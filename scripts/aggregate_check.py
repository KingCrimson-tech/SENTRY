"""Validate clip-level aggregation (the SoniaNeural fix).

The deployed detector scores 1-second windows independently.  Averaging every
window is too conservative for clips whose scores are bimodal: ~35% of
``SoniaNeural`` windows score 1.0, but the plain mean is only 0.44 -> LOG.

This script reproduces the validation behind ``aggregate_clip``:

  * 80 held-out real FoR clips and 80 held-out fake FoR clips (sampled from the
    raw parquet shards, not the feature cache);
  * all 150 local edge-tts clips, grouped by voice;
  * window-mean vs top-30% vs AI-window-fraction rules.

Writes ``models/aggregation.json``.

Run:  .venv/bin/python scripts/aggregate_check.py
"""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pyarrow.parquet as pq

from src.data import _decode
from src.detector import Detector, aggregate_clip, analyze_file, stream_windows

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "models" / "aggregation.json"
DET = Detector()
MIN_RMS = 0.0015


def clip_aggs_from_windows(p, rms) -> dict | None:
    p = np.asarray(p, dtype=float)
    rms = np.asarray(rms, dtype=float)
    valid = rms > MIN_RMS
    if not valid.any():
        return None
    pv = p[valid]
    agg = aggregate_clip(p, rms, DET.th["log"], DET.th["drop"])
    return {
        "n": int(pv.size),
        "mean": float(pv.mean()),
        "max": float(pv.max()),
        "top10": float(agg["top10"]),
        "top30": float(agg["top"]),
        "frac60": float(agg["frac_drop"]),
        "sparse": bool(agg["sparse"]),
        "decision": agg["decision"],
    }


def score_audio(y: np.ndarray, sr: int) -> dict | None:
    probs, rmss = [], []
    for _t, w, rms in stream_windows(y, sr):
        if rms < MIN_RMS:
            continue
        probs.append(DET.score(w, sr))
        rmss.append(rms)
    if not probs:
        return None
    return clip_aggs_from_windows(probs, rmss)


def sample_parquet(path: Path, label: int, n: int = 80) -> list[dict]:
    out: list[dict] = []
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=20):
        d = batch.to_pydict()
        for audio, lab in zip(d["audio"], d["label"]):
            if int(lab) != label:
                continue
            y, sr = _decode(audio)
            agg = score_audio(y, sr)
            if agg:
                out.append(agg)
            if len(out) >= n:
                return out
    return out


def tts_clips() -> dict[str, list[dict]]:
    by_voice: dict[str, list[dict]] = defaultdict(list)
    for f in sorted(glob.glob(str(ROOT / "data" / "tts" / "*.wav"))):
        voice = Path(f).name.split("_")[0]
        res = list(analyze_file(f, DET))[-1]
        agg = clip_aggs_from_windows(res["p"], res["rms"])
        if agg:
            by_voice[voice].append(agg)
    return dict(by_voice)


def summarize(label: str, rows: list[dict]) -> dict:
    if not rows:
        return {}
    mean = np.array([r["mean"] for r in rows])
    top10 = np.array([r["top10"] for r in rows])
    top30 = np.array([r["top30"] for r in rows])
    frac60 = np.array([r["frac60"] for r in rows])
    return {
        "label": label,
        "n": len(rows),
        "mean_median": round(float(np.median(mean)), 4),
        "mean_min": round(float(mean.min()), 4),
        "mean_max": round(float(mean.max()), 4),
        "top10_median": round(float(np.median(top10)), 4),
        "top10_min": round(float(top10.min()), 4),
        "top10_max": round(float(top10.max()), 4),
        "top30_median": round(float(np.median(top30)), 4),
        "top30_min": round(float(top30.min()), 4),
        "top30_max": round(float(top30.max()), 4),
        "frac60_median": round(float(np.median(frac60)), 4),
        "frac60_min": round(float(frac60.min()), 4),
        "frac60_max": round(float(frac60.max()), 4),
        "drop": int(sum(r["decision"] == "DROP" for r in rows)),
        "challenge": int(sum(r["decision"] == "CHALLENGE" for r in rows)),
        "log": int(sum(r["decision"] == "LOG" for r in rows)),
        "sparse": int(sum(r["sparse"] for r in rows)),
    }


def main() -> None:
    print("== sampling raw FoR shards (this takes a few minutes)")
    real = sample_parquet(ROOT / "data" / "for" / "data" / "train-00000-of-00021.parquet", 0)
    print(f"  real clips: {len(real)}")
    fake = sample_parquet(ROOT / "data" / "for" / "data" / "train-00010-of-00021.parquet", 1)
    print(f"  fake clips: {len(fake)}")

    print("\n== local edge-tts clips, by voice")
    voices = tts_clips()
    tts_all = [r for rows in voices.values() for r in rows]
    for v, rows in sorted(voices.items()):
        print(f"  {v:16s} n={len(rows):3d}  mean_med={np.median([r['mean'] for r in rows]):.3f}  "
              f"top30_med={np.median([r['top30'] for r in rows]):.3f}  "
              f"DROP={sum(r['decision'] == 'DROP' for r in rows)}/{len(rows)}")

    print("\n== separation")
    for name, rows in (("real", real), ("fake_for", fake), ("tts_all", tts_all)):
        s = summarize(name, rows)
        print(f"  {name:9s} n={s['n']:3d}  mean(p50/max)={s['mean_median']:.3f}/{s['mean_max']:.3f}  "
              f"top10(p50/max)={s['top10_median']:.3f}/{s['top10_max']:.3f}  "
              f"top30(p50/max)={s['top30_median']:.3f}/{s['top30_max']:.3f}  "
              f"DROP={s['drop']}/{s['n']}  sparse={s['sparse']}")

    print("\n== rule comparison (flag as AI)")
    rules = {
        "mean>=0.60": lambda r: r["mean"] >= 0.60,
        "max>=0.60": lambda r: r["max"] >= 0.60,
        "top30>=0.60": lambda r: r["top30"] >= 0.60,
        "top30>=0.50": lambda r: r["top30"] >= 0.50,
        "top10>=0.45": lambda r: r["top10"] >= 0.45,
        "frac60>=0.15": lambda r: r["frac60"] >= 0.15,
        "sparse_guard": lambda r: r["sparse"],
        "aggregate_clip": lambda r: r["decision"] == "DROP",
        "sparse_or_drop": lambda r: r["decision"] == "DROP" or r["sparse"],
    }
    table = {}
    for rule_name, rule in rules.items():
        row = {}
        for name, rows in (("real", real), ("fake_for", fake), ("tts_all", tts_all),
                           ("sonia", voices.get("SoniaNeural", []))):
            row[name] = sum(rule(r) for r in rows)
        table[rule_name] = row
        print(f"  {rule_name:14s} real {row['real']:2d}/{len(real)}  "
              f"fake {row['fake_for']:2d}/{len(fake)}  "
              f"tts {row['tts_all']:3d}/{len(tts_all)}  "
              f"sonia {row['sonia']:2d}/{len(voices.get('SoniaNeural', []))}")

    payload = {
        "thresholds": DET.th,
        "aggregation": {
            "top_frac": 0.30,
            "drop_frac": 0.15,
            "challenge_frac": 0.05,
            "sparse_frac": 0.10,
            "sparse_top": 0.45,
            "rule": "DROP if top30>=0.50 and frac60>=0.15; "
                    "CHALLENGE if top30>=0.40 or frac60>=0.05 or sparse; else LOG",
            "sparse_rule": "sparse = top10>=0.45 and top30<0.50; escalate LOG -> CHALLENGE",
        },
        "clips": {
            "real": summarize("real", real),
            "fake_for": summarize("fake_for", fake),
            "tts_all": summarize("tts_all", tts_all),
            **{f"tts_{v}": summarize(v, rows) for v, rows in sorted(voices.items())},
        },
        "rules": table,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
