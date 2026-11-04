"""Pre-flight check for the live demo.

Run this five minutes before presenting:

    make check          # or: .venv/bin/python scripts/demo_check.py

It scores the three shipped sample clips plus any held-out clips available, and
fails loudly (exit code 1) if a clip lands on an unexpected verdict, if the
novelty guard flags known material, or if the headline metrics on the console
disagree with the stored evaluation.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.detector import Detector, aggregate_clip, analyze_file

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"


def score(det: Detector, path: Path) -> dict:
    last = list(analyze_file(path, det))[-1]
    return aggregate_clip(
        np.asarray(last["p"]),
        np.asarray(last["rms"]),
        det.th["log"],
        det.th["drop"],
        ood=np.asarray(last.get("ood", [])),
        ood_thresh=det.ood.threshold if det.ood else None,
    )


def check(det: Detector, label: str, path: Path, expect: set[str]) -> bool:
    agg = score(det, path)
    ok = agg["decision"] in expect
    flag = "PASS" if ok else "FAIL"
    print(
        f"  [{flag}] {label:28s} {path.name:22s} "
        f"score={agg['score']:.2f} windows={agg['n']:2d} "
        f"in_scope={1 - agg['ood_frac']:.0%} -> {agg['decision']}"
    )
    return ok


def metrics_ok() -> bool:
    meta = json.loads((MODELS / "meta.json").read_text())
    ev = json.loads((MODELS / "eval.json").read_text())
    agg = json.loads((MODELS / "aggregation.json").read_text())
    hold = ev["combined__holdout"]["accuracy"]
    shown = meta["metrics"]["accuracy"]
    real = agg["clips"]["real"]
    ok = abs(hold - shown) < 0.01 and real["drop"] == 0
    detail = (
        f"meta={shown:.4f} eval={hold:.4f} "
        f"real_drop={real['drop']}/{real['n']}"
    )
    print(f"  [{'PASS' if ok else 'FAIL'}] console metrics match stored evaluation ({detail})")
    return ok


def main() -> int:
    det = Detector()
    print(f"model: {MODELS / 'sentry_lgbm.txt'}")
    print(
        f"novelty guard: "
        + (f"on · threshold {det.ood.threshold:.2f}" if det.ood else "OFF (models/ood.npz missing)")
    )
    print("\nShipped demo clips")
    ok = True
    ok &= check(det, "real speech", ROOT / "samples" / "real.wav", {"LOG"})
    ok &= check(det, "spoofed speech", ROOT / "samples" / "fake.wav", {"DROP"})
    ok &= check(det, "modern TTS", ROOT / "samples" / "tts.wav", {"DROP"})

    optional = [
        ("held-out real", ROOT / "data" / "real" / "for_*.wav", {"LOG"}),
        ("held-out FoR fake", ROOT / "data" / "fake" / "*.wav", {"DROP"}),
        ("held-out modern TTS", ROOT / "data" / "tts" / "*.wav", {"DROP"}),
    ]
    for label, pattern, expect in optional:
        files = [Path(f) for f in sorted(glob.glob(str(pattern))) if "mic_" not in f][:6]
        if not files:
            continue
        print(f"\n{label} ({len(files)} clips)")
        for path in files:
            ok &= check(det, label, path, expect)

    print("\nConsistency")
    ok &= metrics_ok()

    print("\nREADY FOR DEMO" if ok else "\nNOT READY — fix the FAIL rows above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
