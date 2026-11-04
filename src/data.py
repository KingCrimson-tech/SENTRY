"""Dataset download + feature caching for the SENTRY prototype.

We use the Fake-or-Real (FoR) "norm" release on Hugging Face:
  ArissBandoss/fake_or_real_dataset_for_norm   (69.3k clips, 16 kHz mono)

It is a clean real/fake speech corpus with 2019-era TTS/VC spoofs.  We also
support mixing in contemporary neural TTS clips generated locally with
edge-tts (see src/make_samples.py) so the detector does not only learn the
"old TTS" shortcuts -- this is the main extension over the original paper.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from .features import extract

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = ROOT / "cache"
HF_REPO = "ArissBandoss/fake_or_real_dataset_for_norm"
HF_SHARD = "data/train-{i:05d}-of-00021.parquet"


def shard_paths(indices) -> list[Path]:
    return [DATA / "for" / HF_SHARD.format(i=i) for i in indices]


def download(indices) -> list[Path]:
    from huggingface_hub import hf_hub_download

    if isinstance(indices, int):
        indices = range(indices)
    DATA.mkdir(parents=True, exist_ok=True)
    out = []
    for i in indices:
        p = DATA / "for" / HF_SHARD.format(i=i)
        if p.exists():
            out.append(p)
            continue
        print(f"[data] downloading shard {i} ...", flush=True)
        got = hf_hub_download(
            repo_id=HF_REPO,
            filename=HF_SHARD.format(i=i),
            repo_type="dataset",
            local_dir=str(DATA / "for"),
        )
        out.append(Path(got))
    return out


def _decode(audio) -> tuple[np.ndarray, int]:
    if isinstance(audio, dict):
        raw = audio.get("bytes")
        if raw is None and audio.get("array") is not None:
            return np.asarray(audio["array"], dtype=np.float32), int(audio.get("sampling_rate", 16000))
        arr, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    elif isinstance(audio, (bytes, bytearray)):
        arr, sr = sf.read(io.BytesIO(audio), dtype="float32", always_2d=False)
    else:
        arr, sr = np.asarray(audio, dtype=np.float32), 16000
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    return arr, int(sr)


def _windows(y: np.ndarray, sr: int, n_windows: int, win_s: float = 1.0, hop_s: float = 0.5):
    """Take up to n_windows evenly spaced 1 s windows from a clip."""
    need = int(win_s * sr)
    if len(y) < need // 2:
        return []
    if len(y) < need:
        y = np.pad(y, (0, need - len(y)))
    starts = np.linspace(0, max(0, len(y) - need), n_windows).astype(int)
    return [y[s:s + need] for s in starts]


def build_cache(indices=3, n_windows: int = 2, limit_per_shard: int | None = None) -> None:
    if isinstance(indices, int):
        indices = list(range(indices))
    CACHE.mkdir(parents=True, exist_ok=True)
    paths = shard_paths(indices)
    for i, p in zip(indices, paths):
        out = CACHE / f"for_{i:02d}.npz"
        if out.exists():
            print(f"[cache] {out.name} exists, skipping", flush=True)
            continue
        print(f"[cache] {p.name} ...", flush=True)
        X, y, groups = [], [], []
        pf = pq.ParquetFile(p)
        seen = 0
        for batch in pf.iter_batches(batch_size=32):
            cols = batch.to_pydict()
            for j in range(len(cols["label"])):
                audio = cols["audio"][j]
                arr, sr = _decode(audio)
                src = str(cols.get("source", [""] * (j + 1))[j])
                path = ""
                if isinstance(audio, dict):
                    path = str(audio.get("path", ""))
                g = path or src or f"{i}-{seen}"
                for w in _windows(arr, sr, n_windows):
                    try:
                        v, names = extract(w, sr)
                    except Exception:
                        continue
                    X.append(v)
                    y.append(int(cols["label"][j]))
                    groups.append(g)
                seen += 1
                if limit_per_shard and seen >= limit_per_shard:
                    break
            if limit_per_shard and seen >= limit_per_shard:
                break
            if seen % 200 == 0:
                print(f"  {seen} clips -> {len(X)} windows", flush=True)
        np.savez_compressed(
            out, X=np.vstack(X), y=np.array(y, dtype=np.int8),
            groups=np.array(groups), names=np.array(names),
        )
        print(f"[cache] saved {out.name}: {len(X)} windows", flush=True)


def load_cache() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    files = sorted(CACHE.glob("*.npz"))
    if not files:
        raise SystemExit("No cache found. Run: python -m src.data --shards 2")
    Xs, ys, gs = [], [], []
    names = None
    for f in files:
        d = np.load(f, allow_pickle=True)
        Xs.append(d["X"]); ys.append(d["y"]); gs.append(d["groups"])
        names = list(d["names"])
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(gs), names


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=2)
    ap.add_argument("--indices", type=str, default=None,
                    help="comma-separated shard ids, e.g. 10,11,12")
    ap.add_argument("--windows", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    idx = [int(x) for x in a.indices.split(",")] if a.indices else a.shards
    download(idx)
    build_cache(idx, a.windows, a.limit)
