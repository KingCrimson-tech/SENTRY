"""Channel augmentation: noise, low-pass, reverb and gain.

These are the transformations that break a clean-file detector in a live demo
(speaker -> room -> mic).  We extract features from augmented copies of real
FoR clips and mix them into training.
"""

from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq
from scipy.signal import butter, fftconvolve, sosfilt

from .data import CACHE, _decode, _windows
from .features import SR, extract

ROOT_SHARD = {
    0: "data/for/data/train-00000-of-00021.parquet",    # real
    10: "data/for/data/train-00010-of-00021.parquet",   # fake
}


def _reverb(y: np.ndarray, sr: int, rng) -> np.ndarray:
    rt = rng.uniform(0.15, 0.5)
    n = int(rt * sr)
    t = np.arange(n) / sr
    h = rng.normal(0, 1, n) * np.exp(-t / (rt / 4))
    h[0] = 1.0
    h /= np.sqrt((h ** 2).sum()) + 1e-9
    return fftconvolve(y, h)[: len(y)]


def _lowpass(y: np.ndarray, sr: int, rng) -> np.ndarray:
    cut = rng.uniform(3500, 7200)
    sos = butter(6, cut / (sr / 2), btype="low", output="sos")
    return sosfilt(sos, y).astype(np.float32)


def _noise(y: np.ndarray, sr: int, rng) -> np.ndarray:
    snr_db = rng.uniform(8, 22)
    p = (y ** 2).mean() + 1e-12
    sigma = np.sqrt(p / (10 ** (snr_db / 10)))
    return (y + rng.normal(0, sigma, len(y))).astype(np.float32)


def _gain(y: np.ndarray, sr: int, rng) -> np.ndarray:
    return (y * rng.uniform(0.25, 2.0)).astype(np.float32)


def variants(y: np.ndarray, sr: int, k: int, rng) -> list[np.ndarray]:
    out = []
    for _ in range(k):
        v = y.copy()
        ops = rng.choice(["noise", "lowpass", "reverb", "gain"], size=2, replace=False)
        for op in ops:
            v = {"noise": _noise, "lowpass": _lowpass, "reverb": _reverb, "gain": _gain}[op](v, sr, rng)
        out.append(np.nan_to_num(v, nan=0.0).astype(np.float32))
    return out


def build(n_per_shard: int = 600, k: int = 3, n_windows: int = 2, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    X, y, groups = [], [], []
    names = None
    for shard, rel in ROOT_SHARD.items():
        p = CACHE.parent / rel
        pf = pq.ParquetFile(p)
        seen = 0
        for batch in pf.iter_batches(batch_size=32):
            d = batch.to_pydict()
            for j in range(len(d["label"])):
                audio = d["audio"][j]
                arr, sr = _decode(audio)
                g = str(audio.get("path", "")) or f"{shard}-{seen}"
                for w in _windows(arr, sr, n_windows):
                    for v in variants(w, sr, k, rng):
                        try:
                            f, names = extract(v, sr)
                        except Exception:
                            continue
                        X.append(f); y.append(int(d["label"][j])); groups.append(g)
                seen += 1
                if seen >= n_per_shard:
                    break
            if seen >= n_per_shard:
                break
        print(f"[aug] shard {shard}: {seen} clips, total {len(X)} windows", flush=True)
    out_dir = CACHE / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "aug.npz", X=np.vstack(X), y=np.array(y, dtype=np.int8),
        groups=np.array(groups), names=np.array(names),
    )
    print(f"[aug] saved {len(X)} windows")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--per-shard", type=int, default=600)
    ap.add_argument("--variants", type=int, default=3)
    a = ap.parse_args()
    build(a.per_shard, a.variants)
