"""Out-of-distribution guard.

The paper's Track B (an integer Mahalanobis detector fitted on bona-fide speech
only) was weak in closed-set tests.  We keep the *idea* but change the job: a
simple nearest-neighbour distance in standardised feature space tells the console
when a clip does not resemble anything the model was trained on.  In that case
the verdict is UNKNOWN, never LOG -- the score is simply not trustworthy.

Build the reference once from the feature cache:

    python -m src.ood --build

The deployed demo only needs ``models/ood.npz``; the cache is not required.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
MODELS = ROOT / "models"


class OODGuard:
    """Nearest-neighbour distance to a sample of the training manifold."""

    def __init__(self, ref: np.ndarray, mu: np.ndarray, sd: np.ndarray, threshold: float):
        self.ref = ref.astype(np.float32)
        self.mu = mu.astype(np.float32)
        self.sd = sd.astype(np.float32)
        self.threshold = float(threshold)

    @classmethod
    def load(cls, path: Path | None = None) -> "OODGuard | None":
        path = Path(path) if path else MODELS / "ood.npz"
        if not path.exists():
            return None
        d = np.load(path)
        return cls(d["ref"], d["mu"], d["sd"], float(d["threshold"]))

    def distance(self, vector: np.ndarray) -> float:
        z = (np.asarray(vector, dtype=np.float32) - self.mu) / self.sd
        d2 = ((self.ref - z) ** 2).sum(axis=1)
        return float(np.sqrt(d2.min()))


def _load_training(cache_dir: Path) -> np.ndarray:
    files = sorted(glob.glob(str(cache_dir / "*.npz")))
    if not files:
        raise SystemExit("No feature cache found. Run: make data && make aug && make tts")
    return np.vstack([np.load(f, allow_pickle=True)["X"] for f in files]).astype(np.float64)


def build(out: Path | None = None, n_ref: int = 2000, n_cal: int = 600, seed: int = 7) -> Path:
    """Fit the guard and write ``models/ood.npz``.

    The threshold is the 99.5th percentile of the distance from held-back
    in-distribution windows to the reference set.  On the current cache that is
    roughly 15.5; real, FoR-fake and edge-tts clips all sit well below it, so the
    guard does not fire on the demo material.
    """
    out = Path(out) if out else MODELS / "ood.npz"
    X = _load_training(CACHE)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-6] = 1.0
    Z = (X - mu) / sd

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Z), size=min(n_ref + n_cal, len(Z)), replace=False)
    ref = Z[idx[:n_ref]]
    cal = Z[idx[n_ref:]]

    dists = []
    for i in range(0, len(cal), 256):
        chunk = cal[i:i + 256]
        d2 = ((chunk[:, None, :] - ref[None, :, :]) ** 2).sum(axis=-1)
        dists.append(np.sqrt(d2.min(axis=1)))
    cal_d = np.concatenate(dists) if dists else np.array([1.0])
    threshold = float(np.percentile(cal_d, 99.5))

    out.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        out,
        ref=ref.astype(np.float32),
        mu=mu.astype(np.float32),
        sd=sd.astype(np.float32),
        threshold=np.float32(threshold),
    )
    print(f"[ood] wrote {out}  ref={ref.shape}  threshold={threshold:.3f}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--out", default=str(MODELS / "ood.npz"))
    a = ap.parse_args()
    if a.build:
        build(Path(a.out))
    else:
        guard = OODGuard.load(Path(a.out))
        print("no guard" if guard is None else f"guard ok · threshold={guard.threshold:.3f}")
