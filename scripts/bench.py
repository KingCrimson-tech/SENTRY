"""Score a fixed probe set with a chosen model: clean vs speaker→mic channel."""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detector import Detector, analyze_file  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def score(det, f):
    last = list(analyze_file(f, det))[-1]
    p = np.asarray(last["p"]); rms = np.asarray(last["rms"])
    v = rms > 0.0015
    return float(p[v].mean()) if v.any() else 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "sentry_lgbm.txt"))
    a = ap.parse_args()
    det = Detector(Path(a.model))
    groups = {
        "clean real": sorted(glob.glob(str(ROOT / "data/real/for_*.wav")))[:3],
        "clean fake (FoR)": sorted(glob.glob(str(ROOT / "data/fake/*.wav")))[:3],
        "clean TTS": sorted(glob.glob(str(ROOT / "data/tts/*.wav")))[:6],
        "speaker→mic TTS": sorted(glob.glob(str(ROOT / "data/real/mic_*.wav")))[:6],
        "speaker→mic real": sorted(glob.glob(str(ROOT / "data/chan/real/*.wav")))[:4],
    }
    print(f"model: {a.model}")
    for name, files in groups.items():
        if not files:
            continue
        vals = [score(det, f) for f in files]
        print(f"  {name:18s} n={len(vals):2d} mean={np.mean(vals):.2f} "
              f"min={np.min(vals):.2f} max={np.max(vals):.2f}  {['%.2f' % v for v in vals]}")


if __name__ == "__main__":
    main()
