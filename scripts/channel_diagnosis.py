"""Diagnose the speaker -> room -> mic channel.

Answers three questions:
  1. Does room noise alone trigger the detector?          (no)
  2. How noisy were the channel recordings?                (noise-floor SNR)
  3. Is the failure caused by noise, or by the channel?    (controlled tests)

Run:  .venv/bin/python scripts/channel_diagnosis.py
"""

from __future__ import annotations

import glob
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, sosfilt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detector import Detector, analyze_file  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SR = 16000
DET = Detector()


def score(y: np.ndarray) -> float:
    tmp = tempfile.mktemp(suffix=".wav")
    sf.write(tmp, y, SR)
    last = list(analyze_file(tmp, DET))[-1]
    p = np.asarray(last["p"])
    rms = np.asarray(last["rms"])
    os.unlink(tmp)
    valid = rms > 0.0015
    return float(p[valid].mean()) if valid.any() else 0.5


def noise_floor_snr(y: np.ndarray, frame: int = 400, hop: int = 160) -> float:
    """Estimate SNR from the quietest vs loudest 20% of short-time energies."""
    n = 1 + (len(y) - frame) // hop
    e = np.array([np.mean(y[i * hop:i * hop + frame] ** 2) for i in range(n)])
    e = np.sort(e)
    noise = e[: max(1, len(e) // 5)].mean()
    speech = e[-max(1, len(e) // 5):].mean()
    return float(10 * np.log10(max(speech - noise, 1e-12) / max(noise, 1e-12)))


def reverb(y: np.ndarray, rt: float = 0.3) -> np.ndarray:
    n = int(rt * SR)
    t = np.arange(n) / SR
    h = np.random.default_rng(0).normal(0, 1, n) * np.exp(-t / (rt / 4))
    h[0] = 1.0
    h /= np.sqrt((h ** 2).sum())
    return fftconvolve(y, h)[: len(y)].astype(np.float32)


def lowpass(y: np.ndarray, cut: float) -> np.ndarray:
    return sosfilt(butter(6, cut / (SR / 2), btype="low", output="sos"), y).astype(np.float32)


def highpass(y: np.ndarray, cut: float) -> np.ndarray:
    return sosfilt(butter(4, cut / (SR / 2), btype="high", output="sos"), y).astype(np.float32)


def main() -> None:
    print("== 1. Room noise alone")
    rn = Path("/tmp/opencode/room_noise.wav")
    if rn.exists():
        y, _ = sf.read(rn)
        print(f"  rms={np.sqrt((y ** 2).mean()):.4f}  score={score(y):.3f}  "
              f"-> {DET.decision(score(y))}")
    else:
        print("  (no /tmp/opencode/room_noise.wav; record 6 s with arecord to include this)")

    print("\n== 2. Noise-floor SNR of channel recordings (lower = noisier)")
    for kind in ("tts", "real"):
        vals = []
        for f in sorted(glob.glob(str(ROOT / "data" / "chan" / kind / "*.wav")))[:10]:
            y, _ = sf.read(f)
            vals.append(noise_floor_snr(y))
        if vals:
            print(f"  {kind:4s}: median {np.median(vals):4.1f} dB  "
                  f"range {min(vals):4.1f}..{max(vals):4.1f} dB  (n={len(vals)})")

    print("\n== 3a. Clean TTS + recorded fan noise")
    clean_path = sorted(glob.glob(str(ROOT / "data" / "tts" / "*.wav")))[0]
    clean, _ = sf.read(clean_path)
    clean = clean.astype(np.float32)
    print(f"  {Path(clean_path).name}: clean score={score(clean):.3f}")
    if rn.exists():
        noise, _ = sf.read(rn)
        noise = noise[int(0.5 * SR):].astype(np.float32)   # skip arecord click
        noise = np.tile(noise, int(np.ceil(len(clean) / len(noise))))[: len(clean)]
        ps = np.mean(clean ** 2)
        for snr_db in (30, 20, 15, 10):
            sigma = np.sqrt(ps / (10 ** (snr_db / 10)))
            n = noise / (np.sqrt(np.mean(noise ** 2)) + 1e-12) * sigma
            print(f"  + fan noise @ {snr_db:2d} dB SNR -> score={score(clean + n):.3f}")

    print("\n== 3b. Simulated channel without noise (quiet re-record)")
    band = highpass(lowpass(clean, 7000), 100)
    print(f"  speaker band 100-7000 Hz         -> {score(band):.3f}")
    print(f"  reverb 0.3 s only                -> {score(reverb(clean, 0.3)):.3f}")
    print(f"  band + reverb 0.3 s              -> {score(reverb(band, 0.3)):.3f}")
    print(f"  band + reverb 0.15 s             -> {score(reverb(band, 0.15)):.3f}")

    print("\nInterpretation: noise alone can flip a clip, but the speaker/mic channel")
    print("flips it even with zero noise. Re-recording quietly removes one confound,")
    print("not the main one. More speakers/voices and channel augmentation are needed.")


if __name__ == "__main__":
    main()
