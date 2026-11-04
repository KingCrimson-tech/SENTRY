"""
SENTRY-style handcrafted audio features.

This is a *lightweight reimplementation* of the feature family described in:
  SENTRY: A Streaming Edge Network Trust Engine for Real-Time Audio Deepfake
  Detection (arXiv:2610.24468)

The paper's selling point is that every operation is integer-friendly
(add / shift / compare / bitwise), so a detector can run inline at a network
edge on cheap hardware.  We keep the same feature *families* -- wavelet packet
energies, Walsh-Hadamard sequency profiles and waveform micro-statistics --
but use float arithmetic for portability.  The resulting vector is small
(~150 dims) and a LightGBM model scores a 1 s window in well under a
millisecond on a laptop CPU.
"""

from __future__ import annotations

import numpy as np
import pywt
from scipy.fft import dct

SR = 16_000
N_FFT = 512
WIN = 400          # 25 ms
HOP = 160          # 10 ms
N_MELS = 40
N_MFCC = 20

# ---------------------------------------------------------------- mel banks
def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)


def _mel_filterbank(sr=SR, n_fft=N_FFT, n_mels=N_MELS):
    fmax = sr / 2
    mels = np.linspace(_hz_to_mel(0), _hz_to_mel(fmax), n_mels + 2)
    hz = _mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(1, n_mels + 1):
        l, c, r = bins[i - 1], bins[i], bins[i + 1]
        if c == l:
            c = l + 1
        if r <= c:
            r = c + 1
        for k in range(l, min(c, fb.shape[1])):
            fb[i - 1, k] = (k - l) / max(c - l, 1)
        for k in range(c, min(r, fb.shape[1])):
            fb[i - 1, k] = (r - k) / max(r - c, 1)
    return fb


_MEL_FB = _mel_filterbank()


# ---------------------------------------------------------------- framing
def _frames(x: np.ndarray) -> np.ndarray:
    if len(x) < WIN:
        x = np.pad(x, (0, WIN - len(x)))
    n = 1 + (len(x) - WIN) // HOP
    idx = np.arange(WIN)[None, :] + HOP * np.arange(n)[:, None]
    return x[idx]


def _stats(v: np.ndarray, prefix: str) -> dict[str, float]:
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        v = np.zeros(1)
    mu, sd = float(v.mean()), float(v.std())
    out = {
        f"{prefix}_mean": mu,
        f"{prefix}_std": sd,
        f"{prefix}_p10": float(np.percentile(v, 10)),
        f"{prefix}_p90": float(np.percentile(v, 90)),
    }
    # skew / kurtosis without scipy.stats (cheap)
    if sd > 1e-9:
        z = (v - mu) / sd
        out[f"{prefix}_skew"] = float(np.mean(z ** 3))
        out[f"{prefix}_kurt"] = float(np.mean(z ** 4) - 3.0)
    else:
        out[f"{prefix}_skew"] = 0.0
        out[f"{prefix}_kurt"] = 0.0
    return out


# ---------------------------------------------------------------- WHT
def _fwht_batch(seg: np.ndarray) -> np.ndarray:
    """Vectorised Walsh-Hadamard transform over rows of a (n, block) array."""
    n, block = seg.shape
    h = 1
    while h < block:
        # (n, block//(2h), 2, h) -> butterflies
        s = seg.reshape(n, block // (2 * h), 2, h)
        a = s[:, :, 0, :]
        b = s[:, :, 1, :]
        seg = np.concatenate([a + b, a - b], axis=2).reshape(n, block)
        h *= 2
    return seg


def _wht_profile(x: np.ndarray, block: int = 1024, bins: int = 24) -> np.ndarray:
    """Average |WHT| coefficient profile over non-overlapping blocks."""
    n = max(1, len(x) // block)
    seg = x[: n * block].astype(np.float64).reshape(n, block)
    prof = np.abs(_fwht_batch(seg)).mean(axis=0)
    edges = np.linspace(0, block, bins + 1).astype(int)
    binned = np.array([prof[a:b].mean() for a, b in zip(edges[:-1], edges[1:])])
    total = binned.sum() + 1e-9
    p = binned / total
    ent = float(-(p * np.log(p + 1e-12)).sum())
    return np.concatenate([np.log1p(binned), [ent, ent / np.log(bins), p.max()]])


# ---------------------------------------------------------------- main
def extract(x: np.ndarray, sr: int = SR) -> tuple[np.ndarray, list[str]]:
    """Return (feature_vector, feature_names) for one audio window."""
    if sr != SR:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g)
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size < WIN:
        x = np.pad(x, (0, WIN - x.size))
    # remove DC + pre-emphasis (integer-friendly in a fixed point port)
    x = x - x.mean()
    x = np.append(x[0], x[1:] - 0.97 * x[:-1]).astype(np.float32)

    feats: dict[str, float] = {}
    names_order: list[str] = []

    # ---- frame-level spectra -------------------------------------------
    fr = _frames(x) * np.hanning(WIN).astype(np.float32)
    spec = np.abs(np.fft.rfft(fr, n=N_FFT)) ** 2 + 1e-12
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / SR)

    mel = spec @ _MEL_FB.T
    logmel = np.log(mel + 1e-8)
    mfcc = dct(logmel, type=2, axis=1, norm="ortho")[:, :N_MFCC]
    d_mfcc = np.diff(mfcc, axis=0, prepend=mfcc[:1])

    for i in range(N_MFCC):
        feats[f"mfcc{i}_mean"] = float(mfcc[:, i].mean())
        feats[f"mfcc{i}_std"] = float(mfcc[:, i].std())
        feats[f"dmfcc{i}_mean"] = float(d_mfcc[:, i].mean())
        feats[f"dmfcc{i}_std"] = float(d_mfcc[:, i].std())

    # ---- spectral shape ------------------------------------------------
    mag = np.sqrt(spec)
    p = spec / (spec.sum(axis=1, keepdims=True) + 1e-12)
    centroid = (p * freqs[None, :]).sum(1)
    bandwidth = np.sqrt((p * (freqs[None, :] - centroid[:, None]) ** 2).sum(1))
    cum = np.cumsum(p, axis=1)
    rolloff = freqs[np.argmax(cum >= 0.85, axis=1)]
    flatness = np.exp(np.log(mag + 1e-12).mean(1)) / (mag.mean(1) + 1e-12)
    flux = np.sqrt((np.diff(mag, axis=0, prepend=mag[:1]) ** 2).sum(1))
    zcr = np.mean(np.abs(np.diff(np.sign(fr), axis=1)) > 0, axis=1)
    rms = np.sqrt((fr ** 2).mean(1))
    hf = mag[:, freqs >= 4000].sum(1) / (mag.sum(1) + 1e-12)

    for k, v in [
        ("centroid", centroid), ("bandwidth", bandwidth), ("rolloff", rolloff),
        ("flatness", flatness), ("flux", flux), ("zcr", zcr), ("rms", rms),
        ("hf_ratio", hf),
    ]:
        feats.update(_stats(v, k))

    # autocorrelation pitch strength on the whole window (2 lags bands)
    xc = np.correlate(x[:SR], x[:SR], mode="full")[SR - 1:]
    xc = xc / (xc[0] + 1e-9)
    lo, hi = int(SR / 400), int(SR / 60)          # 60-400 Hz
    peak = float(xc[lo:hi].max()) if hi > lo else 0.0
    feats["pitch_strength"] = peak
    feats["pitch_strength_std"] = float(np.std(xc[lo:hi])) if hi > lo else 0.0

    # ---- wavelet packet energies (db4, level 4) -----------------------
    wp = pywt.WaveletPacket(data=x, wavelet="db4", mode="symmetric", maxlevel=4)
    nodes = wp.get_level(4, order="freq")
    energies = np.array([float(np.sum(n.data ** 2)) for n in nodes])
    energies = energies / (energies.sum() + 1e-12)
    for i, e in enumerate(energies):
        feats[f"wpt{i}"] = float(np.log(e + 1e-9))
    feats["wpt_entropy"] = float(-(energies * np.log(energies + 1e-12)).sum())

    # ---- Walsh-Hadamard sequency profile ------------------------------
    wht = _wht_profile(x)
    for i, v in enumerate(wht):
        feats[f"wht{i}"] = float(v)

    # ---- waveform micro statistics ------------------------------------
    feats["wave_kurt"] = float(np.mean((x - x.mean()) ** 4) / (x.std() ** 4 + 1e-12))
    feats["wave_skew"] = float(np.mean((x - x.mean()) ** 3) / (x.std() ** 3 + 1e-12))
    env = np.abs(x)
    feats["env_crest"] = float(env.max() / (env.mean() + 1e-12))
    feats["env_dynamic"] = float(np.log((env.max() + 1e-9) / (np.percentile(env, 10) + 1e-9)))

    # stable order
    names_order = sorted(feats)
    vec = np.array([feats[k] for k in names_order], dtype=np.float32)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return vec, names_order
