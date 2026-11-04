"""Streaming detector: 1 s windows, 0.25 s hop, EMA + hysteresis decisions."""

from __future__ import annotations

import json
import os
import time
from collections import deque
from pathlib import Path

import lightgbm as lgb
import numpy as np
import soundfile as sf

from .features import SR, extract
from .ood import OODGuard

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

LOG, CHALLENGE, DROP = "LOG", "CHALLENGE", "DROP"
UNKNOWN = "UNKNOWN"


class Detector:
    def __init__(self, model_path: Path | None = None, meta_path: Path | None = None):
        model_path = Path(model_path) if model_path else MODELS / "sentry_lgbm.txt"
        meta_path = Path(meta_path) if meta_path else MODELS / "meta.json"
        if not model_path.exists():
            raise SystemExit(
                f"Model not found at {model_path}. Train it first:\n"
                "  python -m src.data --shards 2\n  python -m src.train"
            )
        self.model = lgb.Booster(model_file=str(model_path))
        self.names = self.model.feature_name()
        meta = json.loads(meta_path.read_text())
        self.th = meta.get("thresholds", {"log": 0.40, "drop": 0.60})
        self.meta = meta
        self.ood = OODGuard.load()

    def featurize(self, window: np.ndarray, sr: int = SR) -> np.ndarray:
        """Extract one feature vector in the model's training order."""
        v, names = extract(window, sr)
        if names != self.names:
            idx = [names.index(n) for n in self.names]
            v = v[idx]
        return v

    def predict(self, vector: np.ndarray) -> float:
        return float(self.model.predict(np.asarray(vector)[None, :])[0])

    def score(self, window: np.ndarray, sr: int = SR) -> float:
        return self.predict(self.featurize(window, sr))

    def ood_distance(self, vector: np.ndarray) -> float:
        """Distance from the training manifold; 0.0 when the guard is absent."""
        if self.ood is None:
            return 0.0
        return self.ood.distance(vector)

    def decision(self, p: float) -> str:
        if p >= self.th["drop"]:
            return DROP
        if p >= self.th["log"]:
            return CHALLENGE
        return LOG


class StreamState:
    """Smoothing + hysteresis so the UI does not flicker."""

    def __init__(self, n: int = 3, th_log: float = 0.40, th_drop: float = 0.60):
        self.hist: deque[float] = deque(maxlen=n)
        self.ema: float | None = None
        self.decision = LOG
        self._pending: str | None = None
        self._pending_n = 0
        self.th_log = th_log
        self.th_drop = th_drop

    def update(self, p: float) -> tuple[float, str]:
        self.hist.append(p)
        raw = float(np.median(self.hist))
        self.ema = raw if self.ema is None else 0.65 * self.ema + 0.35 * raw
        d = DROP if self.ema >= self.th_drop else CHALLENGE if self.ema >= self.th_log else LOG
        # require two consecutive windows before switching state
        if d != self.decision:
            if d == self._pending:
                self._pending_n += 1
            else:
                self._pending, self._pending_n = d, 1
            if self._pending_n >= 2:
                self.decision = d
                self._pending_n = 0
        else:
            self._pending_n = 0
        return self.ema, self.decision

    th_log, th_drop = 0.40, 0.60


def aggregate_clip(
    p,
    rms,
    th_log: float = 0.40,
    th_drop: float = 0.60,
    min_rms: float = 0.0015,
    top_frac: float = 0.30,
    clip_drop: float = 0.50,
    drop_frac: float = 0.15,
    challenge_frac: float = 0.05,
    sparse_frac: float = 0.10,
    sparse_top: float = 0.45,
    log_floor: float = 0.20,
    ood=None,
    ood_thresh: float | None = None,
    ood_frac_thresh: float = 0.50,
) -> dict:
    """Clip-level decision from per-window scores.

    A fake clip only needs *some* windows to expose the artefact, so averaging
    every window is too conservative (this is the SoniaNeural case: ~35% of its
    windows score 1.0, but the mean lands at 0.44).  We therefore use the mean of
    the top ``top_frac`` windows plus the fraction of confident windows.

    DROP      : top-k >= clip_drop AND at least ``drop_frac`` of windows >= th_drop
    CHALLENGE : top-k >= th_log, OR at least ``challenge_frac`` of windows >= th_drop,
                OR sparse evidence (see below), OR top-k in the gray zone above
                ``log_floor``.  LOG is reserved for clearly low evidence so a
                borderline novel fake is never called clean.
    UNKNOWN   : the clip is far from the training manifold (OOD guard) and is not
                already DROP.  This overrides LOG so an unfamiliar fake can never
                be presented as authentic.
    LOG       : otherwise

    ``clip_drop`` is deliberately below the per-window DROP threshold (0.50 vs
    0.60) because the top-k mean is more stable on short clips.  Held-out real
    FoR clips never exceed 0.112 on this statistic, so the margin is large.

    Sparse-evidence guard: long, noisy, music-backed clips may expose only a few
    synthetic windows.  If the top ``sparse_frac`` mean is >= ``sparse_top`` but
    the top-30% mean is still below ``clip_drop``, the clip is not DROP-able but
    it is also not clean.  We escalate LOG -> CHALLENGE and set ``sparse=True``
    so the UI can say "unresolved, do not treat as authentic".  Held-out real
    FoR clips max out at 0.212 on this statistic, so the guard keeps 0/80 false
    positives.
    """
    p = np.asarray(p, dtype=float).reshape(-1)
    rms = np.asarray(rms, dtype=float).reshape(-1)
    if p.size == 0:
        p = np.array([0.5])
    if rms.size != p.size:
        rms = np.full_like(p, 1.0)

    valid = rms > min_rms
    pv = p[valid] if valid.any() else p
    if pv.size == 0:
        pv = np.array([0.5])

    mean = float(pv.mean())
    k = max(1, int(np.ceil(top_frac * pv.size)))
    top = float(np.sort(pv)[-k:].mean())
    frac_drop = float(np.mean(pv >= th_drop))
    k_sparse = max(1, int(np.ceil(sparse_frac * pv.size)))
    top_sparse = float(np.sort(pv)[-k_sparse:].mean())
    sparse = bool(top_sparse >= sparse_top and top < clip_drop)

    ood_frac = 0.0
    out_of_scope = False
    if ood is not None and ood_thresh is not None:
        ood = np.asarray(ood, dtype=float).reshape(-1)
        if ood.size == p.size:
            ov = ood[valid] if valid.any() else ood
            ood_frac = float(np.mean(ov > ood_thresh)) if ov.size else 0.0
        out_of_scope = ood_frac >= ood_frac_thresh

    if top >= clip_drop and frac_drop >= drop_frac:
        decision = DROP
    elif top >= th_log or frac_drop >= challenge_frac or sparse:
        decision = CHALLENGE
    elif top >= log_floor:
        # Gray zone: low but non-zero evidence.  Never call this clean.
        decision = CHALLENGE
    else:
        decision = LOG

    # An unfamiliar clip must never fall through to LOG.  A confident DROP is
    # left alone: it is safe to reject something the model already calls fake.
    if out_of_scope and decision != DROP:
        decision = UNKNOWN

    return {
        "n": int(pv.size),
        "mean": mean,
        "top": top,
        "top10": top_sparse,
        "frac_drop": frac_drop,
        "score": top,
        "sparse": sparse,
        "ood_frac": ood_frac,
        "out_of_scope": out_of_scope,
        "ood_thresh": float(ood_thresh) if ood_thresh is not None else None,
        "decision": decision,
    }


def stream_windows(y: np.ndarray, sr: int, win_s: float = 1.0, hop_s: float = 0.25):
    """Yield (end_time_s, window, rms) for a whole clip."""
    win = int(win_s * sr)
    hop = int(hop_s * sr)
    if len(y) < win:
        y = np.pad(y, (0, win - len(y)))
    for end in range(win, len(y) + 1, hop):
        w = y[end - win:end]
        yield end / sr, w, float(np.sqrt(np.mean(w ** 2)))


def analyze_file(path: str | Path, det: Detector):
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(sr), SR)
        y = resample_poly(y, SR // g, int(sr) // g).astype(np.float32)
        sr = SR
    st = StreamState(th_log=det.th["log"], th_drop=det.th["drop"])
    times, probs, decisions, rmss, oods = [], [], [], [], []
    t0 = time.perf_counter()
    n = 0
    compute = 0.0
    for t, w, rms in stream_windows(y, sr):
        oodv = 0.0
        if rms < 0.0015:
            p, d = 0.5, CHALLENGE
        else:
            # software AGC: laptop mics are often very quiet
            c0 = time.perf_counter()
            v = det.featurize(w * min(40.0, 0.06 / (rms + 1e-9)), sr)
            p = det.predict(v)
            oodv = det.ood_distance(v)
            compute += time.perf_counter() - c0
            _, d = st.update(p)
            n += 1
        times.append(t); probs.append(p); decisions.append(d)
        rmss.append(rms); oods.append(oodv)
        yield {
            "t": times, "p": probs, "decisions": decisions, "rms": rmss,
            "ood": oods, "now": t, "score": p, "decision": d,
            "ood_now": oodv,
            "smooth": float(st.ema if st.ema is not None else p),
            "latency_ms": compute * 1000 / max(n, 1),
        }


def list_input_devices():
    """ALSA capture devices that are known to work with arecord."""
    import re
    import subprocess

    devs = ["default"]
    try:
        out = subprocess.run(["arecord", "-l"], capture_output=True, text=True, timeout=5).stdout
        for line in out.splitlines():
            m = re.match(r"card (\d+): (.+?), device (\d+): (.+)", line)
            if m:
                card, _, dev, name = m.groups()
                devs.append(f"plughw:{card},{dev}")
    except Exception:
        pass
    return devs


def mic_stream(det: Detector, duration_s: float = 20.0, hop_s: float = 0.25, device: str | None = None):
    """Server-side mic capture via arecord (robust on PipeWire/ALSA).

    arecord's ALSA plug layer handles rate conversion, so we read 16 kHz mono
    PCM directly and avoid PortAudio heap issues with PipeWire nodes.
    """
    import subprocess

    dev = device or os.environ.get("SENTRY_ALSA_DEV", "plughw:3,0")
    hop = max(1, int(hop_s * SR))
    cmd = ["arecord", "-q", "-D", dev, "-f", "S16_LE", "-r", str(SR),
           "-c", "1", "-t", "raw"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    buf = np.zeros(0, dtype=np.float32)
    st = StreamState(th_log=det.th["log"], th_drop=det.th["drop"])
    times, probs, decisions, rmss = [], [], [], []
    t0 = time.perf_counter()
    n = 0
    compute = 0.0
    try:
        while time.perf_counter() - t0 < duration_s:
            raw = proc.stdout.read(hop * 2)
            if not raw:
                break
            block = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            buf = np.concatenate([buf, block])[-SR:]
            if len(buf) < SR:
                continue
            rms = float(np.sqrt(np.mean(buf ** 2)))
            t = time.perf_counter() - t0
            if rms < 0.0015:
                p, d = 0.5, CHALLENGE
            else:
                c0 = time.perf_counter()
                p = det.score(buf * min(40.0, 0.06 / (rms + 1e-9)), SR)
                compute += time.perf_counter() - c0
                _, d = st.update(p)
                n += 1
            times.append(t); probs.append(p); decisions.append(d); rmss.append(rms)
            yield {
                "t": times, "p": probs, "decisions": decisions, "rms": rmss,
                "now": t, "score": p, "decision": d,
                "smooth": float(st.ema if st.ema is not None else p),
                "latency_ms": compute * 1000 / max(n, 1),
            }
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except Exception:
            proc.kill()
