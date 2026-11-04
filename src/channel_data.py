"""Record a channel-matched dataset: play clips through the laptop speakers and
re-record them with the internal mic.

Why: a detector trained on clean FoR/TTS files collapses when the same audio
goes through a speaker, a room and a mic (reverb + EQ + noise).  The live demo
uses exactly that channel, so we collect a small matched set and mix it into
training.  This is the "channel robustness" extension in the report.
"""

from __future__ import annotations

import glob
import subprocess
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from .augment import variants as aug_variants
from .features import SR, extract

ROOT = Path(__file__).resolve().parent.parent
CHAN = ROOT / "data" / "chan"
REAL = ROOT / "data" / "real"
CACHE = ROOT / "cache"
ALSA = "plughw:3,0"


def extract_real(n: int = 20) -> list[Path]:
    REAL.mkdir(parents=True, exist_ok=True)
    existing = sorted(REAL.glob("for_*.wav"))
    if len(existing) >= n:
        return existing[:n]
    pf = pq.ParquetFile(ROOT / "data" / "for" / "data" / "train-00000-of-00021.parquet")
    out = []
    for b in pf.iter_batches(batch_size=40):
        d = b.to_pydict()
        for a, lab in zip(d["audio"], d["label"]):
            if int(lab) != 0:
                continue
            f = REAL / f"for_{len(out):02d}.wav"
            if not f.exists():
                y, sr = sf.read(__import__("io").BytesIO(a["bytes"]), dtype="float32")
                if y.ndim > 1:
                    y = y.mean(1)
                sf.write(f, y, sr)
            out.append(f)
            if len(out) >= n:
                return out
    return out


def record_one(src: Path, dst: Path) -> bool:
    info = sf.info(str(src))
    dur = max(1.5, min(info.duration, 12.0)) + 1.2
    player = subprocess.Popen(["aplay", "-q", str(src)],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.25)
    try:
        subprocess.run(
            ["arecord", "-q", "-D", ALSA, "-d", str(int(dur + 0.999)), "-f", "S16_LE",
             "-r", str(SR), "-c", "1", str(dst)],
            check=True, timeout=dur + 8,
        )
    except Exception as e:
        print(f"  ! record failed {src.name}: {e}", flush=True)
        player.terminate()
        return False
    finally:
        try:
            player.wait(timeout=3)
        except Exception:
            player.terminate()
    return dst.exists() and dst.stat().st_size > 4000


def collect(n_tts: int = 30, n_real: int = 20) -> None:
    (CHAN / "tts").mkdir(parents=True, exist_ok=True)
    (CHAN / "real").mkdir(parents=True, exist_ok=True)

    tts_files = [Path(f) for f in sorted(glob.glob(str(ROOT / "data" / "tts" / "*.wav")))]
    tts_files = [f for f in tts_files if f.stat().st_size > 10_000][:n_tts]
    real_files = extract_real(n_real)

    jobs = [(f, CHAN / "tts" / f.name, "tts") for f in tts_files]
    jobs += [(f, CHAN / "real" / f.name, "real") for f in real_files]
    for i, (src, dst, kind) in enumerate(jobs, 1):
        if dst.exists() and dst.stat().st_size > 4000:
            continue
        ok = record_one(src, dst)
        print(f"[chan] {i}/{len(jobs)} {kind} {src.name} {'ok' if ok else 'FAIL'}", flush=True)
        time.sleep(0.3)


def build_cache(n_windows: int = 2, n_aug: int = 3) -> None:
    rng = np.random.default_rng(0)
    X, y, groups = [], [], []
    names = None
    for label, kind in [(1, "tts"), (0, "real")]:
        for f in sorted((CHAN / kind).glob("*.wav")):
            try:
                audio, sr = sf.read(str(f), dtype="float32", always_2d=False)
            except Exception:
                continue
            if audio.ndim > 1:
                audio = audio.mean(1)
            if len(audio) < SR // 2:
                continue
            if len(audio) < SR:
                audio = np.pad(audio, (0, SR - len(audio)))
            # loudness-normalise both classes so the model cannot use level as a cue
            rms = float(np.sqrt(np.mean(audio ** 2))) + 1e-9
            audio = np.clip(audio * (0.05 / rms), -1.0, 1.0).astype(np.float32)
            lo = int(0.4 * SR)
            hi = max(lo, len(audio) - SR - int(0.4 * SR))
            starts = np.linspace(lo, hi, n_windows).astype(int)
            # keep the source identity so a clip cannot leak across the split
            if kind == "tts":
                voice = f.stem.split("_")[0]
                g = f"chan-tts-{voice}"
            else:
                g = f"chan-real-{f.stem}"
            for s in starts:
                w = audio[s:s + SR]
                if float(np.sqrt(np.mean(w ** 2))) < 0.004:
                    continue
                for cand in [w] + aug_variants(w, SR, n_aug, rng):
                    v, names = extract(cand, SR)
                    X.append(v); y.append(label); groups.append(g)
    if not X:
        raise SystemExit("No channel clips. Run --collect first.")
    out_dir = CACHE / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "chan.npz", X=np.vstack(X), y=np.array(y, dtype=np.int8),
        groups=np.array(groups), names=np.array(names),
    )
    print(f"[chan-cache] {len(X)} windows from {len(set(groups))} clips")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--n-tts", type=int, default=30)
    ap.add_argument("--n-real", type=int, default=20)
    a = ap.parse_args()
    if a.collect:
        collect(a.n_tts, a.n_real)
    if a.cache:
        build_cache()
