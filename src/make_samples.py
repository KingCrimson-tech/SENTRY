"""Generate contemporary neural-TTS clips (edge-tts) and cache their features.

Why: FoR is a 2019-era corpus.  Modern zero-shot TTS has different artefacts, so
a detector trained only on FoR can look great offline and fail on a live demo.
We generate 2026-era clips, hold out entire voices, and report cross-voice
performance -- that is the honest extension over the original paper.
"""

from __future__ import annotations

import asyncio
import subprocess
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf

from .features import SR, extract

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "tts"
CACHE = ROOT / "cache"

VOICES = [
    "en-US-AriaNeural",
    "en-US-GuyNeural",
    "en-IN-NeerjaNeural",
    "en-GB-SoniaNeural",
    "en-AU-NatashaNeural",
]

TEXTS = [
    "Hello, this is a quick verification call from your bank. Please confirm the OTP you just received.",
    "Congratulations, you have won a prize. To claim it, please share your account number and PIN.",
    "This is technical support. Your computer is infected. Please install this remote access tool.",
    "Hi, it's me. I'm in a meeting and I need you to transfer the invoice amount right away.",
    "Your electricity will be disconnected today unless you pay immediately by gift card.",
    "I am calling about your recent order. Can you confirm your address and card details?",
    "Please stay on the line while I verify your identity with the last four digits of your card.",
    "The refund has been approved. I just need your online banking password to process it.",
    "This is the police. There is a warrant in your name. Pay the fine to avoid arrest.",
    "Your account has been locked for suspicious activity. Click the link I am sending you.",
    "We detected a problem with your tax return. Please read me the verification code.",
    "I'm calling from the hospital. Your family member is in an emergency, send money now.",
    "Your parcel is held at customs. Pay the clearance fee to release it today.",
    "This is your manager. I need you to buy gift cards for a client and send me the codes.",
    "Your subscription will renew. Say yes to cancel, or press one to continue.",
    "I am the IT administrator. We need your password to run a security update.",
    "You have an unpaid toll. Settle it now to avoid a late fee and legal action.",
    "This call is being recorded for quality. Please confirm your mother's maiden name.",
    "Your social media account will be deleted. Verify your identity with this code.",
    "I can offer you a loan at zero percent interest. Just pay the processing fee first.",
    "Hello, I'm doing a short survey. Can you tell me your date of birth and address?",
    "Your insurance claim is ready. We need your bank routing number to release the funds.",
    "This is the airline. Your flight was cancelled. Pay the rebooking fee to keep your seat.",
    "We are from the government. You are eligible for a stimulus payment, share your ID.",
    "Your crypto wallet is at risk. Send your seed phrase to our support agent immediately.",
]

SEM = 6


async def _one(text: str, voice: str, idx: int) -> Path | None:
    mp3 = OUT / f"{voice.split('-')[2]}_{idx:04d}.mp3"
    wav = mp3.with_suffix(".wav")
    if wav.exists():
        return wav
    try:
        import edge_tts

        await edge_tts.Communicate(text, voice).save(str(mp3))
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3),
             "-ar", str(SR), "-ac", "1", str(wav)],
            check=True,
        )
        mp3.unlink(missing_ok=True)
        return wav
    except Exception as e:
        print(f"  ! {voice} {idx}: {e}", flush=True)
        return None


async def generate(n_clips: int = 150) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for i in range(n_clips):
        jobs.append(_one(TEXTS[i % len(TEXTS)], VOICES[i % len(VOICES)], i))
    done = 0
    for coro in asyncio.as_completed(jobs):
        await coro
        done += 1
        if done % 10 == 0:
            print(f"[tts] {done}/{n_clips}", flush=True)


def build_cache(n_windows: int = 2) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    files = sorted(OUT.glob("*.wav"))
    if not files:
        raise SystemExit("No TTS wavs. Run with --generate first.")
    X, y, groups = [], [], []
    names = None
    for i, f in enumerate(files):
        try:
            audio, sr = sf.read(str(f), dtype="float32", always_2d=False)
        except Exception:
            continue
        if audio.ndim > 1:
            audio = audio.mean(1)
        need = SR
        if len(audio) < need:
            audio = np.pad(audio, (0, need - len(audio)))
        starts = np.linspace(0, max(0, len(audio) - need), n_windows).astype(int)
        voice = f.stem.split("_")[0]
        for s in starts:
            v, names = extract(audio[s:s + need], SR)
            X.append(v); y.append(1); groups.append(f"tts-{voice}")
        if i % 20 == 0:
            print(f"[tts-cache] {i}/{len(files)}", flush=True)
    np.savez_compressed(
        CACHE / "tts.npz", X=np.vstack(X), y=np.array(y, dtype=np.int8),
        groups=np.array(groups), names=np.array(names),
    )
    print(f"[tts-cache] saved {len(X)} windows from {len(files)} clips")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", type=int, default=0)
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--windows", type=int, default=2)
    a = ap.parse_args()
    if a.generate:
        asyncio.run(generate(a.generate))
    if a.cache:
        build_cache(a.windows)
