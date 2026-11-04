"""SENTRY web console.

A small FastAPI + vanilla-JS interface around the streaming detector in
``src.detector``:

  * file analysis with clip-level aggregation
  * an edge-tts attack lab
  * an experimental live microphone screen
  * a method / limits page

Run with ``python app.py`` and open http://127.0.0.1:7860.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import mimetypes
import os
import re
import subprocess
import uuid
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.detector import (
    CHALLENGE,
    DROP,
    LOG,
    Detector,
    aggregate_clip,
    analyze_file,
    mic_stream,
)

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
MODELS = ROOT / "models"
UPLOADS = ROOT / "data" / "uploads"
GENERATED = ROOT / "data" / "generated"
REPORT_DIR = ROOT / "reports"
for folder in (UPLOADS, GENERATED, REPORT_DIR):
    folder.mkdir(parents=True, exist_ok=True)

VOICES = [
    "en-IN-NeerjaNeural",
    "en-US-AriaNeural",
    "en-US-GuyNeural",
    "en-GB-SoniaNeural",
    "en-AU-NatashaNeural",
]

ALLOWED_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac", ".webm"}

_DET: Detector | None = None
FILES: dict[str, Path] = {}
REPORTS: dict[str, Path] = {}
NAMES: dict[str, str] = {}
PROVENANCE: dict[str, str] = {}
_KNOWN_REAL: set[str] | None = None


# --------------------------------------------------------------------------- #
# detector + shared helpers
# --------------------------------------------------------------------------- #
def det() -> Detector:
    """Single clean-file model, loaded lazily."""
    global _DET
    if _DET is None:
        _DET = Detector(MODELS / "sentry_lgbm.txt", MODELS / "meta.json")
    return _DET


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(_jsonable(payload), separators=(',', ':'))}\n\n"


def _rows(res: dict[str, Any]) -> dict[str, Any]:
    return {
        "t": res["t"],
        "p": res["p"],
        "decisions": res["decisions"],
        "rms": res["rms"],
        "ood": res.get("ood", []),
        "now": res["now"],
        "score": res["score"],
        "ood_now": res.get("ood_now", 0.0),
        "smooth": res.get("smooth", res["score"]),
        "decision": res["decision"],
        "latency_ms": res["latency_ms"],
    }


def _write_report(
    path: Path, res: dict[str, Any], agg: dict[str, Any], display_name: str | None = None
) -> Path:
    """Write a per-file JSON analysis artefact to reports/."""
    p = np.asarray(res["p"])
    stem = ((Path(display_name).stem if display_name else path.stem)[:48]) or "clip"
    out = REPORT_DIR / f"{stem}.json"
    try:
        out.write_text(
            json.dumps(
                {
                    "file": str(path),
                    "window_scores": [round(float(x), 4) for x in p],
                    "decisions": res["decisions"],
                    "mean": float(np.mean(p)),
                    "max": float(np.max(p)),
                    "top10": round(float(agg["top10"]), 4),
                    "top30": round(float(agg["top"]), 4),
                    "frac_ai_windows": round(float(agg["frac_drop"]), 4),
                    "sparse_evidence": bool(agg["sparse"]),
                    "ood_frac": round(float(agg.get("ood_frac", 0.0)), 4),
                    "out_of_scope": bool(agg.get("out_of_scope", False)),
                    "ood_threshold": agg.get("ood_thresh"),
                    "clip_decision": agg["decision"],
                    "latency_ms_per_window": res["latency_ms"],
                },
                indent=2,
            )
        )
    except Exception:
        pass
    return out


def _known_real_hashes() -> set[str]:
    """Content hashes of the held-out real recordings shipped/local to the repo.

    Only the clean ``for_*.wav`` clips qualify.  The ``mic_*`` recordings in
    ``data/real`` are a mixed channel set (some are TTS played through the
    speaker), so they are deliberately excluded.
    """
    global _KNOWN_REAL
    if _KNOWN_REAL is None:
        hashes: set[str] = set()
        paths = [ROOT / "samples" / "real.wav"]
        paths += sorted((ROOT / "data" / "real").glob("for_*.wav"))
        for path in paths:
            try:
                hashes.add(hashlib.sha256(path.read_bytes()).hexdigest())
            except Exception:
                pass
        _KNOWN_REAL = hashes
    return _KNOWN_REAL


def _register(
    path: Path,
    prefix: str = "clip",
    display_name: str | None = None,
    provenance: str | None = None,
) -> str:
    fid = f"{prefix}-{uuid.uuid4().hex[:10]}"
    FILES[fid] = path
    NAMES[fid] = display_name or path.name
    if provenance:
        PROVENANCE[fid] = provenance
    return fid


def _media_type(path: Path) -> str:
    return mimetypes.guess_type(str(path))[0] or "application/octet-stream"


# --------------------------------------------------------------------------- #
# devices + samples
# --------------------------------------------------------------------------- #
def _devices() -> tuple[list[dict[str, str]], str]:
    devices: list[dict[str, str]] = [{"value": "default", "label": "System default"}]
    default = "default"
    try:
        text = subprocess.run(
            ["arecord", "-l"], capture_output=True, text=True, timeout=5
        ).stdout
        pattern = re.compile(
            r"card (\d+): (.+?) \[(.+?)\], device (\d+): (.+?) \[(.*?)\]"
        )
        for match in pattern.finditer(text):
            card, short, card_name, dev, name, _ = match.groups()
            value = f"plughw:{card},{dev}"
            haystack = f"{short} {card_name} {name}".lower()
            if "dmic" in haystack:
                label = "Internal digital microphone (DMIC)"
                if default == "default":
                    default = value
            elif "analog" in haystack:
                label = f"Analog input · {name}"
            elif "usb" in haystack:
                label = f"USB microphone · {name}"
            else:
                label = f"{card_name} · {name}"
            devices.append({"value": value, "label": label})
    except Exception:
        pass
    known = {d["value"] for d in devices}
    env = os.environ.get("SENTRY_ALSA_DEV")
    if env and env in known:
        default = env
    elif default == "default" and "plughw:3,0" in known:
        default = "plughw:3,0"
    return devices, default


def _samples() -> list[dict[str, str]]:
    """Three deterministic demo clips shipped with the repo."""
    wanted = [
        ("sample-real", "Real speech", "Held-out recording", "samples/real.wav", "data/real/for_*.wav", "known-real"),
        ("sample-fake", "Spoofed speech", "Held-out FoR fake", "samples/fake.wav", "data/fake/*.wav", None),
        ("sample-tts", "Modern TTS", "edge-tts neural voice", "samples/tts.wav", "data/tts/*.wav", None),
    ]
    out: list[dict[str, str]] = []
    for fid, label, description, primary, fallback, provenance in wanted:
        candidates = [ROOT / primary]
        candidates += [f for f in sorted(ROOT.glob(fallback)) if "mic_" not in f.name]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            continue
        FILES[fid] = path
        NAMES[fid] = path.name
        if provenance:
            PROVENANCE[fid] = provenance
        out.append(
            {
                "id": fid,
                "label": label,
                "description": description,
                "filename": path.name,
                "audio_url": f"/api/file/{fid}",
            }
        )
    return out


def _ood_threshold() -> float | None:
    try:
        guard = det().ood
    except Exception:
        return None
    return guard.threshold if guard else None


def _results_summary(meta: dict[str, Any]) -> dict[str, Any]:
    """Single source of truth for the numbers shown in the console.

    The UI reads these at load time instead of hard-coding them, so a retrain
    or a threshold change cannot leave a stale figure on screen.
    """
    try:
        ev = json.loads((MODELS / "eval.json").read_text())
        agg = json.loads((MODELS / "aggregation.json").read_text())
    except Exception:
        return {}

    hold = ev.get("combined__holdout", {})
    tts = ev.get("for_only__tts_test", {})
    chan = ev.get("clean_model__channel", {})
    lovo = [
        v["accuracy"]
        for v in ev.get("leave_one_voice_out", {}).values()
        if v.get("accuracy") is not None
    ]
    clips = agg.get("clips", {})
    real = clips.get("real", {})
    tts_clips = clips.get("tts_all", {})
    metrics = meta.get("metrics", {})

    return {
        "dataset": meta.get("trained_on", "Fake-or-Real (norm) + edge-tts"),
        "holdout_n": hold.get("n"),
        "accuracy": hold.get("accuracy", metrics.get("accuracy")),
        "auc": hold.get("auc", metrics.get("auc")),
        "eer": hold.get("eer", metrics.get("eer")),
        "unseen_tts_accuracy": tts.get("accuracy"),
        "channel_accuracy": chan.get("accuracy"),
        "real_drop": real.get("drop"),
        "real_n": real.get("n"),
        "tts_drop": tts_clips.get("drop"),
        "tts_n": tts_clips.get("n"),
        "lovo_min": min(lovo) if lovo else None,
        "lovo_max": max(lovo) if lovo else None,
    }


# --------------------------------------------------------------------------- #
# app
# --------------------------------------------------------------------------- #
app = FastAPI(title="SENTRY web console", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    meta = json.loads((MODELS / "meta.json").read_text())
    return {
        "status": "ok",
        "model": "LightGBM · clean channel",
        "n_features": meta.get("n_features", 178),
        "thresholds": meta.get("thresholds", {"log": 0.40, "drop": 0.60}),
        "metrics": meta.get("metrics", {}),
        "results": _results_summary(meta),
        "ood_threshold": _ood_threshold(),
    }


@app.get("/api/devices")
def devices() -> dict[str, Any]:
    items, default = _devices()
    return {"devices": items, "default": default}


@app.get("/api/samples")
def samples() -> dict[str, Any]:
    return {"samples": _samples()}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
    name = Path(file.filename or "clip.wav").name
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(415, f"Unsupported file type: {suffix or 'unknown'}")
    blob = await file.read()
    if not blob:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(blob) > 120 * 1024 * 1024:
        raise HTTPException(413, "File is larger than 120 MB.")
    path = UPLOADS / f"{uuid.uuid4().hex[:12]}{suffix}"
    path.write_bytes(blob)
    provenance = "known-real" if hashlib.sha256(blob).hexdigest() in _known_real_hashes() else None
    fid = _register(path, "upload", name, provenance=provenance)
    return {
        "id": fid,
        "filename": name,
        "size": len(blob),
        "audio_url": f"/api/file/{fid}",
        "provenance": provenance,
    }


@app.get("/api/file/{fid}")
def serve_file(fid: str) -> FileResponse:
    path = FILES.get(fid)
    if not path or not path.exists():
        raise HTTPException(404, "Unknown file.")
    return FileResponse(path, media_type=_media_type(path), filename=path.name)


@app.get("/api/report/{fid}")
def serve_report(fid: str) -> FileResponse:
    path = REPORTS.get(fid)
    if not path or not path.exists():
        raise HTTPException(404, "Report not ready.")
    return FileResponse(path, media_type="application/json", filename=path.name)


def _analysis_stream(fid: str) -> Iterator[str]:
    path = FILES.get(fid)
    if not path or not path.exists():
        yield _sse({"error": "That file is no longer available. Please upload it again."})
        return
    try:
        model = det()
    except Exception as exc:  # model missing / corrupt
        yield _sse({"error": f"Detector could not start: {exc}"})
        return

    last: dict[str, Any] | None = None
    try:
        for i, res in enumerate(analyze_file(path, model)):
            last = res
            if i % 2 == 0 or i == 0:
                payload = _rows(res)
                payload["final"] = False
                yield _sse(payload)

        if last is None:
            yield _sse({"error": "No audio could be read from this file."})
            return

        agg = aggregate_clip(
            np.asarray(last["p"]),
            np.asarray(last["rms"]),
            model.th["log"],
            model.th["drop"],
            ood=np.asarray(last.get("ood", [])),
            ood_thresh=model.ood.threshold if model.ood else None,
        )
        display_name = NAMES.get(fid, path.name)
        report = _write_report(path, last, agg, display_name)
        REPORTS[fid] = report
        payload = _rows(last)
        payload.update(
            {
                "final": True,
                "summary": agg,
                "provenance": PROVENANCE.get(fid),
                "title": f"{display_name} · {last['now']:.1f} s audio",
                "report_url": f"/api/report/{fid}",
                "report_name": report.name,
            }
        )
        yield _sse(payload)
    except Exception as exc:
        yield _sse({"error": f"Analysis failed: {exc}"})


@app.get("/api/analyze/{fid}")
def analyze(fid: str) -> StreamingResponse:
    return StreamingResponse(
        _analysis_stream(fid),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _mic_stream(duration: float, device: str) -> Iterator[str]:
    try:
        model = det()
    except Exception as exc:
        yield _sse({"error": f"Detector could not start: {exc}"})
        return

    emitted = 0
    last: dict[str, Any] | None = None
    try:
        for res in mic_stream(model, duration_s=duration, device=device or None):
            last = res
            emitted += 1
            payload = _rows(res)
            payload["final"] = False
            yield _sse(payload)
    except Exception as exc:
        yield _sse({"error": f"Microphone capture failed: {exc}"})
        return

    if last is None or emitted == 0:
        yield _sse(
            {
                "error": "No audio arrived from the selected device. "
                "Check `arecord -l`, pick another input, or set SENTRY_ALSA_DEV."
            }
        )
        return

    payload = _rows(last)
    payload.update({"final": True, "summary": None, "title": "live microphone"})
    yield _sse(payload)


@app.get("/api/mic")
def mic(duration: float = 20.0, device: str = "default") -> StreamingResponse:
    duration = max(3.0, min(90.0, float(duration)))
    return StreamingResponse(
        _mic_stream(duration, device),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/tts")
async def tts(payload: dict[str, Any]) -> JSONResponse:
    text = str(payload.get("text", "")).strip()[:800]
    voice = str(payload.get("voice", VOICES[0]))
    if not text:
        raise HTTPException(400, "Enter a script first.")
    if voice not in VOICES:
        voice = VOICES[0]

    stem = f"tts_{uuid.uuid4().hex[:10]}"
    mp3 = GENERATED / f"{stem}.mp3"
    wav = GENERATED / f"{stem}.wav"

    try:
        import edge_tts

        await edge_tts.Communicate(text, voice).save(str(mp3))
    except Exception as exc:
        raise HTTPException(
            502,
            "Could not reach edge-tts. This tab needs internet access. "
            f"Details: {exc}",
        ) from exc

    audio = mp3
    if mp3.exists():
        try:
            await asyncio.to_thread(
                subprocess.run,
                [
                    "ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3),
                    "-ar", "16000", "-ac", "1", str(wav),
                ],
                check=True,
            )
            if wav.exists():
                audio = wav
        except Exception:
            audio = mp3  # soundfile can often read mp3 directly

    fid = _register(audio, "tts")
    return JSONResponse(
        {
            "id": fid,
            "filename": audio.name,
            "audio_url": f"/api/file/{fid}",
            "voice": voice,
        }
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7860, log_level="info")
