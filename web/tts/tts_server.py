"""Local text-to-speech service for the Live tab → TouchDesigner voice.

Runs Chatterbox-Turbo (ResembleAI, MIT, 350M, English) on the GPU in its own
environment — its pins (torch 2.6, transformers 5.2, numpy < 2) clash with the
main delyrism backend, so it is a separate process:

    uv venv --python 3.11 .venv-tts
    uv pip install --python .venv-tts/Scripts/python.exe torch==2.6.0 torchaudio==2.6.0 \
        --index-url https://download.pytorch.org/whl/cu126
    uv pip install --python .venv-tts/Scripts/python.exe -r web/tts/requirements.txt
    .venv-tts/Scripts/python.exe -m uvicorn tts_server:app --app-dir web/tts --port 8010

POST /speak {"text": "..."} → {"path", "duration", "ms", "text"}
    synthesises the text with the model's built-in voice, peak-normalises it
    and writes a 24 kHz mono WAV into DELYRISM_TTS_DIR (default
    td/runtime/tts), cycling through 8 file names so TouchDesigner can still
    be playing one while the next is written.
GET /healthz → {"ok", "ready", "device"}

Optional: DELYRISM_TTS_VOICE = path to a ~10 s reference WAV to clone a voice.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import torch
import torchaudio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(os.environ.get("DELYRISM_TTS_DIR", ROOT / "td" / "runtime" / "tts"))
VOICE = os.environ.get("DELYRISM_TTS_VOICE") or None
SLOTS = 8

app = FastAPI(title="delyrism TTS (Chatterbox-Turbo)")
_state = {"model": None, "slot": 0, "error": None}
_lock = threading.Lock()


def _load() -> None:
    try:
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        device = "cuda" if torch.cuda.is_available() else "cpu"
        t0 = time.time()
        model = ChatterboxTurboTTS.from_pretrained(device=device)
        model.generate("ready", audio_prompt_path=VOICE)       # warm-up: the first call is slow
        _state["model"] = model
        print(f"[tts] Chatterbox-Turbo ready on {device} in {time.time() - t0:.1f}s", flush=True)
    except Exception as ex:  # pragma: no cover — reported via /healthz
        _state["error"] = f"{type(ex).__name__}: {ex}"
        print("[tts] load failed:", _state["error"], flush=True)


@app.on_event("startup")
def _startup() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=_load, daemon=True).start()


class SpeakRequest(BaseModel):
    text: str


@app.get("/healthz")
def healthz():
    return {"ok": True, "ready": _state["model"] is not None, "error": _state["error"],
            "device": "cuda" if torch.cuda.is_available() else "cpu", "out_dir": str(OUT_DIR)}


@app.post("/speak")
def speak(req: SpeakRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(400, "empty text")
    model = _state["model"]
    if model is None:
        raise HTTPException(503, _state["error"] or "model still loading")
    with _lock:
        t0 = time.time()
        wav = model.generate(text, audio_prompt_path=VOICE).detach().float().cpu()
        if wav.dim() == 1:
            wav = wav[None]
        peak = float(wav.abs().max())
        if peak > 1e-6:
            wav = wav * (0.89 / peak)                       # ≈ −1 dBFS
        slot = _state["slot"] = (_state["slot"] + 1) % SLOTS
        path = OUT_DIR / f"speak_{slot}.wav"
        torchaudio.save(str(path), wav, model.sr)
        ms = (time.time() - t0) * 1000.0
    return {"path": str(path).replace("\\", "/"), "duration": wav.shape[-1] / model.sr,
            "ms": round(ms, 1), "text": text}
