"""Local text-to-speech service for the Live tab → TouchDesigner voice.

Runs Chatterbox-Turbo (ResembleAI, MIT, 350M, English) on the GPU in its own
environment — its pins (torch 2.6, transformers 5.2, numpy < 2) clash with the
main delyrism backend, so it is a separate process:

    uv venv --python 3.11 .venv-tts
    uv pip install --python .venv-tts/Scripts/python.exe torch==2.6.0 torchaudio==2.6.0 \
        --index-url https://download.pytorch.org/whl/cu126
    uv pip install --python .venv-tts/Scripts/python.exe -r web/tts/requirements.txt
    .venv-tts/Scripts/python.exe -m uvicorn tts_server:app --app-dir web/tts --port 8010

The voice: Chatterbox clones timbre and delivery from a reference clip —
web/tts/voices/*.wav (calm, slow readings rendered by make_voices.py; "default"
= the model's built-in voice).  Each spoken word is then slowed down with
Rubber Band (pitch-preserving, `stretch` × longer) and lowered by `pitch`
semitones, for a slow, grounded, meditative delivery.

The WAV has three channels, all the same length (the reverb tail included):
    chan1  the dry voice
    chan2  hall reverb, left    — 100 % wet, a synthetic diffuse impulse
    chan3  hall reverb, right     response (decorrelated noise, exponential
                                  decay, darker as it decays), `predelay` s
                                  then `decay` s to −60 dB
TouchDesigner mixes, filters and moves these live (Controls › Voice).

POST /speak {"text": "..."} → {"path", "duration", "voice_duration", "ms", "text"}
    writes DELYRISM_TTS_DIR (default td/runtime/tts)/speak_{0..7}.wav, cycling
    so TouchDesigner can still be playing one while the next is written.
GET/POST /config {voice, stretch, pitch, decay, predelay, suffix} → the config
GET /voices → available voices · GET /healthz → {"ok", "ready", "device"}

Env: DELYRISM_TTS_VOICE = a voice name or a path to a ≥ 5 s reference WAV.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.signal import fftconvolve, butter, sosfilt

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(os.environ.get("DELYRISM_TTS_DIR", ROOT / "td" / "runtime" / "tts"))
VOICE_DIR = Path(__file__).resolve().parent / "voices"
SLOTS = 8

app = FastAPI(title="delyrism TTS (Chatterbox-Turbo)")
_state = {"model": None, "slot": 0, "error": None, "conds": {}, "ir": None}
_lock = threading.Lock()


class Config(BaseModel):
    voice: str = os.environ.get("DELYRISM_TTS_VOICE") or "george"
    stretch: float = 1.6       # × longer, pitch preserved (1 = as synthesised)
    pitch: float = -1.0        # semitones
    decay: float = 6.0         # reverb RT60, s
    predelay: float = 0.06     # reverb pre-delay, s
    suffix: str = "."          # appended to the word: "." → a settled, falling close


_cfg = Config()


class ConfigUpdate(BaseModel):
    voice: Optional[str] = None
    stretch: Optional[float] = None
    pitch: Optional[float] = None
    decay: Optional[float] = None
    predelay: Optional[float] = None
    suffix: Optional[str] = None


def voices() -> list[str]:
    return ["default"] + sorted(p.stem for p in VOICE_DIR.glob("*.wav"))


def _use_voice(model, voice: str) -> None:
    """Switch the model's conditionals (cached per voice: preparing costs ~0.5 s)."""
    conds = _state["conds"]
    if voice not in conds:
        if voice == "default":
            raise KeyError(voice)                            # captured at load time
        path = VOICE_DIR / f"{voice}.wav"
        if not path.exists():
            path = Path(voice)
        if not path.exists():
            raise FileNotFoundError(f"unknown voice {voice!r}; have {voices()}")
        model.prepare_conditionals(str(path))
        conds[voice] = model.conds
    model.conds = conds[voice]


def _impulse(sr: int, decay: float, predelay: float) -> np.ndarray:
    """Stereo diffuse-hall impulse response (2 × n): decorrelated noise with an
    exponential envelope, darkening as it decays (each band its own RT60)."""
    key = (sr, round(decay, 3), round(predelay, 3))
    if _state["ir"] and _state["ir"][0] == key:
        return _state["ir"][1]
    rng = np.random.default_rng(7)
    n = int(sr * (predelay + decay * 1.05))
    t = np.arange(n) / sr - predelay
    ir = np.zeros((2, n))
    bands = [(None, 600, 1.0), (600, 2500, 0.75), (2500, 7000, 0.45), (7000, None, 0.25)]
    for lo, hi, rt in bands:                                 # rt = fraction of the full RT60
        noise = rng.standard_normal((2, n))
        if lo and hi:
            sos = butter(2, [lo, hi], "bandpass", fs=sr, output="sos")
        elif hi:
            sos = butter(2, hi, "lowpass", fs=sr, output="sos")
        else:
            sos = butter(2, lo, "highpass", fs=sr, output="sos")
        env = np.where(t >= 0, 10 ** (-3 * np.clip(t, 0, None) / max(0.1, decay * rt)), 0.0)
        ir += sosfilt(sos, noise, axis=1) * env
    ramp = np.clip(t / 0.03, 0, 1)                           # soft onset after the pre-delay
    ir *= ramp
    ir /= np.sqrt((ir ** 2).sum(axis=1, keepdims=True)) + 1e-9
    _state["ir"] = (key, ir)
    return ir


def _shape(wav: np.ndarray, sr: int, cfg: Config) -> np.ndarray:
    """Slow down (pitch preserved) and lower the voice."""
    if abs(cfg.stretch - 1.0) < 1e-3 and abs(cfg.pitch) < 1e-3:
        return wav
    import pedalboard
    out = pedalboard.time_stretch(wav[None].astype(np.float32), sr,
                                  stretch_factor=1.0 / max(0.25, cfg.stretch),
                                  pitch_shift_in_semitones=cfg.pitch, high_quality=True)
    return out[0]


def _load() -> None:
    try:
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        device = "cuda" if torch.cuda.is_available() else "cpu"
        t0 = time.time()
        model = ChatterboxTurboTTS.from_pretrained(device=device)
        _state["conds"]["default"] = model.conds
        _use_voice(model, _cfg.voice)
        model.generate("ready")                              # warm-up: the first call is slow
        _shape(np.zeros(2400, np.float32), model.sr, _cfg)
        _state["model"] = model
        print(f"[tts] Chatterbox-Turbo ready on {device} in {time.time() - t0:.1f}s, voice {_cfg.voice}",
              flush=True)
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
            "device": "cuda" if torch.cuda.is_available() else "cpu", "out_dir": str(OUT_DIR),
            "voice": _cfg.voice}


@app.get("/voices")
def list_voices():
    return {"voices": voices(), "current": _cfg.voice}


@app.get("/config")
def get_config():
    return _cfg.model_dump()


@app.post("/config")
def set_config(upd: ConfigUpdate):
    global _cfg
    new = _cfg.model_copy(update={k: v for k, v in upd.model_dump().items() if v is not None})
    new.stretch = float(np.clip(new.stretch, 0.5, 4.0))
    new.pitch = float(np.clip(new.pitch, -12.0, 12.0))
    new.decay = float(np.clip(new.decay, 0.3, 20.0))
    new.predelay = float(np.clip(new.predelay, 0.0, 0.5))
    model = _state["model"]
    if model is not None and new.voice != _cfg.voice:
        with _lock:
            try:
                _use_voice(model, new.voice)
            except (KeyError, FileNotFoundError) as ex:
                raise HTTPException(404, str(ex))
    _cfg = new
    return _cfg.model_dump()


@app.post("/speak")
def speak(req: SpeakRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(400, "empty text")
    model = _state["model"]
    if model is None:
        raise HTTPException(503, _state["error"] or "model still loading")
    cfg = _cfg
    if cfg.suffix and text[-1].isalnum():
        text += cfg.suffix
    with _lock:
        t0 = time.time()
        sr = model.sr
        dry = model.generate(text).detach().float().cpu().numpy().reshape(-1)
        dry = _shape(dry, sr, cfg)
        ir = _impulse(sr, cfg.decay, cfg.predelay)
        wet = np.stack([fftconvolve(dry, ir[c]) for c in range(2)])
        n = wet.shape[1]
        out = np.zeros((n, 3), np.float32)
        out[: len(dry), 0] = dry
        out[:, 1:] = wet.T
        out[:, 0] *= 0.89 / max(1e-6, float(np.abs(out[:, 0]).max()))   # dry ≈ −1 dBFS
        out[:, 1:] *= 0.89 / max(1e-6, float(np.abs(out[:, 1:]).max()))  # wet likewise; TD sets the mix
        slot = _state["slot"] = (_state["slot"] + 1) % SLOTS
        path = OUT_DIR / f"speak_{slot}.wav"
        sf.write(str(path), out, sr, subtype="FLOAT")
        ms = (time.time() - t0) * 1000.0
    return {"path": str(path).replace("\\", "/"), "duration": n / sr, "voice_duration": len(dry) / sr,
            "ms": round(ms, 1), "text": text}
