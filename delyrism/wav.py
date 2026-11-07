"""Mono float audio → 16 kHz 16-bit WAV bytes, the format the speech and
audio-embedding APIs take (Cloudflare Whisper, Gemini Embedding 2)."""
from __future__ import annotations

import io
import wave as _wave

import numpy as np

SR = 16000


def to_wav16k(wave: np.ndarray, sr: int) -> bytes:
    """Mono float32 at any rate → 16 kHz 16-bit PCM WAV bytes."""
    x = np.asarray(wave, dtype=np.float32).reshape(-1)
    if sr != SR and len(x) > 1:
        n = int(round(len(x) * SR / sr))
        x = np.interp(np.linspace(0.0, 1.0, n, endpoint=False),
                      np.linspace(0.0, 1.0, len(x), endpoint=False), x).astype(np.float32)
    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with _wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()
