"""Speech to text through Cloudflare Workers AI Whisper.

The Live tab's speech mode streams short microphone chunks over the socket
(binary kind 7); each chunk is transcribed here and the words join a rolling
transcript that queries the space through its text tower, so speech works
with every embedder, the hosted Cloudflare ones included.

    text = transcribe(wave, sr)                 # float32 mono, any rate
    available()                                 # Cloudflare credentials present?

Env: CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN (the embedders' own),
optional CLOUDFLARE_GATEWAY_ID / CLOUDFLARE_GATEWAY_TOKEN, and
DELYRISM_STT_MODEL (default @cf/openai/whisper-large-v3-turbo, about
$0.0005 per audio minute).
"""
from __future__ import annotations

import base64
import os
from typing import Optional

import numpy as np

from ..wav import SR, to_wav16k  # noqa: F401  (re-exported for callers)

MODEL = "@cf/openai/whisper-large-v3-turbo"
SILENCE_RMS = 0.004       # below this the chunk is room noise: Whisper would invent words


def _creds():
    return os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip(), os.environ.get("CLOUDFLARE_API_TOKEN", "").strip()


def available() -> bool:
    acc, tok = _creds()
    return bool(acc and tok)


def _url(account: str, model: str) -> str:
    gw = os.environ.get("CLOUDFLARE_GATEWAY_ID", "").strip()
    if gw:
        return f"https://gateway.ai.cloudflare.com/v1/{account}/{gw}/workers-ai/{model}"
    return f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{model}"


def is_silent(wave: np.ndarray) -> bool:
    x = np.asarray(wave, dtype=np.float32)
    return len(x) == 0 or float(np.sqrt(np.mean(x * x))) < SILENCE_RMS


def transcribe(wave: np.ndarray, sr: int, language: Optional[str] = None,
               prompt: Optional[str] = None, timeout: float = 30.0) -> str:
    """The words in a mono chunk ("" for silence).  `prompt` is the recent
    transcript, which keeps names and spelling consistent across chunks."""
    if is_silent(wave):
        return ""
    account, token = _creds()
    if not (account and token):
        raise RuntimeError("speech to text needs CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN")
    import requests

    model = os.environ.get("DELYRISM_STT_MODEL", MODEL)
    body = {"audio": base64.b64encode(to_wav16k(wave, sr)).decode("ascii"), "vad_filter": True,
            "condition_on_previous_text": False}
    if language:
        body["language"] = language
    if prompt:
        body["initial_prompt"] = prompt[-200:]
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    gw_tok = os.environ.get("CLOUDFLARE_GATEWAY_TOKEN", "").strip()
    if gw_tok and os.environ.get("CLOUDFLARE_GATEWAY_ID"):
        headers["cf-aig-authorization"] = f"Bearer {gw_tok}"
    r = requests.post(_url(account, model), json=body, headers=headers, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"Cloudflare Whisper {r.status_code}: {r.text[:300]}")
    data = r.json()
    result = data.get("result", data) if isinstance(data, dict) else {}
    return str((result or {}).get("text") or "").strip()
