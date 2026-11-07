"""Speak the query's most activated word — client of the TTS service (web/tts).

When the text query is edited, the live socket waits `delay` seconds after
the last edit, takes the node with the highest query relevance (the brightest
word in the graph), has the TTS service synthesise it and sends
`/delyrism/speak s word  s wav-path  f seconds` over OSC to TouchDesigner,
which plays it.  The same word is not spoken twice in a row unless `repeat`.

Configuration: env DELYRISM_SPEAK (1/0), DELYRISM_SPEAK_DELAY (s),
DELYRISM_TTS_URL (default http://127.0.0.1:8010) — or GET/POST /realtime/speech.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() not in ("0", "false", "no", "off", "")


class SpeechConfig:
    def __init__(self) -> None:
        self.enabled = _env_bool("DELYRISM_SPEAK", True)
        self.delay = float(os.environ.get("DELYRISM_SPEAK_DELAY", "1.0"))
        self.url = os.environ.get("DELYRISM_TTS_URL", "http://127.0.0.1:8010").rstrip("/")
        self.repeat = False
        self.last_word: Optional[str] = None
        self.last_error: Optional[str] = None
        self.last_ms: Optional[float] = None

    def configure(self, **kw: Any) -> Dict[str, Any]:
        for key, cast in (("enabled", bool), ("delay", float), ("url", str), ("repeat", bool)):
            if kw.get(key) is not None:
                setattr(self, key, cast(kw[key]))
        self.delay = max(0.0, self.delay)
        self.url = self.url.rstrip("/")
        return self.status()

    def status(self) -> Dict[str, Any]:
        try:
            tts = requests.get(f"{self.url}/healthz", timeout=1.0).json()
        except Exception as ex:
            tts = {"reachable": False, "error": str(ex)[:120]}
        return dict(enabled=self.enabled, delay=self.delay, url=self.url, repeat=self.repeat,
                    last_word=self.last_word, last_ms=self.last_ms, last_error=self.last_error, tts=tts)


cfg = SpeechConfig()


def synthesize(text: str) -> Optional[Dict[str, Any]]:
    """Blocking call to the TTS service → {path, duration, ms, text} or None."""
    try:
        r = requests.post(f"{cfg.url}/speak", json={"text": text}, timeout=30)
        r.raise_for_status()
        out = r.json()
        cfg.last_word, cfg.last_ms, cfg.last_error = text, out.get("ms"), None
        return out
    except Exception as ex:
        cfg.last_error = f"{type(ex).__name__}: {str(ex)[:200]}"
        return None
