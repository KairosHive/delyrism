"""Live profile — the server-wide defaults a live session starts from.

A host application (an installation such as do-you-dream, or any client that
does not want to send dozens of parameters) sets one profile, either at start

    python -m delyrism.server --profile live_profile.json
    create_app(profile=LiveProfile(...))

or at run time (GET/POST /live/profile), then builds its space in one call
(POST /live/setup, see routes/live.py).  Every live graph socket opened
afterwards (/realtime/graph/{space_id}) starts from the profile's graph
params, layout shape, pose options and smoothing; a client can still override
any of them per socket with the usual messages.

Without a profile everything behaves as before (the defaults below are the
engine's own).
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from .schemas import EmbedderConfig


class OscSettings(BaseModel):
    """OSC stream of the live graph (osc_bridge.py); None fields keep the current value."""
    enabled: Optional[bool] = None
    host: Optional[str] = None
    port: Optional[int] = None
    hz: Optional[float] = None
    scale: Optional[float] = None


class SpeechSettings(BaseModel):
    """Voice of the top activated word (speech.py, needs the TTS service); None keeps the current value."""
    enabled: Optional[bool] = None
    url: Optional[str] = None
    delay: Optional[float] = None
    media: Optional[bool] = None          # also speak in movement modes
    media_kinds: Optional[List[str]] = None   # kinds that speak on a top-node change: "pose", "motion", "speech"
                                              # (others speak `delay` s after the query stops changing)
    min_gap: Optional[float] = None
    mood: Optional[bool] = None
    repeat: Optional[bool] = None


class MotionSettings(BaseModel):
    """Movement mode.  `auto` = the server describes the movement every
    `refresh` s while landmark frames (kind 6) stream in, so a client only
    streams landmarks; off = the client sends {"type": "motion_tick", …}."""
    auto: bool = False
    refresh: float = Field(3.0, ge=0.5, le=30.0)
    window: float = Field(3.0, ge=1.0, le=10.0)
    level: Literal["full", "abstract", "qualities", "body"] = "full"
    contrast: float = Field(0.7, ge=0.0, le=1.0)
    relative: float = Field(1.0, ge=0.0, le=1.0)


class TranscriptSettings(BaseModel):
    """Spoken-text queries ({"type": "transcript", "text": …}): each new text
    sets a target embedding and the query vector glides toward it with time
    constant `tau` s, re-scored `hz` times a second — latent smoothing, so the
    graph follows speech continuously instead of jumping at every update."""
    tau: float = Field(1.5, ge=0.0, le=30.0)     # 0 = jump straight to each new text
    hz: float = Field(20.0, ge=1.0, le=60.0)


class LiveProfile(BaseModel):
    # ---- the space (POST /live/setup) ----
    embedder: EmbedderConfig = Field(default_factory=lambda: EmbedderConfig(backend="sentence-transformer"))
    descriptor_threshold: float = 0.2
    contextual_embeddings: bool = False
    palette: str = "AuroraPop"
    # global mode: "descriptor" removes the descriptor offset and the top
    # `global_pcs` principal directions from the Δ geometry; "none" = raw
    global_mode: Literal["none", "descriptor"] = "none"
    global_pcs: int = Field(1, ge=0, le=5)
    # ---- every new live graph socket ----
    graph: Dict[str, Any] = Field(default_factory=dict)      # overrides of the live graph params (realtime._GRAPH_DEFAULTS)
    shape: Dict[str, float] = Field(default_factory=dict)    # layout knobs (live_sim.SHAPE_DEFAULTS: focus, contraction, …)
    pose: Dict[str, Any] = Field(default_factory=dict)       # pose options (tracker, trace, fit, joints, smooth)
    audio_smoothing: Optional[float] = None
    rel_smoothing: Optional[float] = None
    motion: MotionSettings = Field(default_factory=MotionSettings)
    transcript: TranscriptSettings = Field(default_factory=TranscriptSettings)
    # Lakota background (lakota.py): match the activated words to the ten Lakota
    # symbols by meaning → /delyrism/lakota* OSC + the "lakota" block of sim
    # frames.  Off by default (an embedding pass per space and per query).
    lakota: bool = False
    # ---- server-wide outputs, applied when the profile is set ----
    osc: Optional[OscSettings] = None
    speech: Optional[SpeechSettings] = None


_lock = threading.Lock()
_profile = LiveProfile()


def current() -> LiveProfile:
    with _lock:
        return _profile


def load(src: Union[str, Path, Dict[str, Any], LiveProfile]) -> LiveProfile:
    """A profile from a JSON file path, a dict or a LiveProfile."""
    if isinstance(src, LiveProfile):
        return src
    if isinstance(src, (str, Path)):
        src = json.loads(Path(src).read_text(encoding="utf-8"))
    return LiveProfile.model_validate(src)


def apply(src: Union[str, Path, Dict[str, Any], LiveProfile]) -> LiveProfile:
    """Make `src` the current profile and push its OSC / speech settings."""
    global _profile
    prof = load(src)
    from .osc_bridge import bridge as osc
    from . import speech
    if prof.osc is not None:
        osc.configure(**prof.osc.model_dump(exclude_none=True))
    if prof.speech is not None:
        speech.cfg.configure(**prof.speech.model_dump(exclude_none=True))
    with _lock:
        _profile = prof
    return prof
