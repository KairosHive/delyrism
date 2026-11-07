"""Speak the query's most activated word — client of a TTS service (the do-you-dream repo ships one: tts/tts_server.py).

When the text query is edited, the live socket waits `delay` seconds after
the last edit, takes the node with the highest query relevance (the brightest
word in the graph), has the TTS service synthesise it and sends
`/delyrism/speak s word  s wav-path  f seconds` over OSC to TouchDesigner,
which plays it.  The same word is not spoken twice in a row unless `repeat`.

Movement (MediaPipe) queries — kinds in `media_kinds`, i.e. pose / motion —
have no edits to wait for: the word is spoken whenever the top activated node
changes and then stays on top for `delay` s, at most once every `min_gap` s
(the slowed voice and its reverb take a few seconds to ring out).

Mood: the word is spoken in a mood sampled from the queried sub-network.
Every descriptor of the space has a distribution over MOODS (the Chatterbox
style tags that measurably change delivery).  It comes from the cosine
similarity between the descriptor's embedding and an anchor description of
each mood (same embedder), standardised per mood, softmaxed with `mood_tau`
and then Sinkhorn-balanced: every row stays a distribution while every mood
column receives the same total mass over all descriptors — so across the
space each mood is equally likely and no tone dominates by default.  At speak
time the distributions of the nodes the query activates (relevance ≥
`mood_threshold`; the top node if none) are averaged, weighted by relevance,
sharpened by 1/`mood_temperature`, and a mood is sampled from the result.

Configuration: env DELYRISM_SPEAK (1/0), DELYRISM_SPEAK_DELAY (s), DELYRISM_SPEAK_GAP (s),
DELYRISM_TTS_URL (default http://127.0.0.1:8010) — or GET/POST /realtime/speech.
GET /realtime/speech/moods/{space_id} shows the balance and each mood's most
typical descriptors.
"""
from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import requests

# (mood, Chatterbox-Turbo tag or None, anchor description embedded with the space's model)
MOODS: List[Tuple[str, Optional[str], str]] = [
    ("neutral", None, "calm, steady, ordinary, plain, everyday, matter of fact"),
    ("happy", "[happy]", "joy, delight, celebration, warmth, laughter, hope"),
    ("angry", "[angry]", "anger, rage, fury, violence, conflict, aggression"),
    ("fear", "[fear]", "fear, dread, danger, terror, threat, anxiety"),
    ("sigh", "[sigh]", "sorrow, grief, loss, weariness, longing, melancholy"),
    ("whispering", "[whispering]", "secret, hush, intimacy, sacred silence, mystery, tenderness"),
    ("dramatic", "[dramatic]", "grandeur, power, fate, awe, revelation, epic struggle"),
]


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() not in ("0", "false", "no", "off", "")


class SpeechConfig:
    def __init__(self) -> None:
        self.enabled = _env_bool("DELYRISM_SPEAK", True)
        self.delay = float(os.environ.get("DELYRISM_SPEAK_DELAY", "1.0"))
        self.url = os.environ.get("DELYRISM_TTS_URL", "http://127.0.0.1:8010").rstrip("/")
        self.repeat = False
        self.media = True                # also speak on top-node changes in movement modes
        self.media_kinds = ("pose", "motion")
        self.min_gap = float(os.environ.get("DELYRISM_SPEAK_GAP", "4.0"))   # s between movement-driven words
        self.mood = True                 # speak in a sampled mood (else always neutral)
        self.mood_threshold = 0.5        # relevance (0–1) for a node to count as activated
        self.mood_temperature = 1.0      # < 1 sharpens the averaged distribution, > 1 flattens it
        self.mood_tau = 0.5              # softness of each descriptor's distribution (re-balanced on change)
        self.last_word: Optional[str] = None
        self.last_mood: Optional[Dict[str, Any]] = None
        self.last_error: Optional[str] = None
        self.last_ms: Optional[float] = None

    def configure(self, **kw: Any) -> Dict[str, Any]:
        for key, cast in (("enabled", bool), ("delay", float), ("url", str), ("repeat", bool),
                          ("media", bool), ("min_gap", float),
                          ("mood", bool), ("mood_threshold", float), ("mood_temperature", float),
                          ("mood_tau", float)):
            if kw.get(key) is not None:
                setattr(self, key, cast(kw[key]))
        self.delay = max(0.0, self.delay)
        self.min_gap = max(0.0, self.min_gap)
        self.mood_temperature = max(0.05, self.mood_temperature)
        self.mood_tau = max(0.05, self.mood_tau)
        self.url = self.url.rstrip("/")
        if kw.get("media_kinds") is not None:
            self.media_kinds = tuple(k for k in kw["media_kinds"] if k in ("pose", "motion", "speech"))
        return self.status()

    def status(self) -> Dict[str, Any]:
        try:
            tts = requests.get(f"{self.url}/healthz", timeout=1.0).json()
        except Exception as ex:
            tts = {"reachable": False, "error": str(ex)[:120]}
        return dict(enabled=self.enabled, delay=self.delay, url=self.url, repeat=self.repeat,
                    media=self.media, media_kinds=list(self.media_kinds), min_gap=self.min_gap,
                    mood=self.mood, mood_threshold=self.mood_threshold,
                    mood_temperature=self.mood_temperature, mood_tau=self.mood_tau,
                    moods=[m[0] for m in MOODS], last_word=self.last_word, last_mood=self.last_mood,
                    last_ms=self.last_ms, last_error=self.last_error, tts=tts)


cfg = SpeechConfig()
_rng = np.random.default_rng()


# ---------------------------------------------------------------- moods
def _sinkhorn(M: np.ndarray, iters: int = 200) -> np.ndarray:
    """Rows → distributions, columns → equal total mass (N / K each)."""
    M = M.astype(np.float64).copy()
    n, k = M.shape
    for _ in range(iters):
        M /= M.sum(axis=1, keepdims=True)
        M *= (n / k) / M.sum(axis=0, keepdims=True)
    return M / M.sum(axis=1, keepdims=True)


_mood_cache: Dict[Tuple[int, float], np.ndarray] = {}
_mood_lock = threading.Lock()


def mood_matrix(space) -> np.ndarray:
    """N_descriptors × K mood distribution for this space (balanced, cached)."""
    key = (id(space), round(cfg.mood_tau, 4))
    with _mood_lock:
        if key in _mood_cache:
            return _mood_cache[key]
    A = np.stack([np.asarray(space._sent_vec(text, ignore_override=True), np.float64) for _, _, text in MOODS])
    A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-9
    D = np.asarray(space.D, np.float64)
    D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)
    S = D @ A.T                                              # cosine, N × K
    Z = (S - S.mean(axis=0)) / (S.std(axis=0) + 1e-9)        # remove each anchor's global bias
    Z = Z - Z.max(axis=1, keepdims=True)
    P = _sinkhorn(np.exp(Z / cfg.mood_tau))
    with _mood_lock:
        _mood_cache[key] = P
    return P


def choose_mood(space, relevance: Dict[str, float]) -> Dict[str, Any]:
    """Average the mood distributions of the activated nodes, then sample one."""
    P = mood_matrix(space)
    ids = [i for i in relevance if i in space.desc_idx]
    if not ids:
        return dict(mood="neutral", tag=None, probs={}, nodes=[])
    active = [i for i in ids if relevance[i] >= cfg.mood_threshold] or [max(ids, key=relevance.get)]
    w = np.array([max(1e-6, float(relevance[i])) for i in active])
    p = (w[:, None] * P[[space.desc_idx[i] for i in active]]).sum(axis=0) / w.sum()
    p = np.power(p, 1.0 / cfg.mood_temperature)
    p /= p.sum()
    k = int(_rng.choice(len(MOODS), p=p))
    name, tag, _ = MOODS[k]
    return dict(mood=name, tag=tag, probs={m[0]: round(float(x), 3) for m, x in zip(MOODS, p)},
                nodes=active[:12])


def mood_report(space, top: int = 6) -> Dict[str, Any]:
    """Balance check + the descriptors most typical of each mood."""
    P = mood_matrix(space)
    names = space.descriptors
    marginal = P.mean(axis=0)
    rep = {}
    for k, (m, _, _) in enumerate(MOODS):
        order = np.argsort(-P[:, k])[:top]
        rep[m] = [(names[i], round(float(P[i, k]), 3)) for i in order]
    argmax = np.bincount(P.argmax(axis=1), minlength=len(MOODS)) / len(names)
    return dict(n_descriptors=len(names), tau=cfg.mood_tau,
                mean_distribution={m[0]: round(float(x), 4) for m, x in zip(MOODS, marginal)},
                share_as_top_mood={m[0]: round(float(x), 3) for m, x in zip(MOODS, argmax)},
                most_typical=rep)


def synthesize(text: str, tag: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Blocking call to the TTS service → {path, duration, ms, text} or None.
    `tag` (a Chatterbox style tag such as "[fear]") is prepended to the text."""
    try:
        spoken = f"{tag} {text}" if tag else text
        r = requests.post(f"{cfg.url}/speak", json={"text": spoken}, timeout=30)
        r.raise_for_status()
        out = r.json()
        cfg.last_word, cfg.last_ms, cfg.last_error = text, out.get("ms"), None
        return out
    except Exception as ex:
        cfg.last_error = f"{type(ex).__name__}: {str(ex)[:200]}"
        return None
