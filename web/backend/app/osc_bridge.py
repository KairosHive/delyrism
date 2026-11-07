"""OSC stream of the live Δ-graph for TouchDesigner (or anything that speaks OSC).

One UDP target (default 127.0.0.1:7000).  Every live-graph socket runs its
own `LiveSim`; only the *owner* — the socket that most recently received
input — streams, so two open browser tabs don't fight over the output.

Wire format — columnar: one message per attribute, one argument per node /
edge, so a frame is ~30 messages regardless of graph size.  Each frame goes
out as ONE OSC bundle (falls back to loose messages if it would exceed a
UDP datagram, or when `bundle` is off).  Row i of every /node/* array is the
same node; row j of every /edge/* array is the same edge.

  /delyrism/frame      i seq  f time  i n_nodes  i n_edges  i graph_seq  i query_active  i awake
  /delyrism/bounds     f xmin  f ymin  f xmax  f ymax                 (all nodes, output units)

  /delyrism/node/uid     i×N   stable id per descriptor (survives graph changes; use it to key trails)
  /delyrism/node/x       f×N   position, output units (≈ −1…1, +y up)
  /delyrism/node/y       f×N
  /delyrism/node/vx      f×N   velocity, output units / s
  /delyrism/node/vy      f×N
  /delyrism/node/size    f×N   Σ|Δ| of the node's edges, min-max → 0…1  (browser: node radius)
  /delyrism/node/rel     f×N   query relevance after the focus exponent, 0…1 (0 when no query)
  /delyrism/node/pull    f×N   raw context pull, cos(D′_i, v_ctx)
  /delyrism/node/r|g|b   f×N   archetype colour, 0…1
  /delyrism/node/sym     i×N   archetype index into /delyrism/symbols
  /delyrism/node/alive   i×N   1 = in the current graph, 0 = fading out
  /delyrism/node/fade    f×N   appear / disappear envelope 0…1

  /delyrism/edge/src     i×E   row index into the /node/* arrays of the same frame
  /delyrism/edge/dst     i×E
  /delyrism/edge/x0|y0   f×E   endpoint positions, copied from the nodes (ready for particles)
  /delyrism/edge/x1|y1   f×E
  /delyrism/edge/weight  f×E   |Δ| min-max → 0…1  (browser: line width)
  /delyrism/edge/delta   f×E   signed Δ cosine (after − before)
  /delyrism/edge/sign    i×E   +1 strengthens (teal) · −1 weakens (red)
  /delyrism/edge/overlap f×E   sqrt(rel_src · rel_dst) — how much the query lights this edge
  /delyrism/edge/len     f×E   current length, output units
  /delyrism/edge/alive   i×E
  /delyrism/edge/fade    f×E

  /delyrism/lakota       f×K   activation of each Lakota background symbol (app/lakota.py):
                               the graph's activated words matched to the Lakota symbols, for
                               any space; order = /delyrism/lakota/names, never all zero

  /delyrism/end          i seq     — last message of the frame: commit here

Event, outside the frame bundles (app/speech.py):
  /delyrism/speak        s word  s wav path  f seconds  s mood  — the query's top word, synthesised in a
                         mood sampled from the queried sub-network (neutral, happy, angry, fear, sigh,
                         whispering, dramatic); play it

Sent with the frame when the node order changed, and once a second:
  /delyrism/node/label   s×N   descriptor text
  /delyrism/symbols      s×S   archetype names (index = /node/sym)
  /delyrism/context      s
  /delyrism/query        s text  s kind (text | audio | image | video | av | "")
  /delyrism/lakota/names s×K   Lakota symbol names (delyrism/structures/lakota.json order)

Configuration: env DELYRISM_OSC (1/0), DELYRISM_OSC_HOST, DELYRISM_OSC_PORT,
DELYRISM_OSC_HZ, DELYRISM_OSC_SCALE (sim px per output unit, default 300),
DELYRISM_OSC_BUNDLE (1/0) — or at runtime via GET/POST /realtime/osc.
"""
from __future__ import annotations

import os
import socket
import struct
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

MAX_DATAGRAM = 60_000
PREFIX = "/delyrism"


# ---------------------------------------------------------------- encoding
def _s(text: str) -> bytes:
    b = str(text).replace("\x00", "").encode("utf-8")
    return b + b"\x00" * (4 - len(b) % 4)


def _msg(addr: str, tags: str, payload: bytes) -> bytes:
    return _s(addr) + _s("," + tags) + payload


def floats(addr: str, values) -> bytes:
    a = np.asarray(values, dtype=">f4").ravel()
    return _msg(addr, "f" * a.size, a.tobytes())


def ints(addr: str, values) -> bytes:
    a = np.asarray(values, dtype=">i4").ravel()
    return _msg(addr, "i" * a.size, a.tobytes())


def strings(addr: str, values: Sequence[str]) -> bytes:
    return _msg(addr, "s" * len(values), b"".join(_s(v) for v in values))


def mixed(addr: str, *args: Any) -> bytes:
    tags, body = "", b""
    for v in args:
        if isinstance(v, bool) or isinstance(v, (int, np.integer)):
            tags += "i"; body += struct.pack(">i", int(v))
        elif isinstance(v, (float, np.floating)):
            tags += "f"; body += struct.pack(">f", float(v))
        else:
            tags += "s"; body += _s(str(v))
    return _msg(addr, tags, body)


def bundle(messages: Sequence[bytes]) -> bytes:
    out = [b"#bundle\x00", struct.pack(">Q", 1)]  # timetag 1 = immediately
    for m in messages:
        out.append(struct.pack(">i", len(m)))
        out.append(m)
    return b"".join(out)


def frame_messages(sim, seq: int, t: float, scale: float, meta: Dict[str, Any],
                   labels: bool) -> List[bytes]:
    """Encode one LiveSim frame in the format documented above."""
    P = PREFIX
    k = 1.0 / max(1e-9, scale)
    x, y = sim.x * k, -sim.y * k            # sim is y-down (canvas); output is y-up
    vx, vy = sim.vx * k, -sim.vy * k
    n, e = len(sim.ids), len(sim.e_a)
    rel = sim.rel if sim.query_active else np.zeros(n)
    xmin, ymin, xmax, ymax = (float(x.min()), float(y.min()), float(x.max()), float(y.max())) if n else (0.0,) * 4
    a, b = sim.e_a, sim.e_b
    x0, y0, x1, y1 = x[a], y[a], x[b], y[b]
    msgs = [
        mixed(f"{P}/frame", int(seq), float(t), n, e, int(sim.graph_seq), int(sim.query_active), int(sim.awake)),
        floats(f"{P}/bounds", [xmin, ymin, xmax, ymax]),
        ints(f"{P}/node/uid", sim.uid),
        floats(f"{P}/node/x", x), floats(f"{P}/node/y", y),
        floats(f"{P}/node/vx", vx), floats(f"{P}/node/vy", vy),
        floats(f"{P}/node/size", sim.size), floats(f"{P}/node/rel", rel), floats(f"{P}/node/pull", sim.pull),
        floats(f"{P}/node/r", sim.rgb[:, 0]), floats(f"{P}/node/g", sim.rgb[:, 1]), floats(f"{P}/node/b", sim.rgb[:, 2]),
        ints(f"{P}/node/sym", sim.sym), ints(f"{P}/node/alive", sim.alive.astype(np.int32)),
        floats(f"{P}/node/fade", sim.fade),
        ints(f"{P}/edge/src", a), ints(f"{P}/edge/dst", b),
        floats(f"{P}/edge/x0", x0), floats(f"{P}/edge/y0", y0),
        floats(f"{P}/edge/x1", x1), floats(f"{P}/edge/y1", y1),
        floats(f"{P}/edge/weight", sim.e_weight), floats(f"{P}/edge/delta", sim.e_delta),
        ints(f"{P}/edge/sign", sim.e_sign),
        floats(f"{P}/edge/overlap", sim.e_overlap if sim.query_active else np.zeros(e)),
        floats(f"{P}/edge/len", np.hypot(x1 - x0, y1 - y0)),
        ints(f"{P}/edge/alive", sim.e_alive.astype(np.int32)), floats(f"{P}/edge/fade", sim.e_fade),
    ]
    lak = sim.lakota_activation() if hasattr(sim, "lakota_activation") else None
    if lak is not None:
        msgs.append(floats(f"{P}/lakota", lak))
    if labels:
        msgs += [
            strings(f"{P}/node/label", sim.ids),
            strings(f"{P}/symbols", sim.symbols),
            strings(f"{P}/context", [meta.get("context", "")]),
            strings(f"{P}/query", [meta.get("query", ""), meta.get("kind", "")]),
        ]
        if getattr(sim, "lakota_names", None):
            msgs.append(strings(f"{P}/lakota/names", sim.lakota_names))
    msgs.append(ints(f"{P}/end", [int(seq)]))
    return msgs


# ---------------------------------------------------------------- transport
def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() not in ("0", "false", "no", "off", "")


class OscBridge:
    def __init__(self) -> None:
        self.enabled = _env_bool("DELYRISM_OSC", True)
        self.host = os.environ.get("DELYRISM_OSC_HOST", "127.0.0.1")
        self.port = int(os.environ.get("DELYRISM_OSC_PORT", "7000"))
        self.hz = float(os.environ.get("DELYRISM_OSC_HZ", "60"))
        self.scale = float(os.environ.get("DELYRISM_OSC_SCALE", "300"))
        self.bundle = _env_bool("DELYRISM_OSC_BUNDLE", True)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._lock = threading.Lock()
        self._sessions: List[object] = []
        self._info: Dict[int, Callable[[], Dict[str, Any]]] = {}
        self.owner: Optional[object] = None
        self.owner_epoch = 0
        self.frames_sent = 0
        self.last_error: Optional[str] = None
        self.last_bytes = 0

    # ---- ownership: the most recently active socket streams
    def register(self, token: object, info: Optional[Callable[[], Dict[str, Any]]] = None) -> None:
        """`info` (optional) reports the session's state in status() while it owns the output."""
        with self._lock:
            self._sessions.append(token)
            if info is not None:
                self._info[id(token)] = info
            if self.owner is None:
                self._set_owner(token)

    def unregister(self, token: object) -> None:
        with self._lock:
            self._info.pop(id(token), None)
            if token in self._sessions:
                self._sessions.remove(token)
            if self.owner is token:
                self._set_owner(self._sessions[-1] if self._sessions else None)

    def claim(self, token: object) -> None:
        with self._lock:
            if self.owner is not token and token in self._sessions:
                self._sessions.remove(token)
                self._sessions.append(token)
                self._set_owner(token)

    def _set_owner(self, token: Optional[object]) -> None:
        self.owner = token
        self.owner_epoch += 1

    def is_owner(self, token: object) -> bool:
        return self.enabled and self.owner is token

    # ---- config
    def configure(self, **kw: Any) -> Dict[str, Any]:
        for key, cast in (("host", str), ("port", int), ("hz", float), ("scale", float),
                          ("enabled", bool), ("bundle", bool)):
            if kw.get(key) is not None:
                setattr(self, key, cast(kw[key]))
        self.hz = float(min(240.0, max(1.0, self.hz)))
        self.owner_epoch += 1    # resend labels to the (possibly new) target
        return self.status()

    def status(self) -> Dict[str, Any]:
        owner_info = None
        fn = self._info.get(id(self.owner)) if self.owner is not None else None
        if fn is not None:
            try:
                owner_info = fn()
            except Exception as ex:  # pragma: no cover — diagnostics only
                owner_info = {"error": str(ex)}
        return dict(enabled=self.enabled, host=self.host, port=self.port, hz=self.hz, scale=self.scale,
                    bundle=self.bundle, sessions=len(self._sessions), streaming=self.owner is not None,
                    frames_sent=self.frames_sent, last_bytes=self.last_bytes, last_error=self.last_error,
                    owner=owner_info)

    def send(self, messages: Sequence[bytes]) -> None:
        addr = (self.host, self.port)
        try:
            if self.bundle:
                pkt = bundle(messages)
                if len(pkt) <= MAX_DATAGRAM:
                    self._sock.sendto(pkt, addr)
                    self.last_bytes = len(pkt)
                    self.frames_sent += 1
                    self.last_error = None
                    return
            for m in messages:
                self._sock.sendto(m, addr)
            self.last_bytes = sum(len(m) for m in messages)
            self.frames_sent += 1
            self.last_error = None
        except OSError as ex:   # nobody listening on Windows → WSAECONNRESET etc.; keep going
            self.last_error = str(ex)


bridge = OscBridge()

# Windows: asyncio sleeps round up to the 15.6 ms system tick, which caps the
# OSC pump at ~32 Hz.  Ask for 1 ms timer resolution for this process.
# Windows 11 ignores that request from a process with no visible window (a
# backend started in the background, or a minimised console) unless the
# process opts out of timer-resolution power throttling first; measured here,
# a windowless backend otherwise streams at 31 ms intervals instead of 16.7.
if os.name == "nt":
    try:
        import ctypes
        from ctypes import wintypes

        class _PowerThrottlingState(ctypes.Structure):
            _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

        _k32 = ctypes.windll.kernel32
        _k32.GetCurrentProcess.restype = wintypes.HANDLE
        _k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        _k32.SetProcessInformation.restype = wintypes.BOOL
        # ProcessPowerThrottling (4); IGNORE_TIMER_RESOLUTION (0x4) in ControlMask
        # and cleared in StateMask = always honour this process's timer resolution
        _state = _PowerThrottlingState(1, 0x4, 0)
        _k32.SetProcessInformation(_k32.GetCurrentProcess(), 4, ctypes.byref(_state), ctypes.sizeof(_state))
        ctypes.windll.winmm.timeBeginPeriod(1)
    except Exception:  # pragma: no cover — best effort
        pass
