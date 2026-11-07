"""Delyrism FastAPI backend — `delyrism.server`.

    python -m delyrism.server --port 8000 [--profile live.json] [--warmup]
    uvicorn delyrism.server.main:app --port 8000

`create_app()` builds an app (a host can also mount it inside its own
FastAPI); the module-level `app` is the default instance, configured from the
environment:

    DELYRISM_PROFILE      path to a LiveProfile JSON (see profile.py)
    DELYRISM_STATIC_DIR   a static frontend to serve at /  (default: the
                          repo's web/frontend/out when it exists)
    DELYRISM_WARMUP=1     load the profile's embedder at start (see /healthz)
    CORS_ORIGINS          extra allowed origins (every origin is allowed anyway)
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import profile as live_profile
from .routes import analysis, builder, context, delta, live, realtime, spaces, story, topology

# readiness for process managers: /healthz → "ready" once the warm-up (if any) is done
_state: Dict[str, Any] = {"ready": True, "warming": False, "error": None, "started": time.time()}


def warmup(prof: Optional[live_profile.LiveProfile] = None) -> None:
    """Load the profile's embedder and run one encode, so the first live
    session does not wait for a model download / load."""
    from . import engine_cache
    prof = prof or live_profile.current()
    _state.update(ready=False, warming=True, error=None)
    t0 = time.perf_counter()
    try:
        _, emb = engine_cache.get_embedder(prof.embedder.model_dump())
        emb.encode(["warm up"])
        print(f"[delyrism] warm: {prof.embedder.backend} in {time.perf_counter() - t0:.1f} s")
    except Exception as e:  # the server stays usable; /healthz reports it
        _state["error"] = f"warm-up failed: {e}"
        print(f"[delyrism] {_state['error']}")
    finally:
        _state.update(ready=True, warming=False)


def _default_static_dir() -> Optional[Path]:
    env = os.environ.get("DELYRISM_STATIC_DIR")
    if env:
        return Path(env)
    # a repo checkout: web/frontend/out (Next.js static export)
    p = Path(__file__).resolve().parents[2] / "web" / "frontend" / "out"
    return p if p.is_dir() else None


def create_app(profile: Union[None, str, Path, Dict[str, Any], live_profile.LiveProfile] = None,
               static_dir: Union[None, str, Path] = None, warm: bool = False) -> FastAPI:
    """The Delyrism API.  `profile` sets the live defaults (profile.py);
    `static_dir` serves a frontend at /; `warm` loads the profile's embedder
    in the background (/healthz says ready=false until it is loaded)."""
    if profile is not None:
        live_profile.apply(profile)

    app = FastAPI(
        title="Delyrism API",
        version="0.3.0",
        description="HTTP surface for the SymbolSpace engine and the live Δ-graph.",
    )

    # CORS — common local dev origins plus CORS_ORIGINS; the regex lets any
    # origin through so a narrower env setting never silently breaks a client.
    dev_origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
    env_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list({*dev_origins, *env_origins}),
        allow_origin_regex=".*",
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*", "X-Server-Ms", "Server-Timing"],
        max_age=600,
    )

    @app.middleware("http")
    async def server_timing(request: Request, call_next):
        """`Server-Timing` on every response, so the browser's Network tab
        shows the wall-clock time per route."""
        t0 = time.perf_counter()
        response = await call_next(request)
        ms = (time.perf_counter() - t0) * 1000.0
        existing = response.headers.get("Server-Timing", "")
        tag = f"app;dur={ms:.1f}"
        response.headers["Server-Timing"] = f"{existing}, {tag}" if existing else tag
        response.headers["X-Server-Ms"] = f"{ms:.1f}"
        return response

    for r in (spaces.router, analysis.router, delta.router, context.router, story.router,
              builder.router, builder.legacy_router,      # /miner alias kept for older clients
              topology.router, realtime.router, live.router):
        app.include_router(r)

    @app.get("/healthz")
    def healthz():
        """ok = the process answers; ready = the warm-up (if any) is done."""
        return {"ok": True, "ready": bool(_state["ready"]), "warming": bool(_state["warming"]),
                "error": _state["error"], "uptime": round(time.time() - _state["started"], 1)}

    @app.get("/api")
    def api_info():
        """Cheap probe — JSON even when a static frontend is mounted at /."""
        return {"ok": True, "name": "delyrism-api", "version": app.version}

    @app.get("/backends")
    def backends():
        """Embedder backends and Cloudflare model names — for the frontend dropdowns."""
        return {
            "embedders": [
                {"id": "cloudflare-bge-m3", "label": "Cloudflare · BGE-M3 (multilingual)", "remote": True, "audio": False, "dim": 1024},
                {"id": "cloudflare-qwen3", "label": "Cloudflare · Qwen3 Embedding", "remote": True, "audio": False, "dim": 1024},
                {"id": "cloudflare-embeddinggemma", "label": "Cloudflare · EmbeddingGemma 300M", "remote": True, "audio": False, "dim": 768},
                {"id": "cloudflare-bge-large", "label": "Cloudflare · BGE Large EN", "remote": True, "audio": False, "dim": 1024},
                {"id": "cloudflare-bge-base", "label": "Cloudflare · BGE Base EN", "remote": True, "audio": False, "dim": 768},
                {"id": "qwen3", "label": "Local · Qwen3 Embedding", "remote": False, "audio": False, "dim": 1024},
                {"id": "qwen2", "label": "Local · Qwen2 Embedding", "remote": False, "audio": False, "dim": 1024},
                {"id": "sentence-transformer", "label": "Local · sentence-transformers", "remote": False, "audio": False, "dim": 768},
                {"id": "clap", "label": "Local · CLAP (audio+text)", "remote": False, "audio": True, "dim": 512},
                {"id": "clip", "label": "Local · CLIP ViT-B/32 (image+text)", "remote": False, "audio": False, "image": True, "dim": 512},
                {"id": "peav", "label": "Local · PE-AV small (audio+video+text)", "remote": False, "audio": True, "video": True, "av": True, "dim": 1024},
            ],
        }

    # ---- static frontend: mounted after every API route, so /spaces, /healthz,
    # … still hit FastAPI and everything else falls through to the bundle
    sd = Path(static_dir) if static_dir is not None else _default_static_dir()
    if sd is not None and sd.is_dir():
        app.mount("/", StaticFiles(directory=str(sd), html=True), name="frontend")
        print(f"[delyrism] serving a static frontend from {sd}")

    if warm:
        _state.update(ready=False, warming=True)
        threading.Thread(target=warmup, daemon=True, name="delyrism-warmup").start()
    return app


app = create_app(
    profile=os.environ.get("DELYRISM_PROFILE") or None,
    warm=os.environ.get("DELYRISM_WARMUP", "").strip().lower() in ("1", "true", "yes", "on"),
)
