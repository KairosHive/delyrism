"""Delyrism FastAPI backend.

Run with:
    uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations
import os
import time
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .routes import spaces, analysis, delta, context, story, builder, topology

app = FastAPI(
    title="Delyrism API",
    version="0.1.0",
    description="HTTP surface for the SymbolSpace engine — used by the new Next.js frontend.",
)

# CORS — always allow common local dev origins; let an env var widen the list
# in prod.  `allow_origin_regex=".*"` acts as a safety net so this never
# silently misconfigures when CORS_ORIGINS is set to something narrower than
# what the browser is actually using.
_default_dev_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
env_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()]
allow_origins = list({*_default_dev_origins, *env_origins})

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_origin_regex=".*",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*", "X-Server-Ms", "Server-Timing"],
    max_age=600,
)

print(f"[delyrism] CORS allow_origins={allow_origins} (+ regex=.*)")


@app.middleware("http")
async def server_timing(request: Request, call_next):
    """Tag every response with a `Server-Timing` header so the browser Network
    tab shows wall-clock time per route — handy for spotting which endpoint
    is the bottleneck when sliders feel sluggish."""
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000.0
    # `app;dur=…` is the standard Server-Timing format
    existing = response.headers.get("Server-Timing", "")
    tag = f"app;dur={ms:.1f}"
    response.headers["Server-Timing"] = f"{existing}, {tag}" if existing else tag
    response.headers["X-Server-Ms"] = f"{ms:.1f}"
    return response

app.include_router(spaces.router)
app.include_router(analysis.router)
app.include_router(delta.router)
app.include_router(context.router)
app.include_router(story.router)
app.include_router(builder.router)
app.include_router(builder.legacy_router)  # /miner alias kept for older clients
app.include_router(topology.router)


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/api")
def api_info():
    """Cheap probe — returns OK so /api responds with JSON even when the
    static export's index.html is mounted at /."""
    return {"ok": True, "name": "delyrism-api"}


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
            {"id": "sentence-transformer", "label": "Local · sentence-transformers", "remote": False, "audio": False, "dim": 384},
            {"id": "clap", "label": "Local · CLAP (audio+text)", "remote": False, "audio": True, "dim": 512},
        ],
    }


# ---- Static frontend ---------------------------------------------------------
# In production the Next.js static export (`output: 'export'`) lands at
# web/frontend/out/.  Mounting it AFTER every API route is registered means
# /spaces, /propose, /healthz, etc. still hit FastAPI; everything else falls
# through to the SPA bundle.  In local dev the directory won't exist — we
# silently skip the mount and the user runs `npm run dev` separately.
_STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "frontend" / "out"
if _STATIC_DIR.is_dir():
    # Mount under a real path first so we serve _next/* assets correctly.
    app.mount(
        "/",
        StaticFiles(directory=str(_STATIC_DIR), html=True),
        name="frontend",
    )
    print(f"[delyrism] serving Next.js static export from {_STATIC_DIR}")
else:
    print(
        f"[delyrism] no static frontend at {_STATIC_DIR} "
        "(dev mode — run `npm run dev` separately)"
    )
