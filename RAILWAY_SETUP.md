# Railway Deployment Guide

The app deploys as a **single Railway service**.  The build phase compiles
the Next.js frontend to a static export, and at runtime the FastAPI backend
serves both the API routes (`/spaces`, `/propose`, `/delta-graph`, …) and
the SPA bundle from the same uvicorn process.

That means: no CORS to configure, no second service to manage, no env-var
juggling between two domains.

The previous Streamlit deployment is preserved on the `old` branch — point a
service at that branch if you ever need to roll back.

---

## Deploy

1. Connect the GitHub repo as a Railway service (or keep the existing
   `delyrism` service — it auto-deploys on push to `main`).
2. Railway picks up `nixpacks.toml` and `Procfile` automatically.
3. **Settings → Networking → Generate Domain** for the public URL.

That's it for a baseline deploy.

---

## Environment variables

| Variable                  | Required for                              |
|---------------------------|-------------------------------------------|
| `CLOUDFLARE_ACCOUNT_ID`   | Cloudflare embedder + story generator     |
| `CLOUDFLARE_API_TOKEN`    | Same                                      |
| `CLOUDFLARE_GATEWAY_ID`   | (optional) Cloudflare AI Gateway proxy    |
| `CLOUDFLARE_GATEWAY_TOKEN`| (optional) AI Gateway auth                |
| `EGREGORE_URL`            | (optional) URL of an Egregore miner       |
| `CORS_ORIGINS`            | Unused in single-service mode — set only  |
|                           | if you split frontend & backend across    |
|                           | domains again.                            |

If you skip the Cloudflare vars, the Cloudflare embedder dropdown will
error; users can still pick the local `sentence-transformer` backend in the
sidebar for offline operation.

---

## (Optional) Egregore — real-time miner

Egregore is only needed for the *Archetype Builder* tab's real-time
PDF/image mining flow.  The rest of the explorer works without it.

1. **+ New Service** → same repo.
2. **Settings → Build → Nixpacks Config Path**: `egregore.nixpacks.toml`
3. **Settings → Build → Custom Start Command**:
   ```
   cd delyrism && uvicorn miner_server:app --host 0.0.0.0 --port $PORT
   ```
4. On the main service, set `EGREGORE_URL` to this service's domain.
   The frontend reads it via the `/miner` endpoint.

---

## Shared model cache (recommended if you use local embedders)

Local Qwen3 / sentence-transformers / CLAP models download on first use.
Without a volume Railway redownloads every deploy.  If you use only
Cloudflare-hosted embedders you can skip this entire section.

1. **+ New Volume**, mount path `/app/cache`, attach to the service.
2. Add these variables to the service:

   | Variable          | Value                       |
   |-------------------|-----------------------------|
   | `HF_HOME`         | `/app/cache/huggingface`    |
   | `TORCH_HOME`      | `/app/cache/torch`          |
   | `XDG_CACHE_HOME`  | `/app/cache`                |

---

## Local development

```bash
# 1. Backend  (port 8000)
pip install -r requirements.txt -r web/backend/requirements.txt
uvicorn app.main:app --reload --port 8000 --app-dir web/backend

# 2. Frontend (port 3000)
cd web/frontend
NEXT_PUBLIC_API_BASE=http://localhost:8000 npm install
NEXT_PUBLIC_API_BASE=http://localhost:8000 npm run dev
```

Open <http://localhost:3000>.

Two separate dev processes is the most ergonomic local setup: Next.js HMR
gives instant frontend reloads while the backend autoreloads via uvicorn.
In production the build step compiles the frontend into
`web/frontend/out/` which the backend then serves at `/`.

---

## How it works under the hood

```
  Request                                                    Response
     │                                                          ▲
     ▼                                                          │
  ┌────────────────────────────────────────────────────────────┐
  │  Single uvicorn process (single Railway service)            │
  │                                                             │
  │   FastAPI router (in order):                                │
  │     /spaces        /propose         /attention              │
  │     /reduce-2d     /shift           /delta-graph            │
  │     /subgraph      /similarity      /context/*              │
  │     /story/*       /miner           /backends               │
  │     /healthz       /api                                     │
  │                                                             │
  │   ↓  (everything else)                                      │
  │                                                             │
  │   StaticFiles(directory="web/frontend/out", html=True)      │
  │     → index.html, _next/*, *.js, *.css, …                   │
  └────────────────────────────────────────────────────────────┘
```

FastAPI routes are evaluated first; anything that doesn't match falls
through to the static-file handler.  The Next.js bundle is fully
client-rendered (every component is `"use client"`), so static export is a
clean fit — no SSR or `/api/*` routes to worry about.
