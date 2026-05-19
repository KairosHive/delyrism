# Railway Deployment Guide

The app deploys as a **single Railway service via Dockerfile**.  The image
build compiles the Next.js frontend to a static export and the runtime
FastAPI process serves both the API routes (`/spaces`, `/propose`,
`/delta-graph`, …) and the SPA bundle at the same origin.

That means: no CORS, no second service, no env-var dance between two
domains.

The previous Streamlit deployment is preserved on the `old` branch — point a
service at that branch if you need to roll back.

---

## Deploy

1. Connect the GitHub repo as a Railway service (or keep the existing one).
2. Railway sees the root `Dockerfile` and uses it automatically (it
   supersedes any Nixpacks autodetection — that's the whole point).
3. **Settings → Networking → Generate Domain** for the public URL.

No build-command, start-command or root-directory overrides needed.

---

## Environment variables

| Variable                  | Required for                              |
|---------------------------|-------------------------------------------|
| `CLOUDFLARE_ACCOUNT_ID`   | Cloudflare embedder + story generator     |
| `CLOUDFLARE_API_TOKEN`    | Same                                      |
| `CLOUDFLARE_GATEWAY_ID`   | (optional) Cloudflare AI Gateway proxy    |
| `CLOUDFLARE_GATEWAY_TOKEN`| (optional) AI Gateway auth                |
| `EGREGORE_URL`            | (optional) URL of an Egregore builder     |
| `CORS_ORIGINS`            | Unused in single-service mode             |

If you skip the Cloudflare vars the Cloudflare embedder dropdown will
error; users can still pick the local `sentence-transformer` backend in the
sidebar for offline operation.

---

## (Optional) Egregore — real-time builder

Egregore is only needed for the *Archetype Builder* tab's real-time
PDF/image composition flow.  The rest of the explorer works without it.

1. **+ New Service** → same repo.
2. **Settings → Build → Nixpacks Config Path**: `egregore.nixpacks.toml`
3. **Settings → Build → Custom Start Command**:
   ```
   cd delyrism && uvicorn miner_server:app --host 0.0.0.0 --port $PORT
   ```
4. On the main service set `EGREGORE_URL` to this service's domain.

---

## Shared model cache (recommended if you use local embedders)

Local Qwen3 / sentence-transformers / CLAP models download on first use.
Without a volume, Railway redownloads every deploy.  Skip if you use only
Cloudflare-hosted embedders.

1. **+ New Volume**, mount path `/app/cache`, attach to the service.
2. Add these variables to the service:

   | Variable          | Value                       |
   |-------------------|-----------------------------|
   | `HF_HOME`         | `/app/cache/huggingface`    |
   | `TORCH_HOME`      | `/app/cache/torch`          |
   | `XDG_CACHE_HOME`  | `/app/cache`                |

---

## Local development

The Dockerfile is for production.  For local hacking, run the two
processes directly so you get Next.js HMR and uvicorn autoreload:

```bash
# Backend  (port 8000)
pip install -r requirements.txt -r web/backend/requirements.txt
uvicorn app.main:app --reload --port 8000 --app-dir web/backend

# Frontend (port 3000)
cd web/frontend
NEXT_PUBLIC_API_BASE=http://localhost:8000 npm install
NEXT_PUBLIC_API_BASE=http://localhost:8000 npm run dev
```

Open <http://localhost:3000>.

If you want to test the production single-image setup locally:

```bash
docker build -t delyrism .
docker run --rm -p 8000:8000 \
    -e CLOUDFLARE_ACCOUNT_ID=... \
    -e CLOUDFLARE_API_TOKEN=... \
    delyrism
# open http://localhost:8000
```

---

## How it works under the hood

```
  Request                                                    Response
     │                                                          ▲
     ▼                                                          │
  ┌────────────────────────────────────────────────────────────┐
  │  Single uvicorn process (single Railway service)            │
  │                                                             │
  │   FastAPI router (evaluated first):                         │
  │     /spaces        /propose         /attention              │
  │     /reduce-2d     /shift           /delta-graph            │
  │     /subgraph      /similarity      /context/*              │
  │     /story/*       /builder         /backends               │
  │     /healthz       /api                                     │
  │                                                             │
  │   ↓  (everything else)                                      │
  │                                                             │
  │   StaticFiles(directory="web/frontend/out", html=True)      │
  │     → index.html, _next/*, *.js, *.css, …                   │
  └────────────────────────────────────────────────────────────┘
```

FastAPI routes match first; anything that doesn't match falls through to
the static-file handler.  The Next.js bundle is fully client-rendered
(every component is `"use client"`), so static export is a clean fit — no
SSR or Next API routes to worry about.
