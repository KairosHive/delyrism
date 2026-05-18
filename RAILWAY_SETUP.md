# Railway Deployment Guide

The new stack runs as **two Railway services** sharing the same GitHub repo:

| Service       | Role                                  | Nixpacks config            |
|---------------|---------------------------------------|----------------------------|
| `backend`     | FastAPI server wrapping the engine    | `nixpacks.toml` (root)     |
| `frontend`    | Next.js 14 UI                         | `frontend.nixpacks.toml`   |
| `egregore`    | (optional) Real-time archetype miner  | `egregore.nixpacks.toml`   |

The previous Streamlit deployment lives on the `old` branch — if you ever
need to roll back, point a service at that branch.

---

## 1. Backend service

1. Create a new Railway project, connect the GitHub repo.
2. Railway picks up `nixpacks.toml` and `Procfile` automatically — no extra
   configuration needed.
3. **Environment variables** (Settings → Variables):

   | Variable                  | Required for                              |
   |---------------------------|-------------------------------------------|
   | `CLOUDFLARE_ACCOUNT_ID`   | Cloudflare embedder + story generator     |
   | `CLOUDFLARE_API_TOKEN`    | Same                                      |
   | `CLOUDFLARE_GATEWAY_ID`   | (optional) Cloudflare AI Gateway proxy    |
   | `CLOUDFLARE_GATEWAY_TOKEN`| (optional) AI Gateway auth                |
   | `CORS_ORIGINS`            | Comma-separated list of allowed origins.  |
   |                           | Defaults to localhost; add your frontend  |
   |                           | URL once deployed.                        |
   | `EGREGORE_URL`            | (optional) URL of the Egregore service    |

4. Click **Settings → Networking → Generate Domain** — note the URL, you'll
   need it as `NEXT_PUBLIC_API_BASE` on the frontend.

---

## 2. Frontend service

1. In the same project click **+ New Service → GitHub repo** (same repo).
2. **Settings → Build**:
   - **Nixpacks Config Path**: `frontend.nixpacks.toml`
3. **Settings → Variables** (these must be set **before** the first build —
   Next.js inlines `NEXT_PUBLIC_*` vars into the bundle at build time):

   | Variable                 | Value                                                       |
   |--------------------------|-------------------------------------------------------------|
   | `NEXT_PUBLIC_API_BASE`   | `https://<backend-domain>.up.railway.app`                   |
   | `NODE_ENV`               | `production` (Railway sets this by default)                 |

4. **Settings → Networking → Generate Domain** — this is the public URL of
   the app.
5. Back on the **backend** service, add this URL to `CORS_ORIGINS` so the
   browser can talk to the API.

---

## 3. (Optional) Egregore — real-time miner

The Δ-graph and the rest of the explorer work without Egregore.  Egregore is
only needed for the *Archetype Builder* tab's real-time PDF/image mining
flow with WebSocket progress.

1. **+ New Service** → same repo.
2. **Settings → Build → Nixpacks Config Path**: `egregore.nixpacks.toml`
3. **Settings → Build → Custom Start Command**:
   ```
   cd delyrism && uvicorn miner_server:app --host 0.0.0.0 --port $PORT
   ```
4. On the **backend** service, set `EGREGORE_URL` to this service's domain.
   The frontend reads it via the `/miner` endpoint.

---

## Shared model cache (recommended)

The backend downloads Hugging Face models on first use (Qwen3 embeddings,
sentence-transformers, …).  Without a volume, Railway redownloads on every
deploy.  Attach a single volume to **both** the backend and Egregore services:

1. **+ New Volume**, mount path `/app/cache`, attach to both services.
2. Variables on each service:

   | Variable          | Value                       |
   |-------------------|-----------------------------|
   | `HF_HOME`         | `/app/cache/huggingface`    |
   | `TORCH_HOME`      | `/app/cache/torch`          |
   | `XDG_CACHE_HOME`  | `/app/cache`                |

If you use only Cloudflare-hosted embedders (no local models) you don't need
the volume at all.

---

## Local development

```bash
# Backend  (port 8000)
pip install -r requirements.txt -r web/backend/requirements.txt
uvicorn app.main:app --reload --port 8000 --app-dir web/backend

# Frontend (port 3000)
cd web/frontend
NEXT_PUBLIC_API_BASE=http://localhost:8000 npm install
NEXT_PUBLIC_API_BASE=http://localhost:8000 npm run dev

# Optional: Egregore (port 8765)
cd delyrism && uvicorn miner_server:app --port 8765
```

Open <http://localhost:3000>.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                       Railway project                         │
│                                                               │
│   ┌────────────┐    ┌────────────┐    ┌──────────────┐       │
│   │  frontend  │───▶│  backend   │◀──▶│   egregore   │       │
│   │ (Next.js)  │ JSON│ (FastAPI) │ HTTP│  (optional)  │       │
│   └─────┬──────┘    └─────┬──────┘    └──────┬───────┘       │
│         │                 │                  │                │
│         ▼                 ▼                  ▼                │
│     public domain    public domain      public domain         │
│                                                               │
│              ┌──────────────────────────────────┐             │
│              │  Shared volume — /app/cache      │             │
│              │  (HF model cache, optional)      │             │
│              └──────────────────────────────────┘             │
└──────────────────────────────────────────────────────────────┘
```

The backend keeps engine state in memory (SymbolSpace per session) and
talks to Cloudflare Workers AI for embeddings + story generation.  The
frontend is stateless — every interaction goes through the backend.
