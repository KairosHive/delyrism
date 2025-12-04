# Railway Deployment Guide

This project uses a **two-service architecture**:
- **Delyrism** (Streamlit) — The main explorer & story generator
- **Egregore** (FastAPI) — Real-time archetype miner with WebSocket progress

Both services can run **standalone** or **together**.

---

## Option 1: Single Service (Delyrism Only)

Deploy only the main Streamlit app. The built-in "Classic Miner" tab will work, but you won't have real-time progress visualization.

### Deploy Steps
1. Create a new Railway project
2. Connect your GitHub repo
3. Railway will auto-detect `nixpacks.toml` and `Procfile`
4. Done!

---

## Option 2: Two Services (Full Experience)

Deploy both Delyrism and Egregore for the complete experience with real-time mining.

### A. Create the Delyrism Service
1. Create a new Railway project
2. Add a service → Connect GitHub repo
3. Name it `delyrism`
4. Railway uses `nixpacks.toml` and `Procfile` (default)

### B. Create the Egregore Service
1. In the same project, click **+ New Service**
2. Connect the **same** GitHub repo
3. Name it `egregore`
4. Go to **Settings** → **Build**:
   - Set **Nixpacks Config Path** to `egregore.nixpacks.toml`
   - Set **Custom Start Command** to:
     ```
     cd delyrism && uvicorn miner_server:app --host 0.0.0.0 --port $PORT
     ```

### C. Link the Services
1. Go to **Delyrism** service → **Variables**
2. Add:
   ```
   EGREGORE_URL = https://${{egregore.RAILWAY_PUBLIC_DOMAIN}}
   ```
   (Railway will auto-resolve this to egregore's public URL)

3. Make sure **Egregore** has a public domain:
   - Go to **Egregore** → **Settings** → **Networking**
   - Click **Generate Domain**

---

## Persistent Volume Setup (Recommended)

To avoid downloading models every time your app restarts:

### 1. Create a Shared Volume
1. Click **+ New Volume** in your Railway project
2. Mount path: `/app/cache`
3. Attach to **both** services (Delyrism and Egregore)

### 2. Configure Environment Variables
Add these to **both** services (or use Railway's shared variables):

| Variable Name | Value | Description |
|--------------|-------|-------------|
| `HF_HOME` | `/app/cache/huggingface` | Hugging Face model cache |
| `TORCH_HOME` | `/app/cache/torch` | PyTorch data cache |
| `XDG_CACHE_HOME` | `/app/cache` | General cache directory |

### 3. AI Service Credentials (Optional)
If using Cloudflare Workers AI:

| Variable Name | Description |
|--------------|-------------|
| `CLOUDFLARE_ACCOUNT_ID` | Your Cloudflare account ID |
| `CLOUDFLARE_API_TOKEN` | API token with Workers AI permission |

---

## Environment Variables Reference

### Delyrism (Streamlit)
| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8501` | Streamlit port (Railway sets this) |
| `EGREGORE_URL` | `http://localhost:8765` | URL to Egregore service |

### Egregore (FastAPI)
| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8765` | FastAPI port (Railway sets this) |

---

## Local Development

Run both services locally:

```bash
# Terminal 1 - Egregore (miner)
cd delyrism
uvicorn miner_server:app --port 8765

# Terminal 2 - Delyrism (explorer)
streamlit run delyrism/app.py
```

Access:
- Delyrism: http://localhost:8501
- Egregore: http://localhost:8765

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    Railway Project                       │
│                                                          │
│  ┌─────────────────┐      ┌─────────────────┐           │
│  │    Delyrism     │      │    Egregore     │           │
│  │   (Streamlit)   │ ←──→ │   (FastAPI)     │           │
│  │                 │      │                 │           │
│  │ • Explorer      │      │ • WebSocket     │           │
│  │ • Story Gen     │      │ • Mining API    │           │
│  │ • Classic Miner │      │ • Real-time UI  │           │
│  └────────┬────────┘      └────────┬────────┘           │
│           │                        │                     │
│           └────────────┬───────────┘                     │
│                        │                                 │
│              ┌─────────▼─────────┐                       │
│              │   Shared Volume   │                       │
│              │   /app/cache      │                       │
│              │   (HF models)     │                       │
│              └───────────────────┘                       │
└─────────────────────────────────────────────────────────┘
```
