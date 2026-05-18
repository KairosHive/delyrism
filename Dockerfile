# Delyrism — single-image deploy.
#
# Two stages:
#   1) frontend-builder — compiles the Next.js app into a static export
#   2) runtime          — Python + FastAPI + the static export mounted at /
#
# Railway / Fly / any Docker-compatible host will pick this up automatically
# when a Dockerfile sits at the repo root, overriding Nixpacks.

# ---------- 1. frontend build ------------------------------------------------
FROM node:20-slim AS frontend-builder

WORKDIR /build

# install deps with a deterministic lockfile-driven install
COPY web/frontend/package.json web/frontend/package-lock.json ./
RUN npm ci

# build the static export → /build/out
COPY web/frontend/ ./
RUN npm run build


# ---------- 2. runtime image -------------------------------------------------
FROM python:3.10-slim

# system libs the engine needs (audio, video, build tools for any source
# distributions pip pulls in)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg git curl build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# python deps — CPU-only torch wheels keep the image manageable
COPY requirements.txt ./
COPY web/backend/requirements.txt ./web/backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir -r web/backend/requirements.txt

# source code
COPY web/backend ./web/backend
COPY delyrism ./delyrism

# static export from stage 1
COPY --from=frontend-builder /build/out ./web/frontend/out

# Railway / Heroku-style $PORT, fall back to 8000 for local runs
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --app-dir web/backend"]
