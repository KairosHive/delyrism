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
# 3.11: transformers >= 5.22 requires Python >= 3.11
FROM python:3.11-slim

WORKDIR /app

# Python deps — the package's `web` extra (pyproject.toml): API, Topology,
# Builder's PDF mining, the symbol vectorizer (headless OpenCV, for the
# Generate tab's drawings), on CPU torch.  No camera / movement / audio
# models: MediaPipe, CLAP / CLIP / PE-AV, librosa and the TTS / transcription
# stack are for local installs only.  Installed against stub packages first so
# this layer stays cached when only the code changes.
COPY pyproject.toml README.md ./
RUN mkdir -p delyrism symbol_vectorizer  && touch delyrism/__init__.py symbol_vectorizer/__init__.py  && pip install --no-cache-dir --upgrade pip  && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu  && pip install --no-cache-dir ".[web]"  && pip uninstall -y delyrism  && rm -rf delyrism symbol_vectorizer build *.egg-info

# source code, then the package itself (deps already in place)
COPY delyrism ./delyrism
COPY symbol_vectorizer ./symbol_vectorizer
COPY web/backend ./web/backend
RUN pip install --no-cache-dir --no-deps -e .

# static export from stage 1
COPY --from=frontend-builder /build/out ./web/frontend/out

# Railway / Heroku-style $PORT, fall back to 8000 for local runs.
# The hosted site has no TouchDesigner, no TTS service and no GPU next to it:
# no OSC stream, no spoken word, no audio / image / video embedders in the
# model list.  The live graph itself (context x text query) runs as usual.
ENV PORT=8000     DELYRISM_OSC=0     DELYRISM_SPEAK=0     DELYRISM_MEDIA=0
EXPOSE 8000
CMD ["sh", "-c", "python -m delyrism.server --host 0.0.0.0 --port ${PORT} --static /app/web/frontend/out"]
