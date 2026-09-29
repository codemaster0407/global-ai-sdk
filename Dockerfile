# syntax=docker/dockerfile:1
#
# One image for the whole repo. Which server it runs is chosen at runtime:
#
#   docker build -t global-ai-sdk .
#   docker run --env-file .env -p 8000:8000 global-ai-sdk                      # API model server
#   docker run -p 8001:8000 -e APP_MODULE=serving.local_model_server:app \
#              -e WORKERS=1 global-ai-sdk                                      # local model server
#
# Or start both with: docker compose up --build

# ---------------------------------------------------------------------------
# Stage 1: builder - compilers live here only, so they don't ship in the image
# ---------------------------------------------------------------------------
FROM python:3.13.5-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CPU-only PyTorch first: the default Linux wheels bundle ~3 GB of CUDA libraries.
# Remove --index-url here if you deploy on GPU machines.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copied on its own so this layer stays cached until requirements.txt changes
COPY requirements.txt .
RUN pip install -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM python:3.13.5-slim AS runtime

# System libraries the Python packages call at runtime:
#   tesseract-ocr  -> pytesseract (OCR)           poppler-utils -> PDF parsing
#   ffmpeg         -> audio decoding/playback     libmagic1     -> unstructured file-type detection
#   libgomp1       -> LightGBM / XGBoost          libgl1, libglib2.0-0 -> OpenCV
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr poppler-utils ffmpeg libmagic1 libgomp1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_HOME=/opt/models/torch \
    HF_HOME=/opt/models/huggingface

COPY --from=builder /opt/venv /opt/venv

# Bake the local model's weights into the image: containers start without
# downloading anything, and every replica runs exactly the same weights.
RUN python -c "from torchvision.models import resnet18, ResNet18_Weights; resnet18(weights=ResNet18_Weights.DEFAULT)"

# Never run as root
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/data \
    && chown -R app:app /app /opt/models

WORKDIR /app
COPY --chown=app:app . .
USER app

# Server settings - override with -e / docker compose `environment:`
ENV APP_MODULE=serving.api_model_server:app \
    PORT=8000 \
    WORKERS=4 \
    LIMIT_CONCURRENCY=256 \
    TIMEOUT_GRACEFUL_SHUTDOWN=30

EXPOSE 8000

# start-period covers model loading before failed checks count
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=4)"

# `exec` makes uvicorn PID 1, so `docker stop` (SIGTERM) triggers a graceful shutdown:
# in-flight requests finish (up to TIMEOUT_GRACEFUL_SHUTDOWN seconds) before exit.
CMD ["sh", "-c", "exec uvicorn \"$APP_MODULE\" --host 0.0.0.0 --port \"$PORT\" --workers \"$WORKERS\" --limit-concurrency \"$LIMIT_CONCURRENCY\" --timeout-graceful-shutdown \"$TIMEOUT_GRACEFUL_SHUTDOWN\" --proxy-headers"]
