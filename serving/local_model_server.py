"""Serve a locally loaded model with FastAPI: single-image and batch endpoints, one call at a time.

Concurrency model - ONE model instance, ONE inference at a time, NO threads:
- Endpoints are ``async def`` and call the model directly on the event loop,
  inside ``_model_lock``. FastAPI only moves plain ``def`` endpoints to a
  threadpool, so the model is never touched from two threads.
- Run exactly ONE uvicorn worker per container (each worker would load its own
  copy of the model). Scale by running more containers behind a load balancer.
- Backpressure: start uvicorn with ``--limit-concurrency N``. Past N in-flight
  requests uvicorn answers 503 right away, so clients can retry elsewhere
  instead of queueing until they time out.
- Trade-off: while the model runs, the event loop is busy, so other requests
  (including /health) wait for it. MAX_BATCH_SIZE bounds how long that can be -
  keep one batch well under your health-check timeout.
- For throughput, send images to /v1/predict/batch: one forward pass over N
  images is much faster than N single calls.

Run from the repo root:
    uvicorn serving.local_model_server:app --host 0.0.0.0 --port 8001 --workers 1 --limit-concurrency 64
"""
import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import torch
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from serving.common import add_production_middleware, env_int, require_api_key, setup_logging
from serving.local_model import ImageClassifier

setup_logging()
log = logging.getLogger("serving.local_model_server")

MAX_BATCH_SIZE = env_int("MAX_BATCH_SIZE", 32)
MAX_IMAGE_BYTES = env_int("MAX_IMAGE_BYTES", 10 * 1024 * 1024)
DEVICE = os.getenv("MODEL_DEVICE", "cpu")
TOP_K = env_int("TOP_K", 5)
# Threads PyTorch uses *inside* one forward pass. This is not request concurrency:
# there is still only one inference at a time.
TORCH_THREADS = env_int("TORCH_THREADS", 0)

# The model call has no await inside, so the event loop already runs one call at a
# time. The lock keeps that true if someone later adds an await (or to_thread) here.
_model_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if TORCH_THREADS:
        torch.set_num_threads(TORCH_THREADS)
    start = time.perf_counter()
    app.state.model = ImageClassifier(device=DEVICE, top_k=TOP_K)
    app.state.model.warmup()
    log.info("Model ready in %.1fs (max batch %d)", time.perf_counter() - start, MAX_BATCH_SIZE)
    yield
    app.state.model = None
    log.info("Model unloaded")


app = FastAPI(title="Local model server", version="1.0.0", lifespan=lifespan)
add_production_middleware(app)


class Prediction(BaseModel):
    filename: Optional[str] = None
    predictions: Optional[List[Dict]] = None
    error: Optional[str] = None


class BatchResponse(BaseModel):
    results: List[Prediction]
    batch_size: int
    inference_ms: float


async def _read_image(upload: UploadFile) -> bytes:
    data = await upload.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(f"Image larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB")
    return data


async def _run_exclusive(fn, *args):
    """Run a model call with the model to ourselves."""
    async with _model_lock:
        # Called inline - deliberately NOT asyncio.to_thread / run_in_threadpool
        return fn(*args)


@app.get("/health")
async def health():
    """Liveness: the process is up."""
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    """Readiness: the model is loaded and warmed up."""
    if getattr(app.state, "model", None) is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready", "device": DEVICE}


@app.post("/v1/predict", response_model=Prediction, dependencies=[Depends(require_api_key)])
async def predict(file: UploadFile = File(...)):
    """Classify one image."""
    try:
        image = ImageClassifier.decode(await _read_image(file))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    predictions = await _run_exclusive(app.state.model.predict_one, image)
    return Prediction(filename=file.filename, predictions=predictions)


@app.post("/v1/predict/batch", response_model=BatchResponse, dependencies=[Depends(require_api_key)])
async def predict_batch(files: List[UploadFile] = File(...)):
    """Classify up to MAX_BATCH_SIZE images in one forward pass.

    A bad image doesn't fail the batch: it gets an ``error`` and the rest are still classified.
    Results come back in the same order as the uploaded files.
    """
    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=413, detail=f"At most {MAX_BATCH_SIZE} images per batch")

    results = [Prediction(filename=f.filename) for f in files]
    images, positions = [], []
    for i, upload in enumerate(files):
        try:
            images.append(ImageClassifier.decode(await _read_image(upload)))
            positions.append(i)
        except ValueError as e:
            results[i].error = str(e)

    start = time.perf_counter()
    predictions = await _run_exclusive(app.state.model.predict_batch, images)
    inference_ms = round((time.perf_counter() - start) * 1000, 1)

    for i, preds in zip(positions, predictions):
        results[i].predictions = preds
    return BatchResponse(results=results, batch_size=len(images), inference_ms=inference_ms)


if __name__ == "__main__":
    import uvicorn
    # workers=1 on purpose: one model instance per process, one inference at a time
    uvicorn.run("serving.local_model_server:app", host="0.0.0.0", port=8001, workers=1,
                limit_concurrency=env_int("LIMIT_CONCURRENCY", 64))
