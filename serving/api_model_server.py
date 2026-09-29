"""Serve API-hosted models (Z.ai GLM) with FastAPI, running independent LLM calls in parallel threads.

Why threads work here: an API model call is network I/O. Python releases the GIL
while waiting on the socket, so N calls on N threads take about as long as the
slowest one instead of the sum of all of them.

Concurrency layers:
- Endpoints are plain ``def``: FastAPI runs each request on its own threadpool
  thread, so a request that blocks on its worker threads never blocks the event loop.
- Inside a request, ``run_parallel`` starts one ``threading.Thread`` per LLM call.
- ``_llm_slots`` caps in-flight LLM calls across ALL requests in this process,
  protecting you from provider rate limits (429s). Total capacity is
  MAX_CONCURRENT_LLM_CALLS x uvicorn workers.
- Every request has a deadline. Python threads cannot be killed, so a call that
  outlives the deadline is reported as timed out and finishes in the background;
  the HTTP client timeout (LLM_TIMEOUT_S) bounds how long that can take.

Run from the repo root:
    uvicorn serving.api_model_server:app --host 0.0.0.0 --port 8000 --workers 4
"""
import contextvars
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Callable, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field
from zai import ZaiClient

from serving.common import add_production_middleware, env_float, env_int, require_api_key, setup_logging

load_dotenv()
setup_logging()
log = logging.getLogger("serving.api_models")

LLM_MODEL = os.getenv("LLM_MODEL", "glm-4.5-flash")
LLM_TIMEOUT_S = env_float("LLM_TIMEOUT_S", 60)            # per HTTP call to the provider
REQUEST_TIMEOUT_S = env_float("REQUEST_TIMEOUT_S", 90)    # whole request, including waiting for a slot
MAX_CONCURRENT_LLM_CALLS = env_int("MAX_CONCURRENT_LLM_CALLS", 8)
MAX_BATCH_PROMPTS = env_int("MAX_BATCH_PROMPTS", 20)

_llm_slots = threading.BoundedSemaphore(MAX_CONCURRENT_LLM_CALLS)
_client: Optional[ZaiClient] = None


def call_llm(user_prompt: str, system_prompt: str = "You are a helpful assistant.",
             max_tokens: int = 1024) -> str:
    """One blocking chat completion. Thread-safe: the client uses a thread-safe connection pool."""
    response = _client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt}],
        thinking={"type": "disabled"},   # lower latency for serving; enable for harder tasks
        max_tokens=max_tokens,
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Thread fan-out
# ---------------------------------------------------------------------------
class TaskResult(BaseModel):
    ok: bool
    output: Optional[str] = None
    error: Optional[str] = None
    latency_ms: float = 0.0


def run_parallel(tasks: Dict[str, Callable[[], str]], timeout_s: float = REQUEST_TIMEOUT_S) -> Dict[str, TaskResult]:
    """Run each task on its own thread and wait for all of them, up to ``timeout_s``.

    One failing or slow task never fails the others: each result says ok/error.
    """
    deadline = time.monotonic() + timeout_s
    results: Dict[str, TaskResult] = {}
    results_lock = threading.Lock()

    def worker(name: str, fn: Callable[[], str]) -> None:
        start = time.perf_counter()
        # Wait for a free global slot, but never past the request deadline
        if not _llm_slots.acquire(timeout=max(0.0, deadline - time.monotonic())):
            result = TaskResult(ok=False, error="Server busy: no free LLM slot before the deadline")
        else:
            try:
                result = TaskResult(ok=True, output=fn())
            except Exception as e:
                log.warning("Task %s failed: %s", name, e)
                result = TaskResult(ok=False, error=f"{type(e).__name__}: {e}")
            finally:
                _llm_slots.release()
        result.latency_ms = round((time.perf_counter() - start) * 1000, 1)
        with results_lock:
            results[name] = result

    threads = []
    for name, fn in tasks.items():
        # copy_context() carries the request id into the thread, so its logs are traceable
        ctx = contextvars.copy_context()
        thread = threading.Thread(target=ctx.run, args=(worker, name, fn), name=f"llm-{name}", daemon=True)
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join(timeout=max(0.0, deadline - time.monotonic()))

    with results_lock:
        for name in tasks:
            results.setdefault(name, TaskResult(ok=False, error=f"Timed out after {timeout_s:g}s"))
        return {name: results[name] for name in tasks}


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _client
    _client = ZaiClient(api_key=os.getenv("Z_AI_API_KEY"), timeout=LLM_TIMEOUT_S, max_retries=2)
    log.info("API model server ready: model=%s, max concurrent LLM calls=%d", LLM_MODEL, MAX_CONCURRENT_LLM_CALLS)
    yield
    log.info("Shutting down API model server")


app = FastAPI(title="API model server", version="1.0.0", lifespan=lifespan)
add_production_middleware(app)


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=20_000)
    system_prompt: str = "You are a helpful assistant."


class ChatResponse(BaseModel):
    model: str
    output: str


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)


class BatchChatRequest(BaseModel):
    prompts: List[str] = Field(min_length=1)
    system_prompt: str = "You are a helpful assistant."


class MultiTaskResponse(BaseModel):
    model: str
    results: Dict[str, TaskResult]
    total_latency_ms: float


@app.get("/health")
def health():
    """Liveness: the process is up."""
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """Readiness: the server can take traffic."""
    if not os.getenv("Z_AI_API_KEY"):
        raise HTTPException(status_code=503, detail="Z_AI_API_KEY is not set")
    return {"status": "ready", "model": LLM_MODEL}


@app.post("/v1/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key)])
def chat(req: ChatRequest):
    """Single LLM call. Goes through run_parallel too, so it respects the global slot limit and deadline."""
    result = run_parallel({"chat": lambda: call_llm(req.prompt, req.system_prompt)})["chat"]
    if not result.ok:
        raise HTTPException(status_code=504 if "Timed out" in result.error else 502, detail=result.error)
    return ChatResponse(model=LLM_MODEL, output=result.output)


@app.post("/v1/analyze", response_model=MultiTaskResponse, dependencies=[Depends(require_api_key)])
def analyze(req: AnalyzeRequest):
    """Several DIFFERENT function calls on the same input, run at the same time."""
    text = req.text
    tasks = {
        "summary": lambda: call_llm(f"Summarise in 2 sentences:\n\n{text}", max_tokens=300),
        "sentiment": lambda: call_llm(f"Reply with one word, positive, negative or neutral, for:\n\n{text}",
                                      max_tokens=10),
        "keywords": lambda: call_llm(f"List 5 comma-separated keywords for:\n\n{text}", max_tokens=100),
    }
    start = time.perf_counter()
    results = run_parallel(tasks)
    return MultiTaskResponse(model=LLM_MODEL, results=results,
                             total_latency_ms=round((time.perf_counter() - start) * 1000, 1))


@app.post("/v1/batch-chat", response_model=MultiTaskResponse, dependencies=[Depends(require_api_key)])
def batch_chat(req: BatchChatRequest):
    """The SAME function over many prompts, run at the same time. Results are keyed by prompt index."""
    if len(req.prompts) > MAX_BATCH_PROMPTS:
        raise HTTPException(status_code=413, detail=f"At most {MAX_BATCH_PROMPTS} prompts per request")
    # Bind p in the lambda default, otherwise every task would see the last prompt
    tasks = {str(i): (lambda p=p: call_llm(p, req.system_prompt)) for i, p in enumerate(req.prompts)}
    start = time.perf_counter()
    results = run_parallel(tasks)
    return MultiTaskResponse(model=LLM_MODEL, results=results,
                             total_latency_ms=round((time.perf_counter() - start) * 1000, 1))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serving.api_model_server:app", host="0.0.0.0", port=8000, workers=env_int("WORKERS", 4))
