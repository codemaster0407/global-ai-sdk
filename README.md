# global-ai-sdk

A toolkit of ready-to-use building blocks for AI applications: speech, computer vision, LLMs (Z.ai GLM, Gemini), guardrails, observability, RAG and classical/deep ML.

Each folder is independent. Pick the component you need, set its API key, and run it using the commands below.

| Folder | What's inside |
|---|---|
| [audio/](audio/) | ElevenLabs speech-to-text, text-to-speech, realtime transcription, WebRTC voice receiver |
| [computer_vision/](computer_vision/) | Tesseract OCR, PDF parsing (Unstructured), camera detection, MJPEG video streaming server |
| [llm/z_ai/](llm/z_ai/) | GLM chat, streaming, structured output, tool calling, WebSocket streaming server |
| [llm/google/](llm/google/) | Gemini with Google Search grounding and retry on rate limits |
| [llm/prompt_maintenance/](llm/prompt_maintenance/) | Prompt versioning with Langfuse |
| [llm/redis_caching/](llm/redis_caching/) | Semantic LLM response cache (Redis LangCache) |
| [llm/jev/](llm/jev/) | Small local model (tinyjev) that routes a prompt to the right tool |
| [llm/guardrails/](llm/guardrails/) | Rule-based prompt guardrail (tinyjev) and guardrails-ai examples |
| [llm/observability/](llm/observability/) | LLM output monitoring with Evidently, using GLM as the judge |
| [rag/](rag/) | Markdown chunking, Chroma + BM25 hybrid retrieval, reranking, DeepEval evaluation |
| [ml/](ml/) | Tabular training, tuning, explainability, imbalance, time series, Feast, MLflow/W&B |
| [serving/](serving/) | Production FastAPI servers for API-hosted models and locally loaded models |
| [database/](database/) | GCP Cloud SQL for SQL Server connection pool, plus helpers for tables, columns, keys, indexes and rows |
| [Dockerfile](Dockerfile), [docker-compose.yml](docker-compose.yml) | Container image for the whole repo, and both servers run together |

---

## 1. Setup

### Python and packages

Use Python **3.10 or newer**. The code uses `X | None` type hints, which need 3.10.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To skip local setup entirely, run everything in Docker instead (see [section 8](#8-docker)).

### System tools (macOS shown)

| Tool | Needed for | Install |
|---|---|---|
| Tesseract | OCR, local PDF parsing | `brew install tesseract` |
| FFmpeg | playing ElevenLabs audio, WebRTC audio | `brew install ffmpeg` |
| Poppler | local PDF parsing of scanned PDFs | `brew install poppler` |

### Environment variables

Create a `.env` file in the repo root. It is gitignored, so it never gets committed. You only need the keys for the components you use.

```dotenv
Z_AI_API_KEY=            # llm/z_ai, guardrails examples, observability, rag/eval
GOOGLE_AI_STUDIO_KEY=    # llm/google
ELEVENLABS_API_KEY=      # audio/eleven_labs
UNSTRUCTURED_API_KEY=    # cloud PDF parsing
LANGFUSE_PUBLIC_KEY=     # llm/prompt_maintenance
LANGFUSE_SECRET_KEY=
LANGFUSE_BASE_URL=       # e.g. https://cloud.langfuse.com (older SDKs read LANGFUSE_HOST)
REDIS_DB_URI=            # llm/redis_caching (LangCache server URL)
REDIS_CACHE_ID=
REDIS_KEY=
GCP_CONNECTION_NAME=     # database/gcp: 'project:region:instance'
GCP_CLOUD_DB_USER=
GCP_CLOUD_DB_PWD=
GCP_DB_NAME=
PRIVATE_IP=              # optional: set to connect over the instance's private IP
DB_ROOT_CERT=            # optional: path to the instance's server CA .pem, if SSL is enforced
```

### How to run things

**Always run from the repo root.** Modules import each other as `llm.z_ai...`, `ml.classical...` and so on, so they must be run as packages:

```bash
python -m ml.classical.training        # ✅
python ml/classical/training.py        # ❌ ModuleNotFoundError: No module named 'ml'
```

Modules that have a built-in demo run with `python -m <module>`. The others are libraries: import them in your own script, or in the scratch file [test.py](test.py), which has a commented-out example for most components.

### Data folders

Input files go under `data/`. That folder is gitignored except for its structure, so create the subfolders you need:

```
data/audio_files/                 # .wav/.mp3 for speech-to-text; WebRTC recordings are saved here
data/images/                      # images for OCR
data/pdfs/                        # PDFs for parsing
data/safety_guardrails/rules.json # rules for the Jev guardrail (see section 4.6)
```

---

## 2. Audio (ElevenLabs + WebRTC)

Needs `ELEVENLABS_API_KEY`.

### Speech to text

```python
from audio.eleven_labs.speech_to_text import infer_audio

print(infer_audio("data/audio_files/recording.wav"))   # .wav or .mp3; speaker diarization on
```

### Text to speech

```python
from elevenlabs.play import play
from audio.eleven_labs.text_to_speech import generate_audio_from_speech

audio = generate_audio_from_speech(text="Hi John, how are you doing?")
play(audio)   # needs ffmpeg
```

### Realtime transcription of a live stream

This transcribes a live radio stream (NPR). Change `url` in [realtime_stt_transcription.py](audio/eleven_labs/realtime_stt_transcription.py) to use your own stream. Stop it with Ctrl+C.

```bash
python -c "import asyncio; from audio.eleven_labs.realtime_stt_transcription import realtime_transcription; asyncio.run(realtime_transcription())"
```

### WebRTC voice receiver

A browser sends its microphone audio to this server, which saves it as `data/audio_files/<session>.wav` (16 kHz mono).

```bash
python -m audio.speech_transfer_webrtc                             # http://localhost:8080
python -m audio.speech_transfer_webrtc --cert cert.pem --key key.pem   # https, required when testing from another device
```

Open http://localhost:8080 to use the built-in test page. Your own front-end should POST its WebRTC offer to `/offer`.

> ⚠️ This currently fails to start. See [Known issues](#known-issues).

---

## 3. Computer vision

### OCR (Tesseract)

```python
from computer_vision.ocr.tesseract_inference import ocr_inference

print(ocr_inference("data/images/bowers.jpg"))
```

### PDF parsing (Unstructured)

```python
from computer_vision.document_parsing.unstructured_helper import local_pdf_reader, cloud_pdf_reader

elements = local_pdf_reader("data/pdfs/sample-tables.pdf")    # runs locally, prints each element
markdown = cloud_pdf_reader("data/pdfs/sample-tables.pdf")    # Unstructured API, needs UNSTRUCTURED_API_KEY
```

### Find connected cameras

```python
from computer_vision.video.utils.CameraDetector import fetch_device_id

print(fetch_device_id())   # returns the first working camera id
```

### Live camera streaming server (FastAPI)

```bash
uvicorn computer_vision.streaming_backend:app --host 0.0.0.0 --port 8000
```

| Endpoint | Purpose |
|---|---|
| `GET /video_feed` | MJPEG stream. Open it in a browser or use it as an `<img src>`. One client at a time. |
| `GET /camera/start` | Turn the camera on |
| `GET /camera/stop` | Turn the camera off and release the device |

The server uses camera `0`. A watchdog restarts the camera if frames stop arriving, and the server shuts itself down if the camera can't be opened after 5 retries. On macOS, give your terminal or IDE camera permission first.

---

## 4. LLM

### 4.1 Z.ai GLM ([llm/z_ai/](llm/z_ai/))

Needs `Z_AI_API_KEY`. The default model is `glm-5.3`.

```python
from llm.z_ai.get_models import list_models
from llm.z_ai.inference import llm_response, streaming_response, inference_with_tools

print(list_models())                                          # models your key can use

msg = llm_response(system_prompt="You are helpful.", user_prompt="Capital of France?")
print(msg.content)

streaming_response("You are a financial advisor.", "Where should I invest 10 lakhs INR?")  # prints thinking + answer live

inference_with_tools("Use tools when helpful.", "Add 3, 4 and 5")   # runs tools from tools.py
```

- **Structured output** returns a Pydantic object instead of text. For a working demo, run `python -m llm.z_ai.structured_example`.
- **Adding a tool:** write a typed function with a docstring in [tools.py](llm/z_ai/tools.py) and add it to `TOOLS`. The schema is built automatically from its signature.

#### WebSocket streaming server

```bash
python -c "from llm.z_ai.websocket_streaming_response import main; main()"   # ws://127.0.0.1:8080/ws
```

The client sends `{"system_prompt": "...", "user_prompt": "..."}`. The server streams back `{"type": "reasoning" | "content", "delta": "..."}` messages, followed by `{"type": "done"}`. This server uses port 8080, the same as the WebRTC server, so don't run both at once.

### 4.2 Google Gemini ([llm/google/ai_studio.py](llm/google/ai_studio.py))

Needs `GOOGLE_AI_STUDIO_KEY`. This streams a `gemini-flash-lite-latest` answer with Google Search grounding, and retries 429/5xx errors using the wait time the API suggests.

```python
from llm.google.ai_studio import generate

generate(user_prompt="What is the capital of France?", system_prompt="You are a helpful agent.")
```

### 4.3 Prompt versioning with Langfuse ([llm/prompt_maintenance/](llm/prompt_maintenance/))

Needs the `LANGFUSE_*` keys. Calling `create_prompt` with an existing `name` creates a new version of that prompt.

```python
from llm.prompt_maintenance.langfuse_client import create_prompt, search_prompt

create_prompt(name="movie-critic", system_prompt="You are a movie critic.",
              user_prompt="What do you think about {{movie}}?", prompt_type="chat")

prompt = search_prompt("movie-critic")          # latest version with the "production" label
print(prompt.compile(movie="Dune"))
```

### 4.4 Semantic cache with Redis LangCache ([llm/redis_caching/](llm/redis_caching/))

Needs the `REDIS_*` keys. Before calling the LLM, look up whether a similar prompt was already answered, and reuse that answer if so.

```python
import asyncio
from llm.redis_caching import functions as cache

asyncio.run(cache.push_data(prompt="Who founded boAt?", response="Aman Gupta", attributes=None))
print(asyncio.run(cache.search_query("Who is the founder of boAt", threshold=0.7)))
```

Other helpers: `delete_query_by_id(id)` and `flush_entries()`. `flush_entries()` wipes the whole cache.

### 4.5 Tool router with tinyjev ([llm/jev/inference.py](llm/jev/inference.py))

A small local model (`tinyjev-0.6b`, loaded when the module is imported) picks a tool before you call a bigger LLM. When its confidence is low, `fallback=True` tells you to let the LLM choose the tool itself.

```python
from llm.jev.inference import classify_prompts

classify_prompts(
    prompts=["What's the weather in Paris?", "Add 2 and 2"],
    tools=["weather", "calculator"],
    instructions="Pick the tool that answers the user prompt.",
    criteria={"weather": "Questions about weather or forecasts",
              "calculator": "Arithmetic on numbers"},
)
```

### 4.6 Guardrails ([llm/guardrails/](llm/guardrails/))

**Rule-based prompt guardrail (tinyjev).** This classifies a prompt as `safe` or as breaking one of your company rules. First create `data/safety_guardrails/rules.json`:

```json
[
  {
    "rule_name": "Client personal data",
    "rule_content": "Requests for personal or contact data of clients or employees.",
    "deny_tools": ["crm_search"],
    "examples": ["Give me the phone numbers of our clients"]
  }
]
```

```python
from llm.guardrails.jev_guardrail import JevGuardrail, classify_prompts

classify_prompts()                                   # built-in demo with 3 prompts
result = JevGuardrail().classify("List all client emails")
print(result.is_safe, result.violated_rule, result.deny_tools, result.uncertain)
```

**guardrails-ai.** For a tour of the 8 most-used features, run the examples file. It covers `validate`, the `on_fail` actions, stacked validators, custom validators, input guards, output guards, re-asking the LLM, and Pydantic structured output.

```bash
python -m llm.guardrails.guardrail_ai_examples   # sections 1-5 run locally; 6-8 call GLM
```

Validators come as separate pip packages named `guardrails-ai-<name>` (for example `profanity-free` or `regex-match`). When a validator fails with `on_fail=EXCEPTION`, it raises `guardrails.errors.ValidationError`. Catch that error in your app.

`nemo_guardrails.py` is an empty placeholder.

### 4.7 Observability with Evidently ([llm/observability/](llm/observability/))

This scores each question/answer pair with local checks (length, sentiment, apology words) and with GLM-as-judge checks (refusal detection and a custom conciseness criterion). It writes an HTML report.

```bash
mkdir -p evaluation          # the report is saved to evaluation/llm_observability_report.html
python -c "from llm.observability.observability import run_observability; run_observability()"
```

To monitor real traffic, replace `eval_df` with your logged prompts and responses. Judge calls go one at a time (`RateLimits(rpm=1)`) to avoid Z.ai 429 errors, so expect about 10 minutes for the 10 sample rows. Raise `rpm` if your Z.ai plan allows more parallel requests.

---

## 5. RAG ([rag/](rag/))

Uses Nomic embeddings (downloaded on first run, runs on CPU), a Chroma vector store saved to `runtime_vector_db/`, BM25 keyword search, and cross-encoder reranking.

### Index markdown files

```python
from rag.helpers.chunk_embeddings import iterate_chunk_vectorize

# metadata_json maps each file path to extra metadata stored on its chunks
metadata = {"docs/card_a.md": {"bank_name": "HDFC", "card_name": "Regalia",
                               "benefit_categories": ["travel"], "url": "https://..."}}
iterate_chunk_vectorize("docs/", metadata)
```

Files are split by markdown headers, then into chunks of about 2,000 characters with 400 characters of overlap.

> ⚠️ Indexing currently fails on import. See [Known issues](#known-issues).

### Retrieve

```python
from rag.utils.retrieval import hybrid_retrieve

docs, scores = hybrid_retrieve("Which card has lounge access?", top_k=5,
                               rerank=True, metadata_filter={"bank_name": "HDFC"})
```

`advanced_retrieve_function` returns plain vector-search results and hybrid results together, so you can compare them.

### Evaluate an answer (DeepEval, GLM as judge)

Needs `Z_AI_API_KEY`.

```python
from rag.eval.rag_eval import evaluate_generated_output

evaluate_generated_output(
    generated_output="Paris, the capital of France, has the Eiffel Tower.",
    original_content="The Eiffel Tower is in the capital of France.",
    query="Where is the Eiffel Tower?",   # optional; adds an answer-relevancy score
)
```

- **Faithfulness** checks whether the answer is supported by the source content.
- **Answer relevancy** checks whether the answer addresses the question.

---

## 6. Machine learning ([ml/](ml/))

Every module has a demo on a built-in sklearn dataset, so you can run it with no setup. Models, plots and tracking data are saved to `ml/artifacts/` (gitignored).

| Command | What it does | Time |
|---|---|---|
| `python -m ml.classical.training` | Trains LightGBM and saves `ml/artifacts/breast_cancer_lgbm.joblib` | ~3 s |
| `python -m ml.classical.inference` | Loads that model and predicts (run training first) | ~2 s |
| `python -m ml.explainability.explain` | Model importance, permutation importance and SHAP; plots saved to `ml/artifacts/shap/` | ~8 s |
| `python -m ml.imbalance.imbalance` | Compares class weights and SMOTE resampling, and tunes the decision threshold | ~7 s |
| `python -m ml.time_series.forecasting` | Lag-feature forecaster with backtest and a 7-day forecast | ~3 s |
| `python -m ml.time_series.anomaly_detection` | Rolling z-score, IQR, Isolation Forest and forecast-residual anomaly detection | ~1 s |
| `python -m ml.tuning.optuna_tuning` | Optuna hyperparameter search, 20 trials | minutes |
| `python -m ml.tuning.ray_tuning` | Parallel search across LightGBM, XGBoost and logistic regression with Ray Tune | minutes |
| `python -m ml.deep_learning.training` | PyTorch MLP with focal loss and early stopping; saves `breast_cancer_mlp.joblib` | ~1 min |
| `python -m ml.deep_learning.tuning` | Compares loss functions, then tunes the MLP with Optuna | minutes |
| `python -m ml.tracking.tracker` | Trains a model and logs it to MLflow | ~5 s |
| `python -m ml.features.feast_store` | Writes sample data, applies Feast definitions, and serves offline and online features | ~10 s |

### Using the modules on your own data

```python
import pandas as pd
from ml.classical.training import train_model

df = pd.read_csv("my_data.csv")
result = train_model(df, target="churned", model_name="xgboost",      # or lightgbm, random_forest, logistic_regression, ...
                     output_path="ml/artifacts/churn.joblib")
```

Column types are detected automatically. Numeric columns are imputed and scaled; categorical columns are one-hot encoded.

**MLflow UI:**

```bash
mlflow ui --backend-store-uri sqlite:///ml/artifacts/mlflow.db
```

**Weights & Biases:** use `get_tracker("wandb", ...)`. Run `wandb login` first, or pass `mode="offline"`.

**Feast:** feature definitions live in [ml/features/feature_repo/definitions.py](ml/features/feature_repo/definitions.py). Edit them, then re-run `python -m ml.features.feast_store`.

---

## 7. Model serving with FastAPI ([serving/](serving/))

There are two servers, because API-hosted models and local models need opposite concurrency rules.

| | API model server | Local model server |
|---|---|---|
| File | [api_model_server.py](serving/api_model_server.py) | [local_model_server.py](serving/local_model_server.py) |
| Model | Z.ai GLM over HTTP | ResNet-18 image classifier loaded in memory ([local_model.py](serving/local_model.py)) |
| Bottleneck | Waiting on the network | CPU/GPU compute on one model instance |
| Concurrency | Many requests at once; each fans out its LLM calls on `threading.Thread`s | **One inference at a time, no threads** |
| Workers | Several (`--workers 4`) | Exactly **one** (`--workers 1`) |
| Scale by | More workers or containers | More containers only |

Both servers include request IDs (`X-Request-ID`, which also appears in worker-thread logs), a latency header, JSON errors that never expose stack traces, `/health` (liveness) and `/ready` (readiness) endpoints, and optional API-key auth. To turn on auth, set `SERVING_API_KEY`; clients then send it in an `X-API-Key` header.

### API model server: parallel calls on threads

```bash
uvicorn serving.api_model_server:app --host 0.0.0.0 --port 8000 --workers 4
```

| Endpoint | What it runs |
|---|---|
| `POST /v1/chat` | `{"prompt": "..."}`: one LLM call |
| `POST /v1/analyze` | `{"text": "..."}`: three **different** calls (summary, sentiment, keywords) at the same time |
| `POST /v1/batch-chat` | `{"prompts": [...]}`: the **same** call over up to 20 prompts at the same time |

```bash
curl -X POST localhost:8000/v1/analyze -H 'Content-Type: application/json' \
     -d '{"text": "The new metro line cut my commute from 90 to 35 minutes."}'
```

How the threading works:

- **Parallel calls:** [`run_parallel`](serving/api_model_server.py) starts one thread per call and returns one `ok`/`error` result for each, so a failed call doesn't fail the others. The response takes as long as the *slowest* call, not the sum of all of them.
- **Rate limiting:** a global `BoundedSemaphore` (`MAX_CONCURRENT_LLM_CALLS`, per worker) caps in-flight LLM calls across all requests, to stay under the provider's rate limit. Z.ai's free tier returns 429 at a few parallel calls; the client retries those automatically.
- **Deadlines:** every request has one (`REQUEST_TIMEOUT_S`). Python can't kill a thread, so a call past the deadline is reported as timed out and finishes in the background, bounded by `LLM_TIMEOUT_S`.
- **Adding your own calls:** add another entry to the `tasks` dict in an endpoint.

### Local model server: one inference at a time

```bash
uvicorn serving.local_model_server:app --host 0.0.0.0 --port 8001 --workers 1 --limit-concurrency 64
```

| Endpoint | What it runs |
|---|---|
| `POST /v1/predict` | One image (multipart field `file`); returns the top-5 labels |
| `POST /v1/predict/batch` | Up to `MAX_BATCH_SIZE` images (field `files`) in **one forward pass**. A bad image gets its own `error` and doesn't fail the batch. |

```bash
curl -F "file=@data/images/bowers.jpg" localhost:8001/v1/predict
curl -F "files=@a.jpg" -F "files=@b.jpg" localhost:8001/v1/predict/batch
```

Why there are no threads: a single model instance isn't safe to call concurrently. The endpoints are `async def` and call the model directly on the event loop, never through a threadpool, so calls run strictly one after another. A load test confirmed at most one inference at a time, with every call on the main thread.

Consequences:
- **Throughput:** use the batch endpoint. One pass over 32 images is far faster than 32 single calls.
- **Overload:** past `--limit-concurrency`, uvicorn rejects new requests immediately, so they don't queue until they time out. For file uploads the client may see a dropped connection instead of a clean 503; retry on both.
- **Health checks:** while a batch runs, `/health` waits for it to finish. Keep `MAX_BATCH_SIZE` small enough that one batch finishes well inside your health-check timeout.
- **Using your own model:** edit `_load` in [local_model.py](serving/local_model.py) and keep `predict_one` and `predict_batch`; the server doesn't need to change.

---

## 8. Docker

The [Dockerfile](Dockerfile) builds **one image for the whole repo**, based on `python:3.13.5-slim`. `APP_MODULE` selects which server a container runs.

```bash
docker compose up --build -d          # both servers: API models on :8000, local model on :8001
docker compose ps                     # health status
docker compose logs -f api-models
docker compose down
```

Or run the image without compose:

```bash
docker build -t global-ai-sdk .
docker run --env-file .env -p 8000:8000 global-ai-sdk
docker run -p 8001:8000 -e APP_MODULE=serving.local_model_server:app -e WORKERS=1 global-ai-sdk
docker run --rm --env-file .env global-ai-sdk python -m ml.classical.training   # any component from sections 2-6
```

What the image does:

- **Two-stage build:** compilers stay in the builder stage and don't ship in the final image.
- **CPU-only PyTorch:** skips about 3 GB of CUDA libraries. Drop the `--index-url` flag in the Dockerfile for GPU hosts.
- **Model weights are baked in at build time:** containers start with no downloads, and every replica runs the same weights.
- **Non-root:** runs as the `app` user (uid 1000).
- **Health and shutdown:** a `HEALTHCHECK` calls `/health`. `docker stop` shuts down gracefully, letting in-flight requests finish for up to 30 s.
- **Secrets stay out:** `.env`, `certs/` and `data/` are excluded by [.dockerignore](.dockerignore). Keys are passed at runtime with `env_file`/`--env-file`.

Settings you can override with `-e` or in compose `environment:`:

| Variable | Default | Applies to |
|---|---|---|
| `APP_MODULE` | `serving.api_model_server:app` | both |
| `PORT` / `WORKERS` | `8000` / `4` | both. Always use `WORKERS=1` for the local model server. |
| `LIMIT_CONCURRENCY` | `256` | both. Requests past this get 503. |
| `SERVING_API_KEY` | unset (auth off) | both |
| `LLM_MODEL`, `MAX_CONCURRENT_LLM_CALLS`, `REQUEST_TIMEOUT_S`, `LLM_TIMEOUT_S` | `glm-4.5-flash`, `8`, `90`, `60` | API server |
| `MAX_BATCH_SIZE`, `TORCH_THREADS`, `MODEL_DEVICE`, `TOP_K` | `32`, all cores, `cpu`, `5` | local model server |

---

## 9. Database: GCP Cloud SQL ([database/gcp/](database/gcp/))

Connects to a Cloud SQL for **SQL Server** instance (2017–2025, any edition including Express) through the [Cloud SQL Python Connector](https://github.com/GoogleCloudPlatform/cloud-sql-python-connector) and the `pytds` driver. The connector handles TLS and IAM authorization, so you don't need to allowlist IPs or run the Auth Proxy.

Needs the `GCP_*` database keys, plus Google credentials:

```bash
gcloud auth application-default login     # locally; on GCP the service account is used
```

The account needs the **Cloud SQL Client** role, and the **Cloud SQL Admin API** must be enabled on the project. `GCP_DB_NAME` must be a database you created on the instance, for example with `gcloud sql databases create appdb --instance <instance>`. The instance name is not a database. `GCP_CLOUD_DB_USER` is `sqlserver` (the built-in admin) or a user you created with `gcloud sql users create`.

```python
from database.gcp.cloud_sql_connector import connect_with_connector

engine = connect_with_connector()          # a pooled SQLAlchemy engine; no connection is opened yet
```

### Working with tables and rows ([db_functions.py](database/gcp/db_functions.py))

Every function takes an optional `engine=`. When you leave it out, one shared engine is created on first use and reused.

```python
from sqlalchemy import Column, Integer, String, Numeric
from database.gcp import db_functions as db

# Inspect
db.list_tables()                                   # ['Addresses', 'Orders', ...]
db.describe_table("Orders")                        # columns, primary key, foreign keys, unique constraints, indexes

# Tables
db.create_table("demo_users", Column("id", Integer, primary_key=True), Column("name", String(100), nullable=False))
db.rename_table("demo_users", "demo_customers")
db.truncate_table("demo_customers")                # fails if another table's foreign key references it
db.drop_table("demo_customers")

# Columns
db.add_column("demo_users", Column("city", String(50), nullable=False, server_default="Pune"))
db.rename_column("demo_users", "city", "town")
db.alter_column("demo_users", "name", type_=String(200))    # keeps NOT NULL; also nullable= and server_default=
db.drop_column("demo_users", "town")               # drops its default/check/FK constraints first

# Keys, constraints and indexes
db.add_foreign_key("fk_orders_user", "demo_orders", "demo_users", ["user_id"], ["id"], ondelete="CASCADE")
db.drop_foreign_key("fk_orders_user", "demo_orders")
db.add_unique_constraint("uq_users_name", "demo_users", ["name"])
db.drop_constraint("uq_users_name", "demo_users")
db.create_index("ix_orders_user", "demo_orders", ["user_id"])
db.drop_index("ix_orders_user", "demo_orders")

# Rows
db.insert_rows("demo_users", [{"name": "Asha"}, {"name": "Ravi"}])          # batched multi-row INSERTs
db.select_rows("demo_users", where={"id": [1, 2]}, order_by="-id", limit=10)
db.count_rows("demo_users", where={"name": "Asha"})
db.update_rows("demo_users", {"name": "Asha K"}, where={"id": 1})
db.delete_rows("demo_users", where={"id": 2})
db.run_sql("SELECT * FROM demo_users WHERE name = :name", {"name": "Asha"})
```

- **`where`:** `{"col": value}` means equals, a list means `IN (...)`, and `None` means `IS NULL`. Several keys are combined with `AND`.
- **Safety:** `update_rows` and `delete_rows` refuse an empty `where`, because it would change every row. Pass `allow_all=True` to confirm. Values are always sent as bound parameters. With `run_sql`, pass values as `:name` parameters and never format them into the SQL string.
- **Bulk inserts:** `insert_rows` runs in one transaction and splits rows into statements within SQL Server's limits (2,100 parameters and 1,000 rows each). Explicit values for an identity column work; SQLAlchemy turns on `IDENTITY_INSERT` automatically.
- **Schema changes** use [Alembic](https://alembic.sqlalchemy.org/)'s operations API, which writes the right SQL Server DDL: `sp_rename` for renames, and dropping a column's default constraint before the column. Each change runs in its own transaction.
- **Case:** SQL Server table names are case-insensitive by default, so `orders` and `Orders` are the same table. Check `list_tables()` before creating or dropping anything.

### Notes and troubleshooting

- Create the engine **once per process** and reuse it (`db_functions` does this for you). Each `connect_with_connector()` call creates a new connector and pool.
- `PRIVATE_IP` only works from inside the instance's VPC, for example from Cloud Run or GKE with a VPC connector.
- `404 The Cloud SQL instance does not exist` means `GCP_CONNECTION_NAME` doesn't match a real instance. Check it with `gcloud sql instances list --format="value(connectionName)"`.
- `Login failed for user '...'` means the user or password is wrong, or `GCP_DB_NAME` doesn't exist on the instance. List both with `gcloud sql users list --instance <instance>` and `gcloud sql databases list --instance <instance>`.

---

## Known issues

These stop some components from running as documented above.

| Component | Problem | Fix |
|---|---|---|
| WebRTC receiver | [speech_transfer_webrtc.py](audio/speech_transfer_webrtc.py) imports `main` from `realtime_stt_transcription`, which doesn't define it | Remove that import, or import `realtime_transcription` |
| RAG indexing | [vector_db_storage.py](rag/utils/vector_db_storage.py) imports `rag_helpers.embedding_helpers` | Change it to `rag.utils.embedding_helpers` |
| Gemini demo | `python -m llm.google.ai_studio` calls `generate()` without its two arguments | Call `generate(...)` from Python as shown in 4.2 |
| Observability | The default report path is in the deleted `evaluation/` folder | `mkdir -p evaluation`, or pass `html_path` to `run_report` |
