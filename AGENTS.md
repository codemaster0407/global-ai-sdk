# global-ai-sdk: reference catalog for agents

This repo is a library of working, tested building blocks for AI/ML applications. When you build a feature in **another project** that overlaps with anything below, read the matching file here first and base your code on it. Don't write it from scratch.

Repo root: `/Users/chaitanyasrikanth/Awone Files/global-ai-sdk`
Full usage examples for every component: [README.md](README.md). Read the matching section before adapting a component.

## How to reuse code from here

1. Find the task in the catalog below, then **read the whole source file**, not just the function you need. Retry logic, config and edge-case handling usually sit in private helpers (`_retry_delay`, `_load`, and so on).
2. **Copy and adapt the code into the target project.** Don't import from this repo by path or add it to `sys.path`: it isn't an installable package, and its imports (`from llm.z_ai.client import ...`) only resolve from this repo's root.
   - Rewrite internal imports to fit the target project's layout.
   - Bring over only the dependencies the copied code needs. Don't copy `requirements.txt`, which pulls in torch, ray, unstructured and more.
3. Keep what's proven: model names, retry and backoff logic, concurrency limits, preprocessing choices, and API parameters.
4. **Improve on these known weak spots when you adapt code:**
   - API clients are created at import time (`llm/z_ai/client.py`, `audio/eleven_labs/client.py`, `llm/redis_caching/client.py`). In the target project, create them lazily, or inject them.
   - Some functions catch an exception, `print` it and return `None` (for example `infer_audio`). Raise the error, or log it properly instead.
   - Paths like `data/...` and `runtime_vector_db` are relative to the working directory. Make them configurable.
   - Secrets come from `.env` via `python-dotenv`. Keep that pattern and never hard-code keys.
5. Python 3.10+ (uses `X | None` hints).
6. If a component is listed under [Known issues](#known-issues), apply the fix while adapting it.

## Catalog: task → file

### Audio (`ELEVENLABS_API_KEY`)
| Task | File | Entry point |
|---|---|---|
| Speech-to-text (file, with diarization) | [audio/eleven_labs/speech_to_text.py](audio/eleven_labs/speech_to_text.py) | `infer_audio(audio_file_path)`, model `scribe_v2` |
| Text-to-speech | [audio/eleven_labs/text_to_speech.py](audio/eleven_labs/text_to_speech.py) | `generate_audio_from_speech(text)` |
| Realtime transcription of a live audio stream | [audio/eleven_labs/realtime_stt_transcription.py](audio/eleven_labs/realtime_stt_transcription.py) | `async realtime_transcription()` |
| Browser mic → server over WebRTC, saved as 16 kHz mono WAV | [audio/speech_transfer_webrtc.py](audio/speech_transfer_webrtc.py), [audio/static/webrtc_client.html](audio/static/webrtc_client.html) | aiohttp `create_app()`, `POST /offer`, `consume_audio(track, session_id, on_audio)` |

### Computer vision
| Task | File | Entry point |
|---|---|---|
| OCR on an image | [computer_vision/ocr/tesseract_inference.py](computer_vision/ocr/tesseract_inference.py) | `ocr_inference(image_path)` |
| PDF → elements/markdown (local or Unstructured API) | [computer_vision/document_parsing/unstructured_helper.py](computer_vision/document_parsing/unstructured_helper.py) | `local_pdf_reader(filepath)`, `cloud_pdf_reader(filepath)` |
| Detect working cameras | [computer_vision/video/utils/CameraDetector.py](computer_vision/video/utils/CameraDetector.py) | `CameraDetector`, `fetch_device_id(device_type)` |
| MJPEG camera streaming server with watchdog/auto-restart | [computer_vision/streaming_backend.py](computer_vision/streaming_backend.py) | FastAPI `app`, `StreamManager` |
| Camera as an aiortc WebRTC video track | [computer_vision/video/VideoStreamTrack.py](computer_vision/video/VideoStreamTrack.py) | `VideoStreamTrack` |

### LLM
| Task | File | Entry point |
|---|---|---|
| Chat completion (Z.ai GLM, default `glm-5.3`) | [llm/z_ai/inference.py](llm/z_ai/inference.py) | `llm_response(model, system_prompt, user_prompt)` |
| Structured output → Pydantic model | [llm/z_ai/inference.py](llm/z_ai/inference.py), demo [llm/z_ai/structured_example.py](llm/z_ai/structured_example.py) | `llm_structured_response(output_model, ...)` |
| Streaming (reasoning + content) | [llm/z_ai/inference.py](llm/z_ai/inference.py) | `streaming_response(system_prompt, user_prompt)` |
| **Tool/function calling loop**, schema auto-built from typed Python functions | [llm/z_ai/inference.py](llm/z_ai/inference.py) + [llm/z_ai/tools.py](llm/z_ai/tools.py) | `inference_with_tools(...)`; register functions in `TOOLS` |
| Stream LLM output to a browser over WebSocket | [llm/z_ai/websocket_streaming_response.py](llm/z_ai/websocket_streaming_response.py) | aiohttp `create_app()`, `stream_llm(ws, ...)` |
| List available models | [llm/z_ai/get_models.py](llm/z_ai/get_models.py) | `list_models()` |
| Gemini with Google Search grounding + 429 retry that honours `retryDelay` | [llm/google/ai_studio.py](llm/google/ai_studio.py) | `generate(user_prompt, system_prompt)` |
| Prompt versioning / registry | [llm/prompt_maintenance/langfuse_client.py](llm/prompt_maintenance/langfuse_client.py) | `create_prompt(...)`, `search_prompt(name)` |
| Semantic response cache | [llm/redis_caching/functions.py](llm/redis_caching/functions.py) | async `push_data`, `search_query`, `delete_query_by_id`, `flush_entries` |
| Route a prompt to a tool with a small local model | [llm/jev/inference.py](llm/jev/inference.py) | `classify_prompt(s)(prompts, tools, instructions, criteria, threshold)` → `ToolChoice` |
| Rule-based prompt guardrail (JSON rules → allow/deny tools) | [llm/guardrails/jev_guardrail.py](llm/guardrails/jev_guardrail.py) | `JevGuardrail().classify(prompt)` → `GuardrailResult` |
| guardrails-ai patterns: validators, on_fail, input/output guards, reask, structured output | [llm/guardrails/guardrail_ai_examples.py](llm/guardrails/guardrail_ai_examples.py) | one function per pattern |
| LLM output monitoring / LLM-as-judge report | [llm/observability/observability.py](llm/observability/observability.py) | `build_dataset(df)`, `run_report(dataset, html_path)` |

### RAG
| Task | File | Entry point |
|---|---|---|
| Chunk markdown by headers, then by size | [rag/utils/text_chunk.py](rag/utils/text_chunk.py) | `chunk_text_func(text)` |
| Embeddings (Nomic, LangChain-compatible) | [rag/utils/embedding_helpers.py](rag/utils/embedding_helpers.py) | `NomicEmbeddings` / `nomic_embeddings` |
| Batched insert into Chroma | [rag/utils/vector_db_storage.py](rag/utils/vector_db_storage.py) | `vector_db_storage(chunks, batch_size, persist_directory)` |
| End-to-end indexing of a folder of markdown with per-file metadata | [rag/helpers/chunk_embeddings.py](rag/helpers/chunk_embeddings.py) | `iterate_chunk_vectorize(md_file_dir, metadata_json)` |
| Hybrid retrieval (vector + BM25) + cross-encoder rerank + metadata filter | [rag/utils/retrieval.py](rag/utils/retrieval.py) | `hybrid_retrieve(query, top_k, rerank, metadata_filter)` |
| Evaluate answers (faithfulness, relevancy) and retrieval (P@k, R@k, hit rate, MRR) | [rag/eval/rag_eval.py](rag/eval/rag_eval.py) | `evaluate_generated_output(...)`, `precision_at_k`, `recall_at_k`, `hit_rate_at_k`, `mrr` |
| Use a non-OpenAI LLM as the DeepEval judge | [rag/eval/zai_llm.py](rag/eval/zai_llm.py) | `glm_model` (`DeepEvalBaseLLM` subclass) |

### Classical & deep ML (tabular / time series)
| Task | File | Entry point |
|---|---|---|
| Auto numeric/categorical detection + impute/scale/one-hot | [ml/features/pipelines.py](ml/features/pipelines.py) | `infer_column_types`, `build_preprocessor` |
| Train + evaluate + save a sklearn/XGBoost/LightGBM pipeline | [ml/classical/training.py](ml/classical/training.py) | `train_model(df, target, model_name, task, ...)` |
| Load a saved model and predict with input validation | [ml/classical/inference.py](ml/classical/inference.py) | `ModelPredictor` |
| Hyperparameter search (Optuna, CV) | [ml/tuning/optuna_tuning.py](ml/tuning/optuna_tuning.py) | `tune_model(...)` |
| Parallel search across model families (Ray Tune) | [ml/tuning/ray_tuning.py](ml/tuning/ray_tuning.py) | `tune_with_ray(...)` |
| PyTorch MLP for tabular data, early stopping, schedulers | [ml/deep_learning/training.py](ml/deep_learning/training.py), [models.py](ml/deep_learning/models.py) | `train_tabular(df, target, config)`, `TrainConfig`, `TabularPredictor` |
| Loss functions (focal, label smoothing, Huber, class weights) | [ml/deep_learning/losses.py](ml/deep_learning/losses.py) | `get_loss(name, ...)` |
| Tune the MLP / compare losses | [ml/deep_learning/tuning.py](ml/deep_learning/tuning.py) | `tune_tabular`, `compare_losses` |
| Explainability (native, permutation, SHAP + plots) | [ml/explainability/explain.py](ml/explainability/explain.py) | `model_feature_importance`, `permutation_importance_df`, `shap_explain`, `explain_row` |
| Class imbalance (weights, SMOTE, threshold tuning) | [ml/imbalance/imbalance.py](ml/imbalance/imbalance.py) | `class_weight_params`, `build_resampled_pipeline`, `tune_threshold` |
| Forecasting with lag features + walk-forward backtest | [ml/time_series/forecasting.py](ml/time_series/forecasting.py) | `train_forecaster`, `backtest`, `Forecaster.forecast` |
| Anomaly detection (rolling z, IQR, Isolation Forest, residuals) | [ml/time_series/anomaly_detection.py](ml/time_series/anomaly_detection.py) | `rolling_zscore`, `iqr_outliers`, `IsolationForestDetector`, `residual_anomalies` |
| Experiment tracking (MLflow / W&B behind one interface) | [ml/tracking/tracker.py](ml/tracking/tracker.py) | `get_tracker(backend)`, `train_with_tracking`, `optuna_callback` |
| Feature store (Feast offline/online) | [ml/features/feast_store.py](ml/features/feast_store.py), [feature_repo/](ml/features/feature_repo/) | `get_training_data`, `get_online_features`, `materialize` |

### Serving & deployment
| Task | File | Notes |
|---|---|---|
| FastAPI server for **API-hosted** models: parallel LLM calls on threads, global semaphore, per-request deadline | [serving/api_model_server.py](serving/api_model_server.py) | `run_parallel(tasks, timeout_s)`; many workers |
| FastAPI server for a **locally loaded** model: one inference at a time, batch endpoint, warmup | [serving/local_model_server.py](serving/local_model_server.py), [serving/local_model.py](serving/local_model.py) | exactly 1 worker; swap the model in `_load` |
| Production middleware: request id, latency header, safe JSON errors, API-key auth, env config | [serving/common.py](serving/common.py) | `add_production_middleware`, `require_api_key`, `env_int` |
| Container image (multi-stage, CPU torch, non-root, healthcheck, baked weights) | [Dockerfile](Dockerfile), [docker-compose.yml](docker-compose.yml), [.dockerignore](.dockerignore) | `APP_MODULE` selects the server |

### Databases
| Task | File | Entry point |
|---|---|---|
| Pooled SQLAlchemy engine for GCP Cloud SQL for **SQL Server** (`pytds`) via the Cloud SQL Python Connector | [database/gcp/cloud_sql_connector.py](database/gcp/cloud_sql_connector.py) | `connect_with_connector()`; auth via Application Default Credentials |
| DB helpers: inspect, create/rename/truncate/drop tables, add/rename/alter/drop columns, foreign keys, unique/primary keys, indexes (Alembic ops, SQL Server-safe), and batched insert / select / update / delete with dict `where` filters | [database/gcp/db_functions.py](database/gcp/db_functions.py) | `create_table`, `add_column`, `drop_column`, `alter_column`, `add_foreign_key`, `create_index`, `insert_rows`, `select_rows`, `update_rows`, `delete_rows`, `run_sql`; optional `engine=` |

**Choosing a serving pattern:** if the model is behind an HTTP API, use `api_model_server.py` (threads, many workers). If the model weights are in process memory, use `local_model_server.py` (no threads, one worker, scale by containers).

## Environment variables

`Z_AI_API_KEY`, `GOOGLE_AI_STUDIO_KEY`, `ELEVENLABS_API_KEY`, `UNSTRUCTURED_API_KEY`, `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL`, `REDIS_DB_URI` / `REDIS_CACHE_ID` / `REDIS_KEY`, `GCP_CONNECTION_NAME` / `GCP_CLOUD_DB_USER` / `GCP_CLOUD_DB_PWD` / `GCP_DB_NAME` / `PRIVATE_IP` / `DB_ROOT_CERT`, and the serving settings (`SERVING_API_KEY`, `LLM_MODEL`, `MAX_CONCURRENT_LLM_CALLS`, `REQUEST_TIMEOUT_S`, `LLM_TIMEOUT_S`, `MAX_BATCH_SIZE`, ...). See README sections 1, 8 and 9.

## Known issues

Fix these while adapting the component:

| Component | Problem | Fix |
|---|---|---|
| WebRTC receiver | `speech_transfer_webrtc.py` imports `main` from `realtime_stt_transcription`, which doesn't define it | Drop the import or use `realtime_transcription` |
| RAG indexing | `vector_db_storage.py` imports `rag_helpers.embedding_helpers` | Use `rag.utils.embedding_helpers` |
| Gemini demo | `__main__` calls `generate()` without its two arguments | Pass `user_prompt` and `system_prompt` |
| Observability | Default report path is in a missing `evaluation/` folder | Create it or pass `html_path` |

## When working inside this repo

- Run modules from the repo root as packages: `python -m ml.classical.training`, not `python ml/classical/training.py`.
- Keep each component self-contained, and update this catalog and README.md when you add or rename a public function.
