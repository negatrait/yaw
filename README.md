# YAW: HippoRAG 2 evaluation

YAW evaluates upstream HippoRAG 2 on a small, messy corpus. Its next steps are a local daemon and a FastMCP v4 interface. Read [FOUNDATIONAL](FOUNDATIONAL.md), then the planned [daemon epic](docs/epic-1-hipporag-daemon.md) and [Streamable HTTP epic](docs/epic-2-fastmcp-http.md).

## Available now

Use Python 3.12 and uv:

```powershell
uv sync --locked
uv run --locked python -c "from hipporag import HippoRAG; print('HippoRAG import OK')"
uv run --locked python -m unittest discover -s tests -v
uv run --locked python live_smoke.py
uv run --locked python live_smoke.py --mode eval
uv run --locked python live_smoke.py --mode eval --embedding-task-mode distinct
uv run --locked python live_smoke.py --mode eval --embedding-task-mode distinct-titled
uv run --locked python live_smoke.py --mode eval --chat-model gemma4-it-12b-FLM --embedding-model Qwen3-Embedding-0.6B-GGUF
```

The smoke script uses nine fixture passages and three questions against Lemonade at `http://localhost:13305/api/v1`. It first tests embedding repeatability, then builds a **fresh** index under `outputs/live-smoke/run-*/`. By default (`--mode all`) it runs separate retrieval/QA, combined QA and an explicit gold evaluation with the same five-passage limit. `--mode eval` runs only gold evaluation; `--mode smoke` runs the first two. It writes `run.log` and `report.json`, and exits nonzero on unstable vectors, incorrect ranking, missing evidence, empty `graph_seeds` (silent DPR fallback) or failed answer checks. Source notes are not ingested by this command.

Evaluation passes `GOLD_DOCS` and `GOLD_ANSWERS` to upstream `rag_qa` and also records dense-only baseline recall via `retrieve_dpr`. The console and JSON report show each prediction, gold documents/answers, retrieval recall@1/@2/@5, and per-question plus aggregate answer exact match/F1. F1 is normalized answer-token overlap, not semantic correctness; correct paraphrases may score below 1. Gold references stay fixed independently of generated answers. In `all` mode later calls may reuse upstream caches, so use `--mode eval` for a fresh standalone evaluation.

The default 3-slot development split uses:
- **Embeddings (`--embedding-model`):** `embeddinggemma-300m-GGUF` on GPU CUDA `llamacpp` (`--embedding-task-mode` supports `search`, `qa`, `fact`, `distinct`, `titled`, `distinct-titled`; `--embedding-model` also accepts `embeddinggemma-300m-qat-q8_0-GGUF-Q8_0` and `Qwen3-Embedding-0.6B-GGUF`).
- **OpenIE & Recognition Memory (`--extraction-model`):** `Qwen3-4B-Instruct-2507-GGUF` on GPU CUDA `llamacpp` with tightened OpenIE NER JSON Schema (`minLength: 1`), dynamic entity-anchored triple GBNF grammar, and dynamic candidate-literal `DSPyFilter` GBNF grammar enforcement.
- **QA Reading (`--qa-model`):** `Mistral-7B-Instruct-v0.3-Hybrid` on Ryzen AI Hybrid NPU/iGPU (`ryzenai-llm`) with `Answer:` delimiter validation.
- Passing `--chat-model` overrides both `--extraction-model` and `--qa-model` with a single model (such as step-up `gemma4-it-12b-FLM` or `qwen3.5-9b-FLM`).

There is one embedding adapter (`yaw/embedding.py`), one chat policy subclass (`yaw/chat.py`), and no fallback. See [model contracts and real benchmark protocol](docs/model-evaluation.md). No upstream classes are monkeypatched.

On this machine the models and settings are already installed. For a fresh Lemonade installation, register/download the embedding checkpoint and save its settings explicitly before running the smoke test:

```powershell
$api = 'http://localhost:13305/api/v1'
$pull = @{
    model_name = 'user.embeddinggemma-300m-GGUF'
    checkpoint = 'ggml-org/embeddinggemma-300M-GGUF:embeddinggemma-300M-Q8_0.gguf'
    recipe = 'llamacpp'
    embedding = $true
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "$api/pull" -ContentType 'application/json' -Body $pull
$load = @{
    model_name = 'user.embeddinggemma-300m-GGUF'
    ctx_size = 2048
    llamacpp_backend = 'cuda'
    llamacpp_args = '--pooling mean --parallel 1'
    save_options = $true
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "$api/load" -ContentType 'application/json' -Body $load
```

This is the verified local CUDA configuration; the script does not install models or choose another backend on failure. Keep the selected extraction and QA chat models downloaded in Lemonade too. A different serving configuration requires a new repeatability/quality validation and index identity. The reported 8192 server allocation does not enlarge EmbeddingGemma's 2048-token model context.

HippoRAG is pinned to upstream commit `d5c8329422e0a0b834a15874545cb6a74b4f9b26`, declaring `2.0.0a5`. This is a source installation requiring Git; a published a5 wheel was unavailable when checked. NumPy is explicitly declared for the experiment. Python 3.14.7 resolution failed because upstream pins PyTorch 2.5.1 without a CPython 3.14 wheel, so the verified interpreter is CPython 3.12.14.

Existing source notes remain in `sample-llm-wiki/raw/`; they are not automatically indexed by the current script. Milvus and MCP packages are not installed. HippoRAG's default Parquet stores already support the executed pipeline.

## Measured versus planned

The 2026-10-14 default 3-slot split (`Qwen3-4B-Instruct-2507-GGUF` extraction + `Mistral-7B-Instruct-v0.3-Hybrid` QA + `embeddinggemma-300m-GGUF` embeddings) passed in 29.72s (`search`, `run-gudguenl`) and 29.95s (`distinct-titled`, `run-ewmwi3fx`): answer exact match 0.6667, F1 0.8444, recall@1/@2/@5 0.6111/0.8889/1.0000, with 15 phrase nodes, 10 extracted triples, and non-empty graph seeds on all three questions. Step-up `gemma4-it-12b-FLM` and `qwen3.5-9b-FLM` evaluations also passed (118.41s–244.40s).

Twenty-six offline regression tests pass (`uv run --locked python -m unittest discover -s tests -v`). See [evaluation](docs/evaluation.md) for commands, versions, answer-score limitations and preserved failure diagnostics.

Epic 1 has its smoke-test embedding and chat adapters implemented; multilingual/chunk-boundary validation and daemon work remain. Epic 2 is planned. No daemon commands or MCP endpoint are implemented yet.

Upstream: [pinned HippoRAG source](https://github.com/OSU-NLP-Group/HippoRAG/tree/d5c8329422e0a0b834a15874545cb6a74b4f9b26).
