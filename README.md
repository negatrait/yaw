# YAW: HippoRAG 2 evaluation

YAW evaluates upstream HippoRAG 2 on multi-hop and messy corpora. Its next steps are a local daemon and a FastMCP v4 interface. Read [FOUNDATIONAL](FOUNDATIONAL.md), then the planned [daemon epic](docs/epic-1-hipporag-daemon.md) and [Streamable HTTP epic](docs/epic-2-fastmcp-http.md). Reference papers are preserved locally under [HippoRAG 1 (`2405.14831v3`)](docs/arxiv-2405.14831v3-hipporag.md) and [HippoRAG 2 (`2502.14802v1`)](docs/arxiv-2502.14802v1-hipporag2.md).

## Available now

Use Python 3.12 and uv:

```powershell
uv sync --locked
uv run --locked pytest -v
uv run --locked python -m yaw daemon status --config yaw.toml
uv run --locked python -m yaw daemon run --config yaw.toml
uv run --locked python -m yaw daemon start --config yaw.toml
uv run --locked python -m yaw daemon stop --config yaw.toml
uv run --locked python -m yaw graph build --config yaw.toml
uv run --locked python -m yaw graph rebuild --config yaw.toml
uv run --locked python -m yaw dspy optimize --config yaw.toml --mode gepa --auto light
uv run --locked python -m yaw dspy load-program .yaw/compiled_program_optimize_<datestamp>.json
uv run --locked python -m yaw config get optimization.auto
uv run --locked python -m yaw config set optimization.auto light
uv run --locked python benchmark.py --dataset sample --mode all
uv run --locked python benchmark.py --dataset 2wikimultihopqa --max-queries 10
uv run --locked python benchmark.py --dataset hotpotqa --max-queries 10 --mode retrieval
uv run --locked python benchmark.py --dataset musique --max-queries 10 --mode retrieval
uv run --locked python benchmark.py --dataset sample --chat-model gemma4-it-12b-FLM --embedding-model Qwen3-Embedding-0.6B-GGUF
```

`benchmark.py` replaces the retired `live_smoke.py` fixture script and evaluates upstream HippoRAG 2 (`retrieve` + optional `qa`) head-to-head against StandardRAG (`retrieve_dpr` + optional `qa`) on the upstream reproduction datasets (`sample`, `2wikimultihopqa`, `musique`, `hotpotqa`) against Lemonade at `http://localhost:13305/api/v1`. Pass `--mode retrieval` to evaluate retrieval (`Recall@1/@2/@5`) without invoking the QA reader model. Dataset files are fetched once from pinned commit `d5c8329422e0a0b834a15874545cb6a74b4f9b26` and cached under `outputs/datasets/` (skipping `{dataset}_corpus.json` when `--max-queries` is used without `--full-corpus`). Corpus passages are checkpointed through OpenIE in bounded micro-batches (`--index-batch-size 8` by default via `pre_openie`) and indexed once into deterministic persisted directories under `outputs/benchmarks/indexes/` (keyed by dataset slice hash, resolved embedding identity, and extraction chat policy identity; use `--fresh-index` for a disposable per-run index), and queries/QA pairs are processed 1-by-1 (`--qa-batch-size 1` by default) with in-memory retrieval matrices released after retrieval. Each run writes `run.log` and `report.json` under `outputs/benchmarks/run-<dataset>-*/`, comparing HippoRAG 2 and Standard DPR retrieval Recall@1/@2/@5 (and QA ExactMatch/F1 when QA is enabled) alongside the HippoRAG 2 paper's `Llama-3.3-70B-Instruct` + `NV-Embed-v2` (8B) reference numbers. Source notes in `sample-llm-wiki/raw/` are not ingested by this command.

The default 3-slot development split uses:
- **Embeddings (`--embedding-model`):** `embeddinggemma-300m-GGUF` on GPU CUDA `llamacpp` (`--embedding-task-mode` defaults to `distinct-titled` under `retrieval-prompts-v3` and supports `search`, `qa`, `fact`, `distinct`, `titled`, `distinct-titled`; `--embedding-model` also accepts `embeddinggemma-300m-qat-q8_0-GGUF-Q8_0` and `Qwen3-Embedding-0.6B-GGUF`).
- **OpenIE & Recognition Memory (`--extraction-model`):** `Qwen3-4B-Instruct-2507-GGUF` on GPU CUDA `llamacpp` (`--parallel 1 --cache-ram 0`) with tightened OpenIE NER and triple JSON Schemas (`minLength: 1`) and dynamic candidate-literal `DSPyFilter` GBNF grammar enforcement (`gbnf-v3:require-stop`).
- **QA Reading (`--qa-model`):** `Mistral-7B-Instruct-v0.3-Hybrid` on Ryzen AI Hybrid NPU/iGPU (`ryzenai-llm`) with end-of-turn `QA_TURN_REMINDER` and `Answer:` delimiter normalization (`fold-system:prompt-v4:require-stop`).
- Passing `--chat-model` overrides both `--extraction-model` and `--qa-model` with a single model (such as step-up `gemma4-it-12b-FLM` or `qwen3.5-9b-FLM`).

There is one embedding adapter (`yaw/embedding.py`), one chat policy subclass (`yaw/chat.py`), and no fallback. See [model contracts and real benchmark protocol](docs/model-evaluation.md). No upstream classes are monkeypatched.

On this machine the models and settings are already installed. For a fresh Lemonade installation, register/download the embedding checkpoint and save both CUDA `llamacpp` model settings explicitly before running benchmarks (`--batch-size 2048 --ubatch-size 2048` on EmbeddingGemma prevents `llama-server`'s default 512-token physical microbatch cutoff on 512–2048 token passages, and `--cache-ram 0` without `-dio` on `Qwen3-4B-Instruct-2507-GGUF` prevents Windows Direct-IO and host-RAM prompt-cache growth):

```powershell
$api = 'http://localhost:13305/api/v1'
$pull = @{
    model_name = 'user.embeddinggemma-300m-GGUF'
    checkpoint = 'ggml-org/embeddinggemma-300M-GGUF:embeddinggemma-300M-Q8_0.gguf'
    recipe = 'llamacpp'
    embedding = $true
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "$api/pull" -ContentType 'application/json' -Body $pull
$loadEmbed = @{
    model_name = 'user.embeddinggemma-300m-GGUF'
    ctx_size = 2048
    llamacpp_backend = 'cuda'
    llamacpp_args = '--pooling mean --parallel 1 --batch-size 2048 --ubatch-size 2048'
    save_options = $true
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "$api/load" -ContentType 'application/json' -Body $loadEmbed
$loadChat = @{
    model_name = 'Qwen3-4B-Instruct-2507-GGUF'
    ctx_size = 16384
    llamacpp_backend = 'cuda'
    llamacpp_args = '--parallel 1 --cache-ram 0'
    save_options = $true
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "$api/load" -ContentType 'application/json' -Body $loadChat
```

This is the verified local CUDA configuration; the script does not install models or choose another backend on failure. Keep the selected extraction and QA chat models downloaded in Lemonade too. A different serving configuration requires a new repeatability/quality validation and index identity. The reported 8192 server allocation does not enlarge EmbeddingGemma's 2048-token model context.

HippoRAG is pinned to upstream commit `d5c8329422e0a0b834a15874545cb6a74b4f9b26`, declaring `2.0.0a5`. This is a source installation requiring Git; a published a5 wheel was unavailable when checked. NumPy is explicitly declared for the experiment. Python 3.14.7 resolution failed because upstream pins PyTorch 2.5.1 without a CPython 3.14 wheel, so the verified interpreter is CPython 3.12.14.

Existing source notes remain in `sample-llm-wiki/raw/`; they are not automatically indexed by the current script. Milvus and MCP packages are not installed. HippoRAG's default Parquet stores already support the executed pipeline.

## Measured versus planned

On the 2026-10-14 Tier 0 and Tier 1 upstream benchmark runs with the default split (`embeddinggemma-300m-GGUF` `distinct-titled` `retrieval-prompts-v3` + `Qwen3-4B-Instruct-2507-GGUF` `gbnf-v3:require-stop` + `Mistral-7B-Instruct-v0.3-Hybrid` `fold-system:prompt-v4:require-stop`, `temperature = 0.2`):
- **Tier 0 (`sample`, 1 query, 3 passages, `run-sample-hgj5zm7g`):** HippoRAG 2 Recall@1/@2/@5 = **1.0 / 1.0 / 1.0** (30 phrase nodes, 29 triples).
- **Tier 1 (`2wikimultihopqa`, 10 queries, 96 passages, `run-2wikimultihopqa-fbmnosfw`):** HippoRAG 2 Recall@1/@2/@5 = **0.425 / 0.800 / 1.000** vs. Standard DPR **0.425 / 0.725 / 0.850** (**`+7.5%` Recall@2**, **`+15.0%` Recall@5**; 866 phrase nodes, 905 triples, 8/10 queries with graph seeds); HippoRAG 2 QA EM/F1 = **0.300 / 0.5278** vs. Standard DPR **0.400 / 0.4500** (**`+7.78%` F1**).
- **Tier 1 (`hotpotqa`, 10 queries, 94 passages, `run-hotpotqa-zbp9k8e0`):** HippoRAG 2 Recall@1/@2/@5 = **0.500 / 0.850 / 1.000** vs. Standard DPR **0.450 / 0.700 / 0.950** (**`+5.0%` Recall@1**, **`+15.0%` Recall@2**, **`+5.0%` Recall@5**; 1,206 phrase nodes, 1,286 triples, 10/10 queries with graph seeds); HippoRAG 2 QA EM/F1 = **0.300 / 0.5483** vs. Standard DPR **0.300 / 0.4524** (**`+9.59%` F1**).
- **Tier 1 (`musique`, 10 queries, 197 passages, `run-musique-_5f_uroh`):** HippoRAG 2 Recall@1/@2/@5 = **0.1833 / 0.3167 / 0.5417** vs. Standard DPR **0.2083 / 0.2833 / 0.5167** (**`+3.34%` Recall@2**, **`+2.5%` Recall@5**; 2,268 phrase nodes, 2,346 triples, 7/10 queries with graph seeds); HippoRAG 2 QA EM/F1 = **0.200 / 0.3000** vs. Standard DPR **0.300 / 0.4000**.

Thirty-two offline regression tests pass (`uv run --locked python -m unittest discover -s tests -v`). See [evaluation](docs/evaluation.md) for commands, versions, answer-score limitations and preserved failure diagnostics.

Epic 1 has its embedding and chat adapters plus Tier 0/1 benchmark harness implemented; multilingual/chunk-boundary validation and daemon work remain. Epic 2 is planned. No daemon commands or MCP endpoint are implemented yet.

Upstream: [pinned HippoRAG source](https://github.com/OSU-NLP-Group/HippoRAG/tree/d5c8329422e0a0b834a15874545cb6a74b4f9b26).
