# YAW

YAW is a local-first retrieval and QA library built on a single plain SQLite database (`yaw.db`) with 5 tables (`chunks` ROWID w/ `<f2>` `embedding_f16 BLOB`, contentless `FTS5` `chunk_fts`, `chunk_vocab` `fts5vocab`, `rabitq_index`, `citation_graph`), live declarative `CocoIndex` ingestion, zero-LLM `RetrievalIndex` (`recall`), and bare `DSPy` (`dspy.ReAct` QA + `dspy.GEPA` optimization). Upstream HippoRAG is a historical benchmark reference only; legacy HippoRAG OpenIE/PPR/W3C/OWL/SKOS/SHACL/GBNF/`dspy.RLM` machinery is retired into a modular layer-contained `yaw/` package (`L0`–`L6` DAG) under the [Clean-Slate Replacement Sprint](docs/reviews/bare-bones-replacement-sprint.md) and frozen `M_FINAL_1B` ([Epic 3 Decisions](docs/epic-3-performance.md#decisions)).

Read [FOUNDATIONAL](FOUNDATIONAL.md), [Epic 1](docs/epic-1-hipporag-daemon.md), [Epic 2](docs/epic-2-fastmcp-http.md), and the [Clean-Slate Replacement Sprint](docs/reviews/bare-bones-replacement-sprint.md). Reference papers are preserved locally under [HippoRAG 1 (`2405.14831v3`)](docs/arxiv-2405.14831v3-hipporag.md) and [HippoRAG 2 (`2502.14802v1`)](docs/arxiv-2502.14802v1-hipporag2.md).

## Setup

Use Python `>=3.14` (`.python-version` `3.14.9`) and `uv`:

```powershell
uv sync
```

## Active CLI

```powershell
uv run yaw daemon {run|start|stop|status} [--config PATH] [--json]
uv run yaw recall <query> [--top-k 5] [--config PATH] [--json]
uv run yaw qa <question> [--config PATH] [--json]
uv run yaw optimize dspy [--dataset ...] [--max-queries N] [--query-offset M] [--threads N] [--fresh]
uv run yaw config {show|get [key]|set <key> <value>} [--config PATH] [--json]
uv run python -m scripts.watchdog --once [--artifact-dir PATH]
uv run python -m scripts.download_datasets [--dataset all]
uv run python -m scripts.benchmark --dataset sample --mode retrieval
uv run python -m scripts.verify_paper_parity_gate --mode eval
```

Guarded parity/tier-2 benchmarks (`2wikimultihopqa`, `hotpotqa`, `musique`, `popqa`, `nq`, `lveval`, `narrativeqa`) reserve query slice `[0, 10)` for held-out evaluation; pass `--query-offset 10` or set `[optimization].query_offset >= 10` in [`yaw.toml`](yaw.toml) when optimizing against those datasets.

## Architecture & Model Stack

The replacement CLI handles Ctrl-C, SIGTERM, and Windows Ctrl-Break during optimization. The first signal allows cleanup. Repeat it to force exit if cleanup or a provider worker is stuck. Forced exit can lose unfinished work. Restart an existing process to load this behavior.

- **Declarative Live Indexing (`yaw/pipeline.py`):** Continuous `CocoIndex` (`localfs.walk_dir(..., live=live)`) writes the 5-table plain SQLite schema (`chunks` ROWID via `ChunkRow` with `<f2>` `embedding_f16 BLOB` + source spans and `0` raw chunk text, contentless `chunk_fts` using `unicode61 remove_diacritics 1`, `chunk_vocab` `fts5vocab`, and single-row `rabitq_index` + `citation_graph` materialized via `reconcile_index_tables` from `IndexArtifacts`) with persistent LMDB memoization under `.yaw/coco_lmdb`. Native `CodeSource`/`RecursiveSplitter` parse each file by its real format; titles are temporarily empty (`title: none`) pending native AST metadata support. Prepared v2 Markdown dataset bundles (`# {title}` headings) are indexed directly via `resolve_slice_corpus_scope` with zero `.txt` staging or JSON reparsing. `[ingestion].chunk_size` bounds UTF-8 chunk bytes (an indivisible oversized native chunk fails visibly); `embed_batch_size` sets the maximum separate inputs per native CocoIndex/DSPy batch. `[inference].embed_max_in_flight` bounds concurrent embedding HTTP calls across ingestion and queries. Batch token usage is the sum across inputs; chunk size is not a token limit.
- **Zero-LLM Hybrid `RetrievalIndex` (`yaw/retrieval.py`, `yaw/runtime.py` -> `yaw recall`):** Read-only WAL retrieval (`build_retrieval_index`, `empty_retrieval_index`, `get_retrieval_index` via `PRAGMA data_version` snapshot reload) combining 1-bit extended rotated SoA RaBitQ (`rabitq_index`), `FTS5`/`chunk_vocab` lexical scoring, tangent-steered cyclic citation + 1-bit PMI diffusion (`CitationGraph`, `DiffusionOperators`), rank-1 Cholesky greedy L-ensemble 1-bit DPP reranking (`search_index`, `SearchResult`, `expand_citation_neighbors`), and MaxP span deduplication with SHA-256 disk line hydration.
- **Bare `dspy.ReAct` QA & `dspy.GEPA` (`yaw/qa.py`, `yaw/optimize.py`):** Unwrapped `dspy.ReAct` over 4 read-only uncropped disk-hydration corpus tools (`search_passages`, `lexical_grep`, `expand_neighbors`, `read_chunk`), serialized under `Semaphore(1)` and verified against inline `[chunk_id=<int>]` tool provenance; QA-only `dspy.GEPA` optimization isolates telemetry in `.yaw/optimize_*` (resume by default; `--fresh` wipes `.yaw/optimize_*` while preserving `.yaw/coco_lmdb`).
- **Streamable HTTP Transport (`yaw/mcp_server.py`, [Epic 2](docs/epic-2-fastmcp-http.md)):** `FastMCP("yaw")` on loopback `http://127.0.0.1:13306/mcp` with pre-dispatch `Host`/`Origin` `403` validation, `/healthz`, `/readyz`, and 3 read-only tools (`recall`, `qa`, `daemon_status`).

### Active Local Inference Configuration (`http://127.0.0.1:8888/v1`, [`yaw.toml`](yaw.toml))

- **Embeddings:** `embeddinggemma-2` (`distinct-titled` framing, 768-dim unit-norm vectors, `BoundedSemaphore(2)` shared across indexing and queries).
- **QA & Reflection LLM:** `gemma-4-e4b` on local GPU `llama.cpp` (`--parallel 1`, `temperature = 1.0`).

## Benchmarks & Baselines

`scripts/benchmark.py` and `scripts/verify_paper_parity_gate.py` evaluate YAW against Standard DPR and two consolidated targets (see [evaluation ledger](docs/evaluation.md)):

- **Target 1 (`MONKEYPATCHED_HIPPORAG_BASELINES`):** Archived historical runs of upstream HippoRAG 2 on the older `Qwen3-4B-Instruct-2507-GGUF` + `embeddinggemma-300m-GGUF` stack only (stored in [`outputs/benchmarks/consolidated_historical_baselines.json`](outputs/benchmarks/consolidated_historical_baselines.json)).
- **Target 2 (`PAPER_BASELINES`):** Published HippoRAG 2 paper scores (`Llama-3.3-70B-Instruct` + `NV-Embed-v2`, `arXiv:2502.14802v1`).

Across the `E1-M1` Pass-4 held-out `[0, 10)` shootout (`n=40` queries across `2wikimultihopqa`, `musique`, `hotpotqa`, `popqa`), pure-chunk hybrid retrieval scored macro `R@1/2/5/10/20 = 0.4333 / 0.6583 / 0.8146 / 0.8938 / 0.9854`, clearing `PAPER_BASELINES` `Recall@5` (`0.7828`) and outperforming legacy OpenIE + `ppr.py` (`R@5 = 0.7021`).

## Quality Gates

```powershell
uv run ty check
uv run ruff check --fix
uv run ruff format
uv run --locked pytest
git diff --check
```

See [coverage plan](docs/coverage-plan.md) for full `--cov` verification commands. Remaining upstream algorithm references:

1. **HippoRAG (`arXiv:2405.14831v3`):** [arXiv:2405.14831](https://arxiv.org/abs/2405.14831) / [Local Copy](docs/arxiv-2405.14831v3-hipporag.md)
2. **HippoRAG 2 (`arXiv:2502.14802v1`):** [arXiv:2502.14802](https://arxiv.org/abs/2502.14802) / [Local Copy](docs/arxiv-2502.14802v1-hipporag2.md)
3. **MaxP Passage Retrieval (Dai & Callan, SIGIR 2019):** [arXiv:1905.01988](https://arxiv.org/abs/1905.01988)
4. **DSPy & GEPA (`arXiv:2310.03714`, `arXiv:2406.11695`):** Declarative `ReAct` and reflective Pareto optimization
