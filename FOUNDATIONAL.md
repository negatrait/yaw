# YAW Foundations

## 1. Purpose

`yaw` is a local, zero-LLM-index multi-hop retrieval engine and QA runtime built on plain SQLite (`chunks`, `chunk_fts`, `chunk_vocab`, `rabitq_index`, `citation_graph`).

- **Tier 0 — Primary Librarian Surface (`recall`):** Queries local corpora with **zero generative LLM dependency** (`yaw/retrieval.py`), returning verified disk spans, multi-hop citation/PMI evidence routes, and SHA-256 staleness flags (`RetrievalIndex`, `SearchResult`, `search_index`).
- **Tier 1 — Benchmark QA Harness (`qa`, `optimize dspy`):** Optional local SLM reader (`dspy.ReAct` + `dspy.GEPA`, `temperature=1.0` unconditionally) used for academic EM/F1 benchmarking (`PopQA`, `MuSiQue`, `HotpotQA`, `2WikiMultiHopQA`) and interactive multi-hop synthesis (`yaw/qa.py`, `yaw/optimize.py`).

Upstream `HippoRAG 2` and `Youtu-GraphRAG` are external benchmark baselines only (`0` `hipporag` imports anywhere in `yaw/`, `scripts/`, or `tests/`).

## 2. Non-Goals & Retired Stack

`yaw` does **not** provide backward-compatibility shims, multi-backend storage abstractions, imperative SQLite bypass writers, speculative configuration knobs, wrapper subclasses (`dspy.Module`, `dspy.BaseLM`, `dspy.Adapter`), or fallback retrieval paths.

The following remain permanently retired unless controlled benchmarks prove material gains justifying their dependency weight:
`rdflib`, OWL 2 RL, `pySHACL`, LLM OpenIE triple/ontology extraction, GBNF grammars, `skos:ConceptScheme`, `dspy.RLM` / Deno WASM REPL, and legacy `yaw/daemon/**`.

## 3. Package Boundaries

All production code obeys a strict monotonic layer DAG (`L0`–`L6`, verified by `tests/core/test_architecture.py` with `0` horizontal cross-domain imports, `0` cycles, and `0` `# noqa`/`C901` suppressions):

```text
yaw/
├── core/        # L0: __init__.py, config.py, logger.py, types.py (Python stdlib only; 0 ML/DB imports)
└── ...          # L1 datasets, retrieval -> L2 pipeline, qa -> L3 runtime -> L4 mcp_server, optimize -> L5 cli -> L6 __main__
```

- Modules never import another module's `_private` symbols or access foreign `_` attributes.
- Transport adapters (`cli.py`, `mcp_server.py`) contain zero retrieval or ingestion logic; `mcp_server.py` serves Streamable HTTP on loopback `127.0.0.1:13306/mcp` with `/healthz`, `/readyz`, pre-dispatch `Host`/`Origin` `403`, and exact `3` read-only tools (`recall`, `qa`, `daemon_status`).
- `cli.py` lazy-imports `optimize.py` inside optimization subcommands so Tier-0 commands run standalone outside a git checkout.

## 4. System Invariants

1. **Declarative Ingestion (`yaw/pipeline.py`):**
   - Sole writer to `.yaw/yaw.db` via event-driven CocoIndex (`localfs.walk_dir(..., live=live)`), per-`db_path` `app_name`, native `CodeSource`/`RecursiveSplitter` (`included_patterns` / `chunk_size` UTF-8 byte cap), configurable `embed_batch_size` + `embed_max_in_flight`, and persistent content-addressed `.yaw/coco_lmdb` memoization.
   - Publishes the 5-table plain SQLite schema: `chunks` (ROWID via `ChunkRow` / `sqlite.mount_table_target` with `<f2>` `embedding_f16 BLOB` + source spans, `0` raw chunk text), `chunk_fts` (`unicode61 remove_diacritics 1`) + `chunk_vocab` (`fts5vocab` via `_FtsTableHandler`), and single-row `rabitq_index` + `citation_graph` materialized via `reconcile_index_tables` (`IndexArtifacts`). Native dataset bundles export directly to `.md` files (`<idx>.md`, `# Title` headings) and index without duplicate JSON-to-text re-staging.
2. **Pointer-Only Disk Hydration (`yaw/retrieval.py`):**
   - `yaw.db` stores 1-indexed `(rel_posix_path, start_line, end_line)`, character offsets `(char_start, char_end)`, and SHA-256 `source_hash`—never authoritative copies of corpus text.
   - Passages hydrate pure line/character spans uniformly from disk at query time (`max_chars=None` uncropped by default, `pool_k = max(top_k * 4, 50)` before span deduplication, zero serde JSON reparsing): missing or hash-mismatched files return `stale_on_disk = True` with `text = ""`.
3. **Zero-LLM Hybrid Ranking (`yaw/retrieval.py`, `yaw/runtime.py`):**
   - Combines 1-bit SoA RaBitQ (`rabitq_index`), `FTS5` (`chunk_fts` / `chunk_vocab`), and tangent-steered cyclic citation + 1-bit PMI diffusion (`CitationGraph`, `DiffusionOperators`) inside `RetrievalIndex` (`build_retrieval_index`, `empty_retrieval_index`, `search_index`, `expand_citation_neighbors`) over `?mode=ro` WAL SQLite connections (`busy_timeout=30000`), hot-swapping outside `index_lock` via `PRAGMA data_version` (`get_retrieval_index`).
4. **Runtime Concurrency & Native Tool Calling (`yaw/runtime.py`, `yaw/qa.py`, `yaw/optimize.py`):**
   - `recall` runs under `BoundedSemaphore(32)` over `?mode=ro` SQLite connections with zero LM endpoint dependency.
   - Bare `dspy.ReAct` QA and `dspy.GEPA` instruction proposal declare `ProviderDefinition.chat` (`ModelSupport(function_calling=True)`) via `dspy.lm15.register_provider`, enforce `tool_choice="required"`, serialize under `Semaphore(1)`, run student and GEPA reflection at `temperature=1.0` unconditionally, and validate inline `[chunk_id=<int>]` citations strictly against non-`submit` tool observations (`0` SQLite writes during QA).

## 5. Failure Policy & Acceptance Criteria

- **Fail-Fast Semantics:** Transient model-service retries use `dspy.LM(num_retries=...)` exclusively. Exhausted retries, SQLite operational errors, invalid TOML config, or invariant violations abort immediately with the original exception or typed `YawRuntimeError("[CODE]", ...)`—zero silent catch-and-continue, negative clamping, or degraded fallback loops.
- **Evidence-Gated Acceptance:** Unit/integration tests exercise real SQLite/CocoIndex/FastMCP/filesystem substrates without mocks. Cutovers are complete only when callers, CLI/MCP surfaces, and telemetry (`scripts/watchdog.py`) pass whole-repo quality gates cleanly.

## 6. Governing Specs

- Governing Sprint & Cutover Ledgers: [bare-bones-replacement-sprint.md](docs/reviews/bare-bones-replacement-sprint.md) (`CS-0`..`CS-6` clean-slate cutover, closed on `3cd3952`); [mdl-dag-sprint.md](docs/reviews/mdl-dag-sprint.md) (`EVAL-0`..`EVAL-5` shipped frozen `M_FINAL_1B` on `dev` at `5c4b115`); [epic-3-librarian-wiki-sprint.md](docs/reviews/epic-3-librarian-wiki-sprint.md) (active Sprints `3.1`–`3.3`: modular DAG governance, unified CocoIndex/FastMCP lifespan, `log_bf` + `coverage` slip, and dual-root Librarian `wiki_root`).
- Epics: [Epic 1](docs/epic-1-hipporag-daemon.md), [Epic 2](docs/epic-2-fastmcp-http.md), [Epic 3](docs/epic-3-performance.md) *(where historical `## Decisions` preserve pre-cutover rationale verbatim, §§1–5 above govern active production code)*.
