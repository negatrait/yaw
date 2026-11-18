# YAW Foundations

## 1. Purpose

`yaw` is a local, zero-LLM-index multi-hop retrieval engine and context compiler built on SQLite (`sqlite-vec` + contentless `FTS5`).

- **Tier 0 — Primary Librarian Surface (`recall`, `neighbors`):** Compiles large, messy, ambiguous corpora into a minimal, cycle-free **Evidence DAG** at query time with **zero generative LLM dependency**—returning verified disk spans, topological bridge edges, calibrated completeness (`explained_r2`), and missing-evidence diagnostics (`stalled_anchors`, `near_miss_terms`).
- **Tier 1 — Benchmark QA Harness (`qa`, `optimize dspy`):** Optional local SLM reader (`dspy.ReAct` + `dspy.GEPA`) used strictly for academic end-to-end EM/F1 benchmarking (`PopQA`, `MuSiQue`, `HotpotQA`, `2WikiMultiHopQA`) and optional sub-agent synthesis.

Upstream `HippoRAG 2` and `Youtu-GraphRAG` are external benchmark baselines only (`0` `hipporag` imports anywhere in `yaw/`, `scripts/`, or `tests/`).

## 2. Non-Goals & Retired Stack

`yaw` does **not** provide backward-compatibility shims, multi-backend storage abstractions, imperative SQLite bypass writers, speculative configuration knobs, or fallback retrieval paths.

The following remain permanently retired unless a controlled holdout benchmark proves material Recall@K, latency, and memory gains justifying the dependency weight:
`rdflib`, OWL 2 RL, `pySHACL`, LLM OpenIE triple/ontology extraction, GBNF grammars, `skos:ConceptScheme`, `dspy.RLM`, `spacy` (`en-core-web-sm`), and `optuna` retrieval sweeps.

## 3. Package Boundaries

All production code lives inside a flat 14-file boundary:

```text
yaw/
├── core/        # __init__.py, config.py, logger.py, types.py (Python stdlib only; 0 ML/DB imports)
└── *.py         # __init__, __main__, cli, datasets, mcp_server, optimize, pipeline, qa, retrieval, runtime
```

- Modules never import another module's `_private` symbols.
- Transport adapters (`cli.py`, `mcp_server.py`) contain zero retrieval or ingestion logic.
- `cli.py` lazy-imports `optimize.py` and `datasets.py` inside optimization subcommands so Tier-0 commands run standalone outside a git checkout.

## 4. System Invariants

1. **Declarative Ingestion & Ingest Gate (`yaw/pipeline.py`):**
   - Sole writer to `.yaw/yaw.db` via event-driven CocoIndex (`localfs.walk_dir(..., live=True)`), dataset-agnostic `64`-chunk micro-batches per `app.update()`, and a persistent content-addressed CocoIndex LMDB memo cache. Never encode dataset splits or benchmark presets in production code.
   - `tree-sitter-language-pack` supplies AST/Markdown `(ast_island_start, ast_island_end, title)` and structural symbol/link anchors (`json_document_fields` unwraps benchmark JSON documents).
   - A 64-bit SimHash + conditional `zlib(zdict)` novelty gate (`novelty_q8 < 51`) reuses prototype embeddings without invoking the neural embedder on boilerplate/log clones. Disk embeddings remain chunk-local (never coupled to corpus-wide whitening).
   - Mounts dual contentless `FTS5` indexes plus `fts5vocab(..., 'row')` views: `chunk_fts` (`unicode61` prose/phrases) and `anchor_fts` (`unicode61 tokenchars '_-./:@$'` exact paths/CVEs/identifiers).
2. **Pointer-Only Disk Hydration:**
   - `yaw.db` stores 1-indexed `(rel_posix_path, start_line, end_line)`, SHA-256 `source_hash`, `mtime_ns`, and `novelty_q8`—never authoritative copies of corpus text.
   - Passages hydrate from disk at query time: hash mismatches return `stale_on_disk = True` with `text = ""`; mid-read file mutations raise `CONSISTENCY_ERROR`.
3. **Stage-1 `RaBitQ` & MDL DAG Solver (`yaw/retrieval.py`):**
   - `VectorSearch` holds **no** full-corpus `(N, D)` `float32` array or JSON token/entity dicts in RAM—only `~100B/chunk` Stage-1 `RaBitQ` codes, Marchenko–Pastur shrinkage whitening $(\mu, V_{\text{spike}}, \Delta w, D_{\text{eff}})$, `csls_rk`, and `next_in_file`.
   - Exact Stage-2 vectors are read from `chunks` strictly for bidirectional wavefront survivors (`<= 192`). All `FTS5` queries specify `ORDER BY bm25(...) ASC LIMIT ?`.
   - Candidate chains are scored via orthogonal bridge-subspace projection (`qr`) and assembled into a cycle-free DAG via parameter-free Shannon/MDL `scipy.optimize.nnls` bit accounting + `zlib` path novelty.
4. **Runtime Concurrency & Lazy LLM Admission (`yaw/runtime.py`):**
   - Tier-0 reads (`recall`, `neighbors`) run under `BoundedSemaphore(32)` over `?mode=ro` SQLite connections and succeed when no generative LLM endpoint is configured or reachable.
   - Tier-1 `dspy.LM` initializes lazily on first `qa` call under `Semaphore(1)` (`temperature=1.0` student / `0.7` GEPA reflection; embeddings batched at `max_batch_size = 64`).

## 5. Failure Policy & Acceptance Criteria

- **Fail-Fast Semantics:** Transient model-service retries use `dspy.LM(num_retries=...)` exclusively. Exhausted retries, SQLite operational errors, or invariant violations abort immediately with the original exception or typed `YawRuntimeError("[CODE]", ...)`—zero silent catch-and-continue or degraded fallback loops.
- **Evidence-Gated Acceptance:** Unit tests prove mechanics, not retrieval quality. Cutovers are complete only when all callers, CLI/MCP surfaces, and telemetry (`scripts/watchdog.py`) are wired end-to-end and backed by reproducible benchmark artifacts (`Recall@K`, Reject-mode calibration, RAM/latency envelopes, and Tier-1 EM/F1).

## 6. Governing Specs

- Active migration ledgers: `docs/reviews/` (currently `docs/reviews/mdl-dag-sprint.md`)
- Epics: [Epic 1](docs/epic-1-hipporag-daemon.md), [Epic 2](docs/epic-2-fastmcp-http.md), [Epic 3](docs/epic-3-performance.md) *(where older epic text references pre-cutover OpenIE/SHACL/PPR details, §§1–5 above take precedence)*.
