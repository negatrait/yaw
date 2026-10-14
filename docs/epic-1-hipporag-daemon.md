# Epic 1: Native `yaw.db` Engine & CocoIndex Daemon

Owner: [FOUNDATIONAL](../FOUNDATIONAL.md) | Results: [evaluation](evaluation.md) | Next: [Epic 2](epic-2-fastmcp-http.md)

## Hard Gates

- **Zero Cross-Domain Edges (`E1-S1`):** Lock the containment tree and AST boundary test (`tests/core/test_architecture.py`) **before** writing engine code. Sibling packages never import each other (`0` horizontal edges).
- **Lean Architecture (No Code-Golfing, No LOC Quotas):** Eliminate dead abstractions and multi-file synchronization sprawl (retire the 7-file Parquet/Pickle/JSON store and multi-provider wrappers). Write direct, readable, strongly-typed code without line-count obsession or code-golfing.
- **Dev Gate (`E1-S4`):** Native `yaw.daemon.graph` engine matches or beats upstream `hipporag` on **Tier 1** (`sample`, `2wikimultihopqa`, `musique` 10-query slices) on the 3-model local setup.
- **Epic Gate (`E1-S6`):** Native `yaw.daemon.graph` engine matches or beats upstream `hipporag` baseline on our exact 3-model local setup on **Tier 2** full-corpus benchmarks (`Recall@2`, `Recall@5`, `QA EM`, `QA F1`), with published paper figures recorded in [evaluation](evaluation.md) as reference.

---

## 1. Containment Tree & Import Law

```text
yaw/
  core/                    types.py, config.py, logger.py          (imports: stdlib only)
  daemon/                  runtime.py, cli.py, mcp_server.py       (imports: daemon.*, core)
    ingestion/             corpus.py                               (imports: core, cocoindex)
    graph/                 engine.py, ppr.py                       (imports: graph.*, core, scipy)
      storage/             db.py, hydration.py                     (imports: core, sqlite3, sqlite_vec)
      inference/           models.py, prompts.py, embedding.py, chat.py (imports: core, openai)
  __main__.py                                                      (imports: daemon.cli)
```

- **Containment Law (`tests/core/test_architecture.py`):**
  1. Every `__init__.py` is empty (no parent side-effects when importing child modules).
  2. A module in package `P` imports **only** `P.*` (its own descendants) and `yaw.core.*`.
  3. **Zero horizontal edges:** `ingestion` $\not\leftrightarrow$ `graph`, and `storage` $\not\leftrightarrow$ `inference` $\not\leftrightarrow$ `ppr`. Only a parent orchestrator (`daemon/runtime.py` or `graph/engine.py`) bridges its direct children.

---

## 2. Sprint-by-Sprint Implementation Plan (One Node at a Time, Zero Spaghetti)

Execute strictly in order `E1-S1` $\to$ `E1-S6`. Never start Sprint $k+1$ until Sprint $k$'s exit gate passes.

---

### Sprint 1 (`E1-S1`): `yaw/core/` & Containment Tree Lock
- **Files:** `yaw/core/{types.py, logger.py}`, empty `__init__.py` tree (`yaw/{core,daemon/{ingestion,graph/{storage,inference}}}/__init__.py`), `tests/core/test_architecture.py`.
- **Allowed imports:** `stdlib` only (`dataclasses`, `logging`, `pathlib`, `typing`, `collections`).
- **Copy/Adapt From:**
  - Move current [`yaw/logger.py`](../yaw/logger.py) (`FIFOLineRotatingHandler(max_lines=500)`) $\to$ `yaw/core/logger.py`.
  - Move current [`yaw/{models,embedding,chat}.py`](../yaw/) $\to$ `yaw/daemon/graph/inference/` (after running Tier 2 baseline completes) and update imports in `benchmark.py` & `tests/`.
  - Adapt data structures from `git show d1bc65e:src/yaw/indexer/pipeline.py` (`MarkdownChunk`) and `hipporag/utils/misc_utils.py` (`Chunk`, `QuerySolution`, `NerRawOutput`, `TripleRawOutput`) into 6 frozen `@dataclass(slots=True, frozen=True)` types in `yaw/core/types.py`:
    - `ChunkPointer(chunk_id, rel_posix_path, start_line, end_line, title, content_hash)`
    - `HydratedPassage(pointer, text, score, stale_on_disk=False)`
    - `OpenIERecord(chunk_id, entities: tuple[str, ...], triples: tuple[tuple[str, str, str], ...])`
    - `RetrievalResult(question, passages: tuple[HydratedPassage, ...], answer: str | None = None)`
    - `SyncDelta(added_chunks: tuple[ tuple[ChunkPointer, str], ...], deleted_chunk_ids: tuple[str, ...])`
- **Drop (Anti-Spaghetti):** Drop `d1bc65e:src/yaw/utils.py`, global mutable singletons, and upstream's mutable `QuerySolution.to_dict()` / numpy array fields.
- **Exit Gate (`D1-A1`):** `uv run --locked python -m unittest discover -s tests -v` passes with `tests/core/test_architecture.py` verifying `0` cross-domain imports and empty `__init__.py`s across all `yaw/**/*.py`.

---

### Sprint 2 (`E1-S2`): `yaw/daemon/graph/inference/prompts.py` — Owned DSPy/GEPA Prompts & Parsers
- **Files:** `yaw/daemon/graph/inference/prompts.py`, `tests/daemon/graph/inference/test_prompts.py`.
- **Allowed imports:** `yaw.core.*`, `json`, `re`, `ast`, `difflib`. (No dynamic loaders, no provider wrappers).
- **Copy/Adapt From (Upstream `hipporag`):**
  - **Prompts:** Copy exact prompt strings & one-shot demos from `.venv/Lib/site-packages/hipporag/prompts/templates/{ner.py, triple_extraction.py, rag_qa_musique.py}` and `prompts/filter_default_prompt.py` (`best_dspy_prompt`) into plain module constants (`NER_MESSAGES`, `TRIPLE_MESSAGES`, `FACT_FILTER_MESSAGES`, `RAG_QA_MESSAGES`).
  - **Stateless Parsers:**
    - From `hipporag/information_extraction/openie_openai.py:32-72`: port `_extract_ner_from_response`, `_extract_json_list_field`, and `filter_invalid_triples` (preserving exact `[s, p, o]` length-3 string coercion).
    - From `hipporag/rerank.py:35-120`: port `DSPyFilter.parse_filter` and `difflib.get_close_matches(..., n=1, cutoff=0.0)` candidate fact matcher as a pure function `filter_reranked_facts(candidate_facts, llm_response) -> list[int]`.
    - From `hipporag/utils/misc_utils.py:42-65`: port `compute_mdhash_id(content, prefix)` and `text_processing(text)` (lowercase alphanumeric + whitespace normalization).
- **Drop (Anti-Spaghetti):** Drop `hipporag/prompts/prompt_template_manager.py` (dynamic `importlib` file scanning, `string.Template` AST introspection, role-mapping classes) and `d1bc65e:src/yaw/nlp/*` (spaCy, GLiNER, Ollama wrappers).
- **Exit Gate (`D1-A2`):** `tests/daemon/graph/inference/test_prompts.py` proves 100% byte-for-byte rendered message parity and parser output parity against upstream `PromptTemplateManager`, `OpenIE`, and `DSPyFilter`.

---

### Sprint 3 (`E1-S3`): `yaw/daemon/graph/storage/{db.py, hydration.py}` — Single `yaw.db` & Line Hydration
- **Files:** `yaw/daemon/graph/storage/{db.py, hydration.py}`, `tests/daemon/graph/storage/{test_db.py, test_hydration.py}`.
- **Allowed imports:** `yaw.core.*`, `sqlite3`, `sqlite_vec`, `numpy`, `hashlib`, `itertools`, `pathlib`.
- **Copy/Adapt From:**
  - **SQLite WAL & `sqlite-vec` setup (`db.py`):** Adapt `apply_pragmas` (`WAL`, `busy_timeout=5000`, `synchronous=NORMAL`, `foreign_keys=ON`) and `load_vec_extension` (`sqlite_vec.load(conn)`) from `git show d1bc65e:src/yaw/storage/db.py:55-68`.
  - **Single Fixed Schema (`db.py` — Zero Migrations):**
    ```sql
    CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    CREATE TABLE llm_cache (cache_key TEXT PRIMARY KEY, response TEXT NOT NULL, meta_json TEXT NOT NULL);
    CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, rel_posix_path TEXT NOT NULL, start_line INT NOT NULL, end_line INT NOT NULL, title TEXT NOT NULL, content_hash TEXT NOT NULL);
    CREATE TABLE openie_chunks (chunk_id TEXT PRIMARY KEY, entities_json TEXT NOT NULL, triples_json TEXT NOT NULL);
    CREATE TABLE entities (entity_id TEXT PRIMARY KEY, text TEXT NOT NULL);
    CREATE TABLE facts (fact_id TEXT PRIMARY KEY, subj_id TEXT NOT NULL, pred TEXT NOT NULL, obj_id TEXT NOT NULL, text TEXT NOT NULL);
    CREATE TABLE edges (src_id TEXT NOT NULL, dst_id TEXT NOT NULL, kind TEXT NOT NULL, weight REAL NOT NULL, PRIMARY KEY (src_id, dst_id, kind));
    CREATE TABLE edge_sources (src_id TEXT NOT NULL, dst_id TEXT NOT NULL, kind TEXT NOT NULL, chunk_id TEXT NOT NULL, PRIMARY KEY (src_id, dst_id, kind, chunk_id));
    CREATE VIRTUAL TABLE chunk_fts USING fts5(chunk_id UNINDEXED, title, body, content='', contentless_delete=1);
    -- + sqlite-vec float32 tables: chunk_vec(chunk_id, embedding), entity_vec(entity_id, embedding), fact_vec(fact_id, embedding)
    ```
  - **1-Indexed Line Hydration (`hydration.py`):** Adapt `hydrate_disk_slice` from `git show d1bc65e:src/yaw/retrieval/retriever.py:30-75` (`itertools.islice(f, start_line - 1, end_line)` + `resolved.is_relative_to(corpus_root)` path guard), adding SHA-256 `content_hash` verification (`stale_on_disk = (sha256(text) != pointer.content_hash)`).
- **Drop (Anti-Spaghetti):** Drop all 7 upstream files (`EmbeddingStore` Parquet rewrite loops, `openie_results_*.json`, `chunk_metadata.json`, `graph.pickle`) AND drop `d1bc65e:src/yaw/storage/{rdf.py, schema.py}` (`unified_edges` SQL views, `compiled_violations`, `aliases`, `canon_verdicts`).
- **Exit Gate (`D1-A3`):** Unmocked `test_db.py` & `test_hydration.py` pass top-$k$ `sqlite-vec` KNN, contentless `FTS5` BM25 (`1 / rank` L1-normalized), atomic `chunk_id` deletion cascading through `edge_sources` $\to$ `edges`/`entities`/`facts`, and `stale_on_disk` detection.

---

### Sprint 4 (`E1-S4`): `yaw/daemon/graph/{ppr.py, engine.py}` — Sparse PPR & Tier 1 Parity
- **Files:** `yaw/daemon/graph/{ppr.py, engine.py}`, `tests/daemon/graph/{test_ppr.py, test_engine.py}`.
- **Allowed imports:** `ppr.py` imports `yaw.core.*`, `numpy`, `scipy.sparse` only; `engine.py` imports `yaw.daemon.graph.{storage,inference,ppr}` and `yaw.core.*`. (Zero external graph frameworks).
- **Copy/Adapt From:**
  - **Sparse CSR Power Iteration (`ppr.py`):** Adapt `power_iteration` from `git show d1bc65e:src/yaw/retrieval/retriever.py:280-315`:
    - Build symmetric row-stochastic CSR matrix $P = D^{-1} A$ from `edges(src_id, dst_id, weight)` plus `min_max_normalize` (`hipporag/utils/misc_utils.py:132`).
    - Run $p^{(t+1)} = (1 - \alpha) (P^T p^{(t)} + m_{\text{dangling}} p_0) + \alpha p_0$ with HippoRAG damping $\alpha = 0.5$ (`damping=0.5`), converging in $<3\text{ ms}$ on CSR matrices.
  - **HippoRAG 2 Graph Construction (`engine.py`):** Port directly from `hipporag/HippoRAG.py`:
    - `add_fact_edges` (`HippoRAG.py:1303-1340`): entity $\leftrightarrow$ entity co-occurrence counts across triples.
    - `add_passage_edges` (`HippoRAG.py:1342-1372`): chunk $\leftrightarrow$ entity mention edges (`weight = 1.0`).
    - `add_synonymy_edges` (`HippoRAG.py:1435-1510`): entity $\leftrightarrow$ entity KNN via `entity_vec` where cosine similarity $\ge 0.8$.
  - **HippoRAG 2 Retrieval & Multi-Step QA (`engine.py`):** Port directly from `hipporag/HippoRAG.py`:
    - Fact linking & Recognition Memory (`HippoRAG.py:2125-2175`): query `fact_vec` top-$k$ (`linking_top_k=5`) $\to$ filter via `inference/prompts.py:filter_reranked_facts`.
    - PPR Seeding (`HippoRAG.py:2008-2120`): entity weight $w(e) = \text{mean\_fact\_score}(e) / \max(1, \text{doc\_freq}(e))$; passage weight $w(c) = 0.05 \cdot (\widehat{s}_{\text{vec}}(c) + \widehat{s}_{\text{fts}}(c))$ $\to$ `ppr.run_ppr` $\to$ rank chunk nodes $\to$ hydrate **only** top-$k$ chunks via `storage/hydration.py` (killing the `HippoRAG.py:2098` full-corpus `get_row` loop).
    - QA reader (`HippoRAG.py:804-860, 1117-1165`): format top-$k$ hydrated passages with `Wikipedia Title: {title}\n{text}` and run `inference/chat.py`.
- **Drop (Anti-Spaghetti):** Drop `python-igraph` (`self.graph`, vertex deletion index-shifting bugs, `graph.pickle`) and drop `d1bc65e`'s `compute_upward_conductance`, `in_degrees**0.5` hub dampening, and `InMemoryLookupKB`.
- **Exit Gate (`D1-A4` — Tier 1 Dev Parity Gate):** `uv run --locked python benchmark.py --engine yaw --dataset {sample,2wikimultihopqa,musique} --max-queries 10` matches or beats upstream `hipporag` on `Recall@2`, `Recall@5`, `QA EM`, and `QA F1`.

---

### Sprint 5 (`E1-S5`): `yaw/daemon/ingestion/corpus.py` — CocoIndex Incremental Line-Pointer Pipeline
- **Files:** `yaw/daemon/ingestion/corpus.py`, `tests/daemon/ingestion/test_corpus.py`.
- **Allowed imports:** `yaw.core.*`, `cocoindex`, `hashlib`, `pathlib`. (Zero imports from `graph/`).
- **Copy/Adapt From:**
  - Adapt `chunk_markdown` from `git show d1bc65e:src/yaw/indexer/pipeline.py:122-167`:
    - Walk `corpus.root` using `cocoindex>=1.0.0` (`localfs.walk_`, `PatternFilePathMatcher` for `*.md`, `*.txt`, `RecursiveSplitter`).
    - Compute 1-indexed inclusive `start_line = content[:c.start].count("\n") + 1`, `end_line = start_line + c.text.count("\n")`, and `content_hash = sha256(c.text)`.
    - Diff against committed `chunk_id` set to return a single coalesced `SyncDelta(added_chunks, deleted_chunk_ids)`.
- **Drop (Anti-Spaghetti):** Never call LLMs, spaCy, or SQLite writers inside the CocoIndex flow (`d1bc65e:pipeline.py` had 14 functions calling Ollama/spaCy inside CocoIndex). `corpus.py` is a pure file-to-`SyncDelta` producer.
- **Exit Gate (`D1-A5`):** Unmocked `tests/daemon/ingestion/test_corpus.py` (adapted from `d1bc65e:tests/test_cocoindexer.py`) passes file add, edit, rename, delete, and memoized zero-op re-scans.

---

### Sprint 6 (`E1-S6`): `yaw/daemon/{runtime.py, cli.py}` & Tier 2 Full-Corpus Parity Gate
- **Files:** `yaw/core/config.py`, `yaw/daemon/{runtime.py, cli.py}`, `yaw/__main__.py`, `yaw.toml`, `tests/daemon/test_runtime.py`.
- **Allowed imports:** `runtime.py` & `cli.py` import `yaw.daemon.{ingestion,graph}`, `yaw.core.*`, `filelock`, `tomllib`, `argparse`, `queue`, `threading`.
- **Copy/Adapt From:**
  - Adapt `DaemonLifecycle` and graceful signal handling from `git show d1bc65e:src/yaw/server/daemon.py:260-340` (using `filelock.FileLock("yaw.db.lock")` instead of raw `msvcrt`/PID-alive checks).
  - Adapt `argparse` subcommands from `git show d1bc65e:src/yaw/cli.py:42-180` into `python -m yaw {run|start|status|sync|stop|rebuild} --config yaw.toml`.
  - Wire single-owner 32-slot bounded FIFO worker thread in `runtime.py`: drains `ingestion.corpus.poll()` $\to$ `graph.engine.apply_delta(delta)` and serves `retrieve`/`rag_qa` requests, with non-blocking cached `status()`/`stats()`.
- **Exit Gate (`D1-A6` — Epic 1 Acceptance Gate):**
  1. `tests/daemon/test_runtime.py` passes lock contention, 32-slot `BUSY` overflow, Stage 1 crash-resume, and inference outage recovery.
  2. Native `yaw.daemon.graph` engine matches or beats our upstream `hipporag` 3-model baseline on **Tier 2** full-corpus benchmarks in [evaluation](evaluation.md).

---

## 3. Checklist Summary

- [x] **D1-A0 (`D1-01`):** Embedding repeatability (`cosine >= 0.999`), $T=0.2$ floors, GBNF, 500-line FIFO logger, paper datasets downloaded.
- [x] **D1-A1 (`E1-S1`):** `yaw/core/` + containment tree locked; `test_architecture.py` enforces `0` horizontal/cross-domain imports.
- [ ] **D1-A2 (`E1-S2`):** `inference/prompts.py` ports HippoRAG 2 NER, Triple, Recognition Memory, and QA prompts/parsers with 1:1 unit parity.
- [ ] **D1-A3 (`E1-S3`):** `storage/{db,hydration}.py` implements single-schema `yaw.db` (`sqlite-vec`, contentless `FTS5`, provenance edges) and 1-indexed `itertools.islice` line hydration.
- [ ] **D1-A4 (`E1-S4`):** `graph/{ppr,engine}.py` + `storage/*` achieves **Tier 1 benchmark parity** vs. upstream `hipporag` on the 3-model local setup.
- [ ] **D1-A5 (`E1-S5`):** `ingestion/corpus.py` (`cocoindex>=1.0.0`) emits pure coalesced line-pointer `SyncDelta`s across add/edit/rename/delete.
- [ ] **D1-A6 (`E1-S6`):** `daemon/{runtime,cli}.py` passes lifecycle/lock/recovery tests AND native engine matches or beats upstream `hipporag` 3-model baseline on **Tier 2** full corpus.
