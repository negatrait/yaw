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
      inference/           models.py, prompts.py, embedding.py, chat.py (imports: core, dspy)
  __main__.py                                                      (imports: daemon.cli)
```

- **Containment Law (`tests/core/test_architecture.py`):**
  1. **Empty Intermediate Packages:** Intermediate parent package initializers (`yaw/`, `yaw/daemon/`, `yaw/daemon/graph/`) contain zero executable statements and trigger zero parent-package import side-effects.
  2. **Constricted Leaf Facades:** Leaf domain packages (`yaw.core`, `yaw.daemon.graph.inference`, `yaw.daemon.graph.storage`, `yaw.daemon.ingestion`) provide constricted public API facades (`_PUBLIC_API` + lazy `__getattr__` + `__dir__`) allowing leaf mutation into single-responsibility modules while keeping a stable public API contract.
  3. **Strict Domain Containment:** A module in package `P` imports **only** `P.*` (its own descendants) and `yaw.core.*`. **Zero horizontal edges:** `ingestion` $\not\leftrightarrow$ `graph`, and `storage` $\not\leftrightarrow$ `inference` $\not\leftrightarrow$ `ppr`. Only a parent orchestrator (`daemon/runtime.py` or `graph/engine.py`) bridges its direct children.
  4. **Zero Protected (`_`) or Private (`__`) Cross-Module Imports or Facade Exports:** Any symbol starting with a leading underscore is strictly module-internal. No `name.startswith("_")` may ever appear in a leaf `_PUBLIC_API` facade, in cross-module imports (`from .mod import _helper`), or in cross-module attribute accesses.
  5. **Zero `hipporag` runtime imports inside `yaw/`:** Pinned `hipporag` is imported only in `benchmark.py` and offline test suites as the reference baseline.

---

## 2. Sprint-by-Sprint Implementation Plan (One Node at a Time, Zero Spaghetti)

Execute strictly in order `E1-S1` $\to$ `E1-S6`. Never start Sprint $k+1$ until Sprint $k$'s exit gate passes.

---

### Sprint 1 (`E1-S1`): `yaw/core/` & Containment Tree Lock
- **Files:** `yaw/core/{types.py, logger.py}`, empty `__init__.py` tree (`yaw/{core,daemon/{ingestion,graph/{storage,inference}}}/__init__.py`), `tests/core/test_architecture.py`.
- **Allowed imports:** `stdlib` only (`dataclasses`, `logging`, `pathlib`, `typing`, `collections`).
- **Copy/Adapt From:**
  - Move current [`yaw/logger.py`](../yaw/logger.py) (`FIFOLineRotatingHandler(max_lines=500)`) $\to$ `yaw/core/logger.py`.
  - Move current [`yaw/{models,embedding,chat}.py`](../yaw/) $\to$ `yaw/daemon/graph/inference/` and update imports in `benchmark.py` & `tests/`.
  - Adapt data structures from `git show d1bc65e:src/yaw/indexer/pipeline.py` (`MarkdownChunk`) and `hipporag/utils/misc_utils.py` (`Chunk`, `QuerySolution`, `NerRawOutput`, `TripleRawOutput`) into frozen `@dataclass(slots=True, frozen=True)` types in `yaw/core/types.py`:
    - `ChunkPointer(chunk_id, rel_posix_path, start_line, end_line, title, content_hash)`
    - `HydratedPassage(pointer, text, score, stale_on_disk=False)`
    - `OpenIERecord(chunk_id, entities: tuple[str, ...], triples: tuple[tuple[str, str, str], ...])`
    - `RetrievalResult(question, passages: tuple[HydratedPassage, ...], answer: str | None = None)`
    - `SyncDelta(added_chunks: tuple[tuple[ChunkPointer, str], ...], deleted_chunk_ids: tuple[str, ...])`
- **Drop (Anti-Spaghetti):** Drop `d1bc65e:src/yaw/utils.py`, global mutable singletons, and upstream's mutable `QuerySolution.to_dict()` / numpy array fields.
- **Exit Gate (`D1-A1`):** `uv run --locked python -m unittest discover -s tests -v` passes with `tests/core/test_architecture.py` verifying `0` cross-domain imports and empty `__init__.py`s across all `yaw/**/*.py`.

---

### Sprint 2 (`E1-S2`): `yaw/daemon/graph/inference/` — Native DSPy (`dspy.LM`, `dspy.Embedder`, `dspy.Signature`, `dspy.RLM`, `dspy.GEPA`)
- **Files:** `pyproject.toml`, `uv.lock`, `yaw/daemon/graph/inference/{models.py, embedding.py, chat.py, prompts.py}`, `tests/daemon/graph/inference/test_prompts.py`.
- **Allowed imports:** `yaw.core.*`, `dspy`, `pydantic`, `numpy`, `difflib`, `hashlib`, `json`, `re`, `ast`. (Zero `hipporag` imports in `yaw/**`).
- **Design & Capabilities:**
  - **`embedding.py` (`dspy.Embedder`):**
    - Wrap `dspy.Embedder(model="openai/...", api_base=..., api_key=..., batch_size=..., caching=True, encoding_format="float")` for Lemonade (`embeddinggemma-300m-GGUF`) or `ollama/...` for Ollama.
    - Apply `distinct-titled` passage prefix (`title: {title} | text: {body}`) and query prefix (`task: search result | query: {query}`) + L2 normalization (`np.float32`).
  - **`chat.py` (`dspy.LM` & `create_dspy_rlm_pair`):**
    - Drive `dspy.LM` against Lemonade (`openai/...`) or Ollama (`ollama_chat/...`) with recipe-aware GBNF capability flags (`_yaw_supports_gbnf`):
      - **GBNF Worker & Direct Extraction/Rerank (`Qwen3-4B-Instruct-2507-GGUF`):** Vulkan GPU `llamacpp`, `8192–16384` context window, `_yaw_supports_gbnf=True`, `temp=0.2`, `freq_penalty=0.5`.
      - **RLM Orchestrator & QA Reader (`Mistral-7B-Instruct-v0.3-Hybrid`):** Ryzen AI NPU/OGA `ryzenai-llm`, `32786` context window, `_yaw_supports_gbnf=False` (GBNF is not supported on Ryzen AI NPU/OGA).
      - `create_dspy_rlm_pair()` constructs `(orchestrator_lm, sub_lm)` pairing `Mistral-7B` (NPU orchestrator) with `Qwen3-4B` (GPU GBNF worker).
  - **`prompts.py` (`dspy.Signature`s, `dspy.JSONAdapter`, Heterogeneous `dspy.RLM` + GBNF, `compute_sizecoding_pareto_score`):**
    - Define 4 typed `dspy.Signature`s seeded with HippoRAG 2 instructions and `dspy.Example` demos, formatted via **`dspy.JSONAdapter()`** (pure JSON, zero `[[ ## ... ## ]]` output headers):
      1. `ExtractEntities(dspy.Signature)`: `passage: str -> named_entities: list[str]`
      2. `ExtractTriples(dspy.Signature)`: `passage: str, named_entities: list[str] -> triples: list[list[str]]`
      3. `FilterFacts(dspy.Signature)`: `question: str, fact_before_filter: str -> fact: list[list[str]]` (with `difflib.get_close_matches(..., n=1, cutoff=0.0)` index resolver)
      4. `AnswerQuestion(dspy.Signature)`: `context: str, question: str -> answer: str`
    - Provide composable `HippoRAGProgram(dspy.Module)` supporting both `execution_mode="direct"` and **`execution_mode="rlm"` across all 4 stages**:
      - In `execution_mode="rlm"`, `ner_rlm`, `triples_rlm`, `fact_filter_rlm`, and `qa_rlm` run `Mistral-7B` on Ryzen AI NPU as the outer `dspy.RLM` orchestrator inside `dspy.LocalInterpreter` (zero GBNF), delegating inner `qwen_extract_entities`, `qwen_extract_triples`, `qwen_filter_facts`, `qwen_answer_cot`, and `llm_query` calls to `Qwen3-4B` on `llama.cpp` GPU with pure-JSON GBNF grammars (`build_dspy_ner_grammar`, `build_dspy_triples_grammar`, `build_dspy_rerank_grammar`, `build_dspy_qa_cot_grammar`).
      - Provides `compute_sizecoding_pareto_score` ($\text{pareto} = \text{Recall@5} / \text{tokens}$ with hard Tier 1 baseline floor and multi-objective `objective_scores` for `dspy.GEPA`) and `.save(path)` / `.load(path)`.
    - Keep stateless helpers `compute_mdhash_id(content, prefix)`, `text_processing(text)`, `filter_invalid_triples(triples)`, and `filter_reranked_facts(candidate_facts, matched_facts)`.
- **Drop (Anti-Spaghetti):** Drop `hipporag.llm.openai_gpt.CacheOpenAI`, `hipporag.embedding_model.base.BaseEmbeddingModel`, `hipporag/prompts/prompt_template_manager.py`, and `d1bc65e:src/yaw/nlp/*`.
- **Exit Gate (`D1-A2`):** `tests/daemon/graph/inference/` passes unmocked tests for `dspy.Embedder`, `dspy.LM`, `create_dspy_rlm_pair`, all 4 `dspy.Signature` modules in both direct (`JSONAdapter` + GBNF) and heterogeneous `execution_mode="rlm"` (`Mistral` NPU orchestrator + `Qwen` GPU GBNF worker inside `LocalInterpreter`), `compute_sizecoding_pareto_score`, `.save()`/`.load()` state round-tripping, and `test_architecture.py` enforces `0` `hipporag` imports inside `yaw/`.

---

### Sprint 3 (`E1-S3`): `yaw/daemon/graph/storage/{db.py, hydration.py}` — Single `yaw.db` & Line Hydration
- **Files:** `yaw/daemon/graph/storage/{db.py, hydration.py}`, `tests/daemon/graph/storage/{test_db.py, test_hydration.py}`.
- **Allowed imports:** `yaw.core.*`, `sqlite3`, `sqlite_vec`, `numpy`, `hashlib`, `itertools`, `pathlib`.
- **Copy/Adapt From:**
  - **SQLite WAL & `sqlite-vec` setup (`db.py`):** Adapt `apply_pragmas` (`WAL`, `busy_timeout=5000`, `synchronous=NORMAL`, `foreign_keys=ON`) and `load_vec_extension` (`sqlite_vec.load(conn)`) from `git show d1bc65e:src/yaw/storage/db.py:55-68`.
  - **Single Fixed Schema (`db.py` — Zero Migrations):**
    ```sql
    CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, rel_posix_path TEXT NOT NULL, start_line INT NOT NULL, end_line INT NOT NULL, title TEXT NOT NULL);
    CREATE TABLE openie_chunks (chunk_id TEXT PRIMARY KEY, entities_json TEXT NOT NULL, triples_json TEXT NOT NULL);
    CREATE TABLE entities (entity_id TEXT PRIMARY KEY, text TEXT NOT NULL);
    CREATE TABLE facts (fact_id TEXT PRIMARY KEY, subj_id TEXT NOT NULL, pred TEXT NOT NULL, obj_id TEXT NOT NULL, text TEXT NOT NULL);
    CREATE TABLE edges (src_id TEXT NOT NULL, dst_id TEXT NOT NULL, kind TEXT NOT NULL, weight REAL NOT NULL, PRIMARY KEY (src_id, dst_id, kind));
    CREATE TABLE edge_sources (src_id TEXT NOT NULL, dst_id TEXT NOT NULL, kind TEXT NOT NULL, chunk_id TEXT NOT NULL, PRIMARY KEY (src_id, dst_id, kind, chunk_id));
    CREATE VIRTUAL TABLE chunk_fts USING fts5(chunk_id UNINDEXED, title, body, content='', contentless_delete=1);
    -- + sqlite-vec float32 tables: chunk_vec(chunk_id, embedding), entity_vec(entity_id, embedding), fact_vec(fact_id, embedding)
    ```
  - **1-Indexed Line Hydration (`hydration.py`):** Adapt `hydrate_disk_slice` from `git show d1bc65e:src/yaw/retrieval/retriever.py:30-75` (`itertools.islice(f, start_line - 1, end_line)` + `resolved.is_relative_to(corpus_root)` path guard), with missing file/span `stale_on_disk` detection, plus a callable `hydrate_chunk(chunk_id)` tool helper for `dspy.RLM` on-demand passage inspection.
- **Drop (Anti-Spaghetti):** Drop all 7 upstream files (`EmbeddingStore` Parquet rewrite loops, `openie_results_*.json`, `chunk_metadata.json`, `graph.pickle`) AND drop `d1bc65e:src/yaw/storage/{rdf.py, schema.py}` (`unified_edges` SQL views, `compiled_violations`, `aliases`, `canon_verdicts`).
- **Exit Gate (`D1-A3`):** Unmocked `test_db.py` & `test_hydration.py` pass top-$k$ `sqlite-vec` KNN, contentless `FTS5` BM25 (`1 / rank` L1-normalized), atomic `chunk_id` deletion cascading through `edge_sources` $\to$ `edges`/`entities`/`facts`, and `stale_on_disk` detection.

---

### Sprint 4 (`E1-S4`): `yaw/daemon/graph/{ppr.py, engine.py}` — Sparse PPR, DSPy Graph Engine & Tier 1 Parity
- **Files:** `yaw/daemon/graph/{ppr.py, engine.py}`, `tests/daemon/graph/{test_ppr.py, test_engine.py}`.
- **Allowed imports:** `ppr.py` imports `yaw.core.*`, `numpy`, `scipy.sparse` only; `engine.py` imports `yaw.daemon.graph.{storage,inference,ppr}` and `yaw.core.*`. (Zero external graph frameworks).
- **Copy/Adapt From:**
  - **Sparse CSR Power Iteration (`ppr.py`):** Adapt `power_iteration` from `git show d1bc65e:src/yaw/retrieval/retriever.py:280-315`:
    - Build symmetric row-stochastic CSR matrix $P = D^{-1} A$ from `edges(src_id, dst_id, weight)` plus `min_max_normalize` (`hipporag/utils/misc_utils.py:132`).
    - Run $p^{(t+1)} = (1 - \alpha) (P^T p^{(t)} + m_{\text{dangling}} p_0) + \alpha p_0$ with HippoRAG damping $\alpha = 0.5$ (`damping=0.5`), converging in $<3\text{ ms}$ on CSR matrices.
  - **HippoRAG 2 Graph Construction (`engine.py`):** Port from `hipporag/HippoRAG.py` onto `yaw.db` + `inference/prompts.py` (`HippoRAGProgram`):
    - Stage 1 OpenIE via `HippoRAGProgram.extract_openie(passage)` cached in `openie_chunks`.
    - `add_fact_edges` (`HippoRAG.py:1303-1340`): entity $\leftrightarrow$ entity co-occurrence counts across triples.
    - `add_passage_edges` (`HippoRAG.py:1342-1372`): chunk $\leftrightarrow$ entity mention edges (`weight = 1.0`).
    - `add_synonymy_edges` (`HippoRAG.py:1435-1510`): entity $\leftrightarrow$ entity KNN via `entity_vec` where cosine similarity $\ge 0.8$.
  - **HippoRAG 2 Retrieval & DSPy Multi-Step QA (`engine.py`):**
    - Fact linking & Recognition Memory (`HippoRAG.py:2125-2175`): query `fact_vec` top-$k$ (`linking_top_k=5`) $\to$ filter via `HippoRAGProgram.filter_facts`.
    - PPR Seeding (`HippoRAG.py:2008-2120`): entity weight $w(e) = \text{mean\_fact\_score}(e) / \max(1, \text{doc\_freq}(e))$; passage weight $w(c) = 0.05 \cdot (\widehat{s}_{\text{vec}}(c) + \widehat{s}_{\text{fts}}(c))$ $\to$ `ppr.run_ppr` $\to$ rank chunk nodes $\to$ hydrate **only** top-$k$ chunks via `storage/hydration.py`.
    - QA reader: run `HippoRAGProgram.answer_question` (selectable `ChainOfThought` or `RLM` with `sub_lm` and hydration tool).
  - **End-to-End DSPy Trace Hook (`engine.py`) & Disjoint Benchmark Split (`benchmark.py`):**
    - Wire `engine.retrieve_and_answer(question)` through `HippoRAGProgram` so a single `dspy.Module.forward(question)` trace captures `FilterFacts` $\to$ `<3 ms` CSR PPR + line hydration $\to$ `AnswerQuestion` (`CoT` or `RLM`).
    - Add `--engine {hipporag, yaw}`, `--query-offset <M>`, `--evolve {none, miprov2, gepa, hybrid}`, `--train-queries <K>`, and `--compiled-program <path>` to `benchmark.py` to support strict disjoint learning (`[0:K]`) vs. holdout (`[K:N]`) evaluation.
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

### Sprint 6 (`E1-S6`): `yaw/daemon/{runtime.py, cli.py}`, Two-Phase `MIPROv2` + `GEPA` Evolution & Tier 2 Holdout Gate
- **Files:** `yaw/core/config.py`, `yaw/daemon/{runtime.py, cli.py}`, `yaw/__main__.py`, `yaw.toml`, `tests/daemon/test_runtime.py`.
- **Allowed imports:** `runtime.py` & `cli.py` import `yaw.daemon.{ingestion,graph}`, `yaw.core.*`, `filelock`, `tomllib`, `argparse`, `queue`, `threading`.
- **Copy/Adapt From:**
  - Adapt `DaemonLifecycle` and graceful signal handling from `git show d1bc65e:src/yaw/server/daemon.py:260-340` (using `filelock.FileLock("yaw.db.lock")` instead of raw `msvcrt`/PID-alive checks).
  - Adapt CLI commands into hierarchical grammar `python -m yaw <domain> <action> [--flags]` (`yaw daemon {start|stop|status|run}`, `yaw graph {build|rebuild}`, `yaw dspy {optimize|load-program}`, `yaw config {get|set}`).
  - Wire single-owner 32-slot bounded FIFO worker thread in `runtime.py`: drains `ingestion.corpus.poll()` $\to$ `graph.engine.apply_delta(delta)` and serves `retrieve`/`rag_qa` requests, with non-blocking cached `status()`/`stats()`, loading `compiled_program_path` from `yaw.toml` when present.
  - **Zero-Lock-In `MIPROv2` + `GEPA` Evolution (`engine.evolve_prompts` & `python -m yaw dspy optimize` / `benchmark.py --evolve`):**
    1. **Anti-Overfitting Rule:** Never run prompt evolution on full Tier 2 benchmark corpora (`musique`, `2wikimultihopqa`, `hotpotqa` full sets) to prevent benchmark lock-in.
    2. **Composite Fitness Objective (`compute_sizecoding_pareto_score`):**
       - **Numerator ("Good Things"):** Stage-scoped product across the full quality curve ($\text{Recall@1} \times \text{Recall@2} \times \text{Recall@5} \times \text{Recall@10} \times \text{Recall@20} \times \text{EM} \times \text{F1}$ for end-to-end/QA stages; $\text{Recall@1..20}$ for retrieval/indexing stages, with smoothed floor protection against single-metric zero collapse and multi-objective `objective_scores` dict keys for `dspy.GEPA`).
       - **Denominator ("Bad Things"):** $\text{tokens\_consumed}$.
    3. **Full Optimizable Surface & Heterogeneous Reflection RLM:** Expose all pipeline instructions and bounds to `MIPROv2`/`GEPA` (`EmbeddingGemma` query/passage instructions, Extraction RLM, `FilterFacts`/`ExtractQueryIntent`, QA RLM, and parametric GBNF grammar bounds). Drive `dspy.GEPA(reflection_lm=...)` via the local heterogeneous **Mistral (Ryzen AI NPU) + Qwen (Vulkan GPU) RLM pair** (`create_dspy_rlm_pair`), capturing its own DSPy telemetry trails so the reflection module itself can be traced and optimized.
    4. **Evolve on Tier 1 Calibration Slices or External OOD Corpora:** Run `dspy.MIPROv2` + `dspy.GEPA` (`enable_gbnf=True` for `llama.cpp` GBNF grammar enforcement via `config={"extra_body": {"grammar": ...}}`) strictly on the small **Tier 1 benchmark sets** (`--max-queries 10`) and/or **external out-of-distribution corpora** (`onyx-dot-app/EnterpriseRAG-Bench`, `microsoft/graphrag-benchmarking-datasets`) to produce one universal, generalizable `compiled_program.json`.
    5. **Run Unseen Holdout Benchmarks Zero-Shot:** Evaluate `--engine yaw --compiled-program compiled_program.json --query-offset 10` on the remaining unseen Tier 2 holdout queries (`10..N`).
    6. **Target Consumer-Hardware Victory:** Beat our un-evolved `hipporag` baseline and target the HippoRAG 2 paper (`Llama-3.3-70B` + `NV-Embed-v2`) benchmarks on consumer hardware (`Qwen3-4B` + `Mistral-7B` + `embeddinggemma-300m`).
- **Exit Gate (`D1-A6` — Epic 1 Acceptance Gate):**
  1. `tests/daemon/test_runtime.py` passes lock contention, 32-slot `BUSY` overflow, Stage 1 crash-resume, and inference outage recovery.
  2. Native `yaw.daemon.graph` engine + universal `MIPROv2`/`GEPA`-evolved `HippoRAGProgram` (trained only on Tier 1 slices / external OOD corpora) matches or beats our upstream `hipporag` 3-model baseline on **Tier 2** disjoint holdout benchmarks in [evaluation](evaluation.md).

---

## 3. Checklist Summary

- [x] **D1-A0 (`D1-01`):** Embedding repeatability (`cosine >= 0.999`), $T=0.2$ floors, GBNF, 500-line FIFO logger, paper datasets downloaded.
- [x] **D1-A1 (`E1-S1`):** `yaw/core/` + containment tree locked; `test_architecture.py` enforces `0` horizontal/cross-domain imports.
- [x] **D1-A2 (`E1-S2`):** `inference/` refactored onto native `dspy` (`dspy.LM`, `dspy.Embedder`, `dspy.Signature`s, `dspy.Predict`/`ChainOfThought`/`RLM`, GBNF grammar forwarding, `.save()`/`.load()`) with `0` `hipporag` imports in `yaw/`.
- [x] **D1-A3 (`E1-S3`):** `storage/{db,hydration}.py` implements single-schema `yaw.db` (`sqlite-vec`, contentless `FTS5`, provenance edges) and 1-indexed `itertools.islice` line hydration.
- [x] **D1-A4 (`E1-S4`):** `graph/{ppr,engine}.py` + `storage/*` + `--query-offset`/`--evolve` hooks achieve **Tier 1 benchmark parity** vs. upstream `hipporag` on the 3-model local setup (YAW Recall@2: 0.8250 vs upstream: 0.8000; Recall@5: 1.0000 vs upstream: 1.0000; EM: 0.20 vs 0.20).
- [x] **D1-A5 (`E1-S5`):** `ingestion/corpus.py` (`cocoindex>=1.0.25`) emits pure coalesced line-pointer `SyncDelta`s across add/edit/rename/delete.
- [ ] **D1-A6 (`E1-S6`):** `daemon/{runtime,cli}.py` (`yaw <domain> <action>` CLI & isolated sandbox `dspy optimize` / `load-program`) passes lifecycle/lock/recovery tests AND native engine + universal `MIPROv2`/`GEPA`/`RLM` program (trained only on Tier 1 / OOD corpora) beats baseline on **Tier 2** disjoint holdout benchmarks.
