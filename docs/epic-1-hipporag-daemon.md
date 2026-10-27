# Epic 1: Native `yaw.db` Engine & CocoIndex Daemon

Owner: [FOUNDATIONAL](../FOUNDATIONAL.md) | Results: [evaluation](evaluation.md) | Next: [Epic 2](epic-2-fastmcp-http.md)

## Hard Gates

Runtime model update, 2026-10-22: the user selected `gemma-4-E4B-it-qat-q4_0-gguf-Q4_0`
(`unsloth/gemma-4-E4B-it-qat-q4_0-gguf-Q4_0:Q4_K_M`) for extraction, QA and reflection on the
existing external llama.cpp GPU service (32,768 context). This supersedes Qwen
as the runtime default in the packet descriptions below. GBNF worker / plain
RLM separation, embedding selection, and historical benchmark targets remain.

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
    ingestion/             corpus.py, datasets.py, bundles.py      (imports: core, cocoindex)
    graph/                 engine.py, ppr.py                       (imports: graph.*, core, scipy)
      storage/             db.py, vectors.py, graph_ops.py, shacl.py, hydration.py, candidate_cache.py (imports: core, sqlite3, sqlite_vec)
      inference/           models.py, prompts.py, embedding.py, chat.py, grammars.py, parsers.py, invariants.py, adapter.py, rlm.py, retriever.py, proposer.py, diagnostics.py, reflection.py, optimizer.py, program.py (imports: core, dspy)
  __main__.py                                                      (imports: daemon.cli)
```

- **Containment Law (`tests/core/test_architecture.py`):**
  1. **Empty Intermediate Packages:** Intermediate parent package initializers (`yaw/`, `yaw/daemon/`, `yaw/daemon/graph/`) contain zero executable statements and trigger zero parent-package import side-effects.
  2. **Constricted Leaf Facades:** Leaf domain packages (`yaw.core`, `yaw.daemon.graph.inference`, `yaw.daemon.graph.storage`, `yaw.daemon.ingestion`) provide constricted public API facades (`_PUBLIC_API` + lazy `__getattr__` + `__dir__`) allowing leaf mutation into single-responsibility modules while keeping a stable public API contract.
  3. **Strict Domain Containment:** A module in package `P` imports **only** `P.*` (its own descendants) and `yaw.core.*`. **Zero horizontal edges:** `ingestion` $\not\leftrightarrow$ `graph`, and `storage` $\not\leftrightarrow$ `inference` $\not\leftrightarrow$ `ppr`. Only a parent orchestrator (`daemon/runtime.py` or `graph/engine.py`) bridges its direct children.
  4. **Zero Protected (`_`) or Private (`__`) Cross-Module Imports or Facade Exports:** Any symbol starting with a leading underscore is strictly module-internal. No `name.startswith("_")` may ever appear in a leaf `_PUBLIC_API` facade, in cross-module imports (`from .mod import _helper`), or in cross-module attribute accesses.
  5. **Zero `hipporag` imports anywhere in the repository:** `hipporag` is permanently retired from `pyproject.toml`, `uv.lock`, `yaw/`, `benchmark.py`, `scripts/`, and `tests/`. Benchmark evaluation compares native `yaw` against Target 1 (`MONKEYPATCHED_HIPPORAG_BASELINES`) and Target 2 (`PAPER_BASELINES`).

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
    - `StructuredTriple(subj, pred, obj, spo_def="", subj_def="", obj_def="")`
    - `SHACLViolation(focus_node, source_shape, result_path, message, value="", severity="sh:Violation")`
    - `OpenIERecord(chunk_id, entities: tuple[str, ...], triples: tuple[tuple[str, str, str], ...], entity_definitions: tuple[tuple[str, str], ...] = (), structured_triples: tuple[StructuredTriple, ...] = ())`
    - `RetrievalResult(question, passages: tuple[HydratedPassage, ...], answer: str | None = None)`
    - `SyncDelta(added_chunks: tuple[tuple[ChunkPointer, str], ...], deleted_chunk_ids: tuple[str, ...])`
    - `IndexStats(chunks, entities, facts, edges, config_fingerprint="", openie_checkpoints=0, entity_labels=0, shacl_conforms=True)`
- **Drop (Anti-Spaghetti):** Drop `d1bc65e:src/yaw/utils.py`, global mutable singletons, and upstream's mutable `QuerySolution.to_dict()` / numpy array fields.
- **Exit Gate (`D1-A1`):** `uv run --locked python -m unittest discover -s tests -v` passes with `tests/core/test_architecture.py` verifying `0` cross-domain imports and empty `__init__.py`s across all `yaw/**/*.py`.

---

### Sprint 2 (`E1-S2`): `yaw/daemon/graph/inference/` — Native DSPy (`dspy.LM`, `dspy.Embedder`, Canonical 12 `dspy.Signature`s, First-Class Embedding Trail, `dspy.RLM`, `dspy.GEPA`)
- **Files:** `pyproject.toml`, `uv.lock`, `yaw/daemon/graph/inference/{models.py, embedding.py, chat.py, grammars.py, prompts.py, templates.py, parsers.py, program.py}`, `tests/daemon/graph/inference/test_prompts.py`.
- **Allowed imports:** `yaw.core.*`, `dspy`, `pydantic`, `numpy`, `difflib`, `hashlib`, `json`, `re`, `ast`. (Zero `hipporag` imports in `yaw/**`).
- **Design & Capabilities:**
  - **`embedding.py` (`dspy.Embedder` + First-Class Telemetry Trail):**
    - Wrap `dspy.Embedder(model="openai/...", api_base=..., api_key=..., batch_size=..., caching=True, encoding_format="float")` for Lemonade (`embeddinggemma-300m-GGUF`) or `ollama/...` for Ollama.
    - Apply `distinct-titled` passage prefix (`title: {title} | text: {body}`), fact query prefix (`task: fact verification | query: {query}`), and passage query prefix (`task: search result | query: {query}`) + L2 normalization (`np.float32`).
    - Emit structured telemetry (`self.telemetry`, `bind_telemetry_hook`) on every `encode` / `batch_encode` invocation (`stage`, `predictor="embedder"`, `task_mode`, `active_prefix`, `num_inputs`, `num_pieces`, `num_batches`, `tokens_used`, `usage`) and wire into `HippoRAGProgram.telemetry` and `HippoRAGProgram.embedding_audit_trail`.
  - **`chat.py` (`dspy.LM` & `create_dspy_rlm_pair`):**
    - Drive `dspy.LM` against Lemonade (`openai/...`) or Ollama (`ollama_chat/...`) on unified `Qwen3-4B-Instruct-2507-GGUF` (`llamacpp` on Vulkan GPU, `8192–16384` context window, `temp=0.2`, `freq_penalty=0.5`) with role- and recipe-aware GBNF capability flags (`_yaw_supports_gbnf`):
      - **Direct Extraction, QA Reader & RLM Worker (`role in ("extraction", "qa", "reflection_worker")`):** `_yaw_supports_gbnf=True` (`extra_body={"grammar": ...}`).
      - **RLM Python REPL Orchestrator (`role in ("rlm", "reflection")`):** `_yaw_supports_gbnf=False` so `dspy.RLM` emits unconstrained Python REPL code inside `dspy.PythonInterpreter` (`dspy[deno]>=3.0.0`).
      - `create_dspy_rlm_pair()` constructs `(orchestrator_lm, sub_lm)` pairing an unconstrained `Qwen3-4B` RLM orchestrator (`_yaw_supports_gbnf=False`) with a GBNF-enabled `Qwen3-4B` sub-LM worker (`_yaw_supports_gbnf=True`).
  - **`grammars.py`, `prompts.py`, `parsers.py`, `invariants.py`, `adapter.py`, `rlm.py`, `retriever.py`, `proposer.py`, `diagnostics.py`, `reflection.py`, `optimizer.py`, `program.py` (Canonical 12 DSPy Signatures, `/dspy-rlm-module` Discipline, Polymorphic `ExtractEntities` & `ExtractTriples` GBNF, `DefineMissingEntities`, & `compute_task_quality_score`):**
    - Enforce the **Canonical 12 `dspy.Signature` classes** (9 runtime predictors in `HippoRAGProgram`: 7 core direct GBNF predictors `ExtractEntities`, `ExtractTriples`, `DefineMissingEntities`, `JudgeSynonyms`, `ExtractQueryIntent`, `FilterTriples`, `AnswerQuestion` + 2 SHACL healing predictors `HealSHACLViolations` in `dspy.RLM` and `ProposeSHACLRepairs`; plus 3 reflection predictors in `RLMInstructionProposer`: `ReflectiveInnerDistillerSignature` in `dspy.RLM`, `ProposeInstructionSignature`, `CompressInstructionSignature`).
    - Following `/dspy-rlm-module`, `dspy.RLM` is strictly reserved for multi-step data-exploration / large-context Python REPL loops (`shacl_healer_rlm` and `RLMInstructionProposer.inner_rlm`) backed by `dspy.PythonInterpreter` (`dspy[deno]>=3.0.0`), with kwargs-only sandbox tool dispatch and explicit `sub_lm` wiring (`execution_mode` and `qa_mode` toggles are retired).
    - `ExtractEntities` emits 2-element `[entity_name, contextual_definition]` pairs (`DSPY_NER_GBNF_GRAMMAR`) handed off into Stage 2; `ExtractTriples` emits 4-field `{"sub", "pred", "obj", "def"}` objects where known Stage 1 entities are strings and newly introduced Stage 2 entities are 2-element `[entity_name, contextual_definition]` pairs (`entity-slot ::= known-entity | new-entity-pair`); `DefineMissingEntities` synthesizes definitions for any triple entities lacking definitions (`build_dspy_missing_entity_defs_grammar`, zero heuristic `spo_def` copying); `JudgeSynonyms` adjudicates candidate entity pairs into `skos:altLabel`, `skos:hiddenLabel`, `skos:related`, or `owl:differentFrom`; `ProposeSHACLRepairs` proposes bounded repairs for closed-world SHACL violations.
    - `compute_task_quality_score` (`yaw/daemon/graph/inference/parsers.py`, `task-quality-arithmetic-mean-v2`) implements the unweighted arithmetic mean ($\text{score} = \frac{1}{|M|}\sum_{m \in M} m$) over available `Recall@1/2/5/10/20`, `EM`, and `F1` while preserving all 7 raw metrics in `objective_scores`, with zero cost/token/SHACL terms in the optimization reward.
    - Keep stateless helpers `compute_mdhash_id(content, prefix)`, `text_processing(text)`, `filter_invalid_triples(triples)`, and `match_facts_to_candidate_indices(candidate_facts, matched_facts)`.
- **Drop (Anti-Spaghetti):** Drop `hipporag.llm.openai_gpt.CacheOpenAI`, `hipporag.embedding_model.base.BaseEmbeddingModel`, `hipporag/prompts/prompt_template_manager.py`, and `d1bc65e:src/yaw/nlp/*`.
- **Exit Gate (`D1-A2`):** `tests/daemon/graph/inference/` passes unmocked tests for `dspy.Embedder` (including telemetry trail hooks), `dspy.LM`, `create_dspy_rlm_pair`, all 12 canonical `dspy.Signature` modules, `compute_task_quality_score`, `.save()`/`.load()` state round-tripping, and `test_architecture.py` enforces `0` `hipporag` imports inside `yaw/`.

---

### Sprint 3 (`E1-S3`): `yaw/daemon/graph/storage/{db.py, vectors.py, graph_ops.py, shacl.py, hydration.py, candidate_cache.py}` — Closed-World SKOS/RDF/OWL `yaw.db` & Line Hydration
- **Files:** `yaw/daemon/graph/storage/{db.py, vectors.py, graph_ops.py, shacl.py, hydration.py, candidate_cache.py}`, `tests/daemon/graph/storage/{test_db.py, test_hydration.py, test_candidate_cache.py}`.
- **Allowed imports:** `yaw.core.*`, `sqlite3`, `sqlite_vec`, `numpy`, `hashlib`, `itertools`, `pathlib`.
- **Copy/Adapt From:**
  - **SQLite WAL, `ManagedConnection` & `sqlite-vec` setup (`db.py`):** Provide `open_managed_yaw_db(db_path)` (`isolation_level=None`, `ManagedConnection.transaction()` explicit `BEGIN IMMEDIATE` / `SAVEPOINT` nesting, `synchronous=NORMAL`, `temp_store=MEMORY`, `busy_timeout=30000`, `foreign_keys=ON`) and `sqlite_vec.load(conn)`.
  - **Closed-World SKOS/RDF/OWL Schema (`db.py`):**
    ```sql
    CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, rel_posix_path TEXT NOT NULL, start_line INT NOT NULL, end_line INT NOT NULL, title TEXT NOT NULL);
    CREATE TABLE openie_chunks (chunk_id TEXT PRIMARY KEY, entities_json TEXT NOT NULL, triples_json TEXT NOT NULL, structured_triples_json TEXT NOT NULL DEFAULT '[]'); -- openie_chunks.entities_json stores [name, definition] 2-element pairs alongside triples_json / structured_triples_json storing structured triples
    CREATE TABLE entities (entity_id TEXT PRIMARY KEY, name TEXT NOT NULL, definition TEXT NOT NULL DEFAULT '');
    CREATE TABLE entity_labels (
        label_norm TEXT NOT NULL,
        entity_id TEXT NOT NULL,
        label_kind TEXT NOT NULL CHECK (label_kind IN ('skos:prefLabel', 'skos:altLabel', 'skos:hiddenLabel')),
        raw_label TEXT NOT NULL,
        chunk_id TEXT NOT NULL,
        PRIMARY KEY (label_norm, entity_id, chunk_id)
    );
    CREATE TABLE facts (fact_id TEXT PRIMARY KEY, subj_id TEXT NOT NULL, pred TEXT NOT NULL, obj_id TEXT NOT NULL, raw_triple_json TEXT NOT NULL, definition TEXT NOT NULL DEFAULT '');
    CREATE TABLE edges (src_id TEXT NOT NULL, dst_id TEXT NOT NULL, kind TEXT NOT NULL, weight REAL NOT NULL, PRIMARY KEY (src_id, dst_id, kind));
    CREATE TABLE edge_sources (src_id TEXT NOT NULL, dst_id TEXT NOT NULL, kind TEXT NOT NULL, chunk_id TEXT NOT NULL, PRIMARY KEY (src_id, dst_id, kind, chunk_id));
    CREATE VIRTUAL TABLE chunk_fts USING fts5(chunk_id UNINDEXED, title, body, content='', contentless_delete=1);
    CREATE VIRTUAL TABLE fact_fts USING fts5(fact_id UNINDEXED, text, content='', contentless_delete=1);
    -- + sqlite-vec float32 tables: chunk_vec(chunk_id, embedding), entity_vec(entity_id, embedding), fact_vec(fact_id, embedding)
    ```
  - **SKOS Label Resolution, Merging, Closed-World SHACL Linting & Turtle Export (`db.py`):**
    - `find_entity_disambiguation_candidates` (2-channel dense KNN + lexical prefix/acronym/trigram blocking + unadjudicated same-`label_norm` distinct-`entity_id` pairs), `upsert_entity_labels` (enforcing SKOS `S13` & `S14` across all `chunk_id` rows), `resolve_entity_labels` (homograph query-vector cosine disambiguation), `merge_entities`, `upsert_semantic_edges` (`skos:related` / `owl:differentFrom` with `owl:differentFrom` dominance and exclusion from `load_graph_topology`), `delete_chunks` (including homograph `owl:differentFrom` preservation upon `skos:prefLabel` promotion), `lint_graph_shacl` (all 6 shapes), and `serialize_graph_turtle`.
  - **1-Indexed Line Hydration (`hydration.py`):** Adapt `hydrate_disk_slice` from `git show d1bc65e:src/yaw/retrieval/retriever.py:30-75` (`itertools.islice(f, start_line - 1, end_line)` + `resolved.is_relative_to(corpus_root)` path guard), with missing file/span `stale_on_disk` detection, plus a callable `hydrate_chunk(chunk_id)` tool helper for `dspy.RLM` on-demand passage inspection.
- **Drop (Anti-Spaghetti):** Drop all 7 upstream files (`EmbeddingStore` Parquet rewrite loops, `openie_results_*.json`, `chunk_metadata.json`, `graph.pickle`) AND drop `d1bc65e:src/yaw/storage/{rdf.py, schema.py}` (`unified_edges` SQL views, `compiled_violations`, `aliases`, `canon_verdicts`).
- **Exit Gate (`D1-A3`):** Unmocked `test_db.py` & `test_hydration.py` pass top-$k$ `sqlite-vec` KNN, contentless `FTS5` BM25 (`1 / rank` L1-normalized), SKOS `S13`/`S14`, `owl:differentFrom` dominance, 6-shape SHACL linting, Turtle serialization, atomic `chunk_id` deletion cascading through `edge_sources` $\to$ `edges`/`entities`/`entity_labels`/`facts` (with homograph promotion `owl:differentFrom` preservation), and `stale_on_disk` detection.

---

### Sprint 4 (`E1-S4`): `yaw/daemon/graph/{ppr.py, engine.py}` — Sparse PPR, EDC + SKOS/SHACL Graph Engine & Tier 1 Parity
- **Files:** `yaw/daemon/graph/{ppr.py, engine.py}`, `tests/daemon/graph/{test_ppr.py, test_engine.py}`.
- **Allowed imports:** `ppr.py` imports `yaw.core.*`, `numpy`, `scipy.sparse` only; `engine.py` imports `yaw.daemon.graph.{storage,inference,ppr}` and `yaw.core.*`. (Zero external graph frameworks).
- **Copy/Adapt From:**
  - **Sparse CSR Power Iteration (`ppr.py`):** Adapt `power_iteration` from `git show d1bc65e:src/yaw/retrieval/retriever.py:280-315`:
    - Build symmetric row-stochastic CSR matrix $P = D^{-1} A$ from `edges(src_id, dst_id, weight)` (including `passage`, `fact`, and `skos:related` edges while excluding `owl:differentFrom`) plus `min_max_normalize` (`hipporag/utils/misc_utils.py:132`).
    - Run $p^{(t+1)} = (1 - \alpha) (P^T p^{(t)} + m_{\text{dangling}} p_0) + \alpha p_0$ with HippoRAG damping $\alpha = 0.5$ (`damping=0.5`), converging in `<3 ms` on CSR matrices.
  - **HippoRAG 2 + EDC + SKOS/SHACL Graph Construction (`engine.py`):**
    - Stage 1 OpenIE via `HippoRAGProgram.extract_openie(passage)` cached in `openie_chunks` with Stage 1 `entity_definitions` (`[name, definition]`) and Stage 2 `StructuredTriple` contextual definitions (`subj_def`, `obj_def`, `spo_def`). Materialize all Stage 1 NER entities (including standalone entities that do not appear in Stage 2 triples) into `entities`, `entity_labels`, and bidirectional `(chunk <-> entity)` `passage` edges.
    - **Provisional Same-Label Sense-Split Gate:** Before assigning `compute_mdhash_id(label_norm, prefix="entity-")`, compare contextual embeddings (`# {name}\n{definition}`); if $\cos \ge 0.85$ (`same_sense_cosine_gate`), reuse the existing `entity_id`; if $\cos < 0.85$, mint a provisional `sense_id` (`compute_mdhash_id(f"{label_norm}::{defn.lower()}", prefix="entity-")`) **without** writing any `owl:differentFrom` edge in Step 4 so that `find_entity_disambiguation_candidates` surfaces the pair to `judge_synonyms`.
    - **Atomic Closed-World Transaction (Zero Deterministic Fallbacks):** Inside `with self.managed_conn.transaction() as conn:`, commit nodes/vectors/edges, run 2-channel + same-`label_norm` candidate generation + `self.program.judge_synonyms(candidates)` (`merge_entities` / `upsert_semantic_edges`), and run `lint_graph_shacl(conn, auto_heal=False)` + `self.program.heal_shacl_violations(violations, focus_context=...)`.
  - **HippoRAG 2 Retrieval & DSPy Multi-Step QA (`engine.py`):**
    - Fact linking & Recognition Memory (`HippoRAG.py:2125-2175`): query `fact_vec` + `fact_fts` RRF $\to$ filter via `HippoRAGProgram.filter_triples`.
    - PPR Seeding (`HippoRAG.py:2008-2120`): resolve surface mentions via `resolve_entity_labels(conn, surface_mentions, query_vec=query_fact_vec)`; entity weight $w(e) = \text{mean\_fact\_score}(e) / \max(1, \text{doc\_freq}(e))$; passage weight $w(c) = 0.05 \cdot (\widehat{s}_{\text{vec}}(c) + \widehat{s}_{\text{fts}}(c))$ $\to$ `ppr.run_ppr` $\to$ rank chunk nodes $\to$ hydrate **only** top-$k$ chunks via `storage/hydration.py`.
    - QA reader: run `HippoRAGProgram.answer_question` (`qa` `dspy.ChainOfThought(AnswerQuestion)` with pure-JSON GBNF grammar).
  - **End-to-End DSPy Trace Hook (`engine.py`) & Disjoint Benchmark Split (`benchmark.py`):**
    - Wire `engine.retrieve_and_answer(question)` through `HippoRAGProgram` so a single `dspy.Module.forward(question)` trace captures `FilterTriples` (`triple_filter`) $\to$ `<3 ms` CSR PPR + line hydration $\to$ `AnswerQuestion` (`qa`).
    - Add `--engine {hipporag, yaw}`, `--query-offset <M>`, `--evolve {none, miprov2, gepa, hybrid}`, `--train-queries <K>`, and `--compiled-program <path>` to `benchmark.py` to support strict disjoint learning (`[0:K]`) vs. holdout (`[K:N]`) evaluation.
- **Drop (Anti-Spaghetti):** Drop `python-igraph` (`self.graph`, vertex deletion index-shifting bugs, `graph.pickle`) and drop `d1bc65e`'s `compute_upward_conductance`, `in_degrees**0.5` hub dampening, and `InMemoryLookupKB`.
- **Exit Gate (`D1-A4` — Tier 1 Dev Parity Gate):** `uv run --locked python benchmark.py --engine yaw --dataset {sample,2wikimultihopqa,musique} --max-queries 10` matches or beats upstream `hipporag` on `Recall@2`, `Recall@5`, `QA EM`, and `QA F1`.

---

### Sprint 5 (`E1-S5`): `yaw/daemon/ingestion/corpus.py` — CocoIndex Incremental Line-Pointer Pipeline & Target Sink Compiler Hook
- **2026-10-27 replacement packet:** execute the restart checklist in
  [dataset ingestion refactor](dataset-ingestion-refactor.md). Dataset preparation
  emits isolated document files; persistent CocoIndex snapshots replace fresh-state
  reconstruction. The exit condition includes warm snapshots without reparsing,
  unchanged source values, loader-owned Enterprise stratification and no legacy
  discovery/projection consumers. The packet supersedes benchmark-shape parsing
  and automatic benchmark JSON projection settings described elsewhere.
- **Files:** `yaw/daemon/ingestion/corpus.py`, `tests/daemon/ingestion/test_corpus.py`.
- **Allowed imports:** `yaw.core.*`, `cocoindex`, `hashlib`, `pathlib`. (Zero imports from `graph/`).
- **Copy/Adapt From:**
  - Adapt `chunk_markdown` from `git show d1bc65e:src/yaw/indexer/pipeline.py:122-167`:
    - Walk `corpus.root` using `cocoindex>=1.0.0` (`localfs.walk_`, `PatternFilePathMatcher` for `*.md`, `*.txt`, `RecursiveSplitter`).
    - Compute 1-indexed inclusive `start_line = content[:c.start].count("\n") + 1`, `end_line = start_line + c.text.count("\n")`, and `content_hash = sha256(c.text)`.
    - Diff against committed `chunk_id` set to produce a single coalesced `SyncDelta(added_chunks, deleted_chunk_ids)` and invoke the registered `on_delta: Callable[[SyncDelta], None] | None` callback inside `_ChunkTargetHandler._apply_actions` before CocoIndex commits state to `coco_lmdb` (if `on_delta` raises, CocoIndex aborts the state commit so the next `poll()` retries).
- **Drop (Anti-Spaghetti):** Never import `graph/`, `storage/`, or `inference/` inside `corpus.py`. `corpus.py` bridges to `YawGraphEngine` exclusively through the `Callable[[SyncDelta], None]` injected by `Runtime`.
- **Exit Gate (`D1-A5`):** Unmocked `tests/daemon/ingestion/test_corpus.py` passes file add, edit, rename, delete, memoized zero-op re-scans, and `on_delta` target sink rollback on compiler failure.

---

### Sprint 6 (`E1-S6`): `yaw/daemon/{runtime.py, cli.py}`, Two-Phase `MIPROv2` + `GEPA` Evolution & Tier 2 Holdout Gate
- **Restart acceptance packet (2026-10-22):** `yaw dspy optimize --resume` and
  `--db-path latest` reopen the newest timestamped sandbox, recover its original
  output path and invocation, preserve the GEPA candidate frontier/budget, and
  reuse only database checkpoints with matching durable prompt provenance.
  Incompatible run identities and unverifiable legacy GEPA state fail without
  deleting checkpoints. Observable exit: real SQLite/CocoIndex/DSPy/GEPA restart
  tests retain a previously saved superior candidate and reject changed data.
  This packet does not satisfy the separate Tier 2 parity acceptance gate.
- **Candidate-stage disk-cache packet (2026-10-22):** replace growing in-memory
  embedding/OpenIE/synonym/SHACL cache dictionaries with a shared, bounded SQLite
  cache in `yaw.db`. Preserve typed DSPy/RLM traces and logical token costs;
  hash complete inference inputs/settings and hydrate source content rather than
  copying passages into cache payloads. Observable exit: independent Python
  processes rebuild identical graphs with zero additional loopback inference
  requests, while changed prompts/models/sampling/source content force misses.
  This packet does not eliminate GEPA seed reevaluation or claim Tier 2 parity.
- **Priority 2 response-cache packet (2026-10-22):** enable DSPy's native disk
  request cache during evolution with its memory tier disabled. Keep this change
  in `inference/chat.py` (public cache-interface usage accounting) and `engine.py`
  (configuration). Replay logical usage without altering native response history,
  retain endpoints in request keys, and preserve explicit cache bypass. Observable
  exit: separate-process loopback requests hit disk with no new HTTP calls, and
  independent real GEPA runs produce equal live/cached scores and token costs.
- **Files:** `yaw/core/config.py`, `yaw/daemon/{runtime.py, cli.py}`, `yaw/__main__.py`, `yaw.toml`, `tests/daemon/test_runtime.py`.
- **Allowed imports:** `runtime.py` & `cli.py` import `yaw.daemon.{ingestion,graph}`, `yaw.core.*`, `filelock`, `tomllib`, `argparse`, `queue`, `threading`.
- **Copy/Adapt From:**
  - Adapt `DaemonLifecycle` and graceful signal handling from `git show d1bc65e:src/yaw/server/daemon.py:260-340` (using `filelock.FileLock("yaw.db.lock")` instead of raw `msvcrt`/PID-alive checks).
  - Adapt CLI commands into hierarchical grammar `python -m yaw <domain> <action> [--flags]` (`yaw daemon {start|stop|status|run}`, `yaw graph {build|rebuild|lint [--turtle]}`, `yaw dspy {optimize|load-program|benchmark}`, `yaw config {get|set}`). `yaw graph lint [--turtle]` opens `yaw.db` in read-only URI mode (`mode=ro`) without acquiring `yaw.db.lock`.
  - Wire single-owner 32-slot bounded FIFO worker thread in `runtime.py`: binds `CorpusIngestor(..., on_delta=self._apply_delta)` so `self._ingestor.poll()` compiles and validates each `SyncDelta` inside the CocoIndex target sink transaction, and serves `retrieve`/`rag_qa` requests with non-blocking cached `status()`/`stats()`.
  - **Zero-Lock-In `MIPROv2` + `GEPA` + `SIMBA` Evolution (`engine.evolve_prompts` & `python -m yaw dspy optimize` / `benchmark.py --evolve`):**
    1. **Anti-Overfitting Rule:** Never run prompt evolution on full Tier 2 benchmark corpora (`musique`, `2wikimultihopqa`, `hotpotqa` full sets) to prevent benchmark lock-in.
    2. **Task-Quality Fitness Objective (`compute_task_quality_score`, `yaw/daemon/graph/inference/parsers.py`, `task-quality-arithmetic-mean-v2`):**
       - Unweighted arithmetic mean ($\text{score} = \frac{1}{|M|}\sum_{m \in M} m$) over available quality metrics (`Recall@1, 2, 5, 10, 20`, `EM`, `F1`) with all 7 raw metrics in `objective_scores` for GEPA's validation-set Pareto frontier and compile-time `use_merge=True` (`MergeProposer`). Tokens, latency, SHACL, and repairs are diagnostic telemetry only (never Pareto objectives or penalties). `HippoRAGProgram` is a single textbook `dspy.Module` (`best_candidate` returned by `gepa.compile()`); custom self-cloning ensemble wrappers (`ParetoProgramEnsemble`) are retired in favor of textbook `dspy.GEPA` compile-time merging and native `dspy.Ensemble(reduce_fn=dspy.majority)`.
    3. **Full Optimizable Surface, First-Class Embedding Trail & Single-Level Reflection RLM:** Expose all 12 canonical signatures (`HippoRAGProgram` 9 runtime predictors + `RLMInstructionProposer` 3 reflection predictors), `EmbeddingGemma` query/passage instructions (with full embedding telemetry recorded in `HippoRAGProgram.telemetry` and `HippoRAGProgram.embedding_audit_trail`), and parametric GBNF grammar bounds to canonical `"hybrid"` optimization (`dspy.MIPROv2` warm-up + `dspy.GEPA` handoff + always-on `dspy.SIMBA`). Drive `dspy.GEPA(reflection_lm=...)` via the unified RLM pair (`create_dspy_rlm_pair`: unconstrained RLM REPL orchestrator + GBNF sub-LM worker), persisting `<stem>_proposer_telemetry.json` and `<stem>_graph_rlm_audit.json`.
    4. **Evolve on Tier 1 Calibration Slices or External OOD Corpora:** Run `dspy.MIPROv2` + `dspy.GEPA` (`enable_gbnf=True` for `llama.cpp` GBNF grammar enforcement via `config={"extra_body": {"grammar": ...}}`) strictly on the small **Tier 1 benchmark sets** (`--max-queries 10`) and/or **external out-of-distribution corpora** (`onyx-dot-app/EnterpriseRAG-Bench`, `microsoft/graphrag-benchmarking-datasets`) to produce one universal, generalizable `compiled_program.json`.
    5. **Run Unseen Holdout Benchmarks Zero-Shot:** Evaluate `uv run yaw dspy benchmark --gate benchmark_gate.json` / `--engine yaw --compiled-program compiled_program.json --query-offset 10` on the remaining unseen Tier 2 holdout queries (`10..N`).
    6. **Target Consumer-Hardware Victory:** Beat our un-evolved `hipporag` baseline and target the HippoRAG 2 paper (`Llama-3.3-70B` + `NV-Embed-v2`) benchmarks on consumer hardware (`Qwen3-4B-Instruct-2507-GGUF` + `embeddinggemma-300m-GGUF`). Once Tier 2 HippoRAG 2 parity/beat is confirmed, upstream `hipporag` will be permanently removed.
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
