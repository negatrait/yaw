# Native CocoIndex indexing and live ingestion

Status: active (clean-slate `14`-file `M_FINAL_1B`, `5c4b115`).

| Relation | Links |
| --- | --- |
| Governed by | [FOUNDATIONAL.md](../FOUNDATIONAL.md), [AGENTS.md](../AGENTS.md), [epic-1-hipporag-daemon.md](epic-1-hipporag-daemon.md) |
| Rules | [pipeline.md](../.agents/rules/pipeline.md): `yaw/pipeline.py`, `yaw/datasets.py`; [tests.md](../.agents/rules/tests.md): `tests/**/*.py`; [docs.md](../.agents/rules/docs.md): `docs/**/*.md` |
| Owns code | `yaw/pipeline.py`, `yaw/datasets.py` |
| Peers | [epic-1-hipporag-daemon.md](epic-1-hipporag-daemon.md): owns `yaw/core/` (`YawConfig`, `compute_chunk_int_id`, `format_doc_for_embedding`, `resolve_coco_state_root`, `YawRuntimeError`); [native-cocoindex-graph.md](native-cocoindex-graph.md): owns `yaw/retrieval.py` (`RetrievalIndex`, `compute_db_fingerprints`, `compute_index_artifacts`, `normalize_fts_unicode61`); [dspy-improvements.md](dspy-improvements.md): owns `yaw/qa.py` & `yaw/optimize.py` consuming `start_pipeline`, `update_pipeline_once`, `close_pipeline`, `make_dspy_embed_fn`, and `resolve_slice_corpus_scope`; [epic-2-fastmcp-http.md](epic-2-fastmcp-http.md): owns `yaw/runtime.py`, `yaw/mcp_server.py` & `yaw/cli.py` orchestrating live `PipelineSession`; [enterprise-rag-bench-training-plan.md](enterprise-rag-bench-training-plan.md): owns `scripts/{benchmark,download_datasets,verify_paper_parity_gate,evaluation}.py` consuming `yaw/datasets.py` (with bundled `yaw/baselines.json`) bundles/metrics and constraining the `135/27` split; [epic-3-performance.md](epic-3-performance.md): owns `scripts/watchdog.py` & deferred `E3-P5` / `EVAL-6` proposals; [coverage-plan.md](coverage-plan.md): owns `tests/**/*.py` (`tests/test_pipeline.py`, `tests/test_datasets.py`) |
| References | [arxiv-2502.14802v1-hipporag2.md](arxiv-2502.14802v1-hipporag2.md): anchors HippoRAG 2 indexing and retrieval parity; [model-evaluation.md](model-evaluation.md): anchors local `llama.cpp` serving constraints |
| Supersedes / superseded by | Supersedes legacy `yaw/daemon/` multi-pipeline ingestion (`frozen_indexing_app_main`, `CorpusIndexingLiveComponent`, `source_chunks`, `chunk_inference`, `DatabaseSink`, `vec0`, `CandidateStageCache`) retired under [bare-bones-replacement-sprint.md](reviews/bare-bones-replacement-sprint.md) and `EVAL-5` ([mdl-dag-sprint.md](reviews/mdl-dag-sprint.md)) |
| Records | [mdl-dag-sprint.md](reviews/mdl-dag-sprint.md), [bare-bones-replacement-sprint.md](reviews/bare-bones-replacement-sprint.md), [incremental-publication-work.md](reviews/incremental-publication-work.md), [shacl-healer-publication.md](reviews/shacl-healer-publication.md), [evaluation-ingestion.md](evaluation-ingestion.md), [evaluation.md](evaluation.md) |

## 1. Scope

In scope: Declarative `cocoindex >= 1.2.0` live/catch-up ingestion (`yaw/pipeline.py`) into the plain 5-table SQLite schema (`chunks`, `chunk_fts`, `chunk_vocab`, `rabitq_index`, `citation_graph`), content-addressed LMDB memoization (`.yaw/coco_lmdb`), native AST/recursive byte-capped chunking, IEEE `<f2>` `float16` embedding persistence, shared `_DspyBatchEmbedder` HTTP concurrency gating, `reconcile_index_tables` artifact materialization, fail-fast `exception_handler` propagation, and zero-`dspy` content-addressed v2 `.md` dataset bundles and pure evaluation metrics (`yaw/datasets.py`).

Non-goals: Query-time `RetrievalIndex` scoring, 1-bit SoA RaBitQ LUT matching, cyclic citation diffusion, DPP reranking, and disk passage hydration (owned by [native-cocoindex-graph.md](native-cocoindex-graph.md)); `YawRuntime` lifecycle, `PRAGMA data_version` swaps, and FastMCP HTTP transport (owned by [epic-2-fastmcp-http.md](epic-2-fastmcp-http.md)); legacy OpenIE/W3C/SHACL extraction, `yaw/daemon/` modules, `vec0`, `DatabaseSink`, or `CandidateStageCache` (permanently retired per [FOUNDATIONAL.md §2](../FOUNDATIONAL.md#2-non-goals--retired-stack)).

<a id="2-problem-with-evidence"></a>
<a id="3-goals"></a>

## 2. Architecture & Design Principles

- **Single Declarative Writer (`yaw/pipeline.py`):** Every persistent write to `yaw.db` executes inside CocoIndex `>= 1.2.0` (`sqlite.mount_table_target` for `chunks`, `_FtsTableHandler` / `_FtsRowHandler` for `chunk_fts` + `chunk_vocab`, and `reconcile_index_tables` inside `conn_mgr.transaction()` for `rabitq_index` + `citation_graph`). Production live daemon mode (`live=True`), offline optimization indexing (`live=False`), and benchmarking run the identical `_indexing_main` flow differing only in `CORPUS_ROOT`, `included_patterns`, `db_path`, and per-DB `app_name`.
- **Zero Raw Chunk Text in SQLite:** `chunks` stores only the 63-bit integer `chunk_id`, `<f2>` `embedding_f16 BLOB`, `rel_path`, `source_hash`, 1-indexed inclusive `(start_line, end_line)`, `(char_start, char_end)`, `(ast_island_start, ast_island_end)`, and `title`. `chunk_fts` is strictly contentless (`content='', contentless_delete=1`, `tokenize='unicode61 remove_diacritics 1'`) paired with `chunk_vocab` (`fts5vocab(..., 'row')`).
- **No JSON-to-Text Re-Staging (`yaw/datasets.py`):** v2 dataset bundles write one UTF-8 Markdown file per corpus document (`documents/<document_filename(doc_id)>.md` with `# {title}` H1 headers via `render_document_markdown`), validated via SHA-256 `content_hash` in `manifest.json`. `resolve_slice_corpus_scope` passes slice `documents/*.md` patterns directly to `_indexing_main`.

<a id="3-revision-identity-memo-and-target-isolation"></a>

## 3. Revision Identity, Memoization & Target Isolation

- **Chunk Identity (`compute_chunk_int_id`):** Each chunk's primary key is a deterministic positive signed 63-bit integer derived from SHA-256 over `(rel_posix_path, start_line, end_line, char_start, char_end, source_hash)`, where `source_hash` is the SHA-256 hex digest of normalized UTF-8 slice lines.
- **DB-Invariant `@coco.fn(memo=True)` Keys:**
  - `_compute_chunk_embedding(title_body, task_mode, embedder)` (`@coco.fn(memo=True, version=4)`) takes `(title, body)` plus `task_mode` and `_PipelineEmbedder.__coco_memo_key__()` (provider, `embedding_model`, `base_url`, `embedding_task_mode`, and backend tag) — never `rel_path`, `chunk_id`, or `db_path` — so file moves, renames, or multi-DB index builds hit `.yaw/coco_lmdb` memos without re-embedding.
  - `_process_file` (`@coco.fn(memo=True, version=6)`) takes `FileLike`, targets, `embedder`, and chunking parameters, emitting `ChunkRow` and `(normalize_fts_unicode61(title), normalize_fts_unicode61(body))` target states.
  - Per-database target isolation is enforced by keying `app_name = app_name or f"yaw_{sha256(db_path)[:12]}"` and `_FtsTableKey(SQLITE_DB.key, "chunk_fts")`.
- **Single HTTP Admission Gate (`_DspyBatchEmbedder` & `_PipelineEmbedder`):** `_DspyBatchEmbedder` wraps `dspy.Embedder(..., batch_size=cfg.embed_batch_size, caching=False)` under `threading.BoundedSemaphore(cfg.embed_max_in_flight)` and is shared between indexing and runtime queries. `_PipelineEmbedder` bridges async CocoIndex batching (`coco.fn.as_async(batching=True, max_batch_size=..., max_in_flight=...)`), raising `coco.RetryWithSmallerBatch()` if a multi-item batch mismatches length and `YawRuntimeError("MODEL_ERROR", ...)` on non-finite, zero-norm, or dimension-drifting vectors.
- **Fail-Fast Error Propagation:** `coco.Environment` wires `exception_handler=_on_component_error`, recording the first error on `PipelineSession.error`, setting `ready_event`, and re-raising. `_watch_pipeline` surfaces `snap.stats.total.num_errors` as `YawRuntimeError("CONSISTENCY_ERROR", ...)` when `READY`. Each `start_pipeline` worker thread runs its own `asyncio.run(_session_worker())` event loop.

## 4. Schema & Module Ownership

| Artifact / State | Single Owner (`yaw/pipeline.py` or `yaw/datasets.py`) |
| --- | --- |
| `chunks` table (ROWID PK `chunk_id`, `<f2>` `embedding_f16 BLOB`, source span pointers) | `_indexing_main` -> `sqlite.TableSchema[ChunkRow].from_class(ChunkRow, primary_key=["chunk_id"])` via `sqlite.mount_table_target` |
| `chunk_fts` (contentless `unicode61 remove_diacritics 1`) & `chunk_vocab` (`fts5vocab`) | `_FTS_TABLES` (`_FtsTableHandler` + `_FtsRowHandler`) inside `conn_mgr.transaction()` |
| `rabitq_index` & `citation_graph` (single-row `id = 1` materialized artifacts) | `reconcile_index_tables` (`compute_db_fingerprints` + `compute_index_artifacts` from `yaw.retrieval`) |
| Persistent CocoIndex memoization & target tracking | `<state_dir>/coco_lmdb` (`resolve_coco_state_root(cfg.state_dir)`) with per-`db_path` `app_name` |
| v2 `.md` dataset bundles, `manifest.json`, slicing, and EM/F1/Recall metrics | `yaw/datasets.py` (`prepare_dataset_bundle`, `load_dataset_bundle`, `validate_dataset_bundle`, `resolve_slice_corpus_scope`, `compute_qa_em_f1`, `score_retrieval_recalls`, `score_qa_em_f1`) |

<a id="5-design-current-state"></a>

## 5. Design (current state)

```python
@dataclass(frozen=True, slots=True)
class ChunkRow:
    """Row schema for the plain SQLite `chunks` table storing IEEE float16 embeddings."""

    chunk_id: int
    embedding_f16: bytes
    rel_path: str
    source_hash: str
    start_line: int
    end_line: int
    char_start: int
    char_end: int
    ast_island_start: int
    ast_island_end: int
    title: str


@dataclass(slots=True)
class PipelineSession:
    """Handle bundle owning a CocoIndex Environment, worker thread/loop, and SQLite connection."""

    db_path: Path
    state_dir: Path
    corpus_root: Path
    embedding_model: str
    task_mode: str
    env: coco.Environment
    app: coco.App[Any, Any]
    conn_mgr: sqlite.ManagedConnection
    live: bool
    ready_event: threading.Event
    thread: threading.Thread
    loop: asyncio.AbstractEventLoop
    stop_async: asyncio.Event
    error: BaseException | None = None
```

- **Pipeline Lifecycle (`yaw/pipeline.py`):**
  1. `open_managed_db(db_path)` opens CocoIndex `sqlite.connect(target, timeout=30.0, load_vec=False)` with `PRAGMA busy_timeout=30000`, `PRAGMA journal_mode=WAL`, and `PRAGMA synchronous=NORMAL`.
  2. `start_pipeline(cfg, *, embed_fn=None, live=True, app_name=None)` starts the dedicated `yaw-cocoindex-session` thread (`asyncio.run(_session_worker())`), creates `coco.Environment(..., exception_handler=_on_component_error)`, mounts `_indexing_main`, and when `live=True` runs `_watch_pipeline` to call `reconcile_index_tables` whenever `snap.status.name == "READY"`.
  3. `update_pipeline_once(session)` executes `session.app.update_blocking(live=False)` and runs `reconcile_index_tables` synchronously for offline catch-up / optimization indexing.
  4. `reconcile_index_tables` compares `compute_db_fingerprints(conn)` `(n, dim, vec_sha256, source_sha256)` plus `(embedding_model, task_mode)` and non-empty blob lengths against `rabitq_index` and `citation_graph` (`id = 1`); when stale or missing, it executes `INSERT INTO chunk_fts(chunk_fts) VALUES('optimize')` and replaces both single-row artifact tables atomically from `compute_index_artifacts(rows, corpus_root)` (or deletes `id = 1` when `n == 0`).
  5. `close_pipeline(session)` signals `stop_async`, joins the worker thread (raising `SHUTTING_DOWN` on timeout), and closes `session.env` and `session.conn_mgr`.

## 6. Capability Gates & Dataset Scope Contracts

- **`CG-INDEX-1` (Plain 5-Table Schema & `<f2>` Storage):** `list_user_tables(conn)` yields only `chunks`, `chunk_fts`, `chunk_vocab`, `rabitq_index`, `citation_graph` (`0` `vec0` tables, `0` raw text columns); `chunks.embedding_f16` stores `2 * dim` IEEE float16 little-endian bytes (`<f2>`).
- **`CG-INDEX-2` (Cross-DB Memo Reuse & Target Isolation):** Distinct `db_path` targets share `.yaw/coco_lmdb` via `app_name` and `_FtsTableKey(SQLITE_DB.key, "chunk_fts")`; unchanged `(title, body)` chunks hit `_compute_chunk_embedding` LMDB memos with `0` duplicate embedding HTTP requests and zero cross-DB withdrawals.
- **`CG-INDEX-3` (Oversized Chunk Fail-Fast & Live Recovery):** `chunk_file_content` raises `ValueError` if an indivisible native chunk exceeds `chunk_size` UTF-8 bytes; file additions, edits, renames, and deletions reconcile `chunks`, `chunk_fts`, `chunk_vocab`, `rabitq_index`, and `citation_graph` cleanly in lockstep.
- **`CG-DATA-1` (Zero-`dspy` v2 `.md` Bundles):** `yaw/datasets.py` imports `stdlib`, `filelock`, `numpy`, and `yaw.core` only; `prepare_dataset_bundle` / `validate_dataset_bundle` write and verify SHA-256 `manifest.json` v2 `.md` files; `resolve_slice_corpus_scope` returns `(documents_root, included_patterns)` directly for `_indexing_main`.

<a id="7-removal-inventory"></a>

## 7. Removal Summary

All legacy `yaw/daemon/` ingestion and graph modules (`indexing.py`, `corpus.py`, `source_targets.py`, `bundles.py`, `managed_sqlite.py`, `DatabaseSink`, `CandidateStageCache`, `vec0`, `source_chunks`, `chunk_inference`, `tokens_json`, `entities_json`, `yaw_stage1`) are permanently removed; see [bare-bones-replacement-sprint.md](reviews/bare-bones-replacement-sprint.md) and [mdl-dag-sprint.md](reviews/mdl-dag-sprint.md).

<a id="8-checkpoints"></a>

## 8. Checkpoints & Quality Gates

- `CS-1`..`CS-6` (clean-slate CocoIndex pipeline & v2 `.md` dataset cutover): closed (`3cd3952`).
- `EVAL-5` (`M_FINAL_1B` `<f2>` plain SQLite schema, `unicode61` `chunk_fts` + `chunk_vocab`, `rabitq_index` + `citation_graph` reconciliation): shipped (`5c4b115`).

<a id="9-verification"></a>

## 9. Verification

Real-substrate verification lives in `tests/test_pipeline.py` and `tests/test_datasets.py` (real SQLite WAL databases, real `.yaw/coco_lmdb` environments, real `RecursiveSplitter`, and real filesystem bundles in `tmp_path`).

```powershell
uv run --offline --locked ty check
uv run --offline --locked ruff check
uv run --offline --locked ruff format --check
uv run --offline --locked pytest -o addopts="" tests/test_pipeline.py tests/test_datasets.py
git diff --check
```

Measurements recorded in: [evaluation-ingestion.md](evaluation-ingestion.md) and [evaluation.md](evaluation.md).

## Decisions

- 2026-11-06 *(Superseded)*: "Step 4 should be included in this unit, but obviously, you can implement one unit at a time according to whichever spec you write into @[docs/native-cocoindex-graph.md] or @[docs/native-cocoindex-indexing.md] , also sounds like you'll need to update our rules and tests to prevent these kinds of failures surfacing again." (User). Specs are promoted into `docs/native-cocoindex-graph.md` and `docs/native-cocoindex-indexing.md` and executed `1` `[target]` checkpoint unit at a time with regression tests.
- 2026-11-07 *(Superseded)*: "Audit and review for root cause for batch indexing regressing back to exponential time behavior. Incremental indexing and graph publication was designed and tested to be near-constant time for 64 + 64 chunk batches vs 128 chunk batchless indexing." / "It’s a SHACL Violation CONSISTENCY_ERROR" / "you will need to /spec first." (User). Enforces delta-scoped `64`-chunk `[target]` batch publication (`publish_graph` `version=8` `[observed: commit c6fa080]`, focus-isolated `heal_shacl_violations` `version=4` `[observed: commit c6fa080]`, repair-preserving `canonicalize_fact_definitions`, `"__healed__"` label provenance, `touched_entity_ids`-scoped Active EDC, delta-only `canonicalize_fact_text`, and single-pass post-canonicalization `reembed_facts` embedding).
- 2026-11-07 *(Superseded)*: "SHACL healing is taking extremely long - what is the performance bottleneck here?" / "/spec please spec the remedy for your debug audit above." (User). ~~Requires `source_chunks JOIN chunk_inference` polling via `compute_active_inference_fingerprint` and `load_active_chunk_inferences`~~ (superseded by 2026-11-07 Gaps 1–5 entry below for planner handoff); ~~retains published `chunks` and `meta['graph_input_fingerprints']` matching in `partition_frozen_chunks`~~ (`partition_frozen_chunks` superseded by 2026-11-09 Anti-Patterns 1–6 entry below); retains direct pre-hydrated `SHACLHealInput` dispatch to `YawProgram.shacl_repair` (`mode="direct"`, `heal_shacl_violations` `version=5` `[observed: yaw/daemon/graph/inference/components.py:309, 50be17d]`) and `OpenIEEndpoint.progress_callback` telemetry.
- 2026-11-07 *(Superseded)*: "/spec remedy gaps 1-5 in @[docs/native-cocoindex-indexing.md] and @[docs/native-cocoindex-graph.md] - remember the separation of concerns, one is for indexing and the other is for graphing. Use subagents to protect your own context from bloat." (User). Remedies Gaps 1–5 across the indexing/graph boundary: (1) `process_chunk` returns `ChunkProcessOutput(pointer, text, inference, vector_f32)` via `coco.use_mount` to `publish_graph` while `publish_graph` remains the single atomic `DatabaseSink` producer for all `15` `[target]` graph tables (including `chunk_vec`, with `0` `[target]` duplicate vector columns on `chunk_inference`); (2) `coco.Settings(db_path=...)` points to a shared persistent `<state_dir>/coco_lmdb` with `db_identity`-scoped component mounts and `TableKey(db_identity, name)` target states; (3) retires `load_active_chunk_inferences`, `compute_active_inference_fingerprint`, `ChunkStateTargetHandler`, and `GraphDeltaCoordinator` in favor of direct `coco.use_mount` value flow; (4) delegates `chunk_vec`, `entity_vec`, and `fact_vec` `sqlite-vec` `vec0` virtual table DDL/DML to [native-cocoindex-graph.md](native-cocoindex-graph.md); and (5) retires `candidate_stage_cache` SQLite usage in `runtime.py:1240–1241` (`50be17d`) alongside `CandidateStageCache` deletion in [native-cocoindex-graph.md](native-cocoindex-graph.md).
- 2026-11-07 *(Superseded)*: "Plan doesn’t take subagent orchestration into account, should be explicitly defined." (User). Codifies the mandatory subagent orchestration protocol (`research` -> `simplifier` -> `critic (Model="flash")`, with `ontologist` on W3C/SHACL/identity slices in [native-cocoindex-graph.md §8](native-cocoindex-graph.md#8-checkpoints)) across §8 Checkpoints so the parent agent acts strictly as a context-protected orchestrator.
- 2026-11-08 *(Superseded)*: "I think everything should default to resume, so we do not wipe databases by accident, and `--fresh` flag wipes intentionally. Adjust all flags to behave like this across the codebase" / "Think if we should switch from postfix to prefix across the board so these related files get sorted together in file browser: `optimize_program_telemetry`, `optimize_program.json`, `optimize_program_eval.json`, `optimize_program_audit.json`, `optimize_program_pools.json`, `optimize_status.json`, `optimize_telemetry.json`, `optimize_yaw.db`, `optimize_yaw.db.lock` -> this looks much nicer and well organized when sorted by alphabet" / "Think if we can use cocoindex fingerprinting capabilities instead of handrolling our own." (User). Establishes the canonical `optimize_*` sandbox in `.yaw/` with default-resume on `yaw dspy optimize` and explicit `--fresh` to wipe `.yaw/optimize_*` intentionally (`--resume` retired); unifies both `YawRuntime._build_coco_app` and `bind_native_indexing` onto `<base_dir>/coco_lmdb` (`.yaw/coco_lmdb`) and `<base_dir>/optimize_coco/<dataset>`; replaces hand-rolled `json.dumps` + `hashlib.sha256` predictor/stage fingerprint helpers with CocoIndex's `fingerprint_object(payload).hex()` via `graph_plan_fingerprints`; and retires `_proposer_telemetry.json` while renaming `_graph_rlm_audit.json` to `_audit.json` (`.yaw/optimize_program_audit.json`).
- 2026-11-09 *(Superseded)*: "Review our cocoindex implementation against cocoindex skill and docs, are we inventing our own orchestration, caching, fingerprinting etc when cocoindex already provides it? Make a comprehensive audit." / "Please /spec these changes." (User). Remediates CocoIndex Anti-Patterns 1–5 in this spec (and Anti-Pattern 6 in [native-cocoindex-graph.md](native-cocoindex-graph.md)): (1) deletes `partition_frozen_chunks` (`yaw/daemon/indexing.py:424–478` at `28a1113`) and keys `meta['graph_input_fingerprints']` on `fingerprint_object((settings, pointer, text)).hex()` so micro-batch admission checks `SELECT chunk_id FROM chunks` + `meta['graph_input_fingerprints']` with `0` `[target]` SQL joins on `source_chunks` or `chunk_inference`; (2) decorates mounted `declare_source_file` and `process_chunk` with `@coco.fn(memo=True, version=1)` (and `publish_graph` with `@coco.fn(memo=True, version=2)`) so unchanged mounted files and chunks short-circuit in Rust `ComponentProcessor` while preserving `TABLES.target_state` declarations; (3) deletes `YawRuntime._release_idle_coco_env` (`runtime.py:285–297` at `28a1113`) and maintains a shared process-level `coco.Environment` per `resolved_state_root` (`acquire_coco_environment` / `close_coco_environment`); (4) updates `chunk_file_memo` (`@coco.fn(memo=True, version=2)`) and `declare_source_file` to return `tuple[tuple[SourceMetadata, str], ...]`, eliminating the second file read/re-chunking and eliminating the throwaway `source_snapshot.db` + post-hoc file re-chunking loop in `snapshot_dataset_bundle`; and (5) maintains an incremental `CorpusIndexingLiveComponent._active_files` dictionary updated per-key by `_CorpusFeedSubscriber.update(key, value)` and `.delete(key)` without full-corpus directory re-walks.
- 2026-11-09 *(Superseded)*: "Why exactly do we have the distinction between \"yaw_indexing\" and \"yaw_frozen_indexing\" - what's the point with having both instead of only \"yaw_indexing\"?" / "But why on earth is it indexing in memory instead of from the disk? Sounds super complex just to point the indexing pipeline at two different folders, production-ingestion/** and optimization-ingestion/**, and what happens when we want to optimize from the actual production corpus with GEPA?" / "Basically, the way I'd like to see this work, is that when uv run yaw dspy optimize is called, an optimize cocoindex app is launched, pointed into whatever was set up as the corpus, using the exact same pipeline as the production ingestion pipeline. The only difference, imo, should be the GEPA optimization and the corpus directory, to avoid polluting the production database... that's what the sandbox is supposed to mean." / "The point was to simply remove the hand rolled parallel ingestion path, and point the pipeline to a separate Target.optimize = F(Source.optimize) instead of Target.production = F(Source.production). The F stays the same." / "I don't understand this either. Why is there a separate benchmarking ingestion path?" (User). Retires the parallel in-memory frozen ingestion pipeline (`frozen_indexing_app_main`, `process_frozen_chunk`, `frozen_chunk_metadata`, `partition_frozen_chunks`, `FrozenChunkStream`, `"yaw_frozen_indexing"`) and separate benchmark bundle ingestion app (`_bundle_snapshot_app_main`, `"yaw_bundle_source"`, `BUNDLE_SOURCE_SETTINGS`, `snapshot_dataset_bundle`) completely, unifying production ($\text{Target}_{\text{production}} = F(\text{Source}_{\text{production}})$), GEPA optimization ($\text{Target}_{\text{optimize}} = F(\text{Source}_{\text{optimize}})$), and benchmarking ($\text{Target}_{\text{benchmark}} = F(\text{Source}_{\text{benchmark}})$) onto the single `indexing_app_main` (`"yaw_indexing"`) pipeline (`localfs.walk_dir(SOURCE_ROOT)` -> `CorpusIndexingLiveComponent` -> `declare_source_file` -> `64`-chunk `[target]` micro-batching -> `process_chunk` -> `publish_graph`).

### Embedding deployment identity (2026-11-13 wiring audit) *(Superseded)*

`DSPyEmbeddingAdapter.index_identity` is the sole indexing identity for vectors:
it fingerprints the document-side profile contract, actual configured endpoint
and provider, and `max_batch_chars` (which changes splitting/pooling). Credentials
and query-only prefixes do not identify document vectors. This configured
contract is not evidence of remote weights. `EmbeddingRequest` carries it in the
CocoIndex memo key; endpoints validate it before execution. Chunk inference,
graph planning, resume hydration and reports use that exact value. A server or
splitting-contract change must re-embed unchanged sources; same-contract replay
must retain memo hits. Synthetic integration providers expose an explicit
identity rather than relying on a production profile-only fallback.
