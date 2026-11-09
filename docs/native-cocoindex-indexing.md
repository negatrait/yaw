# Native CocoIndex indexing and live ingestion

Status: active, 2026-11-09. Pinned code: `28a1113`.

| Relation | Links |
| --- | --- |
| Governed by | [FOUNDATIONAL.md](../FOUNDATIONAL.md), [AGENTS.md](../AGENTS.md), [epic-1-hipporag-daemon.md](epic-1-hipporag-daemon.md) |
| Rules | [daemon.md](../.agents/rules/daemon.md): `yaw/daemon/*.py`, `yaw/daemon/ingestion/*.py`; [graph-storage.md](../.agents/rules/graph-storage.md): `yaw/daemon/graph/*.py`, `yaw/daemon/graph/storage/*.py`; [inference.md](../.agents/rules/inference.md): `yaw/daemon/graph/inference/*.py`; [tests.md](../.agents/rules/tests.md): `tests/**/*.py`; [docs.md](../.agents/rules/docs.md): `docs/**/*.md` |
| Owns code | `yaw/daemon/runtime.py`, `yaw/daemon/indexing.py`, `yaw/daemon/ingestion/corpus.py`, `yaw/daemon/ingestion/source_targets.py`, `yaw/daemon/ingestion/datasets.py`, `yaw/daemon/ingestion/bundles.py`, `yaw/daemon/graph/inference/components.py:extract_text`, `yaw/daemon/graph/inference/components.py:embed_text` |
| Peers | [native-cocoindex-graph.md](native-cocoindex-graph.md): owns `plan_graph_publication`, `define_missing_entities`, `judge_synonyms`, `heal_shacl_violations`, `hydration.py`, `chunks.py`, `ManagedTableTarget` / `DatabaseSink` / `vec0` target storage, `CandidateStageCache` retirement, and the paired `64 + 64` vs. `128` `[target]` acceptance gate; [w3c-improvements.md](w3c-improvements.md): owns `synonyms.py` and `canonicalize_entity_labels_edc` (`§3.2`) and constrains SKOS/OWL/SHACL closed-world shapes; [epistemic-confidence-spec.md](epistemic-confidence-spec.md): constrains Bayesian confidence pooling; [dspy-improvements.md](dspy-improvements.md): constrains GBNF adapter and RLM healer architecture; [enterprise-rag-bench-training-plan.md](enterprise-rag-bench-training-plan.md): constrains stratified `135/27` `[target]` training/validation split; [Epic 3](epic-3-performance.md): owns post-Epic 2 `msgspec` sidecar serialization and full-corpus scaling over `64`-chunk `[target]` micro-batches |
| References | [arxiv-2502.14802v1-hipporag2.md](arxiv-2502.14802v1-hipporag2.md): anchors HippoRAG 2 indexing and retrieval parity; [model-evaluation.md](model-evaluation.md): anchors local `llama.cpp` serving constraints (`parallel = 1` `[target]`) |
| Supersedes / superseded by | Supersedes earlier `skos:ConceptScheme` indexing hooks, per-database `candidate_coco/<db_name>` LMDB isolation (`50be17d`), discarded `EmbeddedText` in `process_chunk` (`50be17d`), SQLite `source_chunks JOIN chunk_inference` planner polling (`50be17d`), `partition_frozen_chunks` 3-table SQLite joins (`28a1113`), mutable `FrozenChunkStream` and un-memoized `process_file`/`process_frozen_chunk` component mounts (`28a1113`), ephemeral per-call `coco.Environment` thrashing and `_release_idle_coco_env` (`28a1113`), throwaway `source_snapshot.db` + double file chunking in `snapshot_dataset_bundle` and `process_file` (`28a1113`), and monolithic `self._files.items()` live directory re-scans in `CorpusIndexingLiveComponent` (`28a1113`) |
| Records | [incremental-publication-work.md](reviews/incremental-publication-work.md), [shacl-healer-publication.md](reviews/shacl-healer-publication.md), [resource-vocabulary-boundary.md](reviews/resource-vocabulary-boundary.md), [evaluation-ingestion.md](evaluation-ingestion.md), [evaluation-graph-storage.md](evaluation-graph-storage.md) |

## 1. Scope

In scope: Documentation contract for native CocoIndex source ingestion, `64`-chunk `[target]` micro-batch admission via `FROZEN_CHUNKS` (`coco.ContextKey`), persistent shared LMDB memoization (`coco.Settings(db_path=...)`) with process-level `coco.Environment` reuse (`acquire_coco_environment` / `close_coco_environment`), `@coco.fn(memo=True, version=1)` component memoization on mounted components (`process_file`, `process_frozen_chunk`; keeping plain-awaited `process_chunk` as `@coco.fn` without `memo=True`), single-pass `(SourceMetadata, str)` file chunking memoization (`_chunk_file_memo` `version=2`) without throwaway `source_snapshot.db`, incremental `CorpusIndexingLiveComponent._active_files` live event updates, per-database component/target-state isolation, direct component value handoff (`process_chunk` -> `publish_graph`), provider admission boundaries, and ordered runtime lifecycle across `yaw/daemon/runtime.py`, `yaw/daemon/indexing.py`, `yaw/daemon/ingestion/*.py`, and `yaw/daemon/graph/inference/components.py` (`extract_text`, `embed_text`).

Non-goals: Graph identity canonicalization, missing-entity definition (`define_missing_entities`), synonym candidate mining and adjudication (`judge_synonyms`), SHACL repair semantics (`heal_shacl_violations`), `ManagedTableTarget.delete_row` and `vec0` virtual table DDL/DML, 1-indexed disk hydration (`hydration.py`), PPR retrieval, and the numeric `64 + 64` vs. `128` `[target]` graph parity definitions (all owned by [native-cocoindex-graph.md](native-cocoindex-graph.md)); `skos:ConceptScheme` clustering or tables (permanently removed); live process restarts during this specification turn.

Depends on decisions: [2026-11-06: Step 4 atomic units](#decisions), [2026-11-07: delta-scoped 64-chunk batch publication](#decisions), [2026-11-07: direct pre-hydrated SHACL healing & resume partitioning](#decisions), [2026-11-07: remedy gaps 1–5 across indexing and graph specs](#decisions), [2026-11-08: canonical optimize_* sandbox & fingerprint_object](#decisions), [2026-11-09: remediate CocoIndex anti-patterns 1–6](#decisions), [native-cocoindex-graph.md decisions](native-cocoindex-graph.md#decisions).

## 2. Problem, with evidence

Following the resolution of `50be17d` Gaps 1–5 (recorded in [incremental-publication-work.md](reviews/incremental-publication-work.md)), audit of pinned commit `28a1113` against `cocoindex == 1.1.0` `[measured: uv.lock:330-331, 28a1113]` identified `5` `[observed: yaw/daemon/indexing.py, yaw/daemon/runtime.py, yaw/daemon/ingestion/source_targets.py, yaw/daemon/ingestion/bundles.py, 28a1113]` remaining indexing and ingestion anti-patterns (plus Anti-Pattern 6 in [native-cocoindex-graph.md §2](native-cocoindex-graph.md#2-problem-with-evidence)):

- **Anti-Pattern 1 (Shadow Change Detection — `partition_frozen_chunks`)**: `yaw/daemon/indexing.py:partition_frozen_chunks` (`L424–478` at `28a1113`) executes raw `3`-table `[observed: yaw/daemon/indexing.py:452-464, 28a1113]` SQLite joins (`source_chunks s JOIN chunk_inference ci JOIN chunks c`) in `64`-ID `[observed: yaw/daemon/indexing.py:449, 28a1113]` chunks and recomputes `fingerprint_object((settings, pointer, text, entry[1])).hex()` (`L474` at `28a1113`) in Python to split chunks into `retained` vs `changed` `[observed: yaw/daemon/indexing.py:424-478, 28a1113]`. Because `extract_text` and `embed_text` are deterministic functions of `(text, settings)`, joining `source_chunks` and `chunk_inference` duplicates CocoIndex's own `TABLES.components` / `TABLES.memoization` change tracking.
- **Anti-Pattern 2 (Mutable `FrozenChunkStream` & Quadratic Python Component Re-mounts)**: `yaw/daemon/indexing.py:FrozenChunkStream` (`L481–488` at `28a1113`) is a mutable Python object mutated in place across `app.update()` calls (`L615, 621` at `28a1113`), while `process_file` (`L170` at `28a1113`) and `process_frozen_chunk` (`L396` at `28a1113`) are decorated with `@coco.fn` **without `memo=True`** `[observed: yaw/daemon/indexing.py:170, 396, 481-488, 615-621, 28a1113]`. In `cocoindex == 1.1.0`, mounting an un-memoized `@coco.fn` via `coco.use_mount` forces CocoIndex to re-enter the Python coroutine and re-declare `source_chunks` and `chunk_inference` target rows for every previously mounted chunk on every subsequent `64`-chunk `[target]` micro-batch (for $N$ chunks across $K = \lceil N / 64 \rceil$ batches, executing $\sum_{i=1}^{K} 64 i = 32 K (K + 1)$ `[measured: arithmetic]` Python component entries—e.g., `128 + 64 = 192` `[measured: arithmetic]` entries instead of `128` `[target]` for `128` `[target]` chunks across `2` `[target]` batches—instead of short-circuiting in Rust `ComponentProcessor`).
- **Anti-Pattern 3 (Ephemeral `coco.Environment` Thrashing & `_release_idle_coco_env`)**: `yaw/daemon/indexing.py:bind_native_indexing` (`L597–628` at `28a1113`) instantiates and closes a fresh `coco.Environment` and `coco.App("yaw_frozen_indexing", ...)` on every `engine.index_chunks()` call, and `YawRuntime._release_idle_coco_env` (`yaw/daemon/runtime.py:285–297, 329, 1130` at `28a1113`) forcibly closes idle `YawRuntime._coco_env` handles to avoid LMDB single-open-environment collisions `[observed: yaw/daemon/indexing.py:597-628, yaw/daemon/runtime.py:285-297, 329, 1130, 28a1113]`.
- **Anti-Pattern 4 (Throwaway `source_snapshot.db` & Double File Read / Re-Chunking in `snapshot_dataset_bundle` and `process_file`)**:
  - `yaw/daemon/ingestion/source_targets.py:_chunk_file_memo` (`L78–115` at `28a1113`) calls `chunk_source_text` (`L93` at `28a1113`) and discards the chunk `text` (`for pointer, _ in chunks` at `L113` at `28a1113`), returning only `tuple[SourceMetadata, ...]`.
  - `yaw/daemon/indexing.py:process_file` (`L170–213` at `28a1113`) awaits `declare_source_file` (`L184–190` at `28a1113`) and then invokes `process_chunk` with `frozen_text=None` (`L203` at `28a1113`), causing `process_chunk` (`L117–124` at `28a1113`) to call `resolve_chunk_text`, which re-reads the file from disk and runs `chunk_source_text` a second time `[observed: yaw/daemon/ingestion/source_targets.py:78-115, yaw/daemon/indexing.py:114-124, 184-208, 28a1113]`.
  - `yaw/daemon/ingestion/bundles.py:snapshot_dataset_bundle` (`L236–394` at `28a1113`) creates a throwaway SQLite file `source_snapshot.db` (`L303` at `28a1113`), mounts `source_chunks` (`L247–251` at `28a1113`), queries `SELECT ... FROM source_chunks` (`L316–323` at `28a1113`), closes `db` (`L326` at `28a1113`), and then re-reads every JSON document from disk (`L341` at `28a1113`) and calls `chunk_source_text` a second time (`L368–376` at `28a1113`) `[observed: yaw/daemon/ingestion/bundles.py:236-394, 28a1113]`.
- **Anti-Pattern 5 (Monolithic Live Re-Scan in `_CorpusFeedSubscriber` & `CorpusIndexingLiveComponent`)**: `yaw/daemon/indexing.py:_CorpusFeedSubscriber.update(key, value)` and `.delete(key)` (`L250–258` at `28a1113`) discard `key` and `value` (`_ = key, value`) and trigger `self._operator.update_full()`, which in `CorpusIndexingLiveComponent.process` (`L294` at `28a1113`) executes `[item async for item in self._files.items()]` (`DirWalker._walk_sync()`, a full synchronous directory walk) and runs un-memoized `process_file` across every file in the corpus on every single file event `[observed: yaw/daemon/indexing.py:250-258, 294-309, 28a1113]`.
- **Historical `50be17d` Gaps 1–5 (resolved in Checkpoints `I1–I4`)**:
  - **Gap 1**: `process_chunk` (`L252–331` at `50be17d`) discarded `EmbeddedText.vector_f32` while `plan_graph_publication` (`L1099–1127` at `50be17d`) called `embed_text` a second time `[observed: yaw/daemon/indexing.py:316-319, yaw/daemon/graph/planner.py:1099-1127, 50be17d]`.
  - **Gap 2**: `bind_native_indexing` (`L752` at `50be17d`) used per-DB `candidate_coco/<db_name>` LMDB paths `[observed: yaw/daemon/indexing.py:752, 50be17d]`.
  - **Gap 3**: `indexing.py` (`L101–249, 368–400, 425–524` at `50be17d`) and `planner.py:load_active_chunk_inferences` (`L203–263` at `50be17d`) polled `source_chunks JOIN chunk_inference` `[observed: yaw/daemon/indexing.py:101-249, yaw/daemon/graph/planner.py:203-263, 50be17d]`.
  - **Gap 4**: `chunk_vec` (`yaw/daemon/graph/storage/db.py:147–151` at `50be17d`) was an ordinary `BLOB` table `[observed: yaw/daemon/graph/storage/db.py:147-163, 50be17d]`.
  - **Gap 5**: `yaw/daemon/runtime.py` (`L1240–1241` at `50be17d`) queried the retired `candidate_stage_cache` table `[observed: yaw/daemon/runtime.py:1240-1241, 50be17d]`.

## 3. Goals

### Goal I1: Persistent cross-database CocoIndex memo reuse with strict target isolation
- **Measure**: Provider call count (`extract_text`, `embed_text`, `define_missing_entities`, `judge_synonyms`, `heal_shacl_violations`) and published SQLite row counts across all `15` `[target]` graph tables when indexing an identical `64`-chunk `[target]` snapshot into a fresh second SQLite database (`db_b.sqlite`) sharing `coco.Settings(db_path=shared_lmdb)` with a previously indexed database (`db_a.sqlite`).
- **Reference**: Canonical `15`-table `[target]` SQLite state of `db_a.sqlite` after initial cold indexing of the same `64` `[target]` chunks.
- **Passes when**: `db_b.sqlite` incurs `0` `[target]` provider calls (`100%` `[target]` `TABLES.memoization` hit rate for unchanged inputs) and materializes `100%` `[target]` identical rows across all `15` `[target]` graph tables plus `source_chunks` and `chunk_inference`, without mutating or withdrawing `db_a.sqlite`'s tracked target state.
- **Fails if / does not count**:
  - `state_root` is suffixed with the database filename (`candidate_coco/<db_name>`), requiring `CandidateStageCache` or `.yaw/dspy_cache` to mask LMDB misses.
  - `db_b.sqlite` hits `db_a.sqlite`'s `/publication` component memo (`TABLES.components`) and skips `declare_tables`, leaving `db_b.sqlite` empty and requiring `recovery_nonce` retry loops.
  - Fakes or mocks are cited as proof without a real multi-DB on-disk SQLite + LMDB test.

### Goal I2: Direct component value flow without SQLite polling or duplicate `embed_text` calls
- **Measure**: Number of `embed_text` calls inside `plan_graph_publication`, number of `SELECT` queries against `source_chunks` or `chunk_inference` executed inside `publish_graph`, `plan_graph_publication`, and `bind_native_indexing`, and chunk batch admission size per `app.update()` cycle.
- **Reference**: Paired `64 + 64` vs. `128` `[target]` harness defined in [native-cocoindex-graph.md §3](native-cocoindex-graph.md#3-goals).
- **Passes when**: Each `app.update()` cycle admits at most `64` `[target]` changed chunks; `process_chunk` returns typed `ChunkProcessOutput(inference, vector_f32)` via `coco.use_mount`; `bind_native_indexing`, `publish_graph` (`indexing.py`), and `plan_graph_publication` (`planner.py`) execute `0` `[target]` `SELECT` queries against `source_chunks` or `chunk_inference` and `0` `[target]` `embed_text` calls for chunk vectors; and cumulative timing/parity satisfies [native-cocoindex-graph.md §3 Goal A & Goal B](native-cocoindex-graph.md#3-goals) (`0.80 <= R_time <= 1.25` `[target]`, `0.80 <= R_graph <= 1.25` `[target]`, worker RSS `<= 1 GiB` `[target]`).
- **Fails if / does not count**:
  - `process_chunk` declares `chunk_vec` in a separate component `TargetActionSink` outside `publish_graph`'s atomic transaction, or stores duplicate `embedding` bytes in `chunk_inference`.
  - Moving `SELECT ... FROM source_chunks JOIN chunk_inference` into `indexing.py:publish_graph` or `indexing.py:partition_frozen_chunks` instead of passing `ChunkProcessOutput` in memory from `process_chunk` and checking `chunks` + `meta['graph_input_fingerprints']`.
  - Re-reading/re-chunking disk files via `resolve_chunk_text` during batch admission fingerprint checks.
  - A smaller prefix of an already-indexed snapshot withdraws previously published chunks.

### Goal I3: Component-level Rust memo short-circuiting, shared `coco.Environment` reuse, single-read file chunking, and incremental live updates
- **Measure**: Python coroutine execution counts of `process_frozen_chunk`, `process_chunk`, and `process_file` across multi-batch frozen indexing and single-file live updates; number of `source_snapshot.db` files created and `chunk_source_text` calls per file in `snapshot_dataset_bundle` and `process_file`; and number of `coco.Environment` instances opened per `resolved_state_root`.
- **Reference**: Cold initial execution of each chunk/file in Batch 1 (`1` `[target]` Python execution per new chunk/file).
- **Passes when**:
  - In a `2`-batch `[target]` frozen indexing run, Batch 1 chunks incur `0` `[target]` Python executions of `process_frozen_chunk` or `process_chunk` during Batch 2 (short-circuiting in Rust `ComponentProcessor` via `@coco.fn(memo=True, version=1)` on mounted `process_frozen_chunk` while preserving their `source_chunks` and `chunk_inference` target rows).
  - `snapshot_dataset_bundle` creates `0` `[target]` `source_snapshot.db` files and calls `chunk_source_text` at most `1` `[target]` time per cold file (`0` `[target]` times on warm memo), and `process_file` calls `chunk_source_text` `0` `[target]` times outside `_chunk_file_memo` (`0` `[target]` `resolve_chunk_text` calls).
  - `acquire_coco_environment(state_root)` maintains `1` `[target]` open `coco.Environment` per `resolved_state_root` across `YawRuntime` and `bind_native_indexing` without closing idle runtime apps.
  - On a live single-file create/modify/delete event in `_CorpusFeedSubscriber`, `CorpusIndexingLiveComponent` performs `0` `[target]` `self._files.items()` full directory walks and `0` `[target]` Python `process_file` executions on untouched sibling files.
- **Fails if / does not count**:
  - Keeping `FrozenChunkStream`, `partition_frozen_chunks`, or `_release_idle_coco_env` behind flags or default arguments instead of deleting them (§7).
  - Passing only `changed[:64]` to `FROZEN_CHUNKS` while omitting `retained` chunks (which unmounts prior chunks and deletes their `source_chunks`/`chunk_inference` rows).
  - Decorating plain-awaited `process_chunk` with `@coco.fn(memo=True)` instead of keeping `@coco.fn` without `memo=True` (because plain `await fn(...)` on `@coco.fn(memo=True)` in `cocoindex == 1.1.0` does not replay `declare_target_state` calls when `process_file` re-runs on a partially edited file).
  - Re-reading files from disk in `snapshot_dataset_bundle` or `process_file` after `_chunk_file_memo` has already read and chunked them, or re-reading/re-chunking disk files via `resolve_chunk_text` during batch admission fingerprint checks.
  - Updating `CorpusIndexingLiveComponent._active_files` on live events without performing the initial cold directory scan on startup (`live=False`) or `_CorpusFeedSubscriber.update_all()`.

Thresholds are not loosened after seeing a run.

## 4. Semantics and invariants

<a id="3-revision-identity-memo-and-target-isolation"></a>

- **Equality and revision identity**:
  - A source occurrence revision is uniquely keyed by `(chunk_id, source_pointer, sha256_hash, program_identity, embedding_identity)`:
    - `source_pointer` uses `1`-indexed `[target]` inclusive `(start_line, end_line)` bounds and `0` `[target]` raw text columns in `source_chunks`.
    - Identical text at `2` `[target]` distinct file coordinates retains distinct `chunk_id` occurrence provenance.
    - Same-ID edits change `sha256_hash` or line bounds; ID set difference (`set(new_ids) - set(old_ids)`) is never used for change detection.
    - Because `extract_text(OpenIEInput(chunk_id, passage, settings.program_identity))` and `embed_text` are deterministic functions of `(text, settings)`, `fingerprint_object((settings, pointer, text)).hex()` uniquely identifies a chunk's graph input state in `meta['graph_input_fingerprints']` without reading `source_chunks` or `chunk_inference`.
    - `program_identity` (extraction) and `embedding_identity` (embedding) remain separate columns on `ChunkInference` so changing an embedder does not invalidate `extract_text` memos, and changing an extraction prompt does not invalidate `embed_text` memos. Predictor and stage fingerprints (`judge_stage_fingerprint`, `embed_fingerprint`, `shacl_fingerprint`, `openie_fingerprint`, `extraction_fingerprint`, `candidate_fingerprint`, `query_retrieval_fingerprint`) are computed with CocoIndex's Rust-backed `fingerprint_object(payload).hex()` (`cocoindex.connectorkits.fingerprint`), and the 4-tuple `(program_identity, embedding_identity, synonym_identity, shacl_identity)` is assembled by `graph_plan_fingerprints(program, embedder)` in `yaw/daemon/indexing.py`.
    - The chunk embedding has exactly `1` `[target]` persisted `float32` SQLite copy (in `chunk_vec`), never a second `embedding` column on `chunk_inference` or pickled bytes in a component memo key.
- **Ownership**:

| State | Single owner |
| --- | --- |
| Process-level `coco.Environment` & `coco.ContextProvider` per LMDB path | Shared registry in `yaw/daemon/indexing.py` (`acquire_coco_environment` / `close_coco_environment`), maintaining `1` `[target]` open `(coco.Environment, coco.ContextProvider)` per canonical `<base_dir>/coco_lmdb` (default `.yaw/coco_lmdb`, `0` `[target]` `.yaw/coco_state*` directories) shared by `"yaw_indexing"`, `"yaw_frozen_indexing"`, and `"yaw_bundle_source"`. |
| Persistent `@coco.fn(memo=True)` cache (`TABLES.memoization`: `_chunk_file_memo`, `process_file`, `process_frozen_chunk`, `publish_graph`, `extract_text`, `embed_text`, `define_missing_entities`, `judge_synonyms`, `heal_shacl_violations`) | Shared persistent CocoIndex LMDB environment (`<base_dir>/coco_lmdb`, and `<base_dir>/optimize_coco/<dataset>/coco_lmdb` for dataset bundle snapshots) keyed by `(app_name, parent_ctx.stable_path, memo_fp)`. |
| Per-database component mount state (`TABLES.components`) and target row tracking (`TABLES.target_state`) | CocoIndex LMDB environment scoped by `db_identity` (`<db_canonical_path>:<db_header_nonce>`), isolating each `yaw.db` file's applied target state while keeping component `StablePath`s DB-invariant. |
| `source_chunks` and `chunk_inference` tables in `yaw.db` | Declarative per-chunk targets in `yaw/daemon/indexing.py` (`process_chunk` / `process_frozen_chunk`) mounted via `managed_sqlite.py:mount_table(..., db_identity=db_identity)`, used strictly for provenance, GEPA trace hydration (`engine.py`), and `scripts/watchdog.py` inspection—never read by `bind_native_indexing`, `publish_graph`, or `planner.py`. |
| `publish_graph` component mount (`/publication`) | `yaw/daemon/indexing.py:publish_graph`, which delegates planning to `plan_graph_publication` and target row declarations for all `15` `[target]` graph tables to `declare_tables` (both owned by [native-cocoindex-graph.md](native-cocoindex-graph.md)). |

- **Boundaries**:
  - **Empty input (`chunks = []`)**: Treated as complete corpus withdrawal when explicitly passed as the full desired snapshot; retires all published chunks atomically.
  - **Partial batch admission (`64`-chunk `[target]` micro-batches)**: Admitting `changed[:64]` (`<= 64` `[target]` chunks) alongside `retained` via `context.provide(FROZEN_CHUNKS, active_slice)` retains all previously declared unchanged chunks in the snapshot envelope; a partial batch never withdraws unseen valid chunks.
  - **Missing frozen text (`text is None` or empty)**: Raises `ValueError` immediately; never falls back to `supplied_text or hydrate_from_disk(...)`.
  - **Provider Admission & Retries**: Default LLM concurrency is `Semaphore(1)` (`parallel = 1` `[target]`) for local `llama.cpp`; embedding batches contain at most `64` `[target]` inputs (`max_batch_size = 64` `[target]`) within configured context limits; sampling temperature follows the configured model profile (`1.0` `[target]` in development per [AGENTS.md](../AGENTS.md) to prevent small-model repetition loops; `>= 0.3` `[target]` minimum, reflection `0.7–0.8` `[target]` per [FOUNDATIONAL.md §4.4](../FOUNDATIONAL.md)). Transient model errors retry only via `dspy.LM(num_retries=...)`; exhausted retries raise the original exception (`FatalGraphPublicationError` subclasses `BaseException`).
  - **Crash & Replay**: CocoIndex LMDB and `yaw.db` commit independently. If a process crashes after `extract_text`/`embed_text` memos commit to LMDB but before `publish_graph` commits to `yaw.db`, default-resume detects that `chunks` or `meta['graph_input_fingerprints']` lacks those chunks, places them in `changed`, and replays them in `64`-chunk `[target]` slices while hitting `TABLES.memoization` (`0` `[target]` duplicate provider calls).
  - **Shutdown**: Ordered lifecycle stops source admission, drains in-flight provider/publication tasks, closes live handles, and closes the shared CocoIndex `Environment` (`close_coco_environment`) and SQLite connections.

## 5. Design (current state)

Decided:

```python
FROZEN_CHUNKS = coco.ContextKey[tuple[tuple[ChunkPointer, str], ...]]("yaw/frozen_chunks")
FROZEN_GRAPH_SETTINGS = coco.ContextKey[GraphPlanSettings | None]("yaw/frozen_graph_settings")
FROZEN_INFERENCE_IDENTITY = coco.ContextKey[InferenceIdentity | None]("yaw/frozen_inference_identity")


@dataclass
class _SharedCocoRuntime:
    environment: coco.Environment
    context: coco.ContextProvider
    apps: dict[str, coco.App[Any, Any]]


@dataclass(frozen=True)
class ChunkInference:
    chunk_id: str
    source_hash: str
    program_identity: str
    embedding_identity: str
    extraction_json: str
    traces_json: str


@dataclass(frozen=True, slots=True)
class ChunkProcessOutput:
    pointer: ChunkPointer
    text: str
    inference: ChunkInference
    embedding: bytes  # returned in-memory from embed_text; never stored in chunk_inference

    @property
    def chunk_id(self) -> str:
        return self.pointer.chunk_id

    @property
    def vector_f32(self) -> bytes:
        return self.embedding


def resolve_coco_state_root(state_dir: Path | str) -> Path:
    """Return the single persistent CocoIndex LMDB directory (<base_dir>/coco_lmdb).
    Never append transient per-candidate SQLite filenames or create coco_state* subdirectories.
    """


def acquire_coco_environment(state_root: Path | str) -> _SharedCocoRuntime:
    """Return the shared process-level _SharedCocoRuntime(environment, context, apps) for
    resolved_state_root, creating it on first access so multiple CocoIndex apps share one
    open LMDB handle and reuse registered app instances by name.
    """


def close_coco_environment(state_root: Path | str | None = None) -> None:
    """Close and evict the shared _SharedCocoRuntime for state_root (or all open environments when None)."""


def _partition_published_chunks(
    db: ManagedConnection,
    chunks: Sequence[tuple[ChunkPointer, str]],
    settings: GraphPlanSettings,
) -> tuple[list[tuple[ChunkPointer, str]], list[tuple[ChunkPointer, str]]]:
    """Partition chunks into (retained, changed) using only SELECT chunk_id FROM chunks and
    meta['graph_input_fingerprints'] (fingerprint_object((settings, pointer, text)).hex()),
    with zero SQL queries on source_chunks or chunk_inference.
    """


def graph_plan_fingerprints(program: YawProgram, embedder: DSPyEmbeddingAdapter) -> tuple[str, str, str, str]:
    """Return (program_identity, embedding_identity, synonym_identity, shacl_identity)
    computed via CocoIndex's fingerprint_object(payload).hex().
    """


@coco.fn(memo=True, version=2)
async def _chunk_file_memo(file: FileLike, settings: SourceSettings) -> tuple[tuple[SourceMetadata, str], ...]:
    """Read file once, compute chunk_source_text, and memoize (SourceMetadata, chunk_text) pairs in coco_lmdb."""


@coco.fn
async def declare_source_file(
    file: FileLike,
    table: sqlite.TableTarget[SourceMetadata],
    settings: SourceSettings,
) -> tuple[tuple[SourceMetadata, str], ...]:
    """Declare SourceMetadata rows to source_chunks and return (SourceMetadata, chunk_text) pairs."""


@coco.fn
async def _bundle_snapshot_app_main(
    settings: SourceSettings,
) -> dict[str, tuple[tuple[SourceMetadata, str], ...]]:
    """Walk bundle documents_root and return per-file (SourceMetadata, chunk_text) tuples directly from _chunk_file_memo
    without mounting SQLite tables or creating source_snapshot.db.
    """


@coco.fn
async def process_chunk(
    metadata: SourceMetadata,
    target: sqlite.TableTarget[ChunkInference],
    identity: InferenceIdentity,
    settings: SourceSettings | None = None,
    frozen_text: str | None = None,
    db_identity: str = "",
) -> ChunkProcessOutput:
    """Run memoized extract_text and embed_text in parallel (hydrating AST parent context when
    settings is not None and pointer.start_line > 1); declare ChunkInference to chunk_inference;
    and return ChunkProcessOutput (with vector_f32) to the caller. Kept as @coco.fn (without memo=True)
    because it is plain-awaited inside mounted process_frozen_chunk and process_file components.
    """


@coco.fn(memo=True, version=1)
async def process_file(
    file: FileLike,
    sources: sqlite.TableTarget[SourceMetadata],
    artifacts: sqlite.TableTarget[ChunkInference],
    settings: SourceSettings,
    identity: InferenceIdentity,
    db_identity: str = "",
) -> tuple[ChunkProcessOutput, ...]:
    """Mount declare_source_file and pass each returned (metadata, text) directly to process_chunk
    with frozen_text=text (zero duplicate file.read_text or chunk_source_text calls).
    """


@coco.fn(memo=True, version=1)
async def process_frozen_chunk(
    chunk: tuple[ChunkPointer, str],
    sources: sqlite.TableTarget[SourceMetadata],
    artifacts: sqlite.TableTarget[ChunkInference],
    identity: InferenceIdentity,
    db_identity: str = "",
) -> ChunkProcessOutput:
    """Declare SourceMetadata to source_chunks and return
    await process_chunk(metadata, artifacts, identity, frozen_text=text, db_identity=db_identity).
    """


@coco.fn
async def frozen_indexing_app_main(
    settings: GraphPlanSettings | None = None,
    identity: InferenceIdentity | None = None,
) -> GraphPlanSummary | None:
    """Read active_chunks from coco.use_context(FROZEN_CHUNKS) (and fall back to
    coco.use_context(FROZEN_GRAPH_SETTINGS) / coco.use_context(FROZEN_INFERENCE_IDENTITY) when
    settings or identity are None), mount process_frozen_chunk per chunk via coco.use_mount,
    and mount publish_graph with the resulting tuple[ChunkProcessOutput, ...].
    """


@coco.fn(memo=True, version=2)
async def publish_graph(
    settings: GraphPlanSettings,
    chunk_outputs: tuple[ChunkProcessOutput, ...],
    db_identity: str = "",
) -> GraphPlanSummary:
    """Single atomic graph publication component mounted via coco.use_mount at DB-invariant
    coco.component_subpath('publication'). Delegates to plan_graph_publication(settings, chunk_outputs, db_identity=db_identity)
    and declare_tables(plan.tables, db_identity=db_identity) over shared_sink, returning GraphPlanSummary.
    """
```

- **Shared `coco.Environment` & App Registry (`acquire_coco_environment`, `bind_native_indexing`)**:
  - `EnvironmentInfo.register_app` in `cocoindex == 1.1.0` (`cocoindex/_internal/environment.py:180–183`) raises `ValueError` if two `coco.App` instances with the same `name` are registered on the same `coco.Environment`. Therefore, `_SharedCocoRuntime` caches `apps: dict[str, coco.App[Any, Any]]` (`"yaw_frozen_indexing"`, `"yaw_bundle_source"`, `"yaw_indexing"`) per canonical `resolved_state_root: Path`.
  - In `bind_native_indexing`, `execute(chunks, program)` acquires `shared = acquire_coco_environment(resolved_state_root)`, updates `shared.context.provide(...)` (including `FROZEN_GRAPH_SETTINGS` and `FROZEN_INFERENCE_IDENTITY` so candidate `program` changes in GEPA take effect without recreating `coco.App`), and reuses `shared.apps["yaw_frozen_indexing"]`.
  - Before micro-batching, `_partition_published_chunks(engine.managed_conn, chunks, settings)` reads `SELECT chunk_id FROM chunks` and `meta['graph_input_fingerprints']` from `yaw.db` (`0` `[target]` queries on `source_chunks` or `chunk_inference`). A chunk `(pointer, text)` is placed in `retained` iff `pointer.chunk_id` is present in `chunks` and `published_inputs.get(pointer.chunk_id) == fingerprint_object((settings, pointer, text)).hex()`; otherwise it is placed in `changed`.
  - Each `app.update()` cycle provides `shared.context.provide(FROZEN_CHUNKS, (*retained, *changed[: start + batch_size]))` (`<= 64` `[target]` new chunks per batch) and mounts each chunk at DB-invariant `coco.component_subpath("chunk", pointer.chunk_id)` via `coco.use_mount(..., process_frozen_chunk, ...)`. Because `process_frozen_chunk` is `@coco.fn(memo=True, version=1)` and mounted via `coco.use_mount` (while plain-awaited `process_chunk` remains `@coco.fn` without `memo=True`, since plain `await fn(...)` on `@coco.fn(memo=True)` in `cocoindex == 1.1.0` does not replay `declare_target_state` calls), CocoIndex's Rust `ComponentProcessor` matches `memo_fp` in `TABLES.components` for all `retained` chunks from earlier batches, short-circuiting in Rust without entering Python `process_frozen_chunk` or `process_chunk` while preserving their `TABLES.target_state` row declarations and returning cached `ChunkProcessOutput` values to `publish_graph`.
- **Single-read dataset bundle snapshotting (`snapshot_dataset_bundle`)**:
  - `_bundle_snapshot_app_main` walks `SOURCE_ROOT` and calls `_chunk_file_memo(file, normalize_chunking_settings(settings))` per matched file, returning `dict[str, tuple[tuple[SourceMetadata, str], ...]]` directly from `await app.update().result()` over the shared `acquire_coco_environment(resolved_state / "coco_lmdb")`. `snapshot_dataset_bundle` constructs `list[tuple[ChunkPointer, str]]` directly from those memoized `(SourceMetadata, str)` tuples without creating `source_snapshot.db` or re-reading/re-chunking files from disk.
- **Incremental live corpus handoff (`CorpusIndexingLiveComponent` & `_CorpusFeedSubscriber`)**:
  - `CorpusIndexingLiveComponent.__init__` initializes `self._active_files: dict[str, FileLike] | None = None`.
  - `_CorpusFeedSubscriber.update_all()` sets `self._component._active_files = None` and awaits `self._operator.update_full()` under `self._lock`.
  - `_CorpusFeedSubscriber.update(key, value)` sets `self._component._active_files[str(key)] = value` in $O(1)$ (when `self._component._active_files is not None`) and awaits `self._operator.update_full()` under `self._lock`.
  - `_CorpusFeedSubscriber.delete(key)` calls `self._component._active_files.pop(str(key), None)` in $O(1)$ (when `self._component._active_files is not None`) and awaits `self._operator.update_full()` under `self._lock`.
  - `CorpusIndexingLiveComponent.process()` populates `self._active_files = {str(rel): f async for rel, f in self._files.items()}` only when `self._active_files is None`, and mounts `process_file` (`@coco.fn(memo=True, version=1)`) for each `(rel_path, file)` in `sorted(self._active_files.items())`. Unchanged files hit `FileLike.__coco_memo_state__` (`current_mtime == prev_mtime`) and short-circuit in Rust `ComponentProcessor` without disk reads or Python `process_file` execution, while deleted keys omitted from `self._active_files` are automatically unmounted and reconciled by CocoIndex.

Open: None.

## 6. Capability gates

| Behavior relied on | Locked version | Probe | Pass criterion | Stop rule |
| --- | --- | --- | --- | --- |
| `CG-INDEX-1`: Shared `coco.Settings(db_path=...)` LMDB reuses `@coco.fn(memo=True)` memos across `2` `[target]` distinct SQLite DBs when component mounts and target states are keyed by `db_identity` | `cocoindex == 1.1.0` `[measured: uv.lock:330-331, 28a1113]` | `tests/daemon/graph/test_indexing_overlap.py::test_shared_lmdb_cross_db_memo_reuse_and_target_isolation` | Indexing `64` `[target]` chunks into `db_a.sqlite` then indexing an overlapping `64`-chunk `[target]` set into fresh `db_b.sqlite` sharing `1` `[target]` `state_root` makes `0` `[target]` duplicate `extract_text`/`embed_text` calls for overlapping chunks and populates all `15` `[target]` graph tables in both DBs without withdrawing `db_a.sqlite` rows | Stop and amend this spec with the evidence |
| `CG-INDEX-2`: `coco.use_mount` returns typed `ChunkProcessOutput` payloads directly to `publish_graph` in both frozen (`frozen_indexing_app_main`) and live (`indexing_app_main`) flows without SQLite table polling | `cocoindex == 1.1.0` `[measured: uv.lock:330-331, 28a1113]` | `tests/daemon/graph/test_indexing_overlap.py::test_direct_use_mount_chunk_output_handoff_without_chunk_state_target` | `publish_graph` receives `tuple[ChunkProcessOutput, ...]` (including `vector_f32`) directly in memory; `0` `[target]` `SELECT` queries run against `source_chunks` or `chunk_inference` inside `publish_graph` or `planner.py` | Stop and amend this spec with the evidence |
| `CG-INDEX-3`: `@coco.fn(memo=True, version=1)` on mounted `process_frozen_chunk` and `process_file` (with plain-awaited `process_chunk` kept as `@coco.fn` without `memo=True`) short-circuits in Rust `ComponentProcessor` on unchanged prior-batch chunks (`FROZEN_CHUNKS` `coco.ContextKey`) and untouched live files while retaining `TABLES.target_state` ownership across shared `acquire_coco_environment` apps | `cocoindex == 1.1.0` `[measured: uv.lock:330-331, 28a1113]` | `tests/daemon/graph/test_indexing_overlap.py` & `tests/daemon/ingestion/test_datasets.py` `[measured: inline CocoIndex 1.1.0 substrate probe, 28a1113: Batch 1 (c0, c1) -> ['inner:c0', 'inner:c1', 'mounted:c0', 'mounted:c1'], Batch 2 (c0, c1, c2, c3) -> ['inner:c2', 'inner:c3', 'mounted:c2', 'mounted:c3'] (0 Python re-entries for c0, c1) with COUNT(*) == 4 in both target tables]` | Batch 1 chunks incur `0` `[target]` Python `process_frozen_chunk`/`process_chunk` executions during Batch 2; untouched live sibling files incur `0` `[target]` Python `process_file` executions on single-file live events; `snapshot_dataset_bundle` creates `0` `[target]` `source_snapshot.db` files and calls `chunk_source_text` `0` `[target]` times on warm memo | Stop and amend this spec with the evidence |

## 7. Removal inventory

| Retired item | Location | Replacement or "none" | Exit check |
| --- | --- | --- | --- |
| Per-DB `candidate_coco/<db_name>` LMDB path suffix | `yaw/daemon/indexing.py:bind_native_indexing` (`L752` at `50be17d`) | `resolve_coco_state_root` (`<state_dir>/coco_lmdb`) + `db_identity`-scoped mounts and `TableKey` | `git grep -n -e "candidate_coco" yaw/ scripts/ tests/` has `0` `[target]` active production matches |
| `compute_active_inference_fingerprint` SQLite join | `yaw/daemon/indexing.py:compute_active_inference_fingerprint` (`L368–400` at `50be17d`) | Direct `tuple[ChunkProcessOutput, ...]` argument to `publish_graph` via `coco.use_mount` | `git grep -n -e "compute_active_inference_fingerprint" yaw/ scripts/ tests/` has `0` `[target]` active production matches |
| `ChunkStateTargetHandler`, `CHUNK_STATE_TARGET`, `declare_chunk_state` | `yaw/daemon/indexing.py:205–249, 330–337` (`50be17d`) | Direct `ChunkProcessOutput` return value from `process_chunk` via `coco.use_mount` | `git grep -n -e "ChunkStateTargetHandler" -e "CHUNK_STATE_TARGET" -e "declare_chunk_state" yaw/ scripts/ tests/` has `0` `[target]` active production matches |
| `GraphDeltaCoordinator`, `_SubscriberProxy`, `_DeltaNotifyingLiveItems`, `recovery_nonce` | `yaw/daemon/indexing.py:101–202, 425–524` (`50be17d`) | Native `coco.use_mount` / `LiveComponent` reactive handoff validated by `CG-INDEX-2` | `git grep -n -e "GraphDeltaCoordinator" -e "_SubscriberProxy" -e "_DeltaNotifyingLiveItems" -e "recovery_nonce" yaw/ scripts/ tests/` has `0` `[target]` active production matches |
| Metric-upgrade `candidate_stage_cache` SQLite delete in `runtime.py` | `yaw/daemon/runtime.py:1240–1241` (`50be17d`) | `yaw.db` `meta` table score invalidation alongside `CandidateStageCache` retirement ([native-cocoindex-graph.md §7](native-cocoindex-graph.md#7-removal-inventory)) | `git grep -n -e "candidate_stage_cache" yaw/ scripts/ tests/` has `0` `[target]` active production matches |
| `partition_frozen_chunks` 3-table SQLite join & Python fingerprint helper (Anti-Pattern 1) | `yaw/daemon/indexing.py:partition_frozen_chunks` (`L424–478` at `28a1113`) | `SELECT chunk_id FROM chunks` + `meta['graph_input_fingerprints']` (`fingerprint_object((settings, pointer, text)).hex()`) in `bind_native_indexing` | `git grep -n -e "partition_frozen_chunks" yaw/ scripts/ tests/` has `0` `[target]` matches |
| `FrozenChunkStream` mutable chunk wrapper (Anti-Pattern 2) | `yaw/daemon/indexing.py:FrozenChunkStream` (`L481–488, 598` at `28a1113`) | `FROZEN_CHUNKS` (`coco.ContextKey[tuple[tuple[ChunkPointer, str], ...]]`) + `@coco.fn(memo=True, version=1)` on mounted `process_frozen_chunk` and `process_file` | `git grep -n -e "FrozenChunkStream" yaw/ scripts/ tests/` has `0` `[target]` matches |
| `YawRuntime._release_idle_coco_env` and ad-hoc env-closing registries (Anti-Pattern 3) | `yaw/daemon/runtime.py:_release_idle_coco_env` (`L285–297, 329, 1130` at `28a1113`) | Shared `acquire_coco_environment` / `close_coco_environment` registry in `yaw/daemon/indexing.py` | `git grep -n -e "_release_idle_coco_env" -e "_ActiveCocoAppRegistry" -e "register_active_coco_env_closer" -e "release_idle_coco_env" yaw/ scripts/ tests/` has `0` `[target]` matches |
| Throwaway `source_snapshot.db`, `_process_bundle_file`, and post-hoc bundle file re-chunking loop (Anti-Pattern 4) | `yaw/daemon/ingestion/bundles.py` (`L236–260, 303–394` at `28a1113`) | `_chunk_file_memo` (`@coco.fn(memo=True, version=2)`) returning `tuple[tuple[SourceMetadata, str], ...]` directly from `_bundle_snapshot_app_main` | `git grep -n -e "source_snapshot.db" -e "_process_bundle_file" yaw/ scripts/ tests/` has `0` `[target]` matches |

Consumers updated in the same change: `yaw/daemon/runtime.py`, `yaw/daemon/indexing.py`, `yaw/daemon/ingestion/source_targets.py`, `yaw/daemon/ingestion/bundles.py`, `yaw/daemon/graph/planner.py`, `yaw/daemon/graph/storage/managed_sqlite.py`, `yaw/daemon/graph/storage/rows.py`, `tests/core/test_architecture.py`, `tests/daemon/graph/test_indexing_overlap.py`, `tests/daemon/graph/test_evidence_revisions.py`, `tests/daemon/graph/test_batch_equivalence.py`, `tests/daemon/ingestion/test_datasets.py`, `tests/daemon/ingestion/test_source_targets.py`, `tests/daemon/ingestion/test_corpus.py`, `tests/daemon/test_runtime.py`.

## 8. Checkpoints

Status is tracked in [incremental-publication-work.md](reviews/incremental-publication-work.md).

### Subagent Orchestration Protocol (Context Protection)

To prevent parent-agent context bloat during implementation, the lead agent acts strictly as a lightweight orchestrator and delegates all code reading, TDD implementation, ontology/SHACL review, and post-slice verification to specialized subagents:

| Role | Subagent `TypeName` | `Model` | `Workspace` | Responsibility per Checkpoint |
| --- | --- | --- | --- | --- |
| Read-only discovery & SDK inspection | `research` | `inherit` | `inherit` | Inspects CocoIndex `1.1.0` `[measured: uv.lock:330-331, 28a1113]` internals, locates callers/tests, and returns compact `path:symbol` summaries before code edits. |
| Red-Green-Refactor TDD & deletion | `simplifier` | `inherit` | `inherit` | Authors failing real-substrate tests (`CG-INDEX-1`, `CG-INDEX-2`, `CG-INDEX-3`), implements the minimal code change (`1` `[target]` slice at a time), runs all `5` `[target]` mandatory quality gates plus §7 `git grep` exit checks, and appends evidence to [incremental-publication-work.md](reviews/incremental-publication-work.md). |
| Pre-commit architectural & shortcut audit | `critic` | `flash` | `inherit` | Audits the working tree diff after `simplifier` completes to verify `0` `[target]` masked failures, `0` `[target]` surviving retired symbols, `0` `[target]` relocated SQLite polling shortcuts, and `100%` `[target]` spec compliance before commit. |

### Ordered Slices (Gaps 1–5 from `50be17d`)

- **I1 (Gap 2 — Shared persistent `coco_lmdb` & `db_identity` target isolation)**:
  - *Orchestration*: `research` maps `mount_table` and `indexing.py` mount paths -> `simplifier` writes and passes `CG-INDEX-1` (`test_shared_lmdb_cross_db_memo_reuse_and_target_isolation`), keys `mount_table` (`managed_sqlite.py`, including `source_chunks` and `chunk_inference`) by `TableKey(db_identity, name)`, scopes component mounts by `db_identity`, and replaces `candidate_coco/<db_name>` in `indexing.py:752` (`50be17d`) with `<state_dir>/coco_lmdb` -> `critic (Model="flash")` verifies cross-DB target isolation and §7 exit checks.
  - *Artifact & exit gate*: `outputs/benchmarks/cg_index_1_memo_isolation.xml`; `uv run pytest tests/daemon/graph/test_indexing_overlap.py -k shared_lmdb` passes with `0` `[target]` duplicate provider calls on the second DB and `0` `[target]` matches for `candidate_coco` in `yaw/`.
- **I2 (Gap 1 & Gap 3 — Direct `ChunkProcessOutput` `use_mount` handoff & zero `embed_text` replay)**:
  - *Orchestration*: `research` audits `LiveItems` vs. `use_mount` call sites -> `simplifier` writes and passes `CG-INDEX-2` (`test_direct_use_mount_chunk_inference_handoff`), switches `process_frozen_chunk`/`process_chunk` mounts to `coco.use_mount` returning `ChunkProcessOutput` (with `vector_f32`), passes `tuple[ChunkProcessOutput, ...]` directly into `publish_graph` -> `plan_graph_publication`, and deletes `load_active_chunk_inferences`, `compute_active_inference_fingerprint`, `ChunkStateTargetHandler`, and `GraphDeltaCoordinator` -> `critic (Model="flash")` audits against Shortcut 1 (relocated SQLite polling in `indexing.py` or `planner.py`).
  - *Artifact & exit gate*: `outputs/benchmarks/cg_index_2_direct_handoff.xml`; `0` `[target]` `embed_text` calls inside `planner.py`, `0` `[target]` `SELECT` queries on `source_chunks`/`chunk_inference` in `publish_graph`/`planner.py`, and `0` `[target]` matches for retired coordinator/polling symbols.
- **I3 (Gap 5 — Retire `candidate_stage_cache` SQLite delete in `runtime.py`)**:
  - *Orchestration*: Executed atomically by `simplifier` in lockstep with slice `C5` in [native-cocoindex-graph.md §8](native-cocoindex-graph.md#8-checkpoints), replacing `DELETE FROM candidate_stage_cache WHERE stage = 'simba_task_score'` in `yaw/daemon/runtime.py:1240–1241` (`50be17d`) with `meta` table invalidation -> `critic (Model="flash")` verifies `0` `[target]` `candidate_stage_cache` references across `yaw/`, `scripts/`, and `tests/`.
  - *Artifact & exit gate*: `outputs/benchmarks/c5_candidate_cache_retirement.xml`; `git grep -n -e "candidate_stage_cache" yaw/ scripts/` returns `0` `[target]` matches.
- **I4 (Windows live filesystem & full paired lifecycle acceptance)**:
  - *Orchestration*: `simplifier` executes Windows create/edit/rapid-edit/rename/delete, malformed file replacement, partial enumeration, crash/replay, and ordered drain/shutdown suites alongside `G5–G6` in [native-cocoindex-graph.md §8](native-cocoindex-graph.md#8-checkpoints) -> `critic (Model="flash")` audits resource cleanup and worker RSS.
  - *Artifact & exit gate*: `outputs/benchmarks/live_lifecycle_acceptance.xml`; all fixtures in §9 pass under `<= 1 GiB` `[target]` worker RSS.

### Ordered Slices (Anti-Patterns 1–6 Remediation from `28a1113`)

- **Checkpoint N1 (`ManagedTableTarget.delete_row` & AST Private-Attribute Gate — Anti-Pattern 6)**: Owned by [native-cocoindex-graph.md §8](native-cocoindex-graph.md#8-checkpoints) (`3` `[target]` files: `yaw/daemon/graph/storage/managed_sqlite.py`, `yaw/daemon/graph/storage/rows.py`, `tests/core/test_architecture.py`).
- **Checkpoint N2 (`_chunk_file_memo` Text Memoization & Throwaway DB Elimination in `snapshot_dataset_bundle` and `process_file` — Anti-Pattern 4)**:
  - *Orchestration*: `simplifier (Model="flash")` -> `critic (Model="flash")`.
  - *Bounded inputs* (`4` `[target]` files): `yaw/daemon/ingestion/source_targets.py`, `yaw/daemon/ingestion/bundles.py`, `yaw/daemon/indexing.py`, `tests/daemon/ingestion/test_datasets.py` (plus `declare_source_file` callers in `tests/daemon/ingestion/test_source_targets.py` and `tests/daemon/ingestion/test_corpus.py`).
  - *Exact implementation instructions*:
    1. In `yaw/daemon/ingestion/source_targets.py:78–125` (`28a1113`), bump `_chunk_file_memo` to `@coco.fn(memo=True, version=2)` returning `tuple[tuple[SourceMetadata, str], ...]` (`(SourceMetadata(...), text)` for `pointer, text in chunks`), and update `declare_source_file` to iterate `for meta, _ in chunks: table.declare_row(row=meta)` and return `chunks` (`tuple[tuple[SourceMetadata, str], ...]`).
    2. In `yaw/daemon/indexing.py:143, 184–208` (`28a1113`), update `process_chunk` line `143` from `if frozen_text is None and pointer.start_line > 1:` to `if settings is not None and pointer.start_line > 1:` (so AST parent context is still hydrated when `process_file` passes `frozen_text=text`), and update `process_file` to pass `frozen_text=text` for each `metadata, text in declared` returned by `declare_source_file` (`0` `[target]` `resolve_chunk_text` calls in `process_file`).
    3. In `yaw/daemon/ingestion/bundles.py:236–394` (`28a1113`), delete `_process_bundle_file`, `source_snapshot.db`, `open_managed_yaw_db`, and the post-hoc `read_text`/`chunk_source_text` loop (`L303–394` at `28a1113`); have `_bundle_snapshot_app_main(settings: SourceSettings) -> dict[str, tuple[tuple[SourceMetadata, str], ...]]` call `_chunk_file_memo(file, normalize_chunking_settings(settings))` per matched file and return `dict(zip(rels, per_file_chunks, strict=True))` directly from `await app.update().result()`. (Note: `bundles.py` does **not** call `acquire_coco_environment` in `N2`; shared environment wiring arrives in `N3`.)
  - *Artifact & exit gate*: `outputs/benchmarks/n2_bundle_memo_text.xml` `[target]`; `git grep -n -e "source_snapshot.db" -e "_process_bundle_file" yaw/ scripts/ tests/` returns `0` `[target]` matches, and all `5` `[target]` mandatory quality gates pass.
- **Checkpoint N3 (Shared `coco.Environment` & Named App Registry; Delete `_release_idle_coco_env` — Anti-Pattern 3)**:
  - *Orchestration*: `simplifier (Model="flash")` -> `critic (Model="flash")`.
  - *Bounded inputs* (`4` `[target]` files): `yaw/daemon/indexing.py`, `yaw/daemon/runtime.py`, `yaw/daemon/ingestion/bundles.py`, `tests/daemon/graph/test_indexing_overlap.py`.
  - *Exact implementation instructions*:
    1. In `yaw/daemon/indexing.py:55–71` (`28a1113`), delete `_ActiveCocoAppRegistry`, `register_active_coco_env_closer`, and `release_idle_coco_env`. Define `_SharedCocoRuntime(environment: coco.Environment, context: coco.ContextProvider, apps: dict[str, coco.App[Any, Any]])`, `acquire_coco_environment(state_root: Path | str) -> _SharedCocoRuntime`, and `close_coco_environment(state_root: Path | str | None = None) -> None` guarded by a module lock and keyed by `resolve_coco_state_root(state_root)`.
    2. Cache `"yaw_frozen_indexing"` and `"yaw_bundle_source"` inside `shared.apps` on first use because `EnvironmentInfo.register_app` in `cocoindex == 1.1.0` (`cocoindex/_internal/environment.py:180–183`) raises `ValueError` if two `coco.App` instances with the same `name` are registered on the same `coco.Environment`.
    3. Wire `YawRuntime._build_coco_app()` / `_close_coco_app()` (`yaw/daemon/runtime.py:285–346, 1130` at `28a1113`), `bind_native_indexing` (`yaw/daemon/indexing.py:597–628` at `28a1113`, plus `engine.close()`), and `snapshot_dataset_bundle` (`yaw/daemon/ingestion/bundles.py`) to `acquire_coco_environment` / `close_coco_environment`, deleting `YawRuntime._release_idle_coco_env`.
  - *Artifact & exit gate*: `outputs/benchmarks/n3_coco_memo_env_cutover.xml` `[target]`; `git grep -n -e "_release_idle_coco_env" -e "_ActiveCocoAppRegistry" -e "register_active_coco_env_closer" -e "release_idle_coco_env" yaw/ scripts/ tests/` returns `0` `[target]` matches, and all `5` `[target]` mandatory quality gates pass.
- **Checkpoint N4 (`@coco.fn(memo=True, version=1)` Mounted Components, `FROZEN_CHUNKS` Context Keys, Input-Only `graph_input_fingerprints`, & `partition_frozen_chunks` / `FrozenChunkStream` Removal — Anti-Patterns 1 & 2)**:
  - *Orchestration*: `simplifier (Model="flash")` -> `critic (Model="flash")`.
  - *Bounded inputs* (`5` `[target]` files): `yaw/daemon/indexing.py`, `yaw/daemon/graph/planner.py`, `tests/daemon/graph/test_indexing_overlap.py`, `tests/daemon/graph/test_evidence_revisions.py`, `tests/daemon/graph/test_batch_equivalence.py`.
  - *Exact implementation instructions*:
    1. In `yaw/daemon/indexing.py`, decorate `process_file` (`L170` at `28a1113`) and `process_frozen_chunk` (`L396` at `28a1113`) with `@coco.fn(memo=True, version=1)` while keeping plain-awaited `process_chunk` (`L85` at `28a1113`) as `@coco.fn` without `memo=True`.
    2. Define `FROZEN_CHUNKS = coco.ContextKey[tuple[tuple[ChunkPointer, str], ...]]("yaw/frozen_chunks")`, `FROZEN_GRAPH_SETTINGS = coco.ContextKey[GraphPlanSettings | None]("yaw/frozen_graph_settings")`, and `FROZEN_INFERENCE_IDENTITY = coco.ContextKey[InferenceIdentity | None]("yaw/frozen_inference_identity")`. Update `frozen_indexing_app_main(settings: GraphPlanSettings | None = None, identity: InferenceIdentity | None = None) -> GraphPlanSummary | None` to read `active_chunks = coco.use_context(FROZEN_CHUNKS)`, `effective_settings = settings if settings is not None else coco.use_context(FROZEN_GRAPH_SETTINGS)`, and `effective_identity = identity if identity is not None else coco.use_context(FROZEN_INFERENCE_IDENTITY) or (InferenceIdentity(effective_settings.program_identity, effective_settings.embedding_identity) if effective_settings is not None else InferenceIdentity("", ""))` so both cached `shared.apps["yaw_frozen_indexing"]` (where `program`/`settings` change per GEPA candidate in `execute(chunks, program)`) and direct `test_evidence_revisions.py` callers work seamlessly.
    3. In `yaw/daemon/graph/planner.py:854` (`28a1113`), change `input_fingerprints` to `{out.chunk_id: fingerprint_object((settings, out.pointer, out.text)).hex() for out in ordered_outputs}`.
    4. In `yaw/daemon/indexing.py:424–488` (`28a1113`), delete `partition_frozen_chunks` and `FrozenChunkStream`, and add `_partition_published_chunks(db, chunks, settings)` reading only `SELECT chunk_id FROM chunks` and `meta['graph_input_fingerprints']` (`0` `[target]` `source_chunks`/`chunk_inference` SQL queries).
    5. In `tests/daemon/graph/test_evidence_revisions.py:61–68, 115–117` (`28a1113`), update `native_corpus` to yield `(db, env, app)` (`app = coco.App(coco.AppConfig(name="evidence_revisions", environment=env), frozen_indexing_app_main, settings)`) and update `publish(env, app, chunks)` to call `env.context_provider.provide(FROZEN_CHUNKS, tuple(chunks)); app.update_blocking()`. Update `tests/daemon/graph/test_batch_equivalence.py:669–677` (`28a1113`) to provide `FROZEN_CHUNKS` via `ctx.provide(FROZEN_CHUNKS, tuple(chunks))` and update `tests/daemon/graph/test_indexing_overlap.py:790–914` (`28a1113`) to test `_partition_published_chunks` and verify `0` `[target]` Python `process_frozen_chunk` re-entries on Batch 2.
  - *Artifact & exit gate*: `outputs/benchmarks/n4_component_memo_context_cutover.xml` `[target]`; `git grep -n -e "partition_frozen_chunks" -e "FrozenChunkStream" yaw/ scripts/ tests/` returns `0` `[target]` matches, and all `5` `[target]` mandatory quality gates pass.
- **Checkpoint N5 (Incremental `CorpusIndexingLiveComponent` Event Map & Live Memo Short-Circuit — Anti-Pattern 5)**:
  - *Orchestration*: `simplifier (Model="flash")` -> `critic (Model="flash")`.
  - *Bounded inputs* (`2` `[target]` code/test files + docs): `yaw/daemon/indexing.py`, `tests/daemon/graph/test_indexing_overlap.py`, `docs/reviews/incremental-publication-work.md`.
  - *Exact implementation instructions*:
    1. In `CorpusIndexingLiveComponent.__init__` (`yaw/daemon/indexing.py:265–279` at `28a1113`), initialize `self._active_files: dict[str, FileLike] | None = None`.
    2. In `_CorpusFeedSubscriber` (`yaw/daemon/indexing.py:239–258` at `28a1113`), pass `component: CorpusIndexingLiveComponent` into `__init__`: `update_all()` sets `self._component._active_files = None` and awaits `self._operator.update_full()` under `self._lock`; `update(key, value)` sets `if self._component._active_files is not None: self._component._active_files[str(key)] = value` in $O(1)$ and awaits `self._operator.update_full()` under `self._lock`; `delete(key)` calls `if self._component._active_files is not None: self._component._active_files.pop(str(key), None)` in $O(1)$ and awaits `self._operator.update_full()` under `self._lock`.
    3. In `CorpusIndexingLiveComponent.process()` (`yaw/daemon/indexing.py:293–309` at `28a1113`), populate `if self._active_files is None: self._active_files = {str(rel): f async for rel, f in self._files.items()}` and iterate over `sorted(self._active_files.items())`.
  - *Artifact & exit gate*: `outputs/benchmarks/n5_incremental_live_component.xml` `[target]`; `_CorpusFeedSubscriber.update(key, value)` and `.delete(key)` mutate `CorpusIndexingLiveComponent._active_files` in place with `0` `[target]` `self._files.items()` full directory walks on single-file live events and `0` `[target]` Python `process_file` executions on untouched sibling files, and all `5` `[target]` mandatory quality gates pass.

## 9. Verification

| Fixture | Required result |
| --- | --- |
| `2` `[target]` fresh candidate SQLite DBs (`db_a`, `db_b`) sharing `<state_dir>/coco_lmdb` on overlapping `64`-chunk `[target]` inputs (`CG-INDEX-1`) | `0` `[target]` provider calls for overlapping chunks on `db_b`; identical `15`-table `[target]` graph state in both DBs; `db_a` target state untouched. |
| Direct `ChunkProcessOutput` handoff via `coco.use_mount` (`CG-INDEX-2`) | `plan_graph_publication` receives `vector_f32` and `ChunkInference` in memory; `0` `[target]` `embed_text` calls and `0` `[target]` `source_chunks`/`chunk_inference` SQLite queries in `publish_graph` and `planner.py`. |
| Component-level `@coco.fn(memo=True, version=1)` short-circuit & shared `acquire_coco_environment` (`CG-INDEX-3`, `N3`, `N4`) | On Batch 2 (`chunks[0:128]`), Batch 1 (`chunks[0:64]`) incurs `0` `[target]` Python `process_frozen_chunk`/`process_chunk` executions while retaining `source_chunks`/`chunk_inference` target rows; `YawRuntime` and `bind_native_indexing` share `1` `[target]` `coco.Environment` per `state_root`. |
| Single-read `_chunk_file_memo` (`version=2`) in `snapshot_dataset_bundle` and `process_file` (`CG-INDEX-3`, `N2`) | `0` `[target]` `source_snapshot.db` files created; `chunk_source_text` called `1` `[target]` time per cold file and `0` `[target]` times on warm memo; `process_file` makes `0` `[target]` `resolve_chunk_text` calls. |
| Interrupted indexing after `chunk_inference` write / before `publish_graph` commit (default-resume, `N4`) | Unpublished chunks absent from `chunks` or `meta['graph_input_fingerprints']` (`fingerprint_object((settings, pointer, text)).hex()`) are placed in `changed` with `0` `[target]` SQL queries on `source_chunks`/`chunk_inference`, published in strict `64`-chunk `[target]` batches, and reuse LMDB `extract_text`/`embed_text` memos (`0` `[target]` duplicate LLM/embedder calls). |
| Incremental live `_CorpusFeedSubscriber` update/delete (`CG-INDEX-3`, `N5`) | Single-file live create/modify/delete updates `CorpusIndexingLiveComponent._active_files` in place with `0` `[target]` `self._files.items()` full directory walks and `0` `[target]` Python `process_file` calls on untouched sibling files. |
| Same-ID source edit, file rename, and complete deletion (`chunks = []`) | Edited chunk re-extracts and retracts stale support; shared entities/facts survive; empty snapshot withdraws all rows cleanly. |
| Missing frozen text (`text = ""` or `None`) or partial parser failure | Raises immediately; never hydrates from mutable disk during frozen indexing or deletes unseen valid chunks. |
| Windows live `localfs.walk_dir(..., live=True)` create/edit/delete + ordered `YawRuntime.stop()` | Drains in-flight publication before closing `Environment` and SQLite connection; `0` `[target]` `sqlite3.ProgrammingError` or orphaned worker threads. |

Quality gates:
```powershell
uv sync --locked
uv run --locked ty check
uv run --locked ruff check --fix
uv run --locked ruff format
uv run --locked pytest
git diff --check
```

Measurements recorded in: [evaluation-ingestion.md](evaluation-ingestion.md) and [evaluation-graph-storage.md](evaluation-graph-storage.md). Work notes: [incremental-publication-work.md](reviews/incremental-publication-work.md).

## Decisions

- 2026-11-06: "Step 4 should be included in this unit, but obviously, you can implement one unit at a time according to whichever spec you write into @[docs/native-cocoindex-graph.md] or @[docs/native-cocoindex-indexing.md] , also sounds like you'll need to update our rules and tests to prevent these kinds of failures surfacing again." (User). Specs are promoted into `docs/native-cocoindex-graph.md` and `docs/native-cocoindex-indexing.md` and executed `1` `[target]` checkpoint unit at a time with regression tests.
- 2026-11-07: "Audit and review for root cause for batch indexing regressing back to exponential time behavior. Incremental indexing and graph publication was designed and tested to be near-constant time for 64 + 64 chunk batches vs 128 chunk batchless indexing." / "It’s a SHACL Violation CONSISTENCY_ERROR" / "you will need to /spec first." (User). Enforces delta-scoped `64`-chunk `[target]` batch publication (`publish_graph` `version=8` `[observed: commit c6fa080]`, focus-isolated `heal_shacl_violations` `version=4` `[observed: commit c6fa080]`, repair-preserving `canonicalize_fact_definitions`, `"__healed__"` label provenance, `touched_entity_ids`-scoped Active EDC, delta-only `canonicalize_fact_text`, and single-pass post-canonicalization `reembed_facts` embedding).
- 2026-11-07: "SHACL healing is taking extremely long - what is the performance bottleneck here?" / "/spec please spec the remedy for your debug audit above." (User). ~~Requires `source_chunks JOIN chunk_inference` polling via `compute_active_inference_fingerprint` and `load_active_chunk_inferences`~~ (superseded by 2026-11-07 Gaps 1–5 entry below for planner handoff); ~~retains published `chunks` and `meta['graph_input_fingerprints']` matching in `partition_frozen_chunks`~~ (`partition_frozen_chunks` superseded by 2026-11-09 Anti-Patterns 1–6 entry below); retains direct pre-hydrated `SHACLHealInput` dispatch to `YawProgram.shacl_repair` (`mode="direct"`, `heal_shacl_violations` `version=5` `[observed: yaw/daemon/graph/inference/components.py:309, 50be17d]`) and `OpenIEEndpoint.progress_callback` telemetry.
- 2026-11-07: "/spec remedy gaps 1-5 in @[docs/native-cocoindex-indexing.md] and @[docs/native-cocoindex-graph.md] - remember the separation of concerns, one is for indexing and the other is for graphing. Use subagents to protect your own context from bloat." (User). Remedies Gaps 1–5 across the indexing/graph boundary: (1) `process_chunk` returns `ChunkProcessOutput(pointer, text, inference, vector_f32)` via `coco.use_mount` to `publish_graph` while `publish_graph` remains the single atomic `DatabaseSink` producer for all `15` `[target]` graph tables (including `chunk_vec`, with `0` `[target]` duplicate vector columns on `chunk_inference`); (2) `coco.Settings(db_path=...)` points to a shared persistent `<state_dir>/coco_lmdb` with `db_identity`-scoped component mounts and `TableKey(db_identity, name)` target states; (3) retires `load_active_chunk_inferences`, `compute_active_inference_fingerprint`, `ChunkStateTargetHandler`, and `GraphDeltaCoordinator` in favor of direct `coco.use_mount` value flow; (4) delegates `chunk_vec`, `entity_vec`, and `fact_vec` `sqlite-vec` `vec0` virtual table DDL/DML to [native-cocoindex-graph.md](native-cocoindex-graph.md); and (5) retires `candidate_stage_cache` SQLite usage in `runtime.py:1240–1241` (`50be17d`) alongside `CandidateStageCache` deletion in [native-cocoindex-graph.md](native-cocoindex-graph.md).
- 2026-11-07: "Plan doesn’t take subagent orchestration into account, should be explicitly defined." (User). Codifies the mandatory subagent orchestration protocol (`research` -> `simplifier` -> `critic (Model="flash")`, with `ontologist` on W3C/SHACL/identity slices in [native-cocoindex-graph.md §8](native-cocoindex-graph.md#8-checkpoints)) across §8 Checkpoints so the parent agent acts strictly as a context-protected orchestrator.
- 2026-11-08: "I think everything should default to resume, so we do not wipe databases by accident, and `--fresh` flag wipes intentionally. Adjust all flags to behave like this across the codebase" / "Think if we should switch from postfix to prefix across the board so these related files get sorted together in file browser: `optimize_program_telemetry`, `optimize_program.json`, `optimize_program_eval.json`, `optimize_program_audit.json`, `optimize_program_pools.json`, `optimize_status.json`, `optimize_telemetry.json`, `optimize_yaw.db`, `optimize_yaw.db.lock` -> this looks much nicer and well organized when sorted by alphabet" / "Think if we can use cocoindex fingerprinting capabilities instead of handrolling our own." (User). Establishes the canonical `optimize_*` sandbox in `.yaw/` with default-resume on `yaw dspy optimize` and explicit `--fresh` to wipe `.yaw/optimize_*` intentionally (`--resume` retired); unifies both `YawRuntime._build_coco_app` and `bind_native_indexing` onto `<base_dir>/coco_lmdb` (`.yaw/coco_lmdb`) and `<base_dir>/optimize_coco/<dataset>`; replaces hand-rolled `json.dumps` + `hashlib.sha256` predictor/stage fingerprint helpers with CocoIndex's `fingerprint_object(payload).hex()` via `graph_plan_fingerprints`; and retires `_proposer_telemetry.json` while renaming `_graph_rlm_audit.json` to `_audit.json` (`.yaw/optimize_program_audit.json`).
- 2026-11-09: "Review our cocoindex implementation against cocoindex skill and docs, are we inventing our own orchestration, caching, fingerprinting etc when cocoindex already provides it? Make a comprehensive audit." / "Please /spec these changes." (User). Remediates CocoIndex Anti-Patterns 1–5 in this spec (and Anti-Pattern 6 in [native-cocoindex-graph.md](native-cocoindex-graph.md)): (1) deletes `partition_frozen_chunks` (`yaw/daemon/indexing.py:424–478` at `28a1113`) and keys `meta['graph_input_fingerprints']` on `fingerprint_object((settings, pointer, text)).hex()` so micro-batch admission checks `SELECT chunk_id FROM chunks` + `meta['graph_input_fingerprints']` with `0` `[target]` SQL joins on `source_chunks` or `chunk_inference`; (2) deletes mutable `FrozenChunkStream` (`indexing.py:481–488` at `28a1113`), introduces `FROZEN_CHUNKS` (`coco.ContextKey`), and decorates mounted `process_file` and `process_frozen_chunk` with `@coco.fn(memo=True, version=1)` (and `publish_graph` with `@coco.fn(memo=True, version=2)`, while keeping plain-awaited `process_chunk` as `@coco.fn` without `memo=True`) so unchanged mounted chunks short-circuit in Rust `ComponentProcessor` while preserving `TABLES.target_state` declarations; (3) deletes `YawRuntime._release_idle_coco_env` (`runtime.py:285–297` at `28a1113`) and maintains a shared process-level `coco.Environment` per `resolved_state_root` (`acquire_coco_environment` / `close_coco_environment`) with `coco.App("yaw_frozen_indexing", ...)` reused across `engine.index_chunks()` calls; (4) updates `_chunk_file_memo` (`@coco.fn(memo=True, version=2)`) and `declare_source_file` to return `tuple[tuple[SourceMetadata, str], ...]`, eliminating the second file read/re-chunking in `process_file` and eliminating the throwaway `source_snapshot.db` + post-hoc file re-chunking loop in `snapshot_dataset_bundle`; and (5) maintains an incremental `CorpusIndexingLiveComponent._active_files` dictionary updated per-key by `_CorpusFeedSubscriber.update(key, value)` and `.delete(key)` without full-corpus directory re-walks.
