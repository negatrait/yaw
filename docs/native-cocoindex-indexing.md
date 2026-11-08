# Native CocoIndex indexing and live ingestion

Status: draft, 2026-11-07. Pinned code: `50be17d`.

| Relation | Links |
| --- | --- |
| Governed by | [FOUNDATIONAL.md](../FOUNDATIONAL.md), [AGENTS.md](../AGENTS.md), [epic-1-hipporag-daemon.md](epic-1-hipporag-daemon.md) |
| Rules | [daemon.md](../.agents/rules/daemon.md): `yaw/daemon/*.py`, `yaw/daemon/ingestion/*.py`; [graph-storage.md](../.agents/rules/graph-storage.md): `yaw/daemon/graph/*.py`, `yaw/daemon/graph/storage/*.py`; [inference.md](../.agents/rules/inference.md): `yaw/daemon/graph/inference/*.py`; [tests.md](../.agents/rules/tests.md): `tests/**/*.py`; [docs.md](../.agents/rules/docs.md): `docs/**/*.md` |
| Owns code | `yaw/daemon/runtime.py`, `yaw/daemon/indexing.py`, `yaw/daemon/ingestion/corpus.py`, `yaw/daemon/ingestion/source_targets.py`, `yaw/daemon/ingestion/datasets.py`, `yaw/daemon/ingestion/bundles.py`, `yaw/daemon/graph/inference/components.py:extract_text`, `yaw/daemon/graph/inference/components.py:embed_text` |
| Peers | [native-cocoindex-graph.md](native-cocoindex-graph.md): owns `plan_graph_publication`, `define_missing_entities`, `judge_synonyms`, `heal_shacl_violations`, `hydration.py`, `chunks.py`, `DatabaseSink` / `vec0` target storage, `CandidateStageCache` retirement, and the paired `64 + 64` vs. `128` `[target]` acceptance gate; [w3c-improvements.md](w3c-improvements.md): owns `synonyms.py` and `canonicalize_entity_labels_edc` (`§3.2`) and constrains SKOS/OWL/SHACL closed-world shapes; [epistemic-confidence-spec.md](epistemic-confidence-spec.md): constrains Bayesian confidence pooling; [dspy-improvements.md](dspy-improvements.md): constrains GBNF adapter and RLM healer architecture; [enterprise-rag-bench-training-plan.md](enterprise-rag-bench-training-plan.md): constrains stratified `135/27` `[target]` training/validation split |
| References | [arxiv-2502.14802v1-hipporag2.md](arxiv-2502.14802v1-hipporag2.md): anchors HippoRAG 2 indexing and retrieval parity; [model-evaluation.md](model-evaluation.md): anchors local `llama.cpp` serving constraints (`parallel = 1` `[target]`) |
| Supersedes / superseded by | Supersedes earlier `skos:ConceptScheme` indexing hooks, per-database `candidate_coco/<db_name>` LMDB isolation (`50be17d`), discarded `EmbeddedText` in `process_chunk` (`50be17d`), and SQLite `source_chunks JOIN chunk_inference` planner polling (`50be17d`) |
| Records | [incremental-publication-work.md](reviews/incremental-publication-work.md), [shacl-healer-publication.md](reviews/shacl-healer-publication.md), [resource-vocabulary-boundary.md](reviews/resource-vocabulary-boundary.md), [evaluation-ingestion.md](evaluation-ingestion.md), [evaluation-graph-storage.md](evaluation-graph-storage.md) |

## 1. Scope

In scope: Documentation contract for native CocoIndex source ingestion, `64`-chunk `[target]` micro-batch admission, persistent shared LMDB memoization (`coco.Settings(db_path=...)`), per-database component/target-state isolation, direct component value handoff (`process_chunk` -> `publish_graph`), provider admission boundaries, and ordered runtime lifecycle across `yaw/daemon/runtime.py`, `yaw/daemon/indexing.py`, `yaw/daemon/ingestion/*.py`, and `yaw/daemon/graph/inference/components.py` (`extract_text`, `embed_text`).

Non-goals: Graph identity canonicalization, missing-entity definition (`define_missing_entities`), synonym candidate mining and adjudication (`judge_synonyms`), SHACL repair semantics (`heal_shacl_violations`), `vec0` virtual table DDL/DML, 1-indexed disk hydration (`hydration.py`), PPR retrieval, and the numeric `64 + 64` vs. `128` `[target]` graph parity definitions (all owned by [native-cocoindex-graph.md](native-cocoindex-graph.md)); `skos:ConceptScheme` clustering or tables (permanently removed); live process restarts during this specification turn.

Depends on decisions: [2026-11-06: Step 4 atomic units](#decisions), [2026-11-07: delta-scoped 64-chunk batch publication](#decisions), [2026-11-07: direct pre-hydrated SHACL healing & resume partitioning](#decisions), [2026-11-07: remedy gaps 1–5 across indexing and graph specs](#decisions), [native-cocoindex-graph.md decisions](native-cocoindex-graph.md#decisions).

## 2. Problem, with evidence

At pinned commit `50be17d`, the indexing pipeline exhibits five concrete defects across the indexing/graph boundary:

- **Gap 1 (Chunk-vector discard & ownership split)**: `yaw/daemon/indexing.py:process_chunk` (`L252–331` at `50be17d`) awaits `_, record = await asyncio.gather(embed_text(embed_req), ...)` (`L316–319` at `50be17d`), discarding the returned `EmbeddedText` (`vector_f32: tuple[float, ...]`) to `_`, while `yaw/daemon/graph/planner.py:plan_graph_publication` (`L1099–1127` at `50be17d`) rebuilds `EmbeddingRequest` via `_prepare_embed_req` and calls `embed_text` a second time for every active chunk to populate `chunk_vec` `[observed: yaw/daemon/indexing.py:316-319, yaw/daemon/graph/planner.py:1099-1127, 50be17d]`. Meanwhile, earlier draft sections proposed declaring `chunk_vec` directly inside `process_chunk`, which in `cocoindex == 1.1.0` commits via an independent component `TargetActionSink` outside `publish_graph`'s atomic `DatabaseSink` transaction (`yaw/daemon/graph/storage/managed_sqlite.py:75–112` at `50be17d`) `[observed: cocoindex/_internal/target_state.py:614-629, cocoindex 1.1.0]`.
- **Gap 2 (Per-DB LMDB `state_root` defeating cross-run memo reuse)**: `yaw/daemon/indexing.py:bind_native_indexing` (`L752` at `50be17d`) sets `state_root = Path(engine.db_path).resolve().parent / "candidate_coco" / Path(engine.db_path).name`, creating an isolated LMDB environment per transient candidate database (`.yaw/yaw_optimize_<ts>.db`) `[observed: yaw/daemon/indexing.py:752, 50be17d]`. This violates [FOUNDATIONAL.md §4.1](../FOUNDATIONAL.md) and forces cold re-execution of `@coco.fn(memo=True)` functions unless a secondary cache rescues them. However, naively pointing two databases at the same `state_root` with the current fixed component path `/publication` (`yaw/daemon/indexing.py:493, 724` at `50be17d`) and bare table key `TABLES.target_state(name, spec)` (`yaw/daemon/graph/storage/managed_sqlite.py:195` at `50be17d`) causes `cocoindex == 1.1.0` to hit the first database's `/publication` component memo (`TABLES.components`) or collide on `TABLES.target_state`, leaving the second database empty or missing unchanged rows `[observed: cocoindex/_internal/core.pyi:237-286, function.py:1838-1844, cocoindex 1.1.0]`.
- **Gap 3 (`source_chunks` + `chunk_inference` used as an intermediate SQLite polling bus)**: Despite claiming that `source_chunks` and `chunk_inference` are not scanned by a second planner, `yaw/daemon/indexing.py` (`50be17d`) mounts chunk components with `coco.mount` (which returns `None` rather than the child's return value `[observed: cocoindex/_internal/api.py:359-422, cocoindex 1.1.0]`) and bridges chunk inference to `publish_graph` via `148` lines `[observed: yaw/daemon/indexing.py:101-249, 50be17d]` of custom coordination (`GraphDeltaCoordinator`, `ChunkStateTargetHandler`, `_SubscriberProxy`, `_DeltaNotifyingLiveItems`, `GraphPublicationLiveComponent` at `L101–249, 425–524` at `50be17d`), `compute_active_inference_fingerprint` (`L368–400` at `50be17d`), `partition_frozen_chunks` (`L618–673` at `50be17d`), and `yaw/daemon/graph/planner.py:load_active_chunk_inferences` (`L203–263` at `50be17d`) scanning `source_chunks JOIN chunk_inference` from SQLite.
- **Gap 4 (Ordinary `BLOB` vector tables vs. `sqlite-vec` `vec0` virtual tables)**: `chunk_vec` is currently created as an ordinary `BLOB` table (`yaw/daemon/graph/storage/db.py:147–151` at `50be17d`) rather than a `sqlite-vec` `vec0` virtual table `[observed: yaw/daemon/graph/storage/db.py:147-163, 50be17d]`; see [native-cocoindex-graph.md §2 & §5](native-cocoindex-graph.md#2-problem-with-evidence) for the `vec0` storage contract.
- **Gap 5 (Redundant `candidate_stage_cache` SQLite table coupling in `runtime.py`)**: `yaw/daemon/runtime.py` (`L1240–1241` at `50be17d`) executes `DELETE FROM candidate_stage_cache WHERE stage = 'simba_task_score'` on metric version upgrade `[observed: yaw/daemon/runtime.py:1240-1241, 50be17d]`, coupling runtime lifecycle to the redundant `CandidateStageCache` SQLite table retired in [native-cocoindex-graph.md §7](native-cocoindex-graph.md#7-removal-inventory).
- **Unknown**: Whether `cocoindex == 1.1.0` `localfs.walk_dir(..., live=True)` (`LiveItems`) can pass per-file/per-chunk `ChunkProcessOutput` return values directly into a downstream `publish_graph` mount without an event-driven trigger handle (probed by Capability Gate `CG-INDEX-2` in §6).

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
- **Measure**: Number of `embed_text` calls inside `plan_graph_publication`, number of `SELECT` queries against `source_chunks` or `chunk_inference` executed inside `publish_graph` and `plan_graph_publication`, and chunk batch admission size per `app.update()` cycle.
- **Reference**: Paired `64 + 64` vs. `128` `[target]` harness defined in [native-cocoindex-graph.md §3](native-cocoindex-graph.md#3-goals).
- **Passes when**: Each `app.update()` cycle admits at most `64` `[target]` changed chunks; `process_chunk` returns typed `ChunkProcessOutput(inference, vector_f32)` via `coco.use_mount`; neither `publish_graph` (`indexing.py`) nor `plan_graph_publication` (`planner.py`) executes any (`0` `[target]`) `SELECT` query against `source_chunks` or `chunk_inference` or any (`0` `[target]`) `embed_text` call for chunk vectors; and cumulative timing/parity satisfies [native-cocoindex-graph.md §3 Goal A & Goal B](native-cocoindex-graph.md#3-goals) (`0.80 <= R_time <= 1.25` `[target]`, `0.80 <= R_graph <= 1.25` `[target]`, worker RSS `<= 1 GiB` `[target]`).
- **Fails if / does not count**:
  - `process_chunk` declares `chunk_vec` in a separate component `TargetActionSink` outside `publish_graph`'s atomic transaction, or stores duplicate `embedding` bytes in `chunk_inference`.
  - Moving `SELECT ... FROM source_chunks JOIN chunk_inference` out of `planner.py:load_active_chunk_inferences` into `indexing.py:publish_graph` instead of passing `ChunkProcessOutput` in memory from `process_chunk`.
  - A smaller prefix of an already-indexed snapshot withdraws previously published chunks.

Thresholds are not loosened after seeing a run.

## 4. Semantics and invariants

<a id="3-revision-identity-memo-and-target-isolation"></a>

- **Equality and revision identity**:
  - A source occurrence revision is uniquely keyed by `(chunk_id, source_pointer, sha256_hash, program_identity, embedding_identity)`:
    - `source_pointer` uses `1`-indexed `[target]` inclusive `(start_line, end_line)` bounds and `0` `[target]` raw text columns in `source_chunks`.
    - Identical text at `2` `[target]` distinct file coordinates retains distinct `chunk_id` occurrence provenance.
    - Same-ID edits change `sha256_hash` or line bounds; ID set difference (`set(new_ids) - set(old_ids)`) is never used for change detection.
    - `program_identity` (extraction) and `embedding_identity` (embedding) remain separate columns on `ChunkInference` so changing an embedder does not invalidate `extract_text` memos, and changing an extraction prompt does not invalidate `embed_text` memos. Predictor and stage fingerprints (`judge_stage_fingerprint`, `embed_fingerprint`, `shacl_fingerprint`, `openie_fingerprint`, `extraction_fingerprint`, `candidate_fingerprint`, `query_retrieval_fingerprint`) are computed with CocoIndex's Rust-backed `fingerprint_object(payload).hex()` (`cocoindex.connectorkits.fingerprint`), and the 4-tuple `(program_identity, embedding_identity, synonym_identity, shacl_identity)` is assembled by `graph_plan_fingerprints(program, embedder)` in `yaw/daemon/indexing.py`.
    - The chunk embedding has exactly `1` `[target]` persisted `float32` SQLite copy (in `chunk_vec`), never a second `embedding` column on `chunk_inference` or pickled bytes in a component memo key.
- **Ownership**:

| State | Single owner |
| --- | --- |
| Persistent `@coco.fn(memo=True)` cache (`TABLES.memoization`: `extract_text`, `embed_text`, `define_missing_entities`, `judge_synonyms`, `heal_shacl_violations`) | Shared persistent CocoIndex LMDB environment (`<base_dir>/coco_lmdb`, default `.yaw/coco_lmdb`), resolved identically by both `YawRuntime._build_coco_app` (`yaw/daemon/runtime.py`) and `bind_native_indexing` (`yaw/daemon/indexing.py`) (`0` `[target]` `.yaw/coco_state*` directories) and keyed exclusively by `(fn_version, model_identity, prompt_fingerprint, input_payload)`. Dataset bundle snapshots live at `<base_dir>/optimize_coco/<dataset>`. |
| Per-database component mount state (`TABLES.components`) and target row tracking (`TABLES.target_state`) | CocoIndex LMDB environment scoped by `db_identity` (`<db_canonical_path>:<db_header_nonce>`), isolating each `yaw.db` file's applied target state. |
| `source_chunks` and `chunk_inference` tables in `yaw.db` | Declarative per-chunk targets in `yaw/daemon/indexing.py` (`process_chunk` / `process_frozen_chunk`) mounted via `managed_sqlite.py:mount_table(db_identity, ...)`, used strictly for provenance, default-resume watermarking, GEPA trace hydration (`engine.py`), and `scripts/watchdog.py` inspection—never read by `publish_graph` or `planner.py`. |
| `publish_graph` component mount (`/publication`) | `yaw/daemon/indexing.py:publish_graph`, which delegates planning to `plan_graph_publication` and target row declarations for all `15` `[target]` graph tables to `declare_tables` (both owned by [native-cocoindex-graph.md](native-cocoindex-graph.md)). |

- **Boundaries**:
  - **Empty input (`chunks = []`)**: Treated as complete corpus withdrawal when explicitly passed as the full desired snapshot; retires all published chunks atomically.
  - **Partial batch admission (`64`-chunk `[target]` micro-batches)**: Admitting `changed[:64]` (`<= 64` `[target]` chunks) retains all previously declared unchanged chunks in the snapshot envelope; a partial batch never withdraws unseen valid chunks.
  - **Missing frozen text (`text is None` or empty)**: Raises `ValueError` immediately; never falls back to `supplied_text or hydrate_from_disk(...)`.
  - **Provider Admission & Retries**: Default LLM concurrency is `Semaphore(1)` (`parallel = 1` `[target]`) for local `llama.cpp`; embedding batches contain at most `64` `[target]` inputs (`max_batch_size = 64` `[target]`) within configured context limits; sampling temperature follows the configured model profile (`1.0` `[target]` in development per [AGENTS.md](../AGENTS.md) to prevent small-model repetition loops; `>= 0.3` `[target]` minimum, reflection `0.7–0.8` `[target]` per [FOUNDATIONAL.md §4.4](../FOUNDATIONAL.md)). Transient model errors retry only via `dspy.LM(num_retries=...)`; exhausted retries raise the original exception (`FatalGraphPublicationError` subclasses `BaseException`).
  - **Crash & Replay**: CocoIndex LMDB and `yaw.db` commit independently. If a process crashes after `extract_text`/`embed_text` memos commit to LMDB but before `publish_graph` commits to `yaw.db`, `--resume` detects that `chunks` / `meta['graph_input_fingerprints']` lacks those chunks, places them in `changed`, and replays them in `64`-chunk `[target]` slices while hitting `TABLES.memoization` (`0` `[target]` duplicate provider calls).
  - **Shutdown**: Ordered lifecycle stops source admission, drains in-flight provider/publication tasks, closes live handles, and only then closes the CocoIndex `Environment` and SQLite connections.

## 5. Design (current state)

Decided:

```python
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


def graph_plan_fingerprints(program: YawProgram, embedder: DSPyEmbeddingAdapter) -> tuple[str, str, str, str]:
    """Return (program_identity, embedding_identity, synonym_identity, shacl_identity)
    computed via CocoIndex's fingerprint_object(payload).hex().
    """


def resolve_db_identity(conn: sqlite3.Connection, db_path: Path) -> str:
    """Return stable '<canonical_db_path>:<db_nonce>' stored in meta['db_identity']
    so recreated SQLite files at the same path never collide with stale LMDB target state.
    """


@coco.fn
async def process_chunk(
    pointer: ChunkPointer,
    text: str,
    identity: InferenceIdentity,
    source_lock: Path | None = None,
    *,
    artifacts: TableTarget[ChunkInference] | None = None,
    db_identity: str = "",
) -> ChunkProcessOutput:
    """Run memoized extract_text and embed_text in parallel; declare ChunkInference
    to chunk_inference via mount_table('chunk_inference', ..., db_identity=db_identity);
    and return ChunkProcessOutput (with vector_f32) to the caller via coco.use_mount.
    """


@coco.fn
async def process_frozen_chunk(
    chunk: tuple[ChunkPointer, str],
    identity: InferenceIdentity,
    sources: TableTarget[SourceMetadata],
    artifacts: TableTarget[ChunkInference],
    db_identity: str = "",
) -> ChunkProcessOutput:
    """Declare SourceMetadata to source_chunks via mount_table('source_chunks', ..., db_identity=db_identity)
    and return await process_chunk(pointer, text, identity, artifacts=artifacts, db_identity=db_identity).
    """


@coco.fn
async def publish_graph(
    settings: GraphPlanSettings,
    active_chunks: tuple[ChunkProcessOutput, ...],
    db_identity: str = "",
) -> GraphPlanSummary:
    """Single atomic graph publication component mounted via coco.use_mount at DB-invariant
    coco.component_subpath('publication') (so @coco.fn(memo=True) calls inside plan_graph_publication
    retain an identical parent StablePath across databases in TABLES.memoization while passing
    db_identity as a component argument forces component body execution per database in TABLES.components).
    Delegates to plan_graph_publication(settings, active_chunks) and
    declare_tables(plan.tables, db_identity=db_identity) over shared_sink, returning GraphPlanSummary.
    """
```

- **Frozen micro-batch scheduling (`bind_native_indexing`)**:
  - `partition_frozen_chunks(db, chunks, identity, settings=settings)` partitions the caller's complete snapshot into `(retained, changed)` by comparing `(ChunkPointer, InferenceIdentity, settings)` against committed `chunks` and `meta['graph_input_fingerprints']` in `yaw.db`.
  - Each `app.update()` cycle mounts `retained + changed[:64]` (`<= 64` `[target]` changed chunks) at DB-invariant `coco.component_subpath("chunk", m.chunk_id)` via `await asyncio.gather(*(coco.use_mount(coco.component_subpath("chunk", m.chunk_id), process_frozen_chunk, (m, text), identity, sources, artifacts, db_identity) for m, text in active_slice))` and passes the resulting `tuple[ChunkProcessOutput, ...]` directly to `summary = await coco.use_mount(coco.component_subpath("publication"), publish_graph, settings, active_chunks, db_identity)`, then reports `engine.report_indexing_batch(summary, len(active_slice))`.
  - Because `cocoindex == 1.1.0` (`function.py:1463-1465`) keys `TABLES.memoization` by `(app_name, parent_ctx.stable_path, memo_fp)` while `coco.mount` keys `TABLES.components` by `fingerprint((StablePath, args))`, keeping `coco.component_subpath("chunk", m.chunk_id)` and `coco.component_subpath("publication")` DB-invariant while passing `db_identity` as a component argument ensures `process_frozen_chunk` and `publish_graph` execute their component bodies for each target DB (declaring rows for `TableKey(db_identity, name)`) while hitting `100%` `[target]` of `TABLES.memoization` across databases with `0` `[target]` duplicate provider calls. Unchanged `source_chunks`/`chunk_inference` rows use `mount_table(..., db_identity=db_identity)` (`RowHandler.tracks_value_fingerprint = True`) so unchanged rows incur `0` `[target]` SQLite writes.

- **Live corpus handoff (`CorpusIndexingLiveComponent` & `_CorpusFeedSubscriber`)**:
  - `indexing_app_main` mounts `CorpusIndexingLiveComponent` at `coco.component_subpath("corpus")`, which iterates `self._files.items()`, mounts `process_file` per relative path via `coco.use_mount(coco.component_subpath("process_file", rel_path), process_file, ...)`, flattens the returned `tuple[ChunkProcessOutput, ...]` batches in memory, and mounts `publish_graph` at `coco.component_subpath("publication")` via `coco.use_mount`. In live mode, `items.watch(_CorpusFeedSubscriber(operator))` serializes `await operator.update_full()` across filesystem create/modify/delete notifications under an `asyncio.Lock()`.

Open: None.

## 6. Capability gates

| Behavior relied on | Locked version | Probe | Pass criterion | Stop rule |
| --- | --- | --- | --- | --- |
| `CG-INDEX-1`: Shared `coco.Settings(db_path=...)` LMDB reuses `@coco.fn(memo=True)` memos across `2` `[target]` distinct SQLite DBs when component mounts and target states are keyed by `db_identity` | `cocoindex == 1.1.0` `[measured: uv.lock:330-331, 50be17d]` | `tests/daemon/graph/test_indexing_overlap.py::test_shared_lmdb_cross_db_memo_reuse_and_target_isolation` | Indexing `64` `[target]` chunks into `db_a.sqlite` then indexing an overlapping `64`-chunk `[target]` set into fresh `db_b.sqlite` sharing `1` `[target]` `state_root` makes `0` `[target]` duplicate `extract_text`/`embed_text` calls for overlapping chunks and populates all `15` `[target]` graph tables in both DBs without withdrawing `db_a.sqlite` rows | Stop and amend this spec with the evidence |
| `CG-INDEX-2`: `coco.use_mount` returns typed `ChunkProcessOutput` payloads directly to `publish_graph` in both frozen (`frozen_indexing_app_main`) and live (`indexing_app_main`) flows without SQLite table polling | `cocoindex == 1.1.0` `[measured: uv.lock:330-331, 50be17d]` | `tests/daemon/graph/test_indexing_overlap.py::test_direct_use_mount_chunk_output_handoff_without_chunk_state_target` | `publish_graph` receives `tuple[ChunkProcessOutput, ...]` (including `vector_f32`) directly in memory; `0` `[target]` `SELECT` queries run against `source_chunks` or `chunk_inference` inside `publish_graph` or `planner.py` | Stop and amend this spec with the evidence |

## 7. Removal inventory

| Retired item | Location | Replacement or "none" | Exit check |
| --- | --- | --- | --- |
| Per-DB `candidate_coco/<db_name>` LMDB path suffix | `yaw/daemon/indexing.py:bind_native_indexing` (`L752` at `50be17d`) | `resolve_coco_state_root` (`<state_dir>/coco_lmdb`) + `db_identity`-scoped mounts and `TableKey` | `git grep -n -e "candidate_coco" yaw/ scripts/ tests/` has `0` `[target]` active production matches |
| `compute_active_inference_fingerprint` SQLite join | `yaw/daemon/indexing.py:compute_active_inference_fingerprint` (`L368–400` at `50be17d`) | Direct `tuple[ChunkProcessOutput, ...]` argument to `publish_graph` via `coco.use_mount` | `git grep -n -e "compute_active_inference_fingerprint" yaw/ scripts/ tests/` has `0` `[target]` active production matches |
| `ChunkStateTargetHandler`, `CHUNK_STATE_TARGET`, `declare_chunk_state` | `yaw/daemon/indexing.py:205–249, 330–337` (`50be17d`) | Direct `ChunkProcessOutput` return value from `process_chunk` via `coco.use_mount` | `git grep -n -e "ChunkStateTargetHandler" -e "CHUNK_STATE_TARGET" -e "declare_chunk_state" yaw/ scripts/ tests/` has `0` `[target]` active production matches |
| `GraphDeltaCoordinator`, `_SubscriberProxy`, `_DeltaNotifyingLiveItems`, `recovery_nonce` | `yaw/daemon/indexing.py:101–202, 425–524` (`50be17d`) | Native `coco.use_mount` / `LiveComponent` reactive handoff validated by `CG-INDEX-2` | `git grep -n -e "GraphDeltaCoordinator" -e "_SubscriberProxy" -e "_DeltaNotifyingLiveItems" -e "recovery_nonce" yaw/ scripts/ tests/` has `0` `[target]` active production matches |
| Metric-upgrade `candidate_stage_cache` SQLite delete in `runtime.py` | `yaw/daemon/runtime.py:1240–1241` (`50be17d`) | `yaw.db` `meta` table score invalidation alongside `CandidateStageCache` retirement ([native-cocoindex-graph.md §7](native-cocoindex-graph.md#7-removal-inventory)) | `git grep -n -e "candidate_stage_cache" yaw/ scripts/ tests/` has `0` `[target]` active production matches |

Consumers updated in the same change: `yaw/daemon/runtime.py`, `yaw/daemon/indexing.py`, `yaw/daemon/graph/planner.py`, `yaw/daemon/graph/storage/managed_sqlite.py`, `yaw/daemon/graph/storage/rows.py`, `tests/daemon/graph/test_indexing_overlap.py`, `tests/daemon/ingestion/test_source_targets.py`, `tests/daemon/test_runtime.py`.

## 8. Checkpoints

Status is tracked in [incremental-publication-work.md](reviews/incremental-publication-work.md).

### Subagent Orchestration Protocol (Context Protection)

To prevent parent-agent context bloat during implementation, the lead agent acts strictly as a lightweight orchestrator and delegates all code reading, TDD implementation, ontology/SHACL review, and post-slice verification to specialized subagents:

| Role | Subagent `TypeName` | `Model` | `Workspace` | Responsibility per Checkpoint |
| --- | --- | --- | --- | --- |
| Read-only discovery & SDK inspection | `research` | `inherit` | `inherit` | Inspects CocoIndex `1.1.0` `[measured: uv.lock:330-331, 50be17d]` internals, locates callers/tests, and returns compact `path:symbol` summaries before code edits. |
| Red-Green-Refactor TDD & deletion | `simplifier` | `inherit` | `inherit` | Authors failing real-substrate tests (`CG-INDEX-1`, `CG-INDEX-2`), implements the minimal code change (`1` `[target]` slice at a time), runs all `6` `[target]` mandatory quality gates plus §7 `git grep` exit checks, and appends evidence to [incremental-publication-work.md](reviews/incremental-publication-work.md). |
| Pre-commit architectural & shortcut audit | `critic` | `flash` | `inherit` | Audits the working tree diff after `simplifier` completes to verify `0` `[target]` masked failures, `0` `[target]` surviving retired symbols, `0` `[target]` relocated SQLite polling shortcuts, and `100%` `[target]` spec compliance before commit. |

### Ordered Slices

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

## 9. Verification

| Fixture | Required result |
| --- | --- |
| `2` `[target]` fresh candidate SQLite DBs (`db_a`, `db_b`) sharing `<state_dir>/coco_lmdb` on overlapping `64`-chunk `[target]` inputs (`CG-INDEX-1`) | `0` `[target]` provider calls for overlapping chunks on `db_b`; identical `15`-table `[target]` graph state in both DBs; `db_a` target state untouched. |
| Direct `ChunkProcessOutput` handoff via `coco.use_mount` (`CG-INDEX-2`) | `plan_graph_publication` receives `vector_f32` and `ChunkInference` in memory; `0` `[target]` `embed_text` calls and `0` `[target]` `source_chunks`/`chunk_inference` SQLite queries in `publish_graph` and `planner.py`. |
| Interrupted indexing after `chunk_inference` write / before `publish_graph` commit (`--resume`) | Unpublished chunks are placed in `changed` by `partition_frozen_chunks`, published in strict `64`-chunk `[target]` batches, and reuse LMDB `extract_text`/`embed_text` memos (`0` `[target]` duplicate LLM/embedder calls). |
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
- 2026-11-07: "SHACL healing is taking extremely long - what is the performance bottleneck here?" / "/spec please spec the remedy for your debug audit above." (User). ~~Requires `source_chunks JOIN chunk_inference` polling via `compute_active_inference_fingerprint` and `load_active_chunk_inferences`~~ (superseded by 2026-11-07 Gaps 1–5 entry below for planner handoff); retains published `chunks` and `meta['graph_input_fingerprints']` matching in `partition_frozen_chunks` (`publish_graph` `version=9` `[observed: yaw/daemon/indexing.py:403, 50be17d]`, `heal_shacl_violations` `version=5` `[observed: yaw/daemon/graph/inference/components.py:309, 50be17d]`), direct pre-hydrated `SHACLHealInput` dispatch to `YawProgram.shacl_repair` (`mode="direct"`), and `OpenIEEndpoint.progress_callback` telemetry.
- 2026-11-07: "/spec remedy gaps 1-5 in @[docs/native-cocoindex-indexing.md] and @[docs/native-cocoindex-graph.md] - remember the separation of concerns, one is for indexing and the other is for graphing. Use subagents to protect your own context from bloat." (User). Remedies Gaps 1–5 across the indexing/graph boundary: (1) `process_chunk` returns `ChunkProcessOutput(pointer, text, inference, vector_f32)` via `coco.use_mount` to `publish_graph` while `publish_graph` remains the single atomic `DatabaseSink` producer for all `15` `[target]` graph tables (including `chunk_vec`, with `0` `[target]` duplicate vector columns on `chunk_inference`); (2) `coco.Settings(db_path=...)` points to a shared persistent `<state_dir>/coco_lmdb` with `db_identity`-scoped component mounts and `TableKey(db_identity, name)` target states; (3) retires `load_active_chunk_inferences`, `compute_active_inference_fingerprint`, `ChunkStateTargetHandler`, and `GraphDeltaCoordinator` in favor of direct `coco.use_mount` value flow; (4) delegates `chunk_vec`, `entity_vec`, and `fact_vec` `sqlite-vec` `vec0` virtual table DDL/DML to [native-cocoindex-graph.md](native-cocoindex-graph.md); and (5) retires `candidate_stage_cache` SQLite usage in `runtime.py:1240–1241` (`50be17d`) alongside `CandidateStageCache` deletion in [native-cocoindex-graph.md](native-cocoindex-graph.md).
- 2026-11-07: "Plan doesn’t take subagent orchestration into account, should be explicitly defined." (User). Codifies the mandatory subagent orchestration protocol (`research` -> `simplifier` -> `critic (Model="flash")`, with `ontologist` on W3C/SHACL/identity slices in [native-cocoindex-graph.md §8](native-cocoindex-graph.md#8-checkpoints)) across §8 Checkpoints so the parent agent acts strictly as a context-protected orchestrator.
- 2026-11-08: "I think everything should default to resume, so we do not wipe databases by accident, and `--fresh` flag wipes intentionally. Adjust all flags to behave like this across the codebase" / "Think if we should switch from postfix to prefix across the board so these related files get sorted together in file browser: `optimize_program_telemetry`, `optimize_program.json`, `optimize_program_eval.json`, `optimize_program_audit.json`, `optimize_program_pools.json`, `optimize_status.json`, `optimize_telemetry.json`, `optimize_yaw.db`, `optimize_yaw.db.lock` -> this looks much nicer and well organized when sorted by alphabet" / "Think if we can use cocoindex fingerprinting capabilities instead of handrolling our own." (User). Establishes the canonical `optimize_*` sandbox in `.yaw/` with default-resume on `yaw dspy optimize` and explicit `--fresh` to wipe `.yaw/optimize_*` intentionally (`--resume` retired); unifies both `YawRuntime._build_coco_app` and `bind_native_indexing` onto `<base_dir>/coco_lmdb` (`.yaw/coco_lmdb`) and `<base_dir>/optimize_coco/<dataset>`; replaces hand-rolled `json.dumps` + `hashlib.sha256` predictor/stage fingerprint helpers with CocoIndex's `fingerprint_object(payload).hex()` via `graph_plan_fingerprints`; and retires `_proposer_telemetry.json` while renaming `_graph_rlm_audit.json` to `_audit.json` (`.yaw/optimize_program_audit.json`).
