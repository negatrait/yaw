# Native CocoIndex indexing and live ingestion

Accepted 2026-10-28; lifecycle acceptance reopened 2026-11-04 on `dev`.

`YawRuntime` catch-up, `sync_now`, and `snapshot_dataset_bundle` execute through
`indexing_app_main`. Optimization and benchmark snapshot acquisition use this
path, but candidate indexing still calls the engine's sequential `index_chunks`.
The native cutover is therefore incomplete. Runtime live
activation was added on 2026-11-04: `auto_poll=True` hosts a persistent native
`app.update(live=True)` and waits for readiness. CocoIndex owns file watching and
inference scheduling. The existing command worker drains live work before each
queued command and resumes watching after success, preserving ordered access.
This currently adds catch-up overhead to commands; concurrent read admission is
not established by the current lifecycle tests.

The 2026-11-04 audit also found that shutdown abandoned an active worker after
five seconds and then closed its database. Shutdown now joins that worker and
drains live work before closing resources. Tests cover actual runtime insertion,
deletion, cached stats, reload signals, continued ingestion after reload, parser
failure and resource release. Timeout/rapid-edit/stale-publication races still
need end-to-end coverage. Earlier completion claims and historical sections below
must be read with these remaining gates.

## Restart checklist

- [x] Remove the unconnected source_work journal draft; no callers or database migration existed.
- [x] Built-in SQLite targets for source contributions and inference artifacts.
  `source_chunks` and `chunk_inference` targets pass real SQLite replace, delete,
  restart and failed-replacement tests.
  `declare_source_file` returns pointer-only `SourceMetadata` tuples and `process_chunk`
  hydrates bounded text from `SOURCE_ROOT` via `resolve_chunk_text`, while `chunk_inference`
  retains only extraction JSON and traces (`chunk_vec` owns the single SQLite dense vector copy).
- [x] Native inference components, memoization and embedding batching.
  Embedding, OpenIE, missing-entity definition, synonym judgment, and read-only SHACL RLM
  components pass loopback-HTTP batching, memo and trace tests with float32 vectors.
- [x] Ordered graph planning outside network-spanning SQLite transactions.
  `yaw/daemon/graph/planner.py` (`plan_graph_publication`) prepares entity/fact embeddings,
  missing-entity definitions, homograph sense resolution (including same-document merged-definition
  prebatching), synonym adjudication, and read-only SHACL RLM proposals against an isolated
  in-memory SQLite staging snapshot before emitting `PreparedGraphRows`.
- [x] Idempotent graph target publishing validated graph changes atomically.
  `GraphTargetHandler` applies `PreparedGraphRows` and validates `lint_graph_shacl` inside a short,
  inference-free SQLite transaction, rolling back on any SHACL violation while preserving prior
  published graph rows and `source_chunks` / `chunk_inference`.
- [ ] Native filesystem live source and declarative delta/reload components integrated into daemon lifecycle.
  `localfs.walk_dir(..., live=True, rescan_interval=None)` drives both corpus files and
  `.yaw/reload.signal`, and `GraphPublicationLiveComponent` + `GraphDeltaCoordinator` publish
  committed state-diff transitions without `auto_refresh` or timer polling.
- [ ] Complete persistent snapshots and all consumer updates.
  `snapshot_dataset_bundle` runs on `indexing_app_main`, and `yaw/daemon/ingestion/datasets.py`
  is dataset-agnostic, reading all dataset metadata, source-field mappings, slice traits, and
  baselines from `data/benchmarks/baselines.json`.
  Remaining: replace `YawGraphEngine.index_chunks`, query-triggered candidate
  indexing in `_canonical_retrieval_hook`, and the manual eight-chunk loops in
  `_ensure_candidate_chunks_indexed` and `_prepare_optimization_pools`. Candidate
  preparation must use the native pipeline with a frozen snapshot and isolated
  candidate identity, preserving DSPy traces and task scores. Do not remove the
  loops alone: their callee still owns inference and direct graph writes.
- [x] Remove custom executor, polling and overlap switch.
  Removed `EmbeddingLane`, `indexing_overlap`, `CorpusIngestor`, `sync_corpus_once`,
  `YawRuntime._poll_loop`, `openie_cache`, `ReActV2` QA fallback, `"degraded"` lifecycle mode,
  and all silent parser/proposer/RLM fallback swallowers.
- [x] Real SQLite/HTTP recovery, equivalence and bounded-memory tests.
  Fresh-process interruption, SHACL rollback preservation, and unchanged replay with zero
  additional provider calls pass in `tests/daemon/graph/test_indexing_overlap.py`.
- [ ] Windows daemon live-ingestion lifecycle and failure tests.
  Live filesystem edit/rename/delete, `.yaw/reload.signal`, and live-to-catch-up shutdown pass
  on Windows.
- [ ] Ruff, ty, affected tests, architecture and diff checks.
  Revalidation at `06b35d2`: static checks pass; the engine/synonym regression
  selection has eight failures (see the November 4 evaluation entry).
- [x] Bounded live comparison and actual evidence in evaluation.md.

## Contract

CocoIndex schedules work through mounted components and native batching.
Four processing components, one in-flight request per endpoint,
64 embedding inputs and 3,500 formatted
characters per provider request are the initial bounds. Source files remain
untouched. DSPy remains the inference interface. Native memoization must run
inside processing components and must include source and model/program identity.

File components declare source contributions through native SQLite targets.
CocoIndex owns desired-state tracking, cleanup and recovery; do not introduce
source-generation, pending-work or acknowledgement tables. A single graph
materialization component transforms committed contributions into validated graph
declarations. Only the graph target sink applies graph/FTS changes, in a short
transaction with no inference. CocoIndex's tracking database and yaw.db are
separate durability domains, so target actions are idempotent.
Incomplete scans and failed replacements cannot delete valid contributions.

Filesystem ingestion is event-driven. Native live components observe committed
source transitions and reload signals without a periodic timer. Remove the old
eight-chunk scheduling cadence once native component bounds replace it.
Readiness includes publication of the corresponding validated graph.

Verified public lifecycle for CocoIndex 1.0.26: after a live update, await a
catch-up `app.update().result()` on the same app, then finish consuming the original
live handle. `watch()` consumes its final result; do not also call `result()` on
that handle. This terminates its live sources without private cancellation
APIs. Close the environment and provided database only afterward. A Windows
real-file test confirms edits after this transition are not ingested until a
subsequent catch-up. Timeout expiry alone did not terminate the idle live handle.
Runtime integration must also verify provider draining and graph catch-up.

RLM tools inspect proposed graph state and return typed repairs without mutating
the published graph. Retrieval/inspection connections are read-only. Replace
query-triggered indexing with explicit candidate preparation. Preserve named
DSPy traces on memo hits, with zero newly consumed inference tokens.

### Declarative Target Schemas for Epistemic Confidence (`/cocoindex`)

Under CocoIndex v1, graph rows are declared through typed target classes (`TableSchema.from_class`), replacing all imperative DDL and manual schema migrations in `yaw/daemon/graph/storage/db.py`:

```python
@dataclass
class EntityTargetRow:
    entity_id: str
    name: str
    definition: str
    confidence: float | None = None  # prov:value prior in (0, 1] (NULL if unknown)


@dataclass
class FactTargetRow:
    fact_id: str
    subj_id: str
    pred: str
    obj_id: str | None
    raw_json: str
    definition: str
    kind: str  # 'entity' | 'literal'
    lexical: str | None = None
    datatype: str | None = None
    language: str | None = None
    confidence: float | None = None  # prov:value prior in (0, 1]


@dataclass
class FactSourceTargetRow:
    fact_id: str
    chunk_id: str
    llm_prob: float | None = None
    vec_coherence: float | None = None
    atomic_confidence: float | None = None


@dataclass
class EdgeTargetRow:
    src_id: str
    dst_id: str
    kind: str  # 'fact' | 'skos:broader' | 'skos:related' | 'skos:topConceptOf' | 'owl:differentFrom' | 'passage'
    weight: float  # Bayesian compound confidence in (0, 1)


@dataclass
class EdgeSourceTargetRow:
    src_id: str
    dst_id: str
    kind: str
    chunk_id: str
    atomic_confidence: float | None = None


@dataclass
class ConceptSchemeTargetRow:
    scheme_id: str
    pref_label: str
    definition: str
    history_note: str
    confidence: float | None = None  # prov:value = pool_confidences(p_llm, cos(v_scheme, mu_m), c_mod)


@dataclass
class ConceptSchemeMemberTargetRow:
    scheme_id: str
    entity_id: str
    is_top_concept: int = 0
```

### Epistemic Ingestion Pipeline Dataflow

1. **Extraction Component (`@coco.fn(memo=True)`)**:
   - `ExtractEntities` and `ExtractFacts` capture length-normalized token logprob probabilities $p_{\text{llm}} = \exp\left(\frac{1}{T}\sum \log P(y_t)\right) \in (0, 1]$ from the inference server payload (or `None` if logprobs are absent).
   - Memoized in `chunk_inference` targets alongside structured triples.
2. **Embedding Component (`@coco.fn(memo=True)`)**:
   - Computes float32 dense vectors for chunks (`chunk_vec`), entities (`entity_vec`), and facts (`fact_vec`).
   - Grounding coherence $s_{\text{vec}} = \max(\epsilon, \cos(\mathbf{v}_{\text{item}}, \mathbf{v}_{\text{chunk}}))$ is evaluated at zero additional provider calls.
3. **Graph Materialization & W3C Guardrail Component (`plan_graph_publication`)**:
   - Computes atomic confidence $c_i = \text{pool\_confidences}(p_{\text{llm}, i}, s_{\text{vec}, i})$ for each mention.
   - Suppresses scalar literals from becoming entity nodes (`obj_entity_id = None`); literal values are stored strictly on `FactTargetRow` (`kind = 'literal'`) with their own `confidence`.
   - Enforces the Four W3C Semantic Guardrails during disambiguation (`prepare_synonym_candidates` and `apply_synonym_adjudication`): case-insensitive identity consolidation (`canary == Canary`), head-noun modifier protection against mega-hub `skos:altLabel` collapse (`Redwood` vs. `Redwood Router`), genus-species falsification in `JudgeHierarchy`, and homograph-only `owl:differentFrom`.
   - Aggregates multi-mention evidence via `bayesian_noisy_or` ($1 - \prod (1 - c_i)$).
4. **Emergent `skos:ConceptScheme` Modularity Clustering & Definer (`build_concept_schemes`)**:
   - Partitions the post-healing `skos:Concept` graph into $K^*$ communities via greedy Newman Modularity maximization ($\Delta Q \le 0$ stopping rule, zero magic numbers) on $A_{uv} = \text{pool\_confidences}(w_{\text{edge}}(u, v), \cos(\mathbf{v}_u, \mathbf{v}_v))$.
   - Selects top concepts $T_m$ per scheme and emits typed `skos:topConceptOf` edges into `edges`.
   - Invokes `entity_definer` (`DefineMissingEntities` / `define_concept_schemes`) to synthesize `[scheme_id, pref_label, definition, history_note]`, computes `prov:value` via `pool_confidences(p_llm, cos(v_scheme, mu_m), c_mod)`, and embeds each `skos:ConceptScheme` in `entity_vec` (`scheme-...`) for fallback PPR seeding and QA reader framing.
5. **Idempotent Graph Target Sink (`GraphTargetHandler`)**:
   - Applies declared target rows and executes closed-world validation `lint_graph_shacl(conn)` including `ProvenanceConfidenceShape`.
   - Any unresolved violation rolls back atomically, preserving existing published rows.

## Removal sequence

### Active replacement packet (2026-11-04)

Replace engine-owned `index_chunks` inference with a daemon-composed frozen
CocoIndex update using the same chunk operations, graph planner and atomic target
as live ingestion. Bind that operation explicitly to engines created for runtime,
benchmark and test evaluation. Unbound retrieval engines must fail if indexing is
requested. Frozen updates consume supplied chunk text and never rehydrate that
text from a mutable file. Remove the sequential inference/write implementation
in this packet; remaining query-triggered preparation and cache interfaces stay
unchecked until their consumers are replaced. Acceptance requires native live
and frozen graph equivalence, mutation/replay tests and all affected callers.

Remove each old implementation alongside its replacement: EmbeddingLane and its
executor/futures; indexing_overlap configuration; runtime polling thread; target
callbacks that run inference; serialized pending-delta replay; duplicate indexing
cache orchestration; published-graph mutation tools; query-time indexing. Retire
their exports, settings, tests and documentation together. Keep DSPy response
caching, optimizer checkpoints, SIMBA state, retrieval algorithms and domain SQL.

The daemon uses live mode; optimization and benchmarks use frozen catch-up
snapshots. Stop source scheduling, drain bounded provider work, then close the
database. Native readiness alone does not imply the graph has caught up.
The installed CocoIndex 1.0.26 can complete a root update while a mounted file
component failed. Runtime readiness must inspect the public update handle's
`stats().total.num_errors`, not merely await its result. The real SQLite test
confirms a malformed replacement retains the previous source rows and corrected
content succeeds on the next update; this does not yet prove partial-scan safety.

### Dataset snapshot consumers (`yaw/daemon/ingestion/datasets.py`)

`snapshot_dataset_bundle` (`yaw/daemon/ingestion/bundles.py`) still polls the
legacy `CorpusIngestor` and filters in-memory chunks by the `document_ids`
selected in `datasets.py` (`build_benchmark_slice` for `runtime.benchmark` and
`benchmark.py`; `load_dataset_splits` for `runtime.optimize`). Before rewiring
those consumers to native persistent snapshots—and queued behind the
`yaw dspy optimize` restart—make `datasets.py` lean in place so it holds only
the dataset slice and split logic that benchmarking and optimization need:

1. **No data literals in `yaw/`:** Move `MONKEYPATCHED_HIPPORAG_BASELINES`,
   `PAPER_BASELINES`, and root `benchmark_gate.json` verbatim into one tracked
   `data/benchmarks/baselines.json` file; delete the Python dicts and root
   `benchmark_gate.json`. Auditing baseline provenance and slice mismatches is
   a separate gate-trustworthiness spec.
2. **No invented behaviour:** Remove keyword-overlap gold-passage synthesis and
   `TEXT_STOPWORDS` from `get_gold_docs` and `build_benchmark_slice`. Treat
   `lveval` and `narrativeqa` as QA-only like upstream HippoRAG (`gold_docs=None`,
   full corpus selected for the snapshot slice, `Recall@k` omitted rather than
   defaulted to `0.0`).
3. **No duplicated or dead code:** Collapse `extract_gold_answers` to a single
   rule, share one passage splitter, and drop unused or duplicate exports
   (`DATASET_DESCRIPTIONS`, redundant `ALIASES` / `DATASET_ALIASES`).
4. **Behaviour preserved:** `build_benchmark_slice` and `load_dataset_splits`
   return identical outputs before and after for every dataset except `lveval`
   and `narrativeqa`, verified by a snapshot test on real prepared bundles.

Moving benchmark and dataset code (`datasets.py`, `runtime.benchmark`,
`yaw dspy benchmark`, baselines, and gate evaluation) out of `yaw/` into a
dev-only `benchmarks/` package—while `yaw dspy optimize` remains in the app
consuming a dataset-neutral prepared file—stays a recorded future extraction
once the native snapshot cutover is complete.

## Acceptance

Graph-target draft status (2026-10-29): the persistence-only native target passes
real SQLite tests for unchanged replay, changed source contributions under the
same chunk ID, replacement by a different chunk ID and removal across reopened
environments. Native tracking records retain per-source contribution fingerprints;
replaced contributions are removed in the same transaction as desired rows are
applied. It is not connected to runtime. Global semantic relation reconciliation,
shared-source cases and bounded full-corpus publication still require verification.
These tests do not establish the complete graph-publication gate.

### Vector-storage review (2026-10-29)

Dense vectors now use little-endian float32 bytes in both the memoized inference
result and the native SQLite inference artifact; JSON float arrays and Python
float tuples have been removed from this draft path. This reduces representation
overhead but does not establish single-copy storage. The old runtime has not yet
been connected to these artifacts; a third graph copy is a prospective integration
problem, not evidence that this draft already publishes three copies per run.

Keep outer file/chunk orchestration non-memoized and semantic embedding/OpenIE
identities independent. The restart test verifies a changed program identity
repeats only the two OpenIE calls, retaining the matching embedding.

Do not move provider calls into target sinks to avoid memo storage, and do not
write SQLite from a memoized inference function returning a receipt. Both break
the accepted transformation/publication boundary. Removing the intermediate
vector payload remains part of final graph-target integration and must preserve
recovery of completed inference. Native row tracking fingerprints are not a
measurement of total CocoIndex disk usage: keys, metadata and memo entries also
consume space. Measure actual database sizes before claiming a storage ratio.

Incremental implementation evidence: the native embedding component has a real
loopback-HTTP test demonstrating aggregation of four calls and zero additional
provider calls after reopening the CocoIndex environment. Typed vectors and
request telemetry survive memoization. This is same-process environment restart
evidence, not fresh-process crash recovery or runtime integration. Native batch
bodies run outside component context; resolve resources in the calling component
and use the SDK's bound batched method interface. Batch sizes are scheduler-dependent.
The native OpenIE component also passes a loopback test for concurrent HTTP
admission alongside embeddings, ordered NER/triples, named trace restoration and
zero extra calls on an unchanged update. A server rendezvous isolates concurrency
from client startup costs; this is not throughput evidence. The legacy runtime
does not yet call these components, so its executor cannot yet be retired.

Both native inference and live-ingestion gates are mandatory. Test real local
HTTP, SQLite and Windows file events, including fresh-process interruption at
each durability boundary. Report orchestration evidence separately from model
quality. The optional available-provider pilot is limited to 16 frozen chunks
and 20 minutes, uses disposable state, and never starts GEPA or stops another run.
No automatic commit. Epic 1 D1-A6's Tier 2 gate remains outstanding.

## Native CocoIndex Architectural Proposals (2026-11-04)

Building on the measured 5x throughput gain from native CocoIndex component execution (`5.39 chunks/min`), the following five architectural proposals formalize the remaining cutover items and align production contracts with local resource bounds.

### 1. Chunk-Stream Batching & Incremental Graph Publication (64 Chunks)

2026-11-05 scoped implementation: additive semantic planning now reuses the
validated graph when every previously published chunk's input fingerprint still
matches. Fingerprints cover source text, pointer, extraction and semantic settings,
and are committed atomically with the graph. New chunk contributions alone enter
sense resolution and synonym candidate search; existing senses and adjudicated
relations remain available. Derived community taxonomies are recomputed separately.
Synonym evidence is consumed one packet at a time rather than gathering every
packet and queued request in memory. The native publication memo version is 2.

This is deliberately an additions-first step. Edits, deletions, semantic-setting
changes and graphs without verified fingerprints require complete semantic
recomputation. Graph cloning, source hydration, complete target declarations,
SHACL validation and community clustering still scale with the published graph;
64-chunk dispatch does not yet establish flat heap usage. The declarative per-table
cutover below and Epic 1 D1-A6 remain unfinished.

- **Problem:** Currently, `frozen_indexing_app_main` queues the entire snapshot pool (e.g., 11,507 chunks) in a single pass. Graph publication (`publish_graph`) only triggers after *all* handles reach `.ready()`. Downstream readers, query interfaces, and evaluation harnesses are completely blocked from seeing graph entities, facts, or concept schemes until the final chunk extraction finishes. Production environments do not have static dataset splits or fixed validation chunk counts.
- **Proposal:** Replace the monolithic all-chunks pass with a general chunk-stream batching contract:
  - Chunks are dispatched in streaming batches of **64 chunks** (configurable via `indexing_batch_size`).
  - Each batch completes inference into `chunk_inference` and triggers an incremental declarative target reconciliation and graph publication via `plan_graph_publication`.
  - **Invariants:**
    - Downstream query readers and evaluation loops can begin querying the incrementally published graph after the first 64 chunks land, drastically improving daemon responsiveness.
    - SQLite write transactions remain short, atomic, and bounded, avoiding large memory staging trees and minimizing lock hold times.
    - Zero benchmark-specific splits or heuristic constants in the ingestion engine.

### 2. Universal Declarative Targets for SQLite and sqlite-vec (`TableSchema.from_class`)

- **Problem:** While `source_chunks` and `chunk_inference` use CocoIndex's native `sqlite.TableTarget`, graph entities, facts, labels, edges, and concept schemes in `yaw/daemon/graph/storage/publication.py` are still written via manual imperative SQL helper functions (`upsert_entities`, `upsert_facts`, `upsert_edges`, etc.) inside a custom `GraphTargetHandler`. Hand-rolled DDL, manual schema versioning, and imperative upsert/deletion statements violate the single source of truth.
- **Proposal:** Establish pure declarative target schemas for **everything** landing in `yaw.db` (`sqlite` and `sqlite-vec`):
  - Every table is declared as a typed dataclass mounted via `sqlite.mount_table_target(..., table_schema=await TableSchema.from_class(...))`:
    - `chunks` (`ChunkTargetRow`)
    - `chunk_vec` (via `declare_vector_index` on `Annotated[NDArray, EMBEDDER]`)
    - `entities` (`EntityTargetRow`)
    - `entity_labels` (`EntityLabelTargetRow`)
    - `entity_vec` (via vector index declaration)
    - `facts` (`FactTargetRow`)
    - `fact_sources` (`FactSourceTargetRow`)
    - `fact_vec` (via vector index declaration)
    - `edges` (`EdgeTargetRow`)
    - `edge_sources` (`EdgeSourceTargetRow`)
    - `concept_schemes` (`ConceptSchemeTargetRow`)
    - `concept_scheme_members` (`ConceptSchemeMemberTargetRow`)
  - **Invariants:**
    - CocoIndex's reconciliation engine owns state diffing, insertions, deletions, and cleanup across both graph and vector tables.
    - Closed-world SHACL linting (`lint_graph_shacl`) executes directly within the atomic target transaction, rolling back on invalid states.
    - Zero imperative SQL DDL or manual schema migrations in `yaw/daemon/graph/storage/`.

### 3. OpenIE Concurrency Contract & Local Resource Bounds

- **Problem:** While `EmbeddingEndpoint.embed` uses `@coco.fn.as_async(batching=True, max_batch_size=64)` and `Semaphore(4)`, OpenIE extraction requires careful admission control. On developer workstations, local LLM backends (`llama.cpp` server) fit only a single concurrent execution slot (`parallel = 1`) without exceeding GPU VRAM or thrashing context windows.
- **Proposal:** Establish an explicit runtime concurrency contract:
  - Local developer execution defaults strictly to `admission = asyncio.Semaphore(1)` for `OpenIEEndpoint`.
  - Expose `openie_parallel_slots: int = 1` in `YawConfig`. If connected to a multi-slot server (e.g., `llama.cpp --parallel N` or `vLLM` continuous batching), CocoIndex dynamically scales up to $N$ concurrent components without altering pipeline code.
  - Embedding requests continue to leverage native batched tensor evaluation (`max_batch_size=64`) since GPU memory for embeddings is negligible compared to LLM autoregressive kv-caches.

### 4. Native OS Event-Driven Live Ingestion (`localfs.walk_dir(..., live=True)`)

- **Current State:** The low-level connector implementation (`localfs.walk_dir(..., live=True)` and `GraphDeltaCoordinator`) is implemented in `yaw/daemon/indexing.py`. However, its integration into the main `YawRuntime` lifecycle and complete removal of legacy polling loops remains an unchecked item on the restart checklist.
- **Proposal:** Finalize the runtime live ingestion wiring:
  - `YawRuntime` starts `localfs.walk_dir(corpus_root, live=True, rescan_interval=None)` as the primary source feed for corpus files, and monitors `.yaw/reload.signal` via the same native event-driven mechanism.
  - On Windows, file creation, modification, rename, and deletion events are captured directly via OS file notifications (`ReadDirectoryChangesW`) through CocoIndex's `LiveMapView`.
  - Completely remove all residual polling timers, `auto_refresh` intervals, and manual rescan logic from `YawRuntime`.
  - Acceptance: Pass automated Windows integration tests demonstrating instant, event-driven ingestion upon file save with zero polling overhead.

### 5. Persistent Content-Addressed Memoization Across Optimizer Runs

- **Current State:** `embed_text` and `extract_text` already define `@coco.fn(memo=True)` keyed on `(source_hash, program_identity)`. However, `bind_native_indexing` currently derives the CocoIndex state root dynamically:
  ```python
  state_root = Path(engine.db_path).resolve().parent / "candidate_coco" / Path(engine.db_path).name
  ```
  Because each optimization run generates a new datestamped database (e.g., `yaw_optimize_20261104_132656.db`), the `state_root` directory is transient. A restarted optimizer run creates a fresh, empty CocoIndex database and fails to reuse memoized extractions from previous runs, despite the underlying texts and program signatures being identical.
- **Proposal:** Decouple CocoIndex's persistent environment state root from transient sandbox DB names:
  - Establish a shared, content-addressed cache directory: `.yaw/coco_inference_cache/`.
  - Because CocoIndex memoization keys deterministically on `(input_data, function_code, program_identity, embedding_identity)`, sharing this state root across sandbox instances is cryptographically sound and safe against cache poisoning.
  - When restarting optimization runs or evaluating candidate prompts that modify only QA/retrieval instructions without altering OpenIE extraction, 100% of chunk inferences hit the persistent memo cache instantly, consuming zero inference tokens and accelerating candidate cycles.
