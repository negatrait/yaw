# Native CocoIndex indexing and live ingestion

Accepted 2026-10-28; completed 2026-10-31 on `dev`.

`YawRuntime` (daemon live mode, `sync_now`, `yaw dspy optimize`, and `yaw dspy benchmark`) and `snapshot_dataset_bundle` now execute through the single declarative CocoIndex pipeline (`indexing_app_main`), with all competing ingestion/caching/polling paths and fallback degradations removed.

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
- [x] Native filesystem live source and declarative delta/reload components.
  `localfs.walk_dir(..., live=True, rescan_interval=None)` drives both corpus files and
  `.yaw/reload.signal`, and `GraphPublicationLiveComponent` + `GraphDeltaCoordinator` publish
  committed state-diff transitions without `auto_refresh` or timer polling.
- [x] Complete persistent snapshots and all consumer updates.
  `snapshot_dataset_bundle` runs on `indexing_app_main`, and `yaw/daemon/ingestion/datasets.py`
  is dataset-agnostic, reading all dataset metadata, source-field mappings, slice traits, and
  baselines from `data/benchmarks/baselines.json`.
- [x] Remove custom executor, polling and overlap switch.
  Removed `EmbeddingLane`, `indexing_overlap`, `CorpusIngestor`, `sync_corpus_once`,
  `YawRuntime._poll_loop`, `openie_cache`, `ReActV2` QA fallback, `"degraded"` lifecycle mode,
  and all silent parser/proposer/RLM fallback swallowers.
- [x] Real SQLite/HTTP recovery, equivalence and bounded-memory tests.
  Fresh-process interruption, SHACL rollback preservation, and unchanged replay with zero
  additional provider calls pass in `tests/daemon/graph/test_indexing_overlap.py`.
- [x] Windows live-ingestion lifecycle and failure tests.
  Live filesystem edit/rename/delete, `.yaw/reload.signal`, and live-to-catch-up shutdown pass
  on Windows.
- [x] Ruff, ty, affected tests, architecture and diff checks.
- [x] Bounded live comparison and actual evidence in evaluation.md.

## Contract

CocoIndex schedules work through mounted components, native batching and
auto_refresh. Four processing components, one in-flight request per endpoint,
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

Filesystem ingestion is event-driven. Native auto_refresh coalesces graph
materialization and reload-signal maintenance at 10 seconds. Remove the old
eight-chunk scheduling cadence once native component bounds replace it.
Readiness includes publication of the corresponding validated graph.

Verified public lifecycle for CocoIndex 1.0.26: after a live update, await a
catch-up `app.update().result()` on the same app, then await the original live
handle's result. This terminates its live sources without private cancellation
APIs. Close the environment and provided database only afterward. A Windows
real-file test confirms edits after this transition are not ingested until a
subsequent catch-up. Timeout expiry alone did not terminate the idle live handle.
Runtime integration must also verify provider draining and graph catch-up.

RLM tools inspect proposed graph state and return typed repairs without mutating
the published graph. Retrieval/inspection connections are read-only. Replace
query-triggered indexing with explicit candidate preparation. Preserve named
DSPy traces on memo hits, with zero newly consumed inference tokens.

## Removal sequence

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
