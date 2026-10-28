# Native CocoIndex indexing and live ingestion

Accepted 2026-10-28; implementation in progress on `dev`.

## Restart checklist

- [x] Remove the unconnected source_work journal draft; no callers or database migration existed.
- [ ] Built-in SQLite targets for source contributions and inference artifacts.
- [ ] Native inference components, memoization and embedding batching.
- [ ] Ordered graph planning outside network-spanning SQLite transactions.
- [ ] Idempotent graph target publishing validated graph changes atomically.
- [ ] Native filesystem live source and periodic graph/reload components.
- [ ] Complete persistent snapshots and all consumer updates.
- [ ] Remove custom executor, polling and overlap switch.
- [ ] Real SQLite/HTTP recovery, equivalence and bounded-memory tests.
- [ ] Windows live-ingestion lifecycle and failure tests.
- [ ] Ruff, ty, affected tests, architecture and diff checks.
- [ ] Bounded live comparison and actual evidence in evaluation.md.

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

## Acceptance

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
