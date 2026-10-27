# Epic 2 — Sprint 1: Runtime Contracts and HTTP Lifespan

Parent: [Epic 2](epic-2-fastmcp-http.md) | Constitution: [FOUNDATIONAL](../FOUNDATIONAL.md) | Next: [Sprint 2](epic-2-sprint-2-spec.md)

Status: planned, reconciled 2026-10-27 against `dev`. Implementation starts only
after recorded Epic 1 `D1-A6` acceptance. GEPA and benchmarks remain user-managed.
The parent owns public tool/transport contracts; this sprint establishes their
runtime prerequisites and `E2-A1` evidence.

## 1. Dependency and containment packet

- Add FastMCP deliberately to [pyproject.toml](../pyproject.toml), resolve with
  `uv lock`, install with `uv sync --locked`, and verify the locked package on
  Python 3.14.9. Use that version's public APIs; old isolated resolution evidence
  does not prove project installation or transport behavior.
- Extend [architecture tests](../tests/core/test_architecture.py) with a
  module-specific rule for `yaw.daemon.mcp_server`: internal imports limited to
  `yaw.daemon.runtime` and `yaw.core.*`. Test allowed/rejected imports.
  No private cross-module access or relaxed domain rules.
- Exit: locked import and boundary tests pass, with versions/results recorded.

## 2. Source provenance and citation packet

### Separate identity, source freshness and returned-text identity

`chunk_id` is not a source freshness checksum. Projected JSON subchunks can share
one physical record span, hydration can coalesce/expand spans, and bundle paths
can be namespaced after chunk construction. Recomputing a chunk ID from hydrated
text and its returned path therefore marks unchanged content stale.

Add mandatory `source_content_hash` provenance to `ChunkPointer` and its
persisted chunk representation: `sha256:` plus SHA-256 of the original source
file bytes used to parse the chunk. All chunks from the same source snapshot share
it. Capture bytes and fingerprint together; do not reread after parsing to assign
a potentially newer hash. No source text is stored in SQLite.

Carry the fingerprint through typed ingestion results, `SyncDelta`, persistent
CocoIndex memo results, snapshots, optimization pool/cache serialization and graph
commits. Commit each chunk and its fingerprint together, including resumed or
partial indexing batches; a mutable per-file digest must not relabel old chunks
as newly indexed. Update all constructors, consumers and persisted-format
identities. Require fresh incompatible indexes/caches with a clear diagnostic;
no default-empty provenance, migration shim or reset of unrelated experiments.

At hydration, read each distinct source once per request, hash that same byte
snapshot and validate complete physical span bounds before decoding/projection.
Bound retained buffers; never keep a resident corpus copy. Metadata-only file
stats are not proof of unchanged bytes. Preserve strict JSON whitelist/blacklist
projection, nested exclusions and bundle document/evaluation isolation.

Missing, changed or truncated sources produce `stale_on_disk=True` and empty
returned text while preserving the indexed source pointer. Do not expose changed
raw JSON as a fallback. Propagate other read/parser failures explicitly. When
coalescing passages, validate every contributing pointer's source fingerprint;
aggregation must not erase stale state or mix incompatible source versions.
Freshness describes the source snapshot read for this request, not a promise that
an external writer cannot edit the file afterward.

### One typed hydration and serialization path

- Refactor graph retrieval to return `HydratedPassage` without discarding
  stale/empty hits. Refactor callers rather than adding compatibility wrappers.
- Serialize centrally in the runtime using the parent's exact schema.
  `content_hash = "sha256:" + sha256(text.encode("utf-8")).hexdigest()` describes
  returned text, including empty stale text; it is not the indexed source
  fingerprint. Keep physical coordinates and valid indexed pointers.
- QA uses verified fresh passages and returns evidence from the same captured
  context actually supplied to its predictor. Do not retrieve again to fabricate
  citations, use zero coordinates for missing pointers or hardcode freshness.
  If stale sources leave no usable context, return `CONSISTENCY_ERROR` requesting
  sync; a genuinely empty ready index returns `EMPTY_INDEX`. Unusable pointers
  are consistency failures, not fabricated evidence.
- Exit: real prepared JSON subchunks, namespaced paths, Markdown/code spans and
  coalesced passages survive an unchanged restart without false stale reports;
  edits/deletion/truncation are detected without exposing excluded data.

## 3. Queue, sync and lifecycle packet

Own a small bounded pending FIFO with explicit admission, removal, dequeue and
shutdown methods, for example a `collections.deque` protected by a
`threading.Condition`. Do not mutate Python queue internals. Capacity defaults to
32 pending items, excluding the active operation; status must reflect that rule.

- Admission and dequeue/cancellation have explicit atomic boundaries. A cancelled
  pending future is physically removed immediately, releases capacity and never
  executes. No cancelled tombstones accumulate. If execution won the race,
  complete its transaction and detach the caller.
- Ensure `queued` precedes `started`, even with an idle worker and immediate
  operation. Serialize per-operation stage delivery; emit `ready` only after
  success. Invoke external callbacks outside queue/state locks. Callback failures
  must not kill the worker or leak capacity.
- Atomically reserve and publish the pending-sync future/scan ID with admission.
  Concurrent `request_sync()` callers see one pending/running scan and the same
  ID. Roll back reservations on failed admission; clear on completion,
  cancellation or failure. Retain partial-scan deletion protection.
- Validate runtime inputs and lifecycle/error precedence as specified by the
  parent, including admission/shutdown races.
- Exit: event-controlled tests prove immediate slot reuse, one coalesced scan,
  valid progress ordering and recovery after failure.

## 4. Guarded HTTP mount and daemon packet

Create `yaw/daemon/mcp_server.py` with a server factory and one HTTP runner wrapping
the existing runtime. Use existing host/port/path settings, reject non-loopback
binds and install Sprint 2's Host/Origin guard immediately.

Support deferred initial sync through a public runtime startup API. HTTP must bind
while sync remains in `starting`; successful sync transitions to `ready`,
including an empty corpus. Perform blocking startup/stop work off the ASGI event
loop. Startup failure must leave readiness false and release resources acquired
so far; a bound socket alone does not prove success.

Expose public runtime stop observation; never inspect its private event. Preserve
Windows named-event and POSIX signal behavior. On stop, reject new work, settle
pending futures, finish the active transaction, join the worker before closing
resources, then release the database lock. Route normal exit, failed startup,
bind failure and external stop through idempotent cleanup. Never leave an executor
thread waiting indefinitely for a stop event after server startup failure.

Wire existing `yaw daemon run` and the `yaw daemon start` child into this runner.
Preserve the CLI grammar and external inference service. No parallel headless CLI
server, duplicate database owner or transport shim. Health handlers read cached
runtime snapshots and expose only redacted errors.

## 5. Changes and removal inventory

| Area | Replacement and removal |
| --- | --- |
| Core types, ingestion, graph storage/engine and pool serialization | Durable source fingerprint/format identities; replace obsolete pointer construction/read paths |
| Hydration and runtime | Typed results and one serializer; remove tuple conversions losing freshness, invented pointers and hardcoded non-stale evidence |
| Queue and sync | Owned pending FIFO and atomic reservation; remove cancelled-capacity retention and check-then-publish races |
| Lifecycle, CLI and MCP module | Public deferred startup/stop observation; replace headless CLI wait with guarded HTTP lifespan |
| Dependencies and architecture | Locked FastMCP and module-specific boundary; no retired dependency or CLI redesign work |

The former no-SQLite-schema-change constraint is superseded by the minimal source
provenance extension. One contentless `yaw.db`, native DSPy, persistent CocoIndex
state and domain containment remain mandatory.

## 6. Verification and exit record

Use real files, SQLite, CocoIndex and loopback TCP with workspace CocoIndex and
engineering-real-tests skills. Extend runtime/ingestion/hydration tests and create
`tests/daemon/test_mcp_http.py` now for lifecycle/guard tests. Cover cancellation
races, sync admission failures, callback failures, deferred/failed startup, bind
failure, repeated stop, lock release and Windows stop signaling. An event-controlled
loopback provider holds initial sync open while health is 200 and readiness 503.

Run affected tests with the project's testmon default, architecture tests, Ruff,
ty and `git diff --check`; record selection/deselection and unavailable prerequisites.
Use `uv run --locked typecheck` for ty's fix policy. Report dependency resolution,
installation, runtime tests and real HTTP results separately in
[evaluation](evaluation.md). Mark `E2-A1` complete only from those results.
Do not run GEPA or the user's benchmark workloads as part of this packet.
