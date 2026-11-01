# Epic 2: FastMCP Streamable HTTP Adapter

Owner: [FOUNDATIONAL](../FOUNDATIONAL.md) | Prerequisite: [Epic 1](epic-1-hipporag-daemon.md) | Results: [evaluation](evaluation.md)

## Status and entry gate — reconciled 2026-10-27

This is a planned implementation, reconciled against `dev`. **Epic 1 acceptance
`D1-A6` remains open.** Runtime and ingestion checks do not establish Tier 2
full-corpus parity. Record disjoint holdout results, corpus/query coverage, model
identities and comparison against both consolidated targets in
[evaluation](evaluation.md) before starting Epic 2 implementation. A sampled gate
run is not full-corpus acceptance. GEPA and benchmarks are user-managed; this
reconciliation launches neither and does not waive the gate.

After that gate passes, the first implementation packet is Sprint 1 dependency
and containment verification. Its observable exit is a project-locked FastMCP
import on Python 3.14.9 and passing adapter-boundary tests. Package resolution,
installation, HTTP acceptance and model quality are separate evidence.

## Verified baseline and required work

| Area | Current code | Planned requirement |
| --- | --- | --- |
| Transport | No `fastmcp` dependency, `mcp_server.py` or MCP HTTP tests; host/port/path settings exist | Add the locked dependency and thin adapter; an old isolated dependency probe does not establish project compatibility |
| Ownership | One runtime, database lock, worker and bounded queue | Preserve ownership; serve HTTP in the same daemon process |
| Containment | Package-level rules allow daemon imports too broadly for the adapter | Enforce a module-specific boundary |
| Citations | Some hydration paths lose metadata; runtime reports freshness without a source-content comparison | Persist indexed source fingerprints and preserve typed hydration results |
| Queue | Cancelled futures retain capacity until consumed; admission precedes the queued callback | Own a cancellable pending queue; guarantee progress order and immediate reclamation |
| Sync | Pending-sync lookup and publication are separate critical sections | Atomically coalesce concurrent requests |
| Startup | Initial sync blocks startup; CLI runs a headless runtime | Expose health during deferred sync; couple shutdown through public APIs |
| Validation | Boolean top_k is accepted; empty-index validation can precede lifecycle validation | Strict inputs and consistent error precedence |

Prepared bundles and persistent CocoIndex snapshots are the current ingestion
contract. Scan document roots only; do not restore monolithic corpus discovery,
evaluation-field ingestion, cache resets or legacy-layout fallbacks. Preserve
generic production JSON projection and physical source coordinates.

## Containment and transport contract

- The new `yaw/daemon/mcp_server.py` may import `yaw.daemon.runtime`,
  `yaw.core.*` and public transport-library/stdlib APIs. No direct graph,
  ingestion, storage or inference imports, or cross-module private access.
- Default endpoint: `http://127.0.0.1:13306/mcp`, using existing HTTP settings.
  Validate a loopback bind address. No separate server process, stdio/SSE entry
  points, remote bind or wildcard CORS in this epic.
- One database owner and worker; default capacity is **32 pending operations**,
  excluding the active operation. Status reads bypass the queue.
- Validate Host and Origin before dispatch, including health routes. Sprint 2
  defines the guard; install it with Sprint 1's initial HTTP mount.
- `/healthz` returns 200 while HTTP is responsive. `/readyz` returns 200 only
  for `lifecycle == "ready"`, including a successfully synchronized empty corpus;
  otherwise 503. A bound socket does not prove successful startup.

## Five tool contracts

Query/question inputs must be strings, trimmed, non-empty and at most 1,024 UTF-8
bytes. `top_k` is a strict integer in 1–20, default 5; booleans and coerced
strings/floats are invalid. Runtime validation remains authoritative.

| Tool | Execution and result | readOnly / destructive / idempotent / openWorld hints |
| --- | --- | --- |
| `retrieve(query, top_k=5)` | Await queued retrieval; `{query, top_k, results: Passage[]}` | true / false / true / false |
| `rag_qa(question)` | Await queued QA; `{question, answer, reasoning, evidence: Passage[], last_sync_at: float or null}`; at most five evidence passages from the context actually used to answer | true / false / false / false |
| `index_stats()` | Cached `runtime.stats().to_dict()`; no worker/database wait | true / false / true / false |
| `daemon_status()` | Cached `runtime.status().to_dict()`, including redacted errors; no worker/database wait | true / false / true / false |
| `sync_sources()` | Immediate `{scan_id, status: "accepted" or "already_pending"}` receipt; concurrent requests share the pending/running scan | false / true / true / false |

`Passage` has `{rank, text, score, chunk_id, content_hash, stale_on_disk, source}`;
`source` has `{rel_posix_path, start_line, end_line, title}` with physical,
1-indexed coordinates. Preserve MaxP aggregation by canonical passage span.
`content_hash` is `sha256:` plus the SHA-256 of the **returned text's UTF-8 bytes**.
It is distinct from the durable indexed source fingerprint used for freshness.
The latter requires a small provenance schema extension specified in Sprint 1;
the database still stores no chunk text.

## Execution, progress and errors

- Await worker futures without blocking ASGI. On successful retrieval/QA,
  requested progress is `queued` (1/3), `started` (2/3), `ready` (3/3), in
  order. Failed/cancelled operations emit only the stages reached.
- `sync_sources` acknowledges admission rather than awaiting the scan. An
  accepted receipt may report `queued`; later state is read through
  `daemon_status`. Never retain its MCP request context after returning.
- Explicit cancellation before execution removes pending work immediately.
  Running work finishes its transaction; detach the cancelled caller. A transport
  disconnect is not automatically MCP cancellation: test both against the locked
  transport and clean up request-local progress tasks in every case.
- Tool failures use `isError=true` and `[CODE]` prefixes: `INVALID_INPUT`,
  `NOT_READY`, `EMPTY_INDEX` (QA only), `BUSY`, `MODEL_ERROR`,
  `CONSISTENCY_ERROR`, `SHUTTING_DOWN`. Redact secrets on all exposed paths.
  Validate input, then lifecycle, then operation-specific conditions/admission;
  an empty index cannot override a starting/stopping lifecycle error.

---

## Sprint 1 (`E2-S1`): Runtime Contracts and HTTP Lifespan

### 1.1 Dependency and containment packet

- Add FastMCP deliberately to [pyproject.toml](../pyproject.toml), resolve with
  `uv lock`, install with `uv sync --locked`, and verify the locked package on
  Python 3.14.9. Use that version's public APIs; old isolated resolution evidence
  does not prove project installation or transport behavior.
- Extend [architecture tests](../tests/core/test_architecture.py) with a
  module-specific rule for `yaw.daemon.mcp_server`: internal imports limited to
  `yaw.daemon.runtime` and `yaw.core.*`. Test allowed/rejected imports.
  No private cross-module access or relaxed domain rules.
- Exit: locked import and boundary tests pass, with versions/results recorded.

### 1.2 Source provenance and citation packet

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

- Refactor graph retrieval to return `HydratedPassage` without discarding
  stale/empty hits. Refactor callers rather than adding compatibility wrappers.
- Serialize centrally in the runtime using the exact schema above.
- QA uses verified fresh passages and returns evidence from the same captured
  context actually supplied to its predictor. If stale sources leave no usable
  context, return `CONSISTENCY_ERROR` requesting sync; a genuinely empty ready
  index returns `EMPTY_INDEX`.

### 1.3 Queue, sync and lifecycle packet

Own a small bounded pending FIFO with explicit admission, removal, dequeue and
shutdown methods (e.g. a `collections.deque` protected by a `threading.Condition`).
Do not mutate Python queue internals. Capacity defaults to 32 pending items,
excluding the active operation; status must reflect that rule.

- A cancelled pending future is physically removed immediately, releases capacity
  and never executes. If execution won the race, complete its transaction and
  detach the caller.
- Ensure `queued` precedes `started`, even with an idle worker and immediate
  operation. Serialize per-operation stage delivery; emit `ready` only after
  success. Invoke external callbacks outside queue/state locks.
- Atomically reserve and publish the pending-sync future/scan ID with admission.
  Concurrent `request_sync()` callers see one pending/running scan and the same
  ID. Roll back reservations on failed admission; clear on completion,
  cancellation or failure. Retain partial-scan deletion protection.

### 1.4 Guarded HTTP mount and daemon packet

Create `yaw/daemon/mcp_server.py` with a server factory and one HTTP runner wrapping
the existing runtime. Use existing host/port/path settings, reject non-loopback
binds and install the Host/Origin guard immediately.

Support deferred initial sync through a public runtime startup API. HTTP must bind
while sync remains in `starting`; successful sync transitions to `ready`,
including an empty corpus. Perform blocking startup/stop work off the ASGI event
loop. Startup failure must leave readiness false and release resources acquired
so far; a bound socket alone does not prove success.

Expose public runtime stop observation; never inspect its private event. Preserve
Windows named-event and POSIX signal behavior. On stop, reject new work, settle
pending futures, finish the active transaction, join the worker before closing
resources, then release the database lock. Wire existing `yaw daemon run` and the
`yaw daemon start` child into this runner.

---

## Sprint 2 (`E2-S2`): MCP Tools and Transport Acceptance

### 2.1 Thin transport implementation

Register `retrieve`, `rag_qa`, `index_stats`, `daemon_status` and `sync_sources`
with strict schemas and all four annotations. Delegate to public runtime APIs;
never import graph/ingestion or own a second database writer.

Await retrieval/QA futures with `asyncio.wrap_future`. Status/statistics read
cached snapshots directly. `sync_sources` returns its accepted/already-pending
receipt immediately. Use strict transport types so schema validation cannot coerce
`true`, `"5"` or `5.0` to integer `top_k`. Malformed JSON-RPC remains a protocol
error, not a runtime tool failure.

### 2.2 Progress, cancellation and sync receipts

Bridge synchronous callbacks into the active loop with `loop.call_soon_threadsafe`
and a bounded per-request progress channel. Await `Context.report_progress`
through public APIs of the locked FastMCP version. Detach callbacks and stop/await
progress tasks on every completion, failure, or cancellation path. Test explicit
MCP cancellation and dropped connections separately.

### 2.3 Loopback Host and Origin guard

Apply to `/mcp`, `/healthz` and `/readyz` before route dispatch.

- Recognized request hostnames: `localhost`, `127.0.0.1`, `::1`. Parse
  authorities correctly, including bracketed IPv6 and optional valid ports.
- Reject missing, malformed, duplicate or untrusted Host headers with HTTP 403.
- Missing Origin is valid for non-browser clients. When present, require one
  well-formed HTTP(S) origin with a recognized loopback hostname and valid optional
  port. Reject `null`, userinfo, unexpected path/query/fragment, duplicate/multiple
  origins and external domains. Return a sanitized 403 response.

### 2.4 Acceptance test matrix

| Test packet | Required evidence |
| --- | --- |
| Lifespan/containment (`E2-A1`) | Health/readiness during blocked initial sync, empty-ready, failure and stopping; bind/startup cleanup; lock reacquisition; adapter import boundary |
| Five tools (`E2-A2`) | Exact names, schemas and four annotations; calls without progress; ordered immediate/slow success; valid prefixes on error/cancellation |
| Citations (`E2-A2`) | Actual text/physical coordinates for Markdown/code and prepared JSON; shared-record subchunks, namespaced paths and MaxP coalescing; unchanged fresh-process restart; edit/delete/truncate detection without excluded-data leaks; text hashes and QA evidence match captured predictor context |
| Errors (`E2-A2`) | All seven codes from real validation/provider/storage/lifecycle conditions; empty/oversized UTF-8 and non-string query, `top_k` bounds/bool/string/float; lifecycle before empty-index errors; redaction |
| Responsive reads (`E2-A3`) | Two clients; `daemon_status`, `index_stats`, `/healthz` and `/readyz` finish within two seconds while inference remains blocked on an explicit event |
| Capacity/cancellation (`E2-A3`) | One blocked active operation plus 32 actual pending items; overflow `BUSY`; pending cancellation reduces depth and permits replacement before releasing the worker; cancelled work never executes; running cancellation finishes safely |
| Atomic sync (`E2-A3`) | Simultaneously released clients get one `accepted` and remaining `already_pending` receipts with one scan ID; one actual scan; failed admission leaves no reservation; failed scans retry; partial scans never delete sources |
| Live update (`E2-A3`) | CocoIndex source edit plus sync; new consistent citation/provenance after commit; unchanged warm scans reuse persistent results |
| Security/disconnects (`E2-A3`) | Raw duplicate headers, IPv6, missing Origin, hostile hosts and non-loopback bind rejection; separate cancellation/disconnect/reconnect evidence; no leaked progress tasks |

---

## Sprints and acceptance checklist

| Sprint | Work | Observable exit |
| --- | --- | --- |
| `E2-S1` | Dependency/boundary, provenance, typed hydration, cancellation/coalescing/order, deferred startup, guarded mount and existing CLI integration | Real loopback health/lifecycle tests; runtime/provenance tests; adapter containment |
| `E2-S2` | Five tools, progress/error mapping and complete multi-client acceptance | Real TCP schemas, citations, security, cancellation, capacity, coalescing and responsive status |

- [ ] **E2-A1:** One HTTP/runtime lifespan, database owner and public shutdown path;
  health/readiness observe startup, empty-ready, failure and shutdown; dependency
  and containment checks pass.
- [ ] **E2-A2:** Exactly five tools with specified annotations/errors; ordered
  successful progress; source-isolated citations and QA evidence with correct
  coordinates, hashes and freshness behavior.
- [ ] **E2-A3:** Two TCP clients share one writer; all four status/health surfaces
  respond within two seconds while inference remains blocked; actual 32-slot
  overflow and immediate cancellation reclamation; atomic sync coalescing;
  Host/Origin rejection and post-commit citation updates.

Use workspace CocoIndex and engineering-real-tests skills during implementation.
Record actual checks/failures in [evaluation](evaluation.md). Loopback inference
proves orchestration, not model quality or parity. Update README's available
commands only after the HTTP implementation passes.

## Deferred Epic 3 ideas — outside this epic

Retain these proposals for a separate decision: reorganizing CLI domains around
optimization/evaluation, numerical traversal tuning, grouping traversal/policy/
seeding settings, and editable seed instructions/demonstrations. They are neither
Epic 2 prerequisites nor accepted replacements for current interfaces.
