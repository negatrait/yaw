# Epic 2: FastMCP Streamable HTTP Adapter

Owner: [FOUNDATIONAL](../FOUNDATIONAL.md) | Prerequisite: [Epic 1](epic-1-hipporag-daemon.md) | Results: [evaluation](evaluation.md) | Specs: [Sprint 1](epic-2-sprint-1-spec.md), [Sprint 2](epic-2-sprint-2-spec.md)

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

## Sprints and acceptance

| Sprint | Work | Observable exit |
| --- | --- | --- |
| [E2-S1](epic-2-sprint-1-spec.md) | Dependency/boundary, provenance, typed hydration, cancellation/coalescing/order, deferred startup, guarded mount and existing CLI integration | Real loopback health/lifecycle tests; runtime/provenance tests; adapter containment |
| [E2-S2](epic-2-sprint-2-spec.md) | Five tools, progress/error mapping and complete multi-client acceptance | Real TCP schemas, citations, security, cancellation, capacity, coalescing and responsive status |

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

The previous appendix proposed `yaw optimize {gepa|optuna|hybrid}`,
`yaw evaluate {benchmark|gate}`, `yaw config show`, Optuna sweeps and automatic
configuration writes. These conflict with FOUNDATIONAL's locked CLI and Optuna
retirement policy. Optuna still appearing in the manifest/lock is unresolved
policy drift, not authorization to use it. Resolve that separately before adoption.
No traversal-speed claim is accepted without measurement; configurable prompts
must preserve native DSPy signatures, types, schemas and persistence identities.
