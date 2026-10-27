# Epic 2 — Sprint 2: MCP Tools and Transport Acceptance

Parent: [Epic 2](epic-2-fastmcp-http.md) | Prerequisite: [Sprint 1](epic-2-sprint-1-spec.md) | Constitution: [FOUNDATIONAL](../FOUNDATIONAL.md)

Status: planned, reconciled 2026-10-27 against `dev`. Sprint 1 and Epic 1 `D1-A6`
acceptance are prerequisites, not claims of completed work. This sprint completes
the parent's five tool contracts and verifies `E2-A2`/`E2-A3` over real TCP.

## 1. Thin transport implementation

Register exactly `retrieve`, `rag_qa`, `index_stats`, `daemon_status` and
`sync_sources`, with the parent's schemas and all four annotations. Delegate to
public runtime APIs; never import graph/ingestion or own a second database writer.

Await retrieval/QA futures with `asyncio.wrap_future`. Status/statistics read
cached snapshots directly. `sync_sources` returns its accepted/already-pending
receipt immediately; it does not keep the tool request open until indexing ends.

Use strict transport types so schema validation cannot coerce `true`, `"5"` or
`5.0` to integer `top_k` before runtime validation. Preserve the same rules for
direct callers. Map invalid tool arguments, synchronous runtime failures and
future failures to the parent's seven codes with `isError=true` and `[CODE]`
prefixes. Malformed JSON-RPC remains a protocol error, not a runtime tool failure.
Redact credentials in tool errors, status, readiness and exposed exception details.

## 2. Progress, cancellation and sync receipts

Bridge synchronous callbacks into the active loop with
`loop.call_soon_threadsafe` and a bounded per-request progress channel. Await
`Context.report_progress` through public APIs of the locked FastMCP version.
Preserve runtime stage order; drain successful completion progress before returning.

- Successful retrieval/QA with a progress token emits exactly `queued` (1/3),
  `started` (2/3), `ready` (3/3), even for immediate operations.
- Errors/cancellation emit only the valid prefix reached, never fabricated
  `ready`. Failed admission emits no `queued`. Calls without progress tokens
  still work and must not retain unused progress state.
- Accepted sync receipts may emit `queued` after admission. `already_pending`
  refers to the existing scan, not a second admission. Read later scan state
  through `daemon_status`; never notify an already returned tool context.
- Explicit request cancellation cancels pending work and reclaims its slot
  immediately. Already running work finishes safely without delivering to that caller.
- Test explicit MCP cancellation and dropped connections separately. Closing HTTP
  is not necessarily session-request cancellation. Document the locked transport's
  observed behavior; where cancellation propagates, apply the same queue rules.
  A disconnected session may retain work according to protocol behavior, but must
  not orphan request-local progress tasks.
- Detach callbacks and stop/await progress tasks on every completion/failure/
  cancellation path. Never call a closed loop or let a slow progress consumer
  block the database worker.

## 3. Loopback Host and Origin guard

Implement with Sprint 1's mount; exercise the full matrix here. Apply to `/mcp`,
`/healthz` and `/readyz` before route dispatch.

- Validate configuration as loopback; it cannot extend trust to arbitrary hosts.
  Recognized request hostnames: `localhost`, `127.0.0.1`, `::1`. Parse
  authorities correctly, including bracketed IPv6 and optional valid ports;
  do not split naively on `:`.
- Reject missing, malformed, duplicate or untrusted Host headers with HTTP 403.
- Missing Origin is valid for non-browser clients. When present, require one
  well-formed HTTP(S) origin with a recognized loopback hostname and valid optional
  port. Reject `null`, userinfo, unexpected path/query/fragment, duplicate/multiple
  origins and external domains. Never use hostname suffix matching.
- Return a small sanitized 403 response. No wildcard CORS, stdio/SSE compatibility
  or remote deployment support.

## 4. Acceptance tests

Use workspace engineering-real-tests and CocoIndex skills. Run a real ASGI/FastMCP
server on an ephemeral loopback port, real `fastmcp.Client` sessions, temporary
SQLite/files, CocoIndex and a loopback HTTP inference provider. Do not mock
storage or runtime methods. Synthetic provider outputs establish orchestration
only; model quality remains the user's benchmark work.

| Test packet | Required evidence |
| --- | --- |
| Lifespan/containment (E2-A1 regression) | Health/readiness during blocked initial sync, empty-ready, failure and stopping; bind/startup cleanup; lock reacquisition; adapter import boundary |
| Five tools (E2-A2) | Exact names, schemas and four annotations; calls without progress; ordered immediate/slow success; valid prefixes on error/cancellation |
| Citations (E2-A2) | Actual text/physical coordinates for Markdown/code and prepared JSON; shared-record subchunks, namespaced paths and MaxP coalescing; unchanged fresh-process restart; edit/delete/truncate detection without excluded-data leaks; text hashes and QA evidence match captured predictor context |
| Errors (E2-A2) | All seven codes from real validation/provider/storage/lifecycle conditions; empty/oversized UTF-8 and non-string query, top_k bounds/bool/string/float; lifecycle before empty-index errors; redaction |
| Responsive reads (E2-A3) | Two clients; daemon_status, index_stats, /healthz and /readyz finish within two seconds while inference remains blocked on an explicit event |
| Capacity/cancellation (E2-A3) | One blocked active operation plus 32 actual pending items; overflow BUSY; pending cancellation reduces depth and permits replacement before releasing the worker; cancelled work never executes; running cancellation finishes safely |
| Atomic sync (E2-A3) | Simultaneously released clients get one accepted and remaining already_pending receipts with one scan ID; one actual scan; failed admission leaves no reservation; failed scans retry; partial scans never delete sources |
| Live update (E2-A3) | CocoIndex source edit plus sync; new consistent citation/provenance after commit; unchanged warm scans reuse persistent results |
| Security/disconnects (E2-A3) | Raw duplicate headers, IPv6, missing Origin, hostile hosts and non-loopback bind rejection; separate cancellation/disconnect/reconnect evidence; no leaked progress tasks |

Use barriers/events to control concurrency, not short sleeps and tighter-than-
contract latency thresholds. Release events in `finally` blocks and bound waits.
Assert status arrives **before** releasing inference as well as measuring the
two-second bound. Small-capacity unit cases supplement but do not replace the
real 32-slot acceptance case.

Send raw tool-call arguments when client schema validation would hide server
behavior. Verify tool-result errors independently of JSON-RPC parse errors.
Exercise failures with real invalid bytes, provider errors and resource/lifecycle
conditions, rather than patching runtime methods.

## 5. Delivery and verification

Changes belong in `yaw/daemon/mcp_server.py`, affected runtime/core contracts,
`tests/daemon/test_mcp_http.py` and runtime/architecture tests. Replace obsolete
paths rather than retaining adapters. README must describe only delivered commands
and verified endpoints; keep target commands separate.

After implementation, run affected tests with testmon and repository checks:

```powershell
uv run --locked pytest tests/core/test_architecture.py tests/daemon/test_runtime.py tests/daemon/test_mcp_http.py -vv
uv run --locked preflight
git diff --check
```

Report selection/deselection and coverage scope. If testmon deselects required
acceptance cases, run them explicitly with selection disabled and record why;
deselection is not evidence that a new acceptance condition passed. Do not hide
unavailable resident-provider prerequisites. Record results, failures, versions,
Windows lifecycle evidence and observed disconnect semantics in
[evaluation](evaluation.md); mark E2-A2/E2-A3 only when their tests pass.
Do not launch GEPA or benchmarks or infer quality gains from loopback tests.
