# Epic 2: FastMCP Streamable HTTP Adapter

Owner: [FOUNDATIONAL](../FOUNDATIONAL.md) | Prerequisite: [Epic 1](epic-1-hipporag-daemon.md) | Next: [Epic 3](epic-3-performance.md) | Results: [evaluation](evaluation.md)

## Status and entry gate — reconciled 2026-10-27

This is a planned implementation, reconciled against `dev`. **2026-11-15 / 2026-11-16 Updates
([Epic 1 Decisions 19–20](epic-1-hipporag-daemon.md#decisions)):** Following the
validated `E1-M1` Pass-4 shootout and Decision 20's zero-backward-compatibility
clean-slate rebuild, the [Bare-Bones YAW Replacement Sprint](reviews/bare-bones-replacement-sprint.md)
replaces `yaw/daemon/**` internals onto four bare-bones layers where `FastMCP`
Streamable HTTP (`yaw/mcp_server.py`) and CLI sit directly on top of zero-LLM
`recall` (`VectorSearch` in `yaw/retrieval.py`) and bare `dspy.ReAct` `qa`
(`yaw/qa.py`).

Historic note on the prior pre-bare-bones baseline: **Epic 1 acceptance
`D1-A6` remained open under the legacy OpenIE/`ppr.py` path.** Runtime and ingestion checks do not establish Tier 2
full-corpus parity. Record disjoint holdout results, corpus/query coverage, model
identities and comparison against both consolidated targets in
[evaluation](evaluation.md) before starting Epic 2 implementation. A sampled gate
run is not full-corpus acceptance. GEPA and benchmarks are user-managed; this
reconciliation launches neither and does not waive the gate.

After that gate passes, the first implementation packet is Sprint 1 dependency
and containment verification. Its observable exit is a project-locked FastMCP
import on Python 3.14.9 and passing adapter-boundary tests. Package resolution,
installation, HTTP acceptance and model quality are separate evidence. Because
`yaw/mcp_server.py` delegates exclusively to `runtime.recall` /
`runtime.ask` / `runtime.get_status` and `ChunkPointer.source_hash`, this external
transport boundary remains decoupled from [Epic 3](epic-3-performance.md)'s
future retrieval and performance experiments.

## Verified baseline and required work

| Area | Pre-cutover baseline (`b8b65ec`) | Clean-slate planned requirement (`CS-3a`–`CS-3b`) |
| --- | --- | --- |
| Transport | No `fastmcp` dependency, `yaw/mcp_server.py`, or MCP HTTP tests; host/port/path config exists (`http_port = 8889` before `CS-4a` switch to `13306`) | Add locked `fastmcp` dependency in `CS-3a` and thin `yaw/mcp_server.py` adapter in `CS-3b`; old isolated probes do not prove project compatibility |
| Ownership | Legacy `yaw/daemon/runtime.py` | Clean-slate `yaw/runtime.py`: single `YawRuntime`, `FileLock`, live `PipelineSession`, `BoundedSemaphore(32)` (`query_sem`), `Semaphore(1)` (`lm_sem`), atomic `VectorSearch` swap |
| Containment | Legacy daemon package rules | Enforce module-specific `yaw.mcp_server` import boundary (`yaw.runtime`, `yaw.core.*` only) in `tests/core/test_architecture.py` |
| Citations | Pointer provenance (`source_hash`) + `stale_on_disk` hydration without DB chunk text | Preserve `ChunkPointer.source_hash`, strict `chunk_id: int` (`chunks` `vec0` PK), MaxP coalescing (`pool_k = max(top_k * 4, 50)`), and `stale_on_disk` verification |
| Admission | Queued worker internals in legacy daemon | Non-blocking `BoundedSemaphore(32)` admission returning `BUSY` when saturated; lock-free `daemon_status()` snapshot reads |
| Sync | Manual `request_sync()` alongside watcher | Continuous live `CocoIndex` watcher (`localfs.walk_dir(..., live=True)`) swapping `VectorSearch` outside `index_lock` on `PRAGMA data_version` change |
| Startup | Blocking/deferred initial sync | Deferred/non-blocking initial readiness on `/readyz` vs `/healthz`; Win32 named stop event (`Local\yaw-stop-<hash16>`) + POSIX signals wired via `yaw/cli.py` |
| Validation | Permissive or duplicate validation paths | Validate strict query/question (`<= 1,024` UTF-8 bytes) and strict `int` `top_k` (`1..20`, rejecting `bool`/`str`/`float`) before lifecycle checks |

Prepared bundles and persistent CocoIndex snapshots (`coco_lmdb`) are the current
ingestion contract. Scan document roots only; do not restore monolithic corpus
discovery, evaluation-field ingestion, cache resets or legacy-layout fallbacks.
Preserve generic production JSON projection and physical 1-indexed source coordinates.

## Containment and transport contract

- The new `yaw/mcp_server.py` may import ONLY `yaw.runtime`, `yaw.core.*`, and
  public transport-library/stdlib APIs (`fastmcp`, `starlette`, `uvicorn`). No
  direct ingestion, retrieval, pipeline, or `qa` imports, and no private
  cross-module access.
- Default endpoint: `http://127.0.0.1:13306/mcp` (`YawConfig.http_host`/`http_port`/`http_path`
  in `CS-4a`). Reject non-loopback binds (`INVALID_INPUT`). No separate server
  process, stdio/SSE entry points, remote bind or wildcard CORS in this epic.
- One database owner (`PipelineSession`) and `YawRuntime`; default admission
  capacity is **32 concurrent `recall` calls** (`BoundedSemaphore(32)`).
  `daemon_status()` reads lock-free without acquiring `query_sem`, `lm_sem`, or `index_lock`.
- Validate Host and Origin before dispatch, including health routes (`/mcp`,
  `/healthz`, `/readyz`), returning HTTP `403` on rejection.
- `/healthz` returns `200` `{"status": "ok"}` while HTTP is responsive.
  `/readyz` returns `200` `{"status": "ready"}` only when
  `get_status(rt).lifecycle == "ready"`, including a synchronized empty corpus;
  otherwise `503` `{"status": lifecycle}`. A bound socket alone does not prove readiness.

## Three tool contracts

Query/question inputs must be strings, trimmed, non-empty and at most 1,024 UTF-8
bytes. `top_k` is a strict integer (`type(top_k) is int and 1 <= top_k <= 20`),
default 5; booleans and coerced strings/floats are invalid (`INVALID_INPUT`).
Runtime validation remains authoritative.

| Tool | Execution and result | readOnly / destructive / idempotent / openWorld hints |
| --- | --- | --- |
| `recall(query, top_k=5)` | Delegates to `runtime.recall`; returns `{"query": str, "top_k": int, "results": list[PassageDict]}` (zero LM calls; `pool_k = max(top_k * 4, 50)` before MaxP) | true / false / true / false |
| `qa(question)` | Delegates to `runtime.ask`; returns `{"question": str, "answer": str, "termination_reason": str, "turns": int, "citations": list[PassageDict]}` (`<= 5` verified citations from tool-seen passages) | true / false / false / false |
| `daemon_status()` | Lock-free `runtime.get_status(rt).to_dict()`: `{"lifecycle", "provider_ready", "sync_state", "queue_depth", "active_op", "last_error", "last_sync_at", "pid", "num_chunks"}` with redacted errors | true / false / true / false |

Legacy `sync_sources` and `index_stats` proposals are retired under Decision 20:
live `CocoIndex` keeps `yaw.db` synchronized continuously without a separate manual
sync tool, and `daemon_status()` exposes `num_chunks` alongside lifecycle state.

`PassageDict` has `{"rank": int, "chunk_id": int, "text": str, "score": float, "stale_on_disk": bool, "content_hash": str, "source": {"rel_posix_path": str, "start_line": int, "end_line": int, "title": str}}`
with physical, 1-indexed coordinates. Preserve MaxP aggregation by canonical passage
span `(rel_posix_path, start_line, end_line)`. `content_hash` is `"sha256:"` plus
the SHA-256 of the **returned text's UTF-8 bytes**. It is distinct from
`ChunkPointer.source_hash` (persisted in `chunks` for freshness checks); SQLite
still stores no raw chunk text.

## Execution and error contracts

- Offload blocking synchronous `runtime.recall` / `runtime.ask` execution off the
  ASGI event loop (`asyncio.to_thread` or dedicated worker thread) so `/healthz`,
  `/readyz`, and `daemon_status` remain non-blocking while inference runs.
- On client disconnect or MCP cancellation, clean up request-local async tasks
  cleanly without leaking handles or corrupting active `VectorSearch` state.
- Tool failures map `YawRuntimeError` to redacted `ToolError(f"[{exc.code}] {msg}")`
  using `[CODE]` prefixes: `INVALID_INPUT`, `SHUTTING_DOWN`, `NOT_READY`, `BUSY`,
  `EMPTY_INDEX` (`qa` only; `recall` returns `results: []` on an empty ready index),
  `MODEL_ERROR`, and `CONSISTENCY_ERROR`. Redact secrets (`api_key`, `Bearer ...`,
  `sk-...`, URL userinfo) on every exposed path.
- Validate inputs FIRST, then lifecycle (`SHUTTING_DOWN` / `NOT_READY`), then
  operation-specific conditions/admission (`BUSY`, `EMPTY_INDEX`, `CONSISTENCY_ERROR`,
  `MODEL_ERROR`); an empty index cannot override a `starting`/`stopping` lifecycle error.

---

## Sprint 1 (`E2-S1`): Runtime Contracts and HTTP Lifespan

### 1.1 Dependency and containment packet

- Add FastMCP deliberately to [pyproject.toml](../pyproject.toml), resolve with
  `uv lock`, install with `uv sync --locked`, and verify the locked package on
  Python 3.14.9. Use that version's public APIs; old isolated resolution evidence
  does not prove project installation or transport behavior.
- Extend [architecture tests](../tests/core/test_architecture.py) with a
  module-specific rule for `yaw.mcp_server`: internal imports limited to
  `yaw.runtime` and `yaw.core.*`. Test allowed/rejected imports.
  No private cross-module access or relaxed domain rules.
- Exit: locked import and boundary tests pass, with versions/results recorded.

### 1.2 Source provenance and citation packet

`chunk_id` is not a source freshness checksum. Projected JSON subchunks can share
one physical record span, hydration can coalesce/expand spans, and bundle paths
can be namespaced after chunk construction. Recomputing a chunk ID from hydrated
text and its returned path therefore marks unchanged content stale.

Persist mandatory `source_hash` provenance on `ChunkPointer` and in the `chunks`
`vec0` table alongside `chunk_id: int` (`CHUNK_SCHEMA` in `yaw/pipeline.py`):
`sha256` hex of the original normalized source span/file snapshot captured at
chunk creation. All chunks from the same snapshot carry deterministic IDs and
provenance without storing raw chunk text in SQLite.

Carry the fingerprint through CocoIndex rows and `VectorSearch` snapshots. Require
fresh incompatible indexes/caches with clear diagnostics; no default-empty
provenance, migration shim, or reset of unrelated experiments.

At hydration (`hydrate_passage`), validate complete physical span bounds
(`1 <= start_line <= end_line`, span `<= 5000`) and verify the disk span hash against
`ptr.source_hash`. Bound retained buffers; never keep a resident corpus copy.
Preserve strict JSON projection, AST island expansion (`include_island=True` for
QA `read_chunk`), and bundle document/evaluation isolation.

Missing, changed or truncated sources produce `stale_on_disk=True` and empty
returned text while preserving the indexed source pointer. Do not expose changed
raw JSON as a fallback. Propagate other read/parser failures explicitly. When
coalescing passages via MaxP span deduplication (`pool_k = max(top_k * 4, 50)`),
preserve canonical span ordering without mixing incompatible source versions.

- `search_vectors` + `hydrate_ranked_passages` return `tuple[HydratedPassage, ...]`
  preserving `stale_on_disk` flags cleanly; refactor callers rather than adding
  compatibility wrappers.
- Serialize centrally in `yaw/mcp_server.py` using the `PassageDict` schema above.
- QA uses verified fresh passages and cites inline `[chunk_id=<int>]` verified
  against tool-seen IDs and `vs.id_to_pos` (`MODEL_ERROR` otherwise). If any
  cited passage has `stale_on_disk=True`, return `CONSISTENCY_ERROR`; a
  genuinely empty ready index returns `EMPTY_INDEX` (`recall` returns `results: []`).

### 1.3 Concurrency, live sync and lifecycle packet

Own bounded admission and lock hygiene inside `yaw/runtime.py`:
- `query_sem` (`BoundedSemaphore(32)`): non-blocking acquisition in `recall()`
  raising `YawRuntimeError("BUSY", ...)` immediately when 32 concurrent calls are
  active; always released in `finally`.
- `lm_sem` (`Semaphore(1)`): serializes local LM requests during `qa()`.
- `index_lock` (`threading.RLock`): protects ONLY atomic swap of `(vector_search,
  index_data_version)` in `get_vector_search(rt, conn)` after querying
  `PRAGMA data_version` on a short-lived `with contextlib.closing(open_readonly_db(...))`
  handle and rebuilding `build_vector_search(...)` outside locks — never hold
  `index_lock` across embedder or LM HTTP calls.
- Live `CocoIndex` watcher (`start_pipeline(..., live=True)`) updates `chunks` and
  `chunk_fts` continuously in the background; no manual `sync_sources` tool.

### 1.4 Guarded HTTP mount and daemon packet

Create `yaw/mcp_server.py` with `create_mcp_app` and `serve_http` wrapping
`YawRuntime`. Reject non-loopback `cfg.http_host` binds (`INVALID_INPUT`) and
install the pre-dispatch Host/Origin guard immediately on `/mcp`, `/healthz`,
and `/readyz`.

Support deferred initial readiness via `start_runtime(..., wait_for_ready=False)`
with background readiness transition so HTTP binds while initial sync is `starting`,
returning `503` on `/readyz` until `ready` (including an empty corpus). Perform
blocking startup/stop work off the ASGI event loop. Startup failure leaves
`/readyz` false and releases acquired locks/handles cleanly.

Preserve Windows named-event (`Local\yaw-stop-<hash16>` hashed from normalized
`cfg.lock_path`) and POSIX signal stop handling. On stop, set `lifecycle="stopping"`,
reject new tool work (`SHUTTING_DOWN`), join/close `close_pipeline`, set
`"stopped"`, and release `FileLock`. Wire `yaw daemon run|start|stop|status`
(`yaw/cli.py`) into this runtime and server runner.

The shared executable entrypoint also covers synchronous commands: SIGINT,
SIGTERM, and Windows SIGBREAK initiate exception unwinding; a repeated signal
forces process exit if cleanup or worker joins block. See the replacement
sprint's [CLI contract](reviews/bare-bones-replacement-sprint.md#59-yawclipy-575-loc--yaw__main__py-8-loc).

---

## Sprint 2 (`E2-S2`): MCP Tools and Transport Acceptance

### 2.1 Thin transport implementation

Register `recall`, `qa`, and `daemon_status` with strict schemas and all four
`ToolAnnotations`. Delegate exclusively to `yaw.runtime`; never import
`retrieval`/`pipeline`/`qa` or own a second database writer.

Offload `runtime.recall` / `runtime.ask` off the ASGI loop; `daemon_status`
calls lock-free `runtime.get_status(rt).to_dict()` directly. Use strict Python /
Pydantic typing so transport validation cannot coerce `true`, `"5"`, or `5.0` to
integer `top_k`. Malformed JSON-RPC remains a protocol error, not a tool failure.

### 2.2 Cancellation and disconnect cleanup

Handle client disconnects and explicit MCP request cancellation cleanly across
offloaded tool calls without corrupting `VectorSearch`, leaking readonly SQLite
handles (impossible under `contextlib.closing`), or deadlocking `query_sem`/`lm_sem`.
Test explicit cancellation and dropped connections separately against live TCP HTTP.

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
| Lifespan/containment (`E2-A1`) | Health/readiness during blocked initial sync, empty-ready, failure and stopping; bind/startup cleanup; lock reacquisition; `yaw.mcp_server` import boundary |
| Three tools (`E2-A2`) | Exact `recall`, `qa`, `daemon_status` names, schemas, and four `ToolAnnotations`; `sync_sources`/`index_stats` absent; valid `[CODE]` prefixes on errors |
| Citations (`E2-A2`) | Actual text/physical coordinates for Markdown/code and prepared JSON; shared-record subchunks, namespaced paths and MaxP coalescing (`pool_k = max(4k, 50)`); edit/delete/truncate detection (`stale_on_disk=True`, zero raw JSON leaks); `content_hash` and `qa.citations` verified against tool-seen `[chunk_id=<int>]` IDs |
| Errors (`E2-A2`) | All seven `[CODE]` errors from real validation/provider/storage/lifecycle conditions; empty/oversized (`> 1,024` UTF-8 bytes) and non-string query, `top_k` bounds/bool/string/float; lifecycle before empty-index errors; secret redaction |
| Responsive reads (`E2-A3`) | Two TCP clients; `daemon_status`, `/healthz`, and `/readyz` finish within two seconds while `qa` (`lm_sem`) remains blocked on an explicit event |
| Capacity/release (`E2-A3`) | 32 concurrent active/blocked `recall` slots via `BoundedSemaphore(32)`; 33rd concurrent `recall` returns `BUSY`; releasing waiters restores immediate admission cleanly |
| Live update (`E2-A3`) | Live CocoIndex file edit reflected via `PRAGMA data_version` swap in `VectorSearch` without server restart; new consistent citation/provenance on next `recall`/`qa` |
| Security/disconnects (`E2-A3`) | Raw duplicate headers, IPv6, missing Origin, hostile hosts and non-loopback bind rejection; separate cancellation/disconnect/reconnect evidence |

---

## Sprints and acceptance checklist

| Sprint | Work | Observable exit |
| --- | --- | --- |
| `E2-S1` | `fastmcp` dependency/containment (`CS-3a`), `source_hash` provenance, `ChunkPointer` `int` IDs, bounded `query_sem`/`lm_sem`, lock-free status, guarded HTTP mount & `yaw/cli.py` integration | Real loopback health/lifecycle tests; runtime/provenance tests; adapter containment |
| `E2-S2` | Three tools (`recall`, `qa`, `daemon_status`), error mapping, and multi-client TCP acceptance (`CS-3b`) | Real TCP schemas, citations, security, cancellation, 32-slot saturation, live update, and responsive status |

- [ ] **E2-A1:** One HTTP/runtime lifespan, CocoIndex DB owner (`PipelineSession`), and public shutdown path; health/readiness observe startup, empty-ready, failure and shutdown; dependency and containment checks pass.
- [ ] **E2-A2:** Exactly three read-only tools (`recall`, `qa`, `daemon_status`) with specified annotations/errors; source-isolated citations (`<= 5` for `qa`) and exact coordinates, hashes and freshness behavior.
- [ ] **E2-A3:** Two TCP clients share one runtime; `daemon_status`, `/healthz`, and `/readyz` respond within two seconds while `qa` inference remains blocked; 32-slot `BUSY` saturation + release; live CocoIndex file updates swap `VectorSearch` cleanly; Host/Origin `403` rejection passes.

Use workspace CocoIndex and engineering-real-tests skills during implementation.
Record actual checks/failures in [evaluation](evaluation.md). Loopback inference
proves orchestration, not model quality or parity. Update README's available
commands only after the HTTP implementation passes.

## Deferred Epic 3 ideas — outside this epic

Retain these proposals for a separate decision in [Epic 3](epic-3-performance.md): reorganizing CLI domains around
optimization/evaluation, numerical traversal tuning, grouping traversal/policy/
seeding settings, and editable seed instructions/demonstrations. They are neither
Epic 2 prerequisites nor accepted replacements for current interfaces.

---

## Decisions

- 2026-11-15 `[Superseded on 2026-11-16 by Epic 1 Decision 20]`: User authorized the `4`-layer bare-bones architecture uniting FastMCP HTTP, CLI, constant live CocoIndex indexing, zero-LLM `SubStateManifoldHMM` `recall`, and `dspy.RLM` `qa` in the [Bare-Bones YAW Replacement Sprint](reviews/bare-bones-replacement-sprint.md) ([Epic 1 Decisions 18–19](epic-1-hipporag-daemon.md#decisions)): *"Retain & refine, but remove the ppr.py default. replace with the graphless graph. Pointless to keep ppr.py if we can beat it with less compute. So the mission was a success after all! This means you’ll need to map out a proper replacement sprint where all the bells and whistles of current yaw internals are reduced to utter bare bones. 1. cocoindex owns indexing, constant live daemon mode. 2. DSPy RLM sits on top of that pipeline. QA is routed to the RLM which uses tools to explore the ”graph”. 3. Recall@N passes through verbatim without the RLM 4. FastMCP and CLI sit on top of this, exposing the ”recall” and ”qa” tools (2. and 3.) Your goal has been met. We’ll explore the fancier math in Epic 3."* Open scope detail (`Open 1`: strictly `2` MCP tools `recall` + `qa` vs retaining `daemon_status` + `sync_sources` as transport health/sync controls alongside `/healthz` and `/readyz`) is tracked in [bare-bones-replacement-sprint.md](reviews/bare-bones-replacement-sprint.md#1-scope-non-goals--architecture-overview) pending user sign-off before execution.

- 2026-11-16: Following the user's clean-slate zero-backward-compatibility directive ([Epic 1 Decision 20](epic-1-hipporag-daemon.md#decisions), [bare-bones-replacement-sprint.md](reviews/bare-bones-replacement-sprint.md)), update the FastMCP HTTP and CLI contract:
  - Clean-slate `yaw/mcp_server.py` (`FastMCP("yaw")`, Streamable HTTP on `127.0.0.1:13306/mcp` + `Host`/`Origin` `403` + `/healthz`/`/readyz`) exposes **exact `3` read-only tools**: `recall(query: str, top_k: int = 5)`, `qa(question: str)`, and `daemon_status()` (dropping `sync_sources` and `index_stats`), delegating to `VectorSearch` (`yaw/retrieval.py`) and bare `dspy.ReAct` (`yaw/qa.py`).
  - `chunk_id` in cited passages is `int` (from `chunks` `vec0` `INTEGER primary key` matching physical `(rel_path, start_line, end_line)` + `source_hash` / `stale_on_disk`).
  - CLI `yaw index` / `yaw graph` are deleted: `yaw daemon run|start` brings FastMCP HTTP and live CocoIndex online together.
