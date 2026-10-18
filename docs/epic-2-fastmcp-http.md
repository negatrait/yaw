# Epic 2: FastMCP Streamable HTTP Adapter (`yaw/daemon/mcp_server.py`)

Owner: [FOUNDATIONAL](../FOUNDATIONAL.md) | Prerequisite: [Epic 1](epic-1-hipporag-daemon.md) (`D1-A0..D1-A6` passed) | Results: [evaluation](evaluation.md)

## Containment & Transport Contract

- **Containment Law:** `yaw/daemon/mcp_server.py` resides in `yaw/daemon/` and imports **only** `yaw.daemon.runtime`, `yaw.core.*`, and `fastmcp`. It never imports `ingestion/`, `graph/`, `storage/`, or `inference/` directly (`test_architecture.py` enforced).
- Mount `FastMCP` onto the E1 `Runtime` lifespan at `http://127.0.0.1:13306/mcp` (`mcp.run(transport="http", host="127.0.0.1", port=13306, path="/mcp")`).
- One process, one `yaw.db` owner, one worker queue. No separate `serve-mcp` process, no stdio/SSE entry points, no wildcard CORS.
- Health routes: `/healthz` = 200 liveness; `/readyz` = 200 when committed index is ready/empty, 503 when starting/stopping/failed.

## Tools (5 Fixed Endpoints Delegating to `Runtime`)

All query inputs trimmed, non-empty, $\le 1024$ UTF-8 bytes.

| Tool | Args | Action & Return (`yaw/daemon/runtime.py` queue) |
| --- | --- | --- |
| `retrieve` | `query: str`, `top_k: int = 5` (1–20) | `readOnlyHint=true`. Queue `runtime.retrieve(query, top_k)`. Return `{rank, text, score, chunk_id, content_hash, stale_on_disk, source: {rel_posix_path, start_line, end_line, title}}`. |
| `rag_qa` | `question: str` | `readOnlyHint=true`. Queue `runtime.rag_qa(question)`. Return `{answer, evidence: passages[:5], last_sync_at}`. |
| `index_stats` | none | `readOnlyHint=true`. Read cached `runtime.stats()` (`chunks`, `entities`, `facts`, `edges`, `entity_labels`, `shacl_conforms`, config fingerprint) without blocking on worker thread. |
| `daemon_status` | none | `readOnlyHint=true`. Read cached `runtime.status()` (lifecycle, `provider_ready`, CocoIndex sync state, queue depth, active op, redacted last error). |
| `sync_sources` | none | `destructiveHint=true`. Call `runtime.request_sync()`; return `{scan_id, status: "accepted" \| "already_pending"}`. |

## Execution & Error Rules

- Never block the `asyncio` loop; await `Runtime` queue futures.
- Emit real `Context.report_progress` stages (`queued`, `started`, `ready`).
- Cancel before execution removes the queued item; once running on the worker thread, finish the operation and discard the disconnected client's result.
- Return `isError=true` with fixed codes: `INVALID_INPUT`, `NOT_READY`, `EMPTY_INDEX` (QA only), `BUSY`, `MODEL_ERROR`, `CONSISTENCY_ERROR`, `SHUTTING_DOWN`.

## Sprints & Acceptance Checks

- **Copy/Adapt From:** `git show d1bc65e:src/yaw/server/daemon.py:90-160` (`FastMCP("yaw-mcp")` initialization, `@mcp.tool()` registration, `mcp.run(transport="http", host=..., port=...)`).
- **Drop (Anti-Spaghetti):** Drop `d1bc65e:daemon.py`'s ontology tools (`compile_wiki`, `lint_wiki`, `load_ontology`, `set_extraction_mode`, `get_ontology_schema`), `stdio` transport branches, and direct `yaw.retrieval`/`yaw.storage` imports.

| Sprint | Target Node & Files | Exit Condition |
| --- | --- | --- |
| **Sprint 1 (`E2-S1`)** | `pyproject.toml`, `uv.lock`, `yaw/daemon/mcp_server.py`, `yaw/daemon/cli.py` | Locked `fastmcp` installed; `/mcp`, `/healthz`, `/readyz` bind to `Runtime` lifespan; `test_architecture.py` passes (`0` cross-domain imports) |
| **Sprint 2 (`E2-S2`)** | `yaw/daemon/mcp_server.py`, `tests/daemon/test_mcp_http.py`, `README.md`, `docs/evaluation.md` | All 5 tools, progress notifications, cancellation, `BUSY` queue overflow, and Host/Origin checks pass over real loopback TCP |

- [ ] **E2-A1 (`E2-S1`):** FastMCP server mounts onto `Runtime` lifespan with zero direct `graph`/`storage`/`inference` imports; `/healthz` and `/readyz` reflect live `Runtime` state.
- [ ] **E2-A2 (`E2-S2`):** FastMCP client connects over TCP to `127.0.0.1:13306/mcp`, lists 5 tools, receives progress events, and retrieves 1-indexed line-hydrated passages + QA answers.
- [ ] **E2-A3 (`E2-S2`):** Two concurrent TCP clients share one `yaw.db` owner; `daemon_status` and `/healthz` respond in `<2s` during slow inference; 32-slot overflow returns `BUSY`; live CocoIndex edits update citations atomically.
