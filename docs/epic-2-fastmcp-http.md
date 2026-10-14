# Epic 2: HippoRAG 2 through FastMCP v4 and Streamable HTTP

Status: planned; implementation has not started. Model compatibility follows Epic 1: independent NPU chat/GPU embedding selection, explicit validated profiles, no model fallback. Depends on every acceptance gate in [Epic 1](epic-1-hipporag-daemon.md). Scope owner: [FOUNDATIONAL](../FOUNDATIONAL.md). Results owner: [evaluation](evaluation.md).

## Outcome

An MCP client connects to `http://127.0.0.1:13306/mcp`, retrieves source passages and asks grounded questions against the same index maintained by the daemon. Lemonade remains the inference server at port 13305. Client connections never create another index owner.

Use standalone `fastmcp==4.1.0` and `from fastmcp import FastMCP`, not the similarly named MCP SDK class. Version 4.1.0 was the checked v4 release on 2026-10-13; an isolated uv resolution with pinned HippoRAG, NumPy and filelock succeeded and selected MCP SDK 2.3.0. This does not yet prove import or HTTP integration. Add FastMCP to the real manifest/lock only during this epic. [Release](https://github.com/PrefectHQ/fastmcp/releases/tag/v4.1.0)

The pinned HippoRAG already ships `hipporag.mcp_server`, backed by `mcp.server.mcpserver.MCPServer`, with retrieve, QA, index, delete and stats tools and optional Streamable HTTP. Treat it as a reference for upstream data shapes and serialized execution. YAW needs FastMCP v4 plus the E1 ownership/reconciliation contract, so it supplies one thin adapter. Do not run, proxy, import private helpers from, or copy the upstream service implementation alongside it. No fork is required.

## One runtime and one transport

Create `yaw/mcp_server.py` with a server factory accepting the E1 Runtime/configuration. Its application lifespan enters/exits E1's runtime context once for the whole server. All tools share it. E2 changes the existing `run/start` host to serve FastMCP; there is no separate `serve-mcp` daemon or per-client instance.

Use the documented direct host `mcp.run(transport="http", host="127.0.0.1", port=13306, path="/mcp")`: FastMCP's HTTP transport is Streamable HTTP. Keep one server worker and disable auto-reload. Set streaming responses (`json_response=False`) explicitly and validate that setting against the pinned API in E2-01. Do not expose legacy SSE or stdio as alternate YAW entry points. [HTTP deployment](https://gofastmcp.com/deployment/http)

Configuration adds exactly `mcp.host = "127.0.0.1"`, `mcp.port = 13306`, `mcp.path = "/mcp"`; reject non-loopback binding in this scope. Enable FastMCP Host/Origin protection, permit only the configured loopback authority, and reject unexpected supplied origins. Clients without an Origin header must still work. Do not configure wildcard CORS. Authentication for access from other machines is a later scope decision, not a disabled production feature.

A port conflict exits with its cause and releases the runtime/lock; never pick a different port silently. The HTTP endpoint becomes discoverable after runtime initialization; querying while the first scan is pending reports current state. `/healthz` reports HTTP process liveness, while `/readyz` returns 503 for starting, stopping, failed or needs-rebuild states. Ready or empty committed indexes return 200; a clean prior generation with a scan-read error remains queryable and reports its stale scan timestamp. Health endpoints contain no document text, credentials or absolute source paths.

Server setup and teardown must run once regardless of client count. FastMCP supports application lifespan setup/cleanup and access to shared context; use that public surface. [Lifespans](https://gofastmcp.com/servers/lifespan)

## Tool contract

Advertise these five tools, with typed input/output schemas and descriptive tool annotations. Validate strings after trimming and reject empty queries. Limit questions/queries to 1024 UTF-8 bytes under the E1 input-size contract; reject oversize inputs explicitly. There are no caller-selected file paths, model IDs, backend types or upstream configuration overrides.

| Tool | Input | Behavior and output |
| --- | --- | --- |
| `retrieve` | `query: str`, `top_k: int = 5`, strict range 1–20 | Queue `HippoRAG.retrieve([query], num_to_retrieve=top_k)`. Return `query`, committed generation/scan IDs, last successful scan time and ordered passages. No answer generation. |
| `rag_qa` | `question: str` | Queue `HippoRAG.rag_qa([question])`. Return answer and the actual first `qa_top_k=5` passages supplied to upstream QA, plus the same generation/freshness fields. Do not label every retrieved candidate as evidence used in the answer. |
| `index_stats` | none | Return cached passage/entity/fact/graph counts and model/configuration identities from the last completed worker operation. Include the snapshot time; never inspect mutable HippoRAG objects on the HTTP thread. |
| `daemon_status` | none | Return E1 lifecycle, index state, scan state, queue depth, active phase and redacted last error. Stays responsive while indexing or answering. |
| `sync_sources` | none | Request/coalesce a full scan of the configured root and return `scan_id` with `accepted` or `already_pending`. Completion is observed with `daemon_status`; an acceptance receipt is not an indexing-success claim. |

Each passage has `rank`, `text`, finite `score`, `passage_sha256` and `sources[]`. Each source has relative `path`, original file hash, normalized text hash and start/end offsets. Derive citations from the committed E1 manifest so duplicate text can cite more than one source. Scores describe relevance within that response, never probability or factual confidence. The QA result's evidence list does not imply support for every generated claim.

Use `readOnlyHint=true` for retrieve, QA, stats and status (they do not change the corpus); mark their use of inference/cache in descriptions. `sync_sources` is mutating with `destructiveHint=true`: reconciliation can remove derived passages when files were deleted. Annotations do not authorize execution. No arbitrary `index`, `delete`, `index_path`, `remove_path` or OS stop tool is exposed: filesystem reconciliation is the sole corpus writer, and process control remains local CLI. This prevents a deleted MCP passage from being silently recreated by the next scan.

## Streaming, cancellation and errors

Streamable HTTP describes the MCP transport. HippoRAG's current synchronous `rag_qa` returns a completed answer, so this epic does not promise incremental answer tokens. Report real observable stages with `Context.report_progress`: queued, retrieval/QA started, result ready. Do not invent extraction percentages or token progress. Clients requesting progress must receive at least the queue/start update before a delayed tool completes; clients without progress support receive the same final result. [Progress API](https://gofastmcp.com/servers/progress)

The event loop never calls blocking HippoRAG methods. Await the E1 worker operation. A client cancellation before execution removes the queued operation. Once execution starts, the worker retains ownership until it finishes even if the client disconnects; release neither the serialization slot nor the HippoRAG instance early. Discard a disconnected caller's result. Status and unrelated control requests remain usable. Accepted scans continue as daemon work and are not cancelled by an MCP disconnect.

Use tool error results (`isError=true`) with stable codes and a concise message for `INVALID_INPUT`, `NOT_READY`, `EMPTY_INDEX` (QA only), `BUSY`, `NEEDS_REBUILD`, `MODEL_ERROR`, `CONSISTENCY_ERROR`, and `SHUTTING_DOWN`. Empty retrieval returns `passages=[]`. A model/transport failure never becomes an empty successful answer, backend change or automatic retry through another model. Preserve upstream exception details in local redacted logs.

Internal HippoRAG dense-retrieval branches remain upstream behavior. Report diagnostic evidence when available, and use `not_reported` where the public response does not establish a branch. Do not infer a successful graph traversal merely from a graph existing on disk, or add an independent dense fallback in YAW.

Keep foreground tools as normal MCP calls; do not add the optional FastMCP task extension, a broker, REST query endpoints, websockets or token-streaming protocol in these epics. `sync_sources` is explicitly a scan request rather than a long-running MCP task. Long query waits retain queue/phase observability; measure real latency before promising a service-level target.

## Delivery packets

All items are planned, not completed. Keep the E1 runtime as the sole implementation of operations.

| Packet | Owned files (future) | Exit condition |
| --- | --- | --- |
| E2-01 | `pyproject.toml`, `uv.lock`, `yaw/mcp_server.py` | Install locked v4 dependency; import/server construction and pinned transport settings work on Windows; no optional upstream server is launched |
| E2-02 | `yaw/mcp_server.py`, runtime host wiring | Five typed tool contracts, correct metadata and shared lifespan; one instance for two concurrent clients |
| E2-03 | HTTP/progress/error adapters | Blocking work stays off the event loop; cancellation and busy behavior preserve serialization; real progress messages cross HTTP |
| E2-04 | `tests/test_mcp_http.py`, README | A separate native client process connects over TCP to `/mcp`; discovery and all tool/error scenarios pass |
| E2-05 | `docs/evaluation.md` | Real Lemonade evidence and unchanged E1 lifecycle/reconciliation gates recorded; endpoint instructions work from a second terminal |

## Acceptance gates

Tests that directly call Python tool functions do not satisfy transport acceptance. Use the FastMCP client against the real HTTP socket, plus bounded failure-injection fixtures. No GUI-only evidence or unrecorded manual claims.

- [ ] **E2-A1:** List exactly five tools with documented schemas, strict input validation and annotations. Connect using the installed v4 client; record negotiated protocol version and verify Streamable HTTP progress over the actual connection. Do not hard-code an obsolete handshake sequence.
- [ ] **E2-A2:** Two clients retrieve/ask against one runtime/index owner. A deliberately slow model fixture leaves daemon_status and health responsive within two seconds. Queue exhaustion returns BUSY; cancellation cannot overlap two HippoRAG calls.
- [ ] **E2-A3:** Repeat the nine-document fixture over HTTP. George's evidence is first; both birthplace passages are inside QA's actual evidence window. Answers and evidence meet D1-01. Record ranking, selected facts/graph evidence when available, model IDs and timing. Real model outcomes supplement deterministic tests.
- [ ] **E2-A4:** Add/edit/rename/delete source fixtures while clients remain connected. Requests completed before reconciliation see the previous complete generation; requests queued during reconciliation execute only after the commit and see the new complete generation. They never receive mixed manifest/index citations. Duplicate owners and empty-index behavior match E1.
- [ ] **E2-A5:** Stop/start the daemon and reconnect. Existing sources cause no unnecessary extraction, instance ID changes, committed generation persists, and no stale tool call is replayed as a new mutation. Source/model errors produce defined errors; health and readiness differ correctly.
- [ ] **E2-A6:** Reject invalid top_k values, empty/oversize queries, wrong Host/Origin and remote bind configuration. A busy port never causes an unnoticed port change. Source filenames and secrets do not leak through health or routine error responses.
- [ ] **E2-A7:** `sync_sources` returns an accepted scan ID; status shows its eventual completion or failure. Repeated requests coalesce; the tool cannot ingest arbitrary filesystem paths or delete source files.

Close only after all boxes have command/result evidence in the evaluation record. No Milvus migration or semantic layer is part of completion.
