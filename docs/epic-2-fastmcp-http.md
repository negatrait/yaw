# Epic 2: FastMCP Streamable HTTP Adapter (`yaw/mcp_server.py`, `yaw/runtime.py`, `yaw/cli.py`)

Owner: [FOUNDATIONAL](../FOUNDATIONAL.md) | Prerequisite: [Epic 1](epic-1-hipporag-daemon.md) | Next: [Epic 3](epic-3-performance.md) | Results: [evaluation](evaluation.md)

| Relation | Links |
| --- | --- |
| Governed by | [FOUNDATIONAL.md](../FOUNDATIONAL.md), [AGENTS.md](../AGENTS.md) |
| Rules | [runtime.md](../.agents/rules/runtime.md): `yaw/runtime.py`, `yaw/mcp_server.py`, `yaw/cli.py`; [tests.md](../.agents/rules/tests.md): `tests/**/*.py`; [docs.md](../.agents/rules/docs.md): `docs/**/*.md` |
| Owns code | `yaw/mcp_server.py`, `yaw/runtime.py`, `yaw/cli.py` |
| Peers | [epic-1-hipporag-daemon.md](epic-1-hipporag-daemon.md): owns `yaw/core/` (`YawConfig`, `DaemonStatus`, `configure_logging`, `redact_secrets`) & `yaw/__main__.py` entrypoint; [native-cocoindex-indexing.md](native-cocoindex-indexing.md): owns `yaw/pipeline.py` & `yaw/datasets.py` (`PipelineSession`, `start_pipeline`, `close_pipeline`, `make_dspy_embed_fn`); [native-cocoindex-graph.md](native-cocoindex-graph.md): owns `yaw/retrieval.py` (`RetrievalIndex`, `open_readonly_db`, `build_retrieval_index`, `search_index`, `hydrate_ranked_passages`); [dspy-improvements.md](dspy-improvements.md): owns `yaw/qa.py` & `yaw/optimize.py` (`ask_question`, `build_qa_agent`, `create_lm`, `optimize_dspy`); [epic-3-performance.md](epic-3-performance.md): owns `scripts/{watchdog,preflight}.py` & tracks deferred `EVAL-6` / `E3-P5` / `dspy.RLM` proposals; [coverage-plan.md](coverage-plan.md): owns `tests/**/*.py` (`tests/test_mcp_server.py`, `tests/test_runtime.py`, `tests/test_cli.py`) |
| References | [bare-bones-replacement-sprint.md](reviews/bare-bones-replacement-sprint.md), [mdl-dag-sprint.md](reviews/mdl-dag-sprint.md) |
| Records | [evaluation.md](evaluation.md), [evaluation-ingestion.md](evaluation-ingestion.md) |

## Status & Implemented Architecture (`CS-3`..`CS-6` & `EVAL-5`, `5c4b115`)

`yaw/mcp_server.py`, `yaw/runtime.py`, and `yaw/cli.py` implement the clean-slate FastMCP Streamable HTTP transport and `YawRuntime` lifecycle over live CocoIndex `PipelineSession`, frozen `M_FINAL_1B` `RetrievalIndex`, and bare `dspy.ReAct` QA.

| Area | Implemented Contract |
| --- | --- |
| Transport (`yaw/mcp_server.py`) | `FastMCP("yaw", strict_input_validation=True)` Streamable HTTP on `127.0.0.1:13306/mcp` (`uvicorn` + `_LoopbackGuardMiddleware`), `/healthz`, `/readyz`, 3 read-only tools (`recall`, `qa`, `daemon_status`) |
| Runtime Ownership (`yaw/runtime.py`) | Plain `@dataclass YawRuntime`, exclusive `FileLock(cfg.lock_path)`, live `PipelineSession`, `BoundedSemaphore(32)` (`query_sem`), `Semaphore(1)` (`lm_sem`), atomic `RetrievalIndex` swap outside `index_lock` |
| Containment (`tests/core/test_architecture.py`) | `yaw.mcp_server` imports `fastmcp`, `pydantic`, `starlette`, `uvicorn`, `yaw.core`, `yaw.runtime` only (`0` direct `pipeline`/`retrieval`/`qa` imports) |
| Provenance & MaxP Citations | 63-bit integer `chunk_id` in plain SQLite `chunks` (`embedding_f16 <f2>`, zero raw chunk text), `pool_k = max(top_k * 4, 50)` before `hydrate_ranked_passages`, SHA-256 `source_hash` + `stale_on_disk` verification |
| Concurrency & Admission | Non-blocking `BoundedSemaphore(32)` returns `BUSY` when saturated; `get_status(rt)` reads under short `state_lock` without waiting on `query_sem`, `lm_sem`, or `index_lock` |
| Live WAL Hot-Swapping | `get_retrieval_index(rt, conn)` checks `PRAGMA data_version` on `probe_conn` (`probe_lock`), rebuilds `build_retrieval_index` over `?mode=ro` WAL outside `index_lock`, and preserves prior snapshot if `INDEX_NOT_READY` occurs mid-sync |
| Startup & Signals (`yaw/cli.py`) | Deferred initial readiness via `start_runtime(..., wait_for_ready=False)` (`/readyz` returns `503` until `lifecycle == "ready"`); Win32 stop event (`Local\yaw-stop-<hash16>`) + POSIX signals with double-signal `os._exit` escape hatch |
| Strict Input Validation | Query/question must be non-empty stripped string `<= 1,024` UTF-8 bytes; `top_k` strict `int` (`type(top_k) is int and 1 <= top_k <= 20`, rejecting `bool`/`str`/`float`) before lifecycle checks |

## Containment and transport contract

- `yaw/mcp_server.py` imports only `yaw.runtime`, `yaw.core.*`, and public transport-library/stdlib APIs (`fastmcp`, `pydantic`, `starlette`, `uvicorn`). Zero direct `pipeline`, `retrieval`, or `qa` imports, and zero private cross-module access.
- Default endpoint: `http://127.0.0.1:13306/mcp` (`YawConfig.http_host`/`http_port`/`http_path`). Non-loopback `http_host` binds raise `YawRuntimeError("INVALID_INPUT", ...)`.
- `_LoopbackGuardMiddleware` validates `Host` (exactly 1 recognized loopback authority: `localhost`, `127.0.0.1`, or loopback IPv6 `[::1]`) and `Origin` (absent or 1 well-formed HTTP(S) loopback origin) before dispatch across `/mcp`, `/healthz`, and `/readyz`, returning HTTP `403` on rejection.
- `/healthz` returns `200` `{"status": "ok"}`. `/readyz` returns `200` `{"status": "ready"}` only when `get_status(rt).lifecycle == "ready"` and `provider_ready` is true; otherwise `503` `{"status": lifecycle}`.

## Three tool contracts

| Tool | Execution and result | readOnly / destructive / idempotent / openWorld hints |
| --- | --- | --- |
| `recall(query, top_k=5)` | Offloaded via executor to `runtime.recall`; returns `{"query": str, "top_k": int, "results": list[PassageDict]}` (`0` LM calls; `pool_k = max(top_k * 4, 50)` before MaxP span dedup) | true / false / true / false |
| `qa(question)` | Offloaded via executor to `runtime.ask`; returns `{"question": str, "answer": str, "turns": int, "citations": list[PassageDict]}` (`<= 5` verified citations from tool-seen passages) | true / false / false / false |
| `daemon_status()` | Offloaded lock-free `runtime.get_status(rt).to_dict()`: `{"lifecycle", "provider_ready", "sync_state", "queue_depth", "active_op", "last_error", "last_sync_at", "pid", "num_chunks"}` with secrets redacted | true / false / true / false |

`PassageDict` (`serialize_passage` in `yaw/runtime.py`) emits `{"rank": int, "chunk_id": int, "text": str, "score": float, "stale_on_disk": bool, "content_hash": str, "source": {"rel_posix_path": str, "start_line": int, "end_line": int, "title": str}}` with 1-indexed coordinates. `content_hash` is `"sha256:"` plus the SHA-256 of the returned UTF-8 text (distinct from `ChunkPointer.source_hash` stored in `chunks`); SQLite stores zero raw chunk text.

## Execution and error contracts

- Blocking `runtime.recall` and `runtime.ask` execute on a dedicated `ThreadPoolExecutor` (`_run_offloaded`) off the ASGI event loop so `/healthz`, `/readyz`, and `daemon_status` remain responsive while `qa` (`lm_sem`) blocks.
- `YawRuntimeError` maps to redacted `ToolError(f"[{exc.code}] {msg}")` over the 8 live error codes (`INVALID_INPUT`, `SHUTTING_DOWN`, `NOT_READY`, `BUSY`, `EMPTY_INDEX`, `MODEL_ERROR`, `CONSISTENCY_ERROR`; `INDEX_NOT_READY` on missing/stale artifacts).
- Validation ordering: inputs (`INVALID_INPUT`) -> lifecycle (`SHUTTING_DOWN` / `NOT_READY`) -> operation state (`BUSY`, `EMPTY_INDEX`, `CONSISTENCY_ERROR`, `MODEL_ERROR`). `recall` returns `results: []` on an empty ready index; `qa` raises `EMPTY_INDEX`.

## Sprint Verification & Checklists

- [x] **E2-A1 (Lifespan, Readiness & Containment):** `yaw/mcp_server.py` import boundary enforced by `tests/core/test_architecture.py`; `/healthz`, `/readyz`, startup/stop lifecycle, `FileLock`, and Win32/POSIX stop handling verified in `tests/test_mcp_server.py` and `tests/test_runtime.py`.
- [x] **E2-A2 (Three Tools, Citations & Error Mapping):** `recall`, `qa`, `daemon_status`, annotations, `PassageDict`, `stale_on_disk` detection, strict `[chunk_id=<int>]` grounding, all error codes, and secret redaction verified in `tests/test_mcp_server.py` and `tests/test_cli.py`.
- [x] **E2-A3 (TCP Concurrency, 32-Slot `BUSY`, Live Updates & Guard):** Loopback TCP multi-client responsiveness, 32-slot saturation + release, live CocoIndex `PRAGMA data_version` hot-swap to `RetrievalIndex`, and `Host`/`Origin` `403` guard verified in `tests/test_mcp_server.py` and `tests/test_runtime.py`.

## Deferred Epic 3 ideas — outside this epic

Retain these proposals for a separate decision in [Epic 3](epic-3-performance.md): `EVAL-6` (`SearchResult.completeness` calibration and lazy `.yaw/wiki/` Graph Scout), isolated backend scaling ablations (`E3-P5`), and `dspy.RLM` for `>100K`-token raw documents. They are neither Epic 2 prerequisites nor authorized changes to active `yaw/mcp_server.py` or `yaw/runtime.py` interfaces.

---

## Decisions

- 2026-11-15 `[Superseded on 2026-11-16 by Epic 1 Decision 20]`: User authorized the `4`-layer bare-bones architecture uniting FastMCP HTTP, CLI, constant live CocoIndex indexing, zero-LLM `SubStateManifoldHMM` `recall`, and `dspy.RLM` `qa` in the [Bare-Bones YAW Replacement Sprint](reviews/bare-bones-replacement-sprint.md) ([Epic 1 Decisions 18–19](epic-1-hipporag-daemon.md#decisions)): *"Retain & refine, but remove the ppr.py default. replace with the graphless graph. Pointless to keep ppr.py if we can beat it with less compute. So the mission was a success after all! This means you’ll need to map out a proper replacement sprint where all the bells and whistles of current yaw internals are reduced to utter bare bones. 1. cocoindex owns indexing, constant live daemon mode. 2. DSPy RLM sits on top of that pipeline. QA is routed to the RLM which uses tools to explore the ”graph”. 3. Recall@N passes through verbatim without the RLM 4. FastMCP and CLI sit on top of this, exposing the ”recall” and ”qa” tools (2. and 3.) Your goal has been met. We’ll explore the fancier math in Epic 3."* Open scope detail (`Open 1`: strictly `2` MCP tools `recall` + `qa` vs retaining `daemon_status` + `sync_sources` as transport health/sync controls alongside `/healthz` and `/readyz`) is tracked in [bare-bones-replacement-sprint.md](reviews/bare-bones-replacement-sprint.md#1-scope-non-goals--architecture-overview) pending user sign-off before execution.

- 2026-11-16: Following the user's clean-slate zero-backward-compatibility directive ([Epic 1 Decision 20](epic-1-hipporag-daemon.md#decisions), [bare-bones-replacement-sprint.md](reviews/bare-bones-replacement-sprint.md)), update the FastMCP HTTP and CLI contract:
  - Clean-slate `yaw/mcp_server.py` (`FastMCP("yaw")`, Streamable HTTP on `127.0.0.1:13306/mcp` + `Host`/`Origin` `403` + `/healthz`/`/readyz`) exposes **exact `3` read-only tools**: `recall(query: str, top_k: int = 5)`, `qa(question: str)`, and `daemon_status()` (dropping `sync_sources` and `index_stats`), delegating to `VectorSearch` (`yaw/retrieval.py`) and bare `dspy.ReAct` (`yaw/qa.py`).
  - `chunk_id` in cited passages is `int` (from `chunks` `vec0` `INTEGER primary key` matching physical `(rel_path, start_line, end_line)` + `source_hash` / `stale_on_disk`).
  - CLI `yaw index` / `yaw graph` are deleted: `yaw daemon run|start` brings FastMCP HTTP and live CocoIndex online together.
