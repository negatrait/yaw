"""Thin FastMCP Streamable HTTP adapter with loopback Host/Origin 403 guard and 3 read-only tools."""

from __future__ import annotations

import asyncio
import functools
import ipaddress
import os
import threading
from collections.abc import AsyncGenerator, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Annotated, Any

import uvicorn
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field
from starlette.middleware import Middleware as StarletteMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from yaw.core import DaemonStatus, YawConfig, YawRuntimeError
from yaw.runtime import (
    YawRuntime,
    close_runtime,
    get_status,
    redact_secrets,
    serialize_passage,
    start_runtime,
)
from yaw.runtime import (
    ask as runtime_ask,
)
from yaw.runtime import (
    recall as runtime_recall,
)


def _validate_loopback_bind(host: str) -> None:
    if not isinstance(host, str) or not host or host != host.strip():
        raise YawRuntimeError("INVALID_INPUT", f"Non-loopback http_host {host!r} is forbidden.")
    stripped = host[1:-1] if host.startswith("[") and host.endswith("]") else host
    if stripped.lower() == "localhost":
        return
    try:
        if ipaddress.ip_address(stripped).is_loopback:
            return
    except ValueError:
        pass
    raise YawRuntimeError("INVALID_INPUT", f"Non-loopback http_host {host!r} is forbidden.")


def _valid_port(port_text: str) -> bool:
    return port_text.isascii() and port_text.isdigit() and 1 <= int(port_text) <= 65535


def _is_valid_loopback_authority(raw: str) -> bool:
    if not raw or not raw.isascii() or any(ch.isspace() or ch in "@/?#\\," for ch in raw):
        return False
    if raw.startswith("["):
        end = raw.find("]")
        if end < 0:
            return False
        host_part, rest = raw[1:end], raw[end + 1 :]
        if rest and (not rest.startswith(":") or not _valid_port(rest[1:])):
            return False
        try:
            return ipaddress.IPv6Address(host_part).is_loopback
        except ValueError:
            return False
    if raw.count(":") > 1:
        return False
    host_part = raw
    if ":" in raw:
        host_part, port_s = raw.split(":", 1)
        if not _valid_port(port_s):
            return False
    if host_part.lower() == "localhost":
        return True
    try:
        return ipaddress.IPv4Address(host_part).is_loopback
    except ValueError:
        return False


def _is_valid_loopback_origin(raw: str) -> bool:
    if not raw or raw.lower() == "null":
        return False
    low = raw.lower()
    if low.startswith("http://"):
        return _is_valid_loopback_authority(raw[7:])
    if low.startswith("https://"):
        return _is_valid_loopback_authority(raw[8:])
    return False


class _LoopbackGuardMiddleware:
    """Pre-dispatch ASGI middleware enforcing single loopback Host and optional loopback Origin."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") == "http":
            hosts: list[str] = []
            origins: list[str] = []
            for raw_name, raw_val in scope.get("headers", ()):
                name = raw_name.decode("latin1").lower()
                if name == "host":
                    hosts.append(raw_val.decode("latin1"))
                elif name == "origin":
                    origins.append(raw_val.decode("latin1"))
            host_ok = len(hosts) == 1 and _is_valid_loopback_authority(hosts[0])
            origin_ok = not origins or (len(origins) == 1 and _is_valid_loopback_origin(origins[0]))
            if not (host_ok and origin_ok):
                resp = JSONResponse({"error": "Forbidden"}, status_code=403)
                await resp(scope, receive, send)
                return
        await self.app(scope, receive, send)


def _to_tool_error(exc: YawRuntimeError, api_key: str) -> ToolError:
    return ToolError(redact_secrets(str(exc), api_key))


def create_mcp_app(
    rt: YawRuntime | None = None,
    *,
    runtime_factory: Callable[[], YawRuntime] | None = None,
    http_path: str | None = None,
) -> tuple[FastMCP, Any]:
    """Create the FastMCP server and pre-dispatch-guarded Starlette Streamable HTTP ASGI app."""
    if (rt is None) == (runtime_factory is None):
        raise YawRuntimeError("INVALID_INPUT", "Provide exactly one of rt or runtime_factory.")
    mount_path = http_path if http_path is not None else (rt.config.http_path if rt is not None else "/mcp")
    if not isinstance(mount_path, str) or not mount_path.startswith("/"):
        raise YawRuntimeError("INVALID_INPUT", f"http_path must start with '/', got {mount_path!r}.")
    if rt is not None:
        _validate_loopback_bind(rt.config.http_host)

    holder: dict[str, YawRuntime | None] = {"rt": rt}
    exec_box: list[ThreadPoolExecutor | None] = [None]
    active_sessions = [0]

    def _ensure_executor(active_rt: YawRuntime | None) -> ThreadPoolExecutor | None:
        if exec_box[0] is None:
            cap = active_rt.config.queue_maxsize if active_rt is not None else 32
            exec_box[0] = ThreadPoolExecutor(max_workers=max(cap + 8, 40), thread_name_prefix="yaw-mcp-worker")
        return exec_box[0]

    @asynccontextmanager
    async def _lifespan(_server: FastMCP) -> AsyncGenerator[None]:
        owns_rt = runtime_factory is not None
        active_sessions[0] += 1
        if owns_rt and holder["rt"] is None:
            built = await asyncio.to_thread(runtime_factory)
            try:
                _validate_loopback_bind(built.config.http_host)
            except BaseException:
                await asyncio.to_thread(close_runtime, built)
                active_sessions[0] -= 1
                raise
            holder["rt"] = built
        _ensure_executor(holder["rt"])
        try:
            yield
        finally:
            active_sessions[0] -= 1
            if active_sessions[0] == 0:
                active = holder["rt"]
                if owns_rt:
                    holder["rt"] = None
                pool = exec_box[0]
                exec_box[0] = None
                if owns_rt and active is not None:
                    await asyncio.to_thread(close_runtime, active)
                if pool is not None:
                    await asyncio.to_thread(pool.shutdown, True)

    async def _run_offloaded(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_ensure_executor(holder["rt"]), functools.partial(fn, *args, **kwargs))

    mcp = FastMCP("yaw", strict_input_validation=True, lifespan=_lifespan)

    @mcp.custom_route("/healthz", methods=["GET"])
    async def healthz(_request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"}, status_code=200)

    @mcp.custom_route("/readyz", methods=["GET"])
    async def readyz(_request: Request) -> JSONResponse:
        active = holder["rt"]
        if active is None:
            return JSONResponse({"status": "starting"}, status_code=503)
        st = await _run_offloaded(get_status, active)
        if st.lifecycle == "ready" and not st.provider_ready:
            return JSONResponse({"status": "stopping"}, status_code=503)
        return JSONResponse({"status": st.lifecycle}, status_code=200 if st.lifecycle == "ready" else 503)

    @mcp.tool(
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False}
    )
    async def recall(
        query: Annotated[str, Field(min_length=1)],
        top_k: Annotated[int, Field(ge=1, le=20)] = 5,
    ) -> dict[str, Any]:
        """Retrieve ranked disk-hydrated corpus passages using zero-LLM RetrievalIndex."""
        if (active := holder["rt"]) is None:
            raise ToolError("[NOT_READY] Daemon runtime is still starting.")
        try:
            passages = await _run_offloaded(runtime_recall, active, query, top_k=top_k)
        except YawRuntimeError as exc:
            raise _to_tool_error(exc, active.config.api_key) from exc
        return {
            "query": query.strip(),
            "top_k": top_k,
            "results": [serialize_passage(idx, hp) for idx, hp in enumerate(passages, start=1)],
        }

    @mcp.tool(
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False}
    )
    async def qa(question: Annotated[str, Field(min_length=1)]) -> dict[str, Any]:
        """Answer a multi-hop question using bare ReAct grounded in retrieved corpus passages."""
        if (active := holder["rt"]) is None:
            raise ToolError("[NOT_READY] Daemon runtime is still starting.")
        try:
            res = await _run_offloaded(runtime_ask, active, question)
        except YawRuntimeError as exc:
            raise _to_tool_error(exc, active.config.api_key) from exc
        return {
            "question": question.strip(),
            "answer": res.answer,
            "turns": int(res.turns),
            "citations": [serialize_passage(idx, hp) for idx, hp in enumerate(res.passages, start=1)],
        }

    @mcp.tool(
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False}
    )
    async def daemon_status() -> dict[str, Any]:
        """Return a non-blocking redacted snapshot of daemon lifecycle and chunk counts."""
        if (active := holder["rt"]) is None:
            return DaemonStatus(
                lifecycle="starting",
                provider_ready=False,
                sync_state="syncing",
                pid=os.getpid(),
            ).to_dict()
        return (await _run_offloaded(get_status, active)).to_dict()

    app = mcp.http_app(path=mount_path, middleware=[StarletteMiddleware(_LoopbackGuardMiddleware)])
    return mcp, app


def serve_http(cfg: YawConfig) -> None:
    """Bind loopback Streamable HTTP transport and run the live YawRuntime until stopped."""
    _validate_loopback_bind(cfg.http_host)
    if isinstance(cfg.http_port, bool) or not isinstance(cfg.http_port, int) or not (1 <= cfg.http_port <= 65535):
        raise YawRuntimeError("INVALID_INPUT", f"http_port must be in 1..65535, got {cfg.http_port!r}.")
    if not isinstance(cfg.http_path, str) or not cfg.http_path.startswith("/"):
        raise YawRuntimeError("INVALID_INPUT", f"http_path must start with '/', got {cfg.http_path!r}.")

    server_box: list[uvicorn.Server] = []
    rt_box: list[YawRuntime] = []

    def _build_runtime() -> YawRuntime:
        rt = start_runtime(cfg, live=True, wait_for_ready=False)
        rt_box.append(rt)

        def _watch_stop() -> None:
            rt.stop_event.wait()
            if server_box:
                server_box[0].should_exit = True

        threading.Thread(target=_watch_stop, name="yaw-http-stop-watcher", daemon=True).start()
        return rt

    _, app = create_mcp_app(runtime_factory=_build_runtime, http_path=cfg.http_path)
    host = cfg.http_host[1:-1] if cfg.http_host.startswith("[") and cfg.http_host.endswith("]") else cfg.http_host
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=cfg.http_port, log_level="warning", lifespan="on"))
    server_box.append(server)
    try:
        server.run()
    finally:
        if rt_box:
            rt_box[0].stop_event.set()
