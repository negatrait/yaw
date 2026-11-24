"""Thin FastMCP Streamable HTTP adapter with loopback Host/Origin 403 guard and unified L3 lifespan."""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import ipaddress
import math
import os
import socket
import sys
from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Annotated, Any

import uvicorn
from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field
from starlette.middleware import Middleware as StarletteMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from yaw.core import DaemonStatus, YawConfig, YawRuntimeError
from yaw.runtime import (
    YawRuntime,
    acquire_daemon_control,
    redact_secrets,
    runtime_lifespan,
    serialize_passage,
    snapshot_status,
)
from yaw.runtime import (
    aask as runtime_aask,
)
from yaw.runtime import (
    arecall as runtime_arecall,
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


def _parse_wait_param(request: Request) -> float | None:
    if "wait" not in request.query_params:
        return 0.0
    try:
        val = float(request.query_params["wait"])
    except ValueError:
        return None
    if not math.isfinite(val) or val < 0.0:
        return None
    return val


def _build_mcp_lifespan(
    rt: YawRuntime | None,
    cfg: YawConfig | None,
    runtime_lifespan_factory: Callable[[asyncio.Event], AbstractAsyncContextManager[YawRuntime]] | None,
    holder: dict[str, YawRuntime | None],
    evt: asyncio.Event,
) -> Callable[[FastMCP], AbstractAsyncContextManager[dict[str, YawRuntime | None]]]:
    if rt is not None:

        @asynccontextmanager
        async def _static_lifespan(_server: FastMCP) -> AsyncGenerator[dict[str, YawRuntime | None]]:
            holder["rt"] = rt
            if rt.lifecycle != "starting":
                evt.set()
            yield {"rt": rt}

        return _static_lifespan

    @asynccontextmanager
    async def _managed_lifespan(_server: FastMCP) -> AsyncGenerator[dict[str, YawRuntime | None]]:
        if runtime_lifespan_factory is not None:
            cm = runtime_lifespan_factory(evt)
        else:
            assert cfg is not None
            cm = runtime_lifespan(cfg, live=True, wait_for_ready=False, ready_evt=evt)
        try:
            async with cm as active_rt:
                _validate_loopback_bind(active_rt.config.http_host)
                holder["rt"] = active_rt
                try:
                    yield {"rt": active_rt}
                finally:
                    holder["rt"] = None
        except BaseException:
            evt.set()
            raise

    return _managed_lifespan


def _register_http_routes(
    mcp: FastMCP,
    holder: dict[str, YawRuntime | None],
    ready_evt: asyncio.Event,
    control_token: str | None,
    stop_cb: Callable[[], None] | None,
) -> None:
    @mcp.custom_route("/healthz", methods=["GET"])
    async def healthz(_request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"}, status_code=200)

    @mcp.custom_route("/readyz", methods=["GET"])
    async def readyz(request: Request) -> JSONResponse:
        wait_val = _parse_wait_param(request)
        if wait_val is None:
            return JSONResponse({"error": "Invalid wait query parameter."}, status_code=400)
        active = holder["rt"]
        unsettled = active is None or snapshot_status(active).lifecycle == "starting"
        if unsettled and wait_val > 0.0:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(ready_evt.wait(), timeout=wait_val)
            active = holder["rt"]
        if active is None:
            return JSONResponse({"status": "starting"}, status_code=503)
        st = snapshot_status(active)
        if st.lifecycle == "ready" and not st.provider_ready:
            return JSONResponse({"status": "stopping"}, status_code=503)
        return JSONResponse({"status": st.lifecycle}, status_code=200 if st.lifecycle == "ready" else 503)

    @mcp.custom_route("/status", methods=["GET"])
    async def status_route(_request: Request) -> JSONResponse:
        active = holder["rt"]
        if active is None:
            return JSONResponse(
                DaemonStatus(
                    lifecycle="starting",
                    provider_ready=False,
                    sync_state="syncing",
                    pid=os.getpid(),
                ).to_dict(),
                status_code=200,
            )
        return JSONResponse(snapshot_status(active).to_dict(), status_code=200)

    @mcp.custom_route("/_internal/stop", methods=["POST"])
    async def internal_stop(request: Request) -> JSONResponse:
        hdrs = request.headers.getlist("x-yaw-control-token")
        if control_token is None or len(hdrs) != 1 or not hmac.compare_digest(hdrs[0], control_token):
            return JSONResponse({"error": "Forbidden"}, status_code=403)
        if stop_cb is not None:
            stop_cb()
        ready_evt.set()
        return JSONResponse({"status": "stopping"}, status_code=200, headers={"Connection": "close"})


def _register_mcp_tools(mcp: FastMCP, holder: dict[str, YawRuntime | None]) -> None:
    @mcp.tool(
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False}
    )
    async def recall(
        query: Annotated[str, Field(min_length=1)],
        top_k: Annotated[int, Field(ge=1, le=20)] = 5,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Retrieve ranked disk-hydrated corpus passages using zero-LLM RetrievalIndex."""
        active = (ctx.lifespan_context.get("rt") if ctx is not None else None) or holder["rt"]
        if active is None:
            raise ToolError("[NOT_READY] Daemon runtime is still starting.")
        try:
            passages = await runtime_arecall(active, query, top_k=top_k)
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
    async def qa(question: Annotated[str, Field(min_length=1)], ctx: Context | None = None) -> dict[str, Any]:
        """Answer a multi-hop question using bare ReAct grounded in retrieved corpus passages."""
        active = (ctx.lifespan_context.get("rt") if ctx is not None else None) or holder["rt"]
        if active is None:
            raise ToolError("[NOT_READY] Daemon runtime is still starting.")
        try:
            res = await runtime_aask(active, question)
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
    async def daemon_status(ctx: Context | None = None) -> dict[str, Any]:
        """Return a non-blocking redacted snapshot of daemon lifecycle and chunk counts."""
        active = (ctx.lifespan_context.get("rt") if ctx is not None else None) or holder["rt"]
        if active is None:
            return DaemonStatus(
                lifecycle="starting",
                provider_ready=False,
                sync_state="syncing",
                pid=os.getpid(),
            ).to_dict()
        return snapshot_status(active).to_dict()


def _resolve_mount_path(rt: YawRuntime | None, cfg: YawConfig | None, http_path: str | None) -> str:
    if http_path is not None:
        mount_path = http_path
    elif rt is not None:
        mount_path = rt.config.http_path
    elif cfg is not None:
        mount_path = cfg.http_path
    else:
        mount_path = "/mcp"
    if not isinstance(mount_path, str) or not mount_path.startswith("/"):
        raise YawRuntimeError("INVALID_INPUT", f"http_path must start with '/', got {mount_path!r}.")
    return mount_path


def create_mcp_app(
    rt: YawRuntime | None = None,
    *,
    cfg: YawConfig | None = None,
    runtime_lifespan_factory: Callable[[asyncio.Event], AbstractAsyncContextManager[YawRuntime]] | None = None,
    http_path: str | None = None,
    ready_evt: asyncio.Event | None = None,
    control_token: str | None = None,
    stop_cb: Callable[[], None] | None = None,
) -> tuple[FastMCP, Any]:
    """Create the FastMCP server and pre-dispatch-guarded Starlette Streamable HTTP ASGI app."""
    if sum(arg is not None for arg in (rt, cfg, runtime_lifespan_factory)) != 1:
        raise YawRuntimeError("INVALID_INPUT", "Provide exactly one of rt, cfg, or runtime_lifespan_factory.")
    mount_path = _resolve_mount_path(rt, cfg, http_path)
    if rt is not None:
        _validate_loopback_bind(rt.config.http_host)
    if cfg is not None:
        _validate_loopback_bind(cfg.http_host)

    holder: dict[str, YawRuntime | None] = {"rt": rt}
    evt = (
        ready_evt
        if ready_evt is not None
        else (rt.ready_evt if rt is not None and rt.ready_evt is not None else asyncio.Event())
    )
    lifespan = _build_mcp_lifespan(rt, cfg, runtime_lifespan_factory, holder, evt)
    mcp = FastMCP("yaw", strict_input_validation=True, lifespan=lifespan)
    _register_http_routes(mcp, holder, evt, control_token, stop_cb)
    _register_mcp_tools(mcp, holder)
    app = mcp.http_app(path=mount_path, middleware=[StarletteMiddleware(_LoopbackGuardMiddleware)])
    return mcp, app


def _bind_loopback_socket(host: str, port: int) -> socket.socket:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.listen(128)
        sock.setblocking(False)
        return sock
    except OSError as exc:
        sock.close()
        raise YawRuntimeError("BUSY", f"Port {host}:{port} is already in use or unavailable: {exc}") from exc


def _emit_listening_sentinel_and_silence_stdout() -> None:
    sys.stdout.write("YAW_LISTENING\n")
    sys.stdout.flush()
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 1)
    finally:
        os.close(devnull_fd)


def serve_http(cfg: YawConfig, *, notify_stdout: bool = False) -> None:
    """Bind loopback Streamable HTTP transport and run unified YawRuntime until stopped."""
    _validate_loopback_bind(cfg.http_host)
    if isinstance(cfg.http_port, bool) or not isinstance(cfg.http_port, int) or not (1 <= cfg.http_port <= 65535):
        raise YawRuntimeError("INVALID_INPUT", f"http_port must be in 1..65535, got {cfg.http_port!r}.")
    if not isinstance(cfg.http_path, str) or not cfg.http_path.startswith("/"):
        raise YawRuntimeError("INVALID_INPUT", f"http_path must start with '/', got {cfg.http_path!r}.")

    host = cfg.http_host[1:-1] if cfg.http_host.startswith("[") and cfg.http_host.endswith("]") else cfg.http_host
    with acquire_daemon_control(cfg) as (file_lock, control_token):
        evt = asyncio.Event()
        sock = _bind_loopback_socket(host, cfg.http_port)
        try:
            server_box: list[uvicorn.Server] = []

            def _stop_cb() -> None:
                if server_box:
                    server_box[0].should_exit = True

            _, app = create_mcp_app(
                runtime_lifespan_factory=lambda ready_e: runtime_lifespan(
                    cfg, live=True, wait_for_ready=False, file_lock=file_lock, ready_evt=ready_e
                ),
                http_path=cfg.http_path,
                ready_evt=evt,
                control_token=control_token,
                stop_cb=_stop_cb,
            )
            server = uvicorn.Server(
                uvicorn.Config(
                    app,
                    host=host,
                    port=cfg.http_port,
                    timeout_graceful_shutdown=2,
                    log_level="warning",
                    lifespan="on",
                )
            )
            server_box.append(server)
            if notify_stdout:
                _emit_listening_sentinel_and_silence_stdout()
            server.run(sockets=[sock])
        finally:
            sock.close()
