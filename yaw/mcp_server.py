"""Thin FastMCP Streamable HTTP adapter with loopback Host/Origin 403 guard and 4+4 Librarian surface."""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import ipaddress
import json
import math
import os
import re
import socket
import sys
from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Annotated, Any

import uvicorn
from fastmcp import Context, FastMCP
from fastmcp.exceptions import ResourceError, ToolError
from fastmcp.tools import ToolResult
from pydantic import Field
from starlette.middleware import Middleware as StarletteMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from yaw.core import DaemonStatus, YawConfig, YawRuntimeError
from yaw.runtime import (
    YawRuntime,
    acquire_daemon_control,
    dump_recall_json,
    format_recall_text,
    list_wiki_entries,
    read_wiki_entry,
    redact_secrets,
    runtime_lifespan,
    snapshot_status,
    write_wiki_and_sync,
)
from yaw.runtime import (
    aexpand_neighbors_payload as runtime_aexpand_neighbors_payload,
)
from yaw.runtime import (
    aread_chunk_payload as runtime_aread_chunk_payload,
)
from yaw.runtime import (
    arecall_payload as runtime_arecall_payload,
)

_CHUNK_ID_RE = re.compile(r"[1-9]\d*")
_MCP_INSTRUCTIONS = (
    "YAW Librarian MCP server (zero-LLM hybrid retrieval + user-editable Markdown wiki). "
    "Tools: recall(query, top_k=5), expand_neighbors(chunk_id, top_k=6), read_chunk(chunk_id), "
    "write_wiki(slug, title, body, source_chunk_ids=None, kind='note', wait_for_sync=True). "
    "Resources: yaw://status (JSON DaemonStatus), yaw://wiki (JSON wiki card index), "
    "yaw://wiki/{slug*} (Markdown card), yaw://chunk/{chunk_id} (JSON exact chunk slip). "
    "Multi-hop QA (qa) is CLI-only via `yaw qa` and is not exposed as an MCP tool."
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


def _to_resource_error(exc: YawRuntimeError, api_key: str) -> ResourceError:
    return ResourceError(redact_secrets(str(exc), api_key))


def _resolve_active_rt(ctx: Context | None, holder: dict[str, YawRuntime | None]) -> YawRuntime | None:
    ctx_rt = ctx.lifespan_context.get("rt") if ctx is not None else None
    return ctx_rt if isinstance(ctx_rt, YawRuntime) else holder["rt"]


def _resolve_tool_rt(ctx: Context | None, holder: dict[str, YawRuntime | None]) -> YawRuntime:
    active = _resolve_active_rt(ctx, holder)
    if active is None:
        raise ToolError("[NOT_READY] Daemon runtime is still starting.")
    return active


def _resolve_resource_rt(ctx: Context | None, holder: dict[str, YawRuntime | None]) -> YawRuntime:
    active = _resolve_active_rt(ctx, holder)
    if active is None:
        raise ResourceError("[NOT_READY] Daemon runtime is still starting.")
    return active


def _render_tool_result(payload: dict[str, Any]) -> ToolResult:
    dump_recall_json(payload)
    return ToolResult(content=format_recall_text(payload), structured_content=payload)


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
    ) -> ToolResult:
        """Retrieve ranked disk-hydrated corpus and wiki passages using zero-LLM RetrievalIndex."""
        active = _resolve_tool_rt(ctx, holder)
        try:
            payload = await runtime_arecall_payload(active, query, top_k=top_k)
        except YawRuntimeError as exc:
            raise _to_tool_error(exc, active.config.api_key) from exc
        return _render_tool_result(payload)

    @mcp.tool(
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False}
    )
    async def expand_neighbors(
        chunk_id: Annotated[int, Field(ge=1)],
        top_k: Annotated[int, Field(ge=1, le=20)] = 6,
        ctx: Context | None = None,
    ) -> ToolResult:
        """Expand citation, lexical, semantic, and AST-island neighbors around a verified chunk_id."""
        active = _resolve_tool_rt(ctx, holder)
        try:
            payload = await runtime_aexpand_neighbors_payload(active, chunk_id, top_k=top_k)
        except YawRuntimeError as exc:
            raise _to_tool_error(exc, active.config.api_key) from exc
        return _render_tool_result(payload)

    @mcp.tool(
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False}
    )
    async def read_chunk(
        chunk_id: Annotated[int, Field(ge=1)],
        ctx: Context | None = None,
    ) -> ToolResult:
        """Read the exact verified disk span for a single chunk_id."""
        active = _resolve_tool_rt(ctx, holder)
        try:
            payload = await runtime_aread_chunk_payload(active, chunk_id)
        except YawRuntimeError as exc:
            raise _to_tool_error(exc, active.config.api_key) from exc
        return _render_tool_result(payload)

    @mcp.tool(
        annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False}
    )
    async def write_wiki(
        slug: Annotated[str, Field(min_length=1)],
        title: Annotated[str, Field(min_length=1)],
        body: Annotated[str, Field(min_length=1)],
        source_chunk_ids: list[Annotated[int, Field(ge=1)]] | None = None,
        kind: str = "note",
        wait_for_sync: bool = True,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Write a grounded Markdown wiki card under wiki_root and reactively index it."""
        active = _resolve_tool_rt(ctx, holder)
        try:
            return await write_wiki_and_sync(
                active,
                slug=slug,
                title=title,
                body=body,
                source_chunk_ids=tuple(source_chunk_ids or ()),
                kind=kind,
                wait_for_sync=wait_for_sync,
            )
        except YawRuntimeError as exc:
            raise _to_tool_error(exc, active.config.api_key) from exc


def _register_mcp_resources(mcp: FastMCP, holder: dict[str, YawRuntime | None]) -> None:
    @mcp.resource("yaw://status", mime_type="application/json")
    async def status_resource(ctx: Context | None = None) -> str:
        """Return a non-blocking redacted snapshot of daemon lifecycle and chunk counts."""
        active = _resolve_active_rt(ctx, holder)
        if active is None:
            st = DaemonStatus(
                lifecycle="starting",
                provider_ready=False,
                sync_state="syncing",
                pid=os.getpid(),
            ).to_dict()
        else:
            st = snapshot_status(active).to_dict()
        return json.dumps(st)

    @mcp.resource("yaw://wiki", mime_type="application/json")
    async def wiki_index_resource(ctx: Context | None = None) -> str:
        """List metadata for all active Markdown wiki cards under wiki_root."""
        active = _resolve_resource_rt(ctx, holder)
        try:
            entries = await asyncio.to_thread(list_wiki_entries, active)
        except YawRuntimeError as exc:
            raise _to_resource_error(exc, active.config.api_key) from exc
        return json.dumps(entries)

    @mcp.resource("yaw://wiki/{slug*}", mime_type="text/markdown")
    async def wiki_entry_resource(slug: str, ctx: Context | None = None) -> str:
        """Read a single validated Markdown wiki card confined strictly inside wiki_root."""
        active = _resolve_resource_rt(ctx, holder)
        try:
            return await asyncio.to_thread(read_wiki_entry, active, slug)
        except YawRuntimeError as exc:
            raise _to_resource_error(exc, active.config.api_key) from exc

    @mcp.resource("yaw://chunk/{chunk_id}", mime_type="application/json")
    async def chunk_resource(chunk_id: str, ctx: Context | None = None) -> str:
        """Read the exact verified disk span slip payload for chunk_id."""
        raw_cid = chunk_id.strip()
        if not _CHUNK_ID_RE.fullmatch(raw_cid):
            raise ResourceError(f"[INVALID_INPUT] Invalid chunk_id={chunk_id!r}.")
        active = _resolve_resource_rt(ctx, holder)
        try:
            payload = await runtime_aread_chunk_payload(active, int(raw_cid))
        except YawRuntimeError as exc:
            raise _to_resource_error(exc, active.config.api_key) from exc
        return dump_recall_json(payload)


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
    mcp = FastMCP("yaw", instructions=_MCP_INSTRUCTIONS, strict_input_validation=True, lifespan=lifespan)
    _register_http_routes(mcp, holder, evt, control_token, stop_cb)
    _register_mcp_tools(mcp, holder)
    _register_mcp_resources(mcp, holder)
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
