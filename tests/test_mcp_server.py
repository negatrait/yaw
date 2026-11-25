"""Unmocked integration tests for FastMCP Streamable HTTP adapter, loopback guards, and U3 lifespan contracts."""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, override

import filelock
import httpx2 as httpx
import numpy as np
import pytest
import uvicorn
from fastmcp import Client
from fastmcp.exceptions import ResourceError, ToolError
from mcp.shared.exceptions import MCPError

from yaw.core import ChunkPointer, HydratedPassage, YawConfig, YawRuntimeError
from yaw.mcp_server import create_mcp_app, serve_http
from yaw.pipeline import open_managed_db, reconcile_index_tables
from yaw.qa import create_lm
from yaw.runtime import (
    build_recall_payload,
    close_runtime,
    control_token_path,
    format_recall_text,
    runtime_lifespan,
    start_runtime,
    sync_embed_concurrency_limit,
)


def _topic_embed_vec(text: str, *, dim: int = 16) -> np.ndarray:
    topics = (
        ("orbital", "booster", "valve", "specification", "stage"),
        ("xylophone", "zither", "ohio", "armstrong", "musician"),
        ("apollo", "lunar", "moon", "commander"),
    )
    low = text.lower()
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    base = (np.frombuffer(digest * 2, dtype=np.uint8)[:dim].astype(np.float32) - 127.5) * 0.05
    for t_idx, kws in enumerate(topics):
        for kw in kws:
            if kw in low:
                base[t_idx] += 2.0
    return base / max(float(np.linalg.norm(base)), 1e-12)


def _topic_embed(texts: list[str], *, dim: int = 16) -> np.ndarray:
    if not texts:
        return np.zeros((0, dim), dtype=np.float32)
    return np.vstack([_topic_embed_vec(t, dim=dim) for t in texts]).astype(np.float32, copy=False)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _make_cfg(
    root: Path,
    *,
    http_host: str = "127.0.0.1",
    http_port: int | None = None,
    http_path: str = "/mcp",
    base_url: str = "http://127.0.0.1:8888/v1",
    api_key: str = "test",
    queue_maxsize: int = 32,
    embed_max_in_flight: int = 2,
    wiki_root: Path | None = None,
    qa_model: str = "gemma-4-e4b",
) -> YawConfig:
    return YawConfig(
        workspace_root=root,
        corpus_root=root / "corpus",
        wiki_root=wiki_root,
        db_path=root / ".yaw" / "yaw.db",
        lock_path=root / ".yaw" / "yaw.db.lock",
        state_dir=root / ".yaw" / "coco_lmdb",
        log_path=root / ".yaw" / "yaw.log",
        status_path=root / ".yaw" / "status.json",
        compiled_program_path=root / ".yaw" / "compiled_program.json",
        base_url=base_url,
        api_key=api_key,
        qa_model=qa_model,
        chunk_size=140,
        chunk_overlap=20,
        min_chunk_size=20,
        queue_maxsize=queue_maxsize,
        embed_max_in_flight=embed_max_in_flight,
        http_host=http_host,
        http_port=http_port if http_port is not None else _free_port(),
        http_path=http_path,
    )


async def _read_text_resource(client: Client[Any], uri: str) -> str:
    items = await client.read_resource(uri)
    assert len(items) == 1
    return str(getattr(items[0], "text", ""))


async def _read_json_resource(client: Client[Any], uri: str) -> Any:
    return json.loads(await _read_text_resource(client, uri))


def _tool_reply(thought: str, calls: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "reasoning_content": thought,
        "tool_calls": [
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {"name": call["name"], "arguments": json.dumps(call["args"])},
            }
            for i, call in enumerate(calls)
        ],
    }


def _start_openai_stub(
    chat_responder: Any,
    *,
    embed_gate: threading.Event | None = None,
    hold_query_marker: str | None = None,
    embed_delay_s: float = 0.0,
) -> tuple[ThreadingHTTPServer, dict[str, int]]:
    stats = {"in_flight": 0, "peak": 0}
    lock = threading.Lock()

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if self.path.endswith("/embeddings"):
                raw_inputs = payload.get("input", [])
                texts = raw_inputs if isinstance(raw_inputs, list) else [raw_inputs]
                with lock:
                    stats["in_flight"] += 1
                    stats["peak"] = max(stats["peak"], stats["in_flight"])
                try:
                    if embed_delay_s > 0.0 and (
                        hold_query_marker is None or any(hold_query_marker in str(t) for t in texts)
                    ):
                        time.sleep(embed_delay_s)
                    if embed_gate is not None and (
                        hold_query_marker is None or any(hold_query_marker in str(t) for t in texts)
                    ):
                        embed_gate.wait(timeout=15.0)
                finally:
                    with lock:
                        stats["in_flight"] -= 1
                data = [
                    {
                        "object": "embedding",
                        "index": i,
                        "embedding": _topic_embed_vec(str(t), dim=16).tolist(),
                    }
                    for i, t in enumerate(texts)
                ]
                out = {
                    "object": "list",
                    "data": data,
                    "model": payload.get("model", "embeddinggemma-2"),
                    "usage": {"prompt_tokens": len(texts), "total_tokens": len(texts)},
                }
            else:
                content = chat_responder(payload)
                out = {
                    "id": "chatcmpl-stub",
                    "object": "chat.completion",
                    "created": 1,
                    "model": payload.get("model", "gemma-4-e4b"),
                    "choices": [
                        {
                            "index": 0,
                            "message": content
                            if isinstance(content, dict)
                            else {"role": "assistant", "content": content},
                            "finish_reason": "tool_calls" if isinstance(content, dict) else "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                }
            raw = json.dumps(out).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        @override
        def log_message(self, format: str, *args: Any) -> None:
            _ = (format, args)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, stats


def _stop_via_control_token(cfg: YawConfig) -> httpx.Response:
    tok_path = control_token_path(cfg)
    assert tok_path.is_file()
    token = tok_path.read_text(encoding="ascii").strip()
    return httpx.post(
        f"http://127.0.0.1:{cfg.http_port}/_internal/stop",
        headers={"X-Yaw-Control-Token": token},
        timeout=5.0,
    )


def _wait_http_ready(port: int, srv_errors: list[BaseException]) -> None:
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/readyz", timeout=1.0)
            if r.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.05)
    pytest.fail(f"serve_http did not become ready; errors={srv_errors}")


def test_mcp_server_containment_bind_validation_and_tool_inventory(tmp_path: Path) -> None:
    mcp_path = Path(__file__).resolve().parent.parent / "yaw" / "mcp_server.py"
    src = mcp_path.read_text(encoding="utf-8")
    low = src.lower()
    assert "manifold" not in low and "hmm" not in low and "hipporag" not in low
    for retired in ("ThreadPoolExecutor", "yaw-http-stop-watcher", "filelock"):
        assert retired not in src

    tree = ast.parse(src, filename=str(mcp_path))
    allowed_top = sys.stdlib_module_names | {"yaw", "fastmcp", "pydantic", "starlette", "uvicorn"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] in allowed_top, f"Forbidden import: {alias.name}"
                if alias.name.startswith("yaw"):
                    assert alias.name == "yaw.runtime" or alias.name.startswith("yaw.core")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert mod.split(".")[0] in allowed_top, f"Forbidden from-import: {mod}"
            if mod.startswith("yaw"):
                assert mod == "yaw.runtime" or mod == "yaw.core" or mod.startswith("yaw.core.")

    for bad_host in ("0.0.0.0", "::", "192.168.1.10", "evil.example", ""):
        bad_cfg = _make_cfg(tmp_path / "badh", http_host=bad_host)
        with pytest.raises(YawRuntimeError) as exc_h:
            serve_http(bad_cfg)
        assert exc_h.value.code == "INVALID_INPUT"

    with pytest.raises(YawRuntimeError) as exc_cfg_p:
        _make_cfg(tmp_path / "badp", http_path="mcp")
    assert exc_cfg_p.value.code == "INVALID_INPUT"

    bad_path_cfg = _make_cfg(tmp_path / "badp")
    object.__setattr__(bad_path_cfg, "http_path", "mcp")
    with pytest.raises(YawRuntimeError) as exc_p:
        serve_http(bad_path_cfg)
    assert exc_p.value.code == "INVALID_INPUT"

    with pytest.raises(YawRuntimeError) as exc_none:
        create_mcp_app()
    assert exc_none.value.code == "INVALID_INPUT"

    bad_factory_cfg = _make_cfg(tmp_path / "bad_factory", http_host="0.0.0.0")
    assert bad_factory_cfg.corpus_root is not None and bad_factory_cfg.lock_path is not None
    bad_factory_cfg.corpus_root.mkdir(parents=True)
    _, bad_factory_app = create_mcp_app(
        runtime_lifespan_factory=lambda ready_e: runtime_lifespan(
            bad_factory_cfg,
            live=False,
            wait_for_ready=True,
            embed_fn=_topic_embed,
            lm=create_lm(bad_factory_cfg),
            ready_evt=ready_e,
        )
    )

    async def _check_bad_factory() -> None:
        with pytest.raises(YawRuntimeError) as exc_factory:
            async with bad_factory_app.lifespan(bad_factory_app):
                pass
        assert exc_factory.value.code == "INVALID_INPUT"

    asyncio.run(_check_bad_factory())
    factory_lock = filelock.FileLock(str(bad_factory_cfg.lock_path), timeout=0)
    factory_lock.acquire(timeout=0)
    factory_lock.release()

    cfg = _make_cfg(tmp_path / "ok")
    assert cfg.corpus_root is not None
    cfg.corpus_root.mkdir(parents=True)
    rt = start_runtime(cfg, live=False, wait_for_ready=True, embed_fn=_topic_embed, lm=create_lm(cfg))
    try:
        with pytest.raises(YawRuntimeError) as exc_both:
            create_mcp_app(rt=rt, cfg=cfg)
        assert exc_both.value.code == "INVALID_INPUT"

        mcp, _ = create_mcp_app(rt=rt)

        async def _inspect() -> None:
            async with Client(mcp) as client:
                tools = await client.list_tools()
                by_name = {t.name: t for t in tools}
                assert set(by_name) == {"recall", "expand_neighbors", "read_chunk", "write_wiki"}
                for name, ro_hint in (
                    ("recall", True),
                    ("expand_neighbors", True),
                    ("read_chunk", True),
                    ("write_wiki", False),
                ):
                    ann = by_name[name].annotations
                    assert ann is not None
                    assert ann.read_only_hint is ro_hint
                    assert ann.destructive_hint is False
                    assert ann.idempotent_hint is True
                    assert ann.open_world_hint is False
                assert "query" in by_name["recall"].input_schema.get("properties", {})
                assert "chunk_id" in by_name["expand_neighbors"].input_schema.get("properties", {})
                assert "chunk_id" in by_name["read_chunk"].input_schema.get("properties", {})
                assert {"slug", "title", "body"} <= set(by_name["write_wiki"].input_schema.get("properties", {}))

                resources = {str(r.uri): r for r in await client.list_resources()}
                assert set(resources) == {"yaw://status", "yaw://wiki"}
                assert resources["yaw://status"].mime_type == "application/json"
                assert resources["yaw://wiki"].mime_type == "application/json"

                templates = {str(t.uri_template): t for t in await client.list_resource_templates()}
                assert set(templates) == {"yaw://wiki/{slug*}", "yaw://chunk/{chunk_id}"}
                assert templates["yaw://wiki/{slug*}"].mime_type == "text/markdown"
                assert templates["yaw://chunk/{chunk_id}"].mime_type == "application/json"

                instr = client.instructions or ""
                for token in (
                    "recall",
                    "expand_neighbors",
                    "read_chunk",
                    "write_wiki",
                    "yaw://status",
                    "yaw://wiki",
                    "yaw://wiki/{slug*}",
                    "yaw://chunk/{chunk_id}",
                    "yaw qa",
                ):
                    assert token in instr
            async with Client(mcp, mode="legacy") as legacy:
                assert legacy.initialize_result is not None
                assert legacy.initialize_result.instructions == instr

        asyncio.run(_inspect())
    finally:
        close_runtime(rt)


async def _send_route(hc: httpx.AsyncClient, path: str, headers: dict[str, str]) -> httpx.Response:
    if path in ("/mcp", "/_internal/stop"):
        return await hc.post(path, headers=headers)
    return await hc.get(path, headers=headers)


async def _assert_guard_rejections(hc: httpx.AsyncClient) -> None:
    routes = ("/healthz", "/readyz", "/status", "/_internal/stop", "/mcp")
    for bad_host in (
        "evil.example",
        "127.0.0.1.evil.example",
        "0.0.0.0:13306",
        "127.0.0.1:0",
        "127.0.0.1:70000",
        "::1:13306",
        "",
    ):
        for path in routes:
            res = await _send_route(hc, path, {"Host": bad_host})
            assert res.status_code == 403, f"Expected 403 for Host={bad_host!r} on {path}"

    for bad_origin in (
        "null",
        "http://evil.example",
        "http://user:pass@127.0.0.1:13306",
        "http://127.0.0.1:13306/path",
        "http://127.0.0.1:13306?x=1",
        "http://127.0.0.1:13306#frag",
        "file:///tmp",
    ):
        for path in routes:
            res = await _send_route(hc, path, {"Host": "127.0.0.1:13306", "Origin": bad_origin})
            assert res.status_code == 403, f"Expected 403 for Origin={bad_origin!r} on {path}"

    for ok_host, ok_origin in (
        ("127.0.0.1:13306", None),
        ("localhost:13306", "http://localhost:3000"),
        ("[::1]:13306", "https://[::1]:13306"),
    ):
        headers = {"Host": ok_host}
        if ok_origin is not None:
            headers["Origin"] = ok_origin
        assert (await hc.get("/healthz", headers=headers)).status_code == 200


async def _assert_raw_asgi_guard(app: Any) -> None:
    async def _recv() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    for raw_hdrs in (
        [],
        [(b"host", b"127.0.0.1:13306"), (b"host", b"evil.example")],
        [(b"host", b"127.0.0.1:13306"), (b"origin", b"http://localhost"), (b"origin", b"http://evil.example")],
    ):
        captured_status: list[int] = []

        async def _send(msg: dict[str, Any], out: list[int] = captured_status) -> None:
            if msg["type"] == "http.response.start":
                out.append(int(msg["status"]))

        await app(
            {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "GET",
                "scheme": "http",
                "path": "/healthz",
                "raw_path": b"/healthz",
                "query_string": b"",
                "headers": raw_hdrs,
            },
            _recv,
            _send,
        )
        assert captured_status == [403]


def _assert_tcp_raw_guard(port: int) -> None:
    def _send_raw(req: bytes) -> str:
        with socket.create_connection(("127.0.0.1", port), timeout=2.0) as s:
            s.sendall(req)
            chunks = []
            while chunk := s.recv(4096):
                chunks.append(chunk)
            return b"".join(chunks).decode("latin1")

    r_ready = httpx.get(f"http://127.0.0.1:{port}/readyz", timeout=2.0)
    assert r_ready.status_code == 200 and r_ready.json() == {"status": "ready"}
    dup_origin_req = (
        b"GET /healthz HTTP/1.1\r\nHost: 127.0.0.1\r\n"
        b"Origin: http://127.0.0.1\r\nOrigin: http://evil.example\r\nConnection: close\r\n\r\n"
    )
    assert "403" in _send_raw(dup_origin_req).splitlines()[0]
    assert "403" in _send_raw(b"GET /healthz HTTP/1.0\r\nConnection: close\r\n\r\n").splitlines()[0]
    dup_host_status = _send_raw(
        b"GET /healthz HTTP/1.1\r\nHost: 127.0.0.1\r\nHost: evil.example\r\nConnection: close\r\n\r\n"
    ).splitlines()[0]
    assert ("400" in dup_host_status) or ("403" in dup_host_status)


def test_predispatch_host_origin_guard_and_health_readiness_lifecycle(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    assert cfg.corpus_root is not None and cfg.lock_path is not None
    cfg.corpus_root.mkdir(parents=True)
    (cfg.corpus_root / "init.md").write_text("# Init Note\nApollo 11 landed smoothly.\n", encoding="utf-8")

    probe_started = threading.Event()
    release_probe = threading.Event()
    holder: dict[str, Any] = {}

    def deferred_embed(texts: list[str]) -> np.ndarray:
        if any("Apollo 11" in t for t in texts):
            probe_started.set()
            assert release_probe.wait(timeout=5.0)
        return _topic_embed(texts)

    def _lifespan_factory(ready_e: asyncio.Event) -> Any:
        cm = runtime_lifespan(
            cfg, live=True, wait_for_ready=False, embed_fn=deferred_embed, lm=create_lm(cfg), ready_evt=ready_e
        )

        class _Capture:
            async def __aenter__(self) -> Any:
                holder["rt"] = await cm.__aenter__()
                return holder["rt"]

            async def __aexit__(self, *exc: Any) -> Any:
                return await cm.__aexit__(*exc)

        return _Capture()

    _, app = create_mcp_app(
        runtime_lifespan_factory=_lifespan_factory,
        http_path=cfg.http_path,
        control_token="secret-ctrl-tok",
    )

    async def _check_asgi() -> None:
        async with app.lifespan(app):
            assert await asyncio.to_thread(probe_started.wait, 5.0)
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:13306") as hc:
                assert (await hc.get("/healthz")).json() == {"status": "ok"}
                rz0 = await hc.get("/readyz")
                assert rz0.status_code == 503 and rz0.json()["status"] == "starting"
                for bad_wait in ("nan", "inf", "-1", "abc"):
                    assert (await hc.get(f"/readyz?wait={bad_wait}")).status_code == 400
                st0 = await hc.get("/status")
                assert st0.status_code == 200 and st0.json()["lifecycle"] == "starting"

                await _assert_guard_rejections(hc)
                assert (await hc.post("/_internal/stop")).status_code == 403
                assert (await hc.post("/_internal/stop", headers={"X-Yaw-Control-Token": "wrong"})).status_code == 403

                release_probe.set()
                rz1 = await hc.get("/readyz?wait=10")
                assert rz1.status_code == 200 and rz1.json() == {"status": "ready"}
                holder["rt"].lifecycle = "stopping"
                rz_stop = await hc.get("/readyz")
                assert rz_stop.status_code == 503 and rz_stop.json() == {"status": "stopping"}
            await _assert_raw_asgi_guard(app)

    try:
        asyncio.run(_check_asgi())
    finally:
        release_probe.set()

    lock = filelock.FileLock(str(cfg.lock_path), timeout=0)
    lock.acquire(timeout=0)
    lock.release()

    empty_cfg = _make_cfg(tmp_path / "empty_tcp")
    assert empty_cfg.corpus_root is not None
    empty_cfg.corpus_root.mkdir(parents=True)
    rt_empty = start_runtime(empty_cfg, live=False, wait_for_ready=True, embed_fn=_topic_embed, lm=create_lm(empty_cfg))
    try:
        _, tcp_app = create_mcp_app(rt=rt_empty)
        port = _free_port()
        srv = uvicorn.Server(uvicorn.Config(tcp_app, host="127.0.0.1", port=port, log_level="error", lifespan="on"))
        t = threading.Thread(target=srv.run, daemon=True)
        t.start()
        try:
            while not srv.started:
                time.sleep(0.02)
            _assert_tcp_raw_guard(port)
        finally:
            srv.should_exit = True
            t.join(timeout=5.0)
    finally:
        close_runtime(rt_empty)


def _assert_mcp_recall_tool_result(rec_res: Any, *, expected_text: str | None = None) -> dict[str, Any]:
    rec = rec_res.structured_content
    assert rec is not None and rec_res.data == rec
    assert len(rec_res.content) == 1
    rec_text = getattr(rec_res.content[0], "text", "")
    assert rec_text == format_recall_text(rec)
    if expected_text is not None:
        assert rec_text == expected_text
    assert "query" not in rec and "top_k" not in rec
    return rec


async def _assert_mcp_happy_tools_and_resources(client: Client[Any], secret_key: str) -> tuple[int, int]:
    rec_res = await client.call_tool("recall", {"query": "orbital booster", "top_k": 2})
    rec = _assert_mcp_recall_tool_result(rec_res)
    assert isinstance(rec["log_bf"], float) and isinstance(rec["coverage"], float)
    results = rec["results"]
    assert len(results) == 2
    assert any(r["source"]["rel_posix_path"] == "multi.json" for r in results)
    assert len({r["chunk_id"] for r in results}) == 2
    for rank_i, item in enumerate(results, start=1):
        assert item["rank"] == rank_i
        assert type(item["chunk_id"]) is int
        assert isinstance(item["score"], float)
        assert "stale" not in item and "stale_on_disk" not in item
        assert item["content_hash"] == f"sha256:{hashlib.sha256(item['text'].encode('utf-8')).hexdigest()}"
        expected_source_keys = {"rel_posix_path", "start_line", "end_line"}
        if item["source"]["rel_posix_path"].endswith(".md"):
            expected_source_keys.add("title")
        assert set(item["source"]) == expected_source_keys
        if item["source"]["rel_posix_path"].endswith(".md"):
            assert item["source"]["title"] in {"Apollo 11 Mission", "Booster Backup"}

    apollo_rec = _assert_mcp_recall_tool_result(await client.call_tool("recall", {"query": "Apollo 11 commander"}))
    alpha_cid = next(int(r["chunk_id"]) for r in apollo_rec["results"] if r["source"]["rel_posix_path"] == "alpha.md")
    first_cid = int(results[0]["chunk_id"])

    read_res = _assert_mcp_recall_tool_result(await client.call_tool("read_chunk", {"chunk_id": alpha_cid}))
    assert len(read_res["results"]) == 1 and read_res["results"][0]["chunk_id"] == alpha_cid
    exp_res = _assert_mcp_recall_tool_result(await client.call_tool("expand_neighbors", {"chunk_id": first_cid}))
    assert isinstance(exp_res["results"], list)

    st = await _read_json_resource(client, "yaw://status")
    assert st["lifecycle"] == "ready" and st["num_chunks"] >= 7
    assert secret_key not in json.dumps(st)
    chunk_res = await _read_json_resource(client, f"yaw://chunk/{alpha_cid}")
    assert chunk_res["results"][0]["chunk_id"] == alpha_cid

    long_q = "orbital booster " * 80
    assert len(long_q.encode("utf-8")) > 1024
    long_rec = (await client.call_tool("recall", {"query": long_q, "top_k": 2})).data
    assert len(long_rec["results"]) == 2
    return first_cid, alpha_cid


async def _assert_mcp_bad_inputs(client: Client[Any]) -> None:
    bad_calls = (
        ("recall", {"query": "ok", "top_k": True}),
        ("recall", {"query": "ok", "top_k": False}),
        ("recall", {"query": "ok", "top_k": "5"}),
        ("recall", {"query": "ok", "top_k": 5.0}),
        ("recall", {"query": "ok", "top_k": 0}),
        ("recall", {"query": "ok", "top_k": 21}),
        ("recall", {"query": ""}),
        ("recall", {"query": 123}),
        ("recall", {"query": "ok", "extra": 1}),
        ("expand_neighbors", {"chunk_id": 0}),
        ("expand_neighbors", {"chunk_id": -1}),
        ("expand_neighbors", {"chunk_id": True}),
        ("expand_neighbors", {"chunk_id": 1, "top_k": 21}),
        ("read_chunk", {"chunk_id": 0}),
        ("read_chunk", {"chunk_id": -1}),
        ("read_chunk", {"chunk_id": "1"}),
        ("write_wiki", {"slug": "", "title": "T", "body": "B"}),
        ("write_wiki", {"slug": "ok", "title": "", "body": "B"}),
        ("write_wiki", {"slug": "ok", "title": "T", "body": ""}),
        ("write_wiki", {"slug": "ok", "title": "T", "body": "B", "source_chunk_ids": [0]}),
        ("write_wiki", {"slug": "ok", "title": "T", "body": "B", "source_chunk_ids": [-1]}),
        ("write_wiki", {"slug": "ok", "title": "T", "body": "B", "source_chunk_ids": [True]}),
    )
    for tool_name, bad_args in bad_calls:
        with pytest.raises(ToolError):
            await client.call_tool(tool_name, bad_args)

    ws_calls = (
        ("recall", {"query": "   "}),
        ("expand_neighbors", {"chunk_id": 999999}),
        ("read_chunk", {"chunk_id": 999999}),
        ("write_wiki", {"slug": "../escape", "title": "T", "body": "B"}),
        ("write_wiki", {"slug": "ok-slug", "title": "   ", "body": "B"}),
    )
    for tool_name, ws_args in ws_calls:
        with pytest.raises(ToolError) as exc_ws:
            await client.call_tool(tool_name, ws_args)
        assert str(exc_ws.value).startswith("[INVALID_INPUT]")

    for bad_uri in (
        "yaw://chunk/abc",
        "yaw://chunk/0",
        "yaw://chunk/-1",
        "yaw://chunk/00",
        "yaw://chunk/01",
        "yaw://chunk/999999",
        "yaw://wiki/missing-card",
    ):
        with pytest.raises((ResourceError, MCPError)):
            await client.read_resource(bad_uri)


async def _assert_mcp_not_ready_and_busy(client: Client[Any], rt: Any, gate_embed: threading.Event) -> None:
    rt.lifecycle = "starting"
    try:
        with pytest.raises(ToolError) as exc_prec:
            await client.call_tool("recall", {"query": "   "})
        assert str(exc_prec.value).startswith("[INVALID_INPUT]")

        for tool_name, bad_args in (
            ("expand_neighbors", {"chunk_id": 0}),
            ("expand_neighbors", {"chunk_id": -1}),
            ("read_chunk", {"chunk_id": 0}),
            ("read_chunk", {"chunk_id": -1}),
            ("write_wiki", {"slug": "card", "title": "Card", "body": "Body", "source_chunk_ids": [0]}),
        ):
            with pytest.raises(ToolError) as exc_bad_id:
                await client.call_tool(tool_name, bad_args)
            assert not str(exc_bad_id.value).startswith("[NOT_READY]")

        for bad_uri in ("yaw://chunk/0", "yaw://chunk/-1", "yaw://chunk/00"):
            with pytest.raises((ResourceError, MCPError)) as exc_bad_res:
                await client.read_resource(bad_uri)
            assert "[INVALID_INPUT]" in str(exc_bad_res.value)

        st_starting = await _read_json_resource(client, "yaw://status")
        assert st_starting["lifecycle"] == "starting"

        for tool_name, args in (
            ("recall", {"query": "valid"}),
            ("expand_neighbors", {"chunk_id": 1}),
            ("read_chunk", {"chunk_id": 1}),
            ("write_wiki", {"slug": "not-ready", "title": "Title", "body": "Body."}),
        ):
            with pytest.raises(ToolError) as exc_nr:
                await client.call_tool(tool_name, args)
            assert str(exc_nr.value).startswith("[NOT_READY]")
    finally:
        rt.lifecycle = "ready"

    gate_embed.clear()
    busy_task = asyncio.create_task(client.call_tool("recall", {"query": "busy_probe"}))
    await asyncio.sleep(0.1)
    try:
        with pytest.raises(ToolError) as exc_busy:
            await client.call_tool("recall", {"query": "second caller"})
        assert str(exc_busy.value).startswith("[BUSY]")
    finally:
        gate_embed.set()
        await busy_task


async def _assert_mcp_staleness_and_shutdown(client: Client[Any], rt: Any, alpha_file: Path, alpha_cid: int) -> None:
    alpha_file.write_text("# Altered\nDisk text changed after indexing.\n", encoding="utf-8")
    stale_call = await client.call_tool("recall", {"query": "Apollo 11 commander", "top_k": 3})
    stale_rec = _assert_mcp_recall_tool_result(stale_call)
    alpha_hits = [r for r in stale_rec["results"] if r["source"]["rel_posix_path"] == "alpha.md"]
    assert alpha_hits and alpha_hits[0].get("stale") is True and alpha_hits[0]["text"] == ""
    assert "stale_on_disk" not in alpha_hits[0]
    assert alpha_hits[0]["content_hash"] == f"sha256:{hashlib.sha256(b'').hexdigest()}"
    assert "[STALE ON DISK] alpha.md" in getattr(stale_call.content[0], "text", "")

    stale_read = _assert_mcp_recall_tool_result(await client.call_tool("read_chunk", {"chunk_id": alpha_cid}))
    assert stale_read["results"][0].get("stale") is True

    with pytest.raises(ToolError) as exc_cons:
        await client.call_tool("expand_neighbors", {"chunk_id": alpha_cid})
    assert str(exc_cons.value).startswith("[CONSISTENCY_ERROR]")

    with pytest.raises(ToolError) as exc_wiki_cons:
        await client.call_tool(
            "write_wiki",
            {"slug": "stale-ref", "title": "Stale Ref", "body": "Body.", "source_chunk_ids": [alpha_cid]},
        )
    assert str(exc_wiki_cons.value).startswith("[CONSISTENCY_ERROR]")

    close_runtime(rt)
    for tool_name, args in (
        ("recall", {"query": "after close"}),
        ("expand_neighbors", {"chunk_id": alpha_cid}),
        ("read_chunk", {"chunk_id": alpha_cid}),
        ("write_wiki", {"slug": "after-close", "title": "Closed", "body": "Body."}),
    ):
        with pytest.raises(ToolError) as exc_sd:
            await client.call_tool(tool_name, args)
        assert str(exc_sd.value).startswith("[SHUTTING_DOWN]")


def test_four_tools_four_resources_staleness_and_mcp_error_codes(tmp_path: Path) -> None:
    secret_key = "super-secret-mcp-token"
    cfg = _make_cfg(
        tmp_path / "tools",
        wiki_root=tmp_path / "tools" / "wiki",
        api_key=secret_key,
        queue_maxsize=1,
        qa_model="",
    )
    assert cfg.corpus_root is not None and cfg.wiki_root is not None
    cfg.corpus_root.mkdir(parents=True)
    cfg.wiki_root.mkdir(parents=True)

    alpha_file = cfg.corpus_root / "alpha.md"
    alpha_file.write_text(
        "# Apollo 11 Mission\nNeil Armstrong served as the Apollo 11 Lunar Module commander.\n",
        encoding="utf-8",
    )
    multi_text = "\n\n".join(
        f"Stage {k}: " + " ".join(f"Orbital booster valve item {k}_{j}." for j in range(1, 6)) for k in range(1, 8)
    )
    (cfg.corpus_root / "multi.json").write_text(
        json.dumps({"title": "Booster Manual", "text": multi_text}),
        encoding="utf-8",
    )
    (cfg.corpus_root / "booster_backup.md").write_text(
        "# Booster Backup\nOrbital booster backup valve specification for emergency shutoff.\n",
        encoding="utf-8",
    )

    gate_embed = threading.Event()
    gate_embed.set()

    def embed_fn(texts: list[str]) -> np.ndarray:
        if any("busy_probe" in t for t in texts):
            gate_embed.wait(timeout=5.0)
        return _topic_embed(texts)

    rt = start_runtime(cfg, live=False, wait_for_ready=True, embed_fn=embed_fn, lm=None)
    try:
        mcp, _ = create_mcp_app(rt=rt)

        async def _run() -> None:
            async with Client(mcp) as client:
                _first_cid, alpha_cid = await _assert_mcp_happy_tools_and_resources(client, secret_key)
                await _assert_mcp_bad_inputs(client)
                await _assert_mcp_not_ready_and_busy(client, rt, gate_embed)
                await _assert_mcp_staleness_and_shutdown(client, rt, alpha_file, alpha_cid)

        asyncio.run(_run())
    finally:
        gate_embed.set()
        if rt.lifecycle != "stopped":
            close_runtime(rt)

    e_cfg = _make_cfg(tmp_path / "empty_tool", wiki_root=tmp_path / "empty_tool" / "wiki", qa_model="")
    assert e_cfg.corpus_root is not None and e_cfg.wiki_root is not None
    e_cfg.corpus_root.mkdir(parents=True)
    e_cfg.wiki_root.mkdir(parents=True)
    rt_e = start_runtime(e_cfg, live=False, wait_for_ready=True, embed_fn=_topic_embed, lm=None)
    try:
        mcp_e, _ = create_mcp_app(rt=rt_e)

        async def _empty_checks() -> None:
            async with Client(mcp_e) as client:
                empty_call = await client.call_tool("recall", {"query": "Anything?"})
                empty_rec = _assert_mcp_recall_tool_result(empty_call, expected_text="No matching passages found.")
                assert empty_rec == {"results": []}
                assert await _read_json_resource(client, "yaw://wiki") == []
                with pytest.raises(ToolError) as exc_e:
                    await client.call_tool("read_chunk", {"chunk_id": 1})
                assert str(exc_e.value).startswith("[INVALID_INPUT]")

        asyncio.run(_empty_checks())
    finally:
        close_runtime(rt_e)


def test_mcp_write_wiki_resources_expand_read_and_collision_safety(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path / "wiki_mcp", wiki_root=tmp_path / "wiki_mcp" / "wiki", qa_model="")
    assert cfg.corpus_root is not None and cfg.wiki_root is not None
    cfg.corpus_root.mkdir(parents=True)
    cfg.wiki_root.mkdir(parents=True)
    (cfg.corpus_root / "notes.md").write_text(
        "# Corpus Notes\nCorpus notes track helium thruster baseline calibration.\n",
        encoding="utf-8",
    )
    (cfg.corpus_root / "alpha.md").write_text(
        "# Apollo 11\nNeil Armstrong commanded Apollo 11 lunar descent.\n",
        encoding="utf-8",
    )

    rt = start_runtime(cfg, live=False, wait_for_ready=True, embed_fn=_topic_embed, lm=None)
    try:
        mcp, _ = create_mcp_app(rt=rt)

        async def _verify_wiki_flow() -> None:
            async with Client(mcp) as client:
                apollo_rec = _assert_mcp_recall_tool_result(await client.call_tool("recall", {"query": "Apollo 11"}))
                apollo_cid = next(
                    int(r["chunk_id"]) for r in apollo_rec["results"] if r["source"]["rel_posix_path"] == "alpha.md"
                )
                write_res = (
                    await client.call_tool(
                        "write_wiki",
                        {
                            "slug": "notes",
                            "title": "Wiki Notes",
                            "body": "Synthesized tungsten bridge citing [[alpha.md]] and Apollo 11.",
                            "source_chunk_ids": [apollo_cid],
                            "wait_for_sync": True,
                        },
                    )
                ).data
                assert write_res["changed"] is True
                assert write_res["synced"] is True
                assert write_res["card_queued"] is False
                assert write_res["rel_path"] == "wiki:notes.md"

                entries = await _read_json_resource(client, "yaw://wiki")
                assert len(entries) == 1 and entries[0]["slug"] == "notes" and entries[0]["title"] == "Wiki Notes"
                card_md = await _read_text_resource(client, "yaw://wiki/notes")
                assert "# Wiki Notes" in card_md and "tungsten bridge" in card_md and "rel_path: alpha.md" in card_md

                wiki_rec = _assert_mcp_recall_tool_result(
                    await client.call_tool("recall", {"query": "tungsten bridge", "top_k": 5})
                )
                corpus_rec = _assert_mcp_recall_tool_result(
                    await client.call_tool("recall", {"query": "helium thruster", "top_k": 5})
                )
                assert any(r["source"]["rel_posix_path"] == "wiki:notes.md" for r in wiki_rec["results"])
                assert any(r["source"]["rel_posix_path"] == "notes.md" for r in corpus_rec["results"])

                wiki_cid = next(
                    int(r["chunk_id"]) for r in wiki_rec["results"] if r["source"]["rel_posix_path"] == "wiki:notes.md"
                )
                read_wiki = _assert_mcp_recall_tool_result(await client.call_tool("read_chunk", {"chunk_id": wiki_cid}))
                assert read_wiki["results"][0]["source"]["rel_posix_path"] == "wiki:notes.md"
                res_chunk = await _read_json_resource(client, f"yaw://chunk/{wiki_cid}")
                assert res_chunk["results"][0]["chunk_id"] == wiki_cid

                neighbors = _assert_mcp_recall_tool_result(
                    await client.call_tool("expand_neighbors", {"chunk_id": wiki_cid, "top_k": 4})
                )
                assert any(r["source"]["rel_posix_path"] == "alpha.md" for r in neighbors["results"])

                nested_res = (
                    await client.call_tool(
                        "write_wiki",
                        {
                            "slug": "topic/rover-notes",
                            "title": "Rover Notes",
                            "body": "Nested rover battery note citing Apollo 11.",
                            "source_chunk_ids": [apollo_cid],
                            "wait_for_sync": True,
                        },
                    )
                ).data
                assert nested_res["slug"] == "topic/rover-notes"
                assert nested_res["synced"] is True
                nested_md = await _read_text_resource(client, "yaw://wiki/topic/rover-notes")
                assert "# Rover Notes" in nested_md and "Nested rover battery note" in nested_md
                listed_slugs = {entry["slug"] for entry in await _read_json_resource(client, "yaw://wiki")}
                assert listed_slugs == {"notes", "topic/rover-notes"}

                for bad_uri in (
                    "yaw://wiki/../secret",
                    "yaw://wiki/%2e%2e%2fsecret",
                    "yaw://wiki/topic/%2e%2e/secret",
                ):
                    with pytest.raises((ResourceError, MCPError)):
                        await client.read_resource(bad_uri)

                assert cfg.wiki_root is not None
                (cfg.wiki_root / "malformed.md").write_text(
                    "---\nkind: [unterminated\n---\n\n# Bad\n", encoding="utf-8"
                )
                with pytest.raises((ResourceError, MCPError)) as exc_bad_yaml:
                    await client.read_resource("yaw://wiki")
                assert "[CONSISTENCY_ERROR]" in str(exc_bad_yaml.value)

        asyncio.run(_verify_wiki_flow())
    finally:
        close_runtime(rt)


async def _verify_status_nonblocking_under_embed_hold(
    c1: Client[Any],
    c2: Client[Any],
    hc: httpx.AsyncClient,
    port: int,
    hold_started: threading.Event,
    embed_gate: threading.Event,
) -> None:
    embed_gate.clear()
    hold_task = asyncio.create_task(c1.call_tool("recall", {"query": "hold_barrier nonblock"}))
    t0 = time.monotonic()
    while not hold_started.wait(timeout=0.05):
        if time.monotonic() - t0 > 10.0:
            pytest.fail("recall task did not enter embed hold in time")
        await asyncio.sleep(0.02)

    t_check = time.monotonic()
    st_blocked = await _read_json_resource(c2, "yaw://status")
    hz = await hc.get(f"http://127.0.0.1:{port}/healthz")
    rz = await hc.get(f"http://127.0.0.1:{port}/readyz")
    st_http = await hc.get(f"http://127.0.0.1:{port}/status")
    assert time.monotonic() - t_check < 2.0
    assert hz.status_code == 200 and rz.status_code == 200 and st_http.status_code == 200
    assert st_blocked["active_op"] == "recall" and st_http.json()["active_op"] == "recall"

    embed_gate.set()
    hold_done = _assert_mcp_recall_tool_result(await hold_task)
    assert len(hold_done["results"]) >= 1


async def _verify_32_slot_saturation_and_cancel(
    c1: Client[Any],
    c2: Client[Any],
    hc: httpx.AsyncClient,
    port: int,
    embed_gate: threading.Event,
) -> None:
    embed_gate.clear()
    held_tasks = [asyncio.create_task(c1.call_tool("recall", {"query": f"hold_barrier {i}"})) for i in range(32)]
    try:
        sat_deadline = time.monotonic() + 10.0
        while time.monotonic() < sat_deadline:
            if (await _read_json_resource(c2, "yaw://status"))["queue_depth"] == 32:
                break
            await asyncio.sleep(0.02)
        else:
            pytest.fail("Did not reach 32 active recall slots")

        t0 = time.monotonic()
        hz = await hc.get(f"http://127.0.0.1:{port}/healthz")
        rz = await hc.get(f"http://127.0.0.1:{port}/readyz")
        assert time.monotonic() - t0 < 2.0
        assert hz.status_code == 200 and rz.status_code == 200

        with pytest.raises(ToolError) as exc_33:
            await c2.call_tool("recall", {"query": "slot 33"})
        assert str(exc_33.value).startswith("[BUSY]")
    finally:
        embed_gate.set()
        await asyncio.gather(*held_tasks, return_exceptions=True)

    assert (await _read_json_resource(c2, "yaw://status"))["queue_depth"] == 0
    assert len((await c2.call_tool("recall", {"query": "Apollo commander"})).data["results"]) >= 1

    embed_gate.clear()
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(c1.call_tool("recall", {"query": "hold_barrier cancel"}), timeout=0.05)

    with socket.create_connection(("127.0.0.1", port), timeout=1.0) as raw_sock:
        body = b'{"jsonrpc":"2.0","id":9,"method":"tools/list","params":{}}'
        hdr = (
            f"POST /mcp HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
            f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n"
        ).encode("ascii")
        raw_sock.sendall(hdr + body)
    embed_gate.set()

    rel_deadline = time.monotonic() + 10.0
    while time.monotonic() < rel_deadline:
        if (await _read_json_resource(c2, "yaw://status"))["queue_depth"] == 0:
            break
        await asyncio.sleep(0.05)
    else:
        pytest.fail("Query slots leaked after cancellation")


def test_live_tcp_multi_client_nonblocking_status_32_slot_saturation_live_sync_and_cancellation(
    tmp_path: Path,
) -> None:
    embed_gate = threading.Event()
    embed_gate.set()
    hold_started = threading.Event()

    class _TrackingGate(threading.Event):
        @override
        def wait(self, timeout: float | None = None) -> bool:
            hold_started.set()
            return embed_gate.wait(timeout=timeout)

        @override
        def is_set(self) -> bool:
            return embed_gate.is_set()

    stub_srv, _ = _start_openai_stub(
        lambda _p: _tool_reply("done", [{"name": "submit", "args": {"answer": "ok"}}]),
        embed_gate=_TrackingGate(),
        hold_query_marker="hold_barrier",
    )
    port = _free_port()
    cfg = _make_cfg(
        tmp_path / "live_tcp",
        http_host="127.0.0.1",
        http_port=port,
        base_url=f"http://127.0.0.1:{stub_srv.server_port}/v1",
        queue_maxsize=32,
        embed_max_in_flight=32,
        qa_model="",
    )
    assert cfg.corpus_root is not None and cfg.lock_path is not None
    corpus_root = cfg.corpus_root
    corpus_root.mkdir(parents=True)
    (corpus_root / "apollo.md").write_text(
        "# Apollo 11 Mission\nNeil Armstrong was the Apollo 11 Lunar Module commander.\n",
        encoding="utf-8",
    )

    srv_errors: list[BaseException] = []
    t_srv = threading.Thread(target=lambda: serve_http(cfg), daemon=True)
    t_srv.start()

    try:
        _wait_http_ready(port, srv_errors)

        # Second concurrent serve_http on same workspace must fail immediately with BUSY
        with pytest.raises(YawRuntimeError) as exc_dup:
            serve_http(cfg)
        assert exc_dup.value.code == "BUSY"

        async def _tcp_suite() -> None:
            mcp_url = f"http://127.0.0.1:{port}/mcp"
            async with Client(mcp_url) as c1, Client(mcp_url) as c2, httpx.AsyncClient() as hc:
                await _verify_status_nonblocking_under_embed_hold(c1, c2, hc, port, hold_started, embed_gate)
                hold_started.clear()
                await _verify_32_slot_saturation_and_cancel(c1, c2, hc, port, embed_gate)

                async with Client(mcp_url) as c_fresh:
                    assert len((await c_fresh.call_tool("recall", {"query": "Apollo 11"})).data["results"]) >= 1

                (corpus_root / "live_tcp.md").write_text(
                    "# XylophoneZither Bio\nNeil Armstrong played XylophoneZither in Ohio.\n",
                    encoding="utf-8",
                )
                live_deadline = time.monotonic() + 20.0
                while time.monotonic() < live_deadline:
                    hits = (await c2.call_tool("recall", {"query": "XylophoneZither", "top_k": 3})).data["results"]
                    if any(r["source"]["rel_posix_path"] == "live_tcp.md" for r in hits):
                        break
                    await asyncio.sleep(0.1)
                else:
                    pytest.fail("Live added file was not indexed/searchable over TCP")

                # Unauthenticated /_internal/stop rejected with 403
                assert (await hc.post(f"http://127.0.0.1:{port}/_internal/stop")).status_code == 403
                assert (
                    await hc.post(
                        f"http://127.0.0.1:{port}/_internal/stop",
                        headers={"X-Yaw-Control-Token": "wrong"},
                    )
                ).status_code == 403

        asyncio.run(_tcp_suite())
        assert _stop_via_control_token(cfg).status_code == 200
        t_srv.join(timeout=10.0)
        assert not t_srv.is_alive()
        assert srv_errors == []
        assert not control_token_path(cfg).exists()

        lock = filelock.FileLock(str(cfg.lock_path), timeout=0)
        lock.acquire(timeout=0)
        lock.release()
    finally:
        embed_gate.set()
        if t_srv.is_alive() and control_token_path(cfg).is_file():
            _stop_via_control_token(cfg)
            t_srv.join(timeout=5.0)
        stub_srv.shutdown()


def test_async_acall_32_slot_2600ms_hold_keeps_health_fast(tmp_path: Path) -> None:
    stub_srv, stats = _start_openai_stub(
        lambda _p: _tool_reply("done", [{"name": "submit", "args": {"answer": "ok"}}]),
        hold_query_marker="slow_acall",
        embed_delay_s=2.6,
    )
    port_a = _free_port()
    cfg_a = _make_cfg(
        tmp_path / "async_hold",
        http_port=port_a,
        base_url=f"http://127.0.0.1:{stub_srv.server_port}/v1",
        queue_maxsize=32,
        embed_max_in_flight=32,
    )
    assert cfg_a.corpus_root is not None
    cfg_a.corpus_root.mkdir(parents=True)
    (cfg_a.corpus_root / "apollo.md").write_text("# Apollo\nApollo lunar booster valve.\n", encoding="utf-8")

    srv_errors: list[BaseException] = []
    t_srv = threading.Thread(target=lambda: serve_http(cfg_a), daemon=True)
    t_srv.start()
    try:
        _wait_http_ready(port_a, srv_errors)

        async def _run_async_hold() -> None:
            mcp_url = f"http://127.0.0.1:{port_a}/mcp"
            async with Client(mcp_url) as c1, Client(mcp_url) as c2, httpx.AsyncClient() as hc:
                tasks = [asyncio.create_task(c1.call_tool("recall", {"query": f"slow_acall {i}"})) for i in range(32)]
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline:
                    if (await hc.get(f"http://127.0.0.1:{port_a}/status")).json()["queue_depth"] == 32:
                        break
                    await asyncio.sleep(0.02)
                else:
                    pytest.fail("32 async acall slots did not enter")

                t_h = time.monotonic()
                hz = await hc.get(f"http://127.0.0.1:{port_a}/healthz")
                rz = await hc.get(f"http://127.0.0.1:{port_a}/readyz")
                assert time.monotonic() - t_h < 2.0
                assert hz.status_code == 200 and rz.status_code == 200

                with pytest.raises(ToolError) as exc_busy:
                    await c2.call_tool("recall", {"query": "slot 33"})
                assert str(exc_busy.value).startswith("[BUSY]")
                results = await asyncio.gather(*tasks)
                assert len(results) == 32

        asyncio.run(_run_async_hold())
        assert stats["peak"] == 32
        assert _stop_via_control_token(cfg_a).status_code == 200
        t_srv.join(timeout=10.0)
        assert not t_srv.is_alive()
    finally:
        if t_srv.is_alive() and control_token_path(cfg_a).is_file():
            _stop_via_control_token(cfg_a)
            t_srv.join(timeout=5.0)
        stub_srv.shutdown()


def test_sync_embed_gate_32_slot_headroom_clamp_avoids_executor_deadlock(tmp_path: Path) -> None:
    cfg_b = _make_cfg(tmp_path / "sync_clamp", queue_maxsize=32, embed_max_in_flight=32)
    assert cfg_b.corpus_root is not None and cfg_b.db_path is not None
    corpus_root, db_path = cfg_b.corpus_root, cfg_b.db_path
    corpus_root.mkdir(parents=True)
    (corpus_root / "apollo.md").write_text("# Apollo\nApollo lunar booster valve.\n", encoding="utf-8")

    expected_clamp = sync_embed_concurrency_limit(32)
    assert expected_clamp <= 24
    gate = threading.Event()
    gate.set()
    state_lock = threading.Lock()
    counts = {"in_embed": 0, "peak": 0}

    def sync_embed_gate(texts: list[str]) -> np.ndarray:
        if any("sync_barrier" in t for t in texts):
            with state_lock:
                counts["in_embed"] += 1
                counts["peak"] = max(counts["peak"], counts["in_embed"])
            try:
                assert gate.wait(timeout=10.0)
            finally:
                with state_lock:
                    counts["in_embed"] -= 1
        return _topic_embed(texts, dim=16)

    async def _run_sync_clamp() -> None:
        _, app_b = create_mcp_app(
            runtime_lifespan_factory=lambda ready_e: runtime_lifespan(
                cfg_b,
                live=True,
                wait_for_ready=True,
                embed_fn=sync_embed_gate,
                lm=create_lm(cfg_b),
                ready_evt=ready_e,
            )
        )
        port_b = _free_port()
        srv_b = uvicorn.Server(uvicorn.Config(app_b, host="127.0.0.1", port=port_b, log_level="error", lifespan="on"))
        srv_task = asyncio.create_task(srv_b.serve())
        conn_mgr = open_managed_db(db_path)
        try:
            async with httpx.AsyncClient() as hc:
                for _ in range(100):
                    if srv_b.started:
                        break
                    await asyncio.sleep(0.02)
                mcp_url = f"http://127.0.0.1:{port_b}/mcp"
                async with Client(mcp_url) as c1, Client(mcp_url) as c2:
                    gate.clear()
                    held = [
                        asyncio.create_task(c1.call_tool("recall", {"query": f"sync_barrier {i}"})) for i in range(32)
                    ]
                    try:
                        dl = time.monotonic() + 10.0
                        while time.monotonic() < dl:
                            st = (await hc.get(f"http://127.0.0.1:{port_b}/status")).json()
                            if st["queue_depth"] == 32 and counts["in_embed"] == expected_clamp:
                                break
                            await asyncio.sleep(0.02)
                        else:
                            pytest.fail(f"Did not clamp sync embed workers cleanly: {counts}")

                        t_chk = time.monotonic()
                        hz = await hc.get(f"http://127.0.0.1:{port_b}/healthz")
                        rz = await hc.get(f"http://127.0.0.1:{port_b}/readyz")
                        reconciled = await asyncio.to_thread(
                            reconcile_index_tables,
                            conn_mgr,
                            embedding_model=cfg_b.embedding_model,
                            task_mode=cfg_b.embedding_task_mode,
                            corpus_root=corpus_root,
                        )
                        assert time.monotonic() - t_chk < 2.0
                        assert hz.status_code == 200 and rz.status_code == 200 and reconciled is False

                        with pytest.raises(ToolError) as exc_33:
                            await c2.call_tool("recall", {"query": "slot 33"})
                        assert str(exc_33.value).startswith("[BUSY]")
                    finally:
                        gate.set()
                        await asyncio.gather(*held)
        finally:
            conn_mgr.close()
            srv_b.should_exit = True
            await srv_task

    asyncio.run(_run_sync_clamp())
    assert counts["peak"] == expected_clamp


def test_live_post_ready_pipeline_failure_flips_readyz_503_and_not_ready_mcp(tmp_path: Path) -> None:
    stub_srv, _ = _start_openai_stub(lambda _p: _tool_reply("Abstain", [{"name": "submit", "args": {"answer": "ok"}}]))
    port = _free_port()
    cfg = _make_cfg(
        tmp_path / "post_ready_err",
        http_host="127.0.0.1",
        http_port=port,
        base_url=f"http://127.0.0.1:{stub_srv.server_port}/v1",
    )
    assert cfg.corpus_root is not None and cfg.lock_path is not None
    cfg.corpus_root.mkdir(parents=True)
    (cfg.corpus_root / "apollo.md").write_text(
        "# Apollo 11\nNeil Armstrong commanded Apollo 11.\n",
        encoding="utf-8",
    )

    srv_errors: list[BaseException] = []
    t_srv = threading.Thread(target=lambda: serve_http(cfg), daemon=True)
    t_srv.start()

    try:
        _wait_http_ready(port, srv_errors)
        (cfg.corpus_root / "oversize.md").write_text("X" * 300, encoding="utf-8")

        err_deadline = time.monotonic() + 20.0
        while time.monotonic() < err_deadline:
            r = httpx.get(f"http://127.0.0.1:{port}/readyz", timeout=1.0)
            if r.status_code == 503 and r.json() == {"status": "error"}:
                break
            time.sleep(0.1)
        else:
            pytest.fail("readyz did not transition to 503 {'status': 'error'} after post-ready failure")

        async def _verify_mcp_blocked() -> None:
            async with Client(f"http://127.0.0.1:{port}/mcp") as client:
                st = await _read_json_resource(client, "yaw://status")
                assert st["lifecycle"] == "error"
                assert st["provider_ready"] is False
                assert st["sync_state"] == "error"
                assert st["last_error"] is not None

                for tool_name, args in (
                    ("recall", {"query": "Apollo 11"}),
                    ("expand_neighbors", {"chunk_id": 1}),
                    ("read_chunk", {"chunk_id": 1}),
                    ("write_wiki", {"slug": "post-err", "title": "Err", "body": "Body."}),
                ):
                    with pytest.raises(ToolError) as exc_nr:
                        await client.call_tool(tool_name, args)
                    assert str(exc_nr.value).startswith("[NOT_READY]")

        asyncio.run(_verify_mcp_blocked())
        assert _stop_via_control_token(cfg).status_code == 200
        t_srv.join(timeout=10.0)
        assert not t_srv.is_alive()
        assert srv_errors == []
    finally:
        if t_srv.is_alive() and control_token_path(cfg).is_file():
            _stop_via_control_token(cfg)
            t_srv.join(timeout=5.0)
        stub_srv.shutdown()


def test_mcp_recall_tool_result_next_links_sparse_and_byte_budget(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path / "slip_mcp")
    cfg = YawConfig(
        workspace_root=cfg.workspace_root,
        corpus_root=cfg.corpus_root,
        db_path=cfg.db_path,
        lock_path=cfg.lock_path,
        state_dir=cfg.state_dir,
        log_path=cfg.log_path,
        status_path=cfg.status_path,
        compiled_program_path=cfg.compiled_program_path,
        chunk_size=600,
        chunk_overlap=20,
        min_chunk_size=10,
    )
    assert cfg.corpus_root is not None
    cfg.corpus_root.mkdir(parents=True)
    pad = 'Detailed engineering telemetry log with "quoted" fields and newline records.\n'
    (cfg.corpus_root / "source.md").write_text(
        f"# Source Hub\nSource Hub cites Target Node using quantumflux alpha and orbital booster.\n{pad * 2}",
        encoding="utf-8",
    )
    (cfg.corpus_root / "target.md").write_text(
        f"# Target Node\nTarget Node evaluates quantumflux alpha alongside orbital booster calibration.\n{pad * 2}",
        encoding="utf-8",
    )
    for k in range(4):
        (cfg.corpus_root / f"extra_{k}.md").write_text(
            f"# Extra {k}\nExtra {k} logs orbital booster valve baseline {k}.\n{pad}",
            encoding="utf-8",
        )

    titled_payload = build_recall_payload(
        (
            HydratedPassage(
                pointer=ChunkPointer(
                    chunk_id=11,
                    rel_path="titled.md",
                    source_hash="abc",
                    start_line=1,
                    end_line=2,
                    char_start=0,
                    char_end=10,
                    title="Titled Section",
                ),
                score=0.9,
                text="body",
                stale_on_disk=False,
            ),
        )
    )
    assert titled_payload["results"][0]["source"]["title"] == "Titled Section"
    assert "stale" not in titled_payload["results"][0]

    rt = start_runtime(cfg, live=False, wait_for_ready=True, embed_fn=_topic_embed, lm=create_lm(cfg))
    try:
        mcp, _ = create_mcp_app(rt=rt)

        async def _verify() -> None:
            async with Client(mcp) as client:
                q = "orbital booster quantumflux alpha"
                call_res = await client.call_tool("recall", {"query": q, "top_k": 2})
                rec = _assert_mcp_recall_tool_result(call_res)
                rec_text = getattr(call_res.content[0], "text", "")
                assert len(rec["next"]) == 3 and "next: " in rec_text
                kinds = {lnk["kind"] for lnk in rec["links"]}
                assert {"cite", "pmi"} <= kinds and "links: " in rec_text
                for item in rec["results"]:
                    assert "stale" not in item and "stale_on_disk" not in item
                    assert item["source"]["title"] in {"Source Hub", "Target Node", *(f"Extra {k}" for k in range(4))}

                old_json_payload = {
                    "query": q,
                    "top_k": 2,
                    "results": [
                        {
                            "rank": item["rank"],
                            "chunk_id": item["chunk_id"],
                            "score": item["score"],
                            "text": item["text"],
                            "content_hash": item["content_hash"],
                            "stale_on_disk": False,
                            "source": {**item["source"], "title": ""},
                        }
                        for item in rec["results"]
                    ],
                }
                old_bytes = len(json.dumps(old_json_payload).encode("utf-8"))
                new_bytes = len(rec_text.encode("utf-8"))
                assert new_bytes <= 0.85 * old_bytes

        asyncio.run(_verify())
    finally:
        close_runtime(rt)
