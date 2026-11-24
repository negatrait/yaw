"""Unmocked integration tests for FastMCP Streamable HTTP adapter, loopback guards, and U3 lifespan contracts."""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import re
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
from fastmcp.exceptions import ToolError

from yaw.core import YawConfig, YawRuntimeError
from yaw.mcp_server import create_mcp_app, serve_http
from yaw.pipeline import open_managed_db, reconcile_index_tables
from yaw.qa import create_lm
from yaw.runtime import (
    close_runtime,
    control_token_path,
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
) -> YawConfig:
    return YawConfig(
        workspace_root=root,
        corpus_root=root / "corpus",
        db_path=root / ".yaw" / "yaw.db",
        lock_path=root / ".yaw" / "yaw.db.lock",
        state_dir=root / ".yaw" / "coco_lmdb",
        log_path=root / ".yaw" / "yaw.log",
        status_path=root / ".yaw" / "status.json",
        compiled_program_path=root / ".yaw" / "compiled_program.json",
        base_url=base_url,
        api_key=api_key,
        chunk_size=140,
        chunk_overlap=20,
        min_chunk_size=20,
        queue_maxsize=queue_maxsize,
        embed_max_in_flight=embed_max_in_flight,
        http_host=http_host,
        http_port=http_port if http_port is not None else _free_port(),
        http_path=http_path,
    )


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
                assert set(by_name) == {"recall", "qa", "daemon_status"}
                for name, idem in (("recall", True), ("qa", False), ("daemon_status", True)):
                    ann = by_name[name].annotations
                    assert ann is not None
                    assert ann.read_only_hint is True
                    assert ann.destructive_hint is False
                    assert ann.idempotent_hint is idem
                    assert ann.open_world_hint is False
                assert "query" in by_name["recall"].input_schema.get("properties", {})
                assert "question" in by_name["qa"].input_schema.get("properties", {})

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


def test_three_tools_passage_dict_staleness_and_all_seven_error_codes(tmp_path: Path) -> None:
    secret_key = "super-secret-mcp-token"
    cfg = _make_cfg(tmp_path / "tools", api_key=secret_key, queue_maxsize=1)
    assert cfg.corpus_root is not None
    cfg.corpus_root.mkdir(parents=True)

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

    chat_mode = {"mode": "normal"}
    gate_embed = threading.Event()
    gate_embed.set()

    def embed_fn(texts: list[str]) -> np.ndarray:
        if any("busy_probe" in t for t in texts):
            gate_embed.wait(timeout=5.0)
        return _topic_embed(texts)

    def chat_responder(payload: dict[str, Any]) -> str | dict[str, Any]:
        if chat_mode["mode"] == "secret_garbage":
            return f"garbage Bearer tok_secret999 {secret_key} sk-secret123456"
        msgs = payload.get("messages", [])
        has_tool_obs = any(m.get("role") == "tool" for m in msgs)
        msgs_blob = json.dumps(msgs)
        alpha_match = re.search(r"\[chunk_id=(\d+)\]\s+\(alpha\.md:[^)\n]+\)", msgs_blob)
        cids = re.findall(r"\[chunk_id=(\d+)\]", msgs_blob)
        if not has_tool_obs or not cids:
            return _tool_reply(
                "Search Apollo passages first",
                [{"name": "search_passages", "args": {"query": "Apollo 11 commander", "top_k": 3}}],
            )
        chosen = alpha_match.group(1) if alpha_match is not None else cids[0]
        return _tool_reply(
            "Cite retrieved passage",
            [{"name": "submit", "args": {"answer": f"Neil Armstrong [chunk_id={chosen}]"}}],
        )

    srv, _ = _start_openai_stub(chat_responder)
    cfg_lm = _make_cfg(tmp_path / "tools", base_url=f"http://127.0.0.1:{srv.server_port}/v1", api_key=secret_key)
    rt = start_runtime(cfg, live=False, wait_for_ready=True, embed_fn=embed_fn, lm=create_lm(cfg_lm))
    try:
        mcp, _ = create_mcp_app(rt=rt)

        async def _run() -> None:
            async with Client(mcp) as client:
                rec = (await client.call_tool("recall", {"query": "orbital booster", "top_k": 2})).data
                assert rec["query"] == "orbital booster" and rec["top_k"] == 2
                results = rec["results"]
                assert len(results) == 2
                assert any(r["source"]["rel_posix_path"] == "multi.json" for r in results)
                assert len({r["chunk_id"] for r in results}) == 2
                for rank_i, item in enumerate(results, start=1):
                    assert item["rank"] == rank_i
                    assert type(item["chunk_id"]) is int
                    assert isinstance(item["score"], float)
                    assert item["stale_on_disk"] is False
                    assert item["content_hash"] == f"sha256:{hashlib.sha256(item['text'].encode('utf-8')).hexdigest()}"
                    assert set(item["source"]) == {"rel_posix_path", "start_line", "end_line", "title"}

                qa_res = (await client.call_tool("qa", {"question": "Who commanded Apollo 11?"})).data
                assert qa_res["question"] == "Who commanded Apollo 11?"
                assert "Neil Armstrong" in qa_res["answer"]
                assert qa_res["turns"] >= 2
                assert 1 <= len(qa_res["citations"]) <= 5

                st = (await client.call_tool("daemon_status", {})).data
                assert st["lifecycle"] == "ready" and st["num_chunks"] >= 7

                long_q = "orbital booster " * 80
                assert len(long_q.encode("utf-8")) > 1024
                long_rec = (await client.call_tool("recall", {"query": long_q, "top_k": 2})).data
                assert len(long_rec["results"]) == 2

                for tool_name, bad_args in (
                    ("recall", {"query": "ok", "top_k": True}),
                    ("recall", {"query": "ok", "top_k": False}),
                    ("recall", {"query": "ok", "top_k": "5"}),
                    ("recall", {"query": "ok", "top_k": 5.0}),
                    ("recall", {"query": "ok", "top_k": 0}),
                    ("recall", {"query": "ok", "top_k": 21}),
                    ("recall", {"query": ""}),
                    ("recall", {"query": 123}),
                    ("recall", {"query": "ok", "extra": 1}),
                    ("qa", {"question": ""}),
                    ("qa", {"question": 42}),
                    ("daemon_status", {"x": 1}),
                ):
                    with pytest.raises(ToolError):
                        await client.call_tool(tool_name, bad_args)

                for tool_name, ws_args in (("recall", {"query": "   "}), ("qa", {"question": "   "})):
                    with pytest.raises(ToolError) as exc_ws:
                        await client.call_tool(tool_name, ws_args)
                    assert str(exc_ws.value).startswith("[INVALID_INPUT]")

                rt.lifecycle = "starting"
                with pytest.raises(ToolError) as exc_prec:
                    await client.call_tool("recall", {"query": "   "})
                assert str(exc_prec.value).startswith("[INVALID_INPUT]")
                with pytest.raises(ToolError) as exc_nr:
                    await client.call_tool("recall", {"query": "valid"})
                assert str(exc_nr.value).startswith("[NOT_READY]")
                rt.lifecycle = "ready"

                gate_embed.clear()
                busy_task = asyncio.create_task(client.call_tool("recall", {"query": "busy_probe"}))
                await asyncio.sleep(0.1)
                with pytest.raises(ToolError) as exc_busy:
                    await client.call_tool("recall", {"query": "second caller"})
                assert str(exc_busy.value).startswith("[BUSY]")
                gate_embed.set()
                await busy_task

                chat_mode["mode"] = "secret_garbage"
                with pytest.raises(ToolError) as exc_mod:
                    await client.call_tool("qa", {"question": "What triggers garbage?"})
                err_txt = str(exc_mod.value)
                assert err_txt.startswith("[MODEL_ERROR]")
                for secret in (secret_key, "tok_secret999", "sk-secret123456"):
                    assert secret not in err_txt
                chat_mode["mode"] = "normal"

                alpha_file.write_text("# Altered\nDisk text changed after indexing.\n", encoding="utf-8")
                stale_rec = (await client.call_tool("recall", {"query": "Apollo 11 commander", "top_k": 3})).data
                alpha_hits = [r for r in stale_rec["results"] if r["source"]["rel_posix_path"] == "alpha.md"]
                assert alpha_hits and alpha_hits[0]["stale_on_disk"] is True and alpha_hits[0]["text"] == ""
                assert alpha_hits[0]["content_hash"] == f"sha256:{hashlib.sha256(b'').hexdigest()}"

                with pytest.raises(ToolError) as exc_cons:
                    await client.call_tool("qa", {"question": "Who commanded Apollo 11?"})
                assert str(exc_cons.value).startswith("[CONSISTENCY_ERROR]")

                close_runtime(rt)
                with pytest.raises(ToolError) as exc_sd:
                    await client.call_tool("recall", {"query": "after close"})
                assert str(exc_sd.value).startswith("[SHUTTING_DOWN]")

        asyncio.run(_run())
    finally:
        gate_embed.set()
        if rt.lifecycle != "stopped":
            close_runtime(rt)
        srv.shutdown()

    e_cfg = _make_cfg(tmp_path / "empty_tool")
    assert e_cfg.corpus_root is not None
    e_cfg.corpus_root.mkdir(parents=True)
    rt_e = start_runtime(e_cfg, live=False, wait_for_ready=True, embed_fn=_topic_embed, lm=create_lm(e_cfg))
    try:
        mcp_e, _ = create_mcp_app(rt=rt_e)

        async def _empty_qa() -> None:
            async with Client(mcp_e) as client:
                with pytest.raises(ToolError) as exc_e:
                    await client.call_tool("qa", {"question": "Anything?"})
                assert str(exc_e.value).startswith("[EMPTY_INDEX]")

        asyncio.run(_empty_qa())
    finally:
        close_runtime(rt_e)


async def _verify_qa_nonblocking(
    c1: Client[Any],
    c2: Client[Any],
    hc: httpx.AsyncClient,
    port: int,
    qa_started: threading.Event,
    qa_release: threading.Event,
) -> None:
    qa_task = asyncio.create_task(c1.call_tool("qa", {"question": "Who commanded Apollo 11?"}))
    t0 = time.monotonic()
    while not qa_started.wait(timeout=0.05):
        if time.monotonic() - t0 > 10.0:
            pytest.fail("qa task did not enter chat responder in time")
        await asyncio.sleep(0.02)

    t_check = time.monotonic()
    st_blocked = (await c2.call_tool("daemon_status", {})).data
    hz = await hc.get(f"http://127.0.0.1:{port}/healthz")
    rz = await hc.get(f"http://127.0.0.1:{port}/readyz")
    st_http = await hc.get(f"http://127.0.0.1:{port}/status")
    assert time.monotonic() - t_check < 2.0
    assert hz.status_code == 200 and rz.status_code == 200 and st_http.status_code == 200
    assert st_blocked["active_op"] == "qa" and st_http.json()["active_op"] == "qa"

    qa_release.set()
    qa_done = (await qa_task).data
    assert len(qa_done["citations"]) >= 1


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
            if (await c2.call_tool("daemon_status", {})).data["queue_depth"] == 32:
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

    assert (await c2.call_tool("daemon_status", {})).data["queue_depth"] == 0
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
        if (await c2.call_tool("daemon_status", {})).data["queue_depth"] == 0:
            break
        await asyncio.sleep(0.05)
    else:
        pytest.fail("Query slots leaked after cancellation")


def test_live_tcp_multi_client_nonblocking_status_32_slot_saturation_live_sync_and_cancellation(
    tmp_path: Path,
) -> None:
    qa_started = threading.Event()
    qa_release = threading.Event()
    embed_gate = threading.Event()
    embed_gate.set()

    def chat_responder(payload: dict[str, Any]) -> dict[str, Any]:
        msgs = payload.get("messages", [])
        has_tool_obs = any(m.get("role") == "tool" for m in msgs)
        cids = re.findall(r"\[chunk_id=(\d+)\]", json.dumps(msgs))
        if not has_tool_obs or not cids:
            return _tool_reply(
                "Look up Apollo",
                [{"name": "search_passages", "args": {"query": "Apollo 11 commander", "top_k": 2}}],
            )
        qa_started.set()
        qa_release.wait(timeout=15.0)
        return _tool_reply(
            "Submit grounded answer", [{"name": "submit", "args": {"answer": f"Armstrong [chunk_id={cids[0]}]"}}]
        )

    stub_srv, _ = _start_openai_stub(chat_responder, embed_gate=embed_gate, hold_query_marker="hold_barrier")
    port = _free_port()
    cfg = _make_cfg(
        tmp_path / "live_tcp",
        http_host="127.0.0.1",
        http_port=port,
        base_url=f"http://127.0.0.1:{stub_srv.server_port}/v1",
        queue_maxsize=32,
        embed_max_in_flight=32,
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
                await _verify_qa_nonblocking(c1, c2, hc, port, qa_started, qa_release)
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
        qa_release.set()
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
                st = (await client.call_tool("daemon_status", {})).data
                assert st["lifecycle"] == "error"
                assert st["provider_ready"] is False
                assert st["sync_state"] == "error"
                assert st["last_error"] is not None

                with pytest.raises(ToolError) as exc_rec:
                    await client.call_tool("recall", {"query": "Apollo 11"})
                assert str(exc_rec.value).startswith("[NOT_READY]")

                with pytest.raises(ToolError) as exc_qa:
                    await client.call_tool("qa", {"question": "Who commanded Apollo 11?"})
                assert str(exc_qa.value).startswith("[NOT_READY]")

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
