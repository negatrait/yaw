"""Unmocked integration tests for YawRuntime lifecycle, live WAL data_version swaps, and FastMCP dependency gate."""

from __future__ import annotations

import ast
import asyncio
import contextlib
import hashlib
import json
import os
import sqlite3
import stat
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, override

import dspy
import fastmcp
import filelock
import numpy as np
import pytest
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from numpy.typing import NDArray

from yaw.core import YawConfig, YawRuntimeError
from yaw.pipeline import DspyBatchEmbedder
from yaw.qa import build_qa_agent
from yaw.retrieval import build_retrieval_index, open_readonly_db
from yaw.runtime import (
    YawRuntime,
    aask,
    acquire_daemon_control,
    arecall,
    arecall_payload,
    ask,
    ask_once,
    build_recall_payload,
    close_runtime,
    control_token_path,
    create_lm,
    dump_recall_json,
    format_recall_text,
    get_retrieval_index,
    get_status,
    recall,
    recall_once,
    recall_once_payload,
    redact_secrets,
    reload_retrieval_index,
    require_path,
    resolve_program_path,
    runtime_lifespan,
    snapshot_status,
    start_runtime,
    sync_embed_concurrency_limit,
    wait_ready_async,
)


def _topic_embed(texts: list[str], *, dim: int = 64) -> NDArray[np.float32]:
    topics = (
        ("orbital", "booster", "valve", "specification", "stage"),
        ("xylophone", "zither", "ohio", "armstrong", "musician"),
        ("apollo", "lunar", "moon", "commander"),
    )
    rows = np.zeros((len(texts), dim), dtype=np.float32)
    for row_idx, text in enumerate(texts):
        low = text.lower()
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        repeats = (dim + len(digest) - 1) // len(digest)
        base = (np.frombuffer(digest * repeats, dtype=np.uint8)[:dim].astype(np.float32) - 127.5) * 0.005
        for t_idx, kws in enumerate(topics):
            for kw in kws:
                if kw in low:
                    base[t_idx] += 3.0
        rows[row_idx] = base / max(float(np.linalg.norm(base)), 1e-12)
    return rows


def _make_cfg(root: Path, *, queue_maxsize: int = 32, api_key: str = "test") -> YawConfig:
    return YawConfig(
        workspace_root=root,
        corpus_root=root / "corpus",
        db_path=root / ".yaw" / "yaw.db",
        lock_path=root / ".yaw" / "yaw.db.lock",
        state_dir=root / ".yaw" / "coco_lmdb",
        log_path=root / ".yaw" / "yaw.log",
        status_path=root / ".yaw" / "status.json",
        compiled_program_path=root / ".yaw" / "compiled_program.json",
        chunk_size=140,
        chunk_overlap=20,
        min_chunk_size=20,
        queue_maxsize=queue_maxsize,
        api_key=api_key,
    )


def _start_chat_server(responder: Any) -> tuple[ThreadingHTTPServer, list[dict[str, Any]]]:
    seen: list[dict[str, Any]] = []
    lock = threading.Lock()

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            with lock:
                turn_idx = len(seen)
                seen.append(payload)
            content = responder(turn_idx, payload)
            body = {
                "id": f"chatcmpl-{turn_idx}",
                "object": "chat.completion",
                "created": 1,
                "model": payload.get("model", "gemma-4-e4b"),
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            }
            raw = json.dumps(body).encode("utf-8")
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
    return srv, seen


def test_fastmcp_gate_and_runtime_module_isolation(tmp_path: Path) -> None:
    assert fastmcp.__version__
    mcp = FastMCP("yaw")
    app = mcp.http_app(path="/mcp")
    assert callable(app)
    assert ToolError is not None and ToolAnnotations is not None

    rt_path = Path(__file__).resolve().parent.parent / "yaw" / "runtime.py"
    src = rt_path.read_text(encoding="utf-8")
    low = src.lower()
    assert "manifold" not in low and "hmm" not in low and "hipporag" not in low

    tree = ast.parse(src, filename=str(rt_path))
    class_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    assert class_names == {"YawRuntime"}
    forbidden_prefixes = ("yaw.daemon", "cocoindex", "fastmcp", "openai", "optuna", "starlette", "uvicorn")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith(forbidden_prefixes)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert not mod.startswith(forbidden_prefixes)
    for retired in (
        "win32_stop_event_name",
        "signal_stop_event",
        "PRAGMA data_version",
        "index_data_version",
        "probe_conn",
        "yaw-win32-stop-waiter",
        "yaw-ready-waiter",
        "ThreadPoolExecutor",
    ):
        assert retired not in src

    cfg = _make_cfg(tmp_path)
    lm = create_lm(cfg)
    assert type(lm) is dspy.LM
    assert lm.model == f"yaw-openai/{cfg.qa_model}"
    assert lm.supports_function_calling
    assert lm.model_type == "chat"
    assert lm.cache is False
    assert lm.kwargs.get("temperature") == 1.0


def test_empty_corpus_precedence_filelock_and_lifecycle(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    assert cfg.corpus_root is not None and cfg.status_path is not None
    cfg.corpus_root.mkdir(parents=True)

    embed_calls: list[list[str]] = []

    def tracked_embed(texts: list[str]) -> NDArray[np.float32]:
        embed_calls.append(list(texts))
        return _topic_embed(texts, dim=64)

    with pytest.raises(YawRuntimeError) as exc_live:
        start_runtime(cfg, live=True, wait_for_ready=False, embed_fn=tracked_embed, lm=create_lm(cfg))
    assert exc_live.value.code == "INVALID_INPUT"

    rt = start_runtime(
        cfg,
        live=False,
        wait_for_ready=True,
        embed_fn=tracked_embed,
        lm=create_lm(cfg),
    )
    try:
        # Second runtime on same workspace lock_path must fail immediately with BUSY
        with pytest.raises(YawRuntimeError) as exc_lock:
            start_runtime(cfg, live=False, wait_for_ready=True, embed_fn=tracked_embed, lm=create_lm(cfg))
        assert exc_lock.value.code == "BUSY"

        st = get_status(rt)
        assert st.lifecycle == "ready"
        assert st.provider_ready is True
        assert st.num_chunks == 0
        assert st.pid == os.getpid()
        assert cfg.status_path.is_file()
        disk_st = json.loads(cfg.status_path.read_text(encoding="utf-8"))
        assert disk_st["lifecycle"] == "ready"
        assert disk_st["num_chunks"] == 0

        # Empty ready corpus: recall returns (), arecall_payload returns {"results": []}, ask raises EMPTY_INDEX
        probe_count = len(embed_calls)
        assert recall(rt, "anything", top_k=5) == ()
        assert asyncio.run(arecall_payload(rt, "anything", top_k=5)) == {"results": []}
        assert format_recall_text({"results": []}) == "No matching passages found."
        assert json.loads(dump_recall_json({"results": []})) == {"results": []}
        assert len(embed_calls) == probe_count
        with pytest.raises(YawRuntimeError) as exc_empty:
            ask(rt, "What happened?")
        assert exc_empty.value.code == "EMPTY_INDEX"

        # Validation precedence: INVALID_INPUT wins BEFORE NOT_READY and SHUTTING_DOWN
        rt.lifecycle = "starting"
        bad_queries: list[Any] = ["", "   ", 123]
        for bad_q in bad_queries:
            with pytest.raises(YawRuntimeError) as exc_inp:
                recall(rt, bad_q)
            assert exc_inp.value.code == "INVALID_INPUT"
        bad_top_ks: list[Any] = [True, False, 0, 21, "5", 5.0]
        for bad_k in bad_top_ks:
            with pytest.raises(YawRuntimeError) as exc_k:
                recall(rt, "valid", top_k=bad_k)
            assert exc_k.value.code == "INVALID_INPUT"
        with pytest.raises(YawRuntimeError) as exc_nr:
            recall(rt, "valid query")
        assert exc_nr.value.code == "NOT_READY"
        with pytest.raises(YawRuntimeError) as exc_nr_ask:
            ask(rt, "Valid question?")
        assert exc_nr_ask.value.code == "NOT_READY"
        rt.lifecycle = "ready"
    finally:
        close_runtime(rt)

    assert rt.lifecycle == "stopped"
    with pytest.raises(YawRuntimeError) as exc_inp_stopped:
        ask(rt, "   ")
    assert exc_inp_stopped.value.code == "INVALID_INPUT"
    with pytest.raises(YawRuntimeError) as exc_sd:
        recall(rt, "valid query")
    assert exc_sd.value.code == "SHUTTING_DOWN"

    # Restarting after close_runtime acquires lock cleanly
    rt2 = start_runtime(cfg, live=False, wait_for_ready=True, embed_fn=tracked_embed, lm=create_lm(cfg))
    try:
        assert get_status(rt2).lifecycle == "ready"
    finally:
        close_runtime(rt2)


def test_live_reactive_swap_maxp_idle_noop_and_unlocked_inference(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    assert cfg.corpus_root is not None
    corpus_root = cfg.corpus_root
    corpus_root.mkdir(parents=True)

    multi_text = "\n\n".join(
        f"Stage {k}: " + " ".join(f"Orbital booster valve item {k}_{j}." for j in range(1, 6)) for k in range(1, 8)
    )
    (corpus_root / "multi.json").write_text(
        json.dumps({"title": "Booster Manual", "text": multi_text}),
        encoding="utf-8",
    )
    (corpus_root / "booster_backup.md").write_text(
        "# Booster Backup\nOrbital booster backup valve specification for emergency shutoff.\n",
        encoding="utf-8",
    )

    entered_embed = threading.Event()
    release_embed = threading.Event()
    gate_enabled = threading.Event()

    def gated_embed(texts: list[str]) -> NDArray[np.float32]:
        if gate_enabled.is_set() and any("gateprobe" in t for t in texts):
            entered_embed.set()
            assert release_embed.wait(timeout=5.0)
        return _topic_embed(texts, dim=64)

    threads_before = {t.ident for t in threading.enumerate()}

    async def _run() -> None:
        async with runtime_lifespan(
            cfg, live=True, wait_for_ready=False, embed_fn=gated_embed, lm=create_lm(cfg)
        ) as rt:
            assert isinstance(rt, YawRuntime)
            assert await wait_ready_async(rt, timeout=20.0) is True
            idx0 = get_retrieval_index(rt)
            gen0 = rt.index_generation
            assert idx0.size >= 7 and gen0 >= 1

            hits = await arecall(rt, "orbital booster", top_k=2)
            paths = [h.pointer.rel_posix_path for h in hits]
            assert len(hits) == 2
            assert {"multi.json", "booster_backup.md"} == set(paths)
            assert len({hit.pointer.span_label for hit in hits}) == 2
            assert rt.index_generation == gen0

            long_query = "orbital booster " * 80
            assert len(long_query.encode("utf-8")) > 1024
            assert len(await arecall(rt, long_query, top_k=2)) == 2
            assert rt.index_generation == gen0

            # Idle / empty / ignored files must NOT rebuild RetrievalIndex
            (corpus_root / "empty.md").write_text("  \n", encoding="utf-8")
            (corpus_root / "ignored.bin").write_bytes(b"\x01\x02")
            await asyncio.sleep(0.4)
            assert rt.index_generation == gen0

            # Sequential post-READY create -> edit -> delete advance index_generation +1 each
            live_file = corpus_root / "live_new.md"
            live_file.write_text(
                "# XylophoneZither Bio\nNeil Armstrong played XylophoneZither in Ohio.\n", encoding="utf-8"
            )
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline:
                if rt.index_generation > gen0:
                    break
                await asyncio.sleep(0.05)
            assert rt.index_generation == gen0 + 1
            res = await arecall(rt, "XylophoneZither", top_k=3)
            assert any(h.pointer.rel_posix_path == "live_new.md" for h in res)
            assert get_status(rt).num_chunks > idx0.size
            assert get_retrieval_index(rt).size == get_status(rt).num_chunks

            gen1 = rt.index_generation
            live_file.write_text("# XylophoneZither Bio\nNeil Armstrong mastered XylophoneZither.\n", encoding="utf-8")
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline and rt.index_generation == gen1:
                await asyncio.sleep(0.05)
            assert rt.index_generation == gen1 + 1

            gen2 = rt.index_generation
            live_file.unlink()
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline and rt.index_generation == gen2:
                await asyncio.sleep(0.05)
            assert rt.index_generation == gen2 + 1

            # Pause inside embed_fn -> verify index_lock is NOT held and snapshot_status returns immediately
            gate_enabled.set()
            task = asyncio.create_task(arecall(rt, "gateprobe", top_k=2))
            try:
                assert await asyncio.to_thread(entered_embed.wait, 5.0)
                acquired = rt.index_lock.acquire(blocking=False)
                assert acquired is True
                rt.index_lock.release()
                st = snapshot_status(rt)
                assert st.queue_depth == 1
                assert st.active_op == "recall"
            finally:
                release_embed.set()
                _ = await task

    try:
        asyncio.run(_run())
    finally:
        release_embed.set()

    leaked = [
        t.name
        for t in threading.enumerate()
        if t.ident not in threads_before and (not t.daemon or t.name.startswith("yaw-"))
    ]
    assert leaked == []


def test_bounded_semaphore_busy_program_load_redaction_and_control_token(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path, queue_maxsize=2, api_key="super-secret-xyz987")
    assert cfg.corpus_root is not None and cfg.db_path is not None and cfg.lock_path is not None
    assert cfg.status_path is not None
    cfg.corpus_root.mkdir(parents=True)
    (cfg.corpus_root / "doc.md").write_text(
        "# Xylophone Fact\nNeil Armstrong played the XylophoneZither in Ohio.\n",
        encoding="utf-8",
    )

    two_inside = threading.Event()
    release_all = threading.Event()
    hold_enabled = threading.Event()
    in_embed = 0
    in_lock = threading.Lock()

    def slow_embed(texts: list[str]) -> NDArray[np.float32]:
        nonlocal in_embed
        if hold_enabled.is_set() and any("saturate" in t for t in texts):
            with in_lock:
                in_embed += 1
                if in_embed == 2:
                    two_inside.set()
            assert release_all.wait(timeout=5.0)
            with in_lock:
                in_embed -= 1
        return _topic_embed(texts, dim=64)

    seen_reqs: list[dict[str, Any]] = []

    def chat_responder(_turn_idx: int, payload: dict[str, Any]) -> str:
        seen_reqs.append(payload)
        return (
            "<next_thought>Answer directly</next_thought>\n"
            '<tool_calls>{"tool_calls": [{"name": "submit", "args": {"answer": "XylophoneZither"}}]}</tool_calls>'
        )

    srv, _ = _start_chat_server(chat_responder)
    with contextlib.ExitStack() as stack:
        stack.callback(srv.shutdown)
        lm = dspy.LM(
            model="openai/gemma-4-e4b",
            model_type="chat",
            api_base=f"http://127.0.0.1:{srv.server_port}/v1",
            api_key="test",
            temperature=1.0,
            cache=False,
            num_retries=0,
        )
        rt = start_runtime(cfg, live=False, wait_for_ready=True, embed_fn=slow_embed, lm=lm)
        stack.callback(close_runtime, rt)
        stack.callback(release_all.set)

        # Saturate queue_maxsize=2 -> 3rd concurrent recall and ask fail immediately with BUSY
        hold_enabled.set()
        with ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(recall, rt, "saturate one", top_k=1)
            f2 = pool.submit(recall, rt, "saturate two", top_k=1)
            assert two_inside.wait(timeout=5.0)

            with pytest.raises(YawRuntimeError) as exc_busy_rec:
                recall(rt, "third caller")
            assert exc_busy_rec.value.code == "BUSY"

            with pytest.raises(YawRuntimeError) as exc_busy_ask:
                ask(rt, "Third QA caller?")
            assert exc_busy_ask.value.code == "BUSY"

            release_all.set()
            _ = f1.result(timeout=5.0), f2.result(timeout=5.0)
        hold_enabled.clear()

        # Verify default queue_maxsize == 32 and program loading precedence
        assert YawConfig(workspace_root=tmp_path).queue_maxsize == 32
        with pytest.raises(YawRuntimeError) as exc_req:
            require_path(None, "db_path")
        assert exc_req.value.code == "INVALID_INPUT"
        assert require_path(cfg.db_path, "db_path") == cfg.db_path.resolve()
        assert resolve_program_path(cfg) is None

        recalled_for_ser = recall(rt, "Which valve controls booster separation?", top_k=1)
        assert len(recalled_for_ser) == 1
        slip_payload = build_recall_payload(recalled_for_ser)
        ser = slip_payload["results"][0]
        assert ser["rank"] == 1
        assert ser["chunk_id"] == int(recalled_for_ser[0].pointer.chunk_id)
        assert ser["content_hash"].startswith("sha256:")
        assert ser["source"]["start_line"] == int(recalled_for_ser[0].pointer.start_line)
        assert "stale" not in ser and "stale_on_disk" not in ser
        assert json.loads(dump_recall_json(slip_payload)) == slip_payload
        assert f"[chunk_id={ser['chunk_id']}]" in format_recall_text(slip_payload)

        def _dummy_tool(query: str) -> str:
            return query

        opt_agent = build_qa_agent([_dummy_tool], max_iters=cfg.react_max_iters)
        assert opt_agent.react.signature is not None
        opt_agent.react.signature = opt_agent.react.signature.with_instructions("CUSTOM_OPT_INSTR_MARKER")
        opt_prog_path = cfg.workspace_root / ".yaw" / "qa_program.json"
        opt_agent.save(str(opt_prog_path), save_program=False)
        assert resolve_program_path(cfg) == opt_prog_path

        alt_dir = cfg.workspace_root / "custom_rt"
        alt_dir.mkdir()
        alt_prog = alt_dir / "qa_program.json"
        opt_agent.save(str(alt_prog), save_program=False)
        alt_cfg = replace(cfg, compiled_program_path=None, db_path=alt_dir / "yaw.db")
        assert resolve_program_path(alt_cfg) == alt_prog

        qa_res_opt = ask(rt, "What instrument?")
        assert qa_res_opt.answer == "XylophoneZither"
        assert "CUSTOM_OPT_INSTR_MARKER" in json.dumps(seen_reqs[-1]["messages"])

        assert cfg.compiled_program_path is not None
        comp_agent = build_qa_agent([_dummy_tool], max_iters=cfg.react_max_iters)
        assert comp_agent.react.signature is not None
        comp_agent.react.signature = comp_agent.react.signature.with_instructions("CUSTOM_COMPILED_MARKER")
        comp_agent.save(str(cfg.compiled_program_path), save_program=False)
        assert resolve_program_path(cfg) == cfg.compiled_program_path

        qa_res_comp = ask(rt, "What instrument?")
        assert qa_res_comp.answer == "XylophoneZither"
        assert "CUSTOM_COMPILED_MARKER" in json.dumps(seen_reqs[-1]["messages"])

        # Secret redaction in get_status and persisted status.json
        raw_leak = (
            f"key={cfg.api_key} auth=Bearer tok_abc123 oai=sk-live987654321 "
            "url=http://user:secretpass@127.0.0.1:8888/v1"
        )
        assert "super-secret-xyz987" not in redact_secrets(raw_leak, cfg.api_key)
        rt.last_error = raw_leak
        redacted_st = get_status(rt)
        disk_text = cfg.status_path.read_text(encoding="utf-8")
        for leaked in ("super-secret-xyz987", "tok_abc123", "sk-live987654321", "secretpass"):
            assert redacted_st.last_error is not None and leaked not in redacted_st.last_error
            assert leaked not in disk_text

        # Verify headroom clamp calculation and acquire_daemon_control / owns_file_lock handoff
        expected_cap = max(1, min(32, min(32, (os.process_cpu_count() or 1) + 4) - 4))
        assert sync_embed_concurrency_limit(32) == expected_cap

    ctrl_cfg = _make_cfg(tmp_path / "ctrl_lock")
    assert ctrl_cfg.corpus_root is not None and ctrl_cfg.lock_path is not None
    ctrl_cfg.corpus_root.mkdir(parents=True)
    tok_path = control_token_path(ctrl_cfg)
    with acquire_daemon_control(ctrl_cfg) as (held_lock, token):
        assert len(token) == 64
        assert tok_path.is_file() and tok_path.read_text(encoding="ascii") == token
        if os.name != "nt":
            assert stat.S_IMODE(tok_path.stat().st_mode) == 0o600
        with pytest.raises(YawRuntimeError) as exc_dup, acquire_daemon_control(ctrl_cfg):
            pass
        assert exc_dup.value.code == "BUSY"
        assert tok_path.read_text(encoding="ascii") == token

        rt_owned = start_runtime(
            ctrl_cfg,
            live=False,
            wait_for_ready=True,
            embed_fn=slow_embed,
            lm=create_lm(ctrl_cfg),
            file_lock=held_lock,
        )
        assert rt_owned.owns_file_lock is False
        close_runtime(rt_owned)
        assert held_lock.is_locked is True
        assert tok_path.is_file()
    assert not tok_path.exists()
    probe = filelock.FileLock(str(ctrl_cfg.lock_path), timeout=0)
    probe.acquire(timeout=0)
    probe.release()


def test_live_initial_failure_records_error_lifecycle_and_redacted_status(tmp_path: Path) -> None:
    secret_tok = "tok-initial-live-secret999"
    cfg = _make_cfg(tmp_path, api_key=secret_tok)
    assert cfg.corpus_root is not None and cfg.status_path is not None
    status_path = cfg.status_path
    cfg.corpus_root.mkdir(parents=True)
    (cfg.corpus_root / "boom.md").write_text("# Boom\nTrigger initial live embed failure.\n", encoding="utf-8")

    def failing_embed(texts: list[str]) -> NDArray[np.float32]:
        if any("Trigger initial live" in t for t in texts):
            raise RuntimeError(f"embed backend failed with key={secret_tok}")
        return _topic_embed(texts, dim=64)

    async def _run() -> None:
        async with runtime_lifespan(
            cfg, live=True, wait_for_ready=False, embed_fn=failing_embed, lm=create_lm(cfg)
        ) as rt:
            assert rt.ready_evt is not None
            await asyncio.wait_for(rt.ready_evt.wait(), timeout=10.0)
            assert rt.lifecycle == "error"
            st = get_status(rt)
            assert st.lifecycle == "error"
            assert st.sync_state == "error"
            assert st.provider_ready is False
            assert st.last_error is not None
            assert "RuntimeError" in st.last_error
            assert secret_tok not in st.last_error
            assert rt.last_error is not None and secret_tok not in rt.last_error
            assert "[REDACTED]" in rt.last_error
            rt.last_error = "custom recorded redacted message"
            _ = snapshot_status(rt)
            assert rt.last_error == "custom recorded redacted message"
            rt.last_error = None
            synced_st = snapshot_status(rt)
            assert synced_st.last_error is not None and secret_tok not in synced_st.last_error
            assert rt.last_error is not None and secret_tok not in rt.last_error
            assert "[REDACTED]" in rt.last_error
            disk_text = status_path.read_text(encoding="utf-8")
            assert '"lifecycle": "error"' in disk_text
            assert secret_tok not in disk_text
            with pytest.raises(YawRuntimeError) as exc_nr:
                await arecall(rt, "anything")
            assert exc_nr.value.code == "NOT_READY"

    asyncio.run(_run())


def test_live_initial_base_exception_transitions_to_error_and_redacts_status(tmp_path: Path) -> None:
    class _SimulatedRustPanic(BaseException):
        pass

    secret_tok = "tok-initial-panic-secret888"
    cfg = _make_cfg(tmp_path, api_key=secret_tok)
    assert cfg.corpus_root is not None and cfg.status_path is not None
    status_path = cfg.status_path
    cfg.corpus_root.mkdir(parents=True)
    (cfg.corpus_root / "panic.md").write_text("# Panic\nTrigger initial live non-Exception panic.\n", encoding="utf-8")

    def panicking_embed(texts: list[str]) -> NDArray[np.float32]:
        if any("Trigger initial live non-Exception panic" in t for t in texts):
            raise _SimulatedRustPanic(f"rust panic with key={secret_tok}")
        return _topic_embed(texts, dim=64)

    async def _run() -> None:
        async with runtime_lifespan(
            cfg, live=True, wait_for_ready=False, embed_fn=panicking_embed, lm=create_lm(cfg)
        ) as rt:
            assert rt.ready_evt is not None
            await asyncio.wait_for(rt.ready_evt.wait(), timeout=10.0)
            assert rt.lifecycle == "error"
            assert isinstance(rt.pipeline.error, RuntimeError)
            assert isinstance(rt.pipeline.error.__cause__, BaseException)
            st = get_status(rt)
            assert st.lifecycle == "error"
            assert st.sync_state == "error"
            assert st.provider_ready is False
            assert st.last_error is not None
            assert "RuntimeError" in st.last_error
            assert "_SimulatedRustPanic" in st.last_error
            assert secret_tok not in st.last_error
            assert rt.last_error is not None and secret_tok not in rt.last_error
            disk_text = status_path.read_text(encoding="utf-8")
            assert '"lifecycle": "error"' in disk_text
            assert secret_tok not in disk_text
            with pytest.raises(YawRuntimeError) as exc_nr:
                await arecall(rt, "anything")
            assert exc_nr.value.code == "NOT_READY"

    asyncio.run(_run())


def test_live_initial_http_500_transitions_to_error_and_redacts_status(tmp_path: Path) -> None:
    secret_tok = "tok-initial-http500-secret555"

    class _FailingEmbedHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 0:
                self.rfile.read(length)
            body = json.dumps({"error": {"message": f"upstream 500 for key={secret_tok}"}}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        @override
        def log_message(self, format: str, *args: Any) -> None:
            _ = (format, args)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), _FailingEmbedHandler)
    srv_thread = threading.Thread(target=srv.serve_forever, daemon=True)
    srv_thread.start()
    try:
        cfg = replace(_make_cfg(tmp_path, api_key=secret_tok), base_url=f"http://127.0.0.1:{srv.server_port}/v1")
        assert cfg.corpus_root is not None and cfg.status_path is not None
        status_path = cfg.status_path
        cfg.corpus_root.mkdir(parents=True)
        (cfg.corpus_root / "http500.md").write_text(
            "# HTTP 500\nTrigger real DSPy HTTP 500 embedding failure.\n",
            encoding="utf-8",
        )

        async def _run() -> None:
            async with runtime_lifespan(cfg, live=True, wait_for_ready=False, embed_fn=None, lm=create_lm(cfg)) as rt:
                assert rt.ready_evt is not None
                await asyncio.wait_for(rt.ready_evt.wait(), timeout=15.0)
                assert rt.lifecycle == "error"
                assert isinstance(rt.pipeline.error, RuntimeError)
                assert rt.pipeline.error.__cause__ is not None
                assert type(rt.pipeline.error.__cause__).__name__ == "InternalServerError"
                st = get_status(rt)
                assert st.lifecycle == "error"
                assert st.sync_state == "error"
                assert st.provider_ready is False
                assert st.last_error is not None
                assert "RuntimeError" in st.last_error
                assert "InternalServerError" in st.last_error
                assert secret_tok not in st.last_error
                assert rt.last_error is not None and secret_tok not in rt.last_error
                disk_text = status_path.read_text(encoding="utf-8")
                assert '"lifecycle": "error"' in disk_text
                assert secret_tok not in disk_text
                with pytest.raises(YawRuntimeError) as exc_nr:
                    await arecall(rt, "anything")
                assert exc_nr.value.code == "NOT_READY"

        asyncio.run(_run())
    finally:
        srv.shutdown()
        srv.server_close()
        srv_thread.join(timeout=5.0)


def test_live_post_ready_index_not_ready_fallback_and_consistency_guard(tmp_path: Path) -> None:
    def rank_embed(texts: list[str]) -> NDArray[np.float32]:
        return _topic_embed(texts, dim=64)

    class _OpenAIHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if self.path.endswith("/embeddings"):
                raw_inp = payload.get("input", [])
                texts = [raw_inp] if isinstance(raw_inp, str) else [str(x) for x in raw_inp]
                vecs = rank_embed(texts)
                body = {
                    "object": "list",
                    "data": [
                        {"object": "embedding", "index": i, "embedding": vecs[i].tolist()} for i in range(len(texts))
                    ],
                    "model": payload.get("model", "embeddinggemma-2"),
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                }
            else:
                msgs = payload.get("messages", [])
                blob = json.dumps(msgs)
                if "tool" not in [m.get("role") for m in msgs]:
                    msg = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_search",
                                "type": "function",
                                "function": {
                                    "name": "search_passages",
                                    "arguments": json.dumps({"query": "QuasarPulse99", "top_k": 1}),
                                },
                            }
                        ],
                    }
                else:
                    import re as _re

                    cids = _re.findall(r"\[chunk_id=(\d+)\]", blob)
                    msg = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_sub",
                                "type": "function",
                                "function": {
                                    "name": "submit",
                                    "arguments": json.dumps({"answer": f"Found [chunk_id={cids[0]}]"}),
                                },
                            }
                        ],
                    }
                body = {
                    "id": "chatcmpl-1",
                    "object": "chat.completion",
                    "created": 1,
                    "model": payload.get("model", "gemma-4-e4b"),
                    "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                }
            raw = json.dumps(body).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        @override
        def log_message(self, format: str, *args: Any) -> None:
            _ = (format, args)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), _OpenAIHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        cfg = YawConfig(
            workspace_root=tmp_path,
            corpus_root=tmp_path / "corpus",
            db_path=tmp_path / ".yaw" / "yaw.db",
            lock_path=tmp_path / ".yaw" / "yaw.db.lock",
            state_dir=tmp_path / ".yaw" / "coco_lmdb",
            log_path=tmp_path / ".yaw" / "yaw.log",
            status_path=tmp_path / ".yaw" / "status.json",
            compiled_program_path=tmp_path / ".yaw" / "compiled_program.json",
            base_url=f"http://127.0.0.1:{srv.server_port}/v1",
            chunk_size=240,
            chunk_overlap=0,
            min_chunk_size=20,
        )
        assert cfg.corpus_root is not None and cfg.db_path is not None
        db_path = cfg.db_path
        cfg.corpus_root.mkdir(parents=True)
        (cfg.corpus_root / "doc_lex.md").write_text(
            "# Lexical Target\nQuasarPulse99 appears in lexical target doc_lex.\n",
            encoding="utf-8",
        )
        (cfg.corpus_root / "doc_dense.md").write_text(
            "# Dense Target\nOrbital telemetry station report alpha beta gamma.\n",
            encoding="utf-8",
        )

        with pytest.raises(YawRuntimeError) as exc_empty:
            recall_once(cfg, "QuasarPulse99", top_k=1)
        assert exc_empty.value.code == "EMPTY_INDEX"

        async def _run() -> None:
            async with runtime_lifespan(
                cfg, live=True, wait_for_ready=True, embed_fn=rank_embed, lm=create_lm(cfg)
            ) as rt:
                idx0 = get_retrieval_index(rt)
                gen0 = rt.index_generation
                assert idx0.size >= 2 and gen0 >= 1

                assert recall_once(cfg, "QuasarPulse99", top_k=1)[0].pointer.rel_posix_path == "doc_lex.md"
                once_payload = recall_once_payload(cfg, "QuasarPulse99", top_k=1)
                assert once_payload["results"][0]["source"]["rel_posix_path"] == "doc_lex.md"
                assert isinstance(once_payload["log_bf"], float) and isinstance(once_payload["coverage"], float)
                assert "certainty: log_bf=" in format_recall_text(once_payload)
                qa_res = ask_once(cfg, "Which doc matches QuasarPulse99?")
                assert qa_res.passages[0].pointer.rel_posix_path == "doc_lex.md"

                with contextlib.closing(open_readonly_db(db_path)) as ro_conn:
                    _ = build_retrieval_index(
                        ro_conn,
                        expected_embedding_model=cfg.embedding_model,
                        expected_task_mode=cfg.embedding_task_mode,
                    )
                with pytest.raises(YawRuntimeError) as exc_mm:
                    recall_once(replace(cfg, embedding_model="mismatched-model"), "QuasarPulse99", top_k=1)
                assert exc_mm.value.code == "INDEX_NOT_READY"

                # 1. Transient post-ready INDEX_NOT_READY falls back to prior idx0 without advancing index_generation
                with contextlib.closing(sqlite3.connect(db_path)) as rw_conn:
                    orig_vsha = str(rw_conn.execute("SELECT vec_sha256 FROM rabitq_index WHERE id = 1").fetchone()[0])
                    rw_conn.execute("UPDATE rabitq_index SET vec_sha256 = 'transient_stale_sha' WHERE id = 1")
                    rw_conn.commit()
                fallback_idx = reload_retrieval_index(rt)
                assert fallback_idx is idx0
                assert rt.index_generation == gen0

                # 2. Restore fingerprint and introduce CONSISTENCY_ERROR -> NEVER swallowed by reload_retrieval_index
                with contextlib.closing(sqlite3.connect(db_path)) as rw_conn:
                    rw_conn.execute(
                        "UPDATE rabitq_index SET vec_sha256 = ?, codes_blob = X'0001' WHERE id = 1",
                        (orig_vsha,),
                    )
                    rw_conn.commit()
                with pytest.raises(YawRuntimeError) as exc_cons:
                    reload_retrieval_index(rt)
                assert exc_cons.value.code == "CONSISTENCY_ERROR"

        asyncio.run(_run())
    finally:
        srv.shutdown()


def test_live_post_ready_pipeline_failure_transitions_to_error_and_blocks_queries(tmp_path: Path) -> None:
    secret_tok = "tok-post-ready-secret777"
    cfg = _make_cfg(tmp_path, api_key=secret_tok)
    assert cfg.corpus_root is not None and cfg.status_path is not None
    cfg.corpus_root.mkdir(parents=True)
    (cfg.corpus_root / "apollo.md").write_text(
        "# Apollo 11\nNeil Armstrong commanded Apollo 11 to the Moon.\n",
        encoding="utf-8",
    )

    def post_ready_failing_embed(texts: list[str]) -> NDArray[np.float32]:
        if any("POST_READY_EMBED_BOOM" in t for t in texts):
            raise RuntimeError(f"post-ready embed boom key={secret_tok}")
        return _topic_embed(texts, dim=64)

    srv, _ = _start_chat_server(lambda _t, _p: "Insufficient references in corpus")
    cfg = replace(cfg, base_url=f"http://127.0.0.1:{srv.server_port}/v1", api_key=secret_tok)
    assert cfg.corpus_root is not None and cfg.status_path is not None
    corpus_root, status_path = cfg.corpus_root, cfg.status_path

    async def _run() -> None:
        async with runtime_lifespan(
            cfg, live=True, wait_for_ready=True, embed_fn=post_ready_failing_embed, lm=create_lm(cfg)
        ) as rt:
            assert get_status(rt).lifecycle == "ready"
            assert get_status(rt).provider_ready is True
            assert await arecall(rt, "Apollo", top_k=1)

            (corpus_root / "boom_after.md").write_text(
                "# Trigger\nPOST_READY_EMBED_BOOM breaks live ingestion.\n",
                encoding="utf-8",
            )
            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline and rt.pipeline.error is None:
                await asyncio.sleep(0.05)
            assert rt.pipeline.error is not None

            with pytest.raises(YawRuntimeError) as exc_rec:
                await arecall(rt, "Apollo")
            assert exc_rec.value.code == "NOT_READY"
            with pytest.raises(YawRuntimeError) as exc_ask:
                await aask(rt, "Who commanded Apollo 11?")
            assert exc_ask.value.code == "NOT_READY"

            st = get_status(rt)
            assert st.lifecycle == "error"
            assert st.provider_ready is False
            assert st.sync_state == "error"
            assert st.last_error is not None
            assert "RuntimeError" in st.last_error
            assert secret_tok not in st.last_error
            assert rt.last_error is not None and secret_tok not in rt.last_error
            assert "[REDACTED]" in rt.last_error
            rt.last_error = "custom recorded redacted message"
            _ = snapshot_status(rt)
            assert rt.last_error == "custom recorded redacted message"
            rt.last_error = None
            synced_st = snapshot_status(rt)
            assert synced_st.last_error is not None and secret_tok not in synced_st.last_error
            assert rt.last_error is not None and secret_tok not in rt.last_error
            assert "[REDACTED]" in rt.last_error
            disk_text = status_path.read_text(encoding="utf-8")
            assert '"lifecycle": "error"' in disk_text
            assert secret_tok not in disk_text

    try:
        asyncio.run(_run())
    finally:
        srv.shutdown()


def test_arecall_awaits_dspy_batch_embedder_acall_under_l3_semaphores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _AcEmbedHandler(BaseHTTPRequestHandler):
        @override
        def log_message(self, format: str, *args: Any) -> None:
            _ = (format, args)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            raw_in = payload["input"]
            texts = [raw_in] if isinstance(raw_in, str) else [str(x) for x in raw_in]
            vecs = _topic_embed(texts, dim=16)
            body = json.dumps(
                {
                    "object": "list",
                    "data": [
                        {"object": "embedding", "index": i, "embedding": vecs[i].tolist()} for i in range(len(texts))
                    ],
                    "model": str(payload.get("model", "")),
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), _AcEmbedHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        cfg = replace(_make_cfg(tmp_path), base_url=f"http://127.0.0.1:{srv.server_port}/v1")
        assert cfg.corpus_root is not None
        cfg.corpus_root.mkdir(parents=True)
        (cfg.corpus_root / "doc.md").write_text("# Doc\nOrbital booster valve note.\n", encoding="utf-8")

        aclose_calls = 0

        async def _run() -> None:
            nonlocal aclose_calls
            async with runtime_lifespan(cfg, live=False, wait_for_ready=True, lm=create_lm(cfg)) as rt:
                assert isinstance(rt.embed_fn, DspyBatchEmbedder)
                assert rt.embed_uses_async is True
                acall_calls: list[list[str]] = []
                orig_acall = rt.embed_fn.acall
                orig_aclose = rt.embed_fn.aclose

                async def _spy_acall(texts: list[str]) -> NDArray[np.float32]:
                    acall_calls.append(list(texts))
                    return await orig_acall(texts)

                async def _spy_aclose() -> None:
                    nonlocal aclose_calls
                    aclose_calls += 1
                    await orig_aclose()

                monkeypatch.setattr(rt.embed_fn, "acall", _spy_acall)
                monkeypatch.setattr(rt.embed_fn, "aclose", _spy_aclose)
                monkeypatch.setattr(
                    rt.embed_fn,
                    "__call__",
                    lambda _texts: pytest.fail("arecall must not call sync __call__ on DspyBatchEmbedder"),
                )
                hits = await arecall(rt, "orbital booster", top_k=1)
                assert len(hits) == 1
                payload = await arecall_payload(rt, "orbital booster", top_k=1)
                assert len(payload["results"]) == 1
                assert isinstance(payload["log_bf"], float) and isinstance(payload["coverage"], float)
                assert "certainty: log_bf=" in format_recall_text(payload)
                assert json.loads(dump_recall_json(payload)) == payload
                assert len(acall_calls) == 2

        asyncio.run(_run())
        assert aclose_calls == 1
    finally:
        srv.shutdown()
        srv.server_close()
