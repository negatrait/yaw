"""Unmocked integration tests for YawRuntime lifecycle, live WAL data_version swaps, and FastMCP dependency gate."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import dspy
import fastmcp
import numpy as np
import pytest
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from numpy.typing import NDArray

from yaw.core import YawConfig, YawRuntimeError, write_text_atomic
from yaw.qa import build_qa_agent, make_qa_tools
from yaw.retrieval import build_vector_search, open_readonly_db, resolve_retrieval_params
from yaw.runtime import (
    YawRuntime,
    ask,
    ask_once,
    close_runtime,
    create_lm,
    get_status,
    get_vector_search,
    recall,
    recall_once,
    redact_secrets,
    require_path,
    resolve_program_path,
    serialize_passage,
    signal_stop_event,
    start_runtime,
    wait_ready,
    win32_stop_event_name,
)


def _topic_embed(texts: list[str], *, dim: int = 16) -> NDArray[np.float32]:
    topics = (
        ("orbital", "booster", "valve", "specification", "stage"),
        ("xylophone", "zither", "ohio", "armstrong", "musician"),
        ("apollo", "lunar", "moon", "commander"),
    )
    rows = np.zeros((len(texts), dim), dtype=np.float32)
    for row_idx, text in enumerate(texts):
        low = text.lower()
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        base = (np.frombuffer(digest * 2, dtype=np.uint8)[:dim].astype(np.float32) - 127.5) * 0.05
        for t_idx, kws in enumerate(topics):
            for kw in kws:
                if kw in low:
                    base[t_idx] += 2.0
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
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith(("yaw.daemon", "fastmcp", "optuna", "starlette", "uvicorn"))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert not mod.startswith(("yaw.daemon", "fastmcp", "optuna", "starlette", "uvicorn"))

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
        return _topic_embed(texts, dim=16)

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

        # Empty ready corpus: recall returns (), ask raises EMPTY_INDEX (zero extra embed calls beyond dim probe)
        probe_count = len(embed_calls)
        assert recall(rt, "anything", top_k=5) == ()
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


def test_live_pragma_data_version_swap_maxp_and_unlocked_inference(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    assert cfg.corpus_root is not None
    cfg.corpus_root.mkdir(parents=True)

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

    entered_embed = threading.Event()
    release_embed = threading.Event()
    gate_enabled = threading.Event()

    def gated_embed(texts: list[str]) -> NDArray[np.float32]:
        if gate_enabled.is_set() and any("gateprobe" in t for t in texts):
            entered_embed.set()
            assert release_embed.wait(timeout=5.0)
        return _topic_embed(texts, dim=16)

    rt = start_runtime(cfg, live=True, wait_for_ready=False, embed_fn=gated_embed, lm=create_lm(cfg))
    try:
        assert isinstance(rt, YawRuntime)
        assert wait_ready(rt, timeout=20.0) is True
        vs0 = get_vector_search(rt)
        ver0 = rt.index_data_version
        assert vs0.size >= 7 and ver0 >= 1

        hits = recall(rt, "orbital booster", top_k=2)
        paths = [h.pointer.rel_posix_path for h in hits]
        assert len(hits) == 2
        assert "multi.json" in paths
        assert len({hit.pointer.span_label for hit in hits}) == 2

        # Queries > 1024 bytes succeed cleanly without artificial byte truncation
        long_query = "orbital booster " * 80
        assert len(long_query.encode("utf-8")) > 1024
        assert len(recall(rt, long_query, top_k=2)) == 2

        # Write a new file while live CocoIndex watcher runs and verify PRAGMA data_version auto-swap
        (cfg.corpus_root / "live_new.md").write_text(
            "# XylophoneZither Bio\nNeil Armstrong played XylophoneZither in Ohio.\n",
            encoding="utf-8",
        )
        deadline = time.monotonic() + 20.0
        found_live = False
        while time.monotonic() < deadline:
            res = recall(rt, "XylophoneZither", top_k=3)
            if any(h.pointer.rel_posix_path == "live_new.md" for h in res):
                found_live = True
                break
            time.sleep(0.1)
        assert found_live is True
        assert rt.index_data_version > ver0
        assert get_status(rt).num_chunks > vs0.size

        # Pause inside embed_fn -> verify index_lock is NOT held and get_status returns immediately
        gate_enabled.set()
        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(recall, rt, "gateprobe", top_k=2)
            assert entered_embed.wait(timeout=5.0)
            acquired = rt.index_lock.acquire(blocking=False)
            assert acquired is True
            rt.index_lock.release()
            st = get_status(rt)
            assert st.queue_depth == 1
            assert st.active_op == "recall"
            release_embed.set()
            _ = fut.result(timeout=5.0)
    finally:
        release_embed.set()
        close_runtime(rt)


def test_bounded_semaphore_busy_program_load_redaction_and_win32_stop(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path, queue_maxsize=2, api_key="super-secret-xyz987")
    assert cfg.corpus_root is not None and cfg.lock_path is not None and cfg.status_path is not None
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
        return _topic_embed(texts, dim=16)

    seen_reqs: list[dict[str, Any]] = []

    def chat_responder(_turn_idx: int, payload: dict[str, Any]) -> str:
        seen_reqs.append(payload)
        return (
            "<next_thought>Answer directly</next_thought>\n"
            '<tool_calls>{"tool_calls": [{"name": "submit", "args": {"answer": "XylophoneZither"}}]}</tool_calls>'
        )

    srv, _ = _start_chat_server(chat_responder)
    rt: YawRuntime | None = None
    try:
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
        ser = serialize_passage(1, recalled_for_ser[0])
        assert ser["rank"] == 1
        assert ser["chunk_id"] == int(recalled_for_ser[0].pointer.chunk_id)
        assert ser["content_hash"].startswith("sha256:")
        assert ser["source"]["start_line"] == int(recalled_for_ser[0].pointer.start_line)

        def _dummy_tool(query: str) -> str:
            return query

        opt_agent = build_qa_agent([_dummy_tool], max_iters=cfg.react_max_iters)
        assert opt_agent.react.signature is not None
        opt_agent.react.signature = opt_agent.react.signature.with_instructions("CUSTOM_OPT_INSTR_MARKER")
        opt_prog_path = cfg.workspace_root / ".yaw" / "qa_program.json"
        opt_agent.save(str(opt_prog_path), save_program=False)
        assert resolve_program_path(cfg) == opt_prog_path

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

        # Win32 named stop event signal propagation
        if os.name == "nt":
            assert win32_stop_event_name(cfg.lock_path).startswith("Local\\yaw-stop-")
            assert signal_stop_event(cfg.lock_path) is True
            deadline = time.monotonic() + 2.0
            while not rt.stop_event.is_set() and time.monotonic() < deadline:
                time.sleep(0.02)
            assert rt.stop_event.is_set() is True
    finally:
        release_all.set()
        if rt is not None:
            close_runtime(rt)
        srv.shutdown()


def test_live_initial_failure_records_error_lifecycle_and_redacted_status(tmp_path: Path) -> None:
    secret_tok = "tok-initial-live-secret999"
    cfg = _make_cfg(tmp_path, api_key=secret_tok)
    assert cfg.corpus_root is not None and cfg.status_path is not None
    cfg.corpus_root.mkdir(parents=True)
    (cfg.corpus_root / "boom.md").write_text("# Boom\nTrigger initial live embed failure.\n", encoding="utf-8")

    def failing_embed(texts: list[str]) -> NDArray[np.float32]:
        if any("Trigger initial live" in t for t in texts):
            raise RuntimeError(f"embed backend failed with key={secret_tok}")
        return _topic_embed(texts, dim=16)

    rt = start_runtime(cfg, live=True, wait_for_ready=False, embed_fn=failing_embed, lm=create_lm(cfg))
    try:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if rt.lifecycle == "error":
                break
            time.sleep(0.05)
        assert rt.lifecycle == "error"
        st = get_status(rt)
        assert st.lifecycle == "error"
        assert st.sync_state == "error"
        assert st.provider_ready is False
        assert st.last_error is not None
        assert "RuntimeError" in st.last_error
        assert secret_tok not in st.last_error
        disk_text = cfg.status_path.read_text(encoding="utf-8")
        assert '"lifecycle": "error"' in disk_text
        assert secret_tok not in disk_text
        with pytest.raises(YawRuntimeError) as exc_nr:
            recall(rt, "anything")
        assert exc_nr.value.code == "NOT_READY"
    finally:
        close_runtime(rt)


def test_optuna_best_params_wiring_rank_flip_and_strict_validation(tmp_path: Path) -> None:
    import contextlib

    def rank_embed(texts: list[str]) -> NDArray[np.float32]:
        rows = np.zeros((len(texts), 16), dtype=np.float32)
        for idx, text in enumerate(texts):
            if text.startswith("task:") or "Orbital telemetry" in text:
                rows[idx, 0] = 1.0
            else:
                rows[idx, 1] = 1.0
        return rows

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
        cfg.corpus_root.mkdir(parents=True)
        (cfg.corpus_root / "doc_lex.md").write_text(
            "# Lexical Target\nQuasarPulse99 appears twice QuasarPulse99 in lexical target.\n",
            encoding="utf-8",
        )
        (cfg.corpus_root / "doc_dense.md").write_text(
            "# Dense Target\nOrbital telemetry station report alpha beta gamma.\n",
            encoding="utf-8",
        )

        # Missing params file returns {} cleanly
        assert resolve_retrieval_params(cfg) == {}

        opt_db = cfg.for_optimization(".yaw/optuna_index.db").db_path
        assert opt_db is not None
        params_path = opt_db.parent / "optuna_best_params.json"

        rt = start_runtime(cfg, live=False, wait_for_ready=True, embed_fn=rank_embed, lm=create_lm(cfg))
        try:
            # 1. Lexical-heavy Optuna override (hybrid_alpha=0.05) ranks doc_lex.md first
            write_text_atomic(
                params_path,
                json.dumps({"best_params": {"hybrid_alpha": 0.05, "ppr_alpha": 0.0, "seed_top_k": 5}}),
            )
            assert resolve_retrieval_params(cfg) == {"hybrid_alpha": 0.05, "ppr_alpha": 0.0, "seed_top_k": 5}
            assert recall(rt, "QuasarPulse99", top_k=1)[0].pointer.rel_posix_path == "doc_lex.md"

            with contextlib.closing(open_readonly_db(cfg.db_path)) as ro_conn:
                ro_vs = build_vector_search(ro_conn)
                tools_lex = make_qa_tools(
                    ro_vs, cfg.corpus_root, rank_embed, ro_conn, search_params=resolve_retrieval_params(cfg)
                )
                assert "doc_lex.md" in tools_lex[0].func("QuasarPulse99", top_k=1)

            # 2. Dense-heavy Optuna override (hybrid_alpha=0.95) flips top rank to doc_dense.md
            write_text_atomic(
                params_path,
                json.dumps({"best_params": {"hybrid_alpha": 0.95, "ppr_alpha": 0.0, "seed_top_k": 5}}),
            )
            assert recall(rt, "QuasarPulse99", top_k=1)[0].pointer.rel_posix_path == "doc_dense.md"

            with contextlib.closing(open_readonly_db(cfg.db_path)) as ro_conn:
                ro_vs = build_vector_search(ro_conn)
                tools_dense = make_qa_tools(
                    ro_vs, cfg.corpus_root, rank_embed, ro_conn, search_params=resolve_retrieval_params(cfg)
                )
                assert "doc_dense.md" in tools_dense[0].func("QuasarPulse99", top_k=1)
        finally:
            close_runtime(rt)

        # 3. Standalone recall_once and ask_once (lock-free) honor optuna_best_params.json
        write_text_atomic(
            params_path,
            json.dumps({"best_params": {"hybrid_alpha": 0.05, "ppr_alpha": 0.0, "seed_top_k": 5}}),
        )
        assert recall_once(cfg, "QuasarPulse99", top_k=1)[0].pointer.rel_posix_path == "doc_lex.md"
        qa_lex = ask_once(cfg, "Which doc matches QuasarPulse99?")
        assert qa_lex.passages[0].pointer.rel_posix_path == "doc_lex.md"

        write_text_atomic(
            params_path,
            json.dumps({"best_params": {"hybrid_alpha": 0.95, "ppr_alpha": 0.0, "seed_top_k": 5}}),
        )
        assert recall_once(cfg, "QuasarPulse99", top_k=1)[0].pointer.rel_posix_path == "doc_dense.md"
        qa_dense = ask_once(cfg, "Which doc matches QuasarPulse99?")
        assert qa_dense.passages[0].pointer.rel_posix_path == "doc_dense.md"

        # 4. Corrupt / unknown / boolean / NaN / out-of-range Optuna params fail fast with CONSISTENCY_ERROR
        for bad_raw in (
            "{not valid json",
            json.dumps([1, 2, 3]),
            json.dumps({"best_params": "not-a-dict"}),
            json.dumps({"best_params": {"unknown_key": 0.5}}),
            json.dumps({"best_params": {"hybrid_alpha": True}}),
            json.dumps({"best_params": {"hybrid_alpha": 1.2}}),
            json.dumps({"best_params": {"ppr_alpha": -0.1}}),
            json.dumps({"best_params": {"seed_top_k": 0}}),
            json.dumps({"best_params": {"seed_top_k": 2.5}}),
            '{"best_params": {"hybrid_alpha": NaN}}',
        ):
            write_text_atomic(params_path, bad_raw)
            with pytest.raises(YawRuntimeError) as exc_bad:
                resolve_retrieval_params(cfg)
            assert exc_bad.value.code == "CONSISTENCY_ERROR"
            with pytest.raises(YawRuntimeError) as exc_rec:
                recall_once(cfg, "QuasarPulse99", top_k=1)
            assert exc_rec.value.code == "CONSISTENCY_ERROR"
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
        return _topic_embed(texts, dim=16)

    def chat_responder(_turn_idx: int, payload: dict[str, Any]) -> str:
        _ = payload
        return "Insufficient references in corpus"

    srv, _ = _start_chat_server(chat_responder)
    cfg = YawConfig(
        workspace_root=tmp_path,
        corpus_root=cfg.corpus_root,
        db_path=cfg.db_path,
        lock_path=cfg.lock_path,
        state_dir=cfg.state_dir,
        log_path=cfg.log_path,
        status_path=cfg.status_path,
        compiled_program_path=cfg.compiled_program_path,
        base_url=f"http://127.0.0.1:{srv.server_port}/v1",
        api_key=secret_tok,
        chunk_size=140,
        chunk_overlap=20,
        min_chunk_size=20,
    )
    rt = start_runtime(cfg, live=True, wait_for_ready=True, embed_fn=post_ready_failing_embed, lm=create_lm(cfg))
    try:
        assert get_status(rt).lifecycle == "ready"
        assert get_status(rt).provider_ready is True
        assert recall(rt, "Apollo", top_k=1)

        (cfg.corpus_root / "boom_after.md").write_text(
            "# Trigger\nPOST_READY_EMBED_BOOM breaks live ingestion.\n",
            encoding="utf-8",
        )
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            if rt.pipeline.error is not None:
                break
            time.sleep(0.05)
        else:
            pytest.fail("Live watcher did not capture post-ready pipeline error")

        # _ensure_ready must sync pipeline error -> persist error status & raise NOT_READY even before get_status()
        with pytest.raises(YawRuntimeError) as exc_rec:
            recall(rt, "Apollo")
        assert exc_rec.value.code == "NOT_READY"
        with pytest.raises(YawRuntimeError) as exc_ask:
            ask(rt, "Who commanded Apollo 11?")
        assert exc_ask.value.code == "NOT_READY"

        st = get_status(rt)
        assert st.lifecycle == "error"
        assert st.provider_ready is False
        assert st.sync_state == "error"
        assert st.last_error is not None
        assert "RuntimeError" in st.last_error
        assert secret_tok not in st.last_error
        disk_text = cfg.status_path.read_text(encoding="utf-8")
        assert '"lifecycle": "error"' in disk_text
        assert secret_tok not in disk_text
    finally:
        close_runtime(rt)
        srv.shutdown()
