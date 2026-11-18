"""Unmocked integration tests for yaw.cli, yaw.__main__, and real detached daemon lifecycle."""

from __future__ import annotations

import ast
import contextlib
import hashlib
import json
import os
import re
import runpy
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import filelock
import numpy as np
import pytest

from yaw.core import YawConfig, load_config, write_text_atomic
from yaw.datasets import prepare_dataset_bundle
from yaw.pipeline import close_pipeline, make_dspy_embed_fn, start_pipeline, update_pipeline_once
from yaw.qa import build_qa_agent
from yaw.retrieval import build_vector_search, open_readonly_db
from yaw.runtime import signal_stop_event

_CID_IN_TEXT_RE = re.compile(r"\[chunk_id=(\d+)\]")
_FORBIDDEN_WORDS = ("mani" + "fold", "h" + "mm", "hippo" + "rag")
_FORBIDDEN_COMMENTS = ("# " + "noqa", "# " + "type: ignore")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _topic_embed_vec(text: str, *, dim: int = 16) -> np.ndarray:
    topics = (
        ("orbital", "booster", "valve", "specification", "stage"),
        ("apollo", "lunar", "moon", "armstrong", "commander"),
        ("xylophone", "zither", "ohio", "music", "childhood"),
        ("glacier", "fjord", "bergen", "harbor", "nordic"),
    )
    low = text.lower()
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    base = (np.frombuffer(digest * 2, dtype=np.uint8)[:dim].astype(np.float32) - 127.5) * 0.05
    for idx, kws in enumerate(topics):
        for kw in kws:
            if kw in low:
                base[idx] += 2.0
    return base / max(float(np.linalg.norm(base)), 1e-12)


def _xml_tool_reply(thought: str, calls: list[dict[str, Any]]) -> str:
    return f"<next_thought>{thought}</next_thought>\n<tool_calls>{json.dumps({'tool_calls': calls})}</tool_calls>"


def _start_openai_stub(chat_responder: Any = None) -> tuple[ThreadingHTTPServer, dict[str, int]]:
    counts = {"embeddings": 0, "chat": 0}
    lock = threading.Lock()

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if self.path.endswith("/embeddings"):
                with lock:
                    counts["embeddings"] += 1
                raw_inputs = payload.get("input", [])
                texts = raw_inputs if isinstance(raw_inputs, list) else [raw_inputs]
                data = [
                    {"object": "embedding", "index": i, "embedding": _topic_embed_vec(str(t)).tolist()}
                    for i, t in enumerate(texts)
                ]
                out: dict[str, Any] = {
                    "object": "list",
                    "data": data,
                    "model": payload.get("model", "embeddinggemma-2"),
                    "usage": {"prompt_tokens": len(texts), "total_tokens": len(texts)},
                }
                status_code = 200
            else:
                with lock:
                    turn_idx = counts["chat"]
                    counts["chat"] += 1
                if chat_responder is None:
                    status_code, content = 500, "unexpected chat call"
                else:
                    status_code, content = chat_responder(turn_idx, payload)
                if status_code != 200:
                    err_raw = json.dumps({"error": {"message": str(content)}}).encode("utf-8")
                    self.send_response(status_code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(err_raw)))
                    self.end_headers()
                    self.wfile.write(err_raw)
                    return
                out = {
                    "id": f"chatcmpl-{turn_idx}",
                    "object": "chat.completion",
                    "created": 1,
                    "model": payload.get("model", "gemma-4-e4b"),
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                }
            raw = json.dumps(out).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, format: str, *args: Any) -> None:
            _ = (format, args)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, counts


def _write_toml(
    root: Path,
    *,
    port: int,
    http_port: int | None = None,
    api_key: str = "super-secret-cli-key-999",
    chunk_size: int = 160,
    chunk_overlap: int = 20,
    min_chunk_size: int = 20,
) -> Path:
    toml_path = root / "yaw.toml"
    mcp_port = http_port if http_port is not None else _free_port()
    toml_path.write_text(
        "\n".join(
            [
                "[paths]",
                'corpus_root = "corpus"',
                'db_path = ".yaw/yaw.db"',
                'lock_path = ".yaw/yaw.db.lock"',
                'state_dir = ".yaw/coco_lmdb"',
                'log_path = ".yaw/yaw.log"',
                'status_path = ".yaw/status.json"',
                'compiled_program_path = ".yaw/compiled_program.json"',
                "",
                "[ingestion]",
                f"chunk_size = {chunk_size}",
                f"chunk_overlap = {chunk_overlap}",
                f"min_chunk_size = {min_chunk_size}",
                "poll_interval_seconds = 0.2",
                "",
                "[inference]",
                f'base_url = "http://127.0.0.1:{port}/v1"',
                f'api_key = "{api_key}"',
                'provider = "openai"',
                'embedding_model = "embeddinggemma-2"',
                'qa_model = "gemma-4-e4b"',
                "react_max_iters = 4",
                "",
                "[runtime]",
                'http_host = "127.0.0.1"',
                f"http_port = {mcp_port}",
                'http_path = "/mcp"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    return toml_path


def _index_corpus(cfg: YawConfig) -> None:
    session = start_pipeline(cfg, live=False, embed_fn=make_dspy_embed_fn(cfg))
    try:
        update_pipeline_once(session)
    finally:
        close_pipeline(session)


def _prepare_sample_bundle(root: Path, *, dataset_name: str = "sample", total_questions: int = 6) -> Path:
    corpus_rows = [
        {
            "idx": "doc_apollo",
            "title": "Apollo Mission",
            "text": "Neil Armstrong served as commander of Apollo 11 on the lunar surface.",
        },
        {
            "idx": "doc_instrument",
            "title": "Wapakoneta Music",
            "text": "Neil Armstrong played the XylophoneZither instrument during childhood in Ohio.",
        },
        {
            "idx": "doc_booster",
            "title": "Booster Spec",
            "text": "Orbital booster stage separation relies on the cryo valve specification.",
        },
        {
            "idx": "doc_fjord",
            "title": "Bergen Harbor",
            "text": "Nordic glacier water feeds the deep Bergen fjord harbor year round.",
        },
        {
            "idx": "doc_optics",
            "title": "Prism Optics",
            "text": "The observatory telescope mirror aligns with a fused silica prism lens.",
        },
        {
            "idx": "doc_foundry",
            "title": "Copper Alloy",
            "text": "Copper smelter furnace temperatures control bronze alloy density in the foundry.",
        },
    ]
    question_rows = [
        {
            "question": "Who commanded Apollo 11 on the lunar surface?",
            "answer": "Neil Armstrong",
            "gold_docs": ["Apollo Mission\nNeil Armstrong served as commander of Apollo 11 on the lunar surface."],
        },
        {
            "question": "What childhood instrument did Neil Armstrong play in Ohio?",
            "answer": "XylophoneZither",
            "gold_docs": [
                "Wapakoneta Music\nNeil Armstrong played the XylophoneZither instrument during childhood in Ohio."
            ],
        },
        {
            "question": "Which valve governs orbital booster stage separation?",
            "answer": "cryo valve",
            "gold_docs": ["Booster Spec\nOrbital booster stage separation relies on the cryo valve specification."],
        },
        {
            "question": "What feeds the deep Bergen fjord harbor?",
            "answer": "Nordic glacier water",
            "gold_docs": ["Bergen Harbor\nNordic glacier water feeds the deep Bergen fjord harbor year round."],
        },
        {
            "question": "What aligns with the observatory telescope mirror?",
            "answer": "fused silica prism lens",
            "gold_docs": ["Prism Optics\nThe observatory telescope mirror aligns with a fused silica prism lens."],
        },
        {
            "question": "What controls bronze alloy density in the foundry?",
            "answer": "Copper smelter furnace temperatures",
            "gold_docs": [
                "Copper Alloy\nCopper smelter furnace temperatures control bronze alloy density in the foundry."
            ],
        },
    ]
    for idx in range(len(question_rows), total_questions):
        question_rows.append(
            {
                "question": f"OffsetCheck question {idx}: who commanded Apollo 11?",
                "answer": "Neil Armstrong",
                "gold_docs": ["Apollo Mission\nNeil Armstrong served as commander of Apollo 11 on the lunar surface."],
            }
        )
    src_dir = root / f"_raw_src_{dataset_name}"
    src_dir.mkdir(parents=True, exist_ok=True)
    c_path = src_dir / "corpus.json"
    q_path = src_dir / "questions.json"
    c_path.write_text(json.dumps(corpus_rows), encoding="utf-8")
    q_path.write_text(json.dumps(question_rows), encoding="utf-8")
    return prepare_dataset_bundle(dataset_name, c_path, q_path, root / "outputs" / "datasets").root


def _readyz_get(port: int) -> int:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"http://127.0.0.1:{port}/readyz", timeout=2.0) as resp:
        return int(resp.status)


def test_cli_containment_main_loc_zero_wrappers_and_clean_vocabulary() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    cli_path = repo_root / "yaw" / "cli.py"
    main_path = repo_root / "yaw" / "__main__.py"
    test_path = Path(__file__).resolve()

    main_src = main_path.read_text(encoding="utf-8")
    assert len(main_src.splitlines()) <= 10
    assert "yaw.cli" in main_src
    assert "yaw.daemon" not in main_src

    for path in (cli_path, main_path, test_path):
        low = path.read_text(encoding="utf-8").lower()
        for word in _FORBIDDEN_WORDS:
            assert word not in low, f"Forbidden word {word!r} in {path}"
        for marker in _FORBIDDEN_COMMENTS:
            assert marker not in low, f"Forbidden marker {marker!r} in {path}"

    cli_src = cli_path.read_text(encoding="utf-8")
    tree = ast.parse(cli_src, filename=str(cli_path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                base_txt = ast.unparse(base)
                assert not any(w in base_txt for w in ("Module", "BaseLM", "Adapter"))

    allowed_roots = {"dspy", "filelock", "yaw"}
    allowed_yaw = {
        "yaw.core",
        "yaw.datasets",
        "yaw.pipeline",
        "yaw.retrieval",
        "yaw.qa",
        "yaw.runtime",
        "yaw.mcp_server",
        "yaw.optimize",
    }
    forbidden_pkgs = ("yaw.daemon", "cocoindex", "optuna", "fastmcp", "starlette", "uvicorn", "numpy")
    stdlib_names = sys.stdlib_module_names
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_pkg = alias.name.split(".")[0]
                assert root_pkg in stdlib_names or root_pkg in allowed_roots
                assert not alias.name.startswith(forbidden_pkgs)
                if root_pkg == "yaw":
                    assert any(alias.name == a or alias.name.startswith(f"{a}.") for a in allowed_yaw)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            root_pkg = mod.split(".")[0]
            assert root_pkg in stdlib_names or root_pkg in allowed_roots
            assert not mod.startswith(forbidden_pkgs)
            if root_pkg == "yaw":
                assert any(mod == a or mod.startswith(f"{a}.") for a in allowed_yaw)
                for alias in node.names:
                    assert not alias.name.startswith("_")


def test_cli_help_exit_codes_and_global_flag_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from yaw.cli import cli_entrypoint, main

    assert main(["--help"]) == 0
    assert "daemon" in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", ["yaw", "--help"])
    with pytest.raises(SystemExit) as exited_main:
        runpy.run_module("yaw", run_name="__main__", alter_sys=True)
    assert exited_main.value.code == 0
    assert "daemon" in capsys.readouterr().out

    with pytest.raises(SystemExit) as exited_entry:
        cli_entrypoint()
    assert exited_entry.value.code == 0
    _ = capsys.readouterr()

    for bad_argv in (
        [],
        ["unknown"],
        ["index"],
        ["graph"],
        ["dspy"],
        ["daemon"],
        ["optimize"],
        ["config"],
        ["recall", "query", "--unknown-flag"],
    ):
        assert main(bad_argv) == 2
        _ = capsys.readouterr()

    toml_path = _write_toml(tmp_path, port=_free_port(), api_key="secret-token-err-42")
    missing_toml = tmp_path / "missing.toml"
    assert main(["--json", "config", "show", "--config", str(missing_toml)]) == 1
    err_json = json.loads(capsys.readouterr().err)
    assert "error" in err_json and "Configuration file not found" in err_json["error"]

    for bad_exec in (
        ["recall", "   ", "--config", str(toml_path)],
        ["recall", "q", "--top-k", "0", "--config", str(toml_path)],
        ["recall", "q", "--top-k", "21", "--config", str(toml_path)],
        ["recall", "q", "--config", str(toml_path)],
        ["qa", "   ", "--config", str(toml_path)],
        ["qa", "What happened?", "--config", str(toml_path)],
    ):
        assert main(bad_exec) == 1
        assert capsys.readouterr().err.strip()

    outputs: list[dict[str, Any]] = []
    for argv in (
        ["--json", "--config", str(toml_path), "config", "show"],
        ["config", "--json", "show", "--config", str(toml_path)],
        ["config", "show", "--config", str(toml_path), "--json"],
    ):
        assert main(argv) == 0
        outputs.append(json.loads(capsys.readouterr().out))
    assert outputs[0] == outputs[1] == outputs[2]


def test_cli_config_show_get_set_and_secret_redaction(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from yaw.cli import main

    toml_path = _write_toml(tmp_path, port=_free_port(), api_key="super-secret-cli-key-999")

    assert main(["config", "show", "--config", str(toml_path), "--json"]) == 0
    show_out = capsys.readouterr().out
    assert "super-secret-cli-key-999" not in show_out
    assert json.loads(show_out)["api_key"] == "[REDACTED]"

    assert main(["config", "get", "api_key", "--config", str(toml_path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {"key": "api_key", "value": "[REDACTED]"}

    assert main(["config", "get", "inference.api_key", "--config", str(toml_path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {"key": "inference.api_key", "value": "[REDACTED]"}

    assert main(["config", "get", "--config", str(toml_path), "--json"]) == 0
    get_all = json.loads(capsys.readouterr().out)
    assert get_all["api_key"] == "[REDACTED]"

    assert main(["config", "get", "chunk_size", "--config", str(toml_path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {"key": "chunk_size", "value": 160}

    assert main(["config", "set", "chunk_size", "800", "--config", str(toml_path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {"key": "ingestion.chunk_size", "value": 800}
    assert load_config(toml_path).chunk_size == 800

    assert main(["config", "set", "api_key", "rotated-secret-key-777", "--config", str(toml_path), "--json"]) == 0
    set_out = capsys.readouterr().out
    assert "rotated-secret-key-777" not in set_out
    assert json.loads(set_out) == {"key": "inference.api_key", "value": "[REDACTED]"}
    assert load_config(toml_path).api_key == "rotated-secret-key-777"

    assert main(["config", "get", "nonexistent_key", "--config", str(toml_path), "--json"]) == 1
    assert "error" in json.loads(capsys.readouterr().err)

    assert main(["config", "set", "chunk_size", "100.5", "--config", str(toml_path), "--json"]) == 1
    float_err = json.loads(capsys.readouterr().err)["error"]
    assert "INVALID_INPUT" in float_err
    assert load_config(toml_path).chunk_size == 800

    # Config command failures redact both existing cfg.api_key and failing `set api_key <secret>` values
    assert main(["config", "get", "rotated-secret-key-777", "--config", str(toml_path), "--json"]) == 1
    err_existing = capsys.readouterr().err
    assert "rotated-secret-key-777" not in err_existing and "[REDACTED]" in err_existing

    assert (
        main(["config", "set", "api_key", '"failing-secret-token-888"bad"', "--config", str(toml_path), "--json"]) == 1
    )
    err_pending = capsys.readouterr().err
    assert "failing-secret-token-888" not in err_pending and "[REDACTED]" in err_pending

    # Text-mode outputs (non-JSON) also redact secrets and format values cleanly
    assert main(["config", "show", "--config", str(toml_path)]) == 0
    txt_show = capsys.readouterr().out
    assert "api_key = [REDACTED]" in txt_show and "rotated-secret-key-777" not in txt_show

    assert main(["config", "get", "chunk_size", "--config", str(toml_path)]) == 0
    assert capsys.readouterr().out.strip() == "800"

    assert main(["config", "set", "chunk_size", "700", "--config", str(toml_path)]) == 0
    assert capsys.readouterr().out.strip() == "ingestion.chunk_size = 700"


def test_cli_recall_and_daemon_status_strict_readonly_zero_write_locks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from yaw.cli import main

    srv, counts = _start_openai_stub()
    try:
        toml_path = _write_toml(tmp_path, port=srv.server_port, api_key="secret-readonly-key-321")
        cfg = load_config(toml_path)
        assert cfg.corpus_root and cfg.lock_path and cfg.state_dir and cfg.status_path and cfg.db_path
        cfg.corpus_root.mkdir(parents=True, exist_ok=True)

        # Fresh workspace with no .yaw directory: daemon status creates ZERO files or locks
        fresh_ws = tmp_path / "fresh_ws"
        fresh_ws.mkdir()
        fresh_toml = _write_toml(fresh_ws, port=srv.server_port)
        assert main(["daemon", "status", "--config", str(fresh_toml), "--json"]) == 0
        fresh_st = json.loads(capsys.readouterr().out)
        assert fresh_st["lifecycle"] == "stopped" and fresh_st["num_chunks"] == 0
        assert not (fresh_ws / ".yaw").exists()

        # Empty DB chunks table -> recall returns empty [] with 0 extra HTTP calls; qa exits 1 EMPTY_INDEX
        _index_corpus(cfg)
        probes_before = counts["embeddings"]
        assert counts["chat"] == 0
        assert main(["recall", "anything", "--config", str(toml_path), "--json"]) == 0
        assert json.loads(capsys.readouterr().out) == {"query": "anything", "top_k": 5, "results": []}
        assert counts["embeddings"] == probes_before and counts["chat"] == 0

        assert main(["qa", "Anything?", "--config", str(toml_path), "--json"]) == 1
        assert "EMPTY_INDEX" in json.loads(capsys.readouterr().err)["error"]

        # Populate multi-chunk corpus and second doc to verify full passage hydration + MaxP pool_k
        multi_text = "\n\n".join(
            f"Stage {k}: " + " ".join(f"Orbital booster valve item {k}_{j}." for j in range(1, 6)) for k in range(1, 8)
        )
        (cfg.corpus_root / "multi.md").write_text(f"# Booster Manual\n{multi_text}\n", encoding="utf-8")
        (cfg.corpus_root / "backup.md").write_text(
            "# Booster Backup\nOrbital booster backup valve specification for emergency shutoff.\n",
            encoding="utf-8",
        )
        _index_corpus(cfg)
        counts["chat"] = 0

        # Hold yaw.db.lock externally while running BOTH recall and daemon status!
        cfg.lock_path.parent.mkdir(parents=True, exist_ok=True)
        held = filelock.FileLock(str(cfg.lock_path), timeout=0)
        held.acquire()
        try:
            assert main(["recall", "orbital booster", "--top-k", "2", "--config", str(toml_path), "--json"]) == 0
            rec_payload = json.loads(capsys.readouterr().out)
            assert rec_payload["query"] == "orbital booster"
            assert rec_payload["top_k"] == 2
            assert len(rec_payload["results"]) == 2
            assert counts["chat"] == 0
            for idx, item in enumerate(rec_payload["results"], start=1):
                assert item["rank"] == idx
                assert isinstance(item["chunk_id"], int)
                assert "[... cropped ...]" not in item["text"]
                assert item["stale_on_disk"] is False
                assert item["content_hash"].startswith("sha256:")
                assert {"rel_posix_path", "start_line", "end_line", "title"} <= set(item["source"])

            assert main(["daemon", "status", "--config", str(toml_path), "--json"]) == 0
            st_payload = json.loads(capsys.readouterr().out)
            assert st_payload["lifecycle"] == "stopped"
            assert st_payload["provider_ready"] is False
            assert st_payload["sync_state"] == "idle"
            assert st_payload["pid"] is None
            assert st_payload["num_chunks"] >= 2
        finally:
            held.release()

        # Stale PID + stale status.json must normalize to stopped without deleting files, and redact secrets
        dead_proc = subprocess.Popen([sys.executable, "-c", "pass"])
        dead_proc.wait(timeout=10)
        pid_path = cfg.state_dir / "daemon.pid"
        write_text_atomic(pid_path, f"{dead_proc.pid}\n")
        stale_status = json.dumps(
            {
                "lifecycle": "ready",
                "provider_ready": True,
                "sync_state": "syncing",
                "queue_depth": 4,
                "active_op": "qa",
                "last_error": f"Failed with key={cfg.api_key} Bearer tok_xyz",
                "last_sync_at": 1234.5,
                "pid": dead_proc.pid,
                "num_chunks": 1,
            }
        )
        write_text_atomic(cfg.status_path, stale_status)
        assert main(["daemon", "status", "--config", str(toml_path), "--json"]) == 0
        norm_st = json.loads(capsys.readouterr().out)
        assert norm_st["lifecycle"] == "stopped"
        assert norm_st["provider_ready"] is False
        assert norm_st["sync_state"] == "idle"
        assert norm_st["queue_depth"] == 0
        assert norm_st["active_op"] is None
        assert norm_st["pid"] is None
        assert norm_st["num_chunks"] >= 2
        assert cfg.api_key not in str(norm_st["last_error"]) and "tok_xyz" not in str(norm_st["last_error"])
        assert pid_path.is_file() and cfg.status_path.read_text(encoding="utf-8") == stale_status

        # Corrupt status.json must fail fast (exit 1), never silently suppressed
        cfg.status_path.write_text("{corrupt json", encoding="utf-8")
        assert main(["daemon", "status", "--config", str(toml_path), "--json"]) == 1
        assert "error" in json.loads(capsys.readouterr().err)
    finally:
        srv.shutdown()


def test_cli_qa_and_program_loading_over_loopback_http(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from yaw.cli import main

    seen_messages: list[str] = []

    def chat_responder(_turn_idx: int, payload: dict[str, Any]) -> tuple[int, str]:
        msgs = payload.get("messages", [])
        blob = json.dumps(msgs)
        seen_messages.append(blob)
        has_tool_obs = any(m.get("role") == "tool" for m in msgs)
        cids = [int(m.group(1)) for m in _CID_IN_TEXT_RE.finditer(blob)]
        if not has_tool_obs or not cids:
            return 200, _xml_tool_reply("Search corpus", [{"name": "search_passages", "args": {"query": "Apollo 11"}}])
        return 200, _xml_tool_reply(
            "Submit grounded answer",
            [{"name": "submit", "args": {"answer": f"Neil Armstrong [chunk_id={cids[0]}]"}}],
        )

    srv, _ = _start_openai_stub(chat_responder)
    try:
        toml_path = _write_toml(tmp_path, port=srv.server_port)
        cfg = load_config(toml_path)
        assert cfg.corpus_root and cfg.db_path and cfg.compiled_program_path
        cfg.corpus_root.mkdir(parents=True, exist_ok=True)
        (cfg.corpus_root / "apollo.md").write_text(
            "# Apollo 11\nNeil Armstrong served as commander of Apollo 11 on the lunar surface.\n",
            encoding="utf-8",
        )
        _index_corpus(cfg)

        with contextlib.closing(open_readonly_db(cfg.db_path)) as conn:
            vs = build_vector_search(conn)

            def _dummy_tool(query: str) -> str:
                return query

            tools: list[Callable[..., Any]] = [_dummy_tool]

            opt_agent = build_qa_agent(tools, max_iters=cfg.react_max_iters)
            assert opt_agent.react.signature is not None
            opt_agent.react.signature = opt_agent.react.signature.with_instructions("CLI_OPT_PROG_MARKER")
            opt_prog = cfg.workspace_root / ".yaw" / "qa_program.json"
            opt_prog.parent.mkdir(parents=True, exist_ok=True)
            opt_agent.save(str(opt_prog), save_program=False)

            assert main(["qa", "Who commanded Apollo 11?", "--config", str(toml_path), "--json"]) == 0
            qa_out = json.loads(capsys.readouterr().out)
            assert "Neil Armstrong" in qa_out["answer"]
            assert qa_out["turns"] >= 1
            assert qa_out["turns"] >= 1
            assert len(qa_out["citations"]) == 1
            assert qa_out["citations"][0]["chunk_id"] in vs.id_to_pos
            assert any("CLI_OPT_PROG_MARKER" in m for m in seen_messages)

            seen_messages.clear()
            comp_agent = build_qa_agent(tools, max_iters=cfg.react_max_iters)
            assert comp_agent.react.signature is not None
            comp_agent.react.signature = comp_agent.react.signature.with_instructions("CLI_COMPILED_PROG_MARKER")
            comp_agent.save(str(cfg.compiled_program_path), save_program=False)

            assert main(["qa", "Who commanded Apollo 11?", "--config", str(toml_path), "--json"]) == 0
            _ = json.loads(capsys.readouterr().out)
            assert any("CLI_COMPILED_PROG_MARKER" in m for m in seen_messages)

            assert main(["recall", "Apollo 11", "--top-k", "1", "--config", str(toml_path)]) == 0
            assert "[1] chunk_id=" in capsys.readouterr().out

            assert main(["qa", "Who commanded Apollo 11?", "--config", str(toml_path)]) == 0
            assert "Neil Armstrong" in capsys.readouterr().out

            assert main(["recall", "   ", "--config", str(toml_path)]) == 1
            assert "[INVALID_INPUT]" in capsys.readouterr().err
    finally:
        srv.shutdown()


def test_cli_optimize_dspy_and_optuna_commands(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from yaw.cli import main

    bundle = _prepare_sample_bundle(tmp_path, dataset_name="sample")
    bundle_14 = _prepare_sample_bundle(tmp_path, dataset_name="sample14", total_questions=14)
    seen_chat_blobs: list[str] = []

    def assert_plaintext_staging(stage: str, database: str) -> None:
        source = json.loads((tmp_path / "_raw_src_sample" / "corpus.json").read_text(encoding="utf-8"))
        expected = {
            "sample_"
            + hashlib.sha256(f"sample\0{row['idx']}".encode()).hexdigest()[:16]
            + ".txt": f"{row['title']}\n{row['text']}"
            for row in source
        }
        files = list((tmp_path / ".yaw" / stage).iterdir())
        assert files and all(path.suffix == ".txt" for path in files)
        for path in files:
            assert path.read_bytes() == expected[path.name].encode("utf-8")
        with open_readonly_db(tmp_path / ".yaw" / database) as conn:
            titles = conn.execute("SELECT title FROM chunks").fetchall()
            assert titles and all(row[0] == "" for row in titles)

    def chat_responder(_turn_idx: int, payload: dict[str, Any]) -> tuple[int, str]:
        msgs = payload.get("messages", [])
        sys_text = str(msgs[0].get("content", "")) if msgs else ""
        blob = json.dumps(msgs)
        seen_chat_blobs.append(blob)
        if "improved_instruction" in sys_text:
            return 200, _xml_tool_reply(
                "Submit concise instruction",
                [{"name": "submit", "args": {"improved_instruction": "Use search_passages concisely."}}],
            )
        has_tool_obs = any(m.get("role") == "tool" for m in msgs)
        cids = [int(m.group(1)) for m in _CID_IN_TEXT_RE.finditer(blob)]
        if not has_tool_obs or not cids:
            return 200, _xml_tool_reply(
                "Search passages", [{"name": "search_passages", "args": {"query": "Apollo 11", "top_k": 3}}]
            )
        cid = cids[0]
        ans = f"Neil Armstrong [chunk_id={cid}]" if "Apollo" in blob else f"XylophoneZither [chunk_id={cid}]"
        return 200, _xml_tool_reply("Submit answer", [{"name": "submit", "args": {"answer": ans}}])

    srv, _ = _start_openai_stub(chat_responder)
    try:
        toml_path = _write_toml(tmp_path, port=srv.server_port)
        assert (
            main(
                [
                    "optimize",
                    "dspy",
                    "--config",
                    str(toml_path),
                    "--dataset",
                    str(bundle),
                    "--auto",
                    "light",
                    "--max-metric-calls",
                    "4",
                ]
            )
            == 1
        )
        assert "mutually exclusive" in capsys.readouterr().err

        assert (
            main(
                [
                    "optimize",
                    "dspy",
                    "--config",
                    str(toml_path),
                    "--dataset",
                    str(bundle),
                    "--max-queries",
                    "4",
                    "--query-offset",
                    "0",
                    "--max-metric-calls",
                    "6",
                    "--fresh",
                    "--json",
                ]
            )
            == 0
        )
        dspy_res = json.loads(capsys.readouterr().out)
        program = Path(dspy_res["program_path"])
        assert program.is_file()
        assert_plaintext_staging("dspy_corpus", "optimize_yaw.db")
        assert json.loads(program.with_name("dspy_gepa_audit.json").read_text(encoding="utf-8")) == dspy_res

        # Omitting --query-offset honors [optimization] query_offset = 10 from yaw.toml
        seen_chat_blobs.clear()
        offset_ws = tmp_path / "offset_ws"
        offset_ws.mkdir()
        offset_toml = _write_toml(offset_ws, port=srv.server_port)
        offset_toml.write_text(
            offset_toml.read_text(encoding="utf-8") + "\n[optimization]\nquery_offset = 10\n",
            encoding="utf-8",
        )
        assert (
            main(
                [
                    "optimize",
                    "dspy",
                    "--config",
                    str(offset_toml),
                    "--dataset",
                    str(bundle_14),
                    "--max-queries",
                    "4",
                    "--max-metric-calls",
                    "4",
                    "--fresh",
                    "--json",
                ]
            )
            == 0
        )
        offset_res = json.loads(capsys.readouterr().out)
        assert offset_res["train_size"] + offset_res["val_size"] == 4
        joined_chats = "\n".join(seen_chat_blobs)
        assert "OffsetCheck question 10" in joined_chats
        assert "OffsetCheck question 6" not in joined_chats

        assert (
            main(
                [
                    "optimize",
                    "optuna",
                    "--config",
                    str(toml_path),
                    "--dataset",
                    str(bundle),
                    "--query-offset",
                    "0",
                    "--max-queries",
                    "6",
                    "--trials",
                    "2",
                    "--fresh",
                    "--json",
                ]
            )
            == 0
        )
        optuna_res = json.loads(capsys.readouterr().out)
        assert optuna_res["total_trials"] == 2
        assert_plaintext_staging("optuna_corpus", "optuna_index.db")
        assert (tmp_path / ".yaw" / "optuna_study.db").is_file()
        assert set(optuna_res["best_params"]) == {"hybrid_alpha", "ppr_alpha", "seed_top_k"}
    finally:
        srv.shutdown()


@pytest.mark.parametrize("force", [False, True])
def test_cli_optimizer_interrupt_during_provider_request(tmp_path: Path, force: bool) -> None:
    requested, release = threading.Event(), threading.Event()

    def stalled_chat(_turn: int, _payload: dict[str, Any]) -> tuple[int, str]:
        requested.set()
        assert release.wait(timeout=60)
        return 200, _xml_tool_reply("Finish", [{"name": "finish", "args": {"answer": "Neil Armstrong"}}])

    srv, _ = _start_openai_stub(stalled_chat)
    toml_path = _write_toml(tmp_path, port=srv.server_port)
    bundle = _prepare_sample_bundle(tmp_path)
    cfg = load_config(toml_path).for_optimization(None)
    assert cfg.db_path and cfg.lock_path
    command = (
        [sys.executable, "-m", "yaw"]
        if force
        else [sys.executable, "-c", "from yaw.cli import cli_entrypoint; cli_entrypoint()"]
    )
    proc = subprocess.Popen(
        [
            *command,
            "optimize",
            "dspy",
            "--config",
            str(toml_path),
            "--dataset",
            str(bundle),
            "--max-queries",
            "4",
            "--query-offset",
            "0",
            "--max-metric-calls",
            "6",
        ],
        cwd=tmp_path,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )
    sig = signal.SIGBREAK if os.name == "nt" else signal.SIGINT
    interrupted = threading.Event()
    errors: list[str] = []

    def read_errors() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            errors.append(line)
            if "Interrupt received; signal again to force exit." in line:
                interrupted.set()

    reader = threading.Thread(target=read_errors, daemon=True)
    reader.start()
    try:
        assert requested.wait(timeout=40), "Optimizer never reached the loopback provider"
        assert proc.poll() is None
        proc.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else sig)
        assert interrupted.wait(timeout=5), "First signal was not dispatched"
        if force:
            assert proc.poll() is None, "Blocked provider should keep a worker alive"
            proc.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else sig)
        else:
            release.set()
        proc.wait(timeout=15)
        reader.join(timeout=5)
        assert not reader.is_alive()
        stderr = "".join(errors)
        assert proc.returncode == 128 + sig, stderr
        assert "Traceback" not in stderr
        if not force:
            status = json.loads(cfg.db_path.with_name("dspy_status.json").read_text(encoding="utf-8"))
            assert status == {"state": "interrupted", "pid": None}
        with filelock.FileLock(str(cfg.lock_path), timeout=0):
            pass
    finally:
        release.set()
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        reader.join(timeout=5)
        if proc.stderr is not None:
            proc.stderr.close()
        srv.shutdown()
        srv.server_close()


def test_cli_daemon_detached_start_status_stop_and_run_pid_cleanup(tmp_path: Path) -> None:
    srv, _ = _start_openai_stub()
    mcp_port = _free_port()
    toml_path = _write_toml(tmp_path, port=srv.server_port, http_port=mcp_port)
    cfg = load_config(toml_path)
    assert cfg.corpus_root and cfg.lock_path and cfg.state_dir
    cfg.corpus_root.mkdir(parents=True, exist_ok=True)
    (cfg.corpus_root / "note.md").write_text("# Apollo\nNeil Armstrong commanded Apollo 11.\n", encoding="utf-8")
    pid_path = cfg.state_dir / "daemon.pid"

    def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "yaw", *args],
            cwd=str(tmp_path),
            capture_output=True,
            text=True,
            timeout=45,
        )

    started_pid: int | None = None
    run_proc: subprocess.Popen[str] | None = None
    try:
        stop_missing = _run_cli("daemon", "stop", "--config", str(toml_path), "--json")
        assert stop_missing.returncode == 1
        assert "NOT_READY" in json.loads(stop_missing.stderr)["error"]

        started = _run_cli("daemon", "start", "--config", str(toml_path), "--json")
        assert started.returncode == 0, started.stderr
        start_data = json.loads(started.stdout)
        started_pid = int(start_data["pid"])
        assert start_data["lifecycle"] == "ready"
        assert start_data["provider_ready"] is True
        assert start_data["num_chunks"] >= 1
        assert pid_path.is_file() and pid_path.read_text(encoding="utf-8").strip() == str(started_pid)
        assert _readyz_get(mcp_port) == 200

        status_live = _run_cli("daemon", "status", "--config", str(toml_path), "--json")
        assert status_live.returncode == 0, status_live.stderr
        live_data = json.loads(status_live.stdout)
        assert live_data["lifecycle"] == "ready"
        assert live_data["pid"] == started_pid
        assert live_data["num_chunks"] >= 1

        start_dup = _run_cli("daemon", "start", "--config", str(toml_path), "--json")
        assert start_dup.returncode == 1
        assert "BUSY" in json.loads(start_dup.stderr)["error"]

        stopped = _run_cli("daemon", "stop", "--config", str(toml_path), "--json")
        assert stopped.returncode == 0, stopped.stderr
        assert json.loads(stopped.stdout) == {"lifecycle": "stopped"}
        started_pid = None
        assert not pid_path.exists()

        # Verify yaw.db.lock was cleanly released by the daemon process
        lock = filelock.FileLock(str(cfg.lock_path), timeout=0)
        lock.acquire(timeout=0)
        lock.release()

        status_after = _run_cli("daemon", "status", "--config", str(toml_path), "--json")
        assert status_after.returncode == 0
        after_data = json.loads(status_after.stdout)
        assert after_data["lifecycle"] == "stopped"
        assert after_data["pid"] is None

        # Verify direct `daemon run` writes pid_path and unlinks it in its own finally block
        run_proc = subprocess.Popen(
            [sys.executable, "-m", "yaw", "daemon", "run", "--config", str(toml_path)],
            cwd=str(tmp_path),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if run_proc.poll() is not None:
                pytest.fail(f"daemon run exited early: {run_proc.stderr.read() if run_proc.stderr else ''}")
            try:
                if pid_path.is_file() and _readyz_get(mcp_port) == 200:
                    break
            except OSError:
                pass
            time.sleep(0.1)
        run_pid = int(pid_path.read_text(encoding="utf-8").strip())
        assert run_pid > 0
        assert cfg.status_path is not None
        assert json.loads(cfg.status_path.read_text(encoding="utf-8"))["pid"] == run_pid

        if os.name == "nt":
            assert signal_stop_event(cfg.lock_path) is True
        else:
            run_proc.terminate()
        assert run_proc.wait(timeout=15) == 0
        if run_proc.stderr is not None:
            run_proc.stderr.close()
        run_proc = None
        assert not pid_path.exists()
    finally:
        if started_pid is not None:
            if os.name == "nt":
                signal_stop_event(cfg.lock_path)
            else:
                subprocess.run(["kill", str(started_pid)], check=False)
        if run_proc is not None and run_proc.poll() is None:
            run_proc.kill()
            run_proc.wait(timeout=5)
        srv.shutdown()
