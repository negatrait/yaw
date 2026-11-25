"""Real-substrate integration tests for the clean-slate YAW watchdog monitor."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar, override
from xml.etree import ElementTree

import filelock
import numpy as np
import pytest

import scripts.watchdog as watchdog
from yaw.core import DaemonStatus


def _create_yaw_db(db_path: Path, num_chunks: int) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(str(db_path))) as conn:
        conn.execute("CREATE TABLE chunks (chunk_id INTEGER PRIMARY KEY, embedding_f16 BLOB NOT NULL)")
        conn.execute(
            "CREATE VIRTUAL TABLE chunk_fts USING fts5("
            "title, body, content='', contentless_delete=1, tokenize='unicode61 remove_diacritics 1')"
        )
        conn.execute("CREATE VIRTUAL TABLE chunk_vocab USING fts5vocab(chunk_fts, 'row')")
        conn.execute("CREATE TABLE rabitq_index (id INTEGER PRIMARY KEY CHECK (id = 1))")
        conn.execute("CREATE TABLE citation_graph (id INTEGER PRIMARY KEY CHECK (id = 1))")
        for idx in range(1, num_chunks + 1):
            vec = np.array([1.0 if i == (idx % 4) else 0.0 for i in range(4)], dtype="<f2").tobytes()
            conn.execute("INSERT INTO chunks(chunk_id, embedding_f16) VALUES (?, ?)", (idx, vec))
            conn.execute(
                "INSERT INTO chunk_fts(rowid, title, body) VALUES (?, ?, ?)", (idx, f"title {idx}", f"body {idx}")
            )
        conn.commit()


@pytest.fixture
def prometheus_server():
    seen: list[tuple[str, str | None]] = []

    class _Handler(BaseHTTPRequestHandler):
        routes: ClassVar[list[tuple[str, str | None]]] = seen

        def do_GET(self) -> None:
            self.routes.append((self.path, self.headers.get("Authorization")))
            body = b"# TYPE llamacpp:predicted_tokens_seconds gauge\nllamacpp:predicted_tokens_seconds 98.76\n"
            self.send_response(200 if self.path == "/metrics" else 404)
            self.end_headers()
            self.wfile.write(body)

        @override
        def log_message(self, format: str, *args: object) -> None:
            _ = format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1", seen
    server.shutdown()
    server.server_close()


def _populate_workspace(ws: Path, base_url: str, api_key: str) -> None:
    yaw_dir = ws / ".yaw"
    yaw_dir.mkdir(parents=True, exist_ok=True)
    _create_yaw_db(yaw_dir / "yaw.db", 2)
    _create_yaw_db(yaw_dir / "optimize_yaw.db", 1)

    (ws / "yaw.toml").write_text(
        f'[inference]\nbase_url = "{base_url}"\napi_key = "{api_key}"\n',
        encoding="utf-8",
    )
    status = DaemonStatus(
        lifecycle="ready",
        provider_ready=True,
        sync_state="idle",
        pid=os.getpid(),
        num_chunks=2,
    ).to_dict()
    (yaw_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")
    (yaw_dir / "dspy_status.json").write_text(json.dumps({"state": "running", "pid": os.getpid()}), encoding="utf-8")
    (yaw_dir / "dspy_gepa_audit.json").write_text(
        json.dumps(
            {
                "datasets": ["enterprise_rag_bench"],
                "train_size": 10,
                "val_size": 5,
                "num_candidates": 3,
                "best_val_score": 0.78,
                "val_aggregate_scores": [0.45, 0.61, 0.78],
            }
        ),
        encoding="utf-8",
    )


def test_snapshot_reads_real_sqlite_fts_locks_json_and_metrics(tmp_path: Path, prometheus_server) -> None:
    base_url, seen = prometheus_server
    _populate_workspace(tmp_path, base_url, "tok-123")
    yaw_dir = tmp_path / ".yaw"
    (yaw_dir / "yaw.db.lock").write_text("", encoding="utf-8")

    held_lock = filelock.FileLock(str(yaw_dir / "optimize_yaw.db.lock"), timeout=0)
    held_lock.acquire()
    try:
        snap = watchdog.get_active_snapshot(tmp_path)
    finally:
        held_lock.release()

    assert set(snap["databases"]) == {"yaw.db", "optimize_yaw.db"}
    assert snap["databases"]["yaw.db"]["chunks"] == 2
    assert snap["databases"]["yaw.db"]["wiki_chunks"] == 0
    assert snap["databases"]["yaw.db"]["tables"] == [
        "chunk_fts",
        "chunk_vocab",
        "chunks",
        "citation_graph",
        "rabitq_index",
    ]
    assert snap["databases"]["optimize_yaw.db"]["chunks"] == 1
    assert snap["locks"] == {
        "yaw.db.lock": False,
        "optimize_yaw.db.lock": True,
    }
    assert "optuna_status" not in snap
    assert "optuna_best_params" not in snap
    assert snap["status"]["lifecycle"] == "ready"
    assert snap["dspy_status"]["state"] == "running"
    assert snap["dspy_gepa_audit"]["best_val_score"] == 0.78
    assert snap["llm_metrics"] == {"server_up": True, "llm_tps": 98.8}
    assert seen == [("/metrics", "Bearer tok-123")]
    assert isinstance(snap["resources"]["pipeline_ram_gb"], float)


def test_run_once_and_cli_emit_valid_svg_markdown_and_log(tmp_path: Path, prometheus_server) -> None:
    base_url, _ = prometheus_server
    _populate_workspace(tmp_path, base_url, "cli-secret")
    out_dir = tmp_path / "artifacts"

    rc = watchdog.main(["--once", "--workspace", str(tmp_path), "--artifact-dir", str(out_dir)])
    assert rc == 0

    svg_path = out_dir / "score_progress.svg"
    md_path = out_dir / "benchmark_mobile_report.md"
    assert svg_path.is_file()
    assert md_path.is_file()

    root = ElementTree.fromstring(svg_path.read_text(encoding="utf-8"))
    assert root.tag.endswith("svg")
    svg_text = " ".join(root.itertext())
    assert "DSPy GEPA Score Progress" in svg_text
    assert "0.7800" in svg_text
    assert not watchdog.plt.get_fignums()

    report = md_path.read_text(encoding="utf-8")
    assert "![Score Progress](score_progress.svg)" in report
    assert "optuna" not in report.lower()
    assert "`ready`" in report
    assert "`running`" in report
    assert "`2` chunks" in report
    assert "`1` chunks" in report
    assert "`0.7800`" in report
    assert "98.8 tok/s" in report

    log_path = tmp_path / ".yaw" / "yaw.log"
    assert log_path.is_file()
    assert "Snapshot updated: daemon=ready, yaw_chunks=2" in log_path.read_text(encoding="utf-8")


def test_thermal_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(watchdog, "get_gpu_temperature", lambda: None)
    assert watchdog.check_thermal_guard() is False

    monkeypatch.setattr(watchdog, "get_gpu_temperature", lambda: 70.0)
    assert watchdog.check_thermal_guard(crisis_temp=82.0) is False

    temps = iter([85.0, 80.0, 71.0])
    monkeypatch.setattr(watchdog, "get_gpu_temperature", lambda: next(temps, 71.0))
    assert watchdog.check_thermal_guard(crisis_temp=82.0, target_temp=72.0, max_pause_sec=1.0, poll_sec=0.005)


def test_snapshot_respects_custom_paths_from_yaw_toml(tmp_path: Path) -> None:
    custom_dir = tmp_path / "custom_state"
    status_dir = tmp_path / "runtime_meta"
    status_dir.mkdir(parents=True)

    # Decoy default .yaw/ files must be ignored when yaw.toml configures custom paths.
    decoy_dir = tmp_path / ".yaw"
    decoy_dir.mkdir(parents=True)
    _create_yaw_db(decoy_dir / "yaw.db", 9)
    (decoy_dir / "status.json").write_text(json.dumps({"lifecycle": "error"}), encoding="utf-8")
    (decoy_dir / "dspy_status.json").write_text(json.dumps({"state": "failed"}), encoding="utf-8")

    (tmp_path / "yaw.toml").write_text(
        "[paths]\n"
        'db_path = "custom_state/main.db"\n'
        'lock_path = "custom_state/main.db.lock"\n'
        'status_path = "runtime_meta/daemon_status.json"\n',
        encoding="utf-8",
    )
    _create_yaw_db(custom_dir / "main.db", 4)
    _create_yaw_db(custom_dir / "optimize_yaw.db", 2)

    (status_dir / "daemon_status.json").write_text(
        json.dumps({"lifecycle": "ready", "num_chunks": 4}), encoding="utf-8"
    )
    (custom_dir / "dspy_status.json").write_text(json.dumps({"state": "running"}), encoding="utf-8")
    (custom_dir / "dspy_gepa_audit.json").write_text(json.dumps({"best_val_score": 0.91}), encoding="utf-8")
    (custom_dir / "optimize_yaw.db.lock").write_text("", encoding="utf-8")

    main_lock = filelock.FileLock(str(custom_dir / "main.db.lock"), timeout=0)
    main_lock.acquire()
    try:
        snap = watchdog.get_active_snapshot(tmp_path, include_llm_metrics=False)
    finally:
        main_lock.release()

    assert set(snap["databases"]) == {"yaw.db", "optimize_yaw.db"}
    assert Path(snap["databases"]["yaw.db"]["path"]) == (custom_dir / "main.db").resolve()
    assert snap["databases"]["yaw.db"]["chunks"] == 4
    assert snap["databases"]["optimize_yaw.db"]["chunks"] == 2
    assert snap["locks"] == {
        "yaw.db.lock": True,
        "optimize_yaw.db.lock": False,
    }
    assert "optuna_status" not in snap
    assert "optuna_best_params" not in snap
    assert snap["status"] == {"lifecycle": "ready", "num_chunks": 4}
    assert snap["dspy_status"] == {"state": "running"}
    assert snap["dspy_gepa_audit"] == {"best_val_score": 0.91}


def test_snapshot_counts_wiki_chunks_when_rel_path_present(tmp_path: Path) -> None:
    db_path = tmp_path / ".yaw" / "yaw.db"
    db_path.parent.mkdir(parents=True)
    with closing(sqlite3.connect(str(db_path))) as conn:
        conn.execute("CREATE TABLE chunks (chunk_id INTEGER PRIMARY KEY, rel_path TEXT NOT NULL, embedding_f16 BLOB)")
        conn.execute("INSERT INTO chunks VALUES (1, 'alpha.md', X'0000')")
        conn.execute("INSERT INTO chunks VALUES (2, 'wiki:card.md', X'0000')")
        conn.commit()
    snap = watchdog.get_active_snapshot(tmp_path, include_llm_metrics=False)
    assert snap["databases"]["yaw.db"]["chunks"] == 2
    assert snap["databases"]["yaw.db"]["wiki_chunks"] == 1
