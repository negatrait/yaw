"""Downloader network fallback and CLI contracts over loopback and real files."""

import importlib.util
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TypedDict

import pytest

from yaw.daemon.ingestion import download_dataset_file, prepare_dataset_bundle, select_enterprise_questions


class DownloadState(TypedDict):
    requests: list[str]
    fail_all: bool
    payload: bytes


@pytest.fixture
def downloader():
    path = Path(__file__).resolve().parents[2] / "scripts" / "download-datasets.py"
    spec = importlib.util.spec_from_file_location("download_edges_subject", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def download_server():
    state: DownloadState = {"requests": [], "fail_all": False, "payload": b"x" * (1024 * 1024 + 123)}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state["requests"].append(self.path)
            if state["fail_all"] or self.path.startswith("/primary"):
                self.send_error(503)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(state["payload"])))
            self.end_headers()
            self.wfile.write(state["payload"])

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state, f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_download_fallback_streams_complete_bytes_and_force_replaces_atomically(
    tmp_path, monkeypatch, downloader, download_server
):
    state, url = download_server
    target = download_dataset_file(
        "sample.json", tmp_path, hf_base_url=url + "/primary", upstream_base_url=url + "/fallback"
    )
    assert target.read_bytes() == state["payload"]
    assert state["requests"] == ["/primary/sample.json", "/fallback/sample.json"]
    assert not target.with_suffix(".json.tmp").exists()
    download_dataset_file("sample.json", tmp_path, hf_base_url=url + "/primary", upstream_base_url=url + "/fallback")
    assert len(state["requests"]) == 2
    state["payload"] = b"replacement"
    download_dataset_file(
        "sample.json", tmp_path, force=True, hf_base_url=url + "/primary", upstream_base_url=url + "/fallback"
    )
    assert target.read_bytes() == b"replacement"


def test_failed_download_preserves_previous_target_and_removes_temporary_file(
    tmp_path, monkeypatch, downloader, download_server
):
    state, url = download_server
    state["fail_all"] = True
    target = tmp_path / "sample.json"
    target.write_bytes(b"original")
    temporary = tmp_path / "sample.json.tmp"
    temporary.write_bytes(b"incomplete")
    with pytest.raises(RuntimeError, match="Could not download.*any source"):
        download_dataset_file(
            "sample.json", tmp_path, force=True, hf_base_url=url + "/primary", upstream_base_url=url + "/fallback"
        )
    assert target.read_bytes() == b"original"
    assert not temporary.exists()
    assert len(state["requests"]) == 2


@pytest.mark.parametrize("mode", ["check-valid", "check-invalid", "all-cached", "alias-cached", "invalid-source"])
def test_downloader_cli_reports_real_bundle_integrity_without_external_network(tmp_path, downloader, mode):
    sources, prepared = tmp_path / "sources", tmp_path / "prepared"
    sources.mkdir()
    for dataset, (query, corpus) in downloader.DATASET_REGISTRY.items():
        qp, cp = sources / query, sources / corpus
        qp.write_text('[{"question":"Source","answer":"Original"}]', encoding="utf-8")
        cp.write_text('[{"title":"Original","text":"Content"}]', encoding="utf-8")
        if mode.startswith("check"):
            prepare_dataset_bundle(dataset, cp, qp, prepared)
    if mode == "check-invalid":
        (prepared / "sample/evaluation/questions.json").write_text("invalid", encoding="utf-8")
    if mode == "invalid-source":
        (sources / "sample.json").write_text("invalid", encoding="utf-8")
    arguments = ["--dest", str(prepared), "--source-dir", str(sources)]
    arguments += (
        ["--check"]
        if mode.startswith("check")
        else ["--dataset", "naturalquestions" if mode == "alias-cached" else "all"]
    )
    result = subprocess.run(
        [sys.executable, downloader.__file__, *arguments],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        timeout=60,
    )
    assert (result.returncode != 0) == (mode in ("check-invalid", "invalid-source")), result.stdout + result.stderr
    if mode == "check-invalid":
        assert "[FAIL] sample" in result.stdout
    elif mode != "invalid-source":
        assert "[OK]" in result.stdout


def test_undersized_enterprise_selection_is_rejected_by_loader_without_touching_sources(tmp_path):
    source = tmp_path / "questions.json"
    source.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="Enterprise category"):
        select_enterprise_questions(json.loads(source.read_text(encoding="utf-8")))
    assert source.read_text(encoding="utf-8") == "[]"
