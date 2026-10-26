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
    monkeypatch.setattr(downloader, "HF_BASE_URL", url + "/primary")
    monkeypatch.setattr(downloader, "GITHUB_BASE_URL", url + "/fallback")
    target = downloader.download_file("sample.json", tmp_path)
    assert target.read_bytes() == state["payload"]
    assert state["requests"] == ["/primary/sample.json", "/fallback/sample.json"]
    assert not target.with_suffix(".json.tmp").exists()
    downloader.download_file("sample.json", tmp_path)
    assert len(state["requests"]) == 2
    state["payload"] = b"replacement"
    downloader.download_file("sample.json", tmp_path, force=True)
    assert target.read_bytes() == b"replacement"


def test_failed_download_preserves_previous_target_and_removes_temporary_file(
    tmp_path, monkeypatch, downloader, download_server
):
    state, url = download_server
    monkeypatch.setattr(downloader, "HF_BASE_URL", url + "/primary")
    monkeypatch.setattr(downloader, "GITHUB_BASE_URL", url + "/fallback")
    state["fail_all"] = True
    target = tmp_path / "sample.json"
    target.write_bytes(b"original")
    temporary = tmp_path / "sample.json.tmp"
    temporary.write_bytes(b"incomplete")
    with pytest.raises(RuntimeError, match="Could not download.*any source"):
        downloader.download_file("sample.json", tmp_path, force=True)
    assert target.read_bytes() == b"original"
    assert not temporary.exists()
    assert len(state["requests"]) == 2


@pytest.mark.parametrize("mode", ["check-valid", "check-invalid", "all-cached", "alias-cached", "invalid-cached"])
def test_downloader_cli_reports_real_file_integrity_without_external_network(tmp_path, downloader, mode):
    for query, corpus, _description in downloader.DATASET_REGISTRY.values():
        (tmp_path / query).write_text('[{"question":"Source"}]', encoding="utf-8")
        (tmp_path / corpus).write_text('[{"text":"Content"}]', encoding="utf-8")
    if mode in ("check-invalid", "invalid-cached"):
        (tmp_path / "sample.json").write_text("invalid json", encoding="utf-8")
    arguments = ["--dest", str(tmp_path)]
    if mode.startswith("check"):
        arguments.append("--check")
    else:
        arguments += ["--dataset", "naturalquestions" if mode == "alias-cached" else "all"]
    result = subprocess.run(
        [sys.executable, downloader.__file__, *arguments],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        timeout=30,
    )
    assert result.returncode == (1 if mode == "check-invalid" else 0), result.stdout + result.stderr
    if mode == "check-invalid":
        assert "[MISSING] sample" in result.stdout
    elif mode == "invalid-cached":
        assert "[FAIL] sample" in result.stdout
    else:
        assert "[OK]" in result.stdout
    assert "Downloading sample.json from" not in result.stdout


def test_enterprise_cache_is_used_and_undersized_category_is_rejected(tmp_path, downloader):
    query, corpus, _ = downloader.DATASET_REGISTRY["enterprise_rag_bench"]
    for filename in (query, corpus):
        (tmp_path / filename).write_text(json.dumps([{"preserved": True}]), encoding="utf-8")
    assert downloader.download_enterprise_rag_bench(tmp_path) == (tmp_path / query, tmp_path / corpus)
    before = (tmp_path / query).read_bytes()
    with pytest.raises(ValueError, match="Category.*only 0 questions"):
        downloader.download_enterprise_rag_bench(tmp_path, questions_rows=[], documents_rows=[])
    assert (tmp_path / query).read_bytes() == before
