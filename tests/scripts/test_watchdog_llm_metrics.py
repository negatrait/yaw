"""Watchdog throughput comes from the llama.cpp Prometheus endpoint of the configured inference server."""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import scripts.watchdog as watchdog

METRICS = """# HELP llamacpp:predicted_tokens_seconds Average generation throughput in tokens/s.
# TYPE llamacpp:predicted_tokens_seconds gauge
llamacpp:predicted_tokens_seconds 151.37
llamacpp:requests_processing 1
"""


@pytest.fixture
def metrics_server():
    seen: list[tuple[str, str | None]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append((self.path, self.headers["Authorization"]))
            self.send_response(200 if self.path == "/metrics" else 404)
            self.end_headers()
            self.wfile.write(METRICS.encode())

        def log_message(self, format: str, *args) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1", seen
    server.shutdown()
    server.server_close()


def test_generation_throughput_is_read_from_the_server_root_metrics_route(metrics_server):
    base_url, seen = metrics_server
    metrics = watchdog.get_llm_server_metrics(base_url, "secret")
    assert metrics == {"server_up": True, "llm_tps": 151.4}
    assert seen == [("/metrics", "Bearer secret")]


def test_unreachable_server_reports_down_not_zero():
    assert watchdog.get_llm_server_metrics("http://127.0.0.1:9/v1", "k") == {"server_up": False}


def test_server_without_the_generation_gauge_reports_up_without_throughput(monkeypatch, metrics_server):
    base_url, _ = metrics_server
    monkeypatch.setattr(watchdog, "_PROMETHEUS_TPS", "llamacpp:absent")
    assert watchdog.get_llm_server_metrics(base_url, "k") == {"server_up": True, "llm_tps": None}
