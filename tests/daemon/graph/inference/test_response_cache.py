"""Native DSPy disk-cache reuse and logical-cost parity over actual loopback HTTP."""

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from yaw.daemon.graph.inference import configure_yaw_dspy_cache

_WORKER = r'''
import json, os, sys
from pathlib import Path
import dspy
from dspy.utils.usage_tracker import track_usage
from yaw.daemon.graph.inference import configure_yaw_dspy_cache, create_dspy_lm

root, url, case = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
configure_yaw_dspy_cache(root / "responses", enable_disk_cache=True,
    enable_memory_cache=False, restrict_pickle=True, account_cached_usage=True)
lm = create_dspy_lm("cache-other" if case == "model" else "cache-test",
    base_url=url, api_key="test", temperature=0.4 if case == "temperature" else 0.2,
    max_tokens=200, cache=case != "bypass",
    **({"engine": "litellm"} if case == "legacy" else {}))
dspy.configure(lm=lm, track_usage=True, adapter=dspy.JSONAdapter())
if case == "gepa":
    student = dspy.Predict("question -> answer")
    examples = [dspy.Example(question="Name the cached result.").with_inputs("question")]
    costs = []
    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        usage = pred.get_lm_usage() or {}
        cost = sum(v.get("total_tokens", 0) for v in usage.values())
        costs.append(cost)
        return 1.0 / (1.0 + cost / 100)
    optimizer = dspy.GEPA(metric=metric, reflection_lm=lm, max_metric_calls=1,
        track_stats=True, log_dir=str(root / ("gepa_" + str(os.getpid()))))
    compiled = optimizer.compile(student, trainset=examples, valset=examples)
    report = {"costs": costs, "scores": compiled.detailed_results.val_aggregate_scores}
else:
    messages = [{"role": "user", "content": "A changed request" if case == "prompt"
                 else "Return the cached result."}]
    options = {"extra_body": {"grammar": 'root ::= "cached"'}} if case == "grammar" else {}
    logical, physical, outputs = [], [], []
    for _ in range(2):
        with track_usage() as tracker:
            outputs.append(lm(messages=messages, **options))
        logical.append(sum(v.get("total_tokens", 0) for v in tracker.get_total_tokens().values()))
        physical.append(lm.history[-1]["usage"].get("total_tokens", 0))
    report = {"logical": logical, "physical": physical, "outputs": outputs}
report.update(hits=dspy.cache.cache_hits, cached_tokens=dspy.cache.cached_tokens,
    disk=dspy.cache.enable_disk_cache, memory=dspy.cache.enable_memory_cache)
print(json.dumps(report))
dspy.cache.disk_cache.close()
'''


@pytest.fixture
def inference_server():
    requests: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(request)
            content = (
                '{"answer":"cached","reasoning":"Grounded in the supplied passage."}'
                if request["model"] == "retention-test" else '{"answer":"cached"}'
            )
            response = {
                "id": "cache-test", "object": "chat.completion", "created": 0,
                "model": request["model"], "choices": [{"index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            }
            body = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/v1", requests
        finally:
            server.shutdown()
            thread.join(timeout=5)


def _run_worker(root: Path, url: str, case: str = "normal") -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "-c", _WORKER, str(root), url, case],
        capture_output=True, text=True, timeout=60,
        env=dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[4])),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_long_qa_run_bounds_diagnostics_preserves_cost_and_avoids_history(
    tmp_path: Path, inference_server,
) -> None:
    import gc
    import tracemalloc

    import dspy
    from dspy.clients.base_lm import GLOBAL_HISTORY

    from yaw.daemon.graph.inference import HippoRAGProgram, create_dspy_lm

    url, requests = inference_server
    configure_yaw_dspy_cache(
        tmp_path / "retention", enable_memory_cache=False, account_cached_usage=True,
    )
    lm = create_dspy_lm("retention-test", base_url=url, api_key="test", max_tokens=200)
    program = HippoRAGProgram(enable_gbnf=False, orchestrator_lm=lm, rlm_sub_lm=lm)
    program.max_invariant_retries = 0
    before_global = len(GLOBAL_HISTORY)
    total_tokens = 0
    # This is the same supported DSPy context used throughout engine evolution.
    with dspy.context(lm=lm, disable_history=True, max_history_size=0, track_usage=True):
        tracemalloc.start()
        try:
            for iteration in range(600):
                with dspy.context(trace=[]):
                    prediction = program.answer_question(
                        context="The cached result is documented here. " * 100,
                        question="What is the result?",
                    )
                    assert dspy.settings.trace  # GEPA traces survive history suppression.
                assert prediction.answer == "cached"
                total_tokens += sum(v["total_tokens"] for v in prediction.get_lm_usage().values())
                if iteration == 299:
                    gc.collect()
                    retained_after_warmup = tracemalloc.get_traced_memory()[0]
            gc.collect()
            retained_at_end = tracemalloc.get_traced_memory()[0]
        finally:
            tracemalloc.stop()
    assert total_tokens == 600 * 30
    assert len(requests) == 1
    assert len(program.telemetry) == 250
    assert not lm.history and not program.history and not program.qa.lm.history
    assert len(GLOBAL_HISTORY) == before_global
    assert retained_at_end - retained_after_warmup < 1_000_000
    print({"calls": 600, "provider_requests": len(requests), "logical_tokens": total_tokens,
           "retained_growth_bytes": retained_at_end - retained_after_warmup})


def test_response_disk_cache_across_processes_and_request_invalidation(
    tmp_path: Path, inference_server,
) -> None:
    url, requests = inference_server
    first = _run_worker(tmp_path, url)
    assert len(requests) == 1
    assert first["logical"] == [30, 30]
    assert first["physical"] == [30, 0]
    second = _run_worker(tmp_path, url)
    assert len(requests) == 1
    assert second["logical"] == [30, 30]
    assert second["physical"] == [0, 0]
    assert second["hits"] == 2 and second["cached_tokens"] == 60
    assert second["disk"] and not second["memory"]
    for case in ("prompt", "model", "temperature", "grammar"):
        before = len(requests)
        changed = _run_worker(tmp_path, url, case)
        assert len(requests) == before + 1
        assert changed["logical"] == [30, 30]
    before = len(requests)
    bypass = _run_worker(tmp_path, url, "bypass")
    assert len(requests) == before + 2
    assert bypass["physical"] == [30, 30]


def test_native_gepa_scores_match_after_request_cache_restart(
    tmp_path: Path, inference_server,
) -> None:
    url, requests = inference_server
    first = _run_worker(tmp_path, url, "gepa")
    assert len(requests) == 1
    second = _run_worker(tmp_path, url, "gepa")
    assert len(requests) == 1
    assert first["costs"] == second["costs"] == [30]
    assert first["scores"] == second["scores"]
    assert first["scores"][0] == pytest.approx(1 / 1.3)
    assert second["hits"] == 1


def test_legacy_sdk_response_cache_preserves_logical_usage(
    tmp_path: Path, inference_server,
) -> None:
    url, requests = inference_server
    first = _run_worker(tmp_path, url, "legacy")
    second = _run_worker(tmp_path, url, "legacy")
    assert len(requests) == 1
    assert first["logical"] == second["logical"] == [30, 30]
    assert second["physical"] == [0, 0]


def test_response_cache_does_not_cross_inference_endpoints(
    tmp_path: Path, inference_server,
) -> None:
    url, requests = inference_server
    _run_worker(tmp_path, url)
    # Same server, different endpoint path: DSPy must issue a new request.
    _run_worker(tmp_path, url + "/other")
    assert len(requests) == 2


def test_response_cache_passthrough_and_unreported_usage(tmp_path: Path) -> None:
    import dspy

    configure_yaw_dspy_cache(
        tmp_path, enable_memory_cache=False, account_cached_usage=True,
    )
    try:
        unrelated = {"_fn_identifier": "application.pure_function", "input": "data"}
        dspy.cache.put(unrelated, {"answer": "unchanged"})
        assert dspy.cache.get(unrelated) == {"answer": "unchanged"}
        assert dspy.cache.cache_hits == 0
        request = {"_fn_identifier": "dspy.clients.lm.litellm_completion", "model": "test"}
        record = {"_dspy_format": "counter-test", "usage": {
            "prompt_tokens": None, "completion_tokens": None, "total_tokens": None,
        }}
        dspy.cache.put(request, record)
        with dspy.context(usage_tracker=None):
            assert dspy.cache.get(request) == record
        assert dspy.cache.cached_tokens == 0
    finally:
        configure_yaw_dspy_cache(enable_disk_cache=False, enable_memory_cache=False)
