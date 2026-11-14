"""Native asynchronous embeddings against a real OpenAI-compatible HTTP server."""

import asyncio
import json
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
import pytest

from yaw.daemon.graph.inference.embedding import DSPyEmbeddingAdapter, create_dspy_embedder, resolve_embedding_profile


@pytest.fixture
def embedding_server():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, self.headers.get("Authorization"), body))
            assert self.path == "/v1/embeddings"
            assert self.headers.get("Authorization") == "Bearer fixture-key"
            inputs = body["input"]
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "object": "list",
                        "model": body["model"],
                        "data": [
                            {"object": "embedding", "index": i, "embedding": [3.0, 4.0]} for i in range(len(inputs))
                        ],
                        "usage": {"prompt_tokens": 1, "total_tokens": 1},
                    }
                ).encode()
            )

        def log_message(self, format: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield requests, f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_native_async_embedder_uses_real_http_provider(embedding_server):
    requests, base_url = embedding_server
    profile = replace(resolve_embedding_profile(), dimensions=2)
    embedder = create_dspy_embedder(profile, base_url=base_url, api_key="fixture-key", caching=False)
    adapter = DSPyEmbeddingAdapter(profile, embedder)
    sync = adapter.batch_encode(["sync"])
    np.testing.assert_allclose(sync, [[0.6, 0.8]])
    vectors = asyncio.run(adapter.acall(["first", "second"]))
    np.testing.assert_allclose(vectors, [[0.6, 0.8], [0.6, 0.8]])
    assert requests[0][2]["input"] == [profile.document_prefix + "sync"]
    (path, auth, body) = requests[1]
    assert path == "/v1/embeddings" and auth == "Bearer fixture-key"
    assert body["input"] == [profile.document_prefix + "first", profile.document_prefix + "second"]


def test_factory_preserves_endpoint_identity_without_connecting():
    profile = replace(resolve_embedding_profile(), dimensions=2)
    first = create_dspy_embedder(profile, base_url="http://127.0.0.1:10001/v1", api_key="fixture-key")
    second = create_dspy_embedder(profile, base_url="http://127.0.0.1:10002/v1", api_key="fixture-key")
    assert first.default_kwargs["base_url"] == "http://127.0.0.1:10001/v1"
    assert "api_base" not in first.default_kwargs
    assert DSPyEmbeddingAdapter(profile, first).index_identity != DSPyEmbeddingAdapter(profile, second).index_identity


def test_shared_sync_async_admission_limits_real_http_to_two():
    import time
    from concurrent.futures import ThreadPoolExecutor

    lock = threading.Lock()
    state = {"active": 0, "peak": 0, "count": 0}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
                state["count"] += 1
            time.sleep(0.2)
            response = {
                "object": "list",
                "model": body["model"],
                "data": [
                    {"object": "embedding", "index": i, "embedding": [3.0, 4.0]} for i, _ in enumerate(body["input"])
                ],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            with lock:
                state["active"] -= 1

        def log_message(self, format: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    profile = replace(resolve_embedding_profile(), dimensions=2)
    adapter = DSPyEmbeddingAdapter(profile, base_url=f"http://127.0.0.1:{server.server_port}/v1", api_key="test")
    try:

        async def exercise():
            with ThreadPoolExecutor(max_workers=2) as pool:
                loop = asyncio.get_running_loop()
                sync = [loop.run_in_executor(pool, adapter.batch_encode, f"sync {i}") for i in range(2)]
                await asyncio.gather(*sync, *(adapter.acall(f"async {i}") for i in range(2)))

        asyncio.run(exercise())
        assert state == {"active": 0, "peak": 2, "count": 4}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_cancelled_async_waiter_returns_its_permit():
    adapter = DSPyEmbeddingAdapter(replace(resolve_embedding_profile(), dimensions=2))
    adapter._admission.acquire()
    adapter._admission.acquire()

    async def exercise():
        waiter = asyncio.create_task(adapter.acall("cancelled"))
        await asyncio.sleep(0.05)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        adapter._admission.release()
        await asyncio.sleep(0.05)

    try:
        asyncio.run(exercise())
        assert adapter._admission.acquire(timeout=1)
        adapter._admission.release()
    finally:
        adapter._admission.release()


def test_native_cache_separates_endpoint_and_formatted_prefix(embedding_server):
    import dspy

    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=True)
    requests, base_url = embedding_server
    profile = replace(resolve_embedding_profile(), dimensions=2)
    first = DSPyEmbeddingAdapter(profile, base_url=base_url, api_key="fixture-key")
    second = DSPyEmbeddingAdapter(profile, base_url=base_url.replace("127.0.0.1", "localhost"), api_key="fixture-key")
    first.batch_encode("cache identity proof")
    first.batch_encode("cache identity proof")
    assert len(requests) == 1
    first.update_prefixes(document_prefix="different document: ")
    first.batch_encode("cache identity proof")
    assert len(requests) == 2
    second.batch_encode("cache identity proof")
    assert len(requests) == 3
    assert first.index_identity != second.index_identity


def test_injected_hosted_model_must_match_profile():
    profile = resolve_embedding_profile()
    wrong = create_dspy_embedder("different-model", base_url="http://127.0.0.1:10001/v1")
    with pytest.raises(ValueError, match="does not match"):
        DSPyEmbeddingAdapter(profile, wrong)


def test_large_finite_vectors_normalize_without_overflow():
    from embedding_fixture import IdentifiedEmbeddingCallable

    profile = replace(resolve_embedding_profile(), dimensions=2)
    model = IdentifiedEmbeddingCallable(
        lambda texts: np.tile(np.array([1e30, 1e30], dtype=np.float32), (len(texts), 1)),
        "large-finite-fixture",
    )
    adapter = DSPyEmbeddingAdapter(profile, create_dspy_embedder(model, caching=False))
    for vectors in (adapter.batch_encode(["sync"]), asyncio.run(adapter.abatch_encode(["async"]))):
        assert vectors.dtype == np.float32
        assert np.isfinite(vectors).all()
        np.testing.assert_allclose(vectors, [[2**-0.5, 2**-0.5]], rtol=1e-6)


def test_sync_and_async_embedding_requests_never_exceed_64(embedding_server):
    requests, base_url = embedding_server
    profile = replace(resolve_embedding_profile(), dimensions=2)
    embedder = create_dspy_embedder(profile, base_url=base_url, api_key="fixture-key", batch_size=128, caching=False)
    adapter = DSPyEmbeddingAdapter(profile, embedder, max_batch_chars=100_000)
    texts = [f"input {i}" for i in range(130)]
    assert adapter.batch_encode(texts, batch_size=128).shape == (130, 2)
    assert asyncio.run(adapter.abatch_encode(texts, batch_size=128)).shape == (130, 2)
    assert [len(request[2]["input"]) for request in requests] == [64, 64, 2, 64, 64, 2]
