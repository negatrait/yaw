"""Real HTTP, native DSPy and SQLite checks for bounded indexing overlap."""

import asyncio
import json
import subprocess
import sys
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import cocoindex as coco
import dspy
import litellm
import numpy as np
import pytest
from cocoindex.connectors import sqlite

from yaw.core import ChunkPointer
from yaw.core.config import YawConfig
from yaw.daemon.graph.engine import YawGraphEngine
from yaw.daemon.graph.inference import (
    DSPyEmbeddingAdapter,
    EmbeddingLane,
    EmbeddingProfile,
    HippoRAGProgram,
    ServingTokenCounter,
    create_dspy_lm,
)
from yaw.daemon.graph.inference.components import (
    EMBEDDING_ENDPOINT,
    OPENIE_ENDPOINT,
    EmbeddedText,
    EmbeddingEndpoint,
    ExtractedText,
    OpenIEEndpoint,
    OpenIEInput,
    embed_text,
    extract_text,
)
from yaw.daemon.graph.inference.embedding import EmbeddingRequest
from yaw.daemon.graph.storage import CandidateCacheStore, CandidateStageCache, open_managed_yaw_db
from yaw.daemon.indexing import InferenceIdentity, indexing_app_main
from yaw.daemon.ingestion.source_targets import SOURCE_DB, SOURCE_ROOT, SourceSettings
from yaw.daemon.runtime import YawRuntime


def test_native_file_pipeline_owns_inference_rows(tmp_path, provider):
    url, state = provider
    root = tmp_path / "documents"
    root.mkdir()
    source = root / "source.md"
    source.write_text("Ada studies Algebra.", encoding="utf-8")
    db = sqlite.connect(tmp_path / "yaw.db", load_vec=False)
    lm = create_dspy_lm("overlap", base_url=url + "/v1", cache=False, num_retries=0, max_tokens=512)
    program = HippoRAGProgram(rlm_sub_lm=lm, max_output_tokens=512)
    program.max_invariant_retries = 0
    context = coco.ContextProvider()
    context.provide(SOURCE_DB, db)
    context.provide(SOURCE_ROOT, root)
    context.provide(EMBEDDING_ENDPOINT, EmbeddingEndpoint(adapter(url), asyncio.Semaphore(1)))
    endpoint = OpenIEEndpoint(program, lm, "fixture-v1", asyncio.Semaphore(1))
    context.provide(OPENIE_ENDPOINT, endpoint)
    env = coco.Environment(settings=coco.Settings.from_env(tmp_path / "state"), context_provider=context)
    app = coco.App(
        coco.AppConfig(name="file_pipeline", environment=env, max_inflight_components=4),
        indexing_app_main,
        SourceSettings(("**/*.md",), (), min_chunk_size=1),
        InferenceIdentity("fixture-v1", adapter(url).profile.index_identity),
    )
    try:
        app.update_blocking()
        with db.readonly() as conn:
            rows = conn.execute("SELECT chunk_id, extraction_json, traces_json FROM chunk_inference").fetchall()
            assert len(rows) == 1
            assert conn.execute("SELECT chunk_id FROM source_chunks").fetchall() == [(rows[0][0],)]
        assert json.loads(rows[0][1])["entities"]
        assert len(json.loads(rows[0][2])) == 2
        calls = len(state["intervals"])
        assert calls == 3
        app.update_blocking()
        assert len(state["intervals"]) == calls
        endpoint.program_identity = "fixture-v2"
        env.close()
        env = coco.Environment(settings=coco.Settings.from_env(tmp_path / "state"), context_provider=context)
        app = coco.App(
            coco.AppConfig(name="file_pipeline", environment=env, max_inflight_components=4),
            indexing_app_main,
            SourceSettings(("**/*.md",), (), min_chunk_size=1),
            InferenceIdentity("fixture-v2", adapter(url).profile.index_identity),
        )
        app.update_blocking()
        assert len(state["intervals"]) == calls + 2
        assert sum(span[0] == "embedding" for span in state["intervals"]) == 1
        with db.readonly() as conn:
            assert conn.execute("SELECT program_identity FROM chunk_inference").fetchall() == [("fixture-v2",)]
        calls += 2
        source.unlink()
        app.update_blocking()
        with db.readonly() as conn:
            assert conn.execute("SELECT COUNT(*) FROM source_chunks").fetchone()[0] == 0
            assert conn.execute("SELECT COUNT(*) FROM chunk_inference").fetchone()[0] == 0
        assert len(state["intervals"]) == calls
    finally:
        env.close()
        db.close()


@coco.fn
async def embedding_fixture(requests: tuple[EmbeddingRequest, ...]) -> tuple[EmbeddedText, ...]:
    return tuple(await coco.map(embed_text, requests))


@coco.fn
async def inference_fixture(embedding: EmbeddingRequest, extraction: OpenIEInput) -> tuple[EmbeddedText, ExtractedText]:
    vector, record = await asyncio.gather(embed_text(embedding), extract_text(extraction))
    return vector, record


def test_native_openie_overlaps_embedding_and_restores_traces(tmp_path, provider):
    url, state = provider
    state["rendezvous"] = True
    lm = create_dspy_lm("overlap", base_url=url + "/v1", cache=False, num_retries=0, max_tokens=512)
    program = HippoRAGProgram(rlm_sub_lm=lm, max_output_tokens=512)
    program.max_invariant_retries = 0
    embedder = adapter(url)
    context = coco.ContextProvider()
    context.provide(EMBEDDING_ENDPOINT, EmbeddingEndpoint(embedder, asyncio.Semaphore(1)))
    context.provide(OPENIE_ENDPOINT, OpenIEEndpoint(program, lm, "fixture-v1", asyncio.Semaphore(1)))
    env = coco.Environment(settings=coco.Settings.from_env(tmp_path / "state"), context_provider=context)
    app = coco.App(
        coco.AppConfig(name="inference_fixture", environment=env, max_inflight_components=4),
        inference_fixture,
        embedder.prepare_request(["Ada studies Algebra."], batch_size=64, stage="chunk_embed"),
        OpenIEInput("chunk", "Ada studies Algebra.", "fixture-v1"),
    )
    try:
        first = app.update_blocking()
        spans = list(state["intervals"])
        embeds = [span for span in spans if span[0] == "embedding"]
        chats = [span for span in spans if span[0] == "chat"]
        assert len(embeds) == 1
        assert len(chats) == 2
        assert any(max(embeds[0][1], chat[1]) < min(embeds[0][2], chat[2]) for chat in chats)
        assert first[1].record.chunk_id == "chunk"
        assert {trace.predictor for trace in first[1].traces} == {"ner", "triples"}
        assert not program.telemetry
        assert app.update_blocking() == first
        assert state["intervals"] == spans
    finally:
        env.close()


def test_native_embedding_batches_and_reuses_persistent_results(tmp_path, provider):
    url, state = provider
    embedder = adapter(url)
    requests = tuple(
        embedder.prepare_request([f"Grounded source {index}."], batch_size=64, stage="chunk_embed")
        for index in range(4)
    )
    first = None
    calls = 0
    for _ in range(2):
        context = coco.ContextProvider()
        context.provide(EMBEDDING_ENDPOINT, EmbeddingEndpoint(embedder, asyncio.Semaphore(1)))
        env = coco.Environment(settings=coco.Settings.from_env(tmp_path / "native-state"), context_provider=context)
        app = coco.App(
            coco.AppConfig(name="embedding_fixture", environment=env, max_inflight_components=4),
            embedding_fixture,
            requests,
        )
        try:
            result = app.update_blocking()
            assert len(result) == 4
            assert all(isinstance(item, EmbeddedText) for item in result)
            assert len({item.request_id for item in result}) < len(requests)
            if first is None:
                first = result
                calls = len(state["intervals"])
            else:
                assert result == first
            assert len(state["intervals"]) == calls
            assert not embedder.telemetry
        finally:
            env.close()


@pytest.fixture
def provider():
    state: dict[str, Any] = {"mode": "valid", "intervals": [], "delay": 0.15, "embedding_started": threading.Event()}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def send(self, body):
            raw = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            try:
                self.wfile.write(raw)
            except BrokenPipeError, ConnectionResetError:
                # The timeout case intentionally disconnects before the response.
                self.close_connection = True

        def do_GET(self):
            if self.path.endswith("health"):
                self.send(
                    {
                        "all_models_loaded": [
                            {"id": "overlap", "backend_url": url, "launch_command": ["-m", "overlap.gguf"]}
                        ]
                    }
                )
            else:
                self.send(
                    {
                        "model_path": "overlap.gguf",
                        "chat_template": "bytes",
                        "default_generation_settings": {"n_ctx": 131072},
                    }
                )

        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path.endswith("apply-template"):
                self.send({"prompt": json.dumps(data["messages"])})
                return
            if self.path.endswith("tokenize"):
                self.send({"tokens": list(data["content"].encode())})
                return
            started = time.perf_counter()
            kind = "embedding" if self.path.endswith("embeddings") else "chat"
            if kind == "embedding":
                state["embedding_started"].set()
            elif state.get("rendezvous"):
                assert state["embedding_started"].wait(10), "Embedding request did not overlap the chat request"
            time.sleep(state["delay"])
            if kind == "embedding":
                rows: list[dict[str, Any]] = [
                    {"object": "embedding", "index": i, "embedding": [1.0, 2.0, 3.0, 4.0]}
                    for i, _ in enumerate(data["input"])
                ]
                if state["mode"] == "count":
                    rows = rows[:-1]
                elif state["mode"] == "dimensions":
                    rows[0]["embedding"] = [1.0]
                elif state["mode"] == "nan":
                    rows[0]["embedding"][0] = float("nan")
                body = {"object": "list", "data": rows, "model": "overlap", "usage": {"total_tokens": 7}}
            else:
                text = data["messages"][-1]["content"]
                if "candidate_pairs" in text:
                    packet = json.JSONDecoder().raw_decode(text[text.index('{"format":') :])[0]
                    output = {
                        "decisions": [
                            {
                                "pair_id": p["pair_id"],
                                "relation": "skos:related",
                                "canonical": p["pair_id"].split("|")[0],
                            }
                            for p in packet["pairs"]
                        ]
                    }
                elif "[[ ## named_entities ## ]]" in text:
                    output = {
                        "triples": [{"sub": "Ada", "pred": "studies", "obj": "Algebra", "def": "Ada studies algebra."}]
                    }
                else:
                    output = {"named_entities": [["Ada", "A mathematician"], ["Algebra", "A mathematical discipline"]]}
                if state["mode"] == "chat":
                    output = {"wrong": "invalid output"}
                body = {
                    "id": "overlap",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "overlap",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": json.dumps(output)},
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                }
            with lock:
                state["intervals"].append((kind, started, time.perf_counter()))
            self.send(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        url = f"http://127.0.0.1:{server.server_port}"
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            yield url, state
        finally:
            server.shutdown()
            worker.join()


def adapter(url):
    profile = EmbeddingProfile("overlap", "fixture", 4, 2048, "mean", "query: ", "text: ")
    return DSPyEmbeddingAdapter(profile, base_url=url + "/v1", batch_size=64)


def test_async_embedding_request_matches_sync_without_publishing(provider):
    url, state = provider
    embedder = adapter(url)
    request = embedder.prepare_request(
        ["Ada studies algebra.", "Euler studies graphs."], batch_size=64, stage="chunk_embed"
    )
    expected = embedder.execute_request(request)
    actual = asyncio.run(embedder.aexecute_request(request))
    np.testing.assert_array_equal(actual.vectors, expected.vectors)
    assert actual.telemetry["batch_sizes"] == expected.telemetry["batch_sizes"]
    assert len(state["intervals"]) == 2
    assert not embedder.telemetry


@pytest.mark.parametrize("mode", ["count", "dimensions", "nan"])
def test_async_embedding_request_rejects_invalid_provider_vectors(provider, mode):
    url, state = provider
    state["mode"] = mode
    embedder = adapter(url)
    request = embedder.prepare_request(["Ada studies algebra."], batch_size=64, stage="chunk_embed")
    with pytest.raises(ValueError):
        asyncio.run(embedder.aexecute_request(request))
    assert not embedder.telemetry


def engine(root, url, overlap):
    root.mkdir(exist_ok=True)
    lm = create_dspy_lm("overlap", base_url=url + "/v1", cache=False, num_retries=0, max_tokens=512)
    program = HippoRAGProgram(
        rlm_sub_lm=lm, synonym_token_counter=ServingTokenCounter(url, "overlap"), max_output_tokens=512
    )
    program.max_invariant_retries = 0
    result = YawGraphEngine(db_path=root / "yaw.db", workspace_root=root, program=program, embedder=adapter(url))
    result.config = YawConfig(workspace_root=root, indexing_overlap=overlap)
    return result


def chunks(root):
    text = "Ada studies Algebra."
    (root / "document.md").write_text(text, encoding="utf-8")
    return [(ChunkPointer("chunk", "document.md", 1, 1, "Mathematics"), text)]


def test_overlap_matches_serial_graph_and_preserves_accounting(tmp_path, provider):
    url, state = provider
    graphs = []
    for overlap in (False, True):
        graph = engine(tmp_path / str(overlap), url, overlap)
        state["intervals"].clear()
        try:
            assert graph.index_chunks(chunks(graph.workspace_root)) == 1
            tables = ("chunks", "entities", "facts", "edges", "edge_sources", "entity_labels")
            graphs.append({table: sorted(map(tuple, graph.conn.execute(f"SELECT * FROM {table}"))) for table in tables})
            spans = state["intervals"]
            simultaneous = any(
                max(a, c) < min(b, d)
                for kind, a, b in spans
                for other, c, d in spans
                if kind == "embedding" and other == "chat"
            )
            assert simultaneous is overlap
            embeds = [(a, b) for k, a, b in spans if k == "embedding"]
            assert all(b <= c for (_, b), (c, _) in zip(embeds, embeds[1:], strict=False))
            stages = [t for t in graph.program.telemetry if t.get("stage") in ("ner", "triples")]
            assert stages and all(t["tokens_used"] == 15 for t in stages)
            assert graph.get_stats().shacl_conforms
        finally:
            graph.close()
    assert graphs[0] == graphs[1]
    assert not any(t.name.startswith("yaw-chunk-embedding") for t in threading.enumerate())


@pytest.mark.parametrize("mode", ["count", "dimensions", "nan", "chat"])
def test_failure_keeps_graph_atomic_and_closes_worker(tmp_path, provider, mode):
    url, state = provider
    graph = engine(tmp_path, url, True)
    state["mode"] = mode
    try:
        with pytest.raises((ValueError, dspy.AdapterParseError)):
            graph.index_chunks(chunks(tmp_path))
        assert graph.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
        assert graph.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 0
    finally:
        graph.close()
    assert not any(t.name.startswith("yaw-chunk-embedding") for t in threading.enumerate())


def test_lane_owner_publication_and_bounded_retention(tmp_path, provider):
    url, state = provider
    state["delay"] = 0
    embedder = adapter(url)
    owner = threading.get_ident()
    publication_threads = []
    embedder.bind_telemetry_hook(lambda _: publication_threads.append(threading.get_ident()))
    managed = open_managed_yaw_db(tmp_path / "cache.db")
    with managed.transaction() as conn:
        cache = CandidateStageCache(CandidateCacheStore(conn), managed, "embedding", lambda: "stable")
    try:
        for _ in range(12):
            with EmbeddingLane(
                embedder,
                ["a", "b", "a"],
                batch_size=1,
                lookup=lambda text: cache.get((text,)),
                store=lambda text, vector: cache.__setitem__((text,), vector),
            ) as lane:
                vectors = lane.finish()
                np.testing.assert_equal(vectors[0], vectors[2])
            assert not lane.vectors and not lane.missing and lane.future is None
        assert publication_threads == [owner, owner]
        assert len(state["intervals"]) == 2
    finally:
        managed.close()


def test_schedule_flag_does_not_change_index_identity(tmp_path):
    config = YawConfig(workspace_root=tmp_path)
    assert config.config_fingerprint() == replace(config, indexing_overlap=True).config_fingerprint()


def test_schedule_flag_preserves_optimization_resume_identity(tmp_path, provider):
    graph = engine(tmp_path, provider[0], False)
    runtime = YawRuntime(graph.config)
    runtime.engine = graph
    try:
        target = tmp_path / "program.json"
        runtime.validate_optimization_run(target, [], [], [], "gepa")
        before = graph.get_meta("optimization_run")
        runtime.config = replace(runtime.config, indexing_overlap=True)
        runtime.validate_optimization_run(target, [], [], [], "gepa")
        assert graph.get_meta("optimization_run") == before
        assert "indexing_overlap" not in json.loads(before)["identity"]["settings"]
    finally:
        graph.close()


def test_failed_owner_drains_valid_request_without_scheduling_more(tmp_path, provider):
    url, state = provider
    managed = open_managed_yaw_db(tmp_path / "cache.db")
    with managed.transaction() as conn:
        cache = CandidateStageCache(CandidateCacheStore(conn), managed, "embedding", lambda: "stable")
    try:
        with (
            pytest.raises(RuntimeError, match="OpenIE failed"),
            EmbeddingLane(
                adapter(url),
                ["a", "b", "c"],
                batch_size=1,
                lookup=lambda text: cache.get((text,)),
                store=lambda text, vector: cache.__setitem__((text,), vector),
            ),
        ):
            assert state["embedding_started"].wait(timeout=5)
            raise RuntimeError("OpenIE failed")
        assert cache.get(("a",)) is not None
        assert cache.get(("b",)) is None
        assert len(state["intervals"]) == 1
    finally:
        managed.close()
    assert not any(t.name.startswith("yaw-chunk-embedding") for t in threading.enumerate())


def test_provider_timeout_closes_lane(provider):
    url, _ = provider
    embedder = adapter(url)
    embedder.default_kwargs.update(timeout=0.03, num_retries=0)
    published = []
    with (
        pytest.raises(litellm.Timeout),
        EmbeddingLane(
            embedder,
            ["timeout"],
            batch_size=1,
            lookup=lambda _: None,
            store=lambda text, vector: published.append((text, vector)),
        ) as lane,
    ):
        lane.finish()
    assert published == []
    assert not any(t.name.startswith("yaw-chunk-embedding") for t in threading.enumerate())


def test_fresh_process_first_chat_and_embedding_calls_overlap(tmp_path, provider):
    url, state = provider
    script = """
import sys
from yaw.daemon.graph.inference import (
 DSPyEmbeddingAdapter, EmbeddingProfile, EmbeddingLane, create_dspy_lm,
)
profile=EmbeddingProfile('overlap','fixture',4,2048,'mean','query: ','text: ')
adapter=DSPyEmbeddingAdapter(profile,base_url=sys.argv[1]+'/v1')
lm=create_dspy_lm('overlap',base_url=sys.argv[1]+'/v1',cache=False,num_retries=0,max_tokens=32)
with EmbeddingLane(adapter,['first call'],batch_size=1,lookup=lambda _:None,store=lambda t,v:None) as lane:
 lm(messages=[{'role':'user','content':'Name the entities'}])
 assert len(lane.finish())==1
"""
    result = subprocess.run([sys.executable, "-c", script, url], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    spans = state["intervals"]
    assert any(
        max(a, c) < min(b, d)
        for kind, a, b in spans
        for other, c, d in spans
        if kind == "embedding" and other == "chat"
    )


def test_fresh_process_reuses_persisted_embeddings_after_interruption(tmp_path, provider):
    url, state = provider
    script = """
import sys
from pathlib import Path
from yaw.daemon.graph.inference import DSPyEmbeddingAdapter, EmbeddingProfile, EmbeddingLane
from yaw.daemon.graph.storage import CandidateCacheStore, CandidateStageCache, open_managed_yaw_db
profile = EmbeddingProfile('overlap','fixture',4,2048,'mean','query: ','text: ')
adapter = DSPyEmbeddingAdapter(profile,base_url=sys.argv[2]+'/v1')
db = open_managed_yaw_db(Path(sys.argv[1])/'cache.db')
with db.transaction() as conn:
 cache = CandidateStageCache(CandidateCacheStore(conn),db,'embedding',lambda:'stable')
try:
 with EmbeddingLane(adapter,['a','b'],batch_size=1,lookup=lambda t:cache.get((t,)),
                    store=lambda t,v:cache.__setitem__((t,),v)) as lane:
  assert len(lane.finish()) == 2
  if sys.argv[3]=='crash': raise RuntimeError('before graph commit')
except RuntimeError:
 if sys.argv[3]!='crash': raise
finally: db.close()
"""
    for mode in ("crash", "resume"):
        result = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path), url, mode], capture_output=True, text=True, timeout=60
        )
        assert result.returncode == 0, result.stderr
        assert len(state["intervals"]) == 2
