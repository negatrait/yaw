"""Real HTTP, native DSPy and SQLite checks for bounded indexing overlap."""

import asyncio
import json
import subprocess
import sys
import threading
import time
from collections.abc import AsyncGenerator
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import cocoindex as coco
import numpy as np
import pytest
from cocoindex.connectors import sqlite

from daemon.graph.synonym_protocol import candidate_rows, complete_decisions
from yaw.daemon.graph.inference import (
    DSPyEmbeddingAdapter,
    EmbeddingProfile,
    HippoRAGProgram,
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
from yaw.daemon.graph.planner import GraphPlanSettings
from yaw.daemon.graph.storage import open_managed_yaw_db
from yaw.daemon.graph.storage.publication import GRAPH_DB, GraphAction, GraphTargetHandler, PreparedGraphRows
from yaw.daemon.indexing import (
    RELOAD_SIGNAL_HANDLER,
    RELOAD_SIGNAL_ROOT,
    InferenceIdentity,
    get_graph_delta_coordinator,
    indexing_app_main,
)
from yaw.daemon.ingestion.source_targets import SOURCE_DB, SOURCE_ROOT, SourceSettings


def test_frozen_candidate_replay_uses_native_targets_and_original_text(tmp_path, provider):
    from yaw.core import ChunkPointer
    from yaw.daemon.graph.engine import YawGraphEngine
    from yaw.daemon.indexing import bind_native_indexing

    url, state = provider
    source = tmp_path / "source.md"
    original = "Ada studies Algebra."
    source.write_text("Changed after the evaluation snapshot was frozen.", encoding="utf-8")
    pointer = ChunkPointer("frozen", source.name, 1, 1, "source", 1, 1)
    lm = create_dspy_lm("overlap", base_url=url + "/v1", cache=False, num_retries=0, max_tokens=512)
    program = HippoRAGProgram(orchestrator_lm=lm, rlm_sub_lm=lm, enable_gbnf=False, max_output_tokens=512)
    engine = bind_native_indexing(YawGraphEngine(tmp_path / "yaw.db", tmp_path, program, adapter(url)))
    try:
        assert engine.index_chunks([(pointer, original)]) == 1
        assert [row[0] for row in engine.conn.execute("SELECT chunk_id FROM chunks")] == ["frozen"]
        assert engine.conn.execute("SELECT COUNT(*) FROM chunk_inference").fetchone()[0] == 1
        assert engine.conn.execute("SELECT COUNT(*) FROM openie_chunks").fetchone()[0] == 0
        traces = engine._captured_chunk_traces["frozen"]
        assert {name for name, _, _ in traces} == {"ner", "triples"}
        assert all(inputs["passage"] == original for _, inputs, _ in traces)
        calls = len(state["intervals"])
        assert calls > 0
        assert engine.index_chunks([(pointer, original)]) == 1
        assert len(state["intervals"]) == calls, (state["intervals"][calls:], state["embedding_inputs"])
        assert engine.lint_graph()["conforms"]
    finally:
        engine.close()


@pytest.mark.parametrize("title_aware", [False, True])
def test_native_file_pipeline_owns_inference_rows(tmp_path, provider, title_aware):
    url, state = provider
    root = tmp_path / "documents"
    root.mkdir()
    source = root / "source.md"
    source.write_text("Ada studies Algebra.", encoding="utf-8")
    db = sqlite.connect(tmp_path / "yaw.db", load_vec=False)
    lm = create_dspy_lm("overlap", base_url=url + "/v1", cache=False, num_retries=0, max_tokens=512)
    program = HippoRAGProgram(rlm_sub_lm=lm, max_output_tokens=512)
    context = coco.ContextProvider()
    context.provide(SOURCE_DB, db)
    context.provide(SOURCE_ROOT, root)
    embedder = DSPyEmbeddingAdapter(
        replace(adapter(url).profile, parse_document_title=title_aware), base_url=url + "/v1", batch_size=64
    )
    context.provide(EMBEDDING_ENDPOINT, EmbeddingEndpoint(embedder, asyncio.Semaphore(1)))
    endpoint = OpenIEEndpoint(program, lm, "fixture-v1", asyncio.Semaphore(1))
    context.provide(OPENIE_ENDPOINT, endpoint)
    env = coco.Environment(settings=coco.Settings.from_env(tmp_path / "state"), context_provider=context)
    app = coco.App(
        coco.AppConfig(name="file_pipeline", environment=env, max_inflight_components=4),
        indexing_app_main,
        SourceSettings(("**/*.md",), (), min_chunk_size=1),
        InferenceIdentity("fixture-v1", embedder.profile.index_identity),
    )
    try:
        app.update_blocking()
        with db.readonly() as conn:
            rows = conn.execute("SELECT chunk_id, extraction_json, traces_json FROM chunk_inference").fetchall()
            assert len(rows) == 1
            assert conn.execute("SELECT chunk_id FROM source_chunks").fetchall() == [(rows[0][0],)]
            assert "vector_f32" not in {row[1] for row in conn.execute("PRAGMA table_info(chunk_inference)")}
        assert json.loads(rows[0][1])["entities"]
        assert len(json.loads(rows[0][2])) == 2
        calls = len(state["intervals"])
        assert calls == 3
        if title_aware:
            assert "source" in state["embedding_inputs"][0]
        else:
            assert state["embedding_inputs"] == ["text: Ada studies Algebra."]
        app.update_blocking()
        assert len(state["intervals"]) == calls
        endpoint.program_identity = "fixture-v2"
        env.close()
        env = coco.Environment(settings=coco.Settings.from_env(tmp_path / "state"), context_provider=context)
        app = coco.App(
            coco.AppConfig(name="file_pipeline", environment=env, max_inflight_components=4),
            indexing_app_main,
            SourceSettings(("**/*.md",), (), min_chunk_size=1),
            InferenceIdentity("fixture-v2", embedder.profile.index_identity),
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


def test_native_inference_survives_process_exit_before_graph_publication(tmp_path, provider):
    url, state = provider
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "source.md").write_text("Ada studies Algebra.", encoding="utf-8")
    script = r"""
import asyncio
import os
import sys
from pathlib import Path
import cocoindex as coco
from cocoindex.connectors import sqlite
from yaw.daemon.graph.inference import DSPyEmbeddingAdapter, EmbeddingProfile, HippoRAGProgram, create_dspy_lm
from yaw.daemon.graph.inference.components import EMBEDDING_ENDPOINT, OPENIE_ENDPOINT, EmbeddingEndpoint, OpenIEEndpoint
from yaw.daemon.indexing import InferenceIdentity, indexing_app_main
from yaw.daemon.ingestion.source_targets import SOURCE_DB, SOURCE_ROOT, SourceSettings

root = Path(sys.argv[1])
url = sys.argv[2]
db = sqlite.connect(root / 'yaw.db', load_vec=False)
profile = EmbeddingProfile('overlap', 'fixture', 4, 2048, 'mean', 'query: ', 'text: ')
embedder = DSPyEmbeddingAdapter(profile, base_url=url + '/v1', batch_size=64)
lm = create_dspy_lm('overlap', base_url=url + '/v1', cache=False, max_tokens=512)
program = HippoRAGProgram(rlm_sub_lm=lm, max_output_tokens=512)
context = coco.ContextProvider()
context.provide(SOURCE_DB, db)
context.provide(SOURCE_ROOT, root / 'documents')
context.provide(EMBEDDING_ENDPOINT, EmbeddingEndpoint(embedder, asyncio.Semaphore(1)))
context.provide(OPENIE_ENDPOINT, OpenIEEndpoint(program, lm, 'fixture-v1', asyncio.Semaphore(1)))
env = coco.Environment(settings=coco.Settings.from_env(root / 'state'), context_provider=context)
app = coco.App(coco.AppConfig(name='restart', environment=env, max_inflight_components=4),
               indexing_app_main, SourceSettings(('**/*.md',), (), min_chunk_size=1),
               InferenceIdentity('fixture-v1', profile.index_identity))
app.update_blocking()
with db.readonly() as conn:
    assert conn.execute('SELECT COUNT(*) FROM chunk_inference').fetchone()[0] == 1
    assert 'vector_f32' not in {row[1] for row in conn.execute('PRAGMA table_info(chunk_inference)')}
if sys.argv[3] == 'interrupt':
    os._exit(23)
env.close()
db.close()
"""
    for mode, expected_code in (("interrupt", 23), ("resume", 0)):
        result = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path), url, mode],
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == expected_code, result.stderr
        assert len(state["intervals"]) == 3
        assert sum(kind == "embedding" for kind, _, _ in state["intervals"]) == 1


@coco.fn
async def inference_fixture(embedding: EmbeddingRequest, extraction: OpenIEInput) -> tuple[EmbeddedText, ExtractedText]:
    vector, record = await asyncio.gather(embed_text(embedding), extract_text(extraction))
    return vector, record


def test_native_openie_overlaps_embedding_and_restores_traces(tmp_path, provider):
    url, state = provider
    state["rendezvous"] = True
    lm = create_dspy_lm("overlap", base_url=url + "/v1", cache=False, num_retries=0, max_tokens=512)
    program = HippoRAGProgram(rlm_sub_lm=lm, max_output_tokens=512)
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
                with lock:
                    state.setdefault("embedding_inputs", []).extend(data["input"])
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
                        "decisions": complete_decisions(
                            candidate_rows(text),
                            [
                                {
                                    "pair_id": p["pair_id"],
                                    "relation": "skos:related",
                                    "canonical": p["pair_id"].split("|")[0],
                                }
                                for p in packet["pairs"]
                            ],
                            stage_text=json.dumps(data["messages"]),
                        )
                    }
                elif "missing_entities" in text:
                    output = (
                        {"entity_definitions": []}
                        if state["mode"] == "shacl_unrepaired"
                        else {"entity_definitions": [["London", "Capital of England"]]}
                    )
                elif "violations" in text:
                    all_msg_text = json.dumps(data["messages"])
                    output = (
                        {"reasoning": "Leave violation unrepaired.", "code": "SUBMIT(repairs=[])"}
                        if '"code"' in all_msg_text or "`code`" in all_msg_text
                        else {"repairs": []}
                    )
                elif "[[ ## named_entities ## ]]" in text:
                    if state["mode"] == "shacl_unrepaired" and "London" in text:
                        output = {
                            "triples": [
                                {"sub": "Ada", "pred": "visited", "obj": "London", "def": "Ada visited London."}
                            ]
                        }
                    else:
                        output = {
                            "triples": [
                                {"sub": "Ada", "pred": "studies", "obj": "Algebra", "def": "Ada studies algebra."}
                            ]
                        }
                elif state["mode"] == "shacl_unrepaired" and "London" in text:
                    output = {"named_entities": [["Ada", "A mathematician"]]}
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
            server.server_close()
            worker.join(timeout=5)
            assert not worker.is_alive()


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


def test_declarative_live_graph_publication_without_polling(tmp_path, provider):
    url, state = provider
    state["delay"] = 0.01
    root = tmp_path / "documents"
    root.mkdir()
    source = root / "source.md"
    source.write_text("Ada studies Algebra.", encoding="utf-8")

    db = open_managed_yaw_db(tmp_path / "yaw.db")
    lm = create_dspy_lm("overlap", base_url=url + "/v1", cache=False, num_retries=0, max_tokens=512)
    program = HippoRAGProgram(rlm_sub_lm=lm, max_output_tokens=512)
    embedder = adapter(url)

    signal_dir = tmp_path / ".yaw"
    signal_dir.mkdir()
    reload_payloads: list[str] = []
    reload_event = threading.Event()

    def on_reload_signal(raw: str) -> None:
        reload_payloads.append(raw.strip())
        reload_event.set()

    context = coco.ContextProvider()
    context.provide(SOURCE_DB, db)
    context.provide(GRAPH_DB, db)
    context.provide(SOURCE_ROOT, root)
    context.provide(RELOAD_SIGNAL_ROOT, signal_dir)
    context.provide(RELOAD_SIGNAL_HANDLER, on_reload_signal)
    context.provide(EMBEDDING_ENDPOINT, EmbeddingEndpoint(embedder, asyncio.Semaphore(2)))
    context.provide(OPENIE_ENDPOINT, OpenIEEndpoint(program, lm, "fixture-v1", asyncio.Semaphore(2)))

    graph_settings = GraphPlanSettings(
        corpus_root=root,
        program_identity="fixture-v1",
        embedding_identity=embedder.profile.index_identity,
        synonymy_top_k=0,
    )
    env = coco.Environment(settings=coco.Settings.from_env(tmp_path / "state"), context_provider=context)
    app = coco.App(
        coco.AppConfig(name="live_graph_pipeline", environment=env, max_inflight_components=4),
        indexing_app_main,
        SourceSettings(("**/*.md",), (), min_chunk_size=1),
        InferenceIdentity("fixture-v1", embedder.profile.index_identity),
        graph_settings,
        True,
    )

    async def run_live_cycle() -> None:
        handle = app.update(live=True)
        events = handle.watch()
        assert isinstance(events, AsyncGenerator)
        ready = False

        async def wait_for_published_count(expected_count: int) -> None:
            nonlocal ready
            async with asyncio.timeout(20):
                while True:
                    if not ready:
                        event = await anext(events)
                        assert event.stats.total.num_errors == 0
                        ready = ready or event.status == coco.UpdateStatus.READY
                        if not ready:
                            continue
                    with db.readonly() as conn:
                        count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
                        vec_count = conn.execute("SELECT COUNT(*) FROM chunk_vec").fetchone()[0]
                    if count == expected_count and vec_count == expected_count:
                        return
                    await asyncio.sleep(0.02)

        try:
            await wait_for_published_count(1)
            coordinator = get_graph_delta_coordinator(db)
            assert coordinator.last_plan_summary is not None
            assert len(coordinator.last_plan_summary.chunk_ids) == 1
            assert {t.predictor for t in coordinator.last_plan_summary.traces} == {"ner", "triples", "entity_definer"}

            (signal_dir / "reload.signal").write_text('{"program_path": "compiled.json"}', encoding="utf-8")
            assert await asyncio.to_thread(reload_event.wait, 10.0)
            assert reload_payloads == ['{"program_path": "compiled.json"}']

            second = root / "second.md"
            second.write_text("Ada studies Algebra in London.", encoding="utf-8")
            await wait_for_published_count(2)

            second.unlink()
            await wait_for_published_count(1)

            source.unlink()
            await wait_for_published_count(0)

            await asyncio.wait_for(app.update().result(), timeout=10)
            await asyncio.wait_for(handle.result(), timeout=10)
        finally:
            await events.aclose()

    try:
        asyncio.run(run_live_cycle())
    finally:
        env.close()
        db.close()


def test_publish_graph_shacl_violation_preserves_prior_graph_and_inference_rows(tmp_path, provider):
    url, state = provider
    state["delay"] = 0.01
    root = tmp_path / "documents"
    root.mkdir()
    source = root / "source.md"
    source.write_text("Ada studies Algebra.", encoding="utf-8")

    db = open_managed_yaw_db(tmp_path / "yaw.db")
    lm = create_dspy_lm("overlap", base_url=url + "/v1", cache=False, num_retries=0, max_tokens=512)
    program = HippoRAGProgram(rlm_sub_lm=lm, max_output_tokens=512)
    embedder = adapter(url)

    context = coco.ContextProvider()
    context.provide(SOURCE_DB, db)
    context.provide(GRAPH_DB, db)
    context.provide(SOURCE_ROOT, root)
    context.provide(EMBEDDING_ENDPOINT, EmbeddingEndpoint(embedder, asyncio.Semaphore(2)))
    context.provide(OPENIE_ENDPOINT, OpenIEEndpoint(program, lm, "fixture-v1", asyncio.Semaphore(2)))

    graph_settings = GraphPlanSettings(
        corpus_root=root,
        program_identity="fixture-v1",
        embedding_identity=embedder.profile.index_identity,
        synonymy_top_k=0,
    )
    env = coco.Environment(settings=coco.Settings.from_env(tmp_path / "state"), context_provider=context)
    app = coco.App(
        coco.AppConfig(name="shacl_rollback_pipeline", environment=env, max_inflight_components=4),
        indexing_app_main,
        SourceSettings(("**/*.md",), (), min_chunk_size=1),
        InferenceIdentity("fixture-v1", embedder.profile.index_identity),
        graph_settings,
    )
    try:
        app.update_blocking()
        with db.readonly() as conn:
            assert conn.execute("SELECT COUNT(*) FROM source_chunks").fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM chunk_inference").fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM chunk_vec").fetchone()[0] == 1
            initial_entities = conn.execute("SELECT entity_id, name, definition FROM entities").fetchall()

        # Verify GraphTargetHandler.apply_actions rolls back transaction on SHACL violation
        invalid_rows = PreparedGraphRows(
            chunks=(),
            entities=(("entity-invalid", "broken", ""),),
            entity_labels=(("broken", "entity-invalid", "skos:prefLabel", "Broken", ""),),
            facts=(),
            fact_sources=(),
            edges=(),
            chunk_vectors=(),
            entity_vectors=(),
            fact_vectors=(),
        )
        with pytest.raises(ValueError, match="SHACL violations"):
            GraphTargetHandler().apply_actions(context, [GraphAction(invalid_rows, (), ())])
        with db.readonly() as conn:
            assert conn.execute("SELECT entity_id, name, definition FROM entities").fetchall() == initial_entities

        # Verify publish_graph failure on SHACL violation keeps source_chunks & chunk_inference intact
        # while preserving prior published graph rows
        state["mode"] = "shacl_unrepaired"
        second = root / "second.md"
        second.write_text("Ada visited London.", encoding="utf-8")
        app.update_blocking()
        err = get_graph_delta_coordinator(db).pop_error()
        assert err is not None and "SHACL violations" in str(err)

        with db.readonly() as conn:
            assert conn.execute("SELECT COUNT(*) FROM source_chunks").fetchone()[0] == 2
            assert conn.execute("SELECT COUNT(*) FROM chunk_inference").fetchone()[0] == 2
            assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM chunk_vec").fetchone()[0] == 1
            assert conn.execute("SELECT entity_id, name, definition FROM entities").fetchall() == initial_entities

        # Restoring valid repair mode publishes the 2nd chunk using existing chunk_inference rows
        state["mode"] = "valid"
        get_graph_delta_coordinator(db).recovery_nonce = "recover-1"
        app.update_blocking()
        assert get_graph_delta_coordinator(db).pop_error() is None
        with db.readonly() as conn:
            assert conn.execute("SELECT COUNT(*) FROM source_chunks").fetchone()[0] == 2
            assert conn.execute("SELECT COUNT(*) FROM chunk_inference").fetchone()[0] == 2
            assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 2
            assert conn.execute("SELECT COUNT(*) FROM chunk_vec").fetchone()[0] == 2
    finally:
        env.close()
        db.close()


def test_frozen_candidate_streaming_batches(tmp_path, provider):
    from yaw.core import ChunkPointer
    from yaw.daemon.graph.engine import YawGraphEngine
    from yaw.daemon.indexing import bind_native_indexing

    url, state = provider
    lm = create_dspy_lm("overlap", base_url=url + "/v1", cache=False, num_retries=0, max_tokens=512)
    program = HippoRAGProgram(orchestrator_lm=lm, rlm_sub_lm=lm, enable_gbnf=False, max_output_tokens=512)
    engine = bind_native_indexing(
        YawGraphEngine(tmp_path / "yaw.db", tmp_path, program, adapter(url)),
        batch_size=2,
    )
    chunks = [
        (ChunkPointer(f"c{i}", f"doc{i}.md", 1, 1, f"doc{i}", 1, 1), f"Entity{i} relates to Topic{i}.")
        for i in range(5)
    ]
    try:
        indexed = engine.index_chunks(chunks)
        assert indexed == 5
        assert engine.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 5
        assert engine.conn.execute("SELECT COUNT(*) FROM chunk_inference").fetchone()[0] == 5
        assert engine.conn.execute("SELECT COUNT(*) FROM source_chunks").fetchone()[0] == 5
        assert engine.conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0] == 2
        assert engine.lint_graph()["conforms"]
    finally:
        engine.close()
