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
    YawProgram,
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
from yaw.daemon.graph.storage.db import GRAPH_DB
from yaw.daemon.graph.storage.managed_sqlite import VALIDATE_SQLITE
from yaw.daemon.graph.storage.rows import validate_graph_state
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
    program = YawProgram(orchestrator_lm=lm, rlm_sub_lm=lm, enable_gbnf=False, max_output_tokens=512)
    engine = bind_native_indexing(YawGraphEngine(tmp_path / "yaw.db", tmp_path, program, adapter(url)))
    try:
        assert engine.index_chunks([(pointer, original)]) == 1
        assert [row[0] for row in engine.conn.execute("SELECT chunk_id FROM chunks")] == ["frozen"]
        assert engine.conn.execute("SELECT COUNT(*) FROM chunk_inference").fetchone()[0] == 1
        assert engine.conn.execute("SELECT 1 FROM sqlite_master WHERE name='openie_chunks'").fetchone() is None
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
    program = YawProgram(rlm_sub_lm=lm, max_output_tokens=512)
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
from yaw.daemon.graph.inference import DSPyEmbeddingAdapter, EmbeddingProfile, YawProgram, create_dspy_lm
from yaw.daemon.graph.inference.components import EMBEDDING_ENDPOINT, OPENIE_ENDPOINT, EmbeddingEndpoint, OpenIEEndpoint
from yaw.daemon.indexing import InferenceIdentity, indexing_app_main
from yaw.daemon.ingestion.source_targets import SOURCE_DB, SOURCE_ROOT, SourceSettings

root = Path(sys.argv[1])
url = sys.argv[2]
db = sqlite.connect(root / 'yaw.db', load_vec=False)
profile = EmbeddingProfile('overlap', 'fixture', 4, 2048, 'mean', 'query: ', 'text: ')
embedder = DSPyEmbeddingAdapter(profile, base_url=url + '/v1', batch_size=64)
lm = create_dspy_lm('overlap', base_url=url + '/v1', cache=False, max_tokens=512)
program = YawProgram(rlm_sub_lm=lm, max_output_tokens=512)
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
    program = YawProgram(rlm_sub_lm=lm, max_output_tokens=512)
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
                    {
                        "object": "embedding",
                        "index": i,
                        "embedding": state["embedding_vector"](text)
                        if "embedding_vector" in state
                        else [1.0, 2.0, 3.0, 4.0],
                    }
                    for i, text in enumerate(data["input"])
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
                if "[[ ## fact_before_filter ## ]]" in text:
                    output = {"fact": state["selected_facts"]}
                elif "candidate_pairs" in text:
                    packet = json.JSONDecoder().raw_decode(text[text.index('{"format":') :])[0]
                    output = {
                        "decisions": complete_decisions(
                            candidate_rows(text),
                            [
                                {
                                    "pair_id": p["pair_id"],
                                    "relation": state.get("relation", "skos:related"),
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
                elif "extractions" in state and "[[ ## passage ## ]]" in text:
                    (passage,) = [source for source in state["extractions"] if source in text]
                    field = "triples" if "[[ ## named_entities ## ]]" in text else "named_entities"
                    output = {field: state["extractions"][passage][field]}
                    with lock:
                        state.setdefault("extraction_requests", []).append((field, passage))
                elif "violations" in text:
                    all_msg_text = json.dumps(data["messages"])
                    if "healed_definition" in state:
                        supervisor = '"code"' in all_msg_text or "`code`" in all_msg_text
                        state.setdefault("healer_requests", []).append("rlm" if supervisor else "direct")
                        if supervisor:
                            output = {
                                "reasoning": "Inspect the proposed statement and repair its description.",
                                "code": (
                                    "import json\n"
                                    "v = json.loads(violations)\n"
                                    "d = inspect_focus_node(v[0]['focus_node'])\n"
                                    "assert d['focus_kind'] in {'fact', 'entity'}\n"
                                    "repairs = propose_repairs()\n"
                                    "SUBMIT(repairs=repairs)"
                                ),
                            }
                        else:
                            state.setdefault("healer_contexts", []).append(
                                text.split("[[ ## focus_context ## ]]", 1)[1]
                            )
                            raw = text.split("[[ ## violations ## ]]", 1)[1].strip()
                            violations, _ = json.JSONDecoder().raw_decode(raw)
                            output = {
                                "repairs": [
                                    {
                                        "focus_node": v["focus_node"],
                                        "action": "set_pref_label"
                                        if v["source_shape"] == "ConceptShape/noScalarLiterals"
                                        else "set_definition",
                                        "value": "Rate limit responses"
                                        if v["source_shape"] == "ConceptShape/noScalarLiterals"
                                        else state["healed_definition"],
                                    }
                                    for v in violations
                                ]
                            }
                    else:
                        output = (
                            {"reasoning": "Leave violation unrepaired.", "code": "SUBMIT(repairs=[])"}
                            if '"code"' in all_msg_text or "`code`" in all_msg_text
                            else {"repairs": []}
                        )
                elif "[[ ## named_entities ## ]]" in text:
                    if "visited London" in text:
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
                elif "London" in text:
                    output = {"named_entities": [["Ada", "A mathematician"], ["London", "A city in England"]]}
                    if "Algebra" in text:
                        output["named_entities"].append(["Algebra", "A mathematical discipline"])
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
    program = YawProgram(rlm_sub_lm=lm, max_output_tokens=512)
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
    context.provide(VALIDATE_SQLITE, validate_graph_state)
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
            assert {t.predictor for t in coordinator.last_plan_summary.traces} == {"ner", "triples"}

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


def test_invalid_resources_and_shacl_failure_preserve_prior_published_graph(tmp_path, provider):
    url, state = provider
    state["delay"] = 0.01
    root = tmp_path / "documents"
    root.mkdir()
    source = root / "source.md"
    source.write_text("Ada studies Algebra.", encoding="utf-8")

    db = open_managed_yaw_db(tmp_path / "yaw.db")
    lm = create_dspy_lm("overlap", base_url=url + "/v1", cache=False, num_retries=0, max_tokens=512)
    program = YawProgram(rlm_sub_lm=lm, max_output_tokens=512)
    embedder = adapter(url)

    context = coco.ContextProvider()
    context.provide(SOURCE_DB, db)
    context.provide(GRAPH_DB, db)
    context.provide(VALIDATE_SQLITE, validate_graph_state)
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

        # An unregistered resource aborts extraction before its artifact or graph is published.
        state["mode"] = "shacl_unrepaired"
        second = root / "second.md"
        second.write_text("Ada visited London.", encoding="utf-8")
        app.update_blocking()
        error = get_graph_delta_coordinator(db).pop_error()
        assert isinstance(error, ValueError) and "absent from NER" in str(error)

        with db.readonly() as conn:
            assert conn.execute("SELECT COUNT(*) FROM source_chunks").fetchone()[0] == 2
            assert conn.execute("SELECT COUNT(*) FROM chunk_inference").fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM chunk_vec").fetchone()[0] == 1
            assert conn.execute("SELECT entity_id, name, definition FROM entities").fetchall() == initial_entities

        # Corrected extraction publishes the second chunk; the first artifact remains reusable.
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
    program = YawProgram(orchestrator_lm=lm, rlm_sub_lm=lm, enable_gbnf=False, max_output_tokens=512)
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
        assert engine._graph_rlm_stats["reused_chunks"] == 4
        assert engine._graph_rlm_stats["processed_chunks"] == 1
        assert (
            len(
                json.loads(
                    engine.conn.execute("SELECT value FROM meta WHERE key='graph_input_fingerprints'").fetchone()[0]
                )
            )
            == 5
        )
        assert engine.lint_graph()["conforms"]
    finally:
        engine.close()


def test_addition_preserves_adjudicated_graph_and_edit_invalidates_reuse(tmp_path, provider):
    from yaw.core import ChunkPointer
    from yaw.daemon.graph.engine import YawGraphEngine
    from yaw.daemon.graph.storage import upsert_edges_with_sources, upsert_semantic_edges
    from yaw.daemon.indexing import bind_native_indexing

    url, state = provider
    state["delay"] = 0.01
    lm = create_dspy_lm("overlap", base_url=url + "/v1", cache=False, num_retries=0, max_tokens=512)
    program = YawProgram(orchestrator_lm=lm, rlm_sub_lm=lm, enable_gbnf=False, max_output_tokens=512)
    engine = bind_native_indexing(YawGraphEngine(tmp_path / "yaw.db", tmp_path, program, adapter(url)))
    first = (ChunkPointer("first", "first.md", 1, 1, "first", 1, 1), "Ada studies Algebra.")
    second = (ChunkPointer("second", "second.md", 1, 1, "second", 1, 1), "Ada studies Algebra again.")
    try:
        engine.index_chunks([first])
        entities = [row[0] for row in engine.conn.execute("SELECT entity_id FROM entities ORDER BY entity_id")]
        # A previously validated semantic assertion must survive expansion.
        with engine.managed_conn.transaction() as conn:
            upsert_semantic_edges(conn, [(entities[0], entities[1], "owl:differentFrom", 0.8, "Distinct concepts")])
            upsert_edges_with_sources(
                conn,
                [
                    (a, b, "owl:differentFrom", weight, cid)
                    for a, b in (entities, entities[::-1])
                    for weight, cid in ((0.9, "first"), (0.8, ""))
                ],
            )
        assert engine.lint_graph()["conforms"]
        previous = engine.conn.execute(
            "SELECT src_id, dst_id, kind, weight FROM edges WHERE kind='owl:differentFrom'"
        ).fetchall()
        assert previous
        vectors = engine.conn.execute("SELECT embedding FROM chunk_vec WHERE chunk_id='first'").fetchone()
        engine.index_chunks([first, second])
        assert engine._graph_rlm_stats["reused_chunks"] == 1
        assert engine._graph_rlm_stats["processed_chunks"] == 1
        assert (
            engine.conn.execute(
                "SELECT src_id, dst_id, kind, weight FROM edges WHERE kind='owl:differentFrom'"
            ).fetchall()
            == previous
        )
        assert engine.conn.execute("SELECT embedding FROM chunk_vec WHERE chunk_id='first'").fetchone() == vectors
        assert engine.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 2
        assert engine.lint_graph()["conforms"]
        engine.index_chunks([(first[0], "Ada studies Algebra with a revised source."), second])
        assert engine._graph_rlm_stats["reused_chunks"] == 0
        assert engine._graph_rlm_stats["processed_chunks"] == 2
        assert engine.lint_graph()["conforms"]
    finally:
        engine.close()


def test_partition_frozen_chunks_requires_published_graph_and_emits_substage_progress(tmp_path, provider):
    from yaw.core import ChunkPointer
    from yaw.daemon.graph.engine import YawGraphEngine
    from yaw.daemon.graph.inference.optimizer import (
        hierarchy_judge_fingerprint,
        label_judge_fingerprint,
        openie_fingerprint,
        related_judge_fingerprint,
        shacl_fingerprint,
    )
    from yaw.daemon.indexing import InferenceIdentity, bind_native_indexing, partition_frozen_chunks

    url, state = provider
    state["delay"] = 0
    state["healed_definition"] = "Ada Lovelace collaborated with Ada M. Lovelace on algebra."
    chunks = [
        (
            ChunkPointer(f"c{i}", f"doc{i}.md", 1, 1, f"doc{i}", 1, 1),
            f"Passage {i}: Ada Lovelace collaborated with Ada M. Lovelace.",
        )
        for i in range(4)
    ]
    state["extractions"] = {
        text: {
            "named_entities": [
                ["Ada Lovelace", "English mathematician"],
                ["Ada M. Lovelace", "British mathematician and analyst"],
            ],
            "triples": [
                {
                    "sub": "Ada Lovelace",
                    "pred": "collaborated with",
                    "obj": "Ada M. Lovelace",
                    "def": "English mathematician",
                }
            ],
        }
        for _, text in chunks
    }
    lm = create_dspy_lm("overlap", base_url=url + "/v1", cache=False, num_retries=0, max_tokens=512)
    program = YawProgram(orchestrator_lm=lm, rlm_sub_lm=lm, enable_gbnf=False, max_output_tokens=512)
    embedder = adapter(url)
    engine = bind_native_indexing(
        YawGraphEngine(
            tmp_path / "yaw.db",
            tmp_path,
            program,
            embedder,
            synonymy_top_k=5,
            synonymy_threshold=-1.0,
        ),
        batch_size=2,
    )
    try:
        events: list[str] = []
        assert engine.index_chunks(chunks, progress_callback=events.append) == 4
        assert any("indexing: disambiguation candidate search" in ev for ev in events)
        assert any("indexing: synonym adjudication (" in ev for ev in events)
        assert any("indexing: SHACL healing (" in ev for ev in events)

        program_identity = openie_fingerprint(program)
        embedding_identity = str(embedder.profile.index_identity)
        identity = InferenceIdentity(program_identity, embedding_identity)
        settings = GraphPlanSettings(
            corpus_root=engine.workspace_root,
            program_identity=program_identity,
            embedding_identity=embedding_identity,
            synonym_identity=":".join(
                (
                    label_judge_fingerprint(program),
                    hierarchy_judge_fingerprint(program),
                    related_judge_fingerprint(program),
                )
            ),
            shacl_identity=shacl_fingerprint(program),
            auto_merge_cosine_gate=engine.auto_merge_cosine_gate,
            auto_merge_difflib_gate=engine.auto_merge_difflib_gate,
            synonymy_threshold=engine.synonymy_threshold,
            synonymy_top_k=engine.synonymy_top_k,
            ast_context_hops=engine.ast_context_hops,
            context_window_tokens=program.context_window_tokens,
            max_output_tokens=program.max_output_tokens,
        )

        # Simulate a mid-batch crash after c2 and c3 were extracted into source_chunks/chunk_inference
        # but before publish_graph committed c2 and c3 into chunks and meta['graph_input_fingerprints'].
        with engine.managed_conn.transaction() as conn:
            conn.execute("DELETE FROM chunks WHERE chunk_id IN ('c2', 'c3')")
            fps = dict(
                json.loads(conn.execute("SELECT value FROM meta WHERE key='graph_input_fingerprints'").fetchone()[0])
            )
            fps.pop("c2", None)
            fps.pop("c3", None)
            conn.execute(
                "UPDATE meta SET value=? WHERE key='graph_input_fingerprints'",
                (json.dumps(sorted(fps.items())),),
            )

        retained, changed = partition_frozen_chunks(engine.managed_conn, chunks, identity, settings=settings)
        assert [ptr.chunk_id for ptr, _ in retained] == ["c0", "c1"]
        assert [ptr.chunk_id for ptr, _ in changed] == ["c2", "c3"]

        # Tampering with c0's published input fingerprint forces c0 into changed even though c0 is in chunks.
        with engine.managed_conn.transaction() as conn:
            fps = dict(
                json.loads(conn.execute("SELECT value FROM meta WHERE key='graph_input_fingerprints'").fetchone()[0])
            )
            fps["c0"] = "0" * 64
            conn.execute(
                "UPDATE meta SET value=? WHERE key='graph_input_fingerprints'",
                (json.dumps(sorted(fps.items())),),
            )
        retained_after_tamper, changed_after_tamper = partition_frozen_chunks(
            engine.managed_conn, chunks, identity, settings=settings
        )
        assert [ptr.chunk_id for ptr, _ in retained_after_tamper] == ["c1"]
        assert [ptr.chunk_id for ptr, _ in changed_after_tamper] == ["c0", "c2", "c3"]
    finally:
        engine.close()


def test_shared_lmdb_cross_db_memo_reuse_and_target_isolation(tmp_path, provider):
    from yaw.core import ChunkPointer
    from yaw.daemon.graph.engine import YawGraphEngine
    from yaw.daemon.graph.storage import resolve_coco_state_root, resolve_db_identity
    from yaw.daemon.indexing import bind_native_indexing

    url, state = provider
    state["delay"] = 0
    state["healed_definition"] = "Ada Lovelace collaborated with Ada M. Lovelace on analytical algebra."
    chunks = [
        (
            ChunkPointer(f"c{i:02d}", f"doc{i:02d}.md", 1, 1, f"doc{i:02d}", 1, 1),
            f"Passage {i:02d}: Ada Lovelace collaborated with Ada M. Lovelace in London.",
        )
        for i in range(64)
    ]
    state["extractions"] = {
        text: (
            {
                "named_entities": [
                    ["Ada Lovelace", "English mathematician"],
                    ["Ada M. Lovelace", "British mathematician and analyst"],
                    ["London", ""],
                ],
                "triples": [
                    {
                        "sub": "Ada Lovelace",
                        "pred": "collaborated with",
                        "obj": "Ada M. Lovelace",
                        "def": "English mathematician",
                    },
                    {
                        "sub": "Ada Lovelace",
                        "pred": "visited",
                        "obj": "London",
                        "def": "Ada Lovelace visited London during her research.",
                    },
                ],
            }
            if idx == 0
            else {
                "named_entities": [
                    ["Ada Lovelace", "English mathematician"],
                    ["Ada M. Lovelace", "British mathematician and analyst"],
                ],
                "triples": [
                    {
                        "sub": "Ada Lovelace",
                        "pred": "collaborated with",
                        "obj": "Ada M. Lovelace",
                        "def": "English mathematician",
                    }
                ],
            }
        )
        for idx, (_, text) in enumerate(chunks)
    }

    db_a_path = tmp_path / "db_a.sqlite"
    db_b_path = tmp_path / "db_b.sqlite"
    assert resolve_coco_state_root(db_a_path) == tmp_path / "coco_lmdb"
    assert resolve_coco_state_root(db_b_path) == tmp_path / "coco_lmdb"

    tracked_tables = (
        "source_chunks",
        "chunk_inference",
        "chunks",
        "entities",
        "entity_labels",
        "facts",
        "fact_sources",
        "edges",
        "edge_sources",
        "chunk_vec",
        "entity_vec",
        "fact_vec",
        "entity_mentions",
        "fact_assertions",
    )

    def snapshot_db(conn) -> dict[str, list[tuple[Any, ...]]]:
        snap: dict[str, list[tuple[Any, ...]]] = {
            table: conn.execute(f"SELECT * FROM {table} ORDER BY 1, 2").fetchall() for table in tracked_tables
        }
        snap["chunk_fts"] = conn.execute("SELECT rowid, chunk_id FROM chunk_fts ORDER BY chunk_id").fetchall()
        snap["fact_fts"] = conn.execute("SELECT rowid, fact_id FROM fact_fts ORDER BY fact_id").fetchall()
        snap["meta"] = conn.execute("SELECT key, value FROM meta WHERE key != 'db_identity' ORDER BY key").fetchall()
        return snap

    lm = create_dspy_lm("overlap", base_url=url + "/v1", cache=False, num_retries=0, max_tokens=512)
    program = YawProgram(orchestrator_lm=lm, rlm_sub_lm=lm, enable_gbnf=False, max_output_tokens=512)
    embedder = adapter(url)

    engine_a = bind_native_indexing(
        YawGraphEngine(
            db_a_path,
            tmp_path,
            program,
            embedder,
            synonymy_top_k=5,
            synonymy_threshold=-1.0,
        ),
        batch_size=64,
    )
    try:
        assert engine_a.index_chunks(chunks) == 64
        assert engine_a.lint_graph()["conforms"]
        id_a = resolve_db_identity(engine_a.conn, db_a_path)
        assert engine_a.conn.execute("SELECT value FROM meta WHERE key = 'db_identity'").fetchone()[0] == id_a
        snap_a_initial = snapshot_db(engine_a.conn)
        assert len(snap_a_initial["source_chunks"]) == 64
        assert len(snap_a_initial["chunk_inference"]) == 64
        assert len(snap_a_initial["chunks"]) == 64
        assert len(snap_a_initial["chunk_vec"]) == 64
        assert len(snap_a_initial["entities"]) >= 2
        assert len(snap_a_initial["facts"]) >= 2

        calls_after_a = len(state["intervals"])
        extractions_after_a = len(state.get("extraction_requests", []))
        embeddings_after_a = len(state.get("embedding_inputs", []))
        healers_after_a = len(state.get("healer_requests", []))
        assert calls_after_a > 0
        assert extractions_after_a == 128
        assert embeddings_after_a > 0
        assert healers_after_a > 0

        engine_b = bind_native_indexing(
            YawGraphEngine(
                db_b_path,
                tmp_path,
                program,
                embedder,
                synonymy_top_k=5,
                synonymy_threshold=-1.0,
            ),
            batch_size=64,
        )
        try:
            assert engine_b.index_chunks(chunks) == 64
            assert engine_b.lint_graph()["conforms"]
            id_b = resolve_db_identity(engine_b.conn, db_b_path)
            assert id_b != id_a
            assert engine_b.conn.execute("SELECT value FROM meta WHERE key = 'db_identity'").fetchone()[0] == id_b
            assert len(state["intervals"]) == calls_after_a
            assert len(state.get("extraction_requests", [])) == extractions_after_a
            assert len(state.get("embedding_inputs", [])) == embeddings_after_a
            assert len(state.get("healer_requests", [])) == healers_after_a

            snap_b = snapshot_db(engine_b.conn)
            assert snap_b == snap_a_initial
            assert snapshot_db(engine_a.conn) == snap_a_initial
        finally:
            engine_b.close()
    finally:
        engine_a.close()
