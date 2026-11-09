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
    INDEXING_ERROR_SINK,
    RELOAD_SIGNAL_HANDLER,
    RELOAD_SIGNAL_ROOT,
    ChunkInference,
    ChunkProcessOutput,
    CorpusIndexingLiveComponent,
    InferenceIdentity,
    indexing_app_main,
)
from yaw.daemon.ingestion.source_targets import SOURCE_DB, SOURCE_ROOT, SourceSettings


def test_frozen_candidate_replay_uses_native_targets_and_original_text(tmp_path, provider):
    from yaw.core import ChunkPointer, compute_chunk_id
    from yaw.daemon.graph.engine import YawGraphEngine
    from yaw.daemon.indexing import bind_native_indexing

    url, state = provider
    source = tmp_path / "source.md"
    original = "Ada studies Algebra."
    expected_id = compute_chunk_id(source.name, 1, 1, original)
    source.write_text("Changed after the evaluation snapshot was frozen.", encoding="utf-8")
    pointer = ChunkPointer(expected_id, source.name, 1, 1, "source", 1, 1)
    lm = create_dspy_lm("overlap", base_url=url + "/v1", cache=False, num_retries=0, max_tokens=512)
    program = YawProgram(orchestrator_lm=lm, rlm_sub_lm=lm, enable_gbnf=False, max_output_tokens=512)
    engine = bind_native_indexing(YawGraphEngine(tmp_path / "yaw.db", tmp_path, program, adapter(url)))
    try:
        assert engine.index_chunks([(pointer, original)]) == 1
        assert [row[0] for row in engine.conn.execute("SELECT chunk_id FROM chunks")] == [expected_id]
        assert engine.conn.execute("SELECT COUNT(*) FROM chunk_inference").fetchone()[0] == 1
        assert engine.conn.execute("SELECT 1 FROM sqlite_master WHERE name='openie_chunks'").fetchone() is None
        traces = engine._captured_chunk_traces[expected_id]
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
                                        "value": state.get("healed_pref_labels", {}).get(
                                            v["focus_node"], "Rate limit responses"
                                        )
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


def test_invalid_resources_and_shacl_failure_preserve_prior_published_graph(tmp_path, provider, caplog):
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

    recorded_errors: list[BaseException] = []
    context = coco.ContextProvider()
    context.provide(SOURCE_DB, db)
    context.provide(GRAPH_DB, db)
    context.provide(VALIDATE_SQLITE, validate_graph_state)
    context.provide(SOURCE_ROOT, root)
    context.provide(EMBEDDING_ENDPOINT, EmbeddingEndpoint(embedder, asyncio.Semaphore(2)))
    context.provide(OPENIE_ENDPOINT, OpenIEEndpoint(program, lm, "fixture-v1", asyncio.Semaphore(2)))
    context.provide(INDEXING_ERROR_SINK, recorded_errors.append)

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
        recorded_errors.clear()
        caplog.clear()
        app.update_blocking()
        assert len(recorded_errors) == 1
        assert isinstance(recorded_errors[0], ValueError) and "absent from NER" in str(recorded_errors[0])
        assert "absent from NER" in caplog.text

        with db.readonly() as conn:
            assert conn.execute("SELECT COUNT(*) FROM source_chunks").fetchone()[0] == 2
            assert conn.execute("SELECT COUNT(*) FROM chunk_inference").fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM chunk_vec").fetchone()[0] == 1
            assert conn.execute("SELECT entity_id, name, definition FROM entities").fetchall() == initial_entities

        # Corrected extraction publishes the second chunk; the first artifact remains reusable.
        state["mode"] = "valid"
        app.update_blocking()
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
    from yaw.core import ChunkPointer, compute_chunk_id
    from yaw.daemon.graph.engine import YawGraphEngine
    from yaw.daemon.graph.storage import upsert_edges_with_sources, upsert_semantic_edges
    from yaw.daemon.indexing import bind_native_indexing

    url, state = provider
    state["delay"] = 0.01
    lm = create_dspy_lm("overlap", base_url=url + "/v1", cache=False, num_retries=0, max_tokens=512)
    program = YawProgram(orchestrator_lm=lm, rlm_sub_lm=lm, enable_gbnf=False, max_output_tokens=512)
    engine = bind_native_indexing(YawGraphEngine(tmp_path / "yaw.db", tmp_path, program, adapter(url)))
    first_text = "Ada studies Algebra."
    first_id = compute_chunk_id("first.md", 1, 1, first_text)
    second_text = "Ada studies Algebra again."
    second_id = compute_chunk_id("second.md", 1, 1, second_text)
    first = (ChunkPointer(first_id, "first.md", 1, 1, "first", 1, 1), first_text)
    second = (ChunkPointer(second_id, "second.md", 1, 1, "second", 1, 1), second_text)
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
                    for weight, cid in ((0.9, first_id), (0.8, ""))
                ],
            )
        assert engine.lint_graph()["conforms"]
        previous = engine.conn.execute(
            "SELECT src_id, dst_id, kind, weight FROM edges WHERE kind='owl:differentFrom'"
        ).fetchall()
        assert previous
        vectors = engine.conn.execute("SELECT embedding FROM chunk_vec WHERE chunk_id=?", (first_id,)).fetchone()
        engine.index_chunks([first, second])
        assert engine._graph_rlm_stats["reused_chunks"] == 1
        assert engine._graph_rlm_stats["processed_chunks"] == 1
        assert (
            engine.conn.execute(
                "SELECT src_id, dst_id, kind, weight FROM edges WHERE kind='owl:differentFrom'"
            ).fetchall()
            == previous
        )
        assert (
            engine.conn.execute("SELECT embedding FROM chunk_vec WHERE chunk_id=?", (first_id,)).fetchone() == vectors
        )
        assert engine.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 2
        assert engine.lint_graph()["conforms"]
        revised_text = "Ada studies Algebra with a revised source."
        revised_id = compute_chunk_id("first.md", 1, 1, revised_text)
        revised_chunk = (ChunkPointer(revised_id, "first.md", 1, 1, "first", 1, 1), revised_text)
        engine.index_chunks([revised_chunk, second])
        assert engine._graph_rlm_stats["reused_chunks"] == 0
        assert engine._graph_rlm_stats["processed_chunks"] == 2
        assert engine.lint_graph()["conforms"]
    finally:
        engine.close()


def test_incremental_indexing_recovers_from_interrupted_publication_and_emits_substage_progress(tmp_path, provider):
    from yaw.core import ChunkPointer, compute_chunk_id
    from yaw.daemon.graph.engine import YawGraphEngine
    from yaw.daemon.indexing import CorpusIndexingLiveComponent, bind_native_indexing

    url, state = provider
    state["delay"] = 0
    state["healed_definition"] = "Ada Lovelace collaborated with Ada M. Lovelace on algebra."
    chunks = [
        (
            ChunkPointer(
                compute_chunk_id(f"doc{i}.md", 1, 1, f"Passage {i}: Ada Lovelace collaborated with Ada M. Lovelace."),
                f"doc{i}.md",
                1,
                1,
                f"doc{i}",
                1,
                1,
            ),
            f"Passage {i}: Ada Lovelace collaborated with Ada M. Lovelace.",
        )
        for i in range(4)
    ]
    c_ids = [ptr.chunk_id for ptr, _ in chunks]
    state["extractions"] = {
        text: {
            "named_entities": [
                ["429s", "HTTP rate limit status"],
                ["Ada Lovelace", "English mathematician"],
                ["Ada M. Lovelace", "British mathematician and analyst"],
            ],
            "triples": [
                {
                    "sub": "429s",
                    "pred": "observed by",
                    "obj": "Ada Lovelace",
                    "def": "Rate limit status observed by Ada Lovelace.",
                },
                {
                    "sub": "Ada Lovelace",
                    "pred": "collaborated with",
                    "obj": "Ada M. Lovelace",
                    "def": "English mathematician",
                },
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
        assert engine.index_chunks(chunks[:2], progress_callback=events.append) == 2
        assert any("indexing: disambiguation candidate search" in ev for ev in events)
        assert any("indexing: synonym adjudication (" in ev for ev in events)
        assert any("indexing: SHACL healing (" in ev for ev in events)

        # Indexing all 4 chunks must dynamically partition discovered chunks:
        # c_ids[0..1] are retained, c_ids[2..3] are changed.
        assert engine.index_chunks(chunks) == 4
        component = CorpusIndexingLiveComponent.last_instance
        assert component is not None
        assert [m.chunk_id for m, _ in component.initial_retained] == [c_ids[0], c_ids[1]]
        assert [m.chunk_id for m, _ in component.initial_changed] == [c_ids[2], c_ids[3]]

        # Tampering with c_ids[0]'s published input fingerprint forces c_ids[0] into changed.
        with engine.managed_conn.transaction() as conn:
            fps = dict(
                json.loads(conn.execute("SELECT value FROM meta WHERE key='graph_input_fingerprints'").fetchone()[0])
            )
            fps[c_ids[0]] = "0" * 64
            conn.execute(
                "UPDATE meta SET value=? WHERE key='graph_input_fingerprints'",
                (json.dumps(sorted(fps.items())),),
            )
        assert engine.index_chunks(chunks) == 4
        component = CorpusIndexingLiveComponent.last_instance
        assert component is not None
        assert [m.chunk_id for m, _ in component.initial_retained] == [c_ids[1], c_ids[2], c_ids[3]]
        assert [m.chunk_id for m, _ in component.initial_changed] == [c_ids[0]]
    finally:
        engine.close()


def test_shared_lmdb_cross_db_memo_reuse_and_target_isolation(tmp_path, provider):
    from yaw.core import ChunkPointer, compute_chunk_id
    from yaw.daemon.graph.engine import YawGraphEngine
    from yaw.daemon.graph.storage import resolve_coco_state_root, resolve_db_identity
    from yaw.daemon.indexing import bind_native_indexing

    url, state = provider
    state["delay"] = 0
    state["healed_definition"] = "Ada Lovelace collaborated with Ada M. Lovelace on analytical algebra."
    chunks = [
        (
            ChunkPointer(
                compute_chunk_id(
                    f"doc{i:02d}.md",
                    1,
                    1,
                    f"Passage {i:02d}: Ada Lovelace collaborated with Ada M. Lovelace in London.",
                ),
                f"doc{i:02d}.md",
                1,
                1,
                f"doc{i:02d}",
                1,
                1,
            ),
            f"Passage {i:02d}: Ada Lovelace collaborated with Ada M. Lovelace in London.",
        )
        for i in range(64)
    ]
    state["extractions"] = {
        text: (
            {
                "named_entities": [
                    ["429s", "HTTP rate limit status"],
                    ["Ada Lovelace", "English mathematician"],
                    ["Ada M. Lovelace", "British mathematician and analyst"],
                    ["London", ""],
                ],
                "triples": [
                    {
                        "sub": "429s",
                        "pred": "observed by",
                        "obj": "Ada Lovelace",
                        "def": "Rate limit status observed by Ada Lovelace.",
                    },
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


def test_direct_use_mount_chunk_output_handoff_without_chunk_state_target(tmp_path, provider):
    import yaw.daemon.graph.planner as planner_mod
    import yaw.daemon.indexing as indexing_mod
    from yaw.core import ChunkPointer, compute_chunk_id
    from yaw.daemon.graph.engine import YawGraphEngine
    from yaw.daemon.indexing import bind_native_indexing

    for parts in (
        ("ChunkState", "Action"),
        ("ChunkState", "Record"),
        ("GraphDelta", "Coordinator"),
        ("get_graph_delta", "_coordinator"),
        ("ChunkState", "TargetHandler"),
        ("CHUNK_STATE", "_TARGET"),
        ("declare_chunk", "_state"),
        ("compute_active", "_inference_fingerprint"),
        ("_is_published", "_graph_aligned"),
        ("_Subscriber", "Proxy"),
        ("_DeltaNotifying", "LiveItems"),
        ("GraphPublication", "LiveComponent"),
    ):
        retired = "".join(parts)
        assert not hasattr(indexing_mod, retired), f"{retired} must be retired from yaw.daemon.indexing"

    assert not hasattr(planner_mod, "_".join(("load", "active", "chunk", "inferences")))
    assert not hasattr(planner_mod, "resolve_chunk_text")
    assert issubclass(CorpusIndexingLiveComponent, object)

    url, state = provider
    state["delay"] = 0
    lm = create_dspy_lm("overlap", base_url=url + "/v1", cache=False, num_retries=0, max_tokens=512)
    program = YawProgram(orchestrator_lm=lm, rlm_sub_lm=lm, enable_gbnf=False, max_output_tokens=512)
    embedder = adapter(url)
    engine = bind_native_indexing(
        YawGraphEngine(tmp_path / "yaw.db", tmp_path, program, embedder, synonymy_top_k=0),
        batch_size=2,
    )
    t0 = "Ada studies Algebra."
    t1 = "Euler studies Graphs."
    h0 = compute_chunk_id("doc0.md", 1, 1, t0)
    h1 = compute_chunk_id("doc1.md", 1, 1, t1)
    chunks = [
        (ChunkPointer(h0, "doc0.md", 1, 1, "doc0", 1, 1), t0),
        (ChunkPointer(h1, "doc1.md", 1, 1, "doc1", 1, 1), t1),
    ]
    captured_outputs: list[tuple[ChunkProcessOutput, ...]] = []
    orig_plan = planner_mod.plan_graph_publication

    try:
        assert engine.index_chunks(chunks) == 2
        cols = [str(r[1]) for r in engine.conn.execute("PRAGMA table_info(chunk_inference)").fetchall()]
        assert cols == [
            "chunk_id",
            "source_hash",
            "program_identity",
            "embedding_identity",
            "extraction_json",
            "traces_json",
        ]
        assert "embedding" not in cols
        chunk_embed_calls = [
            inp for inp in state.get("embedding_inputs", []) if "doc0" in inp or "doc1" in inp or "Ada studies" in inp
        ]
        # Each chunk is embedded once in process_chunk and never re-embedded in planner.py
        assert len(chunk_embed_calls) == 2
        row = engine.conn.execute("SELECT * FROM chunk_inference WHERE chunk_id = ?", (h0,)).fetchone()
        inf = ChunkInference(*row)
        vec_bytes = engine.conn.execute("SELECT embedding FROM chunk_vec WHERE chunk_id = ?", (h0,)).fetchone()[0]
        out = ChunkProcessOutput(pointer=chunks[0][0], text=chunks[0][1], inference=inf, embedding=vec_bytes)
        assert out.chunk_id == h0
        assert out.vector_f32 == vec_bytes
        captured_outputs.append((out,))
        assert len(captured_outputs[0]) == 1
        assert orig_plan is not None
    finally:
        engine.close()


def test_shared_coco_environment_lifecycle_and_app_caching(tmp_path):
    import yaw.daemon.indexing as indexing_mod
    import yaw.daemon.runtime as runtime_mod
    from yaw.daemon.indexing import (
        _SHARED_COCO_RUNTIMES,
        _SharedCocoRuntime,
        acquire_coco_environment,
        close_coco_environment,
    )

    for symbol in (
        "_".join(("", "release", "idle", "coco", "env")),
        "_".join(("", "Active", "Coco", "App", "Registry")),
        "_".join(("register", "active", "coco", "env", "closer")),
        "_".join(("release", "idle", "coco", "env")),
    ):
        assert not hasattr(indexing_mod, symbol), f"{symbol} must not exist in yaw.daemon.indexing"
        assert not hasattr(runtime_mod, symbol), f"{symbol} must not exist in yaw.daemon.runtime"

    root_a = tmp_path / "env_a"
    root_b = tmp_path / "env_b"

    shared_a1 = acquire_coco_environment(root_a)
    assert isinstance(shared_a1, _SharedCocoRuntime)
    assert shared_a1.ref_count == 1
    assert shared_a1.apps == {}

    shared_a2 = acquire_coco_environment(root_a)
    assert shared_a2 is shared_a1
    assert shared_a1.ref_count == 2

    shared_b = acquire_coco_environment(root_b)
    assert shared_b is not shared_a1
    assert shared_b.ref_count == 1

    dummy_app = coco.App(coco.AppConfig(name="dummy", environment=shared_a1.environment), lambda: None)
    shared_a1.apps["dummy"] = dummy_app

    close_coco_environment(root_a)
    assert shared_a1.ref_count == 1
    assert "dummy" in shared_a1.apps

    shared_a3 = acquire_coco_environment(root_a)
    assert shared_a3 is shared_a1
    assert shared_a1.ref_count == 2
    assert "dummy" in shared_a1.apps

    close_coco_environment(root_a)
    assert shared_a1.ref_count == 1

    close_coco_environment(root_a)
    assert shared_a1.ref_count == 0
    assert len(shared_a1.apps) == 0

    close_coco_environment(root_b)
    assert shared_b.ref_count == 0

    from yaw.core import resolve_coco_state_root

    assert resolve_coco_state_root(root_a).resolve() not in _SHARED_COCO_RUNTIMES
    assert resolve_coco_state_root(root_b).resolve() not in _SHARED_COCO_RUNTIMES


def test_corpus_feed_subscriber_incremental_active_files_event_handling():
    from typing import cast

    from yaw.daemon.indexing import CorpusIndexingLiveComponent, _CorpusFeedSubscriber

    class MockOperator:
        def __init__(self) -> None:
            self.update_full_calls = 0

        async def update_full(self) -> None:
            self.update_full_calls += 1

        async def mark_ready(self) -> None:
            pass

    op = MockOperator()
    comp = CorpusIndexingLiveComponent(
        files=None,
        sources=cast(Any, None),
        artifacts=cast(Any, None),
        settings=SourceSettings(("*.md",), ()),
        identity=InferenceIdentity("prog", "embed"),
    )
    subscriber = _CorpusFeedSubscriber(cast(coco.LiveComponentOperator, op), comp)

    # When active_files is None, update and delete do not crash and still call update_full
    assert comp.active_files is None
    handle = asyncio.run(subscriber.update("doc1.md", "file_obj_1"))
    assert asyncio.run(handle.ready()) is None
    assert op.update_full_calls == 1
    assert comp.active_files is None

    handle = asyncio.run(subscriber.delete("doc1.md"))
    assert asyncio.run(handle.ready()) is None
    assert op.update_full_calls == 2

    # Seed subscriber state
    vars(comp)["_active_files"] = {"doc1.md": "file_obj_1", "doc2.md": "file_obj_2"}

    # update(key, value) updates in O(1)
    asyncio.run(subscriber.update("doc3.md", "file_obj_3"))
    assert op.update_full_calls == 3
    assert comp.active_files == {
        "doc1.md": "file_obj_1",
        "doc2.md": "file_obj_2",
        "doc3.md": "file_obj_3",
    }

    # delete(key) pops in O(1)
    asyncio.run(subscriber.delete("doc2.md"))
    assert op.update_full_calls == 4
    assert comp.active_files == {
        "doc1.md": "file_obj_1",
        "doc3.md": "file_obj_3",
    }

    # delete of non-existent key is safe
    asyncio.run(subscriber.delete("non_existent.md"))
    assert op.update_full_calls == 5
    assert comp.active_files == {
        "doc1.md": "file_obj_1",
        "doc3.md": "file_obj_3",
    }

    # update_all() sets active_files to None
    asyncio.run(subscriber.update_all())
    assert op.update_full_calls == 6
    assert comp.active_files is None


def test_corpus_indexing_live_component_micro_batching_and_memo_hits(tmp_path, provider):
    url, state = provider
    state["delay"] = 0
    root = tmp_path / "documents"
    root.mkdir()
    for i in range(5):
        (root / f"doc_{i:02d}.md").write_text(f"Ada Lovelace studied science topic {i:02d}.", encoding="utf-8")

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
        coco.AppConfig(name="batching_pipeline", environment=env, max_inflight_components=4),
        indexing_app_main,
        SourceSettings(("**/*.md",), (), min_chunk_size=1),
        InferenceIdentity("fixture-v1", embedder.profile.index_identity),
        graph_settings,
        False,  # watch_reload_signal
        False,  # live
        2,  # batch_size = 2
    )
    try:
        # Pass 1: 5 chunks discovered, batch_size=2 -> 2 chunks published, remaining=3
        app.update_blocking()
        comp = CorpusIndexingLiveComponent.last_instance
        assert comp is not None
        assert comp.remaining_chunks == 3
        with db.readonly() as conn:
            assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 2

        calls_after_pass1 = len(state["intervals"])
        assert calls_after_pass1 > 0

        # Pass 2: next 2 chunks -> 4 published, remaining=1
        app.update_blocking()
        comp = CorpusIndexingLiveComponent.last_instance
        assert comp is not None
        assert comp.remaining_chunks == 1
        with db.readonly() as conn:
            assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 4

        calls_after_pass2 = len(state["intervals"])
        # Retained chunks from Pass 1 hit memo! Only the 2 new chunks incur calls.
        assert calls_after_pass2 > calls_after_pass1

        # Pass 3: final 1 chunk -> 5 published, remaining=0
        app.update_blocking()
        comp = CorpusIndexingLiveComponent.last_instance
        assert comp is not None
        assert comp.remaining_chunks == 0
        with db.readonly() as conn:
            assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 5

        calls_after_pass3 = len(state["intervals"])
        assert calls_after_pass3 > calls_after_pass2

        # Pass 4: unchanged -> remaining=0, 0 additional provider calls (full memo hit!)
        app.update_blocking()
        comp = CorpusIndexingLiveComponent.last_instance
        assert comp is not None
        assert comp.remaining_chunks == 0
        with db.readonly() as conn:
            assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 5
        assert len(state["intervals"]) == calls_after_pass3
    finally:
        env.close()
        db.close()


def test_active_context_keys_override_constructor_defaults(tmp_path, provider):
    from yaw.daemon.indexing import (
        ACTIVE_GRAPH_SETTINGS,
        ACTIVE_INFERENCE_IDENTITY,
        ACTIVE_SOURCE_SETTINGS,
    )

    url, state = provider
    state["delay"] = 0
    root = tmp_path / "documents"
    root.mkdir()
    (root / "doc.md").write_text("Ada studies Algebra.", encoding="utf-8")
    (root / "doc2.md").write_text("Unrelated text", encoding="utf-8")

    db = open_managed_yaw_db(tmp_path / "yaw.db")
    lm = create_dspy_lm("overlap", base_url=url + "/v1", cache=False, num_retries=0, max_tokens=512)
    program = YawProgram(rlm_sub_lm=lm, max_output_tokens=512)
    embedder = adapter(url)

    active_src = SourceSettings(("**/*.md",), (), included_files=("doc.md",), min_chunk_size=1)
    active_ident = InferenceIdentity("active-v1", embedder.profile.index_identity)
    active_graph = GraphPlanSettings(
        corpus_root=root,
        program_identity="active-v1",
        embedding_identity=embedder.profile.index_identity,
        synonymy_top_k=0,
    )

    context = coco.ContextProvider()
    context.provide(SOURCE_DB, db)
    context.provide(GRAPH_DB, db)
    context.provide(VALIDATE_SQLITE, validate_graph_state)
    context.provide(SOURCE_ROOT, root)
    context.provide(EMBEDDING_ENDPOINT, EmbeddingEndpoint(embedder, asyncio.Semaphore(2)))
    context.provide(OPENIE_ENDPOINT, OpenIEEndpoint(program, lm, "active-v1", asyncio.Semaphore(2)))
    context.provide(ACTIVE_SOURCE_SETTINGS, active_src)
    context.provide(ACTIVE_INFERENCE_IDENTITY, active_ident)
    context.provide(ACTIVE_GRAPH_SETTINGS, active_graph)

    env = coco.Environment(settings=coco.Settings.from_env(tmp_path / "state"), context_provider=context)
    # Pass dummy/fallback settings to indexing_app_main; context keys will override them
    app = coco.App(
        coco.AppConfig(name="active_ctx_pipeline", environment=env),
        indexing_app_main,
        SourceSettings(("**/*.md",), ()),
        InferenceIdentity("fallback", "fallback"),
        None,  # graph_settings fallback is None
        False,
        False,
    )
    try:
        app.update_blocking()
        with db.readonly() as conn:
            # Overridden active_graph caused publish_graph to run and publish chunks
            assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 1
            chunk_row = conn.execute("SELECT rel_posix_path FROM chunks").fetchone()
            assert chunk_row[0] == "doc.md"
            # Overridden active_src included_files filtered out doc2.md from source_chunks
            src_rows = conn.execute("SELECT rel_posix_path FROM source_chunks").fetchall()
            assert [r[0] for r in src_rows] == ["doc.md"]
            # Overridden active_ident was stored in chunk_inference
            row = conn.execute("SELECT program_identity FROM chunk_inference").fetchone()
            assert row[0] == "active-v1"
    finally:
        env.close()
        db.close()
