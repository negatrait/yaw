"""Native DSPy/HTTP indexing and durable OpenIE recovery against real SQLite."""

import hashlib
import json
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import dspy
import numpy as np
import pytest

from yaw.core import ChunkPointer
from yaw.daemon.graph.engine import YawGraphEngine
from yaw.daemon.graph.inference import HippoRAGProgram
from yaw.daemon.graph.inference.embedding import DSPyEmbeddingAdapter, create_dspy_embedder, resolve_embedding_profile


@pytest.fixture
def indexing_provider():
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path.endswith("/embeddings"):
                texts = request["input"] if isinstance(request["input"], list) else [request["input"]]
                raw = json.dumps(
                    {
                        "object": "list",
                        "model": request["model"],
                        "data": [
                            {
                                "object": "embedding",
                                "index": i,
                                "embedding": [byte + 1 for byte in hashlib.sha256(text.encode()).digest()[:8]],
                            }
                            for i, text in enumerate(texts)
                        ],
                        "usage": {"prompt_tokens": 10, "total_tokens": 10},
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return
            prompt = str(request["messages"])
            if "query_intent" in prompt:
                stage, result = "intent", {"query_intent": "Who wrote Notes?"}
            elif "fact_before_filter" in prompt:
                stage, result = "filter", {"fact": [["ada", "wrote", "notes"]]}
            elif "reasoning" in prompt.lower():
                stage, result = "qa", {"reasoning": "The source states Ada wrote Notes.", "answer": "Answer: 'Ada'"}
            elif "candidate_pairs" in prompt:
                stage, result = "synonyms", {"decisions": []}
            elif "missing_entities" in prompt:
                stage, result = (
                    "definitions",
                    {"entity_definitions": [["Ada", "Mathematician"], ["Notes", "Publication"]]},
                )
            elif "triples" in prompt:
                stage, result = (
                    "triples",
                    {
                        "triples": [
                            {
                                "sub": "Ada",
                                "pred": "wrote",
                                "obj": "Notes",
                                "def": "Ada wrote Notes.",
                            }
                        ]
                    },
                )
            elif "named_entities" in prompt:
                stage, result = "ner", {"named_entities": [["Ada", "Mathematician"], ["Notes", "Publication"]]}
            else:
                self.send_error(500, "Unexpected inference request")
                return
            calls.append(stage)
            payload = {
                "id": "index",
                "object": "chat.completion",
                "created": 0,
                "model": request["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": json.dumps(result)},
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            }
            raw = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def make_engine(tmp_path, url) -> YawGraphEngine:
    lm = dspy.LM("openai/indexing-contract", api_base=url, api_key="test", cache=False, num_retries=0)
    program = HippoRAGProgram(include_demos=False, enable_gbnf=False, orchestrator_lm=lm, rlm_sub_lm=lm)

    def embed(texts):
        return np.array(
            [[byte + 1 for byte in hashlib.sha256(text.encode()).digest()[:8]] for text in texts], dtype=np.float32
        )

    embedder = DSPyEmbeddingAdapter(
        replace(resolve_embedding_profile(), dimensions=8),
        create_dspy_embedder(embed, caching=False),
    )
    return YawGraphEngine(db_path=tmp_path / "yaw.db", workspace_root=tmp_path, program=program, embedder=embedder)


@pytest.mark.parametrize(
    "checkpoint_kind", ["structured", "triples-only", "missing-definitions", "structured-only", "invalid-statement"]
)
def test_committed_openie_survives_stage_cache_eviction_and_engine_restart(
    tmp_path, indexing_provider, checkpoint_kind
):
    url, calls = indexing_provider
    source = "Ada wrote Notes."
    (tmp_path / "source.txt").write_text(source, encoding="utf-8")
    pointer = ChunkPointer("source", "source.txt", 1, 1, "Source")
    first = make_engine(tmp_path, url)
    try:
        assert first.index_chunks([]) == 0
        assert first.index_chunks([(pointer, source)]) == 1
        assert calls.count("ner") == 1 and calls.count("triples") == 1
        assert first.conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 1
        fingerprints = [
            tuple(row) for row in first.conn.execute("SELECT key,value FROM meta WHERE key LIKE '%fp:%' ORDER BY key")
        ]
        with first.managed_conn.transaction() as conn:
            conn.execute("DELETE FROM candidate_stage_cache")
            if checkpoint_kind in ("triples-only", "missing-definitions"):
                conn.execute("UPDATE openie_chunks SET structured_triples_json='[]'")
            if checkpoint_kind == "missing-definitions":
                conn.execute("UPDATE openie_chunks SET entities_json=?", (json.dumps(["Ada", "Notes"]),))
            if checkpoint_kind == "structured-only":
                conn.execute("UPDATE openie_chunks SET entities_json='[]'")
            if checkpoint_kind == "invalid-statement":
                saved = json.loads(conn.execute("SELECT structured_triples_json FROM openie_chunks").fetchone()[0])
                conn.execute(
                    "UPDATE openie_chunks SET structured_triples_json=?",
                    (json.dumps([{"subj": "", "pred": "invalid", "obj": "Notes"}, *saved]),),
                )
        checkpoint = tuple(
            first.conn.execute(
                "SELECT entities_json,triples_json,structured_triples_json FROM openie_chunks"
            ).fetchone()
        )
    finally:
        first.close()
    before = list(calls)
    second = make_engine(tmp_path, url)
    try:
        second.index_chunks([(pointer, source)])
        assert calls.count("ner") == before.count("ner")
        assert calls.count("triples") == before.count("triples")
        assert calls.count("definitions") - before.count("definitions") == (
            1 if checkpoint_kind == "missing-definitions" else 0
        )
        assert (
            tuple(
                second.conn.execute(
                    "SELECT entities_json,triples_json,structured_triples_json FROM openie_chunks"
                ).fetchone()
            )
            == checkpoint
        )
        assert [
            tuple(row) for row in second.conn.execute("SELECT key,value FROM meta WHERE key LIKE '%fp:%' ORDER BY key")
        ] == fingerprints
        assert second.conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 1
        assert second.retrieve_dpr("Ada", top_k=0) == []
    finally:
        second.close()


def test_empty_engine_operations_do_not_call_inference_or_create_graph_rows(tmp_path, indexing_provider):
    url, calls = indexing_provider
    engine = make_engine(tmp_path, url)
    try:
        assert engine.index([]) == 0
        assert engine.index([""]) == 0
        assert engine.resolve_chunk_pointers([]) == {}
        assert engine.retrieve_with_seeds("question") == ([], [])
        assert engine.retrieve_dpr("question") == []
        assert engine.rag_qa([]) == []
        assert engine.format_qa_context([]) == ""
        engine.delete_chunks([])
        assert calls == []
        assert engine.conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0
    finally:
        engine.close()


def test_unadjudicated_synonym_candidates_below_16_emit_skos_related_fallback(tmp_path, indexing_provider):
    url, calls = indexing_provider
    engine = make_engine(tmp_path, url)
    engine.synonymy_threshold = 0.1
    try:
        assert engine.index(["Ada wrote Notes."]) == 1
        assert "synonyms" in calls
        relations = [
            tuple(row)
            for row in engine.conn.execute(
                "SELECT src_id, dst_id, kind FROM edges WHERE kind = 'skos:related' ORDER BY src_id"
            )
        ]
        assert len(relations) == 2
        assert all(rel == "skos:related" for _, _, rel in relations)
        resolved = engine._resolve_chunks_for_docs(["Ada wrote Notes."])
        assert len(resolved) == 1
        pointer, _ = resolved[0]
        assert (tmp_path / pointer.rel_posix_path).exists()
    finally:
        engine.close()
