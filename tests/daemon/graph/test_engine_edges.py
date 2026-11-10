"""Native DSPy/HTTP indexing and durable OpenIE recovery against real SQLite."""

import hashlib
import json
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import dspy
import numpy as np
import pytest

from daemon.graph.synonym_protocol import candidate_rows, complete_decisions
from yaw.core import ChunkPointer
from yaw.daemon.graph.engine import YawGraphEngine
from yaw.daemon.graph.inference import YawProgram
from yaw.daemon.graph.inference.embedding import DSPyEmbeddingAdapter, create_dspy_embedder, resolve_embedding_profile
from yaw.daemon.indexing import bind_native_indexing


@pytest.fixture
def indexing_provider(request):
    calls = []
    empty_decisions = getattr(request, "param", False)

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, payload):
            raw = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path.endswith("health"):
                assert isinstance(self.server, ThreadingHTTPServer)
                self.send_json(
                    {
                        "all_models_loaded": [
                            {
                                "id": "indexing-contract",
                                "backend_url": f"http://127.0.0.1:{self.server.server_port}",
                                "launch_command": ["-m", "indexing-contract.gguf"],
                            }
                        ]
                    }
                )
            else:
                self.send_json(
                    {
                        "model_path": "indexing-contract.gguf",
                        "chat_template": "bytes",
                        "default_generation_settings": {"n_ctx": 131072},
                    }
                )

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path.endswith("apply-template"):
                self.send_json({"prompt": json.dumps(request["messages"])})
                return
            if self.path.endswith("tokenize"):
                self.send_json({"tokens": list(request["content"].encode())})
                return
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
            elif "candidate_pairs" in prompt or "synonym-evidence" in prompt:
                text = request["messages"][-1]["content"]
                stage, result = (
                    "synonyms",
                    {
                        "decisions": []
                        if empty_decisions
                        else complete_decisions(candidate_rows(text), [], stage_text=prompt)
                    },
                )
            elif "reasoning" in prompt.lower():
                stage, result = "qa", {"reasoning": "The source states Ada wrote Notes.", "answer": "Answer: 'Ada'"}
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
    program = YawProgram(enable_gbnf=False, orchestrator_lm=lm, rlm_sub_lm=lm)

    def embed(texts):
        return np.array(
            [[byte + 1 for byte in hashlib.sha256(text.encode()).digest()[:8]] for text in texts], dtype=np.float32
        )

    embedder = DSPyEmbeddingAdapter(
        replace(resolve_embedding_profile(), dimensions=8),
        create_dspy_embedder(embed, caching=False),
    )
    return bind_native_indexing(
        YawGraphEngine(db_path=tmp_path / "yaw.db", workspace_root=tmp_path, program=program, embedder=embedder)
    )


def test_native_inference_survives_query_cache_clear_and_engine_restart(tmp_path, indexing_provider):
    url, calls = indexing_provider
    source = "Ada wrote Notes."
    (tmp_path / "source.txt").write_text(source, encoding="utf-8")
    pointer = ChunkPointer("source", "source.txt", 1, 1, "Source")
    first = make_engine(tmp_path, url)
    try:
        assert first.index_chunks([(pointer, source)]) == 1
        assert first.conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 1
        first._encode_cached_text.cache_clear()
        checkpoint = tuple(
            first.conn.execute(
                "SELECT program_identity, embedding_identity, extraction_json, traces_json FROM chunk_inference"
            ).fetchone()
        )
    finally:
        first.close()
    before = list(calls)
    second = make_engine(tmp_path, url)
    try:
        second.index_chunks([(pointer, source)])
        assert calls == before
        assert (
            tuple(
                second.conn.execute(
                    "SELECT program_identity, embedding_identity, extraction_json, traces_json FROM chunk_inference"
                ).fetchone()
            )
            == checkpoint
        )
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
        assert engine.index_chunks([]) == 0
        assert calls == []
        assert engine.conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0
    finally:
        engine.close()


@pytest.mark.parametrize("indexing_provider", [True], indirect=True)
def test_unadjudicated_synonym_candidates_abort_graph_publication(tmp_path, indexing_provider):
    url, calls = indexing_provider
    engine = make_engine(tmp_path, url)
    engine.candidate_dense_threshold = 0.1
    try:
        with pytest.raises(ValueError, match="Missing synonym pairs|Empty decisions"):
            engine.index(["Ada wrote Notes."])
        assert "synonyms" in calls
        for table in ("chunks", "entities", "facts", "edges", "edge_sources", "chunk_fts"):
            assert engine.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
        resolved = engine._resolve_chunks_for_docs(["Ada wrote Notes."])
        assert len(resolved) == 1
        pointer, _ = resolved[0]
        assert pointer.rel_posix_path == "passages.json"
        assert (tmp_path / pointer.rel_posix_path).exists()
        assert not (tmp_path / "corpus").exists()
    finally:
        engine.close()


def test_slice5_passages_json_shared_corpus_pools_and_unbounded_eval_hydration(tmp_path, indexing_provider):
    url, _ = indexing_provider
    assert not hasattr(YawGraphEngine, "_materialize_doc_chunk")
    engine = make_engine(tmp_path, url)
    try:
        # 1. Complete raw-document snapshots preserve unchanged spans without materializing corpus/chunk-*.md.
        assert engine.index(["First Title\nAda wrote Notes on the Analytical Engine."]) == 1
        assert (
            engine.index(
                [
                    "First Title\nAda wrote Notes on the Analytical Engine.",
                    "Second Title\nAda wrote Notes on Babbage's machine.",
                ]
            )
            == 2
        )
        assert not (tmp_path / "corpus").exists()
        passages_path = tmp_path / "passages.json"
        assert passages_path.is_file()
        loaded = json.loads(passages_path.read_text(encoding="utf-8"))
        assert len(loaded) == 2
        rows = engine.conn.execute(
            "SELECT rel_posix_path, start_line, end_line, ast_island_start, ast_island_end "
            "FROM chunks ORDER BY start_line"
        ).fetchall()
        assert [tuple(r) for r in rows] == [
            ("passages.json", 2, 6, 2, 6),
            ("passages.json", 7, 11, 7, 11),
        ]

        # 2. _resolve_chunks_for_docs matches title-prepended sub-chunks and never materializes unknown docs to disk
        part1 = "First section of the engineering specification with detailed telemetry."
        part2 = "Second section covering cryogenic phase lock stabilization at 42.5 MHz."
        full_doc = f"[github] Spec Doc\n{part1}\n\n{part2}"
        sub_ptr_1 = ChunkPointer("chunk-sub1", "corpus.json", 2, 10, "[github] Spec Doc", 2, 10)
        sub_ptr_2 = ChunkPointer("chunk-sub2", "corpus.json", 2, 10, "[github] Spec Doc", 2, 10)
        engine._opt_chunks_pool.extend(
            [
                (sub_ptr_1, f"[github] Spec Doc\n{part1}"),
                (sub_ptr_2, f"[github] Spec Doc\n{part2}"),
            ]
        )
        engine._resolved_docs_cache.clear()
        matched = engine._resolve_chunks_for_docs([full_doc, "Unindexed doc that must not be materialized"])
        assert [p.chunk_id for p, _ in matched] == ["chunk-sub1", "chunk-sub2"]
        direct_cid_matched = engine._resolve_chunks_for_docs(["chunk-sub2"])
        assert [p.chunk_id for p, _ in direct_cid_matched] == ["chunk-sub2"]
        assert not (tmp_path / "corpus").exists()

        # 3. Shared-corpus (>64 chunks) pool preservation in _prepare_optimization_pools & _ensure_candidate_chunks

        shared_records = [
            {
                "id": f"chunk-shared-{i:03d}",
                "title": f"Doc {i}",
                "text": f"Doc {i}\nShared corpus passage number {i} discussing Ada and Notes in detail.",
            }
            for i in range(75)
        ]
        rendered = [
            "\n".join("  " + ln for ln in json.dumps(rec, ensure_ascii=False, indent=2).splitlines())
            for rec in shared_records
        ]
        shared_path = tmp_path / "shared_corpus.json"
        shared_path.write_text("[\n" + ",\n".join(rendered) + "\n]\n", encoding="utf-8")

        engine._opt_chunks_pool.clear()
        engine._opt_val_chunks_pool.clear()
        # Leave _resolved_docs_cache populated so _prepare_optimization_pools is verified to clear it
        assert len(engine._resolved_docs_cache) > 0
        shared_pairs: list[tuple[ChunkPointer, str]] = []
        curr_ln = 2
        for rec, item_s in zip(shared_records, rendered, strict=True):
            s_ln = curr_ln
            e_ln = s_ln + item_s.count("\n")
            curr_ln = e_ln + 1
            ptr = ChunkPointer(rec["id"], "shared_corpus.json", s_ln, e_ln, rec["title"], s_ln, e_ln)
            shared_pairs.append((ptr, rec["text"]))

        engine.index_chunks(shared_pairs)

        shared_train = [dspy.Example(question=f"Question {i}?", item_docs=[]).with_inputs("question") for i in range(3)]
        shared_val = [
            dspy.Example(question=f"Val Question {i}?", item_docs=[]).with_inputs("question") for i in range(2)
        ]
        save_target = tmp_path / "opt_out.json"
        engine._prepare_optimization_pools(
            shared_train,
            shared_val,
            None,
            mode="gepa",
            save_path=save_target,
            progress_callback=None,
        )
        assert len(engine._resolved_docs_cache) == 0
        assert len(engine._opt_val_chunks_pool) == 75
        assert engine._graph_rlm_stats["val_pool_size"] == 75
        assert engine._graph_rlm_stats["train_pool_size"] == 75
        assert engine._graph_rlm_stats["corpus_pool_size"] == 75
        pools_meta = json.loads((tmp_path / "opt_out_pools.json").read_text(encoding="utf-8"))
        assert len(pools_meta["val_chunk_ids"]) == 75
        assert len(pools_meta["train_chunk_ids"]) == 75
        assert len(pools_meta["corpus_chunk_ids"]) == 75

        ensured_cids = engine._ensure_candidate_chunks_indexed(engine.program, item_docs=None)
        assert len(ensured_cids) == 75

        # 4. _hydrate_top_chunks_for_eval hydrates all 20 chunks even when max_context_chars is small
        engine.max_context_chars = 300
        top_20 = [f"chunk-shared-{i:03d}" for i in range(20)]
        hydrated = engine._hydrate_top_chunks_for_eval(top_20, engine.program)
        assert len(hydrated) == 20
        assert all(f"Shared corpus passage number {i}" in hydrated[f"chunk-shared-{i:03d}"] for i in range(20))

        # 5. When examples have item_docs=[] and non-empty gold_docs (Enterprise RAG contract),
        #    _prepare_optimization_pools narrows val_pool and train_pool to gold_docs and maps questions.
        gold_train = [
            dspy.Example(
                question=f"Gold Train Q{i}?",
                item_docs=[],
                gold_docs=[shared_records[i]["text"]],
            ).with_inputs("question", "item_docs")
            for i in range(3)
        ]
        gold_val = [
            dspy.Example(
                question=f"Gold Val Q{i}?",
                item_docs=[],
                gold_docs=[shared_records[10 + i]["text"]],
            ).with_inputs("question", "item_docs")
            for i in range(2)
        ]
        save_gold_target = tmp_path / "opt_gold_out.json"
        engine._prepare_optimization_pools(
            gold_train,
            gold_val,
            None,
            mode="gepa",
            save_path=save_gold_target,
            progress_callback=None,
        )
        assert len(engine._opt_val_chunks_pool) == 2
        assert engine._graph_rlm_stats["val_pool_size"] == 2
        assert engine._graph_rlm_stats["train_pool_size"] == 3
        assert engine._graph_rlm_stats["corpus_pool_size"] == 75
        assert engine._opt_docs_by_question["Gold Train Q0?"] == [shared_records[0]["text"]]
        assert engine._opt_docs_by_question["Gold Val Q1?"] == [shared_records[11]["text"]]
        ensured_q0 = engine._ensure_candidate_chunks_indexed(
            engine.program,
            item_docs=engine._opt_docs_by_question["Gold Train Q0?"],
        )
        assert ensured_q0 == ["chunk-shared-000"]
    finally:
        engine.close()


def test_format_qa_context_subgraph_and_dpr_separation(tmp_path: Path, indexing_provider) -> None:
    """Verify format_qa_context prepends induced subgraph when enabled and preserves flat passages when disabled."""
    from yaw.core import ChunkPointer
    from yaw.daemon.graph.storage import (
        upsert_chunk_pointers,
        upsert_entities,
        upsert_entity_labels,
        upsert_fact_sources,
        upsert_facts,
        upsert_semantic_edges,
    )

    url, _ = indexing_provider
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir(parents=True)
    doc_file = corpus_dir / "notes.txt"
    doc_file.write_text("Line 1\nLine 2: Ada Lovelace wrote the first computer algorithm.\nLine 3\n", encoding="utf-8")

    engine = make_engine(tmp_path, url)
    try:
        conn = engine.conn
        assert conn is not None

        ptr = ChunkPointer(
            chunk_id="c_ada",
            rel_posix_path="corpus/notes.txt",
            start_line=2,
            end_line=2,
            title="Ada Notes",
            ast_island_start=1,
            ast_island_end=3,
        )
        upsert_chunk_pointers(conn, [(ptr, "Ada Lovelace wrote the first computer algorithm.")])
        upsert_entities(
            conn,
            [
                ("e_ada", "Ada Lovelace", "English mathematician and writer"),
                ("e_algo", "Computer Algorithm", "Computational instruction sequence"),
            ],
        )
        upsert_entity_labels(
            conn,
            [
                ("ada lovelace", "e_ada", "skos:prefLabel", "Ada Lovelace", "c_ada"),
                ("computer algorithm", "e_algo", "skos:prefLabel", "Computer Algorithm", "c_ada"),
            ],
        )
        upsert_facts(
            conn,
            [
                (
                    "f_ada",
                    "e_ada",
                    "authored",
                    "entity",
                    "e_algo",
                    None,
                    "",
                    "",
                    '["Ada Lovelace", "authored", "Computer Algorithm"]',
                    "Ada authored the algorithm.",
                ),
            ],
        )
        upsert_fact_sources(conn, [("f_ada", "c_ada")])
        upsert_semantic_edges(
            conn,
            [
                ("e_ada", "e_algo", "skos:related", 0.9, "Creator to invention association"),
            ],
        )

        hits = [(ptr, 0.95, "Line 2: Ada Lovelace wrote the first computer algorithm.")]

        # 1. GraphRAG QA Context (include_subgraph=True, default)
        ctx_graph = engine.format_qa_context(
            hits,
            seed_facts=[("Ada Lovelace", "authored", "Computer Algorithm")],
            include_subgraph=True,
        )
        assert "### Knowledge Graph Statements (rdf:Statement)" in ctx_graph
        assert "[Ada Lovelace, authored, Computer Algorithm]" in ctx_graph
        assert "### Taxonomy & Disambiguation Trace (SKOS / OWL)" in ctx_graph
        assert "skos:related" in ctx_graph
        assert "Creator to invention association" in ctx_graph
        assert "### Retrieved Passages" in ctx_graph
        assert "Ada Lovelace wrote the first computer algorithm" in ctx_graph

        # 2. DPR QA Context (include_subgraph=False)
        ctx_dpr = engine.format_qa_context(
            hits,
            include_subgraph=False,
        )
        assert "### Knowledge Graph Statements" not in ctx_dpr
        assert "### Taxonomy & Disambiguation Trace" not in ctx_dpr
        assert "### Retrieved Passages" not in ctx_dpr
        assert "Ada Lovelace wrote the first computer algorithm" in ctx_dpr
    finally:
        engine.close()
