"""Benchmark input, fingerprinting, embedding stability, and loopback integration tests."""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, override

import numpy as np
import pytest
from openai import AuthenticationError

import benchmark
from yaw.core import YawConfig, YawRuntimeError
from yaw.datasets import prepare_dataset_bundle
from yaw.pipeline import list_user_tables
from yaw.qa import build_qa_agent
from yaw.retrieval import open_readonly_db

_CHUNK_ID_RE = re.compile(r"\[chunk_id=(\d+)\]")


@pytest.fixture
def benchmark_inference_server():
    requests: list[tuple[str, str | None, dict[str, Any] | None]] = []

    def _embed_text(text: str) -> list[float]:
        low = text.lower()
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
        vector = np.random.default_rng(seed).standard_normal(64) * 0.05
        for idx, kw in enumerate(("ada", "notes", "wrote", "turing", "cambridge", "worked")):
            if kw in low:
                vector[idx] += 2.0
        vector /= max(float(np.linalg.norm(vector)), 1e-12)
        return [float(x) for x in vector]

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            auth = self.headers.get("Authorization")
            requests.append((self.path, auth, None))
            if auth != "Bearer benchmark-key":
                self.send_error(401)
                return
            self.send_error(404)

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            auth = self.headers.get("Authorization")
            requests.append((self.path, auth, body))
            if auth != "Bearer benchmark-key":
                self.send_error(401)
                return

            if self.path == "/v1/embeddings":
                assert body["model"] == "embeddinggemma-2"
                raw_in = body["input"]
                texts = [raw_in] if isinstance(raw_in, str) else [str(x) for x in raw_in]
                values = [_embed_text(text) for text in texts]
                response: dict[str, Any] = {
                    "object": "list",
                    "model": body["model"],
                    "data": [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(values)],
                    "usage": {"prompt_tokens": 20, "total_tokens": 20},
                }
            elif self.path == "/v1/chat/completions":
                assert body["model"] == "gemma-4-e4b"
                messages = body.get("messages", [])
                tool_msgs = [m for m in messages if m.get("role") == "tool"]
                user_text = " ".join(str(m.get("content", "")) for m in messages if m.get("role") == "user")
                if not tool_msgs:
                    q_match = re.search(r"\[\[ ## question ## \]\]\s*(.+?)(?:\n\[\[|\Z)", user_text, re.DOTALL)
                    query_str = q_match.group(1).strip() if q_match else user_text.strip()
                    call_name = "search_passages"
                    call_args: dict[str, Any] = {"query": query_str, "top_k": 5}
                else:
                    tool_blob = "\n".join(str(m.get("content", "")) for m in tool_msgs)
                    cids = _CHUNK_ID_RE.findall(tool_blob)
                    cite_suffix = f" [chunk_id={cids[0]}]" if cids else ""
                    low_blob = tool_blob.lower()
                    if "cambridge" in low_blob and "turing" in low_blob:
                        ans = "Cambridge"
                    elif "ada wrote notes" in low_blob or "ada" in low_blob:
                        ans = "Ada"
                    else:
                        ans = "Insufficient references in corpus"
                    call_name = "submit"
                    call_args = {"answer": f"{ans}{cite_suffix}"}

                response = {
                    "id": f"benchmark-chat-{len(requests)}",
                    "object": "chat.completion",
                    "created": 1,
                    "model": body["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": f"call_{len(requests)}",
                                        "type": "function",
                                        "function": {
                                            "name": call_name,
                                            "arguments": json.dumps(call_args),
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
                }
            else:
                self.send_error(404)
                return

            raw = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        @override
        def log_message(self, format: str, *args: object) -> None:
            _ = format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("  Ada ", "Ada"), ('Answer: "Ada"', "Ada"), ("Answer: distractor Answer: 'Ada'", "Ada"), ("", "")],
)
def test_qa_answer_cleanup_preserves_only_the_final_answer(raw: str, expected: str) -> None:
    assert benchmark.clean_qa_answer(raw) == expected


@pytest.mark.parametrize(
    ("sample", "expected"),
    [
        ({"reference": "Ada"}, ["Ada"]),
        ({"gold_ans": "Ada", "answer_aliases": ["Lovelace", "Ada"]}, ["Ada", "Lovelace"]),
        ({"obj": "Ada", "possible_answers": '["Lovelace"]', "o_aliases": "not JSON"}, ["Ada", "Lovelace", "not JSON"]),
        ({"obj": "Ada", "possible_answers": ["Lovelace"], "o_aliases": 3}, ["Ada", "Lovelace", "3"]),
        ({"answer": [3, "3"], "answer_aliases": [3]}, ["3"]),
    ],
)
def test_answer_variants_keep_aliases_ordered_and_deduplicated(sample: dict[str, Any], expected: list[str]) -> None:
    assert benchmark.get_gold_answers([sample]) == [expected]


@pytest.mark.parametrize("sample", [{}, {"answer": None}])
def test_absent_or_null_answers_are_rejected(sample: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="answer|Answer"):
        benchmark.get_gold_answers([sample])


def test_benchmark_config_fingerprint_is_deterministic_and_tracks_live_fields(tmp_path: Path) -> None:
    base = YawConfig(workspace_root=tmp_path)
    fp_base = benchmark.benchmark_config_fingerprint(base)
    assert len(fp_base) == 64
    assert fp_base == benchmark.benchmark_config_fingerprint(YawConfig(workspace_root=tmp_path))

    mutations = [
        replace(base, corpus_root=tmp_path / "other_corpus"),
        replace(base, included_patterns=("*.md",)),
        replace(base, excluded_patterns=(".git/*",)),
        replace(base, chunk_size=500),
        replace(base, chunk_overlap=50),
        replace(base, min_chunk_size=25),
        replace(base, embedding_model="other-embed"),
        replace(base, embedding_task_mode="search"),
        replace(base, qa_model="other-qa"),
        replace(base, max_output_tokens=2048),
        replace(base, react_max_iters=4),
    ]
    for mutated in mutations:
        assert benchmark.benchmark_config_fingerprint(mutated) != fp_base


def test_integrity_gate_preserves_provider_errors_and_respects_retrieval_only_mode() -> None:
    records = [
        {"answer": "Ada", "retrieved_chunk_ids": []},
        {"answer": "", "retrieved_chunk_ids": []},
    ]

    def evaluate(include_qa: bool) -> list[str]:
        return benchmark.check_benchmark_integrity(
            records,
            extraction_errors=["source failed"],
            qa_errors=["reader failed"],
            graph_info={"num_passage_nodes": 0},
            expected_doc_count=1,
            include_qa=include_qa,
        )

    errors = evaluate(True)
    assert len(errors) == 6
    assert "extraction_llm: source failed" in errors
    assert "qa_llm: reader failed" in errors
    assert "Question 2: empty QA answer" in errors
    assert any("No chunks indexed" in error for error in errors)
    assert any("empty retrieval results" in error for error in errors)

    retrieval_only = evaluate(False)
    assert len(retrieval_only) == 4
    assert not any("qa_llm" in error or "empty QA answer" in error for error in retrieval_only)


@pytest.mark.parametrize(
    ("sample", "expected"),
    [
        ({"gold_docs": ["one", {"title": "Title", "text": "Body"}, "one"]}, ["one", "Title\nBody"]),
        ({"context": [["Title", ["One", "Two"]]]}, ["Title\nOne Two"]),
        ({"contexts": [{"title": "Title", "text": "Body", "is_supporting": True}]}, ["Title\nBody"]),
        ({"paragraphs": [{"title": "Title", "paragraph_text": "Body", "is_supporting": True}]}, ["Title\nBody"]),
    ],
)
def test_gold_document_fallbacks_preserve_exact_format_and_empty_inputs(
    sample: dict[str, Any], expected: list[str]
) -> None:
    assert benchmark.get_gold_docs([sample]) == [expected]


def test_qa_only_datasets_return_none_for_gold_docs() -> None:
    sample = {"question": "Who?", "answer": "absent", "document": {"id": "doc"}}
    corpus = [
        {"idx": "other_1", "title": "Other", "text": "Not part of this document"},
        {"idx": "doc_1", "title": "First", "text": "Original context"},
    ]
    assert benchmark.get_gold_docs([sample], "lveval") is None
    assert benchmark.get_gold_docs([sample], "narrativeqa", corpus) is None


@pytest.mark.parametrize("dataset", ["custom", "popqa", "lveval", "narrativeqa"])
def test_benchmark_slice_pads_from_real_corpus_without_dropping_gold(dataset: str) -> None:
    source = {"idx": "doc_1", "title": "Notes", "text": "Ada wrote Notes."}
    distractors = [{"idx": f"other_{i}", "title": f"Other {i}", "text": f"Unrelated context {i}."} for i in range(5)]
    sample: dict[str, Any] = {"question": "Who wrote Notes?", "answer": "Ada"}
    if dataset == "custom":
        sample["contexts"] = [{**source, "is_supporting": True}, {**distractors[0], "is_supporting": False}]
    elif dataset == "popqa":
        sample["paragraphs"] = [{**source, "is_supporting": True}]
    elif dataset == "lveval":
        sample["context"] = "### Passage 1\nAda wrote Notes."
    else:
        sample["document"] = {"id": "doc", "summary": {"title": "Summary", "text": "Ada wrote Notes."}}
    docs, queries, gold, answers = benchmark.build_benchmark_slice(
        dataset,
        [sample],
        [source, *distractors],
        max_queries=1,
        index_queries=2,
    )
    assert queries == ["Who wrote Notes?"] and answers == [["Ada"]]
    selected = {f"{row['title']}\n{row['text']}" for row in [source, *distractors] if row["idx"] in docs}
    if dataset in ("lveval", "narrativeqa"):
        assert gold is None
        assert len(docs) == 6
    else:
        assert gold is not None
        assert gold == [["Notes\nAda wrote Notes."]]
        assert all(reference in selected for reference in gold[0])
        assert len(docs) == (2 if dataset == "custom" else 4)
    assert len(docs) == len(set(docs))


def test_dataset_loading_requires_prepared_bundle_and_preserves_source_files(tmp_path: Path) -> None:
    corpus, qa = tmp_path / "corpus.json", tmp_path / "questions.json"
    corpus.write_text(json.dumps([{"title": "Title", "text": "Body"}]), encoding="utf-8")
    qa.write_text(json.dumps([{"question": "Who?", "answer": "Ada"}]), encoding="utf-8")
    original = corpus.read_bytes(), qa.read_bytes()
    with pytest.raises(FileNotFoundError):
        benchmark.load_dataset_files("sample", tmp_path)
    prepare_dataset_bundle("sample", corpus, qa, tmp_path)
    samples, rows = benchmark.load_dataset_files("sample", tmp_path)
    assert samples == json.loads(qa.read_text(encoding="utf-8"))
    assert rows[0]["title"] == "Title" and rows[0]["text"] == "Body"
    assert (corpus.read_bytes(), qa.read_bytes()) == original
    with pytest.raises(ValueError):
        benchmark.load_dataset_files("sample", tmp_path, max_queries=0)


@pytest.mark.parametrize("mode", ["search", "distinct-titled"])
def test_embedding_verification_compares_all_active_roles_under_reordering(mode: str) -> None:
    def embed(texts: list[str]) -> np.ndarray:
        return np.array(
            [[byte + 1 for byte in hashlib.sha256(text.encode()).digest()[:8]] for text in texts],
            dtype=np.float32,
        )

    probes = benchmark.verify_embeddings(
        embed,
        ["Who wrote Notes?", "Who was Ada?"],
        ["Notes\nAda wrote Notes."],
        task_mode=mode,
    )
    expected = ["query", "document"] if mode == "search" else ["query", "fact_query", "document", "titled_document"]
    assert [probe["role"] for probe in probes] == expected
    assert all(probe["repeats"] == 5 and probe["dimensions"] == 8 for probe in probes)
    assert all(probe["min_pairwise_cosine"] == pytest.approx(1.0) for probe in probes)


def test_embedding_verification_rejects_changed_vectors_for_identical_inputs() -> None:
    calls = 0

    def changing_embed(texts: list[str]) -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.tile(np.array([1.0 if calls % 2 else -1.0, 0.0], dtype=np.float32), (len(texts), 1))

    with pytest.raises(RuntimeError, match="embeddings are unstable"):
        benchmark.verify_embeddings(changing_embed, ["question"], ["document"])


@pytest.mark.parametrize("include_qa", [False, True])
def test_full_native_benchmark_reports_real_ingestion_retrieval_and_restart_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, benchmark_inference_server, include_qa: bool
) -> None:
    base_url, requests = benchmark_inference_server
    monkeypatch.setattr(benchmark, "PROJECT_ROOT", tmp_path)
    corpus, qa = tmp_path / "corpus.json", tmp_path / "qa.json"
    corpus.write_text(json.dumps([{"idx": "notes", "title": "Notes", "text": "Ada wrote Notes."}]), encoding="utf-8")
    qa.write_text("[]", encoding="utf-8")
    prepare_dataset_bundle("musique", corpus, qa, tmp_path / "outputs/datasets")
    config = YawConfig(
        workspace_root=tmp_path,
        base_url=base_url,
        api_key="benchmark-key",
        embedding_model="embeddinggemma-2",
        qa_model="gemma-4-e4b",
        min_chunk_size=10,
    )

    def run(report: dict[str, Any]) -> None:
        benchmark.run_yaw_benchmark(
            dataset="musique",
            document_ids=["notes"],
            queries=["Who wrote Notes?"],
            gold_docs=[["Notes\nAda wrote Notes."]],
            gold_answers=[["Ada"]],
            config=config,
            include_qa=include_qa,
            compiled_program=None,
            index_dir=tmp_path / "index",
            report=report,
        )

    report: dict[str, Any] = {}
    run(report)
    assert not (tmp_path / "index" / "corpus").exists()
    assert report["graph"]["num_passage_nodes"] == 1
    assert report["graph"]["num_indexed_docs"] == 1
    assert report["integrity"]["passed"] is True
    metrics = report["evaluation_metrics"]
    assert metrics["hipporag"]["retrieval"]["Recall@1"] == 1.0
    assert metrics["standard_dpr"]["retrieval"]["Recall@1"] == 1.0
    assert metrics["hipporag"]["qa"] == ({"ExactMatch": 1.0, "F1": 1.0} if include_qa else None)
    assert len(report["records"]) == 1
    assert report["records"][0]["retrieved_chunk_ids"]
    assert report["records"][0]["qa_evidence"] == ["# Notes\n\nAda wrote Notes."]
    db_file = tmp_path / "index" / "yaw.db"
    assert db_file.is_file()
    with contextlib.closing(open_readonly_db(db_file)) as conn:
        assert list_user_tables(conn) == ("chunk_fts", "chunk_vocab", "chunks", "citation_graph", "rabitq_index")

    req_idx_before = len(requests)
    resumed: dict[str, Any] = {}
    run(resumed)
    new_embed_inputs = [
        inp
        for route, _, body in requests[req_idx_before:]
        if route == "/v1/embeddings" and body is not None
        for inp in ([body["input"]] if isinstance(body["input"], str) else body["input"])
    ]
    assert not any(s.startswith("title: ") for s in new_embed_inputs)
    assert resumed["evaluation_metrics"] == metrics


@pytest.mark.parametrize(
    ("mode", "authorized"),
    [("embeddings", True), ("embeddings", False), ("retrieval", True), ("all", True)],
)
def test_benchmark_main_persists_success_or_failure_with_disjoint_probe_queries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, benchmark_inference_server, mode: str, authorized: bool
) -> None:
    base_url, requests = benchmark_inference_server
    monkeypatch.setattr(benchmark, "PROJECT_ROOT", tmp_path)
    secret = "benchmark-key" if authorized else "wrong-key"
    monkeypatch.setenv("UNSLOTH_STUDIO_AUTH_TOKEN", secret)
    (tmp_path / "yaw.toml").write_text(
        f'[inference]\nbase_url = "{base_url}"\nembedding_model = "embeddinggemma-2"\nqa_model = "gemma-4-e4b"\n'
        "[ingestion]\nchunk_size = 96\nchunk_overlap = 16\nmin_chunk_size = 30\n",
        encoding="utf-8",
    )
    data = tmp_path / "outputs" / "datasets"
    data.mkdir(parents=True)
    text = "Alan Turing worked in Cambridge. " * 7
    paragraph = {"title": "Alan Turing", "text": text, "is_supporting": True}
    (data / "sample.json").write_text(
        json.dumps(
            [
                {"question": "First question?", "answer": "Cambridge", "paragraphs": [paragraph]},
                {"question": "Where did Alan Turing work?", "answer": "Cambridge", "paragraphs": [paragraph]},
            ]
        ),
        encoding="utf-8",
    )
    (data / "sample_corpus.json").write_text(json.dumps([paragraph]), encoding="utf-8")
    prepare_dataset_bundle("sample", data / "sample_corpus.json", data / "sample.json", data)

    def run() -> dict[str, Any]:
        return benchmark.main(
            dataset="sample",
            mode=mode,
            max_queries=1,
            index_queries=1,
            query_offset=1,
            fresh_index=True,
        )

    if authorized:
        result = run()
        assert result["status"] == "passed"
        assert result["query_offset"] == 1
        assert result["num_queries"] == 1
        report_path = next((tmp_path / "outputs" / "benchmarks").glob("run-sample-*/report.json"))
        serialized = report_path.read_text(encoding="utf-8")
        assert secret not in serialized
        assert json.loads(serialized) == result
        assert requests and all(auth == "Bearer benchmark-key" for _, auth, _ in requests)
        if mode == "embeddings":
            assert len(result["embedding_probes"]) == 4
            assert result["embedding_probes"][0]["inputs"][0].endswith("Where did Alan Turing work?")
            assert {route for route, _, _ in requests} == {"/v1/embeddings"}
        elif mode == "retrieval":
            assert not (Path(result["index_dir"]) / "corpus").exists()
            assert result["graph"]["num_indexed_docs"] == 1
            assert result["integrity"]["passed"] is True
            assert result["evaluation_metrics"]["hipporag"]["retrieval"]["Recall@1"] == 1.0
            assert result["evaluation_metrics"]["hipporag"]["qa"] is None
            assert {route for route, _, _ in requests} == {"/v1/embeddings"}
        else:
            assert not (Path(result["index_dir"]) / "corpus").exists()
            assert len(result["embedding_probes"]) == 4
            assert result["graph"]["num_indexed_docs"] == 1
            assert result["integrity"]["passed"] is True
            assert result["evaluation_metrics"]["hipporag"]["retrieval"]["Recall@1"] == 1.0
            assert result["evaluation_metrics"]["hipporag"]["qa"] == {"ExactMatch": 1.0, "F1": 1.0}
            assert result["records"][0]["answer"] == "Cambridge"
            assert {route for route, _, _ in requests} == {"/v1/embeddings", "/v1/chat/completions"}
            with contextlib.closing(open_readonly_db(Path(result["index_dir"]) / "yaw.db")) as conn:
                assert list_user_tables(conn) == (
                    "chunk_fts",
                    "chunk_vocab",
                    "chunks",
                    "citation_graph",
                    "rabitq_index",
                )
                assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] > 1
    else:
        with pytest.raises(AuthenticationError):
            run()
        report_path = next((tmp_path / "outputs" / "benchmarks").glob("run-sample-*/report.json"))
        serialized = report_path.read_text(encoding="utf-8")
        assert secret not in serialized
        result = json.loads(serialized)
        assert result["status"] == "failed"
        assert "AuthenticationError" in result["error"]
        assert result["elapsed_seconds"] >= 0


def test_run_yaw_benchmark_and_main_fail_fast_on_qa_runtime_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BadCiteHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/v1/embeddings":
                raw_in = body["input"]
                texts = [raw_in] if isinstance(raw_in, str) else [str(x) for x in raw_in]
                response: dict[str, Any] = {
                    "object": "list",
                    "model": body["model"],
                    "data": [
                        {"object": "embedding", "index": i, "embedding": [1.0, 0.0, 0.0, 0.0]}
                        for i in range(len(texts))
                    ],
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                }
            elif self.path == "/v1/chat/completions":
                response = {
                    "id": "bad-cite",
                    "object": "chat.completion",
                    "created": 1,
                    "model": body["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_bad",
                                        "type": "function",
                                        "function": {
                                            "name": "submit",
                                            "arguments": json.dumps({"answer": "Ada [chunk_id=999999]"}),
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            else:
                self.send_error(404)
                return
            raw = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        @override
        def log_message(self, format: str, *args: object) -> None:
            _ = format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), BadCiteHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}/v1"
        monkeypatch.setattr(benchmark, "PROJECT_ROOT", tmp_path)
        monkeypatch.setenv("UNSLOTH_STUDIO_AUTH_TOKEN", "benchmark-key")
        (tmp_path / "yaw.toml").write_text(
            f'[inference]\nbase_url = "{base_url}"\nembedding_model = "embeddinggemma-2"\nqa_model = "gemma-4-e4b"\n'
            "[ingestion]\nmin_chunk_size = 10\n",
            encoding="utf-8",
        )
        corpus, qa = tmp_path / "corpus.json", tmp_path / "qa.json"
        doc_row = {"idx": "notes", "title": "Notes", "text": "Ada wrote Notes."}
        corpus.write_text(json.dumps([doc_row]), encoding="utf-8")
        qa.write_text(
            json.dumps(
                [{"question": "Who wrote Notes?", "answer": "Ada", "paragraphs": [{**doc_row, "is_supporting": True}]}]
            ),
            encoding="utf-8",
        )
        prepare_dataset_bundle("sample", corpus, qa, tmp_path / "outputs/datasets")
        config = YawConfig(
            workspace_root=tmp_path,
            base_url=base_url,
            api_key="benchmark-key",
            embedding_model="embeddinggemma-2",
            qa_model="gemma-4-e4b",
            min_chunk_size=10,
        )
        with pytest.raises(YawRuntimeError, match="ungrounded chunk_id=999999"):
            benchmark.run_yaw_benchmark(
                dataset="sample",
                document_ids=["notes"],
                queries=["Who wrote Notes?"],
                gold_docs=[["Notes\nAda wrote Notes."]],
                gold_answers=[["Ada"]],
                config=config,
                include_qa=True,
                index_dir=tmp_path / "index",
                report={},
            )

        with pytest.raises(YawRuntimeError, match="ungrounded chunk_id=999999"):
            benchmark.main(dataset="sample", mode="eval", max_queries=1, fresh_index=True)
        report_path = next((tmp_path / "outputs" / "benchmarks").glob("run-sample-*/report.json"))
        saved = json.loads(report_path.read_text(encoding="utf-8"))
        assert saved["status"] == "failed"
        assert "YawRuntimeError" in saved["error"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def test_benchmark_main_fails_fast_when_integrity_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, benchmark_inference_server
) -> None:
    base_url, _ = benchmark_inference_server
    monkeypatch.setattr(benchmark, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("UNSLOTH_STUDIO_AUTH_TOKEN", "benchmark-key")
    (tmp_path / "yaw.toml").write_text(
        f'[inference]\nbase_url = "{base_url}"\nembedding_model = "embeddinggemma-2"\nqa_model = "gemma-4-e4b"\n'
        "[ingestion]\nmin_chunk_size = 10\n",
        encoding="utf-8",
    )
    corpus, qa = tmp_path / "corpus.json", tmp_path / "qa.json"
    blank_doc = {"idx": "blank", "title": "", "text": "   "}
    corpus.write_text(json.dumps([blank_doc]), encoding="utf-8")
    qa.write_text(
        json.dumps([{"question": "Who?", "answer": "Ada", "paragraphs": [{**blank_doc, "is_supporting": True}]}]),
        encoding="utf-8",
    )
    prepare_dataset_bundle("sample", corpus, qa, tmp_path / "outputs/datasets")

    with pytest.raises(RuntimeError, match="Benchmark integrity failed"):
        benchmark.main(dataset="sample", mode="retrieval", max_queries=1, fresh_index=True)
    report_path = next((tmp_path / "outputs" / "benchmarks").glob("run-sample-*/report.json"))
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["status"] == "failed"
    assert saved["integrity"]["passed"] is False
    assert "Benchmark integrity failed" in saved["error"]


def test_parity_gate_propagates_configuration_and_invalidates_old_deployment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, benchmark_inference_server
) -> None:
    from scripts import verify_paper_parity_gate as gate

    base_url, requests = benchmark_inference_server
    corpus, questions = tmp_path / "corpus.json", tmp_path / "questions.json"
    text = "Alan Turing worked in Cambridge."
    corpus.write_text(json.dumps([{"idx": "turing", "title": "Alan Turing", "text": text}]), encoding="utf-8")
    questions.write_text(
        json.dumps(
            [
                {
                    "question": "Where did Alan Turing work?",
                    "answer": "Cambridge",
                    "context": [["Alan Turing", [text]]],
                }
            ]
        ),
        encoding="utf-8",
    )
    prepare_dataset_bundle("musique", corpus, questions, tmp_path / "outputs" / "datasets")
    config_path = tmp_path / "yaw.toml"
    config_text = (
        f'[inference]\nbase_url = "{base_url}"\nembedding_model = "embeddinggemma-2"\n'
        'qa_model = "gemma-4-e4b"\nmax_output_tokens = 1024\n'
        "[ingestion]\nmin_chunk_size = 1\n"
    )
    config_path.write_text(config_text, encoding="utf-8")
    monkeypatch.setattr(benchmark, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(gate, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("UNSLOTH_STUDIO_AUTH_TOKEN", "benchmark-key")

    def run_gate() -> dict[str, Any]:
        return gate.run_parity_gate(
            datasets=["musique"],
            queries_per_dataset=1,
            index_queries=1,
            compiled_program=str(tmp_path / "uncompiled.json"),
            mode="eval",
            config_path=config_path,
            output_json=tmp_path / "gate.json",
            output_md=tmp_path / "gate.md",
        )

    first = run_gate()
    assert first["config"]["qa_model"] == "gemma-4-e4b"
    assert first["config"]["inference"]["base_url"] == base_url
    assert first["datasets"]["musique"]["gate_passed"]
    first_run = first["datasets"]["musique"]["yaw"]["run_dir"]
    count = len(requests)
    repeated = run_gate()
    assert repeated["datasets"]["musique"]["yaw"]["run_dir"] == first_run
    assert len(requests) == count

    config_path.write_text(config_text.replace("1024", "2048"), encoding="utf-8")
    changed = run_gate()
    assert changed["config"]["inference"] != first["config"]["inference"]
    assert changed["datasets"]["musique"]["yaw"]["run_dir"] != first_run

    artifact = tmp_path / "uncompiled.json"
    build_qa_agent([lambda query: query]).save(str(artifact))
    changed_program = run_gate()
    assert changed_program["config"]["inference"] != changed["config"]["inference"]
    assert changed_program["datasets"]["musique"]["yaw"]["run_dir"] != changed["datasets"]["musique"]["yaw"]["run_dir"]

    artifact.write_bytes(artifact.read_bytes() + b"\n")
    changed_bytes = run_gate()
    assert changed_bytes["config"]["inference"] != changed_program["config"]["inference"]
    assert (
        changed_bytes["datasets"]["musique"]["yaw"]["run_dir"]
        != changed_program["datasets"]["musique"]["yaw"]["run_dir"]
    )
    assert all(auth == "Bearer benchmark-key" for _, auth, _ in requests)
    assert {route for route, _, _ in requests} <= {"/v1/embeddings", "/v1/chat/completions"}
