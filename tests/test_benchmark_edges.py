"""Benchmark input and repeatability contracts on real files and native arrays."""

import hashlib
import json
from dataclasses import replace
from functools import partial
from typing import Any

import dspy
import numpy as np
import pytest
from daemon.graph.inference.test_registry_edges import embedding_entry
from daemon.graph.inference.test_registry_edges import registry_server as registry_server
from daemon.graph.test_engine_edges import indexing_provider as indexing_provider
from scripts.test_download_edges import download_server as download_server

import benchmark
from yaw.daemon.graph.inference.embedding import (
    MODEL,
    PROFILES,
    DSPyEmbeddingAdapter,
    create_dspy_embedder,
    resolve_embedding_profile,
)
from yaw.daemon.graph.inference.models import embedding_evidence, preflight
from yaw.daemon.ingestion import prepare_dataset_bundle


@pytest.fixture(autouse=True)
def restore_benchmark_lm_binding():
    previous = dspy.settings.lm
    try:
        yield
    finally:
        dspy.configure(lm=previous)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("  Ada ", "Ada"), ('Answer: "Ada"', "Ada"), ("Answer: distractor Answer: 'Ada'", "Ada"), ("", "")],
)
def test_qa_answer_cleanup_preserves_only_the_final_answer(raw, expected):
    assert benchmark.clean_qa_answer(raw) == expected


@pytest.mark.parametrize(
    ("sample", "expected"),
    [
        ({"reference": "Ada"}, ["Ada"]),
        ({"gold_ans": "Ada", "answer_aliases": ["Lovelace", "Ada"]}, ["Ada", "Lovelace"]),
        ({"obj": "Ada", "possible_answers": '"Lovelace"', "o_aliases": "not JSON"}, ["Ada", "Lovelace", "not JSON"]),
        ({"obj": "Ada", "possible_answers": ["Lovelace"], "o_aliases": 3}, ["Ada", "Lovelace", "3"]),
        ({"answer": [3, "3"], "answer_aliases": [3]}, ["3"]),
    ],
)
def test_answer_variants_keep_aliases_ordered_and_deduplicated(sample, expected):
    assert benchmark.get_gold_answers([sample]) == [expected]


@pytest.mark.parametrize("sample", [{}, {"answer": None}])
def test_absent_or_null_answers_are_rejected(sample):
    with pytest.raises(ValueError, match="answer|Answer"):
        benchmark.get_gold_answers([sample])


def test_integrity_gate_preserves_provider_errors_and_respects_retrieval_only_mode():
    records = [
        {"answer": "Ada", "dpr_answer": "", "dpr_qa_metrics": {}, "graph_seeds": []},
        {"answer": "", "graph_seeds": []},
    ]

    def evaluate(include_qa):
        return benchmark.check_benchmark_integrity(
            records,
            extraction_errors=["source failed"],
            qa_errors=["reader failed"],
            graph_info={"num_passage_nodes": 2, "num_fact_nodes": 0},
            expected_doc_count=1,
            include_qa=include_qa,
        )

    errors = evaluate(True)
    assert len(errors) == 7
    assert "extraction_llm: source failed" in errors and "qa_llm: reader failed" in errors
    assert "Question 1: empty StandardRAG (DPR) answer" in errors
    assert "Question 2: empty HippoRAG answer" in errors
    retrieval_only = evaluate(False)
    assert len(retrieval_only) == 4
    assert not any("qa_llm" in error or "empty HippoRAG answer" in error for error in retrieval_only)


@pytest.mark.parametrize(
    ("sample", "expected"),
    [
        ({"gold_docs": ["one", {"title": "Title", "text": "Body"}, "one"]}, ["one", "Title\nBody"]),
        ({"context": [["Title", ["One", "Two"]]]}, ["Title\nOne Two"]),
        ({"context": "", "answer": "missing"}, []),
        ({"context": "unrelated passage", "answer": "missing"}, ["\nunrelated passage"]),
        ({"document": {"summary": "  Summary  "}}, ["Summary"]),
        ({"document": {"text": "  Full text  "}}, ["Full text"]),
        ({"document": {}}, []),
    ],
)
def test_gold_document_fallbacks_preserve_exact_format_and_empty_inputs(sample, expected):
    assert benchmark.get_gold_docs([sample]) == [expected]


def test_narrative_document_without_answer_match_uses_first_matching_corpus_chunk():
    sample = {"question": "Who?", "answer": "absent", "document": {"id": "doc"}}
    corpus = [
        {"idx": "other_1", "title": "Other", "text": "Not part of this document"},
        {"idx": "doc_1", "title": "First", "text": "Original context"},
    ]
    assert benchmark.get_gold_docs([sample], "narrativeqa", corpus) == [["First\nOriginal context"]]


@pytest.mark.parametrize("dataset", ["custom", "popqa", "lveval", "narrativeqa"])
def test_benchmark_slice_pads_from_real_corpus_without_dropping_gold(dataset):
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
    # LV-Eval's labels omit corpus titles; verify body identity in the adapter.
    assert all(
        any(reference.strip() == text or reference.strip() == text.split("\n", 1)[-1] for text in selected)
        for reference in gold[0]
    )
    assert len(docs) == len(set(docs))
    assert len(docs) == (2 if dataset == "custom" else 4)


def test_dataset_loading_requires_prepared_bundle_and_preserves_source_files(tmp_path):
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
def test_embedding_verification_compares_all_active_roles_under_reordering(mode):
    def embed(texts):
        return np.array(
            [[byte + 1 for byte in hashlib.sha256(text.encode()).digest()[:8]] for text in texts], dtype=np.float32
        )

    profile = replace(resolve_embedding_profile(task_mode=mode), dimensions=8)
    adapter = DSPyEmbeddingAdapter(profile, create_dspy_embedder(embed, caching=False))
    probes = benchmark.verify_embeddings(
        adapter, ["Who wrote Notes?", "Who was Ada?"], ["Title: Notes\nAda wrote Notes."]
    )
    expected = ["query", "document"] if mode == "search" else ["query", "fact_query", "document", "titled_document"]
    assert [probe["role"] for probe in probes] == expected
    assert all(probe["repeats"] == 5 and probe["dimensions"] == 8 for probe in probes)
    assert all(probe["min_pairwise_cosine"] == pytest.approx(1.0) for probe in probes)


def test_embedding_verification_rejects_changed_vectors_for_identical_inputs():
    calls = 0

    def changing_embed(texts):
        nonlocal calls
        calls += 1
        return np.tile(np.array([1.0 if calls % 2 else -1.0, 0.0]), (len(texts), 1))

    adapter = DSPyEmbeddingAdapter(
        replace(resolve_embedding_profile(), dimensions=2),
        create_dspy_embedder(changing_embed, caching=False),
    )
    with pytest.raises(RuntimeError, match="embeddings are unstable"):
        benchmark.verify_embeddings(adapter, ["question"], ["document"])


@pytest.mark.parametrize("include_qa", [False, True])
def test_full_native_benchmark_reports_real_ingestion_retrieval_and_restart_reuse(
    tmp_path, monkeypatch, indexing_provider, include_qa
):
    url, calls = indexing_provider
    monkeypatch.setattr(benchmark, "BASE_URL", url)
    # Keep any retry lifecycle lookup inside this disposable environment.
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr(benchmark, "PROJECT_ROOT", tmp_path)
    corpus, qa = tmp_path / "corpus.json", tmp_path / "qa.json"
    corpus.write_text(json.dumps([{"idx": "notes", "title": "Notes", "text": "Ada wrote Notes."}]), encoding="utf-8")
    qa.write_text("[]", encoding="utf-8")
    prepare_dataset_bundle("musique", corpus, qa, tmp_path / "outputs/datasets")
    profile = replace(resolve_embedding_profile(task_mode="distinct-titled"), dimensions=8)

    def run(report):
        benchmark.run_yaw_benchmark(
            dataset="musique",
            document_ids=["notes"],
            queries=["Who wrote Notes?"],
            gold_docs=[["Notes\nAda wrote Notes."]],
            gold_answers=[["Ada"]],
            train_samples=[],
            profile=profile,
            extraction_model="indexing-contract",
            qa_model="indexing-contract",
            include_qa=include_qa,
            enable_gbnf=False,
            evolve="none",
            compiled_program=None,
            index_dir=tmp_path / "index",
            report=report,
        )

    report = {}
    run(report)
    assert report["graph"]["num_passage_nodes"] == 1
    assert report["graph"]["num_fact_nodes"] == 1
    metrics = report["evaluation_metrics"]
    assert metrics["hipporag"]["retrieval"]["Recall@1"] == 1.0
    assert metrics["standard_dpr"]["retrieval"]["Recall@1"] == 1.0
    assert metrics["hipporag"]["qa"] == ({"ExactMatch": 1.0, "F1": 1.0} if include_qa else None)
    assert len(report["records"]) == 1
    assert report["records"][0]["retrieved_chunk_ids"]
    assert (tmp_path / "index" / "yaw.db").is_file()
    before = list(calls)
    resumed = {}
    run(resumed)
    assert calls == before
    assert resumed["index_batches"] == 1
    assert resumed["evaluation_metrics"] == metrics


@pytest.mark.parametrize("downloaded", [True, False])
def test_benchmark_main_persists_success_or_failure_with_disjoint_probe_queries(
    tmp_path, monkeypatch, registry_server, downloaded
):
    state, url = registry_server
    profile = replace(PROFILES[MODEL], dimensions=2)
    monkeypatch.setitem(PROFILES, MODEL, profile)
    entry = embedding_entry(profile)
    entry["downloaded"] = downloaded
    state["registry"] = [entry]
    state["loaded"] = state["after_probe"] = [entry]
    monkeypatch.setattr(benchmark, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(benchmark, "BASE_URL", url)
    # Bind real identity checks to this actual service, including their defaults.
    monkeypatch.setattr(benchmark, "preflight", partial(preflight, base_url=url))
    monkeypatch.setattr(benchmark, "embedding_evidence", partial(embedding_evidence, base_url=url))
    data = tmp_path / "outputs" / "datasets"
    data.mkdir(parents=True)
    paragraph = {"title": "Notes", "text": "Ada wrote Notes.", "is_supporting": True}
    (data / "sample.json").write_text(
        json.dumps(
            [
                {"question": "Who wrote Notes?", "answer": "Ada", "paragraphs": [paragraph]},
                {"question": "What did Ada write?", "answer": "Notes", "paragraphs": [paragraph]},
            ]
        ),
        encoding="utf-8",
    )
    (data / "sample_corpus.json").write_text(json.dumps([paragraph]), encoding="utf-8")

    prepare_dataset_bundle("sample", data / "sample_corpus.json", data / "sample.json", data)

    def run():
        return benchmark.main(
            dataset="sample",
            mode="embeddings",
            max_queries=1,
            index_queries=1,
            train_queries=1,
            evolve="gepa",
            chat_model=benchmark.DEFAULT_QA_CHAT,
        )

    if downloaded:
        result = run()
        assert result["status"] == "passed" and result["query_offset"] == 1
        assert result["train_queries"] == 1 and result["num_queries"] == 1
        assert len(result["embedding_probes"]) == 4
        assert result["embedding_probes"][0]["inputs"][0].endswith("What did Ada write?")
        report_path = next((tmp_path / "outputs" / "benchmarks").glob("run-sample-*/report.json"))
        assert json.loads(report_path.read_text(encoding="utf-8")) == result
    else:
        with pytest.raises(ValueError, match="not downloaded"):
            run()
        report_path = next((tmp_path / "outputs" / "benchmarks").glob("run-sample-*/report.json"))
        result = json.loads(report_path.read_text(encoding="utf-8"))
        assert result["status"] == "failed" and "not downloaded" in result["error"]
        assert result["elapsed_seconds"] >= 0
