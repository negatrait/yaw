"""Unmocked integration tests for centralized datasets registry, shared-corpus loading, and in-place CocoIndex sync."""

import json
from pathlib import Path

from yaw.daemon.ingestion.datasets import (
    BENCHMARK_JSON_BLACKLIST,
    DATASET_ALIASES,
    DATASET_REGISTRY,
    _ensure_multiline_json_file,
    _load_corpus_by_id,
    extract_gold_answers,
    extract_sample_docs_and_gold,
    get_gold_docs,
    load_dataset_splits,
    resolve_dataset_name,
    resolve_dataset_source_files,
    sync_source_files_for_docs,
)


def test_dataset_registry_and_blacklist_keys() -> None:
    assert DATASET_REGISTRY["enterprise_rag_bench"] == (
        "enterprise_rag_bench_questions.json",
        "enterprise_rag_bench_corpus.json",
    )
    assert DATASET_ALIASES["enterprise"] == "enterprise_rag_bench"
    assert DATASET_ALIASES["enterprise_rag_bench.json"] == "enterprise_rag_bench"
    assert DATASET_ALIASES["enterprise_rag_bench_questions"] == "enterprise_rag_bench"
    assert DATASET_ALIASES["enterprise_rag_bench_questions.json"] == "enterprise_rag_bench"

    for blacklisted_key in ("split", "source_type", "source_types", "expected_doc_ids"):
        assert blacklisted_key in BENCHMARK_JSON_BLACKLIST

    assert resolve_dataset_name("enterprise") == "enterprise_rag_bench"
    assert resolve_dataset_name("enterprise_rag_bench.json") == "enterprise_rag_bench"
    assert resolve_dataset_name("enterprise_rag_bench_questions") == "enterprise_rag_bench"
    assert resolve_dataset_name("enterprise_rag_bench_questions.json") == "enterprise_rag_bench"
    assert resolve_dataset_name("outputs/datasets/enterprise_rag_bench_questions.json") == "enterprise_rag_bench"


def test_extract_gold_answers_excludes_answer_facts() -> None:
    sample = {
        "question": "What is the secret?",
        "answer": "Primary Answer",
        "answer_aliases": ["Alias 1", "Alias 2"],
        "answer_facts": [
            "The answer must state that the query cannot be answered",
            "Negative rubric constraint",
        ],
    }
    extracted = extract_gold_answers(sample, strict=False)
    assert extracted == ["Primary Answer", "Alias 1", "Alias 2"]
    assert "The answer must state that the query cannot be answered" not in extracted
    assert "Negative rubric constraint" not in extracted


def test_load_corpus_by_id_and_extract_sample_docs_and_gold(tmp_path: Path) -> None:
    ds_dir = tmp_path / "outputs" / "datasets"
    ds_dir.mkdir(parents=True, exist_ok=True)
    corpus_file = ds_dir / "test_corpus.json"
    corpus_records = [
        {"idx": "doc_1", "title": "Doc 1", "text": "Content of doc 1"},
        {"idx": "doc_2", "title": "Doc 2", "text": "Content of doc 2"},
    ]
    corpus_file.write_text(json.dumps(corpus_records, indent=2), encoding="utf-8")

    lookup = _load_corpus_by_id("test", workspace_root=tmp_path)
    assert lookup["doc_1"] == "Doc 1\nContent of doc 1"
    assert lookup["doc_2"] == "Doc 2\nContent of doc 2"

    # Verify _load_corpus_by_id via direct file path (both questions and corpus path)
    q_file = ds_dir / "test_questions.json"
    q_file.write_text("[]", encoding="utf-8")
    lookup_via_q_path = _load_corpus_by_id(q_file, workspace_root=tmp_path)
    assert lookup_via_q_path["doc_1"] == "Doc 1\nContent of doc 1"

    # 1. Pure question with expected_doc_ids -> item_docs is empty list, gold_docs is resolved
    q_pure = {
        "id": "q1",
        "question": "What is doc 1?",
        "expected_doc_ids": ["doc_1"],
    }
    item_docs, gold_docs = extract_sample_docs_and_gold(q_pure, "test", corpus_by_id=lookup)
    assert item_docs == []
    assert gold_docs == ["Doc 1\nContent of doc 1"]

    # 2. Pure question with empty expected_doc_ids (e.g. Info Not Found)
    q_abstain = {
        "id": "q2",
        "question": "Unanswerable question",
        "expected_doc_ids": [],
    }
    item_docs, gold_docs = extract_sample_docs_and_gold(q_abstain, "test", corpus_by_id=lookup)
    assert item_docs == []
    assert gold_docs == []

    # 3. get_gold_docs with path as dataset_name
    golds = get_gold_docs([q_pure, q_abstain], dataset_name=q_file, workspace_root=tmp_path)
    assert golds == [["Doc 1\nContent of doc 1"], []]

    # 4. Backwards-compatible legacy item with inlined paragraphs
    q_legacy = {
        "id": "q3",
        "question": "Legacy question",
        "paragraphs": [
            {"title": "P1", "text": "Text 1", "is_supporting": True},
            {"title": "P2", "text": "Text 2", "is_supporting": False},
        ],
    }
    item_docs, gold_docs = extract_sample_docs_and_gold(q_legacy, "legacy")
    assert len(item_docs) == 2
    assert len(gold_docs) == 1
    assert gold_docs == ["P1\nText 1"]


def test_ensure_multiline_json_file_formats_minified(tmp_path: Path) -> None:
    single_line_file = tmp_path / "minified.json"
    single_line_file.write_text(json.dumps([{"a": 1}, {"b": 2}]), encoding="utf-8")
    assert len(single_line_file.read_text(encoding="utf-8").splitlines()) == 1

    formatted_path = _ensure_multiline_json_file(single_line_file)
    assert formatted_path == single_line_file.resolve()
    lines = formatted_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) > 5


def test_resolve_dataset_source_files_prefer_corpus(tmp_path: Path) -> None:
    ds_dir = tmp_path / "outputs" / "datasets"
    ds_dir.mkdir(parents=True, exist_ok=True)
    q_path = ds_dir / "enterprise_rag_bench_questions.json"
    c_path = ds_dir / "enterprise_rag_bench_corpus.json"
    q_path.write_text(json.dumps([{"q": 1}], indent=2), encoding="utf-8")
    c_path.write_text(json.dumps([{"c": 1}], indent=2), encoding="utf-8")

    # prefer_corpus=False returns questions file
    q_resolved = resolve_dataset_source_files(["enterprise"], workspace_root=tmp_path, prefer_corpus=False)
    assert len(q_resolved) == 1
    assert q_resolved[0].name == "enterprise_rag_bench_questions.json"

    # prefer_corpus=True returns corpus file
    c_resolved = resolve_dataset_source_files(["enterprise"], workspace_root=tmp_path, prefer_corpus=True)
    assert len(c_resolved) == 1
    assert c_resolved[0].name == "enterprise_rag_bench_corpus.json"

    # prefer_corpus=True when given direct questions file path still returns corpus file!
    c_resolved_from_q = resolve_dataset_source_files([q_path], workspace_root=tmp_path, prefer_corpus=True)
    assert len(c_resolved_from_q) == 1
    assert c_resolved_from_q[0].name == "enterprise_rag_bench_corpus.json"

    # prefer_corpus=True when given direct questions str path still returns corpus file!
    c_resolved_from_str = resolve_dataset_source_files(str(q_path), workspace_root=tmp_path, prefer_corpus=True)
    assert len(c_resolved_from_str) == 1
    assert c_resolved_from_str[0].name == "enterprise_rag_bench_corpus.json"


def test_load_dataset_splits_shared_corpus(tmp_path: Path) -> None:
    ds_dir = tmp_path / "outputs" / "datasets"
    ds_dir.mkdir(parents=True, exist_ok=True)
    q_path = ds_dir / "enterprise_rag_bench_questions.json"
    c_path = ds_dir / "enterprise_rag_bench_corpus.json"

    corpus_records = [
        {"idx": f"doc_{i}", "title": f"Title {i}", "text": f"Corpus body {i}", "source_type": "jira"}
        for i in range(10)
    ]
    c_path.write_text(json.dumps(corpus_records, indent=2), encoding="utf-8")

    question_records = [
        {
            "id": f"q_{i}",
            "question": f"Question {i}?",
            "answer": f"Answer {i}",
            "answer_facts": [f"Fact {i}"],
            "question_type": "Basic",
            "expected_doc_ids": [f"doc_{i % 10}"],
        }
        for i in range(15)
    ]
    q_path.write_text(json.dumps(question_records, indent=2), encoding="utf-8")

    trainset, valset, corpus_docs = load_dataset_splits(
        ["enterprise"],
        workspace_root=tmp_path,
        max_queries=15,
        train_queries=10,
        query_offset=0,
    )
    assert len(trainset) == 10
    assert valset is not None and len(valset) == 5

    # All 10 corpus documents are returned in corpus_docs
    assert len(corpus_docs) == 10
    assert corpus_docs[0] == "Title 0\nCorpus body 0"

    # item_docs is empty list on each example, and metadata is preserved
    for ex in trainset + valset:
        assert ex["item_docs"] == []
        assert "Fact" in ex["answer_facts"][0]
        assert ex["question_type"] == "Basic"


def test_sync_source_files_for_docs_inplace_no_intermediate_markdown(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    state_dir = tmp_path / "state"
    workspace.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    corpus_file = workspace / "test_corpus.json"
    corpus_records = [
        {"idx": "d1", "title": "Doc One", "text": "First body content here for indexing."},
        {"idx": "d2", "title": "Doc Two", "text": "Second body content here for indexing."},
    ]
    corpus_file.write_text(json.dumps(corpus_records, indent=2), encoding="utf-8")

    # Ingest directly from source_files
    chunks = sync_source_files_for_docs(
        workspace,
        state_dir,
        source_files=[corpus_file],
        chunk_size=500,
        chunk_overlap=50,
        min_chunk_size=10,
    )
    assert len(chunks) >= 2

    # Invariant: ZERO intermediate markdown files created
    assert not (workspace / "corpus").exists()
    assert list(workspace.glob("*.md")) == []
    assert list(workspace.rglob("*.md")) == []

    # Test allowed_docs filtering
    filtered_chunks = sync_source_files_for_docs(
        workspace,
        state_dir,
        source_files=[corpus_file],
        allowed_docs=["Doc One\nFirst body content here for indexing."],
        chunk_size=500,
        chunk_overlap=50,
        min_chunk_size=10,
    )
    assert len(filtered_chunks) >= 1
    for _ptr, txt in filtered_chunks:
        assert "Doc One" in txt or "First body" in txt


def test_sync_source_files_for_docs_multichunk_title_indexed_filtering(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    state_dir = tmp_path / "state"
    workspace.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    long_body_1 = " ".join(f"Alpha paragraph sentence {i} with distinct content." for i in range(40))
    long_body_2 = " ".join(f"Beta paragraph sentence {i} with distinct content." for i in range(40))
    long_body_3 = " ".join(f"Gamma paragraph sentence {i} with distinct content." for i in range(40))
    corpus_file = workspace / "multichunk_corpus.json"
    corpus_records = [
        {"idx": "d1", "title": "Multi Doc Alpha", "text": long_body_1},
        {"idx": "d2", "title": "Multi Doc Beta", "text": long_body_2},
        {"idx": "d3", "title": "Multi Doc Alpha", "text": long_body_3},
    ]
    corpus_file.write_text(json.dumps(corpus_records, indent=2), encoding="utf-8")

    # Filter for only d1 (and not d2, nor d3 which shares title "Multi Doc Alpha" with d1)
    filtered = sync_source_files_for_docs(
        workspace,
        state_dir,
        source_files=[corpus_file],
        allowed_docs=[f"Multi Doc Alpha\n{long_body_1}"],
        chunk_size=300,
        chunk_overlap=30,
        min_chunk_size=20,
    )
    assert len(filtered) >= 3
    for ptr, txt in filtered:
        assert ptr.title == "Multi Doc Alpha"
        assert "Alpha paragraph sentence" in txt
        assert "Beta paragraph sentence" not in txt
        assert "Gamma paragraph sentence" not in txt

