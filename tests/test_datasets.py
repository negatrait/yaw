"""Tests for dataset bundle management, slicing, extraction, and pure evaluation metrics."""

import ast
import hashlib
import json
from pathlib import Path

import pytest

from yaw.datasets import (
    DATASET_REGISTRY,
    allow_upstream_github_fallback,
    build_benchmark_slice,
    compute_qa_em_f1,
    doc_matches_gold,
    document_filename,
    document_identity,
    extract_gold_answers,
    get_gold_docs,
    load_dataset_bundle,
    load_dataset_files,
    normalize_answer,
    prepare_dataset_bundle,
    render_document_markdown,
    resolve_dataset_bundle,
    resolve_dataset_name,
    resolve_slice_corpus_scope,
    score_qa_em_f1,
    score_retrieval_recalls,
    validate_dataset_bundle,
)


def test_bundle_lifecycle_and_validation(tmp_path: Path) -> None:
    corpus = [
        {"title": "Doc 1", "text": "Content of document 1.", "idx": "d1"},
        {"title": "Doc 2", "text": "Content of document 2.", "idx": "d2"},
    ]
    questions = [
        {"question": "Q1?", "answer": "Ans 1", "gold_docs": ["Doc 1\nContent of document 1."]},
    ]
    corpus_p = tmp_path / "corpus.json"
    qa_p = tmp_path / "questions.json"
    corpus_p.write_text(json.dumps(corpus), encoding="utf-8")
    qa_p.write_text(json.dumps(questions), encoding="utf-8")

    bundle = prepare_dataset_bundle("sample", corpus_p, qa_p, tmp_path)
    assert bundle.dataset == "sample"
    assert len(bundle.documents) == 2
    assert bundle.documents[0].path == "documents/d1.md"
    assert bundle.documents[1].path == "documents/d2.md"
    assert (bundle.root / "documents" / "d1.md").read_text(encoding="utf-8") == "# Doc 1\n\nContent of document 1.\n"
    assert json.loads(bundle.manifest_path.read_text(encoding="utf-8"))["version"] == 2
    assert bundle.documents_root.exists()
    assert bundle.evaluation_path.exists()
    assert bundle.manifest_path.exists()

    loaded = load_dataset_bundle(bundle.root)
    assert loaded.dataset == "sample"
    assert len(loaded.documents) == 2
    validate_dataset_bundle(loaded)

    # Test read methods round-trip H1 strip and raw markdown lookup
    docs = bundle.read_documents()
    assert docs == [
        {"title": "Doc 1", "text": "Content of document 1.", "idx": "d1"},
        {"title": "Doc 2", "text": "Content of document 2.", "idx": "d2"},
    ]
    inputs_qa, inputs_docs = bundle.read_inputs()
    assert len(inputs_qa) == 1
    assert inputs_docs == docs
    by_id = bundle.corpus_by_id()
    assert by_id["d1"] == "# Doc 1\n\nContent of document 1."

    # Upgrading from a v1 .json manifest removes stale .json files automatically
    stale_json = bundle.documents_root / "legacy_01.json"
    stale_json.write_text('{"title": "Old", "text": "Body"}\n', encoding="utf-8")
    v1_manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    v1_manifest["documents"] = [{"document_id": "old", "path": "documents/legacy_01.json", "content_hash": "0" * 64}]
    bundle.manifest_path.write_text(json.dumps(v1_manifest), encoding="utf-8")
    upgraded = prepare_dataset_bundle("sample", corpus_p, qa_p, tmp_path)
    assert not stale_json.exists()
    validate_dataset_bundle(upgraded)

    # Test resolve_dataset_bundle
    resolved = resolve_dataset_bundle(bundle.root)
    assert resolved.dataset == "sample"

    # Corruption: tampered doc content
    doc_file = bundle.root / bundle.documents[0].path
    orig_bytes = doc_file.read_bytes()
    doc_file.write_text("# Tampered\n\nBad body.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Corrupt|invalid|hash"):
        validate_dataset_bundle(bundle)
    doc_file.write_bytes(orig_bytes)
    validate_dataset_bundle(bundle)

    # Corruption: unmanaged extra file in documents/
    extra_file = bundle.documents_root / "unmanaged.md"
    extra_file.write_text("# Rogue\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory"):
        validate_dataset_bundle(bundle)
    extra_file.unlink()
    validate_dataset_bundle(bundle)

    # Corruption: tampered evaluation file
    eval_orig = bundle.evaluation_path.read_bytes()
    bundle.evaluation_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest|evaluation"):
        validate_dataset_bundle(bundle)
    bundle.evaluation_path.write_bytes(eval_orig)
    validate_dataset_bundle(bundle)

    # Reject conflicting duplicate document IDs and case-insensitive filename collisions
    dup_corpus = tmp_path / "dup_corpus.json"
    dup_corpus.write_text(
        json.dumps([{"idx": "dup1", "title": "A", "text": "One"}, {"idx": "dup1", "title": "A", "text": "Two"}]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        prepare_dataset_bundle("dup", dup_corpus, qa_p, tmp_path / "dup_out")

    case_corpus = tmp_path / "case_corpus.json"
    case_corpus.write_text(
        json.dumps([{"idx": "DocA", "title": "A", "text": "One"}, {"idx": "doca", "title": "B", "text": "Two"}]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="collision"):
        prepare_dataset_bundle("case_dup", case_corpus, qa_p, tmp_path / "case_out")

    # Missing manifest
    with pytest.raises(FileNotFoundError):
        load_dataset_bundle(tmp_path / "nonexistent")


def test_markdown_helpers_scope_and_fallback_metadata(tmp_path: Path) -> None:
    assert allow_upstream_github_fallback("nq") is False
    assert allow_upstream_github_fallback("hotpotqa") is True

    assert document_filename("doc_1") == "doc_1.md"
    assert document_filename("CON") != "CON.md" and document_filename("CON").endswith(".md")
    assert "/" not in document_filename("nested/id:1")
    assert document_filename("..") != "...md"
    assert len(document_filename("x" * 160)) < 120

    assert render_document_markdown("Doc", "Line1\r\nLine2") == "# Doc\n\nLine1\nLine2\n"
    assert render_document_markdown("Plan", "Step 1\\n\\nStep 2\\r\\nStep 3") == "# Plan\n\nStep 1\n\nStep 2\nStep 3\n"
    assert render_document_markdown("Code", "line1\nprint('a\\nb\\nc')") == "# Code\n\nline1\nprint('a\\nb\\nc')\n"
    assert render_document_markdown("Intro", "# Intro\n\nAlready H1") == "# Intro\n\nAlready H1\n"

    # source_fields id_key=None uses SHA-256[:16] of rendered Markdown identically in prepare + slice
    c_path = tmp_path / "c2.json"
    q_path = tmp_path / "q2.json"
    c_rows = [
        {"title": "Alpha", "text": "Alpha body."},
        {"title": "Beta", "text": "Beta body."},
    ]
    q_rows = [
        {
            "question": "What?",
            "answer": "Alpha",
            "supporting_facts": [["Alpha", 0]],
            "context": [["Alpha", ["Alpha body."]], ["Beta", ["Beta body."]]],
        }
    ]
    c_path.write_text(json.dumps(c_rows), encoding="utf-8")
    q_path.write_text(json.dumps(q_rows), encoding="utf-8")
    b1 = prepare_dataset_bundle("2wikimultihopqa", c_path, q_path, tmp_path / "bundles", id_key=None)
    expected_alpha_id = document_identity(c_rows[0], id_key=None)
    expected_beta_id = document_identity(c_rows[1], id_key=None)
    assert len(expected_alpha_id) == 16
    assert b1.documents[0].document_id == expected_alpha_id
    slice_ids, _, _, _ = build_benchmark_slice("2wikimultihopqa", q_rows, c_rows, max_queries=1)
    assert slice_ids == [expected_alpha_id, expected_beta_id]

    assert resolve_slice_corpus_scope([(b1, [b1.documents[0].document_id, b1.documents[1].document_id])]) == (
        b1.documents_root.resolve(),
        ("*.md",),
    )
    assert resolve_slice_corpus_scope([(b1, [expected_alpha_id])]) == (
        b1.documents_root.resolve(),
        (f"{expected_alpha_id}.md",),
    )

    c_other = tmp_path / "c_other.json"
    c_other.write_text(json.dumps([{"idx": "other_1", "title": "Gamma", "text": "Gamma body."}]), encoding="utf-8")
    b2 = prepare_dataset_bundle("sample", c_other, q_path, tmp_path / "bundles")
    common_root, multi_patterns = resolve_slice_corpus_scope([(b1, [expected_alpha_id]), (b2, ["other_1"])])
    assert common_root == (tmp_path / "bundles").resolve()
    assert multi_patterns == (
        f"2wikimultihopqa/documents/{expected_alpha_id}.md",
        "sample/documents/other_1.md",
    )

    with pytest.raises(ValueError):
        resolve_slice_corpus_scope([])
    with pytest.raises(ValueError):
        resolve_slice_corpus_scope([(b1, ["missing_id"])])


def test_dataset_name_resolution() -> None:
    assert resolve_dataset_name("sample") == "sample"
    assert resolve_dataset_name("hotpotqa") == "hotpotqa"
    with pytest.raises(ValueError, match="Unknown dataset"):
        resolve_dataset_name("unknown_invalid_dataset")


def test_schemas_and_slicing() -> None:
    # HotpotQA schema: supporting_facts + context
    hotpot_samples = [
        {
            "question": "What happened?",
            "answer": "Something.",
            "supporting_facts": [["Doc A", 0]],
            "context": [["Doc A", ["Sent 1.", " Sent 2."]], ["Doc B", ["Distractor."]]],
        }
    ]
    hp_docs = get_gold_docs(hotpot_samples, "hotpotqa")
    assert hp_docs == [["Doc A\nSent 1. Sent 2."]]

    # MusiQue schema: paragraphs with is_supporting
    musique_samples = [
        {
            "question": "Who was born?",
            "answer": "Alice",
            "paragraphs": [
                {"title": "P1", "text": "Alice was born in Paris.", "is_supporting": True},
                {"title": "P2", "text": "Unrelated fact.", "is_supporting": False},
            ],
        }
    ]
    mq_docs = get_gold_docs(musique_samples, "musique")
    assert mq_docs == [["P1\nAlice was born in Paris."]]

    # 2WikiMultiHopQA schema: contexts with is_supporting
    twiki_samples = [
        {
            "question": "Where is it?",
            "answer": "Berlin",
            "contexts": [
                {"title": "City", "text": "Berlin is a city.", "is_supporting": True},
                {"title": "Noise", "text": "Random text.", "is_supporting": False},
            ],
        }
    ]
    tw_docs = get_gold_docs(twiki_samples, "2wikimultihopqa")
    assert tw_docs == [["City\nBerlin is a city."]]

    # PopQA schema: obj + o_aliases + possible_answers
    popqa_samples = [
        {
            "question": "Who directed the movie?",
            "obj": "Christopher Nolan",
            "o_wiki_title": "Christopher Nolan (Director)",
            "o_aliases": '["C. Nolan", "Nolan"]',
            "possible_answers": ["Christopher Edward Nolan"],
        }
    ]
    pop_ans = extract_gold_answers(popqa_samples[0])
    assert "Christopher Nolan" in pop_ans
    assert "Nolan" in pop_ans
    assert "Christopher Edward Nolan" in pop_ans

    # LVEval is QA-only: get_gold_docs returns None
    assert get_gold_docs(popqa_samples, "lveval") is None

    # Benchmark slice building
    corpus = [
        {"idx": "doc_a", "title": "Doc A", "text": "Sent 1. Sent 2."},
        {"idx": "doc_b", "title": "Doc B", "text": "Distractor."},
        {"idx": "doc_c", "title": "Doc C", "text": "Extra doc."},
    ]
    doc_ids, questions, gold_docs, gold_answers = build_benchmark_slice(
        "hotpotqa", hotpot_samples, corpus, max_queries=1
    )
    assert len(questions) == 1
    assert len(gold_answers) == 1
    assert gold_docs == [["Doc A\nSent 1. Sent 2."]]
    assert "doc_a" in doc_ids

    # Invalid slice bounds
    with pytest.raises(ValueError):
        build_benchmark_slice("hotpotqa", hotpot_samples, corpus, max_queries=0)
    with pytest.raises(ValueError):
        build_benchmark_slice("hotpotqa", hotpot_samples, corpus, query_offset=-1)

    # Full corpus slice
    fc_ids, _, _, _ = build_benchmark_slice("hotpotqa", hotpot_samples, corpus, full_corpus=True)
    assert len(fc_ids) == len(corpus)


def test_load_dataset_files(tmp_path: Path) -> None:
    samples = [{"question": "Q1?", "answer": "A1"}]
    corpus = [{"title": "T1", "text": "Body 1."}]
    q_file, c_file = DATASET_REGISTRY["sample"]
    (tmp_path / q_file).write_text(json.dumps(samples), encoding="utf-8")
    (tmp_path / c_file).write_text(json.dumps(corpus), encoding="utf-8")

    loaded_s, loaded_c = load_dataset_files("sample", dataset_dir=tmp_path)
    assert len(loaded_s) == 1
    assert len(loaded_c) == 1

    loaded_s_only, empty_c = load_dataset_files("sample", dataset_dir=tmp_path, include_corpus=False)
    assert len(loaded_s_only) == 1
    assert empty_c == []


def test_metrics_and_doc_matching() -> None:
    # normalize_answer
    assert normalize_answer("The quick brown Fox, jumps!") == "quick brown fox jumps"
    assert normalize_answer("an Apple") == "apple"
    assert normalize_answer("a cat") == "cat"

    # compute_qa_em_f1
    em, f1 = compute_qa_em_f1("Barack Obama", ["Barack Hussein Obama", "Obama"])
    assert em == 0.0
    assert f1 > 0.0
    em2, f12 = compute_qa_em_f1("Barack Obama", ["the barack obama"])
    assert em2 == 1.0
    assert f12 == 1.0
    assert compute_qa_em_f1("answer", []) == (0.0, 0.0)

    # score_qa_em_f1
    means, exs = score_qa_em_f1([["Paris"], ["42"]], ["Paris", "43"])
    assert means["ExactMatch"] == 0.5
    assert len(exs) == 2

    # doc_matches_gold
    assert doc_matches_gold("Exact match text", "Exact match text")
    # Title prefix strip (including Markdown H1 "# ")
    assert doc_matches_gold("Wikipedia Title: Paris\nCapital of France.", "Title: Paris\nCapital of France.")
    assert doc_matches_gold("# Paris\n\nCapital of France.", "Paris\nCapital of France.")
    assert doc_matches_gold("Paris\nCapital of France.", "# Paris\n\nCapital of France.")
    # AST section cropped
    assert doc_matches_gold("Paris\nCapital of France.\n[... AST section cropped", "Paris\nCapital of France.")
    # chunk- md5 match
    h = hashlib.md5(b"Some passage text content here.").hexdigest()
    assert doc_matches_gold(f"chunk-{h}", "Some passage text content here.")
    assert doc_matches_gold("Some passage text content here.", f"chunk-{h}")
    # Bidirectional >= 20 char containment
    long_passage = "A sufficiently long passage with twenty or more characters of text."
    assert doc_matches_gold(long_passage[:30], long_passage)
    assert doc_matches_gold(long_passage, long_passage[:30])
    # < 20 chars should not match on partial containment
    assert not doc_matches_gold("short", "short extra content")

    # Proportional Recall@k on 2 gold docs + word-boundary containment check
    assert not doc_matches_gold(
        "one two three four five six seven",
        "onetwothreefourfivesixseven",
    )
    gold = [["doc1", "doc2"], ["doc3"]]
    retrieved = [["doc1", "doc2"], ["doc4", "doc3"]]
    rec_means, rec_exs = score_retrieval_recalls(gold, retrieved, k_list=(1, 2))
    assert rec_exs[0]["Recall@1"] == pytest.approx(0.5)
    assert rec_exs[0]["Recall@2"] == pytest.approx(1.0)
    assert rec_means == {"Recall@1": 0.25, "Recall@2": 1.0}
    with pytest.raises(ValueError):
        score_retrieval_recalls([["d1"]], [["d1"], ["d2"]])


def test_regression_fixes() -> None:
    # 1. extract_gold_answers prioritizes answer over obj when both exist
    sample_both = {
        "question": "Who?",
        "answer": "Primary Answer",
        "obj": "Secondary Obj",
        "o_aliases": ["Alias 1"],
    }
    extracted = extract_gold_answers(sample_both)
    assert extracted == ["Primary Answer"]

    assert compute_qa_em_f1("", ["Paris"]) == (0.0, 0.0)
    assert compute_qa_em_f1("   ", ["Paris"]) == (0.0, 0.0)

    # 2. doc_matches_gold handles body-after-first-newline
    shared_body = "This is a sufficiently long shared body paragraph of text."
    doc_ret = f"Different Title\n{shared_body}"
    doc_gold = f"Original Title\n{shared_body}"
    assert doc_matches_gold(doc_ret, doc_gold)

    # 3. build_benchmark_slice selects matching corpus doc IDs via passage content + gold_docs + min_docs_per_query
    samples = [
        {
            "question": "Where is it?",
            "answer": "Somewhere",
            "supporting_facts": [["Passage Alpha", 0]],
            "context": [
                ["Passage Alpha", ["First sentence of alpha.", "Second sentence."]],
                ["Passage Beta", ["Distractor beta sentence."]],
            ],
        }
    ]
    corpus = [
        {"idx": "doc_id_101", "title": "Passage Alpha", "text": "First sentence of alpha.Second sentence."},
        {"idx": "doc_id_102", "title": "Passage Beta", "text": "Distractor beta sentence."},
        {"idx": "doc_id_103", "title": "Passage Gamma", "text": "Completely unrelated content."},
    ]
    doc_ids, questions, gold_docs, gold_answers = build_benchmark_slice("hotpotqa", samples, corpus, max_queries=1)
    assert "doc_id_101" in doc_ids
    assert len(questions) == 1
    assert gold_docs == [["Passage Alpha\nFirst sentence of alpha.Second sentence."]]
    assert gold_answers == [["Somewhere"]]

    # 4. expected_doc_ids + untitled corpus documents resolve accurately (nq)
    nq_corpus = [
        {"idx": "doc_1", "title": "", "text": "Content of doc 1."},
        {"idx": "doc_2", "title": "", "text": "Content of doc 2."},
    ]
    nq_samples = [{"question": "Where?", "answer": "Here", "expected_doc_ids": ["doc_1"]}]
    nq_ids, nq_q, nq_gd, nq_ga = build_benchmark_slice("nq", nq_samples, nq_corpus, max_queries=1)
    assert nq_ids == ["doc_1"]
    assert nq_q == ["Where?"]
    assert nq_gd == [["Content of doc 1."]]
    assert nq_ga == [["Here"]]


def test_extract_gold_answers_json_and_malformed_fail_fast() -> None:
    plain_sample = {
        "question": "Who?",
        "obj": "Alice",
        "o_aliases": "Plain Alias String",
        "possible_answers": '["Alias Two", "Alias Three"]',
    }
    answers = extract_gold_answers(plain_sample)
    assert answers == ["Alice", "Alias Two", "Alias Three", "Plain Alias String"]

    for field, bad_val in (("possible_answers", "[broken"), ("o_aliases", "{broken")):
        with pytest.raises(ValueError):
            extract_gold_answers({"question": "Who?", "obj": "Alice", field: bad_val})


def test_imports_and_no_oop_metric_wrappers() -> None:
    src_path = Path(__file__).resolve().parent.parent / "yaw" / "datasets.py"
    text = src_path.read_text(encoding="utf-8")

    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("dspy"), f"Forbidden import: {alias.name}"
                assert not alias.name.startswith("yaw.daemon"), f"Forbidden import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert not mod.startswith("dspy"), f"Forbidden import from: {mod}"
            assert not mod.startswith("yaw.daemon"), f"Forbidden import from: {mod}"
        elif isinstance(node, ast.ClassDef):
            assert node.name not in (
                "RetrievalRecall",
                "QAExactMatch",
                "QAF1Score",
            ), f"Forbidden OOP wrapper class: {node.name}"
