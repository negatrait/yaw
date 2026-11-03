"""Prepared files, native persistent snapshots and untouched evaluation records."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from yaw.core import YawRuntimeError
from yaw.daemon.graph.storage import hydrate_chunk_pointer
from yaw.daemon.ingestion import (
    DATASET_REGISTRY,
    DATASET_SOURCE_FIELDS,
    ENTERPRISE_CATEGORY_QUOTAS,
    load_dataset_bundle,
    load_dataset_files,
    load_dataset_splits,
    prepare_dataset_bundle,
    resolve_dataset_name,
    select_enterprise_questions,
    snapshot_dataset_bundle,
)


def inputs(tmp_path: Path, records=None, questions=None):
    corpus = tmp_path / "source.json"
    qa = tmp_path / "qa.json"
    corpus.write_text(
        json.dumps(
            records
            if records is not None
            else [
                {"idx": "a", "title": "  Original ä Title  ", "text": "First\nsecond\r\nline", "is_supporting": True},
                {"idx": "b", "title": "Distractor", "text": "Unrelated content."},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    qa.write_text(
        json.dumps(
            questions
            if questions is not None
            else [
                {"question": "SECRET QUESTION", "answer": "SECRET ANSWER", "expected_doc_ids": ["a"]},
                {"question": "Other question", "answer": "Other answer", "expected_doc_ids": ["b"]},
            ]
        ),
        encoding="utf-8",
    )
    return corpus, qa


@pytest.mark.parametrize("dataset", DATASET_REGISTRY)
def test_preparation_preserves_document_values_qa_bytes_and_full_inventory(tmp_path, dataset):
    corpus, qa = inputs(tmp_path)
    original = corpus.read_bytes(), qa.read_bytes()
    fields = DATASET_SOURCE_FIELDS[dataset]
    bundle = prepare_dataset_bundle(dataset, corpus, qa, tmp_path / "outputs/datasets", **fields)
    assert len(bundle.documents) == 2
    assert bundle.evaluation_path.read_bytes() == qa.read_bytes()
    for entry, row in zip(bundle.documents, json.loads(corpus.read_text(encoding="utf-8")), strict=True):
        exported = json.loads((bundle.root / entry.path).read_text(encoding="utf-8"))
        expected = {"title": row["title"], "text": row["text"]}
        if fields["id_key"]:
            expected[fields["id_key"]] = row[fields["id_key"]]
            assert entry.document_id == str(row[fields["id_key"]])
        assert exported == expected
        assert "SECRET" not in (bundle.root / entry.path).read_text(encoding="utf-8")
    assert original == (corpus.read_bytes(), qa.read_bytes())
    mtimes = [(bundle.root / d.path).stat().st_mtime_ns for d in bundle.documents]
    second = prepare_dataset_bundle(dataset, corpus, qa, tmp_path / "outputs/datasets", **fields)
    assert mtimes == [(second.root / d.path).stat().st_mtime_ns for d in second.documents]
    assert load_dataset_bundle(bundle.root) == bundle


def test_readers_reject_a_stale_bundle_after_preparation_publishes_new_content(tmp_path):
    corpus, qa = inputs(tmp_path)
    bundle = prepare_dataset_bundle("sample", corpus, qa, tmp_path / "prepared")
    rows = json.loads(corpus.read_bytes())
    rows[0]["text"] = "New generation"
    corpus.write_text(json.dumps(rows), encoding="utf-8")
    current = prepare_dataset_bundle("sample", corpus, qa, tmp_path / "prepared")
    with pytest.raises(ValueError, match="bundle changed"):
        bundle.read_inputs()
    questions, documents = current.read_inputs()
    assert documents[0]["text"] == "New generation"
    assert questions == json.loads(qa.read_bytes())


def test_manifest_detects_incomplete_exports_and_rejects_unsafe_paths(tmp_path):
    corpus, qa = inputs(tmp_path)
    bundle = prepare_dataset_bundle("sample", corpus, qa, tmp_path / "prepared")
    path = bundle.root / bundle.documents[0].path
    original = path.read_bytes()
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Incomplete"):
        load_dataset_bundle(bundle.root)
    path.write_bytes(original)
    data = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    data["documents"][0]["path"] = "evaluation/questions.json"
    bundle.manifest_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid document path"):
        load_dataset_bundle(bundle.root)


def test_conflicting_duplicate_ids_fail_before_publishing(tmp_path):
    corpus, qa = inputs(
        tmp_path,
        records=[
            {"idx": "a", "title": "A", "text": "One"},
            {"idx": "a", "title": "A", "text": "Two"},
        ],
    )
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        prepare_dataset_bundle("sample", corpus, qa, tmp_path / "prepared")
    assert not (tmp_path / "prepared/sample/manifest.json").exists()


def test_snapshot_reuses_native_memos_across_processes_and_updates_one_document(tmp_path):
    corpus, qa = inputs(tmp_path)
    bundle = prepare_dataset_bundle("sample", corpus, qa, tmp_path / "prepared")
    worker = """
import json, sys, threading
from yaw.daemon.ingestion import load_dataset_bundle, snapshot_dataset_bundle
calls = []
def profile(frame, event, arg):
    if event == "call" and frame.f_code.co_name == "chunk_source_text":
        calls.append(frame.f_locals["rel_posix_path"])
sys.setprofile(profile)
threading.setprofile(profile)
snapshot = snapshot_dataset_bundle(load_dataset_bundle(sys.argv[1]), sys.argv[2], **json.loads(sys.argv[3]))
sys.setprofile(None)
print(json.dumps([[p.chunk_id for p, _ in snapshot], calls]))
"""
    prev_ids: set[str] = set()

    def scan(**settings):
        nonlocal prev_ids
        result = subprocess.run(
            [sys.executable, "-c", worker, str(bundle.root), str(tmp_path / "state"), json.dumps(settings)],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        chunk_ids, calls = json.loads(result.stdout.strip().splitlines()[-1])
        cur_ids = set(chunk_ids)
        added = len(cur_ids - prev_ids)
        deleted = len(prev_ids - cur_ids)
        prev_ids = cur_ids
        return [added, deleted, len(chunk_ids), calls]

    first = scan()
    assert first[:3] == [2, 0, 2] and len(first[3]) == 2
    assert scan() == [0, 0, 2, []]
    rows = json.loads(corpus.read_text(encoding="utf-8"))
    rows[0]["text"] = "Changed document."
    corpus.write_text(json.dumps(rows), encoding="utf-8")
    prepare_dataset_bundle("sample", corpus, qa, tmp_path / "prepared")
    changed = scan()
    assert changed[:3] == [1, 1, 2] and len(changed[3]) == 1
    rows.pop()
    corpus.write_text(json.dumps(rows), encoding="utf-8")
    prepare_dataset_bundle("sample", corpus, qa, tmp_path / "prepared")
    assert scan() == [0, 1, 1, []]
    rows.append({"idx": "c", "title": "Inserted", "text": "Another original document"})
    corpus.write_text(json.dumps(rows), encoding="utf-8")
    prepare_dataset_bundle("sample", corpus, qa, tmp_path / "prepared")
    inserted = scan()
    assert inserted[:3] == [1, 0, 2] and len(inserted[3]) == 1
    resized = scan(chunk_size=200)
    assert resized[2] == 2 and len(resized[3]) == 2
    assert scan(chunk_size=200) == [0, 0, 2, []]
    projected = scan(chunk_size=200, json_keys_whitelist=["text"])
    assert projected[2] == 2 and len(projected[3]) == 2
    assert scan(chunk_size=200, json_keys_whitelist=["text"]) == [0, 0, 2, []]


def test_complete_bundle_snapshot_isolated_selection_and_real_hydration(tmp_path):
    corpus, qa = inputs(tmp_path)
    bundle = prepare_dataset_bundle("sample", corpus, qa, tmp_path / "outputs/datasets")
    first = snapshot_dataset_bundle(bundle, tmp_path / "state", min_chunk_size=1)
    warm = snapshot_dataset_bundle(bundle, tmp_path / "state", min_chunk_size=1)
    assert first == warm and len(first) == 2
    for ptr, _ in warm:
        text = hydrate_chunk_pointer(ptr, bundle.root.parent).text
        assert "SECRET" not in text
        assert "/documents/" in ptr.rel_posix_path
    selected = snapshot_dataset_bundle(bundle, tmp_path / "state", document_ids=["b"], min_chunk_size=1)
    assert len(selected) == 1 and selected[0][0].title == "Distractor"
    with pytest.raises(ValueError, match="Document IDs missing"):
        snapshot_dataset_bundle(bundle, tmp_path / "state", document_ids=["unknown"])
    samples, documents = load_dataset_files("sample", project_root=tmp_path)
    assert samples == json.loads(qa.read_text(encoding="utf-8"))
    assert len(documents) == 2


def test_enterprise_stratification_is_owned_by_loader_and_does_not_rewrite_rows(tmp_path):
    rows = []
    for category, (nt, nv) in ENTERPRISE_CATEGORY_QUOTAS.items():
        rows.extend(
            {
                "question_id": f"{category}-{i}",
                "question_type": category,
                "question": f"Question {i}",
                "gold_answer": f" Answer {i} ",
                "expected_doc_ids": ["a"],
            }
            for i in range(nt + nv + 3)
        )
    before = json.dumps(rows)
    selected = select_enterprise_questions(rows)
    assert len(selected) == 250 and selected == select_enterprise_questions(rows)
    assert {r["question_id"] for r in selected[:223]}.isdisjoint({r["question_id"] for r in selected[223:]})
    for category, (nt, nv) in ENTERPRISE_CATEGORY_QUOTAS.items():
        assert sum(r["question_type"] == category for r in selected[:223]) == nt
        assert sum(r["question_type"] == category for r in selected[223:]) == nv
    assert before == json.dumps(rows)
    corpus, qa = inputs(tmp_path, questions=rows)
    prepare_dataset_bundle("enterprise_rag_bench", corpus, qa, tmp_path / "outputs/datasets")
    train, val, documents = load_dataset_splits(
        ["enterprise_rag_bench"], workspace_root=tmp_path, max_queries=250, train_queries=223
    )
    assert len(train) == 223 and len(val) == 27 and len(documents) == 2
    assert train[0]["gold_answers"] == [selected[0]["gold_answer"]]
    assert not train[0]["item_docs"]
    small_train, small_val, _ = load_dataset_splits(
        ["enterprise_rag_bench"], workspace_root=tmp_path, max_queries=12, train_queries=7
    )
    assert [row["gold_answers"] for row in small_train] == [[r["gold_answer"]] for r in selected[:7]]
    assert [row["gold_answers"] for row in small_val] == [[r["gold_answer"]] for r in selected[223:228]]
    for alias in ("2wiki", "naturalquestions", "nq_rear", "enterprise"):
        with pytest.raises(ValueError, match="Unknown dataset"):
            resolve_dataset_name(alias)
    with pytest.raises(YawRuntimeError, match="held-out"):
        load_dataset_splits(["nq"], workspace_root=tmp_path)
    presplit = [dict(row, split="train" if i < 223 else "val") for i, row in enumerate(selected)]
    assert select_enterprise_questions(presplit) == presplit
    presplit[223]["question_id"] = presplit[0]["question_id"]
    with pytest.raises(ValueError, match="Duplicate Enterprise question"):
        select_enterprise_questions(presplit)


def test_snapshot_dataset_bundle_fails_on_malformed_document_then_recovers(tmp_path):
    corpus, qa = inputs(tmp_path)
    bundle = prepare_dataset_bundle("sample", corpus, qa, tmp_path / "bundles")
    first_doc = bundle.root / bundle.documents[0].path
    original_bytes = first_doc.read_bytes()
    state_dir = tmp_path / "state"
    assert len(snapshot_dataset_bundle(bundle, state_dir=state_dir, min_chunk_size=1)) == 2
    first_doc.write_text("{malformed", encoding="utf-8")
    with pytest.raises(ValueError, match="Incomplete dataset export"):
        snapshot_dataset_bundle(bundle, state_dir=state_dir, min_chunk_size=1)
    first_doc.write_bytes(original_bytes)
    assert len(snapshot_dataset_bundle(bundle, state_dir=state_dir, min_chunk_size=1)) == 2
