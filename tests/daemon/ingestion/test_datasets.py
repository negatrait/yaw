"""Prepared files, native persistent snapshots and untouched evaluation records."""

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from yaw.daemon.graph.storage import hydrate_chunk_pointer
from yaw.daemon.ingestion import (
    DATASET_REGISTRY,
    DATASET_SOURCE_FIELDS,
    ENTERPRISE_CATEGORY_QUOTAS,
    CorpusIngestor,
    load_dataset_bundle,
    load_dataset_files,
    load_dataset_splits,
    prepare_dataset_bundle,
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
from yaw.daemon.ingestion import CorpusIngestor
calls = []
def profile(frame, event, arg):
    if event == "call" and frame.f_code.co_name == "chunk_source_text":
        calls.append(frame.f_locals["rel_posix_path"])
sys.setprofile(profile)
threading.setprofile(profile)
with CorpusIngestor(sys.argv[1], sys.argv[2], **json.loads(sys.argv[3])) as i:
    delta = i.poll()
    snapshot = i.snapshot()
sys.setprofile(None)
print(json.dumps([len(delta.added_chunks), len(delta.deleted_chunk_ids), len(snapshot), calls]))
"""

    def scan(**settings):
        result = subprocess.run(
            [sys.executable, "-c", worker, str(bundle.documents_root), str(tmp_path / "state"), json.dumps(settings)],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        return json.loads(result.stdout.strip().splitlines()[-1])

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
    assert {r["question_id"] for r in selected[:150]}.isdisjoint({r["question_id"] for r in selected[150:]})
    for category, (nt, nv) in ENTERPRISE_CATEGORY_QUOTAS.items():
        assert sum(r["question_type"] == category for r in selected[:150]) == nt
        assert sum(r["question_type"] == category for r in selected[150:]) == nv
    assert before == json.dumps(rows)
    corpus, qa = inputs(tmp_path, questions=rows)
    prepare_dataset_bundle("enterprise_rag_bench", corpus, qa, tmp_path / "outputs/datasets")
    train, val, documents = load_dataset_splits(
        ["enterprise"], workspace_root=tmp_path, max_queries=250, train_queries=150
    )
    assert len(train) == 150 and len(val) == 100 and len(documents) == 2
    assert train[0]["gold_answers"] == [selected[0]["gold_answer"]]
    assert not train[0]["item_docs"]
    small_train, small_val, _ = load_dataset_splits(
        ["enterprise"], workspace_root=tmp_path, max_queries=12, train_queries=7
    )
    assert [row["gold_answers"] for row in small_train] == [[r["gold_answer"]] for r in selected[:7]]
    assert [row["gold_answers"] for row in small_val] == [[r["gold_answer"]] for r in selected[150:155]]
    presplit = [dict(row, split="train" if i < 150 else "val") for i, row in enumerate(selected)]
    assert select_enterprise_questions(presplit) == presplit
    presplit[150]["question_id"] = presplit[0]["question_id"]
    with pytest.raises(ValueError, match="Duplicate Enterprise question"):
        select_enterprise_questions(presplit)


def test_snapshot_requires_successful_scan_and_malformed_file_fails_then_retries(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    source = docs / "doc.json"
    source.write_text('{"title": "A", "text": "Valid"}', encoding="utf-8")
    with CorpusIngestor(docs, tmp_path / "state") as ingestor:
        with pytest.raises(RuntimeError, match="successful poll"):
            ingestor.snapshot()
        assert ingestor.poll().added_chunks
        source.write_text("{malformed", encoding="utf-8")
        with pytest.raises(ValueError):
            ingestor.poll()
        with pytest.raises(RuntimeError):
            ingestor.snapshot()
        source.write_text('{"title": "A", "text": "Fixed"}', encoding="utf-8")
        assert len(ingestor.poll().added_chunks) == 1
        assert len(ingestor.snapshot()) == 1


def test_failed_scan_does_not_delete_unvisited_sources_from_real_sqlite(tmp_path):
    docs = tmp_path / "documents"
    docs.mkdir()
    a, b = docs / "a.json", docs / "b.json"
    a.write_text('{"title":"A","text":"First"}', encoding="utf-8")
    b.write_text('{"title":"B","text":"Second"}', encoding="utf-8")
    with sqlite3.connect(tmp_path / "target.db") as conn:
        conn.execute("CREATE TABLE chunks(id TEXT PRIMARY KEY, source TEXT)")

        # SQLite callbacks run on CocoIndex's worker thread, so open the real
        # connection in that thread rather than sharing a thread-bound handle.
        def apply(delta):
            with sqlite3.connect(tmp_path / "target.db") as target:
                target.executemany(
                    "INSERT OR REPLACE INTO chunks VALUES (?,?)",
                    [(p.chunk_id, p.rel_posix_path) for p, _ in delta.added_chunks],
                )
                target.executemany("DELETE FROM chunks WHERE id=?", [(cid,) for cid in delta.deleted_chunk_ids])

        with CorpusIngestor(docs, tmp_path / "state", on_delta=apply) as ingestor:
            ingestor.poll()
            before = conn.execute("SELECT * FROM chunks ORDER BY id").fetchall()
            a.write_text("{bad", encoding="utf-8")
            b.unlink()
            with pytest.raises(ValueError):
                ingestor.poll()
            assert conn.execute("SELECT * FROM chunks ORDER BY id").fetchall() == before
            a.write_text('{"title":"A","text":"Fixed"}', encoding="utf-8")
            ingestor.poll()
            assert conn.execute("SELECT source FROM chunks").fetchall() == [("a.json",)]
