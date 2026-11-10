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
    validate_dataset_bundle,
)
from yaw.daemon.ingestion.corpus import chunk_source_text


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
import asyncio, json, sys, threading
from pathlib import Path
import cocoindex as coco
from cocoindex.connectors import localfs
from yaw.daemon.ingestion import load_dataset_bundle
from yaw.daemon.ingestion.source_targets import (
    SOURCE_ROOT,
    SourceSettings,
    acquire_coco_environment,
    build_source_path_matcher,
    chunk_file_memo,
    close_coco_environment,
    normalize_chunking_settings,
)

TEST_SETTINGS = coco.ContextKey[SourceSettings]("yaw/test_settings")

@coco.fn
async def _test_chunk_app():
    settings = coco.use_context(TEST_SETTINGS)
    files = localfs.walk_dir(
        SOURCE_ROOT,
        recursive=False,
        live=False,
        path_matcher=build_source_path_matcher(settings),
        rescan_interval=None,
    )
    chunking_settings = normalize_chunking_settings(settings)
    items = [(rel, file) async for rel, file in files.items()]
    per_file_chunks = await asyncio.gather(*(chunk_file_memo(file, chunking_settings) for _, file in items))
    return [chunk for chunks in per_file_chunks for chunk in chunks]

bundle = load_dataset_bundle(sys.argv[1])
state_dir = Path(sys.argv[2])
kwargs = json.loads(sys.argv[3])
settings = SourceSettings(
    included_patterns=("*.json",),
    excluded_patterns=(),
    chunk_size=kwargs.get("chunk_size", 1200),
    chunk_overlap=kwargs.get("chunk_overlap", 100),
    min_chunk_size=kwargs.get("min_chunk_size", 50),
    json_keys_whitelist=tuple(kwargs.get("json_keys_whitelist", ("*",))),
    json_keys_blacklist=tuple(kwargs.get("json_keys_blacklist", ())),
)

calls = []
def profile(frame, event, arg):
    if event == "call" and frame.f_code.co_name == "chunk_source_text":
        calls.append(frame.f_locals.get("rel_posix_path", ""))

sys.setprofile(profile)
threading.setprofile(profile)

shared = acquire_coco_environment(state_dir)
shared.context.provide(SOURCE_ROOT, bundle.documents_root.resolve())
shared.context.provide(TEST_SETTINGS, settings)
app = coco.App(
    coco.AppConfig(name="test_memo_app", environment=shared.environment, max_inflight_components=4),
    _test_chunk_app,
)

try:
    chunks = asyncio.run(app.update().result())
finally:
    sys.setprofile(None)
    threading.setprofile(None)
    close_coco_environment(state_dir)

chunk_ids = [meta.chunk_id for meta, _ in chunks]
print(json.dumps([chunk_ids, calls]))
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
    assert not any((tmp_path / "state").glob("*.db"))


def test_complete_bundle_document_chunks_isolated_selection_and_real_hydration(tmp_path):
    corpus, qa = inputs(tmp_path)
    bundle = prepare_dataset_bundle("sample", corpus, qa, tmp_path / "outputs/datasets")

    def get_bundle_chunks(bundle, document_ids=None):
        by_name = {Path(d.path).name: d.document_id for d in bundle.documents}
        if document_ids is not None:
            missing = set(document_ids) - set(by_name.values())
            if missing:
                raise ValueError(f"Document IDs missing from {bundle.dataset} manifest: {sorted(missing)}")
        chunks = []
        for doc in bundle.documents:
            if document_ids is not None and doc.document_id not in document_ids:
                continue
            doc_file = bundle.root / doc.path
            text = doc_file.read_text(encoding="utf-8")
            chunks.extend(chunk_source_text(text, f"{bundle.dataset}/{doc.path}", min_chunk_size=1))
        return chunks

    first = get_bundle_chunks(bundle)
    warm = get_bundle_chunks(bundle)
    assert first == warm and len(first) == 2
    for ptr, _ in warm:
        text = hydrate_chunk_pointer(ptr, bundle.root.parent).text
        assert "SECRET" not in text
        assert "/documents/" in ptr.rel_posix_path
    selected = get_bundle_chunks(bundle, document_ids=["b"])
    assert len(selected) == 1 and selected[0][0].title == "Distractor"
    with pytest.raises(ValueError, match="Document IDs missing"):
        get_bundle_chunks(bundle, document_ids=["unknown"])
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
    assert len(selected) == 162 and selected == select_enterprise_questions(rows)
    assert {r["question_id"] for r in selected[:135]}.isdisjoint({r["question_id"] for r in selected[135:]})
    for category, (nt, nv) in ENTERPRISE_CATEGORY_QUOTAS.items():
        assert sum(r["question_type"] == category for r in selected[:135]) == nt
        assert sum(r["question_type"] == category for r in selected[135:]) == nv
    assert before == json.dumps(rows)
    corpus, qa = inputs(tmp_path, questions=rows)
    prepare_dataset_bundle("enterprise_rag_bench", corpus, qa, tmp_path / "outputs/datasets")
    train, val, documents = load_dataset_splits(
        ["enterprise_rag_bench"], workspace_root=tmp_path, max_queries=162, train_queries=135
    )
    assert len(train) == 135 and len(val) == 27 and len(documents) == 1
    assert train[0]["gold_answers"] == [selected[0]["gold_answer"]]
    assert not train[0]["item_docs"]
    small_train, small_val, _ = load_dataset_splits(
        ["enterprise_rag_bench"], workspace_root=tmp_path, max_queries=12, train_queries=7
    )
    assert [row["gold_answers"] for row in small_train] == [[r["gold_answer"]] for r in selected[:7]]
    assert [row["gold_answers"] for row in small_val] == [[r["gold_answer"]] for r in selected[135:140]]
    for alias in ("2wiki", "naturalquestions", "nq_rear", "enterprise"):
        with pytest.raises(ValueError, match="Unknown dataset"):
            resolve_dataset_name(alias)
    with pytest.raises(YawRuntimeError, match="held-out"):
        load_dataset_splits(["nq"], workspace_root=tmp_path)
    presplit = [dict(row, split="train" if i < 135 else "val") for i, row in enumerate(selected)]
    assert select_enterprise_questions(presplit) == presplit
    presplit[135]["question_id"] = presplit[0]["question_id"]
    with pytest.raises(ValueError, match="Duplicate Enterprise question"):
        select_enterprise_questions(presplit)


def test_dataset_bundle_fails_on_malformed_document_then_recovers(tmp_path):
    corpus, qa = inputs(tmp_path)
    bundle = prepare_dataset_bundle("sample", corpus, qa, tmp_path / "bundles")
    first_doc = bundle.root / bundle.documents[0].path
    original_bytes = first_doc.read_bytes()
    validate_dataset_bundle(bundle)
    first_doc.write_text("{malformed", encoding="utf-8")
    with pytest.raises(ValueError, match="Incomplete dataset export"):
        validate_dataset_bundle(bundle)
    first_doc.write_bytes(original_bytes)
    validate_dataset_bundle(bundle)
