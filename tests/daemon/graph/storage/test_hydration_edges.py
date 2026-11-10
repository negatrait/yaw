"""Disk hydration, JSON normalization and AST context boundary contracts."""

import json
import os

import pytest

from yaw.core import ChunkPointer
from yaw.daemon.graph.storage.hydration import (
    MutableSpan,
    coalesce_spans,
    dehydrate_passage_ast,
    get_cached_code_source,
    hydrate_budgeted_passages,
    hydrate_chunk_pointer,
    hydrate_passages,
    normalize_json_passage_text,
    read_corpus_lines,
)


@pytest.mark.parametrize(
    ("raw", "title", "expected"),
    [
        ("   ", "", ""),
        ('"key": "value",', "", '{"key": "value"}'),
        ('" string "', "", " string "),
        ('{"title":"Title","text":"Body"}', "", "Title\nBody"),
        ('{"content":"Body"}', "Title", "Title\nBody"),
    ],
)
def test_json_passage_variants_preserve_source_content(raw, title, expected):
    assert normalize_json_passage_text(raw, title) == expected


@pytest.mark.parametrize(
    "raw",
    [
        '["Title", ["One.", "Two."]]',
        "42",
        "plain prose",
        "{broken",
        '"a":1 "b":2',
        '"a":1,,"b":2',
    ],
)
def test_json_hydration_rejects_unsupported_or_malformed_sources(raw):
    with pytest.raises(ValueError):
        normalize_json_passage_text(raw)


def test_span_aliases_and_coalescing_preserve_titles_scores_and_hit_identity():
    first = MutableSpan("first", "doc", 1, 2, "", 0.3, 0, 0)
    first.ast_start = 1
    first.ast_end = 8
    assert first.ast_start == 1 and first.ast_end == 8
    second = MutableSpan("second", "doc", 3, 4, "Heading", 0.9, 1, 8)
    assert coalesce_spans([]) == []
    merged = coalesce_spans([first, second])
    assert len(merged) == 1
    assert (merged[0].start_line, merged[0].end_line, merged[0].title) == (1, 4, "Heading")
    assert merged[0].hit_chunk_ids == ("first", "second")
    assert merged[0].max_score == 0.9


def test_code_source_cache_reuses_unchanged_bytes_and_invalidates_after_edit(tmp_path):
    path = tmp_path / "source.txt"
    path.write_text("first\nsecond\n", encoding="utf-8")
    original = get_cached_code_source(path)
    assert original is not None
    unchanged = get_cached_code_source(path)
    assert unchanged is not None and unchanged[0] is original[0]
    assert read_corpus_lines(path, 0, 1) is None
    assert read_corpus_lines(path, 2, 1) is None
    assert read_corpus_lines(path, 1, 2) == "first\nsecond"
    previous = path.stat().st_mtime
    path.write_text("changed\n", encoding="utf-8")
    os.utime(path, (previous + 2, previous + 2))
    edited = get_cached_code_source(path)
    assert edited is not None and edited[0] is not original[0]
    assert read_corpus_lines(path, 1, 1) == "changed"
    assert get_cached_code_source(tmp_path) is None
    assert get_cached_code_source(tmp_path / "missing") is None


@pytest.mark.parametrize("suffix", [".md", ".py", ".json", ".unknown"])
def test_dehydration_uses_real_cocoindex_boundaries_and_preserves_header(suffix):
    body = "\n\n".join(f"Paragraph {i}. Unicode äö and context." for i in range(100))
    text = "# Source\n" + body
    result = dehydrate_passage_ast(text, 400, title="Source", rel_posix_path="source" + suffix)
    assert result.startswith("# Source\n")
    assert result.endswith("[... AST section cropped due to context window limitations ...]")
    assert len(result) <= 400
    assert "\ufffd" not in result
    retained = result.split("\n[... AST")[0]
    assert text.startswith(retained)
    assert dehydrate_passage_ast(text, 0) == ""
    assert dehydrate_passage_ast("short", 100) == "short"
    assert "cropped" in dehydrate_passage_ast(text, 90)


def test_batch_hydration_respects_remaining_context_and_keeps_stale_markers(tmp_path):
    body = "\n".join(f"Line {i}: source context and detail." for i in range(100))
    (tmp_path / "source.md").write_text(body, encoding="utf-8")
    first = ChunkPointer("first", "source.md", 1, 100, "")
    missing = ChunkPointer("missing", "missing.md", 1, 1, "")
    output = hydrate_passages([(missing, 0.9), (first, 0.8), (first, 0.7)], tmp_path, max_total_chars=400)
    assert len(output) == 2
    assert output[0].stale_on_disk
    assert output[1].text.endswith("[... AST section cropped due to context window limitations ...]")
    assert len(output[1].text) <= 400
    assert output[1].score == 0.8
    assert len(hydrate_chunk_pointer(first, tmp_path, max_chars=300).text) <= 300
    assert len(hydrate_passages([(first, 1.0)], tmp_path, max_total_chars=0)[0].text) == len(body)


def test_budgeted_hydration_keeps_missing_and_normalizes_json_hits(tmp_path):
    path = tmp_path / "source.json"
    path.write_text(json.dumps({"title": "Source", "text": "Content"}), encoding="utf-8")
    first = ChunkPointer("first", path.name, 1, 1, "Source")
    missing = ChunkPointer("missing", "absent", 1, 1, "")
    assert hydrate_budgeted_passages([], tmp_path) == []
    result = hydrate_budgeted_passages([(first, 1.0), (missing, 0.5)], tmp_path, max_context_chars=1)
    assert result[0].text == "Source\nContent"
    assert result[1].stale_on_disk
    assert result[1].pointer.chunk_id == "missing"


def test_hydration_rejects_invalid_utf8_instead_of_replacing_bytes(tmp_path):
    from yaw.core import ChunkPointer
    from yaw.daemon.graph.storage import hydrate_chunk_pointer

    (tmp_path / "invalid.json").write_bytes(b'{"text":"invalid\xff"}')
    with pytest.raises(UnicodeDecodeError):
        hydrate_chunk_pointer(ChunkPointer("chunk-invalid", "invalid.json", 1, 1), tmp_path)
