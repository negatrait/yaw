"""Real, unmocked tests for 1-indexed line hydration and RLM hydrator tool."""

from pathlib import Path

import pytest

from yaw.core import ChunkPointer
from yaw.daemon.graph.storage.db import open_yaw_db, upsert_chunk_pointers
from yaw.daemon.graph.storage.hydration import (
    hydrate_chunk_pointer,
    hydrate_passages,
    make_chunk_hydrator,
)


def test_hydrate_chunk_pointer_reads_exact_lines(tmp_path: Path) -> None:
    corpus_file = tmp_path / "corpus" / "notes.txt"
    corpus_file.parent.mkdir(parents=True)
    content = (
        "Line 1: Header\n"
        "Line 2: Introduction\n"
        "Line 3: Alan Turing was born in 1912.\n"
        "Line 4: He was a mathematician.\n"
        "Line 5: He worked at Bletchley Park.\n"
        "Line 6: He broke Enigma.\n"
        "Line 7: Conclusion\n"
    )
    corpus_file.write_text(content, encoding="utf-8")

    expected_slice = (
        "Line 3: Alan Turing was born in 1912.\n"
        "Line 4: He was a mathematician.\n"
        "Line 5: He worked at Bletchley Park.\n"
        "Line 6: He broke Enigma."
    )

    pointer = ChunkPointer(
        chunk_id="chunk-3-6",
        rel_posix_path="corpus/notes.txt",
        start_line=3,
        end_line=6,
        title="Turing Bio",
    )

    passage = hydrate_chunk_pointer(pointer, tmp_path, score=0.85)
    assert passage.pointer == pointer
    assert passage.text == expected_slice
    assert passage.score == 0.85
    assert not passage.stale_on_disk


def test_hydration_missing_file_and_empty_slice(tmp_path: Path) -> None:
    pointer_missing = ChunkPointer(
        chunk_id="c_missing",
        rel_posix_path="nonexistent.txt",
        start_line=1,
        end_line=5,
        title="",
    )
    p_missing = hydrate_chunk_pointer(pointer_missing, tmp_path)
    assert p_missing.text == ""
    assert p_missing.stale_on_disk

    # File exists but is shorter than start_line (truncated)
    short_file = tmp_path / "short.txt"
    short_file.write_text("Only one line\n", encoding="utf-8")
    pointer_truncated = ChunkPointer(
        chunk_id="c_trunc",
        rel_posix_path="short.txt",
        start_line=5,
        end_line=10,
        title="",
    )
    p_trunc = hydrate_chunk_pointer(pointer_truncated, tmp_path)
    assert p_trunc.text == ""
    assert p_trunc.stale_on_disk


def test_hydration_rejects_path_traversal(tmp_path: Path) -> None:
    pointer = ChunkPointer(
        chunk_id="c_traversal",
        rel_posix_path="../outside.txt",
        start_line=1,
        end_line=2,
        title="",
    )
    with pytest.raises(ValueError, match="Path traversal detected"):
        hydrate_chunk_pointer(pointer, tmp_path)


def test_hydrate_passages_batch(tmp_path: Path) -> None:
    f1 = tmp_path / "f1.txt"
    f2 = tmp_path / "f2.txt"
    f1.write_text("Line 1 in f1\n", encoding="utf-8")
    f2.write_text("Line 1 in f2\n", encoding="utf-8")

    p1 = ChunkPointer("p1", "f1.txt", 1, 1, "Doc1")
    p2 = ChunkPointer("p2", "f2.txt", 1, 1, "Doc2")

    hydrated = hydrate_passages([(p1, 0.9), (p2, 0.4)], tmp_path)
    assert len(hydrated) == 2
    assert hydrated[0].text == "Line 1 in f1"
    assert hydrated[0].score == 0.9
    assert not hydrated[0].stale_on_disk
    assert hydrated[1].text == "Line 1 in f2"
    assert hydrated[1].score == 0.4
    assert not hydrated[1].stale_on_disk


def test_make_chunk_hydrator_tool(tmp_path: Path) -> None:
    db_file = tmp_path / "yaw.db"
    conn = open_yaw_db(db_file)
    try:
        f = tmp_path / "corpus" / "story.txt"
        f.parent.mkdir(parents=True)
        f.write_text("Once upon a time in a faraway land.\nThere lived a wise queen.\n", encoding="utf-8")

        p = ChunkPointer("c_story", "corpus/story.txt", 1, 2, "Story")
        upsert_chunk_pointers(conn, [(p, f.read_text(encoding="utf-8"))])

        hydrator = make_chunk_hydrator(conn, tmp_path)
        # 1. Existing chunk
        text = hydrator("c_story")
        assert "Once upon a time" in text
        assert "wise queen" in text

        # 2. Non-existent chunk in database
        missing_msg = hydrator("c_unknown")
        assert "Error: Chunk 'c_unknown' not found" in missing_msg

        # 3. File removed from disk surfaces missing error to RLM
        f.unlink()
        missing_file_msg = hydrator("c_story")
        assert "missing or unreadable on disk" in missing_file_msg
    finally:
        conn.close()


def test_hydration_rejects_excessive_line_span(tmp_path: Path) -> None:
    pointer = ChunkPointer(
        chunk_id="c_huge",
        rel_posix_path="huge.txt",
        start_line=1,
        end_line=10000,
        title="",
    )
    with pytest.raises(ValueError, match="exceeds maximum"):
        hydrate_chunk_pointer(pointer, tmp_path)


def test_coalesce_spans_merges_overlapping_and_contiguous_spans() -> None:
    from yaw.daemon.graph.storage.hydration import MutableSpan, coalesce_chunk_spans, coalesce_spans

    spans = [
        MutableSpan("c1", "doc.txt", 1, 10, "TitleA", 0.8, 1, 100),
        MutableSpan("c2", "doc.txt", 11, 20, "TitleA", 0.95, 1, 100),  # contiguous
        MutableSpan("c3", "doc.txt", 15, 25, "TitleA", 0.5, 1, 100),   # overlapping
        MutableSpan("c4", "doc.txt", 35, 40, "TitleB", 0.7, 1, 100),   # disjoint
        MutableSpan("c5", "other.txt", 1, 5, "Other", 0.88, 1, 20),     # different file
    ]

    coalesced = coalesce_spans(spans)
    assert len(coalesced) == 3

    # Most relevant coalesced span first
    assert coalesced[0].rel_posix_path == "doc.txt"
    assert coalesced[0].start_line == 1
    assert coalesced[0].end_line == 25
    assert coalesced[0].max_score == 0.95
    assert set(coalesced[0].hit_chunk_ids) == {"c1", "c2", "c3"}

    # Second is other.txt with 0.88
    assert coalesced[1].rel_posix_path == "other.txt"
    assert coalesced[1].start_line == 1
    assert coalesced[1].end_line == 5
    assert coalesced[1].max_score == 0.88

    # Third is disjoint span with 0.7
    assert coalesced[2].rel_posix_path == "doc.txt"
    assert coalesced[2].start_line == 35
    assert coalesced[2].end_line == 40
    assert coalesced[2].max_score == 0.7

    # Alias check
    assert coalesce_chunk_spans(spans) == coalesced


def test_hydrate_budgeted_passages_inviolate_hits(tmp_path: Path) -> None:
    from yaw.daemon.graph.storage.hydration import hydrate_budgeted_passages

    f = tmp_path / "long_doc.txt"
    lines = [f"This is sentence line {i} with some content.\n" for i in range(1, 51)]
    f.write_text("".join(lines), encoding="utf-8")

    p1 = ChunkPointer("c1", "long_doc.txt", 5, 10, "Sec1", ast_island_start=1, ast_island_end=20)
    p2 = ChunkPointer("c2", "long_doc.txt", 25, 30, "Sec2", ast_island_start=21, ast_island_end=40)

    # Even with max_context_chars smaller than the combined hits, both hit spans are preserved (Inviolate Hit Invariant)
    passages = hydrate_budgeted_passages(
        [(p1, 0.9), (p2, 0.7)],
        corpus_root=tmp_path,
        max_context_chars=100,  # very tight budget
        max_expansion_hops=1,
    )

    assert len(passages) == 2
    assert not passages[0].stale_on_disk
    assert not passages[1].stale_on_disk
    assert "sentence line 5" in passages[0].text
    assert "sentence line 10" in passages[0].text
    assert "sentence line 25" in passages[1].text
    assert "sentence line 30" in passages[1].text


def test_hydrate_budgeted_passages_1hop_ast_expansion(tmp_path: Path) -> None:
    from yaw.daemon.graph.storage.hydration import hydrate_budgeted_passages

    f = tmp_path / "article.txt"
    lines = [f"Line {i:02d}: Context statement for testing.\n" for i in range(1, 41)]
    f.write_text("".join(lines), encoding="utf-8")

    p1 = ChunkPointer("c_hit", "article.txt", 15, 18, "Article", ast_island_start=10, ast_island_end=25)

    # Large budget: expands to ±10 lines bounded by [ast_island_start, ast_island_end] = [10, 25]
    passages = hydrate_budgeted_passages(
        [(p1, 0.9)],
        corpus_root=tmp_path,
        max_context_chars=24000,
        max_expansion_hops=1,
    )
    assert len(passages) == 1
    assert passages[0].pointer.start_line == 10  # max(10, 15 - 10)
    assert passages[0].pointer.end_line == 25    # min(25, 18 + 10)
    assert "Line 10" in passages[0].text
    assert "Line 25" in passages[0].text
    assert "Line 09" not in passages[0].text
    assert "Line 26" not in passages[0].text
