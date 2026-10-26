"""Real, unmocked tests for 1-indexed line hydration and RLM hydrator tool."""

from pathlib import Path

import pytest

from yaw.core import ChunkPointer
from yaw.daemon.graph.storage import open_yaw_db, upsert_chunk_pointers
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
        MutableSpan("c3", "doc.txt", 15, 25, "TitleA", 0.5, 1, 100),  # overlapping
        MutableSpan("c4", "doc.txt", 35, 40, "TitleB", 0.7, 1, 100),  # disjoint
        MutableSpan("c5", "other.txt", 1, 5, "Other", 0.88, 1, 20),  # different file
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
    assert passages[0].pointer.end_line == 25  # min(25, 18 + 10)
    assert "Line 10" in passages[0].text
    assert "Line 25" in passages[0].text
    assert "Line 09" not in passages[0].text
    assert "Line 26" not in passages[0].text


def test_read_corpus_lines_cocoindex_codesource_caching(tmp_path: Path) -> None:
    from yaw.daemon.graph.storage.hydration import read_corpus_lines

    f = tmp_path / "sample.txt"
    f.write_text("Alpha\nBeta\nGamma\nDelta\nEpsilon\n", encoding="utf-8")

    # 1. Normal read with CocoIndex CodeSource cache
    text = read_corpus_lines(f, 2, 4)
    assert text == "Beta\nGamma\nDelta"

    # 2. Modify file on disk -> mtime change invalidates cache automatically
    f.write_text("Alpha\nUpdated Beta\nGamma\nDelta\nEpsilon\n", encoding="utf-8")
    updated = read_corpus_lines(f, 2, 4)
    assert updated == "Updated Beta\nGamma\nDelta"

    # 3. Missing file returns None
    missing = read_corpus_lines(tmp_path / "absent.txt", 1, 2)
    assert missing is None

    # 4. Out of range returns None
    out_of_range = read_corpus_lines(f, 10, 20)
    assert out_of_range is None


@pytest.mark.parametrize("rel_path", ["corpus/records.json", "corpus/records.jsonl", "corpus/records.ndjson"])
def test_coalesce_spans_json_merges_only_identical_line_bounds(rel_path: str) -> None:
    from yaw.daemon.graph.storage.hydration import MutableSpan, coalesce_spans

    spans = [
        # Two sub-chunks of the exact same JSON record at lines 2..7
        MutableSpan("rec1_sub1", rel_path, 2, 7, "SameTitle", 0.65, 2, 7),
        MutableSpan("rec1_sub2", rel_path, 2, 7, "SameTitle", 0.92, 2, 7),
        # Adjacent JSON record at lines 8..13 with identical title and ast_island_start=0
        MutableSpan("rec2_sub1", rel_path, 8, 13, "SameTitle", 0.80, 0, 0),
        # Third adjacent JSON record at lines 14..19 with empty title and ast_island_start=0
        MutableSpan("rec3_sub1", rel_path, 14, 19, "", 0.50, 0, 0),
    ]

    coalesced = coalesce_spans(spans)
    assert len(coalesced) == 3

    # First coalesced span is the merged rec1 (lines 2..7) with max_score=0.92 and both hit chunk IDs
    assert coalesced[0].rel_posix_path == rel_path
    assert coalesced[0].start_line == 2
    assert coalesced[0].end_line == 7
    assert coalesced[0].max_score == 0.92
    assert coalesced[0].hit_chunk_ids == ("rec1_sub1", "rec1_sub2")

    # Second coalesced span is rec2 (lines 8..13), never merged into lines 2..13
    assert coalesced[1].rel_posix_path == rel_path
    assert coalesced[1].start_line == 8
    assert coalesced[1].end_line == 13
    assert coalesced[1].max_score == 0.80
    assert coalesced[1].hit_chunk_ids == ("rec2_sub1",)

    # Third coalesced span is rec3 (lines 14..19), never merged into lines 8..19
    assert coalesced[2].rel_posix_path == rel_path
    assert coalesced[2].start_line == 14
    assert coalesced[2].end_line == 19
    assert coalesced[2].max_score == 0.50
    assert coalesced[2].hit_chunk_ids == ("rec3_sub1",)


def test_resolve_upward_ast_bounds_and_ast_island_hydrator_clamped_for_json(tmp_path: Path) -> None:
    from yaw.daemon.graph.storage.hydration import (
        hydrate_budgeted_passages,
        make_ast_island_hydrator,
        resolve_one_hop_upward_ast_bounds,
    )

    corpus_json = tmp_path / "outputs" / "datasets" / "enterprise_rag_bench_corpus.json"
    corpus_json.parent.mkdir(parents=True, exist_ok=True)
    # Line 1: [
    # Lines 2..7: Record 1
    # Lines 8..13: Record 2
    # Line 14: ]
    json_lines = [
        "[\n",
        "  {\n",
        '    "idx": "doc_1",\n',
        '    "title": "First Document",\n',
        '    "text": "Forbidden sibling record content from doc_1.",\n',
        '    "source_type": "github"\n',
        "  },\n",
        "  {\n",
        '    "idx": "doc_2",\n',
        '    "title": "Second Document",\n',
        '    "text": "Target content inside doc_2 only.",\n',
        '    "source_type": "slack"\n',
        "  }\n",
        "]\n",
    ]
    corpus_json.write_text("".join(json_lines), encoding="utf-8")

    rel_path = "outputs/datasets/enterprise_rag_bench_corpus.json"
    ptr_rec1_sub1 = ChunkPointer(
        chunk_id="chunk_rec1_a",
        rel_posix_path=rel_path,
        start_line=2,
        end_line=7,
        title="[github] First Document",
        ast_island_start=2,
        ast_island_end=7,
    )
    ptr_rec1_sub2 = ChunkPointer(
        chunk_id="chunk_rec1_b",
        rel_posix_path=rel_path,
        start_line=2,
        end_line=7,
        title="[github] First Document",
        ast_island_start=2,
        ast_island_end=7,
    )
    ptr_rec2 = ChunkPointer(
        chunk_id="chunk_rec2",
        rel_posix_path=rel_path,
        start_line=8,
        end_line=13,
        title="[slack] Second Document",
        ast_island_start=8,
        ast_island_end=13,
    )
    ptr_rec2_excised = ChunkPointer(
        chunk_id="chunk_rec2_excised",
        rel_posix_path=rel_path,
        start_line=10,
        end_line=11,
        title="[slack] Second Document",
        ast_island_start=8,
        ast_island_end=13,
    )

    # 1. resolve_one_hop_upward_ast_bounds stays clamped at (8, 13, title) for hops=1 and hops=2
    assert resolve_one_hop_upward_ast_bounds(corpus_json, ptr_rec2, hops=0) == (
        8,
        13,
        "[slack] Second Document",
    )
    assert resolve_one_hop_upward_ast_bounds(corpus_json, ptr_rec2, hops=1) == (
        8,
        13,
        "[slack] Second Document",
    )
    assert resolve_one_hop_upward_ast_bounds(corpus_json, ptr_rec2, hops=2) == (
        8,
        13,
        "[slack] Second Document",
    )
    # Excised inner sub-span expands to enclosing JSON record (8, 13) on hops>=1, never climbing to line 1
    assert resolve_one_hop_upward_ast_bounds(corpus_json, ptr_rec2_excised, hops=0) == (
        10,
        11,
        "[slack] Second Document",
    )
    assert resolve_one_hop_upward_ast_bounds(corpus_json, ptr_rec2_excised, hops=1) == (
        8,
        13,
        "[slack] Second Document",
    )

    # 2. make_ast_island_hydrator hydrates clean normalized text of Record 2 only
    db_file = tmp_path / "yaw.db"
    conn = open_yaw_db(db_file)
    try:
        upsert_chunk_pointers(
            conn,
            [(ptr_rec2, "[slack] Second Document\nTarget content inside doc_2 only.")],
        )
        hydrate_island = make_ast_island_hydrator(conn, tmp_path)
        hydrated_output = hydrate_island("chunk_rec2")

        assert "Lines 8–13, Focus Chunk Lines 8–13" in hydrated_output
        assert "Target content inside doc_2 only." in hydrated_output
        assert "Forbidden sibling record content" not in hydrated_output
        assert "doc_1" not in hydrated_output
    finally:
        conn.close()

    # 3. hydrate_budgeted_passages coalesces sub-chunks of rec1 and keeps adjacent rec2 separate and normalized
    passages = hydrate_budgeted_passages(
        [(ptr_rec2, 0.95), (ptr_rec1_sub1, 0.70), (ptr_rec1_sub2, 0.85)],
        corpus_root=tmp_path,
        max_context_chars=24000,
        max_expansion_hops=1,
    )
    assert len(passages) == 2
    assert passages[0].score == 0.95
    assert passages[0].text == "Second Document\nTarget content inside doc_2 only."
    assert passages[1].score == 0.85
    assert passages[1].text == "First Document\nForbidden sibling record content from doc_1."


@pytest.mark.parametrize("rel_path", ["corpus/data.json", "corpus/data.jsonl", "corpus/data.ndjson"])
def test_hydrate_ast_parent_context_returns_empty_for_json_and_jsonl(tmp_path: Path, rel_path: str) -> None:
    from yaw.daemon.graph.storage.hydration import hydrate_ast_parent_context

    target = tmp_path / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "[\n",
        "  {\n",
        '    "idx": "doc_1",\n',
        '    "title": "First Record",\n',
        '    "text": "Leaked first record text"\n',
        "  },\n",
        "  {\n",
        '    "idx": "doc_2",\n',
        '    "title": "Second Record",\n',
        '    "text": "Second record text"\n',
        "  }\n",
        "]\n",
    ]
    target.write_text("".join(lines), encoding="utf-8")

    # Even when start_line=8 (or inside an excised span with ast_island_start=7, start_line=9) and hops=1 or 2
    ptr = ChunkPointer(
        chunk_id="c_rec2",
        rel_posix_path=rel_path,
        start_line=8,
        end_line=11,
        title="Second Record",
        ast_island_start=7,
        ast_island_end=11,
    )
    assert hydrate_ast_parent_context(ptr, tmp_path, hops=1) == ""
    assert hydrate_ast_parent_context(ptr, tmp_path, hops=2) == ""
