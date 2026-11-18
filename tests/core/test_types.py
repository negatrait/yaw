"""Unit tests for yaw.core.types frozen dataclasses and pure coordinate/ID helpers."""

import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from yaw.core.types import (
    ChunkPointer,
    HydratedPassage,
    QAResult,
    compute_chunk_id,
    compute_chunk_int_id,
    inclusive_end_line,
    resolve_corpus_path,
)


class TypesTests(unittest.TestCase):
    def test_chunk_pointer_invariants(self):
        pointer = ChunkPointer(
            chunk_id=1,
            rel_path="docs/sample.md",
            source_hash="hash",
            start_line=1,
            end_line=25,
            title="Sample Title",
        )
        self.assertEqual(pointer.chunk_id, 1)
        self.assertEqual(pointer.rel_posix_path, "docs/sample.md")
        self.assertEqual(pointer.start_line, 1)
        self.assertEqual(pointer.end_line, 25)
        self.assertEqual(pointer.title, "Sample Title")

        # Frozen contract
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            pointer.start_line = 2  # type: ignore

        # Slots contract (no __dict__)
        self.assertFalse(hasattr(pointer, "__dict__"))

        # Value equality & hashing
        pointer2 = ChunkPointer(
            chunk_id=1,
            rel_path="docs/sample.md",
            source_hash="hash",
            start_line=1,
            end_line=25,
            title="Sample Title",
        )
        self.assertEqual(pointer, pointer2)
        self.assertEqual(hash(pointer), hash(pointer2))
        self.assertEqual(len({pointer, pointer2}), 1)

    def test_hydrated_passage_invariants(self):
        pointer = ChunkPointer(1, "p.md", "hash", 1, 10, title="Title")
        passage = HydratedPassage(pointer=pointer, text="Hello world", score=0.95)
        self.assertEqual(passage.pointer, pointer)
        self.assertEqual(passage.text, "Hello world")
        self.assertEqual(passage.score, 0.95)
        self.assertFalse(passage.stale_on_disk)

        stale_passage = HydratedPassage(pointer=pointer, text="Stale", score=0.5, stale_on_disk=True)
        self.assertTrue(stale_passage.stale_on_disk)

        with self.assertRaises((FrozenInstanceError, AttributeError)):
            passage.score = 1.0  # type: ignore
        self.assertFalse(hasattr(passage, "__dict__"))

    def test_qa_result_keeps_citation_labels_and_hydrated_passages(self):
        pointer = ChunkPointer(1, "p.md", "hash", 1, 10)
        passage = HydratedPassage(pointer=pointer, text="Hello world", score=0.95)
        result = QAResult("What is this?", "A greeting.", ("p.md:1-10",), (passage,))
        self.assertEqual(result.citations, ("p.md:1-10",))
        self.assertEqual(result.passages, (passage,))

    def test_chunk_ids_are_deterministic_and_include_source_coordinates(self):
        chunk_id = compute_chunk_id("doc.md", 1, 10, source_hash="hash")
        self.assertEqual(chunk_id, compute_chunk_id("doc.md", 1, 10, source_hash="hash"))
        self.assertNotEqual(chunk_id, compute_chunk_id("doc.md", 2, 10, source_hash="hash"))
        self.assertGreater(compute_chunk_int_id("doc.md", 1, 10, source_hash="hash"), 0)

    def test_inclusive_end_line(self):
        # When end_column == 1 and end_line > start_line, slice ended at trailing newline
        self.assertEqual(inclusive_end_line(start_line=1, end_line=5, end_column=1), 4)
        # When end_line == start_line, never decrement below start_line
        self.assertEqual(inclusive_end_line(start_line=1, end_line=1, end_column=1), 1)
        # When end_column > 1, slice ended midway on end_line
        self.assertEqual(inclusive_end_line(start_line=1, end_line=5, end_column=15), 5)

    def test_resolve_corpus_path_and_to_rel_posix_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir).resolve()
            sub = root / "nested" / "doc.md"
            sub.parent.mkdir(parents=True)
            sub.write_text("content", encoding="utf-8")

            # Valid resolution
            resolved = resolve_corpus_path(root, "nested/doc.md")
            self.assertEqual(resolved, sub)

            # Traversal attempt rejection
            with self.assertRaises(ValueError):
                resolve_corpus_path(root, "../outside.txt")
