"""Unit tests for yaw.core.types frozen dataclasses and pure coordinate/ID helpers."""

import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from yaw.core.types import (
    ChunkPointer,
    HydratedPassage,
    OpenIERecord,
    RetrievalResult,
    SyncDelta,
    compute_chunk_id,
    compute_mdhash_id,
    inclusive_end_line,
    resolve_corpus_path,
    to_rel_posix_path,
)


class TypesTests(unittest.TestCase):
    def test_chunk_pointer_invariants(self):
        pointer = ChunkPointer(
            chunk_id="chunk-1",
            rel_posix_path="docs/sample.md",
            start_line=1,
            end_line=25,
            title="Sample Title",
        )
        self.assertEqual(pointer.chunk_id, "chunk-1")
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
            chunk_id="chunk-1",
            rel_posix_path="docs/sample.md",
            start_line=1,
            end_line=25,
            title="Sample Title",
        )
        self.assertEqual(pointer, pointer2)
        self.assertEqual(hash(pointer), hash(pointer2))
        self.assertEqual(len({pointer, pointer2}), 1)

    def test_hydrated_passage_invariants(self):
        pointer = ChunkPointer("c1", "p.md", 1, 10, "Title")
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

    def test_openie_record_invariants(self):
        record = OpenIERecord(
            chunk_id="c1",
            entities=("Apple", "California"),
            triples=(("Apple", "headquartered in", "California"),),
        )
        self.assertEqual(record.chunk_id, "c1")
        self.assertEqual(record.entities, ("Apple", "California"))
        self.assertEqual(record.triples, (("Apple", "headquartered in", "California"),))

        with self.assertRaises((FrozenInstanceError, AttributeError)):
            record.chunk_id = "c2"  # type: ignore
        self.assertFalse(hasattr(record, "__dict__"))

    def test_retrieval_result_invariants(self):
        pointer = ChunkPointer("c1", "p.md", 1, 10, "Title")
        passage = HydratedPassage(pointer=pointer, text="Hello world", score=0.95)
        result = RetrievalResult(question="What is this?", passages=(passage,))
        self.assertEqual(result.question, "What is this?")
        self.assertEqual(result.passages, (passage,))
        self.assertIsNone(result.answer)

        result_with_answer = RetrievalResult(question="What is this?", passages=(passage,), answer="This is a test.")
        self.assertEqual(result_with_answer.answer, "This is a test.")

        with self.assertRaises((FrozenInstanceError, AttributeError)):
            result.answer = "mutated"  # type: ignore
        self.assertFalse(hasattr(result, "__dict__"))

    def test_sync_delta_invariants(self):
        delta_empty = SyncDelta()
        self.assertEqual(delta_empty.added_chunks, ())
        self.assertEqual(delta_empty.deleted_chunk_ids, ())

        pointer = ChunkPointer("c1", "p.md", 1, 10, "Title")
        delta = SyncDelta(
            added_chunks=((pointer, "Full chunk body text"),),
            deleted_chunk_ids=("c0",),
        )
        self.assertEqual(len(delta.added_chunks), 1)
        self.assertEqual(delta.added_chunks[0][0], pointer)
        self.assertEqual(delta.added_chunks[0][1], "Full chunk body text")
        self.assertEqual(delta.deleted_chunk_ids, ("c0",))

        with self.assertRaises((FrozenInstanceError, AttributeError)):
            delta.deleted_chunk_ids = ()  # type: ignore
        self.assertFalse(hasattr(delta, "__dict__"))

    def test_compute_mdhash_id_and_chunk_id(self):
        content = "test text"
        raw_hash = compute_mdhash_id(content)
        self.assertEqual(len(raw_hash), 32)
        prefixed = compute_mdhash_id(content, prefix="entity-")
        self.assertTrue(prefixed.startswith("entity-"))
        self.assertEqual(prefixed, f"entity-{raw_hash}")

        chunk_id = compute_chunk_id("doc.md", 1, 10, content)
        self.assertTrue(chunk_id.startswith("chunk-"))
        self.assertEqual(len(chunk_id), 38)  # len("chunk-") + 32 hex

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

            # Valid relative posix path conversion
            rel = to_rel_posix_path(root, sub)
            self.assertEqual(rel, "nested/doc.md")

            # Traversal attempt rejection
            with self.assertRaises(ValueError):
                resolve_corpus_path(root, "../outside.txt")
            with self.assertRaises(ValueError):
                to_rel_posix_path(root, root.parent / "outside.txt")
