"""Unit tests for yaw.core.types frozen dataclasses."""

from dataclasses import FrozenInstanceError
import unittest

from yaw.core.types import (
    ChunkPointer,
    HydratedPassage,
    OpenIERecord,
    RetrievalResult,
    SyncDelta,
)


class TypesTests(unittest.TestCase):
    def test_chunk_pointer_invariants(self):
        pointer = ChunkPointer(
            chunk_id="chunk-1",
            rel_posix_path="docs/sample.md",
            start_line=1,
            end_line=25,
            title="Sample Title",
            content_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        )
        self.assertEqual(pointer.chunk_id, "chunk-1")
        self.assertEqual(pointer.rel_posix_path, "docs/sample.md")
        self.assertEqual(pointer.start_line, 1)
        self.assertEqual(pointer.end_line, 25)
        self.assertEqual(pointer.title, "Sample Title")
        self.assertEqual(pointer.content_hash, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")

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
            content_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        )
        self.assertEqual(pointer, pointer2)
        self.assertEqual(hash(pointer), hash(pointer2))
        self.assertEqual(len({pointer, pointer2}), 1)

    def test_hydrated_passage_invariants(self):
        pointer = ChunkPointer("c1", "p.md", 1, 10, "Title", "hash1")
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
        pointer = ChunkPointer("c1", "p.md", 1, 10, "Title", "hash1")
        passage = HydratedPassage(pointer=pointer, text="Hello world", score=0.95)
        result = RetrievalResult(question="What is this?", passages=(passage,))
        self.assertEqual(result.question, "What is this?")
        self.assertEqual(result.passages, (passage,))
        self.assertIsNone(result.answer)

        result_with_answer = RetrievalResult(
            question="What is this?", passages=(passage,), answer="This is a test."
        )
        self.assertEqual(result_with_answer.answer, "This is a test.")

        with self.assertRaises((FrozenInstanceError, AttributeError)):
            result.answer = "mutated"  # type: ignore
        self.assertFalse(hasattr(result, "__dict__"))

    def test_sync_delta_invariants(self):
        delta_empty = SyncDelta()
        self.assertEqual(delta_empty.added_chunks, ())
        self.assertEqual(delta_empty.deleted_chunk_ids, ())

        pointer = ChunkPointer("c1", "p.md", 1, 10, "Title", "hash1")
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
