"""Unit tests for yaw.core.types frozen dataclasses, live ErrorCode, and pure ID/formatting helpers."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, cast, get_args

import yaw.core.types as core_types
from yaw.core.types import (
    ChunkPointer,
    DaemonStatus,
    ErrorCode,
    HydratedPassage,
    QAResult,
    YawRuntimeError,
    canonical_chunk_span_key,
    compute_chunk_int_id,
    format_doc_for_embedding,
    format_query_for_embedding,
    inclusive_end_line,
    normalize_entity,
    redact_secrets,
    render_document_text,
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
            cast(Any, pointer).start_line = 2

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
            cast(Any, passage).score = 1.0
        self.assertFalse(hasattr(passage, "__dict__"))

    def test_qa_result_and_daemon_status_contracts(self):
        pointer = ChunkPointer(1, "p.md", "hash", 1, 10)
        passage = HydratedPassage(pointer=pointer, text="Hello world", score=0.95)
        result = QAResult("What is this?", "A greeting.", ("p.md:1-10",), (passage,))
        self.assertEqual(result.citations, ("p.md:1-10",))
        self.assertEqual(result.passages, (passage,))
        status = DaemonStatus(lifecycle="ready", provider_ready=True, num_chunks=2)
        self.assertEqual(status.to_dict()["num_chunks"], 2)

    def test_chunk_int_ids_are_deterministic_and_include_source_coordinates(self):
        cid = compute_chunk_int_id("doc.md", 1, 10, source_hash="hash")
        self.assertEqual(cid, compute_chunk_int_id("doc.md", 1, 10, source_hash="hash"))
        self.assertNotEqual(cid, compute_chunk_int_id("doc.md", 2, 10, source_hash="hash"))
        self.assertNotEqual(cid, compute_chunk_int_id("doc.md", 1, 10, char_start=1, char_end=5, source_hash="hash"))
        self.assertTrue(0 < cid < (1 << 63))
        self.assertEqual(canonical_chunk_span_key("a\\b.md", 1, 3, 2, 10, "h"), "a/b.md:1:3:2:10:h")

    def test_inclusive_end_line(self):
        self.assertEqual(inclusive_end_line(start_line=1, end_line=5, end_column=1), 4)
        self.assertEqual(inclusive_end_line(start_line=1, end_line=1, end_column=1), 1)
        self.assertEqual(inclusive_end_line(start_line=1, end_line=5, end_column=15), 5)

    def test_resolve_corpus_path_and_rejection(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir).resolve()
            sub = root / "nested" / "doc.md"
            sub.parent.mkdir(parents=True)
            sub.write_text("content", encoding="utf-8")

            resolved = resolve_corpus_path(root, "nested/doc.md")
            self.assertEqual(resolved, sub)

            with self.assertRaises(ValueError):
                resolve_corpus_path(root, "../outside.txt")

    def test_live_error_codes_redaction_and_embedding_formatters(self):
        codes = set(get_args(ErrorCode))
        self.assertEqual(
            codes,
            {
                "BUSY",
                "CONSISTENCY_ERROR",
                "EMPTY_INDEX",
                "INDEX_NOT_READY",
                "INVALID_INPUT",
                "MODEL_ERROR",
                "NOT_READY",
                "SHUTTING_DOWN",
            },
        )
        err = YawRuntimeError("INVALID_INPUT", "bad input")
        self.assertEqual(str(err), "[INVALID_INPUT] bad input")

        self.assertEqual(normalize_entity("  New   York "), "new york")
        masked = redact_secrets(
            "Bearer abc123 sk-secret123456 https://user:pass@host/api api_key=tok_99 my-raw-secret",
            "my-raw-secret",
        )
        self.assertNotIn("abc123", masked)
        self.assertNotIn("sk-secret123456", masked)
        self.assertNotIn("user:pass", masked)
        self.assertNotIn("tok_99", masked)
        self.assertNotIn("my-raw-secret", masked)

        self.assertEqual(render_document_text("Doc", "Body"), "Doc\nBody")
        self.assertEqual(format_query_for_embedding("who?"), "task: question answering | query: who?")
        self.assertEqual(format_doc_for_embedding("Body", title="Doc"), "title: Doc | text: Body")

    def test_retired_legacy_types_removed(self):
        expected_all = [
            "ChunkPointer",
            "DaemonStatus",
            "EmbedFn",
            "ErrorCode",
            "HydratedPassage",
            "QAResult",
            "YawRuntimeError",
            "canonical_chunk_span_key",
            "compute_chunk_int_id",
            "format_doc_for_embedding",
            "format_query_for_embedding",
            "inclusive_end_line",
            "normalize_entity",
            "redact_secrets",
            "render_document_text",
            "resolve_corpus_path",
        ]
        self.assertEqual(core_types.__all__, expected_all)
        for retired in (
            "json_document_fields",
            "FatalGraphPublicationError",
            "TextChunk",
            "Entity",
            "Triple",
            "Quad",
            "ExtractionRecord",
            "DropReason",
            "PassMetrics",
            "AxiomViolation",
            "ValidationSummary",
            "ManifoldDiagnostics",
            "OntologySummary",
            "PipelineResult",
            "DatasetStats",
            "ChunkMetadata",
            "RecallHit",
            "RecallResult",
            "EvalQuestionResult",
            "EvalAggregate",
            "compute_chunk_id",
            "stable_hash_id",
        ):
            self.assertFalse(hasattr(core_types, retired), f"Retired symbol {retired} still present")
