"""Unit tests for yaw.core.types frozen dataclasses, live ErrorCode, and pure ID/formatting helpers."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, cast, get_args

import yaw.core.types as core_types
from yaw.core.types import (
    WIKI_REL_PREFIX,
    ChunkPointer,
    DaemonStatus,
    ErrorCode,
    HydratedPassage,
    QAResult,
    YawRuntimeError,
    advance_markdown_fence,
    canonical_chunk_span_key,
    compute_chunk_int_id,
    format_doc_for_embedding,
    format_query_for_embedding,
    inclusive_end_line,
    is_wiki_rel_path,
    leading_frontmatter_end,
    make_wiki_rel_path,
    markdown_heading_level,
    normalize_ast_heading,
    redact_secrets,
    render_document_text,
    require_corpus_dir,
    resolve_corpus_path,
    resolve_source_path,
    split_lf_lines,
    strip_wiki_rel_path,
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
        self.assertIsNone(passage.hydrated_start_line)
        self.assertIsNone(passage.hydrated_end_line)
        self.assertEqual(passage.resolved_start_line, 1)
        self.assertEqual(passage.resolved_end_line, 10)

        expanded = HydratedPassage(
            pointer=ChunkPointer(2, "p.md", "hash", 3, 6, ast_island_start=1, ast_island_end=12, title="Title"),
            text="# Title\nExpanded",
            score=0.88,
            hydrated_start_line=1,
            hydrated_end_line=12,
        )
        self.assertEqual(expanded.hydrated_start_line, 1)
        self.assertEqual(expanded.hydrated_end_line, 12)
        self.assertEqual(expanded.resolved_start_line, 1)
        self.assertEqual(expanded.resolved_end_line, 12)

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
        self.assertEqual(split_lf_lines(""), [])
        self.assertEqual(split_lf_lines("a\nb\n"), ["a", "b"])
        self.assertEqual(split_lf_lines("a\nb"), ["a", "b"])
        self.assertEqual(normalize_ast_heading("# [RFC-12] Tagged Header — "), "Tagged Header")
        self.assertEqual(markdown_heading_level(["Title", "==="], 1), 1)
        self.assertEqual(markdown_heading_level(["## Sub", "body"], 1), 2)
        fence, is_delim = advance_markdown_fence("```python", ("", 0))
        self.assertEqual((fence, is_delim), (("`", 3), True))
        fence, is_delim = advance_markdown_fence("# Comment Inside Fence", fence)
        self.assertEqual((fence, is_delim), (("`", 3), False))
        fence, is_delim = advance_markdown_fence("```", fence)
        self.assertEqual((fence, is_delim), (("", 0), True))

    def test_retired_legacy_types_removed(self):
        expected_all = [
            "ChunkPointer",
            "DASH_RE",
            "DaemonStatus",
            "EmbedFn",
            "ErrorCode",
            "HydratedPassage",
            "QAResult",
            "WIKI_REL_PREFIX",
            "YawRuntimeError",
            "advance_markdown_fence",
            "canonical_chunk_span_key",
            "compute_chunk_int_id",
            "format_doc_for_embedding",
            "format_query_for_embedding",
            "inclusive_end_line",
            "is_wiki_rel_path",
            "leading_frontmatter_end",
            "make_wiki_rel_path",
            "markdown_heading_level",
            "normalize_ast_heading",
            "redact_secrets",
            "render_document_text",
            "require_corpus_dir",
            "resolve_corpus_path",
            "resolve_source_path",
            "split_lf_lines",
            "strip_wiki_rel_path",
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

    def test_require_corpus_dir_validates_existing_directory(self):
        from yaw.core import require_corpus_dir as pkg_require_corpus_dir

        self.assertIs(pkg_require_corpus_dir, require_corpus_dir)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.assertEqual(require_corpus_dir(root), root.resolve())
            self.assertEqual(require_corpus_dir(str(root)), root.resolve())
            sample = root / "file.md"
            sample.write_text("# Doc\n", encoding="utf-8")
            for bad in (root / "missing", sample, "", "   ", cast(Any, None)):
                with self.assertRaises(YawRuntimeError) as ctx:
                    require_corpus_dir(bad)
                self.assertEqual(ctx.exception.code, "CONSISTENCY_ERROR")

    def test_wiki_rel_path_helpers_resolve_source_path_and_provenance(self):
        self.assertEqual(WIKI_REL_PREFIX, "wiki:")
        self.assertFalse(is_wiki_rel_path("docs/a.md"))
        self.assertTrue(is_wiki_rel_path("wiki:cards/a.md"))

        self.assertEqual(make_wiki_rel_path("cards\\topic.md"), "wiki:cards/topic.md")
        self.assertEqual(make_wiki_rel_path(" spaced name.md"), "wiki: spaced name.md")
        self.assertEqual(strip_wiki_rel_path("wiki:cards/topic.md"), "cards/topic.md")

        for bad_rel in (
            "",
            "   ",
            "a\nb.md",
            "a\x00b.md",
            "/abs.md",
            "C:\\abs.md",
            "C:rel.md",
            "../x.md",
            "wiki:x.md",
            ".",
            "dir/",
        ):
            with self.assertRaises(ValueError):
                make_wiki_rel_path(bad_rel)

        for bad_strip in ("cards/topic.md", "wiki:", "wiki:   "):
            with self.assertRaises(ValueError):
                strip_wiki_rel_path(bad_strip)

        corpus_ptr = ChunkPointer(1, "docs/sample.md", "h", 1, 5)
        wiki_ptr = ChunkPointer(2, "wiki:cards/topic.md", "h", 1, 5)
        self.assertEqual(corpus_ptr.provenance, "corpus")
        self.assertEqual(wiki_ptr.provenance, "wiki")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            corpus_root = root / "corpus"
            wiki_root = root / "wiki"
            (corpus_root / "docs").mkdir(parents=True)
            (wiki_root / "cards").mkdir(parents=True)
            (corpus_root / "docs" / "sample.md").write_text("# Doc\n", encoding="utf-8")
            (wiki_root / "cards" / "topic.md").write_text("# Topic\n", encoding="utf-8")

            self.assertEqual(
                resolve_source_path("docs/sample.md", corpus_root),
                (corpus_root / "docs" / "sample.md").resolve(),
            )
            self.assertEqual(
                resolve_source_path("wiki:cards/topic.md", corpus_root, wiki_root),
                (wiki_root / "cards" / "topic.md").resolve(),
            )
            with self.assertRaises(YawRuntimeError) as exc_ctx:
                resolve_source_path("wiki:cards/topic.md", corpus_root, None)
            self.assertEqual(exc_ctx.exception.code, "CONSISTENCY_ERROR")
            with self.assertRaises(ValueError):
                resolve_source_path("wiki:../escape.md", corpus_root, wiki_root)

    def test_leading_frontmatter_end_and_markdown_heading_level_edge_cases(self):
        fm_lines = [
            "---",
            "kind: note",
            "stale: false",
            "sources: []",
            "---",
            "",
            "First Line",
            "Second Line",
            "===",
            "",
            "Body text.",
        ]
        self.assertEqual(leading_frontmatter_end(fm_lines), 5)
        self.assertEqual(leading_frontmatter_end(fm_lines, 3), 0)
        for ln in range(1, 7):
            self.assertIsNone(markdown_heading_level(fm_lines, ln), f"Line {ln} must not be a heading")
        self.assertEqual(markdown_heading_level(fm_lines, 7), 1)
        self.assertEqual(markdown_heading_level(fm_lines, 8), 1)
        self.assertIsNone(markdown_heading_level(fm_lines, 9))
        self.assertIsNone(markdown_heading_level(fm_lines, 10))

        # Blank/whitespace lines before === or --- must never be headings.
        self.assertIsNone(markdown_heading_level(["", "==="], 1))
        self.assertIsNone(markdown_heading_level(["   ", "---"], 1))

        # Two horizontal rules with non-YAML body must not trigger frontmatter skipping.
        self.assertEqual(leading_frontmatter_end(["---", "", "# After HR"]), 0)
        self.assertEqual(markdown_heading_level(["---", "", "# After HR"], 3), 1)
        self.assertEqual(leading_frontmatter_end(["---", "Plain prose", "---"]), 0)

        # 3-line Setext H1 & H2 supported; 4-line exceeds lookahead.
        self.assertEqual(markdown_heading_level(["A", "B", "C", "==="], 1), 1)
        self.assertEqual(markdown_heading_level(["A", "B", "C", "---"], 1), 2)
        self.assertIsNone(markdown_heading_level(["A", "B", "C", "D", "==="], 1))

        # CommonMark paragraph interrupters stop Setext lookahead.
        for interrupter in ("# ATX", "## Sub", "```py", "~~~", "> quote", "***", "___", "- item"):
            self.assertIsNone(markdown_heading_level(["Lead", interrupter, "==="], 1), interrupter)
        # 4-space indented underline is not a Setext underline.
        self.assertIsNone(markdown_heading_level(["Lead", "    ===", "Body"], 1))
        # #tag without space is paragraph continuation, not ATX interrupter.
        self.assertEqual(markdown_heading_level(["Lead", "#tag", "==="], 1), 1)
