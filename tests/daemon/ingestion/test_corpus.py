"""Unmocked integration tests for native CocoIndex v1 incremental line-pointer ingestion."""

from pathlib import Path

from yaw.daemon.graph.storage import hydrate_chunk_pointer
from yaw.daemon.ingestion import (
    CorpusIngestor,
    chunk_markdown_text,
    extract_chunk_title,
    extract_markdown_headings,
    resolve_heading_for_line,
    sync_corpus_once,
)


def test_chunk_markdown_text_invariants() -> None:
    content = """# System Architecture

This is the introductory overview of the distributed system.
It covers several primary components.

## Data Layer

The storage engine uses SQLite with WAL mode.
It maintains high read concurrency and ACID guarantees.
"""
    chunks = chunk_markdown_text(content, "docs/architecture.md", chunk_size=300, chunk_overlap=30)
    assert len(chunks) >= 1

    lines = content.splitlines(keepends=True)
    for ptr, text in chunks:
        assert ptr.chunk_id.startswith("chunk-")
        assert ptr.rel_posix_path == "docs/architecture.md"
        assert ptr.start_line >= 1
        assert ptr.end_line >= ptr.start_line
        assert ptr.title in ("System Architecture", "Data Layer")

        # Verify disk line slice integrity
        slice_text = "".join(lines[ptr.start_line - 1 : ptr.end_line]).rstrip("\r\n")
        assert slice_text == text.rstrip("\r\n")


def test_chunk_markdown_text_empty_and_fallback_title() -> None:
    assert chunk_markdown_text("", "empty.md") == ()
    assert chunk_markdown_text("   \n\n  ", "spaces.md") == ()

    # Document without any headings falls back to stem
    plain = "Just a single paragraph without any markdown headings."
    chunks = chunk_markdown_text(plain, "raw/unstructured_notes.txt")
    assert len(chunks) == 1
    ptr, text = chunks[0]
    assert ptr.title == "unstructured_notes"
    assert ptr.start_line == 1
    assert ptr.end_line == 1
    assert text == plain


def test_extract_markdown_headings_and_skips_code_fences() -> None:
    doc = (
        "Intro text before any heading\n"
        "# Chapter 1\n"
        "Body paragraph.\n"
        "```python\n"
        "# This is a python comment, not a markdown heading\n"
        "x = 42\n"
        "```\n"
        "## Section 1.1\n"
        "More body text.\n\n"
        "Setext Chapter 2\n"
        "================\n"
        "Setext body.\n"
    )
    headings = extract_markdown_headings(doc)
    # Must only contain Chapter 1, Section 1.1, and Setext Chapter 2
    titles = [h[1] for h in headings]
    assert "Chapter 1" in titles
    assert "Section 1.1" in titles
    assert "Setext Chapter 2" in titles
    assert not any("comment" in t for t in titles)

    # Verify line numbers
    lines_by_title = {title: line for line, title in headings}
    assert lines_by_title["Chapter 1"] == 2
    assert lines_by_title["Section 1.1"] == 8
    assert lines_by_title["Setext Chapter 2"] == 11


def test_resolve_heading_for_line_and_extract_chunk_title() -> None:
    headings = [(2, "Chapter 1"), (8, "Section 1.1"), (11, "Chapter 2")]
    # Line 1: before any heading -> fallback
    assert resolve_heading_for_line(headings, 1, "default") == "default"
    # Line 2: exact heading line
    assert resolve_heading_for_line(headings, 2, "default") == "Chapter 1"
    # Line 5: between Chapter 1 and Section 1.1
    assert resolve_heading_for_line(headings, 5, "default") == "Chapter 1"
    # Line 8: Section 1.1
    assert resolve_heading_for_line(headings, 8, "default") == "Section 1.1"
    # Line 20: after Chapter 2
    assert resolve_heading_for_line(headings, 20, "default") == "Chapter 2"

    doc = "# Top Title\n\nIntro\n\n## Subheading\n\nLine 6\nLine 7"
    assert extract_chunk_title(doc, start_line=1, rel_posix_path="a.md") == "Top Title"
    assert extract_chunk_title(doc, start_line=5, rel_posix_path="a.md") == "Subheading"
    assert extract_chunk_title(doc, start_line=7, rel_posix_path="a.md") == "Subheading"


def test_ingestion_lifecycle_add_zero_op_edit_rename_delete(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    state_dir = tmp_path / "state"
    corpus_dir.mkdir()
    state_dir.mkdir()

    # Step 1: Initial file add
    doc1 = corpus_dir / "quantum.md"
    doc1.write_text(
        "# Quantum Theory\n\nFormulated by Werner Heisenberg and Erwin Schrodinger.\n",
        encoding="utf-8",
    )
    doc2 = corpus_dir / "relativity.txt"
    doc2.write_text(
        "General Relativity was developed by Albert Einstein between 1907 and 1915.\n",
        encoding="utf-8",
    )

    ingestor = CorpusIngestor(corpus_dir, state_dir)

    delta1 = ingestor.poll()
    assert len(delta1.added_chunks) == 2
    assert delta1.deleted_chunk_ids == ()

    added_by_path = {ptr.rel_posix_path: (ptr, text) for ptr, text in delta1.added_chunks}
    assert "quantum.md" in added_by_path
    assert "relativity.txt" in added_by_path

    q_ptr, q_text = added_by_path["quantum.md"]
    assert q_ptr.title == "Quantum Theory"
    assert "Heisenberg" in q_text

    # Verify line hydration compatibility
    h_passage = hydrate_chunk_pointer(q_ptr, corpus_dir)
    assert not h_passage.stale_on_disk
    assert "Heisenberg" in h_passage.text

    # Step 2: Memoized zero-op re-scan (nothing modified on disk)
    delta2 = ingestor.poll()
    assert delta2.added_chunks == ()
    assert delta2.deleted_chunk_ids == ()

    # Step 3: Incremental file edit (modify quantum.md, leave relativity.txt untouched)
    doc1.write_text(
        "# Quantum Mechanics Extended\n\nMax Planck introduced the quantum hypothesis in 1900.\n",
        encoding="utf-8",
    )
    delta3 = ingestor.poll()
    assert len(delta3.added_chunks) == 1
    assert len(delta3.deleted_chunk_ids) == 1

    new_q_ptr, new_q_text = delta3.added_chunks[0]
    assert new_q_ptr.rel_posix_path == "quantum.md"
    assert new_q_ptr.title == "Quantum Mechanics Extended"
    assert "Max Planck" in new_q_text
    assert delta3.deleted_chunk_ids[0] == q_ptr.chunk_id

    # Step 4: File rename (rename relativity.txt to relativity_v2.txt)
    r_ptr, _ = added_by_path["relativity.txt"]
    doc2_renamed = corpus_dir / "relativity_v2.txt"
    doc2.rename(doc2_renamed)

    delta4 = ingestor.poll()
    assert len(delta4.added_chunks) == 1
    assert len(delta4.deleted_chunk_ids) == 1

    renamed_ptr, renamed_text = delta4.added_chunks[0]
    assert renamed_ptr.rel_posix_path == "relativity_v2.txt"
    assert "Albert Einstein" in renamed_text
    assert delta4.deleted_chunk_ids[0] == r_ptr.chunk_id

    # Step 5: File delete (remove relativity_v2.txt)
    doc2_renamed.unlink()

    delta5 = ingestor.poll()
    assert delta5.added_chunks == ()
    assert len(delta5.deleted_chunk_ids) == 1
    assert delta5.deleted_chunk_ids[0] == renamed_ptr.chunk_id

    # Step 6: Post-deletion zero-op re-scan
    delta6 = ingestor.poll()
    assert delta6.added_chunks == ()
    assert delta6.deleted_chunk_ids == ()


def test_sync_corpus_once_helper(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    state_dir = tmp_path / "state"
    corpus_dir.mkdir()
    state_dir.mkdir()

    note = corpus_dir / "sample.md"
    note.write_text("# Sample Note\nContent here.\n", encoding="utf-8")

    delta = sync_corpus_once(corpus_dir, state_dir)
    assert len(delta.added_chunks) == 1
    assert delta.deleted_chunk_ids == ()
    assert delta.added_chunks[0][0].title == "Sample Note"


def test_ingestion_memoization_skips_untouched_files_unmocked(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    state_dir = tmp_path / "state"
    corpus_dir.mkdir()
    state_dir.mkdir()

    f1 = corpus_dir / "file1.md"
    f2 = corpus_dir / "file2.md"
    f1.write_text("# File 1\nLine 1\n", encoding="utf-8")
    f2.write_text("# File 2\nLine 1\n", encoding="utf-8")

    with CorpusIngestor(corpus_dir, state_dir) as ingestor:
        # Run 1: Both files discovered and processed
        d1 = ingestor.poll()
        assert len(d1.added_chunks) == 2
        assert len(d1.deleted_chunk_ids) == 0

        # Run 2: Neither file modified -> zero delta emitted without any mock
        d2 = ingestor.poll()
        assert len(d2.added_chunks) == 0
        assert len(d2.deleted_chunk_ids) == 0

        # Run 3: Modify only file1.md -> only file1.md is emitted as an update
        f1.write_text("# File 1 Updated\nNew line\n", encoding="utf-8")
        d3 = ingestor.poll()
        assert len(d3.added_chunks) == 1
        assert len(d3.deleted_chunk_ids) == 1
        assert d3.added_chunks[0][0].rel_posix_path == "file1.md"
        assert d3.added_chunks[0][0].title == "File 1 Updated"

    # Run 4: Fresh CorpusIngestor instance on the existing state_dir preserves memoization
    with CorpusIngestor(corpus_dir, state_dir) as ingestor_fresh:
        d4 = ingestor_fresh.poll()
        assert len(d4.added_chunks) == 0
        assert len(d4.deleted_chunk_ids) == 0


def test_target_sink_compiler_callback_and_retry_on_failure(tmp_path: Path) -> None:
    import pytest

    corpus_dir = tmp_path / "corpus"
    state_dir = tmp_path / "state"
    corpus_dir.mkdir()
    state_dir.mkdir()

    note = corpus_dir / "note.md"
    note.write_text("# Target Sink\nFirst paragraph.\n", encoding="utf-8")

    observed_deltas = []
    should_fail = True

    def compiler_callback(delta):
        nonlocal should_fail
        if should_fail:
            raise RuntimeError("Simulated compiler failure inside target sink")
        observed_deltas.append(delta)

    with CorpusIngestor(corpus_dir, state_dir, on_delta=compiler_callback) as ingestor:
        # First poll raises RuntimeError before CocoIndex commits state
        with pytest.raises(RuntimeError, match="Simulated compiler failure"):
            ingestor.poll()
        assert len(observed_deltas) == 0

        # Second poll succeeds and replays the uncommitted file delta
        should_fail = False
        delta = ingestor.poll()
        assert len(delta.added_chunks) == 1
        assert len(observed_deltas) == 1
        assert observed_deltas[0] == delta


def test_parser_helper_edit_invalidates_persistent_memo_across_processes(tmp_path: Path) -> None:
    import json
    import os
    import shutil
    import subprocess
    import sys

    snapshot = tmp_path / "implementation"
    shutil.copytree(
        Path(__file__).resolve().parents[3] / "yaw", snapshot / "yaw",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "note.md").write_text("# Original\nA passage about graph indexing.\n")
    worker = """
import json, sys
sys.path.insert(0, sys.argv[1])
from yaw.daemon.ingestion import CorpusIngestor
with CorpusIngestor(sys.argv[2], sys.argv[3]) as ingestor:
    delta = ingestor.poll()
print(json.dumps([p.title for p, text in delta.added_chunks]))
"""

    def scan() -> list[str]:
        result = subprocess.run(
            [sys.executable, "-c", worker, str(snapshot), str(corpus), str(tmp_path / "state")],
            capture_output=True, text=True, check=True, timeout=60,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        return json.loads(result.stdout.strip().splitlines()[-1])

    assert scan() == ["Original"]
    assert scan() == []
    module = snapshot / "yaw/daemon/ingestion/corpus.py"
    before = module.read_text(encoding="utf-8")
    after = before.replace(
        "title = resolve_heading_for_line(headings, start_line, fallback_title)",
        'title = "Changed by parser helper"',
    )
    assert before != after
    module.write_text(after, encoding="utf-8")
    assert scan() == ["Changed by parser helper"]
    assert scan() == []


def test_zero_markdown_it_in_corpus_module() -> None:
    import ast
    import inspect

    import yaw.daemon.ingestion.corpus as corpus_mod

    source = inspect.getsource(corpus_mod)
    tree = ast.parse(source)
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module.split(".")[0])
    assert "markdown_it" not in imported_modules
    assert "cocoindex" in imported_modules


def test_native_cocoindex_ast_headings_across_languages() -> None:
    from yaw.daemon.ingestion import chunk_source_text, extract_ast_headings

    # 1. HTML (<title>, <h1>, <h2>)
    html_doc = (
        "<html>\n"
        "<head><title>Portal Overview</title></head>\n"
        "<body>\n"
        "<h1>Authentication</h1>\n"
        "<p>OAuth2 flow details.</p>\n"
        "<h2>Token Refresh</h2>\n"
        "<p>Refresh token rotation policy.</p>\n"
        "</body>\n"
        "</html>\n"
    )
    html_headings = extract_ast_headings(html_doc, language="html")
    html_titles = [t for _, t in html_headings]
    assert "Portal Overview" in html_titles
    assert "Authentication" in html_titles
    assert "Token Refresh" in html_titles

    # 2. TOML ([section])
    toml_doc = "[ingestion]\nchunk_size = 1200\n\n[graph.ppr]\ndamping = 0.5\n"
    toml_headings = extract_ast_headings(toml_doc, language="toml")
    assert toml_headings == [(1, "ingestion"), (4, "graph.ppr")]

    # 3. YAML (top-level keys)
    yaml_doc = "service:\n  host: localhost\ndatabase:\n  path: yaw.db\n"
    yaml_headings = extract_ast_headings(yaml_doc, language="yaml")
    assert yaml_headings == [(1, "service"), (3, "database")]

    # 4. Python (class and def AST scopes)
    py_doc = (
        "class TokenVerifier:\n"
        "    def verify_signature(self, payload: bytes) -> bool:\n"
        "        digest = len(payload)\n"
        "        return digest > 0\n"
    )
    py_headings = extract_ast_headings(py_doc, language="python")
    py_titles = [t for _, t in py_headings]
    assert any("class TokenVerifier" in t for t in py_titles)
    assert any("def verify_signature" in t for t in py_titles)

    py_chunks = chunk_source_text(py_doc, "auth/verifier.py")
    assert len(py_chunks) == 1
    assert "TokenVerifier" in py_chunks[0][0].title


def test_corporate_json_secret_blacklisting_and_hydration(tmp_path: Path) -> None:
    import json

    corpus_dir = tmp_path / "corpus"
    state_dir = tmp_path / "state"
    corpus_dir.mkdir()
    state_dir.mkdir()

    corporate_payload = [
        {
            "title": "Internal Payment Gateway",
            "user": "corp_admin_secret_user",
            "password": "super_secret_password_987",
            "credentials": {
                "db_password": "nested_password_456",
                "auth_header": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.secret",
            },
            "endpoint": "https://payments.internal.example/v2",
            "retries": 3,
        }
    ]
    json_file = corpus_dir / "services.json"
    json_file.write_text(json.dumps(corporate_payload, indent=2), encoding="utf-8")

    delta = sync_corpus_once(
        corpus_dir,
        state_dir,
        json_keys_whitelist=("*",),
        json_keys_blacklist=("Bearer*", "*password*", "user"),
    )
    assert len(delta.added_chunks) == 1
    ptr, chunk_text = delta.added_chunks[0]
    assert ptr.title == "Internal Payment Gateway"
    assert "https://payments.internal.example/v2" in chunk_text
    assert "corp_admin_secret_user" not in chunk_text
    assert "super_secret_password_987" not in chunk_text
    assert "nested_password_456" not in chunk_text
    assert "Bearer eyJ" not in chunk_text

    # Physical air-gap check on source file and 1-indexed line hydration
    hydrated = hydrate_chunk_pointer(ptr, corpus_dir)
    assert not hydrated.stale_on_disk
    assert "Internal Payment Gateway" in hydrated.text
    assert "https://payments.internal.example/v2" in hydrated.text
    assert "corp_admin_secret_user" not in hydrated.text
    assert "super_secret_password_987" not in hydrated.text
    assert "Bearer eyJ" not in hydrated.text


def test_benchmark_json_air_gapping_and_exact_passage_hydration(tmp_path: Path) -> None:
    import json

    from benchmark import BENCHMARK_JSON_BLACKLIST, BENCHMARK_JSON_WHITELIST

    corpus_dir = tmp_path / "corpus"
    state_dir = tmp_path / "state"
    corpus_dir.mkdir()
    state_dir.mkdir()

    hotpot_sample = [
        {
            "_id": "hp_leak_test_1",
            "question": "LEAKED_QUESTION_STRING_XYZ?",
            "answer": "LEAKED_ANSWER_STRING_ABC",
            "supporting_facts": [["Montebello", 0], ["Rockland County", 0]],
            "context": [
                [
                    "Montebello",
                    [
                        "Montebello is an incorporated village in the town of Ramapo.",
                        "It is in Rockland County, New York.",
                    ],
                ],
                [
                    "Rockland County",
                    [
                        "Rockland County is the southernmost county on the west side of the Hudson River."
                    ],
                ],
            ],
        }
    ]
    hp_path = corpus_dir / "hotpotqa_sample.json"
    hp_path.write_text(json.dumps(hotpot_sample, indent=2), encoding="utf-8")

    delta = sync_corpus_once(
        corpus_dir,
        state_dir,
        json_keys_whitelist=BENCHMARK_JSON_WHITELIST,
        json_keys_blacklist=BENCHMARK_JSON_BLACKLIST,
    )
    assert len(delta.added_chunks) == 2

    hydrated_texts = []
    for ptr, c_text in sorted(delta.added_chunks, key=lambda item: item[0].start_line):
        assert "LEAKED_QUESTION_STRING_XYZ" not in c_text
        assert "LEAKED_ANSWER_STRING_ABC" not in c_text
        hp = hydrate_chunk_pointer(ptr, corpus_dir)
        assert not hp.stale_on_disk
        assert "LEAKED_QUESTION_STRING_XYZ" not in hp.text
        assert "LEAKED_ANSWER_STRING_ABC" not in hp.text
        hydrated_texts.append(hp.text)

    expected_p1 = (
        "Montebello\n"
        "Montebello is an incorporated village in the town of Ramapo."
        "It is in Rockland County, New York."
    )
    expected_p2 = (
        "Rockland County\n"
        "Rockland County is the southernmost county on the west side of the Hudson River."
    )
    assert hydrated_texts == [expected_p1, expected_p2]
