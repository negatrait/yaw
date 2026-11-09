"""Unmocked integration tests for native CocoIndex v1 incremental line-pointer ingestion."""

from pathlib import Path

from yaw.daemon.graph.storage import hydrate_chunk_pointer
from yaw.daemon.ingestion import (
    chunk_source_text,
    extract_ast_headings,
    resolve_heading_for_line,
)


def test_chunk_markdown_text_invariants() -> None:
    content = """# System Architecture

This is the introductory overview of the distributed system.
It covers several primary components.

## Data Layer

The storage engine uses SQLite with WAL mode.
It maintains high read concurrency and ACID guarantees.
"""
    chunks = chunk_source_text(content, "docs/architecture.md", chunk_size=300, chunk_overlap=30)
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
    assert chunk_source_text("", "empty.md") == ()
    assert chunk_source_text("   \n\n  ", "spaces.md") == ()

    # Document without any headings falls back to stem
    plain = "Just a single paragraph without any markdown headings."
    chunks = chunk_source_text(plain, "raw/unstructured_notes.txt")
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
    headings = extract_ast_headings(doc, language="markdown")
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


def test_parser_helper_edit_invalidates_persistent_memo_across_processes(tmp_path: Path) -> None:
    import json
    import os
    import shutil
    import subprocess
    import sys

    snapshot = tmp_path / "implementation"
    shutil.copytree(
        Path(__file__).resolve().parents[3] / "yaw",
        snapshot / "yaw",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "note.md").write_text("# Original\nA passage about graph indexing.\n", encoding="utf-8")
    worker = """
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import cocoindex as coco
from cocoindex.connectors import localfs, sqlite
from yaw.daemon.ingestion.source_targets import (
    SOURCE_DB,
    SOURCE_ROOT,
    SourceMetadata,
    SourceSettings,
    build_source_path_matcher,
    declare_source_file,
)

@coco.fn
async def app_main(settings: SourceSettings) -> None:
    table = await sqlite.mount_table_target(
        SOURCE_DB,
        table_name="source_chunks",
        table_schema=await sqlite.TableSchema[SourceMetadata].from_class(SourceMetadata, primary_key=["chunk_id"]),
    )
    files = localfs.walk_dir(
        SOURCE_ROOT,
        recursive=True,
        path_matcher=build_source_path_matcher(settings),
        rescan_interval=None,
    )
    _ = [await declare_source_file(file, table, settings) async for _, file in files.items()]

db = sqlite.connect(Path(sys.argv[3]) / "source.db", load_vec=False)
ctx = coco.ContextProvider()
ctx.provide(SOURCE_DB, db)
ctx.provide(SOURCE_ROOT, Path(sys.argv[2]))
env = coco.Environment(settings=coco.Settings.from_env(Path(sys.argv[3]) / "state"), context_provider=ctx)
app = coco.App(
    coco.AppConfig(name="memo_test", environment=env, max_inflight_components=4),
    app_main,
    SourceSettings(("**/*.md",), (), min_chunk_size=1),
)
try:
    app.update_blocking()
    with db.readonly() as conn:
        rows = conn.execute("SELECT chunk_id, title FROM source_chunks").fetchall()
    print(json.dumps([[r[0], r[1]] for r in rows]))
finally:
    env.close()
    db.close()
"""

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    worker_file = tmp_path / "worker.py"
    worker_file.write_text(worker, encoding="utf-8")

    def scan() -> list[list[str]]:
        result = subprocess.run(
            [sys.executable, str(worker_file), str(snapshot), str(corpus), str(state_dir)],
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout.strip().splitlines()[-1])

    rows1 = scan()
    assert len(rows1) == 1
    assert rows1[0][0].startswith("chunk-")
    assert rows1[0][1] == "Original"

    module = snapshot / "yaw/daemon/ingestion/corpus.py"
    before = module.read_text(encoding="utf-8")
    after = before.replace(
        "return heading_title if heading_title else fallback_title",
        'return "Changed by parser helper"',
    )
    assert before != after
    module.write_text(after, encoding="utf-8")
    rows2 = scan()
    assert len(rows2) == 1
    assert rows2[0][0].startswith("chunk-")
    assert rows2[0][1] == "Changed by parser helper"

    # Shared rendering/identity contracts are dependencies too: a helper in
    # yaw.core must invalidate file memos even when corpus.py is unchanged.
    module = snapshot / "yaw/core/types.py"
    before = module.read_text(encoding="utf-8")
    after = before.replace('prefix="chunk-"', 'prefix="updated-chunk-"')
    assert before != after
    module.write_text(after, encoding="utf-8")
    rows3 = scan()
    assert len(rows3) == 1
    assert rows3[0][0].startswith("updated-chunk-")
    assert rows3[0][1] == "Changed by parser helper"


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


def test_strict_json_projection_excludes_secrets_during_real_hydration(tmp_path):
    import json

    source = tmp_path / "record.json"
    source.write_text(
        json.dumps(
            {
                "title": "Service",
                "credentials": {"password": "SECRET"},
                "endpoint": "https://example.test",
                "retries": 3,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    chunks = chunk_source_text(
        source.read_text(encoding="utf-8"),
        source.name,
        json_keys_blacklist=("credentials",),
        min_chunk_size=1,
    )
    assert chunks
    hydrated = [hydrate_chunk_pointer(ptr, tmp_path).text for ptr, _ in chunks]
    assert any("https://example.test" in text for text in hydrated)
    assert all("SECRET" not in text and "password" not in text for text in hydrated)


def test_native_scopes_preserve_unindented_brace_style():
    from yaw.daemon.ingestion import extract_ast_headings

    code = "int add(int a, int b)\n{ return a + b; }\nint main() {\nreturn add(1, 2);\n}\n"
    headings = extract_ast_headings(code, "c")
    assert any("add(" in title for _, title in headings)
    assert any("main(" in title for _, title in headings)


def test_markdown_fences_require_matching_marker_and_closing_length():
    source = "````md\n# Hidden\n~~~\n# Also hidden\n```\n# Still hidden\n````\n# Visible\n"
    assert extract_ast_headings(source, language="markdown") == [(8, "Visible")]


def test_json_projection_rejects_unsafe_compact_spans_and_malformed_input(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="unsafe JSON projection"):
        chunk_source_text(
            '{"title":"Service","password":"SECRET","text":"public"}',
            "unsafe.json",
            json_keys_blacklist=("password",),
        )
    with pytest.raises(ValueError):
        chunk_source_text('{"title": malformed}', "invalid.json")
    assert chunk_source_text("[]", "empty.json") == ()


def test_jsonl_and_escaped_multiline_body_keep_original_source_coordinates(tmp_path):
    import json

    source = tmp_path / "records.jsonl"
    source.write_text(
        json.dumps({"title": "First", "text": "line\n" * 500})
        + "\n\n"
        + json.dumps({"title": "Second", "text": "äö \U0001f600"})
        + "\n",
        encoding="utf-8",
    )
    chunks = chunk_source_text(source.read_text(encoding="utf-8"), source.name, chunk_size=200, min_chunk_size=1)
    assert len(chunks) > 2
    for ptr, _ in chunks:
        assert (ptr.start_line, ptr.end_line) == ((1, 1) if ptr.title == "First" else (3, 3))
        assert ptr.ast_island_start == ptr.start_line and ptr.ast_island_end == ptr.end_line
        assert not hydrate_chunk_pointer(ptr, tmp_path).stale_on_disk


def test_nested_projection_preserves_object_identity_or_rejects_unsafe_partial_object(tmp_path):
    import json

    import pytest

    source = tmp_path / "nested.json"
    source.write_text(json.dumps({"profile": {"name": "Ada"}, "password": "SECRET"}, indent=2), encoding="utf-8")
    chunks = chunk_source_text(
        source.read_text(encoding="utf-8"), source.name, json_keys_blacklist=("password",), min_chunk_size=1
    )
    assert chunks
    text = hydrate_chunk_pointer(chunks[0][0], tmp_path).text
    assert "profile" in text and "Ada" in text and "SECRET" not in text
    source.write_text(json.dumps({"profile": {"name": "Ada", "password": "SECRET"}}, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="enclosing object"):
        chunk_source_text(source.read_text(encoding="utf-8"), source.name, json_keys_blacklist=("password",))


def test_compact_multi_record_array_cannot_masquerade_as_individual_source_spans():
    import pytest

    with pytest.raises(ValueError, match="unsafe JSON projection/source span"):
        chunk_source_text('[{"title":"A","text":"Body"},{"title":"B","text":"Other"}]', "records.json")


def test_whitelist_excludes_empty_objects_without_leaking_their_keys():
    assert chunk_source_text('{"secret":{}}', "record.json", json_keys_whitelist=("text",)) == ()
