"""Unit and integration tests for clean-slate cocoindexer.py invariants."""

from __future__ import annotations

import sqlite3

from yaw import config
from yaw.indexer import (
    chunk_markdown,
    sync_vault,
)
from yaw.server import hydrate_disk_slice
from yaw.storage import search_fts


def test_disk_first_chunking_offsets_and_lines():
    sample = "# Main Title\n\nParagraph 1.\n\n## Subheading\n\nLine 1\nLine 2\n"
    chunks = chunk_markdown(sample)
    assert len(chunks) >= 1
    lines = sample.splitlines(keepends=True)
    for c in chunks:
        assert c.line_start >= 1
        assert c.line_end >= c.line_start
        slice_text = "".join(lines[c.line_start - 1 : c.line_end])
        assert slice_text.rstrip("\r\n") == c.text.rstrip("\r\n")


def test_cocoindexer_invariants_disk_first_and_contentless_fts(tmp_path):
    old_root = config.VAULT_ROOT
    try:
        config.set_vault(tmp_path)
        (tmp_path / "raw").mkdir()
        (tmp_path / "raw" / "rules.md").write_text(
            "# Quantum Mechanics\n\nFormulated by Werner Heisenberg and Erwin Schrodinger.",
            encoding="utf-8",
        )

        sync_vault(vault_dir=tmp_path, force=True)

        db_path = tmp_path / ".cache" / "yaw.db"
        assert db_path.exists()

        with sqlite3.connect(db_path) as conn:
            # 1. Invariant: note_chunks schema & disk-first offsets (NO content column)
            cols = [
                r[1] for r in conn.execute("PRAGMA table_info(note_chunks)").fetchall()
            ]
            assert "content" not in cols, "note_chunks MUST NEVER store raw chunk text"
            assert "byte_offset" not in cols, "note_chunks MUST NOT store byte_offset"
            assert "byte_length" not in cols, "note_chunks MUST NOT store byte_length"
            for req in ("line_start", "line_end"):
                assert req in cols, f"note_chunks missing {req}"

            rows = conn.execute(
                "SELECT chunk_id, entity_uri, rel_path, line_start, line_end FROM note_chunks"
            ).fetchall()
            assert len(rows) > 0
            for r in rows:
                assert r[3] >= 1
                assert r[4] >= r[3]

            # 2. Disk-first hydration verification
            for r in rows:
                _uri, rel, l_start, l_end = r[1], r[2], r[3], r[4]
                text = hydrate_disk_slice(rel, l_start, l_end)
                assert "Heisenberg" in text

            # 3. Invariant: contentless query_fts
            sql = conn.execute(
                "SELECT sql FROM sqlite_master WHERE name='query_fts'"
            ).fetchone()[0]
            assert "content=''" in sql or 'content=""' in sql, (
                f"query_fts must be contentless, got: {sql}"
            )

            # 4. Contentless FTS search verification across chunks and taxonomy
            hits = search_fts("Heisenberg", conn=conn)
            assert len(hits) > 0
            hit_uris = [h[0] for h in hits]
            assert "raw/rules.md#chunk:0" in hit_uris
            assert (
                "entities:werner-heisenberg" in hit_uris
                or "concepts:werner-heisenberg" in hit_uris
            )
            assert hits[0][1] > 0.0

            # 5. Invariant: note_meta populated
            meta_rows = conn.execute(
                "SELECT rel_path, title, category, modified_at FROM note_meta"
            ).fetchall()
            assert len(meta_rows) > 0
            assert meta_rows[0][0] == "raw/rules.md"
            assert "Quantum" in meta_rows[0][1]
            assert meta_rows[0][3] > 0.0

            # 6. Invariant: unified_edges and unified_nodes views operational
            edges = conn.execute("SELECT * FROM unified_edges").fetchall()
            assert isinstance(edges, list)
            nodes = conn.execute("SELECT * FROM unified_nodes").fetchall()
            assert len(nodes) > 0
    finally:
        config.set_vault(old_root)
