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
    raw_bytes = sample.encode("utf-8")
    for c in chunks:
        assert c.byte_offset >= 0
        assert c.byte_length > 0
        assert c.line_start >= 1
        assert c.line_end >= c.line_start
        slice_text = raw_bytes[c.byte_offset : c.byte_offset + c.byte_length].decode(
            "utf-8"
        )
        assert slice_text == c.text


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

        db_path = tmp_path / ".cache" / "vectors.db"
        assert db_path.exists()

        with sqlite3.connect(db_path) as conn:
            # 1. Invariant: note_chunks schema & disk-first offsets (NO content column)
            cols = [
                r[1] for r in conn.execute("PRAGMA table_info(note_chunks)").fetchall()
            ]
            assert "content" not in cols, "note_chunks MUST NEVER store raw chunk text"
            for req in ("byte_offset", "byte_length", "line_start", "line_end"):
                assert req in cols, f"note_chunks missing {req}"

            rows = conn.execute(
                "SELECT chunk_id, entity_uri, rel_path, byte_offset, byte_length, line_start, line_end FROM note_chunks"
            ).fetchall()
            assert len(rows) > 0
            for r in rows:
                assert r[3] >= 0
                assert r[4] > 0
                assert r[5] >= 1
                assert r[6] >= r[5]

            # 2. Disk-first hydration verification
            for r in rows:
                _uri, rel, b_off, b_len = r[1], r[2], r[3], r[4]
                text = hydrate_disk_slice(rel, b_off, b_len)
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
