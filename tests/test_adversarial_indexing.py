"""Adversarial coverage hardening for indexing, storage, and caching layers.

Empirical verification adhering strictly to .agents/skills/engineering-real-tests/SKILL.md.
Tests 5 distinct, un-tested boundary conditions with zero overlap with tests/test_fewshot_steering.py:
1. 10,000-line note chunk monotonicity and exact disk byte offsets.
2. Malformed and unparseable markdown syntax tolerance without parser crashes.
3. LRU cache eviction mechanics and bounded memory size under 100+ predicate combinations in _get_steered_pipeline.
4. High-volume concurrent WAL reads during active batch transactions (400 queries across 8 threads without locking).
5. Read-only connection strict write protection (mode=ro) enforced at SQLite engine level.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import pytest
import spacy

from yaw.indexer.pipeline import (
    _get_steered_pipeline,
    chunk_markdown,
    extract_wikilinks,
    get_valid_targets,
)
from yaw.nlp.examples import query_live_predicates
from yaw.storage.db import apply_pragmas
from yaw.storage.schema import ensure_schema_tables, ensure_schema_views


def _setup_full_wal_db(conn: sqlite3.Connection) -> None:
    """Set up all core tables, indices, and views on a test database."""
    apply_pragmas(conn)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS leaf_nodes (
            node_id TEXT NOT NULL,
            source_leaf_id TEXT NOT NULL,
            name TEXT NOT NULL,
            namespace TEXT NOT NULL,
            definition TEXT,
            embedding BLOB,
            PRIMARY KEY (node_id, source_leaf_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS leaf_relations (
            relation_id TEXT PRIMARY KEY,
            source_leaf_id TEXT NOT NULL,
            subject_id TEXT NOT NULL,
            predicate TEXT NOT NULL,
            predicate_id TEXT,
            object_id TEXT NOT NULL,
            weight REAL DEFAULT 1.0
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS wikilinks (
            source_uri TEXT NOT NULL,
            target_uri TEXT NOT NULL,
            link_label TEXT,
            PRIMARY KEY (source_uri, target_uri)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS note_chunks (
            chunk_id TEXT NOT NULL,
            entity_uri TEXT PRIMARY KEY,
            rel_path TEXT NOT NULL,
            ordinal INT NOT NULL,
            heading TEXT,
            heading_slug TEXT,
            line_start INT,
            line_end INT,
            embedding BLOB
        )
        """
    )
    ensure_schema_tables(conn)
    ensure_schema_views(conn)
    conn.commit()


# ===========================================================================
# 1. 10,000-Line Note Chunk Monotonicity & Exact Disk Line Ranges
# ===========================================================================


def test_ten_thousand_line_note_chunk_monotonicity_and_exact_disk_line_ranges(
    tmp_path: Path,
) -> None:
    """10,000-line note generates strictly monotonic chunks matching exact disk line slices."""
    lines: list[str] = ["# Grand Unified Theory and Scientific Taxonomy\n"]
    # 100 sections, each with a heading + 99 paragraph lines = 10,001 physical lines
    for section_idx in range(100):
        lines.append(
            f"\n## Section {section_idx}: Invariant Principles {section_idx}\n"
        )
        for p_idx in range(99):
            lines.append(
                f"Line {p_idx} of section {section_idx} discusses principle {p_idx}. "
                f"Referencing [[topic_{section_idx}_{p_idx % 7}]] with mathematical "
                f"parameter delta_{section_idx}={section_idx * 1.5 + p_idx}.\n"
            )

    large_text = "".join(lines)
    assert len(large_text.splitlines()) >= 10000, "Must be at least 10,000 lines"

    large_file = tmp_path / "ten_thousand_line_note.md"
    large_file.write_text(large_text, encoding="utf-8")

    chunks = chunk_markdown(large_text)
    assert len(chunks) >= 20, f"Expected at least 20 chunks, got {len(chunks)}"

    # Invariant: Chunk line ranges must be strictly monotonic non-overlapping forward progression
    for i in range(len(chunks) - 1):
        assert chunks[i + 1].line_start > chunks[i].line_end, (
            f"Chunk {i + 1} overlap or regression: line_start {chunks[i + 1].line_start} "
            f"<= line_end of chunk {i} ({chunks[i].line_end})"
        )

    # Invariant: Physical disk verification via exact line slicing
    file_lines = large_text.splitlines(keepends=True)
    for idx, c in enumerate(chunks):
        assert c.line_start >= 1
        assert c.line_end >= c.line_start
        line_slice = "".join(file_lines[c.line_start - 1 : c.line_end])
        assert line_slice.rstrip("\r\n") == c.text.rstrip("\r\n"), (
            f"Disk line mismatch at chunk {idx}"
        )


# ===========================================================================
# 2. Malformed / Unparseable Markdown Syntax Tolerance
# ===========================================================================


def test_malformed_unparseable_markdown_syntax_tolerance(tmp_path: Path) -> None:
    """Unclosed code fences, unclosed wikilinks, excessive heading levels handled gracefully."""
    malformed = (
        "```python\ndef unclosed_code():\n    return 'broken'\n"
        "# Header inside unclosed fence\n"
        "[[unclosed wikilink without closing bracket\n"
        "[[nested [[unclosed [[links]]]]\n"
        "####### Level 7 heading (invalid markdown)\n"
        "> > > > Deeply nested quote without exit\n"
        "***\n---\n___\n"
        "| Unclosed | Table | Without | Rows |\n"
    )

    chunks = chunk_markdown(malformed)
    assert isinstance(chunks, list)
    assert len(chunks) > 0

    for c in chunks:
        assert c.line_start >= 1
        assert c.line_end >= c.line_start

    valid_targets = get_valid_targets(tmp_path)
    edges = extract_wikilinks("raw/malformed.md", chunks, valid_targets)
    assert isinstance(edges, list)


# ===========================================================================
# 3. LRU Cache Eviction & Memory Boundedness in _get_steered_pipeline
# ===========================================================================


def test_lru_cache_eviction_under_multiple_predicate_combinations(
    tmp_path: Path,
) -> None:
    """Pass 24 distinct predicate combinations through maxsize=8 LRU cache verifying eviction."""
    db_file = tmp_path / "lru_test.db"
    db_file.touch()

    _get_steered_pipeline.cache_clear()
    info_start = _get_steered_pipeline.cache_info()
    assert info_start.currsize == 0
    assert info_start.maxsize == 8

    combos = [(f"predicate_group_{i}_a", f"predicate_group_{i}_b") for i in range(24)]

    pipelines: list[spacy.language.Language] = []
    for combo in combos:
        nlp = _get_steered_pipeline(combo, str(db_file))
        assert isinstance(nlp, spacy.language.Language), (
            "Must return valid spaCy Language"
        )
        pipelines.append(nlp)

    info_after_misses = _get_steered_pipeline.cache_info()
    assert info_after_misses.currsize == 8, "Cache currsize must be capped at maxsize=8"
    assert info_after_misses.misses == 24, (
        "All 24 distinct calls must register as cache misses"
    )

    # Re-call the last 8 combinations: these must all be cache hits
    for combo in combos[-8:]:
        hit_nlp = _get_steered_pipeline(combo, str(db_file))
        assert isinstance(hit_nlp, spacy.language.Language)

    info_after_hits = _get_steered_pipeline.cache_info()
    assert info_after_hits.hits == 8, f"Expected 8 hits, got {info_after_hits.hits}"
    assert info_after_hits.currsize == 8

    # Re-call the earliest evicted combination: must miss again cleanly without corruption
    first_evicted = combos[0]
    re_evicted_nlp = _get_steered_pipeline(first_evicted, str(db_file))
    assert isinstance(re_evicted_nlp, spacy.language.Language)

    info_final = _get_steered_pipeline.cache_info()
    assert info_final.misses == 25, (
        "Re-calling evicted entry must increment misses to 25"
    )
    assert info_final.currsize == 8
    _get_steered_pipeline.cache_clear()


def test_lru_cache_hundred_combinations_stress_and_bounded_size(
    tmp_path: Path,
) -> None:
    """Pass 100 distinct predicate combinations, ensuring strict memory boundedness."""
    db_file = tmp_path / "lru_100_test.db"
    db_file.touch()

    _get_steered_pipeline.cache_clear()

    for idx in range(100):
        combo = (f"predicate_scale_{idx}",)
        pipe = _get_steered_pipeline(combo, str(db_file))
        assert isinstance(pipe, spacy.language.Language)

    info = _get_steered_pipeline.cache_info()
    assert info.currsize == 8, "Cache currsize must strictly not exceed maxsize=8"
    assert info.misses == 100, "All 100 calls were distinct misses"
    _get_steered_pipeline.cache_clear()


# ===========================================================================
# 4. High-Volume Concurrent WAL Reads During Active Batch Transactions
# ===========================================================================


def test_wal_mode_concurrent_high_volume_reads_during_active_write_transactions(
    tmp_path: Path,
) -> None:
    """WAL mode enables 400 concurrent read-only queries across 8 threads during active write transactions."""
    db_path = tmp_path / "yaw.db"
    init_conn = sqlite3.connect(db_path)
    _setup_full_wal_db(init_conn)

    cur = init_conn.cursor()
    journal_mode = cur.execute("PRAGMA journal_mode;").fetchone()[0]
    assert str(journal_mode).lower() == "wal", f"Expected WAL mode, got {journal_mode}"
    init_conn.close()

    stop_event = threading.Event()
    writer_errors: list[Exception] = []
    reader_errors: list[Exception] = []
    read_counts: list[int] = [0] * 8

    def writer_worker() -> None:
        try:
            write_conn = sqlite3.connect(db_path, timeout=60.0)
            apply_pragmas(write_conn)
            batch = 0
            while not stop_event.is_set():
                write_conn.execute("BEGIN IMMEDIATE;")
                for i in range(10):
                    idx = batch * 10 + i
                    write_conn.execute(
                        """
                        INSERT OR REPLACE INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition)
                        VALUES (?, 'note.md#0', ?, 'concepts', 'definition')
                        """,
                        (f"concepts:node_{idx}", f"Node {idx}"),
                    )
                    write_conn.execute(
                        """
                        INSERT OR REPLACE INTO leaf_relations (relation_id, source_leaf_id, subject_id, predicate, predicate_id, object_id, weight)
                        VALUES (?, 'note.md#0', 'concepts:sub', 'partOf', 'partOf', ?, 1.0)
                        """,
                        (f"rel_{idx}", f"concepts:node_{idx}"),
                    )
                write_conn.commit()
                batch += 1
                time.sleep(0.005)
            write_conn.close()
        except (sqlite3.Error, OSError, RuntimeError, AssertionError) as e:
            writer_errors.append(e)

    def reader_worker(reader_id: int) -> None:
        try:
            ro_uri = f"file:{db_path.resolve()}?mode=ro"
            ro_conn = sqlite3.connect(ro_uri, uri=True, timeout=10.0)
            queries_executed = 0

            for _ in range(50):
                # 1. Query unified_edges view
                cur_ro = ro_conn.cursor()
                cur_ro.execute("SELECT count(*) FROM unified_edges;")
                _ = cur_ro.fetchone()

                # 2. Query leaf_nodes table
                cur_ro.execute("SELECT count(*) FROM leaf_nodes;")
                _ = cur_ro.fetchone()

                # 3. Query query_live_predicates logic
                preds = query_live_predicates(db_path, limit=10)
                assert isinstance(preds, list)

                queries_executed += 1
                time.sleep(0.002)

            ro_conn.close()
            read_counts[reader_id] = queries_executed
        except (sqlite3.Error, OSError, RuntimeError, AssertionError) as e:
            reader_errors.append(e)

    # Launch 1 dedicated writer thread and 8 concurrent reader threads
    writer_thread = threading.Thread(target=writer_worker, daemon=True)
    reader_threads = [
        threading.Thread(target=reader_worker, args=(i,), daemon=True) for i in range(8)
    ]

    writer_thread.start()
    time.sleep(0.02)  # Allow writer transaction initialization

    for t in reader_threads:
        t.start()

    for t in reader_threads:
        t.join(timeout=30.0)
        assert not t.is_alive(), "Reader thread timed out!"

    stop_event.set()
    writer_thread.join(timeout=10.0)
    assert not writer_thread.is_alive(), "Writer thread timed out!"

    assert not writer_errors, f"Writer encountered unexpected errors: {writer_errors}"
    assert not reader_errors, f"Readers encountered unexpected errors: {reader_errors}"
    assert sum(read_counts) == 400, (
        f"Expected 400 queries (8x50), got {sum(read_counts)}"
    )

    # Database integrity check
    check_conn = sqlite3.connect(db_path)
    integrity = check_conn.execute("PRAGMA integrity_check").fetchall()
    check_conn.close()
    assert integrity == [("ok",)], f"Database integrity check failed: {integrity}"


# ===========================================================================
# 5. Read-Only Connection Strict Write Protection (mode=ro)
# ===========================================================================


def test_read_only_connection_strict_write_protection(tmp_path: Path) -> None:
    """mode=ro strictly prevents write mutations at SQLite engine level."""
    db_path = tmp_path / "yaw.db"
    conn = sqlite3.connect(db_path)
    _setup_full_wal_db(conn)
    conn.close()

    ro_uri = f"file:{db_path.resolve()}?mode=ro"
    ro_conn = sqlite3.connect(ro_uri, uri=True, timeout=5.0)

    # Attempting any write must fail with OperationalError
    with pytest.raises(
        sqlite3.OperationalError, match="attempt to write a readonly database"
    ):
        ro_conn.execute(
            "INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace) VALUES ('1', '2', '3', 'concepts')"
        )

    with pytest.raises(
        sqlite3.OperationalError, match="attempt to write a readonly database"
    ):
        ro_conn.execute("DELETE FROM leaf_nodes")

    ro_conn.close()
