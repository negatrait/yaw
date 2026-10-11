"""Tests for storage layer (db.py) and SQLite FTS5 / schema invariants."""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from pathlib import Path

import pytest

from yaw import config
from yaw.storage import db
from yaw.storage.db import (
    _clean_fts_query,
    apply_pragmas,
    ensure_schema_tables,
    ensure_schema_views,
    get_db,
    insert_alias,
    search_fts,
)


def test_storage_layer_spacy_decoupling() -> None:
    """Verify that storage/db.py does not define module-level _nlp spacy object."""
    assert not hasattr(db, "_nlp"), (
        "storage.db must not maintain a global _nlp instance"
    )


def test_clean_fts_query_tokenization() -> None:
    """Verify regex tokenization in _clean_fts_query extracts lowercase alphanumeric terms."""
    assert _clean_fts_query("Quantum Mechanics") == ["quantum", "mechanics"]
    assert _clean_fts_query("What is a 'qubit'?!") == ["what", "is", "a", "qubit"]
    assert _clean_fts_query("!@#$%^&*()+=-") == []
    assert _clean_fts_query("") == []
    assert _clean_fts_query("   \t\n  ") == []
    assert _clean_fts_query("Qwen3-4B Instruct v1.2") == [
        "qwen3",
        "4b",
        "instruct",
        "v1",
        "2",
    ]


def test_clean_fts_query_stopwords() -> None:
    """Verify _clean_fts_query strips stopwords when requested."""
    assert _clean_fts_query("what is a qubit", strip_stopwords=True) == ["qubit"]
    assert _clean_fts_query("what is a qubit", strip_stopwords=False) == [
        "what",
        "is",
        "a",
        "qubit",
    ]
    # All stopwords query falls back to raw terms
    assert _clean_fts_query("and or not", strip_stopwords=True) == ["and", "or", "not"]


def test_clean_fts_query_stress() -> None:
    """Verify _clean_fts_query handles unicode scripts, SQLi payloads, keywords, and long strings."""
    # Unicode scripts
    assert _clean_fts_query("量子 计算 与 算法") == ["量子", "计算", "与", "算法"]
    assert _clean_fts_query("Квантовая Механика и Физика") == [
        "квантовая",
        "механика",
        "и",
        "физика",
    ]
    assert _clean_fts_query("Café résumé naïve façade") == [
        "café",
        "résumé",
        "naïve",
        "façade",
    ]
    assert _clean_fts_query("🔬 Quantum ⚛️ Computing 💻") == ["quantum", "computing"]
    assert _clean_fts_query("Zero\u200bWidth\u200cJoiner") == [
        "zero",
        "width",
        "joiner",
    ]

    # SQL injection payloads
    payload = "OR 1=1; DROP TABLE query_fts; --"
    assert _clean_fts_query(payload) == [
        "or",
        "1",
        "1",
        "drop",
        "table",
        "query_fts",
    ]

    # FTS5 reserved keywords
    assert _clean_fts_query("MATCH AND OR NOT NEAR col:value * ^") == [
        "match",
        "and",
        "or",
        "not",
        "near",
        "col",
        "value",
    ]

    # Underscores, control chars, and long strings
    assert _clean_fts_query("foo_bar") == ["foo_bar"]
    assert _clean_fts_query("hello\x00world\x07\x08 red") == ["hello", "world", "red"]
    long_query = "word " * 1_000
    assert len(_clean_fts_query(long_query)) == 1_000


@pytest.fixture
def fts_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[sqlite3.Connection]:
    """Initialize a real isolated SQLite database with query_fts and standard schema."""
    db_file = tmp_path / "storage_fts.db"
    monkeypatch.setattr(config, "DB_PATH", db_file)
    conn = get_db(load_vec=False)
    apply_pragmas(conn)
    ensure_schema_tables(conn)
    ensure_schema_views(conn)

    corpus = [
        (
            "raw/quantum.md#0",
            "raw/quantum.md",
            "Quantum Superposition",
            "Quantum superposition is a fundamental principle of quantum mechanics.",
        ),
        (
            "raw/cyrillic.md#0",
            "raw/cyrillic.md",
            "Квантовая Механика",
            "Квантовая механика изучает микромир и квантовые состояния частиц.",
        ),
        (
            "raw/cjk.md#0",
            "raw/cjk.md",
            "量子计算",
            "量子计算 利用 量子力学 的 特性 来 进行 信息 处理 和 算法 执行。",
        ),
        (
            "raw/accents.md#0",
            "raw/accents.md",
            "Le Café Français",
            "Un café et un résumé naïf de la physique moderne.",
        ),
        (
            "raw/code.md#0",
            "raw/code.md",
            "Code Identifier",
            "The variable foo_bar and function get_model_name are defined.",
        ),
    ]
    conn.executemany(
        "INSERT INTO query_fts(entity_uri, rel_path, heading, content) VALUES (?, ?, ?, ?)",
        corpus,
    )
    conn.commit()
    try:
        yield conn
    finally:
        conn.close()


def test_fts5_sql_injection_defense(fts_db: sqlite3.Connection) -> None:
    """Empirically verify SQL injection attempts do not drop tables or execute arbitrary SQL."""
    results = search_fts("OR 1=1; DROP TABLE query_fts; --", conn=fts_db)
    table_check = fts_db.execute(
        "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='query_fts'"
    ).fetchone()[0]
    assert table_check == 1, "query_fts table must not be dropped by SQL injection"
    assert isinstance(results, list)

    results_union = search_fts("' UNION SELECT null, null, null, null --", conn=fts_db)
    assert isinstance(results_union, list)

    results_stacked = search_fts("quantum; SELECT 1; /* comment */", conn=fts_db)
    assert len(results_stacked) >= 1
    assert results_stacked[0][0] == "raw/quantum.md#0"


def test_fts5_reserved_keywords_no_syntax_error(fts_db: sqlite3.Connection) -> None:
    """Empirically verify FTS5 reserved operators do not trigger fts5 syntax errors."""
    assert isinstance(search_fts("AND OR NOT NEAR", conn=fts_db), list)
    assert isinstance(search_fts("OR OR OR OR", conn=fts_db), list)
    assert isinstance(search_fts("NEAR/5 MATCH col:*", conn=fts_db), list)


def test_fts5_multilingual_matching(fts_db: sqlite3.Connection) -> None:
    """Verify FTS5 search across Cyrillic, Accents, and CJK content."""
    res_cyr = search_fts("квантовая механика", conn=fts_db)
    assert len(res_cyr) == 1
    assert res_cyr[0][0] == "raw/cyrillic.md#0"
    assert 0.0 <= res_cyr[0][1] <= 1.0

    res_acc = search_fts("café résumé", conn=fts_db)
    assert len(res_acc) == 1
    assert res_acc[0][0] == "raw/accents.md#0"

    res_cjk = search_fts("量子计算 算法", conn=fts_db)
    assert len(res_cjk) == 1
    assert res_cjk[0][0] == "raw/cjk.md#0"


def test_fts5_ranking_and_contentless_invariants(fts_db: sqlite3.Connection) -> None:
    """Verify BM25 normalization, contentless empty returns, and edge boundaries."""
    results = search_fts("quantum superposition principle", top_n=5, conn=fts_db)
    assert len(results) == 1
    uri, score, heading, content = results[0]
    assert uri == "raw/quantum.md#0"
    assert 0.0 <= score <= 1.0
    assert score == 1.0
    assert heading == ""
    assert content == ""

    # top_n boundaries
    assert search_fts("quantum", top_n=0, conn=fts_db) == []
    assert len(search_fts("quantum", top_n=1, conn=fts_db)) == 1

    # Long query stress
    tokens = [f"word{i}" for i in range(100)] + ["superposition"]
    res_long = search_fts(" ".join(tokens), conn=fts_db)
    assert len(res_long) >= 1
    assert res_long[0][0] == "raw/quantum.md#0"


def test_search_fts_custom_terms(fts_db: sqlite3.Connection) -> None:
    """Verify search_fts honors custom clean terms passed by caller."""
    hits = search_fts("any raw query text", conn=fts_db, terms=["superposition"])
    assert len(hits) == 1
    assert hits[0][0] == "raw/quantum.md#0"

    misses = search_fts("any raw query text", conn=fts_db, terms=["nonexistent"])
    assert misses == []


def test_search_fts_connection_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify search_fts automatically opens and closes DB connection when conn=None."""
    db_file = tmp_path / "lifecycle_fts.db"
    monkeypatch.setattr(config, "DB_PATH", db_file)
    conn = get_db(load_vec=False)
    apply_pragmas(conn)
    ensure_schema_tables(conn)
    ensure_schema_views(conn)
    conn.execute(
        "INSERT INTO query_fts(entity_uri, rel_path, heading, content) VALUES (?, ?, ?, ?)",
        ("note#0", "note.md", "H", "lifecycle content"),
    )
    conn.commit()
    conn.close()

    results = search_fts("lifecycle content", conn=None)
    assert len(results) == 1
    assert results[0][0] == "note#0"


def test_schema_integrity_and_pragmas(tmp_path: Path) -> None:
    """Verify SQLite PRAGMAs and idempotent schema table and view creation."""
    db_file = tmp_path / "test_schema.db"
    conn = sqlite3.connect(str(db_file))
    try:
        apply_pragmas(conn)
        ensure_schema_tables(conn)
        ensure_schema_views(conn)

        mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        assert mode.upper() == config.JOURNAL_MODE.upper()
        fk = conn.execute("PRAGMA foreign_keys;").fetchone()[0]
        assert fk == 1

        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "aliases" in tables
        assert "canon_verdicts" in tables

        # Re-running must be idempotent
        ensure_schema_tables(conn)
        ensure_schema_views(conn)
    finally:
        conn.close()


def test_aliases_method_check_constraint(tmp_path: Path) -> None:
    """WP 5.1.3: Verify aliases table accepts all canonical MatchMethod values and rejects purged methods."""
    db_file = tmp_path / "test_aliases.db"
    conn = sqlite3.connect(str(db_file))
    try:
        ensure_schema_tables(conn)
        valid_methods = ("exact", "alias", "auto", "new", "ontology")
        for m in valid_methods:
            insert_alias(conn, f"alias_{m}", "entity", "concepts:test", method=m)
        count = conn.execute("SELECT count(*) FROM aliases").fetchone()[0]
        assert count == len(valid_methods)

        # Assert purged legacy methods fail immediately
        import pytest

        for m in ("lexical", "llm", "compile"):
            with pytest.raises(ValueError, match="Invalid alias method"):
                insert_alias(
                    conn, f"alias_bad_{m}", "entity", "concepts:test", method=m
                )
    finally:
        conn.close()


def test_compiled_tables_and_unified_views(tmp_path: Path) -> None:
    """WP 5.1.5: Verify compiled_nodes and compiled_relations persist and project cleanly into unified views."""
    db_file = tmp_path / "test_views.db"
    conn = sqlite3.connect(str(db_file))
    try:
        ensure_schema_tables(conn)
        conn.execute(
            "CREATE TABLE note_chunks (entity_uri TEXT PRIMARY KEY, heading TEXT, rel_path TEXT)"
        )
        conn.execute(
            "CREATE TABLE leaf_nodes (node_id TEXT, source_leaf_id TEXT, name TEXT, namespace TEXT, definition TEXT, PRIMARY KEY(node_id, source_leaf_id))"
        )
        conn.execute(
            "CREATE TABLE wikilinks (source_uri TEXT, target_uri TEXT, link_label TEXT, PRIMARY KEY(source_uri, target_uri))"
        )
        conn.execute(
            "CREATE TABLE leaf_relations (relation_id TEXT PRIMARY KEY, source_leaf_id TEXT, subject_id TEXT, predicate TEXT, object_id TEXT, weight REAL)"
        )
        ensure_schema_views(conn)

        conn.execute(
            "INSERT INTO compiled_nodes (node_id, name, namespace, definition) VALUES ('concepts:meso', 'Meso Concept', 'concepts', 'Compiled superclass')"
        )
        conn.execute(
            "INSERT INTO compiled_relations (relation_id, subject_id, predicate, object_id, weight) VALUES ('cr1', 'concepts:leaf', 'subClassOf', 'concepts:meso', 1.0)"
        )

        n_row = conn.execute(
            "SELECT node_id, name, namespace FROM unified_nodes WHERE node_id = 'concepts:meso'"
        ).fetchone()
        assert n_row == ("concepts:meso", "Meso Concept", "concepts")

        e_row = conn.execute(
            "SELECT src_id, dst_id, rel_type FROM unified_edges WHERE src_id = 'concepts:leaf'"
        ).fetchone()
        assert e_row == ("concepts:leaf", "concepts:meso", "subClassOf")
    finally:
        conn.close()


def test_aliases_idempotent_migration_purges_obsolete_methods(
    tmp_path: Path,
) -> None:
    """Verify ensure_schema_tables idempotently migrates legacy aliases table containing obsolete methods."""
    db_file = tmp_path / "legacy_migration.db"
    conn = sqlite3.connect(str(db_file))
    try:
        # Create legacy aliases table with obsolete methods allowed in CHECK constraint
        conn.execute(
            """
            CREATE TABLE aliases (
                alias_id TEXT NOT NULL,
                kind TEXT NOT NULL CHECK(kind IN ('entity', 'predicate')),
                canonical_id TEXT NOT NULL,
                method TEXT NOT NULL CHECK(method IN ('lexical', 'auto', 'llm', 'compile', 'exact', 'alias', 'new')),
                score REAL NOT NULL DEFAULT 1.0 CHECK(score >= 0.0 AND score <= 1.0),
                status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'contested')),
                source_leaf_id TEXT,
                resolved_at REAL DEFAULT (unixepoch()),
                evidence TEXT,
                PRIMARY KEY (alias_id, kind)
            )
            """
        )
        conn.execute(
            "INSERT INTO aliases (alias_id, kind, canonical_id, method, score) VALUES ('a_lex', 'entity', 'c1', 'lexical', 0.9)"
        )
        conn.execute(
            "INSERT INTO aliases (alias_id, kind, canonical_id, method, score) VALUES ('a_llm', 'entity', 'c2', 'llm', 0.8)"
        )
        conn.execute(
            "INSERT INTO aliases (alias_id, kind, canonical_id, method, score) VALUES ('a_comp', 'entity', 'c3', 'compile', 0.7)"
        )
        conn.execute(
            "INSERT INTO aliases (alias_id, kind, canonical_id, method, score) VALUES ('a_exact', 'entity', 'c4', 'exact', 1.0)"
        )
        conn.commit()

        # Run ensure_schema_tables migration
        ensure_schema_tables(conn)

        # Verify schema table SQL no longer contains obsolete methods
        sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='aliases'"
        ).fetchone()[0]
        assert "'lexical'" not in sql
        assert "'llm'" not in sql
        assert "'compile'" not in sql
        assert "'ontology'" in sql

        # Verify rows were safely migrated without data loss
        rows = dict(
            conn.execute(
                "SELECT alias_id, (method || ':' || printf('%.1f', score) || ':' || canonical_id) FROM aliases"
            ).fetchall()
        )
        assert rows["a_lex"] == "exact:0.9:c1"
        assert rows["a_llm"] == "auto:0.8:c2"
        assert rows["a_comp"] == "auto:0.7:c3"
        assert rows["a_exact"] == "exact:1.0:c4"

        # Re-run migration: must be strictly idempotent
        ensure_schema_tables(conn)
        count = conn.execute("SELECT count(*) FROM aliases").fetchone()[0]
        assert count == 4
    finally:
        conn.close()
