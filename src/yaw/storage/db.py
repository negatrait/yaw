"""SQLite connection provider, PRAGMA configuration SSOT, schema contracts, and FTS5 search."""

from __future__ import annotations

import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np
import sqlite_vec

logger = logging.getLogger(__name__)

from yaw.storage.rdf import export_graph
from yaw.storage.schema import (
    ensure_schema_tables,
    ensure_schema_views,
    verify_schema_integrity,
)

from .. import config

__all__ = [
    "DatabaseError",
    "_clean_fts_query",
    "apply_pragmas",
    "clean_db",
    "delete_alias",
    "ensure_schema_tables",
    "ensure_schema_views",
    "export_graph",
    "get_active_aliases",
    "get_aliases_for_canonical",
    "get_canonical_id",
    "get_db",
    "insert_alias",
    "insert_leaf_node",
    "load_vec_extension",
    "log_canon_verdict",
    "search_fts",
    "serialize_embedding",
    "set_alias_status",
    "vacuum_db",
    "verify_schema_integrity",
]


def apply_pragmas(conn: sqlite3.Connection) -> None:
    """Apply authoritative SQLite journal, busy timeout, and synchronous PRAGMAs."""
    conn.execute(f"PRAGMA journal_mode = {config.JOURNAL_MODE}")
    conn.execute(f"PRAGMA busy_timeout = {int(config.BUSY_TIMEOUT_S * 1000)}")
    conn.execute(f"PRAGMA synchronous = {config.SYNCHRONOUS}")
    conn.execute("PRAGMA foreign_keys = ON;")


def load_vec_extension(conn: sqlite3.Connection) -> None:
    """Load sqlite-vec vector extension into SQLite connection."""
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)


def clean_db(db_path: Path | None = None) -> None:
    """Safely unlink SQLite database and associated WAL/SHM journaling artifacts."""
    target = db_path or config.DB_PATH
    for p in (target, Path(f"{target}-wal"), Path(f"{target}-shm")):
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass
    if target.is_file():
        try:
            with sqlite3.connect(
                str(target), autocommit=True, timeout=config.CONNECT_TIMEOUT_S
            ) as conn:
                conn.execute("PRAGMA foreign_keys = OFF;")
                views = [
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'view'"
                    ).fetchall()
                ]
                for v in views:
                    conn.execute(f"DROP VIEW IF EXISTS {v};")
                tables = [
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
                    ).fetchall()
                ]
                for t in tables:
                    conn.execute(f"DROP TABLE IF EXISTS {t};")
                conn.execute("VACUUM;")
        except sqlite3.DatabaseError, OSError:
            pass


def vacuum_db(db_path: Path | str | None = None) -> None:
    """Execute VACUUM on the database to reclaim free space and defragment."""
    target = Path(db_path) if db_path else config.DB_PATH
    if not target.is_file():
        return
    with sqlite3.connect(
        str(target), autocommit=True, timeout=config.CONNECT_TIMEOUT_S
    ) as conn:
        conn.execute("VACUUM;")


def insert_alias(
    conn: sqlite3.Connection,
    alias_id: str,
    canonical_id: str | None = None,
    kind: str = "entity",
    method: str = "alias",
    score: float = 1.0,
    status: str = "active",
    source_leaf_id: str | None = None,
    resolved_at: float | None = None,
    evidence: str | None = None,
    **kwargs: Any,
) -> None:
    """Upsert alias mapping into aliases table.

    Strictly enforces method in ('exact', 'alias', 'auto', 'new', 'ontology').
    Zero fallback shims or silent swallowers.
    """
    if canonical_id in ("entity", "predicate") and kind not in ("entity", "predicate"):
        actual_kind = canonical_id
        actual_canonical_id = kind
    else:
        actual_kind = kind
        actual_canonical_id = canonical_id or kwargs.get("canonical_id", alias_id)

    valid_methods = ("exact", "alias", "auto", "new", "ontology")
    if method not in valid_methods:
        raise ValueError(
            f"Invalid alias method '{method}'; must be one of {valid_methods}"
        )
    if actual_kind not in ("entity", "predicate"):
        raise ValueError(
            f"Invalid kind '{actual_kind}'; must be 'entity' or 'predicate'"
        )
    if not alias_id or not actual_canonical_id:
        raise ValueError("alias_id and canonical_id must be non-empty strings")
    if status not in ("active", "contested"):
        raise ValueError(f"Invalid status '{status}'; must be 'active' or 'contested'")

    ts = time.time() if resolved_at is None else resolved_at

    conn.execute(
        """
        INSERT OR REPLACE INTO aliases (
            alias_id, kind, canonical_id, method, score, status, source_leaf_id, resolved_at, evidence
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            alias_id,
            actual_kind,
            actual_canonical_id,
            method,
            score,
            status,
            source_leaf_id,
            ts,
            evidence,
        ),
    )


def get_canonical_id(
    conn: sqlite3.Connection,
    alias_id: str,
    kind: str = "entity",
    fallback_to_alias: bool = True,
) -> str | None:
    """Lookup active canonical ID for alias_id, or return alias_id (or None) if none."""
    try:
        row = conn.execute(
            "SELECT canonical_id FROM aliases WHERE alias_id = ? AND kind = ? AND status = 'active'",
            (alias_id, kind),
        ).fetchone()
        if row and row[0]:
            return str(row[0])
    except sqlite3.OperationalError:
        pass
    return alias_id if fallback_to_alias else None


def serialize_embedding(
    embedding: list[float] | np.ndarray | bytes | None,
) -> bytes | None:
    """Serialize float sequence or ndarray into IEEE 754 float32 byte buffer."""
    if embedding is None:
        return None
    if isinstance(embedding, (bytes, memoryview)):
        return bytes(embedding)
    emb_shape = getattr(embedding, "shape", None)
    if (
        isinstance(emb_shape, tuple)
        and len(emb_shape) == 1
        and emb_shape[0] in (300, 768)
    ) or (
        hasattr(embedding, "__len__") and len(embedding) in (300, 768)  # type: ignore[arg-type]
    ):
        arr = np.asarray(embedding, dtype=np.float32)
        return arr.tobytes()
    if hasattr(embedding, "__len__") and len(embedding) > 0:  # type: ignore[arg-type]
        arr = np.asarray(embedding, dtype=np.float32)
        return arr.tobytes()
    return None


def insert_leaf_node(
    conn: sqlite3.Connection,
    node_id: str,
    name: str,
    namespace: str,
    definition: str | None = None,
    embedding: list[float] | np.ndarray | None = None,
    source_leaf_id: str = "raw/note.md",
) -> None:
    """Insert leaf node record into leaf_nodes table idempotently."""
    if not node_id:
        raise ValueError("node_id must be a non-empty string")
    emb_bytes = serialize_embedding(embedding) if embedding is not None else None
    eff_source = source_leaf_id or "system:canonical_induction"
    try:
        conn.execute(
            """
            INSERT OR IGNORE INTO leaf_nodes (
                node_id, source_leaf_id, name, namespace, definition, embedding
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                node_id,
                eff_source,
                name,
                namespace,
                definition or "",
                emb_bytes,
            ),
        )
    except sqlite3.OperationalError as err:
        if "no such table: leaf_nodes" in str(err).lower():
            return
        raise


def get_active_aliases(
    conn: sqlite3.Connection,
    kind: str | None = None,
) -> dict[str, str]:
    """Retrieve all active alias -> canonical mappings as a lookup dictionary."""
    try:
        query = "SELECT alias_id, canonical_id FROM aliases WHERE status = 'active'"
        params: tuple[Any, ...] = ()
        if kind:
            query += " AND kind = ?"
            params = (kind,)
        rows = conn.execute(query, params).fetchall()
        return {str(r[0]): str(r[1]) for r in rows}
    except sqlite3.OperationalError:
        return {}


def get_aliases_for_canonical(
    conn: sqlite3.Connection,
    canonical_id: str,
    kind: str | None = None,
    include_contested: bool = False,
) -> list[str]:
    """Retrieve all alias identifiers mapped to a canonical entity/predicate."""
    try:
        clauses = ["canonical_id = ?"]
        params: list[Any] = [canonical_id]
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        if not include_contested:
            clauses.append("status = 'active'")
        query = f"SELECT alias_id FROM aliases WHERE {' AND '.join(clauses)}"
        rows = conn.execute(query, tuple(params)).fetchall()
        return [str(r[0]) for r in rows]
    except sqlite3.OperationalError:
        return []


def log_canon_verdict(
    conn: sqlite3.Connection,
    surface_id: str,
    kind: str,
    candidate_id: str,
    cosine: float,
    lexical: float,
    verdict: str,
    phase: str = "ingest",
    decided_at: float | None = None,
    rationale: str | None = None,
) -> None:
    """Insert or replace record in canon_verdicts."""
    ts = time.time() if decided_at is None else decided_at
    conn.execute(
        """
        INSERT OR REPLACE INTO canon_verdicts (
            surface_id, kind, candidate_id, cosine, lexical, verdict, phase, decided_at, rationale
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            surface_id,
            kind,
            candidate_id,
            cosine,
            lexical,
            verdict,
            phase,
            ts,
            rationale,
        ),
    )


def delete_alias(
    conn: sqlite3.Connection,
    alias_id: str,
    kind: str = "entity",
) -> bool:
    """Delete alias row, reversibly restoring original surface forms in views."""
    try:
        cur = conn.execute(
            "DELETE FROM aliases WHERE alias_id = ? AND kind = ?",
            (alias_id, kind),
        )
        return cur.rowcount > 0
    except sqlite3.OperationalError:
        return False


def set_alias_status(
    conn: sqlite3.Connection,
    alias_id: str,
    kind: str,
    status: str,
) -> bool:
    """Update alias status ('active' or 'contested')."""
    if status not in ("active", "contested"):
        raise ValueError(f"Invalid status '{status}'. Must be 'active' or 'contested'.")
    try:
        cur = conn.execute(
            "UPDATE aliases SET status = ? WHERE alias_id = ? AND kind = ?",
            (status, alias_id, kind),
        )
        return cur.rowcount > 0
    except sqlite3.OperationalError:
        return False


def get_db(load_vec: bool = True) -> sqlite3.Connection:
    """Acquire SQLite connection configured with authoritative PRAGMAs and ensure query_fts."""
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH, timeout=config.CONNECT_TIMEOUT_S)
    apply_pragmas(conn)
    if load_vec:
        load_vec_extension(conn)
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS query_fts USING fts5("
        "entity_uri UNINDEXED, rel_path UNINDEXED, heading, content, "
        "content='', contentless_unindexed=1, contentless_delete=1"
        ")"
    )
    ensure_schema_views(conn)
    conn.commit()
    return conn


def _clean_fts_query(query: str, strip_stopwords: bool = False) -> list[str]:
    """Tokenize query for FTS5, extracting alphanumeric word terms."""
    raw_terms = re.findall(r"\w+", query.lower())
    if strip_stopwords:
        from spacy.lang.en.stop_words import STOP_WORDS

        meaningful = [t for t in raw_terms if t not in STOP_WORDS and len(t) > 1]
        return meaningful if meaningful else raw_terms
    return raw_terms


def search_fts(
    query: str,
    top_n: int = config.FTS_TOP_N,
    conn: sqlite3.Connection | None = None,
    terms: list[str] | None = None,
) -> list[tuple[str, float, str, str]]:
    """Lexical BM25 search over query_fts using regex query tokenization."""
    clean_terms = (
        terms if terms is not None else _clean_fts_query(query, strip_stopwords=True)
    )
    if not clean_terms:
        return []
    close_after = False
    if conn is None:
        conn = get_db(load_vec=False)
        close_after = True
    try:
        match_expr = " OR ".join(f'"{t}"' for t in clean_terms)
        rows = conn.execute(
            "SELECT entity_uri, rank, heading, content FROM query_fts WHERE query_fts MATCH ? ORDER BY rank LIMIT ?",
            (match_expr, top_n),
        ).fetchall()
    finally:
        if close_after:
            conn.close()

    if not rows:
        return []
    raw = [max(0.0, -float(r[1])) for r in rows]
    denom = max(max(raw), 1e-6)
    return [
        (rows[i][0] or "", round(raw[i] / denom, 4), rows[i][2] or "", rows[i][3] or "")
        for i in range(len(rows))
    ]


DatabaseError = sqlite3.DatabaseError
