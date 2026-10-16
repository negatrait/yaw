"""Unified yaw.db SQLite storage engine, sqlite-vec KNN, and contentless FTS5 lexical search."""

import json
import re
import sqlite3
from collections.abc import Generator, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import sqlite_vec

from yaw.core import ChunkPointer, OpenIERecord

_SQL_BATCH_SIZE = 500
_MAX_FTS_TERMS = 24
_FTS_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
        "for", "from", "had", "has", "have", "how", "in", "into", "is", "it",
        "its", "of", "on", "or", "that", "the", "their", "then", "there",
        "these", "they", "this", "to", "was", "were", "what", "when", "where",
        "which", "who", "whom", "why", "with",
    }
)

_VEC_TABLES: dict[str, tuple[str, str]] = {
    "chunk": ("chunk_vec", "chunk_id"),
    "entity": ("entity_vec", "entity_id"),
    "fact": ("fact_vec", "fact_id"),
}

_CREATE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    rel_posix_path TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    title TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(rel_posix_path);
CREATE INDEX IF NOT EXISTS idx_chunks_path_end ON chunks(rel_posix_path, end_line);

CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
    chunk_id UNINDEXED,
    title,
    body,
    content='',
    contentless_delete=1,
    contentless_unindexed=1
);

CREATE TABLE IF NOT EXISTS openie_chunks (
    chunk_id TEXT PRIMARY KEY,
    entities_json TEXT NOT NULL,
    triples_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entities (
    entity_id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS facts (
    fact_id TEXT PRIMARY KEY,
    subj_id TEXT NOT NULL,
    pred TEXT NOT NULL,
    obj_id TEXT NOT NULL,
    raw_triple_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_facts_subj ON facts(subj_id);
CREATE INDEX IF NOT EXISTS idx_facts_obj ON facts(obj_id);
CREATE INDEX IF NOT EXISTS idx_facts_pair ON facts(subj_id, obj_id);

CREATE VIRTUAL TABLE IF NOT EXISTS fact_fts USING fts5(
    fact_id UNINDEXED,
    text,
    content='',
    contentless_delete=1,
    contentless_unindexed=1
);

CREATE TABLE IF NOT EXISTS edges (
    src_id TEXT NOT NULL,
    dst_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    weight REAL NOT NULL,
    PRIMARY KEY (src_id, dst_id, kind)
);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src_id, kind);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst_id, kind);

CREATE TABLE IF NOT EXISTS edge_sources (
    src_id TEXT NOT NULL,
    dst_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    PRIMARY KEY (src_id, dst_id, kind, chunk_id)
);
CREATE INDEX IF NOT EXISTS idx_edge_sources_chunk ON edge_sources(chunk_id);

CREATE TABLE IF NOT EXISTS chunk_vec (
    chunk_id TEXT PRIMARY KEY,
    embedding BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS entity_vec (
    entity_id TEXT PRIMARY KEY,
    embedding BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS fact_vec (
    fact_id TEXT PRIMARY KEY,
    embedding BLOB NOT NULL
);
"""


def _batched(items: Sequence[str], size: int = _SQL_BATCH_SIZE) -> Iterator[Sequence[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


@contextmanager
def _write_tx(conn: sqlite3.Connection) -> Generator[None]:
    """Acquire an immediate write transaction lock to avoid deferred reader->writer deadlocks."""
    owns_tx = not conn.in_transaction
    if owns_tx:
        conn.execute("BEGIN IMMEDIATE")
    try:
        yield
    except Exception:
        if owns_tx and conn.in_transaction:
            conn.rollback()
        raise
    else:
        if owns_tx and conn.in_transaction:
            conn.commit()


def open_yaw_db(db_path: Path | str) -> sqlite3.Connection:
    """Open or create a unified yaw.db SQLite connection with WAL and sqlite-vec loaded."""
    db_str = str(db_path)
    if db_str != ":memory:":
        Path(db_str).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_str, timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row

    if db_str != ":memory:":
        conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA temp_store = MEMORY;")
    conn.execute("PRAGMA busy_timeout = 30000;")

    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    initialize_schema(conn)
    return conn


def initialize_schema(conn: sqlite3.Connection) -> None:
    """Initialize all 11 unified schema tables and indices in yaw.db and auto-backfill fact_fts if needed."""
    with conn:
        conn.executescript(_CREATE_SCHEMA_SQL)
        # Auto-backfill fact_fts if facts exist but fact_fts is empty
        fact_count = int(conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0])
        if fact_count > 0:
            fts_count = int(conn.execute("SELECT COUNT(*) FROM fact_fts").fetchone()[0])
            if fts_count == 0:
                rows = conn.execute("SELECT fact_id, pred, raw_triple_json FROM facts").fetchall()
                fts_items: list[tuple[str, str]] = []
                for row in rows:
                    fid = str(row[0])
                    pred = str(row[1])
                    try:
                        items = json.loads(row[2])
                        fts_items.append((fid, f"{items[0]} {pred} {items[2]}"))
                    except Exception:
                        fts_items.append((fid, pred))
                if fts_items:
                    conn.executemany("INSERT INTO fact_fts (fact_id, text) VALUES (?, ?)", fts_items)


def serialize_f32_vector(vector: np.ndarray | Sequence[float]) -> bytes:
    """Serialize a 1D float vector into contiguous 32-bit float bytes for sqlite-vec."""
    arr = np.asarray(vector, dtype=np.float32)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"Expected 1D non-empty vector, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Vector contains NaN or infinite values")
    return arr.tobytes()


def deserialize_f32_vector(blob: bytes) -> np.ndarray:
    """Deserialize float32 bytes back into a 1D numpy array."""
    if len(blob) % 4 != 0 or len(blob) == 0:
        raise ValueError(f"Invalid float32 byte buffer length: {len(blob)}")
    return np.frombuffer(blob, dtype=np.float32).copy()


# ---------------------------------------------------------------------------
# Vector Upserts & SQL Cosine KNN (sqlite-vec)
# ---------------------------------------------------------------------------


def _upsert_vectors(
    conn: sqlite3.Connection,
    kind: str,
    vectors: Sequence[tuple[str, np.ndarray | Sequence[float]]],
) -> None:
    if not vectors:
        return
    table, id_col = _VEC_TABLES[kind]
    with _write_tx(conn):
        conn.executemany(
            f"INSERT OR REPLACE INTO {table} ({id_col}, embedding) VALUES (?, ?)",
            ((item_id, serialize_f32_vector(vec)) for item_id, vec in vectors),
        )


def _search_knn(
    conn: sqlite3.Connection,
    kind: str,
    query_vec: np.ndarray | Sequence[float],
    top_k: int,
) -> list[tuple[str, float]]:
    if top_k <= 0:
        return []
    table, id_col = _VEC_TABLES[kind]
    blob = serialize_f32_vector(query_vec)
    cursor = conn.execute(
        f"""
        SELECT {id_col}, (1.0 - dist) AS similarity
        FROM (
            SELECT {id_col}, vec_distance_cosine(embedding, ?) AS dist
            FROM {table}
            ORDER BY dist ASC
            LIMIT ?
        )
        """,
        (blob, top_k),
    )
    return [(str(row[0]), float(row[1])) for row in cursor]


def upsert_chunk_vectors(
    conn: sqlite3.Connection,
    vectors: Sequence[tuple[str, np.ndarray | Sequence[float]]],
) -> None:
    """Upsert chunk embedding vectors into chunk_vec."""
    _upsert_vectors(conn, "chunk", vectors)


def upsert_entity_vectors(
    conn: sqlite3.Connection,
    vectors: Sequence[tuple[str, np.ndarray | Sequence[float]]],
) -> None:
    """Upsert entity embedding vectors into entity_vec."""
    _upsert_vectors(conn, "entity", vectors)


def upsert_fact_vectors(
    conn: sqlite3.Connection,
    vectors: Sequence[tuple[str, np.ndarray | Sequence[float]]],
) -> None:
    """Upsert fact embedding vectors into fact_vec."""
    _upsert_vectors(conn, "fact", vectors)


def search_chunk_vectors_knn(
    conn: sqlite3.Connection,
    query_vec: np.ndarray | Sequence[float],
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Bounded SQL KNN top-k cosine search over chunk_vec."""
    return _search_knn(conn, "chunk", query_vec, top_k)


def search_entity_vectors_knn(
    conn: sqlite3.Connection,
    query_vec: np.ndarray | Sequence[float],
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Bounded SQL KNN top-k cosine search over entity_vec."""
    return _search_knn(conn, "entity", query_vec, top_k)


def search_fact_vectors_knn(
    conn: sqlite3.Connection,
    query_vec: np.ndarray | Sequence[float],
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Bounded SQL KNN top-k cosine search over fact_vec."""
    return _search_knn(conn, "fact", query_vec, top_k)


def get_entity_vectors(
    conn: sqlite3.Connection,
    entity_ids: Sequence[str],
) -> dict[str, np.ndarray]:
    """Load entity embedding vectors by entity_id in bounded SQL batches."""
    if not entity_ids:
        return {}
    results: dict[str, np.ndarray] = {}
    for batch in _batched(entity_ids):
        placeholders = ",".join("?" for _ in batch)
        cursor = conn.execute(
            f"SELECT entity_id, embedding FROM entity_vec WHERE entity_id IN ({placeholders})",
            tuple(batch),
        )
        for row in cursor:
            results[str(row[0])] = deserialize_f32_vector(row[1])
    return results


def find_entity_synonym_candidates(
    conn: sqlite3.Connection,
    entity_ids: Sequence[str] | None = None,
    threshold: float = 0.8,
    top_k: int = 5,
) -> list[tuple[str, str, float]]:
    """Mine entity synonymy pairs in SQLite where cosine similarity >= threshold without RAM matrices."""
    if top_k <= 0:
        return []
    max_dist = max(0.0, 1.0 - threshold + 1e-6)
    pair_scores: dict[tuple[str, str], float] = {}

    def _scan_entity_rows(rows_iter: Iterator[sqlite3.Row]) -> None:
        for row in rows_iter:
            src_id = str(row[0])
            blob = row[1]
            knn_cursor = conn.execute(
                """
                SELECT entity_id, (1.0 - dist) AS similarity
                FROM (
                    SELECT entity_id, vec_distance_cosine(embedding, ?) AS dist
                    FROM entity_vec
                    WHERE entity_id != ?
                    ORDER BY dist ASC
                    LIMIT ?
                )
                WHERE dist <= ?
                """,
                (blob, src_id, top_k, max_dist),
            )
            for hit in knn_cursor:
                dst_id = str(hit[0])
                sim = float(hit[1])
                if sim >= threshold:
                    key = (src_id, dst_id) if src_id < dst_id else (dst_id, src_id)
                    if sim > pair_scores.get(key, -1.0):
                        pair_scores[key] = sim

    if entity_ids is None:
        _scan_entity_rows(conn.execute("SELECT entity_id, embedding FROM entity_vec"))
    elif entity_ids:
        for batch in _batched(entity_ids):
            placeholders = ",".join("?" for _ in batch)
            cursor = conn.execute(
                f"SELECT entity_id, embedding FROM entity_vec WHERE entity_id IN ({placeholders})",
                tuple(batch),
            )
            _scan_entity_rows(cursor)

    return [(a, b, sim) for (a, b), sim in sorted(pair_scores.items())]


# ---------------------------------------------------------------------------
# Chunk Metadata & Contentless FTS5 Lexical Search
# ---------------------------------------------------------------------------


def upsert_chunk_pointers(
    conn: sqlite3.Connection,
    chunks: Sequence[tuple[ChunkPointer, str]],
) -> None:
    """Bulk upsert ChunkPointer metadata into chunks and index passage text in contentless chunk_fts."""
    if not chunks:
        return
    with _write_tx(conn):
        conn.executemany(
            """
            INSERT OR REPLACE INTO chunks (chunk_id, rel_posix_path, start_line, end_line, title)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                (p.chunk_id, p.rel_posix_path, p.start_line, p.end_line, p.title)
                for p, _ in chunks
            ),
        )
        conn.executemany(
            "DELETE FROM chunk_fts WHERE chunk_id = ?",
            ((p.chunk_id,) for p, _ in chunks),
        )
        conn.executemany(
            "INSERT INTO chunk_fts (chunk_id, title, body) VALUES (?, ?, ?)",
            ((p.chunk_id, p.title, body) for p, body in chunks),
        )


def _row_to_chunk_pointer(row: sqlite3.Row) -> ChunkPointer:
    return ChunkPointer(
        chunk_id=row["chunk_id"],
        rel_posix_path=row["rel_posix_path"],
        start_line=row["start_line"],
        end_line=row["end_line"],
        title=row["title"],
    )


def get_chunk_pointer(conn: sqlite3.Connection, chunk_id: str) -> ChunkPointer | None:
    """Retrieve ChunkPointer metadata for a single chunk_id."""
    row = conn.execute(
        "SELECT chunk_id, rel_posix_path, start_line, end_line, title FROM chunks WHERE chunk_id = ?",
        (chunk_id,),
    ).fetchone()
    return _row_to_chunk_pointer(row) if row is not None else None


def get_chunk_pointers(
    conn: sqlite3.Connection,
    chunk_ids: Sequence[str],
) -> dict[str, ChunkPointer]:
    """Batch retrieve ChunkPointer metadata for given chunk_ids."""
    if not chunk_ids:
        return {}
    results: dict[str, ChunkPointer] = {}
    for batch in _batched(chunk_ids):
        placeholders = ",".join("?" for _ in batch)
        cursor = conn.execute(
            f"""
            SELECT chunk_id, rel_posix_path, start_line, end_line, title
            FROM chunks WHERE chunk_id IN ({placeholders})
            """,
            tuple(batch),
        )
        for row in cursor:
            results[row["chunk_id"]] = _row_to_chunk_pointer(row)
    return results


def _extract_fts_terms(query: str) -> list[str]:
    raw_tokens = re.findall(r"\w+", query.lower())
    if not raw_tokens:
        return []
    unique_tokens = list(dict.fromkeys(t for t in raw_tokens if len(t) > 1))
    if not unique_tokens:
        unique_tokens = list(dict.fromkeys(raw_tokens))
    filtered = [t for t in unique_tokens if t not in _FTS_STOPWORDS]
    active = filtered if filtered else unique_tokens
    return active[:_MAX_FTS_TERMS]


def search_chunks_fts(
    conn: sqlite3.Connection,
    query: str,
    top_k: int = 20,
) -> list[tuple[str, float]]:
    """Execute contentless FTS5 lexical match returning (chunk_id, score) with L1-normalized BM25 weights."""
    if top_k <= 0:
        return []
    terms = _extract_fts_terms(query)
    if not terms:
        return []

    match_expr = " OR ".join(f'"{t}"' for t in terms)
    cursor = conn.execute(
        """
        SELECT chunk_id, bm25(chunk_fts) AS bm25_raw
        FROM chunk_fts
        WHERE chunk_fts MATCH ?
        ORDER BY bm25_raw ASC
        LIMIT ?
        """,
        (match_expr, top_k),
    )
    hits = [(str(row[0]), max(1e-6, -float(row[1]))) for row in cursor]
    if not hits:
        return []

    total = sum(score for _, score in hits)
    return [(cid, score / total) for cid, score in hits]


def search_facts_fts(
    conn: sqlite3.Connection,
    query: str,
    top_k: int = 50,
) -> list[tuple[str, float]]:
    """Execute contentless FTS5 lexical match over fact triples returning (fact_id, score) with L1-normalized BM25 weights."""
    if top_k <= 0:
        return []
    terms = _extract_fts_terms(query)
    if not terms:
        return []

    match_expr = " OR ".join(f'"{t}"' for t in terms)
    cursor = conn.execute(
        """
        SELECT fact_id, bm25(fact_fts) AS bm25_raw
        FROM fact_fts
        WHERE fact_fts MATCH ?
        ORDER BY bm25_raw ASC
        LIMIT ?
        """,
        (match_expr, top_k),
    )
    hits = [(str(row[0]), max(1e-6, -float(row[1]))) for row in cursor]
    if not hits:
        return []

    total = sum(score for _, score in hits)
    return [(fid, score / total) for fid, score in hits]


# ---------------------------------------------------------------------------
# OpenIE Checkpoints Cache
# ---------------------------------------------------------------------------


def get_openie_checkpoints(
    conn: sqlite3.Connection,
    chunk_ids: Sequence[str],
) -> dict[str, OpenIERecord]:
    """Retrieve cached OpenIE extraction records for chunk_ids in bounded SQL batches."""
    if not chunk_ids:
        return {}
    results: dict[str, OpenIERecord] = {}
    for batch in _batched(chunk_ids):
        placeholders = ",".join("?" for _ in batch)
        cursor = conn.execute(
            f"SELECT chunk_id, entities_json, triples_json FROM openie_chunks WHERE chunk_id IN ({placeholders})",
            tuple(batch),
        )
        for row in cursor:
            cid = str(row[0])
            entities = tuple(json.loads(row[1]))
            triples = tuple((t[0], t[1], t[2]) for t in json.loads(row[2]))
            results[cid] = OpenIERecord(chunk_id=cid, entities=entities, triples=triples)
    return results


def save_openie_checkpoints(
    conn: sqlite3.Connection,
    records: Sequence[OpenIERecord],
) -> None:
    """Save OpenIE extraction records into openie_chunks for crash-resilient caching."""
    if not records:
        return
    with _write_tx(conn):
        conn.executemany(
            "INSERT OR REPLACE INTO openie_chunks (chunk_id, entities_json, triples_json) VALUES (?, ?, ?)",
            (
                (rec.chunk_id, json.dumps(list(rec.entities)), json.dumps([list(t) for t in rec.triples]))
                for rec in records
            ),
        )


# ---------------------------------------------------------------------------
# Graph Upserts & Provenance
# ---------------------------------------------------------------------------


def upsert_entities(
    conn: sqlite3.Connection,
    entities: Sequence[tuple[str, str]],
) -> None:
    """Upsert entity nodes (entity_id, name) into entities."""
    if not entities:
        return
    with _write_tx(conn):
        conn.executemany(
            "INSERT OR REPLACE INTO entities (entity_id, name) VALUES (?, ?)",
            entities,
        )


def upsert_facts(
    conn: sqlite3.Connection,
    facts: Sequence[tuple[str, str, str, str, str]],
) -> None:
    """Upsert fact nodes (fact_id, subj_id, pred, obj_id, raw_triple_json) into facts and index in fact_fts."""
    if not facts:
        return
    with _write_tx(conn):
        conn.executemany(
            """
            INSERT OR REPLACE INTO facts (fact_id, subj_id, pred, obj_id, raw_triple_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            facts,
        )
        conn.executemany(
            "DELETE FROM fact_fts WHERE fact_id = ?",
            ((fid,) for fid, _, _, _, _ in facts),
        )
        fts_items: list[tuple[str, str]] = []
        for fid, _, pred, _, raw_json in facts:
            try:
                items = json.loads(raw_json)
                fts_items.append((fid, f"{items[0]} {pred} {items[2]}"))
            except Exception:
                fts_items.append((fid, pred))
        if fts_items:
            conn.executemany(
                "INSERT INTO fact_fts (fact_id, text) VALUES (?, ?)",
                fts_items,
            )


def upsert_edges_with_sources(
    conn: sqlite3.Connection,
    edges_with_sources: Sequence[tuple[str, str, str, float, str]],
) -> None:
    """Bulk upsert graph edges and record chunk provenance in edge_sources."""
    if not edges_with_sources:
        return
    with _write_tx(conn):
        conn.executemany(
            """
            INSERT OR IGNORE INTO edge_sources (src_id, dst_id, kind, chunk_id)
            VALUES (?, ?, ?, ?)
            """,
            ((src, dst, kind, cid) for src, dst, kind, _, cid in edges_with_sources),
        )
        conn.executemany(
            """
            INSERT INTO edges (src_id, dst_id, kind, weight)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(src_id, dst_id, kind) DO UPDATE SET weight = excluded.weight
            """,
            ((src, dst, kind, weight) for src, dst, kind, weight, _ in edges_with_sources),
        )
        fact_pairs = {
            (src, dst) for src, dst, kind, _, _ in edges_with_sources if kind == "fact"
        }
        if fact_pairs:
            conn.executemany(
                """
                UPDATE edges
                SET weight = (
                    SELECT COUNT(*) FROM edge_sources es
                    WHERE es.src_id = edges.src_id AND es.dst_id = edges.dst_id AND es.kind = 'fact'
                )
                WHERE src_id = ? AND dst_id = ? AND kind = 'fact'
                """,
                fact_pairs,
            )


_EXISTING_ID_TABLES: dict[str, tuple[str, str]] = {
    "chunk": ("chunks", "chunk_id"),
    "chunk_vec": ("chunk_vec", "chunk_id"),
    "entity": ("entities", "entity_id"),
    "entity_vec": ("entity_vec", "entity_id"),
    "fact": ("facts", "fact_id"),
    "fact_vec": ("fact_vec", "fact_id"),
    "openie": ("openie_chunks", "chunk_id"),
}


def get_existing_ids(
    conn: sqlite3.Connection,
    table_kind: str,
    candidate_ids: Sequence[str],
) -> set[str]:
    """Check which candidate IDs already exist in the given table in bounded SQL batches."""
    if not candidate_ids:
        return set()
    if table_kind not in _EXISTING_ID_TABLES:
        raise ValueError(
            f"Unknown table kind: {table_kind!r}. Expected one of {list(_EXISTING_ID_TABLES.keys())}"
        )
    table, col = _EXISTING_ID_TABLES[table_kind]
    existing: set[str] = set()
    for batch in _batched(candidate_ids):
        placeholders = ",".join("?" for _ in batch)
        cursor = conn.execute(
            f"SELECT {col} FROM {table} WHERE {col} IN ({placeholders})",
            tuple(batch),
        )
        for row in cursor:
            existing.add(str(row[0]))
    return existing


def get_facts_by_ids(
    conn: sqlite3.Connection,
    fact_ids: Sequence[str],
) -> dict[str, tuple[str, str, str]]:
    """Retrieve original (subj, pred, obj) triples for candidate fact_ids in bounded batches."""
    if not fact_ids:
        return {}
    results: dict[str, tuple[str, str, str]] = {}
    for batch in _batched(fact_ids):
        placeholders = ",".join("?" for _ in batch)
        cursor = conn.execute(
            f"SELECT fact_id, raw_triple_json FROM facts WHERE fact_id IN ({placeholders})",
            tuple(batch),
        )
        for row in cursor:
            fid = str(row[0])
            raw_list = json.loads(row[1])
            results[fid] = (str(raw_list[0]), str(raw_list[1]), str(raw_list[2]))
    return results


def get_entity_doc_frequencies(
    conn: sqlite3.Connection,
    entity_ids: Sequence[str],
) -> dict[str, int]:
    """Count number of passage edges incident to given entity_ids for PPR seed weighting."""
    if not entity_ids:
        return {}
    freqs: dict[str, int] = {eid: 0 for eid in entity_ids}
    for batch in _batched(entity_ids):
        placeholders = ",".join("?" for _ in batch)
        cursor = conn.execute(
            f"""
            SELECT dst_id, COUNT(*)
            FROM edges
            WHERE kind = 'passage' AND dst_id IN ({placeholders})
            GROUP BY dst_id
            """,
            tuple(batch),
        )
        for row in cursor:
            freqs[str(row[0])] = int(row[1])
    return freqs


def get_max_end_line(
    conn: sqlite3.Connection,
    rel_posix_path: str,
) -> int:
    """Return the maximum 1-indexed end_line for chunks belonging to rel_posix_path, or 0 if none."""
    row = conn.execute(
        "SELECT COALESCE(MAX(end_line), 0) FROM chunks WHERE rel_posix_path = ?",
        (rel_posix_path,),
    ).fetchone()
    return int(row[0]) if row is not None else 0


def get_graph_counts(conn: sqlite3.Connection) -> tuple[int, int, int, int, int]:
    """Return (chunks_count, entities_count, facts_count, edges_count, openie_count) in a single query."""
    row = conn.execute(
        """
        SELECT
            (SELECT COUNT(*) FROM chunks),
            (SELECT COUNT(*) FROM entities),
            (SELECT COUNT(*) FROM facts),
            (SELECT COUNT(*) FROM edges),
            (SELECT COUNT(*) FROM openie_chunks)
        """
    ).fetchone()
    if row is None:
        return (0, 0, 0, 0, 0)
    return (int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]))


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    """Retrieve a metadata value by key from the meta table, or None if not present."""
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return str(row[0]) if row is not None else None


def set_meta(conn: sqlite3.Connection, key: str, value: str | None) -> None:
    """Upsert or delete (when value is None) a metadata entry in the meta table."""
    with _write_tx(conn):
        if value is None:
            conn.execute("DELETE FROM meta WHERE key = ?", (key,))
        else:
            conn.execute(
                """
                INSERT INTO meta (key, value)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, value),
            )


def load_graph_topology(
    conn: sqlite3.Connection,
) -> tuple[tuple[str, ...], tuple[str, ...], list[tuple[str, str, float]]]:
    """Load all active chunk IDs, entity IDs, and max-weight edges from yaw.db to build the CSR graph."""
    chunk_cursor = conn.execute("SELECT chunk_id FROM chunks ORDER BY chunk_id ASC")
    chunk_ids = tuple(str(row[0]) for row in chunk_cursor)

    entity_cursor = conn.execute("SELECT entity_id FROM entities ORDER BY entity_id ASC")
    entity_ids = tuple(str(row[0]) for row in entity_cursor)

    edge_cursor = conn.execute(
        """
        SELECT src_id, dst_id, MAX(weight)
        FROM edges
        WHERE src_id != dst_id AND weight > 0.0
        GROUP BY src_id, dst_id
        """
    )
    edges = [(str(row[0]), str(row[1]), float(row[2])) for row in edge_cursor]

    return chunk_ids, entity_ids, edges


def upsert_synonymy_edges(
    conn: sqlite3.Connection,
    edges: Sequence[tuple[str, str, float]],
) -> None:
    """Bulk upsert bidirectional synonymy edges between entity pairs."""
    if not edges:
        return
    with _write_tx(conn):
        conn.executemany(
            """
            INSERT INTO edges (src_id, dst_id, kind, weight)
            VALUES (?, ?, 'synonymy', ?)
            ON CONFLICT(src_id, dst_id, kind) DO UPDATE SET weight = excluded.weight
            """,
            (
                row
                for ent1, ent2, sim in edges
                for row in ((ent1, ent2, sim), (ent2, ent1, sim))
            ),
        )


# ---------------------------------------------------------------------------
# Atomic Scoped Cascading Chunk Deletion
# ---------------------------------------------------------------------------


def delete_chunks(conn: sqlite3.Connection, chunk_ids: Sequence[str]) -> None:
    """Atomically delete chunks and cascade removal strictly across affected edges, facts, and entities."""
    if not chunk_ids:
        return

    with _write_tx(conn):
        conn.executescript(
            """
            CREATE TEMP TABLE IF NOT EXISTS _del_chunks (chunk_id TEXT PRIMARY KEY);
            CREATE TEMP TABLE IF NOT EXISTS _affected_edges (
                src_id TEXT NOT NULL,
                dst_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                PRIMARY KEY (src_id, dst_id, kind)
            );
            CREATE TEMP TABLE IF NOT EXISTS _affected_entities (entity_id TEXT PRIMARY KEY);
            CREATE TEMP TABLE IF NOT EXISTS _orphan_facts (fact_id TEXT PRIMARY KEY);
            CREATE TEMP TABLE IF NOT EXISTS _orphan_entities (entity_id TEXT PRIMARY KEY);
            DELETE FROM temp._del_chunks;
            DELETE FROM temp._affected_edges;
            DELETE FROM temp._affected_entities;
            DELETE FROM temp._orphan_facts;
            DELETE FROM temp._orphan_entities;
            """
        )
        conn.executemany(
            "INSERT OR IGNORE INTO temp._del_chunks (chunk_id) VALUES (?)",
            ((cid,) for cid in chunk_ids),
        )

        # 1. Capture only edges and entities incident to the deleted chunks before removing provenance
        conn.execute(
            """
            INSERT OR IGNORE INTO temp._affected_edges (src_id, dst_id, kind)
            SELECT src_id, dst_id, kind
            FROM edge_sources
            WHERE chunk_id IN (SELECT chunk_id FROM temp._del_chunks)
            """
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO temp._affected_entities (entity_id)
            SELECT dst_id FROM temp._affected_edges WHERE kind IN ('passage', 'fact')
            UNION
            SELECT src_id FROM temp._affected_edges WHERE kind = 'fact'
            UNION
            SELECT src_id FROM temp._affected_edges WHERE kind = 'passage'
              AND src_id NOT IN (SELECT chunk_id FROM temp._del_chunks)
            """
        )
        # Include synonym-linked neighbors of affected entities so isolated aliases are also checked
        conn.execute(
            """
            INSERT OR IGNORE INTO temp._affected_entities (entity_id)
            SELECT dst_id FROM edges
            WHERE kind = 'synonymy' AND src_id IN (SELECT entity_id FROM temp._affected_entities)
            UNION
            SELECT src_id FROM edges
            WHERE kind = 'synonymy' AND dst_id IN (SELECT entity_id FROM temp._affected_entities)
            """
        )

        # 2. Remove chunk rows and provenance
        conn.executemany(
            "DELETE FROM chunk_fts WHERE chunk_id = ?",
            ((cid,) for cid in chunk_ids),
        )
        conn.execute("DELETE FROM chunk_vec WHERE chunk_id IN (SELECT chunk_id FROM temp._del_chunks)")
        conn.execute("DELETE FROM openie_chunks WHERE chunk_id IN (SELECT chunk_id FROM temp._del_chunks)")
        conn.execute("DELETE FROM chunks WHERE chunk_id IN (SELECT chunk_id FROM temp._del_chunks)")
        conn.execute("DELETE FROM edge_sources WHERE chunk_id IN (SELECT chunk_id FROM temp._del_chunks)")

        # 3. Prune affected passage/fact edges with zero remaining provenance and update surviving fact weights
        conn.execute(
            """
            DELETE FROM edges
            WHERE (src_id, dst_id, kind) IN (SELECT src_id, dst_id, kind FROM temp._affected_edges)
              AND NOT EXISTS (
                  SELECT 1 FROM edge_sources es
                  WHERE es.src_id = edges.src_id
                    AND es.dst_id = edges.dst_id
                    AND es.kind = edges.kind
              )
            """
        )
        conn.execute(
            """
            UPDATE edges
            SET weight = (
                SELECT COUNT(*) FROM edge_sources es
                WHERE es.src_id = edges.src_id
                  AND es.dst_id = edges.dst_id
                  AND es.kind = 'fact'
            )
            WHERE kind = 'fact'
              AND (src_id, dst_id, kind) IN (SELECT src_id, dst_id, kind FROM temp._affected_edges)
            """
        )

        # 4. Prune orphaned facts and fact_vec strictly among affected fact edges
        conn.execute(
            """
            INSERT OR IGNORE INTO temp._orphan_facts (fact_id)
            SELECT f.fact_id
            FROM facts f
            INNER JOIN temp._affected_edges ae
              ON ae.kind = 'fact' AND ae.src_id = f.subj_id AND ae.dst_id = f.obj_id
            WHERE NOT EXISTS (
                SELECT 1 FROM edges e
                WHERE e.src_id = f.subj_id AND e.dst_id = f.obj_id AND e.kind = 'fact'
            )
            """
        )
        conn.execute("DELETE FROM fact_vec WHERE fact_id IN (SELECT fact_id FROM temp._orphan_facts)")
        conn.execute("DELETE FROM fact_fts WHERE fact_id IN (SELECT fact_id FROM temp._orphan_facts)")
        conn.execute("DELETE FROM facts WHERE fact_id IN (SELECT fact_id FROM temp._orphan_facts)")

        # 5. Prune orphaned entities and entity_vec strictly among candidate entities using index seeks
        conn.execute(
            """
            INSERT OR IGNORE INTO temp._orphan_entities (entity_id)
            SELECT ae.entity_id
            FROM temp._affected_entities ae
            WHERE NOT EXISTS (
                SELECT 1 FROM edges e
                WHERE e.src_id = ae.entity_id AND e.kind IN ('passage', 'fact')
            )
            AND NOT EXISTS (
                SELECT 1 FROM edges e
                WHERE e.dst_id = ae.entity_id AND e.kind IN ('passage', 'fact')
            )
            """
        )
        conn.execute("DELETE FROM entity_vec WHERE entity_id IN (SELECT entity_id FROM temp._orphan_entities)")
        conn.execute("DELETE FROM entities WHERE entity_id IN (SELECT entity_id FROM temp._orphan_entities)")

        # 6. Prune synonymy edges incident to deleted entities via indexed src_id/dst_id lookups
        conn.execute(
            """
            DELETE FROM edges
            WHERE kind = 'synonymy'
              AND (
                  src_id IN (SELECT entity_id FROM temp._orphan_entities)
                  OR dst_id IN (SELECT entity_id FROM temp._orphan_entities)
              )
            """
        )
