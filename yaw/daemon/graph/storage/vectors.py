"""Vector serialization, sqlite-vec cosine KNN, and contentless FTS5 BM25 lexical search."""

import re
import sqlite3
from collections.abc import Sequence

import numpy as np

from .db import YawConnection, batched, write_tx

_MAX_FTS_TERMS = 24
FTS_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "why",
        "with",
    }
)

_VEC_TABLES: dict[str, tuple[str, str]] = {
    "chunk": ("chunk_vec", "chunk_id"),
    "entity": ("entity_vec", "entity_id"),
    "fact": ("fact_vec", "fact_id"),
}

_FTS_TABLES: dict[str, tuple[str, str]] = {
    "chunk": ("chunk_fts", "chunk_id"),
    "fact": ("fact_fts", "fact_id"),
}


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


def _upsert_vectors(
    conn: sqlite3.Connection | YawConnection,
    kind: str,
    vectors: Sequence[tuple[str, np.ndarray | Sequence[float]]],
) -> None:
    if not vectors:
        return
    table, id_col = _VEC_TABLES[kind]
    with write_tx(conn):
        conn.executemany(
            f"INSERT OR REPLACE INTO {table} ({id_col}, embedding) VALUES (?, ?)",
            ((item_id, serialize_f32_vector(vec)) for item_id, vec in vectors),
        )


def _search_knn(
    conn: sqlite3.Connection | YawConnection,
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
    conn: sqlite3.Connection | YawConnection,
    vectors: Sequence[tuple[str, np.ndarray | Sequence[float]]],
) -> None:
    """Upsert chunk embedding vectors into chunk_vec."""
    _upsert_vectors(conn, "chunk", vectors)


def upsert_entity_vectors(
    conn: sqlite3.Connection | YawConnection,
    vectors: Sequence[tuple[str, np.ndarray | Sequence[float]]],
) -> None:
    """Upsert entity embedding vectors into entity_vec."""
    _upsert_vectors(conn, "entity", vectors)


def upsert_fact_vectors(
    conn: sqlite3.Connection | YawConnection,
    vectors: Sequence[tuple[str, np.ndarray | Sequence[float]]],
) -> None:
    """Upsert fact embedding vectors into fact_vec."""
    _upsert_vectors(conn, "fact", vectors)


def search_chunk_vectors_knn(
    conn: sqlite3.Connection | YawConnection,
    query_vec: np.ndarray | Sequence[float],
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Bounded SQL KNN top-k cosine search over chunk_vec."""
    return _search_knn(conn, "chunk", query_vec, top_k)


def search_entity_vectors_knn(
    conn: sqlite3.Connection | YawConnection,
    query_vec: np.ndarray | Sequence[float],
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Bounded SQL KNN top-k cosine search over entity_vec."""
    return _search_knn(conn, "entity", query_vec, top_k)


def search_fact_vectors_knn(
    conn: sqlite3.Connection | YawConnection,
    query_vec: np.ndarray | Sequence[float],
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Bounded SQL KNN top-k cosine search over fact_vec."""
    return _search_knn(conn, "fact", query_vec, top_k)


def get_entity_vectors(
    conn: sqlite3.Connection | YawConnection,
    entity_ids: Sequence[str],
) -> dict[str, np.ndarray]:
    """Load entity embedding vectors by entity_id in bounded SQL batches."""
    if not entity_ids:
        return {}
    results: dict[str, np.ndarray] = {}
    for batch in batched(entity_ids):
        placeholders = ",".join("?" for _ in batch)
        cursor = conn.execute(
            f"SELECT entity_id, embedding FROM entity_vec WHERE entity_id IN ({placeholders})",
            tuple(batch),
        )
        for row in cursor:
            results[str(row[0])] = deserialize_f32_vector(row[1])
    return results


def _extract_fts_terms(query: str) -> list[str]:
    raw_tokens = re.findall(r"\w+", query.lower())
    if not raw_tokens:
        return []
    unique_tokens = list(dict.fromkeys(t for t in raw_tokens if len(t) > 1))
    if not unique_tokens:
        unique_tokens = list(dict.fromkeys(raw_tokens))
    filtered = [t for t in unique_tokens if t not in FTS_STOPWORDS]
    active = filtered if filtered else unique_tokens
    return active[:_MAX_FTS_TERMS]


def _search_fts_bm25(
    conn: sqlite3.Connection | YawConnection,
    kind: str,
    query: str,
    top_k: int,
) -> list[tuple[str, float]]:
    if top_k <= 0:
        return []
    terms = _extract_fts_terms(query)
    if not terms:
        return []

    table, id_col = _FTS_TABLES[kind]
    match_expr = " OR ".join(f'"{t}"' for t in terms)
    cursor = conn.execute(
        f"""
        SELECT {id_col}, bm25({table}) AS bm25_raw
        FROM {table}
        WHERE {table} MATCH ?
        ORDER BY bm25_raw ASC
        LIMIT ?
        """,
        (match_expr, top_k),
    )
    hits = [(str(row[0]), max(1e-6, -float(row[1]))) for row in cursor]
    if not hits:
        return []

    total = sum(score for _, score in hits)
    return [(item_id, score / total) for item_id, score in hits]


def search_chunks_fts(
    conn: sqlite3.Connection | YawConnection,
    query: str,
    top_k: int = 20,
) -> list[tuple[str, float]]:
    """Execute contentless FTS5 lexical match returning (chunk_id, score) with L1-normalized BM25 weights."""
    return _search_fts_bm25(conn, "chunk", query, top_k)


def search_facts_fts(
    conn: sqlite3.Connection | YawConnection,
    query: str,
    top_k: int = 50,
) -> list[tuple[str, float]]:
    """Execute contentless FTS5 lexical match over fact triples returning (fact_id, score) with L1-normalized BM25."""
    return _search_fts_bm25(conn, "fact", query, top_k)
