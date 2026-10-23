"""Unified yaw.db SQLite storage engine, sqlite-vec KNN, and contentless FTS5 lexical search."""

import difflib
import json
import re
import sqlite3
from collections.abc import Generator, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import sqlite_vec
from cocoindex.connectors.sqlite import ManagedConnection
from cocoindex.connectors.sqlite import connect as coco_sqlite_connect

from yaw.core import ChunkPointer, OpenIERecord, SHACLViolation, StructuredTriple

_SQL_BATCH_SIZE = 500
_MAX_FTS_TERMS = 24
_FTS_STOPWORDS = frozenset(
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

_LABEL_KIND_PRIORITY: dict[str, int] = {
    "skos:prefLabel": 3,
    "skos:altLabel": 2,
    "skos:hiddenLabel": 1,
}

_CREATE_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id TEXT PRIMARY KEY,
        rel_posix_path TEXT NOT NULL,
        start_line INTEGER NOT NULL,
        end_line INTEGER NOT NULL,
        title TEXT NOT NULL DEFAULT '',
        ast_island_start INTEGER NOT NULL DEFAULT 0,
        ast_island_end INTEGER NOT NULL DEFAULT 0
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(rel_posix_path)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_path_end ON chunks(rel_posix_path, end_line)",
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
        chunk_id UNINDEXED,
        title,
        body,
        content='',
        contentless_delete=1,
        contentless_unindexed=1
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS openie_chunks (
        chunk_id TEXT PRIMARY KEY,
        entities_json TEXT NOT NULL,
        triples_json TEXT NOT NULL,
        structured_triples_json TEXT NOT NULL DEFAULT '[]'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS entities (
        entity_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        definition TEXT NOT NULL DEFAULT ''
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)",
    """
    CREATE TABLE IF NOT EXISTS entity_labels (
        label_norm TEXT NOT NULL,
        entity_id TEXT NOT NULL,
        label_kind TEXT NOT NULL CHECK (label_kind IN ('skos:prefLabel', 'skos:altLabel', 'skos:hiddenLabel')),
        raw_label TEXT NOT NULL DEFAULT '',
        chunk_id TEXT NOT NULL,
        PRIMARY KEY (label_norm, entity_id, chunk_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_entity_labels_norm ON entity_labels(label_norm)",
    "CREATE INDEX IF NOT EXISTS idx_entity_labels_entity ON entity_labels(entity_id)",
    "CREATE INDEX IF NOT EXISTS idx_entity_labels_chunk ON entity_labels(chunk_id)",
    """
    CREATE TABLE IF NOT EXISTS facts (
        fact_id TEXT PRIMARY KEY,
        subj_id TEXT NOT NULL,
        pred TEXT NOT NULL,
        obj_id TEXT NOT NULL,
        raw_triple_json TEXT NOT NULL,
        definition TEXT NOT NULL DEFAULT ''
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_facts_subj ON facts(subj_id)",
    "CREATE INDEX IF NOT EXISTS idx_facts_obj ON facts(obj_id)",
    "CREATE INDEX IF NOT EXISTS idx_facts_pair ON facts(subj_id, obj_id)",
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS fact_fts USING fts5(
        fact_id UNINDEXED,
        text,
        content='',
        contentless_delete=1,
        contentless_unindexed=1
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS edges (
        src_id TEXT NOT NULL,
        dst_id TEXT NOT NULL,
        kind TEXT NOT NULL CHECK (kind IN ('passage', 'fact', 'skos:related', 'owl:differentFrom')),
        weight REAL NOT NULL,
        PRIMARY KEY (src_id, dst_id, kind)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src_id, kind)",
    "CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst_id, kind)",
    """
    CREATE TABLE IF NOT EXISTS edge_sources (
        src_id TEXT NOT NULL,
        dst_id TEXT NOT NULL,
        kind TEXT NOT NULL,
        chunk_id TEXT NOT NULL,
        PRIMARY KEY (src_id, dst_id, kind, chunk_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_edge_sources_chunk ON edge_sources(chunk_id)",
    """
    CREATE TABLE IF NOT EXISTS chunk_vec (
        chunk_id TEXT PRIMARY KEY,
        embedding BLOB NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS entity_vec (
        entity_id TEXT PRIMARY KEY,
        embedding BLOB NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS fact_vec (
        fact_id TEXT PRIMARY KEY,
        embedding BLOB NOT NULL
    )
    """,
)


def _batched(items: Sequence[str], size: int = _SQL_BATCH_SIZE) -> Iterator[Sequence[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


@contextmanager
def _write_tx(conn: sqlite3.Connection) -> Generator[None]:
    """Acquire an immediate write transaction lock or join an active outer transaction."""
    owns_tx = not conn.in_transaction
    if owns_tx:
        conn.execute("BEGIN IMMEDIATE")
    try:
        yield
    except Exception:
        if owns_tx and conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    else:
        if owns_tx and conn.in_transaction:
            conn.execute("COMMIT")


def open_managed_yaw_db(db_path: Path | str) -> ManagedConnection:
    """Open or create a unified yaw.db via CocoIndex ManagedConnection with FIFO RWLock and sqlite-vec."""
    db_str = str(db_path)
    if db_str != ":memory:":
        Path(db_str).parent.mkdir(parents=True, exist_ok=True)

    managed = coco_sqlite_connect(db_str, timeout=30.0, load_vec=True)
    raw_conn = managed._conn
    raw_conn.row_factory = sqlite3.Row
    raw_conn.execute("PRAGMA synchronous = NORMAL;")
    raw_conn.execute("PRAGMA temp_store = MEMORY;")
    raw_conn.execute("PRAGMA busy_timeout = 30000;")
    with managed.transaction() as conn:
        initialize_schema(conn)
    return managed


def open_yaw_db(db_path: Path | str) -> sqlite3.Connection:
    """Open or create a unified yaw.db SQLite connection with WAL and sqlite-vec loaded."""
    db_str = str(db_path)
    if db_str != ":memory:":
        Path(db_str).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_str, timeout=30.0, check_same_thread=False)
    conn.isolation_level = None
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
    """Initialize all unified schema tables and indices in yaw.db with in-place migrations."""
    with _write_tx(conn):
        for stmt in _CREATE_SCHEMA_STATEMENTS:
            conn.execute(stmt)

        # Auto-migrate legacy content_hash column on chunks if present
        chunk_cols = {row[1] for row in conn.execute("PRAGMA table_info(chunks)").fetchall()}
        if "content_hash" in chunk_cols:
            conn.execute("ALTER TABLE chunks DROP COLUMN content_hash")
        if "ast_island_start" not in chunk_cols:
            conn.execute(
                "ALTER TABLE chunks ADD COLUMN ast_island_start INTEGER NOT NULL DEFAULT 0"
            )
        if "ast_island_end" not in chunk_cols:
            conn.execute("ALTER TABLE chunks ADD COLUMN ast_island_end INTEGER NOT NULL DEFAULT 0")

        # Auto-migrate openie_chunks.structured_triples_json
        openie_cols = {
            row[1] for row in conn.execute("PRAGMA table_info(openie_chunks)").fetchall()
        }
        if "structured_triples_json" not in openie_cols:
            conn.execute(
                "ALTER TABLE openie_chunks ADD COLUMN structured_triples_json TEXT NOT NULL DEFAULT '[]'"
            )

        # Auto-migrate entities.definition and drop legacy UNIQUE constraint on entities.name
        entity_cols = {row[1] for row in conn.execute("PRAGMA table_info(entities)").fetchall()}
        if "definition" not in entity_cols:
            conn.execute("ALTER TABLE entities ADD COLUMN definition TEXT NOT NULL DEFAULT ''")
        entity_ddl_row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'entities'"
        ).fetchone()
        if entity_ddl_row is not None and "UNIQUE" in str(entity_ddl_row[0]).upper():
            conn.execute(
                """
                CREATE TABLE _entities_migrated (
                    entity_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    definition TEXT NOT NULL DEFAULT ''
                )
                """
            )
            conn.execute(
                "INSERT INTO _entities_migrated (entity_id, name, definition) SELECT entity_id, name, definition FROM entities"
            )
            conn.execute("DROP TABLE entities")
            conn.execute("ALTER TABLE _entities_migrated RENAME TO entities")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)")

        # Auto-migrate facts.definition
        fact_cols = {row[1] for row in conn.execute("PRAGMA table_info(facts)").fetchall()}
        if "definition" not in fact_cols:
            conn.execute("ALTER TABLE facts ADD COLUMN definition TEXT NOT NULL DEFAULT ''")

        # Auto-migrate edges CHECK constraint and legacy 'synonymy' kind -> 'skos:related'
        edges_ddl_row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'edges'"
        ).fetchone()
        if edges_ddl_row is not None and "skos:related" not in str(edges_ddl_row[0]):
            conn.execute(
                """
                CREATE TABLE _edges_migrated (
                    src_id TEXT NOT NULL,
                    dst_id TEXT NOT NULL,
                    kind TEXT NOT NULL CHECK (kind IN ('passage', 'fact', 'skos:related', 'owl:differentFrom')),
                    weight REAL NOT NULL,
                    PRIMARY KEY (src_id, dst_id, kind)
                )
                """
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO _edges_migrated (src_id, dst_id, kind, weight)
                SELECT
                    src_id,
                    dst_id,
                    CASE WHEN kind = 'synonymy' THEN 'skos:related' ELSE kind END,
                    weight
                FROM edges
                WHERE kind IN ('passage', 'fact', 'synonymy', 'skos:related', 'owl:differentFrom')
                """
            )
            conn.execute("DROP TABLE edges")
            conn.execute("ALTER TABLE _edges_migrated RENAME TO edges")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src_id, kind)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst_id, kind)")

        # Auto-backfill entity_labels for any legacy entities without labels
        missing_label_rows = conn.execute(
            """
            SELECT e.entity_id, e.name
            FROM entities e
            WHERE NOT EXISTS (
                SELECT 1 FROM entity_labels el WHERE el.entity_id = e.entity_id
            )
            """
        ).fetchall()
        if missing_label_rows:
            conn.executemany(
                """
                INSERT OR IGNORE INTO entity_labels (label_norm, entity_id, label_kind, raw_label, chunk_id)
                VALUES (?, ?, 'skos:prefLabel', ?, '__legacy__')
                """,
                ((str(r[1]).strip().lower(), str(r[0]), str(r[1])) for r in missing_label_rows),
            )

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
                    conn.executemany(
                        "INSERT INTO fact_fts (fact_id, text) VALUES (?, ?)", fts_items
                    )


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


def _is_acronym_match(name_a: str, name_b: str) -> bool:
    """Return True if one label is an initialism/acronym of the other multi-token label."""
    clean_a = re.sub(r"[^a-zA-Z0-9\s]", "", name_a).strip()
    clean_b = re.sub(r"[^a-zA-Z0-9\s]", "", name_b).strip()
    if not clean_a or not clean_b:
        return False
    tokens_a = clean_a.split()
    tokens_b = clean_b.split()
    if len(tokens_a) == 1 and len(tokens_b) >= 2:
        short, words = tokens_a[0].lower(), tokens_b
    elif len(tokens_b) == 1 and len(tokens_a) >= 2:
        short, words = tokens_b[0].lower(), tokens_a
    else:
        return False
    if len(short) < 2:
        return False
    initials = "".join(w[0].lower() for w in words if w)
    if short == initials:
        return True
    non_stop_initials = "".join(
        w[0].lower() for w in words if w and w.lower() not in _FTS_STOPWORDS
    )
    return len(non_stop_initials) >= 2 and short == non_stop_initials


def _lexical_similarity_cascade(name_a: str, name_b: str, threshold: float) -> float:
    """Run the 3-stage difflib.SequenceMatcher cascade + acronym detector."""
    norm_a = name_a.strip().lower()
    norm_b = name_b.strip().lower()
    if not norm_a or not norm_b:
        return 0.0
    if norm_a == norm_b:
        return 1.0
    if _is_acronym_match(name_a, name_b):
        return max(threshold, 0.85)

    len_a, len_b = len(norm_a), len(norm_b)
    if min(len_a, len_b) / max(len_a, len_b) < 0.4:
        return 0.0

    matcher = difflib.SequenceMatcher(autojunk=False)
    matcher.set_seq2(norm_a)
    matcher.set_seq1(norm_b)
    if matcher.real_quick_ratio() < threshold:
        return 0.0
    if matcher.quick_ratio() < threshold:
        return 0.0
    ratio = matcher.ratio()
    return ratio if ratio >= threshold else 0.0


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

    adjudicated: set[tuple[str, str]] = {
        (str(r[0]), str(r[1])) if str(r[0]) < str(r[1]) else (str(r[1]), str(r[0]))
        for r in conn.execute(
            "SELECT src_id, dst_id FROM edges WHERE kind IN ('skos:related', 'owl:differentFrom')"
        )
    }

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
                    if key not in adjudicated and sim > pair_scores.get(key, -1.0):
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


def find_entity_disambiguation_candidates(
    conn: sqlite3.Connection,
    entity_ids: Sequence[str] | None = None,
    *,
    dense_threshold: float = 0.75,
    dense_top_k: int = 5,
    lexical_threshold: float = 0.72,
) -> list[tuple[str, str, float, float]]:
    """Two-channel candidate generation combining sqlite-vec dense kNN and 3-stage SequenceMatcher cascade."""
    adjudicated: set[tuple[str, str]] = {
        (str(r[0]), str(r[1])) if str(r[0]) < str(r[1]) else (str(r[1]), str(r[0]))
        for r in conn.execute(
            "SELECT src_id, dst_id FROM edges WHERE kind IN ('skos:related', 'owl:differentFrom')"
        )
    }

    dense_pairs: dict[tuple[str, str], float] = {}
    for a, b, sim in find_entity_synonym_candidates(
        conn, entity_ids=entity_ids, threshold=dense_threshold, top_k=dense_top_k
    ):
        dense_pairs[(a, b)] = sim

    all_entities = [
        (str(r[0]), str(r[1]), str(r[1]).strip().lower())
        for r in conn.execute("SELECT entity_id, name FROM entities ORDER BY entity_id ASC")
    ]
    entity_names = {eid: name for eid, name, _ in all_entities}
    target_set = set(entity_ids) if entity_ids is not None else None

    lexical_pairs: dict[tuple[str, str], float] = {}
    n_ents = len(all_entities)
    matcher = difflib.SequenceMatcher(autojunk=False)
    for i in range(n_ents):
        eid_a, name_a, norm_a = all_entities[i]
        if not norm_a:
            continue
        a_in_target = target_set is None or eid_a in target_set
        matcher_seq2_set = False
        len_a = len(norm_a)
        for j in range(i + 1, n_ents):
            eid_b, name_b, norm_b = all_entities[j]
            if not a_in_target and (target_set is not None and eid_b not in target_set):
                continue
            if not norm_b:
                continue
            key = (eid_a, eid_b) if eid_a < eid_b else (eid_b, eid_a)
            if key in adjudicated:
                continue
            if norm_a == norm_b:
                lexical_pairs[key] = 1.0
                continue
            if _is_acronym_match(name_a, name_b):
                lexical_pairs[key] = max(lexical_threshold, 0.85)
                continue
            if norm_a.isdigit() and norm_b.isdigit():
                continue
            len_b = len(norm_b)
            if min(len_a, len_b) / max(len_a, len_b) < 0.4:
                continue
            if not matcher_seq2_set:
                matcher.set_seq2(norm_a)
                matcher_seq2_set = True
            matcher.set_seq1(norm_b)
            best_ratio = 0.0
            if (
                matcher.real_quick_ratio() >= lexical_threshold
                and matcher.quick_ratio() >= lexical_threshold
            ):
                best_ratio = matcher.ratio()
            if best_ratio < lexical_threshold and (" " in norm_a or " " in norm_b):
                alnum_a = re.sub(r"[^a-z0-9]", "", norm_a)
                alnum_b = re.sub(r"[^a-z0-9]", "", norm_b)
                if alnum_a and alnum_b and not (alnum_a.isdigit() and alnum_b.isdigit()):
                    alnum_ratio = difflib.SequenceMatcher(
                        None, alnum_b, alnum_a, autojunk=False
                    ).ratio()
                    if alnum_ratio > best_ratio:
                        best_ratio = alnum_ratio
            if best_ratio >= lexical_threshold:
                lexical_pairs[key] = best_ratio

    for row in conn.execute(
        """
        SELECT DISTINCT a.entity_id, b.entity_id
        FROM entity_labels a
        INNER JOIN entity_labels b
          ON a.label_norm = b.label_norm
         AND a.entity_id < b.entity_id
        """
    ):
        ea, eb = str(row[0]), str(row[1])
        if target_set is not None and ea not in target_set and eb not in target_set:
            continue
        key = (ea, eb)
        if key not in adjudicated:
            lexical_pairs[key] = 1.0

    combined_keys = sorted(set(dense_pairs.keys()) | set(lexical_pairs.keys()))
    if not combined_keys:
        return []

    missing_vec_ids: set[str] = set()
    for a, b in combined_keys:
        if (a, b) not in dense_pairs:
            missing_vec_ids.add(a)
            missing_vec_ids.add(b)
    extra_vecs = get_entity_vectors(conn, list(missing_vec_ids)) if missing_vec_ids else {}

    results: list[tuple[str, str, float, float]] = []
    for a, b in combined_keys:
        d_sim = dense_pairs.get((a, b))
        if d_sim is None:
            va = extra_vecs.get(a)
            vb = extra_vecs.get(b)
            if va is not None and vb is not None:
                denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
                d_sim = float(np.dot(va, vb) / denom) if denom > 1e-12 else 0.0
            else:
                d_sim = 0.0
        l_sim = lexical_pairs.get((a, b))
        if l_sim is None:
            raw_a = entity_names.get(a, "")
            raw_b = entity_names.get(b, "")
            na = raw_a.strip().lower()
            nb = raw_b.strip().lower()
            if _is_acronym_match(raw_a, raw_b):
                l_sim = max(lexical_threshold, 0.85)
            elif na and nb:
                l_sim = difflib.SequenceMatcher(None, nb, na, autojunk=False).ratio()
            else:
                l_sim = 0.0
        results.append((a, b, float(d_sim), float(l_sim)))

    return results


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
            INSERT OR REPLACE INTO chunks (chunk_id, rel_posix_path, start_line, end_line, title, ast_island_start, ast_island_end)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    p.chunk_id,
                    p.rel_posix_path,
                    p.start_line,
                    p.end_line,
                    p.title,
                    p.ast_island_start,
                    p.ast_island_end,
                )
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
    keys = row.keys()
    ast_s = int(row["ast_island_start"]) if "ast_island_start" in keys else 0
    ast_e = int(row["ast_island_end"]) if "ast_island_end" in keys else 0
    return ChunkPointer(
        chunk_id=row["chunk_id"],
        rel_posix_path=row["rel_posix_path"],
        start_line=row["start_line"],
        end_line=row["end_line"],
        title=row["title"],
        ast_island_start=ast_s,
        ast_island_end=ast_e,
    )


def get_chunk_pointer(conn: sqlite3.Connection, chunk_id: str) -> ChunkPointer | None:
    """Retrieve ChunkPointer metadata for a single chunk_id."""
    row = conn.execute(
        "SELECT chunk_id, rel_posix_path, start_line, end_line, title, ast_island_start, ast_island_end FROM chunks WHERE chunk_id = ?",
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
            SELECT chunk_id, rel_posix_path, start_line, end_line, title, ast_island_start, ast_island_end
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
            f"""
            SELECT chunk_id, entities_json, triples_json, structured_triples_json
            FROM openie_chunks WHERE chunk_id IN ({placeholders})
            """,
            tuple(batch),
        )
        for row in cursor:
            cid = str(row[0])
            raw_entities = json.loads(row[1]) if row[1] else []
            entities_list: list[str] = []
            entity_defs_list: list[tuple[str, str]] = []
            has_any_def = False
            for item in raw_entities:
                if isinstance(item, (list, tuple)) and len(item) >= 1:
                    ename = str(item[0]).strip()
                    edef = str(item[1]).strip() if len(item) >= 2 and item[1] is not None else ""
                    if ename:
                        entities_list.append(ename)
                        entity_defs_list.append((ename, edef))
                        if edef:
                            has_any_def = True
                elif isinstance(item, str):
                    ename = item.strip()
                    if ename:
                        entities_list.append(ename)
                        entity_defs_list.append((ename, ""))
            triples = tuple((t[0], t[1], t[2]) for t in json.loads(row[2]))
            raw_st = json.loads(row[3]) if row[3] else []
            structured_triples = tuple(
                StructuredTriple(
                    subj=str(item["subj"]),
                    subj_def=str(item.get("subj_def", "")),
                    pred=str(item["pred"]),
                    obj=str(item["obj"]),
                    obj_def=str(item.get("obj_def", "")),
                    spo_def=str(item.get("spo_def", "")),
                )
                for item in raw_st
                if isinstance(item, dict) and "subj" in item and "pred" in item and "obj" in item
            )
            results[cid] = OpenIERecord(
                chunk_id=cid,
                entities=tuple(entities_list),
                triples=triples,
                structured_triples=structured_triples,
                entity_definitions=tuple(entity_defs_list) if has_any_def else (),
            )
    return results


def _serialize_openie_entities(rec: OpenIERecord) -> str:
    if rec.entity_definitions and any(edef for _, edef in rec.entity_definitions):
        def_map = {ename: edef for ename, edef in rec.entity_definitions}
        out: list[list[str]] = []
        seen: set[str] = set()
        for ename in rec.entities:
            seen.add(ename)
            out.append([ename, def_map.get(ename, "")])
        for ename, edef in rec.entity_definitions:
            if ename not in seen:
                seen.add(ename)
                out.append([ename, edef])
        return json.dumps(out)
    return json.dumps(list(rec.entities))


def save_openie_checkpoints(
    conn: sqlite3.Connection,
    records: Sequence[OpenIERecord],
) -> None:
    """Save OpenIE extraction records into openie_chunks for crash-resilient caching."""
    if not records:
        return
    with _write_tx(conn):
        conn.executemany(
            """
            INSERT OR REPLACE INTO openie_chunks (
                chunk_id, entities_json, triples_json, structured_triples_json
            ) VALUES (?, ?, ?, ?)
            """,
            (
                (
                    rec.chunk_id,
                    _serialize_openie_entities(rec),
                    json.dumps([list(t) for t in rec.triples]),
                    json.dumps(
                        [
                            {
                                "subj": st.subj,
                                "subj_def": st.subj_def,
                                "pred": st.pred,
                                "obj": st.obj,
                                "obj_def": st.obj_def,
                                "spo_def": st.spo_def,
                            }
                            for st in rec.structured_triples
                        ]
                    ),
                )
                for rec in records
            ),
        )


# ---------------------------------------------------------------------------
# Graph Upserts, SKOS Label Resolution, Entity Merging & Provenance
# ---------------------------------------------------------------------------


def _normalize_label(text: str) -> str:
    """Normalize an entity label or name to lowercase alphanumeric tokens (matching text_processing)."""
    normalized = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in str(text).casefold())
    return " ".join(normalized.split())


def upsert_entities(
    conn: sqlite3.Connection,
    entities: Sequence[tuple[str, str] | tuple[str, str, str]],
) -> None:
    """Upsert entity nodes (entity_id, name[, definition]) into entities and align existing SKOS labels."""
    if not entities:
        return
    normalized_rows: list[tuple[str, str, str]] = []
    for item in entities:
        if len(item) == 2:
            normalized_rows.append((item[0], item[1], ""))
        else:
            normalized_rows.append((item[0], item[1], item[2]))
    with _write_tx(conn):
        conn.executemany(
            """
            INSERT INTO entities (entity_id, name, definition)
            VALUES (?, ?, ?)
            ON CONFLICT(entity_id) DO UPDATE SET
                name = excluded.name,
                definition = CASE
                    WHEN TRIM(excluded.definition) != '' THEN excluded.definition
                    ELSE entities.definition
                END
            """,
            normalized_rows,
        )
        for eid, ename, _ in normalized_rows:
            raw_norm = ename.strip().lower()
            tok_norm = _normalize_label(ename)
            if not raw_norm and not tok_norm:
                continue
            existing_rows = conn.execute(
                "SELECT DISTINCT label_norm, label_kind FROM entity_labels WHERE entity_id = ?",
                (eid,),
            ).fetchall()
            if not existing_rows:
                continue
            existing_norms = {str(r[0]) for r in existing_rows}
            pref_norm = (
                raw_norm
                if raw_norm in existing_norms
                else (tok_norm if tok_norm in existing_norms else raw_norm)
            )
            if pref_norm:
                conn.execute(
                    """
                    UPDATE entity_labels
                    SET label_kind = 'skos:prefLabel'
                    WHERE entity_id = ? AND label_norm = ? AND label_kind != 'skos:prefLabel'
                    """,
                    (eid, pref_norm),
                )
                conn.execute(
                    """
                    UPDATE entity_labels
                    SET label_kind = 'skos:altLabel'
                    WHERE entity_id = ? AND label_norm != ? AND label_kind = 'skos:prefLabel'
                    """,
                    (eid, pref_norm),
                )


def upsert_entity_labels(
    conn: sqlite3.Connection,
    labels: Sequence[tuple[str, str, str, str, str]],
) -> None:
    """Upsert SKOS entity labels while enforcing SKOS S13/S14 pairwise disjointness across all chunk_ids."""
    if not labels:
        return
    with _write_tx(conn):
        affected_eids = list({item[1] for item in labels})
        entity_names: dict[str, str] = {}
        existing_pref: dict[str, str] = {}
        for batch in _batched(affected_eids):
            placeholders = ",".join("?" for _ in batch)
            for row in conn.execute(
                f"""
                SELECT entity_id, name
                FROM entities
                WHERE entity_id IN ({placeholders})
                """,
                tuple(batch),
            ):
                if str(row[1]).strip():
                    entity_names[str(row[0])] = str(row[1]).strip()
            for row in conn.execute(
                f"""
                SELECT entity_id, label_norm
                FROM entity_labels
                WHERE label_kind = 'skos:prefLabel' AND entity_id IN ({placeholders})
                """,
                tuple(batch),
            ):
                existing_pref[str(row[0])] = str(row[1])

        for label_norm, entity_id, label_kind, raw_label, chunk_id in labels:
            norm = label_norm.strip().lower()
            if not norm:
                continue
            if label_kind not in _LABEL_KIND_PRIORITY:
                raise ValueError(f"Invalid SKOS label_kind: {label_kind!r}")

            ename = entity_names.get(entity_id)
            curr_pref = existing_pref.get(entity_id)
            existing_kinds = [
                str(r[0])
                for r in conn.execute(
                    "SELECT DISTINCT label_kind FROM entity_labels WHERE entity_id = ? AND label_norm = ?",
                    (entity_id, norm),
                )
            ]

            if ename is not None:
                raw_ename_norm = ename.lower()
                tok_ename_norm = _normalize_label(ename)
                matches_entity_name = (
                    norm == raw_ename_norm or _normalize_label(norm) == tok_ename_norm
                )
                if matches_entity_name:
                    if curr_pref is not None and curr_pref != norm:
                        curr_matches_name = (
                            curr_pref == raw_ename_norm
                            or _normalize_label(curr_pref) == tok_ename_norm
                        )
                        if curr_matches_name:
                            # Entity already has a canonical prefLabel norm; demote variant punctuation norm to altLabel
                            best_kind = "skos:altLabel"
                        else:
                            # Demote stale non-name prefLabel to altLabel and promote canonical name norm
                            conn.execute(
                                """
                                UPDATE entity_labels
                                SET label_kind = 'skos:altLabel'
                                WHERE entity_id = ? AND label_norm = ? AND label_kind = 'skos:prefLabel'
                                """,
                                (entity_id, curr_pref),
                            )
                            existing_pref[entity_id] = norm
                            best_kind = "skos:prefLabel"
                    else:
                        existing_pref[entity_id] = norm
                        best_kind = "skos:prefLabel"
                else:
                    # Non-canonical label (label_norm != text_processing(entities.name)):
                    # Strictly normalize across all chunk_ids to skos:altLabel or skos:hiddenLabel
                    # with skos:altLabel taking precedence over skos:hiddenLabel.
                    candidate_kind = (
                        "skos:altLabel"
                        if label_kind in ("skos:prefLabel", "skos:altLabel")
                        else "skos:hiddenLabel"
                    )
                    if candidate_kind == "skos:altLabel" or any(
                        ek in ("skos:prefLabel", "skos:altLabel") for ek in existing_kinds
                    ):
                        best_kind = "skos:altLabel"
                    else:
                        best_kind = "skos:hiddenLabel"
            else:
                effective_kind = label_kind
                if curr_pref is not None and curr_pref == norm:
                    effective_kind = "skos:prefLabel"
                elif effective_kind == "skos:prefLabel":
                    if curr_pref is not None and curr_pref != norm:
                        effective_kind = "skos:altLabel"
                    else:
                        existing_pref[entity_id] = norm
                best_kind = effective_kind
                for ek in existing_kinds:
                    if _LABEL_KIND_PRIORITY.get(ek, 0) > _LABEL_KIND_PRIORITY.get(best_kind, 0):
                        best_kind = ek

            conn.execute(
                """
                INSERT INTO entity_labels (label_norm, entity_id, label_kind, raw_label, chunk_id)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(label_norm, entity_id, chunk_id) DO UPDATE SET
                    label_kind = excluded.label_kind,
                    raw_label = CASE WHEN excluded.raw_label != '' THEN excluded.raw_label ELSE entity_labels.raw_label END
                """,
                (norm, entity_id, best_kind, raw_label or label_norm, chunk_id),
            )
            if existing_kinds and any(k != best_kind for k in existing_kinds):
                conn.execute(
                    "UPDATE entity_labels SET label_kind = ? WHERE entity_id = ? AND label_norm = ?",
                    (best_kind, entity_id, norm),
                )


def get_entities_by_label_norm(
    conn: sqlite3.Connection,
    label_norms: Sequence[str],
) -> dict[str, list[str]]:
    """Look up entity_ids matching each normalized label across skos:prefLabel, altLabel, and hiddenLabel."""
    if not label_norms:
        return {}
    clean_norms = list(dict.fromkeys(n.strip().lower() for n in label_norms if n.strip()))
    if not clean_norms:
        return {}
    results: dict[str, list[str]] = {n: [] for n in clean_norms}
    for batch in _batched(clean_norms):
        placeholders = ",".join("?" for _ in batch)
        cursor = conn.execute(
            f"""
            SELECT label_norm, entity_id,
                   MIN(CASE label_kind
                       WHEN 'skos:prefLabel' THEN 1
                       WHEN 'skos:altLabel' THEN 2
                       ELSE 3
                   END) AS prio
            FROM entity_labels
            WHERE label_norm IN ({placeholders})
            GROUP BY label_norm, entity_id
            ORDER BY label_norm ASC, prio ASC, entity_id ASC
            """,
            tuple(batch),
        )
        for row in cursor:
            norm = str(row[0])
            eid = str(row[1])
            if eid not in results[norm]:
                results[norm].append(eid)
    return results


def resolve_entity_labels(
    conn: sqlite3.Connection,
    label_norms: Sequence[str],
    query_vec: np.ndarray | Sequence[float] | None = None,
) -> dict[str, list[str]]:
    """Resolve surface labels to entity_ids, disambiguating polysemous/homograph senses by query_vec cosine."""
    by_norm = get_entities_by_label_norm(conn, label_norms)
    if not by_norm:
        return {}
    if query_vec is None:
        return by_norm

    ambiguous_eids: list[str] = [eid for eids in by_norm.values() if len(eids) > 1 for eid in eids]
    if not ambiguous_eids:
        return by_norm

    q_arr = np.asarray(query_vec, dtype=np.float32)
    q_norm = float(np.linalg.norm(q_arr))
    if q_norm <= 1e-12:
        return by_norm

    vecs = get_entity_vectors(conn, list(dict.fromkeys(ambiguous_eids)))
    resolved: dict[str, list[str]] = {}
    for norm, eids in by_norm.items():
        if len(eids) <= 1:
            resolved[norm] = eids
            continue
        scored: list[tuple[float, str]] = []
        for eid in eids:
            v = vecs.get(eid)
            if v is not None and v.size == q_arr.size:
                v_norm = float(np.linalg.norm(v))
                sim = float(np.dot(q_arr, v) / (q_norm * v_norm)) if v_norm > 1e-12 else -1.0
            else:
                sim = -1.0
            scored.append((sim, eid))
        scored.sort(key=lambda item: (-item[0], item[1]))
        resolved[norm] = [scored[0][1]]
    return resolved


def merge_entities(
    conn: sqlite3.Connection,
    src_id: str,
    dst_id: str,
    label_kind: str = "skos:altLabel",
) -> None:
    """Atomically collapse src_id into dst_id, demoting src_id's prefLabel to label_kind and re-pointing all edges/facts."""
    if src_id == dst_id:
        return
    if label_kind not in ("skos:altLabel", "skos:hiddenLabel"):
        raise ValueError(
            f"merge_entities label_kind must be skos:altLabel or skos:hiddenLabel, got {label_kind!r}"
        )

    with _write_tx(conn):
        src_row = conn.execute(
            "SELECT name, definition FROM entities WHERE entity_id = ?", (src_id,)
        ).fetchone()
        dst_row = conn.execute(
            "SELECT name, definition FROM entities WHERE entity_id = ?", (dst_id,)
        ).fetchone()
        if src_row is None or dst_row is None:
            return

        src_name, src_def = str(src_row[0]), str(src_row[1])
        dst_name, dst_def = str(dst_row[0]), str(dst_row[1])
        dst_pref_norm = dst_name.strip().lower()

        if not dst_def.strip() and src_def.strip():
            conn.execute(
                "UPDATE entities SET definition = ? WHERE entity_id = ?",
                (src_def, dst_id),
            )

        # 1. Re-point and demote src_id labels onto dst_id while maintaining SKOS S13
        src_labels = conn.execute(
            "SELECT label_norm, label_kind, raw_label, chunk_id FROM entity_labels WHERE entity_id = ?",
            (src_id,),
        ).fetchall()
        conn.execute("DELETE FROM entity_labels WHERE entity_id = ?", (src_id,))
        labels_to_upsert: list[tuple[str, str, str, str, str]] = []
        if src_labels:
            for l_norm, l_kind, raw_lbl, cid in src_labels:
                norm_str = str(l_norm)
                new_kind = (
                    "skos:prefLabel"
                    if norm_str == dst_pref_norm
                    else (label_kind if str(l_kind) == "skos:prefLabel" else str(l_kind))
                )
                labels_to_upsert.append((norm_str, dst_id, new_kind, str(raw_lbl), str(cid)))
        else:
            norm_str = src_name.strip().lower()
            new_kind = "skos:prefLabel" if norm_str == dst_pref_norm else label_kind
            labels_to_upsert.append((norm_str, dst_id, new_kind, src_name, "__merged__"))
        upsert_entity_labels(conn, labels_to_upsert)

        # 2. Re-point facts (subj_id and obj_id) and prune self-loop facts
        conn.execute("UPDATE facts SET subj_id = ? WHERE subj_id = ?", (dst_id, src_id))
        conn.execute("UPDATE facts SET obj_id = ? WHERE obj_id = ?", (dst_id, src_id))
        self_loop_fids = [
            str(r[0])
            for r in conn.execute(
                "SELECT fact_id FROM facts WHERE subj_id = ? AND obj_id = ?",
                (dst_id, dst_id),
            ).fetchall()
        ]
        for fid in self_loop_fids:
            conn.execute("DELETE FROM fact_vec WHERE fact_id = ?", (fid,))
            conn.execute("DELETE FROM fact_fts WHERE fact_id = ?", (fid,))
            conn.execute("DELETE FROM facts WHERE fact_id = ?", (fid,))

        # 3. Re-point edge_sources
        conn.execute(
            """
            INSERT OR IGNORE INTO edge_sources (src_id, dst_id, kind, chunk_id)
            SELECT
                CASE WHEN src_id = ? THEN ? ELSE src_id END,
                CASE WHEN dst_id = ? THEN ? ELSE dst_id END,
                kind,
                chunk_id
            FROM edge_sources
            WHERE (src_id = ? OR dst_id = ?)
              AND (CASE WHEN src_id = ? THEN ? ELSE src_id END) != (CASE WHEN dst_id = ? THEN ? ELSE dst_id END)
            """,
            (src_id, dst_id, src_id, dst_id, src_id, src_id, src_id, dst_id, src_id, dst_id),
        )
        conn.execute(
            "DELETE FROM edge_sources WHERE src_id = ? OR dst_id = ?",
            (src_id, src_id),
        )

        # 4. Re-point edges (excluding self-loops)
        incident_edges = conn.execute(
            "SELECT src_id, dst_id, kind, weight FROM edges WHERE src_id = ? OR dst_id = ?",
            (src_id, src_id),
        ).fetchall()
        conn.execute(
            "DELETE FROM edges WHERE src_id = ? OR dst_id = ?",
            (src_id, src_id),
        )
        for e_src, e_dst, e_kind, e_weight in incident_edges:
            new_src = dst_id if str(e_src) == src_id else str(e_src)
            new_dst = dst_id if str(e_dst) == src_id else str(e_dst)
            kind_str = str(e_kind)
            weight_val = float(e_weight)
            if new_src == new_dst:
                continue
            if kind_str == "passage":
                conn.execute(
                    """
                    INSERT INTO edges (src_id, dst_id, kind, weight)
                    VALUES (?, ?, 'passage', ?)
                    ON CONFLICT(src_id, dst_id, kind) DO UPDATE SET
                        weight = MAX(edges.weight, excluded.weight)
                    """,
                    (new_src, new_dst, weight_val),
                )
            elif kind_str == "fact":
                prov_count = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM edge_sources WHERE src_id = ? AND dst_id = ? AND kind = 'fact'",
                        (new_src, new_dst),
                    ).fetchone()[0]
                )
                eff_weight = float(prov_count) if prov_count > 0 else weight_val
                conn.execute(
                    """
                    INSERT INTO edges (src_id, dst_id, kind, weight)
                    VALUES (?, ?, 'fact', ?)
                    ON CONFLICT(src_id, dst_id, kind) DO UPDATE SET
                        weight = CASE WHEN ? > 0 THEN excluded.weight ELSE edges.weight + excluded.weight END
                    """,
                    (new_src, new_dst, eff_weight, prov_count),
                )
            elif kind_str in ("skos:related", "owl:differentFrom"):
                upsert_semantic_edges(conn, [(new_src, new_dst, kind_str, weight_val)])

        # 5. Remove src_id from entity_vec and entities
        conn.execute("DELETE FROM entity_vec WHERE entity_id = ?", (src_id,))
        conn.execute("DELETE FROM entities WHERE entity_id = ?", (src_id,))


def upsert_facts(
    conn: sqlite3.Connection,
    facts: Sequence[tuple[str, str, str, str, str] | tuple[str, str, str, str, str, str]],
) -> None:
    """Upsert fact nodes (fact_id, subj_id, pred, obj_id, raw_triple_json[, definition]) into facts and index in fact_fts."""
    if not facts:
        return
    normalized_facts: list[tuple[str, str, str, str, str, str]] = []
    for item in facts:
        if len(item) == 5:
            normalized_facts.append((item[0], item[1], item[2], item[3], item[4], ""))
        else:
            normalized_facts.append((item[0], item[1], item[2], item[3], item[4], item[5]))
    with _write_tx(conn):
        conn.executemany(
            """
            INSERT INTO facts (fact_id, subj_id, pred, obj_id, raw_triple_json, definition)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(fact_id) DO UPDATE SET
                subj_id = excluded.subj_id,
                pred = excluded.pred,
                obj_id = excluded.obj_id,
                raw_triple_json = excluded.raw_triple_json,
                definition = CASE
                    WHEN TRIM(excluded.definition) != '' THEN excluded.definition
                    ELSE facts.definition
                END
            """,
            normalized_facts,
        )
        conn.executemany(
            "DELETE FROM fact_fts WHERE fact_id = ?",
            ((fid,) for fid, _, _, _, _, _ in normalized_facts),
        )
        fts_items: list[tuple[str, str]] = []
        for fid, _, pred, _, raw_json, _ in normalized_facts:
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
        fact_pairs = {(src, dst) for src, dst, kind, _, _ in edges_with_sources if kind == "fact"}
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


def get_facts_with_meta_by_ids(
    conn: sqlite3.Connection,
    fact_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Retrieve enriched fact records with statement, subject, and object definitions."""
    if not fact_ids:
        return {}
    results: dict[str, dict[str, Any]] = {}
    for batch in _batched(fact_ids):
        placeholders = ",".join("?" for _ in batch)
        cursor = conn.execute(
            f"""
            SELECT
                f.fact_id,
                f.subj_id,
                f.pred,
                f.obj_id,
                f.raw_triple_json,
                f.definition,
                COALESCE(es.definition, '') AS subj_def,
                COALESCE(eo.definition, '') AS obj_def
            FROM facts f
            LEFT JOIN entities es ON es.entity_id = f.subj_id
            LEFT JOIN entities eo ON eo.entity_id = f.obj_id
            WHERE f.fact_id IN ({placeholders})
            """,
            tuple(batch),
        )
        for row in cursor:
            fid = str(row[0])
            raw_list = json.loads(row[4])
            results[fid] = {
                "fact_id": fid,
                "subj_id": str(row[1]),
                "pred": str(row[2]),
                "obj_id": str(row[3]),
                "triple": (str(raw_list[0]), str(raw_list[1]), str(raw_list[2])),
                "definition": str(row[5]),
                "subj_def": str(row[6]),
                "obj_def": str(row[7]),
            }
    return results


def inspect_focus_node(
    conn: sqlite3.Connection,
    focus_node: str,
) -> dict[str, Any]:
    """Read-only neighborhood inspection tool for RLM synonym judgment and SHACL healing."""
    ent_row = conn.execute(
        "SELECT entity_id, name, definition FROM entities WHERE entity_id = ? OR name = ? LIMIT 1",
        (focus_node, focus_node),
    ).fetchone()
    eid = str(ent_row[0]) if ent_row is not None else focus_node
    entity_info = (
        {"entity_id": str(ent_row[0]), "name": str(ent_row[1]), "definition": str(ent_row[2])}
        if ent_row is not None
        else None
    )

    labels = [
        {
            "label_norm": str(r[0]),
            "label_kind": str(r[1]),
            "raw_label": str(r[2]),
            "chunk_id": str(r[3]),
        }
        for r in conn.execute(
            "SELECT label_norm, label_kind, raw_label, chunk_id FROM entity_labels WHERE entity_id = ?",
            (eid,),
        )
    ]

    facts = [
        {
            "fact_id": str(r[0]),
            "subj_id": str(r[1]),
            "pred": str(r[2]),
            "obj_id": str(r[3]),
            "raw_triple": json.loads(r[4]),
            "definition": str(r[5]),
        }
        for r in conn.execute(
            """
            SELECT fact_id, subj_id, pred, obj_id, raw_triple_json, definition
            FROM facts
            WHERE fact_id = ? OR subj_id = ? OR obj_id = ?
            """,
            (focus_node, eid, eid),
        )
    ]

    edges = [
        {
            "src_id": str(r[0]),
            "dst_id": str(r[1]),
            "kind": str(r[2]),
            "weight": float(r[3]),
        }
        for r in conn.execute(
            "SELECT src_id, dst_id, kind, weight FROM edges WHERE src_id IN (?, ?) OR dst_id IN (?, ?)",
            (focus_node, eid, focus_node, eid),
        )
    ]

    chunk_ids_set: set[str] = {
        item["chunk_id"]
        for item in labels
        if item["chunk_id"] and not item["chunk_id"].startswith("__")
    }
    for r in conn.execute(
        "SELECT DISTINCT chunk_id FROM edge_sources WHERE src_id IN (?, ?) OR dst_id IN (?, ?)",
        (focus_node, eid, focus_node, eid),
    ):
        chunk_ids_set.add(str(r[0]))
    for e in edges:
        if e["kind"] == "passage" and e["dst_id"] == eid:
            chunk_ids_set.add(str(e["src_id"]))

    return {
        "focus_node": focus_node,
        "entity": entity_info,
        "labels": labels,
        "facts": facts,
        "edges": edges,
        "chunk_ids": sorted(chunk_ids_set),
    }


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
        WHERE src_id != dst_id AND weight > 0.0 AND kind != 'owl:differentFrom'
        GROUP BY src_id, dst_id
        """
    )
    edges = [(str(row[0]), str(row[1]), float(row[2])) for row in edge_cursor]

    return chunk_ids, entity_ids, edges


def _are_same_label_homographs(conn: sqlite3.Connection, ent1: str, ent2: str) -> bool:
    """Return True if ent1 and ent2 share a skos:prefLabel norm or normalized entity name (e.g. 'a ball' vs 'a ball')."""
    row = conn.execute(
        """
        SELECT 1
        FROM entities a
        INNER JOIN entities b ON a.entity_id = ? AND b.entity_id = ?
        WHERE LOWER(TRIM(a.name)) != '' AND LOWER(TRIM(a.name)) = LOWER(TRIM(b.name))
        UNION ALL
        SELECT 1
        FROM entity_labels la
        INNER JOIN entity_labels lb
          ON la.label_norm = lb.label_norm
         AND la.label_kind = 'skos:prefLabel'
         AND lb.label_kind = 'skos:prefLabel'
        WHERE la.entity_id = ? AND lb.entity_id = ?
        LIMIT 1
        """,
        (ent1, ent2, ent1, ent2),
    ).fetchone()
    return row is not None


def upsert_semantic_edges(
    conn: sqlite3.Connection,
    edges: Sequence[tuple[str, str, str, float]],
) -> None:
    """Bulk upsert symmetric skos:related and owl:differentFrom edges (allowing both on same-label homographs like 'a ball' vs 'a ball')."""
    if not edges:
        return
    with _write_tx(conn):
        for ent1, ent2, kind, weight in edges:
            if ent1 == ent2:
                continue
            if kind not in ("skos:related", "owl:differentFrom"):
                raise ValueError(f"Unsupported semantic edge kind: {kind!r}")
            is_homograph = _are_same_label_homographs(conn, ent1, ent2)
            if kind == "owl:differentFrom":
                if not is_homograph:
                    conn.execute(
                        """
                        DELETE FROM edges
                        WHERE kind = 'skos:related'
                          AND ((src_id = ? AND dst_id = ?) OR (src_id = ? AND dst_id = ?))
                        """,
                        (ent1, ent2, ent2, ent1),
                    )
                for s, d in ((ent1, ent2), (ent2, ent1)):
                    conn.execute(
                        """
                        INSERT INTO edges (src_id, dst_id, kind, weight)
                        VALUES (?, ?, 'owl:differentFrom', 0.0)
                        ON CONFLICT(src_id, dst_id, kind) DO UPDATE SET weight = 0.0
                        """,
                        (s, d),
                    )
            else:
                has_diff = (
                    None
                    if is_homograph
                    else conn.execute(
                        """
                        SELECT 1 FROM edges
                        WHERE kind = 'owl:differentFrom'
                          AND ((src_id = ? AND dst_id = ?) OR (src_id = ? AND dst_id = ?))
                        LIMIT 1
                        """,
                        (ent1, ent2, ent2, ent1),
                    ).fetchone()
                )
                if has_diff is None:
                    rel_weight = max(float(weight), 1e-6)
                    for s, d in ((ent1, ent2), (ent2, ent1)):
                        conn.execute(
                            """
                            INSERT INTO edges (src_id, dst_id, kind, weight)
                            VALUES (?, ?, 'skos:related', ?)
                            ON CONFLICT(src_id, dst_id, kind) DO UPDATE SET weight = excluded.weight
                            """,
                            (s, d, rel_weight),
                        )
                    if is_homograph:
                        for s, d in ((ent1, ent2), (ent2, ent1)):
                            conn.execute(
                                """
                                INSERT INTO edges (src_id, dst_id, kind, weight)
                                VALUES (?, ?, 'owl:differentFrom', 0.0)
                                ON CONFLICT(src_id, dst_id, kind) DO UPDATE SET weight = 0.0
                                """,
                                (s, d),
                            )


def upsert_synonymy_edges(
    conn: sqlite3.Connection,
    edges: Sequence[tuple[str, str, float]],
) -> None:
    """Bulk upsert bidirectional skos:related semantic edges between entity pairs."""
    if not edges:
        return
    upsert_semantic_edges(
        conn,
        [(ent1, ent2, "skos:related", sim) for ent1, ent2, sim in edges],
    )


# ---------------------------------------------------------------------------
# Closed-World SQL SHACL Linter & W3C Turtle Serializer
# ---------------------------------------------------------------------------


def _scan_shacl_violations(conn: sqlite3.Connection) -> list[SHACLViolation]:
    violations: list[SHACLViolation] = []

    # 1. ConceptShape/prefLabel: Every entity must have exactly 1 distinct skos:prefLabel norm
    for row in conn.execute(
        """
        SELECT
            e.entity_id,
            COUNT(DISTINCT CASE WHEN el.label_kind = 'skos:prefLabel' THEN el.label_norm END) AS pref_cnt
        FROM entities e
        LEFT JOIN entity_labels el ON el.entity_id = e.entity_id
        GROUP BY e.entity_id
        HAVING pref_cnt != 1
        ORDER BY e.entity_id ASC
        """
    ):
        eid, cnt = str(row[0]), int(row[1])
        msg = (
            f"Entity {eid} has no skos:prefLabel in entity_labels."
            if cnt == 0
            else f"Entity {eid} has {cnt} distinct skos:prefLabel norms (expected 1)."
        )
        violations.append(
            SHACLViolation(
                shape="ConceptShape/prefLabel",
                focus_node=eid,
                result_path="skos:prefLabel",
                message=msg,
            )
        )

    # 2. ConceptShape/definition: Every entity must have a non-empty definition
    for row in conn.execute(
        """
        SELECT entity_id, name
        FROM entities
        WHERE TRIM(definition) = ''
        ORDER BY entity_id ASC
        """
    ):
        eid, name = str(row[0]), str(row[1])
        violations.append(
            SHACLViolation(
                shape="ConceptShape/definition",
                focus_node=eid,
                result_path="skos:definition",
                message=f"Entity {eid} ({name!r}) has an empty skos:definition.",
            )
        )

    # 3. ConceptShape/S13-disjointLabels:
    # (a) No (entity_id, label_norm) pair may have >1 distinct label_kind across chunks
    #     (covering skos:prefLabel ∩ (skos:altLabel ∪ skos:hiddenLabel) and skos:altLabel ∩ skos:hiddenLabel)
    # (b) No entity's canonical name (LOWER(TRIM(e.name))) may appear as skos:altLabel or skos:hiddenLabel
    seen_s13: set[tuple[str, str]] = set()
    for row in conn.execute(
        """
        SELECT entity_id, label_norm, COUNT(DISTINCT label_kind) AS kind_cnt
        FROM entity_labels
        GROUP BY entity_id, label_norm
        HAVING kind_cnt > 1
        ORDER BY entity_id ASC, label_norm ASC
        """
    ):
        eid, lnorm = str(row[0]), str(row[1])
        seen_s13.add((eid, lnorm))
        violations.append(
            SHACLViolation(
                shape="ConceptShape/S13-disjointLabels",
                focus_node=eid,
                result_path="skos:prefLabel|skos:altLabel|skos:hiddenLabel",
                message=f"Entity {eid} label {lnorm!r} violates SKOS S13 disjointness across chunks.",
            )
        )
    for row in conn.execute(
        """
        SELECT DISTINCT el.entity_id, el.label_norm
        FROM entity_labels el
        INNER JOIN entities e ON e.entity_id = el.entity_id
        WHERE el.label_kind IN ('skos:altLabel', 'skos:hiddenLabel')
          AND el.label_norm = LOWER(TRIM(e.name))
        ORDER BY el.entity_id ASC, el.label_norm ASC
        """
    ):
        eid, lnorm = str(row[0]), str(row[1])
        if (eid, lnorm) not in seen_s13:
            seen_s13.add((eid, lnorm))
            violations.append(
                SHACLViolation(
                    shape="ConceptShape/S13-disjointLabels",
                    focus_node=eid,
                    result_path="skos:prefLabel|skos:altLabel|skos:hiddenLabel",
                    message=f"Entity {eid} canonical name {lnorm!r} appears as non-prefLabel in entity_labels.",
                )
            )

    # 4. ConceptShape/homographDifferentFrom: Distinct entities sharing a skos:prefLabel norm must have owl:differentFrom
    for row in conn.execute(
        """
        SELECT a.entity_id, b.entity_id, a.label_norm
        FROM entity_labels a
        INNER JOIN entity_labels b
          ON a.label_norm = b.label_norm
         AND a.label_kind = 'skos:prefLabel'
         AND b.label_kind = 'skos:prefLabel'
         AND a.entity_id < b.entity_id
        WHERE NOT EXISTS (
            SELECT 1 FROM edges e
            WHERE e.src_id = a.entity_id
              AND e.dst_id = b.entity_id
              AND e.kind = 'owl:differentFrom'
        )
        GROUP BY a.entity_id, b.entity_id, a.label_norm
        ORDER BY a.entity_id ASC, b.entity_id ASC
        """
    ):
        ea, eb, lnorm = str(row[0]), str(row[1]), str(row[2])
        violations.append(
            SHACLViolation(
                shape="ConceptShape/homographDifferentFrom",
                focus_node=f"{ea}|{eb}",
                result_path="owl:differentFrom",
                message=f"Homograph concepts {ea} and {eb} share prefLabel {lnorm!r} without owl:differentFrom.",
            )
        )

    # 5. StatementShape/closedWorld: Facts must have valid subj_id/obj_id in entities and non-empty definition
    for row in conn.execute(
        """
        SELECT
            f.fact_id,
            f.subj_id,
            f.obj_id,
            f.definition,
            es.entity_id IS NOT NULL AS has_subj,
            eo.entity_id IS NOT NULL AS has_obj
        FROM facts f
        LEFT JOIN entities es ON es.entity_id = f.subj_id
        LEFT JOIN entities eo ON eo.entity_id = f.obj_id
        WHERE es.entity_id IS NULL OR eo.entity_id IS NULL OR TRIM(f.definition) = ''
        ORDER BY f.fact_id ASC
        """
    ):
        fid, sid, oid, fdef, has_s, has_o = (
            str(row[0]),
            str(row[1]),
            str(row[2]),
            str(row[3]),
            bool(row[4]),
            bool(row[5]),
        )
        if not has_s or not has_o:
            violations.append(
                SHACLViolation(
                    shape="StatementShape/closedWorld",
                    focus_node=fid,
                    result_path="rdf:subject|rdf:object",
                    message=f"Fact {fid} references missing entity endpoint ({sid} -> {oid}).",
                )
            )
        if not fdef.strip():
            violations.append(
                SHACLViolation(
                    shape="StatementShape/closedWorld",
                    focus_node=fid,
                    result_path="skos:definition",
                    message=f"Fact {fid} has an empty statement definition.",
                )
            )

    # 6. EdgeShape/closedWorld: Endpoints exist, no skos:related + owl:differentFrom collision, symmetric semantic edges
    for row in conn.execute(
        """
        SELECT e.src_id, e.dst_id, e.kind
        FROM edges e
        WHERE (
            e.kind = 'passage'
            AND NOT (
                (
                    EXISTS (SELECT 1 FROM chunks c WHERE c.chunk_id = e.src_id)
                    AND EXISTS (SELECT 1 FROM entities ent WHERE ent.entity_id = e.dst_id)
                )
                OR (
                    EXISTS (SELECT 1 FROM entities ent WHERE ent.entity_id = e.src_id)
                    AND EXISTS (SELECT 1 FROM chunks c WHERE c.chunk_id = e.dst_id)
                )
            )
        )
        OR (
            e.kind IN ('fact', 'skos:related', 'owl:differentFrom')
            AND (
                NOT EXISTS (SELECT 1 FROM entities ent1 WHERE ent1.entity_id = e.src_id)
                OR NOT EXISTS (SELECT 1 FROM entities ent2 WHERE ent2.entity_id = e.dst_id)
            )
        )
        ORDER BY e.src_id ASC, e.dst_id ASC, e.kind ASC
        """
    ):
        s, d, k = str(row[0]), str(row[1]), str(row[2])
        violations.append(
            SHACLViolation(
                shape="EdgeShape/closedWorld",
                focus_node=f"{s}->{d}",
                result_path=k,
                message=f"Edge ({s}, {d}, {k}) has a dangling endpoint outside closed-world tables.",
            )
        )

    for row in conn.execute(
        """
        SELECT a.src_id, a.dst_id
        FROM edges a
        INNER JOIN edges b
          ON a.src_id = b.src_id
         AND a.dst_id = b.dst_id
         AND a.kind = 'skos:related'
         AND b.kind = 'owl:differentFrom'
        WHERE a.src_id < a.dst_id
          AND NOT EXISTS (
              SELECT 1 FROM entities ea
              INNER JOIN entities eb ON ea.entity_id = a.src_id AND eb.entity_id = a.dst_id
              WHERE LOWER(TRIM(ea.name)) != '' AND LOWER(TRIM(ea.name)) = LOWER(TRIM(eb.name))
          )
          AND NOT EXISTS (
              SELECT 1 FROM entity_labels la
              INNER JOIN entity_labels lb
                ON la.label_norm = lb.label_norm
               AND la.label_kind = 'skos:prefLabel'
               AND lb.label_kind = 'skos:prefLabel'
              WHERE la.entity_id = a.src_id AND lb.entity_id = a.dst_id
          )
        ORDER BY a.src_id ASC, a.dst_id ASC
        """
    ):
        s, d = str(row[0]), str(row[1])
        violations.append(
            SHACLViolation(
                shape="EdgeShape/closedWorld",
                focus_node=f"{s}|{d}",
                result_path="skos:related|owl:differentFrom",
                message=f"Node pair ({s}, {d}) has conflicting skos:related and owl:differentFrom edges.",
            )
        )

    for row in conn.execute(
        """
        SELECT a.src_id, a.dst_id, a.kind
        FROM edges a
        WHERE a.kind IN ('skos:related', 'owl:differentFrom')
          AND NOT EXISTS (
              SELECT 1 FROM edges b
              WHERE b.src_id = a.dst_id AND b.dst_id = a.src_id AND b.kind = a.kind
          )
        ORDER BY a.src_id ASC, a.dst_id ASC, a.kind ASC
        """
    ):
        s, d, k = str(row[0]), str(row[1]), str(row[2])
        violations.append(
            SHACLViolation(
                shape="EdgeShape/closedWorld",
                focus_node=f"{s}->{d}",
                result_path=k,
                message=f"Asymmetric semantic edge ({s}, {d}, {k}) missing reverse direction.",
            )
        )

    return violations


def lint_graph_shacl(
    conn: sqlite3.Connection,
    *,
    auto_heal: bool = False,
) -> list[SHACLViolation]:
    """Run closed-world SQL SHACL validation shapes over yaw.db and optionally apply deterministic SQL repairs."""
    violations = _scan_shacl_violations(conn)
    if not violations or not auto_heal:
        return violations

    with _write_tx(conn):
        # Heal 1: ConceptShape/prefLabel
        missing_pref = conn.execute(
            """
            SELECT e.entity_id, e.name
            FROM entities e
            WHERE NOT EXISTS (
                SELECT 1 FROM entity_labels el
                WHERE el.entity_id = e.entity_id AND el.label_kind = 'skos:prefLabel'
            )
            """
        ).fetchall()
        for r in missing_pref:
            eid, ename = str(r[0]), str(r[1])
            alt_row = conn.execute(
                """
                SELECT label_norm, raw_label
                FROM entity_labels
                WHERE entity_id = ?
                ORDER BY CASE label_kind WHEN 'skos:altLabel' THEN 1 ELSE 2 END ASC, label_norm ASC
                LIMIT 1
                """,
                (eid,),
            ).fetchone()
            if alt_row is not None:
                lnorm, rlabel = str(alt_row[0]), str(alt_row[1])
                conn.execute(
                    "UPDATE entity_labels SET label_kind = 'skos:prefLabel' WHERE entity_id = ? AND label_norm = ?",
                    (eid, lnorm),
                )
                if rlabel:
                    conn.execute("UPDATE entities SET name = ? WHERE entity_id = ?", (rlabel, eid))
            else:
                norm = ename.strip().lower() or eid.lower()
                conn.execute(
                    """
                    INSERT OR IGNORE INTO entity_labels (label_norm, entity_id, label_kind, raw_label, chunk_id)
                    VALUES (?, ?, 'skos:prefLabel', ?, '__healed__')
                    """,
                    (norm, eid, ename or eid),
                )

        multi_pref = conn.execute(
            """
            SELECT e.entity_id, e.name
            FROM entities e
            INNER JOIN entity_labels el ON el.entity_id = e.entity_id AND el.label_kind = 'skos:prefLabel'
            GROUP BY e.entity_id
            HAVING COUNT(DISTINCT el.label_norm) > 1
            """
        ).fetchall()
        for r in multi_pref:
            eid, ename = str(r[0]), str(r[1])
            keep_norm = ename.strip().lower()
            norms = [
                str(nr[0])
                for nr in conn.execute(
                    """
                    SELECT DISTINCT label_norm FROM entity_labels
                    WHERE entity_id = ? AND label_kind = 'skos:prefLabel'
                    ORDER BY label_norm ASC
                    """,
                    (eid,),
                )
            ]
            winner = keep_norm if keep_norm in norms else norms[0]
            conn.execute(
                """
                UPDATE entity_labels
                SET label_kind = 'skos:altLabel'
                WHERE entity_id = ? AND label_kind = 'skos:prefLabel' AND label_norm != ?
                """,
                (eid, winner),
            )

        # Heal 2: ConceptShape/S13-disjointLabels
        s13_rows = conn.execute(
            """
            SELECT entity_id, label_norm
            FROM entity_labels
            GROUP BY entity_id, label_norm
            HAVING COUNT(DISTINCT label_kind) > 1
            """
        ).fetchall()
        for r in s13_rows:
            eid, lnorm = str(r[0]), str(r[1])
            kinds = [
                str(kr[0])
                for kr in conn.execute(
                    "SELECT DISTINCT label_kind FROM entity_labels WHERE entity_id = ? AND label_norm = ?",
                    (eid, lnorm),
                )
            ]
            best = max(kinds, key=lambda k: _LABEL_KIND_PRIORITY.get(k, 0))
            conn.execute(
                "UPDATE entity_labels SET label_kind = ? WHERE entity_id = ? AND label_norm = ?",
                (best, eid, lnorm),
            )

        # Heal 3: ConceptShape/homographDifferentFrom
        homographs = conn.execute(
            """
            SELECT a.entity_id, b.entity_id
            FROM entity_labels a
            INNER JOIN entity_labels b
              ON a.label_norm = b.label_norm
             AND a.label_kind = 'skos:prefLabel'
             AND b.label_kind = 'skos:prefLabel'
             AND a.entity_id < b.entity_id
            WHERE NOT EXISTS (
                SELECT 1 FROM edges e
                WHERE e.src_id = a.entity_id
                  AND e.dst_id = b.entity_id
                  AND e.kind = 'owl:differentFrom'
            )
            GROUP BY a.entity_id, b.entity_id
            """
        ).fetchall()
        if homographs:
            upsert_semantic_edges(
                conn,
                [(str(r[0]), str(r[1]), "owl:differentFrom", 0.0) for r in homographs],
            )

        # Heal 4: ConceptShape/definition
        empty_ent_defs = conn.execute(
            "SELECT entity_id, name FROM entities WHERE TRIM(definition) = ''"
        ).fetchall()
        for r in empty_ent_defs:
            eid, ename = str(r[0]), str(r[1])
            fact_row = conn.execute(
                "SELECT raw_triple_json, definition FROM facts WHERE subj_id = ? OR obj_id = ? LIMIT 1",
                (eid, eid),
            ).fetchone()
            if fact_row is not None and str(fact_row[1]).strip():
                synth_def = f"{ename}: {str(fact_row[1]).strip()}"
            elif fact_row is not None:
                try:
                    t = json.loads(fact_row[0])
                    synth_def = f"{ename} participates in relation: {t[0]} {t[1]} {t[2]}."
                except Exception:
                    synth_def = f"Concept representing {ename}."
            else:
                synth_def = f"Concept representing {ename}."
            conn.execute(
                "UPDATE entities SET definition = ? WHERE entity_id = ?",
                (synth_def, eid),
            )

        # Heal 5: StatementShape/closedWorld
        dangling_fids = [
            str(r[0])
            for r in conn.execute(
                """
                SELECT f.fact_id
                FROM facts f
                LEFT JOIN entities es ON es.entity_id = f.subj_id
                LEFT JOIN entities eo ON eo.entity_id = f.obj_id
                WHERE es.entity_id IS NULL OR eo.entity_id IS NULL
                """
            )
        ]
        if dangling_fids:
            for batch in _batched(dangling_fids):
                placeholders = ",".join("?" for _ in batch)
                conn.execute(
                    f"DELETE FROM fact_vec WHERE fact_id IN ({placeholders})", tuple(batch)
                )
                conn.execute(
                    f"DELETE FROM fact_fts WHERE fact_id IN ({placeholders})", tuple(batch)
                )
                conn.execute(f"DELETE FROM facts WHERE fact_id IN ({placeholders})", tuple(batch))

        empty_fact_defs = conn.execute(
            "SELECT fact_id, pred, raw_triple_json FROM facts WHERE TRIM(definition) = ''"
        ).fetchall()
        for r in empty_fact_defs:
            fid, pred, raw_json = str(r[0]), str(r[1]), str(r[2])
            try:
                t = json.loads(raw_json)
                synth_fdef = f"{t[0]} {t[1]} {t[2]}."
            except Exception:
                synth_fdef = f"Statement with predicate {pred}."
            conn.execute(
                "UPDATE facts SET definition = ? WHERE fact_id = ?",
                (synth_fdef, fid),
            )

        # Heal 6: EdgeShape/closedWorld
        conn.execute(
            """
            DELETE FROM edges
            WHERE (
                kind = 'passage'
                AND NOT (
                    (
                        EXISTS (SELECT 1 FROM chunks c WHERE c.chunk_id = edges.src_id)
                        AND EXISTS (SELECT 1 FROM entities ent WHERE ent.entity_id = edges.dst_id)
                    )
                    OR (
                        EXISTS (SELECT 1 FROM entities ent WHERE ent.entity_id = edges.src_id)
                        AND EXISTS (SELECT 1 FROM chunks c WHERE c.chunk_id = edges.dst_id)
                    )
                )
            )
            OR (
                kind IN ('fact', 'skos:related', 'owl:differentFrom')
                AND (
                    NOT EXISTS (SELECT 1 FROM entities ent1 WHERE ent1.entity_id = edges.src_id)
                    OR NOT EXISTS (SELECT 1 FROM entities ent2 WHERE ent2.entity_id = edges.dst_id)
                )
            )
            """
        )
        conn.execute(
            """
            DELETE FROM edges
            WHERE kind = 'skos:related'
              AND EXISTS (
                  SELECT 1 FROM edges b
                  WHERE b.src_id = edges.src_id
                    AND b.dst_id = edges.dst_id
                    AND b.kind = 'owl:differentFrom'
              )
              AND NOT EXISTS (
                  SELECT 1 FROM entities ea
                  INNER JOIN entities eb ON ea.entity_id = edges.src_id AND eb.entity_id = edges.dst_id
                  WHERE LOWER(TRIM(ea.name)) != '' AND LOWER(TRIM(ea.name)) = LOWER(TRIM(eb.name))
              )
              AND NOT EXISTS (
                  SELECT 1 FROM entity_labels la
                  INNER JOIN entity_labels lb
                    ON la.label_norm = lb.label_norm
                   AND la.label_kind = 'skos:prefLabel'
                   AND lb.label_kind = 'skos:prefLabel'
                  WHERE la.entity_id = edges.src_id AND lb.entity_id = edges.dst_id
              )
            """
        )
        asym_rows = conn.execute(
            """
            SELECT a.src_id, a.dst_id, a.kind, a.weight
            FROM edges a
            WHERE a.kind IN ('skos:related', 'owl:differentFrom')
              AND NOT EXISTS (
                  SELECT 1 FROM edges b
                  WHERE b.src_id = a.dst_id AND b.dst_id = a.src_id AND b.kind = a.kind
              )
            """
        ).fetchall()
        for r in asym_rows:
            conn.execute(
                "INSERT OR IGNORE INTO edges (src_id, dst_id, kind, weight) VALUES (?, ?, ?, ?)",
                (str(r[1]), str(r[0]), str(r[2]), float(r[3])),
            )

    return _scan_shacl_violations(conn)


def _escape_turtle_literal(value: str) -> str:
    escaped = (
        value.replace("\\", "\\\\").replace('"', '\\"').replace("\r", "\\r").replace("\n", "\\n")
    )
    return f'"{escaped}"'


def serialize_graph_turtle(
    conn: sqlite3.Connection,
    violations: Sequence[SHACLViolation] | None = None,
) -> str:
    """Serialize the yaw.db SKOS/RDF/OWL closed-world knowledge graph into W3C Turtle format."""
    if violations is None:
        violations = _scan_shacl_violations(conn)

    lines: list[str] = [
        "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
        "@prefix owl: <http://www.w3.org/2002/07/owl#> .",
        "@prefix skos: <http://www.w3.org/2004/02/skos/core#> .",
        "@prefix sh: <http://www.w3.org/ns/shacl#> .",
        "@prefix prov: <http://www.w3.org/ns/prov#> .",
        "@prefix yaw: <https://yaw.local/vocab#> .",
        "",
    ]

    labels_by_eid: dict[str, list[tuple[str, str, str]]] = {}
    for row in conn.execute(
        """
        SELECT entity_id, label_kind, raw_label, label_norm, chunk_id
        FROM entity_labels
        ORDER BY entity_id ASC, label_kind ASC, label_norm ASC
        """
    ):
        eid = str(row[0])
        lkind = str(row[1])
        lbl = str(row[2]) if str(row[2]) else str(row[3])
        cid = str(row[4])
        labels_by_eid.setdefault(eid, []).append((lkind, lbl, cid))

    sem_edges_by_eid: dict[str, list[tuple[str, str]]] = {}
    for row in conn.execute(
        """
        SELECT src_id, kind, dst_id
        FROM edges
        WHERE kind IN ('skos:related', 'owl:differentFrom')
        ORDER BY src_id ASC, kind ASC, dst_id ASC
        """
    ):
        sem_edges_by_eid.setdefault(str(row[0]), []).append((str(row[1]), str(row[2])))

    for row in conn.execute(
        "SELECT entity_id, name, definition FROM entities ORDER BY entity_id ASC"
    ):
        eid, ename, edef = str(row[0]), str(row[1]), str(row[2])
        pred_objs: list[str] = ["a skos:Concept"]
        seen_labels: set[tuple[str, str]] = set()
        prov_chunks: set[str] = set()
        for lkind, lbl, cid in labels_by_eid.get(eid, []):
            if (lkind, lbl) not in seen_labels:
                seen_labels.add((lkind, lbl))
                pred_objs.append(f"{lkind} {_escape_turtle_literal(lbl)}")
            if cid and not cid.startswith("__"):
                prov_chunks.add(cid)
        if not seen_labels:
            pred_objs.append(f"skos:prefLabel {_escape_turtle_literal(ename)}")
        if edef.strip():
            pred_objs.append(f"skos:definition {_escape_turtle_literal(edef.strip())}")
        for kind, dst_id in sem_edges_by_eid.get(eid, []):
            pred_objs.append(f"{kind} yaw:{dst_id}")
        for cid in sorted(prov_chunks):
            pred_objs.append(f"prov:wasDerivedFrom yaw:{cid}")

        lines.append(f"yaw:{eid} " + " ;\n    ".join(pred_objs) + " .")
        lines.append("")

    for row in conn.execute(
        "SELECT fact_id, subj_id, pred, obj_id, definition FROM facts ORDER BY fact_id ASC"
    ):
        fid, sid, pred, oid, fdef = (
            str(row[0]),
            str(row[1]),
            str(row[2]),
            str(row[3]),
            str(row[4]),
        )
        stmt_parts = [
            "a rdf:Statement",
            f"rdf:subject yaw:{sid}",
            f"rdf:predicate {_escape_turtle_literal(pred)}",
            f"rdf:object yaw:{oid}",
        ]
        if fdef.strip():
            stmt_parts.append(f"skos:definition {_escape_turtle_literal(fdef.strip())}")
        lines.append(f"yaw:{fid} " + " ;\n    ".join(stmt_parts) + " .")
        lines.append("")

    conforms_str = "true" if len(violations) == 0 else "false"
    report_parts = [
        "a sh:ValidationReport",
        f"sh:conforms {conforms_str}",
    ]
    for v in violations:
        report_parts.append(
            "sh:result [ "
            "a sh:ValidationResult ; "
            f"sh:sourceShape {_escape_turtle_literal(v.shape)} ; "
            f"sh:focusNode {_escape_turtle_literal(v.focus_node)} ; "
            f"sh:resultPath {_escape_turtle_literal(v.result_path)} ; "
            f"sh:resultMessage {_escape_turtle_literal(v.message)} "
            "]"
        )
    lines.append("yaw:ValidationReport " + " ;\n    ".join(report_parts) + " .")
    lines.append("")

    return "\n".join(lines).strip() + "\n"


# ---------------------------------------------------------------------------
# Atomic Scoped Cascading Chunk Deletion
# ---------------------------------------------------------------------------

_INIT_DEL_TEMP_TABLES: tuple[str, ...] = (
    "CREATE TEMP TABLE IF NOT EXISTS _del_chunks (chunk_id TEXT PRIMARY KEY)",
    """
    CREATE TEMP TABLE IF NOT EXISTS _affected_edges (
        src_id TEXT NOT NULL,
        dst_id TEXT NOT NULL,
        kind TEXT NOT NULL,
        PRIMARY KEY (src_id, dst_id, kind)
    )
    """,
    "CREATE TEMP TABLE IF NOT EXISTS _affected_entities (entity_id TEXT PRIMARY KEY)",
    "CREATE TEMP TABLE IF NOT EXISTS _orphan_facts (fact_id TEXT PRIMARY KEY)",
    "CREATE TEMP TABLE IF NOT EXISTS _orphan_entities (entity_id TEXT PRIMARY KEY)",
    "DELETE FROM temp._del_chunks",
    "DELETE FROM temp._affected_edges",
    "DELETE FROM temp._affected_entities",
    "DELETE FROM temp._orphan_facts",
    "DELETE FROM temp._orphan_entities",
)


def delete_chunks(conn: sqlite3.Connection, chunk_ids: Sequence[str]) -> None:
    """Atomically delete chunks and cascade removal strictly across affected edges, facts, labels, and entities."""
    if not chunk_ids:
        return

    with _write_tx(conn):
        for stmt in _INIT_DEL_TEMP_TABLES:
            conn.execute(stmt)

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
            UNION
            SELECT entity_id FROM entity_labels
            WHERE chunk_id IN (SELECT chunk_id FROM temp._del_chunks)
            """
        )
        # Include semantic-linked neighbors of affected entities so isolated aliases are also checked
        conn.execute(
            """
            INSERT OR IGNORE INTO temp._affected_entities (entity_id)
            SELECT dst_id FROM edges
            WHERE kind IN ('skos:related', 'owl:differentFrom')
              AND src_id IN (SELECT entity_id FROM temp._affected_entities)
            UNION
            SELECT src_id FROM edges
            WHERE kind IN ('skos:related', 'owl:differentFrom')
              AND dst_id IN (SELECT entity_id FROM temp._affected_entities)
            """
        )

        # 2. Remove chunk rows, label provenance, and edge provenance
        conn.executemany(
            "DELETE FROM chunk_fts WHERE chunk_id = ?",
            ((cid,) for cid in chunk_ids),
        )
        conn.execute(
            "DELETE FROM chunk_vec WHERE chunk_id IN (SELECT chunk_id FROM temp._del_chunks)"
        )
        conn.execute(
            "DELETE FROM openie_chunks WHERE chunk_id IN (SELECT chunk_id FROM temp._del_chunks)"
        )
        conn.execute("DELETE FROM chunks WHERE chunk_id IN (SELECT chunk_id FROM temp._del_chunks)")
        conn.execute(
            "DELETE FROM edge_sources WHERE chunk_id IN (SELECT chunk_id FROM temp._del_chunks)"
        )
        conn.execute(
            "DELETE FROM entity_labels WHERE chunk_id IN (SELECT chunk_id FROM temp._del_chunks)"
        )

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
        conn.execute(
            "DELETE FROM fact_vec WHERE fact_id IN (SELECT fact_id FROM temp._orphan_facts)"
        )
        conn.execute(
            "DELETE FROM fact_fts WHERE fact_id IN (SELECT fact_id FROM temp._orphan_facts)"
        )
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
        conn.execute(
            "DELETE FROM entity_labels WHERE entity_id IN (SELECT entity_id FROM temp._orphan_entities)"
        )
        conn.execute(
            "DELETE FROM entity_vec WHERE entity_id IN (SELECT entity_id FROM temp._orphan_entities)"
        )
        conn.execute(
            "DELETE FROM entities WHERE entity_id IN (SELECT entity_id FROM temp._orphan_entities)"
        )

        # 6. Prune semantic edges incident to deleted entities
        conn.execute(
            """
            DELETE FROM edges
            WHERE kind IN ('skos:related', 'owl:differentFrom')
              AND (
                  src_id IN (SELECT entity_id FROM temp._orphan_entities)
                  OR dst_id IN (SELECT entity_id FROM temp._orphan_entities)
              )
            """
        )

        # 7. Promote a surviving label to skos:prefLabel for any surviving affected entity that lost its prefLabel
        surviving_missing_pref = conn.execute(
            """
            SELECT e.entity_id, e.name
            FROM entities e
            INNER JOIN temp._affected_entities ae ON ae.entity_id = e.entity_id
            WHERE NOT EXISTS (
                SELECT 1 FROM entity_labels el
                WHERE el.entity_id = e.entity_id AND el.label_kind = 'skos:prefLabel'
            )
            """
        ).fetchall()
        for r in surviving_missing_pref:
            eid, ename = str(r[0]), str(r[1])
            candidate_lbl = conn.execute(
                """
                SELECT label_norm, raw_label
                FROM entity_labels
                WHERE entity_id = ?
                ORDER BY CASE label_kind WHEN 'skos:altLabel' THEN 1 ELSE 2 END ASC, label_norm ASC
                LIMIT 1
                """,
                (eid,),
            ).fetchone()
            if candidate_lbl is not None:
                lnorm, rlabel = str(candidate_lbl[0]), str(candidate_lbl[1])
                conn.execute(
                    "UPDATE entity_labels SET label_kind = 'skos:prefLabel' WHERE entity_id = ? AND label_norm = ?",
                    (eid, lnorm),
                )
                if rlabel:
                    conn.execute("UPDATE entities SET name = ? WHERE entity_id = ?", (rlabel, eid))
                homograph_peers = [
                    str(hr[0])
                    for hr in conn.execute(
                        """
                        SELECT DISTINCT entity_id FROM entity_labels
                        WHERE label_norm = ? AND label_kind = 'skos:prefLabel' AND entity_id != ?
                        """,
                        (lnorm, eid),
                    ).fetchall()
                ]
                if homograph_peers:
                    upsert_semantic_edges(
                        conn,
                        [(eid, peer_id, "owl:differentFrom", 0.0) for peer_id in homograph_peers],
                    )
            else:
                surviving_chunk = conn.execute(
                    """
                    SELECT chunk_id FROM edge_sources
                    WHERE src_id = ? OR dst_id = ?
                    LIMIT 1
                    """,
                    (eid, eid),
                ).fetchone()
                cid = str(surviving_chunk[0]) if surviving_chunk is not None else "__survived__"
                conn.execute(
                    """
                    INSERT OR IGNORE INTO entity_labels (label_norm, entity_id, label_kind, raw_label, chunk_id)
                    VALUES (?, ?, 'skos:prefLabel', ?, ?)
                    """,
                    (ename.strip().lower(), eid, ename, cid),
                )
