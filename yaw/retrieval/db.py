"""Read-only WAL SQLite snapshot access, chunk fingerprinting, and text normalization."""

from __future__ import annotations

import hashlib
import re
import sqlite3
import threading
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from yaw.core import YawRuntimeError

from .types import CJK_RUN_RE, DASH_RE, POSS_RE, WS_RUN_RE

_CONN_LOCKS_GUARD = threading.Lock()
_CONN_LOCKS: dict[int, threading.RLock] = {}


def _get_conn_lock(conn: sqlite3.Connection) -> threading.RLock:
    cid = id(conn)
    with _CONN_LOCKS_GUARD:
        lock = _CONN_LOCKS.get(cid)
        if lock is None:
            lock = threading.RLock()
            _CONN_LOCKS[cid] = lock
        return lock


def open_readonly_db(db_path: Path | str) -> sqlite3.Connection:
    """Open a read-only URI SQLite connection with WAL-safe snapshot settings (no extensions)."""
    target = Path(db_path).resolve()
    if not target.is_file():
        raise YawRuntimeError("INDEX_NOT_READY", f"SQLite database does not exist: {target}")
    conn = sqlite3.connect(f"file:{target.as_posix()}?mode=ro", uri=True, check_same_thread=False, isolation_level=None)
    try:
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA cache_size=-65536")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA query_only=ON")
    except Exception:
        conn.close()
        raise
    return conn


@contextmanager
def snapshot_tx(conn: sqlite3.Connection, *, require_fts: bool = True) -> Generator[None]:
    """Acquire connection lock and wrap queries in an isolated SQLite read snapshot."""
    with _get_conn_lock(conn):
        if require_fts:
            present = {
                str(r[0])
                for r in conn.execute(
                    "SELECT name FROM main.sqlite_master WHERE type='table' AND name IN ('chunk_fts', 'chunk_vocab')"
                ).fetchall()
            }
            if "chunk_fts" not in present:
                raise YawRuntimeError("CONSISTENCY_ERROR", "Missing chunk_fts table in SQLite index.")
            if "chunk_vocab" not in present:
                raise YawRuntimeError("CONSISTENCY_ERROR", "Missing chunk_vocab table in SQLite index.")
        started_txn = not conn.in_transaction
        if started_txn:
            conn.execute("BEGIN")
        try:
            yield
        finally:
            if started_txn:
                conn.execute("ROLLBACK")


def normalize_fts_unicode61(text: str) -> str:
    """Casefold text and expand CJK runs into overlapping bigrams for unicode61 FTS5 indexing."""
    norm = WS_RUN_RE.sub(" ", text.casefold()).strip()
    if not norm or CJK_RUN_RE.search(norm) is None:
        return norm

    def _cjk_repl(m: re.Match[str]) -> str:
        run = m.group(0)
        if len(run) <= 1:
            return f" {run} "
        return " " + " ".join(run[i : i + 2] for i in range(len(run) - 1)) + " "

    return WS_RUN_RE.sub(" ", CJK_RUN_RE.sub(_cjk_repl, norm)).strip()


def norm_entity_text(text: str) -> str:
    """Normalize dashes and English possessive suffixes before lexical/entity matching."""
    return POSS_RE.sub("", DASH_RE.sub("-", text))


def fingerprint_rows(rows: Sequence[Any]) -> tuple[int, int, str, str]:
    """Compute deterministic SHA-256 fingerprints over ordered chunk embeddings and metadata."""
    n = len(rows)
    if n == 0:
        empty = hashlib.sha256(b"").hexdigest()
        return 0, 0, empty, empty
    first_raw = rows[0][1]
    if first_raw is None:
        raise YawRuntimeError("CONSISTENCY_ERROR", "Missing embedding_f16 blob in chunks table.")
    first_bytes = bytes(first_raw)
    if len(first_bytes) == 0 or (len(first_bytes) % 2) != 0:
        raise YawRuntimeError(
            "CONSISTENCY_ERROR", f"Invalid float16 embedding byte length {len(first_bytes)} in chunks table."
        )
    dim = len(first_bytes) // 2
    if dim <= 0:
        raise YawRuntimeError("CONSISTENCY_ERROR", "Embedding dimension must be positive.")

    vec_h = hashlib.sha256()
    src_h = hashlib.sha256()
    for r in rows:
        cid = int(r[0])
        raw_emb = r[1]
        if raw_emb is None:
            raise YawRuntimeError("CONSISTENCY_ERROR", f"Missing embedding_f16 blob for chunk_id={cid}.")
        ebytes = bytes(raw_emb)
        if len(ebytes) == 0 or (len(ebytes) % 2) != 0 or (len(ebytes) // 2) != dim:
            raise YawRuntimeError(
                "CONSISTENCY_ERROR",
                f"Mismatched float16 embedding byte length {len(ebytes)} for chunk_id={cid} (expected {dim * 2}).",
            )
        vec_h.update(cid.to_bytes(8, "little", signed=True))
        vec_h.update(ebytes)
        meta_str = f"{cid}|{r[2]}|{r[3]}|{r[4]}|{r[5]}|{r[6]}|{r[7]}|{r[8]}|{r[9]}|{r[10]}\n"
        src_h.update(meta_str.encode("utf-8"))
    return n, dim, vec_h.hexdigest(), src_h.hexdigest()


def compute_db_fingerprints(conn: sqlite3.Connection) -> tuple[int, int, str, str, list[tuple[Any, ...]]]:
    """Query ordered chunk rows and compute deterministic vector and metadata SHA-256 fingerprints."""
    rows = conn.execute(
        "SELECT chunk_id, embedding_f16, rel_path, source_hash, start_line, end_line, "
        "char_start, char_end, ast_island_start, ast_island_end, title "
        "FROM chunks ORDER BY chunk_id ASC"
    ).fetchall()
    n, dim, vec_sha, src_sha = fingerprint_rows(rows)
    return n, dim, vec_sha, src_sha, rows
