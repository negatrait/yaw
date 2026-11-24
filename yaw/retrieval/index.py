"""Read-only RetrievalIndex loading and SQLite artifact integrity validation."""

from __future__ import annotations

import math
import sqlite3
from typing import Any

import numpy as np
from numpy.typing import NDArray

from yaw.core import YawRuntimeError

from .db import compute_db_fingerprints, snapshot_tx
from .graph import build_diffusion_operators, build_title_index, unpack_citation_graph_blob
from .hydration import build_chunk_pointers
from .rabitq import build_rotator
from .types import RetrievalIndex, empty_retrieval_index


def _fetch_index_rows(
    conn: sqlite3.Connection,
) -> tuple[int, int, str, str, list[tuple[Any, ...]], tuple[Any, ...], tuple[Any, ...]] | None:
    with snapshot_tx(conn, require_fts=False):
        if conn.execute("SELECT 1 FROM main.sqlite_master WHERE type='table' AND name='chunks'").fetchone() is None:
            return None
        n, dim, vec_sha, src_sha, rows = compute_db_fingerprints(conn)
        if n == 0:
            return None
        if (
            conn.execute("SELECT 1 FROM main.sqlite_master WHERE type='table' AND name='rabitq_index'").fetchone()
            is None
        ):
            raise YawRuntimeError("INDEX_NOT_READY", "Missing rabitq_index table in SQLite index.")
        if (
            conn.execute("SELECT 1 FROM main.sqlite_master WHERE type='table' AND name='citation_graph'").fetchone()
            is None
        ):
            raise YawRuntimeError("INDEX_NOT_READY", "Missing citation_graph table in SQLite index.")
        r_row = conn.execute(
            "SELECT n_chunks, dim, embedding_model, task_mode, vec_sha256, source_sha256, "
            "n_words, codes_blob, ip_norm_blob, mean_blob, d_eff_shannon "
            "FROM rabitq_index WHERE id = 1"
        ).fetchone()
        if r_row is None:
            raise YawRuntimeError("INDEX_NOT_READY", "Missing persisted row id=1 in rabitq_index.")
        g_row = conn.execute(
            "SELECT n_chunks, vec_sha256, source_sha256, length(doc_titles_blob), graph_blob "
            "FROM citation_graph WHERE id = 1"
        ).fetchone()
        if g_row is None:
            raise YawRuntimeError("INDEX_NOT_READY", "Missing persisted row id=1 in citation_graph.")
    return n, dim, vec_sha, src_sha, rows, tuple(r_row), tuple(g_row)


def _validate_rabitq_blobs(
    r_row: tuple[Any, ...],
    n: int,
    dim: int,
    vec_sha: str,
    src_sha: str,
    expected_embedding_model: str | None,
    expected_task_mode: str | None,
) -> tuple[NDArray[np.uint8], NDArray[np.float32], NDArray[np.float32], float]:
    r_n, r_dim, r_model, r_mode, r_vsha, r_ssha = r_row[:6]
    if (int(r_n), int(r_dim), str(r_vsha), str(r_ssha)) != (n, dim, vec_sha, src_sha):
        raise YawRuntimeError("INDEX_NOT_READY", "Stale rabitq_index fingerprint against chunks table.")
    if expected_embedding_model is not None and str(r_model) != expected_embedding_model:
        raise YawRuntimeError(
            "INDEX_NOT_READY", f"Embedding model mismatch in rabitq_index: {r_model!r} != {expected_embedding_model!r}."
        )
    if expected_task_mode is not None and str(r_mode) != expected_task_mode:
        raise YawRuntimeError(
            "INDEX_NOT_READY", f"Task mode mismatch in rabitq_index: {r_mode!r} != {expected_task_mode!r}."
        )
    n_words = int(r_row[6])
    expected_words = (dim + 63) // 64
    if n_words != expected_words:
        raise YawRuntimeError(
            "CONSISTENCY_ERROR", f"Invalid n_words={n_words} in rabitq_index (expected {expected_words})."
        )
    codes_bytes = bytes(r_row[7])
    ip_norm_bytes = bytes(r_row[8])
    mean_bytes = bytes(r_row[9])
    if len(codes_bytes) != n_words * 8 * n:
        raise YawRuntimeError("CONSISTENCY_ERROR", "Corrupt codes_blob length in rabitq_index.")
    if len(ip_norm_bytes) != n * 4:
        raise YawRuntimeError("CONSISTENCY_ERROR", "Corrupt ip_norm_blob length in rabitq_index.")
    if len(mean_bytes) != dim * 4:
        raise YawRuntimeError("CONSISTENCY_ERROR", "Corrupt mean_blob length in rabitq_index.")
    d_eff = float(r_row[10])
    if not math.isfinite(d_eff) or d_eff < 2.0:
        raise YawRuntimeError("CONSISTENCY_ERROR", f"Invalid d_eff_shannon={d_eff!r} in rabitq_index.")

    codes = np.frombuffer(codes_bytes, dtype=np.uint8).reshape(n_words * 8, n).copy()
    ip_norm = np.frombuffer(ip_norm_bytes, dtype="<f4").astype(np.float32, copy=True)
    mean_vec = np.frombuffer(mean_bytes, dtype="<f4").astype(np.float32, copy=True)
    if not np.all(np.isfinite(ip_norm)) or not np.all(ip_norm > 0.0) or not np.all(np.isfinite(mean_vec)):
        raise YawRuntimeError("CONSISTENCY_ERROR", "Non-finite or invalid numeric values in rabitq_index blobs.")
    return codes, ip_norm, mean_vec, d_eff


def build_retrieval_index(
    conn: sqlite3.Connection,
    *,
    expected_embedding_model: str | None = None,
    expected_task_mode: str | None = None,
) -> RetrievalIndex:
    """Load persisted 1-bit SoA RaBitQ codes and CitationGraph from SQLite into a RetrievalIndex."""
    fetched = _fetch_index_rows(conn)
    if fetched is None:
        return empty_retrieval_index()
    n, dim, vec_sha, src_sha, rows, r_row, g_row = fetched

    codes, ip_norm, mean_vec, d_eff = _validate_rabitq_blobs(
        r_row, n, dim, vec_sha, src_sha, expected_embedding_model, expected_task_mode
    )
    g_n, g_vsha, g_ssha = g_row[:3]
    if (int(g_n), str(g_vsha), str(g_ssha)) != (n, vec_sha, src_sha):
        raise YawRuntimeError("INDEX_NOT_READY", "Stale citation_graph fingerprint against chunks table.")
    if int(g_row[3]) <= 0:
        raise YawRuntimeError("CONSISTENCY_ERROR", "Empty doc_titles_blob in citation_graph.")

    chunk_ids, pointers, distinct_paths, path_idx_arr = build_chunk_pointers(rows)
    cg = unpack_citation_graph_blob(bytes(g_row[4]), path_idx_arr, len(distinct_paths))
    doc_titles = {dp: dt for dp, dt in zip(distinct_paths, cg.doc_titles, strict=True) if dt}
    title_idx = build_title_index(pointers, doc_titles, prop_nouns=cg.prop_nouns)
    diffusion_ops = build_diffusion_operators(cg, path_idx_arr, codes=codes, dim=dim, d_eff=d_eff)

    return RetrievalIndex(
        chunk_ids=tuple(chunk_ids),
        id_to_pos={cid: i for i, cid in enumerate(chunk_ids)},
        pointers=tuple(pointers),
        path_idx=path_idx_arr,
        distinct_paths=tuple(distinct_paths),
        dim=dim,
        mean_vec=mean_vec,
        d_eff=d_eff,
        codes=codes,
        ip_norm=ip_norm,
        rotator=build_rotator(dim),
        doc_titles=doc_titles,
        title_by_first_word=title_idx,
        citation_graph=cg,
        diffusion_ops=diffusion_ops,
    )
