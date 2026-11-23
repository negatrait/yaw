"""1-bit SoA RaBitQ + tangent-steered cyclic citation diffusion retrieval and zero-copy disk hydration."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import struct
import threading
import unicodedata
import zlib
from collections.abc import Collection, Generator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy import special

from yaw.core import (
    ChunkPointer,
    HydratedPassage,
    YawRuntimeError,
    resolve_corpus_path,
)

__all__ = [
    "CitationGraph",
    "DiffusionOperators",
    "IndexArtifacts",
    "RetrievalIndex",
    "SearchResult",
    "build_retrieval_index",
    "compute_db_fingerprints",
    "compute_index_artifacts",
    "empty_retrieval_index",
    "expand_citation_neighbors",
    "format_evidence_route",
    "hydrate_passage",
    "hydrate_ranked_passages",
    "lexical_search",
    "normalize_fts_unicode61",
    "open_readonly_db",
    "search_index",
]

_CJK_RANGE = r"\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff\uac00-\ud7af"
_CJK_RUN_RE = re.compile(f"[{_CJK_RANGE}]+")
_WORD_TOK_RE = re.compile(rf"[^\W_{_CJK_RANGE}]+(?:[+#._-][^\W_{_CJK_RANGE}]+|[+]{{2}}|#)?")
_ASCII_NORM_TRANS = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"})
_WS_RUN_RE = re.compile(r"\s+")
_TITLE_TRAILING_SEP_RE = re.compile(r"\s+[-–—:]+\s*$")
_TITLE_TAG_PREFIX_RE = re.compile(r"^\[[^\]]+\]\s+(.+)$")
_DASH_RE = re.compile(r"[\u2010-\u2015\u2212]")
_POSS_RE = re.compile(r"(?<=[0-9a-zA-Z\u00c0-\u024f])['’]s\b")
_SENT_SPLIT_RE = re.compile(r"(?:(?<=[.!?;:\n])\s+|(?<=[。！？；])\s*)")

_CITATION_GRAPH_MAGIC = b"YCG1"
_CITATION_GRAPH_HDR_FMT = "<4s11I"
_MAX_HYDRATE_SPAN = 10_000
_SPECTRUM_SAMPLE_CAP = 2048
_RABITQ_SEED = 0x594157  # Fixed deterministic orthogonal rotator seed ("YAW").
_TANGENT_COS_CLIP = 0.9999
_TANGENT_RHO_CLIP = 0.999
_TANGENT_Z_MAX = math.sqrt(-math.log1p(-_TANGENT_RHO_CLIP * _TANGENT_RHO_CLIP))
_TANGENT_LUT_BINS = 16384

_BYTE_BITS_TBL: NDArray[np.float32] = (
    np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1, bitorder="little").astype(np.float32) * 2.0 - 1.0
)

_CONN_LOCKS_GUARD = threading.Lock()
_CONN_LOCKS: dict[int, threading.RLock] = {}
_CACHE_LOCK = threading.Lock()


def _get_conn_lock(conn: sqlite3.Connection) -> threading.RLock:
    cid = id(conn)
    with _CONN_LOCKS_GUARD:
        lock = _CONN_LOCKS.get(cid)
        if lock is None:
            lock = threading.RLock()
            _CONN_LOCKS[cid] = lock
        return lock


@dataclass(frozen=True, slots=True)
class _Rotator:
    """Lightweight deterministic orthogonal rotator for 1-bit SoA RaBitQ."""

    dim: int
    h64: NDArray[np.float32]
    signs: tuple[NDArray[np.float32], ...]
    perms: tuple[NDArray[np.intp], ...]
    tail_q: NDArray[np.float32] | None


@dataclass(frozen=True, slots=True)
class CitationGraph:
    """Compact int32 CSR title-reference and 1-bit PMI bipartite citation graph."""

    fi: NDArray[np.int32]
    fx: NDArray[np.int32]
    ri: NDArray[np.int32]
    rx: NDArray[np.int32]
    dc_indptr: NDArray[np.int32]
    dc_indices: NDArray[np.int32]
    pmi_indptr: NDArray[np.int32]
    pmi_indices: NDArray[np.int32]
    cpmi_indptr: NDArray[np.int32]
    cpmi_indices: NDArray[np.int32]
    pmi_terms: tuple[str, ...]
    prop_nouns: frozenset[str]
    doc_titles: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DiffusionOperators:
    """Static CSR citation/PMI diffusion operators and 16,384-bin Beta-tail LUT buffers."""

    d_own: sp.csr_array | None
    m_fwd_t: sp.csr_array | None
    r_rev: sp.csr_array | None
    d_id_t: sp.csr_array | None
    p2c: sp.csr_array | None
    c2p_t: sp.csr_array | None
    self_pmi: NDArray[np.float32] | None
    pmi_scale: NDArray[np.float32] | None
    inv_m: NDArray[np.float32]
    edge_f_rows: NDArray[np.int32] | None = None
    edge_f_cols: NDArray[np.int32] | None = None
    edge_c_ud: NDArray[np.float32] | None = None
    edge_inv_s_ud: NDArray[np.float32] | None = None
    edge_fwd_indptr: NDArray[np.int32] | None = None
    edge_t_order: NDArray[np.int32] | None = None
    edge_t_indptr: NDArray[np.int32] | None = None
    edge_t_indices: NDArray[np.int32] | None = None
    edge_beta_lut: NDArray[np.float64] | None = None
    edge_beta_lut_diff: NDArray[np.float64] | None = None
    edge_beta_scale: float = 0.0
    edge_beta_d_tan: float = 0.0
    edge_nd: int = 0


@dataclass(slots=True)
class RetrievalIndex:
    """Read-only M_FINAL_1B retrieval index with 1-bit SoA RaBitQ codes, citation graph, and caches."""

    chunk_ids: tuple[int, ...]
    id_to_pos: dict[int, int]
    pointers: tuple[ChunkPointer, ...]
    path_idx: NDArray[np.uint32]
    distinct_paths: tuple[str, ...]
    dim: int
    mean_vec: NDArray[np.float32]
    d_eff: float
    codes: NDArray[np.uint8]
    ip_norm: NDArray[np.float32]
    rotator: _Rotator | None
    doc_titles: dict[str, str]
    title_by_first_word: dict[str, tuple[int, dict[tuple[str, ...], tuple[int, ...]]]]
    citation_graph: CitationGraph | None
    diffusion_ops: DiffusionOperators | None
    vocab_doc_cache: dict[str, int] = field(default_factory=dict)
    resolved_path_cache: dict[tuple[Path, str], Path] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.chunk_ids)

    @property
    def num_pmi_terms(self) -> int:
        return len(self.citation_graph.pmi_terms) if self.citation_graph is not None else 0

    @property
    def num_citation_edges(self) -> int:
        return int(self.citation_graph.fx.size) if self.citation_graph is not None else 0


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Ranked chunk hits, log Bayes factor, and corpus spectral summary."""

    ranked: tuple[tuple[int, float], ...]
    log_bf: float
    max_chunk_score: float
    top1_cosine: float
    d_eff: float
    n_candidates: int


@dataclass(frozen=True, slots=True)
class IndexArtifacts:
    """Materialized 1-bit SoA RaBitQ and compressed CitationGraph blobs for SQLite persistence."""

    n_chunks: int
    dim: int
    vec_sha256: str
    source_sha256: str
    n_words: int
    codes_blob: bytes
    ip_norm_blob: bytes
    mean_blob: bytes
    d_eff_shannon: float
    doc_titles_blob: bytes
    graph_blob: bytes


def empty_retrieval_index() -> RetrievalIndex:
    """Return an empty zero-chunk RetrievalIndex."""
    return RetrievalIndex(
        chunk_ids=(),
        id_to_pos={},
        pointers=(),
        path_idx=np.zeros(0, dtype=np.uint32),
        distinct_paths=(),
        dim=0,
        mean_vec=np.zeros(0, dtype=np.float32),
        d_eff=0.0,
        codes=np.zeros((0, 0), dtype=np.uint8),
        ip_norm=np.zeros(0, dtype=np.float32),
        rotator=None,
        doc_titles={},
        title_by_first_word={},
        citation_graph=None,
        diffusion_ops=None,
    )


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
def _snapshot_tx(conn: sqlite3.Connection, *, require_fts: bool = True) -> Generator[None]:
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
    norm = _WS_RUN_RE.sub(" ", text.casefold()).strip()
    if not norm or _CJK_RUN_RE.search(norm) is None:
        return norm

    def _cjk_repl(m: re.Match[str]) -> str:
        run = m.group(0)
        if len(run) <= 1:
            return f" {run} "
        return " " + " ".join(run[i : i + 2] for i in range(len(run) - 1)) + " "

    return _WS_RUN_RE.sub(" ", _CJK_RUN_RE.sub(_cjk_repl, norm)).strip()


def _fingerprint_rows(rows: Sequence[Any]) -> tuple[int, int, str, str]:
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
    n, dim, vec_sha, src_sha = _fingerprint_rows(rows)
    return n, dim, vec_sha, src_sha, rows


def _make_hadamard(n: int) -> NDArray[np.float32]:
    h = np.array([[1.0]], dtype=np.float32)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return (h / math.sqrt(n)).astype(np.float32)


def _build_rotator(dim: int) -> _Rotator:
    rng = np.random.default_rng(_RABITQ_SEED + dim)
    h64 = _make_hadamard(64)
    signs = tuple(rng.choice([-1.0, 1.0], size=dim).astype(np.float32) for _ in range(3))
    perms = tuple(rng.permutation(dim) for _ in range(3))
    rem = dim % 64
    tail_q: NDArray[np.float32] | None = None
    if rem > 0:
        q_mat, r_mat = np.linalg.qr(rng.standard_normal((rem, rem)))
        diag = np.sign(np.diag(r_mat))
        diag[diag == 0.0] = 1.0
        tail_q = (q_mat * diag).astype(np.float32)
    return _Rotator(dim=dim, h64=h64, signs=signs, perms=perms, tail_q=tail_q)


def _rotate_vectors(rot: _Rotator, arr: NDArray[np.float32]) -> NDArray[np.float32]:
    is_1d = arr.ndim == 1
    y = arr[None, :].astype(np.float32, copy=True) if is_1d else arr.astype(np.float32, copy=True)
    n_blocks = rot.dim // 64
    head_end = n_blocks * 64
    for s, p in zip(rot.signs, rot.perms, strict=True):
        y = (y * s)[:, p]
        if n_blocks > 0:
            y[:, :head_end] = (y[:, :head_end].reshape(-1, n_blocks, 64) @ rot.h64).reshape(-1, head_end)
        if rot.tail_q is not None:
            y[:, head_end:] = y[:, head_end:] @ rot.tail_q
    return y[0] if is_1d else y


def _transform_vectors(raw_vecs: NDArray[np.float32], mean_vec: NDArray[np.float32]) -> NDArray[np.float32]:
    centered = raw_vecs.astype(np.float32, copy=False) - mean_vec
    norms = np.linalg.norm(centered, axis=-1, keepdims=True).astype(np.float32)
    safe_norms = np.where(norms > 1e-12, norms, np.float32(1.0))
    return np.where(norms > 1e-12, centered / safe_norms, np.float32(0.0)).astype(np.float32, copy=False)


def _compute_shannon_d_eff(centered_unit: NDArray[np.float32]) -> float:
    n, d = centered_unit.shape
    if n < 2:
        return float(max(2, d))
    sample = centered_unit
    if n > _SPECTRUM_SAMPLE_CAP:
        stride_idx = np.linspace(0, n - 1, _SPECTRUM_SAMPLE_CAP, dtype=np.int64)
        sample = centered_unit[stride_idx]
    m = sample.shape[0]
    _, svals, _ = np.linalg.svd(sample.astype(np.float64), full_matrices=False)
    eigvals = (svals * svals) / float(m)
    pos = eigvals[eigvals > 1e-12]
    total = float(pos.sum())
    if pos.size == 0 or total <= 1e-12:
        return float(max(2, min(d, max(2, n))))
    probs = pos / total
    ent = -float(np.sum(probs * np.log(probs)))
    return max(2.0, math.exp(ent))


def _pack_rabitq_soa(unit_vecs: NDArray[np.float32], rot: _Rotator) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
    """Quantize unit vectors into SoA uint8[W*8, N] bit-packed codes + ip_norm[N]."""
    n, d = unit_vecs.shape
    w = (d + 63) // 64
    if n == 0:
        return np.zeros((w * 8, 0), dtype=np.uint8), np.zeros(0, dtype=np.float32)
    rotated = _rotate_vectors(rot, unit_vecs)
    bits = (rotated >= 0.0).astype(np.uint8)
    padded = np.zeros((n, w * 64), dtype=np.uint8)
    padded[:, :d] = bits
    codes = np.ascontiguousarray(np.packbits(padded, axis=1, bitorder="little").T, dtype=np.uint8)
    inv_sqrt_d = 1.0 / math.sqrt(d)
    signed_vertex = np.where(bits, inv_sqrt_d, -inv_sqrt_d).astype(np.float32)
    ip_norm = np.sum(rotated * signed_vertex, axis=1).astype(np.float32)
    ip_norm = np.maximum(ip_norm, inv_sqrt_d * 0.5).astype(np.float32)
    return codes, ip_norm


def _rabitq_dot_lut(codes: NDArray[np.uint8], rotated_q: NDArray[np.float32], dim: int) -> NDArray[np.float32]:
    """Compute <x_bar_i, rotated_q> using vectorized byte-level 256-entry LUT indexing."""
    n = codes.shape[1]
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    active_bytes = (dim + 7) // 8
    inv_sqrt_d = np.float32(1.0 / math.sqrt(dim))
    q_scaled = np.zeros(active_bytes * 8, dtype=np.float32)
    q_scaled[:dim] = rotated_q * inv_sqrt_d
    lut = (q_scaled.reshape(active_bytes, 8) @ _BYTE_BITS_TBL.T).astype(np.float32, copy=False)
    acc = np.empty(n, dtype=np.float32)
    row_idx = np.arange(active_bytes)[:, None]
    block = 4096
    for s in range(0, n, block):
        e = min(n, s + block)
        acc[s:e] = lut[row_idx, codes[:active_bytes, s:e]].sum(axis=0)
    return acc


def _unpack_1bit_signed(codes: NDArray[np.uint8], indices: Sequence[int], dim: int) -> NDArray[np.float32]:
    idx_arr = np.asarray(indices, dtype=np.intp).ravel()
    if idx_arr.size == 0:
        return np.zeros((0, dim), dtype=np.float32)
    active_bytes = (dim + 7) // 8
    sub = np.ascontiguousarray(codes[:active_bytes, idx_arr].T, dtype=np.uint8)
    bits = np.unpackbits(sub, axis=1, bitorder="little")[:, :dim]
    inv_sqrt_d = np.float32(1.0 / math.sqrt(dim))
    return ((bits.astype(np.float32) * np.float32(2.0) - np.float32(1.0)) * inv_sqrt_d).astype(np.float32, copy=False)


def _pack_doc_1bit_codes(
    codes: NDArray[np.uint8],
    dim: int,
    el_idx: NDArray[np.int32],
    pidx: NDArray[np.int32],
    nd: int,
) -> NDArray[np.uint8]:
    """Pack uint8[active_bytes, nd] doc 1-bit sign codes (0.0 ties broken by first eligible chunk)."""
    active_bytes = (dim + 7) // 8
    if nd <= 0 or el_idx.size == 0:
        return np.zeros((active_bytes, max(0, nd)), dtype=np.uint8)
    pidx_el = pidx[el_idx].astype(np.int32, copy=False)
    first_c = np.full(nd, -1, dtype=np.int32)
    first_c[pidx_el[::-1]] = el_idx[::-1]
    is_first = first_c[pidx_el] == el_idx
    w = np.where(is_first, 3, 2).astype(np.int32)
    m_doc = sp.csr_array(
        (w, (pidx_el, np.arange(el_idx.size, dtype=np.int32))), shape=(nd, el_idx.size), dtype=np.int32
    )
    doc_codes = np.zeros((active_bytes, nd), dtype=np.uint8)
    b_step = 8
    for b0 in range(0, active_bytes, b_step):
        b1 = min(active_bytes, b0 + b_step)
        d0 = b0 * 8
        d1 = min(dim, b1 * 8)
        w_d = d1 - d0
        sub = np.ascontiguousarray(codes[b0:b1, el_idx].T, dtype=np.uint8)
        bits = np.unpackbits(sub, axis=1, bitorder="little")[:, :w_d]
        signed = bits.astype(np.int32) * 2 - 1
        sums = m_doc @ signed
        doc_bits = (sums > 0).astype(np.uint8)
        pad_w = (b1 - b0) * 8
        if w_d < pad_w:
            padded = np.zeros((nd, pad_w), dtype=np.uint8)
            padded[:, :w_d] = doc_bits
            doc_bits = padded
        doc_codes[b0:b1, :] = np.packbits(doc_bits, axis=1, bitorder="little").T
    return doc_codes


def _compute_edge_ham_cos(
    codes: NDArray[np.uint8],
    doc_codes: NDArray[np.uint8],
    f_rows: NDArray[np.int32],
    f_cols: NDArray[np.int32],
    dim: int,
) -> NDArray[np.float32]:
    m_edges = int(f_rows.size)
    if m_edges == 0:
        return np.zeros(0, dtype=np.float32)
    active_bytes = (dim + 7) // 8
    h_ud = np.empty(m_edges, dtype=np.int32)
    rem = dim % 8
    last_mask = np.uint8((1 << rem) - 1) if rem != 0 else np.uint8(0xFF)
    block = 16384
    for s in range(0, m_edges, block):
        e = min(m_edges, s + block)
        xb = np.bitwise_xor(codes[:active_bytes, f_rows[s:e]], doc_codes[:active_bytes, f_cols[s:e]])
        if rem != 0:
            xb[active_bytes - 1] &= last_mask
        h_ud[s:e] = np.bitwise_count(xb).sum(axis=0, dtype=np.int32)
    return np.cos(math.pi * h_ud.astype(np.float64) / float(dim)).astype(np.float32)


def _grouped_softmax(group_ids: NDArray[np.int32], logits: NDArray[np.float64], n_groups: int) -> NDArray[np.float32]:
    if group_ids.size == 0:
        return np.zeros(0, dtype=np.float32)
    g_max = np.full(n_groups, -np.inf, dtype=np.float64)
    np.maximum.at(g_max, group_ids, logits)
    ex = np.exp(logits - g_max[group_ids])
    g_sum = np.bincount(group_ids, weights=ex, minlength=n_groups)
    return (ex / np.maximum(g_sum[group_ids], 1e-30)).astype(np.float32)


def _parametric_null_nats_vec(cos_vals: NDArray[np.float64], d_eff: float) -> NDArray[np.float64]:
    out = np.zeros_like(cos_vals, dtype=np.float64)
    pos = cos_vals > 0.0
    if not np.any(pos):
        return out
    c = np.minimum(cos_vals[pos], 1.0 - 1e-15)
    a = max(0.5, 0.5 * (d_eff - 1.0))
    one_m_c2 = np.maximum(1e-30, 1.0 - c * c)
    p_two = special.betainc(a, 0.5, one_m_c2)
    asymp = a * np.log(one_m_c2) - math.log(a) - float(special.betaln(a, 0.5))
    vals = np.where(p_two <= 1e-300, -asymp, -np.log(np.maximum(p_two, 1e-300)))
    out[pos] = np.maximum(0.0, vals)
    return out


def _build_beta_nats_lut(
    d_eff: float, n_bins: int = _TANGENT_LUT_BINS
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    z_grid = np.linspace(0.0, _TANGENT_Z_MAX, n_bins + 1, dtype=np.float64)
    c_grid = np.sqrt(-np.expm1(-z_grid * z_grid))
    lut = _parametric_null_nats_vec(c_grid, d_eff)
    return lut, np.diff(lut), float(n_bins) / _TANGENT_Z_MAX


def _eval_signed_beta_nats_lut(
    rho: NDArray[np.float64],
    lut: NDArray[np.float64],
    lut_diff: NDArray[np.float64],
    scale: float,
    inv_t_path: float,
) -> NDArray[np.float64]:
    z = np.sqrt(-np.log1p(-rho * rho))
    n_bins = lut_diff.size
    pos = np.minimum(z * scale, float(n_bins))
    idx = np.minimum(pos.astype(np.int32), n_bins - 1)
    val = (lut[idx] + (pos - idx) * lut_diff[idx]) * inv_t_path
    return np.copysign(val, rho)


def _extract_markdown_h1(raw_head: str) -> str:
    first = next((ln.strip() for ln in raw_head.splitlines() if ln.strip()), "")
    if not first.startswith("#"):
        return ""
    title = _TITLE_TRAILING_SEP_RE.sub("", first.lstrip("#").strip()).strip()
    if m := _TITLE_TAG_PREFIX_RE.match(title):
        stripped = m.group(1).strip()
        if stripped:
            title = stripped
    return title


def _pack_doc_titles_blob(doc_titles: Mapping[str, str]) -> bytes:
    payload = json.dumps(dict(doc_titles), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return zlib.compress(payload, level=6)


def _build_chunk_pointers(
    rows: Sequence[Any],
) -> tuple[list[int], list[ChunkPointer], list[str], NDArray[np.uint32]]:
    chunk_ids: list[int] = []
    pointers: list[ChunkPointer] = []
    distinct_paths: list[str] = []
    path_to_idx: dict[str, int] = {}
    path_idx_arr = np.zeros(len(rows), dtype=np.uint32)
    for pos, r in enumerate(rows):
        cid = int(r[0])
        rel = str(r[2]).replace("\\", "/")
        p_idx = path_to_idx.get(rel)
        if p_idx is None:
            p_idx = len(distinct_paths)
            path_to_idx[rel] = p_idx
            distinct_paths.append(rel)
        path_idx_arr[pos] = p_idx
        chunk_ids.append(cid)
        pointers.append(
            ChunkPointer(
                chunk_id=cid,
                rel_path=str(r[2]),
                source_hash=str(r[3]),
                start_line=int(r[4]),
                end_line=int(r[5]),
                char_start=int(r[6]),
                char_end=int(r[7]),
                ast_island_start=int(r[8]),
                ast_island_end=int(r[9]),
                title=str(r[10]),
            )
        )
    return chunk_ids, pointers, distinct_paths, path_idx_arr


def _build_dc_csr(path_idx: NDArray[np.uint32], nd: int) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    if nd == 0 or path_idx.size == 0:
        return np.zeros(nd + 1, dtype=np.int32), np.zeros(0, dtype=np.int32)
    counts = np.bincount(path_idx, minlength=nd).astype(np.int32, copy=False)
    indptr = np.empty(nd + 1, dtype=np.int32)
    indptr[0] = 0
    np.cumsum(counts, dtype=np.int32, out=indptr[1:])
    indices = np.argsort(path_idx, kind="stable").astype(np.int32, copy=False)
    return indptr, indices


def _lists_to_csr(postings: Sequence[Sequence[int]], n_rows: int) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    counts = np.fromiter((len(lst) for lst in postings), dtype=np.int32, count=n_rows)
    indptr = np.empty(n_rows + 1, dtype=np.int32)
    indptr[0] = 0
    if n_rows > 0:
        np.cumsum(counts, dtype=np.int32, out=indptr[1:])
    total = int(indptr[-1])
    indices = (
        np.fromiter((x for lst in postings for x in lst), dtype=np.int32, count=total)
        if total
        else np.zeros(0, dtype=np.int32)
    )
    return indptr, indices


def _extract_title_keys(
    title: str,
    *,
    prop_nouns: Collection[str] | None = None,
) -> list[tuple[str, ...]]:
    t_norm = _DASH_RE.sub("-", title) if prop_nouns is not None else title
    t_cf = t_norm.casefold().strip()
    if not t_cf:
        return []
    keys: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()

    def _emit(tok_seq: tuple[str, ...]) -> None:
        if 1 <= len(tok_seq) <= 8 and tok_seq not in seen:
            seen.add(tok_seq)
            keys.append(tok_seq)

    toks = tuple(_WORD_TOK_RE.findall(t_cf))
    if toks:
        _emit(toks)
        if "(" in t_cf:
            base = t_cf.split("(", 1)[0].strip()
            base_toks = tuple(_WORD_TOK_RE.findall(base))
            if base_toks and (prop_nouns is None or len(base_toks) >= 2 or base_toks[0] in prop_nouns):
                _emit(base_toks)
    for run in _CJK_RUN_RE.findall(t_cf):
        if len(run) >= 2:
            _emit((run,))
    return keys


def _build_title_index(
    pointers: Sequence[ChunkPointer],
    doc_titles: Mapping[str, str],
    *,
    prop_nouns: Collection[str] | None = None,
) -> dict[str, tuple[int, dict[tuple[str, ...], tuple[int, ...]]]]:
    raw_by_first: dict[str, dict[tuple[str, ...], list[int]]] = {}
    max_len_by_first: dict[str, int] = {}
    for pos, ptr in enumerate(pointers):
        eff_title = ptr.title or doc_titles.get(ptr.rel_posix_path, "")
        for tk in _extract_title_keys(eff_title, prop_nouns=prop_nouns):
            first = tk[0]
            tbl = raw_by_first.setdefault(first, {})
            lst = tbl.get(tk)
            if lst is None:
                tbl[tk] = [pos]
            elif lst[-1] != pos:
                lst.append(pos)
            if len(tk) > max_len_by_first.get(first, 0):
                max_len_by_first[first] = len(tk)
    return {
        first: (max_len_by_first[first], {tk: tuple(lst) for tk, lst in tbl.items()})
        for first, tbl in raw_by_first.items()
    }


def _norm_entity_text(text: str) -> str:
    return _POSS_RE.sub("", _DASH_RE.sub("-", text))


def _validate_max_chars(max_chars: int | None) -> int | None:
    if max_chars is None:
        return None
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 0:
        raise ValueError(f"max_chars must be a non-negative int or None, got {max_chars!r}")
    return max_chars


def _crop_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    lines = text.splitlines()
    if len(lines) >= 3 and max_chars >= 120:
        half = max(20, (max_chars - 48) // 2)
        head: list[str] = []
        used = 0
        for ln in lines:
            if head and used + len(ln) + 1 > half:
                break
            head.append(ln)
            used += len(ln) + 1
        tail: list[str] = []
        used_t = 0
        for ln in reversed(lines[len(head) :]):
            if tail and used_t + len(ln) + 1 > half:
                break
            tail.append(ln)
            used_t += len(ln) + 1
        tail.reverse()
        omitted_start = len(head) + 1
        omitted_end = len(lines) - len(tail)
        if omitted_end >= omitted_start and head and tail:
            marker = f"[... lines {omitted_start}-{omitted_end} omitted ...]"
            candidate = "\n".join([*head, marker, *tail])
            if len(candidate) <= max_chars:
                return candidate
    marker_c = "\n[... AST section cropped ...]\n"
    if max_chars <= len(marker_c) + 8:
        return text[:max_chars]
    rem = max_chars - len(marker_c)
    left = rem // 2
    right = rem - left
    return f"{text[:left]}{marker_c}{text[-right:]}"


type _FileCache = dict[str, tuple[Path, str, list[str]] | None]


def _resolve_cached_path(
    corpus_root: Path,
    rel_posix_path: str,
    path_cache: dict[tuple[Path, str], Path] | None,
) -> Path:
    if path_cache is None:
        return resolve_corpus_path(corpus_root, rel_posix_path)
    p_key = (corpus_root, rel_posix_path)
    with _CACHE_LOCK:
        cached = path_cache.get(p_key)
    if cached is not None:
        return cached
    target = resolve_corpus_path(corpus_root, rel_posix_path)
    with _CACHE_LOCK:
        path_cache[p_key] = target
    return target


def _hydrate_one(
    corpus_root: Path,
    ptr: ChunkPointer,
    score: float,
    max_chars: int | None,
    include_island: bool,
    full_span: bool,
    *,
    strict: bool = False,
    file_cache: _FileCache | None = None,
    path_cache: dict[tuple[Path, str], Path] | None = None,
) -> HydratedPassage:
    limit = _validate_max_chars(max_chars)
    sl, el = int(ptr.start_line), int(ptr.end_line)
    if sl < 1 or el < sl or (el - sl + 1) > _MAX_HYDRATE_SPAN:
        raise ValueError(f"Invalid hydration line range [{sl}, {el}] for {ptr.rel_posix_path}")

    if file_cache is not None and ptr.rel_posix_path in file_cache:
        loaded = file_cache[ptr.rel_posix_path]
    else:
        target = _resolve_cached_path(corpus_root, ptr.rel_posix_path, path_cache)
        if not target.is_file():
            loaded = None
        else:
            try:
                text = target.read_bytes().decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
            except (OSError, UnicodeDecodeError) as exc:
                raise YawRuntimeError(
                    "CONSISTENCY_ERROR", f"Invalid UTF-8 or unreadable source file {target}: {exc}"
                ) from exc
            loaded = (target, text, text.splitlines())
        if file_cache is not None:
            file_cache[ptr.rel_posix_path] = loaded

    if loaded is None:
        if strict:
            raise YawRuntimeError("CONSISTENCY_ERROR", f"Missing corpus source file: {ptr.rel_posix_path}")
        return HydratedPassage(ptr, "", float(score), stale_on_disk=True)

    target, text, lines = loaded
    if el > len(lines):
        if strict:
            raise YawRuntimeError("CONSISTENCY_ERROR", f"Stale line range [{sl}, {el}] in {target}")
        return HydratedPassage(ptr, "", float(score), stale_on_disk=True)
    raw_slice = "\n".join(lines[sl - 1 : el])
    if ptr.source_hash and hashlib.sha256(raw_slice.encode("utf-8")).hexdigest() != ptr.source_hash:
        if strict:
            raise YawRuntimeError("CONSISTENCY_ERROR", f"SHA-256 drift detected on disk for {target}")
        return HydratedPassage(ptr, "", float(score), stale_on_disk=True)

    if include_island and 0 < ptr.ast_island_start <= ptr.ast_island_end:
        es, ee = min(sl, int(ptr.ast_island_start)), min(len(lines), max(el, int(ptr.ast_island_end)))
        rendered = "\n".join(lines[es - 1 : ee])
    elif not full_span and ptr.char_end > ptr.char_start:
        rendered = text[ptr.char_start : ptr.char_end]
    else:
        rendered = raw_slice
    if (
        ptr.title
        and not rendered.lstrip().startswith("#")
        and not rendered.startswith(f"{ptr.title}\n")
        and rendered != ptr.title
    ):
        rendered = f"{ptr.title}\n{rendered}"

    if limit is not None and limit > 0:
        rendered = _crop_text(rendered, limit)
    return HydratedPassage(ptr, rendered, float(score), stale_on_disk=False)


def hydrate_passage(
    ptr: ChunkPointer,
    corpus_root: Path,
    *,
    score: float = 0.0,
    max_chars: int | None = None,
    include_island: bool = False,
    full_span: bool = False,
) -> HydratedPassage:
    """Hydrate a single ChunkPointer directly from disk, verifying SHA-256 freshness."""
    return _hydrate_one(corpus_root, ptr, score, max_chars, include_island, full_span, strict=False)


def hydrate_ranked_passages(
    idx: RetrievalIndex,
    ranked_chunks: Sequence[tuple[int, float]],
    corpus_root: Path,
    *,
    top_k: int = 5,
    max_chars: int | None = None,
    include_island: bool = False,
    full_span: bool = False,
) -> tuple[HydratedPassage, ...]:
    """Apply MaxP span deduplication and hydrate up to `top_k` passages from disk."""
    limit = _validate_max_chars(max_chars)
    if top_k <= 0 or not ranked_chunks:
        return ()
    out: list[HydratedPassage] = []
    seen_spans: set[tuple[str, int, int, int, int]] = set()
    file_cache: _FileCache = {}
    for cid, score in ranked_chunks:
        if (pos := idx.id_to_pos.get(int(cid))) is None:
            continue
        ptr = idx.pointers[pos]
        span = (ptr.rel_posix_path, int(ptr.start_line), int(ptr.end_line), int(ptr.char_start), int(ptr.char_end))
        if span in seen_spans:
            continue
        seen_spans.add(span)
        out.append(
            _hydrate_one(
                corpus_root,
                ptr,
                float(score),
                limit,
                include_island,
                full_span,
                strict=False,
                file_cache=file_cache,
                path_cache=idx.resolved_path_cache,
            )
        )
        if len(out) >= top_k:
            break
    return tuple(out)


def format_evidence_route(passages: Sequence[HydratedPassage]) -> str:
    """Format hydrated passages with explicit provenance headers for recall and ReAct output."""
    blocks: list[str] = []
    for idx, hp in enumerate(passages, start=1):
        if hp.stale_on_disk:
            blocks.append(
                f"[{idx}] [STALE: {hp.pointer.span_label} | chunk_id={hp.pointer.chunk_id} | score={hp.score:.4f}]"
            )
        else:
            blocks.append(
                f"[{idx}] [{hp.pointer.span_label} | chunk_id={hp.pointer.chunk_id} | score={hp.score:.4f}]\n{hp.text}"
            )
    return "\n\n".join(blocks)


def _hydrate_corpus_chunks(
    pointers: Sequence[ChunkPointer],
    corpus_root: Path,
    *,
    extract_titles_for: Collection[str] = (),
) -> tuple[list[str], dict[str, str]]:
    n = len(pointers)
    if n == 0:
        return [], {}
    c_root = corpus_root if corpus_root.is_absolute() else corpus_root.resolve()
    title_targets = set(extract_titles_for)
    by_rel: dict[str, list[int]] = {}
    for pos, ptr in enumerate(pointers):
        by_rel.setdefault(ptr.rel_posix_path, []).append(pos)

    texts = [""] * n
    titles: dict[str, str] = {}

    def _read_group(item: tuple[str, list[int]]) -> tuple[list[tuple[int, str]], str, str]:
        rel, pos_list = item
        local_cache: _FileCache = {}
        chunk_texts = [
            (
                pos,
                _hydrate_one(
                    c_root,
                    pointers[pos],
                    1.0,
                    None,
                    False,
                    False,
                    strict=True,
                    file_cache=local_cache,
                ).text,
            )
            for pos in pos_list
        ]
        title = ""
        if rel in title_targets and (loaded := local_cache.get(rel)) is not None:
            first_line = next((ln for ln in loaded[2] if ln.strip()), "")
            if first_line:
                title = _extract_markdown_h1(first_line)
        return chunk_texts, rel, title

    with ThreadPoolExecutor(max_workers=16) as pool:
        for grp_res, rel, title in pool.map(_read_group, by_rel.items()):
            for pos, txt in grp_res:
                texts[pos] = txt
            if title:
                titles[rel] = title
    return texts, titles


def _build_citation_graph(
    pointers: Sequence[ChunkPointer],
    distinct_paths: Sequence[str],
    path_idx_arr: NDArray[np.uint32],
    corpus_root: Path | None,
) -> CitationGraph:
    n = len(pointers)
    nd = len(distinct_paths)
    dc_indptr, dc_indices = _build_dc_csr(path_idx_arr, nd)
    raw_dt = [""] * nd
    for pos, ptr in enumerate(pointers):
        di = int(path_idx_arr[pos])
        if not raw_dt[di] and (t_str := ptr.title.strip()):
            raw_dt[di] = _DASH_RE.sub("-", t_str)

    if corpus_root is None:
        return CitationGraph(
            fi=np.zeros(n + 1, dtype=np.int32),
            fx=np.zeros(0, dtype=np.int32),
            ri=np.zeros(nd + 1, dtype=np.int32),
            rx=np.zeros(0, dtype=np.int32),
            dc_indptr=dc_indptr,
            dc_indices=dc_indices,
            pmi_indptr=np.zeros(1, dtype=np.int32),
            pmi_indices=np.zeros(0, dtype=np.int32),
            cpmi_indptr=np.zeros(n + 1, dtype=np.int32),
            cpmi_indices=np.zeros(0, dtype=np.int32),
            pmi_terms=(),
            prop_nouns=frozenset(),
            doc_titles=tuple(raw_dt),
        )

    missing_title_paths = {distinct_paths[di] for di, t in enumerate(raw_dt) if not t}
    ct, disk_titles = _hydrate_corpus_chunks(pointers, corpus_root, extract_titles_for=missing_title_paths)
    for di, dp in enumerate(distinct_paths):
        if not raw_dt[di] and (t_ext := disk_titles.get(dp, "")):
            raw_dt[di] = _DASH_RE.sub("-", t_ext)
    doc_titles_tup = tuple(raw_dt)

    doc_titles_norm = {dp: dt for dp, dt in zip(distinct_paths, doc_titles_tup, strict=True) if dt}
    db = max(4, math.ceil(math.sqrt(max(1, n))))

    cap_cnt: dict[str, int] = {}
    tot_cnt: dict[str, int] = {}
    w_df: dict[str, int] = {}
    bg_post: dict[tuple[str, str], list[int]] = {}

    for u, raw_chunk in enumerate(ct):
        raw_t = _norm_entity_text(raw_chunk)
        for sent in _SENT_SPLIT_RE.split(raw_t):
            stoks = _WORD_TOK_RE.findall(sent)
            for ti, tok in enumerate(stoks):
                w = tok.casefold()
                if (len(w) >= 3 or (len(w) >= 2 and _CJK_RUN_RE.fullmatch(w) is not None)) and (
                    ti > 0 or (len(tok) >= 2 and tok.isupper()) or _CJK_RUN_RE.fullmatch(tok) is not None
                ):
                    tot_cnt[w] = tot_cnt.get(w, 0) + 1
                    if tok[0].isupper() or (len(tok) >= 2 and tok.isupper()) or _CJK_RUN_RE.fullmatch(tok) is not None:
                        cap_cnt[w] = cap_cnt.get(w, 0) + 1
        cf_toks = _WORD_TOK_RE.findall(raw_t.casefold())
        for w in set(cf_toks):
            w_df[w] = w_df.get(w, 0) + 1
        seen_bg: set[tuple[str, str]] = set()
        for j in range(len(cf_toks) - 1):
            bg = (cf_toks[j], cf_toks[j + 1])
            if bg not in seen_bg:
                seen_bg.add(bg)
                bg_post.setdefault(bg, []).append(u)
        for run in _CJK_RUN_RE.findall(raw_t.casefold()):
            if len(run) >= 2:
                cjk_bgs = [run[k : k + 2] for k in range(len(run) - 1)]
                for bg in set(cjk_bgs):
                    w_df[bg] = w_df.get(bg, 0) + 1
                for k in range(len(run) - 3):
                    pair = (run[k : k + 2], run[k + 2 : k + 4])
                    if pair not in seen_bg:
                        seen_bg.add(pair)
                        bg_post.setdefault(pair, []).append(u)

    prop_nouns = frozenset(w for w, tot in tot_cnt.items() if 2 * cap_cnt.get(w, 0) > tot)
    tbfw_clean = _build_title_index(pointers, doc_titles_norm, prop_nouns=prop_nouns)

    tk_to_docs: dict[tuple[str, ...], tuple[int, ...]] = {}
    for _, (_, sub_tbl) in tbfw_clean.items():
        for tk, pos_tup in sub_tbl.items():
            if len(pos_tup) <= db:
                tk_to_docs[tk] = tuple(sorted({int(path_idx_arr[p]) for p in pos_tup}))

    fwd_post: list[list[int]] = [[] for _ in range(n)]
    rev_adj: list[list[int]] = [[] for _ in range(nd)]
    for u, raw_chunk in enumerate(ct):
        du = int(path_idx_arr[u])
        targets: set[int] = set()
        for sent in _SENT_SPLIT_RE.split(_norm_entity_text(raw_chunk)):
            ltoks = _WORD_TOK_RE.findall(sent.casefold())
            nt = len(ltoks)
            i = 0
            while i < nt:
                entry = tbfw_clean.get(ltoks[i])
                if entry is not None:
                    ml, sub_tbl = entry
                    hit = 0
                    for span_len in range(min(ml, nt - i), 0, -1):
                        cand = tuple(ltoks[i : i + span_len])
                        if cand in sub_tbl:
                            docs = tk_to_docs.get(cand)
                            if docs:
                                for di in docs:
                                    if di != du:
                                        targets.add(di)
                            hit = span_len
                            break
                    if hit > 0:
                        i += hit
                        continue
                i += 1
            for run in _CJK_RUN_RE.findall(sent.casefold()):
                rl = len(run)
                for j in range(rl):
                    for span_len in range(min(8, rl - j), 1, -1):
                        docs = tk_to_docs.get((run[j : j + span_len],))
                        if docs:
                            for di in docs:
                                if di != du:
                                    targets.add(di)
                            break
        if targets:
            sorted_t = sorted(targets)
            fwd_post[u] = sorted_t
            for di in sorted_t:
                rev_adj[di].append(u)

    fi, fx = _lists_to_csr(fwd_post, n)
    ri, rx = _lists_to_csr(rev_adj, nd)

    pmi_items = sorted(
        (
            (f"{a} {b}", cl)
            for (a, b), cl in bg_post.items()
            if 2 <= len(cl) <= db
            and len(cl) * n >= 2 * w_df[a] * w_df[b]
            and len({int(path_idx_arr[u]) for u in cl}) >= 2
        ),
        key=lambda item: (len(item[1]), item[0]),
    )
    pmi_terms = tuple(term for term, _ in pmi_items)
    pmi_lists = [cl for _, cl in pmi_items]
    c_pmis: list[list[int]] = [[] for _ in range(n)]
    for eid, cl in enumerate(pmi_lists):
        for u in cl:
            c_pmis[u].append(eid)
    for u in range(n):
        if len(c_pmis[u]) > 1:
            c_pmis[u].sort(key=lambda e: (len(pmi_lists[e]), pmi_terms[e]))
    pmi_indptr, pmi_indices = _lists_to_csr(pmi_lists, len(pmi_items))
    cpmi_indptr, cpmi_indices = _lists_to_csr(c_pmis, n)

    return CitationGraph(
        fi=fi,
        fx=fx,
        ri=ri,
        rx=rx,
        dc_indptr=dc_indptr,
        dc_indices=dc_indices,
        pmi_indptr=pmi_indptr,
        pmi_indices=pmi_indices,
        cpmi_indptr=cpmi_indptr,
        cpmi_indices=cpmi_indices,
        pmi_terms=pmi_terms,
        prop_nouns=prop_nouns,
        doc_titles=doc_titles_tup,
    )


def _pack_citation_graph_blob(cg: CitationGraph) -> bytes:
    n = int(cg.fi.size) - 1
    nd = int(cg.ri.size) - 1
    meta_bytes = json.dumps(
        [list(cg.pmi_terms), sorted(cg.prop_nouns), list(cg.doc_titles)],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    hdr = struct.pack(
        _CITATION_GRAPH_HDR_FMT,
        _CITATION_GRAPH_MAGIC,
        n,
        nd,
        int(cg.fx.size),
        int(cg.rx.size),
        len(cg.pmi_terms),
        int(cg.pmi_indices.size),
        len(meta_bytes),
        0,
        0,
        0,
        0,
    )
    arrays = (
        cg.fi,
        cg.fx,
        cg.ri,
        cg.rx,
        cg.pmi_indptr,
        cg.pmi_indices,
        cg.cpmi_indptr,
        cg.cpmi_indices,
    )
    payload = b"".join([hdr, *(np.ascontiguousarray(a, dtype="<i4").tobytes() for a in arrays), meta_bytes])
    return zlib.compress(payload, level=6)


def _unpack_citation_graph_blob(blob: bytes, path_idx: NDArray[np.uint32], nd_expected: int) -> CitationGraph:
    if not blob:
        raise YawRuntimeError("CONSISTENCY_ERROR", "Empty graph_blob in citation_graph.")
    try:
        raw = zlib.decompress(blob)
    except zlib.error as exc:
        raise YawRuntimeError("CONSISTENCY_ERROR", f"Corrupt zlib stream in citation_graph.graph_blob: {exc}") from exc
    hdr_sz = struct.calcsize(_CITATION_GRAPH_HDR_FMT)
    if len(raw) < hdr_sz:
        raise YawRuntimeError("CONSISTENCY_ERROR", "Truncated citation_graph.graph_blob header.")
    magic, n, nd, n_fx, n_rx, n_pmi, n_pmi_idx, meta_len, *_ = struct.unpack(_CITATION_GRAPH_HDR_FMT, raw[:hdr_sz])
    if magic != _CITATION_GRAPH_MAGIC:
        raise YawRuntimeError("CONSISTENCY_ERROR", f"Invalid citation_graph.graph_blob magic: {magic!r}")
    if n != int(path_idx.size) or nd != nd_expected:
        raise YawRuntimeError(
            "CONSISTENCY_ERROR",
            f"citation_graph dimension mismatch: got (n={n}, nd={nd}), expected (n={path_idx.size}, nd={nd_expected}).",
        )
    expected_ints = (n + 1) + n_fx + (nd + 1) + n_rx + (n_pmi + 1) + n_pmi_idx + (n + 1) + n_pmi_idx
    if len(raw) != hdr_sz + expected_ints * 4 + meta_len:
        raise YawRuntimeError("CONSISTENCY_ERROR", "Corrupt citation_graph.graph_blob byte length.")

    off = hdr_sz

    def _take(cnt: int) -> NDArray[np.int32]:
        nonlocal off
        arr = np.frombuffer(raw, dtype="<i4", count=cnt, offset=off).astype(np.int32, copy=True)
        off += cnt * 4
        return arr

    fi = _take(n + 1)
    fx = _take(n_fx)
    ri = _take(nd + 1)
    rx = _take(n_rx)
    pmi_indptr = _take(n_pmi + 1)
    pmi_indices = _take(n_pmi_idx)
    cpmi_indptr = _take(n + 1)
    cpmi_indices = _take(n_pmi_idx)

    if (
        int(fi[0]) != 0
        or int(fi[-1]) != n_fx
        or int(ri[0]) != 0
        or int(ri[-1]) != n_rx
        or int(pmi_indptr[0]) != 0
        or int(pmi_indptr[-1]) != n_pmi_idx
        or int(cpmi_indptr[0]) != 0
        or int(cpmi_indptr[-1]) != n_pmi_idx
        or bool(np.any(np.diff(fi) < 0))
        or bool(np.any(np.diff(ri) < 0))
        or bool(np.any(np.diff(pmi_indptr) < 0))
        or bool(np.any(np.diff(cpmi_indptr) < 0))
    ):
        raise YawRuntimeError("CONSISTENCY_ERROR", "Corrupt CSR indptr bounds in citation_graph.graph_blob.")
    if fx.size and (int(fx.min()) < 0 or int(fx.max()) >= nd):
        raise YawRuntimeError("CONSISTENCY_ERROR", "Out-of-range document index in citation_graph.graph_blob.")
    if rx.size and (int(rx.min()) < 0 or int(rx.max()) >= n):
        raise YawRuntimeError("CONSISTENCY_ERROR", "Out-of-range chunk index in citation_graph.graph_blob.")
    if pmi_indices.size and (int(pmi_indices.min()) < 0 or int(pmi_indices.max()) >= n):
        raise YawRuntimeError("CONSISTENCY_ERROR", "Out-of-range chunk index in citation_graph.graph_blob.")
    if cpmi_indices.size and (int(cpmi_indices.min()) < 0 or int(cpmi_indices.max()) >= n_pmi):
        raise YawRuntimeError("CONSISTENCY_ERROR", "Out-of-range PMI term index in citation_graph.graph_blob.")

    try:
        meta = json.loads(raw[off:].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise YawRuntimeError("CONSISTENCY_ERROR", f"Corrupt citation_graph metadata JSON: {exc}") from exc
    if (
        not isinstance(meta, list)
        or len(meta) != 3
        or not isinstance(meta[0], list)
        or not isinstance(meta[1], list)
        or not isinstance(meta[2], list)
        or len(meta[0]) != n_pmi
        or len(meta[2]) != nd
    ):
        raise YawRuntimeError("CONSISTENCY_ERROR", "Corrupt citation_graph metadata structure.")
    dc_indptr, dc_indices = _build_dc_csr(path_idx, nd)
    return CitationGraph(
        fi=fi,
        fx=fx,
        ri=ri,
        rx=rx,
        dc_indptr=dc_indptr,
        dc_indices=dc_indices,
        pmi_indptr=pmi_indptr,
        pmi_indices=pmi_indices,
        cpmi_indptr=cpmi_indptr,
        cpmi_indices=cpmi_indices,
        pmi_terms=tuple(str(x) for x in meta[0]),
        prop_nouns=frozenset(str(x) for x in meta[1]),
        doc_titles=tuple(str(x) for x in meta[2]),
    )


def compute_index_artifacts(rows: Sequence[Any], corpus_root: Path | str | None = None) -> IndexArtifacts:
    """Compute 1-bit SoA RaBitQ codes, Shannon d_eff, and compressed CitationGraph blobs from chunk rows."""
    if not rows:
        empty = hashlib.sha256(b"").hexdigest()
        return IndexArtifacts(0, 0, empty, empty, 0, b"", b"", b"", 0.0, b"", b"")
    n, dim, vec_sha, src_sha = _fingerprint_rows(rows)
    raw = np.vstack([np.frombuffer(bytes(r[1]), dtype="<f2").astype(np.float32) for r in rows])
    if not np.all(np.isfinite(raw)):
        raise YawRuntimeError("CONSISTENCY_ERROR", "Non-finite embedding_f16 value in chunks table.")
    mean_vec = raw.mean(axis=0).astype(np.float32) if n >= 2 else np.zeros(dim, dtype=np.float32)
    unit = _transform_vectors(raw, mean_vec)
    d_shannon = _compute_shannon_d_eff(unit)
    rot = _build_rotator(dim)
    codes, ip_norm = _pack_rabitq_soa(unit, rot)
    w = (dim + 63) // 64
    c_root = Path(corpus_root).resolve() if corpus_root is not None else None
    _cids, pointers, distinct_paths, path_idx_arr = _build_chunk_pointers(rows)
    cg = _build_citation_graph(pointers, distinct_paths, path_idx_arr, c_root)
    doc_titles_map = {dp: dt for dp, dt in zip(distinct_paths, cg.doc_titles, strict=True) if dt}
    doc_titles_blob = _pack_doc_titles_blob(doc_titles_map)
    graph_blob = _pack_citation_graph_blob(cg)
    return IndexArtifacts(
        n_chunks=n,
        dim=dim,
        vec_sha256=vec_sha,
        source_sha256=src_sha,
        n_words=w,
        codes_blob=np.ascontiguousarray(codes, dtype=np.uint8).tobytes(),
        ip_norm_blob=np.ascontiguousarray(ip_norm, dtype="<f4").tobytes(),
        mean_blob=np.ascontiguousarray(mean_vec, dtype="<f4").tobytes(),
        d_eff_shannon=float(d_shannon),
        doc_titles_blob=doc_titles_blob,
        graph_blob=graph_blob,
    )


def _build_diffusion_operators(
    cg: CitationGraph,
    path_idx: NDArray[np.uint32],
    *,
    codes: NDArray[np.uint8],
    dim: int,
    d_eff: float,
    eligible: NDArray[np.bool_] | None = None,
) -> DiffusionOperators:
    """Build static O(nnz) forward/transpose citation + non-self-loop PMI diffusion operators and Beta LUT."""
    n = int(path_idx.size)
    nd = len(cg.doc_titles)
    u_all = np.arange(n, dtype=np.int32)
    pidx = path_idx.astype(np.int32, copy=False)
    el = np.ones(n, dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool).reshape(n)
    el_idx = np.flatnonzero(el).astype(np.int32, copy=False)
    n_elig = int(el_idx.size)

    d_own: sp.csr_array | None = None
    d_id_t: sp.csr_array | None = None
    edge_f_rows: NDArray[np.int32] | None = None
    edge_f_cols: NDArray[np.int32] | None = None
    edge_c_ud: NDArray[np.float32] | None = None
    edge_inv_s_ud: NDArray[np.float32] | None = None
    edge_fwd_indptr: NDArray[np.int32] | None = None
    edge_t_order: NDArray[np.int32] | None = None
    edge_t_indptr: NDArray[np.int32] | None = None
    edge_t_indices: NDArray[np.int32] | None = None
    edge_beta_lut: NDArray[np.float64] | None = None
    edge_beta_lut_diff: NDArray[np.float64] | None = None
    edge_beta_scale = 0.0
    edge_beta_d_tan = 0.0
    edge_nd = 0
    has_fwd = np.zeros(n, dtype=np.float32)
    has_rev = np.zeros(n, dtype=np.float32)

    if n > 0 and nd > 0 and cg.fx.size > 0 and n_elig > 0:
        doc_sz = np.bincount(pidx[el_idx], minlength=nd).astype(np.float32)
        f_rows_all = np.repeat(u_all, np.diff(cg.fi)).astype(np.int32, copy=False)
        f_cols_all = cg.fx.astype(np.int32, copy=False)
        f_keep = el[f_rows_all] & (doc_sz[f_cols_all] > 0.0)
        f_rows = f_rows_all[f_keep]
        f_cols = f_cols_all[f_keep]
        if f_rows.size > 0:
            d_own = sp.csr_array(
                (1.0 / np.maximum(doc_sz[pidx[el_idx]], 1.0), (el_idx, pidx[el_idx])),
                shape=(n, nd),
                dtype=np.float32,
            )
            out_deg = np.bincount(f_rows, minlength=n)
            has_fwd = (out_deg > 0).astype(np.float32)
            if f_rows.size > 1:
                is_sorted = bool(
                    np.all((f_rows[:-1] < f_rows[1:]) | ((f_rows[:-1] == f_rows[1:]) & (f_cols[:-1] <= f_cols[1:])))
                )
                if not is_sorted:
                    lex_order = np.lexsort((f_cols, f_rows))
                    f_rows = f_rows[lex_order]
                    f_cols = f_cols[lex_order]
                if bool(np.any((f_rows[:-1] == f_rows[1:]) & (f_cols[:-1] == f_cols[1:]))):
                    raise YawRuntimeError("CONSISTENCY_ERROR", "Duplicate (f_rows, f_cols) edge in citation_graph.")
            doc_codes = _pack_doc_1bit_codes(codes, dim, el_idx, pidx, nd)
            c_ud_raw = _compute_edge_ham_cos(codes, doc_codes, f_rows, f_cols, dim)
            edge_f_rows = f_rows
            edge_f_cols = f_cols
            edge_c_ud = np.clip(c_ud_raw, -_TANGENT_RHO_CLIP, _TANGENT_RHO_CLIP).astype(np.float32)
            edge_inv_s_ud = (1.0 / np.sqrt(np.maximum(1e-6, 1.0 - edge_c_ud.astype(np.float64) ** 2))).astype(
                np.float32
            )
            edge_fwd_indptr = np.zeros(n + 1, dtype=np.int32)
            np.cumsum(out_deg, out=edge_fwd_indptr[1:])
            edge_t_order = np.lexsort((f_rows, f_cols)).astype(np.int32)
            in_deg_doc = np.bincount(f_cols, minlength=nd)
            edge_t_indptr = np.zeros(nd + 1, dtype=np.int32)
            np.cumsum(in_deg_doc, out=edge_t_indptr[1:])
            edge_t_indices = f_rows[edge_t_order].astype(np.int32, copy=False)
            edge_beta_d_tan = max(2.0, float(d_eff) - 1.0)
            edge_beta_lut, edge_beta_lut_diff, edge_beta_scale = _build_beta_nats_lut(edge_beta_d_tan)
            edge_nd = nd
            d_id_t = sp.csr_array(
                (np.ones(el_idx.size, dtype=np.float32), (pidx[el_idx], el_idx)),
                shape=(nd, n),
                dtype=np.float32,
            )
            has_rev = (el & (in_deg_doc[pidx] > 0)).astype(np.float32)

    p2c: sp.csr_array | None = None
    c2p_t: sp.csr_array | None = None
    self_pmi: NDArray[np.float32] | None = None
    pmi_scale: NDArray[np.float32] | None = None
    has_pmi = np.zeros(n, dtype=np.float32)
    n_pmi = len(cg.pmi_terms)
    if n > 0 and n_pmi > 0 and cg.cpmi_indices.size > 0 and n_elig > 0:
        p_rows_all = np.repeat(u_all, np.diff(cg.cpmi_indptr)).astype(np.int32, copy=False)
        p_cols_all = cg.cpmi_indices.astype(np.int32, copy=False)
        p_keep = el[p_rows_all]
        p_rows = p_rows_all[p_keep]
        p_cols = p_cols_all[p_keep]
        if p_rows.size > 0:
            pmi_df = np.bincount(p_cols, minlength=n_pmi).astype(np.float32)
            pmi_idf = np.where(
                pmi_df > 0.0,
                np.log((float(n_elig) + 1.0) / (np.maximum(pmi_df, 1.0) + 0.5)),
                0.0,
            ).astype(np.float32)
            b_pmi = sp.csr_array((pmi_idf[p_cols], (p_rows, p_cols)), shape=(n, n_pmi), dtype=np.float32)
            pr_sum = b_pmi.sum(axis=1).astype(np.float32)
            pc_sum = b_pmi.sum(axis=0).astype(np.float32)
            c2p = (sp.diags_array(np.where(pr_sum > 0.0, 1.0 / np.maximum(pr_sum, 1e-12), 0.0)) @ b_pmi).tocsr()
            p2c = (b_pmi @ sp.diags_array(np.where(pc_sum > 0.0, 1.0 / np.maximum(pc_sum, 1e-12), 0.0))).tocsr()
            self_pmi = c2p.multiply(p2c).sum(axis=1).astype(np.float32)
            non_self_pmi = np.maximum(1.0 - self_pmi, 0.0).astype(np.float32)
            has_pmi = (el & (non_self_pmi > 1e-6)).astype(np.float32)
            pmi_scale = np.where(has_pmi > 0.0, 1.0 / np.maximum(non_self_pmi, 1e-6), 0.0).astype(np.float32)
            c2p_t = c2p.T.tocsr()

    m_cnt = has_fwd + has_rev + has_pmi
    inv_m = np.where(el & (m_cnt > 0.0), 1.0 / np.maximum(m_cnt, 1.0), 0.0).astype(np.float32)
    return DiffusionOperators(
        d_own=d_own,
        m_fwd_t=None,
        r_rev=None,
        d_id_t=d_id_t,
        p2c=p2c,
        c2p_t=c2p_t,
        self_pmi=self_pmi,
        pmi_scale=pmi_scale,
        inv_m=inv_m,
        edge_f_rows=edge_f_rows,
        edge_f_cols=edge_f_cols,
        edge_c_ud=edge_c_ud,
        edge_inv_s_ud=edge_inv_s_ud,
        edge_fwd_indptr=edge_fwd_indptr,
        edge_t_order=edge_t_order,
        edge_t_indptr=edge_t_indptr,
        edge_t_indices=edge_t_indices,
        edge_beta_lut=edge_beta_lut,
        edge_beta_lut_diff=edge_beta_lut_diff,
        edge_beta_scale=edge_beta_scale,
        edge_beta_d_tan=edge_beta_d_tan,
        edge_nd=edge_nd,
    )


def build_retrieval_index(
    conn: sqlite3.Connection,
    *,
    expected_embedding_model: str | None = None,
    expected_task_mode: str | None = None,
) -> RetrievalIndex:
    """Load persisted 1-bit SoA RaBitQ codes and CitationGraph from SQLite into a RetrievalIndex."""
    with _snapshot_tx(conn, require_fts=False):
        if conn.execute("SELECT 1 FROM main.sqlite_master WHERE type='table' AND name='chunks'").fetchone() is None:
            return empty_retrieval_index()
        n, dim, vec_sha, src_sha, rows = compute_db_fingerprints(conn)
        if n == 0:
            return empty_retrieval_index()
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
    g_n, g_vsha, g_ssha = g_row[:3]
    if (int(g_n), str(g_vsha), str(g_ssha)) != (n, vec_sha, src_sha):
        raise YawRuntimeError("INDEX_NOT_READY", "Stale citation_graph fingerprint against chunks table.")

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

    if int(g_row[3]) <= 0:
        raise YawRuntimeError("CONSISTENCY_ERROR", "Empty doc_titles_blob in citation_graph.")
    chunk_ids, pointers, distinct_paths, path_idx_arr = _build_chunk_pointers(rows)
    cg = _unpack_citation_graph_blob(bytes(g_row[4]), path_idx_arr, len(distinct_paths))
    doc_titles = {dp: dt for dp, dt in zip(distinct_paths, cg.doc_titles, strict=True) if dt}
    title_idx = _build_title_index(pointers, doc_titles, prop_nouns=cg.prop_nouns)
    diffusion_ops = _build_diffusion_operators(cg, path_idx_arr, codes=codes, dim=dim, d_eff=d_eff)

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
        rotator=_build_rotator(dim),
        doc_titles=doc_titles,
        title_by_first_word=title_idx,
        citation_graph=cg,
        diffusion_ops=diffusion_ops,
    )


def _eligible_mask(
    idx: RetrievalIndex,
    *,
    path_glob: str | None = None,
    masked_rel_paths: Collection[str] | None = None,
) -> NDArray[np.bool_]:
    n = idx.size
    if path_glob is None and not masked_rel_paths:
        return np.ones(n, dtype=bool)
    masked_set = {p.replace("\\", "/") for p in masked_rel_paths} if masked_rel_paths else set()
    allowed_paths = np.array(
        [
            (p not in masked_set) and (path_glob is None or PurePosixPath(p).full_match(path_glob))
            for p in idx.distinct_paths
        ],
        dtype=bool,
    )
    return allowed_paths[idx.path_idx]


def _fts_quote(term: str) -> str:
    return '"' + term.replace('"', '""') + '"'


def _match_postings(
    conn: sqlite3.Connection,
    idx: RetrievalIndex,
    eligible: NDArray[np.bool_],
    expr: str,
    *,
    row_limit: int | None = None,
    all_eligible: bool | None = None,
    elig_list: list[bool] | None = None,
) -> tuple[set[int], bool]:
    if not expr:
        return set(), False
    if row_limit is not None:
        rows = conn.execute("SELECT rowid FROM chunk_fts WHERE chunk_fts MATCH ? LIMIT ?", (expr, row_limit)).fetchall()
        if len(rows) >= row_limit:
            return set(), True
    else:
        rows = conn.execute("SELECT rowid FROM chunk_fts WHERE chunk_fts MATCH ?", (expr,)).fetchall()
    if not rows:
        return set(), False
    id_get = idx.id_to_pos.get
    is_all = bool(eligible.all()) if all_eligible is None else all_eligible
    if is_all:
        out = {p for (rid,) in rows if (p := id_get(rid)) is not None}
    else:
        el = elig_list if elig_list is not None else eligible.tolist()
        out = {p for (rid,) in rows if (p := id_get(rid)) is not None and el[p]}
    return out, False


def _lookup_u61_postings(
    conn: sqlite3.Connection,
    idx: RetrievalIndex,
    eligible: NDArray[np.bool_],
    term: str,
    *,
    is_cjk: bool = False,
    prefix: bool = False,
    postings_cache: dict[tuple[str, bool, bool], set[int]],
    row_limit: int | None = None,
    oversized_keys: set[tuple[str, bool, bool]],
    all_eligible: bool,
    elig_list: list[bool] | None,
) -> set[int]:
    if not term:
        return set()
    cache_key = (term, is_cjk, prefix)
    if (cached := postings_cache.get(cache_key)) is not None:
        return set(cached)
    if row_limit is not None and cache_key in oversized_keys:
        return set()

    if prefix:
        expr = f"{_fts_quote(term)}*"
    elif is_cjk and len(term) >= 3:
        expr = _fts_quote(" ".join(term[i : i + 2] for i in range(len(term) - 1)))
    elif is_cjk and len(term) == 1:
        expr = f"{_fts_quote(term)}*"
    else:
        expr = _fts_quote(term)

    out, truncated = _match_postings(
        conn, idx, eligible, expr, row_limit=row_limit, all_eligible=all_eligible, elig_list=elig_list
    )
    if truncated:
        oversized_keys.add(cache_key)
        return set()
    postings_cache[cache_key] = set(out)
    return out


def _u61_vocab_key(w: str) -> str:
    """Fold Latin-1/Extended-A diacritics (U+00C0..U+024F) to match FTS5 unicode61 remove_diacritics=1."""
    out: list[str] = []
    for ch in w:
        o = ord(ch)
        if 0x00C0 <= o <= 0x024F:
            out.append("".join(c for c in unicodedata.normalize("NFD", ch) if unicodedata.category(c) != "Mn"))
        else:
            out.append(ch)
    return "".join(out)


def _fetch_vocab_docs(conn: sqlite3.Connection, terms: Sequence[str], target: dict[str, int]) -> None:
    with _CACHE_LOCK:
        missing = list(dict.fromkeys(t for t in terms if t not in target))
    if not missing:
        return
    fetched: dict[str, int] = {}
    if len(missing) == 1:
        t0 = missing[0]
        row = conn.execute("SELECT doc FROM main.chunk_vocab WHERE term = ?", (t0,)).fetchone()
        fetched[t0] = int(row[0]) if row is not None else 0
    else:
        for start in range(0, len(missing), 500):
            batch_t = missing[start : start + 500]
            ph = ",".join("?" for _ in batch_t)
            for term_s, doc_cnt in conn.execute(
                f"SELECT term, doc FROM main.chunk_vocab WHERE term IN ({ph})",
                batch_t,
            ).fetchall():
                fetched[str(term_s)] = int(doc_cnt)
            for t in batch_t:
                fetched.setdefault(t, 0)
    with _CACHE_LOCK:
        for k, v in fetched.items():
            target.setdefault(k, v)


def _umlaut_primary_variants(tok: str) -> list[str]:
    variants = [tok]
    folded = tok.translate(_ASCII_NORM_TRANS)
    if folded != tok:
        variants.append(folded)
    return variants


def _umlaut_reverse_variant(tok: str) -> str | None:
    rev = tok.replace("ae", "ä").replace("oe", "ö").replace("ue", "ü")
    return rev if rev != tok else None


def _extract_multilingual_anchors(
    conn: sqlite3.Connection,
    idx: RetrievalIndex,
    eligible: NDArray[np.bool_],
    query_text: str,
    *,
    max_df_filter: int | None = None,
) -> list[tuple[str, set[int]]]:
    """Extract multilingual anchors: unicode61 words, umlauts, decompounds, PMI bigrams, and CJK spans."""
    q_cf = query_text.casefold()
    n_eff = int(np.sum(eligible))
    if n_eff == 0 or not q_cf.strip():
        return []

    all_eligible = n_eff == idx.size
    elig_list = None if all_eligible else eligible.tolist()
    cache: dict[tuple[str, bool, bool], set[int]] = {}
    oversized: set[tuple[str, bool, bool]] = set()
    word_row_limit = max_df_filter + 1 if (max_df_filter is not None and all_eligible) else None
    anchors: dict[str, set[int]] = {}
    words = _WORD_TOK_RE.findall(q_cf)

    def _add_anchor(label: str, postings: set[int]) -> None:
        if not postings:
            return
        if max_df_filter is not None and len(postings) > max_df_filter:
            return
        prev = anchors.get(label)
        if prev is None:
            anchors[label] = set(postings)
        else:
            prev.update(postings)

    def _lookup_word(w: str) -> set[int]:
        return _lookup_u61_postings(
            conn,
            idx,
            eligible,
            w,
            postings_cache=cache,
            row_limit=word_row_limit,
            oversized_keys=oversized,
            all_eligible=all_eligible,
            elig_list=elig_list,
        )

    unique_words = list(dict.fromkeys(words))
    vocab_doc_counts: dict[str, int] | None = None
    if max_df_filter is not None and unique_words:
        vocab_doc_counts = idx.vocab_doc_cache
        probe_terms: list[str] = []
        for raw_w in unique_words:
            cand_vs = _umlaut_primary_variants(raw_w)
            if (rev := _umlaut_reverse_variant(raw_w)) is not None:
                cand_vs.append(rev)
            for w in cand_vs:
                pkey = (w, False, False)
                if pkey in cache or pkey in oversized:
                    continue
                if w.isalnum():
                    vk = _u61_vocab_key(w)
                    with _CACHE_LOCK:
                        missing_vk = vk not in vocab_doc_counts
                    if missing_vk:
                        probe_terms.append(vk)
        _fetch_vocab_docs(conn, probe_terms, vocab_doc_counts)

    def _run_decomp_for_word(w: str, pts: set[int]) -> None:
        if len(w) < 6 or len(pts) >= 2:
            return
        full_df = len(pts)
        if max_df_filter is not None and full_df > max_df_filter:
            return
        if not w.isalnum():
            return
        min_stem_df = 1 if full_df == 0 else max(2, 2 * full_df)
        min_split_df = max(full_df + 1, 1)
        pref4 = _u61_vocab_key(w[:4])
        pref_rows = conn.execute(
            "SELECT term, doc FROM main.chunk_vocab WHERE term >= ? AND term < ?",
            (pref4, pref4 + "\U0010ffff"),
        ).fetchall()
        pref_terms = [(str(r[0]), int(r[1])) for r in pref_rows]
        if not pref_terms:
            return

        best_stem: tuple[str, set[int]] | None = None
        for cut in range(len(w) - 1, 3, -1):
            pk = _u61_vocab_key(w[:cut])
            if sum(d for t, d in pref_terms if t.startswith(pk)) < min_stem_df:
                continue
            s_pts = _lookup_u61_postings(
                conn,
                idx,
                eligible,
                w[:cut],
                prefix=True,
                postings_cache=cache,
                row_limit=word_row_limit,
                oversized_keys=oversized,
                all_eligible=all_eligible,
                elig_list=elig_list,
            )
            if len(s_pts) >= min_stem_df and (max_df_filter is None or len(s_pts) <= max_df_filter):
                best_stem = (f"stem:{w[:cut]}", s_pts)
                break
        if best_stem is not None:
            _add_anchor(best_stem[0], best_stem[1])

        if len(w) >= 8:
            best_split: tuple[float, str, set[int], str, set[int]] | None = None
            for a in range(4, len(w) - 3):
                lk = _u61_vocab_key(w[:a])
                if sum(d for t, d in pref_terms if t.startswith(lk)) < min_split_df:
                    continue
                rk = _u61_vocab_key(w[a:])
                r_cnt_row = conn.execute(
                    "SELECT COALESCE(SUM(doc), 0) FROM main.chunk_vocab WHERE term >= ? AND term < ?",
                    (rk, rk + "\U0010ffff"),
                ).fetchone()
                if (int(r_cnt_row[0]) if r_cnt_row is not None else 0) < min_split_df:
                    continue
                left_pts = _lookup_u61_postings(
                    conn,
                    idx,
                    eligible,
                    w[:a],
                    prefix=True,
                    postings_cache=cache,
                    row_limit=word_row_limit,
                    oversized_keys=oversized,
                    all_eligible=all_eligible,
                    elig_list=elig_list,
                )
                if len(left_pts) < min_split_df or (max_df_filter is not None and len(left_pts) > max_df_filter):
                    continue
                right_pts = _lookup_u61_postings(
                    conn,
                    idx,
                    eligible,
                    w[a:],
                    prefix=True,
                    postings_cache=cache,
                    row_limit=word_row_limit,
                    oversized_keys=oversized,
                    all_eligible=all_eligible,
                    elig_list=elig_list,
                )
                if len(right_pts) < min_split_df or (max_df_filter is not None and len(right_pts) > max_df_filter):
                    continue
                score = math.sqrt(float(len(left_pts)) * float(len(right_pts)))
                if best_split is None or score > best_split[0]:
                    best_split = (score, w[:a], left_pts, w[a:], right_pts)
            if best_split is not None:
                _add_anchor(f"decomp:{best_split[1]}", best_split[2])
                _add_anchor(f"decomp:{best_split[3]}", best_split[4])

    def _ub_word_variant(w: str) -> int:
        pkey = (w, False, False)
        if (cached := cache.get(pkey)) is not None:
            return len(cached)
        if word_row_limit is not None and pkey in oversized:
            return max_df_filter + 1 if max_df_filter is not None else n_eff
        if vocab_doc_counts is None or not w.isalnum():
            return n_eff
        vk = _u61_vocab_key(w)
        with _CACHE_LOCK:
            return vocab_doc_counts.get(vk, 0)

    def _vocab_prunes_variant(w: str) -> bool:
        if vocab_doc_counts is None or not w.isalnum():
            return False
        pkey = (w, False, False)
        if pkey in cache:
            return False
        ub_v = _ub_word_variant(w)
        if ub_v == 0:
            return True
        if max_df_filter is not None and n_eff == idx.size and ub_v > max_df_filter:
            oversized.add(pkey)
            return True
        return False

    ordered_words = (
        sorted(
            unique_words,
            key=lambda rw: (min(_ub_word_variant(v) for v in _umlaut_primary_variants(rw)), -len(rw), rw),
        )
        if vocab_doc_counts is not None
        else unique_words
    )

    word_postings: dict[str, set[int]] = {}
    for raw_w in ordered_words:
        decomp_candidates: list[tuple[str, set[int]]] = []
        for w in _umlaut_primary_variants(raw_w):
            if _vocab_prunes_variant(w):
                if len(w) >= 6 and (w, False, False) not in oversized:
                    decomp_candidates.append((w, set()))
                continue
            pts = _lookup_word(w)
            if (w, False, False) not in oversized:
                decomp_candidates.append((w, pts))
            if pts:
                word_postings.setdefault(raw_w, set()).update(pts)
                _add_anchor(raw_w, pts)

        if not word_postings.get(raw_w) and (rev := _umlaut_reverse_variant(raw_w)) is not None:
            if _vocab_prunes_variant(rev):
                if len(rev) >= 6 and (rev, False, False) not in oversized:
                    decomp_candidates.append((rev, set()))
            else:
                rev_pts = _lookup_word(rev)
                if rev_pts:
                    word_postings.setdefault(raw_w, set()).update(rev_pts)
                    _add_anchor(raw_w, rev_pts)
                    if (rev, False, False) not in oversized:
                        decomp_candidates = [(rev, rev_pts)]

        for w, pts in decomp_candidates:
            _run_decomp_for_word(w, pts)

    if len(words) >= 2:
        seen_pairs: set[tuple[str, str]] = set()
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            pair = (w1, w2)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            p1 = word_postings.get(w1)
            p2 = word_postings.get(w2)
            if not p1 or not p2:
                continue
            if 2 * len(p1) > n_eff or 2 * len(p2) > n_eff:
                continue
            if max_df_filter is not None and 2 * len(p1) * len(p2) > n_eff * max_df_filter:
                continue
            inter = len(p1 & p2)
            if inter == 0 or n_eff * inter < 2 * len(p1) * len(p2):
                continue
            phrase = f"{w1} {w2}"
            p12 = (
                _lookup_u61_postings(
                    conn,
                    idx,
                    eligible,
                    phrase,
                    postings_cache=cache,
                    row_limit=word_row_limit,
                    oversized_keys=oversized,
                    all_eligible=all_eligible,
                    elig_list=elig_list,
                )
                if len(phrase) >= 3
                else (p1 & p2)
            )
            if p12 and n_eff * len(p12) >= 2 * len(p1) * len(p2):
                _add_anchor(f"pmi:{phrase}", p12)

    for run in _CJK_RUN_RE.findall(q_cf):
        if len(run) == 1:
            pts = _lookup_u61_postings(
                conn,
                idx,
                eligible,
                run,
                is_cjk=True,
                postings_cache=cache,
                oversized_keys=oversized,
                all_eligible=all_eligible,
                elig_list=elig_list,
            )
            _add_anchor(run, pts)
            continue
        bigrams: dict[str, set[int]] = {}
        for i in range(len(run) - 1):
            bg = run[i : i + 2]
            b_pts = _lookup_u61_postings(
                conn,
                idx,
                eligible,
                bg,
                is_cjk=True,
                postings_cache=cache,
                oversized_keys=oversized,
                all_eligible=all_eligible,
                elig_list=elig_list,
            )
            if b_pts:
                bigrams[bg] = b_pts
                _add_anchor(bg, b_pts)
        if len(run) >= 3:
            for span_len in (3, 4):
                for i in range(len(run) - span_len + 1):
                    sub = run[i : i + span_len]
                    b_left = bigrams.get(sub[:2])
                    b_right = bigrams.get(sub[-2:])
                    if not b_left or not b_right:
                        continue
                    s_pts = _lookup_u61_postings(
                        conn,
                        idx,
                        eligible,
                        sub,
                        is_cjk=True,
                        postings_cache=cache,
                        oversized_keys=oversized,
                        all_eligible=all_eligible,
                        elig_list=elig_list,
                    )
                    if s_pts and n_eff * len(s_pts) >= 2 * len(b_left) * len(b_right):
                        _add_anchor(f"cjk_pmi:{sub}", s_pts)

    return sorted(anchors.items(), key=lambda kv: (len(kv[1]), kv[0]))


def _extract_title_references(
    text: str,
    idx: RetrievalIndex,
    conn: sqlite3.Connection,
    eligible: NDArray[np.bool_],
    n_eff: int,
    d_query: int,
    d_bridge: int,
    *,
    all_eligible: bool,
    elig_list: list[bool] | None,
) -> list[tuple[str, tuple[int, ...], float, int]]:
    title_tbl = idx.title_by_first_word
    if not title_tbl or n_eff <= 0 or not text.strip():
        return []
    t_cf = text.casefold()
    toks = _WORD_TOK_RE.findall(t_cf)
    matched_keys: list[tuple[str, ...]] = []
    seen_tk: set[tuple[str, ...]] = set()
    i = 0
    n_toks = len(toks)
    while i < n_toks:
        entry = title_tbl.get(toks[i])
        if entry is not None:
            max_len, sub_tbl = entry
            hit_len = 0
            for span_len in range(min(max_len, n_toks - i), 0, -1):
                cand = tuple(toks[i : i + span_len])
                if cand in sub_tbl:
                    if cand not in seen_tk:
                        seen_tk.add(cand)
                        matched_keys.append(cand)
                    hit_len = span_len
                    break
            if hit_len > 0:
                i += hit_len
                continue
        i += 1

    for run in _CJK_RUN_RE.findall(t_cf):
        rlen = len(run)
        for j in range(rlen):
            for span_len in range(min(8, rlen - j), 1, -1):
                sub = run[j : j + span_len]
                entry = title_tbl.get(sub)
                if entry is not None and (sub,) in entry[1]:
                    cand = (sub,)
                    if cand not in seen_tk:
                        seen_tk.add(cand)
                        matched_keys.append(cand)
                    break

    if not matched_keys:
        return []

    filtered: list[tuple[tuple[str, ...], tuple[int, ...]]] = []
    probe_terms: list[str] = []
    for tk in matched_keys:
        pos_tup = title_tbl[tk[0]][1][tk]
        cand_pos = tuple(v for v in pos_tup if all_eligible or (elig_list[v] if elig_list is not None else eligible[v]))
        df_title = len(cand_pos)
        if df_title == 0 or df_title > d_bridge:
            continue
        filtered.append((tk, cand_pos))
        for w in tk:
            if _CJK_RUN_RE.fullmatch(w) is not None:
                continue
            vk = _u61_vocab_key(w)
            with _CACHE_LOCK:
                missing_vk = vk not in idx.vocab_doc_cache
            if missing_vk:
                probe_terms.append(vk)

    if not filtered:
        return []

    _fetch_vocab_docs(conn, probe_terms, idx.vocab_doc_cache)

    out: list[tuple[str, tuple[int, ...], float, int]] = []
    for tk, cand_pos in filtered:
        df_title = len(cand_pos)
        word_ubs: list[int] = []
        is_cjk_only = True
        with _CACHE_LOCK:
            for w in tk:
                if _CJK_RUN_RE.fullmatch(w) is not None:
                    continue
                is_cjk_only = False
                word_ubs.append(idx.vocab_doc_cache.get(_u61_vocab_key(w), n_eff))
        df_body_hat = max(df_title, min(word_ubs)) if word_ubs else df_title
        eff_q_df = d_bridge if len(tk) == 1 else d_query
        if df_body_hat > eff_q_df:
            continue
        if len(tk) == 1 and len(tk[0]) <= 2 and not is_cjk_only and df_body_hat > d_bridge:
            continue
        mi_ref = math.log(max(1.0, (float(n_eff) + 1.0) / (float(min(df_body_hat, n_eff)) + 0.5)))
        out.append((" ".join(tk), cand_pos, mi_ref, df_title))

    out.sort(key=lambda item: (-item[2], item[3], item[0]))
    return out


def lexical_search(
    idx: RetrievalIndex,
    query_text: str,
    conn: sqlite3.Connection | None,
    *,
    top_k: int = 10,
    path_glob: str | None = None,
    masked_rel_paths: Collection[str] | None = None,
) -> list[tuple[int, float]]:
    """Rank eligible chunks by pure RSJ IDF over multilingual unicode61 + PMI + decompounded anchors."""
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0 or idx.size == 0 or not query_text.strip():
        return []
    if conn is None:
        raise YawRuntimeError("CONSISTENCY_ERROR", "lexical_search requires an active SQLite connection.")
    elig = _eligible_mask(idx, path_glob=path_glob, masked_rel_paths=masked_rel_paths)
    n_eff = int(np.sum(elig))
    if n_eff == 0:
        return []

    with _snapshot_tx(conn, require_fts=True):
        anchors = _extract_multilingual_anchors(conn, idx, elig, query_text, max_df_filter=None)
    if not anchors:
        return []

    scores = np.zeros(idx.size, dtype=np.float64)
    for _, postings in anchors:
        df = len(postings)
        if df == 0:
            continue
        idf = math.log(1.0 + (n_eff - df + 0.5) / (df + 0.5))
        idx_arr = np.fromiter(postings, dtype=np.intp, count=df)
        scores[idx_arr] += idf

    nonzero = np.flatnonzero(scores > 0.0)
    if nonzero.size == 0:
        return []
    order = nonzero[np.argsort(-scores[nonzero], kind="stable")][:top_k]
    return [(idx.chunk_ids[int(i)], float(scores[int(i)])) for i in order]


def _fetch_point_vectors(
    conn: sqlite3.Connection,
    idx: RetrievalIndex,
    positions: Sequence[int],
    cache: dict[int, NDArray[np.float32]],
) -> None:
    missing = [int(p) for p in positions if int(p) not in cache]
    if not missing:
        return
    raw_list: list[tuple[int, NDArray[np.float32]]] = []
    expected_bytes = idx.dim * 2
    for p in missing:
        cid = idx.chunk_ids[p]
        row = conn.execute("SELECT embedding_f16 FROM chunks WHERE chunk_id = ?", (cid,)).fetchone()
        if row is None or row[0] is None:
            raise YawRuntimeError("CONSISTENCY_ERROR", f"Missing chunk_id={cid} in chunks table during point lookup.")
        ebytes = bytes(row[0])
        if len(ebytes) != expected_bytes:
            raise YawRuntimeError(
                "CONSISTENCY_ERROR",
                f"Invalid float16 embedding byte length {len(ebytes)} for chunk_id={cid} (expected {expected_bytes}).",
            )
        raw_list.append((p, np.frombuffer(ebytes, dtype="<f2").astype(np.float32)))
    mat = np.vstack([v for _, v in raw_list]).astype(np.float32, copy=False)
    unit = _transform_vectors(mat, idx.mean_vec)
    for k, (p, _) in enumerate(raw_list):
        cache[p] = unit[k]


def _prepare_query_vector(
    idx: RetrievalIndex,
    q_raw: NDArray[np.float32],
    *,
    exclude_vecs: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]:
    q = np.asarray(q_raw, dtype=np.float32).ravel()
    if q.shape[0] != idx.dim:
        raise YawRuntimeError("INVALID_INPUT", f"Query embedding dim {q.shape[0]} != index dim {idx.dim}")
    q_norm = float(np.linalg.norm(q))
    if q_norm <= 1e-12 or not np.all(np.isfinite(q)):
        raise YawRuntimeError("INVALID_INPUT", "Query embedding must be finite and non-zero.")
    q_unit = (q / q_norm).astype(np.float32, copy=False)
    q_vec = _transform_vectors(q_unit[None, :], idx.mean_vec)[0]
    if float(np.linalg.norm(q_vec)) <= 1e-12:
        q_vec = q_unit

    if exclude_vecs is not None and exclude_vecs.shape[0] > 0:
        q_basis, r_mat = np.linalg.qr(exclude_vecs.T.astype(np.float64), mode="reduced")
        keep = np.abs(np.diag(r_mat)) > 1e-8
        if np.any(keep):
            q_orth = q_basis[:, keep]
            q64 = q_vec.astype(np.float64)
            q_defl = q64 - q_orth @ (q_orth.T @ q64)
            d_norm = float(np.linalg.norm(q_defl))
            if d_norm > 1e-8:
                q_vec = (q_defl / d_norm).astype(np.float32)
    return q_vec


def _compute_all_query_cosines(idx: RetrievalIndex, q_hat: NDArray[np.float32]) -> NDArray[np.float32]:
    assert idx.rotator is not None
    q_rot = _rotate_vectors(idx.rotator, q_hat)
    return (_rabitq_dot_lut(idx.codes, q_rot, idx.dim) / np.maximum(idx.ip_norm, 1e-6)).astype(np.float32)


def _apply_tangent_steer_ops(
    ops: DiffusionOperators,
    all_q_cos: NDArray[np.float32],
    d_eff: float,
    n_size: int,
    t_path: float,
) -> DiffusionOperators:
    if ops.d_own is None or ops.edge_f_rows is None or ops.edge_f_rows.size == 0:
        return ops
    assert (
        ops.edge_f_cols is not None
        and ops.edge_c_ud is not None
        and ops.edge_inv_s_ud is not None
        and ops.edge_fwd_indptr is not None
        and ops.edge_t_order is not None
        and ops.edge_t_indptr is not None
        and ops.edge_t_indices is not None
    )
    f_rows = ops.edge_f_rows
    f_cols = ops.edge_f_cols
    c_ud = ops.edge_c_ud.astype(np.float64, copy=False)
    inv_s_ud = ops.edge_inv_s_ud.astype(np.float64, copy=False)
    c_qd = np.clip((ops.d_own.T @ all_q_cos).astype(np.float64), -_TANGENT_COS_CLIP, _TANGENT_COS_CLIP)
    c_qu = np.clip(all_q_cos.astype(np.float64), -_TANGENT_COS_CLIP, _TANGENT_COS_CLIP)
    inv_sq_u = 1.0 / np.sqrt(np.maximum(1e-6, 1.0 - c_qu * c_qu))
    inv_sq_d = 1.0 / np.sqrt(np.maximum(1e-6, 1.0 - c_qd * c_qd))
    cq_u = c_qu[f_rows]
    cq_d = c_qd[f_cols]
    rho_fwd = np.clip((cq_d - cq_u * c_ud) * (inv_sq_u[f_rows] * inv_s_ud), -_TANGENT_RHO_CLIP, _TANGENT_RHO_CLIP)
    rho_rev = np.clip((cq_u - cq_d * c_ud) * (inv_sq_d[f_cols] * inv_s_ud), -_TANGENT_RHO_CLIP, _TANGENT_RHO_CLIP)
    d_tan = max(2.0, d_eff - 1.0)
    if (
        ops.edge_beta_lut is not None
        and ops.edge_beta_lut_diff is not None
        and abs(ops.edge_beta_d_tan - d_tan) <= 1e-9
    ):
        beta_lut, beta_lut_diff, beta_scale = ops.edge_beta_lut, ops.edge_beta_lut_diff, ops.edge_beta_scale
    else:
        beta_lut, beta_lut_diff, beta_scale = _build_beta_nats_lut(d_tan)
    inv_t_path = 1.0 / t_path
    logits_fwd = _eval_signed_beta_nats_lut(rho_fwd, beta_lut, beta_lut_diff, beta_scale, inv_t_path)
    logits_rev = _eval_signed_beta_nats_lut(rho_rev, beta_lut, beta_lut_diff, beta_scale, inv_t_path)
    w_fwd = _grouped_softmax(f_rows, logits_fwd, n_size)
    w_rev = _grouped_softmax(f_cols, logits_rev, ops.edge_nd)
    m_fwd_t_q = sp.csr_array(
        (w_fwd[ops.edge_t_order], ops.edge_t_indices, ops.edge_t_indptr),
        shape=(ops.edge_nd, n_size),
        copy=False,
    )
    r_rev_q = sp.csr_array(
        (w_rev, f_cols, ops.edge_fwd_indptr),
        shape=(n_size, ops.edge_nd),
        copy=False,
    )
    return replace(ops, m_fwd_t=m_fwd_t_q, r_rev=r_rev_q)


def _apply_citation_step(
    ops: DiffusionOperators,
    x: NDArray[np.float32],
    el_f: NDArray[np.float32],
    size: int,
) -> NDArray[np.float32]:
    xm = (x * el_f) * ops.inv_m
    y = np.zeros(size, dtype=np.float32)
    if ops.d_own is not None:
        assert ops.m_fwd_t is not None and ops.r_rev is not None and ops.d_id_t is not None
        y = y + (ops.d_own @ (ops.m_fwd_t @ xm)) + (ops.r_rev @ (ops.d_id_t @ xm))
    if ops.p2c is not None and ops.c2p_t is not None and ops.self_pmi is not None and ops.pmi_scale is not None:
        xp = xm * ops.pmi_scale
        y = y + np.maximum(np.float32(0.0), (ops.p2c @ (ops.c2p_t @ xp)) - ops.self_pmi * xp)
    return (y * el_f).astype(np.float32, copy=False)


def _diffuse_citation(
    ops: DiffusionOperators,
    a_init: NDArray[np.float32],
    l_vec: NDArray[np.float32],
    el_f: NDArray[np.float32],
    n_eff: int,
) -> NDArray[np.float32]:
    """Converged geometric-restart diffusion (rho_stop = 0.5) until w < 1/max(2, n_eff) or L_inf <= 1e-9."""
    assert ops.d_own is None or (ops.m_fwd_t is not None and ops.r_rev is not None and ops.d_id_t is not None)
    w_stop = 1.0 / float(max(2, n_eff))
    w = 0.5
    acc = (w * a_init).astype(np.float32)
    a_cur = a_init
    size = int(a_init.size)
    while True:
        w *= 0.5
        if w < w_stop:
            break
        step_vec = _apply_citation_step(ops, a_cur, el_f, size)
        raw = step_vec * l_vec
        s = float(raw.sum())
        if s <= 1e-12:
            break
        a_next = (raw / s).astype(np.float32, copy=False)
        acc = acc + w * a_next
        if float(np.abs(a_next - a_cur).max()) <= 1e-9:
            acc = acc + w * a_next
            break
        a_cur = a_next
    st = float(acc.sum())
    return (acc / st).astype(np.float32, copy=False) if st > 1e-15 else acc


def _rerank_dpp_1bit(
    idx: RetrievalIndex,
    post: NDArray[np.float32],
    pool: Sequence[int],
    *,
    top_k: int,
    n_eff: int,
) -> list[int]:
    """Rank-1 Cholesky greedy L-ensemble 1-bit DPP over distinct-doc pool (d_1 = argmax post)."""
    m = len(pool)
    k_sel = min(top_k, m)
    if k_sel <= 1:
        return list(pool[:k_sel])
    x_bar = _unpack_1bit_signed(idx.codes, pool, idx.dim).astype(np.float64)
    gram = x_bar @ x_bar.T
    w = float(n_eff) * np.maximum(post[list(pool)].astype(np.float64), 0.0)
    diags = np.ones(m, dtype=np.float64)
    chol = np.zeros((m, k_sel), dtype=np.float64)
    n_cols = 0
    avail = [True] * m
    selected: list[int] = []
    for step in range(k_sel):
        if step == 0:
            best = 0
        else:
            best = max(
                (i for i in range(m) if avail[i]),
                key=lambda i: (
                    math.log1p(float(w[i]) * min(1.0, max(0.0, float(diags[i])))),
                    -idx.chunk_ids[pool[i]],
                ),
            )
        avail[best] = False
        selected.append(pool[best])
        if step + 1 >= k_sel:
            break
        if diags[best] > 1e-14:
            d_val = math.sqrt(float(diags[best]))
            if d_val > 1e-7:
                proj = chol[:, :n_cols] @ chol[best, :n_cols] if n_cols > 0 else 0.0
                col = (gram[:, best] - proj) / d_val
                chol[:, n_cols] = col
                n_cols += 1
                diags = np.maximum(0.0, diags - col * col)
    return selected


def _sigmoid(x: float) -> float:
    if x >= 30.0:
        return 1.0
    if x <= -30.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def search_index(
    idx: RetrievalIndex,
    query_embedding: NDArray[np.float32],
    *,
    query_text: str = "",
    conn: sqlite3.Connection | None = None,
    top_k: int = 10,
    path_glob: str | None = None,
    masked_rel_paths: Collection[str] | None = None,
    exclude_chunk_ids: Sequence[int] | None = None,
) -> SearchResult:
    """Run frozen M_FINAL_1B retrieval: 1-bit RaBitQ, unicode61 lexical/title anchors, tangent diffusion, and DPP."""
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise YawRuntimeError("INVALID_INPUT", f"top_k must be a positive integer, got {top_k!r}")
    if conn is None:
        raise YawRuntimeError("CONSISTENCY_ERROR", "search_index requires an active SQLite connection.")
    if idx.size == 0:
        return SearchResult((), -math.inf, 0.0, 0.0, idx.d_eff, 0)
    if idx.citation_graph is None or idx.diffusion_ops is None or idx.rotator is None:
        raise YawRuntimeError("CONSISTENCY_ERROR", "RetrievalIndex is missing CitationGraph or DiffusionOperators.")

    eligible = _eligible_mask(idx, path_glob=path_glob, masked_rel_paths=masked_rel_paths)
    excl_pos: list[int] = []
    if exclude_chunk_ids:
        for cid in exclude_chunk_ids:
            if (p := idx.id_to_pos.get(int(cid))) is not None:
                eligible[p] = False
                excl_pos.append(p)

    n_eff = int(np.sum(eligible))
    if n_eff == 0:
        q_chk = np.asarray(query_embedding, dtype=np.float32).ravel()
        if q_chk.shape[0] != idx.dim or float(np.linalg.norm(q_chk)) <= 1e-12 or not np.all(np.isfinite(q_chk)):
            raise YawRuntimeError("INVALID_INPUT", "Query embedding must match dim, be finite, and non-zero.")
        with _snapshot_tx(conn, require_fts=True):
            pass
        return SearchResult((), -math.inf, 0.0, 0.0, idx.d_eff, 0)

    all_elig = n_eff == idx.size
    ops = (
        idx.diffusion_ops
        if all_elig
        else _build_diffusion_operators(
            idx.citation_graph,
            idx.path_idx,
            codes=idx.codes,
            dim=idx.dim,
            d_eff=idx.d_eff,
            eligible=eligible,
        )
    )
    d_bridge = max(4, math.ceil(math.sqrt(n_eff)))
    d_query = max(8, n_eff // 2)
    t_path = math.sqrt(max(1.0, idx.d_eff) / max(1.0, math.log(float(n_eff) + 1.0)))
    vec_cache: dict[int, NDArray[np.float32]] = {}

    with _snapshot_tx(conn, require_fts=True):
        if excl_pos:
            _fetch_point_vectors(conn, idx, excl_pos, vec_cache)
            excl_mat = np.vstack([vec_cache[p] for p in excl_pos])
        else:
            excl_mat = None
        q_hat = _prepare_query_vector(idx, query_embedding, exclude_vecs=excl_mat)
        all_q_cos = _compute_all_query_cosines(idx, q_hat)
        elig_list = None if all_elig else eligible.tolist()
        q_eff = _norm_entity_text(query_text)
        q_anchors = _extract_multilingual_anchors(conn, idx, eligible, q_eff, max_df_filter=d_query)
        q_trefs = _extract_title_references(
            q_eff,
            idx,
            conn,
            eligible,
            n_eff,
            d_query,
            d_bridge,
            all_eligible=all_elig,
            elig_list=elig_list,
        )

    sem_n = _parametric_null_nats_vec(np.maximum(0.0, np.where(eligible, all_q_cos.astype(np.float64), 0.0)), idx.d_eff)
    lex_n = np.zeros(idx.size, dtype=np.float64)
    for _, pts in q_anchors:
        idf_w = math.log((float(n_eff) + 1.0) / (float(len(pts)) + 0.5))
        for p in pts:
            lex_n[p] += idf_w
    for _, pt, _, _ in q_trefs:
        w_t = math.log((float(n_eff) + 1.0) / (float(len(pt)) + 0.5))
        for p in pt:
            lex_n[p] += w_t

    ev = np.where(eligible, (sem_n + lex_n) / t_path, -np.inf)
    ev_max = float(ev[eligible].max())
    ex = np.where(eligible, np.exp(ev - ev_max), 0.0)
    ex_sum = max(float(ex.sum()), 1e-12)
    a0 = (ex / ex_sum).astype(np.float32)

    ops = _apply_tangent_steer_ops(ops, np.where(eligible, all_q_cos, np.float32(0.0)), idx.d_eff, idx.size, t_path)
    el_f = eligible.astype(np.float32)
    mx_sem = float((sem_n[eligible] / t_path).max())
    l_vec = np.where(eligible, np.exp((sem_n / t_path) - mx_sem), 0.0).astype(np.float32)
    post = _diffuse_citation(ops, a0, l_vec, el_f, n_eff)

    if len(q_trefs) >= 2:
        nrel = 1.0 - np.clip(post.astype(np.float64), 0.0, 1.0)
        log_n1 = math.log(float(n_eff) + 1.0)
        for _, pt, _, _ in q_trefs:
            ept = [p for p in pt if eligible[p]]
            if not ept:
                continue
            wg = _sigmoid(max(float(sem_n[p]) for p in ept) - log_n1)
            sc_g = np.array([sem_n[p] / t_path for p in ept], dtype=np.float64)
            ex_g = np.exp(sc_g - float(sc_g.max()))
            ex_g /= max(float(ex_g.sum()), 1e-12)
            ag = np.zeros(idx.size, dtype=np.float32)
            for k_p, p in enumerate(ept):
                ag[p] = float(ex_g[k_p])
            pg = _diffuse_citation(ops, ag, l_vec, el_f, n_eff)
            nrel *= 1.0 - wg * np.clip(pg.astype(np.float64), 0.0, 1.0)
        post = (1.0 - nrel).astype(np.float32)

    def _order_candidates(pos_list: Sequence[int]) -> tuple[list[int], int]:
        ordered = sorted(pos_list, key=lambda p: (-float(post[p]), -float(all_q_cos[p]), idx.chunk_ids[p]))
        seen_rels: set[str] = set()
        first_pass: list[int] = []
        rest_pass: list[int] = []
        for p in ordered:
            rel_p = idx.pointers[p].rel_posix_path
            if rel_p not in seen_rels:
                seen_rels.add(rel_p)
                first_pass.append(p)
            else:
                rest_pass.append(p)
        return first_pass + rest_pass, len(first_pass)

    elig_pos = np.flatnonzero(eligible)
    n_elig_int = int(elig_pos.size)
    m_law = math.ceil(2.0 * math.log(float(n_eff) + 1.0))
    m_target = max(top_k, m_law)
    k_pool = min(n_elig_int, max(m_target * 8, 256))
    post_e = post[elig_pos]
    sub_idx = (
        np.argpartition(-post_e, k_pool - 1)[:k_pool] if k_pool < n_elig_int else np.arange(n_elig_int, dtype=np.int64)
    )
    order, n_distinct_found = _order_candidates([int(elig_pos[int(i)]) for i in sub_idx])
    p_cut = float(post_e[sub_idx].min()) if k_pool < n_elig_int else -math.inf
    need_n = min(m_target, n_elig_int)
    if k_pool < n_elig_int and need_n > 0 and (n_distinct_found < need_n or float(post[order[need_n - 1]]) <= p_cut):
        order, n_distinct_found = _order_candidates([int(i) for i in elig_pos])

    if len(order) > 1:
        m_pool = min(n_distinct_found, m_law)
        pool = order[:m_pool]
        reranked = pool if m_pool <= 1 else _rerank_dpp_1bit(idx, post, pool, top_k=top_k, n_eff=n_eff)
        if len(reranked) < min(top_k, len(order)):
            reranked = reranked + order[m_pool : m_pool + (min(top_k, len(order)) - len(reranked))]
        selected_pos = reranked
    else:
        selected_pos = order[: min(top_k, len(order))]

    ranked_list: list[tuple[int, float]] = []
    prev_sc = math.inf
    for p in selected_pos:
        raw_sc = float(post[p])
        sc = raw_sc if not ranked_list else min(prev_sc - 1e-9, raw_sc)
        prev_sc = sc
        ranked_list.append((idx.chunk_ids[p], sc))
    ranked_out = tuple(ranked_list)

    top1_cos = float(all_q_cos[eligible].max())
    log_bf = float(ev_max + math.log(ex_sum) - math.log(float(n_eff)))
    return SearchResult(
        ranked=ranked_out,
        log_bf=log_bf,
        max_chunk_score=ranked_out[0][1] if ranked_out else 0.0,
        top1_cosine=top1_cos,
        d_eff=idx.d_eff,
        n_candidates=len(selected_pos),
    )


def expand_citation_neighbors(
    idx: RetrievalIndex,
    chunk_id: int,
    *,
    corpus_root: Path | str,
    conn: sqlite3.Connection | None = None,
    top_k: int = 6,
    path_glob: str | None = None,
    masked_rel_paths: Collection[str] | None = None,
) -> list[tuple[int, float]]:
    """Expand isotropic 1-bit cosines, lexical IDF, tangent-steered citation diffusion, and AST-island nats."""
    if isinstance(chunk_id, bool) or not isinstance(chunk_id, int):
        raise YawRuntimeError("INVALID_INPUT", f"chunk_id must be an integer, got {chunk_id!r}")
    if (pos := idx.id_to_pos.get(chunk_id)) is None:
        raise YawRuntimeError("INVALID_INPUT", f"Unknown chunk_id={chunk_id}.")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise YawRuntimeError("INVALID_INPUT", f"top_k must be a positive integer, got {top_k!r}")
    if conn is None:
        raise YawRuntimeError("CONSISTENCY_ERROR", "expand_citation_neighbors requires an active SQLite connection.")
    if idx.size <= 1:
        return []
    if idx.citation_graph is None or idx.diffusion_ops is None or idx.rotator is None:
        raise YawRuntimeError("CONSISTENCY_ERROR", "RetrievalIndex is missing CitationGraph or DiffusionOperators.")

    eligible = _eligible_mask(idx, path_glob=path_glob, masked_rel_paths=masked_rel_paths)
    eligible[pos] = False
    elig_pos = np.flatnonzero(eligible)
    if elig_pos.size == 0:
        return []

    c_root = corpus_root if isinstance(corpus_root, Path) and corpus_root.is_absolute() else Path(corpus_root).resolve()
    src_hp = _hydrate_one(
        c_root,
        idx.pointers[pos],
        0.0,
        None,
        False,
        False,
        strict=True,
        path_cache=idx.resolved_path_cache,
    )
    vec_cache: dict[int, NDArray[np.float32]] = {}
    n_elig_total = int(elig_pos.size) + 1
    lex_cap = min(int(elig_pos.size), max(top_k, math.ceil(2.0 * math.log(float(n_elig_total) + 1.0))))
    with _snapshot_tx(conn, require_fts=True):
        _fetch_point_vectors(conn, idx, [pos], vec_cache)
        u_vec = vec_cache[pos]
        all_u_cos = _compute_all_query_cosines(idx, u_vec)
        cos_scores = all_u_cos[elig_pos].astype(np.float64)
        lex_hits = lexical_search(
            idx,
            src_hp.text,
            conn,
            top_k=lex_cap,
            path_glob=path_glob,
            masked_rel_paths=masked_rel_paths,
        )

    sem_nats = _parametric_null_nats_vec(np.maximum(0.0, cos_scores), idx.d_eff)
    cos_by_pos: dict[int, float] = {}
    score_by_pos: dict[int, float] = {}
    for i, ep in enumerate(elig_pos):
        p_int = int(ep)
        cos_by_pos[p_int] = float(cos_scores[i])
        score_by_pos[p_int] = float(sem_nats[i])
    for cid, lv in lex_hits:
        if (p := idx.id_to_pos.get(cid)) is not None and p in score_by_pos:
            score_by_pos[p] += float(lv)

    elig_with_src = eligible.copy()
    elig_with_src[pos] = True
    t_path = math.sqrt(max(1.0, idx.d_eff) / max(1.0, math.log(float(n_elig_total) + 1.0)))
    ops = (
        idx.diffusion_ops
        if int(elig_with_src.sum()) == idx.size
        else _build_diffusion_operators(
            idx.citation_graph,
            idx.path_idx,
            codes=idx.codes,
            dim=idx.dim,
            d_eff=idx.d_eff,
            eligible=elig_with_src,
        )
    )
    ops = _apply_tangent_steer_ops(
        ops,
        np.where(elig_with_src, all_u_cos, np.float32(0.0)),
        idx.d_eff,
        idx.size,
        t_path,
    )
    sem_full = np.zeros(idx.size, dtype=np.float64)
    sem_full[elig_pos] = sem_nats
    sem_full[pos] = float(sem_nats.max()) if elig_pos.size > 0 else 0.0
    mx_sem = float((sem_full[elig_with_src] / t_path).max())
    l_vec = np.where(elig_with_src, np.exp((sem_full / t_path) - mx_sem), 0.0).astype(np.float32)
    seed = np.zeros(idx.size, dtype=np.float32)
    seed[pos] = 1.0
    p_diff = _diffuse_citation(ops, seed, l_vec, elig_with_src.astype(np.float32), n_elig_total)
    for ep in elig_pos:
        p_int = int(ep)
        pd = float(p_diff[p_int])
        if pd > 0.0:
            score_by_pos[p_int] += math.log1p(float(n_elig_total) * pd)

    ptr = idx.pointers[pos]
    if (ptr.ast_island_start, ptr.ast_island_end) != (1, 1):
        island = (ptr.rel_posix_path, ptr.ast_island_start, ptr.ast_island_end)
        same_island_pos = [
            p
            for p in score_by_pos
            if (idx.pointers[p].rel_posix_path, idx.pointers[p].ast_island_start, idx.pointers[p].ast_island_end)
            == island
        ]
        if same_island_pos:
            island_nats = math.log(max(1.0, float(n_elig_total) / float(len(same_island_pos) + 1)))
            for p in same_island_pos:
                score_by_pos[p] += island_nats

    ordered = sorted(
        score_by_pos.items(),
        key=lambda kv: (-kv[1], -cos_by_pos[kv[0]], idx.chunk_ids[kv[0]]),
    )[:top_k]
    return [(idx.chunk_ids[p], sc) for p, sc in ordered]
