"""Hybrid sqlite-vec + FTS5 retrieval, SciPy entity PPR, and zero-copy disk hydration."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import threading
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import spacy
import sqlite_vec
from numpy.typing import NDArray

from yaw.core import (
    ChunkPointer,
    HydratedPassage,
    YawRuntimeError,
    normalize_entity,
    resolve_corpus_path,
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_PROPER_SPAN_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9_-]*|[A-Z]{2,}[0-9]*)(?:\s+(?:[A-Z][A-Za-z0-9_-]*|[A-Z]{2,}[0-9]*))*"
)
_FTS_SAFE_RE = re.compile(r"[A-Za-z0-9_]+")
_STOPWORDS = frozenset(
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
        "them",
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
_SENTENCE_INITIAL_SKIP = frozenset(
    {"The", "This", "That", "These", "Those", "What", "When", "Where", "Which", "Who", "Why", "How", "In", "On", "At"}
)
_MAX_HYDRATE_SPAN = 10_000
_NLP_LOCK = threading.Lock()
_NLP: Any = None


def _get_spacy_nlp() -> Any:
    global _NLP
    if _NLP is None:
        with _NLP_LOCK:
            if _NLP is None:
                try:
                    _NLP = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer", "textcat"])
                except OSError:
                    _NLP = spacy.blank("en")
    return _NLP


def extract_text_features(title: str, text: str) -> tuple[dict[str, int], tuple[str, ...]]:
    """Extract content-term counts and normalized named/proper-noun entity keys."""
    combined = f"{title}\n{text}" if title else text
    terms = [t for t in _TOKEN_RE.findall(combined.lower()) if len(t) > 1 and t not in _STOPWORDS]
    counts = dict(Counter(terms))

    ents: dict[str, None] = {}
    if title and (clean_t := normalize_entity(title)) and clean_t not in _STOPWORDS:
        ents[clean_t] = None
    nlp = _get_spacy_nlp()
    if "ner" in nlp.pipe_names:
        doc = nlp(combined[:25_000])
        for ent in doc.ents:
            norm = normalize_entity(ent.text)
            if len(norm) > 1 and norm not in _STOPWORDS:
                ents[norm] = None
    for match in _PROPER_SPAN_RE.finditer(combined):
        raw = match.group(0).strip()
        if raw in _SENTENCE_INITIAL_SKIP:
            continue
        norm = normalize_entity(raw)
        if len(norm) > 1 and norm not in _STOPWORDS:
            ents[norm] = None
    return counts, tuple(ents)


def open_readonly_db(db_path: Path | str) -> sqlite3.Connection:
    """Open a read-only SQLite connection with sqlite-vec loaded."""
    target = Path(db_path).resolve()
    if not target.is_file():
        raise YawRuntimeError("INDEX_NOT_READY", f"SQLite database does not exist: {target}")
    conn = sqlite3.connect(f"file:{target.as_posix()}?mode=ro", uri=True, check_same_thread=False)
    try:
        conn.execute("PRAGMA busy_timeout=30000")
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except Exception:
        conn.close()
        raise
    return conn


@dataclass(slots=True)
class VectorSearch:
    """In-memory CSR entity-chunk graph and chunk metadata view backed by SQLite."""

    chunk_ids: tuple[int, ...]
    id_to_pos: dict[int, int]
    pointers: tuple[ChunkPointer, ...]
    embeddings: NDArray[np.float32]
    entity_keys: tuple[str, ...]
    entity_to_idx: dict[str, int]
    entity_to_chunk: sp.csr_matrix
    chunk_to_entity: sp.csr_matrix
    doc_freqs: dict[str, int]
    chunk_tokens: tuple[dict[str, int], ...]
    chunk_lengths: NDArray[np.float32]
    avg_chunk_len: float

    @property
    def size(self) -> int:
        return len(self.chunk_ids)

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1]) if self.embeddings.ndim == 2 and self.embeddings.size else 0


def build_vector_search(conn: sqlite3.Connection) -> VectorSearch:
    """Load chunk metadata, normalized vectors, and sparse entity graph from SQLite `chunks`."""
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunks'").fetchone()
    if row is None:
        empty_csr = sp.csr_matrix((0, 0), dtype=np.float32)
        return VectorSearch(
            (),
            {},
            (),
            np.zeros((0, 0), dtype=np.float32),
            (),
            {},
            empty_csr,
            empty_csr,
            {},
            (),
            np.zeros(0, dtype=np.float32),
            0.0,
        )

    rows = conn.execute(
        "SELECT chunk_id, embedding, rel_path, source_hash, start_line, end_line, "
        "char_start, char_end, ast_island_start, ast_island_end, title, tokens_json, entities_json "
        "FROM chunks ORDER BY chunk_id ASC"
    ).fetchall()
    if not rows:
        empty_csr = sp.csr_matrix((0, 0), dtype=np.float32)
        return VectorSearch(
            (),
            {},
            (),
            np.zeros((0, 0), dtype=np.float32),
            (),
            {},
            empty_csr,
            empty_csr,
            {},
            (),
            np.zeros(0, dtype=np.float32),
            0.0,
        )

    chunk_ids: list[int] = []
    pointers: list[ChunkPointer] = []
    vecs: list[NDArray[np.float32]] = []
    chunk_tokens: list[dict[str, int]] = []
    doc_freqs: dict[str, int] = {}
    ent_to_idx: dict[str, int] = {}
    ent_keys: list[str] = []
    row_ind: list[int] = []
    col_ind: list[int] = []
    lengths = np.zeros(len(rows), dtype=np.float32)

    for pos, r in enumerate(rows):
        cid = int(r[0])
        raw_blob = bytes(r[1])
        vec = np.frombuffer(raw_blob, dtype="<f4").copy()
        norm = float(np.linalg.norm(vec))
        if norm > 1e-12:
            vec /= norm
        vecs.append(vec)
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
        toks_raw = json.loads(r[11]) if r[11] else {}
        tok_map = {str(k): int(v) for k, v in toks_raw.items()} if isinstance(toks_raw, dict) else {}
        chunk_tokens.append(tok_map)
        lengths[pos] = float(sum(tok_map.values()))
        for term in tok_map:
            doc_freqs[term] = doc_freqs.get(term, 0) + 1

        ents_raw = json.loads(r[12]) if r[12] else []
        seen_local: set[int] = set()
        for e in ents_raw:
            ne = normalize_entity(str(e))
            if not ne:
                continue
            e_idx = ent_to_idx.get(ne)
            if e_idx is None:
                e_idx = len(ent_keys)
                ent_to_idx[ne] = e_idx
                ent_keys.append(ne)
            if e_idx not in seen_local:
                seen_local.add(e_idx)
                row_ind.append(e_idx)
                col_ind.append(pos)

    n_chunks = len(chunk_ids)
    n_ents = len(ent_keys)
    mat = np.vstack(vecs).astype(np.float32, copy=False)
    if n_ents and row_ind:
        vals = np.ones(len(row_ind), dtype=np.float32)
        incidence = sp.csr_matrix((vals, (row_ind, col_ind)), shape=(n_ents, n_chunks), dtype=np.float32)
        # Discount hub entities by 1 / sqrt(df) before column-stochastic normalization.
        ent_df = np.asarray(incidence.sum(axis=1)).ravel()
        hub_w = 1.0 / np.sqrt(np.maximum(ent_df, 1.0))
        weighted = sp.diags(hub_w.astype(np.float32)) @ incidence
        e_sums = np.maximum(np.asarray(incidence.sum(axis=1)).ravel(), 1e-12)
        e2c = (sp.diags((1.0 / e_sums).astype(np.float32)) @ incidence).tocsr()
        c_sums = np.maximum(np.asarray(weighted.sum(axis=0)).ravel(), 1e-12)
        c2e = (weighted @ sp.diags((1.0 / c_sums).astype(np.float32))).tocsr()
    else:
        e2c = sp.csr_matrix((n_ents, n_chunks), dtype=np.float32)
        c2e = sp.csr_matrix((n_ents, n_chunks), dtype=np.float32)

    avg_len = float(lengths.mean()) if n_chunks else 0.0
    id_to_pos = {cid: idx for idx, cid in enumerate(chunk_ids)}
    return VectorSearch(
        chunk_ids=tuple(chunk_ids),
        id_to_pos=id_to_pos,
        pointers=tuple(pointers),
        embeddings=mat,
        entity_keys=tuple(ent_keys),
        entity_to_idx=ent_to_idx,
        entity_to_chunk=e2c,
        chunk_to_entity=c2e,
        doc_freqs=doc_freqs,
        chunk_tokens=tuple(chunk_tokens),
        chunk_lengths=lengths,
        avg_chunk_len=max(avg_len, 1.0),
    )


def _minmax_norm(arr: NDArray[np.float32]) -> NDArray[np.float32]:
    if arr.size == 0:
        return arr
    hi = float(arr.max())
    lo = float(arr.min())
    if hi <= 1e-12:
        return np.zeros_like(arr)
    if hi - lo <= 1e-12:
        return np.where(arr > 0.0, 1.0, 0.0).astype(np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def _bm25_scores(vs: VectorSearch, query_text: str, conn: sqlite3.Connection | None) -> NDArray[np.float32]:
    n = vs.size
    scores = np.zeros(n, dtype=np.float32)
    if n == 0 or not query_text.strip():
        return scores

    if conn is not None:
        safe_terms = [t for t in _FTS_SAFE_RE.findall(query_text) if len(t) > 1 and t.lower() not in _STOPWORDS]
        if safe_terms:
            match_expr = " OR ".join(f'"{t}"' for t in dict.fromkeys(safe_terms))
            try:
                fts_rows = conn.execute(
                    "SELECT rowid, -bm25(chunk_fts, 2.0, 1.0) FROM chunk_fts WHERE chunk_fts MATCH ? LIMIT ?",
                    (match_expr, min(n, 256)),
                ).fetchall()
                for rid, val in fts_rows:
                    pos = vs.id_to_pos.get(int(rid))
                    if pos is not None and float(val) > 0.0:
                        scores[pos] = float(val)
                if float(scores.max()) > 0.0:
                    return _minmax_norm(scores)
            except sqlite3.OperationalError:
                pass

    q_terms = [t for t in _TOKEN_RE.findall(query_text.lower()) if len(t) > 1 and t not in _STOPWORDS]
    if not q_terms:
        return scores
    k1, b = 1.5, 0.75
    for term in dict.fromkeys(q_terms):
        df = vs.doc_freqs.get(term, 0)
        if df <= 0:
            continue
        idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        for pos, tok_map in enumerate(vs.chunk_tokens):
            tf = tok_map.get(term, 0)
            if tf > 0:
                denom = tf + k1 * (1.0 - b + b * (float(vs.chunk_lengths[pos]) / vs.avg_chunk_len))
                scores[pos] += float(idf * (tf * (k1 + 1.0)) / denom)
    return _minmax_norm(scores)


def _dense_scores(
    vs: VectorSearch, q_vec: NDArray[np.float32], conn: sqlite3.Connection | None, limit: int
) -> NDArray[np.float32]:
    n = vs.size
    scores = np.zeros(n, dtype=np.float32)
    if n == 0:
        return scores
    q = np.asarray(q_vec, dtype=np.float32).ravel()
    if q.shape[0] != vs.dim:
        raise YawRuntimeError("INVALID_INPUT", f"Query embedding dim {q.shape[0]} != index dim {vs.dim}")
    q_norm = float(np.linalg.norm(q))
    if q_norm <= 1e-12 or not np.all(np.isfinite(q)):
        raise YawRuntimeError("INVALID_INPUT", "Query embedding must be finite and non-zero.")
    q = (q / q_norm).astype(np.float32, copy=False)

    if conn is not None:
        k = min(n, max(limit * 4, 64))
        vec_rows = conn.execute(
            "SELECT chunk_id, distance FROM chunks WHERE embedding MATCH ? AND k = ?",
            (q.astype("<f4", copy=False).tobytes(), k),
        ).fetchall()
        for cid, dist in vec_rows:
            pos = vs.id_to_pos.get(int(cid))
            if pos is not None:
                scores[pos] = max(0.0, 1.0 - float(dist))
        return _minmax_norm(scores)

    raw = np.maximum(0.0, vs.embeddings @ q)
    return _minmax_norm(raw.astype(np.float32, copy=False))


def lexical_search(
    vs: VectorSearch, query_text: str, conn: sqlite3.Connection | None, *, top_k: int
) -> list[tuple[int, float]]:
    """Rank chunks by FTS5/BM25 keyword score alone, with no query embedding."""
    scores = _bm25_scores(vs, query_text, conn)
    order = np.argsort(-scores, kind="stable")[: max(top_k, 0)]
    return [(vs.chunk_ids[int(i)], float(scores[i])) for i in order if scores[i] > 0.0]


def search_vectors(
    vs: VectorSearch,
    query_embedding: NDArray[np.float32],
    *,
    query_text: str = "",
    conn: sqlite3.Connection | None = None,
    top_k: int = 10,
    seed_top_k: int = 20,
    hybrid_alpha: float = 0.7,
    ppr_alpha: float = 0.85,
    max_iter: int = 25,
    tol: float = 1e-6,
) -> list[tuple[int, float]]:
    """Rank chunks via sqlite-vec + FTS5 hybrid seeds diffused over entity-chunk SciPy PPR."""
    if top_k <= 0 or vs.size == 0:
        return []
    alpha_h = float(np.clip(hybrid_alpha, 0.0, 1.0))
    dense = _dense_scores(vs, query_embedding, conn, max(top_k, seed_top_k))
    lexical = _bm25_scores(vs, query_text, conn) if query_text else np.zeros(vs.size, dtype=np.float32)
    hybrid = (alpha_h * dense + (1.0 - alpha_h) * lexical).astype(np.float32) if query_text else dense

    n_ents = len(vs.entity_keys)
    if n_ents > 0 and vs.entity_to_chunk.nnz > 0 and ppr_alpha > 0.0:
        k_seed = min(vs.size, max(1, seed_top_k))
        top_pos = np.argpartition(-hybrid, k_seed - 1)[:k_seed]
        c_seed = np.zeros(vs.size, dtype=np.float32)
        c_seed[top_pos] = np.maximum(hybrid[top_pos], 0.0)

        # Anchor query entities directly in entity space alongside top hybrid passage seeds.
        e_seed = np.asarray(vs.chunk_to_entity @ c_seed).ravel()
        if query_text:
            _, q_ents = extract_text_features("", query_text)
            for qe in q_ents:
                if (ei := vs.entity_to_idx.get(qe)) is not None:
                    e_seed[ei] += 1.0
            for q_tok in _TOKEN_RE.findall(query_text.lower()):
                if len(q_tok) > 2 and q_tok not in _STOPWORDS and (ei := vs.entity_to_idx.get(q_tok)) is not None:
                    e_seed[ei] += 0.5

        seed_sum = float(e_seed.sum())
        if seed_sum > 1e-12:
            p0 = (e_seed / seed_sum).astype(np.float32)
            p = p0.copy()
            damp = float(np.clip(ppr_alpha, 0.0, 0.99))
            for _ in range(max(1, max_iter)):
                c_prob = vs.entity_to_chunk.T @ p
                e_next = vs.chunk_to_entity @ c_prob
                nxt_sum = float(e_next.sum())
                # Redistribute dangling probability mass back to teleportation vector p0.
                leaked = max(0.0, 1.0 - nxt_sum)
                p_next = (1.0 - damp) * p0 + damp * (e_next + leaked * p0)
                if float(np.abs(p_next - p).sum()) <= tol:
                    p = p_next
                    break
                p = p_next
            c_ppr = _minmax_norm(np.asarray(vs.entity_to_chunk.T @ p).ravel().astype(np.float32))
            final = (0.65 * hybrid + 0.35 * c_ppr).astype(np.float32)
        else:
            final = hybrid
    else:
        final = hybrid

    limit = min(vs.size, top_k)
    order = np.argsort(-final, kind="stable")[:limit]
    return [(vs.chunk_ids[int(i)], float(final[int(i)])) for i in order]


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


type _FileCache = dict[Path, tuple[str, list[str]] | None]


def _hydrate_one(
    corpus_root: Path,
    ptr: ChunkPointer,
    score: float,
    max_chars: int | None,
    include_island: bool,
    full_span: bool,
    file_cache: _FileCache | None = None,
) -> HydratedPassage:
    limit = _validate_max_chars(max_chars)
    sl, el = int(ptr.start_line), int(ptr.end_line)
    if sl < 1 or el < sl or (el - sl + 1) > _MAX_HYDRATE_SPAN:
        raise ValueError(f"Invalid hydration line range [{sl}, {el}] for {ptr.rel_posix_path}")
    target = resolve_corpus_path(corpus_root, ptr.rel_posix_path)

    if file_cache is not None and target in file_cache:
        loaded = file_cache[target]
    else:
        if target.is_file():
            norm_text = target.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
            loaded = (norm_text, norm_text.splitlines())
        else:
            loaded = None
        if file_cache is not None:
            file_cache[target] = loaded
    if loaded is None:
        return HydratedPassage(ptr, "", float(score), stale_on_disk=True)

    text, lines = loaded
    if el > len(lines):
        return HydratedPassage(ptr, "", float(score), stale_on_disk=True)
    raw_slice = "\n".join(lines[sl - 1 : el])
    if ptr.source_hash and hashlib.sha256(raw_slice.encode("utf-8")).hexdigest() != ptr.source_hash:
        return HydratedPassage(ptr, "", float(score), stale_on_disk=True)

    # Hydrate pure line/char spans uniformly across all file types without serde JSON parsing.
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
    return _hydrate_one(corpus_root, ptr, score, max_chars, include_island, full_span)


def hydrate_ranked_passages(
    vs: VectorSearch,
    ranked_chunks: Sequence[tuple[int, float]],
    corpus_root: Path,
    *,
    top_k: int = 5,
    max_chars: int | None = None,
    include_island: bool = False,
    full_span: bool = False,
) -> tuple[HydratedPassage, ...]:
    """Apply MaxP deduplication by span and hydrate top `top_k` passages."""
    limit = _validate_max_chars(max_chars)
    if top_k <= 0 or not ranked_chunks:
        return ()
    out: list[HydratedPassage] = []
    seen_spans: set[tuple[str, int, int, int, int]] = set()
    file_cache: _FileCache = {}
    for cid, score in ranked_chunks:
        if (pos := vs.id_to_pos.get(int(cid))) is None:
            continue
        ptr = vs.pointers[pos]
        span = (ptr.rel_posix_path, int(ptr.start_line), int(ptr.end_line), int(ptr.char_start), int(ptr.char_end))
        if span in seen_spans:
            continue
        seen_spans.add(span)
        out.append(_hydrate_one(corpus_root, ptr, float(score), limit, include_island, full_span, file_cache))
        if len(out) >= top_k:
            break
    return tuple(out)


def format_evidence_route(passages: Sequence[HydratedPassage]) -> str:
    """Format hydrated passages with explicit provenance headers for ReAct / recall output."""
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


def query_external_hints(
    conn: sqlite3.Connection,
    user_tables: Sequence[str],
    query_terms: Sequence[str],
    *,
    limit: int = 5,
) -> list[str]:
    """Probe non-chunk user SQLite tables for rows matching lexical query terms."""
    hints: list[str] = []
    clean_terms = [t.lower() for t in query_terms if len(t) > 2 and t.lower() not in _STOPWORDS][:4]
    if not clean_terms or limit <= 0:
        return hints
    for tbl in user_tables:
        if tbl in ("chunks", "chunk_fts") or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", tbl):
            continue
        try:
            cols = [str(r[1]) for r in conn.execute(f'PRAGMA table_info("{tbl}")').fetchall()]
            text_cols = [c for c in cols if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", c)]
            if not text_cols:
                continue
            where = " OR ".join(f'LOWER(CAST("{c}" AS TEXT)) LIKE ?' for c in text_cols[:4] for _ in clean_terms)
            params = [f"%{t}%" for _ in text_cols[:4] for t in clean_terms]
            rows = conn.execute(f'SELECT * FROM "{tbl}" WHERE {where} LIMIT ?', (*params, limit)).fetchall()
            for r in rows:
                rendered = ", ".join(f"{c}={v}" for c, v in zip(cols, r, strict=False) if v is not None)
                hints.append(f"[{tbl}] {rendered[:240]}")
                if len(hints) >= limit:
                    return hints
        except sqlite3.OperationalError:
            continue
    return hints
