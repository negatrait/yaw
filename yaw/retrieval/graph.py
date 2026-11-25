"""Citation graph construction, YCG1 binary serialization, and diffusion operator builders."""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import zlib
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from yaw.core import DASH_RE, ChunkPointer, YawRuntimeError, require_corpus_dir

from .db import fingerprint_rows, norm_entity_text
from .hydration import build_chunk_pointers, hydrate_corpus_chunks, is_markdown_path, pack_doc_titles_blob
from .rabitq import (
    build_beta_nats_lut,
    build_rotator,
    compute_edge_ham_cos,
    compute_shannon_d_eff,
    pack_doc_1bit_codes,
    pack_rabitq_soa,
    transform_vectors,
)
from .types import (
    CJK_RUN_RE,
    TANGENT_RHO_CLIP,
    WORD_TOK_RE,
    CitationGraph,
    DiffusionOperators,
    IndexArtifacts,
)

_SENT_SPLIT_RE = re.compile(r"(?:(?<=[.!?;:\n])\s+|(?<=[。！？；])\s*)")
_CITATION_GRAPH_MAGIC = b"YCG1"
_CITATION_GRAPH_HDR_FMT = "<4s11I"


def build_dc_csr(path_idx: NDArray[np.uint32], nd: int) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Build document-to-chunk CSR arrays from chunk path indices."""
    if nd == 0 or path_idx.size == 0:
        return np.zeros(nd + 1, dtype=np.int32), np.zeros(0, dtype=np.int32)
    counts = np.bincount(path_idx, minlength=nd).astype(np.int32, copy=False)
    indptr = np.empty(nd + 1, dtype=np.int32)
    indptr[0] = 0
    np.cumsum(counts, dtype=np.int32, out=indptr[1:])
    indices = np.argsort(path_idx, kind="stable").astype(np.int32, copy=False)
    return indptr, indices


def lists_to_csr(postings: Sequence[Sequence[int]], n_rows: int) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Convert adjacency posting lists into compact int32 CSR `(indptr, indices)` arrays."""
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


def extract_title_keys(
    title: str,
    *,
    prop_nouns: Collection[str] | None = None,
) -> list[tuple[str, ...]]:
    """Extract normalized word-tuple and CJK keys for matching a document title."""
    t_norm = DASH_RE.sub("-", title) if prop_nouns is not None else title
    t_cf = t_norm.casefold().strip()
    if not t_cf:
        return []
    keys: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()

    def _emit(tok_seq: tuple[str, ...]) -> None:
        if 1 <= len(tok_seq) <= 8 and tok_seq not in seen:
            seen.add(tok_seq)
            keys.append(tok_seq)

    toks = tuple(WORD_TOK_RE.findall(t_cf))
    if toks:
        _emit(toks)
        if "(" in t_cf:
            base = t_cf.split("(", 1)[0].strip()
            base_toks = tuple(WORD_TOK_RE.findall(base))
            if base_toks and (prop_nouns is None or len(base_toks) >= 2 or base_toks[0] in prop_nouns):
                _emit(base_toks)
    for run in CJK_RUN_RE.findall(t_cf):
        if len(run) >= 2:
            _emit((run,))
    return keys


def build_title_index(
    pointers: Sequence[ChunkPointer],
    doc_titles: Mapping[str, str],
    *,
    prop_nouns: Collection[str] | None = None,
) -> dict[str, tuple[int, dict[tuple[str, ...], tuple[int, ...]]]]:
    """Index title key sequences by leading token for longest-span title matching."""
    raw_by_first: dict[str, dict[tuple[str, ...], list[int]]] = {}
    max_len_by_first: dict[str, int] = {}
    for pos, ptr in enumerate(pointers):
        eff_title = doc_titles.get(ptr.rel_posix_path, "")
        for tk in extract_title_keys(eff_title, prop_nouns=prop_nouns):
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


def _resolve_doc_titles_and_chunks(
    pointers: Sequence[ChunkPointer],
    distinct_paths: Sequence[str],
    corpus_root: Path,
) -> tuple[list[str], tuple[str, ...], dict[str, str]]:
    md_paths = [dp for dp in distinct_paths if is_markdown_path(dp)]
    ct, disk_titles = hydrate_corpus_chunks(pointers, corpus_root, extract_titles_for=md_paths)
    doc_titles_tup = tuple(DASH_RE.sub("-", disk_titles.get(dp, "")) for dp in distinct_paths)
    doc_titles_norm = {dp: dt for dp, dt in zip(distinct_paths, doc_titles_tup, strict=True) if dt}
    return ct, doc_titles_tup, doc_titles_norm


def _accum_sentence_cap_stats(raw_t: str, tot_cnt: dict[str, int], cap_cnt: dict[str, int]) -> None:
    for sent in _SENT_SPLIT_RE.split(raw_t):
        for ti, tok in enumerate(WORD_TOK_RE.findall(sent)):
            w = tok.casefold()
            is_cjk = CJK_RUN_RE.fullmatch(tok) is not None
            is_acronym = len(tok) >= 2 and tok.isupper()
            if (len(w) >= 3 or (len(w) >= 2 and CJK_RUN_RE.fullmatch(w) is not None)) and (
                ti > 0 or is_acronym or is_cjk
            ):
                tot_cnt[w] = tot_cnt.get(w, 0) + 1
                if tok[0].isupper() or is_acronym or is_cjk:
                    cap_cnt[w] = cap_cnt.get(w, 0) + 1


def _collect_corpus_vocab_stats(
    ct: Sequence[str],
) -> tuple[frozenset[str], dict[str, int], dict[tuple[str, str], list[int]]]:
    cap_cnt: dict[str, int] = {}
    tot_cnt: dict[str, int] = {}
    w_df: dict[str, int] = {}
    bg_post: dict[tuple[str, str], list[int]] = {}

    for u, raw_chunk in enumerate(ct):
        raw_t = norm_entity_text(raw_chunk)
        _accum_sentence_cap_stats(raw_t, tot_cnt, cap_cnt)
        raw_cf = raw_t.casefold()
        cf_toks = WORD_TOK_RE.findall(raw_cf)
        for w in set(cf_toks):
            w_df[w] = w_df.get(w, 0) + 1
        seen_bg: set[tuple[str, str]] = set()
        for j in range(len(cf_toks) - 1):
            bg = (cf_toks[j], cf_toks[j + 1])
            if bg not in seen_bg:
                seen_bg.add(bg)
                bg_post.setdefault(bg, []).append(u)
        for run in CJK_RUN_RE.findall(raw_cf):
            if len(run) >= 2:
                for bg in {run[k : k + 2] for k in range(len(run) - 1)}:
                    w_df[bg] = w_df.get(bg, 0) + 1
                for k in range(len(run) - 3):
                    pair = (run[k : k + 2], run[k + 2 : k + 4])
                    if pair not in seen_bg:
                        seen_bg.add(pair)
                        bg_post.setdefault(pair, []).append(u)

    prop_nouns = frozenset(w for w, tot in tot_cnt.items() if 2 * cap_cnt.get(w, 0) > tot)
    return prop_nouns, w_df, bg_post


def _match_sent_title_targets(
    sent_cf: str,
    du: int,
    tbfw_clean: dict[str, tuple[int, dict[tuple[str, ...], tuple[int, ...]]]],
    tk_to_docs: dict[tuple[str, ...], tuple[int, ...]],
    targets: set[int],
) -> None:
    ltoks = WORD_TOK_RE.findall(sent_cf)
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
                    if (docs := tk_to_docs.get(cand)) is not None:
                        targets.update(di for di in docs if di != du)
                    hit = span_len
                    break
            if hit > 0:
                i += hit
                continue
        i += 1
    for run in CJK_RUN_RE.findall(sent_cf):
        rl = len(run)
        for j in range(rl):
            for span_len in range(min(8, rl - j), 1, -1):
                if (docs := tk_to_docs.get((run[j : j + span_len],))) is not None:
                    targets.update(di for di in docs if di != du)
                    break


def _build_citation_csr(
    ct: Sequence[str],
    path_idx_arr: NDArray[np.uint32],
    pointers: Sequence[ChunkPointer],
    doc_titles_norm: Mapping[str, str],
    prop_nouns: frozenset[str],
    n: int,
    nd: int,
    db: int,
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.int32], NDArray[np.int32]]:
    tbfw_clean = build_title_index(pointers, doc_titles_norm, prop_nouns=prop_nouns)
    tk_to_docs: dict[tuple[str, ...], tuple[int, ...]] = {}
    for _, sub_tbl in tbfw_clean.values():
        for tk, pos_tup in sub_tbl.items():
            if len(pos_tup) <= db:
                tk_to_docs[tk] = tuple(sorted({int(path_idx_arr[p]) for p in pos_tup}))

    fwd_post: list[list[int]] = [[] for _ in range(n)]
    rev_adj: list[list[int]] = [[] for _ in range(nd)]
    for u, raw_chunk in enumerate(ct):
        du = int(path_idx_arr[u])
        targets: set[int] = set()
        for sent in _SENT_SPLIT_RE.split(norm_entity_text(raw_chunk)):
            _match_sent_title_targets(sent.casefold(), du, tbfw_clean, tk_to_docs, targets)
        if targets:
            sorted_t = sorted(targets)
            fwd_post[u] = sorted_t
            for di in sorted_t:
                rev_adj[di].append(u)

    fi, fx = lists_to_csr(fwd_post, n)
    ri, rx = lists_to_csr(rev_adj, nd)
    return fi, fx, ri, rx


def _build_pmi_csr(
    bg_post: dict[tuple[str, str], list[int]],
    w_df: dict[str, int],
    path_idx_arr: NDArray[np.uint32],
    n: int,
    db: int,
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.int32], NDArray[np.int32], tuple[str, ...]]:
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
    pmi_indptr, pmi_indices = lists_to_csr(pmi_lists, len(pmi_items))
    cpmi_indptr, cpmi_indices = lists_to_csr(c_pmis, n)
    return pmi_indptr, pmi_indices, cpmi_indptr, cpmi_indices, pmi_terms


def build_citation_graph(
    pointers: Sequence[ChunkPointer],
    distinct_paths: Sequence[str],
    path_idx_arr: NDArray[np.uint32],
    corpus_root: Path | str,
) -> CitationGraph:
    """Hydrate corpus chunks once and build CSR title-reference and 1-bit PMI graphs."""
    c_root = require_corpus_dir(corpus_root)
    n = len(pointers)
    nd = len(distinct_paths)
    dc_indptr, dc_indices = build_dc_csr(path_idx_arr, nd)
    ct, doc_titles_tup, doc_titles_norm = _resolve_doc_titles_and_chunks(pointers, distinct_paths, c_root)
    db = max(4, math.ceil(math.sqrt(max(1, n))))
    prop_nouns, w_df, bg_post = _collect_corpus_vocab_stats(ct)
    fi, fx, ri, rx = _build_citation_csr(ct, path_idx_arr, pointers, doc_titles_norm, prop_nouns, n, nd, db)
    pmi_indptr, pmi_indices, cpmi_indptr, cpmi_indices, pmi_terms = _build_pmi_csr(bg_post, w_df, path_idx_arr, n, db)
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


def pack_citation_graph_blob(cg: CitationGraph) -> bytes:
    """Serialize a CitationGraph into a zlib-compressed YCG1 binary payload."""
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


def _decompress_and_parse_cg_header(
    blob: bytes, n_expected: int, nd_expected: int
) -> tuple[bytes, int, int, int, int, int, int, int]:
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
    if n != n_expected or nd != nd_expected:
        raise YawRuntimeError(
            "CONSISTENCY_ERROR",
            f"citation_graph dimension mismatch: got (n={n}, nd={nd}), expected (n={n_expected}, nd={nd_expected}).",
        )
    expected_ints = (n + 1) + n_fx + (nd + 1) + n_rx + (n_pmi + 1) + n_pmi_idx + (n + 1) + n_pmi_idx
    if len(raw) != hdr_sz + expected_ints * 4 + meta_len:
        raise YawRuntimeError("CONSISTENCY_ERROR", "Corrupt citation_graph.graph_blob byte length.")
    return raw, hdr_sz, n, nd, n_fx, n_rx, n_pmi, n_pmi_idx


def _validate_cg_csr_arrays(
    arrays: tuple[NDArray[np.int32], ...], n: int, nd: int, n_fx: int, n_rx: int, n_pmi: int, n_pmi_idx: int
) -> None:
    fi, fx, ri, rx, pmi_indptr, pmi_indices, cpmi_indptr, cpmi_indices = arrays
    indptr_ok = (
        int(fi[0]) == 0
        and int(fi[-1]) == n_fx
        and int(ri[0]) == 0
        and int(ri[-1]) == n_rx
        and int(pmi_indptr[0]) == 0
        and int(pmi_indptr[-1]) == n_pmi_idx
        and int(cpmi_indptr[0]) == 0
        and int(cpmi_indptr[-1]) == n_pmi_idx
        and not bool(np.any(np.diff(fi) < 0))
        and not bool(np.any(np.diff(ri) < 0))
        and not bool(np.any(np.diff(pmi_indptr) < 0))
        and not bool(np.any(np.diff(cpmi_indptr) < 0))
    )
    if not indptr_ok:
        raise YawRuntimeError("CONSISTENCY_ERROR", "Corrupt CSR indptr bounds in citation_graph.graph_blob.")
    if fx.size and (int(fx.min()) < 0 or int(fx.max()) >= nd):
        raise YawRuntimeError("CONSISTENCY_ERROR", "Out-of-range document index in citation_graph.graph_blob.")
    if rx.size and (int(rx.min()) < 0 or int(rx.max()) >= n):
        raise YawRuntimeError("CONSISTENCY_ERROR", "Out-of-range chunk index in citation_graph.graph_blob.")
    if pmi_indices.size and (int(pmi_indices.min()) < 0 or int(pmi_indices.max()) >= n):
        raise YawRuntimeError("CONSISTENCY_ERROR", "Out-of-range chunk index in citation_graph.graph_blob.")
    if cpmi_indices.size and (int(cpmi_indices.min()) < 0 or int(cpmi_indices.max()) >= n_pmi):
        raise YawRuntimeError("CONSISTENCY_ERROR", "Out-of-range PMI term index in citation_graph.graph_blob.")


def _parse_cg_metadata(
    meta_bytes: bytes, n_pmi: int, nd: int
) -> tuple[tuple[str, ...], frozenset[str], tuple[str, ...]]:
    try:
        meta = json.loads(meta_bytes.decode("utf-8"))
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
    return tuple(str(x) for x in meta[0]), frozenset(str(x) for x in meta[1]), tuple(str(x) for x in meta[2])


def unpack_citation_graph_blob(blob: bytes, path_idx: NDArray[np.uint32], nd_expected: int) -> CitationGraph:
    """Decompress and strictly validate a YCG1 CitationGraph binary blob."""
    raw, off, n, nd, n_fx, n_rx, n_pmi, n_pmi_idx = _decompress_and_parse_cg_header(
        blob, int(path_idx.size), nd_expected
    )
    counts = (n + 1, n_fx, nd + 1, n_rx, n_pmi + 1, n_pmi_idx, n + 1, n_pmi_idx)
    decoded: list[NDArray[np.int32]] = []
    for cnt in counts:
        decoded.append(np.frombuffer(raw, dtype="<i4", count=cnt, offset=off).astype(np.int32, copy=True))
        off += cnt * 4
    arrays = tuple(decoded)
    _validate_cg_csr_arrays(arrays, n, nd, n_fx, n_rx, n_pmi, n_pmi_idx)
    pmi_terms, prop_nouns, doc_titles = _parse_cg_metadata(raw[off:], n_pmi, nd)
    dc_indptr, dc_indices = build_dc_csr(path_idx, nd)
    fi, fx, ri, rx, pmi_indptr, pmi_indices, cpmi_indptr, cpmi_indices = arrays
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
        doc_titles=doc_titles,
    )


def compute_index_artifacts(rows: Sequence[Any], corpus_root: Path | str) -> IndexArtifacts:
    """Compute 1-bit SoA RaBitQ codes, Shannon d_eff, and compressed CitationGraph blobs from chunk rows."""
    c_root = require_corpus_dir(corpus_root)
    if not rows:
        empty = hashlib.sha256(b"").hexdigest()
        return IndexArtifacts(0, 0, empty, empty, 0, b"", b"", b"", 0.0, b"", b"")
    n, dim, vec_sha, src_sha = fingerprint_rows(rows)
    raw = np.vstack([np.frombuffer(bytes(r[1]), dtype="<f2").astype(np.float32) for r in rows])
    if not np.all(np.isfinite(raw)):
        raise YawRuntimeError("CONSISTENCY_ERROR", "Non-finite embedding_f16 value in chunks table.")
    mean_vec = raw.mean(axis=0).astype(np.float32) if n >= 2 else np.zeros(dim, dtype=np.float32)
    unit = transform_vectors(raw, mean_vec)
    d_shannon = compute_shannon_d_eff(unit)
    rot = build_rotator(dim)
    codes, ip_norm = pack_rabitq_soa(unit, rot)
    w = (dim + 63) // 64
    _cids, pointers, distinct_paths, path_idx_arr = build_chunk_pointers(rows)
    cg = build_citation_graph(pointers, distinct_paths, path_idx_arr, c_root)
    doc_titles_map = {dp: dt for dp, dt in zip(distinct_paths, cg.doc_titles, strict=True) if dt}
    doc_titles_blob = pack_doc_titles_blob(doc_titles_map)
    graph_blob = pack_citation_graph_blob(cg)
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


def _build_citation_edge_buffers(
    cg: CitationGraph,
    u_all: NDArray[np.int32],
    pidx: NDArray[np.int32],
    el: NDArray[np.bool_],
    el_idx: NDArray[np.int32],
    codes: NDArray[np.uint8],
    dim: int,
    d_eff: float,
) -> tuple[
    sp.csr_array | None,
    sp.csr_array | None,
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.int32] | None,
    NDArray[np.int32] | None,
    NDArray[np.float32] | None,
    NDArray[np.float32] | None,
    NDArray[np.int32] | None,
    NDArray[np.int32] | None,
    NDArray[np.int32] | None,
    NDArray[np.int32] | None,
    NDArray[np.float64] | None,
    NDArray[np.float64] | None,
    float,
    float,
    int,
]:
    n = int(pidx.size)
    nd = len(cg.doc_titles)
    has_fwd = np.zeros(n, dtype=np.float32)
    has_rev = np.zeros(n, dtype=np.float32)
    if n == 0 or nd == 0 or cg.fx.size == 0 or el_idx.size == 0:
        return None, None, has_fwd, has_rev, None, None, None, None, None, None, None, None, None, None, 0.0, 0.0, 0

    doc_sz = np.bincount(pidx[el_idx], minlength=nd).astype(np.float32)
    f_rows_all = np.repeat(u_all, np.diff(cg.fi)).astype(np.int32, copy=False)
    f_cols_all = cg.fx.astype(np.int32, copy=False)
    f_keep = el[f_rows_all] & (doc_sz[f_cols_all] > 0.0)
    f_rows = f_rows_all[f_keep]
    f_cols = f_cols_all[f_keep]
    if f_rows.size == 0:
        return None, None, has_fwd, has_rev, None, None, None, None, None, None, None, None, None, None, 0.0, 0.0, 0

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
    doc_codes = pack_doc_1bit_codes(codes, dim, el_idx, pidx, nd)
    c_ud_raw = compute_edge_ham_cos(codes, doc_codes, f_rows, f_cols, dim)
    edge_c_ud = np.clip(c_ud_raw, -TANGENT_RHO_CLIP, TANGENT_RHO_CLIP).astype(np.float32)
    edge_inv_s_ud = (1.0 / np.sqrt(np.maximum(1e-6, 1.0 - edge_c_ud.astype(np.float64) ** 2))).astype(np.float32)
    edge_fwd_indptr = np.zeros(n + 1, dtype=np.int32)
    np.cumsum(out_deg, out=edge_fwd_indptr[1:])
    edge_t_order = np.lexsort((f_rows, f_cols)).astype(np.int32)
    in_deg_doc = np.bincount(f_cols, minlength=nd)
    edge_t_indptr = np.zeros(nd + 1, dtype=np.int32)
    np.cumsum(in_deg_doc, out=edge_t_indptr[1:])
    edge_t_indices = f_rows[edge_t_order].astype(np.int32, copy=False)
    edge_beta_d_tan = max(2.0, float(d_eff) - 1.0)
    edge_beta_lut, edge_beta_lut_diff, edge_beta_scale = build_beta_nats_lut(edge_beta_d_tan)
    d_id_t = sp.csr_array(
        (np.ones(el_idx.size, dtype=np.float32), (pidx[el_idx], el_idx)),
        shape=(nd, n),
        dtype=np.float32,
    )
    has_rev = (el & (in_deg_doc[pidx] > 0)).astype(np.float32)
    return (
        d_own,
        d_id_t,
        has_fwd,
        has_rev,
        f_rows,
        f_cols,
        edge_c_ud,
        edge_inv_s_ud,
        edge_fwd_indptr,
        edge_t_order,
        edge_t_indptr,
        edge_t_indices,
        edge_beta_lut,
        edge_beta_lut_diff,
        edge_beta_scale,
        edge_beta_d_tan,
        nd,
    )


def _build_pmi_buffers(
    cg: CitationGraph,
    u_all: NDArray[np.int32],
    el: NDArray[np.bool_],
    n: int,
    n_elig: int,
) -> tuple[
    sp.csr_array | None,
    sp.csr_array | None,
    NDArray[np.float32] | None,
    NDArray[np.float32] | None,
    NDArray[np.float32],
]:
    has_pmi = np.zeros(n, dtype=np.float32)
    n_pmi = len(cg.pmi_terms)
    if n == 0 or n_pmi == 0 or cg.cpmi_indices.size == 0 or n_elig == 0:
        return None, None, None, None, has_pmi
    p_rows_all = np.repeat(u_all, np.diff(cg.cpmi_indptr)).astype(np.int32, copy=False)
    p_cols_all = cg.cpmi_indices.astype(np.int32, copy=False)
    p_keep = el[p_rows_all]
    p_rows = p_rows_all[p_keep]
    p_cols = p_cols_all[p_keep]
    if p_rows.size == 0:
        return None, None, None, None, has_pmi
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
    p2c = sp.csr_array((b_pmi @ sp.diags_array(np.where(pc_sum > 0.0, 1.0 / np.maximum(pc_sum, 1e-12), 0.0))).tocsr())
    self_pmi = c2p.multiply(p2c).sum(axis=1).astype(np.float32)
    non_self_pmi = np.maximum(1.0 - self_pmi, 0.0).astype(np.float32)
    has_pmi = (el & (non_self_pmi > 1e-6)).astype(np.float32)
    pmi_scale = np.where(has_pmi > 0.0, 1.0 / np.maximum(non_self_pmi, 1e-6), 0.0).astype(np.float32)
    c2p_t = sp.csr_array(c2p.T.tocsr())
    return p2c, c2p_t, self_pmi, pmi_scale, has_pmi


def build_diffusion_operators(
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
    u_all = np.arange(n, dtype=np.int32)
    pidx = path_idx.astype(np.int32, copy=False)
    el = np.ones(n, dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool).reshape(n)
    el_idx = np.flatnonzero(el).astype(np.int32, copy=False)
    (
        d_own,
        d_id_t,
        has_fwd,
        has_rev,
        edge_f_rows,
        edge_f_cols,
        edge_c_ud,
        edge_inv_s_ud,
        edge_fwd_indptr,
        edge_t_order,
        edge_t_indptr,
        edge_t_indices,
        edge_beta_lut,
        edge_beta_lut_diff,
        edge_beta_scale,
        edge_beta_d_tan,
        edge_nd,
    ) = _build_citation_edge_buffers(cg, u_all, pidx, el, el_idx, codes, dim, d_eff)
    p2c, c2p_t, self_pmi, pmi_scale, has_pmi = _build_pmi_buffers(cg, u_all, el, n, int(el_idx.size))
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
