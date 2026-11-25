"""Tangent-steered cyclic citation diffusion, conditional Noisy-OR, and 1-bit DPP search."""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Collection, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from yaw.core import YawRuntimeError

from .db import norm_entity_text, snapshot_tx
from .graph import build_diffusion_operators
from .hydration import dedup_ranked_positions, hydrate_one
from .lexical import eligible_mask, extract_multilingual_anchors, extract_title_references, lexical_search
from .rabitq import (
    build_beta_nats_lut,
    eval_signed_beta_nats_lut,
    grouped_softmax,
    parametric_null_nats_vec,
    rabitq_dot_lut,
    rotate_vectors,
    sigmoid,
    transform_vectors,
    unpack_1bit_signed,
)
from .slip import sanitize_slip_label
from .types import (
    TANGENT_COS_CLIP,
    TANGENT_RHO_CLIP,
    CitationGraph,
    DiffusionOperators,
    RetrievalIndex,
    SearchResult,
)


def fetch_point_vectors(
    conn: sqlite3.Connection,
    idx: RetrievalIndex,
    positions: Sequence[int],
    cache: dict[int, NDArray[np.float32]],
) -> None:
    """Load and normalize exact float16 chunk embeddings for `positions` into `cache`."""
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
    unit = transform_vectors(mat, idx.mean_vec)
    for k, (p, _) in enumerate(raw_list):
        cache[p] = unit[k]


def prepare_query_vector(
    idx: RetrievalIndex,
    q_raw: NDArray[np.float32],
    *,
    exclude_vecs: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]:
    """Validate, center, L2-normalize, and optionally orthogonally deflate query embeddings."""
    q = np.asarray(q_raw, dtype=np.float32).ravel()
    if q.shape[0] != idx.dim:
        raise YawRuntimeError("INVALID_INPUT", f"Query embedding dim {q.shape[0]} != index dim {idx.dim}")
    q_norm = float(np.linalg.norm(q))
    if q_norm <= 1e-12 or not np.all(np.isfinite(q)):
        raise YawRuntimeError("INVALID_INPUT", "Query embedding must be finite and non-zero.")
    q_unit = (q / q_norm).astype(np.float32, copy=False)
    q_vec = transform_vectors(q_unit[None, :], idx.mean_vec)[0]
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


def compute_all_query_cosines(idx: RetrievalIndex, q_hat: NDArray[np.float32]) -> NDArray[np.float32]:
    """Compute unbiased 1-bit SoA RaBitQ cosines against all indexed chunks."""
    assert idx.rotator is not None
    q_rot = rotate_vectors(idx.rotator, q_hat)
    return (rabitq_dot_lut(idx.codes, q_rot, idx.dim) / np.maximum(idx.ip_norm, 1e-6)).astype(np.float32)


def apply_tangent_steer_ops(
    ops: DiffusionOperators,
    all_q_cos: NDArray[np.float32],
    d_eff: float,
    n_size: int,
    t_path: float,
) -> DiffusionOperators:
    """Rewire forward and reverse CSR transition weights using query-tangent partial correlations."""
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
    c_qd = np.clip((ops.d_own.T @ all_q_cos).astype(np.float64), -TANGENT_COS_CLIP, TANGENT_COS_CLIP)
    c_qu = np.clip(all_q_cos.astype(np.float64), -TANGENT_COS_CLIP, TANGENT_COS_CLIP)
    inv_sq_u = 1.0 / np.sqrt(np.maximum(1e-6, 1.0 - c_qu * c_qu))
    inv_sq_d = 1.0 / np.sqrt(np.maximum(1e-6, 1.0 - c_qd * c_qd))
    cq_u = c_qu[f_rows]
    cq_d = c_qd[f_cols]
    rho_fwd = np.clip((cq_d - cq_u * c_ud) * (inv_sq_u[f_rows] * inv_s_ud), -TANGENT_RHO_CLIP, TANGENT_RHO_CLIP)
    rho_rev = np.clip((cq_u - cq_d * c_ud) * (inv_sq_d[f_cols] * inv_s_ud), -TANGENT_RHO_CLIP, TANGENT_RHO_CLIP)
    d_tan = max(2.0, d_eff - 1.0)
    if (
        ops.edge_beta_lut is not None
        and ops.edge_beta_lut_diff is not None
        and abs(ops.edge_beta_d_tan - d_tan) <= 1e-9
    ):
        beta_lut, beta_lut_diff, beta_scale = ops.edge_beta_lut, ops.edge_beta_lut_diff, ops.edge_beta_scale
    else:
        beta_lut, beta_lut_diff, beta_scale = build_beta_nats_lut(d_tan)
    inv_t_path = 1.0 / t_path
    logits_fwd = eval_signed_beta_nats_lut(rho_fwd, beta_lut, beta_lut_diff, beta_scale, inv_t_path)
    logits_rev = eval_signed_beta_nats_lut(rho_rev, beta_lut, beta_lut_diff, beta_scale, inv_t_path)
    w_fwd = grouped_softmax(f_rows, logits_fwd, n_size)
    w_rev = grouped_softmax(f_cols, logits_rev, ops.edge_nd)
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


def apply_citation_step(
    ops: DiffusionOperators,
    x: NDArray[np.float32],
    el_f: NDArray[np.float32],
    size: int,
) -> NDArray[np.float32]:
    """Execute one uniform-channel citation + non-self-loop PMI transition step."""
    xm = (x * el_f) * ops.inv_m
    y = np.zeros(size, dtype=np.float32)
    if ops.d_own is not None:
        assert ops.m_fwd_t is not None and ops.r_rev is not None and ops.d_id_t is not None
        y = y + (ops.d_own @ (ops.m_fwd_t @ xm)) + (ops.r_rev @ (ops.d_id_t @ xm))
    if ops.p2c is not None and ops.c2p_t is not None and ops.self_pmi is not None and ops.pmi_scale is not None:
        xp = xm * ops.pmi_scale
        y = y + np.maximum(np.float32(0.0), (ops.p2c @ (ops.c2p_t @ xp)) - ops.self_pmi * xp)
    return (y * el_f).astype(np.float32, copy=False)


def diffuse_citation(
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
        step_vec = apply_citation_step(ops, a_cur, el_f, size)
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


def rerank_dpp_1bit(
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
    x_bar = unpack_1bit_signed(idx.codes, pool, idx.dim).astype(np.float64)
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


def _require_positive_int(val: Any, label: str) -> int:
    if isinstance(val, bool) or not isinstance(val, int) or val <= 0:
        raise YawRuntimeError("INVALID_INPUT", f"{label} must be a positive integer, got {val!r}")
    return val


def _validate_search_inputs(
    idx: RetrievalIndex,
    query_embedding: NDArray[np.float32],
    conn: sqlite3.Connection | None,
    top_k: int,
    pool_k: int | None,
    path_glob: str | None,
    masked_rel_paths: Collection[str] | None,
    exclude_chunk_ids: Sequence[int] | None,
) -> tuple[NDArray[np.bool_], list[int], int] | SearchResult:
    _require_positive_int(top_k, "top_k")
    if pool_k is not None:
        _require_positive_int(pool_k, "pool_k")
    if conn is None:
        raise YawRuntimeError("CONSISTENCY_ERROR", "search_index requires an active SQLite connection.")
    if idx.size == 0:
        return SearchResult((), -math.inf, 0.0, 0.0, idx.d_eff, 0)
    if idx.citation_graph is None or idx.diffusion_ops is None or idx.rotator is None:
        raise YawRuntimeError("CONSISTENCY_ERROR", "RetrievalIndex is missing CitationGraph or DiffusionOperators.")

    eligible = eligible_mask(idx, path_glob=path_glob, masked_rel_paths=masked_rel_paths)
    excl_pos: list[int] = []
    if exclude_chunk_ids is not None:
        for cid in exclude_chunk_ids:
            if (p := idx.id_to_pos.get(int(cid))) is not None:
                eligible[p] = False
                excl_pos.append(p)

    n_eff = int(np.sum(eligible))
    if n_eff == 0:
        q_chk = np.asarray(query_embedding, dtype=np.float32).ravel()
        if q_chk.shape[0] != idx.dim or float(np.linalg.norm(q_chk)) <= 1e-12 or not np.all(np.isfinite(q_chk)):
            raise YawRuntimeError("INVALID_INPUT", "Query embedding must match dim, be finite, and non-zero.")
        with snapshot_tx(conn, require_fts=True):
            pass
        return SearchResult((), -math.inf, 0.0, 0.0, idx.d_eff, 0)
    return eligible, excl_pos, n_eff


def _apply_title_ref_noisy_or(
    idx: RetrievalIndex,
    ops: DiffusionOperators,
    post: NDArray[np.float32],
    q_trefs: list[tuple[str, tuple[int, ...], float, int]],
    sem_n: NDArray[np.float64],
    l_vec: NDArray[np.float32],
    el_f: NDArray[np.float32],
    eligible: NDArray[np.bool_],
    n_eff: int,
    t_path: float,
) -> NDArray[np.float32]:
    if len(q_trefs) < 2:
        return post
    nrel = 1.0 - np.clip(post.astype(np.float64), 0.0, 1.0)
    log_n1 = math.log(float(n_eff) + 1.0)
    for _, pt, _, _ in q_trefs:
        ept = [p for p in pt if eligible[p]]
        if not ept:
            continue
        wg = sigmoid(max(float(sem_n[p]) for p in ept) - log_n1)
        sc_g = np.array([sem_n[p] / t_path for p in ept], dtype=np.float64)
        ex_g = np.exp(sc_g - float(sc_g.max()))
        ex_g /= max(float(ex_g.sum()), 1e-12)
        ag = np.zeros(idx.size, dtype=np.float32)
        for k_p, p in enumerate(ept):
            ag[p] = float(ex_g[k_p])
        pg = diffuse_citation(ops, ag, l_vec, el_f, n_eff)
        nrel *= 1.0 - wg * np.clip(pg.astype(np.float64), 0.0, 1.0)
    return (1.0 - nrel).astype(np.float32)


def _score_query_evidence(
    idx: RetrievalIndex,
    query_embedding: NDArray[np.float32],
    query_text: str,
    conn: sqlite3.Connection,
    eligible: NDArray[np.bool_],
    excl_pos: list[int],
    n_eff: int,
    t_path: float,
    ops: DiffusionOperators,
) -> tuple[NDArray[np.float32], NDArray[np.float32], float, float]:
    all_elig = n_eff == idx.size
    d_bridge = max(4, math.ceil(math.sqrt(n_eff)))
    d_query = max(8, n_eff // 2)
    vec_cache: dict[int, NDArray[np.float32]] = {}

    with snapshot_tx(conn, require_fts=True):
        if excl_pos:
            fetch_point_vectors(conn, idx, excl_pos, vec_cache)
            excl_mat = np.vstack([vec_cache[p] for p in excl_pos])
        else:
            excl_mat = None
        q_hat = prepare_query_vector(idx, query_embedding, exclude_vecs=excl_mat)
        all_q_cos = compute_all_query_cosines(idx, q_hat)
        elig_list = None if all_elig else eligible.tolist()
        q_eff = norm_entity_text(query_text)
        q_anchors = extract_multilingual_anchors(conn, idx, eligible, q_eff, max_df_filter=d_query)
        q_trefs = extract_title_references(
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

    sem_n = parametric_null_nats_vec(np.maximum(0.0, np.where(eligible, all_q_cos.astype(np.float64), 0.0)), idx.d_eff)
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

    steered_ops = apply_tangent_steer_ops(
        ops, np.where(eligible, all_q_cos, np.float32(0.0)), idx.d_eff, idx.size, t_path
    )
    el_f = eligible.astype(np.float32)
    mx_sem = float((sem_n[eligible] / t_path).max())
    l_vec = np.where(eligible, np.exp((sem_n / t_path) - mx_sem), 0.0).astype(np.float32)
    post = diffuse_citation(steered_ops, a0, l_vec, el_f, n_eff)
    post = _apply_title_ref_noisy_or(idx, steered_ops, post, q_trefs, sem_n, l_vec, el_f, eligible, n_eff, t_path)
    return post, all_q_cos, ev_max, ex_sum


def _order_candidates_distinct_first(
    idx: RetrievalIndex,
    post: NDArray[np.float32],
    all_q_cos: NDArray[np.float32],
    pos_list: Sequence[int],
) -> tuple[list[int], int]:
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


def _select_ranked_positions(
    idx: RetrievalIndex,
    post: NDArray[np.float32],
    all_q_cos: NDArray[np.float32],
    eligible: NDArray[np.bool_],
    top_k: int,
    n_eff: int,
) -> tuple[tuple[tuple[int, float], ...], int]:
    elig_pos = np.flatnonzero(eligible)
    n_elig_int = int(elig_pos.size)
    m_law = math.ceil(2.0 * math.log(float(n_eff) + 1.0))
    m_target = max(top_k, m_law)
    k_pool = min(n_elig_int, max(m_target * 8, 256))
    post_e = post[elig_pos]
    sub_idx = (
        np.argpartition(-post_e, k_pool - 1)[:k_pool] if k_pool < n_elig_int else np.arange(n_elig_int, dtype=np.int64)
    )
    order, n_distinct_found = _order_candidates_distinct_first(
        idx, post, all_q_cos, [int(elig_pos[int(i)]) for i in sub_idx]
    )
    p_cut = float(post_e[sub_idx].min()) if k_pool < n_elig_int else -math.inf
    need_n = min(m_target, n_elig_int)
    if k_pool < n_elig_int and need_n > 0 and (n_distinct_found < need_n or float(post[order[need_n - 1]]) <= p_cut):
        order, n_distinct_found = _order_candidates_distinct_first(idx, post, all_q_cos, [int(i) for i in elig_pos])

    if len(order) > 1:
        m_pool = min(n_distinct_found, m_law)
        pool = order[:m_pool]
        reranked = pool if m_pool <= 1 else rerank_dpp_1bit(idx, post, pool, top_k=top_k, n_eff=n_eff)
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
    return tuple(ranked_list), len(selected_pos)


def _collect_cite_links(
    idx: RetrievalIndex,
    cg: CitationGraph,
    cand_pos: Sequence[int],
) -> list[tuple[Literal["cite", "pmi"], int, int, str]]:
    doc_best_pos: dict[int, int] = {}
    for p in cand_pos:
        doc_best_pos.setdefault(int(idx.path_idx[p]), p)
    seen_cites: set[tuple[int, int]] = set()
    cite_links: list[tuple[Literal["cite", "pmi"], int, int, str]] = []
    for u in cand_pos:
        src_doc = int(idx.path_idx[u])
        src_cid = int(idx.chunk_ids[u])
        for di in cg.fx[int(cg.fi[u]) : int(cg.fi[u + 1])]:
            v = doc_best_pos.get(int(di))
            if int(di) == src_doc or v is None or v == u:
                continue
            dst_cid = int(idx.chunk_ids[v])
            if (src_cid, dst_cid) not in seen_cites:
                seen_cites.add((src_cid, dst_cid))
                cite_links.append(("cite", src_cid, dst_cid, ""))
    return cite_links


def _find_rarest_pmi_link(
    idx: RetrievalIndex,
    cg: CitationGraph,
    cand_pos: Sequence[int],
) -> tuple[Literal["cite", "pmi"], int, int, str] | None:
    if not cg.pmi_terms or cg.cpmi_indices.size == 0:
        return None
    terms_by_rank = [
        {int(e) for e in cg.cpmi_indices[int(cg.cpmi_indptr[u]) : int(cg.cpmi_indptr[u + 1])]} for u in cand_pos
    ]
    best_key: tuple[int, str, int, int] | None = None
    best_link: tuple[Literal["cite", "pmi"], int, int, str] | None = None
    for rank_u, u in enumerate(cand_pos):
        terms_u = terms_by_rank[rank_u]
        if not terms_u:
            continue
        doc_u = int(idx.path_idx[u])
        for rank_v in range(rank_u + 1, len(cand_pos)):
            v = cand_pos[rank_v]
            if int(idx.path_idx[v]) == doc_u:
                continue
            for eid in terms_u & terms_by_rank[rank_v]:
                clean_term = sanitize_slip_label(cg.pmi_terms[eid])
                if not clean_term:
                    continue
                cand_key = (int(cg.pmi_indptr[eid + 1] - cg.pmi_indptr[eid]), clean_term, rank_u, rank_v)
                if best_key is None or cand_key < best_key:
                    best_key = cand_key
                    best_link = ("pmi", int(idx.chunk_ids[u]), int(idx.chunk_ids[v]), clean_term)
    return best_link


def _extract_candidate_links(
    idx: RetrievalIndex,
    top_k_pos: Sequence[int],
    next_pos: Sequence[int],
) -> tuple[tuple[Literal["cite", "pmi"], int, int, str], ...]:
    cand_pos = [*top_k_pos, *next_pos]
    if len(cand_pos) < 2 or idx.citation_graph is None:
        return ()
    links = _collect_cite_links(idx, idx.citation_graph, cand_pos)
    if (pmi_link := _find_rarest_pmi_link(idx, idx.citation_graph, cand_pos)) is not None:
        links.append(pmi_link)
    return tuple(links)


def search_index(
    idx: RetrievalIndex,
    query_embedding: NDArray[np.float32],
    *,
    query_text: str = "",
    conn: sqlite3.Connection | None = None,
    top_k: int = 10,
    pool_k: int | None = None,
    path_glob: str | None = None,
    masked_rel_paths: Collection[str] | None = None,
    exclude_chunk_ids: Sequence[int] | None = None,
) -> SearchResult:
    """Run frozen M_FINAL_1B retrieval: 1-bit RaBitQ, unicode61 lexical/title anchors, tangent diffusion, and DPP."""
    validated = _validate_search_inputs(
        idx, query_embedding, conn, top_k, pool_k, path_glob, masked_rel_paths, exclude_chunk_ids
    )
    if isinstance(validated, SearchResult):
        return validated
    assert conn is not None and idx.citation_graph is not None and idx.diffusion_ops is not None
    eligible, excl_pos, n_eff = validated
    all_elig = n_eff == idx.size
    ops = (
        idx.diffusion_ops
        if all_elig
        else build_diffusion_operators(
            idx.citation_graph,
            idx.path_idx,
            codes=idx.codes,
            dim=idx.dim,
            d_eff=idx.d_eff,
            eligible=eligible,
        )
    )
    t_path = math.sqrt(max(1.0, idx.d_eff) / max(1.0, math.log(float(n_eff) + 1.0)))
    post, all_q_cos, ev_max, ex_sum = _score_query_evidence(
        idx, query_embedding, query_text, conn, eligible, excl_pos, n_eff, t_path, ops
    )
    dpp_k = pool_k if pool_k is not None else top_k
    ranked_out, n_candidates = _select_ranked_positions(idx, post, all_q_cos, eligible, dpp_k, n_eff)
    deduped = dedup_ranked_positions(idx, ranked_out, limit=top_k + 3, include_island=True)
    top_entries = deduped[:top_k]
    next_entries = deduped[top_k : top_k + 3]
    top_k_pos = [p for p, _, _ in top_entries]
    next_pos = [p for p, _, _ in next_entries]
    coverage = (
        min(1.0, max(0.0, float(post[top_k_pos].sum() / max(float(post[eligible].sum()), 1e-12)))) if top_k_pos else 0.0
    )
    next_candidates = tuple((cid, sc) for _, cid, sc in next_entries)
    links = _extract_candidate_links(idx, top_k_pos, next_pos)
    top1_cos = float(all_q_cos[eligible].max())
    log_bf = float(ev_max + math.log(ex_sum) - math.log(float(n_eff)))
    return SearchResult(
        ranked=ranked_out,
        log_bf=log_bf,
        max_chunk_score=ranked_out[0][1] if ranked_out else 0.0,
        top1_cosine=top1_cos,
        d_eff=idx.d_eff,
        n_candidates=n_candidates,
        coverage=coverage,
        next_candidates=next_candidates,
        links=links,
    )


def _validate_neighbor_inputs(
    idx: RetrievalIndex,
    chunk_id: int,
    conn: sqlite3.Connection | None,
    top_k: int,
    path_glob: str | None,
    masked_rel_paths: Collection[str] | None,
) -> tuple[int, NDArray[np.bool_], NDArray[np.intp]] | None:
    if isinstance(chunk_id, bool) or not isinstance(chunk_id, int):
        raise YawRuntimeError("INVALID_INPUT", f"chunk_id must be an integer, got {chunk_id!r}")
    if (pos := idx.id_to_pos.get(chunk_id)) is None:
        raise YawRuntimeError("INVALID_INPUT", f"Unknown chunk_id={chunk_id}.")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise YawRuntimeError("INVALID_INPUT", f"top_k must be a positive integer, got {top_k!r}")
    if conn is None:
        raise YawRuntimeError("CONSISTENCY_ERROR", "expand_citation_neighbors requires an active SQLite connection.")
    if idx.size <= 1:
        return None
    if idx.citation_graph is None or idx.diffusion_ops is None or idx.rotator is None:
        raise YawRuntimeError("CONSISTENCY_ERROR", "RetrievalIndex is missing CitationGraph or DiffusionOperators.")
    eligible = eligible_mask(idx, path_glob=path_glob, masked_rel_paths=masked_rel_paths)
    eligible[pos] = False
    elig_pos = np.flatnonzero(eligible)
    if elig_pos.size == 0:
        return None
    return pos, eligible, elig_pos


def _add_ast_island_nats(
    idx: RetrievalIndex,
    pos: int,
    n_elig_total: int,
    score_by_pos: dict[int, float],
) -> None:
    ptr = idx.pointers[pos]
    if (ptr.ast_island_start, ptr.ast_island_end) == (1, 1):
        return
    island = (ptr.rel_posix_path, ptr.ast_island_start, ptr.ast_island_end)
    same_island_pos = [
        p
        for p in score_by_pos
        if (idx.pointers[p].rel_posix_path, idx.pointers[p].ast_island_start, idx.pointers[p].ast_island_end) == island
    ]
    if same_island_pos:
        island_nats = math.log(max(1.0, float(n_elig_total) / float(len(same_island_pos) + 1)))
        for p in same_island_pos:
            score_by_pos[p] += island_nats


def expand_citation_neighbors(
    idx: RetrievalIndex,
    chunk_id: int,
    *,
    corpus_root: Path | str,
    wiki_root: Path | str | None = None,
    conn: sqlite3.Connection | None = None,
    top_k: int = 6,
    path_glob: str | None = None,
    masked_rel_paths: Collection[str] | None = None,
) -> list[tuple[int, float]]:
    """Expand isotropic 1-bit cosines, lexical IDF, tangent-steered citation diffusion, and AST-island nats."""
    validated = _validate_neighbor_inputs(idx, chunk_id, conn, top_k, path_glob, masked_rel_paths)
    if validated is None:
        return []
    assert conn is not None and idx.citation_graph is not None and idx.diffusion_ops is not None
    pos, eligible, elig_pos = validated

    c_root = corpus_root if isinstance(corpus_root, Path) and corpus_root.is_absolute() else Path(corpus_root).resolve()
    w_root = (
        None
        if wiki_root is None
        else (wiki_root if isinstance(wiki_root, Path) and wiki_root.is_absolute() else Path(wiki_root).resolve())
    )
    src_hp = hydrate_one(
        c_root,
        idx.pointers[pos],
        0.0,
        None,
        False,
        False,
        strict=True,
        path_cache=idx.resolved_path_cache,
        wiki_root=w_root,
    )
    vec_cache: dict[int, NDArray[np.float32]] = {}
    n_elig_total = int(elig_pos.size) + 1
    lex_cap = min(int(elig_pos.size), max(top_k, math.ceil(2.0 * math.log(float(n_elig_total) + 1.0))))
    with snapshot_tx(conn, require_fts=True):
        fetch_point_vectors(conn, idx, [pos], vec_cache)
        all_u_cos = compute_all_query_cosines(idx, vec_cache[pos])
        cos_scores = all_u_cos[elig_pos].astype(np.float64)
        lex_hits = lexical_search(
            idx, src_hp.text, conn, top_k=lex_cap, path_glob=path_glob, masked_rel_paths=masked_rel_paths
        )

    sem_nats = parametric_null_nats_vec(np.maximum(0.0, cos_scores), idx.d_eff)
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
        else build_diffusion_operators(
            idx.citation_graph,
            idx.path_idx,
            codes=idx.codes,
            dim=idx.dim,
            d_eff=idx.d_eff,
            eligible=elig_with_src,
        )
    )
    ops = apply_tangent_steer_ops(ops, np.where(elig_with_src, all_u_cos, np.float32(0.0)), idx.d_eff, idx.size, t_path)
    sem_full = np.zeros(idx.size, dtype=np.float64)
    sem_full[elig_pos] = sem_nats
    sem_full[pos] = float(sem_nats.max()) if elig_pos.size > 0 else 0.0
    mx_sem = float((sem_full[elig_with_src] / t_path).max())
    l_vec = np.where(elig_with_src, np.exp((sem_full / t_path) - mx_sem), 0.0).astype(np.float32)
    seed = np.zeros(idx.size, dtype=np.float32)
    seed[pos] = 1.0
    p_diff = diffuse_citation(ops, seed, l_vec, elig_with_src.astype(np.float32), n_elig_total)
    for ep in elig_pos:
        p_int = int(ep)
        if (pd := float(p_diff[p_int])) > 0.0:
            score_by_pos[p_int] += math.log1p(float(n_elig_total) * pd)

    _add_ast_island_nats(idx, pos, n_elig_total, score_by_pos)
    ordered = sorted(
        score_by_pos.items(),
        key=lambda kv: (-kv[1], -cos_by_pos[kv[0]], idx.chunk_ids[kv[0]]),
    )[:top_k]
    return [(idx.chunk_ids[p], sc) for p, sc in ordered]
