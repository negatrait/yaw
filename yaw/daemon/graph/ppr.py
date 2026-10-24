"""Sparse CSR matrix Personalized PageRank (PPR) power iteration engine."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.sparse as sp

GraphActivation = Literal[
    "linear",
    "scaled_tanh",
    "hard_tanh",
    "shifted_tanh",
    "threshold_relu",
]

DEFAULT_ACTIVATION_PARAMS: dict[str, float] = {
    "scaled_tanh": 5.0,
    "hard_tanh": 0.1,
    "shifted_tanh": 0.001,
    "threshold_relu": 1e-4,
}


def apply_graph_activation(
    x: np.ndarray,
    activation: GraphActivation | str = "linear",
    param: float | None = None,
) -> np.ndarray:
    """Apply 1-Lipschitz non-linear activation pointwise (arXiv:2609.32929v1)."""
    if activation == "linear":
        return x
    if activation == "scaled_tanh":
        c = float(param) if param is not None and param > 0.0 else DEFAULT_ACTIVATION_PARAMS["scaled_tanh"]
        return np.tanh(c * x) / c
    if activation == "hard_tanh":
        c = float(param) if param is not None and param > 0.0 else DEFAULT_ACTIVATION_PARAMS["hard_tanh"]
        return np.clip(x, 0.0, c)
    if activation == "shifted_tanh":
        c = float(param) if param is not None and param >= 0.0 else DEFAULT_ACTIVATION_PARAMS["shifted_tanh"]
        return np.maximum(0.0, np.tanh(np.maximum(0.0, x) - c))
    if activation == "threshold_relu":
        theta = float(param) if param is not None and param >= 0.0 else DEFAULT_ACTIVATION_PARAMS["threshold_relu"]
        return np.maximum(0.0, x - theta)
    raise ValueError(f"Unknown graph activation: {activation}")


@dataclass(frozen=True, slots=True)
class CSRGraph:
    """Precomputed column-stochastic CSR graph operator for sub-millisecond PPR power iteration."""

    node_ids: tuple[str, ...]
    node_to_idx: Mapping[str, int]
    chunk_indices: np.ndarray
    chunk_ids: tuple[str, ...]
    transition_t: sp.csr_matrix
    dangling_mask: np.ndarray


def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize a float array into [0, 1] range matching upstream HippoRAG."""
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return arr
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    if max_val == min_val:
        return np.ones_like(arr, dtype=np.float64)
    return (arr - min_val) / (max_val - min_val)


def build_transition_matrix(
    chunk_ids: Sequence[str],
    entity_ids: Sequence[str],
    edges: Sequence[tuple[str, str, float] | tuple[str, str, str, float]],
    *,
    beta: float = 0.5,
) -> CSRGraph:
    """Construct a CSR transition operator W(beta) = D^-beta A D^-(1-beta) (arXiv:2609.32929v1)."""
    ordered_chunks = list(dict.fromkeys(chunk_ids))
    chunk_set = set(ordered_chunks)
    ordered_entities = [e for e in dict.fromkeys(entity_ids) if e not in chunk_set]

    all_nodes = tuple(ordered_chunks + ordered_entities)
    node_to_idx = {nid: i for i, nid in enumerate(all_nodes)}
    n_total = len(all_nodes)

    chunk_indices = np.arange(len(ordered_chunks), dtype=np.int32)
    tuple_chunk_ids = tuple(ordered_chunks)

    if n_total == 0:
        empty_csr = sp.csr_matrix((0, 0), dtype=np.float64)
        return CSRGraph(
            node_ids=(),
            node_to_idx={},
            chunk_indices=np.array([], dtype=np.int32),
            chunk_ids=(),
            transition_t=empty_csr,
            dangling_mask=np.array([], dtype=bool),
        )

    u_list: list[int] = []
    v_list: list[int] = []
    w_list: list[float] = []
    for edge in edges:
        src = edge[0]
        dst = edge[1]
        weight = float(edge[2]) if len(edge) == 3 else float(edge[3])
        if weight <= 0.0:
            continue
        u = node_to_idx.get(src)
        v = node_to_idx.get(dst)
        if u is None or v is None or u == v:
            continue
        u_list.append(u)
        v_list.append(v)
        w_list.append(weight)

    if u_list:
        u_arr = np.asarray(u_list, dtype=np.int32)
        v_arr = np.asarray(v_list, dtype=np.int32)
        w_arr = np.asarray(w_list, dtype=np.float64)

        rows = np.concatenate([u_arr, v_arr])
        cols = np.concatenate([v_arr, u_arr])
        data = np.concatenate([w_arr, w_arr])

        # Pack (row, col) into 64-bit integer keys and keep max weight per directed pair
        keys = (rows.astype(np.int64) << 32) | cols.astype(np.int64)
        order = np.lexsort((-data, keys))
        sorted_keys = keys[order]
        first_mask = np.empty(len(sorted_keys), dtype=bool)
        first_mask[0] = True
        first_mask[1:] = sorted_keys[1:] != sorted_keys[:-1]
        keep = order[first_mask]

        adj = sp.csr_matrix(
            (data[keep], (rows[keep], cols[keep])),
            shape=(n_total, n_total),
            dtype=np.float64,
        )
    else:
        adj = sp.csr_matrix((n_total, n_total), dtype=np.float64)

    degrees = np.asarray(adj.sum(axis=1)).ravel()
    dangling_mask = degrees == 0.0

    if adj.nnz > 0:
        if beta == 0.0:
            inv_deg = np.zeros(n_total, dtype=np.float64)
            inv_deg[~dangling_mask] = 1.0 / degrees[~dangling_mask]
            # Standard random-walk P^T = (D^-1 A)^T = A D^-1 -> scale columns in-place
            adj.data *= inv_deg[adj.indices]
        elif beta == 0.5:
            inv_sqrt = np.zeros(n_total, dtype=np.float64)
            inv_sqrt[~dangling_mask] = 1.0 / np.sqrt(degrees[~dangling_mask])
            row_indices = np.repeat(np.arange(n_total, dtype=np.int32), np.diff(adj.indptr))
            # Symmetric GCN normalization: W = D^-0.5 A D^-0.5
            adj.data *= inv_sqrt[row_indices] * inv_sqrt[adj.indices]
        elif beta == 1.0:
            inv_deg = np.zeros(n_total, dtype=np.float64)
            inv_deg[~dangling_mask] = 1.0 / degrees[~dangling_mask]
            row_indices = np.repeat(np.arange(n_total, dtype=np.int32), np.diff(adj.indptr))
            # Row-stochastic normalization: W = D^-1 A
            adj.data *= inv_deg[row_indices]
        else:
            b = float(beta)
            r_scale = np.zeros(n_total, dtype=np.float64)
            c_scale = np.zeros(n_total, dtype=np.float64)
            r_scale[~dangling_mask] = degrees[~dangling_mask] ** (-b)
            c_scale[~dangling_mask] = degrees[~dangling_mask] ** (-(1.0 - b))
            row_indices = np.repeat(np.arange(n_total, dtype=np.int32), np.diff(adj.indptr))
            # Generalized degree normalization: W = D^-beta A D^-(1-beta)
            adj.data *= r_scale[row_indices] * c_scale[adj.indices]

    return CSRGraph(
        node_ids=all_nodes,
        node_to_idx=node_to_idx,
        chunk_indices=chunk_indices,
        chunk_ids=tuple_chunk_ids,
        transition_t=adj,
        dangling_mask=dangling_mask,
    )


def run_ppr(
    graph: CSRGraph,
    reset_weights: Mapping[str, float] | np.ndarray,
    *,
    damping: float = 0.5,
    max_iter: int = 100,
    tol: float = 1e-6,
    activation: GraphActivation | str = "scaled_tanh",
    activation_param: float | None = 5.0,
    renormalize: bool = False,
    dual_weight: float | None = 0.5,
) -> np.ndarray:
    """Run Personalized PageRank or non-linear graph propagation (arXiv:2609.32929v1)."""
    n_nodes = len(graph.node_ids)
    if n_nodes == 0:
        return np.array([], dtype=np.float64)

    if isinstance(reset_weights, Mapping):
        p0 = np.zeros(n_nodes, dtype=np.float64)
        for nid, w in reset_weights.items():
            idx = graph.node_to_idx.get(nid)
            if idx is not None and w > 0.0:
                p0[idx] += float(w)
    else:
        p0 = np.asarray(reset_weights, dtype=np.float64).copy()
        if p0.size != n_nodes:
            raise ValueError(f"Reset vector size {p0.size} does not match graph nodes {n_nodes}")

    total_w = float(np.sum(p0))
    if total_w <= 0.0:
        p0 = np.ones(n_nodes, dtype=np.float64) / n_nodes
    else:
        p0 /= total_w

    p = p0.copy()
    alpha = float(damping)
    teleport = (1.0 - alpha) * p0
    p_raw = p.copy()

    for _ in range(max_iter):
        dangling_mass = float(np.sum(p[graph.dangling_mask]))
        p_raw = alpha * (graph.transition_t.dot(p) + dangling_mass * p0) + teleport
        if activation != "linear":
            p_next = apply_graph_activation(p_raw, activation, activation_param)
            if renormalize:
                s = float(np.sum(p_next))
                if s > 0.0:
                    p_next = p_next / s
        else:
            p_next = p_raw

        err = float(np.sum(np.abs(p_next - p)))
        p = p_next
        if err < tol:
            break

    if dual_weight is not None and activation != "linear":
        lam = float(np.clip(dual_weight, 0.0, 1.0))
        y_norm = min_max_normalize(p_raw)
        z_norm = min_max_normalize(p)
        return (1.0 - lam) * y_norm + lam * z_norm

    return p


def rank_passages_by_ppr(
    graph: CSRGraph,
    ppr_scores: np.ndarray,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Extract and sort passage node PPR probabilities in descending score order."""
    if len(graph.chunk_indices) == 0 or top_k <= 0 or ppr_scores.size == 0:
        return []
    passage_scores = ppr_scores[graph.chunk_indices]
    k = min(top_k, len(graph.chunk_indices))
    # Negative argsort for descending order
    order = np.argsort(-passage_scores)[:k]
    return [(graph.chunk_ids[i], float(passage_scores[i])) for i in order]


__all__ = [
    "CSRGraph",
    "DEFAULT_ACTIVATION_PARAMS",
    "GraphActivation",
    "apply_graph_activation",
    "build_transition_matrix",
    "min_max_normalize",
    "rank_passages_by_ppr",
    "run_ppr",
]
