"""Sparse entity-state HMM relevance inference with separate passage emissions (H1)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from yaw.core import RelationTraversalPolicy


def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    """Normalize dense retrieval features into [0, 1]."""
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return arr
    span = float(arr.max() - arr.min())
    return (arr - arr.min()) / span if span else np.ones_like(arr)


@dataclass(frozen=True, slots=True)
class HMMOperators:
    """Entity transition potentials, passage emissions and independent source likelihoods."""

    forward: sp.csr_matrix
    backward: sp.csr_matrix
    emissions: sp.csr_matrix
    likelihoods: np.ndarray
    forward_dangling: np.ndarray
    backward_dangling: np.ndarray


@dataclass(frozen=True, slots=True)
class CSRGraph:
    node_ids: tuple[str, ...]
    node_to_idx: Mapping[str, int]
    chunk_indices: np.ndarray
    chunk_ids: tuple[str, ...]
    transition_t: sp.csr_matrix
    dangling_mask: np.ndarray
    hmm: HMMOperators


def _noisy_or(shape: tuple[int, int], rows: list[int], cols: list[int], weights: list[float]) -> sp.csr_matrix:
    if not weights:
        return sp.csr_matrix(shape, dtype=np.float64)
    with np.errstate(divide="ignore"):
        failures = np.log1p(-np.asarray(weights, dtype=np.float64))
    matrix = sp.coo_matrix((failures, (rows, cols)), shape=shape).tocsr()
    matrix.data = -np.expm1(matrix.data)
    matrix.eliminate_zeros()
    return matrix


def _normalize_potential(matrix: sp.csr_matrix, beta: float) -> tuple[sp.csr_matrix, np.ndarray]:
    incoming = np.asarray(matrix.sum(axis=1)).ravel()
    outgoing = np.asarray(matrix.sum(axis=0)).ravel()
    row_scale = np.zeros_like(incoming)
    col_scale = np.zeros_like(outgoing)
    np.power(incoming, -beta, out=row_scale, where=incoming > 0)
    np.power(outgoing, beta - 1.0, out=col_scale, where=outgoing > 0)
    return (sp.diags(row_scale) @ matrix @ sp.diags(col_scale)).tocsr(), outgoing == 0


def build_transition_matrix(
    chunk_ids: Sequence[str],
    entity_ids: Sequence[str],
    edges: Sequence[tuple[str, str, str, float] | tuple[str, str, float]],
    *,
    beta: float = 0.5,
    policy: RelationTraversalPolicy | None = None,
    directed: bool = False,
    node_priors: Mapping[str, float | None] | None = None,
) -> CSRGraph:
    """Build positive entity potentials and passage emissions; the full matrix is an inspection view."""
    if not np.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError("beta must be finite and between 0 and 1")
    chunks = tuple(dict.fromkeys(chunk_ids))
    chunk_lookup = {key: i for i, key in enumerate(chunks)}
    entities = tuple(key for key in dict.fromkeys(entity_ids) if key not in chunk_lookup)
    entity_lookup = {key: i for i, key in enumerate(entities)}
    nodes = chunks + entities
    n_c, n_e = len(chunks), len(entities)
    relation_policy = policy or RelationTraversalPolicy()
    if any(not np.isfinite(value) or not 0 <= value <= 1 for value in relation_policy.to_dict().values()):
        raise ValueError("Relation policy weights must be finite and between 0 and 1")
    multipliers = {
        "fact": (relation_policy.fact_forward, relation_policy.fact_reverse),
        "skos:broader": (relation_policy.skos_broader_up, relation_policy.skos_broader_down),
        "skos:related": (relation_policy.skos_related, relation_policy.skos_related),
    }
    emissions: dict[tuple[int, int], float] = {}
    contributions: dict[str, tuple[list[int], list[int], list[float]]] = {kind: ([], [], []) for kind in multipliers}
    for edge in edges:
        source, target = edge[:2]
        weight = float(edge[-1])
        if not np.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError("Graph evidence weights must be finite and between 0 and 1")
        if source == target or weight == 0:
            continue
        kind = str(edge[2]) if len(edge) == 4 else "fact"
        if kind == "owl:differentFrom":
            continue
        if source in chunk_lookup and target in entity_lookup:
            if not directed:
                key = (chunk_lookup[source], entity_lookup[target])
                emissions[key] = max(emissions.get(key, 0.0), weight)
        elif target in chunk_lookup and source in entity_lookup:
            key = (chunk_lookup[target], entity_lookup[source])
            emissions[key] = max(emissions.get(key, 0.0), weight)
        elif source in entity_lookup and target in entity_lookup:
            if kind not in contributions:
                raise ValueError(f"Unsupported entity relation kind: {kind!r}")
            rows, cols, weights = contributions[kind]
            rows.append(entity_lookup[target])
            cols.append(entity_lookup[source])
            weights.append(weight)
    emission_matrix = sp.csr_matrix((n_c, n_e), dtype=np.float64)
    if emissions and relation_policy.passage:
        coordinates = tuple(emissions)
        emission_matrix = sp.coo_matrix(
            (
                [emissions[key] * relation_policy.passage for key in coordinates],
                ([key[0] for key in coordinates], [key[1] for key in coordinates]),
            ),
            shape=(n_c, n_e),
        ).tocsr()
    entity_rows: list[int] = []
    entity_cols: list[int] = []
    entity_weights: list[float] = []
    for kind, (rows, cols, weights) in contributions.items():
        observation = _noisy_or((n_e, n_e), rows, cols, weights)
        if kind == "skos:related" and not directed:
            # Opposing rows are copies of one symmetric observation. Pool
            # same-direction evidence first, then keep the stronger copy once.
            observation = sp.triu(observation.maximum(observation.T), k=1, format="csr")
        observation = observation.tocoo()
        forward_weight, reverse_weight = multipliers[kind]
        if forward_weight:
            entity_rows.extend(observation.row.tolist())
            entity_cols.extend(observation.col.tolist())
            entity_weights.extend((observation.data * forward_weight).tolist())
        if reverse_weight and not directed:
            entity_rows.extend(observation.col.tolist())
            entity_cols.extend(observation.row.tolist())
            entity_weights.extend((observation.data * reverse_weight).tolist())
    adjacency = _noisy_or((n_e, n_e), entity_rows, entity_cols, entity_weights)
    forward, forward_dangling = _normalize_potential(adjacency, beta)
    backward, backward_dangling = _normalize_potential(adjacency.T.tocsr(), beta)
    likelihoods = np.ones(n_e, dtype=np.float64)
    for name, value in (node_priors or {}).items():
        if value is not None:
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Entity confidence must be finite and between 0 and 1")
            if name in entity_lookup:
                likelihoods[entity_lookup[name]] = value
    full = sp.bmat(
        [
            [sp.csr_matrix((n_c, n_c)), emission_matrix],
            [sp.csr_matrix((n_e, n_c)) if directed else emission_matrix.T, forward],
        ],
        format="csr",
    )
    dangling = np.asarray(full.sum(axis=0)).ravel() == 0
    return CSRGraph(
        nodes,
        {key: i for i, key in enumerate(nodes)},
        np.arange(n_c),
        chunks,
        full,
        dangling,
        HMMOperators(forward, backward, emission_matrix, likelihoods, forward_dangling, backward_dangling),
    )


def _filter(
    potential: sp.csr_matrix,
    dangling: np.ndarray,
    seed: np.ndarray,
    likelihoods: np.ndarray,
    damping: float,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    state = seed * likelihoods
    total = float(state.sum())
    if total == 0:
        return state
    state /= total
    for _ in range(max_iter):
        next_state = likelihoods * (
            damping * (potential @ state + float(state[dangling].sum()) * seed) + (1 - damping) * seed
        )
        total = float(next_state.sum())
        if total == 0:
            return next_state
        next_state /= total
        if float(np.abs(next_state - state).sum()) <= tol:
            return next_state
        state = next_state
    return state


def _emit(emissions: sp.csr_matrix, state: np.ndarray) -> np.ndarray:
    result = np.zeros(emissions.shape[0], dtype=np.float64)
    nonempty = np.flatnonzero(np.diff(emissions.indptr))
    if nonempty.size:
        failure = 1 - emissions.data * state[emissions.indices]
        result[nonempty] = 1 - np.multiply.reduceat(failure, emissions.indptr[nonempty])
    return result


def run_hmm(
    graph: CSRGraph,
    reset_weights: Mapping[str, float] | np.ndarray,
    *,
    damping: float = 0.5,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """Infer entity relevance and emit passage relevance; the two layers do not share one mass."""
    if not np.isfinite(damping) or not 0 <= damping <= 1:
        raise ValueError("damping must be finite and between 0 and 1")
    if type(max_iter) is not int or max_iter < 0 or not np.isfinite(tol) or tol < 0:
        raise ValueError("max_iter and tol must be nonnegative finite bounds")
    if isinstance(reset_weights, Mapping):
        seed = np.zeros(len(graph.node_ids), dtype=np.float64)
        for name, weight in reset_weights.items():
            if not np.isfinite(weight) or weight < 0:
                raise ValueError("Seed evidence must be finite and nonnegative")
            if name in graph.node_to_idx:
                seed[graph.node_to_idx[name]] += weight
    else:
        seed = np.array(reset_weights, dtype=np.float64, copy=True).ravel()
        if seed.size != len(graph.node_ids):
            raise ValueError(f"Reset vector size {seed.size} does not match graph nodes {len(graph.node_ids)}")
        if not np.all(np.isfinite(seed)) or np.any(seed < 0):
            raise ValueError("Seed evidence must be finite and nonnegative")
    total = float(seed.sum())
    if total == 0:
        return seed
    seed /= total
    n_c = len(graph.chunk_ids)
    operators = graph.hmm
    entity_seed = seed[n_c:] + operators.emissions.T @ seed[:n_c]
    total = float(entity_seed.sum())
    if total == 0:
        return seed
    entity_seed /= total
    forward = _filter(
        operators.forward, operators.forward_dangling, entity_seed, operators.likelihoods, damping, max_iter, tol
    )
    if np.count_nonzero(entity_seed) > 1:
        backward = _filter(
            operators.backward, operators.backward_dangling, entity_seed, operators.likelihoods, damping, max_iter, tol
        )
        posterior = forward * backward
        mass = float(posterior.sum())
        if mass > 0:
            posterior /= mass
    else:
        posterior = forward
    passage = 1 - (1 - _emit(operators.emissions, posterior)) * (1 - seed[:n_c])
    return np.concatenate((passage, posterior))


def rank_passages_by_hmm(graph: CSRGraph, scores: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
    """Rank actual emitted passage relevance with stable graph-order ties."""
    if scores.size != len(graph.node_ids):
        raise ValueError("HMM score size does not match graph nodes")
    if top_k <= 0:
        return []
    order = np.argsort(-scores[: len(graph.chunk_ids)], kind="stable")
    return [(graph.chunk_ids[i], float(scores[i])) for i in order if scores[i] > 0][:top_k]


__all__ = [
    "CSRGraph",
    "HMMOperators",
    "build_transition_matrix",
    "min_max_normalize",
    "rank_passages_by_hmm",
    "run_hmm",
]
