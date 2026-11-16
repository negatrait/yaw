"""Pure-chunk Sub-State Manifold HMM retrieval with ABTT, CSLS, spaCy bridging, and Ego-kNN."""

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from graphlib import CycleError, TopologicalSorter
from typing import Any, Self

import numpy as np
import spacy
from scipy.linalg import eigh
from scipy.special import softmax

from yaw.core.types import ChunkPointer

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'_-]*")
_CAP_SPAN_RE = re.compile(r"(?:[A-Z][A-Za-z0-9_-]*(?:\s+[A-Z0-9][A-Za-z0-9_-]*)*)")
_BRIDGE_CONN = frozenset({"of", "the", "and", "-", "de", "la", "van", "von"})
_AUX_LEMMAS = frozenset({"be", "have", "do", "sr", "jr", "ii", "iii", "iv"})
type _LinksByBranch = list[list[tuple[int, int, int, float, str]]]


@dataclass(frozen=True, slots=True)
class ManifoldHyperparams:
    """Primary physical hyperparameters governing manifold retrieval."""

    peclet: float = 0.65
    alpha_sprt: float = 0.05
    redundancy_curvature: float = 0.75
    coalition_coupling: float = 0.45
    external_trust: float = 0.35

    def __post_init__(self) -> None:
        bounds = {"peclet": (0.0, 1.0, False, True), "alpha_sprt": (0.0, 0.5, False, False)}
        for name in self.__dataclass_fields__:
            raw = getattr(self, name)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ValueError(f"{name} must be an int or float")
            if not math.isfinite(val := float(raw)):
                raise ValueError(f"{name} must be finite")
            lo, hi, inc_lo, inc_hi = bounds.get(name, (0.0, 2.0 if name == "redundancy_curvature" else 1.0, True, True))
            if not ((val >= lo if inc_lo else val > lo) and (val <= hi if inc_hi else val < hi)):
                raise ValueError(f"{name} out of domain [{lo}, {hi}]: {val}")
            object.__setattr__(self, name, val)

    @classmethod
    def from_overrides(cls, overrides: Mapping[str, float] | Sequence[tuple[str, float]] | None = None) -> Self:
        """Build validated hyperparameters from a mapping or key-value sequence."""
        cleaned = dict(overrides.items() if isinstance(overrides, Mapping) else (overrides or ()))
        if bad := [k for k in cleaned if k not in cls.__dataclass_fields__]:
            raise ValueError(f"Unknown ManifoldHyperparams field: {bad[0]!r}")
        return replace(cls(), **cleaned)


@dataclass(frozen=True, slots=True)
class ManifoldDerivedTelemetry:
    """Execution budgets derived deterministically from ManifoldHyperparams."""

    num_hops: int
    max_silent_hops: int
    route_min_gate: float
    route_max_chars: int
    default_top_k: int
    max_particles: int
    hop0_anchor_protect: float
    hop_post_scale: float
    external_seed_weight: float
    join_bp_weight: float


def derive_manifold_telemetry(params: ManifoldHyperparams | None = None) -> ManifoldDerivedTelemetry:
    """Derive deterministic execution budgets from the 5 physical hyperparameters."""
    hp = params or ManifoldHyperparams()
    gate, protect = max(0.05, min(0.50, 0.15 * (1.0 - hp.alpha_sprt))), hp.peclet + 0.35 * hp.external_trust
    et, cc = hp.external_trust, hp.coalition_coupling
    return ManifoldDerivedTelemetry(8, 2, gate, 600, 10, 3, protect, 1.0 - hp.peclet, et, cc)


@dataclass(frozen=True, slots=True)
class EvidenceHop:
    """One evidence step in an acyclic multi-hop reasoning route."""

    hop: int
    chunk_id: str
    title: str
    snippet: str
    score: float
    bridge: str = ""
    branch: int = 0


@dataclass(frozen=True, slots=True)
class SubStateResult:
    """Ranked chunk posteriors, per-hop trajectories, and acyclic evidence routes."""

    ranked_chunks: list[tuple[str, float]]
    chunk_posteriors: np.ndarray
    hop_posteriors: tuple[np.ndarray, ...]
    hop_weights: tuple[float, ...]
    route: tuple[EvidenceHop, ...]
    num_particles: int = 1
    routes: tuple[tuple[EvidenceHop, ...], ...] = ()


@dataclass(frozen=True, slots=True)
class SubStateManifold:
    """Precomputed isotropic spectral and lexical index over pure document chunks."""

    chunk_ids: tuple[str, ...]
    chunk_titles: tuple[str, ...]
    chunk_title_norms: tuple[str, ...]
    chunk_full_texts: tuple[str, ...]
    u_chunk_raw: np.ndarray
    u_chunk_iso: np.ndarray
    q_rogue: np.ndarray
    abtt_removed_dims: int
    rmt_lambda_plus: float
    csls_chunk: np.ndarray
    vocab_index: dict[str, int]
    postings: tuple[np.ndarray, ...]
    token_df: np.ndarray
    token_idf2: np.ndarray
    chunk_tok_sets: tuple[frozenset[int], ...]
    ast_siblings: tuple[np.ndarray, ...]
    chunk_entities: tuple[tuple[list[str], list[str]], ...]
    params: ManifoldHyperparams = ManifoldHyperparams()


@lru_cache(maxsize=1)
def _load_spacy_pipeline() -> Any:
    return spacy.load("en_core_web_sm", exclude=["ner"])


def _l2_norm(arr: np.ndarray, *, row_wise: bool = False) -> np.ndarray:
    data = np.asarray(arr, dtype=np.float32)
    if not data.size:
        return data
    norm = np.linalg.norm(data, axis=1, keepdims=True) if row_wise else max(float(np.linalg.norm(data)), 1e-12)
    return data / np.maximum(norm, 1e-12) if row_wise else data.reshape(-1) / norm


def _extract_terms(text: str, stops: set[str]) -> list[str]:
    words = [w for m in _WORD_RE.finditer(text) if len(w := m.group(0).lower()) > 1 and w not in stops]
    return words + [f"{words[k]}__{words[k + 1]}" for k in range(len(words) - 1)]


def _robust_z(arr: np.ndarray) -> np.ndarray:
    med = float(np.median(arr))
    return (arr - med) / max(float(np.median(np.abs(arr - med))) * 1.4826, 1e-3)


def _extract_spacy_entities(doc: Any, stops: set[str], *, include_attr: bool = False) -> list[str]:
    spans, toks, idx = [], list(doc), 0
    while idx < len(toks):
        end = idx + 1
        if toks[idx].pos_ == "PROPN":
            while end < len(toks) and (
                toks[end].pos_ == "PROPN"
                or (end + 1 < len(toks) and toks[end + 1].pos_ == "PROPN" and toks[end].text.lower() in _BRIDGE_CONN)
                or (end > idx and toks[end - 1].text == "-" and toks[end].is_alpha)
                or (end + 1 < len(toks) and toks[end].text == "-" and toks[end + 1].is_alpha)
            ):
                end += 1
            spans.append(doc[idx:end].text)
        idx = end
    if doc.has_annotation("DEP"):
        spans.extend(nc.text for nc in doc.noun_chunks if nc.root.pos_ == "PROPN")
        if include_attr:
            for sent in list(doc.sents)[:2]:
                for attr_tok in (t for t in sent if t.pos_ == "NOUN" and not t.is_stop and t.dep_ == "attr"):
                    spans += [
                        attr_tok.text,
                        *(c.text for c in attr_tok.conjuncts if c.pos_ == "NOUN" and not c.is_stop),
                    ]
    dedup: dict[str, str] = {}
    for raw in spans or _CAP_SPAN_RE.findall(doc.text):
        if len(clean := raw.strip(" \t\n\r\"'()[]{}:;,.?!")) >= 2 and _extract_terms(clean, stops):
            dedup.setdefault(re.sub(r"\s+", " ", clean).lower(), clean)
    return list(dedup.values())


def _is_comp_q(doc: Any, ent_spans: Sequence[str]) -> bool:
    non_appos = [sp for sp in ent_spans if not any(t.text in sp and t.dep_ == "appos" for t in doc)]
    parts = [s for sp in (non_appos or ent_spans[:1]) for p in re.split(r"\band\b", sp) if len(s := p.strip()) >= 2]
    if len(parts) >= 2:
        return True
    ent_low, preds = " ".join(ent_spans).lower(), set()
    for tok in doc:
        if not (tok.is_alpha and len(tok.text) > 1 and not tok.is_stop and tok.pos_ in ("NOUN", "VERB", "ADJ")):
            continue
        if (lem := tok.lemma_.lower()) not in _AUX_LEMMAS and tok.text.lower() not in ent_low:
            preds.add(lem)
    return len(preds) >= 2


def _build_ast_siblings(ptrs: Sequence[ChunkPointer]) -> tuple[np.ndarray, ...]:
    by_file: dict[str, list[tuple[int, ChunkPointer]]] = {}
    for idx, ptr in enumerate(ptrs):
        if ptr.ast_island_start > 0 and ptr.ast_island_end >= ptr.ast_island_start:
            by_file.setdefault(ptr.rel_posix_path, []).append((idx, ptr))
    siblings = [np.empty((0,), np.int32)] * len(ptrs)
    for items in by_file.values():
        items.sort(key=lambda pair: (pair[1].start_line, pair[1].end_line, pair[1].chunk_id))
        for idx, ptr in items:
            ps, pe = ptr.ast_island_start, ptr.ast_island_end
            if ovl := [oi for oi, o in items if oi != idx and max(ps, o.ast_island_start) <= min(pe, o.ast_island_end)]:
                siblings[idx] = np.asarray(ovl, np.int32)
    return tuple(siblings)


# fmt: off
def build_substate_manifold(chunk_pointers: Sequence[ChunkPointer], chunk_texts: Mapping[str, str] | Sequence[str],
    chunk_vectors: np.ndarray, *, params: ManifoldHyperparams | None = None) -> SubStateManifold:
# fmt: on
    """Build a pure-chunk SubStateManifold from chunk pointers, texts, and embeddings."""
    hp, n_c, vecs = params or ManifoldHyperparams(), len(chunk_pointers), np.asarray(chunk_vectors, np.float32)
    if n_c == 0:
        dim = vecs.shape[1] if vecs.ndim == 2 else 0
        em, ei = np.zeros((0, dim), np.float32), np.zeros((0,), np.int32)
        ef, er = np.zeros((0,), np.float32), np.zeros((dim, 0), np.float32)
        return SubStateManifold((), (), (), (), em, em, er, 0, 1.0, ef, {}, (), ei, ef, (), (), (), hp)
    raw = [chunk_texts.get(p.chunk_id, "") for p in chunk_pointers] if isinstance(chunk_texts, Mapping) else chunk_texts
    if vecs.ndim != 2 or vecs.shape[0] != n_c or len(raw) != n_c:
        raise ValueError(f"Mismatched chunk count {n_c}: vecs={vecs.shape}, texts={len(raw)}")
    texts, c_ids = tuple(str(t) for t in raw), tuple(str(p.chunk_id) for p in chunk_pointers)
    titles = tuple(p.title for p in chunk_pointers)
    t_norms = tuple(re.sub(r"\W+", " ", t).strip().lower() for t in titles)
    u_raw, dim = _l2_norm(vecs, row_wise=True), vecs.shape[1]
    u_iso, q_rogue, abtt_d, lam = u_raw.copy(), np.zeros((dim, 0), np.float32), 0, 1.0
    if n_c >= 2 and dim >= 2:
        gram = (np.asarray(u_raw, np.float64).T @ np.asarray(u_raw, np.float64)) / float(n_c)
        lam = max(((1.0 + math.sqrt(dim / n_c)) ** 2) / dim, 0.35 if dim < 64 else 0.04)
        evals, evecs = eigh(0.5 * (gram + gram.T), subset_by_index=(dim - min(8, n_c - 1, dim - 1), dim - 1))
        if (keep := np.flatnonzero(evals > lam)[::-1]).size > 0:
            q_rogue, abtt_d = np.asarray(evecs[:, keep], np.float32), int(keep.size)
            u_iso = _l2_norm(u_raw - (u_raw @ q_rogue) @ q_rogue.T, row_wise=True)
    csls = np.zeros((n_c,), np.float32)
    if n_c >= 2:
        k_eff, u_t = min(8, n_c - 1), u_iso.T
        for s in range(0, n_c, 1024):
            sims = u_iso[s : (e := min(n_c, s + 1024))] @ u_t
            sims[np.arange(e - s), np.arange(s, e)] = -2.0
            csls[s:e] = np.maximum(np.partition(sims, -k_eff, axis=1)[:, -k_eff:], 0.0).mean(axis=1)
    stops, post_map = set((nlp := _load_spacy_pipeline()).Defaults.stop_words), {}
    doc_terms = [list(dict.fromkeys(_extract_terms(f"{t}\n{b}", stops))) for t, b in zip(titles, texts, strict=True)]
    for ci, terms in enumerate(doc_terms):
        for term in terms:
            post_map.setdefault(term, []).append(ci)
    v_idx = {term: idx for idx, term in enumerate(sorted(post_map))}
    posts = tuple(np.asarray(post_map[t], np.int32) for t in v_idx)
    t_df = np.asarray([len(p) for p in posts], np.int32)
    idf2 = np.square(np.log((n_c + 1.0) / (t_df.astype(np.float64) + 0.5))).astype(np.float32)
    idf2[t_df > (0.25 * n_c if n_c >= 16 else n_c)] = 0.0
    t_sets = tuple(frozenset(v_idx[t] for t in tms if idf2[v_idx[t]] > 0.0) for tms in doc_terms)
    d_all = list(nlp.pipe((t[:1600] for t in texts), batch_size=64))
    it_s, e_fn = iter(nlp.pipe((t[:320] for t in texts if len(t) > 320), batch_size=64)), _extract_spacy_entities
    c_ents = tuple(
        (e_fn(d, stops)[:16], e_fn(next(it_s) if len(t) > 320 else d, stops, include_attr=True)[:8])
        for t, d in zip(texts, d_all, strict=True)
    )
    # fmt: off
    return SubStateManifold(c_ids, titles, t_norms, texts, u_raw, u_iso, q_rogue, abtt_d,
        float(lam), csls, v_idx, posts, t_df, idf2, t_sets, _build_ast_siblings(chunk_pointers), c_ents, hp)
    # fmt: on


def _scale_hints(n_c: int, scores: Mapping[str, float] | None, id_map: Mapping[str, int]) -> np.ndarray:
    out = np.zeros((n_c,), np.float32)
    pairs = [(id_map[c], float(v)) for c, v in (scores or {}).items() if c in id_map and math.isfinite(v) and v > 0.0]
    if not pairs:
        return out
    idxs, vals = [p[0] for p in pairs], np.asarray([p[1] for p in pairs], dtype=np.float64)
    if vals.size == 1 or float(vals.max() - vals.min()) <= 1e-9:
        out[idxs] = 2.0 if vals.size == 1 else 0.5
        return out
    med = float(np.median(vals))
    mad = max(float(np.median(np.abs(vals - med))) * 1.4826, 1e-4)
    out[idxs] = (1.0 + np.clip((vals - med) / mad, 0.0, 3.0)).astype(np.float32)
    return out


def _score_terms(man: SubStateManifold, terms: Sequence[str], bg_w: float = 1.5) -> tuple[np.ndarray, set[int]]:
    lex, tot_w, active = np.zeros((len(man.chunk_ids),), np.float32), 0.0, set()
    for term in dict.fromkeys(terms):
        if (ti := man.vocab_index.get(term)) is not None and man.token_idf2[ti] > 0.0:
            weight = float(man.token_idf2[ti]) * (bg_w if "__" in term else 1.0)
            tot_w += weight
            lex[man.postings[ti]] += weight
            active.add(ti)
    return (lex / tot_w if tot_w > 0.0 else lex), active


def _build_routes(
    man: SubStateManifold, links_br: _LinksByBranch, seed_info: list[tuple[int, str, float]], gate: float
) -> tuple[tuple[EvidenceHop, ...], tuple[tuple[EvidenceHop, ...], ...]]:
    edges = [(0, -1, s, max(sc, 0.1), lbl, bi) for bi, (s, lbl, sc) in enumerate(seed_info)]
    edges += [(h, p, d, s, b, i) for i, ls in enumerate(links_br) for h, p, d, s, b in ls if s >= gate and p != d]
    if len(edges) == len(seed_info):
        return (), ()
    routes_br: list[list[EvidenceHop]] = [[] for _ in seed_info]
    seen_br, graph = [set() for _ in seed_info], {}
    prev_br: list[int | None] = [None] * len(seed_info)
    for hop, p_ci, d_ci, score, bridge, bi in sorted(edges, key=lambda x: (x[0], -x[3])):
        if d_ci in seen_br[bi]:
            continue
        deps = set(graph.get(d_ci, ())) | {x for x in (p_ci, prev_br[bi]) if x is not None and x >= 0 and x != d_ci}
        cand_g = {**graph, d_ci: deps}
        try:
            list(TopologicalSorter(cand_g).static_order())
        except CycleError:
            continue
        graph, prev_br[bi] = cand_g, d_ci
        seen_br[bi].add(d_ci)
        snip = man.chunk_full_texts[d_ci].strip().replace("\n", " ")[:180]
        routes_br[bi].append(EvidenceHop(hop, man.chunk_ids[d_ci], man.chunk_titles[d_ci], snip, score, bridge, bi))
    active = tuple(tuple(br) for br in routes_br if len(br) >= 2)
    flat = sorted((h for b in active for h in b), key=lambda h: (-h.hop, h.score, -h.branch))
    return tuple({h.chunk_id: h for h in flat}.values()), active


# fmt: off
def run_substate_hmm(manifold: SubStateManifold, query: str, query_vec: np.ndarray | Sequence[float], *,
    top_k: int | None = None, fts_chunk_scores: Mapping[str, float] | None = None, num_hops: int | None = None,
    vec_chunk_scores: Mapping[str, float] | None = None, params: ManifoldHyperparams | None = None) -> SubStateResult:
# fmt: on
    """Execute pure-chunk ABTT+CSLS Hop-0 and Dynamic Max-Node Ego-kNN multi-hop inference."""
    hp, n_c, tele = (p := params or manifold.params), len(manifold.chunk_ids), derive_manifold_telemetry(p)
    eff_k = tele.default_top_k if top_k is None else max(1, int(top_k))
    eff_hops = tele.num_hops if num_hops is None else max(1, int(num_hops))
    if n_c == 0:
        return SubStateResult([], np.zeros((0,), np.float32), (), (), (), 0, ())
    q_raw, dim = _l2_norm(np.asarray(query_vec, np.float32)), manifold.u_chunk_raw.shape[1]
    if q_raw.size != dim:
        raise ValueError(f"query_vec dim {q_raw.size} != manifold dim {dim}")
    q_rogue, u_iso, csls = manifold.q_rogue, manifold.u_chunk_iso, manifold.csls_chunk
    q_iso = _l2_norm(q_raw - q_rogue @ (q_rogue.T @ q_raw)) if q_rogue.shape[1] > 0 else q_raw
    blend, csls_w = 1.0 - hp.peclet, 0.5 * (1.0 - hp.peclet)
    z_d = _robust_z(blend * (manifold.u_chunk_raw @ q_raw) + hp.peclet * (u_iso @ q_iso - csls_w * csls))
    stops = set((nlp := _load_spacy_pipeline()).Defaults.stop_words)
    c_ids, titles, t_norms = manifold.chunk_ids, manifold.chunk_titles, manifold.chunk_title_norms
    v_idx, idf2, t_sets = manifold.vocab_index, manifold.token_idf2, manifold.chunk_tok_sets
    lex_c, q_tids = _score_terms(manifold, _extract_terms(query, stops), 1.5)
    id_map = {cid: idx for idx, cid in enumerate(c_ids)}
    z_fts, z_vec = _scale_hints(n_c, fts_chunk_scores, id_map), _scale_hints(n_c, vec_chunk_scores, id_map)
    s0 = z_d + 2.2 * (1.0 - 0.5 * hp.peclet) * lex_c + hp.external_trust * (2.2 * (z_fts / 4.0) + 1.2 * (z_vec / 4.0))
    h0_post = softmax(np.asarray(s0, np.float64)).astype(np.float32)
    q_ent_spans = _extract_spacy_entities(q_doc := nlp(query), stops)
    comp_q, z_gate = _is_comp_q(q_doc, q_ent_spans), np.maximum(0.1, (z_d + 2.0) / 4.0)
    ent_facets: list[tuple[str, np.ndarray]] = []
    for span in q_ent_spans if len(q_ent_spans) >= 2 else ():
        ef, _ = _score_terms(manifold, _extract_terms(span, stops), 1.6)
        if (mx := float((ef := ef * z_gate).max(initial=0.0))) > 0.0:
            ent_facets.append((span, (ef / mx).astype(np.float32)))
    ent_facets = ent_facets[:3]
    order0, s0_pos = np.argsort(-s0, kind="mergesort"), np.maximum(0.0, s0 - float(np.median(s0)))
    s0_norm = (s0_pos / max(float(s0_pos.max()), 1e-6)).astype(np.float32)
    top2_gap = float(s0_norm[order0[0]] - s0_norm[order0[1]]) if n_c > 1 else 1.0
    seed0, max_p = int(order0[0]), tele.max_particles
    single_peak = (not comp_q) and (top2_gap > 0.30 or float(s0[seed0]) > 4.5)
    seed_map: dict[int, str] = {seed0: q_ent_spans[0] if q_ent_spans else (titles[seed0] or c_ids[seed0])}
    sim_lim, top_lim = max(0.35, min(0.92, hp.redundancy_curvature)), min(0.92, hp.redundancy_curvature + 0.12)
    for span, ef in ent_facets:
        if len(seed_map) < max_p and ef[te := int(np.argmax(ef))] > 0.0 and float(u_iso[te] @ u_iso[seed0]) < sim_lim:
            seed_map.setdefault(te, span)
    if len(seed_map) < max_p and (comp_q or not single_peak):
        for ci in (int(i) for i in order0[1:16] if s0_norm[int(i)] > 0.0 and int(i) not in seed_map):
            if len(seed_map) < max_p and all(float(u_iso[ci] @ u_iso[s]) < top_lim for s in seed_map):
                seed_map[ci] = titles[ci] or c_ids[ci]
    seeds, labels = list(seed_map), list(seed_map.values())
    num_br, frontier_br, visited, links_br = len(seeds), [[s] for s in seeds], [{s} for s in seeds], [[] for _ in seeds]
    branch_hop, hop_scores, q_act = np.zeros((num_br, n_c), np.float32), np.zeros((n_c,), np.float32), q_iso.copy()
    hop_post, hop_w = [h0_post] + [np.zeros((n_c,), np.float32)] * (eff_hops - 1), [1.0] + [0.0] * (eff_hops - 1)
    z_thr = 1.5 + 0.25 * math.log((1.0 - hp.alpha_sprt) / max(1e-6, 19.0 * hp.alpha_sprt))
    rare_cap, cand_cap, ast_b = max(20, n_c // 20), max(16, n_c // 25), 0.30 * blend
    def _steer(q_vec: np.ndarray, p_vec: np.ndarray) -> np.ndarray:
        return _l2_norm(hp.peclet * _l2_norm(q_vec - max(0.0, float(q_vec @ p_vec)) * p_vec) + blend * p_vec)
    for hop in range(1, min(3, eff_hops)):
        if not any(frontier_br):
            break
        lex_bridge_all, step_by_br = np.zeros((n_c,), np.float32), np.zeros((num_br, n_c), np.float32)
        par_by_br, br_by_br = np.full((num_br, n_c), -1, np.int32), [{} for _ in seeds]
        for bi, fr in enumerate(frontier_br):
            b_sc, b_par, b_lbl = step_by_br[bi], par_by_br[bi], br_by_br[bi]
            for p_ci in fr:
                pw = max(0.30, max(0.0, float(s0[p_ci])) / max(float(s0[seed0]), 1e-3)) * (1.0 if hop == 1 else 0.65)
                u_p, u_p_raw = u_iso[p_ci], manifold.u_chunk_raw[p_ci]
                dense_loc = np.zeros((n_c,), np.float32)
                for pr_i, pr_r, w_k in ((u_p, u_p_raw, 0.85 * pw), (_steer(q_act, u_p), _steer(q_raw, u_p_raw), pw)):
                    col = blend * (manifold.u_chunk_raw @ pr_r) + hp.peclet * (u_iso @ pr_i - csls_w * csls)
                    col[p_ci] = -2.0
                    dense_loc = np.maximum(dense_loc, w_k * np.maximum(0.0, _robust_z(col) - z_thr) / 3.0)
                u_bonus = np.zeros((n_c,), np.float32)
                if unmet := q_tids - t_sets[p_ci]:
                    for tid in unmet:
                        u_bonus[manifold.postings[tid]] += float(idf2[tid])
                    u_bonus /= max(float(idf2[list(unmet)].sum()), 1e-12)
                ents_all, ents_attr = manifold.chunk_entities[p_ci]
                attr_only = set(ents_attr) - set(ents_all)
                lex_loc, span_lbls = np.zeros((n_c,), np.float32), {}
                for span in ents_attr if single_peak else ents_all:
                    tids = [(t, v_idx[t]) for t in _extract_terms(span, stops) if t in v_idx and idf2[v_idx[t]] > 0.0]
                    valid = [ti for term, ti in tids if "__" not in term]
                    rare = [t for t in valid if int(manifold.token_df[t]) <= rare_cap]
                    if not valid or not rare or all(t in q_tids for t in valid):
                        continue
                    cand = set.intersection(*(set(map(int, manifold.postings[t])) for t in rare))
                    if 0 < len(cand) <= cand_cap:
                        bg_tids, base_w = [ti for term, ti in tids if "__" in term], (1.15 if len(cand) <= 3 else 0.75)
                        for dst in cand - {p_ci}:
                            if single_peak and span not in attr_only and u_bonus[dst] <= 0.0:
                                continue
                            bg_hit = 0.25 if any(b in t_sets[dst] for b in bg_tids) else 0.0
                            if (sc := pw * (base_w + bg_hit + 0.50 * float(u_bonus[dst]))) > lex_loc[dst]:
                                lex_loc[dst], span_lbls[dst] = sc, span
                s_loc = dense_loc + lex_loc
                if p_toks := t_sets[p_ci]:
                    for sci in (int(r) for r in manifold.ast_siblings[p_ci][:4] if int(r) != p_ci):
                        s_loc[sci] += pw * (ast_b + 0.05 * min(len(p_toks & t_sets[sci]), 4))
                s_loc[mask := list(visited[bi])], lex_loc[mask] = 0.0, 0.0
                lex_bridge_all, better = np.maximum(lex_bridge_all, lex_loc), s_loc > b_sc
                b_sc[better], b_par[better] = s_loc[better], p_ci
                b_lbl.update({int(d): span_lbls.get(int(d), labels[bi]) for d in np.flatnonzero(better)})
        hop_step_all = step_by_br.max(axis=0)
        hs_max, lb_max = float(hop_step_all.max(initial=0.0)), float(lex_bridge_all.max(initial=0.0))
        if single_peak and (hop > 1 or (lb_max == 0.0 and hs_max < 1.35)):
            break
        if (hs_sum := float(hop_step_all.sum())) > 0.0:
            hop_post[hop] = (hop_step_all / hs_sum).astype(np.float32)
            hop_w[hop] = min(1.0, hs_max * (0.04 if single_peak else 1.0))
            hop_scores += hop_step_all
        q_act = _l2_norm(q_act - max(0.0, float(q_act @ (u_f := u_iso[next(f[0] for f in frontier_br if f)]))) * u_f)
        branch_hop = np.maximum(branch_hop, step_by_br)
        for bi in range(num_br):
            b_sc, b_par = step_by_br[bi], par_by_br[bi]
            order_b = [int(d) for d in np.argsort(-b_sc, kind="mergesort")[:6]]
            frontier_br[bi] = nf = [d for d in order_b if b_sc[d] > 0.25 and b_par[d] >= 0 and d not in visited[bi]][:2]
            if nf:
                links_br[bi] += [(hop, int(b_par[d]), d, float(b_sc[d]), br_by_br[bi].get(d, labels[bi])) for d in nf]
                visited[bi].update(nf)
    h_max, multi_q = float(hop_scores.max(initial=0.0)), (comp_q or not single_peak)
    h_norm = (hop_scores / h_max).astype(np.float32) if h_max > 1e-6 else hop_scores
    w_hop = 0.0 if max(hop_w[1:], default=0.0) == 0.0 else (0.50 + 0.45 * hp.coalition_coupling if multi_q else 0.30)
    coal_bonus = np.zeros((n_c,), np.float32)
    if num_br >= 2:
        branch_sig = np.tile(s0_norm, (num_br, 1)) + branch_hop
        for bi in range(num_br):
            branch_sig[bi, seeds[bi]] += 0.5
            for _, _, dst, sc, _ in links_br[bi]:
                branch_sig[bi, dst] += min(1.0, sc)
        s_br = np.sort(branch_sig, axis=0)
        gmean = np.sqrt(np.maximum(s_br[-1], 0.0) * np.maximum(s_br[-2], 0.0))
        coal_bonus = (0.45 * hp.coalition_coupling * gmean / (1.0 + 0.5 * np.maximum(csls, 0.0))).astype(np.float32)
    w_f, curv = 0.35 * hp.coalition_coupling, hp.redundancy_curvature
    gate = max(0.55, 1.0 - 0.30 * curv)
    ef_mat = np.vstack([ef for _, ef in ent_facets]) if ent_facets else np.zeros((0, n_c), np.float32)
    base = s0_norm + w_hop * h_norm + coal_bonus + (w_f * ef_mat.sum(axis=0)).astype(np.float32)
    rem = np.asarray([i for i in np.argsort(-base, kind="mergesort")[: max(64, eff_k * 4)] if i != seed0], np.int32)
    max_sims = (u_iso[rem] @ u_iso[seed0]).astype(np.float32)
    reranked, used_t = [(seed0, float(base[seed0]))], ({tn} if (tn := t_norms[seed0]) else set())
    hop_cov, e_cov, last = min(1.0, 0.5 * float(h_norm[seed0])), np.minimum(1.0, ef_mat[:, seed0]), float(base[seed0])
    for _ in range(1, min(eff_k, 1 + rem.size)):
        e_gain = (w_f * (1.0 - e_cov)) @ ef_mat[:, rem]
        raw_sc = s0_norm[rem] + w_hop * (1.0 - 0.35 * hop_cov) * h_norm[rem] + coal_bonus[rem] + e_gain
        red_pen = np.maximum(0.15, 1.0 - 0.65 * curv * np.maximum(0.0, max_sims - gate) / max(1e-6, 1.0 - gate))
        if curv > 0.0 and used_t:
            red_pen[[i for i, c in enumerate(rem) if t_norms[int(c)] in used_t]] *= 0.85
        pick = int(np.argmax(gains := raw_sc * red_pen))
        best_idx, last = int(rem[pick]), min(last - 1e-6, float(gains[pick]))
        reranked.append((best_idx, last))
        used_t.update((t_norms[best_idx],) if t_norms[best_idx] else ())
        hop_cov, e_cov = min(1.0, hop_cov + 0.5 * float(h_norm[best_idx])), np.minimum(1.0, e_cov + ef_mat[:, best_idx])
        rem = np.delete(rem, pick)
        max_sims = np.maximum(np.delete(max_sims, pick), u_iso[rem] @ u_iso[best_idx])
    seed_info = [(s, lbl, float(s0[s])) for s, lbl in zip(seeds, labels, strict=True)]
    route, routes = _build_routes(manifold, links_br, seed_info, tele.route_min_gate)
    posts, ranked = softmax(base.astype(np.float64) * 3.0).astype(np.float32), [(c_ids[ci], sc) for ci, sc in reranked]
    return SubStateResult(ranked, posts, tuple(hop_post), tuple(hop_w), route, num_br, routes)


def format_evidence_route(
    route: Sequence[EvidenceHop], *, max_chars: int | None = None, params: ManifoldHyperparams | None = None
) -> str:
    """Format an acyclic multi-hop EvidenceHop trajectory within a character budget."""
    budget = derive_manifold_telemetry(params).route_max_chars if max_chars is None else int(max_chars)
    if not route or budget <= 0:
        return ""
    parts, seen, used = [], set(), 0
    for hop in route:
        if (hop.chunk_id, hop.hop) not in seen:
            seen.add((hop.chunk_id, hop.hop))
            head = f"[H{hop.hop}{' B' + str(hop.branch) if hop.branch else ''}] {hop.title + ': ' if hop.title else ''}"
            line = f"{head}{hop.snippet}{' (' + hop.bridge + ')' if hop.bridge else ''}".strip()
            if parts and used + 1 + len(line) > budget:
                break
            parts.append(line[:budget] if not parts else line)
            used += len(parts[-1]) + (1 if len(parts) > 1 else 0)
    return "\n".join(parts)
