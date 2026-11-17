"""Zero-LLM hybrid VectorSearch ranker, read-only SQLite access, shared NLP, and disk hydration."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import threading
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from graphlib import CycleError, TopologicalSorter
from pathlib import Path
from typing import Any

import numpy as np
import spacy
import sqlite_vec
from numpy.typing import NDArray
from scipy.linalg import eigh
from scipy.special import softmax

from yaw.core import (
    ChunkPointer,
    HydratedPassage,
    RetrievalParams,
    YawRuntimeError,
    json_document_fields,
    render_document_text,
    resolve_corpus_path,
)

FTS_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for", "from",
        "had", "has", "have", "how", "in", "into", "is", "it", "its", "of", "on", "or", "that",
        "the", "their", "then", "there", "these", "they", "this", "to", "was", "were", "what",
        "when", "where", "which", "who", "whom", "why", "with",
    }
)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'_-]*")
_CAP_SPAN_RE = re.compile(r"(?:[A-Z][A-Za-z0-9_-]*(?:\s+[A-Z0-9][A-Za-z0-9_-]*)*)")
_BRIDGE_CONN = frozenset({"of", "the", "and", "-", "de", "la", "van", "von"})
_AUX_LEMMAS = frozenset({"be", "have", "do", "sr", "jr", "ii", "iii", "iv"})
_MAX_FTS_TERMS = 24
_MAX_HYDRATE_SPAN = 5000
_SPACY_LOCK = threading.Lock()
type _LinksByBranch = list[list[tuple[int, int, int, float, str]]]
type _FileCache = dict[Path, tuple[str, list[str]] | None]
type _JsonCache = dict[tuple[Path, int, int], Any]


@dataclass(frozen=True, slots=True)
class EvidenceHop:
    """One evidence step in an acyclic multi-hop reasoning route."""

    hop: int
    chunk_id: int
    title: str
    snippet: str
    score: float
    bridge: str = ""
    branch: int = 0


@dataclass(frozen=True, slots=True)
class SearchTrace:
    """Ranked chunk posteriors, per-hop trajectories, and acyclic evidence routes."""

    ranked_chunks: list[tuple[int, float]]
    chunk_posteriors: NDArray[np.float32]
    hop_posteriors: tuple[NDArray[np.float32], ...]
    hop_weights: tuple[float, ...]
    route: tuple[EvidenceHop, ...]
    num_particles: int = 1
    routes: tuple[tuple[EvidenceHop, ...], ...] = ()


@dataclass(frozen=True, slots=True)
class VectorSearch:
    """Precomputed isotropic spectral and lexical search index over indexed document chunks."""

    pointers: tuple[ChunkPointer, ...]
    id_to_pos: dict[int, int]
    chunk_ids: tuple[int, ...]
    chunk_titles: tuple[str, ...]
    chunk_title_norms: tuple[str, ...]
    u_chunk_raw: NDArray[np.float32]
    u_chunk_iso: NDArray[np.float32]
    q_rogue: NDArray[np.float32]
    abtt_removed_dims: int
    rmt_lambda_plus: float
    csls_chunk: NDArray[np.float32]
    vocab_index: dict[str, int]
    postings: tuple[NDArray[np.int32], ...]
    token_df: NDArray[np.int32]
    token_idf2: NDArray[np.float32]
    chunk_tok_sets: tuple[frozenset[int], ...]
    ast_siblings: tuple[NDArray[np.int32], ...]
    chunk_entities: tuple[tuple[list[str], list[str]], ...]
    params: RetrievalParams = RetrievalParams()
    db_data_version: int = 0

    @property
    def size(self) -> int:
        return len(self.chunk_ids)


@lru_cache(maxsize=1)
def _load_spacy_cached() -> Any:
    return spacy.load("en_core_web_sm", exclude=["ner"])


def get_spacy_nlp() -> Any:
    """Return the shared lazy-loaded spaCy `en_core_web_sm` pipeline (NER excluded)."""
    with _SPACY_LOCK:
        return _load_spacy_cached()


def extract_terms(text: str, stops: Collection[str]) -> list[str]:
    """Extract lowercased non-stop unigrams and adjacent bigrams."""
    words = [w for m in _WORD_RE.finditer(text) if len(w := m.group(0).lower()) > 1 and w not in stops]
    return words + [f"{words[k]}__{words[k + 1]}" for k in range(len(words) - 1)]


def extract_entities(
    doc: Any, stops: Collection[str], *, include_attr: bool = False, max_chars: int | None = None
) -> list[str]:
    """Extract clean proper-noun spans, noun chunks, and optional sentence attr nouns."""
    spans: list[str] = []
    # Restrict tokens by char offset so callers can reuse one parsed Doc without re-running spaCy.
    toks = [t for t in doc if t.idx < max_chars] if max_chars is not None else list(doc)
    idx = 0
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
        spans.extend(
            nc.text for nc in doc.noun_chunks
            if nc.root.pos_ == "PROPN" and (max_chars is None or nc.start_char < max_chars)
        )
        if include_attr:
            for sent in list(doc.sents)[:2]:
                if max_chars is not None and sent.start_char >= max_chars:
                    break
                for tok in sent:
                    if max_chars is not None and tok.idx >= max_chars:
                        break
                    if tok.pos_ == "NOUN" and not tok.is_stop and tok.dep_ == "attr":
                        spans.append(tok.text)
                        spans.extend(
                            c.text for c in tok.conjuncts
                            if c.pos_ == "NOUN" and not c.is_stop and (max_chars is None or c.idx < max_chars)
                        )
    fallback_text = doc.text[:max_chars] if max_chars is not None else doc.text
    dedup: dict[str, str] = {}
    for raw in spans or _CAP_SPAN_RE.findall(fallback_text):
        clean = raw.strip(" \t\n\r\"'()[]{}:;,.?!")
        if len(clean) >= 2 and extract_terms(clean, stops):
            dedup.setdefault(re.sub(r"\s+", " ", clean).lower(), clean)
    return list(dedup.values())


def extract_text_features(title: str, body: str) -> tuple[list[str], tuple[list[str], list[str]]]:
    """Extract deduplicated index terms and `(main_entities, attr_entities)` for one chunk."""
    nlp = get_spacy_nlp()
    stops = nlp.Defaults.stop_words
    terms = list(dict.fromkeys(extract_terms(f"{title}\n{body}", stops)))
    doc_main = nlp(body)
    attr_limit = 320 if len(body) > 320 else None
    return terms, (
        extract_entities(doc_main, stops),
        extract_entities(doc_main, stops, include_attr=True, max_chars=attr_limit)[:8],
    )


def open_readonly_db(db_path: Path | str, *, timeout: float = 5.0) -> sqlite3.Connection:
    """Open a dedicated read-only SQLite connection with `sqlite-vec` loaded."""
    resolved = Path(db_path).resolve()
    conn = sqlite3.connect(f"file:{resolved.as_posix()}?mode=ro", uri=True, timeout=timeout, check_same_thread=False)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def _validate_max_chars(max_chars: int | None) -> int | None:
    if max_chars is None:
        return None
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 0:
        raise ValueError(f"max_chars must be a non-negative integer or None, got {max_chars!r}")
    return max_chars


def _crop_text(text: str, max_chars: int | None) -> str:
    limit = _validate_max_chars(max_chars)
    if limit is None or limit == 0 or len(text) <= limit:
        return text
    marker = "\n[... cropped ...]"
    return text[: max(1, limit - len(marker))] + marker if limit > len(marker) else text[:limit]


def _hydrate_one(
    corpus_root: Path,
    ptr: ChunkPointer,
    score: float,
    max_chars: int | None,
    include_island: bool,
    full_span: bool,
    file_cache: _FileCache | None = None,
    json_cache: _JsonCache | None = None,
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

    suffix = Path(ptr.rel_posix_path).suffix.lower()
    if suffix in (".json", ".jsonl", ".ndjson"):
        cache_key = (target, sl, el)
        if json_cache is not None and cache_key in json_cache:
            parsed = json_cache[cache_key]
        else:
            try:
                parsed = json.loads(raw_slice)
            except json.JSONDecodeError as exc:
                raise YawRuntimeError("CONSISTENCY_ERROR", f"Corrupted JSON document at {target}: {exc}") from exc
            if json_cache is not None:
                json_cache[cache_key] = parsed
        fallback = Path(ptr.rel_posix_path).stem
        item = parsed
        if isinstance(parsed, list):
            if not parsed:
                raise YawRuntimeError("CONSISTENCY_ERROR", f"Corrupted JSON document at {target}: empty JSON array")
            item = next((x for x in parsed if json_document_fields(x, fallback)[0] == ptr.title), parsed[0])
        title, body = json_document_fields(item, fallback)
        if not full_span and ptr.char_end > ptr.char_start:
            body = body[ptr.char_start : ptr.char_end]
        rendered = render_document_text(title, body)
    else:
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
            and not rendered.lower().startswith(("title:", "wikipedia title:"))
        ):
            rendered = f"{ptr.title}\n{rendered}"

    if limit is not None and limit > 0:
        rendered = _crop_text(rendered, limit)
    return HydratedPassage(ptr, rendered, float(score), stale_on_disk=False)


def hydrate_passage(
    corpus_root: Path,
    ptr: ChunkPointer,
    score: float = 0.0,
    *,
    max_chars: int | None = None,
    include_island: bool = False,
    full_span: bool = False,
) -> HydratedPassage:
    """Hydrate 1-indexed source lines from disk with SHA-256 freshness verification."""
    return _hydrate_one(corpus_root, ptr, score, max_chars, include_island, full_span)


def _l2_norm(arr: NDArray[np.float32], *, row_wise: bool = False) -> NDArray[np.float32]:
    data = np.asarray(arr, dtype=np.float32)
    if not data.size:
        return data
    if row_wise:
        norm = np.maximum(np.linalg.norm(data, axis=1, keepdims=True), 1e-12)
        return (data / norm).astype(np.float32, copy=False)
    return (data.reshape(-1) / max(float(np.linalg.norm(data)), 1e-12)).astype(np.float32, copy=False)


def _robust_z(arr: NDArray[np.float32]) -> NDArray[np.float32]:
    med = float(np.median(arr))
    return ((arr - med) / max(float(np.median(np.abs(arr - med))) * 1.4826, 1e-3)).astype(np.float32, copy=False)


def _build_ast_siblings(ptrs: Sequence[ChunkPointer]) -> tuple[NDArray[np.int32], ...]:
    by_file: dict[str, list[tuple[int, ChunkPointer]]] = {}
    for idx, ptr in enumerate(ptrs):
        if ptr.ast_island_start > 0 and ptr.ast_island_end >= ptr.ast_island_start:
            by_file.setdefault(ptr.rel_posix_path, []).append((idx, ptr))
    empty = np.empty((0,), np.int32)
    siblings = [empty] * len(ptrs)
    for items in by_file.values():
        items.sort(key=lambda pair: (pair[1].start_line, pair[1].end_line, int(pair[1].chunk_id)))
        for idx, ptr in items:
            ps, pe = ptr.ast_island_start, ptr.ast_island_end
            ovl = [oi for oi, o in items if oi != idx and max(ps, o.ast_island_start) <= min(pe, o.ast_island_end)]
            if ovl:
                siblings[idx] = np.asarray(ovl, np.int32)
    return tuple(siblings)


def build_vector_search(
    conn: sqlite3.Connection, params: RetrievalParams | None = None, *, data_version: int = 0
) -> VectorSearch:
    """Load `chunks` metadata/embeddings from SQLite and precompute ABTT isotropic & lexical structures."""
    hp = params or RetrievalParams()
    rows = conn.execute(
        "SELECT chunk_id, embedding, rel_path, source_hash, start_line, end_line, char_start, char_end, "
        "ast_island_start, ast_island_end, title, tokens_json, entities_json FROM chunks ORDER BY chunk_id"
    ).fetchall()
    n_c = len(rows)
    if n_c == 0:
        empty_m = np.zeros((0, 0), np.float32)
        empty_i, empty_f, empty_r = np.zeros((0,), np.int32), np.zeros((0,), np.float32), np.zeros((0, 0), np.float32)
        return VectorSearch(
            (), {}, (), (), (), empty_m, empty_m, empty_r, 0, 1.0, empty_f, {}, (), empty_i, empty_f,
            (), (), (), hp, data_version,
        )

    ptrs: list[ChunkPointer] = []
    vecs_list: list[NDArray[np.float32]] = []
    doc_terms: list[list[str]] = []
    c_ents: list[tuple[list[str], list[str]]] = []
    for row in rows:
        cid = int(row[0])
        ptrs.append(
            ChunkPointer(
                chunk_id=cid, rel_posix_path=str(row[2]), start_line=int(row[4]), end_line=int(row[5]),
                title=str(row[10]), ast_island_start=int(row[8]), ast_island_end=int(row[9]),
                source_hash=str(row[3]), char_start=int(row[6]), char_end=int(row[7]),
            )
        )
        # Zero-copy view over SQLite BLOB buffer prior to batched vstack/L2 normalization.
        vecs_list.append(np.frombuffer(row[1], dtype="<f4"))
        doc_terms.append(list(dict.fromkeys(str(t) for t in json.loads(str(row[11])))))
        raw_ents = json.loads(str(row[12]))
        c_ents.append(([str(e) for e in raw_ents[0]], [str(e) for e in raw_ents[1][:8]]))

    c_ids = tuple(int(p.chunk_id) for p in ptrs)
    titles = tuple(p.title for p in ptrs)
    t_norms = tuple(re.sub(r"\W+", " ", t).strip().lower() for t in titles)
    u_raw = _l2_norm(np.vstack(vecs_list), row_wise=True)
    dim = int(u_raw.shape[1])
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
            e = min(n_c, s + 1024)
            sims = u_iso[s:e] @ u_t
            sims[np.arange(e - s), np.arange(s, e)] = -2.0
            csls[s:e] = np.maximum(np.partition(sims, -k_eff, axis=1)[:, -k_eff:], 0.0).mean(axis=1)

    post_map: dict[str, list[int]] = {}
    for ci, terms in enumerate(doc_terms):
        for term in terms:
            post_map.setdefault(term, []).append(ci)
    v_idx = {term: idx for idx, term in enumerate(sorted(post_map))}
    posts = tuple(np.asarray(post_map[t], np.int32) for t in v_idx)
    t_df = np.asarray([len(p) for p in posts], np.int32)
    idf2 = np.square(np.log((n_c + 1.0) / (t_df.astype(np.float64) + 0.5))).astype(np.float32)
    idf2[t_df > (0.25 * n_c if n_c >= 16 else n_c)] = 0.0
    t_sets = tuple(frozenset(v_idx[t] for t in tms if idf2[v_idx[t]] > 0.0) for tms in doc_terms)
    return VectorSearch(
        tuple(ptrs), {cid: i for i, cid in enumerate(c_ids)}, c_ids, titles, t_norms, u_raw, u_iso, q_rogue,
        abtt_d, float(lam), csls, v_idx, posts, t_df, idf2, t_sets, _build_ast_siblings(ptrs),
        tuple(c_ents), hp, data_version,
    )


def query_external_hints(
    conn: sqlite3.Connection, query_text: str, query_vec: NDArray[np.float32], *, fts_k: int = 50, vec_k: int = 200
) -> tuple[dict[int, float], dict[int, float]]:
    """Query SQLite contentless FTS5 (`bm25`) and `vec0` KNN cosine hints."""
    fts_scores: dict[int, float] = {}
    if fts_k > 0 and (raw_tokens := re.findall(r"\w+", query_text.lower())):
        uniq = list(dict.fromkeys(t for t in raw_tokens if len(t) > 1)) or list(dict.fromkeys(raw_tokens))
        active = ([t for t in uniq if t not in FTS_STOPWORDS] or uniq)[:_MAX_FTS_TERMS]
        match_expr = " OR ".join(f'"{t}"' for t in active)
        rows = conn.execute(
            "SELECT rowid, bm25(chunk_fts) AS raw FROM chunk_fts WHERE chunk_fts MATCH ? ORDER BY raw ASC LIMIT ?",
            (match_expr, int(fts_k)),
        ).fetchall()
        hits = [(int(r[0]), max(1e-6, -float(r[1]))) for r in rows]
        if total := sum(s for _, s in hits):
            fts_scores = {cid: s / total for cid, s in hits}

    vec_scores: dict[int, float] = {}
    q_arr = np.asarray(query_vec, dtype=np.float32).reshape(-1)
    if vec_k > 0 and q_arr.size > 0:
        # Encode contiguous little-endian float32 bytes directly for sqlite-vec without Python list boxing.
        blob = np.ascontiguousarray(q_arr, dtype="<f4").tobytes()
        rows = conn.execute(
            "WITH knn AS MATERIALIZED (SELECT chunk_id, distance FROM chunks WHERE embedding MATCH ? AND k = ? "
            "ORDER BY distance) SELECT chunk_id, (1.0 - distance) FROM knn ORDER BY distance ASC, chunk_id ASC",
            (blob, min(int(vec_k), 4096)),
        ).fetchall()
        vec_scores = {int(r[0]): float(r[1]) for r in rows}
    return fts_scores, vec_scores


def _scale_hints(n_c: int, scores: Mapping[int, float] | None, id_map: Mapping[int, int]) -> NDArray[np.float32]:
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


def _score_terms(vs: VectorSearch, terms: Sequence[str], bg_w: float = 1.5) -> tuple[NDArray[np.float32], set[int]]:
    lex = np.zeros((vs.size,), np.float32)
    tot_w = 0.0
    active: set[int] = set()
    for term in dict.fromkeys(terms):
        if (ti := vs.vocab_index.get(term)) is not None and vs.token_idf2[ti] > 0.0:
            weight = float(vs.token_idf2[ti]) * (bg_w if "__" in term else 1.0)
            tot_w += weight
            lex[vs.postings[ti]] += weight
            active.add(ti)
    return ((lex / np.float32(tot_w)).astype(np.float32, copy=False) if tot_w > 0.0 else lex), active


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


@lru_cache(maxsize=512)
def _analyze_query_cached(nlp: Any, query_text: str) -> tuple[tuple[str, ...], tuple[str, ...], bool]:
    # Cache deterministic query NLP features to avoid repeating spaCy parses in parameter sweeps.
    stops = nlp.Defaults.stop_words
    q_terms = tuple(extract_terms(query_text, stops))
    q_doc = nlp(query_text)
    q_ent_spans = tuple(extract_entities(q_doc, stops))
    return q_terms, q_ent_spans, _is_comp_q(q_doc, q_ent_spans)


def _build_routes(
    vs: VectorSearch, links_br: _LinksByBranch, seed_info: list[tuple[int, str, float]], gate: float
) -> tuple[tuple[EvidenceHop, ...], tuple[tuple[EvidenceHop, ...], ...]]:
    edges = [(0, -1, s, max(sc, 0.1), lbl, bi) for bi, (s, lbl, sc) in enumerate(seed_info)]
    edges += [(h, p, d, s, b, i) for i, ls in enumerate(links_br) for h, p, d, s, b in ls if s >= gate and p != d]
    if len(edges) == len(seed_info):
        return (), ()
    routes_br: list[list[EvidenceHop]] = [[] for _ in seed_info]
    seen_br: list[set[int]] = [set() for _ in seed_info]
    graph: dict[int, set[int]] = {}
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
        ptr = vs.pointers[d_ci]
        snip = (vs.chunk_titles[d_ci] or f"{ptr.rel_posix_path}:{ptr.start_line}-{ptr.end_line}").replace("\n", " ")
        routes_br[bi].append(EvidenceHop(hop, vs.chunk_ids[d_ci], vs.chunk_titles[d_ci], snip, score, bridge, bi))
    active = tuple(tuple(br) for br in routes_br if len(br) >= 2)
    flat = sorted((h for b in active for h in b), key=lambda h: (-h.hop, h.score, -h.branch))
    return tuple({h.chunk_id: h for h in flat}.values()), active


def search_vectors(
    vs: VectorSearch,
    query_vec: NDArray[np.float32] | Sequence[float],
    query_text: str,
    *,
    top_k: int | None = None,
    fts_scores: Mapping[int, float] | None = None,
    vec_scores: Mapping[int, float] | None = None,
    params: RetrievalParams | None = None,
    num_hops: int | None = None,
) -> SearchTrace:
    """Execute zero-LLM ABTT+CSLS Hop-0 and multi-hop entity-steered vector search."""
    hp = params or vs.params
    n_c = vs.size
    eff_k = 10 if top_k is None else max(1, int(top_k))
    eff_hops = 8 if num_hops is None else max(1, int(num_hops))
    route_gate = max(0.05, min(0.50, 0.15 * (1.0 - hp.alpha_sprt)))
    if n_c == 0:
        return SearchTrace([], np.zeros((0,), np.float32), (), (), (), 0, ())

    q_raw, dim = _l2_norm(np.asarray(query_vec, np.float32)), int(vs.u_chunk_raw.shape[1])
    if q_raw.size != dim:
        raise ValueError(f"query_vec dim {q_raw.size} != index dim {dim}")
    q_rogue, u_iso, csls = vs.q_rogue, vs.u_chunk_iso, vs.csls_chunk
    q_iso = _l2_norm(q_raw - q_rogue @ (q_rogue.T @ q_raw)) if q_rogue.shape[1] > 0 else q_raw
    blend, csls_w = 1.0 - hp.peclet, 0.5 * (1.0 - hp.peclet)
    z_d = _robust_z(blend * (vs.u_chunk_raw @ q_raw) + hp.peclet * (u_iso @ q_iso - csls_w * csls))

    nlp = get_spacy_nlp()
    stops = nlp.Defaults.stop_words
    q_terms, q_ent_spans, comp_q = _analyze_query_cached(nlp, query_text)
    c_ids, titles, t_norms = vs.chunk_ids, vs.chunk_titles, vs.chunk_title_norms
    v_idx, idf2, t_sets = vs.vocab_index, vs.token_idf2, vs.chunk_tok_sets
    lex_c, q_tids = _score_terms(vs, q_terms, 1.5)
    z_fts, z_vec = _scale_hints(n_c, fts_scores, vs.id_to_pos), _scale_hints(n_c, vec_scores, vs.id_to_pos)
    s0 = z_d + 2.2 * (1.0 - 0.5 * hp.peclet) * lex_c + hp.external_trust * (2.2 * (z_fts / 4.0) + 1.2 * (z_vec / 4.0))
    h0_post = softmax(np.asarray(s0, np.float64)).astype(np.float32)

    z_gate = np.maximum(0.1, (z_d + 2.0) / 4.0)
    ent_facets: list[tuple[str, NDArray[np.float32]]] = []
    for span in q_ent_spans if len(q_ent_spans) >= 2 else ():
        ef, _ = _score_terms(vs, extract_terms(span, stops), 1.6)
        scaled = ef * z_gate
        if (mx := float(scaled.max(initial=0.0))) > 0.0:
            ent_facets.append((span, (scaled / mx).astype(np.float32)))
    ent_facets = ent_facets[:3]

    order0 = np.argsort(-s0, kind="mergesort")
    s0_pos = np.maximum(0.0, s0 - float(np.median(s0)))
    s0_norm = (s0_pos / max(float(s0_pos.max()), 1e-6)).astype(np.float32)
    top2_gap = float(s0_norm[order0[0]] - s0_norm[order0[1]]) if n_c > 1 else 1.0
    seed0, max_p = int(order0[0]), 3
    single_peak = (not comp_q) and (top2_gap > 0.30 or float(s0[seed0]) > 4.5)
    seed_map: dict[int, str] = {seed0: q_ent_spans[0] if q_ent_spans else (titles[seed0] or str(c_ids[seed0]))}
    sim_lim, top_lim = max(0.35, min(0.92, hp.redundancy_curvature)), min(0.92, hp.redundancy_curvature + 0.12)
    for span, ef in ent_facets:
        te = int(np.argmax(ef))
        if len(seed_map) < max_p and ef[te] > 0.0 and float(u_iso[te] @ u_iso[seed0]) < sim_lim:
            seed_map.setdefault(te, span)
    if len(seed_map) < max_p and (comp_q or not single_peak):
        for ci in (int(i) for i in order0[1:16] if s0_norm[int(i)] > 0.0 and int(i) not in seed_map):
            if len(seed_map) < max_p and all(float(u_iso[ci] @ u_iso[s]) < top_lim for s in seed_map):
                seed_map[ci] = titles[ci] or str(c_ids[ci])

    seeds, labels = list(seed_map), list(seed_map.values())
    num_br = len(seeds)
    frontier_br, visited, links_br = [[s] for s in seeds], [{s} for s in seeds], [[] for _ in seeds]
    branch_hop = np.zeros((num_br, n_c), np.float32)
    hop_scores, q_act = np.zeros((n_c,), np.float32), q_iso.copy()
    hop_post = [h0_post] + [np.zeros((n_c,), np.float32)] * (eff_hops - 1)
    hop_w = [1.0] + [0.0] * (eff_hops - 1)
    z_thr = 1.5 + 0.25 * math.log((1.0 - hp.alpha_sprt) / max(1e-6, 19.0 * hp.alpha_sprt))
    rare_cap, cand_cap, ast_b = max(20, n_c // 20), max(16, n_c // 25), 0.30 * blend

    def _steer(qv: NDArray[np.float32], pv: NDArray[np.float32]) -> NDArray[np.float32]:
        return _l2_norm(hp.peclet * _l2_norm(qv - max(0.0, float(qv @ pv)) * pv) + blend * pv)

    for hop in range(1, min(3, eff_hops)):
        if not any(frontier_br):
            break
        lex_bridge_all = np.zeros((n_c,), np.float32)
        step_by_br = np.zeros((num_br, n_c), np.float32)
        par_by_br = np.full((num_br, n_c), -1, np.int32)
        br_by_br: list[dict[int, str]] = [{} for _ in seeds]
        for bi, fr in enumerate(frontier_br):
            b_sc, b_par, b_lbl = step_by_br[bi], par_by_br[bi], br_by_br[bi]
            for p_ci in fr:
                pw = max(0.30, max(0.0, float(s0[p_ci])) / max(float(s0[seed0]), 1e-3)) * (1.0 if hop == 1 else 0.65)
                u_p, u_p_raw = u_iso[p_ci], vs.u_chunk_raw[p_ci]
                dense_loc = np.zeros((n_c,), np.float32)
                for pr_i, pr_r, w_k in ((u_p, u_p_raw, 0.85 * pw), (_steer(q_act, u_p), _steer(q_raw, u_p_raw), pw)):
                    col = blend * (vs.u_chunk_raw @ pr_r) + hp.peclet * (u_iso @ pr_i - csls_w * csls)
                    col[p_ci] = -2.0
                    dense_loc = np.maximum(dense_loc, w_k * np.maximum(0.0, _robust_z(col) - z_thr) / 3.0)
                u_bonus = np.zeros((n_c,), np.float32)
                if unmet := q_tids - t_sets[p_ci]:
                    for tid in unmet:
                        u_bonus[vs.postings[tid]] += float(idf2[tid])
                    u_bonus /= max(float(idf2[list(unmet)].sum()), 1e-12)
                ents_all, ents_attr = vs.chunk_entities[p_ci]
                attr_only = set(ents_attr) - set(ents_all)
                lex_loc, span_lbls = np.zeros((n_c,), np.float32), {}
                for span in ents_attr if single_peak else ents_all:
                    tids = [(t, v_idx[t]) for t in extract_terms(span, stops) if t in v_idx and idf2[v_idx[t]] > 0.0]
                    valid = [ti for term, ti in tids if "__" not in term]
                    rare = [t for t in valid if int(vs.token_df[t]) <= rare_cap]
                    if not valid or not rare or all(t in q_tids for t in valid):
                        continue
                    if len(rare) == 1 and int(vs.token_df[rare[0]]) > cand_cap:
                        continue
                    # Intersect sorted unique int32 posting arrays smallest-first without Python set/int boxing.
                    rare.sort(key=lambda t: int(vs.token_df[t]))
                    cand_arr = vs.postings[rare[0]]
                    for t in rare[1:]:
                        cand_arr = np.intersect1d(cand_arr, vs.postings[t], assume_unique=True)
                        if cand_arr.size == 0:
                            break
                    if 0 < cand_arr.size <= cand_cap:
                        bg_tids = [ti for term, ti in tids if "__" in term]
                        base_w = 1.15 if cand_arr.size <= 3 else 0.75
                        for dst_np in cand_arr:
                            dst = int(dst_np)
                            if dst == p_ci or (single_peak and span not in attr_only and u_bonus[dst] <= 0.0):
                                continue
                            bg_hit = 0.25 if any(b in t_sets[dst] for b in bg_tids) else 0.0
                            if (sc := pw * (base_w + bg_hit + 0.50 * float(u_bonus[dst]))) > lex_loc[dst]:
                                lex_loc[dst], span_lbls[dst] = sc, span
                s_loc = dense_loc + lex_loc
                if p_toks := t_sets[p_ci]:
                    for sci in (int(r) for r in vs.ast_siblings[p_ci][:4] if int(r) != p_ci):
                        s_loc[sci] += pw * (ast_b + 0.05 * min(len(p_toks & t_sets[sci]), 4))
                mask = list(visited[bi])
                s_loc[mask], lex_loc[mask] = 0.0, 0.0
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
        u_f = u_iso[next(f[0] for f in frontier_br if f)]
        q_act = _l2_norm(q_act - max(0.0, float(q_act @ u_f)) * u_f)
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
    reranked: list[tuple[int, float]] = [(seed0, float(base[seed0]))]
    if (n_rem := int(rem.size)) > 0 and eff_k > 1:
        # Pre-slice candidate arrays once and mask picked entries to avoid per-step np.delete copies.
        u_rem = u_iso[rem]
        s0_rem, h_rem, coal_rem, ef_rem = s0_norm[rem], h_norm[rem], coal_bonus[rem], ef_mat[:, rem]
        rem_titles = [t_norms[int(c)] for c in rem]
        max_sims = (u_rem @ u_iso[seed0]).astype(np.float32)
        alive = np.ones((n_rem,), dtype=bool)
        title_pen = np.ones((n_rem,), dtype=np.float32)
        used_t: set[str] = set()
        if curv > 0.0 and t_norms[seed0]:
            t0 = t_norms[seed0]
            used_t.add(t0)
            for i, tc in enumerate(rem_titles):
                if tc == t0:
                    title_pen[i] = 0.85
        hop_cov, e_cov = min(1.0, 0.5 * float(h_norm[seed0])), np.minimum(1.0, ef_mat[:, seed0])
        last, denom_gate = float(base[seed0]), max(1e-6, 1.0 - gate)
        for _ in range(1, min(eff_k, 1 + n_rem)):
            e_gain = (w_f * (1.0 - e_cov)) @ ef_rem
            raw_sc = s0_rem + w_hop * (1.0 - 0.35 * hop_cov) * h_rem + coal_rem + e_gain
            red_pen = np.maximum(0.15, 1.0 - 0.65 * curv * np.maximum(0.0, max_sims - gate) / denom_gate)
            gains = np.where(alive, raw_sc * (red_pen * title_pen), -np.inf)
            pick = int(np.argmax(gains))
            best_idx = int(rem[pick])
            last = min(last - 1e-6, float(gains[pick]))
            reranked.append((best_idx, last))
            alive[pick] = False
            if curv > 0.0 and (bt := rem_titles[pick]) and bt not in used_t:
                used_t.add(bt)
                for i, tc in enumerate(rem_titles):
                    if tc == bt:
                        title_pen[i] = 0.85
            hop_cov = min(1.0, hop_cov + 0.5 * float(h_norm[best_idx]))
            e_cov = np.minimum(1.0, e_cov + ef_rem[:, pick])
            max_sims = np.maximum(max_sims, u_rem @ u_rem[pick])

    seed_info = [(s, lbl, float(s0[s])) for s, lbl in zip(seeds, labels, strict=True)]
    route, routes = _build_routes(vs, links_br, seed_info, route_gate)
    posts = softmax(base.astype(np.float64) * 3.0).astype(np.float32)
    return SearchTrace(
        [(c_ids[ci], sc) for ci, sc in reranked], posts, tuple(hop_post), tuple(hop_w), route, num_br, routes
    )


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
    """Apply MaxP deduplication by `(rel_posix_path, start_line, end_line)` and hydrate top `top_k` passages."""
    limit = _validate_max_chars(max_chars)
    if top_k <= 0 or not ranked_chunks:
        return ()
    out: list[HydratedPassage] = []
    seen_spans: set[tuple[str, int, int]] = set()
    # Scope file/JSON caches strictly to this call so repeated chunks from one file read disk once.
    file_cache: _FileCache = {}
    json_cache: _JsonCache = {}
    for cid, score in ranked_chunks:
        if (pos := vs.id_to_pos.get(int(cid))) is None:
            continue
        ptr = vs.pointers[pos]
        span = (ptr.rel_posix_path, int(ptr.start_line), int(ptr.end_line))
        if span in seen_spans:
            continue
        seen_spans.add(span)
        out.append(
            _hydrate_one(
                corpus_root, ptr, float(score), limit, include_island, full_span, file_cache, json_cache
            )
        )
        if len(out) >= top_k:
            break
    return tuple(out)


def format_evidence_route(route: Sequence[EvidenceHop], *, max_chars: int | None = None) -> str:
    """Format an acyclic multi-hop EvidenceHop trajectory, optionally bounded by `max_chars`."""
    limit = _validate_max_chars(max_chars)
    if not route:
        return ""
    budget = limit if limit is not None and limit > 0 else None
    parts: list[str] = []
    seen: set[tuple[int, int]] = set()
    used = 0
    for hop in route:
        if (hop.chunk_id, hop.hop) in seen:
            continue
        seen.add((hop.chunk_id, hop.hop))
        head = f"[H{hop.hop}{' B' + str(hop.branch) if hop.branch else ''}] {hop.title + ': ' if hop.title else ''}"
        line = f"{head}{hop.snippet}{' (' + hop.bridge + ')' if hop.bridge else ''}".strip()
        if budget is not None:
            if parts and used + 1 + len(line) > budget:
                break
            parts.append(line[:budget] if not parts else line)
        else:
            parts.append(line)
        used += len(parts[-1]) + (1 if len(parts) > 1 else 0)
    return "\n".join(parts)
    