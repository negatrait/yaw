"""Multilingual unicode61 FTS5 lexical search, compound decompounding, and title matching."""

from __future__ import annotations

import math
import sqlite3
import unicodedata
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath

import numpy as np
from numpy.typing import NDArray

from yaw.core import YawRuntimeError

from .db import snapshot_tx
from .types import CACHE_LOCK, CACHE_MAX_ENTRIES, CJK_RUN_RE, WORD_TOK_RE, RetrievalIndex

_ASCII_NORM_TRANS = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"})


@dataclass(slots=True)
class _AnchorCtx:
    conn: sqlite3.Connection
    idx: RetrievalIndex
    eligible: NDArray[np.bool_]
    n_eff: int
    all_eligible: bool
    elig_list: list[bool] | None
    max_df_filter: int | None
    word_row_limit: int | None
    vocab_doc_counts: dict[str, int] | None
    cache: dict[tuple[str, bool, bool], set[int]] = field(default_factory=dict)
    oversized: set[tuple[str, bool, bool]] = field(default_factory=set)
    anchors: dict[str, set[int]] = field(default_factory=dict)


def eligible_mask(
    idx: RetrievalIndex,
    *,
    path_glob: str | None = None,
    masked_rel_paths: Collection[str] | None = None,
) -> NDArray[np.bool_]:
    """Compute boolean chunk mask applying path glob and masked relative paths."""
    n = idx.size
    if path_glob is None and (masked_rel_paths is None or not bool(masked_rel_paths)):
        return np.ones(n, dtype=bool)
    masked_set = {p.replace("\\", "/") for p in masked_rel_paths} if masked_rel_paths is not None else set()
    allowed_paths = np.array(
        [
            (p not in masked_set) and (path_glob is None or PurePosixPath(p).full_match(path_glob))
            for p in idx.distinct_paths
        ],
        dtype=bool,
    )
    return allowed_paths[idx.path_idx]


def fts_quote(term: str) -> str:
    """Quote a literal token or phrase safely for FTS5 MATCH queries."""
    return '"' + term.replace('"', '""') + '"'


def match_postings(
    conn: sqlite3.Connection,
    idx: RetrievalIndex,
    eligible: NDArray[np.bool_],
    expr: str,
    *,
    row_limit: int | None = None,
    all_eligible: bool | None = None,
    elig_list: list[bool] | None = None,
) -> tuple[set[int], bool]:
    """Execute one FTS5 MATCH query and map matching rowids to eligible index positions."""
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


def lookup_u61_postings(
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
    """Cached unicode61 FTS5 postings lookup supporting CJK bigrams and prefix stems."""
    if not term:
        return set()
    cache_key = (term, is_cjk, prefix)
    if (cached := postings_cache.get(cache_key)) is not None:
        return set(cached)
    if row_limit is not None and cache_key in oversized_keys:
        return set()

    if prefix:
        expr = f"{fts_quote(term)}*"
    elif is_cjk and len(term) >= 3:
        expr = fts_quote(" ".join(term[i : i + 2] for i in range(len(term) - 1)))
    elif is_cjk and len(term) == 1:
        expr = f"{fts_quote(term)}*"
    else:
        expr = fts_quote(term)

    out, truncated = match_postings(
        conn, idx, eligible, expr, row_limit=row_limit, all_eligible=all_eligible, elig_list=elig_list
    )
    if truncated:
        oversized_keys.add(cache_key)
        return set()
    postings_cache[cache_key] = set(out)
    return out


def u61_vocab_key(w: str) -> str:
    """Fold Latin-1/Extended-A diacritics (U+00C0..U+024F) to match FTS5 unicode61 remove_diacritics=1."""
    out: list[str] = []
    for ch in w:
        o = ord(ch)
        if 0x00C0 <= o <= 0x024F:
            out.append("".join(c for c in unicodedata.normalize("NFD", ch) if unicodedata.category(c) != "Mn"))
        else:
            out.append(ch)
    return "".join(out)


def fetch_vocab_docs(conn: sqlite3.Connection, terms: Sequence[str], target: dict[str, int]) -> None:
    """Populate `target` with `chunk_vocab` document counts for any uncached `terms`."""
    with CACHE_LOCK:
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
    with CACHE_LOCK:
        for k, v in fetched.items():
            target.setdefault(k, v)
            while len(target) > CACHE_MAX_ENTRIES:
                target.pop(next(iter(target)))


def umlaut_primary_variants(tok: str) -> list[str]:
    """Return `[tok]` plus ASCII-folded umlaut/sharp-s variant when different."""
    variants = [tok]
    folded = tok.translate(_ASCII_NORM_TRANS)
    if folded != tok:
        variants.append(folded)
    return variants


def umlaut_reverse_variant(tok: str) -> str | None:
    """Reverse-fold `ae`/`oe`/`ue` digraphs to `ä`/`ö`/`ü` when changed."""
    rev = tok.replace("ae", "ä").replace("oe", "ö").replace("ue", "ü")
    return rev if rev != tok else None


def _add_anchor(ctx: _AnchorCtx, label: str, postings: set[int]) -> None:
    if not postings:
        return
    if ctx.max_df_filter is not None and len(postings) > ctx.max_df_filter:
        return
    prev = ctx.anchors.get(label)
    if prev is None:
        ctx.anchors[label] = set(postings)
    else:
        prev.update(postings)


def _lookup_anchor_term(
    ctx: _AnchorCtx,
    term: str,
    *,
    is_cjk: bool = False,
    prefix: bool = False,
    row_limit: int | None = None,
) -> set[int]:
    return lookup_u61_postings(
        ctx.conn,
        ctx.idx,
        ctx.eligible,
        term,
        is_cjk=is_cjk,
        prefix=prefix,
        postings_cache=ctx.cache,
        row_limit=row_limit,
        oversized_keys=ctx.oversized,
        all_eligible=ctx.all_eligible,
        elig_list=ctx.elig_list,
    )


def _probe_vocab_doc_counts(ctx: _AnchorCtx, unique_words: list[str]) -> None:
    if ctx.vocab_doc_counts is None or not unique_words:
        return
    probe_terms: list[str] = []
    for raw_w in unique_words:
        cand_vs = umlaut_primary_variants(raw_w)
        if (rev := umlaut_reverse_variant(raw_w)) is not None:
            cand_vs.append(rev)
        for w in cand_vs:
            pkey = (w, False, False)
            if pkey in ctx.cache or pkey in ctx.oversized or not w.isalnum():
                continue
            vk = u61_vocab_key(w)
            with CACHE_LOCK:
                missing_vk = vk not in ctx.vocab_doc_counts
            if missing_vk:
                probe_terms.append(vk)
    fetch_vocab_docs(ctx.conn, probe_terms, ctx.vocab_doc_counts)


def _ub_word_variant(ctx: _AnchorCtx, w: str) -> int:
    pkey = (w, False, False)
    if (cached := ctx.cache.get(pkey)) is not None:
        return len(cached)
    if ctx.word_row_limit is not None and pkey in ctx.oversized:
        return ctx.max_df_filter + 1 if ctx.max_df_filter is not None else ctx.n_eff
    if ctx.vocab_doc_counts is None or not w.isalnum():
        return ctx.n_eff
    vk = u61_vocab_key(w)
    with CACHE_LOCK:
        return ctx.vocab_doc_counts.get(vk, 0)


def _vocab_prunes_variant(ctx: _AnchorCtx, w: str) -> bool:
    if ctx.vocab_doc_counts is None or not w.isalnum():
        return False
    pkey = (w, False, False)
    if pkey in ctx.cache:
        return False
    ub_v = _ub_word_variant(ctx, w)
    if ub_v == 0:
        return True
    if ctx.max_df_filter is not None and ctx.n_eff == ctx.idx.size and ub_v > ctx.max_df_filter:
        ctx.oversized.add(pkey)
        return True
    return False


def _find_best_stem_anchor(
    ctx: _AnchorCtx,
    w: str,
    pref_terms: list[tuple[str, int]],
    min_stem_df: int,
) -> tuple[str, set[int]] | None:
    for cut in range(len(w) - 1, 3, -1):
        pk = u61_vocab_key(w[:cut])
        if sum(d for t, d in pref_terms if t.startswith(pk)) < min_stem_df:
            continue
        s_pts = _lookup_anchor_term(ctx, w[:cut], prefix=True, row_limit=ctx.word_row_limit)
        if len(s_pts) >= min_stem_df and (ctx.max_df_filter is None or len(s_pts) <= ctx.max_df_filter):
            return f"stem:{w[:cut]}", s_pts
    return None


def _find_best_decomp_split(
    ctx: _AnchorCtx,
    w: str,
    pref_terms: list[tuple[str, int]],
    min_split_df: int,
) -> tuple[float, str, set[int], str, set[int]] | None:
    best_split: tuple[float, str, set[int], str, set[int]] | None = None
    for a in range(4, len(w) - 3):
        lk = u61_vocab_key(w[:a])
        if sum(d for t, d in pref_terms if t.startswith(lk)) < min_split_df:
            continue
        rk = u61_vocab_key(w[a:])
        r_cnt_row = ctx.conn.execute(
            "SELECT COALESCE(SUM(doc), 0) FROM main.chunk_vocab WHERE term >= ? AND term < ?",
            (rk, rk + "\U0010ffff"),
        ).fetchone()
        if (int(r_cnt_row[0]) if r_cnt_row is not None else 0) < min_split_df:
            continue
        left_pts = _lookup_anchor_term(ctx, w[:a], prefix=True, row_limit=ctx.word_row_limit)
        if len(left_pts) < min_split_df or (ctx.max_df_filter is not None and len(left_pts) > ctx.max_df_filter):
            continue
        right_pts = _lookup_anchor_term(ctx, w[a:], prefix=True, row_limit=ctx.word_row_limit)
        if len(right_pts) < min_split_df or (ctx.max_df_filter is not None and len(right_pts) > ctx.max_df_filter):
            continue
        score = math.sqrt(float(len(left_pts)) * float(len(right_pts)))
        if best_split is None or score > best_split[0]:
            best_split = (score, w[:a], left_pts, w[a:], right_pts)
    return best_split


def _decomp_word(ctx: _AnchorCtx, w: str, pts: set[int]) -> None:
    if len(w) < 6 or len(pts) >= 2:
        return
    full_df = len(pts)
    if (ctx.max_df_filter is not None and full_df > ctx.max_df_filter) or not w.isalnum():
        return
    pref4 = u61_vocab_key(w[:4])
    pref_rows = ctx.conn.execute(
        "SELECT term, doc FROM main.chunk_vocab WHERE term >= ? AND term < ?",
        (pref4, pref4 + "\U0010ffff"),
    ).fetchall()
    pref_terms = [(str(r[0]), int(r[1])) for r in pref_rows]
    if not pref_terms:
        return

    min_stem_df = 1 if full_df == 0 else max(2, 2 * full_df)
    if (best_stem := _find_best_stem_anchor(ctx, w, pref_terms, min_stem_df)) is not None:
        _add_anchor(ctx, best_stem[0], best_stem[1])

    if len(w) >= 8:
        min_split_df = max(full_df + 1, 1)
        if (best_split := _find_best_decomp_split(ctx, w, pref_terms, min_split_df)) is not None:
            _add_anchor(ctx, f"decomp:{best_split[1]}", best_split[2])
            _add_anchor(ctx, f"decomp:{best_split[3]}", best_split[4])


def _process_one_word_anchors(ctx: _AnchorCtx, raw_w: str, word_postings: dict[str, set[int]]) -> None:
    decomp_candidates: list[tuple[str, set[int]]] = []
    for w in umlaut_primary_variants(raw_w):
        if _vocab_prunes_variant(ctx, w):
            if len(w) >= 6 and (w, False, False) not in ctx.oversized:
                decomp_candidates.append((w, set()))
            continue
        pts = _lookup_anchor_term(ctx, w, row_limit=ctx.word_row_limit)
        if (w, False, False) not in ctx.oversized:
            decomp_candidates.append((w, pts))
        if pts:
            word_postings.setdefault(raw_w, set()).update(pts)
            _add_anchor(ctx, raw_w, pts)

    if not bool(word_postings.get(raw_w)) and (rev := umlaut_reverse_variant(raw_w)) is not None:
        if _vocab_prunes_variant(ctx, rev):
            if len(rev) >= 6 and (rev, False, False) not in ctx.oversized:
                decomp_candidates.append((rev, set()))
        else:
            rev_pts = _lookup_anchor_term(ctx, rev, row_limit=ctx.word_row_limit)
            if rev_pts:
                word_postings.setdefault(raw_w, set()).update(rev_pts)
                _add_anchor(ctx, raw_w, rev_pts)
                if (rev, False, False) not in ctx.oversized:
                    decomp_candidates = [(rev, rev_pts)]

    for w, pts in decomp_candidates:
        _decomp_word(ctx, w, pts)


def _collect_word_anchors(ctx: _AnchorCtx, words: list[str]) -> dict[str, set[int]]:
    unique_words = list(dict.fromkeys(words))
    _probe_vocab_doc_counts(ctx, unique_words)
    ordered_words = (
        sorted(
            unique_words,
            key=lambda rw: (min(_ub_word_variant(ctx, v) for v in umlaut_primary_variants(rw)), -len(rw), rw),
        )
        if ctx.vocab_doc_counts is not None
        else unique_words
    )
    word_postings: dict[str, set[int]] = {}
    for raw_w in ordered_words:
        _process_one_word_anchors(ctx, raw_w, word_postings)
    return word_postings


def _collect_word_bigram_pmi_anchors(
    ctx: _AnchorCtx,
    words: list[str],
    word_postings: dict[str, set[int]],
) -> None:
    if len(words) < 2:
        return
    seen_pairs: set[tuple[str, str]] = set()
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]
        pair = (w1, w2)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        p1 = word_postings.get(w1)
        p2 = word_postings.get(w2)
        if p1 is None or not p1 or p2 is None or not p2:
            continue
        if 2 * len(p1) > ctx.n_eff or 2 * len(p2) > ctx.n_eff:
            continue
        if ctx.max_df_filter is not None and 2 * len(p1) * len(p2) > ctx.n_eff * ctx.max_df_filter:
            continue
        inter = len(p1 & p2)
        if inter == 0 or ctx.n_eff * inter < 2 * len(p1) * len(p2):
            continue
        phrase = f"{w1} {w2}"
        p12 = _lookup_anchor_term(ctx, phrase, row_limit=ctx.word_row_limit) if len(phrase) >= 3 else (p1 & p2)
        if p12 and ctx.n_eff * len(p12) >= 2 * len(p1) * len(p2):
            _add_anchor(ctx, f"pmi:{phrase}", p12)


def _collect_cjk_span_pmi(ctx: _AnchorCtx, run: str, bigrams: dict[str, set[int]]) -> None:
    for span_len in (3, 4):
        for i in range(len(run) - span_len + 1):
            sub = run[i : i + span_len]
            b_left = bigrams.get(sub[:2])
            b_right = bigrams.get(sub[-2:])
            if b_left is None or not b_left or b_right is None or not b_right:
                continue
            s_pts = _lookup_anchor_term(ctx, sub, is_cjk=True)
            if s_pts and ctx.n_eff * len(s_pts) >= 2 * len(b_left) * len(b_right):
                _add_anchor(ctx, f"cjk_pmi:{sub}", s_pts)


def _collect_cjk_run_anchors(ctx: _AnchorCtx, run: str) -> None:
    if len(run) == 1:
        _add_anchor(ctx, run, _lookup_anchor_term(ctx, run, is_cjk=True))
        return
    bigrams: dict[str, set[int]] = {}
    for i in range(len(run) - 1):
        bg = run[i : i + 2]
        b_pts = _lookup_anchor_term(ctx, bg, is_cjk=True)
        if b_pts:
            bigrams[bg] = b_pts
            _add_anchor(ctx, bg, b_pts)
    if len(run) >= 3:
        _collect_cjk_span_pmi(ctx, run, bigrams)


def extract_multilingual_anchors(
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
    words = WORD_TOK_RE.findall(q_cf)
    ctx = _AnchorCtx(
        conn=conn,
        idx=idx,
        eligible=eligible,
        n_eff=n_eff,
        all_eligible=all_eligible,
        elig_list=None if all_eligible else eligible.tolist(),
        max_df_filter=max_df_filter,
        word_row_limit=max_df_filter + 1 if (max_df_filter is not None and all_eligible) else None,
        vocab_doc_counts=idx.vocab_doc_cache if (max_df_filter is not None and bool(words)) else None,
    )
    word_postings = _collect_word_anchors(ctx, words)
    _collect_word_bigram_pmi_anchors(ctx, words, word_postings)
    for run in CJK_RUN_RE.findall(q_cf):
        _collect_cjk_run_anchors(ctx, run)
    return sorted(ctx.anchors.items(), key=lambda kv: (len(kv[1]), kv[0]))


def _match_word_title_keys(
    toks: list[str],
    title_tbl: dict[str, tuple[int, dict[tuple[str, ...], tuple[int, ...]]]],
    seen_tk: set[tuple[str, ...]],
    matched_keys: list[tuple[str, ...]],
) -> None:
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


def _match_cjk_title_keys(
    t_cf: str,
    title_tbl: dict[str, tuple[int, dict[tuple[str, ...], tuple[int, ...]]]],
    seen_tk: set[tuple[str, ...]],
    matched_keys: list[tuple[str, ...]],
) -> None:
    for run in CJK_RUN_RE.findall(t_cf):
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


def _collect_title_candidates(
    matched_keys: list[tuple[str, ...]],
    idx: RetrievalIndex,
    conn: sqlite3.Connection,
    eligible: NDArray[np.bool_],
    elig_list: list[bool] | None,
    all_eligible: bool,
    d_bridge: int,
) -> list[tuple[tuple[str, ...], tuple[int, ...]]]:
    title_tbl = idx.title_by_first_word
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
            if CJK_RUN_RE.fullmatch(w) is not None:
                continue
            vk = u61_vocab_key(w)
            with CACHE_LOCK:
                missing_vk = vk not in idx.vocab_doc_cache
            if missing_vk:
                probe_terms.append(vk)
    if filtered:
        fetch_vocab_docs(conn, probe_terms, idx.vocab_doc_cache)
    return filtered


def _score_title_candidates(
    filtered: list[tuple[tuple[str, ...], tuple[int, ...]]],
    idx: RetrievalIndex,
    n_eff: int,
    d_query: int,
    d_bridge: int,
) -> list[tuple[str, tuple[int, ...], float, int]]:
    out: list[tuple[str, tuple[int, ...], float, int]] = []
    for tk, cand_pos in filtered:
        df_title = len(cand_pos)
        word_ubs: list[int] = []
        is_cjk_only = True
        with CACHE_LOCK:
            for w in tk:
                if CJK_RUN_RE.fullmatch(w) is not None:
                    continue
                is_cjk_only = False
                word_ubs.append(idx.vocab_doc_cache.get(u61_vocab_key(w), n_eff))
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


def extract_title_references(
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
    """Extract and IDF-score document title references matched in query text."""
    title_tbl = idx.title_by_first_word
    if not title_tbl or n_eff <= 0 or not text.strip():
        return []
    t_cf = text.casefold()
    toks = WORD_TOK_RE.findall(t_cf)
    matched_keys: list[tuple[str, ...]] = []
    seen_tk: set[tuple[str, ...]] = set()
    _match_word_title_keys(toks, title_tbl, seen_tk, matched_keys)
    _match_cjk_title_keys(t_cf, title_tbl, seen_tk, matched_keys)
    if not matched_keys:
        return []
    filtered = _collect_title_candidates(matched_keys, idx, conn, eligible, elig_list, all_eligible, d_bridge)
    if not filtered:
        return []
    return _score_title_candidates(filtered, idx, n_eff, d_query, d_bridge)


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
    if isinstance(top_k, bool) or top_k <= 0 or idx.size == 0 or not query_text.strip():
        return []
    if conn is None:
        raise YawRuntimeError("CONSISTENCY_ERROR", "lexical_search requires an active SQLite connection.")
    elig = eligible_mask(idx, path_glob=path_glob, masked_rel_paths=masked_rel_paths)
    n_eff = int(np.sum(elig))
    if n_eff == 0:
        return []

    with snapshot_tx(conn, require_fts=True):
        anchors = extract_multilingual_anchors(conn, idx, elig, query_text, max_df_filter=None)
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
