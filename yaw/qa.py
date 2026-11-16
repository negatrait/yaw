"""Bare dspy.ReAct multi-hop QA tools, agent builder, and grounded answer verification."""

from __future__ import annotations

import re
import sqlite3
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import dspy
import numpy as np
from dspy.utils.exceptions import AdapterParseError, ContextWindowExceededError
from numpy.typing import NDArray

from yaw.core import HydratedPassage, QAResult, YawRuntimeError, format_query_for_embedding
from yaw.retrieval import (
    VectorSearch,
    hydrate_passage,
    hydrate_ranked_passages,
    query_external_hints,
    search_vectors,
)

_NO_MATCHES = "No matching passages."
_OBS_SCORE_RE = re.compile(r"\[chunk_id=(\d+)\]\s+\([^\n]+:\d+-\d+\)\s+score=([-+]?\d+(?:\.\d+)?):")
_CHUNK_ID_RE = re.compile(r"\[chunk_id=(\d+)\]")


class EmbedFn(Protocol):
    """Callable encoding text strings into unit-L2-normalized float32 embeddings."""

    def __call__(self, texts: list[str], /) -> NDArray[np.float32]: ...


class AnswerQuestion(dspy.Signature):
    """Deduce a definitive closed-world answer grounded in tool-retrieved corpus passages.
    Use search_passages, lexical_grep, expand_neighbors, and read_chunk to verify multi-hop facts.
    Cite supporting passage IDs inline as [chunk_id=<int>], or reply 'Insufficient references in corpus'.
    """

    question: str = dspy.InputField(desc="Question to answer")
    answer: str = dspy.OutputField(desc="Definitive grounded answer citing [chunk_id=<int>] or refusal")


def _format_passage(hp: HydratedPassage) -> str:
    ptr = hp.pointer
    body = f"[STALE ON DISK] {ptr.rel_posix_path}" if hp.stale_on_disk else hp.text
    head = f"[chunk_id={ptr.chunk_id}] ({ptr.rel_posix_path}:{ptr.start_line}-{ptr.end_line}) score={hp.score:.3f}:"
    return f"{head}\n{body}"


def _format_passages(passages: Sequence[HydratedPassage]) -> str:
    return "\n\n".join(_format_passage(hp) for hp in passages) if passages else _NO_MATCHES


def _clamp_top_k(raw: Any, default: int) -> int:
    if isinstance(raw, bool):
        return default
    try:
        return max(1, min(10, int(raw)))
    except TypeError, ValueError:
        return default


def _resolve_chunk_pos(vs: VectorSearch, chunk_id: Any) -> int | None:
    if isinstance(chunk_id, bool):
        return None
    try:
        cid = int(chunk_id)
    except TypeError, ValueError:
        return None
    return vs.id_to_pos.get(cid)


def make_qa_tools(
    vs: VectorSearch,
    corpus_root: Path,
    embed_fn: EmbedFn,
    conn: sqlite3.Connection,
    *,
    task_mode: str = "distinct-titled",
) -> list[Callable[..., Any]]:
    """Build the 4 read-only passage exploration tools bound to an immutable VectorSearch."""
    format_query_for_embedding("probe", task_mode)

    def search_passages(query: str, top_k: int = 8) -> str:
        """Search corpus passages using hybrid vector and lexical ranking."""
        q = query.strip() if isinstance(query, str) else ""
        if not q or vs.size == 0:
            return _NO_MATCHES
        k = _clamp_top_k(top_k, 8)
        raw_emb = np.asarray(embed_fn([format_query_for_embedding(q, task_mode)]), dtype=np.float32)
        q_vec = raw_emb[0] if raw_emb.ndim == 2 else raw_emb.reshape(-1)
        fts_scores, vec_scores = query_external_hints(conn, q, q_vec)
        trace = search_vectors(
            vs,
            q_vec,
            q,
            top_k=max(k * 4, 50),
            fts_scores=fts_scores,
            vec_scores=vec_scores,
            params=vs.params,
        )
        return _format_passages(hydrate_ranked_passages(vs, trace.ranked_chunks, corpus_root, top_k=k, max_chars=1200))

    def lexical_grep(query: str, top_k: int = 10) -> str:
        """Run zero-embedder FTS5 keyword grep over indexed passages."""
        q = query.strip() if isinstance(query, str) else ""
        if not q or vs.size == 0:
            return _NO_MATCHES
        k = _clamp_top_k(top_k, 10)
        fts_scores, _ = query_external_hints(conn, q, np.zeros((0,), dtype=np.float32), fts_k=max(k * 2, 20), vec_k=0)
        ranked = sorted(fts_scores.items(), key=lambda item: (-item[1], item[0]))
        return _format_passages(hydrate_ranked_passages(vs, ranked, corpus_root, top_k=k, max_chars=1200))

    def expand_neighbors(chunk_id: int, top_k: int = 6) -> str:
        """Expand isotropic, shared-entity, and AST-sibling neighbor passages for chunk_id."""
        idx = _resolve_chunk_pos(vs, chunk_id)
        if idx is None:
            return f"[INVALID chunk_id={chunk_id}]"
        if vs.size <= 1:
            return _NO_MATCHES
        k = _clamp_top_k(top_k, 6)
        sims = vs.u_chunk_iso @ vs.u_chunk_iso[idx]
        src_ents = {e.lower() for e in vs.chunk_entities[idx][0]}
        sibs = {int(s) for s in vs.ast_siblings[idx]}
        scored = [
            (
                vs.chunk_ids[j],
                float(sims[j])
                + 0.35 * len(src_ents & {e.lower() for e in vs.chunk_entities[j][0]})
                + (0.50 if j in sibs else 0.0),
            )
            for j in range(vs.size)
            if j != idx
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return _format_passages(
            hydrate_ranked_passages(vs, scored[: max(k * 4, 50)], corpus_root, top_k=k, max_chars=1200),
        )

    def read_chunk(chunk_id: int, include_siblings: bool = False) -> str:
        """Read a passage by chunk_id, optionally expanding its enclosing AST section."""
        idx = _resolve_chunk_pos(vs, chunk_id)
        if idx is None:
            return f"[INVALID chunk_id={chunk_id}]"
        hp = hydrate_passage(
            corpus_root,
            vs.pointers[idx],
            score=1.0,
            max_chars=2400,
            include_island=bool(include_siblings),
        )
        return _format_passage(hp)

    return [search_passages, lexical_grep, expand_neighbors, read_chunk]


def build_qa_agent(tools: list[Callable[..., Any]], max_iters: int = 6) -> dspy.ReAct:
    """Return a bare dspy.ReAct agent configured with AnswerQuestion and exploration tools."""
    return dspy.ReAct(AnswerQuestion, tools=tools, max_iters=max_iters)


def _extract_seen_chunks(history_msgs: Sequence[Any]) -> dict[int, tuple[float, int]]:
    """Extract {chunk_id: (best_score, first_order)} strictly from non-submit tool outputs."""
    seen: dict[int, tuple[float, int]] = {}
    order = 0
    for msg in history_msgs:
        if not isinstance(msg, Mapping) or (tc := msg.get("tool_calls")) is None:
            continue
        calls = getattr(tc, "tool_calls", None) or ()
        results = getattr(getattr(tc, "tool_call_results", None), "tool_call_results", None) or ()
        for call, res in zip(calls, results, strict=False):
            call_name = call.get("name", "") if isinstance(call, Mapping) else getattr(call, "name", "")
            if call_name == "submit":
                continue
            is_error = bool(res.get("is_error")) if isinstance(res, Mapping) else bool(getattr(res, "is_error", False))
            val = res.get("value") if isinstance(res, Mapping) else getattr(res, "value", None)
            if is_error or not isinstance(val, str):
                continue
            scores_here = {int(m.group(1)): float(m.group(2)) for m in _OBS_SCORE_RE.finditer(val)}
            for match in _CHUNK_ID_RE.finditer(val):
                cid = int(match.group(1))
                score = scores_here.get(cid, 0.0)
                if cid not in seen:
                    seen[cid] = (score, order)
                    order += 1
                elif score > seen[cid][0]:
                    seen[cid] = (score, seen[cid][1])
    return seen


def ask_question(
    agent: dspy.ReAct,
    question: str,
    *,
    corpus_root: Path,
    vs: VectorSearch,
    lm: dspy.LM,
    lm_sem: threading.Semaphore,
    max_citations: int = 5,
) -> QAResult:
    """Execute bounded ReAct QA under lm_sem and verify cited chunks against disk."""
    if not isinstance(question, str) or not (q := question.strip()):
        raise YawRuntimeError("INVALID_INPUT", "Question must be a non-empty string.")
    if isinstance(max_citations, bool) or not isinstance(max_citations, int) or max_citations < 1:
        raise YawRuntimeError("INVALID_INPUT", "max_citations must be a positive integer.")
    if vs.size == 0:
        raise YawRuntimeError("EMPTY_INDEX", "Cannot answer question on an empty corpus index.")

    with lm_sem, dspy.context(lm=lm):
        try:
            pred = agent(question=q)
        except (AdapterParseError, ContextWindowExceededError, ValueError) as exc:
            raise YawRuntimeError("MODEL_ERROR", f"ReAct QA failed: {exc}") from exc

    term = str(getattr(pred, "termination_reason", "") or "")
    if term not in ("submit", "forced_submit"):
        raise YawRuntimeError("MODEL_ERROR", f"Unexpected ReAct termination_reason: {term!r}")
    raw_answer = getattr(pred, "answer", None)
    if not isinstance(raw_answer, str) or not (answer := raw_answer.strip()):
        raise YawRuntimeError("MODEL_ERROR", "ReAct produced an empty answer.")
    history_msgs = getattr(getattr(pred, "history", None), "messages", ()) or ()
    if not history_msgs:
        raise YawRuntimeError("MODEL_ERROR", "ReAct returned an empty interaction history.")

    seen_scores = _extract_seen_chunks(history_msgs)
    inline_ids = list(dict.fromkeys(int(m.group(1)) for m in _CHUNK_ID_RE.finditer(answer)))
    for cid in inline_ids:
        if cid not in seen_scores or cid not in vs.id_to_pos:
            raise YawRuntimeError("MODEL_ERROR", f"Answer cites ungrounded chunk_id={cid}.")

    if inline_ids:
        chosen_ids = inline_ids[:max_citations]
    else:
        ordered_seen = sorted(seen_scores.items(), key=lambda item: (-item[1][0], item[1][1]))
        chosen_ids = [cid for cid, _ in ordered_seen if cid in vs.id_to_pos][:max_citations]

    citations: list[HydratedPassage] = []
    for cid in chosen_ids:
        hp = hydrate_passage(corpus_root, vs.pointers[vs.id_to_pos[cid]], score=seen_scores[cid][0], max_chars=4000)
        if hp.stale_on_disk:
            raise YawRuntimeError("CONSISTENCY_ERROR", f"Cited chunk {cid} is stale or missing on disk.")
        citations.append(hp)

    return QAResult(answer=answer, citations=tuple(citations), termination_reason=term, turns=len(history_msgs))
