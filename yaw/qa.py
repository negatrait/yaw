from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import dspy
import numpy as np
from dspy.lm15 import (
    AccessPolicy,
    EndpointSupport,
    ModelSupport,
    OpenAIChatCompat,
    ProviderDefinition,
    register_provider,
)
from dspy.utils.exceptions import AdapterParseError

from .core import EmbedFn, HydratedPassage, QAResult, YawConfig, YawRuntimeError, format_query_for_embedding
from .retrieval import VectorSearch, hydrate_passage, hydrate_ranked_passages, lexical_search, search_vectors

_MAX_TOP_K = 25
_NO_MATCHES = "No matching passages found."
_NO_ISLAND = (1, 1)  # ChunkPointer default: the chunk carries no AST island.
_PASSAGE_HEAD_RE = re.compile(r"^\[chunk_id=(\d+)\]\s+\([^)\n]+\)\s+score=([-+]?\d+(?:\.\d+)?):$", re.MULTILINE)
_CHUNK_ID_RE = re.compile(r"\[chunk_id=(\d+)\]")
_QUERY_DESC = {"query": "Non-empty search query.", "top_k": f"Max passages to return (1..{_MAX_TOP_K})."}
_CHUNK_DESC = "Chunk ID from a prior [chunk_id=<int>] header."


class QASignature(dspy.Signature):
    """Answer strictly from retrieved corpus passages using inline [chunk_id=<int>] citations,
    or reply 'Insufficient references in corpus' when evidence is lacking."""

    question: str = dspy.InputField(desc="User question to answer from indexed corpus.")
    answer: str = dspy.OutputField(desc="Grounded answer citing [chunk_id=<int>] or refusal.")


def create_lm(cfg: YawConfig, *, temperature: float = 1.0) -> dspy.LM:
    """Declare the required native tool protocol for the configured compatible endpoint."""
    model = cfg.qa_model if "/" in cfg.qa_model else f"{cfg.provider}/{cfg.qa_model}"
    if model.startswith("openai/"):
        register_provider(
            ProviderDefinition.chat(
                AccessPolicy(
                    provider="yaw-openai",
                    base_url=cfg.base_url,
                    supports=EndpointSupport(complete=True),
                    auth_modes=("bearer",),
                ),
                compat=OpenAIChatCompat(max_tokens_field="max_tokens"),
            ),
            supports=ModelSupport(function_calling=True),
            replace=True,
        )
        model = "yaw-openai/" + model.removeprefix("openai/")
    return dspy.LM(
        model,
        model_type="chat",
        engine="lm15",
        temperature=temperature,
        max_tokens=cfg.max_output_tokens,
        cache=False,
        num_retries=3,
        api_base=cfg.base_url,
        api_key=cfg.api_key or "yaw",
    )


def _field(obj: Any, key: str, default: Any = None) -> Any:
    return obj.get(key, default) if isinstance(obj, Mapping) else getattr(obj, key, default)


def _valid_int(val: Any, lo: int = 1, hi: int | None = None) -> bool:
    return isinstance(val, int) and not isinstance(val, bool) and val >= lo and (hi is None or val <= hi)


def _format_passages(passages: Sequence[HydratedPassage]) -> str:
    if not passages:
        return _NO_MATCHES
    return "\n\n".join(
        f"[chunk_id={p.pointer.chunk_id}] ({p.pointer.rel_posix_path}:{p.pointer.start_line}-{p.pointer.end_line})"
        f" score={p.score:.3f}:\n{f'[STALE ON DISK] {p.pointer.rel_posix_path}' if p.stale_on_disk else p.text}"
        for p in passages
    )


def _require_query(query: str) -> str:
    if not isinstance(query, str) or not (q := query.strip()):
        raise ValueError("query must be a non-empty string.")
    return q


def _require_top_k(top_k: int) -> int:
    if not _valid_int(top_k, 1, _MAX_TOP_K):
        raise ValueError(f"top_k must be an integer between 1 and {_MAX_TOP_K}.")
    return top_k


def _require_chunk_pos(vs: VectorSearch, chunk_id: int) -> int:
    if isinstance(chunk_id, bool) or not isinstance(chunk_id, int):
        raise ValueError("chunk_id must be an integer.")
    if (pos := vs.id_to_pos.get(chunk_id)) is None:
        raise ValueError(f"Unknown chunk_id={chunk_id}.")
    return pos


def _hydrate_ids(
    vs: VectorSearch, ranked: Sequence[tuple[int, float]], corpus_root: Path
) -> tuple[HydratedPassage, ...]:
    """Hydrate exactly the requested chunk ids from disk, in order and without span deduplication."""
    return tuple(hydrate_passage(vs.pointers[vs.id_to_pos[cid]], corpus_root, score=score) for cid, score in ranked)


def make_qa_tools(
    vs: VectorSearch,
    corpus_root: Path,
    embed_fn: EmbedFn,
    conn: sqlite3.Connection,
    *,
    task_mode: str = "distinct-titled",
) -> list[dspy.Tool]:
    """Build closure-bound retrieval tools with explicit DSPy schemas for dspy.ReAct."""

    def search_passages(query: str, top_k: int = 8) -> str:
        """Search corpus passages using hybrid vector and lexical ranking."""
        q, k = _require_query(query), _require_top_k(top_k)
        if vs.size == 0:
            return _NO_MATCHES
        raw_emb = np.asarray(embed_fn([format_query_for_embedding(q, task_mode)]), dtype=np.float32)
        q_vec = raw_emb[0] if raw_emb.ndim == 2 else raw_emb.reshape(-1)
        ranked = search_vectors(vs, q_vec, query_text=q, conn=conn, top_k=max(k * 4, 50))
        return _format_passages(hydrate_ranked_passages(vs, ranked, corpus_root, top_k=k))

    def lexical_grep(query: str, top_k: int = 10) -> str:
        """Run zero-embedder FTS5 keyword grep over indexed passages."""
        q, k = _require_query(query), _require_top_k(top_k)
        if vs.size == 0:
            return _NO_MATCHES
        # Over-fetch pool_k = max(k * 4, 50) to prevent sub-chunk starvation during MaxP span dedup.
        ranked = lexical_search(vs, q, conn, top_k=max(k * 4, 50))
        return _format_passages(hydrate_ranked_passages(vs, ranked, corpus_root, top_k=k))

    def expand_neighbors(chunk_id: int, top_k: int = 6) -> str:
        """Expand isotropic, shared-entity, and same-AST-island neighbor passages for chunk_id."""
        idx, k = _require_chunk_pos(vs, chunk_id), _require_top_k(top_k)
        if vs.size <= 1:
            return _NO_MATCHES
        # Embeddings are unit-normalized, so the dot product is the cosine; entity_to_chunk is (entities x chunks).
        scores = (vs.embeddings @ vs.embeddings[idx]).astype(np.float64)
        if (shared := vs.entity_to_chunk[:, idx].nonzero()[0]).size:
            overlap = np.asarray(vs.entity_to_chunk[shared].astype(bool).sum(axis=0)).ravel()
            scores += 0.35 * overlap
        ptr = vs.pointers[idx]
        island = (ptr.rel_posix_path, ptr.ast_island_start, ptr.ast_island_end)
        if (ptr.ast_island_start, ptr.ast_island_end) != _NO_ISLAND:
            siblings = [
                j
                for j, p in enumerate(vs.pointers)
                if (p.rel_posix_path, p.ast_island_start, p.ast_island_end) == island
            ]
            scores[siblings] += 0.50
        scores[idx] = -np.inf
        pool_k = min(max(k * 4, 50), vs.size - 1)
        cutoff = float(np.partition(scores, -pool_k)[-pool_k])
        ranked = sorted(
            ((vs.chunk_ids[int(j)], float(scores[int(j)])) for j in np.flatnonzero(scores >= cutoff) if int(j) != idx),
            key=lambda item: (-item[1], item[0]),
        )[:pool_k]
        return _format_passages(hydrate_ranked_passages(vs, ranked, corpus_root, top_k=k))

    def read_chunk(chunk_id: int) -> str:
        """Read the exact hydrated passage for a single chunk_id."""
        _require_chunk_pos(vs, chunk_id)
        return _format_passages(_hydrate_ids(vs, [(chunk_id, 1.0)], corpus_root))

    return [
        dspy.Tool(search_passages, arg_desc=_QUERY_DESC),
        dspy.Tool(lexical_grep, arg_desc=_QUERY_DESC),
        dspy.Tool(expand_neighbors, arg_desc={"chunk_id": _CHUNK_DESC, "top_k": _QUERY_DESC["top_k"]}),
        dspy.Tool(read_chunk, arg_desc={"chunk_id": _CHUNK_DESC}),
    ]


def iter_tool_observations(history_msgs: Sequence[Any]) -> Iterator[str]:
    """Yield successful text results from live or serialized DSPy ToolCalls."""
    for msg in history_msgs:
        if (tc := _field(msg, "tool_calls")) is None:
            continue
        calls = _field(tc, "tool_calls") or ()
        tcr = _field(tc, "tool_call_results")
        results = tcr if isinstance(tcr, list | tuple) else (_field(tcr, "tool_call_results") or ())
        by_id = {cid: _field(c, "name", "") for c in calls if (cid := _field(c, "id")) is not None}
        for i, res in enumerate(results):
            pos_name = _field(calls[i], "name", "") if i < len(calls) else ""
            name = _field(res, "name") or by_id.get(_field(res, "tool_call_id")) or pos_name
            if name == "submit" or _field(res, "is_error") or not isinstance(val := _field(res, "value"), str):
                continue
            yield val


def _extract_seen_chunks(history_msgs: Sequence[Any]) -> dict[int, tuple[float, int]]:
    seen: dict[int, tuple[float, int]] = {}
    for value in iter_tool_observations(history_msgs):
        for m in _PASSAGE_HEAD_RE.finditer(value):
            cid, score = int(m.group(1)), float(m.group(2))
            prev = seen.get(cid, (score, len(seen)))
            seen[cid] = (max(score, prev[0]), prev[1])
    return seen


def build_qa_agent(
    tools: Sequence[dspy.Tool | Callable[..., Any]], *, max_iters: int = 6, artifact_path: Path | None = None
) -> dspy.ReAct:
    if not _valid_int(max_iters):
        raise YawRuntimeError("INVALID_INPUT", "max_iters must be >= 1.")
    agent = dspy.ReAct(QASignature, tools=list(tools), max_iters=max_iters)
    if artifact_path is not None and artifact_path.is_file():
        agent.load(str(artifact_path))
        agent.set_lm(None)
    agent.react.update_config(tool_choice="required")
    return agent


def ask_question(
    question: str, *, vs: VectorSearch, corpus_root: Path, agent: dspy.Module, max_citations: int = 5
) -> QAResult:
    if not isinstance(question, str) or not (q := question.strip()):
        raise YawRuntimeError("INVALID_INPUT", "question must be a non-empty string.")
    if not _valid_int(max_citations):
        raise YawRuntimeError("INVALID_INPUT", "max_citations must be >= 1.")
    if vs.size == 0:
        raise YawRuntimeError("EMPTY_INDEX", "Index contains 0 chunks.")

    try:
        pred = agent(question=q)
    except AdapterParseError as exc:
        raise YawRuntimeError("MODEL_ERROR", "QA model returned an invalid tool response.") from exc
    if not isinstance(raw_ans := _field(pred, "answer"), str) or not (answer := raw_ans.strip()):
        raise YawRuntimeError("MODEL_ERROR", "QA agent produced an empty answer.")

    # Verify inline [chunk_id] citations against tool-seen IDs and disk SHA-256 hashes.
    history_msgs = _field(_field(pred, "history"), "messages") or ()
    seen_scores = _extract_seen_chunks(history_msgs)
    inline_ids = list(dict.fromkeys(int(m.group(1)) for m in _CHUNK_ID_RE.finditer(answer)))
    for cid in inline_ids:
        if cid not in seen_scores or cid not in vs.id_to_pos:
            raise YawRuntimeError("MODEL_ERROR", f"Answer cites ungrounded chunk_id={cid}.")

    chosen_ids = (
        inline_ids
        or [cid for cid, _ in sorted(seen_scores.items(), key=lambda x: (-x[1][0], x[1][1])) if cid in vs.id_to_pos]
    )[:max_citations]
    citations = _hydrate_ids(vs, [(cid, seen_scores[cid][0]) for cid in chosen_ids], corpus_root)
    if any(hp.stale_on_disk for hp in citations):
        raise YawRuntimeError("CONSISTENCY_ERROR", "Cited passage failed disk SHA-256 verification.")

    return QAResult(
        question=q,
        answer=answer,
        citations=tuple(hp.pointer.span_label for hp in citations),
        passages=citations,
        turns=len(history_msgs),
    )
