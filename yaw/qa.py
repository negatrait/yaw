from __future__ import annotations

import contextlib
import re
import sqlite3
from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
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
from dspy.utils.exceptions import AdapterParseError, ContextWindowExceededError

from .core import EmbedFn, HydratedPassage, QAResult, YawConfig, YawRuntimeError, format_query_for_embedding
from .retrieval import (
    RetrievalIndex,
    build_recall_payload,
    expand_citation_neighbors,
    format_recall_text,
    hydrate_passage,
    hydrate_ranked_passages,
    lexical_search,
    open_readonly_db,
    search_index,
)

_MAX_TOP_K = 25
_PASSAGE_HEAD_RE = re.compile(r"^\[chunk_id=(\d+)\]\s+\([^)\n]+\)\s+score=([-+]?\d+(?:\.\d+)?):$", re.MULTILINE)
_CITATION_BLOCK_RE = re.compile(r"\[(?:chunk_id=\d+(?:\s*[,;]\s*(?:chunk_id=)?\d+)*)\]")
_CHUNK_NUM_RE = re.compile(r"\d+")
_QUERY_DESC = {"query": "Non-empty search query.", "top_k": f"Max passages to return (1..{_MAX_TOP_K})."}
_CHUNK_DESC = "Chunk ID from a prior [chunk_id=<int>] header."


ABSTAIN_REPLY = "Insufficient references in corpus"


class QASignature(dspy.Signature):
    """Verify multi-hop facts via tools and answer with ONLY the minimal canonical answer span grounded in retrieved
    passages plus inline [chunk_id=<int>] citations — 'yes'/'no' for polar questions, bare entity/title/date/number/
    single primary term without qualifiers, lists, or explanatory sentences — or 'Insufficient references in corpus'."""

    question: str = dspy.InputField(desc="Question to answer strictly from the indexed corpus.")
    answer: str = dspy.OutputField(desc="Minimal bare answer span with [chunk_id=<int>] (no prose), or refusal.")


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


def _require_query(query: str) -> str:
    if not isinstance(query, str) or not (q := query.strip()):
        raise ValueError("query must be a non-empty string.")
    return q


def _require_top_k(top_k: int) -> int:
    if not _valid_int(top_k, 1, _MAX_TOP_K):
        raise ValueError(f"top_k must be an integer between 1 and {_MAX_TOP_K}.")
    return top_k


def _require_chunk_pos(index: RetrievalIndex, chunk_id: int) -> int:
    if isinstance(chunk_id, bool) or not isinstance(chunk_id, int):
        raise ValueError("chunk_id must be an integer.")
    if (pos := index.id_to_pos.get(chunk_id)) is None:
        raise ValueError(f"Unknown chunk_id={chunk_id}.")
    return pos


def _hydrate_ids(
    index: RetrievalIndex, ranked: Sequence[tuple[int, float]], corpus_root: Path
) -> tuple[HydratedPassage, ...]:
    """Hydrate exactly the requested chunk ids from disk, in order and without span deduplication."""
    return tuple(
        hydrate_passage(index.pointers[index.id_to_pos[cid]], corpus_root, score=score) for cid, score in ranked
    )


def make_qa_tools(
    index: RetrievalIndex,
    corpus_root: Path,
    embed_fn: EmbedFn,
    conn: sqlite3.Connection | Path,
    *,
    task_mode: str = "distinct-titled",
) -> list[dspy.Tool]:
    """Build closure-bound retrieval tools with explicit DSPy schemas for dspy.ReAct."""

    @contextlib.contextmanager
    def _tool_conn() -> Generator[sqlite3.Connection]:
        if isinstance(conn, Path):
            with contextlib.closing(open_readonly_db(conn)) as local_conn:
                yield local_conn
        else:
            yield conn

    def search_passages(query: str, top_k: int = 8) -> str:
        """Search corpus passages using hybrid 1-bit RaBitQ, lexical, and citation diffusion ranking."""
        q, k = _require_query(query), _require_top_k(top_k)
        if index.size == 0:
            return format_recall_text(build_recall_payload(()))
        raw_emb = np.asarray(embed_fn([format_query_for_embedding(q, task_mode)]), dtype=np.float32)
        q_vec = raw_emb[0] if raw_emb.ndim == 2 else raw_emb.reshape(-1)
        with _tool_conn() as db_conn:
            res = search_index(index, q_vec, query_text=q, conn=db_conn, top_k=k, pool_k=max(k * 4, 50))
        passages = hydrate_ranked_passages(index, res.ranked, corpus_root, top_k=k)
        return format_recall_text(build_recall_payload(passages))

    def lexical_grep(query: str, top_k: int = 10) -> str:
        """Run zero-embedder FTS5 keyword grep over indexed passages."""
        q, k = _require_query(query), _require_top_k(top_k)
        if index.size == 0:
            return format_recall_text(build_recall_payload(()))
        # Over-fetch pool_k = max(k * 4, 50) to prevent sub-chunk starvation during MaxP span dedup.
        with _tool_conn() as db_conn:
            ranked = lexical_search(index, q, db_conn, top_k=max(k * 4, 50))
        return format_recall_text(build_recall_payload(hydrate_ranked_passages(index, ranked, corpus_root, top_k=k)))

    def expand_neighbors(chunk_id: int, top_k: int = 6) -> str:
        """Expand isotropic, lexical, citation-diffusion, and same-AST-island neighbor passages for chunk_id."""
        _require_chunk_pos(index, chunk_id)
        k = _require_top_k(top_k)
        if index.size <= 1:
            return format_recall_text(build_recall_payload(()))
        with _tool_conn() as db_conn:
            ranked = expand_citation_neighbors(
                index, chunk_id, corpus_root=corpus_root, conn=db_conn, top_k=max(k * 4, 50)
            )
        return format_recall_text(build_recall_payload(hydrate_ranked_passages(index, ranked, corpus_root, top_k=k)))

    def read_chunk(chunk_id: int) -> str:
        """Read the exact hydrated passage for a single chunk_id."""
        _require_chunk_pos(index, chunk_id)
        return format_recall_text(build_recall_payload(_hydrate_ids(index, [(chunk_id, 1.0)], corpus_root)))

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
        results = tcr if isinstance(tcr, list | tuple) else _field(tcr, "tool_call_results")
        if not results:
            continue
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


def extract_inline_chunk_ids(text: str) -> list[int]:
    """Return deduplicated `[chunk_id=<int>]` identifiers in order of appearance."""
    ids = [int(n) for m in _CITATION_BLOCK_RE.finditer(text) for n in _CHUNK_NUM_RE.findall(m.group(0))]
    return list(dict.fromkeys(ids))


def strip_inline_citations(text: str) -> str:
    """Remove inline `[chunk_id=<int>]` markers from an answer string."""
    return re.sub(r"\s+([.,;:!?])", r"\1", " ".join(_CITATION_BLOCK_RE.sub("", text).split()))


def collect_tool_history(pred: Any) -> tuple[list[str], int, dict[int, tuple[float, int]]]:
    """Extract tool observations, non-submit call count, and seen chunk scores from `pred.history`."""
    history_msgs = _field(_field(pred, "history"), "messages") or ()
    obs_blocks = list(iter_tool_observations(history_msgs))
    tool_calls = 0
    for msg in history_msgs:
        if (tc := _field(msg, "tool_calls")) is None:
            continue
        for call in _field(tc, "tool_calls") or ():
            if (name := _field(call, "name", "")) and name != "submit":
                tool_calls += 1
    seen = _extract_seen_chunks(history_msgs)
    return obs_blocks, tool_calls, seen


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
    question: str, *, index: RetrievalIndex, corpus_root: Path, agent: dspy.Module, max_citations: int = 5
) -> QAResult:
    if not isinstance(question, str) or not (q := question.strip()):
        raise YawRuntimeError("INVALID_INPUT", "question must be a non-empty string.")
    if not _valid_int(max_citations):
        raise YawRuntimeError("INVALID_INPUT", "max_citations must be >= 1.")
    if index.size == 0:
        raise YawRuntimeError("EMPTY_INDEX", "Index contains 0 chunks.")

    try:
        pred = agent(question=q)
    except (AdapterParseError, ContextWindowExceededError, ValueError) as exc:
        raise YawRuntimeError("MODEL_ERROR", f"QA model returned an invalid tool response: {exc}") from exc
    termination_reason = getattr(pred, "termination_reason", None)
    if termination_reason is not None and termination_reason not in ("submit", "forced_submit"):
        raise YawRuntimeError(
            "MODEL_ERROR",
            f"QA agent terminated without submit (termination_reason={termination_reason!r}).",
        )
    if not isinstance(raw_ans := _field(pred, "answer"), str) or not (answer := raw_ans.strip()):
        raise YawRuntimeError("MODEL_ERROR", "QA agent produced an empty answer.")

    # Verify inline [chunk_id] citations against tool-seen IDs and disk SHA-256 hashes.
    history_msgs = _field(_field(pred, "history"), "messages") or ()
    seen_scores = _extract_seen_chunks(history_msgs)
    inline_ids = extract_inline_chunk_ids(answer)
    for cid in inline_ids:
        if cid not in seen_scores or cid not in index.id_to_pos:
            raise YawRuntimeError("MODEL_ERROR", f"Answer cites ungrounded chunk_id={cid}.")

    chosen_ids = (
        inline_ids
        or [cid for cid, _ in sorted(seen_scores.items(), key=lambda x: (-x[1][0], x[1][1])) if cid in index.id_to_pos]
    )[:max_citations]
    citations = _hydrate_ids(index, [(cid, seen_scores[cid][0]) for cid in chosen_ids], corpus_root)
    if any(hp.stale_on_disk for hp in citations):
        raise YawRuntimeError("CONSISTENCY_ERROR", "Cited passage failed disk SHA-256 verification.")

    return QAResult(
        question=q,
        answer=answer,
        citations=tuple(hp.pointer.span_label for hp in citations),
        passages=citations,
        turns=len(history_msgs),
    )
