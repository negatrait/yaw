"""Canonical sparse recall slip serialization, sanitization, and terse ASCII rendering."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, TypeIs

from yaw.core import HydratedPassage

from .types import RetrievalIndex, SearchResult

NO_MATCHES_TEXT = "No matching passages found."
_CONTROL_WS_RE = re.compile(r"[\x00-\x1f\x7f-\x9f\s]+")


def sanitize_slip_label(raw: str) -> str:
    """Strip double quotes, collapse C0/C1 control chars and whitespace, and cap label at 64 chars."""
    cleaned = _CONTROL_WS_RE.sub(" ", str(raw).replace('"', ""))
    return cleaned.strip()[:64].rstrip()


def _is_finite_number(val: Any) -> TypeIs[int | float]:
    return isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val)


def _serialize_result_item(rank: int, hp: HydratedPassage) -> dict[str, Any]:
    ptr = hp.pointer
    digest = hashlib.sha256(hp.text.encode("utf-8")).hexdigest()
    source: dict[str, Any] = {
        "rel_posix_path": ptr.rel_posix_path,
        "start_line": int(ptr.start_line),
        "end_line": int(ptr.end_line),
    }
    if ptr.title:
        source["title"] = ptr.title
    item: dict[str, Any] = {
        "rank": rank,
        "chunk_id": int(ptr.chunk_id),
        "score": float(hp.score),
        "text": hp.text,
        "content_hash": f"sha256:{digest}",
        "source": source,
    }
    if hp.stale_on_disk:
        item["stale"] = True
    return item


def _resolve_next_label(cid: int, index: RetrievalIndex | None) -> str:
    if index is not None and (pos := index.id_to_pos.get(cid)) is not None:
        ptr = index.pointers[pos]
        for candidate in (ptr.title, index.doc_titles.get(ptr.rel_posix_path, ""), ptr.rel_posix_path):
            if candidate and (cleaned := sanitize_slip_label(candidate)):
                return cleaned
    return str(cid)


def _serialize_links(raw_links: Sequence[tuple[str, int, int, str]]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for kind, src, dst, term in raw_links:
        if kind == "cite":
            serialized.append({"kind": "cite", "src": int(src), "dst": int(dst)})
        elif kind == "pmi":
            clean_t = sanitize_slip_label(term)
            if clean_t:
                serialized.append({"kind": "pmi", "src": int(src), "dst": int(dst), "term": clean_t})
        else:
            raise ValueError(f"Unsupported recall link kind: {kind!r}")
    return serialized


def build_recall_payload(
    passages: Sequence[HydratedPassage],
    *,
    index: RetrievalIndex | None = None,
    search_result: SearchResult | None = None,
    log_bf: float | None = None,
    coverage: float | None = None,
    next_candidates: Sequence[tuple[int, float]] | None = None,
    links: Sequence[tuple[str, int, int, str]] | None = None,
    card_queued: bool = False,
) -> dict[str, Any]:
    """Assemble the canonical sparse-by-truth recall payload with strictly in-RAM next labels."""
    results = [_serialize_result_item(rank, hp) for rank, hp in enumerate(passages, start=1)]
    payload: dict[str, Any] = {"results": results}

    eff_log_bf = log_bf if log_bf is not None else (search_result.log_bf if search_result is not None else None)
    eff_coverage = coverage if coverage is not None else (search_result.coverage if search_result is not None else None)
    eff_next = (
        next_candidates
        if next_candidates is not None
        else (search_result.next_candidates if search_result is not None else ())
    )
    eff_links = links if links is not None else (search_result.links if search_result is not None else ())

    if _is_finite_number(eff_log_bf):
        payload["log_bf"] = float(eff_log_bf)
    if results and _is_finite_number(eff_coverage):
        payload["coverage"] = float(eff_coverage)
    if eff_next:
        payload["next"] = [
            {
                "chunk_id": int(cid),
                "score": float(sc),
                "label": _resolve_next_label(int(cid), index),
            }
            for cid, sc in eff_next
        ]
    if eff_links and (serialized_links := _serialize_links(eff_links)):
        payload["links"] = serialized_links
    if card_queued:
        payload["card_queued"] = True
    return payload


def _format_links_line(raw_links: Sequence[Mapping[str, Any]]) -> str | None:
    parts: list[str] = []
    for link in raw_links:
        kind = link.get("kind")
        src = int(link["src"])
        dst = int(link["dst"])
        if kind == "cite":
            parts.append(f"{src}->{dst}")
        elif kind == "pmi":
            clean_t = sanitize_slip_label(str(link.get("term", "")))
            if clean_t:
                parts.append(f'{src}~pmi("{clean_t}")~{dst}')
    return f"links: {', '.join(parts)}" if parts else None


def format_recall_text(payload: Mapping[str, Any]) -> str:
    """Render a sparse recall payload as terse ASCII matching _PASSAGE_HEAD_RE."""
    results = payload.get("results") or ()
    if not results:
        return NO_MATCHES_TEXT

    lines: list[str] = []
    log_bf = payload.get("log_bf")
    cov = payload.get("coverage")
    if _is_finite_number(log_bf) and _is_finite_number(cov):
        lines.append(f"certainty: log_bf={float(log_bf):.3f} coverage={float(cov):.3f}")

    blocks: list[str] = []
    for item in results:
        src = item["source"]
        rel = str(src["rel_posix_path"])
        sl = int(src["start_line"])
        el = int(src["end_line"])
        cid = int(item["chunk_id"])
        score = float(item["score"])
        body = f"[STALE ON DISK] {rel}" if item.get("stale") else str(item.get("text", ""))
        blocks.append(f"[chunk_id={cid}] ({rel}:{sl}-{el}) score={score:.3f}:\n{body}")
    lines.append("\n\n".join(blocks))

    if raw_next := payload.get("next"):
        next_parts = [
            f'{int(n["chunk_id"])} "{sanitize_slip_label(str(n.get("label", "")))}" ({float(n["score"]):.3f})'
            for n in raw_next
        ]
        lines.append(f"next: {', '.join(next_parts)}")

    if (raw_links := payload.get("links")) and (links_line := _format_links_line(raw_links)) is not None:
        lines.append(links_line)

    return "\n".join(lines)


def dump_recall_json(payload: Any, *, indent: int | None = None) -> str:
    """Serialize payload strictly per RFC 8259, failing fast on non-finite floats."""
    return json.dumps(payload, indent=indent, allow_nan=False)
