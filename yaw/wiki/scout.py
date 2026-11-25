"""Bare DSPy wiki card scouting and synthesis triggered by low-confidence recall."""

from __future__ import annotations

import math
from collections.abc import Sequence

import dspy

from yaw.core import HydratedPassage, normalize_ast_heading
from yaw.retrieval import SearchResult, format_evidence_route

from .cards import validate_wiki_slug


class ScoutCardSignature(dspy.Signature):
    """Synthesize a concise, strictly grounded wiki note card from retrieved passages, or output 'ABSTAIN' in body
    when retrieved passages lack sufficient grounded evidence."""

    query: str = dspy.InputField(desc="Query that triggered wiki card synthesis.")
    passages: str = dspy.InputField(desc="Disk-verified retrieved passages with provenance spans.")
    slug: str = dspy.OutputField(desc="Clean relative kebab-case slug without .md suffix.")
    title: str = dspy.OutputField(desc="Concise H1 heading title for the synthesized wiki card.")
    kind: str = dspy.OutputField(desc="Card kind such as 'note', 'synthesis', or 'contradiction'.")
    body: str = dspy.OutputField(desc="Concise grounded Markdown body, or 'ABSTAIN' if evidence is insufficient.")


def is_fresh_passage(passage: HydratedPassage) -> bool:
    """Return True when a hydrated passage is fresh on disk and contains non-whitespace text."""
    return not passage.stale_on_disk and bool(passage.text.strip())


def should_queue_scout(search_result: SearchResult | None, passages: Sequence[HydratedPassage]) -> bool:
    """Return True when recall produced fresh passages with weak Bayes factor or incomplete top-k coverage."""
    if search_result is None or not any(is_fresh_passage(hp) for hp in passages):
        return False
    if not math.isfinite(search_result.log_bf) or search_result.log_bf <= 0.0:
        return True
    return search_result.coverage < 0.5 and bool(search_result.next_candidates)


def synthesize_scout_card(
    query: str,
    passages: Sequence[HydratedPassage],
    lm: dspy.LM,
) -> tuple[str, str, str, str] | None:
    """Synthesize `(slug, title, body, kind)` using a bare `dspy.Predict` call, or return `None` if abstained."""
    clean_query = query.strip()
    fresh_passages = [hp for hp in passages if is_fresh_passage(hp)]
    if not clean_query or not fresh_passages:
        return None

    formatted = format_evidence_route(fresh_passages)
    predictor = dspy.Predict(ScoutCardSignature)
    with dspy.context(lm=lm):
        pred = predictor(query=clean_query, passages=formatted)

    raw_slug: object = getattr(pred, "slug", "")
    raw_title: object = getattr(pred, "title", "")
    raw_body: object = getattr(pred, "body", "")
    raw_kind: object = getattr(pred, "kind", "")

    if not isinstance(raw_body, str):
        return None
    clean_body = raw_body.strip()
    if not clean_body or clean_body.casefold() == "abstain":
        return None
    if not isinstance(raw_title, str) or not (clean_title := normalize_ast_heading(raw_title)):
        return None
    if not isinstance(raw_slug, str):
        return None
    try:
        clean_slug = validate_wiki_slug(raw_slug.lower().replace(" ", "-"))
    except ValueError:
        return None
    clean_kind = raw_kind.strip().lower() if isinstance(raw_kind, str) and raw_kind.strip() else "note"
    return clean_slug, clean_title, clean_body, clean_kind
