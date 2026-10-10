"""JSON extraction, candidate parsing, and span alignment for LLM output."""

from __future__ import annotations

import ast
import json
import logging
import re
from types import SimpleNamespace
from typing import Any

from spacy.language import Language
from spacy.tokens import Doc, Span
from spacy.util import filter_spans

from yaw import config
from yaw.nlp.components import (
    _filter_code_spans,
    _resolve_base_predicates,
    is_named_entity,
    normalize_predicate,
)
from yaw.nlp.models import (
    ExtractedNode,
    ExtractedTriple,
    LeafTaxonomy,
    normalize_label,
)
from yaw.nlp.utils import is_valid_entity_text

logger = logging.getLogger("yaw.leaf_extractor")


def align_span(doc: Doc, entity_text: str) -> Span | None:
    """Align raw entity string to doc.char_span(...) verifying Invariant E1 byte/token offsets."""
    if doc is None or not entity_text or not entity_text.strip():
        return None
    raw_sub = re.sub(r"\s+", " ", entity_text).strip(" \t\n\r\"'.,:;!?(){}[]<>")
    if not raw_sub:
        return None

    # 1. Exact verbatim match (contract then expand)
    start = doc.text.find(raw_sub)
    if start != -1:
        end = start + len(raw_sub)
        span = doc.char_span(start, end, alignment_mode="contract")
        if span is not None:
            return span
        span = doc.char_span(start, end, alignment_mode="expand")
        if span is not None:
            return span

    # 2. Case-insensitive or flexible-whitespace regex match (contract then expand)
    pattern = r"\s+".join(re.escape(w) for w in raw_sub.split())
    if pattern:
        match = re.search(pattern, doc.text, flags=re.IGNORECASE)
        if match:
            start, end = match.start(), match.end()
            span = doc.char_span(start, end, alignment_mode="contract")
            if span is not None:
                return span
            span = doc.char_span(start, end, alignment_mode="expand")
            if span is not None:
                return span

    return None


_anchor_span = align_span


def _clean_candidate_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Strip and collapse whitespace/newlines across string dictionary fields."""
    cleaned: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, str):
            cleaned[k] = re.sub(r"\s+", " ", v).strip()
        else:
            cleaned[k] = v
    return cleaned


def _parse_json_array(s: str) -> list[dict[str, Any]] | None:
    """Parse outer bracketed JSON array if present."""
    start_arr = s.find("[")
    end_arr = s.rfind("]")
    if start_arr != -1 and end_arr != -1 and end_arr > start_arr:
        chunk = s[start_arr : end_arr + 1]
        try:
            val = json.loads(chunk)
            if isinstance(val, list):
                return [_clean_candidate_dict(x) for x in val if isinstance(x, dict)]
        except json.JSONDecodeError:
            try:
                val = ast.literal_eval(chunk)
                if isinstance(val, list):
                    return [
                        _clean_candidate_dict(x) for x in val if isinstance(x, dict)
                    ]
            except ValueError, SyntaxError:
                pass
    return None


def _parse_json_object(s: str) -> list[dict[str, Any]] | None:
    """Parse outer bracketed JSON object or embedded triples array."""
    start_obj = s.find("{")
    end_obj = s.rfind("}")
    if start_obj != -1 and end_obj != -1 and end_obj >= start_obj:
        chunk = s[start_obj : end_obj + 1]
        try:
            val = json.loads(chunk)
            if isinstance(val, dict):
                triples = val.get("triples")
                if isinstance(triples, list):
                    return [
                        _clean_candidate_dict(x) for x in triples if isinstance(x, dict)
                    ]
                return [_clean_candidate_dict(val)]
        except json.JSONDecodeError:
            try:
                val = ast.literal_eval(chunk)
                if isinstance(val, dict):
                    triples = val.get("triples")
                    if isinstance(triples, list):
                        return [
                            _clean_candidate_dict(x)
                            for x in triples
                            if isinstance(x, dict)
                        ]
                    return [_clean_candidate_dict(val)]
            except ValueError, SyntaxError:
                pass
    return None


def _parse_json_stream(s: str) -> list[dict[str, Any]]:
    """Decode concatenated or multiline JSON objects from string."""
    start_obj = s.find("{")
    if start_obj == -1:
        return []
    decoder = json.JSONDecoder()
    pos = start_obj
    results: list[dict[str, Any]] = []
    while pos < len(s):
        idx = s.find("{", pos)
        if idx == -1:
            break
        try:
            obj, end_pos = decoder.raw_decode(s, idx)
            if isinstance(obj, dict):
                results.append(_clean_candidate_dict(obj))
            pos = end_pos
        except json.JSONDecodeError:
            pos = idx + 1
    return results


def extract_json_payload(raw_text: str | list[str] | None) -> list[dict[str, Any]]:
    """Cleanly parse JSON array, JSON lines, or single JSON object from LLM response
    without fragile regex cleaning or markdown fence stripping loops.
    """
    if raw_text is None:
        return []
    s = (
        "\n".join(str(item) for item in raw_text)
        if isinstance(raw_text, (list, tuple))
        else str(raw_text)
    ).strip()
    if not s:
        return []

    # 1. Fast path: Outer array brackets [ ... ]
    arr = _parse_json_array(s)
    if arr is not None:
        return arr

    # 2. Outer single object brackets { ... }
    obj = _parse_json_object(s)
    if obj is not None:
        return obj

    # 3. Stream / JSON lines fallback
    return _parse_json_stream(s)


def _get_first_str(d: dict[str, Any], keys: tuple[str, ...], default: str = "") -> str:
    """Return the first non-empty string value for candidate keys."""
    for k in keys:
        val = d.get(k)
        if val is not None and str(val).strip():
            return str(val).strip()
    return default


def _resolve_triple_namespace(
    item: dict[str, Any],
    prefix: str,
    text: str = "",
    nlp: Language | None = None,
) -> str:
    """Resolve and normalize namespace for subject or object."""
    ent_text = text or _get_first_str(item, (prefix, "name", "text"), "")
    if ent_text and is_named_entity(ent_text, nlp):
        return "entities"

    raw = _get_first_str(
        item, (f"{prefix}_type", f"{prefix}_label", f"{prefix}_namespace"), ""
    )
    if raw:
        ns = normalize_label(raw)
        if ns in config.TAXONOMY_NAMESPACES:
            return ns

    return "concepts"


def _anchor_triple_in_doc(
    triple: dict[str, Any], doc: Doc, new_spans: list[Span]
) -> None:
    """Anchor triple subject and object spans within doc text."""
    for role in ("subject", "object"):
        text = triple[role]
        span = align_span(doc, text)
        if span is not None:
            ns = triple[f"{role}_type"]
            triple[f"{role}_span"] = (span.start_char, span.end_char)
            new_spans.append(Span(doc, span.start, span.end, label=ns))


def _parse_single_candidate(
    item: dict[str, Any],
    bp_list: list[str] | None,
    nlp: Language | None,
) -> dict[str, Any] | None:
    """Validate and construct a single triple dictionary candidate."""
    raw_sub = _get_first_str(item, ("subject", "sub", "source"))
    sub = re.sub(r"\s+", " ", raw_sub).strip(" \t\n\r\"'.,:;!?(){}[]<>")

    pred = _get_first_str(item, ("predicate", "pred", "relation", "rel")).strip()

    raw_obj = _get_first_str(item, ("object", "obj", "target"))
    obj = re.sub(r"\s+", " ", raw_obj).strip(" \t\n\r\"'.,:;!?(){}[]<>")

    if not (sub and pred and obj) or not (
        is_valid_entity_text(sub) and is_valid_entity_text(obj)
    ):
        return None

    norm_pred = normalize_predicate(pred, base_predicates=bp_list, nlp=nlp)
    if not norm_pred:
        norm_pred = pred or "relates to"

    return {
        "subject": sub,
        "predicate": norm_pred,
        "object": obj,
        "subject_type": _resolve_triple_namespace(item, "subject", text=sub, nlp=nlp),
        "object_type": _resolve_triple_namespace(item, "object", text=obj, nlp=nlp),
    }


def parse_raw_triples_json(
    raw_text: str | list[str] | None,
    doc: Doc | None = None,
    base_predicates: tuple[str, ...] | list[str] | None = None,
    nlp: Language | None = None,
) -> list[dict[str, Any]]:
    """Parse raw LLM output into triple dictionaries, defensively handling markdown code fences,
    trailing whitespace, multiline formatting, and running predicates through normalize_predicate()
    and entities through label/namespace normalization. Anchors text spans via doc.char_span(...) if doc provided.
    """
    candidates = extract_json_payload(raw_text)
    if not candidates:
        return []

    bp_list = _resolve_base_predicates(base_predicates, nlp)

    results: list[dict[str, Any]] = []
    new_spans: list[Span] = []

    for item in candidates:
        triple = _parse_single_candidate(item, bp_list, nlp)
        if triple is None:
            continue
        if doc is not None and doc.text:
            _anchor_triple_in_doc(triple, doc, new_spans)
        results.append(triple)

    if doc is not None and new_spans:
        existing = list(doc.ents) if doc.ents else []
        code_spans = doc.spans.get("code", [])
        clean_spans = _filter_code_spans(existing + new_spans, code_spans)
        doc.ents = filter_spans(clean_spans)

        # Populate doc._.rel for compatibility
        if not hasattr(doc._, "rel") or doc._.rel is None:
            doc._.rel = []
        ent_span_map = {
            (ent.start_char, ent.end_char): idx for idx, ent in enumerate(doc.ents)
        }
        for triple in results:
            sub_span = triple.get("subject_span")
            obj_span = triple.get("object_span")
            if sub_span in ent_span_map and obj_span in ent_span_map:
                dep_idx = ent_span_map[sub_span]
                dest_idx = ent_span_map[obj_span]
                doc._.rel.append(
                    SimpleNamespace(
                        dep=dep_idx,
                        dest=dest_idx,
                        relation=triple["predicate"],
                    )
                )

    return results


def parse_raw_triples(
    raw_text: str | list[str] | None,
    doc: Doc | None = None,
    base_predicates: tuple[str, ...] | list[str] | None = None,
    nlp: Language | None = None,
) -> list[ExtractedTriple]:
    """Parse raw LLM output into validated ExtractedTriple models, running predicates
    through normalize_predicate() and entities through label/namespace normalization.
    """
    dicts = parse_raw_triples_json(
        raw_text, doc=doc, base_predicates=base_predicates, nlp=nlp
    )
    triples: list[ExtractedTriple] = []
    seen: set[tuple[str, str, str]] = set()

    for item in dicts:
        sub_name = str(item["subject"]).strip()
        pred_name = str(item["predicate"]).strip()
        obj_name = str(item["object"]).strip()
        sub_ns = str(item.get("subject_type", "entities")).strip() or "entities"
        obj_ns = str(item.get("object_type", "entities")).strip() or "entities"

        key = (sub_name.casefold(), pred_name.casefold(), obj_name.casefold())
        if key in seen:
            continue
        seen.add(key)

        sub_node = ExtractedNode(
            name=sub_name,
            namespace=sub_ns,
            definition=f"{sub_name} ({sub_ns})",
        )
        pred_node = ExtractedNode(
            name=pred_name,
            namespace="predicates",
            definition=f"Relation '{pred_name}' connecting {sub_name} and {obj_name}.",
        )
        obj_node = ExtractedNode(
            name=obj_name,
            namespace=obj_ns,
            definition=f"{obj_name} ({obj_ns})",
        )
        triples.append(
            ExtractedTriple(subject=sub_node, predicate=pred_node, object=obj_node)
        )
    return triples


def parse_raw_triples_to_taxonomy(
    raw_text: str | list[str] | None,
    doc: Doc | None = None,
    base_predicates: tuple[str, ...] | list[str] | None = None,
    nlp: Language | None = None,
    category: str | None = None,
) -> LeafTaxonomy:
    """Parse raw LLM output directly into a LeafTaxonomy schema."""
    triples = parse_raw_triples(
        raw_text, doc=doc, base_predicates=base_predicates, nlp=nlp
    )
    node_map: dict[str, ExtractedNode] = {}
    for t in triples:
        for n in (t.subject, t.predicate, t.object):
            key = f"{n.namespace}:{n.name.casefold()}"
            if key not in node_map:
                node_map[key] = n
    return LeafTaxonomy(
        nodes=list(node_map.values()), triples=triples, category=category
    )


parse_extracted_triples = parse_raw_triples


__all__ = [
    "_anchor_span",
    "_anchor_triple_in_doc",
    "_clean_candidate_dict",
    "_get_first_str",
    "_parse_json_array",
    "_parse_json_object",
    "_parse_json_stream",
    "_parse_single_candidate",
    "_resolve_triple_namespace",
    "align_span",
    "extract_json_payload",
    "parse_extracted_triples",
    "parse_raw_triples",
    "parse_raw_triples_json",
    "parse_raw_triples_to_taxonomy",
]
