"""Translate synthetic test decisions into the compact request's wire identifiers."""

import json
import math
import re
from typing import Any

from dspy.lm15 import TokenLogprob


def response_logprobs(text: str, probability: float = 0.81) -> tuple[TokenLogprob, ...]:
    """Native DSPy token metadata with measurable Boolean decision tokens."""
    return tuple(
        TokenLogprob(
            token=token,
            bytes=tuple(token.encode()),
            logprob=math.log(probability) if token in ("true", "false") or token.startswith('"') else -0.001,
        )
        for token in re.findall(r'"(?:[^"\\]|\\.)*"|true|false|.', text, re.DOTALL)
    )


def candidate_rows(text: str) -> list[dict[str, Any]]:
    m = re.search(r'\{\s*"format"\s*:\s*"synonym-evidence-v3"', text)
    if not m and '"pairs"' in text:
        m = re.search(r'\{\s*"pairs"\s*:', text)
    if not m:
        raise ValueError("Expected a compact synonym evidence packet")
    packet = json.JSONDecoder().raw_decode(text[m.start() :])[0]
    result = []
    for pair in packet.get("pairs", []):
        pid = pair.get("pair_id", "")
        if "|" in pid:
            a, b = pid.split("|", 1)
        else:
            a, b = pair.get("e0", ""), pair.get("e1", "")
        ea = pair.get("entity_a")
        eb = pair.get("entity_b")
        name_a = ea["name"] if isinstance(ea, dict) else packet.get("entities", {}).get(a, {}).get("name", "")
        name_b = eb["name"] if isinstance(eb, dict) else packet.get("entities", {}).get(b, {}).get("name", "")
        result.append(
            {
                **pair,
                "_source_text": text,
                "pair_id": pid,
                "entity_a": a,
                "entity_b": b,
                "name_a": name_a,
                "name_b": name_b,
            }
        )
    return result


def complete_decisions(
    candidates: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    *,
    stage_text: str = "",
) -> list[dict[str, Any]]:
    chosen = {str(d["pair_id"]): dict(d) for d in decisions if "pair_id" in d}

    if not stage_text:
        stage_text = getattr(candidates, "_source_text", "")
        if not stage_text and candidates and "_source_text" in candidates[0]:
            stage_text = str(candidates[0]["_source_text"])

    # Evidence can mention other stages' predicates; route from the instructions.
    stage_text = stage_text.split("[[ ## candidate_pairs ## ]]", 1)[0]
    text_lower = stage_text.lower()
    is_label_stage = (
        "judgelabels" in text_lower
        or "label_judge" in text_lower
        or "alt_label" in text_lower
        or "hidden_label" in text_lower
        or "equivalent" in text_lower
    )
    is_relation_stage = not is_label_stage and (
        "judgerelations" in text_lower
        or "relationdecision" in text_lower
        or "relation_judge" in text_lower
        or "differentfrom" in text_lower
        or "skos:broader" in text_lower
    )
    if not (is_label_stage or is_relation_stage):
        if any("equivalent" in d or "alt_label" in d or "hidden_label" in d for d in decisions):
            is_label_stage = True
        elif any("relation" in d for d in decisions):
            is_relation_stage = True

    # If still not determined and stage_text is empty, fallback to legacy format
    if not (is_label_stage or is_relation_stage):
        result: list[dict[str, Any]] = []
        for pair in candidates:
            pid = pair["pair_id"]
            names = " / ".join(sorted((str(pair.get("name_a", "")), str(pair.get("name_b", "")))))
            decision = chosen.get(pid, {"pair_id": pid, "relation": "skos:related", "canonical": pair["entity_a"]})
            if decision.get("canonical") in ("A", "B"):
                decision["canonical"] = pair["entity_a" if decision["canonical"] == "A" else "entity_b"]
            for side in ("a", "b"):
                canon = str(decision.get("canonical", ""))
                if canon.casefold() == str(pair.get(f"name_{side}", "")).casefold():
                    decision["canonical"] = pair[f"entity_{side}"]
                    break
            result.append(decision)
        return result

    if is_label_stage:
        res_list: list[dict[str, Any]] = []
        for pair in candidates:
            pid = pair["pair_id"]
            names = " / ".join(sorted((str(pair.get("name_a", "")), str(pair.get("name_b", "")))))
            d = chosen.get(pid, {})
            if "equivalent" in d or "alt_label" in d or "hidden_label" in d:
                hidden = bool(d.get("hidden_label", False))
                alt = bool(d.get("alt_label", False))
                equiv = bool(d.get("equivalent", False)) or alt or hidden
                reason = str(d.get("reasoning", f"Label reasoning for {names}"))
            else:
                rel = str(d.get("relation", "")).strip()
                if rel in ("skos:altLabel", "skos:exactMatch", "exact_match"):
                    equiv, hidden = True, False
                    reason = f"Alias label for {names}"
                elif rel == "skos:hiddenLabel":
                    equiv, hidden = True, True
                    reason = f"Hidden label for {names}"
                else:
                    equiv, hidden = False, False
                    reason = f"Not a label alias for {names}"
            if equiv:
                item: dict[str, Any] = {
                    "pair_id": pid,
                    "reasoning": reason,
                    "equivalent": True,
                    "definition": str(d.get("definition") or f"Unified definition for {names}"),
                    "hidden_label": hidden,
                }
            else:
                item = {
                    "pair_id": pid,
                    "reasoning": reason,
                    "equivalent": False,
                }
            res_list.append(item)
        return res_list

    if is_relation_stage:
        result = []
        for pair in candidates:
            pid = pair["pair_id"]
            decision = chosen.get(pid, {})
            relation = str(decision.get("relation", "skos:related"))
            if relation == "skos:broader":
                canonical = str(decision.get("canonical", "")).strip().casefold()
                if "canonical" in decision and not canonical:
                    raise ValueError(f"invalid canonical: empty canonical in skos:broader decision for {pid}")
                if canonical in (str(pair.get("entity_a", "")).casefold(), str(pair.get("name_a", "")).casefold(), "a"):
                    relation = "skos:narrower"
            if relation in ("different", "different_from"):
                relation = "owl:differentFrom"
            if relation not in ("skos:broader", "skos:narrower", "skos:related", "owl:differentFrom"):
                relation = "skos:related"
            result.append(
                {
                    "pair_id": pid,
                    "reasoning": str(decision.get("reasoning", "Evidence supports relation")),
                    "relation": relation,
                }
            )
        return result

    return []
