"""Translate synthetic test decisions into the compact request's wire identifiers."""

import json
import re
from typing import Any


def candidate_rows(text: str) -> list[dict[str, Any]]:
    m = re.search(r'\{\s*"format"\s*:\s*"synonym-evidence-v2"', text)
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

    # Determine stage from stage_text or decision contents
    text_lower = stage_text.lower()
    is_label_stage = (
        "judgelabels" in text_lower
        or "label_judge" in text_lower
        or "alt_label" in text_lower
        or "hidden_label" in text_lower
        or "equivalent" in text_lower
    )
    is_related_stage = not is_label_stage and (
        "judgerelated" in text_lower
        or "related_judge" in text_lower
        or "differentfrom" in text_lower
        or "different_from" in text_lower
    )
    is_hierarchy_stage = (
        not is_label_stage
        and not is_related_stage
        and ("judgehierarchy" in text_lower or "hierarchy_judge" in text_lower or "narrower" in text_lower)
    )

    if not (is_label_stage or is_hierarchy_stage or is_related_stage):
        if any("equivalent" in d or "alt_label" in d or "hidden_label" in d for d in decisions):
            is_label_stage = True
        elif any("broader" in d or "narrower" in d for d in decisions):
            is_hierarchy_stage = True
        elif any("related" in d and isinstance(d.get("related"), bool) for d in decisions):
            is_related_stage = True

    # If still not determined and stage_text is empty, fallback to legacy format
    if not (is_label_stage or is_hierarchy_stage or is_related_stage):
        result: list[dict[str, Any]] = []
        for pair in candidates:
            pid = pair["pair_id"]
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
            d = chosen.get(pid, {})
            if "equivalent" in d or "alt_label" in d or "hidden_label" in d:
                hidden = bool(d.get("hidden_label", False))
                alt = bool(d.get("alt_label", False))
                equiv = bool(d.get("equivalent", False)) or alt or hidden
                reason = str(d.get("reasoning", f"Label reasoning for {pid}"))
            else:
                rel = str(d.get("relation", "")).strip()
                if rel == "skos:altLabel":
                    equiv, hidden = True, False
                    reason = f"Alias label for {pid}"
                elif rel == "skos:hiddenLabel":
                    equiv, hidden = True, True
                    reason = f"Hidden label for {pid}"
                else:
                    equiv, hidden = False, False
                    reason = f"Not a label alias for {pid}"
            if equiv:
                item: dict[str, Any] = {
                    "pair_id": pid,
                    "reasoning": reason,
                    "equivalent": True,
                    "definition": str(d.get("definition") or f"Unified definition for {pid}"),
                    "hidden_label": hidden,
                }
            else:
                item = {
                    "pair_id": pid,
                    "reasoning": reason,
                    "equivalent": False,
                }
            if "canonical" in d and d["canonical"]:
                canon = str(d["canonical"])
                if canon in ("A", "B"):
                    canon = pair["entity_a" if canon == "A" else "entity_b"]
                for side in ("a", "b"):
                    if canon.casefold() == str(pair.get(f"name_{side}", "")).casefold():
                        canon = pair[f"entity_{side}"]
                        break
                item["canonical"] = canon
            res_list.append(item)
        return res_list

    if is_hierarchy_stage:
        res_list = []
        for pair in candidates:
            pid = pair["pair_id"]
            d = chosen.get(pid, {})
            if "broader" in d or "narrower" in d:
                broader = bool(d.get("broader", False))
                narrower = bool(d.get("narrower", False))
                reason = str(d.get("reasoning", f"Hierarchy reasoning for {pid}"))
            else:
                rel = str(d.get("relation", "")).strip()
                if rel == "skos:broader":
                    if "canonical" in d and not str(d.get("canonical", "")).strip():
                        raise ValueError(f"invalid canonical: empty canonical in skos:broader decision for {pid}")
                    canon = str(d.get("canonical", "")).strip().casefold()
                    a_id = str(pair.get("entity_a", ""))
                    b_id = str(pair.get("entity_b", ""))
                    name_a = str(pair.get("name_a", "")).strip().casefold()
                    name_b = str(pair.get("name_b", "")).strip().casefold()
                    if canon in (b_id.casefold(), name_b, "b"):
                        broader, narrower = True, False
                    elif canon in (a_id.casefold(), name_a, "a"):
                        broader, narrower = False, True
                    else:
                        broader, narrower = True, False
                    reason = f"Broader hierarchy for {pid}"
                elif rel == "skos:narrower":
                    broader, narrower = False, True
                    reason = f"Narrower hierarchy for {pid}"
                else:
                    broader, narrower = False, False
                    reason = f"Not hierarchical for {pid}"
            res_list.append(
                {
                    "pair_id": pid,
                    "reasoning": reason,
                    "broader": broader,
                    "narrower": narrower,
                }
            )
        return res_list

    if is_related_stage:
        res_list = []
        for pair in candidates:
            pid = pair["pair_id"]
            d = chosen.get(pid, {})
            if "related" in d and isinstance(d["related"], bool):
                related = d["related"]
                reason = str(d.get("reasoning", f"Related reasoning for {pid}"))
            else:
                rel = str(d.get("relation", "")).strip()
                if rel in ("owl:differentFrom", "different", "different_from"):
                    related = False
                    reason = f"Different concepts for {pid}"
                else:
                    related = True
                    reason = f"Related concepts for {pid}"
            res_list.append(
                {
                    "pair_id": pid,
                    "reasoning": reason,
                    "related": related,
                }
            )
        return res_list

    return []
