"""Translate synthetic test decisions into the compact request's wire identifiers."""

import json
from typing import Any


def candidate_rows(text: str) -> list[dict[str, Any]]:
    start = text.find('{"format":"synonym-evidence-v1"')
    if start < 0:
        raise ValueError("Expected a compact synonym evidence packet")
    packet = json.JSONDecoder().raw_decode(text[start:])[0]
    result = []
    for pair in packet["pairs"]:
        a, b = pair["pair_id"].split("|")
        result.append(
            {
                **pair,
                "entity_a": a,
                "entity_b": b,
                "name_a": packet["entities"][a]["name"],
                "name_b": packet["entities"][b]["name"],
            }
        )
    return result


def complete_decisions(candidates: list[dict[str, Any]], decisions: list[dict[str, str]]) -> list[dict[str, str]]:
    chosen = {d["pair_id"]: dict(d) for d in decisions}
    result = []
    for pair in candidates:
        decision = chosen.get(
            pair["pair_id"], {"pair_id": pair["pair_id"], "relation": "skos:related", "canonical": pair["entity_a"]}
        )
        if decision["canonical"] in ("A", "B"):
            decision["canonical"] = pair["entity_a" if decision["canonical"] == "A" else "entity_b"]
        for side in ("a", "b"):
            if decision["canonical"].casefold() == pair[f"name_{side}"].casefold():
                decision["canonical"] = pair[f"entity_{side}"]
                break
        result.append(decision)
    return result
