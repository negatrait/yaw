"""Stateless parsing, JSON repair, entity/triple extraction, and candidate fact matching."""

import ast
import difflib
import json
import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

from yaw.core import compute_mdhash_id


def text_processing(text: str | list[Any]) -> str | list[Any]:
    """Entity and alias normalization: lowercase, alphanumeric and whitespace only."""
    if isinstance(text, list):
        return [text_processing(t) for t in text]
    if not isinstance(text, str):
        text = str(text)
    normalized = "".join(
        ch if ch.isalnum() or ch.isspace() else " " for ch in text.casefold()
    )
    return " ".join(normalized.split())


def filter_invalid_triples(triples: Sequence[Any]) -> list[list[str]]:
    """Filter out invalid and duplicate triples, preserving original order."""
    unique_triples: set[tuple[str, str, str]] = set()
    valid_triples: list[list[str]] = []

    for triple in triples:
        if not isinstance(triple, (list, tuple)) or len(triple) != 3:
            continue
        valid_triple = [str(item) for item in triple]
        key = (valid_triple[0], valid_triple[1], valid_triple[2])
        if key not in unique_triples:
            unique_triples.add(key)
            valid_triples.append(valid_triple)

    return valid_triples


def repair_truncated_json(response: str, *, is_triples: bool = False) -> str:
    """String-aware bracket depth scanner to repair JSON responses truncated by max tokens."""
    try:
        json.loads(response)
        return response
    except (ValueError, TypeError):
        pass

    depth, in_str, esc, last_end = 0, False, False, -1
    for i, ch in enumerate(response):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
                if not is_triples and depth == 2:
                    last_end = i + 1
        elif ch == '"':
            in_str = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            if is_triples and depth == 3 and ch == "]":
                last_end = i + 1
            depth -= 1

    if last_end == -1:
        return '{"triples": []}' if is_triples else '{"named_entities": []}'
    return response[:last_end] + "]}"


def _extract_json_list_field(response: str, field_name: str) -> list[Any]:
    """Scan a response string for the first JSON object containing `field_name` as a list."""
    decoder = json.JSONDecoder()
    for start_index, character in enumerate(response):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(response[start_index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or field_name not in payload:
            continue
        value = payload[field_name]
        if not isinstance(value, list):
            raise ValueError(f"OpenIE response field {field_name!r} must be a list.")
        return value
    raise ValueError(f"OpenIE response does not contain a valid JSON object with {field_name!r}.")


def _parse_bare_json_string_list(response: str) -> list[str] | None:
    """Return the response as a list of strings if it is exactly one JSON array of strings, else None."""
    text = response.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text[3:-3]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
        return None
    return payload


def extract_named_entities(response: str) -> tuple[str, ...]:
    """Extract and deduplicate unique named entities from LLM response."""
    try:
        raw_list = _extract_json_list_field(response, "named_entities")
    except ValueError:
        bare = _parse_bare_json_string_list(response)
        if bare is not None:
            raw_list = bare
        else:
            repaired = repair_truncated_json(response, is_triples=False)
            try:
                raw_list = _extract_json_list_field(repaired, "named_entities")
            except ValueError:
                bare_repaired = _parse_bare_json_string_list(repaired)
                if bare_repaired is not None:
                    raw_list = bare_repaired
                else:
                    return ()

    unique: list[str] = []
    seen: set[str] = set()
    for item in raw_list:
        if isinstance(item, str):
            clean = item.strip()
            if clean and clean not in seen:
                seen.add(clean)
                unique.append(clean)
    return tuple(unique)


def extract_triples(response: str) -> tuple[tuple[str, str, str], ...]:
    """Extract, validate, and deduplicate relation triples from LLM response."""
    try:
        raw_list = _extract_json_list_field(response, "triples")
    except ValueError:
        repaired = repair_truncated_json(response, is_triples=True)
        try:
            raw_list = _extract_json_list_field(repaired, "triples")
        except ValueError:
            return ()

    filtered = filter_invalid_triples(raw_list)
    return tuple((t[0], t[1], t[2]) for t in filtered)


def parse_filter_facts(response: str) -> list[list[str]]:
    """Parse candidate facts from DSPy recognition memory response."""
    sections: list[tuple[str | None, list[str]]] = [(None, [])]
    field_header_pattern = re.compile(r"\[\[ ## (\w+) ## \]\]")
    for line in response.splitlines():
        match = field_header_pattern.match(line.strip())
        if match:
            sections.append((match.group(1), []))
        else:
            sections[-1][1].append(line)

    parsed: list[list[str]] = []
    found_section = False
    for k, lines in sections:
        if k == "fact_after_filter":
            found_section = True
            value = "\n".join(lines).strip()
            parsed_value = None
            try:
                parsed_value = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                try:
                    parsed_value = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    parsed_value = value

            if isinstance(parsed_value, dict) and "fact" in parsed_value:
                raw_facts = parsed_value["fact"]
                if isinstance(raw_facts, list):
                    for item in raw_facts:
                        if isinstance(item, (list, tuple)) and len(item) == 3:
                            parsed.append([str(x) for x in item])
            elif isinstance(parsed_value, list):
                for item in parsed_value:
                    if isinstance(item, (list, tuple)) and len(item) == 3:
                        parsed.append([str(x) for x in item])

    if not found_section:
        text = response.strip()
        try:
            val = json.loads(text)
            if isinstance(val, dict) and "fact" in val and isinstance(val["fact"], list):
                for item in val["fact"]:
                    if isinstance(item, (list, tuple)) and len(item) == 3:
                        parsed.append([str(x) for x in item])
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, (list, tuple)) and len(item) == 3:
                        parsed.append([str(x) for x in item])
        except Exception:
            pass

    return parsed


def match_facts_to_candidate_indices(
    candidate_facts: Sequence[tuple[str, str, str] | list[str]],
    generated_facts: Sequence[Sequence[str]],
    max_facts: int | None = 4,
) -> list[int]:
    """Map generated [s, p, o] facts back to original candidate_facts indices via difflib."""
    if not generated_facts or not candidate_facts:
        return []

    candidate_str_list = [
        str(list(f) if isinstance(f, tuple) else f) for f in candidate_facts
    ]
    candidate_list_of_lists = [
        list(f) if isinstance(f, tuple) else f for f in candidate_facts
    ]

    result_indices: list[int] = []
    for generated_fact in generated_facts:
        gen_str = str(list(generated_fact))
        matches = difflib.get_close_matches(
            gen_str, candidate_str_list, n=1, cutoff=0.0
        )
        if matches:
            closest_matched_fact = matches[0]
            try:
                idx = candidate_list_of_lists.index(ast.literal_eval(closest_matched_fact))
                if idx not in result_indices:
                    result_indices.append(idx)
            except (ValueError, SyntaxError):
                pass

    if max_facts is not None:
        return result_indices[:max_facts]
    return result_indices


def filter_reranked_facts(
    candidate_facts: Sequence[tuple[str, str, str] | list[str]],
    response: str,
    max_facts: int | None = None,
) -> list[int]:
    """Map generated facts back to original candidate fact indices via difflib."""
    generated_facts = parse_filter_facts(response)
    if not generated_facts:
        return []
    return match_facts_to_candidate_indices(
        candidate_facts, generated_facts, max_facts=max_facts
    )


def strip_mistral_bracket_prefix(text: str) -> str:
    """Strip leading `]\n` artifact occasionally emitted by Mistral v0.3 after DSPy `[[ ## field ## ]]` headers."""
    return re.sub(r"^\s*\]\s*\n+", "", text).strip()


def coerce_generated_facts(raw_output: Any) -> list[list[str]]:
    """Extract 3-element string lists from a FactList, dict, JSON string, or list."""
    if hasattr(raw_output, "fact"):
        candidates = raw_output.fact
    elif isinstance(raw_output, dict) and "fact" in raw_output:
        candidates = raw_output["fact"]
    elif isinstance(raw_output, list):
        candidates = raw_output
    elif isinstance(raw_output, str):
        return parse_filter_facts(raw_output)
    else:
        return []

    valid: list[list[str]] = []
    if isinstance(candidates, list):
        for item in candidates:
            if isinstance(item, (list, tuple)) and len(item) == 3:
                valid.append([str(x) for x in item])
    return valid


def extract_entity_nodes(
    triples: Sequence[tuple[str, str, str] | Sequence[str]],
) -> list[tuple[str, str]]:
    """Extract unique (entity_id, normalized_name) from triple subjects and objects."""
    seen: set[str] = set()
    result: list[tuple[str, str]] = []
    for t in triples:
        if len(t) < 3:
            continue
        subj = text_processing(str(t[0]))
        obj = text_processing(str(t[2]))
        if isinstance(subj, str) and subj and subj not in seen:
            seen.add(subj)
            result.append((compute_mdhash_id(subj, prefix="entity-"), subj))
        if isinstance(obj, str) and obj and obj not in seen:
            seen.add(obj)
            result.append((compute_mdhash_id(obj, prefix="entity-"), obj))
    return result


def flatten_facts(
    triples: Sequence[tuple[str, str, str] | Sequence[str]],
) -> list[tuple[str, str, str, str, str]]:
    """Convert triples to normalized (fact_id, subj_id, pred, obj_id, raw_triple_json)."""
    seen: set[str] = set()
    result: list[tuple[str, str, str, str, str]] = []
    for t in triples:
        if len(t) < 3:
            continue
        subj = str(text_processing(str(t[0])))
        pred = str(text_processing(str(t[1])))
        obj = str(text_processing(str(t[2])))
        if not subj or not obj or subj == obj:
            continue
        subj_id = compute_mdhash_id(subj, prefix="entity-")
        obj_id = compute_mdhash_id(obj, prefix="entity-")
        triple_tuple = (subj, pred, obj)
        fact_id = compute_mdhash_id(str(triple_tuple), prefix="fact-")
        if fact_id not in seen:
            seen.add(fact_id)
            result.append((fact_id, subj_id, pred, obj_id, json.dumps(list(triple_tuple))))
    return result


def compute_qa_em_f1(pred_str: str, gold_answers: Sequence[str]) -> tuple[float, float]:
    """Compute exact match (EM) and token-level F1 score against gold answers."""
    norm_p = str(text_processing(pred_str))
    p_tokens = norm_p.split()
    p_counts = Counter(p_tokens)
    best_em = 0.0
    best_f1 = 0.0
    for gold in gold_answers:
        norm_g = str(text_processing(gold))
        if norm_p == norm_g or (norm_p and norm_g and norm_g in norm_p):
            best_em = 1.0
        g_tokens = norm_g.split()
        if not p_tokens and not g_tokens:
            best_f1 = 1.0
            continue
        common = sum((p_counts & Counter(g_tokens)).values())
        if common > 0:
            best_f1 = max(best_f1, (2.0 * common) / (len(p_tokens) + len(g_tokens)))
    return best_em, max(best_f1, best_em)


__all__ = [
    "compute_mdhash_id",
    "compute_qa_em_f1",
    "text_processing",
    "filter_invalid_triples",
    "repair_truncated_json",
    "extract_named_entities",
    "extract_triples",
    "parse_filter_facts",
    "match_facts_to_candidate_indices",
    "filter_reranked_facts",
    "strip_mistral_bracket_prefix",
    "coerce_generated_facts",
    "extract_entity_nodes",
    "flatten_facts",
]
