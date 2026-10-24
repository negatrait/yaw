"""Stateless parsing, JSON repair, entity/triple extraction, and candidate fact matching."""

import ast
import difflib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import dspy
from dspy.evaluate.metrics import F1, normalize_text

from yaw.core import StructuredTriple, compute_mdhash_id, text_processing


def _nullable_text(value: Any) -> str:
    """Model null is missing text, never an entity or definition named 'None'."""
    return "" if value is None else str(value).strip()


def _coerce_structured_triple_item(
    item: Any,
    def_lookup: dict[str, str] | None = None,
) -> StructuredTriple | None:
    """Coerce a structured triple dict or 3/4-element sequence into a StructuredTriple."""
    lookup = def_lookup or {}
    if isinstance(item, StructuredTriple):
        return item
    if isinstance(item, dict):
        raw_subj = item.get("sub", item.get("subj", item.get("subject")))
        raw_pred = item.get("pred", item.get("predicate"))
        raw_obj = item.get("obj", item.get("object"))
        raw_spo_def = item.get("def", item.get("spoDef", item.get("spo_def", "")))
        if raw_pred is None or raw_subj is None or raw_obj is None:
            return None
        if isinstance(raw_subj, (list, tuple)) and len(raw_subj) >= 1:
            subj = _nullable_text(raw_subj[0])
            subj_def = str(raw_subj[1]).strip() if len(raw_subj) >= 2 and raw_subj[1] is not None else ""
        elif isinstance(raw_subj, str):
            subj = raw_subj.strip()
            subj_def = _nullable_text(item.get("subjDef", item.get("subj_def", "")))
        else:
            return None

        raw_obj_kind = str(item.get("obj_kind", item.get("objKind", "entity")) or "entity").strip().lower()
        obj_kind: Literal["entity", "literal"] = "literal" if raw_obj_kind == "literal" else "entity"
        obj_datatype = _nullable_text(item.get("obj_datatype", item.get("objDatatype", "")))
        obj_lang = _nullable_text(item.get("obj_lang", item.get("objLang", "")))

        if isinstance(raw_obj, (list, tuple)) and len(raw_obj) >= 1:
            obj = _nullable_text(raw_obj[0])
            obj_def = str(raw_obj[1]).strip() if len(raw_obj) >= 2 and raw_obj[1] is not None else ""
        elif isinstance(raw_obj, dict):
            obj = _nullable_text(raw_obj.get("value", raw_obj.get("lex", raw_obj.get("name", ""))))
            obj_def = _nullable_text(raw_obj.get("def", raw_obj.get("definition", "")))
            if str(raw_obj.get("kind", "")).strip().lower() == "literal" or "datatype" in raw_obj or "lang" in raw_obj:
                obj_kind = "literal"
                obj_datatype = _nullable_text(raw_obj.get("datatype", obj_datatype))
                obj_lang = _nullable_text(raw_obj.get("lang", obj_lang))
        elif isinstance(raw_obj, str):
            obj = raw_obj.strip()
            obj_def = _nullable_text(item.get("objDef", item.get("obj_def", "")))
        else:
            return None

        pred = str(raw_pred).strip()
        if not subj or not pred or not obj:
            return None
        spo_def = str(raw_spo_def).strip() if raw_spo_def is not None else ""
        if not spo_def:
            spo_def = f"{subj} {pred} {obj}."
        if not subj_def:
            subj_def = lookup.get(subj) or lookup.get(str(text_processing(subj))) or ""
        if obj_kind == "literal":
            obj_def = ""
        elif not obj_def:
            obj_def = lookup.get(obj) or lookup.get(str(text_processing(obj))) or ""
        return StructuredTriple(
            subj=subj,
            subj_def=subj_def,
            pred=pred,
            obj=obj,
            obj_def=obj_def,
            spo_def=spo_def,
            obj_kind=obj_kind,
            obj_datatype=obj_datatype if obj_kind == "literal" else "",
            obj_lang=obj_lang if obj_kind == "literal" else "",
        )

    if isinstance(item, (list, tuple)) and len(item) in (5, 6):
        subj = str(item[0]).strip() if item[0] is not None else ""
        subj_def = str(item[1]).strip() if item[1] is not None else ""
        pred = str(item[2]).strip() if item[2] is not None else ""
        obj = str(item[3]).strip() if item[3] is not None else ""
        obj_def = str(item[4]).strip() if item[4] is not None else ""
        if not subj or not pred or not obj:
            return None
        raw_spo = str(item[5]).strip() if len(item) == 6 and item[5] is not None else ""
        spo_def = raw_spo or f"{subj} {pred} {obj}."
        if not subj_def:
            subj_def = lookup.get(subj) or lookup.get(str(text_processing(subj))) or ""
        if not obj_def:
            obj_def = lookup.get(obj) or lookup.get(str(text_processing(obj))) or ""
        return StructuredTriple(
            subj=subj,
            subj_def=subj_def,
            pred=pred,
            obj=obj,
            obj_def=obj_def,
            spo_def=spo_def,
        )

    if isinstance(item, (list, tuple)) and len(item) in (3, 4):
        raw_s, raw_p, raw_o = item[0], item[1], item[2]
        if raw_s is None or raw_p is None or raw_o is None:
            return None
        if isinstance(raw_s, (list, tuple)) and len(raw_s) >= 1:
            subj = _nullable_text(raw_s[0])
            subj_def = str(raw_s[1]).strip() if len(raw_s) >= 2 and raw_s[1] is not None else ""
        else:
            subj = str(raw_s).strip()
            subj_def = ""
        if isinstance(raw_o, (list, tuple)) and len(raw_o) >= 1:
            obj = _nullable_text(raw_o[0])
            obj_def = str(raw_o[1]).strip() if len(raw_o) >= 2 and raw_o[1] is not None else ""
        else:
            obj = str(raw_o).strip()
            obj_def = ""
        pred = str(raw_p).strip()
        if not subj or not pred or not obj:
            return None
        raw_spo = str(item[3]).strip() if len(item) == 4 and item[3] is not None else ""
        spo_def = raw_spo or f"{subj} {pred} {obj}."
        if not subj_def:
            subj_def = lookup.get(subj) or lookup.get(str(text_processing(subj))) or ""
        if not obj_def:
            obj_def = lookup.get(obj) or lookup.get(str(text_processing(obj))) or ""
        return StructuredTriple(
            subj=subj,
            subj_def=subj_def,
            pred=pred,
            obj=obj,
            obj_def=obj_def,
            spo_def=spo_def,
        )
    return None


def filter_invalid_triples(triples: Sequence[Any]) -> list[list[str]]:
    """Filter out invalid and duplicate triples, preserving original order."""
    unique_triples: set[tuple[str, str, str]] = set()
    valid_triples: list[list[str]] = []

    for triple in triples:
        if isinstance(triple, (dict, StructuredTriple)):
            st = _coerce_structured_triple_item(triple)
            if st is None:
                continue
            valid_triple = [st.subj, st.pred, st.obj]
        elif isinstance(triple, (list, tuple)) and len(triple) == 3:
            valid_triple = [str(item) for item in triple]
        else:
            continue
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
    except ValueError, TypeError:
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
            if depth == 3 and ch in "]}":
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


def _parse_bare_json_list(response: str) -> list[Any] | None:
    """Return the response as a list if it is a JSON array of strings or [name, definition] pairs, else None."""
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
    if not isinstance(payload, list) or not all(isinstance(item, (str, list, tuple, dict)) for item in payload):
        return None
    return payload


def extract_entity_definitions(
    raw_input: Any,
    *,
    normalize_keys: bool = False,
) -> tuple[tuple[str, str], ...]:
    """Extract and deduplicate (entity_name, contextual_definition) pairs from Stage 1 NER, \
DefineMissingEntities output, or {name: definition} mappings."""
    raw_list: Sequence[Any]
    if isinstance(raw_input, str):
        try:
            raw_list = _extract_json_list_field(raw_input, "named_entities")
        except ValueError:
            try:
                raw_list = _extract_json_list_field(raw_input, "entity_definitions")
            except ValueError:
                bare = _parse_bare_json_list(raw_input)
                if bare is not None:
                    raw_list = bare
                else:
                    repaired = repair_truncated_json(raw_input, is_triples=False)
                    try:
                        raw_list = _extract_json_list_field(repaired, "named_entities")
                    except ValueError:
                        try:
                            raw_list = _extract_json_list_field(repaired, "entity_definitions")
                        except ValueError:
                            bare_repaired = _parse_bare_json_list(repaired)
                            raw_list = bare_repaired if bare_repaired is not None else ()
    elif isinstance(raw_input, Mapping):
        if "named_entities" in raw_input or "entity_definitions" in raw_input:
            if "named_entities" in raw_input and isinstance(raw_input["named_entities"], list):
                raw_list = raw_input["named_entities"]
            elif "entity_definitions" in raw_input and isinstance(raw_input["entity_definitions"], list):
                raw_list = raw_input["entity_definitions"]
            else:
                raw_list = ()
        else:
            raw_list = [
                (k, v)
                for k, v in raw_input.items()
                if isinstance(k, str) and (v is None or isinstance(v, str))
            ]
    elif isinstance(raw_input, Sequence):
        raw_list = raw_input
    else:
        raw_list = ()

    ordered_pairs: list[tuple[str, str]] = []
    indices_by_name: dict[str, list[int]] = {}
    for item in raw_list:
        name = ""
        defn = ""
        if isinstance(item, str):
            name = item.strip()
        elif isinstance(item, (list, tuple)) and len(item) >= 1:
            name = str(item[0]).strip()
            if len(item) >= 2 and item[1] is not None:
                defn = str(item[1]).strip()
        elif isinstance(item, dict):
            raw_name = item.get("name", item.get("entity", item.get("label", "")))
            raw_def = item.get("definition", item.get("def", item.get("skos:definition", "")))
            name = _nullable_text(raw_name)
            defn = _nullable_text(raw_def)
        if not name:
            continue
        key = (str(text_processing(name)) or name.lower()) if normalize_keys else name
        existing_idxs = indices_by_name.get(key)
        if not existing_idxs:
            indices_by_name[key] = [len(ordered_pairs)]
            ordered_pairs.append((name, defn))
        elif not defn:
            continue
        else:
            matched = False
            for idx in existing_idxs:
                prev_name, prev_def = ordered_pairs[idx]
                if not prev_def:
                    ordered_pairs[idx] = (prev_name, defn)
                    matched = True
                    break
                if prev_def.lower() == defn.lower():
                    if len(defn) > len(prev_def):
                        ordered_pairs[idx] = (prev_name, defn)
                    matched = True
                    break
            if not matched:
                existing_idxs.append(len(ordered_pairs))
                ordered_pairs.append((name, defn))

    return tuple(ordered_pairs)


def extract_named_entities(response: Any) -> tuple[str, ...]:
    """Extract and deduplicate unique named entities from LLM response (supporting [name, def] pairs and 1D strings)."""
    seen: set[str] = set()
    names: list[str] = []
    for name, _ in extract_entity_definitions(response):
        if name not in seen:
            seen.add(name)
            names.append(name)
    return tuple(names)


def extract_structured_triples(
    raw_input: Any,
    *,
    entity_definitions: Sequence[tuple[str, str]] | dict[str, str] | None = None,
) -> tuple[StructuredTriple, ...]:
    """Extract, validate, and deduplicate StructuredTriple records from raw list, dict, or JSON string."""
    def_lookup: dict[str, str] = {}
    multi_defs: dict[str, list[str]] = {}
    if isinstance(entity_definitions, dict):
        for k, v in entity_definitions.items():
            clean_k = str(k).strip()
            clean_v = str(v).strip()
            if clean_k and clean_v:
                def_lookup[clean_k] = clean_v
                def_lookup[str(text_processing(clean_k))] = clean_v
    elif isinstance(entity_definitions, Sequence):
        for pair in entity_definitions:
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                clean_k = str(pair[0]).strip()
                clean_v = str(pair[1]).strip()
                if clean_k and clean_v:
                    norm_k = str(text_processing(clean_k))
                    def_lookup.setdefault(clean_k, clean_v)
                    if norm_k:
                        def_lookup.setdefault(norm_k, clean_v)
                    for key_str in (clean_k, norm_k):
                        if key_str:
                            lst = multi_defs.setdefault(key_str, [])
                            if all(d.lower() != clean_v.lower() for d in lst):
                                lst.append(clean_v)

    raw_list: Sequence[Any]
    if isinstance(raw_input, str):
        try:
            raw_list = _extract_json_list_field(raw_input, "triples")
        except ValueError:
            repaired = repair_truncated_json(raw_input, is_triples=True)
            try:
                raw_list = _extract_json_list_field(repaired, "triples")
            except ValueError:
                try:
                    loaded = json.loads(raw_input)
                    raw_list = loaded if isinstance(loaded, list) else []
                except Exception:
                    raw_list = []
    elif isinstance(raw_input, dict) and "triples" in raw_input and isinstance(raw_input["triples"], list):
        raw_list = raw_input["triples"]
    elif isinstance(raw_input, Sequence):
        raw_list = raw_input
    else:
        raw_list = []

    seen: set[tuple[str, str, str, str, str, str]] = set()
    results: list[StructuredTriple] = []
    for item in raw_list:
        st = _coerce_structured_triple_item(item, def_lookup=def_lookup)
        if st is None:
            continue
        s_norm = str(text_processing(st.subj))
        o_norm = str(text_processing(st.obj))
        if (
            st.obj_kind == "entity"
            and s_norm
            and s_norm == o_norm
            and st.subj_def
            and st.obj_def
            and st.subj_def.lower() == st.obj_def.lower()
        ):
            senses = multi_defs.get(s_norm) or multi_defs.get(st.subj) or []
            if len(senses) >= 2:
                alt_sense = next(
                    (d for d in senses if d.lower() != st.subj_def.lower()),
                    st.obj_def,
                )
                st = StructuredTriple(
                    subj=st.subj,
                    subj_def=st.subj_def,
                    pred=st.pred,
                    obj=st.obj,
                    obj_def=alt_sense,
                    spo_def=st.spo_def,
                    obj_kind=st.obj_kind,
                    obj_datatype=st.obj_datatype,
                    obj_lang=st.obj_lang,
                )
        key = (
            st.subj,
            st.subj_def.strip().lower(),
            st.pred,
            st.obj,
            st.obj_def.strip().lower(),
            st.obj_kind,
        )
        if key not in seen:
            seen.add(key)
            results.append(st)
    return tuple(results)


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


def match_facts_to_candidate_indices(
    candidate_facts: Sequence[tuple[str, str, str] | list[str]],
    generated_facts: Sequence[Sequence[str]],
    max_facts: int | None = 4,
) -> list[int]:
    """Map generated [s, p, o] facts back to original candidate_facts indices via difflib."""
    if not generated_facts or not candidate_facts:
        return []

    # Compare textual coordinates, not representation tags such as np.str_(...).
    candidate_values = [[str(value) for value in fact] for fact in candidate_facts]
    candidate_str_list = [str(fact) for fact in candidate_values]

    result_indices: list[int] = []
    for generated_fact in generated_facts:
        gen_str = str([str(value) for value in generated_fact])
        matches = difflib.get_close_matches(gen_str, candidate_str_list, n=1, cutoff=0.0)
        if matches:
            idx = candidate_str_list.index(matches[0])
            if idx not in result_indices:
                result_indices.append(idx)

    if max_facts is not None:
        return result_indices[:max_facts]
    return result_indices


def coerce_generated_facts(raw_output: Any) -> list[list[str]]:
    """Extract 3-element string lists from a FactList, dict, JSON string, or list."""
    candidates: Any = None
    if hasattr(raw_output, "fact"):
        candidates = raw_output.fact
    elif isinstance(raw_output, dict) and "fact" in raw_output:
        candidates = raw_output["fact"]
    elif isinstance(raw_output, list):
        candidates = raw_output
    elif isinstance(raw_output, str):
        text = raw_output.strip()
        if "[[ ## fact_after_filter ## ]]" in text:
            text = text.split("[[ ## fact_after_filter ## ]]", 1)[1]
            text = text.split("[[ ##", 1)[0].strip()
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError, ValueError:
            try:
                loaded = ast.literal_eval(text)
            except ValueError, SyntaxError:
                loaded = None
        if isinstance(loaded, dict) and "fact" in loaded:
            candidates = loaded["fact"]
        elif isinstance(loaded, list):
            candidates = loaded
    else:
        return []

    valid: list[list[str]] = []
    if isinstance(candidates, list):
        for item in candidates:
            if isinstance(item, (list, tuple)) and len(item) == 3:
                valid.append([str(x) for x in item])
    return valid


def normalize_answer(answer: str) -> str:
    """Normalize a given string for exact match and F1 comparison."""
    return str(normalize_text(str(answer)))


def compute_qa_em_f1(pred_str: str, gold_answers: Sequence[str]) -> tuple[float, float]:
    """Use DSPy's normalized exact match and continuous best-reference token F1.

    ``answer_exact_match(frac < 1)`` thresholds F1 into a boolean; it does not
    return the continuous score needed by the optimization objective.
    """
    if not gold_answers:
        return 0.0, 0.0
    answers = list(gold_answers)
    example = dspy.Example(answer=answers)
    prediction = dspy.Prediction(answer=pred_str)
    return float(dspy.evaluate.answer_exact_match(example, prediction)), float(F1(pred_str, answers))


_OMITTED_LINES_RE = re.compile(r"\n?\[\.\.\.\s*lines\s+\d+-\d+\s+omitted\s*\.\.\.\]\n?")


def doc_matches_gold(retrieved_doc: str, gold_doc: str) -> bool:
    """Canonical passage-to-gold matcher handling AST island crop markers, title prefixes, and body containment."""
    if retrieved_doc == gold_doc:
        return True
    raw_ret = retrieved_doc.split("\n[... AST section cropped", 1)[0].strip()
    clean_gold = gold_doc.strip()
    if not raw_ret or not clean_gold:
        return False
    segments = [s.strip() for s in _OMITTED_LINES_RE.split(raw_ret) if s.strip()]
    clean_ret = "\n".join(segments) if segments else ""
    if not clean_ret:
        return False
    if clean_ret == clean_gold:
        return True
    if clean_ret.startswith("chunk-") and clean_ret == compute_mdhash_id(clean_gold, prefix="chunk-"):
        return True
    if clean_gold.startswith("chunk-") and clean_gold == compute_mdhash_id(clean_ret, prefix="chunk-"):
        return True

    # Strip standard titles/prefixes for content comparison
    for pfx in ("Wikipedia Title: ", "Title: "):
        if clean_ret.startswith(pfx):
            clean_ret = clean_ret[len(pfx) :].strip()
        if clean_gold.startswith(pfx):
            clean_gold = clean_gold[len(pfx) :].strip()
        if segments and segments[0].startswith(pfx):
            segments[0] = segments[0][len(pfx) :].strip()

    if clean_ret == clean_gold:
        return True

    ws_ret = " ".join(clean_ret.split())
    ws_gold = " ".join(clean_gold.split())
    if ws_ret == ws_gold:
        return True

    if len(segments) > 1:
        ws_segs = [" ".join(seg.split()) for seg in segments if " ".join(seg.split())]
        if ws_segs and sum(len(s) for s in ws_segs) >= 20 and all(s in ws_gold for s in ws_segs):
            return True
        if "\n" in segments[0]:
            seg0_body = " ".join(segments[0].split("\n", 1)[1].split())
            body_ws_segs = [seg0_body, *ws_segs[1:]] if seg0_body else ws_segs[1:]
            if body_ws_segs and sum(len(s) for s in body_ws_segs) >= 20 and all(s in ws_gold for s in body_ws_segs):
                return True

    # Check containment in both directions
    if len(ws_ret) >= 20 and (ws_ret in ws_gold or ws_gold in ws_ret):
        return True

    # Check body-only without title line
    if "\n" in clean_ret:
        body_only = " ".join(clean_ret.split("\n", 1)[1].split())
        if len(body_only) >= 20 and (body_only in ws_gold or ws_gold in body_only):
            return True

    if "\n" in clean_gold:
        gold_body_only = " ".join(clean_gold.split("\n", 1)[1].split())
        if len(gold_body_only) >= 20 and gold_body_only in ws_ret:
            return True

    return False


def compute_retrieval_recalls(
    gold_docs: Sequence[str],
    retrieved_docs: Sequence[str],
    k_list: Sequence[int] = (1, 2, 5, 10, 20),
) -> dict[int, float]:
    """Compute Recall@k for a single query using `doc_matches_gold`."""
    gold_seq: Sequence[str] = [gold_docs] if isinstance(gold_docs, str) else (gold_docs or ())
    ret_seq: Sequence[str] = [retrieved_docs] if isinstance(retrieved_docs, str) else (retrieved_docs or ())
    unique_gold = [str(g) for g in dict.fromkeys(gold_seq) if str(g).strip()]
    recalls: dict[int, float] = {}
    for k in k_list:
        k_int = int(k)
        if not unique_gold or k_int < 1:
            recalls[k_int] = 0.0
            continue
        top_k_docs = ret_seq[:k_int]
        matched_gold = sum(1 for g in unique_gold if any(doc_matches_gold(str(r), g) for r in top_k_docs))
        recalls[k_int] = matched_gold / len(unique_gold)
    return recalls


def compute_answer_passage_match(
    gold_answers: Sequence[str],
    context: str | Sequence[str],
) -> bool:
    """Return True if any normalized gold answer appears in the normalized retrieved context."""
    raw_answers: Sequence[str] = [gold_answers] if isinstance(gold_answers, str) else (gold_answers or ())
    clean_answers = [str(a).strip() for a in raw_answers if str(a).strip()]
    if not clean_answers:
        return False
    if isinstance(context, str):
        passages = [p for p in context.split("\n\n") if p.strip()]
    elif isinstance(context, Sequence):
        passages = [str(p) for p in context if str(p).strip()]
    else:
        passages = []
    if not passages:
        return False
    ex_apm = dspy.Example(answer=clean_answers)
    pred_apm = dspy.Prediction(context=passages)
    return bool(dspy.evaluate.answer_passage_match(ex_apm, pred_apm))


_SYNONYM_RELATION_ALIASES: dict[str, str] = {
    "skos:altLabel": "skos:altLabel",
    "altLabel": "skos:altLabel",
    "exact_match": "skos:altLabel",
    "skos:hiddenLabel": "skos:hiddenLabel",
    "hiddenLabel": "skos:hiddenLabel",
    "skos:related": "skos:related",
    "related": "skos:related",
    "skos:broader": "skos:broader",
    "broader": "skos:broader",
    "skos:narrower": "skos:narrower",
    "narrower": "skos:narrower",
    "owl:differentFrom": "owl:differentFrom",
    "differentFrom": "owl:differentFrom",
}
VALID_SYNONYM_RELATIONS: frozenset[str] = frozenset(
    {
        "skos:altLabel",
        "skos:hiddenLabel",
        "skos:related",
        "skos:broader",
        "owl:differentFrom",
    }
)
VALID_SHACL_ACTIONS: frozenset[str] = frozenset(
    {
        "set_definition",
        "rewrite_definition",
        "set_pref_label",
        "demote_to_alt_label",
        "merge_altLabel",
        "merge_hiddenLabel",
        "add_different_from",
        "assert_differentFrom",
        "assert_related",
        "assert_broader",
        "remove_hierarchy_cycle",
        "remove_conflicting_related",
        "delete_dangling",
        "delete_invalid_fact",
    }
)
VALUE_BEARING_SHACL_ACTIONS: frozenset[str] = frozenset(
    {
        "set_definition",
        "rewrite_definition",
        "set_pref_label",
    }
)


def parse_synonym_decisions(
    raw_output: Any,
    num_pairs: int | None = None,
) -> list[dict[str, str]]:
    """Parse and validate SKOS/OWL disambiguation decisions from JudgeSynonyms or OrchestrateSynonymJudgment."""
    items: Sequence[Any] = ()
    if hasattr(raw_output, "decisions"):
        items = raw_output.decisions
    elif isinstance(raw_output, dict) and "decisions" in raw_output:
        items = raw_output["decisions"]
    elif isinstance(raw_output, str):
        try:
            items = _extract_json_list_field(raw_output, "decisions")
        except ValueError:
            try:
                loaded = json.loads(raw_output)
                items = loaded if isinstance(loaded, list) else ()
            except Exception:
                items = ()
    elif isinstance(raw_output, Sequence):
        items = raw_output

    decisions: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        pair_id = str(item.get("pair_id", "")).strip()
        raw_relation = str(item.get("relation", "")).strip()
        relation = _SYNONYM_RELATION_ALIASES.get(raw_relation, "")
        canonical = str(item.get("canonical", "")).strip()
        if relation == "skos:narrower":
            relation = "skos:broader"
            canonical = "B" if canonical in ("", "A") else "A"
        if not pair_id or relation not in VALID_SYNONYM_RELATIONS:
            continue
        decisions.append(
            {
                "pair_id": pair_id,
                "relation": relation,
                "canonical": canonical or "A",
            }
        )
        if num_pairs is not None and num_pairs > 0 and len(decisions) >= num_pairs:
            break
    return decisions


def parse_shacl_repairs(raw_output: Any) -> list[dict[str, str]]:
    """Parse and validate closed-world SHACL repair actions from ProposeSHACLRepairs or HealSHACLViolations."""
    items: Sequence[Any] = ()
    if hasattr(raw_output, "repairs"):
        items = raw_output.repairs
    elif isinstance(raw_output, dict) and "repairs" in raw_output:
        items = raw_output["repairs"]
    elif isinstance(raw_output, str):
        try:
            items = _extract_json_list_field(raw_output, "repairs")
        except ValueError:
            try:
                loaded = json.loads(raw_output)
                items = loaded if isinstance(loaded, list) else ()
            except Exception:
                items = ()
    elif isinstance(raw_output, Sequence):
        items = raw_output

    repairs: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        focus_node = str(item.get("focus_node", "")).strip()
        action = str(item.get("action", "")).strip()
        value = str(item.get("value", item.get("target_node", item.get("target", ""))) or "").strip()
        if not focus_node or action not in VALID_SHACL_ACTIONS:
            continue
        repairs.append(
            {
                "focus_node": focus_node,
                "action": action,
                "value": value,
            }
        )
    return repairs


def _compute_metric_confidence(yaw_val: float, base_val: float, n: int) -> dict[str, Any]:
    """Compute binomial normal-approximation standard error, z-score, and parity/difference confidence."""
    n_eff = max(1, int(n))
    delta_k = float(yaw_val) - float(base_val)
    p_k = min(1.0, max(0.0, float(yaw_val)))
    p0_k = min(1.0, max(0.0, float(base_val)))
    se_k = math.sqrt(max((p_k * (1.0 - p_k) + p0_k * (1.0 - p0_k)) / n_eff, 1e-9))
    z_k = 0.0 if abs(delta_k) < 1e-12 else delta_k / se_k
    c_k = round(0.5 * (1.0 + math.erf(z_k / math.sqrt(2.0))), 4)
    c_diff_k = round(math.erf(abs(z_k) / math.sqrt(2.0)), 4)
    ci_95 = [round(delta_k - 1.96 * se_k, 4), round(delta_k + 1.96 * se_k, 4)]
    return {
        "n": n_eff,
        "delta": round(delta_k, 4),
        "se": round(se_k, 4),
        "z": round(z_k, 4),
        "confidence": c_k,
        "parity_confidence": c_k,
        "difference_confidence": c_diff_k,
        "diff_confidence": c_diff_k,
        "ci_95": ci_95,
    }


def evaluate_benchmark_gate(
    dataset_reports: Mapping[str, Mapping[str, Any]],
    gate_spec: Mapping[str, Any],
    *,
    mode: Literal["eval", "retrieval"] = "eval",
    target: str | None = None,
) -> dict[str, Any]:
    """Evaluate per-dataset benchmark reports against a benchmark_gate.json specification."""
    aliases: dict[str, str] = {
        "nq_rear": "nq",
        "naturalquestions": "nq",
        "2wiki": "2wikimultihopqa",
        "hotpot": "hotpotqa",
        "narrative": "narrativeqa",
        "narrativeqa_dev_10_doc": "narrativeqa",
        "enterprise": "enterprise_rag_bench",
        "popqa": "popqa",
        "lveval": "lveval",
        "narrativeqa": "narrativeqa",
        "2wikimultihopqa": "2wikimultihopqa",
        "hotpotqa": "hotpotqa",
        "nq": "nq",
        "musique": "musique",
        "sample": "sample",
        "case_study_university": "case_study_university",
        "enterprise_rag_bench": "enterprise_rag_bench",
    }
    raw_aliases = gate_spec.get("aliases")
    if isinstance(raw_aliases, Mapping):
        for ak, av in raw_aliases.items():
            aliases[str(ak).strip().lower()] = str(av).strip()

    effective_mode = str(mode or gate_spec.get("mode", "eval")).strip().lower()
    tolerance = float(gate_spec.get("per_dataset_tolerance", gate_spec.get("tolerance", 0.0)))
    primary_target = str(target or gate_spec.get("primary_target", "monkeypatched_hipporag"))
    spec_datasets = gate_spec.get("datasets", {})
    if not isinstance(spec_datasets, Mapping):
        spec_datasets = {}
    targets_map = gate_spec.get("targets", {})
    target_datasets: Mapping[str, Any] = {}
    if isinstance(targets_map, Mapping):
        t_entry = targets_map.get(primary_target, {})
        if isinstance(t_entry, Mapping) and isinstance(t_entry.get("datasets"), Mapping):
            target_datasets = t_entry["datasets"]

    spec_datasets_lower = {str(k).strip().lower(): v for k, v in spec_datasets.items()}
    target_datasets_lower = {str(k).strip().lower(): v for k, v in target_datasets.items()}

    dataset_evaluations: dict[str, dict[str, Any]] = {}
    sample_counts: dict[str, int] = {}
    optional_qa_values: dict[str, dict[str, list[float | None]]] = {
        "yaw_hipporag": {"ExactMatch": [], "F1": []},
        "monkeypatched_hipporag": {"ExactMatch": [], "F1": []},
        "delta_yaw_vs_hipporag": {"ExactMatch": [], "F1": []},
    }
    for raw_name, rep in dataset_reports.items():
        if not isinstance(rep, Mapping):
            continue
        raw_stripped = str(raw_name).strip()
        cleaned = raw_stripped.lower()
        if cleaned in aliases:
            canonical = aliases[cleaned]
        elif cleaned in spec_datasets or cleaned in target_datasets or raw_stripped == cleaned:
            canonical = cleaned
        else:
            canonical = raw_stripped

        eval_metrics = rep.get("evaluation_metrics", {})
        if isinstance(eval_metrics, Mapping) and "hipporag" in eval_metrics:
            hippo_block = eval_metrics.get("hipporag", {})
            ret_block = hippo_block.get("retrieval", {}) if isinstance(hippo_block, Mapping) else {}
            raw_qa = hippo_block.get("qa") if isinstance(hippo_block, Mapping) else None
            qa_block = raw_qa if isinstance(raw_qa, Mapping) else {}
            dpr_block = eval_metrics.get("standard_dpr", {})
            dpr_ret = dpr_block.get("retrieval", {}) if isinstance(dpr_block, Mapping) else {}
            rec_mem = eval_metrics.get("recognition_memory", {}) if isinstance(eval_metrics, Mapping) else {}
        elif "yaw_hipporag" in rep and isinstance(rep["yaw_hipporag"], Mapping):
            yh = rep["yaw_hipporag"]
            ret_block = yh
            qa_block = yh
            dpr_ret = rep.get("yaw_dpr", {}) if isinstance(rep.get("yaw_dpr"), Mapping) else {}
            rec_mem = rep.get("recognition_memory", {}) if isinstance(rep.get("recognition_memory"), Mapping) else {}
        elif "retrieval" in rep and isinstance(rep["retrieval"], Mapping):
            ret_block = rep["retrieval"]
            raw_qa = rep.get("qa")
            qa_block = raw_qa if isinstance(raw_qa, Mapping) else {}
            dpr_ret = rep.get("standard_dpr", {}) if isinstance(rep.get("standard_dpr"), Mapping) else {}
            rec_mem = rep.get("recognition_memory", {}) if isinstance(rep.get("recognition_memory"), Mapping) else {}
        else:
            ret_block = rep
            qa_block = rep
            dpr_ret = rep.get("dpr", {}) if isinstance(rep.get("dpr"), Mapping) else {}
            rec_mem = {}

        yaw_em_raw = qa_block.get("ExactMatch") if isinstance(qa_block, Mapping) else None
        yaw_f1_raw = qa_block.get("F1") if isinstance(qa_block, Mapping) else None
        yaw_em_explicit_none = isinstance(qa_block, Mapping) and "ExactMatch" in qa_block and yaw_em_raw is None
        yaw_f1_explicit_none = isinstance(qa_block, Mapping) and "F1" in qa_block and yaw_f1_raw is None
        yaw_em_opt = round(float(yaw_em_raw), 4) if yaw_em_raw is not None else None
        yaw_f1_opt = round(float(yaw_f1_raw), 4) if yaw_f1_raw is not None else None
        yaw_metrics = {
            "Recall@1": round(float(ret_block.get("Recall@1", 0.0) or 0.0), 4),
            "Recall@2": round(float(ret_block.get("Recall@2", 0.0) or 0.0), 4),
            "Recall@5": round(float(ret_block.get("Recall@5", 0.0) or 0.0), 4),
            "ExactMatch": yaw_em_opt if yaw_em_opt is not None else 0.0,
            "F1": yaw_f1_opt if yaw_f1_opt is not None else 0.0,
        }
        dpr_metrics = {
            "Recall@1": round(float(dpr_ret.get("Recall@1", 0.0) or 0.0), 4),
            "Recall@2": round(float(dpr_ret.get("Recall@2", 0.0) or 0.0), 4),
            "Recall@5": round(float(dpr_ret.get("Recall@5", 0.0) or 0.0), 4),
        }

        ds_spec = (
            spec_datasets.get(canonical)
            or spec_datasets.get(cleaned)
            or spec_datasets.get(raw_stripped)
            or spec_datasets_lower.get(canonical.lower())
            or spec_datasets_lower.get(cleaned)
            or (spec_datasets.get("nq_rear") if canonical == "nq" else None)
            or target_datasets.get(canonical)
            or target_datasets.get(cleaned)
            or target_datasets.get(raw_stripped)
            or target_datasets_lower.get(canonical.lower())
            or target_datasets_lower.get(cleaned)
            or {}
        )
        if not isinstance(ds_spec, Mapping):
            ds_spec = {}

        base_block = ds_spec.get(primary_target, ds_spec.get("monkeypatched_hipporag", ds_spec))
        if not isinstance(base_block, Mapping):
            base_block = {}
        paper_block = ds_spec.get("paper_70b", {})
        if not isinstance(paper_block, Mapping):
            paper_block = {}
        elif isinstance(paper_block.get("hipporag2_llama3.3_70b_nvembed_v2"), Mapping):
            paper_block = paper_block["hipporag2_llama3.3_70b_nvembed_v2"]

        base_em_raw = base_block.get("ExactMatch")
        base_f1_raw = base_block.get("F1")
        base_em_opt = round(float(base_em_raw), 4) if base_em_raw is not None else None
        base_f1_opt = round(float(base_f1_raw), 4) if base_f1_raw is not None else None
        hippo_metrics = {
            "Recall@1": round(float(base_block.get("Recall@1", 0.0) or 0.0), 4),
            "Recall@2": round(float(base_block.get("Recall@2", 0.0) or 0.0), 4),
            "Recall@5": round(float(base_block.get("Recall@5", 0.0) or 0.0), 4),
            "ExactMatch": base_em_opt if base_em_opt is not None else 0.0,
            "F1": base_f1_opt if base_f1_opt is not None else 0.0,
        }
        paper_metrics = {
            k: (round(float(paper_block[k]), 4) if paper_block.get(k) is not None else None)
            for k in ("Recall@2", "Recall@5", "ExactMatch", "F1")
        }

        delta_vs_hippo = {
            k: round(yaw_metrics[k] - hippo_metrics[k], 4)
            for k in ("Recall@1", "Recall@2", "Recall@5", "ExactMatch", "F1")
        }
        delta_vs_paper = {
            "Recall@2": (
                round(yaw_metrics["Recall@2"] - pv, 4) if (pv := paper_metrics["Recall@2"]) is not None else None
            ),
            "Recall@5": (
                round(yaw_metrics["Recall@5"] - pv, 4) if (pv := paper_metrics["Recall@5"]) is not None else None
            ),
            "ExactMatch": (
                round(yaw_em_opt - pv, 4)
                if yaw_em_opt is not None and (pv := paper_metrics["ExactMatch"]) is not None
                else None
            ),
            "F1": (
                round(yaw_f1_opt - pv, 4)
                if yaw_f1_opt is not None and (pv := paper_metrics["F1"]) is not None
                else None
            ),
        }
        delta_vs_dpr = {k: round(yaw_metrics[k] - dpr_metrics[k], 4) for k in ("Recall@1", "Recall@2", "Recall@5")}

        eval_em_enforced = base_em_opt is not None and not yaw_em_explicit_none
        eval_f1_enforced = base_f1_opt is not None and not yaw_f1_explicit_none
        optional_qa_values["yaw_hipporag"]["ExactMatch"].append(yaw_em_opt)
        optional_qa_values["yaw_hipporag"]["F1"].append(yaw_f1_opt)
        optional_qa_values["monkeypatched_hipporag"]["ExactMatch"].append(base_em_opt)
        optional_qa_values["monkeypatched_hipporag"]["F1"].append(base_f1_opt)
        optional_qa_values["delta_yaw_vs_hipporag"]["ExactMatch"].append(
            delta_vs_hippo["ExactMatch"] if eval_em_enforced else None
        )
        optional_qa_values["delta_yaw_vs_hipporag"]["F1"].append(
            delta_vs_hippo["F1"] if eval_f1_enforced else None
        )

        failures: list[str] = []
        integrity = rep.get("integrity", {})
        if isinstance(integrity, Mapping) and not integrity.get("passed", True):
            int_errors = integrity.get("errors", [])
            failures.append(f"Integrity gate failed: {int_errors}")

        if delta_vs_hippo["Recall@5"] < tolerance - 1e-9:
            failures.append(
                f"Recall@5 delta vs historical HippoRAG ({delta_vs_hippo['Recall@5']:+.4f}) < {tolerance:+.2f}"
            )
        if effective_mode == "eval":
            if eval_f1_enforced and delta_vs_hippo["F1"] < tolerance - 1e-9:
                failures.append(f"QA F1 delta vs historical HippoRAG ({delta_vs_hippo['F1']:+.4f}) < {tolerance:+.2f}")
            if eval_em_enforced and delta_vs_hippo["ExactMatch"] < tolerance - 1e-9:
                failures.append(
                    f"ExactMatch delta vs historical HippoRAG ({delta_vs_hippo['ExactMatch']:+.4f}) < {tolerance:+.2f}"
                )

        raw_q = rep.get("num_queries") or rep.get("queries_count")
        if raw_q is None:
            q_field = rep.get("queries")
            if isinstance(q_field, Sequence) and not isinstance(q_field, (str, bytes)):
                raw_q = len(q_field)
            elif isinstance(q_field, (int, float)):
                raw_q = int(q_field)
            elif isinstance(rep.get("records"), Sequence) and not isinstance(rep.get("records"), (str, bytes)):
                raw_q = len(rep["records"])
        queries_count = int(raw_q if raw_q is not None else gate_spec.get("queries_per_dataset", 0))
        n_queries = max(1, int(raw_q or gate_spec.get("queries_per_dataset", 25) or 25))
        sample_counts[canonical] = n_queries

        graph_block = rep.get("graph", {})
        raw_docs = rep.get("docs", rep.get("num_docs", 0))
        docs_fallback = (
            len(raw_docs)
            if isinstance(raw_docs, Sequence) and not isinstance(raw_docs, (str, bytes))
            else int(raw_docs or 0)
        )
        passage_nodes = (
            int(graph_block.get("num_passage_nodes", docs_fallback) or docs_fallback)
            if isinstance(graph_block, Mapping)
            else docs_fallback
        )

        ds_confidence = {
            k: _compute_metric_confidence(yaw_metrics[k], hippo_metrics[k], n_queries)
            for k in ("Recall@5", "ExactMatch", "F1")
        }

        dataset_evaluations[canonical] = {
            "dataset": canonical,
            "queries": queries_count,
            "docs": passage_nodes,
            "yaw_hipporag": yaw_metrics,
            "monkeypatched_hipporag": hippo_metrics,
            "paper_70b": paper_metrics,
            "yaw_dpr": dpr_metrics,
            "recognition_memory": dict(rec_mem) if isinstance(rec_mem, Mapping) else {},
            "delta_yaw_vs_hipporag": delta_vs_hippo,
            "delta_yaw_vs_paper": delta_vs_paper,
            "delta_yaw_vs_dpr": delta_vs_dpr,
            "confidence": ds_confidence,
            "gate_passed": len(failures) == 0,
            "gate_status": "PASS" if len(failures) == 0 else "FAIL",
            "failure_reasons": failures,
        }

    completed = list(dataset_evaluations.values())

    def _mean(vals: Sequence[float | None]) -> float:
        valid = [float(v) for v in vals if v is not None]
        return round(sum(valid) / len(valid), 4) if valid else 0.0

    macro_means: dict[str, Any] = {
        "yaw_hipporag": {
            "Recall@1": _mean([d["yaw_hipporag"]["Recall@1"] for d in completed]),
            "Recall@2": _mean([d["yaw_hipporag"]["Recall@2"] for d in completed]),
            "Recall@5": _mean([d["yaw_hipporag"]["Recall@5"] for d in completed]),
            "ExactMatch": _mean(optional_qa_values["yaw_hipporag"]["ExactMatch"]),
            "F1": _mean(optional_qa_values["yaw_hipporag"]["F1"]),
        },
        "monkeypatched_hipporag": {
            "Recall@1": _mean([d["monkeypatched_hipporag"]["Recall@1"] for d in completed]),
            "Recall@2": _mean([d["monkeypatched_hipporag"]["Recall@2"] for d in completed]),
            "Recall@5": _mean([d["monkeypatched_hipporag"]["Recall@5"] for d in completed]),
            "ExactMatch": _mean(optional_qa_values["monkeypatched_hipporag"]["ExactMatch"]),
            "F1": _mean(optional_qa_values["monkeypatched_hipporag"]["F1"]),
        },
        "paper_70b": {
            k: _mean([d["paper_70b"][k] for d in completed]) for k in ("Recall@2", "Recall@5", "ExactMatch", "F1")
        },
        "yaw_dpr": {k: _mean([d["yaw_dpr"][k] for d in completed]) for k in ("Recall@1", "Recall@2", "Recall@5")},
        "delta_yaw_vs_hipporag": {
            "Recall@1": _mean([d["delta_yaw_vs_hipporag"]["Recall@1"] for d in completed]),
            "Recall@2": _mean([d["delta_yaw_vs_hipporag"]["Recall@2"] for d in completed]),
            "Recall@5": _mean([d["delta_yaw_vs_hipporag"]["Recall@5"] for d in completed]),
            "ExactMatch": _mean(optional_qa_values["delta_yaw_vs_hipporag"]["ExactMatch"]),
            "F1": _mean(optional_qa_values["delta_yaw_vs_hipporag"]["F1"]),
        },
        "delta_yaw_vs_paper": {
            k: _mean([d["delta_yaw_vs_paper"][k] for d in completed])
            for k in ("Recall@2", "Recall@5", "ExactMatch", "F1")
        },
        "delta_yaw_vs_dpr": {
            k: _mean([d["delta_yaw_vs_dpr"][k] for d in completed]) for k in ("Recall@1", "Recall@2", "Recall@5")
        },
    }

    total_queries = sum(sample_counts.values())
    pooled_confidence = (
        {
            k: _compute_metric_confidence(
                macro_means["yaw_hipporag"][k],
                macro_means["monkeypatched_hipporag"][k],
                total_queries,
            )
            for k in ("Recall@5", "ExactMatch", "F1")
        }
        if completed
        else {}
    )
    macro_means["confidence"] = pooled_confidence

    summary_failures: list[str] = []
    if not completed:
        summary_failures.append("No datasets completed")
    for d in completed:
        if not d["gate_passed"]:
            summary_failures.append(f"{d['dataset']}: {'; '.join(d['failure_reasons'])}")

    if completed:
        if bool(gate_spec.get("require_mean_recall5_parity", True)):
            mean_r5_delta = macro_means["delta_yaw_vs_hipporag"]["Recall@5"]
            if mean_r5_delta < -1e-9:
                summary_failures.append(
                    f"Macro-average Recall@5 delta vs historical HippoRAG ({mean_r5_delta:+.4f}) < 0.0000"
                )
        if effective_mode == "eval":
            has_f1_spec = any(
                isinstance(d_spec, Mapping)
                and isinstance(
                    b := d_spec.get(primary_target, d_spec.get("monkeypatched_hipporag", d_spec)),
                    Mapping,
                )
                and b.get("F1") is not None
                for d_spec in (*spec_datasets.values(), *target_datasets.values())
            )
            has_em_spec = any(
                isinstance(d_spec, Mapping)
                and isinstance(
                    b := d_spec.get(primary_target, d_spec.get("monkeypatched_hipporag", d_spec)),
                    Mapping,
                )
                and b.get("ExactMatch") is not None
                for d_spec in (*spec_datasets.values(), *target_datasets.values())
            )
            if bool(gate_spec.get("require_mean_f1_parity", has_f1_spec)):
                mean_f1_delta = macro_means["delta_yaw_vs_hipporag"]["F1"]
                if mean_f1_delta < -1e-9:
                    summary_failures.append(
                        f"Macro-average QA F1 delta vs historical HippoRAG ({mean_f1_delta:+.4f}) < 0.0000"
                    )
            if bool(gate_spec.get("require_mean_em_parity", has_em_spec)):
                mean_em_delta = macro_means["delta_yaw_vs_hipporag"]["ExactMatch"]
                if mean_em_delta < -1e-9:
                    summary_failures.append(
                        f"Macro-average ExactMatch delta vs historical HippoRAG ({mean_em_delta:+.4f}) < 0.0000"
                    )

    overall_passed = len(summary_failures) == 0
    return {
        "passed": overall_passed,
        "overall_gate_passed": overall_passed,
        "overall_gate_status": "PASS" if overall_passed else "FAIL",
        "mode": effective_mode,
        "per_dataset_tolerance": tolerance,
        "datasets_completed": len(completed),
        "datasets_passed": sum(1 for d in completed if d["gate_passed"]),
        "datasets": dataset_evaluations,
        "macro_means": macro_means,
        "confidence": pooled_confidence,
        "failures": summary_failures,
        "failure_reasons": summary_failures,
    }


def _clamp_unit(val: float) -> float:
    return max(0.0, min(1.0, float(val)))


def compute_task_quality_score(
    recall_at_5: float | None,
    *,
    tokens: int,
    recall_at_1: float | None = None,
    recall_at_2: float | None = None,
    recall_at_10: float | None = None,
    recall_at_20: float | None = None,
    em: float | None = None,
    f1: float | None = None,
    shacl_conformance: float | None = None,
    shacl_violations: int | None = None,
    shacl_violations_initial: int = 0,
    shacl_violations_remaining: int = 0,
    synonym_judge_tokens: int = 0,
    shacl_healer_tokens: int = 0,
    missing_entity_defs_count: int = 0,
    missing_entity_defs_tokens: int = 0,
    pred_name: str | None = None,
    missed_items: Sequence[str] = (),
) -> dspy.Prediction:
    """Return the unweighted arithmetic mean of available recall, EM and F1.

    There is no cost denominator, smoothing floor or repeated weight. Absent
    labels contribute no objectives. Tokens and graph integrity remain
    diagnostics only, including in GEPA's per-objective Pareto frontier.
    """
    clamped_recall = _clamp_unit(recall_at_5) if recall_at_5 is not None else None
    safe_tokens = max(int(tokens), 0)
    stage_prefix = f"[{pred_name}] " if pred_name else ""
    raw_metrics = {
        "recall_at_1": recall_at_1,
        "recall_at_2": recall_at_2,
        "recall_at_5": clamped_recall,
        "recall_at_10": recall_at_10,
        "recall_at_20": recall_at_20,
        "em": em,
        "f1": f1,
    }
    obj_scores = {name: _clamp_unit(value) for name, value in raw_metrics.items() if value is not None}
    if not obj_scores:
        raise ValueError("Task quality requires at least one labeled metric")
    score = math.fsum(obj_scores.values()) / len(obj_scores)

    eff_v_rem = max(
        0,
        int(shacl_violations) if shacl_violations is not None else int(shacl_violations_remaining),
    )
    eff_v_init = max(0, int(shacl_violations_initial))
    eff_missing_defs = max(0, int(missing_entity_defs_count))
    eff_missing_def_tokens = max(0, int(missing_entity_defs_tokens))
    g_shacl = _clamp_unit(shacl_conformance) if shacl_conformance is not None else 1.0 / (1.0 + float(eff_v_rem))

    missed_str = f" Missed gold items: {list(missed_items)}." if missed_items else ""
    missing_defs_str = (
        f" [entity_definer] repaired {eff_missing_defs} missing Stage-2 entity definition(s) "
        f"({eff_missing_def_tokens} aux tokens); emit inline [name, def] pairs in Stage 2 or extract in Stage 1."
        if eff_missing_defs > 0
        else ""
    )
    quality_details = ", ".join(f"{name}={value:.4f}" for name, value in obj_scores.items())
    feedback = (
        f"{stage_prefix}Task quality arithmetic mean={score:.6f} ({quality_details}). "
        "Improve retrieval and answer correctness; tokens and graph repairs do not affect reward. "
        f"Diagnostics: {safe_tokens} tokens, SHACL={g_shacl:.4f}."
        f"{missed_str}{missing_defs_str}"
    )

    return dspy.Prediction(
        score=score,
        feedback=feedback,
        gm_good=score,
        shacl_conforms=g_shacl,
        recall_at_1=obj_scores.get("recall_at_1"),
        recall_at_2=obj_scores.get("recall_at_2"),
        recall_at_5=clamped_recall,
        recall_at_10=obj_scores.get("recall_at_10"),
        recall_at_20=obj_scores.get("recall_at_20"),
        em=obj_scores.get("em"),
        f1=obj_scores.get("f1"),
        tokens=safe_tokens,
        synonym_judge_tokens=max(0, int(synonym_judge_tokens)),
        shacl_healer_tokens=max(0, int(shacl_healer_tokens)),
        shacl_violations_initial=eff_v_init,
        shacl_violations_remaining=eff_v_rem,
        missing_entity_defs_count=eff_missing_defs,
        missing_entity_defs_tokens=eff_missing_def_tokens,
        objective_scores=obj_scores,
    )


def check_benchmark_integrity(
    records: Sequence[Mapping[str, Any]],
    graph_info: Mapping[str, Any],
    expected_doc_count: int | None = None,
    extraction_errors: Sequence[str] = (),
    qa_errors: Sequence[str] = (),
    include_qa: bool = True,
) -> list[str]:
    """Fail-fast system integrity checks on indexing, chat completions, and PPR activation."""
    errors: list[str] = []
    if extraction_errors:
        errors.extend(f"extraction_llm: {err}" for err in extraction_errors)
    if include_qa and qa_errors:
        errors.extend(f"qa_llm: {err}" for err in qa_errors)
    if expected_doc_count is not None and graph_info.get("num_passage_nodes") != expected_doc_count:
        errors.append(f"Index passage count mismatch: {graph_info.get('num_passage_nodes')} != {expected_doc_count}")
    if not (graph_info.get("num_extracted_triples") or graph_info.get("num_fact_nodes")):
        errors.append("No triples extracted during OpenIE indexing")
    if include_qa:
        for idx, record in enumerate(records):
            if not str(record.get("answer", "")).strip():
                errors.append(f"Question {idx + 1}: empty HippoRAG answer")
            if (
                "dpr_answer" in record
                and record["dpr_answer"] is not None
                and str(record.get("dpr_answer", "")) == ""
                and record.get("dpr_qa_metrics") is not None
            ):
                errors.append(f"Question {idx + 1}: empty StandardRAG (DPR) answer")
    if len(records) > 1 and not any(record.get("graph_seeds") for record in records):
        errors.append("All queries had empty graph_seeds (recognition memory / PPR never activated)")
    return errors


__all__ = [
    "VALID_SHACL_ACTIONS",
    "VALID_SYNONYM_RELATIONS",
    "VALUE_BEARING_SHACL_ACTIONS",
    "check_benchmark_integrity",
    "coerce_generated_facts",
    "compute_answer_passage_match",
    "compute_mdhash_id",
    "compute_qa_em_f1",
    "compute_retrieval_recalls",
    "compute_task_quality_score",
    "doc_matches_gold",
    "evaluate_benchmark_gate",
    "extract_entity_definitions",
    "extract_named_entities",
    "extract_structured_triples",
    "extract_triples",
    "filter_invalid_triples",
    "match_facts_to_candidate_indices",
    "normalize_answer",
    "parse_shacl_repairs",
    "parse_synonym_decisions",
    "repair_truncated_json",
    "text_processing",
]
