"""Closed-world SHACL/SKOS stage invariant validation."""

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from yaw.core import (
    VALID_SHACL_ACTIONS,
    VALUE_BEARING_SHACL_ACTIONS,
    is_irreflexive_statement,
    text_processing,
)

from .parsers import (
    coerce_generated_facts,
    extract_entity_definitions,
    extract_structured_triples,
    parse_shacl_repairs,
)

_ABSTAIN_PATTERNS: tuple[str, ...] = (
    "i don't know",
    "i do not know",
    "unknown",
    "not enough information",
    "insufficient information",
    "insufficient context",
    "cannot answer",
    "cannot be determined",
    "unable to answer",
    "no information provided",
    "not mentioned in the context",
)


def is_abstaining_answer(answer: str) -> bool:
    """Return True if `answer` is empty or expresses abstention/uncertainty."""
    cleaned = str(answer or "").strip().lower().rstrip(".!")
    if not cleaned or cleaned in ("none", "n/a", "null"):
        return True
    return any(pat in cleaned for pat in _ABSTAIN_PATTERNS)


STAGE_SCHEMA_CONTRACTS: dict[str, tuple[str, tuple[str, ...], str]] = {
    "ner": (
        "named_entities",
        ("named_entities", "named entities", "entities"),
        "Populate `named_entities` with deduplicated resource label and contextual definition pairs.",
    ),
    "triples": (
        "triples",
        ("triples", "statements"),
        "Populate `triples` with passage-supported reified `rdf:Statement` facts.",
    ),
    "entity_definer": (
        "entity_definitions",
        ("entity_definitions", "entity definitions"),
        "Populate `entity_definitions` with contextual definitions paired with each requested entity name.",
    ),
    "label_judge": (
        "decisions",
        ("decisions",),
        (
            "Populate `decisions` with identity equivalence decisions, supplying a unified definition "
            "and `hidden_label` flag only when equivalent."
        ),
    ),
    "relation_judge": (
        "decisions",
        ("decisions",),
        "Populate decisions with one oriented relation per pair.",
    ),
    "shacl_repair": (
        "repairs",
        ("repairs",),
        "Populate `repairs` with closed-world SHACL repair actions for the reported focus nodes.",
    ),
    "shacl_healer_rlm": (
        "repairs",
        ("repairs",),
        "Populate `repairs` with closed-world SHACL repair actions for the reported focus nodes.",
    ),
    "triple_filter": (
        "fact",
        ("fact", "facts"),
        "Populate `fact` with up to 4 verbatim subject-predicate-object triples from `fact_before_filter`.",
    ),
    "query_intent": (
        "query_intent",
        ("query_intent", "query intent"),
        "Populate `query_intent` with a concise declarative statement of the sought fact.",
    ),
    "qa": (
        "answer",
        ("answer",),
        "Populate `answer` with a definitive closed-world answer grounded in `context`.",
    ),
}


def validate_stage_invariants(
    stage_name: str,
    inputs: Mapping[str, Any],
    prediction: Any,
    *,
    shacl_linter: Callable[[], Sequence[Mapping[str, Any]]] | None = None,
) -> tuple[float, str]:
    """Validate stage contracts and closed-world SHACL/SKOS invariants.

    Returns `(score, feedback)` where `score` is in `[0.0, 1.0]` (`1.0` = full pass).
    """
    stage = str(stage_name or "").strip()

    if stage == "ner":
        raw_entities = getattr(prediction, "named_entities", None)
        passage_str = str(inputs.get("passage", "") or "").strip()
        items = list(raw_entities) if isinstance(raw_entities, (list, tuple)) else []
        if not items:
            if len(passage_str) >= 20:
                return (
                    0.0,
                    "[ner] Empty named_entities on informative passage (ConceptShape/prefLabel).",
                )
            return 1.0, "PASS [ner]"

        seen_defs_by_norm: dict[str, set[str]] = {}
        dup_labels: list[str] = []
        bad_defs: list[str] = []
        valid_count = 0
        for item in items:
            if isinstance(item, (list, tuple)):
                ename = str(item[0]).strip() if len(item) >= 1 else ""
                edef = str(item[1]).strip() if len(item) >= 2 and item[1] is not None else ""
            elif isinstance(item, Mapping):
                ename = str(item.get("entity", item.get("name", ""))).strip()
                edef = str(item.get("definition", item.get("def", ""))).strip()
            else:
                ename = str(item).strip()
                edef = ""
            if not ename:
                bad_defs.append("<empty_name>")
                continue
            norm = str(text_processing(ename)) or ename.lower()
            def_key = edef.lower()
            if norm in seen_defs_by_norm:
                if not def_key or def_key in seen_defs_by_norm[norm] or "" in seen_defs_by_norm[norm]:
                    dup_labels.append(ename)
                else:
                    seen_defs_by_norm[norm].add(def_key)
            else:
                seen_defs_by_norm[norm] = {def_key}
            if not edef or edef.lower() == ename.lower():
                bad_defs.append(ename)
            else:
                valid_count += 1

        issues: list[str] = []
        if dup_labels:
            issues.append(f"ConceptShape/prefLabel duplicate normalized labels: {dup_labels}")
        if bad_defs:
            issues.append(f"ConceptShape/definition missing or tautological definitions: {bad_defs}")
        if not issues:
            return 1.0, "PASS [ner]"
        ratio = valid_count / max(1, len(items))
        penalty = 0.35 * (1 if dup_labels else 0) + 0.45 * (len(bad_defs) / max(1, len(items)))
        score = max(0.0, min(0.85, round(ratio * (1.0 - penalty), 4)))
        return score, f"[ner] {'; '.join(issues)}"

    if stage == "triples":
        raw_triples = getattr(prediction, "triples", None)
        structured = list(extract_structured_triples(raw_triples))
        if not structured:
            return 0.0, "[triples] Empty or unparseable triples (StatementShape/closedWorld)."

        stage1_raw = inputs.get("named_entities") or ()
        stage1_norms: set[str] = set()
        stage1_defs: dict[str, str] = {}
        if isinstance(stage1_raw, Sequence) and not isinstance(stage1_raw, str):
            for ent in stage1_raw:
                ename = str(ent[0]).strip() if isinstance(ent, (list, tuple)) and len(ent) >= 1 else str(ent).strip()
                edef = str(ent[1]).strip() if isinstance(ent, (list, tuple)) and len(ent) >= 2 else ""
                if ename:
                    norm_e = str(text_processing(ename)) or ename.lower()
                    stage1_norms.add(norm_e)
                    if edef:
                        stage1_defs[norm_e] = edef.lower()

        closed_world_norms = set(stage1_norms)

        unregistered: list[str] = []
        self_loops: list[str] = []
        incomplete: int = 0
        for st in structured:
            if not (st.subj and st.pred and st.obj and st.spo_def):
                incomplete += 1
                continue
            s_norm = str(text_processing(st.subj)) or st.subj.lower()
            o_norm = str(text_processing(st.obj)) or st.obj.lower()
            spo_clean = st.spo_def.strip().lower()
            s_def_clean = st.subj_def.strip().lower() or stage1_defs.get(s_norm, "")
            o_def_clean = st.obj_def.strip().lower() or stage1_defs.get(o_norm, "")
            if (s_def_clean and spo_clean == s_def_clean) or (
                st.obj_kind != "literal" and o_def_clean and spo_clean == o_def_clean
            ):
                incomplete += 1
                continue
            if not is_irreflexive_statement(st.subj, st.pred, st.obj, s_def_clean, o_def_clean, st.obj_kind):
                self_loops.append(st.subj)
            if s_norm not in closed_world_norms:
                unregistered.append(st.subj)
            if st.obj_kind != "literal" and o_norm not in closed_world_norms:
                unregistered.append(st.obj)

        issues = []
        if incomplete:
            issues.append(f"StatementShape/closedWorld {incomplete} incomplete triple(s)")
        if self_loops:
            issues.append(f"Self-loop triple(s) on {self_loops}")
        if unregistered:
            issues.append(f"StatementShape/closedWorld resources absent from NER: {unregistered}")

        if not issues:
            return 1.0, "PASS [triples]"
        score = max(0.0, round(1.0 - 0.35 * len(issues), 4))
        return score, f"[triples] {'; '.join(issues)}"

    if stage == "entity_definer":
        missing_in = [str(x).strip() for x in (inputs.get("missing_entities") or ()) if str(x).strip()]
        raw_defs = getattr(
            prediction,
            "entity_definitions",
            getattr(prediction, "named_entities", None),
        )
        parsed = extract_entity_definitions(raw_defs)
        def_by_norm: dict[str, str] = {}
        for ename, edef in parsed:
            if ename and edef and edef.strip().lower() != ename.strip().lower():
                def_by_norm[str(text_processing(ename)) or ename.lower()] = edef.strip()
        if not missing_in:
            return (
                (1.0, "PASS [entity_definer]")
                if def_by_norm
                else (0.0, "[entity_definer] ConceptShape/definition empty.")
            )
        unrepaired = [m for m in missing_in if (str(text_processing(m)) or m.lower()) not in def_by_norm]
        if not unrepaired:
            return 1.0, "PASS [entity_definer]"
        cov = (len(missing_in) - len(unrepaired)) / max(1, len(missing_in))
        return (
            round(cov * 0.5, 4),
            f"[entity_definer] ConceptShape/definition missing or tautological for: {unrepaired}",
        )

    if stage in ("label_judge", "relation_judge"):
        raw_decisions = getattr(prediction, "decisions", None)
        items = list(raw_decisions) if isinstance(raw_decisions, (list, tuple)) else []
        if not items:
            return 0.0, f"[{stage}] Empty decisions (EdgeShape/closedWorld)."
        seen_pairs: set[str] = set()
        issues = []
        packet = inputs.get("candidate_pairs", "")
        try:
            packet = json.loads(packet) if isinstance(packet, str) else packet
        except json.JSONDecodeError:
            packet = None
        expected = (
            {pair["pair_id"] for pair in packet["pairs"]}
            if isinstance(packet, dict)
            and packet.get("format") == "synonym-evidence-v3"
            and isinstance(packet.get("pairs"), list)
            else None
        )

        for d in items:
            if not isinstance(d, Mapping):
                issues.append("EdgeShape/closedWorld non-dict decision")
                continue
            pair_id = str(d.get("pair_id", "") or "").strip()
            reasoning = str(d.get("reasoning", "") or "").strip()
            if not reasoning:
                issues.append(f"Empty reasoning string for {pair_id!r}")
            if expected is not None and (pair_id not in expected or pair_id in seen_pairs):
                issues.append(f"Unknown or duplicate pair {pair_id!r}")
            if "|" not in pair_id:
                issues.append(f"EdgeShape/closedWorld invalid pair_id={pair_id!r}")
                continue
            id_a, id_b = [p.strip() for p in pair_id.split("|", 1)]
            if not id_a or not id_b or id_a == id_b:
                issues.append(f"EdgeShape/closedWorld self-loop or empty pair_id={pair_id!r}")
            if stage == "label_judge":
                if "equivalent" not in d:
                    issues.append(f"Missing boolean field 'equivalent' for {pair_id!r}")
                elif not isinstance(d["equivalent"], bool):
                    issues.append(f"Non-boolean value {d['equivalent']!r} for 'equivalent' in {pair_id!r}")
                elif d["equivalent"] is True:
                    raw_def = d.get("definition")
                    if not isinstance(raw_def, str) or not raw_def.strip():
                        issues.append(f"Missing or empty 'definition' for equivalent=True in {pair_id!r}")
                    if "hidden_label" in d and not isinstance(d["hidden_label"], bool):
                        issues.append(f"Non-boolean value {d['hidden_label']!r} for 'hidden_label' in {pair_id!r}")
                    if "alt_label" in d and not isinstance(d["alt_label"], bool):
                        issues.append(f"Non-boolean value {d['alt_label']!r} for 'alt_label' in {pair_id!r}")
                    if d.get("alt_label") is True and d.get("hidden_label") is True:
                        issues.append(f"Conflicting alt_label=True and hidden_label=True in {pair_id!r}")
                else:
                    if "alt_label" in d and not isinstance(d["alt_label"], bool):
                        issues.append(f"Non-boolean value {d['alt_label']!r} for 'alt_label' in {pair_id!r}")
                    elif "hidden_label" in d and not isinstance(d["hidden_label"], bool):
                        issues.append(f"Non-boolean value {d['hidden_label']!r} for 'hidden_label' in {pair_id!r}")
                    elif d.get("alt_label") is True or d.get("hidden_label") is True:
                        issues.append(f"Conflicting alt_label/hidden_label with equivalent=False in {pair_id!r}")
            elif d.get("relation") not in {"skos:broader", "skos:narrower", "skos:related", "owl:differentFrom"}:
                issues.append(f"Invalid relation for {pair_id!r}")
            elif (
                d["relation"] == "owl:differentFrom"
                and isinstance(packet, dict)
                and any(
                    pair.get("pair_id") == pair_id and (pair.get("direct_facts") or pair.get("shared_role_facts"))
                    for pair in packet.get("pairs", [])
                )
            ):
                issues.append(f"Fact-associated pair cannot be differentFrom: {pair_id!r}")
            if pair_id in seen_pairs:
                issues.append(f"Duplicate pair_id={pair_id!r}")
            seen_pairs.add(pair_id)

        if expected is not None and seen_pairs != expected:
            issues.append(f"Missing pairs: {sorted(expected - seen_pairs)}")
        if not issues:
            return 1.0, f"PASS [{stage}]"
        return max(0.0, round(1.0 - 0.35 * len(issues), 4)), f"[{stage}] {'; '.join(issues)}"

    if stage in ("shacl_repair", "shacl_healer_rlm"):
        raw_repairs = getattr(prediction, "repairs", None)
        repairs = list(raw_repairs) if isinstance(raw_repairs, (list, tuple)) else parse_shacl_repairs(raw_repairs)
        allowed_nodes: set[str] = set()
        raw_viol = inputs.get("violations", "")
        try:
            viol_list = json.loads(raw_viol) if isinstance(raw_viol, str) else list(raw_viol or ())
            for v in viol_list:
                if isinstance(v, Mapping) and v.get("focus_node"):
                    allowed_nodes.add(str(v["focus_node"]).strip())
        except json.JSONDecodeError, TypeError, ValueError:
            viol_list = []

        issues = []
        if not repairs and allowed_nodes:
            issues.append("Empty repairs for non-empty SHACL violations")
        for r in repairs:
            if not isinstance(r, Mapping):
                issues.append("Malformed repair item")
                continue
            fnode = str(r.get("focus_node", "") or "").strip()
            act = str(r.get("action", "") or "").strip()
            val = str(r.get("value", r.get("target_node", r.get("target", ""))) or "").strip()
            if allowed_nodes and fnode not in allowed_nodes:
                issues.append(f"StatementShape/closedWorld out-of-scope focus_node={fnode!r}")
            if act not in VALID_SHACL_ACTIONS:
                issues.append(f"EdgeShape/closedWorld invalid action={act!r}")
            elif act in VALUE_BEARING_SHACL_ACTIONS and not val:
                issues.append(f"ConceptShape/definition empty value for action={act!r}")

        if shacl_linter is not None:
            remaining = list(shacl_linter() or ())
            if remaining:
                rem_shapes = sorted(
                    {
                        str(v.get("source_shape", v.get("shape", "SHACLShape")))
                        for v in remaining
                        if isinstance(v, Mapping)
                    }
                )
                issues.append(f"{len(remaining)} violation(s) remaining across shapes {rem_shapes}")
        if not issues:
            return 1.0, f"PASS [{stage}]"
        return max(0.0, round(1.0 - 0.35 * len(issues), 4)), f"[{stage}] {'; '.join(issues)}"

    if stage == "query_intent":
        qi = str(getattr(prediction, "query_intent", "") or "").strip()
        if len(qi) < 3:
            return 0.0, "[query_intent] Empty or trivial query_intent."
        return 1.0, "PASS [query_intent]"

    if stage == "triple_filter":
        raw_fact = getattr(prediction, "fact", getattr(prediction, "fact_after_filter", None))
        generated = coerce_generated_facts(raw_fact)
        max_facts = int(inputs.get("max_facts", 4) or 4)
        cand_norms: set[tuple[str, str, str]] = set()
        raw_before = inputs.get("fact_before_filter", "")
        try:
            parsed_before = json.loads(raw_before) if isinstance(raw_before, str) else raw_before
            fact_list = parsed_before.get("fact", []) if isinstance(parsed_before, Mapping) else parsed_before
            for item in fact_list or ():
                if isinstance(item, (list, tuple)) and len(item) >= 3:
                    cand_norms.add(
                        (
                            str(text_processing(str(item[0]))),
                            str(text_processing(str(item[1]))),
                            str(text_processing(str(item[2]))),
                        )
                    )
        except json.JSONDecodeError, TypeError, ValueError:
            cand_norms = set()
        issues = []
        if len(generated) > max_facts:
            issues.append(f"Exceeded max_facts ({len(generated)} > {max_facts})")
        if cand_norms:
            hallucinated = [
                f
                for f in generated
                if (
                    str(text_processing(f[0])),
                    str(text_processing(f[1])),
                    str(text_processing(f[2])),
                )
                not in cand_norms
            ]
            if hallucinated:
                issues.append(f"Hallucinated facts outside closed-world candidate subset: {hallucinated}")
        if not issues:
            return 1.0, "PASS [triple_filter]"
        return max(0.0, round(1.0 - 0.5 * len(issues), 4)), f"[triple_filter] {'; '.join(issues)}"

    if stage == "qa":
        ans = str(getattr(prediction, "answer", "") or "").strip()
        reas = str(getattr(prediction, "reasoning", "") or "").strip()
        ctx = str(inputs.get("context", "") or "").strip()
        if not ans:
            return 0.0, "[qa] Empty answer."
        issues = []
        if not reas:
            issues.append("Empty reasoning")
        if ctx and is_abstaining_answer(ans):
            issues.append(f"Abstaining answer {ans!r} on non-empty context")
        if not issues:
            return 1.0, "PASS [qa]"
        return max(0.2, round(1.0 - 0.45 * len(issues), 4)), f"[qa] {'; '.join(issues)}"

    if stage in ("llm_proposer", "llm_compressor"):
        inst_field = "shortened_instruction" if stage == "llm_compressor" else "new_instruction"
        new_inst = str(getattr(prediction, inst_field, "") or "").strip()
        max_chars = int(inputs.get("max_instruction_chars", 1500) or 1500)
        comp_name = str(inputs.get("component_name", "") or "").strip()
        comp_stage = comp_name.removesuffix(".predict")
        cur_inst = str(inputs.get("current_instruction", "") or "").strip()
        if not new_inst:
            return 0.0, f"[{stage}] Empty {inst_field}."
        issues = []
        if len(new_inst) > max_chars:
            issues.append(f"Instruction length {len(new_inst)} > {max_chars}")
        for marker in ("<TargetAnswer>", "<TargetPassage>", "<TargetEntity>"):
            if marker in new_inst:
                issues.append(f"Leaked placeholder marker {marker}")
        if comp_stage in STAGE_SCHEMA_CONTRACTS:
            strict_key, alt_markers, _ = STAGE_SCHEMA_CONTRACTS[comp_stage]
            had_strict = strict_key in cur_inst if cur_inst else True
            has_strict = strict_key in new_inst
            has_any = any(m.lower() in new_inst.lower() for m in alt_markers)
            if not (has_strict or (not had_strict and has_any)):
                issues.append(f"Missing required stage schema key {strict_key!r}")
        if not issues:
            return 1.0, f"PASS [{stage}]"
        return max(0.0, round(1.0 - 0.4 * len(issues), 4)), f"[{stage}] {'; '.join(issues)}"

    raise ValueError(f"Unknown stage contract for invariant validation: {stage!r}")


__all__ = [
    "STAGE_SCHEMA_CONTRACTS",
    "is_abstaining_answer",
    "validate_stage_invariants",
]
