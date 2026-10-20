"""Pure-JSON Context-Free Grammars (GBNF) and Sizecoding Pareto Metric for DSPy."""

import json
from collections.abc import Sequence
from typing import Any

import dspy

# ---------------------------------------------------------------------------
# Pure-JSON Context-Free Grammars (GBNF) for dspy.JSONAdapter on llama.cpp
# ---------------------------------------------------------------------------

DSPY_NER_GBNF_GRAMMAR = r"""root ::= "{\"named_entities\": " entity-list "}"
entity-list ::= "[]" | "[" entity-pair (", " entity-pair)* "]"
entity-pair ::= "[" string ", " string "]"
string ::= "\"" char+ "\""
char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
"""

DSPY_TRIPLES_GBNF_GRAMMAR = r"""root ::= "{\"triples\": " triple-list "}"
triple-list ::= "[]" | "[" triple (", " triple)* "]"
triple ::= "{\"sub\": " entity-slot ", \"pred\": " string ", \"obj\": " entity-slot ", \"def\": " string "}"
entity-slot ::= new-entity-pair | string
new-entity-pair ::= "[" string ", " string "]"
string ::= "\"" char+ "\""
char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
"""

DSPY_MISSING_ENTITY_DEFS_GBNF_GRAMMAR = r"""root ::= "{\"entity_definitions\": " entity-list "}"
entity-list ::= "[]" | "[" entity-pair (", " entity-pair)* "]"
entity-pair ::= "[" string ", " string "]"
string ::= "\"" char+ "\""
char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
"""

DSPY_SYNONYM_JUDGE_GBNF_GRAMMAR = r"""root ::= "{\"decisions\": " decision-list "}"
decision-list ::= "[]" | "[" decision (", " decision)* "]"
decision ::= "{\"pair_id\": " string ", \"relation\": " relation-kind ", \"canonical\": " string "}"
relation-kind ::= "\"skos:altLabel\"" | "\"skos:hiddenLabel\"" | "\"skos:related\"" | "\"owl:differentFrom\""
string ::= "\"" char+ "\""
char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
"""
DSPY_SYNONYM_JUDGMENT_GBNF_GRAMMAR = DSPY_SYNONYM_JUDGE_GBNF_GRAMMAR

DSPY_SHACL_REPAIR_GBNF_GRAMMAR = r"""root ::= "{\"repairs\": " repair-list "}"
repair-list ::= "[]" | "[" repair (", " repair)* "]"
repair ::= "{\"focus_node\": " string ", \"action\": " action-kind ", \"value\": " string "}"
action-kind ::= "\"set_definition\"" | "\"rewrite_definition\"" | "\"set_pref_label\"" | "\"demote_to_alt_label\"" | "\"merge_altLabel\"" | "\"merge_hiddenLabel\"" | "\"add_different_from\"" | "\"assert_differentFrom\"" | "\"assert_related\"" | "\"delete_dangling\"" | "\"delete_invalid_fact\""
string ::= "\"" char+ "\""
char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
"""

DSPY_RERANK_GBNF_GRAMMAR = r"""root ::= "{\"fact\": [" fact-list "]}"
fact-list ::= (triple (", " triple)*)?
triple ::= "[" string ", " string ", " string "]"
string ::= "\"" char+ "\""
char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
"""

DSPY_QA_GBNF_GRAMMAR = r"""root ::= "{\"reasoning\": " string ", \"answer\": " string "}"
string ::= "\"" char+ "\""
char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
"""

DSPY_INSTRUCTION_PROPOSAL_GBNF_GRAMMAR = r"""root ::= "{\"new_instruction\": " string "}"
string ::= "\"" char+ "\""
char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
"""

DSPY_QUERY_INTENT_GBNF_GRAMMAR = r"""root ::= "{\"query_intent\": " string "}"
string ::= "\"" char+ "\""
char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
"""


def gbnf_escape_literal(text: str) -> str:
    escaped = (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def build_dspy_ner_grammar(max_entities: int | None = None) -> str:
    """Build pure-JSON GBNF grammar for ExtractEntities, optionally capping entity count for sizecoding."""
    if max_entities is None or max_entities <= 0:
        return DSPY_NER_GBNF_GRAMMAR
    seq = "entity-pair"
    for _ in range(max_entities - 1):
        seq = f'entity-pair (", " {seq})?'
    return (
        'root ::= "{\\"named_entities\\": [" entity-list "]}"\n'
        f"entity-list ::= ({seq})?\n"
        'entity-pair ::= "[" string ", " string "]"\n'
        'string ::= "\\"" char+ "\\""\n'
        'char ::= [^"\\\\\\x7F\\x00-\\x1F] | "\\\\" (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})\n'
    )


def build_dspy_triples_grammar(
    max_triples: int | None = None,
    known_entities: Sequence[Any] | None = None,
) -> str:
    """Build pure-JSON GBNF grammar for ExtractTriples, constraining known Stage 1 entities as bare strings and novel entities as [name, def] pairs."""
    deduped_known: list[str] = []
    if known_entities:
        for item in known_entities:
            if isinstance(item, str):
                name = item.strip()
            elif isinstance(item, (list, tuple)) and item and isinstance(item[0], str):
                name = item[0].strip()
            else:
                continue
            if name and name not in deduped_known:
                deduped_known.append(name)

    if (max_triples is None or max_triples <= 0) and not deduped_known:
        return DSPY_TRIPLES_GBNF_GRAMMAR

    if max_triples is None or max_triples <= 0:
        root_line = 'root ::= "{\\"triples\\": " triple-list "}"\n'
        list_line = 'triple-list ::= "[]" | "[" triple (", " triple)* "]"\n'
    else:
        seq = "triple"
        for _ in range(max_triples - 1):
            seq = f'triple (", " {seq})?'
        root_line = 'root ::= "{\\"triples\\": [" triple-list "]}"\n'
        list_line = f"triple-list ::= ({seq})?\n"

    triple_line = (
        'triple ::= "{\\"sub\\": " entity-slot ", \\"pred\\": " string '
        '", \\"obj\\": " entity-slot ", \\"def\\": " string "}"\n'
    )
    if deduped_known:
        known_rule = " | ".join(
            gbnf_escape_literal(json.dumps(e, ensure_ascii=False)) for e in deduped_known
        )
        slot_lines = (
            "entity-slot ::= known-entity | new-entity-pair\n"
            f"known-entity ::= {known_rule}\n"
            'new-entity-pair ::= "[" string ", " string "]"\n'
        )
    else:
        slot_lines = (
            "entity-slot ::= new-entity-pair | string\n"
            'new-entity-pair ::= "[" string ", " string "]"\n'
        )

    return (
        root_line
        + list_line
        + triple_line
        + slot_lines
        + 'string ::= "\\"" char+ "\\""\n'
        + 'char ::= [^"\\\\\\x7F\\x00-\\x1F] | "\\\\" (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})\n'
    )


def build_dspy_missing_entity_defs_grammar(
    missing_names: Sequence[str] | None = None,
) -> str:
    """Build pure-JSON GBNF grammar for DefineMissingEntities (`{"entity_definitions": [[name, def], ...]}`)."""
    deduped: list[str] = []
    if missing_names:
        for name in missing_names:
            if isinstance(name, str) and name.strip() and name.strip() not in deduped:
                deduped.append(name.strip())
    if not deduped:
        return DSPY_MISSING_ENTITY_DEFS_GBNF_GRAMMAR
    seq = "entity-pair"
    for _ in range(len(deduped) - 1):
        seq = f'entity-pair (", " {seq})?'
    name_rule = " | ".join(
        gbnf_escape_literal(json.dumps(name, ensure_ascii=False)) for name in deduped
    )
    return (
        'root ::= "{\\"entity_definitions\\": [" entity-list "]}"\n'
        f"entity-list ::= ({seq})?\n"
        'entity-pair ::= "[" entity-name ", " string "]"\n'
        f"entity-name ::= {name_rule}\n"
        'string ::= "\\"" char+ "\\""\n'
        'char ::= [^"\\\\\\x7F\\x00-\\x1F] | "\\\\" (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})\n'
    )


def build_dspy_synonym_judge_grammar(max_decisions: int | None = None) -> str:
    """Build pure-JSON GBNF grammar for JudgeSynonyms (`{"decisions": [...]}`)."""
    if max_decisions is None or max_decisions <= 0:
        return DSPY_SYNONYM_JUDGE_GBNF_GRAMMAR
    seq = "decision"
    for _ in range(max_decisions - 1):
        seq = f'decision (", " {seq})?'
    return (
        'root ::= "{\\"decisions\\": [" decision-list "]}"\n'
        f"decision-list ::= ({seq})?\n"
        'decision ::= "{\\"pair_id\\": " string ", \\"relation\\": " relation-kind ", \\"canonical\\": " string "}"\n'
        'relation-kind ::= "\\"skos:altLabel\\"" | "\\"skos:hiddenLabel\\"" | "\\"skos:related\\"" | "\\"owl:differentFrom\\""\n'
        'string ::= "\\"" char+ "\\""\n'
        'char ::= [^"\\\\\\x7F\\x00-\\x1F] | "\\\\" (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})\n'
    )


build_dspy_synonym_judgment_grammar = build_dspy_synonym_judge_grammar


def build_dspy_shacl_repair_grammar(max_repairs: int | None = None) -> str:
    """Build pure-JSON GBNF grammar for ProposeSHACLRepairs (`{"repairs": [...]}`)."""
    if max_repairs is None or max_repairs <= 0:
        return DSPY_SHACL_REPAIR_GBNF_GRAMMAR
    seq = "repair"
    for _ in range(max_repairs - 1):
        seq = f'repair (", " {seq})?'
    return (
        'root ::= "{\\"repairs\\": [" repair-list "]}"\n'
        f"repair-list ::= ({seq})?\n"
        'repair ::= "{\\"focus_node\\": " string ", \\"action\\": " action-kind ", \\"value\\": " string "}"\n'
        'action-kind ::= "\\"set_definition\\"" | "\\"rewrite_definition\\"" | "\\"set_pref_label\\"" | "\\"demote_to_alt_label\\"" | "\\"merge_altLabel\\"" | "\\"merge_hiddenLabel\\"" | "\\"add_different_from\\"" | "\\"assert_differentFrom\\"" | "\\"assert_related\\"" | "\\"delete_dangling\\"" | "\\"delete_invalid_fact\\""\n'
        'string ::= "\\"" char+ "\\""\n'
        'char ::= [^"\\\\\\x7F\\x00-\\x1F] | "\\\\" (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})\n'
    )


def build_dspy_qa_grammar(max_reasoning_chars: int | None = None) -> str:
    """Build pure-JSON GBNF grammar for AnswerQuestion, optionally capping reasoning length."""
    if max_reasoning_chars is None or max_reasoning_chars <= 0:
        return DSPY_QA_GBNF_GRAMMAR
    return (
        'root ::= "{\\"reasoning\\": " reasoning-str ", \\"answer\\": " string "}"\n'
        f'reasoning-str ::= "\\"" char{{1,{int(max_reasoning_chars)}}} "\\""\n'
        'string ::= "\\"" char+ "\\""\n'
        'char ::= [^"\\\\\\x7F\\x00-\\x1F] | "\\\\" (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})\n'
    )


def build_dspy_instruction_proposal_grammar(max_chars: int | None = 1500) -> str:
    """Build pure-JSON GBNF grammar for GEPA/MIPROv2 instruction synthesis on Qwen (`llama.cpp` GPU)."""
    if max_chars is None or max_chars <= 0:
        return DSPY_INSTRUCTION_PROPOSAL_GBNF_GRAMMAR
    return (
        'root ::= "{\\"new_instruction\\": " instruction-str "}"\n'
        f'instruction-str ::= "\\"" char{{1,{int(max_chars)}}} "\\""\n'
        'char ::= [^"\\\\\\x7F\\x00-\\x1F] | "\\\\" (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})\n'
    )


def build_dspy_query_intent_grammar() -> str:
    """Build pure-JSON GBNF grammar for ExtractQueryIntent (`{"query_intent": "..."}`)."""
    return DSPY_QUERY_INTENT_GBNF_GRAMMAR


def build_dspy_rerank_grammar(
    candidate_facts: Sequence[tuple[str, str, str] | list[str]],
    max_facts: int = 4,
) -> str:
    """Build dynamic candidate-literal pure-JSON GBNF grammar for DSPy FilterTriples (JSONAdapter)."""
    candidates: list[str] = []
    for item in candidate_facts:
        if (
            not isinstance(item, (list, tuple))
            or len(item) != 3
            or not all(isinstance(x, str) for x in item)
        ):
            return DSPY_RERANK_GBNF_GRAMMAR
        serialized = json.dumps(list(item), ensure_ascii=False)
        if serialized not in candidates:
            candidates.append(serialized)

    if not candidates:
        return 'root ::= "{\\"fact\\": []}"\n'

    limit = min(max_facts, len(candidates))
    seq = "triple"
    for _ in range(limit - 1):
        seq = f'triple (", " {seq})?'
    triple_rule = " | ".join(gbnf_escape_literal(c) for c in candidates)
    return (
        'root ::= "{\\"fact\\": [" fact-list "]}"\n'
        f"fact-list ::= ({seq})?\n"
        f"triple ::= {triple_rule}\n"
    )


_METRIC_FLOOR = 0.05


def _clamp_unit(val: float) -> float:
    return max(0.0, min(1.0, float(val)))


def _smooth_floor(val: float) -> float:
    return max(_METRIC_FLOOR, _clamp_unit(val))


def _geometric_mean(values: Sequence[float]) -> float:
    if not values:
        return 1.0
    product = 1.0
    for v in values:
        product *= max(_METRIC_FLOOR, float(v))
    return float(product ** (1.0 / len(values)))


def compute_sizecoding_pareto_score(
    recall_at_5: float,
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
    repair_rate: float | None = None,
    latency_ms: float | None = None,
    target_latency_ms: float = 1000.0,
    min_baseline_recall: float = 0.0,
    target_tokens: float = 100.0,
    pred_name: str | None = None,
    missed_items: Sequence[str] = (),
) -> dspy.Prediction:
    """Compute a gate-free sizecoding Pareto metric (`GM_good / GM_bad`) for `dspy.GEPA`.

    - Computes the geometric mean of smoothed quality goods (`Recall@{1,2,5,10,20}`, `EM`, `F1`,
      `shacl_conformance`) divided by the geometric mean of cost/violation bads.
    - No hard step-function gate cliffs: every improvement in recall, QA accuracy, SHACL conformance,
      or token/latency compression provides a smooth, positive evolutionary gradient.
    """
    clamped_recall = _clamp_unit(recall_at_5)
    safe_tokens = max(int(tokens), 1)
    target_budget = max(float(target_tokens), 1.0)
    efficiency = target_budget / (target_budget + safe_tokens)
    stage_prefix = f"[{pred_name}] " if pred_name else ""

    has_multi_rank = any(
        x is not None for x in (recall_at_1, recall_at_2, recall_at_10, recall_at_20, em, f1)
    )
    if has_multi_rank:
        r1 = _clamp_unit(recall_at_1 if recall_at_1 is not None else clamped_recall)
        r2 = _clamp_unit(recall_at_2 if recall_at_2 is not None else clamped_recall)
        r5 = clamped_recall
        r10 = _clamp_unit(recall_at_10 if recall_at_10 is not None else clamped_recall)
        r20 = _clamp_unit(recall_at_20 if recall_at_20 is not None else clamped_recall)
        rank_qa_factors: list[float] = [
            _smooth_floor(r1),
            _smooth_floor(r2),
            _smooth_floor(r5),
            _smooth_floor(r10),
            _smooth_floor(r20),
        ]
        if em is not None or f1 is not None:
            c_em = _clamp_unit(em if em is not None else 1.0)
            c_f1 = _clamp_unit(f1 if f1 is not None else 1.0)
            rank_qa_factors.append(_smooth_floor(c_em))
            rank_qa_factors.append(_smooth_floor(c_f1))
        else:
            c_em = None
            c_f1 = None
        quality_numerator = 1.0
        for factor in rank_qa_factors:
            quality_numerator *= factor
        goods: list[float] = list(rank_qa_factors)
    else:
        r1 = r2 = r5 = r10 = r20 = clamped_recall
        c_em = None
        c_f1 = None
        quality_numerator = clamped_recall
        goods = [_smooth_floor(clamped_recall)]

    eff_v_rem = max(
        0,
        int(shacl_violations) if shacl_violations is not None else int(shacl_violations_remaining),
    )
    eff_v_init = max(0, int(shacl_violations_initial))
    eff_missing_defs = max(0, int(missing_entity_defs_count))
    eff_missing_def_tokens = max(0, int(missing_entity_defs_tokens))
    g_shacl = (
        _clamp_unit(shacl_conformance)
        if shacl_conformance is not None
        else 1.0 / (1.0 + float(eff_v_rem))
    )
    goods.append(_smooth_floor(g_shacl))

    gm_good = _geometric_mean(goods)

    b_tokens = 1.0 + (float(safe_tokens) / target_budget)
    b_shacl_initial = 1.0 + float(eff_v_init + eff_missing_defs)
    b_shacl_remaining = 1.0 + float(eff_v_rem)
    aux_rlm_tokens = max(
        0, int(synonym_judge_tokens) + int(shacl_healer_tokens) + eff_missing_def_tokens
    )
    b_repair_tokens = 1.0 + (float(aux_rlm_tokens) / target_budget)
    bads: list[float] = [
        b_tokens,
        b_shacl_initial,
        b_shacl_remaining,
        b_repair_tokens,
    ]
    if latency_ms is not None and target_latency_ms > 0:
        bads.append(1.0 + (max(0.0, float(latency_ms)) / float(target_latency_ms)))
    if repair_rate is not None and repair_rate > 0.0:
        bads.append(1.0 + float(repair_rate))

    gm_bad = _geometric_mean(bads)
    score = gm_good / gm_bad
    pareto_raw = quality_numerator / safe_tokens

    missed_str = f" Missed gold items: {list(missed_items)}." if missed_items else ""
    missing_defs_str = (
        f" [entity_definer] repaired {eff_missing_defs} missing Stage-2 entity definition(s) "
        f"({eff_missing_def_tokens} aux tokens); emit inline [name, def] pairs in Stage 2 or extract in Stage 1."
        if eff_missing_defs > 0
        else ""
    )
    qa_str = f", EM={c_em:.4f}, F1={c_f1:.4f}" if (c_em is not None and c_f1 is not None) else ""
    baseline_note = (
        f"Below reference floor ({clamped_recall:.4f} < {min_baseline_recall:.4f}); prioritize recall recovery. "
        if clamped_recall < min_baseline_recall
        else f"PASS reference floor ({clamped_recall:.4f} >= {min_baseline_recall:.4f}). "
    )
    feedback = (
        f"{stage_prefix}{baseline_note}"
        f"Gate-free GM_good/GM_bad={score:.6f} (GM_good={gm_good:.4f}, GM_bad={gm_bad:.4f}, "
        f"Pareto={pareto_raw:.6f}, Recall@1/2/5/10/20={r1:.2f}/{r2:.2f}/{r5:.2f}/{r10:.2f}/{r20:.2f}{qa_str}, "
        f"SHACL={g_shacl:.4f} / [{safe_tokens} tokens], efficiency={efficiency:.4f})."
        f"{missed_str}{missing_defs_str}"
    )
    obj_scores: dict[str, float] = {
        "recall_at_5": clamped_recall,
        "pareto_raw": pareto_raw,
        "efficiency": efficiency,
        "gm_good": gm_good,
        "gm_bad": gm_bad,
        "shacl_conforms": g_shacl,
    }
    if has_multi_rank:
        obj_scores.update(
            {
                "recall_at_1": r1,
                "recall_at_2": r2,
                "recall_at_10": r10,
                "recall_at_20": r20,
                "quality_numerator": quality_numerator,
            }
        )
        if c_em is not None and c_f1 is not None:
            obj_scores["em"] = c_em
            obj_scores["f1"] = c_f1

    return dspy.Prediction(
        score=score,
        feedback=feedback,
        gm_good=gm_good,
        gm_bad=gm_bad,
        pareto_raw=pareto_raw,
        quality_numerator=quality_numerator,
        shacl_conforms=g_shacl,
        recall_at_1=r1,
        recall_at_2=r2,
        recall_at_5=clamped_recall,
        recall_at_10=r10,
        recall_at_20=r20,
        em=c_em,
        f1=c_f1,
        tokens=safe_tokens,
        missing_entity_defs_count=eff_missing_defs,
        missing_entity_defs_tokens=eff_missing_def_tokens,
        objective_scores=obj_scores,
    )


def lm_supports_gbnf(lm: Any = None) -> bool:
    """Return True if `lm` runs on a `llama.cpp` backend that supports GBNF grammars.

    Ryzen AI NPU/OGA (`ryzenai-llm`, e.g. `Mistral-7B-Instruct-v0.3-Hybrid`) and FLM
    do NOT support GBNF (`extra_body={"grammar": ...}`). Only `llamacpp` GPU models
    (e.g. `Qwen3-4B-Instruct-2507-GGUF`) support GBNF constrained decoding.
    """
    target_lm = lm if lm is not None else getattr(dspy.settings, "lm", None)
    if target_lm is None:
        return True
    explicit = getattr(target_lm, "_yaw_supports_gbnf", None)
    if isinstance(explicit, bool):
        return explicit
    recipe = getattr(target_lm, "_yaw_recipe", None)
    if recipe in ("ryzenai-llm", "flm"):
        return False
    model_name = str(getattr(target_lm, "model", "")).lower()
    return not ("mistral" in model_name or "ryzenai" in model_name or "flm" in model_name)


__all__ = [
    "DSPY_NER_GBNF_GRAMMAR",
    "DSPY_TRIPLES_GBNF_GRAMMAR",
    "DSPY_MISSING_ENTITY_DEFS_GBNF_GRAMMAR",
    "DSPY_SYNONYM_JUDGE_GBNF_GRAMMAR",
    "DSPY_SYNONYM_JUDGMENT_GBNF_GRAMMAR",
    "DSPY_SHACL_REPAIR_GBNF_GRAMMAR",
    "DSPY_RERANK_GBNF_GRAMMAR",
    "DSPY_QA_GBNF_GRAMMAR",
    "DSPY_INSTRUCTION_PROPOSAL_GBNF_GRAMMAR",
    "DSPY_QUERY_INTENT_GBNF_GRAMMAR",
    "gbnf_escape_literal",
    "build_dspy_ner_grammar",
    "build_dspy_triples_grammar",
    "build_dspy_missing_entity_defs_grammar",
    "build_dspy_synonym_judge_grammar",
    "build_dspy_synonym_judgment_grammar",
    "build_dspy_shacl_repair_grammar",
    "build_dspy_qa_grammar",
    "build_dspy_instruction_proposal_grammar",
    "build_dspy_query_intent_grammar",
    "build_dspy_rerank_grammar",
    "compute_sizecoding_pareto_score",
    "lm_supports_gbnf",
]
