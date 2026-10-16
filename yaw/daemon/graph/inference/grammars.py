"""Pure-JSON Context-Free Grammars (GBNF) and Sizecoding Pareto Metric for DSPy."""

import json
from collections.abc import Sequence
from typing import Any

import dspy

# ---------------------------------------------------------------------------
# Pure-JSON Context-Free Grammars (GBNF) for dspy.JSONAdapter on llama.cpp
# ---------------------------------------------------------------------------

DSPY_NER_GBNF_GRAMMAR = r"""root ::= "{\"named_entities\": " entity-list "}"
entity-list ::= "[]" | "[" string (", " string)* "]"
string ::= "\"" char+ "\""
char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
"""

DSPY_TRIPLES_GBNF_GRAMMAR = r"""root ::= "{\"triples\": " triple-list "}"
triple-list ::= "[]" | "[" triple (", " triple)* "]"
triple ::= "[" string ", " string ", " string "]"
string ::= "\"" char+ "\""
char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
"""

DSPY_RERANK_GBNF_GRAMMAR = r"""root ::= "{\"fact\": [" fact-list "]}"
fact-list ::= (triple (", " triple)*)?
triple ::= "[" string ", " string ", " string "]"
string ::= "\"" char+ "\""
char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
"""

DSPY_QA_COT_GBNF_GRAMMAR = r"""root ::= "{\"reasoning\": " string ", \"answer\": " string "}"
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
    seq = "string"
    for _ in range(max_entities - 1):
        seq = f'string (", " {seq})?'
    return (
        'root ::= "{\\"named_entities\\": [" entity-list "]}"\n'
        f"entity-list ::= ({seq})?\n"
        'string ::= "\\"" char+ "\\""\n'
        'char ::= [^"\\\\\\x7F\\x00-\\x1F] | "\\\\" (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})\n'
    )


def build_dspy_triples_grammar(max_triples: int | None = None) -> str:
    """Build pure-JSON GBNF grammar for ExtractTriples, optionally capping triple count for sizecoding."""
    if max_triples is None or max_triples <= 0:
        return DSPY_TRIPLES_GBNF_GRAMMAR
    seq = "triple"
    for _ in range(max_triples - 1):
        seq = f'triple (", " {seq})?'
    return (
        'root ::= "{\\"triples\\": [" triple-list "]}"\n'
        f"triple-list ::= ({seq})?\n"
        'triple ::= "[" string ", " string ", " string "]"\n'
        'string ::= "\\"" char+ "\\""\n'
        'char ::= [^"\\\\\\x7F\\x00-\\x1F] | "\\\\" (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})\n'
    )


def build_dspy_qa_cot_grammar(max_reasoning_chars: int | None = None) -> str:
    """Build pure-JSON GBNF grammar for AnswerQuestion CoT, optionally capping reasoning length."""
    if max_reasoning_chars is None or max_reasoning_chars <= 0:
        return DSPY_QA_COT_GBNF_GRAMMAR
    return (
        'root ::= "{\\"reasoning\\": " reasoning-str ", \\"answer\\": " string "}"\n'
        f'reasoning-str ::= "\\"" char{{1,{int(max_reasoning_chars)}}} "\\""\n'
        'string ::= "\\"" char+ "\\""\n'
        'char ::= [^"\\\\\\x7F\\x00-\\x1F] | "\\\\" (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})\n'
    )


def build_dspy_rerank_grammar(
    candidate_facts: Sequence[tuple[str, str, str] | list[str]],
    max_facts: int = 4,
) -> str:
    """Build dynamic candidate-literal pure-JSON GBNF grammar for DSPy FilterFacts (JSONAdapter)."""
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


def _clamp_unit(val: float) -> float:
    return max(0.0, min(1.0, float(val)))


def compute_sizecoding_pareto_score(
    recall_at_5: float,
    *,
    tokens: int,
    elapsed_seconds: float,
    recall_at_1: float | None = None,
    recall_at_2: float | None = None,
    recall_at_10: float | None = None,
    recall_at_20: float | None = None,
    em: float | None = None,
    f1: float | None = None,
    min_baseline_recall: float = 0.0,
    target_token_seconds: float = 100.0,
    pred_name: str | None = None,
    missed_items: Sequence[str] = (),
) -> dspy.Prediction:
    """Compute a sizecoding Pareto metric for `dspy.GEPA`.

    - Supports both single-rank (`recall_at_5 / (tokens * time)`) and full composite fitness:
      - Retrieval/indexing stages: `(Recall@1 * Recall@2 * Recall@5 * Recall@10 * Recall@20) / (tokens * time)`
      - End-to-End QA stages: `(Recall@1 * Recall@2 * Recall@5 * Recall@10 * Recall@20 * EM * F1) / (tokens * time)`
    - Hard Tier 1 baseline gate: if `recall_at_5 < min_baseline_recall`, the candidate is
      clamped below `0.05` with zero `pareto_raw`/`efficiency` objective scores so no
      token-saving mutation can ever drop below the locked Tier 1 baseline.
    """
    clamped_recall = _clamp_unit(recall_at_5)
    safe_tokens = max(int(tokens), 1)
    safe_seconds = max(float(elapsed_seconds), 1e-3)
    token_seconds = safe_tokens * safe_seconds
    target_budget = max(float(target_token_seconds), 1e-3)
    efficiency = target_budget / (target_budget + token_seconds)
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
        quality_numerator = r1 * r2 * r5 * r10 * r20
        if em is not None or f1 is not None:
            c_em = _clamp_unit(em if em is not None else 1.0)
            c_f1 = _clamp_unit(f1 if f1 is not None else 1.0)
            quality_numerator = quality_numerator * c_em * c_f1
        else:
            c_em = None
            c_f1 = None
    else:
        r1 = r2 = r5 = r10 = r20 = clamped_recall
        c_em = None
        c_f1 = None
        quality_numerator = clamped_recall

    pareto_raw = quality_numerator / token_seconds

    if clamped_recall < min_baseline_recall:
        floor_ratio = clamped_recall / max(float(min_baseline_recall), 1e-6)
        gated_score = 0.05 * max(0.0, min(0.99, floor_ratio))
        missed_str = f" Missed gold items: {list(missed_items)}." if missed_items else ""
        feedback = (
            f"{stage_prefix}BASELINE GATE VIOLATION: Recall@5={clamped_recall:.4f} dropped below "
            f"locked Tier 1 baseline ({min_baseline_recall:.4f}). Cost={safe_tokens} tokens * "
            f"{safe_seconds:.3f}s ({token_seconds:.2f} tok*s).{missed_str} "
            "Restore missing entity/triple/fact coverage first before compressing output tokens."
        )
        obj_violated: dict[str, float] = {
            "recall_at_5": clamped_recall,
            "pareto_raw": 0.0,
            "efficiency": 0.0,
        }
        if has_multi_rank:
            obj_violated.update(
                {
                    "recall_at_1": r1,
                    "recall_at_2": r2,
                    "recall_at_10": r10,
                    "recall_at_20": r20,
                    "quality_numerator": 0.0,
                }
            )
            if c_em is not None and c_f1 is not None:
                obj_violated["em"] = c_em
                obj_violated["f1"] = c_f1
        return dspy.Prediction(
            score=gated_score,
            feedback=feedback,
            pareto_raw=0.0,
            quality_numerator=0.0,
            recall_at_1=r1,
            recall_at_2=r2,
            recall_at_5=clamped_recall,
            recall_at_10=r10,
            recall_at_20=r20,
            em=c_em,
            f1=c_f1,
            tokens=safe_tokens,
            elapsed_seconds=safe_seconds,
            objective_scores=obj_violated,
        )

    score = 0.10 + 0.90 * quality_numerator * efficiency
    missed_str = f" Missed gold items: {list(missed_items)}." if missed_items else ""
    qa_str = f", EM={c_em:.4f}, F1={c_f1:.4f}" if (c_em is not None and c_f1 is not None) else ""
    feedback = (
        f"{stage_prefix}PASS Tier 1 floor ({clamped_recall:.4f} >= {min_baseline_recall:.4f}). "
        f"Sizecoding Pareto={pareto_raw:.6f} (Quality={quality_numerator:.4f}, "
        f"Recall@1/2/5/10/20={r1:.2f}/{r2:.2f}/{r5:.2f}/{r10:.2f}/{r20:.2f}{qa_str} / "
        f"[{safe_tokens} tokens * {safe_seconds:.3f}s = {token_seconds:.2f} tok*s], "
        f"efficiency={efficiency:.4f}).{missed_str} "
        "To increase Pareto score, preserve 100% of gold retrieval bridges while pruning "
        "redundant triples, verbose entity aliases, and filler reasoning tokens."
    )
    obj_scores: dict[str, float] = {
        "recall_at_5": clamped_recall,
        "pareto_raw": pareto_raw,
        "efficiency": efficiency,
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
        pareto_raw=pareto_raw,
        quality_numerator=quality_numerator,
        recall_at_1=r1,
        recall_at_2=r2,
        recall_at_5=clamped_recall,
        recall_at_10=r10,
        recall_at_20=r20,
        em=c_em,
        f1=c_f1,
        tokens=safe_tokens,
        elapsed_seconds=safe_seconds,
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
    "DSPY_RERANK_GBNF_GRAMMAR",
    "DSPY_QA_COT_GBNF_GRAMMAR",
    "gbnf_escape_literal",
    "build_dspy_ner_grammar",
    "build_dspy_triples_grammar",
    "build_dspy_qa_cot_grammar",
    "build_dspy_rerank_grammar",
    "compute_sizecoding_pareto_score",
    "lm_supports_gbnf",
]
