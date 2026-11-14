"""Per-signature GBNF grammar adapter and usage token summation for DSPy."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import dspy

from . import grammars
from .grammars import lm_supports_gbnf

_Base = dspy.JSONAdapter
_MAP = (dict, Mapping)
_NUM = (int, float)
_ROUTES: dict[str | tuple[str, ...], tuple[str, str | int | None, int | None] | int] = dict(
    (
        ("HealSHACLViolations", 0),
        ("ReflectiveInnerDistillerSignature", 0),
        ("FilterTriples", ("DSPY_RERANK_GBNF_GRAMMAR", None, 0)),
        (("fact",), ("DSPY_RERANK_GBNF_GRAMMAR", None, 0)),
        ("DefineMissingEntities", ("build_dspy_missing_entity_defs_grammar", 0, 0)),
        (("entity_definitions",), ("build_dspy_missing_entity_defs_grammar", 0, 0)),
        ("ExtractQueryIntent", ("build_dspy_query_intent_grammar", 0, 0)),
        (("query_intent",), ("build_dspy_query_intent_grammar", 0, 0)),
        ("JudgeLabels", ("build_dspy_label_judge_grammar", 0, None)),
        ("JudgeRelations", ("build_dspy_relation_judge_grammar", 0, None)),
        ("CompressInstructionSignature", ("build_dspy_instruction_proposal_grammar", "max_instruction_chars", 1)),
        ("CompressInstruction", ("build_dspy_instruction_proposal_grammar", "max_instruction_chars", 1)),
        (("shortened_instruction",), ("build_dspy_instruction_proposal_grammar", "max_instruction_chars", 1)),
        ("ExtractEntities", ("build_dspy_ner_grammar", "max_entities", 0)),
        (("named_entities",), ("build_dspy_ner_grammar", "max_entities", 0)),
        ("ProposeSHACLRepairs", ("build_dspy_shacl_repair_grammar", "max_entities", 0)),
        ("ExtractTriples", ("build_dspy_triples_grammar", "max_triples", 0)),
        (("triples",), ("build_dspy_triples_grammar", "max_triples", 0)),
        ("AnswerQuestion", ("build_dspy_qa_grammar", "max_reasoning_chars", 0)),
        (("answer",), ("build_dspy_qa_grammar", "max_reasoning_chars", 0)),
        (("reasoning", "answer"), ("build_dspy_qa_grammar", "max_reasoning_chars", 0)),
        (("plan", "answer"), ("build_dspy_qa_grammar", "max_reasoning_chars", 0)),
        ("ProposeInstructionSignature", ("build_dspy_instruction_proposal_grammar", "max_instruction_chars", 0)),
        ("ProposeInstruction", ("build_dspy_instruction_proposal_grammar", "max_instruction_chars", 0)),
        (("new_instruction",), ("build_dspy_instruction_proposal_grammar", "max_instruction_chars", 0)),
    )
)


class YawGBNFAdapter(_Base):
    """Supply per-signature GBNF grammars; DSPy owns all formatting and parsing."""

    def __init__(
        self,
        *,
        enable_gbnf: bool = False,
        max_entities: int | None = None,
        max_triples: int | None = None,
        max_facts: int = 4,
        max_reasoning_chars: int | None = None,
        max_instruction_chars: int | None = 1500,
        callbacks: list[Any] | None = None,
        use_native_function_calling: bool = True,
    ) -> None:
        _Base.__init__(self, callbacks=callbacks, use_native_function_calling=use_native_function_calling)
        self.enable_gbnf = bool(enable_gbnf)
        self.max_entities = max_entities
        self.max_triples = max_triples
        self.max_facts = max_facts
        self.max_reasoning_chars = max_reasoning_chars
        self.max_instruction_chars = max_instruction_chars

    def resolve_grammar(self, signature: Any) -> str | None:
        """Resolve the pure-JSON GBNF grammar for any of the 10 direct LM signatures (or None for RLM signatures)."""
        spec = _ROUTES.get(f"{getattr(signature, '__name__', signature)!s}")
        if (
            spec is None
            and isinstance(out_fields := getattr(signature, "output_fields", None), _MAP)
            and (spec := _ROUTES.get(out_keys := tuple(out_fields))) is None
        ):
            if (
                out_keys == ("repairs",)
                and isinstance(in_fields := getattr(signature, "input_fields", None), _MAP)
                and "focus_context" in in_fields
            ):
                spec = ("build_dspy_shacl_repair_grammar", "max_entities", 0)
            elif out_keys == ("decisions",):
                dec = out_fields["decisions"]
                text = f"{getattr(dec, 'annotation', None)!s}"
                for needle, s in (
                    ("LabelDecision", ("build_dspy_label_judge_grammar", 0, None)),
                    ("RelationDecision", ("build_dspy_relation_judge_grammar", 0, None)),
                    ("", 0),
                    ("alt_label", ("build_dspy_label_judge_grammar", 0, None)),
                    ("hidden_label", ("build_dspy_label_judge_grammar", 0, None)),
                    ("relation", ("build_dspy_relation_judge_grammar", 0, None)),
                ):
                    if not s:
                        extra = getattr(dec, "json_schema_extra", None)
                        desc = getattr(dec, "description", None) or (
                            extra.get("desc") if isinstance(extra, _MAP) else None
                        )
                        text = f"{desc!s} {signature.__doc__ or getattr(signature, 'instructions', None)!s}".lower()
                    elif needle in text:
                        spec = s
                        break
        if not isinstance(spec, tuple):
            return None
        fn_name, pos, flag = spec
        fn = getattr(grammars, fn_name)
        if pos is None:
            return fn
        if flag is None:
            return fn(self.max_entities, max_reasoning_chars=self.max_reasoning_chars)
        res = fn(getattr(self, pos)) if isinstance(pos, str) else fn()
        return res.replace("new_instruction", "shortened_instruction") if flag else res

    def _prepare_gbnf_kwargs(self, lm: Any, lm_kwargs: dict[str, Any], signature: Any) -> None:
        if self.enable_gbnf and lm_supports_gbnf(lm) and (g := self.resolve_grammar(signature)):
            lm_kwargs.setdefault("grammar", g)
        return None

    def __call__(
        self,
        lm: Any,
        lm_kwargs: dict[str, Any],
        signature: type[dspy.Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        self._prepare_gbnf_kwargs(lm, lm_kwargs, signature)
        return _Base.__call__(self, lm, lm_kwargs, signature, demos, inputs)

    async def acall(
        self,
        lm: Any,
        lm_kwargs: dict[str, Any],
        signature: type[dspy.Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        self._prepare_gbnf_kwargs(lm, lm_kwargs, signature)
        return await _Base.acall(self, lm, lm_kwargs, signature, demos, inputs)


def sum_usage_tokens(usage: Mapping[str, Any] | None) -> int:
    """Sum total tokens across all model entries in a DSPy usage dictionary."""
    total = 0
    if usage:
        for val in usage.values():
            t = val.get("total_tokens") if (m := isinstance(val, _MAP)) else val
            if isinstance(t, _NUM) and (t > 0 or not m):
                total += int(t)
            elif m:
                for k in ("prompt_tokens", "completion_tokens"):
                    total += int(val.get(k) or 0)
    return total


__all__ = [
    "YawGBNFAdapter",
    "sum_usage_tokens",
]
