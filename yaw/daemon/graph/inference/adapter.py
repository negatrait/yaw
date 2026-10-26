"""Per-signature GBNF grammar adapter and usage token summation for DSPy."""

from collections.abc import Mapping
from typing import Any

import dspy

from .grammars import (
    DSPY_RERANK_GBNF_GRAMMAR,
    build_dspy_instruction_proposal_grammar,
    build_dspy_missing_entity_defs_grammar,
    build_dspy_ner_grammar,
    build_dspy_qa_grammar,
    build_dspy_query_intent_grammar,
    build_dspy_shacl_repair_grammar,
    build_dspy_synonym_judge_grammar,
    build_dspy_triples_grammar,
    lm_supports_gbnf,
)


class YawGBNFAdapter(dspy.JSONAdapter):
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
        super().__init__(
            callbacks=callbacks,
            use_native_function_calling=use_native_function_calling,
        )
        self.enable_gbnf = bool(enable_gbnf)
        self.max_entities = max_entities
        self.max_triples = max_triples
        self.max_facts = max_facts
        self.max_reasoning_chars = max_reasoning_chars
        self.max_instruction_chars = max_instruction_chars

    def resolve_grammar(self, signature: Any) -> str | None:
        """Resolve the pure-JSON GBNF grammar for any of the 10 direct Qwen signatures (or None for RLM signatures)."""
        sig_name = getattr(signature, "__name__", str(signature))
        if sig_name in ("HealSHACLViolations", "ReflectiveInnerDistillerSignature"):
            return None
        if sig_name == "ExtractEntities":
            return build_dspy_ner_grammar(self.max_entities)
        if sig_name == "ExtractTriples":
            return build_dspy_triples_grammar(self.max_triples)
        if sig_name == "DefineMissingEntities":
            return build_dspy_missing_entity_defs_grammar()
        if sig_name == "JudgeSynonyms":
            return build_dspy_synonym_judge_grammar(self.max_entities)
        if sig_name == "ProposeSHACLRepairs":
            return build_dspy_shacl_repair_grammar(self.max_entities)
        if sig_name == "ExtractQueryIntent":
            return build_dspy_query_intent_grammar()
        if sig_name == "FilterTriples":
            return DSPY_RERANK_GBNF_GRAMMAR
        if sig_name == "AnswerQuestion":
            return build_dspy_qa_grammar(self.max_reasoning_chars)
        if sig_name in ("ProposeInstructionSignature", "CompressInstructionSignature"):
            return build_dspy_instruction_proposal_grammar(self.max_instruction_chars)

        out_fields = getattr(signature, "output_fields", None)
        in_fields = getattr(signature, "input_fields", None)
        if isinstance(out_fields, Mapping):
            out_keys = tuple(out_fields.keys())
            in_keys = tuple(in_fields.keys()) if isinstance(in_fields, Mapping) else ()
            if out_keys == ("named_entities",):
                return build_dspy_ner_grammar(self.max_entities)
            if out_keys == ("triples",):
                return build_dspy_triples_grammar(self.max_triples)
            if out_keys == ("entity_definitions",):
                return build_dspy_missing_entity_defs_grammar()
            if out_keys == ("decisions",):
                return build_dspy_synonym_judge_grammar(self.max_entities)
            if out_keys == ("repairs",) and "focus_context" in in_keys:
                return build_dspy_shacl_repair_grammar(self.max_entities)
            if out_keys == ("new_instruction",):
                return build_dspy_instruction_proposal_grammar(self.max_instruction_chars)
            if out_keys == ("shortened_instruction",):
                return build_dspy_instruction_proposal_grammar(self.max_instruction_chars).replace(
                    "new_instruction", "shortened_instruction"
                )
            if out_keys == ("query_intent",):
                return build_dspy_query_intent_grammar()
            if out_keys == ("fact",):
                return DSPY_RERANK_GBNF_GRAMMAR
            if out_keys in (("answer",), ("reasoning", "answer")):
                return build_dspy_qa_grammar(self.max_reasoning_chars)
        return None

    def _prepare_gbnf_kwargs(
        self,
        lm: Any,
        lm_kwargs: dict[str, Any],
        signature: Any,
    ) -> None:
        if not self.enable_gbnf or not lm_supports_gbnf(lm):
            return
        grammar = self.resolve_grammar(signature)
        if not grammar:
            return
        extra_body = lm_kwargs.get("extra_body")
        if isinstance(extra_body, Mapping):
            if "grammar" not in extra_body:
                lm_kwargs["extra_body"] = {**extra_body, "grammar": grammar}
        else:
            lm_kwargs["extra_body"] = {"grammar": grammar}

    def __call__(
        self,
        lm: Any,
        lm_kwargs: dict[str, Any],
        signature: type[dspy.Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        self._prepare_gbnf_kwargs(lm, lm_kwargs, signature)
        return super().__call__(lm, lm_kwargs, signature, demos, inputs)

    async def acall(
        self,
        lm: Any,
        lm_kwargs: dict[str, Any],
        signature: type[dspy.Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        self._prepare_gbnf_kwargs(lm, lm_kwargs, signature)
        return await super().acall(lm, lm_kwargs, signature, demos, inputs)


def sum_usage_tokens(usage: Mapping[str, Any] | None) -> int:
    """Sum total tokens across all model entries in a DSPy usage dictionary."""
    if not usage:
        return 0
    total = 0
    for val in usage.values():
        if isinstance(val, Mapping):
            t = val.get("total_tokens")
            if isinstance(t, (int, float)) and t > 0:
                total += int(t)
            else:
                p = int(val.get("prompt_tokens", 0) or 0)
                c = int(val.get("completion_tokens", 0) or 0)
                total += p + c
        elif isinstance(val, (int, float)):
            total += int(val)
    return total


__all__ = [
    "YawGBNFAdapter",
    "sum_usage_tokens",
]
