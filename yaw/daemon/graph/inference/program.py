"""Native DSPy HippoRAGProgram and _PositionalSafeRLM orchestrator."""

import functools
import inspect
import json
import logging
import math
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import dspy
from dspy.utils.exceptions import AdapterParseError, LMError

from yaw.core.config import YawConfig, context_char_limit
from yaw.core.types import OpenIERecord, StructuredTriple

from .chat import create_dspy_lm
from .diagnostics import DiagnosticHistory, diagnostic_boundary, trim_diagnostics
from .grammars import (
    DSPY_RERANK_GBNF_GRAMMAR,
    build_dspy_instruction_proposal_grammar,
    build_dspy_missing_entity_defs_grammar,
    build_dspy_ner_grammar,
    build_dspy_qa_grammar,
    build_dspy_query_intent_grammar,
    build_dspy_rerank_grammar,
    build_dspy_shacl_repair_grammar,
    build_dspy_synonym_judge_grammar,
    build_dspy_triples_grammar,
    lm_supports_gbnf,
)
from .parsers import (
    coerce_generated_facts,
    extract_entity_definitions,
    extract_named_entities,
    extract_structured_triples,
    extract_triples,
    match_facts_to_candidate_indices,
    parse_shacl_repairs,
    parse_synonym_decisions,
    strip_mistral_bracket_prefix,
    text_processing,
)
from .prompts import (
    NER_DEMOS,
    QA_DEMOS,
    TRIPLE_DEMOS,
    TRIPLE_FILTER_DEMOS,
    AnswerQuestion,
    CompressInstructionSignature,
    DefineMissingEntities,
    ExtractEntities,
    ExtractQueryIntent,
    ExtractTriples,
    FilterTriples,
    HealSHACLViolations,
    JudgeSynonyms,
    ProposeInstructionSignature,
    ProposeSHACLRepairs,
    ReflectiveInnerDistillerSignature,
)
from .reflection import (
    ReflectionBudgetError,
    ReflectionLM,
    checkpoint_object,
    deterministic_evidence,
    reflection_digest,
    reflection_examples,
    resolve_evidence,
    validate_request_checkpoint,
    validate_simba_state,
)

_STAGE_RLM_DELEGATION_RULES: dict[str, str] = {
    "shacl_repair": (
        "CRITICAL RLM SANDBOX RULE:\n"
        "Prioritize aggressive 2-step resolution: do NOT perform exploratory node inspections unless strictly necessary.\n"
        "Immediately delegate to `qwen_propose_shacl_repairs(violations=violations)` on Step 1, verify if needed, and call `SUBMIT(repairs=repairs)` by Step 2 without wandering:\n"
        "```python\n"
        "repairs = qwen_propose_shacl_repairs(violations=violations)\n"
        "SUBMIT(repairs=repairs)\n"
        "```"
    ),
    "instruction_proposer_inner": (
        "Inspect examples_with_feedback in Python. Select evidence by evidence_id and field path; "
        "use start/end character offsets for string excerpts. Include failures and successes. "
        "contrasts. Never write instructions or reconstruct facts. "
        "SUBMIT(selection={'diagnosis': 'brief generalizable diagnosis', "
        "'references': [{'example_id': examples_with_feedback[0]['evidence_id'], 'path': []}]})"
    ),
}


def _build_rlm_action_demos() -> dict[str, list[dspy.Example]]:
    from dspy.primitives.repl_types import REPLHistory

    empty_history = REPLHistory()
    return {
        "shacl_repair": [
            dspy.Example(
                variables_info="Variable: `violations` (type: str): JSON list of SHACLViolation report items",
                repl_history=empty_history,
                iteration="1/10",
                reasoning="Immediately delegate to Qwen worker and submit in step 1 without wandering.",
                code="```python\nrepairs = qwen_propose_shacl_repairs(violations=violations)\nSUBMIT(repairs=repairs)\n```",
            ).with_inputs("variables_info", "repl_history", "iteration")
        ],
        "instruction_proposer_inner": [],
    }


def _normalize_rlm_field(name: str, val: Any, annotation: Any) -> Any:
    if isinstance(val, dict):
        if name in val:
            val = val[name]
        elif len(val) == 1:
            val = next(iter(val.values()))
    if name == "named_entities" and isinstance(val, str):
        defs = extract_entity_definitions(val)
        if defs:
            return [[k, v] if v else k for k, v in defs]
        return list(extract_named_entities(val))
    if name == "entity_definitions":
        if isinstance(val, (str, dict, list, tuple)):
            defs = extract_entity_definitions(val)
            return [[k, v] for k, v in defs if k]
        return val
    if name == "triples":
        if isinstance(val, str):
            try:
                loaded = json.loads(val)
            except json.JSONDecodeError, TypeError, ValueError:
                loaded = None
            if isinstance(loaded, list):
                return loaded
            return [list(t) for t in extract_triples(val)]
        if isinstance(val, (list, tuple)):
            return list(val)
        return val
    if name == "decisions":
        return parse_synonym_decisions(val)
    if name == "repairs":
        return parse_shacl_repairs(val)
    if name == "fact":
        return coerce_generated_facts(val)
    if annotation is str and isinstance(val, str):
        return strip_mistral_bracket_prefix(val)
    return val


def _sanitize_repl_code(code: str) -> str:
    """Strip hallucinated imports of pre-bound sandbox tools (`qwen_*`, `inspect_focus_node`, `hydrate_chunk`, `apply_shacl_repairs`, `lint_shacl`, `synthesize_instruction`, `SUBMIT`, `llm_query`)."""
    cleaned_lines: list[str] = []
    for line in code.splitlines():
        stripped = line.strip()
        if (stripped.startswith("from ") or stripped.startswith("import ")) and (
            "qwen_" in stripped
            or "inspect_node" in stripped
            or "inspect_focus_node" in stripped
            or "hydrate_chunk" in stripped
            or "read_source_chunk" in stripped
            or "apply_shacl_repairs" in stripped
            or "lint_shacl" in stripped
            or "synthesize_instruction" in stripped
            or "SUBMIT" in stripped
            or "llm_query" in stripped
        ):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


_RLM_Base: Any = getattr(dspy, "RLM", dspy.Module)


class _PositionalSafeRLM(_RLM_Base):
    """RLM subclass that binds positional tool arguments, enforces 1-step Qwen delegation, and normalizes SUBMIT payloads."""

    def __init__(self, *args: Any, stage_rule: str = "", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.last_reasoning: str = ""
        self._action_signature = self._action_signature.with_updated_fields(
            "reasoning",
            desc="Concise 1-sentence diagnosis or action plan.",
        )
        if stage_rule:
            updated_instructions = self._action_signature.instructions + "\n\n" + stage_rule
            self._action_signature = self._action_signature.with_instructions(updated_instructions)
        self.generate_action.signature = self._action_signature

    def _execute_iteration(
        self,
        repl: Any,
        variables: list[Any],
        history: Any,
        iteration: int,
        input_args: dict[str, Any],
        output_field_names: list[str],
        interpreter_factory: Callable[[], Any],
        sub_agent_rules: str,
    ) -> Any:
        variables_info = [variable.format() for variable in variables]
        action = self.generate_action(
            signature=self._action_signature_for_current_factory(
                interpreter_factory, sub_agent_rules
            ),
            variables_info=variables_info,
            repl_history=history,
            iteration=f"{iteration + 1}/{self.max_iters}",
        )
        self.last_reasoning = strip_mistral_bracket_prefix(
            str(getattr(action, "reasoning", "") or "")
        ).strip()
        from dspy.predict.rlm import _strip_code_fences, format_error_for_lm

        try:
            code = _sanitize_repl_code(_strip_code_fences(action.code))
        except SyntaxError as e:
            code = action.code
            result = f"[Error] {format_error_for_lm(e)}"
            return self._process_execution_result(action, code, result, history, output_field_names)
        result = self._execute_code(repl, code, input_args)
        return self._process_execution_result(action, code, result, history, output_field_names)

    def _make_interpreter_tool(self, tool: dspy.Tool) -> Callable[..., Any]:
        sig = inspect.signature(tool.func)
        is_async = inspect.iscoroutinefunction(tool.func) or (
            callable(tool.func) and inspect.iscoroutinefunction(tool.func.__call__)
        )
        if is_async:

            async def invoke(*args: Any, **kwargs: Any) -> Any:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                return await tool.acall(**bound.arguments)
        else:

            def invoke(*args: Any, **kwargs: Any) -> Any:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                return tool(**bound.arguments)

        functools.update_wrapper(invoke, tool.func)
        # inspect.signature's documented callable protocol; retain coroutine
        # identity rather than replacing the function with a callable object.
        vars(invoke)["__signature__"] = sig
        return invoke

    def _process_final_output(
        self,
        result: Any,
        output_field_names: list[str],
    ) -> tuple[dict[str, Any] | None, str | None]:
        raw_output = getattr(result, "output", None)
        if isinstance(raw_output, dict):
            if (
                len(output_field_names) == 1
                and output_field_names[0] not in raw_output
                and len(raw_output) == 1
            ):
                raw_output[output_field_names[0]] = next(iter(raw_output.values()))
            for name in output_field_names:
                if name in raw_output:
                    field = self.signature.output_fields[name]
                    annotation = getattr(field, "annotation", str)
                    raw_output[name] = _normalize_rlm_field(name, raw_output[name], annotation)
        return super()._process_final_output(result, output_field_names)

    def _extract_fallback(
        self,
        variables: list[Any],
        history: Any,
        output_field_names: list[str],
    ) -> dspy.Prediction:
        try:
            pred = super()._extract_fallback(variables, history, output_field_names)
        except Exception:
            payload: dict[str, Any] = {}
            for name in output_field_names:
                for var in variables if isinstance(variables, list) else []:
                    if hasattr(var, "name") and var.name == name:
                        payload[name] = var.value
                        break
                    elif isinstance(var, dict) and name in var:
                        payload[name] = var[name]
                        break
                if name not in payload:
                    field = self.signature.output_fields.get(name)
                    annotation = getattr(field, "annotation", None)
                    if annotation is list or getattr(annotation, "__origin__", None) is list:
                        payload[name] = []
                    elif annotation is dict or getattr(annotation, "__origin__", None) is dict:
                        payload[name] = {}
                    elif annotation in (int, float):
                        payload[name] = 0
                    else:
                        payload[name] = ""
            traj: list[Any] = []
            for item in history or []:
                dump_fn = getattr(item, "model_dump", None)
                traj.append(dump_fn() if callable(dump_fn) else str(item))
            pred = dspy.Prediction(
                trajectory=traj,
                final_reasoning="Extract fallback recovered from REPL state",
                **payload,
            )
        for name in output_field_names:
            field = self.signature.output_fields[name]
            annotation = getattr(field, "annotation", str)
            val = getattr(pred, name, None)
            if annotation is str and isinstance(val, str):
                setattr(pred, name, strip_mistral_bracket_prefix(val))
        return pred


def _parse_react_v2_payload(fields: Mapping[str, Any]) -> dict[str, Any] | None:
    """Coerce ReActV2 tool-call payloads (`next_thought`/`tool_calls` or `thought`/`tool_name`/`tool_args`)."""
    from dspy.adapters.types.reasoning import Reasoning
    from dspy.adapters.types.tool import ToolCalls

    if "tool_calls" in fields:
        thought_val = fields.get("next_thought", fields.get("thought", ""))
        tc_raw = fields["tool_calls"]
        if isinstance(tc_raw, str):
            import json_repair

            tc_raw = json_repair.loads(tc_raw)
        return {
            "next_thought": (
                thought_val
                if isinstance(thought_val, Reasoning)
                else Reasoning(content=str(thought_val or ""))
            ),
            "tool_calls": (
                tc_raw if isinstance(tc_raw, ToolCalls) else ToolCalls.model_validate(tc_raw)
            ),
        }
    if "tool_name" in fields:
        raw_name = str(fields.get("tool_name", "") or "").strip()
        tool_name = "submit" if raw_name in ("finish", "submit") else raw_name
        raw_args = fields.get("tool_args", {})
        if isinstance(raw_args, str):
            import json_repair

            loaded_args = json_repair.loads(raw_args)
            raw_args = loaded_args if isinstance(loaded_args, dict) else {}
        elif not isinstance(raw_args, Mapping):
            raw_args = {}
        thought_str = str(fields.get("next_thought") or fields.get("thought") or "")
        return {
            "next_thought": Reasoning(content=thought_str),
            "tool_calls": ToolCalls.model_validate(
                {"tool_calls": [{"name": tool_name, "args": dict(raw_args)}]}
            ),
        }
    return None


class YawJSONAdapter(dspy.JSONAdapter):
    """Robust JSON adapter that recovers from top-level arrays or mismatched single fields."""

    def parse(self, signature: type[dspy.Signature], completion: str) -> dict[str, Any]:
        out_fields = list(signature.output_fields.keys())
        try:
            return super().parse(signature, completion)
        except Exception:
            if "[[ ##" in completion:
                try:
                    return dspy.ChatAdapter().parse(signature, completion)
                except Exception:
                    if set(out_fields) == {"next_thought", "tool_calls"}:
                        sections: dict[str, str] = {}
                        current_key: str | None = None
                        buf: list[str] = []
                        for line in completion.splitlines():
                            m = re.match(r"^\[\[\s*##\s*([^#]+?)\s*##\s*\]\]$", line.strip())
                            if m:
                                if current_key is not None:
                                    sections[current_key] = "\n".join(buf).strip()
                                current_key = m.group(1).strip()
                                buf = []
                            else:
                                buf.append(line)
                        if current_key is not None:
                            sections[current_key] = "\n".join(buf).strip()
                        coerced = _parse_react_v2_payload(sections)
                        if coerced is not None:
                            return coerced
                    raise
            import json_repair
            from dspy.adapters.utils import parse_value

            fields = json_repair.loads(completion)
            if isinstance(fields, list) and len(out_fields) == 1:
                target_field = out_fields[0]
                annotation = getattr(signature.output_fields[target_field], "annotation", Any)
                if annotation is not str:
                    try:
                        cast_val = parse_value(fields, annotation)
                    except Exception:
                        cast_val = fields
                    return {target_field: cast_val}
            if isinstance(fields, dict):
                if set(out_fields) == {"next_thought", "tool_calls"}:
                    coerced = _parse_react_v2_payload(fields)
                    if coerced is not None:
                        return coerced
                for out_f in out_fields:
                    if out_f in fields:
                        ann = getattr(signature.output_fields[out_f], "annotation", Any)
                        return {out_f: parse_value(fields[out_f], ann)}
                if len(out_fields) == 1:
                    target_field = out_fields[0]
                    ann = getattr(signature.output_fields[target_field], "annotation", Any)
                    if target_field in ("decisions", "repairs") and target_field not in fields:
                        return {target_field: []}
                    if len(fields) == 1:
                        v = next(iter(fields.values()))
                        return {target_field: parse_value(v, ann)}
                    if "answer" in fields:
                        return {target_field: parse_value(fields["answer"], ann)}
            raise


class YawGBNFAdapter(YawJSONAdapter):
    """Unified `dspy.JSONAdapter` subclass with automatic per-signature GBNF grammar injection and fallback parsing."""

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


class _PredictorBoundLM(dspy.BaseLM):
    """Per-predictor isolated `dspy.BaseLM` proxy with dedicated `.history` and telemetry."""

    def __init__(
        self,
        predictor_name: str,
        role: Literal["orchestrator", "worker"],
        base_lm_getter: Callable[[], Any],
    ) -> None:
        self._fallback_model = f"yaw-predictor/{predictor_name}"
        self.predictor_name = predictor_name
        self.role: Literal["orchestrator", "worker"] = role
        self._base_lm_getter = base_lm_getter
        super().__init__(model=self._fallback_model)

    def _resolve_target_lm(self) -> Any:
        ctx_lm = getattr(dspy.settings, "lm", None)
        if ctx_lm is not None and not isinstance(ctx_lm, _PredictorBoundLM):
            return ctx_lm
        base = self._base_lm_getter()
        if base is not None and not isinstance(base, _PredictorBoundLM):
            return base
        return None

    @property
    def model(self) -> str:
        target = self._resolve_target_lm()
        if target is not None:
            return str(getattr(target, "model", self._fallback_model))
        return self._fallback_model

    @model.setter
    def model(self, value: str) -> None:
        self._fallback_model = str(value)

    @property
    def _yaw_supports_gbnf(self) -> bool | None:
        target = self._resolve_target_lm()
        return getattr(target, "_yaw_supports_gbnf", None) if target is not None else None

    @property
    def _yaw_recipe(self) -> str | None:
        target = self._resolve_target_lm()
        return getattr(target, "_yaw_recipe", None) if target is not None else None

    def dump_state(self) -> dict[str, Any]:
        return {}

    def cache_identity(self) -> dict[str, Any]:
        """Expose effective inference settings for persistent stage-cache keys."""
        identities = {}
        for role, target in (
            ("effective", self._resolve_target_lm()),
            ("bound", self._base_lm_getter()),
        ):
            identities[role] = {
                "model": getattr(target, "model", self._fallback_model),
                "model_type": getattr(target, "model_type", None),
                "context_window_tokens": getattr(target, "_yaw_context_window_tokens", None),
                "kwargs": {
                    k: v
                    for k, v in (getattr(target, "kwargs", {}) or {}).items()
                    if k not in {"api_key", "api_token"}
                },
            }
        return identities

    def copy(self, **kwargs: Any) -> _PredictorBoundLM:
        cloned = _PredictorBoundLM(
            self.predictor_name,
            self.role,
            self._base_lm_getter,
        )
        if kwargs:
            cloned.kwargs = {**self.kwargs, **kwargs}
        return cloned

    def __deepcopy__(self, memo: dict[int, Any]) -> _PredictorBoundLM:
        cloned = _PredictorBoundLM(
            self.predictor_name,
            self.role,
            self._base_lm_getter,
        )
        memo[id(self)] = cloned
        return cloned

    def __call__(self, prompt: Any = None, *, messages: Any = None, **kwargs: Any) -> Any:
        import warnings

        target = self._resolve_target_lm()
        if target is None:
            raise ValueError(
                f"No LM is configured for predictor {self.predictor_name!r} (role={self.role!r})."
            )
        target_hist = getattr(target, "history", ())
        previous_entry = target_hist[-1] if target_hist else None
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*messages=.*",
                category=DeprecationWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=".*Pydantic serializer warnings.*",
                category=UserWarning,
            )
            if prompt is not None:
                result = target(prompt, messages=messages, **kwargs)
            else:
                result = target(messages=messages, **kwargs)
        target_hist = getattr(target, "history", None)
        if isinstance(target_hist, list):
            new_entries = (
                target_hist[-1:] if target_hist and target_hist[-1] is not previous_entry else ()
            )
            for entry in new_entries:
                if isinstance(entry, Mapping):
                    rec = dict(entry)
                    rec["predictor"] = self.predictor_name
                    self.history.append(rec)
                else:
                    self.history.append({"predictor": self.predictor_name, "entry": entry})
            del self.history[:-100]
        return result


def _sum_usage_tokens(usage: Mapping[str, Any] | None) -> int:
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


class YawPassage(dspy.Prediction):
    """Passage object returned by `YawRM` exposing `.long_text`, `.pid`, `.score`, and `.rank` for `dspy.Retrieve`."""

    def __init__(
        self,
        long_text: str,
        *,
        pid: str = "",
        score: float = 0.0,
        rank: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            long_text=str(long_text),
            pid=str(pid),
            score=float(score),
            rank=int(rank),
            **kwargs,
        )


class YawRM:
    """First-class DSPy Retrieval Model (`dspy.RM`) backed by YAW's Forward-Push PPR & recognition memory hook."""

    def __init__(
        self,
        hook: Callable[
            ...,
            tuple[str, tuple[str, ...], tuple[float, ...]]
            | tuple[str, tuple[str, ...], tuple[float, ...], tuple[str, ...]],
        ]
        | None = None,
        *,
        program: Any = None,
        default_k: int = 10,
    ) -> None:
        self.hook = hook
        self.program = program
        self.default_k = int(default_k)

    def _invoke_hook_with_pids(
        self,
        query: str,
        *,
        k: int | None = None,
        item_docs: Sequence[str] | None = None,
    ) -> tuple[str, tuple[str, ...], tuple[float, ...], tuple[str, ...]]:
        if self.hook is None:
            raise ValueError("No retrieval hook is registered on YawRM.")
        try:
            sig = inspect.signature(self.hook)
            param_names = list(sig.parameters.keys())
            has_var_kw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
            has_var_pos = any(
                p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values()
            )
            accepts_prog = len(param_names) >= 2 or has_var_pos
            accepts_item_docs = "item_docs" in sig.parameters or len(param_names) >= 3 or has_var_kw
            accepts_k = "k" in sig.parameters or has_var_kw
        except TypeError, ValueError:
            accepts_prog = False
            accepts_item_docs = False
            accepts_k = False
        kw: dict[str, Any] = {}
        if accepts_item_docs and self.program is not None:
            kw["item_docs"] = item_docs
        if accepts_k and k is not None:
            kw["k"] = int(k)
        if accepts_prog and self.program is not None:
            raw = self.hook(query, self.program, **kw)
        else:
            raw = self.hook(query, **kw)
        if isinstance(raw, tuple) and len(raw) == 4:
            ctx, passages, scores, pids = raw
            return str(ctx), tuple(passages), tuple(scores), tuple(pids)
        ctx, passages, scores = raw
        pids = tuple(
            str(p) if str(p).startswith("chunk-") else f"chunk-{i}" for i, p in enumerate(passages)
        )
        return str(ctx), tuple(passages), tuple(scores), pids

    def _invoke_hook(
        self,
        query: str,
        *,
        item_docs: Sequence[str] | None = None,
    ) -> tuple[str, tuple[str, ...], tuple[float, ...]]:
        ctx, passages, scores, _ = self._invoke_hook_with_pids(query, item_docs=item_docs)
        return ctx, passages, scores

    def __call__(
        self,
        query_or_queries: str | list[str],
        k: int | None = None,
        **kwargs: Any,
    ) -> list[YawPassage]:
        effective_k = self.default_k if k is None else int(k)
        item_docs = kwargs.get("item_docs")
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else [str(q) for q in query_or_queries]
        )
        results: list[YawPassage] = []
        for q in queries:
            _, passages, scores, pids = self._invoke_hook_with_pids(
                q, k=effective_k, item_docs=item_docs
            )
            for idx, text in enumerate(passages[:effective_k]):
                sc = float(scores[idx]) if idx < len(scores) else 0.0
                pid = str(pids[idx]) if idx < len(pids) else f"chunk-{idx}"
                results.append(
                    YawPassage(
                        long_text=str(text),
                        pid=pid,
                        score=sc,
                        rank=idx + 1,
                    )
                )
        return results


class YawRetriever(dspy.Retrieve):
    """First-class `dspy.Retrieve` module backed by `YawRM` and YAW's Forward-Push PPR + recognition memory."""

    def __init__(
        self,
        k: int = 10,
        *,
        rm: YawRM | None = None,
        hook: Callable[
            ...,
            tuple[str, tuple[str, ...], tuple[float, ...]]
            | tuple[str, tuple[str, ...], tuple[float, ...], tuple[str, ...]],
        ]
        | None = None,
    ) -> None:
        super().__init__(k=k)
        self.rm: YawRM = rm if rm is not None else YawRM(hook=hook, default_k=k)
        if hook is not None and self.rm.hook is None:
            self.rm.hook = hook

    @property
    def hook(
        self,
    ) -> (
        Callable[
            ...,
            tuple[str, tuple[str, ...], tuple[float, ...]]
            | tuple[str, tuple[str, ...], tuple[float, ...], tuple[str, ...]],
        ]
        | None
    ):
        return self.rm.hook

    @hook.setter
    def hook(
        self,
        value: Callable[
            ...,
            tuple[str, tuple[str, ...], tuple[float, ...]]
            | tuple[str, tuple[str, ...], tuple[float, ...], tuple[str, ...]],
        ]
        | None,
    ) -> None:
        self.rm.hook = value

    def dump_state(self, json_mode: bool = True, **_kwargs: Any) -> dict[str, Any]:
        _ = json_mode
        return {"k": self.k}

    def load_state(self, state: dict[str, Any], **_kwargs: Any) -> None:
        if isinstance(state, Mapping) and "k" in state:
            self.k = int(state["k"])

    def forward(
        self,
        query: str,
        k: int | None = None,
        **kwargs: Any,
    ) -> dspy.Prediction:
        effective_k = self.k if k is None else int(k)
        if self.rm.hook is not None:
            hydrated_ctx, raw_passages, raw_scores, raw_pids = self.rm._invoke_hook_with_pids(
                query, k=effective_k, item_docs=kwargs.get("item_docs")
            )
            sliced_passages = list(raw_passages[:effective_k])
            sliced_scores = list(raw_scores[:effective_k])
            return dspy.Prediction(
                passages=sliced_passages,
                context=hydrated_ctx,
                scores=sliced_scores,
                pids=tuple(raw_pids[:effective_k]),
                all_passages=tuple(raw_passages),
                all_scores=tuple(raw_scores),
                all_pids=tuple(raw_pids),
            )
        active_rm = getattr(dspy.settings, "rm", None)
        if active_rm is not None:
            hits = active_rm(query, k=effective_k, **kwargs)
            passages = [
                str(
                    getattr(
                        psg,
                        "long_text",
                        psg.get("long_text", psg) if isinstance(psg, Mapping) else psg,
                    )
                )
                for psg in hits
            ]
            scores = [
                float(
                    getattr(
                        psg,
                        "score",
                        psg.get("score", 0.0) if isinstance(psg, Mapping) else 0.0,
                    )
                )
                for psg in hits
            ]
            pids = tuple(
                str(
                    getattr(
                        psg,
                        "pid",
                        psg.get("pid", f"chunk-{i}")
                        if isinstance(psg, Mapping)
                        else (psg if str(psg).startswith("chunk-") else f"chunk-{i}"),
                    )
                )
                for i, psg in enumerate(hits)
            )
            return dspy.Prediction(
                passages=passages,
                context="\n\n".join(passages),
                scores=scores,
                pids=pids,
                all_passages=tuple(passages),
                all_scores=tuple(scores),
                all_pids=pids,
            )
        raise ValueError("No retrieval hook or dspy.settings.rm is configured for YawRetriever.")


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


class YawAssertionError(AssertionError):
    """Hard invariant violation error raised by `yaw_assert`."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.feedback = str(message)


def yaw_assert(condition: bool, message: str) -> None:
    """Assert a hard structural or closed-world invariant, raising `YawAssertionError` on failure."""
    if not condition:
        raise YawAssertionError(message)


def yaw_suggest(condition: bool, message: str) -> bool:
    """Evaluate a soft invariant suggestion without raising an exception."""
    _ = message
    return bool(condition)


_VALID_SYNONYM_RELATIONS: frozenset[str] = frozenset(
    {
        "skos:altLabel",
        "skos:hiddenLabel",
        "skos:related",
        "owl:differentFrom",
    }
)

_VALID_SHACL_ACTIONS: frozenset[str] = frozenset(
    {
        "set_definition",
        "demote_alt_to_hidden_label",
        "delete_alt_label",
        "replace_edge_with_different_from",
        "delete_semantic_edge",
        "delete_orphan_fact",
    }
)

_VALUE_BEARING_SHACL_ACTIONS: frozenset[str] = frozenset(
    {
        "set_definition",
        "demote_alt_to_hidden_label",
        "delete_alt_label",
    }
)


def _count_connected_components(triples: Sequence[StructuredTriple]) -> int:
    adj: dict[str, set[str]] = {}
    for st in triples:
        u = str(text_processing(st.subj)) or st.subj.strip().lower()
        v = str(text_processing(st.obj)) or st.obj.strip().lower()
        if not u or not v:
            continue
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    if not adj:
        return 0
    visited: set[str] = set()
    components = 0
    for node in adj:
        if node in visited:
            continue
        components += 1
        stack = [node]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            stack.extend(adj.get(cur, set()) - visited)
    return components


def validate_stage_invariants(
    stage_name: str,
    inputs: Mapping[str, Any],
    prediction: Any,
    *,
    shacl_linter: Callable[[], Sequence[Mapping[str, Any]]] | None = None,
) -> tuple[float, str]:
    """Validate all 9 stage contracts and 6 closed-world SHACL/SKOS/connected-node invariants.

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
                if (
                    not def_key
                    or def_key in seen_defs_by_norm[norm]
                    or "" in seen_defs_by_norm[norm]
                ):
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
            issues.append(
                f"ConceptShape/definition missing or tautological definitions: {bad_defs}"
            )
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
        if isinstance(stage1_raw, Sequence) and not isinstance(stage1_raw, str):
            for ent in stage1_raw:
                ename = (
                    str(ent[0]).strip()
                    if isinstance(ent, (list, tuple)) and len(ent) >= 1
                    else str(ent).strip()
                )
                if ename:
                    stage1_norms.add(str(text_processing(ename)) or ename.lower())

        closed_world_norms = set(stage1_norms)
        for st in structured:
            if st.subj and st.subj_def and st.subj_def.strip().lower() != st.subj.strip().lower():
                closed_world_norms.add(str(text_processing(st.subj)) or st.subj.lower())
            if st.obj and st.obj_def and st.obj_def.strip().lower() != st.obj.strip().lower():
                closed_world_norms.add(str(text_processing(st.obj)) or st.obj.lower())

        unregistered: list[str] = []
        self_loops: list[str] = []
        incomplete: int = 0
        for st in structured:
            if not (st.subj and st.pred and st.obj and st.spo_def):
                incomplete += 1
                continue
            s_norm = str(text_processing(st.subj)) or st.subj.lower()
            o_norm = str(text_processing(st.obj)) or st.obj.lower()
            if s_norm == o_norm:
                s_def_clean = st.subj_def.strip().lower()
                o_def_clean = st.obj_def.strip().lower()
                if not s_def_clean or not o_def_clean or s_def_clean == o_def_clean:
                    self_loops.append(st.subj)
            if stage1_norms:
                if s_norm not in closed_world_norms:
                    unregistered.append(st.subj)
                if o_norm not in closed_world_norms:
                    unregistered.append(st.obj)

        issues = []
        if incomplete:
            issues.append(f"StatementShape/closedWorld {incomplete} incomplete triple(s)")
        if self_loops:
            issues.append(f"Self-loop triple(s) on {self_loops}")
        if unregistered:
            issues.append(
                f"StatementShape/closedWorld unregistered novel entities without inline [name, def]: {unregistered}"
            )

        components = _count_connected_components(structured)
        if len(structured) >= 2 and len(stage1_norms) >= 2 and components > 1:
            issues.append(
                f"Disconnected subgraph ({components} components across {len(structured)} triples; missing bridge predicates)"
            )

        if not issues:
            return 1.0, "PASS [triples]"
        score = max(0.0, round(1.0 - 0.35 * len(issues), 4))
        return score, f"[triples] {'; '.join(issues)}"

    if stage == "entity_definer":
        missing_in = [
            str(x).strip() for x in (inputs.get("missing_entities") or ()) if str(x).strip()
        ]
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
        unrepaired = [
            m for m in missing_in if (str(text_processing(m)) or m.lower()) not in def_by_norm
        ]
        if not unrepaired:
            return 1.0, "PASS [entity_definer]"
        cov = (len(missing_in) - len(unrepaired)) / max(1, len(missing_in))
        return (
            round(cov * 0.5, 4),
            f"[entity_definer] ConceptShape/definition missing or tautological for: {unrepaired}",
        )

    if stage == "synonym_judge":
        raw_decisions = getattr(prediction, "decisions", None)
        items = list(raw_decisions) if isinstance(raw_decisions, (list, tuple)) else []
        if not items:
            return 0.0, "[synonym_judge] Empty decisions (EdgeShape/closedWorld)."
        seen_pairs: dict[str, tuple[str, str]] = {}
        issues = []
        for d in items:
            if not isinstance(d, Mapping):
                issues.append("EdgeShape/closedWorld non-dict decision")
                continue
            pair_id = str(d.get("pair_id", "") or "").strip()
            rel = str(d.get("relation", "") or "").strip()
            canon = str(d.get("canonical", "") or "").strip()
            if "|" not in pair_id:
                issues.append(f"EdgeShape/closedWorld invalid pair_id={pair_id!r}")
                continue
            id_a, id_b = [p.strip() for p in pair_id.split("|", 1)]
            if not id_a or not id_b or id_a == id_b:
                issues.append(f"EdgeShape/closedWorld self-loop or empty pair_id={pair_id!r}")
            if rel not in _VALID_SYNONYM_RELATIONS:
                issues.append(f"ConceptShape/S13-disjointLabels invalid relation={rel!r}")
            if canon not in ("A", "B", id_a, id_b):
                issues.append(f"EdgeShape/closedWorld invalid canonical={canon!r} for {pair_id!r}")
            if pair_id in seen_pairs and seen_pairs[pair_id] != (rel, canon):
                issues.append(
                    f"ConceptShape/homographDifferentFrom conflicting decisions for {pair_id!r}"
                )
            seen_pairs[pair_id] = (rel, canon)
        if not issues:
            return 1.0, "PASS [synonym_judge]"
        return max(0.0, round(1.0 - 0.35 * len(issues), 4)), f"[synonym_judge] {'; '.join(issues)}"

    if stage in ("shacl_repair", "shacl_healer_rlm"):
        raw_repairs = getattr(prediction, "repairs", None)
        repairs = (
            list(raw_repairs)
            if isinstance(raw_repairs, (list, tuple))
            else parse_shacl_repairs(raw_repairs)
        )
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
            val = str(r.get("value", "") or "").strip()
            if allowed_nodes and fnode not in allowed_nodes:
                issues.append(f"StatementShape/closedWorld out-of-scope focus_node={fnode!r}")
            if act not in _VALID_SHACL_ACTIONS:
                issues.append(f"EdgeShape/closedWorld invalid action={act!r}")
            elif act in _VALUE_BEARING_SHACL_ACTIONS and not val:
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
            fact_list = (
                parsed_before.get("fact", [])
                if isinstance(parsed_before, Mapping)
                else parsed_before
            )
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
                issues.append(
                    f"Hallucinated facts outside closed-world candidate subset: {hallucinated}"
                )
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

    if stage in ("qwen_proposer", "qwen_compressor"):
        new_inst = str(getattr(prediction, "new_instruction", "") or "").strip()
        max_chars = int(inputs.get("max_instruction_chars", 1500) or 1500)
        comp_name = str(inputs.get("component_name", "") or "").strip()
        cur_inst = str(inputs.get("current_instruction", "") or "").strip()
        if not new_inst:
            return 0.0, f"[{stage}] Empty new_instruction."
        issues = []
        if len(new_inst) > max_chars:
            issues.append(f"Instruction length {len(new_inst)} > {max_chars}")
        for marker in ("<TargetAnswer>", "<TargetPassage>", "<TargetEntity>"):
            if marker in new_inst:
                issues.append(f"Leaked placeholder marker {marker}")
        if comp_name in _STAGE_SCHEMA_CONTRACTS:
            strict_key, alt_markers, _ = _STAGE_SCHEMA_CONTRACTS[comp_name]
            had_strict = strict_key in cur_inst if cur_inst else True
            has_strict = strict_key in new_inst
            has_any = any(m.lower() in new_inst.lower() for m in alt_markers)
            if not (has_strict or (not had_strict and has_any)):
                issues.append(f"Missing required stage schema key {strict_key!r}")
        if not issues:
            return 1.0, f"PASS [{stage}]"
        return max(0.0, round(1.0 - 0.4 * len(issues), 4)), f"[{stage}] {'; '.join(issues)}"

    return 1.0, f"PASS [{stage}]"


def wrap_stage_with_refine(
    module: dspy.Module,
    stage_name: str,
    *,
    N: int = 2,
    threshold: float = 1.0,
    use_refine: bool = True,
    shacl_linter: Callable[[], Sequence[Mapping[str, Any]]] | None = None,
) -> dspy.Module:
    """Wrap a DSPy stage predictor in native `dspy.Refine` or `dspy.BestOfN` scored by `validate_stage_invariants`."""

    def stage_reward_fn(inputs: dict[str, Any], pred: dspy.Prediction) -> float:
        score, _ = validate_stage_invariants(
            stage_name,
            inputs,
            pred,
            shacl_linter=shacl_linter,
        )
        return float(score)

    wrapper_cls = getattr(dspy, "Refine", None) if use_refine else getattr(dspy, "BestOfN", None)
    if wrapper_cls is None:
        wrapper_cls = dspy.BestOfN
    return wrapper_cls(
        module=module,
        N=max(1, int(N)),
        reward_fn=stage_reward_fn,
        threshold=float(threshold),
    )


def _should_retry_stage_invariants(
    stage_name: str,
    pred: Any,
    effective_lm: Any,
) -> bool:
    """Determine if `_invoke_stage` can perform a self-correction retry without exhausting scripted `DummyLM` queues."""
    import copy

    if stage_name == "triples":
        # Stage 2 missing definitions are repaired by Stage 3 (DefineMissingEntities)
        return False
    if stage_name == "ner":
        raw_entities = getattr(pred, "named_entities", None)
        if (
            isinstance(raw_entities, (list, tuple))
            and raw_entities
            and not any(isinstance(x, (list, tuple, Mapping)) for x in raw_entities)
        ):
            return False

    target_lm = (
        effective_lm._resolve_target_lm()
        if isinstance(effective_lm, _PredictorBoundLM)
        else effective_lm
    )
    if target_lm is None:
        return False
    answers_it = getattr(target_lm, "answers", None)
    if answers_it is not None and not isinstance(answers_it, Mapping):
        try:
            peek_it = copy.copy(answers_it)
            next_ans = next(peek_it, None)
        except TypeError:
            return False
        if not isinstance(next_ans, Mapping):
            return False
        expected_key_map = {
            "ner": "named_entities",
            "entity_definer": "entity_definitions",
            "synonym_judge": "decisions",
            "shacl_repair": "repairs",
            "triple_filter": "fact",
            "query_intent": "query_intent",
            "qa": "answer",
        }
        exp_key = expected_key_map.get(stage_name)
        if exp_key and exp_key not in next_ans:
            return False
    return True


class HippoRAGProgram(dspy.Module):
    """Unified DSPy Module for HippoRAG 2 OpenIE, EDC/SKOS Schema Canon, SHACL Repair, Recognition Memory, and Multi-Hop QA.

    Exposes the 9 canonical runtime stage predictors (out of the 12 total DSPy signatures in YAW):
    1. **8 Direct GBNF Predictors (configured chat model on `llama.cpp` GPU):**
       - `ner` (`ExtractEntities`)
       - `triples` (`ExtractTriples`)
       - `entity_definer` (`DefineMissingEntities`)
       - `synonym_judge` (`JudgeSynonyms`)
       - `shacl_repair` (`ProposeSHACLRepairs`)
       - `triple_filter` (`FilterTriples`)
       - `query_intent` (`ExtractQueryIntent`)
       - `qa` (`AnswerQuestion`)
    2. **1 Single-Level RLM Python REPL Orchestrator (`shacl_healer_rlm`):**
       - `shacl_healer_rlm` (`HealSHACLViolations`) running inside `dspy.PythonInterpreter` (managed Deno/Pyodide WASM) with
         bound graph inspection, hydration, repair, and SQL SHACL linting tools (`inspect_focus_node`,
         `hydrate_chunk`, `qwen_propose_shacl_repairs`, `apply_shacl_repairs`, `lint_shacl`).
    3. **First-Class Embedding Trail (`record_embedding_telemetry` / `embedding_audit_trail`):**
       - Tracks every `DSPyEmbeddingAdapter` (`dspy.Embedder`) invocation and the 3 optimizable
         task prefixes (`embedding_query_prefix`, `embedding_fact_prefix`, `embedding_document_prefix`)
         in `self.telemetry` and `self.embedding_audit_trail`.
    """

    def __init__(
        self,
        *,
        include_demos: bool = True,
        enable_gbnf: bool = False,
        gbnf_max_entities: int | None = None,
        gbnf_max_triples: int | None = None,
        gbnf_max_facts: int = 4,
        gbnf_max_reasoning_chars: int | None = None,
        embedding_query_prefix: str | None = None,
        embedding_fact_prefix: str | None = None,
        embedding_document_prefix: str | None = None,
        adapter: dspy.Adapter | None = None,
        orchestrator_lm: Any = None,
        rlm_max_iters: int = 10,
        rlm_max_llm_calls: int = 20,
        rlm_max_output_chars: int = 10000,
        rlm_sub_lm: Any = None,
        rlm_tools: list[Callable[..., Any]] | None = None,
        rlm_interpreter_factory: Callable[[], Any] = dspy.PythonInterpreter,
        compiled_path: str | Path | None = None,
        enable_react_fallback: bool = False,
        context_window_tokens: int = 32768,
        max_output_tokens: int = 4096,
    ) -> None:
        super().__init__()
        context_char_limit(1, context_window_tokens, max_output_tokens)
        self.context_window_tokens = context_window_tokens
        self.max_output_tokens = max_output_tokens
        self.enable_gbnf: bool = enable_gbnf
        self.gbnf_max_entities: int | None = gbnf_max_entities
        self.gbnf_max_triples: int | None = gbnf_max_triples
        self.gbnf_max_facts: int = gbnf_max_facts
        self.gbnf_max_reasoning_chars: int | None = gbnf_max_reasoning_chars
        self.embedding_query_prefix: str | None = embedding_query_prefix
        self.embedding_fact_prefix: str | None = embedding_fact_prefix
        self.embedding_document_prefix: str | None = embedding_document_prefix
        self.json_adapter: dspy.Adapter = adapter or YawGBNFAdapter(
            enable_gbnf=self.enable_gbnf,
            max_entities=self.gbnf_max_entities,
            max_triples=self.gbnf_max_triples,
            max_facts=self.gbnf_max_facts,
            max_reasoning_chars=self.gbnf_max_reasoning_chars,
        )
        self.orchestrator_lm: dspy.LM | None = orchestrator_lm
        self.rlm_sub_lm: dspy.LM | None = rlm_sub_lm
        self._retrieval_hook: (
            Callable[
                ...,
                tuple[str, tuple[str, ...], tuple[float, ...]]
                | tuple[str, tuple[str, ...], tuple[float, ...], tuple[str, ...]],
            ]
            | None
        ) = None
        self.rm: YawRM = YawRM(hook=None, program=self, default_k=10)
        self.retriever: YawRetriever = YawRetriever(k=10, rm=self.rm)
        self._node_inspector: Callable[[str], dict[str, Any]] | None = None
        self._chunk_hydrator: Callable[[str], str] | None = None
        self._shacl_repair_applier: Callable[[Sequence[Mapping[str, Any]]], int] | None = None
        self._shacl_linter: Callable[[], list[dict[str, str]]] | None = None
        self._graph_search_tool: Callable[[str], str] | None = None
        self._last_ner_definitions: dict[str, str] = {}
        self.max_invariant_retries: int = 1
        self._in_ensemble: bool = False
        self.enable_react_fallback: bool = bool(enable_react_fallback)
        self.pareto_frontier_states: list[dict[str, Any]] = []
        self.pareto_ensemble: ParetoProgramEnsemble | None = None
        self.telemetry: list[dict[str, Any]] = DiagnosticHistory()
        self.history = DiagnosticHistory()
        self.embedding_audit_trail: list[dict[str, Any]] = DiagnosticHistory()
        self.entity_def_audit_trail: list[dict[str, Any]] = DiagnosticHistory()
        self.synonym_audit_trail: list[dict[str, Any]] = DiagnosticHistory()
        self.shacl_audit_trail: list[dict[str, Any]] = DiagnosticHistory()

        # 8 Direct GBNF Predictors with distinct Signatures
        self.ner = dspy.Predict(ExtractEntities)
        self.triples = dspy.Predict(ExtractTriples)
        self.entity_definer = dspy.Predict(DefineMissingEntities)
        self.synonym_judge = dspy.Predict(JudgeSynonyms)
        self.shacl_repair = dspy.Predict(ProposeSHACLRepairs)
        self.triple_filter = dspy.Predict(FilterTriples)
        self.query_intent = dspy.Predict(ExtractQueryIntent)
        self.qa = dspy.ChainOfThought(AnswerQuestion)

        shacl_tools = list(self._build_shacl_repl_tools())
        if rlm_tools:
            existing = {getattr(t, "__name__", "") for t in shacl_tools}
            for user_tool in rlm_tools:
                name = getattr(user_tool, "__name__", "")
                if name not in existing:
                    existing.add(name)
                    shacl_tools.append(user_tool)

        # 1 Single-Level RLM Python REPL Predictor for Closed-World SHACL Healing
        self.shacl_healer_rlm = _PositionalSafeRLM(
            HealSHACLViolations,
            stage_rule=_STAGE_RLM_DELEGATION_RULES["shacl_repair"],
            max_iters=rlm_max_iters,
            max_llm_calls=rlm_max_llm_calls,
            max_output_chars=rlm_max_output_chars,
            tools=shacl_tools,
            sub_lm=rlm_sub_lm or orchestrator_lm,
            interpreter_factory=rlm_interpreter_factory,
        )

        self._bind_predictor_lms()

        if include_demos:
            self.ner.demos = list(NER_DEMOS)
            self.triples.demos = list(TRIPLE_DEMOS)
            self.triple_filter.demos = list(TRIPLE_FILTER_DEMOS)
            self.qa.predict.demos = list(QA_DEMOS)
            rlm_demos = _build_rlm_action_demos()
            self.shacl_healer_rlm.generate_action.demos = list(rlm_demos["shacl_repair"])

        if compiled_path is not None:
            path = Path(compiled_path)
            if path.exists():
                self.load(str(path))

    @property
    def shacl_repair_rlm(self) -> _PositionalSafeRLM:
        return self.shacl_healer_rlm

    @shacl_repair_rlm.setter
    def shacl_repair_rlm(self, value: _PositionalSafeRLM) -> None:
        self.shacl_healer_rlm = value

    def record_embedding_telemetry(self, entry: Mapping[str, Any]) -> None:
        """Record a DSPyEmbeddingAdapter invocation in both telemetry and embedding_audit_trail."""
        record = dict(entry)
        self.embedding_audit_trail.append(record)
        self.telemetry.append(record)

    def _bind_predictor_lms(self) -> None:
        """Bind a dedicated `_PredictorBoundLM` instance with isolated `.history` to all 9 stage predictors."""

        def worker_getter() -> Any:
            return self.rlm_sub_lm or self.orchestrator_lm

        def orch_getter() -> Any:
            return self.orchestrator_lm or self.rlm_sub_lm

        for name, pred in (
            ("ner", self.ner),
            ("triples", self.triples),
            ("entity_definer", self.entity_definer),
            ("synonym_judge", self.synonym_judge),
            ("shacl_repair", self.shacl_repair),
            ("triple_filter", self.triple_filter),
            ("query_intent", self.query_intent),
        ):
            pred.lm = _PredictorBoundLM(name, "worker", worker_getter)

        qa_lm = _PredictorBoundLM("qa", "orchestrator", orch_getter)
        self.qa.predict.lm = qa_lm

        action_lm = _PredictorBoundLM(
            "shacl_healer_rlm.generate_action", "orchestrator", orch_getter
        )
        extract_lm = _PredictorBoundLM("shacl_healer_rlm.extract", "orchestrator", orch_getter)
        self.shacl_healer_rlm.generate_action.lm = action_lm
        self.shacl_healer_rlm.extract.lm = extract_lm
        self.shacl_healer_rlm.lm = action_lm

    def all_stage_predictors(self) -> list[tuple[str, Any]]:
        """Return all 9 canonical stage predictors (8 direct GBNF + 1 SHACL RLM healer)."""
        return [
            ("ner", self.ner),
            ("triples", self.triples),
            ("entity_definer", self.entity_definer),
            ("synonym_judge", self.synonym_judge),
            ("shacl_repair", self.shacl_repair),
            ("shacl_healer_rlm", self.shacl_healer_rlm),
            ("triple_filter", self.triple_filter),
            ("query_intent", self.query_intent),
            ("qa", self.qa.predict),
        ]

    def set_graph_tools(
        self,
        *,
        node_inspector: Callable[[str], dict[str, Any]] | None = None,
        chunk_hydrator: Callable[[str], str] | None = None,
        shacl_repair_applier: Callable[[Sequence[Mapping[str, Any]]], int] | None = None,
        shacl_linter: Callable[[], list[dict[str, str]]] | None = None,
    ) -> None:
        """Bind storage tools (`inspect_focus_node`, `hydrate_chunk`, `apply_shacl_repairs`, `lint_shacl`) for the SHACL RLM sandbox."""
        self._node_inspector = node_inspector
        self._chunk_hydrator = chunk_hydrator
        self._shacl_repair_applier = shacl_repair_applier
        self._shacl_linter = shacl_linter

    def set_graph_search_tool(
        self,
        search_fn: Callable[[str], str] | None,
    ) -> None:
        """Bind a graph fact search callback for multi-hop `dspy.ReActV2` fallback exploration."""
        self._graph_search_tool = search_fn

    @classmethod
    def from_config(cls, config: YawConfig) -> HippoRAGProgram:
        """Construct a HippoRAGProgram from a YawConfig."""
        orch_lm = create_dspy_lm(
            config.qa_model,
            base_url=config.base_url,
            api_key=config.api_key,
            provider=config.provider,
            role="qa",
            context_window_tokens=config.context_window_tokens,
            max_tokens=config.max_output_tokens,
        )
        sub_lm = create_dspy_lm(
            config.extraction_model,
            base_url=config.base_url,
            api_key=config.api_key,
            provider=config.provider,
            role="extraction",
            context_window_tokens=config.context_window_tokens,
            max_tokens=config.max_output_tokens,
        )
        program = cls(
            enable_gbnf=config.enable_gbnf,
            gbnf_max_entities=config.gbnf_max_entities,
            gbnf_max_triples=config.gbnf_max_triples,
            gbnf_max_facts=config.gbnf_max_facts,
            gbnf_max_reasoning_chars=config.gbnf_max_reasoning_chars,
            orchestrator_lm=orch_lm,
            rlm_sub_lm=sub_lm,
            compiled_path=config.compiled_program_path,
            context_window_tokens=config.context_window_tokens,
            max_output_tokens=config.max_output_tokens,
            rlm_max_iters=config.rlm_max_iters,
            rlm_max_llm_calls=config.rlm_max_llm_calls,
            rlm_max_output_chars=config.rlm_max_output_chars,
        )
        # Compiled prompts must not override the configured serving capacity.
        members = list(program.pareto_ensemble.programs) if program.pareto_ensemble else []
        for target in [program, *members]:
            target.context_window_tokens = config.context_window_tokens
            target.max_output_tokens = config.max_output_tokens
            target.shacl_healer_rlm.max_iters = config.rlm_max_iters
            target.shacl_healer_rlm.max_llm_calls = config.rlm_max_llm_calls
            target.shacl_healer_rlm.max_output_chars = config.rlm_max_output_chars
        if members:
            program.pareto_frontier_states = [member.dump_state() for member in members]
        return program

    def _context_char_limit(self, base_chars: int) -> int:
        return context_char_limit(base_chars, self.context_window_tokens, self.max_output_tokens)

    def dump_state(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Serialize all DSPy predictors/RLMs plus EmbeddingGemma prefixes, GBNF structural bounds, and Pareto frontier."""
        state: dict[str, Any] = super().dump_state(*args, **kwargs)
        yaw_surface: dict[str, Any] = {
            "gbnf_max_entities": self.gbnf_max_entities,
            "gbnf_max_triples": self.gbnf_max_triples,
            "gbnf_max_facts": self.gbnf_max_facts,
            "gbnf_max_reasoning_chars": self.gbnf_max_reasoning_chars,
            "embedding_query_prefix": self.embedding_query_prefix,
            "embedding_fact_prefix": self.embedding_fact_prefix,
            "embedding_document_prefix": self.embedding_document_prefix,
            "context_window_tokens": self.context_window_tokens,
            "max_output_tokens": self.max_output_tokens,
        }
        if self.pareto_frontier_states:
            yaw_surface["pareto_frontier_states"] = list(self.pareto_frontier_states)
        state["yaw_surface"] = yaw_surface
        return state

    def load_state(self, state: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        """Restore DSPy predictor/RLM state plus EmbeddingGemma prefixes, GBNF structural bounds, and Pareto frontier."""
        state_copy = dict(state)
        state_copy.setdefault("retriever", {"k": getattr(self.retriever, "k", 10)})
        yaw_surface = state_copy.pop("yaw_surface", None)
        # Missing legacy predictors retain constructor defaults. Explicitly
        # saved empty demo lists must survive loading without reseeding.
        for p_name, param in self.named_parameters():
            if p_name not in state_copy and hasattr(param, "dump_state"):
                state_copy[p_name] = param.dump_state()
        res = super().load_state(state_copy, *args, **kwargs)
        if isinstance(yaw_surface, dict):
            for attr in (
                "gbnf_max_entities",
                "gbnf_max_triples",
                "gbnf_max_reasoning_chars",
                "embedding_query_prefix",
                "embedding_fact_prefix",
                "embedding_document_prefix",
                "context_window_tokens",
                "max_output_tokens",
            ):
                if attr in yaw_surface:
                    setattr(self, attr, yaw_surface[attr])
            if yaw_surface.get("gbnf_max_facts") is not None:
                self.gbnf_max_facts = int(yaw_surface["gbnf_max_facts"])
            frontier_states = yaw_surface.get("pareto_frontier_states")
            if isinstance(frontier_states, list) and frontier_states:
                self.pareto_frontier_states = list(frontier_states)
                self.pareto_ensemble = ParetoProgramEnsemble.from_frontier_states(
                    self, self.pareto_frontier_states
                )
        if isinstance(self.json_adapter, YawGBNFAdapter):
            self.json_adapter.enable_gbnf = self.enable_gbnf
            self.json_adapter.max_entities = self.gbnf_max_entities
            self.json_adapter.max_triples = self.gbnf_max_triples
            self.json_adapter.max_facts = self.gbnf_max_facts
            self.json_adapter.max_reasoning_chars = self.gbnf_max_reasoning_chars
        self._bind_predictor_lms()
        return res

    def __deepcopy__(self, memo: dict[int, Any]) -> HippoRAGProgram:
        """Clone all DSPy predictors while preserving live LM clients and the engine retrieval hook."""
        import copy

        cls = self.__class__
        new_prog = cls.__new__(cls)
        memo[id(self)] = new_prog
        for k, v in self.__dict__.items():
            if k in (
                "_retrieval_hook",
                "_node_inspector",
                "_chunk_hydrator",
                "_shacl_repair_applier",
                "_shacl_linter",
                "_graph_search_tool",
                "orchestrator_lm",
                "rlm_sub_lm",
                "json_adapter",
                "rm",
                "retriever",
                "pareto_ensemble",
            ):
                setattr(new_prog, k, v)
            else:
                setattr(new_prog, k, copy.deepcopy(v, memo))
        new_prog.rm = YawRM(
            hook=self._retrieval_hook,
            program=new_prog,
            default_k=getattr(self.retriever, "k", 10),
        )
        new_prog.retriever = YawRetriever(
            k=getattr(self.retriever, "k", 10),
            rm=new_prog.rm,
        )
        new_prog._bind_predictor_lms()
        return new_prog

    def _resolve_worker_lm(self, fallback_lm: dspy.LM | None = None) -> dspy.LM | None:
        return self.rlm_sub_lm if self.rlm_sub_lm is not None else fallback_lm

    def _resolve_orchestrator_lm(self, fallback_lm: dspy.LM | None = None) -> dspy.LM | None:
        return fallback_lm if fallback_lm is not None else self.orchestrator_lm

    def _build_shacl_repl_tools(self) -> list[Callable[..., Any]]:
        """Build the closed-world SHACL healing tools exposed inside `shacl_healer_rlm`."""

        def qwen_propose_shacl_repairs(
            violations: Any = None,
            focus_context: str = "",
            violations_json: Any = None,
        ) -> list[dict[str, str]]:
            """Propose bounded closed-world SHACL repairs using Qwen GPU GBNF."""
            worker_lm = self._resolve_worker_lm()
            raw = violations if violations is not None else violations_json
            return self.propose_shacl_repairs(
                raw,
                focus_context=focus_context,
                mode="direct",
                lm=worker_lm,
                use_gbnf=self.enable_gbnf,
            )

        def inspect_focus_node(entity_id: str) -> dict[str, Any]:
            """Inspect focus node labels, definition, semantic edges, facts, and chunk pointers."""
            if self._node_inspector is None:
                return {}
            return dict(self._node_inspector(str(entity_id)))

        def hydrate_chunk(chunk_id: str) -> str:
            """Hydrate source chunk lines from disk for a given chunk_id."""
            if self._chunk_hydrator is None:
                return ""
            return str(self._chunk_hydrator(str(chunk_id)))

        def read_source_chunk(chunk_id: str) -> str:
            """Hydrate source chunk lines from disk for a given chunk_id."""
            return hydrate_chunk(chunk_id)

        def apply_shacl_repairs(repairs: Any) -> int:
            """Apply bounded SHACL repair actions to the active in-transaction database connection."""
            if self._shacl_repair_applier is None:
                return 0
            parsed = parse_shacl_repairs(repairs)
            return int(self._shacl_repair_applier(parsed))

        def lint_shacl() -> list[dict[str, str]]:
            """Re-run closed-world SQL SHACL validation on the active in-transaction database connection."""
            if self._shacl_linter is None:
                return []
            return list(self._shacl_linter())

        return [
            qwen_propose_shacl_repairs,
            inspect_focus_node,
            hydrate_chunk,
            read_source_chunk,
            apply_shacl_repairs,
            lint_shacl,
        ]

    def _dspy_ctx(self, lm: dspy.LM | None, *, use_json_adapter: bool = True):
        ctx_kwargs: dict[str, Any] = {}
        if lm is not None:
            ctx_kwargs["lm"] = lm
        if use_json_adapter:
            ctx_kwargs["adapter"] = self.json_adapter
        return dspy.context(**ctx_kwargs)

    def _should_use_gbnf(self, use_gbnf: bool | None, lm: dspy.LM | None) -> bool:
        requested = self.enable_gbnf if use_gbnf is None else use_gbnf
        return bool(requested and lm_supports_gbnf(lm))

    def _record_stage_telemetry(
        self,
        stage: str,
        predictor_name: str,
        *,
        signature_name: str = "",
        is_rlm: bool,
        usage: dict[str, Any] | None,
        inputs_summary: str = "",
        outputs_summary: str = "",
        invariant_score: float = 1.0,
        invariant_feedback: str = "",
        invariant_retries: int = 0,
    ) -> int:
        tokens_used = _sum_usage_tokens(usage)
        self.telemetry.append(
            {
                "stage": stage,
                "predictor": predictor_name,
                "signature": signature_name or predictor_name,
                "mode": "rlm" if is_rlm else "direct",
                "tokens_used": tokens_used,
                "usage": dict(usage or {}),
                "inputs_summary": inputs_summary[:240],
                "outputs_summary": outputs_summary[:240],
                "invariant_score": float(invariant_score),
                "invariant_feedback": str(invariant_feedback)[:240],
                "invariant_retries": int(invariant_retries),
            }
        )
        return tokens_used

    def _invoke_stage(
        self,
        direct_mod: Any,
        rlm_mod: Any = None,
        *,
        stage_name: str = "",
        is_rlm: bool = False,
        lm: Any,
        use_gbnf: bool | None,
        grammar_fn: Callable[[], str],
        **inputs: Any,
    ) -> Any:
        from dspy.utils.usage_tracker import track_usage

        retry_count = 0
        inv_score = 1.0
        inv_feedback = ""
        with track_usage() as usage_tracker:
            if is_rlm and rlm_mod is not None:
                orch_lm = self._resolve_orchestrator_lm(lm)
                with self._dspy_ctx(orch_lm, use_json_adapter=False):
                    pred = rlm_mod(**inputs)
                active_mod = rlm_mod
                active_name = "shacl_healer_rlm"
                inv_score, inv_feedback = validate_stage_invariants(
                    stage_name or active_name,
                    inputs,
                    pred,
                    shacl_linter=self._shacl_linter,
                )
            else:
                effective_lm = lm
                if effective_lm is None:
                    if direct_mod is self.qa:
                        effective_lm = self.orchestrator_lm or self.rlm_sub_lm
                    else:
                        effective_lm = self.rlm_sub_lm or self.orchestrator_lm
                direct_inputs = dict(inputs)
                if isinstance(direct_inputs.get("passage"), str) and len(
                    direct_inputs["passage"]
                ) > self._context_char_limit(24000):
                    limit = self._context_char_limit(24000)
                    direct_inputs["passage"] = direct_inputs["passage"][:limit]
                if isinstance(direct_inputs.get("context"), str) and len(
                    direct_inputs["context"]
                ) > self._context_char_limit(40000):
                    limit = self._context_char_limit(40000)
                    direct_inputs["context"] = direct_inputs["context"][:limit]
                call_kwargs: dict[str, Any] = {}
                if self._should_use_gbnf(use_gbnf, effective_lm):
                    call_kwargs["config"] = {"extra_body": {"grammar": grammar_fn()}}
                with self._dspy_ctx(effective_lm, use_json_adapter=True):
                    pred = direct_mod(**direct_inputs, **call_kwargs)
                    inv_score, inv_feedback = validate_stage_invariants(
                        stage_name,
                        direct_inputs,
                        pred,
                        shacl_linter=None,
                    )
                    max_retries = 0 if self._in_ensemble else self.max_invariant_retries
                    while (
                        inv_score < 1.0
                        and retry_count < max_retries
                        and _should_retry_stage_invariants(stage_name, pred, effective_lm)
                    ):
                        retry_count += 1
                        retry_cfg = dict(call_kwargs.get("config") or {})
                        retry_cfg["rollout_id"] = retry_count
                        retry_kwargs = {**call_kwargs, "config": retry_cfg}
                        try:
                            cand_pred = direct_mod(**direct_inputs, **retry_kwargs)
                        except Exception:
                            break
                        cand_score, cand_fb = validate_stage_invariants(
                            stage_name,
                            direct_inputs,
                            cand_pred,
                            shacl_linter=None,
                        )
                        if cand_score >= inv_score:
                            pred = cand_pred
                            inv_score = cand_score
                            inv_feedback = cand_fb
                active_mod = direct_mod
                active_name = "qa" if stage_name == "qa" else (stage_name or "direct")
        tokens = usage_tracker.get_total_tokens()
        if tokens and hasattr(pred, "set_lm_usage"):
            pred.set_lm_usage(tokens)
        if stage_name:
            sig_obj = getattr(active_mod, "signature", None)
            sig_name = getattr(sig_obj, "__name__", str(sig_obj or active_name))
            self._record_stage_telemetry(
                stage_name,
                active_name,
                signature_name=sig_name,
                is_rlm=bool(is_rlm and rlm_mod is not None),
                usage=tokens,
                inputs_summary=str(list(inputs.keys())),
                outputs_summary=str(pred)[:240],
                invariant_score=inv_score,
                invariant_feedback=inv_feedback,
                invariant_retries=retry_count,
            )
        return pred

    def _run_ner_stage(
        self,
        passage: str,
        *,
        lm: Any = None,
        use_gbnf: bool | None = None,
    ) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
        pred = self._invoke_stage(
            self.ner,
            stage_name="ner",
            is_rlm=False,
            lm=lm,
            use_gbnf=use_gbnf,
            grammar_fn=lambda: build_dspy_ner_grammar(self.gbnf_max_entities),
            passage=passage,
        )
        raw_entities = getattr(pred, "named_entities", [])
        parsed_defs = extract_entity_definitions(raw_entities)
        enriched_defs: list[tuple[str, str]] = [
            (ename, edef or self._last_ner_definitions.get(ename, ""))
            for ename, edef in parsed_defs
        ]
        entities = tuple(ename for ename, _ in enriched_defs)
        return entities, tuple(enriched_defs)

    def extract_entity_definitions(
        self,
        passage: str,
        *,
        lm: Any = None,
        use_gbnf: bool | None = None,
    ) -> tuple[tuple[str, str], ...]:
        """Run DSPy NER extraction and return deduplicated (entity_name, contextual_definition) pairs."""
        _, entity_defs = self._run_ner_stage(passage, lm=lm, use_gbnf=use_gbnf)
        return entity_defs

    def extract_entities(
        self,
        passage: str,
        *,
        lm: Any = None,
        use_gbnf: bool | None = None,
    ) -> tuple[str, ...]:
        """Run DSPy NER extraction on Qwen w/ GBNF."""
        entities, _ = self._run_ner_stage(passage, lm=lm, use_gbnf=use_gbnf)
        return entities

    def define_missing_entities(
        self,
        passage: str,
        missing_entities: Sequence[str],
        triples: Sequence[Any] = (),
        *,
        lm: Any = None,
        use_gbnf: bool | None = None,
        _record_audit: bool = True,
    ) -> tuple[tuple[str, str], ...]:
        """Synthesize contextual definitions via DefineMissingEntities for triple entities lacking definitions."""
        deduped: list[str] = []
        for item in missing_entities:
            name = str(item).strip()
            if name and name not in deduped:
                deduped.append(name)
        if not deduped:
            return ()

        triples_payload: list[Any] = []
        for t in triples:
            if isinstance(t, StructuredTriple):
                triples_payload.append(list(t.as_tuple()))
            elif isinstance(t, (list, tuple, dict)):
                triples_payload.append(t)

        prev_telem_len = len(self.telemetry)
        pred = self._invoke_stage(
            self.entity_definer,
            stage_name="entity_definer",
            is_rlm=False,
            lm=lm,
            use_gbnf=use_gbnf,
            grammar_fn=lambda: build_dspy_missing_entity_defs_grammar(deduped),
            passage=passage,
            missing_entities=list(deduped),
            triples=triples_payload,
        )
        raw_defs = getattr(pred, "entity_definitions", getattr(pred, "named_entities", []))
        parsed = extract_entity_definitions(raw_defs)
        exact_map: dict[str, str] = {}
        norm_map: dict[str, str] = {}
        for ename, edef in parsed:
            if ename and edef:
                exact_map[ename] = edef
                norm_map[str(text_processing(ename))] = edef

        resolved: list[tuple[str, str]] = []
        seen_names: set[str] = set()
        for idx, m_name in enumerate(deduped):
            defn = exact_map.get(m_name) or norm_map.get(str(text_processing(m_name))) or ""
            if not defn and len(parsed) == len(deduped):
                defn = parsed[idx][1]
            if defn:
                resolved.append((m_name, defn))
                seen_names.add(m_name)
                if not _record_audit:
                    self._last_ner_definitions[m_name] = defn
        for ename, edef in parsed:
            if ename and edef and ename not in seen_names:
                resolved.append((ename, edef))
                seen_names.add(ename)
                if not _record_audit:
                    self._last_ner_definitions[ename] = edef

        stage_tokens = (
            int(self.telemetry[-1].get("tokens_used", 0))
            if len(self.telemetry) > prev_telem_len
            else 0
        )
        if _record_audit:
            self.entity_def_audit_trail.append(
                {
                    "mode": "direct",
                    "predictor": "entity_definer",
                    "signature": DefineMissingEntities.__name__,
                    "missing_count": len(deduped),
                    "repaired_count": len(resolved),
                    "tokens": stage_tokens,
                    "tokens_used": stage_tokens,
                    "missing_entities": list(deduped),
                    "definitions": [[k, v] for k, v in resolved],
                }
            )
        return tuple(resolved)

    def extract_structured_triples(
        self,
        passage: str,
        named_entities: Sequence[Any],
        *,
        entity_definitions: Sequence[tuple[str, str]] | dict[str, str] | None = None,
        lm: Any = None,
        use_gbnf: bool | None = None,
        _repair_missing_defs: bool = True,
    ) -> tuple[StructuredTriple, ...]:
        """Run DSPy triple extraction and return structured triples with contextual definitions."""
        merged_defs: dict[str, str] = {}
        all_def_pairs: list[tuple[str, str]] = []
        defined_entity_names: list[str] = []
        for k, v in self._last_ner_definitions.items():
            if k and v:
                merged_defs[k] = v
                merged_defs[str(text_processing(k))] = v
                all_def_pairs.append((k, v))
        if isinstance(entity_definitions, Mapping):
            for k, v in entity_definitions.items():
                ck = str(k).strip()
                cv = str(v).strip()
                if ck and cv:
                    merged_defs[ck] = cv
                    merged_defs[str(text_processing(ck))] = cv
                    all_def_pairs.append((ck, cv))
                    if ck not in defined_entity_names:
                        defined_entity_names.append(ck)
        elif isinstance(entity_definitions, Sequence):
            for pair in entity_definitions:
                if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                    ck = str(pair[0]).strip()
                    cv = str(pair[1]).strip() if pair[1] is not None else ""
                    if ck and cv:
                        merged_defs[ck] = cv
                        merged_defs[str(text_processing(ck))] = cv
                        all_def_pairs.append((ck, cv))
                        if ck not in defined_entity_names:
                            defined_entity_names.append(ck)
        for item in named_entities:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                k = str(item[0]).strip()
                v = str(item[1]).strip() if item[1] is not None else ""
                if k and v:
                    merged_defs[k] = v
                    merged_defs[str(text_processing(k))] = v
                    all_def_pairs.append((k, v))
                    if k not in defined_entity_names:
                        defined_entity_names.append(k)

        known_entities_for_grammar: list[str] = defined_entity_names or [
            (str(x[0]).strip() if isinstance(x, (list, tuple)) and len(x) >= 1 else str(x).strip())
            for x in named_entities
            if (
                str(x[0]).strip()
                if isinstance(x, (list, tuple)) and len(x) >= 1
                else str(x).strip()
            )
        ]

        pred = self._invoke_stage(
            self.triples,
            stage_name="triples",
            is_rlm=False,
            lm=lm,
            use_gbnf=use_gbnf,
            grammar_fn=lambda: build_dspy_triples_grammar(
                self.gbnf_max_triples,
                known_entities=known_entities_for_grammar,
            ),
            passage=passage,
            named_entities=list(named_entities),
        )

        raw_triples = getattr(pred, "triples", [])
        structured = list(
            extract_structured_triples(raw_triples, entity_definitions=all_def_pairs or merged_defs)
        )

        # Propagate any inline [name, def] pairs across all triples in the same chunk first
        for st in structured:
            if st.subj and st.subj_def:
                merged_defs[st.subj] = st.subj_def
                merged_defs[str(text_processing(st.subj))] = st.subj_def
            if st.obj and st.obj_def:
                merged_defs[st.obj] = st.obj_def
                merged_defs[str(text_processing(st.obj))] = st.obj_def

        missing_names: list[str] = []
        for idx, st in enumerate(structured):
            s_def = (
                st.subj_def
                or merged_defs.get(st.subj, "")
                or merged_defs.get(str(text_processing(st.subj)), "")
            )
            o_def = (
                st.obj_def
                or merged_defs.get(st.obj, "")
                or merged_defs.get(str(text_processing(st.obj)), "")
            )
            if s_def != st.subj_def or o_def != st.obj_def:
                structured[idx] = StructuredTriple(
                    subj=st.subj,
                    subj_def=s_def,
                    pred=st.pred,
                    obj=st.obj,
                    obj_def=o_def,
                    spo_def=st.spo_def,
                )
            if st.subj and not s_def and st.subj not in missing_names:
                missing_names.append(st.subj)
            if st.obj and not o_def and st.obj not in missing_names:
                missing_names.append(st.obj)

        if _repair_missing_defs and missing_names:
            repaired_defs = self.define_missing_entities(
                passage,
                missing_names,
                triples=structured,
                lm=lm,
                use_gbnf=use_gbnf,
            )
            for ename, edef in repaired_defs:
                if ename and edef:
                    merged_defs[ename] = edef
                    merged_defs[str(text_processing(ename))] = edef
            for idx, st in enumerate(structured):
                s_def = (
                    st.subj_def
                    or merged_defs.get(st.subj, "")
                    or merged_defs.get(str(text_processing(st.subj)), "")
                )
                o_def = (
                    st.obj_def
                    or merged_defs.get(st.obj, "")
                    or merged_defs.get(str(text_processing(st.obj)), "")
                )
                if s_def != st.subj_def or o_def != st.obj_def:
                    structured[idx] = StructuredTriple(
                        subj=st.subj,
                        subj_def=s_def,
                        pred=st.pred,
                        obj=st.obj,
                        obj_def=o_def,
                        spo_def=st.spo_def,
                    )

        return tuple(structured)

    def extract_triples(
        self,
        passage: str,
        named_entities: Sequence[Any],
        *,
        lm: Any = None,
        use_gbnf: bool | None = None,
    ) -> tuple[tuple[str, str, str], ...]:
        """Run DSPy triple extraction on Qwen w/ GBNF."""
        structured = self.extract_structured_triples(
            passage,
            named_entities,
            lm=lm,
            use_gbnf=use_gbnf,
        )
        return tuple(st.as_tuple() for st in structured)

    def extract_openie(
        self,
        chunk_id: str,
        passage: str,
        *,
        lm: Any = None,
        use_gbnf: bool | None = None,
    ) -> OpenIERecord:
        """Execute OpenIE (NER w/ definitions -> Structured Triples -> Missing Entity Definitions) for a passage chunk."""
        entities, entity_definitions = self._run_ner_stage(passage, lm=lm, use_gbnf=use_gbnf)
        has_nonempty_defs = any(defn for _, defn in entity_definitions)
        stage2_entities_input: list[Any] = (
            [[name, defn] for name, defn in entity_definitions]
            if has_nonempty_defs
            else list(entities)
        )
        structured_triples = self.extract_structured_triples(
            passage,
            stage2_entities_input,
            entity_definitions=entity_definitions,
            lm=lm,
            use_gbnf=use_gbnf,
        )
        merged_entities: list[str] = list(entities)
        seen_entity_names: set[str] = set(merged_entities)
        seen_entity_norms: set[str] = {
            str(text_processing(name)) for name in merged_entities if str(text_processing(name))
        }
        merged_def_list: list[tuple[str, str]] = []
        seen_def_keys: dict[str, list[int]] = {}
        for name, defn in entity_definitions:
            norm = str(text_processing(name)) or name.lower()
            d_clean = defn.strip()
            idxs = seen_def_keys.setdefault(norm, [])
            if not idxs:
                idxs.append(len(merged_def_list))
                merged_def_list.append((name, d_clean))
            elif d_clean:
                matched = False
                for idx in idxs:
                    prev_n, prev_d = merged_def_list[idx]
                    if not prev_d:
                        merged_def_list[idx] = (prev_n, d_clean)
                        matched = True
                        break
                    if prev_d.lower() == d_clean.lower():
                        if len(d_clean) > len(prev_d):
                            merged_def_list[idx] = (prev_n, d_clean)
                        matched = True
                        break
                if not matched:
                    idxs.append(len(merged_def_list))
                    merged_def_list.append((name, d_clean))

        for st in structured_triples:
            for ent_name, ent_def in ((st.subj, st.subj_def), (st.obj, st.obj_def)):
                if not ent_name:
                    continue
                ent_norm = str(text_processing(ent_name))
                if ent_name not in seen_entity_names and (
                    not ent_norm or ent_norm not in seen_entity_norms
                ):
                    merged_entities.append(ent_name)
                    seen_entity_names.add(ent_name)
                    if ent_norm:
                        seen_entity_norms.add(ent_norm)
                norm_key = ent_norm or ent_name.lower()
                d_clean = ent_def.strip()
                idxs = seen_def_keys.setdefault(norm_key, [])
                if not idxs:
                    idxs.append(len(merged_def_list))
                    merged_def_list.append((ent_name, d_clean))
                elif d_clean:
                    matched = False
                    for idx in idxs:
                        prev_n, prev_d = merged_def_list[idx]
                        if not prev_d:
                            merged_def_list[idx] = (prev_n, d_clean)
                            matched = True
                            break
                        if prev_d.lower() == d_clean.lower():
                            if len(d_clean) > len(prev_d):
                                merged_def_list[idx] = (prev_n, d_clean)
                            matched = True
                            break
                    if not matched:
                        idxs.append(len(merged_def_list))
                        merged_def_list.append((ent_name, d_clean))

        triples = tuple(st.as_tuple() for st in structured_triples)
        return OpenIERecord(
            chunk_id=chunk_id,
            entities=tuple(merged_entities),
            triples=triples,
            structured_triples=structured_triples,
            entity_definitions=tuple(merged_def_list),
        )

    def judge_synonyms(
        self,
        candidate_pairs: Sequence[Mapping[str, Any]] | str,
        *,
        lm: Any = None,
        use_gbnf: bool | None = None,
    ) -> list[dict[str, str]]:
        """Classify candidate entity pairs into SKOS/OWL relations (`skos:altLabel`, `skos:hiddenLabel`, `skos:related`, `owl:differentFrom`)."""
        if not candidate_pairs:
            return []
        if isinstance(candidate_pairs, str):
            pairs_json = candidate_pairs.strip()
            if not pairs_json or pairs_json == "[]":
                return []
            if len(pairs_json) > self._context_char_limit(12000):
                try:
                    parsed_pairs = json.loads(pairs_json)
                    if isinstance(parsed_pairs, list):
                        pairs_json = json.dumps(parsed_pairs[:16], ensure_ascii=False)
                except json.JSONDecodeError, TypeError:
                    pairs_json = pairs_json[: self._context_char_limit(12000)]
        else:
            pairs_list = [dict(p) for p in candidate_pairs if isinstance(p, Mapping)]
            if not pairs_list:
                return []
            if len(pairs_list) > 16:
                pairs_list = pairs_list[:16]
            pairs_json = json.dumps(pairs_list, ensure_ascii=False)

        prev_telem_len = len(self.telemetry)
        pred = self._invoke_stage(
            self.synonym_judge,
            stage_name="synonym_judge",
            is_rlm=False,
            lm=lm,
            use_gbnf=use_gbnf,
            grammar_fn=build_dspy_synonym_judge_grammar,
            candidate_pairs=pairs_json,
        )
        raw_decisions = getattr(pred, "decisions", [])
        decisions = parse_synonym_decisions(raw_decisions)
        stage_tokens = (
            int(self.telemetry[-1].get("tokens_used", 0))
            if len(self.telemetry) > prev_telem_len
            else 0
        )
        self.synonym_audit_trail.append(
            {
                "mode": "direct",
                "predictor": "synonym_judge",
                "signature": JudgeSynonyms.__name__,
                "tokens_used": stage_tokens,
                "candidate_pairs": pairs_json,
                "decisions": decisions,
            }
        )
        return decisions

    def propose_shacl_repairs(
        self,
        violations: Sequence[Any] | str,
        *,
        focus_context: str = "",
        mode: Literal["direct", "rlm"] | None = None,
        lm: Any = None,
        use_gbnf: bool | None = None,
    ) -> list[dict[str, str]]:
        """Propose bounded closed-world SHACL repair actions for the given violations.

        By default (`mode=None`), executes the single-level RLM Python REPL healer
        (`shacl_healer_rlm` / `HealSHACLViolations`), which inspects focus nodes,
        hydrates source chunks, and delegates structured repair synthesis to its
        inner `qwen_propose_shacl_repairs` tool (`mode="direct"` -> `shacl_repair` /
        `ProposeSHACLRepairs` with pure-JSON GBNF).
        """
        if not violations:
            return []
        is_rlm = True if mode is None else (mode == "rlm")
        if isinstance(violations, str):
            violations_json = violations.strip()
            if not violations_json or violations_json == "[]":
                return []
            if not is_rlm and len(violations_json) > self._context_char_limit(12000):
                try:
                    parsed_v = json.loads(violations_json)
                    if isinstance(parsed_v, list):
                        violations_json = json.dumps(parsed_v[:16], ensure_ascii=False)
                except json.JSONDecodeError, TypeError:
                    violations_json = violations_json[: self._context_char_limit(12000)]
        else:
            serialized: list[dict[str, str]] = []
            for v in violations:
                if isinstance(v, Mapping):
                    serialized.append(
                        {
                            "focus_node": str(v.get("focus_node", "")),
                            "source_shape": str(v.get("source_shape", v.get("shape", ""))),
                            "result_path": str(v.get("result_path", "")),
                            "value": str(v.get("value", "")),
                            "message": str(v.get("message", "")),
                        }
                    )
                elif hasattr(v, "focus_node"):
                    serialized.append(
                        {
                            "focus_node": str(getattr(v, "focus_node", "")),
                            "source_shape": str(
                                getattr(v, "source_shape", getattr(v, "shape", ""))
                            ),
                            "result_path": str(getattr(v, "result_path", "")),
                            "value": str(getattr(v, "value", "")),
                            "message": str(getattr(v, "message", "")),
                        }
                    )
            if not serialized:
                return []
            if not is_rlm and len(serialized) > 16:
                serialized = serialized[:16]
            violations_json = json.dumps(serialized, ensure_ascii=False)

        stage_inputs: dict[str, Any] = {"violations": violations_json}
        if not is_rlm:
            stage_inputs["focus_context"] = (
                focus_context[:12000] if len(focus_context) > 12000 else focus_context
            )
        prev_telem_len = len(self.telemetry)
        pred = self._invoke_stage(
            self.shacl_repair,
            self.shacl_healer_rlm,
            stage_name="shacl_repair",
            is_rlm=is_rlm,
            lm=lm,
            use_gbnf=use_gbnf,
            grammar_fn=build_dspy_shacl_repair_grammar,
            **stage_inputs,
        )
        raw_repairs = getattr(pred, "repairs", [])
        repairs = parse_shacl_repairs(raw_repairs)
        stage_tokens = (
            int(self.telemetry[-1].get("tokens_used", 0))
            if len(self.telemetry) > prev_telem_len
            else 0
        )
        self.shacl_audit_trail.append(
            {
                "mode": "rlm" if is_rlm else "direct",
                "predictor": "shacl_healer_rlm" if is_rlm else "shacl_repair",
                "signature": (
                    HealSHACLViolations.__name__ if is_rlm else ProposeSHACLRepairs.__name__
                ),
                "tokens_used": stage_tokens,
                "violations": violations_json,
                "focus_context": focus_context,
                "repairs": repairs,
            }
        )
        return repairs

    repair_shacl_violations = propose_shacl_repairs
    heal_shacl_violations = propose_shacl_repairs

    def filter_triples(
        self,
        question: str,
        candidate_facts: Sequence[tuple[str, str, str] | list[str]],
        *,
        max_facts: int | None = None,
        lm: Any = None,
        use_gbnf: bool | None = None,
    ) -> list[int]:
        """Filter candidate triples on Qwen w/ dynamic candidate-literal GBNF."""
        if not candidate_facts:
            return []
        effective_max_facts = self.gbnf_max_facts if max_facts is None else max_facts
        fact_before_filter = json.dumps({"fact": [list(f) for f in candidate_facts]})
        pred = self._invoke_stage(
            self.triple_filter,
            stage_name="triple_filter",
            is_rlm=False,
            lm=lm,
            use_gbnf=use_gbnf,
            grammar_fn=lambda: build_dspy_rerank_grammar(
                candidate_facts, max_facts=effective_max_facts
            ),
            question=question,
            fact_before_filter=fact_before_filter,
        )
        raw_after = getattr(pred, "fact", getattr(pred, "fact_after_filter", None))
        generated = coerce_generated_facts(raw_after)
        return match_facts_to_candidate_indices(
            candidate_facts, generated, max_facts=effective_max_facts
        )

    def extract_query_intent(
        self,
        question: str,
        *,
        lm: Any = None,
        use_gbnf: bool | None = None,
    ) -> str:
        """Rewrite question into a declarative query_intent string (GBNF on Qwen GPU, temp=0.4, freq_penalty=0.2)."""
        from dspy.utils.usage_tracker import track_usage

        effective_lm = lm or self.rlm_sub_lm or self.orchestrator_lm
        call_cfg: dict[str, Any] = {
            "temperature": 0.4,
            "frequency_penalty": 0.2,
        }
        if self._should_use_gbnf(use_gbnf, effective_lm):
            call_cfg["extra_body"] = {"grammar": build_dspy_query_intent_grammar()}
        with track_usage() as usage_tracker, self._dspy_ctx(effective_lm, use_json_adapter=True):
            pred = self.query_intent(question=question, config=call_cfg)
        tokens = usage_tracker.get_total_tokens()
        out_intent = str(getattr(pred, "query_intent", "") or "").strip() or question
        self._record_stage_telemetry(
            "query_intent",
            "query_intent",
            signature_name=ExtractQueryIntent.__name__,
            is_rlm=False,
            usage=tokens,
            inputs_summary=question[:120],
            outputs_summary=out_intent[:120],
        )
        return out_intent

    @diagnostic_boundary
    def answer_question(
        self,
        context: str,
        question: str,
        *,
        lm: Any = None,
        use_gbnf: bool | None = None,
    ) -> dspy.Prediction:
        """Answer a question over retrieved context using `qa` (`AnswerQuestion`)."""
        return self._invoke_stage(
            self.qa,
            stage_name="qa",
            is_rlm=False,
            lm=lm,
            use_gbnf=use_gbnf,
            grammar_fn=lambda: build_dspy_qa_grammar(self.gbnf_max_reasoning_chars),
            context=context,
            question=question,
        )

    def set_retrieval_hook(
        self,
        hook: Callable[
            ...,
            tuple[str, tuple[str, ...], tuple[float, ...]]
            | tuple[str, tuple[str, ...], tuple[float, ...], tuple[str, ...]],
        ]
        | None,
    ) -> None:
        """Register an optional end-to-end retrieval hook executed inside forward(question=...)."""
        self._retrieval_hook = hook
        self.rm.hook = hook
        self.retriever.hook = hook

    def _run_react_qa_fallback(
        self,
        question: str,
        initial_context: str,
    ) -> dspy.Prediction | None:
        """Run bounded `dspy.ReActV2` multi-hop graph exploration when single-shot PPR/QA abstains or retrieval confidence < 0.15."""
        react_cls = getattr(dspy, "ReActV2", None)
        if react_cls is None:
            return None

        def search_graph_facts(query: str) -> str:
            """Search the knowledge graph or retrieval engine for facts matching query."""
            if self._graph_search_tool is not None:
                return str(self._graph_search_tool(str(query)))
            if self._retrieval_hook is not None:
                ctx, _, _ = self.rm._invoke_hook(str(query))
                return str(ctx)
            return ""

        def inspect_entity_neighborhood(entity_id: str) -> str:
            """Inspect the 1-hop entity neighborhood and definitions for entity_id."""
            if self._node_inspector is not None:
                return json.dumps(self._node_inspector(str(entity_id)), ensure_ascii=False)
            return "{}"

        def read_passage_chunk(chunk_id: str) -> str:
            """Hydrate the original passage chunk text for chunk_id."""
            if self._chunk_hydrator is not None:
                return str(self._chunk_hydrator(str(chunk_id)))
            return ""

        sig = self.qa.predict.signature
        eff_lm = self.orchestrator_lm or self.rlm_sub_lm
        try:
            agent = react_cls(
                sig,
                tools=[
                    search_graph_facts,
                    inspect_entity_neighborhood,
                    read_passage_chunk,
                ],
                max_iters=10,
            )
            with self._dspy_ctx(eff_lm, use_json_adapter=True):
                return agent(context=initial_context, question=question)
        except Exception as exc:
            logging.getLogger(__name__).debug(
                "ReActV2 fallback skipped (%s: %s)", type(exc).__name__, exc
            )
            return None

    @diagnostic_boundary
    def forward(
        self,
        context: str | None = None,
        question: str | None = None,
        *,
        item_docs: Sequence[str] | None = None,
        use_gbnf: bool | None = None,
    ) -> dspy.Prediction:
        """Default DSPy forward pass executes multi-hop QA or the full extraction+retrieval+QA trace."""
        from dspy.utils.usage_tracker import track_usage

        if context is not None and question is not None:
            if self.pareto_ensemble is not None and not self._in_ensemble:
                return self.pareto_ensemble.answer_question(
                    context=context, question=question, use_gbnf=use_gbnf
                )
            return self.answer_question(context=context, question=question, use_gbnf=use_gbnf)
        if context is not None and question is None:
            return self.forward(
                context=None,
                question=context,
                item_docs=item_docs,
                use_gbnf=use_gbnf,
            )
        if question is not None:
            if self._retrieval_hook is not None or getattr(dspy.settings, "rm", None) is not None:
                with track_usage() as usage_tracker:
                    ret_pred = self.retriever(question, item_docs=item_docs)
                    hydrated_ctx = str(getattr(ret_pred, "context", "") or "")
                    raw_psg = getattr(ret_pred, "all_passages", getattr(ret_pred, "passages", None))
                    passages = tuple(raw_psg) if raw_psg is not None else ()
                    raw_sc = getattr(ret_pred, "all_scores", getattr(ret_pred, "scores", None))
                    scores = tuple(float(s) for s in raw_sc) if raw_sc is not None else ()
                    raw_pd = getattr(ret_pred, "all_pids", getattr(ret_pred, "pids", None))
                    pids = tuple(str(p) for p in raw_pd) if raw_pd is not None else ()
                    if self.pareto_ensemble is not None and not self._in_ensemble:
                        pred = self.pareto_ensemble.answer_question(
                            context=hydrated_ctx, question=question, use_gbnf=use_gbnf
                        )
                    else:
                        pred = self.answer_question(
                            context=hydrated_ctx, question=question, use_gbnf=use_gbnf
                        )
                    top_score = max(scores) if len(scores) > 0 else 1.0
                    if self.enable_react_fallback and (
                        is_abstaining_answer(str(getattr(pred, "answer", "") or ""))
                        or top_score < 0.15
                    ):
                        react_pred = self._run_react_qa_fallback(question, hydrated_ctx)
                        if react_pred is not None and not is_abstaining_answer(
                            str(getattr(react_pred, "answer", "") or "")
                        ):
                            pred = react_pred
                tokens = usage_tracker.get_total_tokens()
                res = dspy.Prediction(
                    context=hydrated_ctx,
                    answer=getattr(pred, "answer", ""),
                    reasoning=getattr(pred, "reasoning", ""),
                    retrieved_passages=passages,
                    retrieved_scores=scores,
                    retrieved_chunk_ids=pids,
                )
                if tokens and hasattr(res, "set_lm_usage"):
                    res.set_lm_usage(tokens)
                return res
            raise ValueError(
                "context must be provided when no retrieval_hook is registered on HippoRAGProgram."
            )
        raise ValueError(
            "Either (context, question) or question must be provided to HippoRAGProgram."
        )

    def as_async(self) -> Any:
        """Return a `dspy.asyncify`-wrapped async callable for this `HippoRAGProgram`."""
        return dspy.asyncify(self)

    def named_predictors(self) -> list[tuple[str, Any]]:
        """Return all 9 canonical stage predictors so `dspy.GEPA` inspects the full program surface."""
        return list(super().named_predictors())


class ParetoProgramEnsemble(dspy.Module):
    """Ensembles the top-M Pareto-optimal `HippoRAGProgram` candidates using `dspy.Ensemble` and `dspy.asyncify`.

    Provides closed-world entity/triple unioning (`reduce_ner_predictions`, `reduce_triple_predictions`)
    and multi-hop QA reasoning reconciliation (`reduce_qa_predictions`).
    """

    def __init__(self, programs: Sequence[HippoRAGProgram]) -> None:
        super().__init__()
        if not programs:
            raise ValueError("ParetoProgramEnsemble requires at least 1 HippoRAGProgram candidate.")
        self.programs: list[HippoRAGProgram] = list(programs)
        self.qa_ensemble = dspy.Ensemble(reduce_fn=self.reduce_qa_predictions).compile(
            self.programs
        )
        # Freeze as compiled so attaching onto HippoRAGProgram.pareto_ensemble never duplicates named_predictors()
        self._compiled = True

    @classmethod
    def from_frontier_states(
        cls,
        template_program: HippoRAGProgram,
        frontier_states: Sequence[Mapping[str, Any]],
    ) -> ParetoProgramEnsemble:
        import copy

        candidates: list[HippoRAGProgram] = []
        for state in frontier_states:
            cloned = copy.deepcopy(template_program)
            cloned.pareto_frontier_states = []
            cloned.pareto_ensemble = None
            state_without_frontier = dict(state)
            if isinstance(state_without_frontier.get("yaw_surface"), Mapping):
                ys = dict(state_without_frontier["yaw_surface"])
                ys.pop("pareto_frontier_states", None)
                state_without_frontier["yaw_surface"] = ys
            cloned.load_state(state_without_frontier)
            candidates.append(cloned)
        return cls(candidates or [template_program])

    @staticmethod
    def reduce_ner_predictions(predictions: Sequence[Any]) -> dspy.Prediction:
        """Union extracted entities across Pareto candidates, preserving the richest non-tautological `skos:definition`."""
        ordered_norms: list[str] = []
        canonical_name_by_norm: dict[str, str] = {}
        best_def_by_norm: dict[str, str] = {}

        for pred in predictions:
            raw_entities = getattr(pred, "named_entities", pred)
            for ename, edef in extract_entity_definitions(raw_entities):
                clean_name = str(ename).strip()
                clean_def = str(edef or "").strip()
                if not clean_name:
                    continue
                norm = str(text_processing(clean_name)) or clean_name.lower()
                if norm not in canonical_name_by_norm:
                    ordered_norms.append(norm)
                    canonical_name_by_norm[norm] = clean_name
                is_taut = clean_def.lower() == clean_name.lower()
                existing_def = best_def_by_norm.get(norm, "")
                if clean_def and not is_taut and len(clean_def) > len(existing_def):
                    best_def_by_norm[norm] = clean_def

        merged_pairs = [
            [canonical_name_by_norm[n], best_def_by_norm.get(n, "")] for n in ordered_norms
        ]
        return dspy.Prediction(named_entities=merged_pairs)

    @staticmethod
    def reduce_triple_predictions(
        predictions: Sequence[Any],
        *,
        valid_entities: Sequence[tuple[str, str]] | Mapping[str, str] | None = None,
    ) -> dspy.Prediction:
        """Union and rank relation triples by vote count across Pareto candidates while enforcing closed-world entity linkage."""
        def_map: dict[str, str] = {}
        if isinstance(valid_entities, Mapping):
            for k, v in valid_entities.items():
                if k and v:
                    def_map[str(k).strip()] = str(v).strip()
                    def_map[str(text_processing(str(k)))] = str(v).strip()
        elif isinstance(valid_entities, Sequence):
            for pair in valid_entities:
                if isinstance(pair, (list, tuple)) and len(pair) >= 2 and pair[0] and pair[1]:
                    def_map[str(pair[0]).strip()] = str(pair[1]).strip()
                    def_map[str(text_processing(str(pair[0])))] = str(pair[1]).strip()

        votes: dict[tuple[str, str, str], int] = {}
        first_seen: dict[tuple[str, str, str], int] = {}
        best_triple_by_key: dict[tuple[str, str, str], StructuredTriple] = {}
        seq_idx = 0

        for pred in predictions:
            raw_triples = getattr(pred, "triples", pred)
            structured_list = (
                list(raw_triples)
                if isinstance(raw_triples, (list, tuple))
                and all(isinstance(x, StructuredTriple) for x in raw_triples)
                else list(extract_structured_triples(raw_triples, entity_definitions=def_map))
            )
            seen_in_cand: set[tuple[str, str, str]] = set()
            for st in structured_list:
                if not (st.subj and st.pred and st.obj):
                    continue
                s_norm = str(text_processing(st.subj)) or st.subj.lower()
                p_norm = str(text_processing(st.pred)) or st.pred.lower()
                o_norm = str(text_processing(st.obj)) or st.obj.lower()
                if s_norm == o_norm:
                    continue
                key = (s_norm, p_norm, o_norm)
                if key not in seen_in_cand:
                    seen_in_cand.add(key)
                    votes[key] = votes.get(key, 0) + 1
                if key not in first_seen:
                    first_seen[key] = seq_idx
                    seq_idx += 1
                prev = best_triple_by_key.get(key)
                s_def = (
                    st.subj_def
                    or (prev.subj_def if prev else "")
                    or def_map.get(st.subj, "")
                    or def_map.get(s_norm, "")
                )
                o_def = (
                    st.obj_def
                    or (prev.obj_def if prev else "")
                    or def_map.get(st.obj, "")
                    or def_map.get(o_norm, "")
                )
                spo_def = (
                    st.spo_def if not prev or len(st.spo_def) >= len(prev.spo_def) else prev.spo_def
                )
                best_triple_by_key[key] = StructuredTriple(
                    subj=prev.subj if prev else st.subj,
                    subj_def=s_def,
                    pred=prev.pred if prev else st.pred,
                    obj=prev.obj if prev else st.obj,
                    obj_def=o_def,
                    spo_def=spo_def,
                )

        ranked_keys = sorted(
            votes.keys(),
            key=lambda k: (-votes[k], first_seen[k]),
        )
        ranked_structured = [best_triple_by_key[k] for k in ranked_keys]
        return dspy.Prediction(
            structured_triples=tuple(ranked_structured),
            triples=[list(st.as_tuple()) for st in ranked_structured],
        )

    @staticmethod
    def reduce_qa_predictions(predictions: Sequence[Any]) -> dspy.Prediction:
        """Select the consensus non-abstaining multi-hop QA answer across Pareto candidates."""
        if not predictions:
            return dspy.Prediction(reasoning="", answer="")

        non_abstaining = [
            p for p in predictions if not is_abstaining_answer(str(getattr(p, "answer", "") or ""))
        ]
        pool = non_abstaining if non_abstaining else list(predictions)
        votes: dict[str, int] = {}
        best_pred_by_norm: dict[str, Any] = {}
        first_idx: dict[str, int] = {}

        for idx, p in enumerate(pool):
            ans = str(getattr(p, "answer", "") or "").strip()
            reas = str(getattr(p, "reasoning", "") or "").strip()
            norm = str(text_processing(ans)) or ans.lower()
            votes[norm] = votes.get(norm, 0) + 1
            if norm not in first_idx:
                first_idx[norm] = idx
            prev = best_pred_by_norm.get(norm)
            prev_reas = (
                str(getattr(prev, "reasoning", "") or "").strip() if prev is not None else ""
            )
            if prev is None or len(reas) > len(prev_reas):
                best_pred_by_norm[norm] = p

        winner_norm = min(
            votes.keys(),
            key=lambda n: (-votes[n], first_idx[n]),
        )
        winner = best_pred_by_norm[winner_norm]
        return dspy.Prediction(
            reasoning=str(getattr(winner, "reasoning", "") or ""),
            answer=str(getattr(winner, "answer", "") or ""),
        )

    def extract_openie(
        self,
        chunk_id: str,
        passage: str,
        *,
        lm: Any = None,
        use_gbnf: bool | None = None,
    ) -> OpenIERecord:
        """Execute Pareto-ensembled OpenIE extraction across all candidate programs."""
        ner_preds: list[dspy.Prediction] = []
        for prog in self.programs:
            prev_flag = prog._in_ensemble
            prog._in_ensemble = True
            try:
                _, defs = prog._run_ner_stage(passage, lm=lm, use_gbnf=use_gbnf)
                ner_preds.append(dspy.Prediction(named_entities=[[k, v] for k, v in defs]))
            finally:
                prog._in_ensemble = prev_flag

        merged_ner = self.reduce_ner_predictions(ner_preds)
        merged_defs = list(extract_entity_definitions(merged_ner.named_entities))
        stage2_input: list[Any] = (
            [[k, v] for k, v in merged_defs]
            if any(v for _, v in merged_defs)
            else [k for k, _ in merged_defs]
        )

        triple_preds: list[dspy.Prediction] = []
        for prog in self.programs:
            prev_flag = prog._in_ensemble
            prog._in_ensemble = True
            try:
                st_list = prog.extract_structured_triples(
                    passage,
                    stage2_input,
                    entity_definitions=merged_defs,
                    lm=lm,
                    use_gbnf=use_gbnf,
                )
                triple_preds.append(dspy.Prediction(triples=list(st_list)))
            finally:
                prog._in_ensemble = prev_flag

        merged_triples_pred = self.reduce_triple_predictions(
            triple_preds,
            valid_entities=merged_defs,
        )
        structured_triples: tuple[StructuredTriple, ...] = tuple(
            merged_triples_pred.structured_triples
        )

        ordered_entities: list[str] = [k for k, _ in merged_defs]
        seen_norms: set[str] = {str(text_processing(k)) or k.lower() for k in ordered_entities}
        def_by_norm: dict[str, str] = {
            (str(text_processing(k)) or k.lower()): v for k, v in merged_defs if v
        }
        for st in structured_triples:
            for ename, edef in ((st.subj, st.subj_def), (st.obj, st.obj_def)):
                if not ename:
                    continue
                enorm = str(text_processing(ename)) or ename.lower()
                if enorm not in seen_norms:
                    seen_norms.add(enorm)
                    ordered_entities.append(ename)
                if edef and not def_by_norm.get(enorm):
                    def_by_norm[enorm] = edef

        final_defs = tuple(
            (ename, def_by_norm.get(str(text_processing(ename)) or ename.lower(), ""))
            for ename in ordered_entities
        )
        return OpenIERecord(
            chunk_id=chunk_id,
            entities=tuple(ordered_entities),
            triples=tuple(st.as_tuple() for st in structured_triples),
            structured_triples=structured_triples,
            entity_definitions=final_defs,
        )

    def answer_question(
        self,
        context: str,
        question: str,
        *,
        lm: Any = None,
        use_gbnf: bool | None = None,
    ) -> dspy.Prediction:
        """Execute Pareto-ensembled multi-hop QA across all candidate programs."""
        preds: list[dspy.Prediction] = []
        for prog in self.programs:
            prev_flag = prog._in_ensemble
            prog._in_ensemble = True
            try:
                preds.append(
                    prog.answer_question(
                        context=context,
                        question=question,
                        lm=lm,
                        use_gbnf=use_gbnf,
                    )
                )
            finally:
                prog._in_ensemble = prev_flag
        return self.reduce_qa_predictions(preds)

    async def aanswer_question(
        self,
        context: str,
        question: str,
        *,
        lm: Any = None,
        use_gbnf: bool | None = None,
    ) -> dspy.Prediction:
        """Async Pareto-ensembled multi-hop QA using `dspy.asyncify` across all candidate programs."""
        import anyio

        results: list[Any] = [None] * len(self.programs)

        async def _run_idx(idx: int, prog: HippoRAGProgram) -> None:
            prev_flag = prog._in_ensemble
            prog._in_ensemble = True
            try:
                async_prog: Any = prog.as_async()
                if lm is not None:
                    with dspy.context(lm=lm):
                        results[idx] = await async_prog(
                            context=context,
                            question=question,
                            use_gbnf=use_gbnf,
                        )
                else:
                    results[idx] = await async_prog(
                        context=context,
                        question=question,
                        use_gbnf=use_gbnf,
                    )
            finally:
                prog._in_ensemble = prev_flag

        async with anyio.create_task_group() as tg:
            for idx, prog in enumerate(self.programs):
                tg.start_soon(_run_idx, idx, prog)

        return self.reduce_qa_predictions([r for r in results if r is not None])

    def forward(
        self,
        context: str | None = None,
        question: str | None = None,
        **kwargs: Any,
    ) -> dspy.Prediction:
        if context is not None and question is not None:
            return self.answer_question(
                context=context,
                question=question,
                use_gbnf=kwargs.get("use_gbnf"),
            )
        return self.qa_ensemble(context=context, question=question, **kwargs)


_STAGE_SCHEMA_CONTRACTS: dict[str, tuple[str, tuple[str, ...], str]] = {
    "ner": (
        "named_entities",
        ("named_entities", "named entities"),
        'Respond with JSON: {"named_entities": [["entity", "concise contextual definition"], ...]}.',
    ),
    "triples": (
        "triples",
        ("triples",),
        'Respond with JSON: {"triples": [{"sub": "...", "pred": "...", "obj": "...", "def": "..."}, ...]}.',
    ),
    "entity_definer": (
        "entity_definitions",
        ("entity_definitions", "entity definitions"),
        'Respond with JSON: {"entity_definitions": [["entity_name", "concise contextual definition"], ...]}.',
    ),
    "synonym_judge": (
        "decisions",
        ("decisions",),
        'Respond with JSON: {"decisions": [{"pair_id": "id_a|id_b", "relation": "skos:altLabel|skos:hiddenLabel|skos:related|owl:differentFrom", "canonical": "id_a_or_id_b"}, ...]}.',
    ),
    "shacl_repair": (
        "repairs",
        ("repairs",),
        'Respond with JSON: {"repairs": [{"focus_node": "...", "action": "...", "value": "..."}, ...]}.',
    ),
    "triple_filter": (
        '"fact"',
        ('"fact"', "'fact'"),
        'Output in JSON: {"fact": [["s1", "p1", "o1"], ...]}.',
    ),
    "fact_filter": (
        '"fact"',
        ('"fact"', "'fact'"),
        'Output in JSON: {"fact": [["s1", "p1", "o1"], ...]}.',
    ),
}


class _InstructionSynthesizerWorker(dspy.Module):
    """Inner GBNF instruction-synthesis worker module of `RLMInstructionProposer`, optimized by `dspy.SIMBA`."""

    def __init__(
        self,
        *,
        qwen_proposer: Any,
        qwen_compressor: Any,
        worker_lm: Any = None,
        orchestrator_lm: Any = None,
        enable_gbnf: bool = True,
        max_instruction_chars: int | None = 1500,
        max_field_chars: int = 600,
        worker_temperature: float = 0.7,
        json_adapter: dspy.Adapter | None = None,
    ) -> None:
        super().__init__()
        self.qwen_proposer = qwen_proposer
        self.qwen_compressor = qwen_compressor
        self.worker_lm = worker_lm
        self.orchestrator_lm = orchestrator_lm
        self.enable_gbnf = bool(enable_gbnf)
        self.max_instruction_chars = max_instruction_chars
        self.max_field_chars = max_field_chars
        self.worker_temperature = float(worker_temperature)
        self.json_adapter: dspy.Adapter = json_adapter or YawJSONAdapter()
        self._rollout_lm: Any = None
        self.sub_telemetry: list[dict[str, Any]] = []

    def get_lm(self) -> Any:
        return (
            self._rollout_lm
            or self.worker_lm
            or self.orchestrator_lm
            or getattr(self.qwen_proposer, "lm", None)
        )

    def set_lm(self, lm: Any) -> None:
        self._rollout_lm = lm

    def __deepcopy__(self, memo: dict[int, Any]) -> _InstructionSynthesizerWorker:
        import copy

        cls = self.__class__
        new_mod = cls.__new__(cls)
        memo[id(self)] = new_mod
        for k, v in self.__dict__.items():
            if k in ("worker_lm", "orchestrator_lm", "json_adapter", "_rollout_lm"):
                setattr(new_mod, k, v)
            else:
                setattr(new_mod, k, copy.deepcopy(v, memo))
        return new_mod

    def forward(
        self,
        current_instruction: str = "",
        examples_with_feedback: Any = None,
        component_name: str = "",
    ) -> dspy.Prediction:
        from dspy.teleprompt.gepa.instruction_proposal import InstructionProposer

        native = InstructionProposer(
            base_instructions=ProposeInstructionSignature.__doc__,
            max_chars=self.max_instruction_chars,
            adapter=self.json_adapter,
        )
        native.propose = self.qwen_proposer
        native.compress = self.qwen_compressor if self.max_instruction_chars else None
        examples = self.fit_evidence(current_instruction, examples_with_feedback, component_name)
        stage = component_name.removesuffix(".predict")
        contract = _STAGE_SCHEMA_CONTRACTS.get(
            stage, ("", (), "Preserve the existing output schema.")
        )[2]
        inputs = {
            "examples_with_feedback": [
                {"component": component_name, "contract": contract},
                *examples,
            ]
        }
        lm = self._rollout_lm or self.worker_lm or self.orchestrator_lm or dspy.settings.lm
        with dspy.context(lm=lm, adapter=self.json_adapter):
            result = native(
                {component_name: current_instruction},
                {component_name: inputs["examples_with_feedback"]},
                [component_name],
            )[component_name]
        return dspy.Prediction(new_instruction=result)

    def fit_evidence(
        self, current_instruction: str, examples_with_feedback: Any, component_name: str
    ) -> list[dict[str, Any]]:
        examples = list(examples_with_feedback or [])
        stage = component_name.removesuffix(".predict")
        contract = _STAGE_SCHEMA_CONTRACTS.get(
            stage, ("", (), "Preserve the existing output schema.")
        )[2]
        mandatory = {"component": component_name, "contract": contract}
        lm = self._rollout_lm or self.worker_lm or self.orchestrator_lm or dspy.settings.lm
        inputs = {
            "current_instruction": current_instruction,
            "examples_with_feedback": [mandatory, *examples],
        }
        if self.max_instruction_chars is not None:
            inputs["length_limit"] = (
                f"The new instruction must be at most {self.max_instruction_chars} characters."
            )
        while isinstance(lm, ReflectionLM):
            try:
                lm.count(
                    self.json_adapter.format(
                        self.qwen_proposer.signature, self.qwen_proposer.demos, inputs
                    )
                )
                break
            except ReflectionBudgetError:
                if not examples:
                    raise
                examples.pop()
                inputs["examples_with_feedback"] = [mandatory, *examples]
        return examples


class RLMInstructionProposer:
    """RLM evidence selection followed by a SIMBA-optimized native DSPy proposer.

    Conforms to `gepa.core.adapter.ProposalFn` and stock DSPy `InstructionProposer`:
    `(candidate, reflective_dataset, components_to_update) -> dict[str, str]`, while its
    inner worker (`self.worker_module`: `_InstructionSynthesizerWorker(dspy.Module)` wrapping
    `qwen_proposer` + `qwen_compressor`) is continuously refined by an always-on `dspy.SIMBA`
    meta-optimizer during prompt evolution.
    """

    def __init__(
        self,
        *,
        orchestrator_lm: Any = None,
        worker_lm: Any = None,
        enable_gbnf: bool = True,
        enable_simba: bool = True,
        simba_bsize: int = 2,
        simba_num_candidates: int = 2,
        simba_max_steps: int = 1,
        simba_max_demos: int = 2,
        max_instruction_chars: int | None = 1500,
        max_field_chars: int = 600,
        max_feedback_examples: int = 5,
        reflection_temperature: float = 0.8,
        worker_temperature: float = 0.7,
        rlm_max_iters: int = 10,
        rlm_max_llm_calls: int = 20,
        rlm_max_output_chars: int = 10000,
        include_demos: bool = True,
        adapter: dspy.Adapter | None = None,
        rlm_interpreter_factory: Callable[[], Any] = dspy.PythonInterpreter,
        task_scorer: Callable[[Mapping[str, str]], float] | None = None,
        save_simba_state: Callable[[dict[str, Any]], None] | None = None,
        reflection_input_tokens: int = 16384,
        token_counter: Any = None,
        checkpoint_get: Callable[[str], Any] | None = None,
        checkpoint_put: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        if max_instruction_chars is not None and max_instruction_chars <= 0:
            raise ValueError(
                f"max_instruction_chars must be > 0 or None, got {max_instruction_chars!r}."
            )

        if reflection_input_tokens < 1:
            raise ValueError("reflection_input_tokens must be positive")
        self.reflection_input_tokens = reflection_input_tokens
        self.token_counter = token_counter
        self.checkpoint_get = checkpoint_get
        self.checkpoint_put = checkpoint_put
        self.orchestrator_lm = orchestrator_lm
        self.worker_lm = worker_lm
        self.enable_gbnf = bool(enable_gbnf)
        self.enable_simba = bool(enable_simba)
        self.simba_bsize = max(1, int(simba_bsize))
        self.simba_num_candidates = max(1, int(simba_num_candidates))
        self.simba_max_steps = max(1, int(simba_max_steps))
        self.simba_max_demos = max(0, int(simba_max_demos))
        self.max_instruction_chars = (
            int(max_instruction_chars) if max_instruction_chars is not None else None
        )
        self.max_field_chars = max(120, int(max_field_chars))
        self.max_feedback_examples = min(5, max(1, int(max_feedback_examples)))
        self.reflection_temperature = float(reflection_temperature)
        self.worker_temperature = float(worker_temperature)
        self.json_adapter: dspy.Adapter = adapter or YawGBNFAdapter(
            enable_gbnf=self.enable_gbnf,
            max_instruction_chars=self.max_instruction_chars or 1500,
        )
        self.telemetry: list[dict[str, Any]] = DiagnosticHistory()
        self.simba_telemetry: list[dict[str, Any]] = DiagnosticHistory()
        self._simba_steps = 0
        self.task_scorer = task_scorer
        self.save_simba_state = save_simba_state
        self._simba_reward_errors = 0
        self._simba_examples: list[dspy.Example] = []
        self._sub_telemetry: list[dict[str, Any]] = []

        from dspy.teleprompt.gepa.instruction_proposal import InstructionProposer

        native = InstructionProposer(
            base_instructions=ProposeInstructionSignature.__doc__,
            max_chars=self.max_instruction_chars,
        )
        self.qwen_proposer = native.propose
        # Keep both native predictors visible to SIMBA and state serialization.
        self.qwen_compressor = native.compress or dspy.Predict(CompressInstructionSignature)
        self.worker_module = _InstructionSynthesizerWorker(
            qwen_proposer=self.qwen_proposer,
            qwen_compressor=self.qwen_compressor,
            worker_lm=self.worker_lm,
            orchestrator_lm=self.orchestrator_lm,
            enable_gbnf=self.enable_gbnf,
            max_instruction_chars=self.max_instruction_chars,
            max_field_chars=self.max_field_chars,
            worker_temperature=self.worker_temperature,
            json_adapter=self.json_adapter,
        )

        self.inner_rlm = _PositionalSafeRLM(
            ReflectiveInnerDistillerSignature,
            stage_rule=_STAGE_RLM_DELEGATION_RULES["instruction_proposer_inner"],
            max_iters=rlm_max_iters,
            max_llm_calls=rlm_max_llm_calls,
            max_output_chars=rlm_max_output_chars,
            tools=[],
            sub_lm=worker_lm or orchestrator_lm,
            interpreter_factory=rlm_interpreter_factory,
        )
        self.inner_rlm.generate_action.config = {"temperature": self.reflection_temperature}
        self.inner_rlm.extract.config = {"temperature": self.reflection_temperature}
        self.rlm = self.inner_rlm

        self._bind_predictor_lms()

        if include_demos:
            rlm_demos = _build_rlm_action_demos()
            self.inner_rlm.generate_action.demos = list(rlm_demos["instruction_proposer_inner"])

    def _bind_predictor_lms(self) -> None:
        """Bind dedicated `_PredictorBoundLM` instances to all 3 reflection predictors."""

        def worker_getter() -> Any:
            return self.worker_lm or self.orchestrator_lm

        def orch_getter() -> Any:
            return self.orchestrator_lm or self.worker_lm

        worker_bound = _PredictorBoundLM("qwen_proposer", "worker", worker_getter)
        comp_bound = _PredictorBoundLM("qwen_compressor", "worker", worker_getter)
        self.qwen_proposer.lm = worker_bound
        self.qwen_compressor.lm = comp_bound
        self.worker_module.qwen_proposer = self.qwen_proposer
        self.worker_module.qwen_compressor = self.qwen_compressor
        self.worker_module.worker_lm = self.worker_lm
        self.worker_module.orchestrator_lm = self.orchestrator_lm

        inner_action_lm = _PredictorBoundLM(
            "inner_rlm.generate_action", "orchestrator", orch_getter
        )
        inner_extract_lm = _PredictorBoundLM("inner_rlm.extract", "orchestrator", orch_getter)
        self.inner_rlm.generate_action.lm = inner_action_lm
        self.inner_rlm.extract.lm = inner_extract_lm
        self.inner_rlm.lm = inner_action_lm

    def get_lm(self) -> Any:
        return self.worker_module.get_lm()

    def set_lm(self, lm: Any) -> None:
        self.worker_module.set_lm(lm)

    def __deepcopy__(self, memo: dict[int, Any]) -> Any:
        import copy

        new = type(self).__new__(type(self))
        memo[id(self)] = new
        callbacks = {
            "orchestrator_lm",
            "worker_lm",
            "json_adapter",
            "token_counter",
            "task_scorer",
            "save_simba_state",
            "checkpoint_get",
            "checkpoint_put",
        }
        for key, value in self.__dict__.items():
            setattr(new, key, value if key in callbacks else copy.deepcopy(value, memo))
        new._bind_predictor_lms()
        return new

    def preflight(self) -> None:
        """Require an exact tokenizer before any reflection or meta-learning call."""
        if self.token_counter is None:
            raise ReflectionBudgetError("Reflection requires a verified serving tokenizer")
        for name in ("worker_lm", "orchestrator_lm"):
            lm = getattr(self, name)
            if lm is not None and not isinstance(lm, ReflectionLM):
                setattr(
                    self,
                    name,
                    ReflectionLM(
                        lm,
                        self.token_counter,
                        self.reflection_input_tokens,
                        getattr(lm, "_yaw_context_window_tokens", None) or 65536,
                        int(lm.kwargs.get("max_tokens") or 4096),
                    ),
                )
        self.inner_rlm.sub_lm = self.worker_lm or self.orchestrator_lm
        self._bind_predictor_lms()

    def all_reflection_predictors(self) -> list[tuple[str, Any]]:
        """Return all 3 active reflection predictors (`inner_rlm`, `qwen_proposer`, `qwen_compressor`)."""
        return [
            ("inner_rlm", self.inner_rlm),
            ("qwen_proposer", self.qwen_proposer),
            ("qwen_compressor", self.qwen_compressor),
        ]

    def dump_simba_state(self) -> dict[str, Any]:
        """JSON-safe DSPy predictor state; exclude clients, traces and optimizer archives."""
        predictors = {}
        for name in ("qwen_proposer", "qwen_compressor"):
            state = getattr(self, name).dump_state()
            state.update(lm=None, traces=[], train=[])
            predictors[name] = state
        return {
            "version": 2,
            "steps": self._simba_steps,
            "examples": [ex.toDict() for ex in self._simba_examples[-max(self.simba_bsize, 8) :]],
            "predictors": predictors,
        }

    def load_simba_state(self, state: object) -> None:
        """Restore learned parameters into fresh predictors, then bind current clients."""
        state = validate_simba_state(state)
        restored = {}
        for name in ("qwen_proposer", "qwen_compressor"):
            predictor = dspy.Predict(getattr(self, name).signature)
            saved = dict(state["predictors"][name])
            saved.update(lm=None, traces=[], train=[])
            signature = predictor.signature
            if signature is None or len(saved["signature"]["fields"]) != len(signature.fields):
                raise ValueError(f"SIMBA predictor {name} has an incompatible field schema")
            try:
                predictor.load_state(saved)
            except (KeyError, TypeError, IndexError) as exc:
                raise ValueError(f"Malformed SIMBA predictor {name}") from exc
            restored[name] = predictor
        examples = [
            dspy.Example(**ex).with_inputs(
                "component_name", "current_instruction", "examples_with_feedback"
            )
            for ex in state["examples"][-max(self.simba_bsize, 8) :]
        ]
        for name, predictor in restored.items():
            # Preserve the identities also held by the synthesis tool and worker.
            getattr(self, name).load_state(predictor.dump_state())
        self._simba_examples = examples
        self._simba_steps = int(state["steps"])
        self._bind_predictor_lms()

    def _score_task_improvement(self, example: Any, prediction: Any) -> float:
        """Reward a counterfactual task-score gain, never instruction formatting."""
        try:
            if self.task_scorer is None:
                raise RuntimeError("SIMBA requires a task scorer")
            candidate = dict(example.task_candidate)
            component = str(example.component_name)
            instruction = self._ensure_stage_schema_contract(
                component,
                str(example.current_instruction),
                str(getattr(prediction, "new_instruction", "") or "").strip(),
            )
            baseline = float(self.task_scorer(candidate))
            candidate[component] = instruction
            improvement = float(self.task_scorer(candidate)) - baseline
            if not math.isfinite(improvement):
                raise ValueError("Non-finite SIMBA task improvement")
            return 0.5 + 0.5 * improvement / (1.0 + abs(improvement))
        except Exception:
            # DSPy can swallow metric exceptions as zero scores. Refuse to adopt
            # a worker selected from incomplete/failed task measurements.
            self._simba_reward_errors += 1
            raise

    def step_simba(self) -> bool:
        """Run `dspy.SIMBA` on the inner worker (`self.worker_module`) over buffered reflection examples."""
        if (
            not self.enable_simba
            or self.task_scorer is None
            or len(self._simba_examples) < self.simba_bsize
        ):
            return False
        if not hasattr(dspy, "SIMBA"):
            return False

        self.preflight()
        eff_worker = self.worker_lm or self.orchestrator_lm or getattr(dspy.settings, "lm", None)
        if eff_worker is None:
            return False

        bsize = min(self.simba_bsize, len(self._simba_examples))
        trainset = list(self._simba_examples[-max(bsize, 8) :])
        prev_inst = str(getattr(self.qwen_proposer.signature, "instructions", "") or "")
        prev_demos = len(getattr(self.qwen_proposer, "demos", ()) or ())

        try:
            self._simba_reward_errors = 0
            simba = dspy.SIMBA(
                metric=self._score_task_improvement,
                bsize=bsize,
                num_candidates=self.simba_num_candidates,
                max_steps=self.simba_max_steps,
                max_demos=self.simba_max_demos,
                demo_input_field_maxlen=sys.maxsize,
                prompt_model=eff_worker,
                num_threads=1,
            )
            with dspy.context(lm=eff_worker, adapter=self.json_adapter, trace=[]):
                compiled_worker = simba.compile(
                    self.worker_module,
                    trainset=trainset,
                    seed=self._simba_steps,
                )
            if self._simba_reward_errors:
                raise RuntimeError("SIMBA task evaluations failed; retaining previous worker")
            if not isinstance(compiled_worker, _InstructionSynthesizerWorker):
                raise TypeError("SIMBA returned an incompatible instruction worker")
            # Learned instructions/demos are also part of the next rendered prompt.
            # Reject an oversized worker before adopting or checkpointing its state.
            compiled_worker._rollout_lm = None
            for example in trainset:
                compiled_worker.fit_evidence(
                    example.current_instruction,
                    example.examples_with_feedback,
                    example.component_name,
                )
            new_inst = str(compiled_worker.qwen_proposer.signature.instructions).strip()
            self.qwen_proposer.signature = compiled_worker.qwen_proposer.signature
            self.qwen_proposer.demos = list(
                getattr(compiled_worker.qwen_proposer, "demos", ()) or ()
            )
            self.qwen_compressor.signature = compiled_worker.qwen_compressor.signature
            self.qwen_compressor.demos = list(
                getattr(compiled_worker.qwen_compressor, "demos", ()) or ()
            )
            self._bind_predictor_lms()

            entry = {
                "optimizer": "dspy.SIMBA",
                "target": "worker_module.qwen_proposer",
                "trainset_size": len(trainset),
                "bsize": bsize,
                "num_candidates": self.simba_num_candidates,
                "max_steps": self.simba_max_steps,
                "demos_before": prev_demos,
                "demos_after": len(self.qwen_proposer.demos),
                "instruction_changed": new_inst != prev_inst.strip(),
                "proposer_instruction": new_inst,
            }
            self.simba_telemetry.append(entry)
            self._simba_steps += 1
            trim_diagnostics(self)
        except (
            AdapterParseError,
            LMError,
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
            StopIteration,
            KeyError,
            IndexError,
            AssertionError,
        ) as exc:
            logging.getLogger(__name__).warning(
                "SIMBA inner-worker step skipped (%s: %s)",
                type(exc).__name__,
                exc,
            )
            return False
        # Persistence failures must reach the caller, rather than look like a
        # successful durable step. SQLite keeps the previous complete snapshot.
        if self.save_simba_state is not None:
            self.save_simba_state(self.dump_simba_state())
        return True

    @staticmethod
    def _prepare_examples(
        examples: Any,
        *,
        max_examples: int = 5,
    ) -> list[dict[str, Any]]:
        """Prepare reflective examples, prioritizing top-5 failing examples and anonymizing gold literals."""
        if isinstance(examples, str):
            examples = json.loads(examples)
        if not examples:
            return []
        return reflection_examples(examples)[:max_examples]

    def _ensure_stage_schema_contract(
        self,
        component_name: str,
        current_instruction: str,
        draft: str,
    ) -> str:
        """Validate the instruction; the immutable DSPy Signature owns the output schema."""
        if not draft.strip():
            raise ValueError("Empty instruction")
        if self.max_instruction_chars is not None and len(draft) > self.max_instruction_chars:
            raise ValueError("Instruction exceeds the native writer length limit")
        return draft

    def cache_identity(self) -> dict[str, Any]:
        """All deployment/policy settings that affect reflection stage results."""
        models = []
        for lm in (self.orchestrator_lm, self.worker_lm):
            models.append(
                {
                    "model": getattr(lm, "model", None),
                    "kwargs": {
                        key: value
                        for key, value in getattr(lm, "kwargs", {}).items()
                        if key not in {"api_key", "api_token"}
                    },
                    "context": getattr(lm, "_yaw_context_window_tokens", None),
                    "tokenizer": getattr(getattr(lm, "counter", None), "identity", None),
                }
            )
        return {
            "version": 2,
            "models": models,
            "tokenizer": getattr(self.token_counter, "identity", None),
            "input_tokens": self.reflection_input_tokens,
            "settings": [
                self.max_instruction_chars,
                self.max_feedback_examples,
                self.worker_temperature,
                self.reflection_temperature,
                self.enable_gbnf,
            ],
            "selector": [
                self.inner_rlm.max_iters,
                self.inner_rlm.max_llm_calls,
                self.inner_rlm.max_output_chars,
            ],
            "adapter": type(self.json_adapter).__qualname__,
        }

    def select_evidence(
        self,
        component_name: str,
        current_instruction: str,
        component_examples: Sequence[Mapping[str, Any]],
    ) -> tuple[str, list[dict[str, Any]], Any]:
        """Checkpoint validated references, retaining exact training evidence."""
        self.preflight()
        originals = reflection_examples(component_examples)
        evidence_key = reflection_digest(
            {
                "version": 2,
                "deployment": self.cache_identity(),
                "component": component_name,
                "instruction": current_instruction,
                "examples": originals,
                "input_tokens": self.reflection_input_tokens,
                "writer_settings": [
                    self.max_instruction_chars,
                    self.max_feedback_examples,
                    self.worker_temperature,
                    self.reflection_temperature,
                    self.enable_gbnf,
                ],
                "selector": self.inner_rlm.generate_action.signature.instructions,
                "selector_limits": [
                    self.inner_rlm.max_iters,
                    self.inner_rlm.max_llm_calls,
                    self.inner_rlm.max_output_chars,
                ],
                "model": self.orchestrator_lm.model if self.orchestrator_lm else None,
                "tokenizer": getattr(self.token_counter, "identity", str(self.token_counter)),
            }
        )
        get = self.checkpoint_get or (lambda key: None)
        put = self.checkpoint_put or (lambda key, value: None)
        saved = get("evidence:" + evidence_key)
        if saved is None:
            fallback = None
            try:
                with dspy.context(lm=self.orchestrator_lm or self.worker_lm, trace=[]):
                    result = self.inner_rlm(
                        current_instruction=current_instruction, examples_with_feedback=originals
                    )
                selection = checkpoint_object(result.selection, "evidence selection")
                resolve_evidence(originals, selection, self.max_feedback_examples)
            except (
                ValueError,
                KeyError,
                TypeError,
                RuntimeError,
                IndexError,
                AdapterParseError,
                LMError,
                OSError,
            ) as exc:
                fallback = type(exc).__name__
                selection = deterministic_evidence(originals, self.max_feedback_examples)
            saved = {"selection": selection, "fallback": fallback}
            put("evidence:" + evidence_key, saved)
        saved = checkpoint_object(saved, "evidence checkpoint")
        selection = checkpoint_object(saved.get("selection"), "evidence selection")
        fallback = saved.get("fallback")
        if fallback is not None and not isinstance(fallback, str):
            raise ValueError("Malformed evidence checkpoint fallback")
        packet = resolve_evidence(originals, selection, self.max_feedback_examples)
        diagnosis = selection.get("diagnosis", "")
        if not isinstance(diagnosis, str):
            raise ValueError("Malformed evidence diagnosis")
        packet.insert(0, {"diagnosis": diagnosis[:2000]})
        return evidence_key, packet, fallback

    @diagnostic_boundary
    def propose_for_component(
        self,
        component_name: str,
        current_instruction: str,
        component_examples: Sequence[Mapping[str, Any]],
    ) -> str:
        evidence_key, packet, fallback = self.select_evidence(
            component_name, current_instruction, component_examples
        )
        get = self.checkpoint_get or (lambda key: None)
        put = self.checkpoint_put or (lambda key, value: None)
        key = "proposal:" + reflection_digest(
            {
                "evidence": evidence_key,
                "state": self.dump_simba_state(),
                "worker": self.worker_lm.model if self.worker_lm else None,
            }
        )
        completed = get(key)
        if completed is not None:
            completed = checkpoint_object(completed, "proposal checkpoint")
            instruction = completed.get("instruction")
            if not isinstance(instruction, str):
                raise ValueError("Malformed proposal checkpoint instruction")
            return self._ensure_stage_schema_contract(
                component_name.removesuffix(".predict"), current_instruction, instruction
            )
        from dspy.utils.usage_tracker import track_usage

        self.worker_module.sub_telemetry.clear()
        with dspy.context(trace=[]), track_usage() as tracker:
            prediction = self.worker_module(
                current_instruction=current_instruction,
                examples_with_feedback=packet,
                component_name=component_name,
            )
        usage = tracker.get_total_tokens()
        instruction = self._ensure_stage_schema_contract(
            component_name.removesuffix(".predict"),
            current_instruction,
            str(prediction.new_instruction),
        )
        put(key, {"instruction": instruction})
        self.telemetry.append(
            {
                "component_name": component_name,
                "mode": "evidence_native",
                "fallback": fallback,
                "evidence_key": evidence_key,
                "usage": usage,
                "tokens_used": _sum_usage_tokens(usage),
                "new_instruction": instruction,
                "simba_steps": self._simba_steps,
            }
        )
        return instruction

    def forward(
        self,
        candidate: dict[str, str] | None = None,
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
        components_to_update: Sequence[str] | None = None,
        *,
        component_name: str = "",
        current_instruction: str = "",
        examples_with_feedback: Any = None,
    ) -> Any:
        """Dual entrypoint: GEPA `ProposalFn` `(candidate, reflective_dataset, components_to_update)` or `dspy.Module` `(component_name=..., current_instruction=..., examples_with_feedback=...)`."""
        if candidate is None or not isinstance(candidate, Mapping):
            self.preflight()
            prepared = self._prepare_examples(
                examples_with_feedback,
                max_examples=self.max_feedback_examples,
            )
            with dspy.context(trace=[]):
                pred = self.worker_module(
                    current_instruction=current_instruction,
                    examples_with_feedback=prepared,
                    component_name=component_name,
                )
            inst = self._ensure_stage_schema_contract(
                component_name,
                current_instruction,
                str(getattr(pred, "new_instruction", "") or "").strip(),
            )
            return dspy.Prediction(new_instruction=inst)

        if not components_to_update or reflective_dataset is None:
            return {}

        self.preflight()
        get = self.checkpoint_get or (lambda key: None)
        put = self.checkpoint_put or (lambda key, value: None)
        request = reflection_digest(
            {
                "version": 2,
                "deployment": self.cache_identity(),
                "candidate": candidate,
                "input_tokens": self.reflection_input_tokens,
                "tokenizer": getattr(self.token_counter, "identity", None),
                "writer_settings": [
                    self.max_instruction_chars,
                    self.max_feedback_examples,
                    self.worker_temperature,
                    self.reflection_temperature,
                    self.enable_gbnf,
                ],
                "data": reflection_examples(
                    [{"component": k, "examples": v} for k, v in reflective_dataset.items()]
                ),
                "components": list(components_to_update),
            }
        )
        checkpoint_key = "request:" + request

        def snapshot(phase: str, **extra: Any) -> dict[str, Any]:
            state = self.dump_simba_state()
            return {
                "phase": phase,
                "state": state,
                "writer": reflection_digest(state["predictors"]),
                **extra,
            }

        record = get(checkpoint_key)
        if record is not None:
            record = validate_request_checkpoint(record)
        current_writer = reflection_digest(self.dump_simba_state()["predictors"])
        if record is not None and record["phase"] == "complete":
            saved_writer = record.get("writer", reflection_digest(record["state"]["predictors"]))
            if saved_writer != current_writer:
                record = None
        if record is not None:
            if record["phase"] != "complete" or record["state"]["steps"] > self._simba_steps:
                self.load_simba_state(record["state"])
            if record["phase"] == "complete":
                allowed = set(components_to_update) & set(candidate)
                if set(record["proposals"]) - allowed:
                    raise ValueError("Completed proposal has an unknown component")
                for name, instruction in record["proposals"].items():
                    self._ensure_stage_schema_contract(name, candidate[name], instruction)
                return record["proposals"]
        else:
            record = snapshot("collecting")
            put(checkpoint_key, record)
        if record["phase"] == "collecting":
            for name in components_to_update:
                if name not in candidate or not reflective_dataset.get(name):
                    continue
                _, packet, _ = self.select_evidence(name, candidate[name], reflective_dataset[name])
                prepared = self.worker_module.fit_evidence(candidate[name], packet, name)
                self._simba_examples.append(
                    dspy.Example(
                        task_candidate=dict(candidate),
                        component_name=name,
                        current_instruction=candidate[name],
                        examples_with_feedback=prepared,
                    ).with_inputs("component_name", "current_instruction", "examples_with_feedback")
                )
            del self._simba_examples[: -max(self.simba_bsize, 8)]
            # The request-local state is authoritative on restart, so an incomplete
            # SIMBA call is retried from its input snapshot, never applied twice.
            record = snapshot("learning")
            put(checkpoint_key, record)
        if record["phase"] == "learning":
            self.step_simba()
            record = snapshot("learned")
            put(checkpoint_key, record)
        updated = {
            name: self.propose_for_component(name, candidate[name], reflective_dataset[name])
            for name in components_to_update
            if name in candidate and reflective_dataset.get(name)
        }
        put(
            checkpoint_key,
            snapshot("complete", proposals=updated),
        )
        if self.save_simba_state is not None:
            self.save_simba_state(self.dump_simba_state())
        return updated

    def propose_mipro_instructions(
        self,
        program: Any,
        trainset: Sequence[Any],
        demo_candidates: Any = None,
        view_data_batch_size: int = 10,
        program_aware_proposer: bool = True,
        data_aware_proposer: bool = True,
        tip_aware_proposer: bool = True,
        fewshot_aware_proposer: bool = True,
        *,
        num_instruct_candidates: int = 1,
    ) -> dict[int, list[str]]:
        """Propose MIPROv2 instruction candidates per predictor using `self.inner_rlm` (`dspy.RLM`) instead of stock `GroundedProposer`."""
        named_preds = list(program.named_predictors())
        program_summary = (
            ", ".join(
                f"{name}:{getattr(getattr(p, 'signature', None), '__name__', type(p).__name__)}"
                for name, p in named_preds
            )
            if program_aware_proposer
            else ""
        )
        batch_limit = max(1, min(len(trainset), int(view_data_batch_size or 5)))
        dataset_records: list[dict[str, Any]] = []
        if data_aware_proposer:
            for ex in list(trainset)[:batch_limit]:
                if hasattr(ex, "inputs"):
                    inp_d = dict(ex.inputs())
                    lbl_d = dict(ex.labels()) if hasattr(ex, "labels") else {}
                elif isinstance(ex, Mapping):
                    inp_d = dict(ex)
                    lbl_d = {}
                else:
                    inp_d = {"example": str(ex)}
                    lbl_d = {}
                dataset_records.append(
                    {
                        "Inputs": inp_d,
                        "Generated Outputs": lbl_d,
                        "Feedback": (
                            f"[MIPROv2 RLM Dataset/Program Context] modules=[{program_summary}]"
                            if program_summary
                            else "[MIPROv2 RLM Dataset Context]"
                        ),
                    }
                )

        tips = (
            "Keep the instruction clear and concise.",
            "Make sure your instruction is very informative and descriptive.",
            "Don't be afraid to be creative when creating the new instruction!",
        )
        instruction_candidates: dict[int, list[str]] = {}
        target_n = max(1, int(num_instruct_candidates))
        for pred_i, (pred_name, pred) in enumerate(named_preds):
            sig = getattr(pred, "signature", None)
            base_inst = str(getattr(sig, "instructions", "") or "")
            cands: list[str] = [base_inst]
            for cand_idx in range(1, target_n):
                tip_str = tips[(cand_idx - 1) % len(tips)] if tip_aware_proposer else ""
                comp_examples: list[dict[str, Any]] = []
                if (
                    fewshot_aware_proposer
                    and isinstance(demo_candidates, Mapping)
                    and pred_i in demo_candidates
                    and demo_candidates[pred_i]
                ):
                    demo_sets = demo_candidates[pred_i]
                    chosen_set = demo_sets[cand_idx % len(demo_sets)] or ()
                    sig_fields = getattr(sig, "fields", {}) or {}
                    in_keys = {
                        k
                        for k, f in sig_fields.items()
                        if getattr(f, "json_schema_extra", {}).get("__dspy_field_type") == "input"
                    }
                    for d_ex in list(chosen_set)[: self.max_feedback_examples]:
                        if isinstance(d_ex, Mapping):
                            d_in = {
                                k: v for k, v in d_ex.items() if k in in_keys and k != "augmented"
                            }
                            d_out = {
                                k: v
                                for k, v in d_ex.items()
                                if k not in in_keys and k != "augmented"
                            }
                            comp_examples.append(
                                {
                                    "Inputs": d_in,
                                    "Generated Outputs": d_out,
                                    "Feedback": f"Bootstrapped demonstration for {pred_name}. {tip_str}".strip(),
                                }
                            )
                if not comp_examples:
                    comp_examples = [
                        {
                            **rec,
                            "Feedback": f"{rec.get('Feedback', '')} {tip_str}".strip(),
                        }
                        for rec in (
                            dataset_records[: self.max_feedback_examples]
                            or [{"Inputs": {}, "Generated Outputs": {}, "Feedback": tip_str}]
                        )
                    ]
                proposed = self.propose_for_component(
                    component_name=pred_name,
                    current_instruction=base_inst,
                    component_examples=comp_examples,
                )
                cands.append(proposed)
            instruction_candidates[pred_i] = cands
        return instruction_candidates

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)


__all__ = [
    "HippoRAGProgram",
    "ParetoProgramEnsemble",
    "RLMInstructionProposer",
    "YawAssertionError",
    "YawGBNFAdapter",
    "YawPassage",
    "YawRM",
    "YawRetriever",
    "is_abstaining_answer",
    "validate_stage_invariants",
    "wrap_stage_with_refine",
    "yaw_assert",
    "yaw_suggest",
]
