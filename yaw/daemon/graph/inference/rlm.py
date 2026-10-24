"""Positional-safe DSPy RLM orchestrator, REPL code sanitizer, and stage delegation rules."""

import functools
import inspect
import json
from collections.abc import Callable
from typing import Any

import dspy

from .parsers import (
    coerce_generated_facts,
    extract_entity_definitions,
    extract_named_entities,
    extract_triples,
    parse_shacl_repairs,
    parse_synonym_decisions,
)

STAGE_RLM_DELEGATION_RULES: dict[str, str] = {
    "shacl_repair": (
        "CRITICAL RLM SANDBOX RULE:\n"
        "Prioritize aggressive 2-step resolution: do NOT perform exploratory node inspections "
        "unless strictly necessary.\n"
        "Immediately delegate to `qwen_propose_shacl_repairs(violations=violations)` on Step 1, "
        "verify if needed, and call `SUBMIT(repairs=repairs)` by Step 2 without wandering:\n"
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


def build_rlm_action_demos() -> dict[str, list[dspy.Example]]:
    """Build few-shot action demonstrations for single-level RLM sandboxes."""
    from dspy.primitives.repl_types import REPLHistory

    empty_history = REPLHistory()
    return {
        "shacl_repair": [
            dspy.Example(
                variables_info="Variable: `violations` (type: str): JSON list of SHACLViolation report items",
                repl_history=empty_history,
                iteration="1/10",
                reasoning="Immediately delegate to Qwen worker and submit in step 1 without wandering.",
                code="```python\n"
                "repairs = qwen_propose_shacl_repairs(violations=violations)\n"
                "SUBMIT(repairs=repairs)\n"
                "```",
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
        return val.strip()
    return val


def sanitize_repl_code(code: str) -> str:
    """Strip hallucinated imports of pre-bound sandbox tools (`qwen_*`, `inspect_focus_node`, `hydrate_chunk`, \
`apply_shacl_repairs`, `lint_shacl`, `synthesize_instruction`, `SUBMIT`, `llm_query`)."""
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


class PositionalSafeRLM(_RLM_Base):
    """RLM subclass that binds positional tool arguments, enforces 1-step Qwen delegation, and normalizes SUBMIT \
payloads."""

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
            signature=self._action_signature_for_current_factory(interpreter_factory, sub_agent_rules),
            variables_info=variables_info,
            repl_history=history,
            iteration=f"{iteration + 1}/{self.max_iters}",
        )
        self.last_reasoning = str(getattr(action, "reasoning", "") or "").strip()
        from dspy.predict.rlm import _strip_code_fences, format_error_for_lm

        try:
            code = sanitize_repl_code(_strip_code_fences(action.code))
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
            if len(output_field_names) == 1 and output_field_names[0] not in raw_output and len(raw_output) == 1:
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
                setattr(pred, name, val.strip())
        return pred


__all__ = [
    "PositionalSafeRLM",
    "STAGE_RLM_DELEGATION_RULES",
    "build_rlm_action_demos",
    "sanitize_repl_code",
]
