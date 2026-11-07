"""Positional-safe DSPy RLM orchestrator, REPL code sanitizer, and stage delegation rules."""

import functools
import inspect
from collections.abc import Callable
from typing import Any

import dspy

from .parsers import parse_shacl_repairs

STAGE_RLM_DELEGATION_RULES: dict[str, str] = {
    "shacl_repair": (
        "RLM SHACL HEALING WORKFLOW (SINGLE-STEP DELEGATION):\n"
        "`propose_repairs()` takes zero arguments and automatically supplies the complete structured "
        "`inspect_focus_node` dossiers, W3C SKOS/RDF/OWL profiles, endpoint entities, and hydrated "
        "`source_passages` for every reported violation in `violations`. Do NOT spend an iteration "
        "printing `violations` or calling `inspect_focus_node` before `propose_repairs()`. In Iteration 1, "
        "delegate once and submit immediately:\n"
        "```python\n"
        "repairs = propose_repairs()\n"
        "SUBMIT(repairs=repairs)\n"
        "```"
    ),
    "instruction_proposer_inner": (
        "Inspect examples_with_feedback directly in Python without calling llm_query or llm_query_batched, and "
        "call SUBMIT immediately once evidence is selected. Select evidence by evidence_id and field path inside "
        "that example dict (use `[]` for the full example or `['Feedback']`, never an example list index like `[0]`); "
        "use start/end character offsets only when `path` resolves to a string. Include failures and successful "
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
                reasoning=(
                    "Delegate once to propose_repairs(), which supplies the complete structured focus dossiers "
                    "and source passages automatically, and immediately submit its repair proposals."
                ),
                code="```python\nrepairs = propose_repairs()\nSUBMIT(repairs=repairs)\n```",
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
    if name == "repairs":
        return parse_shacl_repairs(val)
    if annotation is str and isinstance(val, str):
        return val.strip()
    return val


def sanitize_repl_code(code: str) -> str:
    """Strip hallucinated imports of pre-bound sandbox tools (`llm_*`, `inspect_focus_node`, `inspect_pair`, \
`hydrate_chunk`, `hydrate_ast_island`, `search_graph`, `apply_shacl_repairs`, `lint_shacl`, `synthesize_instruction`, \
`SUBMIT`, `llm_query`)."""
    cleaned_lines: list[str] = []
    for line in code.splitlines():
        stripped = line.strip()
        if (stripped.startswith("from ") or stripped.startswith("import ")) and (
            "llm_" in stripped
            or "inspect_node" in stripped
            or "inspect_focus_node" in stripped
            or "inspect_pair" in stripped
            or "hydrate_chunk" in stripped
            or "hydrate_ast_island" in stripped
            or "read_source_chunk" in stripped
            or "search_graph" in stripped
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


__all__ = [
    "PositionalSafeRLM",
    "STAGE_RLM_DELEGATION_RULES",
    "build_rlm_action_demos",
    "sanitize_repl_code",
]
