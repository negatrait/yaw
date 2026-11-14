"""DSPy RLM orchestrator, REPL code sanitizer, and stage delegation rules."""

from collections.abc import Callable
from typing import Any

import dspy

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
    """RLM subclass with stage instructions and sanitized REPL code."""

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
            history=history,
            iteration=self._iteration_label(iteration),
        )
        self.last_reasoning = str(getattr(action, "reasoning", "") or "").strip()
        from dspy.predict.rlm import _strip_code_fences, format_error_for_lm

        try:
            code = sanitize_repl_code(_strip_code_fences(action.code))
        except SyntaxError as e:
            code = action.code
            result = f"[Error] {format_error_for_lm(e)}"
            return self._process_execution_result(action, code, result, output_field_names)
        result = self._execute_code(repl, code, input_args)
        return self._process_execution_result(action, code, result, output_field_names)


__all__ = [
    "PositionalSafeRLM",
    "STAGE_RLM_DELEGATION_RULES",
    "sanitize_repl_code",
]
