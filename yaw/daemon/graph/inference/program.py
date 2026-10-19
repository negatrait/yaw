"""Native DSPy HippoRAGProgram and _PositionalSafeRLM orchestrator."""

import functools
import inspect
import json
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import dspy

from yaw.core.config import YawConfig
from yaw.core.types import OpenIERecord, StructuredTriple

from .chat import create_dspy_lm
from .grammars import (
    DSPY_INSTRUCTION_PROPOSAL_GBNF_GRAMMAR,
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

_STAGE_RLM_DELEGATION_RULES: dict[str, str] = {
    "shacl_repair": (
        "CRITICAL RLM SANDBOX RULE:\n"
        "Inspect focus nodes via `inspect_focus_node(entity_id=...)` and `hydrate_chunk(chunk_id=...)`, delegate to `qwen_propose_shacl_repairs(violations=violations, focus_context=...)`, optionally apply/verify via `apply_shacl_repairs(repairs=repairs)` and `lint_shacl()`, and call `SUBMIT`:\n"
        "```python\n"
        "repairs = qwen_propose_shacl_repairs(violations=violations)\n"
        "SUBMIT(repairs=repairs)\n"
        "```"
    ),
    "instruction_proposer_inner": (
        "CRITICAL RLM SANDBOX RULE:\n"
        "Do NOT import `synthesize_instruction` (`synthesize_instruction` and `SUBMIT` are already in global scope).\n"
        "Distill `examples_with_feedback` in Python so large inputs fit in context, call `synthesize_instruction` by keyword, and `SUBMIT`:\n"
        "```python\n"
        "digest = [{'Inputs': {k: (v[:400] if isinstance(v, str) else v) for k, v in ex.get('Inputs', {}).items()}, 'Generated Outputs': ex.get('Generated Outputs', {}), 'Feedback': ex.get('Feedback', '')} for ex in examples_with_feedback]\n"
        "res = synthesize_instruction(current_instruction=current_instruction, examples_with_feedback=digest)\n"
        "SUBMIT(new_instruction=res)\n"
        "```"
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
                iteration="1/3",
                reasoning="Delegate SHACL repair generation to the Qwen worker and submit.",
                code="```python\nrepairs = qwen_propose_shacl_repairs(violations=violations)\nSUBMIT(repairs=repairs)\n```",
            ).with_inputs("variables_info", "repl_history", "iteration")
        ],
        "instruction_proposer_inner": [
            dspy.Example(
                variables_info=(
                    "Variable: `current_instruction` (type: str): The instructions currently provided to the assistant.\n"
                    "Variable: `examples_with_feedback` (type: list): Task inputs, assistant responses, and diagnostic evaluation feedback."
                ),
                repl_history=empty_history,
                iteration="1/3",
                reasoning="Distill the examples and feedback in Python and call synthesize_instruction.",
                code=(
                    "```python\n"
                    "digest = [{'Inputs': {k: (v[:400] if isinstance(v, str) else v) for k, v in ex.get('Inputs', {}).items()}, 'Generated Outputs': ex.get('Generated Outputs', {}), 'Feedback': ex.get('Feedback', '')} for ex in examples_with_feedback]\n"
                    "res = synthesize_instruction(current_instruction=current_instruction, examples_with_feedback=digest)\n"
                    "SUBMIT(new_instruction=res)\n"
                    "```"
                ),
            ).with_inputs("variables_info", "repl_history", "iteration")
        ],
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
            except (json.JSONDecodeError, TypeError, ValueError):
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
        if (
            (stripped.startswith("from ") or stripped.startswith("import "))
            and (
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
            )
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
            return self._process_execution_result(
                action, code, result, history, output_field_names
            )
        result = self._execute_code(repl, code, input_args)
        return self._process_execution_result(
            action, code, result, history, output_field_names
        )

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
        setattr(invoke, "__signature__", sig)
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
                for var in (variables if isinstance(variables, list) else []):
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
            for item in (history or []):
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


class YawJSONAdapter(dspy.JSONAdapter):
    """Robust JSON adapter that recovers from top-level arrays or mismatched single fields."""

    def parse(self, signature: type[dspy.Signature], completion: str) -> dict[str, Any]:
        try:
            return super().parse(signature, completion)
        except Exception:
            if "[[ ##" in completion:
                return dspy.ChatAdapter().parse(signature, completion)
            import json_repair
            from dspy.adapters.utils import parse_value

            fields = json_repair.loads(completion)
            out_fields = list(signature.output_fields.keys())
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

    def copy(self, **kwargs: Any) -> "_PredictorBoundLM":
        cloned = _PredictorBoundLM(
            self.predictor_name,
            self.role,
            self._base_lm_getter,
        )
        if kwargs:
            cloned.kwargs = {**self.kwargs, **kwargs}
        return cloned

    def __deepcopy__(self, memo: dict[int, Any]) -> "_PredictorBoundLM":
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
        prev_len = len(getattr(target, "history", ()))
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
            for entry in target_hist[prev_len:]:
                if isinstance(entry, Mapping):
                    rec = dict(entry)
                    rec["predictor"] = self.predictor_name
                    self.history.append(rec)
                else:
                    self.history.append({"predictor": self.predictor_name, "entry": entry})
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


class HippoRAGProgram(dspy.Module):
    """Unified DSPy Module for HippoRAG 2 OpenIE, EDC/SKOS Schema Canon, SHACL Repair, Recognition Memory, and Multi-Hop QA.

    Exposes the 9 canonical runtime stage predictors (out of the 12 total DSPy signatures in YAW):
    1. **8 Direct GBNF Predictors (`Qwen3-4B-Instruct-2507-GGUF` on `llama.cpp` GPU):**
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
    ) -> None:
        super().__init__()
        self.enable_gbnf: bool = enable_gbnf
        self.gbnf_max_entities: int | None = gbnf_max_entities
        self.gbnf_max_triples: int | None = gbnf_max_triples
        self.gbnf_max_facts: int = gbnf_max_facts
        self.gbnf_max_reasoning_chars: int | None = gbnf_max_reasoning_chars
        self.embedding_query_prefix: str | None = embedding_query_prefix
        self.embedding_fact_prefix: str | None = embedding_fact_prefix
        self.embedding_document_prefix: str | None = embedding_document_prefix
        self.json_adapter: dspy.Adapter = adapter or YawJSONAdapter()
        self.orchestrator_lm: dspy.LM | None = orchestrator_lm
        self.rlm_sub_lm: dspy.LM | None = rlm_sub_lm
        self._retrieval_hook: Callable[..., tuple[str, tuple[str, ...], tuple[float, ...]]] | None = None
        self._node_inspector: Callable[[str], dict[str, Any]] | None = None
        self._chunk_hydrator: Callable[[str], str] | None = None
        self._shacl_repair_applier: Callable[[Sequence[Mapping[str, Any]]], int] | None = None
        self._shacl_linter: Callable[[], list[dict[str, str]]] | None = None
        self._last_ner_definitions: dict[str, str] = {}
        self.telemetry: list[dict[str, Any]] = []
        self.embedding_audit_trail: list[dict[str, Any]] = []
        self.entity_def_audit_trail: list[dict[str, Any]] = []
        self.synonym_audit_trail: list[dict[str, Any]] = []
        self.shacl_audit_trail: list[dict[str, Any]] = []

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
            setattr(self.ner, "demos", list(NER_DEMOS))
            setattr(self.triples, "demos", list(TRIPLE_DEMOS))
            setattr(self.triple_filter, "demos", list(TRIPLE_FILTER_DEMOS))
            setattr(self.qa, "demos", list(QA_DEMOS))
            if hasattr(self.qa, "predict"):
                setattr(self.qa.predict, "demos", list(QA_DEMOS))
            rlm_demos = _build_rlm_action_demos()
            setattr(
                self.shacl_healer_rlm.generate_action,
                "demos",
                list(rlm_demos["shacl_repair"]),
            )

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
        setattr(self.qa, "lm", qa_lm)
        setattr(self.qa, "signature", self.qa.predict.signature)

        action_lm = _PredictorBoundLM(
            "shacl_healer_rlm.generate_action", "orchestrator", orch_getter
        )
        extract_lm = _PredictorBoundLM(
            "shacl_healer_rlm.extract", "orchestrator", orch_getter
        )
        self.shacl_healer_rlm.generate_action.lm = action_lm
        self.shacl_healer_rlm.extract.lm = extract_lm
        setattr(self.shacl_healer_rlm, "lm", action_lm)

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
            ("qa", self.qa),
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

    @classmethod
    def from_config(cls, config: YawConfig) -> "HippoRAGProgram":
        """Construct a HippoRAGProgram from a YawConfig."""
        orch_lm = create_dspy_lm(
            config.qa_model,
            base_url=config.base_url,
            api_key=config.api_key,
            provider=config.provider,
            role="qa",
        )
        sub_lm = create_dspy_lm(
            config.extraction_model,
            base_url=config.base_url,
            api_key=config.api_key,
            provider=config.provider,
            role="extraction",
        )
        return cls(
            enable_gbnf=config.enable_gbnf,
            gbnf_max_entities=config.gbnf_max_entities,
            gbnf_max_triples=config.gbnf_max_triples,
            gbnf_max_facts=config.gbnf_max_facts,
            gbnf_max_reasoning_chars=config.gbnf_max_reasoning_chars,
            orchestrator_lm=orch_lm,
            rlm_sub_lm=sub_lm,
            compiled_path=config.compiled_program_path,
        )

    def dump_state(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Serialize all DSPy predictors/RLMs plus EmbeddingGemma prefixes and GBNF structural bounds."""
        state: dict[str, Any] = super().dump_state(*args, **kwargs)
        state["yaw_surface"] = {
            "gbnf_max_entities": self.gbnf_max_entities,
            "gbnf_max_triples": self.gbnf_max_triples,
            "gbnf_max_facts": self.gbnf_max_facts,
            "gbnf_max_reasoning_chars": self.gbnf_max_reasoning_chars,
            "embedding_query_prefix": self.embedding_query_prefix,
            "embedding_fact_prefix": self.embedding_fact_prefix,
            "embedding_document_prefix": self.embedding_document_prefix,
        }
        return state

    def load_state(self, state: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        """Restore DSPy predictor/RLM state plus EmbeddingGemma prefixes and GBNF structural bounds."""
        state_copy = dict(state)
        yaw_surface = state_copy.pop("yaw_surface", None)
        res = super().load_state(state_copy, *args, **kwargs)
        if isinstance(yaw_surface, dict):
            for attr in (
                "gbnf_max_entities",
                "gbnf_max_triples",
                "gbnf_max_reasoning_chars",
                "embedding_query_prefix",
                "embedding_fact_prefix",
                "embedding_document_prefix",
            ):
                if attr in yaw_surface:
                    setattr(self, attr, yaw_surface[attr])
            if yaw_surface.get("gbnf_max_facts") is not None:
                self.gbnf_max_facts = int(yaw_surface["gbnf_max_facts"])
        self._bind_predictor_lms()
        if hasattr(self.qa, "predict") and not getattr(self.qa.predict, "demos", None):
            self.qa.predict.demos = list(QA_DEMOS)
        return res

    def __deepcopy__(self, memo: dict[int, Any]) -> "HippoRAGProgram":
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
                "orchestrator_lm",
                "rlm_sub_lm",
                "json_adapter",
            ):
                setattr(new_prog, k, v)
            else:
                setattr(new_prog, k, copy.deepcopy(v, memo))
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

        with track_usage() as usage_tracker:
            if is_rlm and rlm_mod is not None:
                orch_lm = self._resolve_orchestrator_lm(lm)
                with self._dspy_ctx(orch_lm, use_json_adapter=False):
                    pred = rlm_mod(**inputs)
                active_mod = rlm_mod
                active_name = "shacl_healer_rlm"
            else:
                effective_lm = lm
                if effective_lm is None:
                    if direct_mod is self.qa:
                        effective_lm = self.orchestrator_lm or self.rlm_sub_lm
                    else:
                        effective_lm = self.rlm_sub_lm or self.orchestrator_lm
                direct_inputs = dict(inputs)
                if isinstance(direct_inputs.get("passage"), str) and len(direct_inputs["passage"]) > 24000:
                    direct_inputs["passage"] = direct_inputs["passage"][:24000]
                if isinstance(direct_inputs.get("context"), str) and len(direct_inputs["context"]) > 40000:
                    direct_inputs["context"] = direct_inputs["context"][:40000]
                call_kwargs: dict[str, Any] = {}
                if self._should_use_gbnf(use_gbnf, effective_lm):
                    call_kwargs["config"] = {"extra_body": {"grammar": grammar_fn()}}
                with self._dspy_ctx(effective_lm, use_json_adapter=True):
                    pred = direct_mod(**direct_inputs, **call_kwargs)
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
        raw_defs = getattr(
            pred, "entity_definitions", getattr(pred, "named_entities", [])
        )
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
            defn = (
                exact_map.get(m_name)
                or norm_map.get(str(text_processing(m_name)))
                or ""
            )
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
        defined_entity_names: list[str] = []
        for k, v in self._last_ner_definitions.items():
            if k and v:
                merged_defs[k] = v
                merged_defs[str(text_processing(k))] = v
        if isinstance(entity_definitions, Mapping):
            for k, v in entity_definitions.items():
                ck = str(k).strip()
                cv = str(v).strip()
                if ck and cv:
                    merged_defs[ck] = cv
                    merged_defs[str(text_processing(ck))] = cv
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
                        if ck not in defined_entity_names:
                            defined_entity_names.append(ck)
        for item in named_entities:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                k = str(item[0]).strip()
                v = str(item[1]).strip() if item[1] is not None else ""
                if k and v:
                    merged_defs[k] = v
                    merged_defs[str(text_processing(k))] = v
                    if k not in defined_entity_names:
                        defined_entity_names.append(k)

        known_entities_for_grammar: list[str] = defined_entity_names or [
            (
                str(x[0]).strip()
                if isinstance(x, (list, tuple)) and len(x) >= 1
                else str(x).strip()
            )
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
            extract_structured_triples(raw_triples, entity_definitions=merged_defs)
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
        entities, entity_definitions = self._run_ner_stage(
            passage, lm=lm, use_gbnf=use_gbnf
        )
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
            str(text_processing(name))
            for name in merged_entities
            if str(text_processing(name))
        }
        def_by_name: dict[str, str] = {name: defn for name, defn in entity_definitions}
        def_by_norm: dict[str, str] = {
            str(text_processing(name)): defn
            for name, defn in entity_definitions
            if defn and str(text_processing(name))
        }
        ordered_def_names: list[str] = [name for name, _ in entity_definitions]

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
                if ent_name not in ordered_def_names and (
                    not ent_norm
                    or all(str(text_processing(n)) != ent_norm for n in ordered_def_names)
                ):
                    ordered_def_names.append(ent_name)
                if ent_def:
                    if not def_by_name.get(ent_name):
                        def_by_name[ent_name] = ent_def
                    if ent_norm and not def_by_norm.get(ent_norm):
                        def_by_norm[ent_norm] = ent_def

        merged_def_list: list[tuple[str, str]] = [
            (
                name,
                def_by_name.get(name, "")
                or def_by_norm.get(str(text_processing(name)), ""),
            )
            for name in ordered_def_names
        ]

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
            if len(pairs_json) > 12000:
                try:
                    parsed_pairs = json.loads(pairs_json)
                    if isinstance(parsed_pairs, list):
                        pairs_json = json.dumps(parsed_pairs[:16], ensure_ascii=False)
                except (json.JSONDecodeError, TypeError):
                    pairs_json = pairs_json[:12000]
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
            if not is_rlm and len(violations_json) > 12000:
                try:
                    parsed_v = json.loads(violations_json)
                    if isinstance(parsed_v, list):
                        violations_json = json.dumps(parsed_v[:16], ensure_ascii=False)
                except (json.JSONDecodeError, TypeError):
                    violations_json = violations_json[:12000]
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
                    HealSHACLViolations.__name__
                    if is_rlm
                    else ProposeSHACLRepairs.__name__
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
        hook: Callable[..., tuple[str, tuple[str, ...], tuple[float, ...]]] | None,
    ) -> None:
        """Register an optional end-to-end retrieval hook executed inside forward(question=...)."""
        self._retrieval_hook = hook

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
            return self.answer_question(
                context=context, question=question, use_gbnf=use_gbnf
            )
        if context is not None and question is None:
            return self.forward(
                context=None,
                question=context,
                item_docs=item_docs,
                use_gbnf=use_gbnf,
            )
        if question is not None:
            if self._retrieval_hook is not None:
                with track_usage() as usage_tracker:
                    try:
                        sig = inspect.signature(self._retrieval_hook)
                        param_names = list(sig.parameters.keys())
                        accepts_prog = len(param_names) >= 2
                        accepts_item_docs = "item_docs" in sig.parameters or len(param_names) >= 3
                    except (TypeError, ValueError):
                        accepts_prog = False
                        accepts_item_docs = False
                    if accepts_item_docs:
                        hydrated_ctx, passages, scores = self._retrieval_hook(
                            question, self, item_docs=item_docs
                        )
                    elif accepts_prog:
                        hydrated_ctx, passages, scores = self._retrieval_hook(question, self)
                    else:
                        hydrated_ctx, passages, scores = self._retrieval_hook(question)
                    pred = self.answer_question(
                        context=hydrated_ctx, question=question, use_gbnf=use_gbnf
                    )
                tokens = usage_tracker.get_total_tokens()
                res = dspy.Prediction(
                    context=hydrated_ctx,
                    answer=getattr(pred, "answer", ""),
                    reasoning=getattr(pred, "reasoning", ""),
                    retrieved_passages=passages,
                    retrieved_scores=scores,
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

    def named_predictors(self) -> list[tuple[str, Any]]:
        """Return all 9 canonical stage predictors so `dspy.GEPA` inspects the full program surface."""
        return list(super().named_predictors())



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


def _anonymize_feedback_literals(feedback: str) -> str:
    """Replace specific gold passage and answer literals in diagnostic feedback with abstract placeholders."""
    if not feedback:
        return feedback
    out = re.sub(
        r"Missed passage:\s*(?:'[^']*'|\"[^\"]*\")",
        "Missed passage: '<TargetPassage>'",
        feedback,
    )
    out = re.sub(
        r"Expected=\[[^\]]*\]",
        "Expected=['<TargetAnswer>']",
        out,
    )
    out = re.sub(
        r"gold_answers=\[[^\]]*\]",
        "gold_answers=['<TargetAnswer>']",
        out,
    )
    out = re.sub(
        r"Missed gold items:\s*\[[^\]]*\]",
        "Missed gold items: ['<TargetEntity>']",
        out,
    )
    return out


def _example_failure_priority(ex: Mapping[str, Any], index: int) -> tuple[int, float, int]:
    """Rank reflective examples so the worst-scoring failing examples come strictly before passing ones."""
    fb = str(ex.get("Feedback", "") or "")
    score_match = re.search(r"GM_good/GM_bad=([0-9]+(?:\.[0-9]+)?)", fb)
    score_val = float(score_match.group(1)) if score_match else 0.5
    is_fail = (
        any(
            marker in fb
            for marker in (
                "Below reference floor",
                "Missed",
                "!=",
                "EM=0.0000",
                "violations_remaining",
                "repaired ",
            )
        )
        or ("missed=" in fb and "all_retrieved" not in fb)
    )
    if not is_fail and "all_retrieved" in fb and "EM=1.0000" in fb:
        return (1, score_val, index)
    return (0 if is_fail else 1, score_val if is_fail else 1.0 + score_val, index)


def _bound_nested_strings(value: Any, max_chars: int) -> Any:
    """Bound oversized leaf strings inside reflective examples while keeping structure intact."""
    if isinstance(value, str):
        if len(value) > max_chars:
            cut = len(value) - max_chars
            return f"{value[:max_chars]} [{cut:,} chars omitted]"
        return value
    if isinstance(value, Mapping):
        return {str(k): _bound_nested_strings(v, max_chars) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_bound_nested_strings(v, max_chars) for v in value]
    return value


def _extract_example_literal_phrases(examples: Any, current_instruction: str) -> list[str]:
    """Extract specific input/output literal phrases from reflection examples to detect instance leakage."""
    cur_lower = current_instruction.lower()
    phrases: list[str] = []
    items = examples if isinstance(examples, Sequence) and not isinstance(examples, str) else []
    for ex in items:
        if not isinstance(ex, Mapping):
            continue
        for section_key in ("Inputs", "Generated Outputs"):
            sec = ex.get(section_key)
            if not isinstance(sec, Mapping):
                continue
            for val in sec.values():
                text = str(val)
                for match in re.findall(r"\b[A-Z][A-Za-z0-9]+(?:\s+(?:of|the|and|[A-Z0-9][A-Za-z0-9]*)){1,3}\b", text):
                    cand = match.strip()
                    if len(cand) >= 8 and cand.lower() not in cur_lower:
                        phrases.append(cand)
                for match in re.findall(r"\b\d{1,2}\s+[A-Z][a-z]+\s+\d{3,4}\b", text):
                    cand = match.strip()
                    if cand.lower() not in cur_lower:
                        phrases.append(cand)
    return phrases


def _score_proposer_prediction(example: dspy.Example, prediction: Any) -> float:
    """Meta-metric for `dspy.SIMBA` evaluating the RLM inner worker (`qwen_proposer`) instruction proposals."""
    comp_name = str(getattr(example, "component_name", "") or "")
    cur_inst = str(getattr(example, "current_instruction", "") or "")
    ex_fb = getattr(example, "examples_with_feedback", ())
    new_inst = str(getattr(prediction, "new_instruction", "") or "").strip()

    if not new_inst:
        return 0.0

    schema_ok = True
    if comp_name in _STAGE_SCHEMA_CONTRACTS:
        strict_key, alt_markers, _ = _STAGE_SCHEMA_CONTRACTS[comp_name]
        had_strict = strict_key in cur_inst
        has_strict = strict_key in new_inst
        has_any = any(m.lower() in new_inst.lower() for m in alt_markers)
        schema_ok = bool(has_strict or (not had_strict and has_any))

    leaked_phrases = _extract_example_literal_phrases(ex_fb, cur_inst)
    new_lower = new_inst.lower()
    no_leakage = not any(p.lower() in new_lower for p in leaked_phrases)

    concise_ok = (
        len(new_inst) >= 35
        and len(new_inst) <= 1500
        and new_inst.strip() != cur_inst.strip()
    )

    score = (
        (0.35 if schema_ok else 0.05)
        + (0.35 if no_leakage else 0.0)
        + (0.30 if concise_ok else 0.05)
    )
    return round(float(score), 4)



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
        return self._rollout_lm or getattr(self.qwen_proposer, "lm", None) or self.worker_lm or self.orchestrator_lm

    def set_lm(self, lm: Any) -> None:
        self._rollout_lm = lm

    def __deepcopy__(self, memo: dict[int, Any]) -> "_InstructionSynthesizerWorker":
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
        from dspy.utils.usage_tracker import track_usage

        _ = component_name
        eff_worker = (
            self.worker_lm or self.orchestrator_lm or getattr(dspy.settings, "lm", None)
        )
        parsed_examples: list[dict[str, Any]]
        if isinstance(examples_with_feedback, str):
            try:
                loaded = json.loads(examples_with_feedback)
                parsed_examples = (
                    [dict(x) for x in loaded if isinstance(x, Mapping)]
                    if isinstance(loaded, list)
                    else [{"Feedback": examples_with_feedback}]
                )
            except (json.JSONDecodeError, TypeError, ValueError):
                parsed_examples = [{"Feedback": examples_with_feedback}]
        elif isinstance(examples_with_feedback, Sequence):
            parsed_examples = [
                dict(x) if isinstance(x, Mapping) else {"Feedback": str(x)}
                for x in examples_with_feedback
            ]
        else:
            parsed_examples = [{"Feedback": str(examples_with_feedback)}]

        bounded_examples = _bound_nested_strings(parsed_examples, self.max_field_chars)

        rollout_kwargs = getattr(self._rollout_lm, "kwargs", {}) if self._rollout_lm is not None else {}
        temp = float(rollout_kwargs.get("temperature", self.worker_temperature))
        call_config: dict[str, Any] = {"temperature": temp}
        if "rollout_id" in rollout_kwargs:
            call_config["rollout_id"] = rollout_kwargs["rollout_id"]
        if self.enable_gbnf and lm_supports_gbnf(eff_worker):
            call_config["extra_body"] = {
                "grammar": DSPY_INSTRUCTION_PROPOSAL_GBNF_GRAMMAR
            }
        ctx_kwargs: dict[str, Any] = {"adapter": self.json_adapter}
        if eff_worker is not None:
            ctx_kwargs["lm"] = eff_worker
        with dspy.context(**ctx_kwargs):
            with track_usage() as prop_tracker:
                pred = self.qwen_proposer(
                    current_instruction=str(current_instruction),
                    examples_with_feedback=bounded_examples,
                    config=call_config,
                )
            prop_usage = prop_tracker.get_total_tokens()
            draft = strip_mistral_bracket_prefix(
                str(getattr(pred, "new_instruction", "") or "")
            ).strip()
            self.sub_telemetry.append(
                {
                    "predictor": "qwen_proposer",
                    "signature": ProposeInstructionSignature.__name__,
                    "mode": "direct",
                    "tokens_used": _sum_usage_tokens(prop_usage),
                    "usage": dict(prop_usage or {}),
                    "inputs_summary": str(current_instruction)[:120],
                    "outputs_summary": draft[:120],
                }
            )
            if (
                draft
                and self.max_instruction_chars is not None
                and len(draft) > self.max_instruction_chars
            ):
                compress_config: dict[str, Any] = {
                    "temperature": self.worker_temperature
                }
                if self.enable_gbnf and lm_supports_gbnf(eff_worker):
                    compress_config["extra_body"] = {
                        "grammar": build_dspy_instruction_proposal_grammar(
                            self.max_instruction_chars
                        )
                    }
                with track_usage() as comp_tracker:
                    comp_pred = self.qwen_compressor(
                        instruction=draft,
                        length_limit=(
                            f"The new instruction must be at most {self.max_instruction_chars} characters."
                        ),
                        config=compress_config,
                    )
                comp_usage = comp_tracker.get_total_tokens()
                compressed = strip_mistral_bracket_prefix(
                    str(getattr(comp_pred, "new_instruction", "") or "")
                ).strip()
                self.sub_telemetry.append(
                    {
                        "predictor": "qwen_compressor",
                        "signature": CompressInstructionSignature.__name__,
                        "mode": "direct",
                        "tokens_used": _sum_usage_tokens(comp_usage),
                        "usage": dict(comp_usage or {}),
                        "inputs_summary": draft[:120],
                        "outputs_summary": compressed[:120],
                    }
                )
                if compressed:
                    draft = compressed

        active_trace = getattr(dspy.settings, "trace", None)
        if isinstance(active_trace, list) and not any(
            t[0] is self.qwen_proposer for t in active_trace
        ):
            active_trace.append(
                (
                    self.qwen_proposer,
                    {
                        "current_instruction": str(current_instruction),
                        "examples_with_feedback": bounded_examples,
                    },
                    pred,
                )
            )

        result_inst = draft if draft else str(current_instruction).strip()
        return dspy.Prediction(new_instruction=result_inst)


class RLMInstructionProposer:
    """Single-level RLM + SIMBA-Optimized GBNF Synthesizer (`Qwen3-4B` GPU) GEPA ProposalFn.

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
        rlm_max_iters: int = 3,
        rlm_max_llm_calls: int = 6,
        rlm_max_output_chars: int = 10000,
        include_demos: bool = True,
        adapter: dspy.Adapter | None = None,
        rlm_interpreter_factory: Callable[[], Any] = dspy.PythonInterpreter,
    ) -> None:
        if max_instruction_chars is not None and max_instruction_chars <= 0:
            raise ValueError(
                f"max_instruction_chars must be > 0 or None, got {max_instruction_chars!r}."
            )

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
        self.max_feedback_examples = max(1, int(max_feedback_examples))
        self.reflection_temperature = float(reflection_temperature)
        self.worker_temperature = float(worker_temperature)
        self.json_adapter: dspy.Adapter = adapter or YawJSONAdapter()
        self.telemetry: list[dict[str, Any]] = []
        self.simba_telemetry: list[dict[str, Any]] = []
        self._simba_examples: list[dspy.Example] = []
        self._sub_telemetry: list[dict[str, Any]] = []
        self._synthesized_proposals: list[str] = []

        self.qwen_proposer = dspy.Predict(ProposeInstructionSignature)
        self.qwen_compressor = dspy.Predict(CompressInstructionSignature)
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

        def synthesize_instruction(
            current_instruction: str,
            examples_with_feedback: Any,
        ) -> str:
            """Synthesize an improved instruction using the SIMBA-optimized inner worker on Qwen GPU."""
            self.worker_module.sub_telemetry.clear()
            pred = self.worker_module(
                current_instruction=current_instruction,
                examples_with_feedback=examples_with_feedback,
            )
            self._sub_telemetry.extend(self.worker_module.sub_telemetry)
            result_inst = str(getattr(pred, "new_instruction", "") or "").strip()
            if not result_inst:
                result_inst = str(current_instruction).strip()
            self._synthesized_proposals.append(result_inst)
            return result_inst

        self._synthesize_tool = synthesize_instruction
        self.inner_rlm = _PositionalSafeRLM(
            ReflectiveInnerDistillerSignature,
            stage_rule=_STAGE_RLM_DELEGATION_RULES["instruction_proposer_inner"],
            max_iters=rlm_max_iters,
            max_llm_calls=rlm_max_llm_calls,
            max_output_chars=rlm_max_output_chars,
            tools=[synthesize_instruction],
            sub_lm=worker_lm or orchestrator_lm,
            interpreter_factory=rlm_interpreter_factory,
        )
        self.inner_rlm.generate_action.config = {"temperature": self.reflection_temperature}
        self.inner_rlm.extract.config = {"temperature": self.reflection_temperature}
        self.rlm = self.inner_rlm

        self._bind_predictor_lms()

        if include_demos:
            rlm_demos = _build_rlm_action_demos()
            setattr(
                self.inner_rlm.generate_action,
                "demos",
                list(rlm_demos["instruction_proposer_inner"]),
            )

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

        inner_action_lm = _PredictorBoundLM("inner_rlm.generate_action", "orchestrator", orch_getter)
        inner_extract_lm = _PredictorBoundLM("inner_rlm.extract", "orchestrator", orch_getter)
        self.inner_rlm.generate_action.lm = inner_action_lm
        self.inner_rlm.extract.lm = inner_extract_lm
        setattr(self.inner_rlm, "lm", inner_action_lm)

    def get_lm(self) -> Any:
        return self.worker_module.get_lm()

    def set_lm(self, lm: Any) -> None:
        self.worker_module.set_lm(lm)

    def __deepcopy__(self, memo: dict[int, Any]) -> "RLMInstructionProposer":
        import copy

        cls = self.__class__
        new_prop = cls.__new__(cls)
        memo[id(self)] = new_prop
        for k, v in self.__dict__.items():
            if k in ("orchestrator_lm", "worker_lm", "json_adapter", "_synthesize_tool"):
                setattr(new_prop, k, v)
            else:
                setattr(new_prop, k, copy.deepcopy(v, memo))

        def synthesize_instruction(
            current_instruction: str,
            examples_with_feedback: Any,
        ) -> str:
            new_prop.worker_module.sub_telemetry.clear()
            pred = new_prop.worker_module(
                current_instruction=current_instruction,
                examples_with_feedback=examples_with_feedback,
            )
            new_prop._sub_telemetry.extend(new_prop.worker_module.sub_telemetry)
            result_inst = str(getattr(pred, "new_instruction", "") or "").strip()
            if not result_inst:
                result_inst = str(current_instruction).strip()
            new_prop._synthesized_proposals.append(result_inst)
            return result_inst

        new_prop._synthesize_tool = synthesize_instruction
        new_prop._bind_predictor_lms()
        return new_prop

    def all_reflection_predictors(self) -> list[tuple[str, Any]]:
        """Return all 3 active reflection predictors (`inner_rlm`, `qwen_proposer`, `qwen_compressor`)."""
        return [
            ("inner_rlm", self.inner_rlm),
            ("qwen_proposer", self.qwen_proposer),
            ("qwen_compressor", self.qwen_compressor),
        ]

    def step_simba(self) -> bool:
        """Run `dspy.SIMBA` on the inner worker (`self.worker_module`) over buffered reflection examples."""
        if not self.enable_simba or len(self._simba_examples) < self.simba_bsize:
            return False
        if not hasattr(dspy, "SIMBA"):
            return False

        eff_worker = (
            self.worker_lm or self.orchestrator_lm or getattr(dspy.settings, "lm", None)
        )
        if eff_worker is None:
            return False

        bsize = min(self.simba_bsize, len(self._simba_examples))
        trainset = list(self._simba_examples[-max(bsize, 8) :])
        prev_inst = str(getattr(self.qwen_proposer.signature, "instructions", "") or "")
        prev_demos = len(getattr(self.qwen_proposer, "demos", ()) or ())


        try:
            simba = dspy.SIMBA(
                metric=_score_proposer_prediction,
                bsize=bsize,
                num_candidates=self.simba_num_candidates,
                max_steps=self.simba_max_steps,
                max_demos=self.simba_max_demos,
                prompt_model=eff_worker,
                num_threads=1,
            )
            with dspy.context(lm=eff_worker, adapter=self.json_adapter):
                compiled_worker = simba.compile(
                    self.worker_module,
                    trainset=trainset,
                    seed=len(self.simba_telemetry),
                )
            new_inst = str(compiled_worker.qwen_proposer.signature.instructions).strip()
            if (
                self.max_instruction_chars is not None
                and len(new_inst) > self.max_instruction_chars
            ):
                new_inst = new_inst[: self.max_instruction_chars].rstrip()
                compiled_worker.qwen_proposer.signature = (
                    compiled_worker.qwen_proposer.signature.with_instructions(new_inst)
                )
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
            return True
        except (
            RuntimeError,
            ValueError,
            TypeError,
            StopIteration,
            KeyError,
            IndexError,
            AssertionError,
        ) as exc:
            logging.getLogger(__name__).debug(
                "SIMBA inner-worker step skipped (%s: %s)",
                type(exc).__name__,
                exc,
            )
            return False


    @staticmethod
    def _prepare_examples(
        examples: Any,
        *,
        max_examples: int = 5,
    ) -> list[dict[str, Any]]:
        """Prepare reflective examples, prioritizing top-5 failing examples and anonymizing gold literals."""
        from dspy.adapters.types import History
        from dspy.primitives.repl_types import REPLHistory
        from dspy.teleprompt.gepa.instruction_proposal import (
            compact_history,
            compact_repl_history,
        )

        def prepare(value: Any, *, is_feedback: bool = False) -> Any:
            if isinstance(value, History):
                return compact_history(value)
            if isinstance(value, REPLHistory):
                return compact_repl_history(value)
            if isinstance(value, Mapping):
                return {
                    str(k): prepare(v, is_feedback=(str(k) == "Feedback"))
                    for k, v in value.items()
                }
            if isinstance(value, (list, tuple)):
                return [prepare(v, is_feedback=is_feedback) for v in value]
            if is_feedback and isinstance(value, str):
                return _anonymize_feedback_literals(value)
            return value

        raw_list: list[dict[str, Any]] = []
        if isinstance(examples, str):
            try:
                loaded = json.loads(examples)
                if isinstance(loaded, list):
                    raw_list = [dict(x) for x in loaded if isinstance(x, Mapping)]
                else:
                    raw_list = [{"Feedback": examples}]
            except (json.JSONDecodeError, TypeError, ValueError):
                raw_list = [{"Feedback": examples}]
        elif isinstance(examples, Sequence):
            raw_list = [
                dict(ex) if isinstance(ex, Mapping) else {"Feedback": str(ex)}
                for ex in examples
            ]

        if not raw_list:
            return []

        indexed = list(enumerate(raw_list))
        indexed.sort(key=lambda pair: _example_failure_priority(pair[1], pair[0]))
        limit = max(1, int(max_examples))
        selected = [ex for _, ex in indexed[:limit]]
        return [prepare(ex) for ex in selected]

    def _ensure_stage_schema_contract(
        self,
        component_name: str,
        current_instruction: str,
        draft: str,
    ) -> str:
        """Preserve the component's JSON schema contract when omitted by the proposer."""
        if not draft or component_name not in _STAGE_SCHEMA_CONTRACTS:
            return draft
        strict_key, alt_markers, schema_clause = _STAGE_SCHEMA_CONTRACTS[component_name]
        draft_lower = draft.lower()
        had_strict = strict_key in current_instruction
        has_strict = strict_key in draft
        has_any_marker = any(m.lower() in draft_lower for m in alt_markers)
        if has_strict or (not had_strict and has_any_marker):
            return draft

        combined = f"{draft.rstrip()} {schema_clause}"
        if (
            self.max_instruction_chars is not None
            and len(combined) > self.max_instruction_chars
        ):
            budget = max(32, self.max_instruction_chars - len(schema_clause) - 1)
            combined = f"{draft[:budget].rstrip()} {schema_clause}"
        return combined

    def propose_for_component(
        self,
        component_name: str,
        current_instruction: str,
        component_examples: Sequence[Mapping[str, Any]],
    ) -> str:
        """Run single-level RLM trace distillation -> SIMBA-evolved Qwen GBNF synthesis for a predictor component."""
        from dspy.utils.usage_tracker import track_usage

        prepared_examples = self._prepare_examples(
            component_examples,
            max_examples=self.max_feedback_examples,
        )
        eff_orch = (
            self.orchestrator_lm or self.worker_lm or getattr(dspy.settings, "lm", None)
        )
        self._synthesized_proposals.clear()
        self._sub_telemetry.clear()

        with track_usage() as usage_tracker:
            if eff_orch is not None:
                with dspy.context(lm=eff_orch), track_usage() as rlm_tracker:
                    rlm_pred = self.inner_rlm(
                        current_instruction=current_instruction,
                        examples_with_feedback=prepared_examples,
                    )
                rlm_usage = rlm_tracker.get_total_tokens()
                rlm_inst = strip_mistral_bracket_prefix(
                    str(getattr(rlm_pred, "new_instruction", "") or "")
                ).strip()
                self._sub_telemetry.append(
                    {
                        "predictor": "inner_rlm",
                        "signature": ReflectiveInnerDistillerSignature.__name__,
                        "mode": "rlm",
                        "tokens_used": _sum_usage_tokens(rlm_usage),
                        "usage": dict(rlm_usage or {}),
                        "inputs_summary": current_instruction[:120],
                        "outputs_summary": rlm_inst[:120],
                    }
                )
                if self._synthesized_proposals:
                    proposed = self._synthesized_proposals[-1]
                else:
                    proposed = self._synthesize_tool(
                        current_instruction,
                        prepared_examples,
                    )
            else:
                proposed = self._synthesize_tool(
                    current_instruction,
                    prepared_examples,
                )
        usage = usage_tracker.get_total_tokens()

        final_instruction = proposed.strip() if proposed else current_instruction
        final_instruction = self._ensure_stage_schema_contract(
            component_name,
            current_instruction,
            final_instruction,
        )
        self.telemetry.append(
            {
                "component_name": component_name,
                "predictor": "inner_rlm",
                "signature": ReflectiveInnerDistillerSignature.__name__,
                "mode": "rlm",
                "tokens_used": _sum_usage_tokens(usage),
                "previous_instruction": current_instruction,
                "new_instruction": final_instruction,
                "num_feedback_examples": len(prepared_examples),
                "usage": dict(usage or {}),
                "sub_telemetry": list(self._sub_telemetry),
                "simba_steps": len(self.simba_telemetry),
            }
        )
        return final_instruction

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
            prepared = self._prepare_examples(
                examples_with_feedback,
                max_examples=self.max_feedback_examples,
            )
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

        for comp_name in components_to_update:
            if comp_name not in candidate:
                continue
            comp_examples = reflective_dataset.get(comp_name)
            if not comp_examples:
                continue
            prepared = self._prepare_examples(
                comp_examples,
                max_examples=self.max_feedback_examples,
            )
            if prepared:
                self._simba_examples.append(
                    dspy.Example(
                        component_name=comp_name,
                        current_instruction=candidate[comp_name],
                        examples_with_feedback=prepared,
                    ).with_inputs(
                        "component_name",
                        "current_instruction",
                        "examples_with_feedback",
                    )
                )

        if self.enable_simba and len(self._simba_examples) >= self.simba_bsize:
            self.step_simba()

        updated: dict[str, str] = {}
        for comp_name in components_to_update:
            if comp_name not in candidate:
                continue
            comp_examples = reflective_dataset.get(comp_name)
            if not comp_examples:
                continue
            updated[comp_name] = self.propose_for_component(
                component_name=comp_name,
                current_instruction=candidate[comp_name],
                component_examples=comp_examples,
            )
        return updated

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)


__all__ = [
    "HippoRAGProgram",
    "RLMInstructionProposer",
]
