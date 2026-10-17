"""Native DSPy HippoRAGProgram and _PositionalSafeRLM orchestrator."""

import functools
import inspect
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import dspy

from yaw.core.config import YawConfig
from yaw.core.types import OpenIERecord

from .chat import create_dspy_lm, create_dspy_rlm_pair
from .grammars import (
    DSPY_INSTRUCTION_PROPOSAL_GBNF_GRAMMAR,
    build_dspy_instruction_proposal_grammar,
    build_dspy_ner_grammar,
    build_dspy_qa_cot_grammar,
    build_dspy_query_intent_grammar,
    build_dspy_rerank_grammar,
    build_dspy_triples_grammar,
    lm_supports_gbnf,
)
from .parsers import (
    coerce_generated_facts,
    extract_named_entities,
    extract_triples,
    filter_invalid_triples,
    match_facts_to_candidate_indices,
    strip_mistral_bracket_prefix,
)
from .prompts import (
    FACT_FILTER_DEMOS,
    NER_DEMOS,
    QA_DEMOS,
    TRIPLE_DEMOS,
    AnswerQuestion,
    CompressInstructionSignature,
    ExtractEntities,
    ExtractQueryIntent,
    ExtractTriples,
    FilterFacts,
    ProposeInstructionSignature,
    ReflectiveInstructionTask,
)

_STAGE_RLM_DELEGATION_RULES: dict[str, str] = {
    "ner": (
        "CRITICAL RLM SANDBOX RULE:\n"
        "Do NOT import `nltk` or `spacy`. Delegate directly to `qwen_extract_entities` and call `SUBMIT` in iteration 1:\n"
        "```python\n"
        "entities = qwen_extract_entities(passage=passage)\n"
        "SUBMIT(named_entities=entities)\n"
        "```"
    ),
    "triples": (
        "CRITICAL RLM SANDBOX RULE:\n"
        "Do NOT call `qwen_extract_entities` or loop per entity. Pass `passage` and `named_entities` directly to "
        "`qwen_extract_triples` once and call `SUBMIT` in iteration 1:\n"
        "```python\n"
        "triples = qwen_extract_triples(passage=passage, named_entities=named_entities)\n"
        "SUBMIT(triples=triples)\n"
        "```"
    ),
    "fact_filter": (
        "CRITICAL RLM SANDBOX RULE:\n"
        "Do NOT filter facts manually in Python. Delegate directly to `qwen_filter_facts` and call `SUBMIT` in iteration 1:\n"
        "```python\n"
        "kept = qwen_filter_facts(question=question, fact_before_filter=fact_before_filter)\n"
        "SUBMIT(fact=kept)\n"
        "```"
    ),
    "qa": (
        "CRITICAL RLM SANDBOX RULE:\n"
        "Delegate directly to `qwen_answer_cot` and call `SUBMIT` with `res['answer']` in iteration 1:\n"
        "```python\n"
        "res = qwen_answer_cot(context=context, question=question)\n"
        "SUBMIT(answer=res[\"answer\"])\n"
        "```"
    ),
    "instruction_proposer": (
        "CRITICAL RLM SANDBOX RULE:\n"
        "Do NOT import `qwen_propose_instruction` (`qwen_propose_instruction` and `SUBMIT` are already in global scope).\n"
        "Inspect `examples_with_feedback` in Python, call `qwen_propose_instruction`, and `SUBMIT`:\n"
        "```python\n"
        "improved = qwen_propose_instruction(current_instruction, examples_with_feedback)\n"
        "SUBMIT(new_instruction=improved)\n"
        "```"
    ),
    "instruction_proposer_inner": (
        "CRITICAL RLM SANDBOX RULE:\n"
        "Do NOT import `synthesize_instruction` (`synthesize_instruction` and `SUBMIT` are already in global scope).\n"
        "Distill `examples_with_feedback` in Python so large inputs fit in context, call `synthesize_instruction`, and `SUBMIT`:\n"
        "```python\n"
        "digest = [{'Inputs': {k: (v[:400] if isinstance(v, str) else v) for k, v in ex.get('Inputs', {}).items()}, 'Generated Outputs': ex.get('Generated Outputs', {}), 'Feedback': ex.get('Feedback', '')} for ex in examples_with_feedback]\n"
        "res = synthesize_instruction(current_instruction, digest)\n"
        "SUBMIT(new_instruction=res)\n"
        "```"
    ),
}


def _build_rlm_action_demos() -> dict[str, list[dspy.Example]]:
    from dspy.primitives.repl_types import REPLHistory

    empty_history = REPLHistory()
    return {
        "ner": [
            dspy.Example(
                variables_info="Variable: `passage` (type: str): Source passage text",
                repl_history=empty_history,
                iteration="1/3",
                reasoning="Delegate named entity extraction to the Qwen worker and submit.",
                code="```python\nentities = qwen_extract_entities(passage=passage)\nSUBMIT(named_entities=entities)\n```",
            ).with_inputs("variables_info", "repl_history", "iteration")
        ],
        "triples": [
            dspy.Example(
                variables_info=(
                    "Variable: `passage` (type: str): Source passage text\n"
                    "Variable: `named_entities` (type: list): Named entity list extracted from the passage"
                ),
                repl_history=empty_history,
                iteration="1/3",
                reasoning="Delegate RDF triple extraction to the Qwen worker and submit.",
                code="```python\ntriples = qwen_extract_triples(passage=passage, named_entities=named_entities)\nSUBMIT(triples=triples)\n```",
            ).with_inputs("variables_info", "repl_history", "iteration")
        ],
        "fact_filter": [
            dspy.Example(
                variables_info=(
                    "Variable: `question` (type: str): Query for retrieval\n"
                    "Variable: `fact_before_filter` (type: str): Candidate facts to be filtered"
                ),
                repl_history=empty_history,
                iteration="1/3",
                reasoning="Delegate candidate fact filtering to the Qwen worker and submit.",
                code="```python\nkept = qwen_filter_facts(question=question, fact_before_filter=fact_before_filter)\nSUBMIT(fact=kept)\n```",
            ).with_inputs("variables_info", "repl_history", "iteration")
        ],
        "qa": [
            dspy.Example(
                variables_info=(
                    "Variable: `context` (type: str): Retrieved evidence passages\n"
                    "Variable: `question` (type: str): Question to answer"
                ),
                repl_history=empty_history,
                iteration="1/3",
                reasoning="Delegate reading comprehension to the Qwen worker and submit.",
                code='```python\nres = qwen_answer_cot(context=context, question=question)\nSUBMIT(answer=res["answer"])\n```',
            ).with_inputs("variables_info", "repl_history", "iteration")
        ],
        "instruction_proposer": [
            dspy.Example(
                variables_info=(
                    "Variable: `current_instruction` (type: str): The instructions I provided to the assistant.\n"
                    "Variable: `examples_with_feedback` (type: list): Task inputs provided to the assistant, the assistant's response for each, and feedback on how the response could be better."
                ),
                repl_history=empty_history,
                iteration="1/3",
                reasoning="Inspect the task examples and evaluation feedback and delegate to the inner Qwen RLM.",
                code=(
                    "```python\n"
                    "improved = qwen_propose_instruction(current_instruction, examples_with_feedback)\n"
                    "SUBMIT(new_instruction=improved)\n"
                    "```"
                ),
            ).with_inputs("variables_info", "repl_history", "iteration")
        ],
        "instruction_proposer_inner": [
            dspy.Example(
                variables_info=(
                    "Variable: `current_instruction` (type: str): The instructions I provided to the assistant.\n"
                    "Variable: `examples_with_feedback` (type: list): Task inputs provided to the assistant, the assistant's response for each, and feedback on how the response could be better."
                ),
                repl_history=empty_history,
                iteration="1/3",
                reasoning="Distill the examples and feedback in Python and call synthesize_instruction.",
                code=(
                    "```python\n"
                    "digest = [{'Inputs': {k: (v[:400] if isinstance(v, str) else v) for k, v in ex.get('Inputs', {}).items()}, 'Generated Outputs': ex.get('Generated Outputs', {}), 'Feedback': ex.get('Feedback', '')} for ex in examples_with_feedback]\n"
                    "res = synthesize_instruction(current_instruction, digest)\n"
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
        try:
            loaded = json.loads(val)
        except (json.JSONDecodeError, TypeError, ValueError):
            loaded = None
        if isinstance(loaded, list):
            return [str(x) for x in loaded if isinstance(x, str)]
        return list(extract_named_entities(val))
    if name == "triples":
        if isinstance(val, str):
            try:
                loaded = json.loads(val)
            except (json.JSONDecodeError, TypeError, ValueError):
                loaded = None
            if isinstance(loaded, list):
                return [list(t) for t in filter_invalid_triples(loaded)]
            return [list(t) for t in extract_triples(val)]
        if isinstance(val, (list, tuple)):
            return [list(t) for t in filter_invalid_triples(val)]
        return val
    if name == "fact":
        return coerce_generated_facts(val)
    if annotation is str and isinstance(val, str):
        return strip_mistral_bracket_prefix(val)
    return val


def _sanitize_repl_code(code: str) -> str:
    """Strip hallucinated imports of pre-bound sandbox tools (`qwen_*`, `synthesize_instruction`, `SUBMIT`, `llm_query`)."""
    cleaned_lines: list[str] = []
    for line in code.splitlines():
        stripped = line.strip()
        if (
            (stripped.startswith("from ") or stripped.startswith("import "))
            and (
                "qwen_" in stripped
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
        pred = super()._extract_fallback(variables, history, output_field_names)
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
                    if len(fields) == 1:
                        v = next(iter(fields.values()))
                        return {target_field: parse_value(v, ann)}
                    if "answer" in fields:
                        return {target_field: parse_value(fields["answer"], ann)}
            raise


class HippoRAGProgram(dspy.Module):
    """Unified DSPy Module for HippoRAG 2 OpenIE, Recognition Memory, and Multi-Hop QA.

    Supports two execution styles across **all** stages (`ner`, `triples`, `fact_filter`, `qa`):
    1. **Heterogeneous Mistral Orchestrator + Qwen RLM Worker (`execution_mode="rlm"`):**
       - Outer `dspy.RLM` runs on `orchestrator_lm` (`Mistral-7B-Instruct-v0.3-Hybrid` on
         Ryzen AI NPU/OGA, `ChatAdapter`, **never** GBNF) inside `dspy.LocalInterpreter`.
       - Inner `llm_query` and built-in `qwen_*` sandbox tools (`qwen_extract_entities`,
         `qwen_extract_triples`, `qwen_filter_facts`, `qwen_answer_cot`) execute on `rlm_sub_lm`
         (`Qwen3-4B-Instruct-2507-GGUF` on `llama.cpp` Vulkan GPU) using `dspy.JSONAdapter()`
         and **pure-JSON GBNF grammars** (`extra_body={"grammar": ...}`).
    2. **Direct Single-Call (`execution_mode="direct"`):**
       - Runs `dspy.Predict` / `dspy.ChainOfThought` directly with `dspy.JSONAdapter()` and
         attaches GBNF grammars only when `lm_supports_gbnf(lm)` is `True`.
    """

    def __init__(
        self,
        *,
        execution_mode: Literal["direct", "rlm"] = "direct",
        qa_mode: Literal["cot", "rlm"] | None = None,
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
        rlm_interpreter_factory: Callable[[], Any] = dspy.LocalInterpreter,
        compiled_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        if execution_mode not in ("direct", "rlm"):
            raise ValueError(
                f"Unsupported execution_mode: {execution_mode!r}. Expected 'direct' or 'rlm'."
            )
        resolved_qa_mode: Literal["cot", "rlm"] = (
            qa_mode if qa_mode is not None else ("rlm" if execution_mode == "rlm" else "cot")
        )
        if resolved_qa_mode not in ("cot", "rlm"):
            raise ValueError(
                f"Unsupported qa_mode: {resolved_qa_mode!r}. Expected 'cot' or 'rlm'."
            )
        self.execution_mode: Literal["direct", "rlm"] = execution_mode
        self.qa_mode: Literal["cot", "rlm"] = resolved_qa_mode
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

        self.ner = dspy.Predict(ExtractEntities)
        self.triples = dspy.Predict(ExtractTriples)
        self.fact_filter = dspy.Predict(FilterFacts)
        self.query_intent = dspy.Predict(ExtractQueryIntent)
        self.qa_cot = dspy.ChainOfThought(AnswerQuestion)

        (
            qwen_ner_tool,
            qwen_triples_tool,
            qwen_filter_tool,
            qwen_qa_tool,
        ) = self._build_qwen_worker_tools()

        def _make_stage_rlm(signature: Any, stage_key: str, primary: Callable[..., Any]) -> _PositionalSafeRLM:
            tools = [primary]
            if rlm_tools:
                existing = {getattr(primary, "__name__", "")}
                for user_tool in rlm_tools:
                    name = getattr(user_tool, "__name__", "")
                    if name not in existing:
                        existing.add(name)
                        tools.append(user_tool)
            return _PositionalSafeRLM(
                signature,
                stage_rule=_STAGE_RLM_DELEGATION_RULES[stage_key],
                max_iters=rlm_max_iters,
                max_llm_calls=rlm_max_llm_calls,
                max_output_chars=rlm_max_output_chars,
                tools=tools,
                sub_lm=rlm_sub_lm,
                interpreter_factory=rlm_interpreter_factory,
            )

        self.ner_rlm = _make_stage_rlm(ExtractEntities, "ner", qwen_ner_tool)
        self.triples_rlm = _make_stage_rlm(ExtractTriples, "triples", qwen_triples_tool)
        self.fact_filter_rlm = _make_stage_rlm(FilterFacts, "fact_filter", qwen_filter_tool)
        self.qa_rlm = _make_stage_rlm(AnswerQuestion, "qa", qwen_qa_tool)

        if include_demos:
            setattr(self.ner, "demos", list(NER_DEMOS))
            setattr(self.triples, "demos", list(TRIPLE_DEMOS))
            setattr(self.fact_filter, "demos", list(FACT_FILTER_DEMOS))
            setattr(self.qa_cot, "demos", list(QA_DEMOS))
            rlm_demos = _build_rlm_action_demos()
            setattr(self.ner_rlm.generate_action, "demos", list(rlm_demos["ner"]))
            setattr(self.triples_rlm.generate_action, "demos", list(rlm_demos["triples"]))
            setattr(self.fact_filter_rlm.generate_action, "demos", list(rlm_demos["fact_filter"]))
            setattr(self.qa_rlm.generate_action, "demos", list(rlm_demos["qa"]))

        if compiled_path is not None:
            path = Path(compiled_path)
            if path.exists():
                self.load(str(path))

    @classmethod
    def from_config(cls, config: YawConfig) -> "HippoRAGProgram":
        """Construct a HippoRAGProgram (including heterogeneous or direct LMs) from a YawConfig."""
        if config.execution_mode == "rlm":
            orch_lm, sub_lm = create_dspy_rlm_pair(
                orchestrator_model=config.qa_model,
                sub_model=config.extraction_model,
                base_url=config.base_url,
                api_key=config.api_key,
                provider=config.provider,
            )
        else:
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
            execution_mode=config.execution_mode,
            qa_mode=config.qa_mode,
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
        return res

    def __deepcopy__(self, memo: dict[int, Any]) -> "HippoRAGProgram":
        """Clone all DSPy predictors while preserving live LM clients and the engine retrieval hook."""
        import copy

        cls = self.__class__
        new_prog = cls.__new__(cls)
        memo[id(self)] = new_prog
        for k, v in self.__dict__.items():
            if k in ("_retrieval_hook", "orchestrator_lm", "rlm_sub_lm", "json_adapter"):
                setattr(new_prog, k, v)
            else:
                setattr(new_prog, k, copy.deepcopy(v, memo))
        return new_prog

    def _resolve_worker_lm(self, fallback_lm: dspy.LM | None = None) -> dspy.LM | None:
        return self.rlm_sub_lm if self.rlm_sub_lm is not None else fallback_lm

    def _resolve_orchestrator_lm(self, fallback_lm: dspy.LM | None = None) -> dspy.LM | None:
        return fallback_lm if fallback_lm is not None else self.orchestrator_lm

    def _build_qwen_worker_tools(self) -> list[Callable[..., Any]]:
        """Build GBNF-constrained Qwen (`llama.cpp` GPU) worker tools exposed inside Mistral's RLM REPL."""

        def qwen_extract_entities(passage: str) -> list[str]:
            """Extract deduplicated named entities from a passage using the GBNF-constrained Qwen GPU model."""
            worker_lm = self._resolve_worker_lm()
            return list(
                self.extract_entities(
                    passage,
                    mode="direct",
                    lm=worker_lm,
                    use_gbnf=self.enable_gbnf,
                )
            )

        def qwen_extract_triples(passage: str, named_entities: list[str]) -> list[list[str]]:
            """Extract [subject, predicate, object] triples using the GBNF-constrained Qwen GPU model."""
            worker_lm = self._resolve_worker_lm()
            triples = self.extract_triples(
                passage,
                named_entities,
                mode="direct",
                lm=worker_lm,
                use_gbnf=self.enable_gbnf,
            )
            return [list(t) for t in triples]

        def qwen_filter_facts(
            question: str,
            fact_before_filter: Any = None,
            candidate_facts: list[list[str]] | None = None,
        ) -> list[list[str]]:
            """Filter candidate [s, p, o] facts using the dynamic candidate-literal GBNF Qwen GPU model."""
            raw_source = candidate_facts if candidate_facts is not None else fact_before_filter
            parsed_candidates = coerce_generated_facts(raw_source)
            if not parsed_candidates:
                return []
            worker_lm = self._resolve_worker_lm()
            indices = self.filter_facts(
                question,
                parsed_candidates,
                mode="direct",
                lm=worker_lm,
                use_gbnf=self.enable_gbnf,
            )
            return [
                list(parsed_candidates[i])
                for i in indices
                if 0 <= i < len(parsed_candidates)
            ]

        def qwen_answer_cot(context: str, question: str) -> dict[str, str]:
            """Run GBNF-constrained Chain-of-Thought QA on the Qwen GPU model and return {reasoning, answer}."""
            worker_lm = self._resolve_worker_lm()
            pred = self.answer_question(
                context,
                question,
                mode="cot",
                lm=worker_lm,
                use_gbnf=self.enable_gbnf,
            )
            return {
                "reasoning": strip_mistral_bracket_prefix(str(getattr(pred, "reasoning", ""))),
                "answer": strip_mistral_bracket_prefix(str(getattr(pred, "answer", ""))),
            }

        return [
            qwen_extract_entities,
            qwen_extract_triples,
            qwen_filter_facts,
            qwen_answer_cot,
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

    def _invoke_stage(
        self,
        direct_mod: Any,
        rlm_mod: Any,
        *,
        is_rlm: bool,
        lm: Any,
        use_gbnf: bool | None,
        grammar_fn: Callable[[], str],
        **inputs: Any,
    ) -> Any:
        if is_rlm:
            orch_lm = self._resolve_orchestrator_lm(lm)
            with self._dspy_ctx(orch_lm, use_json_adapter=False):
                return rlm_mod(**inputs)
        effective_lm = lm
        if effective_lm is None:
            if direct_mod is self.qa_cot:
                effective_lm = self.orchestrator_lm
            else:
                effective_lm = self.rlm_sub_lm or self.orchestrator_lm
        call_kwargs: dict[str, Any] = {}
        if self._should_use_gbnf(use_gbnf, effective_lm):
            call_kwargs["config"] = {"extra_body": {"grammar": grammar_fn()}}
        with self._dspy_ctx(effective_lm, use_json_adapter=True):
            return direct_mod(**inputs, **call_kwargs)

    def extract_entities(
        self,
        passage: str,
        *,
        mode: Literal["direct", "rlm"] | None = None,
        lm: Any = None,
        use_gbnf: bool | None = None,
    ) -> tuple[str, ...]:
        """Run DSPy NER extraction (direct on Qwen w/ GBNF, or Mistral RLM orchestrating Qwen)."""
        pred = self._invoke_stage(
            self.ner,
            self.ner_rlm,
            is_rlm=(mode or self.execution_mode) == "rlm",
            lm=lm,
            use_gbnf=use_gbnf,
            grammar_fn=lambda: build_dspy_ner_grammar(self.gbnf_max_entities),
            passage=passage,
        )
        raw_entities = getattr(pred, "named_entities", [])
        if isinstance(raw_entities, str):
            return extract_named_entities(raw_entities)
        if not isinstance(raw_entities, (list, tuple)):
            return ()
        unique: list[str] = []
        seen: set[str] = set()
        for item in raw_entities:
            if isinstance(item, str):
                clean = item.strip()
                if clean and clean not in seen:
                    seen.add(clean)
                    unique.append(clean)
        return tuple(unique)

    def extract_triples(
        self,
        passage: str,
        named_entities: Sequence[str],
        *,
        mode: Literal["direct", "rlm"] | None = None,
        lm: Any = None,
        use_gbnf: bool | None = None,
    ) -> tuple[tuple[str, str, str], ...]:
        """Run DSPy triple extraction (direct on Qwen w/ GBNF, or Mistral RLM orchestrating Qwen)."""
        pred = self._invoke_stage(
            self.triples,
            self.triples_rlm,
            is_rlm=(mode or self.execution_mode) == "rlm",
            lm=lm,
            use_gbnf=use_gbnf,
            grammar_fn=lambda: build_dspy_triples_grammar(self.gbnf_max_triples),
            passage=passage,
            named_entities=list(named_entities),
        )
        raw_triples = getattr(pred, "triples", [])
        if isinstance(raw_triples, str):
            return extract_triples(raw_triples)
        if not isinstance(raw_triples, (list, tuple)):
            return ()
        filtered = filter_invalid_triples(raw_triples)
        return tuple((t[0], t[1], t[2]) for t in filtered)

    def extract_openie(
        self,
        chunk_id: str,
        passage: str,
        *,
        mode: Literal["direct", "rlm"] | None = None,
        lm: Any = None,
        use_gbnf: bool | None = None,
    ) -> OpenIERecord:
        """Execute two-stage OpenIE (NER -> Triples) for a passage chunk."""
        entities = self.extract_entities(
            passage, mode=mode, lm=lm, use_gbnf=use_gbnf
        )
        triples = self.extract_triples(
            passage, entities, mode=mode, lm=lm, use_gbnf=use_gbnf
        )
        return OpenIERecord(
            chunk_id=chunk_id,
            entities=entities,
            triples=triples,
        )

    def filter_facts(
        self,
        question: str,
        candidate_facts: Sequence[tuple[str, str, str] | list[str]],
        *,
        mode: Literal["direct", "rlm"] | None = None,
        max_facts: int | None = None,
        lm: Any = None,
        use_gbnf: bool | None = None,
    ) -> list[int]:
        """Filter candidate facts (direct on Qwen w/ GBNF, or Mistral RLM orchestrating Qwen)."""
        if not candidate_facts:
            return []
        effective_max_facts = self.gbnf_max_facts if max_facts is None else max_facts
        fact_before_filter = json.dumps({"fact": [list(f) for f in candidate_facts]})
        pred = self._invoke_stage(
            self.fact_filter,
            self.fact_filter_rlm,
            is_rlm=(mode or self.execution_mode) == "rlm",
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
        effective_lm = lm or self.rlm_sub_lm or self.orchestrator_lm
        call_cfg: dict[str, Any] = {
            "temperature": 0.4,
            "frequency_penalty": 0.2,
        }
        if self._should_use_gbnf(use_gbnf, effective_lm):
            call_cfg["extra_body"] = {"grammar": build_dspy_query_intent_grammar()}
        with self._dspy_ctx(effective_lm, use_json_adapter=True):
            pred = self.query_intent(question=question, config=call_cfg)
        return str(getattr(pred, "query_intent", "") or "").strip() or question

    def answer_question(
        self,
        context: str,
        question: str,
        *,
        mode: Literal["cot", "rlm"] | None = None,
        lm: Any = None,
        use_gbnf: bool | None = None,
    ) -> dspy.Prediction:
        """Answer a question over retrieved context using either ChainOfThought or RLM."""
        from dspy.utils.usage_tracker import track_usage

        active_mode = mode or self.qa_mode
        if active_mode not in ("cot", "rlm"):
            raise ValueError(f"Unsupported QA mode: {active_mode!r}")
        with track_usage() as usage_tracker:
            pred = self._invoke_stage(
                self.qa_cot,
                self.qa_rlm,
                is_rlm=active_mode == "rlm",
                lm=lm,
                use_gbnf=use_gbnf,
                grammar_fn=lambda: build_dspy_qa_cot_grammar(self.gbnf_max_reasoning_chars),
                context=context,
                question=question,
            )
        tokens = usage_tracker.get_total_tokens()
        if tokens and hasattr(pred, "set_lm_usage"):
            pred.set_lm_usage(tokens)
        return pred

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
        mode: Literal["cot", "rlm"] | None = None,
        use_gbnf: bool | None = None,
    ) -> dspy.Prediction:
        """Default DSPy forward pass executes multi-hop QA or the full retrieval+QA trace."""
        from dspy.utils.usage_tracker import track_usage

        if context is not None and question is not None:
            return self.answer_question(
                context=context, question=question, mode=mode, use_gbnf=use_gbnf
            )
        if context is not None and question is None:
            return self.forward(context=None, question=context, mode=mode, use_gbnf=use_gbnf)
        if question is not None:
            if self._retrieval_hook is not None:
                with track_usage() as usage_tracker:
                    try:
                        sig = inspect.signature(self._retrieval_hook)
                        accepts_prog = len(sig.parameters) >= 2
                    except (TypeError, ValueError):
                        accepts_prog = False
                    if accepts_prog:
                        hydrated_ctx, passages, scores = self._retrieval_hook(question, self)
                    else:
                        hydrated_ctx, passages, scores = self._retrieval_hook(question)
                    pred = self.answer_question(
                        context=hydrated_ctx, question=question, mode=mode, use_gbnf=use_gbnf
                    )
                tokens = usage_tracker.get_total_tokens()
                if tokens and hasattr(pred, "set_lm_usage"):
                    pred.set_lm_usage(tokens)
                setattr(pred, "context", hydrated_ctx)
                setattr(pred, "retrieved_passages", passages)
                setattr(pred, "retrieved_scores", scores)
                return pred
            raise ValueError(
                "context must be provided when no retrieval_hook is registered on HippoRAGProgram."
            )
        raise ValueError(
            "Either (context, question) or question must be provided to HippoRAGProgram."
        )

    def named_predictors(self) -> list[tuple[str, Any]]:
        """Return only predictors active under the current execution_mode, qa_mode, and retrieval_hook."""
        inactive_prefixes: set[str] = set()
        if self.execution_mode != "rlm":
            inactive_prefixes.update({"ner_rlm", "triples_rlm", "fact_filter_rlm"})
        if self.qa_mode != "rlm":
            inactive_prefixes.add("qa_rlm")
        if self._retrieval_hook is not None:
            inactive_prefixes.update({"ner", "triples", "ner_rlm", "triples_rlm"})
        return [
            (name, param)
            for name, param in super().named_predictors()
            if name.split(".", 1)[0] not in inactive_prefixes
        ]


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


class RLMInstructionProposer:
    """Heterogeneous `RLM -> RLM` Cascade (`Mistral-7B` NPU $\\to$ `Qwen3-4B` GPU) GEPA ProposalFn.

    Conforms to `gepa.core.adapter.ProposalFn` and stock DSPy `InstructionProposer`:
    `(candidate, reflective_dataset, components_to_update) -> dict[str, str]`.
    - **Outer RLM (`self.outer_rlm` on `orchestrator_lm`, `Mistral-7B`, NPU):**
      Holds the full untruncated `examples_with_feedback` list in Python memory and delegates
      to `qwen_propose_instruction`.
    - **Inner RLM (`self.inner_rlm` on `worker_lm`, `Qwen3-4B`, GPU):**
      Distills `examples_with_feedback` in Python so neither model's context window is flooded,
      then calls `synthesize_instruction` (backed by stock DSPy's `ProposeInstructionSignature`
      and `CompressInstructionSignature` with `YawJSONAdapter` and pure-JSON GBNF grammars).
    """

    def __init__(
        self,
        *,
        orchestrator_lm: Any = None,
        worker_lm: Any = None,
        enable_gbnf: bool = True,
        max_instruction_chars: int | None = 1500,
        max_field_chars: int = 600,
        reflection_temperature: float = 0.8,
        worker_temperature: float = 0.7,
        rlm_max_iters: int = 3,
        rlm_max_llm_calls: int = 6,
        rlm_max_output_chars: int = 10000,
        include_demos: bool = True,
        adapter: dspy.Adapter | None = None,
        rlm_interpreter_factory: Callable[[], Any] = dspy.LocalInterpreter,
    ) -> None:
        if max_instruction_chars is not None and max_instruction_chars <= 0:
            raise ValueError(
                f"max_instruction_chars must be > 0 or None, got {max_instruction_chars!r}."
            )
        self.orchestrator_lm = orchestrator_lm
        self.worker_lm = worker_lm
        self.enable_gbnf = bool(enable_gbnf)
        self.max_instruction_chars = (
            int(max_instruction_chars) if max_instruction_chars is not None else None
        )
        self.max_field_chars = max(120, int(max_field_chars))
        self.reflection_temperature = float(reflection_temperature)
        self.worker_temperature = float(worker_temperature)
        self.json_adapter: dspy.Adapter = adapter or YawJSONAdapter()
        self.telemetry: list[dict[str, Any]] = []
        self._synthesized_proposals: list[str] = []

        self.qwen_proposer = dspy.Predict(ProposeInstructionSignature)
        self.qwen_compressor = dspy.Predict(CompressInstructionSignature)

        def synthesize_instruction(
            current_instruction: str,
            examples_with_feedback: Any,
        ) -> str:
            """Synthesize an improved instruction using stock DSPy ProposeInstruction + GBNF on Qwen GPU."""
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

            call_config: dict[str, Any] = {"temperature": self.worker_temperature}
            if self.enable_gbnf and lm_supports_gbnf(eff_worker):
                call_config["extra_body"] = {
                    "grammar": DSPY_INSTRUCTION_PROPOSAL_GBNF_GRAMMAR
                }
            ctx_kwargs: dict[str, Any] = {"adapter": self.json_adapter}
            if eff_worker is not None:
                ctx_kwargs["lm"] = eff_worker
            with dspy.context(**ctx_kwargs):
                pred = self.qwen_proposer(
                    current_instruction=str(current_instruction),
                    examples_with_feedback=bounded_examples,
                    config=call_config,
                )
                draft = strip_mistral_bracket_prefix(
                    str(getattr(pred, "new_instruction", "") or "")
                ).strip()
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
                    comp_pred = self.qwen_compressor(
                        instruction=draft,
                        length_limit=(
                            f"The new instruction must be at most {self.max_instruction_chars} characters."
                        ),
                        config=compress_config,
                    )
                    compressed = strip_mistral_bracket_prefix(
                        str(getattr(comp_pred, "new_instruction", "") or "")
                    ).strip()
                    if compressed:
                        draft = compressed

            result_inst = draft if draft else str(current_instruction).strip()
            self._synthesized_proposals.append(result_inst)
            return result_inst

        self._synthesize_tool = synthesize_instruction
        self.inner_rlm = _PositionalSafeRLM(
            ReflectiveInstructionTask,
            stage_rule=_STAGE_RLM_DELEGATION_RULES["instruction_proposer_inner"],
            max_iters=rlm_max_iters,
            max_llm_calls=rlm_max_llm_calls,
            max_output_chars=rlm_max_output_chars,
            tools=[synthesize_instruction],
            sub_lm=worker_lm,
            interpreter_factory=rlm_interpreter_factory,
        )
        self.inner_rlm.generate_action.config = {"temperature": self.worker_temperature}
        self.inner_rlm.extract.config = {"temperature": self.worker_temperature}

        def qwen_propose_instruction(
            arg1: Any,
            arg2: Any = None,
            arg3: Any = None,
            arg4: Any = None,
        ) -> str:
            """Execute the Inner Qwen RLM over examples_with_feedback and synthesize the new instruction."""
            if arg3 is not None:
                curr_inst = str(arg2)
                raw_examples = arg3
            else:
                curr_inst = str(arg1)
                raw_examples = arg2
            prepared = self._prepare_examples(raw_examples)
            eff_worker = (
                self.worker_lm or self.orchestrator_lm or getattr(dspy.settings, "lm", None)
            )
            before_count = len(self._synthesized_proposals)
            with dspy.context(lm=eff_worker):
                inner_pred = self.inner_rlm(
                    current_instruction=curr_inst,
                    examples_with_feedback=prepared,
                )
            if len(self._synthesized_proposals) > before_count:
                return self._synthesized_proposals[-1]
            inner_inst = strip_mistral_bracket_prefix(
                str(getattr(inner_pred, "new_instruction", "") or "")
            ).strip()
            if inner_inst and inner_inst != curr_inst.strip():
                self._synthesized_proposals.append(inner_inst)
                return inner_inst
            return synthesize_instruction(curr_inst, prepared)

        self._qwen_propose_tool = qwen_propose_instruction
        self.outer_rlm = _PositionalSafeRLM(
            ReflectiveInstructionTask,
            stage_rule=_STAGE_RLM_DELEGATION_RULES["instruction_proposer"],
            max_iters=rlm_max_iters,
            max_llm_calls=rlm_max_llm_calls,
            max_output_chars=rlm_max_output_chars,
            tools=[qwen_propose_instruction],
            sub_lm=worker_lm,
            interpreter_factory=rlm_interpreter_factory,
        )
        self.outer_rlm.generate_action.config = {"temperature": self.reflection_temperature}
        self.outer_rlm.extract.config = {"temperature": self.reflection_temperature}
        self.rlm = self.outer_rlm

        if include_demos:
            rlm_demos = _build_rlm_action_demos()
            setattr(
                self.outer_rlm.generate_action,
                "demos",
                list(rlm_demos["instruction_proposer"]),
            )
            setattr(
                self.inner_rlm.generate_action,
                "demos",
                list(rlm_demos["instruction_proposer_inner"]),
            )

    @staticmethod
    def _prepare_examples(examples: Any) -> list[dict[str, Any]]:
        """Prepare reflective examples following stock DSPy InstructionProposer._prepare_examples."""
        from dspy.adapters.types import History
        from dspy.primitives.repl_types import REPLHistory
        from dspy.teleprompt.gepa.instruction_proposal import (
            compact_history,
            compact_repl_history,
        )

        def prepare(value: Any) -> Any:
            if isinstance(value, History):
                return compact_history(value)
            if isinstance(value, REPLHistory):
                return compact_repl_history(value)
            if isinstance(value, Mapping):
                return {str(k): prepare(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [prepare(v) for v in value]
            return value

        if isinstance(examples, str):
            try:
                loaded = json.loads(examples)
                if isinstance(loaded, list):
                    return [prepare(dict(x)) for x in loaded if isinstance(x, Mapping)]
            except (json.JSONDecodeError, TypeError, ValueError):
                return [{"Feedback": examples}]
        if isinstance(examples, Sequence):
            return [
                prepare(dict(ex)) if isinstance(ex, Mapping) else {"Feedback": str(ex)}
                for ex in examples
            ]
        return []

    def propose_for_component(
        self,
        component_name: str,
        current_instruction: str,
        component_examples: Sequence[Mapping[str, Any]],
    ) -> str:
        """Run Outer Mistral RLM -> Inner Qwen RLM -> Qwen GBNF synthesis for a predictor component."""
        prepared_examples = self._prepare_examples(component_examples)
        eff_orch = (
            self.orchestrator_lm or self.worker_lm or getattr(dspy.settings, "lm", None)
        )
        self._synthesized_proposals.clear()

        if eff_orch is not None and not lm_supports_gbnf(eff_orch):
            with dspy.context(lm=eff_orch):
                self.outer_rlm(
                    current_instruction=current_instruction,
                    examples_with_feedback=prepared_examples,
                )
            if self._synthesized_proposals:
                proposed = self._synthesized_proposals[-1]
            else:
                proposed = self._qwen_propose_tool(
                    current_instruction,
                    prepared_examples,
                )
        else:
            proposed = self._qwen_propose_tool(
                current_instruction,
                prepared_examples,
            )

        final_instruction = proposed.strip() if proposed else current_instruction
        self.telemetry.append(
            {
                "component_name": component_name,
                "previous_instruction": current_instruction,
                "new_instruction": final_instruction,
                "num_feedback_examples": len(component_examples),
            }
        )
        return final_instruction

    def __call__(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: Sequence[str],
    ) -> dict[str, str]:
        """ProposalFn entrypoint called by `dspy.GEPA` (`DspyAdapter.propose_new_texts`)."""
        if not components_to_update:
            return {}
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

    forward = __call__


__all__ = [
    "HippoRAGProgram",
    "RLMInstructionProposer",
]
