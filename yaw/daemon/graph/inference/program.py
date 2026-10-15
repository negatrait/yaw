"""Native DSPy HippoRAGProgram and _PositionalSafeRLM orchestrator."""

from collections.abc import Callable, Sequence
import functools
import inspect
import json
from pathlib import Path
from typing import Any, List, Literal, Tuple, Union

import dspy

from yaw.core.types import OpenIERecord

from .grammars import (
    build_dspy_ner_grammar,
    build_dspy_qa_cot_grammar,
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
    ExtractEntities,
    ExtractTriples,
    FilterFacts,
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
                reasoning="Delegate named entity extraction to the GBNF-constrained Qwen GPU worker and submit.",
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
                reasoning="Delegate RDF triple extraction to the GBNF-constrained Qwen GPU worker in a single call and submit.",
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
                reasoning="Delegate candidate fact filtering to the GBNF-constrained Qwen GPU worker and submit the selected facts.",
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
                reasoning="Delegate Chain-of-Thought reading comprehension to the GBNF-constrained Qwen GPU worker and submit its answer.",
                code='```python\nres = qwen_answer_cot(context=context, question=question)\nSUBMIT(answer=res["answer"])\n```',
            ).with_inputs("variables_info", "repl_history", "iteration")
        ],
    }


class _PositionalSafeRLM(dspy.RLM):
    """RLM subclass that binds positional tool arguments, enforces 1-step Qwen delegation, and normalizes SUBMIT payloads."""

    def __init__(self, *args: Any, stage_rule: str = "", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if stage_rule:
            updated_instructions = self._action_signature.instructions + "\n\n" + stage_rule
            self._action_signature = self._action_signature.with_instructions(updated_instructions)
            self.generate_action.signature = self._action_signature

    def _make_interpreter_tool(self, tool: dspy.Tool) -> Callable[..., Any]:
        sig = inspect.signature(tool.func)
        if inspect.iscoroutinefunction(tool.func) or inspect.iscoroutinefunction(
            getattr(tool.func, "__call__", None)
        ):
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
        invoke.__signature__ = sig
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
                if name not in raw_output:
                    continue
                val = raw_output[name]
                field = self.signature.output_fields[name]
                annotation = getattr(field, "annotation", str)
                if isinstance(val, dict):
                    if name in val:
                        val = val[name]
                    elif len(val) == 1:
                        val = next(iter(val.values()))
                if name == "named_entities" and isinstance(val, str):
                    try:
                        loaded = json.loads(val)
                    except Exception:
                        loaded = None
                    if isinstance(loaded, list):
                        val = [str(x) for x in loaded if isinstance(x, str)]
                    else:
                        val = list(extract_named_entities(val))
                elif name == "triples":
                    if isinstance(val, str):
                        try:
                            loaded = json.loads(val)
                        except Exception:
                            loaded = None
                        if isinstance(loaded, list):
                            val = [list(t) for t in filter_invalid_triples(loaded)]
                        else:
                            val = [list(t) for t in extract_triples(val)]
                    elif isinstance(val, (list, tuple)):
                        val = [list(t) for t in filter_invalid_triples(val)]
                elif name == "fact":
                    val = coerce_generated_facts(val)
                elif annotation is str and isinstance(val, str):
                    val = strip_mistral_bracket_prefix(val)
                raw_output[name] = val
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
        adapter: dspy.Adapter | None = None,
        orchestrator_lm: dspy.LM | None = None,
        rlm_max_iters: int = 10,
        rlm_max_llm_calls: int = 20,
        rlm_max_output_chars: int = 10000,
        rlm_sub_lm: dspy.LM | None = None,
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
        self.json_adapter: dspy.Adapter = adapter or dspy.JSONAdapter()
        self.orchestrator_lm: dspy.LM | None = orchestrator_lm
        self.rlm_sub_lm: dspy.LM | None = rlm_sub_lm

        self.ner = dspy.Predict(ExtractEntities)
        self.triples = dspy.Predict(ExtractTriples)
        self.fact_filter = dspy.Predict(FilterFacts)
        self.qa_cot = dspy.ChainOfThought(AnswerQuestion)

        (
            qwen_ner_tool,
            qwen_triples_tool,
            qwen_filter_tool,
            qwen_qa_tool,
        ) = self._build_qwen_worker_tools()

        def _stage_tools(primary: Callable[..., Any]) -> list[Callable[..., Any]]:
            tools = [primary]
            if rlm_tools:
                existing = {getattr(primary, "__name__", "")}
                for user_tool in rlm_tools:
                    name = getattr(user_tool, "__name__", "")
                    if name not in existing:
                        existing.add(name)
                        tools.append(user_tool)
            return tools

        self.ner_rlm = _PositionalSafeRLM(
            ExtractEntities,
            stage_rule=_STAGE_RLM_DELEGATION_RULES["ner"],
            max_iters=rlm_max_iters,
            max_llm_calls=rlm_max_llm_calls,
            max_output_chars=rlm_max_output_chars,
            tools=_stage_tools(qwen_ner_tool),
            sub_lm=rlm_sub_lm,
            interpreter_factory=rlm_interpreter_factory,
        )
        self.triples_rlm = _PositionalSafeRLM(
            ExtractTriples,
            stage_rule=_STAGE_RLM_DELEGATION_RULES["triples"],
            max_iters=rlm_max_iters,
            max_llm_calls=rlm_max_llm_calls,
            max_output_chars=rlm_max_output_chars,
            tools=_stage_tools(qwen_triples_tool),
            sub_lm=rlm_sub_lm,
            interpreter_factory=rlm_interpreter_factory,
        )
        self.fact_filter_rlm = _PositionalSafeRLM(
            FilterFacts,
            stage_rule=_STAGE_RLM_DELEGATION_RULES["fact_filter"],
            max_iters=rlm_max_iters,
            max_llm_calls=rlm_max_llm_calls,
            max_output_chars=rlm_max_output_chars,
            tools=_stage_tools(qwen_filter_tool),
            sub_lm=rlm_sub_lm,
            interpreter_factory=rlm_interpreter_factory,
        )
        self.qa_rlm = _PositionalSafeRLM(
            AnswerQuestion,
            stage_rule=_STAGE_RLM_DELEGATION_RULES["qa"],
            max_iters=rlm_max_iters,
            max_llm_calls=rlm_max_llm_calls,
            max_output_chars=rlm_max_output_chars,
            tools=_stage_tools(qwen_qa_tool),
            sub_lm=rlm_sub_lm,
            interpreter_factory=rlm_interpreter_factory,
        )

        if include_demos:
            self.ner.demos = list(NER_DEMOS)
            self.triples.demos = list(TRIPLE_DEMOS)
            self.fact_filter.demos = list(FACT_FILTER_DEMOS)
            self.qa_cot.demos = list(QA_DEMOS)
            rlm_demos = _build_rlm_action_demos()
            self.ner_rlm.generate_action.demos = list(rlm_demos["ner"])
            self.triples_rlm.generate_action.demos = list(rlm_demos["triples"])
            self.fact_filter_rlm.generate_action.demos = list(rlm_demos["fact_filter"])
            self.qa_rlm.generate_action.demos = list(rlm_demos["qa"])

        if compiled_path is not None:
            path = Path(compiled_path)
            if path.exists():
                self.load(str(path))

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

    def extract_entities(
        self,
        passage: str,
        *,
        mode: Literal["direct", "rlm"] | None = None,
        lm: dspy.LM | None = None,
        use_gbnf: bool | None = None,
    ) -> Tuple[str, ...]:
        """Run DSPy NER extraction (direct on Qwen w/ GBNF, or Mistral RLM orchestrating Qwen)."""
        active_mode = mode or self.execution_mode
        if active_mode == "rlm":
            orch_lm = self._resolve_orchestrator_lm(lm)
            with self._dspy_ctx(orch_lm, use_json_adapter=False):
                pred = self.ner_rlm(passage=passage)
        else:
            call_kwargs: dict[str, Any] = {}
            if self._should_use_gbnf(use_gbnf, lm):
                grammar = build_dspy_ner_grammar(self.gbnf_max_entities)
                call_kwargs["config"] = {"extra_body": {"grammar": grammar}}
            with self._dspy_ctx(lm, use_json_adapter=True):
                pred = self.ner(passage=passage, **call_kwargs)

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
        lm: dspy.LM | None = None,
        use_gbnf: bool | None = None,
    ) -> Tuple[Tuple[str, str, str], ...]:
        """Run DSPy triple extraction (direct on Qwen w/ GBNF, or Mistral RLM orchestrating Qwen)."""
        active_mode = mode or self.execution_mode
        if active_mode == "rlm":
            orch_lm = self._resolve_orchestrator_lm(lm)
            with self._dspy_ctx(orch_lm, use_json_adapter=False):
                pred = self.triples_rlm(
                    passage=passage,
                    named_entities=list(named_entities),
                )
        else:
            call_kwargs: dict[str, Any] = {}
            if self._should_use_gbnf(use_gbnf, lm):
                grammar = build_dspy_triples_grammar(self.gbnf_max_triples)
                call_kwargs["config"] = {"extra_body": {"grammar": grammar}}
            with self._dspy_ctx(lm, use_json_adapter=True):
                pred = self.triples(
                    passage=passage,
                    named_entities=list(named_entities),
                    **call_kwargs,
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
        lm: dspy.LM | None = None,
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
        candidate_facts: Sequence[Union[Tuple[str, str, str], List[str]]],
        *,
        mode: Literal["direct", "rlm"] | None = None,
        max_facts: int | None = None,
        lm: dspy.LM | None = None,
        use_gbnf: bool | None = None,
    ) -> List[int]:
        """Filter candidate facts (direct on Qwen w/ GBNF, or Mistral RLM orchestrating Qwen)."""
        if not candidate_facts:
            return []
        effective_max_facts = self.gbnf_max_facts if max_facts is None else max_facts
        fact_before_filter = json.dumps({"fact": [list(f) for f in candidate_facts]})
        active_mode = mode or self.execution_mode
        if active_mode == "rlm":
            orch_lm = self._resolve_orchestrator_lm(lm)
            with self._dspy_ctx(orch_lm, use_json_adapter=False):
                pred = self.fact_filter_rlm(
                    question=question,
                    fact_before_filter=fact_before_filter,
                )
        else:
            call_kwargs: dict[str, Any] = {}
            if self._should_use_gbnf(use_gbnf, lm):
                grammar = build_dspy_rerank_grammar(
                    candidate_facts, max_facts=effective_max_facts
                )
                call_kwargs["config"] = {"extra_body": {"grammar": grammar}}
            with self._dspy_ctx(lm, use_json_adapter=True):
                pred = self.fact_filter(
                    question=question,
                    fact_before_filter=fact_before_filter,
                    **call_kwargs,
                )
        raw_after = getattr(pred, "fact", getattr(pred, "fact_after_filter", None))
        generated = coerce_generated_facts(raw_after)
        return match_facts_to_candidate_indices(
            candidate_facts, generated, max_facts=effective_max_facts
        )

    def answer_question(
        self,
        context: str,
        question: str,
        *,
        mode: Literal["cot", "rlm"] | None = None,
        lm: dspy.LM | None = None,
        use_gbnf: bool | None = None,
    ) -> dspy.Prediction:
        """Answer a question over retrieved context using either ChainOfThought or RLM."""
        active_mode = mode or self.qa_mode
        if active_mode not in ("cot", "rlm"):
            raise ValueError(f"Unsupported QA mode: {active_mode!r}")
        if active_mode == "rlm":
            orch_lm = self._resolve_orchestrator_lm(lm)
            with self._dspy_ctx(orch_lm, use_json_adapter=False):
                return self.qa_rlm(context=context, question=question)

        call_kwargs: dict[str, Any] = {}
        if self._should_use_gbnf(use_gbnf, lm):
            grammar = build_dspy_qa_cot_grammar(self.gbnf_max_reasoning_chars)
            call_kwargs["config"] = {"extra_body": {"grammar": grammar}}
        with self._dspy_ctx(lm, use_json_adapter=True):
            return self.qa_cot(context=context, question=question, **call_kwargs)

    def forward(
        self,
        context: str,
        question: str,
        *,
        mode: Literal["cot", "rlm"] | None = None,
        use_gbnf: bool | None = None,
    ) -> dspy.Prediction:
        """Default DSPy forward pass executes multi-hop QA (CoT or RLM) for GEPA evaluation."""
        return self.answer_question(
            context=context, question=question, mode=mode, use_gbnf=use_gbnf
        )


__all__ = [
    "HippoRAGProgram",
]
