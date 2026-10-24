"""Native DSPy HippoRAGProgram orchestrator."""

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import dspy

from yaw.core.config import DEFAULT_CONTEXT_WINDOW_TOKENS, YawConfig, context_char_limit, scale_context_budget
from yaw.core.types import OpenIERecord, StructuredTriple

from .adapter import YawGBNFAdapter, sum_usage_tokens
from .chat import create_dspy_lm
from .diagnostics import DiagnosticHistory, diagnostic_boundary
from .grammars import (
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
from .invariants import is_abstaining_answer, should_retry_stage_invariants, validate_stage_invariants
from .parsers import (
    coerce_generated_facts,
    extract_entity_definitions,
    extract_structured_triples,
    match_facts_to_candidate_indices,
    parse_shacl_repairs,
    parse_synonym_decisions,
    text_processing,
)
from .prompts import (
    NER_DEMOS,
    QA_DEMOS,
    TRIPLE_DEMOS,
    TRIPLE_FILTER_DEMOS,
    AnswerQuestion,
    DefineMissingEntities,
    ExtractEntities,
    ExtractQueryIntent,
    ExtractTriples,
    FilterTriples,
    HealSHACLViolations,
    JudgeSynonyms,
    ProposeSHACLRepairs,
)
from .retriever import YawRetriever, YawRM
from .rlm import STAGE_RLM_DELEGATION_RULES, PositionalSafeRLM, build_rlm_action_demos


def _bounded_prompt_json(items: Sequence[Any], max_items: int, max_chars: int) -> str:
    """Fit whole structured input items; never split a JSON item to fit a prompt."""
    selected = list(items[:max_items])
    while selected:
        rendered = json.dumps(selected, ensure_ascii=False)
        if len(rendered) <= max_chars:
            return rendered
        selected.pop()
    return "[]"


def _merge_openie_definitions(
    entities: Sequence[str],
    entity_definitions: Sequence[tuple[str, str]],
    structured_triples: Sequence[StructuredTriple],
) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    """Merge NER entities, NER definitions, and structured-triple inline definitions without duplicates."""
    merged_entities: list[str] = list(entities)
    seen_entity_names: set[str] = set(merged_entities)
    seen_entity_norms: set[str] = {str(text_processing(name)) for name in merged_entities if str(text_processing(name))}
    raw_pairs: list[tuple[str, str]] = list(entity_definitions)
    for st in structured_triples:
        mention_pairs = (
            ((st.subj, st.subj_def),) if st.obj_kind == "literal" else ((st.subj, st.subj_def), (st.obj, st.obj_def))
        )
        for ent_name, ent_def in mention_pairs:
            if not ent_name:
                continue
            ent_norm = str(text_processing(ent_name))
            if ent_name not in seen_entity_names and (not ent_norm or ent_norm not in seen_entity_norms):
                merged_entities.append(ent_name)
                seen_entity_names.add(ent_name)
                if ent_norm:
                    seen_entity_norms.add(ent_norm)
            raw_pairs.append((ent_name, ent_def))

    return tuple(merged_entities), extract_entity_definitions(raw_pairs, normalize_keys=True)


class HippoRAGProgram(dspy.Module):
    """Unified DSPy Module for HippoRAG 2 OpenIE, EDC/SKOS Schema Canon, SHACL Repair, Recognition Memory, and \
Multi-Hop QA.

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
       - `shacl_healer_rlm` (`HealSHACLViolations`) running inside `dspy.PythonInterpreter` (managed \
Deno/Pyodide WASM) with
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
        include_demos: bool = False,
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
        rlm_max_output_chars: int | None = None,
        rlm_sub_lm: Any = None,
        rlm_tools: list[Callable[..., Any]] | None = None,
        rlm_interpreter_factory: Callable[[], Any] = dspy.PythonInterpreter,
        compiled_path: str | Path | None = None,
        enable_react_fallback: bool = False,
        context_window_tokens: int | None = None,
        max_output_tokens: int | None = None,
    ) -> None:
        super().__init__()
        if context_window_tokens is None:
            context_window_tokens = (
                getattr(orchestrator_lm, "_yaw_context_window_tokens", None)
                or getattr(rlm_sub_lm, "_yaw_context_window_tokens", None)
                or DEFAULT_CONTEXT_WINDOW_TOKENS
            )
        if max_output_tokens is None:
            max_output_tokens = max(
                int((getattr(lm, "kwargs", {}) or {}).get("max_tokens") or 0) for lm in (orchestrator_lm, rlm_sub_lm)
            ) or scale_context_budget(4096, context_window_tokens)
        if rlm_max_output_chars is None:
            rlm_max_output_chars = scale_context_budget(10000, context_window_tokens)
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
        self.enable_react_fallback: bool = bool(enable_react_fallback)
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
        self.shacl_healer_rlm = PositionalSafeRLM(
            HealSHACLViolations,
            stage_rule=STAGE_RLM_DELEGATION_RULES["shacl_repair"],
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
            rlm_demos = build_rlm_action_demos()
            self.shacl_healer_rlm.generate_action.demos = list(rlm_demos["shacl_repair"])

        self.loaded_from_path: Path | None = None
        if compiled_path is not None:
            path = Path(compiled_path)
            if path.exists():
                self.load(str(path))

    def record_embedding_telemetry(self, entry: Mapping[str, Any]) -> None:
        """Record a DSPyEmbeddingAdapter invocation in both telemetry and embedding_audit_trail."""
        record = dict(entry)
        self.embedding_audit_trail.append(record)
        self.telemetry.append(record)

    def dump_audit_trails(
        self,
        *,
        bounded: bool = True,
        telemetry_key: str = "telemetry",
    ) -> dict[str, list[dict[str, Any]]]:
        """Return serialized copies of all 5 program telemetry and audit trails."""
        limits = {
            "telemetry": 250,
            "embedding_audit_trail": 250,
            "entity_def_audit_trail": 250,
            "synonym_audit_trail": 100,
            "shacl_audit_trail": 100,
        }
        out: dict[str, list[dict[str, Any]]] = {}
        for name, limit in limits.items():
            items: list[dict[str, Any]] = list(getattr(self, name, ()))
            sliced = items[-limit:] if bounded else items
            key = telemetry_key if name == "telemetry" else name
            out[key] = [dict(x) if isinstance(x, Mapping) else x for x in sliced]
        return out

    def _bind_predictor_lms(self) -> None:
        """Bind native LMs by role; unbound predictors use DSPy's current context."""
        worker = self.rlm_sub_lm or self.orchestrator_lm
        orchestrator = self.orchestrator_lm or self.rlm_sub_lm
        if worker is not None:
            for predictor in (
                self.ner,
                self.triples,
                self.entity_definer,
                self.synonym_judge,
                self.shacl_repair,
                self.triple_filter,
                self.query_intent,
            ):
                predictor.set_lm(worker)
        if orchestrator is not None:
            self.qa.set_lm(orchestrator)
            self.shacl_healer_rlm.set_lm(orchestrator)

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
        """Bind storage tools (`inspect_focus_node`, `hydrate_chunk`, `apply_shacl_repairs`, `lint_shacl`) for the \
SHACL RLM sandbox."""
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

    def copy_runtime_hooks_to(self, target: HippoRAGProgram) -> None:
        """Copy runtime retrieval and graph tool hooks onto a sibling `HippoRAGProgram`."""
        target.set_retrieval_hook(self._retrieval_hook)
        target.set_graph_tools(
            node_inspector=self._node_inspector,
            chunk_hydrator=self._chunk_hydrator,
            shacl_repair_applier=self._shacl_repair_applier,
            shacl_linter=self._shacl_linter,
        )
        target.set_graph_search_tool(self._graph_search_tool)
        if getattr(self, "retriever", None) is not None and getattr(target, "retriever", None) is not None:
            target.retriever.k = self.retriever.k
        if getattr(self, "rm", None) is not None and getattr(target, "rm", None) is not None:
            target.rm.default_k = self.rm.default_k

    @classmethod
    def from_config(cls, config: YawConfig) -> HippoRAGProgram:
        """Construct a HippoRAGProgram from a YawConfig."""
        assert config.max_output_tokens is not None and config.rlm_max_output_chars is not None
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
        program.context_window_tokens = config.context_window_tokens
        program.max_output_tokens = config.max_output_tokens
        program.shacl_healer_rlm.max_iters = config.rlm_max_iters
        program.shacl_healer_rlm.max_llm_calls = config.rlm_max_llm_calls
        program.shacl_healer_rlm.max_output_chars = config.rlm_max_output_chars
        return program

    def _context_char_limit(self, base_chars: int) -> int:
        return context_char_limit(base_chars, self.context_window_tokens, self.max_output_tokens)

    def dump_state(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Serialize all DSPy predictors/RLMs plus EmbeddingGemma prefixes and GBNF structural bounds."""
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
        state["yaw_surface"] = yaw_surface
        return state

    def load_state(self, state: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        """Restore DSPy predictor/RLM state plus EmbeddingGemma prefixes and GBNF structural bounds."""
        state_copy = dict(state)
        state_copy.setdefault("retriever", {"k": getattr(self.retriever, "k", 10)})
        yaw_surface = state_copy.pop("yaw_surface", None)
        deployment_lms = {name: predictor.lm for name, predictor in self.named_predictors() if predictor.lm is not None}
        # Missing legacy predictors retain constructor defaults. Explicitly
        # saved empty demo lists must survive loading without reseeding.
        for p_name, param in self.named_parameters():
            if p_name not in state_copy and hasattr(param, "dump_state"):
                state_copy[p_name] = param.dump_state()
        for name in deployment_lms:
            if name in state_copy:
                # Instructions/demos are portable; explicit deployment LMs own
                # endpoints and credentials. Do not reconstruct clients from a
                # compiled prompt file when the caller already supplied them.
                state_copy[name] = {**state_copy[name], "lm": None}
        res = super().load_state(state_copy, *args, **kwargs)
        for name, predictor in self.named_predictors():
            if name in deployment_lms:
                predictor.set_lm(deployment_lms[name])
        if isinstance(yaw_surface, dict):
            # Serving capacity belongs to this deployment, not a compiled prompt.
            # Preserve it on hot reload as well as startup.
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
        if isinstance(self.json_adapter, YawGBNFAdapter):
            self.json_adapter.enable_gbnf = self.enable_gbnf
            self.json_adapter.max_entities = self.gbnf_max_entities
            self.json_adapter.max_triples = self.gbnf_max_triples
            self.json_adapter.max_facts = self.gbnf_max_facts
            self.json_adapter.max_reasoning_chars = self.gbnf_max_reasoning_chars
        return res

    def load(self, path: str | Path, *args: Any, **kwargs: Any) -> Any:
        """Load compiled state from disk and record `loaded_from_path` for Pareto ensemble discovery."""
        self.loaded_from_path = Path(path).resolve()
        return super().load(str(path), *args, **kwargs)

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

    def _dspy_ctx(self, lm: dspy.BaseLM | None, *, use_json_adapter: bool = True):
        ctx_kwargs: dict[str, Any] = {}
        if lm is not None:
            ctx_kwargs["lm"] = lm
        if use_json_adapter:
            ctx_kwargs["adapter"] = self.json_adapter
        return dspy.context(**ctx_kwargs)

    def _should_use_gbnf(self, use_gbnf: bool | None, lm: dspy.BaseLM | None) -> bool:
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
        tokens_used = sum_usage_tokens(usage)
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
                if lm is not None:
                    rlm_mod = rlm_mod.deepcopy()
                    rlm_mod.set_lm(lm)
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
                predictor = direct_mod.predict if isinstance(direct_mod, dspy.ChainOfThought) else direct_mod
                effective_lm = lm or predictor.lm or dspy.settings.lm
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
                if lm is not None:
                    call_kwargs["lm"] = lm
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
                    max_retries = self.max_invariant_retries
                    while (
                        inv_score < 1.0
                        and retry_count < max_retries
                        and should_retry_stage_invariants(stage_name, pred, effective_lm)
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
            (ename, edef or self._last_ner_definitions.get(ename, "")) for ename, edef in parsed_defs
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

        stage_tokens = int(self.telemetry[-1].get("tokens_used", 0)) if len(self.telemetry) > prev_telem_len else 0
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
            if (str(x[0]).strip() if isinstance(x, (list, tuple)) and len(x) >= 1 else str(x).strip())
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
        structured = list(extract_structured_triples(raw_triples, entity_definitions=all_def_pairs or merged_defs))

        # Propagate any inline [name, def] pairs across all triples in the same chunk first
        for st in structured:
            if st.subj and st.subj_def:
                merged_defs[st.subj] = st.subj_def
                merged_defs[str(text_processing(st.subj))] = st.subj_def
            if st.obj_kind != "literal" and st.obj and st.obj_def:
                merged_defs[st.obj] = st.obj_def
                merged_defs[str(text_processing(st.obj))] = st.obj_def

        missing_names: list[str] = []
        for idx, st in enumerate(structured):
            s_def = st.subj_def or merged_defs.get(st.subj, "") or merged_defs.get(str(text_processing(st.subj)), "")
            o_def = (
                ""
                if st.obj_kind == "literal"
                else (st.obj_def or merged_defs.get(st.obj, "") or merged_defs.get(str(text_processing(st.obj)), ""))
            )
            if s_def != st.subj_def or o_def != st.obj_def:
                structured[idx] = StructuredTriple(
                    subj=st.subj,
                    subj_def=s_def,
                    pred=st.pred,
                    obj=st.obj,
                    obj_def=o_def,
                    spo_def=st.spo_def,
                    obj_kind=st.obj_kind,
                    obj_datatype=st.obj_datatype,
                    obj_lang=st.obj_lang,
                )
            if st.subj and not s_def and st.subj not in missing_names:
                missing_names.append(st.subj)
            if st.obj_kind != "literal" and st.obj and not o_def and st.obj not in missing_names:
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
                    st.subj_def or merged_defs.get(st.subj, "") or merged_defs.get(str(text_processing(st.subj)), "")
                )
                o_def = (
                    ""
                    if st.obj_kind == "literal"
                    else (
                        st.obj_def or merged_defs.get(st.obj, "") or merged_defs.get(str(text_processing(st.obj)), "")
                    )
                )
                if s_def != st.subj_def or o_def != st.obj_def:
                    structured[idx] = StructuredTriple(
                        subj=st.subj,
                        subj_def=s_def,
                        pred=st.pred,
                        obj=st.obj,
                        obj_def=o_def,
                        spo_def=st.spo_def,
                        obj_kind=st.obj_kind,
                        obj_datatype=st.obj_datatype,
                        obj_lang=st.obj_lang,
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
        """Execute OpenIE (NER w/ definitions -> Structured Triples -> Missing Entity Definitions) for a passage \
chunk."""
        entities, entity_definitions = self._run_ner_stage(passage, lm=lm, use_gbnf=use_gbnf)
        has_nonempty_defs = any(defn for _, defn in entity_definitions)
        stage2_entities_input: list[Any] = (
            [[name, defn] for name, defn in entity_definitions] if has_nonempty_defs else list(entities)
        )
        structured_triples = self.extract_structured_triples(
            passage,
            stage2_entities_input,
            entity_definitions=entity_definitions,
            lm=lm,
            use_gbnf=use_gbnf,
        )
        merged_entities, merged_defs = _merge_openie_definitions(entities, entity_definitions, structured_triples)
        triples = tuple(st.as_tuple() for st in structured_triples)
        return OpenIERecord(
            chunk_id=chunk_id,
            entities=merged_entities,
            triples=triples,
            structured_triples=structured_triples,
            entity_definitions=merged_defs,
        )

    def judge_synonyms(
        self,
        candidate_pairs: Sequence[Mapping[str, Any]] | str,
        *,
        lm: Any = None,
        use_gbnf: bool | None = None,
    ) -> list[dict[str, str]]:
        """Classify candidate entity pairs into SKOS/OWL relations (`skos:altLabel`, `skos:hiddenLabel`, \
`skos:related`, `owl:differentFrom`)."""
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
                        pairs_json = _bounded_prompt_json(
                            parsed_pairs,
                            scale_context_budget(16, self.context_window_tokens),
                            self._context_char_limit(12000),
                        )
                except json.JSONDecodeError, TypeError:
                    pairs_json = pairs_json[: self._context_char_limit(12000)]
        else:
            pairs_list = [dict(p) for p in candidate_pairs if isinstance(p, Mapping)]
            if not pairs_list:
                return []
            pairs_json = _bounded_prompt_json(
                pairs_list, scale_context_budget(16, self.context_window_tokens), self._context_char_limit(12000)
            )
        if pairs_json == "[]":
            return []

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
        stage_tokens = int(self.telemetry[-1].get("tokens_used", 0)) if len(self.telemetry) > prev_telem_len else 0
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
                        violations_json = _bounded_prompt_json(
                            parsed_v,
                            scale_context_budget(16, self.context_window_tokens),
                            self._context_char_limit(12000),
                        )
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
                            "source_shape": str(getattr(v, "source_shape", getattr(v, "shape", ""))),
                            "result_path": str(getattr(v, "result_path", "")),
                            "value": str(getattr(v, "value", "")),
                            "message": str(getattr(v, "message", "")),
                        }
                    )
            if not serialized:
                return []
            violations_json = (
                json.dumps(serialized, ensure_ascii=False)
                if is_rlm
                else _bounded_prompt_json(
                    serialized, scale_context_budget(16, self.context_window_tokens), self._context_char_limit(12000)
                )
            )
        if violations_json == "[]":
            return []

        stage_inputs: dict[str, Any] = {"violations": violations_json}
        if not is_rlm:
            stage_inputs["focus_context"] = focus_context[: self._context_char_limit(12000)]
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
        stage_tokens = int(self.telemetry[-1].get("tokens_used", 0)) if len(self.telemetry) > prev_telem_len else 0
        self.shacl_audit_trail.append(
            {
                "mode": "rlm" if is_rlm else "direct",
                "predictor": "shacl_healer_rlm" if is_rlm else "shacl_repair",
                "signature": (HealSHACLViolations.__name__ if is_rlm else ProposeSHACLRepairs.__name__),
                "tokens_used": stage_tokens,
                "violations": violations_json,
                "focus_context": focus_context,
                "repairs": repairs,
            }
        )
        return repairs

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
            grammar_fn=lambda: build_dspy_rerank_grammar(candidate_facts, max_facts=effective_max_facts),
            question=question,
            fact_before_filter=fact_before_filter,
        )
        raw_after = getattr(pred, "fact", getattr(pred, "fact_after_filter", None))
        generated = coerce_generated_facts(raw_after)
        return match_facts_to_candidate_indices(candidate_facts, generated, max_facts=effective_max_facts)

    def extract_query_intent(
        self,
        question: str,
        *,
        lm: Any = None,
        use_gbnf: bool | None = None,
    ) -> str:
        """Rewrite question into a declarative query_intent string (GBNF on Qwen GPU, temp=0.4, freq_penalty=0.2)."""
        from dspy.utils.usage_tracker import track_usage

        effective_lm = lm or self.query_intent.lm or dspy.settings.lm
        call_cfg: dict[str, Any] = {
            "temperature": 0.4,
            "frequency_penalty": 0.2,
        }
        if self._should_use_gbnf(use_gbnf, effective_lm):
            call_cfg["extra_body"] = {"grammar": build_dspy_query_intent_grammar()}
        with track_usage() as usage_tracker, self._dspy_ctx(effective_lm, use_json_adapter=True):
            pred = self.query_intent(question=question, config=call_cfg, **({"lm": lm} if lm is not None else {}))
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
        """Run bounded `dspy.ReActV2` multi-hop graph exploration when single-shot PPR/QA abstains or retrieval \
confidence < 0.15."""
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
            logging.getLogger(__name__).debug("ReActV2 fallback skipped (%s: %s)", type(exc).__name__, exc)
            return None

    @diagnostic_boundary
    def forward(
        self,
        context: str | None = None,
        question: str | None = None,
        *,
        item_docs: Sequence[str] | None = None,
        use_gbnf: bool | None = None,
        retrieve_only: bool = False,
        k: int | None = None,
    ) -> dspy.Prediction:
        """Default DSPy forward pass executes multi-hop QA or the full extraction+retrieval+QA trace."""
        from dspy.utils.usage_tracker import track_usage

        if context is not None and question is not None:
            return self.answer_question(context=context, question=question, use_gbnf=use_gbnf)
        if context is not None and question is None:
            return self.forward(
                context=None,
                question=context,
                item_docs=item_docs,
                use_gbnf=use_gbnf,
                retrieve_only=retrieve_only,
                k=k,
            )
        if question is not None:
            if self._retrieval_hook is not None or getattr(dspy.settings, "rm", None) is not None:
                with track_usage() as usage_tracker:
                    ret_pred = self.retriever(question, k=k, item_docs=item_docs)
                    hydrated_ctx = str(getattr(ret_pred, "context", "") or "")
                    raw_psg = getattr(ret_pred, "all_passages", getattr(ret_pred, "passages", None))
                    passages = tuple(raw_psg) if raw_psg is not None else ()
                    raw_sc = getattr(ret_pred, "all_scores", getattr(ret_pred, "scores", None))
                    scores = tuple(float(s) for s in raw_sc) if raw_sc is not None else ()
                    raw_pd = getattr(ret_pred, "all_pids", getattr(ret_pred, "pids", None))
                    pids = tuple(str(p) for p in raw_pd) if raw_pd is not None else ()
                    if retrieve_only:
                        ans_str = ""
                        reas_str = ""
                    else:
                        pred = self.answer_question(context=hydrated_ctx, question=question, use_gbnf=use_gbnf)
                        top_score = max(scores) if len(scores) > 0 else 1.0
                        if self.enable_react_fallback and (
                            is_abstaining_answer(str(getattr(pred, "answer", "") or "")) or top_score < 0.15
                        ):
                            react_pred = self._run_react_qa_fallback(question, hydrated_ctx)
                            if react_pred is not None and not is_abstaining_answer(
                                str(getattr(react_pred, "answer", "") or "")
                            ):
                                pred = react_pred
                        ans_str = getattr(pred, "answer", "")
                        reas_str = getattr(pred, "reasoning", "")
                tokens = usage_tracker.get_total_tokens()
                res = dspy.Prediction(
                    context=hydrated_ctx,
                    reasoning=reas_str,
                    retrieved_passages=passages,
                    retrieved_scores=scores,
                    retrieved_chunk_ids=pids,
                    answer=ans_str,
                )
                if tokens and hasattr(res, "set_lm_usage"):
                    res.set_lm_usage(tokens)
                return res
            raise ValueError("context must be provided when no retrieval_hook is registered on HippoRAGProgram.")
        raise ValueError("Either (context, question) or question must be provided to HippoRAGProgram.")

    def as_async(self) -> Any:
        """Return a `dspy.asyncify`-wrapped async callable for this `HippoRAGProgram`."""
        return dspy.asyncify(self)


__all__ = [
    "HippoRAGProgram",
]
