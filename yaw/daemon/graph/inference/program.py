"""Native DSPy YawProgram orchestrator."""

import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import dspy

from yaw.core.config import DEFAULT_CONTEXT_WINDOW_TOKENS, YawConfig, context_char_limit, scale_context_budget
from yaw.core.types import OpenIERecord, StructuredTriple, SynonymEvidenceBatch, is_resource_label

from .adapter import YawGBNFAdapter, sum_usage_tokens
from .chat import create_dspy_lm
from .diagnostics import DiagnosticHistory, diagnostic_boundary
from .grammars import (
    build_dspy_hierarchy_judge_grammar,
    build_dspy_label_judge_grammar,
    build_dspy_missing_entity_defs_grammar,
    build_dspy_ner_grammar,
    build_dspy_qa_grammar,
    build_dspy_query_intent_grammar,
    build_dspy_related_judge_grammar,
    build_dspy_rerank_grammar,
    build_dspy_shacl_repair_grammar,
    build_dspy_triples_grammar,
    lm_supports_gbnf,
)
from .invariants import validate_stage_invariants
from .parsers import (
    coerce_generated_facts,
    extract_entity_definitions,
    extract_structured_triples,
    match_facts_to_candidate_indices,
    parse_hierarchy_decisions,
    parse_label_decisions,
    parse_related_decisions,
    parse_shacl_repairs,
    shacl_repair_actions,
    text_processing,
)
from .prompts import (
    AnswerQuestion,
    DefineMissingEntities,
    ExtractEntities,
    ExtractQueryIntent,
    ExtractTriples,
    FilterTriples,
    HealSHACLViolations,
    JudgeHierarchy,
    JudgeLabels,
    JudgeRelated,
    ProposeSHACLRepairs,
)
from .reflection import ServingTokenCounter
from .retriever import YawRetriever, YawRM
from .rlm import STAGE_RLM_DELEGATION_RULES, PositionalSafeRLM
from .synonyms import SynonymRequestBudget


class YawProgram(dspy.Module):
    """Unified DSPy Module for HippoRAG 2 OpenIE, EDC/SKOS Schema Canon, SHACL Repair, Recognition Memory, and \
Multi-Hop QA.

    Exposes the canonical runtime stage predictors (out of the 12 total DSPy signatures in YAW):
    1. **Direct GBNF Predictors (configured chat model on `llama.cpp` GPU):**
       - `ner` (`ExtractEntities`)
       - `triples` (`ExtractTriples`)
       - `entity_definer` (`DefineMissingEntities`)
       - `label_judge` (`JudgeLabels`)
       - `hierarchy_judge` (`JudgeHierarchy`)
       - `related_judge` (`JudgeRelated`)
       - `shacl_repair` (`ProposeSHACLRepairs`)
       - `triple_filter` (`FilterTriples`)
       - `query_intent` (`ExtractQueryIntent`)
       - `qa` (`AnswerQuestion`)
    2. **1 Single-Level RLM Python REPL Orchestrator (`shacl_healer_rlm`):**
       - `shacl_healer_rlm` (`HealSHACLViolations`) running inside `dspy.PythonInterpreter` (managed \
Deno/Pyodide WASM) with
         bound graph inspection, hydration, repair, and SQL SHACL linting tools (`inspect_focus_node`,
         `hydrate_chunk`, `propose_repairs`, `lint_shacl`). Repairs are returned as proposals.
    3. **First-Class Embedding Trail (`record_embedding_telemetry` / `embedding_audit_trail`):**
       - Tracks every `DSPyEmbeddingAdapter` (`dspy.Embedder`) invocation and the 3 optimizable
         task prefixes (`embedding_query_prefix`, `embedding_fact_prefix`, `embedding_document_prefix`)
         in `self.telemetry` and `self.embedding_audit_trail`.
    """

    def __init__(
        self,
        *,
        enable_gbnf: bool = False,
        gbnf_max_entities: int | None = None,
        gbnf_max_triples: int | None = None,
        gbnf_max_facts: int = 4,
        gbnf_max_reasoning_chars: int | None = 360,
        embedding_query_prefix: str | None = None,
        embedding_fact_prefix: str | None = None,
        embedding_document_prefix: str | None = None,
        adapter: dspy.Adapter | None = None,
        orchestrator_lm: Any = None,
        rlm_max_iters: int = 20,
        rlm_max_llm_calls: int = 50,
        rlm_max_output_chars: int | None = None,
        rlm_sub_lm: Any = None,
        rlm_tools: list[Callable[..., Any]] | None = None,
        rlm_interpreter_factory: Callable[[], Any] = dspy.PythonInterpreter,
        compiled_path: str | Path | None = None,
        context_window_tokens: int | None = None,
        max_output_tokens: int | None = None,
        synonym_token_counter: Any = None,
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
        self.synonym_token_counter = synonym_token_counter
        self._synonym_tokenizer_settings: tuple[str, str] | None = None
        self._synonym_budget: SynonymRequestBudget | None = None
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
        self._pair_inspector: Callable[[str, str], str] | None = None
        self._chunk_hydrator: Callable[[str], str] | None = None
        self._ast_island_hydrator: Callable[[str], str] | None = None
        self._shacl_linter: Callable[[], list[dict[str, str]]] | None = None
        self._focus_context: str = ""
        self._graph_search_tool: Callable[[str], str] | None = None
        self.telemetry: list[dict[str, Any]] = DiagnosticHistory()
        self.history = DiagnosticHistory()
        self.embedding_audit_trail: list[dict[str, Any]] = DiagnosticHistory()
        self.entity_def_audit_trail: list[dict[str, Any]] = DiagnosticHistory()
        self.label_judge_audit_trail: list[dict[str, Any]] = DiagnosticHistory()
        self.hierarchy_judge_audit_trail: list[dict[str, Any]] = DiagnosticHistory()
        self.related_judge_audit_trail: list[dict[str, Any]] = DiagnosticHistory()
        self.shacl_audit_trail: list[dict[str, Any]] = DiagnosticHistory()

        # 10 Direct GBNF Predictors with distinct Signatures
        self.ner = dspy.Predict(ExtractEntities)
        self.triples = dspy.Predict(ExtractTriples)
        self.entity_definer = dspy.Predict(DefineMissingEntities)
        self.label_judge = dspy.Predict(JudgeLabels)
        self.hierarchy_judge = dspy.Predict(JudgeHierarchy)
        self.related_judge = dspy.Predict(JudgeRelated)
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
        """Return serialized copies of all program telemetry and audit trails."""
        limits = {
            "telemetry": 250,
            "embedding_audit_trail": 250,
            "entity_def_audit_trail": 250,
            "label_judge_audit_trail": 100,
            "hierarchy_judge_audit_trail": 100,
            "related_judge_audit_trail": 100,
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
                self.label_judge,
                self.hierarchy_judge,
                self.related_judge,
                self.shacl_repair,
                self.triple_filter,
                self.query_intent,
            ):
                predictor.set_lm(worker)
        if orchestrator is not None:
            self.qa.set_lm(orchestrator)
            self.shacl_healer_rlm.set_lm(orchestrator)

    def set_graph_tools(
        self,
        *,
        node_inspector: Callable[[str], dict[str, Any]] | None = None,
        pair_inspector: Callable[[str, str], str] | None = None,
        chunk_hydrator: Callable[[str], str] | None = None,
        ast_island_hydrator: Callable[[str], str] | None = None,
        shacl_linter: Callable[[], list[dict[str, str]]] | None = None,
    ) -> None:
        """Bind storage tools (`inspect_focus_node`, `inspect_pair`, `hydrate_chunk`, `hydrate_ast_island`, \
`lint_shacl`) for the read-only SHACL RLM sandbox."""
        self._node_inspector = node_inspector
        self._pair_inspector = pair_inspector
        self._chunk_hydrator = chunk_hydrator
        self._ast_island_hydrator = ast_island_hydrator
        self._shacl_linter = shacl_linter

    def set_graph_search_tool(
        self,
        search_fn: Callable[[str], str] | None,
    ) -> None:
        """Bind a graph fact search callback for multi-hop `dspy.ReActV2` fallback exploration."""
        self._graph_search_tool = search_fn

    def copy_runtime_hooks_to(self, target: YawProgram) -> None:
        """Copy runtime retrieval and graph tool hooks onto a sibling `YawProgram`."""
        target.set_retrieval_hook(self._retrieval_hook)
        target.set_graph_tools(
            node_inspector=self._node_inspector,
            pair_inspector=self._pair_inspector,
            chunk_hydrator=self._chunk_hydrator,
            ast_island_hydrator=self._ast_island_hydrator,
            shacl_linter=self._shacl_linter,
        )
        target.set_graph_search_tool(self._graph_search_tool)
        if getattr(self, "retriever", None) is not None and getattr(target, "retriever", None) is not None:
            target.retriever.k = self.retriever.k
        if getattr(self, "rm", None) is not None and getattr(target, "rm", None) is not None:
            target.rm.default_k = self.rm.default_k

    @classmethod
    def from_config(cls, config: YawConfig) -> YawProgram:
        """Construct a YawProgram from a YawConfig."""
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
        program._synonym_tokenizer_settings = (config.base_url, config.reflection_tokenizer_url)
        return program

    def _context_char_limit(self, base_chars: int) -> int:
        return context_char_limit(base_chars, self.context_window_tokens, self.max_output_tokens)

    def dump_state(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Serialize all DSPy predictors/RLMs plus EmbeddingGemma prefixes and GBNF structural bounds."""
        state: dict[str, Any] = super().dump_state(*args, **kwargs)
        yaw_surface: dict[str, Any] = {
            "graph_schema": "scheme-free-v1",
            "shacl_tools": "canonical-report-v1",
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
        if not isinstance(yaw_surface, dict) or yaw_surface.get("graph_schema") != "scheme-free-v1":
            raise ValueError("Incompatible program graph schema; recompile against the scheme-free graph")
        if yaw_surface.get("shacl_tools") != "canonical-report-v1":
            raise ValueError("Incompatible SHACL tool contract; start a fresh optimization")
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

    def __deepcopy__(self, memo: dict[int, Any]) -> YawProgram:
        """Clone all DSPy predictors while preserving live LM clients and the engine retrieval hook."""
        import copy

        cls = self.__class__
        new_prog = cls.__new__(cls)
        memo[id(self)] = new_prog
        for k, v in self.__dict__.items():
            if k in (
                "_retrieval_hook",
                "_node_inspector",
                "_pair_inspector",
                "_chunk_hydrator",
                "_ast_island_hydrator",
                "_shacl_linter",
                "_focus_context",
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
        new_prog.shacl_healer_rlm._user_tools = new_prog.shacl_healer_rlm._normalize_tools(
            new_prog._build_shacl_repl_tools()
        )
        return new_prog

    def _resolve_worker_lm(self, fallback_lm: dspy.LM | None = None) -> dspy.LM | None:
        return self.rlm_sub_lm if self.rlm_sub_lm is not None else fallback_lm

    def _resolve_orchestrator_lm(self, fallback_lm: dspy.LM | None = None) -> dspy.LM | None:
        return fallback_lm if fallback_lm is not None else self.orchestrator_lm

    def _build_shacl_repl_tools(self) -> list[Callable[..., Any]]:
        """Build the closed-world SHACL healing tools exposed inside `shacl_healer_rlm`."""

        def propose_repairs() -> list[dict[str, str]]:
            """Propose repairs for the actual batch report with complete structured focus context."""
            if self._shacl_linter is None or (not self._focus_context.strip() and self._node_inspector is None):
                raise ValueError("SHACL repair requires a bound report and node inspector")
            worker_lm = self._resolve_worker_lm()
            raw = self._shacl_linter()
            if self._focus_context.strip():
                resolved_context = self._focus_context
            else:
                assert self._node_inspector is not None
                foci = dict.fromkeys(str(item["focus_node"]) for item in raw)
                resolved_context = json.dumps([self._node_inspector(fn) for fn in foci], ensure_ascii=False)
            return self.propose_shacl_repairs(
                raw,
                focus_context=resolved_context,
                mode="direct",
                lm=worker_lm,
                use_gbnf=self.enable_gbnf,
            )

        def inspect_focus_node(entity_id: str) -> dict[str, Any]:
            """Inspect focus node (`entity-...`, `fact-...`, `entity-a|entity-b`, or `entity-a->entity-b`) labels, \
definition, semantic edges, reified facts, chunk pointers, and markdown_dossier."""
            if self._node_inspector is None:
                return {}
            return dict(self._node_inspector(str(entity_id)))

        def inspect_pair(node_a: str, node_b: str) -> str:
            """Return a structured Markdown + SKOS/RDF/OWL comparison dossier for two entities/literals."""
            if self._pair_inspector is not None:
                return str(self._pair_inspector(str(node_a), str(node_b)))
            if self._node_inspector is not None:
                info = self._node_inspector(f"{node_a}|{node_b}")
                if isinstance(info, Mapping):
                    return str(info.get("markdown_dossier", "")) or json.dumps(dict(info), ensure_ascii=False)
            return ""

        def hydrate_chunk(chunk_id: str) -> str:
            """Hydrate source chunk lines from disk for a given chunk_id."""
            if self._chunk_hydrator is None:
                return ""
            return str(self._chunk_hydrator(str(chunk_id)))

        def hydrate_ast_island(chunk_id: str) -> str:
            """Hydrate the full enclosing CocoIndex AST island for a given chunk_id."""
            if self._ast_island_hydrator is not None:
                return str(self._ast_island_hydrator(str(chunk_id)))
            return hydrate_chunk(chunk_id)

        def read_source_chunk(chunk_id: str) -> str:
            """Hydrate source chunk lines from disk for a given chunk_id."""
            return hydrate_chunk(chunk_id)

        def search_graph(query: str) -> str:
            """Search the graph for entities and reified facts matching a query string."""
            if self._graph_search_tool is None:
                return ""
            return str(self._graph_search_tool(str(query)))

        def lint_shacl() -> list[dict[str, str]]:
            """Read the proposal's violations; repair proposals are applied only after SUBMIT."""
            if self._shacl_linter is None:
                return []
            return list(self._shacl_linter())

        return [
            propose_repairs,
            inspect_focus_node,
            inspect_pair,
            hydrate_chunk,
            hydrate_ast_island,
            read_source_chunk,
            search_graph,
            lint_shacl,
        ]

    def _dspy_ctx(self, lm: dspy.BaseLM | None, *, use_json_adapter: bool = True, adapter: dspy.Adapter | None = None):
        ctx_kwargs: dict[str, Any] = {"warn_on_type_mismatch": False}
        if lm is not None:
            ctx_kwargs["lm"] = lm
        if use_json_adapter:
            ctx_kwargs["adapter"] = adapter or self.json_adapter
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
        request_adapter: dspy.Adapter | None = None,
        **inputs: Any,
    ) -> Any:
        from dspy.utils.usage_tracker import track_usage

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
                    shacl_linter=None,
                )
            else:
                predictor = direct_mod.predict if isinstance(direct_mod, dspy.ChainOfThought) else direct_mod
                effective_lm = lm or getattr(predictor, "lm", None) or dspy.settings.lm
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
                with self._dspy_ctx(effective_lm, use_json_adapter=True, adapter=request_adapter):
                    pred = direct_mod(**direct_inputs, **call_kwargs)
                    inv_score, inv_feedback = validate_stage_invariants(
                        stage_name,
                        direct_inputs,
                        pred,
                        shacl_linter=None,
                    )
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
        literals = [name for name, _ in parsed_defs if not is_resource_label(name)]
        if literals:
            raise ValueError(f"NER resource label lacks lexical content (literals are not resources): {literals!r}")
        entities = tuple(ename for ename, _ in parsed_defs)
        return entities, tuple(parsed_defs)

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
        """Repair definitions of requested NER resources without discovering new resources."""
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
                triples_payload.append(t.as_tuple())
            elif isinstance(t, (list, tuple)) and len(t) == 3:
                triples_payload.append((str(t[0]), str(t[1]), str(t[2])))
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
        requested_norms = {str(text_processing(name)) for name in deduped}
        exact_map: dict[str, str] = {}
        norm_map: dict[str, str] = {}
        for ename, edef in parsed:
            if str(text_processing(ename)) not in requested_norms:
                raise ValueError("Definition repair introduced an unrequested resource")
            if ename and edef:
                exact_map[ename] = edef
                norm_map[str(text_processing(ename))] = edef

        resolved: list[tuple[str, str]] = []
        for m_name in deduped:
            defn = exact_map.get(m_name) or norm_map.get(str(text_processing(m_name))) or ""
            if defn:
                resolved.append((m_name, defn))

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
    ) -> tuple[StructuredTriple, ...]:
        """Link the NER catalog; only literal objects may introduce new values."""
        names = [
            str(item[0]).strip() if isinstance(item, (list, tuple)) else str(item).strip() for item in named_entities
        ]
        if not names:
            return ()
        definitions = list(
            entity_definitions.items() if isinstance(entity_definitions, dict) else entity_definitions or ()
        )
        definitions.extend(
            (name, definition) for name, definition in extract_entity_definitions(named_entities) if definition
        )
        normalized_entities: list[Any] = [
            (str(item[0]), str(item[1]))
            if isinstance(item, (list, tuple)) and len(item) >= 2
            else str(item[0] if isinstance(item, (list, tuple)) and item else item)
            for item in named_entities
        ]
        pred = self._invoke_stage(
            self.triples,
            stage_name="triples",
            is_rlm=False,
            lm=lm,
            use_gbnf=use_gbnf,
            grammar_fn=lambda: build_dspy_triples_grammar(self.gbnf_max_triples, known_entities=named_entities),
            passage=passage,
            named_entities=normalized_entities,
        )
        return extract_structured_triples(
            getattr(pred, "triples", []),
            entity_definitions=definitions,
            known_entities=names,
        )

    def extract_openie(
        self,
        chunk_id: str,
        passage: str,
        *,
        lm: Any = None,
        use_gbnf: bool | None = None,
    ) -> OpenIERecord:
        """Execute OpenIE (NER-owned resources/definitions -> resource-closed statements) for a passage \
chunk."""
        entities, entity_definitions = self._run_ner_stage(passage, lm=lm, use_gbnf=use_gbnf)
        has_nonempty_defs = any(defn for _, defn in entity_definitions)
        stage2_entities_input: list[Any] = list(entity_definitions) if has_nonempty_defs else list(entities)
        structured_triples = self.extract_structured_triples(
            passage,
            stage2_entities_input,
            entity_definitions=entity_definitions,
            lm=lm,
            use_gbnf=use_gbnf,
        )
        triples = tuple(st.as_tuple() for st in structured_triples)
        return OpenIERecord(
            chunk_id=chunk_id,
            entities=entities,
            triples=triples,
            structured_triples=structured_triples,
            entity_definitions=entity_definitions,
        )

    def synonym_request_budget(self, lm: Any = None) -> SynonymRequestBudget:
        """Resolve a verified tokenizer for the active synonym LM, retaining one budget object."""
        effective = lm or self.label_judge.lm or dspy.settings.lm
        if effective is None:
            raise ValueError("Synonym judgment requires an LM")
        counter = self.synonym_token_counter
        if counter is None:
            settings = self._synonym_tokenizer_settings
            base_url = settings[0] if settings else effective.kwargs.get("api_base")
            if not base_url:
                raise ValueError("Synonym judgment requires an explicit serving token counter")
            counter = ServingTokenCounter(base_url, effective.model, settings[1] if settings else "")
            self.synonym_token_counter = counter
        if isinstance(counter, ServingTokenCounter):
            counter = counter.for_model(effective.model)
            self.synonym_token_counter = counter
        if (
            self._synonym_budget is None
            or self._synonym_budget.counter is not counter
            or self._synonym_budget.adapter is not self.json_adapter
            or self._synonym_budget.output_tokens != self.max_output_tokens
            or self._synonym_budget.context_tokens
            != min(self.context_window_tokens, getattr(counter, "context_tokens", self.context_window_tokens))
        ):
            self._synonym_budget = SynonymRequestBudget(
                self.json_adapter,
                counter,
                self.context_window_tokens,
                self.max_output_tokens,
            )
        return self._synonym_budget

    def synonym_batch_fits(self, batch: SynonymEvidenceBatch) -> bool:
        """Check a whole pair packet including signature, demos, output budget and chat template."""
        lm = self.label_judge.lm or dspy.settings.lm
        return self.synonym_request_budget(lm).fits(batch, self.label_judge, lm)

    def _run_judge_stage(
        self,
        stage_name: Literal["label_judge", "hierarchy_judge", "related_judge"],
        batch: SynonymEvidenceBatch,
        *,
        lm: Any = None,
        use_gbnf: bool | None = None,
        captured_traces: list[Any] | None = None,
        rlm_stats: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a single synonym judgment stage."""
        from .diagnostics import extract_named_trace_entries

        if not batch.candidates:
            return []

        predictor_map = {
            "label_judge": (
                self.label_judge,
                build_dspy_label_judge_grammar,
                parse_label_decisions,
                self.label_judge_audit_trail,
            ),
            "hierarchy_judge": (
                self.hierarchy_judge,
                build_dspy_hierarchy_judge_grammar,
                parse_hierarchy_decisions,
                self.hierarchy_judge_audit_trail,
            ),
            "related_judge": (
                self.related_judge,
                build_dspy_related_judge_grammar,
                parse_related_decisions,
                self.related_judge_audit_trail,
            ),
        }
        predictor, grammar_fn, parser_fn, audit_trail = predictor_map[stage_name]
        budget = self.synonym_request_budget(lm)

        prev_telem_len = len(self.telemetry)
        local_stage_trace: list[Any] = []
        with dspy.context(trace=local_stage_trace, usage_tracker=None):
            started = time.perf_counter()
            pred = self._invoke_stage(
                predictor,
                stage_name=stage_name,
                is_rlm=False,
                lm=lm,
                use_gbnf=use_gbnf,
                grammar_fn=lambda: grammar_fn(
                    pair_ids=[pair["pair_id"] for pair in json.loads(batch.payload)["pairs"]],
                    max_reasoning_chars=self.gbnf_max_reasoning_chars,
                ),
                request_adapter=budget,
                candidate_pairs=batch.payload,
            )
            elapsed = time.perf_counter() - started

        score, feedback = validate_stage_invariants(
            stage_name,
            {"candidate_pairs": batch.payload},
            pred,
        )
        if score < 1.0:
            raise ValueError(feedback)

        parsed_decs = parser_fn(pred.decisions)
        stage_tokens = int(self.telemetry[-1].get("tokens_used", 0)) if len(self.telemetry) > prev_telem_len else 0
        summary = {
            "packet_id": batch.cache_identity,
            "pair_count": len(batch.candidates),
            "packet_chars": len(batch.payload),
            "input_tokens": budget.last_count,
            "construction_seconds": batch.construction_seconds,
            "inference_seconds": elapsed,
            "cache_hit": False,
        }
        if len(self.telemetry) > prev_telem_len:
            self.telemetry[-1].update(summary)

        stage_traces = extract_named_trace_entries(
            self,
            local_stage_trace,
            allowed_names={stage_name},
        )
        if captured_traces is not None:
            captured_traces.extend(stage_traces)
        if rlm_stats is not None:
            tok_key = f"{stage_name}_tokens"
            rlm_stats[tok_key] = int(rlm_stats.get(tok_key, 0)) + stage_tokens

        audit_trail.append(
            {
                "mode": "direct",
                "predictor": stage_name,
                "signature": getattr(predictor.signature, "__name__", str(predictor.signature)),
                "tokens_used": stage_tokens,
                **summary,
                "decisions": parsed_decs,
            }
        )

        return parsed_decs

    def judge_synonyms(
        self,
        batch: SynonymEvidenceBatch,
        *,
        lm: Any = None,
        use_gbnf: bool | None = None,
        captured_traces: list[Any] | None = None,
        rlm_stats: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Judge candidate pairs via short-circuiting 3-stage W3C semantic cascade."""
        if not batch.candidates:
            return []

        payload_obj = json.loads(batch.payload)
        pairs_list = payload_obj.get("pairs", [])
        if not pairs_list:
            return []

        pairs_by_id = {p["pair_id"]: p for p in pairs_list}
        resolved_decisions: list[dict[str, Any]] = []

        # 1. Rule W1: Direct facts skip Stage 1 directly to Stage 2
        stage1_pids: list[str] = []
        stage2_pids: list[str] = []
        for p in pairs_list:
            pid = p["pair_id"]
            if p.get("direct_facts"):
                stage2_pids.append(pid)
            else:
                stage1_pids.append(pid)

        # 2. Stage 1: JudgeLabels (skos:altLabel / skos:hiddenLabel)
        if stage1_pids:
            stage1_batch = batch.subset(stage1_pids)
            s1_decs = self._run_judge_stage(
                "label_judge",
                stage1_batch,
                lm=lm,
                use_gbnf=use_gbnf,
                captured_traces=captured_traces,
                rlm_stats=rlm_stats,
            )
            for dec in s1_decs:
                pid = dec["pair_id"]
                equiv = bool(dec.get("equivalent"))
                hidden = bool(dec.get("hidden_label"))
                if equiv and hidden:
                    dec_copy = dict(dec)
                    dec_copy["relation"] = "skos:hiddenLabel"
                    resolved_decisions.append(dec_copy)
                elif equiv:
                    dec_copy = dict(dec)
                    dec_copy["relation"] = "skos:altLabel"
                    resolved_decisions.append(dec_copy)
                else:
                    stage2_pids.append(pid)

        # 4. Stage 2: JudgeHierarchy (skos:broader / skos:narrower)
        stage2_residuals: list[str] = []
        if stage2_pids:
            stage2_batch = batch.subset(stage2_pids)
            s2_decs = self._run_judge_stage(
                "hierarchy_judge",
                stage2_batch,
                lm=lm,
                use_gbnf=use_gbnf,
                captured_traces=captured_traces,
                rlm_stats=rlm_stats,
            )
            for dec in s2_decs:
                pid = dec["pair_id"]
                ea, eb = pid.split("|", 1)
                broader = bool(dec.get("broader"))
                narrower = bool(dec.get("narrower"))
                if broader and not narrower:
                    dec_copy = dict(dec)
                    dec_copy["relation"] = "skos:broader"
                    dec_copy["source_id"] = ea
                    dec_copy["target_id"] = eb
                    resolved_decisions.append(dec_copy)
                elif not broader and narrower:
                    dec_copy = dict(dec)
                    dec_copy["relation"] = "skos:broader"
                    dec_copy["source_id"] = eb
                    dec_copy["target_id"] = ea
                    resolved_decisions.append(dec_copy)
                elif broader and narrower:
                    dec_copy = dict(dec)
                    dec_copy["relation"] = "skos:related"
                    resolved_decisions.append(dec_copy)
                else:
                    stage2_residuals.append(pid)

        # 5. Rule W1 / W2 on Stage 2 (False, False) pairs: direct facts or shared roles -> skos:related
        stage3_pids: list[str] = []
        for pid in stage2_residuals:
            p = pairs_by_id[pid]
            if p.get("direct_facts") or p.get("shared_role_facts"):
                resolved_decisions.append(
                    {
                        "pair_id": pid,
                        "reasoning": "Deterministic skos:related via Rule W1/W2 statement role / direct fact",
                        "relation": "skos:related",
                    }
                )
            else:
                stage3_pids.append(pid)

        # 6. Stage 3: JudgeRelated (skos:related vs owl:differentFrom)
        if stage3_pids:
            stage3_batch = batch.subset(stage3_pids)
            s3_decs = self._run_judge_stage(
                "related_judge",
                stage3_batch,
                lm=lm,
                use_gbnf=use_gbnf,
                captured_traces=captured_traces,
                rlm_stats=rlm_stats,
            )
            for dec in s3_decs:
                dec_copy = dict(dec)
                related = bool(dec.get("related"))
                dec_copy["relation"] = "skos:related" if related else "owl:differentFrom"
                resolved_decisions.append(dec_copy)

        # 7. Aggregate all resolved decisions and return
        return batch.resolve_decisions(resolved_decisions)

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

        By default (`mode=None`), dispatches directly to `shacl_repair` (`ProposeSHACLRepairs`
        with pure-JSON GBNF) when `focus_context` is pre-hydrated, and otherwise executes the
        single-level RLM Python REPL healer (`shacl_healer_rlm` / `HealSHACLViolations`).
        """
        if not violations:
            return []
        is_rlm = (not bool(focus_context.strip())) if mode is None else (mode == "rlm")
        if isinstance(violations, str):
            violations_json = violations.strip()
            if not violations_json or violations_json == "[]":
                return []
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
            violations_json = json.dumps(serialized, ensure_ascii=False)
        if violations_json == "[]":
            return []

        stage_inputs: dict[str, Any] = {"violations": violations_json}
        allowed_actions = shacl_repair_actions(json.loads(violations_json))
        if not is_rlm:
            stage_inputs["focus_context"] = focus_context
        prev_telem_len = len(self.telemetry)
        prev_focus_context = self._focus_context
        if is_rlm:
            self._focus_context = focus_context
        try:
            pred = self._invoke_stage(
                self.shacl_repair,
                self.shacl_healer_rlm,
                stage_name="shacl_repair",
                is_rlm=is_rlm,
                lm=lm,
                use_gbnf=use_gbnf,
                grammar_fn=lambda: build_dspy_shacl_repair_grammar(allowed_actions=allowed_actions),
                **stage_inputs,
            )
        finally:
            if is_rlm:
                self._focus_context = prev_focus_context
        raw_repairs = getattr(pred, "repairs", [])
        repairs = parse_shacl_repairs(raw_repairs, allowed_actions=allowed_actions)
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
        """Filter candidate triples on llama w/ dynamic candidate-literal GBNF."""
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
        """Rewrite question into a declarative query_intent string (GBNF on llama, GPU, temp=1.0)."""
        from dspy.utils.usage_tracker import track_usage

        effective_lm = lm or self.query_intent.lm or dspy.settings.lm
        call_cfg: dict[str, Any] = {
            "temperature": 1.0,
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
                        ans_str = str(getattr(pred, "answer", "") or "")
                        reas_str = str(getattr(pred, "reasoning", "") or "")
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
            raise ValueError("context must be provided when no retrieval_hook is registered on YawProgram.")
        raise ValueError("Either (context, question) or question must be provided to YawProgram.")

    def as_async(self) -> Any:
        """Return a `dspy.asyncify`-wrapped async callable for this `YawProgram`."""
        return dspy.asyncify(self)


__all__ = [
    "YawProgram",
]
