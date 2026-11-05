"""Native DSPy HippoRAG 2 Graph and Retrieval Engine backed by yaw.db and disk line hydration."""

import hashlib
import importlib.metadata
import json
import sqlite3
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Literal

import dspy
import numpy as np

from yaw.core import (
    ChunkPointer,
    IndexStats,
    RelationTraversalPolicy,
    write_text_atomic,
)
from yaw.core.config import context_char_limit
from yaw.daemon.graph.inference import (
    FACT_INSTRUCTION,
    PASSAGE_INSTRUCTION,
    DSPyEmbeddingAdapter,
    GraphOptimizationHooks,
    HippoRAGProgram,
    RLMInstructionProposer,
    YawRetriever,
    YawRM,
    compile_pareto_ensemble,
    compute_mdhash_id,
    configure_simba_task_learning,
    configure_yaw_dspy_cache,
    embed_fingerprint,
    evaluate_program,
    evolve_program_prompts,
    extraction_fingerprint,
    openie_fingerprint,
    query_retrieval_fingerprint,
    text_processing,
    trim_diagnostics,
)
from yaw.daemon.graph.planner import (
    GraphPlanSummary,
)
from yaw.daemon.graph.ppr import (
    CSRGraph,
    build_transition_matrix,
    min_max_normalize,
    rank_passages_by_ppr,
    run_ppr,
)
from yaw.daemon.graph.storage import (
    CandidateCacheStore,
    CandidatePayloadCodec,
    CandidateStageCache,
    ManagedConnection,
    apply_shacl_repairs,
    delete_chunks,
    format_pair_disambiguation_dossier,
    format_subgraph_for_qa,
    get_chunk_pointers,
    get_entity_doc_frequencies,
    get_existing_ids,
    get_facts_by_ids,
    get_graph_counts,
    get_meta,
    hydrate_budgeted_passages,
    hydrate_passages,
    inspect_focus_node,
    lint_graph_shacl,
    load_graph_topology,
    make_ast_island_hydrator,
    make_chunk_hydrator,
    open_managed_yaw_db,
    pool_confidences,
    resolve_entity_labels,
    search_chunk_vectors_knn,
    search_chunks_fts,
    search_fact_vectors_knn,
    search_facts_fts,
    serialize_graph_turtle,
    set_meta,
)

_AUTO_MERGE_COSINE_GATE = 0.95
_AUTO_MERGE_DIFFLIB_GATE = 0.96


class YawGraphEngine:
    """HippoRAG 2 engine backed by yaw.db, CSR Personalized PageRank, and lazy disk line hydration."""

    def __init__(
        self,
        db_path: Path | str,
        workspace_root: Path | str,
        program: HippoRAGProgram,
        embedder: DSPyEmbeddingAdapter,
        *,
        linking_top_k: int = 8,
        fact_candidate_pool_k: int = 50,
        fact_fts_pool_k: int = 50,
        fact_rrf_k: int = 60,
        fact_fts_weight: float = 0.2,
        passage_seed_pool_k: int = 200,
        fts_seed_pool_k: int = 50,
        passage_node_weight: float = 0.05,
        fts_weight: float = 0.2,
        damping: float = 0.5,
        synonymy_threshold: float = 0.8,
        synonymy_top_k: int = 100,
        retrieval_top_k: int = 10,
        embed_batch_size: int = 64,
        activation: str = "scaled_tanh",
        activation_param: float | None = 5.0,
        activation_renormalize: bool = False,
        beta: float = 0.5,
        dual_weight: float | None = 0.5,
        auto_merge_cosine_gate: float = _AUTO_MERGE_COSINE_GATE,
        auto_merge_difflib_gate: float = _AUTO_MERGE_DIFFLIB_GATE,
        max_context_chars: int | None = None,
        max_ast_expansion_hops: int = 1,
        ast_context_hops: int = 0,
        traversal_policy: RelationTraversalPolicy | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.db_path = db_path
        db_path_str = str(db_path)
        if db_path_str != ":memory:" and not db_path_str.startswith("file:"):
            self._dspy_cache_dir = Path(db_path).resolve().parent / "dspy_cache"
        else:
            self._dspy_cache_dir = self.workspace_root / "dspy_cache"
        configure_yaw_dspy_cache(self._dspy_cache_dir, restrict_pickle=True, account_cached_usage=True)
        self._lock = threading.RLock()
        self.managed_conn: ManagedConnection = open_managed_yaw_db(db_path)
        self.conn: sqlite3.Connection = self.managed_conn._conn
        self.program = program
        self.embedder = embedder
        self.config: Any = None
        self.indexing_runner: (
            Callable[[Sequence[tuple[ChunkPointer, str]], HippoRAGProgram], GraphPlanSummary] | None
        ) = None

        self.max_context_chars = (
            max_context_chars
            if max_context_chars is not None
            else context_char_limit(24000, program.context_window_tokens, program.max_output_tokens)
        )
        self.max_ast_expansion_hops = max_ast_expansion_hops
        self.ast_context_hops = max(0, int(ast_context_hops))
        self.linking_top_k = linking_top_k
        self.fact_candidate_pool_k = fact_candidate_pool_k
        self.fact_fts_pool_k = fact_fts_pool_k
        self.fact_rrf_k = fact_rrf_k
        self.fact_fts_weight = fact_fts_weight
        self.passage_seed_pool_k = passage_seed_pool_k
        self.fts_seed_pool_k = fts_seed_pool_k
        self.passage_node_weight = passage_node_weight
        self.fts_weight = fts_weight
        self.damping = damping
        self.synonymy_threshold = synonymy_threshold
        self.synonymy_top_k = synonymy_top_k
        self.retrieval_top_k = retrieval_top_k
        self.embed_batch_size = embed_batch_size
        self.activation = activation
        self.activation_param = activation_param
        self.activation_renormalize = activation_renormalize
        self.beta = beta
        self.dual_weight = dual_weight
        self.auto_merge_cosine_gate = float(auto_merge_cosine_gate)
        self.auto_merge_difflib_gate = float(auto_merge_difflib_gate)
        self.traversal_policy: RelationTraversalPolicy | None = traversal_policy

        self._csr_graph: CSRGraph | None = None
        self._graph_rlm_stats: dict[str, Any] = self._empty_graph_rlm_stats()
        self._attempted_shacl_keys: set[tuple[str, str]] = set()
        self._chunk_rlm_stats: dict[str, dict[str, float]] = {}
        self._active_query_rlm_stats: dict[str, int] | None = None
        self._active_progress_callback: Callable[[str], None] | None = None
        self._last_hook_program: HippoRAGProgram | None = None
        self._last_scope_by_cids: dict[tuple[str, ...], dict[str, int]] = {}
        self._active_extraction_fingerprint: str | None = None
        self._active_openie_fingerprint: str | None = None
        self._baseline_openie_fp: str | None = None
        self._chunk_indexed_ext_fp: dict[str, str] = {}
        self._opt_chunks_pool: list[tuple[ChunkPointer, str]] = []
        self._opt_val_chunks_pool: list[tuple[ChunkPointer, str]] = []
        self._opt_docs_by_question: dict[str, list[str]] = {}
        self._opt_indexed_cids: set[str] = set()
        self._resolved_docs_cache: dict[tuple[int, str], list[tuple[ChunkPointer, str]]] = {}
        self._captured_chunk_traces: dict[str, list[tuple[str, dict[str, Any], Any]]] = {}
        self._captured_graph_traces: list[tuple[str, dict[str, Any], Any]] = []
        self._in_optimization: bool = False
        self._last_memo_question: str | None = None
        self._last_seed_query: str | None = None
        self._last_query_seeds: list[tuple[str, str, str]] = []
        self._query_retrieval_memo: dict[
            tuple[str, int, str],
            tuple[str, tuple[str, ...], tuple[float, ...], tuple[str, ...]],
        ] = {}
        self.ensemble: dspy.Module = self.program
        # One versioned, bounded disk cache in yaw.db. Candidate histories must
        # not accumulate vectors, extraction records and RLM traces in RAM.
        graph_dir = Path(__file__).parent
        self._candidate_cache_code_fp = hashlib.sha256(
            b"".join(
                p.relative_to(graph_dir).as_posix().encode("utf-8") + b"\0" + p.read_bytes()
                for p in sorted(graph_dir.rglob("*.py"))
            )
            + importlib.metadata.version("dspy").encode()
            + np.__version__.encode()
        ).hexdigest()
        with self.managed_conn.transaction():
            cache_store = CandidateCacheStore(self.conn)
        self._text_vec_cache = CandidateStageCache(
            cache_store, self.managed_conn, "embedding", self._candidate_cache_namespace
        )
        self._candidate_cache_store = cache_store

        self._bind_program_hooks()
        self._apply_program_embedding_prefixes()
        self._hydrate_indexed_cids_from_db()

    def _candidate_cache_namespace(self) -> str:
        profile = getattr(self.embedder, "profile", None)
        profile_identity = asdict(profile) if is_dataclass(profile) else str(profile)
        if isinstance(profile_identity, dict):
            # Query prefixes belong to query-vector keys, not document encoding
            # or graph stages. Document prefix/model/endpoint remain dependencies.
            profile_identity.pop("query_prefix", None)
            profile_identity.pop("fact_query_prefix", None)
        payload = {
            "codec": 1,
            "implementation": self._candidate_cache_code_fp,
            "config": self.config.config_fingerprint() if self.config is not None else None,
            "embedder": {
                "class": type(self.embedder).__module__ + "." + type(self.embedder).__qualname__,
                "profile": profile_identity,
                "kwargs": getattr(self.embedder, "default_kwargs", {}),
                **{
                    name: getattr(self.embedder, name, None)
                    for name in (
                        "base_url",
                        "provider",
                        "task_mode",
                        "dim",
                    )
                },
            },
            "gbnf": {
                name: value
                for name, value in vars(self.program).items()
                if name.startswith("gbnf_") or name == "enable_gbnf"
            },
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
        ).hexdigest()

    def _candidate_cache_sources(self) -> dict[str, str]:
        return {hashlib.sha256(text.encode("utf-8")).hexdigest(): text for _, text in self._opt_chunks_pool}

    def _graph_cache_context(self, conn: sqlite3.Connection | None = None) -> str:
        """Hash the graph visible to SHACL's inspection/repair tools, in bounded rows."""
        active_conn = conn if conn is not None else self.conn
        digest = hashlib.sha256()
        for table, order in (
            ("entities", "entity_id"),
            ("facts", "fact_id"),
            ("fact_sources", "fact_id, chunk_id"),
            ("edges", "src_id, dst_id, kind"),
            ("entity_labels", "label_norm, entity_id, label_kind, chunk_id"),
            ("edge_sources", "src_id, dst_id, kind, chunk_id"),
            ("chunks", "chunk_id"),
        ):
            digest.update(table.encode())
            for row in active_conn.execute(f"SELECT * FROM {table} ORDER BY {order}"):
                digest.update(json.dumps(tuple(row), ensure_ascii=False).encode("utf-8"))
        digest.update(json.dumps(sorted(self._candidate_cache_sources())).encode())
        return digest.hexdigest()

    def _hydrate_indexed_cids_from_db(
        self,
        program: HippoRAGProgram | None = None,
    ) -> None:
        """Populate _opt_indexed_cids, _chunk_indexed_ext_fp, and persisted DSPy trails from SQLite."""
        active_prog = program if program is not None else self.program
        provenance: dict[str, list[str]] = {}
        rows = self.conn.execute(
            "SELECT c.chunk_id, m.value, p.value FROM chunks c "
            "INNER JOIN openie_chunks o ON c.chunk_id = o.chunk_id "
            "INNER JOIN meta m ON m.key = 'indexed_fp:' || c.chunk_id "
            "INNER JOIN meta p ON p.key = 'openie_fp:' || c.chunk_id"
        ).fetchall()
        for cid, raw, saved_openie_fp in rows:
            recorded = json.loads(raw)
            if recorded[1] == saved_openie_fp:
                provenance[str(cid)] = recorded
        if (
            self.conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunk_inference' LIMIT 1"
            ).fetchone()
            is not None
        ):
            curr_ext_fp = extraction_fingerprint(active_prog)
            curr_openie_fp = openie_fingerprint(active_prog)
            inf_rows = self.conn.execute(
                "SELECT c.chunk_id, i.program_identity FROM chunks c "
                "INNER JOIN chunk_inference i ON c.chunk_id = i.chunk_id"
            ).fetchall()
            for cid, prog_id in inf_rows:
                cid_str = str(cid)
                if cid_str not in provenance:
                    provenance[cid_str] = [curr_ext_fp, str(prog_id or curr_openie_fp)]
        seeded_cids = list(provenance)
        if not seeded_cids:
            return
        vec_cids = get_existing_ids(self.conn, "chunk_vec", seeded_cids)
        ready_cids = set(seeded_cids) & vec_cids
        if ready_cids:
            self._opt_indexed_cids.update(ready_cids)
            for cid in ready_cids:
                self._chunk_indexed_ext_fp[cid] = provenance[cid][0]
            if self._active_extraction_fingerprint is None:
                self._active_extraction_fingerprint = provenance[next(iter(ready_cids))][0]
            if self._active_openie_fingerprint is None:
                self._active_openie_fingerprint = provenance[next(iter(ready_cids))][1]

            raw_trails = get_meta(self.conn, "dspy_trails_state")
            if raw_trails:
                state_obj = json.loads(raw_trails)
                if isinstance(state_obj, dict):
                    g_stats = state_obj.get("graph_rlm_stats")
                    if isinstance(g_stats, dict):
                        for k, v in g_stats.items():
                            if k in self._graph_rlm_stats and not self._graph_rlm_stats.get(k):
                                self._graph_rlm_stats[k] = v
                    c_stats = state_obj.get("chunk_rlm_stats")
                    if isinstance(c_stats, dict):
                        for cid_k, c_val in c_stats.items():
                            if isinstance(c_val, dict) and cid_k not in self._chunk_rlm_stats:
                                self._chunk_rlm_stats[str(cid_k)] = {str(sk): float(sv) for sk, sv in c_val.items()}
                    for trail_attr in (
                        "telemetry",
                        "embedding_audit_trail",
                        "label_judge_audit_trail",
                        "hierarchy_judge_audit_trail",
                        "related_judge_audit_trail",
                        "shacl_audit_trail",
                        "entity_def_audit_trail",
                    ):
                        saved_list = state_obj.get(trail_attr)
                        curr_list = getattr(active_prog, trail_attr, None)
                        if (
                            isinstance(saved_list, list)
                            and saved_list
                            and isinstance(curr_list, list)
                            and not curr_list
                        ):
                            curr_list.extend(saved_list)
            else:
                rel_cnt_row = self.conn.execute("SELECT count(*) FROM edges WHERE kind = 'skos:related'").fetchone()
                diff_cnt_row = self.conn.execute(
                    "SELECT count(*) FROM edges WHERE kind = 'owl:differentFrom'"
                ).fetchone()
                if rel_cnt_row and not self._graph_rlm_stats.get("related_edges"):
                    self._graph_rlm_stats["related_edges"] = int(rel_cnt_row[0]) // 2
                if diff_cnt_row and not self._graph_rlm_stats.get("different_from_edges"):
                    self._graph_rlm_stats["different_from_edges"] = int(diff_cnt_row[0]) // 2
                self._persist_dspy_trails_to_db(active_prog)

    def _persist_dspy_trails_to_db(
        self,
        program: HippoRAGProgram | None = None,
    ) -> None:
        """Persist graph RLM stats, per-chunk RLM stats, and DSPy audit trails into SQLite meta."""
        active_prog = program if program is not None else self.program
        payload = {
            "graph_rlm_stats": dict(self._graph_rlm_stats),
            "chunk_rlm_stats": dict(self._chunk_rlm_stats),
            **(active_prog.dump_audit_trails(bounded=True) if hasattr(active_prog, "dump_audit_trails") else {}),
        }
        set_meta(self.conn, "dspy_trails_state", json.dumps(payload, ensure_ascii=False))

    def _hydrate_chunk_traces_and_stats_from_db(
        self,
        target_list: Sequence[tuple[ChunkPointer, str]],
    ) -> None:
        """Restore actual named DSPy traces from native inference target rows."""
        wanted = {p.chunk_id for p, _ in target_list}
        if not self.conn.execute("SELECT 1 FROM sqlite_master WHERE name='chunk_inference'").fetchone():
            return
        for cid, raw in self.conn.execute("SELECT chunk_id, traces_json FROM chunk_inference"):
            if cid in wanted:
                self._captured_chunk_traces[cid] = [
                    (trace["predictor"], trace["inputs"], dspy.Prediction(**trace["outputs"]))
                    for trace in json.loads(raw)
                ]
                self._chunk_rlm_stats[cid] = {"openie_tokens": 0.0}

    @staticmethod
    def _empty_graph_rlm_stats() -> dict[str, Any]:
        return {
            "homograph_splits": 0,
            "candidate_pairs": 0,
            "merged_entities": 0,
            "related_edges": 0,
            "broader_edges": 0,
            "different_from_edges": 0,
            "shacl_violations_before": 0,
            "shacl_violations_after": 0,
            "shacl_violations_initial": 0,
            "shacl_violations_remaining": 0,
            "shacl_repairs_applied": 0,
            "shacl_conforms": True,
            "label_judge_tokens": 0,
            "hierarchy_judge_tokens": 0,
            "related_judge_tokens": 0,
            "shacl_healer_tokens": 0,
            "missing_entity_defs_count": 0,
            "missing_entity_defs_tokens": 0,
            "embedding_calls": 0,
            "embedding_tokens": 0,
        }

    def _hydrate_single_chunk(self, chunk_id: str) -> str:
        hydrator = make_chunk_hydrator(self.conn, self.workspace_root)
        return hydrator(chunk_id)

    def _hydrate_single_ast_island(self, chunk_id: str) -> str:
        hydrator = make_ast_island_hydrator(self.conn, self.workspace_root)
        return hydrator(chunk_id)

    def _on_embedding_telemetry(self, entry: Mapping[str, Any]) -> None:
        self._graph_rlm_stats["embedding_calls"] += 1
        self._graph_rlm_stats["embedding_tokens"] += int(entry.get("tokens_used", 0))
        if hasattr(self.program, "record_embedding_telemetry"):
            self.program.record_embedding_telemetry(entry)

    def _bind_program_hooks(self, program: HippoRAGProgram | None = None) -> None:
        target = program if program is not None else self.program
        target_rm = getattr(target, "rm", None)
        if target_rm is not None:
            target_rm.default_k = self.retrieval_top_k
        target_ret = getattr(target, "retriever", None)
        if target_ret is not None:
            target_ret.k = self.retrieval_top_k
        if hasattr(target, "set_retrieval_hook"):
            target.set_retrieval_hook(self._canonical_retrieval_hook)
        if hasattr(target, "set_graph_tools"):
            target.set_graph_tools(
                node_inspector=lambda eid: inspect_focus_node(
                    self.conn,
                    eid,
                    corpus_root=self.workspace_root,
                    ast_context_hops=self.ast_context_hops,
                ),
                pair_inspector=lambda a, b: format_pair_disambiguation_dossier(
                    self.conn,
                    a,
                    b,
                    corpus_root=self.workspace_root,
                    ast_context_hops=self.ast_context_hops,
                ),
                chunk_hydrator=self._hydrate_single_chunk,
                ast_island_hydrator=self._hydrate_single_ast_island,
                shacl_linter=lambda: [v.to_dict() for v in lint_graph_shacl(self.conn)],
            )
        if hasattr(target, "set_graph_search_tool"):
            target.set_graph_search_tool(self._search_graph_for_react)
        if hasattr(self.embedder, "bind_telemetry_hook"):
            self.embedder.bind_telemetry_hook(self._on_embedding_telemetry)
        if target is self.program:
            loaded_path = getattr(self.program, "loaded_from_path", None)
            self.ensemble = compile_pareto_ensemble(self.program, program_path=loaded_path)
            for ep in getattr(self.ensemble, "programs", ()):
                if ep is not self.program:
                    self._bind_program_hooks(ep)

    def refresh_program_bindings(self) -> None:
        """Re-bind runtime hooks, Pareto ensemble siblings, and embedding prefixes after program load."""
        self._query_retrieval_memo.clear()
        self._bind_program_hooks()
        self._apply_program_embedding_prefixes()

    def _canonical_retrieval_hook(
        self,
        question: str,
        program: HippoRAGProgram | None = None,
        *,
        item_docs: Sequence[str] | None = None,
        k: int | None = None,
    ) -> tuple[str, tuple[str, ...], tuple[float, ...], tuple[str, ...]]:
        """Canonical engine retrieval hook returning (formatted_context, hydrated_texts, scores, chunk_ids)."""
        active_program = program if program is not None else self.program
        self._last_hook_program = active_program
        qa_top_k = max(1, int(k if k is not None else self.retrieval_top_k))
        q_str = str(question or "")
        memo_key = (q_str, qa_top_k, query_retrieval_fingerprint(active_program))

        if self._in_optimization:
            eff_item_docs = item_docs or self._opt_docs_by_question.get(q_str.strip())
            target_cids = self._ensure_candidate_chunks_indexed(active_program, item_docs=eff_item_docs)
            active_trace = getattr(dspy.settings, "trace", None)
            if isinstance(active_trace, list):
                named_map = dict(active_program.named_predictors())
                for cid in target_cids:
                    for pred_name, inp_dict, out_obj in self._captured_chunk_traces.get(cid, ()):
                        pred_inst = named_map.get(pred_name)
                        if pred_inst is not None:
                            active_trace.append((pred_inst, dict(inp_dict), out_obj))
                for pred_name, inp_dict, out_obj in self._captured_graph_traces:
                    pred_inst = named_map.get(pred_name)
                    if pred_inst is not None:
                        active_trace.append((pred_inst, dict(inp_dict), out_obj))
        else:
            if not self._opt_indexed_cids:
                self._hydrate_indexed_cids_from_db(self.program)
            active_trace = getattr(dspy.settings, "trace", None)
            if isinstance(active_trace, list) and bool(item_docs):
                req_chunks = self._resolve_chunks_for_docs(item_docs)
                target_list = req_chunks if req_chunks else list(self._opt_chunks_pool)
                self._hydrate_chunk_traces_and_stats_from_db(target_list)
                named_map = dict(active_program.named_predictors())
                for ptr, _ in target_list:
                    for pred_name, inp_dict, out_obj in self._captured_chunk_traces.get(ptr.chunk_id, ()):
                        pred_inst = named_map.get(pred_name)
                        if pred_inst is not None:
                            active_trace.append((pred_inst, dict(inp_dict), out_obj))
                for pred_name, inp_dict, out_obj in self._captured_graph_traces:
                    pred_inst = named_map.get(pred_name)
                    if pred_inst is not None:
                        active_trace.append((pred_inst, dict(inp_dict), out_obj))
            if not bool(item_docs):
                if q_str != self._last_memo_question:
                    self._query_retrieval_memo.clear()
                    self._last_memo_question = q_str
                cached_ret = self._query_retrieval_memo.get(memo_key)
                if cached_ret is not None:
                    return cached_ret

        fetch_top_k = max(20, qa_top_k)
        if self._active_progress_callback is not None:
            q_preview = q_str.strip().replace("\n", " ")[:48]
            self._active_progress_callback(f"retrieving + QA: {q_preview!r}...")
        hits, kept_facts = self.retrieve_with_seeds(
            question,
            top_k=fetch_top_k,
            program=active_program,
        )
        if not hits:
            empty_res: tuple[str, tuple[str, ...], tuple[float, ...], tuple[str, ...]] = ("", (), (), ())
            if not self._in_optimization and not bool(item_docs):
                self._query_retrieval_memo[memo_key] = empty_res
            return empty_res
        texts = tuple(self._format_hydrated_passage(p, text) for p, _, text in hits)
        scores = tuple(float(s) for _, s, _ in hits)
        pids = tuple(p.chunk_id for p, _, _ in hits)
        formatted_context = self.format_qa_context(hits[:qa_top_k], seed_facts=kept_facts)
        res = (formatted_context, texts, scores, pids)
        if not self._in_optimization and not bool(item_docs):
            self._query_retrieval_memo[memo_key] = res
        return res

    def as_rm(
        self,
        k: int | None = None,
        *,
        configure_dspy: bool = False,
    ) -> YawRM:
        "Return a first-class `YawRM` callable bound to this engine (optionally registering `dspy.configure(rm=...)`)."
        eff_k = int(k if k is not None else self.retrieval_top_k)
        rm = YawRM(
            hook=self._canonical_retrieval_hook,
            program=self.program,
            default_k=eff_k,
        )
        if configure_dspy:
            dspy.configure(rm=rm)
        return rm

    def as_retriever(
        self,
        k: int | None = None,
        *,
        configure_dspy: bool = False,
    ) -> YawRetriever:
        """Return a first-class `YawRetriever(dspy.Retrieve)` bound to this engine (optionally registering \
`dspy.configure(rm=...)`)."""
        rm = self.as_rm(k=k, configure_dspy=configure_dspy)
        return YawRetriever(k=rm.default_k, rm=rm)

    def _search_graph_for_react(self, query: str) -> str:
        """Execute hybrid dense + FTS fact and entity lookup in `yaw.db` for `dspy.ReActV2` multi-hop exploration."""
        clean_q = str(query or "").strip()
        if not clean_q:
            return ""
        with self._lock:
            query_fact_vec = self._encode_texts([clean_q], instruction=FACT_INSTRUCTION, stage="react_fact_embed")[0]
            dense_fact_hits = search_fact_vectors_knn(
                self.conn, query_fact_vec, top_k=min(10, self.fact_candidate_pool_k)
            )
            fts_fact_hits = search_facts_fts(self.conn, clean_q, top_k=min(10, self.fact_fts_pool_k))
            ordered_fids: list[str] = []
            seen_fids: set[str] = set()
            for fid, _ in [*fts_fact_hits, *dense_fact_hits]:
                if fid not in seen_fids:
                    seen_fids.add(fid)
                    ordered_fids.append(fid)
                if len(ordered_fids) >= 8:
                    break

            fact_lines: list[str] = []
            if ordered_fids:
                triples_map = get_facts_by_ids(self.conn, ordered_fids)
                placeholders = ",".join("?" for _ in ordered_fids)
                def_rows = {
                    str(r[0]): str(r[1] or "")
                    for r in self.conn.execute(
                        f"SELECT fact_id, COALESCE(definition, '') FROM facts WHERE fact_id IN ({placeholders})",
                        ordered_fids,
                    ).fetchall()
                }
                for fid in ordered_fids:
                    triple = triples_map.get(fid)
                    if triple is not None:
                        s, p, o = triple
                        f_def = def_rows.get(fid, "").strip()
                        if f_def:
                            fact_lines.append(f"- ({s}, {p}, {o}): {f_def}")
                        else:
                            fact_lines.append(f"- ({s}, {p}, {o})")

            query_passage_vec = self._encode_texts(
                [clean_q], instruction=PASSAGE_INSTRUCTION, stage="react_passage_embed"
            )[0]
            chunk_vec_hits = search_chunk_vectors_knn(self.conn, query_passage_vec, top_k=5)
            chunk_fts_hits = search_chunks_fts(self.conn, clean_q, top_k=5)
            merged_chunks: list[tuple[str, float]] = []
            seen_cids: set[str] = set()
            for cid, sc in [*chunk_fts_hits, *chunk_vec_hits]:
                if cid not in seen_cids:
                    seen_cids.add(cid)
                    merged_chunks.append((cid, float(sc)))
            hydrated = self._hydrate_maxp_passages(merged_chunks, top_k=3)
            passage_lines = [f"[{ptr.chunk_id}] {self._format_hydrated_passage(ptr, txt)}" for ptr, _, txt in hydrated]

            sections: list[str] = []
            if fact_lines:
                sections.append("Facts:\n" + "\n".join(fact_lines))
            if passage_lines:
                sections.append("Passages:\n" + "\n\n".join(passage_lines))
            return "\n\n".join(sections) if sections else "No matching graph facts or passages found."

    def _apply_program_embedding_prefixes(self, program: HippoRAGProgram | None = None) -> None:
        target = program if program is not None else self.program
        if hasattr(self.embedder, "update_prefixes"):
            self.embedder.update_prefixes(
                query_prefix=getattr(target, "embedding_query_prefix", None),
                fact_query_prefix=getattr(target, "embedding_fact_prefix", None),
                document_prefix=getattr(target, "embedding_document_prefix", None),
            )
        profile = getattr(self.embedder, "profile", None)
        if profile is not None:
            if getattr(target, "embedding_query_prefix", None) is None:
                target.embedding_query_prefix = getattr(profile, "query_prefix", None)
            if getattr(target, "embedding_fact_prefix", None) is None:
                target.embedding_fact_prefix = getattr(profile, "fact_query_prefix", None)
            if getattr(target, "embedding_document_prefix", None) is None:
                target.embedding_document_prefix = getattr(profile, "document_prefix", None)

    def close(self) -> None:
        """Close SQLite database connection and release any open DSPy disk cache handles."""
        with self._lock:
            active_cache = getattr(dspy, "cache", None)
            disk_cache = getattr(active_cache, "disk_cache", None) if active_cache is not None else None
            close_fn = getattr(disk_cache, "close", None)
            if callable(close_fn):
                close_fn()
            if self.managed_conn is not None:
                self.managed_conn.close()

    def get_stats(self, config_fingerprint: str = "") -> IndexStats:
        """Return current graph index row counts, SKOS label counts, SHACL conformance, and fingerprint."""
        with self._lock:
            chunks, entities, facts, edges, openie = get_graph_counts(self.conn)
            label_row = self.conn.execute("SELECT COUNT(*) FROM entity_labels").fetchone()
            entity_labels = int(label_row[0]) if label_row else 0
            violations = lint_graph_shacl(self.conn)
            return IndexStats(
                chunks=chunks,
                entities=entities,
                facts=facts,
                edges=edges,
                config_fingerprint=config_fingerprint,
                openie_checkpoints=openie,
                entity_labels=entity_labels,
                shacl_conforms=len(violations) == 0,
            )

    def lint_graph(
        self,
        *,
        as_turtle: bool = False,
    ) -> dict[str, Any]:
        """Run closed-world SHACL linting (and optional Turtle serialization) over yaw.db."""
        with self._lock:
            violations = lint_graph_shacl(self.conn)
            report: dict[str, Any] = {
                "conforms": len(violations) == 0,
                "violation_count": len(violations),
                "violations": [v.to_dict() for v in violations],
                "stats": self.get_stats().to_dict(),
            }
            if as_turtle:
                report["turtle"] = serialize_graph_turtle(self.conn, violations)
            return report

    def get_meta(self, key: str) -> str | None:
        """Read a key from the yaw.db meta table."""
        with self._lock:
            return get_meta(self.conn, key)

    def set_meta(self, key: str, value: str | None) -> None:
        """Write or delete (when value is None) a key in the yaw.db meta table."""
        with self._lock:
            set_meta(self.conn, key, value)

    def _ensure_csr_graph(self) -> CSRGraph:
        with self._lock:
            if self._csr_graph is None:
                chunk_ids, entity_ids, edges, entity_priors = load_graph_topology(self.conn)
                self._csr_graph = build_transition_matrix(
                    chunk_ids,
                    entity_ids,
                    edges,
                    beta=self.beta,
                    policy=self.traversal_policy or RelationTraversalPolicy(),
                    node_priors=entity_priors,
                )
            return self._csr_graph

    def _encode_texts(
        self,
        texts: Sequence[str],
        *,
        instruction: str = "",
        stage: str | None = None,
    ) -> Any:
        if not texts:
            return []
        if hasattr(self.embedder, "bind_telemetry_hook"):
            embed_fp = embed_fingerprint(getattr(self.embedder, "profile", None) or self.program, instruction)
            missing_texts: list[str] = []
            missing_keys: set[str] = set()
            vectors: dict[str, np.ndarray] = {}
            for t in texts:
                ts = str(t)
                if ts in vectors or ts in missing_keys:
                    continue
                cached = self._text_vec_cache.get((embed_fp, instruction, ts))
                if cached is None:
                    missing_texts.append(ts)
                    missing_keys.add(ts)
                else:
                    vectors[ts] = cached
            if missing_texts:
                encoded = self.embedder.encode(missing_texts, instruction=instruction, stage=stage)
                for ts, vec in zip(missing_texts, encoded, strict=False):
                    vectors[ts] = np.asarray(vec, dtype=np.float32)
                    self._text_vec_cache[(embed_fp, instruction, ts)] = vectors[ts]
            return [vectors[str(t)] for t in texts]
        return self.embedder.encode(texts, instruction=instruction)

    def _encode_in_batches(
        self,
        texts: Sequence[str],
        instruction: str = "",
        *,
        stage: str | None = None,
    ) -> list[np.ndarray]:
        """Encode texts in bounded micro-batches to prevent HTTP/VRAM OOM on large corpora."""
        if not texts:
            return []
        results: list[np.ndarray] = []
        step = max(1, self.embed_batch_size)
        for i in range(0, len(texts), step):
            batch = texts[i : i + step]
            results.extend(self._encode_texts(batch, instruction=instruction, stage=stage))
        return results

    @staticmethod
    def _format_hydrated_passage(p: ChunkPointer, text: str) -> str:
        if not p.title:
            return text
        if text.startswith(f"{p.title}\n") or text == p.title:
            return f"Wikipedia Title: {text}"
        return f"Wikipedia Title: {p.title}\n{text}"

    def format_qa_context(
        self,
        hits: Sequence[tuple[ChunkPointer, float, str]],
        *,
        seed_facts: Sequence[tuple[str, str, str]] = (),
        include_subgraph: bool = True,
        max_context_chars: int | None = None,
        max_characters: int | None = None,
    ) -> str:
        """Pack induced SKOS/OWL/RDF subgraph header and top-k hydrated AST islands into QA context."""
        if not hits:
            return ""
        budget = (
            max_characters
            if max_characters is not None
            else (max_context_chars if max_context_chars is not None else self.max_context_chars)
        )
        subgraph_header = ""
        passage_budget = budget
        if include_subgraph and self.conn is not None:
            max_subgraph = min(6000, max(0, budget // 4))
            chunk_ids = [p.chunk_id for p, _, _ in hits]
            subgraph_header = format_subgraph_for_qa(
                self.conn,
                chunk_ids=chunk_ids,
                seed_facts=seed_facts,
                max_subgraph_chars=max_subgraph,
            )
            if subgraph_header:
                passage_budget = max(1000, budget - len(subgraph_header) - 2)

        ptrs_scores = [(p, s) for p, s, _ in hits]
        hydrated = hydrate_budgeted_passages(
            ptrs_scores,
            corpus_root=self.workspace_root,
            max_context_chars=passage_budget,
            max_expansion_hops=self.max_ast_expansion_hops,
        )
        blocks: list[str] = []
        if subgraph_header:
            blocks.append(subgraph_header)
            blocks.append("### Retrieved Passages")
        for hp in hydrated:
            formatted = self._format_hydrated_passage(hp.pointer, hp.text)
            if formatted:
                blocks.append(formatted)
        return "\n\n".join(blocks)

    _apply_shacl_repairs = staticmethod(apply_shacl_repairs)

    def _resolve_chunks_for_docs(
        self,
        item_docs: Sequence[str] | None,
    ) -> list[tuple[ChunkPointer, str]]:
        if not item_docs:
            return list(self._opt_chunks_pool)
        pool_len = len(self._opt_chunks_pool)
        pool_by_cid = {ptr.chunk_id: (ptr, txt) for ptr, txt in self._opt_chunks_pool}
        resolved: list[tuple[ChunkPointer, str]] = []
        seen_cids: set[str] = set()
        for raw_doc in item_docs:
            d = str(raw_doc).strip()
            if not d:
                continue
            cache_key = (pool_len, d)
            if cache_key in self._resolved_docs_cache:
                for ptr, c_text in self._resolved_docs_cache[cache_key]:
                    if ptr.chunk_id not in seen_cids:
                        seen_cids.add(ptr.chunk_id)
                        resolved.append((ptr, c_text))
                continue
            doc_matches: list[tuple[ChunkPointer, str]] = []
            direct_cid = d if d in pool_by_cid else compute_mdhash_id(d, prefix="chunk-")
            if direct_cid in pool_by_cid:
                doc_matches.append(pool_by_cid[direct_cid])
            else:
                doc_title = d.split("\n", 1)[0].strip() if "\n" in d else ""
                doc_body = d.split("\n", 1)[1].strip() if "\n" in d else d
                for ptr, c_text in self._opt_chunks_pool:
                    c_strip = c_text.strip()
                    if not c_strip:
                        continue
                    c_title = ptr.title.strip() if ptr.title else ""
                    c_body = (
                        c_strip[len(c_title) + 1 :].strip()
                        if c_title and c_strip.startswith(f"{c_title}\n")
                        else c_strip
                    )
                    title_match = bool(doc_title and c_title and c_title == doc_title)
                    body_match = (
                        c_strip in (d, doc_body)
                        or (bool(c_body) and c_body in (d, doc_body))
                        or (len(c_strip) >= 30 and c_strip in d)
                        or (len(c_body) >= 30 and (c_body in d or c_body in doc_body))
                        or (
                            title_match
                            and (
                                c_strip in doc_body
                                or doc_body in c_strip
                                or (bool(c_body) and (c_body in doc_body or doc_body in c_body))
                            )
                        )
                    )
                    if body_match:
                        doc_matches.append((ptr, c_text))
            self._resolved_docs_cache[(pool_len, d)] = doc_matches
            for ptr, c_text in doc_matches:
                if ptr.chunk_id not in seen_cids:
                    seen_cids.add(ptr.chunk_id)
                    resolved.append((ptr, c_text))
        return resolved

    def _ensure_candidate_chunks_indexed(
        self,
        program: HippoRAGProgram,
        item_docs: Sequence[str] | None = None,
    ) -> list[str]:
        with self._lock:
            ext_fp = extraction_fingerprint(program)
            if self._active_extraction_fingerprint != ext_fp or any(
                p.chunk_id not in self._opt_indexed_cids for p, _ in self._opt_chunks_pool
            ):
                self.index_chunks(
                    self._opt_chunks_pool, program=program, progress_callback=self._active_progress_callback
                )
            target_list = self._resolve_chunks_for_docs(item_docs) if item_docs else list(self._opt_chunks_pool)
            self._hydrate_chunk_traces_and_stats_from_db(target_list)
            target_cids = [p.chunk_id for p, _ in target_list]

            if target_cids and any(cid in self._chunk_rlm_stats for cid in target_cids):
                agg: dict[str, float] = {
                    "label_judge_tokens": 0.0,
                    "hierarchy_judge_tokens": 0.0,
                    "related_judge_tokens": 0.0,
                    "shacl_healer_tokens": 0.0,
                    "shacl_violations_initial": 0.0,
                    "shacl_violations_remaining": 0.0,
                    "missing_entity_defs_count": 0.0,
                    "missing_entity_defs_tokens": 0.0,
                    "openie_tokens": 0.0,
                    "cached_openie_tokens": 0.0,
                }
                for cid in target_cids:
                    c_stats = self._chunk_rlm_stats.get(cid)
                    if c_stats:
                        for k in (
                            "label_judge_tokens",
                            "hierarchy_judge_tokens",
                            "related_judge_tokens",
                            "shacl_healer_tokens",
                            "shacl_violations_initial",
                            "shacl_violations_remaining",
                            "missing_entity_defs_count",
                            "missing_entity_defs_tokens",
                            "openie_tokens",
                        ):
                            agg[k] += float(c_stats.get(k, 0.0))
                        agg["cached_openie_tokens"] += float(c_stats.get("openie_tokens", 0.0))
                q_stats = {k: int(round(v)) for k, v in agg.items()}
                self._active_query_rlm_stats = q_stats
                self._last_scope_by_cids[tuple(target_cids)] = q_stats
            else:
                self._active_query_rlm_stats = None
            return target_cids

    def index(
        self,
        docs: Sequence[str],
        *,
        progress_callback: Callable[[str], None] | None = None,
    ) -> int:
        """Raw document string ingestion adapter for CLI benchmarks and quick experimentation."""
        if not docs:
            return 0

        with self._lock:
            unique_docs: list[str] = []
            seen_docs: set[str] = set()
            for d in docs:
                if d and d not in seen_docs:
                    seen_docs.add(d)
                    unique_docs.append(d)
            if not unique_docs:
                return 0

            chunk_ids = [compute_mdhash_id(d, prefix="chunk-") for d in unique_docs]
            existing_chunks = get_existing_ids(self.conn, "chunk", chunk_ids)
            existing_vecs = get_existing_ids(self.conn, "chunk_vec", chunk_ids)

            incomplete_cids = [cid for cid in chunk_ids if cid in existing_chunks and cid not in existing_vecs]
            new_docs = [d for d, cid in zip(unique_docs, chunk_ids, strict=False) if cid not in existing_chunks]
            if not new_docs and not incomplete_cids:
                return 0

            chunks_to_index: list[tuple[ChunkPointer, str]] = []
            if incomplete_cids:
                existing_ptrs = get_chunk_pointers(self.conn, incomplete_cids)
                doc_by_cid = dict(zip(chunk_ids, unique_docs, strict=False))
                for cid in incomplete_cids:
                    if cid in existing_ptrs and cid in doc_by_cid:
                        chunks_to_index.append((existing_ptrs[cid], doc_by_cid[cid]))

            if new_docs:
                passages_path = self.workspace_root / "passages.json"
                existing_records: list[dict[str, Any]] = []
                if passages_path.is_file():
                    loaded = json.loads(passages_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, list):
                        existing_records = [r for r in loaded if isinstance(r, dict)]

                all_records = list(existing_records)
                seen_rec_ids = {str(r.get("id", "")) for r in all_records}
                new_doc_items: list[tuple[str, str, str]] = []
                for doc in new_docs:
                    cid = compute_mdhash_id(doc, prefix="chunk-")
                    first_line = doc.split("\n", 1)[0].strip()
                    title = first_line if len(first_line) < 100 else ""
                    new_doc_items.append((cid, title, doc))
                    if cid not in seen_rec_ids:
                        all_records.append({"id": cid, "title": title, "text": doc})
                        seen_rec_ids.add(cid)

                rendered_items: list[str] = []
                span_by_cid: dict[str, tuple[int, int]] = {}
                curr_line = 2
                for rec in all_records:
                    item_str = "\n".join("  " + ln for ln in json.dumps(rec, ensure_ascii=False, indent=2).splitlines())
                    rendered_items.append(item_str)
                    s_line = curr_line
                    e_line = s_line + item_str.count("\n")
                    rec_id = str(rec.get("id", ""))
                    if rec_id:
                        span_by_cid[rec_id] = (s_line, e_line)
                    curr_line = e_line + 1

                full_json = "[\n" + ",\n".join(rendered_items) + "\n]\n"
                self.workspace_root.mkdir(parents=True, exist_ok=True)
                write_text_atomic(passages_path, full_json)

                for cid, title, doc in new_doc_items:
                    s_line, e_line = span_by_cid[cid]
                    ptr = ChunkPointer(
                        chunk_id=cid,
                        rel_posix_path="passages.json",
                        start_line=s_line,
                        end_line=e_line,
                        title=title,
                        ast_island_start=s_line,
                        ast_island_end=e_line,
                    )
                    chunks_to_index.append((ptr, doc))

            return self.index_chunks(chunks_to_index, progress_callback=progress_callback)

    def index_chunks(
        self,
        chunks: Sequence[tuple[ChunkPointer, str]],
        *,
        progress_callback: Callable[[str], None] | None = None,
        program: HippoRAGProgram | None = None,
    ) -> int:
        """Publish an explicit candidate snapshot through the bound native runtime."""
        if not chunks:
            return 0
        if self.indexing_runner is None:
            raise RuntimeError("Indexing requires daemon composition with bind_native_indexing")
        active_program = program if program is not None else self.program
        with self._lock:
            self._bind_program_hooks(active_program)
            self._apply_program_embedding_prefixes(active_program)
            by_id = {pointer.chunk_id: (pointer, text) for pointer, text in self._opt_chunks_pool}
            by_id.update({pointer.chunk_id: (pointer, text) for pointer, text in chunks})
            self._opt_chunks_pool = list(by_id.values())
            if progress_callback is not None:
                progress_callback("indexing: native frozen candidate preparation and graph publication")
            summary = self.indexing_runner(self._opt_chunks_pool, active_program)
            self._csr_graph = None
            self._query_retrieval_memo.clear()
            self._last_memo_question = None
            self._active_extraction_fingerprint = extraction_fingerprint(active_program)
            self._active_openie_fingerprint = openie_fingerprint(active_program)
            self._opt_indexed_cids = set(summary.chunk_ids)
            self._chunk_indexed_ext_fp = {cid: self._active_extraction_fingerprint for cid in summary.chunk_ids}
            self._graph_rlm_stats.update(summary.rlm_stats)
            self._captured_graph_traces = [
                (trace.predictor, dict(trace.inputs), dspy.Prediction(**trace.outputs))
                for trace in summary.traces
                if trace.predictor not in {"ner", "triples"}
            ]
            self._captured_chunk_traces.clear()
            self._hydrate_chunk_traces_and_stats_from_db(self._opt_chunks_pool)
            for name in (
                "entity_def_audit_trail",
                "label_judge_audit_trail",
                "hierarchy_judge_audit_trail",
                "related_judge_audit_trail",
                "shacl_audit_trail",
            ):
                getattr(active_program, name).extend(getattr(summary, name))
            trim_diagnostics(active_program)
            return len(chunks)

    def delete_chunks(self, chunk_ids: Sequence[str]) -> None:
        """Atomically delete chunks and cascade through affected edges, facts, and entities."""
        if not chunk_ids:
            return
        with self._lock:
            delete_chunks(self.conn, chunk_ids)
            with self.managed_conn.transaction():
                for cid in chunk_ids:
                    set_meta(self.conn, f"indexed_fp:{cid}", None)
                    set_meta(self.conn, f"openie_fp:{cid}", None)
            for cid in chunk_ids:
                self._opt_indexed_cids.discard(cid)
                self._chunk_indexed_ext_fp.pop(cid, None)
            self._csr_graph = None
            self._query_retrieval_memo.clear()
            self._last_memo_question = None

    def retrieve_with_seeds(
        self,
        query: str,
        top_k: int = 10,
        *,
        allow_intent_path: bool = True,
        program: HippoRAGProgram | None = None,
    ) -> tuple[list[tuple[ChunkPointer, float, str]], list[tuple[str, str, str]]]:
        """Retrieve top-k chunks and return both hydrated hits and kept recognition-memory facts."""
        active_program = program if program is not None else self.program
        with self._lock:
            if allow_intent_path and str(query or "") != self._last_seed_query:
                self._last_query_seeds = []
                self._last_seed_query = str(query or "")
            graph = self._ensure_csr_graph()
            if len(graph.chunk_indices) == 0 or top_k <= 0:
                return [], []

            # 1. Pluggable Hybrid Fact Candidate Retrieval (Dense KNN + FTS5 + RRF)
            # Step 1: Retrieve top-k dense vector hits from fact_vec
            query_fact_vec = self._encode_texts([query], instruction=FACT_INSTRUCTION, stage="query_fact_embed")[0]
            dense_hits = search_fact_vectors_knn(self.conn, query_fact_vec, top_k=self.fact_candidate_pool_k)

            # Step 2: Retrieve top-k lexical FTS hits from fact_fts (plus query_intent expansion)
            fts_hits = search_facts_fts(self.conn, query, top_k=self.fact_fts_pool_k)
            intent_query = active_program.extract_query_intent(query) if allow_intent_path else query
            intent_fts_hits = (
                search_facts_fts(self.conn, intent_query, top_k=self.fact_fts_pool_k)
                if (allow_intent_path and intent_query and intent_query.strip().lower() != query.strip().lower())
                else []
            )

            # Step 3: RRF merge -> max (fact_candidate_pool_k + fact_fts_pool_k) candidate list length
            rrf_scores: dict[str, float] = {}
            for rank, (fid, _) in enumerate(dense_hits, start=1):
                rrf_scores[fid] = rrf_scores.get(fid, 0.0) + (1.0 / (self.fact_rrf_k + rank))
            for rank, (fid, _) in enumerate(fts_hits, start=1):
                rrf_scores[fid] = rrf_scores.get(fid, 0.0) + self.fact_fts_weight * (1.0 / (self.fact_rrf_k + rank))
            for rank, (fid, _) in enumerate(intent_fts_hits, start=1):
                rrf_scores[fid] = rrf_scores.get(fid, 0.0) + 0.5 * self.fact_fts_weight * (
                    1.0 / (self.fact_rrf_k + rank)
                )

            sorted_rrf = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)
            max_pool = self.fact_candidate_pool_k + self.fact_fts_pool_k
            candidate_fids = [fid for fid, _ in sorted_rrf[:max_pool]]

            fact_triples = get_facts_by_ids(self.conn, candidate_fids)
            valid_candidates = [
                (fid, fact_triples[fid], rrf_scores[fid]) for fid in candidate_fids if fid in fact_triples
            ]

            # Step 4: Pass top-n candidates onwards to DSPy Recognition Memory
            top_n = min(self.linking_top_k, len(valid_candidates))
            top_candidates = valid_candidates[:top_n]

            # Calibrate fact candidate scores across the retrieval pool (preserving true positive semantic similarity)
            dense_norm_all = (
                min_max_normalize(np.array([s for _, s in dense_hits], dtype=np.float64))
                if dense_hits
                else np.array([])
            )
            dense_norm_dict = {fid: float(dense_norm_all[i]) for i, (fid, _) in enumerate(dense_hits)}
            fts_norm_all = (
                min_max_normalize(np.array([s for _, s in fts_hits], dtype=np.float64)) if fts_hits else np.array([])
            )
            fts_norm_dict = {fid: float(fts_norm_all[i]) for i, (fid, _) in enumerate(fts_hits)}

            kept_facts_with_scores: list[tuple[str, tuple[str, str, str], float]] = []
            if top_candidates:
                candidate_triples = [t for _, t, _ in top_candidates]
                candidate_scores = [
                    max(
                        dense_norm_dict.get(fid, 0.0),
                        self.fact_fts_weight * fts_norm_dict.get(fid, 0.0),
                        1e-2,
                    )
                    for fid, _, _ in top_candidates
                ]

                filter_indices = active_program.filter_triples(query, candidate_triples)
                for idx in filter_indices:
                    if 0 <= idx < len(top_candidates):
                        kept_facts_with_scores.append(
                            (
                                top_candidates[idx][0],
                                candidate_triples[idx],
                                float(candidate_scores[idx]),
                            )
                        )

            if not kept_facts_with_scores and allow_intent_path:
                return self.retrieve_with_seeds(
                    intent_query,
                    top_k=top_k,
                    allow_intent_path=False,
                    program=active_program,
                )

            # 2. Hybrid Dense + Lexical Passage Seeding
            query_passage_vec = self._encode_texts(
                [query], instruction=PASSAGE_INSTRUCTION, stage="query_passage_embed"
            )[0]
            chunk_vec_hits = search_chunk_vectors_knn(self.conn, query_passage_vec, top_k=self.passage_seed_pool_k)
            fts_hits = search_chunks_fts(self.conn, query, top_k=self.fts_seed_pool_k)

            passage_weights: dict[str, float] = {}
            if chunk_vec_hits:
                v_scores = min_max_normalize(np.array([s for _, s in chunk_vec_hits], dtype=np.float64))
                for (cid, _), s in zip(chunk_vec_hits, v_scores, strict=False):
                    passage_weights[cid] = float(s)

            if fts_hits:
                fts_scores = min_max_normalize(np.array([s for _, s in fts_hits], dtype=np.float64))
                for (cid, _), s in zip(fts_hits, fts_scores, strict=False):
                    passage_weights[cid] = passage_weights.get(cid, 0.0) + self.fts_weight * float(s)

            # 3. PPR Execution or Dense Fallback
            pool_k = max(top_k * 4, 50)
            ranked_chunks: list[tuple[str, float]] = []

            if not kept_facts_with_scores:
                sorted_passages = sorted(passage_weights.items(), key=lambda kv: kv[1], reverse=True)
                ranked_chunks = sorted_passages[:pool_k]
            else:
                kept_fids = [fid for fid, _, _ in kept_facts_with_scores]
                f_placeholders = ",".join("?" for _ in kept_fids)
                f_info_rows = self.conn.execute(
                    f"SELECT fact_id, COALESCE(obj_kind, 'entity'), confidence "
                    f"FROM facts WHERE fact_id IN ({f_placeholders})",
                    kept_fids,
                ).fetchall()
                fact_kind_by_fid: dict[str, str] = {str(r[0]): str(r[1]) for r in f_info_rows}
                fact_conf_by_fid: dict[str, float | None] = {
                    str(r[0]): (float(r[2]) if r[2] is not None else None) for r in f_info_rows
                }
                fact_sources_by_fid: dict[str, list[str]] = {}
                for r in self.conn.execute(
                    f"SELECT fact_id, chunk_id FROM fact_sources WHERE fact_id IN ({f_placeholders})",
                    kept_fids,
                ):
                    fact_sources_by_fid.setdefault(str(r[0]), []).append(str(r[1]))

                surface_mentions: list[str] = []
                for fid, (s, _, o), _ in kept_facts_with_scores:
                    surface_mentions.append(str(text_processing(s)))
                    if fact_kind_by_fid.get(fid, "entity") != "literal":
                        surface_mentions.append(str(text_processing(o)))
                resolved_map = resolve_entity_labels(self.conn, surface_mentions, query_vec=query_fact_vec)

                seed_entities: dict[str, list[float]] = {}
                literal_chunk_seeds: dict[str, list[float]] = {}
                for fid, (s, _, o), score in kept_facts_with_scores:
                    f_prior = fact_conf_by_fid.get(fid)
                    pooled_score = pool_confidences(score, f_prior) or score
                    s_norm = str(text_processing(s))
                    s_ids = resolved_map.get(s_norm) or [compute_mdhash_id(s_norm, prefix="entity-")]
                    for s_id in s_ids:
                        seed_entities.setdefault(s_id, []).append(pooled_score)
                    if fact_kind_by_fid.get(fid, "entity") == "literal":
                        src_cids = fact_sources_by_fid.get(fid, ())
                        if src_cids:
                            share = pooled_score / float(len(src_cids))
                            for src_cid in src_cids:
                                literal_chunk_seeds.setdefault(src_cid, []).append(share)
                    else:
                        o_norm = str(text_processing(o))
                        o_ids = resolved_map.get(o_norm) or [compute_mdhash_id(o_norm, prefix="entity-")]
                        for o_id in o_ids:
                            seed_entities.setdefault(o_id, []).append(pooled_score)

                doc_freqs = get_entity_doc_frequencies(self.conn, list(seed_entities.keys()))

                e_ids_list = list(seed_entities.keys())
                ent_confs: dict[str, float | None] = {}
                if e_ids_list:
                    e_ph = ",".join("?" for _ in e_ids_list)
                    ent_confs = {
                        str(r[0]): (float(r[1]) if r[1] is not None else None)
                        for r in self.conn.execute(
                            f"SELECT entity_id, confidence FROM entities WHERE entity_id IN ({e_ph})",
                            e_ids_list,
                        ).fetchall()
                    }

                entity_reset_weights: list[tuple[str, float]] = []
                for eid, scores in seed_entities.items():
                    mean_score = sum(scores) / len(scores)
                    df = max(1, doc_freqs.get(eid, 0))
                    p_node = ent_confs.get(eid)
                    calibrated_score = pool_confidences(mean_score, p_node) or mean_score
                    w_entity = calibrated_score / float(df)
                    entity_reset_weights.append((eid, w_entity))

                entity_reset_weights.sort(key=lambda kv: kv[1], reverse=True)
                top_entity_seeds = dict(entity_reset_weights[: self.linking_top_k])

                reset_distribution: dict[str, float] = {}
                for eid, w in top_entity_seeds.items():
                    reset_distribution[eid] = w

                for cid, w in passage_weights.items():
                    reset_distribution[cid] = self.passage_node_weight * w

                for cid, l_scores in literal_chunk_seeds.items():
                    reset_distribution[cid] = reset_distribution.get(cid, 0.0) + (sum(l_scores) / len(l_scores))

                ppr_scores = run_ppr(
                    graph,
                    reset_distribution,
                    damping=self.damping,
                    activation=self.activation,
                    activation_param=self.activation_param,
                    renormalize=self.activation_renormalize,
                    dual_weight=self.dual_weight,
                )
                ranked_chunks = rank_passages_by_ppr(graph, ppr_scores, top_k=pool_k)

            kept_facts_triples = [t for _, t, _ in kept_facts_with_scores]
            for seed_t in kept_facts_triples:
                if seed_t not in self._last_query_seeds:
                    self._last_query_seeds.append(seed_t)
            if not ranked_chunks:
                return [], kept_facts_triples

            # 4. MaxP Passage Span Deduplication + 1-Indexed Line Hydration from Disk
            hydrated_hits = self._hydrate_maxp_passages(ranked_chunks, top_k)
            return hydrated_hits, kept_facts_triples

    def hydrate_ranked_chunks(
        self,
        ranked_chunks: Sequence[tuple[str, float]],
        top_k: int,
    ) -> list[tuple[ChunkPointer, float, str]]:
        """Public thread-safe MaxP hydration of ranked `(chunk_id, score)` pairs."""
        with self._lock:
            return self._hydrate_maxp_passages(ranked_chunks, top_k)

    def _hydrate_maxp_passages(
        self,
        ranked_chunks: Sequence[tuple[str, float]],
        top_k: int,
    ) -> list[tuple[ChunkPointer, float, str]]:
        """Deduplicate ranked sub-chunks by parent passage span (MaxP) and hydrate top_k passages."""
        if not ranked_chunks or top_k <= 0:
            return []
        ranked_cids = [cid for cid, _ in ranked_chunks]
        pointers = get_chunk_pointers(self.conn, ranked_cids)
        score_by_cid = dict(ranked_chunks)

        ptrs_with_scores: list[tuple[ChunkPointer, float]] = []
        seen_spans: set[tuple[str, int, int]] = set()
        pool_limit = max(top_k * 2, 10)
        for cid in ranked_cids:
            ptr = pointers.get(cid)
            if ptr is None:
                continue
            span_key = (ptr.rel_posix_path, ptr.start_line, ptr.end_line)
            if span_key in seen_spans:
                continue
            seen_spans.add(span_key)
            ptrs_with_scores.append((ptr, score_by_cid[cid]))
            if len(ptrs_with_scores) >= pool_limit:
                break

        hydrated_passages = hydrate_budgeted_passages(
            ptrs_with_scores,
            corpus_root=self.workspace_root,
            max_context_chars=self.max_context_chars,
            max_expansion_hops=self.max_ast_expansion_hops,
        )
        deduped_hits: list[tuple[ChunkPointer, float, str]] = []
        seen_texts: set[str] = set()
        for hp in hydrated_passages:
            norm_text = hp.text.strip()
            if not norm_text or norm_text in seen_texts:
                continue
            seen_texts.add(norm_text)
            deduped_hits.append((hp.pointer, hp.score, hp.text))
            if len(deduped_hits) >= top_k:
                break
        return deduped_hits

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        *,
        program: HippoRAGProgram | None = None,
    ) -> list[tuple[ChunkPointer, float, str]]:
        """Retrieve top-k chunks for query using hybrid PPR + 1-indexed line hydration."""
        hits, _ = self.retrieve_with_seeds(query, top_k=top_k, program=program)
        return hits

    def retrieve_dpr(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[ChunkPointer, float, str]]:
        """Retrieve top-k chunks using pure dense passage retrieval (Standard DPR baseline)."""
        if top_k <= 0:
            return []
        with self._lock:
            pool_k = max(top_k * 4, 50)
            query_passage_vec = self._encode_texts(
                [query], instruction=PASSAGE_INSTRUCTION, stage="query_passage_embed"
            )[0]
            chunk_vec_hits = search_chunk_vectors_knn(self.conn, query_passage_vec, top_k=pool_k)
            if not chunk_vec_hits:
                return []
            return self._hydrate_maxp_passages(chunk_vec_hits, top_k)

    def resolve_chunk_pointers(
        self,
        chunk_ids: Sequence[str],
    ) -> dict[str, ChunkPointer]:
        """Return ChunkPointer metadata for the requested chunk IDs under the engine lock."""
        if not chunk_ids:
            return {}
        with self._lock:
            return get_chunk_pointers(self.conn, list(chunk_ids))

    def rag_qa(
        self,
        queries: Sequence[str],
        *,
        gold_docs: Sequence[Sequence[str]] | None = None,
        gold_answers: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Multi-hop QA retrieval and answer generation evaluated on task quality."""
        results: list[dict[str, Any]] = []
        active_module = self.ensemble if self.ensemble is not None else self.program
        for i, q in enumerate(queries):
            pred = active_module(question=q)
            ans = str(getattr(pred, "answer", "")).strip()
            reasoning = str(getattr(pred, "reasoning", "")).strip()
            formatted_context = str(getattr(pred, "context", ""))
            cids = list(getattr(pred, "retrieved_chunk_ids", ()))[: self.retrieval_top_k]
            scores = [float(s) for s in list(getattr(pred, "retrieved_scores", ()))[: self.retrieval_top_k]]

            item: dict[str, Any] = {
                "question": q,
                "answer": ans,
                "reasoning": reasoning,
                "retrieved_chunk_ids": cids,
                "retrieved_scores": scores,
                "context": formatted_context,
            }
            if gold_docs is not None and i < len(gold_docs):
                item["gold_docs"] = list(gold_docs[i])
            if gold_answers is not None and i < len(gold_answers):
                item["gold_answers"] = list(gold_answers[i]) if isinstance(gold_answers[i], list) else [gold_answers[i]]

            results.append(item)
        return results

    def _commit_active_reflection_checkpoint(
        self,
        active_key: str,
        encoded_active: str,
        state_key: str | None,
        encoded_state: str | None,
    ) -> None:
        with self.managed_conn.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO meta(key,value) VALUES (?,?)",
                (active_key, encoded_active),
            )
            if state_key is not None and encoded_state is not None:
                conn.execute(
                    "INSERT OR REPLACE INTO meta(key,value) VALUES (?,?)",
                    (state_key, encoded_state),
                )

    def _prepare_optimization_pools(
        self,
        trainset: Sequence[Any],
        valset: Sequence[Any] | None,
        opt_chunks: Sequence[tuple[ChunkPointer, str]] | None,
        mode: str,
        save_path: Path | str | None,
        progress_callback: Callable[[str], None] | None,
    ) -> None:
        self._in_optimization = True
        self._query_retrieval_memo.clear()
        self._resolved_docs_cache.clear()
        self._last_memo_question = None
        if opt_chunks is not None:
            existing_cids = {p.chunk_id for p, _ in self._opt_chunks_pool}
            for ptr, text in opt_chunks:
                if ptr.chunk_id not in existing_cids:
                    self._opt_chunks_pool.append((ptr, text))
                    existing_cids.add(ptr.chunk_id)

        self._active_progress_callback = progress_callback
        self._last_hook_program = None
        self._baseline_openie_fp = openie_fingerprint(self.program)
        self._hydrate_indexed_cids_from_db(self.program)
        # Legacy rows have no durable prompt provenance. Never relabel them as
        # results of the current seed; only verified checkpoints can be reused.
        unverified_cids = [
            str(cid)
            for (cid,) in self.conn.execute("SELECT chunk_id FROM chunks")
            if str(cid) not in self._opt_indexed_cids
        ]
        if unverified_cids:
            self.delete_chunks(unverified_cids)
        self._opt_docs_by_question.clear()
        use_gold_docs = len(self._opt_chunks_pool) > 64
        val_slice = list(valset or trainset)
        val_pool: list[tuple[ChunkPointer, str]] = []
        seen_val_cids: set[str] = set()
        for ex in val_slice:
            ex_docs = getattr(ex, "item_docs", None) or (getattr(ex, "gold_docs", None) if use_gold_docs else None)
            q_str = str(getattr(ex, "question", "") or "").strip()
            if ex_docs:
                if q_str:
                    self._opt_docs_by_question[q_str] = [str(d) for d in ex_docs]
                for p, t in self._resolve_chunks_for_docs(ex_docs):
                    if p.chunk_id not in seen_val_cids:
                        seen_val_cids.add(p.chunk_id)
                        val_pool.append((p, t))
        if val_pool:
            self._opt_val_chunks_pool = val_pool
        else:
            self._opt_val_chunks_pool = list(self._opt_chunks_pool)

        val_cids_set = {p.chunk_id for p, _ in self._opt_val_chunks_pool}
        train_cids_set: set[str] = set()
        for ex in trainset or ():
            ex_docs = getattr(ex, "item_docs", None) or (getattr(ex, "gold_docs", None) if use_gold_docs else None)
            q_str = str(getattr(ex, "question", "") or "").strip()
            if ex_docs:
                if q_str and q_str not in self._opt_docs_by_question:
                    self._opt_docs_by_question[q_str] = [str(d) for d in ex_docs]
                for p, _ in self._resolve_chunks_for_docs(ex_docs):
                    train_cids_set.add(p.chunk_id)
        if not train_cids_set and self._opt_chunks_pool:
            train_cids_set = {p.chunk_id for p, _ in self._opt_chunks_pool}
        all_pool_cids = val_cids_set | train_cids_set | {p.chunk_id for p, _ in self._opt_chunks_pool}
        self._graph_rlm_stats["val_pool_size"] = len(val_cids_set)
        self._graph_rlm_stats["train_pool_size"] = len(train_cids_set)
        self._graph_rlm_stats["corpus_pool_size"] = max(1, len(all_pool_cids))
        if save_path is not None:
            # Publish the real split/pool composition immediately so monitors never guess it.
            p_save = Path(save_path)
            p_save.parent.mkdir(parents=True, exist_ok=True)
            write_text_atomic(
                p_save.parent / f"{p_save.stem}_pools.json",
                json.dumps(
                    {
                        "mode": mode,
                        "train_queries": len(trainset),
                        "val_queries": len(valset or ()),
                        "gepa_val_queries": len(val_slice),
                        "val_chunk_ids": sorted(val_cids_set),
                        "train_chunk_ids": sorted(train_cids_set),
                        "corpus_chunk_ids": sorted(all_pool_cids),
                        "written_at": time.time(),
                    }
                ),
            )

        missing_val_pool = (
            [(p, t) for p, t in self._opt_val_chunks_pool if p.chunk_id not in self._opt_indexed_cids]
            if len(self._opt_val_chunks_pool) <= 64 or not self._opt_docs_by_question
            else []
        )
        if missing_val_pool:
            self.index_chunks(missing_val_pool, program=self.program, progress_callback=progress_callback)

    def _hydrate_top_chunks_for_eval(
        self,
        top_20_cids: Sequence[str],
        active_prog: HippoRAGProgram,
    ) -> dict[str, str]:
        hydrated_by_cid: dict[str, str] = {}
        if top_20_cids:
            with self._lock:
                ptrs = get_chunk_pointers(self.conn, top_20_cids)
            ptrs_scores = [(ptrs[c], 0.0) for c in top_20_cids if c in ptrs]
            if ptrs_scores:
                for hp in hydrate_passages(
                    ptrs_scores,
                    self.workspace_root,
                    max_island_chars=active_prog._context_char_limit(4000),
                    max_total_chars=0,
                ):
                    if hp.text:
                        hydrated_by_cid[hp.pointer.chunk_id] = hp.text.strip()
        return hydrated_by_cid

    def _query_rlm_stats_for_eval(self, gold_item_docs: Sequence[str] | None) -> dict[str, Any]:
        q_stats: dict[str, Any] | None = None
        if gold_item_docs is not None and len(gold_item_docs) > 0:
            gold_cids = tuple(p.chunk_id for p, _ in self._resolve_chunks_for_docs(gold_item_docs))
            q_stats = self._last_scope_by_cids.get(gold_cids)
        if q_stats is None:
            q_stats = (
                self._active_query_rlm_stats if self._active_query_rlm_stats is not None else self._graph_rlm_stats
            )
        return q_stats

    def _finalize_compiled_program(self, compiled: HippoRAGProgram) -> None:
        self.program = compiled
        self._bind_program_hooks()
        self._apply_program_embedding_prefixes()
        self._hydrate_indexed_cids_from_db(compiled)

    def _rehydrate_sandbox_after_optimization(self, compiled: HippoRAGProgram) -> None:
        self._in_optimization = False
        self._query_retrieval_memo.clear()
        self._last_memo_question = None
        if self._opt_chunks_pool and (
            len(self._opt_indexed_cids) < len(self._opt_chunks_pool)
            or self._active_extraction_fingerprint != extraction_fingerprint(compiled)
        ):
            self.index_chunks(self._opt_chunks_pool, program=compiled)
        self._bind_program_hooks()

    def _build_optimization_hooks(self) -> GraphOptimizationHooks:
        codec = CandidatePayloadCodec(self._candidate_cache_sources)
        return GraphOptimizationHooks(
            persist_trails=self._persist_dspy_trails_to_db,
            get_meta=self.get_meta,
            set_meta=self.set_meta,
            candidate_cache_namespace=self._candidate_cache_namespace,
            candidate_cache_sources=self._candidate_cache_sources,
            retrieval_settings={
                name: getattr(self, name)
                for name in (
                    "max_context_chars",
                    "max_ast_expansion_hops",
                    "linking_top_k",
                    "fact_candidate_pool_k",
                    "fact_fts_pool_k",
                    "fact_rrf_k",
                    "fact_fts_weight",
                    "passage_seed_pool_k",
                    "fts_seed_pool_k",
                    "passage_node_weight",
                    "fts_weight",
                    "damping",
                    "synonymy_threshold",
                    "synonymy_top_k",
                    "retrieval_top_k",
                    "activation",
                    "activation_param",
                    "activation_renormalize",
                    "beta",
                    "dual_weight",
                    "auto_merge_cosine_gate",
                    "auto_merge_difflib_gate",
                )
            },
            bind_program_hooks=self._bind_program_hooks,
            open_score_cache=lambda ctx: CandidateStageCache(
                self._candidate_cache_store,
                self.managed_conn,
                "simba_task_score",
                lambda: ctx,
            ),
            open_checkpoint_cache=lambda ctx: CandidateStageCache(
                self._candidate_cache_store,
                self.managed_conn,
                "reflection_checkpoint",
                lambda: ctx,
                sources=self._candidate_cache_sources,
            ),
            encode_payload=codec.encode,
            decode_payload=codec.decode,
            commit_active_checkpoint=self._commit_active_reflection_checkpoint,
            prepare_optimization_pools=self._prepare_optimization_pools,
            hydrate_top_chunks=self._hydrate_top_chunks_for_eval,
            query_rlm_stats=self._query_rlm_stats_for_eval,
            graph_rlm_stats=lambda: dict(self._graph_rlm_stats),
            last_hook_program=lambda: self._last_hook_program,
            clear_progress_callback=lambda: setattr(self, "_active_progress_callback", None),
            finalize_compiled_program=self._finalize_compiled_program,
            rehydrate_sandbox_after_optimization=self._rehydrate_sandbox_after_optimization,
            dspy_cache_dir=self._dspy_cache_dir,
            config=self.config,
        )

    def _configure_simba_task_learning(
        self,
        proposer: RLMInstructionProposer,
        template: HippoRAGProgram,
        trainset: Sequence[Any],
        metric: Callable[[Any, Any], float],
        *,
        save_path: Path | str | None,
        resume: bool,
    ) -> None:
        """Wire counterfactual training rewards and atomic worker snapshots in yaw.db."""
        configure_simba_task_learning(
            proposer,
            template,
            trainset,
            metric,
            self._build_optimization_hooks(),
            save_path=save_path,
            resume=resume,
        )

    def evolve_prompts(
        self,
        trainset: Sequence[Any],
        valset: Sequence[Any] | None = None,
        *,
        mode: Literal["miprov2", "gepa", "hybrid"] = "hybrid",
        save_path: Path | str | None = None,
        reflection_lm: Any = None,
        reflection_worker_lm: Any = None,
        reflection_token_counter: Any = None,
        auto: Literal["light", "medium", "heavy"] | None = "light",
        num_candidates: int | None = None,
        num_trials: int | None = None,
        progress_callback: Callable[[str], None] | None = None,
        opt_chunks: Sequence[tuple[ChunkPointer, str]] | None = None,
        sleep_seconds: float = 0.0,
        max_bootstrapped_demos: int | None = None,
        max_labeled_demos: int | None = None,
    ) -> HippoRAGProgram:
        """Evolve DSPy prompts via MIPROv2, GEPA, or two-phase hybrid with composite Sizecoding feedback."""
        self._in_optimization = True
        try:
            return evolve_program_prompts(
                self.program,
                trainset,
                valset,
                hooks=self._build_optimization_hooks(),
                mode=mode,
                save_path=save_path,
                reflection_lm=reflection_lm,
                reflection_worker_lm=reflection_worker_lm,
                reflection_token_counter=reflection_token_counter,
                auto=auto,
                num_candidates=num_candidates,
                num_trials=num_trials,
                progress_callback=progress_callback,
                opt_chunks=opt_chunks,
                sleep_seconds=sleep_seconds,
                max_bootstrapped_demos=max_bootstrapped_demos,
                max_labeled_demos=max_labeled_demos,
            )
        finally:
            self._in_optimization = False
            self._bind_program_hooks()

    optimize = evolve_prompts

    def evaluate(
        self,
        devset: Sequence[dspy.Example],
        *,
        metric: Callable[..., Any] | None = None,
        num_threads: int = 1,
        display_progress: bool = False,
        display_table: bool | int = False,
        save_as_json: str | Path | None = None,
        save_as_csv: str | Path | None = None,
        include_semantic_metrics: bool = False,
    ) -> dspy.evaluate.EvaluationResult:
        """Evaluate the bound `HippoRAGProgram` (or Pareto `dspy.Ensemble`) on `devset` using `dspy.Evaluate`."""
        return evaluate_program(
            self.ensemble if self.ensemble is not None else self.program,
            devset,
            metric=metric,
            num_threads=num_threads,
            display_progress=display_progress,
            display_table=display_table,
            save_as_json=save_as_json,
            save_as_csv=save_as_csv,
            include_semantic_metrics=include_semantic_metrics,
        )


__all__ = [
    "YawGraphEngine",
]
