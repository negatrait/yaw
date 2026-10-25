"""Native DSPy HippoRAG 2 Graph and Retrieval Engine backed by yaw.db and disk line hydration."""

import contextlib
import hashlib
import importlib.metadata
import json
import re
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
    StructuredTriple,
    YawRuntimeError,
    check_lexical_subsumption_inversion,
    detect_scalar_literal,
    is_acronym_pair,
    is_version_or_ordinal_sibling,
    write_text_atomic,
)
from yaw.core.config import context_char_limit, scale_context_budget
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
    extract_named_trace_entries,
    extraction_fingerprint,
    openie_fingerprint,
    query_retrieval_fingerprint,
    set_yaw_sleep_seconds,
    shacl_fingerprint,
    synonym_fingerprint,
    text_processing,
    trim_diagnostics,
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
    find_entity_disambiguation_candidates,
    get_chunk_pointers,
    get_entities_by_label_norm,
    get_entity_doc_frequencies,
    get_entity_metadata_batch,
    get_entity_vectors,
    get_existing_ids,
    get_facts_by_ids,
    get_graph_counts,
    get_meta,
    get_openie_checkpoints,
    hydrate_budgeted_passages,
    hydrate_passages,
    inspect_focus_node,
    lint_graph_shacl,
    load_graph_topology,
    make_chunk_hydrator,
    merge_entities,
    open_managed_yaw_db,
    resolve_entity_labels,
    save_openie_checkpoints,
    search_chunk_vectors_knn,
    search_chunks_fts,
    search_fact_vectors_knn,
    search_facts_fts,
    serialize_graph_turtle,
    set_meta,
    upsert_chunk_pointers,
    upsert_chunk_vectors,
    upsert_edges_with_sources,
    upsert_entities,
    upsert_entity_labels,
    upsert_entity_vectors,
    upsert_fact_sources,
    upsert_fact_vectors,
    upsert_facts,
    upsert_semantic_edges,
)

_SAME_SENSE_COSINE_GATE = 0.85


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
        same_sense_cosine_gate: float = _SAME_SENSE_COSINE_GATE,
        homograph_cosine_gate: float | None = None,
        max_context_chars: int | None = None,
        max_ast_expansion_hops: int = 1,
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

        self.max_context_chars = (
            max_context_chars
            if max_context_chars is not None
            else context_char_limit(24000, program.context_window_tokens, program.max_output_tokens)
        )
        self.max_ast_expansion_hops = max_ast_expansion_hops
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
        self.same_sense_cosine_gate = float(
            homograph_cosine_gate if homograph_cosine_gate is not None else same_sense_cosine_gate
        )
        self.homograph_cosine_gate = self.same_sense_cosine_gate
        self.traversal_policy: RelationTraversalPolicy | None = traversal_policy

        self._csr_graph: CSRGraph | None = None
        self._active_repl_repair_keys: set[tuple[str, str, str, str]] | None = None
        self._active_repl_repairs_list: list[dict[str, str]] | None = None
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
        self._openie_fp_cache = CandidateStageCache(
            cache_store,
            self.managed_conn,
            "openie",
            self._candidate_cache_namespace,
            sources=self._candidate_cache_sources,
        )
        self._synonym_fp_cache = CandidateStageCache(
            cache_store,
            self.managed_conn,
            "synonym",
            self._candidate_cache_namespace,
            sources=self._candidate_cache_sources,
        )
        self._shacl_fp_cache = CandidateStageCache(
            cache_store,
            self.managed_conn,
            "shacl",
            self._candidate_cache_namespace,
            sources=self._candidate_cache_sources,
        )

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

    def _graph_cache_context(self) -> str:
        """Hash the graph visible to SHACL's inspection/repair tools, in bounded rows."""
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
            for row in self.conn.execute(f"SELECT * FROM {table} ORDER BY {order}"):
                digest.update(json.dumps(tuple(row), ensure_ascii=False).encode("utf-8"))
        digest.update(json.dumps(sorted(self._candidate_cache_sources())).encode())
        return digest.hexdigest()

    def _hydrate_indexed_cids_from_db(
        self,
        program: HippoRAGProgram | None = None,
    ) -> None:
        """Populate _opt_indexed_cids, _chunk_indexed_ext_fp, and persisted DSPy trails from SQLite."""
        active_prog = program if program is not None else self.program
        try:
            rows = self.conn.execute(
                "SELECT c.chunk_id, m.value, p.value FROM chunks c "
                "INNER JOIN openie_chunks o ON c.chunk_id = o.chunk_id "
                "INNER JOIN meta m ON m.key = 'indexed_fp:' || c.chunk_id "
                "INNER JOIN meta p ON p.key = 'openie_fp:' || c.chunk_id"
            ).fetchall()
        except sqlite3.Error:
            return
        if not rows:
            return
        provenance = {}
        for cid, raw, saved_openie_fp in rows:
            recorded = json.loads(raw)
            if recorded[1] == saved_openie_fp:
                provenance[str(cid)] = recorded
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

            try:
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
                            "synonym_audit_trail",
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
            except Exception:
                pass

    def _persist_dspy_trails_to_db(
        self,
        program: HippoRAGProgram | None = None,
    ) -> None:
        """Persist graph RLM stats, per-chunk RLM stats, and DSPy audit trails into SQLite meta."""
        active_prog = program if program is not None else self.program
        try:
            payload = {
                "graph_rlm_stats": dict(self._graph_rlm_stats),
                "chunk_rlm_stats": dict(self._chunk_rlm_stats),
                **active_prog.dump_audit_trails(bounded=True),
            }
            set_meta(self.conn, "dspy_trails_state", json.dumps(payload, ensure_ascii=False))
        except Exception:
            pass

    def _hydrate_chunk_traces_and_stats_from_db(
        self,
        target_list: Sequence[tuple[ChunkPointer, str]],
    ) -> None:
        """Reconstruct DSPy Stage 1/2 traces and per-chunk token stats from openie_chunks for seeded chunks."""
        missing_cids = [
            p.chunk_id
            for p, _ in target_list
            if p.chunk_id not in self._captured_chunk_traces or p.chunk_id not in self._chunk_rlm_stats
        ]
        if not missing_cids:
            return
        try:
            cached_openie = get_openie_checkpoints(self.conn, missing_cids)
        except sqlite3.Error:
            return
        if not cached_openie:
            return
        text_by_cid = {p.chunk_id: t for p, t in target_list}
        for cid, rec in cached_openie.items():
            passage_text = text_by_cid.get(cid, "")
            named_ents = [[str(ename), str(edef)] for ename, edef in (getattr(rec, "entity_definitions", None) or ())]
            if not named_ents and getattr(rec, "entities", None):
                named_ents = [
                    [str(e[0]), str(e[1]) if len(e) >= 2 else ""]
                    if isinstance(e, (list, tuple)) and e
                    else [str(e), ""]
                    for e in rec.entities
                ]
            triples_out = [
                {
                    "sub": st.subj,
                    "pred": st.pred,
                    "obj": st.obj,
                    "def": st.spo_def,
                }
                for st in (getattr(rec, "structured_triples", None) or ())
            ]
            if cid not in self._captured_chunk_traces:
                c_traces: list[tuple[str, dict[str, Any], Any]] = []
                if named_ents:
                    c_traces.append(
                        (
                            "ner",
                            {"passage": passage_text},
                            dspy.Prediction(named_entities=named_ents),
                        )
                    )
                if triples_out:
                    c_traces.append(
                        (
                            "triples",
                            {"passage": passage_text, "named_entities": named_ents},
                            dspy.Prediction(triples=triples_out),
                        )
                    )
                if c_traces:
                    self._captured_chunk_traces[cid] = c_traces
            if cid not in self._chunk_rlm_stats:
                est_json = json.dumps(
                    {"named_entities": named_ents, "triples": triples_out},
                    ensure_ascii=False,
                )
                est_openie_tok = float(max(64, len(est_json) // 4))
                self._chunk_rlm_stats[cid] = {
                    "synonym_judge_tokens": 0.0,
                    "shacl_healer_tokens": 0.0,
                    "shacl_violations_initial": 0.0,
                    "shacl_violations_remaining": 0.0,
                    "missing_entity_defs_count": 0.0,
                    "missing_entity_defs_tokens": 0.0,
                    "openie_tokens": est_openie_tok,
                }

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
            "synonym_judge_tokens": 0,
            "shacl_healer_tokens": 0,
            "missing_entity_defs_count": 0,
            "missing_entity_defs_tokens": 0,
            "embedding_calls": 0,
            "embedding_tokens": 0,
        }

    def _hydrate_single_chunk(self, chunk_id: str) -> str:
        hydrator = make_chunk_hydrator(self.conn, self.workspace_root)
        return hydrator(chunk_id)

    @staticmethod
    def _repair_signature(rep: Mapping[str, Any]) -> tuple[str, str, str, str]:
        focus = str(rep.get("focus_node", "")).strip()
        action = str(rep.get("action", "")).strip()
        val = str(rep.get("value", "")).strip()
        target = str(rep.get("target_node", "")).strip() or val
        return (focus, action, target, val)

    def _apply_shacl_repairs_from_tool(
        self,
        repairs: Sequence[Any],
    ) -> int:
        pending: list[dict[str, str]] = []
        seen_in_call: set[tuple[str, str, str, str]] = set()
        for rep in repairs:
            if not isinstance(rep, Mapping):
                continue
            sig = self._repair_signature(rep)
            if sig in seen_in_call:
                continue
            seen_in_call.add(sig)
            if self._active_repl_repair_keys is not None:
                if sig in self._active_repl_repair_keys:
                    continue
                self._active_repl_repair_keys.add(sig)
            rep_dict = {str(k): str(v) for k, v in rep.items()}
            if self._active_repl_repairs_list is not None:
                self._active_repl_repairs_list.append(rep_dict)
            pending.append(rep_dict)
        if not pending:
            return 0
        applied = self._apply_shacl_repairs(self.conn, pending)
        self._graph_rlm_stats["shacl_repairs_applied"] += applied
        return applied

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
                node_inspector=lambda eid: inspect_focus_node(self.conn, eid),
                chunk_hydrator=self._hydrate_single_chunk,
                shacl_repair_applier=self._apply_shacl_repairs_from_tool,
                shacl_linter=lambda: [v.to_dict() for v in lint_graph_shacl(self.conn, auto_heal=False)],
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

        if self._in_optimization or bool(item_docs):
            target_cids = self._ensure_candidate_chunks_indexed(active_program, item_docs=item_docs)
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
            if not self._opt_indexed_cids and bool(self._opt_chunks_pool):
                self._ensure_candidate_chunks_indexed(self.program, item_docs=item_docs)
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
        hits = self.retrieve(
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
        formatted_context = self.format_qa_context(hits[:qa_top_k])
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
            violations = lint_graph_shacl(self.conn, auto_heal=False)
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
        auto_heal: bool = False,
        as_turtle: bool = False,
    ) -> dict[str, Any]:
        """Run closed-world SHACL linting (and optional Turtle serialization) over yaw.db."""
        with self._lock:
            if auto_heal:
                with self.managed_conn.transaction() as conn:
                    violations = lint_graph_shacl(conn, auto_heal=True)
            else:
                violations = lint_graph_shacl(self.conn, auto_heal=False)
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
                has_broader = (
                    self.conn.execute("SELECT 1 FROM edges WHERE kind = 'skos:broader' LIMIT 1").fetchone() is not None
                )
                use_policy = self.traversal_policy is not None or has_broader
                chunk_ids, entity_ids, edges = load_graph_topology(self.conn, include_kinds=use_policy)
                self._csr_graph = build_transition_matrix(
                    chunk_ids,
                    entity_ids,
                    edges,
                    beta=self.beta,
                    policy=(self.traversal_policy or RelationTraversalPolicy()) if use_policy else None,
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
        max_context_chars: int | None = None,
        max_characters: int | None = None,
    ) -> str:
        """Pack top-k hydrated AST islands into QA context using budgeted AST-expansion and coalescence."""
        if not hits:
            return ""
        budget = (
            max_characters
            if max_characters is not None
            else (max_context_chars if max_context_chars is not None else self.max_context_chars)
        )
        ptrs_scores = [(p, s) for p, s, _ in hits]
        hydrated = hydrate_budgeted_passages(
            ptrs_scores,
            corpus_root=self.workspace_root,
            max_context_chars=budget,
            max_expansion_hops=self.max_ast_expansion_hops,
        )
        blocks: list[str] = []
        for hp in hydrated:
            formatted = self._format_hydrated_passage(hp.pointer, hp.text)
            if formatted:
                blocks.append(formatted)
        return "\n\n".join(blocks)

    _apply_shacl_repairs = staticmethod(apply_shacl_repairs)

    def _reset_sandbox_graph(self, *, clear_openie: bool = True) -> None:
        """Clear graph tables in the sandbox DB when a candidate mutates extraction/graph prompts or query scope."""
        with self._lock, self.managed_conn.transaction() as conn:
            conn.execute("DELETE FROM meta WHERE key GLOB 'indexed_fp:*'")
            conn.execute("DELETE FROM chunk_vec")
            conn.execute("DELETE FROM chunk_fts")
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM fact_sources")
            conn.execute("DELETE FROM fact_vec")
            conn.execute("DELETE FROM fact_fts")
            conn.execute("DELETE FROM facts")
            conn.execute("DELETE FROM edge_sources")
            conn.execute("DELETE FROM edges")
            conn.execute("DELETE FROM entity_labels")
            conn.execute("DELETE FROM entity_vec")
            conn.execute("DELETE FROM entities")
            if clear_openie:
                conn.execute("DELETE FROM openie_chunks")
                conn.execute("DELETE FROM meta WHERE key GLOB 'openie_fp:*'")
                self._captured_chunk_traces.clear()
        self._captured_graph_traces.clear()
        self._opt_indexed_cids.clear()
        self._chunk_indexed_ext_fp.clear()
        self._attempted_shacl_keys.clear()
        self._chunk_rlm_stats.clear()
        self._query_retrieval_memo.clear()
        self._last_memo_question = None
        self._active_query_rlm_stats = None
        self._csr_graph = None
        self._graph_rlm_stats = self._empty_graph_rlm_stats()

    def _materialize_doc_chunk(
        self,
        doc: str,
        *,
        cid: str | None = None,
    ) -> tuple[ChunkPointer, str] | None:
        lines = doc.splitlines(keepends=True)
        n_lines = len(lines)
        if n_lines == 0:
            return None
        chunk_id = cid or compute_mdhash_id(doc, prefix="chunk-")
        corpus_dir = self.workspace_root / "corpus"
        corpus_dir.mkdir(parents=True, exist_ok=True)
        normalized_doc = "".join(line if line.endswith("\n") else line + "\n" for line in lines)
        rel_posix_path = f"corpus/{chunk_id}.md"
        write_text_atomic(self.workspace_root / rel_posix_path, normalized_doc)
        first_line = doc.split("\n", 1)[0].strip()
        title = first_line if len(first_line) < 100 else ""
        ptr = ChunkPointer(
            chunk_id=chunk_id,
            rel_posix_path=rel_posix_path,
            start_line=1,
            end_line=n_lines,
            title=title,
            ast_island_start=1,
            ast_island_end=n_lines,
        )
        return (ptr, doc)

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
            direct_cid = compute_mdhash_id(d, prefix="chunk-")
            if direct_cid in pool_by_cid:
                doc_matches.append(pool_by_cid[direct_cid])
            else:
                doc_title = d.split("\n", 1)[0].strip() if "\n" in d else ""
                doc_body = d.split("\n", 1)[1].strip() if "\n" in d else d
                for ptr, c_text in self._opt_chunks_pool:
                    c_strip = c_text.strip()
                    if not c_strip:
                        continue
                    title_match = bool(doc_title and ptr.title and ptr.title.strip() == doc_title)
                    body_match = (
                        c_strip in (d, doc_body)
                        or (len(c_strip) >= 30 and c_strip in d)
                        or (title_match and (c_strip in doc_body or doc_body in c_strip))
                    )
                    if body_match:
                        doc_matches.append((ptr, c_text))
                if not doc_matches:
                    mat = self._materialize_doc_chunk(d, cid=direct_cid)
                    if mat is not None:
                        if direct_cid not in pool_by_cid:
                            self._opt_chunks_pool.append(mat)
                            pool_by_cid[direct_cid] = mat
                            pool_len = len(self._opt_chunks_pool)
                        doc_matches.append(mat)
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
            self._hydrate_indexed_cids_from_db(program)
            ext_fp = extraction_fingerprint(program)
            req_chunks = self._resolve_chunks_for_docs(item_docs)
            pool_chunks = (
                list(self._opt_val_chunks_pool)
                if self._opt_val_chunks_pool
                else (
                    list(self._opt_chunks_pool)
                    if len(self._opt_chunks_pool) <= 64 and self._opt_chunks_pool
                    else (req_chunks if req_chunks else list(self._opt_chunks_pool[:32]))
                )
            )
            missing_pool = [(p, t) for p, t in pool_chunks if p.chunk_id not in self._opt_indexed_cids]
            if missing_pool:
                for b_start in range(0, len(missing_pool), 8):
                    self.index_chunks(
                        missing_pool[b_start : b_start + 8],
                        program=program,
                        progress_callback=self._active_progress_callback,
                    )

            target_list = req_chunks if req_chunks else pool_chunks
            stale_targets = [
                (p, t)
                for p, t in target_list
                if p.chunk_id in self._opt_indexed_cids and self._chunk_indexed_ext_fp.get(p.chunk_id) != ext_fp
            ]
            if stale_targets:
                stale_cids = [p.chunk_id for p, _ in stale_targets]
                # A synonym/healing-only mutation still uses the same OpenIE.
                # Preserve matching extraction checkpoints across graph rebuilds.
                openie_records = get_openie_checkpoints(self.conn, stale_cids)
                openie_fps = {cid: get_meta(self.conn, f"openie_fp:{cid}") for cid in stale_cids}
                self.delete_chunks(stale_cids)
                with self.managed_conn.transaction():
                    save_openie_checkpoints(self.conn, list(openie_records.values()))
                    for cid, fp in openie_fps.items():
                        if fp is not None:
                            set_meta(self.conn, f"openie_fp:{cid}", fp)
                for cid in stale_cids:
                    self._opt_indexed_cids.discard(cid)
                    self._chunk_indexed_ext_fp.pop(cid, None)

            missing_targets = [(p, t) for p, t in target_list if p.chunk_id not in self._opt_indexed_cids]
            if missing_targets:
                for b_start in range(0, len(missing_targets), 8):
                    self.index_chunks(
                        missing_targets[b_start : b_start + 8],
                        program=program,
                        progress_callback=self._active_progress_callback,
                    )
            self._hydrate_chunk_traces_and_stats_from_db(target_list)
            target_cids = [p.chunk_id for p, _ in target_list]

            if target_cids and any(cid in self._chunk_rlm_stats for cid in target_cids):
                agg: dict[str, float] = {
                    "synonym_judge_tokens": 0.0,
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
                            "synonym_judge_tokens",
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
                if d not in seen_docs:
                    seen_docs.add(d)
                    unique_docs.append(d)

            chunk_ids = [compute_mdhash_id(d, prefix="chunk-") for d in unique_docs]
            existing_chunks = get_existing_ids(self.conn, "chunk", chunk_ids)
            existing_vecs = get_existing_ids(self.conn, "chunk_vec", chunk_ids)
            existing_openie = get_existing_ids(self.conn, "openie", chunk_ids)

            incomplete_cids = [
                cid
                for cid in chunk_ids
                if cid in existing_chunks and (cid not in existing_vecs or cid not in existing_openie)
            ]
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
                for doc in new_docs:
                    mat = self._materialize_doc_chunk(doc)
                    if mat is not None:
                        chunks_to_index.append(mat)

            return self.index_chunks(chunks_to_index, progress_callback=progress_callback)

    def index_chunks(
        self,
        chunks: Sequence[tuple[ChunkPointer, str]],
        *,
        progress_callback: Callable[[str], None] | None = None,
        program: HippoRAGProgram | None = None,
    ) -> int:
        """Incremental chunk indexing into yaw.db, sqlite-vec, SKOS schema, and sparse CSR graph."""
        if not chunks:
            return 0

        opt_sleep = getattr(getattr(self, "config", None), "optimize_sleep_seconds", 0.0)
        if opt_sleep > 0:
            set_yaw_sleep_seconds(opt_sleep)

        active_program = program if program is not None else self.program
        with self._lock:
            trim_diagnostics(active_program)
            self._bind_program_hooks(active_program)
            self._apply_program_embedding_prefixes(active_program)
            openie_fp = openie_fingerprint(active_program)
            ext_fp = extraction_fingerprint(active_program)

            existing_pool_cids = {p.chunk_id for p, _ in self._opt_chunks_pool}
            for ptr, text in chunks:
                if ptr.chunk_id not in existing_pool_cids:
                    self._opt_chunks_pool.append((ptr, text))
                    existing_pool_cids.add(ptr.chunk_id)

            chunk_ptrs = [ptr for ptr, _ in chunks]
            chunk_ids = [ptr.chunk_id for ptr in chunk_ptrs]
            self._captured_graph_traces.clear()
            pre_batch_stats = dict(self._graph_rlm_stats)

            fmt_chunk = getattr(
                self.embedder,
                "format_chunk_for_embedding",
                lambda p, t: DSPyEmbeddingAdapter.format_chunk_for_embedding(self.embedder, p, t),
            )
            fmt_entity = getattr(
                self.embedder,
                "format_entity_for_embedding",
                DSPyEmbeddingAdapter.format_entity_for_embedding,
            )
            fmt_fact = getattr(
                self.embedder,
                "format_fact_for_embedding",
                DSPyEmbeddingAdapter.format_fact_for_embedding,
            )

            # 1. Chunk vector embeddings in bounded batches
            existing_chunk_vecs = get_existing_ids(self.conn, "chunk_vec", chunk_ids)
            missing_chunks = [(p, t) for p, t in chunks if p.chunk_id not in existing_chunk_vecs]
            chunk_vec_pairs: list[tuple[str, np.ndarray]] = []
            if missing_chunks:
                if progress_callback is not None:
                    progress_callback(f"indexing: embedding {len(missing_chunks)} chunk vectors")
                texts_to_embed = [fmt_chunk(p, t) for p, t in missing_chunks]
                vecs = self._encode_in_batches(texts_to_embed, instruction="", stage="chunk_embed")
                chunk_vec_pairs = [(p.chunk_id, v) for (p, _), v in zip(missing_chunks, vecs, strict=False)]

            # 2. Crash-resilient OpenIE extraction (persisted per-chunk immediately)
            cached_openie = get_openie_checkpoints(self.conn, chunk_ids)
            all_records: list[Any] = []
            attempted_missing_by_cid: dict[str, set[str]] = {}
            chunk_openie_tok_by_cid: dict[str, int] = {}
            total_chunks = len(chunks)
            for idx, (ptr, text) in enumerate(chunks, 1):
                # The previous chunk's slices are already consumed. Do not trim
                # inside extract_openie: its caller still needs the new records.
                trim_diagnostics(active_program)
                openie_key = (openie_fp, ptr.chunk_id, hashlib.sha256(text.encode()).hexdigest())
                cached_stage = self._openie_fp_cache.get(openie_key, passage=text)
                if cached_stage is not None:
                    rec, cached_tr, cached_norms, m_cnt, m_tok, o_tok = cached_stage
                    self._captured_chunk_traces[ptr.chunk_id] = list(cached_tr)
                    attempted_missing_by_cid[ptr.chunk_id] = set(cached_norms)
                    chunk_openie_tok_by_cid[ptr.chunk_id] = o_tok
                    self._graph_rlm_stats["missing_entity_defs_count"] += m_cnt
                    self._graph_rlm_stats["missing_entity_defs_tokens"] += m_tok
                    with self.managed_conn.transaction():
                        save_openie_checkpoints(self.conn, [rec])
                        set_meta(self.conn, f"openie_fp:{ptr.chunk_id}", openie_fp)
                    all_records.append(rec)
                elif ptr.chunk_id in cached_openie and (get_meta(self.conn, f"openie_fp:{ptr.chunk_id}") == openie_fp):
                    rec = cached_openie[ptr.chunk_id]
                    if getattr(rec, "entity_definitions", None):
                        cached_norms: set[str] = set()
                        for raw_ent in getattr(rec, "entities", ()) or ():
                            e_s = str(raw_ent[0]) if isinstance(raw_ent, (list, tuple)) and raw_ent else str(raw_ent)
                            en = str(text_processing(e_s))
                            if en:
                                cached_norms.add(en)
                        for st_item in getattr(rec, "structured_triples", ()) or ():
                            sn = str(text_processing(str(getattr(st_item, "subj", ""))))
                            if sn:
                                cached_norms.add(sn)
                            if getattr(st_item, "obj_kind", "entity") != "literal":
                                on = str(text_processing(str(getattr(st_item, "obj", ""))))
                                if on:
                                    cached_norms.add(on)
                        for t_item in getattr(rec, "triples", ()) or ():
                            if len(t_item) >= 3:
                                for side in (t_item[0], t_item[2]):
                                    sn = str(text_processing(str(side)))
                                    if sn:
                                        cached_norms.add(sn)
                        attempted_missing_by_cid[ptr.chunk_id] = cached_norms
                    all_records.append(rec)
                else:
                    if progress_callback is not None:
                        progress_callback(f"indexing: OpenIE extraction ({idx}/{total_chunks})")
                    prev_def_audit_len = len(getattr(active_program, "entity_def_audit_trail", ()))
                    prev_telem_len = len(getattr(active_program, "telemetry", ()))
                    chunk_raw_trace: list[Any] = []
                    with dspy.context(trace=chunk_raw_trace, usage_tracker=None):
                        rec = active_program.extract_openie(ptr.chunk_id, text)
                    chunk_traces = extract_named_trace_entries(active_program, chunk_raw_trace)
                    self._captured_chunk_traces[ptr.chunk_id] = chunk_traces
                    new_telem_entries = list(getattr(active_program, "telemetry", ())[prev_telem_len:])
                    raw_openie_tok = sum(
                        int(t_entry.get("tokens_used", 0)) for t_entry in new_telem_entries if isinstance(t_entry, dict)
                    )
                    new_def_entries = list(getattr(active_program, "entity_def_audit_trail", ())[prev_def_audit_len:])
                    attempted_norms: set[str] = set()
                    chunk_m_cnt = 0
                    chunk_m_tok = 0
                    for entry in new_def_entries:
                        if isinstance(entry, dict):
                            c_cnt = int(entry.get("missing_count", 0))
                            c_tok = int(entry.get("tokens", 0))
                            chunk_m_cnt += c_cnt
                            chunk_m_tok += c_tok
                            self._graph_rlm_stats["missing_entity_defs_count"] += c_cnt
                            self._graph_rlm_stats["missing_entity_defs_tokens"] += c_tok
                            for m_ent in entry.get("missing_entities") or ():
                                m_norm = str(text_processing(str(m_ent)))
                                if m_norm:
                                    attempted_norms.add(m_norm)
                    chunk_openie_tok = max(0, raw_openie_tok - chunk_m_tok)
                    chunk_openie_tok_by_cid[ptr.chunk_id] = chunk_openie_tok
                    attempted_missing_by_cid[ptr.chunk_id] = attempted_norms
                    self._openie_fp_cache[openie_key] = (
                        rec,
                        list(chunk_traces),
                        set(attempted_norms),
                        chunk_m_cnt,
                        chunk_m_tok,
                        chunk_openie_tok,
                    )
                    with self.managed_conn.transaction():
                        save_openie_checkpoints(self.conn, [rec])
                        set_meta(self.conn, f"openie_fp:{ptr.chunk_id}", openie_fp)
                    all_records.append(rec)

            # 3. Resolve Stage 1 NER Entities, Stage 2 StructuredTriples, Provisional Sense-Split Gate, and
            # Contextual Embeddings
            chunk_text_by_id = {ptr.chunk_id: text for ptr, text in chunks}
            record_structured: list[tuple[str, list[StructuredTriple], list[tuple[str, str]]]] = []
            all_norm_labels: set[str] = set()
            surface_by_norm: dict[str, str] = {}
            chunk_surface_by_norm: dict[tuple[str, str], str] = {}

            def _record_surface(norm_key: str, raw_val: str, cid_key: str = "") -> None:
                clean_raw = raw_val.strip()
                if not norm_key or not clean_raw:
                    return
                prev = surface_by_norm.get(norm_key)
                if prev is None or (prev == prev.lower() and clean_raw != clean_raw.lower()):
                    surface_by_norm[norm_key] = clean_raw
                if cid_key:
                    prev_c = chunk_surface_by_norm.get((cid_key, norm_key))
                    if prev_c is None or (prev_c == prev_c.lower() and clean_raw != clean_raw.lower()):
                        chunk_surface_by_norm[(cid_key, norm_key)] = clean_raw

            for rec in all_records:
                cid = rec.chunk_id
                raw_st_items = list(getattr(rec, "structured_triples", ()) or ())
                entity_role_raws: set[str] = set()
                literal_only_raws: set[str] = set()
                for st_item in raw_st_items:
                    s_r = str(getattr(st_item, "subj", "")).strip()
                    o_r = str(getattr(st_item, "obj", "")).strip()
                    if s_r:
                        entity_role_raws.add(s_r)
                        entity_role_raws.add(str(text_processing(s_r)))
                    if getattr(st_item, "obj_kind", "entity") == "literal":
                        if o_r:
                            literal_only_raws.add(o_r)
                    elif o_r:
                        entity_role_raws.add(o_r)
                        entity_role_raws.add(str(text_processing(o_r)))
                literal_only_raws -= entity_role_raws

                stage1_defs: dict[str, str] = {}
                stage1_senses: dict[str, list[str]] = {}
                for raw_pair in getattr(rec, "entity_definitions", ()) or ():
                    if isinstance(raw_pair, (list, tuple)) and len(raw_pair) >= 2:
                        raw_e0 = str(raw_pair[0]).strip()
                        is_lit_norm = raw_e0 in literal_only_raws or detect_scalar_literal(raw_e0)[0]
                        enorm = raw_e0 if is_lit_norm else str(text_processing(raw_e0))
                        _record_surface(enorm, raw_e0, cid)
                        edef = str(raw_pair[1]).strip()
                        if enorm and edef:
                            if enorm not in stage1_defs:
                                stage1_defs[enorm] = edef
                            senses_lst = stage1_senses.setdefault(enorm, [])
                            if all(d.lower() != edef.lower() for d in senses_lst):
                                senses_lst.append(edef)

                ner_entities: list[tuple[str, str]] = []
                seen_ner_norms: set[str] = set()
                seen_ner_sense_keys: set[tuple[str, str]] = set()
                for raw_ent in getattr(rec, "entities", ()) or ():
                    if isinstance(raw_ent, (list, tuple)) and len(raw_ent) >= 1:
                        e_str = str(raw_ent[0]).strip()
                        e_inline_def = str(raw_ent[1]).strip() if len(raw_ent) >= 2 and raw_ent[1] is not None else ""
                    else:
                        e_str = str(raw_ent).strip()
                        e_inline_def = ""
                    is_lit_norm = e_str in literal_only_raws or detect_scalar_literal(e_str)[0]
                    e_norm = e_str if is_lit_norm else str(text_processing(e_str))
                    if not e_norm:
                        continue
                    _record_surface(e_norm, e_str, cid)
                    e_def = e_inline_def or stage1_defs.get(e_norm, "")
                    s_key = (e_norm, e_def.lower())
                    if s_key in seen_ner_sense_keys or (not e_def and e_norm in seen_ner_norms):
                        continue
                    seen_ner_norms.add(e_norm)
                    seen_ner_sense_keys.add(s_key)
                    if e_inline_def:
                        if e_norm not in stage1_defs:
                            stage1_defs[e_norm] = e_inline_def
                        senses_lst = stage1_senses.setdefault(e_norm, [])
                        if all(d.lower() != e_inline_def.lower() for d in senses_lst):
                            senses_lst.append(e_inline_def)
                    all_norm_labels.add(e_norm)
                    ner_entities.append((e_norm, e_def))

                for enorm, senses_lst in stage1_senses.items():
                    for edef in senses_lst:
                        s_key = (enorm, edef.lower())
                        if s_key not in seen_ner_sense_keys:
                            seen_ner_norms.add(enorm)
                            seen_ner_sense_keys.add(s_key)
                            all_norm_labels.add(enorm)
                            ner_entities.append((enorm, edef))

                st_list: list[StructuredTriple] = list(raw_st_items)
                if not st_list and getattr(rec, "triples", ()):
                    for t in rec.triples:
                        if len(t) >= 3:
                            s_raw, p_raw, o_raw = (
                                str(t[0]).strip(),
                                str(t[1]).strip(),
                                str(t[2]).strip(),
                            )
                            spo_synth = f"{s_raw} {p_raw} {o_raw}."
                            s_raw_norm = str(text_processing(s_raw))
                            o_raw_norm = str(text_processing(o_raw))
                            s_def_val = stage1_defs.get(s_raw_norm, "")
                            o_def_val = stage1_defs.get(o_raw_norm, "")
                            if s_raw_norm and s_raw_norm == o_raw_norm and len(stage1_senses.get(s_raw_norm, ())) >= 2:
                                s_def_val = stage1_senses[s_raw_norm][0]
                                o_def_val = stage1_senses[s_raw_norm][1]
                            st_list.append(
                                StructuredTriple(
                                    subj=s_raw,
                                    subj_def=s_def_val,
                                    pred=p_raw,
                                    obj=o_raw,
                                    obj_def=o_def_val,
                                    spo_def=spo_synth,
                                )
                            )
                normalized_sts: list[StructuredTriple] = []
                for st in st_list:
                    s_raw_str = str(st.subj).strip()
                    o_raw_str = str(st.obj).strip()
                    s_norm = str(text_processing(st.subj))
                    p_norm = str(text_processing(st.pred))
                    is_literal_obj = (
                        getattr(st, "obj_kind", "entity") == "literal"
                        or detect_scalar_literal(str(st.obj))[0]
                    )
                    o_val = o_raw_str if is_literal_obj else str(text_processing(st.obj))
                    if not s_norm or not p_norm or not o_val:
                        continue
                    if detect_scalar_literal(s_norm)[0]:
                        if not is_literal_obj:
                            s_norm, o_val = o_val, s_norm
                            s_raw_str, o_raw_str = o_raw_str, s_raw_str
                            is_literal_obj = True
                        else:
                            continue
                    _record_surface(s_norm, s_raw_str, cid)
                    _record_surface(o_val, o_raw_str, cid)
                    all_norm_labels.add(s_norm)
                    all_norm_labels.add(o_val)
                    o_tok_norm = str(text_processing(o_val))
                    if o_tok_norm:
                        all_norm_labels.add(o_tok_norm)
                        _record_surface(o_tok_norm, o_raw_str, cid)
                    spo_d = st.spo_def.strip() or f"{s_norm} {p_norm} {o_val}."
                    raw_subj_d = st.subj_def.strip()
                    subj_d = raw_subj_d or stage1_defs.get(s_norm, "")
                    if subj_d:
                        if s_norm not in stage1_defs:
                            stage1_defs[s_norm] = subj_d
                        senses_lst = stage1_senses.setdefault(s_norm, [])
                        if all(d.lower() != subj_d.lower() for d in senses_lst):
                            senses_lst.append(subj_d)

                    raw_obj_d = st.obj_def.strip()
                    obj_d = raw_obj_d or stage1_defs.get(o_val, "") or stage1_defs.get(o_tok_norm, "")
                    if (
                        not is_literal_obj
                        and s_norm == o_val
                        and subj_d
                        and obj_d
                        and subj_d.lower() == obj_d.lower()
                        and len(stage1_senses.get(o_val, ())) >= 2
                    ):
                        obj_d = next(
                            (d for d in stage1_senses[o_val] if d.lower() != subj_d.lower()),
                            obj_d,
                        )
                    if obj_d:
                        if o_val not in stage1_defs:
                            stage1_defs[o_val] = obj_d
                        senses_lst = stage1_senses.setdefault(o_val, [])
                        if all(d.lower() != obj_d.lower() for d in senses_lst):
                            senses_lst.append(obj_d)

                    normalized_sts.append(
                        StructuredTriple(
                            subj=s_norm,
                            pred=p_norm,
                            obj=o_val,
                            spo_def=spo_d,
                            subj_def=subj_d,
                            obj_def=obj_d,
                            obj_kind="literal" if is_literal_obj else "entity",
                            obj_datatype=str(getattr(st, "obj_datatype", "") or ""),
                            obj_lang=str(getattr(st, "obj_lang", "") or ""),
                        )
                    )

                already_attempted = attempted_missing_by_cid.get(cid, set())
                missing_norms: list[str] = []
                seen_missing: set[str] = set(already_attempted)
                for st in normalized_sts:
                    if not st.subj_def and not stage1_defs.get(st.subj) and st.subj not in seen_missing:
                        seen_missing.add(st.subj)
                        missing_norms.append(st.subj)
                    if (
                        st.obj_kind != "literal"
                        and not st.obj_def
                        and not stage1_defs.get(st.obj)
                        and st.obj not in seen_missing
                    ):
                        seen_missing.add(st.obj)
                        missing_norms.append(st.obj)
                for e_norm, e_def in ner_entities:
                    if not e_def and not stage1_defs.get(e_norm) and e_norm not in seen_missing:
                        seen_missing.add(e_norm)
                        missing_norms.append(e_norm)

                if missing_norms and hasattr(active_program, "define_missing_entities"):
                    prev_def_audit_len = len(getattr(active_program, "entity_def_audit_trail", ()))
                    def_raw_trace: list[Any] = []
                    with dspy.context(trace=def_raw_trace):
                        repaired_pairs = active_program.define_missing_entities(
                            chunk_text_by_id.get(cid, ""),
                            missing_norms,
                            [
                                {
                                    "sub": st.subj,
                                    "pred": st.pred,
                                    "obj": st.obj,
                                    "def": st.spo_def,
                                }
                                for st in normalized_sts
                            ],
                        )
                    self._captured_chunk_traces.setdefault(cid, []).extend(
                        extract_named_trace_entries(active_program, def_raw_trace)
                    )
                    new_def_entries = list(getattr(active_program, "entity_def_audit_trail", ())[prev_def_audit_len:])
                    for entry in new_def_entries:
                        if isinstance(entry, dict):
                            self._graph_rlm_stats["missing_entity_defs_count"] += int(entry.get("missing_count", 0))
                            self._graph_rlm_stats["missing_entity_defs_tokens"] += int(entry.get("tokens", 0))
                    for r_name, r_def in repaired_pairs:
                        r_norm = str(text_processing(str(r_name)))
                        _record_surface(r_norm, str(r_name), cid)
                        r_clean = str(r_def).strip()
                        if r_norm and r_clean:
                            stage1_defs[r_norm] = r_clean

                if stage1_defs:
                    normalized_sts = [
                        StructuredTriple(
                            subj=st.subj,
                            pred=st.pred,
                            obj=st.obj,
                            spo_def=st.spo_def,
                            subj_def=st.subj_def or stage1_defs.get(st.subj, ""),
                            obj_def=(
                                st.obj_def
                                or stage1_defs.get(st.obj, "")
                                or stage1_defs.get(str(text_processing(st.obj)), "")
                            ),
                            obj_kind=st.obj_kind,
                            obj_datatype=st.obj_datatype,
                            obj_lang=st.obj_lang,
                        )
                        for st in normalized_sts
                    ]
                    ner_entities = [(e_norm, e_def or stage1_defs.get(e_norm, "")) for e_norm, e_def in ner_entities]

                record_structured.append((cid, normalized_sts, ner_entities))

            existing_by_label = get_entities_by_label_norm(self.conn, list(all_norm_labels))
            all_existing_eids = list({eid for eids in existing_by_label.values() for eid in eids})
            entity_info_by_id: dict[str, tuple[str, str]] = {}
            existing_vecs_by_id: dict[str, np.ndarray] = {}
            if all_existing_eids:
                entity_info_by_id = get_entity_metadata_batch(self.conn, all_existing_eids)
                raw_vecs = get_entity_vectors(self.conn, all_existing_eids)
                for eid_v, vec_v in raw_vecs.items():
                    v_norm = float(np.linalg.norm(vec_v))
                    existing_vecs_by_id[eid_v] = vec_v / v_norm if v_norm > 0 else vec_v

            # Bridge copular Stage 1 entity definitions to known graph concepts so definitions
            # absorbed by Stage 1 NER still form explicit graph edges and retrievable facts.
            if len(chunks) > 1:
                known_concepts = {
                    lbl
                    for lbl in all_norm_labels
                    if 3 <= len(lbl) <= 30 and not lbl.isdigit() and len(lbl.split()) <= 3
                }
                for ptr, _ in chunks:
                    if ptr.title:
                        t_norm = str(text_processing(ptr.title))
                        _record_surface(t_norm, ptr.title, ptr.chunk_id)
                        if 3 <= len(t_norm) <= 30 and not t_norm.isdigit():
                            known_concepts.add(t_norm)
                for _cid, normalized_sts, ner_entities in record_structured:
                    existing_pairs = {(st.subj, st.obj) for st in normalized_sts} | {
                        (st.obj, st.subj) for st in normalized_sts
                    }
                    for e_norm, e_def in ner_entities[:2]:
                        if not e_def or len(e_norm) < 3:
                            continue
                        first_clause = re.split(r"\.(?:\s|$)", e_def, maxsplit=1)[0].lower()
                        clause_words = set(str(text_processing(first_clause)).split()) | set(
                            re.findall(r"[a-z0-9]+", first_clause)
                        )
                        for c_norm in known_concepts:
                            if c_norm == e_norm or c_norm in e_norm or e_norm in c_norm:
                                continue
                            c_tokens = c_norm.split()
                            if not all(tok in clause_words for tok in c_tokens):
                                continue
                            if (e_norm, c_norm) in existing_pairs:
                                continue
                            if re.search(rf"(?:^|(?<=\s)|(?<=\W)){re.escape(c_norm)}(?=$|\s|\W)", first_clause):
                                existing_pairs.add((e_norm, c_norm))
                                normalized_sts.append(
                                    StructuredTriple(
                                        subj=e_norm,
                                        pred="is a",
                                        obj=c_norm,
                                        spo_def=e_def,
                                        subj_def=e_def,
                                        obj_def="",
                                    )
                                )

            # Track senses per label_norm: list of (entity_id, name, definition, unit_vec_or_None)
            senses_by_label: dict[str, list[tuple[str, str, str, np.ndarray | None]]] = {}
            for lbl, eids in existing_by_label.items():
                senses_by_label[lbl] = [
                    (
                        eid,
                        entity_info_by_id.get(eid, (surface_by_norm.get(lbl, lbl), ""))[0],
                        entity_info_by_id.get(eid, (surface_by_norm.get(lbl, lbl), ""))[1],
                        existing_vecs_by_id.get(eid),
                    )
                    for eid in eids
                ]

            # Pre-batch candidate disambiguation embeddings across the batch to avoid 1-by-1 HTTP calls
            defs_by_label: dict[str, set[str]] = {}
            for lbl, senses in senses_by_label.items():
                for _, _, s_def, _ in senses:
                    if s_def.strip():
                        defs_by_label.setdefault(lbl, set()).add(s_def.strip().lower())
            cand_pairs_by_label: dict[str, list[tuple[str, str]]] = {}
            for _cid, st_list, ner_entities in record_structured:
                for st in st_list:
                    for m_lbl, m_def in ((st.subj, st.subj_def), (st.obj, st.obj_def)):
                        c_def = m_def.strip()
                        if c_def:
                            defs_by_label.setdefault(m_lbl, set()).add(c_def.lower())
                            cand_pairs_by_label.setdefault(m_lbl, []).append((m_lbl, c_def))
                for e_norm, e_def in ner_entities:
                    c_def = e_def.strip()
                    if c_def:
                        defs_by_label.setdefault(e_norm, set()).add(c_def.lower())
                        cand_pairs_by_label.setdefault(e_norm, []).append((e_norm, c_def))
            prebatch_texts: list[str] = []
            seen_prebatch: set[str] = set()
            for lbl, dset in defs_by_label.items():
                if len(dset) >= 2:
                    for m_lbl, c_def in cand_pairs_by_label.get(lbl, ()):
                        sense_cand_id = compute_mdhash_id(f"{m_lbl}::{c_def.lower()}", prefix="entity-")
                        if sense_cand_id in existing_vecs_by_id:
                            continue
                        ct = fmt_entity(surface_by_norm.get(m_lbl, m_lbl), c_def)
                        if ct not in seen_prebatch:
                            seen_prebatch.add(ct)
                            prebatch_texts.append(ct)
                    for _, name_s, prev_def_s, prev_vec_s in senses_by_label.get(lbl, ()):
                        if prev_vec_s is None and prev_def_s:
                            pt = fmt_entity(name_s, prev_def_s)
                            if pt not in seen_prebatch:
                                seen_prebatch.add(pt)
                                prebatch_texts.append(pt)
            if prebatch_texts:
                self._encode_in_batches(prebatch_texts, instruction="", stage="entity_embed")

            entity_map: dict[str, tuple[str, str, str]] = {}
            entity_labels_to_upsert: list[tuple[str, str, str, str, str]] = []
            precomputed_entity_vecs: dict[str, np.ndarray] = {}

            def _label_kind_for(entity_name: str, lbl_norm: str) -> str:
                return (
                    "skos:prefLabel"
                    if (
                        lbl_norm == entity_name
                        or lbl_norm.lower() == entity_name.lower()
                        or lbl_norm == str(text_processing(entity_name))
                    )
                    else "skos:altLabel"
                )

            def _resolve_mention_entity_id(
                label_norm: str,
                defn: str,
                chunk_id: str,
                *,
                exclude_eid: str | None = None,
            ) -> str:
                clean_def = defn.strip()
                canon_surface = surface_by_norm.get(label_norm, label_norm)
                mention_raw = chunk_surface_by_norm.get((chunk_id, label_norm), canon_surface)
                existing_senses = senses_by_label.get(label_norm, [])
                if not existing_senses:
                    eid = compute_mdhash_id(label_norm, prefix="entity-")
                    senses_by_label[label_norm] = [(eid, canon_surface, clean_def, existing_vecs_by_id.get(eid))]
                    entity_map[eid] = (eid, canon_surface, clean_def)
                    entity_labels_to_upsert.append((label_norm, eid, "skos:prefLabel", mention_raw, chunk_id))
                    return eid

                def _best_surface(curr_name: str) -> str:
                    if (
                        curr_name == curr_name.lower()
                        and canon_surface != canon_surface.lower()
                        and str(text_processing(curr_name)) == label_norm
                    ):
                        return canon_surface
                    return curr_name

                if not clean_def:
                    for idx_s, (eid, name, prev_def, prev_vec) in enumerate(existing_senses):
                        if exclude_eid is not None and eid == exclude_eid:
                            continue
                        best_name = _best_surface(name)
                        if best_name != name:
                            existing_senses[idx_s] = (eid, best_name, prev_def, prev_vec)
                            entity_map[eid] = (eid, best_name, prev_def)
                        elif eid not in entity_map:
                            entity_map[eid] = (eid, best_name, prev_def)
                        entity_labels_to_upsert.append(
                            (
                                label_norm,
                                eid,
                                _label_kind_for(best_name, label_norm),
                                mention_raw,
                                chunk_id,
                            )
                        )
                        return eid
                    eid, name, prev_def, prev_vec = existing_senses[0]
                    best_name = _best_surface(name)
                    if best_name != name:
                        existing_senses[0] = (eid, best_name, prev_def, prev_vec)
                        entity_map[eid] = (eid, best_name, prev_def)
                    elif eid not in entity_map:
                        entity_map[eid] = (eid, best_name, prev_def)
                    entity_labels_to_upsert.append(
                        (label_norm, eid, _label_kind_for(best_name, label_norm), mention_raw, chunk_id)
                    )
                    return eid

                # Check if any existing sense has no definition yet or identical definition
                for idx_s, (eid, name, prev_def, prev_vec) in enumerate(existing_senses):
                    if exclude_eid is not None and eid == exclude_eid:
                        continue
                    if not prev_def or prev_def.lower() == clean_def.lower():
                        best_name = _best_surface(name)
                        merged_def = clean_def if len(clean_def) > len(prev_def) else (prev_def or clean_def)
                        existing_senses[idx_s] = (
                            eid,
                            best_name,
                            merged_def,
                            prev_vec if (merged_def == prev_def and best_name == name) else None,
                        )
                        entity_map[eid] = (eid, best_name, merged_def)
                        entity_labels_to_upsert.append(
                            (
                                label_norm,
                                eid,
                                _label_kind_for(best_name, label_norm),
                                mention_raw,
                                chunk_id,
                            )
                        )
                        return eid

                # Both have distinct non-empty definitions:
                # Reuse existing concept only when high-confidence same sense (cos >= 0.85).
                # When cos < 0.85 (whether in [0.65, 0.85) or < 0.65), mint a provisional sense_id
                # WITHOUT writing any owl:differentFrom edge in Step 4 so Step 6 judge_synonyms
                # adjudicates every ambiguous or divergent same-label pair.
                sense_id = compute_mdhash_id(f"{label_norm}::{clean_def.lower()}", prefix="entity-")
                cached_sense_vec = existing_vecs_by_id.get(sense_id)
                if cached_sense_vec is None:
                    cached_sense_vec = precomputed_entity_vecs.get(sense_id)
                if cached_sense_vec is not None:
                    cand_vec = cached_sense_vec
                else:
                    cand_text = fmt_entity(canon_surface, clean_def)
                    cand_vec = self._encode_in_batches([cand_text], instruction="", stage="entity_embed")[0]
                    c_norm = float(np.linalg.norm(cand_vec))
                    if c_norm > 0:
                        cand_vec = cand_vec / c_norm

                for idx_s, (eid, name, prev_def, prev_vec) in enumerate(existing_senses):
                    if exclude_eid is not None and eid == exclude_eid:
                        continue
                    if prev_vec is None:
                        p_text = fmt_entity(name, prev_def)
                        prev_vec = self._encode_in_batches([p_text], instruction="", stage="entity_embed")[0]
                        p_norm = float(np.linalg.norm(prev_vec))
                        if p_norm > 0:
                            prev_vec = prev_vec / p_norm
                        existing_senses[idx_s] = (eid, name, prev_def, prev_vec)
                        precomputed_entity_vecs[eid] = prev_vec
                    cos_sim = float(np.dot(cand_vec, prev_vec))
                    if cos_sim >= self.same_sense_cosine_gate:
                        best_name = _best_surface(name)
                        merged_def = clean_def if len(clean_def) > len(prev_def) else prev_def
                        if merged_def != prev_def or best_name != name:
                            existing_senses[idx_s] = (eid, best_name, merged_def, cand_vec)
                            precomputed_entity_vecs[eid] = cand_vec
                        entity_map[eid] = (eid, best_name, merged_def)
                        entity_labels_to_upsert.append(
                            (
                                label_norm,
                                eid,
                                _label_kind_for(best_name, label_norm),
                                mention_raw,
                                chunk_id,
                            )
                        )
                        return eid

                sense_id = compute_mdhash_id(f"{label_norm}::{clean_def.lower()}", prefix="entity-")
                precomputed_entity_vecs[sense_id] = cand_vec
                existing_senses.append((sense_id, canon_surface, clean_def, cand_vec))
                entity_map[sense_id] = (sense_id, canon_surface, clean_def)
                entity_labels_to_upsert.append((label_norm, sense_id, "skos:prefLabel", mention_raw, chunk_id))
                self._graph_rlm_stats["homograph_splits"] += 1
                return sense_id

            chunk_to_entities: dict[str, set[str]] = {}
            fact_nodes_map: dict[
                str,
                tuple[str, str, str, str | None, str, str, str, str, str, str | None, str, str],
            ] = {}
            fact_sources_batch: list[tuple[str, str]] = []
            edges_with_sources: list[tuple[str, str, str, float, str]] = []
            fact_linked_pairs: set[tuple[str, str]] = set()

            for cid, st_list, ner_entities in record_structured:
                chunk_to_entities.setdefault(cid, set())
                resolved_in_chunk: set[tuple[str, str]] = set()
                for st in st_list:
                    s_id = _resolve_mention_entity_id(st.subj, st.subj_def, cid)
                    resolved_in_chunk.add((st.subj, st.subj_def.strip().lower()))
                    chunk_to_entities[cid].add(s_id)

                    if st.obj_kind == "literal":
                        o_id: str | None = None
                        if not (st.obj_datatype.startswith("xsd:") and not st.obj_def.strip()):
                            eff_obj_def = st.obj_def.strip() or st.spo_def.strip() or f"Literal value {st.obj}."
                            o_id = _resolve_mention_entity_id(st.obj, eff_obj_def, cid)
                            resolved_in_chunk.add((st.obj, st.obj_def.strip().lower()))
                            chunk_to_entities[cid].add(o_id)
                        eff_obj_dt = st.obj_datatype or "rdfs:Literal"
                        fact_oid = o_id if (o_id is not None and o_id != s_id) else None
                        fid = compute_mdhash_id(
                            str((s_id, st.pred, f"literal:{st.obj}:{eff_obj_dt}:{st.obj_lang}")),
                            prefix="fact-",
                        )
                        raw_json = json.dumps([st.subj, st.pred, st.obj])
                        prev_fact = fact_nodes_map.get(fid)
                        if prev_fact is None or (not prev_fact[5] and st.spo_def):
                            fact_nodes_map[fid] = (
                                fid,
                                s_id,
                                st.pred,
                                fact_oid,
                                raw_json,
                                st.spo_def,
                                st.subj_def,
                                st.obj_def,
                                "literal",
                                st.obj,
                                eff_obj_dt,
                                st.obj_lang,
                            )
                        fact_sources_batch.append((fid, cid))
                        if fact_oid is not None:
                            dt_weight = (
                                float(self.traversal_policy.datatype_property)
                                if self.traversal_policy is not None
                                else 0.5
                            )
                            fact_linked_pairs.add((min(s_id, fact_oid), max(s_id, fact_oid)))
                            edges_with_sources.append((s_id, fact_oid, "fact", dt_weight, cid))
                    else:
                        distinct_homograph_pole = (
                            st.subj == st.obj
                            and bool(st.subj_def.strip())
                            and bool(st.obj_def.strip())
                            and st.subj_def.strip().lower() != st.obj_def.strip().lower()
                        )
                        o_id = _resolve_mention_entity_id(
                            st.obj,
                            st.obj_def,
                            cid,
                            exclude_eid=s_id if distinct_homograph_pole else None,
                        )
                        resolved_in_chunk.add((st.obj, st.obj_def.strip().lower()))
                        chunk_to_entities[cid].add(o_id)

                        base_s_id = compute_mdhash_id(st.subj, prefix="entity-")
                        base_o_id = compute_mdhash_id(st.obj, prefix="entity-")
                        if st.subj == st.obj or s_id != base_s_id or o_id != base_o_id:
                            fid = compute_mdhash_id(str((s_id, st.pred, o_id)), prefix="fact-")
                        else:
                            raw_triple_str = str((st.subj, st.pred, st.obj))
                            fid = compute_mdhash_id(raw_triple_str, prefix="fact-")
                        raw_json = json.dumps([st.subj, st.pred, st.obj])
                        prev_fact = fact_nodes_map.get(fid)
                        if prev_fact is None or (not prev_fact[5] and st.spo_def):
                            fact_nodes_map[fid] = (
                                fid,
                                s_id,
                                st.pred,
                                o_id,
                                raw_json,
                                st.spo_def,
                                st.subj_def,
                                st.obj_def,
                                "entity",
                                None,
                                "",
                                "",
                            )
                        fact_sources_batch.append((fid, cid))
                        if s_id != o_id:
                            fact_linked_pairs.add((min(s_id, o_id), max(s_id, o_id)))
                            edges_with_sources.append((s_id, o_id, "fact", 1.0, cid))

                for e_norm, e_def in ner_entities:
                    e_key = (e_norm, e_def.strip().lower())
                    if e_key in resolved_in_chunk:
                        continue
                    if not e_def.strip() and any(k[0] == e_norm for k in resolved_in_chunk):
                        continue
                    resolved_in_chunk.add(e_key)
                    ner_eid = _resolve_mention_entity_id(e_norm, e_def.strip(), cid)
                    chunk_to_entities[cid].add(ner_eid)

            for cid, eids in chunk_to_entities.items():
                for eid in eids:
                    edges_with_sources.append((cid, eid, "passage", 1.0, cid))
                    edges_with_sources.append((eid, cid, "passage", 1.0, cid))

            entity_nodes = list(entity_map.values())
            fact_nodes = list(fact_nodes_map.values())

            # Compute missing entity vectors using contextual `# {name}\n{definition}`
            all_entity_ids = [eid for eid, _, _ in entity_nodes]
            existing_entity_vecs = get_existing_ids(self.conn, "entity_vec", all_entity_ids)
            missing_entities = [
                (eid, name, defn)
                for eid, name, defn in entity_nodes
                if eid not in existing_entity_vecs and eid not in precomputed_entity_vecs
            ]
            entity_vec_pairs: list[tuple[str, np.ndarray]] = [
                (eid, vec) for eid, vec in precomputed_entity_vecs.items() if eid not in existing_entity_vecs
            ]
            if missing_entities:
                if progress_callback is not None:
                    progress_callback(f"indexing: embedding {len(missing_entities)} entity vectors")
                e_texts = [fmt_entity(name, defn) for _, name, defn in missing_entities]
                e_vecs = self._encode_in_batches(e_texts, instruction="", stage="entity_embed")
                entity_vec_pairs.extend((eid, v) for (eid, _, _), v in zip(missing_entities, e_vecs, strict=False))

            # Compute missing fact vectors using contextual `# {s} {p} {o}\n{spo_def} | {subj_def} | {obj_def}`
            all_fact_ids = [f_row[0] for f_row in fact_nodes]
            existing_fact_vecs = get_existing_ids(self.conn, "fact_vec", all_fact_ids)
            missing_facts = [f_row for f_row in fact_nodes if f_row[0] not in existing_fact_vecs]
            fact_vec_pairs: list[tuple[str, np.ndarray]] = []
            if missing_facts:
                if progress_callback is not None:
                    progress_callback(f"indexing: embedding {len(missing_facts)} fact vectors")
                fact_texts: list[str] = []
                for f_row in missing_facts:
                    _, _s_id, pred, _o_id, raw_json, spo_d, subj_d, obj_d = f_row[:8]
                    items = json.loads(raw_json)
                    fact_texts.append(fmt_fact(items[0], pred, items[2], spo_d, subj_d, obj_d))
                f_vecs = self._encode_in_batches(fact_texts, instruction="", stage="fact_embed")
                fact_vec_pairs = [(f_row[0], v) for f_row, v in zip(missing_facts, f_vecs, strict=False)]

            # 4. Atomic Closed-World Transaction: Upsert Graph + EDC Disambiguation + SHACL Self-Healing
            if progress_callback is not None:
                progress_callback("indexing: committing SKOS/RDF/OWL graph & SHACL validation")
            with self.managed_conn.transaction() as conn:
                if chunk_vec_pairs:
                    upsert_chunk_vectors(conn, chunk_vec_pairs)
                # Commit ChunkPointer metadata and FTS5 index first so RLM tools can hydrate chunks
                upsert_chunk_pointers(conn, chunks)
                upsert_entities(conn, entity_nodes)
                upsert_entity_labels(conn, entity_labels_to_upsert)
                upsert_facts(
                    conn,
                    [
                        (fid, s_id, pred, o_id, raw_json, spo_d, okind, olex, odt, olang)
                        for (
                            fid,
                            s_id,
                            pred,
                            o_id,
                            raw_json,
                            spo_d,
                            _subj_d,
                            _obj_d,
                            okind,
                            olex,
                            odt,
                            olang,
                        ) in fact_nodes
                    ],
                )
                if fact_sources_batch:
                    upsert_fact_sources(conn, fact_sources_batch)
                if entity_vec_pairs:
                    upsert_entity_vectors(conn, entity_vec_pairs)
                if fact_vec_pairs:
                    upsert_fact_vectors(conn, fact_vec_pairs)
                upsert_edges_with_sources(conn, edges_with_sources)

                concept_eids = {
                    str(r[0])
                    for r in conn.execute(
                        """
                        SELECT subj_id FROM facts
                        UNION
                        SELECT obj_id FROM facts
                        WHERE COALESCE(obj_kind, 'entity') = 'entity' AND obj_id IS NOT NULL
                        """
                    )
                }
                lit_only_eids = {
                    str(r[0])
                    for r in conn.execute(
                        "SELECT DISTINCT obj_id FROM facts WHERE obj_kind = 'literal' AND obj_id IS NOT NULL"
                    )
                    if str(r[0]) not in concept_eids
                }
                valid_entity_ids = [
                    eid
                    for eid, name, _ in entity_nodes
                    if len(re.sub(r"[^A-Za-z0-9]", "", name)) >= 1
                    and not (detect_scalar_literal(name)[0] and eid not in concept_eids)
                    and eid not in lit_only_eids
                ]
                if valid_entity_ids:
                    raw_candidates = find_entity_disambiguation_candidates(
                        conn,
                        entity_ids=valid_entity_ids,
                        dense_threshold=self.synonymy_threshold,
                        dense_top_k=self.synonymy_top_k,
                    )
                    self._graph_rlm_stats["candidate_pairs"] += len(raw_candidates)

                    cand_eids = list({eid for ea, eb, _, _ in raw_candidates for eid in (ea, eb)})
                    cand_meta = get_entity_metadata_batch(conn, cand_eids)

                    candidates: list[dict[str, Any]] = []
                    for ea, eb, d_sim, l_sim in raw_candidates:
                        name_a, def_a = cand_meta.get(ea, (ea, ""))
                        name_b, def_b = cand_meta.get(eb, (eb, ""))
                        sim = max(float(d_sim), float(l_sim))
                        candidates.append(
                            {
                                "pair_id": f"{ea}|{eb}",
                                "entity_a": ea,
                                "name_a": name_a,
                                "def_a": def_a,
                                "entity_b": eb,
                                "name_b": name_b,
                                "def_b": def_b,
                                "dense_sim": float(d_sim),
                                "lexical_sim": float(l_sim),
                                "similarity": sim,
                            }
                        )

                    candidates.sort(
                        key=lambda c: (
                            text_processing(str(c["name_a"])) == text_processing(str(c["name_b"])),
                            float(c["lexical_sim"]) >= 0.72,
                            0.5 * float(c["lexical_sim"]) + 0.5 * float(c["dense_sim"]),
                            float(c["similarity"]),
                        ),
                        reverse=True,
                    )
                    decisions: list[dict[str, str]] = []
                    if candidates:
                        syn_fp = synonym_fingerprint(active_program)
                        pairs_fp = hashlib.sha256(
                            json.dumps(candidates, sort_keys=True, ensure_ascii=False).encode("utf-8")
                        ).hexdigest()
                        cached_syn = self._synonym_fp_cache.get((syn_fp, pairs_fp))
                        if cached_syn is not None:
                            cached_decs, cached_syn_tr, cached_syn_tok = cached_syn
                            decisions = list(cached_decs)
                            self._captured_graph_traces.extend(cached_syn_tr)
                            self._graph_rlm_stats["synonym_judge_tokens"] += cached_syn_tok
                        else:
                            syn_audit: list[dict[str, Any]] = getattr(active_program, "synonym_audit_trail", [])
                            tel: list[dict[str, Any]] = getattr(active_program, "telemetry", [])
                            prev_syn_audit_len = len(syn_audit)
                            prev_tel_len = len(tel)
                            local_syn_trace: list[Any] = []
                            with dspy.context(trace=local_syn_trace, usage_tracker=None):
                                decisions = active_program.judge_synonyms(candidates)
                            syn_traces = extract_named_trace_entries(
                                active_program,
                                local_syn_trace,
                                allowed_names={"synonym_judge"},
                            )
                            self._captured_graph_traces.extend(syn_traces)
                            syn_tok = 0
                            if len(syn_audit) > prev_syn_audit_len:
                                syn_tok = int(syn_audit[-1].get("tokens_used", 0))
                            elif len(tel) > prev_tel_len:
                                syn_tok = int(tel[-1].get("tokens_used", 0))
                            self._graph_rlm_stats["synonym_judge_tokens"] += syn_tok
                            self._synonym_fp_cache[(syn_fp, pairs_fp)] = (
                                list(decisions),
                                list(syn_traces),
                                syn_tok,
                            )

                    merged_into: dict[str, str] = {}

                    def _resolve_active_eid(eid: str) -> str:
                        curr = eid
                        visited: set[str] = set()
                        while curr in merged_into and curr not in visited:
                            visited.add(curr)
                            curr = merged_into[curr]
                        return curr

                    semantic_batch: list[tuple[str, str, str, float]] = []
                    cand_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
                    for c in candidates:
                        ea, eb = str(c["entity_a"]), str(c["entity_b"])
                        cand_by_pair[(min(ea, eb), max(ea, eb))] = c

                    adjudicated_pairs: set[tuple[str, str]] = set()
                    for dec in decisions:
                        pid = str(dec.get("pair_id", "")).strip()
                        if "|" in pid:
                            ea, eb = pid.split("|", 1)
                            raw_src, raw_tgt = ea.strip(), eb.strip()
                        else:
                            raw_src = str(dec.get("source_id", "")).strip()
                            raw_tgt = str(dec.get("target_id", "")).strip()
                        rel = str(dec.get("relation") or dec.get("decision") or "").strip()
                        canon = str(dec.get("canonical", "")).strip()
                        if not raw_src or not raw_tgt:
                            continue
                        pkey = (min(raw_src, raw_tgt), max(raw_src, raw_tgt))
                        if pkey not in cand_by_pair:
                            continue
                        adjudicated_pairs.add(pkey)
                        src = _resolve_active_eid(raw_src)
                        tgt = _resolve_active_eid(raw_tgt)
                        if src == tgt:
                            continue
                        cand_info = cand_by_pair.get(pkey, {})
                        sim = float(cand_info.get("similarity", 0.5) or 0.5)
                        d_sim = float(cand_info.get("dense_sim", 0.0) or 0.0)
                        name_src, _ = cand_meta.get(src, cand_meta.get(raw_src, (src, "")))
                        name_tgt, _ = cand_meta.get(tgt, cand_meta.get(raw_tgt, (tgt, "")))
                        same_name = text_processing(name_src) == text_processing(name_tgt)
                        active_pkey = (min(src, tgt), max(src, tgt))
                        is_fact_pole_pair = active_pkey in fact_linked_pairs
                        if (
                            rel not in ("owl:differentFrom", "different", "different_from")
                            and is_acronym_pair(name_src, name_tgt)
                            and not (same_name and is_fact_pole_pair)
                        ):
                            if len(name_src) < len(name_tgt):
                                merge_src, merge_dst = src, tgt
                            else:
                                merge_src, merge_dst = tgt, src
                            merge_entities(conn, merge_src, merge_dst, label_kind="skos:altLabel")
                            merged_into[merge_src] = merge_dst
                            self._graph_rlm_stats["merged_entities"] += 1
                            continue
                        if rel in ("skos:altLabel", "skos:hiddenLabel", "exact_match") and not (
                            same_name and is_fact_pole_pair
                        ):
                            if canon and (
                                canon.upper() == "A"
                                or canon in (src, raw_src)
                                or (
                                    text_processing(canon) == text_processing(name_src)
                                    and text_processing(canon) != text_processing(name_tgt)
                                )
                            ):
                                merge_src, merge_dst = tgt, src
                            else:
                                merge_src, merge_dst = src, tgt
                            lkind = (
                                "skos:hiddenLabel"
                                if rel == "skos:hiddenLabel" or str(dec.get("label_type", "")) == "hiddenLabel"
                                else "skos:altLabel"
                            )
                            merge_entities(conn, merge_src, merge_dst, label_kind=lkind)
                            merged_into[merge_src] = merge_dst
                            self._graph_rlm_stats["merged_entities"] += 1
                        elif rel in ("owl:differentFrom", "different", "different_from"):
                            semantic_batch.append((src, tgt, "owl:differentFrom", 0.0))
                            self._graph_rlm_stats["different_from_edges"] += 1
                        elif rel == "skos:broader":
                            # 1. Version/Ordinal Sibling Guard: If version/ordinal siblings, demote to skos:related
                            if is_version_or_ordinal_sibling(name_src, name_tgt):
                                semantic_batch.append((src, tgt, "skos:related", max(0.5, sim)))
                                self._graph_rlm_stats["related_edges"] += 1
                                continue

                            # 2. Missing/Invalid Canonical Guard: If canonical is missing/empty, demote to skos:related
                            if canon and (
                                canon.upper() == "A"
                                or canon in (src, raw_src)
                                or (
                                    text_processing(canon) == text_processing(name_src)
                                    and text_processing(canon) != text_processing(name_tgt)
                                )
                            ):
                                child_id, parent_id = tgt, src
                                child_name, parent_name = name_tgt, name_src
                            elif canon and (
                                canon.upper() == "B"
                                or canon in (tgt, raw_tgt)
                                or (
                                    text_processing(canon) == text_processing(name_tgt)
                                    and text_processing(canon) != text_processing(name_src)
                                )
                            ):
                                child_id, parent_id = src, tgt
                                child_name, parent_name = name_src, name_tgt
                            else:
                                semantic_batch.append((src, tgt, "skos:related", max(0.5, sim)))
                                self._graph_rlm_stats["related_edges"] += 1
                                continue

                            # 3. Lexical Subsumption Inversion Guard:
                            if check_lexical_subsumption_inversion(child_name, parent_name):
                                child_id, parent_id = parent_id, child_id

                            semantic_batch.append((child_id, parent_id, "skos:broader", max(0.5, sim)))
                            self._graph_rlm_stats["broader_edges"] = (
                                int(self._graph_rlm_stats.get("broader_edges", 0)) + 1
                            )
                        elif rel in ("skos:related", "related"):
                            if same_name:
                                semantic_batch.append((src, tgt, "owl:differentFrom", 0.0))
                                self._graph_rlm_stats["different_from_edges"] += 1
                                semantic_batch.append((src, tgt, "skos:related", max(0.5, sim)))
                                self._graph_rlm_stats["related_edges"] += 1
                            else:
                                semantic_batch.append((src, tgt, "skos:related", max(0.5, sim)))
                                self._graph_rlm_stats["related_edges"] += 1
                        elif same_name and is_fact_pole_pair:
                            semantic_batch.append((src, tgt, "owl:differentFrom", 0.0))
                            self._graph_rlm_stats["different_from_edges"] += 1

                    for c in candidates:
                        ea, eb = str(c["entity_a"]), str(c["entity_b"])
                        pkey = (min(ea, eb), max(ea, eb))
                        if pkey in adjudicated_pairs:
                            continue
                        src = _resolve_active_eid(ea)
                        tgt = _resolve_active_eid(eb)
                        if src == tgt:
                            continue
                        name_src, _ = cand_meta.get(src, (str(c["name_a"]), ""))
                        name_tgt, _ = cand_meta.get(tgt, (str(c["name_b"]), ""))
                        same_name = text_processing(name_src) == text_processing(name_tgt)
                        active_pkey = (min(src, tgt), max(src, tgt))
                        if is_acronym_pair(name_src, name_tgt) and not (
                            same_name and active_pkey in fact_linked_pairs
                        ):
                            merge_src, merge_dst = (src, tgt) if len(name_src) < len(name_tgt) else (tgt, src)
                            merge_entities(conn, merge_src, merge_dst, label_kind="skos:altLabel")
                            merged_into[merge_src] = merge_dst
                            self._graph_rlm_stats["merged_entities"] += 1
                        elif same_name:
                            if float(c.get("dense_sim", 0.0)) >= 0.45 and active_pkey not in fact_linked_pairs:
                                merge_entities(conn, src, tgt, label_kind="skos:altLabel")
                                merged_into[src] = tgt
                                self._graph_rlm_stats["merged_entities"] += 1
                            else:
                                semantic_batch.append((src, tgt, "owl:differentFrom", 0.0))
                                self._graph_rlm_stats["different_from_edges"] += 1
                        elif float(c.get("dense_sim", 0.0)) >= self.synonymy_threshold:
                            sim = float(c.get("similarity", 0.5) or 0.5)
                            semantic_batch.append((src, tgt, "skos:related", max(0.5, sim)))
                            self._graph_rlm_stats["related_edges"] += 1

                    if semantic_batch:
                        resolved_semantic_batch = [
                            (_resolve_active_eid(s), _resolve_active_eid(d), k, w)
                            for s, d, k, w in semantic_batch
                            if _resolve_active_eid(s) != _resolve_active_eid(d)
                        ]
                        if resolved_semantic_batch:
                            upsert_semantic_edges(conn, resolved_semantic_batch)

                # In-transaction Closed-World SHACL Linting & Bounded Self-Healing via shacl_healer_rlm
                batch_node_ids = set(chunk_ids) | set(all_entity_ids) | set(all_fact_ids)
                pre_heal_violations = lint_graph_shacl(conn, auto_heal=False)
                pre_heal_batch_violations = [
                    v
                    for v in pre_heal_violations
                    if (v.focus_node, v.shape) not in self._attempted_shacl_keys
                    or v.focus_node in batch_node_ids
                    or ("|" in v.focus_node and any(p.strip() in batch_node_ids for p in v.focus_node.split("|")))
                    or ("->" in v.focus_node and any(p.strip() in batch_node_ids for p in v.focus_node.split("->")))
                ]
                batch_shacl_before = len(pre_heal_batch_violations)
                self._graph_rlm_stats["shacl_violations_before"] += batch_shacl_before
                self._graph_rlm_stats["shacl_violations_initial"] = self._graph_rlm_stats["shacl_violations_before"]
                batch_attempted_keys: set[tuple[str, str]] = {
                    (v.focus_node, v.shape) for v in pre_heal_batch_violations
                }
                all_violations = lint_graph_shacl(conn, auto_heal=True, heal_definitions=False)
                violations = [
                    v
                    for v in all_violations
                    if (v.focus_node, v.shape) not in self._attempted_shacl_keys
                    or v.focus_node in batch_node_ids
                    or ("|" in v.focus_node and any(p.strip() in batch_node_ids for p in v.focus_node.split("|")))
                    or ("->" in v.focus_node and any(p.strip() in batch_node_ids for p in v.focus_node.split("->")))
                ]
                if violations:
                    for v in violations:
                        v_key = (v.focus_node, v.shape)
                        self._attempted_shacl_keys.add(v_key)
                        batch_attempted_keys.add(v_key)
                    bounded_violations = violations
                    focus_contexts = [
                        inspect_focus_node(conn, v.focus_node)
                        for v in violations[: scale_context_budget(16, active_program.context_window_tokens)]
                    ]
                    shacl_fp = shacl_fingerprint(active_program)
                    viol_fp = hashlib.sha256(
                        json.dumps(
                            {
                                "violations": [v.to_dict() for v in bounded_violations],
                                "focus_contexts": focus_contexts,
                                "graph_context": self._graph_cache_context(),
                            },
                            sort_keys=True,
                            ensure_ascii=False,
                            default=str,
                        ).encode("utf-8")
                    ).hexdigest()
                    cached_shacl = self._shacl_fp_cache.get((shacl_fp, viol_fp))
                    if cached_shacl is not None:
                        cached_reps, cached_shacl_tr, cached_shacl_tok = cached_shacl
                        self._captured_graph_traces.extend(cached_shacl_tr)
                        self._graph_rlm_stats["shacl_healer_tokens"] += cached_shacl_tok
                        if cached_reps:
                            applied = self._apply_shacl_repairs(conn, cached_reps)
                            self._graph_rlm_stats["shacl_repairs_applied"] += applied
                        all_violations = lint_graph_shacl(conn, auto_heal=True)
                    else:
                        shacl_audit: list[dict[str, Any]] = getattr(active_program, "shacl_audit_trail", [])
                        shacl_tel: list[dict[str, Any]] = getattr(active_program, "telemetry", [])
                        prev_shacl_audit_len = len(shacl_audit)
                        prev_tel_len = len(shacl_tel)
                        self._active_repl_repair_keys = set()
                        self._active_repl_repairs_list = []
                        local_shacl_trace: list[Any] = []
                        shacl_succeeded = True
                        try:
                            with dspy.context(trace=local_shacl_trace, usage_tracker=None):
                                repairs = active_program.heal_shacl_violations(
                                    bounded_violations,
                                    focus_context=json.dumps(focus_contexts, ensure_ascii=False),
                                )
                        except Exception as exc:
                            import logging

                            logging.warning(
                                "heal_shacl_violations encountered error: %s. Continuing indexing.",
                                exc,
                            )
                            repairs = []
                            shacl_succeeded = False
                        finally:
                            repl_applied_keys = (
                                set(self._active_repl_repair_keys) if self._active_repl_repair_keys else set()
                            )
                            repl_repairs_list = (
                                list(self._active_repl_repairs_list) if self._active_repl_repairs_list else []
                            )
                            self._active_repl_repair_keys = None
                            self._active_repl_repairs_list = None
                        shacl_traces = extract_named_trace_entries(
                            active_program,
                            local_shacl_trace,
                            allowed_names={"shacl_repair", "shacl_healer_rlm"},
                        )
                        self._captured_graph_traces.extend(shacl_traces)
                        shacl_tok = 0
                        if len(shacl_audit) > prev_shacl_audit_len:
                            shacl_tok = int(shacl_audit[-1].get("tokens_used", 0))
                        elif len(shacl_tel) > prev_tel_len:
                            shacl_tok = int(shacl_tel[-1].get("tokens_used", 0))
                        self._graph_rlm_stats["shacl_healer_tokens"] += shacl_tok
                        unapplied_repairs = [
                            {str(k): str(v) for k, v in r.items()}
                            for r in repairs
                            if isinstance(r, Mapping) and self._repair_signature(r) not in repl_applied_keys
                        ]
                        if unapplied_repairs:
                            applied = self._apply_shacl_repairs(conn, unapplied_repairs)
                            self._graph_rlm_stats["shacl_repairs_applied"] += applied
                        all_repairs_for_cache = [*repl_repairs_list, *unapplied_repairs]
                        if shacl_succeeded:
                            self._shacl_fp_cache[(shacl_fp, viol_fp)] = (
                                all_repairs_for_cache,
                                list(shacl_traces),
                                shacl_tok,
                            )
                        all_violations = lint_graph_shacl(conn, auto_heal=True)
                elif all_violations:
                    all_violations = lint_graph_shacl(conn, auto_heal=True)

                # In-transaction Build-Time Gate: Enforce zero remaining SHACL violations
                remaining_violations = lint_graph_shacl(conn, auto_heal=False)
                if remaining_violations:
                    raise YawRuntimeError(
                        "CONSISTENCY_ERROR",
                        f"Graph build left {len(remaining_violations)} SHACL violations: {remaining_violations[:3]}",
                    )
                batch_shacl_remaining = sum(
                    1 for v in remaining_violations if (v.focus_node, v.shape) in batch_attempted_keys
                )
                self._graph_rlm_stats["shacl_violations_after"] = len(remaining_violations)
                self._graph_rlm_stats["shacl_violations_remaining"] = len(remaining_violations)
                self._graph_rlm_stats["shacl_conforms"] = len(remaining_violations) == 0
                for cid in chunk_ids:
                    set_meta(conn, f"indexed_fp:{cid}", json.dumps([ext_fp, openie_fp]))

            n_batch = max(1, len(chunk_ids))
            b_syn_tok = max(
                0,
                int(self._graph_rlm_stats.get("synonym_judge_tokens", 0))
                - int(pre_batch_stats.get("synonym_judge_tokens", 0)),
            )
            b_heal_tok = max(
                0,
                int(self._graph_rlm_stats.get("shacl_healer_tokens", 0))
                - int(pre_batch_stats.get("shacl_healer_tokens", 0)),
            )
            b_def_cnt = max(
                0,
                int(self._graph_rlm_stats.get("missing_entity_defs_count", 0))
                - int(pre_batch_stats.get("missing_entity_defs_count", 0)),
            )
            b_def_tok = max(
                0,
                int(self._graph_rlm_stats.get("missing_entity_defs_tokens", 0))
                - int(pre_batch_stats.get("missing_entity_defs_tokens", 0)),
            )
            for cid in chunk_ids:
                self._chunk_rlm_stats[cid] = {
                    "synonym_judge_tokens": b_syn_tok / n_batch,
                    "shacl_healer_tokens": b_heal_tok / n_batch,
                    "shacl_violations_initial": batch_shacl_before / n_batch,
                    "shacl_violations_remaining": batch_shacl_remaining / n_batch,
                    "missing_entity_defs_count": b_def_cnt / n_batch,
                    "missing_entity_defs_tokens": b_def_tok / n_batch,
                    "openie_tokens": float(chunk_openie_tok_by_cid.get(cid, 0)),
                }

            self._csr_graph = None
            self._query_retrieval_memo.clear()
            self._last_memo_question = None
            self._active_extraction_fingerprint = ext_fp
            self._active_openie_fingerprint = openie_fp
            self._opt_indexed_cids.update(chunk_ids)
            for cid in chunk_ids:
                self._chunk_indexed_ext_fp[cid] = ext_fp
            self._persist_dspy_trails_to_db(active_program)
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
                fact_kind_by_fid: dict[str, str] = {
                    str(r[0]): str(r[1])
                    for r in self.conn.execute(
                        f"SELECT fact_id, COALESCE(obj_kind, 'entity') FROM facts WHERE fact_id IN ({f_placeholders})",
                        kept_fids,
                    )
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
                    s_norm = str(text_processing(s))
                    s_ids = resolved_map.get(s_norm) or [compute_mdhash_id(s_norm, prefix="entity-")]
                    for s_id in s_ids:
                        seed_entities.setdefault(s_id, []).append(score)
                    if fact_kind_by_fid.get(fid, "entity") == "literal":
                        src_cids = fact_sources_by_fid.get(fid, ())
                        if src_cids:
                            share = score / float(len(src_cids))
                            for src_cid in src_cids:
                                literal_chunk_seeds.setdefault(src_cid, []).append(share)
                    else:
                        o_norm = str(text_processing(o))
                        o_ids = resolved_map.get(o_norm) or [compute_mdhash_id(o_norm, prefix="entity-")]
                        for o_id in o_ids:
                            seed_entities.setdefault(o_id, []).append(score)

                doc_freqs = get_entity_doc_frequencies(self.conn, list(seed_entities.keys()))

                entity_reset_weights: list[tuple[str, float]] = []
                for eid, scores in seed_entities.items():
                    mean_score = sum(scores) / len(scores)
                    df = max(1, doc_freqs.get(eid, 0))
                    w_entity = mean_score / float(df)
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
        val_slice = list(valset or trainset)
        val_pool: list[tuple[ChunkPointer, str]] = []
        seen_val_cids: set[str] = set()
        for ex in val_slice:
            ex_docs = getattr(ex, "item_docs", None)
            if ex_docs:
                for p, t in self._resolve_chunks_for_docs(ex_docs):
                    if p.chunk_id not in seen_val_cids:
                        seen_val_cids.add(p.chunk_id)
                        val_pool.append((p, t))
        if val_pool:
            self._opt_val_chunks_pool = val_pool
        elif len(self._opt_chunks_pool) <= 64 and self._opt_chunks_pool:
            self._opt_val_chunks_pool = list(self._opt_chunks_pool)
        else:
            self._opt_val_chunks_pool = list(self._opt_chunks_pool[:64])

        val_cids_set = {p.chunk_id for p, _ in self._opt_val_chunks_pool}
        train_cids_set: set[str] = set()
        for ex in trainset or ():
            ex_docs = getattr(ex, "item_docs", None)
            if ex_docs:
                for p, _ in self._resolve_chunks_for_docs(ex_docs):
                    train_cids_set.add(p.chunk_id)
        all_pool_cids = val_cids_set | train_cids_set | {p.chunk_id for p, _ in self._opt_chunks_pool}
        self._graph_rlm_stats["val_pool_size"] = len(val_cids_set)
        self._graph_rlm_stats["train_pool_size"] = len(train_cids_set)
        self._graph_rlm_stats["corpus_pool_size"] = max(1, len(all_pool_cids))
        if save_path is not None:
            # Publish the real split/pool composition immediately so monitors never guess it.
            with contextlib.suppress(OSError):
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

        missing_val_pool = [(p, t) for p, t in self._opt_val_chunks_pool if p.chunk_id not in self._opt_indexed_cids]
        if missing_val_pool:
            total_val_missing = len(missing_val_pool)
            for b_start in range(0, total_val_missing, 8):
                batch_slice = missing_val_pool[b_start : b_start + 8]
                if progress_callback is not None:
                    progress_callback(
                        f"indexing: validation passages "
                        f"({min(b_start + len(batch_slice), total_val_missing)}/{total_val_missing})"
                    )
                self.index_chunks(
                    batch_slice,
                    program=self.program,
                    progress_callback=progress_callback,
                )

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
                    max_total_chars=self.max_context_chars,
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

    def _rehydrate_sandbox_after_optimization(self, compiled: HippoRAGProgram) -> None:
        self._in_optimization = False
        self._query_retrieval_memo.clear()
        self._last_memo_question = None
        if self._opt_chunks_pool and (
            len(self._opt_indexed_cids) < len(self._opt_chunks_pool)
            or self._active_extraction_fingerprint != extraction_fingerprint(compiled)
        ):
            target_openie_fp = openie_fingerprint(compiled)
            clear_openie = self._active_openie_fingerprint != target_openie_fp
            self._reset_sandbox_graph(clear_openie=clear_openie)
            if len(self._opt_chunks_pool) <= 32:
                rehydrate_pool = list(self._opt_chunks_pool)
            else:
                val_cids = {p.chunk_id for p, _ in self._opt_val_chunks_pool}
                cached_pool = [
                    cp
                    for cp in self._opt_chunks_pool
                    if cp[0].chunk_id not in val_cids
                    and (
                        target_openie_fp,
                        cp[0].chunk_id,
                        hashlib.sha256(cp[1].encode()).hexdigest(),
                    )
                    in self._openie_fp_cache
                ]
                rehydrate_pool = list(self._opt_val_chunks_pool) + (
                    cached_pool[: max(0, 64 - len(self._opt_val_chunks_pool))]
                    if cached_pool
                    else [cp for cp in self._opt_chunks_pool if cp[0].chunk_id not in val_cids][:16]
                )
            self.index_chunks(rehydrate_pool, program=compiled)
        self._bind_program_hooks()

    def _build_optimization_hooks(self) -> GraphOptimizationHooks:
        codec = CandidatePayloadCodec(self._candidate_cache_sources)
        return GraphOptimizationHooks(
            ensure_candidate_indexed=lambda prog, cids=None: self._ensure_candidate_chunks_indexed(
                prog, item_docs=cids
            ),
            reset_sandbox_graph=lambda clear_openie: self._reset_sandbox_graph(clear_openie=clear_openie),
            index_chunks=lambda chunks, prog: self.index_chunks(chunks, program=prog),
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
                    "same_sense_cosine_gate",
                )
            },
            bind_program_hooks=self._bind_program_hooks,
            open_score_cache=lambda ctx: CandidateStageCache(
                self._openie_fp_cache.store,
                self.managed_conn,
                "simba_task_score",
                lambda: ctx,
            ),
            open_checkpoint_cache=lambda ctx: CandidateStageCache(
                self._openie_fp_cache.store,
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
