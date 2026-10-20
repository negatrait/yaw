"""Native DSPy HippoRAG 2 Graph and Retrieval Engine backed by yaw.db and disk line hydration."""

import hashlib
import json
import re
import sqlite3
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import dspy
import numpy as np

from yaw.core import ChunkPointer, IndexStats, StructuredTriple
from yaw.daemon.graph.inference import (
    FACT_INSTRUCTION,
    PASSAGE_INSTRUCTION,
    DSPyEmbeddingAdapter,
    HippoRAGProgram,
    ParetoProgramEnsemble,
    RLMInstructionProposer,
    YawRetriever,
    YawRM,
    compute_mdhash_id,
    compute_qa_em_f1,
    compute_sizecoding_pareto_score,
    configure_yaw_dspy_cache,
    text_processing,
)
from yaw.daemon.graph.ppr import (
    CSRGraph,
    build_transition_matrix,
    min_max_normalize,
    rank_passages_by_ppr,
    run_ppr,
)
from yaw.daemon.graph.storage import (
    ManagedConnection,
    delete_chunks,
    find_entity_disambiguation_candidates,
    get_chunk_pointers,
    get_entities_by_label_norm,
    get_entity_doc_frequencies,
    get_entity_vectors,
    get_existing_ids,
    get_facts_by_ids,
    get_graph_counts,
    get_max_end_line,
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
        max_context_chars: int = 24000,
        max_ast_expansion_hops: int = 1,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.db_path = db_path
        db_path_str = str(db_path)
        if db_path_str != ":memory:" and not db_path_str.startswith("file:"):
            self._dspy_cache_dir = Path(db_path).resolve().parent / "dspy_cache"
        else:
            self._dspy_cache_dir = self.workspace_root / "dspy_cache"
        configure_yaw_dspy_cache(self._dspy_cache_dir, restrict_pickle=True)
        self._lock = threading.RLock()
        self.managed_conn: ManagedConnection = open_managed_yaw_db(db_path)
        self.conn: sqlite3.Connection = self.managed_conn._conn
        self.program = program
        self.embedder = embedder
        self.config: Any = None

        self.max_context_chars = max_context_chars
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
            homograph_cosine_gate
            if homograph_cosine_gate is not None
            else same_sense_cosine_gate
        )
        self.homograph_cosine_gate = self.same_sense_cosine_gate

        self._csr_graph: CSRGraph | None = None
        self._active_repl_repair_keys: set[tuple[str, str, str, str]] | None = None
        self._active_repl_repairs_list: list[dict[str, str]] | None = None
        self._graph_rlm_stats: dict[str, Any] = self._empty_graph_rlm_stats()
        self._attempted_shacl_keys: set[tuple[str, str]] = set()
        self._chunk_rlm_stats: dict[str, dict[str, float]] = {}
        self._active_query_rlm_stats: dict[str, int] | None = None
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
        self._text_vec_cache: dict[tuple[str, str, str], np.ndarray] = {}
        self._openie_fp_cache: dict[
            tuple[str, str],
            tuple[Any, list[tuple[str, dict[str, Any], Any]], set[str], int, int, int],
        ] = {}
        self._synonym_fp_cache: dict[
            tuple[str, str],
            tuple[list[dict[str, str]], list[tuple[str, dict[str, Any], Any]], int],
        ] = {}
        self._shacl_fp_cache: dict[
            tuple[str, str],
            tuple[list[dict[str, str]], list[tuple[str, dict[str, Any], Any]], int],
        ] = {}

        self._bind_program_hooks()
        self._apply_program_embedding_prefixes()
        self._hydrate_indexed_cids_from_db()

    def _hydrate_indexed_cids_from_db(
        self,
        program: HippoRAGProgram | None = None,
    ) -> None:
        """Populate _opt_indexed_cids and _chunk_indexed_ext_fp for chunks already committed in SQLite."""
        active_prog = program if program is not None else self.program
        openie_fp = self._openie_fingerprint(active_prog)
        ext_fp = self._extraction_fingerprint(active_prog)
        if self._baseline_openie_fp is not None and openie_fp != self._baseline_openie_fp:
            return
        try:
            rows = self.conn.execute(
                "SELECT c.chunk_id FROM chunks c INNER JOIN openie_chunks o ON c.chunk_id = o.chunk_id"
            ).fetchall()
        except sqlite3.Error:
            return
        if not rows:
            return
        seeded_cids = [str(r[0]) for r in rows if r[0]]
        if not seeded_cids:
            return
        vec_cids = get_existing_ids(self.conn, "chunk_vec", seeded_cids)
        ready_cids = set(seeded_cids) & vec_cids
        if ready_cids:
            self._opt_indexed_cids.update(ready_cids)
            for cid in ready_cids:
                self._chunk_indexed_ext_fp.setdefault(cid, ext_fp)
            if self._active_extraction_fingerprint is None:
                self._active_extraction_fingerprint = ext_fp
            if self._active_openie_fingerprint is None:
                self._active_openie_fingerprint = openie_fp

    @staticmethod
    def _empty_graph_rlm_stats() -> dict[str, Any]:
        return {
            "homograph_splits": 0,
            "candidate_pairs": 0,
            "merged_entities": 0,
            "related_edges": 0,
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

    @staticmethod
    def _serialize_predictor_surface(pred: Any) -> dict[str, Any]:
        sig = getattr(pred, "signature", None)
        instructions = str(getattr(sig, "instructions", "") or "")
        in_fields = set(getattr(sig, "input_fields", {}) or {})
        out_fields = set(getattr(sig, "output_fields", {}) or {})
        demos_raw = getattr(pred, "demos", None) or ()
        demos_repr: list[str] = []
        for d in demos_raw:
            d_dict = (
                d.toDict()
                if hasattr(d, "toDict")
                else (dict(d) if isinstance(d, Mapping) else None)
            )
            if isinstance(d_dict, dict) and in_fields and out_fields:
                if not (
                    in_fields.issubset(d_dict.keys())
                    and out_fields.issubset(d_dict.keys())
                ):
                    continue
                filtered = {k: d_dict[k] for k in sorted(in_fields | out_fields)}
                demos_repr.append(
                    json.dumps(filtered, sort_keys=True, ensure_ascii=False, default=str)
                )
            else:
                demos_repr.append(str(d_dict) if d_dict is not None else str(d))
        return {"instructions": instructions, "demos": demos_repr}

    @staticmethod
    def _prune_mismatching_demos(program: HippoRAGProgram) -> None:
        if not hasattr(program, "named_predictors"):
            return
        for _, pred in program.named_predictors():
            demos_raw = getattr(pred, "demos", None)
            if not demos_raw:
                continue
            sig = getattr(pred, "signature", None)
            in_fields = set(getattr(sig, "input_fields", {}) or {})
            out_fields = set(getattr(sig, "output_fields", {}) or {})
            if not (in_fields and out_fields):
                continue
            valid_demos: list[Any] = []
            for d in demos_raw:
                d_dict = (
                    d.toDict()
                    if hasattr(d, "toDict")
                    else (dict(d) if isinstance(d, Mapping) else None)
                )
                if (
                    isinstance(d_dict, dict)
                    and in_fields.issubset(d_dict.keys())
                    and out_fields.issubset(d_dict.keys())
                ):
                    valid_demos.append(d)
            pred.demos = valid_demos

    @classmethod
    def _embed_fingerprint(cls, program: HippoRAGProgram) -> str:
        payload = {
            "embedding_query_prefix": getattr(program, "embedding_query_prefix", None),
            "embedding_fact_prefix": getattr(program, "embedding_fact_prefix", None),
            "embedding_document_prefix": getattr(program, "embedding_document_prefix", None),
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @classmethod
    def _synonym_fingerprint(cls, program: HippoRAGProgram) -> str:
        named = (
            dict(program.named_predictors())
            if hasattr(program, "named_predictors")
            else {}
        )
        payload = (
            cls._serialize_predictor_surface(named["synonym_judge"])
            if "synonym_judge" in named
            else {}
        )
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @classmethod
    def _shacl_fingerprint(cls, program: HippoRAGProgram) -> str:
        named = (
            dict(program.named_predictors())
            if hasattr(program, "named_predictors")
            else {}
        )
        shacl_keys = (
            "shacl_repair",
            "shacl_healer_rlm.generate_action",
            "shacl_healer_rlm.extract",
        )
        payload = {
            k: cls._serialize_predictor_surface(named[k])
            for k in shacl_keys
            if k in named
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @classmethod
    def _openie_fingerprint(cls, program: HippoRAGProgram) -> str:
        named = (
            dict(program.named_predictors())
            if hasattr(program, "named_predictors")
            else {}
        )
        payload = {
            "stages": {
                k: cls._serialize_predictor_surface(named[k])
                for k in ("ner", "triples", "entity_definer")
                if k in named
            },
            "gbnf_max_entities": getattr(program, "gbnf_max_entities", None),
            "gbnf_max_triples": getattr(program, "gbnf_max_triples", None),
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @classmethod
    def _extraction_fingerprint(cls, program: HippoRAGProgram) -> str:
        named = (
            dict(program.named_predictors())
            if hasattr(program, "named_predictors")
            else {}
        )
        graph_stage_keys = (
            "ner",
            "triples",
            "entity_definer",
            "synonym_judge",
            "shacl_repair",
            "shacl_healer_rlm.generate_action",
            "shacl_healer_rlm.extract",
        )
        payload = {
            "stages": {
                k: cls._serialize_predictor_surface(named[k])
                for k in graph_stage_keys
                if k in named
            },
            "gbnf_max_entities": getattr(program, "gbnf_max_entities", None),
            "gbnf_max_triples": getattr(program, "gbnf_max_triples", None),
            "embedding_query_prefix": getattr(program, "embedding_query_prefix", None),
            "embedding_fact_prefix": getattr(program, "embedding_fact_prefix", None),
            "embedding_document_prefix": getattr(program, "embedding_document_prefix", None),
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _extract_named_trace_entries(
        program: HippoRAGProgram,
        raw_trace: Sequence[Any],
        *,
        allowed_names: set[str] | None = None,
    ) -> list[tuple[str, dict[str, Any], Any]]:
        named_preds = (
            list(program.named_predictors())
            if hasattr(program, "named_predictors")
            else []
        )
        captured: list[tuple[str, dict[str, Any], Any]] = []
        for item in raw_trace:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue
            pred_inst, inp, out = item[0], item[1], item[2]
            matched_name: str | None = None
            pred_sig = getattr(pred_inst, "signature", None)
            for name, cand_pred in named_preds:
                if pred_inst is cand_pred:
                    matched_name = name
                    break
            if matched_name is None and pred_sig is not None:
                pred_fields = tuple(getattr(pred_sig, "fields", {}).keys())
                for name, cand_pred in named_preds:
                    cand_sig = getattr(cand_pred, "signature", None)
                    if cand_sig is not None:
                        is_eq = (
                            cand_sig.equals(pred_sig)
                            if hasattr(cand_sig, "equals")
                            else cand_sig == pred_sig
                        )
                        cand_fields = tuple(getattr(cand_sig, "fields", {}).keys())
                        if is_eq or (pred_fields and pred_fields == cand_fields):
                            matched_name = name
                            break
            if matched_name is not None:
                if allowed_names is not None and not (
                    matched_name in allowed_names
                    or any(matched_name.startswith(f"{a}.") for a in allowed_names)
                ):
                    continue
                inp_dict = dict(inp) if isinstance(inp, Mapping) else {"input": inp}
                captured.append((matched_name, inp_dict, out))
        return captured

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
                shacl_linter=lambda: [
                    v.to_dict() for v in lint_graph_shacl(self.conn, auto_heal=False)
                ],
            )
        if hasattr(target, "set_graph_search_tool"):
            target.set_graph_search_tool(self._search_graph_for_react)
        ensemble = getattr(target, "pareto_ensemble", None)
        if ensemble is not None and hasattr(ensemble, "programs"):
            for sub_prog in ensemble.programs:
                if sub_prog is not target:
                    sub_rm = getattr(sub_prog, "rm", None)
                    if sub_rm is not None:
                        sub_rm.default_k = self.retrieval_top_k
                    sub_ret = getattr(sub_prog, "retriever", None)
                    if sub_ret is not None:
                        sub_ret.k = self.retrieval_top_k
                    if hasattr(sub_prog, "set_retrieval_hook"):
                        sub_prog.set_retrieval_hook(self._canonical_retrieval_hook)
                    if hasattr(sub_prog, "set_graph_search_tool"):
                        sub_prog.set_graph_search_tool(self._search_graph_for_react)
        if hasattr(self.embedder, "bind_telemetry_hook"):
            self.embedder.bind_telemetry_hook(self._on_embedding_telemetry)

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
        target_cids = self._ensure_candidate_chunks_indexed(
            active_program, item_docs=item_docs
        )
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

        qa_top_k = max(1, int(k if k is not None else self.retrieval_top_k))
        fetch_top_k = max(20, qa_top_k)
        hits = self.retrieve(
            question,
            top_k=fetch_top_k,
            program=active_program,
        )
        if not hits:
            return "", (), (), ()
        texts = tuple(self._format_hydrated_passage(p, text) for p, _, text in hits)
        scores = tuple(float(s) for _, s, _ in hits)
        pids = tuple(p.chunk_id for p, _, _ in hits)
        formatted_context = self.format_qa_context(hits[:qa_top_k])
        return formatted_context, texts, scores, pids

    def as_rm(
        self,
        k: int | None = None,
        *,
        configure_dspy: bool = False,
    ) -> YawRM:
        """Return a first-class `YawRM` callable bound to this engine (optionally registering `dspy.configure(rm=...)`)."""
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
        """Return a first-class `YawRetriever(dspy.Retrieve)` bound to this engine (optionally registering `dspy.configure(rm=...)`)."""
        rm = self.as_rm(k=k, configure_dspy=configure_dspy)
        return YawRetriever(k=rm.default_k, rm=rm)

    def _search_graph_for_react(self, query: str) -> str:
        """Execute hybrid dense + FTS fact and entity lookup in `yaw.db` for `dspy.ReActV2` multi-hop exploration."""
        clean_q = str(query or "").strip()
        if not clean_q:
            return ""
        with self._lock:
            query_fact_vec = self._encode_texts(
                [clean_q], instruction=FACT_INSTRUCTION, stage="react_fact_embed"
            )[0]
            dense_fact_hits = search_fact_vectors_knn(
                self.conn, query_fact_vec, top_k=min(10, self.fact_candidate_pool_k)
            )
            fts_fact_hits = search_facts_fts(
                self.conn, clean_q, top_k=min(10, self.fact_fts_pool_k)
            )
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
            passage_lines = [
                f"[{ptr.chunk_id}] {self._format_hydrated_passage(ptr, txt)}"
                for ptr, _, txt in hydrated
            ]

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
                setattr(target, "embedding_query_prefix", getattr(profile, "query_prefix", None))
            if getattr(target, "embedding_fact_prefix", None) is None:
                setattr(target, "embedding_fact_prefix", getattr(profile, "fact_query_prefix", None))
            if getattr(target, "embedding_document_prefix", None) is None:
                setattr(target, "embedding_document_prefix", getattr(profile, "document_prefix", None))

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
                chunk_ids, entity_ids, edges = load_graph_topology(self.conn)
                self._csr_graph = build_transition_matrix(
                    chunk_ids, entity_ids, edges, beta=self.beta
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
            embed_fp = self._embed_fingerprint(self.program)
            missing_texts: list[str] = []
            seen_missing: set[str] = set()
            for t in texts:
                ts = str(t)
                if (embed_fp, instruction, ts) not in self._text_vec_cache and ts not in seen_missing:
                    seen_missing.add(ts)
                    missing_texts.append(ts)
            if missing_texts:
                encoded = self.embedder.encode(
                    missing_texts, instruction=instruction, stage=stage
                )
                for ts, vec in zip(missing_texts, encoded, strict=False):
                    self._text_vec_cache[(embed_fp, instruction, ts)] = np.asarray(
                        vec, dtype=np.float32
                    )
            return [
                self._text_vec_cache[(embed_fp, instruction, str(t))] for t in texts
            ]
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
            else (
                max_context_chars
                if max_context_chars is not None
                else self.max_context_chars
            )
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

    @staticmethod
    def _format_entity_embedding_text(name: str, definition: str) -> str:
        clean_def = definition.strip()
        if clean_def:
            return f"# {name}\n{clean_def}"
        return name

    @staticmethod
    def _format_fact_embedding_text(
        s: str,
        p: str,
        o: str,
        spo_def: str = "",
        subj_def: str = "",
        obj_def: str = "",
    ) -> str:
        s_d = spo_def.strip()
        if s_d and s_d.lower().rstrip(".") != f"{s} {p} {o}".lower().rstrip("."):
            return f"{s} {p} {o}. {s_d}"
        return f"{s} {p} {o}"

    @staticmethod
    def _apply_shacl_repairs(
        conn: sqlite3.Connection,
        repairs: Sequence[dict[str, str]],
    ) -> int:
        applied = 0
        for rep in repairs:
            focus = str(rep.get("focus_node", "")).strip()
            action = str(rep.get("action", "")).strip()
            val = str(rep.get("value", "")).strip()
            target = str(rep.get("target_node", "")).strip() or val
            if not focus or not action:
                continue
            if action in ("set_definition", "rewrite_definition") and val:
                if focus.startswith("entity-"):
                    cur = conn.execute(
                        "UPDATE entities SET definition = ? WHERE entity_id = ?",
                        (val, focus),
                    )
                    if cur.rowcount > 0:
                        applied += 1
                elif focus.startswith("fact-"):
                    cur = conn.execute(
                        "UPDATE facts SET definition = ? WHERE fact_id = ?",
                        (val, focus),
                    )
                    if cur.rowcount > 0:
                        applied += 1
            elif action == "set_pref_label" and val and focus.startswith("entity-"):
                cur = conn.execute(
                    "UPDATE entities SET name = ? WHERE entity_id = ?",
                    (val, focus),
                )
                if cur.rowcount > 0:
                    upsert_entity_labels(
                        conn, [(val.strip().lower(), focus, "skos:prefLabel", val, "__healed__")]
                    )
                    applied += 1
            elif action in ("demote_to_alt_label", "demote_label") and focus.startswith("entity-"):
                rows = conn.execute(
                    "SELECT label_norm FROM entity_labels WHERE entity_id = ? AND label_kind = 'skos:prefLabel' ORDER BY label_norm ASC",
                    (focus,),
                ).fetchall()
                if len(rows) > 1:
                    for extra in rows[1:]:
                        conn.execute(
                            "UPDATE entity_labels SET label_kind = 'skos:altLabel' WHERE entity_id = ? AND label_norm = ? AND label_kind = 'skos:prefLabel'",
                            (focus, str(extra[0])),
                        )
                    applied += 1
                else:
                    s13_rows = conn.execute(
                        """
                        SELECT label_norm
                        FROM entity_labels
                        WHERE entity_id = ?
                        GROUP BY label_norm
                        HAVING COUNT(DISTINCT label_kind) > 1
                        """,
                        (focus,),
                    ).fetchall()
                    for r in s13_rows:
                        conn.execute(
                            "UPDATE entity_labels SET label_kind = 'skos:altLabel' WHERE entity_id = ? AND label_norm = ?",
                            (focus, str(r[0])),
                        )
                        applied += 1
            elif action in ("merge_altLabel", "merge_hiddenLabel") and (target or "|" in focus):
                if "|" in focus:
                    ea, eb = (x.strip() for x in focus.split("|", 1))
                    if target == ea:
                        src_node, dst_node = eb, ea
                    else:
                        src_node, dst_node = ea, eb
                else:
                    src_node, dst_node = focus, target
                lkind = (
                    "skos:hiddenLabel"
                    if action == "merge_hiddenLabel"
                    else "skos:altLabel"
                )
                if src_node and dst_node and src_node != dst_node:
                    existing_pair = get_existing_ids(conn, "entity", [src_node, dst_node])
                    if src_node in existing_pair and dst_node in existing_pair:
                        merge_entities(conn, src_node, dst_node, label_kind=lkind)
                        applied += 1
            elif action in ("add_different_from", "assert_differentFrom"):
                if "|" in focus:
                    ea, eb = (x.strip() for x in focus.split("|", 1))
                else:
                    ea, eb = focus, target
                if ea and eb and ea != eb:
                    existing_pair = get_existing_ids(conn, "entity", [ea, eb])
                    if ea in existing_pair and eb in existing_pair:
                        upsert_semantic_edges(conn, [(ea, eb, "owl:differentFrom", 0.0)])
                        applied += 1
            elif action == "assert_related":
                if "|" in focus:
                    ea, eb = (x.strip() for x in focus.split("|", 1))
                else:
                    ea, eb = focus, target
                if ea and eb and ea != eb:
                    existing_pair = get_existing_ids(conn, "entity", [ea, eb])
                    if ea in existing_pair and eb in existing_pair:
                        upsert_semantic_edges(conn, [(ea, eb, "skos:related", 0.5)])
                        applied += 1
            elif action in ("delete_dangling", "delete_orphan", "delete_invalid_fact"):
                if "->" in focus:
                    src_node, dst_node = (x.strip() for x in focus.split("->", 1))
                    conn.execute(
                        "DELETE FROM edges WHERE src_id = ? AND dst_id = ?",
                        (src_node, dst_node),
                    )
                    conn.execute(
                        "DELETE FROM edge_sources WHERE src_id = ? AND dst_id = ?",
                        (src_node, dst_node),
                    )
                    applied += 1
                else:
                    affected_fids = [
                        str(r[0])
                        for r in conn.execute(
                            "SELECT fact_id FROM facts WHERE fact_id = ? OR subj_id = ? OR obj_id = ?",
                            (focus, focus, focus),
                        ).fetchall()
                    ]
                    for fid in affected_fids:
                        conn.execute("DELETE FROM fact_vec WHERE fact_id = ?", (fid,))
                        conn.execute("DELETE FROM fact_fts WHERE fact_id = ?", (fid,))
                        conn.execute("DELETE FROM facts WHERE fact_id = ?", (fid,))
                    conn.execute(
                        "DELETE FROM edges WHERE src_id = ? OR dst_id = ?",
                        (focus, focus),
                    )
                    conn.execute(
                        "DELETE FROM edge_sources WHERE src_id = ? OR dst_id = ?",
                        (focus, focus),
                    )
                    conn.execute("DELETE FROM entity_labels WHERE entity_id = ?", (focus,))
                    conn.execute("DELETE FROM entity_vec WHERE entity_id = ?", (focus,))
                    conn.execute("DELETE FROM entities WHERE entity_id = ?", (focus,))
                    applied += 1
        return applied

    def _reset_sandbox_graph(self, *, clear_openie: bool = True) -> None:
        """Clear graph tables in the sandbox DB when a candidate mutates extraction/graph prompts or query scope."""
        with self._lock, self.managed_conn.transaction() as conn:
            conn.execute("DELETE FROM chunk_vec")
            conn.execute("DELETE FROM chunk_fts")
            conn.execute("DELETE FROM chunks")
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
                self._captured_chunk_traces.clear()
        self._captured_graph_traces.clear()
        self._opt_indexed_cids.clear()
        self._chunk_indexed_ext_fp.clear()
        self._attempted_shacl_keys.clear()
        self._chunk_rlm_stats.clear()
        self._active_query_rlm_stats = None
        self._csr_graph = None
        self._graph_rlm_stats = self._empty_graph_rlm_stats()

    def _materialize_raw_doc_to_pool(self, doc: str) -> tuple[ChunkPointer, str] | None:
        """Materialize a raw document string into corpus/passages.txt and _opt_chunks_pool without running LLM indexing."""
        lines = doc.splitlines(keepends=True)
        n_lines = len(lines)
        if n_lines == 0:
            return None
        corpus_dir = self.workspace_root / "corpus"
        corpus_dir.mkdir(parents=True, exist_ok=True)
        passages_path = corpus_dir / "passages.txt"
        pool_max_line = max(
            (
                p.end_line
                for p, _ in self._opt_chunks_pool
                if p.rel_posix_path == "corpus/passages.txt"
            ),
            default=0,
        )
        current_lines = max(
            get_max_end_line(self.conn, "corpus/passages.txt"), pool_max_line
        )
        open_mode = "w" if current_lines == 0 else "a"
        with passages_path.open(open_mode, encoding="utf-8") as f:
            start_line = current_lines + 1
            end_line = current_lines + n_lines
            for line in lines:
                f.write(line if line.endswith("\n") else line + "\n")
        cid = compute_mdhash_id(doc, prefix="chunk-")
        first_line = doc.split("\n", 1)[0].strip()
        title = first_line if len(first_line) < 100 else ""
        ptr = ChunkPointer(
            chunk_id=cid,
            rel_posix_path="corpus/passages.txt",
            start_line=start_line,
            end_line=end_line,
            title=title,
            ast_island_start=start_line,
            ast_island_end=end_line,
        )
        if cid not in {p.chunk_id for p, _ in self._opt_chunks_pool}:
            self._opt_chunks_pool.append((ptr, doc))
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
                    title_match = bool(
                        doc_title and ptr.title and ptr.title.strip() == doc_title
                    )
                    body_match = (
                        c_strip in (d, doc_body)
                        or (len(c_strip) >= 30 and c_strip in d)
                        or (title_match and (c_strip in doc_body or doc_body in c_strip))
                    )
                    if body_match:
                        doc_matches.append((ptr, c_text))
                if not doc_matches:
                    mat = self._materialize_raw_doc_to_pool(d)
                    if mat is not None:
                        pool_by_cid[mat[0].chunk_id] = mat
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
            ext_fp = self._extraction_fingerprint(program)
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
            missing_pool = [
                (p, t)
                for p, t in pool_chunks
                if p.chunk_id not in self._opt_indexed_cids
            ]
            if missing_pool:
                self.index_chunks(missing_pool, program=program)

            target_list = req_chunks if req_chunks else pool_chunks
            stale_targets = [
                (p, t)
                for p, t in target_list
                if p.chunk_id in self._opt_indexed_cids
                and self._chunk_indexed_ext_fp.get(p.chunk_id) != ext_fp
            ]
            if stale_targets:
                stale_cids = [p.chunk_id for p, _ in stale_targets]
                self.delete_chunks(stale_cids)
                for cid in stale_cids:
                    self._opt_indexed_cids.discard(cid)
                    self._chunk_indexed_ext_fp.pop(cid, None)

            missing_targets = [
                (p, t)
                for p, t in target_list
                if p.chunk_id not in self._opt_indexed_cids
            ]
            if missing_targets:
                self.index_chunks(missing_targets, program=program)
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
                        agg["cached_openie_tokens"] += float(
                            c_stats.get("openie_tokens", 0.0)
                        )
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
                if cid in existing_chunks
                and (cid not in existing_vecs or cid not in existing_openie)
            ]
            new_docs = [
                d
                for d, cid in zip(unique_docs, chunk_ids, strict=False)
                if cid not in existing_chunks
            ]
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
                corpus_dir = self.workspace_root / "corpus"
                corpus_dir.mkdir(parents=True, exist_ok=True)
                passages_path = corpus_dir / "passages.txt"

                pool_max_line = max(
                    (
                        p.end_line
                        for p, _ in self._opt_chunks_pool
                        if p.rel_posix_path == "corpus/passages.txt"
                    ),
                    default=0,
                )
                current_lines = max(
                    get_max_end_line(self.conn, "corpus/passages.txt"), pool_max_line
                )
                open_mode = "w" if current_lines == 0 else "a"

                with passages_path.open(open_mode, encoding="utf-8") as f:
                    for doc in new_docs:
                        cid = compute_mdhash_id(doc, prefix="chunk-")
                        first_line = doc.split("\n", 1)[0].strip()
                        title = first_line if len(first_line) < 100 else ""
                        lines = doc.splitlines(keepends=True)
                        n_lines = len(lines)
                        if n_lines == 0:
                            continue
                        start_line = current_lines + 1
                        end_line = current_lines + n_lines
                        for line in lines:
                            f.write(line if line.endswith("\n") else line + "\n")
                        current_lines += n_lines

                        ptr = ChunkPointer(
                            chunk_id=cid,
                            rel_posix_path="corpus/passages.txt",
                            start_line=start_line,
                            end_line=end_line,
                            title=title,
                            ast_island_start=start_line,
                            ast_island_end=end_line,
                        )
                        chunks_to_index.append((ptr, doc))

            return self.index_chunks(chunks_to_index, progress_callback=progress_callback)

    def _format_chunk_for_embedding(self, ptr: ChunkPointer, text: str) -> str:
        """Ensure AST-resolved ChunkPointer.title is passed as a Title: line for title-aware embedders."""
        parse_title = bool(
            getattr(getattr(self.embedder, "profile", None), "parse_document_title", False)
        )
        if not parse_title or not ptr.title or not ptr.title.strip():
            return text
        clean_title = ptr.title.strip()
        first_line = text.split("\n", 1)[0].strip() if "\n" in text else text.strip()
        if first_line.startswith("#") or first_line.lower().startswith(("title:", "wikipedia title:")):
            return text
        if text.startswith(f"{clean_title}\n"):
            body = text[len(clean_title) + 1 :]
            return f"Title: {clean_title}\n{body}" if body.strip() else text
        return f"Title: {clean_title}\n{text}"

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
            from .inference.chat import set_yaw_sleep_seconds

            set_yaw_sleep_seconds(opt_sleep)

        active_program = program if program is not None else self.program
        with self._lock:
            self._bind_program_hooks(active_program)
            self._apply_program_embedding_prefixes(active_program)
            openie_fp = self._openie_fingerprint(active_program)
            ext_fp = self._extraction_fingerprint(active_program)

            existing_pool_cids = {p.chunk_id for p, _ in self._opt_chunks_pool}
            for ptr, text in chunks:
                if ptr.chunk_id not in existing_pool_cids:
                    self._opt_chunks_pool.append((ptr, text))
                    existing_pool_cids.add(ptr.chunk_id)

            chunk_ptrs = [ptr for ptr, _ in chunks]
            chunk_ids = [ptr.chunk_id for ptr in chunk_ptrs]
            self._captured_graph_traces.clear()
            pre_batch_stats = dict(self._graph_rlm_stats)

            # 1. Chunk vector embeddings in bounded batches
            existing_chunk_vecs = get_existing_ids(self.conn, "chunk_vec", chunk_ids)
            missing_chunks = [(p, t) for p, t in chunks if p.chunk_id not in existing_chunk_vecs]
            chunk_vec_pairs: list[tuple[str, np.ndarray]] = []
            if missing_chunks:
                if progress_callback is not None:
                    progress_callback(f"indexing: embedding {len(missing_chunks)} chunk vectors")
                texts_to_embed = [self._format_chunk_for_embedding(p, t) for p, t in missing_chunks]
                vecs = self._encode_in_batches(
                    texts_to_embed, instruction="", stage="chunk_embed"
                )
                chunk_vec_pairs = [
                    (p.chunk_id, v) for (p, _), v in zip(missing_chunks, vecs, strict=False)
                ]

            # 2. Crash-resilient OpenIE extraction (persisted per-chunk immediately)
            cached_openie = get_openie_checkpoints(self.conn, chunk_ids)
            all_records: list[Any] = []
            attempted_missing_by_cid: dict[str, set[str]] = {}
            chunk_openie_tok_by_cid: dict[str, int] = {}
            total_chunks = len(chunks)
            for idx, (ptr, text) in enumerate(chunks, 1):
                if (openie_fp, ptr.chunk_id) in self._openie_fp_cache:
                    rec, cached_tr, cached_norms, m_cnt, m_tok, o_tok = self._openie_fp_cache[
                        (openie_fp, ptr.chunk_id)
                    ]
                    self._captured_chunk_traces[ptr.chunk_id] = list(cached_tr)
                    attempted_missing_by_cid[ptr.chunk_id] = set(cached_norms)
                    chunk_openie_tok_by_cid[ptr.chunk_id] = o_tok
                    self._graph_rlm_stats["missing_entity_defs_count"] += m_cnt
                    self._graph_rlm_stats["missing_entity_defs_tokens"] += m_tok
                    save_openie_checkpoints(self.conn, [rec])
                    all_records.append(rec)
                elif (
                    self._baseline_openie_fp is None
                    or openie_fp == self._baseline_openie_fp
                ) and ptr.chunk_id in cached_openie:
                    rec = cached_openie[ptr.chunk_id]
                    if getattr(rec, "entity_definitions", None):
                        cached_norms: set[str] = set()
                        for raw_ent in getattr(rec, "entities", ()) or ():
                            e_s = (
                                str(raw_ent[0])
                                if isinstance(raw_ent, (list, tuple)) and raw_ent
                                else str(raw_ent)
                            )
                            en = str(text_processing(e_s))
                            if en:
                                cached_norms.add(en)
                        for st_item in getattr(rec, "structured_triples", ()) or ():
                            for side in (getattr(st_item, "subj", ""), getattr(st_item, "obj", "")):
                                sn = str(text_processing(str(side)))
                                if sn:
                                    cached_norms.add(sn)
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
                    prev_def_audit_len = len(
                        getattr(active_program, "entity_def_audit_trail", ())
                    )
                    prev_telem_len = len(getattr(active_program, "telemetry", ()))
                    chunk_raw_trace: list[Any] = []
                    with dspy.context(trace=chunk_raw_trace, usage_tracker=None):
                        rec = active_program.extract_openie(ptr.chunk_id, text)
                    chunk_traces = self._extract_named_trace_entries(
                        active_program, chunk_raw_trace
                    )
                    self._captured_chunk_traces[ptr.chunk_id] = chunk_traces
                    new_telem_entries = list(
                        getattr(active_program, "telemetry", ())[prev_telem_len:]
                    )
                    raw_openie_tok = sum(
                        int(t_entry.get("tokens_used", 0))
                        for t_entry in new_telem_entries
                        if isinstance(t_entry, dict)
                    )
                    new_def_entries = list(
                        getattr(active_program, "entity_def_audit_trail", ())[
                            prev_def_audit_len:
                        ]
                    )
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
                    self._openie_fp_cache[(openie_fp, ptr.chunk_id)] = (
                        rec,
                        list(chunk_traces),
                        set(attempted_norms),
                        chunk_m_cnt,
                        chunk_m_tok,
                        chunk_openie_tok,
                    )
                    save_openie_checkpoints(self.conn, [rec])
                    all_records.append(rec)

            # 3. Resolve Stage 1 NER Entities, Stage 2 StructuredTriples, Provisional Sense-Split Gate, and Contextual Embeddings
            chunk_text_by_id = {ptr.chunk_id: text for ptr, text in chunks}
            record_structured: list[
                tuple[str, list[StructuredTriple], list[tuple[str, str]]]
            ] = []
            all_norm_labels: set[str] = set()
            for rec in all_records:
                cid = rec.chunk_id
                stage1_defs: dict[str, str] = {}
                for raw_pair in getattr(rec, "entity_definitions", ()) or ():
                    if isinstance(raw_pair, (list, tuple)) and len(raw_pair) >= 2:
                        enorm = str(text_processing(str(raw_pair[0])))
                        edef = str(raw_pair[1]).strip()
                        if enorm and edef and (
                            enorm not in stage1_defs
                            or len(edef) > len(stage1_defs[enorm])
                        ):
                            stage1_defs[enorm] = edef

                ner_entities: list[tuple[str, str]] = []
                seen_ner_norms: set[str] = set()
                for raw_ent in getattr(rec, "entities", ()) or ():
                    if isinstance(raw_ent, (list, tuple)) and len(raw_ent) >= 1:
                        e_str = str(raw_ent[0])
                        e_inline_def = (
                            str(raw_ent[1]).strip()
                            if len(raw_ent) >= 2 and raw_ent[1] is not None
                            else ""
                        )
                    else:
                        e_str = str(raw_ent)
                        e_inline_def = ""
                    e_norm = str(text_processing(e_str))
                    if not e_norm or e_norm in seen_ner_norms:
                        continue
                    seen_ner_norms.add(e_norm)
                    if e_inline_def and (
                        e_norm not in stage1_defs
                        or len(e_inline_def) > len(stage1_defs[e_norm])
                    ):
                        stage1_defs[e_norm] = e_inline_def
                    e_def = e_inline_def or stage1_defs.get(e_norm, "")
                    all_norm_labels.add(e_norm)
                    ner_entities.append((e_norm, e_def))

                for enorm, edef in stage1_defs.items():
                    if enorm not in seen_ner_norms:
                        seen_ner_norms.add(enorm)
                        all_norm_labels.add(enorm)
                        ner_entities.append((enorm, edef))

                st_list: list[StructuredTriple] = list(getattr(rec, "structured_triples", ()) or ())
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
                            st_list.append(
                                StructuredTriple(
                                    subj=s_raw,
                                    subj_def=stage1_defs.get(s_raw_norm, ""),
                                    pred=p_raw,
                                    obj=o_raw,
                                    obj_def=stage1_defs.get(o_raw_norm, ""),
                                    spo_def=spo_synth,
                                )
                            )
                normalized_sts: list[StructuredTriple] = []
                for st in st_list:
                    s_norm = str(text_processing(st.subj))
                    p_norm = str(text_processing(st.pred))
                    o_norm = str(text_processing(st.obj))
                    if not s_norm or not p_norm or not o_norm:
                        continue
                    all_norm_labels.add(s_norm)
                    all_norm_labels.add(o_norm)
                    spo_d = st.spo_def.strip() or f"{s_norm} {p_norm} {o_norm}."
                    raw_subj_d = st.subj_def.strip()
                    subj_d = raw_subj_d or stage1_defs.get(s_norm, "")
                    if subj_d and (
                        s_norm not in stage1_defs
                        or len(subj_d) > len(stage1_defs[s_norm])
                    ):
                        stage1_defs[s_norm] = subj_d

                    raw_obj_d = st.obj_def.strip()
                    obj_d = raw_obj_d or stage1_defs.get(o_norm, "")
                    if obj_d and (
                        o_norm not in stage1_defs
                        or len(obj_d) > len(stage1_defs[o_norm])
                    ):
                        stage1_defs[o_norm] = obj_d

                    normalized_sts.append(
                        StructuredTriple(
                            subj=s_norm,
                            pred=p_norm,
                            obj=o_norm,
                            spo_def=spo_d,
                            subj_def=subj_d,
                            obj_def=obj_d,
                        )
                    )

                already_attempted = attempted_missing_by_cid.get(cid, set())
                missing_norms: list[str] = []
                seen_missing: set[str] = set(already_attempted)
                for st in normalized_sts:
                    if not st.subj_def and not stage1_defs.get(st.subj) and st.subj not in seen_missing:
                        seen_missing.add(st.subj)
                        missing_norms.append(st.subj)
                    if not st.obj_def and not stage1_defs.get(st.obj) and st.obj not in seen_missing:
                        seen_missing.add(st.obj)
                        missing_norms.append(st.obj)
                for e_norm, e_def in ner_entities:
                    if not e_def and not stage1_defs.get(e_norm) and e_norm not in seen_missing:
                        seen_missing.add(e_norm)
                        missing_norms.append(e_norm)

                if missing_norms and hasattr(active_program, "define_missing_entities"):
                    prev_def_audit_len = len(
                        getattr(active_program, "entity_def_audit_trail", ())
                    )
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
                        self._extract_named_trace_entries(active_program, def_raw_trace)
                    )
                    new_def_entries = list(
                        getattr(active_program, "entity_def_audit_trail", ())[
                            prev_def_audit_len:
                        ]
                    )
                    for entry in new_def_entries:
                        if isinstance(entry, dict):
                            self._graph_rlm_stats["missing_entity_defs_count"] += int(
                                entry.get("missing_count", 0)
                            )
                            self._graph_rlm_stats["missing_entity_defs_tokens"] += int(
                                entry.get("tokens", 0)
                            )
                    for r_name, r_def in repaired_pairs:
                        r_norm = str(text_processing(str(r_name)))
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
                            obj_def=st.obj_def or stage1_defs.get(st.obj, ""),
                        )
                        for st in normalized_sts
                    ]
                    ner_entities = [
                        (e_norm, e_def or stage1_defs.get(e_norm, ""))
                        for e_norm, e_def in ner_entities
                    ]

                record_structured.append((cid, normalized_sts, ner_entities))

            existing_by_label = get_entities_by_label_norm(self.conn, list(all_norm_labels))
            all_existing_eids = list({eid for eids in existing_by_label.values() for eid in eids})
            entity_info_by_id: dict[str, tuple[str, str]] = {}
            existing_vecs_by_id: dict[str, np.ndarray] = {}
            if all_existing_eids:
                for i in range(0, len(all_existing_eids), 900):
                    batch_eids = all_existing_eids[i : i + 900]
                    placeholders = ",".join("?" for _ in batch_eids)
                    for row in self.conn.execute(
                        f"SELECT entity_id, name, COALESCE(definition, '') FROM entities WHERE entity_id IN ({placeholders})",
                        batch_eids,
                    ).fetchall():
                        entity_info_by_id[str(row[0])] = (str(row[1]), str(row[2]))
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
                    if 3 <= len(lbl) <= 30
                    and not lbl.isdigit()
                    and len(lbl.split()) <= 3
                }
                for ptr, _ in chunks:
                    if ptr.title:
                        t_norm = str(text_processing(ptr.title))
                        if 3 <= len(t_norm) <= 30 and not t_norm.isdigit():
                            known_concepts.add(t_norm)
                for _cid, normalized_sts, ner_entities in record_structured:
                    existing_pairs = {(st.subj, st.obj) for st in normalized_sts} | {
                        (st.obj, st.subj) for st in normalized_sts
                    }
                    for e_norm, e_def in ner_entities[:2]:
                        if not e_def or len(e_norm) < 3:
                            continue
                        first_clause = e_def.split(".", 1)[0].lower()
                        clause_words = set(re.findall(r"[a-z0-9]+", first_clause))
                        for c_norm in known_concepts:
                            if c_norm == e_norm or c_norm in e_norm or e_norm in c_norm:
                                continue
                            c_tokens = c_norm.split()
                            if not all(tok in clause_words for tok in c_tokens):
                                continue
                            if (e_norm, c_norm) in existing_pairs:
                                continue
                            if re.search(rf"\b{re.escape(c_norm)}\b", first_clause):
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
                        entity_info_by_id.get(eid, (lbl, ""))[0],
                        entity_info_by_id.get(eid, (lbl, ""))[1],
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
                        sense_cand_id = compute_mdhash_id(
                            f"{m_lbl}::{c_def.lower()}", prefix="entity-"
                        )
                        if sense_cand_id in existing_vecs_by_id:
                            continue
                        ct = self._format_entity_embedding_text(m_lbl, c_def)
                        if ct not in seen_prebatch:
                            seen_prebatch.add(ct)
                            prebatch_texts.append(ct)
                    for eid_s, name_s, prev_def_s, prev_vec_s in senses_by_label.get(lbl, ()):
                        if prev_vec_s is None and prev_def_s:
                            pt = self._format_entity_embedding_text(name_s, prev_def_s)
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
                    if lbl_norm == str(text_processing(entity_name))
                    else "skos:altLabel"
                )

            def _resolve_mention_entity_id(
                label_norm: str,
                defn: str,
                chunk_id: str,
            ) -> str:
                clean_def = defn.strip()
                existing_senses = senses_by_label.get(label_norm, [])
                if not existing_senses:
                    eid = compute_mdhash_id(label_norm, prefix="entity-")
                    senses_by_label[label_norm] = [
                        (eid, label_norm, clean_def, existing_vecs_by_id.get(eid))
                    ]
                    entity_map[eid] = (eid, label_norm, clean_def)
                    entity_labels_to_upsert.append(
                        (label_norm, eid, "skos:prefLabel", label_norm, chunk_id)
                    )
                    return eid

                if not clean_def:
                    eid, name, prev_def, _ = existing_senses[0]
                    if eid not in entity_map:
                        entity_map[eid] = (eid, name, prev_def)
                    entity_labels_to_upsert.append(
                        (label_norm, eid, _label_kind_for(name, label_norm), label_norm, chunk_id)
                    )
                    return eid

                # Check if any existing sense has no definition yet or identical definition
                for idx_s, (eid, name, prev_def, prev_vec) in enumerate(existing_senses):
                    if not prev_def or prev_def.lower() == clean_def.lower():
                        merged_def = (
                            clean_def
                            if len(clean_def) > len(prev_def)
                            else (prev_def or clean_def)
                        )
                        existing_senses[idx_s] = (
                            eid,
                            name,
                            merged_def,
                            prev_vec if merged_def == prev_def else None,
                        )
                        entity_map[eid] = (eid, name, merged_def)
                        entity_labels_to_upsert.append(
                            (label_norm, eid, _label_kind_for(name, label_norm), label_norm, chunk_id)
                        )
                        return eid

                # Both have distinct non-empty definitions:
                # Reuse existing concept only when high-confidence same sense (cos >= 0.85).
                # When cos < 0.85 (whether in [0.65, 0.85) or < 0.65), mint a provisional sense_id
                # WITHOUT writing any owl:differentFrom edge in Step 4 so Step 6 judge_synonyms
                # adjudicates every ambiguous or divergent same-label pair.
                sense_id = compute_mdhash_id(
                    f"{label_norm}::{clean_def.lower()}", prefix="entity-"
                )
                cached_sense_vec = existing_vecs_by_id.get(sense_id) or precomputed_entity_vecs.get(
                    sense_id
                )
                if cached_sense_vec is not None:
                    cand_vec = cached_sense_vec
                else:
                    cand_text = self._format_entity_embedding_text(label_norm, clean_def)
                    cand_vec = self._encode_in_batches(
                        [cand_text], instruction="", stage="entity_embed"
                    )[0]
                    c_norm = float(np.linalg.norm(cand_vec))
                    if c_norm > 0:
                        cand_vec = cand_vec / c_norm

                for idx_s, (eid, name, prev_def, prev_vec) in enumerate(existing_senses):
                    if prev_vec is None:
                        p_text = self._format_entity_embedding_text(name, prev_def)
                        prev_vec = self._encode_in_batches(
                            [p_text], instruction="", stage="entity_embed"
                        )[0]
                        p_norm = float(np.linalg.norm(prev_vec))
                        if p_norm > 0:
                            prev_vec = prev_vec / p_norm
                        existing_senses[idx_s] = (eid, name, prev_def, prev_vec)
                        precomputed_entity_vecs[eid] = prev_vec
                    cos_sim = float(np.dot(cand_vec, prev_vec))
                    if cos_sim >= self.same_sense_cosine_gate:
                        merged_def = (
                            clean_def if len(clean_def) > len(prev_def) else prev_def
                        )
                        if merged_def != prev_def:
                            existing_senses[idx_s] = (eid, name, merged_def, cand_vec)
                            precomputed_entity_vecs[eid] = cand_vec
                        entity_map[eid] = (eid, name, merged_def)
                        entity_labels_to_upsert.append(
                            (label_norm, eid, _label_kind_for(name, label_norm), label_norm, chunk_id)
                        )
                        return eid

                sense_id = compute_mdhash_id(
                    f"{label_norm}::{clean_def.lower()}", prefix="entity-"
                )
                precomputed_entity_vecs[sense_id] = cand_vec
                existing_senses.append((sense_id, label_norm, clean_def, cand_vec))
                entity_map[sense_id] = (sense_id, label_norm, clean_def)
                entity_labels_to_upsert.append(
                    (label_norm, sense_id, "skos:prefLabel", label_norm, chunk_id)
                )
                self._graph_rlm_stats["homograph_splits"] += 1
                return sense_id

            chunk_to_entities: dict[str, set[str]] = {}
            fact_nodes_map: dict[str, tuple[str, str, str, str, str, str, str, str]] = {}
            edges_with_sources: list[tuple[str, str, str, float, str]] = []

            for cid, st_list, ner_entities in record_structured:
                chunk_to_entities.setdefault(cid, set())
                resolved_in_chunk: set[str] = set()
                for st in st_list:
                    s_id = _resolve_mention_entity_id(st.subj, st.subj_def, cid)
                    o_id = _resolve_mention_entity_id(st.obj, st.obj_def, cid)
                    resolved_in_chunk.add(st.subj)
                    resolved_in_chunk.add(st.obj)
                    chunk_to_entities[cid].add(s_id)
                    chunk_to_entities[cid].add(o_id)

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
                        )
                    if s_id != o_id:
                        edges_with_sources.append((s_id, o_id, "fact", 1.0, cid))
                        edges_with_sources.append((o_id, s_id, "fact", 1.0, cid))

                for e_norm, e_def in ner_entities:
                    if e_norm in resolved_in_chunk:
                        continue
                    resolved_in_chunk.add(e_norm)
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
                (eid, vec)
                for eid, vec in precomputed_entity_vecs.items()
                if eid not in existing_entity_vecs
            ]
            if missing_entities:
                if progress_callback is not None:
                    progress_callback(f"indexing: embedding {len(missing_entities)} entity vectors")
                e_texts = [
                    self._format_entity_embedding_text(name, defn)
                    for _, name, defn in missing_entities
                ]
                e_vecs = self._encode_in_batches(
                    e_texts, instruction="", stage="entity_embed"
                )
                entity_vec_pairs.extend(
                    (eid, v) for (eid, _, _), v in zip(missing_entities, e_vecs, strict=False)
                )

            # Compute missing fact vectors using contextual `# {s} {p} {o}\n{spo_def} | {subj_def} | {obj_def}`
            all_fact_ids = [fid for fid, _, _, _, _, _, _, _ in fact_nodes]
            existing_fact_vecs = get_existing_ids(self.conn, "fact_vec", all_fact_ids)
            missing_facts = [
                f_row for f_row in fact_nodes if f_row[0] not in existing_fact_vecs
            ]
            fact_vec_pairs: list[tuple[str, np.ndarray]] = []
            if missing_facts:
                if progress_callback is not None:
                    progress_callback(f"indexing: embedding {len(missing_facts)} fact vectors")
                fact_texts: list[str] = []
                for _, _s_id, pred, _o_id, raw_json, spo_d, subj_d, obj_d in missing_facts:
                    items = json.loads(raw_json)
                    fact_texts.append(
                        self._format_fact_embedding_text(
                            items[0], pred, items[2], spo_d, subj_d, obj_d
                        )
                    )
                f_vecs = self._encode_in_batches(
                    fact_texts, instruction="", stage="fact_embed"
                )
                fact_vec_pairs = [
                    (f_row[0], v) for f_row, v in zip(missing_facts, f_vecs, strict=False)
                ]

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
                        (fid, s_id, pred, o_id, raw_json, spo_d)
                        for fid, s_id, pred, o_id, raw_json, spo_d, _, _ in fact_nodes
                    ],
                )
                if entity_vec_pairs:
                    upsert_entity_vectors(conn, entity_vec_pairs)
                if fact_vec_pairs:
                    upsert_fact_vectors(conn, fact_vec_pairs)
                upsert_edges_with_sources(conn, edges_with_sources)

                valid_entity_ids = [
                    eid
                    for eid, name, _ in entity_nodes
                    if len(re.sub(r"[^A-Za-z0-9]", "", name)) >= 1
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
                    cand_meta: dict[str, tuple[str, str]] = {}
                    for i in range(0, len(cand_eids), 900):
                        batch_eids = cand_eids[i : i + 900]
                        placeholders = ",".join("?" for _ in batch_eids)
                        for row in conn.execute(
                            f"SELECT entity_id, name, COALESCE(definition, '') FROM entities WHERE entity_id IN ({placeholders})",
                            batch_eids,
                        ).fetchall():
                            cand_meta[str(row[0])] = (str(row[1]), str(row[2]))

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
                            float(c["similarity"]),
                        ),
                        reverse=True,
                    )
                    llm_candidates = candidates[:16]
                    overflow_candidates = candidates[16:]

                    decisions: list[dict[str, str]] = []
                    if llm_candidates:
                        syn_fp = self._synonym_fingerprint(active_program)
                        pairs_fp = hashlib.sha256(
                            json.dumps(
                                llm_candidates, sort_keys=True, ensure_ascii=False
                            ).encode("utf-8")
                        ).hexdigest()
                        if (syn_fp, pairs_fp) in self._synonym_fp_cache:
                            cached_decs, cached_syn_tr, cached_syn_tok = (
                                self._synonym_fp_cache[(syn_fp, pairs_fp)]
                            )
                            decisions = list(cached_decs)
                            self._captured_graph_traces.extend(cached_syn_tr)
                            self._graph_rlm_stats["synonym_judge_tokens"] += cached_syn_tok
                        else:
                            syn_audit: list[dict[str, Any]] = getattr(
                                active_program, "synonym_audit_trail", []
                            )
                            tel: list[dict[str, Any]] = getattr(
                                active_program, "telemetry", []
                            )
                            prev_syn_audit_len = len(syn_audit)
                            prev_tel_len = len(tel)
                            local_syn_trace: list[Any] = []
                            with dspy.context(trace=local_syn_trace, usage_tracker=None):
                                decisions = active_program.judge_synonyms(llm_candidates)
                            syn_traces = self._extract_named_trace_entries(
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
                        if rel in ("skos:altLabel", "skos:hiddenLabel", "exact_match") or (
                            same_name and d_sim >= 0.45
                        ):
                            if canon and (
                                canon in (src, raw_src)
                                or text_processing(canon) == text_processing(name_src)
                            ) and text_processing(canon) != text_processing(name_tgt):
                                merge_src, merge_dst = tgt, src
                            else:
                                merge_src, merge_dst = src, tgt
                            lkind = (
                                "skos:hiddenLabel"
                                if rel == "skos:hiddenLabel"
                                or str(dec.get("label_type", "")) == "hiddenLabel"
                                else "skos:altLabel"
                            )
                            merge_entities(conn, merge_src, merge_dst, label_kind=lkind)
                            merged_into[merge_src] = merge_dst
                            self._graph_rlm_stats["merged_entities"] += 1
                        elif rel in ("skos:related", "related"):
                            if same_name:
                                semantic_batch.append((src, tgt, "owl:differentFrom", 0.0))
                                self._graph_rlm_stats["different_from_edges"] += 1
                            else:
                                semantic_batch.append((src, tgt, "skos:related", max(0.5, sim)))
                                self._graph_rlm_stats["related_edges"] += 1
                        elif rel in ("owl:differentFrom", "different", "different_from"):
                            semantic_batch.append((src, tgt, "owl:differentFrom", 0.0))
                            self._graph_rlm_stats["different_from_edges"] += 1

                    for c in llm_candidates:
                        ea, eb = str(c["entity_a"]), str(c["entity_b"])
                        pkey = (min(ea, eb), max(ea, eb))
                        if pkey in adjudicated_pairs:
                            continue
                        if text_processing(str(c["name_a"])) == text_processing(str(c["name_b"])):
                            src = _resolve_active_eid(ea)
                            tgt = _resolve_active_eid(eb)
                            if src != tgt:
                                if float(c.get("dense_sim", 0.0)) >= 0.45:
                                    merge_entities(conn, src, tgt, label_kind="skos:altLabel")
                                    merged_into[src] = tgt
                                    self._graph_rlm_stats["merged_entities"] += 1
                                else:
                                    semantic_batch.append((src, tgt, "owl:differentFrom", 0.0))
                                    self._graph_rlm_stats["different_from_edges"] += 1

                    for c in overflow_candidates:
                        src = _resolve_active_eid(str(c["entity_a"]))
                        tgt = _resolve_active_eid(str(c["entity_b"]))
                        if src == tgt:
                            continue
                        if text_processing(str(c["name_a"])) == text_processing(str(c["name_b"])):
                            if float(c.get("dense_sim", 0.0)) >= 0.45:
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
                all_violations = lint_graph_shacl(conn, auto_heal=False)
                violations = [
                    v
                    for v in all_violations
                    if (v.focus_node, v.shape) not in self._attempted_shacl_keys
                    or v.focus_node in batch_node_ids
                    or (
                        "|" in v.focus_node
                        and any(p.strip() in batch_node_ids for p in v.focus_node.split("|"))
                    )
                    or (
                        "->" in v.focus_node
                        and any(p.strip() in batch_node_ids for p in v.focus_node.split("->"))
                    )
                ]
                batch_shacl_before = len(violations)
                self._graph_rlm_stats["shacl_violations_before"] += batch_shacl_before
                self._graph_rlm_stats["shacl_violations_initial"] = self._graph_rlm_stats[
                    "shacl_violations_before"
                ]
                batch_attempted_keys: set[tuple[str, str]] = set()
                if violations:
                    for v in violations:
                        v_key = (v.focus_node, v.shape)
                        self._attempted_shacl_keys.add(v_key)
                        batch_attempted_keys.add(v_key)
                    bounded_violations = violations
                    focus_contexts = [
                        inspect_focus_node(conn, v.focus_node)
                        for v in violations[:16]
                    ]
                    shacl_fp = self._shacl_fingerprint(active_program)
                    viol_fp = hashlib.sha256(
                        json.dumps(
                            {
                                "violations": [v.to_dict() for v in bounded_violations],
                                "focus_contexts": focus_contexts,
                            },
                            sort_keys=True,
                            ensure_ascii=False,
                            default=str,
                        ).encode("utf-8")
                    ).hexdigest()
                    if (shacl_fp, viol_fp) in self._shacl_fp_cache:
                        cached_reps, cached_shacl_tr, cached_shacl_tok = (
                            self._shacl_fp_cache[(shacl_fp, viol_fp)]
                        )
                        self._captured_graph_traces.extend(cached_shacl_tr)
                        self._graph_rlm_stats["shacl_healer_tokens"] += cached_shacl_tok
                        if cached_reps:
                            applied = self._apply_shacl_repairs(conn, cached_reps)
                            self._graph_rlm_stats["shacl_repairs_applied"] += applied
                        all_violations = lint_graph_shacl(conn, auto_heal=False)
                    else:
                        shacl_audit: list[dict[str, Any]] = getattr(
                            active_program, "shacl_audit_trail", []
                        )
                        shacl_tel: list[dict[str, Any]] = getattr(
                            active_program, "telemetry", []
                        )
                        prev_shacl_audit_len = len(shacl_audit)
                        prev_tel_len = len(shacl_tel)
                        self._active_repl_repair_keys = set()
                        self._active_repl_repairs_list = []
                        local_shacl_trace: list[Any] = []
                        try:
                            with dspy.context(trace=local_shacl_trace, usage_tracker=None):
                                repairs = active_program.heal_shacl_violations(
                                    bounded_violations,
                                    focus_context=json.dumps(focus_contexts, ensure_ascii=False),
                                )
                        except Exception as exc:
                            import logging
                            logging.warning("heal_shacl_violations encountered error: %s. Continuing indexing.", exc)
                            repairs = []
                        finally:
                            repl_applied_keys = (
                                set(self._active_repl_repair_keys)
                                if self._active_repl_repair_keys
                                else set()
                            )
                            repl_repairs_list = (
                                list(self._active_repl_repairs_list)
                                if self._active_repl_repairs_list
                                else []
                            )
                            self._active_repl_repair_keys = None
                            self._active_repl_repairs_list = None
                        shacl_traces = self._extract_named_trace_entries(
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
                            if isinstance(r, Mapping)
                            and self._repair_signature(r) not in repl_applied_keys
                        ]
                        if unapplied_repairs:
                            applied = self._apply_shacl_repairs(conn, unapplied_repairs)
                            self._graph_rlm_stats["shacl_repairs_applied"] += applied
                        all_repairs_for_cache = [*repl_repairs_list, *unapplied_repairs]
                        self._shacl_fp_cache[(shacl_fp, viol_fp)] = (
                            all_repairs_for_cache,
                            list(shacl_traces),
                            shacl_tok,
                        )
                        all_violations = lint_graph_shacl(conn, auto_heal=False)
                batch_shacl_remaining = sum(
                    1
                    for v in all_violations
                    if (v.focus_node, v.shape) in batch_attempted_keys
                )
                self._graph_rlm_stats["shacl_violations_after"] = len(all_violations)
                self._graph_rlm_stats["shacl_violations_remaining"] = len(all_violations)
                self._graph_rlm_stats["shacl_conforms"] = len(all_violations) == 0

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
            self._active_extraction_fingerprint = ext_fp
            self._active_openie_fingerprint = openie_fp
            self._opt_indexed_cids.update(chunk_ids)
            for cid in chunk_ids:
                self._chunk_indexed_ext_fp[cid] = ext_fp
            return len(chunks)

    def delete_chunks(self, chunk_ids: Sequence[str]) -> None:
        """Atomically delete chunks and cascade through affected edges, facts, and entities."""
        if not chunk_ids:
            return
        with self._lock:
            delete_chunks(self.conn, chunk_ids)
            for cid in chunk_ids:
                self._opt_indexed_cids.discard(cid)
                self._chunk_indexed_ext_fp.pop(cid, None)
            self._csr_graph = None

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
            graph = self._ensure_csr_graph()
            if len(graph.chunk_indices) == 0 or top_k <= 0:
                return [], []

            # 1. Pluggable Hybrid Fact Candidate Retrieval (Dense KNN + FTS5 + RRF)
            # Step 1: Retrieve top-k dense vector hits from fact_vec
            query_fact_vec = self._encode_texts(
                [query], instruction=FACT_INSTRUCTION, stage="query_fact_embed"
            )[0]
            dense_hits = search_fact_vectors_knn(
                self.conn, query_fact_vec, top_k=self.fact_candidate_pool_k
            )

            # Step 2: Retrieve top-k lexical FTS hits from fact_fts (plus query_intent expansion)
            fts_hits = search_facts_fts(
                self.conn, query, top_k=self.fact_fts_pool_k
            )
            intent_query = (
                active_program.extract_query_intent(query)
                if allow_intent_path
                else query
            )
            intent_fts_hits = (
                search_facts_fts(self.conn, intent_query, top_k=self.fact_fts_pool_k)
                if (
                    allow_intent_path
                    and intent_query
                    and intent_query.strip().lower() != query.strip().lower()
                )
                else []
            )

            # Step 3: RRF merge -> max (fact_candidate_pool_k + fact_fts_pool_k) candidate list length
            rrf_scores: dict[str, float] = {}
            for rank, (fid, _) in enumerate(dense_hits, start=1):
                rrf_scores[fid] = rrf_scores.get(fid, 0.0) + (1.0 / (self.fact_rrf_k + rank))
            for rank, (fid, _) in enumerate(fts_hits, start=1):
                rrf_scores[fid] = rrf_scores.get(fid, 0.0) + self.fact_fts_weight * (
                    1.0 / (self.fact_rrf_k + rank)
                )
            for rank, (fid, _) in enumerate(intent_fts_hits, start=1):
                rrf_scores[fid] = rrf_scores.get(fid, 0.0) + 0.5 * self.fact_fts_weight * (
                    1.0 / (self.fact_rrf_k + rank)
                )

            sorted_rrf = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)
            max_pool = self.fact_candidate_pool_k + self.fact_fts_pool_k
            candidate_fids = [fid for fid, _ in sorted_rrf[:max_pool]]

            fact_triples = get_facts_by_ids(self.conn, candidate_fids)
            valid_candidates = [
                (fid, fact_triples[fid], rrf_scores[fid])
                for fid in candidate_fids
                if fid in fact_triples
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
            dense_norm_dict = {
                fid: float(dense_norm_all[i])
                for i, (fid, _) in enumerate(dense_hits)
            }
            fts_norm_all = (
                min_max_normalize(np.array([s for _, s in fts_hits], dtype=np.float64))
                if fts_hits
                else np.array([])
            )
            fts_norm_dict = {
                fid: float(fts_norm_all[i])
                for i, (fid, _) in enumerate(fts_hits)
            }

            kept_facts_with_scores: list[tuple[tuple[str, str, str], float]] = []
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
                        kept_facts_with_scores.append((candidate_triples[idx], float(candidate_scores[idx])))

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
            chunk_vec_hits = search_chunk_vectors_knn(
                self.conn, query_passage_vec, top_k=self.passage_seed_pool_k
            )
            fts_hits = search_chunks_fts(self.conn, query, top_k=self.fts_seed_pool_k)

            passage_weights: dict[str, float] = {}
            if chunk_vec_hits:
                v_scores = min_max_normalize(
                    np.array([s for _, s in chunk_vec_hits], dtype=np.float64)
                )
                for (cid, _), s in zip(chunk_vec_hits, v_scores, strict=False):
                    passage_weights[cid] = float(s)

            if fts_hits:
                fts_scores = min_max_normalize(
                    np.array([s for _, s in fts_hits], dtype=np.float64)
                )
                for (cid, _), s in zip(fts_hits, fts_scores, strict=False):
                    passage_weights[cid] = (
                        passage_weights.get(cid, 0.0) + self.fts_weight * float(s)
                    )

            # 3. PPR Execution or Dense Fallback
            pool_k = max(top_k * 4, 50)
            ranked_chunks: list[tuple[str, float]] = []

            if not kept_facts_with_scores:
                sorted_passages = sorted(
                    passage_weights.items(), key=lambda kv: kv[1], reverse=True
                )
                ranked_chunks = sorted_passages[:pool_k]
            else:
                surface_mentions: list[str] = []
                for (s, _, o), _ in kept_facts_with_scores:
                    surface_mentions.append(str(text_processing(s)))
                    surface_mentions.append(str(text_processing(o)))
                resolved_map = resolve_entity_labels(
                    self.conn, surface_mentions, query_vec=query_fact_vec
                )

                seed_entities: dict[str, list[float]] = {}
                for (s, _, o), score in kept_facts_with_scores:
                    s_norm = str(text_processing(s))
                    o_norm = str(text_processing(o))
                    s_ids = resolved_map.get(s_norm) or [
                        compute_mdhash_id(s_norm, prefix="entity-")
                    ]
                    o_ids = resolved_map.get(o_norm) or [
                        compute_mdhash_id(o_norm, prefix="entity-")
                    ]
                    for s_id in s_ids:
                        seed_entities.setdefault(s_id, []).append(score)
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

            kept_facts_triples = [t for t, _ in kept_facts_with_scores]
            if not ranked_chunks:
                return [], kept_facts_triples

            # 4. MaxP Passage Span Deduplication + 1-Indexed Line Hydration from Disk
            hydrated_hits = self._hydrate_maxp_passages(ranked_chunks, top_k)
            return hydrated_hits, kept_facts_triples

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
        """Multi-hop QA retrieval and answer generation evaluated with Pareto sizecoding."""
        results: list[dict[str, Any]] = []
        for i, q in enumerate(queries):
            pred = self.program(question=q)
            ans = str(getattr(pred, "answer", "")).strip()
            reasoning = str(getattr(pred, "reasoning", "")).strip()
            formatted_context = str(getattr(pred, "context", ""))
            cids = list(getattr(pred, "retrieved_chunk_ids", ()))[: self.retrieval_top_k]
            scores = [
                float(s)
                for s in list(getattr(pred, "retrieved_scores", ()))[: self.retrieval_top_k]
            ]

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
                item["gold_answers"] = (
                    list(gold_answers[i])
                    if isinstance(gold_answers[i], list)
                    else [gold_answers[i]]
                )

            results.append(item)
        return results

    class _ActiveTraceComponentSelector:
        """GEPA component selector that dynamically prioritizes predictors with active execution traces."""

        def __init__(self, program: HippoRAGProgram) -> None:
            self.program = program
            self.cursor = 0

        def __call__(
            self,
            state: Any,
            trajectories: list[Any],
            subsample_scores: list[float],
            candidate_idx: int,
            candidate: dict[str, str],
        ) -> list[str]:
            active_sigs = set()
            for data in trajectories or []:
                for t in data.get("trace", ()):
                    sig = getattr(t[0], "signature", None)
                    if sig is not None:
                        active_sigs.add(sig)

            active_names: list[str] = []
            name_to_pred = dict(self.program.named_predictors())
            for name in getattr(state, "list_of_named_predictors", list(candidate.keys())):
                pred = name_to_pred.get(name)
                pred_sig = getattr(pred, "signature", None)
                if pred_sig is not None:
                    cand_sig = (
                        pred_sig.with_instructions(candidate[name])
                        if name in candidate and hasattr(pred_sig, "with_instructions")
                        else pred_sig
                    )
                    pred_fields = tuple(getattr(pred_sig, "fields", {}).keys())
                    for a_sig in active_sigs:
                        is_eq = (
                            cand_sig.equals(a_sig)
                            if hasattr(cand_sig, "equals")
                            else cand_sig == a_sig
                        )
                        a_fields = tuple(getattr(a_sig, "fields", {}).keys())
                        if is_eq or (pred_fields and pred_fields == a_fields):
                            active_names.append(name)
                            break

            if not active_names:
                pid = getattr(
                    state, "named_predictor_id_to_update_next_for_program_candidate", [0]
                )[candidate_idx]
                num_preds = len(state.list_of_named_predictors)
                state.named_predictor_id_to_update_next_for_program_candidate[candidate_idx] = (
                    pid + 1
                ) % num_preds
                return [state.list_of_named_predictors[pid]]

            selected = active_names[self.cursor % len(active_names)]
            self.cursor += 1
            return [selected]

    def evolve_prompts(
        self,
        trainset: Sequence[Any],
        valset: Sequence[Any] | None = None,
        *,
        mode: Literal["miprov2", "gepa", "hybrid"] = "hybrid",
        save_path: Path | str | None = None,
        reflection_lm: Any = None,
        reflection_worker_lm: Any = None,
        auto: Literal["light", "medium", "heavy"] | None = "light",
        num_candidates: int | None = None,
        num_trials: int | None = None,
        progress_callback: Callable[[str], None] | None = None,
        opt_chunks: Sequence[tuple[ChunkPointer, str]] | None = None,
        sleep_seconds: float = 0.0,
    ) -> HippoRAGProgram:
        """Evolve DSPy prompts via MIPROv2, GEPA, or two-phase hybrid with composite Sizecoding feedback."""
        if not trainset:
            return self.program

        eff_sleep = (
            sleep_seconds
            if sleep_seconds > 0
            else getattr(getattr(self, "config", None), "optimize_sleep_seconds", 0.0)
        )
        from .inference.chat import set_yaw_sleep_seconds

        set_yaw_sleep_seconds(eff_sleep)

        if opt_chunks is not None:
            existing_cids = {p.chunk_id for p, _ in self._opt_chunks_pool}
            for ptr, text in opt_chunks:
                if ptr.chunk_id not in existing_cids:
                    self._opt_chunks_pool.append((ptr, text))
                    existing_cids.add(ptr.chunk_id)

        self._baseline_openie_fp = self._openie_fingerprint(self.program)
        self._hydrate_indexed_cids_from_db(self.program)
        if len(self._opt_chunks_pool) <= 64 and self._opt_chunks_pool:
            self._opt_val_chunks_pool = list(self._opt_chunks_pool)
        else:
            val_pool: list[tuple[ChunkPointer, str]] = []
            seen_val_cids: set[str] = set()
            for ex in valset or trainset:
                ex_docs = getattr(ex, "item_docs", None)
                if ex_docs:
                    for p, t in self._resolve_chunks_for_docs(ex_docs):
                        if p.chunk_id not in seen_val_cids:
                            seen_val_cids.add(p.chunk_id)
                            val_pool.append((p, t))
            self._opt_val_chunks_pool = (
                val_pool if val_pool else list(self._opt_chunks_pool[:64])
            )

        eval_count = 0

        def _build_stage_diagnostic(
            pred_name_str: str,
            pred_trace: Any,
            gold_answers: list[str],
            missed: list[str],
        ) -> str:
            trace_entries = list(pred_trace) if isinstance(pred_trace, (list, tuple)) else []
            out_summaries: list[str] = []
            for entry in trace_entries[:3]:
                if isinstance(entry, (list, tuple)) and len(entry) >= 3:
                    outputs = entry[2]
                    if isinstance(outputs, dict):
                        out_summaries.append(
                            ", ".join(f"{k}={repr(v)[:80]}" for k, v in list(outputs.items())[:3])
                        )
                    else:
                        attrs = [
                            f"{k}={repr(getattr(outputs, k))[:80]}"
                            for k in ("named_entities", "triples", "definitions", "decisions", "repairs", "intent_query", "selected_fact_indices", "answer")
                            if hasattr(outputs, k)
                        ]
                        if attrs:
                            out_summaries.append(", ".join(attrs))
            trace_summary = "; ".join(out_summaries) if out_summaries else "no_trace_outputs"
            target_hint = f"gold_answers={gold_answers[:2]!r}" if gold_answers else "no_gold_answers"
            miss_hint = f"missed={len(missed)}" if missed else "all_retrieved"
            return (
                f"Stage[{pred_name_str}]: trace_calls={len(trace_entries)} "
                f"({trace_summary}) | {target_hint} | {miss_hint}"
            )

        def _evaluate_prediction(
            gold: Any,
            pred: Any,
            pred_name: Any = None,
            pred_trace: Any = None,
        ) -> dspy.Prediction:
            nonlocal eval_count
            eval_count += 1
            if progress_callback is not None:
                progress_callback(f"optimizing ({mode}, auto={auto}): evaluating candidate ({eval_count})")
            gold_answers_raw = getattr(gold, "gold_answers", getattr(gold, "answer", []))
            if isinstance(gold_answers_raw, str):
                gold_answers = [gold_answers_raw]
            else:
                gold_answers = [str(a) for a in (gold_answers_raw or ())]

            pred_answer = str(getattr(pred, "answer", "")).strip()
            context_str = str(getattr(pred, "context", ""))

            gold_docs_raw = getattr(gold, "gold_docs", None)
            retrieved_cids_raw = getattr(pred, "retrieved_chunk_ids", None)
            retrieved_passages_raw = getattr(pred, "retrieved_passages", None)
            retrieved_seq = [
                str(x)
                for x in (
                    retrieved_cids_raw
                    if retrieved_cids_raw
                    else (retrieved_passages_raw or ())
                )
            ]
            retrieved_texts_seq = [str(x) for x in (retrieved_passages_raw or ())]

            missed: list[str] = []
            recalls_by_k: dict[int, float] = {1: 1.0, 2: 1.0, 5: 1.0, 10: 1.0, 20: 1.0}
            if gold_docs_raw:
                total_gold = max(1, len(gold_docs_raw))
                top_20_cids = [c for c in retrieved_seq[:20] if c.startswith("chunk-")]
                hydrated_by_cid: dict[str, str] = {}
                if top_20_cids:
                    with self._lock:
                        ptrs = get_chunk_pointers(self.conn, top_20_cids)
                    ptrs_scores = [(ptrs[c], 0.0) for c in top_20_cids if c in ptrs]
                    if ptrs_scores:
                        for hp in hydrate_passages(ptrs_scores, self.workspace_root):
                            if hp.text:
                                hydrated_by_cid[hp.pointer.chunk_id] = hp.text.strip()
                for k in (1, 2, 5, 10, 20):
                    top_k_cids = retrieved_seq[:k]
                    top_k_set = set(top_k_cids) | set(retrieved_texts_seq[:k])
                    top_k_texts = {
                        hydrated_by_cid[c] for c in top_k_cids if c in hydrated_by_cid
                    } | set(retrieved_texts_seq[:k])
                    hits_k = 0
                    for gd in gold_docs_raw:
                        gd_str = str(gd).strip()
                        gd_cid = compute_mdhash_id(gd_str, prefix="chunk-")
                        island_matched = False
                        for ht in top_k_texts:
                            clean_ht = ht.split("\n[... AST section cropped", 1)[0].strip()
                            if len(clean_ht) >= 60 and clean_ht in gd_str:
                                island_matched = True
                                break
                            if "\n" in clean_ht:
                                body_ht = clean_ht.split("\n", 1)[1].strip()
                                if len(body_ht) >= 60 and body_ht in gd_str:
                                    island_matched = True
                                    break
                        if (
                            gd_cid in top_k_set
                            or gd_str in top_k_set
                            or gd_str in top_k_texts
                            or island_matched
                            or (k >= 5 and gd_str and gd_str in context_str)
                        ):
                            hits_k += 1
                        elif k == 5:
                            first_line = gd_str.splitlines()[0].strip() if gd_str else ""
                            snippet = first_line[:120] if first_line else gd_cid
                            missed.append(f"Missed passage: {snippet!r}")
                    recalls_by_k[k] = hits_k / total_gold

            em_score: float | None = None
            f1_score: float | None = None
            if gold_answers:
                em_score, f1_score = compute_qa_em_f1(pred_answer, gold_answers)
                if em_score < 1.0:
                    missed.append(f"Predicted={pred_answer!r} != Expected={gold_answers!r}")
                if not gold_docs_raw:
                    for k in (1, 2, 5, 10, 20):
                        recalls_by_k[k] = em_score

            usage_tokens = 0
            if hasattr(pred, "get_lm_usage") and callable(pred.get_lm_usage):
                usage_map = pred.get_lm_usage() or {}
                if isinstance(usage_map, (int, float)):
                    usage_tokens = int(usage_map)
                elif isinstance(usage_map, dict):
                    for model_usage in usage_map.values():
                        if isinstance(model_usage, dict):
                            usage_tokens += int(
                                model_usage.get("completion_tokens")
                                or model_usage.get("output_tokens")
                                or model_usage.get("total_tokens")
                                or 0
                            )
                        elif isinstance(model_usage, (int, float)):
                            usage_tokens += int(model_usage)
            if usage_tokens <= 0 and hasattr(self.program, "telemetry") and self.program.telemetry:
                usage_tokens = sum(
                    int(entry.get("tokens_used", 0))
                    for entry in self.program.telemetry[-20:]
                    if isinstance(entry, dict)
                )
            if usage_tokens <= 0:
                raise RuntimeError(
                    "Missing LM token usage on prediction (`pred.get_lm_usage()` returned 0 tokens). "
                    "Ensure `track_usage=True` is active and task LMs report non-zero token usage."
                )
            apm_hit: bool | None = None
            if gold_answers:
                passages_for_apm = (
                    [p for p in context_str.split("\n\n") if p.strip()]
                    if context_str.strip()
                    else []
                )
                if passages_for_apm and hasattr(dspy, "evaluate") and hasattr(
                    dspy.evaluate, "answer_passage_match"
                ):
                    ex_apm = dspy.Example(answer=list(gold_answers))
                    pred_apm = dspy.Prediction(context=passages_for_apm)
                    apm_hit = bool(dspy.evaluate.answer_passage_match(ex_apm, pred_apm))
                else:
                    apm_hit = False

            gold_item_docs = getattr(gold, "item_docs", None)
            q_stats: dict[str, Any] | None = None
            if gold_item_docs:
                gold_cids = tuple(
                    p.chunk_id for p, _ in self._resolve_chunks_for_docs(gold_item_docs)
                )
                q_stats = self._last_scope_by_cids.get(gold_cids)
            if q_stats is None:
                q_stats = (
                    self._active_query_rlm_stats
                    if self._active_query_rlm_stats is not None
                    else self._graph_rlm_stats
                )
            usage_tokens += int(
                q_stats.get("openie_tokens", q_stats.get("cached_openie_tokens", 0))
            )
            score_pred = compute_sizecoding_pareto_score(
                recalls_by_k[5],
                recall_at_1=recalls_by_k[1],
                recall_at_2=recalls_by_k[2],
                recall_at_10=recalls_by_k[10],
                recall_at_20=recalls_by_k[20],
                em=em_score,
                f1=f1_score,
                tokens=usage_tokens,
                synonym_judge_tokens=int(q_stats.get("synonym_judge_tokens", 0)),
                shacl_healer_tokens=int(q_stats.get("shacl_healer_tokens", 0)),
                shacl_violations_initial=int(
                    q_stats.get(
                        "shacl_violations_initial",
                        q_stats.get("shacl_violations_before", 0),
                    )
                ),
                shacl_violations_remaining=int(
                    q_stats.get(
                        "shacl_violations_remaining",
                        q_stats.get("shacl_violations_after", 0),
                    )
                ),
                missing_entity_defs_count=int(
                    q_stats.get("missing_entity_defs_count", 0)
                ),
                missing_entity_defs_tokens=int(
                    q_stats.get("missing_entity_defs_tokens", 0)
                ),
                pred_name=str(pred_name) if pred_name else None,
                missed_items=missed,
            )
            if apm_hit is not None:
                score_pred.answer_passage_match = 1.0 if apm_hit else 0.0
                score_pred.feedback = (
                    f"{score_pred.feedback} | AnswerPassageMatch={1 if apm_hit else 0}"
                )
            if pred_name:
                stage_diag = _build_stage_diagnostic(
                    str(pred_name),
                    pred_trace,
                    gold_answers,
                    missed,
                )
                score_pred.feedback = f"{score_pred.feedback} | {stage_diag}"
            return score_pred

        def _mipro_metric(gold: Any, pred: Any, trace: Any = None) -> float | bool:
            res = _evaluate_prediction(gold, pred)
            if trace is not None:
                f1_val = float(getattr(res, "f1", 0.0) or 0.0)
                apm_val = float(getattr(res, "answer_passage_match", 0.0) or 0.0)
                r5_val = float(getattr(res, "recall_at_5", 0.0) or 0.0)
                return bool(
                    f1_val >= 0.25 or apm_val > 0.0 or (r5_val >= 1.0 and f1_val > 0.0)
                )
            return float(res.score)

        def _gepa_metric(
            gold: Any,
            pred: Any,
            trace: Any = None,
            pred_name: Any = None,
            pred_trace: Any = None,
        ) -> Any:
            return _evaluate_prediction(
                gold,
                pred,
                pred_name=pred_name,
                pred_trace=pred_trace,
            )

        compiled = self.program
        resolved_ref_lm = (
            reflection_lm
            or self.program.orchestrator_lm
            or self.program.rlm_sub_lm
            or getattr(dspy.settings, "lm", None)
        )
        resolved_worker_lm = (
            reflection_worker_lm
            or self.program.rlm_sub_lm
            or self.program.orchestrator_lm
            or resolved_ref_lm
        )
        proposer = RLMInstructionProposer(
            orchestrator_lm=resolved_ref_lm,
            worker_lm=resolved_worker_lm,
            enable_gbnf=True,
            rlm_max_iters=10,
            rlm_max_llm_calls=20,
        )

        # Disable DSPy disk and memory cache during prompt evolution via configure_yaw_dspy_cache
        # so DSPy's `if not result.cache_hit:` in `dspy/clients/execution.py` never drops token usage
        # on repeated predictor calls across GEPA rounds.
        configure_yaw_dspy_cache(
            self._dspy_cache_dir,
            enable_disk_cache=False,
            enable_memory_cache=False,
            restrict_pickle=True,
        )

        try:
            if mode in ("miprov2", "hybrid") and hasattr(dspy, "MIPROv2"):
                import inspect

                auto_val = auto if auto not in ("none", None, "") else None
                optimizer_kwargs: dict[str, Any] = {
                    "metric": _mipro_metric,
                    "prompt_model": resolved_ref_lm,
                    "task_model": resolved_worker_lm,
                    "max_bootstrapped_demos": 2,
                    "max_labeled_demos": 0,
                    "num_threads": 1,
                }
                compile_kwargs: dict[str, Any] = {
                    "trainset": list(trainset),
                    "valset": list(valset) if valset else None,
                }
                if num_trials is not None or num_candidates is not None:
                    optimizer_kwargs["auto"] = None
                    optimizer_kwargs["num_candidates"] = num_candidates or 1
                    compile_kwargs["num_trials"] = num_trials or 1
                    compile_kwargs["minibatch"] = False
                elif mode == "hybrid" and valset and len(valset) >= 25:
                    optimizer_kwargs["auto"] = None
                    optimizer_kwargs["num_candidates"] = 3
                    compile_kwargs["num_trials"] = 4
                    compile_kwargs["max_bootstrapped_demos"] = 2
                    compile_kwargs["max_labeled_demos"] = 0
                    compile_kwargs["minibatch"] = True
                    compile_kwargs["minibatch_size"] = min(25, len(valset))
                    compile_kwargs["minibatch_full_eval_steps"] = 4
                else:
                    optimizer_kwargs["auto"] = auto_val or "light"
                    if valset and len(valset) < 35:
                        compile_kwargs["minibatch"] = False

                baseline_demos = {
                    name: list(getattr(pred, "demos", ()) or ())
                    for name, pred in compiled.named_predictors()
                }
                optimizer = dspy.MIPROv2(**optimizer_kwargs)
                if hasattr(optimizer, "_propose_instructions"):
                    optimizer._propose_instructions = proposer.propose_mipro_instructions
                compile_sig = inspect.signature(optimizer.compile).parameters
                if not any(
                    p.kind == inspect.Parameter.VAR_KEYWORD
                    for p in compile_sig.values()
                ):
                    compile_kwargs = {
                        k: v for k, v in compile_kwargs.items() if k in compile_sig
                    }
                with dspy.context(track_usage=True):
                    compiled = optimizer.compile(
                        compiled,
                        **compile_kwargs,
                    )
                self._prune_mismatching_demos(compiled)
                for name, pred in compiled.named_predictors():
                    if not getattr(pred, "demos", None) and baseline_demos.get(name):
                        pred.demos = list(baseline_demos[name])
                if hasattr(compiled.qa, "predict") and getattr(compiled.qa.predict, "demos", None):
                    compiled.qa.demos = list(compiled.qa.predict.demos)

            if mode in ("gepa", "hybrid") and hasattr(dspy, "GEPA"):
                gepa_log_dir = (
                    str(Path(save_path).parent / f"{Path(save_path).stem}_telemetry")
                    if save_path is not None
                    else None
                )
                gepa_kwargs: dict[str, Any] = {
                    "metric": _gepa_metric,
                    "reflection_lm": resolved_ref_lm,
                    "instruction_proposer": proposer,
                    "reflection_minibatch_size": 3,
                    "candidate_selection_strategy": "pareto",
                    "skip_perfect_score": True,
                    "use_merge": True,
                    "component_selector": self._ActiveTraceComponentSelector(compiled),
                    "num_threads": 1,
                    "track_stats": True,
                    "track_best_outputs": True,
                    "warn_on_score_mismatch": False,
                    "log_dir": gepa_log_dir,
                    "seed": 0,
                    "gepa_kwargs": {"use_cloudpickle": True, "frontier_type": "hybrid"},
                }
                if num_trials is not None:
                    gepa_kwargs["max_metric_calls"] = num_trials
                    gepa_kwargs["auto"] = None
                else:
                    gepa_kwargs["auto"] = auto if auto in ("light", "medium", "heavy") else "light"

                gepa = dspy.GEPA(**gepa_kwargs)
                with dspy.context(track_usage=True):
                    compiled = gepa.compile(
                        compiled,
                        trainset=list(trainset),
                        valset=list(valset) if valset else None,
                    )
        finally:
            configure_yaw_dspy_cache(
                self._dspy_cache_dir,
                enable_disk_cache=True,
                enable_memory_cache=True,
                restrict_pickle=True,
            )

        assert isinstance(compiled, HippoRAGProgram)
        frontier_states = self._extract_pareto_frontier_states(compiled, max_candidates=3)
        if frontier_states:
            compiled.pareto_frontier_states = frontier_states
            if len(frontier_states) >= 2:
                compiled.pareto_ensemble = ParetoProgramEnsemble.from_frontier_states(
                    compiled, frontier_states
                )

        self.program = compiled
        self._bind_program_hooks()
        self._apply_program_embedding_prefixes()

        if save_path is not None:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self.program.save(str(p), save_program=False)
            if proposer.telemetry:
                telemetry_path = p.parent / f"{p.stem}_proposer_telemetry.json"
                telemetry_path.write_text(
                    json.dumps(proposer.telemetry, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            audit_path = p.parent / f"{p.stem}_graph_rlm_audit.json"
            audit_payload = {
                "graph_rlm_stats": dict(self._graph_rlm_stats),
                "program_telemetry": list(getattr(self.program, "telemetry", ())),
                "embedding_audit_trail": list(
                    getattr(self.program, "embedding_audit_trail", ())
                ),
                "synonym_audit_trail": list(
                    getattr(self.program, "synonym_audit_trail", ())
                ),
                "shacl_audit_trail": list(
                    getattr(self.program, "shacl_audit_trail", ())
                ),
                "entity_def_audit_trail": list(
                    getattr(self.program, "entity_def_audit_trail", ())
                ),
                "proposer_simba_telemetry": list(
                    getattr(proposer, "simba_telemetry", ())
                ),
            }
            audit_path.write_text(
                json.dumps(audit_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            if valset:
                eval_path = p.parent / f"{p.stem}_eval.json"
                self.evaluate(valset, save_as_json=eval_path)

        if self._opt_chunks_pool and (
            len(self._opt_indexed_cids) < len(self._opt_chunks_pool)
            or self._active_extraction_fingerprint != self._extraction_fingerprint(self.program)
        ):
            target_openie_fp = self._openie_fingerprint(self.program)
            clear_openie = self._active_openie_fingerprint != target_openie_fp
            self._reset_sandbox_graph(clear_openie=clear_openie)
            if len(self._opt_chunks_pool) <= 32:
                rehydrate_pool = list(self._opt_chunks_pool)
            else:
                cached_pool = [
                    cp
                    for cp in self._opt_chunks_pool
                    if (cp[0].chunk_id, target_openie_fp) in self._openie_fp_cache
                ]
                rehydrate_pool = (
                    cached_pool[:64]
                    if cached_pool
                    else list(self._opt_chunks_pool[:16])
                )
            self.index_chunks(rehydrate_pool, program=self.program)

        return self.program

    optimize = evolve_prompts

    @staticmethod
    def _extract_pareto_frontier_states(
        compiled: Any,
        *,
        max_candidates: int = 3,
    ) -> list[dict[str, Any]]:
        """Extract up to `max_candidates` distinct non-dominated Pareto candidate states from `compiled.detailed_results`."""
        detailed = getattr(compiled, "detailed_results", None)
        if detailed is None:
            return []
        candidates = list(getattr(detailed, "candidates", None) or ())
        if not candidates:
            return []
        scores = [
            float(s)
            for s in (getattr(detailed, "val_aggregate_scores", None) or ())
        ]
        pareto_indices: set[int] = set()
        best_idx = getattr(detailed, "best_idx", None)
        if isinstance(best_idx, int) and 0 <= best_idx < len(candidates):
            pareto_indices.add(best_idx)
        for mapping_attr in (
            "per_val_instance_best_candidates",
            "per_objective_best_candidates",
        ):
            mapping = getattr(detailed, mapping_attr, None)
            if isinstance(mapping, Mapping):
                for idx_group in mapping.values():
                    if isinstance(idx_group, (list, tuple, set)):
                        for idx in idx_group:
                            if isinstance(idx, int) and 0 <= idx < len(candidates):
                                pareto_indices.add(idx)
                    elif isinstance(idx_group, int) and 0 <= idx_group < len(candidates):
                        pareto_indices.add(idx_group)

        ranked_indices = sorted(
            range(len(candidates)),
            key=lambda i: (
                i == best_idx,
                i in pareto_indices,
                scores[i] if i < len(scores) else 0.0,
                -i,
            ),
            reverse=True,
        )

        states: list[dict[str, Any]] = []
        seen_hashes: set[str] = set()
        for idx in ranked_indices:
            cand = candidates[idx]
            if not hasattr(cand, "dump_state"):
                continue
            cand_state = dict(cand.dump_state())
            surface_free = dict(cand_state)
            surface_free.pop("yaw_surface", None)
            raw_hash = hashlib.sha256(
                json.dumps(surface_free, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()
            if raw_hash in seen_hashes:
                continue
            seen_hashes.add(raw_hash)
            states.append(cand_state)
            if len(states) >= max_candidates:
                break
        return states

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
        """Evaluate the bound `HippoRAGProgram` on `devset` using `dspy.Evaluate` and return `EvaluationResult`."""
        semantic_f1_mod = (
            dspy.evaluate.SemanticF1()
            if include_semantic_metrics and hasattr(dspy.evaluate, "SemanticF1")
            else None
        )
        grounded_mod = (
            dspy.evaluate.CompleteAndGrounded()
            if include_semantic_metrics and hasattr(dspy.evaluate, "CompleteAndGrounded")
            else None
        )

        def _extract_gold_answers(example: Any) -> list[str]:
            raw = getattr(example, "gold_answers", getattr(example, "answer", []))
            if isinstance(raw, str):
                return [raw] if raw.strip() else []
            return [str(a) for a in (raw or ()) if str(a).strip()]

        def _compute_apm(gold_answers: list[str], context_val: Any) -> bool:
            if not gold_answers:
                return False
            if isinstance(context_val, str):
                passages = [p for p in context_val.split("\n\n") if p.strip()]
            elif isinstance(context_val, Sequence):
                passages = [str(p) for p in context_val if str(p).strip()]
            else:
                passages = []
            if not passages:
                return False
            ex_apm = dspy.Example(answer=list(gold_answers))
            pred_apm = dspy.Prediction(context=passages)
            return bool(dspy.evaluate.answer_passage_match(ex_apm, pred_apm))

        def _wrapped_metric(example: Any, pred: Any, trace: Any = None) -> float:
            _ = trace
            gold_answers = _extract_gold_answers(example)
            pred_answer = str(getattr(pred, "answer", "") or "").strip()
            context_val = getattr(pred, "context", "")
            apm_hit = _compute_apm(gold_answers, context_val)
            em_score, f1_score = (
                compute_qa_em_f1(pred_answer, gold_answers)
                if gold_answers
                else (0.0, 0.0)
            )
            if isinstance(pred, dspy.Prediction):
                pred.answer_passage_match = 1.0 if apm_hit else 0.0
                pred.em = float(em_score)
                pred.f1 = float(f1_score)
                if include_semantic_metrics and gold_answers:
                    q_str = str(getattr(example, "question", "") or "")
                    ctx_str = (
                        context_val
                        if isinstance(context_val, str)
                        else "\n\n".join(str(x) for x in context_val)
                    )
                    sem_ex = dspy.Example(question=q_str, response=gold_answers[0])
                    sem_pred = dspy.Prediction(response=pred_answer, context=ctx_str)
                    if semantic_f1_mod is not None:
                        sem_res = semantic_f1_mod(sem_ex, sem_pred)
                        pred.semantic_f1 = float(getattr(sem_res, "score", sem_res))
                    if grounded_mod is not None:
                        grd_res = grounded_mod(sem_ex, sem_pred)
                        pred.complete_and_grounded = float(
                            getattr(grd_res, "score", grd_res)
                        )

            if metric is not None:
                if metric is dspy.evaluate.answer_passage_match:
                    return 1.0 if apm_hit else 0.0
                norm_ex = example
                if not hasattr(example, "answer") and gold_answers:
                    norm_ex = dspy.Example(**example.toDict(), answer=list(gold_answers))
                raw_score = metric(norm_ex, pred)
                if hasattr(raw_score, "score"):
                    return float(raw_score.score)
                return float(raw_score)
            return float(f1_score)

        evaluator = dspy.Evaluate(
            devset=list(devset),
            metric=_wrapped_metric,
            num_threads=num_threads,
            display_progress=display_progress,
            display_table=display_table,
            save_as_json=str(save_as_json) if save_as_json is not None else None,
            save_as_csv=str(save_as_csv) if save_as_csv is not None else None,
        )
        return evaluator(self.program)


__all__ = [
    "YawGraphEngine",
]

