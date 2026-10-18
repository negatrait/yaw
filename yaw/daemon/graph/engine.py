"""Native DSPy HippoRAG 2 Graph and Retrieval Engine backed by yaw.db and disk line hydration."""

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
    RLMInstructionProposer,
    compute_mdhash_id,
    compute_qa_em_f1,
    compute_sizecoding_pareto_score,
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
    get_existing_ids,
    get_facts_by_ids,
    get_graph_counts,
    get_max_end_line,
    get_meta,
    get_openie_checkpoints,
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
        retrieval_top_k: int = 5,
        embed_batch_size: int = 64,
        activation: str = "scaled_tanh",
        activation_param: float | None = 5.0,
        activation_renormalize: bool = False,
        beta: float = 0.5,
        dual_weight: float | None = 0.5,
        same_sense_cosine_gate: float = _SAME_SENSE_COSINE_GATE,
        homograph_cosine_gate: float | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.db_path = db_path
        self._lock = threading.RLock()
        self.managed_conn: ManagedConnection = open_managed_yaw_db(db_path)
        self.conn: sqlite3.Connection = self.managed_conn._conn
        self.program = program
        self.embedder = embedder

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
        self._graph_rlm_stats: dict[str, Any] = {
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
        }

        self._bind_program_hooks()
        self._apply_program_embedding_prefixes()

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
            pending.append({str(k): str(v) for k, v in rep.items()})
        if not pending:
            return 0
        applied = self._apply_shacl_repairs(self.conn, pending)
        self._graph_rlm_stats["shacl_repairs_applied"] += applied
        return applied

    def _bind_program_hooks(self) -> None:
        self.program.set_retrieval_hook(self._program_retrieval_hook)
        if hasattr(self.program, "set_graph_tools"):
            self.program.set_graph_tools(
                node_inspector=lambda eid: inspect_focus_node(self.conn, eid),
                chunk_hydrator=self._hydrate_single_chunk,
                shacl_repair_applier=self._apply_shacl_repairs_from_tool,
                shacl_linter=lambda: [
                    v.to_dict() for v in lint_graph_shacl(self.conn, auto_heal=False)
                ],
            )

    def _apply_program_embedding_prefixes(self) -> None:
        if hasattr(self.embedder, "update_prefixes"):
            self.embedder.update_prefixes(
                query_prefix=getattr(self.program, "embedding_query_prefix", None),
                fact_query_prefix=getattr(self.program, "embedding_fact_prefix", None),
                document_prefix=getattr(self.program, "embedding_document_prefix", None),
            )

    def close(self) -> None:
        """Close SQLite database connection."""
        with self._lock:
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

    def _encode_in_batches(
        self,
        texts: Sequence[str],
        instruction: str = "",
    ) -> list[np.ndarray]:
        """Encode texts in bounded micro-batches to prevent HTTP/VRAM OOM on large corpora."""
        if not texts:
            return []
        results: list[np.ndarray] = []
        step = max(1, self.embed_batch_size)
        for i in range(0, len(texts), step):
            batch = texts[i : i + step]
            results.extend(self.embedder.encode(batch, instruction=instruction))
        return results

    @staticmethod
    def _format_hydrated_passage(p: ChunkPointer, text: str) -> str:
        if not p.title:
            return text
        if text.startswith(f"{p.title}\n") or text == p.title:
            return f"Wikipedia Title: {text}"
        return f"Wikipedia Title: {p.title}\n{text}"

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
        sub_d = subj_def.strip()
        obj_d = obj_def.strip()
        if s_d or sub_d or obj_d:
            return f"# {s} {p} {o}\n{s_d} | {sub_d} | {obj_d}"
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
            elif action in ("merge_altLabel", "merge_hiddenLabel") and target:
                src_node = focus.split("|", 1)[0].strip() if "|" in focus else focus
                dst_node = (
                    focus.split("|", 1)[1].strip()
                    if ("|" in focus and target == src_node)
                    else target
                )
                lkind = (
                    "skos:hiddenLabel"
                    if action == "merge_hiddenLabel"
                    else "skos:altLabel"
                )
                if src_node and dst_node and src_node != dst_node:
                    merge_entities(conn, src_node, dst_node, label_kind=lkind)
                    applied += 1
            elif action in ("add_different_from", "assert_differentFrom"):
                if "|" in focus and (not target or target == focus):
                    ea, eb = (x.strip() for x in focus.split("|", 1))
                else:
                    ea, eb = focus, target
                if ea and eb and ea != eb:
                    upsert_semantic_edges(conn, [(ea, eb, "owl:differentFrom", 0.0)])
                    applied += 1
            elif action == "assert_related":
                if "|" in focus and (not target or target == focus):
                    ea, eb = (x.strip() for x in focus.split("|", 1))
                else:
                    ea, eb = focus, target
                if ea and eb and ea != eb:
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

    def _program_retrieval_hook(
        self,
        question: str,
        program: HippoRAGProgram | None = None,
    ) -> tuple[str, tuple[str, ...], tuple[float, ...]]:
        """Internal retrieval hook wired into HippoRAGProgram for end-to-end DSPy trace compilation."""
        hits = self.retrieve(question, top_k=max(20, self.retrieval_top_k), program=program)
        if not hits:
            return "", (), ()
        passages = tuple(p.chunk_id for p, _, _ in hits)
        scores = tuple(s for _, s, _ in hits)
        qa_hits = hits[: self.retrieval_top_k]
        formatted_context = "\n\n".join(
            self._format_hydrated_passage(p, text)
            for p, _, text in qa_hits
        )
        return formatted_context, passages, scores

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

                current_lines = get_max_end_line(self.conn, "corpus/passages.txt")
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
                        )
                        chunks_to_index.append((ptr, doc))

            return self.index_chunks(chunks_to_index, progress_callback=progress_callback)

    def index_chunks(
        self,
        chunks: Sequence[tuple[ChunkPointer, str]],
        *,
        progress_callback: Callable[[str], None] | None = None,
    ) -> int:
        """Incremental chunk indexing into yaw.db, sqlite-vec, SKOS schema, and sparse CSR graph."""
        if not chunks:
            return 0

        with self._lock:
            chunk_ptrs = [ptr for ptr, _ in chunks]
            chunk_ids = [ptr.chunk_id for ptr in chunk_ptrs]

            # 1. Chunk vector embeddings in bounded batches
            existing_chunk_vecs = get_existing_ids(self.conn, "chunk_vec", chunk_ids)
            missing_chunks = [(p, t) for p, t in chunks if p.chunk_id not in existing_chunk_vecs]
            chunk_vec_pairs: list[tuple[str, np.ndarray]] = []
            if missing_chunks:
                if progress_callback is not None:
                    progress_callback(f"indexing: embedding {len(missing_chunks)} chunk vectors")
                texts_to_embed = [t for _, t in missing_chunks]
                vecs = self._encode_in_batches(texts_to_embed, instruction="")
                chunk_vec_pairs = [
                    (p.chunk_id, v) for (p, _), v in zip(missing_chunks, vecs, strict=False)
                ]

            # 2. Crash-resilient OpenIE extraction (persisted per-chunk immediately)
            cached_openie = get_openie_checkpoints(self.conn, chunk_ids)
            all_records: list[Any] = []
            attempted_missing_by_cid: dict[str, set[str]] = {}
            total_chunks = len(chunks)
            for idx, (ptr, text) in enumerate(chunks, 1):
                if ptr.chunk_id in cached_openie:
                    all_records.append(cached_openie[ptr.chunk_id])
                else:
                    if progress_callback is not None:
                        progress_callback(f"indexing: OpenIE extraction ({idx}/{total_chunks})")
                    prev_def_audit_len = len(
                        getattr(self.program, "entity_def_audit_trail", ())
                    )
                    rec = self.program.extract_openie(ptr.chunk_id, text)
                    new_def_entries = list(
                        getattr(self.program, "entity_def_audit_trail", ())[
                            prev_def_audit_len:
                        ]
                    )
                    attempted_norms: set[str] = set()
                    for entry in new_def_entries:
                        if isinstance(entry, dict):
                            self._graph_rlm_stats["missing_entity_defs_count"] += int(
                                entry.get("missing_count", 0)
                            )
                            self._graph_rlm_stats["missing_entity_defs_tokens"] += int(
                                entry.get("tokens", 0)
                            )
                            for m_ent in entry.get("missing_entities") or ():
                                m_norm = str(text_processing(str(m_ent)))
                                if m_norm:
                                    attempted_norms.add(m_norm)
                    attempted_missing_by_cid[ptr.chunk_id] = attempted_norms
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

                if missing_norms and hasattr(self.program, "define_missing_entities"):
                    prev_def_audit_len = len(
                        getattr(self.program, "entity_def_audit_trail", ())
                    )
                    repaired_pairs = self.program.define_missing_entities(
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
                    new_def_entries = list(
                        getattr(self.program, "entity_def_audit_trail", ())[
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
            if all_existing_eids:
                for i in range(0, len(all_existing_eids), 900):
                    batch_eids = all_existing_eids[i : i + 900]
                    placeholders = ",".join("?" for _ in batch_eids)
                    for row in self.conn.execute(
                        f"SELECT entity_id, name, COALESCE(definition, '') FROM entities WHERE entity_id IN ({placeholders})",
                        batch_eids,
                    ).fetchall():
                        entity_info_by_id[str(row[0])] = (str(row[1]), str(row[2]))

            # Track senses per label_norm: list of (entity_id, name, definition, unit_vec_or_None)
            senses_by_label: dict[str, list[tuple[str, str, str, np.ndarray | None]]] = {}
            for lbl, eids in existing_by_label.items():
                senses_by_label[lbl] = [
                    (
                        eid,
                        entity_info_by_id.get(eid, (lbl, ""))[0],
                        entity_info_by_id.get(eid, (lbl, ""))[1],
                        None,
                    )
                    for eid in eids
                ]

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
                    senses_by_label[label_norm] = [(eid, label_norm, clean_def, None)]
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
                cand_text = self._format_entity_embedding_text(label_norm, clean_def)
                cand_vec = self._encode_in_batches([cand_text], instruction="")[0]
                c_norm = float(np.linalg.norm(cand_vec))
                if c_norm > 0:
                    cand_vec = cand_vec / c_norm

                for idx_s, (eid, name, prev_def, prev_vec) in enumerate(existing_senses):
                    if prev_vec is None:
                        p_text = self._format_entity_embedding_text(name, prev_def)
                        prev_vec = self._encode_in_batches([p_text], instruction="")[0]
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
                e_vecs = self._encode_in_batches(e_texts, instruction="")
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
                f_vecs = self._encode_in_batches(fact_texts, instruction="")
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
                    if len(re.sub(r"[^A-Za-z0-9]", "", name)) > 2
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

                    decisions: list[dict[str, str]] = []
                    if candidates:
                        syn_audit: list[dict[str, Any]] = getattr(
                            self.program, "synonym_audit_trail", []
                        )
                        tel: list[dict[str, Any]] = getattr(
                            self.program, "telemetry", []
                        )
                        prev_syn_audit_len = len(syn_audit)
                        prev_tel_len = len(tel)
                        decisions = self.program.judge_synonyms(candidates)
                        if len(syn_audit) > prev_syn_audit_len:
                            self._graph_rlm_stats["synonym_judge_tokens"] += int(
                                syn_audit[-1].get("tokens_used", 0)
                            )
                        elif len(tel) > prev_tel_len:
                            self._graph_rlm_stats["synonym_judge_tokens"] += int(
                                tel[-1].get("tokens_used", 0)
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
                        src = _resolve_active_eid(raw_src)
                        tgt = _resolve_active_eid(raw_tgt)
                        if src == tgt:
                            continue
                        pkey = (min(raw_src, raw_tgt), max(raw_src, raw_tgt))
                        cand_info = cand_by_pair.get(pkey, {})
                        sim = float(cand_info.get("similarity", 0.5) or 0.5)
                        if rel in ("skos:altLabel", "skos:hiddenLabel", "exact_match"):
                            name_src, _ = cand_meta.get(src, cand_meta.get(raw_src, (src, "")))
                            name_tgt, _ = cand_meta.get(tgt, cand_meta.get(raw_tgt, (tgt, "")))
                            if canon and (
                                canon in (src, raw_src)
                                or canon.lower() == name_src.lower()
                            ) and canon.lower() != name_tgt.lower():
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
                            semantic_batch.append((src, tgt, "skos:related", max(0.5, sim)))
                            self._graph_rlm_stats["related_edges"] += 1
                        elif rel in ("owl:differentFrom", "different", "different_from"):
                            semantic_batch.append((src, tgt, "owl:differentFrom", 0.0))
                            self._graph_rlm_stats["different_from_edges"] += 1

                    if semantic_batch:
                        resolved_semantic_batch = [
                            (_resolve_active_eid(s), _resolve_active_eid(d), k, w)
                            for s, d, k, w in semantic_batch
                            if _resolve_active_eid(s) != _resolve_active_eid(d)
                        ]
                        if resolved_semantic_batch:
                            upsert_semantic_edges(conn, resolved_semantic_batch)

                # In-transaction Closed-World SHACL Linting & Bounded Self-Healing via shacl_healer_rlm
                violations = lint_graph_shacl(conn, auto_heal=False)
                self._graph_rlm_stats["shacl_violations_before"] += len(violations)
                self._graph_rlm_stats["shacl_violations_initial"] = self._graph_rlm_stats[
                    "shacl_violations_before"
                ]
                if violations:
                    focus_contexts = [
                        inspect_focus_node(conn, v.focus_node)
                        for v in violations[:16]
                    ]
                    shacl_audit: list[dict[str, Any]] = getattr(
                        self.program, "shacl_audit_trail", []
                    )
                    shacl_tel: list[dict[str, Any]] = getattr(
                        self.program, "telemetry", []
                    )
                    prev_shacl_audit_len = len(shacl_audit)
                    prev_tel_len = len(shacl_tel)
                    self._active_repl_repair_keys = set()
                    try:
                        repairs = self.program.heal_shacl_violations(
                            violations,
                            focus_context=json.dumps(focus_contexts, ensure_ascii=False),
                        )
                    finally:
                        repl_applied_keys = set(self._active_repl_repair_keys)
                        self._active_repl_repair_keys = None
                    if len(shacl_audit) > prev_shacl_audit_len:
                        self._graph_rlm_stats["shacl_healer_tokens"] += int(
                            shacl_audit[-1].get("tokens_used", 0)
                        )
                    elif len(shacl_tel) > prev_tel_len:
                        self._graph_rlm_stats["shacl_healer_tokens"] += int(
                            shacl_tel[-1].get("tokens_used", 0)
                        )
                    unapplied_repairs = [
                        r
                        for r in repairs
                        if self._repair_signature(r) not in repl_applied_keys
                    ]
                    if unapplied_repairs:
                        applied = self._apply_shacl_repairs(conn, unapplied_repairs)
                        self._graph_rlm_stats["shacl_repairs_applied"] += applied
                    violations = lint_graph_shacl(conn, auto_heal=False)
                self._graph_rlm_stats["shacl_violations_after"] = len(violations)
                self._graph_rlm_stats["shacl_violations_remaining"] = len(violations)
                self._graph_rlm_stats["shacl_conforms"] = len(violations) == 0

            self._csr_graph = None
            return len(chunks)

    def delete_chunks(self, chunk_ids: Sequence[str]) -> None:
        """Atomically delete chunks and cascade through affected edges, facts, and entities."""
        if not chunk_ids:
            return
        with self._lock:
            delete_chunks(self.conn, chunk_ids)
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
            query_fact_vec = self.embedder.encode([query], instruction=FACT_INSTRUCTION)[0]
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

                filter_indices = active_program.filter_facts(query, candidate_triples)
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
            query_passage_vec = self.embedder.encode([query], instruction=PASSAGE_INSTRUCTION)[0]
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
            ranked_chunks: list[tuple[str, float]] = []

            if not kept_facts_with_scores:
                sorted_passages = sorted(
                    passage_weights.items(), key=lambda kv: kv[1], reverse=True
                )
                ranked_chunks = sorted_passages[:top_k]
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
                ranked_chunks = rank_passages_by_ppr(graph, ppr_scores, top_k=top_k)

            kept_facts_triples = [t for t, _ in kept_facts_with_scores]
            if not ranked_chunks:
                return [], kept_facts_triples

            # 4. 1-Indexed Line Hydration from Disk
            ranked_cids = [cid for cid, _ in ranked_chunks]
            pointers = get_chunk_pointers(self.conn, ranked_cids)
            score_by_cid = dict(ranked_chunks)

            ptrs_with_scores = [
                (pointers[cid], score_by_cid[cid])
                for cid in ranked_cids
                if cid in pointers
            ]
            hydrated_passages = hydrate_passages(ptrs_with_scores, self.workspace_root)
            hydrated_hits = [(hp.pointer, hp.score, hp.text) for hp in hydrated_passages]

            return hydrated_hits, kept_facts_triples

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
            query_passage_vec = self.embedder.encode([query], instruction=PASSAGE_INSTRUCTION)[0]
            chunk_vec_hits = search_chunk_vectors_knn(self.conn, query_passage_vec, top_k=top_k)
            if not chunk_vec_hits:
                return []
            ranked_cids = [cid for cid, _ in chunk_vec_hits]
            pointers = get_chunk_pointers(self.conn, ranked_cids)
            score_by_cid = dict(chunk_vec_hits)
            ptrs_with_scores = [
                (pointers[cid], score_by_cid[cid])
                for cid in ranked_cids
                if cid in pointers
            ]
            hydrated_passages = hydrate_passages(ptrs_with_scores, self.workspace_root)
            return [(hp.pointer, hp.score, hp.text) for hp in hydrated_passages]

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
            hits = self.retrieve(q, top_k=self.retrieval_top_k)
            retrieved_cids = [p.chunk_id for p, _, _ in hits]
            retrieved_scores = [s for _, s, _ in hits]
            formatted_context = "\n\n".join(
                self._format_hydrated_passage(p, text)
                for p, _, text in hits
            )

            pred = self.program.answer_question(context=formatted_context, question=q)
            ans = str(getattr(pred, "answer", "")).strip()
            reasoning = str(getattr(pred, "reasoning", "")).strip()

            item: dict[str, Any] = {
                "question": q,
                "answer": ans,
                "reasoning": reasoning,
                "retrieved_chunk_ids": retrieved_cids,
                "retrieved_scores": retrieved_scores,
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

    def evolve_prompts(
        self,
        trainset: Sequence[Any],
        valset: Sequence[Any] | None = None,
        *,
        mode: Literal["miprov2", "gepa", "hybrid"] = "hybrid",
        save_path: Path | str | None = None,
        reflection_lm: Any = None,
        reflection_worker_lm: Any = None,
        auto: Literal["light", "medium", "heavy"] = "light",
        progress_callback: Callable[[str], None] | None = None,
    ) -> HippoRAGProgram:
        """Evolve DSPy prompts via MIPROv2, GEPA, or two-phase hybrid with composite Sizecoding feedback."""
        if not trainset:
            return self.program

        eval_count = 0

        def _evaluate_prediction(gold: Any, pred: Any, pred_name: Any = None) -> dspy.Prediction:
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
            retrieved_seq = [str(x) for x in (getattr(pred, "retrieved_passages", None) or ())]

            missed: list[str] = []
            recalls_by_k: dict[int, float] = {1: 1.0, 2: 1.0, 5: 1.0, 10: 1.0, 20: 1.0}
            if gold_docs_raw:
                total_gold = max(1, len(gold_docs_raw))
                for k in (1, 2, 5, 10, 20):
                    top_k_set = set(retrieved_seq[:k])
                    hits_k = 0
                    for gd in gold_docs_raw:
                        gd_str = str(gd).strip()
                        gd_cid = compute_mdhash_id(gd_str, prefix="chunk-")
                        if (
                            gd_cid in top_k_set
                            or gd_str in top_k_set
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
                if isinstance(usage_map, dict):
                    for model_usage in usage_map.values():
                        if isinstance(model_usage, dict):
                            usage_tokens += int(
                                model_usage.get("completion_tokens")
                                or model_usage.get("output_tokens")
                                or model_usage.get("total_tokens")
                                or 0
                            )
            if usage_tokens <= 0:
                raise RuntimeError(
                    "Missing LM token usage on prediction (`pred.get_lm_usage()` returned 0 tokens). "
                    "Ensure `track_usage=True` is active and task LMs report non-zero token usage."
                )
            return compute_sizecoding_pareto_score(
                recalls_by_k[5],
                recall_at_1=recalls_by_k[1],
                recall_at_2=recalls_by_k[2],
                recall_at_10=recalls_by_k[10],
                recall_at_20=recalls_by_k[20],
                em=em_score,
                f1=f1_score,
                tokens=usage_tokens,
                synonym_judge_tokens=int(self._graph_rlm_stats.get("synonym_judge_tokens", 0)),
                shacl_healer_tokens=int(self._graph_rlm_stats.get("shacl_healer_tokens", 0)),
                shacl_violations_initial=int(
                    self._graph_rlm_stats.get(
                        "shacl_violations_initial",
                        self._graph_rlm_stats.get("shacl_violations_before", 0),
                    )
                ),
                shacl_violations_remaining=int(
                    self._graph_rlm_stats.get(
                        "shacl_violations_remaining",
                        self._graph_rlm_stats.get("shacl_violations_after", 0),
                    )
                ),
                missing_entity_defs_count=int(
                    self._graph_rlm_stats.get("missing_entity_defs_count", 0)
                ),
                missing_entity_defs_tokens=int(
                    self._graph_rlm_stats.get("missing_entity_defs_tokens", 0)
                ),
                pred_name=str(pred_name) if pred_name else None,
                missed_items=missed,
            )

        def _mipro_metric(gold: Any, pred: Any, trace: Any = None) -> float:
            return float(_evaluate_prediction(gold, pred).score)

        def _gepa_metric(
            gold: Any,
            pred: Any,
            trace: Any = None,
            pred_name: Any = None,
            pred_trace: Any = None,
        ) -> Any:
            return _evaluate_prediction(gold, pred, pred_name=pred_name)

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
        )

        # Disable DSPy disk cache on task LMs during prompt evolution so DSPy's
        # `if not result.cache_hit:` in `dspy/clients/execution.py` never drops token usage
        # on repeated predictor calls across GEPA rounds.
        task_lms = [
            lm
            for lm in (
                self.program.orchestrator_lm,
                self.program.rlm_sub_lm,
                getattr(dspy.settings, "lm", None),
            )
            if lm is not None and hasattr(lm, "cache")
        ]
        saved_cache_flags = [(lm, bool(getattr(lm, "cache", False))) for lm in task_lms]
        for lm, _ in saved_cache_flags:
            setattr(lm, "cache", False)

        try:
            if mode in ("miprov2", "hybrid") and hasattr(dspy, "MIPROv2"):
                optimizer = dspy.MIPROv2(
                    metric=_mipro_metric,
                    prompt_model=resolved_ref_lm,
                    task_model=resolved_worker_lm,
                    auto=auto,
                )
                with dspy.context(track_usage=True):
                    compiled = optimizer.compile(
                        compiled,
                        trainset=list(trainset),
                        valset=list(valset) if valset else None,
                    )

            if mode in ("gepa", "hybrid") and hasattr(dspy, "GEPA"):
                gepa_log_dir = (
                    str(Path(save_path).parent / f"{Path(save_path).stem}_telemetry")
                    if save_path is not None
                    else None
                )
                gepa = dspy.GEPA(
                    metric=_gepa_metric,
                    auto=auto,
                    reflection_lm=resolved_ref_lm,
                    instruction_proposer=proposer,
                    reflection_minibatch_size=3,
                    candidate_selection_strategy="pareto",
                    skip_perfect_score=True,
                    use_merge=True,
                    component_selector="round_robin",
                    track_stats=True,
                    track_best_outputs=True,
                    warn_on_score_mismatch=False,
                    log_dir=gepa_log_dir,
                    seed=0,
                    gepa_kwargs={"use_cloudpickle": True, "frontier_type": "hybrid"},
                )
                with dspy.context(track_usage=True):
                    compiled = gepa.compile(
                        compiled,
                        trainset=list(trainset),
                        valset=list(valset) if valset else None,
                    )
        finally:
            for lm, orig_flag in saved_cache_flags:
                setattr(lm, "cache", orig_flag)

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
                "synonym_audit_trail": list(
                    getattr(self.program, "synonym_audit_trail", ())
                ),
                "shacl_audit_trail": list(
                    getattr(self.program, "shacl_audit_trail", ())
                ),
                "entity_def_audit_trail": list(
                    getattr(self.program, "entity_def_audit_trail", ())
                ),
            }
            audit_path.write_text(
                json.dumps(audit_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return self.program


__all__ = [
    "YawGraphEngine",
]

