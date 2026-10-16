"""Native DSPy HippoRAG 2 Graph and Retrieval Engine backed by yaw.db and disk line hydration."""

import json
import re
import sqlite3
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import dspy
import numpy as np

from yaw.core import ChunkPointer, IndexStats
from yaw.daemon.graph.inference import (
    FACT_INSTRUCTION,
    PASSAGE_INSTRUCTION,
    DSPyEmbeddingAdapter,
    HippoRAGProgram,
    compute_mdhash_id,
    compute_qa_em_f1,
    compute_sizecoding_pareto_score,
    extract_entity_nodes,
    flatten_facts,
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
    delete_chunks,
    find_entity_synonym_candidates,
    get_chunk_pointers,
    get_entity_doc_frequencies,
    get_existing_ids,
    get_facts_by_ids,
    get_graph_counts,
    get_max_end_line,
    get_meta,
    get_openie_checkpoints,
    hydrate_passages,
    load_graph_topology,
    open_yaw_db,
    save_openie_checkpoints,
    search_chunk_vectors_knn,
    search_chunks_fts,
    search_fact_vectors_knn,
    search_facts_fts,
    set_meta,
    upsert_chunk_pointers,
    upsert_chunk_vectors,
    upsert_edges_with_sources,
    upsert_entities,
    upsert_entity_vectors,
    upsert_fact_vectors,
    upsert_facts,
    upsert_synonymy_edges,
)


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
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.db_path = db_path
        self._lock = threading.RLock()
        self.conn: sqlite3.Connection = open_yaw_db(db_path)
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

        self._csr_graph: CSRGraph | None = None
        self.program.set_retrieval_hook(self._program_retrieval_hook)
        self._apply_program_embedding_prefixes()

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
            if self.conn is not None:
                self.conn.close()

    def get_stats(self, config_fingerprint: str = "") -> IndexStats:
        """Return current graph index row counts and configuration fingerprint."""
        with self._lock:
            chunks, entities, facts, edges = get_graph_counts(self.conn)
            return IndexStats(
                chunks=chunks,
                entities=entities,
                facts=facts,
                edges=edges,
                config_fingerprint=config_fingerprint,
            )

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

    def _program_retrieval_hook(
        self,
        question: str,
    ) -> tuple[str, tuple[str, ...], tuple[float, ...]]:
        """Internal retrieval hook wired into HippoRAGProgram for end-to-end DSPy trace compilation."""
        hits = self.retrieve(question, top_k=max(20, self.retrieval_top_k))
        if not hits:
            return "", (), ()
        passages = tuple(p.chunk_id for p, _, _ in hits)
        scores = tuple(s for _, s, _ in hits)
        qa_hits = hits[: self.retrieval_top_k]
        formatted_context = "\n\n".join(
            f"Wikipedia Title: {p.title}\n{text}" if p.title else text
            for p, _, text in qa_hits
        )
        return formatted_context, passages, scores

    def index(self, docs: Sequence[str]) -> int:
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

            return self.index_chunks(chunks_to_index)

    def index_chunks(self, chunks: Sequence[tuple[ChunkPointer, str]]) -> int:
        """Incremental chunk indexing into yaw.db, sqlite-vec, and sparse CSR graph."""
        if not chunks:
            return 0

        with self._lock:
            chunk_ptrs = [ptr for ptr, _ in chunks]
            chunk_ids = [ptr.chunk_id for ptr in chunk_ptrs]

            # 1. Chunk vector embeddings in bounded batches
            existing_chunk_vecs = get_existing_ids(self.conn, "chunk_vec", chunk_ids)
            missing_chunks = [(p, t) for p, t in chunks if p.chunk_id not in existing_chunk_vecs]
            if missing_chunks:
                texts_to_embed = [t for _, t in missing_chunks]
                vecs = self._encode_in_batches(texts_to_embed, instruction="")
                upsert_chunk_vectors(
                    self.conn,
                    [(p.chunk_id, v) for (p, _), v in zip(missing_chunks, vecs, strict=False)],
                )

            # 2. Crash-resilient OpenIE extraction (persisted per-chunk immediately)
            cached_openie = get_openie_checkpoints(self.conn, chunk_ids)
            all_records: list[Any] = []
            for ptr, text in chunks:
                if ptr.chunk_id in cached_openie:
                    all_records.append(cached_openie[ptr.chunk_id])
                else:
                    rec = self.program.extract_openie(ptr.chunk_id, text)
                    save_openie_checkpoints(self.conn, [rec])
                    all_records.append(rec)

            # 4. Extract entities, facts, and embeddings
            all_triples: list[tuple[str, str, str]] = []
            chunk_to_entities: dict[str, set[str]] = {}
            for rec in all_records:
                cid = rec.chunk_id
                chunk_to_entities[cid] = set()
                for t in rec.triples:
                    if len(t) >= 3:
                        all_triples.append((str(t[0]), str(t[1]), str(t[2])))
                        s_norm = text_processing(str(t[0]))
                        o_norm = text_processing(str(t[2]))
                        if isinstance(s_norm, str) and s_norm:
                            chunk_to_entities[cid].add(compute_mdhash_id(s_norm, prefix="entity-"))
                        if isinstance(o_norm, str) and o_norm:
                            chunk_to_entities[cid].add(compute_mdhash_id(o_norm, prefix="entity-"))

            entity_nodes = extract_entity_nodes(all_triples)
            fact_nodes = flatten_facts(all_triples)

            upsert_entities(self.conn, entity_nodes)
            upsert_facts(self.conn, fact_nodes)

            # Entity vectors in bounded batches
            all_entity_ids = [eid for eid, _ in entity_nodes]
            existing_entity_vecs = get_existing_ids(self.conn, "entity_vec", all_entity_ids)
            missing_entities = [
                (eid, name) for eid, name in entity_nodes if eid not in existing_entity_vecs
            ]
            if missing_entities:
                e_texts = [name for _, name in missing_entities]
                e_vecs = self._encode_in_batches(e_texts, instruction="")
                upsert_entity_vectors(
                    self.conn,
                    [(eid, v) for (eid, _), v in zip(missing_entities, e_vecs, strict=False)],
                )

            # Fact vectors in bounded batches
            all_fact_ids = [fid for fid, _, _, _, _ in fact_nodes]
            existing_fact_vecs = get_existing_ids(self.conn, "fact_vec", all_fact_ids)
            missing_facts = [
                (fid, s_id, pred, o_id, raw_json)
                for fid, s_id, pred, o_id, raw_json in fact_nodes
                if fid not in existing_fact_vecs
            ]
            if missing_facts:
                fact_texts: list[str] = []
                for _, _s_id, pred, _o_id, raw_json in missing_facts:
                    items = json.loads(raw_json)
                    fact_texts.append(f"{items[0]} {pred} {items[2]}")
                f_vecs = self._encode_in_batches(fact_texts, instruction="")
                upsert_fact_vectors(
                    self.conn,
                    [(fid, v) for (fid, _, _, _, _), v in zip(missing_facts, f_vecs, strict=False)],
                )

            # 5. Build passage, fact, and synonymy edges
            edges_with_sources: list[tuple[str, str, str, float, str]] = []

            for cid, eids in chunk_to_entities.items():
                for eid in eids:
                    edges_with_sources.append((cid, eid, "passage", 1.0, cid))
                    edges_with_sources.append((eid, cid, "passage", 1.0, cid))

            for rec in all_records:
                cid = rec.chunk_id
                for t in rec.triples:
                    if len(t) < 3:
                        continue
                    s_norm = str(text_processing(str(t[0])))
                    o_norm = str(text_processing(str(t[2])))
                    if not s_norm or not o_norm or s_norm == o_norm:
                        continue
                    s_id = compute_mdhash_id(s_norm, prefix="entity-")
                    o_id = compute_mdhash_id(o_norm, prefix="entity-")
                    edges_with_sources.append((s_id, o_id, "fact", 1.0, cid))
                    edges_with_sources.append((o_id, s_id, "fact", 1.0, cid))

            upsert_edges_with_sources(self.conn, edges_with_sources)

            # Entity synonymy pairs (cosine similarity >= threshold)
            valid_entity_ids = [
                eid
                for eid, name in entity_nodes
                if len(re.sub(r"[^A-Za-z0-9]", "", name)) > 2
            ]
            if valid_entity_ids:
                synonyms = find_entity_synonym_candidates(
                    self.conn,
                    entity_ids=valid_entity_ids,
                    threshold=self.synonymy_threshold,
                    top_k=self.synonymy_top_k,
                )
                upsert_synonymy_edges(self.conn, synonyms)

            # 6. Commit ChunkPointer metadata and FTS5 index once vectors and graph edges succeed
            upsert_chunk_pointers(self.conn, chunks)

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
    ) -> tuple[list[tuple[ChunkPointer, float, str]], list[tuple[str, str, str]]]:
        """Retrieve top-k chunks and return both hydrated hits and kept recognition-memory facts."""
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

            # Step 2: Retrieve top-k lexical FTS hits from fact_fts
            fts_hits = search_facts_fts(
                self.conn, query, top_k=self.fact_fts_pool_k
            )

            # Step 3: RRF merge -> max (fact_candidate_pool_k + fact_fts_pool_k) candidate list length
            rrf_scores: dict[str, float] = {}
            for rank, (fid, _) in enumerate(dense_hits, start=1):
                rrf_scores[fid] = rrf_scores.get(fid, 0.0) + (1.0 / (self.fact_rrf_k + rank))
            for rank, (fid, _) in enumerate(fts_hits, start=1):
                rrf_scores[fid] = rrf_scores.get(fid, 0.0) + self.fact_fts_weight * (
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
                    dense_norm_dict.get(fid, fts_norm_dict.get(fid, 0.5))
                    for fid, _, _ in top_candidates
                ]

                filter_indices = self.program.filter_facts(query, candidate_triples)
                for idx in filter_indices:
                    if 0 <= idx < len(top_candidates):
                        kept_facts_with_scores.append((candidate_triples[idx], float(candidate_scores[idx])))

            if not kept_facts_with_scores and allow_intent_path:
                return self.retrieve_with_seeds(
                    self.program.extract_query_intent(query),
                    top_k=top_k,
                    allow_intent_path=False,
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
                seed_entities: dict[str, list[float]] = {}
                for (s, _, o), score in kept_facts_with_scores:
                    s_id = compute_mdhash_id(str(text_processing(s)), prefix="entity-")
                    o_id = compute_mdhash_id(str(text_processing(o)), prefix="entity-")
                    seed_entities.setdefault(s_id, []).append(score)
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
    ) -> list[tuple[ChunkPointer, float, str]]:
        """Retrieve top-k chunks for query using hybrid PPR + 1-indexed line hydration."""
        hits, _ = self.retrieve_with_seeds(query, top_k=top_k)
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
                f"Wikipedia Title: {p.title}\n{text}" if p.title else text
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
    ) -> HippoRAGProgram:
        """Evolve DSPy prompts via MIPROv2, GEPA, or two-phase hybrid with composite Sizecoding feedback."""
        if not trainset:
            return self.program

        last_tick = [time.perf_counter()]

        def _evaluate_prediction(gold: Any, pred: Any, pred_name: Any = None) -> dspy.Prediction:
            now = time.perf_counter()
            elapsed = max(1e-3, now - last_tick[0])
            last_tick[0] = now

            gold_answers_raw = getattr(gold, "gold_answers", getattr(gold, "answer", []))
            if isinstance(gold_answers_raw, str):
                gold_answers = [gold_answers_raw]
            else:
                gold_answers = [str(a) for a in (gold_answers_raw or ())]

            pred_answer = str(getattr(pred, "answer", "")).strip()
            reasoning = str(getattr(pred, "reasoning", "")).strip()
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
                            missed.append(gd_cid)
                    recalls_by_k[k] = hits_k / total_gold

            em_score: float | None = None
            f1_score: float | None = None
            if gold_answers:
                em_score, f1_score = compute_qa_em_f1(pred_answer, gold_answers)
                if em_score < 1.0:
                    missed.extend(gold_answers)
                if not gold_docs_raw:
                    for k in (1, 2, 5, 10, 20):
                        recalls_by_k[k] = em_score

            est_tokens = max(1, (len(pred_answer) + len(reasoning) + len(context_str) // 8) // 4)
            return compute_sizecoding_pareto_score(
                recalls_by_k[5],
                recall_at_1=recalls_by_k[1],
                recall_at_2=recalls_by_k[2],
                recall_at_10=recalls_by_k[10],
                recall_at_20=recalls_by_k[20],
                em=em_score,
                f1=f1_score,
                tokens=est_tokens,
                elapsed_seconds=elapsed,
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

        if mode in ("miprov2", "hybrid") and hasattr(dspy, "MIPROv2"):
            optimizer = dspy.MIPROv2(metric=_mipro_metric, auto="light")
            compiled = optimizer.compile(
                compiled,
                trainset=list(trainset),
                valset=list(valset) if valset else None,
            )

        if mode in ("gepa", "hybrid") and hasattr(dspy, "GEPA"):
            resolved_ref_lm = (
                reflection_lm
                or self.program.orchestrator_lm
                or self.program.rlm_sub_lm
                or getattr(dspy.settings, "lm", None)
            )
            gepa = dspy.GEPA(metric=_gepa_metric, reflection_lm=resolved_ref_lm, auto="light")
            compiled = gepa.compile(
                compiled,
                trainset=list(trainset),
                valset=list(valset) if valset else None,
            )

        self.program = compiled
        self.program.set_retrieval_hook(self._program_retrieval_hook)
        self._apply_program_embedding_prefixes()

        if save_path is not None:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self.program.save(str(p))

        return self.program


__all__ = [
    "YawGraphEngine",
]

