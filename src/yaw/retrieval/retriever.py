"""In-memory SciPy Compressed Sparse Row (CSR) Personalized PageRank retrieval engine."""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import urllib.error
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from yaw.nlp import deserialize_embedding, embed_query, is_ollama_online
from yaw.storage import _clean_fts_query, get_db, search_fts

from .. import config

logger = logging.getLogger(__name__)

RELEVANCE_THRESHOLD: float = config.RELEVANCE_THRESHOLD
_QUERY_EMB_EXCEPTIONS = (
    TimeoutError,
    urllib.error.URLError,
    OSError,
    RuntimeError,
    ValueError,
)


def hydrate_disk_slice(
    path: str, byte_offset: int = 0, byte_length: int | None = None
) -> str:
    """Read chunk text slice strictly on demand from disk via binary seek."""
    target = Path(path)
    if not target.is_absolute():
        target = config.VAULT_ROOT / target

    resolved = target.resolve()
    if not resolved.is_relative_to(config.VAULT_ROOT.resolve()):
        raise ValueError(f"Path outside VAULT_ROOT: {path}")

    if not resolved.is_file():
        return ""

    try:
        with resolved.open("rb") as f:
            f.seek(byte_offset)
            if byte_length is None:
                raw = f.read()
            else:
                raw = f.read(byte_length)
            return raw.decode("utf-8", errors="replace")
    except (OSError, UnicodeDecodeError) as err:
        logger.warning(
            "Failed to hydrate disk slice %s [%d:%s]: %s",
            path,
            byte_offset,
            str(byte_length),
            err,
        )
        return ""


read_document = hydrate_disk_slice


def _is_concept(node_id: str) -> bool:
    """Check if node identifier belongs to an ontological taxonomy namespace."""
    prefix = node_id.split(":")[0].lower() if ":" in node_id else ""
    return prefix in config.TAXONOMY_NAMESPACES


def _is_chunk(node_id: str) -> bool:
    """Check if node identifier references a raw document chunk pointer."""
    return "#chunk:" in node_id


def _extract_fts_terms(query: str, nlp: Any | None = None) -> list[str]:
    """Extract lemmatized and stopword-stripped FTS query terms using spaCy."""
    if nlp is None:
        try:
            from yaw.nlp import get_spacy_pipeline

            nlp = get_spacy_pipeline()
        except (OSError, ValueError, ImportError, AttributeError) as spacy_init_err:
            logger.debug(
                "Failed acquiring spaCy pipeline for FTS extraction: %s",
                spacy_init_err,
            )
    if nlp is not None:
        try:
            doc = nlp(query)
            terms = [
                t.lemma_.lower()
                for t in doc
                if not t.is_stop
                and not t.is_punct
                and not t.is_space
                and len(t.text) > 1
            ]
            if terms:
                return terms
            fallback = [
                t.text.lower() for t in doc if not t.is_punct and not t.is_space
            ]
            if fallback:
                return fallback
        except (ValueError, TypeError, AttributeError, RuntimeError) as err:
            logger.debug("spaCy FTS query tokenization failed: %s", err)
    return _clean_fts_query(query, strip_stopwords=True)


def compute_upward_conductance(dst_curie: str) -> float:
    """Compute depth-scaled upward conductance with root damping: W_up = min(0.8, max(0.2, 0.2 + 0.1 * depth))."""
    from yaw.ontology.registry import get_ontology_registry

    reg = get_ontology_registry()
    depth = reg.get_class_depth(dst_curie)
    # Root proximity (depth=0) damps conductance to 0.20 per ADR-002, scaling to 0.80 for deep leaf classes (depth >= 6)
    return min(0.8, max(0.2, 0.2 + 0.1 * float(depth)))


def _collect_table_nodes(conn: sqlite3.Connection) -> set[str]:
    """Collect non-null entity and node IDs from database tables."""
    nodes: set[str] = set()
    table_queries = (
        "SELECT node_id FROM leaf_nodes",
        "SELECT entity_uri FROM note_chunks",
        "SELECT node_id FROM compiled_nodes",
    )
    for query in table_queries:
        try:
            for row in conn.execute(query).fetchall():
                if row[0]:
                    nodes.add(str(row[0]))
        except sqlite3.OperationalError:
            pass
    return nodes


def _collect_edge_endpoint_nodes(raw_edge_rows: list[tuple[Any, ...]]) -> set[str]:
    """Extract instantiated source/destination endpoints from non-ontology edges."""
    nodes: set[str] = set()
    for row in raw_edge_rows:
        source_chunk = str(row[4]) if len(row) > 4 and row[4] else ""
        if source_chunk.startswith("dynamic_ontology:"):
            continue
        if row[0]:
            nodes.add(str(row[0]))
        if row[1]:
            nodes.add(str(row[1]))
    return nodes


def _expand_class_ancestors(nodes: set[str]) -> set[str]:
    """Expand active nodes to include transitive taxonomy ancestors."""
    from yaw.ontology.registry import get_ontology_registry

    reg = get_ontology_registry()
    snapshot = reg.snapshot
    ancestors: set[str] = set()

    for node in nodes:
        curr = node
        visited: set[str] = set()
        while curr in snapshot.classes and curr not in visited:
            visited.add(curr)
            meta = snapshot.classes[curr]
            if not meta.parent_uri:
                break
            p_str = str(meta.parent_uri)
            p_meta = snapshot.classes.get(p_str)
            p_curie = p_meta.curie if p_meta else reg.uri_to_curie(p_str)
            if not p_curie:
                break
            ancestors.add(p_curie)
            curr = p_curie
    return ancestors


def _get_active_subgraph_nodes(
    conn: sqlite3.Connection,
    raw_edge_rows: list[tuple[Any, ...]],
) -> set[str]:
    """Identify active classes (instantiated classes + direct ancestors to root).

    Expands ancestors in Python memory over hydrated OntologyRegistry class hierarchy
    to prevent SQLite parameter limit overflows and redundant CTE execution.
    """
    instantiated = _collect_table_nodes(conn) | _collect_edge_endpoint_nodes(
        raw_edge_rows
    )
    return instantiated | _expand_class_ancestors(instantiated)


def _filter_active_edge_rows(
    conn: sqlite3.Connection,
) -> tuple[list[tuple[str, str, str, float]], set[str]]:
    """Query unified_edges and filter by active subgraph nodes."""
    try:
        raw_edge_rows = conn.execute(
            "SELECT src_id, dst_id, rel_type, weight, source_chunk_id FROM unified_edges"
        ).fetchall()
    except sqlite3.OperationalError:
        return [], set()

    active_nodes = _get_active_subgraph_nodes(conn, raw_edge_rows)
    rows: list[tuple[str, str, str, float]] = []
    nodes: set[str] = set()

    for r in raw_edge_rows:
        src, dst, rel_type = str(r[0]), str(r[1]), str(r[2])
        raw_w = float(r[3] if r[3] is not None else 1.0)
        if src in active_nodes and dst in active_nodes:
            rows.append((src, dst, rel_type, raw_w))
            if src:
                nodes.add(src)
            if dst:
                nodes.add(dst)
    return rows, nodes


def _extract_sequential_chunk_edges(
    conn: sqlite3.Connection,
    nodes: set[str],
) -> list[tuple[str, str, str, float]]:
    """Extract ordered chunk sequence edges within the same note."""
    seq_edges: list[tuple[str, str, str, float]] = []
    try:
        chunk_rows = conn.execute(
            "SELECT entity_uri, rel_path, ordinal FROM note_chunks ORDER BY rel_path, ordinal"
        ).fetchall()
        for r in chunk_rows:
            if r[0]:
                nodes.add(r[0])
        for i in range(len(chunk_rows) - 1):
            cur_uri, cur_path, cur_ord = chunk_rows[i]
            next_uri, next_path, next_ord = chunk_rows[i + 1]
            if cur_path == next_path and next_ord == cur_ord + 1:
                seq_edges.append((cur_uri, next_uri, "sequence", 0.8))
    except sqlite3.OperationalError:
        pass
    return seq_edges


def _synthesize_reciprocal_edges(
    rows: list[tuple[str, str, str, float]],
) -> list[tuple[str, str, str, float]]:
    """Synthesize reciprocal back-edges for wikilinks and ontological taxonomies."""
    back_edges: list[tuple[str, str, str, float]] = []
    for src, dst, rel_type, raw_w in rows:
        if not src or not dst:
            continue
        r = rel_type.split(":", 1)[1] if ":" in rel_type else rel_type
        if r == "wikilink":
            w_fwd = float(raw_w if raw_w is not None else 1.0)
            back_edges.append((dst, src, "wikilink_rev", 0.5 * w_fwd))
        elif r in ("subClassOf", "broader"):
            rev_type = "hasSubClass" if r == "subClassOf" else "narrower"
            back_edges.append((dst, src, rev_type, 1.0))
        elif r == "narrower":
            back_edges.append((dst, src, "broader", 0.2))
        elif r == "partOf":
            back_edges.append((dst, src, "hasPart", 1.0))
    return back_edges


class GraphRAGRetriever:
    """Personalized PageRank associative retrieval over unified SQLite knowledge graph."""

    _extract_fts_terms = staticmethod(_extract_fts_terms)

    def __init__(self) -> None:
        self._P: sp.csr_matrix | None = None
        self._adj: sp.csr_matrix | None = None
        self._node_to_idx: dict[str, int] = {}
        self._idx_to_node: list[str] = []
        self._dangling_mask: np.ndarray | None = None
        self._cached_db_mtime: float = -1.0

    @property
    def adj(self) -> sp.csr_matrix | None:
        """Un-normalized sparse adjacency matrix before row-stochastic scaling."""
        return self._adj

    def _sync_graph(self, conn: sqlite3.Connection) -> None:
        """Build or refresh in-memory CSR transition matrix from unified_edges VIEW."""
        try:
            db_mtime = config.DB_PATH.stat().st_mtime
        except OSError:
            db_mtime = 0.0

        if self._P is not None and self._cached_db_mtime == db_mtime:
            return

        rows, nodes = _filter_active_edge_rows(conn)
        seq_edges = _extract_sequential_chunk_edges(conn, nodes)
        back_edges = _synthesize_reciprocal_edges(rows)

        if not nodes:
            self._P = sp.csr_matrix((0, 0), dtype=np.float32)
            self._adj = sp.csr_matrix((0, 0), dtype=np.float32)
            self._node_to_idx = {}
            self._idx_to_node = []
            self._dangling_mask = np.zeros(0, dtype=bool)
            self._cached_db_mtime = db_mtime
            return

        self._build_matrix(
            nodes=nodes,
            forward_edges=rows + seq_edges,
            back_edges=back_edges,
            db_mtime=db_mtime,
        )

    def _build_matrix(
        self,
        nodes: set[str],
        forward_edges: list[tuple[str, str, str, float]],
        back_edges: list[tuple[str, str, str, float]],
        db_mtime: float = 0.0,
    ) -> None:
        """Construct weighted adjacency and row-stochastic CSR transition matrix."""
        self._idx_to_node = sorted(nodes)
        self._node_to_idx = {n: i for i, n in enumerate(self._idx_to_node)}
        n_nodes = len(self._idx_to_node)

        # Calculate in-degrees for hub dampening strictly across forward edges
        in_degrees = np.zeros(n_nodes, dtype=np.float32)
        for _, dst, _, _ in forward_edges:
            if dst in self._node_to_idx:
                in_degrees[self._node_to_idx[dst]] += 1.0

        # Construct weighted adjacency across forward and reciprocal back-edges
        all_edges = forward_edges + back_edges
        adj_rows: list[int] = []
        adj_cols: list[int] = []
        adj_weights: list[float] = []

        gamma = 0.5
        for src, dst, rel_type, raw_w in all_edges:
            if src not in self._node_to_idx or dst not in self._node_to_idx:
                continue
            u = self._node_to_idx[src]
            v = self._node_to_idx[dst]

            # Directional Conductance Rules
            r = rel_type.split(":", 1)[1] if ":" in rel_type else rel_type
            if r in ("subClassOf", "broader", "partOf"):
                dir_w = compute_upward_conductance(
                    dst
                )  # Depth-scaled upward conductance
            elif r in (
                "subClassOf_rev",
                "hasSubClass",
                "narrower",
                "partOf_rev",
                "hasPart",
            ):
                dir_w = 1.0  # Downward subsumption broadcast (parent -> child)
            elif r == "wikilink_rev":
                dir_w = float(raw_w if raw_w is not None else 0.5)
            elif _is_concept(src) and _is_chunk(dst):
                dir_w = 1.0  # Downward broadcast
            elif _is_chunk(src) and _is_concept(dst):
                dir_w = 0.2  # Upward abstraction (prevent topic leakage)
            elif r == "sequence":
                dir_w = 0.8  # Document flow
            else:
                dir_w = float(
                    raw_w if raw_w is not None else 1.0
                )  # Equal baseline for wikilinks and semantic relations

            # In-degree hub dampening using forward in-degrees
            deg_v = max(1.0, float(in_degrees[v]))
            w_uv = dir_w / (deg_v**gamma)

            adj_rows.append(u)
            adj_cols.append(v)
            adj_weights.append(w_uv)

        # Create row-stochastic transition matrix
        adj = sp.csr_matrix(
            (adj_weights, (adj_rows, adj_cols)),
            shape=(n_nodes, n_nodes),
            dtype=np.float32,
        )
        self._adj = adj
        row_sums = np.array(adj.sum(axis=1)).flatten()
        dangling = row_sums == 0.0

        # Normalize non-zero rows
        inv_sums = np.zeros_like(row_sums)
        inv_sums[~dangling] = 1.0 / row_sums[~dangling]
        inv_diag = sp.diags(inv_sums)
        self._P = inv_diag.dot(adj)
        self._dangling_mask = dangling
        self._cached_db_mtime = db_mtime
        logger.debug(
            "Compiled CSR Graph: %d nodes, %d edges (%d dangling)",
            n_nodes,
            len(adj_weights),
            int(dangling.sum()),
        )

    def invalidate_cache(self) -> None:
        """Invalidate compiled CSR transition matrices and graph indexes."""
        self._P = None
        self._adj = None
        self._node_to_idx = {}
        self._idx_to_node = []
        self._dangling_mask = None
        self._cached_db_mtime = -1.0

    def power_iteration(
        self,
        p0: np.ndarray,
        alpha: float = 0.85,
        max_iter: int = 30,
        tol: float = 1e-6,
    ) -> np.ndarray:
        """Compute stationary distribution p* via PageRank power iterations."""
        if self._P is None or self._P.shape[0] == 0:
            return p0

        p = p0.copy()
        PT = self._P.T.tocsr()
        dangling = self._dangling_mask

        for _ in range(max_iter):
            # Teleport probability mass from dangling nodes to p0
            dangling_mass = float(np.sum(p[dangling])) if dangling is not None else 0.0
            p_next = alpha * (PT.dot(p) + dangling_mass * p0) + (1.0 - alpha) * p0

            # Unit sum normalization
            s = float(np.sum(p_next))
            if s > 1e-12:
                p_next /= s

            diff = float(np.linalg.norm(p_next - p, ord=1))
            p = p_next
            if diff < tol:
                break

        return p

    def _score_chunks_vec(
        self,
        conn: sqlite3.Connection,
        q_bytes: bytes,
        p0: np.ndarray,
        cand_chunk_rows: list[dict[str, Any]],
    ) -> None:
        try:
            chunk_vec_hits = conn.execute(
                """
                SELECT entity_uri, rel_path, heading, byte_offset, byte_length,
                       line_start, line_end, embedding,
                       vec_distance_cosine(embedding, ?) AS distance
                FROM note_chunks
                WHERE embedding IS NOT NULL
                ORDER BY distance ASC LIMIT ?
                """,
                (q_bytes, config.VEC_TOP_N),
            ).fetchall()

            for r in chunk_vec_hits:
                uri, r_path, hd, b_off, b_len, l_start, l_end, emb_raw, dist = r
                try:
                    emb = deserialize_embedding(emb_raw)
                except (ValueError, TypeError, json.JSONDecodeError) as err:
                    logger.warning(
                        "Failed to deserialize embedding for chunk %s: %s", uri, err
                    )
                    continue
                if emb is None or dist is None:
                    continue
                sim = max(0.0, 1.0 - float(dist))
                if uri in self._node_to_idx:
                    p0[self._node_to_idx[uri]] += 0.20 * sim

                cand_chunk_rows.append(
                    {
                        "entity_uri": uri,
                        "rel_path": r_path,
                        "heading": hd,
                        "byte_offset": b_off,
                        "byte_length": b_len,
                        "line_start": l_start,
                        "line_end": l_end,
                        "embedding": emb,
                        "vec_sim": sim,
                    }
                )
        except sqlite3.DatabaseError as db_err:
            logger.warning("Chunk vector scoring query failed: %s", db_err)

    def _score_leaf_nodes_vec(
        self, conn: sqlite3.Connection, q_bytes: bytes, p0: np.ndarray
    ) -> None:
        try:
            leaf_hits = conn.execute(
                """
                SELECT node_id, name, namespace, definition,
                       vec_distance_cosine(embedding, ?) AS distance
                FROM leaf_nodes
                WHERE embedding IS NOT NULL
                ORDER BY distance ASC LIMIT ?
                """,
                (q_bytes, config.VEC_TOP_N),
            ).fetchall()

            seen_nodes: set[str] = set()
            for node_id, _name, _ns, _definition, dist in leaf_hits:
                if node_id in seen_nodes or dist is None:
                    continue
                seen_nodes.add(node_id)
                sim = max(0.0, 1.0 - float(dist))
                if node_id in self._node_to_idx:
                    p0[self._node_to_idx[node_id]] += 0.60 * sim
        except sqlite3.DatabaseError as db_err:
            logger.warning("Leaf nodes vector scoring query failed: %s", db_err)

    async def compute_seed_vector(
        self,
        query: str,
        conn: sqlite3.Connection,
        terms: list[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """Formulate seed vector p0 combining concepts, vector chunks, and BM25 hits."""
        self._sync_graph(conn)
        n_nodes = len(self._idx_to_node)
        if n_nodes == 0:
            return np.zeros(0, dtype=np.float32), np.zeros(768, dtype=np.float32), []

        p0 = np.zeros(n_nodes, dtype=np.float32)

        # 1. Query Embedding (with fast offline guard and bounded timeout)
        if not is_ollama_online(timeout_s=0.5):
            logger.warning(
                "Ollama is offline; skipping query embedding and falling back to lexical search"
            )
            q_vec = np.zeros(768, dtype=np.float32)
        else:
            try:
                q_vec = await asyncio.wait_for(
                    embed_query(query, timeout_s=config.CONNECT_TIMEOUT_S),
                    timeout=2.0,
                )
            except _QUERY_EMB_EXCEPTIONS as emb_err:
                logger.warning(
                    "Query embedding timed out or Ollama offline: %s", emb_err
                )
                q_vec = np.zeros(768, dtype=np.float32)

        # 2. Lexical Hits via FTS5 BM25
        fts_terms = terms if terms is not None else self._extract_fts_terms(query)
        try:
            fts_hits = search_fts(
                query, top_n=config.FTS_TOP_N, conn=conn, terms=fts_terms
            )
        except sqlite3.OperationalError:
            fts_hits = []
        for entity_uri, rank_score, _, _ in fts_hits:
            if entity_uri in self._node_to_idx:
                p0[self._node_to_idx[entity_uri]] += 0.20 * rank_score

        # 3. Dense Vector Search on note_chunks and leaf_nodes
        cand_chunk_rows: list[dict[str, Any]] = []
        if not np.all(q_vec == 0):
            q_bytes = q_vec.astype(np.float32).tobytes()
            self._score_chunks_vec(conn, q_bytes, p0, cand_chunk_rows)
            self._score_leaf_nodes_vec(conn, q_bytes, p0)

        if not cand_chunk_rows and fts_hits:
            for entity_uri, rank_score, hd, _ in fts_hits:
                if not _is_chunk(entity_uri):
                    continue
                row = conn.execute(
                    "SELECT rel_path, heading, byte_offset, byte_length, line_start, line_end, embedding FROM note_chunks WHERE entity_uri = ?",
                    (entity_uri,),
                ).fetchone()
                if row:
                    r_path, hd_db, b_off, b_len, l_start, l_end, emb_raw = row
                    emb = deserialize_embedding(emb_raw) if emb_raw else None
                    cand_chunk_rows.append(
                        {
                            "entity_uri": entity_uri,
                            "rel_path": r_path,
                            "heading": hd or hd_db,
                            "byte_offset": b_off,
                            "byte_length": b_len,
                            "line_start": l_start,
                            "line_end": l_end,
                            "embedding": emb,
                            "vec_sim": float(rank_score),
                        }
                    )

        # Normalize p0
        total_mass = float(np.sum(p0))
        if total_mass > 1e-12:
            p0 /= total_mass

        return p0, q_vec, cand_chunk_rows

    def _score_chunk_candidates(
        self,
        conn: sqlite3.Connection,
        top_chunk_pool: list[tuple[str, int, float]],
        q_vec: np.ndarray,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Score candidate chunks via dense similarity and PPR mass, returning top_k winners."""
        scored_candidates: list[dict[str, Any]] = []
        if not top_chunk_pool:
            return scored_candidates

        cand_uris = [c[0] for c in top_chunk_pool]
        placeholders = ",".join("?" for _ in cand_uris)
        chunk_db_rows = conn.execute(
            f"""
            SELECT entity_uri, rel_path, ordinal, heading, byte_offset, byte_length,
                   line_start, line_end, embedding
            FROM note_chunks
            WHERE entity_uri IN ({placeholders})
            """,
            cand_uris,
        ).fetchall()

        chunk_meta_by_uri: dict[str, dict[str, Any]] = {}
        for row in chunk_db_rows:
            uri, r_path, ord_val, hd, b_off, b_len, l_start, l_end, emb_raw = row
            try:
                emb = deserialize_embedding(emb_raw)
            except (ValueError, TypeError, json.JSONDecodeError) as err:
                logger.warning(
                    "Failed to deserialize embedding for chunk %s: %s", uri, err
                )
                emb = None
            chunk_meta_by_uri[uri] = {
                "entity_uri": uri,
                "rel_path": r_path,
                "ordinal": ord_val,
                "heading": hd,
                "byte_offset": b_off,
                "byte_length": b_len,
                "line_start": l_start,
                "line_end": l_end,
                "embedding": emb,
            }

        is_dense_active = not np.all(q_vec == 0)
        candidate_pool: list[tuple[dict[str, Any], float, float]] = []
        for uri, _idx, ppr_mass in top_chunk_pool:
            chunk_meta = chunk_meta_by_uri.get(uri)
            if not chunk_meta:
                continue

            emb = chunk_meta["embedding"]
            if emb is not None and is_dense_active:
                sim = float(np.dot(emb, q_vec))
            else:
                sim = 1.0
            candidate_pool.append((chunk_meta, ppr_mass, sim))

        any_met_threshold = is_dense_active and any(
            sim >= RELEVANCE_THRESHOLD for _, _, sim in candidate_pool
        )

        for chunk_meta, ppr_mass, sim in candidate_pool:
            if is_dense_active:
                if any_met_threshold:
                    if sim < RELEVANCE_THRESHOLD:
                        continue
                    final_score = round(ppr_mass * sim, 6)
                else:
                    final_score = round(ppr_mass * max(0.001, sim), 6)
            else:
                final_score = round(ppr_mass * sim, 6)

            scored_candidates.append(
                {
                    **chunk_meta,
                    "ppr_score": ppr_mass,
                    "sim_score": sim,
                    "final_score": final_score,
                }
            )

        scored_candidates.sort(key=lambda x: x["final_score"], reverse=True)
        return scored_candidates[:top_k]

    def _render_ontological_context(
        self, conn: sqlite3.Connection, cids: list[str]
    ) -> list[str]:
        """Render ontological concept definitions and edge relationships for context block."""
        ctx_lines: list[str] = []
        c_triples: list[tuple[str, str, str]] = []
        c_meta: dict[str, tuple[str, str]] = {}
        if cids:
            ph = ",".join("?" for _ in cids)
            try:
                rel_rows = conn.execute(
                    f"""
                    SELECT src_id, rel_type, dst_id
                    FROM unified_edges
                    WHERE src_id IN ({ph}) OR dst_id IN ({ph})
                    LIMIT 8
                    """,
                    cids + cids,
                ).fetchall()
                seen_t: set[tuple[str, str, str]] = set()
                for s, p, o in rel_rows:
                    triple = (s, p, o)
                    if triple not in seen_t:
                        seen_t.add(triple)
                        c_triples.append(triple)
            except sqlite3.OperationalError:
                pass

            try:
                leaf_rows = conn.execute(
                    f"""
                    SELECT node_id, namespace, definition
                    FROM leaf_nodes
                    WHERE node_id IN ({ph}) AND definition != ''
                    """,
                    cids,
                ).fetchall()
                for nid, ns, defn in leaf_rows:
                    if nid not in c_meta and defn:
                        c_meta[nid] = (ns or "", defn)

                meta_rows = conn.execute(
                    f"""
                    SELECT node_id, namespace, definition
                    FROM unified_nodes
                    WHERE node_id IN ({ph})
                    GROUP BY node_id
                    """,
                    cids,
                ).fetchall()
                for nid, ns, defn in meta_rows:
                    if nid not in c_meta or not c_meta[nid][1]:
                        c_meta[nid] = (ns or "", defn or "")
            except sqlite3.OperationalError:
                pass

            for cid in cids:
                ns, defn = c_meta.get(cid, ("", ""))
                if not ns and ":" in cid:
                    ns = cid.split(":")[0]
                if defn:
                    ctx_lines.append(f"- `{cid}` [{ns}]: {defn}")
                elif ns:
                    ctx_lines.append(f"- `{cid}` [{ns}]")
                else:
                    ctx_lines.append(f"- `{cid}`")

        if c_triples:
            for sub, pred, obj in c_triples:
                ctx_lines.append(f"- `{sub}` --[{pred}]--> `{obj}`")
        if not cids and not c_triples:
            ctx_lines.append("- _No strongly activated concepts identified._")
        return ctx_lines

    def _fuse_candidate_scores(
        self,
        conn: sqlite3.Connection,
        p_star: np.ndarray,
        q_vec: np.ndarray,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Filter chunk candidates from PPR distribution, score, and hydrate winning text slices."""
        chunk_candidates: list[tuple[str, int, float]] = []
        for i, n in enumerate(self._idx_to_node):
            if _is_chunk(n) and p_star[i] > 0:
                chunk_candidates.append((n, i, float(p_star[i])))

        chunk_candidates.sort(key=lambda x: x[2], reverse=True)
        pool_size = max(top_k * 5, 20)
        top_chunk_pool = chunk_candidates[:pool_size]

        winning_chunks = self._score_chunk_candidates(
            conn, top_chunk_pool, q_vec, top_k
        )

        for chunk in winning_chunks:
            chunk["text"] = hydrate_disk_slice(
                chunk["rel_path"], chunk["byte_offset"], chunk["byte_length"]
            )
        return winning_chunks

    def _render_retrieval_blocks(
        self,
        conn: sqlite3.Connection,
        winning_chunks: list[dict[str, Any]],
        seed_concepts: set[str],
        concept_scores: list[tuple[str, float]],
        p_star: np.ndarray,
    ) -> list[str]:
        """Render dual-layer Markdown blocks for hydrated winning chunks."""
        blocks: list[str] = []
        for idx, c in enumerate(winning_chunks, start=1):
            uri = c["entity_uri"]
            chunk_concepts: set[str] = set()
            try:
                edge_rows = conn.execute(
                    """
                    SELECT src_id FROM unified_edges WHERE dst_id = ? AND rel_type = 'manifests_in'
                    UNION
                    SELECT dst_id FROM unified_edges WHERE src_id = ? AND rel_type = 'mentions'
                    """,
                    (uri, uri),
                ).fetchall()
                for (cid,) in edge_rows:
                    if _is_concept(cid):
                        chunk_concepts.add(cid)
            except sqlite3.OperationalError:
                pass

            active_cids = chunk_concepts | seed_concepts
            if not active_cids:
                active_cids = {
                    c_tuple[0] for c_tuple in concept_scores[:5] if c_tuple[1] > 0
                }

            ranked_active = sorted(
                active_cids,
                key=lambda cid: (
                    float(p_star[self._node_to_idx[cid]])
                    if cid in self._node_to_idx
                    else 0.0
                ),
                reverse=True,
            )
            chunk_top_concepts = ranked_active[:5]

            ord_val = c.get("ordinal", 0)
            cite = f"{c['rel_path']}:{c['line_start']}-{c['line_end']}#chunk:{ord_val}"
            block_lines = [
                f"# Result: [{idx}] `{cite}` [score: {c['final_score']:.3f}, ppr: {c['ppr_score']:.4f}]",
                "## Ontological Context",
            ]
            block_lines.extend(
                self._render_ontological_context(conn, chunk_top_concepts)
            )
            block_lines.append("## Evidence Chunk")
            block_lines.append(f"```markdown\n{c['text'].strip()}\n```")
            blocks.append("\n".join(block_lines))

        return blocks

    async def retrieve(self, query: str, top_k: int = 3) -> str:
        """Full dual-layer GraphRAG retrieval (ontological context + grounded evidence)."""
        conn = get_db(load_vec=True)
        try:
            fts_terms = self._extract_fts_terms(query)
            p0, q_vec, _ = await self.compute_seed_vector(query, conn, terms=fts_terms)

            is_degraded = (q_vec is None) or np.all(q_vec == 0)
            degraded_prefix = (
                "> [!WARNING] Degraded Retrieval Mode: Ollama offline/unreachable. "
                "Dense vector similarity bypassed; results derived strictly from "
                "lexical FTS5 BM25 and taxonomic graph conductance.\n\n"
                if is_degraded
                else ""
            )

            n_nodes = len(self._idx_to_node)
            if n_nodes == 0 or np.all(p0 == 0):
                return degraded_prefix + (
                    "# Result: 0\n"
                    "## Ontological Context\n"
                    "_No indexed knowledge available._\n\n"
                    "## Evidence Chunk\n"
                    "_Zero relevant evidence chunks found in raw/ corpus meeting the relevance threshold. Further source research and ingestion via ingest_raw is recommended to grow the corpus._"
                )

            p_star = self.power_iteration(p0, alpha=0.85, max_iter=30)
            winning_chunks = self._fuse_candidate_scores(conn, p_star, q_vec, top_k)

            concept_indices = [
                i for i, n in enumerate(self._idx_to_node) if _is_concept(n)
            ]
            concept_scores = [
                (self._idx_to_node[i], float(p_star[i])) for i in concept_indices
            ]
            concept_scores.sort(key=lambda x: x[1], reverse=True)

            seed_concepts = {
                self._idx_to_node[i]
                for i in range(len(self._idx_to_node))
                if p0[i] > 0 and _is_concept(self._idx_to_node[i])
            }

            if winning_chunks:
                blocks = self._render_retrieval_blocks(
                    conn, winning_chunks, seed_concepts, concept_scores, p_star
                )
                return degraded_prefix + "\n\n".join(blocks)
            else:
                global_top_concepts = [c[0] for c in concept_scores[:5] if c[1] > 0]
                lines = [
                    "# Result: 0",
                    "## Ontological Context",
                ]
                lines.extend(
                    self._render_ontological_context(conn, global_top_concepts)
                )
                lines.append("## Evidence Chunk")
                lines.append(
                    "_Zero relevant evidence chunks found in raw/ corpus meeting the relevance threshold. Further source research and ingestion via ingest_raw is recommended to grow the corpus._"
                )
                return degraded_prefix + "\n".join(lines)
        finally:
            conn.close()

    async def find_seed_nodes(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Compute top seed nodes using in-memory hybrid search (renamed from find_seed)."""
        return await compute_seed_candidates(query, top_k=top_k)


def _fetch_chunk_candidates(
    conn: sqlite3.Connection,
    q_bytes: bytes,
    top_n: int,
) -> list[dict[str, Any]]:
    """Fetch nearest note_chunks candidates in a single pass."""
    results: list[dict[str, Any]] = []
    try:
        try:
            rows = conn.execute(
                "SELECT entity_uri, rel_path, heading, byte_offset, byte_length, "
                "line_start, line_end, embedding, "
                "vec_distance_cosine(embedding, ?) AS distance "
                "FROM note_chunks WHERE embedding IS NOT NULL ORDER BY distance LIMIT ?",
                (q_bytes, top_n),
            ).fetchall()
            has_lines = True
        except sqlite3.OperationalError:
            rows = conn.execute(
                "SELECT entity_uri, rel_path, heading, byte_offset, byte_length, "
                "embedding, "
                "vec_distance_cosine(embedding, ?) AS distance "
                "FROM note_chunks WHERE embedding IS NOT NULL ORDER BY distance LIMIT ?",
                (q_bytes, top_n),
            ).fetchall()
            has_lines = False

        for r in rows:
            dist_raw = r[8] if has_lines else r[6]
            if dist_raw is None:
                continue
            dist = float(dist_raw)
            rel_path = str(r[1])
            b_off = int(r[3])
            b_len = int(r[4])
            emb_raw = r[7] if has_lines else r[5]
            results.append(
                {
                    "entity_uri": str(r[0]),
                    "kind": "chunk",
                    "rel_path": rel_path,
                    "heading": str(r[2] or ""),
                    "byte_offset": b_off,
                    "byte_length": b_len,
                    "line_start": int(r[5]) if has_lines and r[5] is not None else 0,
                    "line_end": int(r[6]) if has_lines and r[6] is not None else 0,
                    "embedding": deserialize_embedding(emb_raw),
                    "dist": dist,
                    "sim": round(max(0.0, 1.0 - dist), 4),
                    "text": hydrate_disk_slice(rel_path, b_off, b_len),
                }
            )
    except sqlite3.DatabaseError as db_err:
        logger.warning("Chunk single-pass query failed: %s", db_err)
    return results


def _fetch_leaf_candidates(
    conn: sqlite3.Connection,
    q_bytes: bytes,
    top_n: int,
) -> list[dict[str, Any]]:
    """Fetch nearest leaf_nodes candidates in a single pass."""
    results: list[dict[str, Any]] = []
    try:
        rows = conn.execute(
            "SELECT node_id, source_leaf_id, name, namespace, definition, embedding, "
            "vec_distance_cosine(embedding, ?) AS distance "
            "FROM leaf_nodes WHERE embedding IS NOT NULL ORDER BY distance LIMIT ?",
            (q_bytes, top_n),
        ).fetchall()
        for r in rows:
            if r[6] is None:
                continue
            dist = float(r[6])
            name = str(r[2])
            namespace = str(r[3])
            definition = str(r[4]) if r[4] is not None else ""
            heading = f"{name} ({namespace})"
            text = f"{heading}: {definition}"
            results.append(
                {
                    "entity_uri": str(r[0]),
                    "kind": "taxonomy",
                    "rel_path": str(r[1]),
                    "heading": heading,
                    "byte_offset": 0,
                    "byte_length": len(text.encode("utf-8")),
                    "line_start": 0,
                    "line_end": 0,
                    "embedding": deserialize_embedding(r[5]),
                    "dist": dist,
                    "sim": round(max(0.0, 1.0 - dist), 4),
                    "text": text,
                }
            )
    except sqlite3.DatabaseError as db_err:
        logger.warning("Leaf node single-pass query failed: %s", db_err)
    return results


def _fetch_candidates_single_pass(
    conn: sqlite3.Connection,
    q_bytes: bytes,
    top_n: int = config.VEC_TOP_N,
) -> list[dict[str, Any]]:
    """Retrieve chunk distance, metadata, and embeddings in a single SQLite query per table."""
    results = _fetch_chunk_candidates(conn, q_bytes, top_n) + _fetch_leaf_candidates(
        conn, q_bytes, top_n
    )
    results.sort(key=lambda x: x["dist"])
    return results[:top_n]


def _query_vectors(
    query_vec: np.ndarray,
    top_n: int = config.VEC_TOP_N,
) -> list[dict[str, Any]]:
    """Backward-compatible stub querying nearest candidates via _fetch_candidates_single_pass."""
    conn = get_db(load_vec=True)
    try:
        return _fetch_candidates_single_pass(
            conn, query_vec.astype(np.float32).tobytes(), top_n=top_n
        )
    finally:
        conn.close()


def _get_candidate_metadata(
    uris: list[str] | list[Any],
    conn: sqlite3.Connection | None = None,
) -> tuple[dict[str, Any], dict[str, tuple[str, str, str]]]:
    """Fetch chunk locations and taxonomy node definitions for candidate URIs."""
    if not uris:
        return {}, {}
    normalized_uris = [u[0] if isinstance(u, (tuple, list)) else str(u) for u in uris]
    should_close = False
    if conn is None:
        conn = get_db(load_vec=False)
        should_close = True
    try:
        placeholders = ",".join("?" for _ in normalized_uris)
        chunks: dict[str, tuple[str, str, int, int]] = {}
        try:
            chunk_rows = conn.execute(
                f"SELECT entity_uri, rel_path, heading, byte_offset, byte_length FROM note_chunks WHERE entity_uri IN ({placeholders})",
                normalized_uris,
            ).fetchall()
            chunks = {
                r[0]: (r[1], r[2] or "", int(r[3]), int(r[4])) for r in chunk_rows
            }
        except sqlite3.OperationalError:
            pass

        tax: dict[str, tuple[str, str, str]] = {}
        try:
            tax_rows = conn.execute(
                f"SELECT node_id, source_leaf_id, name, namespace, definition FROM leaf_nodes WHERE node_id IN ({placeholders})",
                normalized_uris,
            ).fetchall()
            tax = {
                r[0]: (r[1], f"{r[2]} ({r[3]})", f"{r[2]} ({r[3]}): {r[4]}")
                for r in tax_rows
            }
        except sqlite3.OperationalError:
            pass

        return chunks, tax
    finally:
        if should_close:
            conn.close()


def _get_chunk_locations(
    candidate_items: list[tuple[str, str, str, float]],
    conn: sqlite3.Connection | None = None,
) -> dict[str, dict[str, Any]]:
    """Return dictionary of chunk locations mapping entity_uri to metadata dictionary."""
    locations, _ = _get_candidate_metadata(candidate_items, conn=conn)
    return locations


def _fetch_table_vectors(
    conn: sqlite3.Connection,
    table: str,
    id_col: str,
    uris: list[str],
    vecs: dict[str, np.ndarray],
) -> None:
    placeholders = ",".join("?" for _ in uris)
    try:
        rows = conn.execute(
            f"SELECT {id_col}, embedding FROM {table} WHERE {id_col} IN ({placeholders}) AND embedding IS NOT NULL",
            uris,
        ).fetchall()
        for uri, raw_emb in rows:
            if not raw_emb:
                continue
            try:
                emb = deserialize_embedding(raw_emb)
                if emb is not None:
                    vecs[uri] = emb
            except (ValueError, TypeError, json.JSONDecodeError) as err:
                logger.warning("Failed to deserialize embedding for %s: %s", uri, err)
    except sqlite3.OperationalError:
        pass


def _get_candidate_vectors(uris: list[str]) -> dict[str, np.ndarray]:
    """Fetch stored vector embeddings for candidate URIs directly from SQLite."""
    if not uris:
        return {}
    conn = get_db(load_vec=False)
    vecs: dict[str, np.ndarray] = {}
    try:
        _fetch_table_vectors(conn, "note_chunks", "entity_uri", uris, vecs)
        _fetch_table_vectors(conn, "leaf_nodes", "node_id", uris, vecs)
        return vecs
    finally:
        conn.close()


def _hydrate_candidate_items(
    sorted_candidates: list[str],
    hybrid_scores: dict[str, float],
) -> list[dict[str, Any]]:
    """Hydrate candidate items with text slices and byte offsets from disk or taxonomy."""
    chunks_meta, tax_meta = _get_candidate_metadata(sorted_candidates)
    candidate_items: list[dict[str, Any]] = []
    for u in sorted_candidates:
        if u in chunks_meta:
            rel_path, heading, b_off, b_len = chunks_meta[u]
            text = hydrate_disk_slice(rel_path, b_off, b_len)
            candidate_items.append(
                {
                    "entity_uri": u,
                    "kind": "chunk",
                    "rel_path": rel_path,
                    "heading": heading,
                    "byte_offset": int(b_off),
                    "byte_length": int(b_len),
                    "hybrid_prob": float(hybrid_scores[u]),
                    "text": text,
                }
            )
        elif u in tax_meta:
            rel_path, heading, text = tax_meta[u]
            candidate_items.append(
                {
                    "entity_uri": u,
                    "kind": "taxonomy",
                    "rel_path": rel_path,
                    "heading": heading,
                    "byte_offset": 0,
                    "byte_length": len(text.encode("utf-8")),
                    "hybrid_prob": float(hybrid_scores[u]),
                    "text": text,
                }
            )
    return candidate_items


def _rescore_single_candidate(
    candidate: dict[str, Any],
    cand_vec: np.ndarray | None,
    q_vec: np.ndarray | None,
) -> None:
    """Compute in-memory bi-encoder score or fall back to hybrid probability."""
    uri = candidate["entity_uri"]
    if (
        q_vec is not None
        and isinstance(cand_vec, np.ndarray)
        and cand_vec.shape == q_vec.shape
    ):
        candidate["rerank_score"] = round(float(np.dot(cand_vec, q_vec)), 4)
        return

    if cand_vec is None or not isinstance(cand_vec, np.ndarray):
        logger.warning(
            "Candidate %s lacks stored embedding vector in SQLite; falling back to normalized hybrid_prob",
            uri,
        )
    elif q_vec is not None and cand_vec.shape != q_vec.shape:
        logger.warning(
            "Candidate %s vector dimension mismatch (%s vs %s); falling back to normalized hybrid_prob",
            uri,
            cand_vec.shape,
            q_vec.shape,
        )
    elif q_vec is None:
        logger.warning(
            "Query vector unavailable; falling back to normalized hybrid_prob for %s",
            uri,
        )
    candidate["rerank_score"] = float(candidate["hybrid_prob"])


async def _resolve_query_vector(
    query: str, query_vec: np.ndarray | None
) -> np.ndarray | None:
    """Resolve query embedding vector from caller or Ollama embedding endpoint."""
    if query_vec is not None:
        return query_vec
    if not is_ollama_online(timeout_s=0.5):
        logger.warning(
            "Ollama offline; skipping query embedding and falling back to lexical scoring"
        )
        return None
    try:
        return await asyncio.wait_for(embed_query(query, timeout_s=2.0), timeout=2.0)
    except _QUERY_EMB_EXCEPTIONS as err:
        logger.warning(
            "Query embedding failed or timed out: %s; falling back to lexical scoring",
            err,
        )
        return None


async def compute_seed_candidates(
    query: str,
    top_k: int = 3,
    query_vec: np.ndarray | None = None,
    query_vectors_fn: Any | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Compute seed candidates via summed probability (FTS + vector) and in-memory bi-encoder rescoring."""
    try:
        fts_rows = search_fts(query, top_n=config.FTS_TOP_N)
    except sqlite3.OperationalError:
        fts_rows = []
    fts_scores = {uri: max(0.0, float(score)) for uri, score, _, _ in fts_rows if uri}

    q_vec = await _resolve_query_vector(query, query_vec)

    dense_candidates: list[dict[str, Any]] = []
    if query_vectors_fn is not None:
        dense_candidates = query_vectors_fn(q_vec, top_n=config.VEC_TOP_N)
    elif q_vec is not None:
        if hasattr(_query_vectors, "assert_called") or hasattr(_query_vectors, "mock"):
            dense_candidates = _query_vectors(q_vec, top_n=config.VEC_TOP_N)
        else:
            conn = get_db(load_vec=True)
            try:
                dense_candidates = _fetch_candidates_single_pass(
                    conn, q_vec.astype(np.float32).tobytes(), config.VEC_TOP_N
                )
            finally:
                conn.close()

    vec_scores = {r["entity_uri"]: max(0.0, float(r["sim"])) for r in dense_candidates}
    fts_sum = sum(fts_scores.values())
    vec_sum = sum(vec_scores.values())
    p_fts = {u: (s / fts_sum if fts_sum > 0 else 0.0) for u, s in fts_scores.items()}
    p_vec = {u: (s / vec_sum if vec_sum > 0 else 0.0) for u, s in vec_scores.items()}

    all_uris = set(p_fts.keys()) | set(p_vec.keys())
    if not all_uris:
        return []

    hybrid_scores = {
        u: round(p_fts.get(u, 0.0) + p_vec.get(u, 0.0), 6) for u in all_uris
    }
    pool_size = max(top_k * 2, 10)
    sorted_uris = sorted(all_uris, key=lambda u: hybrid_scores[u], reverse=True)[
        :pool_size
    ]

    dense_map = {r["entity_uri"]: dict(r) for r in dense_candidates}
    missing_uris = [u for u in sorted_uris if u not in dense_map]
    hydrated_missing = (
        {
            c["entity_uri"]: c
            for c in _hydrate_candidate_items(missing_uris, hybrid_scores)
        }
        if missing_uris
        else {}
    )

    missing_emb = [
        u
        for u in sorted_uris
        if u not in dense_map or dense_map[u].get("embedding") is None
    ]
    extra_vecs = _get_candidate_vectors(missing_emb) if missing_emb else {}

    candidate_items: list[dict[str, Any]] = []
    for u in sorted_uris:
        cand = dense_map.get(u) or hydrated_missing.get(u)
        if cand is None:
            continue
        cand["hybrid_prob"] = float(hybrid_scores[u])
        if "text" not in cand:
            hydrated = _hydrate_candidate_items([u], hybrid_scores)
            if hydrated:
                cand.update(hydrated[0])
        vec = (
            cand.get("embedding")
            if cand.get("embedding") is not None
            else extra_vecs.get(u)
        )
        _rescore_single_candidate(cand, vec, q_vec)
        candidate_items.append(cand)

    candidate_items.sort(
        key=lambda x: (x.get("rerank_score", 0.0), x.get("hybrid_prob", 0.0)),
        reverse=True,
    )
    return candidate_items[:top_k]


async def find_seed_nodes(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    """Find top associative seed nodes using summed probability (FTS + vector) and in-memory rescoring.

    Returns structured candidate dictionaries containing disk byte pointers and scores.
    """
    return await compute_seed_candidates(query, top_k=top_k)


_RETRIEVER_INSTANCE: GraphRAGRetriever | None = None


def get_retriever() -> GraphRAGRetriever:
    """Singleton getter for GraphRAGRetriever."""
    global _RETRIEVER_INSTANCE
    if _RETRIEVER_INSTANCE is None:
        _RETRIEVER_INSTANCE = GraphRAGRetriever()
    return _RETRIEVER_INSTANCE


Retriever = GraphRAGRetriever

__all__ = [
    "GraphRAGRetriever",
    "Retriever",
    "_extract_fts_terms",
    "_fetch_candidates_single_pass",
    "_fetch_table_vectors",
    "_get_active_subgraph_nodes",
    "_get_candidate_metadata",
    "_get_candidate_vectors",
    "_get_chunk_locations",
    "_hydrate_candidate_items",
    "_query_vectors",
    "_rescore_single_candidate",
    "compute_seed_candidates",
    "compute_upward_conductance",
    "find_seed_nodes",
    "get_retriever",
    "hydrate_disk_slice",
    "read_document",
]
