"""OWL 2 RL and SKOS taxonomic subsumption inference module for YAW.

Provides deterministic transitive closure computation, subsumption DAG cycle
validation, reversible identity/distinction mapping, and SKOS concept scheme projection.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import rdflib
import scipy.sparse as sp
from rdflib import RDF, Literal, URIRef
from rdflib.namespace import SKOS

from yaw.storage.rdf import (
    YAW_PREFIX,
    curie_to_uri,
    project_knowledge_graph,
)

__all__ = [
    "TaxonomicCycleError",
    "build_skos_concept_scheme",
    "compute_transitive_closure_csr",
    "get_broader_closure",
    "get_different_from_triples",
    "get_ontology_graph",
    "get_same_as_triples",
    "get_subclass_closure",
    "validate_subsumption_dag",
]


class TaxonomicCycleError(ValueError):
    """Raised when a directed cycle is detected in a taxonomic subsumption hierarchy."""


def validate_subsumption_dag(
    edges: Iterable[tuple[str, str]],
    raise_on_cycle: bool = True,
) -> list[list[str]]:
    """Validate that directed subsumption edges form an acyclic directed graph (DAG).

    Uses a 3-color depth-first search (WHITE=0, GRAY=1, BLACK=2) to detect directed cycles.
    If a cycle is detected and raise_on_cycle is True, raises TaxonomicCycleError.
    Otherwise, returns all detected cycle paths as lists of node identifiers.
    """
    adj: dict[str, list[str]] = defaultdict(list)
    nodes: set[str] = set()
    for u, v in edges:
        adj[u].append(v)
        nodes.add(u)
        nodes.add(v)

    color: dict[str, int] = {node: 0 for node in nodes}
    path: list[str] = []
    cycles: list[list[str]] = []

    def dfs(u: str) -> None:
        color[u] = 1  # GRAY (visiting)
        path.append(u)

        for v in adj.get(u, []):
            if color[v] == 1:
                # Directed cycle detected
                try:
                    idx = path.index(v)
                    cycle_path = path[idx:] + [v]
                except ValueError:
                    cycle_path = [u, v]
                cycles.append(cycle_path)
                if raise_on_cycle:
                    cycle_repr = " -> ".join(cycle_path)
                    raise TaxonomicCycleError(
                        f"Taxonomic cycle detected involving nodes: {cycle_repr}"
                    )
            elif color[v] == 0:
                dfs(v)

        path.pop()
        color[u] = 2  # BLACK (visited)

    for node in sorted(nodes):
        if color[node] == 0:
            dfs(node)

    return cycles


def get_subclass_closure(
    node_id: str,
    conn: sqlite3.Connection,
    include_self: bool = False,
) -> set[str]:
    """Compute transitive subclass closure using SQLite recursive CTE with cycle protection.

    Traverses edges where rel_type is bare 'subClassOf', 'rdfs:subClassOf', or 'predicates:subClassOf'.
    Guards against cyclic structures using path-based inclusion checks.
    """
    query = """
    WITH RECURSIVE closure(dst_id, path) AS (
        SELECT dst_id, ',' || ? || ',' || dst_id || ','
        FROM unified_edges
        WHERE src_id = ?
          AND rel_type IN ('subClassOf', 'rdfs:subClassOf', 'predicates:subClassOf')
        UNION
        SELECT e.dst_id, c.path || e.dst_id || ','
        FROM unified_edges e
        JOIN closure c ON e.src_id = c.dst_id
        WHERE e.rel_type IN ('subClassOf', 'rdfs:subClassOf', 'predicates:subClassOf')
          AND instr(c.path, ',' || e.dst_id || ',') = 0
    )
    SELECT DISTINCT dst_id FROM closure
    """
    cursor = conn.execute(query, (node_id, node_id))
    results = {str(row[0]) for row in cursor.fetchall()}
    if include_self:
        results.add(node_id)
    return results


def get_broader_closure(
    node_id: str,
    conn: sqlite3.Connection,
    include_self: bool = False,
) -> set[str]:
    """Compute transitive SKOS broader closure using SQLite recursive CTE with cycle protection.

    Traverses edges where rel_type is bare 'broader', 'skos:broader', or 'predicates:broader'.
    Guards against cyclic structures using path-based inclusion checks.
    """
    query = """
    WITH RECURSIVE closure(dst_id, path) AS (
        SELECT dst_id, ',' || ? || ',' || dst_id || ','
        FROM unified_edges
        WHERE src_id = ?
          AND rel_type IN ('broader', 'skos:broader', 'predicates:broader')
        UNION
        SELECT e.dst_id, c.path || e.dst_id || ','
        FROM unified_edges e
        JOIN closure c ON e.src_id = c.dst_id
        WHERE e.rel_type IN ('broader', 'skos:broader', 'predicates:broader')
          AND instr(c.path, ',' || e.dst_id || ',') = 0
    )
    SELECT DISTINCT dst_id FROM closure
    """
    cursor = conn.execute(query, (node_id, node_id))
    results = {str(row[0]) for row in cursor.fetchall()}
    if include_self:
        results.add(node_id)
    return results


def compute_transitive_closure_csr(
    adj_matrix: sp.csr_matrix,
    max_depth: int | None = None,
) -> sp.csr_matrix:
    """Compute boolean transitive closure of a CSR adjacency matrix.

    Uses repeated matrix multiplication / squaring to find all reachable pairs (i, j)
    where there is a directed path from i to j of length >= 1.
    Terminates when no new paths are discovered or when max_depth is reached.
    """
    n = adj_matrix.shape[0]
    if n <= 1:
        if adj_matrix.shape == (0, 0):
            return sp.csr_matrix((0, 0), dtype=np.float32)
        val = 1.0 if adj_matrix[0, 0] > 0 else 0.0
        return sp.csr_matrix([[val]], dtype=np.float32)

    reach = (adj_matrix > 0).astype(np.float32).tocsr()

    if max_depth is not None:
        if max_depth <= 0:
            return sp.csr_matrix(adj_matrix.shape, dtype=np.float32)
        if max_depth == 1:
            return reach

        frontier = reach.copy()
        for _ in range(2, max_depth + 1):
            next_hop = frontier @ reach
            next_hop = (next_hop > 0).astype(np.float32).tocsr()
            new_reach = reach + next_hop
            new_reach = (new_reach > 0).astype(np.float32).tocsr()
            if new_reach.nnz == reach.nnz:
                break
            reach = new_reach
            frontier = next_hop
        return reach

    # Repeated squaring when max_depth is None: M_{t+1} = M_t + M_t^2
    while True:
        m_sq = reach @ reach
        new_reach = reach + m_sq
        new_reach = (new_reach > 0).astype(np.float32).tocsr()
        if new_reach.nnz == reach.nnz:
            break
        reach = new_reach

    return reach


def get_same_as_triples(conn: sqlite3.Connection) -> list[tuple[str, str, str]]:
    """Generate bidirectional owl:sameAs triples for active aliases.

    Inactive or contested aliases are excluded.
    Returns list of (subject_curie, 'owl:sameAs', object_curie) triples.
    """
    cursor = conn.execute(
        "SELECT alias_id, canonical_id FROM aliases WHERE status = 'active'"
    )
    rows = cursor.fetchall()
    triples: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    for alias_id, canonical_id in rows:
        t1 = (str(alias_id), "owl:sameAs", str(canonical_id))
        t2 = (str(canonical_id), "owl:sameAs", str(alias_id))
        if t1 not in seen:
            seen.add(t1)
            triples.append(t1)
        if t2 not in seen:
            seen.add(t2)
            triples.append(t2)

    return triples


def get_different_from_triples(
    conn: sqlite3.Connection,
) -> list[tuple[str, str, str]]:
    """Generate bidirectional owl:differentFrom triples for distinct canon verdicts.

    Non-distinct (match) verdicts are excluded.
    Returns list of (subject_curie, 'owl:differentFrom', object_curie) triples.
    """
    cursor = conn.execute(
        "SELECT surface_id, candidate_id FROM canon_verdicts WHERE verdict = 'distinct'"
    )
    rows = cursor.fetchall()
    triples: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    for surface_id, candidate_id in rows:
        t1 = (str(surface_id), "owl:differentFrom", str(candidate_id))
        t2 = (str(candidate_id), "owl:differentFrom", str(surface_id))
        if t1 not in seen:
            seen.add(t1)
            triples.append(t1)
        if t2 not in seen:
            seen.add(t2)
            triples.append(t2)

    return triples


def _project_skos_aliases(g: rdflib.Graph, conn: sqlite3.Connection) -> None:
    alias_rows = conn.execute(
        """
        SELECT a.alias_id, a.canonical_id, n.name
        FROM aliases a
        LEFT JOIN unified_nodes n ON a.alias_id = n.node_id
        WHERE a.status = 'active'
        """
    ).fetchall()
    for alias_id, canonical_id, node_name in alias_rows:
        canonical_uri = curie_to_uri(str(canonical_id))
        raw_alias = str(alias_id)
        slug = raw_alias.split(":", 1)[1] if ":" in raw_alias else raw_alias
        g.add((canonical_uri, SKOS.altLabel, Literal(slug)))
        if node_name is not None and str(node_name).strip():
            name_str = str(node_name).strip()
            if name_str != slug:
                g.add((canonical_uri, SKOS.altLabel, Literal(name_str)))


def build_skos_concept_scheme(conn: sqlite3.Connection) -> rdflib.Graph:
    """Project SQLite unified_nodes, aliases, and unified_edges into an rdflib SKOS Graph."""
    vault_scheme_uri = URIRef(f"{YAW_PREFIX}VaultScheme")
    g = rdflib.Graph()
    g.add((vault_scheme_uri, RDF.type, SKOS.ConceptScheme))

    project_knowledge_graph(
        conn=conn,
        target_graph=g,
        include_skos=True,
        vault_scheme_uri=vault_scheme_uri,
        named_quads=False,
    )
    _project_skos_aliases(g, conn)
    return g


def get_ontology_graph() -> rdflib.Graph:
    """Parse and return the static Turtle ontology graph from src/yaw/ontology.ttl."""
    ttl_path = Path(__file__).resolve().parent.parent / "ontology.ttl"
    if not ttl_path.is_file():
        ttl_path = Path("src/yaw/ontology.ttl")
    g = rdflib.Graph()
    g.parse(ttl_path, format="turtle")
    return g
