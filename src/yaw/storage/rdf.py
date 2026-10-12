"""W3C RDF and Named Quad projection substrate for YAW.

Projects SQLite unified_nodes and unified_edges views into an in-process rdflib ConjunctiveGraph.
Provides serialization to Turtle, JSON-LD, N-Triples, and N-Quads with source chunk provenance.
"""

from __future__ import annotations

import re
import sqlite3
import warnings
from typing import Any

import rdflib
from rdflib import ConjunctiveGraph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SH, SKOS, XSD

from yaw.storage.schema import verify_schema_integrity

__all__ = [
    "FORMAT_MAP",
    "ROOT_NAMESPACES",
    "W3C_NAMESPACES",
    "YAW",
    "YAW_PREFIX",
    "bind_namespaces",
    "chunk_to_context_uri",
    "curie_to_uri",
    "export_graph",
    "project_knowledge_graph",
    "to_rdflib_graph",
]

# Universal root namespace and base URI
YAW_PREFIX: str = "http://yaw.local/"
YAW: Namespace = Namespace(YAW_PREFIX)

ROOT_NAMESPACES: dict[str, Namespace] = {
    "concepts": Namespace(f"{YAW_PREFIX}concepts/"),
    "entities": Namespace(f"{YAW_PREFIX}entities/"),
    "predicates": Namespace(f"{YAW_PREFIX}predicates/"),
    "sources": Namespace(f"{YAW_PREFIX}sources/"),
    "contexts": Namespace(f"{YAW_PREFIX}contexts/"),
    "rules": Namespace(f"{YAW_PREFIX}rules/"),
    "queries": Namespace(f"{YAW_PREFIX}queries/"),
    "claims": Namespace(f"{YAW_PREFIX}claims/"),
    "invariants": Namespace(f"{YAW_PREFIX}invariants/"),
    "decisions": Namespace(f"{YAW_PREFIX}decisions/"),
    "processes": Namespace(f"{YAW_PREFIX}processes/"),
}

W3C_NAMESPACES: dict[str, Any] = {
    "yaw": YAW,
    "rdf": RDF,
    "rdfs": RDFS,
    "owl": OWL,
    "sh": SH,
    "skos": SKOS,
    "xsd": XSD,
}

FORMAT_MAP: dict[str, str] = {
    "turtle": "turtle",
    "ttl": "turtle",
    "json-ld": "json-ld",
    "jsonld": "json-ld",
    "nt": "nt",
    "ntriples": "nt",
    "n-triples": "nt",
    "nquads": "nquads",
    "nq": "nquads",
    "n-quads": "nquads",
}


def curie_to_uri(curie: str | None) -> URIRef:
    """Convert a CURIE, relative path, or URI string to a canonical W3C URIRef."""
    if not curie or not curie.strip():
        return URIRef(f"{YAW_PREFIX}unknown")

    raw = curie.strip()
    if raw.startswith(("http://", "https://", "urn:")):
        return URIRef(raw)

    if raw.startswith("yaw:"):
        raw = raw[4:].strip()

    if ":" in raw:
        prefix, suffix = raw.split(":", 1)
        if prefix in W3C_NAMESPACES:
            return URIRef(W3C_NAMESPACES[prefix][suffix])

    if "/" in raw or raw.endswith(".md") or ".md#" in raw:
        path = raw.lstrip("/")
        return URIRef(
            f"{YAW_PREFIX}{path}"
            if path.startswith("sources/")
            else f"{YAW_PREFIX}sources/{path}"
        )

    if ":" in raw:
        ns, slug = raw.split(":", 1)
        slug_clean = re.sub(r"\s+", "-", slug.strip())
        return URIRef(f"{YAW_PREFIX}{ns}/{slug_clean}")

    slug_clean = re.sub(r"\s+", "-", raw)
    return URIRef(f"{YAW_PREFIX}concepts/{slug_clean}")


def chunk_to_context_uri(chunk_id: str | None) -> URIRef | None:
    """Convert a source chunk identifier or rule into a canonical named graph context URIRef."""
    if not chunk_id or not chunk_id.strip():
        return None

    raw = chunk_id.strip()
    if raw.startswith(("http://", "https://")):
        return URIRef(raw)

    if raw.startswith("compiled:"):
        return URIRef(f"{YAW_PREFIX}contexts/{raw}")

    if raw.startswith("yaw:sources/"):
        return URIRef(f"{YAW_PREFIX}sources/{raw[12:].lstrip('/')}")

    if raw.startswith("yaw:"):
        return URIRef(f"{YAW_PREFIX}{raw[4:].lstrip('/')}")

    clean = raw.lstrip("/")
    if clean.startswith("sources/"):
        return URIRef(f"{YAW_PREFIX}{clean}")
    return URIRef(f"{YAW_PREFIX}sources/{clean}")


def bind_namespaces(graph: rdflib.Graph | rdflib.ConjunctiveGraph) -> None:
    """Bind authoritative W3C and domain root namespaces to graph."""
    graph.bind("yaw", YAW)
    graph.bind("rdf", RDF)
    graph.bind("rdfs", RDFS)
    graph.bind("owl", OWL)
    graph.bind("sh", SH)
    graph.bind("skos", SKOS)
    graph.bind("xsd", XSD)
    for prefix, ns in ROOT_NAMESPACES.items():
        graph.bind(prefix, ns)


_bind_namespaces = bind_namespaces


def _get_root_class(ns_str: str) -> URIRef:
    """Resolve Root class URIRef for a namespace prefix."""
    if ns_str in ("chunk", "sources"):
        return ROOT_NAMESPACES["sources"]["Root"]
    if ns_str in ROOT_NAMESPACES:
        return ROOT_NAMESPACES[ns_str]["Root"]
    return Namespace(f"{YAW_PREFIX}{ns_str}/")["Root"]


def _project_nodes(
    g: rdflib.Graph,
    conn: sqlite3.Connection,
    include_skos: bool,
    vault_scheme_uri: URIRef | None,
) -> None:
    """Project unified_nodes rows into RDF triples."""
    node_rows = conn.execute(
        "SELECT node_id, name, namespace, definition FROM unified_nodes"
    ).fetchall()
    seen_nodes: set[URIRef] = set()

    for raw_nid, raw_name, raw_ns, raw_def in node_rows:
        node_uri = curie_to_uri(str(raw_nid))
        if node_uri in seen_nodes:
            continue
        seen_nodes.add(node_uri)

        if include_skos:
            g.add((node_uri, RDF.type, SKOS.Concept))
            if vault_scheme_uri is not None:
                g.add((node_uri, SKOS.inScheme, vault_scheme_uri))

        ns_str = str(raw_ns).strip() if raw_ns else "concepts"
        g.add((node_uri, RDF.type, _get_root_class(ns_str)))

        if raw_name is not None and str(raw_name).strip():
            name_lit = Literal(str(raw_name).strip())
            g.add((node_uri, RDFS.label, name_lit))
            g.add((node_uri, SKOS.prefLabel, name_lit))

        if raw_def is not None and str(raw_def).strip():
            g.add((node_uri, SKOS.definition, Literal(str(raw_def).strip())))


def _is_cross_scheme_endpoint(endpoint: str | None) -> bool:
    """Return True if endpoint CURIE or URI belongs to an external ontology scheme."""
    if not endpoint or not endpoint.strip():
        return False
    raw = endpoint.strip()
    if raw.startswith(("http://", "https://", "urn:")):
        return not raw.startswith(YAW_PREFIX)
    if ":" in raw:
        prefix = raw.split(":", 1)[0].strip()
        return prefix not in ("yaw", *ROOT_NAMESPACES.keys())
    return False


def _resolve_edge_predicate(
    rel: str,
    rel_norm: str,
    src_id: str | None = None,
    dst_id: str | None = None,
) -> tuple[URIRef, URIRef | None]:
    """Resolve predicate URI and optional reciprocal predicate URI."""
    if rel_norm == "subClassOf":
        return RDFS.subClassOf, None
    if rel_norm == "broader":
        return SKOS.broader, SKOS.narrower
    if rel_norm == "narrower":
        return SKOS.narrower, SKOS.broader
    if rel_norm == "closeMatch":
        if _is_cross_scheme_endpoint(src_id) or _is_cross_scheme_endpoint(dst_id):
            return SKOS.closeMatch, None
        return URIRef(f"{YAW_PREFIX}closeMatch"), None
    if ":" in rel:
        return curie_to_uri(rel), None
    rel_clean = re.sub(r"\s+", "-", rel)
    return URIRef(f"{YAW_PREFIX}predicates/{rel_clean}"), None


def _project_edges(
    g: rdflib.Graph,
    conn: sqlite3.Connection,
    named_quads: bool,
) -> None:
    """Project unified_edges rows into RDF graph or quad context."""
    edge_rows = conn.execute(
        "SELECT src_id, rel_type, dst_id, weight, source_chunk_id FROM unified_edges"
    ).fetchall()
    is_conjunctive = named_quads and isinstance(g, ConjunctiveGraph)

    for raw_src, raw_rel, raw_dst, _weight, raw_chunk in edge_rows:
        src_str = str(raw_src)
        dst_str = str(raw_dst)
        s_uri = curie_to_uri(src_str)
        o_uri = curie_to_uri(dst_str)

        rel = str(raw_rel).strip()
        rel_norm = rel.split(":", 1)[1] if ":" in rel else rel

        ctx_graph = g
        if is_conjunctive:
            ctx_uri = chunk_to_context_uri(str(raw_chunk) if raw_chunk else None)
            if ctx_uri is not None:
                ctx_graph = g.get_context(ctx_uri)

        pred_uri, reciprocal_pred = _resolve_edge_predicate(
            rel, rel_norm, src_id=src_str, dst_id=dst_str
        )
        ctx_graph.add((s_uri, pred_uri, o_uri))
        if reciprocal_pred is not None:
            ctx_graph.add((o_uri, reciprocal_pred, s_uri))


def project_knowledge_graph(
    conn: sqlite3.Connection,
    target_graph: rdflib.Graph | None = None,
    include_skos: bool = False,
    vault_scheme_uri: URIRef | None = None,
    named_quads: bool = False,
) -> rdflib.Graph:
    """Project SQLite unified_nodes and unified_edges views into an rdflib Graph or ConjunctiveGraph."""
    verify_schema_integrity(conn)
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') AND name IN ('unified_nodes', 'unified_edges')"
    ).fetchall()
    found = {r[0] for r in rows}
    if "unified_nodes" not in found or "unified_edges" not in found:
        raise sqlite3.OperationalError(
            "Missing required unified views: unified_nodes and unified_edges"
        )

    if target_graph is not None:
        g = target_graph
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            g = ConjunctiveGraph() if named_quads else rdflib.Graph()

    bind_namespaces(g)
    _project_nodes(
        g, conn, include_skos=include_skos, vault_scheme_uri=vault_scheme_uri
    )
    _project_edges(g, conn, named_quads=named_quads)
    return g


def to_rdflib_graph(conn: sqlite3.Connection) -> ConjunctiveGraph:
    """Project unified SQLite views into an in-process rdflib ConjunctiveGraph with named quads."""
    cg = project_knowledge_graph(conn, named_quads=True)
    if isinstance(cg, ConjunctiveGraph):
        return cg
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        res = ConjunctiveGraph()
    for s, p, o in cg:
        res.add((s, p, o))
    return res


def export_graph(conn: sqlite3.Connection, format: str = "turtle") -> str:
    """Export the projected RDF graph in the specified serialization format."""
    if type(format) is not str:
        raise ValueError(f"Invalid format argument: {format!r} (must be str)")

    clean_fmt = format.strip().lower()
    if not clean_fmt or clean_fmt not in FORMAT_MAP:
        raise ValueError(
            f"Invalid format or unsupported RDF format: {format!r}. Supported: {sorted(set(FORMAT_MAP.keys()))}"
        )

    canonical_target = FORMAT_MAP[clean_fmt]
    cg = to_rdflib_graph(conn)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        serialized = cg.serialize(format=canonical_target)

    if isinstance(serialized, bytes):
        return serialized.decode("utf-8")
    return str(serialized)
