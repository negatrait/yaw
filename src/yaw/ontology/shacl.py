"""In-process W3C SHACL validation engine for closed-world wiki knowledge graphs.

Executes pyshacl validation over SQLite unified views with OWL 2 RL inference
and zero external graph daemons or background server processes.
"""

from __future__ import annotations

import gc
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

import pyshacl
import rdflib
from rdflib import RDF, RDFS, SH, SKOS, Literal, Namespace, URIRef

from yaw.ontology.owl import get_ontology_graph
from yaw.ontology.registry import get_ontology_registry
from yaw.storage.rdf import YAW_PREFIX, curie_to_uri, project_knowledge_graph

logger = logging.getLogger(__name__)
YAW_NS = Namespace(YAW_PREFIX)
_CACHED_SHAPES_GRAPH: rdflib.Graph | None = None
_CACHED_FAST_SHAPES_GRAPH: rdflib.Graph | None = None


class ValidationReport(list[dict[str, Any]]):
    """Structured list of SHACL violations with backwards-compatible dict access."""

    def __init__(
        self,
        items: Any = (),
        conforms: bool | None = None,
    ) -> None:
        super().__init__(items)
        self._conforms = conforms

    @property
    def status(self) -> str:
        """Conformance status: 'clean' when empty, 'violations' otherwise."""
        return "clean" if len(self) == 0 else "violations"

    @property
    def conforms(self) -> bool:
        """Boolean conformance: True if zero violations, False otherwise."""
        if self._conforms is not None:
            return self._conforms
        return len(self) == 0

    @property
    def violations(self) -> list[dict[str, Any]]:
        """List of diagnostic violation dictionaries."""
        return list(self)

    @property
    def orphaned_nodes(self) -> int:
        """Count of orphaned or dangling target/subject violations."""
        return sum(
            1
            for v in self
            if "dangling" in v.get("message", "").lower()
            or "orphan" in v.get("message", "").lower()
        )

    @property
    def contested_aliases(self) -> int:
        """Count of contested alias merge violations."""
        return sum(1 for v in self if "contested" in v.get("message", "").lower())

    @property
    def dialectical_contradictions(self) -> int:
        """Count of dialectical contradiction violations."""
        return sum(
            1
            for v in self
            if "contradiction" in v.get("message", "").lower()
            or "causes and prevents" in v.get("message", "").lower()
            or "supports and contradicts" in v.get("message", "").lower()
            or "supports and refutes" in v.get("message", "").lower()
        )

    def to_dict(self) -> dict[str, Any]:
        """Export report summary as standard dictionary."""
        return {
            "status": self.status,
            "conforms": self.conforms,
            "violations": self.violations,
            "orphaned_nodes": self.orphaned_nodes,
            "contested_aliases": self.contested_aliases,
            "dialectical_contradictions": self.dialectical_contradictions,
        }

    def keys(self) -> list[str]:
        return list(self.to_dict().keys())

    def values(self) -> list[Any]:
        return list(self.to_dict().values())

    def items(self) -> list[tuple[str, Any]]:
        return list(self.to_dict().items())

    def __getitem__(self, key: Any) -> Any:
        mapping = self.to_dict()
        if key in mapping:
            return mapping[key]
        if isinstance(key, str):
            raise KeyError(key)
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def __contains__(self, key: object) -> bool:
        if key in self.to_dict():
            return True
        return super().__contains__(key)


def load_shapes_graph(reload: bool = False) -> rdflib.Graph:
    """Parse and return the declarative SHACL shapes graph from wiki_shapes.ttl.

    Caches the parsed graph in memory and returns an isolated defensive clone
    to prevent cross-call mutation during in-process validation reasoning.
    """
    global _CACHED_SHAPES_GRAPH
    if not reload and _CACHED_SHAPES_GRAPH is not None:
        clone = rdflib.Graph()
        for triple in _CACHED_SHAPES_GRAPH:
            clone.add(triple)
        return clone

    shapes_path = Path(__file__).resolve().parent / "shapes" / "wiki_shapes.ttl"
    if not shapes_path.is_file():
        shapes_path = Path("src/yaw/ontology/shapes/wiki_shapes.ttl")

    fresh = rdflib.Graph()
    fresh.parse(shapes_path, format="turtle")
    _CACHED_SHAPES_GRAPH = fresh

    clone = rdflib.Graph()
    for triple in fresh:
        clone.add(triple)
    return clone


def _project_contested_aliases(conn: sqlite3.Connection, g: rdflib.Graph) -> None:
    """Project SQLite contested aliases into the data graph for SHACL inspection."""
    tables = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
        ).fetchall()
    }
    if "aliases" not in tables:
        return

    cursor = conn.execute(
        """
        SELECT alias_id, canonical_id FROM aliases
        WHERE status = 'contested' AND alias_id IS NOT NULL AND canonical_id IS NOT NULL
        """
    )
    for alias_id, canonical_id in cursor.fetchall():
        alias_uri = curie_to_uri(str(alias_id))
        canonical_uri = curie_to_uri(str(canonical_id))
        g.add((alias_uri, YAW_NS.aliasStatus, Literal("contested")))
        g.add((alias_uri, YAW_NS.canonicalTarget, canonical_uri))
        g.add((alias_uri, YAW_NS.contestedMerge, canonical_uri))


def build_validation_data_graph(conn: sqlite3.Connection) -> rdflib.Graph:
    """Stream SQLite unified_nodes and unified_edges into an rdflib validation Graph."""
    if conn is None:
        raise ValueError("A valid open sqlite3.Connection is required")
    try:
        g = project_knowledge_graph(conn=conn, include_skos=False, named_quads=False)
    except sqlite3.OperationalError as op_err:
        if "Missing required unified views" in str(op_err):
            return rdflib.Graph()
        raise
    _project_contested_aliases(conn, g)
    return g


def _extract_result_item(
    report_graph: rdflib.Graph, result: rdflib.term.Node
) -> dict[str, str]:
    """Extract single diagnostic dictionary from an sh:ValidationResult subject."""
    focus_node = report_graph.value(result, SH.focusNode)
    path = report_graph.value(result, SH.resultPath)
    severity = report_graph.value(result, SH.resultSeverity)
    component = report_graph.value(result, SH.sourceConstraintComponent)
    source_shape = report_graph.value(result, SH.sourceShape)
    message = report_graph.value(result, SH.resultMessage)
    val = report_graph.value(result, SH.value)

    sev_name = (
        str(severity).rsplit("#", 1)[-1].rsplit("/", 1)[-1] if severity else "Violation"
    )
    comp_name = (
        str(component).rsplit("#", 1)[-1].rsplit("/", 1)[-1]
        if component
        else "ConstraintComponent"
    )
    shape_name = (
        str(source_shape).rsplit("#", 1)[-1].rsplit("/", 1)[-1]
        if source_shape
        else comp_name
    )

    path_str = str(path) if path else ""
    val_str = str(val) if val else ""

    return {
        "focus_node": str(focus_node) if focus_node else "",
        "path": path_str,
        "result_path": path_str,
        "severity": sev_name,
        "constraint": comp_name,
        "shape_id": shape_name,
        "message": str(message) if message else "Integrity constraint violation",
        "value": val_str,
        "conflicting_chunk_id": val_str,
    }


def parse_validation_report(report_graph: rdflib.Graph | None) -> list[dict[str, str]]:
    """Transform W3C sh:ValidationReport RDF graph into structured diagnostic dicts."""
    if report_graph is None:
        return []

    violations: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str, str, str]] = set()

    for result in report_graph.subjects(RDF.type, SH.ValidationResult):
        item = _extract_result_item(report_graph, result)
        key = (
            item["focus_node"],
            item["path"],
            item["severity"],
            item["constraint"],
            item["message"],
            item["value"],
        )
        if key not in seen:
            seen.add(key)
            violations.append(item)

    return violations


def _matches_filter(violation: dict[str, Any], query: str) -> bool:
    """Check if diagnostic violation matches namespace or keyword query."""
    fn = violation.get("focus_node", "").lower()
    p = violation.get("path", "").lower()
    val = violation.get("value", "").lower()
    msg = violation.get("message", "").lower()
    c = violation.get("constraint", "").lower()
    return query in fn or query in p or query in val or query in msg or query in c


def filter_violations(
    violations: list[dict[str, Any]], query: str | None = None
) -> list[dict[str, Any]]:
    """Filter diagnostic violations cleanly by namespace or keyword."""
    if not query or not query.strip() or query.strip() == "*":
        return violations
    q = query.strip().lower()
    return [v for v in violations if _matches_filter(v, q)]


def validate_wiki_graph(
    conn: sqlite3.Connection, query: str | None = None
) -> tuple[bool, ValidationReport]:
    """Execute in-process pySHACL closed-world validation over SQLite unified views.

    Returns tuple of (conforms: bool, report: ValidationReport).
    """
    if conn is None:
        raise ValueError("A valid open sqlite3.Connection is required")

    data_graph = build_validation_data_graph(conn)
    shapes_graph = load_shapes_graph()
    ont_graph = get_ontology_graph()

    conforms, report_graph, _text = pyshacl.validate(
        data_graph=data_graph,
        shacl_graph=shapes_graph,
        ont_graph=ont_graph,
        inference="owlrl",
        advanced=True,
    )

    all_violations = parse_validation_report(report_graph)
    clean_query = (
        query.strip()
        if (query is not None and query.strip() and query.strip() != "*")
        else None
    )
    filtered = filter_violations(all_violations, query=clean_query)
    is_conforming = (
        bool(conforms and len(filtered) == 0)
        if clean_query is None
        else (len(filtered) == 0)
    )

    return is_conforming, ValidationReport(filtered, conforms=is_conforming)


def precompute_rdfs_closure(conn: sqlite3.Connection, g: rdflib.Graph) -> None:
    """Pre-compute transitive subClassOf, broader, and subPropertyOf closure via SQLite recursive CTEs."""
    query = """
    WITH RECURSIVE class_closure(sub, parent, depth) AS (
        SELECT src_id, dst_id, 1 FROM unified_edges
        WHERE rel_type IN ('subClassOf', 'rdfs:subClassOf')
        UNION ALL
        SELECT cc.sub, e.dst_id, cc.depth + 1 FROM unified_edges e
        JOIN class_closure cc ON e.src_id = cc.parent
        WHERE e.rel_type IN ('subClassOf', 'rdfs:subClassOf') AND cc.depth < 5
    ),
    concept_closure(sub, parent, depth) AS (
        SELECT src_id, dst_id, 1 FROM unified_edges
        WHERE rel_type IN ('broader', 'skos:broader')
        UNION ALL
        SELECT cc.sub, e.dst_id, cc.depth + 1 FROM unified_edges e
        JOIN concept_closure cc ON e.src_id = cc.parent
        WHERE e.rel_type IN ('broader', 'skos:broader') AND cc.depth < 5
    ),
    prop_closure(sub, parent, depth) AS (
        SELECT src_id, dst_id, 1 FROM unified_edges
        WHERE rel_type IN ('subPropertyOf', 'rdfs:subPropertyOf')
        UNION ALL
        SELECT pc.sub, e.dst_id, pc.depth + 1 FROM unified_edges e
        JOIN prop_closure pc ON e.src_id = pc.parent
        WHERE e.rel_type IN ('subPropertyOf', 'rdfs:subPropertyOf') AND pc.depth < 5
    )
    SELECT 'class', sub, parent FROM class_closure WHERE depth > 1
    UNION ALL
    SELECT 'concept', sub, parent FROM concept_closure WHERE depth > 1
    UNION ALL
    SELECT 'prop', sub, parent FROM prop_closure WHERE depth > 1;
    """
    try:
        rows = conn.execute(query).fetchall()
    except sqlite3.OperationalError:
        return
    uri_cache: dict[str, URIRef] = {}
    for kind, sub, parent in rows:
        s_str = str(sub)
        p_str = str(parent)
        if s_str not in uri_cache:
            uri_cache[s_str] = curie_to_uri(s_str)
        if p_str not in uri_cache:
            uri_cache[p_str] = curie_to_uri(p_str)
        sub_uri = uri_cache[s_str]
        parent_uri = uri_cache[p_str]
        if kind == "class":
            g.add((sub_uri, RDFS.subClassOf, parent_uri))
        elif kind == "concept":
            g.add((sub_uri, SKOS.broader, parent_uri))
        elif kind == "prop":
            g.add((sub_uri, RDFS.subPropertyOf, parent_uri))


def generate_dynamic_whitelist_shape() -> rdflib.Graph:
    """Synthesize dynamic SHACL namespace whitelist shape from active OntologyRegistry prefixes."""
    reg = get_ontology_registry()
    prefixes = reg.get_active_prefixes()
    base_prefixes = [
        "http://yaw.local/",
        "http://www.w3.org/",
        "https://www.w3.org/",
    ]
    all_prefixes = list(base_prefixes)
    for p in prefixes.values():
        if p and p not in all_prefixes:
            all_prefixes.append(p)

    safe_prefixes = [p.replace(".", "[.]") for p in all_prefixes]
    combined_regex = f"^({'|'.join(safe_prefixes)})"

    shape_ttl = f"""
    @prefix sh: <http://www.w3.org/ns/shacl#> .
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    @prefix yaw: <http://yaw.local/> .

    yaw:DynamicNamespaceShape a sh:NodeShape ;
        sh:targetSubjectsOf rdf:type ;
        sh:nodeKind sh:IRI ;
        sh:pattern "{combined_regex}" ;
        sh:message "URI does not match any registered dynamic ontology namespace."@en .
    """
    g = rdflib.Graph()
    g.parse(data=shape_ttl, format="turtle")
    return g


def load_fast_shapes_graph(reload: bool = False) -> rdflib.Graph:
    """Return lightweight Core SHACL shapes graph for fast in-process validation (<200ms)."""
    global _CACHED_FAST_SHAPES_GRAPH
    if not reload and _CACHED_FAST_SHAPES_GRAPH is not None:
        return _CACHED_FAST_SHAPES_GRAPH

    fast_shapes = """
    @prefix sh: <http://www.w3.org/ns/shacl#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
    @prefix skos: <http://www.w3.org/2004/02/skos/core#> .
    @prefix yaw: <http://yaw.local/> .
    @prefix concepts: <http://yaw.local/concepts/> .
    @prefix predicates: <http://yaw.local/predicates/> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .

    yaw:ConceptShape a sh:NodeShape ;
        sh:targetSubjectsOf skos:definition ;
        sh:property [
            sh:path skos:definition ;
            sh:minCount 1 ;
            sh:datatype xsd:string ;
        ] .

    yaw:RelationShape a sh:NodeShape ;
        sh:targetSubjectsOf rdfs:subClassOf ;
        sh:nodeKind sh:IRI .

    yaw:DialecticalIntegrityShape a sh:NodeShape ;
        sh:targetClass owl:Thing ;
        sh:targetSubjectsOf yaw:contestedMerge, yaw:aliasStatus, predicates:causes, predicates:prevents, predicates:supports, predicates:contradicts, predicates:refutes ;
        sh:sparql [
            a sh:SPARQLConstraint ;
            sh:message "Dialectical contradiction: Node simultaneously causes and prevents target."@en ;
            sh:severity sh:Violation ;
            sh:select '''
                PREFIX predicates: <http://yaw.local/predicates/>
                SELECT $this ?path (?target AS ?value)
                WHERE {
                    $this predicates:causes ?target .
                    $this predicates:prevents ?target .
                    { BIND(<http://yaw.local/predicates/causes> AS ?path) }
                    UNION
                    { BIND(<http://yaw.local/predicates/prevents> AS ?path) }
                }
            ''' ;
        ] ;
        sh:sparql [
            a sh:SPARQLConstraint ;
            sh:message "Dialectical contradiction: Node simultaneously supports and contradicts target."@en ;
            sh:severity sh:Violation ;
            sh:select '''
                PREFIX predicates: <http://yaw.local/predicates/>
                SELECT $this ?path (?target AS ?value)
                WHERE {
                    {
                        $this predicates:supports ?target .
                        $this predicates:contradicts ?target .
                        { BIND(<http://yaw.local/predicates/supports> AS ?path) }
                        UNION
                        { BIND(<http://yaw.local/predicates/contradicts> AS ?path) }
                    }
                    UNION
                    {
                        $this predicates:supports ?target .
                        ?target predicates:contradicts $this .
                        { BIND(<http://yaw.local/predicates/supports> AS ?path) }
                        UNION
                        { BIND(<http://yaw.local/predicates/contradicts> AS ?path) }
                    }
                }
            ''' ;
        ] ;
        sh:sparql [
            a sh:SPARQLConstraint ;
            sh:message "Dialectical contradiction: Node simultaneously supports and refutes target."@en ;
            sh:severity sh:Violation ;
            sh:select '''
                PREFIX predicates: <http://yaw.local/predicates/>
                SELECT $this ?path (?target AS ?value)
                WHERE {
                    {
                        $this predicates:supports ?target .
                        $this predicates:refutes ?target .
                        { BIND(<http://yaw.local/predicates/supports> AS ?path) }
                        UNION
                        { BIND(<http://yaw.local/predicates/refutes> AS ?path) }
                    }
                    UNION
                    {
                        $this predicates:supports ?target .
                        ?target predicates:refutes $this .
                        { BIND(<http://yaw.local/predicates/supports> AS ?path) }
                        UNION
                        { BIND(<http://yaw.local/predicates/refutes> AS ?path) }
                    }
                }
            ''' ;
        ] ;
        sh:sparql [
            a sh:SPARQLConstraint ;
            sh:message "Contested alias merge detected."@en ;
            sh:severity sh:Violation ;
            sh:select '''
                PREFIX yaw: <http://yaw.local/>
                SELECT DISTINCT $this (yaw:aliasStatus AS ?path) (?canonical AS ?value)
                WHERE {
                    { $this yaw:aliasStatus "contested" .
                      OPTIONAL { $this yaw:canonicalTarget ?canonical } }
                    UNION
                    { $this yaw:contestedMerge ?canonical }
                }
            ''' ;
        ] .
    """
    fresh = rdflib.Graph().parse(data=fast_shapes, format="turtle")
    _CACHED_FAST_SHAPES_GRAPH = fresh
    return fresh


_CACHED_FAST_SHAPES_COMBINED: rdflib.Graph | None = None
_CACHED_PREFIX_SIGNATURE: tuple[tuple[str, str], ...] | None = None


def get_combined_fast_shapes(reload: bool = False) -> rdflib.Graph:
    """Return pre-composed and cached Core SHACL shapes graph."""
    global _CACHED_FAST_SHAPES_COMBINED, _CACHED_PREFIX_SIGNATURE
    reg = get_ontology_registry()
    prefixes = reg.get_active_prefixes()
    sig = tuple(sorted(prefixes.items()))

    if (
        not reload
        and _CACHED_FAST_SHAPES_COMBINED is not None
        and _CACHED_PREFIX_SIGNATURE == sig
    ):
        return _CACHED_FAST_SHAPES_COMBINED

    base_shapes = load_fast_shapes_graph(reload=reload)
    whitelist_shapes = generate_dynamic_whitelist_shape()
    combined = rdflib.Graph()
    for triple in base_shapes:
        combined.add(triple)
    for triple in whitelist_shapes:
        combined.add(triple)

    _CACHED_FAST_SHAPES_COMBINED = combined
    _CACHED_PREFIX_SIGNATURE = sig
    return combined


def validate_wiki_graph_fast(
    conn: sqlite3.Connection,
    query: str | None = None,
) -> tuple[bool, ValidationReport]:
    """Execute lean SHACL validation in <200ms using relational pre-closure and inference=None."""
    if conn is None:
        raise ValueError("A valid open sqlite3.Connection is required")

    t0 = time.perf_counter()
    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()
    try:
        data_graph = build_validation_data_graph(conn)
        precompute_rdfs_closure(conn, data_graph)

        shapes_graph = get_combined_fast_shapes()

        conforms, report_graph, _text = pyshacl.validate(
            data_graph=data_graph,
            shacl_graph=shapes_graph,
            ont_graph=None,
            inference=None,
            advanced=True,
            inplace=True,
        )
    finally:
        if gc_was_enabled:
            gc.enable()

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.debug("pySHACL lean validation completed in %.1f ms", elapsed_ms)

    all_violations = parse_validation_report(report_graph)
    clean_query = (
        query.strip()
        if (query is not None and query.strip() and query.strip() != "*")
        else None
    )
    filtered = filter_violations(all_violations, query=clean_query)
    is_conforming = (
        bool(conforms and len(filtered) == 0)
        if clean_query is None
        else (len(filtered) == 0)
    )

    return is_conforming, ValidationReport(filtered, conforms=is_conforming)


__all__ = [
    "ValidationReport",
    "build_validation_data_graph",
    "filter_violations",
    "generate_dynamic_whitelist_shape",
    "load_fast_shapes_graph",
    "load_shapes_graph",
    "parse_validation_report",
    "precompute_rdfs_closure",
    "validate_wiki_graph",
    "validate_wiki_graph_fast",
]
