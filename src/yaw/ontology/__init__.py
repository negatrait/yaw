"""OWL 2 RL, SKOS, and Closed-World SHACL validation package for YAW."""

from yaw.ontology.loader import ingest_ontology
from yaw.ontology.owl import (
    TaxonomicCycleError,
    build_skos_concept_scheme,
    compute_transitive_closure_csr,
    get_broader_closure,
    get_different_from_triples,
    get_ontology_graph,
    get_same_as_triples,
    get_subclass_closure,
    validate_subsumption_dag,
)
from yaw.ontology.registry import (
    DynamicClassMeta,
    DynamicPropertyMeta,
    OntologyRegistry,
    RegistrySnapshot,
    get_ontology_registry,
)
from yaw.ontology.shacl import (
    ValidationReport,
    build_validation_data_graph,
    filter_violations,
    generate_dynamic_whitelist_shape,
    load_shapes_graph,
    parse_validation_report,
    precompute_rdfs_closure,
    validate_wiki_graph,
    validate_wiki_graph_fast,
)

__all__ = [
    "DynamicClassMeta",
    "DynamicPropertyMeta",
    "OntologyRegistry",
    "RegistrySnapshot",
    "TaxonomicCycleError",
    "ValidationReport",
    "build_skos_concept_scheme",
    "build_validation_data_graph",
    "compute_transitive_closure_csr",
    "filter_violations",
    "generate_dynamic_whitelist_shape",
    "get_broader_closure",
    "get_different_from_triples",
    "get_ontology_graph",
    "get_ontology_registry",
    "get_same_as_triples",
    "get_subclass_closure",
    "ingest_ontology",
    "load_shapes_graph",
    "parse_validation_report",
    "precompute_rdfs_closure",
    "validate_subsumption_dag",
    "validate_wiki_graph",
    "validate_wiki_graph_fast",
]
