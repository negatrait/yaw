"""Centralized W3C SKOS/RDF/OWL/SHACL vocabulary constants and pure schema sanitizers."""

from typing import Any

SCHEMA_PLACEHOLDER_DEFS: frozenset[str] = frozenset(
    {
        "skos:definition",
        "skos:concept",
        "skos:preflabel",
        "skos:altlabel",
        "skos:hiddenlabel",
        "rdfs:literal",
        "owl:thing",
        "owl:objectproperty",
        "owl:datatypeproperty",
        "rdf:statement",
    }
)

SKOS_LABEL_KINDS: frozenset[str] = frozenset(
    {
        "skos:prefLabel",
        "skos:altLabel",
        "skos:hiddenLabel",
    }
)

LABEL_KIND_PRIORITY: dict[str, int] = {
    "skos:prefLabel": 3,
    "skos:altLabel": 2,
    "skos:hiddenLabel": 1,
}

SEMANTIC_EDGE_KINDS: frozenset[str] = frozenset(
    {
        "skos:related",
        "skos:broader",
        "owl:differentFrom",
        "skos:topConceptOf",
    }
)

SYNONYM_RELATION_ALIASES: dict[str, str] = {
    "skos:altLabel": "skos:altLabel",
    "altLabel": "skos:altLabel",
    "exact_match": "skos:altLabel",
    "skos:hiddenLabel": "skos:hiddenLabel",
    "hiddenLabel": "skos:hiddenLabel",
    "skos:related": "skos:related",
    "related": "skos:related",
    "skos:broader": "skos:broader",
    "broader": "skos:broader",
    "skos:narrower": "skos:narrower",
    "narrower": "skos:narrower",
    "skos:topConceptOf": "skos:topConceptOf",
    "topConceptOf": "skos:topConceptOf",
    "top_concept_of": "skos:topConceptOf",
    "owl:differentFrom": "owl:differentFrom",
    "differentFrom": "owl:differentFrom",
}

VALID_SYNONYM_RELATIONS: frozenset[str] = frozenset(
    {
        "skos:altLabel",
        "skos:hiddenLabel",
        "skos:related",
        "skos:broader",
        "owl:differentFrom",
    }
)

VALID_SHACL_ACTIONS: frozenset[str] = frozenset(
    {
        "set_definition",
        "rewrite_definition",
        "set_pref_label",
        "demote_to_alt_label",
        "merge_altLabel",
        "merge_hiddenLabel",
        "add_different_from",
        "assert_differentFrom",
        "assert_related",
        "assert_broader",
        "invert_broader",
        "demote_sibling_to_related",
        "remove_hierarchy_cycle",
        "remove_conflicting_related",
        "delete_dangling",
        "delete_invalid_fact",
    }
)

VALUE_BEARING_SHACL_ACTIONS: frozenset[str] = frozenset(
    {
        "set_definition",
        "rewrite_definition",
        "set_pref_label",
    }
)


def clean_schema_definition(value: Any) -> str:
    """Strip whitespace from a definition string and reject echoed W3C schema property CURIEs."""
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in SCHEMA_PLACEHOLDER_DEFS else text
