"""Tests for typed predicate disambiguation, scalar literal fallback, sibling, and acronym contracts."""

import pytest

from yaw.core import (
    RelationTraversalPolicy,
    StructuredTriple,
    check_lexical_subsumption_inversion,
    detect_scalar_literal,
    is_acronym_pair,
    is_version_or_ordinal_sibling,
)
from yaw.daemon.graph.inference.grammars import (
    DSPY_SHACL_REPAIR_GBNF_GRAMMAR,
    DSPY_TRIPLES_GBNF_GRAMMAR,
    build_dspy_shacl_repair_grammar,
    build_dspy_triples_grammar,
)
from yaw.daemon.graph.inference.parsers import (
    VALID_SHACL_ACTIONS,
    extract_entity_definitions,
    extract_named_entities,
    extract_structured_triples,
    extract_triples,
    filter_invalid_triples,
    parse_shacl_repairs,
)
from yaw.daemon.graph.inference.prompts import ExtractTriples, ProposeSHACLRepairs


def test_typed_predicate_datatype_and_object_properties_preserve_definitions() -> None:
    """Verify owl:DatatypeProperty and owl:ObjectProperty disambiguate obj_kind while preserving obj_def."""
    raw = {
        "triples": [
            {
                "sub": ["The Program", "definition of the program"],
                "pred": ["was released", "owl:DatatypeProperty"],
                "obj": ["19.5.1995", "Release date of The Program"],
                "def": "The Program was released on 19.5.1995.",
            },
            {
                "sub": ["The Program", "definition of the program"],
                "pred": ["programmed by", "owl:ObjectProperty"],
                "obj": ["Bob", "Lead developer"],
                "def": "The Program was programmed by Bob.",
            },
        ]
    }
    sts = extract_structured_triples(raw)
    assert len(sts) == 2

    assert sts[0] == StructuredTriple(
        subj="The Program",
        subj_def="definition of the program",
        pred="was released",
        obj="19.5.1995",
        obj_def="Release date of The Program",
        spo_def="The Program was released on 19.5.1995.",
        obj_kind="literal",
        obj_datatype="rdfs:Literal",
        obj_lang="",
    )
    assert sts[1] == StructuredTriple(
        subj="The Program",
        subj_def="definition of the program",
        pred="programmed by",
        obj="Bob",
        obj_def="Lead developer",
        spo_def="The Program was programmed by Bob.",
        obj_kind="entity",
        obj_datatype="",
        obj_lang="",
    )


def test_typed_predicate_case_insensitivity_and_object_property_override() -> None:
    """Verify case-insensitive predicate typing and owl:ObjectProperty overriding numeric lookalikes."""
    raw = {
        "triples": [
            {
                "sub": ["George Orwell", "English novelist"],
                "pred": ["wrote", "owl:objectproperty"],
                "obj": ["1984", "Dystopian social science fiction novel by George Orwell"],
                "def": "George Orwell wrote the novel 1984.",
            },
            {
                "sub": ["Server", "Backend HTTP server"],
                "pred": ["has timeout", "OWL:DATATYPEPROPERTY"],
                "obj": ["30 seconds", "Request timeout threshold"],
                "def": "Server has a timeout of 30 seconds.",
            },
        ]
    }
    sts = extract_structured_triples(raw)
    assert len(sts) == 2

    assert sts[0].pred == "wrote"
    assert sts[0].obj == "1984"
    assert sts[0].obj_def == "Dystopian social science fiction novel by George Orwell"
    assert sts[0].obj_kind == "entity"
    assert sts[0].obj_datatype == ""

    assert sts[1].pred == "has timeout"
    assert sts[1].obj == "30 seconds"
    assert sts[1].obj_def == "Request timeout threshold"
    assert sts[1].obj_kind == "literal"
    assert sts[1].obj_datatype == "rdfs:Literal"


def test_backward_compatible_plain_string_predicate() -> None:
    """Verify backward-compatible plain string 'pred' works for both entities and fallback scalar literals."""
    raw = {
        "triples": [
            {
                "sub": "The Program",
                "pred": "programmed by",
                "obj": ["Bob", "Lead developer"],
                "def": "The Program was programmed by Bob.",
            },
            {
                "sub": "YAW Daemon",
                "pred": "listens on port",
                "obj": "8080",
                "def": "YAW Daemon listens on port 8080.",
            },
            {
                "sub": "Sprint Release",
                "pred": "scheduled for date",
                "obj": ["2026-10-25", "Target sprint release date"],
                "def": "Sprint Release scheduled for date 2026-10-25.",
            },
            {
                "sub": "Service",
                "pred": "has version",
                "obj": "v1.2.3",
                "def": "Service has version v1.2.3.",
            },
        ]
    }
    sts = extract_structured_triples(raw)
    assert len(sts) == 4

    assert sts[0].pred == "programmed by"
    assert sts[0].obj == "Bob"
    assert sts[0].obj_def == "Lead developer"
    assert sts[0].obj_kind == "entity"
    assert sts[0].obj_datatype == ""

    assert sts[1].subj == "YAW Daemon"
    assert sts[1].pred == "listens on port"
    assert sts[1].obj == "8080"
    assert sts[1].obj_kind == "literal"
    assert sts[1].obj_datatype == "rdfs:Literal"
    assert sts[1].obj_def == ""

    assert sts[2].subj == "Sprint Release"
    assert sts[2].obj == "2026-10-25"
    assert sts[2].obj_kind == "literal"
    assert sts[2].obj_datatype == "rdfs:Literal"
    assert sts[2].obj_def == "Target sprint release date"

    assert sts[3].subj == "Service"
    assert sts[3].obj == "v1.2.3"
    assert sts[3].obj_kind == "literal"
    assert sts[3].obj_datatype == "rdfs:Literal"
    assert sts[3].obj_def == ""


def test_extract_structured_triples_handles_literal_subject() -> None:
    """Verify that literal subjects are swapped with entity objects or rejected if both are literals."""
    raw_swappable = {
        "triples": [
            {
                "sub": "2026-10-25",
                "pred": "is release date of",
                "obj": "YAW Engine",
                "def": "2026-10-25 is release date of YAW Engine.",
            },
            {
                "sub": ["2026-10-25", "Release date of YAW Engine"],
                "pred": ["is release date of", "owl:ObjectProperty"],
                "obj": ["YAW Engine", "Native HippoRAG 2 engine"],
                "def": "2026-10-25 is release date of YAW Engine.",
            },
        ]
    }
    sts = extract_structured_triples(raw_swappable)
    assert len(sts) == 2
    assert sts[0].subj == "YAW Engine"
    assert sts[0].obj == "2026-10-25"
    assert sts[0].obj_kind == "literal"
    assert sts[0].obj_datatype == "rdfs:Literal"

    assert sts[1].subj == "YAW Engine"
    assert sts[1].subj_def == "Native HippoRAG 2 engine"
    assert sts[1].obj == "2026-10-25"
    assert sts[1].obj_def == "Release date of YAW Engine"
    assert sts[1].obj_kind == "literal"
    assert sts[1].obj_datatype == "rdfs:Literal"

    raw_both_literals = {
        "triples": [
            {
                "sub": "2024",
                "pred": "preceded",
                "obj": "2025",
                "def": "2024 preceded 2025.",
            }
        ]
    }
    sts_empty = extract_structured_triples(raw_both_literals)
    assert len(sts_empty) == 0

    seq_filtered = filter_invalid_triples(
        [
            [
                ["The Program", "definition of the program"],
                ["was released", "owl:DatatypeProperty"],
                ["19.5.1995", "Release date"],
            ]
        ]
    )
    assert seq_filtered == [["The Program", "was released", "19.5.1995"]]
    assert extract_triples(
        '{"triples": [[["The Program", "def"], ["was released", "owl:DatatypeProperty"], ["19.5.1995", "date"]]]}'
    ) == (("The Program", "was released", "19.5.1995"),)


@pytest.mark.parametrize(
    "literal_str",
    [
        "2026-10-25",
        "2026-10-25T05:53:00Z",
        "2026-10-25T12:30:00Z",
        "2026 10 25",
        "8080",
        "443",
        "123456",
        "+42",
        "-17",
        "3.14",
        "3.14159",
        "99.9%",
        "99.9 %",
        "100%",
        "22,000",
        "true",
        "false",
        "False",
        "v1.0",
        "v2.1",
        "v1.2.3",
        "v2.1.0-alpha",
        "3.14.9",
        "v1 2 3",
        "v2 1",
    ],
)
def test_detect_scalar_literal_positive_cases(literal_str: str) -> None:
    """Verify minimal fallback detection of bare ISO dates, numbers, percentages, booleans, and versions."""
    is_lit, dt = detect_scalar_literal(literal_str)
    assert is_lit is True, f"Expected {literal_str!r} to be detected as scalar literal"
    assert dt == "rdfs:Literal"


@pytest.mark.parametrize(
    "concept_str",
    [
        "Cinderella",
        "PaymentService",
        "HTTP 2",
        "OAuth 2.0",
        "IPv6",
        "SHA-256",
        "Python 3.10",
        "Python 3.14",
        "Y2K Bug",
        "Q4 Freeze Window",
        "Apple Inc",
        "Room 101",
        "",
        "   ",
    ],
)
def test_detect_scalar_literal_rejects_normal_concepts(concept_str: str) -> None:
    """Verify normal domain concepts and empty strings are not classified as scalar literals."""
    is_lit, dt = detect_scalar_literal(concept_str)
    assert is_lit is False, f"Expected {concept_str!r} NOT to be a scalar literal"
    assert dt == ""


@pytest.mark.parametrize(
    ("name_a", "name_b"),
    [
        ("http 2", "http 3"),
        ("v1", "v2"),
        ("ipv4", "ipv6"),
        ("tls 1.2", "tls 1.3"),
        ("osi layer 2", "osi layer 3"),
        ("tls version 1.2", "tls version 1.3"),
        ("severity level 1", "severity level 2"),
        ("priority p0", "priority p1"),
        ("sev 1", "sev 2"),
        ("sev1", "sev2"),
        ("tier 1", "tier 2"),
        ("p0", "p1"),
        ("python 3 10", "python 3 11"),
        ("http 1 1", "http 2"),
    ],
)
def test_is_version_or_ordinal_sibling_positive(name_a: str, name_b: str) -> None:
    """Verify coordinate peers differing only by version, ordinal, severity, priority, or tier."""
    assert is_version_or_ordinal_sibling(name_a, name_b) is True
    assert is_version_or_ordinal_sibling(name_b, name_a) is True


@pytest.mark.parametrize(
    ("name_a", "name_b"),
    [
        ("http 2", "tls 2"),
        ("priority p0", "sev 0"),
        ("http 2", "http 2"),
        ("http 2", "http 2.0"),
        ("oauth 2", "oauth 2.0"),
        ("oauth 2", "oauth 2 0"),
        ("v1", "v1.0"),
        ("sev1", "sev 1"),
        ("tls 1.2", "tls 1 2"),
        ("osi layer 2", "tcp layer 3"),
        ("", "http 2"),
        ("http 2", ""),
    ],
)
def test_is_version_or_ordinal_sibling_negative(name_a: str, name_b: str) -> None:
    """Verify non-siblings and identical/spacing/precision-variant names return False."""
    assert is_version_or_ordinal_sibling(name_a, name_b) is False


def test_is_acronym_pair() -> None:
    """Verify is_acronym_pair handles initialisms, plurals, and parenthesized forms while rejecting siblings."""
    assert is_acronym_pair("SLOs", "service level objectives") is True
    assert is_acronym_pair("service level objectives", "SLOs") is True
    assert is_acronym_pair("API", "application programming interface") is True
    assert is_acronym_pair("LLM", "Large Language Model") is True
    assert is_acronym_pair("LLM", "LLM (Large Language Model)") is True
    assert is_acronym_pair("LLM", "Large Language Model (LLM)") is True
    assert is_acronym_pair("DoS", "Denial of Service (DoS)") is True

    assert is_acronym_pair("sev1", "sev 1") is False
    assert is_acronym_pair("http 2", "http 3") is False
    assert is_acronym_pair("p0", "p 0") is False
    assert is_acronym_pair("s1", "s 1") is False
    assert is_acronym_pair("API", "API") is False
    assert is_acronym_pair("in a", "internal network architecture") is False
    assert is_acronym_pair("of", "optical fiber") is False
    assert is_acronym_pair("in", "internal network") is False


def test_check_lexical_subsumption_inversion() -> None:
    """Verify lexical subsumption inversion detects inverted compound hierarchies and excludes siblings."""
    assert check_lexical_subsumption_inversion("engineering", "security engineering") is True
    assert check_lexical_subsumption_inversion("security engineering", "engineering") is False
    assert check_lexical_subsumption_inversion("tls 1", "tls 1.2") is False
    assert check_lexical_subsumption_inversion("LLM", "Large Language Model (LLM)") is False


def test_relation_traversal_policy_datatype_property_and_aliases() -> None:
    """Verify RelationTraversalPolicy exposes datatype_property=0.5 and owl_different_from=0.0."""
    pol = RelationTraversalPolicy()
    assert pol.datatype_property == pytest.approx(0.5)
    assert pol.owl_different_from == pytest.approx(0.0)
    assert pol.passage == pytest.approx(1.0)
    assert pol.fact_fwd == pytest.approx(1.0)
    assert pol.fact_rev == pytest.approx(1.0)
    assert pol.skos_related == pytest.approx(1.0)
    assert pol.skos_broader == pytest.approx(1.0)
    assert pol.skos_narrower == pytest.approx(0.5)


def test_gbnf_and_prompts_typed_predicate_and_shacl_actions() -> None:
    """Verify ExtractTriples, ProposeSHACLRepairs, GBNF grammars, and VALID_SHACL_ACTIONS contracts."""
    assert "owl:ObjectProperty" in ExtractTriples.instructions
    assert "owl:DatatypeProperty" in ExtractTriples.instructions
    assert "pred-slot ::=" in DSPY_TRIPLES_GBNF_GRAMMAR
    assert "owl:ObjectProperty" in DSPY_TRIPLES_GBNF_GRAMMAR
    assert "owl:DatatypeProperty" in DSPY_TRIPLES_GBNF_GRAMMAR

    dyn_g = build_dspy_triples_grammar(max_triples=2, known_entities=["The Program"])
    assert "pred-slot ::=" in dyn_g
    assert "owl:ObjectProperty" in dyn_g
    assert "owl:DatatypeProperty" in dyn_g

    for action in ("invert_broader", "demote_sibling_to_related"):
        assert action in VALID_SHACL_ACTIONS
        assert action in ProposeSHACLRepairs.instructions
        assert action in DSPY_SHACL_REPAIR_GBNF_GRAMMAR
        assert action in build_dspy_shacl_repair_grammar(max_repairs=2)

    repairs = parse_shacl_repairs(
        [
            {"focus_node": "ent-a->ent-b", "action": "invert_broader", "value": ""},
            {"focus_node": "ent-c->ent-d", "action": "demote_sibling_to_related", "value": ""},
        ]
    )
    assert len(repairs) == 2
    assert repairs[0]["action"] == "invert_broader"
    assert repairs[1]["action"] == "demote_sibling_to_related"


def test_extract_entity_definitions_filters_scalar_literals() -> None:
    """Verify that extract_entity_definitions excludes scalar literals when exclude_scalars=True."""
    raw = {
        "named_entities": [
            ["YAW Engine", "Native HippoRAG 2 graph storage system."],
            ["2026-10-25", "Date of the sprint."],
            ["8080", "Port number."],
            ["v1.0.0", "Initial release version."],
            ["FastMCP", "Model Context Protocol streaming transport."],
        ]
    }
    defs = extract_entity_definitions(raw, exclude_scalars=True)
    names = [d[0] for d in defs]
    assert "YAW Engine" in names
    assert "FastMCP" in names
    assert "2026-10-25" not in names
    assert "8080" not in names
    assert "v1.0.0" not in names

    entities = extract_named_entities(raw, exclude_scalars=True)
    assert "YAW Engine" in entities
    assert "FastMCP" in entities
    assert "2026-10-25" not in entities
    assert "8080" not in entities
    assert "v1.0.0" not in entities
