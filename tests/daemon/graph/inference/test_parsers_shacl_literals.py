"""Tests for typed predicate disambiguation, scalar literal fallback, sibling, and acronym contracts."""

import pytest

from yaw.core import (
    RelationTraversalPolicy,
    StructuredTriple,
    detect_scalar_literal,
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
    extract_structured_triples,
    parse_shacl_repairs,
)
from yaw.daemon.graph.inference.prompts import (
    ExtractTriples,
    JudgeLabels,
    JudgeRelations,
    ProposeSHACLRepairs,
)


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
        obj_datatype="xsd:string",
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
    assert sts[1].obj_datatype == "xsd:duration"


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
    assert sts[1].obj_datatype == "xsd:integer"
    assert sts[1].obj_def == ""

    assert sts[2].subj == "Sprint Release"
    assert sts[2].obj == "2026-10-25"
    assert sts[2].obj_kind == "literal"
    assert sts[2].obj_datatype == "xsd:date"
    assert sts[2].obj_def == "Target sprint release date"

    assert sts[3].subj == "Service"
    assert sts[3].obj == "v1.2.3"
    assert sts[3].obj_kind == "entity"
    assert sts[3].obj_datatype == ""
    assert sts[3].obj_def == ""


def test_extract_structured_triples_preserves_literal_subject_for_shacl() -> None:
    """Verify that literal subjects are not silently dropped in Python before SHACL validation."""
    raw_literal_subj = {
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
    sts = extract_structured_triples(raw_literal_subj)
    assert len(sts) == 2

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
    sts_both = extract_structured_triples(raw_both_literals)
    assert len(sts_both) == 1
    assert sts_both[0].obj_kind == "literal"
    assert sts_both[0].obj_datatype == "xsd:integer"

    sts_nested = extract_structured_triples(
        [
            [
                ["The Program", "definition of the program"],
                ["was released", "owl:DatatypeProperty"],
                ["19.5.1995", "Release date"],
            ]
        ]
    )
    assert [list(t.as_tuple()) for t in sts_nested] == [["The Program", "was released", "19.5.1995"]]
    assert tuple(
        t.as_tuple()
        for t in extract_structured_triples(
            '{"triples": [[["The Program", "def"], ["was released", "owl:DatatypeProperty"], ["19.5.1995", "date"]]]}'
        )
    ) == (("The Program", "was released", "19.5.1995"),)


@pytest.mark.parametrize(
    ("literal_str", "expected_dt"),
    [
        ("2026-10-25", "xsd:date"),
        ("2026-10-25T05:53:00Z", "xsd:dateTime"),
        ("2026-10-25T12:30:00Z", "xsd:dateTime"),
        ("2026-10-25 12:30:00+02:00", "xsd:dateTime"),
        ("8080", "xsd:integer"),
        ("443", "xsd:integer"),
        ("123456", "xsd:integer"),
        ("+42", "xsd:integer"),
        ("-17", "xsd:integer"),
        ("3.14", "xsd:decimal"),
        ("3.14159", "xsd:decimal"),
        ("true", "xsd:boolean"),
        ("false", "xsd:boolean"),
        ("False", "xsd:boolean"),
        ("99.9%", "xsd:string"),
        ("$100", "xsd:string"),
        ("15ms", "xsd:duration"),
        ("PT10M", "xsd:duration"),
        ("12:30:00", "xsd:time"),
        ("14:30", "xsd:time"),
        ("2:30 pm", "xsd:time"),
        ("100 requests", "xsd:string"),
    ],
)
def test_detect_scalar_literal_positive_cases(literal_str: str, expected_dt: str) -> None:
    """Verify clear-cut W3C XSD detection of booleans, integers, decimals, dates, and dateTimes."""
    is_lit, dt = detect_scalar_literal(literal_str)
    assert is_lit is True, f"Expected {literal_str!r} to be detected as scalar literal"
    assert dt == expected_dt


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
        "v1.2.3",
        "Y2K Bug",
        "Q4 Freeze Window",
        "Apple Inc",
        "Room 101",
        "",
        "   ",
    ],
)
def test_detect_scalar_literal_rejects_normal_concepts(concept_str: str) -> None:
    """Verify normal domain concepts and empty strings default to (False, 'xsd:string')."""
    is_lit, dt = detect_scalar_literal(concept_str)
    assert is_lit is False, f"Expected {concept_str!r} NOT to be a scalar literal"
    assert dt == "xsd:string"


def test_relation_traversal_policy_datatype_property() -> None:
    """Verify RelationTraversalPolicy exposes policy multipliers and aliases."""
    pol = RelationTraversalPolicy()
    assert not hasattr(pol, "owl_different_from")
    assert pol.passage == pytest.approx(1.0)
    assert pol.fact_forward == pytest.approx(1.0)
    assert pol.fact_reverse == pytest.approx(1.0)
    assert pol.skos_related == pytest.approx(1.0)
    assert pol.skos_broader_up == pytest.approx(1.0)
    assert pol.skos_broader_down == pytest.approx(0.5)


def test_gbnf_and_prompts_typed_predicate_and_shacl_actions() -> None:
    """Verify ExtractTriples, JudgeLabels, JudgeRelations, ProposeSHACLRepairs,
    GBNF grammars, and VALID_SHACL_ACTIONS contracts."""
    assert "owl:ObjectProperty" in ExtractTriples.instructions
    assert "owl:DatatypeProperty" in ExtractTriples.instructions
    assert "skos:altLabel" in JudgeLabels.instructions
    assert "skos:hiddenLabel" in JudgeLabels.instructions
    assert "skos:broader" in JudgeRelations.instructions
    assert "skos:related" in JudgeRelations.instructions
    assert "pred-obj ::=" in DSPY_TRIPLES_GBNF_GRAMMAR
    assert "owl:ObjectProperty" in DSPY_TRIPLES_GBNF_GRAMMAR
    assert "owl:DatatypeProperty" in DSPY_TRIPLES_GBNF_GRAMMAR

    dyn_g = build_dspy_triples_grammar(max_triples=2, known_entities=["The Program"])
    assert "subject-0 ::=" in dyn_g
    assert "owl:ObjectProperty" in dyn_g
    assert "owl:DatatypeProperty" in dyn_g

    for action in ("invert_broader", "assert_related"):
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


def test_extract_entity_definitions_preserves_pairs_for_normalization() -> None:
    """Verify that extract_entity_definitions preserves all extracted (name, def) pairs for normalize_openie_record."""
    raw = {
        "named_entities": [
            ["YAW Engine", "Native HippoRAG 2 graph storage system."],
            ["2026-10-25", "Date of the sprint."],
            ["8080", "Port number."],
            ["v1.0.0", "Initial release version."],
            ["FastMCP", "Model Context Protocol streaming transport."],
        ]
    }
    defs = extract_entity_definitions(raw)
    names = [d[0] for d in defs]
    assert names == ["YAW Engine", "2026-10-25", "8080", "v1.0.0", "FastMCP"]


def test_normalize_openie_record_filters_degenerate_self_loops_and_preserves_homographs() -> None:
    """Verify normalize_openie_record rejects same-sense reflexive triples while keeping distinct homograph senses."""
    import pytest

    from yaw.core import OpenIERecord, StructuredTriple
    from yaw.daemon.graph.normalization import normalize_openie_record

    rec_reflexive = OpenIERecord(
        chunk_id="chunk-reflexive",
        entities=("Hyperslate", "a ball"),
        entity_definitions=(
            ("Hyperslate", "Enterprise customer evaluating KV-cache tiering."),
            ("a ball", "Formal royal dance party."),
        ),
        triples=(),
        structured_triples=(
            StructuredTriple(
                subj="Hyperslate",
                subj_def="Enterprise customer evaluating KV-cache tiering.",
                pred="is trialing",
                obj="Hyperslate",
                obj_def="Enterprise customer evaluating KV-cache tiering.",
                spo_def="Hyperslate is trialing Hyperslate.",
                obj_kind="entity",
            ),
        ),
    )
    with pytest.raises(ValueError, match="Statement relates a resource sense to itself"):
        normalize_openie_record(rec_reflexive)

    rec_valid = OpenIERecord(
        chunk_id="chunk-homograph",
        entities=("Hyperslate", "a ball"),
        entity_definitions=(
            ("Hyperslate", "Enterprise customer evaluating KV-cache tiering."),
            ("a ball", "Formal royal dance party."),
        ),
        triples=(),
        structured_triples=(
            StructuredTriple(
                subj="Hyperslate",
                subj_def="Enterprise customer evaluating KV-cache tiering.",
                pred="has lead source",
                obj="Hyperslate",
                obj_def="",
                spo_def="Hyperslate has lead source Hyperslate.",
                obj_kind="literal",
            ),
            StructuredTriple(
                subj="a ball",
                subj_def="Spherical golden toy object.",
                pred="was handled in",
                obj="a ball",
                obj_def="Formal royal dance party.",
                spo_def="A spherical golden toy ball was handled in a formal royal dance ball.",
                obj_kind="entity",
            ),
        ),
    )
    norm = normalize_openie_record(rec_valid)
    assert len(norm.triples) == 2
    assert norm.triples[0].obj_kind == "literal"
    assert norm.triples[0].obj_datatype == "xsd:string"
    assert norm.triples[1].subj == "a ball"
    assert norm.triples[1].obj == "a ball"
    assert norm.triples[1].subj_def == "Spherical golden toy object."
    assert norm.triples[1].obj_def == "Formal royal dance party."
    assert ("a ball", "Formal royal dance party.") in norm.entities

    dyn_g = build_dspy_triples_grammar(max_triples=2, known_entities=["Hyperslate"])
    assert "subject-0" in dyn_g
    assert "literal-statement" in dyn_g
    assert "objects-0" not in dyn_g
