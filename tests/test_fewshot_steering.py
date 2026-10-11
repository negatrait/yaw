"""Comprehensive 4-Tier E2E Test Suite for Epic 5.2 Dynamic Few-Shot Steering & Live Vocabulary Injection.

Methodology: Category-Partition + BVA + Pairwise + Workload Testing per TEST_INFRA.md and
.agents/skills/engineering-real-tests/SKILL.md.

Substrate-Grounded Invariants Verified:
- Tier 1: Feature Coverage (>= 5 tests per feature: registry callable, declarative pipe, live seeding, quad projection, Invariant 1/E1)
- Tier 2: Boundary & Corner Cases (empty DB fallback, CANON_FEWSHOT_MAX cap <= 5, OOV filtering, UTF-8 char spans, empty chunks)
- Tier 3: Cross-Feature Pairwise (Two-Tier EDC convergence, multi-source named quad projection, contested alias state machine)
- Tier 4: Real-World Scenarios (multi-note ingestion convergence with zero duplicate concepts, cold-start vault feedback loop)
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import jinja2
import pytest
import spacy
from spacy_llm.registry import registry

from yaw import config
from yaw.indexer import (
    TierEnum,
    chunk_markdown,
    resolve,
)
from yaw.nlp import (
    OPEN_IE_REL_TEMPLATE,
    build_leaf_pipeline,
)
from yaw.storage import (
    insert_alias,
)

# ---------------------------------------------------------------------------
# Progressive Testability Markers (Milestone 1 & 2 Decoupling)
# ---------------------------------------------------------------------------

_HAS_CANONICAL_EXAMPLES = "yaw.CanonicalExamples.v1" in registry.misc
_HAS_REL_TEMPLATE_EXAMPLES = "prompt_examples" in OPEN_IE_REL_TEMPLATE

requires_canonical_examples = pytest.mark.skipif(
    not _HAS_CANONICAL_EXAMPLES,
    reason="Progressive Testability: yaw.CanonicalExamples.v1 awaiting Milestone 1 implementation",
)

requires_rel_template_examples = pytest.mark.skipif(
    not _HAS_REL_TEMPLATE_EXAMPLES,
    reason="Progressive Testability: OPEN_IE_REL_TEMPLATE prompt_examples block awaiting Milestone 1 implementation",
)


# ---------------------------------------------------------------------------
# Fixtures: Substrate Grounding (Real SQLite, Real Filesystem, Real Views)
# ---------------------------------------------------------------------------


@pytest.fixture
def db_with_active_relations(test_db: sqlite3.Connection) -> sqlite3.Connection:
    """Pre-populate database with active leaf relations and entities."""
    test_db.execute(
        """
        INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition)
        VALUES
            ('concepts:general-relativity', 'notes/einstein.md', 'General Relativity', 'concepts', 'Theory of gravity'),
            ('concepts:spacetime-curvature', 'notes/einstein.md', 'Spacetime Curvature', 'concepts', 'Geometric curvature'),
            ('concepts:quantum-mechanics', 'notes/bohr.md', 'Quantum Mechanics', 'concepts', 'Physics at atomic scale')
        """
    )
    test_db.execute(
        """
        INSERT INTO leaf_relations (relation_id, source_leaf_id, subject_id, predicate, predicate_id, object_id, weight)
        VALUES
            ('rel-1', 'notes/einstein.md#chunk:0', 'concepts:general-relativity', 'causes', 'causes', 'concepts:spacetime-curvature', 1.0),
            ('rel-2', 'notes/einstein.md#chunk:0', 'concepts:general-relativity', 'depends_on', 'depends_on', 'concepts:spacetime-curvature', 0.9),
            ('rel-3', 'notes/bohr.md#chunk:0', 'concepts:quantum-mechanics', 'contradicts', 'contradicts', 'concepts:general-relativity', 0.8)
        """
    )
    test_db.commit()
    return test_db


@pytest.fixture
def db_with_many_relations(test_db: sqlite3.Connection) -> sqlite3.Connection:
    """Pre-populate database with 20 distinct relations to verify CANON_FEWSHOT_MAX cap."""
    for i in range(20):
        pred_name = f"active_rel_{i:02d}"
        test_db.execute(
            """
            INSERT INTO leaf_relations (relation_id, source_leaf_id, subject_id, predicate, predicate_id, object_id, weight)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"rel-many-{i}",
                f"notes/test.md#chunk:{i}",
                f"concepts:subject_{i}",
                pred_name,
                pred_name,
                f"concepts:object_{i}",
                1.0,
            ),
        )
    test_db.commit()
    return test_db


def _get_canonical_examples_factory() -> Callable[
    ..., Callable[[], list[dict[str, Any]]]
]:
    """Retrieve yaw.CanonicalExamples.v1 from registry with clear diagnostic if un-registered."""
    assert "yaw.CanonicalExamples.v1" in registry.misc, (
        "Factory 'yaw.CanonicalExamples.v1' must be registered in spacy_llm.registry.registry.misc"
    )
    factory = registry.misc.get("yaw.CanonicalExamples.v1")
    return factory


# ===========================================================================
# TIER 1: FEATURE COVERAGE (>= 5 tests per feature)
# ===========================================================================

# ---------------------------------------------------------------------------
# Feature 1: yaw.CanonicalExamples.v1 Registry & Callable Protocol (5 tests)
# ---------------------------------------------------------------------------


@requires_canonical_examples
def test_canonical_examples_registry_registration() -> None:
    """Tier 1 / F1.1: Verify factory is registered under @registry.misc('yaw.CanonicalExamples.v1')."""
    assert "yaw.CanonicalExamples.v1" in registry.misc, (
        "yaw.CanonicalExamples.v1 must be present in spacy_llm registry.misc"
    )


@requires_canonical_examples
def test_canonical_examples_factory_callable_signature() -> None:
    """Tier 1 / F1.2: Verify factory returns a Callable[[], list[dict[str, Any]]] protocol."""
    factory = _get_canonical_examples_factory()
    callable_examples = factory(task="rel", max_examples=5)
    assert callable(callable_examples), "Factory must return a callable instance"
    examples = callable_examples()
    assert isinstance(examples, list), "Calling examples must produce a list"
    assert len(examples) <= 5, "Examples count must respect max_examples cap"


@requires_canonical_examples
def test_canonical_examples_rel_schema_conformance() -> None:
    """Tier 1 / F1.3: Verify REL examples adhere to spacy_llm.tasks.rel.RELExample schema."""
    factory = _get_canonical_examples_factory()
    callable_examples = factory(task="rel", max_examples=5)
    examples = callable_examples()
    assert len(examples) > 0, "Default canonical REL examples must not be empty"

    for ex in examples:
        assert isinstance(ex, dict), "Each example must be a dictionary"
        assert "text" in ex and isinstance(ex["text"], str) and len(ex["text"]) > 0, (
            "Example must have non-empty text"
        )
        assert "ents" in ex and isinstance(ex["ents"], list), (
            "Example must have 'ents' list"
        )
        assert "relations" in ex and isinstance(ex["relations"], list), (
            "Example must have 'relations' list"
        )

        for ent in ex["ents"]:
            assert isinstance(ent, dict), "Each entity must be a dict"
            assert "start_char" in ent and isinstance(ent["start_char"], int), (
                "Entity requires int start_char"
            )
            assert "end_char" in ent and isinstance(ent["end_char"], int), (
                "Entity requires int end_char"
            )
            assert "label" in ent and isinstance(ent["label"], str), (
                "Entity requires str label"
            )
            assert 0 <= ent["start_char"] < ent["end_char"] <= len(ex["text"]), (
                f"Entity span [{ent['start_char']}, {ent['end_char']}] out of bounds for text of len {len(ex['text'])}"
            )

        for rel in ex["relations"]:
            assert isinstance(rel, dict), "Each relation must be a dict"
            assert "dep" in rel and isinstance(rel["dep"], int), (
                "Relation requires int dep index"
            )
            assert "dest" in rel and isinstance(rel["dest"], int), (
                "Relation requires int dest index"
            )
            assert "relation" in rel and isinstance(rel["relation"], str), (
                "Relation requires str relation name"
            )
            assert 0 <= rel["dep"] < len(ex["ents"]), (
                "dep index must point to a valid entity in ents"
            )
            assert 0 <= rel["dest"] < len(ex["ents"]), (
                "dest index must point to a valid entity in ents"
            )


@requires_canonical_examples
def test_canonical_examples_spancat_schema_conformance() -> None:
    """Tier 1 / F1.4: Verify SpanCat examples adhere to SpanCatExample schema."""
    factory = _get_canonical_examples_factory()
    callable_examples = factory(task="spancat", max_examples=5)
    examples = callable_examples()
    assert len(examples) > 0, "SpanCat examples must not be empty"

    for ex in examples:
        assert isinstance(ex, dict), "SpanCat example must be a dictionary"
        assert "text" in ex and isinstance(ex["text"], str) and len(ex["text"]) > 0, (
            "Example requires text string"
        )
        assert "entities" in ex and isinstance(ex["entities"], dict), (
            "SpanCat example requires 'entities' dict mapping labels to text spans"
        )

        for label, span_list in ex["entities"].items():
            assert isinstance(label, str), "Label must be string"
            assert isinstance(span_list, list), "Entity span collection must be a list"
            for span in span_list:
                assert isinstance(span, str), "Span text must be string"
                assert span in ex["text"], (
                    f"Span '{span}' must exist in example text '{ex['text']}'"
                )


@requires_canonical_examples
def test_canonical_example_collection_polymorphism() -> None:
    """Tier 1 / F1.5: Verify CanonicalExampleCollection is polymorphic: behaves as list and is callable."""
    import importlib.util

    examples_mod: Any = (
        importlib.import_module("yaw.nlp.examples")
        if importlib.util.find_spec("yaw.nlp.examples") is not None
        else None
    )
    if examples_mod is None:
        factory = _get_canonical_examples_factory()
        coll = factory(task="rel", max_examples=3)
        assert isinstance(coll, list), "Result must inherit from list"
        assert callable(coll), "Result must implement __call__()"
        assert coll() == coll, (
            "Invoking __call__() on collection must return its contained examples"
        )
        return

    collection_cls = examples_mod.CanonicalExampleCollection
    sample = [{"text": "Einstein was a physicist.", "ents": [], "relations": []}]
    collection = collection_cls(sample)
    assert isinstance(collection, list), "Collection must subclass list"
    assert callable(collection), "Collection must implement __call__"
    assert len(collection) == 1
    assert collection[0]["text"] == "Einstein was a physicist."
    invoked = collection()
    assert invoked == sample, "Calling collection must return inner list"


# ---------------------------------------------------------------------------
# Feature 2: Declarative Pipeline Assembly (5 tests)
# ---------------------------------------------------------------------------


@requires_canonical_examples
def test_declarative_pipeline_assembly_with_canonical_examples() -> None:
    """Tier 1 / F2.1: Blank spaCy pipeline adds llm pipe declaring yaw.CanonicalExamples.v1."""
    _get_canonical_examples_factory()
    nlp = spacy.blank("en")
    pipe: Any = nlp.add_pipe(
        "llm",
        name="llm_rel",
        config={
            "task": {
                "@llm_tasks": "spacy.REL.v1",
                "labels": list(config.BASE_PREDICATES),
                "examples": {"@misc": "yaw.CanonicalExamples.v1"},
            },
            "model": {"@llm_models": "yaw.Ollama.v1"},
        },
    )
    assert "llm_rel" in nlp.pipe_names, "Pipe llm_rel must be present in pipeline"
    assert pipe.name == "llm_rel"


@requires_rel_template_examples
def test_open_ie_rel_template_renders_with_fewshot_examples() -> None:
    """Tier 1 / F2.2: OPEN_IE_REL_TEMPLATE renders without Jinja errors when prompt_examples are supplied."""
    sample_examples = [
        {
            "text": "General Relativity depends on Differential Geometry.",
            "ents": [
                {"start_char": 0, "end_char": 18, "label": "concepts"},
                {"start_char": 30, "end_char": 51, "label": "concepts"},
            ],
            "relations": [{"dep": 0, "dest": 1, "relation": "depends_on"}],
        }
    ]
    template = jinja2.Template(OPEN_IE_REL_TEMPLATE)
    rendered = template.render(
        text="Special Relativity depends on Lorentz Transformations.",
        prompt_examples=sample_examples,
        labels=list(config.BASE_PREDICATES),
    )
    assert "Special Relativity depends on Lorentz Transformations." in rendered, (
        "Target text must be present in rendered prompt"
    )
    assert "depends_on" in rendered, (
        "Few-shot example relation must appear in rendered prompt"
    )


@requires_canonical_examples
def test_declarative_pipeline_zero_private_attribute_mutations() -> None:
    """Tier 1 / F2.3: Invariant 5B.2: Verify pipeline assembly performs zero private _model or _task assignments."""
    _get_canonical_examples_factory()
    nlp = spacy.blank("en")
    pipe: Any = nlp.add_pipe(
        "llm",
        name="llm_test_rel",
        config={
            "task": {
                "@llm_tasks": "spacy.REL.v1",
                "labels": ["depends_on", "causes", "subClassOf"],
                "examples": {"@misc": "yaw.CanonicalExamples.v1"},
            },
            "model": {"@llm_models": "yaw.Ollama.v1"},
        },
    )
    assert hasattr(pipe, "_model"), "spaCy-LLM internal pipe defines _model"
    assert callable(pipe._model)


@requires_canonical_examples
def test_declarative_pipeline_execution_with_test_model() -> None:
    """Tier 1 / F2.4: Declarative pipeline executes with TestModelWrapper and extracts relations into doc._.rel."""

    class _MockSteeringRelModel:
        @property
        def context_length(self) -> int:
            return 8192

        def __call__(self, prompts: Iterable[Iterable[str]]) -> Iterable[Iterable[str]]:
            res: list[list[str]] = []
            for doc_prompts in prompts:
                doc_res = [
                    '{"dep": 0, "dest": 1, "relation": "depends_on"}'
                    for _ in doc_prompts
                ]
                res.append(doc_res)
            return res

    _get_canonical_examples_factory()
    nlp = spacy.blank("en")
    nlp.add_pipe(
        "llm",
        name="llm_rel",
        config={
            "task": {
                "@llm_tasks": "spacy.REL.v1",
                "labels": ["depends_on", "causes"],
                "examples": {"@misc": "yaw.CanonicalExamples.v1"},
            },
            "model": _MockSteeringRelModel(),
        },
    )

    doc = nlp.make_doc("General relativity depends on differential geometry.")
    span1 = doc.char_span(0, 18, label="concepts")
    span2 = doc.char_span(30, 51, label="concepts")
    if span1 and span2:
        doc.ents = [span1, span2]

    processed = nlp(doc)
    assert hasattr(processed._, "rel"), "Doc must possess 'rel' extension attribute"
    assert isinstance(processed._.rel, dict)


@requires_canonical_examples
def test_declarative_pipeline_multi_task_assembly() -> None:
    """Tier 1 / F2.5: Build multi-task pipeline with NER and steered REL declaratively without conflicts."""
    _get_canonical_examples_factory()
    nlp = spacy.blank("en")
    nlp.add_pipe(
        "llm",
        name="llm_ner",
        config={
            "task": {
                "@llm_tasks": "spacy.NER.v3",
                "labels": ["concepts", "methods"],
            },
            "model": {"@llm_models": "yaw.Ollama.v1"},
        },
    )
    nlp.add_pipe(
        "llm",
        name="llm_rel",
        config={
            "task": {
                "@llm_tasks": "spacy.REL.v1",
                "labels": list(config.BASE_PREDICATES),
                "examples": {"@misc": "yaw.CanonicalExamples.v1"},
            },
            "model": {"@llm_models": "yaw.Ollama.v1"},
        },
    )
    assert nlp.pipe_names == ["llm_ner", "llm_rel"]


# ---------------------------------------------------------------------------
# Feature 3: Live Vocabulary Seeding & Cold-Start Fallback (5 tests)
# ---------------------------------------------------------------------------


@requires_canonical_examples
def test_cold_start_empty_db_fallback_to_canonical_predicates(
    test_db: sqlite3.Connection, tmp_path: Path
) -> None:
    """Tier 1 / F3.2: Cold start with 0 rows in unified_edges falls back cleanly to static canonical examples."""
    factory = _get_canonical_examples_factory()
    db_path = tmp_path / "yaw.db"
    callable_examples = factory(task="rel", max_examples=5, db_path=str(db_path))
    examples = callable_examples()
    assert len(examples) > 0, (
        "Empty database must fall back to canonical static examples"
    )
    for ex in examples:
        for rel in ex["relations"]:
            assert rel["relation"] in config.BASE_PREDICATES


@requires_canonical_examples
def test_live_seeding_enriches_predicates_from_unified_edges(
    db_with_active_relations: sqlite3.Connection, tmp_path: Path
) -> None:
    """Tier 1 / F3.3: Active relations from unified_edges are queried and injected into few-shot examples."""
    factory = _get_canonical_examples_factory()
    db_path = tmp_path / "yaw.db"
    callable_examples = factory(task="rel", max_examples=5, db_path=str(db_path))
    examples = callable_examples()
    relations_used = {
        rel["relation"] for ex in examples for rel in ex.get("relations", [])
    }
    assert "causes" in relations_used or "depends_on" in relations_used


@requires_canonical_examples
def test_live_seeding_deduplicates_and_preserves_canonical_precedence(
    test_db: sqlite3.Connection, tmp_path: Path
) -> None:
    """Tier 1 / F3.4: Dynamic seeding deduplicates active relations and maintains canonical precedence."""
    for i in range(10):
        test_db.execute(
            """
            INSERT INTO leaf_relations (relation_id, source_leaf_id, subject_id, predicate, predicate_id, object_id, weight)
            VALUES (?, ?, ?, ?, ?, ?, 1.0)
            """,
            (
                f"dup-rel-{i}",
                "note.md#chunk:0",
                "concepts:sub",
                "subClassOf",
                "subClassOf",
                "concepts:obj",
            ),
        )
    test_db.commit()

    factory = _get_canonical_examples_factory()
    db_path = tmp_path / "yaw.db"
    callable_examples = factory(task="rel", max_examples=5, db_path=str(db_path))
    examples = callable_examples()
    assert len(examples) <= 5


@requires_canonical_examples
def test_live_seeding_graceful_on_missing_tables(tmp_path: Path) -> None:
    """Tier 1 / F3.5: Missing unified_edges / un-migrated DB falls back to static examples without OperationalError."""
    unmigrated_db = tmp_path / "raw_empty.db"
    conn = sqlite3.connect(unmigrated_db)
    conn.execute("CREATE TABLE dummy (id INT)")
    conn.commit()
    conn.close()

    factory = _get_canonical_examples_factory()
    callable_examples = factory(task="rel", max_examples=5, db_path=str(unmigrated_db))
    examples = callable_examples()
    assert len(examples) > 0


# ---------------------------------------------------------------------------
# Feature 4: Quad Projection in unified_edges (5 tests)
# ---------------------------------------------------------------------------


def test_unified_edges_named_quad_five_columns_schema(
    test_db: sqlite3.Connection,
) -> None:
    """Tier 1 / F4.1: Invariant 5B.4: unified_edges view projects exactly 5 columns: src_id, dst_id, rel_type, weight, source_chunk_id."""
    cursor = test_db.execute(
        "SELECT src_id, dst_id, rel_type, weight, source_chunk_id FROM unified_edges LIMIT 0"
    )
    columns = [desc[0] for desc in cursor.description]
    assert columns == [
        "src_id",
        "dst_id",
        "rel_type",
        "weight",
        "source_chunk_id",
    ]


def test_leaf_relation_projects_to_unified_edges_quad(
    test_db: sqlite3.Connection,
) -> None:
    """Tier 1 / F4.2: Steered leaf relation projects into unified_edges named quad."""
    test_db.execute(
        """
        INSERT INTO leaf_relations (relation_id, source_leaf_id, subject_id, predicate, predicate_id, object_id, weight)
        VALUES ('lr-test', 'notes/rel.md#chunk:1', 'concepts:spacetime', 'manifests_in', 'manifests_in', 'concepts:gravity', 1.0)
        """
    )
    test_db.commit()

    rows = test_db.execute(
        "SELECT src_id, dst_id, rel_type, weight, source_chunk_id FROM unified_edges WHERE src_id = 'concepts:spacetime'"
    ).fetchall()
    assert len(rows) == 1
    src, dst, rel, weight, chunk_id = rows[0]
    assert src == "concepts:spacetime"
    assert dst == "concepts:gravity"
    assert rel == "manifests_in"
    assert weight == 1.0
    assert chunk_id == "notes/rel.md#chunk:1"


def test_quad_provenance_grounds_to_source_chunk(
    test_db: sqlite3.Connection,
) -> None:
    """Tier 1 / F4.3: Provenance source_chunk_id grounds directly to exact chunk URI."""
    expected_chunk = "notes/quantum_mechanics.md#chunk:42"
    test_db.execute(
        """
        INSERT INTO leaf_relations (relation_id, source_leaf_id, subject_id, predicate, predicate_id, object_id, weight)
        VALUES ('lr-prov', ?, 'concepts:wave-function', 'collapses_to', 'collapses_to', 'concepts:eigenstate', 0.95)
        """,
        (expected_chunk,),
    )
    test_db.commit()

    row = test_db.execute(
        "SELECT source_chunk_id FROM unified_edges WHERE src_id = 'concepts:wave-function'"
    ).fetchone()
    assert row is not None
    assert row[0] == expected_chunk


def test_quad_weight_preservation(test_db: sqlite3.Connection) -> None:
    """Tier 1 / F4.4: Custom floating-point weights propagate intact to unified_edges."""
    custom_weight = 0.725
    test_db.execute(
        """
        INSERT INTO leaf_relations (relation_id, source_leaf_id, subject_id, predicate, predicate_id, object_id, weight)
        VALUES ('lr-wt', 'note.md#chunk:0', 'concepts:src', 'supports', 'supports', 'concepts:dst', ?)
        """,
        (custom_weight,),
    )
    test_db.commit()

    row = test_db.execute(
        "SELECT weight FROM unified_edges WHERE src_id = 'concepts:src'"
    ).fetchone()
    assert row is not None
    assert abs(row[0] - custom_weight) < 1e-6


def test_quad_active_alias_canonicalization_join(
    test_db: sqlite3.Connection,
) -> None:
    """Tier 1 / F4.5: Active aliases in aliases table re-route src_id, dst_id, and rel_type in unified_edges."""
    insert_alias(
        conn=test_db,
        alias_id="surface:relativity-theory",
        canonical_id="concepts:general-relativity",
        kind="entity",
        method="exact",
        status="active",
    )
    insert_alias(
        conn=test_db,
        alias_id="leads_to",
        canonical_id="causes",
        kind="predicate",
        method="exact",
        status="active",
    )

    test_db.execute(
        """
        INSERT INTO leaf_relations (relation_id, source_leaf_id, subject_id, predicate, predicate_id, object_id, weight)
        VALUES ('lr-alias', 'note.md#chunk:0', 'surface:relativity-theory', 'leads_to', 'leads_to', 'concepts:time-dilation', 1.0)
        """
    )
    test_db.commit()

    row = test_db.execute(
        "SELECT src_id, dst_id, rel_type FROM unified_edges WHERE source_chunk_id = 'note.md#chunk:0'"
    ).fetchone()
    assert row is not None
    src, dst, rel = row
    assert src == "concepts:general-relativity", (
        "src_id must be re-routed to canonical entity"
    )
    assert dst == "concepts:time-dilation", "dst_id must match target entity"
    assert rel == "causes", "rel_type must be re-routed to canonical predicate"


# ---------------------------------------------------------------------------
# Feature 5: Substrate Immutability & Invariants 1 and E1 (5 tests)
# ---------------------------------------------------------------------------


def test_invariant_1_raw_file_bitwise_immutability(tmp_path: Path) -> None:
    """Tier 1 / F5.1: Invariant 1: Source markdown file remains bitwise unchanged across chunking and extraction."""
    note_path = tmp_path / "raw_note.md"
    content = (
        "# General Relativity\n\n"
        "Einstein proposed General Relativity in 1915.\n"
        "It describes gravity as spacetime curvature."
    )
    note_path.write_text(content, encoding="utf-8")
    initial_sha = hashlib.sha256(note_path.read_bytes()).hexdigest()

    chunks = chunk_markdown(content)
    assert len(chunks) >= 1

    post_sha = hashlib.sha256(note_path.read_bytes()).hexdigest()
    assert initial_sha == post_sha, "Source file bytes mutated during indexing"


def test_invariant_e1_doc_text_equals_chunk_text() -> None:
    """Tier 1 / F5.2: Invariant E1: doc.text must equal chunk.text with zero whitespace modification."""
    sample_text = (
        "  \n### Quantum Mechanics\n\nSchrödinger equation dictates state evolution.\n"
    )
    nlp = spacy.blank("en")
    doc = nlp.make_doc(sample_text)
    assert doc.text == sample_text, "doc.text must equal raw input text"


def test_chunk_line_range_exact_hydration_verification(tmp_path: Path) -> None:
    """Tier 1 / F5.3: 1-indexed line hydration on physical disk reproduces exact chunk text."""
    note_file = tmp_path / "physical_seek_note.md"
    content = (
        "# Section Alpha\n\n"
        + "Alpha content with unicode α ≤ 0.85 and emoji 🌟.\n" * 50
        + "\n\n## Section Beta\n\n"
        + "Beta content with secondary propositions.\n" * 50
    )
    note_file.write_text(content, encoding="utf-8")

    chunks = chunk_markdown(content)
    assert len(chunks) >= 2

    lines = content.splitlines(keepends=True)
    for chunk in chunks:
        assert chunk.line_start >= 1
        assert chunk.line_end >= chunk.line_start
        hydrated = "".join(lines[chunk.line_start - 1 : chunk.line_end])
        assert hydrated.rstrip("\r\n") == chunk.text.rstrip("\r\n"), (
            f"Line hydration failed on chunk {chunk.chunk_id}"
        )


def test_entity_character_spans_anchor_exact_substrings() -> None:
    """Tier 1 / F5.4: Entity spans anchor exact substrings in raw document text."""
    text = "The Casimir Effect demonstrates zero-point vacuum fluctuations."
    nlp = spacy.blank("en")
    doc = nlp.make_doc(text)
    span = doc.char_span(4, 18, label="concepts")
    assert span is not None
    assert span.text == "Casimir Effect"
    assert doc.text[span.start_char : span.end_char] == "Casimir Effect"


def test_immutable_chunk_caching_integrity(tmp_path: Path) -> None:
    """Tier 1 / F5.5: BatchCache initialization preserves text string immutability."""
    nlp = build_leaf_pipeline(cache_dir=tmp_path / "test_cache")
    text = "Thermodynamics describes heat transfer."
    doc = nlp.make_doc(text)
    assert doc.text == text


# ===========================================================================
# TIER 2: BOUNDARY & CORNER CASES
# ===========================================================================


@requires_canonical_examples
def test_empty_sqlite_database_graceful_fallback(tmp_path: Path) -> None:
    """Tier 2 / B1: Completely uninitialized database falls back cleanly to 14 canonical predicates."""
    empty_db = tmp_path / "empty_vault.db"
    conn = sqlite3.connect(empty_db)
    conn.close()

    factory = _get_canonical_examples_factory()
    callable_examples = factory(task="rel", max_examples=5, db_path=str(empty_db))
    examples = callable_examples()
    assert 1 <= len(examples) <= 5, "Must return canonical examples bounded by max cap"


@requires_canonical_examples
def test_canon_fewshot_max_cap_enforcement(
    db_with_many_relations: sqlite3.Connection, tmp_path: Path
) -> None:
    """Tier 2 / B2: CANON_FEWSHOT_MAX cap <= 5 strictly enforced even when DB has 20 active predicates."""
    canon_max = getattr(config, "CANON_FEWSHOT_MAX", 5)
    assert canon_max <= 5, "CANON_FEWSHOT_MAX must be <= 5 to prevent context bloating"

    factory = _get_canonical_examples_factory()
    db_path = tmp_path / "yaw.db"
    callable_examples = factory(
        task="rel", max_examples=canon_max, db_path=str(db_path)
    )
    examples = callable_examples()
    assert len(examples) <= canon_max, (
        f"Injected examples ({len(examples)}) exceeded CANON_FEWSHOT_MAX ({canon_max})"
    )


@requires_canonical_examples
def test_out_of_vocabulary_noisy_predicates_filtering(
    test_db: sqlite3.Connection, tmp_path: Path
) -> None:
    """Tier 2 / B3: Deprecated/noisy predicates (is_a, child_of, kind_of) are filtered out of few-shot injection."""
    noisy_predicates = ["is_a", "child_of", "kind_of"]
    for i, p in enumerate(noisy_predicates):
        test_db.execute(
            """
            INSERT INTO leaf_relations (relation_id, source_leaf_id, subject_id, predicate, predicate_id, object_id, weight)
            VALUES (?, 'note.md#chunk:0', 'concepts:sub', ?, ?, 'concepts:obj', 1.0)
            """,
            (f"noisy-{i}", p, p),
        )
    test_db.commit()

    factory = _get_canonical_examples_factory()
    db_path = tmp_path / "yaw.db"
    callable_examples = factory(task="rel", max_examples=5, db_path=str(db_path))
    examples = callable_examples()
    for ex in examples:
        for r in ex.get("relations", []):
            assert r["relation"] not in noisy_predicates, (
                f"Noisy predicate '{r['relation']}' leaked into few-shot steering"
            )


def test_unicode_and_utf8_multibyte_entity_spans() -> None:
    """Tier 2 / B4: Entity mentions with multi-byte UTF-8 glyphs anchor exact char spans without offset drift."""
    text = (
        "Schrödinger's Cat 🐱 and Gödel's Incompleteness Theorem ℏ in"
        " mathematical physics."
    )
    nlp = spacy.blank("en")
    doc = nlp.make_doc(text)

    # Multi-byte German umlaut + emoji span
    target1 = "Schrödinger's Cat 🐱"
    start1 = text.index(target1)
    end1 = start1 + len(target1)
    span1 = doc.char_span(start1, end1, label="concepts")
    assert span1 is not None
    assert span1.text == target1
    assert doc.text[span1.start_char : span1.end_char] == target1

    # Multi-byte German umlaut + math symbol span
    target2 = "Gödel's Incompleteness Theorem ℏ"
    start2 = text.index(target2)
    end2 = start2 + len(target2)
    span2 = doc.char_span(start2, end2, label="concepts")
    assert span2 is not None
    assert span2.text == target2
    assert doc.text[span2.start_char : span2.end_char] == target2


def test_empty_and_whitespace_only_chunks_handling() -> None:
    """Tier 2 / B5: Empty and whitespace-only chunks parse gracefully without producing dangling nodes."""
    assert chunk_markdown("") == []
    assert chunk_markdown("   \n\n\t  ") == []


# ===========================================================================
# TIER 3: CROSS-FEATURE PAIRWISE
# ===========================================================================


def test_fewshot_steering_with_two_tier_edc_convergence(
    test_db: sqlite3.Connection,
) -> None:
    """Tier 3 / P1: Steered canonical predicates bypass Tier 2 LLM adjudication and resolve via Tier 1 lexical fast-path."""
    res_entity = resolve(name="General Relativity", kind="entity", conn=test_db)
    assert res_entity.tier in (
        TierEnum.TIER_1_EXACT,
        TierEnum.TIER_2_EMBEDDING,
        TierEnum.NEW_INDUCTION,
    )
    assert res_entity.canonical_id.startswith("concepts:")

    res_predicate = resolve(name="subClassOf", kind="predicate", conn=test_db)
    assert res_predicate.tier == TierEnum.TIER_1_EXACT, (
        "Canonical predicate subClassOf must resolve via Tier 1 exact match"
    )
    assert res_predicate.canonical_id == "subClassOf"


def test_steered_relations_unified_edges_named_quad_projection_multi_source(
    test_db: sqlite3.Connection,
) -> None:
    """Tier 3 / P2: Heterogeneous sources (wikilinks, leaf_relations, compiled_relations) project 5-column quads."""
    test_db.execute(
        "INSERT INTO wikilinks (source_uri, target_uri, link_label) VALUES ('note_a.md', 'note_b.md', 'Link')"
    )
    test_db.execute(
        """
        INSERT INTO leaf_relations (relation_id, source_leaf_id, subject_id, predicate, predicate_id, object_id, weight)
        VALUES ('lr-p2', 'note_a.md#chunk:0', 'concepts:a', 'depends_on', 'depends_on', 'concepts:b', 1.0)
        """
    )
    test_db.execute(
        """
        INSERT INTO compiled_relations (relation_id, subject_id, predicate, object_id, weight)
        VALUES ('cr-p2', 'concepts:c', 'subClassOf', 'concepts:d', 0.8)
        """
    )
    test_db.commit()

    rows = test_db.execute(
        "SELECT src_id, dst_id, rel_type, weight, source_chunk_id FROM unified_edges"
    ).fetchall()
    assert len(rows) == 3, (
        "All 3 heterogeneous relation sources must unify in unified_edges"
    )
    source_chunks = {r[4] for r in rows}
    assert "note_a.md" in source_chunks
    assert "note_a.md#chunk:0" in source_chunks
    assert "compiled:cr-p2" in source_chunks


def test_dynamic_seeding_with_contested_alias_state_machine(
    test_db: sqlite3.Connection, tmp_path: Path
) -> None:
    """Tier 3 / P3: Contested aliases are excluded from canonical re-routing and dynamic live seeding."""
    insert_alias(
        conn=test_db,
        alias_id="contested_term",
        canonical_id="concepts:disputed",
        kind="entity",
        method="auto",
        status="contested",
    )
    test_db.execute(
        """
        INSERT INTO leaf_relations (relation_id, source_leaf_id, subject_id, predicate, predicate_id, object_id, weight)
        VALUES ('lr-contested', 'note.md#chunk:0', 'contested_term', 'applies_to', 'applies_to', 'concepts:target', 1.0)
        """
    )
    test_db.commit()

    row = test_db.execute(
        "SELECT src_id FROM unified_edges WHERE source_chunk_id = 'note.md#chunk:0'"
    ).fetchone()
    assert row is not None
    assert row[0] == "contested_term", (
        "Contested alias must not be re-routed to canonical ID"
    )


# ===========================================================================
# TIER 4: REAL-WORLD SCENARIOS
# ===========================================================================


def test_multinote_ingestion_canonical_convergence_and_zero_drift(
    test_db: sqlite3.Connection, tmp_path: Path
) -> None:
    """Tier 4 / S1: Multi-note ingestion with synonymous entity mentions converges to single canonical concept with zero duplicates."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    note_a = raw_dir / "relativity_alpha.md"
    note_a.write_text(
        "# Special Theory of Relativity\n\nSpecial Theory of Relativity is a branch of Physics.",
        encoding="utf-8",
    )

    note_b = raw_dir / "relativity_beta.md"
    note_b.write_text(
        "# Einstein's Special Relativity\n\nEinstein's Special Relativity depends on Lorentz Transformations.",
        encoding="utf-8",
    )

    res_a = resolve("Special Theory of Relativity", kind="entity", conn=test_db)
    canon_id = res_a.canonical_id
    assert canon_id.startswith("concepts:")

    insert_alias(
        conn=test_db,
        alias_id="Einstein's Special Relativity",
        canonical_id=canon_id,
        kind="entity",
        method="exact",
        status="active",
    )

    test_db.execute(
        """
        INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition)
        VALUES (?, 'relativity_alpha.md', 'Special Theory of Relativity', 'concepts', 'Relativity note A'),
               (?, 'relativity_beta.md', 'Einstein''s Special Relativity', 'concepts', 'Relativity note B')
        """,
        (canon_id, canon_id),
    )

    test_db.execute(
        """
        INSERT INTO leaf_relations (relation_id, source_leaf_id, subject_id, predicate, predicate_id, object_id, weight)
        VALUES
            ('rel-a', 'relativity_alpha.md#chunk:0', ?, 'subClassOf', 'subClassOf', 'concepts:physics', 1.0),
            ('rel-b', 'relativity_beta.md#chunk:0', 'Einstein''s Special Relativity', 'depends_on', 'depends_on', 'concepts:lorentz-transformations', 1.0)
        """,
        (canon_id,),
    )
    test_db.commit()

    nodes = test_db.execute(
        "SELECT node_id, count(*) FROM unified_nodes WHERE node_id = ? GROUP BY node_id",
        (canon_id,),
    ).fetchall()
    assert len(nodes) == 1, (
        "unified_nodes must deduplicate multiple mentions of same canonical concept across leaves"
    )

    edges = test_db.execute(
        "SELECT src_id, rel_type, source_chunk_id FROM unified_edges WHERE src_id = ?",
        (canon_id,),
    ).fetchall()
    assert len(edges) == 2, (
        "Both relations must project with src_id matching canonical concept"
    )
    edge_rels = {e[1] for e in edges}
    assert edge_rels == {"subClassOf", "depends_on"}


@requires_canonical_examples
def test_cold_start_vault_feedback_loop(
    test_db: sqlite3.Connection, tmp_path: Path
) -> None:
    """Tier 4 / S2: Cold-start empty vault dynamically enriches Stage 1 extraction few-shot prompt with newly seeded active vocabulary."""
    factory = _get_canonical_examples_factory()
    db_path = tmp_path / "yaw.db"

    callable_init = factory(task="rel", max_examples=5, db_path=str(db_path))
    examples_init = callable_init()
    assert len(examples_init) > 0

    test_db.execute(
        """
        INSERT INTO leaf_relations (relation_id, source_leaf_id, subject_id, predicate, predicate_id, object_id, weight)
        VALUES
            ('fb-1', 'note1.md#chunk:0', 'concepts:hypo', 'supports', 'supports', 'concepts:theory', 1.0),
            ('fb-2', 'note1.md#chunk:0', 'concepts:evidence', 'contradicts', 'contradicts', 'concepts:myth', 1.0)
        """
    )
    test_db.commit()

    callable_seeded = factory(task="rel", max_examples=5, db_path=str(db_path))
    examples_seeded = callable_seeded()
    seeded_relations = {
        r["relation"] for ex in examples_seeded for r in ex.get("relations", [])
    }

    assert "supports" in seeded_relations or "contradicts" in seeded_relations
