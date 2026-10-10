"""Stage 1 leaf-taxonomy provenance, spacy-llm extraction, and failure handling."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import spacy
import spacy.tokens
from spacy.language import Language
from spacy.tokens import Doc

from yaw import config
from yaw.nlp import (
    ExtractedNode,
    LeafTaxonomy,
    build_leaf_pipeline,
    extract_leaf_taxonomy,
)


class _MockRelModel:
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses

    @property
    def context_length(self) -> int:
        return 8192

    def __call__(self, prompt_batches):
        for prompts in prompt_batches:
            res = []
            for p in prompts:
                p_str = str(p)
                if "extract named entities" in p_str or "Entities must have" in p_str:
                    if "Tautological tests" in p_str:
                        res.append(
                            "1. Tautological tests | True | concepts | concept\n"
                            "2. source code | True | concepts | concept\n"
                            "3. coding agents | True | entities | agent\n"
                            "4. context rot | True | concepts | problem"
                        )
                    else:
                        res.append(
                            "1. Albert Einstein | True | concepts | physicist\n"
                            "2. Princeton University | True | concepts | university\n"
                            "3. New Jersey | True | concepts | location"
                        )
                elif "relations" in p_str or "assigned between" in p_str:
                    res.append(self._responses[0] if self._responses else "")
                else:
                    res.append(self._responses[0] if self._responses else "")
            yield res


def test_extract_leaf_doc_returns_annotated_doc_and_taxonomy(caplog, tmp_path):
    text = "Albert Einstein worked at Princeton University in New Jersey."
    mock_model = _MockRelModel(
        [
            json.dumps(
                [
                    {
                        "subject": "Albert Einstein",
                        "predicate": "employed_at",
                        "object": "Princeton University",
                    },
                    {
                        "subject": "Princeton University",
                        "predicate": "located_in",
                        "object": "New Jersey",
                    },
                ]
            )
        ]
    )
    nlp = build_leaf_pipeline(
        model=mock_model,
        cache_dir=tmp_path / "cache",
        labels=["employed_at", "located_in"],
    )
    assert isinstance(nlp, Language)
    assert "ner" not in nlp.pipe_names
    assert "markdown_structure" in nlp.pipe_names
    assert "llm_raw" in nlp.pipe_names
    assert "llm_spancat" not in nlp.pipe_names
    assert "llm_rel" not in nlp.pipe_names
    assert "candidate_concepts" not in nlp.pipe_names
    assert "leaf_taxonomy_compiler" in nlp.pipe_names

    with caplog.at_level(logging.DEBUG, logger="yaw.leaf_extractor"):
        doc = nlp(text)

    assert isinstance(doc, Doc)
    assert len(doc.ents) == 3
    assert len(doc._.rel) == 2

    # Check relation typing and attachments
    r0 = doc._.rel[0]
    assert r0.dep == 0
    assert r0.dest == 1
    assert r0.relation == "employ at"

    tax = doc._.leaf_taxonomy
    assert isinstance(tax, LeafTaxonomy)
    assert len(tax.triples) == 2
    assert tax.triples[0].subject.name == "Albert Einstein"
    assert tax.triples[0].predicate.name == "employ at"
    assert tax.triples[0].object.name == "Princeton University"
    assert tax.triples[0].predicate.namespace == "predicates"

    # Assert debug logs recorded extraction steps
    logs = caplog.text
    assert "Built LeafTaxonomy" in logs


def test_test_material_document_extraction(tmp_path):
    candidates = [
        config.VAULT_ROOT / "raw" / "test-material" / "test-document.md",
        config.VAULT_ROOT
        / "sample-llm-wiki"
        / "raw"
        / "test-material"
        / "test-document.md",
        Path(__file__).resolve().parent.parent
        / "sample-llm-wiki"
        / "raw"
        / "test-material"
        / "test-document.md",
        config.VAULT_ROOT / "llm-wiki" / "raw" / "test-material" / "test-document.md",
        Path(__file__).resolve().parent.parent
        / "llm-wiki"
        / "raw"
        / "test-material"
        / "test-document.md",
    ]
    test_doc = next((p for p in candidates if p.is_file()), None)
    assert test_doc is not None, "Test material test-document.md must exist"

    content = test_doc.read_text(encoding="utf-8")
    mock_model = _MockRelModel(
        [
            json.dumps(
                [
                    {
                        "subject": "Tautological tests",
                        "predicate": "is_a",
                        "object": "source code",
                    },
                    {
                        "subject": "source code",
                        "predicate": "partOf",
                        "object": "coding agents",
                    },
                ]
            )
        ]
    )
    nlp = build_leaf_pipeline(model=mock_model, cache_dir=tmp_path / "cache")
    doc = nlp(content)
    assert isinstance(doc, Doc)

    assert len(doc.ents) >= 2
    assert any(
        e.label_
        in {"concepts", "entities", "claims", "invariants", "PERSON", "CARDINAL"}
        for e in doc.ents
    )

    tax = extract_leaf_taxonomy(doc)
    assert isinstance(tax, LeafTaxonomy)
    assert len(tax.triples) == 2
    assert all(t.predicate.namespace == "predicates" for t in tax.triples)


def test_custom_configured_namespaces_and_predicates(monkeypatch, tmp_path):
    monkeypatch.setattr(
        config, "TAXONOMY_NAMESPACES", ("entities", "concepts", "tactics", "predicates")
    )
    monkeypatch.setattr(config, "BASE_PREDICATES", ("subClassOf", "countered_by"))

    # Valid custom namespace
    node = ExtractedNode(
        name="Zone Defense", namespace="tactics", definition="Defensive formation."
    )
    assert node.namespace == "tactics"

    # Freeform unconfigured namespace allowed
    freeform_node = ExtractedNode(
        name="Custom Mechanism", namespace="bogus", definition="Freeform namespace."
    )
    assert freeform_node.namespace == "bogus"

    # Reject empty namespace
    with pytest.raises(ValueError, match="Namespace cannot be empty"):
        ExtractedNode(name="Bad Node", namespace="   ", definition="Empty namespace.")

    # Base predicates dynamically recognized in normalize_predicate
    from yaw.nlp import normalize_predicate

    assert normalize_predicate("countered_by") == "countered_by"
    assert normalize_predicate("subClassOf") == "subClassOf"
    assert normalize_predicate("sub_class_of") == "subClassOf"

    nlp = build_leaf_pipeline(model=_MockRelModel([]), cache_dir=tmp_path / "cache")
    assert "llm_raw" in nlp.pipe_names


def test_open_evolving_namespaces_accepted_when_unconfigured(monkeypatch):
    from yaw.nlp import map_namespace

    monkeypatch.setattr(config, "TAXONOMY_NAMESPACES", ())
    node = ExtractedNode(
        name="AlphaZero", namespace="algorithms", definition="RL engine."
    )
    assert node.namespace == "algorithms"

    nlp = spacy.blank("en")
    doc = nlp("AlphaZero")
    ent = doc.char_span(0, 9, label="algorithms")
    assert ent is not None
    assert map_namespace(ent) == "algorithms"


def test_format_proposition_and_similarity_matrix():
    from yaw.nlp import embeddings as embedding_gemma

    prop = embedding_gemma.format_proposition(
        "Albert Einstein", "worked at", "Princeton University"
    )
    assert prop == "Albert Einstein worked at Princeton University."

    empty_mat = embedding_gemma.compute_similarity_matrix([])
    assert empty_mat.shape == (0, 0)

    vecs = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    sim = embedding_gemma.compute_similarity_matrix(vecs)
    assert sim.shape == (3, 3)
    assert np.isclose(sim[0, 1], 1.0)
    assert np.isclose(sim[0, 2], 0.0)


def test_doc_to_taxonomy_deduplicates_intra_chunk_triples():
    from yaw.nlp import doc_to_taxonomy

    if not Doc.has_extension("rel"):
        Doc.set_extension("rel", default=[])
    nlp = spacy.blank("en")
    doc = nlp("Alpha computes Beta. Alpha calculates Beta.")
    doc.ents = [
        spacy.tokens.Span(doc, 0, 1, label="entities"),
        spacy.tokens.Span(doc, 2, 3, label="entities"),
    ]
    doc._.rel = [
        SimpleNamespace(dep=0, dest=1, relation="computes"),
        SimpleNamespace(dep=0, dest=1, relation="computes"),
    ]
    tax = doc_to_taxonomy(doc)
    assert len(tax.triples) == 1
    assert tax.triples[0].predicate.name == "compute"


def test_entity_typing_without_sentence_contamination():
    from yaw.nlp import doc_to_taxonomy

    nlp = spacy.load("en_core_web_lg")
    doc = nlp("Albert Einstein worked at Princeton University in New Jersey.")
    tax = doc_to_taxonomy(doc)

    node_names = {n.name: n for n in tax.nodes}
    assert "Albert Einstein" in node_names
    assert "Princeton University" in node_names
    assert "New Jersey" in node_names

    # Distinct definitions, not identical raw sentence across all nodes
    defs = {
        node_names[k].definition
        for k in ("Albert Einstein", "Princeton University", "New Jersey")
    }
    assert len(defs) == 3
    assert not any(
        n.definition == "Albert Einstein worked at Princeton University in New Jersey."
        for n in tax.nodes
    )
    assert "person" in node_names["Albert Einstein"].definition.lower()
    assert "org" in node_names["Princeton University"].definition.lower()
    assert "gpe" in node_names["New Jersey"].definition.lower()


def test_textcat_v3_pipe_creation_and_classification():
    class _MockTextCatModel:
        @property
        def context_length(self):
            return 4096

        def __call__(self, prompt_batches):
            for prompts in prompt_batches:
                res = []
                for p in prompts:
                    if "Classif" in p:
                        res.append("concepts")
                    else:
                        res.append("")
                yield res

    text = "Vector embeddings and personalized pagerank form neural knowledge."
    model = _MockTextCatModel()
    nlp = build_leaf_pipeline(
        model=model,
        textcat_model=model,
        textcat_labels=["entities", "concepts", "decisions"],
        labels=[],
        use_llm_if_ner_below=0,
    )
    assert "llm_textcat" in nlp.pipe_names

    doc = nlp(text)
    assert doc.cats is not None
    assert doc.cats.get("concepts") == 1.0
    assert max(doc.cats, key=lambda k: float(doc.cats.get(k) or 0.0)) == "concepts"

    tax = doc._.leaf_taxonomy
    assert isinstance(tax, LeafTaxonomy)
    assert tax.category == "concepts"

    from yaw.nlp import map_namespace

    span = doc[0:2]
    span.label_ = "entities"
    assert map_namespace(span) == "entities"
    span._.category = "concepts"
    assert map_namespace(span) == "concepts"


def test_normalize_predicate_open_ie_and_base_predicates():
    from yaw.nlp import normalize_predicate

    # Empty string handling
    assert normalize_predicate("") == ""
    assert normalize_predicate("   ") == ""

    # Open IE lemmatization: verbs, inflections, snake_case, camelCase
    assert normalize_predicate("DEVELOPED_BY", base_predicates=()) == "develop by"
    assert normalize_predicate("ranks", base_predicates=()) == "rank"
    assert normalize_predicate("composed_of", base_predicates=()) == "compose of"
    assert normalize_predicate("runs_on", base_predicates=()) == "run on"
    assert normalize_predicate("partOf", base_predicates=()) == "part of"
    assert normalize_predicate("inventedBy", base_predicates=()) == "invent by"

    # Preserves configured base predicates when present
    bp = ("is_a", "partOf", "subClassOf")
    assert normalize_predicate("is_a", base_predicates=bp) == "is_a"
    assert normalize_predicate("partOf", base_predicates=bp) == "partOf"
    assert normalize_predicate("sub_class_of", base_predicates=bp) == "subClassOf"
    # Unmatched predicate in base_predicates mode still gets lemmatized
    assert normalize_predicate("invented_by", base_predicates=bp) == "invent by"


def test_open_ie_pipeline_empty_labels_normalizes_freeform_relations(tmp_path):
    from yaw.nlp import build_leaf_pipeline

    class _OpenIERelModel:
        context_length = 4096

        def __call__(self, prompt_batches):
            for prompts in prompt_batches:
                res = []
                for _ in prompts:
                    res.append(
                        json.dumps(
                            [
                                {
                                    "subject": "Albert Einstein",
                                    "predicate": "invented_by",
                                    "object": "Princeton University",
                                },
                                {
                                    "subject": "Princeton University",
                                    "predicate": "runs_on",
                                    "object": "Albert Einstein",
                                },
                            ]
                        )
                    )
                yield res

    nlp = build_leaf_pipeline(
        model=_OpenIERelModel(),
        cache_dir=tmp_path / "cache",
        labels=[],
    )
    doc = nlp("Albert Einstein visited Princeton University.")
    tax = doc._.leaf_taxonomy
    assert tax is not None
    assert len(tax.triples) == 2
    preds = {t.predicate.name for t in tax.triples}
    assert "invent by" in preds
    assert "run on" in preds


def test_strategy_b_clean_entity_extraction_and_markdown_stripping():
    from yaw.nlp import doc_to_taxonomy, extract_leaf_doc
    from yaw.nlp.utils import is_valid_entity_text

    # 1. markdown_structure detects fenced blocks and inline code
    raw_md = (
        "# Header\n"
        "Prose before code.\n"
        "```python\n"
        "for idx, embedding in enumerate(embeddings):\n"
        '    print(f"Embedding {idx+1}: {embedding.shape}")\n'
        "```\n"
        "Prose after with inline `embedding.shape` reference.\n"
    )
    nlp = spacy.blank("en")
    nlp.add_pipe("markdown_structure")
    doc = nlp(raw_md)
    assert "code" in doc.spans
    assert len(doc.spans["code"]) == 2
    code_tokens = [t.text for t in doc if t._.is_code]
    assert any("print" in t for t in code_tokens)
    assert "embedding.shape" in code_tokens

    # 2. Invariant E1: doc.text is unmodified raw_md (exact byte provenance)
    annotated = extract_leaf_doc(raw_md, nlp=nlp)
    assert annotated.text == raw_md

    # 3. is_valid_entity_text allows human phrases and dots/underscores in legitimate names
    assert is_valid_entity_text("EmbeddingGemma")
    assert is_valid_entity_text("Vector Search")
    assert is_valid_entity_text("Albert Einstein")
    assert is_valid_entity_text("Princeton University")
    assert is_valid_entity_text('  "EmbeddingGemma"  ')
    assert is_valid_entity_text("Node.js")
    assert is_valid_entity_text("scikit_learn")
    assert not is_valid_entity_text("1")
    assert not is_valid_entity_text("2026")
    assert not is_valid_entity_text("---")
    assert not is_valid_entity_text('print(f"Embedding {idx+1}: {embedding.shape}")')

    # 4. doc_to_taxonomy drops candidate entities that overlap with code spans
    doc_with_ents = nlp(raw_md)
    span_valid = doc_with_ents.char_span(2, 8, label="concepts")
    code_start = raw_md.find("print")
    span_in_code = doc_with_ents.char_span(code_start, code_start + 5, label="concepts")
    inline_start = raw_md.find("embedding.shape", raw_md.find("inline"))
    span_inline_code = doc_with_ents.char_span(
        inline_start, inline_start + len("embedding.shape"), label="concepts"
    )

    candidates = [
        s for s in (span_valid, span_in_code, span_inline_code) if s is not None
    ]
    doc_with_ents.ents = candidates
    tax = doc_to_taxonomy(doc_with_ents)
    node_names = [n.name for n in tax.nodes]
    assert "Header" in node_names
    assert "print" not in node_names
    assert "embedding.shape" not in node_names


def test_invariant_e1_e2_store_chunk_concept_edges_provenance_and_no_fallback():
    from yaw.indexer.pipeline import MarkdownChunk, _store_chunk_concept_edges

    chunks = [
        MarkdownChunk(
            "c0", "H0", "h0", "Alpha text with no concept name.", 0, 0, 32, 1, 2
        ),
        MarkdownChunk(
            "c1", "H1", "h1", "Beta text contains Encoder clearly.", 1, 33, 35, 3, 4
        ),
        MarkdownChunk(
            "c2",
            "H2",
            "h2",
            "Gamma text with completely unrelated content.",
            2,
            69,
            45,
            5,
            6,
        ),
    ]

    from types import SimpleNamespace

    target_rows: list[Any] = []
    target = SimpleNamespace(
        rows=target_rows, declare_row=lambda *, row: target_rows.append(row)
    )
    seen: set[str] = set()

    # Case 1: Node with provenance matching chunk 0 only, but whose text does not appear in chunk 0
    raw_nodes = {
        "concepts:transformer": (
            "raw/test.md#chunk:0",
            ExtractedNode(
                name="Transformer", namespace="concepts", definition="Neural net"
            ),
        )
    }
    node_prov = {"concepts:transformer": {"raw/test.md#chunk:0"}}
    _store_chunk_concept_edges(
        raw_nodes,
        chunks,
        "raw/test.md",
        cast(Any, target),
        seen,
        node_provenance=node_prov,
    )

    # Must link ONLY to chunk 0; MUST NOT fallback to linking to all chunks
    manifest_chunks = [
        r.object_id  # type: ignore[attr-defined]
        for r in target.rows
        if r.predicate == "manifests_in"  # type: ignore[attr-defined]
    ]
    assert manifest_chunks == ["raw/test.md#chunk:0"], (
        f"Expected grounding only to chunk:0, got: {manifest_chunks}"
    )

    # Case 2: Node matching substring in chunk 1 with no provenance
    target.rows.clear()
    seen.clear()
    raw_nodes_2 = {
        "concepts:encoder": (
            "",
            ExtractedNode(name="Encoder", namespace="concepts", definition="Module"),
        )
    }
    _store_chunk_concept_edges(
        raw_nodes_2, chunks, "raw/test.md", cast(Any, target), seen, node_provenance={}
    )
    manifest_chunks_2 = [
        r.object_id  # type: ignore[attr-defined]
        for r in target.rows
        if r.predicate == "manifests_in"  # type: ignore[attr-defined]
    ]
    assert manifest_chunks_2 == ["raw/test.md#chunk:1"]

    # Case 3 (Critical Invariant E1/E2): Ghost node with neither provenance nor substring match
    target.rows.clear()
    seen.clear()
    raw_nodes_3 = {
        "concepts:ghost": (
            "",
            ExtractedNode(
                name="Ghost", namespace="concepts", definition="Non-existent concept"
            ),
        )
    }
    _store_chunk_concept_edges(
        raw_nodes_3, chunks, "raw/test.md", cast(Any, target), seen, node_provenance={}
    )
    assert len(target.rows) == 0, (
        f"Critical Invariant E1/E2 violated: ghost node should produce 0 edges, got {len(target.rows)}"
    )


def test_batch_cache_persists_and_replays_without_network(tmp_path: Path):
    """Verify spacy.BatchCache.v1 caches LLM calls to disk and replays identically without invoking model."""
    call_count = 0

    class _CountingMockModel:
        context_length = 4096

        def __call__(self, prompt_batches):
            nonlocal call_count
            for prompts in prompt_batches:
                res = []
                for _ in prompts:
                    call_count += 1
                    res.append(
                        json.dumps(
                            [
                                {
                                    "subject": "FastMCP",
                                    "predicate": "powers",
                                    "object": "knowledge graphs",
                                }
                            ]
                        )
                    )
                yield res

    cache_dir = tmp_path / "spacy_cache"
    nlp = build_leaf_pipeline(
        model=_CountingMockModel(),
        cache_dir=cache_dir,
        cache_batch_size=1,
    )

    text = "FastMCP powers knowledge graphs effectively."

    # First call: cache miss, model invoked
    doc1 = nlp(text)
    assert call_count == 1
    assert doc1._.llm_reply is not None
    assert doc1._.leaf_taxonomy is not None
    assert len(doc1._.leaf_taxonomy.triples) == 1

    # Verify cache files were persisted to disk
    index_file = cache_dir / "index.jsonl"
    assert index_file.is_file()
    assert index_file.stat().st_size > 0

    # Second call with identical text: cache hit, model NOT invoked!
    doc2 = nlp(text)
    assert call_count == 1, f"Expected cache hit (call_count=1), but got {call_count}"
    assert doc2._.llm_reply is not None
    assert doc2._.leaf_taxonomy is not None
    assert len(doc2._.leaf_taxonomy.triples) == 1

    # Third call on a brand new pipeline instance with same cache_dir: cross-instance disk hit!
    nlp2 = build_leaf_pipeline(
        model=_CountingMockModel(),
        cache_dir=cache_dir,
        cache_batch_size=1,
    )
    doc3 = nlp2(text)
    assert call_count == 1, (
        f"Expected cross-instance disk hit (call_count=1), but got {call_count}"
    )
    assert doc3._.leaf_taxonomy is not None
    assert len(doc3._.leaf_taxonomy.triples) == 1


def test_extract_json_payload_robustness():
    """Verify extract_json_payload handles multiple formats, fences, stream objects, and edge cases."""
    from yaw.nlp.extractor import extract_json_payload

    # 1. Clean JSON array
    clean_arr = '[{"subject": "A", "predicate": "P", "object": "B"}]'
    assert extract_json_payload(clean_arr) == [
        {"subject": "A", "predicate": "P", "object": "B"}
    ]

    # 2. Markdown fenced array with ```json and whitespace
    fenced_arr = '```json\n[{"subject": "A", "predicate": "P", "object": "B"}]\n```'
    assert extract_json_payload(fenced_arr) == [
        {"subject": "A", "predicate": "P", "object": "B"}
    ]

    # 3. Quadruple backticks fenced array
    quad_arr = '````json\n[{"subject": "A", "predicate": "P", "object": "B"}]\n````'
    assert extract_json_payload(quad_arr) == [
        {"subject": "A", "predicate": "P", "object": "B"}
    ]

    # 4. Single JSON object
    single_obj = '{"subject": "A", "predicate": "P", "object": "B"}'
    assert extract_json_payload(single_obj) == [
        {"subject": "A", "predicate": "P", "object": "B"}
    ]

    # 5. Object wrapping "triples" key
    wrapped_obj = '{"triples": [{"subject": "A", "predicate": "P", "object": "B"}]}'
    assert extract_json_payload(wrapped_obj) == [
        {"subject": "A", "predicate": "P", "object": "B"}
    ]

    # 6. JSON lines stream
    json_lines = '{"subject": "A", "predicate": "P", "object": "B"}\n{"subject": "C", "predicate": "Q", "object": "D"}'
    assert len(extract_json_payload(json_lines)) == 2

    # 7. Commentary / conversational noise before and after
    noisy = 'Here are the extracted triples:\n[{"subject": "A", "predicate": "P", "object": "B"}]\nHope this helps!'
    assert extract_json_payload(noisy) == [
        {"subject": "A", "predicate": "P", "object": "B"}
    ]

    # 8. Edge cases: None, empty string, whitespace, non-JSON text
    assert extract_json_payload(None) == []
    assert extract_json_payload("") == []
    assert extract_json_payload("   \n\t  ") == []
    assert (
        extract_json_payload("Sorry, I could not find any relationships in the text.")
        == []
    )


def test_normalize_predicate_non_recursive_and_vector_sim():
    """Verify normalize_predicate makes zero recursive LLM calls and utilizes GloVe vector cosine similarity."""
    from yaw.nlp import normalize_predicate

    # 1. Zero recursion verification: pass pipeline with llm_raw and assert model is never called
    model_called = False

    class _FailIfCalledModel:
        context_length = 4096

        def __call__(self, prompt_batches):
            nonlocal model_called
            model_called = True
            raise AssertionError("normalize_predicate must NEVER invoke the LLM model!")

    nlp = build_leaf_pipeline(model=_FailIfCalledModel())
    res = normalize_predicate("inventedBy", base_predicates=(), nlp=nlp)
    assert res == "invent by"
    assert not model_called

    # 2. Vector similarity verification using synthetic vectors in vocab
    base_nlp = spacy.blank("en")
    base_nlp.vocab.reset_vectors(width=3)
    v_sim = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v_base = np.array([0.95, 0.05, 0.0], dtype=np.float32)
    base_nlp.vocab.set_vector("similar predicate", cast(Any, v_sim))
    base_nlp.vocab.set_vector("base predicate", cast(Any, v_base))

    matched = normalize_predicate(
        "similar predicate",
        base_predicates=("base_predicate", "other_predicate"),
        nlp=base_nlp,
        threshold=0.80,
    )
    assert matched == "base_predicate"


def test_all_17_predicate_aliases_normalize_correctly():
    """Assert all 17 predicate aliases in PREDICATE_ALIASES normalize correctly across formats."""
    from yaw.nlp.extractor import PREDICATE_ALIASES, normalize_predicate

    bp = ("is_a", "partOf", "subClassOf")
    assert len(PREDICATE_ALIASES) == 17

    for alias, expected in PREDICATE_ALIASES.items():
        # Verbatim alias
        assert normalize_predicate(alias, base_predicates=bp) == expected, (
            f"Failed for alias {alias!r}: expected {expected}"
        )
        # Underscored variant
        underscored = alias.replace(" ", "_")
        assert normalize_predicate(underscored, base_predicates=bp) == expected, (
            f"Failed for underscored {underscored!r}: expected {expected}"
        )
        # Spaced variant
        spaced = alias.replace("_", " ")
        assert normalize_predicate(spaced, base_predicates=bp) == expected, (
            f"Failed for spaced {spaced!r}: expected {expected}"
        )


def test_multi_word_predicate_vector_matching_doc_vectors():
    """Verify multi-word predicate vector matching using Doc.vector instead of single vocab Lexeme."""
    from thinc.types import Floats1d

    from yaw.nlp.extractor import normalize_predicate

    nlp = spacy.blank("en")
    nlp.vocab.reset_vectors(width=300)
    vec = np.zeros(300, dtype=np.float32)
    vec[0] = 1.0
    nlp.vocab.set_vector("composed", cast(Floats1d, vec))
    nlp.vocab.set_vector("part", cast(Floats1d, vec))

    res = normalize_predicate(
        "composed of", base_predicates=("partOf",), nlp=nlp, threshold=0.80
    )
    assert res == "partOf"


def test_entity_text_whitespace_and_newline_cleaning():
    """Verify entity texts with trailing newlines or multiline strings are cleaned and pass validation."""
    from yaw.nlp.extractor import parse_raw_triples

    raw = """[
        {
            "subject": "Title\\n",
            "predicate": "relates to",
            "object": "High Temperature Superconductivity\\nWerner Heisenberg"
        }
    ]"""
    triples = parse_raw_triples(raw)
    assert len(triples) == 1
    assert triples[0].subject.name == "Title"
    assert (
        triples[0].object.name == "High Temperature Superconductivity Werner Heisenberg"
    )


def test_named_entity_namespace_resolution():
    """Verify named entities (e.g. Werner Heisenberg) resolve to 'entities' namespace."""
    from yaw.nlp.extractor import parse_raw_triples

    raw = """[
        {
            "subject": "Werner Heisenberg",
            "predicate": "relates to",
            "object": "Erwin Schrodinger",
            "subject_type": "concepts",
            "object_type": "concepts"
        }
    ]"""
    triples = parse_raw_triples(raw)
    assert len(triples) == 1
    assert triples[0].subject.name == "Werner Heisenberg"
    assert triples[0].subject.namespace == "entities"
    assert triples[0].object.name == "Erwin Schrodinger"
    assert triples[0].object.namespace == "entities"


def test_extract_json_payload_trailing_commas_and_single_quotes():
    """Verify extract_json_payload handles trailing commas and single quotes via literal_eval fallback."""
    from yaw.nlp.extractor import extract_json_payload

    trailing_comma = '[{"subject": "A", "predicate": "B", "object": "C",}]'
    res = extract_json_payload(trailing_comma)
    assert len(res) == 1
    assert res[0]["subject"] == "A"

    single_quotes = "[{'subject': 'A', 'predicate': 'B', 'object': 'C'}]"
    res2 = extract_json_payload(single_quotes)
    assert len(res2) == 1
    assert res2[0]["subject"] == "A"
