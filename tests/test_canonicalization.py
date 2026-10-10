"""Invariant and regression tests for Epic 5 Sprint 5A: Entity Disambiguation and Canonicalization (EDC).

Validates the 5 Grounded Invariants:
- Invariant 5A.1: Lexical Fast-Path (Exact slug / active alias hit & SequenceMatcher >= 0.85)
- Invariant 5A.2: Semantic Vector Canopy (Cosine similarity >= 0.95 auto-merge, 0 LLM calls)
- Invariant 5A.3: Gray-Zone Adjudication (Cosine similarity in [0.80, 0.95) & permanent canon_verdicts logging)
- Invariant 5A.4: Non-Destructive View Re-Routing & Reversibility (Raw tables immutable, delete_alias reverts)
- Invariant 5A.5: Namespace Inheritance (Alias endpoints strictly inherit canonical namespace)
- Smoke Directive: spacy.Raw.v1 defensive code-fence stripping, whitespace handling, and doc.char_span anchoring
"""

from __future__ import annotations

import json
import re
from typing import cast
from urllib.request import Request, urlopen

import numpy as np
import spacy
from thinc.types import Floats1d

from yaw import config
from yaw.indexer import (
    AliasStatus,
    MatchMethod,
    ResolutionResult,
    TierEnum,
    resolve,
)
from yaw.nlp import (
    HYPERNYM_INFERENCE_TEMPLATE,
    OPEN_IE_RAW_TRIPLES_TEMPLATE,
    parse_raw_triples,
    parse_raw_triples_json,
)
from yaw.nlp.extractor import get_base_spacy
from yaw.storage import (
    delete_alias,
    insert_alias,
    set_alias_status,
)


def _make_orthogonal_vectors(
    dim: int = 768, target_cosine: float = 0.97
) -> tuple[np.ndarray, np.ndarray]:
    """Generate two unit vectors with exact target cosine similarity."""
    rng = np.random.default_rng(42)
    v1 = rng.standard_normal(dim).astype(np.float32)
    v1 /= np.linalg.norm(v1)

    v_rand = rng.standard_normal(dim).astype(np.float32)
    v_perp = v_rand - np.dot(v_rand, v1) * v1
    v_perp /= np.linalg.norm(v_perp)

    alpha = float(target_cosine)
    beta = float(np.sqrt(max(0.0, 1.0 - alpha**2)))
    v2 = alpha * v1 + beta * v_perp
    v2 /= np.linalg.norm(v2)
    return v1, v2


# ==============================================================================
# INVARIANT 5A.1: LEXICAL FAST-PATH
# ==============================================================================


def test_invariant_5a1_lexical_exact_alias_hit(test_db, mock_ollama_requests):
    """Exact slug hit against known active alias returns immediately with zero network calls."""
    insert_alias(
        conn=test_db,
        alias_id="concept:shannon-entropy",
        kind="entity",
        canonical_id="concepts:entropy",
        method="lexical",
        score=1.0,
        status="active",
    )
    test_db.commit()

    initial_req_count = len(mock_ollama_requests)

    res = resolve(name="shannon-entropy", kind="entity", conn=test_db)

    assert isinstance(res, ResolutionResult)
    assert res.canonical_id == "concepts:entropy"
    assert res.method == MatchMethod.ALIAS.value
    assert res.status == AliasStatus.ACTIVE.value
    assert res.tier == TierEnum.TIER_1_EXACT
    assert res.score == 1.0
    assert res.is_new is False

    # Zero vector/LLM requests invoked
    assert len(mock_ollama_requests) == initial_req_count


def test_invariant_5a1_lexical_exact_node_hit(test_db, mock_ollama_requests):
    """Exact canonical node match in leaf_nodes returns instant resolution with zero network calls."""
    test_db.execute(
        """
        INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition)
        VALUES ('concepts:entropy', 'note_entropy.md', 'entropy', 'concepts', 'Measure of uncertainty')
        """
    )
    test_db.commit()

    initial_req_count = len(mock_ollama_requests)

    res = resolve(name="entropy", namespace="concepts", conn=test_db)

    assert res.canonical_id == "concepts:entropy"
    assert res.method == MatchMethod.EXACT.value
    assert res.tier == TierEnum.TIER_1_EXACT
    assert res.is_new is False
    assert len(mock_ollama_requests) == initial_req_count

    # Full URI invocation also hits exact match
    res_full = resolve(name="concepts:entropy", kind="entity", conn=test_db)
    assert res_full.canonical_id == "concepts:entropy"
    assert res_full.method == MatchMethod.EXACT.value
    assert len(mock_ollama_requests) == initial_req_count


def test_invariant_5a2_near_match_vector_consolidation(test_db, mock_ollama_requests):
    """Near match with vector similarity > CANON_SIM_THRESHOLD (0.80) consolidates via Tier 2."""
    v_canon, v_cand = _make_orthogonal_vectors(dim=768, target_cosine=0.88)
    test_db.execute(
        """
        INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition, embedding)
        VALUES ('concepts:entropy', 'note_entropy.md', 'entropy', 'concepts', 'Thermodynamic entropy', ?)
        """,
        (v_canon.astype(np.float32).tobytes(),),
    )
    test_db.commit()

    initial_req_count = len(mock_ollama_requests)

    res = resolve(
        name="entrophy",
        kind="entity",
        conn=test_db,
        embedding=v_cand.tolist(),
    )

    assert res.canonical_id == "concepts:entropy"
    assert res.method == MatchMethod.AUTO.value
    assert res.status == AliasStatus.ACTIVE.value
    assert res.tier == TierEnum.TIER_2_EMBEDDING
    assert res.score >= config.CANON_SIM_THRESHOLD
    assert res.is_new is False

    # Row inserted in aliases table with auto method
    alias_row = test_db.execute(
        "SELECT canonical_id, method, status FROM aliases WHERE alias_id = 'concepts:entrophy' AND kind = 'entity'"
    ).fetchone()
    assert alias_row is not None
    assert alias_row[0] == "concepts:entropy"
    assert alias_row[1] == "auto"
    assert alias_row[2] == "active"

    # Zero LLM requests
    assert len(mock_ollama_requests) == initial_req_count


def test_invariant_5a1_lexical_predicate_fastpath(test_db, mock_ollama_requests):
    """Predicate resolution fast-paths exact base predicates and lexical typos."""
    test_db.execute(
        """
        INSERT INTO leaf_relations (relation_id, source_leaf_id, subject_id, predicate, object_id)
        VALUES ('rel_1', 'note_1.md', 'concepts:a', 'subClassOf', 'concepts:b')
        """
    )
    test_db.commit()

    initial_req_count = len(mock_ollama_requests)

    # Exact predicate resolution
    res_exact = resolve(name="subClassOf", kind="predicate", conn=test_db)
    assert (
        res_exact.canonical_id in config.BASE_PREDICATES
        or res_exact.canonical_id == "subClassOf"
    )
    assert res_exact.tier in (TierEnum.TIER_1_EXACT, TierEnum.TIER_1_LEXICAL)

    # Typo predicate matches lexical ratio >= 0.85 against 'subClassOf'
    res_typo = resolve(name="subclassof", kind="predicate", conn=test_db)
    assert "subclass" in res_typo.canonical_id.lower()
    assert res_typo.tier in (TierEnum.TIER_1_EXACT, TierEnum.TIER_1_LEXICAL)

    assert len(mock_ollama_requests) == initial_req_count


# ==============================================================================
# INVARIANT 5A.2: SEMANTIC VECTOR CANOPY
# ==============================================================================


def test_invariant_5a2_semantic_vector_canopy_auto_merge(test_db, mock_ollama_requests):
    """Cosine similarity >= CANON_SIM_HIGH (0.95) auto-merges with method='auto' and zero LLM calls."""
    v_canon, v_cand = _make_orthogonal_vectors(dim=768, target_cosine=0.97)

    # Seed canonical node with high-dimensional embedding
    test_db.execute(
        """
        INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition, embedding)
        VALUES ('concepts:space-time-curvature', 'note_relativity.md', 'space-time-curvature', 'concepts',
                'Curvature of four-dimensional space-time under general relativity', ?)
        """,
        (v_canon.astype(np.float32).tobytes(),),
    )
    test_db.commit()

    initial_req_count = len(mock_ollama_requests)

    # 'space-time-warp' has lexical similarity < 0.85 with 'space-time-curvature', but cosine 0.97 >= 0.95
    res = resolve(
        name="space-time-warp",
        kind="entity",
        definition_ctx="Warping of spacetime continuum",
        conn=test_db,
        embedding=v_cand.tolist(),
    )

    assert res.canonical_id == "concepts:space-time-curvature"
    assert res.method == MatchMethod.AUTO.value
    assert res.status == AliasStatus.ACTIVE.value
    assert res.tier == TierEnum.TIER_2_EMBEDDING
    assert res.score >= config.CANON_SIM_HIGH
    assert res.is_new is False

    # Alias record stored with method='auto'
    alias_row = test_db.execute(
        "SELECT canonical_id, method, status FROM aliases WHERE alias_id = 'concepts:space-time-warp'"
    ).fetchone()
    assert alias_row is not None
    assert alias_row[0] == "concepts:space-time-curvature"
    assert alias_row[1] == "auto"

    # Zero LLM chat requests triggered
    assert len(mock_ollama_requests) == initial_req_count
    chat_requests = [
        r for r in mock_ollama_requests if "messages" in r or "prompt" in r
    ]
    assert len(chat_requests) == 0


# ==============================================================================
# INVARIANT 5A.3: GRAY-ZONE ADJUDICATION & VERDICT LOGGING
# ==============================================================================


def test_invariant_5a2_vector_similarity_above_threshold_auto_merges_without_llm(
    test_db, mock_ollama_requests
):
    """Cosine similarity > CANON_SIM_THRESHOLD (0.80) consolidates via Tier 2 near match with zero LLM calls."""
    v_canon, v_cand = _make_orthogonal_vectors(dim=768, target_cosine=0.88)

    test_db.execute(
        """
        INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition, embedding)
        VALUES ('concepts:information-theory', 'note_info.md', 'information-theory', 'concepts',
                'Mathematical treatment of communication and information storage', ?)
        """,
        (v_canon.astype(np.float32).tobytes(),),
    )
    test_db.commit()

    initial_req_count = len(mock_ollama_requests)

    # Surface name 'communication-theory' has cosine ~ 0.88 >= 0.80 threshold
    res = resolve(
        name="communication-theory",
        kind="entity",
        definition_ctx="Mathematical theory of signal communication",
        conn=test_db,
        embedding=v_cand.tolist(),
    )

    assert res.canonical_id == "concepts:information-theory"
    assert res.method == MatchMethod.AUTO.value
    assert res.tier == TierEnum.TIER_2_EMBEDDING
    assert res.status == AliasStatus.ACTIVE.value
    assert res.score >= config.CANON_SIM_THRESHOLD
    assert res.is_new is False

    # Verify alias created with method='auto'
    alias_row = test_db.execute(
        "SELECT canonical_id, method FROM aliases WHERE alias_id = 'concepts:communication-theory'"
    ).fetchone()
    assert alias_row is not None
    assert alias_row[0] == "concepts:information-theory"
    assert alias_row[1] == "auto"

    # Zero LLM chat requests triggered
    assert len(mock_ollama_requests) == initial_req_count


def test_fallback_vector_similarity_below_threshold_creates_new_node(
    test_db, mock_ollama_requests
):
    """Cosine similarity below CANON_SIM_THRESHOLD (0.80) falls back to new canonical node with zero LLM calls."""
    v_canon, v_cand = _make_orthogonal_vectors(dim=768, target_cosine=0.65)

    test_db.execute(
        """
        INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition, embedding)
        VALUES ('concepts:thermodynamics', 'note_thermo.md', 'thermodynamics', 'concepts',
                'Branch of physics dealing with heat and temperature', ?)
        """,
        (v_canon.astype(np.float32).tobytes(),),
    )
    test_db.commit()

    initial_req_count = len(mock_ollama_requests)

    res = resolve(
        name="cryogenics",
        kind="entity",
        definition_ctx="Production and behavior of materials at extremely low temperatures",
        conn=test_db,
        embedding=v_cand.tolist(),
    )

    assert res.is_new is True
    assert res.method == MatchMethod.NEW.value
    assert res.tier == TierEnum.NEW_INDUCTION
    assert res.canonical_id == "concepts:cryogenics"
    assert res.score == 1.0

    # No alias created pointing to thermodynamics
    alias_row = test_db.execute(
        "SELECT canonical_id FROM aliases WHERE alias_id = 'concepts:cryogenics'"
    ).fetchone()
    assert alias_row is None

    # Zero LLM requests triggered
    assert len(mock_ollama_requests) == initial_req_count


def test_two_tier_convergence_bypasses_llm_completely(test_db, mock_ollama_requests):
    """Two-Tier Convergence Engine resolve() consolidates near-matches via Tier 2 with zero LLM calls."""
    v_canon, v_cand = _make_orthogonal_vectors(dim=768, target_cosine=0.88)

    test_db.execute(
        """
        INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition, embedding)
        VALUES ('concepts:deep-reinforcement-learning', 'note_rl.md', 'deep-reinforcement-learning', 'concepts',
                'Combination of artificial neural networks and reinforcement learning', ?)
        """,
        (v_canon.astype(np.float32).tobytes(),),
    )
    test_db.commit()

    initial_req_count = len(mock_ollama_requests)

    res = resolve(
        name="deep-rl",
        kind="entity",
        definition_ctx="Reinforcement learning with deep function approximators",
        conn=test_db,
        embedding=v_cand.tolist(),
    )

    assert res.canonical_id == "concepts:deep-reinforcement-learning"
    assert res.method == MatchMethod.AUTO.value
    assert res.tier == TierEnum.TIER_2_EMBEDDING
    assert res.score >= config.CANON_SIM_THRESHOLD
    assert res.is_new is False

    # Zero Ollama requests should have been dispatched
    assert len(mock_ollama_requests) == initial_req_count


# ==============================================================================
# INVARIANT 5A.4: NON-DESTRUCTIVE VIEW RE-ROUTING & REVERSIBILITY
# ==============================================================================


def test_invariant_5a4_non_destructive_view_rerouting_and_reversibility(test_db):
    """Raw extraction records remain immutable while unified views rewrite endpoints and revert on delete_alias."""
    # 1. Insert raw mentions into leaf_nodes and leaf_relations
    test_db.execute(
        """
        INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition)
        VALUES ('entities:heisenberg', 'note_quantum.md', 'Heisenberg', 'entities', 'Theoretical physicist'),
               ('concepts:entrophy', 'note_quantum.md', 'Entrophy', 'concepts', 'Surface typo mention')
        """
    )
    test_db.execute(
        """
        INSERT INTO leaf_relations (relation_id, source_leaf_id, subject_id, predicate, object_id, weight)
        VALUES ('rel_101', 'note_quantum.md', 'entities:heisenberg', 'formulated', 'concepts:entrophy', 1.0)
        """
    )
    test_db.commit()

    # Query initial views: reflects raw surface forms
    initial_edge = test_db.execute(
        "SELECT src_id, dst_id, rel_type FROM unified_edges WHERE src_id = 'entities:heisenberg'"
    ).fetchone()
    assert initial_edge == ("entities:heisenberg", "concepts:entrophy", "formulated")

    # 2. Insert active alias rewriting concepts:entrophy -> concepts:entropy
    insert_alias(
        conn=test_db,
        alias_id="concepts:entrophy",
        kind="entity",
        canonical_id="concepts:entropy",
        method="lexical",
        score=0.86,
        status="active",
    )
    test_db.commit()

    # Query unified_edges view: destination dynamically rewritten to canonical URI
    rewritten_edge = test_db.execute(
        "SELECT src_id, dst_id, rel_type FROM unified_edges WHERE src_id = 'entities:heisenberg'"
    ).fetchone()
    assert rewritten_edge == ("entities:heisenberg", "concepts:entropy", "formulated")

    # Query unified_nodes view: node_id dynamically rewritten to canonical URI
    canon_node = test_db.execute(
        "SELECT node_id, namespace FROM unified_nodes WHERE node_id = 'concepts:entropy'"
    ).fetchone()
    assert canon_node is not None
    assert canon_node[0] == "concepts:entropy"
    assert canon_node[1] == "concepts"

    # Query raw substrate: raw records remain completely untouched and immutable!
    raw_rel = test_db.execute(
        "SELECT subject_id, predicate, object_id FROM leaf_relations WHERE relation_id = 'rel_101'"
    ).fetchone()
    assert raw_rel == ("entities:heisenberg", "formulated", "concepts:entrophy")

    raw_node = test_db.execute(
        "SELECT node_id FROM leaf_nodes WHERE node_id = 'concepts:entrophy'"
    ).fetchone()
    assert raw_node == ("concepts:entrophy",)

    # 3. Execute delete_alias: views immediately revert to raw surface forms
    deleted = delete_alias(test_db, alias_id="concepts:entrophy", kind="entity")
    assert deleted is True
    test_db.commit()

    reverted_edge = test_db.execute(
        "SELECT src_id, dst_id, rel_type FROM unified_edges WHERE src_id = 'entities:heisenberg'"
    ).fetchone()
    assert reverted_edge == ("entities:heisenberg", "concepts:entrophy", "formulated")

    reverted_node = test_db.execute(
        "SELECT node_id FROM unified_nodes WHERE node_id = 'concepts:entrophy'"
    ).fetchone()
    assert reverted_node is not None


def test_invariant_5a4_predicate_view_rerouting_and_reversibility(test_db):
    """Predicate alias rewrites rel_type in unified_edges and reversibly restores original predicate on deletion."""
    test_db.execute(
        """
        INSERT INTO leaf_relations (relation_id, source_leaf_id, subject_id, predicate, object_id, weight)
        VALUES ('rel_202', 'note_physics.md', 'concepts:energy', 'has_equivalence', 'concepts:mass', 1.0)
        """
    )
    test_db.commit()

    # Initial view check
    edge_init = test_db.execute(
        "SELECT rel_type FROM unified_edges WHERE src_id = 'concepts:energy'"
    ).fetchone()
    assert edge_init[0] == "has_equivalence"

    # Insert predicate alias
    insert_alias(
        conn=test_db,
        alias_id="has_equivalence",
        kind="predicate",
        canonical_id="relates_to",
        method="lexical",
        score=0.9,
        status="active",
    )
    test_db.commit()

    # Rewritten predicate in view
    edge_rewritten = test_db.execute(
        "SELECT rel_type FROM unified_edges WHERE src_id = 'concepts:energy'"
    ).fetchone()
    assert edge_rewritten[0] == "relates_to"

    # Raw relation predicate remains untouched
    raw_pred = test_db.execute(
        "SELECT predicate FROM leaf_relations WHERE relation_id = 'rel_202'"
    ).fetchone()
    assert raw_pred[0] == "has_equivalence"

    # Delete predicate alias
    deleted = delete_alias(test_db, alias_id="has_equivalence", kind="predicate")
    assert deleted is True
    test_db.commit()

    edge_reverted = test_db.execute(
        "SELECT rel_type FROM unified_edges WHERE src_id = 'concepts:energy'"
    ).fetchone()
    assert edge_reverted[0] == "has_equivalence"


# ==============================================================================
# INVARIANT 5A.5: NAMESPACE INHERITANCE
# ==============================================================================


def test_invariant_5a5_namespace_inheritance(test_db):
    """Surface entity mention 'entities:shannon-entropy' inherits canonical namespace 'concepts:entropy'."""
    test_db.execute(
        """
        INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition)
        VALUES ('concepts:entropy', 'note_core.md', 'Entropy', 'concepts', 'Core concept of information and thermodynamics'),
               ('entities:shannon-entropy', 'note_surface.md', 'Shannon Entropy', 'entities', 'Entity mention in document')
        """
    )
    test_db.execute(
        """
        INSERT INTO leaf_relations (relation_id, source_leaf_id, subject_id, predicate, object_id)
        VALUES ('rel_303', 'note_surface.md', 'entities:shannon-entropy', 'relates_to', 'concepts:probabilistic-model')
        """
    )
    insert_alias(
        conn=test_db,
        alias_id="entities:shannon-entropy",
        kind="entity",
        canonical_id="concepts:entropy",
        method="lexical",
        score=0.92,
        status="active",
    )
    test_db.commit()

    # Query unified_nodes: node_id is concepts:entropy and namespace inherits 'concepts' (not 'entities')
    unified_node = test_db.execute(
        "SELECT node_id, namespace FROM unified_nodes WHERE node_id = 'concepts:entropy'"
    ).fetchone()
    assert unified_node is not None
    assert unified_node[0] == "concepts:entropy"
    assert unified_node[1] == "concepts"

    # Query unified_edges: subject resolves to canonical concepts:entropy
    unified_edge = test_db.execute(
        "SELECT src_id, dst_id FROM unified_edges WHERE dst_id = 'concepts:probabilistic-model'"
    ).fetchone()
    assert unified_edge is not None
    assert unified_edge[0] == "concepts:entropy"


def test_invariant_5a5_slug_match_cross_namespace_inheritance(test_db):
    """Resolve surface mention across namespace boundaries inherits canonical root namespace."""
    test_db.execute(
        """
        INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition)
        VALUES ('concepts:shannon-entropy', 'note_entropy.md', 'Shannon Entropy', 'concepts', 'Information measure')
        """
    )
    test_db.commit()

    # Candidate passed with 'entities' namespace but matching existing slug 'shannon-entropy'
    res = resolve(name="entities:shannon-entropy", kind="entity", conn=test_db)

    assert res.canonical_id == "concepts:shannon-entropy"
    assert res.tier == TierEnum.TIER_1_EXACT

    # Active alias automatically inserted linking entities:shannon-entropy -> concepts:shannon-entropy
    alias_row = test_db.execute(
        "SELECT canonical_id FROM aliases WHERE alias_id = 'entities:shannon-entropy'"
    ).fetchone()
    assert alias_row is not None
    assert alias_row[0] == "concepts:shannon-entropy"


# ==============================================================================
# SMOKE DIRECTIVE: spacy.Raw.v1 TRIPLE PARSING
# ==============================================================================


def test_smoke_directive_code_fences_and_whitespace():
    """Verify parse_raw_triples_json defensively handles markdown fences, quadruple backticks, and whitespace."""
    raw_payload = """
    ```json
    [
      {
        "subject": "Shannon entropy",
        "predicate": "measures",
        "object": "information content",
        "subject_type": "concepts",
        "object_type": "concepts"
      }
    ]
    ```
    """
    triples = parse_raw_triples_json(raw_payload)
    assert len(triples) == 1
    assert triples[0]["subject"] == "Shannon entropy"
    assert triples[0]["object"] == "information content"
    assert triples[0]["subject_type"] == "concepts"

    # Quadruple backticks test
    quad_payload = """
    ````json
    {"subject": "Claude Shannon", "predicate": "authored", "object": "Mathematical Theory of Communication"}
    ````
    """
    triples_quad = parse_raw_triples_json(quad_payload)
    assert len(triples_quad) == 1
    assert triples_quad[0]["subject"] == "Claude Shannon"
    assert triples_quad[0]["object"] == "Mathematical Theory of Communication"


def test_smoke_directive_span_anchoring():
    """Verify spacy.Raw.v1 triple parsing anchors spans via doc.char_span into doc.ents."""
    nlp = spacy.blank("en")
    doc = nlp("Claude Shannon established information theory in 1948.")

    raw_json = json.dumps(
        [
            {
                "subject": "Claude Shannon",
                "predicate": "established",
                "object": "information theory",
                "subject_type": "entities",
                "object_type": "concepts",
            }
        ]
    )

    triples = parse_raw_triples_json(raw_json, doc=doc, nlp=nlp)
    assert len(triples) == 1
    t = triples[0]

    assert "subject_span" in t
    assert "object_span" in t
    s_start, s_end = t["subject_span"]
    o_start, o_end = t["object_span"]

    # Assert exact character alignment
    assert doc.text[s_start:s_end] == "Claude Shannon"
    assert doc.text[o_start:o_end] == "information theory"

    # Assert doc.ents populated
    ent_texts = [e.text for e in doc.ents]
    assert "Claude Shannon" in ent_texts
    assert "information theory" in ent_texts


def test_smoke_directive_json_lines_and_multiline():
    """Verify parse_raw_triples handles JSON lines stream format and multiline spacing."""
    json_lines = """
    {"subject": "Albert Einstein", "predicate": "developed", "object": "General Relativity"}
    {"subject": "Isaac Newton", "predicate": "formulated", "object": "Classical Mechanics"}
    """
    triples = parse_raw_triples(json_lines)
    assert len(triples) == 2
    assert triples[0].subject.name == "Albert Einstein"
    assert triples[0].object.name == "General Relativity"
    assert triples[1].subject.name == "Isaac Newton"
    assert triples[1].object.name == "Classical Mechanics"


def test_smoke_directive_edge_cases_and_malformed():
    """Verify parse_raw_triples defensively handles None, empty, malformed, and missing fields."""
    assert parse_raw_triples(None) == []
    assert parse_raw_triples("") == []
    assert parse_raw_triples("   \n\t  ") == []
    assert parse_raw_triples("This is not valid JSON at all.") == []

    # Incomplete triple missing required fields
    incomplete = '{"subject": "Only Subject"}'
    assert parse_raw_triples(incomplete) == []


# ==============================================================================
# MOCK OLLAMA HANDLER CONFORMANCE & ADJUDICATION
# ==============================================================================


def test_mock_ollama_handler_canonicalization():
    """Verify _MockOllamaHandler responds with valid JSON to /api/chat and /api/generate for canonicalization."""
    # Test /api/chat canonicalization prompt
    payload_chat = {
        "model": config.CHAT_MODEL,
        "messages": [
            {
                "role": "user",
                "content": (
                    "You are an authoritative knowledge graph ontology curator adjudicating entity canonicalization.\n"
                    "Surface Mention: entropy-variant\n"
                    "Candidate Canonical Entity:\n"
                    "- ID: concepts:entropy\n"
                    "- Name: Entropy\n"
                    "Requirements:\n"
                    '1. Respond STRICTLY with JSON: {"match": "concepts:entropy"} if identical concept.'
                ),
            }
        ],
    }
    url_chat = f"{config.OLLAMA_HOST.rstrip('/')}/api/chat"
    req_chat = Request(
        url_chat,
        data=json.dumps(payload_chat).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req_chat) as resp:
        data_chat = json.loads(resp.read().decode())
        content = data_chat["message"]["content"]
        parsed = json.loads(content)
        assert parsed.get("match") == "concepts:entropy"

    # Test /api/generate canonicalization prompt
    payload_gen = {
        "model": config.CHAT_MODEL,
        "prompt": (
            "You are an authoritative knowledge graph ontology curator adjudicating entity canonicalization.\n"
            "Surface Mention: entropy-variant\n"
            "Candidate Canonical Entity:\n"
            "- ID: concepts:entropy\n"
        ),
    }
    url_gen = f"{config.OLLAMA_HOST.rstrip('/')}/api/generate"
    req_gen = Request(
        url_gen,
        data=json.dumps(payload_gen).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req_gen) as resp:
        data_gen = json.loads(resp.read().decode())
        content_gen = data_gen["response"]
        parsed_gen = json.loads(content_gen)
        assert parsed_gen.get("match") == "concepts:entropy"


def test_mock_ollama_handler_open_ie_and_hypernym():
    """Verify _MockOllamaHandler responds to OPEN_IE_RAW_TRIPLES_TEMPLATE and HYPERNYM_INFERENCE_TEMPLATE."""
    # 1. Open IE raw triples template
    prompt_open_ie = (
        f"{OPEN_IE_RAW_TRIPLES_TEMPLATE}\n"
        "Text:\n'''\n[[Claude Shannon]] invented [[Information Theory]] in 1948.\n'''\nTriples:"
    )
    payload_open_ie = {
        "model": config.CHAT_MODEL,
        "messages": [{"role": "user", "content": prompt_open_ie}],
    }
    url_chat = f"{config.OLLAMA_HOST.rstrip('/')}/api/chat"
    req_open_ie = Request(
        url_chat,
        data=json.dumps(payload_open_ie).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req_open_ie) as resp:
        data = json.loads(resp.read().decode())
        content = data["message"]["content"]
        # Code fence stripping handles mock output
        triples = parse_raw_triples_json(content)
        assert len(triples) >= 1
        assert triples[0]["subject"] in ("Claude Shannon", "Information Theory")

    # 2. Hypernym inference template
    prompt_hypernym = (
        f"{HYPERNYM_INFERENCE_TEMPLATE}\n"
        "Parent Concept:\n- Name: Information Theory\n- Definition: Mathematical theory of communication\n"
        "Cluster Member Concepts:\n- Name: Shannon Entropy\n- Name: Channel Capacity\n"
    )
    payload_hypernym = {
        "model": config.CHAT_MODEL,
        "messages": [{"role": "user", "content": prompt_hypernym}],
    }
    req_hypernym = Request(
        url_chat,
        data=json.dumps(payload_hypernym).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req_hypernym) as resp:
        data_hyp = json.loads(resp.read().decode())
        hyp_content = data_hyp["message"]["content"]
        hyp_dict = json.loads(hyp_content)
        assert "hypernym_name" in hyp_dict
        assert "definition" in hyp_dict


def test_alias_status_contestability(test_db):
    """Updating alias status to 'contested' excludes it from view rewrites; restoring to 'active' re-enables it."""
    test_db.execute(
        """
        INSERT INTO leaf_relations (relation_id, source_leaf_id, subject_id, predicate, object_id)
        VALUES ('rel_404', 'note_a.md', 'concepts:x', 'relates_to', 'concepts:y_surface')
        """
    )
    insert_alias(
        conn=test_db,
        alias_id="concepts:y_surface",
        kind="entity",
        canonical_id="concepts:y_canonical",
        method="lexical",
        score=0.9,
        status="active",
    )
    test_db.commit()

    # Active alias rewrites edge
    edge_active = test_db.execute(
        "SELECT dst_id FROM unified_edges WHERE src_id = 'concepts:x'"
    ).fetchone()
    assert edge_active[0] == "concepts:y_canonical"

    # Contest alias
    updated = set_alias_status(
        test_db, alias_id="concepts:y_surface", kind="entity", status="contested"
    )
    assert updated is True
    test_db.commit()

    # Contested alias is excluded by view WHERE a.status = 'active'
    edge_contested = test_db.execute(
        "SELECT dst_id FROM unified_edges WHERE src_id = 'concepts:x'"
    ).fetchone()
    assert edge_contested[0] == "concepts:y_surface"

    # Re-activate alias
    updated_act = set_alias_status(
        test_db, alias_id="concepts:y_surface", kind="entity", status="active"
    )
    assert updated_act is True
    test_db.commit()

    edge_reactivated = test_db.execute(
        "SELECT dst_id FROM unified_edges WHERE src_id = 'concepts:x'"
    ).fetchone()
    assert edge_reactivated[0] == "concepts:y_canonical"


def test_fallback_novel_node_creation(test_db):
    """Completely novel surface forms with zero lexical or vector hits fallback to new canonical URI."""
    res = resolve(
        name="quantum-superposition-principle",
        kind="entity",
        namespace="concepts",
        conn=test_db,
    )
    assert res.is_new is True
    assert res.method == MatchMethod.NEW.value
    assert res.tier == TierEnum.NEW_INDUCTION
    assert res.canonical_id == "concepts:quantum-superposition-principle"
    assert res.score == 1.0


import unicodedata


def test_resolve_unicode_and_extreme_slugs(tmp_path):
    db_path = tmp_path / "test_unicode.db"
    import sqlite3

    conn = sqlite3.connect(db_path)
    from yaw.storage.db import apply_pragmas, ensure_schema_tables, ensure_schema_views

    apply_pragmas(conn)
    ensure_schema_tables(conn)
    ensure_schema_views(conn)
    conn.commit()

    # regex for canonical id
    valid_id_re = re.compile(r"^[a-z_]+:[a-z0-9_-]+$")

    cases = ["E = mc²", "Schrödinger's Cat", "量子力学"]

    for case in cases:
        res = resolve(
            name=case,
            kind="entity",
            definition_ctx="Test context",
            conn=conn,
            namespace="concepts",
        )
        assert valid_id_re.match(res.canonical_id) is not None, (
            f"Invalid canonical ID generated: {res.canonical_id}"
        )

    # Test NFC vs NFD consistency
    nfc = unicodedata.normalize("NFC", "Schrödinger")
    nfd = unicodedata.normalize("NFD", "Schrödinger")

    res_nfc = resolve(
        name=nfc,
        kind="entity",
        definition_ctx="Test context",
        conn=conn,
        namespace="concepts",
    )
    res_nfd = resolve(
        name=nfd,
        kind="entity",
        definition_ctx="Test context",
        conn=conn,
        namespace="concepts",
    )

    assert res_nfc.canonical_id == res_nfd.canonical_id, (
        f"NFC vs NFD mismatch: {res_nfc.canonical_id} != {res_nfd.canonical_id}"
    )

    conn.close()


def test_unmocked_two_tier_convergence_real_sqlite_and_alias_persistence(test_db):
    """Real SQLite integration validating Tier 1, Tier 2 near-match, alias persistence, and Fallback."""
    conn = test_db

    # 1. Seed canonical node with real 768-dim vector
    v_canon, v_near = _make_orthogonal_vectors(dim=768, target_cosine=0.92)
    _, v_far = _make_orthogonal_vectors(dim=768, target_cosine=0.45)

    conn.execute(
        """
        INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition, embedding)
        VALUES ('concepts:quantum-entanglement', 'note_qm.md', 'quantum-entanglement', 'concepts',
                'Phenomenon where quantum particles share states', ?)
        """,
        (v_canon.astype(np.float32).tobytes(),),
    )
    conn.commit()

    # Step A: Tier 1 - Exact canonical slug hit
    res_tier1 = resolve(
        name="quantum-entanglement",
        kind="entity",
        namespace="concepts",
        conn=conn,
    )
    assert res_tier1.tier == TierEnum.TIER_1_EXACT
    assert res_tier1.canonical_id == "concepts:quantum-entanglement"
    assert res_tier1.score == 1.0
    assert res_tier1.is_new is False

    # Step B: Tier 2 - Near match with vector similarity > 0.80
    res_tier2 = resolve(
        name="spooky-action-at-a-distance",
        kind="entity",
        namespace="concepts",
        conn=conn,
        embedding=v_near.tolist(),
    )
    assert res_tier2.tier == TierEnum.TIER_2_EMBEDDING
    assert res_tier2.canonical_id == "concepts:quantum-entanglement"
    assert res_tier2.method == MatchMethod.AUTO.value
    assert res_tier2.score >= config.CANON_SIM_THRESHOLD
    assert res_tier2.is_new is False

    # Verify alias row genuinely persisted into real SQLite aliases table
    alias_row = conn.execute(
        """
        SELECT alias_id, canonical_id, method, score, status
        FROM aliases
        WHERE alias_id = 'concepts:spooky-action-at-a-distance'
        """
    ).fetchone()
    assert alias_row is not None
    assert alias_row[0] == "concepts:spooky-action-at-a-distance"
    assert alias_row[1] == "concepts:quantum-entanglement"
    assert alias_row[2] == "auto"
    assert alias_row[3] >= 0.80
    assert alias_row[4] == "active"

    # Step C: Subsequent resolution of the alias hits Tier 1 immediately!
    res_subsequent = resolve(
        name="spooky-action-at-a-distance",
        kind="entity",
        namespace="concepts",
        conn=conn,
    )
    assert res_subsequent.tier == TierEnum.TIER_1_EXACT
    assert res_subsequent.canonical_id == "concepts:quantum-entanglement"
    assert res_subsequent.method == MatchMethod.ALIAS.value
    assert res_subsequent.score == 1.0
    assert res_subsequent.is_new is False

    # Step D: Fallback - Dissimilar novel concept (< 0.80) creates new canonical node
    res_novel = resolve(
        name="classical-mechanics",
        kind="entity",
        namespace="concepts",
        conn=conn,
        embedding=v_far.tolist(),
    )
    assert res_novel.tier == TierEnum.NEW_INDUCTION
    assert res_novel.canonical_id == "concepts:classical-mechanics"
    assert res_novel.method == MatchMethod.NEW.value
    assert res_novel.is_new is True

    # No alias created for novel concept
    novel_alias = conn.execute(
        "SELECT canonical_id FROM aliases WHERE alias_id = 'concepts:classical-mechanics'"
    ).fetchone()
    assert novel_alias is None


def test_unmocked_two_tier_convergence_with_spacy_vectors(test_db):
    """Real SQLite and in-process spaCy vector similarity without synthetic mocks."""
    conn = test_db

    nlp = get_base_spacy()
    nlp.vocab.reset_vectors(width=300)
    v_base = np.ones(300, dtype=np.float32)
    nlp.vocab.set_vector("relativity", cast(Floats1d, v_base))
    nlp.vocab.set_vector("general-relativity", cast(Floats1d, v_base))

    conn.execute(
        """
        INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition, embedding)
        VALUES ('concepts:relativity', 'note_physics.md', 'relativity', 'concepts', 'Einstein theory of spacetime', NULL)
        """
    )
    conn.commit()

    res = resolve(
        name="general-relativity",
        kind="entity",
        namespace="concepts",
        conn=conn,
        nlp=nlp,
    )
    assert res.tier == TierEnum.TIER_2_EMBEDDING
    assert res.canonical_id == "concepts:relativity"
    assert res.method == MatchMethod.AUTO.value
    assert res.score >= config.CANON_SIM_THRESHOLD
    assert res.is_new is False

    alias_row = conn.execute(
        "SELECT canonical_id, method, status FROM aliases WHERE alias_id = 'concepts:general-relativity'"
    ).fetchone()
    assert alias_row is not None
    assert alias_row[0] == "concepts:relativity"
    assert alias_row[1] == "auto"
    assert alias_row[2] == "active"


def test_create_canonical_node_vector_dimensions_and_predicate_guard(test_db):
    """Verify _create_canonical_node handles 300d and 768d vectors, and guards predicates."""
    from yaw.indexer.canonicalize import _create_canonical_node

    # 300d vector list
    vec_300 = [0.1] * 300
    _create_canonical_node(
        conn=test_db,
        surface_id="concepts:dim300",
        name="dim300",
        namespace="concepts",
        source_leaf_id="doc.md#0",
        embedding=vec_300,
    )
    row = test_db.execute(
        "SELECT embedding FROM leaf_nodes WHERE node_id = 'concepts:dim300'"
    ).fetchone()
    assert row is not None
    assert row[0] is not None
    assert len(row[0]) == 300 * 4

    # 768d numpy vector with fallback provenance
    vec_768 = np.ones(768, dtype=np.float32)
    _create_canonical_node(
        conn=test_db,
        surface_id="concepts:dim768",
        name="dim768",
        namespace="concepts",
        source_leaf_id=None,
        embedding=vec_768,
    )
    row768 = test_db.execute(
        "SELECT source_leaf_id, embedding FROM leaf_nodes WHERE node_id = 'concepts:dim768'"
    ).fetchone()
    assert row768 is not None
    assert row768[0] == "system:canonical_induction"
    assert len(row768[1]) == 768 * 4

    # Predicate namespace guarded
    _create_canonical_node(
        conn=test_db,
        surface_id="causes",
        name="causes",
        namespace="predicates",
        source_leaf_id="doc.md#0",
    )
    assert (
        test_db.execute("SELECT * FROM leaf_nodes WHERE node_id = 'causes'").fetchone()
        is None
    )

    # Predicates: prefix guarded
    _create_canonical_node(
        conn=test_db,
        surface_id="predicates:relates_to",
        name="relates_to",
        namespace="concept",
        source_leaf_id="doc.md#0",
    )
    assert (
        test_db.execute(
            "SELECT * FROM leaf_nodes WHERE node_id = 'predicates:relates_to'"
        ).fetchone()
        is None
    )


def test_wp517_candidate_resolution_threshold_mismatch_records_contested(test_db):
    """WP 5.1.7: Candidate with high cosine (0.82) failing higher threshold (0.90) records contested alias."""
    from yaw.indexer.canonicalize import get_contested_aliases

    v_canon, v_cand = _make_orthogonal_vectors(dim=768, target_cosine=0.82)

    test_db.execute(
        """
        INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition, embedding)
        VALUES ('concepts:quantum-mechanics', 'note_qm.md', 'quantum-mechanics', 'concepts',
                'Fundamental physics of atomic scales', ?)
        """,
        (v_canon.astype(np.float32).tobytes(),),
    )
    test_db.commit()

    res = resolve(
        name="quantum-field-theory",
        kind="entity",
        definition_ctx="Theoretical framework combining classical field theory and quantum mechanics",
        conn=test_db,
        embedding=v_cand.tolist(),
        threshold=0.90,
    )

    assert res.is_new is True
    assert res.canonical_id == "concepts:quantum-field-theory"

    contested = get_contested_aliases(test_db)
    assert len(contested) == 1
    assert contested[0]["alias_id"] == "concepts:quantum-field-theory"
    assert contested[0]["canonical_id"] == "concepts:quantum-mechanics"
    assert contested[0]["status"] == "contested"
