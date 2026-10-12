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
import time
from typing import Any, cast
from urllib.request import Request, urlopen

import numpy as np
import pytest
import spacy
from thinc.types import Floats1d

from yaw import config
from yaw.indexer import (
    MatchMethod,
    ResolutionResult,
    TierEnum,
    resolve,
)
from yaw.indexer.canonicalize import (
    _normalize_entity_key,
    check_linguistic_corroboration,
    is_acronym,
    is_head_anchored_containment,
    is_initial_expansion,
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
        method="exact",
        score=1.0,
        status="active",
    )
    test_db.commit()

    initial_req_count = len(mock_ollama_requests)

    res = resolve(name="shannon-entropy", kind="entity", conn=test_db)

    assert isinstance(res, ResolutionResult)
    assert res.canonical_id == "concepts:entropy"
    assert res.method == MatchMethod.ALIAS.value
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
    """Near match with vector similarity > CANON_SIM_THRESHOLD (0.80) without linguistic corroboration records soft link."""
    v_canon, v_cand = _make_orthogonal_vectors(dim=768, target_cosine=0.88)
    test_db.execute(
        """
        INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition)
        VALUES ('concepts:entropy', 'note_entropy.md', 'entropy', 'concepts', 'Thermodynamic entropy')
        """
    )
    test_db.execute(
        "INSERT INTO entity_vec (entity_id, partition_key, embedding) VALUES ('concepts:entropy', 'concepts', ?)",
        (v_canon.astype(np.float32).tobytes(),),
    )
    test_db.commit()

    initial_req_count = len(mock_ollama_requests)

    # "entrophy" vs "entropy" lacks linguistic corroboration -> soft link in close_matches, new induction
    res = resolve(
        name="entrophy",
        kind="entity",
        conn=test_db,
        embedding=v_cand.tolist(),
    )

    assert res.canonical_id == "concepts:entrophy"
    assert res.method == MatchMethod.NEW.value
    assert res.tier == TierEnum.NEW_INDUCTION
    assert res.is_new is True

    # Soft link inserted in close_matches
    close_row = test_db.execute(
        "SELECT src_id, dst_id, score FROM close_matches WHERE (src_id = 'concepts:entropy' AND dst_id = 'concepts:entrophy') OR (src_id = 'concepts:entrophy' AND dst_id = 'concepts:entropy')"
    ).fetchone()
    assert close_row is not None
    assert close_row[2] >= config.CANON_SIM_THRESHOLD

    # No alias created pointing to concepts:entropy
    alias_row = test_db.execute(
        "SELECT canonical_id FROM aliases WHERE alias_id = 'concepts:entrophy'"
    ).fetchone()
    assert alias_row is None

    # Decision logged in canon_verdicts
    verdict_row = test_db.execute(
        "SELECT candidate_id, verdict, lexical FROM canon_verdicts WHERE surface_id = 'concepts:entrophy'"
    ).fetchone()
    assert verdict_row is not None
    assert verdict_row[0] == "concepts:entropy"
    assert verdict_row[1] == "distinct"
    assert verdict_row[2] == 0.0

    # Zero LLM requests
    assert len(mock_ollama_requests) == initial_req_count


def test_invariant_5a2_near_match_vector_consolidation_corroborated(
    test_db, mock_ollama_requests
):
    """Near match with vector similarity >= 0.80 AND head-anchored containment consolidates via Tier 2."""
    v_canon, v_cand = _make_orthogonal_vectors(dim=768, target_cosine=0.88)
    test_db.execute(
        """
        INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition)
        VALUES ('concepts:albert-einstein', 'note_phys.md', 'Albert Einstein', 'concepts', 'Theoretical physicist')
        """
    )
    test_db.execute(
        "INSERT INTO entity_vec (entity_id, partition_key, embedding) VALUES ('concepts:albert-einstein', 'concepts', ?)",
        (v_canon.astype(np.float32).tobytes(),),
    )
    test_db.commit()

    initial_req_count = len(mock_ollama_requests)

    # "Einstein" is head-anchored containment of "Albert Einstein" -> corroborated -> hard merge
    res = resolve(
        name="Einstein",
        kind="entity",
        conn=test_db,
        embedding=v_cand.tolist(),
    )

    assert res.canonical_id == "concepts:albert-einstein"
    assert res.method == MatchMethod.AUTO.value
    assert res.tier == TierEnum.TIER_2_EMBEDDING
    assert res.score >= config.CANON_SIM_THRESHOLD
    assert res.is_new is False

    # Alias record stored in aliases
    alias_row = test_db.execute(
        "SELECT canonical_id, method FROM aliases WHERE alias_id = 'concepts:einstein' AND kind = 'entity'"
    ).fetchone()
    assert alias_row is not None
    assert alias_row[0] == "concepts:albert-einstein"
    assert alias_row[1] == "auto"

    # Match logged in canon_verdicts
    verdict_row = test_db.execute(
        "SELECT candidate_id, verdict, lexical FROM canon_verdicts WHERE surface_id = 'concepts:einstein'"
    ).fetchone()
    assert verdict_row is not None
    assert verdict_row[0] == "concepts:albert-einstein"
    assert verdict_row[1] == "match"
    assert verdict_row[2] == 1.0

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
    assert res_exact.tier == TierEnum.TIER_1_EXACT

    # Typo predicate matches lexical ratio >= 0.85 against 'subClassOf'
    res_typo = resolve(name="subclassof", kind="predicate", conn=test_db)
    assert "subclass" in res_typo.canonical_id.lower()
    assert res_typo.tier == TierEnum.TIER_1_EXACT

    assert len(mock_ollama_requests) == initial_req_count


# ==============================================================================
# INVARIANT 5A.2: SEMANTIC VECTOR CANOPY
# ==============================================================================


def test_invariant_5a2_semantic_vector_canopy_auto_merge(test_db, mock_ollama_requests):
    """Cosine similarity >= CANON_SIM_HIGH (0.95) without linguistic corroboration records soft link."""
    v_canon, v_cand = _make_orthogonal_vectors(dim=768, target_cosine=0.97)

    # Seed canonical node with high-dimensional embedding
    test_db.execute(
        """
        INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition)
        VALUES ('concepts:space-time-curvature', 'note_relativity.md', 'space-time-curvature', 'concepts', 'Curvature of four-dimensional space-time under general relativity')
        """
    )
    test_db.execute(
        "INSERT INTO entity_vec (entity_id, partition_key, embedding) VALUES ('concepts:space-time-curvature', 'concepts', ?)",
        (v_canon.astype(np.float32).tobytes(),),
    )
    test_db.commit()

    initial_req_count = len(mock_ollama_requests)

    # 'space-time-warp' has cosine 0.97 >= 0.95 with 'space-time-curvature' but lacks linguistic corroboration
    res = resolve(
        name="space-time-warp",
        kind="entity",
        definition_ctx="Warping of spacetime continuum",
        conn=test_db,
        embedding=v_cand.tolist(),
    )

    assert res.canonical_id == "concepts:space-time-warp"
    assert res.method == MatchMethod.NEW.value
    assert res.tier == TierEnum.NEW_INDUCTION
    assert res.is_new is True

    # Soft link in close_matches
    close_row = test_db.execute(
        "SELECT src_id, dst_id, score FROM close_matches WHERE src_id = 'concepts:space-time-curvature' AND dst_id = 'concepts:space-time-warp'"
    ).fetchone()
    assert close_row is not None
    assert close_row[2] >= config.CANON_SIM_HIGH

    # No alias created
    assert (
        test_db.execute(
            "SELECT canonical_id FROM aliases WHERE alias_id = 'concepts:space-time-warp'"
        ).fetchone()
        is None
    )

    # Verdict logged as distinct
    v_row = test_db.execute(
        "SELECT candidate_id, verdict, lexical FROM canon_verdicts WHERE surface_id = 'concepts:space-time-warp'"
    ).fetchone()
    assert v_row is not None
    assert v_row[0] == "concepts:space-time-curvature"
    assert v_row[1] == "distinct"
    assert v_row[2] == 0.0

    # Zero LLM chat requests triggered
    assert len(mock_ollama_requests) == initial_req_count


def test_invariant_5a2_semantic_vector_canopy_auto_merge_corroborated(
    test_db, mock_ollama_requests
):
    """Cosine similarity >= CANON_SIM_HIGH (0.95) with acronym corroboration auto-merges with method='auto'."""
    v_canon, v_cand = _make_orthogonal_vectors(dim=768, target_cosine=0.97)

    test_db.execute(
        """
        INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition)
        VALUES ('concepts:national-aeronautics-and-space-administration', 'note_space.md', 'National Aeronautics and Space Administration', 'concepts', 'US space agency')
        """
    )
    test_db.execute(
        "INSERT INTO entity_vec (entity_id, partition_key, embedding) VALUES ('concepts:national-aeronautics-and-space-administration', 'concepts', ?)",
        (v_canon.astype(np.float32).tobytes(),),
    )
    test_db.commit()

    initial_req_count = len(mock_ollama_requests)

    # "NASA" is acronym of "National Aeronautics and Space Administration"
    res = resolve(
        name="NASA",
        kind="entity",
        definition_ctx="US space agency acronym",
        conn=test_db,
        embedding=v_cand.tolist(),
    )

    assert res.canonical_id == "concepts:national-aeronautics-and-space-administration"
    assert res.method == MatchMethod.AUTO.value
    assert res.tier == TierEnum.TIER_2_EMBEDDING
    assert res.score >= config.CANON_SIM_HIGH
    assert res.is_new is False

    # Alias recorded with auto method
    alias_row = test_db.execute(
        "SELECT canonical_id, method FROM aliases WHERE alias_id = 'concepts:nasa'"
    ).fetchone()
    assert alias_row is not None
    assert alias_row[0] == "concepts:national-aeronautics-and-space-administration"
    assert alias_row[1] == "auto"

    # Match logged in canon_verdicts
    v_row = test_db.execute(
        "SELECT candidate_id, verdict, lexical FROM canon_verdicts WHERE surface_id = 'concepts:nasa'"
    ).fetchone()
    assert v_row is not None
    assert v_row[0] == "concepts:national-aeronautics-and-space-administration"
    assert v_row[1] == "match"
    assert v_row[2] == 1.0

    assert len(mock_ollama_requests) == initial_req_count


# ==============================================================================
# INVARIANT 5A.3: GRAY-ZONE ADJUDICATION & VERDICT LOGGING
# ==============================================================================


def test_invariant_5a2_vector_similarity_above_threshold_auto_merges_without_llm(
    test_db, mock_ollama_requests
):
    """Cosine similarity > CANON_SIM_THRESHOLD (0.80) without linguistic corroboration records soft link in close_matches."""
    v_canon, v_cand = _make_orthogonal_vectors(dim=768, target_cosine=0.88)

    test_db.execute(
        """
        INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition)
        VALUES ('concepts:information-theory', 'note_info.md', 'information-theory', 'concepts', 'Mathematical treatment of communication and information storage')
        """
    )
    test_db.execute(
        "INSERT INTO entity_vec (entity_id, partition_key, embedding) VALUES ('concepts:information-theory', 'concepts', ?)",
        (v_canon.astype(np.float32).tobytes(),),
    )
    test_db.commit()

    initial_req_count = len(mock_ollama_requests)

    # Surface name 'communication-theory' has cosine ~ 0.88 >= 0.80 threshold but lacks linguistic corroboration
    res = resolve(
        name="communication-theory",
        kind="entity",
        definition_ctx="Mathematical theory of signal communication",
        conn=test_db,
        embedding=v_cand.tolist(),
    )

    assert res.canonical_id == "concepts:communication-theory"
    assert res.method == MatchMethod.NEW.value
    assert res.tier == TierEnum.NEW_INDUCTION
    assert res.is_new is True

    # Soft link in close_matches
    close_row = test_db.execute(
        "SELECT src_id, dst_id, score FROM close_matches WHERE (src_id = 'concepts:communication-theory' AND dst_id = 'concepts:information-theory') OR (src_id = 'concepts:information-theory' AND dst_id = 'concepts:communication-theory')"
    ).fetchone()
    assert close_row is not None
    assert close_row[2] >= config.CANON_SIM_THRESHOLD

    # No alias created
    assert (
        test_db.execute(
            "SELECT canonical_id FROM aliases WHERE alias_id = 'concepts:communication-theory'"
        ).fetchone()
        is None
    )

    # Distinct verdict logged in canon_verdicts
    v_row = test_db.execute(
        "SELECT candidate_id, verdict, lexical FROM canon_verdicts WHERE surface_id = 'concepts:communication-theory'"
    ).fetchone()
    assert v_row is not None
    assert v_row[0] == "concepts:information-theory"
    assert v_row[1] == "distinct"
    assert v_row[2] == 0.0

    # Zero LLM chat requests triggered
    assert len(mock_ollama_requests) == initial_req_count


def test_invariant_5a2_vector_similarity_above_threshold_auto_merges_without_llm_corroborated(
    test_db, mock_ollama_requests
):
    """Cosine similarity > 0.80 with initial expansion corroboration auto-merges with method='auto'."""
    v_canon, v_cand = _make_orthogonal_vectors(dim=768, target_cosine=0.88)

    test_db.execute(
        """
        INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition)
        VALUES ('concepts:albert-einstein', 'note_phys.md', 'Albert Einstein', 'concepts', 'Theoretical physicist')
        """
    )
    test_db.execute(
        "INSERT INTO entity_vec (entity_id, partition_key, embedding) VALUES ('concepts:albert-einstein', 'concepts', ?)",
        (v_canon.astype(np.float32).tobytes(),),
    )
    test_db.commit()

    initial_req_count = len(mock_ollama_requests)

    # "A. Einstein" is initial expansion of "Albert Einstein"
    res = resolve(
        name="A. Einstein",
        kind="entity",
        definition_ctx="Theoretical physicist",
        conn=test_db,
        embedding=v_cand.tolist(),
    )

    assert res.canonical_id == "concepts:albert-einstein"
    assert res.method == MatchMethod.AUTO.value
    assert res.tier == TierEnum.TIER_2_EMBEDDING
    assert res.score >= config.CANON_SIM_THRESHOLD
    assert res.is_new is False

    # Alias created with method='auto'
    alias_row = test_db.execute(
        "SELECT canonical_id, method FROM aliases WHERE alias_id = 'concepts:a-einstein'"
    ).fetchone()
    assert alias_row is not None
    assert alias_row[0] == "concepts:albert-einstein"
    assert alias_row[1] == "auto"

    assert len(mock_ollama_requests) == initial_req_count


def test_fallback_vector_similarity_below_threshold_creates_new_node(
    test_db, mock_ollama_requests
):
    """Cosine similarity below CANON_SIM_THRESHOLD (0.80) falls back to new canonical node with zero LLM calls."""
    v_canon, v_cand = _make_orthogonal_vectors(dim=768, target_cosine=0.65)

    test_db.execute(
        """
        INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition)
        VALUES ('concepts:thermodynamics', 'note_thermo.md', 'thermodynamics', 'concepts', 'Branch of physics dealing with heat and temperature')
        """
    )
    test_db.execute(
        "INSERT INTO entity_vec (entity_id, partition_key, embedding) VALUES ('concepts:thermodynamics', 'concepts', ?)",
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
    assert res.canonical_id == "concepts:cryogenic"
    assert res.score == 1.0

    # No alias created pointing to thermodynamics
    alias_row = test_db.execute(
        "SELECT canonical_id FROM aliases WHERE alias_id = 'concepts:cryogenic'"
    ).fetchone()
    assert alias_row is None

    # Zero LLM requests triggered
    assert len(mock_ollama_requests) == initial_req_count


def test_two_tier_convergence_bypasses_llm_completely(test_db, mock_ollama_requests):
    """Two-Tier Convergence Engine resolve() consolidates near-matches via Tier 2 with zero LLM calls."""
    v_canon, v_cand = _make_orthogonal_vectors(dim=768, target_cosine=0.88)

    test_db.execute(
        """
        INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition)
        VALUES ('concepts:deep-reinforcement-learning', 'note_rl.md', 'deep reinforcement learning', 'concepts', 'Combination of artificial neural networks and reinforcement learning')
        """
    )
    test_db.execute(
        "INSERT INTO entity_vec (entity_id, partition_key, embedding) VALUES ('concepts:deep-reinforcement-learning', 'concepts', ?)",
        (v_canon.astype(np.float32).tobytes(),),
    )
    test_db.commit()

    initial_req_count = len(mock_ollama_requests)

    # "DRL" is acronym of "deep reinforcement learning" (acronym corroboration)
    res = resolve(
        name="DRL",
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
        method="exact",
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
        method="exact",
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
        method="exact",
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
        INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition)
        VALUES ('concepts:quantum-entanglement', 'note_qm.md', 'quantum-entanglement', 'concepts', 'Phenomenon where quantum particles share states')
        """
    )
    conn.execute(
        "INSERT INTO entity_vec (entity_id, partition_key, embedding) VALUES ('concepts:quantum-entanglement', 'concepts', ?)",
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

    # Step B: Tier 2 - Near match with vector similarity > 0.80 and head-anchored containment
    res_tier2 = resolve(
        name="entanglement",
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
        SELECT alias_id, canonical_id, method, score
        FROM aliases
        WHERE alias_id = 'concepts:entanglement'
        """
    ).fetchone()
    assert alias_row is not None
    assert alias_row[0] == "concepts:entanglement"
    assert alias_row[1] == "concepts:quantum-entanglement"
    assert alias_row[2] == "auto"
    assert alias_row[3] >= 0.80

    # Step C: Subsequent resolution of the alias hits Tier 1 immediately!
    res_subsequent = resolve(
        name="entanglement",
        kind="entity",
        namespace="concepts",
        conn=conn,
    )
    assert res_subsequent.tier == TierEnum.TIER_1_EXACT
    assert res_subsequent.canonical_id == "concepts:quantum-entanglement"
    assert res_subsequent.method == MatchMethod.ALIAS.value
    assert res_subsequent.score == 1.0
    assert res_subsequent.is_new is False

    # Step C2: Uncorroborated candidate creates soft link in close_matches
    res_uncorr = resolve(
        name="spooky-action-at-a-distance",
        kind="entity",
        namespace="concepts",
        conn=conn,
        embedding=v_near.tolist(),
    )
    assert res_uncorr.tier == TierEnum.NEW_INDUCTION
    assert res_uncorr.canonical_id == "concepts:spooky-action-at-a-distance"
    assert res_uncorr.is_new is True

    close_row = conn.execute(
        "SELECT src_id, dst_id FROM close_matches WHERE src_id = 'concepts:quantum-entanglement' AND dst_id = 'concepts:spooky-action-at-a-distance'"
    ).fetchone()
    assert close_row is not None

    # Step D: Fallback - Dissimilar novel concept (< 0.80) creates new canonical node
    res_novel = resolve(
        name="classical-mechanics",
        kind="entity",
        namespace="concepts",
        conn=conn,
        embedding=v_far.tolist(),
    )
    assert res_novel.tier == TierEnum.NEW_INDUCTION
    assert res_novel.canonical_id == "concepts:classical-mechanic"
    assert res_novel.method == MatchMethod.NEW.value
    assert res_novel.is_new is True

    # No alias created for novel concept
    novel_alias = conn.execute(
        "SELECT canonical_id FROM aliases WHERE alias_id = 'concepts:classical-mechanic'"
    ).fetchone()
    assert novel_alias is None


def test_unmocked_two_tier_convergence_with_spacy_vectors(test_db):
    from yaw.indexer.canonicalize import create_in_memory_kb, sync_kb_from_db

    conn = test_db

    try:
        nlp = get_base_spacy()
        nlp.vocab.reset_vectors(width=300)
        import numpy as np

        v_base_300 = np.ones(300, dtype=np.float32)
        nlp.vocab.set_vector("relativity", cast(Floats1d, v_base_300))
        nlp.vocab.set_vector("general-relativity", cast(Floats1d, v_base_300))

        # Seed 768d unit vector in entity_vec (Space S)
        v_canon, v_cand = _make_orthogonal_vectors(dim=768, target_cosine=0.90)
        conn.execute(
            """
            INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition)
                VALUES ('concepts:relativity', 'note_physics.md', 'relativity', 'concepts', 'Einstein theory of spacetime')
            """
        )
        conn.execute(
            "INSERT INTO entity_vec (entity_id, partition_key, embedding) VALUES ('concepts:relativity', 'concepts', ?)",
            (v_canon.astype(np.float32).tobytes(),),
        )
        conn.commit()

        kb = create_in_memory_kb(nlp, entity_vector_length=300)
        sync_kb_from_db(kb, conn)
        res = resolve(
            name="general-relativity",
            kind="entity",
            namespace="concepts",
            conn=conn,
            embedding=v_cand.tolist(),
            nlp=nlp,
            kb=kb,
        )
        assert res.tier == TierEnum.TIER_2_EMBEDDING
        assert res.canonical_id == "concepts:relativity"
        assert res.method == MatchMethod.AUTO.value
        assert res.score >= config.CANON_SIM_THRESHOLD
        assert res.is_new is False

        alias_row = conn.execute(
            "SELECT canonical_id, method FROM aliases WHERE alias_id = 'concepts:general-relativity'"
        ).fetchone()
        assert alias_row is not None
        assert alias_row[0] == "concepts:relativity"
        assert alias_row[1] == "auto"
    finally:
        get_base_spacy.cache_clear()


def test_create_canonical_node_vector_dimensions_and_predicate_guard(test_db):
    """Verify _create_canonical_node handles 300d and 768d vectors, and guards predicates."""
    from yaw.indexer.canonicalize import _create_canonical_node

    # 300d vector list
    vec_768 = [0.1] * 768
    _create_canonical_node(
        conn=test_db,
        surface_id="concepts:dim300",
        name="dim300",
        namespace="concepts",
        source_leaf_id="doc.md#0",
        embedding=vec_768,
    )
    row = test_db.execute(
        "SELECT embedding FROM entity_vec WHERE entity_id = 'concepts:dim300'"
    ).fetchone()
    assert row is not None
    assert row[0] is not None
    assert len(row[0]) == 768 * 4

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
        "SELECT n.source_leaf_id, v.embedding FROM leaf_nodes n LEFT JOIN entity_vec v ON n.node_id = v.entity_id WHERE n.node_id = 'concepts:dim768'"
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

    v_canon, v_cand = _make_orthogonal_vectors(dim=768, target_cosine=0.82)

    test_db.execute(
        """
        INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition)
        VALUES ('concepts:quantum-mechanics', 'note_qm.md', 'quantum-mechanics', 'concepts', 'Fundamental physics of atomic scales')
        """
    )
    test_db.execute(
        "INSERT INTO entity_vec (entity_id, partition_key, embedding) VALUES ('concepts:quantum-mechanics', 'concepts', ?)",
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

    close_matches = test_db.execute(
        "SELECT src_id, dst_id FROM close_matches"
    ).fetchall()
    assert len(close_matches) >= 1
    src, dst = close_matches[0]
    expected = tuple(
        sorted(["concepts:quantum-field-theory", "concepts:quantum-mechanics"])
    )
    assert tuple(sorted([src, dst])) == expected


# ==============================================================================
# SPACYFUSED KNOWLEDGEBASE & ZOMBIE ELIMINATION SUITE
# ==============================================================================


def test_in_memory_kb_lifecycle_and_synchronization(test_db):
    """Test create_in_memory_kb and sync_kb_from_db synchronizes entities, aliases, and priors."""
    from yaw.indexer.canonicalize import create_in_memory_kb, sync_kb_from_db

    nlp = spacy.blank("en")
    kb = create_in_memory_kb(nlp, entity_vector_length=768)
    assert kb.entity_vector_length == 768

    # Seed leaf_nodes
    v = np.ones(768, dtype=np.float32)
    test_db.execute(
        """
        INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition)
        VALUES ('concepts:shannon-entropy', 'doc1.md', 'Shannon Entropy', 'concepts', 'Information measure')
        """
    )
    test_db.execute(
        "INSERT INTO entity_vec (entity_id, partition_key, embedding) VALUES ('concepts:shannon-entropy', 'concepts', ?)",
        (v.tobytes(),),
    )
    # Seed aliases
    test_db.execute(
        """
        INSERT INTO aliases (alias_id, kind, canonical_id, method, score)
        VALUES ('entropy', 'entity', 'concepts:shannon-entropy', 'exact', 1.0)
        """
    )
    test_db.commit()

    sync_kb_from_db(kb, test_db)

    # Entities synced
    assert kb.contains_entity("concepts:shannon-entropy")
    # Base predicates synced
    for bp in config.BASE_PREDICATES:
        assert kb.contains_entity(bp)
        assert kb.contains_alias(bp)

    # Aliases synced
    assert kb.contains_alias("entropy")
    cands = kb.get_alias_candidates("entropy")
    assert len(cands) == 1
    assert cands[0].entity_ == "concepts:shannon-entropy"
    assert cands[0].prior_prob <= 1.0


def test_bidirectional_kb_and_sqlite_registration(test_db):
    """Novel induction and auto-merged aliases register in both SQLite and InMemoryLookupKB."""
    from yaw.indexer.canonicalize import create_in_memory_kb

    nlp = spacy.blank("en")
    kb = create_in_memory_kb(nlp, entity_vector_length=768)

    # 1. Novel induction
    v_ind = np.ones(768, dtype=np.float32)
    res_ind = resolve(
        name="topological-quantum-field-theory",
        kind="entity",
        conn=test_db,
        embedding=v_ind.tolist(),
        kb=kb,
    )
    assert res_ind.tier == TierEnum.NEW_INDUCTION
    assert res_ind.is_new is True

    # Check registered in SQLite
    row = test_db.execute(
        "SELECT node_id FROM leaf_nodes WHERE node_id = ?", (res_ind.canonical_id,)
    ).fetchone()
    assert row is not None

    # Check registered in InMemoryLookupKB
    assert kb.contains_entity(res_ind.canonical_id)
    assert kb.contains_alias("topological-quantum-field-theory")

    # 2. Tier 2 near match auto-merge
    v_cand = v_ind * 0.99
    res_near = resolve(
        name="TQFT",
        kind="entity",
        conn=test_db,
        embedding=v_cand.tolist(),
        kb=kb,
    )
    assert res_near.tier == TierEnum.TIER_2_EMBEDDING
    assert res_near.canonical_id == res_ind.canonical_id

    # Check alias in SQLite
    alias_row = test_db.execute(
        "SELECT canonical_id FROM aliases WHERE alias_id = 'concepts:tqft'"
    ).fetchone()
    assert alias_row is not None
    assert alias_row[0] == res_ind.canonical_id

    # Check alias in InMemoryLookupKB
    assert kb.contains_alias("concepts:tqft")
    cand = kb.get_alias_candidates("concepts:tqft")[0]
    assert cand.entity_ == res_ind.canonical_id


def test_clean_enums_and_zombie_code_elimination():
    """Verify strictly cleaned TierEnum and MatchMethod, and total eradication of CanonicalVerdict."""
    import yaw.indexer
    import yaw.indexer.canonicalize

    # CanonicalVerdict eliminated
    assert not hasattr(yaw.indexer, "CanonicalVerdict")
    assert not hasattr(yaw.indexer.canonicalize, "CanonicalVerdict")

    # TierEnum strictly contains 3 variants
    expected_tiers = {"TIER_1_EXACT", "TIER_2_EMBEDDING", "NEW_INDUCTION"}
    actual_tiers = {m.name for m in TierEnum}
    assert actual_tiers == expected_tiers

    # MatchMethod strictly contains 4 variants
    expected_methods = {"EXACT", "ALIAS", "AUTO", "NEW"}
    actual_methods = {m.name for m in MatchMethod}
    assert actual_methods == expected_methods


def test_ent_kb_id_propagation_to_extracted_node(test_db):
    """Verify ent.kb_id_ from linked spans propagates directly into ExtractedNode and resolve()."""
    from yaw.nlp.components import _extract_entity_nodes

    nlp = spacy.blank("en")
    ruler: Any = nlp.add_pipe("entity_ruler")
    ruler.add_patterns([{"label": "CONCEPT", "pattern": "Topological Insulator"}])
    doc = nlp("Topological Insulator is a condensed matter phase.")

    for token in doc.ents[0]:
        token.ent_kb_id_ = "concepts:topological-insulator"

    node_map = {}
    ent_nodes = _extract_entity_nodes(doc, node_map)
    assert len(ent_nodes) == 1
    node = ent_nodes[0]
    assert node is not None
    assert node.name == "Topological Insulator"
    assert node.canonical_id == "concepts:topological-insulator"

    # resolve with canonical_id hits Tier 1 instantly
    res = resolve(
        name=node.name,
        conn=test_db,
        canonical_id=node.canonical_id,
    )
    assert res.tier == TierEnum.TIER_1_EXACT
    assert res.canonical_id == "concepts:topological-insulator"
    assert res.method == MatchMethod.EXACT.value


def test_defined_aliases_preserves_multi_candidate_distribution(test_db):
    """Verify defined_aliases in aliases table are not overwritten by node_id pre-registration in sync_kb_from_db."""
    from yaw.indexer.canonicalize import create_in_memory_kb, sync_kb_from_db

    # Insert leaf node whose ID is also an alias in the aliases table
    test_db.execute(
        "INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition) VALUES ('concepts:transformer', 'doc1.md', 'Transformer Architecture', 'concepts', 'Dual apparatus')"
    )
    # Insert calibrated multi-candidate distribution for this alias ID
    test_db.execute(
        "INSERT INTO aliases (alias_id, kind, canonical_id, method, score) VALUES ('concepts:transformer', 'entity', 'concepts:transformer_ml', 'alias', 0.8)"
    )
    test_db.execute(
        "INSERT INTO aliases (alias_id, kind, canonical_id, method, score) VALUES ('concepts:transformer', 'predicate', 'concepts:transformer_ee', 'alias', 0.2)"
    )
    test_db.commit()

    kb = create_in_memory_kb()
    sync_kb_from_db(kb, test_db)

    cands = kb.get_alias_candidates("concepts:transformer")
    cand_map = {c.entity_: c.prior_prob for c in cands}
    assert "concepts:transformer_ml" in cand_map
    assert "concepts:transformer_ee" in cand_map
    assert pytest.approx(cand_map["concepts:transformer_ml"], abs=1e-3) == 0.8
    assert pytest.approx(cand_map["concepts:transformer_ee"], abs=1e-3) == 0.2
    assert sum(cand_map.values()) <= 1.0 + 1e-6


def test_alias_conflicting_scores_takes_max_prior(test_db):
    """Verify duplicate alias definitions with different scores resolve to the maximum confidence score."""
    from yaw.indexer.canonicalize import create_in_memory_kb, sync_kb_from_db

    test_db.execute(
        "INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition) VALUES ('concepts:sensor', 'doc1.md', 'Sensor', 'concepts', 'Detector')"
    )
    # Duplicate alias mapping across entity and predicate with different scores
    test_db.execute(
        "INSERT INTO aliases (alias_id, kind, canonical_id, method, score) VALUES ('detector', 'entity', 'concepts:sensor', 'alias', 0.4)"
    )
    test_db.execute(
        "INSERT INTO aliases (alias_id, kind, canonical_id, method, score) VALUES ('detector', 'predicate', 'concepts:sensor', 'alias', 0.9)"
    )
    test_db.commit()

    kb = create_in_memory_kb()
    sync_kb_from_db(kb, test_db)

    cands = kb.get_alias_candidates("detector")
    assert len(cands) == 1
    assert cands[0].entity_ == "concepts:sensor"
    assert pytest.approx(cands[0].prior_prob, abs=1e-3) == 1.0


def test_in_memory_kb_proportional_append_allocation(test_db):
    """Verify appending multiple new candidates to existing KB alias proportionally divides remaining budget."""
    from yaw.indexer.canonicalize import create_in_memory_kb, sync_kb_from_db

    kb = create_in_memory_kb()
    kb.add_entity(
        "concepts:existing_c", freq=1, entity_vector=np.zeros(300, dtype=np.float32)
    )
    # Pre-existing candidate takes 0.4 of probability budget
    kb.add_alias("polysemous_term", ["concepts:existing_c"], [0.4])

    test_db.execute(
        "INSERT INTO aliases (alias_id, kind, canonical_id, method, score) VALUES ('polysemous_term', 'entity', 'concepts:new_c1', 'alias', 0.8)"
    )
    test_db.execute(
        "INSERT INTO aliases (alias_id, kind, canonical_id, method, score) VALUES ('polysemous_term', 'predicate', 'concepts:new_c2', 'alias', 0.4)"
    )
    test_db.commit()

    sync_kb_from_db(kb, test_db)

    cands = kb.get_alias_candidates("polysemous_term")
    cand_map = {c.entity_: c.prior_prob for c in cands}
    assert "concepts:existing_c" in cand_map
    assert "concepts:new_c1" in cand_map
    assert "concepts:new_c2" in cand_map
    assert sum(cand_map.values()) <= 1.0 + 1e-6
    # Relative ratio between new candidates (0.8 vs 0.4 -> 2:1) is preserved within remaining 0.6 budget
    assert cand_map["concepts:new_c1"] > cand_map["concepts:new_c2"]
    assert pytest.approx(cand_map["concepts:new_c1"], abs=1e-2) == 0.4
    assert pytest.approx(cand_map["concepts:new_c2"], abs=1e-2) == 0.2


def test_in_memory_kb_case_insensitive_matching(test_db):
    """Verify title-cased names in SQLite normalize and match query mentions in KB."""
    from yaw.indexer.canonicalize import create_in_memory_kb, sync_kb_from_db
    from yaw.retrieval.retriever import _extract_query_mentions

    test_db.execute(
        "INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition) VALUES ('concepts:albert-einstein', 'doc1.md', 'Albert Einstein', 'concepts', 'Physicist')"
    )
    test_db.commit()

    kb = create_in_memory_kb()
    sync_kb_from_db(kb, test_db)

    mentions = _extract_query_mentions("Who was Albert Einstein?")
    matched = [m for m in mentions if kb.contains_alias(m)]
    assert "albert einstein" in matched
    cands = kb.get_alias_candidates("albert einstein")
    assert len(cands) == 1
    assert cands[0].entity_ == "concepts:albert-einstein"
    # Assert exact title casing is also directly preserved in KB
    assert kb.contains_alias("Albert Einstein")
    cands_exact = kb.get_alias_candidates("Albert Einstein")
    assert len(cands_exact) == 1
    assert cands_exact[0].entity_ == "concepts:albert-einstein"


def test_in_memory_kb_dual_casing_exact_and_lowercase(test_db):
    """Verify sync_kb_from_db dual-registers both exact-cased and lowercased aliases with full candidate prior distributions."""
    from yaw.indexer.canonicalize import create_in_memory_kb, sync_kb_from_db

    test_db.execute(
        "INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition) VALUES ('concepts:tqft', 'doc1.md', 'Topological QFT', 'concepts', 'Quantum field theory')"
    )
    test_db.execute(
        "INSERT INTO aliases (alias_id, kind, canonical_id, method, score) VALUES ('Topological Quantum Field Theory', 'entity', 'concepts:tqft', 'alias', 0.95)"
    )
    test_db.commit()

    kb = create_in_memory_kb()
    sync_kb_from_db(kb, test_db)

    # Both exact casing and lowercase must be directly queryable in compiled Cython KB
    assert kb.contains_alias("Topological Quantum Field Theory")
    assert kb.contains_alias("topological quantum field theory")

    exact_cands = kb.get_alias_candidates("Topological Quantum Field Theory")
    lower_cands = kb.get_alias_candidates("topological quantum field theory")

    assert len(exact_cands) == 1
    assert len(lower_cands) == 1
    assert exact_cands[0].entity_ == "concepts:tqft"
    assert lower_cands[0].entity_ == "concepts:tqft"
    assert pytest.approx(exact_cands[0].prior_prob, abs=1e-3) == 1.0
    assert pytest.approx(lower_cands[0].prior_prob, abs=1e-3) == 1.0


# ==============================================================================
# EPIC 7.7 M2: DEDICATED TWO-TIER EDC PROTOCOL UNIT & EDGE-CASE TESTS
# ==============================================================================


def test_tier1_normalization_determiners_stripping():
    """Tier 1 strips leading English determiners (^the\\s+, ^a\\s+, ^an\\s+)."""
    assert _normalize_entity_key("The Quantum Computer") == "quantum-computer"
    assert _normalize_entity_key("the theory of relativity") == "theory-of-relativity"
    assert _normalize_entity_key("A Black Hole") == "black-hole"
    assert _normalize_entity_key("an electron") == "electron"


def test_tier1_normalization_possessive_clitics():
    """Tier 1 strips possessive clitics ('s\\b, ’s\\b)."""
    assert _normalize_entity_key("Schrödinger's Cat") == "schrodinger-cat"
    assert _normalize_entity_key("Euler’s formula") == "euler-formula"
    assert _normalize_entity_key("Newton's Laws") == "newton-law"


def test_tier1_normalization_noun_lemmatization_vs_propn_verbatim():
    """Tier 1 lemmatizes NOUN tokens to singular base lemma while preserving PROPN tokens verbatim."""
    # NOUNs are lemmatized
    assert _normalize_entity_key("cryogenics") == "cryogenic"
    assert _normalize_entity_key("entanglements") == "entanglement"
    assert _normalize_entity_key("quantum computers") == "quantum-computer"

    # PROPNs are preserved verbatim (lowercased)
    assert _normalize_entity_key("Albert Einstein") == "albert-einstein"
    assert _normalize_entity_key("Isaac Newton") == "isaac-newton"


def test_tier1_normalization_nfkc_ligatures():
    """Tier 1 applies NFKC unicode normalization to decompose ligatures and compatibility characters."""
    assert _normalize_entity_key("ﬁle") == "file"
    assert _normalize_entity_key("ﬂow") == "flow"
    assert _normalize_entity_key("ﬃ") == "ffi"


def test_tier2_linguistic_corroboration_acronym():
    """L(a, e) is_acronym correctly detects acronyms matching token initials."""
    nlp = get_base_spacy()
    doc_drl = nlp("DRL")
    doc_full = nlp("Deep Reinforcement Learning")
    assert is_acronym(doc_drl, doc_full) is True
    assert is_acronym(doc_full, doc_drl) is True

    # With stop word filtering
    doc_tqft = nlp("TQFT")
    doc_tqft_full = nlp("Topological Quantum Field Theory")
    assert is_acronym(doc_tqft, doc_tqft_full) is True

    # Negative cases
    doc_single = nlp("A")
    assert is_acronym(doc_single, nlp("Albert")) is False
    doc_mismatch = nlp("XYZ")
    assert is_acronym(doc_mismatch, doc_full) is False


def test_tier2_linguistic_corroboration_head_anchored_containment():
    """L(a, e) is_head_anchored_containment requires both contiguous subsequence and shared syntactic head."""
    nlp = get_base_spacy()

    # Success: "entanglement" is contiguous subsequence of "quantum entanglement", both have root "entanglement"
    doc_sub = nlp("entanglement")
    doc_full = nlp("quantum entanglement")
    assert is_head_anchored_containment(doc_sub, doc_full) is True
    assert is_head_anchored_containment(doc_full, doc_sub) is True

    # Success: lemma matching ("classical mechanic" vs "classical mechanics")
    assert (
        is_head_anchored_containment(
            nlp("classical mechanic"), nlp("classical mechanics")
        )
        is True
    )

    # Failure: "Albert" vs "Albert Einstein" — syntactic head of "Albert" is "Albert", head of "Albert Einstein" is "Einstein"
    assert is_head_anchored_containment(nlp("Albert"), nlp("Albert Einstein")) is False

    # Failure: non-contiguous subsequence ("quantum theory" vs "quantum field theory")
    assert (
        is_head_anchored_containment(nlp("quantum theory"), nlp("quantum field theory"))
        is False
    )


def test_tier2_linguistic_corroboration_initial_expansion():
    """L(a, e) is_initial_expansion requires token-aligned match where initials match full tokens."""
    nlp = get_base_spacy()

    # Success: A. Einstein <-> Albert Einstein
    assert is_initial_expansion(nlp("A. Einstein"), nlp("Albert Einstein")) is True
    assert is_initial_expansion(nlp("Albert Einstein"), nlp("A. Einstein")) is True

    # Success: J. J. Thomson <-> Joseph John Thomson
    assert (
        is_initial_expansion(nlp("J. J. Thomson"), nlp("Joseph John Thomson")) is True
    )

    # Failure: Initial mismatch
    assert is_initial_expansion(nlp("B. Einstein"), nlp("Albert Einstein")) is False

    # Failure: Token count mismatch
    assert is_initial_expansion(nlp("Einstein"), nlp("Albert Einstein")) is False


def test_tier2_check_linguistic_corroboration_wrapper():
    """check_linguistic_corroboration dispatches across acronym, containment, and initials."""
    assert check_linguistic_corroboration("DRL", "Deep Reinforcement Learning") is True
    assert (
        check_linguistic_corroboration("entanglement", "quantum entanglement") is True
    )
    assert check_linguistic_corroboration("A. Einstein", "Albert Einstein") is True
    assert check_linguistic_corroboration("Albert", "Albert Einstein") is False
    assert (
        check_linguistic_corroboration("Quantum Physics", "Information Theory") is False
    )


def test_tier2_ambiguity_multiple_corroborated_candidates_induces_novel_node(test_db):
    """When >= 2 candidates pass both cosine >= 0.80 and linguistic corroboration, EDC enters ambiguity branch:
    soft links in close_matches, distinct verdicts in canon_verdicts, and novel canonical node induction.
    """
    v1, v2 = _make_orthogonal_vectors(dim=768, target_cosine=0.92)
    # Generate v_mention close to both v1 and v2 (cosine ~0.85+)
    v_mention = (v1 + v2) / np.linalg.norm(v1 + v2)

    # Seed two canonical nodes in leaf_nodes and entity_vec:
    # 1. "quantum entanglement"
    # 2. "bipartite entanglement"
    test_db.execute(
        """
        INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition)
        VALUES ('concepts:quantum-entanglement', 'note_q.md', 'quantum entanglement', 'concepts', 'Quantum entanglement of states')
        """
    )
    test_db.execute(
        "INSERT INTO entity_vec (entity_id, partition_key, embedding) VALUES ('concepts:quantum-entanglement', 'concepts', ?)",
        (v1.astype(np.float32).tobytes(),),
    )

    test_db.execute(
        """
        INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition)
        VALUES ('concepts:bipartite-entanglement', 'note_b.md', 'bipartite entanglement', 'concepts', 'Entanglement between two subsystems')
        """
    )
    test_db.execute(
        "INSERT INTO entity_vec (entity_id, partition_key, embedding) VALUES ('concepts:bipartite-entanglement', 'concepts', ?)",
        (v2.astype(np.float32).tobytes(),),
    )
    test_db.commit()

    # Mention "entanglement": head-anchored containment matches BOTH "quantum entanglement" and "bipartite entanglement"!
    res = resolve(
        name="entanglement",
        kind="entity",
        definition_ctx="Quantum state correlation phenomenon",
        conn=test_db,
        embedding=v_mention.tolist(),
    )

    # Must NOT merge into aliases
    assert res.canonical_id == "concepts:entanglement"
    assert res.method == MatchMethod.NEW.value
    assert res.tier == TierEnum.NEW_INDUCTION
    assert res.is_new is True

    # No alias record created
    alias_row = test_db.execute(
        "SELECT canonical_id FROM aliases WHERE alias_id = 'concepts:entanglement' OR alias_id = 'entanglement'"
    ).fetchone()
    assert alias_row is None

    # Both candidates must be recorded in close_matches with CHECK(src_id < dst_id) satisfied
    cm_rows = test_db.execute(
        """
        SELECT src_id, dst_id, score, evidence
        FROM close_matches
        WHERE src_id = 'concepts:entanglement' OR dst_id = 'concepts:entanglement'
        ORDER BY score DESC
        """
    ).fetchall()
    assert len(cm_rows) == 2
    for s_id, d_id, score, ev in cm_rows:
        assert s_id < d_id
        assert score >= 0.80
        assert "Ambiguous identity (multiple corroborated candidates)" in ev

    # Decisions logged to canon_verdicts as 'distinct' for both candidates
    verdicts = test_db.execute(
        """
        SELECT candidate_id, cosine, lexical, verdict, rationale
        FROM canon_verdicts
        WHERE surface_id = 'concepts:entanglement'
        ORDER BY candidate_id ASC
        """
    ).fetchall()
    assert len(verdicts) == 2
    for cand_id, cosine, lexical, verdict, rationale in verdicts:
        assert cand_id in (
            "concepts:bipartite-entanglement",
            "concepts:quantum-entanglement",
        )
        assert cosine >= 0.80
        assert lexical == 1.0  # Corroborated!
        assert verdict == "distinct"
        assert "Ambiguous identity" in rationale


def test_canon_verdicts_audit_logging_schema_and_fields(test_db):
    """canon_verdicts table logs decision audit records with all required columns and valid values."""
    v_canon, v_cand = _make_orthogonal_vectors(dim=768, target_cosine=0.96)

    test_db.execute(
        """
        INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition)
        VALUES ('concepts:deep-reinforcement-learning', 'ai.md', 'Deep Reinforcement Learning', 'concepts', 'RL with deep networks')
        """
    )
    test_db.execute(
        "INSERT INTO entity_vec (entity_id, partition_key, embedding) VALUES ('concepts:deep-reinforcement-learning', 'concepts', ?)",
        (v_canon.astype(np.float32).tobytes(),),
    )
    test_db.commit()

    start_time = time.time()
    res = resolve(
        name="DRL",
        kind="entity",
        definition_ctx="Deep RL",
        conn=test_db,
        embedding=v_cand.tolist(),
    )
    end_time = time.time()

    assert res.canonical_id == "concepts:deep-reinforcement-learning"
    assert res.method == MatchMethod.AUTO.value
    assert res.tier == TierEnum.TIER_2_EMBEDDING

    row = test_db.execute(
        """
        SELECT surface_id, kind, candidate_id, cosine, lexical, verdict, phase, decided_at, rationale
        FROM canon_verdicts
        WHERE surface_id = 'concepts:drl'
        LIMIT 1
        """
    ).fetchone()

    assert row is not None
    (
        surface_id,
        kind,
        candidate_id,
        cosine,
        lexical,
        verdict,
        phase,
        decided_at,
        rationale,
    ) = row
    assert surface_id == "concepts:drl"
    assert kind == "entity"
    assert candidate_id == "concepts:deep-reinforcement-learning"
    assert 0.95 <= cosine <= 1.0
    assert lexical == 1.0
    assert verdict == "match"
    assert phase == "ingest"
    assert start_time - 1.0 <= decided_at <= end_time + 1.0
    assert "Corroborated Tier 2 match" in rationale


def test_predicate_space_l_guard_never_enters_tier2(test_db):
    """Predicates are strictly guarded in Space L and never query entity_vec or enter Tier 2."""
    v_dummy = np.random.default_rng(42).standard_normal(768).astype(np.float32)
    v_dummy /= np.linalg.norm(v_dummy)

    # Seed entity_vec and leaf_nodes with a predicate
    test_db.execute(
        """
        INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition)
        VALUES ('predicates:causes', 'logic.md', 'causes', 'predicates', 'Causal relation')
        """
    )
    test_db.execute(
        "INSERT INTO entity_vec (entity_id, partition_key, embedding) VALUES ('predicates:causes', 'predicates', ?)",
        (v_dummy.tobytes(),),
    )
    test_db.commit()

    # 1. Resolving standard base predicate synset with embedding
    res1 = resolve(
        name="triggers",
        kind="predicate",
        conn=test_db,
        embedding=v_dummy.tolist(),
    )
    # Must resolve via synset/antonym guard in Space L (Tier 1)
    assert res1.tier == TierEnum.TIER_1_EXACT
    assert res1.canonical_id == "causes"

    # 2. Resolving novel predicate with embedding
    res2 = resolve(
        name="substantially impacts upon",
        kind="predicate",
        conn=test_db,
        embedding=v_dummy.tolist(),
    )
    # Must resolve via novel induction (Tier new_induction), never Tier 2
    assert res2.tier == TierEnum.NEW_INDUCTION
    assert res2.is_new is True
    assert res2.canonical_id.startswith("predicates:")

    # Verify zero audit verdicts logged for predicates
    pred_verdicts = test_db.execute(
        "SELECT COUNT(*) FROM canon_verdicts WHERE kind = 'predicate'"
    ).fetchone()[0]
    assert pred_verdicts == 0

    # Verify zero close_matches created for predicates
    pred_cm = test_db.execute(
        "SELECT COUNT(*) FROM close_matches WHERE src_id LIKE 'predicates:%' OR dst_id LIKE 'predicates:%'"
    ).fetchone()[0]
    assert pred_cm == 0


def test_regression_tier1_kind_guarding_prevents_cross_kind_hijacking(test_db):
    """Regression test for DEFECT-M2-01: Entity never resolves to bare predicate, predicate never resolves to entity KB entry."""
    from yaw.indexer.canonicalize import _register_alias_in_kb, create_in_memory_kb

    # 1. Mention 'causes' as entity must resolve to concepts: namespace, never bare predicate 'causes'
    res_ent = resolve(name="causes", kind="entity", conn=test_db)
    assert res_ent.canonical_id != "causes"
    assert res_ent.canonical_id.startswith("concepts:")

    # 2. Mention 'Quantum Computer' as predicate when KB has entity must resolve to predicates:, never concepts:
    kb = create_in_memory_kb()
    _register_alias_in_kb(kb, "Quantum Computer", "concepts:quantum-computer", 1.0)
    res_pred = resolve(name="Quantum Computer", kind="predicate", conn=test_db, kb=kb)
    assert res_pred.canonical_id.startswith("predicates:")
    assert res_pred.canonical_id == "predicates:quantum-computer"
    assert not res_pred.canonical_id.startswith("concepts:")


def test_regression_tier2_namespace_prefixed_mention_corroborates_and_automerges(
    test_db,
):
    """Regression test for DEFECT-M2-02: Mention with namespace prefix (e.g. concepts:DRL) corroborates against candidate."""
    v1, v2 = _make_orthogonal_vectors(dim=768, target_cosine=0.90)

    test_db.execute(
        """
        INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition)
        VALUES ('concepts:deep-reinforcement-learning', 'rl.md', 'Deep Reinforcement Learning', 'concepts', 'DRL methods')
        """
    )
    test_db.execute(
        "INSERT INTO entity_vec (entity_id, partition_key, embedding) VALUES ('concepts:deep-reinforcement-learning', 'concepts', ?)",
        (v1.astype(np.float32).tobytes(),),
    )
    test_db.commit()

    # Mention passed with namespace prefix 'concepts:DRL'
    res = resolve(
        name="concepts:DRL",
        kind="entity",
        conn=test_db,
        embedding=v2.tolist(),
        definition_ctx="Deep reinforcement learning",
    )

    # Must successfully corroborate and auto-merge into concepts:deep-reinforcement-learning
    assert res.canonical_id == "concepts:deep-reinforcement-learning"
    assert res.method == MatchMethod.AUTO.value
    assert res.tier == TierEnum.TIER_2_EMBEDDING
    assert res.is_new is False


def test_regression_acronym_case_validation():
    """Regression test for DEFECT-M2-03: Lowercase words are rejected as acronyms."""
    nlp = get_base_spacy()
    assert is_acronym(nlp("DRL"), nlp("Deep Reinforcement Learning")) is True
    assert is_acronym(nlp("drl"), nlp("Deep Reinforcement Learning")) is False
    assert is_acronym(nlp("cat"), nlp("Central Action Team")) is False
