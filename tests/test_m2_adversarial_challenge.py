"""Adversarial stress-test suite for Milestone 2: Two-Tier EDC Protocol.

Empirically probes:
1. Deterministic normalization edge cases: foreign names with diacritics, determiners with punctuation, multiple possessives, hyphenated compound nouns.
2. Linguistic corroboration edge cases: false acronyms (lowercase or mismatched lengths), inverted initials, containment with different syntactic roots.
3. Multiple candidates: assert that when 2 candidates both pass corroboration, ambiguity prevents hard merge, records both in close_matches, and logs distinct verdicts in canon_verdicts.
4. Predicates guard: assert predicates never query entity_vec or enter Tier 2.
"""

from __future__ import annotations

import sqlite3

import numpy as np

from yaw.indexer import (
    MatchMethod,
    TierEnum,
    resolve,
)
from yaw.indexer.canonicalize import (
    _normalize_entity_key,
    is_acronym,
    is_head_anchored_containment,
    is_initial_expansion,
)
from yaw.nlp.extractor import get_base_spacy


def _make_orthogonal_vectors(
    dim: int = 768, target_cosine: float = 0.95
) -> tuple[np.ndarray, np.ndarray]:
    """Generate two unit vectors with exact target cosine similarity."""
    rng = np.random.default_rng(12345)
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
# CHALLENGE 1: DETERMINISTIC NORMALIZATION EDGE CASES (SPACE L - TIER 1)
# ==============================================================================


def test_challenge_normalization_diacritics_and_foreign_names():
    """Probe diacritic stripping, NFKC normalization, and foreign name transliteration."""
    # Umlauts and accents
    assert _normalize_entity_key("Gödel") == "godel"
    assert _normalize_entity_key("Erwin Schrödinger") == "erwin-schrodinger"
    assert _normalize_entity_key("René Descartes") == "rene-descartes"
    assert _normalize_entity_key("Henri Poincaré") == "henri-poincare"
    assert _normalize_entity_key("Étienne Klein") == "etienne-klein"
    assert _normalize_entity_key("Anders Jonas Ångström") == "anders-jonas-angstrom"

    # Extreme diacritics / ligatures
    assert _normalize_entity_key("coöperation") == "cooperation"
    assert _normalize_entity_key("œuvre") == "oeuvre"


def test_challenge_normalization_determiners_and_punctuation():
    """Probe determiner stripping with surrounding punctuation and whitespace."""
    # Standard determiners
    assert _normalize_entity_key("The Quantum Computer") == "quantum-computer"
    assert _normalize_entity_key("A Neural Network") == "neural-network"
    assert _normalize_entity_key("An Algorithm") == "algorithm"

    # Determiners followed by hyphens or colons
    assert _normalize_entity_key("The - Graph") == "graph"
    assert _normalize_entity_key("The... Universe") == "universe"

    # Pure determiners (must NOT collapse to empty)
    res_the = _normalize_entity_key("The")
    assert res_the == "the"
    res_a = _normalize_entity_key("A")
    assert res_a == "a"
    res_an = _normalize_entity_key("an")
    assert res_an == "an"


def test_challenge_normalization_multiple_possessives():
    """Probe possessive clitic stripping across single and multiple possessives."""
    # Single possessive
    assert _normalize_entity_key("Schrödinger's Cat") == "schrodinger-cat"
    assert _normalize_entity_key("Euler’s Formula") == "euler-formula"

    # Multiple possessives in chain
    assert _normalize_entity_key("Schrödinger's cat's state") == "schrodinger-cat-state"
    assert _normalize_entity_key("Alice's bob's car") == "alice-bob-car"

    # Coordinated possessives
    assert (
        _normalize_entity_key("Euler's and Fermat's theorem")
        == "euler-and-fermat-theorem"
    )


def test_challenge_normalization_hyphenated_compound_nouns():
    """Probe hyphenated compound nouns and punctuation collapsing."""
    assert _normalize_entity_key("wave-particle duality") == "wave-particle-duality"
    assert _normalize_entity_key("cost-benefit-analysis") == "cost-benefit-analysis"
    assert _normalize_entity_key("state-of-the-art") == "state-of-the-art"
    assert (
        _normalize_entity_key("high-dimensional vector space")
        == "high-dimensional-vector-space"
    )

    # Extreme symbols & empty boundaries
    assert _normalize_entity_key("").strip() == ""
    assert _normalize_entity_key("   ").strip() == ""
    assert _normalize_entity_key("---").startswith("id-")
    assert _normalize_entity_key("???").startswith("id-")
    assert _normalize_entity_key("3-SAT") == "3-sat"
    assert _normalize_entity_key("H2O") == "h2o"


# ==============================================================================
# CHALLENGE 2: LINGUISTIC CORROBORATION EDGE CASES
# ==============================================================================


def test_challenge_linguistic_corroboration_false_acronyms_case_and_length():
    """Probe acronym detection against false acronyms: lowercase, single-char, overlong, and mismatch."""
    nlp = get_base_spacy()

    # Valid uppercase acronyms (2-6 chars)
    doc_drl = nlp("DRL")
    doc_full = nlp("Deep Reinforcement Learning")
    assert is_acronym(doc_drl, doc_full) is True
    assert is_acronym(doc_full, doc_drl) is True

    doc_tqft = nlp("TQFT")
    doc_tqft_full = nlp("Topological Quantum Field Theory")
    assert is_acronym(doc_tqft, doc_tqft_full) is True

    # Punctuation acronyms (e.g. D.R.L.)
    doc_drl_dots = nlp("D.R.L.")
    assert is_acronym(doc_drl_dots, doc_full) is True

    # Negative case: Lowercase token (ADR-012 requires single upper-case token)
    doc_drl_lower = nlp("drl")
    assert is_acronym(doc_drl_lower, doc_full) is False, (
        "Lowercase tokens must NOT be accepted as acronyms"
    )
    assert is_acronym(nlp("cat"), nlp("Central Action Team")) is False, (
        "Common lowercase word must NOT be accepted as acronym"
    )

    # Negative case: Single character (too short, < 2)
    assert is_acronym(nlp("A"), nlp("Albert")) is False

    # Negative case: Overlong (> 6 chars)
    doc_long = nlp("ABCDEFG")
    doc_long_full = nlp("A B C D E F G")
    assert is_acronym(doc_long, doc_long_full) is False

    # Negative case: Token count mismatch
    # "DRL" (len 3) vs "Deep Learning" (len 2)
    assert is_acronym(nlp("DRL"), nlp("Deep Learning")) is False
    # "DL" (len 2) vs "Deep Reinforcement Learning" (len 3)
    assert is_acronym(nlp("DL"), nlp("Deep Reinforcement Learning")) is False


def test_challenge_linguistic_corroboration_inverted_initials():
    """Probe initial expansion against inverted or scrambled initials."""
    nlp = get_base_spacy()

    # Valid expansion: A. Einstein <-> Albert Einstein
    assert is_initial_expansion(nlp("A. Einstein"), nlp("Albert Einstein")) is True
    assert is_initial_expansion(nlp("Albert Einstein"), nlp("A. Einstein")) is True

    # Inverted initials: "Einstein A." vs "Albert Einstein" -> MUST FAIL
    assert is_initial_expansion(nlp("Einstein A."), nlp("Albert Einstein")) is False

    # Scrambled initials: "E. A." vs "Albert Einstein" -> MUST FAIL
    assert is_initial_expansion(nlp("E. A."), nlp("Albert Einstein")) is False

    # Cross initials: "A. E." vs "Einstein Albert" -> MUST FAIL
    assert is_initial_expansion(nlp("A. E."), nlp("Einstein Albert")) is False


def test_challenge_linguistic_corroboration_containment_different_roots():
    """Probe head-anchored containment where tokens match but syntactic roots differ."""
    nlp = get_base_spacy()

    # Shared root ("entanglement"):
    # "entanglement" (root: entanglement) vs "quantum entanglement" (root: entanglement) -> True
    assert (
        is_head_anchored_containment(nlp("entanglement"), nlp("quantum entanglement"))
        is True
    )

    # Different roots:
    # "quantum theory" (root: theory) vs "quantum theory group" (root: group) -> MUST FAIL
    assert (
        is_head_anchored_containment(nlp("quantum theory"), nlp("quantum theory group"))
        is False
    )

    # Different roots:
    # "particle physics" (root: physics) vs "physics department" (root: department) -> MUST FAIL
    assert (
        is_head_anchored_containment(nlp("particle physics"), nlp("physics department"))
        is False
    )

    # Different roots:
    # "Albert" (root: Albert) vs "Albert Einstein" (root: Einstein) -> MUST FAIL
    assert is_head_anchored_containment(nlp("Albert"), nlp("Albert Einstein")) is False

    # Non-contiguous subsequence with matching root:
    # "quantum theory" vs "quantum field theory" -> MUST FAIL (not contiguous)
    assert (
        is_head_anchored_containment(nlp("quantum theory"), nlp("quantum field theory"))
        is False
    )


# ==============================================================================
# CHALLENGE 3: MULTIPLE CANDIDATES AMBIGUITY BRANCH
# ==============================================================================


def test_challenge_multiple_candidates_ambiguity_prevents_hard_merge(
    test_db: sqlite3.Connection,
):
    """Assert that when 2 candidates both pass corroboration, ambiguity prevents hard merge,
    records both in close_matches, and logs distinct verdicts in canon_verdicts.
    """
    v1, v2 = _make_orthogonal_vectors(dim=768, target_cosine=0.92)
    # Target vector close to both candidates (cosine ~0.85+)
    v_target = (v1 + v2) / np.linalg.norm(v1 + v2)

    # Insert candidate 1: "quantum mechanics" (root: mechanics, lemma: mechanic)
    test_db.execute(
        """
        INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition)
        VALUES ('concepts:quantum-mechanics', 'qm.md', 'quantum mechanics', 'concepts', 'Quantum mechanics theory')
        """
    )
    test_db.execute(
        "INSERT INTO entity_vec (entity_id, partition_key, embedding) VALUES ('concepts:quantum-mechanics', 'concepts', ?)",
        (v1.astype(np.float32).tobytes(),),
    )

    # Insert candidate 2: "statistical mechanics" (root: mechanics, lemma: mechanic)
    test_db.execute(
        """
        INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition)
        VALUES ('concepts:statistical-mechanics', 'sm.md', 'statistical mechanics', 'concepts', 'Statistical mechanics theory')
        """
    )
    test_db.execute(
        "INSERT INTO entity_vec (entity_id, partition_key, embedding) VALUES ('concepts:statistical-mechanics', 'concepts', ?)",
        (v2.astype(np.float32).tobytes(),),
    )
    test_db.commit()

    # Mention: "mechanics" (root: mechanics, lemma: mechanic)
    # Head-anchored containment matches BOTH "quantum mechanics" and "statistical mechanics".
    res = resolve(
        name="mechanics",
        kind="entity",
        conn=test_db,
        embedding=v_target.tolist(),
        definition_ctx="Mechanics branch of physics",
    )

    # 1. Hard merge MUST NOT occur
    assert res.canonical_id == "concepts:mechanic"
    assert res.method == MatchMethod.NEW.value
    assert res.tier == TierEnum.NEW_INDUCTION
    assert res.is_new is True

    alias_check = test_db.execute(
        "SELECT canonical_id FROM aliases WHERE alias_id = 'concepts:mechanic' OR alias_id = 'mechanics'"
    ).fetchone()
    assert alias_check is None, "Ambiguous mention must NOT be registered in aliases"

    # 2. Both candidates must be recorded in close_matches
    cm_records = test_db.execute(
        """
        SELECT src_id, dst_id, score, evidence
        FROM close_matches
        WHERE src_id = 'concepts:mechanic' OR dst_id = 'concepts:mechanic'
        ORDER BY dst_id ASC
        """
    ).fetchall()
    assert len(cm_records) == 2, "Both candidates must be recorded in close_matches"
    for src_id, dst_id, score, ev in cm_records:
        assert src_id < dst_id, f"CHECK(src_id < dst_id) violated: {src_id} >= {dst_id}"
        assert score >= 0.80
        assert "Ambiguous identity (multiple corroborated candidates)" in ev

    # 3. Decisions in canon_verdicts logged as 'distinct' for both candidates
    verdict_records = test_db.execute(
        """
        SELECT candidate_id, cosine, lexical, verdict, rationale
        FROM canon_verdicts
        WHERE surface_id = 'concepts:mechanic'
        ORDER BY candidate_id ASC
        """
    ).fetchall()
    assert len(verdict_records) == 2, (
        "Both candidate evaluations must be logged to canon_verdicts"
    )
    for cand_id, cosine, lexical, verdict, rationale in verdict_records:
        assert cand_id in (
            "concepts:quantum-mechanics",
            "concepts:statistical-mechanics",
        )
        assert cosine >= 0.80
        assert lexical == 1.0, "Lexical must be 1.0 since both were corroborated"
        assert verdict == "distinct", "Ambiguity must yield 'distinct' verdict"
        assert "Ambiguous identity" in rationale


def test_challenge_gerund_lemma_divergence_causes_false_merge(
    test_db: sqlite3.Connection,
):
    """Demonstrates DEFECT: spaCy POS tagger lemma divergence between isolated noun phrases
    and compound noun phrases causes is_head_anchored_containment to silently fail corroboration,
    leading to an erroneous hard merge instead of ambiguous soft-linking.
    """
    v1, v2 = _make_orthogonal_vectors(dim=768, target_cosine=0.92)
    v_target = (v1 + v2) / np.linalg.norm(v1 + v2)

    test_db.execute(
        """
        INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition)
        VALUES ('concepts:supervised-learning', 'ml1.md', 'supervised learning', 'concepts', 'Learning with labels')
        """
    )
    test_db.execute(
        "INSERT INTO entity_vec (entity_id, partition_key, embedding) VALUES ('concepts:supervised-learning', 'concepts', ?)",
        (v1.astype(np.float32).tobytes(),),
    )
    test_db.execute(
        """
        INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition)
        VALUES ('concepts:reinforcement-learning', 'ml2.md', 'reinforcement learning', 'concepts', 'Learning with rewards')
        """
    )
    test_db.execute(
        "INSERT INTO entity_vec (entity_id, partition_key, embedding) VALUES ('concepts:reinforcement-learning', 'concepts', ?)",
        (v2.astype(np.float32).tobytes(),),
    )
    test_db.commit()

    # Mention: "learning"
    # Even though both 'supervised learning' and 'reinforcement learning' share head token 'learning',
    # 'supervised learning' has lemma 'learning' (NOUN) while 'learning' has lemma 'learn' (VERB).
    # Consequently, 'supervised learning' is rejected, leaving only 'reinforcement learning',
    # causing an unintended hard merge!
    res = resolve(
        name="learning",
        kind="entity",
        conn=test_db,
        embedding=v_target.tolist(),
        definition_ctx="Machine learning paradigm",
    )
    # If the bug is present, res will be hard-merged into reinforcement-learning!
    assert res.canonical_id == "concepts:learning", (
        f"DEFECT: Mention 'learning' falsely hard-merged into '{res.canonical_id}' "
        "because lemma divergence rejected 'supervised learning'!"
    )


# ==============================================================================
# CHALLENGE 4: PREDICATES SPACE L GUARD
# ==============================================================================


def test_challenge_predicates_guard_never_queries_entity_vec_or_enters_tier2(
    test_db: sqlite3.Connection,
):
    """Assert predicates never query entity_vec, never insert into close_matches or canon_verdicts,
    and never resolve via Tier 2.
    """
    v_pred = np.random.default_rng(999).standard_normal(768).astype(np.float32)
    v_pred /= np.linalg.norm(v_pred)

    # Seed an entity in entity_vec with the exact same vector
    test_db.execute(
        """
        INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition)
        VALUES ('predicates:causes', 'logic.md', 'causes', 'predicates', 'Causal relation')
        """
    )
    test_db.execute(
        "INSERT INTO entity_vec (entity_id, partition_key, embedding) VALUES ('predicates:causes', 'predicates', ?)",
        (v_pred.tobytes(),),
    )
    test_db.commit()

    # 1. Base predicate match
    res_base = resolve(
        name="causes",
        kind="predicate",
        conn=test_db,
        embedding=v_pred.tolist(),
    )
    assert res_base.tier == TierEnum.TIER_1_EXACT
    assert res_base.canonical_id == "causes"

    # 2. Synset/antonym-guarded match in Space L
    res_synset = resolve(
        name="leads to",
        kind="predicate",
        conn=test_db,
        embedding=v_pred.tolist(),
    )
    assert res_synset.tier == TierEnum.TIER_1_EXACT
    assert res_synset.canonical_id == "causes"

    # 3. Novel predicate with high vector similarity to existing predicates:causes
    # Must NOT enter Tier 2, must NOT auto-merge via embedding
    res_novel = resolve(
        name="fundamentally modulates the probability of",
        kind="predicate",
        conn=test_db,
        embedding=v_pred.tolist(),
    )
    assert res_novel.tier == TierEnum.NEW_INDUCTION
    assert res_novel.is_new is True
    assert res_novel.canonical_id.startswith("predicates:")
    assert res_novel.method == MatchMethod.NEW.value

    # 4. Assert zero rows in canon_verdicts for predicates
    verdict_count = test_db.execute(
        "SELECT COUNT(*) FROM canon_verdicts WHERE kind = 'predicate'"
    ).fetchone()[0]
    assert verdict_count == 0, "Predicates must NEVER log to canon_verdicts"

    # 5. Assert zero rows in close_matches for predicates
    cm_count = test_db.execute(
        "SELECT COUNT(*) FROM close_matches WHERE src_id LIKE 'predicates:%' OR dst_id LIKE 'predicates:%'"
    ).fetchone()[0]
    assert cm_count == 0, "Predicates must NEVER create close_matches"


def test_challenge_predicate_hijacked_by_kb_entity(test_db: sqlite3.Connection):
    """When an entity in leaf_nodes shares a surface name with a base predicate (e.g. 'causes'),
    _resolve_tier1_direct queries InMemoryLookupKB before checking kind == 'predicate',
    erroneously resolving the predicate to the entity node ID!
    """
    test_db.execute(
        """
        INSERT INTO leaf_nodes (node_id, source_leaf_id, name, namespace, definition)
        VALUES ('concepts:causes-of-war', 'war.md', 'causes', 'concepts', 'Factors leading to war')
        """
    )
    test_db.commit()

    res = resolve(
        name="causes",
        kind="predicate",
        conn=test_db,
    )
    # The predicate 'causes' is one of the 14 base predicates.
    # It must resolve to 'causes', NEVER to an entity 'concepts:causes-of-war'!
    assert res.canonical_id == "causes", (
        f"Predicate was hijacked by entity node: {res.canonical_id}"
    )
