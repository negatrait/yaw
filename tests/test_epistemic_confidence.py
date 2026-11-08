"""Real, un-mocked SQLite tests for Epistemic Confidence and Bayesian Markov Graph Engine."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from yaw.core.types import detect_scalar_literal
from yaw.daemon.graph.inference.chat import extract_completion_logprob_confidence
from yaw.daemon.graph.ppr import build_transition_matrix, run_ppr
from yaw.daemon.graph.storage.db import initialize_schema, open_managed_yaw_db
from yaw.daemon.graph.storage.graph_ops import (
    bayesian_noisy_or,
    load_graph_topology,
    pool_confidences,
    upsert_semantic_edges,
)
from yaw.daemon.graph.storage.shacl import (
    auto_heal_deterministic_shacl,
    format_subgraph_turtle,
    lint_graph_shacl,
    query_graph_provenance,
    serialize_graph_turtle,
)


def test_pool_confidences_scale_idempotency_and_unknown() -> None:
    """Verify Logarithmic Opinion Pool / Geometric Mean scale idempotency and honest unknown."""
    # Scale idempotency: pooling identical confidences preserves scale
    assert pool_confidences(0.8, 0.8) == pytest.approx(0.8)
    assert pool_confidences(0.8, 0.8, 0.8) == pytest.approx(0.8)
    assert pool_confidences(0.5, 0.5) == pytest.approx(0.5)

    # Honest unknown: None signals are omitted from geometric weighting
    assert pool_confidences(None, 0.8) == pytest.approx(0.8)
    assert pool_confidences(0.8, None) == pytest.approx(0.8)
    assert pool_confidences(None, 0.8, None) == pytest.approx(0.8)

    # Strict unknown: if all signals are None or empty, return None
    assert pool_confidences(None, None) is None
    assert pool_confidences() is None

    # Multi-signal geometric mean: prod(c_i) ** (1/k)
    c1, c2 = 0.9, 0.4
    expected_geom = math.exp((math.log(0.9) + math.log(0.4)) / 2.0)
    assert pool_confidences(c1, c2) == pytest.approx(expected_geom)

    # extract_completion_logprob_confidence contract
    # 1. With token logprobs structure
    resp_with_lp = {
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {"token": "Alan", "logprob": -0.10536, "bytes": [65, 108, 97, 110]},
                        {"token": " Turing", "logprob": -0.05129, "bytes": [32, 84, 117, 114, 105, 110, 103]},
                    ]
                }
            }
        ]
    }
    full_conf = extract_completion_logprob_confidence(resp_with_lp)
    assert full_conf is not None
    assert 0.0 < full_conf <= 1.0
    # Mean logprob: (-0.10536 + -0.05129) / 2 = -0.078325 -> exp(-0.078325) ~ 0.92466
    assert full_conf == pytest.approx(math.exp((-0.10536 + -0.05129) / 2.0), abs=1e-4)

    # Span-specific confidence
    span_conf = extract_completion_logprob_confidence(resp_with_lp, span_text="Turing")
    assert span_conf is not None
    assert span_conf == pytest.approx(math.exp(-0.05129), abs=1e-4)

    # 2. Strict honest unknown when logprobs are absent
    resp_without_lp = {"choices": [{"message": {"content": "Alan Turing"}}]}
    assert extract_completion_logprob_confidence(resp_without_lp) is None
    assert extract_completion_logprob_confidence(None) is None


def test_bayesian_noisy_or_additive_evidence() -> None:
    """Verify Bayesian Noisy-OR complement aggregation compounding independent evidence."""
    # 1 - (1 - 0.8) * (1 - 0.8) = 1 - 0.04 = 0.96
    assert bayesian_noisy_or([0.8, 0.8]) == pytest.approx(0.96)
    # 1 - (1 - 0.5) ** 3 = 1 - 0.125 = 0.875
    assert bayesian_noisy_or([0.5, 0.5, 0.5]) == pytest.approx(0.875)

    # Honest unknown handling
    assert bayesian_noisy_or([None, 0.8]) == pytest.approx(0.8)
    assert bayesian_noisy_or([0.8, None]) == pytest.approx(0.8)
    assert bayesian_noisy_or([None, None]) is None
    assert bayesian_noisy_or([]) is None

    # Single evidence preservation
    assert bayesian_noisy_or([0.7]) == pytest.approx(0.7)


def test_literal_disjointness_in_yaw_db(tmp_path: Path) -> None:
    """Verify strict literal disjointness in real yaw.db: scalar literals never enter entities or edges."""
    # 1. Test detect_scalar_literal pattern coverage
    assert detect_scalar_literal("$100")[0] is True
    assert detect_scalar_literal("€50.50")[0] is True
    assert detect_scalar_literal("100 USD")[0] is True
    assert detect_scalar_literal("15ms")[0] is True
    assert detect_scalar_literal("p99 100ms")[0] is True
    assert detect_scalar_literal("99.9%")[0] is True
    assert detect_scalar_literal("100MB")[0] is True
    assert detect_scalar_literal("4GB")[0] is True
    assert detect_scalar_literal("500 requests")[0] is True
    assert detect_scalar_literal("1000 tokens")[0] is True
    assert detect_scalar_literal("0.28")[0] is True
    assert detect_scalar_literal("401")[0] is True
    # Non-literals (concepts)
    assert detect_scalar_literal("Alan Turing")[0] is False
    assert detect_scalar_literal("Bletchley Park")[0] is False
    assert detect_scalar_literal("Gateway Alpha")[0] is False

    # 2. Real SQLite yaw.db indexing
    db_path = tmp_path / "test_literals.db"
    managed = open_managed_yaw_db(db_path)
    with managed.transaction() as conn:
        initialize_schema(conn)

        # Prepare concept entities
        e_srv = ("ent-server", "API Server", "Core HTTP API service", 0.95)
        e_gw = ("ent-gw", "API Gateway", "Reverse proxy gateway", 0.90)

        # Insert chunks
        conn.execute(
            """
            INSERT INTO chunks (chunk_id, rel_posix_path, start_line, end_line, title)
            VALUES ('c1', 'doc1.md', 1, 10, 'Server metrics')
            """
        )

        # Insert entities
        conn.execute(
            "INSERT INTO entities (entity_id, name, definition, confidence) VALUES (?, ?, ?, ?)",
            e_srv,
        )
        conn.execute(
            "INSERT INTO entities (entity_id, name, definition, confidence) VALUES (?, ?, ?, ?)",
            e_gw,
        )
        conn.execute(
            "INSERT INTO entity_labels (label_norm, entity_id, label_kind, raw_label, chunk_id) VALUES (?, ?, ?, ?, ?)",
            ("api server", "ent-server", "skos:prefLabel", "API Server", "c1"),
        )
        conn.execute(
            "INSERT INTO entity_labels (label_norm, entity_id, label_kind, raw_label, chunk_id) VALUES (?, ?, ?, ?, ?)",
            ("api gateway", "ent-gw", "skos:prefLabel", "API Gateway", "c1"),
        )

        # Insert scalar literal facts
        literal_samples = [
            ("fact-1", "ent-server", "latency", "15ms", "xsd:string", 0.92),
            ("fact-2", "ent-server", "availability", "99.9%", "xsd:string", 0.98),
            ("fact-3", "ent-gw", "cost", "$100", "xsd:string", 0.88),
            ("fact-4", "ent-gw", "rate_limit", "500 requests", "xsd:string", 0.90),
            ("fact-5", "ent-server", "status_code", "401", "xsd:integer", 0.95),
        ]
        for fid, sid, pred, lit_lex, dt, conf in literal_samples:
            raw_json = json.dumps([sid, pred, lit_lex])
            defn = f"{sid} {pred} is {lit_lex}"
            conn.execute(
                """
                INSERT INTO facts (
                    fact_id, subj_id, pred, obj_id, definition, obj_kind,
                    obj_literal_lex, obj_datatype, raw_triple_json, confidence
                ) VALUES (?, ?, ?, NULL, ?, 'literal', ?, ?, ?, ?)
                """,
                (fid, sid, pred, defn, lit_lex, dt, raw_json, conf),
            )
            conn.execute(
                "INSERT INTO fact_sources (fact_id, chunk_id, atomic_confidence) VALUES (?, 'c1', ?)",
                (fid, conf),
            )

        # Also insert an ObjectProperty fact between concepts
        conn.execute(
            """
            INSERT INTO facts (
                fact_id, subj_id, pred, obj_id, definition, obj_kind,
                obj_literal_lex, obj_datatype, raw_triple_json, confidence
            ) VALUES (
                'fact-obj', 'ent-gw', 'routes to', 'ent-server',
                'Gateway routes to server.', 'entity', NULL, '',
                '["ent-gw", "routes to", "ent-server"]', 0.85
            )
            """
        )
        conn.execute("INSERT INTO fact_sources (fact_id, chunk_id, atomic_confidence) VALUES ('fact-obj', 'c1', 0.85)")
        conn.execute("INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('ent-gw', 'ent-server', 'fact', 0.85)")

    with managed.readonly() as conn:
        # Assert ZERO scalar literal rows in entities
        literal_names = ["15ms", "99.9%", "$100", "500 requests", "401"]
        placeholders = ",".join("?" for _ in literal_names)
        ent_hits = conn.execute(
            f"SELECT name FROM entities WHERE name IN ({placeholders})", tuple(literal_names)
        ).fetchall()
        assert len(ent_hits) == 0

        # Assert ZERO edges to literal objects in edges table
        edge_hits = conn.execute(
            f"SELECT dst_id FROM edges WHERE dst_id IN ({placeholders})", tuple(literal_names)
        ).fetchall()
        assert len(edge_hits) == 0

        # Assert literal facts exist strictly in facts table with obj_id IS NULL and calibrated confidence
        lit_fact_rows = conn.execute(
            """
            SELECT fact_id, subj_id, pred, obj_id, obj_kind, obj_literal_lex, confidence
            FROM facts WHERE obj_kind = 'literal'
            """
        ).fetchall()
        assert len(lit_fact_rows) == 5
        for r in lit_fact_rows:
            assert r[3] is None  # obj_id IS NULL
            assert r[4] == "literal"
            assert r[6] is not None and 0.0 < float(r[6]) <= 1.0

        # Assert SHACL linting passes cleanly
        violations = lint_graph_shacl(conn)
        assert violations == []


def test_empirical_bayes_s27_and_cagle_resolution(tmp_path: Path) -> None:
    """Verify empirical Bayes factor resolution and Cagle reification in auto_heal_deterministic_shacl."""
    db_path = tmp_path / "test_bayes_healing.db"
    managed = open_managed_yaw_db(db_path)
    with managed.transaction() as conn:
        initialize_schema(conn)

        # Setup 4 concepts: A, B, C, D across chunks c1 and c2
        for cid, name in [
            ("ent-a", "Concept A"),
            ("ent-b", "Concept B"),
            ("ent-c", "Concept C"),
            ("ent-d", "Concept D"),
        ]:
            conn.execute(
                "INSERT INTO entities (entity_id, name, definition, confidence) VALUES (?, ?, '', 0.9)", (cid, name)
            )
            conn.execute(
                """
                INSERT INTO entity_labels (label_norm, entity_id, label_kind, raw_label, chunk_id)
                VALUES (?, ?, 'skos:prefLabel', ?, 'c1')
                """,
                (name.lower(), cid, name),
            )

        # --- Case 1: Step 1 (Existing fact between A and C prunes redundant skos:related) ---
        conn.execute(
            """
            INSERT INTO facts (fact_id, subj_id, pred, obj_id, obj_kind, raw_triple_json, confidence)
            VALUES ('f-ac', 'ent-a', 'connects', 'ent-c', 'entity', '["ent-a", "connects", "ent-c"]', 0.8)
            """
        )
        conn.execute("INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('ent-a', 'ent-c', 'skos:broader', 0.8)")
        conn.execute("INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('ent-a', 'ent-c', 'skos:related', 0.7)")
        conn.execute("INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('ent-c', 'ent-a', 'skos:related', 0.7)")

        # Run auto-healing
        healed_step1 = auto_heal_deterministic_shacl(conn)
        assert healed_step1 >= 1
        rel_ac = conn.execute(
            "SELECT COUNT(*) FROM edges WHERE kind = 'skos:related' AND src_id = 'ent-a' AND dst_id = 'ent-c'"
        ).fetchone()[0]
        assert rel_ac == 0  # Redundant skos:related pruned because fact already exists!
        broader_ac = conn.execute(
            "SELECT COUNT(*) FROM edges WHERE kind = 'skos:broader' AND src_id = 'ent-a' AND dst_id = 'ent-c'"
        ).fetchone()[0]
        assert broader_ac == 1  # Broader edge preserved

        # --- Case 2: Step 2 Cagle Reification (above-average confidences with shared chunk reify to fact) ---
        conn.execute("DELETE FROM edges;")
        upsert_semantic_edges(conn, [("ent-a", "ent-b", "skos:broader", 0.9)])
        upsert_semantic_edges(conn, [("ent-b", "ent-d", "skos:broader", 0.9)])
        upsert_semantic_edges(conn, [("ent-a", "ent-d", "skos:related", 0.88)])
        # Both ent-a and ent-d share chunk c1 from entity_labels
        healed_cagle = auto_heal_deterministic_shacl(conn)
        assert healed_cagle >= 1
        # skos:related edge between A and D was deleted
        rel_ad = conn.execute(
            "SELECT COUNT(*) FROM edges WHERE kind = 'skos:related' AND src_id = 'ent-a' AND dst_id = 'ent-d'"
        ).fetchone()[0]
        assert rel_ad == 0
        # Reified fact was inserted into facts table
        reified_fact = conn.execute(
            "SELECT pred, confidence FROM facts WHERE subj_id = 'ent-a' AND obj_id = 'ent-d'"
        ).fetchone()
        assert reified_fact is not None
        assert reified_fact[0] == "relatedTo"
        assert reified_fact[1] is not None and float(reified_fact[1]) > 0.8

        # --- Case 3: Step 3 Case A (w_chain >= w_rel prunes skos:related when no shared chunk) ---
        # Disjoint chunks: ent-a on c1, ent-d on c2
        conn.execute("DELETE FROM edges;")
        conn.execute("DELETE FROM facts;")
        conn.execute("UPDATE entity_labels SET chunk_id = 'c2' WHERE entity_id = 'ent-d'")
        upsert_semantic_edges(conn, [("ent-a", "ent-b", "skos:broader", 0.9)])
        upsert_semantic_edges(conn, [("ent-b", "ent-d", "skos:broader", 0.9)])
        upsert_semantic_edges(conn, [("ent-a", "ent-d", "skos:related", 0.6)])

        healed_case_a = auto_heal_deterministic_shacl(conn)
        assert healed_case_a >= 1
        rel_ad_after = conn.execute(
            "SELECT COUNT(*) FROM edges WHERE kind = 'skos:related' AND src_id = 'ent-a' AND dst_id = 'ent-d'"
        ).fetchone()[0]
        assert rel_ad_after == 0  # Pruned because broader chain has higher epistemic confidence!
        broader_ab = conn.execute(
            "SELECT weight FROM edges WHERE kind = 'skos:broader' AND src_id = 'ent-a' AND dst_id = 'ent-b'"
        ).fetchone()
        assert broader_ab is not None

        # --- Case 4: Step 3 Case B (w_rel > w_chain keeps skos:related and demotes weakest broader link) ---
        # Reset: broader A -> B (w=0.8), B -> C (w=0.3)
        # Related: A -> C (w=0.85)
        # w_chain = pool_confidences(0.8, 0.3) ~ 0.49 < w_rel (0.85) -> keep related, demote weakest broader (B -> C)
        conn.execute("DELETE FROM edges;")
        conn.execute("UPDATE entity_labels SET chunk_id = 'c3' WHERE entity_id = 'ent-c'")
        upsert_semantic_edges(conn, [("ent-a", "ent-b", "skos:broader", 0.8)])
        upsert_semantic_edges(conn, [("ent-b", "ent-c", "skos:broader", 0.3)])
        upsert_semantic_edges(conn, [("ent-a", "ent-c", "skos:related", 0.85)])

        healed_case_b = auto_heal_deterministic_shacl(conn)
        assert healed_case_b >= 1
        # Related edge A <-> C is preserved at calibrated weight 0.85
        rel_row = conn.execute(
            "SELECT weight FROM edges WHERE kind = 'skos:related' AND src_id = 'ent-a' AND dst_id = 'ent-c'"
        ).fetchone()
        assert rel_row is not None
        assert abs(float(rel_row[0]) - 0.85) < 1e-6

        # Weakest broader link B -> C (weight 0.3) was broken/demoted to skos:related
        # while preserving its calibrated weight 0.3!
        broader_bc = conn.execute(
            "SELECT COUNT(*) FROM edges WHERE kind = 'skos:broader' AND src_id = 'ent-b' AND dst_id = 'ent-c'"
        ).fetchone()[0]
        assert broader_bc == 0  # Broken!
        demoted_bc = conn.execute(
            "SELECT weight FROM edges WHERE kind = 'skos:related' AND src_id = 'ent-b' AND dst_id = 'ent-c'"
        ).fetchone()
        assert demoted_bc is not None
        assert abs(float(demoted_bc[0]) - 0.3) < 1e-6  # Preserves calibrated weight!


def test_markov_walk_divergence_and_turtle_payload(tmp_path: Path) -> None:
    """Verify calibrated Markov PPR rankings diverge from uniform weights, and Turtle conforms to W3C."""
    db_path = tmp_path / "test_markov_divergence.db"
    managed = open_managed_yaw_db(db_path)
    with managed.transaction() as conn:
        initialize_schema(conn)

        # Graph structure:
        # S connects to N1 and N2 with passage edges.
        # N1 connects to T1 with fact edge.
        # N2 connects to T2 with fact edge.
        # T1 has high confidence prior (0.95), T2 has low confidence prior (0.10).
        conn.execute(
            """
            INSERT INTO chunks (chunk_id, rel_posix_path, start_line, end_line, title)
            VALUES ('chunk-s', 'doc1.md', 1, 10, 'Seed chunk')
            """
        )
        conn.execute(
            "INSERT INTO entities (entity_id, name, definition, confidence) VALUES ('ent-n1', 'Node 1', 'Int 1', 0.90)"
        )
        conn.execute(
            "INSERT INTO entities (entity_id, name, definition, confidence) VALUES ('ent-n2', 'Node 2', 'Int 2', 0.90)"
        )
        conn.execute(
            "INSERT INTO entities (entity_id, name, definition, confidence) VALUES (?, ?, ?, ?)",
            ("ent-t1", "Target 1", "Dest 1", 0.95),
        )
        conn.execute(
            "INSERT INTO entities (entity_id, name, definition, confidence) VALUES (?, ?, ?, ?)",
            ("ent-t2", "Target 2", "Dest 2", 0.10),
        )

        for eid, name in [("ent-n1", "Node 1"), ("ent-n2", "Node 2"), ("ent-t1", "Target 1"), ("ent-t2", "Target 2")]:
            conn.execute(
                """
                INSERT INTO entity_labels (label_norm, entity_id, label_kind, raw_label, chunk_id)
                VALUES (?, ?, 'skos:prefLabel', ?, 'chunk-s')
                """,
                (name.lower(), eid, name),
            )

        # Passage edges from chunk-s to N1 and N2
        conn.execute("INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('chunk-s', 'ent-n1', 'passage', 1.0)")
        conn.execute("INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('chunk-s', 'ent-n2', 'passage', 1.0)")

        # Fact edges: N1 -> T1, N2 -> T2
        conn.execute(
            """
            INSERT INTO facts (
                fact_id, subj_id, pred, obj_id, definition, obj_kind, raw_triple_json, confidence
            ) VALUES (
                'f1', 'ent-n1', 'leads to', 'ent-t1', 'Node 1 leads to Target 1.',
                'entity', '["ent-n1", "leads to", "ent-t1"]', 0.9
            )
            """
        )
        conn.execute(
            """
            INSERT INTO facts (
                fact_id, subj_id, pred, obj_id, definition, obj_kind, raw_triple_json, confidence
            ) VALUES (
                'f2', 'ent-n2', 'leads to', 'ent-t2', 'Node 2 leads to Target 2.',
                'entity', '["ent-n2", "leads to", "ent-t2"]', 0.9
            )
            """
        )
        conn.execute("INSERT INTO fact_sources (fact_id, chunk_id, atomic_confidence) VALUES ('f1', 'chunk-s', 0.9)")
        conn.execute("INSERT INTO fact_sources (fact_id, chunk_id, atomic_confidence) VALUES ('f2', 'chunk-s', 0.9)")
        conn.execute("INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('ent-n1', 'ent-t1', 'fact', 0.9)")
        conn.execute("INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('ent-n2', 'ent-t2', 'fact', 0.9)")

    with managed.readonly() as conn:
        chunk_ids, entity_ids, topo_edges, priors = load_graph_topology(conn)

        # 1. Build transition matrix with real calibrated Bayesian node priors
        csr_calibrated = build_transition_matrix(chunk_ids, entity_ids, topo_edges, node_priors=priors)
        ppr_calibrated = run_ppr(csr_calibrated, {"chunk-s": 1.0})

        idx_t1 = csr_calibrated.node_to_idx["ent-t1"]
        idx_t2 = csr_calibrated.node_to_idx["ent-t2"]

        score_t1 = float(ppr_calibrated[idx_t1])
        score_t2 = float(ppr_calibrated[idx_t2])

        # Under calibrated priors, T1 (prior 0.95) transitions significantly more probability than T2 (prior 0.10)
        assert score_t1 > score_t2
        assert score_t1 > 0.0

        # 2. Turtle serialization verification
        turtle_full = serialize_graph_turtle(conn)
        subgraph_turtle = format_subgraph_turtle(conn, ["chunk-s"])

        for ttl in (turtle_full, subgraph_turtle):
            # Standard W3C prefix header with default prefix : <urn:graph:>
            assert "@prefix : <urn:graph:> ." in ttl
            assert "@prefix skos: <http://www.w3.org/2004/02/skos/core#> ." in ttl
            assert "@prefix prov: <http://www.w3.org/ns/prov#> ." in ttl

            # Zero custom or legacy namespaces
            assert "yaw:" not in ttl
            assert "dct:" not in ttl

            # prov:value calibration emission
            assert 'prov:value "' in ttl

        # 3. ProvenanceConfidenceShape verification
        prov = query_graph_provenance(conn)
        assert "provenance_sources" in prov
        assert prov["provenance_sources"]["fact_sources"] == 2

        # Zero violations initially
        violations = lint_graph_shacl(conn)
        assert violations == []

    # Verify ProvenanceConfidenceShape flags invalid confidence (> 1.0 or <= 0.0)
    with managed.transaction() as conn:
        conn.execute("UPDATE fact_sources SET atomic_confidence = 1.85 WHERE fact_id = 'f1'")
    with managed.readonly() as conn:
        violations = lint_graph_shacl(conn)
        conf_violations = [v for v in violations if v.shape == "ProvenanceConfidenceShape"]
        assert len(conf_violations) == 1
        assert "out of valid bounds" in conf_violations[0].message


def test_w3c_guardrail_1_homograph_only_different_from(tmp_path: Path) -> None:
    """Verify Guardrail 1: owl:differentFrom is strictly emitted for homographs/high lexical overlap, not noise."""
    from yaw.daemon.graph.planner import apply_synonym_adjudication

    db_path = tmp_path / "yaw.db"
    managed = open_managed_yaw_db(db_path)
    with managed.transaction() as conn:
        initialize_schema(conn)
        # 3 entity pairs:
        # Pair 1 (Homograph): "Region" (cloud region) vs "Region" (anatomical region) -> same normalized name
        # Pair 2 (High lexical overlap >= avg_alt_label_sim): "Microservice" vs "Microservices"
        # Pair 3 (Low lexical / completely distinct): "Region" vs "regressions"
        conn.execute(
            "INSERT INTO chunks (chunk_id, rel_posix_path, start_line, end_line) VALUES ('c1', 'a.txt', 1, 10)"
        )
        conn.execute("INSERT INTO entities (entity_id, name, definition) VALUES ('e1', 'Region', 'A cloud region')")
        conn.execute(
            "INSERT INTO entities (entity_id, name, definition) VALUES ('e2', 'Region', 'An anatomical body area')"
        )
        conn.execute(
            "INSERT INTO entities (entity_id, name, definition) VALUES ('e3', 'regressions', 'Statistical models')"
        )
        conn.execute(
            "INSERT INTO entities (entity_id, name, definition) VALUES ('e4', 'Microservice', 'A small service')"
        )
        conn.execute(
            "INSERT INTO entities (entity_id, name, definition) VALUES ('e5', 'Microservices', 'Multiple services')"
        )

    with managed.transaction() as conn:
        candidates = [
            {
                "pair_id": "e1|e2",
                "entity_a": "e1",
                "name_a": "Region",
                "entity_b": "e2",
                "name_b": "Region",
                "lexical_sim": 1.0,
                "similarity": 0.5,
            },
            {
                "pair_id": "e1|e3",
                "entity_a": "e1",
                "name_a": "Region",
                "entity_b": "e3",
                "name_b": "regressions",
                "lexical_sim": 0.35,
                "similarity": 0.4,
            },
            {
                "pair_id": "e4|e5",
                "entity_a": "e4",
                "name_a": "Microservice",
                "entity_b": "e5",
                "name_b": "Microservices",
                "lexical_sim": 0.95,
                "similarity": 0.95,
            },
        ]
        cand_meta = {
            "e1": ("Region", "A cloud region"),
            "e2": ("Region", "An anatomical body area"),
            "e3": ("regressions", "Statistical models"),
            "e4": ("Microservice", "A small service"),
            "e5": ("Microservices", "Multiple services"),
        }
        decisions = [
            # altLabel decision establishing avg_alt_label_sim ~ 0.95
            {"pair_id": "e4|e5", "relation": "skos:altLabel", "canonical": "Microservice"},
            # Stage 3 returning differentFrom on homograph (e1|e2)
            {"pair_id": "e1|e2", "relation": "owl:differentFrom", "reasoning": "Distinct meanings"},
            # Stage 3 returning differentFrom on non-homograph (e1|e3)
            {"pair_id": "e1|e3", "relation": "owl:differentFrom", "reasoning": "Distinct concepts"},
        ]
        rlm_stats: dict[str, Any] = {"merged_entities": 0, "different_from_edges": 0}
        _, sem_batch = apply_synonym_adjudication(
            conn,
            candidates,
            cand_meta,
            decisions,
            fact_linked_pairs=set(),
            synonymy_threshold=0.8,
            rlm_stats=rlm_stats,
        )

        kinds = {(s, o, k) for s, o, k, _, _ in sem_batch}
        # Homograph pair e1|e2 MUST receive owl:differentFrom
        assert ("e1", "e2", "owl:differentFrom") in kinds or ("e2", "e1", "owl:differentFrom") in kinds
        # Non-homograph pair e1|e3 MUST NOT receive owl:differentFrom (emits NO edge)
        assert ("e1", "e3", "owl:differentFrom") not in kinds
        assert ("e3", "e1", "owl:differentFrom") not in kinds
        assert rlm_stats["different_from_edges"] == 1


def test_w3c_guardrail_2_head_noun_modifier_specialization(tmp_path: Path) -> None:
    """Verify Guardrail 2: Head-Noun Modifier Specialization routes to skos:broader; disjoint tokens to skos:related."""
    from yaw.daemon.graph.planner import (
        _has_disjoint_non_acronym_tokens,
        apply_synonym_adjudication,
        is_modifier_head_specialization,
    )

    # Invariant checks
    assert is_modifier_head_specialization("Redwood", "Redwood Router") is True
    assert is_modifier_head_specialization("Redwood Router", "Redwood") is True
    assert is_modifier_head_specialization("Redwood", "Redwood secrets") is True
    assert is_modifier_head_specialization("Redwood", "RedwoodTenant") is True
    assert is_modifier_head_specialization("Redwood", "Redwood") is False

    # Disjoint tokens check
    assert _has_disjoint_non_acronym_tokens("Redwood", "CRDs") is True
    assert _has_disjoint_non_acronym_tokens("CRD", "Custom Resource Definition") is False  # Acronym correspondence
    assert _has_disjoint_non_acronym_tokens("Redwood", "Redwood Router") is False  # Shares "redwood"

    # Behavior in apply_synonym_adjudication
    db_path = tmp_path / "yaw.db"
    managed = open_managed_yaw_db(db_path)
    with managed.transaction() as conn:
        initialize_schema(conn)
        conn.execute(
            "INSERT INTO chunks (chunk_id, rel_posix_path, start_line, end_line) VALUES ('c1', 'a.txt', 1, 10)"
        )
        conn.execute("INSERT INTO entities (entity_id, name, definition) VALUES ('e1', 'Redwood', 'Core system')")
        conn.execute(
            "INSERT INTO entities (entity_id, name, definition) VALUES ('e2', 'Redwood Router', 'Routing component')"
        )
        conn.execute("INSERT INTO entities (entity_id, name, definition) VALUES ('e3', 'CRDs', 'Custom resources')")

    with managed.transaction() as conn:
        candidates = [
            {
                "pair_id": "e1|e2",
                "entity_a": "e1",
                "name_a": "Redwood",
                "entity_b": "e2",
                "name_b": "Redwood Router",
                "lexical_sim": 0.7,
                "similarity": 0.8,
            },
            {
                "pair_id": "e1|e3",
                "entity_a": "e1",
                "name_a": "Redwood",
                "entity_b": "e3",
                "name_b": "CRDs",
                "lexical_sim": 0.1,
                "similarity": 0.75,
            },
        ]
        cand_meta = {
            "e1": ("Redwood", "Core system"),
            "e2": ("Redwood Router", "Routing component"),
            "e3": ("CRDs", "Custom resources"),
        }
        # Stage 1 LLM mistakenly proposes equivalence / altLabel on both
        decisions = [
            {"pair_id": "e1|e2", "relation": "skos:altLabel", "reasoning": "Related to redwood"},
            {"pair_id": "e1|e3", "relation": "skos:altLabel", "reasoning": "Related Kubernetes terms"},
        ]
        rlm_stats: dict[str, Any] = {"merged_entities": 0, "broader_edges": 0, "related_edges": 0}
        merges, sem_batch = apply_synonym_adjudication(
            conn,
            candidates,
            cand_meta,
            decisions,
            fact_linked_pairs=set(),
            synonymy_threshold=0.7,
            rlm_stats=rlm_stats,
        )

        # Neither must be merged into a single entity!
        assert merges == []
        assert rlm_stats["merged_entities"] == 0

        # Redwood Router must be routed to skos:broader Redwood (child=e2, parent=e1)
        broader_edges = [(s, o) for s, o, k, _, _ in sem_batch if k == "skos:broader"]
        assert ("e2", "e1") in broader_edges
        assert rlm_stats["broader_edges"] == 1

        # Redwood vs CRDs must be routed to skos:related
        related_edges = [(s, o) for s, o, k, _, _ in sem_batch if k == "skos:related"]
        assert ("e1", "e3") in related_edges or ("e3", "e1") in related_edges
        assert rlm_stats["related_edges"] == 1


def test_w3c_guardrail_3_case_insensitive_consolidation(tmp_path: Path) -> None:
    """Verify Guardrail 3: Case-insensitive label matches consolidate to properly cased prefLabel."""
    from yaw.daemon.graph.planner import prepare_synonym_candidates

    db_path = tmp_path / "yaw.db"
    managed = open_managed_yaw_db(db_path)
    with managed.transaction() as conn:
        initialize_schema(conn)
        conn.execute(
            "INSERT INTO chunks (chunk_id, rel_posix_path, start_line, end_line) VALUES ('c1', 'a.txt', 1, 10)"
        )
        conn.execute("INSERT INTO entities (entity_id, name, definition) VALUES ('e1', 'canary', 'Deployment test')")
        conn.execute(
            "INSERT INTO entities (entity_id, name, definition) VALUES ('e2', 'Canary', 'Deployment pattern test')"
        )
        conn.execute(
            """
            INSERT INTO entity_labels (label_norm, entity_id, label_kind, raw_label, chunk_id)
            VALUES ('canary', 'e1', 'skos:prefLabel', 'canary', 'c1')
            """
        )
        conn.execute(
            """
            INSERT INTO entity_labels (label_norm, entity_id, label_kind, raw_label, chunk_id)
            VALUES ('canary', 'e2', 'skos:prefLabel', 'Canary', 'c1')
            """
        )

        import numpy as np

        from yaw.daemon.graph.storage.vectors import upsert_entity_vectors

        vec = np.ones(384, dtype=np.float32) / math.sqrt(384)
        upsert_entity_vectors(conn, [("e1", vec), ("e2", vec)])

    with managed.transaction() as conn:
        entity_nodes = [
            ("e1", "canary", "Deployment test"),
            ("e2", "Canary", "Deployment pattern test"),
        ]
        rlm_stats: dict[str, Any] = {"merged_entities": 0}
        cands, cand_meta, _, auto_merges = prepare_synonym_candidates(
            conn,
            entity_nodes,
            synonymy_threshold=0.8,
            synonymy_top_k=5,
            rlm_stats=rlm_stats,
        )

        # Auto-merged under Guardrail 3
        assert len(auto_merges) == 1
        assert rlm_stats["merged_entities"] == 1
        # Prefers properly capitalized "Canary"
        dst_id = auto_merges[0].canonical_id
        assert cand_meta[dst_id][0] == "Canary"
