"""Real SQLite and W3C tests for graph planner and apply_synonym_adjudication."""

import sqlite3
from pathlib import Path

from yaw.daemon.graph.planner import apply_synonym_adjudication
from yaw.daemon.graph.storage import open_yaw_db, upsert_entities, upsert_entity_labels


def _init_test_db(db_path: Path) -> sqlite3.Connection:
    conn = open_yaw_db(db_path)
    return conn


def test_apply_synonym_adjudication_directed_broader(tmp_path: Path) -> None:
    """Verify apply_synonym_adjudication creates directed skos:broader edges from source_id to target_id."""
    conn = _init_test_db(tmp_path / "yaw.db")
    try:
        # Materialize entity A (pilot tenants) and entity B (tenants)
        upsert_entities(
            conn,
            [
                ("e_pilot", "pilot tenants", "Low volume test tenants"),
                ("e_tenants", "tenants", "Provisioned customer organizations"),
            ],
        )
        upsert_entity_labels(
            conn,
            [
                ("pilot tenants", "e_pilot", "skos:prefLabel", "pilot tenants", "c1"),
                ("tenants", "e_tenants", "skos:prefLabel", "tenants", "c1"),
            ],
        )

        candidates = [
            {"entity_a": "e_pilot", "entity_b": "e_tenants", "similarity": 0.88},
        ]
        cand_meta = {
            "e_pilot": ("pilot tenants", "Low volume test tenants"),
            "e_tenants": ("tenants", "Provisioned customer organizations"),
        }
        decisions = [
            {
                "pair_id": "e_pilot|e_tenants",
                "relation": "skos:broader",
                "source_id": "e_pilot",
                "target_id": "e_tenants",
                "reasoning": "pilot tenants is a narrower subtype of tenants",
            }
        ]
        rlm_stats: dict[str, int] = {}

        staged_merges, semantic_batch = apply_synonym_adjudication(
            conn,
            candidates,
            cand_meta,
            decisions,
            fact_linked_pairs=set(),
            synonymy_threshold=0.5,
            rlm_stats=rlm_stats,
            corpus_root=tmp_path,
        )

        assert not staged_merges
        assert len(semantic_batch) == 1
        s_act, o_act, r_kind, _ = semantic_batch[0]
        assert s_act == "e_pilot"
        assert o_act == "e_tenants"
        assert r_kind == "skos:broader"
        assert rlm_stats.get("broader_edges") == 1

        # Check in SQLite edges table
        edge = conn.execute("SELECT src_id, dst_id, kind FROM edges WHERE kind = 'skos:broader'").fetchone()
        assert edge is not None
        assert edge[0] == "e_pilot"
        assert edge[1] == "e_tenants"
        assert edge[2] == "skos:broader"
    finally:
        conn.close()


def test_apply_synonym_adjudication_hidden_label_and_edc_canonicalization(tmp_path: Path) -> None:
    """Verify apply_synonym_adjudication demotes hidden labels and runs end-of-round EDC canonicalization."""
    conn = _init_test_db(tmp_path / "yaw.db")
    try:
        # Create entity A (multi-chunk heading) and entity B (1-chunk typo variant)
        upsert_entities(
            conn,
            [
                ("e_good", "Necessary", "Required condition"),
                ("e_typo", "Neccessary", "Typo of necessary"),
            ],
        )
        upsert_entity_labels(
            conn,
            [
                ("necessary", "e_good", "skos:prefLabel", "Necessary", "c1"),
                ("necessary", "e_good", "skos:altLabel", "necessary", "c2"),
                ("neccessary", "e_typo", "skos:prefLabel", "Neccessary", "c3"),
            ],
        )

        candidates = [
            {"entity_a": "e_typo", "entity_b": "e_good", "similarity": 0.95},
        ]
        cand_meta = {
            "e_typo": ("Neccessary", "Typo of necessary"),
            "e_good": ("Necessary", "Required condition"),
        }
        # Model classifies e_typo as skos:hiddenLabel of e_good
        decisions = [
            {
                "pair_id": "e_typo|e_good",
                "source_id": "e_typo",
                "target_id": "e_good",
                "relation": "skos:hiddenLabel",
                "label_type": "hiddenLabel",
                "reasoning": "Neccessary is a misspelled variant / hidden label of Necessary",
            }
        ]
        rlm_stats: dict[str, int] = {}

        staged_merges, semantic_batch = apply_synonym_adjudication(
            conn,
            candidates,
            cand_meta,
            decisions,
            fact_linked_pairs=set(),
            synonymy_threshold=0.5,
            rlm_stats=rlm_stats,
            corpus_root=tmp_path,
        )

        assert len(staged_merges) == 1
        assert staged_merges[0][0] == "e_typo"
        assert staged_merges[0][1] == "e_good"
        assert staged_merges[0][2] == "skos:hiddenLabel"
        assert rlm_stats.get("merged_entities") == 1

        # Check entity labels on e_good: 'necessary' must be skos:prefLabel, 'neccessary' must be skos:hiddenLabel
        pref = conn.execute(
            "SELECT DISTINCT label_norm FROM entity_labels WHERE entity_id = 'e_good' AND label_kind = 'skos:prefLabel'"
        ).fetchall()
        assert len(pref) == 1
        assert pref[0][0] == "necessary"

        hidden = conn.execute(
            "SELECT DISTINCT label_norm FROM entity_labels "
            "WHERE entity_id = 'e_good' AND label_kind = 'skos:hiddenLabel'"
        ).fetchall()
        assert any(h[0] == "neccessary" for h in hidden)

        # Entity name on entities table must be canonical 'Necessary'
        ent_name = conn.execute("SELECT name FROM entities WHERE entity_id = 'e_good'").fetchone()[0]
        assert ent_name == "Necessary"
    finally:
        conn.close()


def test_apply_synonym_adjudication_related_and_different_from(tmp_path: Path) -> None:
    """Verify apply_synonym_adjudication records skos:related and owl:differentFrom edges."""
    conn = _init_test_db(tmp_path / "yaw.db")
    try:
        upsert_entities(
            conn,
            [
                ("e1", "Consumption price", "Price per seat"),
                ("e2", "consumption", "Usage of resources"),
                ("e3", "Finance", "Financial department"),
                ("e4", "Legal", "Legal department"),
            ],
        )
        candidates = [
            {"entity_a": "e1", "entity_b": "e2", "similarity": 0.82},
            {"entity_a": "e3", "entity_b": "e4", "similarity": 0.75},
        ]
        cand_meta = {
            "e1": ("Consumption price", ""),
            "e2": ("consumption", ""),
            "e3": ("Finance", ""),
            "e4": ("Legal", ""),
        }
        decisions = [
            {
                "pair_id": "e1|e2",
                "source_id": "e1",
                "target_id": "e2",
                "relation": "skos:related",
                "reasoning": "Domain association between price and consumption",
            },
            {
                "pair_id": "e3|e4",
                "source_id": "e3",
                "target_id": "e4",
                "relation": "owl:differentFrom",
                "reasoning": "Distinct organizational functions",
            },
        ]
        rlm_stats: dict[str, int] = {}

        staged_merges, semantic_batch = apply_synonym_adjudication(
            conn,
            candidates,
            cand_meta,
            decisions,
            fact_linked_pairs=set(),
            synonymy_threshold=0.5,
            rlm_stats=rlm_stats,
            corpus_root=tmp_path,
        )

        assert not staged_merges
        assert len(semantic_batch) == 2
        assert rlm_stats.get("related_edges") == 1
        assert rlm_stats.get("different_from_edges") == 1

        rel_edge = conn.execute("SELECT src_id, dst_id FROM edges WHERE kind = 'skos:related'").fetchone()
        assert rel_edge is not None
        assert set(rel_edge) == {"e1", "e2"}

        diff_edge = conn.execute("SELECT src_id, dst_id FROM edges WHERE kind = 'owl:differentFrom'").fetchone()
        assert diff_edge is not None
        assert set(diff_edge) == {"e3", "e4"}
    finally:
        conn.close()
