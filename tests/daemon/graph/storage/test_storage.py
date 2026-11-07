"""Tests for W3C Semantic Cascade Slice 1: Denormalized Pair Evidence & Structural Queries."""

import asyncio
import inspect
import json
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from yaw.core import ChunkPointer
from yaw.daemon.graph.storage import (
    apply_shacl_repairs,
    canonicalize_entity_labels_edc,
    iter_synonym_evidence,
    lint_graph_shacl,
    merge_entities,
    open_yaw_db,
    upsert_chunk_pointers,
    upsert_edges_with_sources,
    upsert_entities,
    upsert_entity_labels,
    upsert_semantic_edges,
)
from yaw.daemon.graph.storage.evidence import (
    EntityMention,
    FactAssertion,
    canonicalize_fact_definitions,
    canonicalize_fact_text,
    evidence_schemas,
    upsert_evidence,
)
from yaw.daemon.graph.storage.graph_ops import upsert_facts
from yaw.daemon.graph.storage.managed_sqlite import create_table
from yaw.daemon.graph.storage.synonyms import (
    _Packet,
    query_broader_tc,
    query_shared_role_facts,
)
from yaw.daemon.graph.storage.synonyms import (
    iter_synonym_evidence as synonyms_iter,
)


def _seed_w3c_test_graph(conn: Any, tmp_path: Path) -> dict[str, str]:
    """Seed test entities, facts, sources, and skos:broader edges for W1, W2, W3 validation."""
    entities = [
        ("ent-tenants", "tenants", "Customer organizations provisioned on the platform."),
        ("ent-pilot-tenants", "pilot tenants", "A group of low-volume tenants used for initial testing."),
        ("ent-invoices", "invoices", "Billing documents issued to customers."),
        ("ent-billing-system", "billing system", "Core payment collection platform."),
        ("ent-root-concept", "organizations", "Top-level enterprise entities."),
        ("ent-unrelated", "widget", "An independent component."),
    ]
    upsert_entities(conn, entities)

    doc_file = tmp_path / "pricing.md"
    doc_file.write_text("Pricing documentation for all tenants and pilot tenants.\n", encoding="utf-8")
    pointer = ChunkPointer("c-pricing", doc_file.name, 1, 1, "Pricing Guide")
    upsert_chunk_pointers(conn, [(pointer, "Pricing documentation for all tenants and pilot tenants.")])
    upsert_edges_with_sources(conn, [("c-pricing", "ent-tenants", "passage", 1.0, "c-pricing")])
    upsert_edges_with_sources(conn, [("c-pricing", "ent-pilot-tenants", "passage", 1.0, "c-pricing")])

    # Rule W1: direct fact connecting ent-tenants and ent-pilot-tenants
    direct_fact = (
        "f-direct-1",
        "ent-pilot-tenants",
        "participate in",
        "ent-tenants",
        json.dumps(["pilot tenants", "participate in", "tenants"]),
        "Pilot tenants are provisioned within standard tenant infrastructure.",
    )

    # Rule W2: Shared role facts
    # Both are subject of same predicate ('billed via') with same object ('ent-invoices')
    role_fact_1 = (
        "f-role-1",
        "ent-pilot-tenants",
        "billed via",
        "ent-invoices",
        json.dumps(["pilot tenants", "billed via", "invoices"]),
        "Pilot tenants receive invoices for consumption.",
    )
    role_fact_2 = (
        "f-role-2",
        "ent-tenants",
        "billed via",
        "ent-invoices",
        json.dumps(["tenants", "billed via", "invoices"]),
        "Tenants receive invoices for consumption.",
    )
    # Both are object of same predicate ('manages') with same subject ('ent-billing-system')
    obj_role_fact_1 = (
        "f-role-3",
        "ent-billing-system",
        "manages",
        "ent-tenants",
        json.dumps(["billing system", "manages", "tenants"]),
        "Billing system manages tenant records.",
    )
    obj_role_fact_2 = (
        "f-role-4",
        "ent-billing-system",
        "manages",
        "ent-pilot-tenants",
        json.dumps(["billing system", "manages", "pilot tenants"]),
        "Billing system manages pilot tenant records.",
    )
    fact_sources = [
        ("f-direct-1", "c-pricing"),
        ("f-role-1", "c-pricing"),
        ("f-role-2", "c-pricing"),
        ("f-role-3", "c-pricing"),
        ("f-role-4", "c-pricing"),
    ]
    upsert_facts(
        conn,
        [direct_fact, role_fact_1, role_fact_2, obj_role_fact_1, obj_role_fact_2],
        fact_sources=fact_sources,
    )

    # Rule W3: Transitive skos:broader hierarchy
    # ent-pilot-tenants skos:broader ent-tenants skos:broader ent-root-concept
    # (src_id is narrower, dst_id is broader)
    upsert_semantic_edges(
        conn,
        [
            ("ent-pilot-tenants", "ent-tenants", "skos:broader", 1.0),
            ("ent-tenants", "ent-root-concept", "skos:broader", 1.0),
        ],
    )
    return {eid: name for eid, name, _ in entities}


def test_synonym_evidence_v2_denormalized_structure(tmp_path: Path) -> None:
    """Verify synonym-evidence-v2 emits denormalized pair structures with zero pointer indirection."""
    with closing(open_yaw_db(tmp_path / "v2_test.db")) as conn:
        _seed_w3c_test_graph(conn, tmp_path)
        candidates = [
            {
                "pair_id": "ent-tenants|ent-pilot-tenants",
                "entity_a": "ent-tenants",
                "entity_b": "ent-pilot-tenants",
                "dense_sim": 0.88,
                "lexical_sim": 0.72,
            }
        ]
        batches = list(iter_synonym_evidence(conn, candidates, tmp_path))
        assert len(batches) == 1
        batch = batches[0]

        parsed = json.loads(batch.payload)
        assert parsed["format"] == "synonym-evidence-v2"
        # Zero top-level entities, facts, or sources lookup tables
        assert "entities" not in parsed
        assert "facts" not in parsed
        assert "sources" not in parsed
        assert "pairs" in parsed
        assert len(parsed["pairs"]) == 1

        pair = parsed["pairs"][0]
        assert pair["pair_id"] == "e0|e1"
        assert pair["dense_sim"] == pytest.approx(0.88)
        assert pair["lexical_sim"] == pytest.approx(0.72)
        assert isinstance(pair["shared_documents"], list)
        assert isinstance(pair["shared_islands"], list)

        # Inline entity_a and entity_b
        for side in ("entity_a", "entity_b"):
            ent = pair[side]
            assert "name" in ent
            assert "definition" in ent
            assert ent["type"] == "skos:Concept"
            assert "labels" in ent
            assert "skos:prefLabel" in ent["labels"]
            assert isinstance(ent["labels"]["skos:prefLabel"], list)
            assert "skos:altLabel" in ent["labels"]
            assert isinstance(ent["labels"]["skos:altLabel"], list)
            assert "facts" in ent
            assert isinstance(ent["facts"], list)
            assert "sources" in ent
            assert isinstance(ent["sources"], list)
            if ent["sources"]:
                src = ent["sources"][0]
                assert "path" in src
                assert "title" in src
                assert "text" in src


def test_synonym_evidence_w3c_structural_queries(tmp_path: Path) -> None:
    """Verify Rule W1 (direct_facts), Rule W2 (shared_role_facts), and Rule W3 (broader_tc)."""
    with closing(open_yaw_db(tmp_path / "w3c_rules.db")) as conn:
        _seed_w3c_test_graph(conn, tmp_path)

        candidates = [
            {
                "pair_id": "ent-tenants|ent-pilot-tenants",
                "entity_a": "ent-tenants",
                "entity_b": "ent-pilot-tenants",
                "dense_sim": 0.88,
                "lexical_sim": 0.72,
            },
            {
                "pair_id": "ent-pilot-tenants|ent-root-concept",
                "entity_a": "ent-pilot-tenants",
                "entity_b": "ent-root-concept",
                "dense_sim": 0.75,
                "lexical_sim": 0.60,
            },
            {
                "pair_id": "ent-tenants|ent-unrelated",
                "entity_a": "ent-tenants",
                "entity_b": "ent-unrelated",
                "dense_sim": 0.50,
                "lexical_sim": 0.40,
            },
        ]
        batches = list(iter_synonym_evidence(conn, candidates, tmp_path, batch_size=10))
        assert len(batches) == 1
        pairs = json.loads(batches[0].payload)["pairs"]
        assert len(pairs) == 3

        p0 = pairs[0]
        # Rule W1: direct_facts connects pilot tenants and tenants directly
        assert len(p0["direct_facts"]) >= 1
        assert ["pilot tenants", "participate in", "tenants"] in p0["direct_facts"]

        # Rule W2: shared_role_facts connects via shared predicate roles
        assert len(p0["shared_role_facts"]) >= 2
        # Either shared subject role ("billed via invoices") or shared object role ("billing system manages")
        triples = [list(f) for f in p0["shared_role_facts"]]
        has_subj_role = ["pilot tenants", "billed via", "invoices"] in triples and [
            "tenants",
            "billed via",
            "invoices",
        ] in triples
        has_obj_role = ["billing system", "manages", "pilot tenants"] in triples and [
            "billing system",
            "manages",
            "tenants",
        ] in triples
        assert has_subj_role or has_obj_role

        # Rule W3: broader_tc check
        # ent-pilot-tenants is narrower than ent-tenants. Since ent-tenants is entity_a, ent-pilot-tenants is entity_b:
        # entity_b is under entity_a -> "b_under_a"
        assert p0["broader_tc"] == "b_under_a"

        # Pair 1: ent-pilot-tenants (entity_a) is transitively narrower than ent-root-concept (entity_b) -> "a_under_b"
        p1 = pairs[1]
        assert p1["broader_tc"] == "a_under_b"

        # Pair 2: ent-tenants and ent-unrelated have no broader relation
        p2 = pairs[2]
        assert p2["broader_tc"] == ""
        assert p2["direct_facts"] == []
        assert p2["shared_role_facts"] == []


def test_synonym_evidence_batch_resolve_decisions_without_canonical(tmp_path: Path) -> None:
    """Verify SynonymEvidenceBatch.resolve_decisions resolves aliases without requiring 'canonical'."""
    with closing(open_yaw_db(tmp_path / "resolve.db")) as conn:
        _seed_w3c_test_graph(conn, tmp_path)
        candidates = [
            {
                "pair_id": "ent-tenants|ent-pilot-tenants",
                "entity_a": "ent-tenants",
                "entity_b": "ent-pilot-tenants",
                "dense_sim": 0.88,
                "lexical_sim": 0.72,
            }
        ]
        batch = next(iter_synonym_evidence(conn, candidates, tmp_path))

        # 1. Decision without 'canonical' should resolve pair_id cleanly
        decisions_no_canon = [{"pair_id": "e0|e1", "relation": "skos:related", "reasoning": "Valid test reasoning"}]
        resolved = batch.resolve_decisions(decisions_no_canon)
        assert len(resolved) == 1
        assert resolved[0]["pair_id"] == "ent-tenants|ent-pilot-tenants"
        assert resolved[0]["relation"] == "skos:related"
        assert "canonical" not in resolved[0]

        # 2. Decision with source_id and target_id (Stage 2 hierarchy format)
        decisions_directed = [
            {
                "pair_id": "e0|e1",
                "relation": "skos:broader",
                "source_id": "e1",
                "target_id": "e0",
                "reasoning": "Subtype hierarchy",
            }
        ]
        resolved_directed = batch.resolve_decisions(decisions_directed)
        assert resolved_directed[0]["source_id"] == "ent-pilot-tenants"
        assert resolved_directed[0]["target_id"] == "ent-tenants"

        # 3. Decision with canonical should resolve alias cleanly
        decisions_with_canon = [{"pair_id": "e0|e1", "relation": "skos:altLabel", "canonical": "e0"}]
        resolved_canon = batch.resolve_decisions(decisions_with_canon)
        assert resolved_canon[0]["canonical"] == "ent-tenants"

        # 4. Unknown pair or invalid canonical should fail hard
        with pytest.raises(ValueError, match="Unknown or duplicate"):
            batch.resolve_decisions([{"pair_id": "e0|e99", "relation": "skos:related"}])
        with pytest.raises(ValueError, match="Invalid canonical"):
            batch.resolve_decisions([{"pair_id": "e0|e1", "relation": "skos:related", "canonical": "foreign"}])


def test_synonym_evidence_batch_subset(tmp_path: Path) -> None:
    """Verify SynonymEvidenceBatch.subset() produces valid filtered packets for cascade stages."""
    with closing(open_yaw_db(tmp_path / "subset.db")) as conn:
        _seed_w3c_test_graph(conn, tmp_path)
        candidates = [
            {
                "pair_id": "ent-tenants|ent-pilot-tenants",
                "entity_a": "ent-tenants",
                "entity_b": "ent-pilot-tenants",
                "dense_sim": 0.88,
                "lexical_sim": 0.72,
            },
            {
                "pair_id": "ent-pilot-tenants|ent-root-concept",
                "entity_a": "ent-pilot-tenants",
                "entity_b": "ent-root-concept",
                "dense_sim": 0.75,
                "lexical_sim": 0.60,
            },
        ]
        batch = next(iter_synonym_evidence(conn, candidates, tmp_path, batch_size=10))
        assert len(batch.candidates) == 2

        # Subset using alias pair ID
        sub_alias = batch.subset(["e0|e1"])
        assert len(sub_alias.candidates) == 1
        assert sub_alias.candidates[0]["pair_id"] == "ent-tenants|ent-pilot-tenants"
        sub_payload = json.loads(sub_alias.payload)
        assert sub_payload["format"] == "synonym-evidence-v2"
        assert len(sub_payload["pairs"]) == 1
        assert sub_payload["pairs"][0]["pair_id"] == "e0|e1"

        # Subset using real pair ID
        sub_real = batch.subset(["ent-pilot-tenants|ent-root-concept"])
        assert len(sub_real.candidates) == 1
        assert sub_real.candidates[0]["pair_id"] == "ent-pilot-tenants|ent-root-concept"
        sub_real_payload = json.loads(sub_real.payload)
        assert len(sub_real_payload["pairs"]) == 1
        assert sub_real_payload["pairs"][0]["pair_id"] == "e1|e2"

        # Empty subset
        sub_empty = batch.subset([])
        assert len(sub_empty.candidates) == 0
        assert json.loads(sub_empty.payload)["pairs"] == []


def test_iter_synonym_evidence_default_batch_size() -> None:
    """Verify default micro-batch size in iter_synonym_evidence is 6."""
    sig = inspect.signature(synonyms_iter)
    assert sig.parameters["batch_size"].default == 6


def test_packet_has_no_obsolete_indirection_methods() -> None:
    """Verify obsolete indirection methods are completely removed from _Packet."""
    for obsolete in ("fact", "source", "facts", "sources", "spans"):
        assert not hasattr(_Packet, obsolete), f"_Packet still has obsolete attribute/method {obsolete!r}"


def test_structural_queries_batched_efficiency(tmp_path: Path) -> None:
    """Verify query_broader_tc and query_shared_role_facts execute efficiently in a single batch."""
    with closing(open_yaw_db(tmp_path / "efficiency.db")) as conn:
        _seed_w3c_test_graph(conn, tmp_path)

        entity_ids = ["ent-tenants", "ent-pilot-tenants", "ent-root-concept"]
        pairs = [
            ("ent-tenants", "ent-pilot-tenants"),
            ("ent-pilot-tenants", "ent-root-concept"),
            ("ent-tenants", "ent-root-concept"),
        ]

        # Both queries execute in single queries without per-pair looping
        broader_set = query_broader_tc(conn, entity_ids)
        assert ("ent-pilot-tenants", "ent-tenants") in broader_set
        assert ("ent-tenants", "ent-root-concept") in broader_set
        assert ("ent-pilot-tenants", "ent-root-concept") in broader_set

        roles = query_shared_role_facts(conn, pairs)
        assert ("ent-pilot-tenants", "ent-tenants") in roles or ("ent-tenants", "ent-pilot-tenants") in roles


def _setup_necessary_typo_graph(conn: Any, tmp_path: Path) -> tuple[str, str]:
    """Setup multi-chunk heading-backed 'Necessary' and 1-chunk typo 'Neccessary'."""
    doc_file = tmp_path / "guide.md"
    doc_file.write_text(
        "# Necessary Procedures\n"
        "Necessary features are essential to operational reliability.\n"
        "Further Necessary controls must be applied at all times.\n",
        encoding="utf-8",
    )
    p1 = ChunkPointer("c-nec-1", doc_file.name, 1, 1, "Necessary Procedures")
    p2 = ChunkPointer("c-nec-2", doc_file.name, 2, 3, "Operational Controls")
    upsert_chunk_pointers(
        conn,
        [
            (p1, "# Necessary Procedures"),
            (
                p2,
                "Necessary features are essential to operational reliability.\n"
                "Further Necessary controls must be applied at all times.",
            ),
        ],
    )

    entities = [
        ("ent-necessary", "Necessary", "Essential and mandatory requirements."),
        ("ent-neccessary", "Neccessary", "Misspelled variant of necessary."),
    ]
    upsert_entities(conn, entities)

    labels = [
        ("necessary", "ent-necessary", "skos:prefLabel", "Necessary", "c-nec-1"),
        ("necessary", "ent-necessary", "skos:prefLabel", "Necessary", "c-nec-2"),
        ("neccessary", "ent-neccessary", "skos:prefLabel", "Neccessary", "c-nec-1"),
    ]
    upsert_entity_labels(conn, labels)

    upsert_edges_with_sources(
        conn,
        [
            ("c-nec-1", "ent-necessary", "passage", 1.0, "c-nec-1"),
            ("c-nec-2", "ent-necessary", "passage", 1.0, "c-nec-2"),
            ("c-nec-1", "ent-neccessary", "passage", 1.0, "c-nec-1"),
        ],
    )

    fact = (
        "f-nec-1",
        "ent-necessary",
        "regulates",
        "ent-necessary",
        json.dumps(["Necessary", "regulates", "operational stability"]),
        "Necessary standards regulate stability.",
    )
    upsert_facts(conn, [fact], fact_sources=[("f-nec-1", "c-nec-1")])

    return "ent-necessary", "ent-neccessary"


def test_canonicalize_entity_labels_edc_typo_merged_into(tmp_path: Path) -> None:
    """Verify 1-chunk typo ('Neccessary') merged into multi-chunk entity is demoted to skos:hiddenLabel."""
    with closing(open_yaw_db(tmp_path / "edc_into.db")) as conn:
        good_id, typo_id = _setup_necessary_typo_graph(conn, tmp_path)
        merged = merge_entities(conn, src_id=typo_id, dst_id=good_id, corpus_root=tmp_path)
        good_id = merged.canonical_id

        # 1. Surviving entity name is 'Necessary'
        row = conn.execute("SELECT name FROM entities WHERE entity_id = ?", (good_id,)).fetchone()
        assert row is not None
        assert row[0] == "Necessary"

        # 2. 'neccessary' norm is demoted to skos:hiddenLabel
        hidden_rows = conn.execute(
            "SELECT label_kind FROM entity_labels WHERE entity_id = ? AND label_norm = 'neccessary'",
            (good_id,),
        ).fetchall()
        assert len(hidden_rows) >= 1
        assert all(r[0] == "skos:hiddenLabel" for r in hidden_rows)

        # 3. 'necessary' norm is skos:prefLabel
        pref_rows = conn.execute(
            "SELECT label_kind FROM entity_labels WHERE entity_id = ? AND label_norm = 'necessary'",
            (good_id,),
        ).fetchall()
        assert len(pref_rows) >= 1
        assert all(r[0] == "skos:prefLabel" for r in pref_rows)

        # 4. Zero label violations from SHACL linter
        violations = lint_graph_shacl(conn)
        label_violations = [
            v for v in violations if "prefLabel" in v.shape or "disjointLabels" in v.shape or "S13" in v.shape
        ]
        assert label_violations == []


def test_canonicalize_entity_labels_edc_typo_merged_onto(tmp_path: Path) -> None:
    """Verify good entity merged onto 1-chunk typo updates entities.name to 'Necessary' and demotes typo."""
    with closing(open_yaw_db(tmp_path / "edc_onto.db")) as conn:
        good_id, typo_id = _setup_necessary_typo_graph(conn, tmp_path)
        # Merged onto: src is good, dst is typo
        merge_entities(conn, src_id=good_id, dst_id=typo_id, corpus_root=tmp_path)

        # 1. Surviving entity (typo_id) now has name 'Necessary'
        row = conn.execute("SELECT name FROM entities WHERE entity_id = ?", (typo_id,)).fetchone()
        assert row is not None
        assert row[0] == "Necessary"

        # 2. 'neccessary' norm is demoted to skos:hiddenLabel
        hidden_rows = conn.execute(
            "SELECT label_kind FROM entity_labels WHERE entity_id = ? AND label_norm = 'neccessary'",
            (typo_id,),
        ).fetchall()
        assert len(hidden_rows) >= 1
        assert all(r[0] == "skos:hiddenLabel" for r in hidden_rows)

        # 3. 'necessary' norm is promoted to skos:prefLabel
        pref_rows = conn.execute(
            "SELECT label_kind FROM entity_labels WHERE entity_id = ? AND label_norm = 'necessary'",
            (typo_id,),
        ).fetchall()
        assert len(pref_rows) >= 1
        assert all(r[0] == "skos:prefLabel" for r in pref_rows)

        # 4. Zero label violations from SHACL linter
        violations = lint_graph_shacl(conn)
        label_violations = [
            v for v in violations if "prefLabel" in v.shape or "disjointLabels" in v.shape or "S13" in v.shape
        ]
        assert label_violations == []


def test_canonicalize_entity_labels_edc_single_label_safety(tmp_path: Path) -> None:
    """Verify an entity with only 1 label is never left with 0 skos:prefLabel rows."""
    with closing(open_yaw_db(tmp_path / "single_label.db")) as conn:
        upsert_entities(conn, [("ent-single", "lonely", "A solitary concept.")])
        upsert_entity_labels(conn, [("lonely", "ent-single", "skos:prefLabel", "lonely", "__auto__")])

        # Explicitly demand that 'lonely' be hidden
        canonicalize_entity_labels_edc(
            conn,
            corpus_root=tmp_path,
            entity_ids=["ent-single"],
            hidden_label_norms_by_entity={"ent-single": {"lonely"}},
        )

        # Entity must still have exactly 1 skos:prefLabel
        pref_rows = conn.execute(
            """
            SELECT label_norm, label_kind
            FROM entity_labels
            WHERE entity_id = 'ent-single' AND label_kind = 'skos:prefLabel'
            """
        ).fetchall()
        assert len(pref_rows) == 1
        assert pref_rows[0][0] == "lonely"

        violations = lint_graph_shacl(conn)
        pref_violations = [v for v in violations if v.shape == "ConceptShape/prefLabel"]
        assert pref_violations == []


def test_canonicalize_entity_labels_edc_explicit_hidden(tmp_path: Path) -> None:
    """Verify hidden_label_norms_by_entity explicitly demotes target norms to skos:hiddenLabel."""
    with closing(open_yaw_db(tmp_path / "explicit_hidden.db")) as conn:
        upsert_entities(conn, [("ent-apple", "Apple", "A widely cultivated fruit.")])
        upsert_entity_labels(
            conn,
            [
                ("apple", "ent-apple", "skos:prefLabel", "Apple", "__source__"),
                ("malus", "ent-apple", "skos:altLabel", "Malus", "__source__"),
            ],
        )

        canonicalize_entity_labels_edc(
            conn,
            entity_ids=["ent-apple"],
            hidden_label_norms_by_entity={"ent-apple": {"malus"}},
        )

        malus_rows = conn.execute(
            "SELECT label_kind FROM entity_labels WHERE entity_id = 'ent-apple' AND label_norm = 'malus'"
        ).fetchall()
        assert len(malus_rows) == 1
        assert malus_rows[0][0] == "skos:hiddenLabel"

        apple_rows = conn.execute(
            "SELECT label_kind FROM entity_labels WHERE entity_id = 'ent-apple' AND label_norm = 'apple'"
        ).fetchall()
        assert len(apple_rows) == 1
        assert apple_rows[0][0] == "skos:prefLabel"

        violations = lint_graph_shacl(conn)
        label_violations = [
            v for v in violations if "prefLabel" in v.shape or "disjointLabels" in v.shape or "S13" in v.shape
        ]
        assert label_violations == []


def test_canonicalize_entity_labels_edc_disk_missing_fails_grounding(tmp_path: Path) -> None:
    """Verify missing/unreadable chunk content on disk increments disk_failed and demotes ungrounded label."""
    with closing(open_yaw_db(tmp_path / "disk_missing.db")) as conn:
        doc_file = tmp_path / "pear.md"
        doc_file.write_text("Fresh organic pears from local orchard.\n", encoding="utf-8")
        p_grounded = ChunkPointer("c-pear", doc_file.name, 1, 1, "Pears")
        p_missing = ChunkPointer("c-missing", "does_not_exist.md", 1, 1, "Missing Doc")
        upsert_chunk_pointers(conn, [(p_grounded, "Fresh organic pears"), (p_missing, "Ghost text")])

        upsert_entities(conn, [("ent-fruit", "Fruit", "Edible produce.")])
        upsert_entity_labels(
            conn,
            [
                ("pear", "ent-fruit", "skos:prefLabel", "Pear", "c-pear"),
                ("ghost", "ent-fruit", "skos:altLabel", "Ghost", "c-missing"),
            ],
        )

        canonicalize_entity_labels_edc(conn, corpus_root=tmp_path, entity_ids=["ent-fruit"])

        # 'ghost' label had missing file, so disk_failed > 0 -> verbatim_in_source = 0 -> skos:hiddenLabel
        ghost_rows = conn.execute(
            "SELECT label_kind FROM entity_labels WHERE entity_id = 'ent-fruit' AND label_norm = 'ghost'"
        ).fetchall()
        assert len(ghost_rows) == 1
        assert ghost_rows[0][0] == "skos:hiddenLabel"

        # 'pear' was grounded on disk -> skos:prefLabel
        pear_rows = conn.execute(
            "SELECT label_kind FROM entity_labels WHERE entity_id = 'ent-fruit' AND label_norm = 'pear'"
        ).fetchall()
        assert len(pear_rows) == 1
        assert pear_rows[0][0] == "skos:prefLabel"


def test_canonicalize_entity_labels_edc_illegal_path_traversal_raises(tmp_path: Path) -> None:
    """Verify illegal path traversal in chunk pointer raises ValueError immediately without being masked."""
    with closing(open_yaw_db(tmp_path / "traversal.db")) as conn:
        p_bad = ChunkPointer("c-bad", "../../outside.txt", 1, 1, "Bad Pointer")
        upsert_chunk_pointers(conn, [(p_bad, "Bad text")])
        upsert_entities(conn, [("ent-bad", "Bad", "Illegal entity.")])
        upsert_entity_labels(conn, [("bad", "ent-bad", "skos:prefLabel", "Bad", "c-bad")])

        with pytest.raises(ValueError, match="Path traversal detected"):
            canonicalize_entity_labels_edc(conn, corpus_root=tmp_path, entity_ids=["ent-bad"])


def test_canonicalize_fact_definitions_validity_precedence_and_repair_preservation(tmp_path: Path) -> None:
    """Verify multi-assertion validity precedence, cross-batch repair preservation, and collision invalidation."""
    with closing(open_yaw_db(tmp_path / "fact_defs.db")) as conn:
        for schema in asyncio.run(evidence_schemas()):
            create_table(conn, schema)
        p1 = ChunkPointer("c1", "doc.md", 1, 1, "Doc")
        p2 = ChunkPointer("c2", "doc.md", 2, 2, "Doc")
        upsert_chunk_pointers(conn, [(p1, "Chunk 1 text."), (p2, "Chunk 2 text.")])
        upsert_entities(
            conn,
            [
                ("ent-subj", "Gateway", "An API gateway component."),
                ("ent-obj", "Backend", "A backend service."),
            ],
        )
        upsert_entity_labels(
            conn,
            [
                ("gateway", "ent-subj", "skos:prefLabel", "Gateway", "c1"),
                ("backend", "ent-obj", "skos:prefLabel", "Backend", "c1"),
            ],
        )

        # 1. Multi-assertion validity precedence:
        # assertion-001 (sorts first) has endpoint-duplicating spo_def; assertion-999 has valid spo_def.
        upsert_facts(
            conn,
            [
                (
                    "fact-multi",
                    "ent-subj",
                    "routes to",
                    "ent-obj",
                    json.dumps(["Gateway", "routes to", "Backend"]),
                    "An API gateway component.",
                ),
                (
                    "fact-healed",
                    "ent-subj",
                    "monitors",
                    "ent-obj",
                    json.dumps(["Gateway", "monitors", "Backend"]),
                    "An API gateway component.",
                ),
            ],
            fact_sources=[("fact-multi", "c1"), ("fact-multi", "c2"), ("fact-healed", "c1")],
        )
        mentions = (
            EntityMention(
                "m-s",
                "c1",
                "rev-1",
                "sense-s",
                "Gateway",
                "An API gateway component.",
                "An API gateway component.",
                "ent-subj",
            ),
            EntityMention(
                "m-o", "c1", "rev-1", "sense-o", "Backend", "A backend service.", "A backend service.", "ent-obj"
            ),
        )
        assertions = (
            FactAssertion(
                "assertion-001",
                "c1",
                "rev-1",
                "m-s",
                "m-o",
                json.dumps({"spo_def": "An API gateway component."}),
                "fact-multi",
            ),
            FactAssertion(
                "assertion-999",
                "c2",
                "rev-1",
                "m-s",
                "m-o",
                json.dumps({"spo_def": "Gateway forwards validated requests to Backend."}),
                "fact-multi",
            ),
            FactAssertion(
                "assertion-100",
                "c1",
                "rev-1",
                "m-s",
                "m-o",
                json.dumps({"spo_def": "An API gateway component."}),
                "fact-healed",
            ),
        )
        upsert_evidence(conn, mentions, assertions)

        updated = canonicalize_fact_definitions(conn)
        assert updated == 1
        row_multi = conn.execute("SELECT definition FROM facts WHERE fact_id = 'fact-multi'").fetchone()
        assert row_multi is not None
        assert row_multi[0] == "Gateway forwards validated requests to Backend."

        # 2. Cross-batch repair preservation:
        # Heal fact-healed via apply_shacl_repairs, then simulate next batch re-asserting fact-healed with invalid def
        healed_def = "Gateway continuously monitors health of Backend."
        apply_shacl_repairs(conn, [{"focus_node": "fact-healed", "action": "set_definition", "value": healed_def}])
        prior_defs = {str(r[0]): str(r[1]) for r in conn.execute("SELECT fact_id, definition FROM facts")}
        # Next batch upsert_facts overwrites facts.definition with unhealed duplicate definition
        upsert_facts(
            conn,
            [
                (
                    "fact-healed",
                    "ent-subj",
                    "monitors",
                    "ent-obj",
                    json.dumps(["Gateway", "monitors", "Backend"]),
                    "An API gateway component.",
                )
            ],
            fact_sources=[("fact-healed", "c2")],
        )
        upsert_evidence(
            conn,
            (),
            (
                FactAssertion(
                    "assertion-050",
                    "c2",
                    "rev-2",
                    "m-s",
                    "m-o",
                    json.dumps({"spo_def": ""}),
                    "fact-healed",
                ),
            ),
        )
        canonicalize_fact_definitions(conn, prior_definitions=prior_defs)
        row_healed = conn.execute("SELECT definition FROM facts WHERE fact_id = 'fact-healed'").fetchone()
        assert row_healed is not None
        assert row_healed[0] == healed_def
        fact_violations = [v for v in lint_graph_shacl(conn) if v.focus_node == "fact-healed"]
        assert fact_violations == []

        # Calling canonicalize_fact_definitions(conn) without prior_definitions also preserves valid current definition
        assert canonicalize_fact_definitions(conn) == 0
        row_healed_again = conn.execute("SELECT definition FROM facts WHERE fact_id = 'fact-healed'").fetchone()
        assert row_healed_again is not None
        assert row_healed_again[0] == healed_def

        # 2b. Re-keyed fact ID not present in prior_definitions falls back to valid current_def:
        conn.execute("UPDATE facts SET definition = ? WHERE fact_id = 'fact-healed'", (healed_def,))
        conn.commit()
        assert canonicalize_fact_definitions(conn, prior_definitions={"other-fact": "Unrelated."}) == 0
        row_rekeyed = conn.execute("SELECT definition FROM facts WHERE fact_id = 'fact-healed'").fetchone()
        assert row_rekeyed is not None
        assert row_rekeyed[0] == healed_def

        # 3. Endpoint definition collision invalidation:
        # Update ent-obj definition to collide with healed_def -> valid_spo_def becomes False -> falls back to assertion
        conn.execute("UPDATE entities SET definition = ? WHERE entity_id = 'ent-obj'", (healed_def,))
        conn.commit()
        changed_on_collision = canonicalize_fact_definitions(conn, prior_definitions=prior_defs)
        assert changed_on_collision == 1
        row_collided = conn.execute("SELECT definition FROM facts WHERE fact_id = 'fact-healed'").fetchone()
        assert row_collided is not None
        assert row_collided[0] == "An API gateway component."
        flagged = [
            v
            for v in lint_graph_shacl(conn)
            if v.focus_node == "fact-healed" and v.shape == "StatementShape/closedWorld"
        ]
        assert len(flagged) == 1


def test_canonicalize_fact_text_delta_only(tmp_path: Path) -> None:
    """Verify canonicalize_fact_text updates only facts whose canonical surface triple changed."""
    with closing(open_yaw_db(tmp_path / "fact_text.db")) as conn:
        p1 = ChunkPointer("c1", "doc.md", 1, 1, "Doc")
        upsert_chunk_pointers(conn, [(p1, "Text.")])
        upsert_entities(
            conn,
            [
                ("ent-s", "Gateway", "An API gateway."),
                ("ent-o", "Backend", "A backend service."),
            ],
        )
        upsert_facts(
            conn,
            [
                (
                    "fact-1",
                    "ent-s",
                    "routes to",
                    "ent-o",
                    json.dumps(["gateway", "routes to", "backend"]),
                    "Gateway routes traffic to Backend.",
                ),
                (
                    "fact-2",
                    "ent-s",
                    "uses port",
                    None,
                    json.dumps(["Gateway", "uses port", "8080"]),
                    "Gateway listens on port 8080.",
                    "literal",
                    "8080",
                    "xsd:integer",
                    "",
                ),
            ],
            fact_sources=[("fact-1", "c1"), ("fact-2", "c1")],
        )

        # First call: fact-1 raw_triple_json changes ("gateway" -> "Gateway") and both need FTS sync
        updated_first = canonicalize_fact_text(conn)
        assert updated_first >= 1
        row_f1 = conn.execute("SELECT raw_triple_json FROM facts WHERE fact_id = 'fact-1'").fetchone()
        assert row_f1 is not None
        assert json.loads(row_f1[0]) == ["Gateway", "routes to", "Backend"]

        # Second call on already-canonical facts: 0 updates and 0 SQLite row modifications
        before_changes = conn.total_changes
        updated_second = canonicalize_fact_text(conn)
        assert updated_second == 0
        assert conn.total_changes == before_changes


def test_shacl_repairs_set_pref_label_healed_provenance_survives_edc(tmp_path: Path) -> None:
    """Verify set_pref_label stores '__healed__', demotes scalar label to hiddenLabel, and survives EDC."""
    with closing(open_yaw_db(tmp_path / "healed_label.db")) as conn:
        doc_file = tmp_path / "status.md"
        doc_file.write_text("The gateway returned 429 during peak traffic.\n", encoding="utf-8")
        p1 = ChunkPointer("c-status", doc_file.name, 1, 1, "Traffic Status")
        upsert_chunk_pointers(conn, [(p1, "The gateway returned 429 during peak traffic.")])

        upsert_entities(
            conn,
            [
                ("ent-status", "429", "HTTP status code indicating too many requests."),
                ("ent-gateway", "Gateway", "Edge routing service."),
            ],
        )
        upsert_entity_labels(
            conn,
            [
                ("429", "ent-status", "skos:prefLabel", "429", "c-status"),
                ("gateway", "ent-gateway", "skos:prefLabel", "Gateway", "c-status"),
            ],
        )
        upsert_facts(
            conn,
            [
                (
                    "fact-status",
                    "ent-gateway",
                    "returned",
                    "ent-status",
                    json.dumps(["Gateway", "returned", "429"]),
                    "The gateway returned HTTP 429 during peak traffic.",
                )
            ],
            fact_sources=[("fact-status", "c-status")],
        )

        initial_violations = [v for v in lint_graph_shacl(conn) if v.focus_node == "ent-status"]
        assert any(v.shape == "ConceptShape/noScalarLiterals" for v in initial_violations)

        # Reject scalar or non-resource repair values
        for bad_val in ("429", "---"):
            with pytest.raises(ValueError):
                apply_shacl_repairs(
                    conn,
                    [{"focus_node": "ent-status", "action": "set_pref_label", "value": bad_val}],
                )

        # Apply valid lexical set_pref_label repair
        applied = apply_shacl_repairs(
            conn,
            [{"focus_node": "ent-status", "action": "set_pref_label", "value": "HTTP 429 Too Many Requests"}],
        )
        assert applied == 1

        # Verify __healed__ provenance and hiddenLabel demotion of '429'
        label_rows = conn.execute(
            "SELECT label_norm, label_kind, raw_label, chunk_id FROM entity_labels WHERE entity_id = 'ent-status'"
        ).fetchall()
        by_norm = {str(r[0]): (str(r[1]), str(r[2]), str(r[3])) for r in label_rows}
        assert by_norm["429"] == ("skos:hiddenLabel", "429", "c-status")
        assert by_norm["http 429 too many requests"] == (
            "skos:prefLabel",
            "HTTP 429 Too Many Requests",
            "__healed__",
        )

        # Subsequent canonicalize_entity_labels_edc with real corpus_root must preserve the healed label
        canonicalize_entity_labels_edc(conn, corpus_root=tmp_path, entity_ids=["ent-status"])
        ent_row = conn.execute("SELECT name FROM entities WHERE entity_id = 'ent-status'").fetchone()
        assert ent_row is not None
        assert ent_row[0] == "HTTP 429 Too Many Requests"
        assert [v for v in lint_graph_shacl(conn) if v.focus_node == "ent-status"] == []


def test_shacl_repairs_maintains_validate_graph_state_across_actions(tmp_path: Path) -> None:
    """Verify delete_invalid_fact, remove_scalar_entity, delete_dangling, and merge_altLabel keep evidence valid."""
    from yaw.daemon.graph.storage.evidence import fact_id, sense_id
    from yaw.daemon.graph.storage.rows import validate_graph_state

    with closing(open_yaw_db(tmp_path / "shacl_evidence_state.db")) as conn:
        for schema in asyncio.run(evidence_schemas()):
            create_table(conn, schema)
        doc_file = tmp_path / "doc.md"
        doc_file.write_text("Gateway, Gateway Alias, Backend, 429, and Orphan.\n", encoding="utf-8")
        p1 = ChunkPointer("c1", doc_file.name, 1, 1, "Doc")
        upsert_chunk_pointers(conn, [(p1, "Gateway, Gateway Alias, Backend, 429, and Orphan.")])

        e_subj = sense_id("Gateway", "Edge API gateway.")
        e_dup = sense_id("Gateway Alias", "Secondary edge gateway label.")
        e_obj = sense_id("Backend", "Core backend service.")
        e_scalar = sense_id("429", "Numeric status literal.")
        e_orphan = sense_id("Orphan", "Isolated concept node.")

        upsert_entities(
            conn,
            [
                (e_subj, "Gateway", "Edge API gateway."),
                (e_dup, "Gateway Alias", "Secondary edge gateway label."),
                (e_obj, "Backend", "Core backend service."),
                (e_scalar, "429", "Numeric status literal."),
                (e_orphan, "Orphan", "Isolated concept node."),
            ],
        )
        upsert_entity_labels(
            conn,
            [
                ("gateway", e_subj, "skos:prefLabel", "Gateway", "c1"),
                ("gateway alias", e_dup, "skos:prefLabel", "Gateway Alias", "c1"),
                ("backend", e_obj, "skos:prefLabel", "Backend", "c1"),
                ("429", e_scalar, "skos:prefLabel", "429", "c1"),
                ("orphan", e_orphan, "skos:prefLabel", "Orphan", "c1"),
            ],
        )
        upsert_edges_with_sources(
            conn,
            [
                ("c1", e_subj, "passage", 1.0, "c1"),
                ("c1", e_dup, "passage", 1.0, "c1"),
                ("c1", e_obj, "passage", 1.0, "c1"),
                ("c1", e_scalar, "passage", 1.0, "c1"),
                ("c1", e_orphan, "passage", 1.0, "c1"),
            ],
        )

        f_keep = fact_id(e_subj, "routes to", "entity", e_obj)
        f_dup = fact_id(e_dup, "routes to", "entity", e_obj)
        f_invalid = fact_id(e_subj, " drops ", "entity", e_obj)
        f_scalar_obj = fact_id(e_subj, "returned", "entity", e_scalar)
        f_scalar_subj = fact_id(e_scalar, "indicates", "entity", e_obj)
        f_orphan = fact_id(e_orphan, "monitors", "entity", e_obj)

        upsert_facts(
            conn,
            [
                (
                    f_keep,
                    e_subj,
                    "routes to",
                    e_obj,
                    json.dumps(["Gateway", "routes to", "Backend"]),
                    "Gateway routes requests to Backend.",
                ),
                (
                    f_dup,
                    e_dup,
                    "routes to",
                    e_obj,
                    json.dumps(["Gateway Alias", "routes to", "Backend"]),
                    "Gateway Alias routes requests to Backend.",
                ),
                (
                    f_invalid,
                    e_subj,
                    " drops ",
                    e_obj,
                    json.dumps(["Gateway", " drops ", "Backend"]),
                    "Invalid fact to be deleted.",
                ),
                (
                    f_scalar_obj,
                    e_subj,
                    "returned",
                    e_scalar,
                    json.dumps(["Gateway", "returned", "429"]),
                    "Gateway returned status code 429.",
                ),
                (
                    f_scalar_subj,
                    e_scalar,
                    "indicates",
                    e_obj,
                    json.dumps(["429", "indicates", "Backend"]),
                    "Status 429 indicates Backend overload.",
                ),
                (
                    f_orphan,
                    e_orphan,
                    "monitors",
                    e_obj,
                    json.dumps(["Orphan", "monitors", "Backend"]),
                    "Orphan monitors Backend.",
                ),
            ],
            fact_sources=[
                (f_keep, "c1"),
                (f_dup, "c1"),
                (f_invalid, "c1"),
                (f_scalar_obj, "c1"),
                (f_scalar_subj, "c1"),
                (f_orphan, "c1"),
            ],
        )

        mentions = (
            EntityMention("m-subj", "c1", "rev-1", e_subj, "Gateway", "Edge API gateway.", "Edge API gateway.", e_subj),
            EntityMention(
                "m-dup",
                "c1",
                "rev-1",
                e_dup,
                "Gateway Alias",
                "Secondary edge gateway label.",
                "Secondary edge gateway label.",
                e_dup,
            ),
            EntityMention(
                "m-obj", "c1", "rev-1", e_obj, "Backend", "Core backend service.", "Core backend service.", e_obj
            ),
            EntityMention(
                "m-scalar",
                "c1",
                "rev-1",
                e_scalar,
                "429",
                "Numeric status literal.",
                "Numeric status literal.",
                e_scalar,
            ),
            EntityMention(
                "m-orphan",
                "c1",
                "rev-1",
                e_orphan,
                "Orphan",
                "Isolated concept node.",
                "Isolated concept node.",
                e_orphan,
            ),
        )
        assertions = (
            FactAssertion(
                "a-keep",
                "c1",
                "rev-1",
                "m-subj",
                "m-obj",
                json.dumps({"spo_def": "Gateway routes requests to Backend."}),
                f_keep,
            ),
            FactAssertion(
                "a-dup",
                "c1",
                "rev-1",
                "m-dup",
                "m-obj",
                json.dumps({"spo_def": "Gateway Alias routes requests to Backend."}),
                f_dup,
            ),
            FactAssertion(
                "a-invalid",
                "c1",
                "rev-1",
                "m-subj",
                "m-obj",
                json.dumps({"spo_def": "Invalid fact to be deleted."}),
                f_invalid,
            ),
            FactAssertion(
                "a-scalar-obj",
                "c1",
                "rev-1",
                "m-subj",
                "m-scalar",
                json.dumps({"spo_def": "Gateway returned status code 429."}),
                f_scalar_obj,
            ),
            FactAssertion(
                "a-scalar-subj",
                "c1",
                "rev-1",
                "m-scalar",
                "m-obj",
                json.dumps({"spo_def": "Status 429 indicates Backend overload."}),
                f_scalar_subj,
            ),
            FactAssertion(
                "a-orphan",
                "c1",
                "rev-1",
                "m-orphan",
                "m-obj",
                json.dumps({"spo_def": "Orphan monitors Backend."}),
                f_orphan,
            ),
        )
        upsert_evidence(conn, mentions, assertions)

        # Execute all 4 repair actions:
        # 1) delete_invalid_fact on f_invalid
        # 2) remove_scalar_entity on e_scalar (literalizes f_scalar_obj; deletes f_scalar_subj & m-scalar)
        # 3) delete_dangling on e_orphan (deletes f_orphan, a-orphan, and m-orphan)
        # 4) merge_altLabel on e_dup|e_subj (merges entities and binds merged evidence)
        repairs = [
            {"focus_node": f_invalid, "action": "delete_invalid_fact", "value": ""},
            {"focus_node": e_scalar, "action": "remove_scalar_entity", "value": ""},
            {"focus_node": e_orphan, "action": "delete_dangling", "value": ""},
            {"focus_node": f"{e_dup}|{e_subj}", "action": "merge_altLabel", "value": e_subj},
        ]
        applied = apply_shacl_repairs(conn, repairs)
        assert applied == 4

        # Verify a-invalid has fact_id = NULL
        row_inv = conn.execute("SELECT fact_id FROM fact_assertions WHERE assertion_id = 'a-invalid'").fetchone()
        assert row_inv is not None and row_inv[0] is None

        # Verify a-scalar-obj was literalized (object_mention_id = NULL) and re-keyed to canonical literal fact_id
        canonical_winner = min(e_subj, e_dup)
        expected_lit_fid = fact_id(canonical_winner, "returned", "literal", "429", "xsd:integer", "")
        row_sc_obj = conn.execute(
            "SELECT object_mention_id, fact_id FROM fact_assertions WHERE assertion_id = 'a-scalar-obj'"
        ).fetchone()
        assert row_sc_obj is not None
        assert row_sc_obj[0] is None
        assert row_sc_obj[1] == expected_lit_fid

        # Verify scalar and orphan entity mentions and subject assertions were removed
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM entity_mentions WHERE entity_id IN (?, ?)", (e_scalar, e_orphan)
            ).fetchone()[0]
            == 0
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM fact_assertions WHERE assertion_id IN ('a-scalar-subj', 'a-orphan')"
            ).fetchone()[0]
            == 0
        )

        # Verify full graph state and SHACL conformance pass without any dangling/noncanonical bindings
        validate_graph_state(conn)
