"""Tests for W3C Semantic Cascade Slice 1: Denormalized Pair Evidence & Structural Queries."""

import inspect
import json
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from yaw.core import ChunkPointer
from yaw.daemon.graph.storage import (
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
from yaw.daemon.graph.storage.graph_ops import upsert_facts
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
