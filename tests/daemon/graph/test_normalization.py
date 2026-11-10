"""Record normalization retains literal evidence and distinct entity senses."""

from contextlib import closing
from dataclasses import asdict
from pathlib import Path

import numpy as np

from daemon.graph.candidate_fixtures import seed_candidate_vectors
from yaw.core import OpenIERecord, StructuredTriple, text_processing
from yaw.daemon.graph.normalization import normalize_openie_record
from yaw.daemon.graph.planner import (
    materialize_entities_and_facts,
    prepare_synonym_candidates,
)
from yaw.daemon.graph.storage import (
    open_yaw_db,
    upsert_entities,
    upsert_entity_labels,
    upsert_entity_vectors,
)


def test_literal_lexical_value_datatype_and_input_are_preserved():
    record = OpenIERecord(
        "c",
        ("Release",),
        (),
        structured_triples=(
            StructuredTriple(
                subj="Release",
                subj_def="A software release",
                pred="released on",
                obj="2024-01-01",
                obj_kind="literal",
                obj_datatype="xsd:date",
                spo_def="Release date is 2024-01-01.",
            ),
        ),
        entity_definitions=(("Release", "A software release"),),
    )
    original = asdict(record)
    normalized = normalize_openie_record(record)
    triple = normalized.triples[0]
    assert triple.obj == "2024-01-01"
    assert triple.obj_kind == "literal"
    assert triple.obj_datatype == "xsd:date"
    assert normalized.surfaces[triple.subj] == "Release"
    assert asdict(record) == original
    assert normalize_openie_record(record) == normalized


def test_homograph_definitions_remain_distinct_in_triples_and_mentions():
    record = OpenIERecord(
        "c",
        ("Mercury",),
        (("Mercury", "differs from", "Mercury"),),
        entity_definitions=(("Mercury", "A planet"), ("Mercury", "A chemical element")),
    )
    normalized = normalize_openie_record(record)
    assert {definition for _, definition in normalized.entities} == {"A planet", "A chemical element"}
    assert normalized.triples == []
    assert normalized.surfaces["mercury"] == "Mercury"


def test_provisional_senses_require_definition_agreement_despite_high_cosine(tmp_path: Path):
    def1 = (
        "A metric used to confirm streaming degradation, specifically indicating "
        "the rate of duplicate data in the hedge process."
    )
    def2 = "A metric for which no existing alert was present, contributing to the incident."
    def3 = "An unrelated financial hedging duplicate ratio in portfolio accounting."

    record_structured = [
        (
            "c1",
            [
                StructuredTriple(
                    subj="hedge-duplicate-rate",
                    subj_def=def1,
                    pred="confirmed",
                    obj="streaming degradation",
                    obj_def="Degradation in streaming pipeline.",
                    spo_def="Hedge-duplicate-rate confirmed streaming degradation.",
                )
            ],
            [("hedge-duplicate-rate", def1), ("streaming degradation", "Degradation in streaming pipeline.")],
        ),
        (
            "c2",
            [
                StructuredTriple(
                    subj="incident",
                    subj_def="Production streaming incident.",
                    pred="involved",
                    obj="hedge-duplicate-rate",
                    obj_def=def2,
                    spo_def="The incident involved hedge-duplicate-rate.",
                )
            ],
            [("incident", "Production streaming incident."), ("hedge-duplicate-rate", def2)],
        ),
        (
            "c3",
            [
                StructuredTriple(
                    subj="hedge-duplicate-rate",
                    subj_def=def3,
                    pred="measured",
                    obj="portfolio",
                    obj_def="Financial portfolio.",
                    spo_def="Hedge-duplicate-rate measured portfolio.",
                )
            ],
            [("hedge-duplicate-rate", def3), ("portfolio", "Financial portfolio.")],
        ),
    ]
    surface_by_norm = {
        "hedge-duplicate-rate": "Hedge-duplicate-rate",
        "streaming degradation": "Streaming degradation",
        "incident": "Incident",
        "portfolio": "Portfolio",
    }
    surface_by_norm = {str(text_processing(label)): surface for label, surface in surface_by_norm.items()}
    chunk_surface_by_norm = {
        (cid, str(text_processing(label))): surface_by_norm[str(text_processing(label))]
        for cid, _, entities in record_structured
        for label, _ in entities
    }

    batch = materialize_entities_and_facts(
        record_structured=record_structured,
        surface_by_norm=surface_by_norm,
        chunk_surface_by_norm=chunk_surface_by_norm,
        existing_bindings={},
    )
    defs = {defn for _, name, defn in batch.entity_nodes if name == "Hedge-duplicate-rate"}
    assert defs == {def1, def2, def3}
    assert batch.homograph_splits == 2

    with closing(open_yaw_db(tmp_path / "stage.db")) as conn:
        upsert_entities(conn, batch.entity_nodes)
        upsert_entity_labels(conn, batch.entity_labels_to_upsert)
        vec_pairs = [
            (
                eid,
                np.array([1.0, 0.0], dtype=np.float32) if defn != def3 else np.array([0.0, 1.0], dtype=np.float32),
            )
            for eid, _, defn in batch.entity_nodes
        ]
        upsert_entity_vectors(conn, vec_pairs)
        seed_candidate_vectors(conn)
        candidates, _meta, _raw_cnt, auto_merges = prepare_synonym_candidates(
            conn,
            batch.entity_nodes,
            candidate_dense_threshold=0.8,
            candidate_lexical_threshold=0.55,
            fact_linked_pairs=batch.fact_linked_pairs,
        )
        assert auto_merges == []
        remaining_defs = {
            str(r[0])
            for r in conn.execute("SELECT definition FROM entities WHERE name = 'Hedge-duplicate-rate'").fetchall()
        }
        assert remaining_defs == {def1, def2, def3}
        homograph_cands = [
            c for c in candidates if c["name_a"] == "Hedge-duplicate-rate" and c["name_b"] == "Hedge-duplicate-rate"
        ]
        assert len(homograph_cands) == 3
        assert {definition for pair in homograph_cands for definition in (pair["def_a"], pair["def_b"])} == {
            def1,
            def2,
            def3,
        }


def test_clean_surface_label_and_single_surface_pref_label_alignment(tmp_path: Path):
    from yaw.core import clean_surface_label
    from yaw.daemon.graph.normalization import preferred_surface
    from yaw.daemon.graph.storage import canonicalize_entity_labels_edc

    assert clean_surface_label(":feature flag") == "feature flag"
    assert clean_surface_label("Control plane,") == "Control plane"
    assert clean_surface_label(" ; Latency: ") == "Latency"
    assert preferred_surface("Control plane,", "control plane", "Control plane", label_norm="control plane") == (
        "Control plane"
    )
    assert preferred_surface(":feature flag", "feature flag", label_norm="feature flag") == "feature flag"

    with closing(open_yaw_db(tmp_path / "single_pref.db")) as conn:
        upsert_entities(conn, [("ent-cp", "control plane", "Central orchestration layer.")])
        for cid, raw in [
            ("c1", "control plane"),
            ("c2", "Control-plane"),
            ("c3", "Control plane,"),
            ("c4", "Control plane"),
            ("c5", ":feature flag"),
        ]:
            norm = "feature flag" if "feature" in raw else "control plane"
            kind = "skos:altLabel" if "feature" in raw else "skos:prefLabel"
            conn.execute(
                "INSERT INTO entity_labels (label_norm, entity_id, label_kind, raw_label, chunk_id) VALUES (?,?,?,?,?)",
                (norm, "ent-cp", kind, raw, cid),
            )
        canonicalize_entity_labels_edc(conn)
        pref_raws = [
            str(r[0])
            for r in conn.execute(
                "SELECT DISTINCT raw_label FROM entity_labels "
                "WHERE entity_id = 'ent-cp' AND label_kind = 'skos:prefLabel'"
            ).fetchall()
        ]
        assert pref_raws == ["Control plane"]
        alt_raws = [
            str(r[0])
            for r in conn.execute(
                "SELECT DISTINCT raw_label FROM entity_labels "
                "WHERE entity_id = 'ent-cp' AND label_kind = 'skos:altLabel'"
            ).fetchall()
        ]
        assert alt_raws == ["feature flag"]


def test_scalar_literal_object_evidence_materialization_and_binding():
    from yaw.core import ChunkPointer
    from yaw.daemon.graph.storage.evidence import build_evidence

    raw_record = OpenIERecord(
        chunk_id="chunk-lat-1",
        entities=("API Gateway", "<= 200ms"),
        triples=(("API Gateway", "has latency target", "<= 200ms"),),
        structured_triples=(
            StructuredTriple(
                subj="API Gateway",
                subj_def="Edge entry point for incoming client traffic.",
                pred="has latency target",
                obj="<= 200ms",
                obj_kind="entity",  # Raw OpenIE extracted as entity
                spo_def="API Gateway latency target is <= 200ms.",
            ),
        ),
        entity_definitions=(
            ("API Gateway", "Edge entry point for incoming client traffic."),
            ("<= 200ms", "Maximum acceptable response time."),
        ),
    )
    normalized = normalize_openie_record(raw_record)
    assert len(normalized.triples) == 1
    st = normalized.triples[0]
    assert st.obj == "<= 200ms"
    assert st.obj_kind == "literal"
    assert st.obj_datatype == "xsd:string"

    surface_by_norm = {str(text_processing("API Gateway")): "API Gateway"}
    chunk_surface_by_norm = {("chunk-lat-1", str(text_processing("API Gateway"))): "API Gateway"}
    record_structured = [("chunk-lat-1", normalized.triples, normalized.entities)]

    batch = materialize_entities_and_facts(
        record_structured=record_structured,
        surface_by_norm=surface_by_norm,
        chunk_surface_by_norm=chunk_surface_by_norm,
        existing_bindings={},
    )
    # Scalar literal object should NOT create an entity node
    assert len(batch.entity_nodes) == 1
    assert batch.entity_nodes[0][1] == "API Gateway"
    assert len(batch.fact_nodes) == 1
    fact_node = batch.fact_nodes[0]
    assert fact_node[8] == "literal"  # obj_kind
    assert fact_node[9] == "<= 200ms"  # obj_val
    assert fact_node[10] == "xsd:string"  # obj_datatype

    # build_evidence must succeed without ValueError on raw_record containing '<= 200ms'
    pointer = ChunkPointer("chunk-lat-1", "doc.md", 1, 10)
    mentions, assertions = build_evidence(
        pointer=pointer,
        text="API Gateway latency target is <= 200ms.",
        program_identity="test-program",
        record=raw_record,
        bindings=batch.bindings,
    )
    assert len(mentions) == 1
    assert mentions[0].name == "API Gateway"
    assert len(assertions) == 1
    assert assertions[0].object_mention_id is None
    assert assertions[0].fact_id == fact_node[0]
