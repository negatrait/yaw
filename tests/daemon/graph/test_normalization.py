"""Record normalization retains literal evidence and distinct entity senses."""

from contextlib import closing
from dataclasses import asdict
from pathlib import Path

import numpy as np

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


def test_provisional_senses_auto_merge_on_high_cosine_and_difflib_while_homographs_go_to_candidates(tmp_path: Path):
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
        candidates, _meta, _raw_cnt, auto_merges = prepare_synonym_candidates(
            conn,
            batch.entity_nodes,
            synonymy_threshold=0.8,
            synonymy_top_k=5,
            auto_merge_cosine_gate=0.90,
            auto_merge_difflib_gate=0.92,
            fact_linked_pairs=batch.fact_linked_pairs,
        )
        assert len(auto_merges) == 1
        remaining_defs = {
            str(r[0])
            for r in conn.execute("SELECT definition FROM entities WHERE name = 'Hedge-duplicate-rate'").fetchall()
        }
        assert remaining_defs == {def1, def3}
        homograph_cands = [
            c for c in candidates if c["name_a"] == "Hedge-duplicate-rate" and c["name_b"] == "Hedge-duplicate-rate"
        ]
        assert len(homograph_cands) == 1
        assert {homograph_cands[0]["def_a"], homograph_cands[0]["def_b"]} == {def1, def3}
