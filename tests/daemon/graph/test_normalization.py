"""Record normalization retains literal evidence and distinct entity senses."""

from dataclasses import asdict

import numpy as np

from yaw.core import ChunkPointer, OpenIERecord, StructuredTriple
from yaw.daemon.graph.normalization import normalize_openie_record
from yaw.daemon.graph.planner import materialize_entities_and_facts, prepare_sense_resolution_state


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
    triple = normalized.triples[0]
    assert triple.subj == triple.obj
    assert triple.subj_def == "A planet"
    assert triple.obj_def == "A chemical element"
    assert normalized.surfaces[triple.subj] == "Mercury"


def test_same_document_merged_definition_is_prebatched_for_subsequent_sense_resolution():
    def1 = (
        "A metric used to confirm streaming degradation, specifically indicating "
        "the rate of duplicate data in the hedge process."
    )
    def2 = "A metric for which no existing alert was present, contributing to the incident."
    def3 = "An unrelated financial hedging duplicate ratio in portfolio accounting."
    expected_merged = f"{def1.rstrip('.')} — {def2}"

    chunks = [
        (ChunkPointer("c1", "incidents/postmortem.md", 1, 10, "Postmortem", 1, 20), "chunk 1"),
        (ChunkPointer("c2", "incidents/postmortem.md", 11, 20, "Postmortem", 1, 20), "chunk 2"),
        (ChunkPointer("c3", "finance/portfolio.md", 1, 10, "Portfolio", 1, 10), "chunk 3"),
    ]
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
    chunk_surface_by_norm = {
        ("c1", "hedge-duplicate-rate"): "Hedge-duplicate-rate",
        ("c2", "hedge-duplicate-rate"): "Hedge-duplicate-rate",
        ("c3", "hedge-duplicate-rate"): "Hedge-duplicate-rate",
    }

    def fmt_entity(name: str, defn: str) -> str:
        return f"# {name}\n{defn}"

    sense_state = prepare_sense_resolution_state(
        conn=None,
        chunks=chunks,
        all_norm_labels=set(surface_by_norm),
        record_structured=record_structured,
        surface_by_norm=surface_by_norm,
        fmt_entity=fmt_entity,
    )
    merged_text = fmt_entity("Hedge-duplicate-rate", expected_merged)
    assert merged_text in sense_state.prebatch_texts

    prebatch_vec_by_text = {
        text: np.array([1.0, 0.0], dtype=np.float32) if def3 not in text else np.array([0.0, 1.0], dtype=np.float32)
        for text in sense_state.prebatch_texts
    }
    batch = materialize_entities_and_facts(
        record_structured=record_structured,
        surface_by_norm=surface_by_norm,
        chunk_surface_by_norm=chunk_surface_by_norm,
        sense_state=sense_state,
        prebatch_vec_by_text=prebatch_vec_by_text,
        fmt_entity=fmt_entity,
        same_sense_cosine_gate=0.85,
        embed_one=None,
    )
    defs = {defn for _, name, defn in batch.entity_nodes if name == "Hedge-duplicate-rate"}
    assert expected_merged in defs
    assert def3 in defs
    assert batch.homograph_splits == 1
