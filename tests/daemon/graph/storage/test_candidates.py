"""Sense-preserving pair-local discovery and the independent automatic merge gate."""

from contextlib import closing

import pytest
from daemon.graph.candidate_fixtures import seed_candidate_vectors

from yaw.core import YawRuntimeError, load_config
from yaw.daemon.graph.planner import prepare_synonym_candidates
from yaw.daemon.graph.storage import (
    find_entity_candidates,
    merge_entities,
    open_yaw_db,
    upsert_entities,
    upsert_entity_labels,
    upsert_entity_vectors,
)


@pytest.mark.parametrize(
    ("a", "b", "cosine", "merged"),
    [
        (("Canary", "A deployment test."), ("canary", "A deployment test"), 1.0, True),
        (("Canary", "A deployment test."), ("canary", "A deployment test"), 0.97, False),
        (("ball", "A spherical toy."), ("ball", "A formal dance event."), 1.0, False),
        (("Cindy Real", "A factual person."), ("Cinderella", "A fictional person."), 1.0, False),
        (("system", "Allows public access."), ("system", "Disallows public access."), 1.0, False),
    ],
)
def test_automatic_merge_requires_extreme_cosine_label_and_definition(tmp_path, a, b, cosine, merged):
    import math

    with closing(open_yaw_db(tmp_path / "merge.db")) as conn:
        entities = [("a", *a), ("b", *b)]
        upsert_entities(conn, entities)
        upsert_entity_vectors(conn, [("a", [1.0, 0.0]), ("b", [cosine, math.sqrt(1 - cosine * cosine)])])
        seed_candidate_vectors(conn)
        candidates, _, _, merges = prepare_synonym_candidates(
            conn,
            entities,
            candidate_dense_threshold=0.5,
            candidate_lexical_threshold=0.4,
            auto_merge_threshold=0.98,
        )
        assert bool(merges) == merged
        assert bool(candidates) != merged
        assert conn.execute("SELECT count(*) FROM entities").fetchone()[0] == (1 if merged else 2)


def test_candidates_are_pair_local_across_blocks_and_keep_orthogonal_homographs(tmp_path):
    with closing(open_yaw_db(tmp_path / "pairs.db")) as conn:
        entities = [(f"e{i:02}", f"Resource {i}", f"Distinct definition {i}") for i in range(35)]
        entities[0] = ("e00", "ball", "A formal dance event.")
        entities[-1] = ("e34", "ball", "A spherical toy.")
        upsert_entities(conn, entities)
        upsert_entity_vectors(conn, [("e00", [1.0, 0.0]), ("e34", [0.0, 1.0])])
        seed_candidate_vectors(conn)
        full = find_entity_candidates(conn, dense_threshold=0.9, lexical_threshold=0.8)
        incremental = {}
        for batch in ([e[0] for e in entities[:16]], [e[0] for e in entities[16:]]):
            for a, b, dense, lexical in find_entity_candidates(conn, batch, dense_threshold=0.9, lexical_threshold=0.8):
                incremental[a, b] = (dense, lexical)
        assert {(a, b): (dense, lexical) for a, b, dense, lexical in full} == incremental
        assert ("e00", "e34", 0.0, 1.0) in full
        assert find_entity_candidates(conn, []) == []


def test_fuzzy_source_label_candidates_survive_canonical_name_changes(tmp_path):
    with closing(open_yaw_db(tmp_path / "labels.db")) as conn:
        upsert_entities(conn, [("a", "Northern coast institute"), ("b", "NCI"), ("c", "Northern coast institution")])
        upsert_entity_vectors(conn, [("a", [1.0, 0.0]), ("b", [1.0, 0.0]), ("c", [0.0, 1.0])])
        upsert_entity_labels(
            conn,
            [
                ("northern coast institute", "a", "skos:prefLabel", "Northern coast institute", "c1"),
                ("nci", "b", "skos:prefLabel", "NCI", "c1"),
            ],
        )
        seed_candidate_vectors(conn)
        assert any(a == "a" and b == "c" for a, b, _, _ in find_entity_candidates(conn, ["c"]))
        merge_entities(conn, "a", "b", label_kind="skos:altLabel")
        assert conn.execute("SELECT name FROM entities WHERE entity_id='a'").fetchone()[0] == "NCI"
        assert any(a == "a" and b == "c" for a, b, _, _ in find_entity_candidates(conn, ["c"]))


def test_display_label_changes_cannot_create_source_candidates(tmp_path):
    with closing(open_yaw_db(tmp_path / "display.db")) as conn:
        upsert_entities(conn, [("a", "Northern coast institute"), ("b", "QZX")])
        upsert_entity_vectors(conn, [("a", [1.0, 0.0]), ("b", [0.0, 1.0])])
        seed_candidate_vectors(conn)
        assert find_entity_candidates(conn) == []
        upsert_entities(conn, [("b", "Northern coast institution")])
        assert find_entity_candidates(conn) == []


def test_automatic_merge_keeps_difflib_orientation_after_representative_remapping(tmp_path):
    x, y = "x" * 98 + "tide", "x" * 98 + "diet"
    entities = [
        ("a", "X" * 98 + "tide", "A deployment test."),
        ("b", y, "A deployment test."),
        ("c", x, "A deployment test"),
    ]
    remaining = []
    for incremental in (False, True):
        with closing(open_yaw_db(tmp_path / f"orientation-{incremental}.db")) as conn:
            for batch in [entities[::2], entities[1:2]] if incremental else [entities]:
                upsert_entities(conn, batch)
                upsert_entity_vectors(conn, [(eid, [1.0, 0.0]) for eid, _, _ in batch])
                seed_candidate_vectors(conn)
                prepare_synonym_candidates(
                    conn, batch, candidate_dense_threshold=0.8, candidate_lexical_threshold=0.625
                )
            remaining.append([row[0] for row in conn.execute("SELECT entity_id FROM entities ORDER BY entity_id")])
    assert remaining == [["a", "b"], ["a", "b"]]


@pytest.mark.parametrize("kind", ["owl:differentFrom", "skos:broader", "skos:related", "fact"])
def test_extreme_merge_cannot_cross_established_distinctness_transitively(tmp_path, kind):
    with closing(open_yaw_db(tmp_path / "distinct.db")) as conn:
        entities = [
            (eid, "A sufficiently long shared surface name for a transitive test " + eid, "A shared definition.")
            for eid in "abc"
        ]
        upsert_entities(conn, entities)
        upsert_entity_vectors(conn, [(eid, [1.0, 0.0]) for eid in "abc"])
        conn.execute("INSERT INTO edges VALUES (?,?,?,?,?)", ("a", "c", kind, 1.0, "Established separation"))
        seed_candidate_vectors(conn)
        prepare_synonym_candidates(conn, entities, candidate_dense_threshold=0.8, candidate_lexical_threshold=0.55)
        assert conn.execute("SELECT count(*) FROM entities").fetchone()[0] == 2
        assert conn.execute("SELECT count(*) FROM edges WHERE kind=?", (kind,)).fetchone()[0] > 0


def test_calibration_uses_configured_volume_with_explicit_override(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path

    db = tmp_path / "calibration.db"
    with closing(open_yaw_db(db)) as conn:
        upsert_entities(conn, [("a", "Alpha"), ("b", "Alpha Service"), ("c", "QZX")])
        upsert_entity_vectors(conn, [("a", [1.0, 0.0]), ("b", [0.9, 0.43589]), ("c", [-1.0, 0.0])])
        seed_candidate_vectors(conn)
    config = tmp_path / "yaw.toml"
    config.write_text("[graph]\ncandidate_volume_ratio = 2.0\n", encoding="utf-8")
    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps({"cases": [{"entity_ids": ["a", "b", "c"], "candidate_count": 1}]}), encoding="utf-8"
    )
    script = Path(__file__).resolve().parents[4] / "scripts" / "calibrate_candidates.py"
    cmd = [sys.executable, str(script), "--db", str(db), "--reference", str(reference), "--config", str(config)]
    configured = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert configured.returncode != 0
    assert "No fixed cutoff pair" in configured.stderr
    overridden = subprocess.run([*cmd, "--volume-ratio", "1.0"], capture_output=True, text=True, timeout=30, check=True)
    result = json.loads(overridden.stdout)
    assert result["candidate_volume_ratio"] == 1.0
    assert result["candidate_counts"] == [1]


@pytest.mark.parametrize("key", ["synonymy_threshold", "synonymy_top_k"])
def test_retired_candidate_settings_fail_instead_of_silently_using_defaults(tmp_path, key):
    path = tmp_path / "yaw.toml"
    path.write_text(f"[graph]\n{key} = 1\n", encoding="utf-8")
    with pytest.raises(YawRuntimeError, match="Retired candidate setting"):
        load_config(path)


@pytest.mark.parametrize("key", ["candidate_dense_threshold", "candidate_lexical_threshold"])
@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_nonfinite_admission_thresholds_fail(tmp_path, key, value):
    path = tmp_path / "yaw.toml"
    path.write_text(f"[graph]\n{key} = {value}\n", encoding="utf-8")
    with pytest.raises(YawRuntimeError, match="must be finite"):
        load_config(path)


def test_dense_member_candidates_survive_automatic_merge(tmp_path):
    import asyncio
    import math

    from yaw.daemon.graph.storage.evidence import evidence_schemas
    from yaw.daemon.graph.storage.managed_sqlite import create_table

    definition = "A common definition for the same service processor."
    entities = [
        ("a", "The exceptionally long canonical service Processor A", definition),
        ("b", "the exceptionally long canonical service processor b", definition),
        ("c", "Orthogonal unrelated surface", "A related domain object."),
    ]
    vectors = {"a": [1.0, 0.0], "b": [0.98, math.sqrt(1 - 0.98**2)], "c": [0.66, math.sqrt(1 - 0.66**2)]}
    results = []
    for incremental in (False, True):
        with closing(open_yaw_db(tmp_path / f"dense-{incremental}.db")) as conn:
            for schema in asyncio.run(evidence_schemas()):
                create_table(conn, schema)
            for batch in [entities[:2], entities[2:]] if incremental else [entities]:
                import numpy as np

                upsert_entities(conn, batch)
                upsert_entity_vectors(conn, [(eid, vectors[eid]) for eid, _, _ in batch])
                conn.executemany(
                    "INSERT INTO entity_candidate_vectors VALUES (?,?,?,?,?)",
                    [
                        (eid, eid, name, defn, np.array(vectors[eid], dtype="<f4").tobytes())
                        for eid, name, defn in batch
                    ],
                )
                candidates, _, _, _ = prepare_synonym_candidates(
                    conn, batch, candidate_dense_threshold=0.7925, candidate_lexical_threshold=0.63
                )
            results.append({(pair["entity_a"], pair["entity_b"]) for pair in candidates})
    assert results == [{("a", "c")}, {("a", "c")}]


@pytest.mark.parametrize("below_admission", [False, True])
def test_remapped_candidate_scores_use_all_source_members(tmp_path, below_admission):
    import math

    entities = [
        ("a", "Canary", "A deployment test."),
        ("b", "canary", "A deployment test"),
        ("c", "ZQX", "A related domain concept."),
    ]
    angles = (0.0, 0.2, 0.6)
    expected = math.cos(0.4)
    if below_admission:
        entities = [
            ("a", "A" * 100 + "b", "A deployment test."),
            ("b", "a" * 100 + "c", "A deployment test."),
            ("c", "aaaaac x", "A related domain concept."),
        ]
        angles = (0.0, math.acos(0.98), -math.acos(0.79))
        expected = 0.79
    scores = []
    for incremental in (False, True):
        with closing(open_yaw_db(tmp_path / f"scores-{incremental}.db")) as conn:
            for batch in [entities[:2], entities[2:]] if incremental else [entities]:
                upsert_entities(conn, batch)
                upsert_entity_vectors(
                    conn,
                    [
                        (eid, [math.cos(angle), math.sin(angle)])
                        for eid, angle in zip("abc", angles, strict=True)
                        if eid in {row[0] for row in batch}
                    ],
                )
                seed_candidate_vectors(conn)
                candidates, _, _, _ = prepare_synonym_candidates(
                    conn, batch, candidate_dense_threshold=0.8, candidate_lexical_threshold=0.625
                )
            assert len(candidates) == 1
            scores.append(candidates[0]["dense_sim"])
    assert scores[0] == pytest.approx(expected)
    assert scores[0] == pytest.approx(scores[1])
