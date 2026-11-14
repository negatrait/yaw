"""Real SQLite identity propagation and order-independent decision application."""

import json
import math
from contextlib import closing
from itertools import permutations

import cocoindex as coco
import pytest
from daemon.graph.test_evidence_revisions import native_environment
from daemon.graph.test_indexing_overlap import provider as provider

from yaw.core import SynonymEvidenceBatch
from yaw.daemon.graph.inference.components import JudgedSynonyms, SemanticAdjudicationInput, judge_semantics
from yaw.daemon.graph.planner import apply_synonym_adjudication
from yaw.daemon.graph.storage import load_graph_topology, open_yaw_db, upsert_entities, upsert_entity_labels
from yaw.daemon.graph.storage.synonyms import SynonymConstraints


def packet(pairs):
    candidates = tuple({"entity_a": a, "entity_b": b, "pair_id": f"{a}|{b}"} for a, b in pairs)
    return SynonymEvidenceBatch(
        candidates,
        json.dumps(
            {
                "format": "synonym-evidence-v3",
                "entities": {e: {"name": e, "sources": []} for pair in pairs for e in pair},
                "sources": {},
                "pairs": [
                    {
                        **candidate,
                        "entity_a": candidate["entity_a"],
                        "entity_b": candidate["entity_b"],
                        "direct_facts": [],
                        "shared_role_facts": [],
                    }
                    for candidate in candidates
                ],
            }
        ),
        {},
    )


def test_union_remaps_separation_and_omits_only_collapsed_pairs(tmp_path):
    with closing(open_yaw_db(tmp_path / "constraints.db")) as conn:
        constraints = SynonymConstraints(conn, (("a", "d"),))
        assert constraints.merge("b", "c")
        assert constraints.merge("a", "b")
        assert constraints.resolve("c") == "a"
        assert constraints.separated("c", "d")
        assert not constraints.merge("c", "d")
        batch = constraints.prune(packet((("a", "c"), ("c", "d"), ("d", "e"))))
        rows = json.loads(batch.payload)["pairs"]
        assert [row["pair_id"] for row in rows] == ["c|d", "d|e"]
        assert rows[0]["merge_forbidden"] is True
        assert "merge_forbidden" not in rows[1]


@coco.fn
async def judge_once(request: SemanticAdjudicationInput) -> JudgedSynonyms:
    return await judge_semantics(request)


def test_separation_skips_labels_but_keeps_real_judge_votes(tmp_path, provider):
    url, state = provider
    state["delay"] = 0
    state["logprob_probability"] = lambda output: 0.64 if "broader" in output["decisions"][0] else 0.81
    with closing(open_yaw_db(tmp_path / "constraints.db")) as conn:
        batch = SynonymConstraints(conn, (("a", "b"),)).prune(packet((("a", "b"),)))
    with native_environment(tmp_path, tmp_path / "index", url) as (_, env, settings):
        app = coco.App(
            coco.AppConfig(name="constraint_judge", environment=env),
            judge_once,
            SemanticAdjudicationInput(batch, settings.program_identity, settings.synonym_identity),
        )
        result = app.update_blocking()
        assert isinstance(result, JudgedSynonyms)
        assert {entry["predictor"] for entry in result.audit_trail} == {"relation_judge"}
        assert result.decisions[0]["llm_prob"] == pytest.approx(0.81)
        assert len(state["synonym_requests"]) == 1
        assert all(request["logprobs"] is True for request in state["synonym_requests"])
        assert all("logprobs" not in trace.outputs for trace in result.traces)
        assert app.update_blocking() == result
        assert len(state["synonym_requests"]) == 1


def test_decision_permutations_preserve_graph_and_bayesian_weights(tmp_path):
    pairs = (("a", "b"), ("b", "c"), ("a", "c"), ("d", "e"))
    candidates = [
        {"pair_id": f"{a}|{b}", "entity_a": a, "entity_b": b, "similarity": 0.81, "lexical_sim": 0.8} for a, b in pairs
    ]
    decisions = [
        {"pair_id": f"{a}|{b}", "relation": "skos:altLabel" if index < 2 else "skos:related", "llm_prob": 0.64}
        for index, (a, b) in enumerate(pairs)
    ]
    expected = None
    for index, order in enumerate(permutations(decisions)):
        with closing(open_yaw_db(tmp_path / f"order-{index}.db")) as conn:
            upsert_entities(conn, [(entity, f"Service {entity}", "A service.") for entity in "abcde"])
            upsert_entity_labels(
                conn, [(f"service {e}", e, "skos:prefLabel", f"Service {e}", "__seed__") for e in "abcde"]
            )
            apply_synonym_adjudication(conn, candidates, {}, order, set(), rlm_stats={})
            snapshot = {
                table: sorted(tuple(row) for row in conn.execute(f"SELECT * FROM {table}"))
                for table in ("entities", "entity_labels", "edges", "edge_sources")
            }
            if expected is None:
                expected = snapshot
            assert snapshot == expected
            assert conn.execute("SELECT weight FROM edges WHERE src_id='d' AND dst_id='e'").fetchone()[0] == (
                pytest.approx(math.sqrt(0.64 * 0.81))
            )


@pytest.mark.parametrize("separation", ["skos:related", "skos:broader", "owl:differentFrom"])
@pytest.mark.parametrize("probability", [0.16, 0.64])
def test_early_separation_survives_later_merge_proposals(tmp_path, separation, probability):
    pairs = (("a", "b"), ("b", "c"), ("a", "c"))
    candidates = [
        {"pair_id": f"{a}|{b}", "entity_a": a, "entity_b": b, "similarity": 0.81, "lexical_sim": 0.8} for a, b in pairs
    ]
    decisions = [
        {"pair_id": f"{a}|{b}", "relation": separation if index == 0 else "skos:altLabel", "llm_prob": probability}
        for index, (a, b) in enumerate(pairs)
    ]
    expected = None
    for index, order in enumerate(permutations(decisions)):
        with closing(open_yaw_db(tmp_path / f"separation-{index}.db")) as conn:
            upsert_entities(conn, [(entity, f"Service {entity}", "A service.") for entity in "abc"])
            upsert_entity_labels(
                conn, [(f"service {e}", e, "skos:prefLabel", f"Service {e}", "__seed__") for e in "abc"]
            )
            apply_synonym_adjudication(conn, candidates, {}, order, set(), rlm_stats={})
            assert conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0] == 2
            edges = [tuple(row) for row in conn.execute("SELECT * FROM edges ORDER BY src_id,dst_id,kind")]
            assert {edge[2] for edge in edges} == {separation}
            assert all(edge[3] == pytest.approx(math.sqrt(probability * 0.81)) for edge in edges)
            if separation == "owl:differentFrom":
                assert all(edge[2] != "owl:differentFrom" for edge in load_graph_topology(conn)[2])
            if expected is None:
                expected = edges
            assert edges == expected
