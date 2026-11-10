"""Fixed-evidence native publication parity with measured judge probabilities."""

import pytest

from daemon.graph.test_batch_equivalence import assert_same_graph, canonical_graph
from daemon.graph.test_evidence_revisions import native_corpus, publish
from daemon.graph.test_indexing_overlap import provider as provider
from yaw.core import ChunkPointer
from yaw.daemon.indexing import CorpusIndexingLiveComponent


@pytest.mark.parametrize("reverse_decisions", [False, True])
def test_constraint_propagation_64_plus_64_equals_128(tmp_path, provider, reverse_decisions):
    url, state = provider
    state["delay"] = 0
    state["relation"] = "skos:altLabel"
    state["reverse_synonym_decisions"] = reverse_decisions
    state["embedding_vector"] = lambda value: [0.0, 1.0, 0.0, 0.0] if "destination" in value else [1.0, 0.0, 0.0, 0.0]
    names = ("Orion North", "Orion South", "Orion East", "Orion West", "Orion Central", "Orion Remote")
    chunks = []
    state["extractions"] = {}
    for index in range(128):
        sense = index % 3 if index < 64 else 3 + index % 3
        name = names[sense]
        text = f"Evidence {index}: {name} routes telemetry through Anchor."
        pointer = ChunkPointer(f"chunk-{index}", f"source-{index}.md", 1, 1)
        (tmp_path / pointer.rel_posix_path).write_text(text, encoding="utf-8")
        chunks.append((pointer, text))
        state["extractions"][text] = {
            "named_entities": [
                [name, f"The telemetry routing service described in regional account {name}."],
                ["Anchor", "The destination for telemetry routing."],
            ],
            "triples": [
                {
                    "sub": name,
                    "pred": "routes through",
                    "obj": "Anchor",
                    "def": "The telemetry routing service sends its output to Anchor.",
                }
            ],
        }
    expected = None
    skipped = 0
    for arm in ("incremental", "reference"):
        with native_corpus(tmp_path, tmp_path / arm, url) as (db, env, app):
            if arm == "incremental":
                publish(env, app, chunks[:64])
            publish(env, app, chunks)
            component = CorpusIndexingLiveComponent.last_instance
            assert component is not None and component.last_summary is not None
            skipped += component.last_summary.rlm_stats.get("constraint_resolved_pairs", 0)
            with db.readonly() as conn:
                actual = canonical_graph(conn)
                assert conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0] == 2
                assert conn.execute("SELECT COUNT(*) FROM fact_assertions").fetchone()[0] == 128
                if expected is None:
                    expected = actual
                else:
                    assert_same_graph(actual, expected)
    assert skipped > 0
    assert state["synonym_requests"]
    assert all(request["logprobs"] is True for request in state["synonym_requests"])
