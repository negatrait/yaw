"""Healed derived statements publish without rewriting or losing source evidence."""

import json
from contextlib import closing

import pytest

from daemon.graph.test_evidence_revisions import native_corpus, publish
from daemon.graph.test_indexing_overlap import provider as provider
from yaw.core import ChunkPointer, YawRuntimeError
from yaw.daemon.graph.inference.grammars import build_dspy_shacl_repair_grammar
from yaw.daemon.graph.inference.parsers import parse_shacl_repairs, shacl_repair_actions
from yaw.daemon.graph.inference.program import YawProgram
from yaw.daemon.graph.storage import apply_shacl_repairs, lint_graph_shacl, open_yaw_db
from yaw.daemon.indexing import CorpusIndexingLiveComponent


@pytest.mark.parametrize("contract", [None, "bound-report-v1", "bound-report-v2", "bound-report-v3"])
def test_saved_program_requires_the_bound_report_tool_contract(contract):
    program = YawProgram()
    state = program.dump_state()
    state["yaw_surface"]["shacl_tools"] = contract
    with pytest.raises(ValueError, match="Incompatible SHACL tool contract"):
        program.load_state(state)


@pytest.mark.parametrize("action", ["assert_differentFrom", "rewrite_definition"])
def test_retired_action_is_rejected(tmp_path, action):
    repair = {"focus_node": "entity-a|entity-b", "action": action, "value": "entity-b"}
    with pytest.raises(ValueError, match="Invalid SHACL repair"):
        parse_shacl_repairs([repair])
    with closing(open_yaw_db(tmp_path / "graph.db")) as conn:
        with pytest.raises(ValueError, match="Invalid SHACL repair action"):
            apply_shacl_repairs(conn, [repair])
        assert conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 0


def test_nonempty_repairs_cover_the_report():
    allowed = {focus: {"set_definition"} for focus in ("fact-a", "fact-b")}
    repairs = [{"focus_node": focus, "action": "set_definition", "value": "A relates to B."} for focus in allowed]
    with pytest.raises(ValueError, match="Incomplete SHACL repairs"):
        parse_shacl_repairs(repairs[:1], allowed_actions=allowed)
    assert parse_shacl_repairs(repairs, allowed_actions=allowed) == repairs
    assert parse_shacl_repairs([], allowed_actions=allowed) == []


@pytest.mark.parametrize("action", ["delete_dangling", "delete_invalid_fact", "set_pref_label"])
def test_definition_repairs_cannot_remove_or_relabel_statements(action):
    allowed = shacl_repair_actions([{"focus_node": "fact-a", "result_path": "skos:definition"}])
    repair = {"focus_node": "fact-a", "action": action, "value": "Repaired relation description."}
    with pytest.raises(ValueError, match="not permitted"):
        parse_shacl_repairs([repair], allowed_actions=allowed)
    repaired = {**repair, "action": "set_definition"}
    assert parse_shacl_repairs([repaired], allowed_actions=allowed) == [repaired]
    rules = dict(
        line.split(" ::= ", 1) for line in build_dspy_shacl_repair_grammar(allowed_actions=allowed).splitlines()
    )
    assert rules["repair"] == "repair-0"
    assert "set_definition" in rules["repair-0"] and action not in rules["repair-0"]


@pytest.mark.parametrize("repair", [False, True])
@pytest.mark.parametrize("label", ["Rate Limits", "429s"])
@pytest.mark.parametrize("statements", [1, 33])
def test_native_shacl_healer_definition_survives_publication(tmp_path, provider, repair, label, statements):
    url, state = provider
    state["delay"] = 0
    definition = "HTTP request limits enforced by the service."
    text = f"{label} require retries after {', '.join(str(n + 1) for n in range(statements))} requests."
    state["extractions"] = {
        text: {
            "named_entities": [[label, definition], ["Retries", "Later attempts to execute a request."]],
            "triples": [
                {"sub": label, "pred": f"require after {n + 1} requests", "obj": "Retries", "def": definition}
                for n in range(statements)
            ],
        }
    }
    if repair:
        state["healed_definition"] = "Rate-limited HTTP responses require retries after waiting."
    state["embedding_vector"] = lambda value: (
        [1.0, 0.0, 0.0, 0.0] if "limits enforced" in value else [0.0, 1.0, 0.0, 0.0]
    )
    pointer = ChunkPointer("source", "source.md", 1, 1)
    (tmp_path / pointer.rel_posix_path).write_text(text, encoding="utf-8")
    with native_corpus(tmp_path, tmp_path / "index", url) as (db, stream, app):
        if label == "429s" and not repair:
            with pytest.raises(YawRuntimeError, match="Graph build left .* SHACL violations"):
                publish(stream, app, [(pointer, text)])
            with db.readonly() as conn:
                assert conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 0
                assert conn.execute("SELECT 1 FROM sqlite_master WHERE name='fact_assertions'").fetchone() is None
            return
        publish(stream, app, [(pointer, text)])
        if label == "Rate Limits":
            assert state.get("healer_requests", []) == []
            expected_subj = "Rate Limits"
        else:
            assert set(state["healer_requests"]) == {"direct"}
            contexts = [json.JSONDecoder().raw_decode(context.strip())[0] for context in state["healer_contexts"]]
            assert all(text in info["source_passages"].values() for context in contexts for info in context)
            assert all(info.get("focus_kind") == "entity" for context in contexts for info in context)
            expected_subj = "Rate limit responses"
        with db.readonly() as conn:
            assert lint_graph_shacl(conn) == []
            assert (
                conn.execute("SELECT definition FROM facts ORDER BY pred ASC").fetchone()[0]
                == f"{expected_subj} require after 1 requests Retries."
            )
            raw = json.loads(conn.execute("SELECT raw_json FROM fact_assertions").fetchone()[0])
            assert raw["spo_def"] == definition
            assert raw["subj"] == label
            assert (
                conn.execute("SELECT COUNT(*) FROM fact_assertions WHERE fact_id IS NOT NULL").fetchone()[0]
                == statements
            )
            assert conn.execute("SELECT COUNT(*) FROM entity_mentions").fetchone()[0] == 2
            assert conn.execute("SELECT COUNT(*) FROM fact_vec").fetchone()[0] == statements
            assert conn.execute("SELECT COUNT(*) FROM fact_fts").fetchone()[0] == statements
        calls = len(state["intervals"])
        changes = conn.total_changes
        publish(stream, app, [(pointer, text)])
        assert len(state["intervals"]) == calls
        assert conn.total_changes == changes


def test_native_assertion_backed_provenance_repair_bypasses_healer(tmp_path, provider):
    url, state = provider
    state["delay"] = 0
    definition = "A telemetry service."
    chunks = []
    for index, destination in enumerate(("Anchor", "Beacon")):
        text = f"Orion reports to {destination}."
        pointer = ChunkPointer(f"chunk-{index}", f"source-{index}.md", 1, 1)
        chunks.append((pointer, text))
        (tmp_path / pointer.rel_posix_path).write_text(text, encoding="utf-8")
        state.setdefault("extractions", {})[text] = {
            "named_entities": [["Orion", definition], [destination, f"The {destination} destination."]],
            "triples": [
                {
                    "sub": "Orion",
                    "pred": "reports to",
                    "obj": destination,
                    "def": f"Orion sends telemetry to {destination}.",
                }
            ],
        }
    state["embedding_vector"] = lambda value: (
        [1.0, 0.0, 0.0, 0.0]
        if "telemetry service" in value
        else [0.0, 1.0, 0.0, 0.0]
        if "Anchor" in value
        else [0.0, 0.0, 1.0, 0.0]
    )
    with native_corpus(tmp_path, tmp_path / "index", url) as (db, env, app):
        publish(env, app, chunks[:1])
        # Seed an incomplete snapshot without stale CocoIndex target fingerprints.
        # Out-of-band damage to an already-owned target is a separate recovery problem.
        repair_dir = tmp_path / "repair"
        repair_dir.mkdir()
        with db.readonly() as source, closing(open_yaw_db(repair_dir / "yaw.db")) as snapshot:
            source.backup(snapshot)
    with closing(open_yaw_db(repair_dir / "yaw.db")) as conn:
        original_fact = conn.execute("SELECT fact_id FROM facts").fetchone()[0]
        raw = conn.execute("SELECT raw_json FROM fact_assertions").fetchone()[0]
        original_chunk = conn.execute("SELECT chunk_id FROM fact_assertions").fetchone()[0]
        conn.execute("DELETE FROM fact_sources")
        conn.commit()
        assert any(v.shape == "ProvenanceShape/factSources" for v in lint_graph_shacl(conn))
    with native_corpus(tmp_path, repair_dir, url) as (db, env, app):
        publish(env, app, chunks)
        component = CorpusIndexingLiveComponent.last_instance
        assert component is not None and component.last_summary is not None
        stats = component.last_summary.rlm_stats
        assert stats["reused_chunks"] == 1
        assert stats["shacl_deterministic_healed"] == 1
        assert state.get("healer_requests", []) == []
        with db.readonly() as conn:
            assert lint_graph_shacl(conn) == []
            assert tuple(
                conn.execute(
                    "SELECT chunk_id,llm_prob,vec_coherence,atomic_confidence FROM fact_sources WHERE fact_id=?",
                    (original_fact,),
                ).fetchone()
            ) == (original_chunk, 0.81, pytest.approx(1e-6), pytest.approx((0.81 * 1e-6) ** 0.5))
            assert (
                conn.execute("SELECT raw_json FROM fact_assertions WHERE fact_id=?", (original_fact,)).fetchone()[0]
                == raw
            )
