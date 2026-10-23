"""Reliability contracts over native DSPy objects, real SQLite and loopback inference."""

import asyncio
import copy
import inspect
import sqlite3
from contextlib import closing

import dspy
import pytest

from daemon.graph.inference.test_reflection import proposer
from daemon.graph.inference.test_reflection import provider as provider
from yaw.core import ChunkPointer, YawRuntimeError
from yaw.daemon.graph.candidate_cache import CandidatePayloadCodec
from yaw.daemon.graph.engine import YawGraphEngine
from yaw.daemon.graph.inference import HippoRAGProgram, RLMInstructionProposer
from yaw.daemon.graph.inference.chat import YawDSPyLM, create_dspy_lm


def test_native_qa_predictor_state_survives_json_restore_and_deepcopy(tmp_path):
    program = HippoRAGProgram(enable_gbnf=False)
    assert type(program.qa) is dspy.ChainOfThought
    predictor = dict(program.named_predictors())["qa.predict"]
    assert dict(program.all_stage_predictors())["qa"] is predictor
    lm = create_dspy_lm("Gemma-4-E4B-it-GGUF", cache=False)
    program.qa.set_lm(lm)
    assert predictor.lm is lm
    predictor.signature = predictor.signature.with_instructions("Use source names.")
    predictor.demos = [dspy.Example(context="A", question="Q", answer="A")]
    assert not {"signature", "lm", "demos"}.intersection(vars(program.qa))
    # State-only saves must not require loading a custom LM class.
    program.qa.set_lm(None)
    saved = tmp_path / "qa.json"
    program.save(str(saved))
    restored = HippoRAGProgram(enable_gbnf=False)
    restored.load(str(saved))
    assert type(restored.qa) is dspy.ChainOfThought
    assert restored.qa.predict.signature.instructions == "Use source names."
    assert restored.qa.predict.demos == [{"context": "A", "question": "Q", "answer": "A"}]
    clone = restored.deepcopy()
    clone.qa.predict.signature = clone.qa.predict.signature.with_instructions("Other instructions.")
    clone.qa.predict.demos.clear()
    assert restored.qa.predict.signature.instructions == "Use source names."
    assert len(restored.qa.predict.demos) == 1


def test_native_lm_capability_defaults_and_copy():
    bare = YawDSPyLM(model="openai/test", cache=False)
    assert bare._yaw_recipe is None
    assert bare._yaw_context_window_tokens is None
    assert bare._yaw_supports_gbnf is False
    configured = create_dspy_lm("Gemma-4-E4B-it-GGUF", context_window_tokens=65536)
    cloned = configured.copy(temperature=0.7)
    assert cloned._yaw_recipe == configured._yaw_recipe
    assert cloned._yaw_context_window_tokens == 65536
    assert cloned._yaw_supports_gbnf == configured._yaw_supports_gbnf
    assert cloned.kwargs["temperature"] == 0.7


@pytest.mark.parametrize("asynchronous", [False, True])
def test_rlm_tool_preserves_signature_binding_and_coroutine_identity(asynchronous):
    def add(left: int, right: int = 2) -> int:
        return left + right

    async def async_add(left: int, right: int = 2) -> int:
        return left + right

    function = async_add if asynchronous else add
    program = HippoRAGProgram(enable_gbnf=False)
    wrapped = program.shacl_healer_rlm._make_interpreter_tool(dspy.Tool(function))
    assert inspect.signature(wrapped) == inspect.signature(function)
    assert inspect.iscoroutinefunction(wrapped) is asynchronous
    result = wrapped(3, right=4)
    assert (asyncio.run(result) if asynchronous else result) == 7


@pytest.mark.parametrize("have_lm", [False, True])
def test_bad_reflection_setup_preserves_existing_rows_before_indexing(tmp_path, have_lm):
    lm = create_dspy_lm("reflection-model", cache=False) if have_lm else None
    program = HippoRAGProgram(enable_gbnf=False, orchestrator_lm=lm, rlm_sub_lm=lm)
    engine = YawGraphEngine(tmp_path / "yaw.db", tmp_path, program, dspy.Embedder(lambda texts: []))
    try:
        from yaw.daemon.graph.storage import upsert_chunk_pointers

        (tmp_path / "source.md").write_text("Preserve this source.\n", encoding="utf-8")
        with engine.managed_conn.transaction() as conn:
            upsert_chunk_pointers(conn, [(ChunkPointer("existing", "source.md", 1, 1), "Preserve this source.")])
        gold = dspy.Example(question="Who?", item_docs=["A new passage."]).with_inputs("question")
        with dspy.context(lm=None), pytest.raises(YawRuntimeError) as caught:
            engine.evolve_prompts([gold])
        assert caught.value.code == ("INVALID_INPUT" if have_lm else "MODEL_ERROR")
        assert [row[0] for row in engine.conn.execute("SELECT chunk_id FROM chunks")] == ["existing"]
        assert engine._opt_chunks_pool == []
    finally:
        engine.close()


@pytest.mark.parametrize("bad_steps", [-1, True, "1", 0.5])
def test_invalid_simba_progress_is_rejected_without_state_changes(bad_steps):
    prop = RLMInstructionProposer()
    before = prop.dump_simba_state()
    broken = copy.deepcopy(before)
    broken["steps"] = bad_steps
    with pytest.raises(ValueError, match="steps"):
        prop.load_simba_state(broken)
    assert prop.dump_simba_state() == before


def test_invalid_second_predictor_does_not_partially_restore_first():
    prop = RLMInstructionProposer()
    before = prop.dump_simba_state()
    broken = copy.deepcopy(before)
    broken["predictors"]["qwen_proposer"]["signature"]["instructions"] = "A different writer."
    broken["predictors"]["qwen_compressor"]["signature"]["fields"] = []
    with pytest.raises(ValueError, match="schema"):
        prop.load_simba_state(broken)
    assert prop.dump_simba_state() == before


def learning(tmp_path, url, *, resume=False):
    prop = proposer(url, enable_simba=False)
    task = HippoRAGProgram(enable_gbnf=False, orchestrator_lm=prop.worker_lm, rlm_sub_lm=prop.worker_lm)
    engine = YawGraphEngine(tmp_path / "yaw.db", tmp_path, task, dspy.Embedder(lambda texts: []))
    train = [dspy.Example(question="Who?", context="Source.").with_inputs("question", "context")]
    try:
        engine._configure_simba_task_learning(
            prop, task, train, lambda gold, pred: 0.0, save_path=tmp_path / "compiled.json", resume=resume
        )
    except ValueError:
        engine.close()
        raise
    return engine, prop


@pytest.mark.parametrize("corrupt", ["phase", "state", "request"])
def test_malformed_active_sqlite_checkpoint_preserves_durable_bytes(tmp_path, provider, corrupt):
    url, calls = provider
    engine, prop = learning(tmp_path, url)
    try:
        prop.checkpoint_put("request:one", {"phase": "collecting", "state": prop.dump_simba_state()})
        prop.checkpoint_put("request:one", {"phase": "learning", "state": prop.dump_simba_state()})
        key = engine.conn.execute("SELECT key FROM meta WHERE key LIKE '%:active_reflection'").fetchone()[0]
        codec = CandidatePayloadCodec(lambda: {})
        active = codec.decode(engine.get_meta(key))
        if corrupt == "phase":
            active["records"]["request:one"]["phase"] = "mystery"
        elif corrupt == "state":
            active["records"]["request:one"]["state"] = "broken"
        else:
            active["request"] = "request:missing"
        payload = codec.encode(active)
        engine.set_meta(key, payload)
    finally:
        engine.close()
    with pytest.raises(ValueError, match="Malformed"):
        learning(tmp_path, url, resume=True)
    with closing(sqlite3.connect(tmp_path / "yaw.db")) as conn:
        assert conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()[0] == payload
    assert calls == []


def test_invalid_request_is_rejected_before_any_sqlite_write(tmp_path, provider):
    url, calls = provider
    engine, prop = learning(tmp_path, url)
    try:
        before = engine.conn.total_changes
        with pytest.raises(ValueError, match="phase"):
            prop.checkpoint_put("request:invalid", {"phase": "unknown", "state": prop.dump_simba_state()})
        assert engine.conn.total_changes == before
        assert engine.conn.execute("SELECT COUNT(*) FROM candidate_stage_cache").fetchone()[0] == 0
    finally:
        engine.close()
    assert calls == []


@pytest.mark.parametrize("corrupt", ["instruction", "component", "length"])
def test_corrupted_completed_proposals_fail_without_generation_or_writes(tmp_path, provider, corrupt):
    url, calls = provider
    engine, prop = learning(tmp_path, url)
    candidate = {"qa.predict": "Use evidence."}
    data = {"qa.predict": [{"Inputs": {"question": "Who?"}, "Feedback": "Expected Turing"}]}
    try:
        assert prop(candidate, data, ["qa.predict"])
        generated = len(calls)
        key = engine.conn.execute("SELECT key FROM meta WHERE key LIKE '%:active_reflection'").fetchone()[0]
        codec = CandidatePayloadCodec(lambda: {})
        active = codec.decode(engine.get_meta(key))
        record = active["records"][active["request"]]
        if corrupt == "instruction":
            record["proposals"] = {"qa.predict": None}
        elif corrupt == "component":
            record["proposals"] = {"unknown": "Use evidence."}
        else:
            record["proposals"] = {"qa.predict": "x" * 2000}
        payload = codec.encode(active)
        engine.set_meta(key, payload)
    finally:
        engine.close()
    if corrupt == "instruction":
        with pytest.raises(ValueError, match="proposal instruction"):
            learning(tmp_path, url, resume=True)
    else:
        restored_engine, restored = learning(tmp_path, url, resume=True)
        try:
            with pytest.raises(ValueError, match="unknown component|length limit"):
                restored(candidate, data, ["qa.predict"])
        finally:
            restored_engine.close()
    assert len(calls) == generated
    with closing(sqlite3.connect(tmp_path / "yaw.db")) as conn:
        assert conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()[0] == payload


def test_recent_assertions_have_fixed_six_column_contract(tmp_path):
    from scripts.watchdog import recent_relation_assertions
    from yaw.daemon.graph.storage import open_managed_yaw_db

    db = open_managed_yaw_db(tmp_path / "yaw.db")
    try:
        with db.transaction() as conn:
            conn.executemany(
                "INSERT INTO entities(entity_id,name,definition) VALUES (?,?,?)",
                [("a", "A", "First concept"), ("b", "B", "Second concept")],
            )
            conn.execute(
                "INSERT INTO edges(src_id,dst_id,kind,weight) VALUES (?,?,?,?)", ("a", "b", "skos:related", 1.0)
            )
        assert recent_relation_assertions(conn, "skos:related") == [
            ("a", "A", "First concept", "b", "B", "Second concept")
        ]
    finally:
        db.close()
