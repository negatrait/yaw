"""Complete fields reach the actual Gemma transport under native DSPy."""

import dspy

from daemon.graph.inference.test_completions_engine import completions_server as completions_server
from yaw.core.config import YawConfig
from yaw.daemon.graph.inference.chat import create_dspy_lm
from yaw.daemon.graph.inference.grammars import DSPY_NER_GBNF_GRAMMAR, DSPY_TRIPLES_GBNF_GRAMMAR
from yaw.daemon.graph.inference.program import YawProgram


class PreserveEvidence(dspy.Signature):
    """Read the supplied evidence without discarding text."""

    passage: str = dspy.InputField()
    context: str = dspy.InputField()
    answer: str = dspy.OutputField()


def test_direct_fields_are_not_sliced_before_dispatch(completions_server):
    passage = "Î±" * 70000 + " PASSAGE_TAIL_Î©"
    context = "Evidence paragraph.\n" * 9000 + " CONTEXT_TAIL_Î»"
    lm = create_dspy_lm("gemma-4-e4b", base_url=completions_server["base_url"], api_key="k", cache=False)
    program = YawProgram(enable_gbnf=False, orchestrator_lm=lm, rlm_sub_lm=lm)
    prediction = program._invoke_stage(
        dspy.Predict(PreserveEvidence),
        lm=lm,
        use_gbnf=False,
        grammar_fn=lambda: "",
        stage_name="qa",
        passage=passage,
        context=context,
    )
    assert prediction.answer == "ok"
    ((_, _, body),) = completions_server["requests"]
    assert passage in body["prompt"] and context in body["prompt"]
    assert body["prompt"].count("PASSAGE_TAIL_Î©") == 1
    assert body["prompt"].count("CONTEXT_TAIL_Î»") == 1


def test_default_generation_grammar_has_no_hidden_semantic_field_caps():
    assert YawConfig().gbnf_max_reasoning_chars is None
    assert YawProgram().gbnf_max_reasoning_chars is None
    for grammar in (DSPY_NER_GBNF_GRAMMAR, DSPY_TRIPLES_GBNF_GRAMMAR):
        assert "{1,512}" not in grammar and "{0,39}" not in grammar


def test_compiled_artifact_does_not_restore_an_old_reasoning_cutoff(tmp_path):
    path = tmp_path / "compiled.json"
    old = YawProgram(gbnf_max_reasoning_chars=360)
    old.save(str(path))
    restored = YawProgram(compiled_path=path)
    assert restored.gbnf_max_reasoning_chars is None
    from yaw.daemon.graph.inference.adapter import YawGBNFAdapter

    assert isinstance(restored.json_adapter, YawGBNFAdapter)
    assert restored.json_adapter.max_reasoning_chars is None
