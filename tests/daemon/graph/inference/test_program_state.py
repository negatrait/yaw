"""Real DSPy JSON persistence preserves the saved demo policy."""

import dspy
import pytest

from yaw.daemon.graph.inference import YawProgram


@pytest.mark.parametrize("demo_policy", ["empty", "custom"])
def test_saved_qa_demos_replace_constructor_defaults(tmp_path, demo_policy):
    program = YawProgram(enable_gbnf=False)
    assert not program.qa.predict.demos
    if demo_policy == "custom":
        program.qa.predict.demos = [
            dspy.Example(
                context="Aster is a project.",
                question="What is Aster?",
                reasoning="The evidence identifies its type.",
                answer="A project.",
            ).with_inputs("context", "question")
        ]
    expected = program.qa.predict.dump_state()["demos"]
    saved = tmp_path / "program.json"
    program.save(str(saved))

    restored = YawProgram(enable_gbnf=False)
    restored.qa.predict.demos = [
        dspy.Example(
            context="Other.",
            question="Other?",
            reasoning="Other.",
            answer="Other.",
        ).with_inputs("context", "question")
    ]
    restored.load(str(saved))
    assert restored.qa.predict.dump_state()["demos"] == expected


def test_legacy_missing_qa_state_retains_constructor_policy():
    program = YawProgram(enable_gbnf=False)
    expected = program.qa.predict.dump_state()["demos"]
    state = program.dump_state()
    del state["qa.predict"]

    restored = YawProgram(enable_gbnf=False)
    restored.load_state(state)
    assert restored.qa.predict.dump_state()["demos"] == expected
