"""Real DSPy JSON persistence preserves the saved demo policy."""

import dspy
import pytest

from yaw.daemon.graph.inference import HippoRAGProgram


@pytest.mark.parametrize("demo_policy", ["empty", "default", "custom"])
def test_saved_qa_demos_replace_constructor_defaults(tmp_path, demo_policy):
    program = HippoRAGProgram(enable_gbnf=False, include_demos=True)
    assert program.qa.predict.demos
    if demo_policy == "empty":
        program.qa.predict.demos = []
    elif demo_policy == "custom":
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

    restored = HippoRAGProgram(enable_gbnf=False, include_demos=True)
    restored.load(str(saved))
    assert restored.qa.predict.dump_state()["demos"] == expected


@pytest.mark.parametrize("include_demos", [False, True])
def test_legacy_missing_qa_state_retains_constructor_policy(include_demos):
    program = HippoRAGProgram(enable_gbnf=False, include_demos=include_demos)
    expected = program.qa.predict.dump_state()["demos"]
    state = program.dump_state()
    del state["qa.predict"]

    restored = HippoRAGProgram(enable_gbnf=False, include_demos=include_demos)
    restored.load_state(state)
    assert restored.qa.predict.dump_state()["demos"] == expected
