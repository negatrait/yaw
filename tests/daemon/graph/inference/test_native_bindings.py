"""Real HTTP invariants for native predictor binding, copying and contracts."""

import asyncio
import copy
import inspect
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import dspy
import pytest
from dspy.utils.exceptions import AdapterParseError

from yaw.daemon.graph.inference import HippoRAGProgram, RLMInstructionProposer, create_dspy_lm
from yaw.daemon.graph.inference.program import YawGBNFAdapter
from yaw.daemon.graph.inference.prompts import ExtractEntities


@pytest.fixture
def provider():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(request)
            body = json.dumps(
                {
                    "id": "native-binding",
                    "object": "chat.completion",
                    "model": request["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": json.dumps(
                                    {
                                        "named_entities": [["Widget", "An object described in the source."]],
                                        "reasoning": "Supported by the source.",
                                        "answer": "Widget",
                                        "query_intent": "Widget is an object",
                                    }
                                ),
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/v1", requests
        finally:
            server.shutdown()
            thread.join(timeout=5)


def lm_for(url, name="bound", **kwargs):
    return create_dspy_lm(name, base_url=url, api_key="test", cache=False, **kwargs)


def test_native_binding_and_copy_send_actual_generation_overrides(provider):
    url, requests = provider
    lm = lm_for(url, temperature=0.1, max_tokens=77)
    program = HippoRAGProgram(orchestrator_lm=lm, rlm_sub_lm=lm)
    assert program.ner.lm is lm
    assert program.get_lm() is lm
    copied = program.ner.lm.copy(temperature=0.9, max_tokens=123)
    with dspy.context(adapter=program.json_adapter):
        result = program.ner(passage="Widget is an object.", lm=copied)
    assert result.named_entities[0][0] == "Widget"
    assert requests[-1]["temperature"] == 0.9
    assert requests[-1].get("max_completion_tokens", requests[-1].get("max_tokens")) == 123
    assert lm.kwargs["temperature"] == 0.1
    assert lm.kwargs["max_tokens"] == 77


def test_native_predict_acall_reaches_real_provider(provider):
    url, requests = provider
    lm = lm_for(url)
    program = HippoRAGProgram(orchestrator_lm=lm, rlm_sub_lm=lm)

    async def invoke():
        with dspy.context(adapter=program.json_adapter, disable_history=True, track_usage=True):
            return await program.ner.acall(passage="Widget is an object.")

    result = asyncio.run(invoke())
    assert result.named_entities[0][0] == "Widget"
    assert len(requests) == 1
    assert not lm.history


def test_native_precedence_per_call_then_bound_then_context(provider):
    url, requests = provider
    bound = lm_for(url, "bound")
    ambient = lm_for(url, "ambient")
    override = lm_for(url, "override")
    program = HippoRAGProgram(orchestrator_lm=bound, rlm_sub_lm=bound)
    with dspy.context(lm=ambient, adapter=program.json_adapter):
        program.ner(passage="first")
        program.ner(passage="second", lm=override)
        program.extract_entities("third", lm=override)
        program.answer_question(context="Widget is an object.", question="What is it?", lm=override)
    assert [request["model"] for request in requests] == ["bound", "override", "override", "override"]
    assert program.ner.lm is bound


def test_unbound_program_uses_context_set_after_construction(provider):
    url, requests = provider
    program = HippoRAGProgram()
    assert all(predictor.lm is None for _, predictor in program.named_predictors())
    with dspy.context(lm=lm_for(url, "contextual"), adapter=program.json_adapter):
        assert program.extract_entities("Widget is an object.") == ("Widget",)
    assert requests[-1]["model"] == "contextual"


def test_task_roles_and_query_intent_override_keep_models_independent(provider):
    url, requests = provider
    qa = lm_for(url, "qa")
    extraction = lm_for(url, "extraction")
    override = lm_for(url, "override")
    program = HippoRAGProgram(orchestrator_lm=qa, rlm_sub_lm=extraction)
    with dspy.context(lm=override):
        program.extract_entities("Widget is an object.")
        program.answer_question(context="Widget is an object.", question="What is it?")
        assert program.extract_query_intent("What is Widget?", lm=override) == "Widget is an object"
    assert [request["model"] for request in requests] == ["extraction", "qa", "override"]


def test_candidate_deepcopy_preserves_native_lm_copy_settings(provider):
    url, requests = provider
    lm = lm_for(url, temperature=0.1, max_tokens=77)
    program = HippoRAGProgram(orchestrator_lm=lm, rlm_sub_lm=lm)
    program.ner.set_lm(lm.copy(temperature=0.8, max_tokens=119))
    candidate = copy.deepcopy(program)
    candidate.extract_entities("Widget is an object.")
    assert requests[-1]["temperature"] == 0.8
    assert requests[-1].get("max_completion_tokens", requests[-1].get("max_tokens")) == 119
    assert program.qa.predict.lm.kwargs["temperature"] == 0.1


def test_native_state_roundtrip_keeps_demos_and_explicit_deployment_lms(tmp_path, provider):
    url, requests = provider
    lm = lm_for(url)
    program = HippoRAGProgram(orchestrator_lm=lm, rlm_sub_lm=lm)
    program.ner.demos = [dspy.Example(passage="A source", named_entities=[["A", "An example concept."]])]
    program.qa.predict.signature = program.qa.predict.signature.with_instructions("Use only supplied evidence.")
    path = tmp_path / "compiled.json"
    program.save(str(path), save_program=False)
    replacement = lm_for(url, "replacement")
    restored = HippoRAGProgram(orchestrator_lm=replacement, rlm_sub_lm=replacement)
    restored.load(str(path))
    assert restored.ner.demos[0]["named_entities"] == [["A", "An example concept."]]
    assert restored.qa.predict.signature.instructions == "Use only supplied evidence."
    assert restored.get_lm() is replacement
    restored.extract_entities("Widget is an object.")
    assert requests[-1]["model"] == "replacement"


def test_fresh_programs_are_zero_shot_and_examples_require_explicit_opt_in():
    program = HippoRAGProgram()
    proposer = RLMInstructionProposer()
    assert all(not predictor.demos for _, predictor in program.named_predictors())
    assert not proposer.qwen_proposer.demos and not proposer.qwen_compressor.demos
    assert all(not predictor.demos for _, predictor in proposer.inner_rlm.named_predictors())
    examples = HippoRAGProgram(include_demos=True)
    assert len(examples.ner.demos) == 1
    assert len(examples.triples.demos) == 1
    assert len(examples.triple_filter.demos) == 10
    assert len(examples.qa.predict.demos) == 2
    assert not program.ner.demos


def test_legacy_state_missing_a_predictor_uses_native_configured_defaults(provider):
    url, _requests = provider
    lm = lm_for(url)
    program = HippoRAGProgram(orchestrator_lm=lm, rlm_sub_lm=lm)
    state = program.dump_state()
    del state["entity_definer"]
    restored = HippoRAGProgram(orchestrator_lm=lm, rlm_sub_lm=lm)
    restored.load_state(state)
    assert restored.entity_definer.lm is lm
    assert not restored.entity_definer.demos


@pytest.mark.parametrize("payload", ['{"reasoning":"Incomplete"}', '{"answer":"Incomplete"}', "{}"])
def test_repaired_multifield_responses_must_satisfy_complete_signature(payload):
    signature = HippoRAGProgram().qa.predict.signature
    with pytest.raises(AdapterParseError):
        YawGBNFAdapter().parse(signature, payload)


def test_repaired_multifield_response_preserves_all_typed_outputs():
    signature = HippoRAGProgram().qa.predict.signature
    result = YawGBNFAdapter().parse(signature, "{'reasoning': 'Evidence', 'answer': 'Widget'}")
    assert result == {"reasoning": "Evidence", "answer": "Widget"}


def test_native_parser_rejects_wrong_field_names_types_and_bare_arrays():
    class Count(dspy.Signature):
        number: int = dspy.OutputField()

    with pytest.raises(AdapterParseError):
        YawGBNFAdapter().parse(Count, '{"number": "not an integer"}')
    with pytest.raises(AdapterParseError):
        YawGBNFAdapter().parse(Count, '{"alternate": 42}')
    with pytest.raises(AdapterParseError):
        YawGBNFAdapter().parse(ExtractEntities, '[["Widget", "An object."]]')
    assert inspect.unwrap(YawGBNFAdapter.parse) is inspect.unwrap(dspy.JSONAdapter.parse)


def test_history_remains_explicit_conversation_input_not_diagnostic_storage():
    class Conversation(dspy.Signature):
        question: str = dspy.InputField()
        history: dspy.History = dspy.InputField()
        answer: str = dspy.OutputField()

    history = dspy.History(messages=[{"question": "Which object?", "answer": "Widget"}])
    messages = YawGBNFAdapter().format(Conversation, [], {"question": "What was its name?", "history": history})
    assert [message["role"] for message in messages] == ["system", "user", "assistant", "user"]
    assert "Widget" in messages[-2]["content"]
    assert len(history.messages) == 1
