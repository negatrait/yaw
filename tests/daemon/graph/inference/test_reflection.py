"""Native DSPy reflection over real HTTP, SQLite and process boundaries."""

import gc
import json
import subprocess
import sys
import threading
import weakref
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import dspy
import pytest
from dspy.clients.requests import build_request

from yaw.daemon.graph.inference.chat import create_dspy_lm
from yaw.daemon.graph.inference.proposer import RLMInstructionProposer
from yaw.daemon.graph.inference.reflection import (
    ReflectionBudgetError,
    ReflectionLM,
    reflection_examples,
    resolve_evidence,
)
from yaw.daemon.graph.storage import (
    get_meta,
    open_managed_yaw_db,
    set_meta,
)


@pytest.fixture
def provider(request: pytest.FixtureRequest):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def reply(self, value):
            raw = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path != "/v1/models":
                self.send_error(404)
                return
            self.reply(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": name,
                            "object": "model",
                            "created": 0,
                            "owned_by": "fixture",
                            "loaded": True,
                            "context_length": 131072 if name == "gemma-4-e4b" else 8192,
                            "alias_for": f"fixture/{name}:Q8",
                        }
                        for name in ("gemma-4-e4b", "embeddinggemma-2")
                    ],
                }
            )

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            is_completion = self.path.endswith("/completions") and not self.path.endswith("/chat/completions")
            calls.append(request)
            prompt = str(request["prompt"] if is_completion else request["messages"])
            if "variables_info" in prompt and "RLM_TOOL_CHECK" in prompt:
                result = {"plan": "Call the tool.", "code": "SUBMIT(answer=double(21))"}
            elif "variables_info" in prompt and "RLM_FAILURE" in prompt:
                self.send_error(500, "Selector unavailable")
                return
            elif "variables_info" in prompt and "code" in prompt and "INVALID_REF" not in prompt:
                result = {
                    "plan": "Inspect the end of the original feedback.",
                    "code": "SUBMIT(selection={'diagnosis': 'Use exact evidence', "
                    "'references': [{'example_id': examples_with_feedback[0]"
                    "['evidence_id'], 'path': []}]})",
                }
            elif "variables_info" in prompt and "INVALID_REF" in prompt:
                result = {
                    "plan": "Select a reference.",
                    "code": "SUBMIT(selection={'diagnosis':'test', 'references':"
                    "[{'example_id':'invented','path':[]}]})",
                }
            elif "shortened_instruction" in prompt:
                result = {"shortened_instruction": "Use source evidence; preserve the output schema."}
            elif "OVERSIZED_DRAFT" in prompt:
                result = {"new_instruction": "x" * 20000}
            elif "LONG_DRAFT" in prompt:
                result = {"new_instruction": "x" * 2000}
            else:
                result = {"new_instruction": "Use source evidence; preserve the output schema."}
            if is_completion:
                self.reply(
                    {
                        "id": "reflection",
                        "object": "text_completion",
                        "model": request["model"],
                        "choices": [{"index": 0, "text": json.dumps(result), "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
                    }
                )
                return
            self.reply(
                {
                    "id": "reflection",
                    "object": "chat.completion",
                    "created": 0,
                    "model": request["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": json.dumps(result)},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
                }
            )

        def log_message(self, format: str, *args: object) -> None:
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/v1", calls
        finally:
            server.shutdown()
            thread.join(5)


def proposer(url, **kwargs):
    lm = create_dspy_lm("reflection-model", base_url=url, api_key="test", cache=False)
    return RLMInstructionProposer(
        worker_lm=lm,
        orchestrator_lm=lm,
        enable_gbnf=False,
        max_instruction_chars=kwargs.pop("max_instruction_chars", 1500),
        **kwargs,
    )


def test_exact_evidence_and_training_answers():
    examples = reflection_examples(
        [
            {
                "Inputs": {"passage": "x" * 10000 + "needle"},
                "Generated Outputs": {"answer": "wrong"},
                "Feedback": "Expected=['Turing']",
            }
        ]
    )
    selection = {
        "references": [
            {
                "example_id": examples[0]["evidence_id"],
                "path": ["Inputs", "passage"],
                "start": 10000,
                "end": 10006,
            }
        ]
    }
    assert resolve_evidence(examples, selection)[0]["evidence"] == "needle"
    selection["references"][0]["end"] = 10007
    with pytest.raises(ValueError, match="span"):
        resolve_evidence(examples, selection)
    selection["references"][0]["example_id"] = "invented"
    with pytest.raises(KeyError):
        resolve_evidence(examples, selection)


def test_native_writer_and_complete_budget(provider):
    url, calls = provider
    prop = proposer(url, enable_simba=False)
    result = prop(
        component_name="qa.predict",
        current_instruction="Use evidence.",
        examples_with_feedback=[{"Feedback": "Expected=['Turing']"}],
    )
    assert "source evidence" in result.new_instruction
    assert "Turing" in str(calls[-1])
    assert "qa.predict" in str(calls[-1])
    assert prop.llm_compressor.signature.output_fields.keys() == {"shortened_instruction"}
    prop.reflection_input_chars = 10
    prop.worker_lm = prop.worker_lm.target
    prop.orchestrator_lm = prop.orchestrator_lm.target
    before = len(calls)
    with pytest.raises(ReflectionBudgetError):
        prop(component_name="qa.predict", current_instruction="Use evidence.")
    assert len(calls) == before


def test_optional_evidence_removed_before_inference(provider):
    url, calls = provider
    prop = proposer(url, enable_simba=False, reflection_input_chars=10000)
    prop(
        component_name="qa.predict",
        current_instruction="Use evidence.",
        examples_with_feedback=[{"Feedback": "x" * 50000}],
    )
    assert "x" * 1000 not in str(calls[-1])
    assert "Use evidence." in str(calls[-1])


def test_native_compression_uses_its_actual_signature(provider):
    url, calls = provider
    prop = proposer(url, enable_simba=False)
    result = prop(component_name="qa.predict", current_instruction="LONG_DRAFT")
    assert result.new_instruction == "Use source evidence; preserve the output schema."
    assert len(calls) == 2
    assert "shortened_instruction" in str(calls[-1])


def test_compression_and_demos_cannot_bypass_complete_request_budget(provider):
    url, calls = provider
    prop = proposer(url, enable_simba=False, reflection_input_chars=10000)
    with pytest.raises(ReflectionBudgetError):
        prop(component_name="qa.predict", current_instruction="OVERSIZED_DRAFT")
    assert len(calls) == 1  # The oversized compression request never reaches inference.
    prop.llm_proposer.demos = [
        dspy.Example(
            current_instruction="Use evidence.",
            examples_with_feedback=[{"Feedback": "x" * 50000}],
            length_limit="1500",
            new_instruction="Use evidence.",
        )
    ]
    with pytest.raises(ReflectionBudgetError):
        prop(component_name="qa.predict", current_instruction="Use evidence.")
    assert len(calls) == 1


def test_character_bound_survives_copy(provider):
    url, calls = provider
    prop = proposer(url)
    lm = ReflectionLM(prop.worker_lm, input_chars=800)
    with pytest.raises(ReflectionBudgetError):
        lm.copy(max_tokens=800)(build_request(lm.copy(max_tokens=800), "x" * 1000))
    assert not calls


def test_rlm_selects_original_evidence_in_real_sandbox(provider):
    url, calls = provider
    prop = proposer(url, enable_simba=False, rlm_max_iters=2)
    outer_trace = []
    with dspy.context(trace=outer_trace):
        result = prop.propose_for_component(
            "qa.predict",
            "Use evidence.",
            [{"Inputs": {"question": "Who?"}, "Feedback": "Expected=['Turing']"}],
        )
    assert outer_trace == []
    assert "source evidence" in result
    assert "fallback" not in prop.telemetry[-1]
    assert 2 <= len(calls) <= 3
    assert "Turing" in str(calls[-1])


@pytest.mark.parametrize("instruction", ["RLM_FAILURE", "INVALID_REF"])
def test_failed_selector_raises_without_fallback(provider, instruction):
    url, _calls = provider
    prop = proposer(url, enable_simba=False, rlm_max_iters=2)
    with pytest.raises((KeyError, ValueError, RuntimeError, Exception)):
        prop.propose_for_component(
            "qa.predict",
            instruction,
            [{"Inputs": {"question": "Who?"}, "Feedback": "Expected=['Turing']"}],
        )


def test_invalid_character_bound_and_legacy_state_rejected():
    prop = RLMInstructionProposer()
    with pytest.raises(ValueError, match="positive"):
        RLMInstructionProposer(reflection_input_chars=0)
    state = prop.dump_simba_state()
    state["version"] = 1
    with pytest.raises(ValueError, match="Unsupported"):
        prop.load_simba_state(state)


@pytest.mark.parametrize("interrupt_after_proposal", [False, True])
def test_proposal_checkpoint_survives_fresh_process(tmp_path, provider, interrupt_after_proposal):
    url, calls = provider
    prop = proposer(url, enable_simba=False)
    db = open_managed_yaw_db(tmp_path / "yaw.db")
    prop.checkpoint_get = lambda key: (
        json.loads(raw) if (raw := get_meta(db._conn, f"reflection:{key}")) is not None else None
    )
    prop.checkpoint_put = lambda key, value: set_meta(db._conn, f"reflection:{key}", json.dumps(value))
    candidate = {"qa.predict": "Use evidence."}
    data = {"qa.predict": [{"Inputs": {"question": "Who?"}, "Feedback": "Expected Turing"}]}
    # Select and persist evidence through the actual DSPy sandbox first.
    prop.select_evidence("qa.predict", candidate["qa.predict"], data["qa.predict"])
    evidence_calls = len(calls)
    if interrupt_after_proposal:
        save = prop.checkpoint_put

        def interrupted(key, value):
            save(key, value)
            if key.startswith("proposal:"):
                raise RuntimeError("Power lost after durable proposal")

        prop.checkpoint_put = interrupted
        with pytest.raises(RuntimeError, match="Power lost"):
            prop(candidate, data, ["qa.predict"])
        result = {"qa.predict": "Use source evidence; preserve the output schema."}
    else:
        result = prop(candidate, data, ["qa.predict"])
    count = len(calls)
    assert count == evidence_calls + 1
    db.close()
    script = """
import importlib.util,json,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('reflection_tests',sys.argv[1])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
p=module.proposer(sys.argv[2],enable_simba=False)
db=module.open_managed_yaw_db(Path(sys.argv[3]))
p.checkpoint_get=lambda k: json.loads(r) if (r:=module.get_meta(db._conn,f'reflection:{k}')) is not None else None
try:
 candidate={'qa.predict':'Use evidence.'}
 data={'qa.predict':[{'Inputs':{'question':'Who?'},'Feedback':'Expected Turing'}]}
 print(json.dumps(p(candidate,data,['qa.predict'])))
finally: db.close()
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, __file__, url, str(tmp_path / "yaw.db")],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout.strip().splitlines()[-1]) == result
    assert len(calls) == count


def test_each_reflection_role_binds_a_character_guard_without_discovery(provider):
    url, calls = provider
    prop = proposer(url)
    prop.orchestrator_lm = create_dspy_lm("another-selector", base_url=url, api_key="test")
    prop.preflight()
    assert isinstance(prop.orchestrator_lm, ReflectionLM)
    assert isinstance(prop.worker_lm, ReflectionLM)
    assert prop.orchestrator_lm.input_chars == prop.worker_lm.input_chars == 65536
    assert calls == []


def test_repeated_native_proposals_release_inputs_and_histories(provider):
    url, calls = provider
    prop = proposer(url, enable_simba=False)

    class TraceRecord(dict):
        pass

    retained = []
    outer_trace = []
    with dspy.context(trace=outer_trace, disable_history=True):
        for index in range(12):
            record = TraceRecord(Feedback=f"Expected training answer {index}: " + "x" * 50000)
            retained.append(weakref.ref(record))
            result = prop(
                component_name="qa.predict",
                current_instruction="Use evidence.",
                examples_with_feedback=[record],
            )
            assert result.new_instruction
            del record, result
            calls.clear()  # Provider inspection is test instrumentation, not application retention.
    gc.collect()
    assert all(reference() is None for reference in retained)
    assert outer_trace == []
    assert prop.worker_lm.history == prop.worker_lm.target.history == []
    assert prop.llm_proposer.traces == prop.llm_compressor.traces == []
    assert prop.worker_module.sub_telemetry == []


def test_completed_request_invalidates_when_learned_writer_changes(tmp_path, provider):
    url, calls = provider
    prop = proposer(url, enable_simba=False)
    db = open_managed_yaw_db(tmp_path / "yaw.db")
    try:

        def get(key):
            raw = get_meta(db._conn, f"reflection:{key}")
            return json.loads(raw) if raw is not None else None

        prop.checkpoint_get = get
        prop.checkpoint_put = lambda key, value: set_meta(db._conn, f"reflection:{key}", json.dumps(value))
        candidate = {"qa.predict": "Use evidence."}
        data = {"qa.predict": [{"Inputs": {"question": "Who?"}, "Feedback": "Expected Turing"}]}
        first = prop(candidate, data, ["qa.predict"])
        completed_calls = len(calls)
        assert prop(candidate, data, ["qa.predict"]) == first
        assert len(calls) == completed_calls
        prop.llm_proposer.signature = prop.llm_proposer.signature.with_instructions(
            prop.llm_proposer.signature.instructions + "\nPreserve exact source names in task outputs."
        )
        assert prop(candidate, data, ["qa.predict"]) == first
        assert len(calls) == completed_calls + 1  # Evidence is reused; the new writer runs once.
        assert "Preserve exact source names in task outputs." in str(calls[-1])
        assert prop(candidate, data, ["qa.predict"]) == first
        assert len(calls) == completed_calls + 1
    finally:
        db.close()


def test_budget_includes_default_lm_template_options(provider):
    url, calls = provider
    prop = proposer(url)
    prop.worker_lm.kwargs["test_option"] = "x" * 100000
    with pytest.raises(ReflectionBudgetError, match="characters"):
        prop(component_name="qa.predict", current_instruction="Use evidence.")
    assert calls == []


@pytest.mark.parametrize("provider", [65536, 131072], indirect=True)
def test_default_reflection_character_bound_is_independent_of_context(provider, request):
    url, calls = provider
    context = request.node.callspec.params["provider"]
    lm = create_dspy_lm("reflection-model", base_url=url, cache=False, context_window_tokens=context)
    guarded = ReflectionLM(lm)
    assert guarded.input_chars == 65536
    assert guarded.kwargs["max_tokens"] is None
    prop = RLMInstructionProposer(orchestrator_lm=lm, worker_lm=lm)
    prop.preflight()
    assert prop.reflection_input_chars == prop.worker_lm.input_chars == 65536
    assert prop.max_instruction_chars == 1500 * context // 65536
    with pytest.raises(ReflectionBudgetError, match="characters"):
        guarded.check_request(build_request(guarded, "x" * 65536))
    assert not calls


@pytest.mark.parametrize("gbnf", [False, True])
def test_evidence_fit_includes_final_lm_defaults_and_adapter_grammar(provider, gbnf):
    url, calls = provider
    lm = create_dspy_lm("gemma-4-e4b" if gbnf else "reflection-model", base_url=url, api_key="test", cache=False)
    lm.kwargs["stop"] = ["tool options " * 100]
    prop = RLMInstructionProposer(
        worker_lm=lm,
        orchestrator_lm=lm,
        enable_gbnf=gbnf,
        enable_simba=False,
        max_instruction_chars=1500,
    )
    prop.preflight()
    prop.llm_proposer.config = {"temperature": 0.71}
    examples = [{"Feedback": "optional evidence " * 100}]
    from yaw.daemon.graph.inference.invariants import STAGE_SCHEMA_CONTRACTS

    mandatory = {"component": "qa.predict", "contract": STAGE_SCHEMA_CONTRACTS["qa"][2]}
    inputs = {
        "current_instruction": "Use evidence.",
        "examples_with_feedback": [mandatory, *examples],
        "length_limit": "The new instruction must be at most 1500 characters.",
    }
    options = dict(prop.llm_proposer.config)
    from yaw.daemon.graph.inference.adapter import YawGBNFAdapter

    assert isinstance(prop.json_adapter, YawGBNFAdapter)
    signature = prop.llm_proposer.signature
    assert signature is not None
    prop.json_adapter._prepare_gbnf_kwargs(prop.worker_lm, options, signature)
    prompt = prop.json_adapter.format(signature, prop.llm_proposer.demos, inputs)
    full_request = build_request(prop.worker_lm, prompt, options)
    assert ("grammar" in (full_request.config.extensions or {})) is gbnf
    prop.worker_lm.input_chars = len(json.dumps(asdict(full_request), ensure_ascii=False)) - 50
    result = prop.worker_module(
        current_instruction="Use evidence.", examples_with_feedback=examples, component_name="qa.predict"
    )
    assert result.new_instruction
    assert len(calls) == 1
    rendered = str(calls[0].get("prompt", calls[0].get("messages")))
    assert "optional evidence" not in rendered
    assert "Use evidence." in rendered


def test_native_rlm_binds_positional_tool_over_loopback(provider):
    from yaw.daemon.graph.inference.rlm import PositionalSafeRLM

    url, calls = provider
    lm = create_dspy_lm("reflection-model", base_url=url, api_key="test", cache=False)

    def double(value: int) -> int:
        """Double a number."""
        return value * 2

    rlm = PositionalSafeRLM("question -> answer", tools=[double], max_iters=1, max_llm_calls=1)
    with dspy.context(lm=lm):
        result = rlm(question="RLM_TOOL_CHECK")
    assert str(result.answer) == "42"
    assert calls


@pytest.mark.parametrize(
    ("model", "overrides", "expected"),
    [
        ("gemma-4-e4b", {}, (1.0, 1.0)),
        ("Qwen3-4B-Instruct-2507-GGUF", {}, (0.8, 0.7)),
        ("gemma-4-e4b", {"reflection_temperature": 0.6, "worker_temperature": 0.5}, (0.6, 0.5)),
    ],
)
def test_reflection_role_temperatures_reach_native_http(provider, model, overrides, expected):
    url, calls = provider
    orchestrator = create_dspy_lm(model, base_url=url, api_key="test", cache=False, role="reflection")
    worker = create_dspy_lm(model, base_url=url, api_key="test", cache=False, role="reflection_worker")
    prop = RLMInstructionProposer(
        orchestrator_lm=orchestrator, worker_lm=worker, enable_simba=False, enable_gbnf=False, **overrides
    )
    candidate = {"qa.predict": "Use evidence."}
    data = {"qa.predict": [{"Inputs": {"question": "Who?"}, "Feedback": "Expected Turing"}]}
    assert prop(candidate, data, ["qa.predict"])["qa.predict"]
    assert len(calls) >= 2
    assert all(call["temperature"] == expected[0] for call in calls[:-1])
    assert calls[-1]["temperature"] == expected[1]


def test_unbound_reflection_defaults_to_development_temperature():
    prop = RLMInstructionProposer(enable_simba=False)
    assert prop.inner_rlm.generate_action.config["temperature"] == 1.0
    assert prop.llm_proposer.config["temperature"] == 1.0
