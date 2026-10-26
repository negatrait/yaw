"""Persistence-boundary and evidence provenance rejection contracts."""

import asyncio
import copy
import json
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import dspy
import pytest
from pydantic import BaseModel

from daemon.graph.inference.test_reflection import provider as provider
from yaw.daemon.graph.inference.reflection import (
    ReflectionBudgetError,
    ReflectionLM,
    ServingTokenCounter,
    deterministic_evidence,
    reflection_digest,
    reflection_examples,
    resolve_evidence,
    validate_active_checkpoint,
    validate_request_checkpoint,
    validate_simba_state,
)


def simba_state():
    predictor = {
        "signature": {"instructions": "Preserve schema", "fields": [{"prefix": "Input:", "description": "Evidence"}]},
        "demos": [],
    }
    return {
        "version": 2,
        "steps": 0,
        "examples": [{"component_name": "ner", "current_instruction": "Extract", "examples_with_feedback": []}],
        "predictors": {"qwen_proposer": predictor, "qwen_compressor": copy.deepcopy(predictor)},
    }


@pytest.mark.parametrize("defect", ["examples", "example-input", "signature", "field", "demo"])
def test_simba_checkpoint_rejects_malformed_nested_contracts(defect):
    state = simba_state()
    if defect == "examples":
        state["examples"] = "not-list"
    elif defect == "example-input":
        state["examples"][0]["component_name"] = None
    elif defect == "signature":
        state["predictors"]["qwen_proposer"]["signature"]["instructions"] = None
    elif defect == "field":
        state["predictors"]["qwen_proposer"]["signature"]["fields"][0]["description"] = 2
    else:
        state["predictors"]["qwen_proposer"]["demos"] = ["not-object"]
    with pytest.raises(ValueError, match="Malformed"):
        validate_simba_state(state)


def test_checkpoint_writer_identity_and_active_request_are_validated():
    state = simba_state()
    request = {"phase": "collecting", "state": state, "writer": reflection_digest(state["predictors"])}
    assert validate_request_checkpoint(request)["writer"] == request["writer"]
    for writer in (42, "wrong-hash"):
        with pytest.raises(ValueError, match="writer identity"):
            validate_request_checkpoint({**request, "writer": writer})
    active = {"context": "context", "request": "request:one", "records": {"request:one": request}}
    assert validate_active_checkpoint(active, "context")["request"] == "request:one"
    with pytest.raises(ValueError, match="does not match this run"):
        validate_active_checkpoint(active, "different-run")
    with pytest.raises(ValueError, match="request identity"):
        validate_active_checkpoint({**active, "request": "missing"}, "context")


def test_deterministic_selector_handles_nested_malformed_and_nonfinite_scores():
    examples = reflection_examples(
        [
            {"Feedback": {"score": 1.0}},
            {"Score": "invalid"},
            {"Score": "nan"},
            {"Score": {}},
            {"Feedback": "score=0.8"},
        ]
    )
    selection = deterministic_evidence(examples)
    resolved = resolve_evidence(examples, selection)
    assert len(resolved) == 5
    assert resolved[-2]["evidence"]["Feedback"]["score"] == 1.0
    assert resolved[-1]["evidence"]["Feedback"] == "score=0.8"


def test_reflection_examples_serialize_native_schema_values_and_keep_stable_ids():
    class Payload(BaseModel):
        answer: str

    examples = reflection_examples([{"output": Payload(answer="Source"), "evidence_id": "stale"}])
    assert examples[0]["output"] == {"answer": "Source"}
    assert examples[0]["evidence_id"] != "stale"
    assert reflection_examples(examples) == examples


@pytest.mark.parametrize(
    "change",
    [
        {"example_id": 1},
        {"path": "input"},
        {"path": ["items", -1]},
        {"path": ["items", True]},
        {"path": ["items", 2]},
        {"path": ["items", 0, 0]},
    ],
)
def test_evidence_references_cannot_escape_original_structure(change):
    examples = reflection_examples([{"items": ["source"]}])
    reference = {"example_id": examples[0]["evidence_id"], **change}
    with pytest.raises(ValueError, match="evidence|Evidence"):
        resolve_evidence(examples, {"references": [reference]})


def test_evidence_diagnosis_and_reference_count_have_explicit_contracts():
    examples = reflection_examples([{"items": ["source"]}])
    with pytest.raises(ValueError, match="diagnosis"):
        resolve_evidence(examples, {"diagnosis": 2, "references": []})
    for references in ([], "not-list"):
        with pytest.raises(ValueError, match="one to five"):
            resolve_evidence(examples, {"references": references})


def test_native_reflection_async_copy_and_state_use_the_verified_serving_counter(provider):
    url, calls = provider
    counter = ServingTokenCounter(url, "reflection-model", backend_url=url)
    lm = dspy.LM("openai/reflection-model", api_base=url, api_key="test", cache=False, max_tokens=128)
    guarded = ReflectionLM(lm, counter, input_tokens=16384, context_tokens=65536, output_tokens=128)
    result = asyncio.run(guarded.acall(prompt="Write a general instruction"))
    assert "new_instruction" in result[0]
    assert len(calls) == 1
    assert calls[0].get("max_completion_tokens", calls[0].get("max_tokens")) == 128
    assert not guarded.history and not lm.history
    cloned = copy.deepcopy(guarded)
    assert cloned is not guarded
    assert cloned.counter is counter
    assert cloned.dump_state()["model"] == lm.model
    assert cloned.dump_state()["max_tokens"] == 128


def test_tokenizer_rejects_nonresident_model_and_actual_missing_backend_route(provider):
    url, _calls = provider
    with pytest.raises(ReflectionBudgetError, match="not resident"):
        ServingTokenCounter(url, "absent-model")
    with pytest.raises(ReflectionBudgetError, match="Serving tokenizer unavailable"):
        ServingTokenCounter(url, "reflection-model", backend_url=url + "/not-a-tokenizer")


@pytest.mark.parametrize("instruction", ["", "  ", 42])
def test_completed_checkpoint_rejects_empty_or_nontext_instructions(instruction):
    with pytest.raises(ValueError, match="completed proposal instruction"):
        validate_request_checkpoint({"phase": "complete", "state": simba_state(), "proposals": {"ner": instruction}})


def test_tokenizer_rejects_unadvertised_url_and_invalid_props_from_real_http_files(tmp_path):
    (tmp_path / "health").write_text(
        json.dumps(
            {
                "all_models_loaded": [{"id": "reflection-model", "launch_command": ["server", "-m", "weights.gguf"]}],
            }
        ),
        encoding="utf-8",
    )
    props = tmp_path / "props"
    props.write_text(
        json.dumps({"model_path": "weights.gguf", "default_generation_settings": {"n_ctx": 0}}), encoding="utf-8"
    )
    handler = partial(SimpleHTTPRequestHandler, directory=str(tmp_path))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        with pytest.raises(ReflectionBudgetError, match="No serving tokenizer URL"):
            ServingTokenCounter(url, "reflection-model")
        with pytest.raises(ReflectionBudgetError, match="actual context window"):
            ServingTokenCounter(url, "reflection-model", backend_url=url)
        props.write_text(
            json.dumps(
                {
                    "model_path": "different-weights.gguf",
                    "default_generation_settings": {"n_ctx": 65536},
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ReflectionBudgetError, match="weights do not match"):
            ServingTokenCounter(url, "reflection-model", backend_url=url)
        props.write_text("malformed JSON", encoding="utf-8")
        with pytest.raises(ReflectionBudgetError, match="Serving tokenizer unavailable"):
            ServingTokenCounter(url, "reflection-model", backend_url=url)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
