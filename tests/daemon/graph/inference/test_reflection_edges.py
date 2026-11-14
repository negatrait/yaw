"""Persistence-boundary and evidence provenance rejection contracts."""

import asyncio
import copy

import pytest
from dspy.clients.requests import build_request
from pydantic import BaseModel

from daemon.graph.inference.test_reflection import provider as provider
from yaw.daemon.graph.inference.chat import create_dspy_lm
from yaw.daemon.graph.inference.reflection import (
    ReflectionBudgetError,
    ReflectionLM,
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
        "predictors": {"llm_proposer": predictor, "llm_compressor": copy.deepcopy(predictor)},
    }


@pytest.mark.parametrize("defect", ["examples", "example-input", "signature", "field", "demo"])
def test_simba_checkpoint_rejects_malformed_nested_contracts(defect):
    state = simba_state()
    if defect == "examples":
        state["examples"] = "not-list"
    elif defect == "example-input":
        state["examples"][0]["component_name"] = None
    elif defect == "signature":
        state["predictors"]["llm_proposer"]["signature"]["instructions"] = None
    elif defect == "field":
        state["predictors"]["llm_proposer"]["signature"]["fields"][0]["description"] = 2
    else:
        state["predictors"]["llm_proposer"]["demos"] = ["not-object"]
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


def test_native_reflection_async_copy_and_state_preserve_character_bound(provider):
    url, calls = provider
    lm = create_dspy_lm("reflection-model", base_url=url, api_key="test", cache=False, max_tokens=128)
    guarded = ReflectionLM(lm, input_chars=65536)
    result = asyncio.run(guarded.acall(build_request(guarded, "Write a general instruction")))
    assert "new_instruction" in result.text
    assert len(calls) == 1
    assert calls[0].get("max_completion_tokens", calls[0].get("max_tokens")) == 128
    assert not guarded.history and not lm.history
    cloned = copy.deepcopy(guarded)
    assert cloned is not guarded
    assert cloned.input_chars == guarded.input_chars == 65536
    assert cloned.dump_state()["model"] == lm.model
    assert cloned.dump_state()["max_tokens"] == 128


def test_reflection_rejects_oversized_messages_before_any_provider_call(provider):
    url, calls = provider
    lm = create_dspy_lm("reflection-model", base_url=url, api_key="test", cache=False)
    guarded = ReflectionLM(lm, input_chars=100)
    with pytest.raises(ReflectionBudgetError, match="characters"):
        guarded(build_request(guarded, "x" * 100))
    assert calls == []


@pytest.mark.parametrize("instruction", ["", "  ", 42])
def test_completed_checkpoint_rejects_empty_or_nontext_instructions(instruction):
    with pytest.raises(ValueError, match="completed proposal instruction"):
        validate_request_checkpoint({"phase": "complete", "state": simba_state(), "proposals": {"ner": instruction}})
