"""Evidence provenance and exact serving-token budgets for GEPA reflection."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any, TypedDict

import dspy
from dspy.clients.requests import build_request


def reflection_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, default=str, allow_nan=False).encode("utf-8")
    ).hexdigest()


class ReflectionBudgetError(ValueError):
    """A reflection request cannot safely be sent to the configured model."""


class SimbaState(TypedDict):
    version: int
    steps: int
    examples: list[dict[str, Any]]
    predictors: dict[str, dict[str, Any]]


class ActiveReflectionCheckpoint(TypedDict):
    context: str
    request: str
    records: dict[str, dict[str, Any]]


def checkpoint_object(value: object, name: str) -> dict[str, Any]:
    """Validate a JSON object at the persistence boundary, retaining native fields."""
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"Malformed {name}: expected an object with string keys")
    return dict(value)


def validate_simba_state(value: object) -> SimbaState:
    state = checkpoint_object(value, "SIMBA state")
    steps = state.get("steps")
    if type(state.get("version")) is not int or state["version"] != 2:
        raise ValueError("Unsupported SIMBA state version")
    if type(steps) is not int or steps < 0:
        raise ValueError("Malformed SIMBA state: steps must be a non-negative integer")
    examples = state.get("examples")
    if not isinstance(examples, list):
        raise ValueError("Malformed SIMBA state: examples must be a list")
    checked_examples = []
    for example in examples:
        example = checkpoint_object(example, "SIMBA training example")
        if (
            not isinstance(example.get("component_name"), str)
            or not isinstance(example.get("current_instruction"), str)
            or not isinstance(example.get("examples_with_feedback"), list)
        ):
            raise ValueError("Malformed SIMBA training example inputs")
        checked_examples.append(example)
    predictors = checkpoint_object(state.get("predictors"), "SIMBA predictors")
    checked_predictors = {}
    for name in ("llm_proposer", "llm_compressor"):
        predictor = checkpoint_object(predictors.get(name), f"SIMBA predictor {name}")
        signature = checkpoint_object(predictor.get("signature"), "SIMBA signature")
        if (
            not isinstance(signature.get("instructions"), str)
            or not isinstance(signature.get("fields"), list)
            or not isinstance(predictor.get("demos"), list)
        ):
            raise ValueError(f"Malformed SIMBA predictor {name}")
        for field in signature["fields"]:
            field = checkpoint_object(field, "SIMBA signature field")
            if not all(isinstance(field.get(key), str) for key in ("prefix", "description")):
                raise ValueError("Malformed SIMBA signature field")
        for demo in predictor["demos"]:
            checkpoint_object(demo, "SIMBA demonstration")
        checked_predictors[name] = predictor
    return {"version": 2, "steps": steps, "examples": checked_examples, "predictors": checked_predictors}


def validate_request_checkpoint(value: object) -> dict[str, Any]:
    record = checkpoint_object(value, "reflection request")
    if record.get("phase") not in ("collecting", "learning", "learned", "complete"):
        raise ValueError("Malformed reflection request phase")
    record["state"] = validate_simba_state(record.get("state"))
    if "writer" in record and not isinstance(record["writer"], str):
        raise ValueError("Malformed reflection writer identity")
    if "writer" in record and record["writer"] != reflection_digest(record["state"]["predictors"]):
        raise ValueError("Reflection writer identity does not match its saved state")
    if record["phase"] == "complete":
        proposals = checkpoint_object(record.get("proposals"), "completed proposals")
        if any(not isinstance(instruction, str) or not instruction.strip() for instruction in proposals.values()):
            raise ValueError("Malformed completed proposal instruction")
    return record


def validate_active_checkpoint(value: object, context: str) -> ActiveReflectionCheckpoint:
    active = checkpoint_object(value, "active reflection checkpoint")
    if active.get("context") != context:
        raise ValueError("Reflection checkpoint does not match this run's task/model/data")
    request = active.get("request")
    records = checkpoint_object(active.get("records"), "active reflection records")
    if not isinstance(request, str) or not request.startswith("request:") or request not in records:
        raise ValueError("Malformed active reflection request identity")
    checked = {key: checkpoint_object(record, "reflection record") for key, record in records.items()}
    checked[request] = validate_request_checkpoint(checked[request])
    return {"context": context, "request": request, "records": checked}


class ReflectionLM(dspy.BaseLM):
    """Bound complete reflection requests by characters; the provider owns token overflow."""

    def __init__(self, target: Any, input_chars: int = 65536) -> None:
        super().__init__(model=target.model)
        self.kwargs = dict(target.kwargs)
        if input_chars < 1:
            raise ValueError("reflection_input_chars must be positive")
        self.target = target
        self.input_chars = input_chars
        self._yaw_context_window_tokens = getattr(target, "_yaw_context_window_tokens", None)
        self._yaw_supports_gbnf = getattr(target, "_yaw_supports_gbnf", False)

    def _build_adapter_request(self, prompt: Any, options: dict[str, Any] | None) -> dspy.lm15.Request:
        return build_request(self.target, prompt, options)

    def check_request(self, request: dspy.lm15.Request) -> None:
        count = len(json.dumps(asdict(request), ensure_ascii=False))
        if count > self.input_chars:
            raise ReflectionBudgetError(f"Reflection request has {count} characters; limit={self.input_chars}")

    def __call__(self, prompt: dspy.lm15.Request | str | None = None, **kwargs: Any) -> Any:
        if isinstance(prompt, str):
            options = {k: v for k, v in kwargs.items() if k not in ("cache", "rollout_id")}
            request = build_request(self.target, prompt, options)
        elif isinstance(prompt, dspy.lm15.Request):
            request = prompt
        else:
            raise TypeError("Reflection requires a native request or text prompt")
        self.check_request(request)
        with dspy.context(disable_history=True):
            return self.target(prompt, **kwargs)

    async def acall(self, prompt: dspy.lm15.Request | str | None = None, **kwargs: Any) -> Any:
        if isinstance(prompt, str):
            options = {k: v for k, v in kwargs.items() if k not in ("cache", "rollout_id")}
            request = build_request(self.target, prompt, options)
        elif isinstance(prompt, dspy.lm15.Request):
            request = prompt
        else:
            raise TypeError("Reflection requires a native request or text prompt")
        self.check_request(request)
        with dspy.context(disable_history=True):
            return await self.target.acall(prompt, **kwargs)

    def copy(self, **kwargs: Any) -> Any:
        return ReflectionLM(self.target.copy(**kwargs), self.input_chars)

    def __deepcopy__(self, memo: dict[int, Any]) -> Any:
        result = self.copy()
        memo[id(self)] = result
        return result

    def dump_state(self) -> dict[str, Any]:
        return self.target.dump_state()


def reflection_examples(examples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Retain task facts and derive content-addressed example identities."""
    result = []
    for example in examples:
        value = dict(example)
        value.pop("evidence_id", None)
        # DSPy histories expose structured values without lossy string-prefix clipping.
        value = json.loads(
            json.dumps(
                value,
                default=lambda item: (
                    item.toDict()
                    if hasattr(item, "toDict")
                    else item.model_dump()
                    if hasattr(item, "model_dump")
                    else str(item)
                ),
                ensure_ascii=False,
            )
        )
        result.append({"evidence_id": reflection_digest(value), **value})
    return result


def resolve_evidence(
    examples: Sequence[Mapping[str, Any]], selection: Mapping[str, Any], limit: int = 5
) -> list[dict[str, Any]]:
    """Resolve IDs/paths/spans in host code; no model-written facts are accepted."""
    selection = checkpoint_object(selection, "evidence selection")
    if not isinstance(selection.get("diagnosis", ""), str):
        raise ValueError("Malformed evidence diagnosis")
    references = selection["references"]
    if not isinstance(references, list) or not 0 < len(references) <= limit:
        raise ValueError("Evidence must contain one to five references")
    originals = {ex["evidence_id"]: ex for ex in examples}
    result = []
    for ref in references:
        ref = checkpoint_object(ref, "evidence reference")
        if not isinstance(ref.get("example_id"), str):
            raise ValueError("Malformed evidence example identity")
        original = originals[ref["example_id"]]
        value = original
        path = ref.get("path", [])
        if not isinstance(path, list):
            raise ValueError("Evidence path must be a list")
        for key in path:
            if (
                isinstance(value, Mapping)
                and isinstance(key, str)
                or isinstance(value, (list, tuple))
                and type(key) is int
                and 0 <= key < len(value)
            ):
                value = value[key]
            else:
                raise ValueError("Invalid evidence field path")
        if "start" in ref or "end" in ref:
            start, end = ref["start"], ref["end"]
            if (
                not isinstance(value, str)
                or type(start) is not int
                or type(end) is not int
                or not 0 <= start < end <= len(value)
            ):
                raise ValueError("Invalid evidence span")
            value = value[start:end]
        result.append({"reference": dict(ref), "evidence": value})
    return result
