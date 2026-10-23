"""Evidence provenance and exact serving-token budgets for GEPA reflection."""

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, TypedDict
from urllib.error import URLError
from urllib.request import Request, urlopen

import dspy


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
    for name in ("qwen_proposer", "qwen_compressor"):
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


class ServingTokenCounter:
    """Use the serving model's template and tokenizer, never a surrogate tokenizer.

    Construction only inspects an already running backend; it never loads a model.
    The backend URL must be supplied explicitly or advertised by Lemonade health.
    """

    def __init__(self, base_url: str, model: str, backend_url: str = "") -> None:
        self.base_url = base_url
        self.model = model.removeprefix("openai/")
        health = self.request(base_url.rstrip("/") + "/health")
        loaded = health.get("all_models_loaded", [])
        match = next(
            (item for item in loaded if (item.get("id", item.get("model_name")) == self.model)),
            None,
        )
        if match is None:
            raise ReflectionBudgetError(
                f"Reflection model {self.model!r} is not resident; start the external service first"
            )
        self.url = (backend_url or match.get("backend_url", "")).rstrip("/")
        if self.url.endswith("/v1"):
            self.url = self.url[:-3]
        if not self.url:
            raise ReflectionBudgetError("No serving tokenizer URL advertised; set inference.reflection_tokenizer_url")
        props = self.request(self.url + "/props")
        model_path = str(props.get("model_path", ""))
        command = match.get("launch_command", [])
        weights = command[command.index("-m") + 1] if "-m" in command else ""
        if not weights or not model_path or model_path.replace("\\", "/") != str(weights).replace("\\", "/"):
            raise ReflectionBudgetError("Tokenizer backend weights do not match the selected model")
        self.context_tokens = int(props.get("default_generation_settings", {}).get("n_ctx", 0))
        if self.context_tokens <= 0:
            raise ReflectionBudgetError("Serving backend did not report its actual context window")
        self.identity = reflection_digest(
            {
                "model": self.model,
                "checkpoint": match.get("checkpoint"),
                "weights": model_path,
                "template": props.get("chat_template"),
                "context": self.context_tokens,
                "generation_defaults": props.get("default_generation_settings"),
            }
        )
        self([{"role": "user", "content": "Tokenizer preflight."}])

    def for_model(self, model: str) -> ServingTokenCounter:
        """Each reflection role must use its own verified serving tokenizer."""
        if model.removeprefix("openai/") == self.model:
            return self
        return ServingTokenCounter(self.base_url, model)

    @staticmethod
    def request(url: str, payload: Any = None) -> Any:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        try:
            with urlopen(Request(url, data=data, headers={"Content-Type": "application/json"}), timeout=10) as response:
                return json.load(response)
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ReflectionBudgetError(f"Serving tokenizer unavailable at {url}: {exc}") from exc

    def __call__(self, messages: list[dict[str, Any]], **kwargs: Any) -> int:
        payload = {"messages": messages, "add_generation_prompt": True}
        merged = {**kwargs, **kwargs.get("extra_body", {})}
        for key in ("tools", "tool_choice", "chat_template_kwargs"):
            if key in merged:
                payload[key] = merged[key]
        rendered = self.request(self.url + "/apply-template", payload)["prompt"]
        return len(
            self.request(
                self.url + "/tokenize",
                {
                    "content": rendered,
                    "add_special": False,
                    "parse_special": True,
                },
            )["tokens"]
        )


class ReflectionLM(dspy.BaseLM):
    """Guard the final messages of every reflection call, including SIMBA advice."""

    def __init__(
        self,
        target: Any,
        counter: Any,
        input_tokens: int = 16384,
        context_tokens: int = 65536,
        output_tokens: int = 4096,
    ) -> None:
        super().__init__(model=target.model, **target.kwargs)
        self.kwargs["max_tokens"] = self.kwargs.get("max_tokens") or output_tokens
        self.target = target
        counter = counter.for_model(target.model) if isinstance(counter, ServingTokenCounter) else counter
        self.counter = counter
        self.input_tokens = input_tokens
        self.context_tokens = min(context_tokens, getattr(counter, "context_tokens", 0) or context_tokens)
        self.output_tokens = output_tokens
        self._yaw_context_window_tokens = self.context_tokens
        self._yaw_supports_gbnf = getattr(target, "_yaw_supports_gbnf", False)

    def count(self, messages: list[dict[str, Any]], **kwargs: Any) -> int:
        request = {**self.kwargs, **kwargs}
        count = self.counter(messages, **request)
        output = int(request.get("max_completion_tokens") or request.get("max_tokens") or self.output_tokens)
        if count > self.input_tokens or count + output + 512 > self.context_tokens:
            raise ReflectionBudgetError(
                f"Reflection prompt has {count} tokens; input limit={self.input_tokens}, "
                f"context={self.context_tokens}, output={output}, margin=512"
            )
        return count

    def __call__(self, prompt: Any = None, messages: Any = None, **kwargs: Any) -> Any:
        messages = messages if messages is not None else [{"role": "user", "content": prompt}]
        self.count(messages, **kwargs)
        request = self.request_kwargs(kwargs)
        with dspy.context(disable_history=True):
            return self.target(messages=messages, **request)

    async def acall(self, prompt: Any = None, messages: Any = None, **kwargs: Any) -> Any:
        messages = messages if messages is not None else [{"role": "user", "content": prompt}]
        self.count(messages, **kwargs)
        request = self.request_kwargs(kwargs)
        with dspy.context(disable_history=True):
            return await self.target.acall(messages=messages, **request)

    def request_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        request = {**self.kwargs, **kwargs}
        identity = getattr(self.counter, "identity", None)
        if identity is not None:
            # DSPy's native rollout ID participates in its disk request key but
            # is not sent as a provider parameter. Preserve rollout diversity
            # while invalidating responses when serving weights/defaults change.
            request["rollout_id"] = int(reflection_digest([identity, request.get("rollout_id")])[:15], 16)
        return request

    def copy(self, **kwargs: Any) -> Any:
        return ReflectionLM(
            self.target.copy(**kwargs),
            self.counter,
            self.input_tokens,
            self.context_tokens,
            self.output_tokens,
        )

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


def deterministic_evidence(examples: Sequence[Mapping[str, Any]], limit: int = 5) -> dict[str, Any]:
    def score(example: Mapping[str, Any]) -> float:
        value = example.get("Score", example.get("score"))
        feedback = example.get("Feedback", "")
        if value is None and isinstance(feedback, Mapping):
            value = feedback.get("score")
        if value is None:
            match = re.search(r"(?:GM_good/GM_bad|score)\s*=\s*([0-9.eE+-]+)", str(feedback))
            value = match.group(1) if match else 0
        try:
            result = float(value)
            return result if math.isfinite(result) else 0.0
        except ValueError, TypeError:
            return 0.0

    ordered = sorted(examples, key=lambda example: (score(example), example["evidence_id"]))
    failures = min(3, limit, len(ordered))
    chosen = ordered[:failures]
    chosen += [ex for ex in reversed(ordered[failures:])][: limit - failures]
    return {
        "diagnosis": "Contrast lower-scoring failures with higher-scoring examples.",
        "references": [{"example_id": ex["evidence_id"], "path": []} for ex in chosen],
    }


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
