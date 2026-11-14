"""Measured token confidence for source extraction and semantic decisions."""

import json
import math
import re
from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from typing import Any

from dspy.lm15 import TokenLogprob

_JUDGE_JSON = re.compile(rb'"(?:[^"\\]|\\.)*"|true|false|[{}:]')
_VOTE_FIELDS = {"equivalent", "hidden_label", "broader", "narrower", "related"}
_JSON_TOKENS = re.compile(rb'"(?:[^"\\]|\\.)*"|true|false|null|[{}\[\]:,]|-?\d+(?:\.\d+)?')


class LogprobMeasurement:
    """Reconstruct one response once; missing measurements stay unknown."""

    def __init__(self, logprobs: Sequence[TokenLogprob] | None) -> None:
        self.content = logprobs
        self.text: bytes | None = None
        self.ends: list[int] = []
        if not self.content or any(token.bytes is None for token in self.content):
            return
        parts = [bytes(token.bytes) for token in self.content if token.bytes is not None]
        offset = 0
        for part in parts:
            offset += len(part)
            self.ends.append(offset)
        self.text = b"".join(parts)

    def probability(self, fields: Sequence[tuple[int, int]], *, normalize_length: bool = False) -> float | None:
        if self.text is None or self.content is None or not fields:
            return None
        total = 0.0
        for start, end in fields:
            first, last = bisect_right(self.ends, start), bisect_left(self.ends, end)
            value_total = 0.0
            for index in range(first, last + 1):
                value = self.content[index].logprob
                if not math.isfinite(value) or value > 0:
                    raise ValueError("Invalid decision token logprob")
                value_total += value
            total += value_total / (last - first + 1) if normalize_length else value_total
        return math.exp(total / len(fields))


def judge_logprob_votes(logprobs: Sequence[TokenLogprob] | None, decisions: list[dict[str, Any]]) -> None:
    """Measure decision values, never reasoning or structural syntax."""
    for decision in decisions:
        decision["llm_prob"] = None
    measurement = LogprobMeasurement(logprobs)
    if measurement.text is None:
        return
    stack: list[dict[str, Any]] = []
    spans: dict[str, tuple[list[tuple[int, int]], dict[str, Any]]] = {}
    key = ""
    last_string = b'""'
    for match in _JUDGE_JSON.finditer(measurement.text):
        value = match.group()
        if value == b"{":
            stack.append({"pair_id": None, "spans": [], "values": {}})
        elif value == b"}":
            entry = stack.pop()
            if entry["pair_id"] is not None:
                if entry["pair_id"] in spans:
                    raise ValueError("Duplicate pair in judge logprobs")
                spans[entry["pair_id"]] = entry["spans"], entry["values"]
            key = ""
        elif value == b":":
            key = json.loads(last_string)
        elif value.startswith(b'"'):
            if key == "pair_id":
                if stack[-1]["pair_id"] is not None:
                    raise ValueError("Duplicate pair identifier in judge logprobs")
                stack[-1]["pair_id"] = json.loads(value)
            elif key == "relation":
                if key in stack[-1]["values"]:
                    raise ValueError("Duplicate decision field in judge logprobs")
                stack[-1]["spans"].append((match.start() + 1, match.end() - 1))
                stack[-1]["values"][key] = json.loads(value)
            last_string = value
            key = ""
        else:
            if key in _VOTE_FIELDS:
                if key in stack[-1]["values"]:
                    raise ValueError("Duplicate decision field in judge logprobs")
                stack[-1]["spans"].append(match.span())
                stack[-1]["values"][key] = value == b"true"
            key = ""
    if stack or set(spans) != {decision["pair_id"] for decision in decisions}:
        raise ValueError("Judge logprobs do not cover the parsed decision pairs")
    for decision in decisions:
        fields, values = spans[decision["pair_id"]]
        expected = {key: decision[key] for key in _VOTE_FIELDS if key in decision}
        if not expected and "relation" in decision:
            expected["relation"] = decision["relation"]
        if decision.get("equivalent") is False:
            expected.pop("hidden_label", None)
        if values != expected:
            raise ValueError("Judge logprobs do not match the parsed decision values")
        if not fields:
            raise ValueError("Judge logprobs contain no decision values")
        decision["llm_prob"] = measurement.probability(fields)


def _unique_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate extraction field in logprobs")
        result[key] = value
    return result


def extraction_logprob_votes(
    logprobs: Sequence[TokenLogprob] | None, field: str, records: Sequence[Any]
) -> tuple[float | None, ...]:
    """Measure each source record's value strings, validating the exact decoded records."""
    measurement = LogprobMeasurement(logprobs)
    if measurement.text is None:
        return (None,) * len(records)
    tokens = list(_JSON_TOKENS.finditer(measurement.text))
    field_token = json.dumps(field).encode()
    index = next(
        (
            i + 3
            for i in range(len(tokens) - 2)
            if tokens[i].group() == field_token and tokens[i + 1].group() == b":" and tokens[i + 2].group() == b"["
        ),
        None,
    )
    if index is None:
        raise ValueError("Extraction logprobs do not contain the output field")
    probabilities = []
    while index < len(tokens) and tokens[index].group() != b"]":
        start = index
        depth = 0
        while index < len(tokens):
            value = tokens[index].group()
            depth += (value in (b"[", b"{")) - (value in (b"]", b"}"))
            index += 1
            if depth == 0:
                break
        raw = measurement.text[tokens[start].start() : tokens[index - 1].end()]
        position = len(probabilities)
        if position >= len(records) or json.dumps(
            json.loads(raw, object_pairs_hook=_unique_fields), sort_keys=True
        ) != json.dumps(records[position], sort_keys=True):
            raise ValueError("Extraction logprobs do not match the parsed records")
        fields = [
            (tokens[i].start() + 1, tokens[i].end() - 1)
            for i in range(start, index)
            if tokens[i].group().startswith(b'"')
            and len(tokens[i].group()) > 2
            and (i + 1 == index or tokens[i + 1].group() != b":")
        ]
        probabilities.append(measurement.probability(fields, normalize_length=True))
        if index < len(tokens) and tokens[index].group() == b",":
            index += 1
    if len(probabilities) != len(records) or index == len(tokens):
        raise ValueError("Extraction logprobs do not cover the parsed records")
    return tuple(probabilities)
