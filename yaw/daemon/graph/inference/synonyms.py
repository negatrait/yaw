"""Token-budget enforcement at the native DSPy adapter boundary."""

import hashlib
import json
from typing import Any

import dspy

from yaw.core import SynonymEvidenceBatch


class SynonymRequestBudget(dspy.Adapter):
    """Preserve native LM dispatch and check complete adapter-formatted requests."""

    def __init__(self, adapter: dspy.Adapter, counter: Any, context_tokens: int, output_tokens: int) -> None:
        super().__init__()
        self.adapter, self.counter = adapter, counter
        self.context_tokens = min(context_tokens, getattr(counter, "context_tokens", context_tokens))
        self.output_tokens = output_tokens
        self.last_key = ""
        self.last_count = 0

    def count(self, lm: Any, lm_kwargs: dict[str, Any], signature: Any, demos: list, inputs: dict) -> int:
        messages = self.adapter.format(signature, demos, inputs)
        kwargs = {**lm.kwargs, **lm_kwargs}
        key = hashlib.sha256(json.dumps([messages, kwargs], sort_keys=True, default=str).encode()).hexdigest()
        if key != self.last_key:
            self.last_count = self.counter(messages, **kwargs)
            self.last_key = key
        return self.last_count

    def fits(self, batch: SynonymEvidenceBatch, predictor: dspy.Predict, lm: Any) -> bool:
        count = self.count(lm, {}, predictor.signature, predictor.demos, {"candidate_pairs": batch.payload})
        output = int(lm.kwargs.get("max_completion_tokens") or lm.kwargs.get("max_tokens") or self.output_tokens)
        return count + output + 512 <= self.context_tokens

    def __call__(self, lm: Any, lm_kwargs: dict, signature: Any, demos: list, inputs: dict) -> list[dict[str, Any]]:
        count = self.count(lm, lm_kwargs, signature, demos, inputs)
        kwargs = {**lm.kwargs, **lm_kwargs}
        output = int(kwargs.get("max_completion_tokens") or kwargs.get("max_tokens") or self.output_tokens)
        if count + output + 512 > self.context_tokens:
            raise ValueError(
                f"Synonym request exceeds context: input={count}, output={output}, margin=512, "
                f"context={self.context_tokens}"
            )
        return self.adapter(lm, lm_kwargs, signature, demos, inputs)
