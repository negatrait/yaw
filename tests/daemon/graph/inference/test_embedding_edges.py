"""Native embedding input, failure and identity contracts."""

import asyncio
from dataclasses import replace

import numpy as np
import pytest

from yaw.daemon.graph.inference.embedding import DSPyEmbeddingAdapter, create_dspy_embedder, resolve_embedding_profile


class IdentifiedCallable:
    def __init__(self, fn, identity):
        self.fn = fn
        self.index_identity = identity

    def __call__(self, texts):
        return self.fn(texts)


def test_callable_requires_identity():
    with pytest.raises(ValueError, match="index_identity"):
        DSPyEmbeddingAdapter(embedder=create_dspy_embedder(lambda texts: []))


@pytest.mark.parametrize("asynchronous", [False, True])
def test_full_long_input_and_fail_fast(asynchronous):
    seen = []

    def embed(texts):
        seen.extend(texts)
        raise RuntimeError("maximum context length exceeded")

    adapter = DSPyEmbeddingAdapter(
        replace(resolve_embedding_profile(), dimensions=2),
        create_dspy_embedder(IdentifiedCallable(embed, "edge-fixture-v1")),
    )
    text = "long input " * 1000
    with pytest.raises(RuntimeError, match="maximum context"):
        if asynchronous:
            asyncio.run(adapter.acall(text))
        else:
            adapter.batch_encode(text)
    assert seen == [adapter.profile.document_prefix + text]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("vector", [[0.0, 0.0], [float("nan"), 1.0], [1.0]])
def test_invalid_vectors_fail(asynchronous, vector):
    def embed(texts):
        return np.array([vector for _ in texts])

    adapter = DSPyEmbeddingAdapter(
        replace(resolve_embedding_profile(), dimensions=2),
        create_dspy_embedder(IdentifiedCallable(embed, "edge-fixture-v1")),
    )
    with pytest.raises(ValueError):
        if asynchronous:
            asyncio.run(adapter.acall("input"))
        else:
            adapter.batch_encode("input")
