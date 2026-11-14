"""Native embedding requests honor item limits without splitting whole inputs."""

import dspy
import numpy as np

from yaw.daemon.graph.inference import DSPyEmbeddingAdapter


def test_configured_batch_limits_reach_native_embedder() -> None:
    class Model:
        index_identity = "batch-budget-fixture"

        def __init__(self) -> None:
            self.batches: list[list[str]] = []

        def __call__(self, texts: list[str]) -> np.ndarray:
            self.batches.append(texts)
            return np.ones((len(texts), 768), dtype=np.float32)

    model = Model()
    adapter = DSPyEmbeddingAdapter(embedder=dspy.Embedder(model, batch_size=64, caching=False), max_batch_chars=100000)
    texts = ["x" * 1200] * 64
    adapter.batch_encode(texts, batch_size=16)
    assert [len(batch) for batch in model.batches] == [16] * 4
    assert all("x" * 1200 in text for batch in model.batches for text in batch)
