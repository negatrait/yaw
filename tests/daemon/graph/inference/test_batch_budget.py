"""Configured embedding request budgets operate independently of item counts."""

from pathlib import Path

from yaw.core import YawConfig
from yaw.daemon.graph.inference import DSPyEmbeddingAdapter


def test_configured_batch_limits_reach_the_actual_request_partitioner(tmp_path: Path) -> None:
    adapter = DSPyEmbeddingAdapter.from_config(
        YawConfig(
            workspace_root=tmp_path,
            embed_batch_size=64,
            embed_batch_chars=20000,
        )
    )
    texts = ["A short input " * 8] * 64
    assert len(adapter._prepare_pieces_and_batches(texts)[-1]) == 1
    assert len(adapter._prepare_pieces_and_batches(texts, batch_size=16)[-1]) == 4
    adapter.max_batch_chars = 3500
    requests = adapter._prepare_pieces_and_batches(["x" * 1200] * 64)[-1]
    assert len(requests) == 32
    assert max(map(len, requests)) == 2
