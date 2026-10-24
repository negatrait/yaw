"""Native callable embedding, async recovery and input-boundary invariants."""

import asyncio
from dataclasses import replace

import numpy as np
import pytest

from yaw.daemon.graph.inference.chat import set_yaw_sleep_seconds
from yaw.daemon.graph.inference.embedding import (
    FACT_INSTRUCTION,
    DSPyEmbeddingAdapter,
    create_dspy_embedder,
    pool_subchunk_embeddings,
    resolve_embedding_profile,
    split_text_gracefully,
)


@pytest.mark.parametrize("separator", ["\n\n", "\n", ". ", "; ", " ", ""])
def test_graceful_splits_preserve_every_character_without_overlap(separator):
    text = separator.join(f"token{i:03d}" for i in range(100))
    pieces = split_text_gracefully(text, max_chars=90, overlap_chars=0)
    assert "".join(pieces) == text
    assert all(0 < len(piece) <= 90 for piece in pieces)
    assert split_text_gracefully("") == [""]
    assert split_text_gracefully("short") == ["short"]


def test_pooling_zero_weights_and_single_vector_have_defined_normalization():
    vectors = np.array([[2.0, 0.0], [0.0, 4.0]], dtype=np.float32)
    np.testing.assert_allclose(pool_subchunk_embeddings(vectors, [0, 0], norm=False), [1.0, 2.0])
    np.testing.assert_allclose(pool_subchunk_embeddings(vectors[:1], [1]), [1.0, 0.0])


def test_profile_validates_raw_types_titles_and_modes():
    profile = resolve_embedding_profile(task_mode="titled")
    with pytest.raises(TypeError, match="must all be strings"):
        profile.format(None, query=False)
    assert profile.format("Heading\nbody", query=False, title="Heading") == "title: Heading | text: body"
    assert profile.format("body", query=False, title="Heading") == "title: Heading | text: body"
    with pytest.raises(ValueError, match="Unsupported embedding task mode"):
        resolve_embedding_profile(task_mode="unknown")
    for batch_size, max_batch_chars in ((0, 3500), (32, 0)):
        with pytest.raises(ValueError, match="must be positive"):
            DSPyEmbeddingAdapter(batch_size=batch_size, max_batch_chars=max_batch_chars)


@pytest.mark.parametrize("native", [False, True])
def test_sync_async_and_aliases_produce_identical_vectors_and_telemetry(native):
    profile = replace(resolve_embedding_profile(), dimensions=2)
    observed = []

    def embed(texts, **kwargs):
        observed.extend(texts)
        return np.tile(np.array([3.0, 4.0]), (len(texts), 1))

    adapter = DSPyEmbeddingAdapter(
        profile,
        create_dspy_embedder(embed, caching=False) if native else embed,
        max_batch_chars=200,
    )
    set_yaw_sleep_seconds(0.001)
    try:
        sync = adapter.forward(["first", "second"])
        asynchronous = asyncio.run(adapter.acall(["first", "second"]))
        np.testing.assert_allclose(sync, [[0.6, 0.8], [0.6, 0.8]])
        np.testing.assert_allclose(asynchronous, sync)
        np.testing.assert_allclose(adapter.encode(("first", "second")), sync)
        np.testing.assert_allclose(asyncio.run(adapter.as_async()(["first", "second"])), sync)
        np.testing.assert_allclose(asyncio.run(adapter.abatch_encode(["a" * 150, "b" * 150])), sync)
    finally:
        set_yaw_sleep_seconds(0)
    assert "first" in observed[0]
    assert adapter.telemetry[-1]["num_batches"] == 2
    for invalid in ([],):
        with pytest.raises(ValueError, match="cannot be empty"):
            adapter.batch_encode(invalid)
    with pytest.raises(ValueError, match="Unsupported retrieval instruction"):
        adapter.batch_encode("question", instruction="invented")


@pytest.mark.parametrize("asynchronous", [False, True])
def test_embedding_overflow_recovers_by_splitting_and_pooling_without_loss(asynchronous):
    accepted = []

    def limited_embed(texts, **kwargs):
        if len(texts) > 1 or len(texts[0]) > 700:
            raise RuntimeError("maximum context length exceeded")
        accepted.extend(texts)
        return np.array([3.0, 4.0])

    adapter = DSPyEmbeddingAdapter(replace(resolve_embedding_profile(), dimensions=2), limited_embed)
    text = "source sentence. " * 70
    result = asyncio.run(adapter.abatch_encode(text)) if asynchronous else adapter.batch_encode(text)
    np.testing.assert_allclose(result, [[0.6, 0.8]])
    assert len(accepted) >= 2
    assert all(len(piece) <= 700 for piece in accepted)
    assert all(f"{word}" in "".join(accepted) for word in ("source", "sentence"))


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("message", ["token limit", "backend unavailable"])
def test_embedding_failure_without_a_smaller_valid_request_propagates(asynchronous, message):
    def reject(texts, **kwargs):
        raise RuntimeError(message)

    adapter = DSPyEmbeddingAdapter(embedder=reject)
    with pytest.raises(RuntimeError, match=message):
        if asynchronous:
            asyncio.run(adapter.abatch_encode("short"))
        else:
            adapter.batch_encode("short")


def test_long_queries_and_untitled_documents_keep_every_piece_prefix():
    inputs = []

    def embed(texts):
        inputs.extend(texts)
        return np.ones((len(texts), 2))

    profile = replace(resolve_embedding_profile(task_mode="titled"), dimensions=2)
    adapter = DSPyEmbeddingAdapter(profile, create_dspy_embedder(embed, caching=False))
    text = "source word " * 400
    adapter.batch_encode(text, instruction=FACT_INSTRUCTION)
    assert len(inputs) > 1 and all(piece.startswith(profile.query_prefix) for piece in inputs)
    inputs.clear()
    adapter.batch_encode(text)
    assert len(inputs) > 1 and all(piece.startswith(profile.document_prefix) for piece in inputs)


def test_prefix_updates_change_index_identity_and_actual_requests():
    inputs = []

    def embed(texts):
        inputs.extend(texts)
        return np.ones((len(texts), 2))

    adapter = DSPyEmbeddingAdapter(
        replace(resolve_embedding_profile(), dimensions=2),
        create_dspy_embedder(embed, caching=False),
    )
    previous = adapter.profile.index_identity
    adapter.update_prefixes(query_prefix="query: ", fact_query_prefix="fact: ", document_prefix="document: ")
    assert adapter.profile.index_identity != previous
    adapter.batch_encode("document")
    adapter.batch_encode("question", instruction=FACT_INSTRUCTION)
    assert inputs == ["document: document", "fact: question"]
    unchanged = adapter.profile
    adapter.update_prefixes()
    assert adapter.profile is unchanged
