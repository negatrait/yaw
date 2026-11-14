"""Embedding profile request-format and vector-identity contracts."""

import unittest
from dataclasses import replace

import numpy as np

from yaw.daemon.graph.inference.embedding import (
    GEMMA_TASK_MODES,
    MODEL,
    PASSAGE_INSTRUCTION,
    PROFILES,
    DSPyEmbeddingAdapter,
    create_dspy_embedder,
    resolve_embedding_profile,
)


class ProfileTests(unittest.TestCase):
    def test_each_profile_preserves_its_request_format_and_dimensions(self):
        for profile in PROFILES.values():
            with self.subTest(model=profile.model):
                recorded_inputs: list[list[str]] = []

                class ProfileEmbedder:
                    def __init__(self, dims: int, sink: list[list[str]]):
                        self.dims = dims
                        self.sink = sink
                        self.index_identity = f"profile-fixture-{dims}"

                    def __call__(self, texts: list[str]) -> np.ndarray:
                        self.sink.append(list(texts))
                        return np.ones((len(texts), self.dims), dtype=np.float32)

                adapter = DSPyEmbeddingAdapter(
                    profile=profile,
                    embedder=create_dspy_embedder(
                        ProfileEmbedder(profile.dimensions, recorded_inputs), batch_size=2, caching=False
                    ),
                )
                adapter.batch_encode(["a document"])
                self.assertEqual(recorded_inputs[-1], [profile.document_prefix + "a document"])

                result = adapter.batch_encode("who?", instruction=PASSAGE_INSTRUCTION)
                self.assertEqual(recorded_inputs[-1], [profile.query_prefix + "who?"])
                self.assertEqual(result.shape, (1, profile.dimensions))

                if profile.model.startswith("Qwen"):
                    self.assertIn("\nQuery:", recorded_inputs[-1][0])
                    bad_adapter = DSPyEmbeddingAdapter(
                        profile=profile,
                        embedder=create_dspy_embedder(
                            ProfileEmbedder(768, recorded_inputs), batch_size=2, caching=False
                        ),
                    )
                    with self.assertRaises(ValueError):
                        bad_adapter.batch_encode(["wrong dimensions"])

    def test_index_identity_changes_only_for_vector_producing_settings(self):
        self.assertEqual(len({p.index_identity for p in PROFILES.values()}), 3)
        profile = next(iter(PROFILES.values()))
        for changed in (
            replace(profile, document_prefix="new document: "),
            replace(profile, dimensions=256),
            replace(profile, pooling="last"),
        ):
            self.assertNotEqual(profile.index_identity, changed.index_identity)
        self.assertEqual(profile.index_identity, replace(profile, query_prefix="new query: ").index_identity)
        mode_identities = {resolve_embedding_profile(MODEL, mode).index_identity for mode in GEMMA_TASK_MODES}
        self.assertEqual(len(mode_identities), 2)
        with self.assertRaises(ValueError):
            resolve_embedding_profile("Qwen3-Embedding-0.6B-GGUF", "qa")


if __name__ == "__main__":
    unittest.main()
