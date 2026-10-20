"""Native DSPy embedding adapter and profile formatting tests."""

import unittest
from typing import Any

import numpy as np

import yaw.daemon.graph.inference as inference_facade
import yaw.daemon.graph.inference.embedding as embedding_module
from yaw.daemon.graph.inference.embedding import (
    DOCUMENT_PREFIX,
    FACT_CHECK_PREFIX,
    FACT_INSTRUCTION,
    MODEL,
    PASSAGE_INSTRUCTION,
    QA_QUERY_PREFIX,
    QUERY_PREFIX,
    DSPyEmbeddingAdapter,
    create_dspy_embedder,
    resolve_embedding_profile,
)


class EmbeddingTests(unittest.TestCase):
    def test_titled_and_distinct_titled_modes_format_multiline_passages_and_preserve_single_line_inputs(self):
        base_profile = resolve_embedding_profile(MODEL, "search")
        titled_profile = resolve_embedding_profile(MODEL, "titled")
        distinct_titled_profile = resolve_embedding_profile(MODEL, "distinct-titled")

        self.assertFalse(base_profile.parse_document_title)
        self.assertTrue(titled_profile.parse_document_title)
        self.assertTrue(distinct_titled_profile.parse_document_title)
        self.assertEqual(titled_profile.query_instruction_mode, "shared")
        self.assertEqual(distinct_titled_profile.query_instruction_mode, "distinct")
        self.assertEqual(
            len({base_profile.index_identity, titled_profile.index_identity, distinct_titled_profile.index_identity}),
            3,
        )

        # Explicit Title: or Markdown #..###### heading on line 1 -> title: {heading} | text: {body}
        for raw_doc, expected in (
            (
                "Title: Montebello\nMontebello is a part of Rockland County.",
                "title: Montebello | text: Montebello is a part of Rockland County.",
            ),
            (
                "Wikipedia Title: Montebello\nMontebello is a part of Rockland County.",
                "title: Montebello | text: Montebello is a part of Rockland County.",
            ),
            (
                "## Section 2.1: Montebello\nMontebello is a part of Rockland County.",
                "title: Section 2.1: Montebello | text: Montebello is a part of Rockland County.",
            ),
        ):
            self.assertEqual(titled_profile.format(raw_doc, query=False), expected)

        # Untitled multi-line passages, single-line passages, fact tuples, and entities -> title: none | text: {text}
        for untitled_input in (
            "Montebello\nMontebello is a part of Rockland County.",
            "First paragraph of an enterprise policy.\nSecond paragraph with details.",
            "George Rankin is a politician.",
            "('george rankin', 'is', 'a politician')",
            "george rankin",
            "\nBody with empty title line",
            "## Heading with empty body\n   ",
            "Title:   \nBody with blank title value",
        ):
            self.assertEqual(
                titled_profile.format(untitled_input, query=False),
                DOCUMENT_PREFIX + untitled_input,
            )
        # Reject already-prefixed titled document input
        with self.assertRaises(ValueError):
            titled_profile.format("title: Montebello | text: Already prefixed", query=False)

    def test_dspy_embedder_and_adapter_preserve_prefixes_and_l2_normalization(self):
        # Verify Lemonade (openai/...) and Ollama (ollama/...) dspy.Embedder initialization
        lemonade_emb = create_dspy_embedder(MODEL, base_url="http://localhost:13305/api/v1", provider="openai")
        self.assertEqual(lemonade_emb.model, f"openai/{MODEL}")
        self.assertEqual(lemonade_emb.default_kwargs.get("encoding_format"), "float")
        self.assertEqual(lemonade_emb.default_kwargs.get("api_base"), "http://localhost:13305/api/v1")

        ollama_emb = create_dspy_embedder("nomic-embed-text", base_url="http://localhost:11434", provider="ollama")
        self.assertEqual(ollama_emb.model, "ollama/nomic-embed-text")
        self.assertEqual(ollama_emb.default_kwargs.get("api_base"), "http://localhost:11434")

        # Real callable dspy.Embedder through DSPyEmbeddingAdapter (no mocks)
        recorded_inputs: list[list[str]] = []

        def fake_embed_fn(texts: list[str]) -> np.ndarray:
            recorded_inputs.append(list(texts))
            rows = []
            for idx, _ in enumerate(texts):
                vec = np.zeros(768, dtype=np.float32)
                vec[0] = float(idx + 2)
                vec[1] = float(idx + 3)
                rows.append(vec)
            return np.stack(rows)

        custom_dspy_embedder = create_dspy_embedder(fake_embed_fn, batch_size=2, caching=False)
        adapter = DSPyEmbeddingAdapter(
            profile=resolve_embedding_profile(MODEL, "distinct-titled"),
            embedder=custom_dspy_embedder,
        )

        hook_entries: list[dict[str, Any]] = []
        adapter.bind_telemetry_hook(hook_entries.append)

        doc_vecs = adapter.batch_encode(
            ["Title: Montebello\nMontebello is a village.", "Single line fact."],
            norm=True,
            stage="chunk_embed",
        )
        self.assertEqual(
            recorded_inputs[-1],
            [
                "title: Montebello | text: Montebello is a village.",
                "title: none | text: Single line fact.",
            ],
        )
        self.assertEqual(doc_vecs.shape, (2, 768))
        np.testing.assert_allclose(np.linalg.norm(doc_vecs, axis=1), 1.0, atol=1e-6)

        fact_q_vecs = adapter.batch_encode(
            "Where is Montebello?",
            instruction=FACT_INSTRUCTION,
            norm=True,
            stage="query_fact_embed",
        )
        self.assertEqual(recorded_inputs[-1], [FACT_CHECK_PREFIX + "Where is Montebello?"])
        self.assertEqual(fact_q_vecs.shape, (1, 768))

        passage_q_vecs = adapter.batch_encode(
            "Where is Montebello?",
            instruction=PASSAGE_INSTRUCTION,
            norm=True,
            stage="query_passage_embed",
        )
        self.assertEqual(recorded_inputs[-1], [QA_QUERY_PREFIX + "Where is Montebello?"])
        self.assertEqual(passage_q_vecs.shape, (1, 768))

        # Shared query mode uses QUERY_PREFIX for both instructions
        shared_adapter = DSPyEmbeddingAdapter(
            profile=resolve_embedding_profile(MODEL, "search"),
            embedder=custom_dspy_embedder,
        )
        shared_adapter.batch_encode("Where is Montebello?", instruction=FACT_INSTRUCTION, norm=True)
        self.assertEqual(recorded_inputs[-1], [QUERY_PREFIX + "Where is Montebello?"])

        self.assertEqual(len(adapter.telemetry), 3)
        self.assertEqual(len(hook_entries), 3)
        self.assertEqual(
            [e["stage"] for e in adapter.telemetry],
            ["chunk_embed", "query_fact_embed", "query_passage_embed"],
        )
        self.assertEqual(adapter.telemetry[0]["predictor"], "embedder")
        self.assertEqual(adapter.telemetry[0]["num_inputs"], 2)
        adapter.reset_telemetry()
        self.assertEqual(adapter.telemetry, [])

    def test_legacy_embedding_shim_is_retired(self):
        """Verify RetrievalEmbedding is completely removed from embedding and inference facade."""
        self.assertFalse(hasattr(embedding_module, "RetrievalEmbedding"))
        self.assertFalse(hasattr(inference_facade, "RetrievalEmbedding"))

    def test_adaptive_batch_splitting_on_physical_batch_overflow(self):
        """Verify DSPyEmbeddingAdapter gracefully splits without truncation on physical batch overflow."""
        def overflow_embed_fn(texts: list[str]) -> np.ndarray:
            if len(texts) > 1 or any(len(x) > 4000 for x in texts):
                raise RuntimeError(
                    "input (4056 tokens) is too large to process. increase the physical batch size (current batch size: 2048)"
                )
            vec = np.zeros((1, 768), dtype=np.float32)
            vec[0, 0] = 1.0
            return vec

        long_doc = "A" * 12000
        adapter = DSPyEmbeddingAdapter(
            profile=resolve_embedding_profile(MODEL, "distinct-titled"),
            embedder=create_dspy_embedder(overflow_embed_fn, batch_size=2, caching=False),
        )
        dspy_vecs = adapter.batch_encode([long_doc, "short doc"], norm=True)
        self.assertEqual(dspy_vecs.shape, (2, 768))
        np.testing.assert_allclose(np.linalg.norm(dspy_vecs, axis=1), 1.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
