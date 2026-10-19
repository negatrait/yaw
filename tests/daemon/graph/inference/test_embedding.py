"""Offline regression tests; run with unittest discover."""
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
from hipporag import HippoRAG
from hipporag.embedding_model.OpenAI import OpenAIEmbeddingModel
from hipporag.prompts.linking import get_query_instruction
from hipporag.utils.config_utils import BaseConfig

from yaw.daemon.graph.inference.embedding import (
    DOCUMENT_PREFIX,
    FACT_CHECK_PREFIX,
    MODEL,
    QA_QUERY_PREFIX,
    QUERY_PREFIX,
    resolve_embedding_profile,
)
from yaw.daemon.graph.inference.legacy_embedding import RetrievalEmbedding


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.embedding = RetrievalEmbedding(BaseConfig(
            embedding_model_name=MODEL, embedding_base_url="http://localhost:13305/api/v1",
            embedding_batch_size=2,
        ))
        self.embedding.client.close()
        self.mock_client: Any = Mock()
        self.embedding.client = self.mock_client
        self.mock_client.embeddings.create.side_effect = self.response
        self.addCleanup(self.embedding.close)

    @staticmethod
    def response(**request):
        # Deliberately reversed response rows with distinct, non-unit vectors.
        rows = [
            SimpleNamespace(index=i, embedding=[float(i + 1)] + [0.] * 767)
            for i in range(len(request["input"]))
        ]
        return SimpleNamespace(data=rows[::-1], usage=SimpleNamespace(
            prompt_tokens=len(rows), total_tokens=len(rows),
        ))

    def test_documents_preserve_batch_order_usage_and_normalization(self):
        vectors = self.embedding.batch_encode(["one\ntwo", "three", "four"], norm=False)
        calls = self.mock_client.embeddings.create.call_args_list
        self.assertEqual([c.kwargs["input"] for c in calls], [
            [DOCUMENT_PREFIX + "one\ntwo", DOCUMENT_PREFIX + "three"],
            [DOCUMENT_PREFIX + "four"],
        ])
        self.assertTrue(all(c.kwargs["model"] == MODEL for c in calls))
        self.assertTrue(all(c.kwargs["encoding_format"] == "float" for c in calls))
        np.testing.assert_array_equal(vectors[:, 0], [1., 2., 1.])
        self.assertEqual(self.embedding.last_usage, {"prompt_tokens": 3, "total_tokens": 3})
        normalized = self.embedding.batch_encode(["one", "two"], norm=True)
        np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), 1)

    def test_actual_hipporag_query_dispatch_supplies_query_role(self):
        state: Any = SimpleNamespace(
            global_config=BaseConfig(), embedding_model=self.embedding,
            fact_node_keys=["fact"], passage_node_keys=["passage"],
            query_to_embedding={"triple": {}, "passage": {}},
        )
        HippoRAG.get_query_embeddings(state, ["occupation?", "occupation?"])
        request = self.mock_client.embeddings.create.call_args.kwargs
        self.assertEqual(request["input"], [QUERY_PREFIX + "occupation?"])
        self.assertIs(state.query_to_embedding["triple"]["occupation?"],
                      state.query_to_embedding["passage"]["occupation?"])
        self.mock_client.embeddings.create.assert_called_once()
        # The base class is untouched, so other users do not inherit our task policy.
        self.assertEqual(OpenAIEmbeddingModel.query_instruction_mode, "ignored")

    def test_distinct_query_instruction_mode_dispatches_separate_fact_and_passage_prefixes(self):
        distinct_profile = resolve_embedding_profile(MODEL, "distinct")
        self.assertEqual(distinct_profile.query_instruction_mode, "distinct")
        distinct_embedding = RetrievalEmbedding(BaseConfig(
            embedding_model_name=MODEL, embedding_base_url="http://localhost:13305/api/v1",
        ), distinct_profile)
        distinct_embedding.client.close()
        distinct_mock: Any = Mock()
        distinct_embedding.client = distinct_mock
        distinct_mock.embeddings.create.side_effect = self.response
        self.addCleanup(distinct_embedding.close)

        state: Any = SimpleNamespace(
            global_config=BaseConfig(), embedding_model=distinct_embedding,
            fact_node_keys=["fact"], passage_node_keys=["passage"],
            query_to_embedding={"triple": {}, "passage": {}},
        )
        HippoRAG.get_query_embeddings(state, ["occupation?"])
        calls = distinct_mock.embeddings.create.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].kwargs["input"], [FACT_CHECK_PREFIX + "occupation?"])
        self.assertEqual(calls[1].kwargs["input"], [QA_QUERY_PREFIX + "occupation?"])
        with self.assertRaises(ValueError):
            distinct_embedding.batch_encode([FACT_CHECK_PREFIX + "already prefixed"])

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

    def test_both_query_instructions_and_invalid_input(self):
        for task in ("query_to_fact", "query_to_passage"):
            self.embedding.batch_encode("who?", instruction=get_query_instruction(task))
            self.assertEqual(self.mock_client.embeddings.create.call_args.kwargs["input"],
                             [QUERY_PREFIX + "who?"])
        for texts, options, exception in [
            ([], {}, ValueError), ([1], {}, TypeError),
            ([QUERY_PREFIX + "who?"], {}, ValueError),
            (["text"], {"instruction": "unsupported"}, ValueError),
            (["text"], {"batch_size": 0}, ValueError),
            (["text"], {"unexpected": True}, TypeError),
        ]:
            with self.subTest(texts=texts, options=options), self.assertRaises(exception):
                self.embedding.batch_encode(texts, **options)

    def test_malformed_server_responses_fail(self):
        for vector in ([0.] * 768, [float("nan")] * 768, [float("inf")] * 768, [1.] * 767):
            self.mock_client.embeddings.create.side_effect = None
            self.mock_client.embeddings.create.return_value = SimpleNamespace(
                data=[SimpleNamespace(index=0, embedding=vector)],
                usage=SimpleNamespace(prompt_tokens=1, total_tokens=1),
            )
            with self.subTest(vector=vector[:1], size=len(vector)), self.assertRaises(ValueError):
                self.embedding.batch_encode(["text"])
            assert self.embedding.last_usage is not None
            self.assertFalse(self.embedding.last_usage["complete"])
        for response in (
            SimpleNamespace(data=[], usage=None),
            SimpleNamespace(data=[SimpleNamespace(index=1, embedding=[1.] * 768)],
                            usage=SimpleNamespace(prompt_tokens=1, total_tokens=1)),
        ):
            self.mock_client.embeddings.create.return_value = response
            with self.assertRaises(ValueError):
                self.embedding.batch_encode(["text"])

    def test_dspy_embedder_and_adapter_preserve_prefixes_and_l2_normalization(self):
        from yaw.daemon.graph.inference.embedding import (
            FACT_INSTRUCTION,
            PASSAGE_INSTRUCTION,
            DSPyEmbeddingAdapter,
            create_dspy_embedder,
        )

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

    def test_legacy_embedding_and_inference_facade_exports(self):
        """Verify RetrievalEmbedding is detangled to legacy_embedding and accessible via inference facade."""
        from yaw.daemon.graph.inference import RetrievalEmbedding as FacadeRetrievalEmbedding
        from yaw.daemon.graph.inference.legacy_embedding import (
            RetrievalEmbedding as DirectRetrievalEmbedding,
        )

        self.assertIs(DirectRetrievalEmbedding, RetrievalEmbedding)
        self.assertIs(FacadeRetrievalEmbedding, RetrievalEmbedding)

        import yaw.daemon.graph.inference.embedding as embedding_module
        self.assertFalse(hasattr(embedding_module, "RetrievalEmbedding"))

    def test_adaptive_batch_splitting_and_truncation_on_physical_batch_overflow(self):
        """Verify both RetrievalEmbedding and DSPyEmbeddingAdapter gracefully split without truncation on physical batch overflow."""
        from yaw.daemon.graph.inference.embedding import (
            DSPyEmbeddingAdapter,
            create_dspy_embedder,
        )

        def overflow_create(**request):
            inputs = request["input"]
            if len(inputs) > 1 or any(len(x) > 4000 for x in inputs):
                raise RuntimeError(
                    "Error code: 500 - {'error': {'code': 500, 'message': "
                    "'input (4056 tokens) is too large to process. increase the physical batch size (current batch size: 2048)'}}"
                )
            return self.response(**request)

        self.mock_client.embeddings.create.side_effect = overflow_create
        long_doc = "A" * 12000
        vecs = self.embedding.batch_encode([long_doc, "short doc"], norm=True)
        self.assertEqual(vecs.shape, (2, 768))
        np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-6)
        # Without truncation, the 12,000 char doc is gracefully divided into sub-chunks plus the short doc (6 total pieces)
        self.assertEqual(self.embedding.last_usage, {"prompt_tokens": 6, "total_tokens": 6})

        def overflow_embed_fn(texts: list[str]) -> np.ndarray:
            if len(texts) > 1 or any(len(x) > 4000 for x in texts):
                raise RuntimeError(
                    "input (4056 tokens) is too large to process. increase the physical batch size (current batch size: 2048)"
                )
            vec = np.zeros((1, 768), dtype=np.float32)
            vec[0, 0] = 1.0
            return vec

        adapter = DSPyEmbeddingAdapter(
            profile=resolve_embedding_profile(MODEL, "distinct-titled"),
            embedder=create_dspy_embedder(overflow_embed_fn, batch_size=2, caching=False),
        )
        dspy_vecs = adapter.batch_encode([long_doc, "short doc"], norm=True)
        self.assertEqual(dspy_vecs.shape, (2, 768))
        np.testing.assert_allclose(np.linalg.norm(dspy_vecs, axis=1), 1.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
