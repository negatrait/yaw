"""Offline regression tests; run with unittest discover."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
from hipporag import HippoRAG
from hipporag.embedding_model.OpenAI import OpenAIEmbeddingModel
from hipporag.prompts.linking import get_query_instruction
from hipporag.utils.config_utils import BaseConfig

from yaw.embedding import (
    DOCUMENT_PREFIX,
    FACT_CHECK_PREFIX,
    MODEL,
    QA_QUERY_PREFIX,
    QUERY_PREFIX,
    RetrievalEmbedding,
    resolve_embedding_profile,
)


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.embedding = RetrievalEmbedding(BaseConfig(
            embedding_model_name=MODEL, embedding_base_url="http://localhost:13305/api/v1",
            embedding_batch_size=2,
        ))
        self.embedding.client.close()
        self.embedding.client = Mock()
        self.embedding.client.embeddings.create.side_effect = self.response
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
        calls = self.embedding.client.embeddings.create.call_args_list
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
        state = SimpleNamespace(
            global_config=BaseConfig(), embedding_model=self.embedding,
            fact_node_keys=["fact"], passage_node_keys=["passage"],
            query_to_embedding={"triple": {}, "passage": {}},
        )
        HippoRAG.get_query_embeddings(state, ["occupation?", "occupation?"])
        request = self.embedding.client.embeddings.create.call_args.kwargs
        self.assertEqual(request["input"], [QUERY_PREFIX + "occupation?"])
        self.assertIs(state.query_to_embedding["triple"]["occupation?"],
                      state.query_to_embedding["passage"]["occupation?"])
        self.embedding.client.embeddings.create.assert_called_once()
        # The base class is untouched, so other users do not inherit our task policy.
        self.assertEqual(OpenAIEmbeddingModel.query_instruction_mode, "ignored")

    def test_distinct_query_instruction_mode_dispatches_separate_fact_and_passage_prefixes(self):
        distinct_profile = resolve_embedding_profile(MODEL, "distinct")
        self.assertEqual(distinct_profile.query_instruction_mode, "distinct")
        distinct_embedding = RetrievalEmbedding(BaseConfig(
            embedding_model_name=MODEL, embedding_base_url="http://localhost:13305/api/v1",
        ), distinct_profile)
        distinct_embedding.client.close()
        distinct_embedding.client = Mock()
        distinct_embedding.client.embeddings.create.side_effect = self.response
        self.addCleanup(distinct_embedding.close)

        state = SimpleNamespace(
            global_config=BaseConfig(), embedding_model=distinct_embedding,
            fact_node_keys=["fact"], passage_node_keys=["passage"],
            query_to_embedding={"triple": {}, "passage": {}},
        )
        HippoRAG.get_query_embeddings(state, ["occupation?"])
        calls = distinct_embedding.client.embeddings.create.call_args_list
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

        # Multi-line titled passage -> title: {first_line} | text: {body}
        self.assertEqual(
            titled_profile.format("Montebello\nMontebello is a part of Rockland County.", query=False),
            "title: Montebello | text: Montebello is a part of Rockland County.",
        )
        # Single-line passage, fact tuple string, and entity string -> title: none | text: {text}
        for single_line in (
            "George Rankin is a politician.",
            "('george rankin', 'is', 'a politician')",
            "george rankin",
            "\nBody with empty title line",
            "Title with empty body\n   ",
        ):
            self.assertEqual(
                titled_profile.format(single_line, query=False),
                DOCUMENT_PREFIX + single_line,
            )
        # Reject already-prefixed titled document input
        with self.assertRaises(ValueError):
            titled_profile.format("title: Montebello | text: Already prefixed", query=False)

    def test_both_query_instructions_and_invalid_input(self):
        for task in ("query_to_fact", "query_to_passage"):
            self.embedding.batch_encode("who?", instruction=get_query_instruction(task))
            self.assertEqual(self.embedding.client.embeddings.create.call_args.kwargs["input"],
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
            self.embedding.client.embeddings.create.side_effect = None
            self.embedding.client.embeddings.create.return_value = SimpleNamespace(
                data=[SimpleNamespace(index=0, embedding=vector)],
                usage=SimpleNamespace(prompt_tokens=1, total_tokens=1),
            )
            with self.subTest(vector=vector[:1], size=len(vector)), self.assertRaises(ValueError):
                self.embedding.batch_encode(["text"])
            self.assertFalse(self.embedding.last_usage["complete"])
        for response in (
            SimpleNamespace(data=[], usage=None),
            SimpleNamespace(data=[SimpleNamespace(index=1, embedding=[1.] * 768)],
                            usage=SimpleNamespace(prompt_tokens=1, total_tokens=1)),
        ):
            self.embedding.client.embeddings.create.return_value = response
            with self.assertRaises(ValueError):
                self.embedding.batch_encode(["text"])


if __name__ == "__main__":
    unittest.main()

