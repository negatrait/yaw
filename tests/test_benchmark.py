"""Executable specifications for HippoRAG benchmark slicing, evaluation, and integrity gates."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from benchmark import (
    PAPER_BASELINES,
    build_benchmark_slice,
    check_benchmark_integrity,
    evaluate_benchmark,
    get_gold_answers,
    get_gold_docs,
    index_documents_in_batches,
    load_dataset_files,
)


class DatasetSliceTests(unittest.TestCase):
    def setUp(self):
        self.musique_samples = [
            {
                "id": "m1",
                "question": "What county is Erik Hort's birthplace a part of?",
                "answer": "Rockland County",
                "answer_aliases": ["Rockland County, New York"],
                "paragraphs": [
                    {"title": "Erik Hort", "paragraph_text": "Erik Hort was born in Montebello.", "is_supporting": True},
                    {"title": "Montebello", "paragraph_text": "Montebello is in Rockland County.", "is_supporting": True},
                    {"title": "Distractor 1", "paragraph_text": "Unrelated text.", "is_supporting": False},
                ],
            },
            {
                "id": "m2",
                "question": "What is George Rankin's occupation?",
                "answer": ["Politician"],
                "paragraphs": [
                    {"title": "George Rankin", "paragraph_text": "George Rankin is a politician.", "is_supporting": True},
                    {"title": "Distractor 2", "paragraph_text": "More unrelated text.", "is_supporting": False},
                ],
            },
        ]
        self.musique_corpus = [
            {"title": "Erik Hort", "text": "Erik Hort was born in Montebello."},
            {"title": "Montebello", "text": "Montebello is in Rockland County."},
            {"title": "Distractor 1", "text": "Unrelated text."},
            {"title": "George Rankin", "text": "George Rankin is a politician."},
            {"title": "Distractor 2", "text": "More unrelated text."},
            {"title": "Extra Corpus Doc", "text": "Only in full corpus."},
        ]

    def test_musique_gold_docs_and_answers_preserve_aliases_and_order(self):
        gold_docs = get_gold_docs(self.musique_samples, "musique")
        self.assertEqual(
            gold_docs,
            [
                [
                    "Erik Hort\nErik Hort was born in Montebello.",
                    "Montebello\nMontebello is in Rockland County.",
                ],
                ["George Rankin\nGeorge Rankin is a politician."],
            ],
        )
        gold_answers = get_gold_answers(self.musique_samples)
        self.assertEqual(
            gold_answers,
            [["Rockland County", "Rockland County, New York"], ["Politician"]],
        )

    def test_2wiki_and_hotpotqa_supporting_facts_separator_contract(self):
        sample = {
            "question": "Who directed film X?",
            "answer": "Director Y",
            "supporting_facts": [["Film X", 0]],
            "context": [
                ["Film X", ["Sentence one.", "Sentence two."]],
                ["Other", ["Ignored sentence."]],
            ],
        }
        wiki_gold = get_gold_docs([sample], "2wikimultihopqa")
        hotpot_gold = get_gold_docs([sample], "hotpotqa")
        self.assertEqual(wiki_gold, [["Film X\nSentence one. Sentence two."]])
        self.assertEqual(hotpot_gold, [["Film X\nSentence one.Sentence two."]])

    def test_slice_invariant_gold_docs_always_subset_of_candidate_docs(self):
        for max_q in (1, 2):
            docs, queries, gold_docs, gold_answers = build_benchmark_slice(
                "musique", self.musique_samples, self.musique_corpus, max_queries=max_q, full_corpus=False
            )
            self.assertEqual(len(queries), max_q)
            self.assertEqual(len(gold_docs), max_q)
            self.assertEqual(len(gold_answers), max_q)
            self.assertEqual(len(docs), len(set(docs)), "Candidate docs must be deduplicated")
            for g in gold_docs:
                self.assertTrue(set(g).issubset(set(docs)))
            self.assertNotIn("Extra Corpus Doc\nOnly in full corpus.", docs)

    def test_full_corpus_flag_includes_entire_corpus(self):
        docs, queries, _, _ = build_benchmark_slice(
            "musique", self.musique_samples, self.musique_corpus, max_queries=1, full_corpus=True
        )
        self.assertEqual(len(queries), 1)
        self.assertEqual(len(docs), 6)
        self.assertIn("Extra Corpus Doc\nOnly in full corpus.", docs)

    def test_negative_boundaries_on_malformed_samples_and_max_queries(self):
        with self.assertRaises(ValueError):
            build_benchmark_slice("musique", self.musique_samples, self.musique_corpus, max_queries=0)
        with self.assertRaises(ValueError):
            get_gold_docs([{"question": "No context fields"}], "musique")
        with self.assertRaises(ValueError):
            get_gold_answers([{"question": "No answer field"}])
        with self.assertRaises(ValueError):
            get_gold_answers([{"question": "Null answer", "answer": None}])

    def test_load_dataset_files_reads_local_disk_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "sample.json").write_text(json.dumps(self.musique_samples), encoding="utf-8")
            (tmp_path / "sample_corpus.json").write_text(json.dumps(self.musique_corpus[:3]), encoding="utf-8")
            samples, corpus = load_dataset_files("sample", dataset_dir=tmp_path)
            self.assertEqual(len(samples), 2)
            self.assertEqual(len(corpus), 3)

            # When include_corpus=False and max_queries=1, corpus file is not required and samples are sliced
            (tmp_path / "sample_corpus.json").unlink()
            sliced_samples, empty_corpus = load_dataset_files(
                "sample", dataset_dir=tmp_path, max_queries=1, include_corpus=False,
            )
            self.assertEqual(len(sliced_samples), 1)
            self.assertEqual(empty_corpus, [])
            with self.assertRaises(ValueError):
                load_dataset_files("sample", dataset_dir=tmp_path, max_queries=0, include_corpus=False)


class EvaluationAndIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.queries = [
            "What county is Erik Hort's birthplace a part of?",
            "What is George Rankin's occupation?",
        ]
        self.gold_docs = [
            [
                "Erik Hort\nErik Hort was born in Montebello.",
                "Montebello\nMontebello is in Rockland County.",
            ],
            ["George Rankin\nGeorge Rankin is a politician."],
        ]
        self.gold_answers = [["Rockland County"], ["Politician"]]

    def test_evaluate_benchmark_compares_hipporag_and_standard_dpr(self):
        with self.assertRaisesRegex(ValueError, "qa_batch_size"):
            evaluate_benchmark(Mock(), self.queries, self.gold_docs, self.gold_answers, qa_batch_size=0)

        graph_results = [
            SimpleNamespace(
                question=self.queries[0],
                answer="Rockland County",
                docs=list(self.gold_docs[0]),
                doc_scores=[0.9, 0.8],
                graph_seeds=[("erik hort", "born in", "montebello")],
            ),
            SimpleNamespace(
                question=self.queries[1],
                answer="Politician",
                docs=list(self.gold_docs[1]),
                doc_scores=[0.95],
                graph_seeds=[],
            ),
        ]
        dpr_results = [
            SimpleNamespace(
                question=self.queries[0],
                answer="Montebello",
                docs=[self.gold_docs[0][0], "Distractor\nUnrelated"],
                doc_scores=[0.7, 0.6],
                graph_seeds=[],
            ),
            SimpleNamespace(
                question=self.queries[1],
                answer="Politician",
                docs=list(self.gold_docs[1]),
                doc_scores=[0.95],
                graph_seeds=[],
            ),
        ]
        hipporag = Mock()
        hipporag.rag_qa.return_value = (
            graph_results,
            [],
            [],
            {"Recall@1": 0.75, "Recall@2": 1.0, "Recall@5": 1.0, "Recall@200": 1.0},
            {"ExactMatch": 1.0, "F1": 1.0},
        )
        hipporag.rag_qa_dpr.return_value = (
            dpr_results,
            [],
            [],
            {"Recall@1": 0.75, "Recall@2": 0.75, "Recall@5": 0.75, "Recall@200": 1.0},
            {"ExactMatch": 0.5, "F1": 0.5},
        )

        records, metrics = evaluate_benchmark(
            hipporag,
            self.queries,
            self.gold_docs,
            self.gold_answers,
            dataset_name="musique",
            qa_top_k=5,
        )
        self.assertEqual(len(records), 2)
        self.assertEqual(metrics["hipporag"]["retrieval"]["Recall@5"], 1.0)
        self.assertEqual(metrics["standard_dpr"]["retrieval"]["Recall@5"], 0.75)
        self.assertAlmostEqual(metrics["delta_vs_dpr"]["Recall@5"], 0.25)
        self.assertAlmostEqual(metrics["delta_vs_dpr"]["F1"], 0.5)
        self.assertEqual(
            metrics["recognition_memory"],
            {
                "queries_with_graph_seeds": 1,
                "dpr_fallback_count": 1,
                "dpr_fallback_rate": 0.5,
            },
        )
        self.assertEqual(metrics["paper_baselines"], PAPER_BASELINES["musique"])

    def test_evaluate_benchmark_staged_one_by_one_qa_and_memory_release(self):
        class StagedHippoRAGStub:
            def __init__(self, queries, gold_docs):
                self._queries = queries
                self._gold_docs = gold_docs
                self.retrieve_batches = []
                self.retrieve_dpr_batches = []
                self.qa_batches = []
                self.memory_released_before_qa = False
                self._openie_info = [{"idx": "c1"}]
                self.entity_embeddings = [1, 2, 3]
                self.passage_embeddings = [4, 5, 6]
                self.fact_embeddings = [7, 8, 9]
                self.query_to_embedding: dict[str, dict[str, object]] = {"triple": {"q": 1}, "passage": {"q": 2}}
                self.ready_to_retrieve = True

            def retrieve(self, queries, num_to_retrieve=None):
                self.retrieve_batches.append((list(queries), num_to_retrieve))
                self.query_to_embedding["triple"][queries[0]] = "vec"
                self.query_to_embedding["passage"][queries[0]] = "vec"
                q = queries[0]
                idx = self._queries.index(q)
                return [
                    SimpleNamespace(
                        question=q,
                        answer=None,
                        docs=list(self._gold_docs[idx]) + [f"Extra {i}" for i in range(10)],
                        doc_scores=[0.9] * 12,
                        doc_metadata=[{}] * 12,
                        graph_seeds=[("erik hort", "born in", "montebello")] if idx == 0 else [],
                        _mode="graph",
                    )
                ]

            def retrieve_dpr(self, queries, num_to_retrieve=None):
                self.retrieve_dpr_batches.append((list(queries), num_to_retrieve))
                q = queries[0]
                idx = self._queries.index(q)
                return [
                    SimpleNamespace(
                        question=q,
                        answer=None,
                        docs=[self._gold_docs[idx][0], "Distractor\nUnrelated"],
                        doc_scores=[0.7, 0.6],
                        doc_metadata=[{}, {}],
                        graph_seeds=[],
                        _mode="dpr",
                    )
                ]

            def qa(self, query_solutions):
                if (
                    self.entity_embeddings is None
                    and self.passage_embeddings is None
                    and self.fact_embeddings is None
                    and self._openie_info is None
                    and not self.ready_to_retrieve
                ):
                    self.memory_released_before_qa = True
                self.qa_batches.append(len(query_solutions))
                for sol in query_solutions:
                    if sol.question == self._queries[0]:
                        sol.answer = "Rockland County" if sol._mode == "graph" else "Montebello"
                    else:
                        sol.answer = "Politician"
                return query_solutions, ["raw"], [{}]

        stub = StagedHippoRAGStub(self.queries, self.gold_docs)
        records, metrics = evaluate_benchmark(
            stub,
            self.queries,
            self.gold_docs,
            self.gold_answers,
            dataset_name="2wikimultihopqa",
            qa_top_k=5,
            qa_batch_size=1,
        )
        self.assertEqual(stub.retrieve_batches, [([self.queries[0]], 5), ([self.queries[1]], 5)])
        self.assertEqual(stub.retrieve_dpr_batches, [([self.queries[0]], 5), ([self.queries[1]], 5)])
        self.assertEqual(stub.qa_batches, [1, 1, 1, 1])
        self.assertTrue(stub.memory_released_before_qa)
        self.assertEqual(stub.query_to_embedding, {"triple": {}, "passage": {}})
        self.assertEqual(len(records[0]["qa_evidence"]), 5)
        self.assertEqual(metrics["hipporag"]["qa"]["ExactMatch"], 1.0)
        self.assertEqual(metrics["standard_dpr"]["qa"]["ExactMatch"], 0.5)

        # Retrieval-only mode (include_qa=False) skips qa() calls entirely and sets qa metrics to None
        retrieval_stub = StagedHippoRAGStub(self.queries, self.gold_docs)
        ret_records, ret_metrics = evaluate_benchmark(
            retrieval_stub,
            self.queries,
            self.gold_docs,
            self.gold_answers,
            dataset_name="hotpotqa",
            qa_top_k=5,
            qa_batch_size=1,
            include_qa=False,
        )
        self.assertEqual(retrieval_stub.qa_batches, [])
        self.assertEqual(ret_metrics["hipporag"]["retrieval"]["Recall@5"], 1.0)
        self.assertIsNone(ret_metrics["hipporag"]["qa"])
        self.assertIsNone(ret_metrics["standard_dpr"]["qa"])
        self.assertIsNone(ret_metrics["delta_vs_dpr"]["ExactMatch"])
        self.assertIsNone(ret_metrics["delta_vs_dpr"]["F1"])
        self.assertIsNone(ret_records[0]["qa_metrics"])
        self.assertEqual(
            check_benchmark_integrity(ret_records, {"num_passage_nodes": 5, "num_extracted_triples": 10}, 5, include_qa=False),
            [],
        )

    def test_integrity_gate_catches_silent_failures(self):
        valid_records = [
            {
                "question": self.queries[0],
                "answer": "Rockland County",
                "dpr_answer": "Montebello",
                "graph_seeds": [("erik hort", "born in", "montebello")],
            },
            {
                "question": self.queries[1],
                "answer": "Politician",
                "dpr_answer": "Politician",
                "graph_seeds": [],
            },
        ]
        valid_graph = {"num_passage_nodes": 5, "num_extracted_triples": 12}
        self.assertEqual(
            check_benchmark_integrity(valid_records, valid_graph, expected_doc_count=5),
            [],
        )

        # Trap 1: chat errors
        errs = check_benchmark_integrity(
            valid_records, valid_graph, expected_doc_count=5, extraction_errors=["ValueError: bad"]
        )
        self.assertTrue(any("extraction_llm" in e for e in errs))

        # Trap 2: passage count mismatch or zero triples
        errs = check_benchmark_integrity(
            valid_records, {"num_passage_nodes": 4, "num_extracted_triples": 0}, expected_doc_count=5
        )
        self.assertTrue(any("passage count" in e.lower() for e in errs))
        self.assertTrue(any("no triples" in e.lower() for e in errs))

        # Trap 3: all graph_seeds empty on multi-query benchmark
        no_seed_records = [{**r, "graph_seeds": []} for r in valid_records]
        errs = check_benchmark_integrity(no_seed_records, valid_graph, expected_doc_count=5)
        self.assertTrue(any("graph_seeds" in e for e in errs))

    def test_index_documents_in_batches_handles_fresh_partial_and_cached_stores(self):
        docs = [f"Doc {i}\nText {i}" for i in range(18)]

        # 1. Invalid batch size rejected
        with self.assertRaisesRegex(ValueError, "index_batch_size"):
            index_documents_in_batches(Mock(), docs, batch_size=0)

        # 2. Empty docs returns 0 calls
        empty_rag = Mock()
        self.assertEqual(index_documents_in_batches(empty_rag, [], batch_size=8), 0)
        empty_rag.index.assert_not_called()

        # 3. Fresh index (_graph_state_available=False) indexes in 3 batches of 8, 8, 2 and releases transient state
        fresh_rag = Mock(_graph_state_available=False, _openie_info=[{"idx": "1"}], entity_id_to_row={"e": 1})
        self.assertEqual(index_documents_in_batches(fresh_rag, docs, batch_size=8), 3)
        self.assertEqual(
            [call.kwargs["docs"] for call in fresh_rag.index.call_args_list],
            [docs[0:8], docs[8:16], docs[16:18]],
        )
        self.assertIsNone(fresh_rag._openie_info)
        self.assertIsNone(fresh_rag.entity_id_to_row)

        # 4. Partially built index (_graph_state_available=True) indexes only remaining docs in batches
        partial_rag = Mock(_graph_state_available=True)
        partial_rag.chunk_embedding_store.get_all_texts.return_value = set(docs[:8])
        self.assertEqual(index_documents_in_batches(partial_rag, docs, batch_size=8), 2)
        self.assertEqual(
            [call.kwargs["docs"] for call in partial_rag.index.call_args_list],
            [docs[8:16], docs[16:18]],
        )

        # 5. Store has all docs but graph has fewer passage nodes: reconciles with 1 bounded micro-batch (not full docs)
        reconcile_rag = Mock(_graph_state_available=True)
        reconcile_rag.chunk_embedding_store.get_all_texts.return_value = set(docs)
        reconcile_rag.get_graph_info.return_value = {"num_passage_nodes": 10}
        self.assertEqual(index_documents_in_batches(reconcile_rag, docs, batch_size=8), 1)
        reconcile_rag.index.assert_called_once_with(docs=docs[:8])

        # 6. Fully built index (_graph_state_available=True, all docs in store and graph) skips index() altogether
        ready_rag = Mock(_graph_state_available=True)
        ready_rag.chunk_embedding_store.get_all_texts.return_value = set(docs)
        ready_rag.get_graph_info.return_value = {"num_passage_nodes": len(docs)}
        self.assertEqual(index_documents_in_batches(ready_rag, docs, batch_size=8), 0)
        ready_rag.index.assert_not_called()

        # 7. Real HippoRAG pre_openie method checkpoints OpenIE in micro-batches and builds graph once
        class StagedIndexStub:
            def __init__(self):
                self._graph_state_available = False
                self._openie_info = [{"idx": "1"}]
                self.pre_openie_batches = []
                self.index_calls = []

            def pre_openie(self, batch_docs):
                self.pre_openie_batches.append(list(batch_docs))
                raise RuntimeError("Offline OpenIE completed. Run indexing again with openie_mode='online' to build the graph.")

            def index(self, docs):
                self.index_calls.append(list(docs))

        staged_index = StagedIndexStub()
        self.assertEqual(index_documents_in_batches(staged_index, docs, batch_size=8), 3)
        self.assertEqual(staged_index.pre_openie_batches, [docs[0:8], docs[8:16], docs[16:18]])
        self.assertEqual(staged_index.index_calls, [docs])
        self.assertIsNone(staged_index._openie_info)


if __name__ == "__main__":
    unittest.main()
