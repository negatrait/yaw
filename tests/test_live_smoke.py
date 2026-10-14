"""Check the acceptance guard rejects the previously silent retrieval failure."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from live_smoke import DOCS, GOLD_ANSWERS, GOLD_DOCS, QUERIES, check_results, evaluate


class SmokeChecks(unittest.TestCase):
    def setUp(self):
        self.records = [
            {"question": question, "answer": answer, "qa_evidence": list(docs)}
            for question, answer, docs in zip(
                QUERIES,
                ["Politician", "The slipper fit and she reunited with the prince.", "Rockland County"],
                GOLD_DOCS, strict=True,
            )
        ]

    def test_expected_fixture_passes(self):
        self.assertEqual(check_results(self.records), [])

    def test_original_unknown_answer_and_missing_george_fail(self):
        self.records[0].update(answer="Unknown", qa_evidence=[DOCS[3], DOCS[4]])
        errors = check_results(self.records)
        self.assertTrue(any("rank first" in error for error in errors))
        self.assertTrue(any("Unknown" in error for error in errors))

    def test_missing_bridge_passage_fails_even_with_correct_answer(self):
        self.records[2]["qa_evidence"] = [DOCS[8]]
        self.assertTrue(any("missing evidence" in error for error in check_results(self.records)))

    def test_empty_graph_seeds_fails_silent_dpr_fallback(self):
        self.records[0]["graph_seeds"] = []
        self.assertTrue(any("empty graph_seeds" in error for error in check_results(self.records)))


class EvaluationChecks(unittest.TestCase):
    def run_evaluation(self, answers):
        results = [SimpleNamespace(question=q, answer=a, docs=list(docs),
                                   doc_scores=[1.] * len(docs), graph_seeds=[("a", "b", "c")])
                   for q, a, docs in zip(QUERIES, answers, GOLD_DOCS, strict=True)]
        hipporag = Mock()
        hipporag.rag_qa.return_value = (
            results, [], [],
            {"Recall@1": .6111, "Recall@2": .8889, "Recall@5": 1., "Recall@200": 1.},
            {"ExactMatch": .6667, "F1": .8},
        )
        hipporag.retrieve_dpr.return_value = (
            results,
            {"Recall@1": .6111, "Recall@2": .7222, "Recall@5": .9444, "Recall@200": 1.},
        )
        rows, metrics = evaluate(hipporag)
        hipporag.rag_qa.assert_called_once_with(
            queries=QUERIES, gold_docs=GOLD_DOCS, gold_answers=GOLD_ANSWERS,
        )
        hipporag.retrieve_dpr.assert_called_once_with(
            queries=QUERIES, num_to_retrieve=5, gold_docs=GOLD_DOCS,
        )
        self.assertNotIn("Recall@200", metrics["retrieval"])
        self.assertNotIn("Recall@200", metrics["dpr_retrieval"])
        self.assertEqual(metrics["qa"], hipporag.rag_qa.return_value[4])
        return rows

    def test_golds_and_per_question_metrics_are_recorded(self):
        rows = self.run_evaluation([answers[0] for answers in GOLD_ANSWERS])
        for i, row in enumerate(rows):
            self.assertEqual(row["gold_docs"], GOLD_DOCS[i])
            self.assertEqual(row["gold_answers"], GOLD_ANSWERS[i])
            self.assertEqual(row["qa_metrics"], {"ExactMatch": 1., "F1": 1.})

    def test_f1_records_partial_overlap_without_claiming_exact_match(self):
        rows = self.run_evaluation(["Politician", "unrelated", "Rockland"])
        self.assertEqual(rows[1]["qa_metrics"], {"ExactMatch": 0., "F1": 0.})
        self.assertEqual(rows[2]["qa_metrics"]["ExactMatch"], 0.)
        self.assertAlmostEqual(rows[2]["qa_metrics"]["F1"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
