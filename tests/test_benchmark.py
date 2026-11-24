"""Executable specifications for YAW benchmark slicing, evaluation, and integrity gates."""

from __future__ import annotations

import contextlib
import json
import re
import tempfile
import unittest
from importlib import resources
from pathlib import Path
from typing import Any, override

import numpy as np
from numpy.typing import NDArray

from scripts.benchmark import (
    BASELINES,
    DATASET_REGISTRY,
    DATASETS,
    build_benchmark_slice,
    check_benchmark_integrity,
    get_gold_answers,
    get_gold_docs,
    load_dataset_files,
)
from scripts.evaluation import RetrievalRecall
from yaw.core import ChunkPointer, YawConfig, render_document_text
from yaw.datasets import document_identity, prepare_dataset_bundle, resolve_dataset_name
from yaw.pipeline import chunk_file_content, close_pipeline, start_pipeline, update_pipeline_once
from yaw.retrieval import (
    build_retrieval_index,
    hydrate_passage,
    hydrate_ranked_passages,
    open_readonly_db,
    search_index,
)


def selected_texts(ids: list[str], corpus: list[dict[str, Any]]) -> list[str]:
    by_id = {document_identity(row): render_document_text(str(row["title"]), str(row["text"])) for row in corpus}
    return [by_id[document_id] for document_id in ids]


class DatasetSliceTests(unittest.TestCase):
    @override
    def setUp(self) -> None:
        self.musique_samples = [
            {
                "id": "m1",
                "question": "What county is Erik Hort's birthplace a part of?",
                "answer": "Rockland County",
                "answer_aliases": ["Rockland County, New York"],
                "paragraphs": [
                    {
                        "title": "Erik Hort",
                        "paragraph_text": "Erik Hort was born in Montebello.",
                        "is_supporting": True,
                    },
                    {
                        "title": "Montebello",
                        "paragraph_text": "Montebello is in Rockland County.",
                        "is_supporting": True,
                    },
                    {
                        "title": "Distractor 1",
                        "paragraph_text": "Unrelated text.",
                        "is_supporting": False,
                    },
                ],
            },
            {
                "id": "m2",
                "question": "What is George Rankin's occupation?",
                "answer": ["Politician"],
                "paragraphs": [
                    {
                        "title": "George Rankin",
                        "paragraph_text": "George Rankin is a politician.",
                        "is_supporting": True,
                    },
                    {
                        "title": "Distractor 2",
                        "paragraph_text": "More unrelated text.",
                        "is_supporting": False,
                    },
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

    def test_musique_gold_docs_and_answers_preserve_aliases_and_order(self) -> None:
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

    def test_2wiki_and_hotpotqa_supporting_facts_separator_contract(self) -> None:
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

    def test_slice_invariant_gold_docs_always_subset_of_candidate_docs(self) -> None:
        for max_q in (1, 2):
            doc_ids, queries, gold_docs, gold_answers = build_benchmark_slice(
                "musique",
                self.musique_samples,
                self.musique_corpus,
                max_queries=max_q,
                full_corpus=False,
            )
            docs = selected_texts(doc_ids, self.musique_corpus)
            self.assertIsNotNone(gold_docs)
            assert gold_docs is not None
            self.assertEqual(len(queries), max_q)
            self.assertEqual(len(gold_docs), max_q)
            self.assertEqual(len(gold_answers), max_q)
            self.assertEqual(len(docs), len(set(docs)), "Candidate docs must be deduplicated")
            for g in gold_docs:
                self.assertTrue(set(g).issubset(set(docs)))
            self.assertNotIn("Extra Corpus Doc\nOnly in full corpus.", docs)

    def test_full_corpus_flag_includes_entire_corpus(self) -> None:
        doc_ids, queries, _, _ = build_benchmark_slice(
            "musique", self.musique_samples, self.musique_corpus, max_queries=1, full_corpus=True
        )
        docs = selected_texts(doc_ids, self.musique_corpus)
        self.assertEqual(len(queries), 1)
        self.assertEqual(len(docs), 6)
        self.assertIn("Extra Corpus Doc\nOnly in full corpus.", docs)

    def test_negative_boundaries_on_malformed_samples_and_max_queries(self) -> None:
        with self.assertRaises(ValueError):
            build_benchmark_slice("musique", self.musique_samples, self.musique_corpus, max_queries=0)
        with self.assertRaises(ValueError):
            get_gold_docs([{"question": "No context fields"}], "musique")
        with self.assertRaises(ValueError):
            get_gold_answers([{"question": "No answer field"}])
        with self.assertRaises(ValueError):
            get_gold_answers([{"question": "Null answer", "answer": None}])

    def test_load_dataset_files_reads_local_disk_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "sample.json").write_text(json.dumps(self.musique_samples), encoding="utf-8")
            (tmp_path / "sample_corpus.json").write_text(json.dumps(self.musique_corpus[:3]), encoding="utf-8")
            prepare_dataset_bundle("sample", tmp_path / "sample_corpus.json", tmp_path / "sample.json", tmp_path)
            samples, corpus = load_dataset_files("sample", dataset_dir=tmp_path)
            self.assertEqual(len(samples), 2)
            self.assertEqual(len(corpus), 3)

            (tmp_path / "sample_corpus.json").unlink()
            sliced_samples, empty_corpus = load_dataset_files(
                "sample",
                dataset_dir=tmp_path,
                max_queries=1,
                include_corpus=False,
            )
            self.assertEqual(len(sliced_samples), 1)
            self.assertEqual(empty_corpus, [])
            with self.assertRaises(ValueError):
                load_dataset_files("sample", dataset_dir=tmp_path, max_queries=0, include_corpus=False)

    def test_dataset_aliasing_and_registry_resolution(self) -> None:
        rejected_aliases = (
            "2wiki",
            "hotpot",
            "naturalquestions",
            "nq_rear",
            "narrative",
            "enterprise",
        )
        for alias in rejected_aliases:
            with self.assertRaises(ValueError):
                resolve_dataset_name(alias)

        expected_registry = {
            "sample": ("sample.json", "sample_corpus.json"),
            "2wikimultihopqa": ("2wikimultihopqa.json", "2wikimultihopqa_corpus.json"),
            "hotpotqa": ("hotpotqa.json", "hotpotqa_corpus.json"),
            "musique": ("musique.json", "musique_corpus.json"),
            "nq": ("nq_rear.json", "nq_rear_corpus.json"),
            "popqa": ("popqa.json", "popqa_corpus.json"),
            "lveval": ("lveval.json", "lveval_corpus.json"),
            "narrativeqa": ("narrativeqa_dev_10_doc.json", "narrativeqa_dev_10_doc_corpus.json"),
            "case_study_university": (
                "case_study_university.json",
                "case_study_university_corpus.json",
            ),
            "enterprise_rag_bench": (
                "enterprise_rag_bench_questions.json",
                "enterprise_rag_bench_corpus.json",
            ),
        }
        for name, files in expected_registry.items():
            self.assertEqual(resolve_dataset_name(name), name)
            self.assertEqual(DATASET_REGISTRY.get(name), files)

        for ds in DATASETS:
            self.assertIn(ds, DATASET_REGISTRY)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "nq_rear.json").write_text(
                json.dumps([{"question": "Q1", "reference": ["A1"], "contexts": []}]),
                encoding="utf-8",
            )
            (tmp_path / "nq_rear_corpus.json").write_text(json.dumps([]), encoding="utf-8")
            prepare_dataset_bundle("nq", tmp_path / "nq_rear_corpus.json", tmp_path / "nq_rear.json", tmp_path)
            samples, _ = load_dataset_files("nq", dataset_dir=tmp_path)
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0]["question"], "Q1")

            for alias in ("naturalquestions", "nq_rear"):
                with self.assertRaises(ValueError):
                    load_dataset_files(alias, dataset_dir=tmp_path)

    def test_distractor_padding_in_benchmark_slice(self) -> None:
        samples: list[dict[str, Any]] = [
            {
                "question": f"Question {i}?",
                "reference": [f"Ans {i}"],
                "contexts": [
                    {"title": f"Doc {i}", "text": f"Content {i}", "is_supporting": True},
                    {"title": f"Distractor {i}", "text": f"Noise {i}", "is_supporting": False},
                ],
            }
            for i in range(15)
        ]
        corpus = [{"title": c["title"], "text": c["text"]} for q in samples for c in q["contexts"]]
        doc_ids, queries, gold_docs, gold_answers = build_benchmark_slice(
            "nq", samples, corpus=corpus, max_queries=2, index_queries=4, full_corpus=False
        )
        docs = selected_texts(doc_ids, corpus)
        self.assertIsNotNone(gold_docs)
        assert gold_docs is not None
        self.assertEqual(len(queries), 2)
        self.assertEqual(len(gold_docs), 2)
        self.assertEqual(len(gold_answers), 2)
        self.assertIn("Doc 0\nContent 0", docs)
        self.assertIn("Distractor 0\nNoise 0", docs)
        self.assertIn("Doc 3\nContent 3", docs)
        self.assertIn("Distractor 3\nNoise 3", docs)
        self.assertNotIn("Doc 4\nContent 4", docs)

        doc_ids_def, queries_def, gold_docs_def, _ = build_benchmark_slice(
            "nq", samples, corpus=corpus, max_queries=10, index_queries=None, full_corpus=False
        )
        docs_def = selected_texts(doc_ids_def, corpus)
        self.assertIsNotNone(gold_docs_def)
        assert gold_docs_def is not None
        self.assertEqual(len(queries_def), 10)
        self.assertEqual(len(gold_docs_def), 10)
        self.assertIn("Doc 13\nContent 13", docs_def)
        self.assertNotIn("Doc 14\nContent 14", docs_def)

        with self.assertRaises(ValueError):
            build_benchmark_slice("nq", samples, corpus=corpus, max_queries=2, index_queries=0)

    def test_nq_gold_extraction(self) -> None:
        samples = [
            {
                "question": "who has the oldest team in the nba",
                "reference": ["Sacramento Kings", "New York Knickerbockers"],
                "answer_aliases": ["Kings", "Knicks"],
                "contexts": [
                    {
                        "title": "Eddie Gottlieb",
                        "text": "Basketball Association...",
                        "is_supporting": True,
                    },
                    {"title": "Unrelated", "text": "Some text...", "is_supporting": False},
                ],
            }
        ]
        gold_answers = get_gold_answers(samples)
        self.assertEqual(
            gold_answers,
            [["Sacramento Kings", "New York Knickerbockers", "Kings", "Knicks"]],
        )
        gold_docs = get_gold_docs(samples, "nq")
        self.assertEqual(gold_docs, [["Eddie Gottlieb\nBasketball Association..."]])

    def test_popqa_gold_extraction(self) -> None:
        samples = [
            {
                "question": "What is George Rankin's occupation?",
                "obj": "politician",
                "o_wiki_title": "Politician",
                "possible_answers": '["politician", "political leader", "political figure"]',
                "o_aliases": '["statesman"]',
                "paragraphs": [
                    {
                        "title": "George Rankin",
                        "text": "George was a politician.",
                        "is_supporting": True,
                    },
                    {"title": "Irrelevant", "text": "Irrelevant text.", "is_supporting": False},
                ],
            },
            {
                "question": "Native list possible answers test",
                "obj": "author",
                "possible_answers": ["author", "writer"],
                "paragraphs": [
                    {
                        "title": "Author Doc",
                        "paragraph_text": "Author text.",
                        "is_supporting": True,
                    },
                ],
            },
        ]
        gold_answers = get_gold_answers(samples)
        self.assertEqual(
            gold_answers[0],
            ["politician", "Politician", "political leader", "political figure", "statesman"],
        )
        self.assertEqual(gold_answers[1], ["author", "writer"])

        gold_docs = get_gold_docs(samples, "popqa")
        assert gold_docs is not None
        self.assertEqual(gold_docs[0], ["George Rankin\nGeorge was a politician."])
        self.assertEqual(gold_docs[1], ["Author Doc\nAuthor text."])

    def test_lveval_gold_extraction(self) -> None:
        samples = [
            {
                "question": "Who is Poly Ii, Duke Of Bob's paternal grandfather?",
                "answers": ["Arenas"],
                "gold_ans": "Arenas",
                "context": (
                    "### Passage 1\n"
                    "Arenas de Savoie, duc de Bob was a military commander.\n"
                    "### Passage 2\n"
                    "Formula Renault season was in 2003.\n"
                ),
            }
        ]
        gold_answers = get_gold_answers(samples)
        self.assertEqual(gold_answers, [["Arenas"]])
        self.assertIsNone(get_gold_docs(samples, "lveval"))

        corpus = [
            {"title": "", "text": text.strip()}
            for text in re.split(r"### Passage \d+\n", samples[0]["context"])
            if text.strip()
        ]
        doc_ids, queries, s_gold, _ = build_benchmark_slice(
            "lveval", samples, corpus=corpus, max_queries=1, index_queries=2, full_corpus=False
        )
        slice_docs = selected_texts(doc_ids, corpus)
        self.assertEqual(len(queries), 1)
        self.assertIsNone(s_gold)
        self.assertEqual(len(slice_docs), len(corpus))

    def test_narrativeqa_gold_extraction(self) -> None:
        samples = [
            {
                "question": "What is Mary Horowitz's job?",
                "answer": [
                    "She is a crossword writer for the Sacramento Herald.",
                    "Crossword puzzle writer.",
                ],
                "document": {
                    "id": "doc123",
                    "summary": {
                        "title": "All About Steve",
                        "text": "Mary Horowitz, a crossword puzzle writer for the Sacramento Herald.",
                    },
                },
            }
        ]
        gold_answers = get_gold_answers(samples)
        self.assertEqual(
            gold_answers,
            [["She is a crossword writer for the Sacramento Herald.", "Crossword puzzle writer."]],
        )
        self.assertIsNone(get_gold_docs(samples, "narrativeqa"))

        corpus = [
            {"idx": "doc123_0", "title": "Script", "text": "Unrelated scene."},
            {"idx": "doc123_1", "title": "Script", "text": "Mary works as a crossword writer for the Herald."},
            {"idx": "other_0", "title": "Other", "text": "Different document."},
        ]
        self.assertIsNone(get_gold_docs(samples, "narrativeqa", corpus=corpus))

        doc_ids, queries, s_gold, _ = build_benchmark_slice(
            "narrativeqa", samples, corpus, max_queries=1, index_queries=2, full_corpus=False
        )
        slice_docs = selected_texts(doc_ids, corpus)
        self.assertEqual(len(queries), 1)
        self.assertIsNone(s_gold)
        self.assertEqual(len(slice_docs), len(corpus))

    def test_enterprise_rag_bench_expected_doc_ids_slicing_and_bundle(self) -> None:
        documents = [
            {"idx": "doc-a", "title": "Spec A", "text": "Service A listens on port 8080."},
            {"idx": "doc-b", "title": "Spec B", "text": "Service B uses SQLite WAL."},
            {"idx": "distractor", "title": "Noise", "text": "Unrelated archive notes."},
        ]
        questions = [
            {
                "question_id": "q1",
                "question": "What port does Service A use?",
                "gold_answer": "8080",
                "expected_doc_ids": ["doc-a"],
            }
        ]
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            cp, qp = root / "corpus.json", root / "questions.json"
            cp.write_text(json.dumps(documents), encoding="utf-8")
            qp.write_text(json.dumps(questions), encoding="utf-8")
            bundle = prepare_dataset_bundle("enterprise_rag_bench", cp, qp, root / "outputs/datasets")
            loaded_q, loaded_c = load_dataset_files("enterprise_rag_bench", dataset_dir=root / "outputs/datasets")
            doc_ids, queries, gold_docs, gold_answers = build_benchmark_slice(
                "enterprise_rag_bench", loaded_q, loaded_c, max_queries=1
            )
            self.assertEqual(len(bundle.documents), 3)
            self.assertEqual(doc_ids, ["doc-a", "doc-b", "distractor"])
            self.assertEqual(queries, ["What port does Service A use?"])
            self.assertEqual(gold_docs, [["Spec A\nService A listens on port 8080."]])
            self.assertEqual(gold_answers, [["8080"]])

    def test_paper_baselines_and_catalog_consistency(self) -> None:
        for name in ("musique", "2wikimultihopqa", "hotpotqa", "nq", "popqa", "lveval", "narrativeqa"):
            self.assertIn(name, BASELINES)
            entry = BASELINES[name]["monkeypatched_hipporag"]
            for key in ("Recall@1", "Recall@2", "Recall@5", "ExactMatch", "F1"):
                self.assertIn(key, entry)
            self.assertGreater(entry["indexing_seconds"], 0.0)
            self.assertGreater(entry["evaluation_seconds"], 0.0)
            self.assertGreater(entry["elapsed_seconds"], 0.0)

        spec = json.loads(resources.files("yaw").joinpath("baselines.json").read_text(encoding="utf-8"))
        self.assertEqual(set(spec["datasets"]), set(DATASETS))
        self.assertEqual(spec["baselines"], BASELINES)


class EvaluationAndIntegrityTests(unittest.TestCase):
    def test_integrity_gate_catches_silent_failures(self) -> None:
        valid_records = [
            {
                "question": "Q1?",
                "answer": "Rockland County",
                "retrieved_chunk_ids": [101],
            },
            {
                "question": "Q2?",
                "answer": "Politician",
                "retrieved_chunk_ids": [],
            },
        ]
        valid_graph = {"num_passage_nodes": 5, "num_indexed_docs": 5}
        self.assertEqual(
            check_benchmark_integrity(valid_records, valid_graph, expected_doc_count=5),
            [],
        )

        # Trap 1: indexing/extraction errors and qa errors
        errs = check_benchmark_integrity(
            valid_records,
            valid_graph,
            expected_doc_count=5,
            extraction_errors=["ValueError: bad"],
            qa_errors=["TimeoutError: qa"],
        )
        self.assertTrue(any("extraction_llm" in e for e in errs))
        self.assertTrue(any("qa_llm" in e for e in errs))

        # Trap 2: passage/doc count mismatch, partial drop hidden behind multi-chunk count, and zero chunks
        errs = check_benchmark_integrity(
            valid_records,
            {"num_passage_nodes": 0, "num_indexed_docs": 0},
            expected_doc_count=5,
        )
        self.assertTrue(any("passage count" in e.lower() for e in errs))
        self.assertTrue(any("no chunks indexed" in e.lower() for e in errs))

        drop_errs = check_benchmark_integrity(
            valid_records,
            {"num_passage_nodes": 8, "num_indexed_docs": 4},
            expected_doc_count=5,
        )
        self.assertEqual(drop_errs, ["Index passage count mismatch: 4 != 5"])

        # Trap 3: all retrieved_chunk_ids/graph_seeds empty on multi-query benchmark
        no_seed_records = [{**r, "retrieved_chunk_ids": []} for r in valid_records]
        errs = check_benchmark_integrity(no_seed_records, valid_graph, expected_doc_count=5)
        self.assertTrue(any("empty retrieval results" in e for e in errs))


class PaperParityGateTests(unittest.TestCase):
    def test_popqa_slice_pads_to_at_least_two_passages_per_index_query(self) -> None:
        samples: list[dict[str, Any]] = [
            {
                "question": f"PopQA Q{i}?",
                "obj": f"Ans{i}",
                "paragraphs": [
                    {"title": f"PopDoc{i}", "text": f"PopText{i}", "is_supporting": True},
                ],
            }
            for i in range(10)
        ]
        corpus = [{"title": p["title"], "text": p["text"]} for q in samples for p in q["paragraphs"]]
        doc_ids, queries, gold_docs, _ = build_benchmark_slice(
            "popqa", samples, corpus=corpus, max_queries=2, index_queries=4, full_corpus=False
        )
        self.assertEqual(len(queries), 2)
        self.assertIsNotNone(gold_docs)
        assert gold_docs is not None
        self.assertEqual(len(gold_docs), 2)
        self.assertEqual(len(doc_ids), 8)
        docs = selected_texts(doc_ids, corpus)
        for g in gold_docs:
            self.assertTrue(set(g).issubset(set(docs)))

    def test_compare_and_aggregate_gate_reports_and_markdown_table(self) -> None:
        from scripts.verify_paper_parity_gate import (
            aggregate_gate_summary,
            build_monkeypatched_baseline_report,
            compare_engine_reports,
            format_gate_markdown,
        )

        baseline_rep = build_monkeypatched_baseline_report("2wikimultihopqa", queries_per_dataset=25)
        self.assertEqual(baseline_rep["engine"], "hipporag")
        self.assertAlmostEqual(
            baseline_rep["evaluation_metrics"]["hipporag"]["retrieval"]["Recall@5"],
            BASELINES["2wikimultihopqa"]["monkeypatched_hipporag"]["Recall@5"],
        )

        hippo_rep = {
            "engine": "hipporag",
            "status": "passed",
            "num_docs": 307,
            "num_queries": 25,
            "indexing_seconds": 42.5,
            "evaluation_seconds": 18.2,
            "elapsed_seconds": 60.7,
            "run_dir": "outputs/benchmarks/run-2wikimultihopqa-hippo",
            "index_dir": "outputs/benchmarks/indexes/2wiki_h",
            "graph": {"num_passage_nodes": 307},
            "evaluation_metrics": {
                "hipporag": {
                    "retrieval": {"Recall@1": 0.60, "Recall@2": 0.72, "Recall@5": 0.84},
                    "qa": {"ExactMatch": 0.52, "F1": 0.64},
                },
                "standard_dpr": {
                    "retrieval": {"Recall@1": 0.48, "Recall@2": 0.60, "Recall@5": 0.70},
                },
                "recognition_memory": {
                    "queries_with_graph_seeds": 22,
                    "dpr_fallback_count": 3,
                    "dpr_fallback_rate": 0.12,
                },
            },
        }
        yaw_rep = {
            "engine": "yaw",
            "status": "passed",
            "num_docs": 307,
            "num_queries": 25,
            "indexing_seconds": 38.0,
            "evaluation_seconds": 15.4,
            "elapsed_seconds": 53.4,
            "run_dir": "outputs/benchmarks/run-2wikimultihopqa-yaw",
            "index_dir": "outputs/benchmarks/indexes/2wiki_y",
            "graph": {"num_passage_nodes": 307},
            "evaluation_metrics": {
                "hipporag": {
                    "retrieval": {"Recall@1": 0.64, "Recall@2": 0.76, "Recall@5": 0.88},
                    "qa": {"ExactMatch": 0.56, "F1": 0.68},
                },
                "standard_dpr": {
                    "retrieval": {"Recall@1": 0.48, "Recall@2": 0.60, "Recall@5": 0.70},
                },
                "recognition_memory": {
                    "queries_with_graph_seeds": 24,
                    "dpr_fallback_count": 1,
                    "dpr_fallback_rate": 0.04,
                },
            },
        }

        comp = compare_engine_reports(
            "2wikimultihopqa",
            hippo_rep,
            yaw_rep,
            queries_per_dataset=25,
            index_queries=35,
            mode="eval",
        )
        self.assertEqual(comp["dataset"], "2wikimultihopqa")
        self.assertTrue(comp["gate_passed"])
        self.assertEqual(comp["gate_status"], "PASS")
        self.assertAlmostEqual(comp["delta_yaw_vs_hipporag"]["Recall@5"], 0.04)
        self.assertAlmostEqual(comp["delta_yaw_vs_hipporag"]["F1"], 0.04)
        self.assertAlmostEqual(comp["delta_yaw_vs_paper"]["Recall@5"], round(0.88 - 0.904, 4))

        summary = aggregate_gate_summary({"2wikimultihopqa": comp}, mode="eval")
        self.assertTrue(summary["overall_gate_passed"])
        self.assertEqual(summary["overall_gate_status"], "PASS")
        self.assertEqual(summary["datasets_completed"], 1)
        self.assertEqual(summary["datasets_passed"], 1)

        payload = {
            "updated_at": "2026-10-18T10:20:00Z",
            "config": {
                "datasets": ["2wikimultihopqa", "musique"],
                "queries_per_dataset": 25,
                "index_queries": 35,
                "compiled_program": ".yaw/compiled_program.json",
                "mode": "eval",
            },
            "datasets": {"2wikimultihopqa": comp},
            "summary": summary,
        }
        md = format_gate_markdown(payload)
        self.assertIn("# Pre-Epic 2 HippoRAG 2 Paper Parity Gate Report", md)
        self.assertIn("| `2wikimultihopqa` | 307 | 25 |", md)
        self.assertIn("| `musique` | — | — |", md)
        self.assertIn("**+0.0400**", md)

    def test_maxp_passage_hydration_and_retrieval_recall_parity_on_split_passages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp_path = Path(tmp_str)
            corpus_dir = tmp_path / "corpus"
            corpus_dir.mkdir()

            long_body = (
                "Montebello is an incorporated village in the town of Ramapo, Rockland County, New York. "
                "Erik Hort was born in Montebello and grew up near the Hudson Valley region.\n\n"
                "Rockland County is the southernmost county on the west side of the Hudson River in New York. "
                "The county seat of Rockland County is New City, and it borders Bergen County, New Jersey.\n\n"
                "Additional historical records confirm that Montebello was incorporated as a village in 1986 "
                "within the town of Ramapo in Rockland County."
            )
            full_gold_passage = f"Montebello\n{long_body}"
            (corpus_dir / "montebello.md").write_text(f"# Montebello\n\n{long_body}\n", encoding="utf-8")
            (corpus_dir / "albany.md").write_text(
                "# Albany\n\nAlbany is the capital city of the U.S. state of New York and seat of Albany County.\n",
                encoding="utf-8",
            )

            def semantic_embed(texts: list[str]) -> NDArray[np.float32]:
                out = []
                for t in texts:
                    low = t.lower()
                    v = np.array(
                        [
                            1.0 if "montebello" in low or "erik hort" in low else 0.1,
                            1.0 if "rockland" in low else 0.1,
                            0.5,
                            0.2,
                        ],
                        dtype=np.float32,
                    )
                    v /= np.linalg.norm(v)
                    out.append(v)
                return np.vstack(out)

            cfg = YawConfig(
                workspace_root=tmp_path,
                corpus_root=corpus_dir,
                db_path=tmp_path / ".yaw" / "yaw.db",
                lock_path=tmp_path / ".yaw" / "yaw.db.lock",
                state_dir=tmp_path / ".yaw" / "coco_lmdb",
                log_path=tmp_path / ".yaw" / "yaw.log",
                status_path=tmp_path / ".yaw" / "status.json",
                chunk_size=180,
                chunk_overlap=20,
                min_chunk_size=30,
            )
            session = start_pipeline(cfg, embed_fn=semantic_embed, live=False)
            try:
                update_pipeline_once(session)
            finally:
                close_pipeline(session)

            assert cfg.db_path is not None
            with contextlib.closing(open_readonly_db(cfg.db_path)) as conn:
                idx = build_retrieval_index(
                    conn,
                    expected_embedding_model=cfg.embedding_model,
                    expected_task_mode=cfg.embedding_task_mode,
                )
                self.assertGreaterEqual(idx.size, 3)
                q_vec = semantic_embed(["task: question answering | query: Which county is Erik Hort born in?"])[0]
                q_text = "Which county is Erik Hort's birthplace part of?"
                res_pool = search_index(idx, q_vec, query_text=q_text, conn=conn, top_k=5, pool_k=50)
                ranked_legacy = search_index(idx, q_vec, query_text=q_text, conn=conn, top_k=50).ranked
                self.assertEqual(res_pool.ranked, ranked_legacy)
                ranked = res_pool.ranked
                hits = hydrate_ranked_passages(idx, ranked, corpus_dir, top_k=5)
                self.assertGreaterEqual(len(hits), 2)
                overall, _ = RetrievalRecall().calculate_metric_scores(
                    gold_docs=[[full_gold_passage]],
                    retrieved_docs=[[hp.text for hp in hits]],
                    k_list=[1, 2, 5],
                )
                self.assertEqual(overall["Recall@1"], 1.0)

    def test_benchmark_bundle_snapshot_is_complete_and_source_aligned(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            cp, qp = root / "corpus.json", root / "questions.json"
            cp.write_text(json.dumps([{"idx": "a", "title": "A", "text": "Grounded text"}]), encoding="utf-8")
            qp.write_text(json.dumps([{"question": "Evaluation only", "answer": "Secret"}]), encoding="utf-8")
            bundle = prepare_dataset_bundle("sample", cp, qp, root / "prepared")
            doc = bundle.documents[0]
            self.assertEqual(doc.path, "documents/a.md")
            raw_md = (bundle.root / doc.path).read_text(encoding="utf-8")
            self.assertEqual(raw_md, "# A\n\nGrounded text\n")
            pieces = chunk_file_content(f"{bundle.dataset}/{doc.path}", raw_md, min_chunk_size=1)
            self.assertEqual(len(pieces), 1)
            piece = pieces[0]
            ptr = ChunkPointer(
                chunk_id=piece.chunk_id,
                rel_path=piece.rel_path,
                source_hash=piece.source_hash,
                start_line=piece.start_line,
                end_line=piece.end_line,
                char_start=piece.char_start,
                char_end=piece.char_end,
                ast_island_start=piece.ast_island_start,
                ast_island_end=piece.ast_island_end,
                title=piece.title,
            )
            self.assertIn("/documents/", ptr.rel_posix_path)
            self.assertEqual(hydrate_passage(ptr, bundle.root.parent).text, piece.body)


if __name__ == "__main__":
    unittest.main()
