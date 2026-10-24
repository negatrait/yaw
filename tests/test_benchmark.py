"""Executable specifications for HippoRAG benchmark slicing, evaluation, and integrity gates."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from benchmark import (
    ALIASES,
    DATASET_REGISTRY,
    DATASETS,
    MONKEYPATCHED_HIPPORAG_BASELINES,
    PAPER_BASELINES,
    TEXT_STOPWORDS,
    build_benchmark_slice,
    check_benchmark_integrity,
    get_gold_answers,
    get_gold_docs,
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
                "musique",
                self.musique_samples,
                self.musique_corpus,
                max_queries=max_q,
                full_corpus=False,
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
                "sample",
                dataset_dir=tmp_path,
                max_queries=1,
                include_corpus=False,
            )
            self.assertEqual(len(sliced_samples), 1)
            self.assertEqual(empty_corpus, [])
            with self.assertRaises(ValueError):
                load_dataset_files("sample", dataset_dir=tmp_path, max_queries=0, include_corpus=False)

    def test_dataset_aliasing_and_registry_resolution(self):
        expected_aliases = {
            "2wiki": "2wikimultihopqa",
            "hotpot": "hotpotqa",
            "naturalquestions": "nq",
            "nq_rear": "nq",
            "narrative": "narrativeqa",
            "enterprise": "enterprise_rag_bench",
        }
        for alias, canonical in expected_aliases.items():
            self.assertEqual(ALIASES.get(alias), canonical)

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
                "enterprise_rag_bench.json",
                "enterprise_rag_bench_corpus.json",
            ),
        }
        for name, files in expected_registry.items():
            self.assertEqual(DATASET_REGISTRY.get(name), files)

        for ds in DATASETS:
            self.assertIn(ds, DATASET_REGISTRY)

        # Verify load_dataset_files with aliases resolves files correctly
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "nq_rear.json").write_text(
                json.dumps([{"question": "Q1", "reference": ["A1"], "contexts": []}]),
                encoding="utf-8",
            )
            (tmp_path / "nq_rear_corpus.json").write_text(json.dumps([]), encoding="utf-8")
            samples, _ = load_dataset_files("naturalquestions", dataset_dir=tmp_path)
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0]["question"], "Q1")

            samples_nq, _ = load_dataset_files("nq_rear", dataset_dir=tmp_path)
            self.assertEqual(len(samples_nq), 1)

    def test_distractor_padding_in_benchmark_slice(self):
        samples = [
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
        # Explicit index_queries: max_queries=2, index_queries=4 -> pools passages from samples 0..3
        docs, queries, gold_docs, gold_answers = build_benchmark_slice(
            "nq", samples, corpus=[], max_queries=2, index_queries=4, full_corpus=False
        )
        self.assertEqual(len(queries), 2)
        self.assertEqual(len(gold_docs), 2)
        self.assertEqual(len(gold_answers), 2)
        # Docs must contain passages from samples 0..3 (both supporting and distractors)
        self.assertIn("Doc 0\nContent 0", docs)
        self.assertIn("Distractor 0\nNoise 0", docs)
        self.assertIn("Doc 3\nContent 3", docs)
        self.assertIn("Distractor 3\nNoise 3", docs)
        # Sample 4 must not be in docs
        self.assertNotIn("Doc 4\nContent 4", docs)

        # Default index_queries: max_queries=10, index_queries=None -> defaults to int(10 * 1.4) = 14
        docs_def, queries_def, gold_docs_def, _ = build_benchmark_slice(
            "nq", samples, corpus=[], max_queries=10, index_queries=None, full_corpus=False
        )
        self.assertEqual(len(queries_def), 10)
        self.assertEqual(len(gold_docs_def), 10)
        # Sample 13 must be in docs (pooled up to index 14)
        self.assertIn("Doc 13\nContent 13", docs_def)
        # Sample 14 must not be in docs
        self.assertNotIn("Doc 14\nContent 14", docs_def)

        # Negative boundary on index_queries
        with self.assertRaises(ValueError):
            build_benchmark_slice("nq", samples, corpus=[], max_queries=2, index_queries=0)

    def test_nq_gold_extraction(self):
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

    def test_popqa_gold_extraction(self):
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
        self.assertEqual(gold_docs[0], ["George Rankin\nGeorge was a politician."])
        self.assertEqual(gold_docs[1], ["Author Doc\nAuthor text."])

    def test_lveval_gold_extraction(self):
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

        gold_docs = get_gold_docs(samples, "lveval")
        self.assertEqual(gold_docs, [["\nArenas de Savoie, duc de Bob was a military commander."]])

        # Test bounded slice generation for lveval
        slice_docs, queries, s_gold, _ = build_benchmark_slice(
            "lveval", samples, corpus=[], max_queries=1, index_queries=2, full_corpus=False
        )
        self.assertEqual(len(queries), 1)
        self.assertTrue(set(gold_docs[0]).issubset(set(slice_docs)))

        # Test top-2 capping when multiple passages match
        multi_passage_sample = [
            {
                "question": "Who is Duke Bob?",
                "gold_ans": "Bob",
                "answers": ["Bob"],
                "context": (
                    "### Passage 1\n"
                    "Bob is a famous duke and military leader.\n"
                    "### Passage 2\n"
                    "Bob was born in 1800.\n"
                    "### Passage 3\n"
                    "Bob ruled the province peacefully.\n"
                ),
            }
        ]
        top2_gold = get_gold_docs(multi_passage_sample, "lveval")
        self.assertEqual(len(top2_gold[0]), 2)
        # Passage 1 has "bob", "duke" (both answer and question match, highest rank)
        self.assertEqual(top2_gold[0][0], "\nBob is a famous duke and military leader.")

    def test_narrativeqa_gold_extraction(self):
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
        gold_docs = get_gold_docs(samples, "narrativeqa")
        self.assertEqual(
            gold_docs,
            [["All About Steve\nMary Horowitz, a crossword puzzle writer for the Sacramento Herald."]],
        )

        # Test with corpus matching chunks by doc_id and answer keywords
        corpus = [
            {"idx": "doc123_0", "title": "Script", "text": "Unrelated scene."},
            {
                "idx": "doc123_1",
                "title": "Script",
                "text": "Mary works as a crossword writer for the Herald.",
            },
            {"idx": "other_0", "title": "Other", "text": "Different document."},
        ]
        corpus_gold = get_gold_docs(samples, "narrativeqa", corpus=corpus)
        self.assertEqual(corpus_gold, [["Script\nMary works as a crossword writer for the Herald."]])

        # Test bounded slice generation for narrativeqa
        slice_docs, queries, s_gold, _ = build_benchmark_slice(
            "narrativeqa", samples, corpus, max_queries=1, index_queries=2, full_corpus=False
        )
        self.assertEqual(len(queries), 1)
        self.assertTrue(set(corpus_gold[0]).issubset(set(slice_docs)))

        # Test top-2 capping when multiple chunks match
        multi_chunk_corpus = [
            {
                "idx": "doc123_0",
                "title": "Script",
                "text": "Mary Horowitz is a crossword writer for the Sacramento Herald.",
            },
            {"idx": "doc123_1", "title": "Script", "text": "Crossword puzzle writer in the city."},
            {"idx": "doc123_2", "title": "Script", "text": "Mary has a job."},
            {"idx": "doc123_3", "title": "Script", "text": "Unrelated line."},
        ]
        multi_gold = get_gold_docs(samples, "narrativeqa", corpus=multi_chunk_corpus)
        self.assertEqual(len(multi_gold[0]), 2)
        self.assertEqual(
            multi_gold[0][0],
            "Script\nMary Horowitz is a crossword writer for the Sacramento Herald.",
        )

    def test_paper_baselines_contain_all_required_datasets(self):
        for name in (
            "musique",
            "2wikimultihopqa",
            "hotpotqa",
            "nq",
            "popqa",
            "lveval",
            "narrativeqa",
        ):
            self.assertIn(name, PAPER_BASELINES)

        # NQ
        self.assertEqual(PAPER_BASELINES["nq"]["hipporag2_llama3.3_70b_nvembed_v2"]["Recall@5"], 0.780)
        self.assertEqual(PAPER_BASELINES["nq"]["hipporag2_llama3.3_70b_nvembed_v2"]["ExactMatch"], 0.485)
        self.assertEqual(PAPER_BASELINES["nq"]["hipporag2_llama3.3_70b_nvembed_v2"]["F1"], 0.633)
        self.assertEqual(PAPER_BASELINES["nq"]["standard_nvembed_v2_llama3.3_70b"]["Recall@5"], 0.754)
        self.assertEqual(PAPER_BASELINES["nq"]["standard_nvembed_v2_llama3.3_70b"]["F1"], 0.619)
        self.assertEqual(PAPER_BASELINES["nq"]["hipporag_reproduced"]["Recall@5"], 0.444)
        self.assertEqual(PAPER_BASELINES["nq"]["hipporag_reproduced"]["F1"], 0.553)

        # PopQA
        self.assertEqual(PAPER_BASELINES["popqa"]["hipporag2_llama3.3_70b_nvembed_v2"]["Recall@5"], 0.517)
        self.assertEqual(PAPER_BASELINES["popqa"]["hipporag2_llama3.3_70b_nvembed_v2"]["ExactMatch"], 0.490)
        self.assertEqual(PAPER_BASELINES["popqa"]["hipporag2_llama3.3_70b_nvembed_v2"]["F1"], 0.562)
        self.assertEqual(PAPER_BASELINES["popqa"]["standard_nvembed_v2_llama3.3_70b"]["Recall@5"], 0.510)
        self.assertEqual(PAPER_BASELINES["popqa"]["standard_nvembed_v2_llama3.3_70b"]["F1"], 0.557)
        self.assertEqual(PAPER_BASELINES["popqa"]["hipporag_reproduced"]["Recall@5"], 0.538)
        self.assertEqual(PAPER_BASELINES["popqa"]["hipporag_reproduced"]["F1"], 0.559)

        # LV-Eval
        self.assertEqual(PAPER_BASELINES["lveval"]["hipporag2_llama3.3_70b_nvembed_v2"]["ExactMatch"], 0.080)
        self.assertEqual(PAPER_BASELINES["lveval"]["hipporag2_llama3.3_70b_nvembed_v2"]["F1"], 0.129)
        self.assertEqual(PAPER_BASELINES["lveval"]["standard_nvembed_v2_llama3.3_70b"]["F1"], 0.098)
        self.assertEqual(PAPER_BASELINES["lveval"]["hipporag_reproduced"]["F1"], 0.084)

        # NarrativeQA
        self.assertEqual(PAPER_BASELINES["narrativeqa"]["hipporag2_llama3.3_70b_nvembed_v2"]["ExactMatch"], 0.180)
        self.assertEqual(PAPER_BASELINES["narrativeqa"]["hipporag2_llama3.3_70b_nvembed_v2"]["F1"], 0.259)
        self.assertEqual(PAPER_BASELINES["narrativeqa"]["standard_nvembed_v2_llama3.3_70b"]["F1"], 0.257)
        self.assertEqual(PAPER_BASELINES["narrativeqa"]["hipporag_reproduced"]["F1"], 0.163)

    def test_monkeypatched_hipporag_baselines_contain_all_required_datasets(self):
        for name in (
            "musique",
            "2wikimultihopqa",
            "hotpotqa",
            "nq",
            "popqa",
            "lveval",
            "narrativeqa",
        ):
            self.assertIn(name, MONKEYPATCHED_HIPPORAG_BASELINES)
            entry = MONKEYPATCHED_HIPPORAG_BASELINES[name]
            self.assertIn("Recall@1", entry)
            self.assertIn("Recall@2", entry)
            self.assertIn("Recall@5", entry)
            self.assertIn("ExactMatch", entry)
            self.assertIn("F1", entry)
            self.assertGreater(entry["indexing_seconds"], 0.0)
            self.assertGreater(entry["evaluation_seconds"], 0.0)
            self.assertGreater(entry["elapsed_seconds"], 0.0)

    def test_enterprise_rag_bench_10_category_stratification_and_distractors(self):
        from collections import Counter

        from yaw.daemon.cli import _load_dataset_splits

        spec = importlib.util.spec_from_file_location(
            "download_datasets",
            Path(__file__).resolve().parent.parent / "scripts" / "download-datasets.py",
        )
        assert spec is not None and spec.loader is not None
        dl_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dl_mod)

        quotas = dl_mod.ENTERPRISE_CATEGORY_QUOTAS
        self.assertEqual(len(quotas), 10)
        self.assertEqual(sum(tr for tr, _ in quotas.values()), 150)
        self.assertEqual(sum(va for _, va in quotas.values()), 100)

        synthetic_questions = []
        synthetic_docs = []
        tools = ["slack", "github", "google_drive", "confluence", "jira"]
        q_idx = 0
        for cat, (n_tr, n_va) in quotas.items():
            for _i in range(n_tr + n_va + 4):
                qid = f"q_{q_idx}"
                tool = tools[q_idx % len(tools)]
                gold_id = f"gold_{q_idx}"
                synthetic_docs.append(
                    {
                        "doc_id": gold_id,
                        "source_type": tool,
                        "title": f"Gold Doc {q_idx}",
                        "content": f"Supporting enterprise content for {qid} in {cat}.",
                    }
                )
                synthetic_questions.append(
                    {
                        "question_id": qid,
                        "question": f"Enterprise question {q_idx} ({cat})?",
                        "gold_answer": f"Answer {q_idx}",
                        "answer_facts": [f"Fact {q_idx}"],
                        "question_type": cat,
                        "source_types": [tool],
                        "expected_doc_ids": [] if cat == "Info Not Found" else [gold_id],
                    }
                )
                q_idx += 1

        for d_idx in range(800):
            tool = tools[d_idx % len(tools)]
            synthetic_docs.append(
                {
                    "doc_id": f"dist_{d_idx}",
                    "source_type": tool,
                    "title": f"Distractor {d_idx}",
                    "content": f"Hard distractor passage {d_idx} for {tool}.",
                }
            )

        with tempfile.TemporaryDirectory() as tmp_str:
            tmp_root = Path(tmp_str)
            ds_dir = tmp_root / "outputs" / "datasets"
            q_path, c_path = dl_mod.download_enterprise_rag_bench(
                ds_dir,
                distractor_factor=2.0,
                seed=42,
                force=True,
                questions_rows=synthetic_questions,
                documents_rows=synthetic_docs,
            )
            self.assertTrue(q_path.exists())
            self.assertTrue(c_path.exists())

            queries_data = json.loads(q_path.read_text(encoding="utf-8"))
            self.assertEqual(len(queries_data), 250)
            train_slice = queries_data[:150]
            val_slice = queries_data[150:]

            train_counts = Counter(q["question_type"] for q in train_slice)
            val_counts = Counter(q["question_type"] for q in val_slice)
            for cat, (exp_tr, exp_va) in quotas.items():
                self.assertEqual(train_counts[cat], exp_tr)
                self.assertEqual(val_counts[cat], exp_va)

            for q in queries_data:
                supp = [p for p in q["paragraphs"] if p["is_supporting"]]
                dists = [p for p in q["paragraphs"] if not p["is_supporting"]]
                supp_ids = {p["idx"] for p in supp}
                dist_ids = {p["idx"] for p in dists}
                self.assertTrue(supp_ids.isdisjoint(dist_ids))
                if q["question_type"] == "Info Not Found":
                    self.assertEqual(len(supp), 0)
                    self.assertIn("Insufficient information", q["answer_aliases"])
                else:
                    self.assertEqual(len(supp), 1)
                self.assertEqual(len(dists), 2)

            trainset, valset, corpus_docs = _load_dataset_splits(
                ["enterprise"],
                workspace_root=tmp_root,
                max_queries=250,
                train_queries=150,
                query_offset=0,
            )
            self.assertEqual(len(trainset), 150)
            assert valset is not None
            self.assertEqual(len(valset), 100)
            self.assertGreater(len(corpus_docs), 300)
            for ex in trainset[:5] + valset[:5]:
                self.assertTrue(hasattr(ex, "item_docs"))
                self.assertGreaterEqual(len(ex.item_docs), 2)
                self.assertIn("item_docs", ex.inputs())

    def test_text_stopwords(self):
        self.assertGreater(len(TEXT_STOPWORDS), 50)
        for expected in ("the", "and", "that", "with", "from", "they"):
            self.assertIn(expected, TEXT_STOPWORDS)


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
            valid_records,
            {"num_passage_nodes": 4, "num_extracted_triples": 0},
            expected_doc_count=5,
        )
        self.assertTrue(any("passage count" in e.lower() for e in errs))
        self.assertTrue(any("no triples" in e.lower() for e in errs))

        # Trap 3: all graph_seeds empty on multi-query benchmark
        no_seed_records = [{**r, "graph_seeds": []} for r in valid_records]
        errs = check_benchmark_integrity(no_seed_records, valid_graph, expected_doc_count=5)
        self.assertTrue(any("graph_seeds" in e for e in errs))


class PaperParityGateTests(unittest.TestCase):
    def test_popqa_slice_pads_to_at_least_two_passages_per_index_query(self):
        samples = [
            {
                "question": f"PopQA Q{i}?",
                "obj": f"Ans{i}",
                "paragraphs": [
                    {"title": f"PopDoc{i}", "text": f"PopText{i}", "is_supporting": True},
                ],
            }
            for i in range(10)
        ]
        docs, queries, gold_docs, _ = build_benchmark_slice(
            "popqa", samples, corpus=[], max_queries=2, index_queries=4, full_corpus=False
        )
        self.assertEqual(len(queries), 2)
        self.assertEqual(len(gold_docs), 2)
        # 4 index_queries * 2 = 8 passages padded from subsequent samples
        self.assertEqual(len(docs), 8)
        for g in gold_docs:
            self.assertTrue(set(g).issubset(set(docs)))

    def test_compare_and_aggregate_gate_reports_and_markdown_table(self):
        from scripts.verify_paper_parity_gate import (
            aggregate_gate_summary,
            build_monkeypatched_baseline_report,
            compare_engine_reports,
            format_gate_markdown,
        )

        baseline_rep = build_monkeypatched_baseline_report("2wiki", queries_per_dataset=25)
        self.assertEqual(baseline_rep["engine"], "hipporag")
        self.assertAlmostEqual(
            baseline_rep["evaluation_metrics"]["hipporag"]["retrieval"]["Recall@5"],
            MONKEYPATCHED_HIPPORAG_BASELINES["2wikimultihopqa"]["Recall@5"],
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
            "2wiki",
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

    def test_maxp_passage_hydration_and_retrieval_recall_parity_on_split_json_passages(self):
        import json
        import tempfile
        from pathlib import Path

        import numpy as np

        from benchmark import BENCHMARK_JSON_BLACKLIST, BENCHMARK_JSON_WHITELIST
        from scripts.evaluation import RetrievalRecall as ScriptRetrievalRecall
        from yaw.core import OpenIERecord
        from yaw.daemon.graph.engine import YawGraphEngine
        from yaw.daemon.graph.inference.embedding import resolve_embedding_profile
        from yaw.daemon.ingestion import sync_corpus_once

        with tempfile.TemporaryDirectory() as tmp_str:
            tmp_path = Path(tmp_str)
            corpus_dir = tmp_path / "corpus"
            state_dir = tmp_path / "state"
            corpus_dir.mkdir()
            state_dir.mkdir()

            long_body = (
                "Montebello is an incorporated village in the town of Ramapo, Rockland County, New York. "
                "Erik Hort was born in Montebello and grew up near the Hudson Valley region.\n\n"
                "Rockland County is the southernmost county on the west side of the Hudson River in New York. "
                "The county seat of Rockland County is New City, and it borders Bergen County, New Jersey.\n\n"
                "Additional historical records confirm that Montebello was incorporated as a village in 1986 "
                "within the town of Ramapo in Rockland County."
            )
            full_gold_passage = f"Montebello\n{long_body}"
            distractor_passage = (
                "Albany\nAlbany is the capital city of the U.S. state of New York and seat of Albany County."
            )

            dataset_payload = [
                {
                    "id": "musique_maxp_1",
                    "question": "Which county is Erik Hort's birthplace part of?",
                    "answer": "Rockland County",
                    "paragraphs": [
                        {
                            "idx": 0,
                            "title": "Montebello",
                            "paragraph_text": long_body,
                            "is_supporting": True,
                        },
                        {
                            "idx": 1,
                            "title": "Albany",
                            "paragraph_text": distractor_passage.split("\n", 1)[1],
                            "is_supporting": False,
                        },
                    ],
                }
            ]
            (corpus_dir / "musique.json").write_text(
                json.dumps(dataset_payload, indent=2),
                encoding="utf-8",
            )

            delta = sync_corpus_once(
                corpus_dir,
                state_dir,
                chunk_size=180,
                chunk_overlap=20,
                min_chunk_size=30,
                json_keys_whitelist=BENCHMARK_JSON_WHITELIST,
                json_keys_blacklist=BENCHMARK_JSON_BLACKLIST,
            )
            # Long passage is split into multiple sub-chunks, plus 1 distractor chunk
            self.assertGreaterEqual(len(delta.added_chunks), 3)
            montebello_subchunks = [(ptr, txt) for ptr, txt in delta.added_chunks if ptr.title == "Montebello"]
            self.assertGreaterEqual(len(montebello_subchunks), 2)
            # All Montebello sub-chunks share the exact same parent record AST island
            ast_islands = {
                (ptr.rel_posix_path, ptr.ast_island_start, ptr.ast_island_end) for ptr, _ in montebello_subchunks
            }
            self.assertEqual(len(ast_islands), 1)

            profile = resolve_embedding_profile(task_mode="distinct-titled")
            formatted_doc_inputs: list[str] = []

            class TitleCheckingEmbedder:
                def __init__(self, prof):
                    self.profile = prof
                    self.dimensions = 4

                def encode(self, texts, *, instruction=""):
                    if not instruction:
                        for t in texts:
                            formatted_doc_inputs.append(self.profile.format(t, query=False))
                    out = []
                    for t in texts:
                        v = np.array(
                            [
                                1.0 if "montebello" in t.lower() or "erik hort" in t.lower() else 0.1,
                                1.0 if "rockland" in t.lower() else 0.1,
                                0.5,
                                0.2,
                            ],
                            dtype=np.float32,
                        )
                        v /= np.linalg.norm(v)
                        out.append(v)
                    return np.vstack(out)

                def batch_encode(self, texts, *, instruction="", batch_size=64):
                    return self.encode(texts, instruction=instruction)

            class MinimalProgram:
                context_window_tokens = 32768
                max_output_tokens = 4096

                def set_retrieval_hook(self, hook):
                    self.hook = hook

                def extract_openie(self, chunk_id, text):
                    if "montebello" in text.lower():
                        return OpenIERecord(
                            chunk_id=chunk_id,
                            entities=("erik hort", "montebello", "rockland county"),
                            triples=(
                                ("erik hort", "born in", "montebello"),
                                ("montebello", "part of", "rockland county"),
                            ),
                        )
                    return OpenIERecord(
                        chunk_id=chunk_id,
                        entities=("albany", "new york"),
                        triples=(("albany", "capital of", "new york"),),
                    )

                def extract_query_intent(self, query):
                    return query

                def filter_triples(self, query, candidates):
                    return [0] if candidates else []

                def judge_synonyms(self, candidates):
                    return []

                def heal_shacl_violations(self, violations, focus_context="", *args, **kwargs):
                    return []

            engine = YawGraphEngine(
                db_path=tmp_path / "yaw.db",
                workspace_root=corpus_dir,
                program=MinimalProgram(),
                embedder=TitleCheckingEmbedder(profile),
                retrieval_top_k=5,
            )
            try:
                engine.index_chunks(delta.added_chunks)
                # Verify zero titled chunks degraded to 'title: none'
                self.assertTrue(len(formatted_doc_inputs) >= len(delta.added_chunks))
                for fmt_str in formatted_doc_inputs[: len(delta.added_chunks)]:
                    self.assertFalse(
                        fmt_str.startswith("title: none"),
                        f"Expected AST title in formatted embedding input, got: {fmt_str!r}",
                    )
                    self.assertTrue(
                        fmt_str.startswith(("title: Montebello | text:", "title: Albany | text:")),
                    )

                hits, _ = engine.retrieve_with_seeds("Which county is Erik Hort's birthplace part of?", top_k=5)
                dpr_hits = engine.retrieve_dpr("Which county is Erik Hort's birthplace part of?", top_k=5)
                # MaxP deduplicates the multiple Montebello sub-chunks into 1 full passage hit
                self.assertEqual(len(hits), 2)
                self.assertEqual(len(dpr_hits), 2)
                self.assertEqual(hits[0][2], full_gold_passage)
                self.assertEqual(dpr_hits[0][2], full_gold_passage)

                overall, _ = ScriptRetrievalRecall().calculate_metric_scores(
                    gold_docs=[[full_gold_passage]],
                    retrieved_docs=[[text for _, _, text in hits]],
                    k_list=[1, 2, 5],
                )
                self.assertEqual(overall["Recall@1"], 1.0)
            finally:
                engine.close()


if __name__ == "__main__":
    unittest.main()
