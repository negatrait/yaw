"""Verify 100% byte-for-byte prompt parity and parser parity against upstream HippoRAG."""

import json
import unittest
from copy import deepcopy
from typing import Any

from hipporag.information_extraction.openie_openai import (
    _extract_ner_from_response as upstream_extract_ner,
)
from hipporag.prompts import PromptTemplateManager
from hipporag.rerank import DSPyFilter
from hipporag.utils.llm_utils import filter_invalid_triples as upstream_filter_invalid_triples
from hipporag.utils.misc_utils import (
    compute_mdhash_id as upstream_compute_mdhash_id,
)
from hipporag.utils.misc_utils import (
    text_processing as upstream_text_processing,
)

from yaw.daemon.graph.inference.parsers import (
    compute_mdhash_id,
    extract_named_entities,
    extract_triples,
    filter_invalid_triples,
    filter_reranked_facts,
    repair_truncated_json,
    text_processing,
)
from yaw.daemon.graph.inference.templates import (
    render_fact_filter_messages,
    render_ner_messages,
    render_rag_qa_messages,
    render_triple_messages,
)


class PromptsAndParsersParityTests(unittest.TestCase):
    def setUp(self):
        self.ptm = PromptTemplateManager()

    def test_ner_messages_byte_for_byte_parity(self):
        """Verify render_ner_messages matches upstream PromptTemplateManager byte-for-byte."""
        test_passages = [
            "Alan Turing was an English mathematician and computer scientist.",
            "Radio City is an FM radio station in India.\nIt operates in English and Hindi.",
            "Special characters: 'quotes', \"double quotes\", $dollar signs, and \\backslashes.",
        ]
        for passage in test_passages:
            upstream_msgs = self.ptm.render(name="ner", passage=passage)
            our_msgs = render_ner_messages(passage)
            self.assertEqual(
                our_msgs,
                upstream_msgs,
                f"NER message mismatch for passage: {passage!r}",
            )

    def test_triple_messages_byte_for_byte_parity(self):
        """Verify render_triple_messages matches upstream PromptTemplateManager byte-for-byte."""
        test_cases = [
            (
                "Alan Turing was born in Maida Vale, London.",
                ["Alan Turing", "Maida Vale", "London"],
            ),
            (
                "DeepMind is based in London and was acquired by Google in 2014.",
                ["DeepMind", "London", "Google", "2014"],
            ),
        ]
        for passage, entities in test_cases:
            named_entity_json = json.dumps({"named_entities": entities})
            upstream_msgs = self.ptm.render(
                name="triple_extraction",
                passage=passage,
                named_entity_json=named_entity_json,
            )
            our_msgs = render_triple_messages(passage, entities)
            self.assertEqual(
                our_msgs,
                upstream_msgs,
                f"Triple message mismatch for passage: {passage!r}",
            )

    def test_rag_qa_messages_byte_for_byte_parity(self):
        """Verify render_rag_qa_messages matches upstream PromptTemplateManager byte-for-byte."""
        test_prompts = [
            (
                "Wikipedia Title: Southampton\nSouthampton is a port city.\n\n"
                "Question: Where is Southampton located?\nThought: "
            ),
            (
                "Wikipedia Title: Neville A. Stanton\nNeville Stanton works at University of Southampton.\n\n"
                "Question: Who employs Neville Stanton?\nThought: "
            ),
        ]
        for prompt_user in test_prompts:
            upstream_msgs = self.ptm.render(name="rag_qa_musique", prompt_user=prompt_user)
            our_msgs = render_rag_qa_messages(prompt_user)
            self.assertEqual(
                our_msgs,
                upstream_msgs,
                f"RAG QA message mismatch for prompt_user: {prompt_user!r}",
            )

    def test_fact_filter_messages_byte_for_byte_parity(self):
        """Verify render_fact_filter_messages matches upstream DSPyFilter prompt structure."""
        candidate_facts = [
            ("apple", "founded in", "california"),
            ("apple", "created", "iphone"),
            ("steve jobs", "co-founded", "apple"),
        ]
        query = "Where was Apple founded?"

        class DummyHippoRAG:
            class global_config:
                rerank_dspy_file_path = None
            class extraction_llm:
                infer = None

        upstream_filter = DSPyFilter(DummyHippoRAG())
        fact_before_filter = {"fact": [list(item) for item in candidate_facts]}
        expected_messages = deepcopy(upstream_filter.message_template)
        expected_messages.append({
            "role": "user",
            "content": upstream_filter.one_input_template.format(
                question=query, fact_before_filter=json.dumps(fact_before_filter)
            ),
        })

        our_messages = render_fact_filter_messages(query, candidate_facts)
        self.assertEqual(our_messages, expected_messages)

    def test_compute_mdhash_id_parity(self):
        """Verify compute_mdhash_id matches upstream."""
        test_strings = [
            ("", ""),
            ("hello world", ""),
            ("test content", "chunk-"),
            ("Radio City is India's first private FM radio station", "ent-"),
        ]
        for text, prefix in test_strings:
            self.assertEqual(
                compute_mdhash_id(text, prefix),
                upstream_compute_mdhash_id(text, prefix),
            )

    def test_text_processing_parity(self):
        """Verify text_processing matches upstream entity normalization."""
        test_inputs = [
            "Apple Inc.",
            "Radio-City (FM Station) #1!",
            "   Multiple   Spaces   and   Tabs\t\n",
            "123 numbers & Symbols_underscore",
            ["Apple Inc.", "Google LLC", "DeepMind (UK)"],
        ]
        for inp in test_inputs:
            self.assertEqual(
                text_processing(inp),
                upstream_text_processing(inp),
            )

    def test_filter_invalid_triples_parity(self):
        """Verify filter_invalid_triples matches upstream."""
        input_triples: list[Any] = [
            ["Radio City", "located in", "India"],
            ["invalid", "only two"],
            ["invalid", "too", "many", "items"],
            ["Radio City", "located in", "India"],  # duplicate
            ["Radio City", "plays songs in", "Hindi"],
            [123, 456, 789],  # non-string coercion
        ]
        expected = upstream_filter_invalid_triples(input_triples)
        actual = filter_invalid_triples(input_triples)
        self.assertEqual(actual, expected)

    def test_extract_named_entities_parity(self):
        """Verify extract_named_entities extracts entities cleanly."""
        # Standard JSON object response
        resp1 = json.dumps({"named_entities": ["Alan Turing", "London", "1912"]})
        self.assertEqual(
            extract_named_entities(resp1),
            tuple(upstream_extract_ner(resp1)),
        )

        # Bare JSON list response (common with local instruction-tuned models)
        resp2 = json.dumps(["Alan Turing", "London", "1912"])
        self.assertEqual(
            extract_named_entities(resp2),
            tuple(upstream_extract_ner(resp2)),
        )

        # Markdown fenced code block response
        resp3 = "```json\n" + json.dumps(["Alan Turing", "London"]) + "\n```"
        self.assertEqual(
            extract_named_entities(resp3),
            tuple(upstream_extract_ner(resp3)),
        )

    def test_extract_triples_parity(self):
        """Verify extract_triples parses and filters triples."""
        resp = json.dumps({
            "triples": [
                ["Alan Turing", "born in", "London"],
                ["London", "located in", "United Kingdom"],
                ["invalid", "pair"],
                ["Alan Turing", "born in", "London"],  # duplicate
            ]
        })
        triples = extract_triples(resp)
        self.assertEqual(
            triples,
            (
                ("Alan Turing", "born in", "London"),
                ("London", "located in", "United Kingdom"),
            ),
        )

    def test_filter_reranked_facts_parity(self):
        """Verify filter_reranked_facts matches upstream DSPyFilter candidate selection."""
        candidates = [
            ("imperial river", "is located in", "florida"),
            ("imperial river", "is a river in", "united states"),
            ("amaradia", "flows through", "ro ia de amaradia"),
            ("unrelated fact", "has predicate", "object"),
        ]

        llm_response = (
            "[[ ## fact_after_filter ## ]]\n"
            '{"fact":[["imperial river","is located in","florida"],["amaradia","flows through","ro ia de amaradia"]]}\n\n'
            "[[ ## completed ## ]]"
        )

        # Upstream DSPyFilter parsing:
        class DummyHippoRAG:
            class global_config:
                rerank_dspy_file_path = None
            class extraction_llm:
                infer = None

        dspy_filter = DSPyFilter(DummyHippoRAG())
        dspy_filter.parse_filter(llm_response)

        # Expected indices: candidate 0 and candidate 2
        matched_indices = filter_reranked_facts(candidates, llm_response)
        self.assertEqual(matched_indices, [0, 2])

        # Test with max_facts limit
        self.assertEqual(filter_reranked_facts(candidates, llm_response, max_facts=1), [0])

    def test_repair_truncated_json_handles_openie_boundaries(self):
        """Verify repair_truncated_json fixes max_tokens truncations without losing valid elements."""
        truncated_triples = (
            '{"triples": [\n'
            '  ["A", "is", "B"],\n'
            '  ["C", "located in", "D"],\n'
            '  ["E", "broken triple without close'
        )
        repaired = repair_truncated_json(truncated_triples, is_triples=True)
        data = json.loads(repaired)
        self.assertEqual(
            data["triples"],
            [["A", "is", "B"], ["C", "located in", "D"]],
        )

        truncated_ner = '{"named_entities": ["Alan Turing", "London", "United King'
        repaired_ner = repair_truncated_json(truncated_ner, is_triples=False)
        data_ner = json.loads(repaired_ner)
        self.assertEqual(data_ner["named_entities"], ["Alan Turing", "London"])

    def test_dspy_signatures_and_default_demos(self):
        """Verify all 4 dspy.Signatures and seeded dspy.Example demos."""
        import dspy

        from yaw.daemon.graph.inference.prompts import (
            FACT_FILTER_DEMOS,
            NER_DEMOS,
            QA_DEMOS,
            TRIPLE_DEMOS,
            AnswerQuestion,
            ExtractEntities,
            ExtractTriples,
            FactList,
            FilterFacts,
        )

        self.assertTrue(issubclass(ExtractEntities, dspy.Signature))
        self.assertTrue(issubclass(ExtractTriples, dspy.Signature))
        self.assertTrue(issubclass(FilterFacts, dspy.Signature))
        self.assertTrue(issubclass(AnswerQuestion, dspy.Signature))

        self.assertEqual(list(ExtractEntities.input_fields.keys()), ["passage"])
        self.assertEqual(list(ExtractEntities.output_fields.keys()), ["named_entities"])
        self.assertEqual(list(ExtractTriples.input_fields.keys()), ["passage", "named_entities"])
        self.assertEqual(list(ExtractTriples.output_fields.keys()), ["triples"])
        self.assertEqual(list(FilterFacts.input_fields.keys()), ["question", "fact_before_filter"])
        self.assertEqual(list(FilterFacts.output_fields.keys()), ["fact"])
        self.assertEqual(list(AnswerQuestion.input_fields.keys()), ["context", "question"])
        self.assertEqual(list(AnswerQuestion.output_fields.keys()), ["answer"])

        self.assertEqual(len(NER_DEMOS), 1)
        self.assertEqual(len(TRIPLE_DEMOS), 1)
        self.assertEqual(len(FACT_FILTER_DEMOS), 10)
        self.assertIsInstance(FACT_FILTER_DEMOS[0].fact_before_filter, str)
        self.assertIsInstance(FACT_FILTER_DEMOS[0].fact, list)
        self.assertIsInstance(FactList(fact=FACT_FILTER_DEMOS[0].fact), FactList)
        self.assertEqual(len(QA_DEMOS), 1)

    def test_hipporag_program_openie_fact_filter_and_cot_qa(self):
        """Verify HippoRAGProgram executes OpenIE, fact filtering, and CoT QA via dspy.JSONAdapter (pure JSON output)."""
        import dspy
        from dspy.utils import DummyLM

        from yaw.daemon.graph.inference.program import HippoRAGProgram

        program = HippoRAGProgram(qa_mode="cot")
        lm = DummyLM(
            [
                {
                    "named_entities": [
                        ["Alan Turing", "English mathematician and computer scientist"],
                        ["London", "Capital of the United Kingdom"],
                        ["Alan Turing", "English mathematician and computer scientist"],
                    ]
                },
                {
                    "triples": [
                        {
                            "sub": "Alan Turing",
                            "pred": "born in",
                            "obj": "London",
                            "def": "Alan Turing was born in London.",
                        },
                        {
                            "sub": "London",
                            "pred": "capital of",
                            "obj": ["United Kingdom", "Sovereign country in Europe"],
                            "def": "London is the capital of the United Kingdom.",
                        },
                        ["invalid", "only_two"],
                        {
                            "sub": "Alan Turing",
                            "pred": "born in",
                            "obj": "London",
                            "def": "Alan Turing was born in London.",
                        },
                    ]
                },
                {
                    "fact": [
                        ["imperial river", "is located in", "florida"],
                        ["amaradia", "flows through", "ro ia de amaradia"],
                    ]
                },
                {
                    "reasoning": "Neville Stanton works at the University of Southampton, founded in 1862.",
                    "answer": "1862",
                },
            ],
            adapter=dspy.JSONAdapter(),
        )

        record = program.extract_openie(
            "chunk-1",
            "Alan Turing was born in London, the capital of the United Kingdom.",
            lm=lm,
        )
        self.assertEqual(record.chunk_id, "chunk-1")
        self.assertEqual(
            record.entities, ("Alan Turing", "London", "United Kingdom")
        )
        self.assertEqual(
            record.entity_definitions,
            (
                ("Alan Turing", "English mathematician and computer scientist"),
                ("London", "Capital of the United Kingdom"),
                ("United Kingdom", "Sovereign country in Europe"),
            ),
        )
        self.assertEqual(
            record.triples,
            (
                ("Alan Turing", "born in", "London"),
                ("London", "capital of", "United Kingdom"),
            ),
        )
        self.assertEqual(len(record.structured_triples), 2)
        self.assertEqual(
            record.structured_triples[0].subj_def,
            "English mathematician and computer scientist",
        )
        self.assertEqual(
            record.structured_triples[0].obj_def,
            "Capital of the United Kingdom",
        )
        self.assertEqual(
            record.structured_triples[1].obj_def,
            "Sovereign country in Europe",
        )

        candidates = [
            ("imperial river", "is located in", "florida"),
            ("imperial river", "is a river in", "united states"),
            ("amaradia", "flows through", "ro ia de amaradia"),
        ]
        selected_indices = program.filter_facts(
            "Where are Imperial River and Amaradia located?",
            candidates,
            lm=lm,
        )
        self.assertEqual(selected_indices, [0, 2])

        qa_pred = program.answer_question(
            "University of Southampton was founded in 1862.",
            "When was it founded?",
            mode="cot",
            lm=lm,
        )
        self.assertEqual(qa_pred.answer, "1862")
        self.assertIn("1862", qa_pred.reasoning)

    def test_hipporag_program_rlm_qa_executes_in_local_interpreter(self):
        """Verify HippoRAGProgram RLM mode executes real Python code inside dspy.LocalInterpreter."""
        from dspy.utils import DummyLM

        from yaw.daemon.graph.inference.program import HippoRAGProgram

        program = HippoRAGProgram(qa_mode="rlm", rlm_max_iters=4, rlm_max_llm_calls=5)
        lm = DummyLM([
            {
                "reasoning": "Search the context in Python for the founding year and submit.",
                "code": (
                    "import re\n"
                    "m = re.search(r'founded in (\\d{4})', context)\n"
                    "SUBMIT(answer=m.group(1) if m else 'unknown')"
                ),
            }
        ])

        rlm_pred = program.answer_question(
            "The University of Southampton was founded in 1862 and received its Royal Charter in 1952.",
            "When was the University of Southampton founded?",
            mode="rlm",
            lm=lm,
        )
        self.assertEqual(rlm_pred.answer, "1862")
        self.assertEqual(len(rlm_pred.trajectory), 1)

    def test_hipporag_program_save_and_load_gepa_state(self):
        """Verify HippoRAGProgram saves and reloads mutated GEPA instructions and demos cleanly."""
        from pathlib import Path
        from tempfile import TemporaryDirectory

        import dspy

        from yaw.daemon.graph.inference.program import HippoRAGProgram

        with TemporaryDirectory() as tmp_dir:
            state_path = Path(tmp_dir) / "gepa_compiled.json"
            prog1 = HippoRAGProgram()
            sig1: Any = prog1.ner.signature
            prog1.ner.signature = sig1.with_instructions(
                "GEPA optimized NER instruction."
            )
            prog1.ner.demos = [
                dspy.Example(
                    passage="Custom GEPA demo passage.",
                    named_entities=["GEPA"],
                ).with_inputs("passage")
            ]
            prog1.save(str(state_path))
            self.assertTrue(state_path.exists())

            prog2 = HippoRAGProgram(compiled_path=state_path)
            sig2: Any = prog2.ner.signature
            self.assertEqual(
                sig2.instructions,
                "GEPA optimized NER instruction.",
            )
            self.assertEqual(len(prog2.ner.demos), 1)
            self.assertEqual(prog2.ner.demos[0]["named_entities"], ["GEPA"])

    def test_hipporag_program_gbnf_grammar_forwarding(self):
        """Verify HippoRAGProgram forwards pure-JSON static and dynamic GBNF grammars via extra_body."""
        import dspy
        from dspy.utils import DummyLM

        from yaw.daemon.graph.inference.grammars import (
            DSPY_NER_GBNF_GRAMMAR,
            DSPY_QA_COT_GBNF_GRAMMAR,
            DSPY_TRIPLES_GBNF_GRAMMAR,
            build_dspy_rerank_grammar,
            build_dspy_triples_grammar,
        )
        from yaw.daemon.graph.inference.program import HippoRAGProgram

        program = HippoRAGProgram(enable_gbnf=True)
        stage1_pairs = [
            ["Alan Turing", "English mathematician"],
            ["London", "Capital of the United Kingdom"],
        ]
        lm = DummyLM(
            [
                {"named_entities": stage1_pairs},
                {
                    "triples": [
                        {
                            "sub": "Alan Turing",
                            "pred": "born in",
                            "obj": "London",
                            "def": "Alan Turing was born in London.",
                        }
                    ]
                },
                {"fact": [["alan turing", "born in", "london"]]},
                {"reasoning": "Born in London.", "answer": "London"},
            ],
            adapter=dspy.JSONAdapter(),
        )

        program.extract_openie("c-1", "Alan Turing was born in London.", lm=lm)
        self.assertNotIn("[[ ##", DSPY_NER_GBNF_GRAMMAR)
        self.assertNotIn("[[ ##", DSPY_TRIPLES_GBNF_GRAMMAR)
        self.assertNotIn("[[ ##", DSPY_QA_COT_GBNF_GRAMMAR)
        self.assertEqual(
            lm.history[0]["kwargs"].get("extra_body"),
            {"grammar": DSPY_NER_GBNF_GRAMMAR},
        )
        self.assertEqual(
            lm.history[1]["kwargs"].get("extra_body"),
            {"grammar": build_dspy_triples_grammar(None, stage1_pairs)},
        )

        candidates = [
            ("alan turing", "born in", "london"),
            ("cinderella", "attended", "royal ball"),
        ]
        selected = program.filter_facts("Where was Alan Turing born?", candidates, lm=lm)
        self.assertEqual(selected, [0])
        expected_rerank_grammar = build_dspy_rerank_grammar(candidates, max_facts=4)
        self.assertNotIn("[[ ##", expected_rerank_grammar)
        self.assertIn('"[\\"alan turing\\", \\"born in\\", \\"london\\"]"', expected_rerank_grammar)
        self.assertEqual(
            lm.history[2]["kwargs"].get("extra_body"),
            {"grammar": expected_rerank_grammar},
        )

        program.answer_question("Alan Turing was born in London.", "Where?", mode="cot", lm=lm, use_gbnf=True)
        self.assertEqual(
            lm.history[3]["kwargs"].get("extra_body"),
            {"grammar": DSPY_QA_COT_GBNF_GRAMMAR},
        )

    def test_sizecoding_gbnf_caps_and_gepa_pareto_metric(self):
        """Verify sizecoding GBNF caps on forward() and compute_sizecoding_pareto_score for dspy.GEPA."""
        import dspy
        from dspy.teleprompt.gepa.gepa_utils import _extract_score_and_objective_scores
        from dspy.utils import DummyLM

        from yaw.daemon.graph.inference.grammars import (
            build_dspy_ner_grammar,
            build_dspy_qa_cot_grammar,
            build_dspy_triples_grammar,
            compute_sizecoding_pareto_score,
        )
        from yaw.daemon.graph.inference.program import HippoRAGProgram

        program = HippoRAGProgram(
            enable_gbnf=True,
            gbnf_max_entities=3,
            gbnf_max_triples=2,
            gbnf_max_facts=2,
            gbnf_max_reasoning_chars=180,
        )
        stage1_pairs = [
            ["Alan Turing", "English mathematician"],
            ["London", "Capital of the United Kingdom"],
        ]
        lm = DummyLM(
            [
                {"named_entities": stage1_pairs},
                {
                    "triples": [
                        {
                            "sub": "Alan Turing",
                            "pred": "born in",
                            "obj": "London",
                            "def": "Alan Turing was born in London.",
                        }
                    ]
                },
                {"reasoning": "Born in London.", "answer": "London"},
            ],
            adapter=dspy.JSONAdapter(),
        )
        program.extract_openie("c-1", "Alan Turing was born in London.", lm=lm)
        self.assertEqual(
            lm.history[0]["kwargs"].get("extra_body"),
            {"grammar": build_dspy_ner_grammar(3)},
        )
        self.assertEqual(
            lm.history[1]["kwargs"].get("extra_body"),
            {"grammar": build_dspy_triples_grammar(2, stage1_pairs)},
        )

        # Calling program(...) (GEPA's forward pass) automatically attaches the capped CoT GBNF grammar
        with dspy.context(lm=lm):
            pred = program(context="Alan Turing was born in London.", question="Where?")
        self.assertEqual(pred.answer, "London")
        self.assertIn("char{1,180}", build_dspy_qa_cot_grammar(180))
        self.assertEqual(
            lm.history[2]["kwargs"].get("extra_body"),
            {"grammar": build_dspy_qa_cot_grammar(180)},
        )

        # 1. Below reference floor -> gate-free GM_good / GM_bad (zero hard-gate cliff)
        violated = compute_sizecoding_pareto_score(
            0.60,
            tokens=20,
            min_baseline_recall=0.75,
            pred_name="triples",
            missed_items=["doc-2"],
        )
        self.assertGreater(violated.score, 0.0)
        self.assertAlmostEqual(violated.pareto_raw, 0.60 / 20)
        self.assertIn("Below reference floor", violated.feedback)
        self.assertIn("Gate-free GM_good/GM_bad", violated.feedback)

        # 2. Above reference floor -> rewards higher recall / fewer tokens / fewer SHACL violations
        bloated = compute_sizecoding_pareto_score(
            0.80,
            tokens=400,
            min_baseline_recall=0.75,
            target_tokens=100.0,
        )
        sizecoded = compute_sizecoding_pareto_score(
            0.80,
            tokens=80,
            min_baseline_recall=0.75,
            target_tokens=100.0,
        )
        higher_recall_same_tokens = compute_sizecoding_pareto_score(
            0.80,
            tokens=20,
            min_baseline_recall=0.75,
            target_tokens=100.0,
        )
        self.assertGreater(sizecoded.score, bloated.score)
        self.assertGreater(sizecoded.pareto_raw, bloated.pareto_raw)
        self.assertGreater(higher_recall_same_tokens.score, violated.score)
        self.assertAlmostEqual(sizecoded.pareto_raw, 0.80 / 80)

        # Verify dspy.GEPA's internal DspyAdapter extracts both scalar score and multi-objective dict
        scalar_score, obj_scores = _extract_score_and_objective_scores(sizecoded)
        self.assertAlmostEqual(scalar_score, sizecoded.score)
        self.assertEqual(
            set(obj_scores.keys()),
            {
                "recall_at_5",
                "pareto_raw",
                "efficiency",
                "gm_good",
                "gm_bad",
                "shacl_conforms",
            },
        )
        self.assertAlmostEqual(obj_scores["recall_at_5"], 0.80)
        self.assertAlmostEqual(obj_scores["shacl_conforms"], 1.0)

        # 3. Smoothed multi-rank gradient: when EM=0.0 or Recall@1=0.0, improving Recall@2/@5/@10
        # or F1 MUST strictly increase scalar score (never collapse to flat 0.10).
        em_zero_low_recall = compute_sizecoding_pareto_score(
            0.50,
            recall_at_1=0.0,
            recall_at_2=0.50,
            recall_at_10=0.50,
            recall_at_20=1.0,
            em=0.0,
            f1=0.40,
            tokens=100,
        )
        em_zero_high_recall = compute_sizecoding_pareto_score(
            1.0,
            recall_at_1=0.50,
            recall_at_2=1.0,
            recall_at_10=1.0,
            recall_at_20=1.0,
            em=0.0,
            f1=0.40,
            tokens=100,
        )
        self.assertGreater(em_zero_low_recall.quality_numerator, 0.0)
        self.assertGreater(em_zero_low_recall.score, 0.10)
        self.assertGreater(em_zero_high_recall.score, em_zero_low_recall.score)
        self.assertNotIn("preserve 100% of gold retrieval bridges", em_zero_high_recall.feedback)

        # Full active predictor surface visibility for GEPA (never hides OpenIE/EDC/SHACL predictors behind _retrieval_hook)
        direct_prog = HippoRAGProgram(execution_mode="direct", qa_mode="cot")
        direct_names = [name for name, _ in direct_prog.named_predictors()]
        self.assertEqual(
            direct_names,
            [
                "ner",
                "triples",
                "entity_definer",
                "synonym_judge",
                "shacl_repair",
                "fact_filter",
                "query_intent",
                "qa_cot.predict",
            ],
        )
        direct_prog.set_retrieval_hook(lambda q: ("ctx", ("c1",), (1.0,)))
        hooked_names = [name for name, _ in direct_prog.named_predictors()]
        self.assertEqual(
            hooked_names,
            [
                "ner",
                "triples",
                "entity_definer",
                "synonym_judge",
                "shacl_repair",
                "fact_filter",
                "query_intent",
                "qa_cot.predict",
            ],
        )

    def test_heterogeneous_mistral_orchestrator_and_qwen_gbnf_rlm_across_all_stages(self):
        """Verify execution_mode='rlm' runs Mistral NPU orchestrator (zero GBNF) + Qwen GPU worker (pure-JSON GBNF) on every stage."""
        import dspy
        from dspy.utils import DummyLM

        from yaw.daemon.graph.inference.grammars import (
            DSPY_NER_GBNF_GRAMMAR,
            DSPY_QA_COT_GBNF_GRAMMAR,
            build_dspy_rerank_grammar,
            build_dspy_triples_grammar,
            lm_supports_gbnf,
        )
        from yaw.daemon.graph.inference.program import HippoRAGProgram

        # Mistral orchestrator on Ryzen AI NPU/OGA (ChatAdapter, NO GBNF allowed)
        mistral_lm = DummyLM([
            {
                "reasoning": "Delegate entity extraction to GBNF-constrained Qwen GPU worker.",
                "code": (
                    "ents = qwen_extract_entities(passage=passage)\n"
                    "SUBMIT(named_entities=ents)"
                ),
            },
            {
                "reasoning": "Delegate triple extraction to GBNF-constrained Qwen GPU worker.",
                "code": (
                    "trips = qwen_extract_triples(passage=passage, named_entities=named_entities)\n"
                    "SUBMIT(triples=trips)"
                ),
            },
            {
                "reasoning": "Parse candidates and delegate fact filtering to GBNF-constrained Qwen GPU worker.",
                "code": (
                    "import json\n"
                    "cands = json.loads(fact_before_filter)['fact']\n"
                    "kept = qwen_filter_facts(question=question, candidate_facts=cands)\n"
                    "SUBMIT(fact=kept)"
                ),
            },
            {
                "reasoning": "Delegate multi-hop CoT reasoning to GBNF-constrained Qwen GPU worker.",
                "code": (
                    "res = qwen_answer_cot(context=context, question=question)\n"
                    "SUBMIT(answer=res['answer'])"
                ),
            },
        ])
        setattr(mistral_lm, "_yaw_recipe", "ryzenai-llm")
        setattr(mistral_lm, "_yaw_supports_gbnf", False)
        self.assertFalse(lm_supports_gbnf(mistral_lm))

        stage1_pairs = [
            ["Alan Turing", "English mathematician"],
            ["London", "Capital of the United Kingdom"],
        ]
        # Qwen worker on llama.cpp Vulkan GPU (JSONAdapter, pure-JSON GBNF enabled)
        qwen_lm = DummyLM(
            [
                {"named_entities": stage1_pairs},
                {
                    "triples": [
                        {
                            "sub": "Alan Turing",
                            "pred": "born in",
                            "obj": "London",
                            "def": "Alan Turing was born in London.",
                        }
                    ]
                },
                {"fact": [["alan turing", "born in", "london"]]},
                {"reasoning": "Alan Turing was born in London.", "answer": "London"},
            ],
            adapter=dspy.JSONAdapter(),
        )
        setattr(qwen_lm, "_yaw_recipe", "llamacpp")
        setattr(qwen_lm, "_yaw_supports_gbnf", True)
        self.assertTrue(lm_supports_gbnf(qwen_lm))

        program = HippoRAGProgram(
            execution_mode="rlm",
            enable_gbnf=True,
            orchestrator_lm=mistral_lm,
            rlm_sub_lm=qwen_lm,
            rlm_max_iters=3,
            rlm_max_llm_calls=8,
        )

        # 1 & 2: OpenIE (NER + Triples) via Mistral RLM -> Qwen GBNF worker
        record = program.extract_openie("c-1", "Alan Turing was born in London.")
        self.assertEqual(record.entities, ("Alan Turing", "London"))
        self.assertEqual(record.triples, (("Alan Turing", "born in", "London"),))

        # 3: Fact filter via Mistral RLM -> Qwen GBNF worker
        candidates = [
            ("alan turing", "born in", "london"),
            ("cinderella", "attended", "royal ball"),
        ]
        selected = program.filter_facts("Where was Alan Turing born?", candidates)
        self.assertEqual(selected, [0])

        # 4: Multi-hop QA via Mistral RLM -> Qwen GBNF worker
        qa_pred = program("Alan Turing was born in London.", "Where was Alan Turing born?")
        self.assertEqual(qa_pred.answer, "London")

        # Verify Mistral NPU orchestrator received 4 RLM calls and ZERO GBNF grammars
        self.assertEqual(len(mistral_lm.history), 4)
        for entry in mistral_lm.history:
            self.assertIsNone(entry["kwargs"].get("extra_body"))

        # Verify Qwen GPU worker received 4 inner calls, ALL with pure-JSON GBNF grammars
        self.assertEqual(len(qwen_lm.history), 4)
        self.assertEqual(
            qwen_lm.history[0]["kwargs"].get("extra_body"),
            {"grammar": DSPY_NER_GBNF_GRAMMAR},
        )
        self.assertEqual(
            qwen_lm.history[1]["kwargs"].get("extra_body"),
            {"grammar": build_dspy_triples_grammar(None, stage1_pairs)},
        )
        self.assertEqual(
            qwen_lm.history[2]["kwargs"].get("extra_body"),
            {"grammar": build_dspy_rerank_grammar(candidates, max_facts=4)},
        )
        self.assertEqual(
            qwen_lm.history[3]["kwargs"].get("extra_body"),
            {"grammar": DSPY_QA_COT_GBNF_GRAMMAR},
        )

    def test_rlm_positional_tool_calls_and_mistral_submit_quirks(self):
        """Verify _PositionalSafeRLM handles real Mistral NPU outputs: positional tool args, dict-wrapped SUBMIT, JSON-string triples, and leading ']\\n'."""
        import dspy
        from dspy.utils import DummyLM

        from yaw.daemon.graph.inference.program import HippoRAGProgram

        # Real-silicon failure patterns observed on Mistral-7B-Instruct-v0.3-Hybrid (ryzenai-llm):
        # 1. Positional tool calls: qwen_extract_entities(passage) instead of keyword args
        # 2. JSON-serialized triples passed to SUBMIT(triples=json.dumps(...))
        # 3. Raw fact_before_filter JSON string passed positionally + dict-wrapped SUBMIT({"fact": kept})
        # 4. Single-key dict SUBMIT({"when": ...}) with leading "]\n" ChatAdapter bracket artifact
        mistral_lm = DummyLM([
            {
                "reasoning": "]\nExtract entities positionally.",
                "code": (
                    "ents = qwen_extract_entities(passage)\n"
                    "SUBMIT({'named_entities': ents})"
                ),
            },
            {
                "reasoning": "]\nExtract triples positionally and JSON-dump.",
                "code": (
                    "import json\n"
                    "trips = qwen_extract_triples(passage, named_entities)\n"
                    "SUBMIT(triples=json.dumps(trips))"
                ),
            },
            {
                "reasoning": "]\nFilter facts passing fact_before_filter positionally and dict-wrapping SUBMIT.",
                "code": (
                    "kept = qwen_filter_facts(question, fact_before_filter)\n"
                    "SUBMIT({'fact': kept})"
                ),
            },
            {
                "reasoning": "]\nAnswer question positionally and submit dict.",
                "code": (
                    "res = qwen_answer_cot(context, question)\n"
                    "SUBMIT({'when': ']\\n' + res['answer']})"
                ),
            },
        ])
        setattr(mistral_lm, "_yaw_recipe", "ryzenai-llm")
        setattr(mistral_lm, "_yaw_supports_gbnf", False)

        qwen_lm = DummyLM(
            [
                {
                    "named_entities": [
                        ["Radio City", "Private FM radio station in India"],
                        ["Bangalore", "Capital city of Karnataka"],
                        ["Abraham Thomas", "Chief executive of Radio City"],
                    ]
                },
                {
                    "triples": [
                        {
                            "sub": "Radio City",
                            "pred": "started in",
                            "obj": "Bangalore",
                            "def": "Radio City was started in Bangalore.",
                        }
                    ]
                },
                {"fact": [["Radio City", "started in", "Bangalore"]]},
                {"reasoning": "]\nStarted on 3 July 2001.", "answer": "]\n3 July 2001"},
            ],
            adapter=dspy.JSONAdapter(),
        )
        setattr(qwen_lm, "_yaw_recipe", "llamacpp")
        setattr(qwen_lm, "_yaw_supports_gbnf", True)

        program = HippoRAGProgram(
            execution_mode="rlm",
            enable_gbnf=True,
            orchestrator_lm=mistral_lm,
            rlm_sub_lm=qwen_lm,
            rlm_max_iters=2,
        )

        record = program.extract_openie(
            "c-pos",
            "Radio City was started on 3 July 2001 in Bangalore by Abraham Thomas.",
        )
        self.assertEqual(record.entities, ("Radio City", "Bangalore", "Abraham Thomas"))
        self.assertEqual(record.triples, (("Radio City", "started in", "Bangalore"),))

        kept = program.filter_facts(
            "Where was Radio City started?",
            [("Radio City", "started in", "Bangalore"), ("Apple", "founded by", "Steve Jobs")],
        )
        self.assertEqual(kept, [0])

        qa_pred = program(
            "Radio City was started on 3 July 2001 in Bangalore by Abraham Thomas.",
            "When was Radio City started?",
        )
        self.assertEqual(qa_pred.answer, "3 July 2001")
        # Every stage must succeed on iteration 1 without falling back to self.extract
        self.assertEqual(len(mistral_lm.history), 4)
        self.assertEqual(len(qwen_lm.history), 4)

    @staticmethod
    def _probe_live_lemonade_models() -> set[str]:
        import urllib.request

        try:
            with urllib.request.urlopen("http://localhost:13305/api/v1/health", timeout=2.0) as resp:
                if resp.status != 200:
                    return set()
                payload = json.loads(resp.read().decode("utf-8"))
                loaded = payload.get("all_models_loaded", [])
                return {
                    str(item.get("model_name"))
                    for item in loaded
                    if isinstance(item, dict) and item.get("model_name")
                }
        except Exception:
            return set()

    def test_live_silicon_direct_qwen_gbnf_and_heterogeneous_mistral_qwen_rlm(self):
        """Run un-mocked end-to-end HippoRAGProgram against live Lemonade Server (Qwen GPU GBNF + Mistral NPU RLM)."""
        loaded_models = self._probe_live_lemonade_models()
        required = {"Qwen3-4B-Instruct-2507-GGUF", "Mistral-7B-Instruct-v0.3-Hybrid"}
        if not required.issubset(loaded_models):
            self.skipTest(
                f"Live Lemonade Server models {sorted(required)} not loaded (found {sorted(loaded_models)})"
            )

        from yaw.daemon.graph.inference.chat import create_dspy_lm, create_dspy_rlm_pair
        from yaw.daemon.graph.inference.program import HippoRAGProgram

        passage = (
            "Radio City is India's first private FM radio station and was started on "
            "3 July 2001 in Bangalore by Abraham Thomas."
        )
        candidates = [
            ("Radio City", "started in", "Bangalore"),
            ("Apple", "founded by", "Steve Jobs"),
        ]

        # 1. Live Direct Mode on Qwen3-4B-Instruct-2507-GGUF (llama.cpp GPU + pure-JSON GBNF)
        qwen_direct_lm = create_dspy_lm(
            "Qwen3-4B-Instruct-2507-GGUF",
            base_url="http://localhost:13305/api/v1",
            role="extraction",
            max_tokens=512,
            cache=True,
        )
        direct_prog = HippoRAGProgram(
            execution_mode="direct",
            enable_gbnf=True,
        )
        direct_record = direct_prog.extract_openie("c-live-direct", passage, lm=qwen_direct_lm)
        self.assertIn("Radio City", direct_record.entities)
        self.assertIn("Bangalore", direct_record.entities)
        self.assertTrue(len(direct_record.triples) >= 1)

        direct_kept = direct_prog.filter_facts(
            "Where was Radio City started?", candidates, lm=qwen_direct_lm
        )
        self.assertEqual(direct_kept, [0])

        direct_qa = direct_prog.answer_question(
            passage, "When was Radio City started?", lm=qwen_direct_lm
        )
        self.assertIn("2001", direct_qa.answer)
        self.assertEqual(len(qwen_direct_lm.history), 4)
        for call_entry in qwen_direct_lm.history:
            self.assertIn("grammar", (call_entry["kwargs"].get("extra_body") or {}))

        # 2. Live RLM Mode: Mistral-7B-Instruct-v0.3-Hybrid (Ryzen AI NPU) Orchestrator +
        #    Qwen3-4B-Instruct-2507-GGUF (llama.cpp GPU + pure-JSON GBNF) Worker
        orch_lm, sub_lm = create_dspy_rlm_pair(
            base_url="http://localhost:13305/api/v1",
            cache=True,
        )
        rlm_prog = HippoRAGProgram(
            execution_mode="rlm",
            enable_gbnf=True,
            orchestrator_lm=orch_lm,
            rlm_sub_lm=sub_lm,
            rlm_max_iters=3,
        )
        rlm_record = rlm_prog.extract_openie("c-live-rlm", passage)
        self.assertIn("Radio City", rlm_record.entities)
        self.assertIn("Bangalore", rlm_record.entities)
        self.assertTrue(len(rlm_record.triples) >= 1)

        rlm_kept = rlm_prog.filter_facts("Where was Radio City started?", candidates)
        self.assertEqual(rlm_kept, [0])

        rlm_qa = rlm_prog.answer_question(passage, "When was Radio City started?")
        self.assertIn("2001", rlm_qa.answer)
        self.assertFalse(rlm_qa.answer.startswith("]"))

        # Verify 1-iteration RLM completion across all 4 stages (4 Mistral NPU calls, 4 Qwen GPU GBNF calls)
        self.assertEqual(len(orch_lm.history), 4)
        for orch_entry in orch_lm.history:
            self.assertIsNone(orch_entry["kwargs"].get("extra_body"))
        self.assertEqual(len(sub_lm.history), 4)
        for sub_entry in sub_lm.history:
            self.assertIn("grammar", (sub_entry["kwargs"].get("extra_body") or {}))

    def test_leaf_modularization_and_inference_facade_integrity(self):
        """Verify program, grammars, and prompts are cleanly modularized and accessible via facade."""
        import yaw.daemon.graph.inference.prompts as prompts_module
        from yaw.daemon.graph.inference import (
            HippoRAGProgram as FacadeHippoRAGProgram,
        )
        from yaw.daemon.graph.inference import (
            build_dspy_ner_grammar as FacadeBuildNer,
        )
        from yaw.daemon.graph.inference import (
            compute_sizecoding_pareto_score as FacadePareto,
        )
        from yaw.daemon.graph.inference.grammars import (
            build_dspy_ner_grammar as GrammarBuildNer,
        )
        from yaw.daemon.graph.inference.grammars import (
            compute_sizecoding_pareto_score as GrammarPareto,
        )
        from yaw.daemon.graph.inference.program import HippoRAGProgram as ProgramHippoRAGProgram

        self.assertIs(ProgramHippoRAGProgram, FacadeHippoRAGProgram)
        self.assertIs(GrammarBuildNer, FacadeBuildNer)
        self.assertIs(GrammarPareto, FacadePareto)

        # Comfort cushions off: prompts.py does NOT re-export program or grammars
        self.assertFalse(hasattr(prompts_module, "HippoRAGProgram"))
        self.assertFalse(hasattr(prompts_module, "build_dspy_ner_grammar"))
        self.assertFalse(hasattr(prompts_module, "compute_sizecoding_pareto_score"))

    def test_instruction_proposal_gbnf_grammar_and_reflection_lm_contract(self):
        """TDD Contract: instruction proposal GBNF grammar and reflection RLM temperatures."""
        from yaw.daemon.graph.inference import (
            DSPY_INSTRUCTION_PROPOSAL_GBNF_GRAMMAR,
            build_dspy_instruction_proposal_grammar,
            create_dspy_lm,
            create_dspy_rlm_pair,
        )

        # 1. Happy path across bounded character limits
        for max_chars in (64, 512, 1500, 3000):
            grammar = build_dspy_instruction_proposal_grammar(max_chars=max_chars)
            self.assertIn('root ::= "{\\"new_instruction\\": " instruction-str "}"', grammar)
            self.assertIn(f"char{{1,{max_chars}}}", grammar)
            self.assertNotIn("named_entities", grammar)
            self.assertNotIn("triples", grammar)
            self.assertNotIn("answer", grammar)

        # 2. Boundary envelope: None or non-positive max_chars returns unbounded grammar
        for unbounded in (None, 0, -10):
            self.assertEqual(
                build_dspy_instruction_proposal_grammar(max_chars=unbounded),
                DSPY_INSTRUCTION_PROPOSAL_GBNF_GRAMMAR,
            )

        # 3. Reflection LM temperature & GBNF capability flags
        mistral_ref = create_dspy_lm("Mistral-7B-Instruct-v0.3-Hybrid", role="reflection")
        qwen_ref = create_dspy_lm("Qwen3-4B-Instruct-2507-GGUF", role="reflection_worker")
        self.assertEqual(mistral_ref.kwargs["temperature"], 0.8)
        self.assertEqual(mistral_ref.kwargs["max_tokens"], 4096)
        self.assertFalse(getattr(mistral_ref, "_yaw_supports_gbnf", True))

        self.assertEqual(qwen_ref.kwargs["temperature"], 0.7)
        self.assertEqual(qwen_ref.kwargs["max_tokens"], 4096)
        self.assertEqual(qwen_ref.kwargs["frequency_penalty"], 0.3)
        self.assertTrue(getattr(qwen_ref, "_yaw_supports_gbnf", False))

        orch_ref, worker_ref = create_dspy_rlm_pair(reflection=True)
        self.assertEqual(orch_ref.kwargs["temperature"], 0.8)
        self.assertEqual(worker_ref.kwargs["temperature"], 0.7)
        self.assertFalse(getattr(orch_ref, "_yaw_supports_gbnf", True))
        self.assertTrue(getattr(worker_ref, "_yaw_supports_gbnf", False))

    def test_rlm_instruction_proposer_contract_and_boundary_envelope(self):
        """TDD Contract: RLMInstructionProposer executes Outer Mistral RLM -> Inner Qwen RLM -> Qwen GBNF synthesis."""
        import dspy
        from dspy.utils.dummies import DummyLM

        from yaw.daemon.graph.inference import (
            DSPY_INSTRUCTION_PROPOSAL_GBNF_GRAMMAR,
            RLMInstructionProposer,
        )

        # Negative boundary envelope: invalid max_instruction_chars raises ValueError
        with self.assertRaises(ValueError):
            RLMInstructionProposer(max_instruction_chars=0)

        mistral_lm = DummyLM([
            {
                "reasoning": "Inspect examples_with_feedback in Python and delegate to Qwen inner RLM.",
                "code": (
                    "improved = qwen_propose_instruction(current_instruction, examples_with_feedback)\n"
                    "SUBMIT(new_instruction=improved)"
                ),
            }
        ])
        setattr(mistral_lm, "_yaw_recipe", "ryzenai-llm")
        setattr(mistral_lm, "_yaw_supports_gbnf", False)

        qwen_lm = DummyLM(
            [
                {
                    "reasoning": "Distill examples_with_feedback in Python and call GBNF synthesize_instruction.",
                    "code": (
                        "digest = [{'Inputs': {'passage': ex['Inputs']['passage'][:200]}, "
                        "'Generated Outputs': ex['Generated Outputs'], 'Feedback': ex['Feedback']} "
                        "for ex in examples_with_feedback]\n"
                        "res = synthesize_instruction(current_instruction, digest)\n"
                        "SUBMIT(new_instruction=res)"
                    ),
                },
                {"new_instruction": "Extract all bridge named entities concisely without redundant titles."},
            ]
        )
        setattr(qwen_lm, "_yaw_recipe", "llamacpp")
        setattr(qwen_lm, "_yaw_supports_gbnf", True)

        proposer = RLMInstructionProposer(
            orchestrator_lm=mistral_lm,
            worker_lm=qwen_lm,
            enable_gbnf=True,
            max_instruction_chars=900,
        )
        self.assertFalse(isinstance(proposer, dspy.Module))

        candidate = {
            "ner": "Extract named entities from the passage.",
            "qa_cot": "Answer the question concisely.",
        }
        reflective_dataset = {
            "ner": [
                {
                    "Inputs": {"passage": "A" * 1200},
                    "Generated Outputs": {"named_entities": ["Radio City"]},
                    "Feedback": "Missed gold items: ['Bangalore', 'Abraham Thomas'].",
                }
            ]
        }

        # Boundary: empty or missing component updates make zero LM calls
        self.assertEqual(proposer(candidate, reflective_dataset, []), {})
        self.assertEqual(proposer(candidate, reflective_dataset, ["missing_comp"]), {})
        self.assertEqual(len(mistral_lm.history), 0)
        self.assertEqual(len(qwen_lm.history), 0)

        # Happy path: updates 'ner' via Outer Mistral RLM -> Inner Qwen RLM -> Qwen GBNF synthesis
        with dspy.context(track_usage=True):
            updated = proposer(candidate, reflective_dataset, ["ner"])
        self.assertEqual(
            updated,
            {"ner": "Extract all bridge named entities concisely without redundant titles."},
        )
        self.assertEqual(len(mistral_lm.history), 1)
        self.assertEqual(len(qwen_lm.history), 2)

        # Verify Mistral got instruction proposal prompt (not SPO/QA tools) and zero GBNF
        mistral_call = mistral_lm.history[0]
        self.assertIsNone(mistral_call["kwargs"].get("extra_body"))
        self.assertEqual(mistral_call["kwargs"].get("temperature"), 0.8)
        mistral_prompt_text = str(mistral_call.get("messages") or mistral_call.get("prompt"))
        self.assertIn("qwen_propose_instruction", mistral_prompt_text)
        self.assertIn("new_instruction", mistral_prompt_text)
        self.assertNotIn("qwen_extract_triples", mistral_prompt_text)

        # Verify Qwen Call 1 was Inner RLM (no GBNF) and Call 2 was GBNF ProposeInstruction synthesis
        qwen_rlm_call = qwen_lm.history[0]
        self.assertIsNone(qwen_rlm_call["kwargs"].get("extra_body"))
        qwen_synth_call = qwen_lm.history[1]
        self.assertEqual(qwen_synth_call["kwargs"].get("temperature"), 0.7)
        self.assertEqual(
            (qwen_synth_call["kwargs"].get("extra_body") or {}).get("grammar"),
            DSPY_INSTRUCTION_PROPOSAL_GBNF_GRAMMAR,
        )

        # Verify telemetry was recorded on the proposer
        self.assertEqual(len(proposer.telemetry), 1)
        self.assertEqual(proposer.telemetry[0]["component_name"], "ner")
        self.assertEqual(
            proposer.telemetry[0]["new_instruction"],
            "Extract all bridge named entities concisely without redundant titles.",
        )

    def test_live_silicon_rlm_instruction_proposer(self):
        """Run un-mocked RLMInstructionProposer (RLM -> RLM cascade) against live Lemonade Server."""
        loaded_models = self._probe_live_lemonade_models()
        required = {"Qwen3-4B-Instruct-2507-GGUF", "Mistral-7B-Instruct-v0.3-Hybrid"}
        if not required.issubset(loaded_models):
            self.skipTest(
                f"Live Lemonade Server models {sorted(required)} not loaded (found {sorted(loaded_models)})"
            )

        from yaw.daemon.graph.inference import (
            DSPY_INSTRUCTION_PROPOSAL_GBNF_GRAMMAR,
            RLMInstructionProposer,
            create_dspy_rlm_pair,
        )

        orch_lm, worker_lm = create_dspy_rlm_pair(
            base_url="http://localhost:13305/api/v1",
            reflection=True,
            cache=False,
        )
        proposer = RLMInstructionProposer(
            orchestrator_lm=orch_lm,
            worker_lm=worker_lm,
            enable_gbnf=True,
            max_instruction_chars=800,
            rlm_max_iters=3,
        )
        candidate = {
            "qa_cot": "Read the retrieved Wikipedia passages and answer the multi-hop question."
        }
        reflective_dataset = {
            "qa_cot": [
                {
                    "Inputs": {
                        "context": "Wikipedia Title: Erik Hort\nErik Hort was born in Montebello, New York.\n\nWikipedia Title: Montebello, New York\nMontebello is an incorporated village in the town of Ramapo, Rockland County, New York.",
                        "question": "What county is Erik Hort's birthplace a part of?",
                    },
                    "Generated Outputs": {
                        "reasoning": "Erik Hort was born in Montebello, which is in Rockland County, New York.",
                        "answer": "Erik Hort's birthplace is part of Rockland County in New York.",
                    },
                    "Feedback": "[qa_cot] PASS Tier 1 floor (1.0000 >= 0.0000). EM=0.0000, F1=0.3636. Missed gold items: ['Rockland County'].",
                }
            ]
        }

        updated = proposer(candidate, reflective_dataset, ["qa_cot"])
        self.assertIn("qa_cot", updated)
        new_inst = updated["qa_cot"].strip()
        self.assertGreater(len(new_inst), 15)
        self.assertLessEqual(len(new_inst), 800)
        self.assertFalse(new_inst.startswith("]"))

        self.assertGreaterEqual(len(orch_lm.history), 1)
        self.assertIsNone(orch_lm.history[0]["kwargs"].get("extra_body"))
        self.assertEqual(orch_lm.history[0]["kwargs"].get("temperature"), 0.8)

        # Inner RLM on Qwen + GBNF ProposeInstruction synthesis
        self.assertGreaterEqual(len(worker_lm.history), 2)
        gbnf_calls = [
            c for c in worker_lm.history if (c["kwargs"].get("extra_body") or {}).get("grammar")
        ]
        self.assertGreaterEqual(len(gbnf_calls), 1)
        self.assertEqual(
            (gbnf_calls[0]["kwargs"].get("extra_body") or {}).get("grammar"),
            DSPY_INSTRUCTION_PROPOSAL_GBNF_GRAMMAR,
        )
        self.assertEqual(len(proposer.telemetry), 1)

    def test_extract_query_intent_gbnf_and_sampling_contract(self):
        """TDD Contract: extract_query_intent enforces DSPY_QUERY_INTENT_GBNF_GRAMMAR on llamacpp with temp=0.4 and freq_penalty=0.2."""
        from dspy.utils.dummies import DummyLM

        from yaw.daemon.graph.inference import (
            DSPY_QUERY_INTENT_GBNF_GRAMMAR,
            HippoRAGProgram,
            build_dspy_query_intent_grammar,
        )

        self.assertEqual(build_dspy_query_intent_grammar(), DSPY_QUERY_INTENT_GBNF_GRAMMAR)
        self.assertIn("query_intent", DSPY_QUERY_INTENT_GBNF_GRAMMAR)

        qwen_lm = DummyLM([{"query_intent": "Erik Hort birthplace village town county"}])
        setattr(qwen_lm, "_yaw_recipe", "llamacpp")
        setattr(qwen_lm, "_yaw_supports_gbnf", True)

        prog = HippoRAGProgram(
            execution_mode="direct",
            qa_mode="cot",
            enable_gbnf=True,
            rlm_sub_lm=qwen_lm,
        )
        intent = prog.extract_query_intent("What county is Erik Hort's birthplace a part of?")
        self.assertEqual(intent, "Erik Hort birthplace village town county")
        self.assertEqual(len(qwen_lm.history), 1)
        call_kwargs = qwen_lm.history[0]["kwargs"]
        self.assertAlmostEqual(call_kwargs.get("temperature"), 0.4)
        self.assertAlmostEqual(call_kwargs.get("frequency_penalty"), 0.2)
        self.assertEqual(
            (call_kwargs.get("extra_body") or {}).get("grammar"),
            DSPY_QUERY_INTENT_GBNF_GRAMMAR,
        )

        # Boundary: non-GBNF LM (e.g. ryzenai-llm) omits extra_body.grammar
        npu_lm = DummyLM([{"query_intent": "Erik Hort birthplace county"}])
        setattr(npu_lm, "_yaw_recipe", "ryzenai-llm")
        setattr(npu_lm, "_yaw_supports_gbnf", False)
        intent_npu = prog.extract_query_intent("What county is Erik Hort's birthplace a part of?", lm=npu_lm)
        self.assertEqual(intent_npu, "Erik Hort birthplace county")
        self.assertIsNone(npu_lm.history[0]["kwargs"].get("extra_body"))

    def test_strict_17_predictor_separation_of_concerns_and_distinct_signatures(self):
        """TDD Contract: All 19 predictors across HippoRAGProgram (15) and RLMInstructionProposer (4) have distinct Signatures and dedicated .lm instances."""
        import dspy
        from dspy.utils.dummies import DummyLM

        from yaw.daemon.graph.inference import (
            DSPY_MISSING_ENTITY_DEFS_GBNF_GRAMMAR,
            DSPY_SHACL_REPAIR_GBNF_GRAMMAR,
            DSPY_SYNONYM_JUDGE_GBNF_GRAMMAR,
            HippoRAGProgram,
            RLMInstructionProposer,
            parse_shacl_repairs,
            parse_synonym_decisions,
        )

        primary_lm = DummyLM([{"named_entities": [["Ada Lovelace", "English mathematician"]]}])
        sub_lm = DummyLM(
            [
                {
                    "triples": [
                        {
                            "sub": "Ada Lovelace",
                            "pred": "wrote",
                            "obj": ["Note G", "First computer algorithm"],
                            "def": "Ada Lovelace wrote the Note G algorithm.",
                        }
                    ]
                },
                {
                    "decisions": [
                        {"pair_id": "ent-1|ent-2", "relation": "skos:altLabel", "canonical": "A"}
                    ]
                },
                {
                    "repairs": [
                        {
                            "focus_node": "ent-1",
                            "action": "set_definition",
                            "value": "Valid non-empty definition",
                        }
                    ]
                },
            ]
        )
        setattr(sub_lm, "_yaw_recipe", "llamacpp")
        setattr(sub_lm, "_yaw_supports_gbnf", True)

        dspy.settings.configure(lm=primary_lm)
        prog = HippoRAGProgram(
            execution_mode="direct",
            qa_mode="cot",
            enable_gbnf=True,
            rlm_sub_lm=sub_lm,
        )
        proposer = RLMInstructionProposer(
            orchestrator_lm=primary_lm,
            worker_lm=sub_lm,
            enable_gbnf=True,
        )

        stage_preds = prog.all_stage_predictors()
        refl_preds = proposer.all_reflection_predictors()
        self.assertEqual(len(stage_preds), 15)
        self.assertEqual(len(refl_preds), 4)

        all_19 = dict(stage_preds + refl_preds)
        self.assertEqual(len(all_19), 19)

        # Pairwise-distinct Signature classes across all 19 predictors
        sig_classes = [p.signature for p in all_19.values()]
        self.assertEqual(len(set(sig_classes)), 19)

        # Pairwise-distinct .lm instances across all 19 predictors
        lm_ids = [id(getattr(p, "lm", None)) for p in all_19.values()]
        self.assertTrue(all(getattr(p, "lm", None) is not None for p in all_19.values()))
        self.assertEqual(len(set(lm_ids)), 19)

        # Test extract_structured_triples, judge_synonyms, propose_shacl_repairs
        structured = prog.extract_structured_triples(
            "Ada Lovelace wrote Note G.",
            [["Ada Lovelace", "English mathematician"]],
        )
        self.assertEqual(len(structured), 1)
        self.assertEqual(structured[0].subj, "Ada Lovelace")
        self.assertEqual(structured[0].subj_def, "English mathematician")
        self.assertEqual(structured[0].obj, "Note G")
        self.assertEqual(structured[0].obj_def, "First computer algorithm")
        self.assertEqual(structured[0].spo_def, "Ada Lovelace wrote the Note G algorithm.")

        decisions = prog.judge_synonyms('[{"pair_id": "ent-1|ent-2"}]')
        self.assertEqual(decisions, [{"pair_id": "ent-1|ent-2", "relation": "skos:altLabel", "canonical": "A"}])

        repairs = prog.propose_shacl_repairs(
            '[{"focus_node": "ent-1"}]',
            focus_context='{"entity": {"entity_id": "ent-1"}}',
        )
        self.assertEqual(
            repairs,
            [{"focus_node": "ent-1", "action": "set_definition", "value": "Valid non-empty definition"}],
        )

        # Verify GBNF grammars and parser edge cases
        self.assertIn("entity_definitions", DSPY_MISSING_ENTITY_DEFS_GBNF_GRAMMAR)
        self.assertIn("skos:altLabel", DSPY_SYNONYM_JUDGE_GBNF_GRAMMAR)
        self.assertIn("set_definition", DSPY_SHACL_REPAIR_GBNF_GRAMMAR)
        self.assertEqual(parse_synonym_decisions("invalid json"), [])
        self.assertEqual(parse_shacl_repairs("invalid json"), [])

    def test_stage1_to_stage2_handoff_polymorphic_triples_and_define_missing_entities(self):
        """Verify Stage 1 -> Stage 2 [[name, def], ...] handoff, dynamic known_entities GBNF, DefineMissingEntities repair, and Pareto penalty."""
        import dspy
        from dspy.utils.dummies import DummyLM

        from yaw.daemon.graph.inference import (
            DefineMissingEntities,
            HippoRAGProgram,
            OrchestrateMissingEntityDefinitions,
            build_dspy_missing_entity_defs_grammar,
            build_dspy_triples_grammar,
            compute_sizecoding_pareto_score,
        )
        from yaw.daemon.graph.inference.parsers import extract_structured_triples

        self.assertTrue(issubclass(DefineMissingEntities, dspy.Signature))
        self.assertTrue(issubclass(OrchestrateMissingEntityDefinitions, dspy.Signature))

        # 1. Dynamic GBNF grammar for known_entities and missing_entity_defs
        g_known = build_dspy_triples_grammar(
            max_triples=3,
            known_entities=[["Radio City", "FM station"], ["Bangalore", "City"]],
        )
        self.assertIn("known-entity ::= \"\\\"Radio City\\\"\" | \"\\\"Bangalore\\\"\"", g_known)
        self.assertIn("entity-slot ::= known-entity | new-entity-pair", g_known)
        self.assertIn('triple-list ::= (triple (", " triple (", " triple)?)?)?', g_known)

        g_unknown = build_dspy_triples_grammar(max_triples=None, known_entities=[])
        self.assertNotIn("known-entity ::=", g_unknown)
        self.assertIn("entity-slot ::= new-entity-pair | string", g_unknown)

        g_missing = build_dspy_missing_entity_defs_grammar(["Note G", "Charles Babbage"])
        self.assertIn('entity-name ::= "\\"Note G\\"" | "\\"Charles Babbage\\""', g_missing)

        # 2. Parser returns empty string (never heuristic "(in statement: ...)") when definitions are omitted
        parsed_bare = extract_structured_triples(
            json.dumps(
                {
                    "triples": [
                        {
                            "sub": "Ada Lovelace",
                            "pred": "collaborated with",
                            "obj": "Charles Babbage",
                            "def": "Ada Lovelace collaborated with Charles Babbage.",
                        }
                    ]
                }
            )
        )
        self.assertEqual(len(parsed_bare), 1)
        self.assertEqual(parsed_bare[0].subj_def, "")
        self.assertEqual(parsed_bare[0].obj_def, "")

        # 3. Direct mode: Stage 1 defines "Ada Lovelace", Stage 2 introduces bare "Charles Babbage" without inline pair -> DefineMissingEntities repairs it
        direct_lm = DummyLM(
            [
                {
                    "named_entities": [
                        ["Ada Lovelace", "English mathematician and writer"],
                    ]
                },
                {
                    "triples": [
                        {
                            "sub": "Ada Lovelace",
                            "pred": "collaborated with",
                            "obj": "Charles Babbage",
                            "def": "Ada Lovelace collaborated with Charles Babbage on the Analytical Engine.",
                        }
                    ]
                },
                {
                    "entity_definitions": [
                        ["Charles Babbage", "English polymath and mechanical computer pioneer"],
                    ]
                },
            ],
            adapter=dspy.JSONAdapter(),
        )
        setattr(direct_lm, "_yaw_recipe", "llamacpp")
        setattr(direct_lm, "_yaw_supports_gbnf", True)

        prog_direct = HippoRAGProgram(
            execution_mode="direct",
            enable_gbnf=True,
            rlm_sub_lm=direct_lm,
        )
        with dspy.context(track_usage=True):
            rec = prog_direct.extract_openie(
                "chunk-repair",
                "Ada Lovelace collaborated with Charles Babbage on the Analytical Engine.",
                lm=direct_lm,
            )
        self.assertEqual(len(direct_lm.history), 3)
        # Call 1 (ExtractTriples) received [[name, def]] in named_entities
        triples_prompt_str = str(direct_lm.history[1].get("messages") or direct_lm.history[1].get("prompt"))
        self.assertIn("English mathematician and writer", triples_prompt_str)
        # Call 2 (DefineMissingEntities) received dynamic GBNF locked to ["Charles Babbage"]
        self.assertEqual(
            direct_lm.history[2]["kwargs"].get("extra_body"),
            {"grammar": build_dspy_missing_entity_defs_grammar(["Charles Babbage"])},
        )
        # Resulting OpenIERecord has both entities and definitions populated and audit trail recorded
        self.assertEqual(rec.entities, ("Ada Lovelace", "Charles Babbage"))
        self.assertEqual(
            rec.entity_definitions,
            (
                ("Ada Lovelace", "English mathematician and writer"),
                ("Charles Babbage", "English polymath and mechanical computer pioneer"),
            ),
        )
        self.assertEqual(len(rec.structured_triples), 1)
        self.assertEqual(rec.structured_triples[0].subj_def, "English mathematician and writer")
        self.assertEqual(
            rec.structured_triples[0].obj_def,
            "English polymath and mechanical computer pioneer",
        )
        self.assertEqual(len(prog_direct.entity_def_audit_trail), 1)
        self.assertEqual(prog_direct.entity_def_audit_trail[0]["missing_count"], 1)
        self.assertEqual(prog_direct.entity_def_audit_trail[0]["repaired_count"], 1)

        # 3b. Cross-chunk isolation: Chunk 2 referencing bare "Charles Babbage" must NOT leak Chunk 1's repaired definition
        chunk2_lm = DummyLM(
            [
                {
                    "named_entities": [
                        ["Difference Engine", "Early mechanical calculator"],
                    ]
                },
                {
                    "triples": [
                        {
                            "sub": "Charles Babbage",
                            "pred": "designed",
                            "obj": "Difference Engine",
                            "def": "Charles Babbage designed the Difference Engine.",
                        }
                    ]
                },
                {
                    "entity_definitions": [
                        ["Charles Babbage", "19th-century inventor of the Difference Engine"],
                    ]
                },
            ],
            adapter=dspy.JSONAdapter(),
        )
        setattr(chunk2_lm, "_yaw_recipe", "llamacpp")
        setattr(chunk2_lm, "_yaw_supports_gbnf", True)
        rec_chunk2 = prog_direct.extract_openie(
            "chunk-2",
            "Charles Babbage designed the Difference Engine.",
            lm=chunk2_lm,
        )
        self.assertEqual(len(chunk2_lm.history), 3)
        self.assertEqual(len(prog_direct.entity_def_audit_trail), 2)
        self.assertEqual(
            rec_chunk2.structured_triples[0].subj_def,
            "19th-century inventor of the Difference Engine",
        )

        # 3c. Cinderella Stage 1 -> Stage 2 handoff: known entities as strings + inline new entity pair ["glass slipper", "..."]
        cinderella_lm = DummyLM(
            [
                {
                    "named_entities": [
                        ["Cinderella", "Young woman mistreated by her stepmother who attends the royal ball"],
                        ["a ball", "Royal dance hosted by the prince"],
                    ]
                },
                {
                    "triples": [
                        {
                            "sub": "Cinderella",
                            "pred": "attended",
                            "obj": "a ball",
                            "def": "Cinderella attended the royal ball.",
                        },
                        {
                            "sub": "Cinderella",
                            "pred": "lost",
                            "obj": [
                                "glass slipper",
                                "Delicate footwear lost by Cinderella at the ball",
                            ],
                            "def": "Cinderella lost her glass slipper while fleeing the ball.",
                        },
                        {
                            "sub": "glass slipper",
                            "pred": "fit",
                            "obj": "Cinderella",
                            "def": "The glass slipper fit Cinderella.",
                        },
                    ]
                },
            ],
            adapter=dspy.JSONAdapter(),
        )
        setattr(cinderella_lm, "_yaw_recipe", "llamacpp")
        setattr(cinderella_lm, "_yaw_supports_gbnf", True)
        prog_cinderella = HippoRAGProgram(
            execution_mode="direct",
            enable_gbnf=True,
            rlm_sub_lm=cinderella_lm,
        )
        rec_cin = prog_cinderella.extract_openie(
            "chunk-cinderella",
            "Cinderella attended a ball and lost her glass slipper, which later fit Cinderella.",
            lm=cinderella_lm,
        )
        # Only 2 LM calls (Stage 1 NER + Stage 2 Triples) — zero DefineMissingEntities needed
        self.assertEqual(len(cinderella_lm.history), 2)
        self.assertEqual(len(prog_cinderella.entity_def_audit_trail), 0)
        self.assertEqual(
            cinderella_lm.history[1]["kwargs"].get("extra_body"),
            {
                "grammar": build_dspy_triples_grammar(
                    None,
                    known_entities=["Cinderella", "a ball"],
                )
            },
        )
        self.assertEqual(rec_cin.entities, ("Cinderella", "a ball", "glass slipper"))
        self.assertEqual(
            rec_cin.entity_definitions,
            (
                ("Cinderella", "Young woman mistreated by her stepmother who attends the royal ball"),
                ("a ball", "Royal dance hosted by the prince"),
                ("glass slipper", "Delicate footwear lost by Cinderella at the ball"),
            ),
        )
        self.assertEqual(len(rec_cin.structured_triples), 3)
        # Third triple referencing plain "glass slipper" receives the inline definition propagated from the second triple
        self.assertEqual(
            rec_cin.structured_triples[2].subj_def,
            "Delicate footwear lost by Cinderella at the ball",
        )
        self.assertEqual(
            rec_cin.structured_triples[2].obj_def,
            "Young woman mistreated by her stepmother who attends the royal ball",
        )

        # 4. RLM mode: OrchestrateMissingEntityDefinitions delegates to qwen_define_missing_entities
        mistral_rlm = DummyLM(
            [
                {
                    "reasoning": "Delegate missing entity definitions to Qwen GBNF worker.",
                    "code": (
                        "defs = qwen_define_missing_entities(passage, missing_entities, triples)\n"
                        "SUBMIT(entity_definitions=defs)"
                    ),
                }
            ]
        )
        setattr(mistral_rlm, "_yaw_recipe", "ryzenai-llm")
        setattr(mistral_rlm, "_yaw_supports_gbnf", False)
        qwen_rlm = DummyLM(
            [
                {
                    "entity_definitions": [
                        ["Analytical Engine", "Proposed mechanical general-purpose computer"],
                    ]
                }
            ],
            adapter=dspy.JSONAdapter(),
        )
        setattr(qwen_rlm, "_yaw_recipe", "llamacpp")
        setattr(qwen_rlm, "_yaw_supports_gbnf", True)

        prog_rlm = HippoRAGProgram(
            execution_mode="rlm",
            enable_gbnf=True,
            orchestrator_lm=mistral_rlm,
            rlm_sub_lm=qwen_rlm,
            rlm_max_iters=2,
        )
        repaired_rlm = prog_rlm.define_missing_entities(
            "Ada Lovelace worked on the Analytical Engine.",
            ["Analytical Engine"],
            [{"sub": "Ada Lovelace", "pred": "worked on", "obj": "Analytical Engine", "def": "Ada Lovelace worked on the Analytical Engine."}],
        )
        self.assertEqual(
            repaired_rlm,
            (("Analytical Engine", "Proposed mechanical general-purpose computer"),),
        )
        self.assertEqual(len(mistral_rlm.history), 1)
        self.assertIsNone(mistral_rlm.history[0]["kwargs"].get("extra_body"))
        self.assertEqual(len(qwen_rlm.history), 1)
        self.assertEqual(
            qwen_rlm.history[0]["kwargs"].get("extra_body"),
            {"grammar": build_dspy_missing_entity_defs_grammar(["Analytical Engine"])},
        )
        # Only outer RLM call is recorded at top level (no inner double-counting)
        self.assertEqual(len(prog_rlm.entity_def_audit_trail), 1)
        self.assertEqual(prog_rlm.entity_def_audit_trail[0]["mode"], "rlm")

        # 5. Pareto score penalizes missing_entity_defs_count and missing_entity_defs_tokens
        clean_pareto = compute_sizecoding_pareto_score(0.80, tokens=100)
        repair_pareto = compute_sizecoding_pareto_score(
            0.80,
            tokens=100,
            missing_entity_defs_count=2,
            missing_entity_defs_tokens=60,
        )
        self.assertLess(repair_pareto.score, clean_pareto.score)
        self.assertGreater(repair_pareto.gm_bad, clean_pareto.gm_bad)
        self.assertIn(
            "[entity_definer] repaired 2 missing Stage-2 entity definition(s) (60 aux tokens)",
            repair_pareto.feedback,
        )


if __name__ == "__main__":
    unittest.main()
