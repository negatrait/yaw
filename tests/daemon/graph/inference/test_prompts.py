"""Verify native prompt template rendering, parser invariants, and canonical 12 DSPy signatures."""

import hashlib
import json
import unittest
from pathlib import Path
from typing import Any, TypedDict

import dspy
import pytest
from dspy.utils.callback import BaseCallback
from dspy.utils.dummies import DummyLM as NativeDummyLM

from yaw.core import load_config
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


class CapabilityDummyLM(NativeDummyLM):
    """Native deterministic LM with the capabilities used by YAW adapters."""

    _yaw_recipe: str | None = None
    _yaw_supports_gbnf: bool | None = None

    def __init__(self, answers: Any, **kwargs: Any) -> None:
        kwargs.setdefault("adapter", dspy.JSONAdapter())
        super().__init__(answers, **kwargs)


class LMRequestAudit(BaseCallback):
    """Inspect actual DSPy requests without enabling unbounded LM histories."""

    def __init__(self):
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def on_lm_start(self, call_id: str, instance: Any, inputs: dict[str, Any]):
        self.calls.append((instance, {**instance.kwargs, **inputs.get("kwargs", {})}))


class RecallInputs(TypedDict):
    recall_at_1: float
    recall_at_2: float
    recall_at_10: float
    recall_at_20: float
    em: float


class PromptsAndParsersParityTests(unittest.TestCase):
    def test_ner_messages_byte_for_byte_parity(self):
        """Verify render_ner_messages produces the canonical 4-turn NER chat message structure."""
        test_passages = [
            "Alan Turing was an English mathematician and computer scientist.",
            "Radio City is an FM radio station in India.\nIt operates in English and Hindi.",
            "Special characters: 'quotes', \"double quotes\", $dollar signs, and \\backslashes.",
        ]
        for passage in test_passages:
            our_msgs = render_ner_messages(passage)
            self.assertEqual(len(our_msgs), 4)
            self.assertEqual([m["role"] for m in our_msgs], ["system", "user", "assistant", "user"])
            self.assertEqual(our_msgs[-1]["content"], passage)

    def test_triple_messages_byte_for_byte_parity(self):
        """Verify render_triple_messages produces the canonical 4-turn OpenIE triple extraction structure."""
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
            our_msgs = render_triple_messages(passage, entities)
            self.assertEqual(len(our_msgs), 4)
            self.assertEqual([m["role"] for m in our_msgs], ["system", "user", "assistant", "user"])
            self.assertIn(passage, our_msgs[-1]["content"])
            self.assertIn(named_entity_json, our_msgs[-1]["content"])

    def test_rag_qa_messages_byte_for_byte_parity(self):
        """Verify render_rag_qa_messages produces the canonical 4-turn MuSiQue QA message structure."""
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
            our_msgs = render_rag_qa_messages(prompt_user)
            self.assertEqual(len(our_msgs), 4)
            self.assertEqual(
                [m["role"] for m in our_msgs],
                ["system", "user", "assistant", "user"],
            )
            self.assertEqual(our_msgs[-1]["content"], prompt_user)

    def test_fact_filter_messages_byte_for_byte_parity(self):
        """Verify render_fact_filter_messages produces the canonical 10-demo fact filter structure."""
        candidate_facts = [
            ("apple", "founded in", "california"),
            ("apple", "created", "iphone"),
            ("steve jobs", "co-founded", "apple"),
        ]
        query = "Where was Apple founded?"

        fact_before_filter = {"fact": [list(item) for item in candidate_facts]}
        our_messages = render_fact_filter_messages(query, candidate_facts)
        self.assertEqual(our_messages[0]["role"], "system")
        self.assertEqual(our_messages[-1]["role"], "user")
        self.assertIn(query, our_messages[-1]["content"])
        self.assertIn(json.dumps(fact_before_filter), our_messages[-1]["content"])

    def test_compute_mdhash_id_parity(self):
        """Verify compute_mdhash_id matches MD5 hex digest with prefix."""
        test_strings = [
            ("", ""),
            ("hello world", ""),
            ("test content", "chunk-"),
            ("Radio City is India's first private FM radio station", "ent-"),
        ]
        for text, prefix in test_strings:
            expected = prefix + hashlib.md5(text.encode()).hexdigest()
            self.assertEqual(compute_mdhash_id(text, prefix), expected)

    def test_text_processing_parity(self):
        """Verify text_processing normalizes alphanumeric lowercase tokens."""
        self.assertEqual(text_processing("Apple Inc."), "apple inc")
        self.assertEqual(text_processing("Radio-City (FM Station) #1!"), "radio city fm station 1")
        self.assertEqual(text_processing("   Multiple   Spaces   and   Tabs\t\n"), "multiple spaces and tabs")
        self.assertEqual(text_processing("123 numbers & Symbols_underscore"), "123 numbers symbols underscore")
        self.assertEqual(
            text_processing(["Apple Inc.", "Google LLC", "DeepMind (UK)"]),
            ["apple inc", "google llc", "deepmind uk"],
        )

    def test_filter_invalid_triples_parity(self):
        """Verify filter_invalid_triples deduplicates 3-element triples and coerces items to strings."""
        input_triples: list[Any] = [
            ["Radio City", "located in", "India"],
            ["invalid", "only two"],
            ["invalid", "too", "many", "items"],
            ["Radio City", "located in", "India"],  # duplicate
            ["Radio City", "plays songs in", "Hindi"],
            [123, 456, 789],  # non-string coercion
        ]
        actual = filter_invalid_triples(input_triples)
        self.assertEqual(
            actual,
            [
                ["Radio City", "located in", "India"],
                ["Radio City", "plays songs in", "Hindi"],
                ["123", "456", "789"],
            ],
        )

    def test_extract_named_entities_parity(self):
        """Verify extract_named_entities extracts entities cleanly."""
        resp1 = json.dumps({"named_entities": ["Alan Turing", "London", "1912"]})
        self.assertEqual(
            extract_named_entities(resp1),
            ("Alan Turing", "London", "1912"),
        )

        resp2 = json.dumps(["Alan Turing", "London", "1912"])
        self.assertEqual(
            extract_named_entities(resp2),
            ("Alan Turing", "London", "1912"),
        )

        resp3 = "```json\n" + json.dumps(["Alan Turing", "London"]) + "\n```"
        self.assertEqual(
            extract_named_entities(resp3),
            ("Alan Turing", "London"),
        )

    def test_extract_triples_parity(self):
        """Verify extract_triples parses and filters triples."""
        resp = json.dumps(
            {
                "triples": [
                    ["Alan Turing", "born in", "London"],
                    ["London", "located in", "United Kingdom"],
                    ["invalid", "pair"],
                    ["Alan Turing", "born in", "London"],  # duplicate
                ]
            }
        )
        triples = extract_triples(resp)
        self.assertEqual(
            triples,
            (
                ("Alan Turing", "born in", "London"),
                ("London", "located in", "United Kingdom"),
            ),
        )

    def test_filter_reranked_facts_parity(self):
        """Verify filter_reranked_facts selects matching candidate indices."""
        candidates = [
            ("imperial river", "is located in", "florida"),
            ("imperial river", "is a river in", "united states"),
            ("amaradia", "flows through", "ro ia de amaradia"),
            ("unrelated fact", "has predicate", "object"),
        ]

        llm_response = (
            "[[ ## fact_after_filter ## ]]\n"
            '{"fact":[["imperial river","is located in","florida"],["amaradia","flows through","ro ia '
            'de amaradia"]]}\n'
            "\n"
            "[[ ## completed ## ]]"
        )

        matched_indices = filter_reranked_facts(candidates, llm_response)
        self.assertEqual(matched_indices, [0, 2])
        self.assertEqual(filter_reranked_facts(candidates, llm_response, max_facts=1), [0])

    def test_repair_truncated_json_handles_openie_boundaries(self):
        """Verify repair_truncated_json fixes max_tokens truncations without losing valid elements."""
        truncated_triples = (
            '{"triples": [\n  ["A", "is", "B"],\n  ["C", "located in", "D"],\n  ["E", "broken triple without close'
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
            NER_DEMOS,
            QA_DEMOS,
            TRIPLE_DEMOS,
            TRIPLE_FILTER_DEMOS,
            AnswerQuestion,
            ExtractEntities,
            ExtractTriples,
            FactList,
            FilterTriples,
        )

        self.assertTrue(issubclass(ExtractEntities, dspy.Signature))
        self.assertTrue(issubclass(ExtractTriples, dspy.Signature))
        self.assertTrue(issubclass(FilterTriples, dspy.Signature))
        self.assertTrue(issubclass(AnswerQuestion, dspy.Signature))

        self.assertEqual(list(ExtractEntities.input_fields.keys()), ["passage"])
        self.assertEqual(list(ExtractEntities.output_fields.keys()), ["named_entities"])
        self.assertEqual(list(ExtractTriples.input_fields.keys()), ["passage", "named_entities"])
        self.assertEqual(list(ExtractTriples.output_fields.keys()), ["triples"])
        self.assertEqual(list(FilterTriples.input_fields.keys()), ["question", "fact_before_filter"])
        self.assertEqual(list(FilterTriples.output_fields.keys()), ["fact"])
        self.assertEqual(list(AnswerQuestion.input_fields.keys()), ["context", "question"])
        self.assertEqual(list(AnswerQuestion.output_fields.keys()), ["answer"])

        self.assertEqual(len(NER_DEMOS), 1)
        self.assertEqual(len(TRIPLE_DEMOS), 1)
        self.assertEqual(len(TRIPLE_FILTER_DEMOS), 10)
        self.assertIsInstance(TRIPLE_FILTER_DEMOS[0].fact_before_filter, str)
        self.assertIsInstance(TRIPLE_FILTER_DEMOS[0].fact, list)
        self.assertIsInstance(FactList(fact=TRIPLE_FILTER_DEMOS[0].fact), FactList)
        self.assertEqual(len(QA_DEMOS), 2)

    def test_hipporag_program_openie_triple_filter_and_qa(self):
        """Verify HippoRAGProgram executes OpenIE, triple filtering, and QA via dspy.JSONAdapter (pure JSON output)."""
        import dspy

        from yaw.daemon.graph.inference.program import HippoRAGProgram

        program = HippoRAGProgram()
        lm = CapabilityDummyLM(
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
        self.assertEqual(record.entities, ("Alan Turing", "London", "United Kingdom"))
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
        selected_indices = program.filter_triples(
            "Where are Imperial River and Amaradia located?",
            candidates,
            lm=lm,
        )
        self.assertEqual(selected_indices, [0, 2])

        qa_pred = program.answer_question(
            "University of Southampton was founded in 1862.",
            "When was it founded?",
            lm=lm,
        )
        self.assertEqual(qa_pred.answer, "1862")
        self.assertIn("1862", qa_pred.reasoning)

    def test_shacl_healer_rlm_executes_in_python_interpreter_with_sub_lm_and_kwargs_tools(self):
        """Verify shacl_healer_rlm executes Python REPL code in dspy.PythonInterpreter with sub_lm wired and kwargs \
tools."""
        import dspy

        from yaw.daemon.graph.inference.grammars import DSPY_SHACL_REPAIR_GBNF_GRAMMAR
        from yaw.daemon.graph.inference.program import HippoRAGProgram

        orch_lm = CapabilityDummyLM(
            [
                {
                    "reasoning": "Propose SHACL repairs via GBNF tool using keyword argument violations=violations.",
                    "code": ("reps = qwen_propose_shacl_repairs(violations=violations)\nSUBMIT(repairs=reps)"),
                }
            ],
            adapter=dspy.ChatAdapter(),
        )
        orch_lm._yaw_recipe = "llamacpp"
        orch_lm._yaw_supports_gbnf = False

        sub_lm = CapabilityDummyLM(
            [
                {
                    "repairs": [
                        {
                            "focus_node": "ent-1",
                            "action": "set_definition",
                            "value": "Repaired entity definition from GBNF sub-model.",
                        }
                    ]
                }
            ],
            adapter=dspy.JSONAdapter(),
        )
        sub_lm._yaw_recipe = "llamacpp"
        sub_lm._yaw_supports_gbnf = True

        program = HippoRAGProgram(
            enable_gbnf=True,
            orchestrator_lm=orch_lm,
            rlm_sub_lm=sub_lm,
            rlm_max_iters=3,
            rlm_max_llm_calls=5,
        )
        self.assertIsNotNone(program.shacl_healer_rlm.sub_lm)

        repairs = program.propose_shacl_repairs(
            '[{"focus_node": "ent-1", "source_shape": "yaw:EntityConceptShape"}]',
            mode="rlm",
        )
        self.assertEqual(
            repairs,
            [
                {
                    "focus_node": "ent-1",
                    "action": "set_definition",
                    "value": "Repaired entity definition from GBNF sub-model.",
                }
            ],
        )
        self.assertEqual(len(orch_lm.history), 1)
        self.assertIsNone(orch_lm.history[0]["kwargs"].get("extra_body"))
        self.assertEqual(len(sub_lm.history), 1)
        self.assertEqual(
            sub_lm.history[0]["kwargs"].get("extra_body"),
            {"grammar": DSPY_SHACL_REPAIR_GBNF_GRAMMAR},
        )

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
            prog1.ner.signature = sig1.with_instructions("GEPA optimized NER instruction.")
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

        from yaw.daemon.graph.inference.grammars import (
            DSPY_NER_GBNF_GRAMMAR,
            DSPY_QA_GBNF_GRAMMAR,
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
        lm = CapabilityDummyLM(
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
        self.assertNotIn("[[ ##", DSPY_QA_GBNF_GRAMMAR)
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
        selected = program.filter_triples("Where was Alan Turing born?", candidates, lm=lm)
        self.assertEqual(selected, [0])
        expected_rerank_grammar = build_dspy_rerank_grammar(candidates, max_facts=4)
        self.assertNotIn("[[ ##", expected_rerank_grammar)
        self.assertIn('"[\\"alan turing\\", \\"born in\\", \\"london\\"]"', expected_rerank_grammar)
        self.assertEqual(
            lm.history[2]["kwargs"].get("extra_body"),
            {"grammar": expected_rerank_grammar},
        )

        program.answer_question("Alan Turing was born in London.", "Where?", lm=lm, use_gbnf=True)
        self.assertEqual(
            lm.history[3]["kwargs"].get("extra_body"),
            {"grammar": DSPY_QA_GBNF_GRAMMAR},
        )

    def test_sizecoding_gbnf_caps_and_gepa_pareto_metric(self):
        """Verify sizecoding GBNF caps on forward() and compute_sizecoding_pareto_score for dspy.GEPA."""
        import dspy
        from dspy.teleprompt.gepa.gepa_utils import _extract_score_and_objective_scores

        from yaw.daemon.graph.inference.grammars import (
            build_dspy_ner_grammar,
            build_dspy_qa_grammar,
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
        lm = CapabilityDummyLM(
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

        # Calling program(...) (GEPA's forward pass) automatically attaches the capped QA GBNF grammar
        with dspy.context(lm=lm):
            pred = program(context="Alan Turing was born in London.", question="Where?")
        self.assertEqual(pred.answer, "London")
        self.assertIn("char{1,180}", build_dspy_qa_grammar(180))
        self.assertEqual(
            lm.history[2]["kwargs"].get("extra_body"),
            {"grammar": build_dspy_qa_grammar(180)},
        )

        # 1. Task quality has no reference gate or cost denominator.
        violated = compute_sizecoding_pareto_score(
            0.60,
            tokens=20,
            pred_name="triples",
            missed_items=["doc-2"],
        )
        self.assertGreater(violated.score, 0.0)
        self.assertAlmostEqual(violated.score, 0.60)
        self.assertIn("Task quality geometric mean", violated.feedback)

        # 2. Identical quality receives the same score at any token count.
        bloated = compute_sizecoding_pareto_score(
            0.80,
            tokens=400,
        )
        sizecoded = compute_sizecoding_pareto_score(
            0.80,
            tokens=80,
        )
        higher_recall_same_tokens = compute_sizecoding_pareto_score(
            0.80,
            tokens=20,
        )
        self.assertEqual(sizecoded.score, bloated.score)
        self.assertGreater(higher_recall_same_tokens.score, violated.score)
        self.assertAlmostEqual(sizecoded.score, 0.80)

        # Verify dspy.GEPA's internal DspyAdapter extracts both scalar score and multi-objective dict
        scalar_score, obj_scores = _extract_score_and_objective_scores(sizecoded)
        self.assertAlmostEqual(scalar_score, sizecoded.score)
        self.assertEqual(
            set(obj_scores.keys()),
            {
                "recall_at_5",
            },
        )
        # Every GEPA objective is higher-is-better (no cost-as-good keys like gm_bad).
        self.assertNotIn("gm_bad", obj_scores)
        self.assertNotIn("pareto_raw", obj_scores)

        # F1 improvements change quality; token count has no effect.
        base_q: RecallInputs = dict(recall_at_1=0.5, recall_at_2=0.7, recall_at_10=0.95, recall_at_20=1.0, em=0.05)
        f1_gain = compute_sizecoding_pareto_score(0.87, f1=0.45, tokens=22_000, **base_q)
        f1_base = compute_sizecoding_pareto_score(0.87, f1=0.35, tokens=22_000, **base_q)
        half_tokens = compute_sizecoding_pareto_score(0.87, f1=0.35, tokens=11_000, **base_q)
        self.assertGreater(f1_gain.score / f1_base.score, half_tokens.score / f1_base.score)
        self.assertAlmostEqual(obj_scores["recall_at_5"], 0.80)
        self.assertNotIn("shacl_conforms", obj_scores)
        self.assertEqual(half_tokens.score, f1_base.score)

        # 3. A true geometric mean preserves zero; individual objectives still improve.
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
        self.assertEqual(em_zero_low_recall.score, 0.0)
        self.assertEqual(em_zero_high_recall.score, 0.0)
        self.assertGreater(
            em_zero_high_recall.objective_scores["recall_at_5"],
            em_zero_low_recall.objective_scores["recall_at_5"],
        )
        self.assertNotIn("preserve 100% of gold retrieval bridges", em_zero_high_recall.feedback)

        # Full canonical 9-predictor surface visibility for GEPA (never hides OpenIE/EDC/SHACL predictors behind
        # _retrieval_hook)
        direct_prog = HippoRAGProgram()
        direct_names = [name for name, _ in direct_prog.named_predictors()]
        expected_canonical_9 = [
            "ner",
            "triples",
            "entity_definer",
            "synonym_judge",
            "shacl_repair",
            "triple_filter",
            "query_intent",
            "qa.predict",
            "shacl_healer_rlm.generate_action",
            "shacl_healer_rlm.extract",
        ]
        self.assertEqual(direct_names, expected_canonical_9)
        direct_prog.set_retrieval_hook(lambda q: ("ctx", ("c1",), (1.0,)))
        hooked_names = [name for name, _ in direct_prog.named_predictors()]
        self.assertEqual(hooked_names, expected_canonical_9)

    @staticmethod
    def _probe_live_lemonade_models(base_url: str) -> set[str]:
        import urllib.request

        with urllib.request.urlopen(base_url.rstrip("/") + "/health", timeout=10) as resp:
            payload = json.load(resp)
        return {item["model_name"] for item in payload["all_models_loaded"] if item.get("loaded")}

    @pytest.mark.live
    def test_live_silicon_direct_gemma_gbnf(self):
        """Run actual extraction, filtering and QA with the configured resident model."""
        import dspy

        from yaw.daemon.graph.inference.chat import create_dspy_lm
        from yaw.daemon.graph.inference.program import HippoRAGProgram

        config = load_config(Path(__file__).resolve().parents[4] / "yaw.toml")
        loaded = self._probe_live_lemonade_models(config.base_url)
        self.assertIn(config.extraction_model, loaded, "Configured extraction model must already be resident")
        passage = (
            "Radio City is India's first private FM radio station and was started on "
            "3 July 2001 in Bangalore by Abraham Thomas."
        )
        candidates = [("Radio City", "started in", "Bangalore"), ("Apple", "founded by", "Steve Jobs")]
        direct_lm = create_dspy_lm(
            config.extraction_model,
            base_url=config.base_url,
            api_key=config.api_key,
            role="extraction",
            max_tokens=1024,
            context_window_tokens=config.context_window_tokens,
            cache=False,
        )
        program = HippoRAGProgram(enable_gbnf=True, orchestrator_lm=direct_lm, rlm_sub_lm=direct_lm)
        audit = LMRequestAudit()
        with dspy.context(callbacks=[audit], disable_history=True):
            record = program.extract_openie("c-live-direct", passage, lm=direct_lm)
            self.assertIn("Radio City", record.entities)
            self.assertIn("Bangalore", record.entities)
            self.assertGreaterEqual(len(record.triples), 1)
            kept = program.filter_triples("Where was Radio City started?", candidates, lm=direct_lm)
            self.assertEqual(kept, [0])
            answer = program.answer_question(passage, "When was Radio City started?", lm=direct_lm)
            self.assertIn("2001", answer.answer)
        requests = [request for lm, request in audit.calls if lm is direct_lm]
        self.assertEqual(len(requests), 4)
        for request in requests:
            self.assertIn("grammar", request.get("extra_body") or {})
        self.assertEqual(direct_lm.history, [])

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

        # 3. Reflection LM temperature & GBNF capability flags on Qwen3-4B-Instruct-2507-GGUF
        qwen_orch_ref = create_dspy_lm("Qwen3-4B-Instruct-2507-GGUF", role="reflection")
        qwen_ref = create_dspy_lm("Qwen3-4B-Instruct-2507-GGUF", role="reflection_worker")
        self.assertEqual(qwen_orch_ref.kwargs["temperature"], 0.8)
        self.assertEqual(qwen_orch_ref.kwargs["max_tokens"], 4096)
        self.assertFalse(getattr(qwen_orch_ref, "_yaw_supports_gbnf", True))

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
        from yaw.daemon.graph.inference import RLMInstructionProposer
        from yaw.daemon.graph.inference.reflection import ReflectionBudgetError

        with self.assertRaises(ValueError):
            RLMInstructionProposer(max_instruction_chars=0)
        proposer = RLMInstructionProposer()
        self.assertEqual(proposer({}, {}, []), {})
        with self.assertRaises(ReflectionBudgetError):
            proposer(component_name="qa.predict", current_instruction="Use evidence.")

    @pytest.mark.live
    def test_live_silicon_rlm_instruction_proposer(self):
        """Run un-mocked single-level RLMInstructionProposer against live Lemonade Server."""
        import dspy

        config = load_config(Path(__file__).resolve().parents[4] / "yaw.toml")
        loaded = self._probe_live_lemonade_models(config.base_url)
        self.assertTrue(
            {config.qa_model, config.extraction_model}.issubset(loaded), "Reflection models must be resident"
        )

        from yaw.daemon.graph.inference import (
            RLMInstructionProposer,
            ServingTokenCounter,
            build_dspy_instruction_proposal_grammar,
            create_dspy_rlm_pair,
        )

        orch_lm, worker_lm = create_dspy_rlm_pair(
            base_url=config.base_url,
            orchestrator_model=config.qa_model,
            sub_model=config.extraction_model,
            api_key=config.api_key,
            context_window_tokens=config.context_window_tokens,
            max_tokens=config.max_output_tokens,
            reflection=True,
            cache=False,
        )
        proposer = RLMInstructionProposer(
            orchestrator_lm=orch_lm,
            worker_lm=worker_lm,
            enable_gbnf=True,
            max_instruction_chars=800,
            rlm_max_iters=3,
            token_counter=ServingTokenCounter(config.base_url, worker_lm.model),
        )
        candidate = {"qa": "Read the retrieved Wikipedia passages and answer the multi-hop question."}
        reflective_dataset = {
            "qa": [
                {
                    "Inputs": {
                        "context": "Wikipedia Title: Erik Hort\n"
                        "Erik Hort was born in Montebello, New York.\n"
                        "\n"
                        "Wikipedia Title: Montebello, New York\n"
                        "Montebello is an incorporated village in the town of Ramapo, Rockland County, New York.",
                        "question": "What county is Erik Hort's birthplace a part of?",
                    },
                    "Generated Outputs": {
                        "reasoning": "Erik Hort was born in Montebello, which is in Rockland County, New York.",
                        "answer": "Erik Hort's birthplace is part of Rockland County in New York.",
                    },
                    "Feedback": "[qa] PASS Tier 1 floor (1.0000 >= 0.0000). EM=0.0000, F1=0.3636. Missed gold items: "
                    "['Rockland County'].",
                }
            ]
        }

        audit = LMRequestAudit()
        with dspy.context(callbacks=[audit], disable_history=True):
            updated = proposer(candidate, reflective_dataset, ["qa"])
        self.assertIn("qa", updated)
        new_inst = updated["qa"].strip()
        self.assertGreater(len(new_inst), 15)
        self.assertLessEqual(len(new_inst), 800)
        self.assertFalse(new_inst.startswith("]"))

        orchestrator_calls = [request for lm, request in audit.calls if lm is orch_lm]
        self.assertGreaterEqual(len(orchestrator_calls), 1)
        self.assertIsNone(orchestrator_calls[0].get("extra_body"))
        self.assertEqual(orchestrator_calls[0].get("temperature"), 0.8)
        worker_calls = [request for lm, request in audit.calls if lm is worker_lm]
        self.assertGreaterEqual(len(worker_calls), 1)
        gbnf_calls = [request for request in worker_calls if (request.get("extra_body") or {}).get("grammar")]
        self.assertGreaterEqual(len(gbnf_calls), 1)
        self.assertEqual(gbnf_calls[0]["extra_body"]["grammar"], build_dspy_instruction_proposal_grammar(max_chars=800))
        self.assertEqual(len(proposer.telemetry), 1)
        self.assertEqual(orch_lm.history, [])
        self.assertEqual(worker_lm.history, [])

    def test_extract_query_intent_gbnf_and_sampling_contract(self):
        """TDD Contract: extract_query_intent enforces DSPY_QUERY_INTENT_GBNF_GRAMMAR on llamacpp with temp=0.4 and \
freq_penalty=0.2."""

        from yaw.daemon.graph.inference import (
            DSPY_QUERY_INTENT_GBNF_GRAMMAR,
            HippoRAGProgram,
            build_dspy_query_intent_grammar,
        )

        self.assertEqual(build_dspy_query_intent_grammar(), DSPY_QUERY_INTENT_GBNF_GRAMMAR)
        self.assertIn("query_intent", DSPY_QUERY_INTENT_GBNF_GRAMMAR)

        qwen_lm = CapabilityDummyLM([{"query_intent": "Erik Hort birthplace village town county"}])
        qwen_lm._yaw_recipe = "llamacpp"
        qwen_lm._yaw_supports_gbnf = True

        prog = HippoRAGProgram(
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

        # Boundary: non-GBNF LM omits extra_body.grammar
        non_gbnf_lm = CapabilityDummyLM([{"query_intent": "Erik Hort birthplace county"}])
        non_gbnf_lm._yaw_recipe = "flm"
        non_gbnf_lm._yaw_supports_gbnf = False
        intent_non_gbnf = prog.extract_query_intent("What county is Erik Hort's birthplace a part of?", lm=non_gbnf_lm)
        self.assertEqual(intent_non_gbnf, "Erik Hort birthplace county")
        self.assertIsNone(non_gbnf_lm.history[0]["kwargs"].get("extra_body"))

    def test_canonical_12_predictor_separation_of_concerns_and_distinct_signatures(self):
        """Canonical predictors retain distinct signatures and native role/context LM bindings."""
        import dspy

        from yaw.daemon.graph.inference import (
            DSPY_MISSING_ENTITY_DEFS_GBNF_GRAMMAR,
            DSPY_SHACL_REPAIR_GBNF_GRAMMAR,
            DSPY_SYNONYM_JUDGE_GBNF_GRAMMAR,
            HippoRAGProgram,
            RLMInstructionProposer,
            parse_shacl_repairs,
            parse_synonym_decisions,
        )

        primary_lm = CapabilityDummyLM([{"named_entities": [["Ada Lovelace", "English mathematician"]]}])
        sub_lm = CapabilityDummyLM(
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
                {"decisions": [{"pair_id": "ent-1|ent-2", "relation": "skos:altLabel", "canonical": "A"}]},
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
        sub_lm._yaw_recipe = "llamacpp"
        sub_lm._yaw_supports_gbnf = True

        dspy.settings.configure(lm=primary_lm)
        prog = HippoRAGProgram(
            enable_gbnf=True,
            orchestrator_lm=primary_lm,
            rlm_sub_lm=sub_lm,
        )
        proposer = RLMInstructionProposer(
            orchestrator_lm=primary_lm,
            worker_lm=sub_lm,
            enable_gbnf=True,
        )

        stage_preds = prog.all_stage_predictors()
        refl_preds = proposer.all_reflection_predictors()
        self.assertEqual(len(stage_preds), 9)
        self.assertEqual(len(refl_preds), 3)

        all_12 = dict(stage_preds + refl_preds)
        self.assertEqual(len(all_12), 12)
        self.assertEqual(
            list(all_12.keys()),
            [
                "ner",
                "triples",
                "entity_definer",
                "synonym_judge",
                "shacl_repair",
                "shacl_healer_rlm",
                "triple_filter",
                "query_intent",
                "qa",
                "inner_rlm",
                "qwen_proposer",
                "qwen_compressor",
            ],
        )

        # Pairwise-distinct Signature classes across all 12 canonical predictors
        sig_classes = [p.signature for p in all_12.values()]
        self.assertEqual(len(set(sig_classes)), 12)

        # Native role bindings share the actual LM, not artificial per-predictor
        # proxies. The native writer reads the worker's scoped rollout context.
        self.assertIs(prog.ner.lm, sub_lm)
        self.assertIs(prog.qa.predict.lm, primary_lm)
        self.assertIs(prog.shacl_healer_rlm.get_lm(), primary_lm)
        self.assertIs(proposer.inner_rlm.get_lm(), primary_lm)
        self.assertIsNone(proposer.qwen_proposer.lm)
        self.assertIsNone(proposer.qwen_compressor.lm)

        # Test first-class embedding telemetry trail on HippoRAGProgram
        prog.record_embedding_telemetry(
            {
                "stage": "fact_embed",
                "predictor": "embedder",
                "model": "openai/embeddinggemma-300m-GGUF",
                "mode": "embed",
                "num_inputs": 4,
                "tokens_used": 64,
            }
        )
        self.assertEqual(len(prog.embedding_audit_trail), 1)
        self.assertEqual(prog.embedding_audit_trail[0]["stage"], "fact_embed")
        self.assertEqual(prog.telemetry[-1]["predictor"], "embedder")

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
            mode="direct",
        )
        self.assertEqual(
            repairs,
            [
                {
                    "focus_node": "ent-1",
                    "action": "set_definition",
                    "value": "Valid non-empty definition",
                }
            ],
        )

        # Verify GBNF grammars and parser edge cases
        self.assertIn("entity_definitions", DSPY_MISSING_ENTITY_DEFS_GBNF_GRAMMAR)
        self.assertIn("skos:altLabel", DSPY_SYNONYM_JUDGE_GBNF_GRAMMAR)
        self.assertIn("set_definition", DSPY_SHACL_REPAIR_GBNF_GRAMMAR)
        self.assertEqual(parse_synonym_decisions("invalid json"), [])
        self.assertEqual(parse_shacl_repairs("invalid json"), [])

    def test_stage1_to_stage2_handoff_polymorphic_triples_and_define_missing_entities(self):
        """Verify Stage 1 -> Stage 2 [[name, def], ...] handoff, dynamic known_entities GBNF, DefineMissingEntities \
repair, and Pareto penalty."""
        import dspy

        from yaw.daemon.graph.inference import (
            DefineMissingEntities,
            HippoRAGProgram,
            build_dspy_missing_entity_defs_grammar,
            build_dspy_triples_grammar,
            compute_sizecoding_pareto_score,
        )
        from yaw.daemon.graph.inference.parsers import extract_structured_triples

        self.assertTrue(issubclass(DefineMissingEntities, dspy.Signature))

        # 1. Dynamic GBNF grammar for known_entities and missing_entity_defs
        g_known = build_dspy_triples_grammar(
            max_triples=3,
            known_entities=[["Radio City", "FM station"], ["Bangalore", "City"]],
        )
        self.assertIn('known-entity ::= "\\"Radio City\\"" | "\\"Bangalore\\""', g_known)
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

        # 3. Direct GBNF: Stage 1 defines "Ada Lovelace", Stage 2 introduces bare "Charles Babbage" without inline
        # pair -> DefineMissingEntities repairs it
        direct_lm = CapabilityDummyLM(
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
        direct_lm._yaw_recipe = "llamacpp"
        direct_lm._yaw_supports_gbnf = True

        prog_direct = HippoRAGProgram(
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

        # 3b. Cross-chunk isolation: Chunk 2 referencing bare "Charles Babbage" must NOT leak Chunk 1's repaired
        # definition
        chunk2_lm = CapabilityDummyLM(
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
        chunk2_lm._yaw_recipe = "llamacpp"
        chunk2_lm._yaw_supports_gbnf = True
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

        # 3c. Cinderella Stage 1 -> Stage 2 handoff: known entities as strings + inline new entity pair ["glass
        # slipper", "..."]
        cinderella_lm = CapabilityDummyLM(
            [
                {
                    "named_entities": [
                        [
                            "Cinderella",
                            "Young woman mistreated by her stepmother who attends the royal ball",
                        ],
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
        cinderella_lm._yaw_recipe = "llamacpp"
        cinderella_lm._yaw_supports_gbnf = True
        prog_cinderella = HippoRAGProgram(
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
                (
                    "Cinderella",
                    "Young woman mistreated by her stepmother who attends the royal ball",
                ),
                ("a ball", "Royal dance hosted by the prince"),
                ("glass slipper", "Delicate footwear lost by Cinderella at the ball"),
            ),
        )
        self.assertEqual(len(rec_cin.structured_triples), 3)
        # Third triple referencing plain "glass slipper" receives the inline definition propagated from the second
        # triple
        self.assertEqual(
            rec_cin.structured_triples[2].subj_def,
            "Delicate footwear lost by Cinderella at the ball",
        )
        self.assertEqual(
            rec_cin.structured_triples[2].obj_def,
            "Young woman mistreated by her stepmother who attends the royal ball",
        )

        # 4. Repairs are diagnostic and do not penalize task quality.
        clean_pareto = compute_sizecoding_pareto_score(0.80, tokens=100)
        repair_pareto = compute_sizecoding_pareto_score(
            0.80,
            tokens=100,
            missing_entity_defs_count=2,
            missing_entity_defs_tokens=60,
        )
        self.assertEqual(repair_pareto.score, clean_pareto.score)
        self.assertEqual(repair_pareto.objective_scores, clean_pareto.objective_scores)
        self.assertIn(
            "[entity_definer] repaired 2 missing Stage-2 entity definition(s) (60 aux tokens)",
            repair_pareto.feedback,
        )

    def test_reflection_preserves_expected_training_answers(self):
        from yaw.daemon.graph.inference.reflection import (
            deterministic_evidence,
            reflection_examples,
            resolve_evidence,
        )

        examples = reflection_examples([{"score": i, "Feedback": f"Expected=['answer-{i}']"} for i in range(8)])
        evidence = resolve_evidence(examples, deterministic_evidence(examples))
        self.assertEqual(len(evidence), 5)
        self.assertEqual([row["evidence"]["score"] for row in evidence], [0, 1, 2, 7, 6])
        self.assertIn("answer-0", evidence[0]["evidence"]["Feedback"])

    def test_rlm_instruction_proposer_requires_task_reward_for_simba(self):
        from yaw.daemon.graph.inference import RLMInstructionProposer

        proposer = RLMInstructionProposer()
        self.assertTrue(proposer.enable_simba)
        self.assertFalse(proposer.step_simba())
        self.assertEqual(proposer.dump_simba_state()["version"], 2)
        self.assertIn("shortened_instruction", proposer.qwen_compressor.signature.output_fields)

    def test_sprint1_configure_yaw_dspy_cache_and_zero_deprecation_warning(self) -> None:
        import tempfile
        import warnings
        from pathlib import Path

        import dspy

        import yaw.daemon.graph.inference as inf_pkg
        from yaw.daemon.graph.inference.chat import (
            YawDSPyLM,
            configure_yaw_dspy_cache,
            create_dspy_lm,
        )

        self.assertIn("configure_yaw_dspy_cache", inf_pkg.__all__)
        self.assertIs(inf_pkg.configure_yaw_dspy_cache, configure_yaw_dspy_cache)
        self.assertNotIn("forward", YawDSPyLM.__dict__)

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            cache_path = Path(tmp_dir) / "dspy_cache"
            configure_yaw_dspy_cache(
                cache_path,
                enable_disk_cache=True,
                enable_memory_cache=True,
                restrict_pickle=True,
            )
            self.assertTrue(getattr(dspy.cache, "enable_disk_cache", False))
            self.assertTrue(getattr(dspy.cache, "enable_memory_cache", False))

            configure_yaw_dspy_cache(
                cache_path,
                enable_disk_cache=False,
                enable_memory_cache=False,
                restrict_pickle=True,
            )
            self.assertFalse(getattr(dspy.cache, "enable_disk_cache", True))
            self.assertFalse(getattr(dspy.cache, "enable_memory_cache", True))

        # Restore default cache outside temp dir so temp dir can be removed cleanly
        configure_yaw_dspy_cache(enable_disk_cache=True, enable_memory_cache=True)

        lm = create_dspy_lm()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            copied = lm.copy(temperature=0.35)
            self.assertIsInstance(copied, YawDSPyLM)
            legacy_warnings = [w for w in caught if "BaseLM.forward()" in str(w.message)]
            self.assertEqual(len(legacy_warnings), 0)

    def test_sprint1_yaw_gbnf_adapter_contract(self) -> None:
        import dspy

        import yaw.daemon.graph.inference as inf_pkg
        from yaw.daemon.graph.inference import (
            AnswerQuestion,
            CompressInstructionSignature,
            DefineMissingEntities,
            ExtractEntities,
            ExtractQueryIntent,
            ExtractTriples,
            FilterTriples,
            HealSHACLViolations,
            HippoRAGProgram,
            JudgeSynonyms,
            ProposeInstructionSignature,
            ProposeSHACLRepairs,
            ReflectiveInnerDistillerSignature,
            RLMInstructionProposer,
            YawGBNFAdapter,
            create_dspy_lm,
        )

        self.assertIn("YawGBNFAdapter", inf_pkg.__all__)
        self.assertTrue(issubclass(YawGBNFAdapter, dspy.JSONAdapter))

        prog = HippoRAGProgram()
        proposer = RLMInstructionProposer()
        self.assertIsInstance(prog.json_adapter, YawGBNFAdapter)
        self.assertIsInstance(proposer.json_adapter, YawGBNFAdapter)

        adapter = YawGBNFAdapter(enable_gbnf=True, max_entities=16, max_facts=4)
        for direct_sig in (
            ExtractEntities,
            ExtractTriples,
            DefineMissingEntities,
            JudgeSynonyms,
            ProposeSHACLRepairs,
            ExtractQueryIntent,
            FilterTriples,
            AnswerQuestion,
            ProposeInstructionSignature,
            CompressInstructionSignature,
        ):
            grammar = adapter.resolve_grammar(direct_sig)
            self.assertIsInstance(grammar, str, f"Expected grammar string for {direct_sig.__name__}")
            self.assertIn("root ::=", str(grammar))

        for rlm_sig in (HealSHACLViolations, ReflectiveInnerDistillerSignature):
            self.assertIsNone(adapter.resolve_grammar(rlm_sig))

        gbnf_lm = create_dspy_lm(role="extraction")
        rlm_lm = create_dspy_lm(role="rlm")

        kw_enabled: dict[str, Any] = {}
        adapter._prepare_gbnf_kwargs(gbnf_lm, kw_enabled, ExtractEntities)
        self.assertIn("extra_body", kw_enabled)
        self.assertIn("grammar", kw_enabled["extra_body"])

        kw_rlm: dict[str, Any] = {}
        adapter._prepare_gbnf_kwargs(rlm_lm, kw_rlm, ExtractEntities)
        self.assertNotIn("extra_body", kw_rlm)

        disabled_adapter = YawGBNFAdapter(enable_gbnf=False)
        kw_disabled: dict[str, Any] = {}
        disabled_adapter._prepare_gbnf_kwargs(gbnf_lm, kw_disabled, ExtractEntities)
        self.assertNotIn("extra_body", kw_disabled)

    def test_sprint1_dspy_embedding_adapter_asyncify_and_abatch_encode(self) -> None:
        import anyio
        import dspy
        import numpy as np

        from yaw.daemon.graph.inference.embedding import (
            FACT_INSTRUCTION,
            DSPyEmbeddingAdapter,
            resolve_embedding_profile,
        )

        profile = resolve_embedding_profile("embeddinggemma-300m-GGUF", "distinct-titled")
        call_records: list[dict[str, Any]] = []

        def fake_embed_fn(batch_inputs: list[str], **_kwargs: Any) -> np.ndarray:
            call_records.append(
                {
                    "batch_len": len(batch_inputs),
                    "async_max_workers": dspy.settings.async_max_workers,
                }
            )
            out = np.zeros((len(batch_inputs), profile.dimensions), dtype=np.float32)
            for idx, text in enumerate(batch_inputs):
                seed = sum(ord(ch) for ch in text) % 97
                out[idx, seed % profile.dimensions] = 1.0
                out[idx, (seed + 1) % profile.dimensions] = 0.5
            return out

        embedder = dspy.Embedder(fake_embed_fn, batch_size=2, caching=False)
        adapter = DSPyEmbeddingAdapter(profile=profile, embedder=embedder, batch_size=2, caching=False)
        self.assertTrue(issubclass(DSPyEmbeddingAdapter, dspy.Embedder))
        self.assertIsInstance(adapter, dspy.Embedder)

        texts = [
            "# Alpha\nAlpha is the first concept.",
            "# Beta\nBeta is the second concept.",
            "# Gamma\nGamma is the third concept.",
            "# Delta\nDelta is the fourth concept.",
        ]
        sync_vecs = adapter.batch_encode(texts, batch_size=2, stage="entity_embed")
        self.assertEqual(sync_vecs.shape, (4, profile.dimensions))

        call_records.clear()
        with dspy.context(async_max_workers=3):
            async_vecs = anyio.run(lambda: adapter.abatch_encode(texts, batch_size=2, stage="entity_embed"))
            acall_vecs = anyio.run(lambda: adapter.acall(texts, batch_size=2, stage="entity_embed"))
            async_fn = adapter.as_async()
            asyncified_vecs = anyio.run(lambda: async_fn(texts, instruction=FACT_INSTRUCTION, batch_size=2))

        np.testing.assert_allclose(sync_vecs, async_vecs, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(sync_vecs, acall_vecs, rtol=1e-5, atol=1e-6)
        self.assertEqual(asyncified_vecs.shape, (4, profile.dimensions))
        self.assertTrue(all(r["async_max_workers"] == 3 for r in call_records))

        with self.assertRaises(ValueError):
            anyio.run(lambda: adapter.abatch_encode([]))
        with self.assertRaises(ValueError):
            anyio.run(lambda: adapter.abatch_encode(["hello"], instruction="not-valid"))

    def test_sprint2_yaw_rm_and_yaw_retriever_contract(self) -> None:
        import dspy

        import yaw.daemon.graph.inference as inf_pkg
        from yaw.daemon.graph.inference import (
            HippoRAGProgram,
            YawPassage,
            YawRetriever,
            YawRM,
        )

        for sym in ("YawPassage", "YawRM", "YawRetriever"):
            self.assertIn(sym, inf_pkg.__all__)

        psg = YawPassage(long_text="Erik Hort was born in Montebello.", pid="chunk-0", score=0.91, rank=1)
        self.assertEqual(psg.long_text, "Erik Hort was born in Montebello.")
        self.assertEqual(psg.pid, "chunk-0")
        self.assertAlmostEqual(psg.score, 0.91)
        self.assertEqual(psg.rank, 1)

        unbound_rm = YawRM(hook=None, default_k=2)
        with self.assertRaises(ValueError):
            unbound_rm("What county is Montebello in?")

        def fake_hook(
            query: str,
            program: Any = None,
            *,
            item_docs: Any = None,
        ) -> tuple[str, tuple[str, ...], tuple[float, ...]]:
            _ = (query, program, item_docs)
            passages = (
                "Montebello is in Rockland County.",
                "Erik Hort was born in Montebello.",
                "Distractor passage.",
            )
            scores = (0.95, 0.82, 0.12)
            return "\n\n".join(passages[:2]), passages, scores

        rm = YawRM(hook=fake_hook, default_k=2)
        rm_hits = rm("What county is Erik Hort's birthplace a part of?", k=2)
        self.assertEqual(len(rm_hits), 2)
        self.assertIsInstance(rm_hits[0], YawPassage)
        self.assertEqual(rm_hits[0].long_text, "Montebello is in Rockland County.")
        self.assertAlmostEqual(rm_hits[0].score, 0.95)

        # Standard dspy.Retrieve works out-of-the-box when YawRM is configured in dspy.context
        with dspy.context(rm=rm):
            stock_retrieve = dspy.Retrieve(k=2)
            stock_pred = stock_retrieve("What county is Erik Hort's birthplace a part of?")
            self.assertEqual(
                stock_pred.passages,
                [
                    "Montebello is in Rockland County.",
                    "Erik Hort was born in Montebello.",
                ],
            )

        self.assertTrue(issubclass(YawRetriever, dspy.Retrieve))
        qa_lm = CapabilityDummyLM([{"reasoning": "Montebello is in Rockland County.", "answer": "Rockland County"}])
        prog = HippoRAGProgram(orchestrator_lm=qa_lm, rlm_sub_lm=qa_lm)
        self.assertIsInstance(prog.rm, YawRM)
        self.assertIsInstance(prog.retriever, YawRetriever)
        self.assertEqual(len(prog.named_predictors()), 10)

        prog.set_retrieval_hook(fake_hook)
        self.assertIs(prog.rm.hook, fake_hook)
        ret_pred = prog.retriever("What county?", k=2)
        self.assertEqual(len(ret_pred.passages), 2)
        self.assertEqual(len(ret_pred.scores), 2)

        fwd_pred = prog(question="What county is Erik Hort's birthplace a part of?")
        self.assertEqual(fwd_pred.answer, "Rockland County")
        self.assertEqual(len(fwd_pred.retrieved_passages), 3)

        # Verify dump_state / load_state round-trip AND backward compatibility when 'retriever' is absent
        dumped = prog.dump_state()
        self.assertIn("retriever", dumped)
        legacy_state = {k: v for k, v in dumped.items() if k != "retriever"}
        prog2 = HippoRAGProgram(orchestrator_lm=qa_lm, rlm_sub_lm=qa_lm)
        prog2.load_state(legacy_state)
        self.assertEqual(prog2.retriever.k, 10)

    def test_sprint2_validate_stage_invariants_and_refine_retry(self) -> None:
        import dspy

        import yaw.daemon.graph.inference as inf_pkg
        from yaw.daemon.graph.inference import (
            HippoRAGProgram,
            YawAssertionError,
            is_abstaining_answer,
            validate_stage_invariants,
            wrap_stage_with_refine,
            yaw_assert,
            yaw_suggest,
        )

        for sym in (
            "YawAssertionError",
            "yaw_assert",
            "yaw_suggest",
            "validate_stage_invariants",
            "wrap_stage_with_refine",
            "is_abstaining_answer",
        ):
            self.assertIn(sym, inf_pkg.__all__)

        with self.assertRaises(YawAssertionError):
            yaw_assert(False, "closed-world violation")
        self.assertFalse(yaw_suggest(False, "soft suggestion"))
        self.assertTrue(yaw_suggest(True, "ok"))

        self.assertTrue(is_abstaining_answer("I don't know"))
        self.assertTrue(is_abstaining_answer("Not enough information provided in the context."))
        self.assertFalse(is_abstaining_answer("Rockland County"))

        # 1. Stage 1: ner (ConceptShape/prefLabel & ConceptShape/definition)
        ner_ok_score, ner_ok_fb = validate_stage_invariants(
            "ner",
            {"passage": "Ada Lovelace was an English mathematician who wrote Note G."},
            dspy.Prediction(
                named_entities=[
                    ["Ada Lovelace", "English mathematician who wrote Note G"],
                    ["Note G", "First published computer algorithm"],
                ]
            ),
        )
        self.assertEqual(ner_ok_score, 1.0)
        self.assertIn("PASS", ner_ok_fb)

        ner_taut_score, ner_taut_fb = validate_stage_invariants(
            "ner",
            {"passage": "Ada Lovelace was an English mathematician who wrote Note G."},
            dspy.Prediction(named_entities=[["Ada Lovelace", "Ada Lovelace"]]),
        )
        self.assertLess(ner_taut_score, 1.0)
        self.assertIn("ConceptShape/definition", ner_taut_fb)

        ner_dup_score, ner_dup_fb = validate_stage_invariants(
            "ner",
            {"passage": "New York is a state in the Northeastern United States."},
            dspy.Prediction(
                named_entities=[
                    ["New York", "State in Northeastern US"],
                    ["new york", "State in Northeastern US"],
                ]
            ),
        )
        self.assertLess(ner_dup_score, 1.0)
        self.assertIn("ConceptShape/prefLabel", ner_dup_fb)

        # Same label with distinct definitions remains a provisional sense split;
        # semantic adjudication belongs to the synonym stage.
        homograph_score, _ = validate_stage_invariants(
            "ner",
            {"passage": "Mercury can refer to a planet or a chemical element."},
            dspy.Prediction(
                named_entities=[
                    ["Mercury", "Planet nearest the Sun"],
                    ["mercury", "Chemical element with symbol Hg"],
                ]
            ),
        )
        self.assertEqual(homograph_score, 1.0)

        # 2. Stage 2: triples (StatementShape/closedWorld & Connected Nodes)
        stage1_ents = [
            ["Ada Lovelace", "English mathematician"],
            ["Note G", "First computer algorithm"],
        ]
        triples_ok_score, _ = validate_stage_invariants(
            "triples",
            {"passage": "Ada Lovelace wrote Note G.", "named_entities": stage1_ents},
            dspy.Prediction(
                triples=[
                    {
                        "sub": "Ada Lovelace",
                        "pred": "wrote",
                        "obj": "Note G",
                        "def": "Ada Lovelace wrote the Note G algorithm.",
                    }
                ]
            ),
        )
        self.assertEqual(triples_ok_score, 1.0)

        # Novel entity WITHOUT inline [name, def] violates StatementShape/closedWorld
        triples_cw_score, triples_cw_fb = validate_stage_invariants(
            "triples",
            {
                "passage": "Ada Lovelace collaborated with Charles Babbage.",
                "named_entities": stage1_ents,
            },
            dspy.Prediction(
                triples=[
                    {
                        "sub": "Ada Lovelace",
                        "pred": "collaborated with",
                        "obj": "Charles Babbage",
                        "def": "Ada Lovelace collaborated with Charles Babbage.",
                    }
                ]
            ),
        )
        self.assertLess(triples_cw_score, 1.0)
        self.assertIn("StatementShape/closedWorld", triples_cw_fb)

        # Novel entity WITH inline [name, def] satisfies StatementShape/closedWorld
        triples_inline_score, _ = validate_stage_invariants(
            "triples",
            {
                "passage": "Ada Lovelace collaborated with Charles Babbage.",
                "named_entities": [["Ada Lovelace", "English mathematician"]],
            },
            dspy.Prediction(
                triples=[
                    {
                        "sub": "Ada Lovelace",
                        "pred": "collaborated with",
                        "obj": ["Charles Babbage", "English mechanical computer pioneer"],
                        "def": "Ada Lovelace collaborated with Charles Babbage.",
                    }
                ]
            ),
        )
        self.assertEqual(triples_inline_score, 1.0)

        # Disconnected components across >= 2 entities violates connected-node topology
        triples_disc_score, triples_disc_fb = validate_stage_invariants(
            "triples",
            {
                "passage": "Alpha connects to Beta while Gamma connects to Delta.",
                "named_entities": [
                    ["Alpha", "Node A"],
                    ["Beta", "Node B"],
                    ["Gamma", "Node C"],
                    ["Delta", "Node D"],
                ],
            },
            dspy.Prediction(
                triples=[
                    {"sub": "Alpha", "pred": "links", "obj": "Beta", "def": "Alpha links to Beta."},
                    {
                        "sub": "Gamma",
                        "pred": "links",
                        "obj": "Delta",
                        "def": "Gamma links to Delta.",
                    },
                ]
            ),
        )
        self.assertLess(triples_disc_score, 1.0)
        self.assertIn("Disconnected subgraph", triples_disc_fb)

        # 3. Stage 3: entity_definer
        ed_ok, _ = validate_stage_invariants(
            "entity_definer",
            {"missing_entities": ["Charles Babbage"]},
            dspy.Prediction(entity_definitions=[["Charles Babbage", "English polymath and inventor"]]),
        )
        self.assertEqual(ed_ok, 1.0)
        ed_bad, ed_bad_fb = validate_stage_invariants(
            "entity_definer",
            {"missing_entities": ["Charles Babbage", "Analytical Engine"]},
            dspy.Prediction(entity_definitions=[["Charles Babbage", "Charles Babbage"]]),
        )
        self.assertLess(ed_bad, 1.0)
        self.assertIn("ConceptShape/definition", ed_bad_fb)

        # 4. Stage 4: synonym_judge (S13, homographDifferentFrom, EdgeShape/closedWorld)
        syn_ok, _ = validate_stage_invariants(
            "synonym_judge",
            {"candidate_pairs": '[{"pair_id": "ent-1|ent-2"}]'},
            dspy.Prediction(decisions=[{"pair_id": "ent-1|ent-2", "relation": "skos:altLabel", "canonical": "A"}]),
        )
        self.assertEqual(syn_ok, 1.0)
        syn_bad, syn_bad_fb = validate_stage_invariants(
            "synonym_judge",
            {"candidate_pairs": '[{"pair_id": "ent-1|ent-1"}]'},
            dspy.Prediction(
                decisions=[
                    {"pair_id": "ent-1|ent-1", "relation": "skos:altLabel", "canonical": "Z"},
                ]
            ),
        )
        self.assertLess(syn_bad, 1.0)
        self.assertIn("EdgeShape/closedWorld", syn_bad_fb)

        # 5. Stage 5: shacl_repair & shacl_healer_rlm (6 Closed-World SHACL Shapes)
        shacl_ok, _ = validate_stage_invariants(
            "shacl_repair",
            {"violations": '[{"focus_node": "ent-1", "source_shape": "ConceptShape/definition"}]'},
            dspy.Prediction(
                repairs=[
                    {
                        "focus_node": "ent-1",
                        "action": "set_definition",
                        "value": "Valid contextual definition",
                    }
                ]
            ),
            shacl_linter=lambda: [],
        )
        self.assertEqual(shacl_ok, 1.0)
        shacl_bad, shacl_bad_fb = validate_stage_invariants(
            "shacl_repair",
            {"violations": '[{"focus_node": "ent-1", "source_shape": "ConceptShape/definition"}]'},
            dspy.Prediction(repairs=[{"focus_node": "ent-999", "action": "set_definition", "value": ""}]),
            shacl_linter=lambda: [{"focus_node": "ent-1", "source_shape": "ConceptShape/definition"}],
        )
        self.assertLess(shacl_bad, 1.0)
        self.assertIn("ConceptShape/definition", shacl_bad_fb)

        # 6. Stage 6: query_intent
        qi_ok, _ = validate_stage_invariants(
            "query_intent",
            {"question": "What county is Montebello in?"},
            dspy.Prediction(query_intent="Montebello village town county location"),
        )
        self.assertEqual(qi_ok, 1.0)
        qi_bad, _ = validate_stage_invariants(
            "query_intent",
            {"question": "What county is Montebello in?"},
            dspy.Prediction(query_intent=""),
        )
        self.assertEqual(qi_bad, 0.0)

        # 7. Stage 7: triple_filter
        tf_ok, _ = validate_stage_invariants(
            "triple_filter",
            {
                "question": "Where was Erik Hort born?",
                "fact_before_filter": '{"fact": [["Erik Hort", "born in", "Montebello"]]}',
                "max_facts": 4,
            },
            dspy.Prediction(fact=[["Erik Hort", "born in", "Montebello"]]),
        )
        self.assertEqual(tf_ok, 1.0)
        tf_bad, tf_bad_fb = validate_stage_invariants(
            "triple_filter",
            {
                "question": "Where was Erik Hort born?",
                "fact_before_filter": '{"fact": [["Erik Hort", "born in", "Montebello"]]}',
                "max_facts": 4,
            },
            dspy.Prediction(fact=[["Hallucinated", "born in", "Nowhere"]]),
        )
        self.assertLess(tf_bad, 1.0)
        self.assertIn("closed-world", tf_bad_fb)

        # 8. Stage 8: qa
        qa_ok, _ = validate_stage_invariants(
            "qa",
            {"context": "Montebello is in Rockland County.", "question": "What county?"},
            dspy.Prediction(reasoning="Montebello is in Rockland County.", answer="Rockland County"),
        )
        self.assertEqual(qa_ok, 1.0)
        qa_abstain, _ = validate_stage_invariants(
            "qa",
            {"context": "Montebello is in Rockland County.", "question": "What county?"},
            dspy.Prediction(reasoning="Unclear.", answer="I don't know"),
        )
        self.assertLess(qa_abstain, 1.0)

        # 9. Stage 9: qwen_proposer / qwen_compressor
        prop_ok, _ = validate_stage_invariants(
            "qwen_proposer",
            {
                "component_name": "ner",
                "current_instruction": "Return JSON with key `named_entities`.",
            },
            dspy.Prediction(
                new_instruction="Extract all salient entities with definitions. Return JSON with key `named_entities`."
            ),
        )
        self.assertEqual(prop_ok, 1.0)
        prop_leak, prop_leak_fb = validate_stage_invariants(
            "qwen_proposer",
            {
                "component_name": "ner",
                "current_instruction": "Return JSON with key `named_entities`.",
            },
            dspy.Prediction(new_instruction="Extract <TargetEntity> from the passage."),
        )
        self.assertLess(prop_leak, 1.0)
        self.assertIn("named_entities", prop_leak_fb)

        # Verify wrap_stage_with_refine and feedback-guided _invoke_stage retry
        retry_lm = CapabilityDummyLM(
            [
                {"named_entities": [["Ada Lovelace", "Ada Lovelace"]]},
                {"named_entities": [["Ada Lovelace", "English mathematician and writer"]]},
            ]
        )
        prog = HippoRAGProgram(orchestrator_lm=retry_lm, rlm_sub_lm=retry_lm)
        defs = prog.extract_entity_definitions("Ada Lovelace was an English mathematician and writer.")
        self.assertEqual(defs, (("Ada Lovelace", "English mathematician and writer"),))
        self.assertEqual(prog.telemetry[-1].get("invariant_retries"), 1)
        self.assertEqual(prog.telemetry[-1].get("invariant_score"), 1.0)

        refined_mod = wrap_stage_with_refine(prog.ner, "ner", N=2, threshold=1.0, use_refine=False)
        self.assertIsInstance(refined_mod, dspy.BestOfN)

    def test_sprint2_pareto_program_ensemble_and_react_v2_fallback(self) -> None:
        import anyio

        import yaw.daemon.graph.inference as inf_pkg
        from yaw.daemon.graph.inference import (
            HippoRAGProgram,
            ParetoProgramEnsemble,
        )

        self.assertIn("ParetoProgramEnsemble", inf_pkg.__all__)

        lm1 = CapabilityDummyLM(
            [
                {"named_entities": [["Erik Hort", "American soccer player"]]},
                {
                    "triples": [
                        {
                            "sub": "Erik Hort",
                            "pred": "born in",
                            "obj": ["Montebello", "Village in Rockland County, New York"],
                            "def": "Erik Hort was born in Montebello.",
                        }
                    ]
                },
                {
                    "reasoning": "Erik Hort was born in Montebello, Rockland County.",
                    "answer": "Rockland County",
                },
                {
                    "reasoning": "Erik Hort was born in Montebello, Rockland County.",
                    "answer": "Rockland County",
                },
            ]
        )
        lm2 = CapabilityDummyLM(
            [
                {
                    "named_entities": [
                        ["Erik Hort", "American professional soccer defender born in Montebello"],
                        ["Rockland County", "County in the state of New York"],
                    ]
                },
                {
                    "triples": [
                        {
                            "sub": "Erik Hort",
                            "pred": "born in",
                            "obj": ["Montebello", "Village in Rockland County, New York"],
                            "def": "Erik Hort was born in Montebello, New York.",
                        },
                        {
                            "sub": "Montebello",
                            "pred": "located in",
                            "obj": "Rockland County",
                            "def": "Montebello is located in Rockland County.",
                        },
                    ]
                },
                {"reasoning": "Not sure.", "answer": "I don't know"},
                {"reasoning": "Montebello is in Rockland County.", "answer": "Rockland County"},
            ]
        )

        cand1 = HippoRAGProgram(orchestrator_lm=lm1, rlm_sub_lm=lm1)
        cand2 = HippoRAGProgram(orchestrator_lm=lm2, rlm_sub_lm=lm2)
        ensemble = ParetoProgramEnsemble([cand1, cand2])
        self.assertTrue(getattr(ensemble, "_compiled", False))

        openie_rec = ensemble.extract_openie(
            "chunk-ens",
            "Erik Hort was born in Montebello, a village located in Rockland County.",
        )
        self.assertIn("Erik Hort", openie_rec.entities)
        self.assertIn("Montebello", openie_rec.entities)
        self.assertIn("Rockland County", openie_rec.entities)
        # Richest non-tautological definition preserved for Erik Hort
        def_map = dict(openie_rec.entity_definitions)
        self.assertIn("defender", def_map["Erik Hort"])
        # Shared triple ranked first
        self.assertEqual(openie_rec.triples[0], ("Erik Hort", "born in", "Montebello"))

        # QA ensemble selects non-abstaining consensus answer
        qa_ens_pred = ensemble.answer_question(
            "Erik Hort was born in Montebello, Rockland County.",
            "What county is Erik Hort's birthplace a part of?",
        )
        self.assertEqual(qa_ens_pred.answer, "Rockland County")

        async_qa_pred = anyio.run(
            lambda: ensemble.aanswer_question(
                "Erik Hort was born in Montebello, Rockland County.",
                "What county is Erik Hort's birthplace a part of?",
            )
        )
        self.assertEqual(async_qa_pred.answer, "Rockland County")

        # Verify Pareto frontier persistence in HippoRAGProgram.dump_state / load_state without polluting
        # named_predictors
        cand1.pareto_frontier_states = [cand1.dump_state(), cand2.dump_state()]
        cand1.pareto_ensemble = ensemble
        self.assertEqual(len(cand1.named_predictors()), 10)
        dumped_with_frontier = cand1.dump_state()
        self.assertEqual(len(dumped_with_frontier["yaw_surface"]["pareto_frontier_states"]), 2)

        restored_prog = HippoRAGProgram(orchestrator_lm=lm1, rlm_sub_lm=lm1)
        restored_prog.load_state(dumped_with_frontier)
        self.assertEqual(len(restored_prog.pareto_frontier_states), 2)
        self.assertIsInstance(restored_prog.pareto_ensemble, ParetoProgramEnsemble)
        self.assertEqual(len(restored_prog.named_predictors()), 10)

        # Verify dspy.ReActV2 multi-hop fallback when single-shot QA abstains or retrieval confidence < 0.15
        react_lm = CapabilityDummyLM(
            [
                # Single-shot QA call 1 & retry 2 abstain
                {"reasoning": "Missing second hop.", "answer": "I don't know"},
                {"reasoning": "Missing second hop.", "answer": "I don't know"},
                # ReActV2 step 1: call search_graph_facts
                {
                    "next_thought": {"content": "Search graph facts for Montebello county."},
                    "tool_calls": {"tool_calls": [{"name": "search_graph_facts", "args": {"query": "Montebello"}}]},
                },
                # ReActV2 step 2: finish with final answer
                {
                    "next_thought": {"content": "Montebello is in Rockland County."},
                    "tool_calls": {
                        "tool_calls": [
                            {
                                "name": "submit",
                                "args": {
                                    "reasoning": "Erik Hort was born in Montebello, which is in Rockland County.",
                                    "answer": "Rockland County",
                                },
                            }
                        ]
                    },
                },
            ]
        )
        react_prog = HippoRAGProgram(orchestrator_lm=react_lm, rlm_sub_lm=react_lm)
        react_prog.enable_react_fallback = True
        react_prog.set_retrieval_hook(
            lambda q, *_a, **_kw: (
                "Erik Hort was born in Montebello.",
                ("Erik Hort was born in Montebello.",),
                (0.08,),
            )
        )
        react_prog.set_graph_search_tool(lambda q: f"Fact for {q}: Montebello is located in Rockland County, New York.")
        react_pred = react_prog(question="What county is Erik Hort's birthplace a part of?")
        self.assertEqual(react_pred.answer, "Rockland County")
        self.assertEqual(len(react_prog.named_predictors()), 10)

        # Verify HippoRAGProgram.as_async()
        async_prog_lm = CapabilityDummyLM(
            [{"reasoning": "Montebello is in Rockland County.", "answer": "Rockland County"}]
        )
        async_prog = HippoRAGProgram(orchestrator_lm=async_prog_lm, rlm_sub_lm=async_prog_lm)
        async_prog_fn = async_prog.as_async()
        async_res = anyio.run(
            lambda: async_prog_fn(
                context="Montebello is in Rockland County.",
                question="What county is Montebello in?",
            )
        )
        self.assertEqual(async_res.answer, "Rockland County")


if __name__ == "__main__":
    unittest.main()
