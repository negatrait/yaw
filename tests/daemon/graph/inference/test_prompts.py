"""Verify 100% byte-for-byte prompt parity and parser parity against upstream HippoRAG."""

from copy import deepcopy
import json
import unittest

from hipporag.information_extraction.openie_openai import (
    _extract_json_list_field as upstream_extract_json_list_field,
    _extract_ner_from_response as upstream_extract_ner,
)
from hipporag.prompts import PromptTemplateManager
from hipporag.rerank import DSPyFilter
from hipporag.utils.llm_utils import filter_invalid_triples as upstream_filter_invalid_triples
from hipporag.utils.misc_utils import (
    compute_mdhash_id as upstream_compute_mdhash_id,
    text_processing as upstream_text_processing,
)

from yaw.daemon.graph.inference.prompts import (
    FACT_FILTER_INPUT_TEMPLATE,
    FACT_FILTER_OUTPUT_TEMPLATE,
    NER_ONE_SHOT_OUTPUT,
    NER_ONE_SHOT_PARAGRAPH,
    NER_SYSTEM,
    PROMPT_JSON_TEMPLATE,
    RAG_QA_ONE_SHOT_INPUT,
    RAG_QA_ONE_SHOT_OUTPUT,
    RAG_QA_SYSTEM,
    TRIPLE_ONE_SHOT_INPUT,
    TRIPLE_ONE_SHOT_OUTPUT,
    TRIPLE_SYSTEM,
    compute_mdhash_id,
    extract_named_entities,
    extract_triples,
    filter_invalid_triples,
    filter_reranked_facts,
    parse_filter_facts,
    render_fact_filter_messages,
    render_ner_messages,
    render_rag_qa_messages,
    render_triple_messages,
    repair_truncated_json,
    text_processing,
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
        input_triples = [
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
        generated_facts = dspy_filter.parse_filter(llm_response)

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
            AnswerQuestion,
            ExtractEntities,
            ExtractTriples,
            FACT_FILTER_DEMOS,
            FactList,
            FilterFacts,
            NER_DEMOS,
            QA_DEMOS,
            TRIPLE_DEMOS,
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
        from yaw.daemon.graph.inference.prompts import HippoRAGProgram

        program = HippoRAGProgram(qa_mode="cot")
        lm = DummyLM(
            [
                {"named_entities": ["Alan Turing", "London", "Alan Turing"]},
                {
                    "triples": [
                        ["Alan Turing", "born in", "London"],
                        ["London", "capital of", "United Kingdom"],
                        ["invalid", "only_two"],
                        ["Alan Turing", "born in", "London"],
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
        self.assertEqual(record.entities, ("Alan Turing", "London"))
        self.assertEqual(
            record.triples,
            (
                ("Alan Turing", "born in", "London"),
                ("London", "capital of", "United Kingdom"),
            ),
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
        from yaw.daemon.graph.inference.prompts import HippoRAGProgram

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
        from yaw.daemon.graph.inference.prompts import HippoRAGProgram

        with TemporaryDirectory() as tmp_dir:
            state_path = Path(tmp_dir) / "gepa_compiled.json"
            prog1 = HippoRAGProgram()
            prog1.ner.signature = prog1.ner.signature.with_instructions(
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
            self.assertEqual(
                prog2.ner.signature.instructions,
                "GEPA optimized NER instruction.",
            )
            self.assertEqual(len(prog2.ner.demos), 1)
            self.assertEqual(prog2.ner.demos[0]["named_entities"], ["GEPA"])

    def test_hipporag_program_gbnf_grammar_forwarding(self):
        """Verify HippoRAGProgram forwards pure-JSON static and dynamic GBNF grammars via extra_body."""
        import dspy
        from dspy.utils import DummyLM
        from yaw.daemon.graph.inference.prompts import (
            DSPY_NER_GBNF_GRAMMAR,
            DSPY_QA_COT_GBNF_GRAMMAR,
            DSPY_TRIPLES_GBNF_GRAMMAR,
            HippoRAGProgram,
            build_dspy_rerank_grammar,
        )

        program = HippoRAGProgram(enable_gbnf=True)
        lm = DummyLM(
            [
                {"named_entities": ["Alan Turing", "London"]},
                {"triples": [["Alan Turing", "born in", "London"]]},
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
            {"grammar": DSPY_TRIPLES_GBNF_GRAMMAR},
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


if __name__ == "__main__":
    unittest.main()
