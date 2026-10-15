"""Verify recipe-specific chat policies, split extraction/QA instances, and strict completion accounting."""
from contextlib import closing
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from hipporag.prompts import PromptTemplateManager
from hipporag.prompts.filter_default_prompt import best_dspy_prompt
from hipporag.utils.config_utils import BaseConfig
from hipporag.utils.llm_utils import PROMPT_JSON_TEMPLATE
from yaw.daemon.graph.inference.chat import CacheOpenAI, RERANK_GBNF_GRAMMAR, LocalChat, chat_identity
from yaw.daemon.graph.inference.models import CHAT_MODELS


class ChatTests(unittest.TestCase):
    def test_recipe_specific_init_and_model_override_preserve_base_config(self):
        with TemporaryDirectory() as directory:
            base = BaseConfig(
                save_dir=directory, llm_name="Phi-4-mini-instruct-GGUF",
                llm_base_url="http://localhost:13305/api/v1", max_retry_attempts=0,
            )
            for name, (_, recipe) in CHAT_MODELS.items():
                with closing(LocalChat.from_experiment_config(base, model_name=name)) as chat:
                    self.assertEqual(base.llm_name, "Phi-4-mini-instruct-GGUF")
                    params = chat.llm_config.generate_params
                    self.assertEqual(chat.llm_name, name)
                    self.assertEqual(params["model"], name)
                    self.assertEqual(chat.max_retries, 0)
                    self.assertTrue(chat.cache_file_name.endswith(f"{name.replace('/', '_')}_cache.sqlite"))
                    if recipe == "flm":
                        self.assertEqual(params["extra_body"], {"think": False})
                    else:
                        self.assertNotIn("extra_body", params)

    def test_ryzenai_folds_leading_system_message_and_unescapes_markdown_underscores(self):
        from yaw.daemon.graph.inference.chat import QA_TURN_REMINDER, TIGHTENED_QA_SYSTEM

        with TemporaryDirectory(ignore_cleanup_errors=True) as directory, closing(
            LocalChat.from_experiment_config(BaseConfig(
                save_dir=directory, llm_name="Mistral-7B-Instruct-v0.3-Hybrid",
                llm_base_url="http://localhost:13305/api/v1", max_retry_attempts=0,
            ))
        ) as chat:
            chat.openai_client.close()
            chat.openai_client = MagicMock()
            chat.openai_client.chat.completions.create.return_value = SimpleNamespace(
                id="resp-1", model="mistral",
                choices=[SimpleNamespace(message=SimpleNamespace(content=r'{"named\_entities": ["George"]}'), finish_reason="stop")],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            )
            ptm = PromptTemplateManager()
            raw_messages = ptm.render(name="ner", passage="George Rankin is a politician.")
            text, meta, hit = chat.infer(raw_messages, max_new_tokens=128)
            self.assertFalse(hit)
            self.assertEqual(text, '{"named_entities": ["George"]}')
            sent_messages = chat.openai_client.chat.completions.create.call_args.kwargs["messages"]
            self.assertEqual([m["role"] for m in sent_messages], ["user", "assistant", "user"])
            self.assertTrue(sent_messages[0]["content"].startswith(raw_messages[0]["content"].strip()))
            self.assertIn(raw_messages[1]["content"], sent_messages[0]["content"])
            self.assertNotIn(QA_TURN_REMINDER, sent_messages[-1]["content"])

            chat.openai_client.chat.completions.create.return_value = SimpleNamespace(
                id="resp-2", model="mistral",
                choices=[SimpleNamespace(message=SimpleNamespace(content="Reasoning.\nAnswer: George."), finish_reason="stop")],
                usage=SimpleNamespace(prompt_tokens=20, completion_tokens=5, total_tokens=25),
            )
            qa_messages = ptm.render(
                name="rag_qa_musique",
                prompt_user="Wikipedia Title: George Rankin\nGeorge Rankin is a politician.\n\nQuestion: Who is George?\nThought: ",
            )
            chat.infer(qa_messages)
            sent_qa = chat.openai_client.chat.completions.create.call_args.kwargs["messages"]
            self.assertEqual([m["role"] for m in sent_qa], ["user", "assistant", "user"])
            self.assertTrue(sent_qa[0]["content"].startswith(TIGHTENED_QA_SYSTEM))
            self.assertIn(f"\n{QA_TURN_REMINDER}\nQuestion: Who is George?\nThought: ", sent_qa[-1]["content"])

            self.assertEqual(chat.summary()["calls"], 2)
            self.assertEqual(chat.summary()["total_tokens"], 40)
            self.assertEqual(chat.errors, [])

    def test_llamacpp_applies_openie_json_schemas_and_dspy_rerank_gbnf_grammar(self):
        from yaw.daemon.graph.inference.chat import QA_GBNF_GRAMMAR, TIGHTENED_QA_SYSTEM

        with TemporaryDirectory(ignore_cleanup_errors=True) as directory, closing(
            LocalChat.from_experiment_config(BaseConfig(
                save_dir=directory, llm_name="Phi-4-mini-instruct-GGUF",
                llm_base_url="http://localhost:13305/api/v1", max_retry_attempts=0,
            ))
        ) as chat:
            chat.openai_client.close()
            chat.openai_client = MagicMock()
            chat.openai_client.chat.completions.create.return_value = SimpleNamespace(
                id="resp-phi", model="phi4",
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content='Reasoning.\nAnswer: George\n{"named_entities": ["George"]}'),
                    finish_reason="stop",
                )],
                usage=SimpleNamespace(prompt_tokens=12, completion_tokens=6, total_tokens=18),
            )
            ptm = PromptTemplateManager()
            chat.infer(ptm.render(name="ner", passage="George Rankin is a politician."))
            ner_kwargs = chat.openai_client.chat.completions.create.call_args.kwargs
            ner_schema = ner_kwargs["response_format"]["json_schema"]["schema"]
            self.assertEqual(ner_schema["properties"]["named_entities"]["items"], {"type": "string", "minLength": 1})
            self.assertEqual(PROMPT_JSON_TEMPLATE["ner"]["properties"]["named_entities"]["items"], {"type": "string"})

            chat.infer(ptm.render(name="triple_extraction", passage="George Rankin is a politician.", named_entity_json='{"named_entities": ["George"]}'))
            triple_kwargs = chat.openai_client.chat.completions.create.call_args.kwargs
            triple_schema = triple_kwargs["response_format"]["json_schema"]["schema"]
            self.assertEqual(
                triple_schema["properties"]["triples"]["items"]["items"],
                {"type": "string", "minLength": 1},
            )
            self.assertEqual(
                PROMPT_JSON_TEMPLATE["triples"]["properties"]["triples"]["items"]["items"],
                {"type": "string"},
            )

            chat.infer([
                {"role": "system", "content": best_dspy_prompt["prog"]["system"]},
                {"role": "user", "content": "[[ ## question ## ]]\nWho?"},
            ])
            rerank_kwargs = chat.openai_client.chat.completions.create.call_args.kwargs
            self.assertEqual(rerank_kwargs["extra_body"], {"grammar": RERANK_GBNF_GRAMMAR})

            chat.infer(ptm.render(name="rag_qa_musique", prompt_user="Question: Who?\nThought: "))
            qa_kwargs = chat.openai_client.chat.completions.create.call_args.kwargs
            self.assertEqual(qa_kwargs["messages"][0]["content"], TIGHTENED_QA_SYSTEM)
            self.assertNotIn("response_format", qa_kwargs)
            self.assertEqual(qa_kwargs["extra_body"], {"grammar": QA_GBNF_GRAMMAR})

    def test_qa_delimiter_normalization(self):
        from yaw.daemon.graph.inference.chat import _normalize_qa_response

        # QA delimiter normalization handles Markdown bolding, internal 'Answer:' mentions, and missing 'Answer:'
        normalized = _normalize_qa_response("The doc says Answer: maybe X.\n**Answer:** The glass slipper fit Cinderella.")
        self.assertEqual(normalized.split("Answer:")[1].strip(), "The glass slipper fit Cinderella.")
        self.assertEqual(normalized.count("Answer:"), 1)

        with self.assertRaisesRegex(ValueError, "omitted required 'Answer:'"):
            _normalize_qa_response("Just a thought without the final marker.")
        fallback = _normalize_qa_response("Step 1. Therefore, the answer is X.", allow_missing_delimiter=True)
        self.assertEqual(fallback.split("Answer:")[1].strip(), "Therefore, the answer is X.")
        with self.assertRaisesRegex(ValueError, "omitted required 'Answer:' content"):
            _normalize_qa_response("   ", allow_missing_delimiter=True)

    def test_dynamic_rerank_grammar_handles_all_candidate_counts_and_corpus_updates(self):
        from yaw.daemon.graph.inference.chat import _build_rerank_grammar

        def make_messages(fact_json_str):
            user_content = (
                f"[[ ## question ## ]]\nWho?\n\n"
                f"[[ ## fact_before_filter ## ]]\n{fact_json_str}\n\n"
                "Respond with [[ ## fact_after_filter ## ]]"
            )
            return [
                {"role": "system", "content": best_dspy_prompt["prog"]["system"]},
                {"role": "user", "content": user_content},
            ]

        # 1. Missing or malformed fact_before_filter -> static fallback
        self.assertEqual(_build_rerank_grammar([]), RERANK_GBNF_GRAMMAR)
        self.assertEqual(_build_rerank_grammar(make_messages("not-json")), RERANK_GBNF_GRAMMAR)
        self.assertEqual(_build_rerank_grammar(make_messages('{"fact": "bad"}')), RERANK_GBNF_GRAMMAR)
        self.assertEqual(_build_rerank_grammar(make_messages('{"fact": [["only", "two"]]}')), RERANK_GBNF_GRAMMAR)

        # 2. N = 0 (empty candidate list) -> forces {"fact": []} without empty triple ::= rule
        grammar_zero = _build_rerank_grammar(make_messages('{"fact": []}'))
        self.assertIn('"{\\"fact\\": []}"', grammar_zero)
        self.assertNotIn("triple ::=", grammar_zero)

        # 3. 1 <= N < 4 (N = 2) -> caps selections at 2 and includes both candidates
        grammar_two = _build_rerank_grammar(make_messages(
            '{"fact": [["george rankin", "is", "politician"], ["cinderella", "attended", "royal ball"]]}'
        ))
        self.assertIn('fact-list ::= (triple (", " triple)?)?', grammar_two)
        self.assertIn('"[\\"george rankin\\", \\"is\\", \\"politician\\"]"', grammar_two)
        self.assertIn('"[\\"cinderella\\", \\"attended\\", \\"royal ball\\"]"', grammar_two)

        # 4. N > 4 (N = 6 unique + 1 duplicate) -> includes all 6 unique candidates, caps selections at 4
        six_facts = [
            [f"s{i}", f"p{i}", f"o{i}"] for i in range(6)
        ] + [["s0", "p0", "o0"]]
        import json
        grammar_six = _build_rerank_grammar(make_messages(json.dumps({"fact": six_facts})))
        self.assertIn(
            'fact-list ::= (triple (", " triple (", " triple (", " triple)?)?)?)?',
            grammar_six,
        )
        self.assertEqual(grammar_six.count("triple"), 5)  # 4 in fact-list RHS + 1 in triple ::= LHS
        for i in range(6):
            self.assertIn(f'"[\\"s{i}\\", \\"p{i}\\", \\"o{i}\\"]"', grammar_six)

        # 5. Special characters (quotes, backslashes, unicode) are safely escaped
        grammar_special = _build_rerank_grammar(make_messages(
            json.dumps({"fact": [['cinderella "glass"', r"path\to", "münchen"]]})
        ))
        self.assertIn("münchen", grammar_special)
        self.assertIn('fact-list ::= (triple)?', grammar_special)

        # 6. Corpus update changes fact_before_filter -> real SQLite CacheOpenAI invalidates cache
        with TemporaryDirectory(ignore_cleanup_errors=True) as directory, closing(
            LocalChat.from_experiment_config(BaseConfig(
                save_dir=directory, llm_name="Qwen3-4B-Instruct-2507-GGUF",
                llm_base_url="http://localhost:13305/api/v1", max_retry_attempts=0,
            ))
        ) as chat:
            chat.openai_client.close()
            chat.openai_client = MagicMock()
            chat.openai_client.chat.completions.create.return_value = SimpleNamespace(
                id="resp-qwen", model="qwen3",
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content='[[ ## fact_after_filter ## ]]\n{"fact": []}\n\n[[ ## completed ## ]]'),
                    finish_reason="stop",
                )],
                usage=SimpleNamespace(prompt_tokens=20, completion_tokens=10, total_tokens=30),
            )
            msgs_v1 = make_messages('{"fact": [["a", "b", "c"]]}')
            _, _, hit1 = chat.infer(msgs_v1)
            self.assertFalse(hit1)
            _, _, hit1_repeat = chat.infer(msgs_v1)
            self.assertTrue(hit1_repeat)

            msgs_v2 = make_messages('{"fact": [["a", "b", "c"], ["d", "e", "f"]]}')
            _, _, hit2 = chat.infer(msgs_v2)
            self.assertFalse(hit2)
            self.assertEqual(chat.openai_client.chat.completions.create.call_count, 2)

    def test_truncation_and_nonterminal_responses_fail_and_record_errors(self):
        chat = object.__new__(LocalChat)
        chat.llm_name = "gemma4-it-12b-FLM"
        chat.calls = []
        chat.errors = []
        for finish in ("length", "tool_calls", None):
            with patch.object(CacheOpenAI, "infer", return_value=("partial", {"finish_reason": finish}, False)):
                with self.assertRaisesRegex(ValueError, "did not finish normally"):
                    chat.infer([])
        self.assertEqual(len(chat.errors), 3)
        expected = ("answer", {"finish_reason": "stop", "prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}, False)
        with patch.object(CacheOpenAI, "infer", return_value=expected):
            self.assertEqual(chat.infer([]), expected)
        self.assertEqual(len(chat.calls), 1)

    def test_openie_allows_length_truncation_for_json_repair(self):
        from hipporag.prompts.templates.triple_extraction import ner_conditioned_re_system
        chat = object.__new__(LocalChat)
        chat.llm_name = "Qwen3-4B-Instruct-2507-GGUF"
        chat.calls = []
        chat.errors = []
        chat.global_config = BaseConfig()
        openie_msgs = [{"role": "system", "content": ner_conditioned_re_system}, {"role": "user", "content": "text"}]
        raw = '{"triples": [["A", "B", "C"], ["D", "rambling with, commas, inside'
        meta = {"finish_reason": "length", "prompt_tokens": 100, "completion_tokens": 4096, "total_tokens": 4196}
        with patch.object(CacheOpenAI, "infer", return_value=(raw, meta, False)):
            res = chat.infer(openie_msgs)
            self.assertEqual(res, ('{"triples": [["A", "B", "C"]]}', meta, False))
        self.assertEqual(len(chat.errors), 0)
        self.assertEqual(len(chat.calls), 1)

    def test_chat_identity_distinguishes_all_registered_models_and_policies(self):
        self.assertIn("Phi-4-mini-instruct-GGUF", CHAT_MODELS)
        self.assertIn("Mistral-7B-Instruct-v0.3-Hybrid", CHAT_MODELS)
        self.assertEqual(len({chat_identity(name) for name in CHAT_MODELS}), len(CHAT_MODELS))
        self.assertTrue(all("chat-v1" in chat_identity(name) for name in CHAT_MODELS))

    def test_create_dspy_lm_configures_lemonade_and_ollama_models(self):
        from yaw.daemon.graph.inference.chat import create_dspy_lm

        # Qwen3-4B extraction on Lemonade (8192-16384 ctx window)
        qwen_lm = create_dspy_lm("Qwen3-4B-Instruct-2507-GGUF", role="extraction")
        self.assertEqual(qwen_lm.model, "openai/Qwen3-4B-Instruct-2507-GGUF")
        self.assertAlmostEqual(qwen_lm.kwargs["temperature"], 0.2)
        self.assertEqual(qwen_lm.kwargs["max_tokens"], 4096)
        self.assertAlmostEqual(qwen_lm.kwargs["frequency_penalty"], 0.5)

        # Mistral-7B QA/RLM on Lemonade (32786 ctx window)
        mistral_lm = create_dspy_lm("Mistral-7B-Instruct-v0.3-Hybrid", role="qa")
        self.assertEqual(mistral_lm.model, "openai/Mistral-7B-Instruct-v0.3-Hybrid")
        self.assertAlmostEqual(mistral_lm.kwargs["temperature"], 0.2)
        self.assertEqual(mistral_lm.kwargs["max_tokens"], 2048)

        # Gemma4 FLM recipe sets think=False
        flm_lm = create_dspy_lm("gemma4-it-12b-FLM", role="qa")
        self.assertEqual(flm_lm.kwargs.get("extra_body"), {"think": False})

        # Ollama provider configures ollama_chat/...
        ollama_lm = create_dspy_lm(
            "qwen3:4b",
            base_url="http://localhost:11434",
            provider="ollama",
            role="extraction",
        )
        self.assertEqual(ollama_lm.model, "ollama_chat/qwen3:4b")
        self.assertEqual(ollama_lm.kwargs.get("api_base"), "http://localhost:11434")


if __name__ == "__main__":
    unittest.main()
