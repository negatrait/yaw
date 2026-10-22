"""Verify native DSPy LM initialization, recipe policies, and retired legacy chat shims."""

import unittest

import yaw.daemon.graph.inference as inference_facade
import yaw.daemon.graph.inference.chat as chat_module
from yaw.daemon.graph.inference.chat import (
    create_dspy_lm,
    create_dspy_rlm_pair,
    lm_supports_gbnf,
)
from yaw.daemon.graph.inference.models import CHAT_MODELS


class ChatTests(unittest.TestCase):
    def test_chat_models_registry_contains_supported_models(self):
        self.assertIn("Phi-4-mini-instruct-GGUF", CHAT_MODELS)
        self.assertIn("Qwen3-4B-Instruct-2507-GGUF", CHAT_MODELS)
        self.assertIn("Gemma-4-E4B-it-GGUF", CHAT_MODELS)
        self.assertNotIn("Mistral-7B-Instruct-v0.3-Hybrid", CHAT_MODELS)

    def test_create_dspy_lm_configures_lemonade_and_ollama_models(self):
        # Qwen3-4B extraction on Lemonade (16384 ctx window, llamacpp GPU -> GBNF supported)
        qwen_lm = create_dspy_lm("Qwen3-4B-Instruct-2507-GGUF", role="extraction")
        self.assertEqual(qwen_lm.model, "openai/Qwen3-4B-Instruct-2507-GGUF")
        self.assertAlmostEqual(qwen_lm.kwargs["temperature"], 0.2)
        self.assertEqual(qwen_lm.kwargs["max_tokens"], 4096)
        self.assertAlmostEqual(qwen_lm.kwargs["frequency_penalty"], 0.5)
        self.assertTrue(lm_supports_gbnf(qwen_lm))

        # Qwen3-4B QA on Lemonade (llamacpp GPU -> GBNF supported for direct QA)
        qwen_qa_lm = create_dspy_lm("Qwen3-4B-Instruct-2507-GGUF", role="qa")
        self.assertEqual(qwen_qa_lm.model, "openai/Qwen3-4B-Instruct-2507-GGUF")
        self.assertAlmostEqual(qwen_qa_lm.kwargs["temperature"], 0.2)
        self.assertEqual(qwen_qa_lm.kwargs["max_tokens"], 4096)
        self.assertTrue(lm_supports_gbnf(qwen_qa_lm))

        # Configured default: Gemma RLM without GBNF and Gemma worker with GBNF.
        orch_lm, sub_lm = create_dspy_rlm_pair()
        self.assertEqual(orch_lm.model, "openai/Gemma-4-E4B-it-GGUF")
        self.assertEqual(sub_lm.model, "openai/Gemma-4-E4B-it-GGUF")
        self.assertFalse(lm_supports_gbnf(orch_lm))
        self.assertTrue(lm_supports_gbnf(sub_lm))
        self.assertNotIn("frequency_penalty", sub_lm.kwargs)

        # Gemma4 FLM recipe sets think=False and disables GBNF
        flm_lm = create_dspy_lm("gemma4-it-12b-FLM", role="qa")
        self.assertEqual(flm_lm.kwargs.get("extra_body"), {"think": False})
        self.assertFalse(lm_supports_gbnf(flm_lm))

        # Ollama provider configures ollama_chat/...
        ollama_lm = create_dspy_lm(
            "qwen3:4b",
            base_url="http://localhost:11434",
            provider="ollama",
            role="extraction",
        )
        self.assertEqual(ollama_lm.model, "ollama_chat/qwen3:4b")
        self.assertEqual(ollama_lm.kwargs.get("api_base"), "http://localhost:11434")

    def test_legacy_chat_shims_are_retired(self):
        """Verify legacy HippoRAG chat adapters are completely removed from chat and inference facade."""
        for attr in ("LocalChat", "CacheOpenAI", "chat_identity"):
            self.assertFalse(hasattr(chat_module, attr))
            self.assertFalse(hasattr(inference_facade, attr))


if __name__ == "__main__":
    unittest.main()
