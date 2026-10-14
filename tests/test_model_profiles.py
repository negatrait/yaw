"""Model-profile and serving identity contracts, without local model calls."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from hipporag.utils.config_utils import BaseConfig
from yaw.embedding import GEMMA_TASK_MODES, MODEL, PROFILES, RetrievalEmbedding, resolve_embedding_profile
from hipporag.prompts.linking import get_query_instruction
from yaw.models import CHAT_MODELS, chat_evidence, check_registry
from unittest.mock import patch


class ProfileTests(unittest.TestCase):
    def test_llamacpp_chat_evidence_requires_cache_ram_zero_and_no_dio(self):
        name = "Qwen3-4B-Instruct-2507-GGUF"
        ckpt, recipe = CHAT_MODELS[name]
        base_entry = {
            "id": name,
            "checkpoint": ckpt,
            "recipe": recipe,
            "backend_health": "ready",
            "launch_command": ["llama-server.exe", "--parallel", "1", "--cache-ram", "0"],
        }
        with patch("yaw.models.get_json", return_value={"all_models_loaded": [base_entry]}):
            self.assertEqual(chat_evidence(name)["id"], name)
        with patch(
            "yaw.models.get_json",
            return_value={
                "all_models_loaded": [
                    {**base_entry, "launch_command": ["llama-server.exe", "-dio", "--cache-ram", "0"]}
                ]
            },
        ):
            with self.assertRaisesRegex(ValueError, "-dio"):
                chat_evidence(name)
        with patch(
            "yaw.models.get_json",
            return_value={"all_models_loaded": [{**base_entry, "launch_command": ["llama-server.exe", "--parallel", "1"]}]},
        ):
            with self.assertRaisesRegex(ValueError, "--cache-ram 0"):
                chat_evidence(name)
    def test_each_profile_preserves_its_request_format_and_dimensions(self):
        for profile in PROFILES.values():
            with self.subTest(model=profile.model):
                model = RetrievalEmbedding(BaseConfig(
                    embedding_model_name=profile.model,
                    embedding_base_url="http://localhost:13305/api/v1",
                ), profile)
                model.client.close()
                model.client = Mock()
                model.client.embeddings.create.return_value = SimpleNamespace(
                    usage=SimpleNamespace(prompt_tokens=1, total_tokens=1),
                    data=[SimpleNamespace(index=0, embedding=[1.] * profile.dimensions)],
                )
                try:
                    model.batch_encode(["a document"])
                    self.assertEqual(model.client.embeddings.create.call_args.kwargs["input"],
                                     [profile.document_prefix + "a document"])
                    result = model.batch_encode("who?", instruction=get_query_instruction("query_to_passage"))
                    request = model.client.embeddings.create.call_args.kwargs
                    self.assertEqual(request["model"], profile.model)
                    self.assertEqual(request["input"], [profile.query_prefix + "who?"])
                    self.assertEqual(result.shape, (1, profile.dimensions))
                    if profile.model.startswith("Qwen"):
                        self.assertIn("\nQuery:", request["input"][0])
                        model.client.embeddings.create.return_value.data[0].embedding = [1.] * 768
                        with self.assertRaises(ValueError):
                            model.batch_encode(["wrong dimensions"])
                finally:
                    model.close()

    def test_identity_changes_for_quant_prompt_pooling_dimensions_and_task_modes(self):
        self.assertEqual(len({p.index_identity for p in PROFILES.values()}), 3)
        profile = next(iter(PROFILES.values()))
        for changed in (
            replace(profile, query_prefix="new query: "),
            replace(profile, dimensions=256),
            replace(profile, pooling="last"),
        ):
            self.assertNotEqual(profile.index_identity, changed.index_identity)
        mode_identities = {
            resolve_embedding_profile(MODEL, mode).index_identity for mode in GEMMA_TASK_MODES
        }
        self.assertEqual(len(mode_identities), len(GEMMA_TASK_MODES))
        with self.assertRaises(ValueError):
            resolve_embedding_profile("Qwen3-Embedding-0.6B-GGUF", "qa")

    def test_config_cannot_claim_another_embedding_model(self):
        profile = next(iter(PROFILES.values()))
        with self.assertRaises(ValueError):
            RetrievalEmbedding(BaseConfig(embedding_model_name="another-model"), profile)

    def test_no_download_or_substitution_for_missing_mistral(self):
        profile = next(iter(PROFILES.values()))
        name = "Mistral-7B-Instruct-v0.3-Hybrid"
        checkpoint, recipe = CHAT_MODELS[name]
        registry = [
            {"id": profile.model, "checkpoint": profile.checkpoint, "recipe": "llamacpp",
             "labels": ["embeddings"], "downloaded": True},
            {"id": name, "checkpoint": checkpoint, "recipe": recipe,
             "labels": ["chat"], "downloaded": False},
        ]
        with self.assertRaisesRegex(ValueError, "not downloaded"):
            check_registry(registry, profile, name)

    def test_split_extraction_and_qa_models_are_both_validated(self):
        profile = next(iter(PROFILES.values()))
        ext_name = "Phi-4-mini-instruct-GGUF"
        qa_name = "Mistral-7B-Instruct-v0.3-Hybrid"
        ext_ckpt, ext_recipe = CHAT_MODELS[ext_name]
        qa_ckpt, qa_recipe = CHAT_MODELS[qa_name]
        registry = [
            {"id": profile.model, "checkpoint": profile.checkpoint, "recipe": "llamacpp",
             "labels": ["embeddings"], "downloaded": True},
            {"id": ext_name, "checkpoint": ext_ckpt, "recipe": ext_recipe,
             "labels": ["chat"], "downloaded": True},
            {"id": qa_name, "checkpoint": qa_ckpt, "recipe": qa_recipe,
             "labels": ["chat"], "downloaded": True},
        ]
        checked = check_registry(registry, profile, ext_name, qa_name)
        self.assertEqual(checked["chat"]["id"], ext_name)
        self.assertEqual(checked["qa_chat"]["id"], qa_name)

    def test_wrong_role_or_checkpoint_is_rejected(self):
        profile = next(iter(PROFILES.values()))
        model = {"id": profile.model, "checkpoint": profile.checkpoint, "recipe": "llamacpp",
                 "labels": ["embeddings"], "downloaded": True}
        for change in ({"labels": ["chat"]}, {"checkpoint": "different/weights"}):
            with self.assertRaises(ValueError):
                check_registry([{**model, **change}], profile)

    def test_embeddings_only_does_not_require_chat(self):
        profile = next(iter(PROFILES.values()))
        model = {"id": profile.model, "checkpoint": profile.checkpoint, "recipe": "llamacpp",
                 "labels": ["embeddings"], "downloaded": True}
        self.assertEqual(check_registry([model], profile), {"embeddings": model})


if __name__ == "__main__":
    unittest.main()


