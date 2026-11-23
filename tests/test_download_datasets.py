import json
import tempfile
import unittest
from pathlib import Path

import scripts.download_datasets as download_datasets
from yaw.datasets import load_dataset_bundle, prepare_dataset_bundle, resolve_dataset_name


class DownloadDatasetsTests(unittest.TestCase):
    def test_registry_contains_all_paper_benchmarks(self):
        expected_benchmarks = {
            "sample",
            "2wikimultihopqa",
            "hotpotqa",
            "musique",
            "nq",
            "popqa",
            "lveval",
            "narrativeqa",
            "case_study_university",
            "enterprise_rag_bench",
        }
        self.assertEqual(set(download_datasets.DATASET_REGISTRY.keys()), expected_benchmarks)

    def test_aliases_are_rejected_and_not_exported(self):
        self.assertFalse(hasattr(download_datasets, "ALIASES"))
        for alias in ("2wiki", "naturalquestions", "nq_rear"):
            with self.assertRaises(ValueError):
                resolve_dataset_name(alias)

    def test_bundle_verification_detects_missing_and_valid_files(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            with self.assertRaises(FileNotFoundError):
                load_dataset_bundle(root / "sample")
            corpus, questions = root / "corpus.json", root / "questions.json"
            corpus.write_text(json.dumps([{"idx": "doc1", "title": "A", "text": "Body"}]), encoding="utf-8")
            questions.write_text(json.dumps([{"question": "Q", "answer": "A"}]), encoding="utf-8")
            bundle = prepare_dataset_bundle("sample", corpus, questions, root)
            self.assertEqual(load_dataset_bundle(bundle.root), bundle)
            self.assertEqual(bundle.documents[0].path, "documents/doc1.md")
            self.assertEqual((bundle.root / bundle.documents[0].path).read_text(encoding="utf-8"), "# A\n\nBody\n")
            self.assertEqual(json.loads(bundle.manifest_path.read_text(encoding="utf-8"))["version"], 2)

    def test_delegates_fallback_lookup_to_yaw_datasets(self):
        self.assertFalse(hasattr(download_datasets, "_CATALOG_PATH"))
        self.assertFalse(hasattr(download_datasets, "_DATASET_SPECS"))
        self.assertFalse(hasattr(download_datasets, "_dataset_spec"))
        self.assertTrue(download_datasets.allow_upstream_github_fallback("sample.json"))
        self.assertFalse(download_datasets.allow_upstream_github_fallback("nq_rear.json"))

    def test_download_file_reuses_existing_cache_without_network(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dest = Path(tmp_dir)
            target = dest / "test.json"
            target.write_text("{}", encoding="utf-8")

            # With force=False, returns target without network
            result = download_datasets.download_dataset_file("test.json", dest, force=False)
            self.assertEqual(result, target)


if __name__ == "__main__":
    unittest.main()
