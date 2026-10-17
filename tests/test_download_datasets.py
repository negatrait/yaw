import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "download-datasets.py"

spec = importlib.util.spec_from_file_location("download_datasets", SCRIPT_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load module from {SCRIPT_PATH}")
download_datasets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download_datasets)


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
        }
        self.assertEqual(set(download_datasets.DATASET_REGISTRY.keys()), expected_benchmarks)

    def test_aliases_map_to_canonical_keys(self):
        for _alias, canonical in download_datasets.ALIASES.items():
            self.assertIn(canonical, download_datasets.DATASET_REGISTRY)

    def test_verify_dataset_detects_missing_and_valid_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dest = Path(tmp_dir)
            # 1. Missing files
            missing_status = download_datasets.verify_dataset("sample", dest)
            self.assertFalse(missing_status["ready"])
            self.assertEqual(missing_status["error"], "Files missing")

            # 2. Present and valid JSON
            (dest / "sample.json").write_text(json.dumps([{"q": 1}]), encoding="utf-8")
            (dest / "sample_corpus.json").write_text(json.dumps([{"c": 1}, {"c": 2}]), encoding="utf-8")
            valid_status = download_datasets.verify_dataset("sample", dest)
            self.assertTrue(valid_status["ready"])
            self.assertEqual(valid_status["queries"], 1)
            self.assertEqual(valid_status["corpus_passages"], 2)

    def test_download_file_reuses_existing_cache_without_network(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dest = Path(tmp_dir)
            target = dest / "test.json"
            target.write_text("{}", encoding="utf-8")

            # With force=False, returns target without network
            result = download_datasets.download_file("test.json", dest, force=False)
            self.assertEqual(result, target)


if __name__ == "__main__":
    unittest.main()
