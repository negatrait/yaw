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
            "enterprise_rag_bench",
        }
        self.assertEqual(set(download_datasets.DATASET_REGISTRY.keys()), expected_benchmarks)

    def test_aliases_map_to_canonical_keys(self):
        for _alias, canonical in download_datasets.ALIASES.items():
            self.assertIn(canonical, download_datasets.DATASET_REGISTRY)

    def test_bundle_verification_detects_missing_and_valid_files(self):
        from yaw.daemon.ingestion import load_dataset_bundle, prepare_dataset_bundle

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            with self.assertRaises(FileNotFoundError):
                load_dataset_bundle(root / "sample")
            corpus, questions = root / "corpus.json", root / "questions.json"
            corpus.write_text(json.dumps([{"title": "A", "text": "Body"}]), encoding="utf-8")
            questions.write_text(json.dumps([{"question": "Q", "answer": "A"}]), encoding="utf-8")
            bundle = prepare_dataset_bundle("sample", corpus, questions, root)
            self.assertEqual(load_dataset_bundle(bundle.root), bundle)

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
