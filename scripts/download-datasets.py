"""Download script for HippoRAG 2 benchmark datasets from official upstream sources.

Sources:
  Primary: Hugging Face (https://huggingface.co/datasets/osunlp/HippoRAG_2)
  Fallback: GitHub pinned commit (https://github.com/OSU-NLP-Group/HippoRAG/tree/d5c8329422e0a0b834a15874545cb6a74b4f9b26/reproduce/dataset)

Usage:
  uv run --locked python scripts/download-datasets.py --dataset all
  uv run --locked python scripts/download-datasets.py --dataset nq
  uv run --locked python scripts/download-datasets.py --dataset popqa
  uv run --locked python scripts/download-datasets.py --dataset lveval
  uv run --locked python scripts/download-datasets.py --dataset narrativeqa
  uv run --locked python scripts/download-datasets.py --check
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HF_BASE_URL = "https://huggingface.co/datasets/osunlp/HippoRAG_2/resolve/main"
GITHUB_PINNED_COMMIT = "d5c8329422e0a0b834a15874545cb6a74b4f9b26"
GITHUB_BASE_URL = (
    f"https://raw.githubusercontent.com/OSU-NLP-Group/HippoRAG/{GITHUB_PINNED_COMMIT}/reproduce/dataset"
)

# Registry of dataset keys to (query_filename, corpus_filename, description)
DATASET_REGISTRY = {
    "sample": ("sample.json", "sample_corpus.json", "Tier 0 validation fixture (1 query, 3 passages)"),
    "2wikimultihopqa": (
        "2wikimultihopqa.json",
        "2wikimultihopqa_corpus.json",
        "Multi-hop QA from Wikipedia (1,000 queries, 6,119 passages)",
    ),
    "hotpotqa": (
        "hotpotqa.json",
        "hotpotqa_corpus.json",
        "Multi-hop QA with diverse reasoning (1,000 queries, 9,811 passages)",
    ),
    "musique": (
        "musique.json",
        "musique_corpus.json",
        "Complex multi-hop QA with 2-4 hops (1,000 queries, 11,656 passages)",
    ),
    "nq": (
        "nq_rear.json",
        "nq_rear_corpus.json",
        "NaturalQuestions single-hop QA (1,000 queries, 9,633 passages)",
    ),
    "popqa": (
        "popqa.json",
        "popqa_corpus.json",
        "PopQA entity-centric long-tail QA (1,000 queries, 8,676 passages)",
    ),
    "lveval": (
        "lveval.json",
        "lveval_corpus.json",
        "LV-Eval hotpotwikiqa-mixup 256k long-context QA (124 queries, 22,849 passages)",
    ),
    "narrativeqa": (
        "narrativeqa_dev_10_doc.json",
        "narrativeqa_dev_10_doc_corpus.json",
        "NarrativeQA discourse understanding on novels (293 queries, 4,111 passages)",
    ),
    "case_study_university": (
        "case_study_university.json",
        "case_study_university_corpus.json",
        "University Case Study graph reasoning demo",
    ),
}

# Aliases for convenience
ALIASES = {
    "2wiki": "2wikimultihopqa",
    "hotpot": "hotpotqa",
    "naturalquestions": "nq",
    "nq_rear": "nq",
    "narrative": "narrativeqa",
}


def download_file(filename: str, dest_dir: Path, force: bool = False) -> Path:
    """Download a dataset file from Hugging Face (or GitHub fallback) to dest_dir."""
    target = dest_dir / filename
    if target.exists() and not force and target.stat().st_size > 0:
        size_mb = target.stat().st_size / (1024 * 1024)
        print(f"  [cached] {filename} ({size_mb:.2f} MB)")
        return target

    urls = [f"{HF_BASE_URL}/{filename}"]
    # GitHub hosts sample, 2wiki, hotpot, and musique files
    if any(k in filename for k in ("sample", "2wiki", "hotpot", "musique")):
        urls.append(f"{GITHUB_BASE_URL}/{filename}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp_target = dest_dir / f"{filename}.tmp"

    last_error = None
    for url in urls:
        try:
            print(f"  Downloading {filename} from {url}...")
            t0 = time.perf_counter()
            req = urllib.request.Request(url, headers={"User-Agent": "YAW-HippoRAG-DatasetDownloader/1.0"})
            with urllib.request.urlopen(req) as resp, open(tmp_target, "wb") as out_f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    out_f.write(chunk)
            tmp_target.replace(target)
            dt = time.perf_counter() - t0
            size_mb = target.stat().st_size / (1024 * 1024)
            print(f"  -> Saved {filename} ({size_mb:.2f} MB in {dt:.2f}s)")
            return target
        except urllib.error.URLError as exc:
            last_error = exc
            if tmp_target.exists():
                tmp_target.unlink()
            print(f"  Failed from {url}: {exc}")

    raise RuntimeError(f"Could not download {filename} from any source: {last_error}")


def verify_dataset(name: str, dest_dir: Path) -> dict:
    """Check integrity of a downloaded dataset."""
    q_file, c_file, desc = DATASET_REGISTRY[name]
    q_path = dest_dir / q_file
    c_path = dest_dir / c_file

    status = {"name": name, "description": desc, "ready": False}
    if not q_path.exists() or not c_path.exists():
        status["error"] = "Files missing"
        return status

    try:
        queries = json.loads(q_path.read_text(encoding="utf-8"))
        corpus = json.loads(c_path.read_text(encoding="utf-8"))
        status["ready"] = True
        status["queries"] = len(queries)
        status["corpus_passages"] = len(corpus)
        status["query_size_mb"] = round(q_path.stat().st_size / (1024 * 1024), 2)
        status["corpus_size_mb"] = round(c_path.stat().st_size / (1024 * 1024), 2)
    except Exception as exc:
        status["error"] = str(exc)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset",
        choices=["all", *tuple(DATASET_REGISTRY), *tuple(ALIASES)],
        default="all",
        help="Dataset name to download or 'all' to fetch all datasets.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("outputs/datasets"),
        help="Destination directory for dataset JSON files (default: outputs/datasets).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-downloading files even if they already exist on disk.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the integrity and print statistics of downloaded datasets without downloading.",
    )
    args = parser.parse_args()

    dest_dir = args.dest.resolve()

    if args.check:
        print(f"\nVerifying datasets in {dest_dir}:")
        all_ok = True
        for name in DATASET_REGISTRY:
            st = verify_dataset(name, dest_dir)
            if st["ready"]:
                print(f"  [OK] {name:22s}: {st['queries']:5d} queries, {st['corpus_passages']:6d} passages ({st['query_size_mb'] + st['corpus_size_mb']:.1f} MB)")
            else:
                print(f"  [MISSING] {name:22s}: {st.get('error', 'not found')}")
                all_ok = False
        sys.exit(0 if all_ok else 1)

    selected = [args.dataset] if args.dataset != "all" else list(DATASET_REGISTRY)
    canonical_keys = [ALIASES.get(k, k) for k in selected]

    print(f"\n=== Downloading HippoRAG 2 datasets to {dest_dir} ===")
    for key in canonical_keys:
        q_file, c_file, desc = DATASET_REGISTRY[key]
        print(f"\nDataset: {key} ({desc})")
        download_file(q_file, dest_dir, force=args.force)
        download_file(c_file, dest_dir, force=args.force)

    print("\n=== Verification ===")
    for key in canonical_keys:
        st = verify_dataset(key, dest_dir)
        if st["ready"]:
            print(f"  [OK] {key}: {st['queries']} queries, {st['corpus_passages']} corpus passages")
        else:
            print(f"  [FAIL] {key}: {st.get('error')}")


if __name__ == "__main__":
    main()
