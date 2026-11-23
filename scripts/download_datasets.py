#!/usr/bin/env python
"""Download configured inputs and copy document records into native .md document files."""

from __future__ import annotations

import argparse
import time
import urllib.error
import urllib.request
from pathlib import Path

from yaw.datasets import (
    DATASET_REGISTRY,
    DATASET_SOURCE_FIELDS,
    HF_BASE_URL,
    UPSTREAM_DATASET_BASE_URL,
    allow_upstream_github_fallback,
    load_dataset_bundle,
    prepare_dataset_bundle,
    resolve_dataset_name,
)

__all__ = [
    "DATASET_REGISTRY",
    "DATASET_SOURCE_FIELDS",
    "HF_BASE_URL",
    "UPSTREAM_DATASET_BASE_URL",
    "allow_upstream_github_fallback",
    "download_dataset_file",
    "load_dataset_bundle",
    "main",
    "prepare_dataset_bundle",
    "resolve_dataset_name",
]


def download_dataset_file(
    filename: str,
    dest_dir: Path,
    *,
    force: bool = False,
    hf_base_url: str | None = None,
    upstream_base_url: str | None = None,
    verbose: bool = False,
) -> Path:
    """Download a dataset file from Hugging Face (or GitHub fallback) to `dest_dir`."""
    if (
        not filename
        or "/" in filename
        or "\\" in filename
        or Path(filename).is_absolute()
        or Path(filename).name != filename
        or filename in {".", ".."}
    ):
        raise ValueError(f"Dataset filename must be a single file name: {filename}")
    target = dest_dir / filename
    if target.exists() and not force and target.stat().st_size > 0:
        if verbose:
            size_mb = target.stat().st_size / (1024 * 1024)
            print(f"  [cached] {filename} ({size_mb:.2f} MB)")
        return target

    urls = [f"{hf_base_url or HF_BASE_URL}/{filename}"]
    if allow_upstream_github_fallback(filename):
        urls.append(f"{upstream_base_url or UPSTREAM_DATASET_BASE_URL}/{filename}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp_target = dest_dir / f"{filename}.tmp"

    last_error: Exception | None = None
    for url in urls:
        try:
            if verbose:
                print(f"  Downloading {filename} from {url}...")
            t0 = time.perf_counter()
            req = urllib.request.Request(url, headers={"User-Agent": "YAW-DatasetDownloader/1.0"})
            with urllib.request.urlopen(req) as resp, tmp_target.open("wb") as out_f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    out_f.write(chunk)
            tmp_target.replace(target)
            if verbose:
                dt = time.perf_counter() - t0
                size_mb = target.stat().st_size / (1024 * 1024)
                print(f"  -> Saved {filename} ({size_mb:.2f} MB in {dt:.2f}s)")
            return target
        except (urllib.error.URLError, OSError, RuntimeError) as exc:
            last_error = exc
            if tmp_target.exists():
                tmp_target.unlink(missing_ok=True)
            if verbose:
                print(f"  Failed from {url}: {exc}")

    raise RuntimeError(f"Could not download {filename} from any source: {last_error}") from last_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset", choices=["all", *DATASET_REGISTRY], default="all")
    parser.add_argument("--dest", type=Path, default=Path("outputs/datasets"))
    parser.add_argument("--source-dir", type=Path, default=Path("outputs/datasets/sources"))
    parser.add_argument("--corpus", type=Path, help="Configured corpus JSON for a single dataset")
    parser.add_argument("--questions", type=Path, help="Configured evaluation JSON for a single dataset")
    parser.add_argument("--force", action="store_true", help="Redownload sources; unchanged documents stay untouched")
    parser.add_argument("--check", action="store_true", help="Validate prepared bundles without changing files")
    args = parser.parse_args()
    if args.dataset == "all" and (args.corpus or args.questions):
        parser.error("--corpus/--questions require a single --dataset")
    selected = list(DATASET_REGISTRY) if args.dataset == "all" else [args.dataset]
    failed = False
    for name in selected:
        if args.check:
            try:
                bundle = load_dataset_bundle(args.dest / name)
            except (OSError, ValueError, KeyError) as exc:
                print(f"[FAIL] {name}: {exc}")
                failed = True
                continue
        else:
            q_name, c_name = DATASET_REGISTRY[name]
            corpus = args.corpus or args.source_dir / c_name
            questions = args.questions or args.source_dir / q_name
            if name == "enterprise_rag_bench" and (not corpus.is_file() or not questions.is_file()):
                parser.error(
                    "Enterprise requires configured corpus and question JSON inputs; "
                    "use --source-dir or --corpus/--questions"
                )
            if not args.corpus and name != "enterprise_rag_bench":
                corpus = download_dataset_file(c_name, args.source_dir, force=args.force)
            if not args.questions and name != "enterprise_rag_bench":
                questions = download_dataset_file(q_name, args.source_dir, force=args.force)
            bundle = prepare_dataset_bundle(name, corpus, questions, args.dest, **DATASET_SOURCE_FIELDS[name])
        print(f"[OK] {name}: {len(bundle.documents)} documents, {bundle.evaluation_path}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
