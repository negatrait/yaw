#!/usr/bin/env python
"""Download configured inputs and copy document records into isolated JSON files."""

import argparse
from pathlib import Path

from yaw.daemon.ingestion import (
    ALIASES,
    DATASET_REGISTRY,
    DATASET_SOURCE_FIELDS,
    download_dataset_file,
    load_dataset_bundle,
    prepare_dataset_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset", choices=["all", *DATASET_REGISTRY, *ALIASES], default="all")
    parser.add_argument("--dest", type=Path, default=Path("outputs/datasets"))
    parser.add_argument("--source-dir", type=Path, default=Path("outputs/datasets/sources"))
    parser.add_argument("--corpus", type=Path, help="Configured corpus JSON for a single dataset")
    parser.add_argument("--questions", type=Path, help="Configured evaluation JSON for a single dataset")
    parser.add_argument("--force", action="store_true", help="Redownload sources; unchanged documents stay untouched")
    parser.add_argument("--check", action="store_true", help="Validate prepared bundles without changing files")
    args = parser.parse_args()
    if args.dataset == "all" and (args.corpus or args.questions):
        parser.error("--corpus/--questions require a single --dataset")
    selected = list(DATASET_REGISTRY) if args.dataset == "all" else [ALIASES.get(args.dataset, args.dataset)]
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
