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
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

try:
    from yaw.daemon.ingestion import (
        ALIASES,
        DATASET_DESCRIPTIONS,
        HF_BASE_URL,
        download_dataset_file,
    )
    from yaw.daemon.ingestion import (
        DATASET_REGISTRY as _INGESTION_DATASET_REGISTRY,
    )
    from yaw.daemon.ingestion import (
        UPSTREAM_DATASET_BASE_URL as GITHUB_BASE_URL,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from yaw.daemon.ingestion import (
        ALIASES,
        DATASET_DESCRIPTIONS,
        HF_BASE_URL,
        download_dataset_file,
    )
    from yaw.daemon.ingestion import (
        DATASET_REGISTRY as _INGESTION_DATASET_REGISTRY,
    )
    from yaw.daemon.ingestion import (
        UPSTREAM_DATASET_BASE_URL as GITHUB_BASE_URL,
    )

ENTERPRISE_RAG_BENCH_HF_REPO = "onyx-dot-app/EnterpriseRAG-Bench"
ENTERPRISE_RAG_BENCH_HF_URL = f"https://huggingface.co/datasets/{ENTERPRISE_RAG_BENCH_HF_REPO}/resolve/main"

# Stratified 250-query quotas (150 train, 100 val) across all 10 EnterpriseRAG-Bench categories
ENTERPRISE_CATEGORY_QUOTAS: dict[str, tuple[int, int]] = {
    "Basic": (35, 24),
    "Semantic": (27, 18),
    "Intra-Document Reasoning": (16, 11),
    "Project Related": (16, 11),
    "Constrained": (12, 8),
    "Conflicting Info": (10, 6),
    "Completeness": (10, 6),
    "Miscellaneous": (9, 6),
    "Info Not Found": (9, 6),
    "High Level": (6, 4),
}

_CATEGORY_NORMALIZATION: dict[str, str] = {
    "basic": "Basic",
    "Basic": "Basic",
    "semantic": "Semantic",
    "Semantic": "Semantic",
    "intra_document_reasoning": "Intra-Document Reasoning",
    "Intra-Document Reasoning": "Intra-Document Reasoning",
    "project_related": "Project Related",
    "Project Related": "Project Related",
    "constrained": "Constrained",
    "Constrained": "Constrained",
    "conflicting_info": "Conflicting Info",
    "Conflicting Info": "Conflicting Info",
    "completeness": "Completeness",
    "Completeness": "Completeness",
    "miscellaneous": "Miscellaneous",
    "Miscellaneous": "Miscellaneous",
    "info_not_found": "Info Not Found",
    "Info Not Found": "Info Not Found",
    "high_level": "High Level",
    "High Level": "High Level",
}

# Registry of dataset keys to (query_filename, corpus_filename, description)
DATASET_REGISTRY = {
    name: (*files, DATASET_DESCRIPTIONS.get(name, ""))
    for name, files in _INGESTION_DATASET_REGISTRY.items()
}


def _normalize_category(raw_cat: str) -> str:
    cleaned = str(raw_cat or "").strip()
    if cleaned in _CATEGORY_NORMALIZATION:
        return _CATEGORY_NORMALIZATION[cleaned]
    lower = cleaned.lower().replace("-", "_").replace(" ", "_")
    return _CATEGORY_NORMALIZATION.get(lower, cleaned)


def _normalize_source_types(row: dict[str, Any]) -> list[str]:
    raw = row.get("source_types")
    if raw is None:
        raw = row.get("source_type")
    if isinstance(raw, str):
        return [raw.strip()] if raw.strip() else ["unknown"]
    if isinstance(raw, (list, tuple)):
        out = [str(x).strip() for x in raw if str(x).strip()]
        return out if out else ["unknown"]
    return ["unknown"]


def _fetch_hf_enterprise_rows(
    *,
    distractor_factor: float = 2.0,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fetch questions and required gold + distractor documents from onyx-dot-app/EnterpriseRAG-Bench."""
    helper_code = """
import json, sys, urllib.request
import fsspec, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq

q_url = sys.argv[1]
d_url = sys.argv[2]
out_path = sys.argv[3]
seed = int(sys.argv[4])
distractor_factor = float(sys.argv[5])
quotas = json.loads(sys.argv[6])
cat_map = json.loads(sys.argv[7])

import random, tempfile
with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tf:
    q_tmp = tf.name
urllib.request.urlretrieve(q_url, q_tmp)
q_table = pq.read_table(q_tmp)
q_rows = q_table.to_pylist()

# Determine stratified 250 queries to know which gold doc_ids we need
by_cat = {}
for r in q_rows:
    raw_c = str(r.get("question_type") or "").strip()
    c = cat_map.get(raw_c, raw_c)
    by_cat.setdefault(c, []).append(r)

rng = random.Random(seed)
selected = []
for cat, (n_tr, n_va) in quotas.items():
    pool = sorted(by_cat.get(cat, []), key=lambda x: str(x.get("question_id") or x.get("id") or ""))
    rng.shuffle(pool)
    selected.extend(pool[: n_tr + n_va])

gold_ids = set()
tool_counts = {}
for r in selected:
    eids = list(r.get("expected_doc_ids") or ())
    for did in eids:
        gold_ids.add(str(did))
    st_list = r.get("source_types") or ["unknown"]
    if isinstance(st_list, str):
        st_list = [st_list]
    n_needed = max(1, int(round(max(1, len(eids)) * distractor_factor)))
    per_tool = max(1, int(round(n_needed / max(1, len(st_list)))))
    for st in st_list:
        tool_counts[str(st)] = tool_counts.get(str(st), 0) + per_tool + 4

fs = fsspec.filesystem("https")
with fs.open(d_url) as f:
    meta_table = pq.read_table(f, columns=["doc_id", "source_type"])
meta_by_tool = {}
all_non_gold = []
for did, st in zip(meta_table["doc_id"].to_pylist(), meta_table["source_type"].to_pylist()):
    if did in gold_ids:
        continue
    meta_by_tool.setdefault(str(st), []).append(str(did))

dist_rng = random.Random(seed + 101)
target_ids = set(gold_ids)
for st, cands in sorted(meta_by_tool.items()):
    needed = min(len(cands), tool_counts.get(st, 50) * 2)
    picked = dist_rng.sample(cands, needed)
    target_ids.update(picked)

target_arr = pa.array(list(target_ids))
matched_docs = []
with fs.open(d_url) as f:
    pf = pq.ParquetFile(f)
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=["doc_id", "source_type", "title", "content"])
        mask = pc.is_in(t["doc_id"], value_set=target_arr)
        filtered = t.filter(mask)
        if filtered.num_rows > 0:
            matched_docs.extend(filtered.to_pylist())

with open(out_path, "w", encoding="utf-8") as out_f:
    json.dump({"questions": q_rows, "documents": matched_docs}, out_f)
"""
    q_url = f"{ENTERPRISE_RAG_BENCH_HF_URL}/data/questions/test.parquet"
    d_url = f"{ENTERPRISE_RAG_BENCH_HF_URL}/data/documents/test.parquet"
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_json = Path(tmp_dir) / "hf_extracted.json"
        cmd = [
            "uv",
            "run",
            "--with",
            "pyarrow",
            "--with",
            "fsspec",
            "--with",
            "aiohttp",
            "python",
            "-c",
            helper_code,
            q_url,
            d_url,
            str(out_json),
            str(seed),
            str(distractor_factor),
            json.dumps(ENTERPRISE_CATEGORY_QUOTAS),
            json.dumps(_CATEGORY_NORMALIZATION),
        ]
        print(f"  Fetching stratified EnterpriseRAG-Bench slice from {ENTERPRISE_RAG_BENCH_HF_REPO}...")
        t0 = time.perf_counter()
        subprocess.run(cmd, check=True)
        dt = time.perf_counter() - t0
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        print(
            f"  -> Fetched {len(payload['questions'])} questions and "
            f"{len(payload['documents'])} candidate documents in {dt:.1f}s"
        )
        return payload["questions"], payload["documents"]


def download_enterprise_rag_bench(
    dest_dir: Path,
    *,
    distractor_factor: float = 2.0,
    seed: int = 42,
    force: bool = False,
    questions_rows: list[dict[str, Any]] | None = None,
    documents_rows: list[dict[str, Any]] | None = None,
) -> tuple[Path, Path]:
    "Build stratified 250-query EnterpriseRAG-Bench dataset (150 train + 100 val) and 2.0x same-tool distractor corpus."
    q_file, c_file, _ = DATASET_REGISTRY["enterprise_rag_bench"]
    q_target = dest_dir / q_file
    c_target = dest_dir / c_file

    if (
        questions_rows is None
        and documents_rows is None
        and not force
        and q_target.exists()
        and c_target.exists()
        and q_target.stat().st_size > 0
        and c_target.stat().st_size > 0
    ):
        q_mb = q_target.stat().st_size / (1024 * 1024)
        c_mb = c_target.stat().st_size / (1024 * 1024)
        print(f"  [cached] {q_file} ({q_mb:.2f} MB), {c_file} ({c_mb:.2f} MB)")
        return q_target, c_target

    if questions_rows is None or documents_rows is None:
        fetched_q, fetched_d = _fetch_hf_enterprise_rows(
            distractor_factor=distractor_factor,
            seed=seed,
        )
        if questions_rows is None:
            questions_rows = fetched_q
        if documents_rows is None:
            documents_rows = fetched_d

    by_category: dict[str, list[dict[str, Any]]] = {}
    for row in questions_rows:
        cat = _normalize_category(str(row.get("question_type") or ""))
        by_category.setdefault(cat, []).append(row)

    rng = random.Random(seed)
    train_raw: list[tuple[str, dict[str, Any]]] = []
    val_raw: list[tuple[str, dict[str, Any]]] = []

    for cat, (n_train, n_val) in ENTERPRISE_CATEGORY_QUOTAS.items():
        pool = sorted(
            by_category.get(cat, []),
            key=lambda r: str(r.get("question_id") or r.get("id") or r.get("question") or ""),
        )
        if len(pool) < n_train + n_val:
            raise ValueError(f"Category '{cat}' has only {len(pool)} questions, need {n_train + n_val}")
        rng.shuffle(pool)
        for item in pool[:n_train]:
            train_raw.append((cat, item))
        for item in pool[n_train : n_train + n_val]:
            val_raw.append((cat, item))

    rng.shuffle(train_raw)
    rng.shuffle(val_raw)
    ordered_split = [("train", cat, item) for cat, item in train_raw] + [("val", cat, item) for cat, item in val_raw]

    docs_by_id: dict[str, dict[str, Any]] = {}
    non_gold_by_tool: dict[str, list[dict[str, Any]]] = {}
    all_gold_ids: set[str] = {str(did) for _, _, q in ordered_split for did in (q.get("expected_doc_ids") or ())}

    for d in documents_rows:
        did = str(d.get("doc_id") or d.get("id") or "")
        if not did:
            continue
        docs_by_id[did] = d
        if did not in all_gold_ids:
            st = str(d.get("source_type") or "unknown").strip() or "unknown"
            non_gold_by_tool.setdefault(st, []).append(d)

    for st in non_gold_by_tool:
        non_gold_by_tool[st].sort(key=lambda x: str(x.get("doc_id") or x.get("id") or ""))
        rng.shuffle(non_gold_by_tool[st])

    tool_cursors: dict[str, int] = {st: 0 for st in non_gold_by_tool}
    all_non_gold_docs = [d for d in documents_rows if str(d.get("doc_id") or d.get("id") or "") not in all_gold_ids]
    all_non_gold_docs.sort(key=lambda x: str(x.get("doc_id") or x.get("id") or ""))
    rng.shuffle(all_non_gold_docs)
    fallback_cursor = 0

    def _format_doc_entry(doc_obj: dict[str, Any]) -> tuple[str, str, str, str]:
        did = str(doc_obj.get("doc_id") or doc_obj.get("id") or "")
        st = str(doc_obj.get("source_type") or "unknown").strip() or "unknown"
        raw_title = str(doc_obj.get("title") or did).strip() or did
        prefix = f"[{st}] "
        title = raw_title if raw_title.startswith("[") else f"{prefix}{raw_title}"
        text = str(doc_obj.get("content") if "content" in doc_obj else doc_obj.get("text") or "").strip()
        return did, st, title, text

    formatted_queries: list[dict[str, Any]] = []
    corpus_map: dict[str, dict[str, Any]] = {}

    for split_name, canonical_cat, q in ordered_split:
        qid = str(q.get("question_id") or q.get("id") or "")
        question_text = str(q.get("question") or "").strip()
        gold_ans = str(q.get("gold_answer") if "gold_answer" in q else q.get("answer") or "").strip()
        raw_facts = q.get("answer_facts") or q.get("answer_aliases") or []
        aliases = [str(f).strip() for f in raw_facts if str(f).strip()]
        if canonical_cat == "Info Not Found":
            for abstain in (
                "Insufficient references in corpus to answer properly, consider ingesting more supporting documents.",
                "Insufficient information",
                "Information not found",
                "Not mentioned",
                "Cannot be determined from the provided context",
            ):
                if abstain not in aliases:
                    aliases.append(abstain)

        source_types = _normalize_source_types(q)
        expected_ids = [str(x) for x in (q.get("expected_doc_ids") or ()) if str(x)]

        paragraphs: list[dict[str, Any]] = []
        for did in expected_ids:
            doc_obj = docs_by_id.get(did)
            if doc_obj is None:
                continue
            doc_id, st, title, text = _format_doc_entry(doc_obj)
            paragraphs.append(
                {
                    "idx": doc_id,
                    "title": title,
                    "text": text,
                    "paragraph_text": text,
                    "source_type": st,
                    "is_supporting": True,
                }
            )
            corpus_map.setdefault(
                doc_id,
                {"idx": doc_id, "title": title, "text": text, "source_type": st},
            )

        n_gold_effective = max(1, len(expected_ids))
        n_distractors = max(1, int(round(n_gold_effective * distractor_factor)))
        picked_distractors: list[dict[str, Any]] = []
        picked_ids: set[str] = set(expected_ids)

        for i in range(n_distractors):
            target_st = source_types[i % len(source_types)]
            pool = non_gold_by_tool.get(target_st, [])
            chosen: dict[str, Any] | None = None
            if pool:
                start_cur = tool_cursors.get(target_st, 0)
                for offset in range(len(pool)):
                    cand = pool[(start_cur + offset) % len(pool)]
                    cid = str(cand.get("doc_id") or cand.get("id") or "")
                    if cid not in picked_ids:
                        chosen = cand
                        tool_cursors[target_st] = (start_cur + offset + 1) % len(pool)
                        break
            if chosen is None and all_non_gold_docs:
                for offset in range(len(all_non_gold_docs)):
                    cand = all_non_gold_docs[(fallback_cursor + offset) % len(all_non_gold_docs)]
                    cid = str(cand.get("doc_id") or cand.get("id") or "")
                    if cid not in picked_ids:
                        chosen = cand
                        fallback_cursor = (fallback_cursor + offset + 1) % len(all_non_gold_docs)
                        break
            if chosen is not None:
                cid = str(chosen.get("doc_id") or chosen.get("id") or "")
                picked_ids.add(cid)
                picked_distractors.append(chosen)

        for d_obj in picked_distractors:
            doc_id, st, title, text = _format_doc_entry(d_obj)
            paragraphs.append(
                {
                    "idx": doc_id,
                    "title": title,
                    "text": text,
                    "paragraph_text": text,
                    "source_type": st,
                    "is_supporting": False,
                }
            )
            corpus_map.setdefault(
                doc_id,
                {"idx": doc_id, "title": title, "text": text, "source_type": st},
            )

        formatted_queries.append(
            {
                "id": qid,
                "question": question_text,
                "answer": gold_ans,
                "answer_aliases": aliases,
                "question_type": canonical_cat,
                "source_types": source_types,
                "split": split_name,
                "paragraphs": paragraphs,
            }
        )

    dest_dir.mkdir(parents=True, exist_ok=True)
    q_target.write_text(
        json.dumps(formatted_queries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    c_target.write_text(
        json.dumps(list(corpus_map.values()), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"  -> Wrote {len(formatted_queries)} stratified queries to {q_file} and {len(corpus_map)} passages to {c_file}"
    )
    return q_target, c_target


def download_file(filename: str, dest_dir: Path, force: bool = False) -> Path:
    """Download a dataset file from Hugging Face (or GitHub fallback) to dest_dir."""
    return download_dataset_file(
        filename,
        dest_dir,
        force=force,
        hf_base_url=HF_BASE_URL,
        upstream_base_url=GITHUB_BASE_URL,
        verbose=True,
    )


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
                print(
                    f"  [OK] {name:22s}: {st['queries']:5d} queries, "
                    f"{st['corpus_passages']:6d} passages "
                    f"({st['query_size_mb'] + st['corpus_size_mb']:.1f} MB)"
                )
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
        if key == "enterprise_rag_bench":
            download_enterprise_rag_bench(dest_dir, force=args.force)
        else:
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
