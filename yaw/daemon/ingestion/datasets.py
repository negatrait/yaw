"""Centralized dataset registry, downloader, slice builder, and CocoIndex direct-source ingestion helpers."""

import json
import random
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from yaw.core import YawRuntimeError

from .bundles import DatasetBundle, document_identity, document_text, load_dataset_bundle

_CATALOG_PATH = Path(__file__).resolve().parents[3] / "data" / "benchmarks" / "baselines.json"
_CATALOG: dict[str, Any] = json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))

PINNED_COMMIT: str = str(_CATALOG["pinned_commit"])
HF_BASE_URL: str = str(_CATALOG["hf_base_url"])
UPSTREAM_DATASET_BASE_URL: str = str(_CATALOG["upstream_dataset_base_url"])

_DATASET_SPECS: dict[str, dict[str, Any]] = dict(_CATALOG["datasets"])
DATASETS: tuple[str, ...] = tuple(_DATASET_SPECS.keys())
DEFAULT_PARITY_DATASETS: tuple[str, ...] = tuple(k for k, v in _DATASET_SPECS.items() if v.get("default_parity"))
TIER2_CORPORA: frozenset[str] = frozenset(k for k, v in _DATASET_SPECS.items() if v.get("tier2"))
QA_ONLY_DATASETS: frozenset[str] = frozenset(k for k, v in _DATASET_SPECS.items() if v.get("qa_only"))
DATASET_REGISTRY: dict[str, tuple[str, str]] = {
    k: (str(v["questions_file"]), str(v["corpus_file"])) for k, v in _DATASET_SPECS.items()
}
DATASET_SOURCE_FIELDS: dict[str, dict[str, Any]] = {k: dict(v["source_fields"]) for k, v in _DATASET_SPECS.items()}
ENTERPRISE_CATEGORY_QUOTAS: dict[str, tuple[int, int]] = next(
    (
        {str(k): (int(q[0]), int(q[1])) for k, q in v["category_quotas"].items()}
        for v in _DATASET_SPECS.values()
        if "category_quotas" in v
    ),
    {},
)


def _dataset_spec(name: str | Path | None) -> dict[str, Any]:
    if name is None:
        return {}
    raw = str(name).strip()
    if not raw:
        return {}
    lowered = raw.lower()
    if lowered in _DATASET_SPECS:
        return _DATASET_SPECS[lowered]
    stem = Path(raw).stem.strip().lower()
    if stem in _DATASET_SPECS:
        return _DATASET_SPECS[stem]
    file_name = Path(raw).name.strip().lower()
    for k, spec in _DATASET_SPECS.items():
        if file_name in (str(spec.get("questions_file", "")).lower(), str(spec.get("corpus_file", "")).lower()):
            return spec
        if stem.startswith(k):
            return spec
    return {}


def resolve_dataset_name(name: str) -> str:
    """Validate a canonical dataset key against DATASET_REGISTRY."""
    cleaned = name.strip().lower()
    if cleaned not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Valid datasets: {', '.join(DATASETS)}")
    return cleaned


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
    if Path(filename).is_absolute() or Path(filename).name != filename or filename in {".", ".."}:
        raise ValueError(f"Dataset filename must be a single file name: {filename}")
    target = dest_dir / filename
    if target.exists() and not force and target.stat().st_size > 0:
        if verbose:
            size_mb = target.stat().st_size / (1024 * 1024)
            print(f"  [cached] {filename} ({size_mb:.2f} MB)")
        return target

    urls = [f"{hf_base_url or HF_BASE_URL}/{filename}"]
    if bool(_dataset_spec(filename).get("allow_upstream_github_fallback", True)):
        urls.append(f"{upstream_base_url or UPSTREAM_DATASET_BASE_URL}/{filename}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp_target = dest_dir / f"{filename}.tmp"

    last_error: Exception | None = None
    for url in urls:
        try:
            if verbose:
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


def resolve_dataset_bundle(dataset: Path | str, *, workspace_root: Path | None = None) -> DatasetBundle:
    path = Path(dataset)
    if path.is_dir():
        return load_dataset_bundle(path)
    if path.name == "manifest.json" and path.is_file():
        return load_dataset_bundle(path.parent)
    if path.suffix or len(path.parts) > 1:
        raise ValueError(f"Expected a prepared bundle directory or manifest, got {dataset}")
    canonical = resolve_dataset_name(str(dataset))
    root = (workspace_root or Path(__file__).resolve().parents[3]) / "outputs" / "datasets" / canonical
    if not (root / "manifest.json").is_file():
        raise ValueError(f"Prepared dataset missing: {root}. Run scripts/download-datasets.py --dataset {canonical}.")
    return load_dataset_bundle(root)


def load_dataset_files(
    dataset_name: str,
    dataset_dir: Path | None = None,
    *,
    max_queries: int | None = None,
    include_corpus: bool = True,
    project_root: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bundle = (
        load_dataset_bundle(dataset_dir / resolve_dataset_name(dataset_name))
        if dataset_dir
        else resolve_dataset_bundle(dataset_name, workspace_root=project_root)
    )
    samples, document_rows = bundle.read_inputs()
    if max_queries is not None:
        if max_queries < 1:
            raise ValueError("max_queries must be positive")
        samples = samples[:max_queries]
    corpus = (
        [dict(row, idx=doc.document_id) for doc, row in zip(bundle.documents, document_rows, strict=True)]
        if include_corpus
        else []
    )
    return samples, corpus


def extract_gold_answers(sample: Mapping[str, Any]) -> list[str]:
    """Extract gold answers and aliases from a single dataset sample."""
    if "answers" in sample and sample["answers"] is not None:
        gold_answer: Any = sample["answers"]
    elif "answer" in sample or "gold_ans" in sample or "gold_answer" in sample:
        gold_answer = sample.get("answer", sample.get("gold_ans", sample.get("gold_answer")))
    elif "gold_answers" in sample and sample["gold_answers"] is not None:
        gold_answer = sample["gold_answers"]
    elif "reference" in sample:
        gold_answer = sample["reference"]
    elif "obj" in sample:
        gold_answer = [sample["obj"]]
        if sample.get("o_wiki_title"):
            gold_answer.append(sample["o_wiki_title"])
        for field in ("possible_answers", "o_aliases"):
            value = sample.get(field, [])
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        gold_answer.extend(parsed)
                    elif parsed:
                        gold_answer.append(parsed)
                except json.JSONDecodeError, TypeError:
                    gold_answer.append(value)
            elif isinstance(value, list):
                gold_answer.extend(value)
            elif value:
                gold_answer.append(value)
    else:
        raise ValueError("Each query sample must contain an answer field")
    if gold_answer is None:
        raise ValueError("Answer fields must not be null")
    answers = [gold_answer] if isinstance(gold_answer, str) else list(gold_answer)
    answers.extend(sample.get("answer_aliases", []))
    return list(dict.fromkeys(str(answer) for answer in answers))


def get_gold_answers(samples: Sequence[Mapping[str, Any]]) -> list[list[str]]:
    """Extract gold answer sets and aliases using upstream main.py's exact contract."""
    return [extract_gold_answers(sample) for sample in samples]


def _extract_sample_passages(
    sample: Mapping[str, Any],
    *,
    separator: str = " ",
    supporting_only: bool = False,
    corpus_by_id: Mapping[str, str] | None = None,
) -> list[str]:
    if "gold_docs" in sample and sample["gold_docs"]:
        return [
            f"{item['title']}\n{item['text']}" if isinstance(item, Mapping) else str(item)
            for item in sample["gold_docs"]
        ]
    if supporting_only and "supporting_facts" in sample:
        gold_titles = {item[0] for item in sample["supporting_facts"]}
        return [f"{item[0]}\n{separator.join(item[1])}" for item in sample["context"] if item[0] in gold_titles]
    if "contexts" in sample:
        items = (
            [item for item in sample["contexts"] if item.get("is_supporting", False)]
            if supporting_only
            else sample["contexts"]
        )
        return [f"{item['title']}\n{item['text']}" for item in items]
    if "paragraphs" in sample:
        items = (
            [item for item in sample["paragraphs"] if item.get("is_supporting", True)]
            if supporting_only
            else sample["paragraphs"]
        )
        return [f"{item['title']}\n{item['text'] if 'text' in item else item['paragraph_text']}" for item in items]
    if "context" in sample and isinstance(sample["context"], list):
        return [f"{item[0]}\n{separator.join(item[1])}" for item in sample["context"]]
    if "expected_doc_ids" in sample:
        lookup = corpus_by_id or {}
        return [lookup[str(did)] for did in sample.get("expected_doc_ids", []) if supporting_only or str(did) in lookup]
    if supporting_only:
        raise ValueError(
            "Each sample needs supporting_facts, contexts, paragraphs, context, or expected_doc_ids "
            "for retrieval evaluation"
        )
    return []


def get_gold_docs(
    samples: Sequence[Mapping[str, Any]],
    dataset_name: str | Path | None = None,
    corpus: Sequence[Mapping[str, Any]] | None = None,
    *,
    corpus_by_id: Mapping[str, str] | None = None,
) -> list[list[str]] | None:
    """Extract gold supporting passages using upstream main.py's exact formatting contract."""
    spec = _dataset_spec(dataset_name)
    if spec.get("qa_only"):
        return None
    separator = str(spec.get("sentence_separator", " "))
    c_by_id: dict[str, str] = dict(corpus_by_id) if corpus_by_id else {}
    if not c_by_id and corpus:
        for c in corpus:
            cid = str(c.get("idx") or c.get("id") or c.get("doc_id") or "")
            if cid:
                t = str(c.get("title", "")).strip()
                tx = str(c.get("text") if "text" in c else c.get("content", "")).strip()
                c_by_id[cid] = f"{t}\n{tx}" if t else tx
    return [
        list(
            dict.fromkeys(
                _extract_sample_passages(
                    sample,
                    separator=separator,
                    supporting_only=True,
                    corpus_by_id=c_by_id,
                )
            )
        )
        for sample in samples
    ]


def build_benchmark_slice(
    dataset_name: str,
    samples: Sequence[Mapping[str, Any]],
    corpus: Sequence[Mapping[str, Any]],
    max_queries: int | None = None,
    query_offset: int = 0,
    full_corpus: bool = False,
    index_queries: int | None = None,
) -> tuple[list[str], list[str], list[list[str]] | None, list[list[str]]]:
    """Build manifest document IDs and evaluation labels, retaining benchmark subset policies."""
    if max_queries is not None and max_queries < 1:
        raise ValueError("max_queries must be a positive integer when specified.")
    if index_queries is not None and index_queries < 1:
        raise ValueError("index_queries must be a positive integer when specified.")

    start = max(0, query_offset)
    end = None if max_queries is None else start + max_queries
    selected = list(samples[start:end])
    queries = [str(sample["question"]) for sample in selected]
    spec = _dataset_spec(dataset_name)
    gold_docs = get_gold_docs(selected, dataset_name, corpus=corpus)
    gold_answers = get_gold_answers(selected)
    minimum_document_count = 0

    if full_corpus or max_queries is None or bool(spec.get("full_corpus_slice")):
        docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]
    else:
        effective_index_queries = int(max_queries * 1.4) if index_queries is None else index_queries
        pooled_end = start + effective_index_queries
        separator = str(spec.get("sentence_separator", " "))
        docs = [
            passage
            for sample in samples[start:pooled_end]
            for passage in _extract_sample_passages(sample, separator=separator)
        ]
        if (min_per_query := int(spec.get("min_docs_per_query", 0))) > 0:
            minimum_document_count = effective_index_queries * min_per_query
            docs = list(dict.fromkeys(docs))
            if len(docs) < minimum_document_count:
                for sample in samples[pooled_end:]:
                    for cand in _extract_sample_passages(sample, separator=separator):
                        if cand not in docs:
                            docs.append(cand)
                            if len(docs) >= minimum_document_count:
                                break
                    if len(docs) >= minimum_document_count:
                        break

    if gold_docs is not None:
        for gold_list in gold_docs:
            docs.extend(gold_list)
    selected_text = set(docs)
    document_ids = [document_identity(c) for c in corpus if f"{c['title']}\n{c['text']}" in selected_text]
    seen_ids = set(document_ids)
    for record in corpus:
        if len(document_ids) >= minimum_document_count:
            break
        document_id = document_identity(record)
        if document_id not in seen_ids:
            document_ids.append(document_id)
            seen_ids.add(document_id)
    return document_ids, queries, gold_docs, gold_answers


def select_enterprise_questions(
    rows: Sequence[dict[str, Any]],
    *,
    seed: int = 42,
    quotas: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Select the existing 150/100 cohort without modifying any source row."""
    effective_quotas: dict[str, tuple[int, int]] = {
        str(k): (int(v[0]), int(v[1])) if isinstance(v, Sequence) else (int(v) * 3 // 5, int(v) - int(v) * 3 // 5)
        for k, v in (quotas or ENTERPRISE_CATEGORY_QUOTAS).items()
    }
    total_cohort = sum(nt + nv for nt, nv in effective_quotas.values())
    identities = [str(row.get("question_id", row.get("id", row["question"]))) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate Enterprise question identities cannot form disjoint splits")
    if len(rows) == total_cohort and all(row.get("split") in ("train", "val") for row in rows):
        for category, (train_count, val_count) in effective_quotas.items():
            for split, count in (("train", train_count), ("val", val_count)):
                if sum(row["question_type"] == category and row["split"] == split for row in rows) != count:
                    raise ValueError(f"Invalid Enterprise {category}/{split} split count")
        return [r for r in rows if r["split"] == "train"] + [r for r in rows if r["split"] == "val"]
    rng = random.Random(seed)
    train, val = [], []
    for category, (nt, nv) in effective_quotas.items():
        pool = sorted(
            (r for r in rows if r["question_type"] == category),
            key=lambda r: str(r.get("question_id", r.get("id", r["question"]))),
        )
        if len(pool) < nt + nv:
            raise ValueError(f"Enterprise category {category!r} needs {nt + nv} rows, got {len(pool)}")
        rng.shuffle(pool)
        train.extend(pool[:nt])
        val.extend(pool[nt : nt + nv])
    rng.shuffle(train)
    rng.shuffle(val)
    return train + val


def load_dataset_splits(
    dataset_args: Sequence[str | Path] | str | None,
    *,
    workspace_root: Path | None = None,
    max_queries: int = 20,
    train_queries: int = 7,
    query_offset: int = 0,
    example_factory: Callable[..., Any] | None = None,
) -> tuple[list[Any], list[Any] | None, list[str]]:
    if not dataset_args:
        raise ValueError("An explicit prepared dataset is required")
    if max_queries < 2 or not 0 < train_queries < max_queries or query_offset < 0:
        raise ValueError("Select at least two queries with distinct nonempty training and validation partitions")
    args = str(dataset_args).split(",") if isinstance(dataset_args, str) else dataset_args
    trains, vals, docs = [], [], []
    for arg in args:
        spec = _dataset_spec(arg)
        if spec.get("held_out") or ((spec.get("tier2") or spec.get("default_parity")) and max_queries > 50):
            raise YawRuntimeError(
                "INVALID_INPUT",
                f"Held-out evaluation dataset '{arg}' is a held-out benchmark "
                "and cannot be used for training/optimization.",
            )
        bundle = resolve_dataset_bundle(arg, workspace_root=workspace_root)
        spec = _dataset_spec(bundle.dataset) or spec
        if spec.get("held_out") or ((spec.get("tier2") or spec.get("default_parity")) and max_queries > 50):
            raise YawRuntimeError(
                "INVALID_INPUT",
                f"Held-out evaluation dataset '{bundle.dataset}' is a held-out benchmark "
                "and cannot be used for training/optimization.",
            )
        rows, document_rows = bundle.read_inputs()
        if quotas := spec.get("category_quotas"):
            cohort = select_enterprise_questions(rows, quotas=quotas)
            train_part = int(spec.get("train_partition_size", 150))
            rows = (
                cohort[:train_part][query_offset : query_offset + train_queries]
                + cohort[train_part:][query_offset : query_offset + max_queries - train_queries]
            )
            if len(rows) != max_queries:
                raise ValueError(
                    f"Requested Enterprise slice exceeds its fixed {train_part}/{len(cohort) - train_part} partitions"
                )
        else:
            rows = rows[query_offset : query_offset + max_queries]
        if len(rows) <= train_queries:
            raise ValueError("Insufficient distinct validation examples in selected dataset slice")
        corpus_rows = [
            dict(row, idx=entry.document_id) for entry, row in zip(bundle.documents, document_rows, strict=True)
        ]
        by_id = {row["idx"]: document_text(row, bundle.title_key, bundle.body_key) for row in corpus_rows}
        gold = get_gold_docs(rows, bundle.dataset, corpus=corpus_rows, corpus_by_id=by_id)
        gold_lists: Sequence[list[str]] = gold if gold is not None else [[] for _ in rows]
        examples = []
        for row, references in zip(rows, gold_lists, strict=True):
            values = dict(
                question=row["question"], gold_answers=extract_gold_answers(row), gold_docs=references, item_docs=[]
            )
            for key in ("answer_facts", "question_type", "expected_doc_ids"):
                if key in row:
                    values[key] = row[key]
            examples.append(example_factory(_inputs=("question", "item_docs"), **values) if example_factory else values)
        trains.append(examples[:train_queries])
        vals.append(examples[train_queries:])
        docs.extend(by_id.values())
    train = [group[i] for i in range(max(map(len, trains))) for group in trains if i < len(group)]
    val = [group[i] for i in range(max(map(len, vals))) for group in vals if i < len(group)]
    return train, val, list(dict.fromkeys(docs))
