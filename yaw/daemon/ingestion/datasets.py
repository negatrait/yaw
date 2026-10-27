"""Centralized dataset registry, downloader, slice builder, and CocoIndex direct-source ingestion helpers."""

import json
import random
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from yaw.core import YawRuntimeError

from .bundles import DatasetBundle, document_identity, document_text, load_dataset_bundle

PINNED_COMMIT = "d5c8329422e0a0b834a15874545cb6a74b4f9b26"
HF_BASE_URL = "https://huggingface.co/datasets/osunlp/HippoRAG_2/resolve/main"
UPSTREAM_DATASET_BASE_URL = (
    f"https://raw.githubusercontent.com/OSU-NLP-Group/HippoRAG/{PINNED_COMMIT}/reproduce/dataset"
)

DATASETS: tuple[str, ...] = (
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
)
DEFAULT_PARITY_DATASETS: tuple[str, ...] = ("2wikimultihopqa", "hotpotqa", "musique")
TIER2_CORPORA: frozenset[str] = frozenset({"nq", "nq_rear", "popqa", "lveval", "narrativeqa"})

DATASET_ALIASES: dict[str, str] = {
    "2wiki": "2wikimultihopqa",
    "hotpot": "hotpotqa",
    "naturalquestions": "nq",
    "nq_rear": "nq",
    "narrative": "narrativeqa",
    "narrativeqa_dev_10_doc": "narrativeqa",
    "enterprise": "enterprise_rag_bench",
}
ALIASES: dict[str, str] = DATASET_ALIASES

DATASET_REGISTRY: dict[str, tuple[str, str]] = {
    "sample": ("sample.json", "sample_corpus.json"),
    "2wikimultihopqa": ("2wikimultihopqa.json", "2wikimultihopqa_corpus.json"),
    "hotpotqa": ("hotpotqa.json", "hotpotqa_corpus.json"),
    "musique": ("musique.json", "musique_corpus.json"),
    "nq": ("nq_rear.json", "nq_rear_corpus.json"),
    "popqa": ("popqa.json", "popqa_corpus.json"),
    "lveval": ("lveval.json", "lveval_corpus.json"),
    "narrativeqa": ("narrativeqa_dev_10_doc.json", "narrativeqa_dev_10_doc_corpus.json"),
    "case_study_university": ("case_study_university.json", "case_study_university_corpus.json"),
    "enterprise_rag_bench": ("enterprise_rag_bench_questions.json", "enterprise_rag_bench_corpus.json"),
}

DATASET_SOURCE_FIELDS: dict[str, dict[str, Any]] = {
    name: {
        "id_key": None if name in ("2wikimultihopqa", "musique", "popqa") else "idx",
        "title_key": "title",
        "body_key": "text",
    }
    for name in DATASET_REGISTRY
}

DATASET_DESCRIPTIONS: dict[str, str] = {
    "sample": "Tier 0 validation fixture (1 query, 3 passages)",
    "2wikimultihopqa": "Multi-hop QA from Wikipedia (1,000 queries, 6,119 passages)",
    "hotpotqa": "Multi-hop QA with diverse reasoning (1,000 queries, 9,811 passages)",
    "musique": "Complex multi-hop QA with 2-4 hops (1,000 queries, 11,656 passages)",
    "nq": "NaturalQuestions single-hop QA (1,000 queries, 9,633 passages)",
    "popqa": "PopQA entity-centric long-tail QA (1,000 queries, 8,676 passages)",
    "lveval": "LV-Eval hotpotwikiqa-mixup 256k long-context QA (124 queries, 22,849 passages)",
    "narrativeqa": "NarrativeQA discourse understanding on novels (293 queries, 4,111 passages)",
    "case_study_university": "University Case Study graph reasoning demo",
    "enterprise_rag_bench": (
        "EnterpriseRAG-Bench stratified 10-category multi-tool dataset (250 queries: 150 train + 100 val)"
    ),
}

TEXT_STOPWORDS: frozenset[str] = frozenset(
    {
        "about",
        "after",
        "again",
        "against",
        "all",
        "also",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "back",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        " come",
        "could",
        "day",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "even",
        "few",
        "first",
        "for",
        "from",
        "further",
        "get",
        "give",
        "go",
        "good",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "know",
        "like",
        "look",
        "make",
        "many",
        "may",
        "me",
        "might",
        "more",
        "most",
        "much",
        "must",
        "my",
        "myself",
        "new",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "one",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "part",
        "people",
        "place",
        "same",
        "see",
        "she",
        "should",
        "so",
        "some",
        "state",
        "such",
        "take",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "think",
        "this",
        "time",
        "to",
        "two",
        "up",
        "use",
        "was",
        "way",
        "we",
        "well",
        "were",
        "what",
        "when",
        "which",
        "who",
        "will",
        "with",
        "work",
        "would",
        "year",
        "you",
        "your",
    }
)

MONKEYPATCHED_HIPPORAG_BASELINES: dict[str, dict[str, Any]] = {
    "2wikimultihopqa": {
        "Recall@1": 0.4545,
        "Recall@2": 0.6591,
        "Recall@5": 0.9091,
        "ExactMatch": 0.6364,
        "F1": 0.7227,
        "indexing_seconds": 616.76,
        "evaluation_seconds": 173.44,
        "elapsed_seconds": 790.73,
    },
    "musique": {
        "Recall@1": 0.3636,
        "Recall@2": 0.5682,
        "Recall@5": 0.6818,
        "ExactMatch": 0.4091,
        "F1": 0.5485,
        "indexing_seconds": 1152.93,
        "evaluation_seconds": 284.18,
        "elapsed_seconds": 1437.36,
    },
    "hotpotqa": {
        "Recall@1": 0.4750,
        "Recall@2": 0.7750,
        "Recall@5": 0.9000,
        "ExactMatch": 0.4500,
        "F1": 0.6426,
        "indexing_seconds": 599.21,
        "evaluation_seconds": 135.75,
        "elapsed_seconds": 736.13,
    },
    "nq": {
        "Recall@1": 0.5833,
        "Recall@2": 0.8083,
        "Recall@5": 0.9250,
        "ExactMatch": 0.4500,
        "F1": 0.6327,
        "indexing_seconds": 807.52,
        "evaluation_seconds": 110.22,
        "elapsed_seconds": 917.85,
    },
    "popqa": {
        "Recall@1": 0.8667,
        "Recall@2": 0.9467,
        "Recall@5": 1.0000,
        "ExactMatch": 0.6800,
        "F1": 0.7419,
        "indexing_seconds": 236.44,
        "evaluation_seconds": 274.38,
        "elapsed_seconds": 511.12,
    },
    "lveval": {
        "Recall@1": 0.6000,
        "Recall@2": 0.8500,
        "Recall@5": 0.9500,
        "ExactMatch": 0.5000,
        "F1": 0.5750,
        "indexing_seconds": 182.23,
        "evaluation_seconds": 87.34,
        "elapsed_seconds": 269.67,
    },
    "narrativeqa": {
        "Recall@1": 0.5000,
        "Recall@2": 0.6500,
        "Recall@5": 0.8000,
        "ExactMatch": 0.1000,
        "F1": 0.4557,
        "indexing_seconds": 333.89,
        "evaluation_seconds": 146.52,
        "elapsed_seconds": 480.52,
    },
}

PAPER_BASELINES: dict[str, dict[str, dict[str, float]]] = {
    "musique": {
        "hipporag2_llama3.3_70b_nvembed_v2": {
            "Recall@2": 0.561,
            "Recall@5": 0.747,
            "ExactMatch": 0.372,
            "F1": 0.486,
        },
        "standard_nvembed_v2_llama3.3_70b": {
            "Recall@2": 0.527,
            "Recall@5": 0.697,
            "ExactMatch": 0.347,
            "F1": 0.457,
        },
        "hipporag2_gpt4o_mini_nvembed_v2": {
            "Recall@2": 0.535,
            "Recall@5": 0.742,
            "ExactMatch": 0.350,
            "F1": 0.493,
        },
        "contriever_llama3.3_70b": {
            "Recall@2": 0.348,
            "Recall@5": 0.466,
            "ExactMatch": 0.240,
            "F1": 0.313,
        },
    },
    "2wikimultihopqa": {
        "hipporag2_llama3.3_70b_nvembed_v2": {
            "Recall@2": 0.762,
            "Recall@5": 0.904,
            "ExactMatch": 0.650,
            "F1": 0.710,
        },
        "standard_nvembed_v2_llama3.3_70b": {
            "Recall@2": 0.671,
            "Recall@5": 0.765,
            "ExactMatch": 0.575,
            "F1": 0.615,
        },
        "hipporag2_gpt4o_mini_nvembed_v2": {
            "Recall@2": 0.746,
            "Recall@5": 0.902,
            "ExactMatch": 0.605,
            "F1": 0.697,
        },
        "contriever_llama3.3_70b": {
            "Recall@2": 0.466,
            "Recall@5": 0.575,
            "ExactMatch": 0.381,
            "F1": 0.419,
        },
    },
    "hotpotqa": {
        "hipporag2_llama3.3_70b_nvembed_v2": {
            "Recall@2": 0.835,
            "Recall@5": 0.963,
            "ExactMatch": 0.627,
            "F1": 0.755,
        },
        "standard_nvembed_v2_llama3.3_70b": {
            "Recall@2": 0.841,
            "Recall@5": 0.945,
            "ExactMatch": 0.628,
            "F1": 0.753,
        },
        "hipporag2_gpt4o_mini_nvembed_v2": {
            "Recall@2": 0.805,
            "Recall@5": 0.957,
            "ExactMatch": 0.563,
            "F1": 0.711,
        },
        "contriever_llama3.3_70b": {
            "Recall@2": 0.584,
            "Recall@5": 0.753,
            "ExactMatch": 0.513,
            "F1": 0.623,
        },
    },
    "nq": {
        "hipporag2_llama3.3_70b_nvembed_v2": {
            "Recall@2": 0.456,
            "Recall@5": 0.780,
            "ExactMatch": 0.485,
            "F1": 0.633,
        },
        "standard_nvembed_v2_llama3.3_70b": {"Recall@2": 0.453, "Recall@5": 0.754, "F1": 0.619},
        "hipporag_reproduced": {"Recall@2": 0.213, "Recall@5": 0.444, "F1": 0.553},
    },
    "popqa": {
        "hipporag2_llama3.3_70b_nvembed_v2": {
            "Recall@2": 0.439,
            "Recall@5": 0.517,
            "ExactMatch": 0.490,
            "F1": 0.562,
        },
        "standard_nvembed_v2_llama3.3_70b": {"Recall@2": 0.453, "Recall@5": 0.510, "F1": 0.557},
        "hipporag_reproduced": {"Recall@2": 0.400, "Recall@5": 0.538, "F1": 0.559},
    },
    "lveval": {
        "hipporag2_llama3.3_70b_nvembed_v2": {"ExactMatch": 0.080, "F1": 0.129},
        "standard_nvembed_v2_llama3.3_70b": {"F1": 0.098},
        "hipporag_reproduced": {"F1": 0.084},
    },
    "narrativeqa": {
        "hipporag2_llama3.3_70b_nvembed_v2": {"ExactMatch": 0.180, "F1": 0.259},
        "standard_nvembed_v2_llama3.3_70b": {"F1": 0.257},
        "hipporag_reproduced": {"F1": 0.163},
    },
}

_OPTIMIZE_TIER2_CORPORA = frozenset({"2wikimultihopqa", "musique", "hotpotqa", *TIER2_CORPORA})


def resolve_dataset_name(name: str) -> str:
    """Resolve CLI/benchmark dataset aliases to canonical dataset keys."""
    cleaned = name.strip().lower()
    resolved = DATASET_ALIASES.get(cleaned, cleaned)
    if resolved not in DATASET_REGISTRY:
        stem = Path(cleaned).stem.lower()
        resolved = DATASET_ALIASES.get(stem, stem)
    if resolved not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. Valid datasets: {', '.join(DATASETS)} (aliases: {', '.join(DATASET_ALIASES)})"
        )
    return resolved


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
    if any(k in filename for k in ("sample", "2wiki", "hotpot", "musique")):
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


def extract_gold_answers(sample: Mapping[str, Any], *, strict: bool = False) -> list[str]:
    """Extract gold answers and aliases from a single dataset sample."""
    if not strict:
        if "gold_answers" in sample:
            raw = sample["gold_answers"]
        elif "answer" in sample or "gold_ans" in sample or "gold_answer" in sample:
            raw = sample.get("answer", sample.get("gold_ans", sample.get("gold_answer")))
        elif "reference" in sample:
            raw = sample["reference"]
        elif "answers" in sample and sample["answers"] is not None:
            raw = sample["answers"]
        else:
            raw = []
        ans = [raw] if isinstance(raw, str) else [str(a) for a in (raw or ())]
        ans.extend(str(a) for a in sample.get("answer_aliases", ()))
        return list(dict.fromkeys(ans))

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
    return [extract_gold_answers(sample, strict=True) for sample in samples]


def get_gold_docs(
    samples: Sequence[Mapping[str, Any]],
    dataset_name: str | Path | None = None,
    corpus: Sequence[Mapping[str, Any]] | None = None,
    *,
    corpus_by_id: Mapping[str, str] | None = None,
) -> list[list[str]]:
    """Extract gold supporting passages using upstream main.py's exact formatting contract."""
    gold_docs: list[list[str]] = []
    ds_str = str(dataset_name).strip() if dataset_name is not None else ""
    ds_stem = Path(ds_str).stem.lower() if ds_str else ""
    canonical_dataset = ALIASES.get(ds_stem, ALIASES.get(ds_str.lower(), ds_str)) if ds_str else ""
    separator = "" if canonical_dataset.startswith("hotpotqa") else " "
    c_by_id: dict[str, str] = dict(corpus_by_id) if corpus_by_id else {}
    if not c_by_id and corpus:
        for c in corpus:
            cid = str(c.get("idx") or c.get("id") or c.get("doc_id") or "")
            if cid:
                t = str(c.get("title", "")).strip()
                tx = str(c.get("text") if "text" in c else c.get("content", "")).strip()
                c_by_id[cid] = f"{t}\n{tx}" if t else tx

    for sample in samples:
        if "gold_docs" in sample and sample["gold_docs"]:
            gold_doc = [
                f"{item['title']}\n{item['text']}" if isinstance(item, Mapping) else str(item)
                for item in sample["gold_docs"]
            ]
        elif "supporting_facts" in sample:
            gold_titles = {item[0] for item in sample["supporting_facts"]}
            supporting_contexts = [item for item in sample["context"] if item[0] in gold_titles]
            gold_doc = [f"{item[0]}\n{separator.join(item[1])}" for item in supporting_contexts]
        elif "contexts" in sample:
            gold_doc = [
                f"{item['title']}\n{item['text']}" for item in sample["contexts"] if item.get("is_supporting", False)
            ]
        elif "paragraphs" in sample:
            paragraphs = [item for item in sample["paragraphs"] if item.get("is_supporting", True)]
            gold_doc = [
                f"{item['title']}\n{item['text'] if 'text' in item else item['paragraph_text']}" for item in paragraphs
            ]
        elif "context" in sample and isinstance(sample["context"], list):
            gold_doc = [f"{item[0]}\n{separator.join(item[1])}" for item in sample["context"]]
        elif isinstance(sample.get("context"), str):
            passages = [p.strip() for p in re.split(r"\n*### Passage \d+\n*", sample["context"]) if p.strip()]
            ans_set: set[str] = set()
            if sample.get("gold_ans"):
                ans_set.add(str(sample["gold_ans"]).strip())
            for a in sample.get("answers", []):
                if a:
                    ans_set.add(str(a).strip())
            if "answer" in sample and sample["answer"]:
                if isinstance(sample["answer"], list):
                    ans_set.update(str(x).strip() for x in sample["answer"] if x)
                else:
                    ans_set.add(str(sample["answer"]).strip())

            question = str(sample.get("question", ""))
            q_keywords = {w for w in re.findall(r"\w+", question.lower()) if len(w) >= 3 and w not in TEXT_STOPWORDS}
            a_keywords = {
                w for a in ans_set for w in re.findall(r"\w+", a.lower()) if len(w) >= 3 and w not in TEXT_STOPWORDS
            }
            ans_lower = [a.lower() for a in ans_set if a]

            scored_passages: list[tuple[tuple[int, int, int, int], str]] = []
            for idx, p in enumerate(passages):
                p_lower = p.lower()
                p_words = set(re.findall(r"\w+", p_lower))
                exact_ans_matches = sum(1 for a in ans_lower if a in p_lower)
                ans_kw_matches = len(a_keywords & p_words)
                q_kw_matches = len(q_keywords & p_words)

                if exact_ans_matches > 0 or ans_kw_matches > 0:
                    scored_passages.append(((exact_ans_matches, ans_kw_matches, q_kw_matches, -idx), p))

            if scored_passages:
                scored_passages.sort(key=lambda x: x[0], reverse=True)
                top_passages = [p for _, p in scored_passages[:2]]
            elif passages:
                top_passages = [passages[0]]
            else:
                top_passages = []

            gold_doc = [f"\n{p}" for p in top_passages]
        elif "expected_doc_ids" in sample:
            lookup = c_by_id
            gold_doc = [lookup[str(did)] for did in sample.get("expected_doc_ids", [])]
        elif "document" in sample and isinstance(sample["document"], Mapping):
            doc = sample["document"]
            doc_id = doc.get("id")
            gold_doc = []
            if corpus and doc_id:
                doc_chunks = [c for c in corpus if str(c.get("idx", "")).startswith(f"{doc_id}_")]
                answers = get_gold_answers([sample])[0]
                ans_lower = [a.lower().strip() for a in answers if a.strip()]
                a_keywords = {
                    w for a in answers for w in re.findall(r"\w+", a.lower()) if len(w) >= 3 and w not in TEXT_STOPWORDS
                }
                question = str(sample.get("question", ""))
                q_keywords = {
                    w for w in re.findall(r"\w+", question.lower()) if len(w) >= 3 and w not in TEXT_STOPWORDS
                }

                scored_chunks: list[tuple[tuple[int, int, int, int], str]] = []
                for idx, c in enumerate(doc_chunks):
                    c_text = str(c.get("text", ""))
                    c_lower = c_text.lower()
                    c_words = set(re.findall(r"\w+", c_lower))
                    exact_ans_matches = sum(1 for a in ans_lower if a in c_lower)
                    ans_kw_matches = len(a_keywords & c_words)
                    q_kw_matches = len(q_keywords & c_words)

                    if exact_ans_matches > 0 or ans_kw_matches > 0:
                        formatted = f"{c['title']}\n{c_text}"
                        scored_chunks.append(((exact_ans_matches, ans_kw_matches, q_kw_matches, -idx), formatted))

                if scored_chunks:
                    scored_chunks.sort(key=lambda x: x[0], reverse=True)
                    gold_doc = [chunk for _, chunk in scored_chunks[:2]]
                elif doc_chunks:
                    gold_doc = [f"{doc_chunks[0]['title']}\n{doc_chunks[0]['text']}"]
            if not gold_doc:
                summary = doc.get("summary")
                if isinstance(summary, Mapping) and summary.get("text"):
                    gold_doc = [f"{summary.get('title', '')}\n{summary['text']}".strip()]
                elif isinstance(summary, str) and summary.strip():
                    gold_doc = [summary.strip()]
                elif doc.get("text"):
                    gold_doc = [str(doc["text"]).strip()]
                else:
                    gold_doc = []
        else:
            raise ValueError(
                "Each sample needs supporting_facts, contexts, paragraphs, context, expected_doc_ids, or "
                "document for retrieval evaluation"
            )
        gold_docs.append(list(dict.fromkeys(gold_doc)))
    return gold_docs


def build_benchmark_slice(
    dataset_name: str,
    samples: Sequence[Mapping[str, Any]],
    corpus: Sequence[Mapping[str, Any]],
    max_queries: int | None = None,
    query_offset: int = 0,
    full_corpus: bool = False,
    index_queries: int | None = None,
) -> tuple[list[str], list[str], list[list[str]], list[list[str]]]:
    """Build manifest document IDs and evaluation labels, retaining benchmark subset policies."""
    if max_queries is not None and max_queries < 1:
        raise ValueError("max_queries must be a positive integer when specified.")
    if index_queries is not None and index_queries < 1:
        raise ValueError("index_queries must be a positive integer when specified.")

    start = max(0, query_offset)
    end = None if max_queries is None else start + max_queries
    selected = list(samples[start:end])
    queries = [str(sample["question"]) for sample in selected]
    canonical_dataset = ALIASES.get(dataset_name or "", dataset_name or "")
    gold_docs = get_gold_docs(selected, canonical_dataset, corpus=corpus)
    gold_answers = get_gold_answers(selected)
    minimum_document_count = 0

    if full_corpus or max_queries is None or canonical_dataset in ("sample", "enterprise_rag_bench"):
        docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]
    else:
        effective_index_queries = int(max_queries * 1.4) if index_queries is None else index_queries
        pooled_end = start + effective_index_queries
        pooled_samples = list(samples[start:pooled_end])

        docs = []
        separator = "" if canonical_dataset.startswith("hotpotqa") else " "

        if canonical_dataset in ("lveval", "narrativeqa"):
            for gold_list in gold_docs:
                docs.extend(gold_list)
            gold_unique = list(dict.fromkeys(docs))
            docs = list(gold_unique)

            target_pool_size = max(
                len(gold_unique) + (effective_index_queries - max_queries) * 2,
                effective_index_queries * 2,
            )
            minimum_document_count = target_pool_size

            candidate_pool: list[str] = []
            if canonical_dataset == "lveval":
                for sample in pooled_samples:
                    if isinstance(sample.get("context"), str):
                        passages = [
                            p.strip() for p in re.split(r"\n*### Passage \d+\n*", sample["context"]) if p.strip()
                        ]
                        for p in passages:
                            candidate_pool.append(f"\n{p}")
                if corpus and len(candidate_pool) < target_pool_size:
                    for c in corpus:
                        candidate_pool.append(f"{c['title']}\n{c['text']}")
            else:
                for sample in pooled_samples:
                    doc = sample.get("document", {})
                    doc_id = doc.get("id") if isinstance(doc, Mapping) else None
                    if doc_id and corpus:
                        for c in corpus:
                            if str(c.get("idx", "")).startswith(f"{doc_id}_"):
                                candidate_pool.append(f"{c['title']}\n{c['text']}")
                    summary = doc.get("summary") if isinstance(doc, Mapping) else None
                    if isinstance(summary, Mapping) and summary.get("text"):
                        candidate_pool.append(f"{summary.get('title', '')}\n{summary['text']}".strip())
                if corpus and len(candidate_pool) < target_pool_size:
                    for c in corpus:
                        candidate_pool.append(f"{c['title']}\n{c['text']}")

            for cand in candidate_pool:
                if len(docs) >= target_pool_size:
                    break
                if cand not in docs:
                    docs.append(cand)
        else:
            for sample in pooled_samples:
                if "context" in sample:
                    if isinstance(sample["context"], list):
                        for title, sents in sample["context"]:
                            docs.append(f"{title}\n{separator.join(sents)}")
                    elif isinstance(sample["context"], str):
                        passages = [
                            p.strip() for p in re.split(r"\n*### Passage \d+\n*", sample["context"]) if p.strip()
                        ]
                        for p in passages:
                            docs.append(f"\n{p}")
                elif "contexts" in sample:
                    for item in sample["contexts"]:
                        docs.append(f"{item['title']}\n{item['text']}")
                elif "paragraphs" in sample:
                    for item in sample["paragraphs"]:
                        text = item["text"] if "text" in item else item["paragraph_text"]
                        docs.append(f"{item['title']}\n{text}")
            if canonical_dataset == "popqa":
                min_popqa_docs = effective_index_queries * 2
                minimum_document_count = min_popqa_docs
                docs = list(dict.fromkeys(docs))
                if len(docs) < min_popqa_docs:
                    for sample in samples[pooled_end:]:
                        if "paragraphs" in sample:
                            for item in sample["paragraphs"]:
                                text = item["text"] if "text" in item else item["paragraph_text"]
                                cand = f"{item['title']}\n{text}"
                                if cand not in docs:
                                    docs.append(cand)
                                    if len(docs) >= min_popqa_docs:
                                        break
                        if len(docs) >= min_popqa_docs:
                            break
                if len(docs) < min_popqa_docs and corpus:
                    for c in corpus:
                        cand = f"{c['title']}\n{c['text']}"
                        if cand not in docs:
                            docs.append(cand)
                            if len(docs) >= min_popqa_docs:
                                break

    for gold_list in gold_docs:
        docs.extend(gold_list)
    selected_text = set(docs)
    document_ids = [document_identity(c) for c in corpus if f"{c['title']}\n{c['text']}" in selected_text]
    # Multiple evaluation views (summary/title/body) can identify one source.
    # Pad after resolving identities so the pool contains real distinct documents.
    seen_ids = set(document_ids)
    for record in corpus:
        if len(document_ids) >= minimum_document_count:
            break
        document_id = document_identity(record)
        if document_id not in seen_ids:
            document_ids.append(document_id)
            seen_ids.add(document_id)
    return document_ids, queries, gold_docs, gold_answers


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


def select_enterprise_questions(rows: Sequence[dict[str, Any]], *, seed: int = 42) -> list[dict[str, Any]]:
    """Select the existing 150/100 cohort without modifying any source row."""
    identities = [str(row.get("question_id", row.get("id", row["question"]))) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate Enterprise question identities cannot form disjoint splits")
    if len(rows) == 250 and all(row.get("split") in ("train", "val") for row in rows):
        for category, (train_count, val_count) in ENTERPRISE_CATEGORY_QUOTAS.items():
            for split, count in (("train", train_count), ("val", val_count)):
                if sum(row["question_type"] == category and row["split"] == split for row in rows) != count:
                    raise ValueError(f"Invalid Enterprise {category}/{split} split count")
        return [r for r in rows if r["split"] == "train"] + [r for r in rows if r["split"] == "val"]
    rng = random.Random(seed)
    train, val = [], []
    for category, (nt, nv) in ENTERPRISE_CATEGORY_QUOTAS.items():
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
        canonical = ALIASES.get(str(arg).lower(), str(arg).lower())
        if canonical in _OPTIMIZE_TIER2_CORPORA:
            raise YawRuntimeError("INVALID_INPUT", f"{canonical} is a held-out benchmark, not optimization data")
        bundle = resolve_dataset_bundle(arg, workspace_root=workspace_root)
        if bundle.dataset in _OPTIMIZE_TIER2_CORPORA:
            raise YawRuntimeError("INVALID_INPUT", f"{bundle.dataset} is a held-out benchmark, not optimization data")
        rows, document_rows = bundle.read_inputs()
        if bundle.dataset == "enterprise_rag_bench":
            cohort = select_enterprise_questions(rows)
            # Smaller optimization slices still respect the fixed cohort's
            # training/validation ownership instead of promoting training rows
            # into validation merely because max_queries is smaller than 250.
            rows = (
                cohort[:150][query_offset : query_offset + train_queries]
                + cohort[150:][query_offset : query_offset + max_queries - train_queries]
            )
            if len(rows) != max_queries:
                raise ValueError("Requested Enterprise slice exceeds its fixed 150/100 partitions")
        else:
            rows = rows[query_offset : query_offset + max_queries]
        if len(rows) <= train_queries:
            raise ValueError("Insufficient distinct validation examples in selected dataset slice")
        corpus_rows = [
            dict(row, idx=entry.document_id) for entry, row in zip(bundle.documents, document_rows, strict=True)
        ]
        by_id = {row["idx"]: document_text(row, bundle.title_key, bundle.body_key) for row in corpus_rows}
        gold = get_gold_docs(rows, bundle.dataset, corpus=corpus_rows, corpus_by_id=by_id)
        examples = []
        for row, references in zip(rows, gold, strict=True):
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
