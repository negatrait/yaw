"""Centralized dataset registry, downloader, slice builder, and CocoIndex direct-source ingestion helpers."""

import json
import re
import shutil
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from yaw.core import (
    DEFAULT_EXCLUDED_PATTERNS,
    DEFAULT_INCLUDED_PATTERNS,
    ChunkPointer,
    YawRuntimeError,
    compute_mdhash_id,
    write_text_atomic,
)

from .corpus import chunk_source_text, sync_corpus_once

PINNED_COMMIT = "d5c8329422e0a0b834a15874545cb6a74b4f9b26"
HF_BASE_URL = "https://huggingface.co/datasets/osunlp/HippoRAG_2/resolve/main"
UPSTREAM_DATASET_BASE_URL = (
    f"https://raw.githubusercontent.com/OSU-NLP-Group/HippoRAG/{PINNED_COMMIT}/reproduce/dataset"
)

BENCHMARK_JSON_WHITELIST: tuple[str, ...] = (
    "title",
    "text",
    "paragraph_text",
    "contents",
    "context",
    "context.*",
    "contexts",
    "contexts.*",
    "paragraphs",
    "paragraphs.*",
    "document",
    "document.title",
    "document.text",
    "document.summary",
    "document.summary.*",
)
BENCHMARK_JSON_BLACKLIST: tuple[str, ...] = (
    "question*",
    "answer*",
    "answers*",
    "gold*",
    "reference*",
    "supporting_facts*",
    "is_supporting*",
    "evidences*",
    "obj*",
    "subj*",
    "prop*",
    "*_id",
    "id",
    "idx",
    "type",
    "level",
    "all_classes",
    "distractor",
    "length",
    "language",
    "dataset",
    "possible_answers*",
    "s_*",
    "o_*",
    "question_decomposition*",
    "answerable*",
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
    "enterprise_rag_bench": ("enterprise_rag_bench.json", "enterprise_rag_bench_corpus.json"),
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
_OPTIMIZE_KNOWN_DATASETS = frozenset(
    {
        *DATASETS,
        *DATASET_ALIASES.keys(),
        "enterprise_rag_bench",
        "enterprise",
    }
)
_OPTIMIZE_DATASET_ALIASES = {**DATASET_ALIASES, "enterprise": "enterprise_rag_bench"}


def resolve_dataset_name(name: str) -> str:
    """Resolve CLI/benchmark dataset aliases to canonical dataset keys."""
    cleaned = name.strip().lower()
    resolved = DATASET_ALIASES.get(cleaned, cleaned)
    if resolved not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. Valid datasets: {', '.join(DATASETS)} "
            f"(aliases: {', '.join(DATASET_ALIASES)})"
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


def load_dataset_files(
    dataset_name: str,
    dataset_dir: Path | None = None,
    *,
    max_queries: int | None = None,
    include_corpus: bool = True,
    hf_base_url: str | None = None,
    upstream_base_url: str | None = None,
    project_root: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load dataset and (when needed) corpus JSON files from local cache, downloading if absent."""
    if Path(dataset_name).name != dataset_name:
        raise ValueError("dataset_name must be a dataset name, not a path.")
    if max_queries is not None and max_queries < 1:
        raise ValueError("max_queries must be a positive integer when specified.")

    canonical_name = ALIASES.get(dataset_name, dataset_name)
    if canonical_name in DATASET_REGISTRY:
        q_file, c_file = DATASET_REGISTRY[canonical_name]
    else:
        q_file = f"{canonical_name}.json"
        c_file = f"{canonical_name}_corpus.json"

    root = (project_root or Path(__file__).resolve().parents[3]).resolve()
    cache_dir = dataset_dir or (root / "outputs" / "datasets")
    cache_dir.mkdir(parents=True, exist_ok=True)
    required_files = [q_file]
    if include_corpus:
        required_files.append(c_file)

    for filename in required_files:
        target = cache_dir / filename
        if not target.exists():
            download_dataset_file(
                filename,
                cache_dir,
                hf_base_url=hf_base_url,
                upstream_base_url=upstream_base_url,
            )

    with open(cache_dir / q_file, encoding="utf-8") as f:
        samples = json.load(f)
    if max_queries is not None:
        samples = samples[:max_queries]
    if include_corpus:
        with open(cache_dir / c_file, encoding="utf-8") as f:
            corpus = json.load(f)
    else:
        corpus = []
    return samples, corpus


def extract_gold_answers(sample: Mapping[str, Any], *, strict: bool = False) -> list[str]:
    """Extract gold answers and aliases from a single dataset sample."""
    if not strict:
        if "gold_answers" in sample:
            raw = sample["gold_answers"]
        elif "answer" in sample or "gold_ans" in sample:
            raw = sample.get("answer", sample.get("gold_ans"))
        elif "reference" in sample:
            raw = sample["reference"]
        elif "answers" in sample and sample["answers"] is not None:
            raw = sample["answers"]
        else:
            raw = []
        ans = [raw] if isinstance(raw, str) else [str(a) for a in (raw or ())]
        ans.extend(str(a) for a in sample.get("answer_aliases", ()))
        ans.extend(str(a) for a in sample.get("answer_facts", ()))
        return list(dict.fromkeys(ans))

    if "answers" in sample and sample["answers"] is not None:
        gold_answer: Any = sample["answers"]
    elif "answer" in sample or "gold_ans" in sample:
        gold_answer = sample.get("answer", sample.get("gold_ans"))
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
                except (json.JSONDecodeError, TypeError):
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
    dataset_name: str | None = None,
    corpus: Sequence[Mapping[str, Any]] | None = None,
) -> list[list[str]]:
    """Extract gold supporting passages using upstream main.py's exact formatting contract."""
    gold_docs: list[list[str]] = []
    canonical_dataset = ALIASES.get(dataset_name or "", dataset_name or "")
    separator = "" if canonical_dataset.startswith("hotpotqa") else " "
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
                "Each sample needs supporting_facts, contexts, paragraphs, context, or document for "
                "retrieval evaluation"
            )
        gold_docs.append(list(dict.fromkeys(gold_doc)))
    return gold_docs


def extract_sample_docs_and_gold(
    sample: Mapping[str, Any],
    dataset_name: str = "",
) -> tuple[list[str], list[str]]:
    """Extract `(all_docs, gold_docs)` from a single dataset item."""
    if "gold_docs" in sample and "paragraphs" not in sample and "context" not in sample and "contexts" not in sample:
        g_docs = [
            f"{d['title']}\n{d['text']}" if isinstance(d, Mapping) else str(d)
            for d in sample.get("gold_docs", ())
        ]
        return g_docs, g_docs

    cleaned_ds = dataset_name.strip().lower()
    canonical_dataset = ALIASES.get(cleaned_ds, cleaned_ds)
    separator = "" if canonical_dataset.startswith("hotpotqa") else " "
    all_docs: list[str] = []
    gold_docs: list[str] = []

    if "supporting_facts" in sample and "context" in sample:
        gold_titles = {sf[0] for sf in sample["supporting_facts"]}
        for title, sents in sample["context"]:
            passage = f"{title}\n{separator.join(sents)}"
            all_docs.append(passage)
            if title in gold_titles:
                gold_docs.append(passage)
    elif "contexts" in sample:
        for ctx in sample["contexts"]:
            passage = f"{ctx['title']}\n{ctx['text']}"
            all_docs.append(passage)
            if ctx.get("is_supporting", True):
                gold_docs.append(passage)
    elif "paragraphs" in sample:
        for p in sample["paragraphs"]:
            text = p["text"] if "text" in p else p.get("paragraph_text", "")
            passage = f"{p['title']}\n{text}"
            all_docs.append(passage)
            if p.get("is_supporting", True):
                gold_docs.append(passage)
    elif "context" in sample and isinstance(sample["context"], list):
        for title, sents in sample["context"]:
            passage = f"{title}\n{separator.join(sents)}"
            all_docs.append(passage)
            gold_docs.append(passage)
    elif isinstance(sample.get("context"), str):
        passages = [p.strip() for p in re.split(r"\n*### Passage \d+\n*", sample["context"]) if p.strip()]
        all_docs = [f"\n{p}" for p in passages]
        gold_docs = get_gold_docs([sample], canonical_dataset)[0]
    elif "document" in sample and isinstance(sample["document"], Mapping):
        gold_docs = get_gold_docs([sample], canonical_dataset)[0]
        all_docs = list(gold_docs)

    return list(dict.fromkeys(all_docs)), list(dict.fromkeys(gold_docs))


def build_benchmark_slice(
    dataset_name: str,
    samples: Sequence[Mapping[str, Any]],
    corpus: Sequence[Mapping[str, Any]],
    max_queries: int | None = None,
    query_offset: int = 0,
    full_corpus: bool = False,
    index_queries: int | None = None,
) -> tuple[list[str], list[str], list[list[str]], list[list[str]]]:
    """Build (docs, queries, gold_docs, gold_answers) for a full or N-query slice with distractor padding."""
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

    if full_corpus or max_queries is None or canonical_dataset == "sample":
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
    return list(dict.fromkeys(docs)), queries, gold_docs, gold_answers


def resolve_dataset_source_files(
    dataset_args: Sequence[str] | str | None,
    *,
    workspace_root: Path | None = None,
) -> list[Path]:
    """Resolve CLI `--dataset` arguments (names, aliases, or file paths) to concrete JSON file paths."""
    if isinstance(dataset_args, str):
        ds_list = [p.strip() for p in dataset_args.split(",") if p.strip()]
    elif dataset_args:
        ds_list = []
        for item in dataset_args:
            for sub in str(item).split(","):
                if sub.strip():
                    ds_list.append(sub.strip())
    else:
        ds_list = []

    resolved_paths: list[Path] = []
    for ds_clean in ds_list:
        candidate_path = Path(ds_clean)
        if not candidate_path.exists() and ds_clean in _OPTIMIZE_KNOWN_DATASETS:
            resolved_ds = _OPTIMIZE_DATASET_ALIASES.get(ds_clean, ds_clean)
            candidate_filenames = list(
                dict.fromkeys(
                    [
                        f"{resolved_ds}.json",
                        f"{ds_clean}.json",
                        *([DATASET_REGISTRY[resolved_ds][0]] if resolved_ds in DATASET_REGISTRY else []),
                    ]
                )
            )
            roots = [Path.cwd(), Path(__file__).resolve().parents[3]]
            if workspace_root is not None:
                roots.insert(0, Path(workspace_root))
            found = False
            for root in roots:
                for fname in candidate_filenames:
                    cached = root / "outputs" / "datasets" / fname
                    if cached.exists():
                        candidate_path = cached
                        found = True
                        break
                if found:
                    break
        if not candidate_path.exists():
            raise YawRuntimeError("INVALID_INPUT", f"Training dataset not found: {ds_clean}")
        resolved_paths.append(candidate_path.resolve())
    return resolved_paths


def load_dataset_splits(
    dataset_args: Sequence[str] | str | None,
    *,
    workspace_root: Path | None = None,
    max_queries: int = 20,
    train_queries: int = 7,
    query_offset: int = 0,
    example_factory: Callable[..., Any] | None = None,
) -> tuple[list[Any], list[Any] | None, list[str]]:
    """Load and split dataset records into (trainset, valset, dedup_docs) with multi-dataset round-robin support."""
    if max_queries < 1:
        raise YawRuntimeError("INVALID_INPUT", "max_queries must be >= 1.")

    def _make_example(*, inputs: tuple[str, ...] = ("question", "item_docs"), **kwargs: Any) -> Any:
        if example_factory is not None:
            return example_factory(_inputs=inputs, **kwargs)
        return dict(kwargs)

    if not dataset_args:
        return (
            [
                _make_example(
                    inputs=("question",),
                    question="When was the University of Southampton founded?",
                    gold_answers=["1862"],
                    gold_docs=[],
                )
            ],
            None,
            [],
        )

    if isinstance(dataset_args, str):
        ds_list = [p.strip() for p in dataset_args.split(",") if p.strip()]
    else:
        ds_list = []
        for item in dataset_args:
            for sub in str(item).split(","):
                if sub.strip():
                    ds_list.append(sub.strip())

    for ds_name in ds_list:
        ds_clean = ds_name.strip()
        ds_lower = Path(ds_clean).stem.lower()
        resolved_clean = _OPTIMIZE_DATASET_ALIASES.get(ds_clean.lower(), ds_clean.lower())
        resolved_lower = _OPTIMIZE_DATASET_ALIASES.get(ds_lower, ds_lower)
        if (
            ds_clean in _OPTIMIZE_TIER2_CORPORA
            or ds_lower in _OPTIMIZE_TIER2_CORPORA
            or resolved_clean in _OPTIMIZE_TIER2_CORPORA
            or resolved_lower in _OPTIMIZE_TIER2_CORPORA
        ) and max_queries > 50:
            raise YawRuntimeError(
                "INVALID_INPUT",
                f"Anti-overfitting policy forbids prompt evolution on >50 queries "
                f"of Tier 2 dataset '{ds_clean}' (got max_queries={max_queries}).",
            )

    paths = resolve_dataset_source_files(ds_list, workspace_root=workspace_root)
    train_per_ds: list[list[Any]] = []
    val_per_ds: list[list[Any]] = []
    all_corpus_docs: list[str] = []

    for candidate_path in paths:
        ds_lower = candidate_path.stem.lower()
        raw_items = json.loads(candidate_path.read_text(encoding="utf-8"))
        start = max(0, int(query_offset))
        sliced = list(raw_items)[start : start + max_queries]
        d_examples: list[Any] = []
        d_docs: list[str] = []
        for item in sliced:
            item_docs, gold_docs = extract_sample_docs_and_gold(item, ds_lower)
            d_docs.extend(item_docs)
            d_examples.append(
                _make_example(
                    inputs=("question", "item_docs"),
                    question=str(item["question"]),
                    gold_answers=extract_gold_answers(item, strict=False),
                    gold_docs=gold_docs,
                    item_docs=item_docs,
                )
            )
        all_corpus_docs.extend(d_docs)
        if 0 < train_queries < len(d_examples):
            train_per_ds.append(d_examples[:train_queries])
            val_per_ds.append(d_examples[train_queries:])
        else:
            train_per_ds.append(d_examples)
            val_per_ds.append(list(d_examples))

    trainset: list[Any] = []
    max_t = max((len(t) for t in train_per_ds), default=0)
    for i in range(max_t):
        for t in train_per_ds:
            if i < len(t):
                trainset.append(t[i])

    valset_list: list[Any] = []
    max_v = max((len(v) for v in val_per_ds), default=0)
    for i in range(max_v):
        for v in val_per_ds:
            if i < len(v):
                valset_list.append(v[i])

    valset = valset_list if valset_list else None
    return trainset, valset, list(dict.fromkeys(all_corpus_docs))


def _chunk_matches_doc(ptr: ChunkPointer, text: str, doc: str) -> bool:
    doc_stripped = doc.strip()
    txt_stripped = text.strip()
    if not doc_stripped or not txt_stripped:
        return False
    if txt_stripped == doc_stripped:
        return True
    if ptr.title:
        titled = f"{ptr.title.strip()}\n{txt_stripped}"
        if titled == doc_stripped:
            return True
        if doc_stripped.startswith(f"{ptr.title.strip()}\n") and txt_stripped in doc_stripped:
            return True
    return len(txt_stripped) >= 20 and txt_stripped in doc_stripped


def _has_non_synthesized_sources(root: Path) -> bool:
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        try:
            rel = p.relative_to(root).as_posix()
        except ValueError:
            continue
        if rel.startswith(".") or "/." in rel or rel.startswith("coco_state/"):
            continue
        if rel.startswith("corpus/chunk-") and rel.endswith(".md"):
            continue
        if rel in {"yaw.db", "yaw.db-wal", "yaw.db-shm", "yaw.db.lock", "status.json", "yaw.log"}:
            continue
        return True
    return False


def sync_source_files_for_docs(
    corpus_root: Path | str,
    state_dir: Path | str,
    *,
    included_patterns: Sequence[str] = (),
    excluded_patterns: Sequence[str] = (),
    json_keys_whitelist: Sequence[str] = BENCHMARK_JSON_WHITELIST,
    json_keys_blacklist: Sequence[str] = BENCHMARK_JSON_BLACKLIST,
    allowed_docs: Sequence[str] | None = None,
    chunk_size: int = 1200,
    chunk_overlap: int = 100,
    min_chunk_size: int = 50,
    source_files: Sequence[Path | str] | None = None,
) -> list[tuple[ChunkPointer, str]]:
    """Sync source files in `corpus_root` via CocoIndex and return `(ChunkPointer, text)` pairs."""
    root = Path(corpus_root).resolve()
    state = Path(state_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    state.mkdir(parents=True, exist_ok=True)

    effective_excluded = tuple(excluded_patterns) if excluded_patterns else DEFAULT_EXCLUDED_PATTERNS
    explicit_patterns: list[str] = list(included_patterns)
    if source_files:
        for sf in source_files:
            sf_path = Path(sf).resolve()
            if sf_path.exists() and sf_path.is_file():
                try:
                    rel = sf_path.relative_to(root).as_posix()
                    if rel not in explicit_patterns:
                        explicit_patterns.append(rel)
                except ValueError:
                    pass

    matched_chunks: list[tuple[ChunkPointer, str]] = []
    covered_docs: set[str] = set()

    if explicit_patterns or allowed_docs is None or _has_non_synthesized_sources(root):
        patterns_to_use = tuple(explicit_patterns) if explicit_patterns else DEFAULT_INCLUDED_PATTERNS
        scan_excluded = (
            effective_excluded
            if allowed_docs is None or explicit_patterns
            else (*effective_excluded, "corpus/chunk-*.md")
        )
        if (state / "coco_lmdb").exists():
            shutil.rmtree(state / "coco_lmdb", ignore_errors=True)
        delta = sync_corpus_once(
            root,
            state,
            included_patterns=patterns_to_use,
            excluded_patterns=scan_excluded,
            json_keys_whitelist=json_keys_whitelist,
            json_keys_blacklist=json_keys_blacklist,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
        )
        raw_chunks = list(delta.added_chunks)
        if allowed_docs is None:
            return raw_chunks

        dedup_allowed = [d for d in dict.fromkeys(allowed_docs) if d and d.strip()]
        for ptr, txt in raw_chunks:
            matched_any = False
            for d in dedup_allowed:
                if _chunk_matches_doc(ptr, txt, d):
                    covered_docs.add(d)
                    matched_any = True
            if matched_any:
                matched_chunks.append((ptr, txt))
        if len(covered_docs) == len(dedup_allowed) and matched_chunks:
            return matched_chunks

    dedup_docs = [
        d for d in dict.fromkeys(allowed_docs or ()) if d and d.strip() and d not in covered_docs
    ]
    if not dedup_docs:
        return matched_chunks

    corpus_dir = root / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    expected_files: dict[str, str] = {}
    for d in dedup_docs:
        cid = compute_mdhash_id(d, prefix="chunk-")
        fname = f"{cid}.md"
        expected_files[fname] = d
        fpath = corpus_dir / fname
        if not fpath.exists() or fpath.read_text(encoding="utf-8") != d:
            write_text_atomic(fpath, d)

    for existing in corpus_dir.glob("chunk-*.md"):
        if existing.name not in expected_files:
            existing.unlink(missing_ok=True)

    effective_min_chunk = min(min_chunk_size, 1)
    md_state = state / "md_pool"
    md_state.mkdir(parents=True, exist_ok=True)
    md_delta = sync_corpus_once(
        root,
        md_state,
        included_patterns=("corpus/chunk-*.md",),
        excluded_patterns=effective_excluded,
        json_keys_whitelist=json_keys_whitelist,
        json_keys_blacklist=json_keys_blacklist,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=effective_min_chunk,
    )

    by_rel_path: dict[str, list[tuple[ChunkPointer, str]]] = {}
    for ptr, txt in md_delta.added_chunks:
        by_rel_path.setdefault(ptr.rel_posix_path, []).append((ptr, txt))

    if any(f"corpus/{fname}" not in by_rel_path for fname in expected_files) and (md_state / "coco_lmdb").exists():
        shutil.rmtree(md_state / "coco_lmdb", ignore_errors=True)
        md_delta = sync_corpus_once(
            root,
            md_state,
            included_patterns=("corpus/chunk-*.md",),
            excluded_patterns=effective_excluded,
            json_keys_whitelist=json_keys_whitelist,
            json_keys_blacklist=json_keys_blacklist,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=effective_min_chunk,
        )
        by_rel_path.clear()
        for ptr, txt in md_delta.added_chunks:
            by_rel_path.setdefault(ptr.rel_posix_path, []).append((ptr, txt))

    for fname, doc_text in expected_files.items():
        rel_posix = f"corpus/{fname}"
        if rel_posix in by_rel_path:
            matched_chunks.extend(by_rel_path[rel_posix])
        else:
            reconstructed = chunk_source_text(
                doc_text,
                rel_posix,
                corpus_root=root,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                min_chunk_size=effective_min_chunk,
                json_keys_whitelist=json_keys_whitelist,
                json_keys_blacklist=json_keys_blacklist,
            )
            matched_chunks.extend(reconstructed)

    return matched_chunks


__all__ = [
    "ALIASES",
    "BENCHMARK_JSON_BLACKLIST",
    "BENCHMARK_JSON_WHITELIST",
    "DATASETS",
    "DATASET_ALIASES",
    "DATASET_DESCRIPTIONS",
    "DATASET_REGISTRY",
    "DEFAULT_PARITY_DATASETS",
    "HF_BASE_URL",
    "MONKEYPATCHED_HIPPORAG_BASELINES",
    "PAPER_BASELINES",
    "PINNED_COMMIT",
    "TEXT_STOPWORDS",
    "TIER2_CORPORA",
    "UPSTREAM_DATASET_BASE_URL",
    "build_benchmark_slice",
    "download_dataset_file",
    "extract_gold_answers",
    "extract_sample_docs_and_gold",
    "get_gold_answers",
    "get_gold_docs",
    "load_dataset_files",
    "load_dataset_splits",
    "resolve_dataset_name",
    "resolve_dataset_source_files",
    "sync_source_files_for_docs",
]
