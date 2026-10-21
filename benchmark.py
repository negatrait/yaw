"""Live Native YAW HippoRAG 2 vs. Dual-Target Baselines & StandardRAG (DPR) benchmark runner.

Tier 0 (sample):
  uv run --locked python benchmark.py --dataset sample --mode all
Tier 1 (calibration slice with pooled candidate passages):
  uv run --locked python benchmark.py --dataset musique --max-queries 10
  uv run --locked python benchmark.py --dataset 2wikimultihopqa --max-queries 10
  uv run --locked python benchmark.py --dataset hotpotqa --max-queries 10
"""

import argparse
import gc
import hashlib
import json
import logging
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import numpy as np

from scripts.evaluation import (
    QAExactMatch as ScriptQAExactMatch,
)
from scripts.evaluation import (
    QAF1Score as ScriptQAF1Score,
)
from scripts.evaluation import (
    RetrievalRecall as ScriptRetrievalRecall,
)
from yaw.core.logger import configure_logging
from yaw.daemon.graph.inference.embedding import (
    FACT_INSTRUCTION,
    GEMMA_TASK_MODES,
    MODEL,
    PASSAGE_INSTRUCTION,
    PROFILES,
    DSPyEmbeddingAdapter,
    create_dspy_embedder,
    resolve_embedding_profile,
)
from yaw.daemon.graph.inference.models import (
    BASE_URL,
    CHAT_MODELS,
    DEFAULT_EXTRACTION_CHAT,
    DEFAULT_QA_CHAT,
    chat_evidence,
    embedding_evidence,
    preflight,
)
from yaw.daemon.ingestion import sync_corpus_once

PROJECT_ROOT = Path(__file__).resolve().parent
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

DATASETS = (
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

ALIASES = {
    "2wiki": "2wikimultihopqa",
    "hotpot": "hotpotqa",
    "naturalquestions": "nq",
    "nq_rear": "nq",
    "narrative": "narrativeqa",
    "enterprise": "enterprise_rag_bench",
}

DATASET_REGISTRY = {
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

QA_TOP_K = 5
RETRIEVAL_TOP_K = 10
LINKING_TOP_K = 8
FACT_CANDIDATE_POOL_K = 50
FACT_FTS_POOL_K = 50
FACT_RRF_K = 60
FACT_FTS_WEIGHT = 0.2
FTS_WEIGHT = 0.2
PASSAGE_NODE_WEIGHT = 0.05
DAMPING = 0.5
PASSAGE_SEED_POOL_K = 200
FTS_SEED_POOL_K = 50

TEXT_STOPWORDS = {
    "a",
    "about",
    "after",
    "all",
    "also",
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
    "but",
    "by",
    "can",
    "come",
    "could",
    "day",
    "do",
    "even",
    "first",
    "for",
    "from",
    "give",
    "good",
    "have",
    "he",
    "her",
    "him",
    "his",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "just",
    "know",
    "like",
    "look",
    "make",
    "me",
    "most",
    "my",
    "new",
    "no",
    "not",
    "now",
    "of",
    "on",
    "one",
    "only",
    "or",
    "other",
    "our",
    "out",
    "over",
    "people",
    "see",
    "she",
    "so",
    "some",
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

# Target 1: Consolidated Historical Monkeypatched Upstream HippoRAG 2 Baselines (Qwen3-4B + EmbeddingGemma).
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

# Target 2: Reference numbers from HippoRAG 2 (arXiv:2502.14802v1, Tables 2, 3, 8 & 9).
PAPER_BASELINES = {
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


def clean_qa_answer(raw_ans: str) -> str:
    """Normalize extracted QA answer by stripping conversational prefixes and enclosing quotes."""
    ans = str(raw_ans or "").strip()
    if "Answer:" in ans:
        ans = ans.split("Answer:")[-1].strip()
    if (ans.startswith('"') and ans.endswith('"')) or (ans.startswith("'") and ans.endswith("'")):
        ans = ans[1:-1].strip()
    return ans


def recycle_lemonade_model(model_name: str) -> bool:
    """Safely bounce a local Lemonade model worker (e.g. Ryzen AI NPU) if frozen or between heavy runs."""
    lemonade_bin = shutil.which("lemonade")
    if not lemonade_bin:
        return False
    try:
        logging.info("Recycling Lemonade model '%s'...", model_name)
        is_ryzen = "ryzenai" in model_name.lower() or "mistral" in model_name.lower()
        if is_ryzen:
            subprocess.run(
                ["taskkill", "/F", "/IM", "ryzenai-server.exe"], capture_output=True, timeout=5
            )
            time.sleep(1.0)
        else:
            subprocess.run([lemonade_bin, "unload", model_name], capture_output=True, timeout=15)
            time.sleep(1.0)
        subprocess.run([lemonade_bin, "load", model_name], capture_output=True, timeout=45)
        time.sleep(2.0)
        return True
    except Exception as exc:
        logging.warning("recycle_lemonade_model(%s) failed: %s", model_name, exc)
        return False


def get_gold_docs(
    samples: list[dict],
    dataset_name: str | None = None,
    corpus: list[dict] | None = None,
) -> list[list[str]]:
    """Extract gold supporting passages using upstream main.py's exact formatting contract."""
    gold_docs = []
    canonical_dataset = ALIASES.get(dataset_name or "", dataset_name or "")
    separator = "" if canonical_dataset.startswith("hotpotqa") else " "
    for sample in samples:
        if "gold_docs" in sample and sample["gold_docs"]:
            gold_doc = [
                f"{item['title']}\n{item['text']}" if isinstance(item, dict) else str(item)
                for item in sample["gold_docs"]
            ]
        elif "supporting_facts" in sample:
            gold_titles = {item[0] for item in sample["supporting_facts"]}
            supporting_contexts = [item for item in sample["context"] if item[0] in gold_titles]
            gold_doc = [f"{item[0]}\n{separator.join(item[1])}" for item in supporting_contexts]
        elif "contexts" in sample:
            gold_doc = [
                f"{item['title']}\n{item['text']}"
                for item in sample["contexts"]
                if item.get("is_supporting", False)
            ]
        elif "paragraphs" in sample:
            paragraphs = [item for item in sample["paragraphs"] if item.get("is_supporting", True)]
            gold_doc = [
                f"{item['title']}\n{item['text'] if 'text' in item else item['paragraph_text']}"
                for item in paragraphs
            ]
        elif "context" in sample and isinstance(sample["context"], list):
            gold_doc = [f"{item[0]}\n{separator.join(item[1])}" for item in sample["context"]]
        elif isinstance(sample.get("context"), str):
            passages = [
                p.strip()
                for p in re.split(r"\n*### Passage \d+\n*", sample["context"])
                if p.strip()
            ]
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

            question = sample.get("question", "")
            q_keywords = {
                w
                for w in re.findall(r"\w+", question.lower())
                if len(w) >= 3 and w not in TEXT_STOPWORDS
            }
            a_keywords = {
                w
                for a in ans_set
                for w in re.findall(r"\w+", a.lower())
                if len(w) >= 3 and w not in TEXT_STOPWORDS
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
                    scored_passages.append(
                        ((exact_ans_matches, ans_kw_matches, q_kw_matches, -idx), p)
                    )

            if scored_passages:
                scored_passages.sort(key=lambda x: x[0], reverse=True)
                top_passages = [p for _, p in scored_passages[:2]]
            elif passages:
                top_passages = [passages[0]]
            else:
                top_passages = []

            gold_doc = [f"\n{p}" for p in top_passages]
        elif "document" in sample and isinstance(sample["document"], dict):
            doc = sample["document"]
            doc_id = doc.get("id")
            gold_doc = []
            if corpus and doc_id:
                doc_chunks = [c for c in corpus if c.get("idx", "").startswith(f"{doc_id}_")]
                answers = get_gold_answers([sample])[0]
                ans_lower = [a.lower().strip() for a in answers if a.strip()]
                a_keywords = {
                    w
                    for a in answers
                    for w in re.findall(r"\w+", a.lower())
                    if len(w) >= 3 and w not in TEXT_STOPWORDS
                }
                question = sample.get("question", "")
                q_keywords = {
                    w
                    for w in re.findall(r"\w+", question.lower())
                    if len(w) >= 3 and w not in TEXT_STOPWORDS
                }

                scored_chunks: list[tuple[tuple[int, int, int, int], str]] = []
                for idx, c in enumerate(doc_chunks):
                    c_text = c.get("text", "")
                    c_lower = c_text.lower()
                    c_words = set(re.findall(r"\w+", c_lower))
                    exact_ans_matches = sum(1 for a in ans_lower if a in c_lower)
                    ans_kw_matches = len(a_keywords & c_words)
                    q_kw_matches = len(q_keywords & c_words)

                    if exact_ans_matches > 0 or ans_kw_matches > 0:
                        formatted = f"{c['title']}\n{c_text}"
                        scored_chunks.append(
                            ((exact_ans_matches, ans_kw_matches, q_kw_matches, -idx), formatted)
                        )

                if scored_chunks:
                    scored_chunks.sort(key=lambda x: x[0], reverse=True)
                    gold_doc = [chunk for _, chunk in scored_chunks[:2]]
                elif doc_chunks:
                    gold_doc = [f"{doc_chunks[0]['title']}\n{doc_chunks[0]['text']}"]
            if not gold_doc:
                summary = doc.get("summary")
                if isinstance(summary, dict) and summary.get("text"):
                    gold_doc = [f"{summary.get('title', '')}\n{summary['text']}".strip()]
                elif isinstance(summary, str) and summary.strip():
                    gold_doc = [summary.strip()]
                elif doc.get("text"):
                    gold_doc = [doc["text"].strip()]
                else:
                    gold_doc = []
        else:
            raise ValueError(
                "Each sample needs supporting_facts, contexts, paragraphs, context, or document for retrieval evaluation"
            )
        gold_docs.append(list(dict.fromkeys(gold_doc)))
    return gold_docs


def get_gold_answers(samples: list[dict]) -> list[list[str]]:
    """Extract gold answer sets and aliases using upstream main.py's exact contract."""
    gold_answers = []
    for sample in samples:
        if "answers" in sample and sample["answers"] is not None:
            gold_answer = sample["answers"]
        elif "answer" in sample or "gold_ans" in sample:
            gold_answer = sample.get("answer", sample.get("gold_ans"))
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
        gold_answers.append(list(dict.fromkeys(answers)))
    return gold_answers


def build_benchmark_slice(
    dataset_name: str,
    samples: list[dict],
    corpus: list[dict],
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
    selected = samples[start:end]
    queries = [sample["question"] for sample in selected]
    canonical_dataset = ALIASES.get(dataset_name or "", dataset_name or "")
    gold_docs = get_gold_docs(selected, canonical_dataset, corpus=corpus)
    gold_answers = get_gold_answers(selected)

    if full_corpus or max_queries is None or canonical_dataset == "sample":
        docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]
    else:
        effective_index_queries = int(max_queries * 1.4) if index_queries is None else index_queries
        pooled_end = start + effective_index_queries
        pooled_samples = samples[start:pooled_end]

        docs = []
        separator = "" if canonical_dataset.startswith("hotpotqa") else " "

        if canonical_dataset in ("lveval", "narrativeqa"):
            # 1. Guarantee all gold docs for the evaluated queries are in docs
            for gold_list in gold_docs:
                docs.extend(gold_list)
            gold_unique = list(dict.fromkeys(docs))
            docs = list(gold_unique)

            # Target pool size: bounded padding for long-context benchmarks
            target_pool_size = max(
                len(gold_unique) + (effective_index_queries - max_queries) * 2,
                effective_index_queries * 2,
            )

            # 2. Gather candidate distractors from pooled samples/corpus
            candidate_pool: list[str] = []
            if canonical_dataset == "lveval":
                for sample in pooled_samples:
                    if isinstance(sample.get("context"), str):
                        passages = [
                            p.strip()
                            for p in re.split(r"\n*### Passage \d+\n*", sample["context"])
                            if p.strip()
                        ]
                        for p in passages:
                            candidate_pool.append(f"\n{p}")
                if corpus and len(candidate_pool) < target_pool_size:
                    for c in corpus:
                        candidate_pool.append(f"{c['title']}\n{c['text']}")
            else:  # narrativeqa
                for sample in pooled_samples:
                    doc = sample.get("document", {})
                    doc_id = doc.get("id") if isinstance(doc, dict) else None
                    if doc_id and corpus:
                        for c in corpus:
                            if c.get("idx", "").startswith(f"{doc_id}_"):
                                candidate_pool.append(f"{c['title']}\n{c['text']}")
                    summary = doc.get("summary") if isinstance(doc, dict) else None
                    if isinstance(summary, dict) and summary.get("text"):
                        candidate_pool.append(
                            f"{summary.get('title', '')}\n{summary['text']}".strip()
                        )
                if corpus and len(candidate_pool) < target_pool_size:
                    for c in corpus:
                        candidate_pool.append(f"{c['title']}\n{c['text']}")

            # 3. Add non-gold distractor passages up to target_pool_size
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
                            p.strip()
                            for p in re.split(r"\n*### Passage \d+\n*", sample["context"])
                            if p.strip()
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


def load_dataset_files(
    dataset_name: str,
    dataset_dir: Path | None = None,
    *,
    max_queries: int | None = None,
    include_corpus: bool = True,
) -> tuple[list[dict], list[dict]]:
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

    cache_dir = dataset_dir or (PROJECT_ROOT / "outputs" / "datasets")
    cache_dir.mkdir(parents=True, exist_ok=True)
    required_files = [q_file]
    if include_corpus:
        required_files.append(c_file)

    for filename in required_files:
        target = cache_dir / filename
        if not target.exists():
            urls = [f"{HF_BASE_URL}/{filename}"]
            if any(k in filename for k in ("sample", "2wiki", "hotpot", "musique")):
                urls.append(f"{UPSTREAM_DATASET_BASE_URL}/{filename}")
            downloaded = False
            last_err = None
            for url in urls:
                try:
                    logging.info("Downloading %s to %s", url, target)
                    req = urllib.request.Request(
                        url, headers={"User-Agent": "YAW-HippoRAG-DatasetDownloader/1.0"}
                    )
                    with urllib.request.urlopen(req) as resp:
                        target.write_bytes(resp.read())
                    downloaded = True
                    break
                except Exception as exc:
                    last_err = exc
                    if target.exists():
                        target.unlink()
            if not downloaded:
                raise RuntimeError(f"Could not download {filename} from any source: {last_err}")

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


def verify_embeddings(
    embedding: DSPyEmbeddingAdapter, queries: list[str], docs: list[str]
) -> list[dict[str, Any]]:
    """Five repeats of batch, individual and reversed inputs across all active roles."""
    probe_queries = queries[:3]
    probe_docs = docs[:3]
    probes = []
    roles: list[tuple[str, list[str], dict[str, Any]]] = [
        ("query", probe_queries, {"instruction": PASSAGE_INSTRUCTION}),
    ]
    if embedding.profile.query_instruction_mode == "distinct":
        roles.append(("fact_query", probe_queries, {"instruction": FACT_INSTRUCTION}))
    untitle_docs = [doc.split("\n", 1)[-1].replace("\n", " ") for doc in probe_docs]
    roles.append(("document", untitle_docs, {}))
    if embedding.profile.parse_document_title:
        roles.append(("titled_document", probe_docs, {}))
    for role, texts, options in roles:
        observations = []
        for _ in range(5):
            observations.extend(
                [
                    embedding.batch_encode(texts, norm=True, **options),
                    np.concatenate(
                        [embedding.batch_encode([text], norm=True, **options) for text in texts]
                    ),
                    embedding.batch_encode(texts[::-1], norm=True, **options)[::-1],
                ]
            )
        samples = np.stack(observations)
        cosine = np.einsum("rid,sid->irs", samples, samples)
        minimum = float(cosine.min())
        probes.append(
            {
                "role": role,
                "inputs": [
                    embedding.profile.format(
                        text, query=bool(options), instruction=options.get("instruction", "")
                    )
                    for text in texts
                ],
                "repeats": 5,
                "dimensions": embedding.profile.dimensions,
                "min_pairwise_cosine": minimum,
            }
        )
        if minimum < 0.999:
            raise RuntimeError(f"{role} embeddings are unstable: min cosine {minimum:.6f} < 0.999")
    return probes


def check_benchmark_integrity(
    records: list[dict],
    graph_info: dict,
    expected_doc_count: int,
    extraction_errors: list[str] | tuple[str, ...] = (),
    qa_errors: list[str] | tuple[str, ...] = (),
    include_qa: bool = True,
) -> list[str]:
    """Fail-fast system integrity checks on indexing, chat completions, and PPR activation."""
    errors = []
    if extraction_errors:
        errors.extend(f"extraction_llm: {err}" for err in extraction_errors)
    if include_qa and qa_errors:
        errors.extend(f"qa_llm: {err}" for err in qa_errors)
    if graph_info.get("num_passage_nodes") != expected_doc_count:
        errors.append(
            f"Index passage count mismatch: {graph_info.get('num_passage_nodes')} != {expected_doc_count}"
        )
    if not (graph_info.get("num_extracted_triples") or graph_info.get("num_fact_nodes")):
        errors.append("No triples extracted during OpenIE indexing")
    if include_qa:
        for idx, record in enumerate(records):
            if not str(record.get("answer", "")).strip():
                errors.append(f"Question {idx + 1}: empty HippoRAG answer")
            if (
                "dpr_answer" in record
                and record["dpr_answer"] is not None
                and str(record.get("dpr_answer", "")) == ""
                and record.get("dpr_qa_metrics") is not None
            ):
                errors.append(f"Question {idx + 1}: empty StandardRAG (DPR) answer")
    if len(records) > 1 and not any(record.get("graph_seeds") for record in records):
        errors.append(
            "All queries had empty graph_seeds (recognition memory / PPR never activated)"
        )
    return errors


def print_benchmark_summary(
    dataset: str,
    num_docs: int,
    records: list[dict],
    metrics: dict,
) -> None:
    print(f"\n{'=' * 72}")
    print(
        f"=== Benchmark Summary [NATIVE YAW ENGINE]: {dataset} ({len(records)} queries, {num_docs} passages) ==="
    )
    print(f"{'=' * 72}")
    print("Native YAW Retrieval:   ", metrics["hipporag"]["retrieval"])
    print("Standard DPR Retrieval: ", metrics["standard_dpr"]["retrieval"])
    if metrics["hipporag"]["qa"] is not None:
        print("Native YAW QA:          ", metrics["hipporag"]["qa"])
    print("Delta (YAW - DPR):      ", metrics["delta_vs_dpr"])
    print("Recognition Memory:     ", metrics["recognition_memory"])
    if "monkeypatched_hipporag_baseline" in metrics:
        print(
            "Target 1 (Monkeypatched HippoRAG 2 Local Baseline):",
            metrics["monkeypatched_hipporag_baseline"],
        )
    if (
        "paper_baselines" in metrics
        and "hipporag2_llama3.3_70b_nvembed_v2" in metrics["paper_baselines"]
    ):
        print(
            "Target 2 (Paper Reference arXiv:2502.14802):       ",
            metrics["paper_baselines"]["hipporag2_llama3.3_70b_nvembed_v2"],
        )
    print(f"{'=' * 72}\n", flush=True)


def run_yaw_benchmark(
    *,
    dataset: str,
    docs: list[str],
    queries: list[str],
    gold_docs: list[list[str]],
    gold_answers: list[list[str]],
    train_samples: list[dict],
    profile: Any,
    extraction_model: str,
    qa_model: str,
    include_qa: bool,
    enable_gbnf: bool,
    evolve: str,
    compiled_program: str | None,
    index_dir: Path,
    report: dict[str, Any],
    fact_candidate_pool_k: int = FACT_CANDIDATE_POOL_K,
    fact_fts_pool_k: int = FACT_FTS_POOL_K,
    fact_rrf_k: int = FACT_RRF_K,
    fact_fts_weight: float = FACT_FTS_WEIGHT,
    linking_top_k: int = LINKING_TOP_K,
    passage_node_weight: float = PASSAGE_NODE_WEIGHT,
    fts_weight: float = FTS_WEIGHT,
    damping: float = DAMPING,
    passage_seed_pool_k: int = PASSAGE_SEED_POOL_K,
    fts_seed_pool_k: int = FTS_SEED_POOL_K,
    graph_activation: str = "scaled_tanh",
    graph_activation_param: float | None = 5.0,
    graph_activation_renormalize: bool = False,
    graph_beta: float = 0.5,
    graph_dual_weight: float | None = 0.5,
) -> dict[str, Any]:
    """Execute live benchmark using native YawGraphEngine and DSPy HippoRAG 2 pipeline."""
    import dspy

    from yaw.daemon.graph.engine import YawGraphEngine
    from yaw.daemon.graph.inference import (
        HippoRAGProgram,
        create_dspy_lm,
        create_dspy_rlm_pair,
    )

    extraction_lm = create_dspy_lm(
        model_name=extraction_model,
        base_url=BASE_URL,
        role="extraction",
    )
    qa_lm = (
        create_dspy_lm(model_name=qa_model, base_url=BASE_URL, role="qa")
        if qa_model != extraction_model
        else extraction_lm
    )
    program = HippoRAGProgram(
        orchestrator_lm=qa_lm,
        rlm_sub_lm=extraction_lm,
        enable_gbnf=enable_gbnf,
        compiled_path=compiled_program,
    )
    dspy.settings.configure(lm=qa_lm)

    embed_client = create_dspy_embedder(profile, base_url=BASE_URL)
    embedder = DSPyEmbeddingAdapter(profile=profile, embedder=embed_client)

    index_dir.mkdir(parents=True, exist_ok=True)
    db_path = index_dir / "yaw.db"

    yaw_engine = YawGraphEngine(
        db_path=db_path,
        workspace_root=index_dir,
        program=program,
        embedder=embedder,
        retrieval_top_k=RETRIEVAL_TOP_K,
        linking_top_k=linking_top_k,
        fact_candidate_pool_k=fact_candidate_pool_k,
        fact_fts_pool_k=fact_fts_pool_k,
        fact_rrf_k=fact_rrf_k,
        fact_fts_weight=fact_fts_weight,
        passage_node_weight=passage_node_weight,
        fts_weight=fts_weight,
        damping=damping,
        passage_seed_pool_k=passage_seed_pool_k,
        fts_seed_pool_k=fts_seed_pool_k,
        activation=graph_activation,
        activation_param=graph_activation_param,
        activation_renormalize=graph_activation_renormalize,
        beta=graph_beta,
        dual_weight=graph_dual_weight,
    )
    try:
        index_start = perf_counter()
        num_docs = len(docs)
        corpus_dir = index_dir / "corpus"
        corpus_dir.mkdir(parents=True, exist_ok=True)
        slice_rel_path = f"corpus/{dataset}_slice.json"
        slice_json_path = index_dir / slice_rel_path
        slice_records: list[dict[str, Any]] = []
        for doc_str in docs:
            t_part = doc_str.split("\n", 1)[0] if "\n" in doc_str else ""
            b_part = doc_str.split("\n", 1)[1] if "\n" in doc_str else doc_str
            slice_records.append(
                {
                    "title": t_part,
                    "text": b_part,
                }
            )
        slice_json_path.write_text(
            json.dumps(slice_records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        delta = sync_corpus_once(
            corpus_root=index_dir,
            state_dir=index_dir / "coco_state",
            included_patterns=(slice_rel_path,),
            excluded_patterns=(".*", "*/.*", "__pycache__/*"),
            json_keys_whitelist=BENCHMARK_JSON_WHITELIST,
            json_keys_blacklist=BENCHMARK_JSON_BLACKLIST,
        )
        report["index_batches"] = (
            yaw_engine.index_chunks(
                delta.added_chunks,
                progress_callback=lambda msg: print(f"[Native YAW] [{dataset}] {msg}", flush=True),
            )
            if delta.added_chunks
            else int(yaw_engine.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        )
        del docs
        gc.collect()
        report["indexing_seconds"] = round(perf_counter() - index_start, 2)
        num_facts = int(yaw_engine.conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0])
        report["graph"] = {
            "num_passage_nodes": int(
                yaw_engine.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            ),
            "num_entity_nodes": int(
                yaw_engine.conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
            ),
            "num_fact_nodes": num_facts,
            "num_extracted_triples": num_facts,
            "num_edges": int(yaw_engine.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]),
        }

        # Disjoint Prompt Evolution
        if evolve != "none" and train_samples:
            print(
                f"[Prompt Evolution] Running {evolve} on {len(train_samples)} training queries...",
                flush=True,
            )
            train_golds = get_gold_answers(train_samples)
            train_gold_docs = get_gold_docs(train_samples, dataset)
            trainset = [
                dspy.Example(
                    question=s["question"],
                    gold_answers=train_golds[idx],
                    gold_docs=train_gold_docs[idx],
                ).with_inputs("question")
                for idx, s in enumerate(train_samples)
            ]
            valset: list[Any] | None = None
            train_q_set = {s["question"] for s in train_samples}
            disjoint_val = [
                (q, gold_answers[idx], gold_docs[idx])
                for idx, q in enumerate(queries)
                if q not in train_q_set
            ]
            if disjoint_val:
                valset = [
                    dspy.Example(
                        question=q,
                        gold_answers=ga,
                        gold_docs=gd,
                    ).with_inputs("question")
                    for q, ga, gd in disjoint_val
                ]
            ref_lm, ref_worker_lm = create_dspy_rlm_pair(
                orchestrator_model=DEFAULT_QA_CHAT,
                sub_model=extraction_model,
                base_url=BASE_URL,
                reflection=True,
            )
            yaw_engine.evolve_prompts(
                trainset,
                valset=valset,
                mode=cast(Any, evolve),
                save_path=compiled_program,
                reflection_lm=ref_lm,
                reflection_worker_lm=ref_worker_lm,
                auto="light",
            )

        eval_start = perf_counter()
        records: list[dict[str, Any]] = []
        all_retrieved_passages: list[list[str]] = []
        all_dpr_passages: list[list[str]] = []
        kept_facts_per_query: list[list[str]] = []
        qa_answers: list[str] = []
        timing_records: list[tuple[float, float]] = []
        retrieved_ids_list: list[list[str]] = []

        total_queries = len(queries)
        for q_idx, query in enumerate(queries, start=1):
            t_q0 = perf_counter()
            hits, kept_facts = yaw_engine.retrieve_with_seeds(query, top_k=RETRIEVAL_TOP_K)
            retrieval_elapsed = perf_counter() - t_q0

            dpr_hits = yaw_engine.retrieve_dpr(query, top_k=RETRIEVAL_TOP_K)

            retrieved_passages = [text for _, _, text in hits]
            dpr_passages = [text for _, _, text in dpr_hits]
            retrieved_ids = [p.chunk_id for p, _, _ in hits]
            all_retrieved_passages.append(retrieved_passages)
            all_dpr_passages.append(dpr_passages)
            retrieved_ids_list.append(retrieved_ids)
            kept_facts_per_query.append([f"({t[0]}, {t[1]}, {t[2]})" for t in kept_facts])

            qa_answer = ""
            qa_elapsed = 0.0
            if include_qa:
                t_qa0 = perf_counter()
                context = yaw_engine.format_qa_context(hits[:QA_TOP_K])
                pred = None
                for qa_attempt in range(3):
                    try:
                        pred = yaw_engine.program.answer_question(context=context, question=query)
                        break
                    except Exception as exc:
                        if qa_attempt < 2:
                            print(
                                f"[Native YAW] [{dataset}] QA query {q_idx} encountered {type(exc).__name__}: {exc}. "
                                f"Recycling {qa_model} and retrying (attempt {qa_attempt + 1}/3)...",
                                flush=True,
                            )
                            recycle_lemonade_model(qa_model)
                            try:
                                if hasattr(yaw_engine.program, "orchestrator_lm") and hasattr(
                                    yaw_engine.program.orchestrator_lm, "close"
                                ):
                                    yaw_engine.program.orchestrator_lm.close()
                            except Exception:
                                pass
                        else:
                            print(
                                f"[Native YAW] [{dataset}] QA query {q_idx} failed after 3 attempts: {exc}. "
                                "Falling back to empty prediction.",
                                flush=True,
                            )
                            pred = None
                qa_elapsed = perf_counter() - t_qa0
                qa_raw = str(getattr(pred, "answer", "")).strip() if pred is not None else ""
                qa_answer = clean_qa_answer(qa_raw)
                qa_answers.append(qa_answer)
            timing_records.append((retrieval_elapsed, qa_elapsed))
            if q_idx % 5 == 0 or q_idx == total_queries:
                print(
                    f"[Native YAW] [{dataset}] Query {q_idx}/{total_queries} completed (retrieval={retrieval_elapsed:.2f}s, qa={qa_elapsed:.2f}s)",
                    flush=True,
                )

        gold_docs_norm = [[d.strip() for d in gd] for gd in gold_docs]
        ret_docs_norm = [[p.strip() for p in rp] for rp in all_retrieved_passages]
        dpr_docs_norm = [[p.strip() for p in dp] for dp in all_dpr_passages]

        _, ret_per_query = ScriptRetrievalRecall().calculate_metric_scores(
            gold_docs=gold_docs_norm,
            retrieved_docs=ret_docs_norm,
            k_list=[1, 2, 5, 10],
        )
        _, dpr_per_query = ScriptRetrievalRecall().calculate_metric_scores(
            gold_docs=gold_docs_norm,
            retrieved_docs=dpr_docs_norm,
            k_list=[1, 2, 5, 10],
        )
        recalls_1 = [round(float(r["Recall@1"]), 4) for r in ret_per_query]
        recalls_2 = [round(float(r["Recall@2"]), 4) for r in ret_per_query]
        recalls_5 = [round(float(r["Recall@5"]), 4) for r in ret_per_query]
        recalls_10 = [round(float(r["Recall@10"]), 4) for r in ret_per_query]
        dpr_recalls_1 = [round(float(r["Recall@1"]), 4) for r in dpr_per_query]
        dpr_recalls_2 = [round(float(r["Recall@2"]), 4) for r in dpr_per_query]
        dpr_recalls_5 = [round(float(r["Recall@5"]), 4) for r in dpr_per_query]
        dpr_recalls_10 = [round(float(r["Recall@10"]), 4) for r in dpr_per_query]

        qa_metrics = None
        em_list: list[float] = []
        f1_list: list[float] = []
        if include_qa:
            em_overall, per_query_em_raw = ScriptQAExactMatch().calculate_metric_scores(
                gold_answers, qa_answers
            )
            f1_overall, per_query_f1_raw = ScriptQAF1Score().calculate_metric_scores(
                gold_answers, qa_answers
            )
            em_list = [float(item["ExactMatch"]) for item in per_query_em_raw]
            f1_list = [float(item["F1"]) for item in per_query_f1_raw]
            qa_metrics = {
                "ExactMatch": round(float(em_overall["ExactMatch"]), 4),
                "F1": round(float(f1_overall["F1"]), 4),
            }

        for q_idx, query in enumerate(queries):
            retrieval_elapsed, qa_elapsed = timing_records[q_idx]
            records.append(
                {
                    "query_index": q_idx,
                    "question": query,
                    "answer": qa_answers[q_idx] if include_qa else "",
                    "dpr_answer": "",
                    "gold_answers": gold_answers[q_idx],
                    "gold_docs": gold_docs[q_idx],
                    "qa_evidence": all_retrieved_passages[q_idx],
                    "dpr_evidence": all_dpr_passages[q_idx],
                    "graph_seeds": kept_facts_per_query[q_idx],
                    "recall": {
                        "@1": recalls_1[q_idx],
                        "@2": recalls_2[q_idx],
                        "@5": recalls_5[q_idx],
                        "@10": recalls_10[q_idx],
                    },
                    "dpr_recall": {
                        "@1": dpr_recalls_1[q_idx],
                        "@2": dpr_recalls_2[q_idx],
                        "@5": dpr_recalls_5[q_idx],
                        "@10": dpr_recalls_10[q_idx],
                    },
                    "retrieved_chunk_ids": retrieved_ids_list[q_idx],
                    "exact_match": em_list[q_idx] if include_qa else None,
                    "f1": f1_list[q_idx] if include_qa else None,
                    "retrieval_seconds": round(retrieval_elapsed, 4),
                    "qa_seconds": round(qa_elapsed, 4),
                }
            )

        report["evaluation_seconds"] = round(perf_counter() - eval_start, 2)
        seeds_active = sum(1 for seeds in kept_facts_per_query if seeds)
        fallback_count = len(records) - seeds_active
        avg_facts_kept = (
            round(float(np.mean([len(s) for s in kept_facts_per_query])), 4)
            if kept_facts_per_query
            else 0.0
        )
        hipporag_ret = {
            "Recall@1": round(float(np.mean(recalls_1)), 4) if recalls_1 else 0.0,
            "Recall@2": round(float(np.mean(recalls_2)), 4) if recalls_2 else 0.0,
            "Recall@5": round(float(np.mean(recalls_5)), 4) if recalls_5 else 0.0,
            "Recall@10": round(float(np.mean(recalls_10)), 4) if recalls_10 else 0.0,
        }
        dpr_ret = {
            "Recall@1": round(float(np.mean(dpr_recalls_1)), 4) if dpr_recalls_1 else 0.0,
            "Recall@2": round(float(np.mean(dpr_recalls_2)), 4) if dpr_recalls_2 else 0.0,
            "Recall@5": round(float(np.mean(dpr_recalls_5)), 4) if dpr_recalls_5 else 0.0,
            "Recall@10": round(float(np.mean(dpr_recalls_10)), 4) if dpr_recalls_10 else 0.0,
        }
        metrics: dict[str, Any] = {
            "hipporag": {
                "retrieval": hipporag_ret,
                "qa": qa_metrics,
            },
            "standard_dpr": {
                "retrieval": dpr_ret,
                "qa": None,
            },
            "delta_vs_dpr": {
                "Recall@1": round(hipporag_ret["Recall@1"] - dpr_ret["Recall@1"], 4),
                "Recall@2": round(hipporag_ret["Recall@2"] - dpr_ret["Recall@2"], 4),
                "Recall@5": round(hipporag_ret["Recall@5"] - dpr_ret["Recall@5"], 4),
                "Recall@10": round(hipporag_ret["Recall@10"] - dpr_ret["Recall@10"], 4),
            },
            "recognition_memory": {
                "queries_with_graph_seeds": seeds_active,
                "dpr_fallback_count": fallback_count,
                "dpr_fallback_rate": round(fallback_count / len(records), 4) if records else 0.0,
                "average_facts_kept": avg_facts_kept,
            },
        }
        if dataset in MONKEYPATCHED_HIPPORAG_BASELINES:
            mp = MONKEYPATCHED_HIPPORAG_BASELINES[dataset]
            metrics["monkeypatched_hipporag_baseline"] = mp
            metrics["delta_vs_monkeypatched_hipporag"] = {
                "Recall@1": round(hipporag_ret["Recall@1"] - float(mp["Recall@1"]), 4),
                "Recall@2": round(hipporag_ret["Recall@2"] - float(mp["Recall@2"]), 4),
                "Recall@5": round(hipporag_ret["Recall@5"] - float(mp["Recall@5"]), 4),
                "ExactMatch": (
                    round(float(qa_metrics["ExactMatch"]) - float(mp["ExactMatch"]), 4)
                    if qa_metrics is not None and mp.get("ExactMatch") is not None
                    else None
                ),
                "F1": (
                    round(float(qa_metrics["F1"]) - float(mp["F1"]), 4)
                    if qa_metrics is not None and mp.get("F1") is not None
                    else None
                ),
            }
        if dataset in PAPER_BASELINES:
            metrics["paper_baselines"] = PAPER_BASELINES[dataset]
        report["hyperparameters"] = {
            "linking_top_k": linking_top_k,
            "fact_candidate_pool_k": fact_candidate_pool_k,
            "fact_fts_pool_k": fact_fts_pool_k,
            "fact_rrf_k": fact_rrf_k,
            "fact_fts_weight": fact_fts_weight,
            "passage_node_weight": passage_node_weight,
            "fts_weight": fts_weight,
            "damping": damping,
            "passage_seed_pool_k": passage_seed_pool_k,
            "fts_seed_pool_k": fts_seed_pool_k,
            "graph_activation": graph_activation,
            "graph_activation_param": graph_activation_param,
            "graph_activation_renormalize": graph_activation_renormalize,
        }
        report["evaluation_metrics"] = metrics
        report["records"] = records
        print_benchmark_summary(dataset, num_docs, records, metrics)
    finally:
        yaw_engine.close()
    return report


def main(
    dataset: str = "sample",
    max_queries: int | None = None,
    index_queries: int | None = None,
    full_corpus: bool = False,
    mode: str = "eval",
    enable_gbnf: bool = False,
    query_offset: int = 0,
    evolve: str = "none",
    train_queries: int = 0,
    compiled_program: str | None = None,
    chat_model: str | None = None,
    embedding_model: str = MODEL,
    extraction_model: str = DEFAULT_EXTRACTION_CHAT,
    qa_model: str = DEFAULT_QA_CHAT,
    embedding_task_mode: str = "distinct-titled",
    fresh_index: bool = False,
    index_batch_size: int = 8,
    qa_batch_size: int = 1,
    linking_top_k: int = LINKING_TOP_K,
    fact_candidate_pool_k: int = FACT_CANDIDATE_POOL_K,
    fact_fts_pool_k: int = FACT_FTS_POOL_K,
    fact_rrf_k: int = FACT_RRF_K,
    fact_fts_weight: float = FACT_FTS_WEIGHT,
    passage_node_weight: float = PASSAGE_NODE_WEIGHT,
    fts_weight: float = FTS_WEIGHT,
    damping: float = DAMPING,
    passage_seed_pool_k: int = PASSAGE_SEED_POOL_K,
    fts_seed_pool_k: int = FTS_SEED_POOL_K,
    graph_activation: str = "scaled_tanh",
    graph_activation_param: float | None = 5.0,
    graph_activation_renormalize: bool = False,
    graph_beta: float = 0.5,
    graph_dual_weight: float | None = 0.5,
):
    dataset = ALIASES.get(dataset, dataset)
    if chat_model is not None:
        extraction_model = chat_model
        qa_model = chat_model
    include_qa = mode not in ("embeddings", "retrieval")
    profile = resolve_embedding_profile(embedding_model, embedding_task_mode)
    include_corpus = bool(
        full_corpus or max_queries is None or dataset in ("sample", "narrativeqa", "lveval")
    )

    total_needed = None
    if max_queries is not None:
        effective_index_q = index_queries if index_queries is not None else int(max_queries * 1.4)
        pool_needed = max(
            max_queries,
            effective_index_q * 2 if dataset == "popqa" else effective_index_q,
        )
        total_needed = pool_needed + max(query_offset, train_queries)

    samples, corpus = load_dataset_files(
        dataset,
        max_queries=total_needed,
        include_corpus=include_corpus,
    )

    train_samples: list[dict] = []
    eval_offset = query_offset
    if train_queries > 0 and evolve != "none":
        train_samples = samples[0:train_queries]
        if max_queries is not None and train_queries >= max_queries:
            eval_offset = max(query_offset, train_queries)

    docs, queries, gold_docs, gold_answers = build_benchmark_slice(
        dataset,
        samples,
        corpus,
        max_queries=max_queries,
        query_offset=eval_offset,
        full_corpus=full_corpus,
        index_queries=index_queries,
    )
    if train_samples:
        for gd in get_gold_docs(train_samples, dataset):
            docs.extend(gd)
        docs = list(dict.fromkeys(docs))

    del samples, corpus
    gc.collect()
    num_docs = len(docs)

    started = perf_counter()
    root = PROJECT_ROOT / "outputs" / "benchmarks"
    root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix=f"run-{dataset}-", dir=root))

    slice_hash = hashlib.sha256("\n".join(docs).encode("utf-8")).hexdigest()[:10]
    id_hash = hashlib.sha256(
        f"yaw|{profile.index_identity}|{extraction_model}".encode()
    ).hexdigest()[:10]
    if fresh_index:
        index_dir = run_dir / "index"
    else:
        index_dir = root / "indexes" / f"{dataset}_d{num_docs}_{slice_hash}_{id_hash}"

    configure_logging(log_file=run_dir / "run.log", max_lines=500)
    for name in (
        "httpx",
        "httpx2",
        "httpcore",
        "openai",
        "urllib3",
        "litellm",
        "LiteLLM",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)
    report: dict[str, Any] = {
        "status": "running",
        "dataset": dataset,
        "engine": "yaw",
        "enable_gbnf": enable_gbnf,
        "query_offset": eval_offset,
        "evolve": evolve,
        "train_queries": len(train_samples),
        "num_queries": len(queries),
        "index_queries": index_queries,
        "num_docs": num_docs,
        "full_corpus": full_corpus,
        "index_batch_size": index_batch_size,
        "qa_batch_size": qa_batch_size,
        "embedding_model": profile.model,
        "checkpoint": profile.checkpoint,
        "embedding_task_mode": embedding_task_mode,
        "query_instruction_mode": profile.query_instruction_mode,
        "parse_document_title": profile.parse_document_title,
        "extraction_model": extraction_model,
        "qa_model": qa_model if include_qa else None,
        "chat_model": (
            extraction_model
            if not include_qa or extraction_model == qa_model
            else f"{extraction_model} -> {qa_model}"
        ),
        "base_url": BASE_URL,
        "index_identity": profile.index_identity,
        "index_dir": str(index_dir),
        "run_dir": str(run_dir),
        "mode": mode,
        "linking_top_k": linking_top_k,
        "fact_candidate_pool_k": fact_candidate_pool_k,
        "fact_fts_pool_k": fact_fts_pool_k,
        "fact_rrf_k": fact_rrf_k,
        "fact_fts_weight": fact_fts_weight,
        "passage_node_weight": passage_node_weight,
        "fts_weight": fts_weight,
        "damping": damping,
        "passage_seed_pool_k": passage_seed_pool_k,
        "fts_seed_pool_k": fts_seed_pool_k,
        "graph_activation": graph_activation,
        "graph_activation_param": graph_activation_param,
        "graph_activation_renormalize": graph_activation_renormalize,
        "graph_beta": graph_beta,
        "graph_dual_weight": graph_dual_weight,
    }
    print(
        f"\n[Native YAW] Benchmark run: {run_dir} (dataset={dataset}, queries={len(queries)}, docs={num_docs})",
        flush=True,
    )
    try:
        report["model_registry"] = preflight(
            profile,
            None if mode == "embeddings" else extraction_model,
            None if not include_qa or qa_model == extraction_model else qa_model,
        )
        if mode in ("embeddings", "all"):
            embed_client = create_dspy_embedder(profile, base_url=BASE_URL)
            embedder = DSPyEmbeddingAdapter(profile=profile, embedder=embed_client)
            report["embedding_probes"] = verify_embeddings(embedder, queries, docs)
            report["server"] = embedding_evidence(profile)
            report["index_identity"] = report["server"]["resolved_index_identity"]
            if mode == "embeddings":
                report["status"] = "passed"
                return report
        report["chat_backends"] = {
            "extraction": chat_evidence(extraction_model),
            "qa": chat_evidence(qa_model) if include_qa else None,
        }
        report["status"] = "passed"
        return run_yaw_benchmark(
            dataset=dataset,
            docs=docs,
            queries=queries,
            gold_docs=gold_docs,
            gold_answers=gold_answers,
            train_samples=train_samples,
            profile=profile,
            extraction_model=extraction_model,
            qa_model=qa_model,
            include_qa=include_qa,
            enable_gbnf=enable_gbnf,
            evolve=evolve,
            compiled_program=compiled_program,
            index_dir=index_dir,
            report=report,
            fact_candidate_pool_k=fact_candidate_pool_k,
            fact_fts_pool_k=fact_fts_pool_k,
            fact_rrf_k=fact_rrf_k,
            fact_fts_weight=fact_fts_weight,
            linking_top_k=linking_top_k,
            passage_node_weight=passage_node_weight,
            fts_weight=fts_weight,
            damping=damping,
            passage_seed_pool_k=passage_seed_pool_k,
            fts_seed_pool_k=fts_seed_pool_k,
            graph_activation=graph_activation,
            graph_activation_param=graph_activation_param,
            graph_activation_renormalize=graph_activation_renormalize,
            graph_beta=graph_beta,
            graph_dual_weight=graph_dual_weight,
        )
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["elapsed_seconds"] = round(perf_counter() - started, 2)
        (run_dir / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"\n{report['status'].upper()}: {run_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[*DATASETS, *ALIASES], default="sample")
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Number of queries to evaluate (pools candidate passages up to --index-queries unless --full-corpus).",
    )
    parser.add_argument(
        "--index-queries",
        type=int,
        default=None,
        help="Number of queries from which to pool candidate passages for indexing (defaults to int(max_queries * 1.4)).",
    )
    parser.add_argument(
        "--full-corpus",
        action="store_true",
        help="Index the full dataset corpus even when --max-queries is set.",
    )
    parser.add_argument(
        "--mode",
        choices=("eval", "retrieval", "all", "embeddings"),
        default="eval",
        help="Run full benchmark ('eval'), retrieval-only ('retrieval'), include probes ('all'), or probes only.",
    )
    parser.add_argument(
        "--chat-model",
        choices=tuple(CHAT_MODELS),
        default=None,
        help="Override both extraction and QA chat models with a single model.",
    )
    parser.add_argument(
        "--extraction-model", choices=tuple(CHAT_MODELS), default=DEFAULT_EXTRACTION_CHAT
    )
    parser.add_argument("--qa-model", choices=tuple(CHAT_MODELS), default=DEFAULT_QA_CHAT)
    parser.add_argument("--embedding-model", choices=tuple(PROFILES), default=MODEL)
    parser.add_argument(
        "--embedding-task-mode", choices=GEMMA_TASK_MODES, default="distinct-titled"
    )
    parser.add_argument(
        "--fresh-index",
        action="store_true",
        help="Force indexing in a fresh disposable directory instead of reusing outputs/benchmarks/indexes.",
    )
    parser.add_argument(
        "--enable-gbnf",
        action="store_true",
        default=False,
        help="Enable pure-JSON GBNF grammar constraints for local llama.cpp GPU worker.",
    )
    parser.add_argument(
        "--query-offset",
        type=int,
        default=0,
        help="Starting index offset into dataset queries for evaluation.",
    )
    parser.add_argument(
        "--evolve",
        choices=("none", "miprov2", "gepa", "hybrid"),
        default="none",
        help="Prompt evolution strategy on training slice: 'none', 'miprov2', 'gepa', or 'hybrid'.",
    )
    parser.add_argument(
        "--train-queries",
        type=int,
        default=0,
        help="Number of queries from dataset head reserved for prompt evolution (disjoint from eval).",
    )
    parser.add_argument(
        "--compiled-program",
        type=str,
        default=None,
        help="Path to load or save compiled HippoRAGProgram state.",
    )
    parser.add_argument(
        "--index-batch-size",
        type=int,
        default=8,
        help="Number of passages per indexing micro-batch for incremental SSD checkpointing.",
    )
    parser.add_argument(
        "--qa-batch-size",
        type=int,
        default=1,
        help="Number of queries per retrieve() and qa() micro-batch (default: 1 for 1-by-1 processing).",
    )
    parser.add_argument(
        "--linking-top-k",
        type=int,
        default=LINKING_TOP_K,
        help="Top-N fact candidates passed to recognition memory filter (default: 8).",
    )
    parser.add_argument(
        "--fact-candidate-pool-k",
        type=int,
        default=FACT_CANDIDATE_POOL_K,
        help="Dense KNN fact candidate pool size before RRF merge (default: 50).",
    )
    parser.add_argument(
        "--fact-fts-pool-k",
        type=int,
        default=FACT_FTS_POOL_K,
        help="FTS5 BM25 fact candidate pool size before RRF merge (default: 50).",
    )
    parser.add_argument(
        "--fact-rrf-k",
        type=int,
        default=FACT_RRF_K,
        help="RRF smoothing constant k_rrf for reciprocal rank fusion (default: 60).",
    )
    parser.add_argument(
        "--fact-fts-weight",
        type=float,
        default=FACT_FTS_WEIGHT,
        help="Weight applied to FTS5 reciprocal ranks in fact RRF merge (default: 0.2).",
    )
    parser.add_argument(
        "--passage-node-weight",
        type=float,
        default=PASSAGE_NODE_WEIGHT,
        help="Weight applied to passage reset distribution in PPR (default: 0.05).",
    )
    parser.add_argument(
        "--fts-weight",
        type=float,
        default=FTS_WEIGHT,
        help="Weight applied to FTS5 lexical scores in passage reset weights (default: 0.2).",
    )
    parser.add_argument(
        "--damping",
        type=float,
        default=DAMPING,
        help="PPR damping restart probability parameter (default: 0.5).",
    )
    parser.add_argument(
        "--passage-seed-pool-k",
        type=int,
        default=PASSAGE_SEED_POOL_K,
        help="Dense KNN passage candidate pool size (default: 200).",
    )
    parser.add_argument(
        "--fts-seed-pool-k",
        type=int,
        default=FTS_SEED_POOL_K,
        help="FTS5 lexical passage candidate pool size (default: 50).",
    )
    parser.add_argument(
        "--graph-activation",
        choices=["linear", "scaled_tanh", "hard_tanh", "shifted_tanh", "threshold_relu"],
        default="scaled_tanh",
        help="Graph propagation activation (arXiv:2609.32929v1, default: scaled_tanh).",
    )
    parser.add_argument(
        "--graph-activation-param",
        type=float,
        default=5.0,
        help="Optional parameter for graph activation (scale for scaled_tanh, cap for hard_tanh, threshold for relu, default: 5.0).",
    )
    parser.add_argument(
        "--graph-activation-renormalize",
        action="store_true",
        help="Renormalize L1 mass to 1.0 after non-linear activation at each step (default: False).",
    )
    parser.add_argument(
        "--graph-beta",
        type=float,
        default=0.5,
        help="Normalization exponent beta for W = D^-beta A D^-(1-beta) (arXiv:2609.32929v1: 0.0=random walk, 0.5=symmetric GCN, default: 0.5).",
    )
    parser.add_argument(
        "--graph-dual-weight",
        type=float,
        default=0.5,
        help="Weight lambda in [0, 1] for dual combination (1-lambda)*y + lambda*z (arXiv:2609.32929v1, default: 0.5).",
    )
    args = parser.parse_args()
    main(
        dataset=args.dataset,
        max_queries=args.max_queries,
        index_queries=args.index_queries,
        full_corpus=args.full_corpus,
        mode=args.mode,
        enable_gbnf=args.enable_gbnf,
        query_offset=args.query_offset,
        evolve=args.evolve,
        train_queries=args.train_queries,
        compiled_program=args.compiled_program,
        chat_model=args.chat_model,
        embedding_model=args.embedding_model,
        extraction_model=args.extraction_model,
        qa_model=args.qa_model,
        embedding_task_mode=args.embedding_task_mode,
        fresh_index=args.fresh_index,
        index_batch_size=args.index_batch_size,
        qa_batch_size=args.qa_batch_size,
        linking_top_k=args.linking_top_k,
        fact_candidate_pool_k=args.fact_candidate_pool_k,
        fact_fts_pool_k=args.fact_fts_pool_k,
        fact_rrf_k=args.fact_rrf_k,
        fact_fts_weight=args.fact_fts_weight,
        passage_node_weight=args.passage_node_weight,
        fts_weight=args.fts_weight,
        damping=args.damping,
        passage_seed_pool_k=args.passage_seed_pool_k,
        fts_seed_pool_k=args.fts_seed_pool_k,
        graph_activation=args.graph_activation,
        graph_activation_param=args.graph_activation_param,
        graph_activation_renormalize=args.graph_activation_renormalize,
        graph_beta=args.graph_beta,
        graph_dual_weight=args.graph_dual_weight,
    )
