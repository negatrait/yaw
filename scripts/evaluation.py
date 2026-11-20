"""Self-contained evaluation metrics, parity gate, and integrity checks for benchmarks."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import numpy as np

from yaw.datasets import compute_qa_em_f1, doc_matches_gold, score_qa_em_f1, score_retrieval_recalls

__all__ = [
    "QAExactMatch",
    "QAF1Score",
    "RetrievalRecall",
    "check_benchmark_integrity",
    "evaluate_benchmark_gate",
]


class RetrievalRecall:
    """Calculates Recall@k for retrieved passages against gold passages."""

    metric_name: str = "retrieval_recall"

    def __init__(self, global_config: Any = None) -> None:
        self.global_config = global_config

    @staticmethod
    def _doc_matches_gold(retrieved_doc: str, gold_doc: str) -> bool:
        return doc_matches_gold(retrieved_doc, gold_doc)

    def calculate_metric_scores(
        self,
        gold_docs: list[list[str]],
        retrieved_docs: list[list[str]],
        k_list: list[int] | tuple[int, ...] = (1, 5, 10, 20),
    ) -> tuple[dict[str, float], list[dict[str, float]]]:
        return score_retrieval_recalls(gold_docs, retrieved_docs, k_list=k_list)


class QAExactMatch:
    """Calculates Exact Match (EM) score across multiple reference answers."""

    metric_name: str = "qa_exact_match"

    def __init__(self, global_config: Any = None) -> None:
        self.global_config = global_config

    def calculate_metric_scores(
        self,
        gold_answers: list[list[str]],
        predicted_answers: list[str],
        aggregation_fn: Callable[[list[float]], float] = np.max,
    ) -> tuple[dict[str, float], list[dict[str, float]]]:
        if aggregation_fn is np.max:
            pooled, per_ex = score_qa_em_f1(gold_answers, predicted_answers)
            return {"ExactMatch": pooled["ExactMatch"]}, [{"ExactMatch": row["ExactMatch"]} for row in per_ex]

        if len(gold_answers) != len(predicted_answers):
            raise ValueError("gold_answers and predicted_answers must have the same length.")

        example_eval_results: list[dict[str, float]] = []
        total_em = 0.0

        for gold_list, predicted in zip(gold_answers, predicted_answers, strict=False):
            em_scores = [compute_qa_em_f1(predicted, [gold])[0] for gold in gold_list]
            aggregated_em = float(aggregation_fn(em_scores)) if em_scores else 0.0
            example_eval_results.append({"ExactMatch": aggregated_em})
            total_em += aggregated_em

        avg_em = total_em / len(gold_answers) if gold_answers else 0.0
        pooled_eval_results = {"ExactMatch": avg_em}
        return pooled_eval_results, example_eval_results


class QAF1Score:
    """Calculates token-level F1 score across multiple reference answers."""

    metric_name: str = "qa_f1_score"

    def __init__(self, global_config: Any = None) -> None:
        self.global_config = global_config

    def calculate_metric_scores(
        self,
        gold_answers: list[list[str]],
        predicted_answers: list[str],
        aggregation_fn: Callable[[list[float]], float] = np.max,
    ) -> tuple[dict[str, float], list[dict[str, float]]]:
        if aggregation_fn is np.max:
            pooled, per_ex = score_qa_em_f1(gold_answers, predicted_answers)
            return {"F1": pooled["F1"]}, [{"F1": row["F1"]} for row in per_ex]

        if len(gold_answers) != len(predicted_answers):
            raise ValueError("gold_answers and predicted_answers must have the same length.")

        example_eval_results: list[dict[str, float]] = []
        total_f1 = 0.0

        for gold_list, predicted in zip(gold_answers, predicted_answers, strict=False):
            f1_scores = [compute_qa_em_f1(predicted, [gold])[1] for gold in gold_list]
            aggregated_f1 = float(aggregation_fn(f1_scores)) if f1_scores else 0.0
            example_eval_results.append({"F1": aggregated_f1})
            total_f1 += aggregated_f1

        avg_f1 = total_f1 / len(gold_answers) if gold_answers else 0.0
        pooled_eval_results = {"F1": avg_f1}
        return pooled_eval_results, example_eval_results


def _compute_metric_confidence(yaw_val: float, base_val: float, n: int) -> dict[str, Any]:
    """Compute binomial normal-approximation standard error, z-score, and parity/difference confidence."""
    n_eff = max(1, int(n))
    delta_k = float(yaw_val) - float(base_val)
    p_k = min(1.0, max(0.0, float(yaw_val)))
    p0_k = min(1.0, max(0.0, float(base_val)))
    se_k = math.sqrt(max((p_k * (1.0 - p_k) + p0_k * (1.0 - p0_k)) / n_eff, 1e-9))
    z_k = 0.0 if abs(delta_k) < 1e-12 else delta_k / se_k
    c_k = round(0.5 * (1.0 + math.erf(z_k / math.sqrt(2.0))), 4)
    c_diff_k = round(math.erf(abs(z_k) / math.sqrt(2.0)), 4)
    ci_95 = [round(delta_k - 1.96 * se_k, 4), round(delta_k + 1.96 * se_k, 4)]
    return {
        "n": n_eff,
        "delta": round(delta_k, 4),
        "se": round(se_k, 4),
        "z": round(z_k, 4),
        "confidence": c_k,
        "parity_confidence": c_k,
        "difference_confidence": c_diff_k,
        "diff_confidence": c_diff_k,
        "ci_95": ci_95,
    }


def evaluate_benchmark_gate(
    dataset_reports: Mapping[str, Mapping[str, Any]],
    gate_spec: Mapping[str, Any],
    *,
    mode: Literal["eval", "retrieval"] = "eval",
    target: str | None = None,
) -> dict[str, Any]:
    """Evaluate per-dataset benchmark reports against a baselines.json specification."""
    effective_mode = str(mode or gate_spec.get("mode", "eval")).strip().lower()
    tolerance = float(gate_spec.get("per_dataset_tolerance", gate_spec.get("tolerance", 0.0)))
    primary_target = str(target or gate_spec.get("primary_target", "monkeypatched_hipporag"))
    spec_datasets = gate_spec.get("baselines", gate_spec.get("datasets", {}))
    if not isinstance(spec_datasets, Mapping):
        spec_datasets = {}
    targets_map = gate_spec.get("targets", {})
    target_datasets: Mapping[str, Any] = {}
    if isinstance(targets_map, Mapping):
        t_entry = targets_map.get(primary_target, {})
        if isinstance(t_entry, Mapping) and isinstance(t_entry.get("datasets"), Mapping):
            target_datasets = t_entry["datasets"]

    spec_datasets_lower = {str(k).strip().lower(): v for k, v in spec_datasets.items()}
    target_datasets_lower = {str(k).strip().lower(): v for k, v in target_datasets.items()}

    dataset_evaluations: dict[str, dict[str, Any]] = {}
    sample_counts: dict[str, int] = {}
    optional_qa_values: dict[str, dict[str, list[float | None]]] = {
        "yaw_hipporag": {"ExactMatch": [], "F1": []},
        "monkeypatched_hipporag": {"ExactMatch": [], "F1": []},
        "delta_yaw_vs_hipporag": {"ExactMatch": [], "F1": []},
    }
    for raw_name, rep in dataset_reports.items():
        if not isinstance(rep, Mapping):
            continue
        raw_stripped = str(raw_name).strip()
        cleaned = raw_stripped.lower()
        canonical = (
            cleaned
            if (cleaned in spec_datasets or cleaned in target_datasets or raw_stripped == cleaned)
            else raw_stripped
        )

        eval_metrics = rep.get("evaluation_metrics", {})
        if isinstance(eval_metrics, Mapping) and "hipporag" in eval_metrics:
            hippo_block = eval_metrics.get("hipporag", {})
            ret_block = hippo_block.get("retrieval", {}) if isinstance(hippo_block, Mapping) else {}
            raw_qa = hippo_block.get("qa") if isinstance(hippo_block, Mapping) else None
            qa_block = raw_qa if isinstance(raw_qa, Mapping) else {}
            dpr_block = eval_metrics.get("standard_dpr", {})
            dpr_ret = dpr_block.get("retrieval", {}) if isinstance(dpr_block, Mapping) else {}
            rec_mem = eval_metrics.get("recognition_memory", {}) if isinstance(eval_metrics, Mapping) else {}
        elif "yaw_hipporag" in rep and isinstance(rep["yaw_hipporag"], Mapping):
            yh = rep["yaw_hipporag"]
            ret_block = yh
            qa_block = yh
            dpr_ret = rep.get("yaw_dpr", {}) if isinstance(rep.get("yaw_dpr"), Mapping) else {}
            rec_mem = rep.get("recognition_memory", {}) if isinstance(rep.get("recognition_memory"), Mapping) else {}
        elif "retrieval" in rep and isinstance(rep["retrieval"], Mapping):
            ret_block = rep["retrieval"]
            raw_qa = rep.get("qa")
            qa_block = raw_qa if isinstance(raw_qa, Mapping) else {}
            dpr_ret = rep.get("standard_dpr", {}) if isinstance(rep.get("standard_dpr"), Mapping) else {}
            rec_mem = rep.get("recognition_memory", {}) if isinstance(rep.get("recognition_memory"), Mapping) else {}
        else:
            ret_block = rep
            qa_block = rep
            dpr_ret = rep.get("dpr", {}) if isinstance(rep.get("dpr"), Mapping) else {}
            rec_mem = {}

        yaw_em_raw = qa_block.get("ExactMatch") if isinstance(qa_block, Mapping) else None
        yaw_f1_raw = qa_block.get("F1") if isinstance(qa_block, Mapping) else None
        yaw_em_opt = round(float(yaw_em_raw), 4) if yaw_em_raw is not None else None
        yaw_f1_opt = round(float(yaw_f1_raw), 4) if yaw_f1_raw is not None else None
        yaw_r1 = (
            round(float(rv), 4)
            if isinstance(ret_block, Mapping) and (rv := ret_block.get("Recall@1")) is not None
            else None
        )
        yaw_r2 = (
            round(float(rv), 4)
            if isinstance(ret_block, Mapping) and (rv := ret_block.get("Recall@2")) is not None
            else None
        )
        yaw_r5 = (
            round(float(rv), 4)
            if isinstance(ret_block, Mapping) and (rv := ret_block.get("Recall@5")) is not None
            else None
        )
        yaw_metrics: dict[str, float | None] = {
            "Recall@1": yaw_r1,
            "Recall@2": yaw_r2,
            "Recall@5": yaw_r5,
            "ExactMatch": yaw_em_opt,
            "F1": yaw_f1_opt,
        }
        dpr_metrics = {
            "Recall@1": round(float(dpr_ret.get("Recall@1", 0.0) or 0.0), 4),
            "Recall@2": round(float(dpr_ret.get("Recall@2", 0.0) or 0.0), 4),
            "Recall@5": round(float(dpr_ret.get("Recall@5", 0.0) or 0.0), 4),
        }

        ds_spec = (
            spec_datasets.get(canonical)
            or spec_datasets.get(cleaned)
            or spec_datasets.get(raw_stripped)
            or spec_datasets_lower.get(canonical.lower())
            or spec_datasets_lower.get(cleaned)
            or target_datasets.get(canonical)
            or target_datasets.get(cleaned)
            or target_datasets.get(raw_stripped)
            or target_datasets_lower.get(canonical.lower())
            or target_datasets_lower.get(cleaned)
            or {}
        )
        if not isinstance(ds_spec, Mapping):
            ds_spec = {}

        base_block = ds_spec.get(primary_target, ds_spec.get("monkeypatched_hipporag", ds_spec))
        if not isinstance(base_block, Mapping):
            base_block = {}
        paper_block = ds_spec.get("hipporag2_llama3.3_70b_nvembed_v2", ds_spec.get("paper_70b", {}))
        if not isinstance(paper_block, Mapping):
            paper_block = {}
        elif isinstance(paper_block.get("hipporag2_llama3.3_70b_nvembed_v2"), Mapping):
            paper_block = paper_block["hipporag2_llama3.3_70b_nvembed_v2"]

        base_em_raw = base_block.get("ExactMatch")
        base_f1_raw = base_block.get("F1")
        base_em_opt = round(float(base_em_raw), 4) if base_em_raw is not None else None
        base_f1_opt = round(float(base_f1_raw), 4) if base_f1_raw is not None else None
        hippo_metrics: dict[str, float] = {
            "Recall@1": round(float(base_block.get("Recall@1", 0.0) or 0.0), 4),
            "Recall@2": round(float(base_block.get("Recall@2", 0.0) or 0.0), 4),
            "Recall@5": round(float(base_block.get("Recall@5", 0.0) or 0.0), 4),
            "ExactMatch": base_em_opt if base_em_opt is not None else 0.0,
            "F1": base_f1_opt if base_f1_opt is not None else 0.0,
        }
        paper_metrics: dict[str, float | None] = {
            k: (round(float(paper_block[k]), 4) if paper_block.get(k) is not None else None)
            for k in ("Recall@2", "Recall@5", "ExactMatch", "F1")
        }

        delta_vs_hippo: dict[str, float | None] = {
            "Recall@1": (
                round(float(yv) - hippo_metrics["Recall@1"], 4) if (yv := yaw_metrics["Recall@1"]) is not None else None
            ),
            "Recall@2": (
                round(float(yv) - hippo_metrics["Recall@2"], 4) if (yv := yaw_metrics["Recall@2"]) is not None else None
            ),
            "Recall@5": (
                round(float(yv) - hippo_metrics["Recall@5"], 4) if (yv := yaw_metrics["Recall@5"]) is not None else None
            ),
            "ExactMatch": (
                round(yaw_em_opt - base_em_opt, 4) if yaw_em_opt is not None and base_em_opt is not None else None
            ),
            "F1": round(yaw_f1_opt - base_f1_opt, 4) if yaw_f1_opt is not None and base_f1_opt is not None else None,
        }
        delta_vs_paper: dict[str, float | None] = {
            "Recall@2": (
                round(float(yv) - float(pv), 4)
                if (yv := yaw_metrics["Recall@2"]) is not None and (pv := paper_metrics["Recall@2"]) is not None
                else None
            ),
            "Recall@5": (
                round(float(yv) - float(pv), 4)
                if (yv := yaw_metrics["Recall@5"]) is not None and (pv := paper_metrics["Recall@5"]) is not None
                else None
            ),
            "ExactMatch": (
                round(yaw_em_opt - float(pv), 4)
                if yaw_em_opt is not None and (pv := paper_metrics["ExactMatch"]) is not None
                else None
            ),
            "F1": (
                round(yaw_f1_opt - float(pv), 4)
                if yaw_f1_opt is not None and (pv := paper_metrics["F1"]) is not None
                else None
            ),
        }
        delta_vs_dpr: dict[str, float | None] = {
            k: (round(float(yv) - dpr_metrics[k], 4) if (yv := yaw_metrics[k]) is not None else None)
            for k in ("Recall@1", "Recall@2", "Recall@5")
        }

        optional_qa_values["yaw_hipporag"]["ExactMatch"].append(yaw_em_opt)
        optional_qa_values["yaw_hipporag"]["F1"].append(yaw_f1_opt)
        optional_qa_values["monkeypatched_hipporag"]["ExactMatch"].append(base_em_opt)
        optional_qa_values["monkeypatched_hipporag"]["F1"].append(base_f1_opt)
        optional_qa_values["delta_yaw_vs_hipporag"]["ExactMatch"].append(delta_vs_hippo["ExactMatch"])
        optional_qa_values["delta_yaw_vs_hipporag"]["F1"].append(delta_vs_hippo["F1"])

        failures: list[str] = []
        integrity = rep.get("integrity", {})
        if isinstance(integrity, Mapping) and not integrity.get("passed", True):
            int_errors = integrity.get("errors", [])
            failures.append(f"Integrity gate failed: {int_errors}")

        if (d_r5 := delta_vs_hippo["Recall@5"]) is not None and d_r5 < tolerance - 1e-9:
            failures.append(f"Recall@5 delta vs historical HippoRAG ({d_r5:+.4f}) < {tolerance:+.2f}")
        if effective_mode == "eval":
            if base_f1_opt is not None:
                if yaw_f1_opt is None:
                    failures.append("Missing QA F1 metric in eval mode")
                elif (d_f1 := delta_vs_hippo["F1"]) is not None and d_f1 < tolerance - 1e-9:
                    failures.append(f"QA F1 delta vs historical HippoRAG ({d_f1:+.4f}) < {tolerance:+.2f}")
            if base_em_opt is not None:
                if yaw_em_opt is None:
                    failures.append("Missing QA ExactMatch metric in eval mode")
                elif (d_em := delta_vs_hippo["ExactMatch"]) is not None and d_em < tolerance - 1e-9:
                    failures.append(f"ExactMatch delta vs historical HippoRAG ({d_em:+.4f}) < {tolerance:+.2f}")

        raw_q = rep.get("num_queries") or rep.get("queries_count")
        if raw_q is None:
            q_field = rep.get("queries")
            if isinstance(q_field, Sequence) and not isinstance(q_field, (str, bytes)):
                raw_q = len(q_field)
            elif isinstance(q_field, (int, float)):
                raw_q = int(q_field)
            elif isinstance(rep.get("records"), Sequence) and not isinstance(rep.get("records"), (str, bytes)):
                raw_q = len(rep["records"])
        queries_count = int(raw_q if raw_q is not None else gate_spec.get("queries_per_dataset", 0))
        n_queries = max(1, int(raw_q or gate_spec.get("queries_per_dataset", 25) or 25))
        sample_counts[canonical] = n_queries

        graph_block = rep.get("graph", {})
        raw_docs = rep.get("docs", rep.get("num_docs", 0))
        docs_fallback = (
            len(raw_docs)
            if isinstance(raw_docs, Sequence) and not isinstance(raw_docs, (str, bytes))
            else int(raw_docs or 0)
        )
        passage_nodes = (
            int(graph_block.get("num_passage_nodes", docs_fallback) or docs_fallback)
            if isinstance(graph_block, Mapping)
            else docs_fallback
        )

        ds_confidence = {
            k: _compute_metric_confidence(float(yaw_metrics[k] or 0.0), hippo_metrics[k], n_queries)
            for k in ("Recall@5", "ExactMatch", "F1")
        }

        dataset_evaluations[canonical] = {
            "dataset": canonical,
            "queries": queries_count,
            "docs": passage_nodes,
            "yaw_hipporag": yaw_metrics,
            "monkeypatched_hipporag": hippo_metrics,
            "paper_70b": paper_metrics,
            "yaw_dpr": dpr_metrics,
            "recognition_memory": dict(rec_mem) if isinstance(rec_mem, Mapping) else {},
            "delta_yaw_vs_hipporag": delta_vs_hippo,
            "delta_yaw_vs_paper": delta_vs_paper,
            "delta_yaw_vs_dpr": delta_vs_dpr,
            "confidence": ds_confidence,
            "gate_passed": len(failures) == 0,
            "gate_status": "PASS" if len(failures) == 0 else "FAIL",
            "failure_reasons": failures,
        }

    completed = list(dataset_evaluations.values())

    def _mean(vals: Sequence[float | None]) -> float:
        valid = [float(v) for v in vals if v is not None]
        return round(sum(valid) / len(valid), 4) if valid else 0.0

    macro_means: dict[str, Any] = {
        "yaw_hipporag": {
            "Recall@1": _mean([d["yaw_hipporag"]["Recall@1"] for d in completed]),
            "Recall@2": _mean([d["yaw_hipporag"]["Recall@2"] for d in completed]),
            "Recall@5": _mean([d["yaw_hipporag"]["Recall@5"] for d in completed]),
            "ExactMatch": _mean(optional_qa_values["yaw_hipporag"]["ExactMatch"]),
            "F1": _mean(optional_qa_values["yaw_hipporag"]["F1"]),
        },
        "monkeypatched_hipporag": {
            "Recall@1": _mean([d["monkeypatched_hipporag"]["Recall@1"] for d in completed]),
            "Recall@2": _mean([d["monkeypatched_hipporag"]["Recall@2"] for d in completed]),
            "Recall@5": _mean([d["monkeypatched_hipporag"]["Recall@5"] for d in completed]),
            "ExactMatch": _mean(optional_qa_values["monkeypatched_hipporag"]["ExactMatch"]),
            "F1": _mean(optional_qa_values["monkeypatched_hipporag"]["F1"]),
        },
        "paper_70b": {
            k: _mean([d["paper_70b"][k] for d in completed]) for k in ("Recall@2", "Recall@5", "ExactMatch", "F1")
        },
        "yaw_dpr": {k: _mean([d["yaw_dpr"][k] for d in completed]) for k in ("Recall@1", "Recall@2", "Recall@5")},
        "delta_yaw_vs_hipporag": {
            "Recall@1": _mean([d["delta_yaw_vs_hipporag"]["Recall@1"] for d in completed]),
            "Recall@2": _mean([d["delta_yaw_vs_hipporag"]["Recall@2"] for d in completed]),
            "Recall@5": _mean([d["delta_yaw_vs_hipporag"]["Recall@5"] for d in completed]),
            "ExactMatch": _mean(optional_qa_values["delta_yaw_vs_hipporag"]["ExactMatch"]),
            "F1": _mean(optional_qa_values["delta_yaw_vs_hipporag"]["F1"]),
        },
        "delta_yaw_vs_paper": {
            k: _mean([d["delta_yaw_vs_paper"][k] for d in completed])
            for k in ("Recall@2", "Recall@5", "ExactMatch", "F1")
        },
        "delta_yaw_vs_dpr": {
            k: _mean([d["delta_yaw_vs_dpr"][k] for d in completed]) for k in ("Recall@1", "Recall@2", "Recall@5")
        },
    }

    total_queries = sum(sample_counts.values())
    pooled_confidence = (
        {
            k: _compute_metric_confidence(
                float(macro_means["yaw_hipporag"][k]),
                float(macro_means["monkeypatched_hipporag"][k]),
                total_queries,
            )
            for k in ("Recall@5", "ExactMatch", "F1")
        }
        if completed
        else {}
    )
    macro_means["confidence"] = pooled_confidence

    summary_failures: list[str] = []
    if not completed:
        summary_failures.append("No datasets completed")
    for d in completed:
        if not d["gate_passed"]:
            summary_failures.append(f"{d['dataset']}: {'; '.join(d['failure_reasons'])}")

    if completed:
        if bool(gate_spec.get("require_mean_recall5_parity", True)):
            mean_r5_delta = float(macro_means["delta_yaw_vs_hipporag"]["Recall@5"])
            if mean_r5_delta < -1e-9:
                summary_failures.append(
                    f"Macro-average Recall@5 delta vs historical HippoRAG ({mean_r5_delta:+.4f}) < 0.0000"
                )
        if effective_mode == "eval":
            has_f1_spec = any(
                isinstance(d_spec, Mapping)
                and isinstance(
                    b := d_spec.get(primary_target, d_spec.get("monkeypatched_hipporag", d_spec)),
                    Mapping,
                )
                and b.get("F1") is not None
                for d_spec in (*spec_datasets.values(), *target_datasets.values())
            )
            has_em_spec = any(
                isinstance(d_spec, Mapping)
                and isinstance(
                    b := d_spec.get(primary_target, d_spec.get("monkeypatched_hipporag", d_spec)),
                    Mapping,
                )
                and b.get("ExactMatch") is not None
                for d_spec in (*spec_datasets.values(), *target_datasets.values())
            )
            if bool(gate_spec.get("require_mean_f1_parity", has_f1_spec)):
                mean_f1_delta = float(macro_means["delta_yaw_vs_hipporag"]["F1"])
                if mean_f1_delta < -1e-9:
                    summary_failures.append(
                        f"Macro-average QA F1 delta vs historical HippoRAG ({mean_f1_delta:+.4f}) < 0.0000"
                    )
            if bool(gate_spec.get("require_mean_em_parity", has_em_spec)):
                mean_em_delta = float(macro_means["delta_yaw_vs_hipporag"]["ExactMatch"])
                if mean_em_delta < -1e-9:
                    summary_failures.append(
                        f"Macro-average ExactMatch delta vs historical HippoRAG ({mean_em_delta:+.4f}) < 0.0000"
                    )

    overall_passed = len(summary_failures) == 0
    return {
        "passed": overall_passed,
        "overall_gate_passed": overall_passed,
        "overall_gate_status": "PASS" if overall_passed else "FAIL",
        "mode": effective_mode,
        "per_dataset_tolerance": tolerance,
        "datasets_completed": len(completed),
        "datasets_passed": sum(1 for d in completed if d["gate_passed"]),
        "datasets": dataset_evaluations,
        "macro_means": macro_means,
        "confidence": pooled_confidence,
        "failures": summary_failures,
        "failure_reasons": summary_failures,
    }


def check_benchmark_integrity(
    records: Sequence[Mapping[str, Any]],
    graph_info: Mapping[str, Any],
    expected_doc_count: int | None = None,
    extraction_errors: Sequence[str] = (),
    qa_errors: Sequence[str] = (),
    include_qa: bool = True,
) -> list[str]:
    """Fail-fast integrity checks on indexed chunk counts, retrieval outputs, and QA answers."""
    errors: list[str] = []
    if extraction_errors:
        errors.extend(f"extraction_llm: {err}" for err in extraction_errors)
    if include_qa and qa_errors:
        errors.extend(f"qa_llm: {err}" for err in qa_errors)
    num_indexed = graph_info.get("num_indexed_docs", graph_info.get("num_passage_nodes"))
    if expected_doc_count is not None and num_indexed != expected_doc_count:
        errors.append(f"Index passage count mismatch: {num_indexed} != {expected_doc_count}")
    raw_chunks = graph_info.get("num_passage_nodes", 0)
    num_chunks = int(raw_chunks) if isinstance(raw_chunks, (int, float)) and not isinstance(raw_chunks, bool) else 0
    if (expected_doc_count or records) and num_chunks <= 0:
        errors.append("No chunks indexed in SQLite index")
    if include_qa:
        for idx, record in enumerate(records):
            if not str(record.get("answer", "")).strip():
                errors.append(f"Question {idx + 1}: empty QA answer")
    if len(records) > 1 and not any(
        record.get("retrieved_chunk_ids") or record.get("graph_seeds") for record in records
    ):
        errors.append("All queries returned empty retrieval results (retrieved_chunk_ids/graph_seeds)")
    return errors
