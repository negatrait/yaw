"""DSPy prompt optimization (MIPROv2, GEPA, SIMBA) and evaluation."""

from __future__ import annotations

import contextlib
import copy
import functools
import hashlib
import inspect
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import dspy
from gepa.core.callbacks import CompositeCallback

from yaw.core import ChunkPointer, YawRuntimeError, compute_mdhash_id, write_text_atomic

from .chat import configure_yaw_dspy_cache, set_yaw_sleep_seconds
from .embedding import FACT_INSTRUCTION
from .parsers import (
    compute_answer_passage_match,
    compute_qa_em_f1,
    compute_retrieval_recalls,
    compute_task_quality_score,
    doc_matches_gold,
    normalize_answer,
)
from .program import HippoRAGProgram
from .proposer import RLMInstructionProposer
from .reflection import (
    ActiveReflectionCheckpoint,
    ServingTokenCounter,
    checkpoint_object,
    validate_active_checkpoint,
    validate_request_checkpoint,
    validate_simba_state,
)


def _optimization_history_scope(fn: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        # GEPA consumes DSPy traces and usage, not inspection histories containing
        # full prompts/responses on every LM, caller module and GLOBAL_HISTORY.
        with dspy.context(disable_history=True, max_history_size=0):
            return fn(*args, **kwargs)

    return wrapped


class GepaProgress:
    """Read native callbacks/state without changing optimizer decisions."""

    def __init__(self, publish: Callable[[], None]) -> None:
        self.publish = publish
        self.data: dict[str, Any] = {
            "version": 1,
            "phase": "waiting for GEPA",
            "iteration": 0,
            "frontier": None,
            "frontier_history": [],
            "validation": [],
            "phase_evaluations": 0,
            "phase_expected": None,
        }

    def snapshot(self) -> dict[str, Any]:
        return dict(self.data, updated_at=time.time())

    def _phase(self, phase: str, event: Any, expected: int | None = None) -> None:
        self.data.update(phase=phase, iteration=event.get("iteration", 0), phase_evaluations=0, phase_expected=expected)
        self.publish()

    def metric_evaluated(self) -> None:
        if self.data["phase"] == "minibatch complete; acceptance / validation pending":
            self.data["phase"] = "validation evaluation (metric calls observed)"
        self.data["phase_evaluations"] += 1

    def _front(self, members: list[int], iteration: int, *, force: bool = False) -> None:
        previous = self.data["frontier"]
        members = sorted(set(members))
        if force or members != previous:
            history = self.data["frontier_history"]
            history.append(
                {
                    "event": self.data.get("frontier_events", 0) + 1,
                    "iteration": iteration,
                    "members": members,
                    "added": sorted(set(members) - set(previous)) if previous is not None else [],
                    "removed": sorted(set(previous or []) - set(members)),
                }
            )
            self.data["frontier_events"] = history[-1]["event"]
            del history[:-250]
        self.data["frontier"] = members

    def restore_state(self, state: Any, total: int) -> None:
        """Copy only scalar scores/counts and exact native frontier membership."""
        self.data["validation"] = [
            {
                "candidate_idx": idx,
                "score": sum(scores.values()) / len(scores),
                "evaluated": len(scores),
                "total": total,
                "source": "GEPA state",
            }
            for idx, scores in enumerate(state.prog_candidate_val_subscores)
            if scores
        ]
        self.data["frontier_type"] = state.frontier_type
        self._front([idx for group in state.get_pareto_front_mapping().values() for idx in group], state.i + 1)
        self.data["native_metric_calls"] = state.total_num_evals

    def on_optimization_start(self, event: Any) -> None:
        self.data["valset_size"] = int(event["valset_size"])
        if self.data["validation"]:
            self._phase("initializing from checkpoint", event)
        else:
            self._phase("initializing / seed validation", event, int(event["valset_size"]))

    def on_candidate_selected(self, event: Any) -> None:
        self.data["selected_parent"] = int(event["candidate_idx"])
        self.publish()

    def on_budget_updated(self, event: Any) -> None:
        self.data["native_metric_calls"] = int(event["metric_calls_used"])
        self.publish()

    def on_error(self, event: Any) -> None:
        phase = "error; continuing" if event["will_continue"] else "error; stopping"
        self._phase(phase, event)

    def on_iteration_start(self, event: Any) -> None:
        self.restore_state(event["state"], self.data["valset_size"])
        self._phase("selecting candidate", event)

    def on_proposal_start(self, event: Any) -> None:
        self._phase("reflecting / generating instructions", event)

    def on_proposal_end(self, event: Any) -> None:
        self._phase("proposal generated; awaiting evaluation", event)

    def on_evaluation_start(self, event: Any) -> None:
        self._phase("minibatch evaluation", event, int(event["batch_size"]))

    def on_evaluation_end(self, event: Any) -> None:
        self._phase("minibatch complete; acceptance / validation pending", event)

    def on_pareto_front_updated(self, event: Any) -> None:
        self._front(list(event["new_front"]), int(event["iteration"]), force=True)
        self.publish()

    def on_valset_evaluated(self, event: Any) -> None:
        rows = self.data["validation"]
        idx = int(event["candidate_idx"])
        row = {
            "candidate_idx": idx,
            "score": float(event["average_score"]),
            "evaluated": int(event["num_examples_evaluated"]),
            "total": int(event["total_valset_size"]),
            "source": "GEPA validation callback",
        }
        rows[:] = [r for r in rows if r["candidate_idx"] != idx] + [row]
        self._phase("validation complete", event)

    def on_iteration_end(self, event: Any) -> None:
        self.restore_state(event["state"], self.data["valset_size"])
        self._phase("iteration complete: " + ("accepted" if event["proposal_accepted"] else "rejected"), event)

    def on_optimization_end(self, event: Any) -> None:
        self.restore_state(event["final_state"], self.data["valset_size"])
        self._phase("optimization complete", {"iteration": event["total_iterations"]})


@dataclass(frozen=True, slots=True)
class GraphOptimizationHooks:
    """Explicit callback contract passed by `YawGraphEngine` into `optimizer.py` (zero horizontal imports)."""

    ensure_candidate_indexed: Callable[[HippoRAGProgram, Sequence[str] | None], list[str]]
    reset_sandbox_graph: Callable[[bool], None]
    index_chunks: Callable[[Sequence[tuple[ChunkPointer, str]], HippoRAGProgram], int]
    persist_trails: Callable[[HippoRAGProgram], None]
    get_meta: Callable[[str], str | None]
    set_meta: Callable[[str, str | None], None]
    candidate_cache_namespace: Callable[[], str]
    candidate_cache_sources: Callable[[], dict[str, str]]
    retrieval_settings: Mapping[str, Any] = field(default_factory=dict)
    bind_program_hooks: Callable[[HippoRAGProgram], None] = field(default=lambda _prog: None)
    open_score_cache: Callable[[str], Any] = field(default=lambda _ctx: {})
    open_checkpoint_cache: Callable[[str], Any] = field(default=lambda _ctx: {})
    encode_payload: Callable[[Any], str] = field(
        default=lambda val: json.dumps(val, sort_keys=True, ensure_ascii=False, default=str)
    )
    decode_payload: Callable[[str], Any] = field(default=json.loads)
    commit_active_checkpoint: Callable[[str, str, str | None, str | None], None] | None = None
    prepare_optimization_pools: (
        Callable[
            [
                Sequence[Any],
                Sequence[Any] | None,
                Sequence[tuple[ChunkPointer, str]] | None,
                str,
                Path | str | None,
                Callable[[str], None] | None,
            ],
            None,
        ]
        | None
    ) = None
    hydrate_top_chunks: Callable[[Sequence[str], HippoRAGProgram], dict[str, str]] = field(
        default=lambda _cids, _prog: {}
    )
    query_rlm_stats: Callable[[Sequence[str] | None], dict[str, Any]] = field(default=lambda _docs: {})
    graph_rlm_stats: Callable[[], dict[str, Any]] = field(default=dict)
    last_hook_program: Callable[[], HippoRAGProgram | None] = field(default=lambda: None)
    clear_progress_callback: Callable[[], None] = field(default=lambda: None)
    finalize_compiled_program: Callable[[HippoRAGProgram], None] = field(default=lambda _prog: None)
    rehydrate_sandbox_after_optimization: Callable[[HippoRAGProgram], None] = field(default=lambda _prog: None)
    dspy_cache_dir: Path | None = None
    config: Any = None


def serialize_predictor_surface(pred: Any) -> dict[str, Any]:
    """Serialize a DSPy predictor's instruction, demos, field types, config, and LM cache identity."""
    sig = getattr(pred, "signature", None)
    instructions = str(getattr(sig, "instructions", "") or "")
    in_fields = set(getattr(sig, "input_fields", {}) or {})
    out_fields = set(getattr(sig, "output_fields", {}) or {})
    demos_raw = getattr(pred, "demos", None) or ()
    demos_repr: list[str] = []
    for d in demos_raw:
        d_dict = d.toDict() if hasattr(d, "toDict") else (dict(d) if isinstance(d, Mapping) else None)
        if isinstance(d_dict, dict) and in_fields and out_fields:
            if not (in_fields.issubset(d_dict.keys()) and out_fields.issubset(d_dict.keys())):
                continue
            filtered = {k: d_dict[k] for k in sorted(in_fields | out_fields)}
            demos_repr.append(json.dumps(filtered, sort_keys=True, ensure_ascii=False, default=str))
        else:
            demos_repr.append(str(d_dict) if d_dict is not None else str(d))
    lm = getattr(pred, "lm", None) or dspy.settings.lm
    identity = getattr(lm, "cache_identity", None)
    return {
        "instructions": instructions,
        "demos": demos_repr,
        "fields": {name: str(f.annotation) for name, f in sig.fields.items()} if sig else {},
        "config": getattr(pred, "config", {}),
        "lm": identity()
        if callable(identity)
        else {
            "model": getattr(lm, "model", None),
            "model_type": getattr(lm, "model_type", None),
            "context_window_tokens": getattr(lm, "_yaw_context_window_tokens", None),
            "kwargs": {
                key: value
                for key, value in (getattr(lm, "kwargs", {}) or {}).items()
                if key not in {"api_key", "api_token"}
            },
        },
    }


def prune_mismatching_demos(program: HippoRAGProgram) -> None:
    """Remove any few-shot demos whose keys do not cover their predictor's input and output fields."""
    if not hasattr(program, "named_predictors"):
        return
    for _, pred in program.named_predictors():
        demos_raw = getattr(pred, "demos", None)
        if not demos_raw:
            continue
        sig = getattr(pred, "signature", None)
        in_fields = set(getattr(sig, "input_fields", {}) or {})
        out_fields = set(getattr(sig, "output_fields", {}) or {})
        if not (in_fields and out_fields):
            continue
        valid_demos: list[Any] = []
        for d in demos_raw:
            d_dict = d.toDict() if hasattr(d, "toDict") else (dict(d) if isinstance(d, Mapping) else None)
            if isinstance(d_dict, dict) and in_fields.issubset(d_dict.keys()) and out_fields.issubset(d_dict.keys()):
                valid_demos.append(d)
        pred.demos = valid_demos


def embed_fingerprint(program: Any, instruction: str = "") -> str:
    """Compute SHA-256 fingerprint for the active embedding prefix."""
    query_prefix = getattr(program, "query_prefix", getattr(program, "embedding_query_prefix", None))
    fact_prefix = getattr(program, "fact_query_prefix", getattr(program, "embedding_fact_prefix", None))
    if not instruction:
        prefix = getattr(program, "document_prefix", getattr(program, "embedding_document_prefix", None))
    elif instruction == FACT_INSTRUCTION:
        prefix = fact_prefix or query_prefix
    else:
        prefix = query_prefix
    payload = {"prefix": prefix}
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def synonym_fingerprint(program: HippoRAGProgram) -> str:
    """Compute SHA-256 fingerprint for Stage 6 synonym judgment predictor surface."""
    named = dict(program.named_predictors()) if hasattr(program, "named_predictors") else {}
    payload = serialize_predictor_surface(named["synonym_judge"]) if "synonym_judge" in named else {}
    payload["context_budget"] = [program.context_window_tokens, program.max_output_tokens]
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def shacl_fingerprint(program: HippoRAGProgram) -> str:
    """Compute SHA-256 fingerprint for SHACL repair and RLM healer predictor surfaces."""
    named = dict(program.named_predictors()) if hasattr(program, "named_predictors") else {}
    shacl_keys = (
        "shacl_repair",
        "shacl_healer_rlm.generate_action",
        "shacl_healer_rlm.extract",
    )
    payload: dict[str, object] = {k: serialize_predictor_surface(named[k]) for k in shacl_keys if k in named}
    healer = getattr(program, "shacl_healer_rlm", None)
    payload["rlm_limits"] = {
        name: getattr(healer, name, None) for name in ("max_iters", "max_llm_calls", "max_output_chars")
    }
    payload["context_budget"] = [program.context_window_tokens, program.max_output_tokens]
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def openie_fingerprint(program: HippoRAGProgram) -> str:
    """Compute SHA-256 fingerprint for OpenIE extraction predictor surfaces."""
    named = dict(program.named_predictors()) if hasattr(program, "named_predictors") else {}
    payload = {
        "stages": {
            k: serialize_predictor_surface(named[k]) for k in ("ner", "triples", "entity_definer") if k in named
        },
        "gbnf_max_entities": getattr(program, "gbnf_max_entities", None),
        "gbnf_max_triples": getattr(program, "gbnf_max_triples", None),
        "context_budget": [program.context_window_tokens, program.max_output_tokens],
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def extraction_fingerprint(program: HippoRAGProgram) -> str:
    """Compute SHA-256 fingerprint across all graph extraction, synonym, SHACL, and document embedding stages."""
    named = dict(program.named_predictors()) if hasattr(program, "named_predictors") else {}
    graph_stage_keys = (
        "ner",
        "triples",
        "entity_definer",
        "synonym_judge",
        "shacl_repair",
        "shacl_healer_rlm.generate_action",
        "shacl_healer_rlm.extract",
    )
    payload = {
        "stages": {k: serialize_predictor_surface(named[k]) for k in graph_stage_keys if k in named},
        "gbnf_max_entities": getattr(program, "gbnf_max_entities", None),
        "gbnf_max_triples": getattr(program, "gbnf_max_triples", None),
        "embedding_document_prefix": getattr(program, "embedding_document_prefix", None),
        "context_budget": [program.context_window_tokens, program.max_output_tokens],
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def candidate_fingerprint(program: HippoRAGProgram) -> str:
    """Identify an optimizer candidate by hashing every predictor surface and embedding prefix."""
    named = dict(program.named_predictors()) if hasattr(program, "named_predictors") else {}
    payload = {
        "stages": {k: serialize_predictor_surface(v) for k, v in sorted(named.items())},
        "embedding_query_prefix": getattr(program, "embedding_query_prefix", None),
        "embedding_fact_prefix": getattr(program, "embedding_fact_prefix", None),
        "embedding_document_prefix": getattr(program, "embedding_document_prefix", None),
        "context_budget": [program.context_window_tokens, program.max_output_tokens],
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def query_retrieval_fingerprint(program: HippoRAGProgram) -> str:
    """Compute SHA-256 fingerprint for query-time retrieval predictor surfaces (`query_intent`, `triple_filter`)."""
    named = dict(program.named_predictors()) if hasattr(program, "named_predictors") else {}
    payload = {
        "stages": {
            k: serialize_predictor_surface(named[k]) for k in ("query_intent", "triple_filter") if k in named
        },
        "embedding_query_prefix": getattr(program, "embedding_query_prefix", None),
        "embedding_fact_prefix": getattr(program, "embedding_fact_prefix", None),
        "context_budget": [program.context_window_tokens, program.max_output_tokens],
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _query_qa_fingerprint(program: HippoRAGProgram) -> str:
    """Compute SHA-256 fingerprint for query-time + QA predictor surfaces (`query_intent`, `triple_filter`, `qa`)."""
    named = dict(program.named_predictors()) if hasattr(program, "named_predictors") else {}
    qa_keys = ("query_intent", "triple_filter", "qa", "qa.predict")
    payload = {
        "stages": {k: serialize_predictor_surface(named[k]) for k in qa_keys if k in named},
        "embedding_query_prefix": getattr(program, "embedding_query_prefix", None),
        "embedding_fact_prefix": getattr(program, "embedding_fact_prefix", None),
        "context_budget": [program.context_window_tokens, program.max_output_tokens],
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def select_pareto_frontier_programs(
    compiled: HippoRAGProgram,
    *,
    max_programs: int = 3,
    program_path: Path | str | None = None,
) -> list[HippoRAGProgram]:
    """Select up to `max_programs` strictly non-dominated Pareto programs covering both `recall@N` and `QA (F1, EM)`."""
    if max_programs <= 1:
        return [compiled]

    selected: list[HippoRAGProgram] = [compiled]
    seen_fps: set[str] = {_query_qa_fingerprint(compiled)}

    details = getattr(compiled, "detailed_results", None)
    raw_candidates = list(getattr(details, "candidates", ()) or ()) if details is not None else []
    if raw_candidates:
        raw_scores = [float(s) for s in (getattr(details, "val_aggregate_scores", ()) or ())]
        raw_subscores = list(getattr(details, "val_aggregate_subscores", ()) or ())
        best_idx = int(getattr(details, "best_idx", 0) or 0)

        eligible_indices: list[int] = []
        subscores_by_idx: dict[int, dict[str, float]] = {}
        scores_by_idx: dict[int, float] = {}
        for idx, cand in enumerate(raw_candidates):
            if not isinstance(cand, HippoRAGProgram):
                continue
            sc = raw_scores[idx] if idx < len(raw_scores) else 0.0
            if sc <= 0.0 and idx != best_idx:
                continue
            scores_by_idx[idx] = sc
            raw_sub = raw_subscores[idx] if idx < len(raw_subscores) else None
            sub_dict: dict[str, float] = {}
            if isinstance(raw_sub, Mapping):
                for k, v in raw_sub.items():
                    if isinstance(v, (int, float)):
                        sub_dict[str(k)] = float(v)
            if not sub_dict:
                sub_dict = {"score": sc}
            subscores_by_idx[idx] = sub_dict
            eligible_indices.append(idx)

        def _dominates(j: int, i: int) -> bool:
            sub_j = subscores_by_idx[j]
            sub_i = subscores_by_idx[i]
            keys = sorted(set(sub_j.keys()) | set(sub_i.keys()))
            if not keys:
                return scores_by_idx[j] > scores_by_idx[i] + 1e-9
            all_ge = all(sub_j.get(k, 0.0) >= sub_i.get(k, 0.0) - 1e-9 for k in keys)
            any_gt = any(sub_j.get(k, 0.0) > sub_i.get(k, 0.0) + 1e-9 for k in keys)
            return all_ge and any_gt

        non_dominated = [
            i for i in eligible_indices if i != best_idx and not any(_dominates(j, i) for j in eligible_indices if j != i)
        ]

        # Deduplicate by query-time + QA predictor surface while preserving highest scalar score per surface
        unique_pool: list[int] = []
        for idx in sorted(non_dominated, key=lambda i: (-scores_by_idx[i], i)):
            cand = raw_candidates[idx]
            fp = _query_qa_fingerprint(cand)
            if fp in seen_fps:
                continue
            unique_pool.append(idx)

        def _retrieval_mean(idx: int) -> float:
            sub = subscores_by_idx[idx]
            vals = [
                float(sub[k])
                for k in ("recall_at_1", "recall_at_2", "recall_at_5", "recall_at_10", "recall_at_20")
                if k in sub
            ]
            return sum(vals) / len(vals) if vals else scores_by_idx[idx]

        def _qa_mean(idx: int) -> float:
            sub = subscores_by_idx[idx]
            vals = [float(sub[k]) for k in ("f1", "em") if k in sub]
            return sum(vals) / len(vals) if vals else scores_by_idx[idx]

        def _accept_idx(idx: int) -> None:
            cand = raw_candidates[idx]
            compiled.copy_runtime_hooks_to(cand)
            seen_fps.add(_query_qa_fingerprint(cand))
            selected.append(cand)
            if idx in unique_pool:
                unique_pool.remove(idx)

        # Slot 1: recall@N Pareto leader
        if unique_pool and len(selected) < max_programs:
            r_leader = max(unique_pool, key=lambda i: (_retrieval_mean(i), scores_by_idx[i], -i))
            _accept_idx(r_leader)

        # Slot 2: QA (F1, EM) Pareto leader
        if unique_pool and len(selected) < max_programs:
            unique_pool = [i for i in unique_pool if _query_qa_fingerprint(raw_candidates[i]) not in seen_fps]
            if unique_pool:
                q_leader = max(unique_pool, key=lambda i: (_qa_mean(i), scores_by_idx[i], -i))
                _accept_idx(q_leader)

        # Fill any remaining slots up to max_programs by balanced perspective score 0.5 * R_bar + 0.5 * Q_bar
        while unique_pool and len(selected) < max_programs:
            unique_pool = [i for i in unique_pool if _query_qa_fingerprint(raw_candidates[i]) not in seen_fps]
            if not unique_pool:
                break
            b_leader = max(
                unique_pool,
                key=lambda i: (0.5 * _retrieval_mean(i) + 0.5 * _qa_mean(i), scores_by_idx[i], -i),
            )
            _accept_idx(b_leader)

        return selected

    eff_path = Path(program_path) if program_path is not None else getattr(compiled, "loaded_from_path", None)
    if eff_path is not None:
        telem_dir = eff_path.parent / f"{eff_path.stem}_telemetry"
        if telem_dir.exists() and telem_dir.is_dir():
            for member_file in sorted(telem_dir.glob("pareto_member_*.json")):
                if len(selected) >= max_programs:
                    break
                try:
                    sibling = copy.deepcopy(compiled)
                    sibling.load(str(member_file))
                    compiled.copy_runtime_hooks_to(sibling)
                    fp = _query_qa_fingerprint(sibling)
                    if fp not in seen_fps:
                        seen_fps.add(fp)
                        selected.append(sibling)
                except Exception:
                    continue

    return selected


def reduce_pareto_predictions(predictions: Sequence[dspy.Prediction]) -> dspy.Prediction:
    """Fuse `recall@N` rankings via RRF + normalized PPR scores and resolve `QA (F1, EM)` via majority EM + F1 medoid."""
    if not predictions:
        return dspy.Prediction(
            context="",
            reasoning="",
            retrieved_passages=(),
            retrieved_scores=(),
            retrieved_chunk_ids=(),
            answer="",
        )
    if len(predictions) == 1:
        return predictions[0]

    # 1. recall@N Perspective: Reciprocal Rank Fusion (1 / (60 + r)) + min-max normalized PPR score
    fused_scores: dict[str, float] = {}
    passage_by_cid: dict[str, str] = {}
    first_seen_order: dict[str, int] = {}
    max_k = 0
    for pred in predictions:
        raw_pids = tuple(str(x) for x in (getattr(pred, "retrieved_chunk_ids", None) or ()))
        raw_passages = tuple(str(x) for x in (getattr(pred, "retrieved_passages", None) or ()))
        raw_scores = tuple(float(x) for x in (getattr(pred, "retrieved_scores", None) or ()))
        if not raw_pids and raw_passages:
            raw_pids = tuple(compute_mdhash_id(p, prefix="chunk-") for p in raw_passages)
        max_k = max(max_k, len(raw_pids))
        if not raw_pids:
            continue
        max_s = max(raw_scores) if raw_scores else 0.0
        min_s = min(raw_scores) if raw_scores else 0.0
        span_s = max_s - min_s
        for idx, cid in enumerate(raw_pids):
            rank_1 = idx + 1
            raw_s = raw_scores[idx] if idx < len(raw_scores) else 0.0
            if span_s > 1e-12:
                norm_s = (raw_s - min_s) / span_s
            elif max_s > 0.0:
                norm_s = raw_s / max_s
            else:
                norm_s = 0.0
            fused_scores[cid] = fused_scores.get(cid, 0.0) + (1.0 / (60.0 + rank_1)) + norm_s
            if cid not in first_seen_order:
                first_seen_order[cid] = len(first_seen_order)
            if cid not in passage_by_cid and idx < len(raw_passages) and raw_passages[idx]:
                passage_by_cid[cid] = raw_passages[idx]

    ranked_cids = sorted(
        fused_scores.keys(),
        key=lambda c: (-fused_scores[c], first_seen_order.get(c, 0)),
    )[: max(20, max_k)]
    fused_pids = tuple(ranked_cids)
    fused_passages = tuple(passage_by_cid.get(c, "") for c in fused_pids)
    fused_score_tuple = tuple(round(float(fused_scores[c]), 6) for c in fused_pids)

    primary_pids = tuple(str(x) for x in (getattr(predictions[0], "retrieved_chunk_ids", None) or ()))
    if fused_pids == primary_pids and getattr(predictions[0], "context", ""):
        fused_context = str(getattr(predictions[0], "context", ""))
    else:
        non_empty_psgs = [p for p in fused_passages[:10] if p]
        fused_context = (
            "\n\n".join(non_empty_psgs)
            if non_empty_psgs
            else str(getattr(predictions[0], "context", "") or "")
        )

    # 2. QA (F1, EM) Perspective: dspy.majority EM vote when max count >= 2, else token-F1 medoid tie-break
    non_empty_members: list[tuple[int, dspy.Prediction, str, str]] = []
    norm_counts: dict[str, int] = {}
    for idx, pred in enumerate(predictions):
        raw_ans = str(getattr(pred, "answer", "") or "").strip()
        norm_ans = normalize_answer(raw_ans) if raw_ans else ""
        if norm_ans:
            non_empty_members.append((idx, pred, raw_ans, norm_ans))
            norm_counts[norm_ans] = norm_counts.get(norm_ans, 0) + 1

    if not non_empty_members:
        winner = predictions[0]
    elif max(norm_counts.values()) >= 2:
        winner = dspy.majority( [p for _, p, _, _ in non_empty_members], field="answer")
    elif len(non_empty_members) == 1:
        winner = non_empty_members[0][1]
    else:
        best_medoid_score = -1.0
        winner = non_empty_members[0][1]
        for i_pos, (_orig_i, pred_i, ans_i, _norm_i) in enumerate(non_empty_members):
            pairwise_f1s = [
                compute_qa_em_f1(ans_i, [ans_j])[1]
                for j_pos, (_orig_j, _pred_j, ans_j, _norm_j) in enumerate(non_empty_members)
                if j_pos != i_pos
            ]
            mean_f1 = sum(pairwise_f1s) / len(pairwise_f1s) if pairwise_f1s else 0.0
            if mean_f1 > best_medoid_score + 1e-9:
                best_medoid_score = mean_f1
                winner = pred_i

    winning_answer = str(getattr(winner, "answer", "") or "")
    winning_reasoning = str(getattr(winner, "reasoning", "") or "")
    if not winning_reasoning:
        for _, p_cand, _, n_cand in non_empty_members:
            if n_cand == normalize_answer(winning_answer) and getattr(p_cand, "reasoning", ""):
                winning_reasoning = str(getattr(p_cand, "reasoning", ""))
                break

    res = dspy.Prediction(
        context=fused_context,
        reasoning=winning_reasoning,
        retrieved_passages=fused_passages,
        retrieved_scores=fused_score_tuple,
        retrieved_chunk_ids=fused_pids,
        answer=winning_answer,
    )
    total_usage: dict[str, dict[str, int]] = {}
    for pred in predictions:
        u = pred.get_lm_usage() if hasattr(pred, "get_lm_usage") else None
        if isinstance(u, Mapping):
            for model_k, counts in u.items():
                if isinstance(counts, Mapping):
                    dest = total_usage.setdefault(str(model_k), {})
                    for ck, cv in counts.items():
                        if isinstance(cv, (int, float)):
                            dest[str(ck)] = dest.get(str(ck), 0) + int(cv)
    if total_usage and hasattr(res, "set_lm_usage"):
        res.set_lm_usage(total_usage)
    return res


def compile_pareto_ensemble(
    compiled: HippoRAGProgram,
    *,
    max_programs: int = 3,
    program_path: Path | str | None = None,
) -> dspy.Module:
    """Compile an external `dspy.Ensemble(reduce_fn=reduce_pareto_predictions)` over up to 3 non-dominated Pareto programs."""
    programs = select_pareto_frontier_programs(
        compiled,
        max_programs=max_programs,
        program_path=program_path,
    )
    if len(programs) <= 1:
        return compiled
    return dspy.Ensemble(reduce_fn=reduce_pareto_predictions).compile(programs)


def extract_named_trace_entries(
    program: HippoRAGProgram,
    raw_trace: Sequence[Any],
    *,
    allowed_names: set[str] | None = None,
) -> list[tuple[str, dict[str, Any], Any]]:
    """Map raw DSPy trace tuples back to their named predictor names on `program`."""
    named_preds = list(program.named_predictors()) if hasattr(program, "named_predictors") else []
    captured: list[tuple[str, dict[str, Any], Any]] = []
    for item in raw_trace:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        pred_inst, inp, out = item[0], item[1], item[2]
        matched_name: str | None = None
        pred_sig = getattr(pred_inst, "signature", None)
        for name, cand_pred in named_preds:
            if pred_inst is cand_pred:
                matched_name = name
                break
        if matched_name is None and pred_sig is not None:
            pred_fields = tuple(getattr(pred_sig, "fields", {}).keys())
            for name, cand_pred in named_preds:
                cand_sig = getattr(cand_pred, "signature", None)
                if cand_sig is not None:
                    is_eq = cand_sig.equals(pred_sig) if hasattr(cand_sig, "equals") else cand_sig == pred_sig
                    cand_fields = tuple(getattr(cand_sig, "fields", {}).keys())
                    if is_eq or (pred_fields and pred_fields == cand_fields):
                        matched_name = name
                        break
        if matched_name is not None:
            if allowed_names is not None and not (
                matched_name in allowed_names or any(matched_name.startswith(f"{a}.") for a in allowed_names)
            ):
                continue
            inp_dict = dict(inp) if isinstance(inp, Mapping) else {"input": inp}
            captured.append((matched_name, inp_dict, out))
    return captured


class ActiveTraceComponentSelector:
    """GEPA component selector that dynamically prioritizes predictors with active execution traces."""

    def __init__(self, program: HippoRAGProgram) -> None:
        self.program = program
        self.cursor = 0

    def __call__(
        self,
        state: Any,
        trajectories: list[Any],
        subsample_scores: list[float],
        candidate_idx: int,
        candidate: dict[str, str],
    ) -> list[str]:
        _ = subsample_scores
        active_sigs = set()
        for data in trajectories or []:
            for t in data.get("trace", ()):
                sig = getattr(t[0], "signature", None)
                if sig is not None:
                    active_sigs.add(sig)

        active_names: list[str] = []
        name_to_pred = dict(self.program.named_predictors())
        for name in getattr(state, "list_of_named_predictors", list(candidate.keys())):
            pred = name_to_pred.get(name)
            pred_sig = getattr(pred, "signature", None)
            if pred_sig is not None:
                cand_sig = (
                    pred_sig.with_instructions(candidate[name])
                    if name in candidate and hasattr(pred_sig, "with_instructions")
                    else pred_sig
                )
                pred_fields = tuple(getattr(pred_sig, "fields", {}).keys())
                for a_sig in active_sigs:
                    is_eq = cand_sig.equals(a_sig) if hasattr(cand_sig, "equals") else cand_sig == a_sig
                    a_fields = tuple(getattr(a_sig, "fields", {}).keys())
                    if is_eq or (pred_fields and pred_fields == a_fields):
                        active_names.append(name)
                        break

        if not active_names:
            pid = getattr(state, "named_predictor_id_to_update_next_for_program_candidate", [0])[candidate_idx]
            num_preds = len(state.list_of_named_predictors)
            state.named_predictor_id_to_update_next_for_program_candidate[candidate_idx] = (pid + 1) % num_preds
            return [state.list_of_named_predictors[pid]]

        selected = active_names[self.cursor % len(active_names)]
        self.cursor += 1
        return [selected]


def configure_simba_task_learning(
    proposer: RLMInstructionProposer,
    template: HippoRAGProgram,
    trainset: Sequence[Any],
    metric: Callable[[Any, Any], float],
    hooks: GraphOptimizationHooks,
    *,
    save_path: Path | str | None,
    resume: bool,
) -> None:
    """Wire counterfactual training rewards and atomic worker snapshots in yaw.db."""
    # Fixed paired calibration queries keep each inner step bounded. Labels
    # never come from the GEPA validation set or enter the worker's inputs.
    proposer.preflight()
    calibration = list(trainset[:2])
    if not calibration:
        raise ValueError("SIMBA task learning requires training queries")
    fallback_task_lm = getattr(dspy.settings, "lm", None)
    template = template.deepcopy()
    identity = {
        "version": 1,
        "metric": "task-quality-arithmetic-mean-v2",
        "reflection_deployment": proposer.cache_identity(),
        "namespace": hooks.candidate_cache_namespace(),
        "retrieval": dict(hooks.retrieval_settings),
        "template": candidate_fingerprint(template),
        "trainset": [{"data": ex.toDict(), "inputs": sorted(ex.inputs())} for ex in trainset],
        "corpus": sorted(hooks.candidate_cache_sources()),
        "worker": serialize_predictor_surface(proposer.qwen_proposer),
        "orchestrator": serialize_predictor_surface(proposer.inner_rlm.generate_action),
        "settings": [
            proposer.simba_bsize,
            proposer.simba_num_candidates,
            proposer.simba_max_steps,
            proposer.simba_max_demos,
            proposer.max_instruction_chars,
            proposer.max_field_chars,
            proposer.max_feedback_examples,
            proposer.enable_gbnf,
            proposer.worker_temperature,
            proposer.reflection_temperature,
            proposer.reflection_input_tokens,
            getattr(proposer.token_counter, "identity", None),
        ],
    }
    context = hashlib.sha256(
        json.dumps(identity, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    scores = hooks.open_score_cache(context)

    def score_candidate(instructions: Mapping[str, str]) -> float:
        key = (json.dumps(dict(instructions), sort_keys=True, ensure_ascii=False),)
        cached = scores.get(key)
        if cached is not None:
            return float(cached)
        task = template.deepcopy()
        named = dict(task.named_predictors())
        for name, instruction in instructions.items():
            named[name].signature = named[name].signature.with_instructions(instruction)
        hooks.bind_program_hooks(task)
        total = 0.0
        # SIMBA's reflection LM and traces must not override task LMs or
        # pollute its meta-training trajectory with downstream task calls.
        task_lm = None if template.rlm_sub_lm or template.orchestrator_lm else fallback_task_lm
        with dspy.context(lm=task_lm, trace=[], usage_tracker=None, track_usage=True):
            for gold in calibration:
                prediction = task(**gold.inputs())
                total += float(metric(gold, prediction))
        result = total / len(calibration)
        scores[key] = result
        return result

    run = str(Path(save_path).resolve()) if save_path is not None else "default"
    state_key = "simba_worker:" + hashlib.sha256(run.encode("utf-8")).hexdigest()

    def save_state(state: dict[str, Any]) -> None:
        hooks.set_meta(state_key, hooks.encode_payload({"context": context, "state": state}))

    saved_state = None
    if resume:
        raw = hooks.get_meta(state_key)
        if raw is not None:
            saved = checkpoint_object(hooks.decode_payload(raw), "SIMBA checkpoint")
            if saved.get("context") != context:
                raise ValueError("SIMBA checkpoint does not match this run's task/model/data")
            saved_state = validate_simba_state(saved.get("state"))
    checkpoints = hooks.open_checkpoint_cache(context)
    active_key = state_key + ":active_reflection"
    active_raw = hooks.get_meta(active_key) if resume else None
    active: ActiveReflectionCheckpoint | None = (
        validate_active_checkpoint(hooks.decode_payload(active_raw), context) if active_raw else None
    )
    if active is not None:
        saved_state = validate_simba_state(active["records"][active["request"]].get("state"))
    if saved_state is not None:
        proposer.load_simba_state(saved_state)

    def get_checkpoint(key: str) -> Any:
        if active is not None and key in active["records"]:
            return active["records"][key]
        raw = checkpoints.get((key,))
        return hooks.decode_payload(raw) if raw is not None else None

    def put_checkpoint(key: str, value: dict[str, Any]) -> None:
        nonlocal active
        if key.startswith("request:"):
            value = validate_request_checkpoint(value)
        checkpoints[(key,)] = hooks.encode_payload(value)
        next_active: ActiveReflectionCheckpoint
        if key.startswith("request:") and value["phase"] == "collecting":
            next_active = {"context": context, "request": key, "records": {key: value}}
        elif active is not None:
            next_active = {**active, "records": {**active["records"], key: value}}
        else:
            return
        # Pin only the active request, outside FIFO candidate-cache eviction.
        # Request state and the learned worker are committed in one transaction.
        encoded_active = hooks.encode_payload(next_active)
        encoded_state = (
            hooks.encode_payload({"context": context, "state": value["state"]}) if key.startswith("request:") else None
        )
        if hooks.commit_active_checkpoint is not None:
            hooks.commit_active_checkpoint(
                active_key,
                encoded_active,
                state_key if key.startswith("request:") else None,
                encoded_state,
            )
        else:
            hooks.set_meta(active_key, encoded_active)
            if encoded_state is not None:
                hooks.set_meta(state_key, encoded_state)
        active = next_active

    proposer.checkpoint_get = get_checkpoint
    proposer.checkpoint_put = put_checkpoint
    proposer.task_scorer = score_candidate
    proposer.save_simba_state = save_state


def evaluate_program(
    program: dspy.Module,
    devset: Sequence[dspy.Example],
    *,
    metric: Callable[..., Any] | None = None,
    num_threads: int = 1,
    display_progress: bool = False,
    display_table: bool | int = False,
    save_as_json: str | Path | None = None,
    save_as_csv: str | Path | None = None,
    include_semantic_metrics: bool = False,
) -> dspy.evaluate.EvaluationResult:
    """Evaluate `program` on `devset` using `dspy.Evaluate` and return `EvaluationResult`."""
    semantic_f1_mod = (
        dspy.evaluate.SemanticF1() if include_semantic_metrics and hasattr(dspy.evaluate, "SemanticF1") else None
    )
    grounded_mod = (
        dspy.evaluate.CompleteAndGrounded()
        if include_semantic_metrics and hasattr(dspy.evaluate, "CompleteAndGrounded")
        else None
    )

    def _extract_gold_answers(example: Any) -> list[str]:
        raw = getattr(example, "gold_answers", getattr(example, "answer", []))
        if isinstance(raw, str):
            return [raw] if raw.strip() else []
        return [str(a) for a in (raw or ()) if str(a).strip()]

    def _extract_gold_docs(example: Any) -> list[str]:
        raw = getattr(example, "gold_docs", getattr(example, "item_docs", ()))
        if isinstance(raw, str):
            return [raw] if raw.strip() else []
        return [str(d) for d in (raw or ()) if str(d).strip()]

    def _wrapped_metric(example: Any, pred: Any, trace: Any = None) -> float:
        _ = trace
        gold_answers = _extract_gold_answers(example)
        gold_docs = _extract_gold_docs(example)
        pred_answer = str(getattr(pred, "answer", "") or "").strip()
        context_val = getattr(pred, "context", "")
        apm_hit = compute_answer_passage_match(gold_answers, context_val)
        em_score, f1_score = compute_qa_em_f1(pred_answer, gold_answers) if gold_answers else (0.0, 0.0)
        if isinstance(pred, dspy.Prediction):
            pred.answer_passage_match = 1.0 if apm_hit else 0.0
            pred.em = float(em_score)
            pred.f1 = float(f1_score)
            if gold_docs:
                ret_passages = [str(p) for p in (getattr(pred, "retrieved_passages", None) or ()) if str(p).strip()]
                if not ret_passages:
                    ret_pids = [str(p) for p in (getattr(pred, "retrieved_chunk_ids", None) or ()) if str(p).strip()]
                    if ret_pids:
                        ret_passages = ret_pids
                    elif isinstance(context_val, str) and context_val.strip():
                        ret_passages = [context_val]
                recalls = compute_retrieval_recalls(gold_docs, ret_passages, k_list=(1, 2, 5, 10, 20))
                pred.recall_at_1 = float(recalls.get(1, 0.0))
                pred.recall_at_2 = float(recalls.get(2, 0.0))
                pred.recall_at_5 = float(recalls.get(5, 0.0))
                pred.recall_at_10 = float(recalls.get(10, 0.0))
                pred.recall_at_20 = float(recalls.get(20, 0.0))
            if include_semantic_metrics and gold_answers:
                q_str = str(getattr(example, "question", "") or "")
                ctx_str = context_val if isinstance(context_val, str) else "\n\n".join(str(x) for x in context_val)
                sem_ex = dspy.Example(question=q_str, response=gold_answers[0])
                sem_pred = dspy.Prediction(response=pred_answer, context=ctx_str)
                if semantic_f1_mod is not None:
                    sem_res = semantic_f1_mod(sem_ex, sem_pred)
                    pred.semantic_f1 = float(getattr(sem_res, "score", sem_res))
                if grounded_mod is not None:
                    grd_res = grounded_mod(sem_ex, sem_pred)
                    pred.complete_and_grounded = float(getattr(grd_res, "score", grd_res))

        if metric is not None:
            if metric is dspy.evaluate.answer_passage_match:
                return 1.0 if apm_hit else 0.0
            norm_ex = example
            if not hasattr(example, "answer") and gold_answers:
                norm_ex = dspy.Example(**example.toDict(), answer=list(gold_answers))
            raw_score = metric(norm_ex, pred)
            if hasattr(raw_score, "score"):
                return float(raw_score.score)
            return float(raw_score)
        return float(f1_score)

    evaluator = dspy.Evaluate(
        devset=list(devset),
        metric=_wrapped_metric,
        num_threads=num_threads,
        display_progress=display_progress,
        display_table=display_table,
        save_as_json=str(save_as_json) if save_as_json is not None else None,
        save_as_csv=str(save_as_csv) if save_as_csv is not None else None,
    )
    return evaluator(program)


@_optimization_history_scope
def evolve_program_prompts(
    program: HippoRAGProgram,
    trainset: Sequence[Any],
    valset: Sequence[Any] | None = None,
    *,
    hooks: GraphOptimizationHooks,
    mode: Literal["miprov2", "gepa", "hybrid"] = "hybrid",
    save_path: Path | str | None = None,
    reflection_lm: Any = None,
    reflection_worker_lm: Any = None,
    reflection_token_counter: Any = None,
    auto: Literal["light", "medium", "heavy"] | None = "light",
    num_candidates: int | None = None,
    num_trials: int | None = None,
    progress_callback: Callable[[str], None] | None = None,
    opt_chunks: Sequence[tuple[ChunkPointer, str]] | None = None,
    sleep_seconds: float = 0.0,
    max_bootstrapped_demos: int | None = None,
    max_labeled_demos: int | None = None,
) -> HippoRAGProgram:
    """Evolve DSPy prompts via MIPROv2, GEPA, or two-phase hybrid with composite Sizecoding feedback."""
    if not trainset:
        return program

    gepa_seed_path = (
        Path(save_path).parent / f"{Path(save_path).stem}_telemetry" / "seed_program.json"
        if save_path is not None and mode in ("gepa", "hybrid")
        else None
    )
    resumed_seed = gepa_seed_path is not None and gepa_seed_path.exists()
    if resumed_seed:
        program.load(str(gepa_seed_path))
        hooks.bind_program_hooks(program)
        hooks.finalize_compiled_program(program)

    resolved_ref_lm = (
        reflection_lm or program.orchestrator_lm or program.rlm_sub_lm or getattr(dspy.settings, "lm", None)
    )
    resolved_worker_lm = reflection_worker_lm or program.rlm_sub_lm or program.orchestrator_lm or resolved_ref_lm
    if resolved_worker_lm is None:
        raise YawRuntimeError("MODEL_ERROR", "Optimization requires a reflection LM")
    if reflection_token_counter is None:
        if hooks.config is None:
            raise YawRuntimeError(
                "INVALID_INPUT",
                "Optimization requires inference configuration or an explicit verified tokenizer",
            )
        reflection_token_counter = ServingTokenCounter(
            hooks.config.base_url,
            resolved_worker_lm.model,
            hooks.config.reflection_tokenizer_url,
        )
    proposer = RLMInstructionProposer(
        orchestrator_lm=resolved_ref_lm,
        worker_lm=resolved_worker_lm,
        enable_gbnf=True,
        rlm_max_iters=getattr(hooks.config, "rlm_max_iters", 10),
        rlm_max_llm_calls=getattr(hooks.config, "rlm_max_llm_calls", 20),
        rlm_max_output_chars=getattr(hooks.config, "rlm_max_output_chars", None),
        max_field_chars=program._context_char_limit(600),
        reflection_input_tokens=getattr(hooks.config, "reflection_input_tokens", None),
        token_counter=reflection_token_counter,
    )

    proposer.preflight()

    eff_sleep = sleep_seconds if sleep_seconds > 0 else getattr(hooks.config, "optimize_sleep_seconds", 0.0)
    set_yaw_sleep_seconds(eff_sleep)

    if hooks.prepare_optimization_pools is not None:
        hooks.prepare_optimization_pools(
            trainset,
            valset,
            opt_chunks,
            mode,
            save_path,
            progress_callback,
        )

    eval_count = 0
    eval_history: list[dict[str, Any]] = []
    last_eval_ts = time.time()
    # Per-candidate aggregates keyed by full predictor-surface fingerprint (first-seen order = candidate #).
    candidate_stats: dict[str, dict[str, Any]] = {}
    opt_phase = "gepa" if mode == "gepa" else "miprov2"
    last_eval_fp: str | None = None
    last_event_eval_count = 0
    active_program_ref: HippoRAGProgram = program
    _metric_keys = ("score", "r1", "r2", "r5", "r10", "r20", "em", "f1", "s_per_it", "tokens")

    def _record_candidate_eval(rec: dict[str, Any]) -> None:
        cs = candidate_stats.get(rec["cand_fp"])
        if cs is None:
            cs = candidate_stats[rec["cand_fp"]] = {
                "cand": len(candidate_stats) + 1,
                "fp": rec["cand_fp"],
                "optimizer": rec["optimizer"],
                "n": 0,
                "first_eval": rec["eval"],
                "score_min": float(rec["score"]),
                "score_max": float(rec["score"]),
                **{f"{k}_sum": 0.0 for k in _metric_keys},
            }
        cs["n"] += 1
        cs["last_eval"] = rec["eval"]
        cs["score_min"] = min(cs["score_min"], float(rec["score"]))
        cs["score_max"] = max(cs["score_max"], float(rec["score"]))
        for k in _metric_keys:
            cs[f"{k}_sum"] += float(rec[k])

    def _candidate_records() -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for cs in sorted(candidate_stats.values(), key=lambda c: int(c["cand"])):
            n = max(1, int(cs["n"]))
            row = {k: cs[k] for k in ("cand", "fp", "optimizer", "n", "first_eval", "last_eval") if k in cs}
            row.update({k: round(cs[f"{k}_sum"] / n, 4) for k in _metric_keys})
            row["score_min"] = round(cs["score_min"], 4)
            row["score_max"] = round(cs["score_max"], 4)
            for k in ("gepa_idx", "gepa_iteration", "val_score", "is_best"):
                if k in cs:
                    row[k] = cs[k]
            out.append(row)
        return out

    def _publish_optimization_audit() -> None:
        if save_path is None:
            return
        p_inc = Path(save_path)
        p_inc.parent.mkdir(parents=True, exist_ok=True)
        inc_payload = {
            "graph_rlm_stats": dict(hooks.graph_rlm_stats()),
            "candidate_records": _candidate_records(),
            "eval_records": eval_history[-500:],
            "gepa_progress": gepa_progress.snapshot(),
            **active_program_ref.dump_audit_trails(bounded=True, telemetry_key="program_telemetry"),
        }
        # Diagnostic write failures must not change optimization decisions.
        with contextlib.suppress(OSError):
            write_text_atomic(
                p_inc.parent / f"{p_inc.stem}_graph_rlm_audit.json",
                json.dumps(inc_payload, ensure_ascii=False, indent=2),
            )

    gepa_progress = GepaProgress(_publish_optimization_audit)

    def _build_stage_diagnostic(
        pred_name_str: str,
        pred_trace: Any,
        gold_answers: list[str],
        missed: list[str],
    ) -> str:
        trace_entries = list(pred_trace) if isinstance(pred_trace, (list, tuple)) else []
        out_summaries: list[str] = []
        for entry in trace_entries[:3]:
            if isinstance(entry, (list, tuple)) and len(entry) >= 3:
                outputs = entry[2]
                if isinstance(outputs, dict):
                    out_summaries.append(", ".join(f"{k}={repr(v)[:80]}" for k, v in list(outputs.items())[:3]))
                else:
                    attrs = [
                        f"{k}={repr(getattr(outputs, k))[:80]}"
                        for k in (
                            "named_entities",
                            "triples",
                            "definitions",
                            "decisions",
                            "repairs",
                            "intent_query",
                            "selected_fact_indices",
                            "answer",
                        )
                        if hasattr(outputs, k)
                    ]
                    if attrs:
                        out_summaries.append(", ".join(attrs))
        trace_summary = "; ".join(out_summaries) if out_summaries else "no_trace_outputs"
        target_hint = f"gold_answers={gold_answers[:2]!r}" if gold_answers else "no_gold_answers"
        miss_hint = f"missed={len(missed)}" if missed else "all_retrieved"
        return (
            f"Stage[{pred_name_str}]: trace_calls={len(trace_entries)} ({trace_summary}) | {target_hint} | {miss_hint}"
        )

    def _evaluate_prediction(
        gold: Any,
        pred: Any,
        pred_name: Any = None,
        pred_trace: Any = None,
        *,
        record: bool = True,
    ) -> dspy.Prediction:
        nonlocal eval_count, last_eval_ts, last_eval_fp
        if record and pred_name is None:
            eval_count += 1
        gold_answers_raw = getattr(gold, "gold_answers", getattr(gold, "answer", None))
        if gold_answers_raw is None:
            gold_answers = []
        elif isinstance(gold_answers_raw, str):
            gold_answers = [gold_answers_raw]
        else:
            gold_answers = [str(a) for a in gold_answers_raw]

        pred_answer = str(getattr(pred, "answer", "")).strip()
        context_str = str(getattr(pred, "context", ""))

        gold_docs_raw = getattr(gold, "gold_docs", None)
        retrieved_cids_raw = getattr(pred, "retrieved_chunk_ids", None)
        retrieved_passages_raw = getattr(pred, "retrieved_passages", None)
        if retrieved_cids_raw is not None and len(retrieved_cids_raw) > 0:
            retrieved_seq = [str(x) for x in retrieved_cids_raw]
        elif retrieved_passages_raw is not None and len(retrieved_passages_raw) > 0:
            retrieved_seq = [str(x) for x in retrieved_passages_raw]
        else:
            retrieved_seq = []
        retrieved_texts_seq = [str(x) for x in retrieved_passages_raw] if retrieved_passages_raw is not None else []

        missed: list[str] = []
        recalls_by_k: dict[int, float] = {1: 0.0, 2: 0.0, 5: 0.0, 10: 0.0, 20: 0.0}
        has_retrieval_labels = gold_docs_raw is not None and len(gold_docs_raw) > 0
        if gold_docs_raw is not None and len(gold_docs_raw) > 0:
            top_20_cids = [c for c in retrieved_seq[:20] if c.startswith("chunk-")]
            hydrated_by_cid = hooks.hydrate_top_chunks(top_20_cids, active_program_ref) if top_20_cids else {}
            n_ranks = min(20, max(len(retrieved_seq), len(retrieved_texts_seq)))
            ranked_docs = [
                hydrated_by_cid.get(retrieved_seq[i] if i < len(retrieved_seq) else "")
                or (retrieved_texts_seq[i] if i < len(retrieved_texts_seq) else "")
                or (retrieved_seq[i] if i < len(retrieved_seq) else "")
                for i in range(n_ranks)
            ]
            recalls_by_k = compute_retrieval_recalls(gold_docs_raw, ranked_docs, k_list=(1, 2, 5, 10, 20))
            if not ranked_docs and context_str.strip():
                ctx_recall = compute_retrieval_recalls(gold_docs_raw, [context_str], k_list=(1,))[1]
                for k_ctx in (5, 10, 20):
                    recalls_by_k[k_ctx] = max(recalls_by_k[k_ctx], ctx_recall)
            top_5_docs = ranked_docs[:5]
            for gd in gold_docs_raw:
                gd_str = str(gd).strip()
                if not any(doc_matches_gold(r, gd_str) for r in top_5_docs) and not (gd_str and gd_str in context_str):
                    first_line = gd_str.splitlines()[0].strip() if gd_str else ""
                    snippet = first_line[:120] if first_line else compute_mdhash_id(gd_str, prefix="chunk-")
                    missed.append(f"Missed passage: {snippet!r}")

        em_score: float | None = None
        f1_score: float | None = None
        if gold_answers is not None and len(gold_answers) > 0:
            em_score, f1_score = compute_qa_em_f1(pred_answer, gold_answers)
            if em_score < 1.0:
                missed.append(f"Predicted={pred_answer!r} != Expected={gold_answers!r}")

        usage_tokens = 0
        if hasattr(pred, "get_lm_usage") and callable(pred.get_lm_usage):
            usage_map = pred.get_lm_usage() or {}
            if isinstance(usage_map, (int, float)):
                usage_tokens = int(usage_map)
            elif isinstance(usage_map, dict):
                for model_usage in usage_map.values():
                    if isinstance(model_usage, dict):
                        usage_tokens += int(
                            model_usage.get("completion_tokens")
                            or model_usage.get("output_tokens")
                            or model_usage.get("total_tokens")
                            or 0
                        )
                    elif isinstance(model_usage, (int, float)):
                        usage_tokens += int(model_usage)
        if usage_tokens <= 0 and hasattr(active_program_ref, "telemetry") and active_program_ref.telemetry:
            usage_tokens = sum(
                int(entry.get("tokens_used", 0))
                for entry in active_program_ref.telemetry[-20:]
                if isinstance(entry, dict)
            )
        apm_hit: bool | None = None
        if gold_answers is not None and len(gold_answers) > 0:
            apm_hit = compute_answer_passage_match(gold_answers, context_str)

        gold_item_docs = getattr(gold, "item_docs", None)
        q_stats = hooks.query_rlm_stats(gold_item_docs)
        usage_tokens += int(q_stats.get("openie_tokens", q_stats.get("cached_openie_tokens", 0)))
        score_pred = compute_task_quality_score(
            recalls_by_k[5] if has_retrieval_labels else None,
            recall_at_1=recalls_by_k[1] if has_retrieval_labels else None,
            recall_at_2=recalls_by_k[2] if has_retrieval_labels else None,
            recall_at_10=recalls_by_k[10] if has_retrieval_labels else None,
            recall_at_20=recalls_by_k[20] if has_retrieval_labels else None,
            em=em_score,
            f1=f1_score,
            tokens=usage_tokens,
            synonym_judge_tokens=int(q_stats.get("synonym_judge_tokens", 0)),
            shacl_healer_tokens=int(q_stats.get("shacl_healer_tokens", 0)),
            shacl_violations_initial=int(
                q_stats.get(
                    "shacl_violations_initial",
                    q_stats.get("shacl_violations_before", 0),
                )
            ),
            shacl_violations_remaining=int(
                q_stats.get(
                    "shacl_violations_remaining",
                    q_stats.get("shacl_violations_after", 0),
                )
            ),
            missing_entity_defs_count=int(q_stats.get("missing_entity_defs_count", 0)),
            missing_entity_defs_tokens=int(q_stats.get("missing_entity_defs_tokens", 0)),
            pred_name=str(pred_name) if pred_name else None,
            missed_items=missed,
        )
        if apm_hit is not None:
            score_pred.answer_passage_match = 1.0 if apm_hit else 0.0
            score_pred.feedback = f"{score_pred.feedback} | AnswerPassageMatch={1 if apm_hit else 0}"
        if pred_name:
            stage_diag = _build_stage_diagnostic(
                str(pred_name),
                pred_trace,
                gold_answers,
                missed,
            )
            score_pred.feedback = f"{score_pred.feedback} | {stage_diag}"
        if record and pred_name is None:
            now_ts = time.time()
            dt_eval = max(1.0, now_ts - last_eval_ts)
            last_eval_ts = now_ts
            sc_val = float(score_pred.score)
            em_val = float(em_score or 0.0)
            f1_val = float(f1_score or 0.0)
            cand_prog = hooks.last_hook_program() or active_program_ref
            cand_fp = candidate_fingerprint(cand_prog)
            last_eval_fp = cand_fp
            eval_rec: dict[str, Any] = {
                "eval": eval_count,
                "cand_fp": cand_fp,
                "optimizer": opt_phase,
                "score": round(sc_val, 4),
                "r1": round(recalls_by_k[1], 4),
                "r2": round(recalls_by_k[2], 4),
                "r5": round(recalls_by_k[5], 4),
                "r10": round(recalls_by_k[10], 4),
                "r20": round(recalls_by_k[20], 4),
                "em": round(em_val, 4),
                "f1": round(f1_val, 4),
                "tokens": int(usage_tokens),
                "s_per_it": round(dt_eval, 2),
                "ts": round(now_ts, 2),
            }
            _record_candidate_eval(eval_rec)
            eval_rec["cand"] = candidate_stats[cand_fp]["cand"]
            eval_history.append(eval_rec)
            del eval_history[:-500]
            gepa_progress.metric_evaluated()
            if progress_callback is not None:
                progress_callback(
                    f"optimizing ({mode}, auto={auto}): eval #{eval_count} "
                    f"[cand #{eval_rec['cand']}] -> "
                    f"score={sc_val:.3f}, R@5={recalls_by_k[5]:.2f}, "
                    f"F1={f1_val:.2f}, EM={em_val:.2f}, dt={dt_eval:.1f}s, tok={usage_tokens}"
                )
        if record and pred_name is None:
            hooks.persist_trails(active_program_ref)
            _publish_optimization_audit()
        return score_pred

    def _mipro_metric(gold: Any, pred: Any, trace: Any = None) -> float | bool:
        res = _evaluate_prediction(gold, pred)
        if trace is not None:
            f1_val = float(getattr(res, "f1", 0.0) or 0.0)
            apm_val = float(getattr(res, "answer_passage_match", 0.0) or 0.0)
            r5_val = float(getattr(res, "recall_at_5", 0.0) or 0.0)
            return bool(f1_val >= 0.25 or apm_val > 0.0 or (r5_val >= 1.0 and f1_val > 0.0))
        return float(res.score)

    def _gepa_metric(
        gold: Any,
        pred: Any,
        trace: Any = None,
        pred_name: Any = None,
        pred_trace: Any = None,
    ) -> Any:
        _ = trace
        return _evaluate_prediction(
            gold,
            pred,
            pred_name=pred_name,
            pred_trace=pred_trace,
        )

    compiled = program
    # Let DSPy reuse identical requests on disk while keeping logical token
    # cost comparable across live and cached candidates. Avoid a RAM cache
    # during evolution; request keys and cache=False remain DSPy's own.
    if hooks.dspy_cache_dir is not None:
        configure_yaw_dspy_cache(
            hooks.dspy_cache_dir,
            enable_disk_cache=True,
            enable_memory_cache=False,
            restrict_pickle=True,
            account_cached_usage=True,
        )

    try:
        if mode in ("miprov2", "hybrid") and not resumed_seed and hasattr(dspy, "MIPROv2"):
            auto_val = auto if auto not in ("none", None, "") else None
            eff_bootstrapped_demos = (
                max_bootstrapped_demos
                if max_bootstrapped_demos is not None
                else getattr(
                    hooks.config,
                    "optimize_max_bootstrapped_demos",
                    0,
                )
            )
            eff_labeled_demos = (
                max_labeled_demos
                if max_labeled_demos is not None
                else getattr(
                    hooks.config,
                    "optimize_max_labeled_demos",
                    0,
                )
            )
            optimizer_kwargs: dict[str, Any] = {
                "metric": _mipro_metric,
                "prompt_model": resolved_ref_lm,
                "task_model": resolved_worker_lm,
                "max_bootstrapped_demos": eff_bootstrapped_demos,
                "max_labeled_demos": eff_labeled_demos,
                "num_threads": 1,
            }
            compile_kwargs: dict[str, Any] = {
                "trainset": list(trainset),
                "valset": list(valset) if valset else None,
            }
            if num_trials is not None or num_candidates is not None:
                optimizer_kwargs["auto"] = None
                optimizer_kwargs["num_candidates"] = num_candidates or 1
                compile_kwargs["num_trials"] = num_trials or 1
                compile_kwargs["minibatch"] = False
            elif mode == "hybrid" and valset and len(valset) >= 25:
                optimizer_kwargs["auto"] = None
                optimizer_kwargs["num_candidates"] = 3
                compile_kwargs["num_trials"] = 4
                compile_kwargs["max_bootstrapped_demos"] = eff_bootstrapped_demos
                compile_kwargs["max_labeled_demos"] = eff_labeled_demos
                compile_kwargs["minibatch"] = True
                compile_kwargs["minibatch_size"] = min(25, len(valset))
                compile_kwargs["minibatch_full_eval_steps"] = 4
            else:
                optimizer_kwargs["auto"] = auto_val or "light"
                if valset and len(valset) < 35:
                    compile_kwargs["minibatch"] = False

            baseline_demos = {
                name: list(getattr(pred, "demos", ()) or ()) for name, pred in compiled.named_predictors()
            }
            optimizer = dspy.MIPROv2(**optimizer_kwargs)
            if hasattr(optimizer, "_propose_instructions"):

                def _mipro_propose(
                    prog: Any,
                    tset: Any,
                    demo_candidates: Any = None,
                    view_data_batch_size: int = 10,
                    program_aware_proposer: bool = True,
                    data_aware_proposer: bool = True,
                    tip_aware_proposer: bool = True,
                    fewshot_aware_proposer: bool = True,
                    num_instruct_candidates: int = 1,
                ) -> dict[int, list[str]]:
                    _ = (
                        demo_candidates,
                        program_aware_proposer,
                        data_aware_proposer,
                        tip_aware_proposer,
                        fewshot_aware_proposer,
                    )
                    target_n = max(1, int(num_instruct_candidates or optimizer_kwargs.get("num_candidates") or 1))
                    batch_lim = max(1, min(len(tset), int(view_data_batch_size or 5)))
                    sample_records = [
                        {
                            "Inputs": dict(ex.inputs()) if hasattr(ex, "inputs") else {},
                            "Generated Outputs": dict(ex.labels()) if hasattr(ex, "labels") else {},
                            "Feedback": "[MIPROv2 RLM Dataset Context]",
                        }
                        for ex in list(tset)[:batch_lim]
                    ] or [{"Inputs": {}, "Generated Outputs": {}, "Feedback": "[MIPROv2 RLM Dataset Context]"}]
                    out_cands: dict[int, list[str]] = {}
                    for pred_i, (pred_name, pred_obj) in enumerate(prog.named_predictors()):
                        base_inst = str(getattr(getattr(pred_obj, "signature", None), "instructions", "") or "")
                        out_cands[pred_i] = [
                            base_inst,
                            *[
                                proposer.propose_for_component(
                                    component_name=pred_name,
                                    current_instruction=base_inst,
                                    component_examples=sample_records,
                                )
                                for _ in range(1, target_n)
                            ],
                        ]
                    return out_cands

                _mipro_propose_fn: Any = _mipro_propose
                optimizer._propose_instructions = _mipro_propose_fn
            compile_sig = inspect.signature(optimizer.compile).parameters
            if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in compile_sig.values()):
                compile_kwargs = {k: v for k, v in compile_kwargs.items() if k in compile_sig}
            with dspy.context(track_usage=True):
                compiled = optimizer.compile(
                    compiled,
                    **compile_kwargs,
                )
            prune_mismatching_demos(compiled)
            if eff_bootstrapped_demos == 0 and eff_labeled_demos == 0:
                for _, pred in compiled.named_predictors():
                    pred.demos = []
            else:
                for name, pred in compiled.named_predictors():
                    if not getattr(pred, "demos", None) and baseline_demos.get(name):
                        pred.demos = list(baseline_demos[name])

        if mode in ("gepa", "hybrid") and hasattr(dspy, "GEPA"):
            if gepa_seed_path is not None and not resumed_seed:
                gepa_seed_path.parent.mkdir(parents=True, exist_ok=True)
                seed_tmp = gepa_seed_path.with_name("seed_program.tmp.json")
                compiled.save(str(seed_tmp), save_program=False)
                seed_tmp.replace(gepa_seed_path)
            gepa_log_dir = (
                str(Path(save_path).parent / f"{Path(save_path).stem}_telemetry") if save_path is not None else None
            )
            gepa_valset = list(valset) if valset else None
            if gepa_log_dir is not None:
                state_bin = Path(gepa_log_dir) / "gepa_state.bin"
                if state_bin.exists():
                    from gepa.core.state import GEPAState

                    # Use GEPA's loader: its on-disk representation is a dict,
                    # not the object the previous preflight expected. Never
                    # delete a checkpoint to conceal an incompatibility.
                    saved_state = GEPAState.load(gepa_log_dir)
                    expected_val_len = len(gepa_valset or trainset)
                    if len(saved_state.prog_candidate_val_subscores[0]) != expected_val_len:
                        raise YawRuntimeError(
                            "INVALID_INPUT",
                            "GEPA checkpoint validation set changed; start a fresh run. The checkpoint was preserved.",
                        )
                    gepa_progress.restore_state(saved_state, expected_val_len)

            class _GepaProgressCallback:
                def on_optimization_start(self, event: Any) -> None:
                    if progress_callback is not None:
                        progress_callback(
                            f"GEPA started (train={event.get('trainset_size')}, val={event.get('valset_size')})"
                        )

                def on_iteration_start(self, event: Any) -> None:
                    if progress_callback is not None:
                        progress_callback(f"GEPA iteration {event.get('iteration')}: selecting candidate...")

                def on_proposal_start(self, event: Any) -> None:
                    if progress_callback is not None:
                        comps = ",".join(str(c) for c in (event.get("components") or ()))
                        progress_callback(f"GEPA iteration {event.get('iteration')}: RLM reflecting on [{comps}]...")

                def on_proposal_end(self, event: Any) -> None:
                    if progress_callback is not None:
                        comps = ",".join(str(c) for c in (event.get("new_instructions") or {}))
                        progress_callback(
                            f"GEPA iteration {event.get('iteration')}: proposed new [{comps}], scoring subsample..."
                        )

                def on_valset_evaluated(self, event: Any) -> None:
                    nonlocal last_event_eval_count
                    # The valset pass that triggered this event scored the candidate whose
                    # fingerprint was seen last; skip attribution if no new evals ran (resumed state).
                    if last_eval_fp is not None and eval_count > last_event_eval_count:
                        cs = candidate_stats.get(last_eval_fp)
                        if cs is not None:
                            cs["gepa_idx"] = int(event.get("candidate_idx", -1))
                            cs["gepa_iteration"] = int(event.get("iteration", 0))
                            cs["val_score"] = round(float(event.get("average_score", 0.0)), 4)
                            cs["is_best"] = bool(event.get("is_best_program"))
                    last_event_eval_count = eval_count
                    if progress_callback is not None:
                        progress_callback(
                            f"GEPA iteration {event.get('iteration')}: candidate #{event.get('candidate_idx')} "
                            f"val_score={float(event.get('average_score', 0.0)):.4f} "
                            f"(best={event.get('is_best_program')})"
                        )

            gepa_kwargs: dict[str, Any] = {
                "metric": _gepa_metric,
                "reflection_lm": proposer.orchestrator_lm or proposer.worker_lm,
                "instruction_proposer": proposer,
                # 3 examples let single lucky answers accept/reject candidates; 10 is the floor.
                "reflection_minibatch_size": 10,
                "candidate_selection_strategy": "pareto",
                "skip_perfect_score": True,
                "use_merge": True,
                "component_selector": ActiveTraceComponentSelector(compiled),
                "num_threads": 1,
                "track_stats": True,
                "track_best_outputs": True,
                "warn_on_score_mismatch": False,
                "log_dir": gepa_log_dir,
                "seed": 0,
                "gepa_kwargs": {
                    "use_cloudpickle": True,
                    "frontier_type": "hybrid",
                    "callbacks": [CompositeCallback([_GepaProgressCallback(), gepa_progress])],
                },
            }
            if num_trials is not None:
                gepa_kwargs["max_metric_calls"] = num_trials
                gepa_kwargs["auto"] = None
            else:
                gepa_kwargs["auto"] = auto if auto in ("light", "medium", "heavy") else "light"

            configure_simba_task_learning(
                proposer,
                compiled,
                trainset,
                lambda gold, pred: float(_evaluate_prediction(gold, pred, record=False).score),
                hooks,
                save_path=save_path,
                resume=resumed_seed,
            )
            gepa = dspy.GEPA(**gepa_kwargs)
            opt_phase = "gepa"
            with dspy.context(track_usage=True):
                compiled = gepa.compile(
                    compiled,
                    trainset=list(trainset),
                    valset=gepa_valset,
                )
    finally:
        hooks.clear_progress_callback()
        if hooks.dspy_cache_dir is not None:
            configure_yaw_dspy_cache(
                hooks.dspy_cache_dir,
                enable_disk_cache=True,
                enable_memory_cache=True,
                restrict_pickle=True,
                account_cached_usage=True,
            )

    assert isinstance(compiled, HippoRAGProgram)
    active_program_ref = compiled

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        compiled.save(str(p), save_program=False)
        compiled.loaded_from_path = p.resolve()
        pareto_programs = select_pareto_frontier_programs(compiled, max_programs=3)
        telem_dir = p.parent / f"{p.stem}_telemetry"
        if telem_dir.exists():
            for stale_member in telem_dir.glob("pareto_member_*.json"):
                with contextlib.suppress(OSError):
                    stale_member.unlink(missing_ok=True)
        if len(pareto_programs) > 1:
            telem_dir.mkdir(parents=True, exist_ok=True)
            for idx, sibling in enumerate(pareto_programs[1:], start=1):
                sibling.save(str(telem_dir / f"pareto_member_{idx}.json"), save_program=False)

    hooks.finalize_compiled_program(compiled)
    hooks.rehydrate_sandbox_after_optimization(compiled)

    if save_path is not None:
        p = Path(save_path)
        if proposer.telemetry:
            telemetry_path = p.parent / f"{p.stem}_proposer_telemetry.json"
            write_text_atomic(
                telemetry_path,
                json.dumps(proposer.telemetry, ensure_ascii=False, indent=2),
            )
        audit_path = p.parent / f"{p.stem}_graph_rlm_audit.json"
        audit_payload = {
            "graph_rlm_stats": dict(hooks.graph_rlm_stats()),
            "candidate_records": _candidate_records(),
            "eval_records": eval_history[-500:],
            "gepa_progress": gepa_progress.snapshot(),
            **compiled.dump_audit_trails(bounded=False, telemetry_key="program_telemetry"),
            "proposer_simba_telemetry": list(getattr(proposer, "simba_telemetry", ())),
        }
        write_text_atomic(audit_path, json.dumps(audit_payload, ensure_ascii=False, indent=2))
        if valset:
            eval_path = p.parent / f"{p.stem}_eval.json"
            eval_target = compile_pareto_ensemble(compiled, program_path=p)
            evaluate_program(eval_target, valset, save_as_json=eval_path)

    return compiled


__all__ = [
    "ActiveTraceComponentSelector",
    "GepaProgress",
    "GraphOptimizationHooks",
    "candidate_fingerprint",
    "compile_pareto_ensemble",
    "configure_simba_task_learning",
    "embed_fingerprint",
    "evaluate_program",
    "evolve_program_prompts",
    "extract_named_trace_entries",
    "extraction_fingerprint",
    "openie_fingerprint",
    "prune_mismatching_demos",
    "query_retrieval_fingerprint",
    "reduce_pareto_predictions",
    "select_pareto_frontier_programs",
    "serialize_predictor_surface",
    "shacl_fingerprint",
    "synonym_fingerprint",
]
