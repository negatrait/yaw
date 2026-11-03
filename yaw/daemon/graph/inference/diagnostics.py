"""Recent diagnostic records, separate from DSPy's optimization traces and usage."""

import functools
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import dspy


class DiagnosticHistory(list):
    """List-compatible working history; candidate copies start with no diagnostics.

    Trim only at operation boundaries: callers use lengths to collect new records.
    These records are observational, never predictor parameters or GEPA traces.
    """

    def __deepcopy__(self, memo: dict[int, Any]) -> DiagnosticHistory:
        result = DiagnosticHistory()
        memo[id(self)] = result
        return result


def trim_diagnostics(owner: Any) -> None:
    """Retain recent diagnostics after their current operation has consumed them."""
    for name, limit in (
        ("telemetry", 250),
        ("embedding_audit_trail", 250),
        ("entity_def_audit_trail", 250),
        ("label_judge_audit_trail", 100),
        ("hierarchy_judge_audit_trail", 100),
        ("related_judge_audit_trail", 100),
        ("shacl_audit_trail", 100),
        ("simba_telemetry", 100),
    ):
        records = getattr(owner, name, None)
        if isinstance(records, list):
            del records[:-limit]


def diagnostic_boundary(fn: Callable) -> Callable:
    """Release old diagnostics after success or failure, outside accounting slices."""

    @functools.wraps(fn)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return fn(self, *args, **kwargs)
        finally:
            trim_diagnostics(self)

    return wrapped


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


def judge_stage_fingerprint(
    program: Any,
    stage_name: str,
) -> str:
    """Compute SHA-256 fingerprint for a specific synonym judge stage predictor surface."""
    named = dict(program.named_predictors()) if hasattr(program, "named_predictors") else {}
    payload = serialize_predictor_surface(named[stage_name]) if stage_name in named else {}
    payload["stage_name"] = stage_name
    payload["context_budget"] = [
        getattr(program, "context_window_tokens", 8192),
        getattr(program, "max_output_tokens", 1024),
    ]
    payload["schema_version"] = "skos-shacl-v3"
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def label_judge_fingerprint(program: Any) -> str:
    """Compute SHA-256 fingerprint for Stage 1 label_judge predictor surface."""
    return judge_stage_fingerprint(program, "label_judge")


def hierarchy_judge_fingerprint(program: Any) -> str:
    """Compute SHA-256 fingerprint for Stage 2 hierarchy_judge predictor surface."""
    return judge_stage_fingerprint(program, "hierarchy_judge")


def related_judge_fingerprint(program: Any) -> str:
    """Compute SHA-256 fingerprint for Stage 3 related_judge predictor surface."""
    return judge_stage_fingerprint(program, "related_judge")


def extract_named_trace_entries(
    program: Any,
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
