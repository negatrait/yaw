"""Checks for the active optimizer's scoring and workspace boundaries."""

import dspy
import pytest
from filelock import FileLock

from yaw.core import YawConfig, YawRuntimeError
from yaw.optimize import optimize_dspy, optimize_optuna, qa_gepa_metric


def test_gepa_scores_inline_citations_against_react_observations():
    text = "Neil Armstrong commanded Apollo 11."
    gold = dspy.Example(gold_answers=["Neil Armstrong"], gold_docs=[text])
    history = {
        "messages": [
            {
                "tool_calls": {
                    "tool_calls": [{"id": "search", "name": "search_passages"}],
                    "tool_call_results": [
                        {"tool_call_id": "search", "value": f"[chunk_id=42] (apollo.md:1-2) score=1.000:\n{text}"}
                    ],
                }
            }
        ]
    }
    grounded = qa_gepa_metric(gold, dspy.Prediction(answer="Neil Armstrong [chunk_id=42]", history=history))
    invented = qa_gepa_metric(gold, dspy.Prediction(answer="Neil Armstrong [chunk_id=99]", history=history))
    assert grounded.score == pytest.approx(1.0)
    assert invented.score == pytest.approx(0.85)
    assert "citation_grounding=0.00" in invented.feedback


def test_optimizers_reject_held_out_overlap_before_indexing(tmp_path):
    cfg = YawConfig(workspace_root=tmp_path)
    for optimize in (optimize_dspy, optimize_optuna):
        with pytest.raises(YawRuntimeError, match="held-out"):
            optimize(cfg, datasets="2wikimultihopqa", query_offset=0)
    assert not list(tmp_path.rglob("*.db"))


def test_gepa_rejects_competing_writer(tmp_path):
    cfg = YawConfig(workspace_root=tmp_path)
    lock_path = cfg.for_optimization().lock_path
    assert lock_path is not None
    lock_path.parent.mkdir(parents=True)
    with FileLock(str(lock_path), timeout=0), pytest.raises(YawRuntimeError, match="BUSY"):
        optimize_dspy(cfg, datasets="2wikimultihopqa", query_offset=10)
