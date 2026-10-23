"""Diagnostic retention must not duplicate history in optimization candidates."""

import copy

import dspy
import pytest

from yaw.daemon.graph.inference import HippoRAGProgram
from yaw.daemon.graph.inference.diagnostics import (
    DiagnosticHistory,
    diagnostic_boundary,
    trim_diagnostics,
)


def test_candidate_copy_keeps_predictors_but_drops_diagnostic_payloads() -> None:
    program = HippoRAGProgram()
    predictor = program.qa.predict
    predictor.signature = predictor.signature.with_instructions("Keep this candidate instruction.")
    predictor.demos = [dspy.Example(context="source", question="q", answer="a", reasoning="r")]
    program.telemetry.append({"payload": "x" * 100_000})
    program.entity_def_audit_trail.append({"missing_count": 3, "tokens": 42})
    clone = copy.deepcopy(program)
    cloned_predictor = clone.qa.predict
    assert cloned_predictor.signature.instructions == "Keep this candidate instruction."
    assert cloned_predictor.demos[0].answer == "a"
    assert not clone.telemetry and not clone.entity_def_audit_trail
    assert program.telemetry and program.entity_def_audit_trail
    clone.telemetry.append({"tokens_used": 17})
    assert program.telemetry[0] != clone.telemetry[0]


def test_boundaries_preserve_current_operation_slices_and_trim_on_failure() -> None:
    class Operation:
        telemetry = DiagnosticHistory()

        @diagnostic_boundary
        def run(self, fail: bool = False) -> int:
            previous = len(self.telemetry)
            self.telemetry.extend({"tokens_used": 7} for _ in range(300))
            total = sum(row["tokens_used"] for row in self.telemetry[previous:])
            if fail:
                raise ValueError("failed after producing diagnostics")
            return total

    operation = Operation()
    for _ in range(20):
        assert operation.run() == 2100
        assert len(operation.telemetry) == 250
    with pytest.raises(ValueError):
        operation.run(fail=True)
    assert len(operation.telemetry) == 250
    trim_diagnostics(operation)
