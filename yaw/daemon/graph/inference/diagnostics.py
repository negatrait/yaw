"""Recent diagnostic records, separate from DSPy's optimization traces and usage."""

import functools
from collections.abc import Callable
from typing import Any


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
        ("synonym_audit_trail", 100),
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
