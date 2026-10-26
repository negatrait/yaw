# Complete developer-gate repair

This work closes developer-check failures in the first unfinished Epic 1 packet,
E1-S6. It does not replace the separate Tier 2 benchmark acceptance gate.

Exit condition: full-repository Ruff and ty checks pass, and every collected
pytest case executes successfully with no skips, xfails or testmon deselection.
Final verification runs serially with testmon selection disabled. Resident-model
tests use the configured external service; unavailable prerequisites fail clearly.

Restart checklist:

- [x] Repair typed fixtures and malformed runtime/optimizer test setup.
- [x] Wrap source literals without changing prompt, grammar or SQL values.
- [x] Remove automatic and conditional test skipping; verify live contracts.
- [x] Repair all failures revealed by the complete suite.
- [x] Run complete static/test gates and record actual results in evaluation.md.

Initial measured baseline: 52 ty diagnostics and 218 Ruff findings at the
user-selected 120-character limit. The work began from clean commit 2e9eded.
Tracked source snapshots for literal-value comparison are under ignored
outputs/quality-repair/baseline/.

The first full pass executed 259 cases with no skips: 257 passed and two failed
because the new deterministic fixture incorrectly declared GBNF unsupported.
Restoring its original unspecified capability fixed both unchanged assertions.
That run exposed native diskcache thread-local SQLite connection leaks. YAW now
closes construction-time and per-operation thread handles without deleting disk
entries; real two-worker tests cover both logical-usage modes and persistence.
The native-cache control reproduces two ResourceWarnings, while the managed
worker tests and CLI-resume/grammar regressions pass with ResourceWarnings
escalated to errors. Final full preflight remains the completion gate.

Completed: final full preflight passed Ruff, ty and all 261 tests with no skipped
or deselected cases. No ResourceWarnings remain. There are 23 unsuppressed
Pydantic/DSPy API/metadata notices. Coverage is 81.75%; excluded lines: zero.
The detailed result and failure history are in evaluation.md section 39.
Final artifacts: outputs/preflight/20261023T164646Z-31068/.
