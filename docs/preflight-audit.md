# Preflight audit — 2026-10-23

Scope: the designated Windows checkout, its upgraded dependencies, Python
3.14.9, and the complete offline test suite. This is a developer verification
audit, not a new corpus benchmark or proof of unattended optimizer stability.

## Completed repair

The complete gate now passes: Ruff 0 findings, ty 0 diagnostics, **261 passed,
0 skipped**, with testmon selection disabled. ResourceWarnings are eliminated;
23 Pydantic/DSPy API/metadata notices remain unsuppressed. The baseline counts
and failures below are historical. See [quality repair](quality-repair.md) and
[evaluation section 39](evaluation.md#39-complete-ruff-ty-and-pytest-repair-2026-10-23).

## Available commands

```powershell
uv sync --locked
uv run preflight
uv run preflight --full
uv run preflight -- --lf
uv run typecheck
uv run ruff check .
uv run ruff format .
uv run pytest
```

Preflight applies Ruff formatting, safe Ruff fixes, and ty fixes before pytest.
Ruff's `fix = true` and pytest's `addopts = ["--testmon"]` are project defaults.
The installed ty supports `--fix`, but rejects `fix = true` as configuration;
the `typecheck` entry point supplies that CLI flag without shadowing ty itself.
Fixes do not imply all diagnostics are automatically repairable. No rules are
disabled and no suppression comments are added.

Coverage uses **coverage.py 7.17.0 + pytest-cov 7.1.0**, with testmon 2.2.0.
SlipCover is removed from the development dependencies and lock. Testmon has
explicit pytest-cov integration that merges its per-test line data into the
coverage report. Branch coverage is not enabled: this integration supports line
coverage. `COVERAGE_CORE=ctrace` is selected because Python 3.14's default sysmon
core cannot supply the per-test dynamic contexts required by testmon.

Every run gets its own directory under `outputs/preflight/`; `latest.json` points
to its Markdown and JSON summaries. Full diagnostics, commands, tool versions,
exit codes, timings, JUnit and coverage JSON are retained. The console shows
progress and a short summary; `--verbose` streams complete diagnostics. Checks
run serially, continue after ordinary failures and return nonzero on failure.
The default per-check timeout is 1,800 seconds; `--timeout 0` disables it.

Incremental coverage measures only this invocation's selected tests, including
zero selected tests when everything is unchanged. It is never presented as
full-suite coverage or merged with stale results. Python subprocess execution
is measured through coverage.py's native subprocess patch in `.coveragerc`.
Coverage includes `yaw/`, `benchmark.py` and all production Python in `scripts/`.
`--full` disables testmon deselection while
updating its database. A lock-file hash separates dependency environments.
Testmon tracks in-process dependencies; code used only in child workers and
external input files can escape that dependency graph. Use `--full` as the
final gate before unattended optimization, especially after worker changes.
The initial audit required `--live`. The subsequent complete-gate repair
supersedes that policy: every collected test runs, including live inference, and
unavailable prerequisites fail explicitly. `--live` remains a compatibility flag.

## Initial findings and repair order

The measured static baseline, including the installed tools' available fixes,
was **602 Ruff findings** and **67 ty diagnostics** under the original
100-character line limit. The user subsequently selected 120 characters; those
counts are historical, not the current remaining count. Ruff formatting alone does
not repair long strings/comments. ty reported zero automatic fixes for these
particular errors.

The full offline run on Python 3.14.9 finished with **230 passed, 7 failed,
2 skipped**, in 624.29 seconds of pytest time (630.74 seconds for its process).
Coverage.py measured **7,916 / 9,882 executable lines = 80.11%**. This includes
the failing tests' executed paths; it is not a successful acceptance gate.
The full artifacts are in
`outputs/preflight/20261023T114056Z-53964/`.

### Fix the seven test/contract failures first

| Failure group | Cause observed | Repair and verification |
| --- | --- | --- |
| `test_engine.py::test_sprint3_engine_evaluate_and_pareto_frontier_export` | The test constructs an engine without config; reflection setup dereferences `self.config.base_url`. | Supply a verified loopback counter through the existing explicit seam for orchestration tests. Also make missing production config produce a deliberate configuration error, rather than `AttributeError`. Preserve the resident-tokenizer requirement. |
| Five optimization/resume tests in `test_runtime.py` | Their in-process `test-runtime-lm` is not resident at the configured inference endpoint. The new tokenizer preflight rejects it. | Add real loopback health/props/template/tokenize endpoints and configure those tests to use them, or propagate an explicit tokenizer dependency through a supported runtime seam. Keep validation enabled; do not replace it with a character estimate in production. Prefer native DSPy optimizers over incompatible assigned stubs. Rerun CLI resume, custom output, split isolation and reload assertions. |
| `test_benchmark.py::test_maxp_passage_hydration_and_retrieval_recall_parity_on_split_json_passages` | `MinimalProgram` lacks `context_window_tokens`; fingerprinting also requires `max_output_tokens`. | Use the real program or a complete declared program contract, including inference capacities. Retain the passage hydration/Recall assertions and real SQLite. This failure occurs before retrieval scoring, so it establishes no retrieval regression. |

The five runtime test names are
`test_cli_resume_real_gepa_checkpoint_and_custom_output`,
`test_optimize_and_reload_compiled_program_with_composite_fitness`,
`test_cli_optimize_ergonomics_and_anti_overfitting_gate`,
`test_cli_optimize_multi_dataset_interleaved_splits`, and
`test_cli_dspy_optimize_sandbox_isolation_and_status`.
All seven repairs are now verified by the complete gate recorded in section 39.

The suite also emitted 53 warnings. ResourceWarnings include unclosed SQLite
connections: trace their allocations and fix fixture/runtime ownership before
turning them into strict gates; the warnings alone do not establish a production
leak. DSPy warns that OpenAI-style `messages=` calls are scheduled for removal in
3.5. Migrate the central LM adapters to the native normalized request API before
upgrading to that release. Do not suppress either warning class to make a report
look clean. The current lock retains the measured DSPy 3.4.1.

| Order | Work packet | Observed scope and exit condition |
| --- | --- | --- |
| 1 | Real runtime type contracts (completed in `cd167ae`) | 14 diagnostics in graph engine/inference, plus one watchdog return annotation. Declare LM capability fields, use native `qa.predict` attributes, narrow SIMBA's compiled module, validate checkpoint/evidence mappings and missing LMs, and type mixed fingerprint payloads. Verified: runtime ty passes and 65 focused native/SQLite/recovery checks pass; see `runtime-contracts.md`. |
| 2 | Typed test fixtures and metric arguments (completed) | 52 diagnostics in tests; 43 are in `test_prompts.py`. Its dynamic DummyLM attributes and three untyped metric-kwargs expansions cause most of these. Use a capability-declaring native DummyLM subclass and explicit or precisely typed metric arguments. Match HTTP-handler override signatures, use a declared callable counter object, narrow optional paths, and replace incompatible optimizer assignment with a supported test seam. Exit: tests pass ty without hiding diagnostics. |
| 3 | Three semantic lint findings (completed) | Two unused loop bindings (`B007`) and one unnecessary `.keys()` membership (`SIM118`), all in `engine.py`. Rename/remove unused bindings and use direct dictionary membership; retain behavior. Exit: these three findings disappear. |
| 4 | Long-line cleanup (completed at 120 characters) | 599 `E501` findings across 34 files. Highest counts: engine tests 71, parity-gate script 68, storage DB 68, inference program 55, graph engine 43, templates 42. Wrap comments, signatures and SQL/prompt literals using adjacent strings. Compare parsed string values before/after so user prompt edits and SQL semantics survive. Exit: full Ruff pass at the current 120-character limit, followed by affected tests. |

Do runtime contracts before broad literal wrapping so compatibility problems do
not get buried in a large formatting diff. Keep the user-selected 120-character policy and enabled rules; do not disable
rules to conceal remaining findings.

## Corrections already made

- The architecture stdlib-only test used an incomplete hand-written list. It
  now uses `sys.stdlib_module_names`, preserving the third-party import ban.
- The NER duplicate test previously supplied two different definitions while
  expecting the exact-duplicate guard to reject them. The duplicate fixture now
  repeats the definition; a separate Mercury homograph case verifies that
  distinct provisional senses remain permitted pending synonym adjudication.
- The initial run skipped resident-model tests without `--live`. The subsequent
  complete-gate repair removes that skipping and replaces obsolete history
  assertions with observations through native DSPy callbacks.
- A Windows redirected-output encoding bug found during runner development was
  fixed; diagnostics and logs use UTF-8. Only this task's verified audit process
  trees were stopped during the coverage-tool/interpreter migration.

## Verification record

The runner's real integration test creates a disposable project, invokes the
actual formatter, linter, typechecker, pytest-cov and testmon, and verifies that
static failures do not prevent tests or reports. It then repairs the real files,
checks a successful incremental rerun, and confirms a third unchanged run
selects zero tests. Executed source lines must appear in coverage data.

The first runner/architecture verification passed 10 tests on Python 3.14.9;
the extended no-change/coverage regression passed independently. The new runner,
its test and conftest pass Ruff and ty. Full-suite outcome and measured coverage
are also recorded in `docs/evaluation.md`.

The raw JUnit suite header said 242 tests, but contained 239 unique test-case
records, matching pytest's 230 + 7 + 2 terminal result. Preflight now derives
counts from those records. The original XML is retained. Its slowest cases were
candidate-stage restart reuse (136.83 s), response-cache process restart and
invalidation (105.91 s), and native optimizer orchestration (53.78 s). Preserve
these physical recovery checks; testmon avoids rerunning them when unaffected.

Python installation note: global uv 0.13.0 did not know 3.14.9. An isolated
current uv downloaded the official managed interpreter; global uv was left in
place. `.python-version` and `.venv` use 3.14.9; the current project compatibility
requirement is `>=3.14`.
The dependency lock was updated deliberately for coverage.py/pytest-cov and the
pinned interpreter. No new production runtime dependency was added for preflight.
