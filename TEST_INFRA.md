# YAW verification contract for coding agents

**Status:** Selected test plan; actual results belong in [TEST_READY](TEST_READY.md).

**Authority:** [FOUNDATIONAL](FOUNDATIONAL.md), [Epic 7.8 packets](docs/epic7.8-open-ie-ontology-alignment.md#delivery-packets), [E78 scenarios](docs/USER_STORIES.md).

## Entry commands

Run from the user-designated repository, currently `C:\Users\Harri.Ahokas\Documents\yaw`. Inspect staged and unstaged changes before writing:

```text
git rev-parse --show-toplevel
git remote get-url origin
git rev-parse HEAD
git status --short
git diff --stat
git diff --cached --stat
```

Confirm the root and `negatrait/yaw` remote. An attached workspace does not override this repository. If Git reports ownership differences, use a per-command `-c safe.directory=<verified-root>` for this explicitly trusted repository; do not silently change global Git configuration.

For implementation work, establish a separate platform environment. In PowerShell:

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".cache/venv-windows"
$env:PYTHONPATH = Join-Path (Get-Location) "src"
uv sync --locked --python 3.14
```

On Unix, use the same checkout contracts with `UV_PROJECT_ENVIRONMENT=.cache/venv-unix` and `PYTHONPATH="$PWD/src"`, then the same sync command. The explicit source path is required for source-checkout commands while [pyproject.toml](pyproject.toml) has `package = false`; it is not a packaging verification result.

Record the interpreter, locked dependency versions, configured model digests, effective endpoint, resource hashes and working-tree identity. `--locked` fails instead of updating an inconsistent lock; fix and document an intended metadata/lock change rather than silently upgrading. [uv locking](https://docs.astral.sh/uv/concepts/projects/sync/).

Before process/lifecycle tests, replace the known Windows `os.kill(pid, 0)` probe with a non-terminating platform implementation. Use temporary vaults/databases. An unavailable dependency/platform leaves its gate open; do not run unsafe lifecycle code to demonstrate a known failure.

## Test-target map

These are delivery targets, not pass claims. **EXTEND** means the file exists but its current tests do not certify the new contract; **NEW** means create it in the owning packet. Preserve independent expected outcomes while replacing obsolete assertions. Every implemented test identifies its E78 obligation in its name or docstring; TEST_READY records exact collected node IDs, including parameter cases. A scenario with several clauses is complete only when all clauses and controls are covered.

| Acceptance | Test owner / delivery | Required boundary |
| --- | --- | --- |
| E78-01, E78-02 | EXTEND `tests/test_pluggable_ontology.py`; 7.8.4–5 | Real source/resource changes and open/scoped projections |
| E78-03, E78-06 | EXTEND `tests/test_leaf_pipeline.py`; 7.8.4 | One real parser, independent records, qualification and failure/empty distinctions |
| E78-04 | EXTEND `tests/test_provenance.py`; 7.8.4 | Original Unicode/CRLF bytes, overlapping/repeated occurrences and stale evidence |
| E78-05 | EXTEND `tests/test_canonicalization.py`, `tests/test_rdf_substrate.py`; 7.8.5 | Candidate versus accepted identity; actual RDF kinds and binding removal |
| E78-07 | EXTEND `tests/test_cocoindexer.py`; start 7.8.3, finish 7.8.4–5 | Actual shared CocoIndex inputs/targets, compatible cache reuse and invalidation |
| E78-08, E78-18 | EXTEND `tests/test_shacl_lint.py`, `tests/test_owl_ontology.py`, `tests/test_compile_shacl_cache.py`; start 7.8.3, finish 7.8.5 | Selected RDFLib/pySHACL sequence, rule interaction, findings, provenance and retraction |
| E78-09, E78-15 | NEW `tests/test_compiler_lifecycle.py`; 7.8.5 | Production compiler, small/grouped inputs, source/resource mutation and publication |
| E78-10 | EXTEND `tests/test_storage.py`, `tests/test_representation_spaces.py`; 7.8.4 and 7.8.6 | Actual legacy database migration, ID mappings, vector spaces and rollback |
| E78-11 | EXTEND `tests/test_retriever.py`; 7.8.5 | Real vectors/matrices, known numerical outputs, candidate union and counterevidence |
| E78-12 | EXTEND `tests/test_yaw.py`; NEW `tests/test_materialize.py`; 7.8.5 | Real FastMCP client, same production handlers and stored-block rendering |
| E78-13 | EXTEND `tests/test_cli_lifecycle.py`; safe probe in 7.8.3, full native checks in 7.8.6 | Native Windows and Unix paths, locks, startup/shutdown |
| E78-14 | NEW `eval/pilot.py`, `eval/manifests/epic78-pilot.json`; 7.8.5 | Actual configured models; independent labelled positive/negative outcomes |
| E78-16 | NEW `tests/test_claim_comparison.py`; 7.8.5, plus actual-model pilot | April/May/supplier fixture and independent scope/unit/authority controls |
| E78-17 | NEW `tests/test_compiler_failures.py`; 7.8.5 | Faults at real boundaries and evidence that no alternate implementation ran |
| E78-19 | EXTEND `tests/test_leaf_pipeline.py`, `tests/test_provenance.py`; 7.8.4, plus actual-model pilot | Original non-English/mixed text, exact evidence and explicit identity support |
| E78-20 | NEW compiler lifecycle/materializer tests above; EXTEND `tests/test_yaw.py`; 7.8.5 | Membership invalidation, superseded publication, scoped states and consumer agreement |

Existing loopback-fixture support lives in `tests/conftest.py`; registry/transport checks extend `tests/test_spacy_llm_registry.py`. Do not duplicate production schemas/DDL in fixtures. Use the production schema owner for normal databases; an explicitly versioned legacy migration fixture may retain old DDL to test migration.

Source-specific, independently authored model replies may isolate HTTP transport. Test positive/negative/partial/failure controls. Never replace the parser, database, compiler, semantic engine or scorer being tested, and never treat adjacent capitalized names chained with “relates to” as semantic ground truth.

## Focused commands

Use paths from the map after delivering the corresponding tests. For example, 7.8.4 extraction/grounding work:

```text
uv run --locked pytest --no-testmon --collect-only -q tests/test_leaf_pipeline.py tests/test_provenance.py
uv run --locked pytest --no-testmon -q tests/test_leaf_pipeline.py tests/test_provenance.py
```

The checkout enables testmon in pytest `addopts`. `--no-testmon` disables it even when configured by default; selected or zero-test success is not proof of a gate. Check collection counts and deselections as well as the exit code. [testmon plugin options](https://github.com/tarpas/pytest-testmon/blob/main/testmon/pytest_testmon.py).

If a planned test is absent, implement it; do not substitute another test file and claim the obligation passed. A meaningful failure assertion observes stored state, evidence or the component result, not a mock returning the expected value.

## Completion commands

After all required files exist, run these against the same working-tree identity. Do not include `-k`, `-m`, `--lf`, testmon selection or other subset filters in the full regression command:

```text
uv run --locked pytest --no-testmon --collect-only -q tests
uv run --locked pytest --no-testmon -ra tests
uv run --locked ruff check src tests eval tools
uv run --locked ruff format --check src tests eval tools
uv run --locked ty check
uv run --locked python eval/pilot.py --manifest eval/manifests/epic78-pilot.json
python tools/check_docs.py
git diff --check
```

`eval/` and the pilot are required 7.8 deliverables; their absence is not a reason to omit them at handoff. Record required-case skips/xfails as incomplete, even with exit 0. Run native platform gates on each claimed platform. Loopback pytest results and actual-model results are separate evidence categories.

## Actual-model pilot

Deliver `eval/pilot.py` in 7.8.5. Its command above is **PLANNED until implemented**. Run it as a separate process, not through pytest: the current session-autouse fixture in `tests/conftest.py` replaces `YAW_HOST` with a synthetic server. The pilot must import production entry points, verify its effective endpoint/model digests against the manifest, and never load pytest fixtures or canned generation outputs.

Before the first prediction, version the input source fixtures, labels and manifest. The manifest fixes case IDs, required E78 outcomes, accepted supporting/opposing source IDs, permitted equivalent phrasings, model/resource/configuration identity and output-report path. It includes positive extraction, a valid empty case, polarity/direction, literal/definition grounding, explicit and ambiguous identity, the deadline-revision case, non-English/mixed-language evidence and supported/contested synthesis. Labels must not be copied from the engine's own predictions.

The runner writes per-case observations, expected outcomes, required-outcome pass/fail, model calls/tokens and elapsed time to the manifest's report path. Exit nonzero on a failed required sentinel or an execution error; keep those two causes distinct. Save all failures rather than silently rerunning until green. An all-empty/all-unknown output fails positive cases. The 8.2 benchmark runner reuses this support; no benchmark superiority is claimed by passing a small pilot.

## Run record

Append/update the current packet's record in TEST_READY using these fields:

```text
Package / step and acceptance IDs:
Repository / HEAD / working-tree hash manifest (including relevant untracked files):
Platform / Python / lock hash / model digests / resource and config hashes:
Exact commands and collected test node IDs / counts:
Results: passed, failed, skipped, xfailed, deselected; exit codes:
Pilot input/label manifest and per-case report (if applicable):
Retired symbols/flags -> replacement owner -> call-site audit evidence:
Remaining failures or unrun obligations:
Next packet / step:
```

A pass record needs retained evidence, not only a branch name or an old test total. Required failures keep the packet open. Do not weaken an expected result or add an implementation branch to clear the gate.

## Documentation checks

Run `python tools/check_docs.py` from the repository root with Python 3.10 or newer. It is a standard-library check of local Markdown targets/heading anchors, portable hyperlinks, active table widths, the E78 inventory/test-map coverage and the epic package inventory. It does not import YAW, call models, fetch URLs, validate source line anchors, assess prose semantics or certify that planned tests exist. Historical evidence/clippings have reduced semantic/table checks; their file links still resolve. Manual review must check owner agreement, dependency order and obsolete instructions.

Run `git diff --check` separately. Documentation-only changes do not require application tests or model runs. Keep the documentation results and runtime results separate in TEST_READY; a temporary script outside the repository is not a reusable verification entry point.
