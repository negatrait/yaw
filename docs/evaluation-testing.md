# Evaluation Record: Testing, Preflight & Production Coverage

Domain evaluation ledger for Python 3.14.9 toolchain, `uv run preflight`, Ruff/ty/pytest repairs, and 100% production coverage packets. See the [Evaluation Index](evaluation.md) for all other domain ledgers.

## 50. Context-Routed `.agents/rules`, Subagent Mounting & `docs/` Consolidation (2026-10-31)

- **Scope:**
  - **Pointer-Only Glob Rules (`.agents/rules/{yaw,daemon,graph-storage,inference,tests,canonical-files,docs,agent-rules}.md`):** Replaced unquoted YAML frontmatter and `@[label](path)` full-file context includes with `yaml.safe_load()`-valid frontmatter (`trigger: glob`, quoted `globs`), `< 4 KB` rule files (`1,210`–`2,350` bytes each), strictly disjoint `yaw/`, `docs/`, and `.agents/` globs, solitary behavioral responsibilities per rule (zero repeated `AGENTS.md` directives or cross-rule prose duplication), and full pointer reachability from each domain rule to all cross-cutting and domain-specific `docs/*.md` specifications governing its files. Scoped [canonical-files.md](../.agents/rules/canonical-files.md) strictly to `globs: "tests/core/test_architecture.py"`, [docs.md](../.agents/rules/docs.md) to `_CANONICAL_DOCS_INVENTORY` rationale and the Tier 0–3 documentation dependency DAG, and [agent-rules.md](../.agents/rules/agent-rules.md) to `AGENTS.md` and `.agents/` customization governance.
  - **Custom Subagents (`.agents/agents/{critic,simplifier,ontologist}/agent.md`):** Enabled `hidden: false` and `inheritCustomizations: true` across all three subagents so `critic`, `simplifier`, and `ontologist` mount visibly in `<subagents>`. Updated `tools:` to current tool names (`read_file`, `write_file`, `edit_file`, `run_command`, `manage_task`, `send_message`, `search_web`, `schedule`), keeping `critic` strictly read-only, and grounded `ontologist` in `yaw.db` (`db.py`, `shacl.py`, `publication.py`).
  - **Canonical `docs/` Inventory & Split Evaluation Ledgers:** Split the 2,670-line `docs/evaluation.md` into four domain ledgers ([evaluation-ingestion.md](evaluation-ingestion.md), [evaluation-graph-storage.md](evaluation-graph-storage.md), [evaluation-inference.md](evaluation-inference.md), [evaluation-testing.md](evaluation-testing.md)) indexed by [evaluation.md](evaluation.md). Consolidated residual live invariants into [dspy-improvements.md](dspy-improvements.md) and [epic-1-hipporag-daemon.md](epic-1-hipporag-daemon.md), merged `epic-2-sprint-1-spec.md` and `epic-2-sprint-2-spec.md` into [epic-2-fastmcp-http.md](epic-2-fastmcp-http.md), removed 14 superseded audit/sprint files, and replaced all `[[wiki-links]]` with standard Markdown links. Left [native-cocoindex-indexing.md](native-cocoindex-indexing.md) untouched after the earlier `datasets.py` goal consolidation.
  - **Architecture Gate (`tests/core/test_architecture.py`):** Added Rules 7–12 (`test_agent_rules_yaml_frontmatter_and_size_caps`, `test_custom_subagents_yaml_frontmatter_and_tool_validity`, `test_every_repo_target_file_matches_at_least_one_rule_glob`, `test_zero_broken_markdown_links_and_zero_wiki_links`, `test_every_doc_is_reachable_from_rules_and_canonical_inventory`, and `test_canonical_module_purity_and_behavioral_contracts`).
- **Verification:**
  - `uv run --locked pytest -o addopts= -v tests/core/test_architecture.py` -> **15 passed**.
  - `uv run --locked ruff check tests/core/test_architecture.py` -> **All checks passed!**
  - `uv run --locked ruff format --check tests/core/test_architecture.py` -> **1 file already formatted**.
  - `uv run --locked ty check tests/core/test_architecture.py` -> **All checks passed!**

## 35. Python 3.14.9 preflight and full developer-check audit (2026-10-23)

- User-requested toolchain update: `.python-version`, `requires-python`, the
  project guidance and the recreated `.venv` now use **Python 3.14.9**. Global uv
  0.13.0 lacked that release in its catalogue; an isolated current uv installed
  the managed interpreter. Global uv was not replaced. The lock and environment
  were synchronized deliberately. The initial offline lock attempt lacked cached
  pytest-cov; normal online resolution succeeded.
- Added **`uv run preflight`** and **`uv run typecheck`**. Preflight runs Ruff
  formatting, safe Ruff fixes, ty fixes and serial pytest/testmon/coverage checks.
  It continues after ordinary failures, returns nonzero for failures and stores
  full UTF-8 logs, commands, tool versions, timings, JUnit, coverage JSON and
  Markdown/JSON summaries under ignored `outputs/preflight/`. `latest.json`
  points to the latest completed report. `--full` disables testmon deselection;
  `--verbose` streams details; the default console includes periodic progress.
- Ruff `fix=true` and pytest `addopts=["--testmon"]` are project defaults.
  The installed ty has `--fix` but rejects a persistent `fix=true` setting;
  `uv run typecheck` provides the default flag without shadowing the ty binary.
  No diagnostic rules were disabled and no suppression comments were added.
- Per the user's correction, **SlipCover was removed**. Coverage uses
  **coverage.py 7.17.0 / pytest-cov 7.1.0**, integrated with testmon 2.2.0.
  Its C tracer preserves the per-test contexts that Python 3.14's default sysmon
  core cannot provide. Reports label incremental coverage as current-run coverage,
  never a full-suite or stale merged result. Child-process coverage/dependencies
  are not automatically observed; use a full run before an unattended optimizer.
- Added explicit `--live` opt-in for the two resident-model tests. Existing
  offline tests retain real SQLite/files/loopback HTTP; no live LLM generation,
  corpus rebuild or unattended optimization was launched. The earlier incomplete
  SlipCover audit was stopped during the user-requested tool/interpreter change;
  only this task's verified process trees were terminated.
- Corrected the architecture test's incomplete stdlib whitelist using
  `sys.stdlib_module_names`. Corrected the NER duplicate fixture to repeat its
  definition and added a separate same-label/distinct-sense Mercury assertion.
  These repaired assertions do not weaken containment or semantic-sense rules.

### Measured full offline run

`uv run --locked preflight --full` completed with **230 passed, 7 failed,
2 skipped**, 53 warnings, **624.29 s** pytest time (630.74 s process time).
Coverage.py recorded **7,916 / 9,882 executable lines = 80.11%**. The failed
paths contribute coverage; this is not an all-passing acceptance result.

Available safe fixes left **602 Ruff findings** (599 E501, two B007, one SIM118)
and **67 ty diagnostics**, with zero ty automatic fixes for this inventory.
The seven failures are one engine optimization setup without config/token counter,
five CLI/runtime optimization fixtures whose in-process LM lacks a verified
resident tokenizer, and one benchmark MinimalProgram missing inference-capacity
fields. They fail before establishing an optimizer-quality or retrieval regression.
The concrete repair packets and full test names are tracked in [coverage-plan.md](coverage-plan.md).
Production config validation also needs a deliberate error rather than the observed
`None.base_url` AttributeError; the verified-tokenizer requirement must remain.

Full artifacts: `outputs/preflight/20261023T114056Z-53964/`. Its raw JUnit suite
header counted 242, but its 239 unique test-case records match pytest's terminal
result. The runner now counts those records. That full report was postprocessed
from preserved logs/XML to correct counts and add diagnostic groups; checks were
not rerun during that report correction. Raw artifacts remain unchanged.

The slowest full-run tests were candidate-stage process-restart reuse (136.83 s),
response-cache process restart/invalidation (105.91 s), and native optimizer
orchestration (53.78 s). Keep these physical recovery checks; use incremental
selection for development and a complete run for the final gate.

### Runner verification and remaining limits

- A real disposable-project integration test invokes the actual tools, verifies
  that static failures do not prevent pytest/reporting, repairs the real files,
  reruns incrementally, checks executed source coverage, and verifies a third
  unchanged run selects zero tests. The initial runner/architecture run passed
  **10 tests** (52.15 s); the final extended runner regression passed again
  (**1 passed**, 21.98 s). New runner/test/conftest pass both Ruff and ty.
- Final command smoke: `uv run --locked preflight -- --lf` completed with
  **4 passed, 7 failed, 1 skipped**, 112.6 s pytest-process time; static counts
  remained 602 / 67. This demonstrates reporting and conservative incremental
  selection, not repair of the seven audited failures. Its coverage is partial.
  Latest artifacts: `outputs/preflight/20261023T115835Z-48344/`.
- The suite reports unclosed SQLite connection ResourceWarnings and DSPy's
  upcoming normalized-LM API migration. Allocation ownership and migration are
  follow-up work, not suppressed warnings or proven production failures.
- `git diff --check` passes. No Tier 1/Tier 2 benchmark parity, live inference
  quality gain, or unattended stability result is claimed by this developer audit.

## 36. Runtime contract reliability repair (2026-10-23)

The first preflight repair packet validates persisted reflection/SIMBA objects
before adoption and SQLite writes, delegates QA aliases to the native predictor,
declares LM capabilities, preserves callable signatures and checks optimizer
configuration before corpus mutation.

Verification on Python 3.14.9 / DSPy 3.4.1 / CocoIndex 1.0.26:

- The 15 runtime diagnostics in the audit packet are cleared: targeted
  `ty check --fix` passes over engine, chat, program, reflection, watchdog and
  the new runtime regression file.
- Native DSPy, real SQLite, loopback inference, fresh-process recovery and
  architecture checks: **61 passed**, 20 warnings, 144.58 s. An isolated export
  of the staged commit also passed **61 tests**, 145.15 s, without the pending
  preflight, ingestion or dashboard presentation changes.
- Affected optimizer/hybrid engine checks: **4 passed**, 75.09 s; the isolated
  staged export also passed all four, 73.23 s. The final
  signature-metadata adjustment passed its two sync/async tool regressions.
- New reflection/runtime regression files pass Ruff. The repair adds no lint
  diagnostics relative to its starting snapshot under the current workspace
  policy; existing lint debt is not reported as repaired.
- No live-model optimization or corpus quality benchmark was launched. Existing
  DSPy temperature/cache notices and normalized-message API deprecations remain.
  The other audited full-suite fixture/type/lint failures remain separate work.

## 37. Ingestion model and runtime-capacity wiring (2026-10-23)

Committed the pending exact embedding alias and runtime capacity wiring.
`embedding.MODEL` shares the configured default; explicit embedding choices
remain independent of chat models. The runtime passes hydration context and AST
expansion limits into the engine and context/output limits into reflection LMs.
The repository profile declares the external server's 65,536-token context.

Verification: 17 ingestion/inference-budget tests passed (42.11 s), including
fresh-process parser-helper invalidation and loopback embedding routing. Three
configuration/runtime/crash-resume tests passed (27.28 s), including a new real
runtime/SQLite check for hydration capacity. Runtime and embedding modules pass
ty. These checks do not establish corpus quality or live-model convergence.

## 38. Preflight command delivery (2026-10-23)

Committed the pending serial preflight runner, Python 3.14.9 environment pin,
coverage.py/pytest-cov and testmon integration, safe-fix defaults and reports.
The user-selected 120-character Ruff limit and current Python compatibility
range (`>=3.14`) are preserved; audit counts at 100 characters remain historical.

The actual-tool disposable-project test passed (19.43 s): it verifies reporting
through static failures, measured coverage, successful incremental reruns and
zero selected tests when unchanged. Runner, test and conftest pass Ruff and ty;
`uv run --locked preflight --help` succeeds. Existing formatting-only edits were
checked for equivalent Python ASTs before inclusion. No full-repository green
claim is made: other audited fixture/type/lint failures remain to be repaired.

## 39. Complete Ruff, ty and pytest repair (2026-10-23)

Started from clean commit `2e9eded`. The baseline was 52 ty diagnostics and
218 Ruff findings at the existing 120-character limit. Typed native-DSPy
fixtures, exact tokenizer callables and HTTP handlers now declare their actual
contracts. Runtime optimizer tests use loopback tokenizer endpoints, one
persistence test now uses native GEPA instead of replacing its class, and the
retrieval-only fixture declares inference capacities. No static rules or test
assertions were disabled. Prompt/template/grammar/SQL literal values were
checked against the original source ASTs; only non-signature docstring trailing
whitespace required normalized comparison.

Automatic/conditional live-test skipping has been removed. Tests use the
configured resident Gemma model and native DSPy callbacks instead of requiring
Qwen or unbounded inspection histories. `--live` is retained only for command
compatibility. Full verification disables testmon deselection; incremental
preflight remains available for development.

The first full preflight ran all 259 cases: **257 passed, 2 failed, 0 skipped**,
731.95 s, with 81.37% line coverage. Both failures were in a new typed DummyLM
fixture whose capability default incorrectly said GBNF unsupported. Restoring
the original unspecified default fixed the unchanged assertions. Both actual
resident-model tests passed in this full run. Initial artifacts:
`outputs/preflight/20261023T162322Z-54704/`.

The run also exposed unclosed native diskcache SQLite connections. An allocation
trace located them in diskcache/core.py, rather than graph storage. YAW closes
constructor handles on their creating thread and per-operation handles on the
using thread, retaining persistent entries and native key/serialization logic.
An unmanaged two-worker control reproduced two ResourceWarnings. New real-cache
regressions cover both usage-accounting modes, thread teardown and persisted
reconfiguration reuse; **3 passed** with ResourceWarnings escalated to errors
(15.23 s). CLI resume and the two corrected grammar cases also passed under that
strict warning setting (**3 passed**, 24.18 s). Three transaction-only SQLite
test contexts now explicitly close their connections.

Final `uv run --locked preflight --full` passed all four gates. Ruff and ty
report **0 diagnostics**. Pytest executed **261 collected / 261 unique cases:
261 passed, 0 failures, 0 errors, 0 skipped**, with testmon selection deactivated.
Pytest took 721.85 s (728.5 s process time). Coverage.py measured **8,168 / 9,992
lines = 81.75%**, with zero excluded lines. Both configured-resident-model tests
and all native/SQLite/fresh-process recovery checks executed successfully.

The final run emitted **0 ResourceWarnings**. Its 23 remaining notices are one
Pydantic ReadOnly qualifier warning, four DSPy legacy message-API deprecations,
and 18 DSPy temperature-zero/rollout notices. None were suppressed. These API
migration/metadata notices do not represent skipped tests or failed assertions.

Artifacts: `outputs/preflight/20261023T164646Z-31068/`, including the actual logs,
JUnit XML, coverage JSON and `completion-audit.json`. The completion audit
independently reconciles collected, executed and unique case counts, confirms
required live/recovery cases, and rejects any failed/errored/skipped case. Tracked
Python source timestamps were checked to ensure no source changed during this
final run. `git diff --check` passes. This establishes developer-gate correctness,
not new corpus quality or Tier 2 benchmark parity.

## 40. Complete production coverage work in progress (2026-10-23)

The user confirmed that 100% coverage includes all production Python: yaw/,
benchmark.py and the four executable/support modules in scripts/. This goal is
not achieved. Development measurements below are accumulated focused-test
coverage, not a fresh full-suite acceptance result.

New tests exercise actual subprocess output/timeout/interruption cleanup,
durable reports, configuration writes, native DSPy objects, evidence provenance,
real sparse matrices, SQLite payload corruption, loopback model identity and
async embeddings, and CocoIndex AST/disk hydration. No production lines were
excluded. Configuration rejects unknown field writes before touching the file.
Parser fixes reject null coordinates/definitions, preserve case-variant
homograph senses and normalize NumPy string triples before fuzzy matching.

Coverage.py's native subprocess patch is enabled in the tracked .coveragerc;
the .gitignore exception retains that configuration. Actual Python children now
contribute coverage of the real source tree. Copied fixture implementations do
not count as production execution. Preflight's reporting text was reconciled
and its coverage command now includes benchmark and scripts. A disposable
project test checks all three source locations in the actual coverage JSON.

Verified focused runs: 82 parser/preflight edge cases passed in 134.71 seconds;
9 real sparse/cache cases passed in 11.73 seconds; the combined inference,
registry, evidence, graph and hydration edge suite passed 80 cases in 33.83
seconds. A subsequent real async HTTP embedding/registry run passed 15 cases in
35.75 seconds. All 15 preflight cases passed after the reporting-text update
(162.36 seconds). The expanded-scope command, actual three-source coverage
report and newest prefix-update case subsequently passed in a 33-case run
(177.76 seconds). Repository-wide Ruff and ty currently report zero
diagnostics. Earlier new-test failures were corrected fixtures: JSON-valid
nonfinite score text and explicit optional-value/type narrowing.

Before the expanded-scope command change, accumulated yaw-only coverage was
8,743 / 9,994 executable lines (87.48%); core, parsers, model registry, graph
candidate cache and PPR measured 100%. Hydration measured 95%, reflection 93%,
and embeddings 89%. The separate first benchmark/script measurement passed
28 tests in 25.80 seconds, covering 907 / 2,206 lines (41.12%). These separate
percentages must not be presented as complete production coverage.

An additional 15 benchmark metric cases passed in 2.28 seconds. They establish
exact recall denominators, rank ordering, title/crop/whitespace matching,
empty/malformed inputs, answer aliases and configurable EM/F1 aggregation.
scripts/evaluation.py now measures 126 / 126 lines (100%) in the development
data; the aggregate script scope is 983 / 2,206 lines (44.56%). One unsuppressed
coverage notice reports that benchmark was not imported in this narrowly
selected metric run; benchmark data was retained from the preceding run.

The runtime and script databases were then combined with coverage.py's native
combine operation: **9,735 / 12,200 lines = 79.7951%, 2,465 missing, 0 excluded**
across all 36 production files. The lower percentage reflects the expanded
denominator, not a regression. See all-production-coverage.json. This combined
database is now the append baseline; future focused runs must keep all three
source flags (`--cov=yaw --cov=benchmark --cov=scripts`). Production source
identity was verified after combination. No source file changed during merging.

Artifacts are under outputs/coverage-work/: coverage.json, scripts-coverage.json,
separate coverage databases and the production source-hash manifest. Changed
production files are purged before development append runs; a fresh complete
suite with exact zero missing/excluded lines remains required for acceptance.
Large remaining gaps include engine/program/storage/runtime/CLI, executable
scripts, and genuine platform/error paths. No unattended optimization was
started by this coverage work; the user's existing optimizer/watchdog processes
were inspected and left running.

## 41. SQLite, downloader and report coverage packet (2026-10-23)

New real SQLite cases migrate an actual legacy database with old entity name
uniqueness, chunk hashes, missing AST/definition columns, synonymy edges and
malformed historical fact JSON. They verify row preservation, SKOS backfill,
FTS reconstruction and migration idempotence. Transaction cases distinguish
owned rollback from an active outer transaction. Additional cases verify
global cross-source label priority, missing/zero vector sense resolution and
independent dense/lexical candidate scores. Fourteen new storage cases pass;
db.py now measures 855 / 965 lines (88.60%).

Downloader tests stream more than 1 MiB through real loopback HTTP, exercise
fallback and forced replacement, and verify failed requests preserve the prior
target and remove the temporary file. Actual CLI processes check valid/invalid
cached datasets and aliases without external network. Eight cases pass;
download-datasets.py measures 229 / 261 lines (87.74%).

Ten dashboard cases render actual Matplotlib SVGs and read audit JSON from disk.
They verify candidate ordering, validation-score precedence, cumulative best
scores, last-good snapshots after malformed writes, observed throughput,
placeholders and configuration fallback. watchdog.py measures 514 / 825 lines
(62.30%). No live dashboard output was overwritten by these tests.

Parity rendering tests exposed two source defects: latency SVG groups were
unclosed and baseline bars were missing, while the quality chart could report
PASS despite an ExactMatch failure in the actual gate. The latency chart now
closes groups, advances rows and renders both engines; the quality chart uses
the recorded gate decision and includes ExactMatch when deriving legacy status.
XML parsing tests verify real outputs, escaping, zero-duration comparisons and
agreement with gate decisions. The source coverage for this edited module was
purged before remeasurement. Native benchmark comparison regressions still pass.

Final focused packet: **60 passed in 43.39 seconds**, with testmon deselection
disabled. Repository-wide Ruff and ty report zero diagnostics; git diff --check
passes. Production source hashes match all 36 measured files. The accumulated
all-production report is **10,255 / 12,211 lines = 83.9817%, 1,956 missing and
0 excluded**. The additional executable lines implement the verified SVG fixes.
Earlier focused runs emitted coverage-only notices about benchmark not being
imported or no new parent-process data; no warnings were suppressed. The final
60-case run includes benchmark execution and reports no warnings.

This packet makes progress toward the full 100% goal; it does not complete it.
Remaining gaps and the final fresh complete-suite acceptance remain open.

## 42. Native engine recovery and reflection boundaries (2026-10-23)

The new engine cases use native dspy.LM, HippoRAGProgram and dspy.Embedder,
actual loopback HTTP chat responses, real source files and yaw.db. They first
index a source, delete the bounded candidate-stage cache in a transaction,
close the engine and recreate its objects. Structured, legacy triples-only,
structured-only and partially invalid statement checkpoints make no new NER
or triple requests. A legacy checkpoint lacking definitions makes exactly one
definition-repair request and still reuses its committed NER/triples. Persisted
OpenIE JSON, prompt fingerprints and the single fact survive replay. Empty
index/retrieval/context/deletion operations leave graph rows empty and make no
inference calls. These are orchestration guarantees for the exercised cases,
not a quality measurement or an unattended-run stability guarantee.

Reflection tests execute the native async wrapper with the verified serving
tokenizer, check output allowance, disabled history, deepcopy and serialized
state, and reject incompatible checkpoint context. Real HTTP-served JSON tests
reject missing tokenizer discovery, zero serving context, different weights,
malformed JSON, nonresident models and missing backend routes. Completed
proposal state rejects blank or nontext instructions. The installed DSPy
OpenAI adapter uses max_completion_tokens on the wire; the test checks the
actual 128-token allowance instead of assuming the legacy max_tokens name.

Final focused packet: **29 passed in 42.40 seconds**, including the prior native
engine cache/resume and prompt-mutation invalidation regressions. Ruff and ty
pass repository-wide, git diff --check passes, and all 36 production hashes
match. No production code changed in this packet. reflection.py measures
**220 / 220 executable lines (100%)** in accumulated development data. Engine
coverage rose to 2,053 / 2,351 lines; additional genuine definition-repair paths
also execute in program.py.

The complete production development report is **10,325 / 12,211 lines =
84.5549%, 1,886 missing, 0 excluded**. Remaining notices are DSPy's existing
legacy-message API deprecation and the focused-run notice that benchmark was
not imported; neither is suppressed. The overall goal remains incomplete and
requires the remaining cases plus a fresh complete-suite acceptance run.

## 43. Benchmark input, persistence and end-to-end coverage (2026-10-23)

New benchmark cases exercise answer cleanup, ordered aliases, null/missing
answer rejection, document fallbacks, source-document matching, four dataset
padding formats, local-cache reuse and actual HTTP dataset fallback/failure.
Native dspy.Embedder callables verify repeatability under batching, individual
calls and reversed inputs across the active embedding roles; an intentionally
changing mathematical embedding function is rejected. Integrity checks retain
provider errors and distinguish QA checks from retrieval-only validation.

The numeric-alias case exposed a real runtime type-contract mismatch:
get_gold_answers was declared to return strings but could return numeric
aliases. Values are now normalized to strings before ordered deduplication;
the new regression verifies that numeric 3 and string "3" collapse to one
answer. The benchmark module's old measurements were purged before running
its tests against the changed implementation. User prompt edits are preserved.

Two complete run_yaw_benchmark cases use real CocoIndex ingestion, SQLite,
native DSPy HTTP chat/embedding calls and sparse retrieval. Retrieval-only and
QA reports contain the expected source and fixture scores; repeated invocation
reuses the committed index and cached LM responses without new LM requests.
These oracle-provider scores verify orchestration, not local-model quality.
Any retry executable lookup is confined to the disposable test environment.

Actual benchmark.main tests persist success and registry-rejection reports,
verify the serialized report against the returned data, and keep training
queries out of evaluation embedding probes. Real model-identity functions are
bound to loopback HTTP endpoints. Each test restores the previous DSPy LM
binding, preventing a stopped fixture server from leaking into later tests.

Verification: **63 passed in 51.44 seconds** across benchmark, new edge cases,
native engine recovery and parity rendering. After adding LM-binding isolation,
the full benchmark edge file passed **29 cases in 43.74 seconds**. Four final
gold-retention/distractor-padding variants passed in **26.33 seconds**. Ruff,
ty and git diff --check pass. The remaining three notices in the full packet
are existing Pydantic ReadOnly metadata and DSPy temperature/rollout warnings;
none were suppressed. All 36 production source hashes match the final data.

benchmark.py now measures **538 / 627 lines (85.81%)**. Complete production
development coverage is **10,589 / 12,211 lines = 86.7169%, 1,622 missing and
0 excluded**. The 100% goal and fresh complete-suite acceptance remain open.
