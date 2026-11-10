# Dead code eradication

Started 2026-11-09 on dev at e58688f, after N1-N5. Existing uncommitted work is
preserved; the cleanup is limited to unreachable helpers and test-only APIs.

Observable contract: production entry points and framework callbacks remain;
tests exercise active implementations rather than keeping alternate production
implementations alive. No dependency, schema, provider or retrieval change.

Definition of done: removed definitions and exports have no callers; retained
integration coverage uses production APIs; ty, Ruff, full pytest and diff checks
pass. Commit the cleanup separately from preexisting changes.

Removed inventory:

- Unused edge-weight aliases and reverse path-conversion helper.
- Unused prompt JSON helper, tuple-only triple and names-only NER wrappers.
- Test-only predictor inventories on the task and reflection programs.
- Uncalled retrieval-hook wrapper and engine SIMBA wiring wrapper.
- Unused synchronous embedding request executor and telemetry-reset wrapper.
- Unused completion-logprob confidence parser and its export.
- Test-only subgraph Turtle formatter, prefix constant, provenance summary and
  enriched fact lookup, including their exports.
- Heading/title convenience wrappers and live-component test initialization hook.

Tests of active confidence validation, literal conversion, inference binding,
reflection learning and AST headings are retained through canonical APIs or
direct real SQLite assertions. Tests exclusive to retired implementations are
removed. No new compatibility wrappers are introduced.

Reference inventory scans yaw, scripts and benchmark.py, followed by tests and
exports. Logging filters, CocoIndex reconcile/live/subscriber callbacks, DSPy RLM
overrides, optimizer callbacks and declared CLI/application entry points are
externally dispatched contracts, not dead code merely because the scan finds
no direct call. This review does not assert whole-program reachability of every
public API or eliminate test fixtures themselves.

Evidence: outputs/dead-code-review/. Initial collection and migration failures
are retained separately. Focused native bindings/budgets, ingestion and core
checks: 125 passed. Follow-up traversal policy and architecture checks: 61 passed.

Final validation on the concurrently committed N6 tree (1b33dac):

- `uv run --locked ty check`, `ruff check`, `ruff format --check` and
  `git diff --check`: passed.
- Full suite, including live inference and 20-trial paired batch acceptance:
  571 passed, three subtests passed, then one failure at
  `test_engine_step7_heal_shacl_violations_rlm_repl_tools_and_token_stats`.
  It expects one label-judge batch but receives two after the concurrent N6
  batch-limit change. Evidence: `outputs/dead-code-review/full-verbose.log`.
  This is not a passing full-suite gate; the remaining 382 tests are recorded
  separately in `outputs/dead-code-review/remainder.log`: 372 passed and ten
  failed. Combined coverage is 943 passed and 11 failed across all 954 collected
  tests, with three subtests passed. Remaining failures concern chunk IDs and
  entity expectations, SHACL error expectations, and optimization output JSON
  parsing. None reports a missing retired API. Their baseline status has not
  been isolated; do not assert they are preexisting or that the suite is green.
- Automatic approval review rejected a proposed additional storage/indexing
  API cleanup because production-use uncertainty remained. That script was
  not executed; the additional APIs remain. This is a bounded cleanup, not a
  claim that every test-only surface has been eradicated.
- A concurrent process committed the main cleanup with N6 while verification
  was running. Follow-up spec corrections and the stale alias test correction
  remain uncommitted until the full quality gate passes.
- Governing confidence spec corrected to retire unused helper requirements and
  distinguish actual Turtle export from unverified QA-prompt injection claims.

## DC2: finish removal and restore acceptance (2026-11-10)

Authorized scope: finish the remaining confirmed test-only storage/indexing
helpers, resolve all eleven recorded failures, run the complete locked quality
gates and a blind critic audit, then commit. Resume at 1b33dac with the three
existing follow-up modifications retained.

Acceptance: no remaining callers or exports of the retired checkpoint lookup,
entity-KNN wrapper, private runtime aliases or test-state property; tests retain
real SQLite/vector and lifecycle invariants via active contracts. Canonical
source IDs and extraction evidence must survive the file pipeline. CLI stdout
must contain the promised JSON document. Preserve original error propagation
and exact token accounting across the configured batch partition. Do not weaken
tests to accept fabricated IDs, lost source evidence, malformed output or errors.

Required gates: locked sync, ty, Ruff check/format, full pytest (no testmon
deselection), diff check and the mandatory unguided critic audit before commit.

Implementation:

- Removed checkpoint lookup and entity-KNN wrappers and their exports; test
  evidence uses direct SQLite reads or active chunk/fact KNN implementations.
- Removed private shared-runtime aliases and the live-component test-state
  property. Tests import the actual owner and seed their fixture state locally.
- Fixed optimize progress output to stderr, retaining JSON-only stdout.
- Graph build fixtures now materialize complete JSON source snapshots, preserve
  per-passage extraction evidence, and use CocoIndex's actual chunk IDs. The
  multi-record fixture keeps consecutive source records in the same file.
- Native publication test driver forwards the original indexing error sink
  exception. Unrepaired SHACL failures must still abort; no expected failure
  was removed or downgraded.
- Token accounting checks the current update's audit trails, including both
  batches after N6's cap; prior hierarchy history is explicitly cleared.
- Retained YawGraphEngine.index as an application API: automatic approval review
  rejected a speculative retirement of that public source-ingestion contract.
  Its implementation and callers were not changed by that proposed script.
- The requested flash critic model is unavailable. The user selected
  gpt-6-luna as the replacement for the mandatory blind critic dispatch.

Focused evidence: dc2-focused.log (32 passed, one stale ID assertion subsequently
fixed), dc2-failures.log (23 passed, one accounting assertion subsequently fixed),
dc2-token.log (one passed), dc2-storage.log (188 passed). Artifacts are under
outputs/dead-code-review/. Full gates and critic result follow below.
