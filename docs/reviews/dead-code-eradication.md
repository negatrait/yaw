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
checks: 125 passed. Final gates pending.
