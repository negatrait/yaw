# Incremental publication: atomic execution units

Started 2026-11-05 on `dev`, from verified scheme-free checkpoint `c01f4aa`.
Controlling contract: [native graph publication](../native-cocoindex-graph.md).
This record implements steps 2 and 3 of the agreed sequence: establish fixed-evidence
batch equivalence, then replace global work with bounded updates. Live diagnostic
pairs and baseline capture are deferred until these contracts pass.

Definition of done: fixed extraction/adjudication evidence produces the same
canonical graph for 64+64 and 128; source changes schedule only affected work;
publication does not hydrate all sources, clone a graph database or redeclare
unaffected rows. Final typed CocoIndex targets commit one validated semantic
revision with provenance, vectors, FTS and source checkpoints. Unrelated semantic
islands and distinct homographs survive. Existing experiments remain untouched.

Each unit is a separate reviewable change. Run focused real-substrate tests during
development and the mandatory project gates before its conventional commit.
Never hide a failing semantic contract with an xfail, model fallback or looser
comparison. A failing adversarial fixture identifies the next implementation unit.

| Unit | Change and boundary | Observable acceptance / dependency |
| --- | --- | --- |
| U1 | Add one fixed-evidence paired harness through the current native runtime. No production planner replacement. | Same 128 occurrence IDs/text/NER/facts and pair decisions; actual 64+64 vs one 128 publication. Compare entity/fact IDs, metadata, typed edges, labels, support/confidence and vector/FTS ownership. Exercise chunk64/65, last-chunk bridge, homographs, literal objects and restart. Record work and semantic differences; loopback providers make no quality claim. |
| U2 | Persist occurrence-local raw evidence and explicit canonical bindings; choose canonical representatives independently of arrival. | Identical evidence, reversed arrival and changed representative converge without losing source predicates/literal types. Raw evidence survives merge/retraction. U1 supplies the comparator. |
| U3 | Make candidate coverage and decision dependency keys explicit; persist positive and negative decisions. | Changed-to-old/new and reverse-neighbor cases match the reference. Identical evidence reuses decisions; changed support reopens affected decisions. No relation edge is treated as a decision receipt. Depends on U2. |
| U4 | Prove locked CocoIndex SQLite transaction behavior for the final typed targets, vec0, FTS and checkpoints. | Real connector probes demonstrate grouping, read-your-proposal validation, rollback and crash/restart behavior. Failure stops the cutover; do not bolt on a second writer/journal. Can be investigated after U1, before U5. |
| U5 | Replace the global planner/sink in one vertical cutover with an indexed affected-owner proposal and final typed target declarations. | Only affected source/identity/fact/edge/aggregate owners are read, hydrated, embedded and declared. Delete graph cloning, `:memory:`, global survivor proposals and the single `graph` target. One active implementation remains. All callers/watchdog use its schema. Depends on U2-U4. |
| U6 | Complete replacement/deletion, no-op/restart, late evidence and bounded-work gates. | Source edit retracts correct support; shared support survives; no-op/restart has zero provider work and target writes. Failures preserve the previous complete revision. Adding unrelated base corpus does not increase the measured affected work; genuine semantic fanout is counted. U1 exact equivalence and required bridge checks pass after every mutation. Depends on U5. |

U2 and U3 may expose affected sets larger than the incoming 64 chunks. Measure
that necessary fanout; never enforce an arbitrary limit that changes semantics.
Candidate KNN is not constant-time and a top-k cap is not a completeness proof.
Whole-graph retrieval CSR construction is measured separately from publication.

After U6: capture the scheme-free timing/count baseline, then paired live trials
and broader quality/scaling acceptance. The live model cannot establish equality
of fixed evidence; do not spend model compute proving an already-known global
planning defect.

## Checkpoint

- [x] U1: paired fixed-evidence harness and recorded result (fixture scope below).
- [ ] U2: stable evidence/bindings and canonical identities.
- [ ] U3: candidate equivalence and invalidation.
- [ ] U4: final-target atomicity proof.
- [ ] U5: bounded planner/target cutover; global path deleted.
- [ ] U6: mutation/restart/work-bound acceptance.

Current constraints: the existing planner loads all active inference rows, hydrates
their source text, clones all graph/vector tables for additions and proposes a
complete graph to the atomic sink. Scheme removal did not solve these issues.
This is a source-verified limitation, not a benchmark result.

### U2 incremental step: deterministic merge identities

Implemented one identity rule: a resource sense is keyed by normalized label and
resolved definition; an accepted equivalence class uses its smallest member ID.
The preferred display label remains independent of that representative. Merges
return explicit entity/fact binding changes, which the planner applies to
source-owned evidence without rewriting raw mention or assertion fields.
Affected fact IDs are derived from canonical endpoints, the original predicate
and the typed object, including literal datatype/language. Colliding facts retain
their source supports, confidence fields and vector ownership. Collapsed
self-links retain their raw assertions with a null canonical fact binding.
The cosine/difflib admission rules are unchanged. Missing merge endpoints raise.

The native publication function is now mounted as a CocoIndex component because
it declares child table targets; calling it as a memoized helper violated that
SDK contract. Final publication remains typed row declarations through the
shared sink. Isolated whole-graph planning/staging remains U5 debt.

Verification, 2026-11-05: 73 focused tests passed in 9.23 seconds, including
storage, synonym/inference contracts, graph integration, shared-sink recovery and
architecture checks. The new real SQLite fixture compares all six arrival orders
of three equivalent senses, in both merge directions, including a smaller
representative arriving later. It checks exact state, raw evidence preservation,
typed/directed facts, source confidence, vector keys/bytes and FTS ownership.
Evidence: `outputs/graph-spec-review/U2-step-focused.{log,xml}`.

`uv sync --locked` and production `ty check yaw` pass. The complete `ty check`
still reports three diagnostics in obsolete publisher tests. The full pytest
run stops at four collection errors importing the removed publisher/API
(`test_publication`, `test_scheme_removal`, `test_indexing_overlap`, and the paired
harness importing its fixture). Next atomic unit: migrate those tests to final
typed declarations and restore full-suite collection, then verify occurrence
replacement/retraction and canonical rebinding across restart. U2 remains open;
no acceptance/baseline claim or commit is made with these gates failing.

### U1 result and first repair

The native paired harness passes for a 128-chunk document with interleaved topics,
homographs across chunk64/65, literal observations and a final-chunk bridge. Both
same-process and restarted 64+64 arms match the single-publication 128 arm for
stored entity/fact IDs, labels, source bindings, numeric confidence (1e-6), typed
edges, literal datatypes, extraction artifacts, vector bytes and FTS ownership.
No-op replay has zero provider calls and zero SQLite changes. This is controlled
orchestration evidence, not live-model quality, timing acceptance or general proof
for positive identity merges, reordered evidence and top-k invalidation.

The harness exposed prefix replay: an unchanged 128-chunk snapshot was redeclared
as 64, then 128, causing 190 unnecessary provider calls. Frozen admission now keeps
matching native source/artifact revisions declared and batches only changed
revisions. It reads metadata using indexed chunk-ID batches; it does not introduce
a second cache or claim to remove the global planner reads. Those remain U5 work.

An initial homograph failure was a fixture error: its embedding endpoint returned
the same vector for every resource, triggering the intentional high-similarity
merge policy. The production automatic merge path and its existing tests were
restored in full. The corrected fixture gives the Mercury senses orthogonal
vectors. Retain the user-confirmed difflib/cosine fast merge policy; changing its
risk tradeoff requires separate evidence. This fixture does not establish that
perfectly colliding embeddings are separable by the current automatic rule.
The general automatic gate defaults to cosine >=0.95 and difflib >=0.96. The
existing same-normalized-label branch also allows cosine >=`synonymy_threshold`
(default 0.8), with its fact-pole guard. Neither branch was changed in U1.

Focused U1 evidence: `outputs/graph-spec-review/U1-admission.{log,xml}`, two tests
passed in 19.91 seconds. Earlier failed/stalled fixture runs are retained separately.
Final U1 project gates, 2026-11-05:

- `uv sync --locked`: 100 packages resolved/checked.
- `uv run --locked ty check` and `uv run --locked ruff check --fix`: passed.
- `uv run --locked ruff format`: 164 files unchanged.
- Focused native indexing, replay, normalization, candidate cache and confidence
  regressions: 53 passed in 129.00 seconds.
- `uv run --locked pytest --no-testmon -q --tb=short -o junit_family=legacy`:
  **850 passed, three subtests passed, 78 warnings**, 627.45 seconds. No test
  deselection or live-test skip. Evidence: `outputs/graph-spec-review/U1-full.{log,xml}`.
- `git diff --check`: passed. Commit only this unit's seven source/test/doc paths;
  leave the unrelated `.obsidian/workspace.json` change untouched.

Both fixture arms have 128 chunks, eight entities, 133 facts, 525 aggregate edges,
256 label/source links, 256 fact supports and 648 edge supports. Exact row
comparisons pass. The legacy `entity_chunks` table is empty on this path; source
links are verified through labels and fact/edge support, not asserted as a complete
raw-occurrence binding representation. U2 must establish that representation and
remove superseded state when safe.

### U4 connector boundary established during U2

The locked connector probe observed a node at revision 2 and its checkpoint at
revision 1 after a checkpoint write failure. This is documented CocoIndex behavior,
not an SDK defect: separate sinks are not one transaction. CocoIndex reconciles
possible previous states on the next update instead of undoing external writes.
See the upstream [processing component contract](https://github.com/cocoindex-io/cocoindex/blob/v1.1.0/docs/src/content/docs/programming_guide/processing_component.mdx)
and [custom connector contract](https://github.com/cocoindex-io/cocoindex/blob/v1.1.0/docs/src/content/docs/advanced_topics/custom_target_connector.mdx).

Coherent graph rows, canonical bindings and their applied input checkpoint must
therefore be declared by one producing component through one shared
`TargetActionSink`. Its connector applies the reconciled actions in one SQLite
transaction and validates the proposed state before commit. Typed row schemas
remain `TableSchema.from_class`; application processing declares desired rows.
Independent components and sinks retain native convergent roll-forward behavior.
Do not add application retry loops or assume an entire `app.update()` is atomic.

The focused real SQLite/CocoIndex test
`tests/daemon/graph/storage/test_managed_sqlite.py` passes: rejected revision 2
rolls back both tables, the next update publishes both at revision 2, and a no-op
performs zero row writes. This establishes the shared-sink mechanism only. U2
integration, graph-wide failure/restart coverage and bounded publication remain
unfinished; neither U2 nor U4 is marked complete.

### U2 increment: native-target test migration (2026-11-06)

Resumed from `43399a3`, which includes the resource-boundary and fail-fast
publication work. The earlier grammar/parser mismatch is already fixed there;
the remaining U2 collection failures came from tests importing the deleted
`storage.publication` module and its whole-graph API.

Changed publication, scheme-removal and indexing-overlap fixtures to declare
typed desired rows through the active CocoIndex target connector. Removed direct
calls to the retired sink. The publication tests now exercise real `app.update()`
rollback/reconciliation across graph tables, vector rows, FTS and the applied
checkpoint, plus shared-source withdrawal and zero-write no-op replay. Planner
withdrawal is also checked across environment restart, including surviving
mentions and assertions. All custom graph contexts supply the required validator.

The fixed-evidence paired harness now compares `entity_mentions` and
`fact_assertions` instead of the retired `entity_chunks` table. Both arms require
256 mentions and 256 bound assertions; comparisons also include edge reasoning.
Both same-process and restarted 64+64 arms match the 128 reference exactly, with
the existing floating-point tolerance and byte-exact vector checks unchanged.
The first migrated run exposed the stale comparator (23 passed, two failed);
the corrected paired run passed both tests in 30.63 seconds. Evidence:
`outputs/graph-spec-review/U2-native-tests.{log,xml}` and
`outputs/graph-spec-review/U2-native-pair.{log,xml}`.

Promoted the verified one-component/one-shared-sink publication boundary into
both governing native CocoIndex plans. Production behavior is unchanged by this
unit. This does not complete U2 or establish replacement/retraction confluence,
bounded indexing, `vec0` publication, crash-window recovery or live quality parity.
Those acceptance gates remain open. Full project gate results follow below.

Gate results for this increment:

- `uv sync --locked`: 100 packages resolved/checked.
- `uv run --locked ty check`: passed with no test exclusions. Evidence:
  `outputs/graph-spec-review/U2-native-final-ty.log`.
- `uv run --locked ruff check --fix` and `uv run --locked ruff format`: passed;
  final format check left 170 files unchanged. Evidence:
  `outputs/graph-spec-review/U2-native-final-{ruff,format}.log`.
- Architecture and updated Markdown links: 15 tests passed in 5.15 seconds.
  Evidence: `outputs/graph-spec-review/U2-native-final-architecture.log`.
- Unfiltered `uv run --locked pytest --no-testmon ...` collected successfully but
  stalled in `test_live_silicon_direct_gemma_gbnf`. The resident backend reported
  task 866067 at 137 decoded tokens at both 09:34:58 and 09:45:30 UTC. Stopped the
  exact owned full-suite test worker after preserving the timeout stack and
  backend sample; no inference service was stopped. Evidence:
  `outputs/graph-spec-review/U2-native-full.log` and
  `outputs/graph-spec-review/U2-native-live-stall.json`. This is not a passing live
  gate; a service restart was requested from the user.
- Separate `uv run --locked pytest --no-testmon -m 'not live' ...`: **893 passed,
  three subtests passed, two live tests deselected, 71 warnings**, 482.73 seconds.
  Evidence: `outputs/graph-spec-review/U2-native-nonlive.{log,xml}`.
- `git diff --check`: passed.

Changes remain uncommitted because the mandatory live-inclusive gate is pending.
After the inference service recovers, rerun the unfiltered suite before committing
this increment. No model retries, timeouts, test markers or failure handling were
changed to obtain these results.

Recovery verification after the user restarted Lemonade:

- Confirmed fresh embedding/Gemma backend PIDs (45972/47312), both ready and
  idle, and no active Gemma slot before issuing new inference.
- `uv run --locked pytest tests/daemon/graph/inference/test_prompts.py
  --no-testmon -m live ...`: **two passed**, 28 non-live tests deselected,
  nine warnings, 209.97 seconds. Evidence:
  `outputs/graph-spec-review/U2-native-live-restarted.{log,xml}`.
- The live-inclusive suite is being rerun without exclusions before committing;
  its results will be recorded separately from the interrupted run.

Final live-inclusive gate: `uv run --locked pytest --no-testmon -q --tb=short
-o junit_family=legacy --junitxml=outputs/graph-spec-review/U2-native-full-restarted.xml`
exited 0: **895 passed, three subtests passed, 79 warnings**, 707.74 seconds.
No tests were skipped or deselected. Evidence:
`outputs/graph-spec-review/U2-native-full-restarted.{log,xml}`. This supersedes
the earlier pending test-gate status for this increment.

After pytest's passing summary, interpreter shutdown emitted a pending cancelled
CocoIndex `_process_live_wrapper` task diagnostic and a ContextVar token reset
`ValueError`. The complete log retains it; nothing was suppressed. Its trigger
is not established here. This increment restores native-target coverage and full
suite collection; it does not establish clean live-component teardown or close
the U4 lifecycle/crash gates. That diagnostic requires separate lifecycle work.

The next U2 unit remains occurrence replacement/retraction and canonical rebinding
equivalence across restart. U2 and the bounded-planner cutover remain open.
