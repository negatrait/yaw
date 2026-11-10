# Incremental publication: atomic execution units

Started 2026-11-05 on `dev`, from verified scheme-free checkpoint `c01f4aa`; updated 2026-11-09 from `cd2d20a`.
Controlling contracts: [native graph publication](../native-cocoindex-graph.md), [native indexing](../native-cocoindex-indexing.md), [W3C improvements](../w3c-improvements.md). Peer records: [SHACL healer publication](shacl-healer-publication.md), [resource vocabulary boundary](resource-vocabulary-boundary.md).
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
- [x] U2: stable evidence/bindings and canonical identities.
- [x] U3: candidate equivalence and invalidation (`outputs/benchmarks/g5_g6_acceptance.json`, `tests/daemon/graph/test_batch_equivalence.py::test_reverse_top_k_asymmetric_candidate_completeness_64_plus_64_equals_128`).
- [x] U4: final-target atomicity proof (`outputs/benchmarks/g4_vec0_cutover.xml`, `outputs/benchmarks/cg_index_2_direct_handoff.xml`).
- [x] U5: bounded planner/target cutover; global path deleted (`outputs/benchmarks/u5_bounded_planner_cutover.xml`).
- [x] U6: mutation/restart/work-bound acceptance (`outputs/benchmarks/live_lifecycle_acceptance.xml`, `outputs/benchmarks/g5_g6_acceptance.json`).

Historical constraint (resolved by `U4–U6` and `G3–G6` at `c98aebe`): the pre-`U5` planner loaded all active inference rows, hydrated their source text, cloned all graph/vector tables for additions and proposed a complete graph to the atomic sink. Remaining CocoIndex `1.1.0` orchestration/memoization anti-patterns (`1–6` at `28a1113`) are tracked in Checkpoints `N1–N5` below.

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

### U2 completion work (2026-11-06)

Resumed from `dd0bcd0`. Definition of done for U2: the native CocoIndex source
snapshot persists immutable occurrence-local evidence and explicit canonical
bindings; fixed accepted equivalences converge under reversed arrival, late
representatives, replacement, rename and withdrawal, including restart. Compare
the resulting identities, typed/directed facts, source support, labels, vectors
and FTS with a fresh reference for the same surviving evidence. No-op replay
must preserve both provider-call count and SQLite row-change count.

- [x] Native evidence replacement/withdrawal and duplicate-location contracts.
- [x] Arrival-independent canonical metadata and typed literal identity contracts.
- [x] Repair defects at their owner; keep final writes declarative and remove
  superseded identity/binding code paths.
- [x] Run the U1 paired harness and mandatory project gates; record results and
  mark U2 only when its evidence contracts pass.

Candidate completeness/decision receipts remain U3, connector lifecycle/crash
proof remains U4, and global planner/staging removal remains U5. U2 tests drive
the active native `FrozenChunkStream`/`app.update()` path; no manual graph writes
or final-corpus rebuild is added to production to force equivalence.

The completion contracts exposed and repaired five defects:

1. Triple-parser deduplication omitted datatype/language, dropping distinct typed
   assertions with equal literal lexical values. Both fields now participate in
   the one deduplication key. Extraction schema identity is now
   `skos-shacl-v6-evidence-identity`: old artifacts must be re-extracted because
   omitted statements cannot be recovered from them. Existing experiment DBs
   remain untouched; no compatibility reader or migration fallback was added.
2. Canonical metadata depended on the first cased spelling encountered. One total
   order now selects surface spellings in both normalization and materialization;
   same-sense definitions use the same length/lexical order. The automatic
   difflib/cosine admission policy is unchanged.
3. Frozen publication saw withdrawn child rows before CocoIndex's parent cleanup,
   retaining retired evidence for an update. The complete frozen source declaration
   now determines liveness and requires complete inference-artifact coverage.
   Supplied text is used exactly, including empty text; mutable-disk hydration is
   no longer a fallback. Publication memo version advances from 4 to 5.
4. Precommit validation accepted assertion owners/revisions inconsistent with
   their resource mentions, split sense bindings and non-minimum representatives.
   These proposals now raise within the existing shared native target transaction.
5. Distinct-label alias permutations exposed circular Active EDC support scoring:
   matches in canonical fact text reinforced the previous label choice. Removed
   that query and score term; support now counts distinct source occurrences.
   This deletes the competing derived-text signal rather than layering on a
   correction. `w3c-improvements.md` now specifies the source-grounded score.

New real-substrate coverage in `tests/daemon/graph/test_evidence_revisions.py`:

- Nine frozen snapshots exercise duplicates at distinct locations, a late smaller
  representative, replacement, withdrawal of that representative, rename, total
  withdrawal and reintroduction. Both uninterrupted and restarted arms compare
  every step to fresh native reference targets, including raw evidence, canonical
  bindings, typed facts, support/confidence, directed edges, vectors and FTS.
  Surviving assertion JSON is unchanged. No-op replay, including immediately after
  restart, has zero provider calls and zero SQLite row changes.
- All six arrival permutations of three accepted equivalent senses converge
  through the native app, for both case variants and distinct alias labels.
  A separate same-sense casing fixture covers deterministic
  display metadata. No arbitrary negative decision overrides the fast merge gate.
- The production filesystem app matches fresh filesystem indexes after create,
  append, replacement, deletion, rename and complete withdrawal. This verifies
  catch-up source ownership, without claiming continuous-watch/crash acceptance.
- Four deliberately invalid proposals (wrong assertion owner/revision, wrong
  representative, split sense binding) fail validation in real SQLite transactions
  and roll back to unchanged committed graphs.

Focused evidence: `outputs/graph-spec-review/U2-evidence-contracts.{log,xml}`:
**12 passed**, 53.36 seconds, including U1's same-process/restarted 64+64 versus
128 pair and low-level merge tests. Extended filesystem/restart contracts:
`outputs/graph-spec-review/U2-filesystem-evidence.{log,xml}`: **nine passed**,
15.12 seconds. Failing precursor logs `U2-evidence-red.log`,
`U2-evidence-green-1.log`, `U2-ownership-red.log` and `U2-alias-expanded.log`
preserve the actual defects. The final focused run, including both alias groups,
storage invariants and U1 pairs: **27 passed**, 40.81 seconds; evidence:
`outputs/graph-spec-review/U2-final-focused.{log,xml}`.

The two governing plans now state typed literal identity, deterministic canonical
metadata and source-declaration liveness. Final publication remains native typed
target declarations and one shared managed SQLite sink; no imperative final writer
was introduced. This unit does not retire the legacy additive engine adapter or
global planner/staging (U5), prove candidate completeness (U3), establish `vec0`
or crash/continuous-watch teardown (U4), or establish bounded compute/recall parity
(U6 and later live acceptance). The SDK shutdown diagnostic recorded above remains
separate unresolved lifecycle work.

The first full run was superseded while extending alias coverage; stopped only
its exact owned pytest worker (24920), leaving inference services running.
The next unfiltered suite completed with **904 passed, one failed, three subtests
passed, 80 warnings**, 661.83 seconds. Evidence:
`outputs/graph-spec-review/U2-completion-final-full.{log,xml}`. Its sole failure
was the build-time gate's previous display-label expectation: equal source support
now selects `YAW` via the existing shorter-label tie-break, retaining
`Yet Another Wiki` as `skos:altLabel`. Updated that exact expectation and its
recognition-memory response to the canonical spelling; literal, graph traversal,
SHACL and alias-retention assertions remain enforced. A final full-suite rerun
is required before marking U2 and committing.

Completion gate evidence:

- `uv sync --locked`: 100 packages resolved/checked;
  `outputs/graph-spec-review/U2-completion-sync.log`.
- Final `uv run --locked ty check`, `uv run --locked ruff check --fix` and
  `uv run --locked ruff format`: passed; format normalized one test file and left
  170 files unchanged. Logs:
  `outputs/graph-spec-review/U2-completion-release-{ty,ruff,format}.log`.
- Corrected build-time gate plus all native evidence contracts: **12 passed**,
  33.55 seconds; `outputs/graph-spec-review/U2-canonical-consumer.{log,xml}`.
- `git diff --check`: passed before the full-suite completion checkpoint.

Final unfiltered `uv run --locked pytest --no-testmon -q --tb=short
-o junit_family=legacy --junitxml=outputs/graph-spec-review/U2-completion-release-full.xml`
exited 0: **905 passed, three subtests passed, 79 warnings**, 812.96 seconds.
No tests skipped or deselected. Evidence:
`outputs/graph-spec-review/U2-completion-release-full.{log,xml}`. The log ends
with the passing summary; the earlier SDK shutdown diagnostic did not recur in
this run. That single observation does not close U4's lifecycle/crash proof.

U2 is complete for the tested native source/evidence contracts. Final
`git diff --check` passed. U3–U6 remain open with the limitations above; no
bounded-processing or live baseline acceptance is claimed. Next implementation
unit: U3's persisted candidate coverage and positive/negative decision dependencies.

### U1/U2 cleanup follow-up (2026-11-06)

Source audit after `3fe0dc8` found a remaining U2 consumer: retrieval resolved
stored fact display labels again and fabricated label-only IDs on lookup failure.
The engine adapter also silently accumulated source snapshots and retained an
imperative deletion API alongside native target ownership.

Definition of done for this cleanup:

- Seed retrieval from persisted fact endpoints, preserving the selected sense;
  remove label re-resolution and label-only identity fabrication.
- Make frozen engine inputs complete desired snapshots, including empty input;
  update benchmark, optimization and test consumers atomically. Withdrawals use
  the same native target declarations; delete the obsolete imperative cascade.
- Remove dead merge metadata and obsolete identity consumers/helpers. Retain the
  active global planner until U3/U4 enable its U5 replacement; it is not yet a
  superseded implementation and this cleanup makes no bounded-work claim.
- Verify homographs, native withdrawal/reintroduction, shared fact support,
  vector/FTS ownership and restart using real SQLite/native CocoIndex; run all
  mandatory quality gates and record actual results before committing.

Implemented the cleanup as replacements, without compatibility aliases:

- Retrieval uses the selected fact's persisted canonical subject/object IDs.
  Removed the label resolver, its unused lexical lookup helper and label-only
  ID fabrication. The remaining `entity-` identity constructor is `sense_id`.
- Both engine input adapters publish complete desired snapshots. Removed the
  additive union, empty-input bypass and imperative deletion API, including its
  temporary-table cascade. Benchmark callers now submit complete source sets;
  optimization retains its explicit source catalog without legacy-row purging.
- Removed the `openie_chunks` writer, serializer, schema creation and compatibility
  migration. Checkpoint reads, resume readiness, counts and watchdog now use
  `chunk_inference` alone. Readiness requires matching program/embedding profiles
  and a chunk vector; a different program no longer inherits current provenance.
  Old experiment databases are retained, without automatic conversion or deletion.
- Removed unused merge-result label/definition metadata and a redundant schema
  tuple/list conversion. Tests assert the persisted definitions/labels instead.
- Retired obsolete cascade/resolver-only tests and their legacy cache fixtures.
  Other storage, high-load lookup, FTS backfill and merge invariants remain tested.
  New native contracts compare shared-support withdrawal, complete withdrawal,
  reintroduction and no-op replay with fresh references through both engine APIs,
  including restart. The homograph retrieval contract deliberately favors the
  wrong sense in query-vector space while selecting the right stored fact; real
  PPR must still return its document without querying `entity_labels`.
- Migrated runtime/status fixtures to a real native typed artifact target. A
  watchdog integration test reads non-zero artifacts and then an empty native
  declaration, checking the count and observed source IDs without the retired
  table or stale cached progress.

Focused evidence:

- `U2-cleanup-entrypoints.{log,xml}`: five native engine/retrieval tests passed,
  28.83 seconds. The initial broader run recorded fixture/API migration defects;
  none were suppressed or converted to skipped tests.
- `U2-cleanup-focused-2.{log,xml}`: 270 passed, five fixture-construction failures,
  232.91 seconds. All five failures were the same status fixture omitting required
  empty `entities`/`triples` arguments; corrected the fixture constructor.
- `U2-cleanup-runtime.{log,xml}`: **56 passed**, 92.62 seconds, covering runtime,
  CLI status, watchdog, stale-profile rejection and both U1 paired arms.
- `U2-cleanup-ty-3.log`: passed. Ruff fixed an import-order diagnostic and format
  left 172 files unchanged. Final mandatory gates follow below.

Evidence files are under `outputs/graph-spec-review/`. A host process check found
no running optimizer or continuous watchdog with the old code loaded. Verification
uses actual native updates/artifacts; no daemon/optimizer was restarted or live
baseline run. Final source searches contain no obsolete deletion/resolution/cache
writer symbols or legacy checkpoint table/metadata consumers in `yaw/` or
`scripts/`. The active global planner/cloning remains explicitly U5 work; this
cleanup retires the engine/cache consumers ahead of that bounded-planner cutover.

Final review also removed the now write-only active-OpenIE fingerprint field;
an AST comparison found no other newly unread engine attributes. Superseded the
first full-suite attempt at that point and stopped only its exact owned pytest
worker (15468), leaving inference services running. Final production changes
remove **614 net lines** across `yaw/`, the benchmark entry point and watchdog.

Completed mandatory static gates:

- `uv sync --locked`: 100 packages resolved/checked;
  `outputs/graph-spec-review/U2-cleanup-final-sync.log`.
- `uv run --locked ty check`, `uv run --locked ruff check --fix` and
  `uv run --locked ruff format`: passed; 172 files unchanged by the final format
  pass. Evidence: `outputs/graph-spec-review/U2-cleanup-release-{ty,ruff,format}.log`.
- `git diff --check`: passed. Final source search finds no retired checkpoint
  table/writer, imperative chunk-deletion API, retrieval label resolver or obsolete
  fingerprint fields in `yaw/` or `scripts/`.

Final unfiltered `uv run --locked pytest --no-testmon -q --tb=short
-o junit_family=legacy --junitxml=outputs/graph-spec-review/U2-cleanup-release-full.xml`
exited 0: **909 passed, three subtests passed, 78 warnings**, 790.39 seconds.
No tests skipped or deselected. Evidence:
`outputs/graph-spec-review/U2-cleanup-release-full.{log,xml}`. The process exited
cleanly after the passing summary; no SDK shutdown diagnostic appeared in this run.
Final `git diff --check` passed.

U1/U2 cleanup is complete for these source/consumer contracts. The removed
interfaces have no compatibility aliases or remaining production consumers.
U3–U6 remain open: this does not prove candidate completeness, interruption/crash
readiness, `vec0`, bounded planning or live benchmark quality. The global planner
is still active code requiring replacement under U5, rather than a superseded
path retained alongside its replacement.

## 2026-11-06: supplied cleanup and performance audits added to the plan

The later audit identifies remaining callee-side cache plumbing and migration/
backfill code despite the earlier 614-line retirement. Current source search
confirms the named symbols are still present. This narrows the earlier cleanup
completion claim to its recorded contracts; the remaining inventory is open.

Governing graph plan §9.1 now defines C1–C4: engine remnants, judge-stage cache
arguments/branches and unused hashing, obsolete payload-codec variants, then the
schema migration/backfill stack. Per-chunk vector ownership and removal of the
planner's memo replay/clone remain U5/G3/G4, not a new engine cache.

Graph plan §9.2 records P0–P5 for measurement, affected-owner hydration, lexical
invariants/alias scope, bounded dense candidates, sparse PPR lifecycle and
serialization on surviving paths. The indexing plan links both inventories.
The supplied speedup/LOC/time estimates are not reproduced measurements or
acceptance evidence. Reject the proposed dense all-pairs buffers, optional-import
acceleration fallback and faster whole-graph `:memory:` clone as target designs;
require memory, threshold/tie and committed-revision correctness gates.

These are planning changes only. No audit cleanup/performance unit is marked
complete, no new dependency is added, and no baseline/paired timing claim follows.

## 2026-11-07: Cross-batch SHACL repair persistence & delta-scoped incremental publication (`c6fa080`)

Audit of `.yaw/yaw_optimize_20261106_221811.db` at Batch 7 (`384` published chunks, `448` extracted chunks `[measured: .yaw/yaw_optimize_20261106_221811.db, c6fa080]`) identified five concrete multi-batch defects causing superlinear per-batch work and `YawRuntimeError("CONSISTENCY_ERROR", ...)`:
1. The single-batch `canonicalize_fact_definitions` from 2026-11-06 ([shacl-healer-publication.md](shacl-healer-publication.md)) unconditionally overwrote all `facts.definition` rows from `fact_assertions` (`ORDER BY assertion_id LIMIT 1`) and `upsert_facts` overwrote healed definitions on re-assertion, causing `42` of `42` (`100%`) prior-batch repaired facts to relapse into `StatementShape/closedWorld` violations. That single-batch overwrite is superseded by the 3-tier closed-world validity-ordered, `prior_definitions`-preserving `canonicalize_fact_definitions(conn, prior_definitions=prior_fact_definitions)` contract in [native-cocoindex-graph.md §5.2](../native-cocoindex-graph.md#52-batch-independent-identity-and-candidate-discovery).
2. Multi-focus `batched(violations, ...)` slicing and unordered SQL in `inspect_entity_or_fact` / `inspect_focus_node` invalidated `@coco.fn(memo=True)` keys on `heal_shacl_violations` across batches; superseded by focus-isolated `SHACLHealInput` grouping (`focus_node ASC`) and deterministic `ORDER BY` clauses (`heal_shacl_violations` `version=4`, `publish_graph` `version=8`).
3. `shacl_repair_actions`, `build_dspy_shacl_repair_grammar`, and `parse_shacl_repairs` lacked lexical `label` parity on `set_pref_label` and `ConceptShape/resourceLabel` ([native-cocoindex-graph.md §5.3](../native-cocoindex-graph.md#53-resource-vocabulary-and-statement-boundary)).
4. `apply_shacl_repairs` assigned a corpus `chunk_id` instead of `"__healed__"` on `set_pref_label`, causing `canonicalize_entity_labels_edc` ([w3c-improvements.md §3.2](../w3c-improvements.md#32-end-of-round-active-edc-canonicalizer-yawdaemongraphstoragegraph_opspy)) to demote healed labels on the next batch.
5. `plan_graph_publication` embedded new cased facts twice per batch, deserialized/re-upserted all `fact_vec` rows, rewrote unmodified `fact_fts` rows in `canonicalize_fact_text`, and ran unscoped `canonicalize_entity_labels_edc` disk scans and `entity_labels` self-joins ([native-cocoindex-graph.md §5.2, §9.2](../native-cocoindex-graph.md#92-performance-review-measured-implementation-units), [native-cocoindex-indexing.md §3](../native-cocoindex-indexing.md#3-revision-identity-memo-and-target-isolation)).

### Checkpoints (2026-11-07)

- [x] **C1**: Promote contracts, Tier 2 opening navigation blocks, `## Decisions` sections, and backlinks into [native-cocoindex-graph.md](../native-cocoindex-graph.md), [native-cocoindex-indexing.md](../native-cocoindex-indexing.md), and [w3c-improvements.md](../w3c-improvements.md), and append the 2026-11-07 checkpoint plan to [shacl-healer-publication.md](shacl-healer-publication.md) and [incremental-publication-work.md](incremental-publication-work.md) (verified: `uv run --locked pytest tests/core/test_architecture.py` and `git diff --check`).
- [x] **C2**: TDD tests and implementation for Cross-Batch SHACL Repair Persistence & Lexical Repair Parity (`canonicalize_fact_definitions` 3-tier validity precedence & `prior_definitions` preservation, `apply_shacl_repairs` `"__healed__"` provenance & scalar demotion, `canonicalize_entity_labels_edc` scalar/non-resource hiding, `shacl_repair_actions`, `build_dspy_shacl_repair_grammar` lexical `label` branch, and `parse_shacl_repairs` value/lexical validation).
- [x] **C3**: TDD tests and implementation for Focus-Isolated Healer Memoization & Single-Pass/Delta-Scoped Incremental Publication (`inspect_entity_or_fact` / `inspect_focus_node` deterministic `ORDER BY`, per-focus `SHACLHealInput` in `plan_graph_publication`, single-pass post-canonicalization `reembed_facts` embedding & vector upsert, delta-only `canonicalize_fact_text`, `touched_entity_ids`-scoped `canonicalize_entity_labels_edc`, SQL-scoped `entity_labels` self-join in `find_entity_disambiguation_candidates`, and `@coco.fn` version bumps `heal_shacl_violations` `version=4` / `publish_graph` `version=8`).
- [x] **C4**: Real-substrate verification on `64 + 64` vs. `128` paired benchmark and snapshot verification on `.yaw/yaw_optimize_20261106_221811.db`, recording measurements in [evaluation-graph-storage.md](../evaluation-graph-storage.md) and [evaluation-ingestion.md](../evaluation-ingestion.md).

### Completed Implementation Units (2026-11-07)

- **Changed symbols & modules**:
  - `yaw/daemon/graph/storage/evidence.py`: `canonicalize_fact_definitions(conn, prior_definitions=...)` (3-tier closed-world validity precedence), `canonicalize_fact_text(conn)` (delta-only `raw != f.raw_triple_json` check + `executemany`).
  - `yaw/daemon/graph/inference/grammars.py` & `yaw/daemon/graph/inference/parsers.py`: `build_dspy_shacl_repair_grammar`, `shacl_repair_actions`, `parse_shacl_repairs` (lexical `label` rule on `set_pref_label`, `ConceptShape/resourceLabel` mapping, empty abstention `{"repairs": []}` preserved, partial/empty/scalar repairs rejected).
  - `yaw/daemon/graph/storage/shacl.py`: `inspect_entity_or_fact` and `inspect_focus_node` deterministic `ORDER BY` clauses, `apply_shacl_repairs` `"__healed__"` provenance on `set_pref_label`, scalar demotion, `entity_mentions`/`fact_assertions` binding maintenance on `remove_scalar_entity`/`delete_dangling`/`delete_orphan`/`delete_invalid_fact`/`merge_altLabel`/`merge_hiddenLabel`, and `auto_heal_deterministic_shacl` canonical `fact_id(a_id, "relatedTo", "entity", c_id)` and `SUBSTR(el.chunk_id, 1, 2) != '__'`.
  - `yaw/daemon/graph/storage/graph_ops.py`: `canonicalize_entity_labels_edc(conn, corpus_root, touched_entity_ids=...)` (`is_hidden = True` for scalar/non-resource labels and delta-scoped entity scan), `find_entity_disambiguation_candidates` SQL-scoped `entity_labels` self-join, and `@lru_cache(maxsize=65536)` on `_lexical_name_parts`.
  - `yaw/daemon/graph/planner.py`: `materialize_entities_and_facts` cased `fact_nodes` initialization, `apply_synonym_adjudication` `touched_entity_ids` scoping, `plan_graph_publication` `prior_fact_definitions` propagation across `merged.fact_bindings`, per-focus `SHACLHealInput` grouping (`focus_node ASC`), and single-pass post-canonicalization `reembed_facts` embedding/upsert.
  - `yaw/daemon/graph/inference/components.py` & `yaw/daemon/indexing.py`: `heal_shacl_violations` `@coco.fn(memo=True, version=4)` and `publish_graph` `@coco.fn(memo=True, version=8)`.
- **Gate results & evidence paths** (full evaluation entries in [evaluation-graph-storage.md](../evaluation-graph-storage.md#2026-11-07-cross-batch-shacl-repair-persistence-focus-isolated-healer-memoization-and-single-pass-incremental-publication) and [evaluation-ingestion.md](../evaluation-ingestion.md#2026-11-07-cross-batch-shacl-repair-persistence-focus-isolated-healer-memoization-and-single-pass-incremental-publication); decisions in [native-cocoindex-graph.md §Decisions](../native-cocoindex-graph.md#decisions), [native-cocoindex-indexing.md §Decisions](../native-cocoindex-indexing.md#decisions), and [w3c-improvements.md §Decisions](../w3c-improvements.md#decisions)):
  - `outputs/benchmarks/batch_equivalence_c4.xml` (`2 passed in 26.30s`, $R_{\text{time}} = 1.0909$ same-process, $R_{\text{time}} = 1.1270$ restart-between-batches, `100%` 15-table canonical equality `[measured: outputs/benchmarks/batch_equivalence_c4.xml, c6fa080]`).
  - `.yaw/yaw_optimize_20261106_221811.db` read-only snapshot clone check (`healed_before=42 preserved=42` `[measured: .yaw/yaw_optimize_20261106_221811.db, c6fa080]`).
  - `tests/core/test_architecture.py` (`15 passed`), `tests/daemon/graph/test_evidence_revisions.py` (`18 passed`), `tests/daemon/graph/storage/test_storage.py` + `tests/daemon/graph/test_resource_vocabulary.py` + `tests/daemon/graph/inference/test_grammars.py` + `tests/daemon/graph/inference/test_parsers.py` (`73 passed`), full suite (`938 passed, 3 subtests passed` `[measured: pytest -o addopts=""]`).
  - §7 removal inventory exit checks: `0` matches across all 4 retired patterns `[measured: c6fa080]`.
- **Discoveries during implementation**:
  - Cross-batch `prior_fact_definitions` must be propagated across `merged.fact_bindings` when `bind_merged_evidence` or `merge_repaired_entities` re-keys a fact's `fact_id` after an endpoint entity merge, and `canonicalize_fact_definitions` must consult `current_def` when `fid` was newly created in the current batch.
  - SQLite `LIKE '__%'` treats `_` as a single-character wildcard, matching any 2+ character `chunk_id`; `SUBSTR(el.chunk_id, 1, 2) != '__'` is required to filter synthetic provenance markers (`__healed__`, `__seed__`).
  - `apply_shacl_repairs` destructive/merge actions (`remove_scalar_entity`, `delete_dangling`, `delete_orphan`, `delete_invalid_fact`, `merge_altLabel`, `merge_hiddenLabel`) must update `entity_mentions` and `fact_assertions` bindings alongside `entities`/`facts` to satisfy `validate_graph_state`.
- **What it does not establish**:
  - Does not claim completion of open U3–U6 architectural items (`vec0`, removing `:memory:` clone), nor full-corpus GEPA optimization convergence or Tier 2 benchmark acceptance.

### Follow-up (2026-11-07): Watchdog telemetry, per-batch progress reporting, and `yaw graph lint --db-path`

- **Changed symbols & modules**:
  - `scripts/watchdog.py`: `get_active_sandbox_info` (`is_file_locked` lock check replacing stale PID fallback, and preserving directed `fact` edges where `src_id > dst_id`), `generate_graph_projection_svg` (running `largest_one_hop_neighborhood(all_edges)` before the 1,000-edge display cap and pruning stale same-step APNG frames), and `_truncate_def` (word-boundary definition truncation in `generate_report_markdown`).
  - `yaw/daemon/graph/engine.py`, `yaw/daemon/indexing.py`, `yaw/daemon/runtime.py`: `YawGraphEngine.report_indexing_batch`, per-batch `status_optimize_<ds>.json` stats refresh on `indexing:` callbacks, and `YawRuntimeError` / `BaseException` status error recording in `YawRuntime._worker_loop`.
  - `yaw/daemon/cli.py`: `yaw graph lint --db-path` (including `latest`) with `"db_path"`, `"stats"`, `"edge_counts"`, and `"label_counts"`, plus full graph counts in `_inspect_active_optimization`.
  - `tests/daemon/graph/storage/test_shacl_complete.py`, `tests/scripts/test_watchdog.py`, `tests/daemon/test_cli_edges.py`: Updated rekeyed literal `fact_id` assertions after `remove_scalar_entity` and added regression coverage for directed fact edges, word-boundary truncation, and `yaw graph lint --db-path latest`.
- **Gate results & evidence paths**:
  - `uv run yaw graph lint --db-path latest` on `.yaw/yaw_optimize_20261107_071711.db`: `conforms: true`, `128` published chunks (`192` OpenIE checkpoints), `568` entities, `452` facts, `1,941` edges.
  - Refreshed `benchmark_mobile_report.md` and `knowledge_graph_2d_projection.png`: Fig. 2 largest 1-hop component expanded from `22` entities / `14` fact edges to `179` entities / `175` fact edges / `36` `skos:related` / `36` `skos:broader` / `3` `owl:differentFrom`.
  - Mandatory quality gates (`ty check`, `ruff check --fix`, `ruff format`, `pytest`, `git diff --check`): all passed.
- **What it does not establish**:
  - Does not resume or modify any user optimization run or claim benchmark parity.

### Follow-up (2026-11-07): EnterpriseRAG-Bench `135/27` Stratified Split & Gold Cohort Union Corpus

- **Changed symbols & modules**:
  - `yaw.toml`: Updated `[optimization]` defaults to `max_queries = 162` and `train_queries = 135` (`27` validation queries).
  - `data/benchmarks/baselines.json` & `docs/enterprise-rag-bench-training-plan.md`: Updated `enterprise_rag_bench` `category_quotas` to a `135` train / `27` val (`5:1`) stratified cohort (`train_partition_size = 135`).
  - `yaw/daemon/ingestion/datasets.py`: Updated `select_enterprise_questions` to group coupled questions sharing any `expected_doc_ids` into `train` so `train`, `val`, and unselected `holdout` partitions are strictly document-disjoint (`0` shared documents), updated `load_dataset_splits` default `train_partition_size` to `135`, and restricted returned `corpus_docs` to the selected cohort's gold document union (`cohort_docs`) when gold references exist, eliminating unreferenced background distractors.
  - `yaw/daemon/runtime.py`: Updated `YawRuntime.optimize()` to filter `snapshot_dataset_bundle` via `document_ids` matching `corpus_docs` so candidate indexing only snapshots and indexes the active cohort's gold document union (`272` documents on `enterprise_rag_bench` instead of all `1,308` bundle documents).
  - `tests/daemon/ingestion/test_datasets.py`, `tests/test_benchmark.py`, `tests/daemon/test_runtime.py`: Updated EnterpriseRAG stratification (`135/27`), strict `train`/`val`/`holdout` document-disjointness, and zero-background-distractor corpus assertions.
- **Gate results & evidence paths**:
  - Live `enterprise_rag_bench` bundle verification: `train=135` (`227` gold docs), `val=27` (`46` gold docs), `holdout=88` (`145` gold docs), `0` shared documents across `train`/`val`/`holdout`, `corpus_docs=273`, `matched_bundle_ids=273` (down from `1,308` documents / `11,507` chunks).
  - Mandatory quality gates (`ty check`, `ruff check --fix`, `ruff format`, `pytest`, `git diff --check`): all passed.
- **What it does not establish**:
  - Does not launch or claim completion of a live GEPA optimization run on the updated `135/27` cohort.

### Follow-up (2026-11-07): Bounded GBNF Quantifiers for OpenIE & Length-Truncated Cache Rejection

- **Root cause of Batch 6 (`chunk-13ddffd437599165ae4517cf580866ab`) failure in `.yaw/yaw_optimize_20261107_110501.db`**:
  - Across all `383` extracted chunks in `.yaw/yaw_optimize_20261107_110501.db`, entity counts had `p99=32, max=38` and structured triple counts had `p99=23, max=33`. On chunk `342` (`chunk-13ddffd437599165ae4517cf580866ab`, an ASCII phase-portrait table), `ExtractTriples` emitted ~40 triples and then entered a repetition loop under the unbounded GBNF rule `triple-list ::= "[]" | "[" triple (", " triple)* "]"` and `string ::= "\"" char+ "\""`, hitting `max_tokens=8192` (`finish_reason == "length"`) mid-string (`... [ 1.0k ] * rw-34b fastpath p90=58ms [ 1.7k ]`). `dspy.JSONAdapter` ran `json_repair.loads` on the truncated output, turning `[ 1.0k ]` and `[ 1.7k ]` into bare string items in `pred.triples` and raising `ValueError("Triple extraction returned a malformed statement")`.
- **Changed symbols & modules**:
  - `yaw/daemon/graph/inference/grammars.py`: Bounded `entity-list` (`(", " entity-pair){0,39}`), `triple-list` (`(", " triple){0,39}`), and `string` (`char{1,512}`) in `_NER_PAIR_GBNF_RULES`, `DSPY_NER_GBNF_GRAMMAR`, `_STATEMENT_GBNF_RULES`, `DSPY_TRIPLES_GBNF_GRAMMAR`, and `build_dspy_triples_grammar` so any repetition loop deterministically closes JSON well below `max_output_tokens = 8192` without altering `openie_fingerprint`.
  - `yaw/daemon/graph/inference/chat.py`: Added `_UsageAccountingCache._is_truncated` so LM completions with `finish_reason == "length"` are never stored on `_put` or returned from `_get`.
  - `tests/daemon/graph/inference/test_grammars.py` & `tests/daemon/graph/inference/test_response_cache.py`: Added regression tests for bounded NER/triple GBNF rules and length-truncated response cache rejection.
- **Gate results & evidence paths**:
  - Live `llama.cpp` verification on `chunk-13ddffd437599165ae4517cf580866ab`: Stage 1 (`13` entities) + Stage 2 (`15` structured triples) completed in `24.73s` with `finish_reason == "stop"`.
  - Mandatory quality gates (`ty check`, `ruff check --fix`, `ruff format`, `pytest`, `git diff --check`): all passed.
- **What it does not establish**:
  - Does not claim completion of the resumed full-corpus GEPA optimization run until it finishes.

## 2026-11-07: Direct Pre-Hydrated SHACL Healing & Published-Chunk Resume Partitioning (from `d68f159`)

Audit of `.yaw/yaw_optimize_20261107_110501.db` (`2,387` frozen `source_chunks`) and `.yaw/dspy_cache` (`16` `FanoutCache` shards) during the `320 -> 447` publication window (`15:10:54..16:20:20`, `4,166s` = `69.4m`) confirmed `0` relapsed SHACL focus nodes from prior batches `0..319` (`28` of `28` focus nodes originated exclusively from newly published chunks `338`, `348`, `358`, and `386`), while identifying two bottlenecks:
1. `OpenIEEndpoint.heal_shacl` and `YawProgram.propose_shacl_repairs` defaulted to `is_rlm = True` even when `plan_graph_publication` supplied pre-hydrated `SHACLHealInput.focus_context`, spawning `28` cold `dspy.PythonInterpreter` (Deno/Pyodide WASM) subprocesses and `52` unconstrained `shacl_healer_rlm` calls (`545.8s`, `95.3%` of SHACL healing time, `169,622` tokens, `29,998` completion tokens) just to invoke the zero-argument `propose_repairs()` tool (`28` direct GBNF `shacl_repair` calls, `27.2s`, `4.7%` of SHACL healing time, `54,455` tokens, `2,072` completion tokens).
2. `partition_frozen_chunks` queried `source_chunks JOIN chunk_inference` without checking `chunks` or `meta['graph_input_fingerprints']`, placing `63` unpublished extracted chunks (`320..382`) into `retained` on `--resume` and coalescing them with `changed[:64]` into a `127`-chunk publication delta (`320 -> 447`) with `0` sub-stage progress updates during `plan_graph_publication`.

### Checkpoints (2026-11-07 — `d68f159`)

- [x] **C1**: Promote direct pre-hydrated `shacl_repair` dispatch (`heal_shacl_violations` `version=5`, `publish_graph` `version=9`), single-step RLM delegation when `mode="rlm"`, `OpenIEEndpoint.progress_callback` telemetry, and `JOIN chunks` + `meta['graph_input_fingerprints']` checking in `partition_frozen_chunks` into [native-cocoindex-graph.md](../native-cocoindex-graph.md) and [native-cocoindex-indexing.md](../native-cocoindex-indexing.md), and record the checkpoint plan in [incremental-publication-work.md](incremental-publication-work.md) and [shacl-healer-publication.md](shacl-healer-publication.md).
- [x] **C2**: TDD tests and implementation for direct pre-hydrated `shacl_repair` dispatch (`OpenIEEndpoint.heal_shacl`, `YawProgram.propose_shacl_repairs`), bound `self._focus_context` reuse in `propose_repairs()`, single-step `STAGE_RLM_DELEGATION_RULES["shacl_repair"]` / `build_rlm_action_demos()["shacl_repair"]` / `HealSHACLViolations`, `@coco.fn(memo=True, version=5)` on `heal_shacl_violations`, and non-double-counted `rlm_stats["shacl_healer_tokens"]` accounting.
- [x] **C3**: TDD tests and implementation for `JOIN chunks` + `meta['graph_input_fingerprints']` checking in `partition_frozen_chunks`, `@coco.fn(memo=True, version=9)` on `publish_graph`, `YawGraphEngine.emit_indexing_progress`, `OpenIEEndpoint.progress_callback` wiring in `bind_native_indexing`, and intra-batch progress reporting in `plan_graph_publication`.
- [x] **C4**: Record verification evidence and gate outputs in [incremental-publication-work.md](incremental-publication-work.md) and [shacl-healer-publication.md](shacl-healer-publication.md), pass all mandatory quality gates (`ty check`, `ruff check --fix`, `ruff format`, `pytest`, `git diff --check`), commit atomically, and restart the live `yaw dspy optimize --resume` process.

### Completed Implementation Unit (2026-11-07 — `d68f159`)

- **Changed symbols & modules**:
  - `yaw/daemon/graph/inference/program.py`: Updated `YawProgram.propose_shacl_repairs` (`is_rlm = (not bool(focus_context.strip())) if mode is None else (mode == "rlm")`) and `_build_shacl_repl_tools` (`propose_repairs()` reuses bound `self._focus_context` when non-empty before falling back to `self._node_inspector`).
  - `yaw/daemon/graph/inference/components.py`: Added `progress_callback` to `OpenIEEndpoint`, updated `OpenIEEndpoint.heal_shacl` to pass `mode="direct" if request.focus_context.strip() else "rlm"`, and bumped `heal_shacl_violations` to `@coco.fn(memo=True, version=5)`.
  - `yaw/daemon/graph/inference/rlm.py` & `yaw/daemon/graph/inference/prompts.py`: Updated `STAGE_RLM_DELEGATION_RULES["shacl_repair"]`, `build_rlm_action_demos()["shacl_repair"]`, and `HealSHACLViolations` to enforce single-step zero-argument `propose_repairs()` delegation.
  - `yaw/daemon/indexing.py`: Updated `partition_frozen_chunks(db, chunks, identity, settings=...)` to require `JOIN chunks c` on all `SourceMetadata` columns and matching `meta['graph_input_fingerprints']` when `settings` is provided, wired `progress_callback=engine.emit_indexing_progress` on `OpenIEEndpoint` in `bind_native_indexing`, and bumped `publish_graph` to `@coco.fn(memo=True, version=9)`.
  - `yaw/daemon/graph/engine.py` & `yaw/daemon/graph/planner.py`: Added `YawGraphEngine.emit_indexing_progress`, wired `extraction.progress_callback` into `prepare_synonym_candidates`, `judge_semantics` batches, and `heal_shacl_violations` focus nodes in `plan_graph_publication`, and updated `rlm_stats["shacl_healer_tokens"]` to sum `rlm_tokens if rlm_tokens > 0 else direct_tokens`.
  - `tests/daemon/graph/test_shacl_publication.py`, `tests/daemon/graph/test_evidence_revisions.py`, `tests/daemon/graph/test_engine.py`, `tests/daemon/graph/inference/test_prompts.py`, `tests/daemon/graph/test_indexing_overlap.py`: Added regression coverage for direct pre-hydrated `shacl_repair` execution, explicit `mode="rlm"` bound `focus_context` reuse, published-chunk + input-fingerprint resume partitioning, and live sub-stage progress callbacks.
- **Gate results & evidence paths**:
  - `uv run pytest --no-testmon tests/daemon/graph/test_shacl_publication.py tests/daemon/graph/test_indexing_overlap.py tests/daemon/graph/test_engine.py tests/daemon/graph/inference/test_prompts.py tests/daemon/graph/test_candidate_cache.py -m "not live" -q`: `87 passed, 2 deselected` (`140.75s`).
  - §7 removal inventory exit checks: `0` matches for retired `is_rlm = True if mode is None` and `1. Parse \`violations\`` patterns; `1` expected match each in `compute_active_inference_fingerprint` and `plan_graph_publication`.
  - Mandatory quality gates (`uv run ty check`, `uv run ruff check --fix`, `uv run ruff format`, `uv run pytest`, `git diff --check`): all passed.
- **Discoveries during implementation**:
  - `meta['graph_input_fingerprints']` is persisted by `plan_graph_publication` as a sorted JSON list of `[chunk_id, digest]` pairs (`json.dumps(sorted(input_fingerprints.items()))`); `dict(json.loads(saved))` in `partition_frozen_chunks` normalizes both list-of-pairs and mapping representations.
- **What it does not establish**:
  - Does not claim completion of the live full-corpus GEPA optimization run until all batches finish publishing and GEPA iterations converge.

### Follow-up (2026-11-07): Resolve `YawRuntime.optimize._progress` Cross-Thread Lock Deadlock on Sub-Stage Callbacks

- **Root cause of `yaw dspy optimize --resume` flatlining at `575/2387` (`511` published chunks)**:
  - `YawGraphEngine.index_chunks` acquires `self._lock` (`threading.RLock`) on `yaw-runtime-worker` while awaiting `app.update().result()`.
  - CocoIndex runs `@coco.fn` coroutines (`publish_graph` -> `plan_graph_publication`) on its internal background event-loop thread (`Thread-1 (_runner)`).
  - When `plan_graph_publication` emitted its first `"indexing: disambiguation candidate search (...)"` callback on `Thread-1 (_runner)`, `YawRuntime.optimize._progress` matched `msg.startswith("indexing:")` and invoked `self.engine.get_stats(...)`, attempting to acquire `self.engine._lock` on `Thread-1 (_runner)` while `yaw-runtime-worker` held `self.engine._lock` waiting for `Thread-1 (_runner)` to finish `app.update().result()`.
- **Changed symbols & modules**:
  - `yaw/daemon/runtime.py`: Captured `worker_thread = threading.current_thread()` in `YawRuntime.optimize._run_optimize` and guarded `self.engine.get_stats(...)` inside `_progress` with `threading.current_thread() is worker_thread` so cross-thread sub-stage callbacks update `active_op` and notify `progress_callback` without acquiring `self.engine._lock`.
  - `tests/daemon/test_runtime.py`: Updated `test_cli_dspy_optimize_sandbox_isolation_and_status` to assert that `"[yaw optimize] indexing: disambiguation candidate search"` is emitted to `stderr` during `yaw dspy optimize` without deadlocking.
- **Gate results & evidence paths**:
  - `uv run pytest tests/daemon/test_runtime.py -k test_cli_dspy_optimize_sandbox_isolation_and_status`: `1 passed` in `9.69s` (previously deadlocked indefinitely).
  - Mandatory quality gates (`uv run ty check`, `uv run ruff check --fix`, `uv run ruff format`, `uv run pytest`, `git diff --check`): all passed.
- **What it does not establish**:
  - Does not claim completion of the live full-corpus GEPA optimization run until all batches finish publishing and GEPA iterations converge.

## 2026-11-07: Gaps 1–5 Specification Remedy Across Indexing & Graph Specs (from `50be17d`)

Controlling contracts: [native-cocoindex-indexing.md §8](../native-cocoindex-indexing.md#8-checkpoints), [native-cocoindex-graph.md §8](../native-cocoindex-graph.md#8-checkpoints), and [Decisions](../native-cocoindex-graph.md#decisions).

### Checkpoints (Gaps 1–5 Cutover Plan from `50be17d` — Subagent-Orchestrated)

Each unit is executed by delegating to `research` (read-only call-graph/SDK discovery), `ontologist` (W3C SKOS/OWL/SHACL & candidate equivalence verification), `simplifier` (Red-Green-Refactor TDD implementation, quality gates, and §7 `git grep` exit checks), and `critic (Model="flash")` (pre-commit audit against shortcuts and surviving dead symbols) so the parent orchestrator's context remains lean:

- [x] **C1–C2 (`simplifier` -> `critic (Model="flash")`)**: Remove dead engine helpers (`_attempted_shacl_keys`, `_graph_cache_context`, `_encode_in_batches`, `_apply_shacl_repairs`) in `engine.py` and dead `stage_cache` / `stage_caches` parameters in `program.py` (`L966–1259`).
- [x] **C4 (`simplifier` -> `critic (Model="flash")`)**: Remove legacy schema migration and backfill branches (`migrate_edges_reasoning`, `migrate_epistemic_confidence_columns`, `_entities_migrated`, `_facts_migrated`, `_edges_migrated`, `__legacy__` backfill) in `yaw/daemon/graph/storage/db.py`.
- [x] **I1 (Gap 2 — `research` -> `simplifier` -> `critic (Model="flash")`)**: Probe `CG-INDEX-1`, key `mount_table` (`managed_sqlite.py`, including `source_chunks` and `chunk_inference`) by `TableKey(db_identity, name)`, scope component mounts by `db_identity`, and point `coco.Settings(db_path=...)` at shared persistent `<state_dir>/coco_lmdb` (`indexing.py:752`).
- [x] **C5 & I3 (Gap 5 — `simplifier` -> `critic (Model="flash")`)**: Retire `CandidateStageCache`, `CandidateCacheStore`, `CandidatePayloadCodec`, `candidate_stage_cache` / `candidate_cache_size` DDL (`candidate_cache.py`), `_text_vec_cache` (`engine.py`), and `candidate_stage_cache` metric-upgrade SQL (`runtime.py:1240–1241`), replacing query embedding caching with bounded in-memory `lru_cache(maxsize=4096)` and SIMBA score/reflection checkpoints with `yaw.db` `meta` (`outputs/benchmarks/c5_candidate_cache_retirement.xml`).
- [x] **G4-VEC0 (Gap 4 — `ontologist` -> `simplifier` -> `critic (Model="flash")`)**: Probe `CG-GRAPH-1` and cut over `chunk_vec`, `entity_vec`, and `fact_vec` from ordinary `BLOB` tables to `sqlite-vec == 0.1.9` `vec0` virtual tables (`USING vec0(<pk_col> TEXT PRIMARY KEY, embedding float[N] distance_metric=cosine)`) with `DELETE`+`INSERT` virtual-table upserts (`DatabaseSink`, `rekey_facts`) and `WHERE embedding MATCH ? AND k = ?` KNN queries (`vectors.py`, `graph_ops.py`) (`outputs/benchmarks/g4_vec0_cutover.xml`).
- [x] **I2 & G3/G4/U5/P1 (Gap 1 & Gap 3 — `research` + `ontologist` -> `simplifier` -> `critic (Model="flash")`)**: Probe `CG-INDEX-2` and `CG-GRAPH-2`, pass `ChunkProcessOutput` (with `vector_f32`) directly via `coco.use_mount` into `publish_graph` -> `plan_graph_publication`, delete `load_active_chunk_inferences`, `compute_active_inference_fingerprint`, `ChunkStateTargetHandler`, `GraphDeltaCoordinator`, `_prepare_embed_req` / chunk `embed_text` replay, `clone_graph_to_memory`, and full-table `SELECT *` survivor scans, and verify bounded affected-owner planning with single-component `DatabaseSink` commits (`outputs/benchmarks/cg_index_2_direct_handoff.xml`, `outputs/benchmarks/u5_bounded_planner_cutover.xml`).
- [x] **U3/P3–P4 & I4/G5–G6 (`ontologist` -> `simplifier` -> `critic (Model="flash")`)**: Complete reverse top-$k$ candidate completeness (`U3`/`P3`), sparse CSR PPR caching (`P4`), Windows live lifecycle (`I4`), paired `64 + 64` vs. `128` fixed/live trials (`G5`), and Tier 1 / Tier 2 benchmark parity (`G6`) (`outputs/benchmarks/live_lifecycle_acceptance.xml`, `outputs/benchmarks/g5_g6_acceptance.json`).

### Completed Unit: C1–C2 & C4 (2026-11-07)

- **Changed symbols & modules** (governing contracts: [native-cocoindex-graph.md §7 & §8](../native-cocoindex-graph.md#91-remaining-cleanup-after-the-614-line-retirement) and [Decisions](../native-cocoindex-graph.md#decisions)):
  - `yaw/daemon/graph/engine.py`: Deleted `YawGraphEngine._attempted_shacl_keys`, `YawGraphEngine._graph_cache_context`, `YawGraphEngine._encode_in_batches`, `YawGraphEngine._apply_shacl_repairs`, and the unused `apply_shacl_repairs` import (`C1`).
  - `yaw/daemon/graph/inference/program.py`: Deleted `stage_cache` parameter, cache lookup/store branches, and unused `judge_stage_fingerprint` call from `YawProgram._run_judge_stage`, and deleted `stage_caches` parameter and pass-through arguments from `YawProgram.judge_semantics` while preserving `judge_stage_fingerprint` in `diagnostics.py` and `cache_identity` on `SynonymEvidenceBatch` and `RLMInstructionProposer` (`C2`).
  - `yaw/daemon/graph/storage/db.py` & `yaw/daemon/graph/storage/__init__.py`: Deleted `migrate_edges_reasoning`, `migrate_epistemic_confidence_columns`, `_entities_migrated` / `_facts_migrated` / `_edges_migrated` table migrations, `ALTER TABLE chunks` column drop/add branches, `fact_sources` / `'__legacy__'` `entity_labels` / `fact_fts` backfills in `initialize_schema`, unused `json` and `text_processing` imports in `db.py`, and the `migrate_edges_reasoning` re-export in `__init__.py` (`C4`). Retained `validate_graph_schema` obsolete-schema rejection and `_CREATE_SCHEMA_STATEMENTS` initialization.
  - `tests/daemon/graph/test_engine.py`, `tests/daemon/graph/inference/test_program.py`, `tests/daemon/graph/storage/test_db.py`, `tests/daemon/graph/storage/test_db_edges.py`, `tests/daemon/graph/storage/test_storage.py`: Updated `test_engine.py` to call `apply_shacl_repairs` directly; updated `test_per_stage_fingerprint_isolation_on_signature_mutation` in `test_program.py` to assert absence of `stage_cache` / `stage_caches` parameters while verifying per-stage fingerprint isolation and `cache_identity`; updated `test_fact_fts_indexing_and_lexical_search` (`test_db.py`) and `test_initialize_schema_creates_fresh_tables_without_legacy_migrations_or_backfills` (`test_db_edges.py`) to assert absence of deleted migration symbols and verify that `initialize_schema` does not backfill `entity_labels`, `fact_sources`, or `fact_fts`.
- **Gate results & evidence paths**:
  - `outputs/benchmarks/c1_c2_dead_code_removal.xml`: `33 passed in 43.34s` (`tests/daemon/graph/inference/test_program.py` and `tests/daemon/graph/test_engine.py -k "not live"`).
  - `outputs/benchmarks/c4_schema_migration_removal.xml`: `36 passed in 1.34s` (`tests/daemon/graph/storage/test_db.py`, `tests/daemon/graph/storage/test_db_edges.py`, `tests/daemon/graph/storage/test_scheme_removal.py`).
  - §7 removal inventory exit checks:
    - `git grep -n -e "_attempted_shacl_keys" -e "_graph_cache_context" -e "_encode_in_batches" -e "_apply_shacl_repairs" yaw/ scripts/ tests/`: `0` matches.
    - `git grep -n -w -e "stage_cache" -e "stage_caches" yaw/ scripts/ tests/` and `git grep -n -e "stage_cache" -e "stage_caches" yaw/daemon/graph/inference/ tests/daemon/graph/inference/`: `0` matches.
    - `git grep -n -e "migrate_edges_reasoning" -e "migrate_epistemic_confidence_columns" -e "_entities_migrated" -e "_facts_migrated" -e "_edges_migrated" yaw/ scripts/ tests/`: `0` matches.
  - Mandatory quality gates: `uv run --locked ty check` (`All checks passed!`), `uv run --locked ruff check --fix` (`All checks passed!`), `uv run --locked ruff format --check` (`174 files already formatted`), `uv run --locked pytest` (`261 passed, 89 deselected in 282.43s`), `git diff --check` (passed).
- **Discoveries during implementation**:
  - Unscoped `git grep -n -e "stage_cache"` without `-w` matches the substring inside `candidate_stage_cache` (in `candidate_cache.py`, `runtime.py`, and `CandidateStageCache` tests, which are scheduled for removal in Checkpoint `C5 & I3`). Whole-word `git grep -n -w -e "stage_cache" -e "stage_caches" yaw/ scripts/ tests/` returns `0` matches.
- **What it does not establish**:
  - Does not retire `CandidateStageCache` / `candidate_stage_cache` (`C5 & I3`), scope `mount_table` by `TableKey(db_identity, name)` (`I1`), cut over vector tables to `vec0` (`G4-VEC0`), or remove `load_active_chunk_inferences` / `clone_graph_to_memory` (`I2 & G3/G4/U5/P1`).

### Completed Unit: I1 (2026-11-08)

- **Changed symbols & modules** (governing contracts: [native-cocoindex-indexing.md §3 Goal I1, §5, §6 CG-INDEX-1, §7, §8](../native-cocoindex-indexing.md#3-goals)):
  - `yaw/daemon/graph/storage/db.py` & `yaw/daemon/graph/storage/__init__.py`: Implemented and exported `resolve_coco_state_root(state_dir)` (returning `<state_dir>/coco_lmdb`, resolving file paths via `.parent / "coco_lmdb"`) and `resolve_db_identity(conn, db_path=None)` (persisting `<canonical_db_path>:<uuid4_hex_12>` in `meta['db_identity']` and initializing it in `initialize_schema`).
  - `yaw/daemon/graph/storage/managed_sqlite.py`: Added `TableKey(db_identity, name)` (`NamedTuple`) and `TableSpec.db_identity`; updated `mount_table` to accept `db_identity: str = ""` and mount targets under `TableKey(db_identity, name)` in `TABLES.target_state`; updated `DatabaseSink.__call__` to fall back from `GRAPH_DB` to `"yaw/source_db"`, filter out foreign-DB actions where `action.spec.db_identity != active_db_identity`, and guard `VALIDATE_SQLITE` to graph-table `RowAction`s (excluding `source_chunks`/`chunk_inference`-only writes).
  - `yaw/daemon/graph/storage/rows.py` & `yaw/daemon/graph/planner.py`: Threaded `db_identity` into `graph_tables`, `empty_graph_tables`, and `declare_tables`, preserving `MetaRow("db_identity", db_identity)` in `meta` during table reconciliation and passing `db_identity` into `mount_table`.
  - `yaw/daemon/indexing.py` & `yaw/daemon/runtime.py`: Replaced `candidate_coco/<db_name>` (`indexing.py:bind_native_indexing`) and `indexing_coco` (`runtime.py:_build_coco_app`) with `resolve_coco_state_root(...)`; mounted `source_chunks` and `chunk_inference` via `mount_table(..., db_identity=db_identity)` (`shared_sink`); kept component mount subpaths DB-invariant (`coco.component_subpath("chunk", ...)` and `coco.component_subpath("publication")`) while passing `db_identity` as a component argument to `process_chunk`, `process_file`, `process_frozen_chunk`, `GraphPublicationLiveComponent`, and `publish_graph`; keyed `declare_chunk_state` and `ChunkStateTargetHandler.reconcile` by `(db_identity, record.chunk_id)`.
  - `yaw/daemon/graph/inference/components.py`: Added `SemanticAdjudicationInput.__post_init__` to zero out wall-clock `batch.construction_seconds` before CocoIndex input fingerprinting.
  - `docs/native-cocoindex-indexing.md` (`§5`): Updated `process_chunk`, `process_frozen_chunk`, `publish_graph`, and `bind_native_indexing` snippets to document DB-invariant `coco.component_subpath` mounts with `db_identity` passed as a component argument.
  - `tests/daemon/graph/test_indexing_overlap.py`: Added `test_shared_lmdb_cross_db_memo_reuse_and_target_isolation` (`CG-INDEX-1`) and ordered `"missing_entities"` before `"extractions"` in the loopback HTTP fixture.
- **Gate results & evidence paths**:
  - `outputs/benchmarks/cg_index_1_memo_isolation.xml`: `1 passed in 8.33s` (`tests/daemon/graph/test_indexing_overlap.py::test_shared_lmdb_cross_db_memo_reuse_and_target_isolation`, verifying `64` chunks indexed into `db_a.sqlite` then `db_b.sqlite` sharing `tmp_path / "coco_lmdb"` with `0` provider calls on `db_b`, `100%` identical rows across `source_chunks`, `chunk_inference`, and all `15` graph tables, and untouched `db_a.sqlite` state).
  - `uv run pytest --no-testmon tests/daemon/graph/test_indexing_overlap.py`: `16 passed in 32.86s`.
  - §7 removal inventory exit check: `git grep -n -e "candidate_coco" yaw/ scripts/ tests/` returned `0` matches.
  - Mandatory quality gates: `uv run --locked ty check` (`All checks passed!`), `uv run --locked ruff check --fix` (`All checks passed!`), `uv run --locked ruff format --check` (`174 files already formatted`), `uv run --locked pytest` (`124 passed, 1 deselected in 201.87s`), `git diff --check` (passed).
- **Discoveries during implementation**:
  - In `cocoindex == 1.1.0` (`cocoindex/_internal/function.py:1463-1465`), `@coco.fn(memo=True)` reserves memos via `core.reserve_memoization_async(parent_ctx._core_processor_ctx, memo_fp)`, keying `TABLES.memoization` by `(app_name, parent_ctx.stable_path, memo_fp)`. Putting `db_identity` inside `coco.component_subpath(...)` changes `parent_ctx.stable_path` between databases and causes `100%` memo misses. Keeping `coco.component_subpath("chunk", ...)` and `coco.component_subpath("publication")` DB-invariant while passing `db_identity` as a component argument causes `TABLES.components` (`fingerprint((StablePath, args))`) to execute component bodies per DB while preserving an identical parent `StablePath` for `TABLES.memoization`.
  - `SynonymEvidenceBatch.construction_seconds` (`time.perf_counter() - started` in `iter_synonym_evidence`) previously polluted `SemanticAdjudicationInput`'s CocoIndex memo fingerprint on every run, causing `judge_semantics` to miss `TABLES.memoization` across runs until normalized to `0.0` in `SemanticAdjudicationInput.__post_init__`.
- **What it does not establish**:
  - Does not retire `CandidateStageCache` / `candidate_stage_cache` (`C5 & I3`), cut over vector tables to `vec0` (`G4-VEC0`), or replace `load_active_chunk_inferences` / `GraphDeltaCoordinator` with direct `coco.use_mount` `ChunkProcessOutput` handoff (`I2 & G3/G4/U5/P1`).

### Completed Unit: C5 & I3 (2026-11-08)

- **Changed symbols & modules** (governing contracts: [native-cocoindex-graph.md §4, §5, §7, §8](../native-cocoindex-graph.md#5-design-current-state) and [native-cocoindex-indexing.md §7, §8](../native-cocoindex-indexing.md#7-removal-inventory)):
  - `yaw/daemon/graph/storage/candidate_cache.py` & `yaw/daemon/graph/storage/__init__.py`: Deleted `candidate_cache.py` (`CandidateCacheStore`, `CandidatePayloadCodec`, `CandidateStageCache`, and `candidate_stage_cache` / `candidate_cache_size` SQLite tables/triggers) and removed their re-exports from `yaw/daemon/graph/storage/__init__.py` (`C5`).
  - `yaw/daemon/graph/engine.py`: Replaced `CandidateCacheStore` / `CandidateStageCache` (`_text_vec_cache`) with an instance-scoped bounded `functools.lru_cache(maxsize=4096)` (`_encode_cached_text`) keyed by `(self._embedding_cache_namespace(instruction), text)`, removed `graph_dir.rglob("*.py")` `_candidate_cache_code_fp` hashing, and removed `CandidatePayloadCodec` / `open_score_cache` / `open_checkpoint_cache` from `_build_optimization_hooks` (`C5`).
  - `yaw/daemon/graph/inference/optimizer.py`: Removed `open_score_cache` and `open_checkpoint_cache` from `GraphOptimizationHooks`; updated `configure_simba_task_learning` to persist SIMBA calibration scores (`simba_task_score:<sha256>`) and reflection checkpoints (`reflection_checkpoint:<sha256>`) in `yaw.db`'s `meta` table via `hooks.get_meta` / `hooks.set_meta` (`C5`).
  - `yaw/daemon/runtime.py`: Replaced `DELETE FROM candidate_stage_cache WHERE stage = 'simba_task_score'` on metric upgrade with `DELETE FROM meta WHERE key LIKE 'simba_task_score:%'` (`I3`).
  - `tests/daemon/graph/test_candidate_cache.py`, `tests/daemon/graph/candidate_cache_worker.py`, `tests/daemon/graph/inference/test_reflection.py`, `tests/daemon/graph/inference/test_synonym_packets.py`, `tests/daemon/graph/test_engine_edges.py`, `tests/daemon/graph/test_optimization_progress.py`, `tests/daemon/graph/test_runtime_contracts.py`, `tests/daemon/graph/test_simba_learning.py`, `tests/daemon/test_runtime.py`: Deleted obsolete `tests/daemon/graph/storage/test_candidate_cache.py` and `tests/daemon/graph/test_cache_edges.py`; added `test_query_embedding_lru_cache_and_absence_of_retired_cache_tables` in `tests/daemon/graph/test_candidate_cache.py` and updated reflection, SIMBA, synonym packet, engine edge, and runtime tests to verify `meta`-backed checkpoints/scores, `lru_cache(maxsize=4096)` query embedding reuse, and shared `coco_lmdb` memoization.
- **Gate results & evidence paths**:
  - `outputs/benchmarks/c5_candidate_cache_retirement.xml`: `108 passed in 173.56s` (`tests/core/test_architecture.py`, `tests/daemon/graph/test_candidate_cache.py`, `tests/daemon/graph/inference/test_reflection.py`, `tests/daemon/graph/inference/test_synonym_packets.py`, `tests/daemon/graph/test_engine_edges.py`, `tests/daemon/graph/test_optimization_progress.py`, `tests/daemon/graph/test_runtime_contracts.py`, `tests/daemon/graph/test_simba_learning.py`, `tests/daemon/test_runtime.py`).
  - §7 removal inventory exit checks:
    - `git grep -n -e "CandidateStageCache" -e "CandidateCacheStore" -e "CandidatePayloadCodec" -e "candidate_stage_cache" -e "candidate_cache_size" -e "_text_vec_cache" -e "open_score_cache" -e "open_checkpoint_cache" yaw/ scripts/ tests/`: `0` matches.
    - `git grep -n -w -e "stage_cache" -e "stage_caches" yaw/ scripts/ tests/`: `0` matches.
  - Mandatory quality gates: `uv run --locked ty check` (`All checks passed!`), `uv run --locked ruff check --fix` (`All checks passed!`), `uv run --locked ruff format --check` (`171 files already formatted`), `uv run --locked pytest` (`112 passed, 85 deselected in 276.13s`), `git diff --check` (passed).
- **Discoveries during implementation**:
  - All `RLMInstructionProposer` checkpoint payloads (`request:<digest>`, `evidence:<digest>`, `proposal:<digest>`, and `dump_simba_state()`) are pure JSON-serializable dictionaries containing span offsets rather than raw corpus text, allowing direct `json.dumps` / `json.loads` persistence in `yaw.db`'s `meta` table without `CandidatePayloadCodec`.
- **What it does not establish**:
  - Does not cut over `chunk_vec`, `entity_vec`, and `fact_vec` to `sqlite-vec` `vec0` virtual tables (`G4-VEC0`) or replace `load_active_chunk_inferences` / `clone_graph_to_memory` with direct `coco.use_mount` `ChunkProcessOutput` handoff and affected-owner planning (`I2 & G3/G4/U5/P1`).

### Completed Unit: G4-VEC0 (2026-11-08)

- **Changed symbols & modules** (governing contracts: [native-cocoindex-graph.md §2.1, §4, §5, §6 CG-GRAPH-1, §7, §8, §9](../native-cocoindex-graph.md#5-design-current-state)):
  - `yaw/daemon/graph/storage/db.py` & `yaw/daemon/graph/storage/__init__.py`: Replaced `CREATE TABLE IF NOT EXISTS chunk_vec / entity_vec / fact_vec` in `_CREATE_SCHEMA_STATEMENTS` with `CREATE VIRTUAL TABLE IF NOT EXISTS ... USING vec0(<pk_col> TEXT PRIMARY KEY, embedding float[768] distance_metric=cosine)`; added and exported `DEFAULT_VEC_DIMENSION = 768`, `vec0_table_dimension(conn, table_name)`, and `ensure_vec0_table(conn, table_name, pk_col, dim, *, column="embedding", metric="cosine")`.
  - `yaw/daemon/graph/storage/managed_sqlite.py`: Added `VecSpec(column="embedding", dim=None, distance_metric="cosine")` and `TableSpec.vec`; updated `create_table` to delegate `spec.vec` tables to `ensure_vec0_table`; updated `DatabaseSink.__call__` to ensure `vec0` table dimension (`len(blob) // 4`) and execute `DELETE` + `INSERT` without `ON CONFLICT DO UPDATE` whenever `spec.fts_unindexed is not None or spec.vec is not None`; updated `TableHandler.reconcile` to preserve prior `vec.dim` when `desired.vec.dim is None` (`empty_graph_tables()` / empty vector table on a batch) and set `create = True` when `desired.vec.dim` changes; added `vec: VecSpec | None = None` to `mount_table`.
  - `yaw/daemon/graph/storage/rows.py`: Added `DesiredTable.vec` and `_VEC_TABLE_NAMES = frozenset({"chunk_vec", "entity_vec", "fact_vec"})`; attached `VecSpec` in `graph_tables` and `empty_graph_tables`; updated `declare_tables` to propagate `shared_vec_dim` across `chunk_vec`, `entity_vec`, and `fact_vec`.
  - `yaw/daemon/graph/storage/vectors.py`: Updated `_upsert_vectors` to deduplicate input `(item_id, vec)` pairs (last-write-wins), call `ensure_vec0_table(conn, table, id_col, dim)`, and execute `DELETE` + `INSERT`; updated `_search_knn` to return `[]` on empty tables and execute `WITH knn AS MATERIALIZED (SELECT {id_col}, distance FROM {table} WHERE embedding MATCH ? AND k = ? ORDER BY distance) SELECT {id_col}, (1.0 - distance) AS similarity FROM knn ORDER BY distance ASC, {id_col} ASC` with `k = min(int(top_k), 4096)`.
  - `yaw/daemon/graph/storage/graph_ops.py` & `yaw/daemon/graph/planner.py`: Updated `find_entity_synonym_candidates` to materialize outer rows before inner `vec0` `MATCH` KNN queries (`k = min(int(top_k) + 1, 4096)`); updated `rekey_facts` to replace `ON CONFLICT(fact_id) DO UPDATE` on `fact_vec` with `SELECT` + `min(bytes(old_vec), bytes(new_vec))` + `DELETE` + `INSERT`; updated `clone_graph_to_memory` to sync `stage_conn` `vec0` dimensions via `ensure_vec0_table(stage_conn, vec_table, pk_col, vec0_table_dimension(conn, vec_table))` before copying rows.
  - `docs/native-cocoindex-graph.md` (`§5`): Updated `VecSpec` / `TableSpec` definitions, unquoted `USING vec0(...)` column DDL, and `WITH knn AS MATERIALIZED` `_search_knn` SQL snippet.
  - `tests/daemon/graph/storage/test_vectors.py`, `tests/daemon/graph/storage/test_db.py`, `tests/daemon/graph/storage/test_canonical_evidence.py`, `tests/test_epistemic_confidence.py`: Added `test_vec0_managed_transaction_upsert_rollback_and_knn_parity` (`CG-GRAPH-1`) and updated schema/vector fixtures to use `vec0` virtual tables.
- **Gate results & evidence paths**:
  - `outputs/benchmarks/g4_vec0_cutover.xml`: `22 passed in 2.66s` (`tests/daemon/graph/storage/test_vectors.py`, `tests/daemon/graph/storage/test_db.py`, `tests/daemon/graph/storage/test_publication.py`).
  - §7 removal inventory exit check:
    - `git grep -n -e "vec_distance_cosine" -e "CREATE TABLE IF NOT EXISTS chunk_vec" -e "CREATE TABLE IF NOT EXISTS entity_vec" -e "CREATE TABLE IF NOT EXISTS fact_vec" yaw/`: `0` matches.
  - Mandatory quality gates: `uv run --locked ty check` (`All checks passed!`), `uv run --locked ruff check --fix` (`All checks passed!`), `uv run --locked ruff format --check` (`172 files already formatted`), `uv run --locked pytest` (`105 passed, 4 deselected in 167.15s`), `git diff --check` (passed).
- **Discoveries during implementation**:
  - `sqlite-vec == 0.1.9` `vec0` module constructor rejects double-quoted column identifiers inside `USING vec0(...)` (`OperationalError: Double-quoting is not allowed in vec0() options`); validating identifiers against `[A-Za-z_][A-Za-z_0-9]*` and emitting unquoted `{pk_col} TEXT PRIMARY KEY, {column} float[{dim}] distance_metric={metric}` inside `USING vec0(...)` is required.
  - `sqlite-vec == 0.1.9` `vec0` virtual tables reject `INSERT OR REPLACE` (`UNIQUE constraint failed`) when a key already exists, reject `ON CONFLICT DO UPDATE` (`OperationalError: UPSERT not implemented for virtual table`), cap `k <= 4096`, and require `WITH knn AS MATERIALIZED (...) ORDER BY distance ASC, <id_col> ASC` for deterministic tie-breaking because `vec0`'s `xBestIndex` only consumes `WHERE embedding MATCH ? AND k = ?` when `ORDER BY` has a single `distance` term.
- **What it does not establish**:
  - Does not replace `load_active_chunk_inferences`, `clone_graph_to_memory`, or full-table `SELECT *` survivor scans with direct `coco.use_mount` `ChunkProcessOutput` handoff and affected-owner planning (`I2 & G3/G4/U5/P1`), nor complete `U3/P3–P4 & I4/G5–G6`.

### Completed Unit: I2 & G3/G4/U5/P1 (2026-11-08)

- **Changed symbols & modules** (governing contracts: [native-cocoindex-indexing.md §2.1 CG-INDEX-2, §3.1, §3.3, §5, §6, §7, §8](../native-cocoindex-indexing.md#5-design-current-state) and [native-cocoindex-graph.md §2.1 CG-GRAPH-2, §3.3, §3.4, §5, §6, §7, §8](../native-cocoindex-graph.md#5-design-current-state)):
  - `yaw/daemon/graph/inference/components.py` & `yaw/daemon/indexing.py`: Defined `ChunkInference` and `ChunkProcessOutput(pointer, text, inference, embedding)` with `@property def chunk_id` and `@property def vector_f32` in `components.py` and re-exported from `indexing.py` (preserving the 6-column `chunk_inference` schema without a duplicate `embedding` column).
  - `yaw/daemon/indexing.py` & `yaw/daemon/runtime.py`: Updated `process_chunk`, `process_file`, and `process_frozen_chunk` to return `ChunkProcessOutput` / `tuple[ChunkProcessOutput, ...]` via `coco.use_mount`; decorated `publish_graph` with `@coco.fn` (without `memo=True`) accepting `chunk_outputs: tuple[ChunkProcessOutput, ...]`; replaced `GraphPublicationLiveComponent`, `_DeltaNotifyingLiveItems`, `_SubscriberProxy`, `CHUNK_STATE_TARGET`, `ChunkStateTargetHandler`, `declare_chunk_state`, `GraphDeltaCoordinator`, `get_graph_delta_coordinator`, `compute_active_inference_fingerprint`, and `_is_published_graph_aligned` with `CorpusIndexingLiveComponent`, `_CorpusFeedSubscriber`, and `INDEXING_ERROR_SINK` (`coco.ContextKey` wired into `YawRuntime` to propagate original component exceptions across `LiveComponentOperator.update_full()`).
  - `yaw/daemon/graph/planner.py`: Deleted `load_active_chunk_inferences`, `clone_graph_to_memory`, `_prepare_embed_req`, and Step 1 chunk `embed_text` replay; updated `plan_graph_publication` to consume `chunk_outputs: Sequence[ChunkProcessOutput]` directly in memory, deserialize `OpenIERecord` / `PredictorTrace` JSON only for `outputs_to_process`, and seed `:memory:` `staging_conn` via `load_affected_neighborhood` and `snapshot_table_rows`.
  - `yaw/daemon/graph/storage/graph_ops.py` & `yaw/daemon/graph/storage/__init__.py`: Implemented and exported `load_affected_neighborhood` using only indexed `WHERE ... IN (...)`, `WHERE label_norm >= ? AND label_norm < ?`, and `vec0` `MATCH` queries on `ro_conn` (expanding `seed_entity_ids` across `entity_vec` KNN, lexical/token-norm overlap, transitive `skos:broader` hierarchy, co-hyponym siblings, 1-hop `skos:related`/`owl:differentFrom` neighbors, and incident `facts`/`edges`/`chunks`); added reverse top-$k$ symmetry in `find_entity_synonym_candidates` when `entity_ids` is scoped.
  - `yaw/daemon/graph/storage/rows.py` & `yaw/daemon/graph/storage/managed_sqlite.py`: Added `snapshot_table_rows`, `empty_delta_graph_tables`, `DesiredTable.deleted_keys`, `DesiredTable.is_delta`, `TableSpec.is_delta`, and `DELETE_ROW`; updated `graph_tables` and `empty_graph_tables` to compute delta `changed_rows` and `deleted_keys` against `seeded_rows_by_table`; updated `declare_tables` and `RowHandler.reconcile` so `is_delta=True` ignores `coco.NON_EXISTENCE` for omitted survivor keys while declaring `DELETE_ROW` (`RowAction(spec, key, None)`) for `deleted_keys` inside the single atomic `DatabaseSink` transaction.
  - `tests/daemon/graph/test_indexing_overlap.py`, `tests/daemon/graph/test_batch_equivalence.py`, `tests/daemon/graph/storage/test_publication.py`, `tests/daemon/graph/test_evidence_revisions.py`: Added `test_direct_use_mount_chunk_output_handoff_without_chunk_state_target` (`CG-INDEX-2`), `test_incremental_publication_without_memory_clone_or_full_survivor_scan` (`CG-GRAPH-2`), and `test_delta_declare_tables_preserves_survivors_and_applies_deletions`; updated publication and revision tests to use `ChunkProcessOutput`.
- **Gate results & evidence paths**:
  - `outputs/benchmarks/cg_index_2_direct_handoff.xml`: `17 passed` (`tests/daemon/graph/test_indexing_overlap.py`) and `27 passed` in `tests/daemon/test_runtime.py`.
  - `outputs/benchmarks/u5_bounded_planner_cutover.xml`: `9 passed in 31.74s` (`tests/daemon/graph/test_batch_equivalence.py` and `tests/daemon/graph/storage/test_publication.py`).
  - `uv run --locked pytest --no-testmon tests/daemon/graph/test_evidence_revisions.py tests/daemon/ingestion/test_source_targets.py`: `22 passed in 39.80s`.
  - §7 removal inventory exit checks:
    - `git grep -n -e "ChunkStateAction" -e "ChunkStateRecord" -e "GraphDeltaCoordinator" -e "get_graph_delta_coordinator" -e "ChunkStateTargetHandler" -e "CHUNK_STATE_TARGET" -e "declare_chunk_state" -e "compute_active_inference_fingerprint" -e "_is_published_graph_aligned" -e "GraphPublicationLiveComponent" -e "load_active_chunk_inferences" -e "clone_graph_to_memory" -e "_prepare_embed_req" -e "_SubscriberProxy" -e "_DeltaNotifyingLiveItems" -e "recovery_nonce" yaw/ scripts/ tests/`: `0` matches.
  - Mandatory quality gates: `uv run --locked ty check` (`All checks passed!`), `uv run --locked ruff check --fix` (`All checks passed!`), `uv run --locked ruff format --check` (`172 files already formatted`), `uv run --locked pytest` (`133 passed, 12 deselected in 212.44s`), `git diff --check` (passed).
- **Discoveries during implementation**:
  - In `cocoindex == 1.1.0`, `coco.mount_target` wraps `declare_target_state_with_child` in its own child component (`use_mount(subpath, declare_target_state_with_child, target_state)`), causing `TableAction`s to flush to `DatabaseSink.__call__` per table during `mount_table` before `publish_graph` declares child `RowAction`s. Representing `deleted_keys` as child target states declared with `DELETE_ROW` in `declare_tables` ensures all upserts and deletions across all `15` graph tables execute atomically in the single post-`publish_graph` `RowAction` batch in `DatabaseSink.__call__` before `validate_graph_state(conn)` runs.
  - When a non-incremental rebuild (`not incremental and published_inputs`) follows an `is_delta=True` incremental batch, CocoIndex's LMDB only tracks the keys declared by the most recent `is_delta=True` run; snapshotting `seeded_rows_by_table = snapshot_table_rows(ro_conn)` on non-incremental rebuilds over an existing database ensures all stale keys from earlier batches are explicitly declared with `DELETE_ROW` and removed atomically.
- **What it does not establish**:
  - Does not complete `U3/P3–P4 & I4/G5–G6` (`> 10` entity reverse-neighbor fixture, sparse CSR PPR caching, Windows live lifecycle, paired `64 + 64` vs. `128` multi-trial benchmarks, and Tier 1 / Tier 2 benchmark parity).

### Completed Unit: U3/P3–P4 & I4/G5–G6 (2026-11-08)

- **Changed symbols & modules** (governing contracts: [native-cocoindex-indexing.md §8 & §9 Checkpoint I4](../native-cocoindex-indexing.md#8-checkpoints) and [native-cocoindex-graph.md §8 & §9 Checkpoints U3, P3–P4, G5–G6](../native-cocoindex-graph.md#8-checkpoints)):
  - `yaw/daemon/graph/storage/graph_ops.py`: Set `k_wide = 4096` (bounded by `AND distance <= max_dist`) in both `find_entity_synonym_candidates` (`L153`) and `load_affected_neighborhood` (`L1661`), and expanded `k_self = min(int(synonymy_top_k) + 1, 4096)` over all `core_entity_ids` via a batched `SELECT entity_id, embedding FROM entity_vec WHERE entity_id IN (...)` query in `load_affected_neighborhood` (`L1674–1695`) (`U3`/`P3`).
  - `yaw/daemon/graph/engine.py` & `yaw/daemon/runtime.py`: Added `YawGraphEngine._csr_cache_key: tuple[Any, ...] | None = None` and `_current_csr_cache_key(self) -> tuple[Any, ...]` keyed by `(PRAGMA data_version, int(self.conn.total_changes), get_meta(self.conn, "graph_input_fingerprints") or "", float(self.beta), policy)`; updated `_ensure_csr_graph`, `_canonical_retrieval_hook`, `report_indexing_batch`, `index_chunks`, and `YawRuntime._refresh_published_state` to reuse `self._csr_graph` across read-only queries and invalidate `_csr_graph`, `_csr_cache_key`, and `_query_retrieval_memo` on any committed graph mutation (`P3–P4`).
  - `yaw/daemon/runtime.py`, `yaw/daemon/indexing.py`, & `yaw/daemon/graph/storage/hydration.py`: Added `_CorpusFeedSubscriber.mark_ready` delegation, caught transient `ValueError` in `process_file` when a watched file is deleted or renamed mid-flight on Windows (`if not target_file.is_file(): return ()`), guarded `hydrate_chunk_lines` against concurrently unlinked files (`except FileNotFoundError: return ""`), and added a retry loop on `PermissionError` in `YawRuntime._emit_reload_signal` for Windows NTFS file-watcher reads (`I4`).
  - `tests/daemon/graph/test_batch_equivalence.py`: Added `test_reverse_top_k_asymmetric_candidate_completeness_64_plus_64_equals_128` (`14` distinct entities, `synonymy_top_k=5`, verifying asymmetric reverse top-$k$ candidate discovery and 15-table canonical equivalence between `64 + 64` and `128`) and `test_g5_g6_paired_64_plus_64_vs_128_acceptance_report` (`N = 20` paired fixed-evidence trials: `10` `cold_memo` + `10` `warm_stage1_memo` with alternated `incremental -> reference` and `reference -> incremental` arm ordering, writing `outputs/benchmarks/g5_g6_acceptance.json`) (`U3`, `G5–G6`).
  - `tests/daemon/graph/test_engine.py`: Added `test_csr_graph_cached_across_queries_and_invalidated_on_graph_commit` proving CSR reuse across consecutive queries (`csr1 is csr2`, `load_graph_topology` called once) and immediate invalidation across `report_indexing_batch`, same-connection commits (`total_changes`), and external-connection commits (`PRAGMA data_version`) (`P3–P4`).
  - `tests/daemon/test_runtime.py`: Added `test_windows_live_lifecycle_create_modify_rename_delete_and_lock_release` exercising live `doc_a.md` / `doc_b.md` creation, rapid consecutive in-place edits (`UniqueA_v1` -> `UniqueA_v2` -> `UniqueA_v3`), rename (`doc_a.md` -> `doc_a_renamed.md`), deletion (retracting `UniqueA_v3` while preserving `SharedEntity` and `UniqueB` with `shacl_conforms == True`), and clean Windows file-lock release on `runtime.stop()` (`I4`).
- **Gate results & evidence paths**:
  - `outputs/benchmarks/live_lifecycle_acceptance.xml`: `22 passed, 23 deselected in 39.31s` (`tests/daemon/test_runtime.py` and `tests/daemon/graph/test_indexing_overlap.py -k "live or overlap or crash_resume or partition_frozen"`).
  - `outputs/benchmarks/g5_g6_acceptance.json`: `N = 20` paired `64 + 64` vs. `128` trials (`10` `cold_memo` + `10` `warm_stage1_memo`):
    - `R_time`: `mean = 1.1376`, `median = 1.1422`, `min = 1.0793`, `max = 1.1837` (`within_0_80_to_1_25 = true`).
    - `R_graph`: `mean = 1.1067`, `median = 1.0964`, `min = 1.0872`, `max = 1.1703` (`within_0_80_to_1_25 = true`).
    - `canonical_15_table_parity_trials_passed = 20/20`, `shacl_conforms = true`, `batch2_reembedded_batch1_chunks = 0`, `batch2_unindexed_readonly_selects = 0`, `batch2_unaffected_row_writes = 0`, `peak_rss_bytes = 481595392 <= 1073741824`.
    - `verify_paper_parity_gate.py` (`--datasets popqa --mode retrieval`): `returncode = 0` (`Recall@5 Δ = +0.0400`, `Overall Gate Status: PASS`, `live_endpoint_reachable = true`).
  - Mandatory quality gates: `uv run --locked ty check` (`All checks passed!`), `uv run --locked ruff check --fix` (`All checks passed!`), `uv run --locked ruff format --check` (`172 files already formatted`), `uv run --locked pytest` (`20 passed, 3 deselected in 463.97s` following `51 passed` in the prior testmon sweep), `git diff --check` (passed).
- **Discoveries during implementation**:
  - On Windows NTFS, `localfs.walk_dir(..., live=True)` can emit a file event while a file is concurrently renamed or unlinked, or hold a brief read handle on `reload.signal`; guarding `process_file` against `not target_file.is_file()` on `ValueError`, guarding `hydrate_chunk_lines` against `FileNotFoundError`, and retrying `PermissionError` in `_emit_reload_signal` eliminates transient Windows file-watcher races without masking genuine errors on existing files.
  - When `_ensure_csr_graph()` builds `self._csr_graph` for the first time during an ensemble query (`_csr_graph is None` while `_csr_cache_key == cache_key`), clearing `_query_retrieval_memo` or resetting `_last_memo_question = None` would evict the active question's ensemble memo; guarding memo invalidation on `self._csr_cache_key != cache_key` preserves intra-query Pareto ensemble memoization while invalidating immediately on any graph commit.
- **What it does not establish**:
  - Loopback paired trials (`N = 20`) and the single-dataset (`popqa`) retrieval parity gate establish `64 + 64` vs. `128` canonical 15-table equivalence, `0.80 <= R_time, R_graph <= 1.25` bounded incremental overhead, and Windows live file-watch lifecycle conformance; full 7-dataset Tier 2 holdout evaluation remains gated on a dedicated full-corpus live benchmark run.

## 2026-11-08: W3C Assertion Cleanup, Strict `JudgeLabels` Equivalence, Non-Transitive `1`-Hop `skos:broader` S27 Disjointness & Deterministic SHACL Healing (from `c98aebe`)

Controlling contracts: [w3c-improvements.md §12, §13, §Decisions](../w3c-improvements.md#12-strict-judgelabels-identity-equivalence--canonical-gemma-4-3-stage-cascade-without-python-token-overrides-or-auto-merge-gates), [native-cocoindex-graph.md §4, §Decisions](../native-cocoindex-graph.md#decisions), and [epistemic-confidence-spec.md §2.4, §Decisions](../epistemic-confidence-spec.md#24-s27-disjointness-resolution-cagle-reification-pattern--direct-1-hop-bayes-factor).

### Checkpoints (2026-11-08 — `c98aebe`)

- [x] **C1 (Documentation & Rule Promotion)**: Promote Invariants `W1–W5`, `CG-W3C-1`, `CG-W3C-2`, and the Removal Inventory into [w3c-improvements.md](../w3c-improvements.md), [native-cocoindex-graph.md](../native-cocoindex-graph.md), [epistemic-confidence-spec.md](../epistemic-confidence-spec.md), [graph-storage.md](../../.agents/rules/graph-storage.md), and [inference.md](../../.agents/rules/inference.md).
- [x] **C2 (Strict `JudgeLabels` Equivalence, Retirement of `broader_tc` Rule W3, and Retirement of Python Token Overrides / Auto-Merge Gates)**: TDD implementation of `CG-W3C-1`, `2`-branch `equiv-branch` GBNF grammar (`max_reasoning_chars = 360`), `judge_semantics` `@coco.fn(memo=True, version=2)`, removal of `query_broader_tc` / `broader_tc` / `Rule W3`, and removal of `_is_acronym`, `is_modifier_head_specialization`, `_has_disjoint_non_acronym_tokens`, `auto_merge_cosine_gate`, and `auto_merge_difflib_gate`.
- [x] **C3 (Tier 3 Deterministic Statement Definition Synthesis, Irreflexive Literal Filtering, `1`-Hop S27 & Single-Surface `skos:prefLabel`)**: TDD implementation of `CG-W3C-2`, `clean_surface_label` and `is_irreflexive_statement` in `yaw.core`, Tier 3 deterministic synthesis in `canonicalize_fact_definitions`, universal literal/entity irreflexivity across `evidence.py` / `planner.py` / `graph_ops.py` / `shacl.py` / `invariants.py`, single-surface `skos:prefLabel` alignment in `normalization.py` / `planner.py` / `graph_ops.py`, direct `1`-hop `ConceptShape/S27-disjointBroaderRelated` + fixed-point `ConceptShape/broaderCycle` healing in `shacl.py`, and post-repair `auto_heal_deterministic_shacl` in `plan_graph_publication`.
- [x] **C4 (Critic Audit, Review Ledger Update, Full Quality Gates & Live Optimization Restart)**: Critic audit (`0` blocking findings), review ledger updates in [shacl-healer-publication.md](shacl-healer-publication.md) and [incremental-publication-work.md](incremental-publication-work.md), full quality gates, atomic commit, and restart of `uv run yaw dspy optimize --resume`.

### Completed Implementation Unit (2026-11-08)

- **Changed symbols & modules**:
  - See full entry in [shacl-healer-publication.md](shacl-healer-publication.md#2026-11-08-w3c-assertion-cleanup-strict-judgelabels-equivalence-non-transitive-1-hop-skosbroader-s27-disjointness--deterministic-shacl-healing-from-c98aebe).
- **Gate results & evidence paths**:
  - `git grep -n -e "_is_acronym" -e "is_modifier_head_specialization" -e "_has_disjoint_non_acronym_tokens" -e "auto_merge_cosine_gate" -e "auto_merge_difflib_gate" -e "_AUTO_MERGE_COSINE_GATE" -e "_AUTO_MERGE_DIFFLIB_GATE" -e "query_broader_tc" -e "broader_tc" -e "broaderTransitiveCapExceeded" yaw/ tests/` -> `0` matches.
  - `uv run --locked ty check`, `uv run --locked ruff check --fix`, `uv run --locked ruff format --check`, `uv run --locked pytest` (`741 passed` across `tests/daemon/graph/`, `tests/test_epistemic_confidence.py`, and `tests/core/`), and `git diff --check`: all passed.
- **What it does not establish**:
  - Does not claim completion of the live full-corpus GEPA optimization run until all batches finish publishing and GEPA iterations converge.

## 2026-11-08: Canonical `optimize_*` Sandbox, Default-Resume + `--fresh` CLI, CocoIndex `fingerprint_object`, Passive-Endpoint Homograph Fix & Epic 1 Gate Audit (from `b788578`)

Controlling contracts: [epic-1-hipporag-daemon.md §2 & §Decisions](../epic-1-hipporag-daemon.md#decisions), [native-cocoindex-indexing.md §3, §5, & §Decisions](../native-cocoindex-indexing.md#decisions), and [native-cocoindex-graph.md §3](../native-cocoindex-graph.md#3-goals).

### Checkpoints (2026-11-08 — `b788578`)

| Checkpoint | Subagent | Scope | Gate / Status |
| --- | --- | --- | --- |
| **C1** | `self` (`Workspace="inherit"`) | Promote canonical `optimize_*` sandbox (`optimize_yaw.db`, `optimize_status.json`, `optimize_program.json`, `optimize_program_audit.json`, `optimize_program_eval.json`, `optimize_program_pools.json`, `optimize_program_telemetry/`, `optimize_telemetry.json`, `optimize_kg_frames/`, `optimize_coco/<dataset>/`), default-resume + `--fresh` CLI contract (`--resume` retired), unified `<base_dir>/coco_lmdb` state root, CocoIndex `fingerprint_object(payload).hex()` (`graph_plan_fingerprints`), and retirement of `_proposer_telemetry.json` / rename of `_graph_rlm_audit.json` -> `_audit.json` into `AGENTS.md`, `README.md`, `docs/epic-1-hipporag-daemon.md`, `docs/native-cocoindex-indexing.md`, and `docs/reviews/incremental-publication-work.md`. | [x] Complete (`git diff --check`, `uv run --locked pytest tests/core/test_architecture.py --no-testmon -q`, `0` matches for `proposer_telemetry` / `_graph_rlm_audit.json` in `docs/epic-1-hipporag-daemon.md`). |
| **C2** | `simplifier` (`Workspace="inherit"`) | TDD tests and implementation for `YawConfig.for_optimization` / `for_benchmark` (`dataclasses.replace`), `write_via_temp_atomic`, `parse_dataset_list`, `resolve_coco_state_root` (`<base_dir>/coco_lmdb`), passive-endpoint homograph neighborhood completeness in `load_affected_neighborhood` (`CG-OPT-2`), CocoIndex `fingerprint_object` + `graph_plan_fingerprints`, `YawRuntime._create_bound_engine`, `YawGraphEngine.emit_indexing_progress` logging, default-resume + `--fresh` CLI in `yaw/daemon/cli.py` (`CG-OPT-1`), and `_audit.json` sidecar cleanup. | [x] Complete (`816` non-live + live tests passed across `tests/core/` and `tests/daemon/`, `CG-OPT-1` & `CG-OPT-2` passed, `0` matches across all C2 removal inventory exit checks). |
| **C3** | `simplifier` (`Workspace="inherit"`) | TDD tests and implementation for `scripts/watchdog.py` canonical `optimize_*` target discovery (`run_tag == "canonical"`, durable `meta['optimization_run']` `"started_at"`), `write_via_temp_atomic` + `get_logger("yaw.watchdog")`, and removal of `5` dead/zombie code blocks (`graph_cache` per-chunk JSON double-parse loop, `read_config_info`, `LEGACY_FIG_FILES`, `legacy` untagged-eval fallback, and `diff_peer_map`). | [x] Complete (`19 passed` in `tests/scripts/test_watchdog.py` & `test_watchdog_edges.py`, `0` matches across all 8 Removal Inventory exit checks, net production LOC delta `-277` across `yaw/` and `scripts/watchdog.py`). |
| **C4** | `critic` (`Model="flash"`) + `self` | Critic audit against all Removal Inventory exit checks, full mandatory quality gates (`ty check`, `ruff check --fix`, `ruff format`, `pytest`, `git diff --check`), `>= 250` net LOC reduction verification, atomic commit, migration of `.yaw/yaw_optimize_20261108_075400.db` -> `.yaw/optimize_yaw.db`, and resume of `uv run yaw dspy optimize` from `448/2387` chunks to completion. | [x] Complete (`critic` `00b62ea7-1b8e-47a2-a698-574c4894d1dc` `ZERO FINDINGS`; all 8 Removal Inventory exit checks `0` matches; `-277` net production LOC; `ty check`, `ruff`, `pytest`, and `git diff --check` passed; `.yaw/optimize_yaw.db` resumed). |
| **C5** | `self` + `critic` | Promote `.yaw/optimize_program.json` -> `.yaw/compiled_program.json`, run live `64 + 64` vs. `128` batch invariance gate (`Goal A` & `Goal B`), run clean-slate 7-dataset HippoRAG 2 slice-based paper parity gate (`verify_paper_parity_gate.py --compiled-program .yaw/compiled_program.json --mode eval --fresh`), tick `D1-A6` in `docs/epic-1-hipporag-daemon.md`, and record evaluation results as the decision gate for Epic 3 full-corpus scaling (after Epic 2 FastMCP feature completion). | [ ] Pending |

### Completed Unit: Checkpoint C1 (2026-11-08)

- **Changed files**:
  - `AGENTS.md`: Updated `## Mandatory Quality Gates & Commands` to document default-resume `uv run yaw dspy optimize` in `.yaw/optimize_yaw.db` and `--fresh` to wipe prior `optimize_*` state.
  - `README.md`: Updated CLI snippet and prose to reflect default-resume in `.yaw/optimize_yaw.db` (producing `.yaw/optimize_program.json` and `optimize_program_*` sidecars) and explicit `--fresh` wipe semantics.
  - `docs/epic-1-hipporag-daemon.md`: Updated `§2` (`Sprint 6`) and appended `2026-11-08` `## Decisions` entry for canonical `optimize_*` sandbox prefixing, default-resume + `--fresh` CLI semantics, unified `.yaw/coco_lmdb`, CocoIndex `fingerprint_object(payload).hex()` (`graph_plan_fingerprints`), and retirement of `_proposer_telemetry.json` / rename of `_graph_rlm_audit.json` -> `_audit.json`.
  - `docs/native-cocoindex-indexing.md`: Updated `§4` (`#3-revision-identity-memo-and-target-isolation`), `§5` (`resolve_coco_state_root` and `graph_plan_fingerprints`), and `## Decisions` (`2026-11-08`) to document `<base_dir>/coco_lmdb` (`.yaw/coco_lmdb`), `<base_dir>/optimize_coco/<dataset>`, and CocoIndex `fingerprint_object` fingerprinting.
  - `docs/reviews/incremental-publication-work.md`: Added the `2026-11-08` Checkpoint C1–C5 tracking table and Checkpoint C1 completion log.
- **Gate results**:
  - `git diff --check`: passed (`0` whitespace/conflict errors).
  - `uv run --locked pytest tests/core/test_architecture.py --no-testmon -q`: `15 passed`.
  - `git grep -n -e "proposer_telemetry" -e "_graph_rlm_audit.json" docs/epic-1-hipporag-daemon.md`: `0` matches.
- **What it does not establish**:
  - Documentation promotion only; production code and test implementation follow in Checkpoints C2–C5.

### Completed Unit: Checkpoint C2 (2026-11-08)

- **Changed symbols & modules** (governing contracts: [epic-1-hipporag-daemon.md §2 & §Decisions](../epic-1-hipporag-daemon.md#decisions), [native-cocoindex-indexing.md §3, §5, & §Decisions](../native-cocoindex-indexing.md#decisions), and [native-cocoindex-graph.md §3](../native-cocoindex-graph.md#3-goals)):
  - `yaw/core/config.py` & `yaw/core/__init__.py`: Added and exported `write_via_temp_atomic(path, write_fn)` and `parse_dataset_list(raw)`; delegated `write_text_atomic` to `write_via_temp_atomic`; updated `_DEFAULT_RELATIVE_PATHS["state_dir"]` to `".yaw/coco_lmdb"`; refactored `YawConfig.for_optimization(self, db_override=None)` (removing `datestamp`) and `YawConfig.for_benchmark` to use `dataclasses.replace(self, ...)` with canonical `.yaw/optimize_yaw.db`, `.yaw/optimize_status.json`, `.yaw/optimize_program.json`, and `.yaw/coco_lmdb` paths.
  - `yaw/daemon/graph/storage/db.py`: Updated `resolve_coco_state_root(state_dir)` so `"coco_lmdb"` returns `resolved`, `"coco_state*"` or `.db`/file paths return `resolved.parent / "coco_lmdb"`, and other directories return `resolved / "coco_lmdb"` without nesting `coco_lmdb/coco_lmdb`.
  - `yaw/daemon/graph/storage/graph_ops.py`: Updated `load_affected_neighborhood` (`CG-OPT-2`) to include same-`label_norm` peers of `core_entity_ids`, run a fixed-point loop over `2d`/`2e` promoting passive-endpoint homograph peers in `affected_entity_ids` into `core_entity_ids` (ignoring `passage` edge chunk endpoints), and copy all `passage` edges incident to `affected_entity_ids` plus all inter-`affected_entity_ids` edges into `staging_conn`.
  - `yaw/daemon/graph/inference/diagnostics.py`, `yaw/daemon/graph/inference/optimizer.py`, & `yaw/daemon/indexing.py`: Replaced `json.dumps(..., sort_keys=True) + hashlib.sha256` with `fingerprint_object(payload).hex()` across `judge_stage_fingerprint`, `embed_fingerprint`, `shacl_fingerprint`, `openie_fingerprint`, `extraction_fingerprint`, `candidate_fingerprint`, and `query_retrieval_fingerprint` (removing `hashlib` from `diagnostics.py`); added `graph_plan_fingerprints(program, embedder)` in `yaw/daemon/indexing.py` and wired it across `bind_native_indexing`, `YawRuntime._build_coco_app`, and `YawRuntime._submit_program_reload`; renamed `{stem}_graph_rlm_audit.json` -> `{stem}_audit.json`, deleted the `_proposer_telemetry.json` write, and slimmed `_publish_optimization_audit` to `{"graph_rlm_stats", "candidate_records", "eval_records", "gepa_progress"}`.
  - `yaw/daemon/graph/engine.py`, `yaw/daemon/runtime.py`, & `yaw/daemon/cli.py`: Added `logger.info("%s", message)` in `YawGraphEngine.emit_indexing_progress`; extracted `YawRuntime._create_bound_engine` and `YawRuntime._close_coco_app` / `_release_idle_coco_env`; pointed `snapshot_dataset_bundle` in `YawRuntime.optimize` at `<db_parent>/optimize_coco/<dataset>`; recorded/preserved `"started_at"` in `validate_optimization_run`; removed unused `progress_callback` parameters on `request_sync`, `sync_now`, `retrieve_async`, `retrieve`, `rag_qa_async`, `rag_qa`, and `rebuild`; replaced `--resume` with `--fresh` and added `_clean_optimization_artifacts` and `_read_db_summary` in `yaw/daemon/cli.py`.
  - `tests/core/test_config.py`, `tests/daemon/graph/test_batch_equivalence.py`, `tests/daemon/graph/test_engine.py`, `tests/daemon/graph/test_optimization_progress.py`, `tests/daemon/graph/inference/test_prompts.py`, `tests/daemon/test_cli_edges.py`, & `tests/daemon/test_runtime.py`: Added `test_passive_endpoint_homograph_preserves_edges_and_facts_across_batches` (`CG-OPT-2`) and `test_canonical_optimize_resume_and_fresh_lifecycle` (`CG-OPT-1`) and updated CLI/config/engine/optimizer tests to the canonical `optimize_*` contract.
- **Gate results & evidence paths**:
  - `uv run --locked ty check`, `uv run --locked ruff check --fix`, `uv run --locked ruff format --check`: all passed (`0` errors, `172` files formatted).
  - `uv run --locked pytest tests/daemon/graph/test_batch_equivalence.py -k test_passive_endpoint_homograph_preserves_edges_and_facts_across_batches --no-testmon -q`: `1 passed in 3.34s` (`CG-OPT-2`).
  - `uv run --locked pytest tests/daemon/test_runtime.py -k test_canonical_optimize_resume_and_fresh_lifecycle --no-testmon -q`: `1 passed in 10.81s` (`CG-OPT-1`).
  - `uv run --locked pytest tests/core/ tests/daemon/ --no-testmon -q`: `817 passed, 3 subtests passed` (`816` on first sweep + `1` live RLM retry pass).
  - Removal inventory exit checks across `yaw/` and `tests/core/ tests/daemon/`: `0` matches for all retired patterns (`coco_state_optimize`, `compiled_program_optimize`, `status_optimize`, `for_optimization(datestamp`, `_cli_progress`, `proposer_telemetry`, `_graph_rlm_audit.json`, `".yaw/coco_state"`, `state_dir) / "optimize_coco"`, and `hashlib.sha256` in `diagnostics.py`).
- **Discoveries during implementation**:
  - Unifying `YawConfig.state_dir` and `resolve_coco_state_root` to `.yaw/coco_lmdb` means `YawRuntime._build_coco_app()` (`yaw_indexing`) and `bind_native_indexing` (`yaw_frozen_indexing`) target the same LMDB directory; because CocoIndex's Rust `core.Environment` enforces LMDB's single-open-handle-per-process rule, `YawRuntime._release_idle_coco_env` closes idle (non-live) in-process `_coco_env` handles before opening `coco.Environment` or running `evolve_prompts`, and `_execute_sync_worker` / `_start_live_update` lazily rebuild `_coco_app` on demand.
- **What it does not establish**:
  - `scripts/watchdog.py` and `tests/scripts/test_watchdog*.py` remain on Checkpoint C3; live optimization migration/resume and Tier 2 parity gate follow in Checkpoints C4–C5.

### Completed Unit: Checkpoint C3 (2026-11-08)

- **Changed symbols & modules** (governing contracts: [epic-1-hipporag-daemon.md §2 & §Decisions](../epic-1-hipporag-daemon.md#decisions) and [native-cocoindex-indexing.md §3, §5, & §Decisions](../native-cocoindex-indexing.md#decisions)):
  - `scripts/watchdog.py`: Updated `get_active_sandbox_info` to discover canonical `.yaw/optimize_yaw.db` and `.yaw/optimize_yaw_*.db` (deduplicated by `p.resolve()`), derive `suffix` and `run_tag = suffix or "canonical"`, read `optimize_status_{suffix}.json` / `optimize_status.json`, read durable `"started_at"` from `meta['optimization_run']` on `ro_conn`, and write timeseries to `optimize_telemetry_{suffix}.json` / `optimize_telemetry.json`; updated `_audit_path`, `_pools_path`, and `generate_graph_projection_svg` (`optimize_kg_frames` / `optimize_kg_frames_{tag}`) for canonical `optimize_*` paths; replaced `_atomic_write` with `write_via_temp_atomic`; wired `configure_logging` + `get_logger("yaw.watchdog")` in `main()` and replaced raw `print` calls; removed `LEGACY_FIG_FILES`, `read_config_info` (and unused `cfg` parameter on `generate_kg_dashboard_svg`), the `legacy` untagged-eval fallback (`load_candidate_records` now returns 2-tuple `(rows, n_evals)`), `diff_peer_map` + commented-out homograph label block, the 52-line per-chunk JSON double-parsing `graph_cache` loop, and unread `get_lemonade_server_metrics` / `get_pipeline_resource_metrics` keys (including the 250ms CPU sleep).
  - `yaw/core/logger.py` & `yaw/core/__init__.py`: Added and exported `get_logger(name="yaw")` and accepted `log_file_path` keyword / level-first call pattern in `configure_logging`.
  - `tests/scripts/test_watchdog.py` & `tests/scripts/test_watchdog_edges.py`: Updated all tests to exercise canonical `optimize_yaw.db` (`datestamp == "canonical"`, durable `meta['optimization_run']` `"started_at"`), `optimize_status.json`, `optimize_program_audit.json`, `optimize_program_pools.json`, `optimize_telemetry.json`, `optimize_kg_frames/`, the 2-tuple `load_candidate_records(sandbox)` signature, the 2-arg `generate_kg_dashboard_svg(dest, sandbox)` signature, and the 4-arg/5-arg candidate/report renderer signatures; removed obsolete tests for deleted `read_config_info` and `legacy` untagged fallback.
- **Gate results & evidence paths**:
  - `uv run --locked ty check`, `uv run --locked ruff check --fix`, `uv run --locked ruff format --check`: all passed (`0` errors, `172` files formatted).
  - `uv run --locked pytest tests/scripts/test_watchdog.py tests/scripts/test_watchdog_edges.py tests/core/test_logger.py tests/core/test_architecture.py --no-testmon -q`: `38 passed in 13.55s` (`19 passed` in `tests/scripts/`).
  - All 8 Removal Inventory exit checks (§7) across `yaw/`, `scripts/`, `tests/`, and `docs/epic-1-hipporag-daemon.md`: `0` matches.
  - Net production LOC delta (`git diff --stat yaw/ scripts/watchdog.py`): `12 files changed, 543 insertions(+), 820 deletions(-)` (`-277` lines net, satisfying `>= 250` lines removed).
  - `git diff --check`: passed (`0` whitespace/conflict errors).
- **Discoveries during implementation**:
  - `get_pipeline_resource_metrics()` in `scripts/watchdog.py` previously executed a 250ms `time.sleep(0.25)` on every `get_active_sandbox_info()` call solely to compute per-process CPU percentages that had `0` readers in the dashboard or report; removing the unread CPU/token metrics eliminated both the 250ms blocking delay and 47 lines of dead dictionary construction.
- **What it does not establish**:
  - Critic diff audit, atomic commit, live `.yaw/yaw_optimize_20261108_075400.db` -> `.yaw/optimize_yaw.db` migration/resume, and Clean-Slate Epic 1 Completion Gate (`D1-A6`) follow in Checkpoints C4–C5.

### Completed Unit: Checkpoint C4 (2026-11-08)

- **Changed symbols & modules** (governing contracts: [epic-1-hipporag-daemon.md §2 & §Decisions](../epic-1-hipporag-daemon.md#decisions), [native-cocoindex-indexing.md §3, §5, & §Decisions](../native-cocoindex-indexing.md#decisions), and [native-cocoindex-graph.md §3](../native-cocoindex-graph.md#3-goals)):
  - Audited full `C1–C3` diff across `AGENTS.md`, `README.md`, `docs/`, `yaw/`, `scripts/watchdog.py`, and `tests/` via `Checkpoint C4 Implementation Critic Auditor` (`00b62ea7-1b8e-47a2-a698-574c4894d1dc`): `ZERO FINDINGS`.
- **Gate results & evidence paths**:
  - All 8 Removal Inventory exit checks (`git grep`): `0` matches across `yaw/`, `scripts/`, `tests/`, and `docs/epic-1-hipporag-daemon.md`.
  - Net production LOC delta (`git diff --stat yaw/ scripts/watchdog.py`): `12 files changed, 544 insertions(+), 821 deletions(-)` (`-277` net lines, exceeding `>= 250` target).
  - Mandatory quality gates: `uv run --locked ty check` (`All checks passed!`), `uv run --locked ruff check --fix` (`All checks passed!`), `uv run --locked ruff format --check` (`172 files already formatted`), `uv run --locked pytest` (`836 passed, 3 subtests passed` across `tests/core/`, `tests/daemon/`, and `tests/scripts/`), and `git diff --check` (`0` errors).
- **What it does not establish**:
  - Clean-Slate Epic 1 Completion Gate (`D1-A6` via `verify_paper_parity_gate.py --compiled-program .yaw/compiled_program.json --mode eval --fresh` and `Goal A` / `Goal B` live batch invariance) follows in Checkpoint C5 once the resumed `uv run yaw dspy optimize` run completes `.yaw/optimize_program.json`.

## 2026-11-08: High-Throughput SQLite PRAGMAs, Vectorized BLAS Synonym & Neighborhood Mining, Lexical Pruning, Zero-Copy Vectors, Fused Fact Join & Epic 3 Performance Spec

Controlling contracts: [epic-3-performance.md §1.2](../epic-3-performance.md#12-baseline-already-in-place-from-epic-1-do-not-re-invent) and [native-cocoindex-graph.md §3](../native-cocoindex-graph.md#3-goals).

- **Changed symbols & modules**:
  - `yaw/daemon/graph/storage/db.py`: Added `PRAGMA cache_size = -65536;` (`64 MiB`) and `PRAGMA mmap_size = 268435456;` (`256 MiB`) to `_configure_connection_pragmas` and unified `open_readonly_yaw_db` onto `_configure_connection_pragmas(conn, busy_timeout_ms=busy_ms)` so `open_yaw_db`, `open_managed_yaw_db`, and `open_readonly_yaw_db` share a single pragma configuration.
  - `yaw/daemon/graph/storage/graph_ops.py`: Replaced per-entity `vec0` `MATCH` SQLite loops in both `find_entity_synonym_candidates` and Step `1d` of `load_affected_neighborhood` with contiguous `float32` NumPy/BLAS matrix passes preserving exact IEEE-754 `float32 -> float64` cosine distance semantics, deterministic `(distance ASC, entity_id ASC)` tie-breaking (`np.argsort(..., kind="stable")`), `adjudicated` pair exclusion, and both forward and reverse top-$k$ candidate discovery; added the exact length-ratio upper bound $2 \min(|a|, |b|) / (|a| + |b|) < \tau$ and pre-tokenized `_lexical_name_parts` / `last_seq2_norm` caching to `_lexical_similarity_cascade` and `find_entity_disambiguation_candidates`.
  - `yaw/daemon/graph/storage/vectors.py`: Removed redundant `.copy()` in `deserialize_f32_vector(blob)` so deserialized float32 vectors are zero-copy `np.frombuffer` views.
  - `yaw/daemon/graph/engine.py`: Fused the separate `SELECT ... FROM facts` and `SELECT ... FROM fact_sources` queries in `YawGraphEngine.retrieve_with_seeds` into a single `LEFT JOIN` query.
  - `tests/daemon/graph/storage/test_db.py` & `tests/daemon/graph/test_engine.py`: Added `test_connection_pragmas_configured_across_open_modes`, `test_find_entity_synonym_candidates_blas_vectorized_without_match_loops`, `test_quick_fixes_blas_neighborhood_lexical_pruning_and_zero_copy_vectors`, and `LEFT JOIN` SQL trace assertions in `test_engine_fact_score_weighting_and_thread_safe_retrieval`.
  - `docs/epic-3-performance.md`: Recorded all implemented quick-fix optimizations in §1.2.
- **Gate results & evidence paths**:
  - Mandatory quality gates (`uv run --locked ty check`, `uv run --locked ruff check --fix`, `uv run --locked ruff format --check`, `uv run --locked pytest tests/daemon/graph/storage/test_db.py tests/daemon/graph/test_batch_equivalence.py tests/daemon/graph/test_engine.py tests/core/test_architecture.py -k "not test_g5_g6_paired_64_plus_64_vs_128_acceptance_report" --no-testmon -q`, `git diff --check`): all passed.
- **What it does not establish**:
  - Does not hot-reload the already-running background `uv run yaw dspy optimize` process in memory, nor complete Checkpoint C5 (`D1-A6` slice-based paper parity gate) until `.yaw/optimize_program.json` finishes.

## 2026-11-09: Suffix Lexical Cascade Neighborhood Completeness, `existing_bindings` Seed Unpacking & Shared-Definition Passive Endpoint Promotion

Controlling contracts: [native-cocoindex-graph.md §3 & §5.2](../native-cocoindex-graph.md#3-goals) and [w3c-improvements.md §12](../w3c-improvements.md#12-strict-judgelabels-identity-equivalence--canonical-gemma-4-3-stage-cascade-without-python-token-overrides-or-auto-merge-gates).

- **Root cause of Batch 20 `StatementShape/closedWorld` missing-endpoint failure (`"SSE disconnect"` vs `"streaming disconnect"`)**:
  - In `yaw/daemon/graph/storage/graph_ops.py` Step `1c` of `load_affected_neighborhood`, lexical candidate seeding previously scanned `entity_labels` using only 3-character prefixes (`WHERE label_norm >= pfx AND label_norm < upper`). When an earlier batch had `"streaming disconnect"` (`label_norm = "streaming disconnect"`, starting with `"str"`) and a later batch introduced `"SSE disconnect"` (`prefixes = {"sse", "dis"}`), `"streaming disconnect"` had `_lexical_similarity_cascade("SSE disconnect", "streaming disconnect", 0.70) == 0.7647 >= 0.70` yet did not start with `"sse"` or `"dis"`. When `"streaming disconnect"` was pulled passively into `affected_entity_ids - core_entity_ids` via a shared entity (`"sse-gateway"`) while having another fact from an unrelated earlier-batch entity (`"Nimbus client SDK"`), `find_entity_disambiguation_candidates` in `stage_conn` paired and merged `"streaming disconnect"` into `"SSE disconnect"`, deleting `"streaming disconnect"` from `entities` while leaving `"Nimbus client SDK -> streaming disconnect"` in `ro_conn` with a dangling endpoint (`StatementShape/closedWorld` and `EdgeShape/closedWorld`).
  - In `yaw/daemon/graph/planner.py:1055`, `seed_entity_ids` unpacked `*existing_bindings.values()` (3-tuples `(eid, name, definition)`) instead of `*(eid for eid, _, _ in existing_bindings.values())`.
  - In `yaw/daemon/graph/storage/graph_ops.py` fixed-point loop, passive entities in `affected_entity_ids` sharing a casefolded `definition` in `entities` were not promoted to `core_entity_ids`.
- **Changed symbols & modules**:
  - `yaw/daemon/graph/planner.py`: Unpacked `*(eid for eid, _, _ in existing_bindings.values())` when building `seed_entity_ids` in `plan_graph_publication`.
  - `yaw/daemon/graph/storage/graph_ops.py`: Replaced the 3-character prefix scan in Step `1c` of `load_affected_neighborhood` with an exact, length-pruned (`2 * min(|a|, |b|) / (|a| + |b|) >= 0.70`) lexical cascade scan over `ro_conn` (`entities` `UNION` `entity_labels` with `WHERE` clauses for bounded-query compliance); updated `_lexical_similarity_cascade` to call `matcher.set_seq2(norm_a)` when `matcher.b != norm_a` on caller-supplied `SequenceMatcher` instances; promoted passive entities in `affected_entity_ids` sharing a non-empty casefolded `definition` into `core_entity_ids` inside the fixed-point expansion loop.
  - `tests/daemon/graph/test_batch_equivalence.py` & `tests/daemon/graph/synonym_protocol.py`: Added `test_suffix_lexical_match_on_passive_endpoint_preserves_closed_world_across_batches` (verified failing in RED with `StatementShape/closedWorld` and `EdgeShape/closedWorld` on `entity-b82f4d657df42a68907a4780664a351c`, and passing in GREEN with 15-table canonical equivalence between `incremental` and `reference` arms).
- **Gate results & evidence paths**:
  - Mandatory quality gates (`uv run --locked ty check`, `uv run --locked ruff check --fix`, `uv run --locked ruff format --check`, `uv run --locked pytest tests/daemon/graph/test_batch_equivalence.py tests/daemon/graph/storage/test_db.py tests/core/test_architecture.py -k "not test_g5_g6_paired_64_plus_64_vs_128_acceptance_report" --no-testmon -q`, `git diff --check`): all passed.
- **What it does not establish**:
  - Does not hot-reload any running background optimization process in memory until restarted, nor complete Checkpoint C5 (`D1-A6` slice-based paper parity gate) until `.yaw/optimize_program.json` finishes.

## 2026-11-09: Unified 3-Layer Typed DSPy Signatures, Discriminated Union Schemas, Schema Hint Deduplication, Dead Demo Removal & MIPROv2 Demo Defaults

Controlling contracts: [w3c-improvements.md §12](../w3c-improvements.md#12-strict-judgelabels-identity-equivalence--canonical-gemma-4-3-stage-cascade-without-python-token-overrides-or-auto-merge-gates), [native-cocoindex-graph.md §4 & §5.3](../native-cocoindex-graph.md#5-design-current-state), and [epic-1-hipporag-daemon.md §2](../epic-1-hipporag-daemon.md#2-sprint-plan).

- **Root cause of `[skos:prefLabel, skos:definition]` & `# note: the value you produce must adhere to JSON schema: {}` extraction corruption**:
  - Previously, DSPy signatures in `yaw/daemon/graph/inference/prompts.py` typed complex outputs as `Any` (`named_entities: Any`, `triples: Any`, `entity_definitions: Any`, `decisions: Any`, `repairs: Any`, `fact: Any`, `selection: Any`), causing `dspy.JSONAdapter.format_field_structure(sig)` to emit `# note: the value you produce must adhere to JSON schema: {}` while signature docstrings and `STAGE_SCHEMA_CONTRACTS` duplicated inline JSON wrapper syntax and bracketed placeholder tokens (`[skos:prefLabel, skos:definition]`, `[entity_name, contextual_definition]`). Under `temperature = 1.0`, small local models echoed those placeholder tokens directly as entity labels.
- **Changed symbols & modules**:
  - `yaw/daemon/graph/inference/prompts.py` & `yaw/daemon/graph/inference/__init__.py`: Defined strict `TypedDict` schemas decorated with `with_config(ConfigDict(extra="forbid"))` (`EntitySensePair`, `FactTriple`, `ObjectPropertyPredicate`, `DatatypePropertyPredicate`, `LiteralObject`, `ObjectPropertyStatement`, `DatatypePropertyStatement`, `ReifiedStatement = ObjectPropertyStatement | DatatypePropertyStatement`, `EquivalentLabelDecision`, `DistinctLabelDecision`, `LabelDecision = EquivalentLabelDecision | DistinctLabelDecision`, `HierarchyDecision`, `RelatedDecision`, `SHACLRepairAction`, `SHACLRepair`, `DistillerReference`, `DistillerSelection`); replaced all `Any` field annotations across all DSPy signatures; removed competing inline JSON syntax templates and `[skos:prefLabel, skos:definition]` placeholders from signature docstrings and `OutputField(desc=...)`; compacted `ProposeInstructionSignature`; deleted ~460 lines of unused static demo scaffolding (`NER_ONE_SHOT_PARAGRAPH`, `RAG_QA_DOCS`, `best_dspy_prompt`, `_NER_DEMO_ENTITY_DEFINITIONS`, `_TRIPLE_DEMO_TRIPLES`, `_build_default_*_demos`, `NER_DEMOS`, `TRIPLE_DEMOS`, `TRIPLE_FILTER_DEMOS`, `QA_DEMOS`) and dead `FactList` / `PredicatePropertyKind` / `PredicateSlot` symbols.
  - `yaw/daemon/graph/inference/grammars.py`, `yaw/daemon/graph/inference/adapter.py`, & `yaw/daemon/graph/inference/invariants.py`: Updated `_OPEN_TRIPLE_GBNF_RULES` (`DSPY_TRIPLES_GBNF_GRAMMAR`) to accept `resource-slot ::= string | "[" string ", " string "]"` for homograph sense pairs; fixed `CompressInstructionSignature` vs `ProposeInstructionSignature` and annotation-first `Judge*` dispatch in `YawGBNFAdapter.resolve_grammar`; added `"qa"`, `"query_intent"`, and `"shacl_healer_rlm"` to `STAGE_SCHEMA_CONTRACTS` (removing dead `"fact_filter"` and raw JSON templates), stripped `.predict` in `validate_stage_invariants`, and checked `shortened_instruction` when `stage == "llm_compressor"`.
  - `yaw/daemon/graph/inference/program.py`, `yaw/daemon/graph/inference/proposer.py`, `yaw/daemon/graph/inference/rlm.py`, `yaw/daemon/graph/inference/components.py`, & `yaw/daemon/graph/planner.py`: Removed `include_demos` and `build_rlm_action_demos`; removed dead `canonical` merge fallback branches in `planner.py`; bumped `@coco.fn(memo=True, version=...)` on `extract_chunk_semantics`, `judge_semantics`, and `repair_missing_definitions` for the updated schema fingerprints.
  - `yaw/core/config.py`, `yaw.toml`, `yaw/daemon/cli.py`, & `yaw/daemon/graph/inference/optimizer.py`: Defaulted `optimize_max_bootstrapped_demos = 1` and `optimize_max_labeled_demos = 1` while preserving explicit `0` when passed.
  - `tests/`: Added comprehensive schema, discriminated-union, `extra="forbid"`, GBNF, and invariant tests in `test_native_bindings.py` and `test_grammars.py`, and aligned mock responses across `synonym_protocol.py`, `test_engine.py`, `test_engine_build_shacl.py`, `test_shacl_build_time_gate.py`, and `test_runtime.py`.
- **Gate results & evidence paths**:
  - `git grep -n -e best_dspy_prompt -e NER_ONE_SHOT_PARAGRAPH -e RAG_QA_DOCS -e NER_DEMOS -e TRIPLE_DEMOS -e TRIPLE_FILTER_DEMOS -e QA_DEMOS -e build_rlm_action_demos -e include_demos -e FactList -e PredicatePropertyKind -e PredicateSlot -- yaw/ scripts/ tests/`: `0` matches.
  - Mandatory quality gates (`uv run --locked ty check`, `uv run --locked ruff check --fix`, `uv run --locked ruff format --check`, `uv run --locked pytest`, `git diff --check`): all passed.
- **What it does not establish**:
  - Full-corpus GEPA optimization (`uv run yaw dspy optimize`) and Checkpoint C5 (`D1-A6` slice-based paper parity gate) remain pending next user-initiated run.

## 2026-11-09: CocoIndex Anti-Patterns 1–6 Remediation (from `28a1113`)

Controlling contracts: [native indexing §2, §5, §7, §8, & §Decisions](../native-cocoindex-indexing.md#8-checkpoints) and [native graph publication §2, §5, §7, §8, & §Decisions](../native-cocoindex-graph.md#8-checkpoints).

This specification remediates the `6` CocoIndex `>= 1.1.0` (`uv.lock:330–331` at `28a1113`) architectural anti-patterns identified at pinned commit `28a1113` across `yaw/daemon/graph/storage/{managed_sqlite,rows}.py`, `yaw/daemon/graph/planner.py`, `yaw/daemon/ingestion/{source_targets,bundles}.py`, `yaw/daemon/indexing.py`, and `yaw/daemon/runtime.py`:

1. **Anti-Pattern 1 (Shadow Change Detection)**: `partition_frozen_chunks` (`yaw/daemon/indexing.py:424–478` at `28a1113`) executes raw `3`-table SQLite joins (`source_chunks s JOIN chunk_inference ci JOIN chunks c`) and recomputes `fingerprint_object((settings, pointer, text, entry[1])).hex()` (`planner.py:854` at `28a1113`) in Python before CocoIndex mounts components. Remediated by deleting `partition_frozen_chunks` and computing `fingerprint_object((settings, out.pointer, out.text)).hex()` in `plan_graph_publication` (`meta['graph_input_fingerprints']`), because `extract_text(text, settings.program_identity)` and `embed_text(text, settings.embedding_identity)` are deterministic functions of `(text, settings)`.
2. **Anti-Pattern 2 (Parallel In-Memory Frozen Pipeline & Un-Memoized Orchestration)**: `yaw/daemon/indexing.py` (`L376–536, 557–631` at `28a1113` and `cd2d20a`) maintains a separate hand-rolled in-memory ingestion pipeline (`frozen_indexing_app_main`, `process_frozen_chunk`, `frozen_chunk_metadata`, `FrozenChunkStream`, `"yaw_frozen_indexing"`) duplicating `indexing_app_main` (`"yaw_indexing"`), and mounts `declare_source_file` and `process_chunk` without `@coco.fn(memo=True)`, forcing $O(N^2)$ Python coroutine re-executions across batches. Remediated by retiring the parallel frozen pipeline completely (`frozen_indexing_app_main`, `process_frozen_chunk`, `frozen_chunk_metadata`, `partition_frozen_chunks`, `FrozenChunkStream`, `"yaw_frozen_indexing"`), unifying production ($\text{Target}_{\text{production}} = F(\text{Source}_{\text{production}})$), GEPA optimization ($\text{Target}_{\text{optimize}} = F(\text{Source}_{\text{optimize}})$), and benchmarking ($\text{Target}_{\text{benchmark}} = F(\text{Source}_{\text{benchmark}})$) onto the single `indexing_app_main` (`"yaw_indexing"`) pipeline (`CorpusIndexingLiveComponent`), and decorating mounted `declare_source_file` and `process_chunk` with `@coco.fn(memo=True, version=1)` (alongside `@coco.fn(memo=True, version=2)` on `publish_graph`).
3. **Anti-Pattern 3 (Ephemeral `coco.Environment` Thrashing & `_release_idle_coco_env`)**: `bind_native_indexing` (`indexing.py:597–628` at `28a1113`) opens and closes a new `coco.Environment` and `coco.App` on every `engine.index_chunks()` call and tears down `YawRuntime._coco_env` via `_ActiveCocoAppRegistry`, `register_active_coco_env_closer`, `release_idle_coco_env` (`indexing.py:55–71`), and `YawRuntime._release_idle_coco_env` (`runtime.py:285–297, 345, 1130`). Remediated by replacing the closer registry with a process-level `coco.Environment` registry per `resolved_state_root: Path` (`acquire_coco_environment` / `close_coco_environment`) shared across `"yaw_indexing"` (`indexing_app_main`).
4. **Anti-Pattern 4 (Throwaway `source_snapshot.db` & Double File Read / Re-Chunking)**: `chunk_file_memo` (`source_targets.py:78–115` at `28a1113`) discards chunk `text`, causing `process_file` (`indexing.py:181–201`) to re-read and re-chunk every file a second time and `snapshot_dataset_bundle` (`bundles.py:236–395`) to materialize a throwaway `source_snapshot.db`, query `SELECT * FROM source_chunks`, and re-read/re-chunk every file on disk (`bundles.py:334–393`). Remediated by bumping `chunk_file_memo` to `@coco.fn(memo=True, version=2)` returning `tuple[tuple[SourceMetadata, str], ...]`, returning `(metadata, text)` from `declare_source_file` directly into `process_file`, and returning `dict[str, tuple[tuple[SourceMetadata, str], ...]]` directly from `app.update().result()` in `snapshot_dataset_bundle` with `0` throwaway SQLite databases.
5. **Anti-Pattern 5 (Monolithic Live Re-Scan in `_CorpusFeedSubscriber` & `CorpusIndexingLiveComponent`)**: `_CorpusFeedSubscriber.update(key, value)` and `.delete(key)` (`indexing.py:249–255` at `28a1113`) discard `key` and `value`, triggering `self._files.items()` (`DirWalker._walk_sync()`, an $O(N)$ synchronous directory walk) and re-running un-memoized `process_file` across the entire corpus on every single file event. Remediated by maintaining `self._active_files: dict[str, FileLike]` with $O(1)$ per-key updates/deletions on live events and `@coco.fn(memo=True, version=1)` short-circuiting on `FileLike.__coco_memo_state__`.
6. **Anti-Pattern 6 (`ManagedTableTarget._provider` Encapsulation Bypass)**: `declare_tables` (`rows.py:333` at `28a1113`) executes `coco.declare_target_state(target._provider.target_state(key, DELETE_ROW))`, reaching into `sqlite.TableTarget`'s private `_provider` attribute across module boundaries. Remediated by exposing `ManagedTableTarget[RowT].delete_row(self, key: tuple[Any, ...]) -> None` in `managed_sqlite.py`, calling `target.delete_row(key)` in `rows.py`, and enforcing the AST cross-object private attribute gate in `tests/core/test_architecture.py`.

### Checkpoints (N1–N5)

- [x] **Checkpoint N1 (`ManagedTableTarget.delete_row` & AST Private-Attribute Gate — Anti-Pattern 6)**:
  - **Subagent Pipeline**: `simplifier (Model="flash")` -> `critic (Model="flash")`.
  - **Bounded Input Files** (`3` files): `yaw/daemon/graph/storage/managed_sqlite.py`, `yaw/daemon/graph/storage/rows.py`, `tests/core/test_architecture.py`.
  - **Named Artifacts**: `outputs/benchmarks/n1_managed_table_target_gate.xml`; `ManagedTableTarget[RowT].delete_row(self, key: tuple[Any, ...]) -> None` in `managed_sqlite.py`; `target.delete_row(key)` in `rows.py`; AST private-attribute gate in `tests/core/test_architecture.py` scanning `yaw/daemon/graph/storage/*.py` (excluding `db.py`'s `managed._conn`), `yaw/daemon/indexing.py`, `yaw/daemon/ingestion/*.py`, and `yaw/daemon/graph/{planner,normalization,ppr}.py` for `ast.Attribute` where `node.attr.startswith("_") and not node.attr.startswith("__")` and `node.value` is not `ast.Name(id in {"self", "cls"})` or `ast.Call(func=ast.Name(id="super"))`, plus forbidding `node.attr == "_provider"` across all `yaw/**/*.py` outside `managed_sqlite.py`.
  - **Exit Gate**: `git grep -n "_provider" -- yaw/daemon/graph/storage/rows.py` returns `0` matches; `uv run --locked ty check`, `uv run --locked ruff check`, `uv run --locked pytest tests/core/test_architecture.py tests/daemon/graph/storage/test_publication.py --no-testmon -q` pass.
- [x] **Checkpoint N2 (`chunk_file_memo` Text Memoization & Throwaway DB Elimination in `snapshot_dataset_bundle` and `process_file` — Anti-Pattern 4)**:
  - **Subagent Pipeline**: `simplifier (Model="flash")` -> `critic (Model="flash")`.
  - **Bounded Input Files** (`4` files): `yaw/daemon/ingestion/source_targets.py`, `yaw/daemon/ingestion/bundles.py`, `yaw/daemon/indexing.py`, `tests/daemon/ingestion/test_datasets.py` (plus `declare_source_file` callers in `tests/daemon/ingestion/test_source_targets.py` and `tests/daemon/ingestion/test_corpus.py`).
  - **Named Artifacts**: `outputs/benchmarks/n2_bundle_memo_text.xml`; `chunk_file_memo(file: FileLike, settings: SourceSettings) -> tuple[tuple[SourceMetadata, str], ...]` (`@coco.fn(memo=True, version=2)`) and `declare_source_file` in `source_targets.py`; single-pass `process_file` (`frozen_text=text`, `if settings is not None and pointer.start_line > 1:`) in `indexing.py`; `_bundle_snapshot_app_main` returning `dict[str, tuple[tuple[SourceMetadata, str], ...]]` via `app.update().result()` in `bundles.py` without `source_snapshot.db`, `_process_bundle_file`, or duplicate `read_text()`/`chunk_source_text()` calls (note: `acquire_coco_environment` arrives in `N3`).
  - **Exit Gate**: `git grep -n -e "source_snapshot.db" -e "_process_bundle_file" -- yaw/ scripts/ tests/` returns `0` matches; `uv run --locked ty check`, `uv run --locked ruff check`, `uv run --locked pytest tests/daemon/ingestion/ --no-testmon -q` pass.
- [x] **Checkpoint N3 (Shared `coco.Environment` & Named App Registry; Delete `_release_idle_coco_env` — Anti-Pattern 3)**:
  - **Subagent Pipeline**: `simplifier (Model="flash")` -> `critic (Model="flash")`.
  - **Bounded Input Files** (`4` files): `yaw/daemon/indexing.py`, `yaw/daemon/runtime.py`, `yaw/daemon/ingestion/bundles.py`, `tests/daemon/graph/test_indexing_overlap.py`.
  - **Named Artifacts**: `outputs/benchmarks/n3_coco_memo_env_cutover.xml`; `_SharedCocoRuntime(environment, context, apps)`, `acquire_coco_environment(state_root)`, and `close_coco_environment(state_root)` in `indexing.py` caching `"yaw_frozen_indexing"` and `"yaw_bundle_source"` on the shared environment record; removal of `_ActiveCocoAppRegistry`, `register_active_coco_env_closer`, `release_idle_coco_env`, and `YawRuntime._release_idle_coco_env`.
  - **Exit Gate**: `git grep -n -e "_ActiveCocoAppRegistry" -e "register_active_coco_env_closer" -e "release_idle_coco_env" -e "_release_idle_coco_env" -- yaw/ scripts/ tests/` returns `0` matches; `uv run --locked ty check`, `uv run --locked ruff check`, `uv run --locked pytest tests/daemon/graph/test_indexing_overlap.py tests/daemon/test_runtime.py --no-testmon -q` pass.
- [x] **Checkpoint N4 (Upgrade $F = \text{indexing\_app\_main}$: Incremental `_active_files`, `@coco.fn(memo=True, version=1)` on `declare_source_file` & `process_chunk`, `64`-Chunk Micro-Batching Inside `CorpusIndexingLiveComponent`, and Input-Only `input_fingerprints` in `planner.py` — Anti-Patterns 1, 2, 5)**:
  - **Subagent Pipeline**: `simplifier (Model="flash")` -> `critic (Model="flash")`.
  - **Bounded Input Files** (`4` files): `yaw/daemon/indexing.py`, `yaw/daemon/ingestion/source_targets.py`, `yaw/daemon/graph/planner.py`, `tests/daemon/graph/test_indexing_overlap.py`.
  - **Named Artifacts**: `outputs/benchmarks/n4_unified_indexing_component.xml`; `@coco.fn(memo=True, version=1)` on mounted `declare_source_file` and `process_chunk`; `ACTIVE_SOURCE_SETTINGS`, `ACTIVE_INFERENCE_IDENTITY`, and `ACTIVE_GRAPH_SETTINGS` context keys; incremental `CorpusIndexingLiveComponent._active_files: dict[str, FileLike] | None` and `self.remaining_chunks: int = 0` with `64`-chunk micro-batch admission (`(*retained, *changed[:batch_size])`) against `SELECT chunk_id FROM chunks` and `meta['graph_input_fingerprints']` inside `CorpusIndexingLiveComponent.process()`; input-only `fingerprint_object((settings, out.pointer, out.text)).hex()` in `plan_graph_publication` (`planner.py:854`).
  - **Exit Gate**: `uv run --locked ty check`, `uv run --locked ruff check`, `uv run --locked pytest tests/daemon/graph/test_indexing_overlap.py --no-testmon -q` pass.
- [x] **Checkpoint N5 (Retire Parallel Frozen Pipeline & Cut Callers Over to $\text{Target} = F(\text{Source})$ — Anti-Patterns 1 & 2 Completion)**:
  - **Subagent Pipeline**: `simplifier (Model="flash")` -> `critic (Model="flash")`.
  - **Bounded Input Files** (`5` files): `yaw/daemon/indexing.py`, `yaw/daemon/runtime.py`, `tests/daemon/graph/test_indexing_overlap.py`, `tests/daemon/graph/test_evidence_revisions.py`, `tests/daemon/graph/test_batch_equivalence.py`.
  - **Named Artifacts**: `outputs/benchmarks/n5_unified_pipeline_cutover.xml`; complete deletion of `frozen_indexing_app_main`, `process_frozen_chunk`, `frozen_chunk_metadata`, `partition_frozen_chunks`, `FrozenChunkStream`, `"yaw_frozen_indexing"`, and `process_file`; `bind_native_indexing` and `YawRuntime.optimize` wired onto `indexing_app_main` (`"yaw_indexing"`) looping `await app.update(live=False).result()` while `component.remaining_chunks > 0`; updated test fixtures in `test_evidence_revisions.py`, `test_batch_equivalence.py`, and `test_indexing_overlap.py` running through `indexing_app_main` over files in `tmp_path` with canonical `compute_chunk_id`; completed review record entry in `docs/reviews/incremental-publication-work.md`.
  - **Exit Gate**: `git grep -n -e "frozen_indexing_app_main" -e "process_frozen_chunk" -e "frozen_chunk_metadata" -e "partition_frozen_chunks" -e "FrozenChunkStream" -e "yaw_frozen_indexing" -- yaw/ scripts/ tests/` returns `0` matches; `uv run --locked ty check`, `uv run --locked ruff check`, `uv run --locked ruff format --check`, `uv run --locked pytest`, and `git diff --check` pass (`0` errors).

### Completed Unit: Checkpoint N1 (2026-11-09)

- **Changed symbols & modules** (governing contracts: [native-cocoindex-graph.md §2, §5, §7, §8](../native-cocoindex-graph.md#8-checkpoints) and [native-cocoindex-indexing.md §2, §5, §7, §8](../native-cocoindex-indexing.md#8-checkpoints)):
  - `yaw/daemon/graph/storage/managed_sqlite.py`: Defined `ManagedTableTarget[RowT](sqlite.TableTarget[RowT])` inheriting `TableTarget.__init__` and defining `delete_row(self, key: tuple[Any, ...]) -> None` declaring `DELETE_ROW` target state through internal `coco.declare_target_state(self._provider.target_state(key, DELETE_ROW))` without leaking `provider` as a public attribute; updated `mount_table` return type annotation and instantiation to `ManagedTableTarget[RowT]`.
  - `yaw/daemon/graph/storage/rows.py`: Replaced cross-module `coco.declare_target_state(target._provider.target_state(key, DELETE_ROW))` with `target.delete_row(key)`, annotated `targets: list[ManagedTableTarget[Any]]`, and removed unused `DELETE_ROW` and `cocoindex as coco` imports.
  - `tests/core/test_architecture.py`: Added AST containment test `test_zero_cross_module_private_attributes` scanning `yaw/daemon/graph/storage/*.py` (exempting `managed._conn` in `db.py`), `yaw/daemon/indexing.py`, `yaw/daemon/ingestion/*.py`, and `yaw/daemon/graph/{planner,normalization,ppr}.py` for non-self/cls/super private attribute access, and strictly forbidding `node.attr == "_provider"` across all `yaw/**/*.py` outside `managed_sqlite.py`.
  - `tests/daemon/graph/storage/test_managed_sqlite.py`: Added `test_managed_table_target_delete_row` verifying live row insertion, `isinstance(nodes, ManagedTableTarget)`, and subsequent row deletion through `.delete_row(key)`.
- **Gate results & evidence paths**:
  - `git grep -n "_provider" -- yaw/daemon/graph/storage/rows.py`: `0` matches.
  - `outputs/benchmarks/n1_managed_table_target_gate.xml`: `24 passed in 4.48s`.
  - Mandatory quality gates: `uv run --locked ty check` (`All checks passed!`), `uv run --locked ruff check --fix` (`All checks passed!`), `uv run --locked ruff format --check` (`173 files already formatted`), `uv run --locked pytest tests/core/test_architecture.py tests/daemon/graph/storage/test_publication.py tests/daemon/graph/storage/test_managed_sqlite.py --no-testmon -q` (`24 passed`), `git diff --check` (`0` errors).
- **What it does not establish**:
  - Checkpoints N2–N5 (`chunk_file_memo` text memoization, shared `coco.Environment` registry, `@coco.fn(memo=True)` mounted components, and incremental live component event maps) follow in their respective units.

### Completed Unit: Checkpoint N2 (2026-11-09)

- **Changed symbols & modules** (governing contracts: [native-cocoindex-indexing.md §2, §5, §7, §8](../native-cocoindex-indexing.md#8-checkpoints) and [native-cocoindex-graph.md §2, §5, §7, §8](../native-cocoindex-graph.md#8-checkpoints)):
  - `yaw/daemon/ingestion/source_targets.py`: Bumped `chunk_file_memo` to `@coco.fn(memo=True, version=2)` returning `tuple[tuple[SourceMetadata, str], ...]`, exposed public `chunk_file_memo`, and updated `declare_source_file` to iterate `for meta, _ in contributions: table.declare_row(row=meta)` and return `contributions: tuple[tuple[SourceMetadata, str], ...]`.
  - `yaw/daemon/indexing.py`: In `process_chunk`, checked `if settings is not None and pointer.start_line > 1:` so AST parent context is hydrated when settings are passed; in `process_file`, passed `text` from `(metadata, text)` returned by `declare_source_file` directly into `process_chunk(..., text, ...)` with zero secondary `file.read_text()` or `chunk_source_text()` calls.
  - `yaw/daemon/ingestion/bundles.py`: Removed `_process_bundle_file`, `source_snapshot.db`, and the secondary document re-read/re-chunking loop; rewrote `_bundle_snapshot_app_main` to return `dict[str, tuple[tuple[SourceMetadata, str], ...]]` directly from `await app.update().result()` via public `chunk_file_memo`; in `snapshot_dataset_bundle`, constructed `ChunkPointer` directly from the returned `(meta, text)` pairs with 0 temporary databases.
  - `yaw/daemon/ingestion/__init__.py`: Registered `"chunk_file_memo": ".source_targets"` in `_PUBLIC_API`.
  - `tests/daemon/ingestion/test_datasets.py`: Added explicit assertions `assert not any((tmp_path / "state").glob("*.db"))` in both multi-process memo-update and complete bundle snapshot hydration fixtures.
  - `tests/daemon/ingestion/test_source_targets.py`: Updated `declare_files_fixture` to annotate and return `tuple[tuple[tuple[SourceMetadata, str], ...], ...]`, verified `meta` is `SourceMetadata` without `text` attribute and chunk text is received in the tuple pair, and updated `test_source_settings_included_files_selects_only_requested_paths` unpack pattern.
- **Gate results & evidence paths**:
  - Removal inventory exit check (`git grep -n -e "source_snapshot.db" -e "_process_bundle_file" -- yaw/ scripts/ tests/`): `0` matches.
  - Private import check (`uv run --locked pytest tests/core/test_architecture.py -k test_zero_cross_module_private_imports_and_exports`): `1 passed`.
  - Named artifact `outputs/benchmarks/n2_bundle_memo_text.xml`: `36 passed in 10.85s`.
  - Mandatory quality gates: `uv run --locked ty check` (`All checks passed!`), `uv run --locked ruff check --fix` (`All checks passed!`), `uv run --locked ruff format --check` (`174 files already formatted`), `uv run --locked pytest tests/daemon/ingestion/test_datasets.py tests/daemon/ingestion/test_source_targets.py tests/daemon/ingestion/test_corpus.py --no-testmon -q` (`36 passed`), `git diff --check` (`0` errors).
- **What it does not establish**:
  - Checkpoints N3–N5 (`acquire_coco_environment` shared environment registry, `@coco.fn(memo=True, version=1)` mounted components with `FROZEN_CHUNKS` context keys, and incremental `CorpusIndexingLiveComponent` event maps) follow in their respective units.

### Completed Unit: Checkpoint N3 (2026-11-09)

- **Changed symbols & modules** (governing contracts: [native-cocoindex-indexing.md §2, §5, §7, §8](../native-cocoindex-indexing.md#8-checkpoints) and [native-cocoindex-graph.md §2, §5, §7, §8](../native-cocoindex-graph.md#8-checkpoints)):
  - `yaw/core/config.py` & `yaw/core/__init__.py`: Added pure `Path` helper `resolve_coco_state_root(state_dir)` in `config.py` and exported through `yaw.core` / `_PUBLIC_API`.
  - `yaw/daemon/graph/storage/db.py`: Re-exported `resolve_coco_state_root` from `yaw.core` for graph callers.
  - `yaw/daemon/ingestion/source_targets.py` & `yaw/daemon/ingestion/__init__.py`: Implemented `SharedCocoRuntime(environment, context, apps, ref_count)`, `acquire_coco_environment(state_root)`, and `close_coco_environment(state_root, force=False)` guarded by module lock `_COCO_RUNTIME_LOCK` and keyed by `resolve_coco_state_root(state_root)`; registered in ingestion `_PUBLIC_API`.
  - `yaw/daemon/indexing.py`: Re-exported `SharedCocoRuntime`, `acquire_coco_environment`, `close_coco_environment` from `yaw.daemon.ingestion.source_targets`; in `bind_native_indexing`, lazily acquired shared environment on first `execute()` call without eager leaks, eliminated inner double-acquire and finally-close, registered close callback via `engine.add_close_callback`, removed spurious `SOURCE_ROOT` provide from `execute()`, and made `process_chunk` access `SOURCE_ROOT` lazily only when hydrating from disk; cached `coco.App("yaw_frozen_indexing", ...)` and dynamic `FrozenChunkStream` in module registries; updated `frozen_indexing_app_main` to read dynamic `effective_settings`.
  - `yaw/daemon/runtime.py`: Deleted `_release_idle_coco_env` classmethod and call sites at lines 329 and 1130; deleted surviving closer shim `self._close_coco_app()` in `evolve_prompts`; updated `_close_coco_app` to invoke `close_coco_environment(self.config.state_dir)`; updated `_build_coco_app` to use `acquire_coco_environment(state_dir)` and reuse `shared.apps["yaw_indexing"]`; updated `_close_components` to call `_close_coco_app`; updated `_create_bound_engine` to pass `state_root=cfg.state_dir` to `bind_native_indexing` without redundant second close callbacks.
  - `yaw/daemon/graph/engine.py`: Added `self._close_callbacks: list[Callable[[], None]] = []`, `YawGraphEngine.add_close_callback(self, callback: Callable[[], None]) -> None`, and executed close callbacks in `YawGraphEngine.close()`.
  - `yaw/daemon/ingestion/bundles.py`: Replaced isolated `coco.Environment(...)` instantiation in `snapshot_dataset_bundle` with local `from .source_targets import acquire_coco_environment, close_coco_environment` (zero inverted cross-domain imports) passing `state_dir if state_dir is not None else bundle.root` directly without nested `coco_lmdb/coco_lmdb` paths; cached `coco.App("yaw_bundle_source", ...)` in `shared.apps["yaw_bundle_source"]`; added `BUNDLE_SOURCE_SETTINGS` context key and wired `_bundle_snapshot_app_main` to read from context when called through the cached app.
  - `tests/core/test_architecture.py`: Preserved strict domain containment rules with zero punch-outs.
  - `tests/daemon/graph/test_indexing_overlap.py`: Added `test_shared_coco_environment_lifecycle_and_app_caching` validating reference counting, environment deduplication, app caching on `shared.apps`, isolated cleanup assertions for the roots under test, and zero presence of retired symbols `_release_idle_coco_env`, `_ActiveCocoAppRegistry`, `register_active_coco_env_closer`, and `release_idle_coco_env`.
- **Gate results & evidence paths**:
  - Removal inventory exit check (`git grep -n -e "_release_idle_coco_env" -e "_ActiveCocoAppRegistry" -e "register_active_coco_env_closer" -e "release_idle_coco_env" -- yaw/ scripts/ tests/`): `0` matches.
  - Named artifact `outputs/benchmarks/n3_coco_memo_env_cutover.xml`: `78 passed in 149.52s`.
  - Mandatory quality gates: `uv run --locked ty check` (`All checks passed!`), `uv run --locked ruff check --fix` (`All checks passed!`), `uv run --locked ruff format --check` (`174 files already formatted`), `uv run --locked pytest tests/core/test_architecture.py tests/daemon/test_runtime.py tests/daemon/graph/test_indexing_overlap.py tests/daemon/ingestion/test_datasets.py --no-testmon -q` (`78 passed`), `git diff --check` (`0` errors).
- **What it does not establish**:
  - Checkpoints N4–N5 (upgrading $F = \text{indexing\_app\_main}$ with incremental `_active_files`, `@coco.fn(memo=True, version=1)` on `declare_source_file` and `process_chunk`, `64`-chunk micro-batching inside `CorpusIndexingLiveComponent`, input-only `graph_input_fingerprints` in `planner.py`, and retiring the parallel frozen pipeline `frozen_indexing_app_main`, `process_frozen_chunk`, `frozen_chunk_metadata`, `partition_frozen_chunks`, `FrozenChunkStream`, `"yaw_frozen_indexing"`, and `process_file`) follow in their respective units.

### Completed Unit: Checkpoint N4 (2026-11-09)

- **Changed symbols & modules** (governing contracts: [native-cocoindex-indexing.md §2, §5, §7, §8](../native-cocoindex-indexing.md#8-checkpoints) and [native-cocoindex-graph.md §2, §5, §7, §8](../native-cocoindex-graph.md#8-checkpoints)):
  - `yaw/daemon/ingestion/source_targets.py`: Decorated `declare_source_file` with `@coco.fn(memo=True, version=1)`.
  - `yaw/daemon/graph/planner.py`: In `plan_graph_publication:854`, keyed `input_fingerprints` purely on `fingerprint_object((settings, out.pointer, out.text)).hex()`, eliminating extraction AST dependencies from input change detection since `settings` contains `program_identity` and `embedding_identity`.
  - `yaw/daemon/indexing.py`:
    - Defined and exported `ACTIVE_SOURCE_SETTINGS`, `ACTIVE_INFERENCE_IDENTITY`, and `ACTIVE_GRAPH_SETTINGS` `coco.ContextKey`s.
    - Decorated `process_chunk` with `@coco.fn(memo=True, version=1)`.
    - In `CorpusIndexingLiveComponent`:
      - Maintained `self._active_files: dict[str, FileLike] | None = None`, `self.remaining_chunks: int = 0`, and `self._batch_size: int = batch_size` (default `64`).
      - Exposed encapsulated methods `reset_active_files()`, `init_active_files(files)`, `set_active_file(key, value)`, `delete_active_file(key)`, and property `active_files` to avoid cross-object private attribute access under Rule 5b.
      - In `_CorpusFeedSubscriber`: `update_all()` resets active files; `update(key, value)` sets in $O(1)$; `delete(key)` pops in $O(1)$; both await `operator.update_full()`.
      - In `process()`: populates `_active_files` from `self._files.items()` only when `None`; reads dynamic `ACTIVE_SOURCE_SETTINGS`, `ACTIVE_INFERENCE_IDENTITY`, and `ACTIVE_GRAPH_SETTINGS` falling back to constructor defaults; filters `self._active_files` using `build_source_path_matcher(effective_settings)`; mounts `declare_source_file` (`@coco.fn(memo=True, version=1)`) per file at `coco.component_subpath("source_file", rel_path)`; queries only `SELECT chunk_id FROM chunks` (zero queries on `chunk_inference` or `source_chunks`) and partitions discovered chunks against `meta['graph_input_fingerprints']` (`fingerprint_object((graph_settings, pointer, text)).hex()`) into `(retained, changed)`; admits `(*retained, *changed[:batch_size])`; sets `self.remaining_chunks = max(0, len(changed) - batch_size)`; mounts `process_chunk` (`@coco.fn(memo=True, version=1)`) per active chunk at `coco.component_subpath("chunk", metadata.chunk_id)`; and mounts `publish_graph` at `coco.component_subpath("publication")`.
      - Deleted unused `process_file` helper.
      - Aligned `partition_frozen_chunks:607` to pure 3-tuple `fingerprint_object((settings, pointer, text)).hex()`.
      - Exposed `batch_size: int = 64` on `indexing_app_main` forwarded to `CorpusIndexingLiveComponent`.
  - `tests/daemon/graph/test_indexing_overlap.py`:
    - Added `test_corpus_feed_subscriber_incremental_active_files_event_handling` verifying O(1) file addition/deletion and bulk invalidation.
    - Added `test_corpus_indexing_live_component_micro_batching_and_memo_hits` verifying 3-pass incremental batch admission (`5` chunks with `batch_size=2`: `remaining_chunks` progresses `3 -> 1 -> 0`), database state progression (`2 -> 4 -> 5` chunks), component-level memo hits, and zero provider calls on no-op replay.
    - Added `test_active_context_keys_override_constructor_defaults` proving dynamic context key overrides over constructor fallbacks, verifying that `included_files=("doc.md",)` in `ACTIVE_SOURCE_SETTINGS` filters out `doc2.md` from `source_chunks` and `chunks`.
- **Gate results & evidence paths**:
  - Named artifact `outputs/benchmarks/n4_unified_indexing_component.xml`: `81 passed in 80.35s`.
  - Mandatory quality gates: `uv run --locked ty check` (`All checks passed!`), `uv run --locked ruff check --fix` (`All checks passed!`), `uv run --locked ruff format --check` (`174 files already formatted`), `uv run --locked pytest tests/core/test_architecture.py tests/daemon/test_runtime.py tests/daemon/graph/test_indexing_overlap.py tests/daemon/ingestion/test_datasets.py --no-testmon -q` (`81 passed`), `git diff --check` (`0` errors).
- **Discoveries during implementation**:
  - `_CorpusFeedSubscriber` previously accessed `self._component._active_files` directly; encapsulated through `reset_active_files()`, `set_active_file()`, and `delete_active_file()` methods on `CorpusIndexingLiveComponent` to strictly satisfy Rule 5b AST checks without punch-outs.
  - CocoIndex component instances are re-instantiated on discrete `app.update_blocking()` cycles in non-live runs; setting `CorpusIndexingLiveComponent.last_instance = self` on `__init__` enables test suites and lifecycle harnesses to observe `remaining_chunks` and `last_summary` reliably across discrete batch passes.
- **What it does not establish**:
  - Checkpoint N5 (retiring parallel frozen pipeline and cutting callers over to Target = F(Source)) is completed below.

### Completed Unit: Checkpoint N5 (2026-11-09)

- **Changed symbols & modules** (governing contracts: [native-cocoindex-indexing.md §2, §5, §7, §8](../native-cocoindex-indexing.md#8-checkpoints) and [native-cocoindex-graph.md §2, §5, §7, §8](../native-cocoindex-graph.md#8-checkpoints)):
  - `yaw/daemon/indexing.py`:
    - Completely deleted parallel frozen pipeline: `frozen_indexing_app_main`, `process_frozen_chunk`, `frozen_chunk_metadata`, `partition_frozen_chunks`, `FrozenChunkStream`, `_FROZEN_STREAMS`, and `"yaw_frozen_indexing"`.
    - Declared active context keys with `detect_change=True`: `ACTIVE_SOURCE_SETTINGS`, `ACTIVE_INFERENCE_IDENTITY`, `ACTIVE_GRAPH_SETTINGS`, and `ACTIVE_BATCH_STEP = coco.ContextKey[tuple[int, int]]("yaw/active_batch_step", detect_change=True)`.
    - In `indexing_app_main`: dynamic pattern matching via `effective_settings = coco.use_context(ACTIVE_SOURCE_SETTINGS)` to discover `.json` bundles alongside `.md` files; reads `ACTIVE_BATCH_STEP` and mounts `CorpusIndexingLiveComponent` with `(..., effective_batch_size, batch_step)`.
    - In `bind_native_indexing:execute`: groups chunks by `ptr.rel_posix_path` with incremental file content check on disk, resets active files when modified, streams micro-batches in `while True: await app.update(live=False).result()` incrementing `step_index` until `remaining_chunks == 0`, and reports progress via `engine.report_indexing_batch(summary, total_chunks)`.
  - `yaw/daemon/ingestion/bundles.py`:
    - Completely deleted `BUNDLE_SOURCE_SETTINGS`, `_bundle_snapshot_app_main`, `_run_bundle_snapshot_app`, and `snapshot_dataset_bundle`.
    - Preserved `validate_dataset_bundle` for dataset verification.
  - `yaw/daemon/ingestion/__init__.py`: Removed `snapshot_dataset_bundle` and registered `validate_dataset_bundle` in `_PUBLIC_API`.
  - `yaw/daemon/graph/engine.py`: Hydrated `resolve_chunk_text` import from `yaw.daemon.graph.storage` for `_hydrate_opt_chunks_pool_from_db`.
  - `yaw/daemon/runtime.py`:
    - Cut `YawRuntime.optimize()` and `benchmark()` over to single pipeline $F = \text{indexing\_app\_main}$ (`"yaw_indexing"`).
    - Multi-dataset bundle support: resolves `roots = {bundle.root.parent for bundle in dataset_bundles}`, sets `workspace_root = bundle_root`, and indexes relative paths `rel_path = (bundle.root / doc.path).relative_to(bundle_root).as_posix()`.
  - `benchmark.py`: Points `workspace_root = bundle.root` and calls `index_chunks(included_files=tuple(matched_files))` directly through `indexing_app_main`.
  - `tests/`:
    - `tests/daemon/graph/test_indexing_overlap.py`: Replaced deleted `partition_frozen_chunks` with incremental step admission (`index_chunks(chunks[:2])` then `index_chunks(chunks)`), canonical `compute_chunk_id` in legacy fixtures, and updated memo-reuse tests.
    - `tests/daemon/graph/test_evidence_revisions.py`: Replaced frozen indexing references with native `indexing_app_main` and canonical `compute_chunk_id`.
    - `tests/daemon/graph/test_batch_equivalence.py`: Updated `frozen_document` to write single-chunk files (`doc_{index:03d}.md`) with canonical IDs, updated `b1_chunk_ids` to eliminate substring hex collision, and verified 20-trial 64+64 vs 128 batch equivalence.
    - `tests/daemon/test_runtime.py`: Removed obsolete `optimize_coco` assertions and aligned path assertions to `"documents/" in ...`.
- **Gate results & evidence paths**:
  - Removal inventory exit check (`git grep -n -e "frozen_indexing_app_main" -e "process_frozen_chunk" -e "frozen_chunk_metadata" -e "partition_frozen_chunks" -e "FrozenChunkStream" -e "yaw_frozen_indexing" -e "yaw_bundle_source" -e "_bundle_snapshot_app_main" -e "BUNDLE_SOURCE_SETTINGS" -e "snapshot_dataset_bundle" -- yaw/ scripts/ tests/`): `0` matches.
  - Named artifact `outputs/benchmarks/n5_unified_pipeline_cutover.xml`.
  - Paired 64+64 vs 128 acceptance report `outputs/benchmarks/g5_g6_acceptance.json`: 20/20 trials passed with exact canonical 15-table parity, 0 re-embedded chunks, 0 unbounded selects, 0 unaffected writes, and conforming SHACL.
  - Mandatory quality gates: `uv run --locked ty check` (`All checks passed!`), `uv run --locked ruff check --fix` (`All checks passed!`), `uv run --locked ruff format --check` (`174 files already formatted`), `git diff --check` (`0` errors).
- **What it does not establish**:
  - Does not hot-reload any long-running daemon in memory until restarted; does not complete full-corpus GEPA optimization runs (separate user-initiated command).

### Completed Unit: Checkpoint N6 (2026-11-10): Optimization Sandbox Fresh Lifecycle, Configurable Source Chunking, and Path Matcher Invariants

- **Changed symbols & modules** (governing contracts: [native-cocoindex-indexing.md](../native-cocoindex-indexing.md) and [AGENTS.md](../../AGENTS.md)):
  - `yaw/daemon/runtime.py`:
    - In `_create_bound_engine`: constructed `SourceSettings` populated directly from `cfg` (`chunk_size`, `chunk_overlap`, `min_chunk_size`, `included_patterns`, `excluded_patterns`, `json_keys_whitelist`, `json_keys_blacklist`) and passed it to `bind_native_indexing(..., source_settings=src_settings)`.
  - `yaw/daemon/indexing.py`:
    - In `bind_native_indexing`: extracted `base_src = source_settings if source_settings is not None else SourceSettings((), ())` and passed `chunk_size`, `chunk_overlap`, `min_chunk_size` into `GraphPlanSettings` so graph planning matches source chunking.
    - Preserved `included_patterns=("**/*",)` when `eff_included is not None` so explicit target file lists (e.g. dataset bundles) bypass directory pattern exclusions.
  - `yaw/daemon/ingestion/source_targets.py`:
    - Fixed `_SelectedFilesPathMatcher`: when `included_files` is specified, `is_file_included(path)` checks `path.as_posix() in self._included_files` directly without redundant pattern rejection; `is_dir_included` allows parent directories and workspace root `.`.
  - `yaw/daemon/cli.py`:
    - Added `optimize_coco` cleanup to `_clean_optimization_artifacts` so `--fresh` purges optimization state directories alongside SQLite databases.
  - `yaw/daemon/graph/engine.py`:
    - In `_setup_optimization_splits`: added hydration fallback `if not self._opt_chunks_pool and self._opt_indexed_cids: self._hydrate_opt_chunks_pool_from_db()` to ensure bare resume reconstitutes chunk pool metadata.
  - `tests/daemon/graph/test_indexing_overlap.py`:
    - Realigned substage progress assertion with `indexing: synonym adjudication (` emitted by planner.
- **Gate results & evidence paths**:
  - `uv run ty check` (`All checks passed!`).
  - `uv run ruff check --fix` and `uv run ruff format` (`All checks passed!`, `187 files left unchanged`).
  - `uv run pytest tests/daemon/test_runtime.py` (`7 passed`).
  - `uv run pytest tests/daemon/graph/inference/test_parsers.py` (`11 passed`).
  - `uv run pytest tests/daemon/graph/test_shacl_publication.py` (`9 passed`).
  - `uv run pytest tests/daemon/graph/test_indexing_overlap.py` (`10 passed`).
  - `uv run pytest tests/daemon/ingestion/test_source_targets.py` (`1 passed`).
  - `git diff --check` (`0` errors).
- **Discoveries during implementation**:
  - `chunk_file_memo` in CocoIndex is keyed on `SourceSettings`. When `runtime._create_bound_engine` omitted `source_settings`, it defaulted to 1200 chars / 100 overlap, preventing user-configured `chunk_size = 1000` from taking effect.
  - When callers provide `included_files`, `_SelectedFilesPathMatcher` must not conjoin with pattern filters that may only include `*.md`, allowing dataset bundles (`.json`) to be loaded when explicitly requested.
  - Critic Audit Finding: Resource label SHACL repair actions must remain strictly `{"set_pref_label"}` to preserve three-layer parity with GBNF grammar (`actions == {"set_pref_label"}` activates `value_rule = "label"`). Restored strict mapping.
- **What it does not establish**:
  - Full-corpus overnight GEPA optimization run (separate long-running job).
