# Incremental publication: atomic execution units

Started 2026-11-05 on `dev`, from verified scheme-free checkpoint `c01f4aa`; updated 2026-11-07 from `c6fa080`.
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
  - `yaw/daemon/graph/engine.py` & `yaw/daemon/graph/planner.py`: Added `YawGraphEngine.emit_indexing_progress`, wired `extraction.progress_callback` into `prepare_synonym_candidates`, `judge_synonyms` batches, and `heal_shacl_violations` focus nodes in `plan_graph_publication`, and updated `rlm_stats["shacl_healer_tokens"]` to sum `rlm_tokens if rlm_tokens > 0 else direct_tokens`.
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
