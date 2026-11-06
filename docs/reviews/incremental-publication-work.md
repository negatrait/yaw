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
