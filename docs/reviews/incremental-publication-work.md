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

U4 preparation: inspection of the locked connector shows a row handler/sink per
table, each opening `ManagedConnection.transaction()`. Sharing a connection alone
does not demonstrate multi-table atomicity. Real connector probes are still
required before selecting the public grouping/validation mechanism; do not assume
independent `mount_table_target` calls implement a semantic revision transaction.
