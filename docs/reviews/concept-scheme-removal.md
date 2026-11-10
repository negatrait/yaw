# Scheme removal and baseline restart record

Started 2026-11-05 on `dev`. Governing plan: [native graph publication](../native-cocoindex-graph.md), checkpoint G1.

Scope: remove all ConceptScheme implementation/consumers, repair genuine test
contracts and quality failures, and capture a scheme-free baseline for subsequent
128-chunk work. Preserve existing source experiments and documentation changes.
No new batching/planner architecture or dependency is introduced at this checkpoint.

Additional authorized scope: enforce NER ownership of resource discovery and
definitions, and restrict triple extraction to supplied resource endpoints plus
new literal objects. Preserve homograph senses already present in the NER catalog.
Reject out-of-catalog endpoints; never repair them by inventing resources.

Definition of done: no active scheme path/schema, ordinary entity definitions
and source-supported graph behavior retained, real SQLite/HTTP/restart tests and
full project checks pass, and baseline evidence records code/profile/source/cache
identities and measured counts/timings. No live cutover claim without process
verification. Any unavailable live service is reported as a failed gate, not skipped.

- [x] Remove scheme inference, grammar/parser, planner and traversal surfaces.
- [x] Remove scheme storage/serialization/watchdog and reject obsolete databases
  without modifying them; retain the existing atomic publication path.
- [x] Replace scheme tests with absence/rejection and retained graph invariants.
- [x] Repair baseline failures at their real contract boundaries; no silent fallbacks.
- [x] Pass locked sync, ty, Ruff, full pytest and diff checks; record results.
- [ ] Capture scheme-free baseline under ignored `outputs/`, reconcile G1 status
  and commit only the completed assigned work with conventional messages.

Initial observation: `test_engine_index_and_caching` fails because its synthetic
embedder lacks the `.profile.index_identity` required by native indexing. Fix the
test provider contract rather than weakening production identity validation.
Earlier suite evidence: 57 failed, 318 passed, 49 deselected (testmon selection),
11 lint errors and three formatting discrepancies. These are not new acceptance
results. All checkpoints above remain open until reverified.

Implementation evidence (2026-11-05):

- Retired stores fail before persistent changes; real SQLite tests verify their
  bytes and saved experiment rows remain unchanged. Fresh stores contain no
  scheme tables, scheme edges or scheme Turtle. Shared-entity PPR connectivity
  remains present.
- NER alone owns resource discovery/definitions. Statements can introduce literal
  objects, not resources. Exact existing NER sense pairs disambiguate homographs.
  Default demonstrations and optimizer feedback obey this boundary. Definition
  repair cannot append unrequested resources or match definitions by array position.
- Real HTTP tests cover two-stage handoff, literal objects and rejected resource
  invention. Native file/watcher tests verify a rejected extraction leaves the
  published graph intact and can recover with corrected extraction.
- Baseline repairs include provider profile contracts, persistent native artifact
  assertions, query-only prefix cache reuse, source hydration during SHACL repair,
  and non-duplicated nested repair token accounting. No service fallback or retry
  reduction was introduced.
- Targeted final checks: 148 parser/model/engine tests (plus three subtests),
  14 native indexing/watcher tests, and eight learning tests passed.

Final gates, 2026-11-05, on unchanged implementation code:

- `uv sync --locked`: 100 packages resolved/checked.
- `uv run --locked ty check`: passed.
- `uv run --locked ruff check --fix`: passed; `uv run --locked ruff format` completed.
- `uv run --locked pytest --no-testmon -q --tb=short`: **848 passed, three subtests
  passed, 79 warnings**, 535.69 seconds. No deselection or live-test skip.
- `git diff --check`: passed.

Full-suite artifacts: `outputs/graph-spec-review/full-complete.log` and
`outputs/graph-spec-review/full-complete.xml`. Earlier failed and interrupted
runs remain preserved separately. No long-running daemon, watchdog or optimizer
was restarted; this evidence does not claim an operational cutover or corpus parity.

The 128-vs-64+64 timing/count baseline is not captured by these tests. G1 as a
whole remains open until that artifact exists. The current code is the scheme-free
checkpoint; baseline capture is now deferred until fixed-evidence equivalence and
bounded publication pass under [atomic execution units](incremental-publication-work.md).
Graph-wide staging and later G2-G6 work remain open.
