# Synonym candidate search: investigation and proposed Epic 1 unblocker

2026-11-10. CS1 implementation and final verification are recorded below. The
initial investigation and superseded proposals are retained as historical evidence. Baseline: `dev`, HEAD
`1b33dac86558c094cf152b438c300f9946df4ad2`, with pre-existing working-tree edits
preserved, including `graph_ops.py`. Controlling contracts:
[native graph publication](../native-cocoindex-graph.md),
[W3C improvements](../w3c-improvements.md),
[incremental publication review](incremental-publication-work.md), and
[Epic 3](../epic-3-performance.md). This proposes a bounded Epic 1 throughput
repair, not starting the remaining Epic 3 program.

Updated after the user's 2026-11-10 review: prioritize speed, small code and low
memory over reproducing the old heuristic. Exact scores, candidate lists,
ordering and vector bytes are not acceptance targets against the old algorithm.
The new algorithm must still produce the same canonical identities, facts,
relations and evidence bindings for fixed-evidence 64+64 and 128 publication.
"Roughly the same ballpark" applies to usefulness relative to the old heuristic;
it does not permit batch-dependent identities or lost source evidence.
The subsequent review adds a hard calibration target: candidate volume within
90–110% of the baseline at the default volume setting. Large volume changes
are not acceptable merely because CPU scoring becomes faster.

## Findings

`yaw/daemon/graph/storage/graph_ops.py` has two lexical scans:

- `find_entity_disambiguation_candidates`: changed entities against all staged
  entity names, canonical entity-ID orientation, adjudicated-pair exclusion,
  independent dense and lexical channels, then exact normalized-label joins.
- `load_affected_neighborhood`, step 1c: delta names against stored names and
  raw labels, before staging. Its matcher orientation is delta-as-seq2, unlike
  the canonical-ID orientation of disambiguation. These are baseline details,
  not requirements for the replacement scorer.

The first already precomputes normalization and applies a length upper bound.
Nevertheless, for 64 sampled changed entities and 8,659 entities it visits
552,096 pairs, invokes the quick-ratio path 277,122 times, and performs only
6,025 final ratios in the isolated lexical baseline. Profiling the actual full
function, after existing adjudication exclusions, records 277,094 quick-ratio
calls and 132,949 `set_seq2` calls. Canonical orientation means the supposedly
hoisted seq2 changes repeatedly for lower-ID passive partners. The profile
contains 22,675,031 function calls; quick-ratio consumes 2.263 seconds and
`set_seq2`/index construction 1.034 seconds under profiling. These are profiler
timings, not normal wall-time measurements.

Dense search already uses float32 matrix products. It still fully stable-sorts
each distance row to retain a small k. Incremental reverse discovery can compare
R passive rows with all N entities: worst-case O(N^2 d), even with only 64
changed chunks. Step 1d of affected-neighborhood loading also uses full sorts,
including its distinct 4,096-wide neighborhood policy. Do not replace that
policy with synonym top-k merely to emulate the baseline. The proposed
replacement below removes ranking as the candidate-selection contract.

## Revised proposal: pairwise thresholds, sparse trigrams, no ranking

Use one canonical API, `find_entity_candidates`, for neighborhood discovery and
staged adjudication. Refactor consumers and exports together; remove superseded
candidate functions, duplicate scans, obsolete top-k settings and call sites.
No forwarding aliases, compatibility wrappers or alternate production path.
Stored source labels are semantic evidence, not Python API aliases.

1. Normalize names once. Include exact normalized matches through a hash lookup.
   Represent each distinct owner/normalized-label pair by a binary CSR row of character
   trigrams with word-boundary padding, including short names. Use actual trigram
   keys, not collision-prone hashing; accumulate intersections in int32.
2. Compute lexical Dice similarity from sparse intersections:
   `2 * shared_trigrams / (trigrams_a + trigrams_b)`. Apply a lexical threshold.
   Remove SequenceMatcher, its cascade and its order-dependent orientation.
3. Compute cosine scores for changed-to-existing/new pairs in bounded NumPy
   blocks, normalize each vector once, and retain pairs above the dense threshold.
   Remove full sorting, top-k and reverse-neighbor rescans. Normalization and
   floating-point implementation may change; byte-for-byte scores are unnecessary.
4. Union both channels and exact matches, canonicalize endpoint IDs and deduplicate.
   Similarity proposes pairs; existing semantic adjudication decides identity.
   Preserve resource/literal boundaries and source-supported facts.

The pool feeds the complete `JudgeLabels` -> `JudgeHierarchy` -> `JudgeRelated`
cascade. Useful outputs include merges/altLabel/hiddenLabel, broader/narrower,
related, and explicit differentFrom for confusable distinct senses. Exact-name
lookup must retain every distinct sense ID. Sharing a lexical feature row is
only a computation optimization; never collapse resources by normalized name.
Keep same-name pairs even when their definition embeddings have low cosine.

No new dependency. No retained secondary index or cache in the first version.
Normalize the hydrated float32 matrix in place; use 16-row query blocks and reuse
dense score/mask buffers. Keep lexical results sparse until a block needs them.
Allocate Python pair records only for emitted candidates. Avoid a dictionary,
tuple or temporary array per compared pair. Sparse multiplication still needs
block-local allocations; this is bounded working memory, not zero allocation.
Memory is O(N*d + total trigrams + B*N + emitted pairs), with fixed block size B;
never allocate an N-by-N matrix. Dense work per update is O(M*N*d), where M is
changed entities, rather than the reverse search's potential O(N^2*d) per update.
Across a full corpus build exhaustive pairwise dense search is still quadratic.
Sparse lexical work depends on shared-gram posting density.

This is the preferred design to measure, not a claim that its speed is already
established. The old 0.70 lexical threshold does not transfer automatically to
trigram Dice. Choose thresholds with one small frozen-input comparison; inspect
candidate volume and known duplicate/bridge examples. A faster CPU stage that
floods the LLM judge with pairs is a throughput regression.

### Candidate-volume control and calibration

Expose one volume-control hyperparameter, `candidate_volume_ratio`, default `1.0`.
It is an offline calibration target relative to the old algorithm, not a runtime
top-k, per-chunk quota or adaptive per-batch threshold. Calibration emits fixed
dense and lexical cutoffs; production only applies these cutoffs. Changing the
volume target requires recalibration. Persist cutoffs with the program/settings
identity so a changed policy invalidates affected candidate/judge work.

Count unique eligible unordered pairs at the same boundary in both algorithms:
after changed-endpoint filtering, self-pair removal, decision exclusions and
exact-label union, before judge batching or deterministic consolidation. For a
baseline count C and ratio r, aim for r*C pairs within ±10% of that target.
At the default, require `ceil(0.9*C) <= new_count <= floor(1.1*C)`. Check each
calibration input separately, not just a pooled total. If C=0, require zero.
Exact-match floors or score ties may make the band unattainable; report this
rather than inventing quota tie-breaks or changing thresholds per batch.

Fit cutoffs once on representative fixed inputs, then freeze them. No old
difflib or reverse-search baseline runs in production. Preserve the band on the
calibration inputs; this does not promise ±10% on every unseen corpus or batch.
Use a separate small input/changed-ID sample for a volume sanity check before
accepting defaults. Retain known duplicate/bridge examples: equal counts can
hide a useless selection. Batch equality is checked under one frozen policy.

Initial read-only calibration used the same two 64-entity samples as the earlier
probe. Reproducer: `outputs/synonym-search-investigation/calibrate.py`; results:
`calibration.json`; pair listings: `optimize_yaw-calibration-pairs.json` and
`yaw-calibration-pairs.json` in that directory.

| Input | Old pairs | New pairs | Change | Dense cutoff | Trigram Dice cutoff | Old pairs retained |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `.yaw/optimize_yaw.db` | 21 | 21 | 0% | 0.7545 | 0.6195 | 3 |
| `.yaw/yaw.db` | 59 | 64 | +8.47% | 0.7995 | 0.5245 | 36 |

The volume requirement is feasible on these fitting samples. Each row was fitted
independently, not with a common default. The offline count search used a small
score histogram, then validated the chosen cutoffs against actual float scores.
Those calibration scratch buffers are not part of the production algorithm.
CSR buffers occupied approximately 0.57 MB and 1.38 MB respectively, excluding
vocabulary, embeddings and scratch space. No whole-process peak was measured.

Selection changes materially: retained old pairs were 3/21 and 36/59. Spot
inspection includes old pairs such as `server-side encryption`/`server-side
retries` and new pairs such as `Commuter benefits`/`Health insurance`; neither
count nor old-pair overlap is an identity-quality oracle. These measurements
establish count calibration only. They do not establish useful duplicate recall,
held-out volume, actual judge workload or canonical graph equality. No universal
default or production acceptance is claimed.

### Manual sensibility review (2026-11-10)

Reviewed every old/new pair in both calibration listings, then read stored
definitions for ambiguous optimizer pairs and representative corpus pairs.
Correction after the user's cascade clarification: the earlier review judged
some distinct-sense pairs too narrowly as failed synonym proposals. The pool
serves all three stages, so hierarchy, association and useful differentFrom
decisions are successful candidate outcomes. The optimizer selection includes
plausible candidates for these outcomes. Count fitting alone still does not
establish cascade coverage or production acceptance.

| Candidate | Assessment |
| --- | --- |
| `Commuter benefits` / `Benefits` | Strong hierarchy candidate. Definitions say employee benefit and employee advantages/rewards. Distinct concepts, not synonymous. |
| `Commuter benefits` / `Health insurance` or `Parental leave` | Reasonable sibling/relatedness candidates, not identities. |
| `Victor Alvarez` / `Network reachability` | Useful relation candidate: stored definition explicitly identifies Victor as responsible for network reachability. Different resource roles; never merge. |
| `rollback_scope` / `Scoped rollback` | Useful relationship to judge. Definitions distinguish configuration scope from a rollback method; treating these as synonyms would be wrong. |
| `resources` / `ephemeral resources`, `cost` / `Compute-cost` | Plausible general/specific relationships. No identity claim from labels alone. |
| `SBOM upload` / `SBOM` | Sensible action/object relation. Definition distinguishes submitting a document from the document itself. |
| Route-agent percentile latency / end-to-end percentile latency | Related metrics worth distinguishing; shared percentiles do not make them the same measurement. |
| `customer data` / `Customer Tier` | Distinct concepts. A negative decision may be useful if the shared customer wording causes confusion; no identity claim. Priority depends on source context. |
| `ServiceNow` / `service` | Valid confusable-sense discrimination candidate: an IT portal versus a KMS log field. Distinctness is a useful possible outcome, not proof that selection was wasteful. |
| `actions` / `GitHub Actions` | Useful general-term/named-system ambiguity to adjudicate using definitions and source context. Do not decide identity or hierarchy from the shared word. |
| `promotional single entitled raining sunshine` / `raining sunshine` | Promising surface-variant identity candidate, but older corpus has no stored definition to establish the referent. |
| `what is peace` / `what is money`, players / international law | Shared wording alone gives weak priority, but explicit distinctness can still be useful for confusable resources. Definitions/source context are needed to assess their actual value. |
| Dates sharing `august`, `1 70 m` / `110 m` | Wasteful candidates. Different dates/measurements should not be proposed as resource identities. |

Dropping old pairs such as `ServiceNow` / `Service Down` or `home` / `rome` is
not automatically an improvement: these can be useful negative-disambiguation
examples. Low old/new overlap is neither proof of regression nor improvement.
Use the fixed candidate-volume budget for likely identity, hierarchy, association
and confusable-sense discrimination, rather than maximizing merge yield alone.

The larger `.yaw/yaw.db` supplies empty definitions for every representative
ambiguous pair queried, and includes dates, measurements and sentence fragments
as entities. It is a useful stress/volume sample, not reliable current-semantic
ground truth. No missing definitions were filled with guesses.

Conclusion: count calibration succeeds and several inspected pairs are sensible
inputs to the full cascade. The earlier low-merge-yield objection does not justify
rejecting the sparse trigram design. Keep the ±10% volume requirement and check
a small set spanning all three stages, including positive equivalence, hierarchy,
association and same-label differentFrom. This manual review used stored
definitions, not a live cascade evaluation.

Required sense-discrimination example: `ball` in "Cinderella had a ball"
(formal dance) versus `ball` in "Cindy Real had a ball" (spherical toy). Retain
two source-supported senses despite identical labels. The pair must enter
JudgeLabels, survive non-equivalence and no hierarchy, and reach JudgeRelated
for `owl:differentFrom`. Test this across the chunk boundary in 64+64 and within
128; exact lexical matching must admit the pair, not merge it. This is one small
existing-harness scenario, not a new evaluation framework.

Existing-code question to resolve during implementation:
`prepare_synonym_candidates` pre-consolidates same-normalized-name pairs when
dense similarity meets the threshold and no active fact links them. Thus not
every same-label pair currently reaches JudgeLabels. Check this path against the
ball example; do not claim universal cascade routing from candidate selection
alone or accidentally reproduce it by deduplicating normalized names.

A pairwise threshold depends on its two inputs, not unrelated neighbors or batch
size. Therefore fixed raw evidence yields the same eligible pair union whether
processed 64+64 or 128. This removes the reason for reverse top-k. It does not by
itself prove identical final graphs: merged representatives, changed definitions
and judge evidence still require correct invalidation and stable resolution.
Use stable source-supported sense inputs for pair eligibility, and verify actual
publication with the existing fixed-evidence batch test.

## Historical diagnostic measurements: exact-filter prototype

These measurements concern the earlier multiset-overlap filter followed by the
original difflib scorer. They establish the cost of the old Python loop and a
feasible native-arithmetic speedup, not the performance of the revised proposal.

Reproducer and raw JSON: `outputs/synonym-search-investigation/probe.py` and
`results.json`. Full cProfile records: `optimize_yaw-profile.txt`,
`yaw-profile.txt`; bytecode: `disassembly.txt` in that directory.

Environment: Python 3.14.9, NumPy 2.5.4, SciPy 1.18.1. Existing databases opened
with SQLite `mode=ro` and an explicit read transaction per database. 64 entity IDs
sampled deterministically across sorted ID order; these are not asserted to be
the actual changed entities of a running optimizer. Three repetitions, median
wall time, warm in-process normalized-name cache. All filter construction costs
included. Baseline and proposal timings were grouped, not alternating randomized
trials; these are diagnostic results, not final performance acceptance.

| Frozen database input | Entities | Lexical baseline | CSR + original ratio | Speedup | Full function baseline | Full function prototype |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `.yaw/optimize_yaw.db` | 3,729 | 0.535 s | 0.061 s | 8.76x | 2.478 s | 1.984 s (1.25x) |
| `.yaw/yaw.db` | 8,659 | 1.161 s | 0.241 s | 4.82x | 1.521 s | 0.487 s (3.13x) |

Lexical pair sets and float scores matched exactly: 100 and 74 pairs before
adjudication exclusion/alias union. Full-function prototypes replaced only the
lexical loop in an isolated compiled function copy, leaving SQL, dense search,
alias union and final scoring intact. Complete ordered output tuples matched
exactly: 21 and 59 candidates respectively. No source module was patched.

Filter storage was 519,320 and 1,255,424 bytes for CSR buffers alone; this excludes
feature dictionaries, temporary blocks and process RSS. Peak RSS was not measured.
Additional exact-equivalence assertions covered 265 Unicode, empty, numeric,
token-equal, repeated-character, long and seeded random labels, using both all
entities and a changed subset. This is evidence, not exhaustive proof or the
production test suite.

Database execute/fetch costs remained material in the full-function profiles,
particularly on the optimizer database. These profiles do not identify one SQL
statement as the sole culprit. The isolated lexical result must not be advertised
as an 8.76x indexing or optimizer throughput improvement.

Disassembly confirms Python pair allocation/unpacking, sorting, normalization
lookups and matcher calls within the hot loop. The baseline function has 725
static bytecode instructions. Static count alone is not dynamic work: removing
hundreds of thousands of pair visits and repeated matcher indexing is the
important reduction. No dynamic opcode-count reduction was measured.

## Historical alternative: partial top-k selection

Tie-safe partial selection was investigated to preserve the old ranking policy.
It is superseded by the threshold-only proposal; do not implement it alongside
that proposal or retain top-k solely to match historical output.

An isolated synthetic 256-by-8,659 float32 distance matrix with deliberate ties,
k=5, gave 0.1093 s full sort versus 0.00808 s partial selection (13.53x), with
identical selected indices. This excludes real cosine generation, reverse search,
SQL and indexing. It does not establish a real-vector or end-to-end speedup.

## Proposed implementation and acceptance

Implementation checkpoint `CS1` (2026-11-10, in progress): one
`find_entity_candidates` path, sparse character trigrams, 16-row normalized cosine
blocks, fixed calibrated admission cutoffs, and no top-k/reverse rescan or old API
aliases. Calibrate volume on a complete fixed evidence pool; arbitrary partitions
can vary in old/new volume and must not get individual runtime quotas. The union
of new candidates must be partition independent. A separate configurable
`auto_merge_threshold` requires cosine AND difflib ratios for both label and
definition to pass. Candidate admission cutoffs never control automatic merging.
`candidate_volume_ratio` controls offline calibration only. The production
calibration command consumes frozen baseline counts, fits fixed cutoffs and
rejects unattainable targets. No old candidate implementation is retained in
production. Quality gates, critic audit and final evidence remain pending.

1. Implement the single candidate path and update every caller. Delete replaced
   code and settings. Keep development probes under ignored outputs only.
2. Run the existing fixed-evidence 64+64 versus 128 publication check, including
   its asymmetric-neighbor and bridge cases. Compare canonical identities, facts,
   relations and bindings; do not require old heuristic scores, vector bytes,
   candidate ordering or judge prompt bytes to match.
3. Make one small before/after measurement covering candidate CPU time, peak
   memory and pair/judge-call count on the existing frozen corpus. Calibrate
   `candidate_volume_ratio=1.0` to ±10% candidate volume per input; freeze the
   cutoffs and check a separate small sample. Spot-check
   known duplicates, short labels, suffix variants and distinct homographs.
   Reuse existing tests; no new evaluation framework or full retrieval campaign.
4. Run mandatory project checks and the required critic audit before committing
   completed implementation. No implementation checkpoint is complete here.

Definition of done: visibly faster candidate discovery, small implementation,
bounded working memory, ±10% calibrated candidate volume, preserved source evidence and
the same canonical graph across batch layouts. Old/new heuristic parity is
explicitly not required. Canonical graph equality does not mean identical DB
files, floating-point similarity scores or physical index bytes.

Initial investigation boundary (before CS1): no new dependency was needed. No
LLM inference, live-process inspection/restart,
full ingestion, retrieval-parity evaluation, daemon acceptance or full project
quality-gate run was performed. Only the diagnostic assertions and read-only
candidate-function calls above ran. Existing experiments were not rewritten.


## CS1 implementation evidence (2026-11-10)

### Active contract and changes

- One source-only `find_entity_candidates` API serves neighborhood discovery and
  staged adjudication. Sparse trigram overlaps stay sparse; cosine uses reusable
  16-row float32 buffers. Removed the old two candidate APIs, all-pairs difflib,
  full sorting, reverse top-k and obsolete settings. No compatibility aliases or
  canonical-vector fallback remain. Bare storage tests declare their source
  inputs explicitly in `candidate_fixtures.py`.
  Canonical display/healed labels cannot add eligibility. After automatic merges,
  one additional search completes scores over all retained members, including
  raw pairs individually below admission; the usual path has no extra pass.
- Default candidate cutoffs are cosine `0.8`, lexical Dice `0.625`, fitted to the
  active optimizer source representation. `candidate_volume_ratio=1.0` controls
  offline fitting through `scripts/calibrate_candidates.py`, which reads the
  configured target and accepts an explicit CLI override. Cutoffs are frozen for
  runtime selection. Retired settings and nonfinite cutoffs fail explicitly.
- `auto_merge_threshold=0.98` is independent: cosine AND difflib ratios for both
  normalized label and nonempty definition must pass. Only the extreme-cosine
  shortlist runs difflib. Resolved representatives cannot cross existing
  differentFrom/broader/related separation or established fact poles.
- Retain source-sense vectors in the typed `entity_candidate_vectors` target,
  keyed by immutable source sense ID, with canonical owner, normalized name,
  definition and embedding. It uses the same SQLite DatabaseSink transaction as
  the graph. No second writer, external vector store, resident projection cache
  or dependency was introduced. Canonical retrieval vectors remain separate
  from source evidence used to propose identity/semantic decisions.
- Candidate vectors use normalized source labels and casefolded/stripped resolved
  definitions, formatted through the existing entity embedding adapter. This
  prevents casing/arrival from changing a source-sense embedding request. Source
  mentions/assertions keep their original evidence. Merges remap owners without
  discarding member vectors or fuzzy source labels; source withdrawal and SHACL
  entity deletion remove unowned rows. Publication validates source ownership.
- `publish_graph` memo version is `3`. Existing indexes predating source-vector
  retention require rebuilding from source; the planner fails explicitly rather
  than reconstructing discarded member embeddings from a canonical vector.
  Existing production/optimizer databases and running processes were not cut over.

### Calibration on the active representation

Earlier 33/122-count measurements on canonical-only databases do not validate the
new source representation. Diagnostic copies were populated using real local
`embeddinggemma-300m-qat-q8_0-GGUF-Q8_0` inference with the normalized source format
and typed CocoIndex target publication. Original databases remained read-only.
The optimizer copy uses 5,922 distinct raw senses from occurrence definitions
behind 3,729 canonical entities. The historical stress copy re-encodes 8,659
existing entities because that older corpus lacks current occurrence evidence.
Its dates/fragments are stress inputs, not current parser-quality ground truth.

| Fixed complete pool | Old unique pairs | Source-aware pairs | Change | Cosine cutoff | Dice cutoff |
| --- | ---: | ---: | ---: | ---: | ---: |
| Optimizer, 128 sampled canonical owners | 33 | 35 | +6.06% | 0.8 | 0.625 |
| Historical stress corpus, 128 sampled owners | 122 | 122 | 0% | 0.8025 | 0.6175 |

Cutoffs are corpus-specific. Both pools satisfy the default ±10% volume band.
Independent partition counts need not satisfy it: the optimizer's two 64-owner
subsets give 13 and 25 candidates versus 21 and 12 historically, with three new
pairs incident to both subsets. Their deduplicated new union is exactly the
35-pair combined pool. This is calibration of a complete evidence pool, not a
universal per-delta quota or held-out recall claim. Forcing a quota per batch
would violate pair-local discovery. New corpora need calibration of their own.

The final optimizer pair listing includes sensible candidates across the cascade:
PrivateLink/endpoints, Key ID/api_key_id, RBAC changes, scoped rollback,
KMS/SSE-KMS, policy/store, resources/ephemeral resources and log variants. Some
pairs remain ambiguous (Jordan/Jordan Blake, actions/GitHub Actions,
ServiceNow/service); explicit distinction is useful adjudication, not automatic
evidence of candidate failure. This is a manual sensibility check, not a semantic
quality score. Listing: `outputs/synonym-search-investigation/final-source-pairs.json`.

Separate real-embedding merge calibration used the actual normalized source
format. At threshold `0.98`, case-only agreement and a long punctuation-only
definition variant pass; Cindy Real/Cinderella, toy/dance ball, factual/fictional
and the negation counterexample do not. A short punctuation-only variant scores
about `0.954` and is deliberately deferred to judging. `0.95` admits that positive
too, but the default retains the stricter margin. These seven cases calibrate
known behavior, not a universal semantic proof. Evidence: `merge-calibration.json`
and `calibrate_merge.py` under `outputs/synonym-search-investigation/`.

### Performance and boundaries

The final source-aware measurement is in
`outputs/synonym-search-investigation/source-performance.json`, reproduced by
`measure.py` using the captured preimplementation functions and source-projection
copies. It includes database reads, normalization, feature construction, candidate
scoring and emission, with five timed repetitions after warming, in a separate
worker per arm. Reference-function AST construction is excluded from worker peaks.
The old arm uses the original canonical vectors; the new arm includes the richer
retained/re-encoded source vectors. This compares the actual replacement workload,
not identical vector bytes. Diagnostic host contention limits timing precision.

| Pool | Old median | New median | Speedup | Old peak worker | New peak worker |
| --- | ---: | ---: | ---: | ---: | ---: |
| Optimizer | 1.148 s | 0.127 s | 9.0x | 135.4 MiB | 133.9 MiB |
| Historical stress | 2.743 s | 0.147 s | 18.7x | 215.8 MiB | 169.1 MiB |

Both peaks are lower, while the optimizer retains 59% more source vectors.
Earlier contended runs measured 10–19x and 14–19x; the table is the final rerun
with no concurrent project test suite. These are whole-worker peaks, not
isolated scratch-buffer sizes. No sub-millisecond or full optimization-throughput
claim is made. Initial re-encoding/model work is not included in these CPU-search
times. The maintained projection/index needed to pursue another order of
magnitude is deferred to Epic 3 at the user's request.

### Verification record

The user-supplied pipeline order is now documented in
[native graph publication §1.1](../native-cocoindex-graph.md#11-pipeline-order-extract-local-evidence-before-global-adjudication):
NER -> local triples -> neighborhood candidates -> three-stage judging ->
canonical rebinding -> SHACL healing/validation -> atomic publication. The
description identifies fact-based short circuits, the four-pair maximum and the
active validation entry points from the implementation.

- Critic audit: unguided CS1 reviews found and drove fixes for fuzzy label loss,
  discarded member vectors, transitive separation, inactive volume configuration
  and nonfinite cutoffs, plus post-merge score completion and canonical difflib
  orientation. The final audit reports no remaining actionable findings. The requested
  `flash` model is not exposed in this session; the available inherited agent
  model performed the audits. No claim of a flash-model run is made.
- Real SQLite/CocoIndex fixed-evidence tests cover 64+64 versus 128, restart,
  homographs, late bridges, source replacement/retraction, and the adversarial
  case where only a merged-away member's vector crosses the admission cutoff.
  The comparator includes source-vector rows as well as canonical graph evidence.
  The final 20-trial report (2026-11-10 14:24 EET) passed all canonical comparisons
  and SHACL checks: mean cumulative time ratio `1.1701`, graph-work ratio `1.1587`,
  zero batch-one chunk re-embeddings, zero unaffected writes and zero unindexed
  readonly survivor scans in the gate's tracked canonical tables. Candidate
  source-vector scans remain linear and are not included in that zero count.
  Whole-suite peak at this gate was 507,043,840 bytes.
  Evidence: `outputs/benchmarks/g5_g6_acceptance.json`. The report's optional live
  parity probe does not establish full-corpus retrieval acceptance.
- Final full run: **980 passed, one test-fixture failure, three subtests passed**
  in 1,236.95 seconds (`complete-suite.xml` and `complete-suite.txt` under the
  investigation output directory). The success-path HTTP provider had been
  updated to supply complete cascade decisions; the rejection test still expected
  empty decisions. Its empty response is now an explicit fixture parameter,
  preserving the rejection assertion. All 34 consumers in `test_engine_edges.py`
  and `test_benchmark_edges.py` then passed. The explicit failed-test rerun passed
  (`final-failed-rerun.xml`); no production code changed after the full run.
  This reconciles coverage of all 981 tests; it is not a claim that the recorded
  full-run XML has zero failures. The failed-test cache selector unexpectedly
  selected the whole suite and was stopped in favor of the explicit test node.
- Final `uv run --locked ty check`, `ruff check --fix`, `ruff format` and
  `git diff --check` passed. Intermediate failures were fixed rather than
  skipped or weakened: storage fixtures declare source inputs and semantic
  decisions instead of relying on the old 0.8 merge shortcut.
- Diagnostic source-projection setup failed first for a missing validation context,
  then for stale target-create state in that failed diagnostic namespace. Clean
  diagnostic target state plus explicit projection ownership validation succeeded;
  completed embedding evidence was reused by content hash. This did not retry a
  production indexing publication or alter the original databases.
- Full-corpus retrieval parity, GEPA acceptance and daemon/live cutover are outside
  this bounded candidate-search unit.
