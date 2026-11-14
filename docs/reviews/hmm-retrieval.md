# H1: decoupled graph-HMM retrieval cutover

2026-11-13, dev at 876e524 plus retained uncommitted work. The user selected
"Finish the HMM cutover" during the wiring audit. This supersedes the PPR
algorithm target only; it does not approve a retrieval-quality or paper-parity
claim. Existing PPR experiment measurements remain historical evidence.

## Contract fixed before implementation

- Hidden states are entities; passages are observations, not walk states.
  Positive typed entity relations form sparse transition potentials. Passage
  edges form a separate sparse emission matrix B (passage by entity).
- Pool independent same-direction contributions with Noisy-OR within each
  relation kind, apply that kind's directional policy multiplier, then combine
  resulting typed directional potentials with Noisy-OR. Reciprocal copies of
  symmetric `skos:related` edges are one observation, using the larger pooled
  direction before undirected expansion, not two confidence votes. owl:differentFrom supplies neither
  relevance transitions nor negative evidence: distinct resources can be jointly
  relevant. Its dead inhibition path is removed.
- Entity transition potentials use D_in^-beta A D_out^-(1-beta). beta applies only
  to entity transitions, not passage emissions. The transpose direction has its
  own degree normalization. These are weighted potentials, not a stochastic
  operator over mixed passage/entity nodes.
- Node confidence is a separate observation likelihood. Missing confidence is
  neutral; known zero remains zero. Confidence must not disappear when an entity
  has only one incoming/outgoing edge, or be synthesized from missing metadata.
- Normalize the supplied nonnegative seed evidence once. Project chunk seeds
  through B^T, add direct entity seeds, then normalize the entity initial state q.
  With no usable seed, return zero evidence; never seed unrelated semantic islands.
- Forward filtering repeatedly normalizes L * ((1-d)q + d(T alpha + dangling*q)).
  d is exactly configured damping; max_iter and tol are actual bounds. With
  multiple entity observations, use the reverse potential to compute a backward
  message and normalize their elementwise product; single-observation retrieval
  uses the forward message alone. These are retrieval relevance scores, not
  calibrated truth probabilities or a convergence theorem for PPR.
- Passage relevance is Noisy-OR over B_ce * entity relevance, combined by Noisy-OR
  with the normalized direct passage seed. Entity state mass and passage
  relevance belong to separate layers; their concatenation need not sum to one.
  Rank actual passage relevance descending with deterministic ties. Remove the
  additional fixed 0.85 explaining-away heuristic: it did not rank the emitted
  relevance and had no governing measurement.
- Remove obsolete activation, activation_param, renormalize and dual_weight
  parameters and their CLI/report/caller paths. Retain active beta and damping.
  Rename the passage ranking helper to rank_passages_by_hmm without an alias.
  No fixed damping floor, iteration cap, global stationary subtraction, arbitrary
  posterior maximum of 0.90, or hidden blend/seed boost remains.

## Definition of done and quality gates

Real sparse matrices independently verify separated state/emission operators,
Noisy-OR, beta, directed relation policy, priors, damping, layer bounds, empty and
seedless graphs, legitimate islands, deterministic passage ordering, and rejection
of retired parameters. Real SQLite topology verifies confidence affects active
retrieval. Existing controlled multi-hop and runtime tests remain required.
Update current governing descriptions and all active consumers. Run type, Ruff,
format, pytest, diff checks and an unguided independent critic. Parent owns the
combined full suite. No checkpoint or commit before evidence; full-corpus Recall,
EM/F1 and real-model optimizer acceptance remain separate, mandatory gates.

## H1 implementation evidence (2026-11-14)

- Symmetric `skos:related` observations are pooled within direction, with reciprocal copies collapsed to the stronger observation before undirected expansion. Explicit directed input retains its supplied orientations. Independent facts still pool by Noisy-OR before their policy; cross-kind directional potentials then pool by Noisy-OR. SQLite topology loading preserves the stronger reciprocal stored edge.
- Backward filtering now activates from the projected entity seed, including observations originating only from passage seeds. Invalid negative graph weights fail rather than being discarded. Passage-only queries now use the same HMM path; a real SQLite indexed graph test proves a seeded passage can reach a second passage via entity relations with no retained facts.
- `tests/daemon/graph/test_ppr.py` and `test_ppr_edges.py`: 30 passed before the retrieval-version test was added. Independent arithmetic covers `.6425` cross-kind potential, `.7` reciprocal copies and projected-seed forward/backward messages; real SQLite covers topology loading. The targeted engine/runtime/policy group passed 7 tests, including two positive-evidence retrieval cases and four updated zero-island expectations.
- Before changing those expectations, an instrumented real graph run observed passage scores `[0, .757281]`, `[.998947, 0]`, `[0, .790642]`, and `[.757281, 0]`; the missing hit had zero evidence. Controlled two-island seeding produced `[.090909, .909090]`, while a hybrid positive-passage-seed case produced `[.263775, .791450]`; both returned two hits. Crash/restart also preserved the completed extraction and both published chunks.
- The retrieval-version identity enters query memo, candidate/reward and SIMBA calibration identities, while OpenIE and embedding fingerprints remain independent. The version regression passed. Benchmark damping help now identifies damping as the transition weight rather than restart probability.
- Scoped Ruff and ty passed. No checkpoint is ticked; full quality gates, independent critic, benchmark parity, real-model quality and running-process acceptance remain outstanding.

### Final integration audit follow-up (2026-11-14)

Independent Terra review found two additional active-path defects: no-fact intent retrieval could replace the original passage query, and SQLite loading filtered invalid negative weights before HMM validation. Removed recursive intent retrieval: one intent expansion contributes fact candidates, while passage dense/lexical seeding and fact relevance judging retain the original user query. Stored negative weights now fail before reciprocal pooling or filtering can hide them. Real SQLite tests cover empty, unrelated and shortened intents, and an invalid reciprocal edge. The focused HMM/retrieval suite passed 36 tests; scoped Ruff/type/format checks passed. The separately edited prompts.py was untouched by these fixes.

The earlier broader HMM/runtime/parser run passed 128 tests and failed one remaining zero-island expectation in the old recursive-intent fixture; that fixture was migrated to the single original-query contract and included in the passing focused follow-up. A fresh broader gate and independent re-review are pending; no checkpoint or commit is made.

Separate canonical-ingestion acceptance was rechecked without altering tests: outputs/wiring-audit/hmm/known-batch-parity.xml and .log record 5 passed and 6 failed in 147.12 seconds. Failures remain differentFrom/related disagreement, lost merged-member relations, and arrival-order-dependent unified definitions. These pre-existing publication/contribution issues are not repaired or waived by H1. No full-system or corpus-quality acceptance follows from HMM unit evidence.

### Parent verification and current acceptance boundary

- Fresh parent run of test_ppr.py, test_ppr_edges.py and all test_engine.py tests: 60 passed in 42.58 seconds; outputs/wiring-audit/hmm/retrieval-final.xml and .log. This includes the final original-query/intent handling and negative stored-weight fixes.
- Core capacity-binding regression was migrated to validate against resolved serving capacity rather than an offline configuration snapshot. Core/HMM run: 52 passed in 2.58 seconds (core-hmm-final.xml).
- Final independent Terra H1 re-review reported no additional actionable findings after both integration fixes. It was read-only, not an independent test execution; no Flash audit is claimed.
- uv sync --locked, type checking, Ruff and formatting passed when run with prompts.py excluded from mutation, as explicitly requested by the user because another chat is editing it. The outside-prompt whitespace check passed. No prompt instructions were changed by HMM fixes.
- The complete suite is running in full-suite.xml/.log. Its separate capacity regression run exposed concurrently pinned default output=16384 and RLM-output=65536 values replacing automatic None defaults. The budget decision is pending with the user; those configuration changes have not been silently reverted or tests weakened. These failures are separate from the six verified pre-existing canonical batch-parity failures.
- H1 implementation and focused integration are verified; global acceptance, full-corpus retrieval quality, and a reload of any existing long-running process are not established. No checkpoint is ticked and no commit is made while mandatory gates remain unresolved.

Budget decision is no longer pending: the user selected endpoint-derived automatic defaults, restored and verified by 46 passing configuration/capacity checks. Complete full-run evidence is 1072 passed/17 failed before the follow-up fixes; subsequent configuration, cache-restart and native reflection checks passed. The six independently verified canonical batch-parity failures still block global acceptance. H1's 60 passing retrieval/engine tests and clean Terra re-review remain distinct from that system-level acceptance barrier. No running optimizer/daemon was reloaded and no checkpoint was ticked or commit made.
