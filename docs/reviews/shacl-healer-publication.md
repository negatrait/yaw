# SHACL healer publication repair

Started 2026-11-06 on `dev`, after `d566a42`.

Goal: a closed-world statement violation reaches the native DSPy healer and its
validated repair survives canonicalization and declarative publication. Retain
raw extraction evidence; unresolved violations still abort with the original
error. No custom retries, relaxed shapes or alternative publisher.

The latest failed sandbox, `.yaw/yaw_optimize_20261106_162830.db`, has 64 native
extraction artifacts and no published graph. Its retained audit has no completed
healer result; the daemon log predates this run. A fresh optimizer reproduction
is captured in `outputs/graph-spec-review/shacl-optimize-live-before.log` without
resuming or modifying that experiment.

Read-only replay of those 64 artifacts into isolated SQLite found 74 duplicated
statement definitions, 32 homograph violations before candidate adjudication and
one scalar-label violation. Deterministic healing removed the `429s` resource and
three of its facts. The initial classifier-parity hypothesis was disproved:
both raw and normalized classifiers identify this label as scalar. The live
definition fix then exposed dangling mention bindings because deterministic pruning
had deleted that resource and its statements. Resources participating in statements
must reach semantic label healing instead of destructive scalar pruning. Evidence:
`shacl-artifact-{lint,healing-replay}.log` under
`outputs/graph-spec-review/`.

Source review also found that post-healer `canonicalize_fact_text()` restores
`facts.definition` from raw assertions, overwriting a repaired definition. Raw
assertions must remain immutable, but their malformed description must not replace
the validated derived statement description after healing.

Acceptance before commit:

- Reproduce a duplicate statement definition through native CocoIndex, DSPy RLM
  and a loopback structured repair worker; assert the repaired definition and
  unchanged raw evidence in real SQLite. First capture the failing test.
- Verify that unresolved statement violations still fail publication.
- Preserve scalar-looking resources with statement roles for semantic healing.
  The existing `set_pref_label` action may choose a lexical name grounded in the
  resource definition/passage; genuinely invalid resources remain unresolved.
- Separate canonical source-description selection from display-text refresh so
  post-repair refresh cannot undo healing. Keep final publication declarative.
- Verify the captured live failure, relevant existing contracts and all mandatory
  project gates. Record actual results and remaining limits below.

## Results in progress

- Fresh configured `uv run --locked yaw dspy optimize --output ...` reproduced the
  crash on the initial 64-chunk publication: **60** endpoint-duplicated statement
  definitions. Exit 1; evidence: `shacl-optimize-live-before.log`.
- Native RLM/worker regression initially failed at the same final definition
  check. Source descriptions now have one bulk canonical selection **before**
  healing; subsequent text/FTS refresh updates display text alone and cannot undo
  repaired definitions. Publication memo version advances from 5 to 6; extraction
  identity is unchanged, retaining valid cached extraction evidence.
- Fresh live verification of that fix reached precommit binding validation,
  exposing deterministic deletion of the scalar-looking `429s` resource. Exit 1;
  evidence: `shacl-optimize-live-fixed.log`. Scalar pruning now excludes resources
  used as subjects/resource objects. The direct repair signature instructs the
  healer to choose a source-grounded lexical name for a semantically valid
  resource. No shape is relaxed and no unresolved violation is accepted.
- `shacl-healer-scalar-focused.{log,xml}`: **148 passed**, 16.83 seconds, including
  both resource-label cases with/without a repair, storage, resource-boundary and
  build-time SHACL contracts. The earlier definition-only paired run also passed
  **55 tests**, 71.78 seconds (`shacl-healer-focused-final.{log,xml}`).
- A resumed diagnostic was stopped at its exact owned PID (12504) after it began
  filling the next 64 additions; it did not reproduce the original boundary.
  The final fresh diagnostic uses `shacl-optimize-live-verified.log`. User
  experiments and inference services were not stopped or modified.

All evidence paths above are under `outputs/graph-spec-review/`. Final live
publication evidence and mandatory gates follow before committing.

The next live run reduced the failure from 60 violations to two (one statement
definition and the scalar-looking resource). Traced incomplete healer context:
the snapshot retained only the first context-scaled 16 foci, while the worker tool
used at most eight Markdown dossiers. Fact inspection has structured data but no
Markdown dossier, so that path omitted the actual statements. The direct worker
also silently clipped violation lists and context strings. Evidence:
`shacl-optimize-live-verified.log`; an unbatched full-context attempt remained
unreliable (`shacl-optimize-live-complete.log`). Neither is a passing live gate.

The final invocation uses bounded disjoint batches (four foci at 64K serving
context, eight at the configured 128K) against one staged proposal. Every focus
has its structured neighborhood; worker requests retain all supplied items/context
instead of silently clipping JSON. Gather proposals for every batch before one
repair application and final validation. This is partitioned work, not a retry
loop. Publication/healer memo versions advance to 6/2; NER/triples remain cached.
The worker's existing alias/coercion interface is otherwise unchanged.

`shacl-batched-publish.{log,xml}`: **nine passed**, 110.49 seconds. Eight native
cases cover one/33 statements, ordinary/scalar-looking resource labels and
successful/unresolved repairs, including unchanged source evidence, live fact
bindings, vectors/FTS, complete structured context and no-op replay. The original
RLM tool/token accounting test also passes after its synthetic LM reads foci from
the violations field instead of treating every neighboring ID as a repair target.

Further live checks exposed two remaining context defects. The batched attempt
hit the RLM iteration limit and retained nine violations. The proposal-only
signature then passed SHACL but failed source assertion binding validation
(`shacl-optimize-live-proposal.log`); cached worker responses included destructive
`delete_invalid_fact` actions. These runs are not passing live gates.

The RLM's separate delegation rules and few-shot example still required the old
first-eight Markdown dossiers, contradicting the corrected signature. Replace
that example with one delegation followed by SUBMIT; the tool supplies complete
structured context. Fact dossiers now include endpoint entity profiles, making
existing endpoints explicit despite the normal empty top-level entity field on
a fact dossier. The worker signature explicitly preserves facts when repairing
their definitions. No constraints are relaxed and no custom retries are added.

`shacl-release-full.{log,xml}`: **917 passed, three subtests passed**, 813.05 seconds.
This full gate precedes only the final delegation-example and endpoint-context
corrections; focused/final verification follows. Static checks and locked sync
passed. The updated lossless prompt-budget contracts pass all 16 tests
(`shacl-budget-contracts-fixed.log`).

`shacl-endpoints-focused.{log,xml}`: **48 passed**, 162.53 seconds, covering native
publication, the engine and prompt-budget contracts after those final changes.
Ruff, formatting and ty pass (`shacl-endpoints-{ruff,format,ty}.log`).

The endpoint diagnostic still failed binding validation. Its captured proposal
confirmed deletion of source-backed facts. Worker-input inspection then exposed
a verification error: reusing the inspection output directory restored its older
GEPA seed instructions, despite fresh database names. The same applies to the
subsequent `shacl-optimize-live-worker.log` replay, which was stopped at its exact
owned PID. These do not establish final-prompt behavior.

The genuinely fresh diagnostic (`shacl-optimize-live-final-fresh.log`) confirmed
the final instructions and endpoint context in captured worker inputs, yet worker
responses still chose deletion for definition defects. It was stopped at its owned
PID once those invalid actions were observed; this is not a successful live gate.

The final boundary is therefore executable rather than prompt-only: one shared
report-to-action map binds each generated repair to an actual focus and its allowed
actions. Definition defects permit `set_definition`/`rewrite_definition`;
scalar-label defects permit `set_pref_label`; other existing action contracts stay
unchanged. The existing contextual GBNF builder prevents incompatible generation;
the parser rejects incompatible returns, including RLM returns, before application.
No new predictor, publisher or retry mechanism. Publication/healer memo versions
are now **7/3** so invalid intermediate proposals are not reused; extraction stays
cached. Add three parser/generation contract cases for destructive/relabel actions.

The active diagnostic is `shacl-optimize-live-actions-v3.log`, with a new output
directory. An ignored inspection script records worker inputs/proposals without
changing final publication. The original failed experiment remains untouched.
No complete GEPA optimization or benchmark acceptance is claimed.

`shacl-actions-focused.{log,xml}`: **96 passed**, 165.39 seconds, including the
three new incompatible-action cases plus native publication, engine, scalar/literal
parser and full-context budget contracts. `shacl-final-full.{log,xml}`: **917 passed,
three subtests passed**, 893.48 seconds, on the preceding endpoint/example revision.
The final action-contract full gate is `shacl-actions-full.{log,xml}`; its result
and final live publication evidence follow below before commit.

The action-constrained live run produced valid definition repairs for its first
three batches, then hit the RLM iteration limit. Safe inspection of cached response
strings showed repeated misspellings of the long worker-tool identifier (`NameError`),
followed by an invented extraction response. That owned run was stopped; no passing
live gate is claimed for it.

Replace the tool with `propose_repairs(focus_context="")`, reading the actual bound
report through the existing linter callback. Remove the two competing report
arguments and their coercion logic; retain optional additional hydrated context.
Update every caller, rule and demo without an alias. Missing storage hooks raise.
Saved DSPy state now carries `yaw_surface.shacl_tools="bound-report-v1"`; older
states fail immediately with instructions to start a fresh optimization. The graph
schema and NER/triple fingerprints remain unchanged. This also prevents the stale
seed verification error discovered above from silently recurring.

The new live log is `shacl-optimize-live-bound-tool.log`. The previous full run
was stopped after a failure while tool/grammar fixtures were being updated; it is
not a passing gate. Final source verification follows below.

The bound-tool run completed all workers but retained **52** copied endpoint
definitions (`shacl-optimize-live-bound-tool.log`). Strict validation correctly
rejected that proposal. The tool plumbing was fixed; definition task semantics
still needed correction. The full gate at that checkpoint passed **921 tests and
three subtests**, 925.52 seconds (`shacl-bound-tool-full.{log,xml}`).

Captured eight-fact worker A/B, same model and report-bound action grammar:

- Existing signature with compact inputs copied **8/8** endpoint definitions.
- Explicit statement-relationship task with compact inputs produced **8/8**
  relational definitions, zero copies and zero missing foci.
- The clarified general task with the original full inspector context produced
  one valid repair but omitted seven foci. Adding grammar coverage produced all
  **8/8**, zero copies and zero missing foci; multiple actions per focus remain
  possible, or the model can return an empty abstention. Parser checks the same
  nonempty-coverage contract. No extra predictor or context-formatting layer.

Evidence: `shacl-captured-statement-{probe.log,results.json}`,
`shacl-captured-rich-{probe.log,results.json}`, and
`shacl-captured-rich-coverage-{probe.log,results.json}` under
`outputs/graph-spec-review/`. These are captured live-model probes, not a corpus
retrieval/QA benchmark.

Upstream probe reuses the exact recorded passage and NER resources from
`chunk-d2883b1a219af8d59ae1734f99b2d44c`. Raw predictor output confirms that
definition/predicate copies originate in the model, rather than parser rewriting.
Clarifying resource definitions versus statement `def` reduced endpoint copies
**12 -> 0** and predicate-definition copies **1 -> 0**. All 22 new statements pass
the production parser; the saved baseline had 23. This does not establish recall
parity. Evidence: `openie-captured-{probe.log,results.json}`. The clarification is
applied to the existing triple signature; its native extraction fingerprint
changes, so older extraction artifacts are never re-tagged as new output.

`shacl-coverage-focused.{log,xml}`: **100 passed**, 202.33 seconds. Static gates
pass. Final full gate: `shacl-final-acceptance.{log,xml}`. A fixed-evidence native
replay uses the original 64 extraction records, real embedding/adjudication/healing
and declarative final targets in an isolated database; evidence is
`shacl-captured-native-publication.log` and its output directory. It is distinct
from running a fresh Open IE corpus extraction or completing GEPA.

The user-supplied dead-code and performance audits are incorporated in graph plan
§9.1/§9.2 and cross-linked in the indexing plan; the incremental review records
that planning checkpoint. Those cleanup/performance changes are not implemented
as part of this healer remedy and remain unchecked.

The first full native captured replay failed with **nine** residual violations
(`shacl-captured-native-publication.log`). Cache inspection showed a worker
abstaining on a fact batch and a scalar-label repair repeating `429s`, rather than
proof that the entire batch was successfully repaired. Adding exact frozen source
passages resolved the captured mixed scalar/fact batch; the other captured batch
still abstained (`shacl-remaining-grounded-{probe.log,results.json}`).

The typed, single-purpose existing-statement task then produced all eight
definitions for that batch with zero endpoint copies or omitted foci, with
abstention still permitted (`shacl-remaining-typed-{probe.log,results.json}`).
Apply its role clarification and a three-string `SHACLRepair` TypedDict to the
existing signatures; no new predictor. The planner supplies source passages from
the exact complete frozen chunk set. `propose_repairs()` now takes zero arguments,
so RLM-generated context cannot replace or precede the authoritative dossiers.
Keep the empty-abstention option and all final semantic/source-binding checks.

`shacl-grounded-focused.{log,xml}`: **136 passed, two live tests deselected**,
106.82 seconds, including the fresh-process SIMBA fixture. The prior full suite
had **921 passed, one failed, three subtests passed** and overlapped a source/
saved-state contract edit while the parent still held old code. The fresh-process
fixture subsequently passed in the focused run above.
That run (`shacl-release-acceptance.log`) is not a passing final gate.

User review correctly challenged `assert_differentFrom`: it was only an alias for
`add_different_from`. Remove it from constants, grammars, prompts, application and
positive tests. Likewise consolidate `rewrite_definition` into `set_definition`;
both performed the same SQL update. Unknown/malformed repair records and unknown
application actions now raise instead of silently disappearing. Negative tests
cover both retired spellings and transaction preservation. Remove the redundant
fixed action-count assertion; verify the exact canonical set. Saved state now
requires `yaw_surface.shacl_tools="canonical-report-v1"` and rejects the older
bound-report contracts. No compatibility alias or automatic artifact migration.

Final gate names: `shacl-canonical-final-{ruff,format,ty}.log` and
`shacl-canonical-final-full.{log,xml}`. The final real-model fixed-evidence replay
is `shacl-captured-native-grounded-publication.log`, using the same isolated
native target database and persistent memo state. Results follow before commit.

## Final verified checkpoint — 2026-11-06

- `shacl-canonical-final-full.{log,xml}` exited **0**: **927 passed, three subtests
  passed, 79 warnings**, 798.41 seconds. No tests skipped/deselected in this full
  run. Ruff, formatting and ty passed; formatting left 174 files unchanged.
  Locked sync passed earlier (`shacl-final-sync.log`); dependencies did not change.
- The final captured native publication exited **0** and committed **64 chunks,
  319 entities, 196 facts, 357 mentions and 197 assertions**. Strict SHACL reports
  **zero violations** and `validate_graph_state` passes. Vectors match those rows:
  chunk/entity/fact counts **64/319/196**; fact FTS count **196**.
- Independent read-only comparison confirms that **all 197 raw assertion JSON
  records equal the original captured records**, including multiplicities. All
  197 assertions have live bindings to 196 distinct canonical facts. **64**
  canonical fact descriptions differ from their selected raw representative;
  **65** joined assertion descriptions differ because a fact has shared support.
  The raw `429s` mention remains unchanged while its graph label becomes
  `HTTP status code representing rate limiting.`
- Evidence: `shacl-captured-native-grounded-publication.log`,
  `shacl-captured-native-publication/{result,evidence-check}.json`, and
  `shacl-captured-evidence-check.log`, under `outputs/graph-spec-review/`.
  Raw assertion digest:
  `b029278ef484b0cbca8ff4875b0ddd47978303d86d40dfb799ccbc2a81e70cd0`.
- The native replay emitted an upstream aiohttp/LiteLLM `_wait_for_close` shutdown
  warning after the committed result, then exited 0. No cleanup exception was
  suppressed or workaround added. This is not transport/lifecycle acceptance.
- README now describes read-only planning, authoritative `propose_repairs()` and
  declarative publication; the older evaluation checkpoint is explicitly historical.
  Current source/scripts contain neither retired action spelling nor the retired
  worker-tool name; retired actions occur only in negative tests/history.

This verifies the captured failure through real model inference and native final
targets. The upstream producer probe covers one recorded passage, not full-corpus
recall/EM/F1. No complete fresh-corpus GEPA run, 64+64/128 timing comparison,
<=1-GiB scaling acceptance or U3–U6 completion is claimed. The requested C1–C4 and
P0–P5 plans remain open; no acceleration dependency or alternate writer was added.
Older saved healer contracts require a fresh optimization rather than migration.
