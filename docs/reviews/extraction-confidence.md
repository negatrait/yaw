# EX1: NER and triple extraction confidence

2026-11-10. User requested the same measured-logprob pooling and provenance
maintenance for existing NER and triple extraction calls, after CP1. Preserve all
uncommitted EP1/CP1 work on `dev` (`7bd5ea9`).

Plan: share token-span measurement with the judge collector; measure semantic
value tokens from each NER pair/triple, excluding property names and structural
syntax. Request logprobs on the existing calls only. Unknown is None; malformed
or mismatched metadata fails, with no substitute scorer or extra generation.
String-value confidence must be length-normalized before geometric field pooling;
token likelihood is not a calibrated probability of factual truth.

Carry per-entity probabilities in `OpenIERecord` and per-statement probability in
`StructuredTriple`; preserve through normalization, missing-definition completion,
serialization and CocoIndex memoization. Bump extraction memo/fingerprint contract
so old unmeasured memo results cannot masquerade as measured results.

Retain original NER measurements in `entity_mentions.llm_prob` and memoized
`chunk_inference.extraction_json`; retain statement measurements in raw assertion
JSON. The single new mention column requires source reindexing for old databases;
schema validation rejects them explicitly, without deleting or migrating data.
Project measurements into passage `edge_sources` and statement `fact_sources`.
Assertion identity excludes score metadata; updating a vote updates its original
observation instead of adding evidence.

Semantic similarity is also a vote, even when model logprobs are unavailable.
Measure source-sense/context cosine using retained vectors, with no new embedding
calls. Statements pool subject/context and object/context cosines; literals have
only a subject resource role. Pool the semantic measurement with the optional
model vote using the unchanged core geometric operator. Source vectors remain
immutable through canonical renaming and merges. Incremental staging now copies
the referenced chunk vectors alongside their provenance.

Keep the best complete observation per canonical owner/chunk, with stable
origin-ID tie ordering; never combine components taken from different observations.
Aggregate surviving unique chunks with existing Bayesian Noisy-OR. Recompute after
merges and source withdrawals so stale probabilities disappear. Missing logprobs
remain None; required vectors must be present and finite. Edge weights stay NOT
NULL; absence of model metadata does not exclude an edge from scored traversal.
`owl:differentFrom` retains its measured confidence but is excluded by relation
type. Never copy an aggregate posterior into an atomic source observation.

Definition of done: transport tests verify same call count and measured/None
results; real SQLite verifies scoring/provenance after merge, withdrawal and
restoration; final-source 64+64/128 graph parity, quality gates and unguided Terra
audit. No live cutover or optimizer acceptance claim follows from these tests.

Status: implemented; combined acceptance remains incomplete.

- `outputs/extraction-confidence/verified-unit.xml`: 51 passed on the retained
  implementation. Covers measured/missing metadata, source merging and
  withdrawal, invalid high-confidence assertion bindings, provenance restoration,
  normal/reversed 64+64/128 alias propagation, edge direction, retrieval with
  controlled topic geometry and isolated SHACL memo reuse.
- `prompt-audit.xml`: 28 passed, two real-inference nodes deselected in that
  targeted run. Those live nodes passed in the broad suite. Request callbacks
  replace tests' dependence on retained LM response histories.
- `final-suite.xml`: 1,011 passed, 13 failed before the final fixture corrections.
  Corrected history/geometry/fixed-score fixtures pass in the targeted runs.
  General late-evidence parity still fails as documented in CP1. The unchanged
  Windows preflight owned-child termination test also fails in that broad run;
  a focused rerun passed that node. No clean final-source full-suite claim.
- Terra found an invalid-assertion scoring path; scoring now applies the same
  revision, source ownership and endpoint checks as provenance restoration.
  Five adversarial cases pass. Subsequent combined audits found no additional
  actionable issues until the separate CP1 late-evidence projection attempt,
  which was rejected and removed.
- Type, Ruff, format and diff checks pass; no implementation checkpoint is
  marked complete and no commit or live YAW cutover has been made. Existing
  indexes require the documented source rebuild for the mention probability
  column. Historical diagnostic timings do not measure the final projection.
