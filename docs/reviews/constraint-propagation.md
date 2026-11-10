# Constraint propagation CP1

2026-11-10. Baseline `dev`, `7bd5ea9`; preserve the uncommitted EP1 provenance
repair. Owning contract: [native graph publication](../native-cocoindex-graph.md#constraint-propagation-during-adjudication).

Plan: one temporary union-find with separation adjacency, shared by merge preview
and final application. Consume judge batches lazily; omit already-collapsed
pairs, bypass labels for established separation, retain real votes and the
existing scoring functions. Normalize decision order to candidate order before
preview/application. No new dependency, schema, persistent cache or voting rule.

The broader proposal to reuse another pair's relation wholesale is excluded:
it can suppress a distinct judge probability and alter confidence. Existing
geometric opinion pooling remains active, but the current judge signatures and
parsers do not supply `llm_prob`; the planner accepts supplied values. CP1 must
preserve those values, not assume a functioning logprob collector.

User-authorized extension: collect logprobs from existing judge calls without
additional generation. Request `logprobs=True`; use DSPy's per-prediction
metadata, never global LM history. Reconstruct UTF-8 byte spans and measure only
Boolean value tokens; sum token logprobs within a value, then geometrically pool
the fields. Pool measured stage votes along the actual cascade path using
`pool_confidences`. Missing metadata/value measurements yield None; malformed
metadata fails. No default confidence, self-reported probability, alternative
scorer, extra sample or inference retry loop. Remove raw token arrays from the
prediction after scoring. Preserve numeric/None values across the memo boundary;
`judge_semantics` memo version becomes 4.

Definition of done: real SQLite constraint/merge tests, shuffled-output and
contradictory-separation cases, native loopback verification of avoided calls,
fixed-evidence 64+64 versus 128 canonical equality including scores, mandatory
quality gates and unguided critic review. No live-model accuracy or full-corpus
speedup claim follows from loopback tests.

Implementation and evidence (commit pending):

- Added plan-local union-find/separation adjacency, lazy evidence-batch consumption,
  stable candidate-order application, and label-stage routing for established
  separation. Accepted related/broader/differentFrom decisions protect later
  merges. No persistent decision cache or second writer.
- Retired the homograph/global-mean-altLabel differentFrom veto: removing other
  pairs otherwise changes its acceptance boundary. Keep explicit final-stage
  distinctness for admitted pairs and its pooled confidence; traversal excludes
  the relation by kind rather than forcing its confidence to zero.
- GPT-6.1 Terra is the user-selected critic, superseding the AGENTS.md flash
  model preference. Its findings caught lost newly judged separations, the
  batch-dependent veto, discarded distinctness confidence, stale documentation,
  and unchecked metadata Boolean values. Corrections and regressions added;
  final audit pending.
- `focused.xml`: 15 passed. `ordering.xml`: 8 passed. `final-affected.xml`:
  32 passed across planner, merge guards, confidence and native publication.
  These targeted runs were repeated as critic corrections landed; final metadata
  matching is covered in `votes-final.xml`.
- Native 128-chunk alias fixture compares all 16 logical tables for 64+64 versus
  128, under normal and reversed judge-output order. It verifies positive skipped
  pair counts, retained 128 assertions and actual logprob requests. Unit tests
  additionally permute conflicting separation/merge decisions and retain numeric
  pooled weights, including differentFrom excluded from topology.
- `live_judge.json`: real resident Gemma 4, GBNF enabled, one stage call, measured
  confidence 0.9295082427906514, 3.73 seconds. No duplicate generation. The first
  preflight failed because no model was resident; loading the existing external
  model enabled this check. No daemon/optimizer restart or source DB mutation.
- Broad run `full-suite.xml`: 998 passed, 8 failed; collected before final critic
  corrections. Two failures were nonresident live-service preflights, subsequently
  `2 passed` in `live-gates.xml`. Six were old in-process `pair_id` assumptions;
  the final affected planner/merge runs above pass. The 20-trial fixed-evidence
  gate completed in that run (`outputs/benchmarks/g5_g6_acceptance.json`), but
  this is not a claim of a clean final-source full-suite run. The earlier EP1
  full-suite process was interrupted and its partial log is not a passing gate.
- Static type, Ruff, format and diff checks passed; repeat after the requested
  extraction-confidence extension. No checkpoint is marked complete yet.

Next authorized unit: [EX1 extraction confidence](extraction-confidence.md),
followed by final-source gates, Terra audit and conventional commit(s).

Final integration discovery: retaining explicit negative edges exposed a stale
decision boundary on a late direct fact. The incremental arm retains an earlier
`owl:differentFrom` for Algebra / Graph theory, while the complete arm derives
`skos:related` under W1/W2. `final-graph.xml` records four failing general batch
checks and 47 passing targeted cases. The fixture now excludes user evidence
from stage detection and uses stable endpoint names in synthetic reasoning.

An attempted unconditional deletion of negative edges made the three direct
batch comparisons pass, but Terra correctly rejected it: a fact does not disprove
distinctness, and deleting the edge loses the measured judgment's history. That
change was removed. The subsequent `final-graph-2.txt` run used the rejected
change and was stopped; it is not acceptance evidence. No parity checkpoint is
complete.

The missing boundary is U3's existing evidence-keyed decision receipt contract:
edges currently serve as adjudication receipts. The user's PROV-O suggestion is
the appropriate representation: a judgment is a `prov:Entity` generated by a
judging `prov:Activity`, derived from versioned source evidence. Retain measured
votes and earlier judgment entities; change the active graph projection only
after the new decision validates. PROV-O supplies semantics, while the existing
typed CocoIndex publisher must supply durable, indexed storage and atomicity.
Do not insert model judgments into author-originated `fact_assertions` or scored
NER entities. A cache hit must not become another independent Bayesian vote.
This receipt boundary remains unimplemented; source withdrawal/re-addition,
non-admitted pairs and failed-judge rollback must be included in its validation.

## CP2: PROV-O judgment receipts

User steering: "But couldn't this be included as a PROV-O entry?" Yes: implement
one typed `synonym_judgments` metadata table through the existing CocoIndex
writer, export judgments as `prov:Entity`, judging runs as `prov:Activity`, and
link source revisions with `prov:wasDerivedFrom` / `prov:used`. Preserve both
individual measurements and the pooled result with `prov:value`; no custom
vocabulary, RDF runtime, extra model call or second writer.

Use stable endpoint IDs, versioned source occurrences, per-pair exposed evidence
and relevant program/policy identities as the content key. Packet-local IDs and
derived stage-routing flags do not identify source evidence. Keep historical
results, including negative judgments. Index endpoints; lookup a result by its
exact evidence key, without cloning the history into staging. The previous
projection is superseded only after a replacement judgment validates. Publish
history and projection together in the existing atomic transaction. A memo or
receipt replay adds no independent confidence vote. Keep this metadata outside
the scored NER graph and author-originated source assertions.

History uses delta target declarations, including empty graph snapshots. Its
`active` marker selects the current receipt for a pair; reuse still requires
validating the current source/evidence key. Original decisions and measurements
remain retained when that marker changes. CP2 does not establish complete U3
reverse candidate coverage or finish the broader publication cutover.

Acceptance pending: native late-bridge add/withdraw/re-add, admission misses,
unchanged receipt replay without inference, failing replacement rollback,
64+64/128 canonical equality, confidence/provenance regressions, Terra audit and
mandatory quality gates. No CP2 checkpoint is complete yet.

Prototype outcome: removed, not shipped. Terra found two blocking gaps: effective
stage-routing proofs must participate in reuse validation, and replacing one
receipt must not delete an edge supported by another active contribution. A
per-pair receipt alone did not solve contribution ownership. The initial
`prov-receipts.txt` probe also caught non-JSON SQLite Row support records; it is
not acceptance evidence. All CP2 code and schema changes were removed, retaining
the verified EP1/CP1/EX1 implementation. PROV-O remains the selected representation
for the required source-grounded receipt/contribution boundary; no new dependency
or second RDF engine is needed. The standard is
[W3C PROV-O](https://www.w3.org/TR/prov-o/).
