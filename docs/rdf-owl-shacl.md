# RDF, OWL, SKOS and SHACL in YAW

**Status:** Selected semantic implementation; runtime verification pending [Epic 7.8](epic7.8-open-ie-ontology-alignment.md).\
**Authority:** [FOUNDATIONAL](../FOUNDATIONAL.md), [ADR-014](adr/ADR-014.md).\
**Orchestration:** [Continuous compilation](knowledge-compilation.md).

## Resource and graph ownership

One loader consumes versioned built-in/imported ontologies, lexical frames, shapes and admitted SHACL-AF rules. SQLite projections and the NLP registry are rebuildable indexes. Preserve external IRIs, namespace bindings, multiple parents, label languages and supported property traits. No Python predicate/synonym/polarity inventory or SQL whitelist competes with these resources.

RDF represents statements and identified resources; qualified source claims need their own explicit assertion model. Preserve speaker, modality, polarity, time, conditions and evidence. Merely using a named graph does not define these meanings. [RDF concepts](https://www.w3.org/TR/rdf11-concepts/).

Keep three graph roles separate:

| Role | Contents and admission |
| --- | --- |
| Claim graph | All valid source assertions with qualifications, original evidence, interpretations and status; opposing accounts can coexist |
| Inference scope | Positive, non-modal facts admitted under one explicit compatible attribution/time/condition/authority scope, plus supported resource axioms; conclusions inherit that scope |
| Retrieval projection | Evidence links, candidate associations, logical results and labelled organization; navigation edges are not logical axioms |

Unresolved argument identity, unknown required scope or ambiguous property meaning prevents admission to an unconditional inference scope. Preserve the claim and uncertainty instead. A model comparison cannot change source facts or admit `owl:sameAs`. Authority/scope policy is resource data; no generic newest-document-wins rule exists.

## Semantic kinds and identifiers

Projection distinguishes mentions, individuals, classes, concepts, properties and literals. Instance membership, class subsumption/equivalence, individual identity and SKOS mapping are different operations. Check endpoint kinds; namespace or local-name equality is insufficient. Literal datatype/language/unit claims retain their support.

YAW prohibits one URI being both an OWL class and a SKOS concept; separate identifiers can express their relationship. This is a project policy, not a claim that OWL forbids all punning. Preserve direct thesaurus hierarchy separately from transitive closure. Source descriptions of an invariant/workflow do not become executable shapes/events. [OWL primer](https://www.w3.org/TR/owl2-primer/), [SKOS reference](https://www.w3.org/TR/skos-reference/).

Use FOUNDATION's single RDF identifier constructor and migration contract. Do not introduce another chunk-URI or singular/plural kind scheme in an adapter.

## One enrichment and validation sequence

The existing `src/yaw/ontology/shacl.py` is the adapter owner; `src/yaw/storage/rdf.py` owns projection/IRIs. Replace SQL/Python semantic pre-closure and duplicated inline shapes. Use this sequence for every compiler scope:

1. Build the scoped RDFLib graph from current admitted assertions and supported resource axioms. Keep originals and the full input manifest available for provenance.
2. Call `pyshacl.shacl_rules` with **`inference="both"`**, **`iterate_rules=False`** and the selected shapes/rules graph. pySHACL owns RDFS/OWL-RL expansion and the supplied AF rules; YAW does not implement a parallel reasoner.
3. Compare the expanded RDF graph with the prior round. Repeat step 2 until no triples are added, so rule consequences can feed the next entailment round. One configured positive round/triple/time budget bounds this loop. Limit exhaustion or an engine error fails the scope; no partial closure is published as complete.
4. Run `pyshacl.validate` on the completed graph with **`inference=None`** and **`advanced=False`**. Validation does not run another enrichment pass. Validate the qualified claim graph with its applicable shapes through the same adapter as well.
5. Persist derived triples and validation findings as separate generation-owned outputs. Publish them through the compiler's SQLite transaction, never by mutating source assertions.

This sequence is YAW's selected integration. Verify the pinned library API with the small 7.8.3 adapter fixture; finish the full supported-profile and rule/entailment interaction cases in 7.8.5. Inability to execute the selected sequence blocks its gate rather than selecting another engine. [pySHACL rule expansion](https://github.com/RDFLib/pySHACL#inference-and-rules), [public entry points](https://github.com/RDFLib/pySHACL/blob/v0.40.1/pyshacl/entrypoints.py).

## Supported profile and termination

The required project profile covers subclass/subproperty/type propagation, declared domain/range, inverse/symmetric/transitive properties, admitted same-kind identity and disjointness checks. Test these capabilities individually and together. Advertise the tested profile, not full OWL 2 conformance. [OWL profiles](https://www.w3.org/TR/owl2-profiles/).

AF admission is restricted to resource-declared, monotonic TripleRules and SPARQL CONSTRUCT rules over the finite input vocabulary. No fresh blank nodes, unbounded new-term construction, network SERVICE, JavaScript, side effects or negation-as-failure rules enter this profile. Rule order is deterministic by declared order then rule IRI. A resource loader rejects unsupported executable rules before compilation; it preserves unsupported descriptive axioms as inactive data with diagnostics. Do not pass them through and hope the engine ignores them. [SHACL-AF](https://www.w3.org/TR/shacl-af/#rules).

Empty admitted rules produce no AF enrichment. There is no LLM rule-authoring stage. Candidate learned axioms/shapes remain non-executable proposals until an explicit resource admission/migration supplies tests and a new version. An observed regularity is not automatically a universal rule.

The adapter checks monotonic graph growth, finite supported terms and configured bounds. Unsupported reasoning constructs such as unadmitted property chains, cardinality/class expressions or executable extensions cannot silently contribute derived facts.

Check budgets before and after each library call and between rounds. The time budget detects an overrun; it is not a claim of hard preemption inside a synchronous library call. Rule resources must have stable IRIs for deterministic ordering. Record overrun and engine timings explicitly; do not add another reasoner or silently skip rules to meet a budget.

## Derivation provenance and retraction

Use **scope-level derivation bundles**: input scope/generation, all admitted input assertion IDs, resource/shape/rule-set digests, engine/profile version and resulting derived triple IDs. A graph difference identifies new triples but does not prove which individual rule or minimal premise set produced them. Do not populate `source_rule` with a guessed shape ID.

This conservative dependency set is the selected initial implementation for 100–500 documents. Rebuild the scope when any input changes or is removed; another remaining premise can still support a conclusion after recomputation. Report the scope-level provenance granularity to consumers. Precise minimal proofs are not promised and no second provenance engine is introduced.

## Findings and claim conflict

SHACL checks the supplied graph against declared constraints. A constraint may compare attributed claims under an explicit temporal/conditional/authority policy. Distinguish structural violations, potential source disagreement, supported supersession and logical incompatibility within one inference scope. Conformance is not factual truth; missing data under OWL's open-world semantics is not false. [SHACL specification](https://www.w3.org/TR/shacl/).

The compiler's LLM comparison can propose `potential_conflict` with evidence; it is not labelled a SHACL violation unless a shape actually produced that finding. A newer statement alone cannot supersede another. The [comparison contract](knowledge-compilation.md#3-cross-document-claim-comparison) owns these interpretation states.

Persist findings with scope, focus/path/value, severity, supporting references, resource versions and completeness. `lint_wiki` reads these records; refresh schedules the same compiler. No live-lint validator, unversioned cache, empty-on-error result or universal consistency/constant-time claim is allowed.

`conforms=False` with a valid report is a completed validation containing findings. It does not fail ingestion or suppress opposing claims. Engine exceptions, malformed shapes and incomplete execution are failures. Keep that distinction in component state and tests; YAW must be able to compile and retrieve a corpus that contains real disagreements.

## Acceptance

[E78-01/05/08/15–20](USER_STORIES.md) cover new properties, kinds, entailment/rule interaction, finite closure, conflicting scopes, conservative provenance, support removal and explicit failures. Real RDFLib/pySHACL tests are mandatory. Current source modules/resources are migration inputs, not evidence that this selected sequence already runs.
