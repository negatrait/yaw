# Pluggable ontology integration

**Status:** Reconciled target specification, 2026-10-12; [Epic 7.8](epic7.8-open-ie-ontology-alignment.md) implements it.\
**Contract:** [FOUNDATIONAL](../FOUNDATIONAL.md); [selected implementation](adr/ADR-014.md); [semantic guide](rdf-owl-shacl.md). The [single compiler](knowledge-compilation.md) owns scheduling/publication.

The former embedded implementation plan, predicate tables and copied SQL/Python are retired. This document defines integration behavior; executable schemas and resource definitions each have their own owner.

## Load, compile and publish

1. Parse the selected resource and identify its version, base/prefix bindings, kinds, labels, axioms, lexical annotations and shapes.
2. Validate resource syntax and YAW modeling-policy constraints. Report unsupported reasoning constructs distinctly from invalid RDF or forbidden class/concept mixing.
3. Persist/retain authoritative semantic data and build consumer indexes transactionally. Preserve required multiple parents, label languages, inverse orientation and supported property characteristics.
4. Supply the complete versioned resource snapshot to the shared CocoIndex compiler manifest. Consumers cannot mix partially updated resource versions.
5. The same compiler recomputes affected alignment, enrichment, validation, comparison/synthesis dependencies and retrieval projections, then publishes through the one SQLite target adapter. Raw extraction and compatible content embeddings remain intact. There is no separate ontology-triggered compiler or live validator.

Idempotent reloading of the same version does not duplicate resources or FTS/vector rows. Removing a resource revokes dependent interpretations and refreshes projections, while unknown/source relations remain represented. No API may implement removal by deleting unrelated source assertions.

## Four view configurations

The legacy `open_ie` setting selects a view's vocabulary scope. Extraction remains open in every configuration.

| Resource selection | Open view | Scoped view |
| --- | --- | --- |
| Built-in/default profile | Source-grounded known and unfamiliar relations | Relations admitted by the selected profile |
| External profile(s) | External alignments plus unfamiliar source-grounded relations | Relations admitted by those selected profiles |

Membership is derived from resource identity and accepted alignments, never a literal SQL/Python relation list. A bare local name does not identify a property across namespaces. An alias existing somewhere in the database is not sufficient evidence of membership in the active profile.

An empty selected optional inventory stays empty. In an open view, unknown relations remain available; in a scoped view, no domain property is admitted merely through a hidden fallback. Structural evidence/navigation edges remain governed by the declared engine profile and traversal configuration.

Ontology/SpanRuler/NER hits are alignment hints, not an upstream filter on discovery. A restricted extractor is not part of this implementation. Both settings feed the same extraction/compilation path and differ only in the selected output vocabulary.

## Registry and alignment

Compile runtime lexical indexes from the resources. Keep original qualified identifiers and semantic kinds. Match phrase/context to candidates; accepted predicate alignment records target property, argument orientation, evidence and version. Preserve original surfaces and qualifiers.

Generate source-grounded open identifiers when alignment is unresolved. Their identity is not defined solely by a normalized string: equal phrases may have distinct senses. Candidate identity, concept mapping, class membership and property equivalence are separate decisions.

The NLP-facing projection must expose the information its consumers need; an incomplete registry must not cause fallback to a second hardcoded domain vocabulary.

## Incremental dependency boundaries

| Change | Affected work |
| --- | --- |
| Retrieval scope only | View selection and traversal snapshot |
| Ontology/lexical resource | Dependent alignment, inference, projection, validation and relevant resource vectors |
| Shapes/validation profile | Relevant validation cache and reported completeness |
| Source revision | Occurrences, extraction and source-dependent derived records |
| Extraction model/task/rendered prompt | Affected extraction cache and derived records |
| Fact renderer/retrieval encoder | Relevant fact/chunk vector indexes; no unnecessary source rewrite |

A resource update may require work; it is not a universal sub-five-millisecond operation. The compiler's generation/dependency contract governs readers: only current completed scopes contribute compiled knowledge, pending scopes are explicit, and failed scopes return their errors. No stale generation is substituted as current.

## Acceptance and migration

Use the [Epic 7.8 acceptance scenarios](USER_STORIES.md), especially the unfamiliar-property slice and resource removal. Validate real RDF, real SQLite and actual consumers. Tests must show changes in a resource affect behavior without Python relation-table edits.

Re-extract records whose original predicates or qualifiers were lost; do not infer their original wording from normalized values. Reassess old heuristic identities and rebuild dependent projections with a migration/audit record. Implementation and platform status belong in [TEST_READY](../TEST_READY.md).
