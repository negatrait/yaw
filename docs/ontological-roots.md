# Corpus organization and supported hierarchy

**Status:** Selected compiler stage; implement and verify in [Epic 7.8](epic7.8-open-ie-ontology-alignment.md).\
**Authority:** [FOUNDATIONAL](../FOUNDATIONAL.md), [ADR-014](adr/ADR-014.md).\
**Orchestrator:** [Continuous compilation](knowledge-compilation.md). This guide owns grouping; the [PPR guide](power-pagerank.md) owns traversal weights.

## Selected algorithm

Implement this procedure inside `src/yaw/indexer/compiler.py`; do not add a second clustering service or runtime algorithm selector.

1. Collect active entity/concept description records with source support and compatible space-S vectors. Exclude properties, literals, already generated group labels and unsupported descriptions. Empty input is a legitimate empty organization result; a missing/invalid required vector is an explicit stage error.
2. Sort records by stable ID and L2-normalize their vectors. With zero records return empty organization; with one record retain it as a supported root without calling `linkage`. When the supported root count is already at or below `target_roots`, retain those roots and continue page synthesis without inventing a merge. Otherwise run SciPy hierarchical `linkage` with **`method="ward"`** on those Euclidean observations. Pin the SciPy version and record input/order/configuration identity. Reject zero/nonfinite vectors rather than manufacture values. The resulting dendrogram is a candidate organization, not an ontology.
3. Walk the linkage rows in their returned bottom-up order, using the sorted input IDs and pinned SciPy version to resolve the candidate sequence. Count only accepted merges toward the root target. A rejected candidate and ancestors requiring it are ineligible; do not generate a second clustering to rescue the target. The shared synthesis task receives the members and supporting claims and proposes an umbrella label, definition and evidence-linked membership. Validate its structured references and semantic kinds. Record accepted organizational proposals or explicit rejection/insufficient-support diagnostics; acceptance here does not admit a logical axiom.
4. An internal group can be retained only when its required child groups/members remain supported. Rejected groups do not force a broader merge. Stop merging when the selected root target is met or no supported candidate remains. Enforce the configured positive depth/member/work budgets; distinguish an unmet root target from truncated/incomplete processing.
5. Persist organization and its full membership/input fingerprint as derived records. Source edits or membership changes invalidate the group and dependent synthesis. Exact compatible generated proposals are memoized; deterministic clustering is recomputed from the current manifest.

The starting `target_roots` is **9**, expressing the owner's fewer-than-ten preference. Fewer source topics may yield fewer groups; unrelated topics may require more. Report actual roots, rejected candidates and `target_met` instead of inventing common semantics to satisfy the count. This target differs from the built-in nine-folder organization profile.

SciPy Ward requires Euclidean geometry and uses quadratic memory; it does not promise balanced branches, stable partitions after edits or improved retrieval. These are library constraints and evaluation questions, not reasons to introduce a hidden alternative. [SciPy linkage](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html).

## Semantic and retrieval boundaries

Persist generated grouping as the built-in profile's **organizational membership** relation, with generated status, member evidence and model/input versions. Define that relation once in the engine RDF resources and project it generically. Do not emit `rdfs:subClassOf`, `rdf:type` or `skos:broader` merely because Ward placed records together.

Imported/source-supported class or thesaurus hierarchies retain their own meaning and are admitted through the semantic compiler. Individuals, classes and concepts cannot be interchanged; a label collision is not identity. YAW's class/concept non-punning policy remains.

The production retrieval graph includes supported organizational links with their status. Whether they improve results is measured through an **offline evaluation ablation**; there is no automatic hierarchy-on/off recovery path. Failures remain visible. Proposed hierarchy and inferred facts must be distinguishable in returned paths.

## Acceptance and cost evidence

E78-09 and E78-15 test zero/one/already-within-target inputs, a real accepted merge, fixed-input reuse, stable ordering, rejected group preservation, an unmet root target, evidence removal, input membership invalidation and no automatic class axioms. The live-model pilot measures label/member support and false conceptual bridges. Record node count, vector dimensions, clustering memory/time and synthesis calls.

This capability must work before Epics 8.x. Epic 8.2 evaluates its benefit; it does not implement the missing compiler. The historical B-tree/degree-10 bisection and automatic hypernym-subclass recipe in [ADR-002](adr/ADR-002.md) is superseded.
