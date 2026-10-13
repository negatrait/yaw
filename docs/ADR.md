# Architecture Decision Records

[FOUNDATIONAL](../FOUNDATIONAL.md) owns the current architectural contract. ADRs explain decisions and amendments. [ADR-013](adr/ADR-013.md) reconciles the review; [ADR-014](adr/ADR-014.md) selects the one functioning path required before Epics 8.x. [Epic 7.8](epic7.8-open-ie-ontology-alignment.md) tracks implementation and verification.

An accepted architectural direction does not imply implemented or tested behavior. Historical bodies retained in amended ADRs are explicitly non-normative; ADR-014's exact amendments take precedence over conflicting earlier summaries. Coding agents follow the current epic's execution map rather than reconstructing a plan from historical bodies.

| ADR | Current subject | Status |
| --- | --- | --- |
| [001](adr/ADR-001.md) | Single-pass structured open extraction | Amended by 013 |
| [002](adr/ADR-002.md) | Provenanced hierarchy induction | Amended by 013 |
| [003](adr/ADR-003.md) | Candidate resolution and reversible identity | Amended by 012 and 013 |
| [004](adr/ADR-004.md) | Typed vector storage | Amended by 012 and 013 |
| [005](adr/ADR-005.md) | Default organization and derived views | Amended by 012 and 013 |
| [006](adr/ADR-006.md) | Non-recursive linguistic/resource alignment | Amended by 013 |
| [007](adr/ADR-007.md) | Loopback HTTP isolation for fixture tests | Retained; actual-model evaluation is separate |
| [008](adr/ADR-008.md) | W3C resource and projection alignment | Amended by 012 and 013 |
| [009](adr/ADR-009.md) | Pluggable ontologies and lossless scope selection | Amended by 013 |
| [010](adr/ADR-010.md) | Candidate weights and PPR relevance | Amended by 012 and 013 |
| [011](adr/ADR-011.md) | Versioned compiled validation | Amended by 012 and 013 |
| [012](adr/ADR-012.md) | L/S/R representation separation and query-to-fact seeding | Retained in part; amended by 013 |
| [013](adr/ADR-013.md) | Open discovery, declarative semantics and revisable interpretation | Amended by 014; implementation pending Epic 7.8 |
| [014](adr/ADR-014.md) | One evidence-preserving corpus compiler, zero fallbacks and all-gates handoff to 8.x | Selected implementation; runtime gates pending |

## Governance

Amendments identify affected sections, retained decisions, evidence, migration implications and verification gates. Reconcile FOUNDATION, the ledger and affected guides together. Record actual test outcomes in [TEST_READY](../TEST_READY.md); never label target behavior implemented solely because an ADR is accepted.
