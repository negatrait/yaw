# Architecture Decision Records (ADRs)

This directory documents the core architectural decisions governing YAW (Yet Another Wiki), an LLM-Wiki indexer and Ontology GraphRAG engine. Each record captures the context, architectural decision, governing canon, and trade-offs of critical technical choices.

## Architectural Decision Ledger

| ADR ID | Title | Status | Scope | File Link |
| :--- | :--- | :--- | :--- | :--- |
| **ADR-001** | Defensive Single-Pass Triples Fallback | Accepted | NLP / Open IE Extraction | [`ADR-001`](file:///home/harri/yaw/docs/adr/ADR-001.md) |
| **ADR-002** | B-Tree Hypernym Induction for Saturated Hubs | Accepted | Ontology / Graph Topology | [`ADR-002`](file:///home/harri/yaw/docs/adr/ADR-002.md) |
| **ADR-003** | Two-Tier Entity Disambiguation & Canonicalization Convergence Protocol | Accepted (Tier 2 amended by ADR-012) | Ingestion / Entity Resolution | [`ADR-003`](file:///home/harri/yaw/docs/adr/ADR-003.md) |
| **ADR-004** | Direct Vector Storage & Cosine Projection for Canopy Indexing | Accepted (amended by ADR-012) | Storage / Vector Indexing | [`ADR-004`](file:///home/harri/yaw/docs/adr/ADR-004.md) |
| **ADR-005** | 9-Root Universal Hierarchy & Dynamic View Namespace Inheritance | Accepted (amended by ADR-012) | Ontology / SQLite Views | [`ADR-005`](file:///home/harri/yaw/docs/adr/ADR-005.md) |
| **ADR-006** | spaCy-Native Token Lemmatization for Predicate Normalization | Accepted | NLP / Predicate Normalization | [`ADR-006`](file:///home/harri/yaw/docs/adr/ADR-006.md) |
| **ADR-007** | In-Process Loopback HTTP Network Isolation for Test Suites | Accepted | Testing / Network Isolation | [`ADR-007`](file:///home/harri/yaw/docs/adr/ADR-007.md) |
| **ADR-008** | W3C Semantic Alignment (SKOS/OWL/PROV-O Mappings, RDF Quad View Projections, and Seeded Predicates) | Accepted (amended by ADR-012) | Ontology / Semantic Web & NLP | [`ADR-008`](file:///home/harri/yaw/docs/adr/ADR-008.md) |
| **ADR-009** | Pluggable Ontology Substrates, Lossless Ingestion Matrix, and Non-Punning Dual-Epistemic Architecture | Accepted | Ontology / GraphRAG & Storage | [`ADR-009`](file:///home/harri/yaw/docs/adr/ADR-009.md) |
| **ADR-010** | Probabilistic Candidate Priors and Topological Disambiguation in PPR GraphRAG | Accepted (§2 amended by ADR-012) | Retrieval / Ontology & Entity Disambiguation | [`ADR-010`](file:///home/harri/yaw/docs/adr/ADR-010.md) |
| **ADR-011** | Compile-Time Closed-World SHACL Validation and O(1) Contradiction Caching | Accepted (§4 amended by ADR-012) | Compilation / SHACL Validation & MCP Server | [`ADR-011`](file:///home/harri/yaw/docs/adr/ADR-011.md) |
| **ADR-012** | Representation-Space Separation, Corroborated Identity Resolution, and Query-to-Fact Seeding | Accepted | Ingestion / Entity Resolution & Retrieval | [`ADR-012`](file:///home/harri/yaw/docs/adr/ADR-012.md) |

---

## Governance & Invariants

All architecture decisions must align strictly with the Seven Invariants defined in [`FOUNDATIONAL.md §2`](file:///home/harri/yaw/FOUNDATIONAL.md) and the operational roadmap in [`docs/epics-alignment-roadmap.md`](file:///home/harri/yaw/docs/epics-alignment-roadmap.md).
Modifications to accepted ADRs require formal review, empirical test validation, and cross-cutting architectural consistency checks.
