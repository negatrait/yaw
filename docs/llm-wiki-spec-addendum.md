# Architectural Addendum: Medallion Architecture (SUPERSEDED)

> [!WARNING]
> **SUPERSEDED SPECIFICATION:** This document outlines an early two-stage Medallion architecture (`Bronze -> Silver -> Gold`). It is completely superseded by [`FOUNDATIONAL.md`](file:///home/harri/yaw/FOUNDATIONAL.md) (Single-Stage Continuous Ingestion) and [`docs/epics-alignment-roadmap.md`](file:///home/harri/yaw/docs/epics-alignment-roadmap.md). Refer strictly to `FOUNDATIONAL.md` as the authoritative Single Source of Truth.

**Specification Status:** Historical / Superseded  
**Governing Canon:** [`FOUNDATIONAL.md §1, §2`](file:///home/harri/yaw/FOUNDATIONAL.md)  
**Authoritative Replacement:** Single-Stage Continuous Ingestion DAG in [`FOUNDATIONAL.md §4, §5`](file:///home/harri/yaw/FOUNDATIONAL.md)  

---

## Historical Context & Reasons for Supercession

The initial 2.0.0 prototype proposed splitting ingestion into an artificial two-stage Medallion pipeline:
1. **Stage 1 (Bronze $\to$ Silver):** Raw markdown notes were parsed into unlinked leaf triples and stored in staging tables (`staging_triples`).
2. **Stage 2 (Silver $\to$ Gold):** A delayed, heavy batch clustering and LLM synthesis pass was required before graph views could be queried or compiled wiki pages materialized.

### Why This Architecture Was Purged:
1. **High Ingestion Friction:** Agents could not query newly added notes without triggering an expensive, high-latency Stage 2 batch clustering cycle.
2. **Substrate Violation:** Mutating intermediate staging tables created non-deterministic graph states and violated Invariant 1 (unidirectional, immutable extraction substrate).
3. **SSOT Duplication:** Staging triples and leaf nodes duplicated entity identities across multiple intermediate representations, causing entity fragmentation.

### The Production Single-Stage Solution
YAW replaced the Medallion CQRS pipeline with **Single-Stage Continuous Ingestion**:
- Ingestion occurs in a single pass via CocoIndex and `spacy-llm`.
- Entities and predicates are resolved on the fly via the Two-Tier EDC Resolver with Cython `InMemoryLookupKB` candidate priors ([`docs/adr/ADR-003.md`](file:///home/harri/yaw/docs/adr/ADR-003.md), [`docs/adr/ADR-010.md`](file:///home/harri/yaw/docs/adr/ADR-010.md)).
- Dynamic SQLite views ([`unified_nodes`](file:///home/harri/yaw/src/yaw/storage/db.py#L170-L200), [`unified_edges`](file:///home/harri/yaw/src/yaw/storage/db.py#L85-L168)) immediately expose the interconnected graph as W3C-compliant named quads ([`docs/adr/ADR-008.md`](file:///home/harri/yaw/docs/adr/ADR-008.md)).
- SciPy Personalized PageRank executes immediately over the unified views without batch delays.