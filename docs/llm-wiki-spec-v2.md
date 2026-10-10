# Architecture Specification v2.0 (SUPERSEDED)

> [!WARNING]
> **SUPERSEDED SPECIFICATION:** This document is an early 2.0.0 prototype specification. The authoritative Single Source of Truth (SSOT) for all architecture, system invariants, database schemas, and retrieval algorithms is [`FOUNDATIONAL.md`](file:///home/harri/yaw/FOUNDATIONAL.md) and [`docs/epics-alignment-roadmap.md`](file:///home/harri/yaw/docs/epics-alignment-roadmap.md). Do not implement or reference patterns or schemas from this document.

**Document Version:** 2.0.0 (Historical / Superseded)  
**Governing Canon:** [`FOUNDATIONAL.md §0, §1, §2`](file:///home/harri/yaw/FOUNDATIONAL.md)  
**Authoritative Replacement:** [`FOUNDATIONAL.md`](file:///home/harri/yaw/FOUNDATIONAL.md)  

---

## Architectural Evolution

The early 2.0.0 prototype outlined preliminary invariants and table definitions that have since been standardized and hardened:

| Legacy Prototype Feature | Authoritative Production Architecture | Key Reference |
| :--- | :--- | :--- |
| 5 Invariants | **7 System Invariants** (added Invariant 6: Real Testing, Invariant 7: SSOT) | [`FOUNDATIONAL.md §2`](file:///home/harri/yaw/FOUNDATIONAL.md#2-the-seven-invariants-of-yaw) |
| Multiple DB files (`vault_knowledge.db`) | **Single in-process database (`.cache/vectors.db`)** | [`FOUNDATIONAL.md §2 Invariant 3`](file:///home/harri/yaw/FOUNDATIONAL.md) |
| `raw_documents` and `raw_chunks` tables | **`note_meta` and `note_chunks` with byte pointers** | [`FOUNDATIONAL.md §5.2`](file:///home/harri/yaw/FOUNDATIONAL.md) |
| Hardcoded `is_a` / `part_of` relations | **14 Canonical Base Predicates & W3C Semantic Alignment** | [`docs/adr/ADR-008.md`](file:///home/harri/yaw/docs/adr/ADR-008.md) |
| Singular `concept:` namespace | **9 Plural Root Namespaces (`concepts:`, etc.)** | [`docs/adr/ADR-005.md`](file:///home/harri/yaw/docs/adr/ADR-005.md) |
| Static graph views | **Reversible `unified_edges` named quads & `aliases`** | [`docs/adr/ADR-003.md`](file:///home/harri/yaw/docs/adr/ADR-003.md), [`FOUNDATIONAL.md §5.2`](file:///home/harri/yaw/FOUNDATIONAL.md) |