# Architectural Evaluation: Neo4j vs In-Process SQLite + SciPy (SUPERSEDED)

> [!WARNING]
> **SUPERSEDED SPECIFICATION:** This document explores an external Neo4j / property graph database integration. It is superseded by Invariant 3 in [`FOUNDATIONAL.md`](file:///home/harri/yaw/FOUNDATIONAL.md) and [`docs/adr/ADR-008.md`](file:///home/harri/yaw/docs/adr/ADR-008.md). YAW strictly uses a single in-process SQLite database with SciPy sparse matrix PageRank and in-process W3C RDF/SHACL validation.

**Status:** Architectural Decision Record / Superseded Evaluation  
**Governing Canon:** [`FOUNDATIONAL.md §2 Invariant 3`](file:///home/harri/yaw/FOUNDATIONAL.md), [`docs/adr/ADR-008.md`](file:///home/harri/yaw/docs/adr/ADR-008.md)  
**Authoritative Implementation:** In-process SQLite quad substrate in [`src/yaw/storage/db.py`](file:///home/harri/yaw/src/yaw/storage/db.py) and SciPy PPR in [`src/yaw/retrieval/retriever.py`](file:///home/harri/yaw/src/yaw/retrieval/retriever.py).

---

## Architectural Decision Summary

### The External Graph Database Anti-Pattern
Early designs considered delegating knowledge graph storage and Personalized PageRank traversal to an external property graph database (e.g. Neo4j, Memgraph). This was rejected for the following decisive reasons:

1. **Operational Complexity:** Requiring Docker or an external Java/JVM process breaks local-first, zero-setup developer experience.
2. **Network Overhead & IPC Latency:** Serializing multi-hop graph queries over Bolt/HTTP sockets adds tens of milliseconds of latency compared to in-memory SQLite and SciPy CSR matrices ($< 2\text{ms}$).
3. **Cagle's Law of Relational Graph Superiority:** For knowledge graphs under 30 tables and fewer than $10^6$ edges, relational B-tree index lookups with SQL recursive CTEs and sparse matrix vector multiplications outperform standalone graph database engines.
4. **Substrate Unification:** SQLite unifies full-text search (`fts5`), vector similarity (`sqlite-vec`), relational leaf extractions, and W3C named quads within a single in-process transaction boundary (`WAL` mode).
