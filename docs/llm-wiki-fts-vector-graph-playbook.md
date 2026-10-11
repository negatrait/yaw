# Autonomous GraphRAG & Emergent Ontology Playbook (SUPERSEDED)

> [!WARNING]
> **SUPERSEDED SPECIFICATION:** This playbook contains prototype code and early 3-phase architectural sketches. The authoritative implementation contracts, SQLite schemas, single-stage ingestion DAG, and FastMCP tool interfaces are defined in [`FOUNDATIONAL.md`](file:///home/harri/yaw/FOUNDATIONAL.md) and [`docs/epics-alignment-roadmap.md`](file:///home/harri/yaw/docs/epics-alignment-roadmap.md).

**Specification Status:** Historical Prototype Reference / Superseded  
**Governing Documents:**  
- [`FOUNDATIONAL.md`](file:///home/harri/yaw/FOUNDATIONAL.md) (The Single Source of Truth)  
- [`docs/epics-alignment-roadmap.md`](file:///home/harri/yaw/docs/epics-alignment-roadmap.md)  
- [`docs/power-pagerank.md`](file:///home/harri/yaw/docs/power-pagerank.md)  
- [`docs/adr/ADR-008.md`](file:///home/harri/yaw/docs/adr/ADR-008.md)  

---

## Executive Summary & Mapping to Production Architecture

The early GraphRAG playbook explored combining full-text search (BM25), dense vector embeddings, and Personalized PageRank. Its mathematical core has been formalized and hardened in production:

| Prototype Playbook Concept | Production Implementation | Governing Reference |
| :--- | :--- | :--- |
| Ad-hoc SQLite schema | Hardened schema in `src/yaw/storage/db.py` with covering indexes and WAL PRAGMAs | [`FOUNDATIONAL.md §5.2`](file:///home/harri/yaw/FOUNDATIONAL.md) |
| Multi-stage batch clustering | Single-Stage continuous ingestion + Two-Tier EDC Resolver with Cython `InMemoryLookupKB` candidate priors | [`docs/adr/ADR-003.md`](file:///home/harri/yaw/docs/adr/ADR-003.md), [`docs/adr/ADR-010.md`](file:///home/harri/yaw/docs/adr/ADR-010.md), [`FOUNDATIONAL.md §5.1`](file:///home/harri/yaw/FOUNDATIONAL.md) |
| Unconstrained predicate extraction | 14 W3C Canonical Base Predicates across 4 categories | [`docs/adr/ADR-008.md`](file:///home/harri/yaw/docs/adr/ADR-008.md) |
| Naive graph traversal | SciPy CSR sparse matrix power iteration with hub dampening ($\gamma=0.5$) | [`docs/power-pagerank.md`](file:///home/harri/yaw/docs/power-pagerank.md), [`src/yaw/retrieval/retriever.py`](file:///home/harri/yaw/src/yaw/retrieval/retriever.py) |
| Disconnected concept lists | Grounded Bridging Subgraphs per result coupling context and disk bytes | [`FOUNDATIONAL.md §5.3`](file:///home/harri/yaw/FOUNDATIONAL.md) |
| Custom HTTP daemon | Official FastMCP streaming server (`daemon.py`) exposing the 4 production tools | [`FOUNDATIONAL.md §5.4`](file:///home/harri/yaw/FOUNDATIONAL.md), [`src/yaw/server/daemon.py`](file:///home/harri/yaw/src/yaw/server/daemon.py) |