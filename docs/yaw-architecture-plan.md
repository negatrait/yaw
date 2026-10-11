# YAW Architecture Plan (Reference)

**Document Version:** 2.0.0 (Consolidated Architecture Reference)  
**Governing SSOT:** [`FOUNDATIONAL.md`](file:///home/harri/yaw/FOUNDATIONAL.md)  
**Active Roadmap:** [`docs/epics-alignment-roadmap.md`](file:///home/harri/yaw/docs/epics-alignment-roadmap.md)  
**Target:** `yaw` Core Engine (`src/yaw/`)  

---

> [!IMPORTANT]
> [`FOUNDATIONAL.md`](file:///home/harri/yaw/FOUNDATIONAL.md) is the primary authoritative Single Source of Truth (SSOT) for all product requirements, invariants, database contracts, and algorithms. [`docs/epics-alignment-roadmap.md`](file:///home/harri/yaw/docs/epics-alignment-roadmap.md) is the authoritative implementation roadmap. This document provides an aligned architectural synthesis and must remain strictly consistent with both.

---

## 1. Architectural Summary & System Boundaries

YAW is an autonomous, local-first GraphRAG engine combining **Cocoindex**, **spaCy + spacy-llm**, **SQLite** (`fts5` + `sqlite-vec`), and **SciPy** (Personalized PageRank through the ontology layer) served via **FastMCP**.

### The 4-Tier Semantic Stack
1. **Tier 1: SQLite In-Process Quad Substrate (`.cache/yaw.db`)**
   - Single-stage continuous ingestion DAG storing byte offsets and contentless FTS5 indexes.
   - Dynamic views ([`unified_edges`](file:///home/harri/yaw/src/yaw/storage/db.py#L85-L168), [`unified_nodes`](file:///home/harri/yaw/src/yaw/storage/db.py#L170-L200)) mapping 5-column named quads `(src_id, dst_id, rel_type, weight, source_chunk_id)`.
   - Engine-owned [`aliases`](file:///home/harri/yaw/src/yaw/storage/db.py#L44-L60) table providing reversible canonicalization without mutating raw extractions.
2. **Tier 2: OWL 2 RL Axiomatic Invariants ([`docs/adr/ADR-008.md`](file:///home/harri/yaw/docs/adr/ADR-008.md))**
   - 14 Canonical Base Predicates across 4 categories (Subsumption, Dialectical, Process, Grounding).
   - 9 Universal Root Namespaces ([`docs/adr/ADR-005.md`](file:///home/harri/yaw/docs/adr/ADR-005.md)).
   - B-Tree hypernym induction and cluster splitting ([`docs/adr/ADR-002.md`](file:///home/harri/yaw/docs/adr/ADR-002.md)).
3. **Tier 3: SciPy Power Personalized PageRank ([`docs/power-pagerank.md`](file:///home/harri/yaw/docs/power-pagerank.md), [`docs/adr/ADR-010.md`](file:///home/harri/yaw/docs/adr/ADR-010.md))**
   - Sparse CSR matrix power iteration with hub dampening ($\gamma=0.5$).
   - Candidate prior distributions $P(e \mid a)$ in seed personalization and topological disambiguation.
   - Taxonomic conductance asymmetry ($1.0$ downward broadcast, $0.2$ upward damping).
   - In-memory Bayesian evidence fusion for chunk candidate rescoring.
4. **Tier 4: SHACL Closed-World Validation & Contradiction Cache ([`docs/rdf-owl-shacl.md`](file:///home/harri/yaw/docs/rdf-owl-shacl.md), [`docs/adr/ADR-011.md`](file:///home/harri/yaw/docs/adr/ADR-011.md))**
   - Lean compile-time validation (`validate_wiki_graph_fast` < 200 ms) via relational pre-closure without heavy DL tableaux reasoners.
   - Transactional persistence into `compiled_violations` SQLite table.
   - Instantaneous $\mathcal{O}(1)$ `lint_wiki()` reads (< 5 ms) surfacing dialectical contradictions (`causes` vs `prevents`, `supports` vs `refutes`) and contested alias merges.
