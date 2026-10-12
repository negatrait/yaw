# YAW Engineering Epics: Architecture Alignment Roadmap

**Document Status:** Authoritative Implementation Roadmap (Audited & Reconciled)  
**Location:** [`docs/epics-alignment-roadmap.md`](file:///home/harri/yaw/docs/epics-alignment-roadmap.md)  
**Governing Documents:**  
- [`FOUNDATIONAL.md`](file:///home/harri/yaw/FOUNDATIONAL.md) (The Single Source of Truth)
- [`docs/ADR.md`](file:///home/harri/yaw/docs/ADR.md) ([`ADR-001`](file:///home/harri/yaw/docs/adr/ADR-001.md) through [`ADR-010`](file:///home/harri/yaw/docs/adr/ADR-010.md))
- [`docs/rdf-owl-shacl.md`](file:///home/harri/yaw/docs/rdf-owl-shacl.md)
- [`docs/alignment-sprint-for-e5s2.md`](file:///home/harri/yaw/docs/alignment-sprint-for-e5s2.md)
- [`docs/USER_STORIES.md`](file:///home/harri/yaw/docs/USER_STORIES.md)
- [`docs/how-to-handle-spacy-llm.md`](file:///home/harri/yaw/docs/how-to-handle-spacy-llm.md)
- [`docs/open-ie-authoritative-source.md`](file:///home/harri/yaw/docs/open-ie-authoritative-source.md)
- [`docs/power-pagerank.md`](file:///home/harri/yaw/docs/power-pagerank.md)
- [`docs/efficient-ppr-algorithms.md`](file:///home/harri/yaw/docs/efficient-ppr-algorithms.md)
- [`docs/karpathy-llm-wiki.md`](file:///home/harri/yaw/docs/karpathy-llm-wiki.md)

---

## 1. Executive Overview & Epic Status Matrix

All Epics are strict mathematical and operational derivatives of [`FOUNDATIONAL.md`](file:///home/harri/yaw/FOUNDATIONAL.md). Zero architectural drift or un-cited speculative patterns are permitted.

| Epic / Phase | Title | Scope & Deliverable | Status | Governing Architecture Docs | Verification Suite |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Epic 1** | **spaCy-LLM Detangling & Canonical Model Registry** | Eliminate monkeypatching, register `@registry.llm_models("yaw.Ollama.v1")`, loopback HTTP fixtures. | **COMPLETE** | [`docs/how-to-handle-spacy-llm.md`](file:///home/harri/yaw/docs/how-to-handle-spacy-llm.md) | `tests/test_spacy_llm_registry.py` (Commit `a9f5d34`) |
| **Epic 2** | **Vault Purity, Substrate Decoupling & Graph Topology** | Purge meta-docs from `raw/`, enforce composite primary keys in SQLite, bidirectional chunk-concept edges (`manifests_in`/`mentions`). | **COMPLETE** | [`FOUNDATIONAL.md §6`](file:///home/harri/yaw/FOUNDATIONAL.md#6-repository-architecture--directory-boundaries) | `tests/test_vault_purity.py` (Commit `001ecae`) |
| **Epic 3** | **CLI, Production Daemon Lifecycle & In-Memory Rescoring** | Stdlib `argparse` subparsers (`index`, `serve`, `query`, `schema`, `shutdown`), lock cleanup, WAL truncation, reciprocal wikilink conductance. | **COMPLETE** | [`docs/power-pagerank.md`](file:///home/harri/yaw/docs/power-pagerank.md), [`FOUNDATIONAL.md §5.4`](file:///home/harri/yaw/FOUNDATIONAL.md) | `tests/test_cli_lifecycle.py` (Commits `8672f0d`, `67c33fa`) |
| **Epic 4** | **Doc-Integrity Extraction & Bridging Subgraph Retrieval** | Invariant E1 (`doc.text == chunk_text`), native `spacy.SpanCat.v3` + `spacy.REL.v1`, post-diffusion chunk selection, token-dense Bridging Subgraph per result. | **COMPLETE** | [`docs/open-ie-authoritative-source.md`](file:///home/harri/yaw/docs/open-ie-authoritative-source.md), [`docs/how-to-handle-spacy-llm.md`](file:///home/harri/yaw/docs/how-to-handle-spacy-llm.md) | `tests/test_leaf_pipeline.py`, `tests/test_retriever.py` (Commits `4f46778`, `22d9fc0`) |
| **Epic 5.0 (Sprint 5A)** | **Two-Tier EDC Convergence & Reversible Aliases** | `aliases` + `canon_verdicts` tables, Two-Tier convergence resolver (`resolve()`), `InMemoryLookupKB` prior distribution $P(e \mid a)$, SQLite graph view joins. | **COMPLETE** | [`docs/ADR.md`](file:///home/harri/yaw/docs/ADR.md), [`FOUNDATIONAL.md §5.1-5.2`](file:///home/harri/yaw/FOUNDATIONAL.md), [`docs/adr/ADR-010.md`](file:///home/harri/yaw/docs/adr/ADR-010.md) | `tests/test_canonicalization.py` |
| **Epic 5.0b (Sprint 5-Align)** | **Forensic Semantic Web Alignment & Quad Preparation** | Standardize 14 canonical predicates ([`ADR-008`](file:///home/harri/yaw/docs/adr/ADR-008.md)), 9 plural roots ([`ADR-005`](file:///home/harri/yaw/docs/adr/ADR-005.md)), quad view contracts. | **COMPLETE** | [`docs/adr/ADR-008.md`](file:///home/harri/yaw/docs/adr/ADR-008.md), [`docs/rdf-owl-shacl.md`](file:///home/harri/yaw/docs/rdf-owl-shacl.md) | Commit `dd25de9` |
| **Epic 5.0c (Remediation Prior)** | **Native spaCy & spacy-llm Remediation Prior** | SSOT base `en_core_web_lg`, `spacy.BatchCache.v1`, purge regex heuristics, 2-tier convergence (0 LLM canonicalization calls), FTS5 stopword stripping, `yaw shutdown`. | **COMPLETE** | [`spacy/SKILL.md`](file:///home/harri/yaw/.agents/skills/spacy/SKILL.md), [`FOUNDATIONAL.md`](file:///home/harri/yaw/FOUNDATIONAL.md) | 208/208 tests pass (Commits `b70d8d4`, `bfdd064`, `67c33fa`) |
| **Epic 5.1 (Sprint 5.1)** | **Substrate Hardening, Quad Views & 14-Predicate Alignment** | Reconcile `aliases.method` CHECK, purge `is_a`, enforce 14 ADR-008 predicates, plural `concepts:`, 5-col `unified_edges` quads, `compiled_relations` tables, subsumption conductance. | **COMPLETE** | [`docs/alignment-sprint-for-e5s2.md`](file:///home/harri/yaw/docs/alignment-sprint-for-e5s2.md), [`docs/adr/ADR-008.md`](file:///home/harri/yaw/docs/adr/ADR-008.md) | `tests/test_storage.py`, `tests/test_canonicalization.py` (Commit `481d5bd`) |
| **Epic 5.2 (Sprint 5B / E5S2)** | **Dynamic Few-Shot Steering & Live Vocabulary Injection** | `@registry.misc("yaw.CanonicalExamples.v1")`, ADR-008 predicate seeding, live canonical vocabulary steering, quad projection compatibility, multi-note audit. | **COMPLETE** | [`docs/adr/ADR-008.md`](file:///home/harri/yaw/docs/adr/ADR-008.md), [`docs/how-to-handle-spacy-llm.md`](file:///home/harri/yaw/docs/how-to-handle-spacy-llm.md), [`FOUNDATIONAL.md §5.1`](file:///home/harri/yaw/FOUNDATIONAL.md) | `tests/test_fewshot_steering.py` (35/35 pass, 261 total) |
| **Epic 6.1 (Sprint 6A)** | **In-Process RDF Triple/Quad Substrate & Serialization** | 5-col SQLite quad view $\to$ `rdflib` graph extraction, CURIE prefixes, Turtle/N-Quads export, zero external graph server, code smell remediation. | **COMPLETE** | [`FOUNDATIONAL.md §1, §5.4`](file:///home/harri/yaw/FOUNDATIONAL.md), [`docs/rdf-owl-shacl.md`](file:///home/harri/yaw/docs/rdf-owl-shacl.md) | `tests/test_rdf_substrate.py` (Commit `462ad1e`) |
| **Epic 6.2 (Sprint 6B)** | **OWL 2 RL & SKOS Taxonomic Subsumption Inference** | In-process OWL 2 RL inference rules, `owl:sameAs` transitive closure via `aliases`, `skos:broader`/`skos:narrower` concept schemes. | **COMPLETE** | [`docs/ADR.md`](file:///home/harri/yaw/docs/ADR.md), [`docs/rdf-owl-shacl.md`](file:///home/harri/yaw/docs/rdf-owl-shacl.md) | `tests/test_owl_ontology.py` (Commit `79ce19d`, 25/25 pass) |
| **Epic 6.3 (Sprint 6C)** | **Closed-World SHACL Wiki Lint & Validation** | Declarative `sh:NodeShape` & `sh:PropertyShape` validation via `pyshacl` in-process, orphan concept & contradiction detection in `lint_wiki`. | **COMPLETE** | [`FOUNDATIONAL.md §5.4`](file:///home/harri/yaw/FOUNDATIONAL.md), [`docs/rdf-owl-shacl.md`](file:///home/harri/yaw/docs/rdf-owl-shacl.md) | `tests/test_shacl_lint.py` (34/34 pass) |
| **Epic 7 (Sprint 7A)** | **Pluggable RDF/OWL Ontology Substrate & Configurable Extraction** | Dynamic OWL/RDFS ingestion (`infer_ontologies`), dynamic catalog tables, dual extraction modes (`open_ie=true|false`), dynamic SHACL whitelist, Ollama offline warning callouts. | **COMPLETE** | [`docs/ADR.md`](file:///home/harri/yaw/docs/ADR.md), [`FOUNDATIONAL.md §5.4`](file:///home/harri/yaw/FOUNDATIONAL.md), [`docs/epic7-pluggable-ontology-integration-spec.md`](file:///home/harri/yaw/docs/epic7-pluggable-ontology-integration-spec.md) | `tests/test_pluggable_ontology.py` (11/11 pass, 429 total) |
| **Epic 7.5** | **Architectural Convergence & Pre-Epic 8 Substrate Realignment** | Cython `InMemoryLookupKB` prior distribution $P(e \mid a)$, Bayesian evidence chunk scoring, clean `aliases.method` CHECK, zero-fallback enforcement. | **COMPLETE** | [`FOUNDATIONAL.md §3, §5.1-5.3`](file:///home/harri/yaw/FOUNDATIONAL.md), [`docs/adr/ADR-010.md`](file:///home/harri/yaw/docs/adr/ADR-010.md) | `tests/test_retriever.py`, `tests/test_canonicalization.py` (98/98 pass) |
| **Epic 7.6 (Sprint 7F)** | **Compile-Time Closed-World SHACL & O(1) Contradiction Caching** | Wire `validate_wiki_graph_fast` into `compile_wiki()`, transactional `compiled_violations` SQLite table, O(1) `lint_wiki()` reads, ontology FTS5 indexing (vector half superseded by ADR-012). | **COMPLETE** | [`FOUNDATIONAL.md §4.3-4.4, §5.5`](file:///home/harri/yaw/FOUNDATIONAL.md), [`docs/adr/ADR-011.md`](file:///home/harri/yaw/docs/adr/ADR-011.md) | `tests/test_compile_shacl_cache.py` (Commit `e64ed3c`) |
| **Epic 7.7** | **Representation-Space Separation, Corroborated Identity & Query-to-Fact Seeding** | One vector space per store (`entity_vec` S, `fact_vec` R), Tier 1 normalized keys, blocked + corroborated Tier 2 with `close_matches`, storage-backed $P(e \mid a)$, query-to-fact PPR seeding. | **SPECIFIED (BLOCKS EPIC 8)** | [`docs/adr/ADR-012.md`](file:///home/harri/yaw/docs/adr/ADR-012.md), [`FOUNDATIONAL.md §3, §5.1-5.3`](file:///home/harri/yaw/FOUNDATIONAL.md) | `tests/test_representation_spaces.py` |
| **Epic 8.1 (Sprint 8A)** | **Compilation Triggers, B-Tree Ward HAC & FastMCP Production Lifecycle** | Ingestion post-sync compilation hook, B-Tree Ward HAC cluster splitting, hypernym induction, full FastMCP production lifecycle (`compile_wiki`, `lint_wiki`, `query_wiki`, `ingest_raw`). | **SPECIFIED (TARGET v1.0)** | [`docs/adr/ADR-002.md`](file:///home/harri/yaw/docs/adr/ADR-002.md), [`docs/adr/ADR-011.md`](file:///home/harri/yaw/docs/adr/ADR-011.md), [`FOUNDATIONAL.md §4`](file:///home/harri/yaw/FOUNDATIONAL.md) | `tests/test_fastmcp_contract.py`, `tests/test_btree_hac.py` |
| **Epic 8.2 (Sprint 8B)** | **2WikiMultiHopQA & MuSiQue Benchmark Evaluation Harness & Materialization** | Ingest official benchmark slices into `sample-llm-wiki/raw/benchmark/`, execute official eval scripts (Exact Match, Token F1, Recall@k, Supporting Fact F1), materialize `vault/wiki/`. | **SPECIFIED (TARGET v1.0)** | [`FOUNDATIONAL.md §1, §6`](file:///home/harri/yaw/FOUNDATIONAL.md), [`docs/karpathy-llm-wiki.md`](file:///home/harri/yaw/docs/karpathy-llm-wiki.md) | `eval/run_benchmark.py`, `tests/test_wiki_materializer.py` |
| **Epic 8.3 (Sprint 8C)** | **Packaging, CI Matrix & GitHub v1.0 Release** | Multi-OS GitHub Actions CI, packaging, documentation audit, release tag `v1.0.0`. | **SPECIFIED (TARGET v1.0)** | [`README.md`](file:///home/harri/yaw/README.md), [`FOUNDATIONAL.md`](file:///home/harri/yaw/FOUNDATIONAL.md) | `tests/test_e2e_launch.py` |

---

## 2. Reconciled Architectural Contradictions

| Topic | Contradiction / Drift Identified | Root Cause | Authoritative SSOT Resolution |
| :--- | :--- | :--- | :--- |
| **Doc Integrity vs Preprocessing** | Stripping markdown code blocks (`946bd44`) broke byte provenance. | Treating spaCy doc as mutable text string instead of annotation substrate. | **Invariant E1:** `doc.text == chunk_text` byte-for-byte. `markdown_structure` annotates code regions into `doc.spans["code"]`; filtering happens strictly in taxonomy compiler. |
| **Open IE Relation Extraction** | Default `spacy.REL.v1` prompt loops over `labels`. Empty list produces empty prompt. | Standard `rel.v1.jinja` requires label definitions for prompt generation. | **`OPEN_IE_REL_TEMPLATE`:** Custom Jinja template extracts freeform semantic predicates without fixed schemas. Lemmatized via native spaCy infix tokenizer ([`docs/adr/ADR-006.md`](file:///home/harri/yaw/docs/adr/ADR-006.md)). |
| **Categorization vs Grounding** | Classifying chunks into `invariants` / `concepts` / `decisions`. | Rylean category mistake: chunks are evidence, not universals. | **Entity-Level Typing:** `spacy.TextCat.v3` is applied exclusively to entity definitions (`Doc._.description`), determining the Universal/Type across the 9 root namespaces ([`docs/adr/ADR-005.md`](file:///home/harri/yaw/docs/adr/ADR-005.md)). |
| **Offline Seeding & Math Continuity** | Creating an ad-hoc "offline lexical fallback" branch in retrieval. | Viewing offline mode as an exception rather than zero vector continuity. | **Unified Vector Equation:** $\mathbf{p}_0 = \beta_1 \mathbf{s}_{\text{concepts}} + \beta_2 \mathbf{s}_{\text{chunks\_vec}} + \beta_3 \mathbf{s}_{\text{chunks\_fts}}$. When $\mathbf{q} = \mathbf{0}$, the vector channels vanish and the alias-prior and FTS5 channels continuously seed PPR ([`FOUNDATIONAL.md §5.3`](file:///home/harri/yaw/FOUNDATIONAL.md), [`ADR-012`](file:///home/harri/yaw/docs/adr/ADR-012.md)). |
| **Candidate Chunk Selection** | Pre-filtering candidate chunks before PPR diffusion ran. | Confusing candidate seed generation with post-diffusion graph ranking. | **Post-Diffusion Selection:** Candidates are extracted from converged distribution $\mathbf{p}^*[c] > 0$, allowing unseeded chunks to surface via graph propagation. |
| **Retrieval Output Architecture** | Disconnected global concept lists separated from chunk evidence. | Outputting macro concept canopy without linking to evidence. | **Bridging Subgraph Per Result:** Each result couples rank, citation, local ontological context (seed concepts $\cup$ chunk-grounded concepts), and evidence slice. $K=0$ provides explicit corpus growth recommendation. |

---

## 3. Epic 1: spaCy-LLM Detangling & Canonical Model Registry
**Status:** **COMPLETE** (Commit `a9f5d34`)  
**Governing Document:** [`docs/how-to-handle-spacy-llm.md`](file:///home/harri/yaw/docs/how-to-handle-spacy-llm.md)  
**Target Test File:** `tests/test_spacy_llm_registry.py`

### Summary of Completed Invariants
1. Registered official `@registry.llm_models("yaw.Ollama.v1")` adhering to spaCy-LLM's public contract.
2. Eliminated all private attribute monkeypatching (`_model`, `_task`).
3. Registered custom Jinja template `yaw.OpenIERelation.v1` for open schema extraction.
4. Bound extraction pipelines to Ollama `/api/chat` with configurable temperature and timeout.
5. Implemented isolated in-process loopback mock handler in [`tests/conftest.py`](file:///home/harri/yaw/tests/conftest.py) ([`docs/adr/ADR-007.md`](file:///home/harri/yaw/docs/adr/ADR-007.md)).

---

## 4. Epic 2: Vault Purity, Substrate Decoupling & Canonical Graph Topology
**Status:** **COMPLETE** (Commit `001ecae`)  
**Governing Document:** [`FOUNDATIONAL.md §6`](file:///home/harri/yaw/FOUNDATIONAL.md#6-repository-architecture--directory-boundaries)  
**Target Test File:** `tests/test_vault_purity.py`

### Summary of Completed Invariants
1. Purged engine specifications from `sample-llm-wiki/raw/`, strictly confining notes to human markdown.
2. Verified all 13 core engine specifications exist under `docs/`.
3. Standardized canonical URI format `^[a-z_]+:[a-z0-9_-]+$` across all extracted entities and concepts.
4. Enforced SQLite composite primary keys `PRIMARY KEY(node_id, source_leaf_id)` in `leaf_nodes` resolving cross-document overwrite bugs (`ARCH-SSOT-001`).
5. Synthesized bidirectional chunk-concept groundings (`manifests_in` / `mentions`).

---

## 5. Epic 3: CLI, Production Daemon Lifecycle & In-Memory Rescoring
**Status:** **COMPLETE** (Commit `8672f0d`)  
**Governing Document:** [`docs/power-pagerank.md`](file:///home/harri/yaw/docs/power-pagerank.md), [`FOUNDATIONAL.md §5.4`](file:///home/harri/yaw/FOUNDATIONAL.md)  
**Target Test File:** `tests/test_cli_lifecycle.py`

### Summary of Completed Invariants
1. Built standard CLI interface in `src/yaw/cli.py` with subparsers: `index`, `serve`, `query`, `schema`.
2. Implemented production daemon lifecycle in `src/yaw/server/daemon.py` with `SIGINT`/`SIGTERM` signal traps, stale PID unlinking, and WAL truncation on shutdown.
3. Implemented in-memory dot product rescoring of candidate chunk vectors against query vectors.
4. Added reciprocal reverse edges for wikilinks ($W_{\text{rev}} = 0.5 \times W_{\text{fwd}}$) in `GraphRAGRetriever`.

---

## 6. Epic 4: Doc-Integrity Extraction & Bridging Subgraph Retrieval
**Status:** **COMPLETE** (Commits `4f46778`, `22d9fc0`)  
**Governing Document:** [`docs/open-ie-authoritative-source.md`](file:///home/harri/yaw/docs/open-ie-authoritative-source.md), [`docs/how-to-handle-spacy-llm.md`](file:///home/harri/yaw/docs/how-to-handle-spacy-llm.md)  
**Target Test Files:** `tests/test_leaf_pipeline.py`, `tests/test_retriever.py`, `tests/test_provenance.py`

### Summary of Completed Invariants
1. Enforced Invariant E1: chunk text is preserved byte-for-byte (`doc.text == chunk_text`). Code blocks are annotated into `doc.spans["code"]` without mutating the underlying text.
2. Extracted open-schema relations using native `spacy.SpanCat.v3` + `spacy.REL.v1` with token-based infix decamelization and spaCy lemmatization ([`docs/adr/ADR-006.md`](file:///home/harri/yaw/docs/adr/ADR-006.md)).
3. Bound entity definitions to $\le 50$ words via `spacy.Summarization.v1`.
4. Extracted candidate chunks strictly from the converged PPR distribution ($\mathbf{p}^*[c] > 0$).
5. Returned token-dense per-result Bridging Subgraphs coupling rank, citation, local graph context, and disk-hydrated evidence slices.

---

## 7. Epic 5: Ontological Convergence & Incremental Maintenance (EDC Protocol)
**Status:** **IN PROGRESS (SPRINTS 5A & 5-ALIGN COMPLETE; SPRINT 5.1 READY; SPRINT 5B SPECIFIED)**  
**Governing Document:** [`docs/ADR.md`](file:///home/harri/yaw/docs/ADR.md), [`docs/adr/ADR-008.md`](file:///home/harri/yaw/docs/adr/ADR-008.md), [`FOUNDATIONAL.md §5.1 step 6, §5.2`](file:///home/harri/yaw/FOUNDATIONAL.md)

---

### 7.1 Epic 5.0 (Sprint 5A): Reversible Aliases & Schema Contracts
**Status:** **COMPLETE** (21 passing tests in [`tests/test_canonicalization.py`](file:///home/harri/yaw/tests/test_canonicalization.py))  
**Governing Document:** [`docs/adr/ADR-003.md`](file:///home/harri/yaw/docs/adr/ADR-003.md), [`docs/adr/ADR-004.md`](file:///home/harri/yaw/docs/adr/ADR-004.md), [`docs/adr/ADR-005.md`](file:///home/harri/yaw/docs/adr/ADR-005.md)

#### 1. Goal & Architecture
Prevent ontological fragmentation by implementing the core **Entity Disambiguation & Canonicalization (EDC)** protocol in SQLite and Python without touching extraction templates:
- **Reversible Alias Persistence:** Merges write strictly to `aliases` table. `unified_edges` and `unified_nodes` rewrite endpoints via `LEFT JOIN aliases ON ... COALESCE(a.canonical_id, x.id)`. Raw extraction tables (`leaf_nodes`, `leaf_relations`) remain immutable (preserving Invariant 1).
- **Two-Tier Convergence Engine ([`src/yaw/indexer/canonicalize.py`](file:///home/harri/yaw/src/yaw/indexer/canonicalize.py)):**
  1. *Tier 1 (Direct Match)*: Exact slug match or registered active alias lookup in `aliases` or `InMemoryLookupKB`. Instant resolution with 0 vector or LLM calls.
  2. *Tier 2 (Near Match)*: In-process vector similarity > `CANON_SIM_THRESHOLD` (0.80) using static GloVe 300d vectors (`en_core_web_lg`). Consolidates to existing canonical node. *(Superseded by [`ADR-012`](file:///home/harri/yaw/docs/adr/ADR-012.md): Tier 2 now blocks in space S and merges only on linguistic corroboration; Epic 7.7.)*
  3. *Induction (Novel Node)*: Novel entity registered with URI `concepts:{slug}` (`TierEnum.NEW_INDUCTION`, `is_new=True`).
- **Compiled C/Cython Knowledge Base & Prior Distributions ([`docs/adr/ADR-010.md`](file:///home/harri/yaw/docs/adr/ADR-010.md)):** `InMemoryLookupKB` maintains an alias candidate distribution with prior probabilities $P(\text{entity} \mid \text{alias}) \in (0, 1]$ satisfying unit sum $\sum_e P(e \mid a) = 1.0$. Microsecond candidate evaluation executes natively in Cython, and `aliases.score` records calibrated prior probabilities without premature winner-take-all discretization.
- **Namespace Inheritance:** Merged aliases strictly inherit the canonical node's root namespace (`concepts`, `entities`, `invariants`, `decisions`, `processes`, `claims`, `sources`, `contexts`, `predicates`).
- **Ingestion Hook:** `process_vault_raw` calls `resolve()` for nodes and predicates, writing canonical URIs to SQLite while recording aliases.

---

### 7.2 Epic 5.0b (Sprint 5-Align): Forensic Semantic Web Alignment & Quad Preparation
**Status:** **COMPLETE** (Commit `dd25de9`)  
**Governing Document:** [`docs/adr/ADR-008.md`](file:///home/harri/yaw/docs/adr/ADR-008.md), [`docs/rdf-owl-shacl.md`](file:///home/harri/yaw/docs/rdf-owl-shacl.md)

#### 1. Goal & Architecture
Conducted a comprehensive forensic audit across the architecture and standardized W3C Semantic Web alignments:
- Formulated [`docs/adr/ADR-008.md`](file:///home/harri/yaw/docs/adr/ADR-008.md) establishing the 14 canonical predicates across 4 categories (Subsumption, Dialectical, Dependency/Process, Provenance/Grounding).
- Standardized the 4-tier semantic stack: SQLite Quad substrate $\to$ OWL 2 RL axioms $\to$ SciPy Power PPR $\to$ SHACL closed-world validation.
- Authored the authoritative integration manual [`docs/rdf-owl-shacl.md`](file:///home/harri/yaw/docs/rdf-owl-shacl.md).
- Formulated Sprint 5.1 remediation plan to resolve all uncovered substrate discrepancies.

---

### 7.3 Epic 5.0c (Remediation Prior): Native spaCy & spacy-llm Remediation Prior
**Status:** **COMPLETE** (Commits `b70d8d4`, `bfdd064`, `67c33fa`)  
**Governing Document:** [`spacy/SKILL.md`](file:///home/harri/yaw/.agents/skills/spacy/SKILL.md), [`FOUNDATIONAL.md §3`](file:///home/harri/yaw/FOUNDATIONAL.md)

#### 1. Goal & Architecture
Remediated architectural drift by enforcing canonical spaCy and `spacy-llm` idioms across the entire codebase:
- **SSOT Base Model:** Standardized on `SPACY_MODEL = "en_core_web_lg"` with static 300d GloVe vectors for all in-process post-indexing operations, concept canonicalization, and token similarity. Restricted `embed-gemma` exclusively to chunk indexing and query embedding.
- **Disk Caching:** Mounted `spacy.BatchCache.v1` directly at `.cache/spacy_llm`, eliminating redundant Ollama `/api/chat` requests for identical note chunks.
- **Two-Tier Convergence:** Replaced 3-tier procedural ladder with native Two-Tier Convergence Engine in `canonicalize.py` (0 LLM pairwise adjudication calls during canonicalization, net -166L dead code pruned).
- **Novel Node Induction:** Replaced degraded "fallback" node creation with clean `TierEnum.NEW_INDUCTION`, supporting 300d/768d vector persistence and guarding predicate purity.
- **Linguistic FTS5 Query Cleaning:** Replaced naive `\w+` regex with spaCy `token.is_stop` and `token.lemma_` filtering in `Retriever._extract_fts_terms()`, preventing stopword pollution of the $p_0$ personalization vector.
- **Production Daemon Lifecycle:** Added clean `yaw shutdown` CLI subcommand and decomposed lifecycle termination handlers.

---

### 7.4 Epic 5.1 (Sprint 5.1): Mid-Epic 5 Remediation & Pre-E5S2 Architectural Alignment
**Status:** **COMPLETE** (Commit `481d5bd`, 226/226 tests green, Sentrux modularity 5234)  
**Governing Document:** [`docs/alignment-sprint-for-e5s2.md`](file:///home/harri/yaw/docs/alignment-sprint-for-e5s2.md), [`docs/adr/ADR-008.md`](file:///home/harri/yaw/docs/adr/ADR-008.md), [`FOUNDATIONAL.md §1, §5.4`](file:///home/harri/yaw/FOUNDATIONAL.md)  
**Target Test Files:** `tests/test_storage.py`, `tests/test_canonicalization.py`

#### 1. Goal & Architecture
Surgically remediate schema, configuration, and namespace discrepancies identified during the cross-epic forensic audit before launching Sprint 5B:
1. **Predicate SSOT Alignment:** Update `config.BASE_PREDICATES` to the authoritative 14 ADR-008 canonical predicates across 4 categories, explicitly purging `is_a`. Update `daemon.get_ontology_schema()` to reflect this SSOT.
2. **Plural Root Namespace Alignment:** Change `canonicalize.py` defaults from singular `"concept"` to plural `"concepts"`, validating against `config.TAXONOMY_NAMESPACES` to guarantee W3C URI regex compliance.
3. **SQLite Aliases Method CHECK Constraint & Calibrated Priors:** Reconcile `db.py` `CHECK(method IN ('exact', 'alias', 'auto', 'new', 'ontology'))` with `MatchMethod` enum (with `'lexical'`, `'llm'`, and `'compile'` completely purged in Epic 7.5), while formalizing `aliases.score` as the calibrated candidate prior confidence / conditional probability $P(\text{canonical\_id} \mid \text{alias\_id})$ feeding into PPR seeding ([`docs/adr/ADR-010.md`](file:///home/harri/yaw/docs/adr/ADR-010.md)).
4. **5-Column `unified_edges` Quad Projection:** Extend `unified_edges` view to project `(src_id, dst_id, rel_type, weight, source_chunk_id)` for seamless Sprint 6A named quad export.
5. **Compiled Knowledge Persistence:** Architect `compiled_nodes` and `compiled_relations` tables in `db.py` and union into views, preserving Invariant 1 (extraction substrate purity) during ADR-002 saturated hub bisections.
6. **Subsumption Conductance Asymmetry:** Implement 1.0 downward / 0.2 upward conductance in `retriever.py` for `subClassOf`/`partOf` with reciprocal reverse edge synthesis.
7. **Contested Alias State Machine:** Implement automated transition to `status = 'contested'` when candidate confidence is low ($<0.80$) or when conflicting candidates emerge, flagging them for SHACL closed-world linting without live LLM calls. *(Retired by [`ADR-012`](file:///home/harri/yaw/docs/adr/ADR-012.md): ambiguity is now ≥2 `close_matches` edges reported as an ambiguous-identity `sh:Warning`; Epic 7.7.)*

#### 2. Verification
- `uv run pytest tests/test_sprint5_1_alignment.py -v` (100% pass across all 7 tests).
- Zero regressions across existing test suite (226/226 tests pass across full suite).
- `uv run ruff check` clean, `uv run ty check` clean.
- Sentrux modularity elevated to 5234 (raw $Q = 0.2851$, cross-module edges: 9, quality signal: 7414).

---

### 7.5 Epic 5.2 (Sprint 5B): Dynamic Few-Shot Steering & Live Vocabulary Injection (E5S2)
**Status:** **COMPLETE** (35/35 new tests in `tests/test_fewshot_steering.py`, 261/261 total repo tests pass, Sentrux modularity 5224)  
**Governing Document:** [`docs/adr/ADR-008.md`](file:///home/harri/yaw/docs/adr/ADR-008.md), [`docs/how-to-handle-spacy-llm.md`](file:///home/harri/yaw/docs/how-to-handle-spacy-llm.md), [`FOUNDATIONAL.md §5.1`](file:///home/harri/yaw/FOUNDATIONAL.md)  
**Target Test File:** `tests/test_fewshot_steering.py`

#### 1. Goal & Architecture
Close the active feedback loop between SQLite and initial extraction. Guide the LLM during Stage 1 extraction to prefer established canonical entities and predicates, preventing fragmentation before strings even reach the resolver:
- **Canonical Vocabulary Provider:** Query `aliases` and `leaf_nodes` for top frequency/conductance nodes in the vault.
- **ADR-008 Predicate Seeding:** Feed categorized canonical seed vocabulary from `config.BASE_PREDICATES` (Subsumption: `subClassOf`, `partOf`, `instanceOf`; Dialectical: `supports`, `refutes`, `contradicts`, `supersedes`; Dependency/Process: `dependsOn`, `causes`, `precedes`, `produces`; Provenance: `derivedFrom`, `cites`, `manifestsIn`) into extraction prompts to prevent open-ended synonym proliferation (`is_a`, `child_of`, `kind_of`).
- **Live Predicate Injection:** Query `unified_edges` for top active canonical predicates and combine with `config.BASE_PREDICATES` fallback, ensuring dynamic prompts reflect both grounded vault usage and standard W3C taxonomy.
- **spaCy-LLM Integration:** Register `@registry.misc("yaw.CanonicalExamples.v1")` providing dynamic few-shot examples for `spacy.REL.v1` and `spacy.SpanCat.v3`. Adhere strictly to Epic 1 (pure declarative configuration, zero private `_model` or `_task` attribute access).
- **Quad Projection Compatibility:** Ensure relations extracted via steered prompts project losslessly into `(subject, predicate, object, graph_uri)` quad tuples in `unified_edges` with source chunk provenance.
- **Multi-Note Convergence Verification:** Ingest multi-note corpus (e.g. `sample-llm-wiki`) and audit `yaw.db` for zero duplicate concept nodes and zero uncontrolled predicate synonyms.

#### 2. Test Authoring Plan (TDD & Grounded Invariants)
- **Target Test File:** `tests/test_fewshot_steering.py`.
- **Invariant 5B.1 (Registry Factory Registration & Schema):**
  - Assert `@registry.misc("yaw.CanonicalExamples.v1")` exists in spaCy global registry.
  - Calling factory returns valid `Callable[[], list[dict]]` format expected by `spacy.REL.v1`.
- **Invariant 5B.2 (Declarative Pipe Integration):**
  - Add pipe `nlp.add_pipe("llm", config={"task": {"@llm_tasks": "spacy.REL.v1", "examples": {"@misc": "yaw.CanonicalExamples.v1"}}})`.
  - Assert pipe builds cleanly without exceptions or private attribute mutation.
- **Invariant 5B.3 (Multi-Note Ingestion Convergence Audit):**
  - Ingest two notes referencing "special relativity" and "Einsteinian relativity".
  - Assert graph converges to single canonical URI with alias mapping and zero disconnected duplicate nodes.
- **Invariant 5B.4 (Canonical Predicate Seeding & Quad-Ready Projection Compatibility):**
  - Assert dynamic few-shot steering injects top predicates from `unified_edges` combined with `config.BASE_PREDICATES` ([`ADR-008`](file:///home/harri/yaw/docs/adr/ADR-008.md)).
  - Assert extracted relation types align with canonical seed predicates (<5% out-of-vocabulary rate).

#### 3. Verification Outcomes
- `uv run pytest tests/test_fewshot_steering.py -v`: 35/35 passed (100%).
- Full repository regression suite: 261/261 passed (100%).
- Adversarial challenger suites: 39/39 passed (100%).
- Total passing tests: 335 passed, 0 failed, 0 regressions.
- Static analysis: `ruff check` clean, `ruff format` clean, `ty check` clean.
- Sentrux architecture gate: Modularity score 5224 (> 5000), 0 cycles, coupling 0.04.

---

## 8. Epic 6: Full RDF/OWL/SHACL Semantic Web Compliance
**Status:** **IN PROGRESS (EPIC 6.1 COMPLETE; EPIC 6.2 NEXT TARGET — READY FOR TEAMWORK)**  
**Governing Document:** [`FOUNDATIONAL.md §1, §5.4`](file:///home/harri/yaw/FOUNDATIONAL.md), [`docs/ADR.md`](file:///home/harri/yaw/docs/ADR.md), [`docs/adr/ADR-002.md`](file:///home/harri/yaw/docs/adr/ADR-002.md), [`docs/adr/ADR-008.md`](file:///home/harri/yaw/docs/adr/ADR-008.md), [`docs/rdf-owl-shacl.md`](file:///home/harri/yaw/docs/rdf-owl-shacl.md)  
**Target Test Files:** `tests/test_rdf_substrate.py`, `tests/test_owl_ontology.py`, `tests/test_shacl_lint.py`

### 8.1 Goal & Architecture
Elevate YAW's graph substrate, taxonomic inference, and wiki validation into 100% formal Semantic Web compliance across the layered W3C stack without sacrificing zero-latency SQLite speed:
1. **RDF Layer (Data Structures, CURIEs & Quad Store Integration):**
   - Unify graph storage as canonical Subject-Predicate-Object triples with standard CURIE namespaces (`yaw:<namespace>:<slug>`), supporting lossless in-process streaming and round-trip serialization to Turtle (`.ttl`), JSON-LD, N-Triples, and N-Quads via `rdflib` without external graph daemons or triple-store servers.
   - Adhere to Kurt Cagle's <30-table relational threshold (relational SQLite B-tree views outperform standalone graph daemons at this schema scale).
   - Preserve chunk provenance and document origin context via named quad representations (`yaw:sources/{rel_path}#chunk:{ordinal}`).
2. **OWL & SKOS Layer (OWL 2 RL / RDFS Ontology & Taxonomic Subsumption):**
   - Formalize knowledge representation strictly within the pragmatic **OWL 2 RL (Rule Language) / RDFS** profile, combined with W3C SKOS vocabulary alignment (`skos:Concept`, `skos:prefLabel`/`altLabel`, `skos:broader`/`narrower`).
   - Explicitly avoid heavy OWL 2 DL reasoners (tableaux solvers like Pellet or HermiT) that suffer from exponential worst-case complexity and require Open World Assumption (OWA).
   - Enforce strict `rdfs:subClassOf` and `skos:broader` hierarchies, B-tree hypernym induction ([`ADR-002`](file:///home/harri/yaw/docs/adr/ADR-002.md)), and dynamic identity resolution via `owl:sameAs` / `skos:altLabel` backed by `aliases` and `canon_verdicts` view joins.
   - Compute transitive closures deterministically using SQLite recursive CTEs (`WITH RECURSIVE`) and SciPy sparse matrix powers ($P^k$).
   - Maintain asymmetric graph conductance (1.0 downward broadcast, 0.2 upward damping) for deterministic spectral diffusion without undecidable Description Logic (DL) tableau bottlenecks.
   - Enforce subsumption graph Directed Acyclic Graph (DAG) validation to eliminate circular inheritance.
3. **SHACL Layer (Wiki Lint & Closed-World Validation):**
   - Implement declarative closed-world graph validation for `lint_wiki` via in-process `pyshacl` / `rdflib` executed directly against SQLite views, requiring zero background daemons.
   - Define `sh:NodeShape` and `sh:PropertyShape` constraints enforcing valid definitions ($\le 50$ words), canonical URI syntax, predicate domain/range bounds, non-orphaned relations, and detection of `contested` canonical merges and dialectical contradictions, returning standard W3C `sh:ValidationReport` objects.

---

### 8.2 Epic 6.1 (Sprint 6A): RDF Data Structures & Quad Store Integration
**Status:** **COMPLETE** (Commit `462ad1e`, 90/90 tests green, 353 total repo tests passed, Sentrux modularity 5703, quality 7565)  
- **Goal:** Expose YAW's internal SQLite graph as an authoritative W3C RDF triple and named quad store via in-process `rdflib` streaming and serialization of `unified_edges` to Turtle, JSON-LD, and N-Triples without external graph daemons, while completing codebase hardening by eliminating production code smells and elevating Sentrux modularity.
- **Deliverables:**
  - Export module (`src/yaw/storage/rdf.py`) providing:
    - `export_graph(format="turtle" | "json-ld" | "nt" | "nquads") -> str` (in-process streaming serialization from SQLite `unified_edges`)
    - `to_rdflib_graph() -> rdflib.ConjunctiveGraph`
  - Maps `unified_edges` dynamically to `(s, p, o)` RDF triples using canonical CURIE URIs (`yaw:<namespace>:<slug>`).
  - Quad provenance context: attaches named graph identifier corresponding to source note chunk (`yaw:sources/{rel_path}#chunk:{ordinal}`).
  - Standard namespace bindings (`yaw:`, `rdfs:`, `owl:`, `rdf:`, `sh:`, `skos:`).
  - **Code Smell Remediation & Sentrux Hardening:**
    - Decompose `src/yaw/indexer/pipeline.py` to $<30$ symbols (inlining trivial private helpers to eliminate God-File smell).
    - Decompose `src/yaw/config.py::_populate` into `_populate_paths()` and `_populate_nlp()` helpers (drops main fn from 166L to $<50$L, eliminating long-function smell).
    - Extract `_fuse_candidate_scores()` helper from `src/yaw/retrieval/retriever.py::retrieve` (drops method from 122L to $<75$L, eliminating long-function smell).
    - Rename retriever method `find_seed` to `find_seed_nodes(...)` (MCP tool in daemon remains `find_seed`) to eliminate duplicate symbol collision.
    - Elevate Sentrux modularity score to $>5500$ (baseline 5069) and quality score $>7500$ while maintaining 0 circular dependencies (acyclicity 10,000).
- **Test File:** `tests/test_rdf_substrate.py`
  - Real SQLite to Turtle / JSON-LD / N-Quads export round-trip via in-process `rdflib.ConjunctiveGraph()`.
  - Asserts exact SPO triple preservation, valid CURIE resolution, named quad chunk provenance retention, and zero external daemon processes.

---

### 8.3 Epic 6.2 (Sprint 6B): OWL 2 RL & SKOS Ontology Semantics & Axiom Inference
**Status:** **COMPLETE** (Commit `79ce19d`, 25/25 tests green, 375 total repo tests passed, Sentrux quality 7270, 0 cycles)  
- **Goal:** Formalize taxonomy, hypernym clusters, and alias resolution using the **OWL 2 RL / RDFS profile** and SKOS vocabulary alignment with recursive SQL and SciPy transitive closure, explicitly avoiding heavy OWL 2 DL tableaux solvers.
- **Deliverables:**
  - OWL/SKOS taxonomy module (`src/yaw/ontology/owl.py`):
    - Aligns nodes with `skos:Concept`, canonical names with `skos:prefLabel`, and active aliases with `skos:altLabel`.
    - Maps ADR-002 B-tree hypernym splits to `rdfs:subClassOf` and `skos:broader`/`skos:narrower` axioms.
    - Reversible alias mapping translating active `aliases` rows to `owl:sameAs` equivalence axioms via view joins, and distinct verdicts to `owl:differentFrom`. *(The `owl:differentFrom` projection is retired by [`ADR-012`](file:///home/harri/yaw/docs/adr/ADR-012.md) §5: unsound under OWA and inconsistent with later merges via OWL 2 RL `eq-diff1`; Epic 7.7.)*
    - Transitive closure computation for taxonomic subsumption queries (`get_subclass_closure(node_id)`) via SQLite recursive CTEs and SciPy sparse matrix powers.
    - Subsumption tree DAG validator asserting zero cycle formation.
  - Asymmetric conductance weighting (1.0 downward / 0.2 upward) formalizing directed property inheritance for GraphRAG.
- **Test File:** `tests/test_owl_ontology.py`
  - Verifies subsumption tree acyclicity, DAG validation, and `rdfs:subClassOf` / `skos:broader` transitive closure.
  - Verifies dynamic `owl:sameAs` equivalence across canonical merges and `owl:differentFrom` distinction.
  - Verifies zero reliance on external DL reasoners.

---

### 8.4 Epic 6.3 (Sprint 6C): SHACL Closed-World Wiki Linter (`lint_wiki` Tool)
- **Goal:** Replace heuristic graph linting with formal in-process `pyshacl` closed-world shape validation over SQLite views backing the `lint_wiki` FastMCP tool, returning standard W3C `sh:ValidationReport`.
- **Deliverables:**
  - Declarative SHACL shape definitions (`src/yaw/ontology/shapes/wiki_shapes.ttl`):
    - `ConceptShape`: Requires non-empty `definition` ($\le 50$ words) and canonical URI syntax matching 9 universal root namespaces.
    - `RelationShape`: Validates subject/predicate/object domain and range constraints, detecting dangling pointers and invalid target nodes.
    - `DialecticalIntegrityShape`: Flags contradictory claims (e.g. `A causes B` vs `A prevents B`) and `contested` alias merges (`aliases.status = 'contested'`). *(Replaced by the ambiguous-identity `sh:Warning`, [`ADR-012`](file:///home/harri/yaw/docs/adr/ADR-012.md); Epic 7.7.)*
  - Integration into `lint_wiki`: Runs in-process `pyshacl` validation against the SQLite RDF projection with support for namespace/keyword filtering (`lint_wiki(query: str | None = None)`), yielding standard W3C `sh:ValidationReport` conformance status and structured violation items without external daemons.
- **Test File:** `tests/test_shacl_lint.py`
  - Asserts detection of missing definitions, invalid namespaces, broken links, contested aliases, and dialectical contradictions.
  - Asserts valid W3C `sh:ValidationReport` output structure and zero external daemon processes.

---

## 9. Epic 7: Pluggable RDF/OWL Ontology Substrates & Configurable Open-IE Extraction (Sprint 7A)
**Status:** **SPECIFIED (NEXT TARGET v0.95)**  
**Governing Documents:** [`FOUNDATIONAL.md §5.4`](file:///home/harri/yaw/FOUNDATIONAL.md), [`docs/rdf-owl-shacl.md`](file:///home/harri/yaw/docs/rdf-owl-shacl.md), [`docs/adr/ADR-008.md`](file:///home/harri/yaw/docs/adr/ADR-008.md)  
**Target Test Files:** `tests/test_pluggable_ontology.py`

### 9.1 Goal & Architecture
Transform the hardcoded 9-namespace taxonomy into a pluggable RDF/OWL substrate supporting external ontologies (e.g. Digital Construction Ontologies, domain SKOS vocabularies) and configurable Open-IE extraction modes:
1. **Dynamic Ontology Ingestion (`infer_ontologies`):**
   - Ingest RDF/OWL/Turtle files from local disk paths or remote URIs via in-process `rdflib`.
   - Materialize concepts, object properties, and subsumption hierarchies into SQLite dynamic taxonomy tables (`dynamic_ontologies`, `dynamic_classes`, `dynamic_properties`).
   - Default to built-in universal schema (`src/yaw/ontology.ttl`), with zero hardcoded code restrictions.
2. **Schema Generalization & Constraint Decoupling:**
   - Migrate SQLite tables (`leaf_nodes`, `leaf_edges`, `unified_nodes`, `unified_edges`) to relax rigid 9-namespace `CHECK` constraints in favor of a dynamic schema registry lookup.
   - Dynamic namespace URI expansion and CURIE prefix registration.
3. **Dual Extraction Pipeline (`open_ie` mode):**
   - `open_ie = false`: Strict closed-world extraction; spaCy-LLM extracts entities and relations mapped strictly to the loaded ontology classes and properties.
   - `open_ie = true`: Unconstrained Open Information Extraction; dynamically mints leaf entities and open predicates with confidence scoring.
4. **Dynamic Closed-World SHACL Whitelisting:**
   - Dynamically synthesize or bind domain-specific SHACL shape graphs (`shapes_graph`) based on imported OWL classes and restrictions.
   - Closed-world pySHACL validation enforces domain-specific shape constraints without modifying core engine logic.
5. **Dynamic Knowledge Base Vocabulary & Alias Prior Synchronization ([`docs/adr/ADR-010.md`](file:///home/harri/yaw/docs/adr/ADR-010.md)):**
   - Imported RDF/OWL/SKOS ontologies dynamically synchronize terms, class labels, and `skos:altLabel` synonyms directly into `InMemoryLookupKB` via `sync_kb_from_db`.
   - Each imported synonym registers a normalized candidate prior $P(e \mid a)$, ensuring external domain ontologies immediately participate in probabilistic seed formulation and topological disambiguation during GraphRAG retrieval without manual prompt tuning.

### 9.2 Test Authoring Plan (TDD & Grounded Invariants)
- **Invariant 1 (Pluggable Ingestion Invariant):**
  - Ingest an external OWL ontology (e.g., DiCo or test Turtle ontology) via `yaw.ontology.loader.ingest_ontology()`.
  - Assert classes and properties are registered in SQLite dynamic taxonomy tables with exact `rdfs:subClassOf` and `rdfs:subPropertyOf` closures.
- **Invariant 2 (Dual Extraction Invariant):**
  - Verify `open_ie=false` filters extracted triples to registered ontology terms.
  - Verify `open_ie=true` allows arbitrary predicates while maintaining valid URI syntax and SQLite storage.
- **Invariant 3 (Dynamic SHACL Conformance):**
  - Assert pySHACL validates instances against imported ontology shapes without hardcoded namespace assertions.

---

## 9.5. Epic 7.5: Architectural Convergence & Pre-Epic 8 Substrate Realignment
**Status:** **IMPLEMENTED & VERIFIED**  
**Governing Document:** `pre_epic8_realignment_plan.md`, [`FOUNDATIONAL.md §3, §4, §5`](file:///home/harri/yaw/FOUNDATIONAL.md), [`ADR-003`](file:///home/harri/yaw/docs/adr/ADR-003.md), [`ADR-010`](file:///home/harri/yaw/docs/adr/ADR-010.md)  
**Target Test Files:** `tests/test_retriever.py`, `tests/test_pluggable_ontology.py`, `tests/test_canonicalization.py`, `tests/test_fewshot_steering.py`

### 9.5.1 Goal & Architectural Pillars
Harden the core ingestion, retrieval, storage, and server layers to eliminate architectural smells, align with ADR-010 candidate prior semantics, and enforce zero-fallback guarantees across the substrate ahead of Epic 8:
1. **Bayesian Evidence Chunk Scoring & Canopy Seeding (R1, ADR-010):**
   - Candidate prior probabilities $P(e \mid a)$ from `InMemoryLookupKB` are blended proportionally into seed vector $\mathbf{p}_0$ ($\sum_a P(e \mid a) \cdot \max(0.0, \text{sim}(q, e))$), simplifying continuously to $\sum_a P(e \mid a)$ when $\mathbf{q} = \mathbf{0}$ (Ollama offline).
   - Compound Bayesian evidence chunk scoring:
     $$\text{Score}(c) = \left( \sum_{e \in \text{entities}(c)} P(e \mid a) \cdot \mathbf{p}^*[e] + \mathbf{p}^*[c] \right) \cdot \text{sim}(q, c)$$
   - Full query mention extraction spanning exact query, spaCy doc entities, noun chunks, tokens, and sliding n-grams.
2. **Storage Schema & DB Constraint Purification (R2, Invariant 4):**
   - Purged dead-code methods (`'lexical'`, `'llm'`, `'compile'`) from `aliases` table `CHECK` constraints. Canonical set strictly enforced: `('exact', 'alias', 'auto', 'new', 'ontology')`.
   - Purged silent exception swallowers and legacy argument-order shims in `insert_alias` and `insert_leaf_node`, enforcing immediate `ValueError` and zero fallbacks.
   - Idempotent migration in `ensure_schema_tables` rebuilding obsolete `aliases` tables non-destructively.
   - Synchronized `FOUNDATIONAL.md` SQLite schema contracts.
3. **Dynamic Ontology Loader KB Synchronization (R3, ADR-010):**
   - Implemented `sync_dynamic_ontology_to_kb` registering dynamically ingested OWL/RDFS classes, properties, and synonym aliases with normalized candidate priors into `InMemoryLookupKB`.
   - Connected dynamic sync directly to `ingest_ontology(..., kb=...)`.
4. **Daemon KB Wiring & Cache Lifecycle (R4):**
   - Implemented module-level `get_daemon_kb(conn)` and `reset_daemon_kb()`.
   - Mounted `InMemoryLookupKB` into `GraphRAGRetriever`, `query_wiki`, and `find_seed`.
   - Wired cache invalidation and re-synchronization into `compile_wiki` and `load_ontology`.

---

## 9.6. Epic 7.6: Compile-Time Closed-World SHACL Validation & O(1) Contradiction Caching
**Status:** **COMPLETE** (Commit `e64ed3c`; ontology vector indexing superseded by Epic 7.7)  
**Governing Document:** [`docs/adr/ADR-011.md`](file:///home/harri/yaw/docs/adr/ADR-011.md), [`FOUNDATIONAL.md §4.3, §4.4, §5.5`](file:///home/harri/yaw/FOUNDATIONAL.md), [`docs/rdf-owl-shacl.md`](file:///home/harri/yaw/docs/rdf-owl-shacl.md)  
**Target Test Files:** `tests/test_compile_shacl_cache.py`, `tests/test_shacl_lint.py`

### 9.6.1 Goal & Architectural Pillars
Decouple streaming chunk ingestion from heavy SHACL validation while eliminating the multi-second latency of on-demand `lint_wiki()` invocations:
1. **Compile-Time Fast SHACL Execution (`compile_wiki`):**
   - Wire `validate_wiki_graph_fast(conn)` into `compile_wiki()` in `src/yaw/server/daemon.py`.
   - Executes lean closed-world validation in $<200\text{ ms}$ over SQLite relational pre-closure (`unified_edges`, `unified_nodes`) with `inference=None`.
   - Surfaces dialectical contradictions (`causes` vs `prevents`, `supports` vs `refutes`, contested alias merges) and structural violations (`yaw:ConceptShape`, `yaw:RelationShape`).
2. **Persistent `compiled_violations` SQLite Storage:**
   - Add `compiled_violations` schema table and indexes (`idx_violations_focus`, `idx_violations_shape`, `idx_violations_severity`) to `src/yaw/storage/schema.py`.
   - Add idempotent migration in `ensure_schema_tables`.
   - Implement `persist_compiled_violations(conn, violations)` and `fetch_compiled_violations(conn, focus_node=None, query=None)`.
   - Transactionally refresh compilation results during each `compile_wiki()` run.
3. **$\mathcal{O}(1)$ Instantaneous `lint_wiki` Protocol:**
   - Refactor `lint_wiki(query=None, vault_dir=None, force=False)` to query `compiled_violations` directly from SQLite by default.
   - Returns structured violation records in $<5\text{ ms}$.
   - Scoped query (`query="X"`) performs fast indexed SQLite `WHERE` filter.
   - Live pySHACL is reserved exclusively for `force=True`.
4. **Ontology Layer FTS & Vector Indexing:**
   - In `src/yaw/ontology/loader.py` (`sync_dynamic_ontology_to_kb` / `ingest_ontology`), insert dynamic class names and definitions into `query_fts` (FTS5).
   - Store class embeddings in `sqlite-vec` during dynamic ontology ingestion and compile-time hypernym induction.
   - Ensures `compute_seed_candidates` retrieves ontology nodes as first-class seed candidates.

### 9.6.2 Test Authoring Plan (TDD & Unmocked Invariants)
- `tests/test_compile_shacl_cache.py`:
  - `test_compile_wiki_runs_fast_shacl_and_persists_violations`: Seed contradictory relations (`A causes B` and `A prevents B`), call `compile_wiki()`, assert `compiled_violations` contains 2 records and execution completes in $<250\text{ ms}$.
  - `test_lint_wiki_serves_from_sqlite_cache_in_sub_10ms`: Assert `lint_wiki()` queries the SQLite cache with zero `pyshacl.validate` calls when `force=False`.
  - `test_lint_wiki_scoped_query_filtering`: Assert `lint_wiki(query="target_concept")` returns only violations where `focus_node` or message matches target.
  - `test_lint_wiki_force_revalidation_updates_cache`: Assert `lint_wiki(force=True)` triggers live validation and refreshes the cache.
  - `test_dynamic_ontology_ingestion_inserts_fts_and_vectors`: Assert `ingest_ontology` indexes classes into `query_fts` and `leaf_nodes.embedding`.

> [!NOTE]
> Pillar 4's vector half (ontology class embeddings in `leaf_nodes.embedding`, query-vector seeding of ontology nodes) is superseded by [`ADR-012`](file:///home/harri/yaw/docs/adr/ADR-012.md) §5 and re-implemented in Epic 7.7. The FTS5 half stands.

---

## 9.7. Epic 7.7: Representation-Space Separation, Corroborated Identity Resolution & Query-to-Fact Seeding
**Status:** **SPECIFIED (BLOCKING PREREQUISITE OF EPIC 8)**  
**Governing Document:** [`docs/adr/ADR-012.md`](file:///home/harri/yaw/docs/adr/ADR-012.md), [`FOUNDATIONAL.md §3, §5.1–§5.3, §5.5`](file:///home/harri/yaw/FOUNDATIONAL.md), [`docs/adr/ADR-003.md`](file:///home/harri/yaw/docs/adr/ADR-003.md), [`docs/adr/ADR-010.md`](file:///home/harri/yaw/docs/adr/ADR-010.md)  
**Target Test Files:** `tests/test_representation_spaces.py` (new), `tests/test_canonicalization.py`, `tests/test_retriever.py`, `tests/test_pluggable_ontology.py`, `tests/test_compile_shacl_cache.py`, `tests/test_owl_ontology.py`, `tests/test_shacl_lint.py`

### 9.7.1 Goal
Every vector store holds exactly one representation space, entity identity is decided by evidence that measures identity rather than relatedness, and retrieval seeds the graph the way the multi-hop evidence says it should (query-to-fact). Epic 8 benchmark numbers are meaningless until this holds, so Epic 8 does not start before this Epic's quality gates pass.

### 9.7.2 Architectural Pillars
1. **Schema migration (`src/yaw/storage/schema.py`, `db.py`):**
   - Drop `leaf_nodes.embedding` and `aliases.status`. Remove every `status = 'active'` filter from views and queries.
   - Add `close_matches` (with `CHECK (src_id < dst_id)`), `entity_vec` (space S, `PARTITION KEY` namespace, `distance_metric=cosine`), `fact_vec` (space R).
   - Idempotent migration in `ensure_schema_tables()`. Rows with `status = 'contested'` migrate to `close_matches` and are removed from `aliases`.
   - `unified_edges` gains the symmetric `closeMatch` branch exactly as in [`FOUNDATIONAL.md §5.2`](file:///home/harri/yaw/FOUNDATIONAL.md).
2. **Tier 1 normalized key (`src/yaw/indexer/canonicalize.py`):** NFKC + casefold, leading determiner and possessive clitic removed, `NOUN` lemmatized, `PROPN` verbatim, punctuation / hyphen / whitespace collapsed. Induced IDs are `{namespace}:{normalized-key}`. Existing slugs whose key changes receive a Tier 1 alias row so no existing reference breaks.
3. **Tier 2 blocked + corroborated matching:** `entity_vec` KNN (`k = CANON_TOP_K`, cosine ≥ τ) restricted to the mention namespace plus active ontology partitions; merge only when exactly one candidate passes the linguistic corroboration predicate (acronym, head-anchored containment, initial expansion); otherwise write `close_matches`. Every evaluated pair is logged to `canon_verdicts`.
4. **Predicates are space L only:** zero embedding calls for predicates. `_find_vec_nearest` is never reached for `kind = 'predicate'`.
5. **Dead and duplicated code removal:** delete `_find_kb_nearest`, the spaCy-vector branch of `_get_or_compute_vector`, and the copy-pasted KB vector blocks. `InMemoryLookupKB` becomes an alias → candidate prior index whose priors are computed solely from `aliases` + `close_matches` (ADR-012 §3). Entity vector width is no longer a KB concern.
6. **Embedding prompts (`src/yaw/nlp/embeddings.py`, SSOT):** one formatter per space. Space S uses `task: sentence similarity | query: {name}`; space R documents use `title: {title} | text: {text}` and queries `task: search result | query: {q}` ([EmbeddingGemma model card](https://ai.google.dev/gemma/docs/embeddinggemma/model_card)). The `Classification` prompt is removed from all entity paths.
7. **Fact embedding in the pipeline (`src/yaw/indexer/pipeline.py`):** each `leaf_relations` row is rendered as `title: {note_title} | text: {subject_name} {surface_predicate} {object_name}` and embedded in the same batched `/api/embed` call as its chunks; written to `fact_vec`. Fact text is never stored (Invariant 2).
8. **Retriever query-to-fact seeding (`src/yaw/retrieval/retriever.py`):** implement $\mathbf{s}_{\text{concepts}} = \lambda \widehat{\mathbf{s}}_{\text{alias}} + (1-\lambda)\widehat{\mathbf{r}}$ exactly as [`FOUNDATIONAL.md §5.3`](file:///home/harri/yaw/FOUNDATIONAL.md); delete `_score_leaf_nodes_vec` and all query → `leaf_nodes` vector seeding. Add `closeMatch` conductance ($W = \text{score}$, both directions).
9. **Ontology loader (`src/yaw/ontology/loader.py`):** stop writing ontology classes to `leaf_nodes`; write one space-S vector per class to `entity_vec` (partition = prefix); make `query_fts` ontology rows idempotent on re-ingestion. A `close_matches` edge to an ontology class instantiates it as a ghost class in the PPR graph.
10. **SHACL & OWL (`src/yaw/ontology/shapes/wiki_shapes.ttl`, `shacl.py`, `owl.py`):** replace the contested-merge check with the ambiguous-identity `sh:Warning`; project `close_matches` to `skos:closeMatch` (cross-scheme) / `yaw:closeMatch` (intra-vault, `owl:SymmetricProperty`); delete the `canon_verdicts` → `owl:differentFrom` projection (ADR-012 §5).
11. **Fail-loud dimensions:** every vector write goes through `vec0`, so a wrong-width vector raises `sqlite3.OperationalError`; no `try/except` downgrades it to a log line.
12. **Entity Definition & Grounding Invariant (`src/yaw/nlp/components.py`, `pipeline.py`):**
    - Purge legacy `_entity_definition` placeholder (`f"{name} ({ent.label_})"`). Raw note entity mentions use their enclosing sentential context (`ent.sent.text.strip()`) for definition grounding.
    - When no definition or sentential context exists, definition defaults to `""`, cleanly emitting `task: sentence similarity | query: {name}` in Space S without tautological repetitions or dangling colons.
    - Saturated hubs and compile-time wiki synthesis can selectively route ambiguous entities to `spacy-llm` intensional definition induction ("what is X") if sentential context is too coarse.

### 9.7.3 Test Authoring Plan (TDD, real SQLite + loopback Ollama, zero mocks of DB/FS)
- **Space integrity:** inserting a 300-float vector into `entity_vec` or `fact_vec` raises `sqlite3.OperationalError`; `PRAGMA table_info(leaf_nodes)` has no `embedding` column; `aliases` has no `status` column.
- **Identity precision:** "Barack Obama" vs existing "Michelle Obama" → no `aliases` row, one `close_matches` row, `canon_verdicts.verdict = 'distinct'`.
- **Identity recall:** "A. Einstein" and "Einstein" each merge into existing "Albert Einstein" when it is the unique corroborated candidate; "NASA" merges into "National Aeronautics and Space Administration" (acronym).
- **Ambiguity:** a mention with two corroborated candidates → zero merges, two `close_matches` rows, and `compile_wiki()` persists one ambiguous-identity `Warning` in `compiled_violations`.
- **Tier 1:** "The transformers" and "Transformer" resolve to the same canonical with zero embedding calls (assert loopback request count).
- **Predicates:** resolving 20 predicate surface forms issues zero `/api/embed` requests.
- **Fact index:** after ingesting a note with *n* relations, `fact_vec` has exactly *n* rows keyed by `relation_id`, and chunk + fact embeddings arrive in one batched request per chunk group.
- **Query-to-fact seeding:** a two-hop fixture (*A* —r1→ *B* —r2→ *C*, question names *A*, answer is *C*): *A* receives seed mass via $\mathbf{s}_{\text{alias}}$ even when no top-`FACT_TOP_K` fact contains *A*; the chunk holding r2 ranks in the top-k.
- **Offline continuity:** with Ollama unreachable, $\mathbf{p}_0$ is a valid distribution built from alias + FTS5 channels only, and no exception escapes.
- **Ontology:** ingesting the same TTL twice leaves `query_fts` row counts unchanged; `leaf_nodes` gains zero rows; a `close_matches` edge to an ontology class makes that class a node in the PPR matrix.
- **W3C projection:** `to_rdflib_graph()` contains `yaw:closeMatch` in both directions for an intra-vault pair, `skos:closeMatch` for a vault ↔ ontology pair, and zero `owl:differentFrom` triples.

### 9.7.4 Quality Gates
- `uv run pytest --no-testmon`: 100% pass, 0 regressions.
- `uv run ruff check src/ tests/`, `uv run ruff format --check src/ tests/`, `uv run ty check src/ tests/`: 0 diagnostics.
- `git grep -n -e "leaf_nodes.embedding" -e "status = 'active'" -e "contested" -e "_find_kb_nearest" -e "differentFrom" -- src/` returns 0 matches.
- Ontologist critic review: zero unaddressed findings against ADR-012 and FOUNDATIONAL.md.

---

## 10. Epic 8: Production FastMCP Server, Compiled Wiki Materialization & GitHub v1.0 Launch
**Status:** **SPECIFIED (TARGET v1.0 LAUNCH)**  
**Governing Document:** [`docs/karpathy-llm-wiki.md`](file:///home/harri/yaw/docs/karpathy-llm-wiki.md), [`FOUNDATIONAL.md §4, §5.4, §6`](file:///home/harri/yaw/FOUNDATIONAL.md), [`docs/adr/ADR-002.md`](file:///home/harri/yaw/docs/adr/ADR-002.md), [`docs/adr/ADR-011.md`](file:///home/harri/yaw/docs/adr/ADR-011.md)  
**Target Test Files:** `tests/test_fastmcp_contract.py`, `tests/test_btree_hac.py`, `tests/test_wiki_materializer.py`, `eval/run_benchmark.py`, `tests/test_e2e_launch.py`

### 10.1 Goal & Architecture
Achieve production feature completeness (v1.0) and prepare the repository for public GitHub launch. Deliver the full Karpathy LLM-Wiki contract over FastMCP with streaming HTTP/SSE access, non-blocking background continuous indexing, and benchmark evaluation against published QA datasets:
1. **Compilation Trigger & B-Tree Ward HAC Deepening (Epic 8.1 / Sprint 8A):**
   - **Post-Sync Ingestion Hook:** CocoIndex file watcher / batch `sync_vault()` invokes debounced `compile_wiki(force=False)` upon ingestion settling.
   - **B-Tree Hub Splitting:** When node degree $> K_{\max} = 10$, executes deterministic SciPy Ward's HAC bisection and induces hypernym superclass via `temperature=0.1` LLM, rewiring `subClassOf` edges with $1.0 / 0.2$ conductance asymmetry ([`ADR-002`](file:///home/harri/yaw/docs/adr/ADR-002.md)).
   - **Compile-Time SHACL Validation:** Automatically invokes `validate_wiki_graph_fast` and persists dialectical contradictions into `compiled_violations` ([`ADR-011`](file:///home/harri/yaw/docs/adr/ADR-011.md)).
2. **2WikiMultiHopQA & MuSiQue Benchmark Evaluation Harness (Epic 8.2 / Sprint 8B):**
   - Ingest official multi-hop benchmark slices into `sample-llm-wiki/raw/benchmark/`.
   - Evaluation harness in `eval/run_benchmark.py` running official published evaluation scripts:
     - **Exact Match (EM)** and **Token F1** on final answer extraction.
     - **Paragraph Recall@k** ($k \in \{2, 5, 10\}$) for retrieved evidence chunks.
     - **Supporting Fact F1** and **Evidentiary Path F1** against ground-truth multi-hop reasoning graphs.
     - **Strict Invariant:** Evaluated strictly against published benchmark metrics (zero handrolled metrics).
   - Materialize deterministic read-only Markdown and Turtle pages into `vault/wiki/{namespace}/{slug}.md`.
3. **Remote FastMCP Daemon over Streaming HTTP/SSE (Epic 8.1):**
   - Standard FastMCP server exposes Server-Sent Events (SSE) streaming transport (`/sse` endpoint) paired with streaming HTTP POST execution endpoints (`/messages/`).
   - Remote LLM agents interact exclusively via MCP protocol framing, making zero assumptions about local shell execution, subshell spawning, or CLI availability on the client host.
4. **Non-Blocking Background Concurrency & Zero SQLite Busy Locks:**
   - Background file-watcher and ingestion threads operate concurrently with query serving.
   - SQLite concurrency architecture: WAL journal mode (`PRAGMA journal_mode = WAL`), non-blocking busy timeouts (`PRAGMA busy_timeout = 5000`), and connection pooling separating read-only query connections from dedicated write transactions.
   - Heavy ingestion bursts (chunking, embedding, EDC canonicalization) run asynchronously without causing `sqlite3.OperationalError: database is locked`, query latency spikes, or read stalls.
5. **Production Packaging & CI Matrix (Epic 8.3 / Sprint 8C):**
   - GitHub Actions CI workflow covering Python 3.10, 3.11, 3.12, and 3.13.
   - Zero-warning `ty` typecheck and `ruff` lint.
   - Sentrux architectural score $\ge 7200$.

### 10.2 Test Authoring Plan (TDD & Grounded Invariants)
- **Target Test Files:** `tests/test_fastmcp_contract.py`, `tests/test_btree_hac.py`, `tests/test_wiki_materializer.py`, `eval/run_benchmark.py`, `tests/test_e2e_launch.py`.
- **Invariant 1 (Substrate Mutation Isolation):**
  - Materialization writes strictly to `vault/wiki/`.
  - Assert zero file creation, modification, or timestamp mutation inside `vault/raw/`.
- **Invariant 2 (Valid Obsidian Frontmatter & Bidirectional Wikilinks):**
  - Assert generated files under `vault/wiki/{namespace}/{slug}.md` contain valid YAML frontmatter and connected relations rendered as standard `[[wikilinks]]`.
- **Invariant 3 (Idempotent Compilation & Orphan Pruning):**
  - Re-running `compile_wiki()` cleanly overwrites existing `vault/wiki/` pages without content duplication.
  - Removing a concept from the index cleanly prunes its orphaned `.md` file from `vault/wiki/`.
- **Invariant 4 (Dialectical Linter & Cached Violation Surfacing):**
  - Seed one ambiguous identity (a node with two unlinked `close_matches` targets) and two contradictory relations (`A causes B` vs `A prevents B`).
  - Assert `compile_wiki()` detects both conflicts and records them in `compiled_violations`.
  - Assert `lint_wiki()` retrieves both in $<5\text{ ms}$ without live pySHACL re-execution.
- **Invariant 5 (Benchmark Evaluation Rigor & Calibration Protocol):**
  - Execute evaluation harness over 2Wiki and MuSiQue benchmark slices using the official splits and official evaluation scripts.
  - **Calibration on dev only:** the uncalibrated priors of [`ADR-012`](file:///home/harri/yaw/docs/adr/ADR-012.md) (λ, $\beta_{1..3}$, `FACT_TOP_K`, τ) and the PPR damping α are tuned exclusively on a calibration split. The chosen values and the search grid are committed to `eval/` before any held-out run.
  - **Report on held-out data only:** headline numbers come from a single run on a held-out split with the committed configuration. Where official test labels are not public, the official dev set is partitioned once into disjoint calibration and held-out subsets by a committed seed, and the partition IDs are committed. No held-out result feeds back into configuration.
  - **Ablations reported alongside:** query-to-fact channel off ($\lambda = 1$), `closeMatch` edges off, and alias-prior channel off ($\lambda = 0$), so each ADR-012 decision has a measured effect.
- **Invariant 6 (FastMCP Streamable HTTP & SSE Transport End-to-End):**
  - Launch FastMCP daemon over loopback HTTP (`127.0.0.1:8765`) using SSE transport.
  - Connect remote FastMCP client over SSE `/sse` and message streaming `/messages/`.
  - Execute end-to-end client session: `ingest_raw` $\to$ `compile_wiki` $\to$ `query_wiki` $\to$ `lint_wiki`.
  - Assert all 4 production operations succeed with zero protocol framing errors and zero local CLI invocations.
- **Invariant 7 (Non-Blocking Concurrency & Zero SQLite Busy Locks):**
  - Run continuous high-throughput CocoIndex ingestion of 50 multi-chunk notes in background task.
  - Concurrently execute 30 parallel `query_wiki()` requests across multiple worker threads.
  - Assert 100% of queries succeed with zero `database is locked` operational errors and mean query latency $\le 50 \text{ ms}$.

### 10.3 Implementation Steps & File Locations
1. **`src/yaw/materialize/emitter.py` (New Module):**
   - Implement `WikiEmitter` that reads `unified_nodes` and `unified_edges`, formats YAML frontmatter and markdown wikilinks, and writes pages to `vault/wiki/`.
2. **`src/yaw/server/daemon.py`:**
   - Implement production FastMCP endpoints:
     - `@mcp.tool() async def compile_wiki(force: bool = False) -> dict`
     - `@mcp.tool() async def lint_wiki(query: str | None = None) -> list[dict]`
     - `@mcp.tool() async def ingest_raw(content: str, title: str | None = None, url: str | None = None) -> dict`
     - `@mcp.tool() async def query_wiki(query: str, top_k: int = 3) -> str`
   - Configure SSE streaming transport and non-blocking background task runner.
3. **`src/yaw/indexer/pipeline.py` & `src/yaw/storage/db.py`:**
   - Optimize connection concurrency: ensure dedicated read connections with `PRAGMA query_only = ON` for search, and serialized write transactions for indexing.
4. **Repository Packaging & Release:**
   - Configure `pyproject.toml` console scripts (`yaw = "yaw:main"`).
   - Add `.github/workflows/ci.yml` running `uv run ruff check`, `uv run ty check`, and `uv run pytest`.
   - Update `README.md` with architectural diagram, FastMCP client configuration snippet, and quickstart.

### 10.4 Verification & Quality Gates
- Full regression test suite: 100% pass across all test modules (`tests/`).
- `sentrux gate .`: assert Sentrux score $\ge 7200$ with 0 circular dependencies, 0 god files, and 0 regression warnings.
- Public GitHub release ready tagged `v1.0.0`.
