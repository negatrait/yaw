# FOUNDATIONAL.md: Product Requirements & Architectural Specification for YAW

**Document Status:** Authoritative Root Document & Single Source of Truth (SSOT)  
**System Name:** YAW (Yet Another Wiki / FastMCP Ontology GraphRAG Engine)  
**Current State:** Epic 6.2 Complete (375 passing tests, 0 lint warnings, Sentrux quality 7270, 0 circular dependencies)  
**Target Audience:** Autonomous Coding Agents & Core Systems Engineers  

---

## 0. Progressive Disclosure Architecture & Reading Guide

`FOUNDATIONAL.md` is the authoritative **Root Document and Single Source of Truth (SSOT)** for YAW. Every engineering epic, architectural decision, test invariant, and operational contract is a direct, traceable derivative of this document.

To navigate the system efficiently without cognitive thrashing or context decay, follow this progressive disclosure hierarchy:

### 1. Core Architectural Canon, Governance & Master Roadmaps
| System Layer | Document Link | Scope & Purpose |
| :--- | :--- | :--- |
| **System Overview & Invariants** | [`FOUNDATIONAL.md`](file:///home/harri/yaw/FOUNDATIONAL.md) (This Document) | Root SSOT: Vision, 7 Invariants, Model Stack, Single-Stage DAG, SQLite Contracts, SciPy PPR Algorithms. |
| **Architectural Decision Records** | [`docs/ADR.md`](file:///home/harri/yaw/docs/ADR.md) | Formal ADR ledger ([`ADR-001`](file:///home/harri/yaw/docs/adr/ADR-001.md) through [`ADR-012`](file:///home/harri/yaw/docs/adr/ADR-012.md)). |
| **Engineering Implementation Roadmap** | [`docs/epics-alignment-roadmap.md`](file:///home/harri/yaw/docs/epics-alignment-roadmap.md) | Authoritative master plan (Epics 1–7.6 complete; Epics 7.7, 8.1–8.3 specified), quality gates. |
| **Executable Acceptance Criteria** | [`docs/USER_STORIES.md`](file:///home/harri/yaw/docs/USER_STORIES.md) | Formal Gherkin / BDD user stories for Epics 5, 6, 7, and 8. |
| **LLM-Wiki Foundational Pattern** | [`docs/karpathy-llm-wiki.md`](file:///home/harri/yaw/docs/karpathy-llm-wiki.md) | Andrej Karpathy's primary source defining the 3-layer persistent compounding PKM pattern. |
| **Architecture Plan & Reference** | [`docs/yaw-architecture-plan.md`](file:///home/harri/yaw/docs/yaw-architecture-plan.md) | Consolidated 4-tier semantic stack reference and architectural synthesis. |

### 2. Retrieval, Graph Diffusion & GraphRAG Algorithms
| Algorithm Layer | Document Link | Scope & Purpose |
| :--- | :--- | :--- |
| **Power PageRank Query Engine** | [`docs/power-pagerank.md`](file:///home/harri/yaw/docs/power-pagerank.md) | Query-time SciPy CSR matrix PageRank diffusion, directional conductance, hub dampening, Bayesian evidence chunk rescoring. |
| **Personalized PageRank Survey** | [`docs/efficient-ppr-algorithms.md`](file:///home/harri/yaw/docs/efficient-ppr-algorithms.md) | Comprehensive algorithmic survey; mathematical derivation of multi-modal candidate prior seed vectors ($\mathbf{p}_0$). |
| **Geodesic Traversal & Linear Solvers** | [`docs/scipy-graphrag-extensions.md`](file:///home/harri/yaw/docs/scipy-graphrag-extensions.md) | Multi-hop geodesic path traces via `scipy.sparse.csgraph.shortest_path` and exact linear solvers (`spsolve`). |
| **Hierarchical Deepening & Waveguides** | [`docs/ontological-roots.md`](file:///home/harri/yaw/docs/ontological-roots.md) | Mathematical justification for B-Tree Ward HAC hub splitting, hypernym induction, and spectral energy concentration. |
| **Key Retrieval ADRs** | [`docs/adr/ADR-002.md`](file:///home/harri/yaw/docs/adr/ADR-002.md)<br>[`docs/adr/ADR-010.md`](file:///home/harri/yaw/docs/adr/ADR-010.md)<br>[`docs/adr/ADR-012.md`](file:///home/harri/yaw/docs/adr/ADR-012.md) | ADR-002 (B-Tree Hub Splitting); ADR-010 (Candidate Priors & Topological Disambiguation); ADR-012 (Query-to-Fact Seeding, storage-backed $P(e \mid a)$). |

### 3. W3C Semantic Web, Ontologies & Closed-World SHACL
| Semantic Subsystem | Document Link | Scope & Purpose |
| :--- | :--- | :--- |
| **RDF, OWL & SHACL Manual** | [`docs/rdf-owl-shacl.md`](file:///home/harri/yaw/docs/rdf-owl-shacl.md) | 4-tier semantic stack manual: RDF named quads, OWL 2 RL inference, SKOS concept schemes, pySHACL validation. |
| **Pluggable Ontology Substrates** | [`docs/epic7-pluggable-ontology-integration-spec.md`](file:///home/harri/yaw/docs/epic7-pluggable-ontology-integration-spec.md) | Complete specification for pluggable external OWL/SKOS domain models, 2x2 extraction matrix, and antonym masks. |
| **Key Semantic Web ADRs** | [`docs/adr/ADR-005.md`](file:///home/harri/yaw/docs/adr/ADR-005.md)<br>[`docs/adr/ADR-008.md`](file:///home/harri/yaw/docs/adr/ADR-008.md)<br>[`docs/adr/ADR-009.md`](file:///home/harri/yaw/docs/adr/ADR-009.md)<br>[`docs/adr/ADR-011.md`](file:///home/harri/yaw/docs/adr/ADR-011.md) | ADR-005 (9 Universal Roots); ADR-008 (W3C Alignment); ADR-009 (Pluggable Substrates); ADR-011 (Compile-Time SHACL & $\mathcal{O}(1)$ Contradiction Cache). |

### 4. NLP Pipeline, Ingestion & Single-Stage Extraction
| NLP / Ingestion Layer | Document Link | Scope & Purpose |
| :--- | :--- | :--- |
| **Open Information Extraction (Open IE)** | [`docs/open-ie-authoritative-source.md`](file:///home/harri/yaw/docs/open-ie-authoritative-source.md) | Academic research foundation establishing open-vocabulary relation extraction without schema blindness. |
| **spaCy-LLM Integration Patterns** | [`docs/how-to-handle-spacy-llm.md`](file:///home/harri/yaw/docs/how-to-handle-spacy-llm.md) | Jinja2 prompt authoring, zero-monkeypatching registry contracts, and model/task pipelines. |
| **Zero-Shot GLiNER Extraction** | [`docs/how-to-handle-spacy-gliner.md`](file:///home/harri/yaw/docs/how-to-handle-spacy-gliner.md) | Architectural alternative and evaluation for zero-shot bi-encoder entity span extraction. |
| **spaCy-LLM Architecture & APIs** | [`docs/spacy-llm-api.md`](file:///home/harri/yaw/docs/spacy-llm-api.md)<br>[`docs/spacy-llm-docs.md`](file:///home/harri/yaw/docs/spacy-llm-docs.md) | Official architectural documentation and component API references for `spacy-llm`. |
| **CocoIndex Incremental Engine** | [`docs/cocoindex.md`](file:///home/harri/yaw/docs/cocoindex.md) | CocoIndex declarative dataflow engine, Rust filesystem watcher, and incremental sync manual. |
| **In-Process SQLite vs Neo4j** | [`docs/cocoindex-neo4j.md`](file:///home/harri/yaw/docs/cocoindex-neo4j.md) | Architectural evaluation establishing in-process SQLite and SciPy superiority over external graph daemons. |
| **Key Ingestion ADRs** | [`docs/adr/ADR-001.md`](file:///home/harri/yaw/docs/adr/ADR-001.md)<br>[`docs/adr/ADR-003.md`](file:///home/harri/yaw/docs/adr/ADR-003.md)<br>[`docs/adr/ADR-004.md`](file:///home/harri/yaw/docs/adr/ADR-004.md)<br>[`docs/adr/ADR-006.md`](file:///home/harri/yaw/docs/adr/ADR-006.md)<br>[`docs/adr/ADR-012.md`](file:///home/harri/yaw/docs/adr/ADR-012.md) | ADR-001 (Single-Pass Triples); ADR-003 (Two-Tier Canonicalization, Tier 2 amended by ADR-012); ADR-004 (Embedding Boundary, amended by ADR-012); ADR-006 (spaCy Token Lemmatization); ADR-012 (Representation-Space Separation & Corroborated Identity Resolution). |

### 5. Testing Rigor, Verification & Test Infrastructure
| Infrastructure Layer | Document Link | Scope & Purpose |
| :--- | :--- | :--- |
| **End-to-End Test Infrastructure** | [`TEST_INFRA.md`](file:///home/harri/yaw/TEST_INFRA.md) | Unmocked test philosophy, 2x2 matrix coverage, and 7 Invariant verification matrix. |
| **Test Readiness & Verification Matrix** | [`TEST_READY.md`](file:///home/harri/yaw/TEST_READY.md) | Milestone readiness audits and verification pass/fail status across test suites. |
| **Sprint 5.1 Remediation & Alignment** | [`docs/alignment-sprint-for-e5s2.md`](file:///home/harri/yaw/docs/alignment-sprint-for-e5s2.md) | Forensic audit, remediation actions, and TDD specifications for quad alignment. |
| **Loopback Network Isolation ADR** | [`docs/adr/ADR-007.md`](file:///home/harri/yaw/docs/adr/ADR-007.md) | In-process loopback HTTP network isolation for fast, hermetic offline test suites. |

### 6. Historical & Superseded Architecture Specifications
| Document Link | Original Scope | Reason for Supercession |
| :--- | :--- | :--- |
| [`docs/llm-wiki-fts-vector-graph-playbook.md`](file:///home/harri/yaw/docs/llm-wiki-fts-vector-graph-playbook.md) | Prototype GraphRAG playbook | Formalized in [`FOUNDATIONAL.md §5.2`](file:///home/harri/yaw/FOUNDATIONAL.md) and [`docs/power-pagerank.md`](file:///home/harri/yaw/docs/power-pagerank.md). |
| [`docs/llm-wiki-spec-v2.md`](file:///home/harri/yaw/docs/llm-wiki-spec-v2.md) | Architecture Spec v2.0 prototype | Superseded by 7 Invariants and Single-Stage Continuous Ingestion in `FOUNDATIONAL.md`. |
| [`docs/llm-wiki-spec-addendum.md`](file:///home/harri/yaw/docs/llm-wiki-spec-addendum.md) | Medallion Two-Stage Architecture | Purged two-stage barrier (`Bronze -> Silver -> Gold`) in favor of Single-Stage Continuous Ingestion. |

---

## 1. Vision & Core Philosophy

### What is YAW?
**YAW** is an autonomous, local-first knowledge compiler and associative retrieval engine. It operates directly over an Obsidian markdown vault, using **CocoIndex**, **spaCy + spacy-llm**, **SQLite** (`fts5` + `sqlite-vec`), and **SciPy** (Personalized PageRank through an emergent ontology layer) served via **FastMCP**.

### The Problem It Solves
1. **Semantic Blindness of Naive RAG:** Naive chunk-based vector search retrieves isolated fragments based purely on surface vocabulary, completely missing macro-level conceptual architecture, cross-document causality, and domain invariants.
2. **The Fragility of Manual Knowledge Bases:** Human note-takers cannot maintain dense, cross-cutting taxonomic hierarchies across hundreds of documents. We cannot trust the human to be an expert ontology engineer.
3. **The Overhead of External Graph Databases:** Cloud or multi-process graph engines (Neo4j, Memgraph, PostgreSQL) introduce severe operational complexity, background daemon bloat, network latency, and synchronization failures.

### The Single-Stage Continuous Solution
Previous architectures introduced severe friction by splitting ingestion into an artificial two-stage barrier: raw leaf staging in Stage 1, followed by delayed, heavy batch clustering and LLM synthesis passes in Stage 2 before the graph could even be queried.

YAW eliminates this barrier through **Single-Stage Continuous Ingestion**:
- Notes in `vault/raw/` are parsed and chunked in a single pipeline pass.
- Leaf ontologies (entities, definitions, domain concepts, and relational triples) are extracted concurrently via **`spacy-llm`**.
- Cross-document canonicalization occurs in-process via a **Two-Tier Identity Protocol** ([`docs/adr/ADR-003.md`](file:///home/harri/yaw/docs/adr/ADR-003.md), [`docs/adr/ADR-012.md`](file:///home/harri/yaw/docs/adr/ADR-012.md)): Tier 1 decides deterministically, Tier 2 proposes and merges only on linguistic corroboration.
- Dynamic SQLite views ([`unified_nodes`](file:///home/harri/yaw/src/yaw/storage/db.py#L170-L200), [`unified_edges`](file:///home/harri/yaw/src/yaw/storage/db.py#L85-L168)) instantly expose an interconnected, W3C-aligned RDF named quad graph ([`docs/adr/ADR-008.md`](file:///home/harri/yaw/docs/adr/ADR-008.md)), anchored to **9 Universal Root Namespaces** ([`docs/adr/ADR-005.md`](file:///home/harri/yaw/docs/adr/ADR-005.md)).
- Retrieval executes immediately via **SciPy Personalized PageRank (PPR)** over the unified graph without waiting for batch consolidation passes.
- Materialization of compiled wiki pages into `vault/wiki/` is an optional, decoupled, asynchronous export (`dist/`) generated during compilation.

---

## 2. The Seven Invariants of YAW

Every line of code and test in YAW is governed by seven non-negotiable invariants:

### Invariant 1: Unidirectional Substrate Mutation
- The `vault/raw/` directory is strictly **append-only and human-authoritative**. The engine never modifies, overwrites, or deletes files in `vault/raw/`.
- All automated synthesis, compiled wiki pages, and human audit trails are cleanly segregated into `vault/wiki/` (the compiled distribution target, `dist/`).
- B-Tree hypernym induction and cluster splits ([`docs/adr/ADR-002.md`](file:///home/harri/yaw/docs/adr/ADR-002.md)) persist in engine-owned tables (`compiled_nodes`, `compiled_relations`) without mutating raw extraction tables (`leaf_nodes`, `leaf_relations`).
- **Lossless Ingestion Invariant across 2x2 Extraction Matrix ([`docs/adr/ADR-009.md`](file:///home/harri/yaw/docs/adr/ADR-009.md)):** `leaf_nodes` and `leaf_relations` are append-only and strictly lossless. Toggling between internal/external ontologies or enabling/disabling `open_ie` is a pure read-time dynamic view filter (`unified_edges`). Switching extraction quadrants never purges raw triples, drops out-of-vocabulary relations, or wipes CocoIndex content-hash / EmbeddingGemma caches.

### Invariant 2: Zero-Prose Disk Pointers
- SQLite tables store **1-indexed line pointers** (`line_start`, `line_end`) into raw files rather than duplicating raw text payloads.
- FTS5 operates in `contentless_delete=1` mode to prevent duplicate text storage while providing full-text search indexing.
- Winning retrieval chunks hydrate directly on demand from disk via 1-indexed line slicing ([`hydrate_disk_slice()`](file:///home/harri/yaw/src/yaw/retrieval/retriever.py)), guaranteeing zero text drift and zero prose redundancy.

### Invariant 3: Single In-Process SQLite Database
- Exactly **one** SQLite database (`.cache/yaw.db`) houses all state ([`docs/adr/ADR-004.md`](file:///home/harri/yaw/docs/adr/ADR-004.md)).
- There are no separate databases for chunks, vectors, and graphs.
- Zero external graph servers or background daemons (Neo4j, Blazegraph, Virtuoso, Jena): SQLite relational tables and views act as the complete triple/quad store and vector store in-process ([`docs/rdf-owl-shacl.md`](file:///home/harri/yaw/docs/rdf-owl-shacl.md)). All RDF serialization (`rdflib`) and SHACL validation (`pyshacl`) execute in-process against this single database.
- Concurrency and reliability are enforced by SQLite engine PRAGMAs:
  ```sql
  PRAGMA journal_mode = WAL;
  PRAGMA synchronous = NORMAL;
  PRAGMA busy_timeout = 60000;
  ```

### Invariant 4: Machine Semantics Equal Human Wikilinks
- Machine-extracted semantic relations are just as important as human-authored Obsidian wikilinks. We cannot trust humans to be expert ontology engineers.
- In the retrieval matrix, human `[[wikilinks]]` and machine-extracted semantic triples (`subject -> predicate -> object`) are first-class peers with equal baseline weight ($1.0$), aligned to W3C standards ([`docs/adr/ADR-008.md`](file:///home/harri/yaw/docs/adr/ADR-008.md)).
- **Probabilistic Alias Distributions & Anti-Winner-Take-All Grounding**: Human note text exhibits high lexical entropy and polysemy (e.g., surface mentions like *"Transformer"*, *"Mercury"*, or *"GraphRAG"*). Discretizing such mentions into a single canonical node at ingestion time via greedy "winner-take-all" heuristic collapses vital semantic uncertainty. Instead, YAW anchors lexical mentions to ontological nodes via a candidate distribution $P(\text{entity } e \mid \text{alias } a) \in (0, 1]$ (normalized evidence weights, not calibrated probabilities) maintained natively in spaCy's compiled C/Cython `InMemoryLookupKB` and backed by the SQLite `aliases` table (hard identity) and `close_matches` table (soft identity candidates, [`docs/adr/ADR-012.md`](file:///home/harri/yaw/docs/adr/ADR-012.md)). This preserves multiple epistemic interpretations simultaneously without premature hard clustering, deferring disambiguation to global graph topology at retrieval time.

### Invariant 5: Dual Epistemic Retrieval Contract
- Retrieval must never return isolated chunk fragments without context, nor abstract theories without evidence.
- Every retrieval query explains **both**:
  1. **The Ontological Context:** Active concept nodes, entity paths, and discovered semantic triples explaining *why* topics relate (the Bridging Subgraph).
  2. **The Grounded Evidence:** Direct hits to explicit raw chunks hydrated from disk with exact byte and line citations.
- **Bayesian Epistemic Bridging & Topological Disambiguation**: The formal epistemic bridge connecting concrete textual evidence chunks to ontological universals is evaluated by fusing candidate alias priors $P(e \mid a)$ with spectral Personalized PageRank (PPR) random walk diffusion $\mathbf{p}^*[e]$ and chunk dense vector similarity $\text{sim}(q, c)$. Instead of ungrounded LLM inference, polysemous mentions are resolved topologically by propagating probability mass across the collective graph context, ensuring every retrieved assertion is mathematically grounded in disk evidence.

### Invariant 6: Real-Invariant Testing (Zero Tautological Cushion Mocks)
- Tests must verify real invariants against real disk files, real SQLite instances, real `sqlite-vec` binary operations, and real SciPy sparse matrices.
- Never mock internal subsystem interfaces or database connections. Tautological cushion mocks (mocking queries or matrix math) test only the mock definitions and conceal fatal structural bugs.
- Boundary isolation mocking is strictly confined to external remote network services (the local Ollama HTTP server, mocked via a fast in-process loopback HTTP server in [`tests/conftest.py`](file:///home/harri/yaw/tests/conftest.py) per [`docs/adr/ADR-007.md`](file:///home/harri/yaw/docs/adr/ADR-007.md)).
- Test architecture, 2x2 matrix coverage, and grounded invariant verification contracts are codified in [`TEST_INFRA.md`](file:///home/harri/yaw/TEST_INFRA.md) and tracked in [`TEST_READY.md`](file:///home/harri/yaw/TEST_READY.md).

### Invariant 7: Code Cleanliness, Strict Formatting & Single Source of Truth
- Code must adhere to Ruff formatting standards (`uv run ruff format --check`) and clean static lint/type analysis via Ruff and Ty (`uv run ruff check`, `uv run ty check`) with zero unused imports, explicit typing, and zero circular dependencies.
- Architecture adheres to strict Single Source of Truth (SSOT): schema definitions, PRAGMAs, and configurations are declared once in [`src/yaw/storage/db.py`](file:///home/harri/yaw/src/yaw/storage/db.py) or [`src/yaw/config.py`](file:///home/harri/yaw/src/yaw/config.py) and imported everywhere, never copy-pasted or shadowed.

---

## 3. The Model Stack Contract

To eliminate hallucinations, non-determinism, and memory bloat, YAW is bound to a strict local model contract served via Ollama and in-process spaCy:

| Role | Model / Mechanism | Endpoint / Execution | Task Format / Prefix / Contract |
| :--- | :--- | :--- | :--- |
| **Retrieval Document Embedding (space R)** | `embed-gemma-300m-FLM` | `/api/embed` | `title: {note_title \| none} \| text: {content}` — note chunks (`note_chunks.embedding`) and rendered facts (`fact_vec`) |
| **Retrieval Query Embedding (space R)** | `embed-gemma-300m-FLM` | `/api/embed` | `task: search result \| query: {query}` — query → chunk and query → fact relevance |
| **Entity Similarity Embedding (space S)** | `embed-gemma-300m-FLM` | `/api/embed` | `task: sentence similarity \| query: {name}: {definition}` — Tier 2 identity **candidate generation only** (`entity_vec`); never used for retrieval |
| **Bi-Encoder Reranker** | `embed-gemma-300m-FLM` | In-Memory Dot Product | Re-scores candidate chunk vectors (space R) against the query vector; zero redundant re-embedding |
| **Linguistic Canonicalization (space L)** | `en_core_web_lg` pipeline (lemmatizer, parser, NER, polarity mask; static 300d vectors only as the guarded last predicate tier) + `InMemoryLookupKB` | In-Process Cython | Predicate canonicalization, Tier 1 normalized keys, Tier 2 corroboration, query mention extraction; alias priors $P(e \mid a)$; zero LLM adjudication ([`docs/adr/ADR-003.md`](file:///home/harri/yaw/docs/adr/ADR-003.md), [`docs/adr/ADR-010.md`](file:///home/harri/yaw/docs/adr/ADR-010.md), [`docs/adr/ADR-012.md`](file:///home/harri/yaw/docs/adr/ADR-012.md)) |
| **Open IE Triplet Extraction & Typing** | `Qwen3-4B-Instruct-2507-GGUF` | `/api/chat` via `spacy.Raw.v1` | Single-pass schema-free Open IE (`OPEN_IE_RAW_TRIPLES_TEMPLATE`; [`docs/adr/ADR-001.md`](file:///home/harri/yaw/docs/adr/ADR-001.md)) with `spacy.BatchCache.v1` disk caching |
| **Node Definition Generation** | `Qwen3-4B-Instruct-2507-GGUF` | `/api/chat` | Off-the-shelf `spacy.Summarization.v1` (`field="description"`, `max_n_words=50` via `config.DEFINITION_MAX_WORDS`); `vault/wiki/` pages materialize deterministically from SQLite |

### Representation-Space Contract ([`docs/adr/ADR-012.md`](file:///home/harri/yaw/docs/adr/ADR-012.md))
Representations are assigned **per job, never per model**, because the three jobs have different correctness criteria:

| Space | Job | Correctness criterion | Why this representation |
| :--- | :--- | :--- | :--- |
| **L: Linguistic** (spaCy) | Predicate canonicalization, Tier 1 normalized keys, Tier 2 corroboration, query mentions | Meaning and **polarity** | Distributional vectors place antonyms close together ([Mrkšić et al., 2016](https://aclanthology.org/N16-1018/)); lemmas, dependency heads and negation carry the semantics. |
| **S: Symmetric similarity** (Gemma, `sentence similarity` prompt) | Entity-identity **candidate generation** | **Recall** of duplicate candidates; never the merge decision alone | Similarity measures relatedness, not identity ([Hill et al., 2015](https://aclanthology.org/J15-4004/)); the model card marks this prompt *"not intended for retrieval use cases"* ([EmbeddingGemma](https://ai.google.dev/gemma/docs/embeddinggemma/model_card)). |
| **R: Asymmetric retrieval** (Gemma, query ↔ document prompts) | Query → chunk and query → **fact** relevance | Ranking relevance to a natural-language question | Query-to-triple linking beats NER-to-node by 12.5% Recall@5; query-to-node gives no gain ([HippoRAG 2 §6.1](https://arxiv.org/abs/2502.14802)). |

- **One space per vector store.** New vector stores are `sqlite-vec` `vec0` tables (`entity_vec`, `fact_vec`), which reject dimension-mismatched inserts. Similarity is computed only within one space using its prescribed prompt pairing. Cross-space comparison is a defect, never a fallback.
- **`InMemoryLookupKB` is the alias → candidate prior index, not a similarity source.** No code path computes cosine against KB entity vectors.
- **Calibrated Alias Prior Distribution ($P(e \mid a)$)**: for a mention node $a$ with hard canonical $\hat{a}$, $\text{cands}(a) = \{\hat{a}\} \cup \text{closeMatch}(\hat{a})$, with $P(e \mid a) = w(a, e) / \sum_{e'} w(a, e')$, $w(a, \hat{a}) = 1$, and $w(a, e) = $ `close_matches.score`. Persisted in SQLite and mirrored into the KB via `kb.add_alias(alias, entities, probabilities)` ([`docs/adr/ADR-010.md`](file:///home/harri/yaw/docs/adr/ADR-010.md), [`docs/adr/ADR-012.md`](file:///home/harri/yaw/docs/adr/ADR-012.md)).
- **Zero LLM Adjudication**: 0 generative LLM calls during canonicalization and retrieval. EmbeddingGemma encoder calls are batched: one space-S embedding per **new** canonical entity, and one space-R embedding per chunk and per fact. Predicates trigger **zero** embedding calls.
- **Two-Tier Identity Protocol: Tier 1 decides, Tier 2 proposes** ([`docs/adr/ADR-003.md`](file:///home/harri/yaw/docs/adr/ADR-003.md), [`docs/adr/ADR-012.md`](file:///home/harri/yaw/docs/adr/ADR-012.md)):
  - **Tier 1 (Deterministic, space L)**: exact canonical id, **normalized key** (NFKC + casefold, leading determiner and possessive clitic removed, `NOUN` lemmatized / `PROPN` verbatim, punctuation and whitespace collapsed), active alias, or ontology label/altLabel alias.
  - **Tier 2 (Blocked + Corroborated, spaces S + L)**:
    - **Blocking:** `vec0` KNN (`k = CANON_TOP_K`, default 5) within the mention's namespace partition plus active ontology partitions; keep cosine $\ge \tau$ (`CANON_SIM_THRESHOLD`).
    - **Matching:** a candidate is a merge candidate only if a deterministic linguistic corroboration also holds: acronym, head-anchored containment (shared `Span.root`), or initial expansion.
    - **Exactly one merge candidate:** hard merge (active alias, `owl:sameAs`).
    - **Otherwise:** soft `closeMatch` links weighted by cosine, with no merge. Soft links follow the HippoRAG synonymy-edge design, which has its largest measured effect on 2WikiMultiHopQA ([HippoRAG](https://arxiv.org/abs/2405.14831)).
  - **Predicates never enter Tier 2**: lemma → synset/alias dictionary → base predicate → polarity-guarded static-vector fallback → open `predicates:{lemma-slug}`.
  - **Induction**: novel canonical node creation (`TierEnum.NEW_INDUCTION`).
- **Native Prompt-Response Disk Caching**: `spacy.BatchCache.v1` is declared and mounted as the native disk cache for all LLM calls (`path = ".cache/spacy_llm"`), ensuring automatic disk caching of prompt-response pairs without redundant network calls.

---

## 4. System Components & Architecture

```txt
═══════════════════════════════════════════════════════════════════════════════════════════════════
                         YAW SINGLE-STAGE ARCHITECTURE & DATAFLOW
═══════════════════════════════════════════════════════════════════════════════════════════════════

  [ vault/raw/*.md ] (Append-Only Markdown Notes)
          │
          ▼
  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ 1. SINGLE-STAGE INGESTION & LEAF ONTOLOGY EXTRACTION (src/yaw/indexer/ & src/yaw/nlp/)              │
  │ • CocoIndex Watcher: Rust core change detection (internal content-hash deltas)                      │
  │ • Markdown Chunker: Deterministic boundary splitting, byte offsets & line ranges                    │
  │ • Representation Spaces (ADR-012): L = spaCy linguistics, S = Gemma similarity, R = Gemma retrieval │
  │ • Space R: chunks + rendered facts embedded with document prompt (batched /api/embed)               │
  │ • Native Disk Cache: spacy.BatchCache.v1 at .cache/spacy_llm for all LLM calls                      │
  │ • Markdown Structure: code spans in doc.spans["code"]; text never mutated                           │
  │ • Unified spacy-llm Pipeline (Qwen3-4B via yaw.Ollama.v1 with spacy.Raw.v1):                        │
  │   - OPEN_IE_RAW_TRIPLES_TEMPLATE: Single-pass (s,p,o) schema-free extraction                        │
  │   - Zero embedding calls for predicates; one space-S embedding per new canonical entity             │
  │ • Compiled C/Cython InMemoryLookupKB: Alias candidate distribution with priors P(e|a)               │
  │   - Built from aliases (hard identity) + close_matches (soft candidates) in SQLite                  │
  │ • In-Process Two-Tier Identity Protocol (Tier 1 decides, Tier 2 proposes):                          │
  │   - Tier 1: exact id / spaCy normalized key / active alias / ontology label (space L)               │
  │   - Tier 2: vec0 KNN block (space S) + linguistic corroboration (space L)                           │
  │       unique corroborated candidate -> hard merge; otherwise -> closeMatch soft links               │
  │   - Induction: Novel canonical node creation (0 LLM adjudication)                                   │
  │ • Predicate Normalization: lemma + synset/alias dictionary + polarity-guarded vector tier           │
  │ • Node & Wiki Summaries: spacy.Summarization.v1 (field="description")                               │
  └──────────────────────────────────────┬──────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ 2. STORAGE CONTRACTS & SQLITE QUAD SUBSTRATE (src/yaw/storage/db.py)                                │
  │ • Single SQLite DB at sample-llm-wiki/.cache/yaw.db (WAL mode, busy_timeout=60000ms)                │
  │ • Physical Extraction Tables: note_meta, note_chunks, wikilinks, leaf_nodes, leaf_relations         │
  │ • Engine-Owned Tables: aliases, close_matches, canon_verdicts, compiled_nodes, compiled_relations   │
  │ • Contentless FTS5: query_fts (contentless_delete=1) indexing headings and chunks                   │
  │ • sqlite-vec: note_chunks.embedding (R), fact_vec (R, vec0), entity_vec (S, vec0 by namespace)      │
  │ • Dynamic Reversible Views:                                                                         │
  │   - unified_edges: 5-column named quad projection (src_id, dst_id, rel_type, weight, chunk)         │
  │   - unified_nodes: Canonical node projections with namespace inheritance                            │
  └──────────────────────────────────────┬──────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ 3. GRAPHRAG RETRIEVAL & SCI-PY DIFFUSION ENGINE (src/yaw/retrieval/retriever.py)                    │
  │ • Probabilistic Seed: p0 = beta1*s_concepts + beta2*s_chunks_vec + beta3*s_chunks_fts               │
  │   - Concept Seeding: lambda*alias priors sum_a P(e|a) + (1-lambda)*query-to-fact r(q, e)            │
  │ • Continuous Math: offline (q=0) -> r = 0, s_chunks_vec = 0; alias + FTS5 seeds remain              │
  │ • SciPy Sparse CSR Matrix: In-memory row-stochastic transition matrix P                             │
  │ • Hub Dampening: In-degree power scaling (gamma = 0.5)                                              │
  │ • Conductance Asymmetry: 1.0 downward broadcast, 0.2 upward damping (ADR-002, Sprint 5.1)           │
  │ • Topological Disambiguation: Random walk diffusion p^(t+1) = alpha*p^(t)*P + (1-alpha)*p0          │
  │ • Bayesian Evidence Scoring: Score(c) = (sum_{e in c} P(e|a)*p*[e] + p*[c]) * sim(q, c)             │
  │ • Grounding: Zero-copy disk hydration via byte offsets (file.seek, file.read)                       │
  │ • Payload: Dual-layer Bridging Subgraph (Ontological Context + Grounded Evidence)                   │
  └──────────────────────────────────────┬──────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ 4. FASTMCP PRODUCTION INTERFACE (src/yaw/server/daemon.py)                                          │
  │ • query_wiki: Full GraphRAG associative retrieval returning dual-layer Bridging Subgraph            │
  │ • ingest_raw: Immutable append-only note ingestion into vault/raw/                                  │
  │ • compile_wiki: B-Tree HAC splitting, hypernym induction, fast SHACL (<200ms), SQLite cache         │
  │ • lint_wiki: O(1) SQLite read of compiled contradictions, topic queries, live force re-lint         │
  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Component Deep Dive & Business Logic

### Component 1: Single-Stage Ingestion & Leaf Ontology (`src/yaw/indexer/` & `src/yaw/nlp/`)
1. **Delta Tracking:** CocoIndex's Rust core filesystem watcher ([`docs/cocoindex.md`](file:///home/harri/yaw/docs/cocoindex.md)) monitors `vault/raw/` recursively. File content hashes are maintained internally by CocoIndex; no manual document hash column is required in `note_meta`.
2. **Deterministic Chunking:** Documents are split along semantic boundaries (default ~1000 characters with 150 overlap). Each chunk records:
   - `chunk_id`: sha256 digest of heading + content.
   - `line_start` & `line_end`: 1-based human-readable line ranges for direct disk hydration.
3. **Embedding (space R, [`docs/adr/ADR-012.md`](file:///home/harri/yaw/docs/adr/ADR-012.md)):** Chunks are embedded using `embed-gemma-300m-FLM` with the document prompt and note title:
   `title: {note_title} | text: {chunk_text}`.
   After extraction (step 5), every `leaf_relations` fact is embedded in the **same batched call** with `title: {note_title} | text: {subject_name} {surface_predicate} {object_name}` into `fact_vec`. Fact text is rendered deterministically and never stored (Invariant 2).
4. **Wikilink Extraction:** Regex parser extracts Obsidian `[[wikilinks]]`, mapping them to `wikilinks` records with source and target chunk pointers.
5. **Open IE Extraction & Typing:**
   - Single-pass Open IE ([`docs/adr/ADR-001.md`](file:///home/harri/yaw/docs/adr/ADR-001.md), [`docs/open-ie-authoritative-source.md`](file:///home/harri/yaw/docs/open-ie-authoritative-source.md), [`docs/how-to-handle-spacy-llm.md`](file:///home/harri/yaw/docs/how-to-handle-spacy-llm.md)) extracts relational triples `(subject, predicate, object)`.
   - Freeform predicates are decamelized and lemmatized via native spaCy `Token.lemma_` ([`docs/adr/ADR-006.md`](file:///home/harri/yaw/docs/adr/ADR-006.md)).
   - Predicates align with the **14 Canonical Base Predicates** ([`docs/adr/ADR-008.md`](file:///home/harri/yaw/docs/adr/ADR-008.md)):
     - *Subsumption & Mereology:* `subClassOf`, `partOf`, `instanceOf`
     - *Dialectical & Argumentation:* `supports`, `refutes`, `contradicts`, `supersedes`
     - *Dependency & Process:* `dependsOn`, `causes`, `precedes`, `produces`
     - *Provenance & Grounding:* `derivedFrom`, `cites`, `manifestsIn`
   - Entities are typed into **9 Universal Root Namespaces** ([`docs/adr/ADR-005.md`](file:///home/harri/yaw/docs/adr/ADR-005.md)):
     `entities`, `concepts`, `predicates`, `invariants`, `decisions`, `claims`, `processes`, `sources`, `contexts`.
6. **In-Memory KB Synchronization & Candidate Prior Distribution ([`docs/adr/ADR-010.md`](file:///home/harri/yaw/docs/adr/ADR-010.md), [`docs/adr/ADR-012.md`](file:///home/harri/yaw/docs/adr/ADR-012.md)):**
   - The engine synchronizes an in-process, Cython-compiled `InMemoryLookupKB` from SQLite (`sync_kb_from_db(kb, conn)`) as an **alias → candidate prior index** (never a similarity source):
     - Registers canonical node ids, base predicates (`config.BASE_PREDICATES`) and `leaf_relations` predicates as candidate identities.
     - For each mention node $a$ with hard canonical $\hat{a}$ (from `aliases`), registers $\text{cands}(a) = \{\hat{a}\} \cup \text{closeMatch}(\hat{a})$ via `InMemoryLookupKB.add_alias(alias, entities, probabilities)`, with
       $$P(e \mid a) = \frac{w(a, e)}{\sum_{e'} w(a, e')}, \quad w(a, \hat{a}) = 1, \quad w(a, e) = \texttt{close\_matches.score}(\hat{a}, e)$$
7. **Two-Tier Identity Protocol: Tier 1 decides, Tier 2 proposes ([`docs/adr/ADR-003.md`](file:///home/harri/yaw/docs/adr/ADR-003.md), [`docs/adr/ADR-012.md`](file:///home/harri/yaw/docs/adr/ADR-012.md)):**
   - **Tier 1 (Deterministic, space L):** exact canonical id, spaCy **normalized key** (NFKC + casefold, leading determiner and possessive clitic removed, `NOUN` lemmatized / `PROPN` verbatim, punctuation and whitespace collapsed), active alias in `aliases`, or ontology label/altLabel alias (`method = 'ontology'`). 0 vectors, 0 LLM calls.
   - **Tier 2 (Blocked + Corroborated, spaces S + L):**
     - **Blocking:** the new entity's space-S vector (`task: sentence similarity | query: {name}: {definition}`) queries `entity_vec` with `vec0` KNN, `k = CANON_TOP_K` (5), within its namespace partition plus active ontology partitions. Candidates with cosine $\ge \tau$ (`CANON_SIM_THRESHOLD`) are kept.
     - **Matching:** a candidate is a merge candidate only if deterministic linguistic corroboration holds: acronym, head-anchored containment (shared `Span.root`), or initial expansion.
     - **Exactly one merge candidate:** hard merge (active alias, `method = 'auto'`, `score = cosine`).
     - **Otherwise:** every kept candidate becomes a soft link in `close_matches` (no merge).
     - Every evaluated pair is logged to `canon_verdicts` (`cosine`, `lexical`, `verdict`) as calibration data for τ.
   - **Predicates never enter Tier 2:** lemma → synset/alias dictionary → base predicate → polarity-guarded static-vector fallback → open `predicates:{lemma-slug}`. Zero embedding calls.
   - **Induction:** create a new canonical node `{namespace}:{normalized-key}` in its typed root namespace via `TierEnum.NEW_INDUCTION`, and insert its space-S vector into `entity_vec`.
   - **Zero LLM Adjudication:** 0 generative LLM calls during entity or predicate canonicalization.

### Component 2: Storage & Schema Contracts (`src/yaw/storage/db.py`)
Single SQLite database at `sample-llm-wiki/.cache/yaw.db` governed by WAL pragmas (reaffirming in-process relational speed over external graph daemons per [`docs/cocoindex-neo4j.md`](file:///home/harri/yaw/docs/cocoindex-neo4j.md)):

- **`note_meta`**: `(rel_path TEXT PRIMARY KEY, title TEXT, category TEXT, modified_at REAL)`.
- **`note_chunks`**: `(chunk_id TEXT PRIMARY KEY, entity_uri TEXT, rel_path TEXT, ordinal INTEGER, heading TEXT, heading_slug TEXT, line_start INTEGER, line_end INTEGER, embedding BLOB)`. `embedding` is space R, float32[768].
- **`wikilinks`**: `(source_uri TEXT, target_uri TEXT, link_label TEXT, PRIMARY KEY(source_uri, target_uri))`.
- **`leaf_nodes`**: `(node_id TEXT, source_leaf_id TEXT, name TEXT, namespace TEXT, definition TEXT, PRIMARY KEY(node_id, source_leaf_id))`. Note extractions only (Invariant 1). No vector column: identity vectors live once per canonical node in `entity_vec` ([`docs/adr/ADR-012.md`](file:///home/harri/yaw/docs/adr/ADR-012.md)).
- **`leaf_relations`**: `(relation_id TEXT PRIMARY KEY, source_leaf_id TEXT, subject_id TEXT, predicate TEXT, object_id TEXT, weight REAL, predicate_id TEXT)`.
- **`aliases`**: hard identity only ([`docs/adr/ADR-003.md`](file:///home/harri/yaw/docs/adr/ADR-003.md), [`docs/adr/ADR-012.md`](file:///home/harri/yaw/docs/adr/ADR-012.md)):
  ```sql
  CREATE TABLE IF NOT EXISTS aliases (
      alias_id TEXT NOT NULL,
      kind TEXT NOT NULL CHECK(kind IN ('entity', 'predicate')),
      canonical_id TEXT NOT NULL,
      method TEXT NOT NULL CHECK(method IN ('exact', 'alias', 'auto', 'new', 'ontology')),
      score REAL NOT NULL DEFAULT 1.0 CHECK(score >= 0.0 AND score <= 1.0),
      source_leaf_id TEXT,
      resolved_at REAL DEFAULT (unixepoch()),
      evidence TEXT,
      PRIMARY KEY (alias_id, kind)
  );
  ```
  *Semantics*: each row is a hard identity assertion (`owl:sameAs`). `score` is 1.0 for Tier 1, and the space-S cosine for a corroborated Tier 2 merge. The former `status` column (`active` / `contested`) is retired: ambiguity is represented by `close_matches`, never by a half-merged alias.
- **`close_matches`**: soft identity candidates ([`docs/adr/ADR-012.md`](file:///home/harri/yaw/docs/adr/ADR-012.md)):
  ```sql
  CREATE TABLE IF NOT EXISTS close_matches (
      src_id TEXT NOT NULL,
      dst_id TEXT NOT NULL,
      score REAL NOT NULL CHECK(score > 0.0 AND score <= 1.0),
      created_at REAL DEFAULT (unixepoch()),
      PRIMARY KEY (src_id, dst_id),
      CHECK (src_id < dst_id)
  );
  ```
  *Semantics*: symmetric and non-transitive (SKOS S44 and §10). Stored once per unordered pair and projected in both directions. Supplies $w(a, e)$ for $P(e \mid a)$. RDF projection: `skos:closeMatch` across concept schemes (vault ↔ external ontology), `yaw:closeMatch` within the vault.
- **`entity_vec`** (space S) and **`fact_vec`** (space R): `sqlite-vec` `vec0` tables. Dimensions are enforced at insert.
  ```sql
  CREATE VIRTUAL TABLE IF NOT EXISTS entity_vec USING vec0(
      node_id TEXT PRIMARY KEY,
      namespace TEXT PARTITION KEY,
      embedding float[768] distance_metric=cosine
  );
  CREATE VIRTUAL TABLE IF NOT EXISTS fact_vec USING vec0(
      relation_id TEXT PRIMARY KEY,
      embedding float[768] distance_metric=cosine
  );
  ```
  `entity_vec` holds one row per canonical node and per external ontology class (partition = ontology prefix). `fact_vec` holds one row per `leaf_relations.relation_id`.
- **`canon_verdicts`** (Audit & Self-Calibration):
  ```sql
  CREATE TABLE IF NOT EXISTS canon_verdicts (
      surface_id TEXT NOT NULL,
      kind TEXT NOT NULL,
      candidate_id TEXT NOT NULL,
      cosine REAL NOT NULL,
      lexical REAL NOT NULL,
      verdict TEXT NOT NULL CHECK(verdict IN ('match', 'distinct')),
      phase TEXT NOT NULL DEFAULT 'ingest' CHECK(phase IN ('ingest', 'compile')),
      decided_at REAL DEFAULT (unixepoch()),
      rationale TEXT,
      PRIMARY KEY (surface_id, kind, candidate_id, phase)
  );
  ```
  *Semantics*: append-only Tier 2 audit log ([`docs/adr/ADR-012.md`](file:///home/harri/yaw/docs/adr/ADR-012.md)). One row per evaluated (mention, candidate) pair: `cosine` is the space-S similarity, `lexical ∈ {0, 1}` records whether linguistic corroboration held, and `verdict` is `match` (hard merge written) or `distinct` (no merge; a `close_matches` link if cosine $\ge \tau$). It is the labelled data for calibrating τ, never an input to resolution.
- **`compiled_nodes` & `compiled_relations`** ([`docs/adr/ADR-002.md`](file:///home/harri/yaw/docs/adr/ADR-002.md)):
  ```sql
  CREATE TABLE IF NOT EXISTS compiled_nodes (
      node_id TEXT PRIMARY KEY,
      name TEXT NOT NULL,
      namespace TEXT NOT NULL DEFAULT 'concepts',
      definition TEXT NOT NULL,
      parent_hub_id TEXT,
      compiled_at REAL DEFAULT (unixepoch())
  );
  CREATE TABLE IF NOT EXISTS compiled_relations (
      relation_id TEXT PRIMARY KEY,
      subject_id TEXT NOT NULL,
      predicate TEXT NOT NULL,
      object_id TEXT NOT NULL,
      weight REAL NOT NULL DEFAULT 1.0,
      source_rule TEXT NOT NULL DEFAULT 'b_tree_split',
      compiled_at REAL DEFAULT (unixepoch())
  );
  ```
- **`query_fts`**: Contentless FTS5 table indexing `heading` and `content` with `entity_uri UNINDEXED` and `contentless_delete=1`.
- **`unified_edges` VIEW (Named Quad Projection with Dynamic Ontology Support, [`docs/adr/ADR-008.md`](file:///home/harri/yaw/docs/adr/ADR-008.md), [`docs/adr/ADR-009.md`](file:///home/harri/yaw/docs/adr/ADR-009.md)):**
  ```sql
  CREATE VIEW IF NOT EXISTS unified_edges AS
  SELECT
      COALESCE(a_src.canonical_id, w.source_uri) AS src_id,
      COALESCE(a_dst.canonical_id, w.target_uri) AS dst_id,
      'wikilink' AS rel_type,
      1.0 AS weight,
      w.source_uri AS source_chunk_id
  FROM wikilinks w
  LEFT JOIN aliases a_src ON a_src.alias_id = w.source_uri AND a_src.kind = 'entity'
  LEFT JOIN aliases a_dst ON a_dst.alias_id = w.target_uri AND a_dst.kind = 'entity'
  UNION ALL
  SELECT
      COALESCE(a_sub.canonical_id, lr.subject_id) AS src_id,
      COALESCE(a_obj.canonical_id, lr.object_id) AS dst_id,
      COALESCE(a_pred.canonical_id, lr.predicate) AS rel_type,
      lr.weight AS weight,
      lr.source_leaf_id AS source_chunk_id
  FROM leaf_relations lr
  LEFT JOIN aliases a_sub ON a_sub.alias_id = lr.subject_id AND a_sub.kind = 'entity'
  LEFT JOIN aliases a_obj ON a_obj.alias_id = lr.object_id AND a_obj.kind = 'entity'
  LEFT JOIN aliases a_pred ON a_pred.alias_id = lr.predicate_id AND a_pred.kind = 'predicate'
  UNION ALL
  SELECT subject_id AS src_id, object_id AS dst_id, predicate AS rel_type, weight, 'compiled' AS source_chunk_id
  FROM compiled_relations
  UNION ALL
  -- Soft identity (ADR-012): symmetric, non-transitive, projected in both directions.
  SELECT src_id, dst_id, 'closeMatch' AS rel_type, score AS weight, 'close_match' AS source_chunk_id FROM close_matches
  UNION ALL
  SELECT dst_id AS src_id, src_id AS dst_id, 'closeMatch' AS rel_type, score AS weight, 'close_match' AS source_chunk_id FROM close_matches
  UNION ALL
  SELECT curie AS src_id, parent_class_uri AS dst_id, CASE WHEN epistemic_type = 'concept' THEN 'broader' ELSE 'subClassOf' END AS rel_type, 1.0 AS weight, 'dynamic_ontology' AS source_chunk_id
  FROM dynamic_classes WHERE parent_class_uri IS NOT NULL;
  ```
- **`unified_nodes` VIEW:**
  ```sql
  CREATE VIEW IF NOT EXISTS unified_nodes AS
  SELECT entity_uri AS node_id, heading AS name, 'chunk' AS namespace, rel_path AS definition FROM note_chunks
  UNION ALL
  SELECT DISTINCT
      COALESCE(a.canonical_id, ln.node_id) AS node_id,
      ln.name,
      CASE WHEN a.canonical_id IS NOT NULL AND instr(a.canonical_id, ':') > 0 THEN substr(a.canonical_id, 1, instr(a.canonical_id, ':') - 1) ELSE ln.namespace END AS namespace,
      ln.definition
  FROM leaf_nodes ln
  LEFT JOIN aliases a ON a.alias_id = ln.node_id AND a.kind = 'entity'
  UNION ALL
  SELECT node_id, name, namespace, definition FROM compiled_nodes
  UNION ALL
  SELECT curie AS node_id, label AS name, ontology_id AS namespace, label AS definition FROM dynamic_classes;
  ```

### Component 3: GraphRAG PPR Retrieval Engine (`src/yaw/retrieval/retriever.py`, [`docs/power-pagerank.md`](file:///home/harri/yaw/docs/power-pagerank.md), [`docs/efficient-ppr-algorithms.md`](file:///home/harri/yaw/docs/efficient-ppr-algorithms.md), [`docs/scipy-graphrag-extensions.md`](file:///home/harri/yaw/docs/scipy-graphrag-extensions.md))
1. **Sparse Adjacency Matrix:** Reads `unified_edges` and constructs an in-memory SciPy Compressed Sparse Row (`csr_matrix`) structure with bidirectional conductance and topological diffusion extensions ([`docs/scipy-graphrag-extensions.md`](file:///home/harri/yaw/docs/scipy-graphrag-extensions.md)).
2. **Directional Conductance & Hub Dampening:**
   $$W_{uv} = \frac{W_{\text{direction}}(u, v)}{(\text{degree}(v))^\gamma}, \quad P_{uv} = \frac{W_{uv}}{\sum_k W_{uk}}$$
   - Downward broadcast (`CONCEPT` $\to$ `CHUNK`/`ENTITY`): $1.0$.
   - Upward abstraction (`CHUNK`/`ENTITY` $\to$ `CONCEPT`): $0.2$ (prevents topic leakage).
   - Downward subsumption broadcast (`Parent` $\to$ `Child` via reciprocal `hasSubClass` / `narrower`): $1.0$.
   - Upward taxonomic abstraction (`Child` $\to$ `Parent` via `subClassOf` / `broader`, [`docs/adr/ADR-002.md`](file:///home/harri/yaw/docs/adr/ADR-002.md), [`docs/adr/ADR-009.md`](file:///home/harri/yaw/docs/adr/ADR-009.md)):
     $$W_{\text{up}}(u, v) = \min\left(0.8, \max\left(0.2, 0.2 + 0.1 \times \text{depth}(v)\right)\right)$$
     Damps to $0.20$ at root proximity ($\text{depth}=0$) to eliminate topic mass dilution into generic roots (`owl:Thing`, `concepts:Root`), while scaling smoothly to $0.80$ for deep leaf classes ($\text{depth} \ge 6$).
   - Document sequence (`chunk_i` $\to$ `chunk_{i+1}`): $0.8$.
   - Semantic relations and forward wikilinks: $1.0$ (equal baseline weight).
   - Reciprocal reverse wikilinks: $0.5$ (attenuated back-propagation).
   - Soft identity `closeMatch` ([`docs/adr/ADR-012.md`](file:///home/harri/yaw/docs/adr/ADR-012.md)): $W = \text{score}$ in both directions (symmetric, non-transitive).
   - Hub degree dampening exponent: $\gamma = 0.5$.
3. **Probabilistic Seed Vector Formulation ($\mathbf{p}_0$):**
   $$\mathbf{p}_0 = \beta_1 \cdot \mathbf{s}_{\text{concepts}} + \beta_2 \cdot \mathbf{s}_{\text{chunks\_vec}} + \beta_3 \cdot \mathbf{s}_{\text{chunks\_fts}}$$
   Where $\beta_1 = 0.60$ (concept canopy bias), $\beta_2 = 0.20$, $\beta_3 = 0.20$.
   - **Concept Canopy Seeding: Alias Priors + Query-to-Fact ([`docs/adr/ADR-010.md`](file:///home/harri/yaw/docs/adr/ADR-010.md), [`docs/adr/ADR-012.md`](file:///home/harri/yaw/docs/adr/ADR-012.md)):**
     $$\mathbf{s}_{\text{concepts}} = \lambda \cdot \widehat{\mathbf{s}}_{\text{alias}} + (1 - \lambda) \cdot \widehat{\mathbf{r}}, \qquad \mathbf{s}_{\text{alias}}[e] = \sum_{a \in M(q)} P(e \mid a)$$
     $$r(q, e) = \operatorname*{mean}_{f \in F_q,\; e \in \{\text{subj}(f), \text{obj}(f)\}} \cos(\mathbf{q}, \mathbf{f}) \quad (0 \text{ if } e \text{ occurs in no } f \in F_q)$$
     - $M(q)$: query mentions extracted in space L; $P(e \mid a)$ from `InMemoryLookupKB` (backed by `aliases` + `close_matches`).
     - $F_q$: top-`FACT_TOP_K` facts from `fact_vec` (space R) for the query vector $\mathbf{q}$. Query-to-triple matching outperforms query-to-node matching ([HippoRAG 2 §6.1](https://arxiv.org/abs/2502.14802)); entity seeds average their facts' scores (§3.5).
     - $\widehat{\cdot}$ is L1 normalization. The channels are additive, so an entity named in the question but absent from $F_q$ still seeds.
     - There is **no** query → `leaf_nodes` vector seeding.
   - When EmbeddingGemma is offline ($\mathbf{q} = \mathbf{0}$): $\mathbf{r} = \mathbf{0}$ and $\mathbf{s}_{\text{chunks\_vec}} = \mathbf{0}$, so $\mathbf{p}_0$ reduces continuously to the alias-prior and FTS5 channels (no artificial fallback branches).
   - The combined seed vector $\mathbf{p}_0$ is normalized to unit sum: $\sum_u \mathbf{p}_0[u] = 1.0$.
   - **Hyperparameters** $\lambda = 0.5$, `FACT_TOP_K` $= 5$, $\beta = (0.6, 0.2, 0.2)$ and τ $= 0.80$ are **uncalibrated priors**. They are calibrated on a benchmark calibration split only, with reporting on a disjoint held-out split (Epic 8 protocol).

4. **Topological Disambiguation via Power Iteration Diffusion:**
   $$\mathbf{p}^{(t+1)} = \alpha \mathbf{P}^T \mathbf{p}^{(t)} + (1 - \alpha) \mathbf{p}_0, \quad \alpha = 0.85 \quad (30 \text{ iterations})$$
   Where $\mathbf{P}$ is the row-stochastic transition matrix of `unified_edges` with directional conductance ($1.0$ down, $0.2$ up) and hub dampening $\gamma = 0.5$.
   - **Topological Disambiguation**: By diffusing probability mass across the transposed transition matrix $\mathbf{P}^T$, polysemous candidates initially seeded via $P(e \mid a)$ are disambiguated by graph topology. Entities whose structural neighbors correlate with the query context accumulate stationary mass through coherent walk reinforcement, whereas spurious candidates bleed mass into unconnected sinks.

5. **Bayesian Evidence Fusion for Candidate Chunk Selection & Rescoring:**
   - Candidate chunks are drawn directly from the converged distribution $\mathbf{p}^*$ for nodes where $\text{node} \in \text{chunks}$ and $\mathbf{p}^*[\text{chunk}] > 0$.
   - When $\mathbf{q} \ne \mathbf{0}$: embeds query once using `embed_query()`, computes $\text{sim}(q, c) = \mathbf{q} \cdot \mathbf{c}$ against pre-indexed SQLite chunk embeddings, and ranks by Bayesian evidence fusion:
     $$\text{Score}(c) = \left( \sum_{e \in c} P(e \mid a) \cdot \mathbf{p}^*[e] + \mathbf{p}^*[c] \right) \cdot \text{sim}(q, c)$$
     with relevance cutoff $\text{sim}(q, c) \ge \text{RELEVANCE\_THRESHOLD}$ ($0.10$).
   - When $\mathbf{q} = \mathbf{0}$: ranks candidates directly by converged graph mass $\text{Score}(c) = \mathbf{p}^*[c]$.
6. **Zero-Copy Disk Slice Hydration:** Winning top-$k$ chunk IDs look up disk byte offsets in `note_chunks` and seek directly to raw text on disk.
7. **Dual-Layer Output Presentation (The Bridging Subgraph):**
   - **Adaptive Semantic Scope**:
     - **When Evidence Chunks Hit ($K > 0$)**: The Ontological Context is strictly **retrieval-grounded**. It extracts the **Bridging Subgraph** connecting the query to the evidence:
       $$\text{active\_concepts} = (\text{seed\_concepts} \cup \text{chunk\_concepts}) \quad \text{sorted by } \mathbf{p}^*$$
       Where $\text{seed\_concepts} = \{u \mid \mathbf{p}_0[u] > 0 \land \text{is\_concept}(u)\}$ and $\text{chunk\_concepts} = \{u \mid \exists c \in W : (u \xrightarrow{\text{manifests\_in}} c) \lor (c \xrightarrow{\text{mentions}} u)\}$.
     - **When No Chunks Hit ($K = 0$)**: The Ontological Context falls back continuously to global top-$N$ concepts by $\mathbf{p}^*$, while Ground-Truth Evidence informs the agent that zero chunks were found and recommends `ingest_raw`.

### Component 4: FastMCP Protocol Interface & Karpathy LLM-Wiki Contract (`src/yaw/server/`)
YAW implements the **Karpathy LLM-Wiki pattern** over FastMCP (Streamable HTTP `127.0.0.1:8765` and STDIO).

#### 1. The 4 Production Tools (Client-Facing Contract)
1. **`query_wiki(query: str, top_k: int = 3) -> str` (Associative Retrieval):**
   Executes full GraphRAG retrieval: seed candidate generation, in-memory bi-encoder reranking, SciPy PPR graph diffusion across `unified_edges`, and zero-copy disk byte slice hydration. Returns the dual-layer payload: Ontological Context explaining *why* concepts connect, plus hydrated Ground-Truth Evidence.
2. **`ingest_raw(content: str, title: str | None = None, url: str | None = None) -> dict` (Source Accumulation):**
   Ingests an external URL or raw text string into a timestamped, slugged Markdown note in `vault/raw/` with standardized YAML frontmatter (`title`, `source_url`, `ingested_at`). Strictly append-only (preserves Invariant 1: historical raw notes are immutable).
3. **`compile_wiki(force: bool = False) -> dict` (Ontology Compilation Trigger & Hierarchical Deepening):**
   Triggers asynchronous background indexing, ontology re-consolidation, B-Tree hierarchical deepening ([`docs/adr/ADR-002.md`](file:///home/harri/yaw/docs/adr/ADR-002.md)), and compile-time closed-world SHACL validation ([`docs/adr/ADR-011.md`](file:///home/harri/yaw/docs/adr/ADR-011.md)).
   - **Delta Ingestion:** By default (`force=False`), it only processes newly modified/added raw notes via CocoIndex content-hash deltas.
   - **B-Tree Cluster Splitting & Hypernym Inducing:** When any hub node accumulates out-degree $> K_{\max} = 10$, SciPy Ward's HAC partitions the subtree. LLM induces a hypernym superclass, rewiring `member --[subClassOf]--> hypernym` and `hypernym --[subClassOf]--> parent_hub` with asymmetric conductance ($1.0$ downward broadcast, $0.2$ upward damping).
   - **Compile-Time Fast SHACL Validation (<200ms):** Executes lean closed-world validation (`validate_wiki_graph_fast(conn)`) over `unified_edges` and `unified_nodes`. Detects shape violations and dialectical contradictions (`causes` vs `prevents`, `supports` vs `refutes`, ambiguous identity warnings) and transactionally commits them to the `compiled_violations` SQLite table.
   - **Deterministic View Materialization:** Emits read-only `vault/wiki/{namespace}/{slug}.md` pages directly from SQLite `unified_nodes` (frontmatter) and `unified_edges` (`[[wikilinks]]`) with zero LLM authoring.
4. **`lint_wiki(query: str | None = None, force: bool = False) -> list[dict]` (SHACL Graph & Dialectical Conflict Linter):**
   Surfaces dialectical contradictions, ambiguous identities, and schema violations in the ontology layer ([`docs/adr/ADR-011.md`](file:///home/harri/yaw/docs/adr/ADR-011.md), [`docs/rdf-owl-shacl.md`](file:///home/harri/yaw/docs/rdf-owl-shacl.md)).
   - **$\mathcal{O}(1)$ Cached Retrieval:** Reads pre-compiled violations directly from the `compiled_violations` SQLite table in $<5\text{ ms}$ by default (`force=False`).
   - **Query Filtering:** When `query` is provided, executes an indexed SQLite filter (`WHERE focus_node = ? OR message LIKE ?`) returning violations localized to a specific concept or topic.
   - **Epistemic Contradiction Steering:** Surfaced contradictions prompt the agent to accumulate additional authoritative sources (`ingest_raw`) or author a reconciliation note in `vault/raw/`.
   - **Live Validation Override:** When `force=True`, bypasses SQLite cache and runs a live in-process pySHACL pass over the active graph, updating `compiled_violations`.

#### 2. Development & Diagnostic Probes (Internal / Debugging Only)
- **`find_seed(query: str, top_k: int = 3) -> list[dict]`:** Inspects intermediate candidate generation prior to graph traversal.
- **`read_document(path: str, line_start: int | None = None, line_end: int | None = None) -> str`:** Direct line-based disk hydration reader for testing.
- **`get_ontology_schema() -> dict`:** Low-level introspection tool returning database statistics, active taxonomy namespaces, and base predicates.

### Component 5: W3C Semantic Web Substrate (RDF, OWL, SKOS, SHACL)

YAW formalizes knowledge representation by anchoring in-process SQLite graph views to W3C Semantic Web standards ([`docs/adr/ADR-008.md`](file:///home/harri/yaw/docs/adr/ADR-008.md), [`docs/rdf-owl-shacl.md`](file:///home/harri/yaw/docs/rdf-owl-shacl.md), [`docs/ontological-roots.md`](file:///home/harri/yaw/docs/ontological-roots.md), [`docs/epic7-pluggable-ontology-integration-spec.md`](file:///home/harri/yaw/docs/epic7-pluggable-ontology-integration-spec.md)). To eliminate architectural confusion and prevent semantic collisions, each standard governs a strictly partitioned, non-overlapping concern:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        W3C SEMANTIC STACK ARCHITECTURAL ROLES                          │
├───────────────────┬────────────────────────────────────────────────────────────────────┤
│ Standard          │ System Role & Invariant Responsibility                             │
├───────────────────┼────────────────────────────────────────────────────────────────────┤
│ RDF 1.1           │ Graph Data Model & Quad Substrate (unified_edges, unified_nodes)   │
│ OWL 2 RL / RDFS   │ Domain Schema Logic, Subsumption & Hard Identity (owl:sameAs)      │
│ SKOS              │ Epistemic Categorization, Thesaurus & Soft Identity (closeMatch)   │
│ SHACL             │ Closed-World Data Integrity, Dialectical Constraints & Linting     │
└───────────────────┴────────────────────────────────────────────────────────────────────┘
```

#### 1. RDF 1.1: Graph Data Model & Named Quad Substrate
- **Role:** Foundational substrate representing all knowledge as directed, typed relational edges: `(Subject, Predicate, Object, GraphContext)`.
- **Substrate Realization:** Streamed dynamically from SQLite relational views ([`unified_edges`](file:///home/harri/yaw/src/yaw/storage/db.py#L85-L168) and [`unified_nodes`](file:///home/harri/yaw/src/yaw/storage/db.py#L170-L200)). Zero external graph servers (Neo4j, Blazegraph, Jena) are permitted (Invariant 3).
- **URI Minting & Namespace Anchoring:** Every node is canonically projected into one of the **9 Universal Root Namespaces** ([`docs/adr/ADR-005.md`](file:///home/harri/yaw/docs/adr/ADR-005.md)), e.g., `concepts:{slug}`, `entities:{slug}`, `predicates:{slug}`, or `chunk:{file}:{start}-{end}#chunk:{ordinal}`.
- **Lossless Extraction Invariant:** Machine-extracted leaf triples are persisted immutably in `leaf_relations`. Raw triples are never discarded during ingestion; configuration constraints (e.g., closed-world open IE filtering) are applied non-destructively at query/view time.

#### 2. OWL 2 RL / RDFS: Domain Schema Logic & Identity
- **Role:** Formal definitional domain ontology, class subsumption hierarchies, and reversible entity identity.
- **Computational Profile:** Strictly confined to the **OWL 2 RL / RDFS** rule-based profile. All inference executes in-process via recursive SQLite CTEs (`WITH RECURSIVE`) and sparse SciPy matrix multiplications in polynomial time ($O(N)$ to $O(N^2)$). Zero heavy DL tableaux reasoners (Pellet, HermiT) are invoked.
- **Subsumption & Transitive Closure:** Hierarchical domain taxonomies are modeled via standard `rdfs:subClassOf`. Multi-hop inheritance is resolved deterministically via `get_subclass_closure` without synthetic wrapper properties.
- **Identity Mapping (Hard vs Soft):**
  - `aliases` rows (hard identity) project to `owl:sameAs` equivalence:
    $$(a, c) \in \texttt{aliases} \implies a \equiv_{\text{owl:sameAs}} c$$
  - `close_matches` rows (soft identity) project to `skos:closeMatch` across concept schemes and `yaw:closeMatch` (symmetric, non-transitive) within the vault ([`docs/adr/ADR-012.md`](file:///home/harri/yaw/docs/adr/ADR-012.md)).
  - `canon_verdicts` is **not** projected. A `distinct` verdict means "not merged" (insufficient corroboration), not evidence of non-identity, so emitting `owl:differentFrom` would be unsound under the open-world assumption and would make any later `owl:sameAs` between the pair an OWL 2 RL inconsistency (rule `eq-diff1`, [OWL 2 Profiles §4.3, Table 4](https://www.w3.org/TR/owl2-profiles/#Reasoning_in_OWL_2_RL_and_RDF_Graphs_using_Rules)).
- **Property Axioms:** Encodes property characteristics (`owl:TransitiveProperty`) and mutual exclusion (`owl:propertyDisjointWith`) to govern dialectical reasoning and asymmetric conductance.

#### 3. SKOS: Epistemic Categorization & Vault Thesaurus
- **Role:** Human-centered knowledge organization, epistemic topic indexing, controlled vocabulary management, and Obsidian vault thesaurus navigation.
- **Core Primitives:** `skos:Concept`, `skos:ConceptScheme` (`yaw:VaultScheme`), `skos:prefLabel`, `skos:altLabel`, `skos:broader`, `skos:narrower`, and `skos:related`.
- **The Non-Punning Invariant (Metamodeling Separation):**
  - An architectural boundary separates schema classes from thesaurus concepts. An entity URI is either an `owl:Class` (domain classification schema) or a `skos:Concept` (thesaurus entry/epistemic topic), **never both on the same URI**.
  - Punning `concepts:Foo a owl:Class, skos:Concept` is strictly forbidden.
  - Subsumption between domain classes is strictly `rdfs:subClassOf`; hierarchical navigation between broader and narrower thesaurus topics is strictly `skos:broader` and `skos:narrower`.

#### 4. SHACL: Closed-World Data Integrity & Dialectical Linter
- **Role:** Structural and semantic invariant enforcement, dialectical conflict detection, and data cleanliness validation.
- **Execution & Storage Lifecycle ([`docs/adr/ADR-011.md`](file:///home/harri/yaw/docs/adr/ADR-011.md)):**
  - **Ingestion Decoupling:** Continuous single-stage ingestion (`pipeline.py`) does *not* invoke SHACL, preserving sub-second streaming throughput per chunk.
  - **Compile-Time Validation:** Executed in-process via `validate_wiki_graph_fast(conn)` during `compile_wiki()` in $<200\text{ ms}$ using SQLite relational pre-closure and `inference=None`.
  - **Persistent SQLite Contradiction Cache:** Detected shape violations and dialectical contradictions are transactionally committed to `compiled_violations` in `yaw.db`:
    ```sql
    CREATE TABLE IF NOT EXISTS compiled_violations (
        violation_id TEXT PRIMARY KEY,
        focus_node TEXT NOT NULL,
        shape_id TEXT NOT NULL,
        result_path TEXT,
        severity TEXT NOT NULL CHECK(severity IN ('Info', 'Warning', 'Violation')),
        message TEXT NOT NULL,
        source_chunk_id TEXT,
        conflicting_chunk_id TEXT,
        compiled_at REAL NOT NULL DEFAULT (unixepoch())
    );
    CREATE INDEX IF NOT EXISTS idx_violations_focus ON compiled_violations(focus_node);
    CREATE INDEX IF NOT EXISTS idx_violations_shape ON compiled_violations(shape_id);
    ```
  - **$\mathcal{O}(1)$ Client Linting:** `@mcp.tool() async def lint_wiki()` queries `compiled_violations` in $\mathcal{O}(1)$ time ($< 5\text{ ms}$), returning all violations or filtering by topic (`query="<term>"`). Live pySHACL is reserved for `force=True`.
- **Closed-World Assumption (CWA):** While OWL operates under Open-World Assumption (missing info is unknown), SHACL operates under Closed-World Assumption with Unique Name Assumption (UNA) at validation time: unfulfilled constraints are explicit actionable violations.
- **Core Declarative Shapes (`src/yaw/ontology/shapes/wiki_shapes.ttl`):**
  - **`yaw:ConceptShape`:** Requires non-empty definitions conforming to word limits ($\le 50$ words via `config.DEFINITION_MAX_WORDS`) and canonical URI syntax across universal root namespaces.
  - **`yaw:RelationShape`:** Enforces referential integrity on subjects, predicates, and objects, detecting dangling edges and unregistered namespaces.
  - **`yaw:DialecticalIntegrityShape`:** Flags dialectical contradictions (`causes` vs `prevents`, `supports` vs `refutes`, `supports` vs `contradicts`, reciprocal conflicts).
  - **Ambiguous identity (`sh:Warning`, [`docs/adr/ADR-012.md`](file:///home/harri/yaw/docs/adr/ADR-012.md)):** Flags a node with $\ge 2$ `close_matches` targets that are not linked to each other. This is an epistemic tension signal, not a data error.
- **Ontology Layer FTS & Vector Indexing Invariant ([`docs/adr/ADR-012.md`](file:///home/harri/yaw/docs/adr/ADR-012.md) §5):**
  Whenever dynamic ontologies are loaded (`ingest_ontology()`) or hypernyms are induced at compile time, their names and definitions **must be inserted into `query_fts` (FTS5)**, idempotently (re-ingestion replaces rows). External ontology classes additionally get one space-S vector in **`entity_vec`** (partition = ontology prefix), making them Tier 2 candidates for note mentions. They are never written to `leaf_nodes`. Materialized markdown pages in `vault/wiki/` are read-only views for human navigation, never re-indexed by CocoIndex (Invariant 1).

---

## 6. Repository Architecture & Directory Boundaries

The codebase enforces strict separation between engine execution, test verification, system documentation, and vault data:

```txt
yaw/
├── src/yaw/                  # Core Engine Implementation (Single Source of Truth)
│   ├── config.py             # Authoritative configuration, 9 root namespaces, 14 base predicates
│   ├── storage/              # SQLite SSOT: DB connection, PRAGMAs, sqlite-vec, fts5, views (db.py)
│   ├── indexer/              # CocoIndex pipeline, markdown chunking, disk byte offsets (pipeline.py)
│   │   └── canonicalize.py   # Two-Tier Identity Protocol: aliases, close_matches, KB prior index (ADR-012)
│   ├── nlp/                  # spaCy + spacy-llm Open IE, predicate normalization, Ollama client
│   │   ├── client.py         # Minimal Ollama HTTP client
│   │   ├── embeddings.py     # EmbeddingGemma model card formatting & cosine similarity
│   │   └── extractor.py      # spaCy pipeline, spacy.NER.v3 (primary), Open IE relations, TextCat
│   ├── retrieval/            # GraphRAG retriever: SciPy CSR, conductance, PPR diffusion, hydration
│   │   └── retriever.py      # GraphRAGRetriever engine with taxonomic conductance asymmetry
│   └── server/               # FastMCP daemon exposing retrieval & seed tools (daemon.py)
│
├── tests/                    # Grounded Test Suite (Mirroring src/yaw/ 1:1)
│   ├── conftest.py           # Ephemeral mock Ollama HTTP server for fast offline testing
│   ├── test_canonicalization.py # Invariant tests: Two-Tier Cython KB resolver, aliases, view rerouting
│   ├── test_cli_lifecycle.py # End-to-end CLI lifecycle & daemon management tests
│   ├── test_cocoindexer.py   # Invariant tests: disk byte offsets, contentless FTS, views
│   ├── test_embedding_gemma.py # EmbeddingGemma prompt formatting and bi-encoder reranking
│   ├── test_fewshot_steering.py # Few-shot LLM steering & Open IE extraction tests
│   ├── test_leaf_pipeline.py # spaCy + spacy.NER.v3, Doc integrity, Open IE, predicate normalization
│   ├── test_pluggable_ontology.py # Dynamic ontology ingestion, class hierarchies, and KB sync
│   ├── test_provenance.py    # Doc integrity, line ranges, byte hydration
│   ├── test_rdf_substrate.py # RDF 1.1 / OWL 2 RL quad views and graph projections
│   ├── test_retriever.py     # SciPy CSR matrix, directional conductance, PPR convergence, Bayesian scoring
│   ├── test_shacl_lint.py    # Closed-world SHACL dialectical contradiction validation tests
│   ├── test_storage.py       # Physical SQLite schema, pragmas, migrations, and unified views
│   ├── test_vault_purity.py  # Vault purity tests asserting zero engine specs in raw/
│   └── test_yaw.py           # FastMCP server integration and end-to-end tool execution
│
├── docs/                     # Engine Specifications, Architectural Blueprints & Playbooks
│   │                         # (SSOT for developer documentation; kept strictly outside the vault!)
│   ├── ADR.md                # Architectural Decision Records ledger (ADR-001 through ADR-012)
│   ├── adr/                  # Authoritative ADR files (ADR-001.md ... ADR-012.md)
│   ├── alignment-sprint-for-e5s2.md # Sprint 5.1 turnkey execution DAG & prompt cards
│   ├── epics-alignment-roadmap.md # Authoritative engineering implementation roadmap (Epics 1 through 8)
│   ├── ontological-roots.md  # 9 Universal root namespaces and ontology alignment
│   ├── epic7-pluggable-ontology-integration-spec.md # Dynamic OWL/SKOS ontology ingestion specification
│   ├── scipy-graphrag-extensions.md # SciPy sparse matrix GraphRAG diffusion extensions
│   ├── cocoindex.md          # CocoIndex streaming ingestion & incremental processing
│   ├── cocoindex-neo4j.md    # CocoIndex SQLite in-process relational speed benchmark comparison
│   ├── rdf-owl-shacl.md      # Authoritative W3C Semantic Web integration manual
│   ├── USER_STORIES.md       # BDD user stories across Epics 5, 6, 7, 8
│   ├── how-to-handle-spacy-llm.md # Canonical spaCy-LLM model & task integration guide
│   ├── open-ie-authoritative-source.md # Open IE extraction specifications
│   ├── power-pagerank.md     # Personalized PageRank mathematical specification
│   ├── efficient-ppr-algorithms.md # Fast PPR diffusion algorithms
│   └── karpathy-llm-wiki.md  # Karpathy LLM-Wiki contract specification
│
└── sample-llm-wiki/          # Canonical Reference Vault (Domain Knowledge Substrate)
    ├── raw/                  # User notes and external knowledge material (e.g. test-material/)
    ├── .cache/               # Local SQLite database (yaw.db) and CocoIndex state
    └── wiki/                 # (Optional) Asynchronous materialized Markdown summaries
```

### The Law of Vault Purity
Engine documentation, architectural plans, and implementation playbooks must **never** be placed inside `sample-llm-wiki/raw/` or any note vault.
1. **Preventing Semantic Pollution:** The indexer treats everything in `raw/` as domain knowledge. Storing engine meta-specifications inside `raw/` corrupts the knowledge graph with self-referential development concepts (e.g. `fastmcp`, `sqlite_vec`, `pytest`).
2. **Distinct Lifecycles:** Engine documentation evolves with the codebase in `docs/` and is versioned alongside `src/yaw/`. Vault notes in `raw/` are external user artifacts.

---

## 7. Testing Rigor, Code Cleanliness & Static Hygiene

### Anti-Tautological Unit Testing Principles
1. **The Fallacy of Cushion Mocks:**
   A "tautological cushion mock" occurs when a test mocks a database query to return a pre-baked array, runs a retriever method, and asserts that the method returned that same array. This proves nothing about the code: if the SQLite schema changes, if `sqlite-vec` rejects numpy array types, or if a SQL view syntax error occurs, the tautological test continues to pass with flying colors.
2. **Grounded Invariant Testing:**
   All unit and integration tests must run against real systems:
   - **Real Temporary SQLite:** Tests use `tmp_path` with actual SQLite databases, load the real `sqlite-vec` binary extension, and execute real `vec_distance_cosine` queries.
   - **Real Disk Hydration:** Tests write real markdown files to disk and verify that binary byte offsets (`file.seek()`, `file.read()`) reconstitute exact UTF-8 chunks without drift.
   - **Real Sparse Matrix Math:** Tests build real SciPy `csr_matrix` instances and verify physical mathematical properties (row-stochasticity, Perron-Frobenius conservation, power iteration convergence).
3. **Boundary Isolation Mocking:**
   The *only* acceptable mock boundary in YAW is external remote network transport:
   - In [`tests/conftest.py`](file:///home/harri/yaw/tests/conftest.py), an in-process ephemeral HTTP server (`BaseHTTPRequestHandler` on `127.0.0.1:<random_port>`) handles `/api/embed`, `/api/chat`, and `/api/version`.
   - Embeddings are generated deterministically via an in-memory lexical hashing algorithm (`_hash_embed`), producing real 768-dimensional unit vectors without requiring a GPU or running Ollama daemon.
   - This keeps the entire test suite executing in under 4 seconds while verifying 100% of internal SQL, graph, and disk logic.

### Ruff & Ty Static Cleanliness
1. **Ruff Determinism:**
   All code in `src/` and `tests/` is formatted with Ruff (`uv run ruff format --check`). AST-level code formatting is completely automated, eliminating stylistic diff noise in version control.
2. **Static Hygiene (Ruff & Ty):**
   - Zero unused imports, zero undefined variables, zero wildcard imports (`from module import *`), verified via `uv run ruff check` and `uv run ty check`.
   - Strict typing across all function signatures and return types.
   - Explicit exception handling: catch targeted operational errors (`sqlite3.OperationalError`, `urllib.error.HTTPError`), never silent bare `except: pass` on unexpected failures.
3. **Single Source of Truth (SSOT):**
   - Configuration defaults live strictly in [`src/yaw/config.py`](file:///home/harri/yaw/src/yaw/config.py) (loaded from `config.toml` via Python 3.14 `tomllib`).
   - Database schemas and PRAGMAs live strictly in [`src/yaw/storage/db.py`](file:///home/harri/yaw/src/yaw/storage/db.py).
   - Never duplicate constants, prompt templates, or column names across disparate modules.
