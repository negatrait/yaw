# FOUNDATIONAL.md: Product Requirements & Architectural Specification for YAW

**Document Status:** Authoritative Root Document & Single Source of Truth (SSOT)  
**System Name:** YAW (Yet Another Wiki / FastMCP Ontology GraphRAG Engine)  
**Current State:** Epic 6.2 Complete (375 passing tests, 0 lint warnings, Sentrux quality 7270, 0 circular dependencies)  
**Target Audience:** Autonomous Coding Agents & Core Systems Engineers  

---

## 0. Progressive Disclosure Architecture & Reading Guide

`FOUNDATIONAL.md` is the authoritative **Root Document and Single Source of Truth (SSOT)** for YAW. Every engineering epic, architectural decision, test invariant, and operational contract is a direct, traceable derivative of this document.

To navigate the system efficiently without cognitive thrashing or context decay, follow this progressive disclosure hierarchy:

| System Layer | Document Link | Scope & Purpose |
| :--- | :--- | :--- |
| **System Overview & Invariants** | [`FOUNDATIONAL.md`](file:///home/harri/yaw/FOUNDATIONAL.md) (This Document) | Vision, 7 Invariants, Model Stack, Single-Stage DAG, SQLite Contracts, SciPy PPR Algorithms. |
| **Active Execution Target** | [`docs/alignment-sprint-for-e5s2.md`](file:///home/harri/yaw/docs/alignment-sprint-for-e5s2.md) | Sprint 5.1 turnkey execution DAG, Prompt Cards for WP 5.1.1–5.1.7, TDD test specification. |
| **Architectural Roadmap** | [`docs/epics-alignment-roadmap.md`](file:///home/harri/yaw/docs/epics-alignment-roadmap.md) | 7-Epic master plan (Epics 1–4 complete, Epic 5 active, Epics 6–7 specified), quality gates. |
| **Architectural Decision Records** | [`docs/ADR.md`](file:///home/harri/yaw/docs/ADR.md) | Formal ADR ledger ([`ADR-001`](file:///home/harri/yaw/docs/adr/ADR-001.md) through [`ADR-009`](file:///home/harri/yaw/docs/adr/ADR-009.md)). |
| **Semantic Web Integration** | [`docs/rdf-owl-shacl.md`](file:///home/harri/yaw/docs/rdf-owl-shacl.md) | W3C RDF quad mapping, OWL 2 RL inference, SKOS concept schemes, closed-world SHACL validation. |
| **Acceptance Criteria (BDD)** | [`docs/USER_STORIES.md`](file:///home/harri/yaw/docs/USER_STORIES.md) | Formal Gherkin / BDD user stories for Epics 5, 6, and 7. |
| **Core Algorithm Specs** | [`docs/power-pagerank.md`](file:///home/harri/yaw/docs/power-pagerank.md)<br>[`docs/efficient-ppr-algorithms.md`](file:///home/harri/yaw/docs/efficient-ppr-algorithms.md) | Mathematical formulation of sparse SciPy Personalized PageRank, hub dampening, and convergence proofs. |
| **NLP & Extraction Contracts** | [`docs/how-to-handle-spacy-llm.md`](file:///home/harri/yaw/docs/how-to-handle-spacy-llm.md)<br>[`docs/open-ie-authoritative-source.md`](file:///home/harri/yaw/docs/open-ie-authoritative-source.md) | Official `spacy-llm` integration, Jinja2 template definitions, zero-monkeypatching registry contracts. |
| **LLM-Wiki Specification** | [`docs/karpathy-llm-wiki.md`](file:///home/harri/yaw/docs/karpathy-llm-wiki.md) | Karpathy LLM-Wiki contract, FastMCP streaming interface, and wiki compiler specifications. |

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
- Cross-document canonicalization occurs in-process via a **Two-Tier Convergence Engine** ([`docs/adr/ADR-003.md`](file:///home/harri/yaw/docs/adr/ADR-003.md)).
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

### Invariant 5: Dual Epistemic Retrieval Contract
- Retrieval must never return isolated chunk fragments without context, nor abstract theories without evidence.
- Every retrieval query explains **both**:
  1. **The Ontological Context:** Active concept nodes, entity paths, and discovered semantic triples explaining *why* topics relate (the Bridging Subgraph).
  2. **The Grounded Evidence:** Direct hits to explicit raw chunks hydrated from disk with exact byte and line citations.

### Invariant 6: Real-Invariant Testing (Zero Tautological Cushion Mocks)
- Tests must verify real invariants against real disk files, real SQLite instances, real `sqlite-vec` binary operations, and real SciPy sparse matrices.
- Never mock internal subsystem interfaces or database connections. Tautological cushion mocks (mocking queries or matrix math) test only the mock definitions and conceal fatal structural bugs.
- Boundary isolation mocking is strictly confined to external remote network services (the local Ollama HTTP server, mocked via a fast in-process loopback HTTP server in [`tests/conftest.py`](file:///home/harri/yaw/tests/conftest.py) per [`docs/adr/ADR-007.md`](file:///home/harri/yaw/docs/adr/ADR-007.md)).

### Invariant 7: Code Cleanliness, Strict Formatting & Single Source of Truth
- Code must adhere to Ruff formatting standards (`uv run ruff format --check`) and clean static lint/type analysis via Ruff and Ty (`uv run ruff check`, `uv run ty check`) with zero unused imports, explicit typing, and zero circular dependencies.
- Architecture adheres to strict Single Source of Truth (SSOT): schema definitions, PRAGMAs, and configurations are declared once in [`src/yaw/storage/db.py`](file:///home/harri/yaw/src/yaw/storage/db.py) or [`src/yaw/config.py`](file:///home/harri/yaw/src/yaw/config.py) and imported everywhere, never copy-pasted or shadowed.

---

## 3. The Model Stack Contract

To eliminate hallucinations, non-determinism, and memory bloat, YAW is bound to a strict local model contract served via Ollama and in-process spaCy:

| Role | Model / Mechanism | Endpoint / Execution | Task Format / Prefix / Contract |
| :--- | :--- | :--- | :--- |
| **Document Embedding** | `embed-gemma-300m-FLM` | `/api/embed` | `title: {doc_title} \| text: {content}` (or `title: none \| text: {content}`) — strictly long document chunk indexing (`note_chunks.embedding`) |
| **Query Embedding** | `embed-gemma-300m-FLM` | `/api/embed` | `task: search result \| query: {query}` — retrieval queries (`s_chunks_vec`) |
| **Bi-Encoder Reranker** | `embed-gemma-300m-FLM` | In-Memory Dot Product | Re-scores candidate chunk vectors against query vector; zero redundant re-embedding |
| **Entity/Predicate Canonicalization** | `en_core_web_lg` static 300d vectors + `InMemoryLookupKB` | In-Process | Two-Tier Convergence Protocol (Tier 1 direct match, Tier 2 near match > 0.80, 0 LLM adjudication; [`docs/adr/ADR-003.md`](file:///home/harri/yaw/docs/adr/ADR-003.md)) |
| **Open IE Triplet Extraction & Typing** | `Qwen3-4B-Instruct-2507-GGUF` | `/api/chat` via `spacy.Raw.v1` | Single-pass schema-free Open IE (`OPEN_IE_RAW_TRIPLES_TEMPLATE`; [`docs/adr/ADR-001.md`](file:///home/harri/yaw/docs/adr/ADR-001.md)) with `spacy.BatchCache.v1` disk caching |
| **Node Definition Generation** | `Qwen3-4B-Instruct-2507-GGUF` | `/api/chat` | Off-the-shelf `spacy.Summarization.v1` (`field="description"`, `max_n_words=50` via `config.DEFINITION_MAX_WORDS`); `vault/wiki/` pages materialize deterministically from SQLite |

### Strict Embedding Boundary Contract
- **Strict Embedding Boundary**: `embed-gemma-300m-FLM` is restricted **exclusively** to long document chunk indexing (`note_chunks.embedding`) and retrieval queries (`s_chunks_vec`). All post-indexing lexical operations, concept canonicalization, and (s,p,o) triplet treatments belong 100% to the spaCy pipeline (`en_core_web_lg` static 300d vectors, `InMemoryLookupKB`, token similarity). Zero external Gemma calls for entities or predicates.
- **Two-Tier Convergence Protocol**:
  - **Tier 1 (Direct Match)**: Exact slug or active alias lookup in `aliases` or `InMemoryLookupKB`.
  - **Tier 2 (Near Match)**: In-process vector similarity > `CANON_SIM_THRESHOLD` (0.80) using `en_core_web_lg` static 300d GloVe vectors. Consolidate into the existing canonical node.
  - **Fallback**: Novel canonical node creation.
  - **Zero LLM Adjudication**: 0 LLM calls during entity or predicate canonicalization.
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
  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
  │ 1. SINGLE-STAGE INGESTION & LEAF ONTOLOGY EXTRACTION (src/yaw/indexer/ & src/yaw/nlp/)      │
  │ • CocoIndex Watcher: Rust core change detection (internal content-hash deltas)             │
  │ • Markdown Chunker: Deterministic boundary splitting, byte offsets & line ranges           │
  │ • Strict Embedding Boundary: embed-gemma-300m-FLM strictly for chunk indexing              │
  │ • Native Disk Cache: spacy.BatchCache.v1 at .cache/spacy_llm for all LLM calls             │
  │ • Markdown Structure: code spans in doc.spans["code"]; text never mutated                   │
  │ • Unified spacy-llm Pipeline (Qwen3-4B via yaw.Ollama.v1 with spacy.Raw.v1):                │
  │   - OPEN_IE_RAW_TRIPLES_TEMPLATE: Single-pass (s,p,o) schema-free extraction                │
  │   - Zero External Gemma Calls: 100% in-process spaCy for entities and predicates            │
  │ • In-Process Two-Tier Convergence Protocol:                                                 │
  │   - Tier 1: Direct match (exact slug / active alias in aliases or InMemoryLookupKB)        │
  │   - Tier 2: Near match (in-process static vector similarity > 0.80 via en_core_web_lg)     │
  │   - Fallback: Novel canonical node creation (0 LLM adjudication)                           │
  │ • Predicate Normalization: lemma + base predicate / alias dictionary + vector similarity    │
  │ • Node & Wiki Summaries: spacy.Summarization.v1 (field="description")                       │
  └──────────────────────────────────────┬──────────────────────────────────────────────────────┘
                                         │
                                         ▼
  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
  │ 2. STORAGE CONTRACTS & SQLITE QUAD SUBSTRATE (src/yaw/storage/db.py)                        │
  │ • Single SQLite DB at sample-llm-wiki/.cache/yaw.db (WAL mode, busy_timeout=60000ms)    │
  │ • Physical Extraction Tables: note_meta, note_chunks, wikilinks, leaf_nodes, leaf_relations│
  │ • Engine-Owned Tables: aliases, canon_verdicts, compiled_nodes, compiled_relations          │
  │ • Contentless FTS5: query_fts (contentless_delete=1) indexing headings and chunks           │
  │ • sqlite-vec: Direct cosine distance queries against note_chunks embedding column           │
  │ • Dynamic Reversible Views:                                                                 │
  │   - unified_edges: 5-column named quad projection (src_id, dst_id, rel_type, weight, chunk)│
  │   - unified_nodes: Canonical node projections with namespace inheritance                    │
  └──────────────────────────────────────┬──────────────────────────────────────────────────────┘
                                         │
                                         ▼
  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
  │ 3. GRAPHRAG RETRIEVAL & SCI-PY DIFFUSION ENGINE (src/yaw/retrieval/retriever.py)             │
  │ • Seed Formulation: p0 = beta1*s_concepts + beta2*s_chunks_vec + beta3*s_chunks_fts        │
  │ • Continuous Math: When embedding is offline (q=0), p0 simplifies to s_chunks_fts           │
  │ • SciPy Sparse CSR Matrix: In-memory row-stochastic transition matrix P                    │
  │ • Hub Dampening: In-degree power scaling (gamma = 0.5)                                      │
  │ • Conductance Asymmetry: 1.0 downward broadcast, 0.2 upward damping (ADR-002, Sprint 5.1)   │
  │ • Power Iteration: p^(t+1) = alpha * p^(t) * P + (1 - alpha) * p0 (alpha = 0.85, 30 iters)  │
  │ • Candidate Extraction: chunks where p*[chunk] > 0; in-memory bi-encoder reranking          │
  │ • Grounding: Zero-copy disk hydration via byte offsets (file.seek, file.read)               │
  │ • Payload: Dual-layer Bridging Subgraph (Ontological Context + Grounded Evidence)           │
  └──────────────────────────────────────┬──────────────────────────────────────────────────────┘
                                         │
                                         ▼
  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
  │ 4. FASTMCP PRODUCTION INTERFACE (src/yaw/server/daemon.py)                                   │
  │ • query_wiki: Full GraphRAG associative retrieval returning dual-layer Bridging Subgraph    │
  │ • ingest_raw: Immutable append-only note ingestion into vault/raw/                          │
  │ • compile_wiki: B-Tree hub splitting, hypernym induction, wiki/ materialization (ADR-002)   │
  │ • lint_wiki: Closed-world SHACL shape validation, contested alias & contradiction linter    │
  └─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Component Deep Dive & Business Logic

### Component 1: Single-Stage Ingestion & Leaf Ontology (`src/yaw/indexer/` & `src/yaw/nlp/`)
1. **Delta Tracking:** CocoIndex's Rust core filesystem watcher monitors `vault/raw/` recursively. File content hashes are maintained internally by CocoIndex; no manual document hash column is required in `note_meta`.
2. **Deterministic Chunking:** Documents are split along semantic boundaries (default ~1000 characters with 150 overlap). Each chunk records:
   - `chunk_id`: sha256 digest of heading + content.
   - `line_start` & `line_end`: 1-based human-readable line ranges for direct disk hydration.
3. **Embedding:** Chunks are embedded using `embed-gemma-300m-FLM` formatted with the document title:
   `title: {doc_title} | text: {chunk_text}`.
4. **Wikilink Extraction:** Regex parser extracts Obsidian `[[wikilinks]]`, mapping them to `wikilinks` records with source and target chunk pointers.
5. **Open IE Extraction & Typing:**
   - Single-pass Open IE ([`docs/adr/ADR-001.md`](file:///home/harri/yaw/docs/adr/ADR-001.md)) extracts relational triples `(subject, predicate, object)`.
   - Freeform predicates are decamelized and lemmatized via native spaCy `Token.lemma_` ([`docs/adr/ADR-006.md`](file:///home/harri/yaw/docs/adr/ADR-006.md)).
   - Predicates align with the **14 Canonical Base Predicates** ([`docs/adr/ADR-008.md`](file:///home/harri/yaw/docs/adr/ADR-008.md)):
     - *Subsumption & Mereology:* `subClassOf`, `partOf`, `instanceOf`
     - *Dialectical & Argumentation:* `supports`, `refutes`, `contradicts`, `supersedes`
     - *Dependency & Process:* `dependsOn`, `causes`, `precedes`, `produces`
     - *Provenance & Grounding:* `derivedFrom`, `cites`, `manifestsIn`
   - Entities are typed into **9 Universal Root Namespaces** ([`docs/adr/ADR-005.md`](file:///home/harri/yaw/docs/adr/ADR-005.md)):
     `entities`, `concepts`, `predicates`, `invariants`, `decisions`, `claims`, `processes`, `sources`, `contexts`.
6. **Two-Tier Convergence Protocol ([`docs/adr/ADR-003.md`](file:///home/harri/yaw/docs/adr/ADR-003.md)):**
   - **Tier 1 (Direct Match):** Exact slug or active alias lookup in `aliases` or `InMemoryLookupKB`. 0 vectors, 0 LLM calls.
   - **Tier 2 (Near Match):** In-process vector similarity > `CANON_SIM_THRESHOLD` ($0.80$) using `en_core_web_lg` static 300d GloVe vectors. Consolidate into existing canonical node. 0 LLM calls.
   - **Fallback:** Create new canonical node with immutable plural root namespace (`concepts:{slug}`).
   - **Zero LLM Adjudication:** 0 LLM calls during entity or predicate canonicalization; procedural pairwise adjudication and external Gemma embedding calls are completely eliminated.

### Component 2: Storage & Schema Contracts (`src/yaw/storage/db.py`)
Single SQLite database at `sample-llm-wiki/.cache/yaw.db` governed by WAL pragmas:

- **`note_meta`**: `(rel_path TEXT PRIMARY KEY, title TEXT, category TEXT, modified_at REAL)`.
- **`note_chunks`**: `(chunk_id TEXT PRIMARY KEY, entity_uri TEXT, rel_path TEXT, ordinal INTEGER, heading TEXT, heading_slug TEXT, line_start INTEGER, line_end INTEGER, embedding BLOB/JSON)`.
- **`wikilinks`**: `(source_uri TEXT, target_uri TEXT, link_label TEXT, PRIMARY KEY(source_uri, target_uri))`.
- **`leaf_nodes`**: `(node_id TEXT, source_leaf_id TEXT, name TEXT, namespace TEXT, definition TEXT, embedding BLOB/JSON, PRIMARY KEY(node_id, source_leaf_id))`.
- **`leaf_relations`**: `(relation_id TEXT PRIMARY KEY, source_leaf_id TEXT, subject_id TEXT, predicate TEXT, object_id TEXT, weight REAL, predicate_id TEXT)`.
- **`aliases`** ([`docs/adr/ADR-003.md`](file:///home/harri/yaw/docs/adr/ADR-003.md)):
  ```sql
  CREATE TABLE IF NOT EXISTS aliases (
      alias_id TEXT NOT NULL,
      kind TEXT NOT NULL CHECK(kind IN ('entity', 'predicate')),
      canonical_id TEXT NOT NULL,
      method TEXT NOT NULL CHECK(method IN ('exact', 'alias', 'lexical', 'auto', 'llm', 'compile', 'new')),
      score REAL NOT NULL DEFAULT 1.0 CHECK(score >= 0.0 AND score <= 1.0),
      status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'contested')),
      source_leaf_id TEXT,
      resolved_at REAL DEFAULT (unixepoch()),
      evidence TEXT,
      PRIMARY KEY (alias_id, kind)
  );
  ```
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
  LEFT JOIN aliases a_src ON a_src.alias_id = w.source_uri AND a_src.kind = 'entity' AND a_src.status = 'active'
  LEFT JOIN aliases a_dst ON a_dst.alias_id = w.target_uri AND a_dst.kind = 'entity' AND a_dst.status = 'active'
  UNION ALL
  SELECT
      COALESCE(a_sub.canonical_id, lr.subject_id) AS src_id,
      COALESCE(a_obj.canonical_id, lr.object_id) AS dst_id,
      COALESCE(a_pred.canonical_id, lr.predicate) AS rel_type,
      lr.weight AS weight,
      lr.source_leaf_id AS source_chunk_id
  FROM leaf_relations lr
  LEFT JOIN aliases a_sub ON a_sub.alias_id = lr.subject_id AND a_sub.kind = 'entity' AND a_sub.status = 'active'
  LEFT JOIN aliases a_obj ON a_obj.alias_id = lr.object_id AND a_obj.kind = 'entity' AND a_obj.status = 'active'
  LEFT JOIN aliases a_pred ON a_pred.alias_id = lr.predicate_id AND a_pred.kind = 'predicate' AND a_pred.status = 'active'
  UNION ALL
  SELECT subject_id AS src_id, object_id AS dst_id, predicate AS rel_type, weight, 'compiled' AS source_chunk_id
  FROM compiled_relations
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
  LEFT JOIN aliases a ON a.alias_id = ln.node_id AND a.kind = 'entity' AND a.status = 'active'
  UNION ALL
  SELECT node_id, name, namespace, definition FROM compiled_nodes
  UNION ALL
  SELECT curie AS node_id, label AS name, ontology_id AS namespace, label AS definition FROM dynamic_classes;
  ```

### Component 3: GraphRAG PPR Retrieval Engine (`src/yaw/retrieval/`)
1. **Sparse Adjacency Matrix:** Reads `unified_edges` and constructs an in-memory SciPy Compressed Sparse Row (`csr_matrix`) structure.
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
   - Hub degree dampening exponent: $\gamma = 0.5$.
3. **Seed Vector Formulation ($\mathbf{p}_0$):**
   $$\mathbf{p}_0 = \beta_1 \cdot \mathbf{s}_{\text{concepts}} + \beta_2 \cdot \mathbf{s}_{\text{chunks\_vec}} + \beta_3 \cdot \mathbf{s}_{\text{chunks\_fts}}$$
   Where $\beta_1 = 0.60$ (concept canopy bias), $\beta_2 = 0.20$, $\beta_3 = 0.20$.
   - When EmbeddingGemma is offline ($\mathbf{q} = \mathbf{0}$): $\mathbf{s}_{\text{concepts}} = \mathbf{0}$ and $\mathbf{s}_{\text{chunks\_vec}} = \mathbf{0}$, simplifying continuously to $\mathbf{p}_0 = \mathbf{s}_{\text{chunks\_fts}}$ (pure FTS5 seeding without artificial fallback branches).
4. **Power Iteration Diffusion:**
   $$\mathbf{p}^{(t+1)} = \alpha \mathbf{p}^{(t)} \mathbf{P} + (1 - \alpha) \mathbf{p}_0, \quad \alpha = 0.85 \quad (30 \text{ iterations})$$
   Stationary distribution $\mathbf{p}^*$ accumulates mass across the graph topology (`manifests_in`, `mentions`, `wikilinks`, semantic predicates), including unseeded chunks connected to active concepts.
5. **Candidate Chunk Selection & Bi-Encoder Similarity Rescoring:**
   - Candidate chunks are drawn directly from the converged distribution $\mathbf{p}^*$ for nodes where $\text{node} \in \text{chunks}$ and $\mathbf{p}^*[\text{chunk}] > 0$.
   - When $\mathbf{q} \ne \mathbf{0}$: embeds query once using `embed_query()`, computes $\text{sim}(q, d) = \mathbf{q} \cdot \mathbf{d}$ against pre-indexed SQLite chunk embeddings, and ranks by $\text{Score} = \mathbf{p}^*[\text{chunk}] \times \text{sim}(q, d)$ with filter $\text{sim}(q, d) \ge \text{RELEVANCE\_THRESHOLD}$ ($0.10$).
   - When $\mathbf{q} = \mathbf{0}$: ranks candidates directly by converged graph mass $\text{Score} = \mathbf{p}^*[\text{chunk}]$.
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
   Triggers asynchronous background indexing, ontology re-consolidation, and B-Tree hierarchical deepening ([`docs/adr/ADR-002.md`](file:///home/harri/yaw/docs/adr/ADR-002.md)).
   - **Delta Ingestion:** By default (`force=False`), it only processes newly modified/added raw notes via CocoIndex content-hash deltas.
   - **B-Tree Cluster Splitting & Hypernym Inducing:** When any hub node accumulates out-degree $> K_{\max} = 10$, SciPy Ward's HAC partitions the subtree. LLM induces a hypernym superclass, rewiring `member --[subClassOf]--> hypernym` and `hypernym --[subClassOf]--> parent_hub` with asymmetric conductance ($1.0$ downward broadcast, $0.2$ upward damping).
   - **Deterministic View Materialization:** Emits `vault/wiki/{namespace}/{slug}.md` pages directly from SQLite `unified_nodes` (frontmatter) and `unified_edges` (`[[wikilinks]]`) with zero LLM authoring.
4. **`lint_wiki(query: str | None = None) -> str | list[dict]` (SHACL Graph & Conflict Linter):**
   Surfaces contradictions, contested claims, dialectical disputes, and schema violations in the ontology layer via in-process `pyshacl` / `rdflib` against SQLite views ([`docs/rdf-owl-shacl.md`](file:///home/harri/yaw/docs/rdf-owl-shacl.md)).

#### 2. Development & Diagnostic Probes (Internal / Debugging Only)
- **`find_seed(query: str, top_k: int = 3) -> list[dict]`:** Inspects intermediate candidate generation prior to graph traversal.
- **`read_document(path: str, line_start: int | None = None, line_end: int | None = None) -> str`:** Direct line-based disk hydration reader for testing.
- **`get_ontology_schema() -> dict`:** Low-level introspection tool returning database statistics, active taxonomy namespaces, and base predicates.

### Component 5: W3C Semantic Web Substrate (RDF, OWL, SKOS, SHACL)

YAW formalizes knowledge representation by anchoring in-process SQLite graph views to W3C Semantic Web standards ([`docs/adr/ADR-008.md`](file:///home/harri/yaw/docs/adr/ADR-008.md), [`docs/rdf-owl-shacl.md`](file:///home/harri/yaw/docs/rdf-owl-shacl.md)). To eliminate architectural confusion and prevent semantic collisions, each standard governs a strictly partitioned, non-overlapping concern:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        W3C SEMANTIC STACK ARCHITECTURAL ROLES                          │
├───────────────────┬────────────────────────────────────────────────────────────────────┤
│ Standard          │ System Role & Invariant Responsibility                             │
├───────────────────┼────────────────────────────────────────────────────────────────────┤
│ RDF 1.1           │ Graph Data Model & Quad Substrate (unified_edges, unified_nodes)   │
│ OWL 2 RL / RDFS   │ Domain Schema Logic, Subsumption & Identity (sameAs / differentFrom)│
│ SKOS              │ Epistemic Categorization, Vault Thesaurus & Controlled Vocabularies │
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
- **Reversible Identity & Distinction Mapping:**
  - Active aliases project seamlessly to `owl:sameAs` equivalence:
    $$(a, \text{status}=\text{'active'}, c) \implies a \equiv_{\text{owl:sameAs}} c$$
  - Distinct calibration verdicts project to `owl:differentFrom` distinction:
    $$(a, \text{verdict}=\text{'distinct'}, c) \implies a \not\equiv_{\text{owl:differentFrom}} c$$
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
- **Execution Engine:** Evaluated in-process via `pyshacl` over the relational knowledge graph (`src/yaw/ontology/shacl.py`), surfaced through `@mcp.tool() async def lint_wiki()` in FastMCP.
- **Closed-World Assumption (CWA):** While OWL operates under Open-World Assumption (missing info is unknown), SHACL operates under Closed-World Assumption with Unique Name Assumption (UNA) at validation time: unfulfilled constraints are explicit actionable violations.
- **Core Declarative Shapes (`src/yaw/ontology/shapes/wiki_shapes.ttl`):**
  - **`yaw:ConceptShape`:** Requires non-empty definitions conforming to word limits ($\le 50$ words via `config.DEFINITION_MAX_WORDS`) and canonical URI syntax across universal root namespaces.
  - **`yaw:RelationShape`:** Enforces referential integrity on subjects, predicates, and objects, detecting dangling edges and unregistered namespaces.
  - **`yaw:DialecticalIntegrityShape`:** Flags dialectical contradictions (`causes` vs `prevents`, `supports` vs `refutes`, `supports` vs `contradicts`, reciprocal conflicts) and contested alias merges (`aliases.status = 'contested'`).

---

## 6. Repository Architecture & Directory Boundaries

The codebase enforces strict separation between engine execution, test verification, system documentation, and vault data:

```txt
yaw/
├── src/yaw/                  # Core Engine Implementation (Single Source of Truth)
│   ├── config.py             # Authoritative configuration, 9 root namespaces, 14 base predicates
│   ├── storage/              # SQLite SSOT: DB connection, PRAGMAs, sqlite-vec, fts5, views (db.py)
│   ├── indexer/              # CocoIndex pipeline, markdown chunking, disk byte offsets (pipeline.py)
│   │   └── canonicalize.py   # Two-Tier Convergence engine, aliases, in-process GloVe vectors
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
│   ├── test_canonicalization.py # Invariant tests: Two-Tier Convergence resolver, aliases, view rerouting
│   ├── test_cocoindexer.py   # Invariant tests: disk byte offsets, contentless FTS, views
│   ├── test_embedding_gemma.py # EmbeddingGemma prompt formatting and bi-encoder reranking
│   ├── test_leaf_pipeline.py # spaCy + spacy.NER.v3, Doc integrity, Open IE, predicate normalization
│   ├── test_provenance.py    # Doc integrity, line ranges, byte hydration
│   ├── test_retriever.py     # SciPy CSR matrix, directional conductance, PPR convergence
│   ├── test_sprint5_1_alignment.py # Sprint 5.1 alignment invariant verification (WP 5.1.1–5.1.7)
│   ├── test_vault_purity.py  # Vault purity tests asserting zero engine specs in raw/
│   └── test_yaw.py           # FastMCP server integration and end-to-end tool execution
│
├── docs/                     # Engine Specifications, Architectural Blueprints & Playbooks
│   │                         # (SSOT for developer documentation; kept strictly outside the vault!)
│   ├── ADR.md                # Architectural Decision Records ledger (ADR-001 through ADR-008)
│   ├── adr/                  # Authoritative ADR files (ADR-001.md ... ADR-008.md)
│   ├── alignment-sprint-for-e5s2.md # Sprint 5.1 turnkey execution DAG & prompt cards
│   ├── epics-alignment-roadmap.md # Authoritative engineering implementation roadmap (7 Epics)
│   ├── rdf-owl-shacl.md      # Authoritative W3C Semantic Web integration manual
│   ├── USER_STORIES.md       # BDD user stories across Epics 5, 6, 7
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
