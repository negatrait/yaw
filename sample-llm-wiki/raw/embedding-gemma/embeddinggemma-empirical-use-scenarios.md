# EmbeddingGemma: Empirical Evaluation & Practical Use Scenarios

**Model Identifier:** `google/embeddinggemma-300M` (`embed-gemma-300m-FLM` via Ollama)  
**Architecture:** 300M-parameter dense bi-encoder with Matryoshka Representation Learning (MRL)  
**Embedding Geometry:** 768-dim float32 unit-normalized vectors (truncatable to 256/512)  
**Evaluation Target:** Knowledge graph indexing, vector search, PPR GraphRAG through the ontology layer, and ontology extraction in `yaw`.

---

## 1. Architectural Boundary & Core Constraint

EmbeddingGemma is strictly a **dense bi-encoder (encoder-only)**:
* **Input:** Raw text prefixed with a task-specific prompt string.
* **Output:** A single dense vector $\mathbf{v} \in \mathbb{R}^{768}$ with $\|\mathbf{v}\|_2 = 1.0$.
* **Boundary:** It possesses no autoregressive decoder. It **cannot generate text**, cannot propose token span boundaries, and cannot emit structured AST/JSON schemas (`LeafTaxonomy`, triples, definitions) from unstructured prose.

Any taxonomy extraction pipeline must separate **span proposal** (generative LLM or deterministic rule-based NLP) from **vector representation/classification**.

---

## 2. Measured Outcomes & Empirical Evidence

### A. Stage 1 Chunk Vectorization (`Retrieval-document`)
* **Prompt Format:** `title: none | text: {chunk_content}`
* **Execution Engine:** Batched via CocoIndex `@coco.fn.as_async(batching=True, max_batch_size=64)` over Ollama `/api/embed`.
* **Measured Performance:**
  - Amortized latency: ~2.1ms per chunk.
  - 162 chunks indexed across 15 technical documents in under 1.2s.
* **Empirical Finding (Prefix Tuning):**
  - Injecting synthetic document titles (`title: {stem} | text: ...`) degraded cosine discrimination against out-of-domain chunks.
  - The canonical `title: none | text: {content}` prompt yielded the highest discriminatory power across heterogeneous technical documentation (GraphRAG, spaCy, PPR GraphRAG through the ontology layer, Karpathy LLM-Wiki).

### B. Query Retrieval (`Retrieval-query`)
* **Prompt Format:** `task: search result | query: {query_text}`
* **Measured Performance:**
  - Inference latency: ~12ms per query on CPU/Ollama.
  - sqlite-vec `vec_distance_cosine(embedding, ?)` retrieval: achieves continuous top similarity scores ($\ge 0.85$ on target concepts vs $0.60 - 0.72$ on background noise).

### C. Bi-Encoder Reranking (`Reranking`)
* **Prompt Format:** `task: search result | query: {query_text}`
* **Measured Performance vs BGE Cross-Encoder:**
  - **Memory & VRAM:** Eliminated the separate 1.5GB `bge-reranker-v2-m3-GGUF` model from memory completely.
  - **Latency:** Cross-encoder joint token forward pass took 180ms–400ms for 40 candidate chunks. EmbeddingGemma bi-encoder cosine dot product takes **< 2ms** (reusing cached document chunk embeddings; only 1 query embedding needed).
  - **Ranking Fidelity:** In benchmark queries (`"Personalized PageRank"`, `"spaCy NER pipeline"`, `"CocoIndex dataflow"`), the bi-encoder cosine similarity yielded high ranking accuracy matching the ground truth relevance ordering.
  - **Score Formulation:** $\text{FinalScore} = \text{PPR\_Probability} \times \cos(\mathbf{q}_{\text{rerank}}, \mathbf{d}_{\text{chunk}})$.

### D. Zero-Shot Namespace Classification (`Classification`)
* **Prompt Format:** `task: classification | query: {text}`
* **Evaluation:** Tested on isolated technical phrases against ontology namespaces (`entities`, `concepts`, `predicates`, `claims`, `decisions`).
* **Measured Outcome:**
  - Cosine similarities across candidate label definitions were tightly compressed in the $[0.70, 0.92]$ band.
  - Because isolated noun chunks lack grammatical context, short phrases can collapse into the wrong class (e.g., `"FastMCP"` scoring 0.770 on `predicates` vs 0.742 on `entities`).
  - **Conclusion:** `task: classification` is effective for coarse-grained document routing or user intent classification, but noisy for fine-grained isolated noun-phrase taxonomy labeling without contextual framing.

---

## 3. High-Value vs Anti-Pattern Use Scenarios

| Scenario | Suitability | Optimal Prompt Prefix | Technical Rationale |
| :--- | :--- | :--- | :--- |
| **Document Chunk Indexing** | **Optimal** | `title: none \| text: ` | High throughput, sub-3ms batched vector generation, clean 768-dim unit sphere. |
| **Dense Search Querying** | **Optimal** | `task: search result \| query: ` | Aligns asymmetric query embedding with document embedding space. |
| **First-Stage Reranking** | **Optimal** | `task: search result \| query: ` | Replaces cross-encoder. Dot product against pre-computed doc vectors in microsecond regime. |
| **Entity Resolution & Linking** | **Strong** | `task: sentence similarity \| query: ` | Maps surface entity mentions (`"PPR"`, `"Personalized PageRank"`) to canonical nodes. |
| **Cross-Doc Topic Clustering** | **Strong** | `task: clustering \| query: ` | Generates cluster-friendly vectors for Stage 2 global wiki ontology synthesis. |
| **Triples / Relation Extraction** | **Anti-Pattern** | N/A (Unsupported) | Bi-encoders cannot emit relational tokens or extract structured triples from raw text. |
| **Zero-Shot Isolated POS Tagging** | **Marginal** | `task: classification \| query: ` | Compressed cosine separation on out-of-context tokens; rule-based spaCy POS is 100x faster and deterministic. |

---

## 4. Recommended Engine Architecture

```
                    ┌─────────────────────────┐
                    │    Raw Markdown Note    │
                    └───────────┬─────────────┘
                                │
               ┌────────────────┴────────────────┐
               ▼                                 ▼
   [Stage 1: Vector Search]          [Stage 1: Graph & Entities]
   Chunk via RecursiveSplitter       spaCy Linguistic Parser
   Embed with EmbeddingGemma         Extract Noun Spans & Heads
   (Retrieval-document)              Zero-Shot / Similarity Link
               │                     (task: sentence similarity)
               ▼                                 ▼
     sqlite-vec Table                 Ontology Diffusion Graph
   (768 float32 embeddings)          (next, child_of, mention edges)
               │                                 │
               └────────────────┬────────────────┘
                                │
                                ▼
                       PPR GraphRAG Retrieval
                      1. FTS + Vector Seed
                      2. PPR Diffusion Matrix (Ontology Layer)
                      3. EmbeddingGemma Bi-Encoder Rerank
```
