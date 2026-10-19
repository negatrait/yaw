# EnterpriseRAG-Bench Training & Evolution Plan

**Status**: Planning / Spec  
**Target Milestone**: Post-Parity Evolution (Retiring Upstream `hipporag`)  
**Owner**: [`FOUNDATIONAL.md`](../FOUNDATIONAL.md) | Related: [`docs/pre-epic2-memo.md`](pre-epic2-memo.md), [Epic 1](epic-1-hipporag-daemon.md)

---

## 1. Architectural Philosophy: Decoupled Training & Eval

To avoid benchmark contamination and prompt overfitting:
1. **Academic Evaluation Corpus**: The 7 paper datasets (`popqa`, `lveval`, `narrativeqa`, `2wikimultihopqa`, `hotpotqa`, `nq`, `musique`) serve as the **frozen acceptance gate**.
2. **Enterprise Optimization Corpus**: [EnterpriseRAG-Bench](https://arxiv.org/html/2605.05253v2) (`onyx-dot-app/EnterpriseRAG-Bench`) serves as the **training and prompt compilation corpus** for DSPy GEPA / MIPROv2 and the Reflection RLM.

### The Upstream Retirement Principle
We do not make premature claims about production superiority. The progression is strictly gated:
- **Gate 1**: Achieve verified parity or beat against upstream `hipporag` on the 7 paper datasets under the unified local model stack.
- **Gate 2**: Once parity is proven and accepted, **scrap all upstream `hipporag` dependencies** and retire all legacy 7-file Parquet/Pickle/JSON storage remnants from the codebase.
- **Gate 3**: With upstream ancestry cleanly removed, evolve YAW natively by training the **Canonical 12 DSPy signatures + EmbeddingGemma task prefixes** on EnterpriseRAG-Bench, verifying zero-shot generalizability back against the paper benchmark suite.

---

## 2. Dataset Profile: EnterpriseRAG-Bench

- **Source Repository**: [`onyx-dot-app/EnterpriseRAG-Bench`](https://github.com/onyx-dot-app/EnterpriseRAG-Bench) (Hugging Face: `onyx-dot-app/EnterpriseRAG-Bench`).
- **Domain**: Fictional AI provider ("Redwood Inference") simulating realistic internal company knowledge.
- **Source Modalities (9 Tools)**: Slack threads, Gmail messages, Linear tickets, Google Drive docs, HubSpot CRM records, Fireflies meeting transcripts, GitHub code/issues, Jira tickets, and Confluence wiki spaces.
- **Key Challenges Stressed**:
  - **Entity Homographs & Acronyms**: Disambiguating "Redwood" (company vs. project vs. repository vs. Slack channel) directly exercises YAW's SKOS entity definitions (`skos:definition`, `entity_labels`, `owl:differentFrom`).
  - **Temporal & Source Conflicts**: Conflicting information between outdated design documents and recent Slack/ticket decisions forces the graph and triple extractor to preserve provenance.
  - **Unanswerable Queries**: 10 question categories include unanswerable cases, training the QA reader to detect missing graph paths rather than hallucinating false bridges.

---

## 3. Stratified Training Slice (`scripts/download_enterprise_rag_bench.py`)

A standalone script will download and curate a balanced training slice from Hugging Face:

```powershell
uv run python scripts/download_enterprise_rag_bench.py --num-train-queries 60 --num-val-queries 30 --distractor-factor 2.0
```

### 3.1 Sampling Strategy
- **Questions ($N=90$)**:
  - 60 Train / 30 Validation questions sampled evenly across the 10 categories:
    1. Single-document lookup
    2. Multi-document synthesis
    3. Constrained temporal / organizational retrieval
    4. Conflict resolution
    5. Unanswerable / absent information
    6. Code / technical schema queries
    7. Meeting transcript action-item queries
    8. Policy / HR compliance queries
    9. Cross-tool reconciliation (e.g. Jira + Slack)
    10. Ambiguous / clarification queries
- **Corpus Slice ($M \approx 600 - 1,200\text{ documents}$)**:
  - All gold supporting documents for the 90 sampled queries.
  - Hard distractor documents from the same tools and author clusters (2× distractor padding).

### 3.2 Ingestion via Native CocoIndex
Ingest the curated slice directly into `indexes/enterprise_train/yaw.db` using textbook CocoIndex AST chunking (`CodeSource`, `RecursiveSplitter`, `detect_code_language`):
- Markdown, HTML, JSON, and source code chunked with zero intermediate files.
- Each chunk pointer records exact source line spans and Tree-sitter section headings.

---

## 4. The Canonical 12 DSPy Signatures + First-Class Embedding Trail

Per `/dspy-rlm-module` discipline and `FOUNDATIONAL.md`, YAW strictly forbids redundant pass-through RLM wrappers (`execution_mode` and `qa_mode` are retired). Only the **Canonical 12 `dspy.Signature` classes** + the **First-Class Embedding Telemetry Trail** (`DSPyEmbeddingAdapter`) exist:
- **9 Runtime Stage Signatures in `HippoRAGProgram`** (7 core direct GBNF signatures + 2 SHACL integrity healing signatures).
- **3 Reflection Optimizer Signatures in `RLMInstructionProposer`** (single-level RLM trace distiller + GBNF instruction synthesizer/compressor).

Every predictor owns a unique Signature class, a dedicated `_PredictorBoundLM` client, and a dedicated `track_usage()` telemetry stream, alongside `DSPyEmbeddingAdapter`'s first-class embedding telemetry hook (`HippoRAGProgram.telemetry` & `HippoRAGProgram.embedding_audit_trail`):

| # | Component | Predictor / Adapter Attribute | Role / Concern | Unique `dspy.Signature` / Profile | Execution Mode |
| :- | :--- | :--- | :--- | :--- | :--- |
| **1** | `HippoRAGProgram` | `ner` | Core: Named Entity & Definition Extraction | `ExtractEntities` | Direct GBNF JSON |
| **2** | `HippoRAGProgram` | `triples` | Core: Polymorphic OpenIE Statement Extraction | `ExtractTriples` | Direct GBNF JSON |
| **3** | `HippoRAGProgram` | `entity_definer` | Core: Missing Entity Definition Synthesis | `DefineMissingEntities` | Direct GBNF JSON |
| **4** | `HippoRAGProgram` | `synonym_judge` | Core: 4-Way SKOS/OWL Synonym Disambiguation | `JudgeSynonyms` | Direct GBNF JSON |
| **5** | `HippoRAGProgram` | `query_intent` | Core: Query Intent & Seed Expansion | `ExtractQueryIntent` | Direct GBNF JSON |
| **6** | `HippoRAGProgram` | `triple_filter` | Core: Recognition Memory Triple Reranker | `FilterTriples` | Direct GBNF JSON |
| **7** | `HippoRAGProgram` | `qa` | Core: Multi-Hop QA Reader | `AnswerQuestion` | Direct GBNF JSON |
| **8** | `HippoRAGProgram` | `shacl_healer_rlm` | SHACL: In-Transaction Graph Healer Loop | `HealSHACLViolations` | Single-Level `dspy.RLM` REPL (`dspy.PythonInterpreter`) |
| **9** | `HippoRAGProgram` | `shacl_repair` | SHACL: Inner Closed-World Repair Tool | `ProposeSHACLRepairs` | Direct GBNF JSON |
| **10** | `RLMInstructionProposer` | `inner_rlm` (`rlm`) | GEPA: Trace Distiller REPL Loop | `ReflectiveInnerDistillerSignature` | Single-Level `dspy.RLM` REPL (`dspy.PythonInterpreter`) |
| **11** | `RLMInstructionProposer` | `qwen_proposer` | GEPA: Instruction Synthesizer | `ProposeInstructionSignature` | Direct GBNF JSON |
| **12** | `RLMInstructionProposer` | `qwen_compressor`| GEPA: Instruction Length Compressor | `CompressInstructionSignature` | Direct GBNF JSON |
| **+E** | `DSPyEmbeddingAdapter` | `embedder` | First-Class Embedding Trail (`chunk_embed`, `entity_embed`, `fact_embed`, `query_fact_embed`, `query_passage_embed`) | `EmbeddingProfile` (`query_prefix`, `fact_query_prefix`, `document_prefix`) | `dspy.Embedder` + Telemetry Hook |

---

## 5. Optimization & Verification Protocol

### 5.1 Optimization Execution
Run DSPy GEPA optimization using the gate-free Pareto metric:
```powershell
uv run python scripts/optimize_dspy_pipeline.py `
    --dataset enterprise_rag_bench `
    --train-queries 60 `
    --val-queries 30 `
    --max-rounds 3 `
    --output .yaw/compiled_program_enterprise.json
```

- **Fitness Metric**: $\text{Fitness} = \frac{\text{GM}(\text{Goods})}{\text{GM}(\text{Bads})}$
  - Goods: Retrieval Recall@1..20, Answer F1, Exact Match, SHACL Conformance.
  - Bads: Normalized Token Volume (including LLM + Embedding tokens), SHACL Violations, Missing Entity Definitions, Repair Tokens.
- **Reflection Proposer**: Qwen in the single-level sandboxed Python REPL (`RLMInstructionProposer.inner_rlm`) analyzes execution failure traces and embedding telemetry trails on enterprise documents and proposes prompt/prefix mutations across the Canonical 12 signatures and `EmbeddingGemma` task prefixes.

### 5.2 Zero-Shot Evaluation Transfer Gate
Evaluate the compiled enterprise program `.yaw/compiled_program_enterprise.json` directly against the 7 paper datasets:

```powershell
uv run python scripts/verify_paper_parity_gate.py `
    --compiled-program .yaw/compiled_program_enterprise.json `
    --queries-per-dataset 25 `
    --index-queries 35 `
    --mode eval
```

- **Acceptance Criterion**: The enterprise-compiled program must match or beat uncompiled baseline performance across the 7 paper datasets without ever having been exposed to Wikipedia training samples.
