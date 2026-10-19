# YAW: HippoRAG 2 Graph & Storage Engine

YAW is a native HippoRAG 2 graph & storage engine backed by a single SQLite database (`yaw.db`), CocoIndex (`cocoindex>=1.0.0`) single-parse Rust Tree-sitter AST ingestion, and DSPy GBNF/RLM/GEPA inference. Read [FOUNDATIONAL](FOUNDATIONAL.md), then the [daemon epic](docs/epic-1-hipporag-daemon.md) and [Streamable HTTP epic](docs/epic-2-fastmcp-http.md). Reference papers are preserved locally under [HippoRAG 1 (`2405.14831v3`)](docs/arxiv-2405.14831v3-hipporag.md) and [HippoRAG 2 (`2502.14802v1`)](docs/arxiv-2502.14802v1-hipporag2.md).

## Available now

Use Python `>=3.14.7` (`.python-version` `3.14.7`) and `uv`:

```powershell
uv sync --locked
uv run --locked pytest -v
uv run --locked python scripts/download-datasets.py --dataset enterprise_rag_bench
uv run --locked yaw daemon status
uv run --locked yaw daemon run
uv run --locked yaw daemon start
uv run --locked yaw daemon stop
uv run --locked yaw graph build
uv run --locked yaw graph rebuild
uv run --locked yaw graph lint
uv run --locked yaw graph lint --turtle
uv run --locked yaw optimize
uv run --locked yaw load-program .yaw/compiled_program_optimize_<datestamp>.json
uv run --locked yaw config get optimization.auto
uv run --locked yaw config set optimization.auto heavy
uv run --locked python benchmark.py --dataset sample --mode all
uv run --locked python benchmark.py --dataset 2wikimultihopqa --max-queries 25
uv run --locked python benchmark.py --dataset hotpotqa --max-queries 25 --mode retrieval
uv run --locked python benchmark.py --dataset musique --max-queries 25 --mode retrieval
uv run --locked python scripts/verify_paper_parity_gate.py
```

`benchmark.py` and `scripts/verify_paper_parity_gate.py` evaluate the native `yaw` engine (`YawGraphEngine` + `yaw.db` + `HippoRAGProgram`) against Standard DPR (`retrieve_dpr`) and two consolidated reference targets across the 7 HippoRAG 2 paper datasets (`popqa`, `lveval`, `narrativeqa`, `2wikimultihopqa`, `hotpotqa`, `nq`, `musique`):
- **Target 1 (`MONKEYPATCHED_HIPPORAG_BASELINES`):** Consolidated historical runs of upstream HippoRAG 2 on the exact same `Qwen3-4B-Instruct-2507-GGUF` + `embeddinggemma-300m-GGUF` local stack (archived in [`outputs/benchmarks/consolidated_historical_baselines.json`](outputs/benchmarks/consolidated_historical_baselines.json)).
- **Target 2 (`PAPER_BASELINES`):** Published HippoRAG 2 paper results (`Llama-3.3-70B-Instruct` + `NV-Embed-v2`, `arXiv:2502.14802v1`).

Upstream `hipporag` and `optuna` are permanently retired from `pyproject.toml`, `uv.lock`, and runtime/test code.

The unified local model stack (`http://localhost:13305/api/v1`) uses:
- **Embeddings (`--embedding-model`):** `embeddinggemma-300m-GGUF` on GPU `llamacpp` (`--embedding-task-mode` defaults to `distinct-titled` under `retrieval-prompts-v3`).
- **Unified Chat, GBNF Worker, QA Reader & RLM Orchestrator (`--extraction-model`, `--qa-model`):** `Qwen3-4B-Instruct-2507-GGUF` on GPU `llamacpp` (`--parallel 1 --cache-ram 0`) with pure-JSON GBNF grammar enforcement (`_yaw_supports_gbnf=True`) for direct/worker predictors and unconstrained Python REPL execution (`_yaw_supports_gbnf=False`) inside DSPy's managed Deno/Pyodide WASM sandbox (`dspy[deno]`, `dspy.PythonInterpreter`) for `shacl_healer_rlm` and `RLMInstructionProposer.inner_rlm`.
- **OOD Enterprise Prompt Evolution (`uv run yaw optimize`):** Configured by default in [`yaw.toml`](yaw.toml) to train on the stratified 250-query (`150` train + `100` val) **EnterpriseRAG-Bench** (`enterprise_rag_bench`, `onyx-dot-app/EnterpriseRAG-Bench`) dataset across all 10 reasoning categories (`Basic`, `Semantic`, `Intra-Document Reasoning`, `Project Related`, `Constrained`, `Conflicting Info`, `Completeness`, `Miscellaneous`, `Info Not Found`, `High Level`) with $2.0\times$ same-tool hard distractors (`mode = "gepa"`, `auto = "heavy"`).

On this machine the models and settings are already installed. For a fresh Lemonade installation, register/download the embedding checkpoint and save both `llamacpp` model settings explicitly before running benchmarks (`--batch-size 2048 --ubatch-size 2048` on EmbeddingGemma prevents `llama-server`'s default 512-token physical microbatch cutoff on 512–2048 token passages, and `--cache-ram 0` without `-dio` on `Qwen3-4B-Instruct-2507-GGUF` prevents Windows Direct-IO and host-RAM prompt-cache growth):

```powershell
$api = 'http://localhost:13305/api/v1'
$pull = @{
    model_name = 'user.embeddinggemma-300m-GGUF'
    checkpoint = 'ggml-org/embeddinggemma-300M-GGUF:embeddinggemma-300M-Q8_0.gguf'
    recipe = 'llamacpp'
    embedding = $true
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "$api/pull" -ContentType 'application/json' -Body $pull
$loadEmbed = @{
    model_name = 'user.embeddinggemma-300m-GGUF'
    ctx_size = 2048
    llamacpp_backend = 'cuda'
    llamacpp_args = '--pooling mean --parallel 1 --batch-size 2048 --ubatch-size 2048'
    save_options = $true
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "$api/load" -ContentType 'application/json' -Body $loadEmbed
$loadChat = @{
    model_name = 'Qwen3-4B-Instruct-2507-GGUF'
    ctx_size = 16384
    llamacpp_backend = 'cuda'
    llamacpp_args = '--parallel 1 --cache-ram 0'
    save_options = $true
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "$api/load" -ContentType 'application/json' -Body $loadChat
```

## Measured versus planned

Across the consolidated benchmark evaluations ([`outputs/benchmarks/consolidated_historical_baselines.json`](outputs/benchmarks/consolidated_historical_baselines.json) and [evaluation](docs/evaluation.md)):
- **Native YAW vs. Target 1 (Monkeypatched Upstream HippoRAG 2 on `Qwen3-4B`):**
  - `popqa` (25 queries): YAW `Recall@5 = 0.9600`, `F1 = 0.7867` (`+0.0448` F1 vs. upstream HippoRAG 2 `0.7419`; `2.3x` faster wall-clock).
  - `lveval` (25 queries): YAW `Recall@5 = 0.9600`, `F1 = 0.5965` (`+0.0100` R@5, `+0.0215` F1 vs. upstream HippoRAG 2).
  - `narrativeqa` (25 queries): YAW `Recall@5 = 0.8800`, `F1 = 0.4399` (`+0.0800` R@5 vs. upstream HippoRAG 2 `0.8000`).
  - `2wikimultihopqa` (22 queries): YAW `Recall@5 = 0.9091`, `F1 = 0.6914` (`1.6x` faster indexing/eval).
- **Native YAW vs. Target 2 (Published Paper 70B `Llama-3.3-70B` + `NV-Embed-v2`):**
  - `popqa`: YAW `Recall@5 = 0.9600` vs. Paper `0.5170` (**`+44.30%`**), `F1 = 0.7867` vs. Paper `0.5620` (**`+22.47%`**).
  - `lveval`: YAW `F1 = 0.5965` vs. Paper `0.1290` (**`+46.75%`**).
  - `narrativeqa`: YAW `F1 = 0.4399` vs. Paper `0.2590` (**`+18.09%`**).
  - `2wikimultihopqa`: YAW `Recall@5 = 0.9091` vs. Paper `0.9040` (**`+0.51%`**).

161 offline and live-silicon integration regression tests pass (`uv run --locked pytest -q`). See [evaluation](docs/evaluation.md) for benchmark runs, closed-world SHACL validations, and live-silicon GEPA optimization results.

The native HippoRAG 2 engine (`yaw/daemon/graph/engine.py`), persistent single-owner daemon runtime (`yaw/daemon/runtime.py`), single `yaw.db` SQLite storage (`sqlite-vec` + FTS5 + sparse CSR PPR), native CocoIndex single-parse Rust Tree-sitter AST ingestion across 30+ formats with declarative $\pi_{\mathcal{W}, \mathcal{B}}$ key projection, title-aware `EmbeddingGemma` formatting, Dai & Callan MaxP passage aggregation, and DSPy GEPA prompt evolution with hot-reload program promotion are implemented and verified. Epic 2 (FastMCP over Streamable HTTP) follows as the next planned phase.

## Native W3C Semantic Graph & Knowledge Representation (`yaw.db`)

YAW replaces upstream HippoRAG 2's flat, unvalidated 7-file Parquet/Pickle/JSON storage with an in-process, closed-world SQLite graph database ([`yaw/daemon/graph/storage/db.py`](yaw/daemon/graph/storage/db.py)) that natively operationalizes W3C Semantic Web recommendations (SKOS, RDF 1.1, OWL 2, SHACL) and the EDC (*Extract, Define, Canonicalize*, arXiv:2404.03868) framework with zero heavy external RDF library bloat (`rdflib`, `pySHACL`, and `spaCy` are retired):

- **Concepts & Contextual Definitions (`skos:Concept`, `skos:definition`):**
  Graph nodes are persisted in `entities(entity_id, name, definition)`. Following EDC, every extracted entity must have a contextual `skos:definition`. This provides lexical disambiguation for `EmbeddingGemma` (`title: {name} | text: {definition}`), separating homographs (e.g. *"ball"* as a dance vs. *"ball"* as a spherical toy) into distinct semantic nodes before candidate generation.
- **Multi-Label Synonymy & SKOS S13 Constraint (`entity_labels`):**
  Synonyms are mapped via `entity_labels(label_norm, entity_id, label_kind, raw_label, chunk_id)` supporting `skos:prefLabel`, `skos:altLabel`, and `skos:hiddenLabel`. YAW enforces W3C SKOS integrity constraint **S13** (pairwise disjointness among `prefLabel`, `altLabel`, and `hiddenLabel` for any given concept).
- **Reversible Merges & Lineage Tracking (`entity_alt_labels`, `entity_redirects`, `entity_merge_lineage`):**
  When synonym candidates are verified by LLM adjudication (`judge_synonyms`), `merge_entities` aliases duplicate concepts into `entity_alt_labels` while preserving complete merge lineage and redirects. This ensures that incremental document removals (`delete_chunks`) via CocoIndex `SyncDelta` remain fully reversible without corrupting the graph.
- **Typed Semantic Edges & Contrastive Semantics (`edges`, `edge_sources`):**
  Edges model semantic relations explicitly:
  - `passage`: Bipartite passage-to-entity links.
  - `fact`: Directed assertion triples (`subj` $\to$ `obj`) enriched with contextual triple `skos:definition`. Predicates remain typed edge properties rather than graph concept hubs, preventing PPR super-node collapse.
  - `skos:related`: Symmetric associative links (weighted by $\beta$) traversed by sparse CSR Personalized PageRank.
  - `owl:differentFrom`: Symmetric contrastive non-identity links between distinct polysemous/homograph senses. These edges are strictly dominant over merging and are excluded from PPR random walks.
- **Two-Channel Candidate Combiner & Provisional Sense-Split Gate:**
  `find_entity_disambiguation_candidates` surfaces candidate pairs using dense vector kNN, lexical `difflib` similarity, and unadjudicated surface matches. When two mentions share identical surface labels but have divergent definitions ($\cos < 0.85$), YAW allocates a provisional `sense_id` instead of colliding or prematurely asserting `owl:differentFrom`.
- **In-Transaction Closed-World SHACL Validation (`lint_graph_shacl`):**
  Enforces 6 deterministic W3C SHACL shapes natively in SQLite SQL CTEs in $<1\text{ ms}$:
  1. `ConceptShape/prefLabel`: Exactly one `skos:prefLabel` per concept.
  2. `ConceptShape/S13-DisjointLabels`: Pairwise disjointness across `prefLabel`, `altLabel`, and `hiddenLabel`.
  3. `ConceptShape/definition`: Mandatory non-empty definition per concept.
  4. `ConceptShape/ContrastiveDisjointness`: Mutual exclusion between `owl:differentFrom` and synonym identity.
  5. `FactStatementShape/endpoints`: Subject and object must reference existing valid concepts.
  6. `ProvenanceShape/edgeSources`: Structural edges must link to at least one valid source chunk.
- **Autonomous DSPy RLM Self-Healing (`heal_shacl_violations`):**
  When SHACL shape violations occur during indexing transactions, YAW triggers a self-healing loop powered by `shacl_healer_rlm` with interactive REPL inspection tools (`hydrate_chunk`, `inspect_focus_node`, `qwen_propose_shacl_repairs`, `apply_shacl_repairs`) to heal the graph prior to transaction commit.
- **Zero-Dependency W3C Turtle 1.1 Export (`serialize_graph_turtle`):**
  `yaw graph lint --turtle` streams the entire closed-world graph directly to standard Turtle syntax with standard prefix namespaces (`rdf:`, `rdfs:`, `skos:`, `owl:`, `sh:`, `yaw:`).

## Theoretical & Empirical Foundations (Scientific References)

YAW is engineered directly from principles and algorithms established in the following peer-reviewed and preprint literature:

### Information Retrieval & Language Model Architecture (arXiv & Peer-Reviewed)

1. **HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models**
   - *Bernal, et al. (2024)* — [arXiv:2405.14831](https://arxiv.org/abs/2405.14831) / [Local Copy](docs/arxiv-2405.14831v3-hipporag.md)
   - *Key Foundations:* Hippocampal memory indexing theory, OpenIE knowledge graph construction, and Personalized PageRank (PPR) associative activation over passage-entity-fact graphs.
2. **HippoRAG 2: From RAG to Memory via Personalized PageRank**
   - *Bernal, et al. (2025)* — [arXiv:2502.14802](https://arxiv.org/abs/2502.14802) / [Local Copy](docs/arxiv-2502.14802v1-hipporag2.md)
   - *Key Foundations:* Continuous online graph updating, Recognition Memory fact reranking with bidirectional phrase linking, and sparse CSR matrix PPR power iteration without external vector database dependencies.
3. **EDC: Extract, Define, Canonicalize: An LLM-based Framework for Knowledge Graph Construction**
   - *arXiv:2404.03868 (2024)* — [arXiv:2404.03868](https://arxiv.org/abs/2404.03868)
   - *Key Foundations:* Augmenting triplet extraction with contextual entity and triple definitions (`skos:definition`), resolving homographs and polysemous senses prior to entity deduplication, disambiguation, and canonicalization.
4. **Deconvolving Passage Retrieval with MaxP**
   - *Dai & Callan (SIGIR 2019)* — [arXiv:1905.01988](https://arxiv.org/abs/1905.01988)
   - *Key Foundations:* Max-Passage (MaxP) aggregation ($S(\text{passage}) = \max_{c \in \text{passage}} S(c)$), scoring multi-chunk passages and long documents by their maximal scoring sub-chunk without allowing redundant chunks to displace distinct documents from top-$K$ recall evaluation.
5. **Dense Passage Retrieval for Open-Domain Question Answering (DPR)**
   - *Karpukhin, et al. (EMNLP 2020)* — [arXiv:2004.04906](https://arxiv.org/abs/2004.04906)
   - *Key Foundations:* Dense dual-encoder representation and retrieval baseline against which associative graph retrieval is calibrated and benchmarked.
6. **DSPy: Compiling Declarative Language Model Calls into State-of-the-Art Pipelines**
   - *Khattab, et al. (2023)* — [arXiv:2310.03714](https://arxiv.org/abs/2310.03714)
   - *Key Foundations:* Declarative Signatures, Modules, and programmatic prompt compilation against objective metric evaluators with strict separation of concerns.
7. **MIPROv2 & GEPA: Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs**
   - *Opsahl-Ong, et al. (2024)* — [arXiv:2406.11695](https://arxiv.org/abs/2406.11695)
   - *Key Foundations:* Multi-stage instruction proposal and few-shot demonstration search with reflective, evolutionary Pareto optimization ($\text{GM}_{\text{good}} / \text{GM}_{\text{bad}}$).
8. **Recursive Language Modeling for Long Context Reasoning (RLM)**
   - *Zhang, et al. (2025)* — [arXiv:2502.06733](https://arxiv.org/abs/2502.06733)
   - *Key Foundations:* Decomposing complex multi-step reasoning, self-healing graph validation, and synonym adjudication into programmatic REPL tool loops with sandboxed Python evaluation.

### Knowledge Representation & Semantic Web Standards (W3C Recommendations)

9. **W3C SKOS Simple Knowledge Organization System Reference**
   - *Miles & Bechhofer (W3C Recommendation, 2009)* — [W3C SKOS Reference](https://www.w3.org/TR/skos-reference/)
   - *Key Foundations:* Modeling knowledge graph nodes as `skos:Concept` instances; multi-label indexing with `skos:prefLabel`, `skos:altLabel`, and `skos:hiddenLabel`; enforcing SKOS integrity constraint **S13** label disjointness; associative semantic edge linking via symmetric `skos:related`; and lexical documentation via `skos:definition`.
10. **W3C SHACL Shapes Constraint Language**
    - *Knublauch & Kontokostas (W3C Recommendation, 2017)* — [W3C SHACL](https://www.w3.org/TR/shacl/)
    - *Key Foundations:* Declarative closed-world structural validation. Implemented natively in pure SQLite SQL without external RDF libraries (`lint_graph_shacl`), enforcing 6 deterministic closed-world shapes (`ConceptShape/prefLabel`, `ConceptShape/S13-DisjointLabels`, `ConceptShape/definition`, `ConceptShape/ContrastiveDisjointness`, `FactStatementShape/endpoints`, `ProvenanceShape/edgeSources`).
11. **W3C RDF 1.1 Turtle: Terse RDF Triple Language**
    - *Prud'hommeaux & Carothers (W3C Recommendation, 2014)* — [W3C Turtle](https://www.w3.org/TR/turtle/)
    - *Key Foundations:* Standardized text serialization of RDF graphs. Implemented natively in YAW (`serialize_graph_turtle`, `yaw graph lint --turtle`) to export the closed-world knowledge graph directly to standard Turtle syntax with zero external dependency bloat.
12. **W3C OWL 2 Web Ontology Language Document Overview**
    - *W3C OWL Working Group (W3C Recommendation, 2012)* — [W3C OWL 2 Overview](https://www.w3.org/TR/owl2-overview/)
    - *Key Foundations:* Expressing contrastive non-identity between distinct polysemous/homograph senses using symmetric `owl:differentFrom` assertions, dominant over transitive merging and excluded from CSR PPR associative retrieval walks.

Upstream baseline: [pinned HippoRAG source](https://github.com/OSU-NLP-Group/HippoRAG/tree/d5c8329422e0a0b834a15874545cb6a74b4f9b26).
