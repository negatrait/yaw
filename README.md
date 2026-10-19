# YAW: HippoRAG 2 evaluation

YAW evaluates upstream HippoRAG 2 on multi-hop and messy corpora. Its next steps are a local daemon and a FastMCP v4 interface. Read [FOUNDATIONAL](FOUNDATIONAL.md), then the planned [daemon epic](docs/epic-1-hipporag-daemon.md) and [Streamable HTTP epic](docs/epic-2-fastmcp-http.md). Reference papers are preserved locally under [HippoRAG 1 (`2405.14831v3`)](docs/arxiv-2405.14831v3-hipporag.md) and [HippoRAG 2 (`2502.14802v1`)](docs/arxiv-2502.14802v1-hipporag2.md).

## Available now

Use Python 3.12 and uv:

```powershell
uv sync --locked
uv run --locked pytest -v
uv run --locked yaw daemon status
uv run --locked yaw daemon run
uv run --locked yaw daemon start
uv run --locked yaw daemon stop
uv run --locked yaw graph build
uv run --locked yaw graph rebuild
uv run --locked yaw graph lint
uv run --locked yaw graph lint --turtle
uv run --locked yaw optimize --mode gepa --auto light
uv run --locked yaw load-program .yaw/compiled_program_optimize_<datestamp>.json
uv run --locked yaw config get optimization.auto
uv run --locked yaw config set optimization.auto light
uv run --locked python benchmark.py --dataset sample --mode all
uv run --locked python benchmark.py --dataset 2wikimultihopqa --max-queries 10
uv run --locked python benchmark.py --dataset hotpotqa --max-queries 10 --mode retrieval
uv run --locked python benchmark.py --dataset musique --max-queries 10 --mode retrieval
uv run --locked python benchmark.py --dataset sample --chat-model gemma4-it-12b-FLM --embedding-model Qwen3-Embedding-0.6B-GGUF
```

`benchmark.py` replaces the retired `live_smoke.py` fixture script and evaluates upstream HippoRAG 2 (`retrieve` + optional `qa`) head-to-head against StandardRAG (`retrieve_dpr` + optional `qa`) on the upstream reproduction datasets (`sample`, `2wikimultihopqa`, `musique`, `hotpotqa`) against Lemonade at `http://localhost:13305/api/v1`. Pass `--mode retrieval` to evaluate retrieval (`Recall@1/@2/@5`) without invoking the QA reader model. Dataset files are fetched once from pinned commit `d5c8329422e0a0b834a15874545cb6a74b4f9b26` and cached under `outputs/datasets/` (skipping `{dataset}_corpus.json` when `--max-queries` is used without `--full-corpus`). Corpus passages are checkpointed through OpenIE in bounded micro-batches (`--index-batch-size 8` by default via `pre_openie`) and indexed once into deterministic persisted directories under `outputs/benchmarks/indexes/` (keyed by dataset slice hash, resolved embedding identity, and extraction chat policy identity; use `--fresh-index` for a disposable per-run index), and queries/QA pairs are processed 1-by-1 (`--qa-batch-size 1` by default) with in-memory retrieval matrices released after retrieval. Each run writes `run.log` and `report.json` under `outputs/benchmarks/run-<dataset>-*/`, comparing HippoRAG 2 and Standard DPR retrieval Recall@1/@2/@5 (and QA ExactMatch/F1 when QA is enabled) alongside the HippoRAG 2 paper's `Llama-3.3-70B-Instruct` + `NV-Embed-v2` (8B) reference numbers. Source notes in `sample-llm-wiki/raw/` are not ingested by this command.

The default 3-slot development split uses:
- **Embeddings (`--embedding-model`):** `embeddinggemma-300m-GGUF` on GPU CUDA `llamacpp` (`--embedding-task-mode` defaults to `distinct-titled` under `retrieval-prompts-v3` and supports `search`, `qa`, `fact`, `distinct`, `titled`, `distinct-titled`; `--embedding-model` also accepts `embeddinggemma-300m-qat-q8_0-GGUF-Q8_0` and `Qwen3-Embedding-0.6B-GGUF`).
- **OpenIE & Recognition Memory (`--extraction-model`):** `Qwen3-4B-Instruct-2507-GGUF` on GPU CUDA `llamacpp` (`--parallel 1 --cache-ram 0`) with tightened OpenIE NER and triple JSON Schemas (`minLength: 1`) and dynamic candidate-literal `DSPyFilter` GBNF grammar enforcement (`gbnf-v3:require-stop`).
- **QA Reading (`--qa-model`):** `Mistral-7B-Instruct-v0.3-Hybrid` on Ryzen AI Hybrid NPU/iGPU (`ryzenai-llm`) with end-of-turn `QA_TURN_REMINDER` and `Answer:` delimiter normalization (`fold-system:prompt-v4:require-stop`).
- Passing `--chat-model` overrides both `--extraction-model` and `--qa-model` with a single model (such as step-up `gemma4-it-12b-FLM` or `qwen3.5-9b-FLM`).

There is one embedding adapter (`yaw/embedding.py`), one chat policy subclass (`yaw/chat.py`), and no fallback. See [model contracts and real benchmark protocol](docs/model-evaluation.md). No upstream classes are monkeypatched.

On this machine the models and settings are already installed. For a fresh Lemonade installation, register/download the embedding checkpoint and save both CUDA `llamacpp` model settings explicitly before running benchmarks (`--batch-size 2048 --ubatch-size 2048` on EmbeddingGemma prevents `llama-server`'s default 512-token physical microbatch cutoff on 512–2048 token passages, and `--cache-ram 0` without `-dio` on `Qwen3-4B-Instruct-2507-GGUF` prevents Windows Direct-IO and host-RAM prompt-cache growth):

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

This is the verified local CUDA configuration; the script does not install models or choose another backend on failure. Keep the selected extraction and QA chat models downloaded in Lemonade too. A different serving configuration requires a new repeatability/quality validation and index identity. The reported 8192 server allocation does not enlarge EmbeddingGemma's 2048-token model context.

HippoRAG is pinned to upstream commit `d5c8329422e0a0b834a15874545cb6a74b4f9b26`, declaring `2.0.0a5`. This is a source installation requiring Git; a published a5 wheel was unavailable when checked. NumPy is explicitly declared for the experiment. Python 3.14.7 resolution failed because upstream pins PyTorch 2.5.1 without a CPython 3.14 wheel, so the verified interpreter is CPython 3.12.14.

Existing source notes remain in `sample-llm-wiki/raw/`; they are not automatically indexed by the current script. Milvus and MCP packages are not installed. HippoRAG's default Parquet stores already support the executed pipeline.

## Measured versus planned

On the 2026-10-14 Tier 0 and Tier 1 upstream benchmark runs with the default split (`embeddinggemma-300m-GGUF` `distinct-titled` `retrieval-prompts-v3` + `Qwen3-4B-Instruct-2507-GGUF` `gbnf-v3:require-stop` + `Mistral-7B-Instruct-v0.3-Hybrid` `fold-system:prompt-v4:require-stop`, `temperature = 0.2`):
- **Tier 0 (`sample`, 1 query, 3 passages, `run-sample-hgj5zm7g`):** HippoRAG 2 Recall@1/@2/@5 = **1.0 / 1.0 / 1.0** (30 phrase nodes, 29 triples).
- **Tier 1 (`2wikimultihopqa`, 10 queries, 96 passages, `run-2wikimultihopqa-fbmnosfw`):** HippoRAG 2 Recall@1/@2/@5 = **0.425 / 0.800 / 1.000** vs. Standard DPR **0.425 / 0.725 / 0.850** (**`+7.5%` Recall@2**, **`+15.0%` Recall@5**; 866 phrase nodes, 905 triples, 8/10 queries with graph seeds); HippoRAG 2 QA EM/F1 = **0.300 / 0.5278** vs. Standard DPR **0.400 / 0.4500** (**`+7.78%` F1**).
- **Tier 1 (`hotpotqa`, 10 queries, 94 passages, `run-hotpotqa-zbp9k8e0`):** HippoRAG 2 Recall@1/@2/@5 = **0.500 / 0.850 / 1.000** vs. Standard DPR **0.450 / 0.700 / 0.950** (**`+5.0%` Recall@1**, **`+15.0%` Recall@2**, **`+5.0%` Recall@5**; 1,206 phrase nodes, 1,286 triples, 10/10 queries with graph seeds); HippoRAG 2 QA EM/F1 = **0.300 / 0.5483** vs. Standard DPR **0.300 / 0.4524** (**`+9.59%` F1**).
- **Tier 1 (`musique`, 10 queries, 197 passages, `run-musique-_5f_uroh`):** HippoRAG 2 Recall@1/@2/@5 = **0.1833 / 0.3167 / 0.5417** vs. Standard DPR **0.2083 / 0.2833 / 0.5167** (**`+3.34%` Recall@2**, **`+2.5%` Recall@5**; 2,268 phrase nodes, 2,346 triples, 7/10 queries with graph seeds); HippoRAG 2 QA EM/F1 = **0.200 / 0.3000** vs. Standard DPR **0.300 / 0.4000**.

174 offline and integration regression tests pass (`uv run --locked pytest -v`). See [evaluation](docs/evaluation.md) for benchmark runs, closed-world SHACL validations, and live-silicon MIPROv2/GEPA optimization results.

The native HippoRAG 2 engine (`yaw/daemon/graph/engine.py`), persistent single-owner daemon runtime (`yaw/daemon/runtime.py`), single `yaw.db` SQLite storage (`sqlite-vec` + FTS5 + sparse CSR PPR), native CocoIndex single-parse Rust Tree-sitter AST ingestion across 30+ formats with declarative $\pi_{\mathcal{W}, \mathcal{B}}$ key projection, title-aware `EmbeddingGemma` formatting, Dai & Callan MaxP passage aggregation, and DSPy MIPROv2 & GEPA prompt evolution with hot-reload program promotion are implemented and verified. Epic 2 (FastMCP over Streamable HTTP) follows as the next planned phase.

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
