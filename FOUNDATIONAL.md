# FOUNDATIONAL

Baseline: 2026-10-14. YAW is a native HippoRAG 2 graph & storage engine (`yaw/daemon/graph/{engine.py,ppr.py,storage/*}`) backed by one SQLite file (`yaw.db`), CocoIndex (`cocoindex>=1.0.0`) ingestion, sparse PPR with 1-indexed line hydration, and FastMCP over Streamable HTTP. Pinned upstream `hipporag` (`2.0.0a5`) is kept only in [`benchmark.py`](benchmark.py) as the reference baseline and evaluator. No `rdflib`, OWL, `pySHACL`, `spaCy`, Parquet, or Pickle in runtime.

## 6 Core Rules (Max Perf, Min Bloat)

1. **`yaw.db` SSOT + 1-Indexed Line Hydration + W3C SKOS/RDF/OWL/SHACL Closed-World Schema:**
   - Zero file sprawl; zero chunk text stored in `yaw.db`.
   - `chunks(chunk_id, rel_posix_path, start_line, end_line, title)` stores 1-indexed line spans.
   - Closed-World Assumption (CWA) graph lives in `entities(entity_id, name, definition)`, `entity_labels(label_norm, entity_id, label_kind, raw_label, chunk_id)` (`skos:prefLabel`, `skos:altLabel`, `skos:hiddenLabel` with SKOS `S13` disjointness), `facts(fact_id, subj_id, pred, obj_id, raw_triple_json, definition)`, and `edges(src_id, dst_id, kind, weight)` + `edge_sources(src_id, dst_id, kind, chunk_id)` (`kind IN ('passage', 'fact', 'skos:related', 'owl:differentFrom')`, with `owl:differentFrom` dominance and exclusion from CSR PPR walks).
   - Closed-world SQL SHACL validation (`lint_graph_shacl`) enforces 6 deterministic shapes (`ConceptShape/prefLabel`, `ConceptShape/S13-DisjointLabels`, `ConceptShape/definition`, `ConceptShape/ContrastiveDisjointness`, `FactStatementShape/endpoints`, `ProvenanceShape/edgeSources`) and serializes W3C Turtle (`serialize_graph_turtle`, `yaw graph lint [--turtle]`) with zero external RDF libraries.
   - Run sparse PPR (`scipy.sparse.csr_matrix` power iteration or Forward Push) in `<3 ms`. Hydrate disk lines **only** for the final top-$k$ chunks via `itertools.islice`.
2. **`sqlite-vec` + Contentless `FTS5` + Multi-Label SKOS Seeding:**
   - Vectors live in `chunk_vec`, `entity_vec`, `fact_vec` (`sqlite-vec` / `float32` BLOBs). Concept and fact embeddings use `TITLE_LINE_RE` (`# {name}\n{definition}`) so `EmbeddingGemma` separates `title:` from `text:`. Use bounded SQL top-$k$ queries, never duplicate full-corpus NumPy matrices in RAM.
   - Lexical BM25 seeding uses contentless `FTS5` (`content=''`, L1-normalized) + multi-label `entity_labels` resolution (`resolve_entity_labels`, disambiguating polysemous/homograph senses by query-vector cosine).
3. **Zero Unproven Bloat:**
   - Every added piece of graph/ranking logic must prove measurable `Recall@k` or `F1` lift on [`benchmark.py`](benchmark.py).
4. **CocoIndex Target Sink Compilation Ownership + `ManagedConnection` + Read-Only External Access:**
   - `cocoindex>=1.0.0` (`CorpusIngestor`) watches `corpus.root`, chunks files on line boundaries, and invokes the bound `on_delta` target compiler inside `_ChunkTargetHandler._apply_actions` before CocoIndex commits state to `coco_lmdb`.
   - `YawGraphEngine` opens `yaw.db` via `cocoindex.connectors.sqlite.ManagedConnection` (`open_managed_yaw_db`, FIFO RWLock + `sqlite-vec`) and executes Stage 1 `ExtractEntities` (`[name, definition]`) + Stage 2 `ExtractTriples` indexing (preserving all standalone Stage 1 NER entities with non-empty `skos:definition`), the Provisional Same-Label Sense-Split Gate ($\cos \ge 0.85 \implies$ reuse concept; $\cos < 0.85 \implies$ provisional `sense_id` without premature `owl:differentFrom`), Two-Channel Candidate Combiner (`find_entity_disambiguation_candidates` surfacing kNN + `difflib` + unadjudicated same-`label_norm` pairs $\to$ `judge_synonyms` $\to$ `merge_entities` / `upsert_semantic_edges`), and in-transaction SHACL linting + RLM self-healing (`lint_graph_shacl(conn, auto_heal=False)` $\to$ `heal_shacl_violations`) inside `with self.managed_conn.transaction() as conn:`.
   - Enforce single-owner write lock via `filelock.FileLock("yaw.db.lock")`. External CLI/inspection processes (`yaw daemon status`, `yaw graph lint`) read `yaw.db` strictly via read-only `file:...?mode=ro` SQLite WAL snapshots.
   - Treat inference (`http://localhost:13305/api/v1` or configured URL) as an external persistent server; never load/unload models from the daemon.
5. **FastMCP Adapter:**
   - Mount `fastmcp` Streamable HTTP (`127.0.0.1:13306/mcp` + `/healthz` + `/readyz`) onto the single daemon runtime.
6. **Native DSPy Inference, 19-Predictor Separation of Concerns & Gate-Free $\text{GM}_{\text{good}} / \text{GM}_{\text{bad}}$ Pareto Evolution:**
   - Drive all inference through `dspy` (`yaw/daemon/graph/inference/`): `dspy.LM` (`chat.py`) and `dspy.Embedder` (`embedding.py`) against Lemonade (`openai/...`) or Ollama (`ollama_chat/...`, `ollama/...`) with zero `hipporag` runtime imports.
   - **Strict 19-Predictor Separation of Concerns:** Define **19 pairwise-distinct `dspy.Signature` classes** (8 Worker Signatures: `ExtractEntities`, polymorphic `ExtractTriples`, `DefineMissingEntities`, `JudgeSynonyms`, `ProposeSHACLRepairs`, `ExtractQueryIntent`, `FilterFacts`, `AnswerQuestion`; 7 RLM Orchestrator Signatures: `OrchestrateEntityExtraction`, `OrchestrateTripleExtraction`, `OrchestrateMissingEntityDefinitions`, `OrchestrateSynonymJudgment`, `HealSHACLViolations`, `OrchestrateFactFiltering`, `OrchestrateQuestionAnswering`; and 4 GEPA Reflection Signatures in `RLMInstructionProposer`: `ReflectiveOuterOrchestratorSignature`, `ReflectiveInnerDistillerSignature`, `ProposeInstructionSignature`, `CompressInstructionSignature`). Every predictor has its own dedicated `_PredictorBoundLM` (`.lm`) clone and records per-invocation `track_usage()` telemetry (`self.telemetry`, `self.entity_def_audit_trail`, `self.synonym_audit_trail`, `self.shacl_audit_trail`), with all active `HippoRAGProgram` predictors exposed in `named_predictors()`.
   - **Heterogeneous Mistral Orchestrator + Qwen GBNF Worker (`execution_mode="rlm"`):** GBNF (`extra_body={"grammar": ...}`) is supported **only** on `llama.cpp` (`Qwen3-4B-Instruct-2507-GGUF` on Vulkan GPU) and **never** on Ryzen AI NPU/OGA (`Mistral-7B-Instruct-v0.3-Hybrid`, enforced by `lm_supports_gbnf`). In `execution_mode="rlm"`, every orchestrator (`ner_rlm`, `triples_rlm`, `entity_definer_rlm`, `synonym_judge_rlm`, `shacl_healer_rlm`, `fact_filter_rlm`, `qa_rlm`) runs `Mistral-7B` on Ryzen AI NPU inside `dspy.LocalInterpreter` (zero GBNF), delegating inner `qwen_*` worker tools and `hydrate_chunk` / `inspect_focus_node` graph tools to `Qwen3-4B` on `llama.cpp` GPU with pure-JSON GBNF grammars (`build_dspy_ner_grammar`, `build_dspy_triples_grammar`, `build_dspy_missing_entity_defs_grammar`, `build_dspy_synonym_judge_grammar`, `build_dspy_shacl_repair_grammar`, `build_dspy_rerank_grammar`, `build_dspy_qa_cot_grammar`).
   - **Gate-Free $\text{GM}_{\text{good}} / \text{GM}_{\text{bad}}$ Sizecoding Pareto Evolution:** Never evolve prompts on full Tier 2 benchmark corpora. Evolve one universal `compiled_program.json` via `dspy.MIPROv2` / `dspy.GEPA` using `compute_sizecoding_pareto_score` ($\text{score} = \text{GM}_{\text{good}} / \text{GM}_{\text{bad}}$, combining floor-smoothed `Recall@1..20`, optional `EM`/`F1`, and `shacl_conforms` in $\text{GM}_{\text{good}}$ against normalized token, initial/remaining SHACL violation, missing entity definition count/tokens, and canonicalization/repair token factors in $\text{GM}_{\text{bad}}$ with zero hard-gate cliffs) strictly on **Tier 1 small slices** and/or **external OOD RAG datasets**, then evaluate zero-shot on unseen Tier 2 holdouts.

## Domain Containment Tree & Architectural Law (Zero Cross-Domain / Sibling Edges)

Every package owns its children; **horizontal imports between sibling domains are forbidden (`0` cross-domain edges)**. A module in package `X` may only import its own descendants `X.*` or `yaw.core.*`:

```text
yaw/
  core/                    types.py, config.py, logger.py          (imports: stdlib only)
  daemon/                  runtime.py, cli.py, mcp_server.py       (imports: daemon.*, core)
    ingestion/             corpus.py                               (imports: core, cocoindex)
    graph/                 engine.py, ppr.py                       (imports: graph.*, core, scipy)
      storage/             db.py, hydration.py                     (imports: core, sqlite3, sqlite_vec)
      inference/           models.py, prompts.py, embedding.py, chat.py (imports: core, dspy)
  __main__.py                                                      (imports: daemon.cli)
```

### Architectural Containment Law (`tests/core/test_architecture.py`):
1. **Empty Intermediate Packages:** Intermediate parent package initializers (`yaw/`, `yaw/daemon/`, `yaw/daemon/graph/`) contain zero executable statements and trigger zero parent-package import side-effects.
2. **Constricted Leaf Facades:** Leaf domain packages (`yaw.core`, `yaw.daemon.graph.inference`, `yaw.daemon.graph.storage`, `yaw.daemon.ingestion`) provide constricted public API facades (`_PUBLIC_API` + lazy `__getattr__` + `__dir__`) allowing leaf mutation into single-responsibility modules while keeping a stable public API contract.
3. **Strict Domain Containment:** A module in package `P` imports only `P.*` and `yaw.core.*`. Horizontal sibling imports (`storage` $\leftrightarrow$ `inference`, `ingestion` $\leftrightarrow$ `graph`) are strictly forbidden. Only top-level orchestrators (`runtime.py`, `mcp_server.py`, `cli.py`, `benchmark.py`) wire sibling leaf domains together.
4. **Zero Protected (`_`) or Private (`__`) Cross-Module Imports or Facade Exports:** Any symbol starting with a leading underscore is strictly module-internal. No `name.startswith("_")` may ever appear in a leaf `_PUBLIC_API` facade, in cross-module imports (`from .mod import _helper`), or in cross-module attribute accesses.
5. **Zero `hipporag` Runtime Imports Inside `yaw/`:** Pinned `hipporag` is imported only in `benchmark.py` and offline test suites as the reference baseline.

## Validated 3-Slot Local Model Stack (`http://localhost:13305/api/v1`)

- **Embeddings (`dspy.Embedder` on Vulkan GPU `llamacpp`):** `embeddinggemma-300m-GGUF` (`Q8_0`, mean pooling, `--batch-size 2048 --ubatch-size 2048`, `encoding_format="float"`, `distinct-titled` / `retrieval-prompts-v3`).
- **GBNF RLM Sub-LM Worker & Direct Extraction/Rerank (`dspy.LM` on Vulkan GPU `llamacpp`):** `Qwen3-4B-Instruct-2507-GGUF` (`Q4_K_M`, `8192–16384` ctx, `--parallel 1 --cache-ram 0`, no `-dio`, `temp=0.2`, `freq_penalty=0.5`, `_yaw_supports_gbnf=True`).
- **RLM Orchestrator & QA Reader (`dspy.LM` on Ryzen AI Hybrid NPU/OGA `ryzenai-llm`):** `Mistral-7B-Instruct-v0.3-Hybrid` (`32786` ctx, `temp=0.2`, `_yaw_supports_gbnf=False`, system-prompt folded into first user turn for RyzenAI Jinja).

## Execution & Hard Gates

1. **[Epic 1 (`E1-S1..E1-S6`)](docs/epic-1-hipporag-daemon.md):** Build strictly bottom-up along the containment tree: `E1-S1` (`core` + AST import lock) $\to$ `E1-S2` (`inference` native `dspy.LM`, `dspy.Embedder`, `dspy.Signature`s, `dspy.RLM` & `dspy.GEPA` loader) $\to$ `E1-S3` (`storage` `yaw.db` + line hydration) $\to$ `E1-S4` (`graph` sparse PPR + engine, **Tier 1 parity gate**) $\to$ `E1-S5` (`ingestion` CocoIndex delta pipeline) $\to$ `E1-S6` (`daemon` runtime/CLI + `dspy.GEPA` / `dspy.RLM` optimization + **Tier 2 full-corpus parity or beat gate**).
2. **[Epic 2 (`E2-S1..E2-S2`)](docs/epic-2-fastmcp-http.md):** Starts after `E1-S6`; mounts `yaw/daemon/mcp_server.py` onto `yaw/daemon/runtime.py` (`0` direct `graph`/`storage`/`inference` imports).
3. Logs use 500-line FIFO rotation (`yaw/core/logger.py`). Record all measured runs in [`docs/evaluation.md`](docs/evaluation.md).
