# FOUNDATIONAL

Accepted development-tool update (2026-10-23): Python is pinned to 3.14.9.
`uv run preflight` applies Ruff formatting/safe fixes and ty fixes, then runs
pytest with testmon and coverage.py through pytest-cov. SlipCover is removed.
Ruff fixes and pytest testmon are project defaults. `uv run typecheck` supplies
ty's CLI-only `--fix` default. Every collected test runs by default, including
resident-model tests; unavailable live prerequisites fail instead of skipping.
Reports distinguish incremental test coverage from complete-suite coverage.

Accepted GEPA reflection repair (2026-10-23): RLM selects original training evidence
by stable reference; DSPy's native InstructionProposer is the sole instruction
writer/compressor. SIMBA remains enabled with task-improvement rewards. Full
rendered reflection requests use the serving tokenizer, a capacity-derived input
budget (32,768 tokens at 128K), configured output allowance and 512-token context margin. Completed
stages and version-2 worker state persist in yaw.db; the active request is pinned
outside candidate-cache eviction. Legacy worker states require a compatible new
run. Reflection never consumes held-out answers. See docs/gepa-reflection-repair.md.

Accepted capacity/profile update (2026-10-24): use the exact embedding alias
`embeddinggemma-300m-qat-q8_0-GGUF-Q8_0` by default. The observed Gemma server
serving window is 131,072 tokens; `yaw.toml` and the Gemma profile declare that
default explicitly. The user reports approximately 65 TPS without flash attention;
this is deployment evidence, not a newly measured YAW benchmark. Output tokens,
reflection input, RLM output characters and prompt-size safeguards scale with
context unless explicitly pinned. At 128K these default to 8,192 output tokens,
32,768 reflection input tokens and 20,000 RLM output characters. Independent
RLM turn/call counts, diagnostic retention and embedding profiles do not scale.
Context/output capacities and RLM execution budgets are deployment
settings, and changing them or embedding identity requires a compatible new
optimization/index run. Existing running processes keep their original settings.

Accepted runtime update (2026-10-22): the configured/default chat, extraction,
QA and reflection model is now `Gemma-4-E4B-it-GGUF`, checkpoint
`unsloth/gemma-4-E4B-it-GGUF:Q4_K_M`, on the existing external llama.cpp GPU
service (`--parallel 1 --cache-ram 0`; serving capacity superseded above). This supersedes the Qwen
runtime-model selection below; historical Qwen benchmark targets remain valid
as historical comparisons. Existing embedding selection is unchanged. Gemma's
direct/worker roles retain GBNF; RLM code generation remains unconstrained.

Baseline: 2026-10-19. YAW is a native HippoRAG 2 graph & storage engine (`yaw/daemon/graph/{engine.py,ppr.py,storage/*}`) backed by one SQLite file (`yaw.db`), CocoIndex (`cocoindex>=1.0.0`) ingestion, sparse PPR with 1-indexed line hydration, and FastMCP over Streamable HTTP on Python `>=3.14.7`. Upstream `hipporag` and `optuna` are permanently retired from `pyproject.toml`, `uv.lock`, and runtime/test code; [`benchmark.py`](benchmark.py) and [`scripts/verify_paper_parity_gate.py`](scripts/verify_paper_parity_gate.py) evaluate native `yaw` against two consolidated reference targets: **Target 1 (`MONKEYPATCHED_HIPPORAG_BASELINES`)** (historical upstream HippoRAG 2 runs on `Qwen3-4B` + `EmbeddingGemma`, archived in [`outputs/benchmarks/consolidated_historical_baselines.json`](outputs/benchmarks/consolidated_historical_baselines.json)) and **Target 2 (`PAPER_BASELINES`)** (published HippoRAG 2 paper results on `Llama-3.3-70B` + `NV-Embed-v2`). No `hipporag`, `rdflib`, OWL, `pySHACL`, `spaCy`, Parquet, or Pickle in runtime.

## 6 Core Rules (Max Perf, Min Bloat)

1. **`yaw.db` SSOT + 1-Indexed Line Hydration + W3C SKOS/RDF/OWL/SHACL Closed-World Schema + MaxP Aggregation:**
   - Zero file sprawl; zero chunk text stored in `yaw.db`.
   - `chunks(chunk_id, rel_posix_path, start_line, end_line, title, ast_island_start, ast_island_end)` stores 1-indexed line spans pointing directly to workspace sources with enclosing AST island boundaries.
   - Closed-World Assumption (CWA) graph lives in `entities(entity_id, name, definition)`, `entity_labels(label_norm, entity_id, label_kind, raw_label, chunk_id)` (`skos:prefLabel`, `skos:altLabel`, `skos:hiddenLabel` with SKOS `S13` disjointness), `facts(fact_id, subj_id, pred, obj_id, raw_triple_json, definition)`, and `edges(src_id, dst_id, kind, weight)` + `edge_sources(src_id, dst_id, kind, chunk_id)` (`kind IN ('passage', 'fact', 'skos:related', 'owl:differentFrom')`, with `owl:differentFrom` dominance and exclusion from CSR PPR walks).
   - Closed-world SQL SHACL validation (`lint_graph_shacl`) enforces 6 deterministic shapes (`ConceptShape/prefLabel`, `ConceptShape/S13-DisjointLabels`, `ConceptShape/definition`, `ConceptShape/ContrastiveDisjointness`, `FactStatementShape/endpoints`, `ProvenanceShape/edgeSources`) and serializes W3C Turtle (`serialize_graph_turtle`, `yaw graph lint [--turtle]`) with zero external RDF libraries.
   - Run sparse PPR (`scipy.sparse.csr_matrix` power iteration or Forward Push) in `<3 ms`. Over-fetch candidates (`pool_k = max(top_k * 4, 50)`) and apply Dai & Callan (SIGIR 2019) MaxP aggregation ($S(\text{passage}_i) = \max_{c_{i,j} \in \text{passage}_i} S(c_{i,j})$) grouped by canonical passage span `(rel_posix_path, start_line, end_line)` before hydrating disk lines for the top-$k$ distinct passages via `itertools.islice`.
2. **`sqlite-vec` + Contentless `FTS5` + Multi-Label SKOS Seeding + Title-Aware Embeddings:**
   - Vectors live in `chunk_vec`, `entity_vec`, `fact_vec` (`sqlite-vec` / `float32` BLOBs). Concept and fact embeddings use `TITLE_LINE_RE` (`# {name}\n{definition}`) and chunk embeddings use AST-resolved `ChunkPointer.title` (`Title: {title}\n{body}`) so `EmbeddingGemma` (`distinct-titled` under `retrieval-prompts-v3`) explicitly formats `title: {title} | text: {body}` without silent fallback degradation to `title: none`. Use bounded SQL top-$k$ queries, never duplicate full-corpus NumPy matrices in RAM.
   - Lexical BM25 seeding uses contentless `FTS5` (`content=''`, L1-normalized) + multi-label `entity_labels` resolution (`resolve_entity_labels`, disambiguating polysemous/homograph senses by query-vector cosine).
3. **Zero Unproven Bloat:**
   - Every added piece of graph/ranking logic must prove measurable `Recall@k` or `F1` lift on [`benchmark.py`](benchmark.py).
4. **Single-Parse CocoIndex Rust Tree-Sitter AST + Target Sink Compilation + `ManagedConnection`:**
   - `cocoindex>=1.0.0` (`CorpusIngestor`) walks `corpus_root` across 30+ supported file formats (Markdown, JSON/JSONL, HTML/XML, YAML/TOML, SQL, Python, Rust, TypeScript, etc.). All syntax-aware chunking and enclosing scope/heading resolution are executed purely via CocoIndex's single-parse Rust Tree-sitter engine (`CodeSource`, `RecursiveSplitter`, `_coco_core.render_ranges`, and `CodePattern`) with zero `markdown-it` or competing third-party parsers.
   - Declarative key projection operator $\pi_{\mathcal{W}, \mathcal{B}}$ (`json_keys_whitelist` & `json_keys_blacklist` with blacklist dominance) prunes blacklisted AST subtrees (corporate secrets like `Bearer*` and `*password*`; benchmark labels like `question*`, `answer*`, and `supporting_facts*`), guaranteeing zero test contamination ($I(\mathcal{I}; Q, A, D^*_{\text{labels}}) = 0$) and physical air-gapping directly on source files for 1-indexed line hydration without intermediate projection files.
   - `YawGraphEngine.index()`, `YawRuntime.optimize()`, and `benchmark.py` (`run_yaw_benchmark`) all ingest through `CorpusIngestor` (`sync_corpus_once`) into `yaw.db` via `cocoindex.connectors.sqlite.ManagedConnection` (`open_managed_yaw_db`, FIFO RWLock + `sqlite-vec`). The engine executes Stage 1 `ExtractEntities` (`[name, definition]`) + Stage 2 `ExtractTriples` indexing (preserving all standalone Stage 1 NER entities with non-empty `skos:definition`), the Provisional Same-Label Sense-Split Gate ($\cos \ge 0.85 \implies$ reuse concept; $\cos < 0.85 \implies$ provisional `sense_id` without premature `owl:differentFrom`), Two-Channel Candidate Combiner (`find_entity_disambiguation_candidates` surfacing kNN + `difflib` + unadjudicated same-`label_norm` pairs $\to$ `judge_synonyms` $\to$ `merge_entities` / `upsert_semantic_edges`), and in-transaction SHACL linting + RLM self-healing (`lint_graph_shacl(conn, auto_heal=False)` $\to$ `heal_shacl_violations`) inside `with self.managed_conn.transaction() as conn:`.
   - Enforce single-owner write lock via `filelock.FileLock("yaw.db.lock")`. External CLI/inspection processes (`yaw daemon status`, `yaw graph lint`) read `yaw.db` strictly via read-only `file:...?mode=ro` SQLite WAL snapshots.
   - Treat inference (`http://localhost:13305/api/v1` or configured URL) as an external persistent server; never load/unload models from the daemon.
5. **FastMCP Adapter:**
   - Mount `fastmcp` Streamable HTTP (`127.0.0.1:13306/mcp` + `/healthz` + `/readyz`) onto the single daemon runtime.
6. **Native DSPy Inference, Canonical 12 Signatures, First-Class Embedding Trail & Task-Quality Pareto Evolution:**
   - Drive all inference through `dspy` (`yaw/daemon/graph/inference/`): `dspy.LM` (`chat.py`) and `dspy.Embedder` (`embedding.py`) against Lemonade (`openai/...`) or Ollama (`ollama_chat/...`, `ollama/...`) with zero `hipporag` runtime imports and zero mode-toggle bloat (`execution_mode` and `qa_mode` are retired).
   - **Canonical 12 DSPy Signatures & First-Class Embedding Trail:** The **12 canonical predictor roles** use distinct signatures; DSPy owns its internal generated signatures:
     1. **7 Core Direct GBNF Signatures (`dspy.Predict` / `dspy.ChainOfThought`):** `ExtractEntities` (`ner`), polymorphic `ExtractTriples` (`triples`), `DefineMissingEntities` (`entity_definer`), `JudgeSynonyms` (`synonym_judge`), `ExtractQueryIntent` (`query_intent`), `FilterTriples` (`triple_filter`), and `AnswerQuestion` (`qa`).
     2. **2 SHACL Integrity Healing Signatures:** `HealSHACLViolations` (`shacl_healer_rlm` — single-level `dspy.RLM` Python REPL loop with 5 graph inspection/lint/repair tools) and `ProposeSHACLRepairs` (`shacl_repair` — inner direct GBNF tool).
     3. **3 GEPA Reflection Optimizer Signatures (`RLMInstructionProposer`):** `ReflectiveInnerDistillerSignature` (`inner_rlm` / `rlm` — sandbox evidence selector), native DSPy `ProposeInstruction` (`qwen_proposer`), and native `CompressInstruction` (`qwen_compressor`). The established public proposal/compression signature names are compatibility aliases; no second writer exists.
     Predictors use native DSPy LM binding by role, or `dspy.context` when unbound; no per-predictor LM proxy exists. Explicit per-call LM overrides and native `LM.copy()` settings are honored, including async predictor calls and candidate copies. The native writer uses the scoped worker/rollout context with the verified reflection budget guard. `YawGBNFAdapter` supplies grammar only; DSPy owns message formatting, JSON parsing and complete output validation. Fresh task/reflection programs have no demonstrations unless explicitly opted in; compiled demos are preserved. Conversation `dspy.History` is an explicit signature input, separate from diagnostic histories. Per-invocation `track_usage()` telemetry (`self.telemetry`, `self.entity_def_audit_trail`, `self.synonym_audit_trail`, `self.shacl_audit_trail`) remains bounded; native `named_predictors()` exposes direct and generated RLM predictors. Furthermore, `DSPyEmbeddingAdapter` (`dspy.Embedder`) is wired into `HippoRAGProgram.telemetry` and `HippoRAGProgram.embedding_audit_trail` via `bind_telemetry_hook`, recording every embedding invocation (`chunk_embed`, `entity_embed`, `fact_embed`, `query_fact_embed`, `query_passage_embed`), active task prefix (`query_prefix`, `fact_query_prefix`, `document_prefix`), sub-chunk splits, micro-batches, and token counts into `*_graph_rlm_audit.json`.
   - **Strict `/dspy-rlm-module` Discipline + Managed Deno WASM Sandbox (`dspy[deno]`, `dspy.PythonInterpreter`) + Configured Gemma GBNF Execution:** Chat, extraction, QA, RLM orchestration, and reflection roles use the configured model, currently `Gemma-4-E4B-it-GGUF` (`llamacpp` on Vulkan GPU; `Mistral-7B-Instruct-v0.3-Hybrid` on Ryzen AI NPU is retired). Following `/dspy-rlm-module`, `dspy.RLM` is strictly reserved for multi-step data-exploration or large-context Python REPL loops (`shacl_healer_rlm` and `RLMInstructionProposer.inner_rlm`) backed by DSPy's managed Deno/Pyodide WASM runtime (`dspy[deno]>=3.0.0`, `dspy.PythonInterpreter`); single-pass extraction, reranking, and QA never wrap `dspy.RLM`. Direct/worker roles (`extraction`, `qa`, `reflection_worker`) set `_yaw_supports_gbnf=True` (`extra_body={"grammar": ...}` with pure-JSON GBNF grammars), while RLM orchestrator roles (`rlm`, `reflection`) set `_yaw_supports_gbnf=False` so `dspy.RLM` emits unconstrained Python REPL code inside `dspy.PythonInterpreter`, invokes sandbox tools strictly with explicit keyword arguments (`current_instruction=..., examples_with_feedback=...`), and delegates `llm_query` / `llm_query_batched` to an explicit `sub_lm` (`create_dspy_rlm_pair`).
   - **Task-Quality Pareto Evolution (2026-10-24):** Never evolve prompts on full Tier 2 benchmark corpora. Evolve one universal `compiled_program.json` via `dspy.GEPA` using `compute_sizecoding_pareto_score` (historical function name): the unweighted geometric mean of labeled `Recall@1/2/5/10/20` and optional QA `EM`/token `F1`, with no denominator, smoothing floor or repeated weights. A zero component yields zero. QA-only examples use only EM/F1; retrieval-only examples use only recall. DSPy owns normalized EM and continuous best-reference token F1. Tokens, latency, SHACL and repairs are diagnostics, never Pareto objectives or penalties. SIMBA uses the same task reward; changed metric identity requires fresh optimizer/worker state. Optimize strictly on the stratified 250-query (`150` train + `100` val) **EnterpriseRAG-Bench** (`enterprise_rag_bench`, `uv run yaw optimize`) OOD dataset, then evaluate zero-shot on unseen Tier 2 holdouts.

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
5. **Zero `hipporag` Imports Across Entire Repository:** `hipporag` is completely retired and banned across `yaw/`, `benchmark.py`, `scripts/`, and `tests/`.
6. **Zero External AST / Markdown Parsers (`markdown_it`):** Parsing, syntax-aware chunking, and scope/heading resolution rely strictly on CocoIndex's single-parse Rust Tree-sitter engine (`CodeSource`, `RecursiveSplitter`, `_coco_core.render_ranges`, `CodePattern`). No third-party parsers like `markdown_it`, `BeautifulSoup`, or `ast.parse` are permitted in runtime ingestion.

## Validated Unified Local Model Stack (`http://localhost:13305/api/v1`)

- **Embeddings (`dspy.Embedder` on Vulkan GPU `llamacpp`):** `embeddinggemma-300m-GGUF` (`Q8_0`, mean pooling, `--batch-size 2048 --ubatch-size 2048`, `encoding_format="float"`, `distinct-titled` / `retrieval-prompts-v3`).
- **Unified Chat, GBNF Worker, QA Reader & RLM Orchestrator (`dspy.LM` on Vulkan GPU `llamacpp`):** `Qwen3-4B-Instruct-2507-GGUF` (`Q4_K_M`, `8192–16384` ctx, `--parallel 1 --cache-ram 0`, no `-dio`, `temp=0.2`, `freq_penalty=0.5`; `_yaw_supports_gbnf=True` for direct/worker roles `extraction`/`qa`/`reflection_worker`, and `_yaw_supports_gbnf=False` for RLM Python REPL orchestrator roles `rlm`/`reflection`).

## Execution & Hard Gates

1. **[Epic 1 (`E1-S1..E1-S6`)](docs/epic-1-hipporag-daemon.md):** Build strictly bottom-up along the containment tree: `E1-S1` (`core` + AST import lock) $\to$ `E1-S2` (`inference` native `dspy.LM`, `dspy.Embedder`, `dspy.Signature`s, `dspy.RLM` & `dspy.GEPA` loader) $\to$ `E1-S3` (`storage` `yaw.db` + line hydration) $\to$ `E1-S4` (`graph` sparse PPR + engine, **Tier 1 parity gate**) $\to$ `E1-S5` (`ingestion` CocoIndex delta pipeline) $\to$ `E1-S6` (`daemon` runtime/CLI + `dspy.GEPA` / `dspy.RLM` optimization + **Tier 2 full-corpus parity or beat gate**).
2. **[Epic 2 (`E2-S1..E2-S2`)](docs/epic-2-fastmcp-http.md):** Starts after `E1-S6`; mounts `yaw/daemon/mcp_server.py` onto `yaw/daemon/runtime.py` (`0` direct `graph`/`storage`/`inference` imports).
3. Logs use 500-line FIFO rotation (`yaw/core/logger.py`). Record all measured runs in [`docs/evaluation.md`](docs/evaluation.md).
