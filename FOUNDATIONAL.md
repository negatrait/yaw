# FOUNDATIONAL

Baseline: 2026-10-14. YAW is a native HippoRAG 2 graph & storage engine (`yaw/daemon/graph/{engine.py,ppr.py,storage/*}`) backed by one SQLite file (`yaw.db`), CocoIndex (`cocoindex>=1.0.0`) ingestion, sparse PPR with 1-indexed line hydration, and FastMCP over Streamable HTTP. Pinned upstream `hipporag` (`2.0.0a5`) is kept only in [`benchmark.py`](benchmark.py) as the reference baseline and evaluator. No `rdflib`, OWL, `pySHACL`, `spaCy`, Parquet, or Pickle in runtime.

## 6 Core Rules (Max Perf, Min Bloat)

1. **`yaw.db` SSOT + 1-Indexed Line Hydration:**
   - Zero file sprawl; zero chunk text stored in `yaw.db`.
   - `chunks(chunk_id, rel_posix_path, start_line, end_line, title, content_hash)` stores 1-indexed line spans.
   - Graph lives in `edges(src_id, dst_id, edge_kind, weight)` + `edge_sources(src_id, dst_id, edge_kind, chunk_id)`.
   - Run sparse PPR (`scipy.sparse.csr_matrix` power iteration or Forward Push) in `<3 ms`. Hydrate disk lines **only** for the final top-$k$ chunks; verify `content_hash` on read (`stale_on_disk=True` on mismatch).
2. **`sqlite-vec` + Contentless `FTS5` Seeding:**
   - Vectors live in `chunk_vec`, `entity_vec`, `fact_vec` (`sqlite-vec` / `float32` BLOBs). Use bounded SQL top-$k$ queries, never duplicate full-corpus NumPy matrices in RAM.
   - Lexical BM25 seeding uses contentless `FTS5` (`content=''`, `1 / rank` L1-normalized) + exact entity/alias lookup.
3. **Zero Unproven Bloat:**
   - Every added piece of graph/ranking logic must prove measurable `Recall@k` or `F1` lift on [`benchmark.py`](benchmark.py).
4. **CocoIndex Daemon + External Inference:**
   - `cocoindex>=1.0.0` watches `corpus.root`, chunks files on line boundaries, and feeds coalesced `(added_chunks, deleted_chunk_ids)` deltas into a single serialized `yaw.db` worker thread.
   - Stage 1 OpenIE caches per-request LLM outputs in `yaw.db` (`llm_cache`) for instant crash resume; Stage 2 graph/vector commit runs in one SQLite transaction.
   - Enforce single-owner lock via `filelock.FileLock("yaw.db.lock")` + bound port (never PID-alive checks).
   - Treat inference (`http://localhost:13305/api/v1` or configured URL) as an external persistent server; never load/unload models from the daemon.
5. **FastMCP Adapter:**
   - Mount `fastmcp` Streamable HTTP (`127.0.0.1:13306/mcp` + `/healthz` + `/readyz`) onto the single daemon runtime.
6. **Native DSPy Inference, Heterogeneous Mistral RLM + Qwen GBNF & Anti-Lock-In `MIPROv2`/`GEPA` Evolution (`dspy.LM` + `dspy.Embedder` + `dspy.RLM` + `dspy.MIPROv2`/`dspy.GEPA`):**
   - Drive all inference through `dspy` (`yaw/daemon/graph/inference/`): `dspy.LM` (`chat.py`) and `dspy.Embedder` (`embedding.py`) against Lemonade (`openai/...`) or Ollama (`ollama_chat/...`, `ollama/...`) with zero `hipporag` runtime imports.
   - Express NER (`ExtractEntities`), Triple Extraction (`ExtractTriples`), Recognition Memory (`FilterFacts`), and Multi-Hop QA (`AnswerQuestion`) as typed `dspy.Signature`s and `dspy.Module`s in `prompts.py` using **`dspy.JSONAdapter()`** (pure JSON, zero `[[ ## ... ## ]]` output headers).
   - **Heterogeneous Mistral Orchestrator + Qwen GBNF Worker (`execution_mode="rlm"` across all stages):** GBNF (`extra_body={"grammar": ...}`) is supported **only** on `llama.cpp` (`Qwen3-4B-Instruct-2507-GGUF` on Vulkan GPU) and **never** on Ryzen AI NPU/OGA (`Mistral-7B-Instruct-v0.3-Hybrid`, enforced by `lm_supports_gbnf`). In `execution_mode="rlm"`, **every** stage (`ner_rlm`, `triples_rlm`, `fact_filter_rlm`, `qa_rlm`) runs `Mistral-7B` on Ryzen AI NPU as the outer `dspy.RLM` Python REPL orchestrator (zero GBNF), delegating inner `llm_query` and built-in `qwen_*` worker tools (`qwen_extract_entities`, `qwen_extract_triples`, `qwen_filter_facts`, `qwen_answer_cot`) to `Qwen3-4B` on `llama.cpp` GPU with pure-JSON GBNF grammars (`build_dspy_ner_grammar`, `build_dspy_triples_grammar`, `build_dspy_rerank_grammar`, `build_dspy_qa_cot_grammar`).
   - **Zero-Lock-In Sizecoding Pareto Evolution Policy:** Never evolve or fine-tune prompts on the full Tier 2 benchmark corpora. Evolve one universal `compiled_program.json` via `dspy.MIPROv2` + `dspy.GEPA` using `compute_sizecoding_pareto_score` ($\text{pareto} = \text{Recall@5} / (\text{tokens} \times \text{time})$ with a hard minimum floor locked at the Tier 1 baseline) strictly on **Tier 1 small slices** (`--max-queries 10`) and/or **external out-of-distribution RAG datasets** (`onyx-dot-app/EnterpriseRAG-Bench`, `microsoft/graphrag-benchmarking-datasets`), then evaluate that frozen program zero-shot across the unseen Tier 2 full-corpus holdouts (`musique`, `2wikimultihopqa`, `hotpotqa`).

## Domain Containment Tree (Zero Cross-Domain / Sibling Edges)

Every package owns its children; **horizontal imports between sibling domains are forbidden (`0` cross-domain edges)**. A module in package `X` may only import its own children `X.*` or `yaw.core.*`:

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

## Validated 3-Slot Local Model Stack (`http://localhost:13305/api/v1`)

- **Embeddings (`dspy.Embedder` on Vulkan GPU `llamacpp`):** `embeddinggemma-300m-GGUF` (`Q8_0`, mean pooling, `--batch-size 2048 --ubatch-size 2048`, `encoding_format="float"`, `distinct-titled` / `retrieval-prompts-v3`).
- **GBNF RLM Sub-LM Worker & Direct Extraction/Rerank (`dspy.LM` on Vulkan GPU `llamacpp`):** `Qwen3-4B-Instruct-2507-GGUF` (`Q4_K_M`, `8192–16384` ctx, `--parallel 1 --cache-ram 0`, no `-dio`, `temp=0.2`, `freq_penalty=0.5`, `_yaw_supports_gbnf=True`).
- **RLM Orchestrator & QA Reader (`dspy.LM` on Ryzen AI Hybrid NPU/OGA `ryzenai-llm`):** `Mistral-7B-Instruct-v0.3-Hybrid` (`32786` ctx, `temp=0.2`, `_yaw_supports_gbnf=False`, system-prompt folded into first user turn for RyzenAI Jinja).

## Execution & Hard Gates

1. **[Epic 1 (`E1-S1..E1-S6`)](docs/epic-1-hipporag-daemon.md):** Build strictly bottom-up along the containment tree: `E1-S1` (`core` + AST import lock) $\to$ `E1-S2` (`inference` native `dspy.LM`, `dspy.Embedder`, `dspy.Signature`s, `dspy.RLM` & `dspy.GEPA` loader) $\to$ `E1-S3` (`storage` `yaw.db` + line hydration) $\to$ `E1-S4` (`graph` sparse PPR + engine, **Tier 1 parity gate**) $\to$ `E1-S5` (`ingestion` CocoIndex delta pipeline) $\to$ `E1-S6` (`daemon` runtime/CLI + `dspy.GEPA` / `dspy.RLM` optimization + **Tier 2 full-corpus parity or beat gate**).
2. **[Epic 2 (`E2-S1..E2-S2`)](docs/epic-2-fastmcp-http.md):** Starts after `E1-S6`; mounts `yaw/daemon/mcp_server.py` onto `yaw/daemon/runtime.py` (`0` direct `graph`/`storage`/`inference` imports).
3. Logs use 500-line FIFO rotation (`yaw/core/logger.py`). Record all measured runs in [`docs/evaluation.md`](docs/evaluation.md).
