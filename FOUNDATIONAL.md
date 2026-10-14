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
6. **Lean Native Graph Engine + Owned DSPy/GEPA Prompts:**
   - Own the NER, Triple Extraction, Recognition Memory (`DSPyFilter`), and QA prompts directly in `yaw/daemon/graph/inference/prompts.py` for direct DSPy/GEPA optimization on local 4B/7B models. Keep implementations direct and self-contained; no code-golfing, no artificial LOC quotas, and no multi-store synchronization sprawl.

## Domain Containment Tree (Zero Cross-Domain / Sibling Edges)

Every package owns its children; **horizontal imports between sibling domains are forbidden (`0` cross-domain edges)**. A module in package `X` may only import its own children `X.*` or `yaw.core.*`:

```text
yaw/
  core/                    types.py, config.py, logger.py          (imports: stdlib only)
  daemon/                  runtime.py, cli.py, mcp_server.py       (imports: daemon.*, core)
    ingestion/             corpus.py                               (imports: core, cocoindex)
    graph/                 engine.py, ppr.py                       (imports: graph.*, core, scipy)
      storage/             db.py, hydration.py                     (imports: core, sqlite3, sqlite_vec)
      inference/           models.py, prompts.py, embedding.py, chat.py (imports: core, openai)
  __main__.py                                                      (imports: daemon.cli)
```

## Validated 3-Slot Local Model Stack (`http://localhost:13305/api/v1`)

- **Embeddings (CUDA `llamacpp`):** `embeddinggemma-300m-GGUF` (`Q8_0`, mean pooling, `--batch-size 2048 --ubatch-size 2048`, `distinct-titled` / `retrieval-prompts-v3`).
- **Extraction & Rerank (CUDA `llamacpp`):** `Qwen3-4B-Instruct-2507-GGUF` (`Q4_K_M`, `--parallel 1 --cache-ram 0`, no `-dio`, `temp=0.2`, `freq_penalty=0.5`, `gbnf-v3:require-stop`).
- **QA Reader (Ryzen AI Hybrid NPU/iGPU `ryzenai-llm`):** `Mistral-7B-Instruct-v0.3-Hybrid` (`temp=0.2`, `fold-system:prompt-v4:require-stop`).

## Execution & Hard Gates

1. **[Epic 1 (`E1-S1..E1-S6`)](docs/epic-1-hipporag-daemon.md):** Build strictly bottom-up along the containment tree: `E1-S1` (`core` + AST import lock) $\to$ `E1-S2` (`inference` HippoRAG prompt/parser port) $\to$ `E1-S3` (`storage` `yaw.db` + line hydration) $\to$ `E1-S4` (`graph` sparse PPR + engine, **Tier 1 parity gate**) $\to$ `E1-S5` (`ingestion` CocoIndex delta pipeline) $\to$ `E1-S6` (`daemon` runtime/CLI + **Tier 2 full-corpus parity or beat gate**).
2. **[Epic 2 (`E2-S1..E2-S2`)](docs/epic-2-fastmcp-http.md):** Starts after `E1-S6`; mounts `yaw/daemon/mcp_server.py` onto `yaw/daemon/runtime.py` (`0` direct `graph`/`storage`/`inference` imports).
3. Logs use 500-line FIFO rotation (`yaw/core/logger.py`). Record all measured runs in [`docs/evaluation.md`](docs/evaluation.md).
