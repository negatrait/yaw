# DSPy Practical Improvements & Roadmap

Practical design patterns, lessons learned, and future architecture investigations for YAW.

---

## 1. Zero-Shot Optimization on Local Silicon (Active Epic 1 Rule)

- **The Issue**: Bootstrapping few-shot demonstrations on local 4B models (`Qwen3-4B-Instruct`) causes prompt bloat ($3,500+$ tokens), generation timeouts, repetition loops, and severe Pareto penalties ($b_{\text{tokens}} = 1 + \text{tokens}/100$).
- **The Rule**: Always set `max_bootstrapped_demos = 0` and `max_labeled_demos = 0` in `yaw.toml`. Optimize pure instructions zero-shot.

---

## 2. Query-Time Embedding Prefix Tuning

- **The Idea**: `EmbeddingGemma` uses asymmetric task prefixes (e.g. `task: search query | query: `).
- **The Leverage**: Mutating query prefixes only affects search-time encoding ($\sim 0.05\text{s}$ per query). It avoids re-embedding the corpus while allowing `dspy.GEPA` to align dense retrieval with the reasoning prompts.
- **Implementation**: Expose `query_embedding_prefix` as a `dspy.Predict` signature in `HippoRAGProgram.named_predictors()`, syncing instructions into `DSPyEmbeddingAdapter.update_prefixes()`.

---

## 3. Dynamic Hyperparameter Tuning: Simple `dspy.Predict` + GBNF

If the objective is simply tuning runtime hyperparameters (`top_k`, `ppr_alpha`, `temperature`, `max_tokens`):
- Do **not** use complex sandboxed code execution for simple configuration.
- A lightweight `dspy.Predict` with a strict GBNF JSON grammar (`{"top_k": int, "ppr_alpha": float}`) is faster, safer, and avoids interpreter overhead.

---

## 4. Post-Epic-2 Roadmap: Tool-Driven Orchestration via `dspy.Flex`

The real leverage of `dspy.Flex` is **arbitrary code and tool execution**, not parameter tweaking:

- **Concept**: Expose YAW's core retrieval and graph primitives as native tools to `dspy.Flex(Signature, tools=[...])`:
  - `dense_search(query, top_k)`
  - `ppr_walk(seed_entities, alpha, top_k)`
  - `query_fts(text, top_k)`
  - `inspect_node(entity_id)`
  - `hydrate_passages(chunk_ids)`
- **Mechanism**:
  - `dspy.Flex` defaults to an internal `dspy.RLM` over these tools.
  - `dspy.GEPA` rewrites the Python orchestrator class (`module_src`), allowing the model to write arbitrary multi-step retrieval loops, adaptive backtrack heuristics, and custom graph traversal logic.
  - All generated code executes isolated inside `BridgeRuntime` (fresh sandbox per query).
- **Status**: Deferred for investigation **after Epic 2** (FastMCP HTTP transport).
