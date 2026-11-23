# DSPy QA, `lm15` Provider & GEPA Optimization (`yaw/qa.py`, `yaw/optimize.py`)

Status: active (`CS-4`..`CS-6` & `EVAL-5` `M_FINAL_1B`, `5c4b115`).

| Relation | Links |
| --- | --- |
| Governed by | [FOUNDATIONAL.md](../FOUNDATIONAL.md), [AGENTS.md](../AGENTS.md), [epic-1-hipporag-daemon.md](epic-1-hipporag-daemon.md) |
| Rules | [qa.md](../.agents/rules/qa.md): `yaw/qa.py`, `yaw/optimize.py`; [tests.md](../.agents/rules/tests.md): `tests/**/*.py`; [docs.md](../.agents/rules/docs.md): `docs/**/*.md` |
| Owns code | `yaw/qa.py`, `yaw/optimize.py` |
| Peers | [epic-1-hipporag-daemon.md](epic-1-hipporag-daemon.md): owns `yaw/core/` (`YawConfig.for_optimization`, `QAResult`, `YawRuntimeError`, `write_text_atomic`); [native-cocoindex-indexing.md](native-cocoindex-indexing.md): owns `yaw/pipeline.py` & `yaw/datasets.py` (`start_pipeline`, `update_pipeline_once`, `close_pipeline`, `make_dspy_embed_fn`, `resolve_slice_corpus_scope`, `compute_qa_em_f1`); [native-cocoindex-graph.md](native-cocoindex-graph.md): owns `yaw/retrieval.py` (`RetrievalIndex`, `search_index`, `lexical_search`, `expand_citation_neighbors`, `hydrate_ranked_passages`, `hydrate_passage`); [epic-2-fastmcp-http.md](epic-2-fastmcp-http.md): owns `yaw/runtime.py`, `yaw/mcp_server.py` & `yaw/cli.py` (`ask_question`, `resolve_program_path`, lazy `yaw optimize dspy`); [enterprise-rag-bench-training-plan.md](enterprise-rag-bench-training-plan.md): owns `scripts/{benchmark,download_datasets,verify_paper_parity_gate,evaluation}.py` consuming `yaw/qa.py` & `.yaw/qa_program.json` and defining the `135/27` EnterpriseRAG-Bench split; [epic-3-performance.md](epic-3-performance.md): owns `scripts/watchdog.py` (`dspy_status.json` & `dspy_gepa_audit.json` telemetry) & deferred `dspy.RLM` / `dspy.Flex` / `dspy.BetterTogether` research; [coverage-plan.md](coverage-plan.md): owns `tests/**/*.py` (`tests/test_qa.py`, `tests/test_optimize.py`) |
| References | [model-evaluation.md](model-evaluation.md) (historical local hardware profiles), [DSPy lm15 migration guide](https://dspy.ai/current/community/normalized-lm-api-migration/), [Unsloth API](https://unsloth.ai/docs/basics/api) |
| Records | [inference-backend.md](reviews/inference-backend.md), [bare-bones-replacement-sprint.md](reviews/bare-bones-replacement-sprint.md), [evaluation-inference.md](evaluation-inference.md), [evaluation.md](evaluation.md) |

---

<a id="1-zero-shot-optimization-reward-contract--runtime-binding-active-epic-1-rules"></a>

## 1. Bare `dspy.ReAct` QA & Zero Wrapper Subclasses (`yaw/qa.py`)

- **Zero Wrapper Subclasses (`FOUNDATIONAL.md §2`):** Production code never subclasses `dspy.Module`, `dspy.BaseLM`, or `dspy.Adapter`. `build_qa_agent(tools, *, max_iters=6, artifact_path=None) -> dspy.ReAct` instantiates bare `dspy.ReAct(QASignature, tools=list(tools), max_iters=max_iters)`. When loading a compiled artifact, it calls `agent.load(str(artifact_path))` and `agent.set_lm(None)` so deployment-owned endpoints and capacities from `yaw.toml` are never overwritten by serialized prompt state, then enforces `agent.react.update_config(tool_choice="required")`.
- **Zero-Shot, `temperature = 1.0` & Zero Silent Truncation:** Both student QA episodes (`yaw/qa.py`) and GEPA reflection (`yaw/optimize.py`) execute unconditionally at `temperature = 1.0` (`create_lm(cfg, temperature=1.0)`) so `gemma-4-e4b` tool calling avoids greedy repetition loops. Prompts, tool outputs, and GEPA instructions are never truncated (`[:max_chars]`); provider context overflows raise `ContextWindowExceededError` -> `YawRuntimeError("MODEL_ERROR", ...)` cleanly. When `cfg.max_output_tokens` is `None`, `max_tokens=None` leaves generation budget to the serving endpoint.
- **Four Read-Only Corpus Tools (`make_qa_tools`):**
  1. `search_passages(query, top_k=8)`: executes `search_index` over `RetrievalIndex` with `pool_k = max(k * 4, 50)` followed by `hydrate_ranked_passages` (`max_chars=None` uncropped).
  2. `lexical_grep(query, top_k=10)`: executes zero-embedder `lexical_search` with `pool_k = max(k * 4, 50)` + `hydrate_ranked_passages`.
  3. `expand_neighbors(chunk_id, top_k=6)`: validates `chunk_id` in `index.id_to_pos`, executes `expand_citation_neighbors` with `pool_k = max(k * 4, 50)` + `hydrate_ranked_passages`.
  4. `read_chunk(chunk_id)`: hydrates the exact requested `chunk_id` via `_hydrate_ids` without span deduplication.
  Every DB access uses read-only `?mode=ro` handles (`open_readonly_db`) with zero SQLite writes during QA.
- **Strict `[chunk_id=<int>]` Citation Grounding (`ask_question`):**
  - Verifies `termination_reason in (None, "submit", "forced_submit")` and non-empty `answer` (`MODEL_ERROR` otherwise).
  - Parses `[chunk_id=<int>]` blocks (`_CITATION_BLOCK_RE`, including comma/semicolon lists) and validates every cited integer against non-`submit` `iter_tool_observations` (`_extract_seen_chunks`) and `index.id_to_pos` (`YawRuntimeError("MODEL_ERROR", ...)` on ungrounded citations).
  - Re-hydrates cited passages from disk (`_hydrate_ids`, up to `max_citations=5`); any `stale_on_disk=True` citation aborts immediately with `YawRuntimeError("CONSISTENCY_ERROR", ...)`. `ABSTAIN_REPLY = "Insufficient references in corpus"`.

---

## 2. Canonical `dspy.GEPA` Instruction Optimization (`yaw/optimize.py`)

- **Rich-Feedback Metric (`qa_gepa_metric`):**
  - 5-argument GEPA metric `(gold, pred, trace, pred_name, pred_trace) -> ScoreWithFeedback`.
  - Evaluates `strip_inline_citations(raw_ans)` against `gold_answers` via `compute_qa_em_f1` (`yaw/datasets.py`), inspects tool calls and observations via `collect_tool_history`, and scores inline citation grounding `cite_score`.
  - When `gold_docs` are provided, computes `score = 0.55 * f1 + 0.15 * em + 0.15 * cite_score + 0.15 * ev_recall` (otherwise `0.65 * f1 + 0.20 * em + 0.15 * cite_score`) alongside diagnostic feedback on F1/EM, citation grounding, and missing evidence coverage.
- **Bare `dspy.ReAct` Instruction Proposer (`make_react_instruction_proposer`):**
  - Strips duplicated `"tools"` schemas from `reflective_dataset` `Inputs` and exposes `list_examples()` and `inspect_example(index, field="")` tools (raising native `TypeError`/`IndexError`/`KeyError` -> `is_error=True` on invalid arguments) to bare `dspy.ReAct(ProposeInstruction, tools=[list_examples, inspect_example], max_iters=max_iters)` with `tool_choice="required"`.
  - Runs at `temperature = 1.0` under `with dspy.context(lm=reflection_lm):`, rejecting un-submitted trajectories or empty instructions without silent truncation.
- **Sandbox Lifecycle & Held-Out Slice Guard (`optimize_dspy`, `_compile_dspy`):**
  - Resolves optimization paths via `cfg.for_optimization()` (`optimize_yaw.db`, `.yaw/qa_program.json`, `dspy_gepa_audit.json`, `dspy_status.json`, `gepa_logs/`) while keeping `.yaw/coco_lmdb` shared for memoization, serialized by `FileLock(opt_cfg.lock_path, timeout=0)` (`BUSY` when contended). (At inference time, `resolve_program_path` in `yaw/runtime.py` checks `cfg.compiled_program_path` `.yaw/compiled_program.json` first and falls back to `.yaw/qa_program.json`.)
  - Enforces `_guard_optimization_slice`: refuses `query_offset < 10` on parity/Tier-2 corpora (`INVALID_INPUT`) so `[0, 10)` remains strictly uncontaminated.
  - Indexes prepared v2 `.md` slices directly via `resolve_slice_corpus_scope` and `start_pipeline` / `update_pipeline_once` (`--fresh` runs `drop_blocking()` and wipes `.yaw/optimize_*`), builds `RetrievalIndex`, compiles `student` via `dspy.GEPA(..., reflection_lm=r_lm, instruction_proposer=proposer, track_stats=True, seed=42)`, saves `.yaw/qa_program.json`, and writes atomic status/audit JSON.

---

<a id="5-post-epic-2-roadmap-tool-driven-orchestration-via-dspyflex"></a>

## 3. Deferred `dspy.RLM` / `dspy.Flex` Scope

`dspy.RLM` (Deno WASM REPL) and `dspy.Flex` are retired from `yaw/qa.py` and `yaw/optimize.py` in favor of bare `dspy.ReAct` over bounded read-only corpus tools; any future `>100K`-token REPL or `dspy.Flex` investigation remains strictly parked in [epic-3-performance.md](epic-3-performance.md) pending controlled benchmarks.

---

<a id="6-inference-backend-contract-dspy-lm15-zero-provider-plumbing"></a>

## 4. Inference Backend Contract: DSPy `lm15` (`create_lm` in `yaw/qa.py`)

- **Config-Only Backend Selection:** `create_lm(cfg, *, temperature=1.0) -> dspy.LM` targets any OpenAI-compatible `/v1/chat/completions` endpoint via `cfg.provider`, `cfg.qa_model`, `cfg.base_url`, and `cfg.api_key` (`UNSLOTH_STUDIO_AUTH_TOKEN` or `OPENAI_API_KEY`).
- **Native `lm15` Provider Declaration:** When `model.startswith("openai/")`, `create_lm` registers `"yaw-openai"` via public `dspy.lm15.register_provider(ProviderDefinition.chat(AccessPolicy(provider="yaw-openai", base_url=cfg.base_url, supports=EndpointSupport(complete=True), auth_modes=("bearer",)), compat=OpenAIChatCompat(max_tokens_field="max_tokens")), supports=ModelSupport(function_calling=True), replace=True)` and returns `dspy.LM(model, model_type="chat", engine="lm15", temperature=temperature, max_tokens=cfg.max_output_tokens, cache=False, num_retries=3, api_base=cfg.base_url, api_key=cfg.api_key or "yaw")`.
- **Fail-Fast & Zero Custom Plumbing:** Transient retries use `dspy.LM(num_retries=3)` exclusively (`0` custom retry loops, `0` custom `BaseLM` / `Adapter` subclasses, `0` GBNF/XML/token parsers). Exhausted retries, auth errors, or overflow errors propagate immediately as typed failures.

## 5. Quality Gates & Verification

Real-substrate tests live in `tests/test_qa.py` and `tests/test_optimize.py` (loopback HTTP `OpenAI`-compatible chat/tool endpoints, unmocked `RetrievalIndex` + SQLite `?mode=ro` fixtures, inline `[chunk_id=<int>]` verification, `stale_on_disk` `CONSISTENCY_ERROR`, `[0, 10)` slice guard, and `FileLock` contention).

```powershell
uv run --offline --locked ty check
uv run --offline --locked ruff check
uv run --offline --locked ruff format --check
uv run --offline --locked pytest -o addopts="" tests/test_qa.py tests/test_optimize.py
git diff --check
```

### Decisions

- 2026-11-12, user: "use lm15 (https://dspy.ai/current/community/normalized-lm-api-migration/) and zero custom plumbing! Spec it."
- 2026-11-12, user: "Would maybe be nice to have a solution that allows changing the inference [...] what if, at some stage, I want to use OpenAI paid API to run yaw? And production usually likes vLLM?"
- 2026-11-12, user: inference backend is Unsloth at `http://127.0.0.1:8888/v1`, aliases `embeddinggemma-2` and `gemma-4-e4b` (implemented in commit `7fbec2b`).
- 2026-11-13, user (pointing at the completions endpoint): "You get logprobs by calling the OpenAI-compatible **completions** endpoint with the `logprobs` parameter, then reading `response.choices[0].logprobs.content[0][\"top_logprobs\"]`."
- 2026-11-13, user *(Superseded by `CS-4` / `EVAL-5`)*: Route B, minimal code change for now. The 150 tok/s average compared to 60 tok/s on lemonade is what I need right now, to be able to pass the epic 1 gates in reasonable time. Might be worth revisiting when epic 1 is officially closed. You'll also need to check the embeddinggemma-2 embedder. Also: The gemma-4 thinking contract requires the <|think|> token as the first token, see the model card.\n- Resolved by the 2026-11-13 decision: structured-output and logprob route is Route B (text completions). Implemented in inference/completions.py:GemmaCompletionsEngine (a dspy.LM(engine=...) that renders the Gemma 4 turn format and posts grammar + integer logprobs to /v1/completions); worker roles only, orchestrator roles keep the chat path. Revisit Route A after Epic 1 closes.\n- Still pending: reasoning mode for orchestrator roles on the chat route, registry preflight (§6.4 Open items). Pre-call sizing was resolved by the bounded-batch decision below.

- 2026-11-13, wiring audit user decision: "Apply the spec’s bounded-batch approach".
  Pre-call exact token counting is retired. Semantic packets use
  `semantic_adjudication_batch_size` pairs. Reflection final formatted requests use
  the explicit `reflection_input_chars` cap (default 65536 characters), without
  claiming equivalence to tokens. Native provider context errors propagate. Actual
  token usage comes only from DSPy response usage; no surrogate tokenizer.

### Generation capacity ownership

Omitted `max_output_tokens` imposes no YAW generation-token limit. DSPy and both completion transports must omit the outbound token-limit field, leaving available context and generation capacity to the endpoint. Only explicit positive output limits are validated against active context and transmitted. This supersedes automatic output-token scaling; RLM tool-output character safeguards remain separate.
