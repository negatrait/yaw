# Local model compatibility and benchmark comparison

Decision: 2026-10-14. This document owns the selected model matrix and the comparison protocol. FOUNDATIONAL owns runtime scope; Epic 1 owns implementation acceptance. **Report differences; the user will decide what constitutes substantial degradation.** No automatic F1 cutoff selects a winner.

## Selected development models

| Role | Lemonade ID | Checkpoint | Model contract |
| --- | --- | --- | --- |
| Embedding (default) | `embeddinggemma-300m-GGUF` | `ggml-org/embeddinggemma-300M-GGUF:embeddinggemma-300M-Q8_0.gguf` | 768 dimensions, 2048 model tokens, mean pooling, GPU CUDA `llamacpp`; supports `search`, `qa`, `fact`, `distinct`, `titled`, `distinct-titled` task modes |
| Embedding (QAT variant) | `embeddinggemma-300m-qat-q8_0-GGUF-Q8_0` | `ggml-org/embeddinggemma-300m-qat-q8_0-GGUF:Q8_0` | 768 dimensions, 2048 model tokens, mean pooling; alternate QAT `Q8_0` weights (same 300M architecture, not a capacity step-up) |
| Embedding (step-up) | `Qwen3-Embedding-0.6B-GGUF` | `Qwen/Qwen3-Embedding-0.6B-GGUF:Qwen3-Embedding-0.6B-Q8_0.gguf` | 1024 dimensions, 32768 model tokens, last-token pooling |
| Extraction / rerank (`DEFAULT_EXTRACTION_CHAT`) | `Qwen3-4B-Instruct-2507-GGUF` | `unsloth/Qwen3-4B-Instruct-2507-GGUF:Qwen3-4B-Instruct-2507-Q4_K_M.gguf` | GPU CUDA `llamacpp` (fits in VRAM alongside `embeddinggemma-300m-GGUF`); tightened OpenIE NER JSON Schema (`minLength: 1`) + dynamic entity-anchored triple GBNF + dynamic candidate-literal `DSPyFilter` GBNF (`gbnf-v3:require-stop`) |
| QA reading (`DEFAULT_QA_CHAT`) | `Mistral-7B-Instruct-v0.3-Hybrid` | `amd/Mistral-7B-Instruct-v0.3-onnx-ryzenai-1.7-hybrid` | Ryzen AI Hybrid NPU/iGPU (`ryzenai-llm`, `fold-system:prompt-v3:require-stop`); folds leading `system` turn into first `user` turn, normalizes `\_`, and enforces `Answer:` delimiter contract |
| Chat / step-up | `gemma4-it-12b-FLM` | `gemma4-it:12b` | NPU step-up baseline (`flm`, `think=False`) |
| Chat / step-up | `qwen3.5-9b-FLM` | `qwen3.5:9b` | NPU step-up comparison (`flm`, `think=False`) |
| Extraction candidate | `Phi-4-mini-instruct-GGUF` | `unsloth/Phi-4-mini-instruct-GGUF:Phi-4-mini-instruct-Q4_K_M.gguf` | GPU CUDA `llamacpp`; passes with GBNF grammar enforcement, lower triple richness than Qwen3-4B |
| Extraction candidate | `Llama-3.2-3B-Instruct-GGUF` | `unsloth/Llama-3.2-3B-Instruct-GGUF:Llama-3.2-3B-Instruct-UD-Q4_K_XL.gguf` | GPU CUDA `llamacpp`; failed `DSPyFilter` seed selection on Cinderella (`graph_seeds: []`) |

All listed models are downloaded in Lemonade's local registry. In the default 3-slot development configuration (`max_models = 3`), `embeddinggemma-300m-GGUF` and `Qwen3-4B-Instruct-2507-GGUF` reside in discrete GPU VRAM (~6.4 / 8.0 GB utilized) while `Mistral-7B-Instruct-v0.3-Hybrid` runs on the Ryzen AI Hybrid NPU + Radeon 890M iGPU path (~81% total system RAM). Avoid keeping multiple `llamacpp` chat models loaded simultaneously in GPU VRAM during latency comparisons. Long-input/boundary validation remains required.

EmbeddingGemma-2/multimodal work is **deferred**: no exact compatible checkpoint is selected, and this text-only adapter does not claim multimodal support.

## One adapter, explicit contracts

`yaw/embedding.py` defines the three immutable base profiles, `resolve_embedding_profile(model, task_mode)` for EmbeddingGemma task modes (`search`, `qa`, `fact`, `distinct`, `titled`, `distinct-titled`), and one `RetrievalEmbedding` adapter. Chat uses `yaw/chat.py`, a small policy subclass of upstream HippoRAG's `CacheOpenAI` client passed to `HippoRAG(extraction_llm=..., qa_llm=...)`:

- Recipe-specific request policies are explicit and isolated in `LocalChat`:
  - `flm` (`gemma4-it-12b-FLM`, `qwen3.5-9b-FLM`): sets `extra_body={"think": False}` to disable reasoning chains on structured extraction and QA calls.
  - `ryzenai-llm` (`Mistral-7B-Instruct-v0.3-Hybrid`): folds a leading `system` message into the first `user` message before caching/inference so Ryzen AI's Mistral `[INST]` template does not place `system` after few-shot demonstrations, normalizes Markdown-escaped `\_` in JSON keys, and normalizes/validates the final `Answer:` delimiter on QA responses.
  - `llamacpp` (`Qwen3-4B-Instruct-2507-GGUF`, `Phi-4-mini-instruct-GGUF`, `Llama-3.2-3B-Instruct-GGUF`): enforces a tightened copy of HippoRAG's `PROMPT_JSON_TEMPLATE["ner"]` (`minLength: 1` on string items) via `response_format` on NER prompts; enforces a per-chunk dynamic entity-anchored GBNF grammar (`_build_triple_grammar(messages)`, falling back to tightened `PROMPT_JSON_TEMPLATE["triples"]` when `named_entities` is absent/malformed) on `triple_extraction` prompts so every triple anchors at least its subject or object to an exact Stage 1 entity literal; enforces a per-call dynamic candidate-literal GBNF grammar (`_build_rerank_grammar(messages)`, falling back to `RERANK_GBNF_GRAMMAR`) on `DSPyFilter` recognition-memory prompts so `llamacpp` can only emit exact candidate triples from `fact_before_filter`; and enforces `QA_GBNF_GRAMMAR` (`thought "\nAnswer: " answer`) on QA prompts.
- Both Gemmas default to Google's retrieval query prefix (`task: search result | query: `) and document prefix (`title: none | text: `), and support `--embedding-task-mode distinct` (`query_instruction_mode = "distinct"`, routing `query_to_fact` to `task: fact checking | query: ` and `query_to_passage` to `task: question answering | query: `), `--embedding-task-mode titled` (`parse_document_title = True`, formatting multi-line `f"{title}\n{body}"` passages as `title: {title} | text: {body}` while keeping `title: none | text: ` for single-line passages, entities and facts), and `--embedding-task-mode distinct-titled` (both). Qwen uses its instruction plus newline/Query separator, and unprefixed documents.
- Preserve prepared inputs, including Qwen's newline. The pinned upstream `encode` flattens newlines, so the shared adapter owns that small request/response boundary while inheriting SDK transport/retries, batching, usage aggregation and normalization.
- Reject unknown instructions, already prefixed inputs, wrong model/config pairs, wrong vector dimensions, invalid row indices/usage, nonfinite and zero vectors.
- `yaw/models.py` verifies downloaded registry identity, role and recipe for embedding, extraction chat and QA chat models before inference. It checks loaded backend health and explicit pooling, and adds the resolved local weight path to the profile fingerprint. Persisted index identity combines the resolved embedding identity and the `extraction_llm` policy identity (`chat_identity(extraction_model)`).
- Smoke extraction uses one worker and zero automatic request retries. Both injected chat clients and the embedding client are closed by the caller. Each smoke run uses a fresh directory and records both graph retrieval recall (`rag_qa`) and dense passage retrieval recall (`retrieve_dpr`), plus per-client call/token/error summaries.

## Commands available now

```powershell
uv run --locked python live_smoke.py
uv run --locked python live_smoke.py --mode eval
uv run --locked python live_smoke.py --mode eval --embedding-task-mode distinct
uv run --locked python live_smoke.py --mode eval --embedding-task-mode distinct-titled
uv run --locked python live_smoke.py --mode eval --extraction-model Phi-4-mini-instruct-GGUF --qa-model Mistral-7B-Instruct-v0.3-Hybrid
uv run --locked python live_smoke.py --mode eval --chat-model gemma4-it-12b-FLM --embedding-model embeddinggemma-300m-GGUF
uv run --locked python live_smoke.py --mode eval --chat-model gemma4-it-12b-FLM --embedding-model Qwen3-Embedding-0.6B-GGUF
```

`--mode embeddings` exercises only embeddings, with five batch/single/reordered repeats per active role and cosine >= 0.999. `--mode eval` performs extraction, graph indexing, graph retrieval, DPR baseline comparison and QA against the nine-passage fixture and prints gold recall/EM/F1. `--extraction-model` and `--qa-model` default to `Qwen3-4B-Instruct-2507-GGUF` and `Mistral-7B-Instruct-v0.3-Hybrid`; passing `--chat-model` overrides both with a single model.

Lemonade settings for these installed embedding models are CUDA and `--parallel 1`, with `--pooling mean` and `ctx_size=2048` for each Gemma, or `--pooling last` and `ctx_size=32768` for Qwen. Restart/reload an already loaded backend after a settings change; saving options alone need not change a running backend. Do not silently change its pooling in response to a failed test.

## Real benchmark gate: implementation still required

Use the pinned HippoRAG revision `d5c8329422e0a0b834a15874545cb6a74b4f9b26`, its `reproduce/dataset` corpora and query files for **HotpotQA, 2WikiMultiHopQA and MuSiQue**. Each supplied query file contains 1000 questions; these are the upstream reproduction sets, not a claim to evaluate every question in the original datasets. The source's `main.py` defines document rendering and gold construction. Its CLI does not inject our embedding adapter, so invoking it unchanged would not reproduce this integration.

The nine-document smoke fixture validates adapter contracts, GBNF grammar enforcement, task-mode dispatch and hardware residency; corpus-wide quality comparisons require the full benchmark datasets below.

The next benchmark implementation packet owns `benchmark.py`, its data-conversion tests and versioned reports under ignored `outputs/benchmarks/`. It must use the **same** profiles, preflight and injected adapter with upstream `index` and `rag_qa`; no copied retrieval algorithm. Deliver in this order:

1. Obtain all six corpus/query JSON files from the pinned revision. Record origin URLs, SHA-256 hashes, query IDs, query and corpus counts. Render corpus passages as exact `title + "\n" + text`. Mirror pinned `main.py` gold handling: HotpotQA joins supporting context sentences with an empty separator; 2Wiki uses a space; MuSiQue uses supporting paragraphs and answer aliases. Assert every gold passage exists in the corpus; fail rather than disabling retrieval evaluation.
2. Keep the full upstream corpus, including distractors. Never index only gold passages or only the documents belonging to selected questions. If a short development query subset is used, fix its IDs/seed, retain the full corpus, and label it as a subset. Full benchmark conclusions require the full 1000-question reproduction set for each dataset.
3. Verify actual token limits for formatted embedding inputs and chat prompts before the sweep. Do not silently truncate text, drop documents or let a larger server context setting override the model limit. An over-limit corpus blocks the unchanged benchmark; any deliberately modified chunking experiment needs a separate label and consistent gold mapping.
4. Fix dataset-specific upstream QA templates, rerank prompt file/revision, retrieval settings (`retrieval_top_k=200`, `qa_top_k=5`, `linking_top_k=5`), generation settings and extraction policy. Record resolved checkpoint/quantization, backend versions, device placement, input/output limits, index identity and raw per-query results. Reject malformed/truncated extraction or QA responses; retain failures and their denominator.
5. **Chat comparison:** hold one embedding profile fixed and rebuild extraction/graph state separately for Gemma, Qwen and Mistral. Measure their end-to-end pipeline results. A later QA-only ablation may freeze extraction state, but must be labelled separately.
6. **Embedding comparison:** hold extraction/chat settings and corpus fixed while rebuilding vectors/graph for the two Gemma weights and Qwen. Keep passage boundaries fixed for this comparison. Test Qwen's larger context separately so context/chunking benefits are not confused with an embedding-only difference.
7. Report retrieval recall@1/@2/@5 and relevant wider upstream cutoffs, answer EM and token F1, per-dataset paired score differences relative to Gemma chat, failure counts, cold indexing/query time, token usage and observed RAM/VRAM/device use. Preserve per-question results so the user can inspect where changes occur. Include uncertainty for aggregate differences where practical. No automatic degradation threshold or model promotion.
8. Run the full suite for final Mistral assessment after explicit installation and serving compatibility checks. Benchmark evidence supplements the daemon/MCP acceptance gates; it does not establish source deletion/restart correctness.

## Sources

- [Pinned HippoRAG benchmark entry point](https://github.com/OSU-NLP-Group/HippoRAG/blob/d5c8329422e0a0b834a15874545cb6a74b4f9b26/main.py) and [reproduction datasets](https://github.com/OSU-NLP-Group/HippoRAG/tree/d5c8329422e0a0b834a15874545cb6a74b4f9b26/reproduce/dataset).
- [FLM request settings](https://github.com/ROCm/FastFlowLM/blob/v1.0.6/src/server/rest_handler.cpp) (`think` is the supported field).
- [Qwen3-Embedding-0.6B model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) and [official GGUF serving instructions](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF).
- [EmbeddingGemma model card](https://huggingface.co/google/embeddinggemma-300m) and [QAT GGUF checkpoint](https://huggingface.co/ggml-org/embeddinggemma-300m-qat-q8_0-GGUF).

