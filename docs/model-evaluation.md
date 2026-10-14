# Local model compatibility and benchmark comparison

Decision: 2026-10-14. This document owns the selected model matrix and the comparison protocol. FOUNDATIONAL owns runtime scope; Epic 1 owns implementation acceptance. **Report differences; the user will decide what constitutes substantial degradation.** No automatic F1 cutoff selects a winner.

## Selected development models

| Role | Lemonade ID | Checkpoint | Model contract |
| --- | --- | --- | --- |
| Embedding (default) | `embeddinggemma-300m-GGUF` | `ggml-org/embeddinggemma-300M-GGUF:embeddinggemma-300M-Q8_0.gguf` | 768 dimensions, 2048 model tokens, mean pooling, GPU CUDA `llamacpp`; supports `search`, `qa`, `fact`, `distinct`, `titled`, `distinct-titled` task modes |
| Embedding (QAT variant) | `embeddinggemma-300m-qat-q8_0-GGUF-Q8_0` | `ggml-org/embeddinggemma-300m-qat-q8_0-GGUF:Q8_0` | 768 dimensions, 2048 model tokens, mean pooling; alternate QAT `Q8_0` weights (same 300M architecture, not a capacity step-up) |
| Embedding (step-up) | `Qwen3-Embedding-0.6B-GGUF` | `Qwen/Qwen3-Embedding-0.6B-GGUF:Qwen3-Embedding-0.6B-Q8_0.gguf` | 1024 dimensions, 32768 model tokens, last-token pooling |
| Extraction / rerank (`DEFAULT_EXTRACTION_CHAT`) | `Qwen3-4B-Instruct-2507-GGUF` | `unsloth/Qwen3-4B-Instruct-2507-GGUF:Qwen3-4B-Instruct-2507-Q4_K_M.gguf` | GPU CUDA `llamacpp` (`--parallel 1` without `-dio`, fits in VRAM alongside `embeddinggemma-300m-GGUF`); tightened OpenIE NER and triple JSON Schemas (`minLength: 1`) + dynamic candidate-literal `DSPyFilter` GBNF (`gbnf-v4:require-stop`) |
| QA reading (`DEFAULT_QA_CHAT`) | `Mistral-7B-Instruct-v0.3-Hybrid` | `amd/Mistral-7B-Instruct-v0.3-onnx-ryzenai-1.7-hybrid` | Ryzen AI Hybrid NPU/iGPU (`ryzenai-llm`, `fold-system:prompt-v4:require-stop`); folds leading `system` turn into first `user` turn, re-anchors `QA_TURN_REMINDER` before the final `\nQuestion: `, normalizes `\_`, and normalizes/validates `Answer:` delimiter |
| Chat / step-up | `gemma4-it-12b-FLM` | `gemma4-it:12b` | NPU step-up baseline (`flm`, `think=False`) |
| Chat / step-up | `qwen3.5-9b-FLM` | `qwen3.5:9b` | NPU step-up comparison (`flm`, `think=False`) |
| Extraction candidate | `Phi-4-mini-instruct-GGUF` | `unsloth/Phi-4-mini-instruct-GGUF:Phi-4-mini-instruct-Q4_K_M.gguf` | GPU CUDA `llamacpp`; passes with GBNF grammar enforcement, lower triple richness than Qwen3-4B |
| Extraction candidate | `Llama-3.2-3B-Instruct-GGUF` | `unsloth/Llama-3.2-3B-Instruct-GGUF:Llama-3.2-3B-Instruct-UD-Q4_K_XL.gguf` | GPU CUDA `llamacpp`; failed `DSPyFilter` seed selection on Cinderella (`graph_seeds: []`) |

All listed models are downloaded in Lemonade's local registry. In the default 3-slot development configuration (`max_models = 3`), `embeddinggemma-300m-GGUF` and `Qwen3-4B-Instruct-2507-GGUF` reside in discrete GPU VRAM while `Mistral-7B-Instruct-v0.3-Hybrid` runs on the Ryzen AI Hybrid NPU + Radeon 890M iGPU path. Avoid keeping multiple `llamacpp` chat models loaded simultaneously in GPU VRAM during latency comparisons. Long-input/boundary validation remains required.

EmbeddingGemma-2/multimodal work is **deferred**: no exact compatible checkpoint is selected, and this text-only adapter does not claim multimodal support.

## One adapter, explicit contracts

`yaw/embedding.py` defines the three immutable base profiles (`retrieval-prompts-v3`), `resolve_embedding_profile(model, task_mode)` for EmbeddingGemma task modes (`search`, `qa`, `fact`, `distinct`, `titled`, `distinct-titled`), and one `RetrievalEmbedding` adapter. Chat uses `yaw/chat.py`, a small policy subclass of upstream HippoRAG's `CacheOpenAI` client passed to `HippoRAG(extraction_llm=..., qa_llm=...)`:

- Recipe-specific request policies are explicit and isolated in `LocalChat`:
  - `flm` (`gemma4-it-12b-FLM`, `qwen3.5-9b-FLM`): sets `extra_body={"think": False}` to disable reasoning chains on structured extraction and QA calls.
  - `ryzenai-llm` (`Mistral-7B-Instruct-v0.3-Hybrid`): folds a leading `system` message into the first `user` message before caching/inference so Ryzen AI's Mistral `[INST]` template does not place `system` after few-shot demonstrations, injects a single corpus-agnostic `QA_TURN_REMINDER` immediately before the final `\nQuestion: ` in `messages[-1]` on QA calls, normalizes Markdown-escaped `\_` in JSON keys, and normalizes/validates the final `Answer:` delimiter on QA responses.
  - `llamacpp` (`Qwen3-4B-Instruct-2507-GGUF`, `Phi-4-mini-instruct-GGUF`, `Llama-3.2-3B-Instruct-GGUF`): enforces tightened copies of HippoRAG's `PROMPT_JSON_TEMPLATE["ner"]` and `PROMPT_JSON_TEMPLATE["triples"]` (`minLength: 1` on string items) via `response_format` on OpenIE prompts; enforces a per-call dynamic candidate-literal GBNF grammar (`_build_rerank_grammar(messages)`, falling back to `RERANK_GBNF_GRAMMAR`) on `DSPyFilter` recognition-memory prompts so `llamacpp` can only emit exact candidate triples from `fact_before_filter`; and enforces `QA_GBNF_GRAMMAR` (`thought "\nAnswer: " answer`) on QA prompts.
- Both Gemmas default to Google's retrieval query prefix (`task: search result | query: `) and document prefix (`title: none | text: `), and support `--embedding-task-mode distinct` (`query_instruction_mode = "distinct"`, routing `query_to_fact` to `task: fact checking | query: ` and `query_to_passage` to `task: question answering | query: `), `--embedding-task-mode titled` (`parse_document_title = True`, extracting `title: {title} | text: {body}` **only** when the first line matches `TITLE_LINE_RE` — i.e., a `#`..`###### ` Markdown heading or `Title:` / `Wikipedia Title:` label — while keeping `title: none | text: {text}` for untitled multi-line passages, single-line passages, entities and facts), and `--embedding-task-mode distinct-titled` (both). Qwen uses its instruction plus newline/Query separator, and unprefixed documents.
- Preserve prepared inputs, including Qwen's newline. The pinned upstream `encode` flattens newlines, so the shared adapter owns that small request/response boundary while inheriting SDK transport/retries, batching, usage aggregation and normalization.
- Reject unknown instructions, already prefixed inputs, wrong model/config pairs, wrong vector dimensions, invalid row indices/usage, nonfinite and zero vectors.
- `yaw/models.py` verifies downloaded registry identity, role and recipe for embedding, extraction chat and QA chat models before inference. It checks loaded backend health and explicit pooling, and adds the resolved local weight path to the profile fingerprint. Persisted index identity combines the resolved embedding identity and the `extraction_llm` policy identity (`chat_identity(extraction_model)`).
- Benchmark extraction uses one worker, zero automatic request retries, bounded document micro-batches (`--index-batch-size 8` by default) so OpenIE and graph state flush incrementally to SSD, and 1-by-1 staged retrieval/QA micro-batches (`--qa-batch-size 1` by default) with explicit pre-QA retrieval memory release. Both injected chat clients and the embedding client are closed by the caller.

## Commands available now

```powershell
uv run --locked python benchmark.py --dataset sample --mode all
uv run --locked python benchmark.py --dataset 2wikimultihopqa --max-queries 10
uv run --locked python benchmark.py --dataset musique --max-queries 5
uv run --locked python benchmark.py --dataset hotpotqa --max-queries 10
uv run --locked python benchmark.py --dataset sample --extraction-model Phi-4-mini-instruct-GGUF --qa-model Mistral-7B-Instruct-v0.3-Hybrid
uv run --locked python benchmark.py --dataset sample --chat-model gemma4-it-12b-FLM --embedding-model embeddinggemma-300m-GGUF
uv run --locked python benchmark.py --dataset sample --chat-model gemma4-it-12b-FLM --embedding-model Qwen3-Embedding-0.6B-GGUF
```

`--mode embeddings` exercises only embeddings, with five batch/single/reordered repeats per active role and cosine >= 0.999. `--mode eval` performs extraction, graph indexing, HippoRAG 2 retrieval/QA (`retrieve` + `qa`), and StandardRAG DPR baseline comparison (`retrieve_dpr` + `qa`) against the selected dataset and prints gold Recall@1/@2/@5 and QA ExactMatch/F1 alongside the HippoRAG 2 paper's reference numbers. `--extraction-model` and `--qa-model` default to `Qwen3-4B-Instruct-2507-GGUF` and `Mistral-7B-Instruct-v0.3-Hybrid`; passing `--chat-model` overrides both with a single model.

Lemonade settings for these installed models are CUDA and `--parallel 1`:
- `embeddinggemma-300m-GGUF` and `embeddinggemma-300m-qat-q8_0-GGUF-Q8_0`: `--pooling mean --parallel 1 --batch-size 2048 --ubatch-size 2048` and `ctx_size=2048` (`--ubatch-size 2048` is required so `llama-server` does not reject passages between 512 and 2048 tokens with HTTP 500).
- `Qwen3-Embedding-0.6B-GGUF`: `--pooling last --parallel 1 --batch-size 2048 --ubatch-size 2048` and `ctx_size=32768`.
- `Qwen3-4B-Instruct-2507-GGUF`: `--parallel 1 --cache-ram 0` (without `-dio`) and `ctx_size=16384`. Omitting `-dio` (`--direct-io`) avoids the Windows Direct-IO system RAM growth (~60 MB per constrained-decoding request) and `--cache-ram 0` disables `llama-server`'s 8,192 MiB host-RAM prompt cache so Private Committed Memory stays flat over hundreds or thousands of OpenIE requests while retaining single-slot GPU VRAM prefix caching.

Restart/reload an already loaded backend after a settings change; saving options alone need not change a running backend. Do not silently change its pooling in response to a failed test.

## Real benchmark harness (`benchmark.py`)

`benchmark.py` and `tests/test_benchmark.py` implement the upstream HippoRAG 2 benchmark harness against pinned revision `d5c8329422e0a0b834a15874545cb6a74b4f9b26` (`reproduce/dataset` corpora and query files for **`sample`, `2wikimultihopqa`, `musique`, and `hotpotqa`**):

1. Fetches and caches `{dataset}.json` and (only when `--full-corpus` or `sample` is selected) `{dataset}_corpus.json` from the pinned upstream commit under `outputs/datasets/`, slicing `samples[:max_queries]` immediately and rendering corpus passages as exact `f"{title}\n{text}"` while mirroring upstream `main.py` gold construction (`""` sentence join for `hotpotqa`, `" "` join for `2wikimultihopqa`, supporting paragraphs and answer aliases for `musique`/`sample`). Asserts every gold passage exists in the active candidate slice before indexing and releases the raw JSON structures immediately.
2. Supports Tier 0 (`--dataset sample`), Tier 1 query-pooled slices (`--max-queries N`, pooling all candidate passages across the selected queries in deterministic order without loading `{dataset}_corpus.json`), Tier 2 full-corpus runs (`--full-corpus`), and retrieval-only runs (`--mode retrieval`, skipping QA model checks and `hipporag.qa`).
3. Checkpoints OpenIE in bounded micro-batches (`--index-batch-size 8` by default via `hipporag.pre_openie(batch)`), then calls `hipporag.index(docs=pending)` once to encode unique entity/fact embeddings and write `graph.pickle` once (skipping redundant `index()` calls when `graph.pickle` already matches `vdb_chunk.parquet`) into deterministic persisted directories under `outputs/benchmarks/indexes/`. Executes `retrieve` and `retrieve_dpr` 1-by-1 (`--qa-batch-size 1` by default), releases in-memory retrieval matrices, and (when `--mode eval` or `--mode all` is selected) runs `hipporag.qa` 1-by-1 against the HippoRAG 2 paper's reference numbers (`Llama-3.3-70B-Instruct` + `NV-Embed-v2`).

See [evaluation](evaluation.md) and the local arXiv references ([HippoRAG 1](arxiv-2405.14831v3-hipporag.md), [HippoRAG 2](arxiv-2502.14802v1-hipporag2.md)) for measured Tier 0 and Tier 1 results.

## Sources

- [Pinned HippoRAG benchmark entry point](https://github.com/OSU-NLP-Group/HippoRAG/blob/d5c8329422e0a0b834a15874545cb6a74b4f9b26/main.py) and [reproduction datasets](https://github.com/OSU-NLP-Group/HippoRAG/tree/d5c8329422e0a0b834a15874545cb6a74b4f9b26/reproduce/dataset).
- Local reference papers: [HippoRAG 1 (`2405.14831v3`)](arxiv-2405.14831v3-hipporag.md) and [HippoRAG 2 (`2502.14802v1`)](arxiv-2502.14802v1-hipporag2.md).
- [FLM request settings](https://github.com/ROCm/FastFlowLM/blob/v1.0.6/src/server/rest_handler.cpp) (`think` is the supported field).
- [Qwen3-Embedding-0.6B model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) and [official GGUF serving instructions](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF).
- [EmbeddingGemma model card](https://huggingface.co/google/embeddinggemma-300m) and [QAT GGUF checkpoint](https://huggingface.co/ggml-org/embeddinggemma-300m-qat-q8_0-GGUF).

