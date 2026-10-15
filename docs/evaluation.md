# Evaluation record

Date: 2026-10-13. This file owns measured results and open failures. The [daemon](epic-1-hipporag-daemon.md) and [FastMCP](epic-2-fastmcp-http.md) epic checklists remain uncompleted until implementation evidence is added.

## Installed baseline

Actual checkout: `C:\Users\Harri.Ahokas\Documents\yaw`, origin `https://github.com/negatrait/yaw.git`. CPython 3.12.14; HippoRAG 2.0.0a5 from commit `d5c8329422e0a0b834a15874545cb6a74b4f9b26`; Lemonade API `http://localhost:13305/api/v1`.

The installed environment was checked with `importlib.util.find_spec`: NumPy exists; `mcp`, `fastmcp`, `pymilvus` and `milvus_lite` do not. The experiment's current manifest declares HippoRAG and NumPy. Missing optional packages do not disable the default Parquet/index/retrieval/QA path.

Earlier in this session, the one-sentence smoke run returned "Politician" in 5.9 seconds and extracted one triple. That was basic execution evidence. With a single passage, it could not establish useful ranking, correct embedding task prompts or endpoint repeatability. `live_smoke.py` has since replaced that script in the working tree.

## Nine-document diagnostic

Read and ran the user's current `live_smoke.py` without changing it, redirecting only its `save_dir` to a fresh temporary directory. INFO logging was enabled. The source fixture contains nine passages and three questions (George Rankin, Cinderella and Erik Hort's birthplace). Chat model: `Qwen3-4B-Instruct-2507-GGUF`. The script declares embedding alias `text-embedding-gemma` but its monkeypatch actually calls `embed-gemma-300m-FLM`.

Equivalent reproduction from the project environment:

```python
import logging, runpy, tempfile
logging.basicConfig(level=logging.INFO)
script = runpy.run_path("live_smoke.py", run_name="diagnostic")
script["main"].__globals__["save_dir"] = tempfile.mkdtemp(prefix="yaw-live-smoke-")
script["main"]()
```

Observed result: exit 0, elapsed 40.0 seconds inside the script. The run stored Parquet vectors, performed NER and triple extraction, built the graph, and ran separate retrieval/QA, combined QA and evaluation. Final graph info: 9 passage nodes, 15 phrase nodes, 24 total nodes, 10 extracted triples, 81 synonymy triples and 104 total triples. The synonym count is an observation, not a quality pass.

| Question | Combined QA result | Evidence result |
| --- | --- | --- |
| George Rankin's occupation | `Unknown` | Correct passage ranked sixth; `qa_top_k=5` kept it out of the actual QA prompt |
| Cinderella's happy ending | Matching slipper allowed reunion with the prince | Relevant Cinderella passages led the result list |
| County containing Erik Hort's birthplace | `Rockland County.` | Montebello/county and birthplace passages were the first two |

The log reported `No facts found after reranking, return DPR results` for two queries on a three-query pass. Thus indexing and QA executed, but some retrieval followed upstream's dense branch. A nonzero graph does not mean every query used graph traversal. Investigate selected facts/graph seeds before attributing a failure to PPR.

Aggregate retrieval: recall@1 0.2778, @2 0.5556, @5 0.6667, @10 1.0. Aggregate QA exact match 0.3333, F1 0.3636. These are diagnostic values from the current imperfect fixture. Recall@10 on nine passages is not useful discrimination. The Cinderella gold answer ("By going to the ball.") inadequately represents the supplied reunion evidence. The script evaluates a stricter two-passage path separately from the combined path; those are different evidence budgets.

## Embedding defects and unresolved serving behavior

Confirmed by pinned-source inspection:

- The OpenAI provider inherits `query_instruction_mode = "ignored"`. In `HippoRAG.get_query_embeddings`, that mode makes shared query embedding calls omit `instruction`.
- The script's monkeypatch uses the presence of two specific instructions to identify queries. Without the argument, it assigns `title: none | text:` to queries too. Changing only the prefix function is insufficient; an injected adapter must declare its instruction behavior.
- Replacing both upstream methods globally also bypasses upstream batching, row-order/usage checks and zero-vector validation. It makes the stored model identity disagree with the actual request model, and does not identify prompt-policy changes in the index. Reusing old vectors after changing that policy would make the test inconclusive.

Google's [EmbeddingGemma model card](https://ai.google.dev/gemma/docs/embeddinggemma/model_card#prompt-instructions) specifies different retrieval query/document prefixes. Their presence does not alone prove that this local server produces stable embeddings.

A direct HTTP probe, outside HippoRAG, sent these strings as one batch and individually with `model="embed-gemma-300m-FLM"` and `encoding_format="float"`:

```text
task: search result | query: What is George Rankin's occupation?
title: none | text: George Rankin is a politician.
title: none | text: Cinderella attended the royal ball.
title: none | text: Montebello is a part of Rockland County.
```

Every returned row had 768 dimensions and a matching row index. Cosine agreement for batch versus individual embeddings was respectively **0.4480, 0.9997, 0.9996, 1.0000**. In a follow-up of three byte-identical single-input query requests, pairwise cosines were **0.8153, 0.8153, 1.0000**. Vectors were finite and had norms around 0.997 before normalization.

This establishes material context/repetition variation in the observed endpoint outputs. It does not establish whether Lemonade routing, FLM runtime state, model packaging or another serving detail caused it. The service was not restarted, upgraded or replaced during diagnosis. D1-01 must isolate this before accepting an embedding baseline; no claim that a wrapper fix alone solves it.

## Milvus decision

Retain the pinned upstream `parquet` backend for both epics. The backend is functional, and neither MCP nor Milvus is required to run indexing/QA. Milvus changes vector persistence/search facilities; it does not supply query task prefixes, file ownership, graph consistency or daemon control.

In this HippoRAG revision, `prepare_retrieval_objects()` loads entity, passage and fact vectors into NumPy arrays, and fact/passage scoring uses dot products. Choosing the Milvus store does not itself replace these operations with server-side vector search or make the whole graph/index transaction atomic. See [retrieval code](https://github.com/OSU-NLP-Group/HippoRAG/blob/d5c8329422e0a0b834a15874545cb6a74b4f9b26/src/hipporag/HippoRAG.py) and [Milvus adapter](https://github.com/OSU-NLP-Group/HippoRAG/blob/d5c8329422e0a0b834a15874545cb6a74b4f9b26/src/hipporag/vector_stores/milvus_store.py).

Revisit Milvus after measuring passage/entity/fact counts, vector memory, load time and query latency on the 100–500-document corpus, or if external vector storage is required. Document count alone is not a vector-capacity estimate. A proposed adoption must demonstrate a benefit in this integration and pass source update/delete/restart tests.

Platform support requires version-specific checking: Milvus's operational FAQ describes older Windows limitations, while the current [Milvus Lite repository](https://github.com/milvus-io/milvus-lite) describes a newer Python implementation with conditional Windows support and incompatible older on-disk files. Neither a blanket "Windows cannot run it" nor an untested installation guarantee is justified. No Milvus package or server was installed for this planning task.

## FastMCP dependency check and source correction

An isolated uv project containing the pinned HippoRAG revision, NumPy, filelock and `fastmcp==4.1.0` resolved successfully on Python 3.12.14: 140 packages in the universal lock, including MCP SDK 2.3.0, filelock 4.1.1, Starlette 1.8.0 and Uvicorn 0.54.0. The repository manifest, lock and installed environment were not changed by that check. Installation/import/HTTP compatibility are still E2-01 work.

The pasted DeepWiki answer says there is no MCP code and implies automatic crash recovery from graph persistence. Both need qualification for our pin: `hipporag.mcp_server` already exists with a serialized service, five core tools, a console entry point and optional HTTP transport. Persistence spans multiple files, so a daemon must explicitly handle interrupted writes. The E1/E2 plans use the pinned source as the implementation reference.

## Repaired EmbeddingGemma smoke run (2026-10-13)

The failures above describe the original script/FLM configuration and are retained as history. The repaired path uses one explicit model ID: `user.embeddinggemma-300m-GGUF`, checkpoint `ggml-org/embeddinggemma-300M-GGUF:embeddinggemma-300M-Q8_0.gguf`. Lemonade downloaded revision `0f741b5a6585bd53aeb15cd1372c56f2a0f65e12`. Backend: llama.cpp CUDA, mean pooling, parallel 1; binary version 0.4.0-dev, build 10820, commit `74a7c897f`. Lemonade 2026.41.1. Qwen chat model remains unchanged. No Milvus/MCP package or new Python dependency was added.

FLM 1.0.6 was isolated by calling its backend directly at `http://127.0.0.1:8002/v1/embeddings`, bypassing Lemonade. Five batch/individual repetitions on three raw strings still differed materially: the George query showed cosine 0.55216 against the initial batch, and a George document individual call dropped to 0.82953. This localizes the observed instability to the FLM serving path, without identifying the native kernel cause. Its [v1.0.6 request handler](https://github.com/ROCm/FastFlowLM/blob/v1.0.6/src/server/rest_handler.cpp) also unconditionally passes `task_query` to the embedding engine. That engine adds the query prefix itself, wrapping even manually prefixed document requests. The smoke repair **does not repair FLM**: it explicitly replaces that serving path with llama.cpp while retaining EmbeddingGemma. The FLM installation was left intact and is never selected by the new script.

Implemented `yaw/embedding.py` with `query_instruction_mode="shared"`, distinct document/query prefixes, truthful model and versioned index identity, finite/nonzero 768-dimensional response validation and inherited upstream transport/batching/row ordering/usage/L2 normalization. HippoRAG and its separately owned injected client are closed explicitly. Every smoke invocation creates a new index, so no old FLM vectors can contaminate it. Configuration/embedding/retrieval/QA errors cause nonzero exit and a failed JSON report.

First successful fresh run: `outputs/live-smoke/run-smq0_7bv/report.json`, elapsed 19.70 seconds inside the script. Five repetitions per role covered batch, individual and reversed-batch contexts; the minimum pairwise cosine was **0.9999998808** for both roles, exceeding 0.999. All vectors were finite, nonzero and 768-dimensional. An offline test calls the real pinned `HippoRAG.get_query_embeddings` dispatcher and captures the actual query-prefix request, guarding against the original ignored-instruction bug.

| Question | Answer | Evidence supplied to QA / graph seeds |
| --- | --- | --- |
| George Rankin's occupation | `politician` | Correct passage rank 1; seed `(george rankin, is, a politician)` |
| Cinderella's happy ending | Slipper fit, allowing reunion with the prince | All three relevant passages ranks 1–3; reunion and ball seeds |
| Erik Hort's birthplace county | `Rockland County` | Birthplace and county passages ranks 1–2; both bridge facts selected |

Both separate and combined APIs passed with `qa_top_k=5`; the cutoff was not enlarged to conceal poor ranking. The second API pass reuses upstream caches, so it is API consistency evidence, not a second independent inference trial. All three queries used graph seeds; no `No facts found after reranking, return DPR results` message occurred. Graph: 9 passages, 15 phrase nodes, 10 extracted triples, 1 synonymy triple and 30 total edges. The earlier 81 synonymy triples fell to 1 without changing the upstream 0.8 similarity threshold.

Aggregate recall@1/@2/@5: **0.6111 / 0.8889 / 1.0**. QA exact match **0.6667**, F1 **0.8462**: the correct Cinderella paraphrase differs from the independently corrected gold wording. The fixture's answer-concept checks all pass; these are modest regression checks, not a general semantic evaluator. Upstream also emits recall values above the retrieved five documents; do not interpret these as additional quality evidence.

Verification: `uv run --locked python -m unittest discover -s tests -v` — **7 tests passed**, covering actual query dispatch, document prefixes, batching/order/usage/normalization, malformed vectors/indices/usage, invalid options and original QA-evidence failure. Run the live check with `uv run --locked python live_smoke.py`; exact model setup is in README. A small initial implementation attempt rejected the incorrect `BaseConfig.llm_model_name` keyword before inference; corrected to the pinned API's `llm_name`.

Final verification after adding the backend identity guard: `uv run --locked python live_smoke.py` exited 0, report `outputs/live-smoke/run-7wu1s5lz/report.json`, 19.53 seconds. All three answers, evidence rankings, graph seeds and repeatability checks passed again on a newly built index. `git diff --check` passed.

## Next result to record

D1-01 remains partial only for English/multilingual 1024-byte chunk-boundary validation against actual embedding/chat limits. The three original smoke defects are resolved on the explicitly selected EmbeddingGemma llama.cpp path. All daemon source-reconciliation/lifecycle and MCP runtime acceptance remains unrun. Preserve failed outcomes alongside successful runs.


## Explicit gold evaluation mode (2026-10-14)

Added `--mode eval` for a standalone gold evaluation and `--mode smoke` for separate/combined QA; the default `--mode all` runs all three. Evaluation calls upstream `rag_qa(queries=QUERIES, gold_docs=GOLD_DOCS, gold_answers=GOLD_ANSWERS)`. Console output and `report.json` now explicitly include gold references, per-question recall and answer EM/F1, plus aggregate metrics. Upstream's QA evaluators provide per-question scores; no replacement scoring algorithm or new dependency was added. Reported recall is limited to 1/2/5, within the five-passage retrieval window.

Preserved the user's chat-model selection `gemma4-it-12b-FLM` for the development NPU; EmbeddingGemma 300M Q8_0 remains on llama.cpp CUDA. This is independent chat/embedding selection, not an embedding fallback. The earlier FLM embedding failure does not exclude FLM chat.

Command: `uv run --locked python live_smoke.py --mode eval`. Exit 0, report `outputs/live-smoke/run-gx2_ci1o/report.json`, elapsed 405.57 seconds including model startup and embedding probes. Fresh index: 9 passages, 14 phrase nodes, 11 extracted triples, 0 synonymy triples and 29 total edges. All three queries had graph seeds; George's evidence ranked first and both birthplace passages ranked first/second. All evidence/answer smoke assertions passed.

| Question | Prediction | Exact match | Answer F1 |
| --- | --- | --- | --- |
| George Rankin | Politician | 1.0 | 1.0 |
| Cinderella | When the slipper fit perfectly, she was reunited with the prince. | 0.0 | 0.470588 |
| Erik Hort | Rockland County | 1.0 | 1.0 |

Aggregate recall@1/@2/@5: **0.6111 / 0.8889 / 1.0000**. Aggregate answer exact match **0.6667**, F1 **0.8235**. The Cinderella answer is supported by the evidence; its wording differs from the fixed gold, so lexical metrics are lower. Gold references were not edited to match predictions. These three fixture questions do not establish general model quality or multilingual/context-limit compatibility.

`uv run --locked python -m unittest discover -s tests -v`: **9 tests passed**, including exact gold argument forwarding, serialized references, per-query exact/partial/no-overlap scores and exclusion of recall cutoffs beyond retrieved evidence. `git diff --check` passed. Further EmbeddingGemma-2/Qwen embedding profiles require exact checkpoint selection and their own contract/live evaluation gates in Epic 1.


## Explicit model profiles (2026-10-14)

User selections are recorded in [model evaluation](model-evaluation.md). Implemented one `RetrievalEmbedding` adapter with three explicit profiles and independent chat/embedding CLI selectors. The local encode boundary preserves Qwen's required newline; inherited batching/SDK/normalization remains in use. Profile/config mismatches, incorrect dimensions/usage/indices, and unavailable or wrongly registered models fail explicitly. Index identity covers profile policy and the resolved local embedding weight path; every smoke invocation still builds fresh state.

Three installed embedding profiles passed five repeated batch/single/reordered probes per role (768 dimensions for Gemma variants; 1024 for Qwen):

| Model | Report | Minimum pairwise cosine (query / document) |
| --- | --- | --- |
| `Qwen3-Embedding-0.6B-GGUF` | `outputs/live-smoke/run-g4wo6zkg/report.json` | 0.999449134 / 0.999739528 |
| `embeddinggemma-300m-qat-q8_0-GGUF-Q8_0` | `outputs/live-smoke/run-mtu79fc_/report.json` | 0.999999881 / 0.999999821 |
| `embeddinggemma-300m-GGUF` | `outputs/live-smoke/run-0zbeiaw6/report.json` | 0.999999881 / 0.999999881 |

Qwen's explicit last-token pooling/32768 context setting and QAT Gemma's mean-pooling/2048 setting were saved in Lemonade. Actual full-context inference has not been certified by these short probes. Fifteen offline tests passed at that stage, including Qwen newline preservation, profile-specific dimensions, distinct index identities and rejection of undownloaded Mistral. The selected benchmark policy is to report differences and let the user decide afterward, without an automatic quality cutoff.

## Split extraction/QA architecture, GBNF grammar enforcement, and task modes (2026-10-14)

Extended `yaw/models.py`, `yaw/embedding.py`, `yaw/chat.py`, and `live_smoke.py` to support HippoRAG's native `HippoRAG(global_config=..., embedding_model=..., extraction_llm=..., qa_llm=..., index_identity=...)` split without modifying upstream HippoRAG source code:

1. **Three-slot heterogeneous hardware split in Lemonade (`max_models = 3`):**
   - `embedding_model`: `embeddinggemma-300m-GGUF` on discrete GPU 0 (`llamacpp` CUDA, port 8002, ~0.31 GB VRAM).
   - `extraction_llm` (`DEFAULT_EXTRACTION_CHAT`): `Qwen3-4B-Instruct-2507-GGUF` on discrete GPU 0 (`llamacpp` CUDA, port 8003, 32768 context, 6.4 / 8.0 GB combined VRAM with EmbeddingGemma) for high-frequency OpenIE NER, triple extraction, and `DSPyFilter` recognition memory.
   - `qa_llm` (`DEFAULT_QA_CHAT`): `Mistral-7B-Instruct-v0.3-Hybrid` on Ryzen AI Hybrid NPU 0 + Radeon 890M iGPU 1 (`ryzenai-llm`, port 8001, 32786 context, ~25.2 / 31.1 GB total system RAM at 81% peak utilization) for final QA reading comprehension.
2. **Diagnosed failures and permanent adapter guards (`yaw/chat.py`, `live_smoke.py`):**
   - `outputs/live-smoke/run-ybzw0eur/report.json` (failed): Unmodified `Mistral-7B-Instruct-v0.3-Hybrid` failed 9/9 NER chunks (`RuntimeError: NER failed for 9 chunk(s)`) because the Ryzen AI ONNX `chat_template.jinja` injects `system` into the final `[INST]` turn *after* all one-shot demonstrations, overriding the `{"named_entities": [...]}` format example. Fixed in `LocalChat.infer` by folding a leading `system` prompt into the first `user` turn for `recipe == "ryzenai-llm"`.
   - `outputs/live-smoke/run-2xn9m4fr/report.json` (failed): `Mistral-7B-Instruct-v0.3-Hybrid` as single-model extraction/QA failed 1/9 NER chunks (`chunk-d6df73e3b8e71d69e39075792fb855cf`) because Mistral v0.3 occasionally emits Markdown-escaped underscores (`{"named\_entities": [...]}`). Fixed by unescaping `r"\_"` to `"_"` on `ryzenai-llm` outputs; verified passing in `outputs/live-smoke/run-fe_mufil/report.json` (103.99s).
   - `outputs/live-smoke/run-99yf7wab/report.json` (failed): `Phi-4-mini-instruct-GGUF` with OpenIE JSON Schema (`PROMPT_JSON_TEMPLATE["ner"]` and `PROMPT_JSON_TEMPLATE["triples"]`) passed OpenIE but emitted a trailing double brace (`{"fact": [["cinderella", "was reunited with", "the prince"]]}}`) during `DSPyFilter` recognition memory on Question 2. Upstream `DSPyFilter` caught the parse error and returned `[]`, causing `HippoRAG.retrieve` to silently fall back to DPR (`graph_seeds: []`). Caught immediately by the new `empty graph_seeds` guard in `live_smoke.check_results`. Fixed in `LocalChat.infer` by enforcing `RERANK_GBNF_GRAMMAR` via `extra_body={"grammar": ...}` whenever `recipe == "llamacpp"` and the system prompt matches `DSPyFilter`.
   - `outputs/live-smoke/run-xcb7d980/report.json` (failed): `Llama-3.2-3B-Instruct-GGUF` passed OpenIE (18 triples) and GBNF syntax, but selected zero facts (`{"fact": []}`) during `DSPyFilter` recognition memory on Question 2 (Cinderella), failing the `empty graph_seeds` guard. Retained in `CHAT_MODELS` for diagnostic comparison, not selected as default.
   - `outputs/live-smoke/run-osf9ddda/report.json` (failed): Intermediate experiment overriding `ner_system` and `ner_conditioned_re_system` while instructing `Mistral-7B` not to start answers with `"By "`. Inspecting `Mistral-7B`'s SQLite cache showed it wrote the full synthesis in `Thought:` (`"Cinderella reached her happy ending when she was reunited with the prince after he found her using the lost glass slipper."`) but compressed `Answer:` to only the second clause (`"She was reunited with the prince."`), omitting `"slipper"` and failing `check_results`. Kept upstream OpenIE system prompts aligned with the `Radio City` one-shot demonstrations and updated `TIGHTENED_QA_SYSTEM` to require stating both the condition/means and the resulting outcome on `"How"`/`"Why"` questions.
   - **Dynamic entity-anchored triple GBNF, dynamic candidate-literal rerank GBNF & QA delimiter contract (`gbnf-v3:require-stop`, `prompt-v3:require-stop`):**
     - **Stage 1 (`ner`):** Enforces tightened `PROMPT_JSON_TEMPLATE["ner"]` (`minLength: 1` on string items).
     - **Stage 2 (`triple_extraction`):** `_build_triple_grammar(messages)` parses Stage 1's `{"named_entities": [...]}` from the user prompt and enforces `triple ::= "[" entity ", " string ", " string "]" | "[" string ", " string ", " entity "]"` via GBNF so every extracted triple anchors at least its subject or object to an exact Stage 1 entity literal (or emits `{"triples": []}` when `named_entities == []`). This pruned noisy unanchored/redundant triples, reducing `Qwen3-4B` extraction completion tokens from `536` to `452` and graph triples from `12` to `10` (removing the abstract non-entity node `"a tool for searching"`).
     - **Stage 3 (`DSPyFilter` recognition memory):** `_build_rerank_grammar(messages)` parses `[[ ## fact_before_filter ## ]]` and restricts `triple ::=` to exact JSON literals from the candidate list (up to $\min(4, N)$ selections, `{"fact": []}` when $N=0$, or static `RERANK_GBNF_GRAMMAR` fallback if malformed), preventing `difflib.get_close_matches(..., cutoff=0.0)` from matching paraphrased/hallucinated triples.
     - **Stage 4 (`rag_qa_musique`):** `TIGHTENED_QA_SYSTEM` instructs the reader to state only the exact entity/fact for `Who`/`What`/`When`/`Where` questions and both the key condition/means and resulting outcome for `How`/`Why` questions. `_normalize_qa_response` normalizes `**Answer:**` / `Answer :`, neutralizes any earlier `"Answer:"` occurrences inside `Thought:`, and fails fast if `"Answer:"` is omitted; `QA_GBNF_GRAMMAR` enforces `thought "\nAnswer: " answer` when a `llamacpp` model is used for QA.
3. **EmbeddingGemma task modes (`yaw/embedding.py`):**
   - Added `resolve_embedding_profile(model, task_mode)` supporting `"search"`, `"qa"`, `"fact"`, `"distinct"` (`query_instruction_mode = "distinct"`, dispatching `task: fact checking | query: ` for `query_to_fact` and `task: question answering | query: ` for `query_to_passage`), `"titled"` (`parse_document_title = True`, formatting multi-line `f"{title}\n{body}"` passages as `title: {title} | text: {body}` while keeping `title: none | text: ` for single-line passages, entity nodes and fact tuples), and `"distinct-titled"` (combining `distinct` and `titled`).
4. **Measured live smoke runs (`uv run --locked python live_smoke.py ...`):**

| Run report | Extraction LLM | QA LLM | Embedding / Task mode | Status | Elapsed (s) | Graph (phrase / triples / synonymy) | Graph seeds (Q1 / Q2 / Q3) | Recall@1 / @2 / @5 (Graph & DPR) | QA EM / F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `outputs/live-smoke/run-gudguenl/report.json` (`gbnf-v3`) | `Qwen3-4B-Instruct-2507-GGUF` | `Mistral-7B-Instruct-v0.3-Hybrid` | `embeddinggemma-300m-GGUF` (`search`) | passed | **29.72** | 15 / 10 / 1 | 1 / 2 / 2 | 0.6111 / 0.8889 / 1.0 | 0.6667 / **0.8444** |
| `outputs/live-smoke/run-ewmwi3fx/report.json` (`gbnf-v3`) | `Qwen3-4B-Instruct-2507-GGUF` | `Mistral-7B-Instruct-v0.3-Hybrid` | `embeddinggemma-300m-GGUF` (`distinct-titled`) | passed | **29.95** | 15 / 10 / 1 | 1 / 2 / 2 | 0.6111 / 0.8889 / 1.0 | 0.6667 / **0.8444** |
| `outputs/live-smoke/run-zo40rpc7/report.json` (`gbnf-v2`) | `Qwen3-4B-Instruct-2507-GGUF` | `Mistral-7B-Instruct-v0.3-Hybrid` | `embeddinggemma-300m-GGUF` (`search`) | passed | **35.14** | 16 / 12 / 1 | 1 / 2 / 2 | 0.6111 / 0.8889 / 1.0 | 0.6667 / 0.7843 |
| `outputs/live-smoke/run-1jqzrvln/report.json` (`gbnf-v2`) | `Qwen3-4B-Instruct-2507-GGUF` | `Mistral-7B-Instruct-v0.3-Hybrid` | `embeddinggemma-300m-GGUF` (`distinct`) | passed | **38.06** | 16 / 12 / 1 | 1 / 2 / 2 | 0.6111 / 0.8889 / 1.0 | 0.6667 / 0.7843 |
| `outputs/live-smoke/run-haz6loa6/report.json` (`gbnf-v2`) | `Qwen3-4B-Instruct-2507-GGUF` | `Mistral-7B-Instruct-v0.3-Hybrid` | `embeddinggemma-300m-GGUF` (`distinct-titled`) | passed | **37.03** | 16 / 12 / 1 | 1 / 2 / 2 | 0.6111 / 0.8889 / 1.0 | 0.6667 / 0.7843 |
| `outputs/live-smoke/run-t38n9k7c/report.json` (`gbnf-v1`) | `Qwen3-4B-Instruct-2507-GGUF` | `Mistral-7B-Instruct-v0.3-Hybrid` | `embeddinggemma-300m-GGUF` (`search`) | passed | **32.03** | 16 / 12 / 1 | 1 / 2 / 2 | 0.6111 / 0.8889 / 1.0 | 0.6667 / 0.7843 |
| `outputs/live-smoke/run-1hr11rjw/report.json` (`gbnf-v1`) | `Qwen3-4B-Instruct-2507-GGUF` | `Mistral-7B-Instruct-v0.3-Hybrid` | `embeddinggemma-300m-GGUF` (`distinct`) | passed | **31.84** | 16 / 12 / 1 | 1 / 2 / 2 | 0.6111 / 0.8889 / 1.0 | 0.6667 / 0.7843 |
| `outputs/live-smoke/run-90vdspsx/report.json` (`--mode all`) | `Qwen3-4B-Instruct-2507-GGUF` | `Mistral-7B-Instruct-v0.3-Hybrid` | `embeddinggemma-300m-GGUF` (`search`) | passed | **30.24** | 16 / 12 / 1 | 1 / 2 / 2 | 0.6111 / 0.8889 / 1.0 | 0.6667 / 0.7843 |
| `outputs/live-smoke/run-di4xs63q/report.json` | `Phi-4-mini-instruct-GGUF` | `Mistral-7B-Instruct-v0.3-Hybrid` | `embeddinggemma-300m-GGUF` (`search`) | passed | **29.52** | 14 / 10 / 0 | 1 / 1 / 2 | 0.6111 / 0.8889 / 1.0 | 0.6667 / 0.8333 |
| `outputs/live-smoke/run-il91m74y/report.json` | `Phi-4-mini-instruct-GGUF` | `Mistral-7B-Instruct-v0.3-Hybrid` | `embeddinggemma-300m-GGUF` (`distinct`) | passed | **29.37** | 14 / 10 / 0 | 1 / 1 / 2 | 0.6111 / 0.8889 / 1.0 | 0.6667 / 0.8333 |
| `outputs/live-smoke/run-fe_mufil/report.json` | `Mistral-7B-Instruct-v0.3-Hybrid` | `Mistral-7B-Instruct-v0.3-Hybrid` | `embeddinggemma-300m-GGUF` (`search`) | passed | **103.99** | 17 / 12 / 1 | 1 / 1 / 1 | 0.6111 / 0.8889 / 1.0 | 0.6667 / 0.8333 |
| `outputs/live-smoke/run-5f63m_9o/report.json` | `gemma4-it-12b-FLM` | `gemma4-it-12b-FLM` | `embeddinggemma-300m-GGUF` (`search`) | passed | **118.41** | 14 / 11 / 0 | 1 / 2 / 2 | 0.6111 / 0.8889 / 1.0 | 0.6667 / 0.8235 |
| `outputs/live-smoke/run-qdb6y78m/report.json` | `gemma4-it-12b-FLM` | `gemma4-it-12b-FLM` | `Qwen3-Embedding-0.6B-GGUF` (`search`) | passed | **161.36** | 14 / 11 / 2 | 1 / 2 / 2 | 0.6111 / 0.8889 / 1.0 | 0.6667 / 0.8235 |
| `outputs/live-smoke/run-170u4419/report.json` | `qwen3.5-9b-FLM` | `qwen3.5-9b-FLM` | `embeddinggemma-300m-GGUF` (`search`) | passed | **244.40** | 15 / 11 / 1 | 1 / 2 / 2 | 0.6111 / 0.8889 / 1.0 | 0.6667 / 0.8571 |

Offline test suite: `uv run --locked python -m unittest discover -s tests -v` — **26 tests passed**.

## Tier 0 & Tier 1 Upstream HippoRAG Benchmarks (`sample`, `2wikimultihopqa`, `hotpotqa`, `musique`), Serving Memory Fixes (`-dio`, `--cache-ram 0`), and Two-Stage Micro-Batch Indexing (2026-10-14)

Retired `live_smoke.py` and `tests/test_live_smoke.py` in favor of `benchmark.py` and `tests/test_benchmark.py` targeting the upstream HippoRAG benchmark datasets (`sample`, `2wikimultihopqa`, `musique`, `hotpotqa`) with support for `--mode retrieval` (retrieval-only evaluation without loading or invoking the QA model), `--mode eval` (retrieval + QA), `--mode embeddings`, and `--mode all`:

1. **Diagnosed benchmark failures, serving bottlenecks, and permanent fixes:**
   - `outputs/benchmarks/run-2wikimultihopqa-j5vxaome/report.json` (**failed**): `embeddinggemma-300m-GGUF` returned HTTP 500 (`input is larger than the max_batch_size (564 > 512 tokens)`) on passage 92 of `2wikimultihopqa`. Fixed by reloading `embeddinggemma-300m-GGUF` in Lemonade with `ctx_size = 2048`, `llamacpp_args = "--batch-size 2048 --ubatch-size 2048"`, `save_options = true`.
   - `outputs/benchmarks/run-2wikimultihopqa-rhqxraw7/report.json` and `outputs/benchmarks/run-2wikimultihopqa-4534o27b/report.json` (**failed**): `Mistral-7B-Instruct-v0.3-Hybrid` on `ryzenai-llm` omitted the literal `"Answer:"` delimiter on Q1 and Q9 of `2wikimultihopqa` (`The birth date of Vsevolod Nestayko is January 30, 1930.` and `Both films do not have the directors from the same country.`). Fixed in `yaw/chat.py` via `_normalize_qa_response(..., allow_missing_delimiter=(recipe != "llamacpp"))`, keeping strict delimiter enforcement (`QA_GBNF_GRAMMAR`) on `llamacpp` models while permitting `ryzenai-llm` direct answers to pass through `rag_qa_musique`'s `len(response_content) == 1` fallback.
   - `outputs/benchmarks/run-musique-tljkd0zf/report.json` (**failed**): 4 of 99 `musique` passages (`chunk-9d8c93e7faec40008389cf8a7905dc77`, `chunk-816fa9eb9be5654e9211bb359d65d78c`, `chunk-bd342bdba240413ab5a72276a496f49b`, `chunk-b81ca5f9174fbbe75c828d18a97ba13b`) hit `finish_reason: "length"` (4,096 tokens) during Stage 2 triple extraction because `_build_triple_grammar` forced every triple to contain an exact Stage 1 NER literal as subject or object. When a sentence had a non-NER subject (`"The cover"`) and descriptive non-NER object (`"floating dandelion seeds"`), the grammar forced `Qwen3-4B` to emit a Stage 1 NER literal (`"11th novel"`) as the object, corrupting its generation context and triggering a 4,096-token repetition loop. Fixed in `yaw/chat.py` (`gbnf-v3:require-stop`) by replacing `_build_triple_grammar` with the corpus-agnostic tightened JSON Schema (`OPENIE_JSON_SCHEMAS[ner_conditioned_re_system]`, `minItems: 3, maxItems: 3, minLength: 1`) while retaining dynamic candidate-literal GBNF (`_build_rerank_grammar`) for Stage 3 `DSPyFilter` recognition memory.
   - `outputs/benchmarks/run-musique-z63c9iae/report.json` (**failed**): Interrupted test run before fixing `force_index_from_scratch=not any(index_dir.rglob("graph.pickle"))` in `benchmark.py`, where a directory containing only `vdb_chunk.parquet` and `llm_cache/` from an interrupted run skipped indexing without creating `graph.pickle`.
   - **System RAM exhaustion and overnight memory-spill elimination:**
     1. **`llama-server` Direct-IO (`-dio`) CPU heap growth:** `Qwen3-4B-Instruct-2507-GGUF` had been loaded with `llamacpp_args = "-dio --parallel 1"`. Under `llama-server` build 10820 on Windows, `-dio` (`--direct-io`) leaked ~59 MB of CPU heap in System RAM per constrained-decoding request.
     2. **`llama-server` `--cache-ram` 8,192 MiB host RAM prompt cache:** By default, `llama-server` sets `--cache-ram 8192` (`8 GiB`) and `--cache-idle-slots`, archiving every completed request's KV state (~72 MB per unique passage across NER + RE for `Qwen3-4B`) into host RAM until `llama-server.exe` grew from `~5.5 GB` to `13,641.4 MB` Private Committed Memory (`4,293.5 MB` Working Set). Reloading `Qwen3-4B-Instruct-2507-GGUF` with `llamacpp_args = "--parallel 1 --cache-ram 0"`, `ctx_size = 16384`, `save_options = true` (and enforcing `--cache-ram 0` and absence of `-dio` in `yaw/models.py` `chat_evidence()`) eliminated host RAM growth completely: across the 10-query `musique` run (`197` passages, `404` chat requests, `271,548` total tokens), `llama-server.exe` stayed flat at `2,811.3 MB -> 2,859.9 MB` WS and `5,553.2 MB -> 5,562.4 MB` Private Committed (**`+9.2 MB` total delta across 404 requests; `0.0%` drift**) while still reusing `131,118` cached prefix tokens in GPU VRAM at `3,119.5 prompt tok/s` and `78.5 gen tok/s`.
     3. **Two-stage micro-batch indexing (`pre_openie` per 8 docs + single final `index`) and PyTorch CUDA cache release:** Calling `hipporag.index(batch)` in a loop re-reads/rewrites `vdb_entity.parquet` and `vdb_fact.parquet` via Pandas, deep-copies all entity rows, runs PyTorch CUDA `retrieve_knn`, and rewrites `graph.pickle` on every 8-doc batch ($O(N^2)$ over full corpora). Updated `index_documents_in_batches(hipporag, docs, batch_size=8)` to call `hipporag.pre_openie(batch)` every 8 documents (checkpointing `vdb_chunk.parquet` and `openie_state.json` to SSD with `1%` NVMe utilization), and then call `hipporag.index(docs=pending)` **once** at the end to encode unique entity/fact embeddings and write `graph.pickle` once, followed by `_release_indexing_memory` (which also calls `torch.cuda.empty_cache()`).
   - **Explicit heading/title extraction (`yaw/embedding.py`, `retrieval-prompts-v3`), end-of-turn `ryzenai-llm` QA reminder (`yaw/chat.py`, `fold-system:prompt-v4:require-stop`), lazy corpus loading, and 1-by-1 retrieval/QA processing (`benchmark.py`):**
     1. **`TITLE_LINE_RE` (`retrieval-prompts-v3`):** Rather than assuming line 1 of every multi-line passage is a document title (which matched the `f"{title}\n{text}"` layout of academic benchmarks), `parse_document_title` now extracts `title: {title} | text: {body}` **only** when the first line explicitly starts with a Markdown ATX heading (`#`..`###### `) or an explicit `Title:` / `Wikipedia Title:` label (`r"^(?:#{1,6}\s+|(?:wikipedia\s+)?title:\s*)(.+)$"`). Untitled multi-line chunks and raw `f"{title}\n{text}"` benchmark passages use `title: none | text: {text}`, while `distinct-titled` still dispatches EmbeddingGemma's distinct `task: fact checking | query: ` (`query_to_fact`) and `task: question answering | query: ` (`query_to_passage`) instructions.
     2. **`QA_TURN_REMINDER` (`fold-system:prompt-v4:require-stop`):** Because `ryzenai-llm` folds the `system` message into `messages[0]` (the first few-shot turn, ~2,500 tokens before the target question), `LocalChat.infer` appends a single corpus-agnostic reminder (`"Based on the passages above, reason step by step and conclude on a new line with 'Answer: ' followed only by the concise final answer."`) immediately before the final `\nQuestion: ` in `messages[-1]` on `ryzenai-llm` QA calls.
     3. **Lazy corpus loading & 1-by-1 staged retrieval/QA (`--qa-batch-size 1`, `--mode retrieval`):** `load_dataset_files` skips downloading/parsing `{dataset}_corpus.json` on query-pooled slices (`include_corpus=False`) and slices `samples[:max_queries]` immediately; `main()` frees `samples`, `corpus`, and `docs` as soon as indexing finishes; and `evaluate_benchmark` runs `retrieve` and `retrieve_dpr` 1-by-1 (`num_to_retrieve=5` instead of `200`, clearing `query_to_embedding` after each query), frees in-memory retrieval matrices (`_release_retrieval_memory`) after retrieval, and skips `hipporag.qa` entirely when `--mode retrieval` (`include_qa=False`) is selected.
2. **Measured Tier 0 & Tier 1 benchmark results (`uv run --locked python benchmark.py ...`):**
   - Active stack: `Qwen3-4B-Instruct-2507-GGUF` (extraction, `ctx_size = 16384`, `--parallel 1 --cache-ram 0`, `extraction_temperature = 0.2`, `frequency_penalty = 0.5`, `gbnf-v3:require-stop`) + `Mistral-7B-Instruct-v0.3-Hybrid` (QA, NPU/iGPU Hybrid, `qa_temperature = 0.2`, `fold-system:prompt-v4:require-stop`) + `embeddinggemma-300m-GGUF` (`distinct-titled`, `retrieval-prompts-v3`, `ctx_size = 2048`, `--batch-size 2048 --ubatch-size 2048`).
   - In greedy mode (`temperature = 0`), both `Qwen3-4B` and `Mistral-7B` are prone to autoregressive repetition loops on repetitive phrases; setting a permanent floor of `temperature = 0.2` in `ChatProfile` and `yaw/chat.py` (plus `frequency_penalty = 0.5` on OpenIE triple extraction) eliminated repetition loops completely, reducing QA reading latency from 30–60s loops to 4–8s concise responses.
   - Across all three 10-query Tier 1 datasets (`387` passages, `30` queries with full retrieval and Mistral QA):

| Dataset (`N` queries / `P` docs) | Run report | Graph (phrases / triples / synonymy / edges) | Method | Recall@1 | Recall@2 | Recall@5 | QA EM | QA F1 | Queries w/ seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Tier 0: `sample`** (`1 / 3`, batch=8) | `outputs/benchmarks/run-sample-hgj5zm7g/report.json` | 30 / 29 / 4 / 72 | **HippoRAG 2** | **1.0000** | **1.0000** | **1.0000** | 0.0000 | **0.6667** | 1 / 1 |
| **Tier 0: `sample`** (`1 / 3`, batch=2) | `outputs/benchmarks/run-sample-ehdci6mn/report.json` | 29 / 28 / 4 / 70 | **HippoRAG 2** | 0.5000 | **1.0000** | **1.0000** | 0.0000 | **0.6667** | 1 / 1 |
| **Tier 1: `2wikimultihopqa`** (`10 / 96`, `temp=0.2`) | `outputs/benchmarks/run-2wikimultihopqa-fbmnosfw/report.json` | 866 / 905 / 156 / 1,949 | **HippoRAG 2 (`rag_qa`)** | **0.4250** | **0.8000** | **1.0000** | 0.3000 | **0.5278** | 8 / 10 |
| **Tier 1: `2wikimultihopqa`** (`10 / 96`, `temp=0.2`) | `outputs/benchmarks/run-2wikimultihopqa-fbmnosfw/report.json` | — | Standard DPR (`rag_qa_dpr`) | 0.4250 | 0.7250 | 0.8500 | **0.4000** | 0.4500 | — |
| *Paper ref: `2wikimultihopqa`* (`1,000 / 6,119`) | *`Llama-3.3-70B` + `NV-Embed-v2` (8B)* | — | *HippoRAG 2 / StandardRAG* | — | *0.762 / 0.671* | *0.904 / 0.765* | *0.650 / 0.575* | *0.710 / 0.615* | — |
| **Tier 1: `hotpotqa`** (`10 / 94`, `temp=0.2`) | `outputs/benchmarks/run-hotpotqa-zbp9k8e0/report.json` | 1,206 / 1,286 / 275 / 2,819 | **HippoRAG 2 (`rag_qa`)** | **0.5000** | **0.8500** | **1.0000** | **0.3000** | **0.5483** | 10 / 10 |
| **Tier 1: `hotpotqa`** (`10 / 94`, `temp=0.2`) | `outputs/benchmarks/run-hotpotqa-zbp9k8e0/report.json` | — | Standard DPR (`rag_qa_dpr`) | 0.4500 | 0.7000 | 0.9500 | **0.3000** | 0.4524 | — |
| *Paper ref: `hotpotqa`* (`1,000 / 9,221`) | *`Llama-3.3-70B` + `NV-Embed-v2` (8B)* | — | *HippoRAG 2 / StandardRAG* | — | *0.835 / 0.841* | *0.963 / 0.945* | *0.627 / 0.628* | *0.755 / 0.753* | — |
| **Tier 1: `musique`** (`10 / 197`, `temp=0.2`) | `outputs/benchmarks/run-musique-_5f_uroh/report.json` | 2,268 / 2,346 / 714 / 5,442 | **HippoRAG 2 (`rag_qa`)** | 0.1833 | **0.3167** | **0.5417** | 0.2000 | 0.3000 | 7 / 10 |
| **Tier 1: `musique`** (`10 / 197`, `temp=0.2`) | `outputs/benchmarks/run-musique-_5f_uroh/report.json` | — | Standard DPR (`rag_qa_dpr`) | **0.2083** | 0.2833 | 0.5167 | **0.3000** | **0.4000** | — |
| *Paper ref: `musique`* (`1,000 / 11,656`) | *`Llama-3.3-70B` + `NV-Embed-v2` (8B)* | — | *HippoRAG 2 / StandardRAG* | — | *0.561 / 0.527* | *0.747 / 0.697* | *0.372 / 0.347* | *0.486 / 0.457* | — |

Offline test suite: `uv run --locked python -m unittest discover -s tests -v` — **32 tests passed**.

## Epic 1 Sprint 1 (`D1-A1`) & Sprint 2 (`D1-A2`): Domain Containment Tree, Native DSPy (`dspy.LM`, `dspy.Embedder`, `dspy.RLM`, `dspy.GEPA`), and Zero `hipporag` Runtime Imports (2026-10-15)

1. **Sprint 1 (`E1-S1`, `D1-A1`):**
   - Established the permanent domain-oriented containment tree (`yaw/{core,daemon/{ingestion,graph/{storage,inference}}}/`) with frozen, zero-dependency contracts (`ChunkPointer`, `HydratedPassage`, `OpenIERecord`, `RetrievalResult`, `SyncDelta`) in `yaw/core/types.py` and the 500-line FIFO logger in `yaw/core/logger.py`.
   - Enforced zero horizontal/cross-domain imports, empty `__init__.py` files, stdlib-only `yaw/core/*`, and zero `hipporag` runtime imports inside `yaw/**` via AST inspection in `tests/core/test_architecture.py`.
2. **Sprint 2 (`E1-S2`, `D1-A2`):**
   - Added `dspy>=3.0.0` (`dspy==3.4.1`, `gepa==0.1.4`) to `pyproject.toml` and synced `uv.lock`.
   - Eliminated all `hipporag` runtime imports from `yaw/daemon/graph/inference/{models.py, embedding.py, chat.py, prompts.py}` while keeping `RetrievalEmbedding` and `LocalChat` 100% compatible with `benchmark.py`.
   - Added `create_dspy_embedder` and `DSPyEmbeddingAdapter` (`yaw/daemon/graph/inference/embedding.py`) wrapping `dspy.Embedder` for Lemonade (`openai/...`, `encoding_format="float"`) and Ollama (`ollama/...`) with `distinct-titled` formatting and L2 normalization.
   - Added `create_dspy_lm` (`yaw/daemon/graph/inference/chat.py`) wrapping `dspy.LM` for Lemonade (`openai/...`) and Ollama (`ollama_chat/...`), configured for our 3-slot stack (`Qwen3-4B-Instruct-2507-GGUF` on CUDA `llamacpp` with `8192–16384` context window, `temp=0.2`, `freq_penalty=0.5`, and `Mistral-7B-Instruct-v0.3-Hybrid` on Ryzen AI `ryzenai-llm` with `32786` context window).
   - Implemented the 4 typed `dspy.Signature`s (`ExtractEntities`, `ExtractTriples`, `FilterFacts` with `FactList`, `AnswerQuestion`), seeded `dspy.Example` demos (`NER_DEMOS`, `TRIPLE_DEMOS`, `FACT_FILTER_DEMOS`, `QA_DEMOS`), and `HippoRAGProgram(dspy.Module)` (`yaw/daemon/graph/inference/prompts.py`) supporting `dspy.Predict`, `dspy.ChainOfThought`, `dspy.RLM(..., interpreter_factory=dspy.LocalInterpreter, sub_lm=...)`, and `.save()` / `.load()` for `dspy.GEPA`-compiled state.
   - Verification: `uv run --locked python -m unittest discover -s tests -v` — **67 tests passed** in 4.69s (including real `dspy.RLM` Python execution in `dspy.LocalInterpreter` and AST verification of `0` `hipporag` imports in `yaw/`).
