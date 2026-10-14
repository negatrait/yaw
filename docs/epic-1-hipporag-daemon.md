# Epic 1: HippoRAG 2 daemon mode

Status: D1-01 adapter and nine-document regression implemented; multilingual/chunk-boundary validation and daemon implementation remain. Owner contract: [FOUNDATIONAL](../FOUNDATIONAL.md). Evidence: [evaluation record](evaluation.md). Next epic: [FastMCP v4](epic-2-fastmcp-http.md).

## Outcome and boundary

A native Windows background process keeps one HippoRAG instance alive and reconciles a configured source folder into one persistent index. Operators can start, inspect, synchronize and stop it. New text, edits, renames and deletions have defined behavior. An interrupted update never becomes a silently trusted index.

This is the first epic after the MVP reset; its number does not continue the retired Epic 7/8 series. Use the pinned upstream package. Scope is local UTF-8 Markdown/text input, Lemonade, the upstream Parquet stores and lifecycle glue. PDF/Office conversion, operating-system service installation, restart-at-login, remote hosting, Milvus, ontology compilation and custom retrieval are outside these two epics.

## Prerequisite: fix and prove embeddings (D1-01)

The initial `live_smoke.py` exposed query-role and FLM serving defects. The corrected adapter/smoke run now passes items 1–5 below, using an explicit EmbeddingGemma llama.cpp backend change. These remain regression requirements. The 1024-byte English/multilingual chunk-boundary checks below remain unfinished; the daemon is not implemented.

1. Replace process-wide assignments to `OpenAIEmbeddingModel.encode/batch_encode` with one small injected embedding adapter. Set the actual model ID `user.embeddinggemma-300m-GGUF` and explicit provider `openai`. Keep upstream SDK/batching/normalization and preserve ordering, float encoding, response validation, usage reporting and zero-vector handling. The shared adapter owns encode because upstream newline flattening would alter Qwen instructions. Do not substitute a made-up model name to select a provider.
2. Set `query_instruction_mode = "shared"`: query-to-fact and query-to-passage both use `task: search result | query: ...`; passage, entity and fact inputs use `title: none | text: ...`. Prefix exactly once. The pinned HippoRAG omits instruction arguments for a provider declaring `ignored`, which is why the former monkeypatch misclassified queries. Use an explicit `index_identity` containing adapter/prompt/normalization versions; close the injected client explicitly when its owner shuts down.
3. Probe the real endpoint with identical inputs alone, in a batch, and after reordered batches. Record input strings, model/backend versions, dimensions, row indices, finite/nonzero vectors and pairwise cosine. Acceptance: the same input has cosine >= 0.999 across those contexts on five repeated probes. This is a proposed regression threshold, not a model vendor guarantee. The corrected llama.cpp path passes; FLM 1.0.6 failed directly as well as through Lemonade. Diagnose the selected serving backend if it fails; do not hide it by silently switching model, single-item requests or vector backend.
4. Rebuild into a fresh disposable index after changing prompts or adapter identity. Log the indexing stages, extracted triples, retrieval branch/seeds and the actual passages supplied to QA. The politician fact must rank first; both birthplace passages must be within QA's five-passage evidence window. Record the three fixture answers and useful recall@1/@2/@5; retrieving all nine documents is not a discriminating recall test.
5. Correct the Cinderella reference answer before evaluating the corrected run: the supplied evidence links the matching slipper to reunion with the prince. Keep retrieval and answer-quality scores separate, and report a correct paraphrase separately from exact-match scoring. Do not tune the expected answer to an observed prediction.

Use the checkpoint and mean-pooling CUDA setup in README. The switch preserves EmbeddingGemma 300M while explicitly replacing its faulty FLM serving path; model selection is explicit and no runtime fallback is added.

Failure of this gate blocks claims that the daemon has a usable model baseline. Lifecycle unit work can be prepared, but D1 cannot close on installation/import evidence alone. Google documents the two prompt forms and a 2K input context for EmbeddingGemma; Lemonade's advertised 8192 setting does not establish a larger effective embedding window. [Model card](https://ai.google.dev/gemma/docs/embeddinggemma/model_card#prompt-instructions)

## Model compatibility gate (2026-10-14 clarification)

Preserve the heterogeneous device split across Lemonade's 3-slot pool (`max_models = 3`): `embeddinggemma-300m-GGUF` and `Qwen3-4B-Instruct-2507-GGUF` (`extraction_llm`) on discrete GPU CUDA (`llamacpp`), and `Mistral-7B-Instruct-v0.3-Hybrid` (`qa_llm`) on Ryzen AI Hybrid NPU/iGPU (`ryzenai-llm`), with `gemma4-it-12b-FLM` and `qwen3.5-9b-FLM` retained as step-up NPU chat profiles. FLM's diagnosed embedding defects do not imply a ban on FLM chat.

Before admitting further models, extend the existing adapter/configuration with explicitly selected, validated model profiles, not parallel pipelines or error-triggered provider switches:

- Keep extraction chat, QA chat, and embedding model IDs and endpoints independently configured. Chat uses the upstream OpenAI-compatible client with the explicit recipe-specific policy in `yaw/chat.py` (`gbnf-grammar:require-stop` on `llamacpp`, `fold-system:require-stop` on `ryzenai-llm`, `think-off:require-stop` on `flm`); exercise extraction, recognition/reranking and QA with each selected chat combination. Unsupported request parameters or malformed responses must surface as failures rather than silent parameter/model substitution.
- Each embedding profile specifies exact checkpoint/revision, query/document formatting (including EmbeddingGemma's `search`, `qa`, `fact`, and `distinct` task modes), whether the server adds formatting, expected dimensions, normalization, pooling and supported context limit. Do not apply EmbeddingGemma's prefixes or 768-dimensional check to Qwen by assumption. Keep inherited SDK/batching and tested row-order/usage validation in one adapter implementation.
- Test single/batched/reordered/repeated inputs and active document/query roles for every profile. Run `live_smoke.py --mode eval` on a fresh index for each approved chat/embedding combination, retaining gold documents, gold answers, per-question and aggregate recall/EM/F1 (both graph retrieval and DPR baseline) plus actual QA evidence and non-empty `graph_seeds`. An import, device health response or successful HTTP request cannot close this gate.
- Include model weights/revision and prompt/dimension/pooling/normalization policy in index identity, along with the `extraction_llm` policy identity. Changing extraction chat or embedding profile must reject incompatible persisted state and require an explicit rebuild. The smoke command always creates fresh state.

Exact profiles and a shared adapter are now implemented for the two selected Gemma quantizations and Qwen3-Embedding-0.6B, with independent `extraction_llm` and `qa_llm` selection. Read [model evaluation](model-evaluation.md) for evidence boundaries and the real-benchmark implementation packet. EmbeddingGemma-2/multimodal support remains deferred. Report model differences without an automatic degradation cutoff. Treat compatibility and benchmark evidence as required before closing the development epics; retain the English/multilingual chunk-boundary checks.

## Runtime and commands

Target interface, to be implemented; these commands do not exist yet:

```powershell
uv run --locked python -m yaw run --config yaw.toml
uv run --locked python -m yaw start --config yaw.toml
uv run --locked python -m yaw status --config yaw.toml
uv run --locked python -m yaw sync --config yaw.toml
uv run --locked python -m yaw stop --config yaw.toml
uv run --locked python -m yaw rebuild --config yaw.toml
```

- `run` is the foreground entry point. `start` launches exactly that entry point with the environment's interpreter, hidden window and redirected logs; it never runs a second implementation. Wait up to 30 seconds for the control endpoint to acknowledge the matching instance ID. Report `starting` if model initialization is still underway, not `ready`.
- One canonical state directory has one lifetime `filelock.FileLock`. A second owner exits before constructing HippoRAG. A stale lock-file pathname alone is not a running process. Do not use `os.kill(pid, 0)`, Unix `fcntl`, PID-only termination, or port-killing shortcuts on Windows.
- Use a local Windows named pipe via standard-library `multiprocessing.connection` for control. Derive the pipe name from the canonical state path. Use a per-run random auth key and instance ID stored under that user's state directory. Exchange bounded JSON using `send_bytes/recv_bytes`, never pickle `send/recv`. The controls are status, sync and stop; no RAG RPC is built here.
- `status` reads a live control response and cached state, without waiting for an LLM or touching HippoRAG on another thread. A failed control connection is reported as unresponsive/stopped based on the owner lock, never as healthy from a PID file. A stale response from an earlier instance is rejected.
- `sync` coalesces an immediate full scan and returns the scan ID. It does not directly mutate HippoRAG from the CLI process. `stop` stops admission and scanning, rejects queued work with SHUTTING_DOWN, then waits for the active operation and closes resources. After a 120-second client wait, report that shutdown is still pending; do not force-kill. Ctrl+C in foreground uses the same shutdown path.
- `rebuild` requires the daemon to be stopped and the same owner lock. It builds a fresh local index from the currently readable source corpus and publishes it only after successful completion. It is explicit repair, not automatic recovery or an alternate retrieval path.

Use one `Runtime` async context manager for initialization, polling, state and shutdown. Construct, call and close HippoRAG on one worker thread. Index, delete, prepare-for-retrieval, retrieve, QA and graph-stat collection are serialized through a bounded FIFO queue of 32 waiting operations. Cache status separately. Coalesce scan requests to one pending scan. A busy queue returns an explicit error; do not create extra HippoRAG instances for concurrent clients.

After Epic 2, the FastMCP server lifespan enters this same context manager. The daemon controls and runtime remain the same; the foreground host gains the HTTP adapter. Do not expose a second independent upstream MCP process against this state directory.

## Configuration and identity

Create one `yaw.toml` example with these settings; relative paths resolve against its location. Validate unknown keys and unsupported values instead of ignoring them.

| Setting | Initial value / rule |
| --- | --- |
| `corpus.root` | `sample-llm-wiki/raw`; one existing directory |
| `corpus.poll_seconds` | 5; positive integer |
| `corpus.chunk_bytes` | 1024; fixed for this initial baseline |
| `index.state_dir` | `outputs/daemon`; outside the corpus root |
| `models.base_url` | `http://localhost:13305/api/v1` for chat and embeddings |
| `models.extraction_chat` | `Qwen3-4B-Instruct-2507-GGUF`; GPU CUDA `llamacpp` |
| `models.qa_chat` | `Mistral-7B-Instruct-v0.3-Hybrid`; Ryzen AI Hybrid NPU/iGPU (`models.chat` may set both when single-model mode is desired) |
| `models.embedding` | `user.embeddinggemma-300m-GGUF` (with optional `models.embedding_task_mode = "search"`) |
| `models.api_key_env` | Optional environment-variable name, never a literal secret |
| storage | Explicit upstream `vector_store_type="parquet"`; no configurable backend selector |

Pin the HippoRAG commit and Python 3.12 baseline. NumPy is already a direct dependency of the experiment. Declare `filelock` directly when importing it in the daemon; it already exists transitively. All other E1 orchestration uses the standard library. Do not install upstream's MCP or Milvus extras in E1.

Persist a configuration fingerprint including canonical source root, HippoRAG revision, model IDs/endpoints, adapter/prompt/normalization identity and chunker version. Refuse to open an incompatible existing index. A changed fingerprint requires the explicit rebuild command. A successful HTTP response is not proof that the model served the requested weights; record the server-reported model/backend version during validation.

## Folder reconciliation and source ownership

1. Poll the full tree on startup and every configured interval. For 100–500 files this avoids adding a watcher dependency. Accept `.md` and `.txt`, case-insensitively; ignore hidden directories and report unsupported extensions in status counts. Do not traverse symlinks/junctions or paths that escape the configured root. Do not write source files.
2. Read UTF-8 with optional BOM and normalize CRLF/CR to LF. Hash content rather than trusting modification time. Require the same path/content hash on two consecutive scans before admitting additions/edits; require absence on two complete scans for deletions. A missing root, failed directory enumeration or unreadable supported file invalidates the scan: retain the previous committed index, report the error and admit no deletions from that scan.
3. Split deterministically into contiguous passages of at most 1024 UTF-8 bytes. Prefer the last newline within the limit; otherwise split at a Unicode character boundary. No overlap, summarization, lowercasing, title injection or silent truncation. Record offsets in the normalized source and original byte hash. Skip whitespace-only spans but report empty sources. D1-01 must verify boundary-size English and multilingual inputs against the actual embedding and chat limits before this chunking policy passes; bytes are a conservative sizing policy, not a claimed token count.
4. Maintain a single committed source manifest: relative path, original file hash, normalized text hash, and ordered passage entries containing exact text, SHA-256, start/end offsets. Derive the set of source owners of each exact passage from it. Reject case-colliding relative paths on this Windows baseline. Renames are removal/addition of source ownership, so unchanged text need not be embedded again.
5. Diff the unique exact texts of the previous and next manifests. Call `index(new_texts)` for texts with no previous owner and `delete(orphaned_texts)` only when no next owner remains. Shared text is indexed once, and remains searchable when only one of its source files disappears. Update the manifest for an ownership-only change without re-extracting text. Do not rely on HippoRAG's single `source_id` slot to represent duplicate owners.
6. A query's citations are reconstructed by matching each returned exact passage against the committed manifest. Return every owning relative path/hash/offset span. Unknown text is a consistency error. Passages and scores stay in upstream order; the wrapper does not rerank or infer canonical facts.

## Commit, failure and restart contract

HippoRAG writes multiple files during a mutation. Atomic graph-file replacement is not an atomic transaction across graph, vectors, OpenIE and source manifest.

Before calling an index/delete mutation or publishing an ownership change, durably write a dirty marker containing the intended scan ID. Execute no query during the reconciliation; admitted queries wait in the bounded queue and recheck index state when they start. Apply additions, then removals, then prepare retrieval objects when nonempty; publish the next source manifest using a flushed temporary file and atomic replacement. Clear the dirty marker only after all steps succeed. An unchanged scan calls neither index nor delete. A query queued before a failed mutation must receive NEEDS_REBUILD rather than running against the damaged state.

Any exception after marking dirty puts the runtime into `needs_rebuild`: reject queries and mutations, retain the last error, keep status/stop responsive. A dirty marker on restart has the same meaning. Do not replay a half-applied batch or silently query a partially updated store. Failed reads before mutation leave the last committed index usable with `sync_error` and its last successful scan time visible.

Rebuild uses a fresh generation directory under the same state root. On success, atomically replace a small `current.json` pointer; the manifest and HippoRAG files belong to that generation. A failed rebuild leaves the prior pointer untouched and reports failure. Never import legacy `yaw.db` or experiment outputs. Local index generations are disposable derived state, not external project backups.

Status fields: instance ID, lifecycle (`starting`, `running`, `stopping`, `failed`), index state (`empty`, `ready`, `updating`, `sync_error`, `needs_rebuild`), configuration fingerprint, current/last scan IDs, last successful scan timestamp, file/passage/entity/fact counts, queue depth, active operation, last error and log path. `empty` is healthy with zero sources; retrieval returns an empty collection and QA returns `EMPTY_INDEX` without a model call. Log operation IDs and phase timings; redact credentials and avoid routine logging of full document text.

## Delivery packets and verification

Select the first unfinished packet; record its command, result and outstanding failure in [evaluation](evaluation.md) before moving on. All boxes start unchecked.

| Packet | Owned files (future) | Observable exit condition |
| --- | --- | --- |
| D1-01 (partial) | `yaw/embedding.py`, `live_smoke.py`, `tests/test_embedding.py`, `tests/test_live_smoke.py` | Adapter, stable probes and clean nine-document evaluation pass; remaining: English/multilingual 1024-byte boundary validation against embedding and chat limits |
| D1-02 | `yaw/config.py`, `yaw/runtime.py` | One validated config and one worker-owned instance; bounded queue and explicit cleanup including injected client |
| D1-03 | `yaw/corpus.py` | Stable scans, deterministic passages, exact text manifest and duplicate-owner reconciliation |
| D1-04 | `yaw/daemon.py`, `yaw/__main__.py`, `yaw.toml` | Foreground/background lifecycle and control protocol operate on native Windows |
| D1-05 | Same modules, `tests/test_daemon.py`, `tests/test_corpus.py` | Dirty-state detection and explicit rebuild; failure-injection checks pass |
| D1-06 | README, evaluation record | Real Lemonade acceptance and restart evidence recorded; all limitations stated |

Use standard-library `unittest` for deterministic lifecycle/source-ownership tests; use controlled fakes for injected write failures, never as evidence of real model quality. Place embedding contract tests in `tests/test_embedding.py`. Execute the real run separately against Lemonade with disposable source/index fixtures.

- [ ] **D1-A1:** Start twice against one state root: second start cannot load or mutate the index. Stop/restart releases clients and owner lock. Killing a disposable test child leaves no false-live PID behavior.
- [ ] **D1-A2:** Stable new file is indexed; unchanged scan/restart causes zero embedding or extraction requests. Changed file removes unique old evidence before queries resume. Rename and deleting one of two identical sources retain the correct remaining citation.
- [ ] **D1-A3:** Delete a source sharing an entity/triple with another: surviving evidence remains; deleted source contributes no stale citation. Deleting the last file yields the defined empty-index behavior.
- [ ] **D1-A4:** Missing root, invalid UTF-8, unreadable file, reparse-point escape and file changing mid-read produce explicit outcomes and no inferred mass deletion.
- [ ] **D1-A5:** Inject failure after embedding persistence, after OpenIE persistence, after deletion, and before manifest publication. Each affected run/restart requires explicit rebuild. Rebuild failure never publishes the new generation.
- [ ] **D1-A6:** A slow index operation cannot overlap retrieval or shutdown cleanup; status remains responsive within two seconds in the local test. Saturating 32 waiting slots produces a busy error without extra workers. Cancelling a queued request removes it; cancelling a running awaiter never allows the next operation to overlap its thread.
- [ ] **D1-A7:** Real nine-document evaluation, multilingual/boundary-size passages and a real add/edit/delete/restart sequence pass with model/version/latency evidence. Do not certify relevance from answer text alone.

Epic 2 depends on D1-01 through D1-06 and all D1 acceptance checks. Required failures stay visible as blockers; adding a different database is not the repair for an embedding or source-ownership defect.

## Source references

The attached DeepWiki analysis describes an older revision. Our pinned source includes an optional MCP service and explicit state validation; do not copy the attachment's unconditional crash-recovery or MCP-absence claims. See upstream [HippoRAG implementation](https://github.com/OSU-NLP-Group/HippoRAG/blob/d5c8329422e0a0b834a15874545cb6a74b4f9b26/src/hipporag/HippoRAG.py), [MCP service](https://github.com/OSU-NLP-Group/HippoRAG/blob/d5c8329422e0a0b834a15874545cb6a74b4f9b26/src/hipporag/mcp_server.py) and [OpenAI embedding adapter](https://github.com/OSU-NLP-Group/HippoRAG/blob/d5c8329422e0a0b834a15874545cb6a74b4f9b26/src/hipporag/embedding_model/OpenAI.py). These sources inform the integration boundary; YAW's lifecycle and reconciliation contracts above are project decisions.
