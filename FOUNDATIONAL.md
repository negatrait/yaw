# Foundation: HippoRAG 2 evaluation and local service

Decision baseline: 2026-10-13. YAW uses upstream HippoRAG 2 as its indexing, graph retrieval and QA engine. The earlier YAW application, ontology compiler, CocoIndex/spaCy stack and Epic 7/8 plans remain retired. Git history is reference material, not an active implementation contract.

## Current evidence and planned work

The installed baseline is Python 3.12.14 with HippoRAG 2.0.0a5 pinned to `d5c8329422e0a0b834a15874545cb6a74b4f9b26`. NumPy is also declared for the adapter. [benchmark.py](benchmark.py) (which replaces the retired `live_smoke.py` fixture script) passes repeatability and upstream multi-hop retrieval/QA benchmark checks (`sample`, `2wikimultihopqa`, `musique`) with EmbeddingGemma through llama.cpp. The former FLM embedding path failed and is retired from this evaluation. See the dated [evaluation record](docs/evaluation.md); this benchmark pass is not acceptance of a daemon or source-deletion lifecycle.

The next work is planned in exactly two epics:

1. [Epic 1: HippoRAG 2 daemon mode](docs/epic-1-hipporag-daemon.md): establish correct embeddings first, then add source reconciliation, one persistent runtime, native Windows lifecycle controls and explicit repair after interrupted writes.
2. [Epic 2: FastMCP v4 over Streamable HTTP](docs/epic-2-fastmcp-http.md): expose the same runtime through one local HTTP endpoint. Its implementation follows the Epic 1 acceptance gates.

These epics are not completed features; only the embedding/chat adapters and benchmark harness are implemented. The daemon commands and MCP endpoint described in them are target interfaces.

## Fixed integration decisions

- One process, one state owner, one HippoRAG instance and one serialized operation queue. File reconciliation is the sole writer of corpus membership. MCP uses this runtime rather than starting another upstream server.
- Lemonade remains the local OpenAI-compatible inference server at `http://localhost:13305/api/v1`. In the default 3-slot development configuration (`max_models = 3`), `embeddinggemma-300m-GGUF` (checkpoint `ggml-org/embeddinggemma-300M-GGUF:embeddinggemma-300M-Q8_0.gguf`, mean pooling, `--batch-size 2048 --ubatch-size 2048`, default `distinct-titled` task mode) and the extraction/reranking chat model `Qwen3-4B-Instruct-2507-GGUF` (checkpoint `unsloth/Qwen3-4B-Instruct-2507-GGUF:Qwen3-4B-Instruct-2507-Q4_K_M.gguf`, `--parallel 1` without `-dio`, tightened OpenIE NER and triple JSON Schemas + dynamic candidate-literal `DSPyFilter` GBNF) run on discrete GPU CUDA (`llamacpp`), while the QA reading model `Mistral-7B-Instruct-v0.3-Hybrid` (`amd/Mistral-7B-Instruct-v0.3-onnx-ryzenai-1.7-hybrid`) runs on the Ryzen AI Hybrid NPU/iGPU (`ryzenai-llm`). The FLM 1.0.6 embedding path is excluded because it forces query formatting and failed repeatability; this does not exclude FLM chat. The daemon must pass the explicit embedding provider, split `extraction_llm`/`qa_llm` instances, and truthful model/adapter identity.
- The upstream Parquet backend remains selected for both epics. Milvus is optional and is not a remedy for incorrect embeddings or source lifecycle behavior. A future adoption requires measured benefit in the pinned integration.
- Extraction chat, QA chat, and embeddings are independently selected through the same upstream API (`HippoRAG(embedding_model=..., extraction_llm=..., qa_llm=..., index_identity=...)`). The selected embedding profiles are the standard and QAT EmbeddingGemma 300M weights (supporting `search`, `qa`, `fact`, `distinct`, `titled`, and `distinct-titled` task modes) plus Qwen3-Embedding-0.6B. Registered chat profiles include `Qwen3-4B-Instruct-2507-GGUF`, `Mistral-7B-Instruct-v0.3-Hybrid`, step-up FLM models (`gemma4-it-12b-FLM`, `qwen3.5-9b-FLM`), and secondary `llamacpp` candidates (`Phi-4-mini-instruct-GGUF`, `Llama-3.2-3B-Instruct-GGUF`). Exact checkpoints, compatibility evidence and the real HippoRAG benchmark gate are in [model evaluation](docs/model-evaluation.md). EmbeddingGemma-2/multimodal support is deferred. Explicitly selecting a validated model profile is configuration, not a fallback; no model is silently substituted. Each extraction-chat/embedding combination must pass fresh gold-document/answer evaluation and index-compatibility checks before daemon/MCP acceptance.
- Correct EmbeddingGemma behavior is implemented through its public injection point with a small adapter; no global monkeypatch, copied retrieval implementation or fork. Existing experimental vectors are rebuilt when prompt/adapter identity changes.
- FastMCP 4.1.0 is the selected v4 target. Use Streamable HTTP on `127.0.0.1:13306/mcp` with a shared application lifespan. HTTP transport does not imply answer-token streaming.
- Python stays on 3.12 while the upstream `torch==2.5.1` pin prevents installation on 3.14.7. Add direct dependencies when the integration actually imports them: filelock in E1 and FastMCP in E2. Pin/lock and verify them at implementation time; do not install optional database stacks preemptively.

## Corpus and correctness boundary

The intended corpus has 100–500 documents, mainly English with occasional other languages. Start with UTF-8 `.md` and `.txt` files under one configured root. The first daemon uses deterministic bounded passages and a source manifest. Document conversion is a separate future decision. No processing-time SLA exists; measure model calls, scan cost, vector counts, memory and latency.

Preserve exact passage text and every source owner. Relevance scores are not truth confidence. Contradictions remain evidence to retrieve and report; the wrapper does not decide canonical truth. RDF/OWL/SKOS/SHACL remain deferred until retrieval evidence establishes a concrete need.

No YAW-selected model/backend fallback is permitted. The selected HippoRAG version still has internal retrieval branches; observe and report them rather than claiming their removal. Dirty or inconsistent state is surfaced and explicitly rebuilt. A health response, package import or generated answer is insufficient evidence of correct indexing, ranking, source deletion or persistence.

## Document ownership and execution

FOUNDATIONAL owns scope and shared decisions. `docs/model-evaluation.md` owns selected model contracts and the benchmark comparison protocol; report differences without an automatic degradation cutoff. Each epic owns its contracts, delivery order and acceptance tests. `docs/evaluation.md` owns measured outcomes and unresolved failures. README owns commands available now and clearly labels future ones.

Continue D1-01 with multilingual/chunk-boundary validation; its adapter and nine-document smoke checks now pass. Complete the first unfinished packet, run its targeted checks and record commands/results before advancing. Retain explicit failures and use live model evidence for quality claims. Reconcile these documents if an accepted decision changes; do not resurrect retired architecture to fill an ambiguity.
