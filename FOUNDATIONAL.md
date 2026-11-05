# YAW Foundations

## 1. Purpose

YAW is a native HippoRAG 2-style retrieval engine built around a single SQLite database.

The project exists to reproduce or exceed HippoRAG 2 retrieval quality with a smaller, inspectable, locally operable architecture. YAW owns its ingestion, storage, retrieval, graph traversal, inference orchestration, and source hydration.

YAW is not a wrapper around HippoRAG. Upstream HippoRAG is a behavioral and benchmark reference only.

## 2. Success Criteria

YAW succeeds only when it demonstrates all of the following:

1. **Retrieval quality**
   - Tier 1 benchmark parity during Epic 1 development.
   - Tier 2 full-corpus parity or better before Epic 1 acceptance.
2. **Native implementation**
   - No `hipporag` imports in production code, benchmarks, scripts, or tests.
   - No hidden delegation to an upstream HippoRAG runtime.
3. **Architectural containment**
   - Each domain owns its implementation and internal dependencies.
   - Shared code is limited to genuinely domain-neutral primitives in `yaw.core`.
   - Cross-domain imports are rejected by automated tests.
4. **Operational simplicity**
   - One runtime.
   - One authoritative database.
   - One database writer.
   - Explicit failures instead of silent degradation.
5. **Traceable evidence**
   - Evaluation records distinguish setup success from retrieval and inference correctness.
   - Acceptance claims are supported by reproducible benchmark outputs.
   - Architectural and schema cutovers are complete only when all runtime lifecycles, callers, and telemetry consumers (`scripts/watchdog.py`) are wired to the active schema and verified end-to-end.

Passing unit tests alone does not establish benchmark parity or system acceptance.

## 3. Non-Goals

YAW does not aim to provide:

- Backward compatibility with abandoned internal APIs.
- A compatibility layer for HippoRAG.
- A general-purpose ontology platform.
- Multiple interchangeable storage backends.
- Distributed ingestion or storage.
- Silent fallbacks between retrieval or inference paths.
- Abstractions created only for possible future use.

The following technologies remain retired unless controlled benchmarks demonstrate a material improvement:

- `rdflib`
- OWL 2 RL
- `pySHACL`
- spaCy-based ontology processing

A benchmark improvement must justify the additional dependency, complexity, runtime cost, and maintenance burden.

## 4. System Model

YAW consists of five conceptual capabilities:

1. **Ingestion**
   - Observes corpus changes through OS event-driven CocoIndex (`localfs.walk_dir(..., live=True)`).
   - Streams ingestion and candidate indexing in dataset-agnostic, bounded micro-batches (`64` chunks per `app.update()` cycle) so heap/VRAM usage stays flat and `yaw.db` publishes incremental graph state every 64 chunks. Never encode benchmark presets or train/val splits in production ingestion.
   - Persists CocoIndex LMDB memoization state (`coco.Settings(db_path=...)`) in a shared, content-addressed cache across runs—never tied to transient per-run database filenames.
2. **Storage**
   - Incremental publication reuses unchanged validated contributions and evidence-keyed adjudications. Its input checkpoint commits with the graph; unresolved SHACL violations abort publication. For fixed evidence, 64+64 and 128 chunks must yield the same canonical graph with comparable cumulative compute, as specified in [native graph publication](docs/native-cocoindex-graph.md). Existing graph-wide staging is an unfinished cutover, not accepted target behavior.
   - Persists source metadata, text, graph data, vectors, and lexical indexes in `yaw.db`.
   - Declares every SQLite table and `sqlite-vec` `vec0` index through typed CocoIndex targets (`TableSchema.from_class`), with one writer and no competing imperative `upsert_*` sinks.
3. **Retrieval & Ontology**
   - Produces lexical, vector, and graph-derived evidence using sparse CSR matrices and Forward-Push Personalized PageRank.
   - Maintains closed-world W3C concept/entity semantics, source-supported facts and relations, and revisable canonical identities across documents and ingestion batches. Legitimate semantic islands are allowed; incomplete documents are not reported as semantically complete.
   - Removes `skos:ConceptScheme` and all scheme clustering, storage, inference, embeddings and retrieval/QA/telemetry consumers from the current implementation target. Reintroduction requires a separate experiment proving held-out Recall@N, EM or F1 improvement against the scheme-free baseline with acceptable compute, memory and other quality metrics.
4. **Inference**
   - NER alone discovers and defines resources: particular `owl:NamedIndividual` instances and general `skos:Concept` categories. Triple extraction selects existing NER resources as subjects/resource objects and discovers literal objects while expressing passage-supported `rdf:Statement` facts. It cannot add resources or redefine them; an ambiguous label requires selecting an existing NER sense. Unrelated supported statements need not form one connected component.
   - Uses native DSPy interfaces (`dspy.LM`, `dspy.Embedder`, `dspy.Signatures`, `dspy.RLM`, `dspy.GEPA`) and treats model services as external dependencies.
   - Defaults LLM extraction and reasoning admission strictly to `parallel = 1` (`Semaphore(1)`) to respect local `llama.cpp` single-slot VRAM bounds, while batching dense embeddings (`max_batch_size = 64`).
   - Enforces sampling temperatures $\ge 0.3$ (defaulting to $0.35$ for extraction and QA, up to $0.7$–$0.8$ for reflection) on small local models to avoid looping; temperatures below $0.3$ are prohibited.
5. **Runtime**
   - Coordinates ingestion, retrieval, inference, and source hydration in `yaw/daemon/runtime.py`.
   - Exposes one application-level API used by transports and evaluation.

Transport adapters do not contain application logic.

## 5. Repository Boundaries

The implementation is contained within:

```text

yaw/

├── core/
└── daemon/
    ├── ingestion/
    └── graph/
        ├── storage/
        └── inference/

```

## 6. Failure Policy

Until YAW is feature complete, every failure is a hard failure.

- DSPy's native retry policy is the only retry mechanism for transient model-service errors.
- When those retries are exhausted, or a model service is unavailable, the operation aborts with the original error. Nothing is skipped, defaulted or degraded.
- Fallbacks, degraded modes and recovery paths are out of scope until the core pipeline is complete. Each is then introduced as an explicit, scoped decision, with tests that exercise the failure it handles.

At this stage a fallback hides defects and corrupts benchmark evidence. A visible failure is more useful than a run that completes.

## 7. Further Context

- [HippoRAG (`2405.14831v3`)](docs/arxiv-2405.14831v3-hipporag.md)
- [HippoRAG 2 (`2502.14802v1`)](docs/arxiv-2502.14802v1-hipporag2.md)
- [Epic 1: Native `yaw.db` Engine & CocoIndex Daemon](docs/epic-1-hipporag-daemon.md)
- [Epic 2: FastMCP Streamable HTTP Adapter](docs/epic-2-fastmcp-http.md)
