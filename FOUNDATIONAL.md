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
   - Observes corpus changes through CocoIndex.
   - Coalesces changes into deterministic synchronization deltas.
   - Submits each logical update through the sole database-writing path.
2. **Storage**
   - Persists source metadata, text, graph data, vectors, and lexical indexes in `yaw.db`.
   - Owns schema creation, migration, transactions, and storage-specific queries.
3. **Retrieval**
   - Produces lexical, vector, and graph-derived evidence.
   - Uses sparse graph structures and Personalized PageRank where required by the retrieval contract.
4. **Inference**
   - Uses native DSPy interfaces for language models, embeddings, signatures, recursive language-model behavior, and optimization.
   - Treats model services as external dependencies.
5. **Runtime**
   - Coordinates ingestion, retrieval, inference, and source hydration.
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