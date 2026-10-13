# YAW architecture reference

**Status:** Navigation and component responsibility guide; [FOUNDATIONAL](../FOUNDATIONAL.md) is the contract owner.\
**Active refactoring:** [Epic 7.8](epic7.8-open-ie-ontology-alignment.md), selected by [ADR-014](adr/ADR-014.md). Coding agents start with its [execution contract](epic7.8-open-ie-ontology-alignment.md#coding-agent-execution-contract).

YAW continuously turns notes into source-grounded knowledge and retrieves their connections and evidence. Open discovery, ontology-driven interpretation and revisable identities are central. Imported vocabularies enrich knowledge without exhausting what may be discovered.

| Component | Responsibility | Detailed guide |
| --- | --- | --- |
| CocoIndex | One application for source processing and shared corpus compilation | [Continuous compilation](knowledge-compilation.md) |
| spaCy / spacy-llm | Linguistic analysis and validated mention/assertion extraction | [spaCy integration](how-to-handle-spacy-llm.md) |
| Semantic resources and registry | Domain meaning, lexical alignment, kinds, supported inference and shapes | [Semantic substrate](rdf-owl-shacl.md), [loading](epic7-pluggable-ontology-integration-spec.md) |
| SQLite | Original assertions, versioned interpretations and derived views, FTS and typed vectors | [Storage contract](../FOUNDATIONAL.md#component-2-storage--schema-contracts) |
| SciPy | Candidate/fact/chunk relevance and graph traversal | [PPR](power-pagerank.md), [derived hierarchy](ontological-roots.md) |
| FastMCP | Ingest, compile, query and lint interfaces | [FOUNDATIONAL interface](../FOUNDATIONAL.md#component-4-fastmcp-protocol-interface) |
| Wiki materializer | Optional deterministic rendering of stored authored blocks; zero generation at export | [Compiler output](knowledge-compilation.md#5-organization-and-authored-knowledge) |

A change to semantic resources refreshes dependent interpretation; it does not overwrite the source extraction. Runtime registry indexes are compiled projections. Ranking weights are algorithm parameters; domain relationships are resource data. Windows/Unix adapters affect execution mechanics, not semantics.

[The roadmap](epics-alignment-roadmap.md) requires all 7.8 gates before 8.x. [The ADR ledger](ADR.md) records amendments. [Readiness](../TEST_READY.md) records verified behavior. Errors never select another extractor, compiler or retriever. This navigation guide does not duplicate execution checklists, schemas or algorithms.
