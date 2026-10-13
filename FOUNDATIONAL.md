# Foundation: evaluate HippoRAG 2

Decision: 2026-10-13. YAW starts again as a HippoRAG 2 evaluation workspace. This document replaces all previous YAW foundations, epics, ADRs and implementation plans. Old specifications in Git history are historical material, not active instructions.

## Current boundary

Use the pinned upstream HippoRAG package through its public Python API. HippoRAG owns extraction, indexing, retrieval and answer generation. The project has one direct dependency and no custom application, service or retrieval implementation. The exact source revision and transitive dependencies are recorded in `pyproject.toml` and `uv.lock`.

The intended corpus is 100–500 documents, mainly English with occasional other languages. Content may contain disagreements, revisions and ambiguous entities. Processing-time requirements are not yet defined; measure actual latency before setting a target.

RDF, OWL, SKOS and SHACL remain possible future tools for evidence and knowledge management. No semantic layer is required or approved for implementation at this stage. The former CocoIndex, spaCy, ontology compiler, MCP service and Epic 8 plans are retired. Do not port them as part of this evaluation.

## Execute in this order

1. Install the locked environment and verify `from hipporag import HippoRAG`. Record interpreter, package revision and any installation failures. An import pass establishes installation only.
2. Run the README example against Lemonade. Record actual model IDs, extraction and embedding success, retrieved evidence, answer and elapsed time. Stop and report errors; do not silently select another model or backend.
3. Select a few existing source notes. Prepare explicit text passages for `index(docs=...)` and retain their source paths in the evaluation record. Do not assume HippoRAG handles arbitrary file formats or automatically splits a document into suitable passages.
4. Write expected evidence and answers before running queries. Include a direct fact, a question requiring two passages, contradictory sources, a revised claim, an absent answer and a non-English sample. Inspect retrieved passages separately from the generated answer. Do not score contradictions as resolved simply because one answer sounds plausible.
5. Exercise repeated indexing, addition, replacement and deletion using the pinned API. Check the resulting retrieval evidence, including shared facts and duplicate text. Do not equate adding a new version with removing the old version. Record restart/persistence behavior and actual failures.
6. Expand toward the intended corpus only after the small evaluation works. Record quality, runtime, memory and passage-preparation limits. Propose the smallest change justified by the observed failures.

Use one evaluation record when results exist, including commands, versions, selected inputs, expected evidence, actual outputs and unresolved failures. Do not create speculative delivery epics or describe unrun checks as passing. The reset itself does not establish model quality or readiness for the full corpus.

## Decisions deferred until evidence exists

Whether to add document conversion, provenance mappings, semantic validation or an upstream contribution depends on the evaluation. Prefer the public API before subclassing; consider a fork only for a demonstrated requirement that the pinned API cannot support. No fork, copied internals, alternate implementation or compatibility overrides are part of this baseline.

The earlier zero-fallback requirement is not a claim about upstream behavior. HippoRAG has internal retrieval branches and error handling; record their effects during evaluation. YAW adds no fallback path. Source attribution, contradiction handling and deletion correctness must be demonstrated rather than inferred from a graph-based design.
