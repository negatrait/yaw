# YAW — Yet Another Wiki

YAW is a local knowledge compiler and ontology GraphRAG engine for **100–500 Markdown documents**, mainly English with occasional other languages. It is intended to retain source claims, disagreements and revisions, compile supported cross-document knowledge, and retrieve that knowledge with verified evidence.

**Current state:** documentation selects the implementation; production migration and runtime verification remain open in [Epic 7.8](docs/epic7.8-open-ie-ontology-alignment.md). See [TEST_READY](TEST_READY.md). No Epic 8 implementation starts until every 7.8 gate passes.

## Start here when implementing

1. Read [AGENTS](AGENTS.md), then [FOUNDATIONAL](FOUNDATIONAL.md).
2. Follow the [coding-agent execution contract](docs/epic7.8-open-ie-ontology-alignment.md#coding-agent-execution-contract), then take the first unfinished step in its [delivery packet](docs/epic7.8-open-ie-ontology-alignment.md#delivery-packets).
3. Read only the relevant [compiler](docs/knowledge-compilation.md), [spaCy](docs/how-to-handle-spacy-llm.md), [semantics](docs/rdf-owl-shacl.md), [organization](docs/ontological-roots.md) or [retrieval](docs/power-pagerank.md) guide.
4. Use the [test-target map and commands](TEST_INFRA.md#test-target-map) to prove the applicable E78 cases; record actual results and the next step in TEST_READY.

For documentation-only changes, run `python tools/check_docs.py` and `git diff --check`; no YAW environment or model is required for the documentation checker.

[ADR-014](docs/adr/ADR-014.md) resolves earlier alternatives. Historical ADR bodies and research clippings are background, not competing implementation plans. [The roadmap](docs/epics-alignment-roadmap.md) defines 8.1 hardening, 8.2 evaluation and 8.3 release.

## Selected architecture

`raw source revisions → spaCy/Raw qualified claims → shared CocoIndex corpus compiler → SQLite → query / cached lint / Markdown`

- **spaCy/spacy-llm + Ollama:** linguistic assistance and typed extraction, with separately scheduled claim comparison and synthesis through the same adapter.
- **CocoIndex:** one application for source and cross-document dependencies; no post-sync fallback compiler.
- **RDF/OWL/SKOS/SHACL + pySHACL:** resource-defined semantics, scoped inference, admitted rules and validation.
- **SQLite:** source pointers/claims, revisable interpretations, dependencies, generated wiki blocks, findings, FTS and typed vectors.
- **SciPy:** the single candidate/fact/chunk/lexical PPR retrieval algorithm.
- **FastMCP:** shared ingestion, compilation, query and lint handlers.
- **Markdown:** optional deterministic rendering of stored authored blocks; no export-time model calls or re-ingestion of generated pages.

Unknown relations remain available. `open_ie` selects an output vocabulary view, never another extractor. Source assertions remain separate from interpreted identities and generated knowledge. Required-component failures are explicit errors; no alternate model, parser, compiler or retriever takes over.

## Runtime preparation

The initial target is Python **3.14**, uv and the configured local Ollama models. Model names/settings have one owner in `config.toml` / `src/yaw/config.py`; implementations pin resolved digests and locked dependencies. Broader version support and performance are measured release claims.

This repository includes `pyproject.toml`, `uv.lock` and the source entry points. Verify its locked environment and configured models in 7.8.3 before treating setup commands or CLI operation as proven. Documentation reconciliation does not certify an application runtime.

Keep platform environments separate when sharing a checkout:

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".cache/venv-windows"
uv sync --locked
```

```sh
export UV_PROJECT_ENVIRONMENT=.cache/venv-unix
uv sync --locked
```

## Interface contract

| Interface | Responsibility |
| --- | --- |
| `ingest_raw` | Append supplied source material and schedule the one application; report actual job state |
| `compile_wiki` | Refresh the same application's derived inputs and report generation/state |
| `query_wiki` | Return structured ranked passages, current context, counterevidence and coverage |
| `lint_wiki` | Read scoped versioned findings; refresh through `compile_wiki` |

Streamable HTTP uses the configured `/mcp` endpoint; stdio uses the same handlers. Neither transport selects another engine. Source/generation state and typed errors remain visible. Exact launcher/package verification belongs to 7.8 and the subsequent production gates.
