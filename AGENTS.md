Your mission is to create a FastMCP Cocoindex LLM-Wiki indexer + Ontology GraphRAG engine.

## Your Process

**You are about to crash into one of your usual pitfalls again.** If you want to be helpful and avoid this, **consult the table below before proceeding:**

| Principle: | Which of your usual pitfalls it adresses? | The actionable steps to avoid failure. |
| :--- | :--- | :--- |
| **Think Before Coding** | Wrong assumptions, hidden confusion, missing tradeoffs | State assumptions, surface tradeoffs, push back when warranted. |
| **Simplicity First** | Overcomplication, bloated abstractions | Minimum code that solves the problem. Nothing speculative. |
| **Surgical Changes** | Orthogonal edits, touching code you shouldn't | Touch only what you must. Clean up only your own mess. |
| **Goal-Driven Execution** | Leverage through-tests first, verifiable success criteria | Define success criteria. Loop until verified.|
| **Never Guess, Allways Cite** | Making up claims and criteria, copping out of background research | Research and cite sources before you make decisions. |
| **Allways SSOT** | Hardcoding and handrolling several near-similar solutions | Import from a single source of truth, never re-export or copy-paste. |

## Context

The repository is Python 3, with `uv` package manager.
- **FastMCP** for the MCP interface.
	- **Scipy** for PPR GraphRAG ranking through the ontology layer
- **Cocoindex** for Single-Stage continuous indexing, chunking, and embedding the `raw/` notes into SQLite (`vectors.db`).
	- **spacy-llm** with Pydantic for extracting leaf ontologies, named entities, and Open IE relations.
- **Cocoindex & SQLite Views** for instant global interconnection (`unified_edges`, `unified_nodes`) and optional compiled wiki materialization (`vault/wiki/`).

### Progressive Disclosure: Architecture & Implementation Plans
Before authoring code or proposing architectural changes, consult these authoritative documents based on your task scope:
- **Core System Specification:** Read [`FOUNDATIONAL.md`](file:///home/harri/yaw/FOUNDATIONAL.md) **first** for the 7 system invariants, single-stage ingestion flow, SQLite schema contracts, and retrieval algorithms.
- **Implementation & Roadmap Epics:** Consult [`docs/epics-alignment-roadmap.md`](file:///home/harri/yaw/docs/epics-alignment-roadmap.md) when executing or verifying any of the 5 engineering Epics, reviewing TDD test authoring plans, or resolving cross-cutting smells (`ARCH-SSOT-001`, `ARCH-PERF-002`).
- **spaCy-LLM Integration Patterns:** Consult [`docs/how-to-handle-spacy-llm.md`](file:///home/harri/yaw/docs/how-to-handle-spacy-llm.md) whenever touching [`src/yaw/nlp/`](file:///home/harri/yaw/src/yaw/nlp/), configuring models/tasks, or implementing Jinja2 Open IE relation extraction without private attribute monkeypatching.

## Tools

Ad-hoc commands often waste turns. Consult this list to find out repeating, succesful commands:
- `uv run <command>`: Runs the python command in the correct venv.

<!-- lean-ctx -->
## lean-ctx

lean-ctx is active — the MCP tools replace native equivalents.
Full rules: LEAN-CTX.md (open on demand — do not auto-load).
<!-- /lean-ctx -->
