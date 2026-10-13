Your mission is to create a FastMCP Cocoindex LLM-Wiki indexer + Ontology GraphRAG engine.

## Your Process

Before editing, apply these rules to the selected work item:

| Principle: | Failure to prevent | The actionable steps to avoid failure. |
| :--- | :--- | :--- |
| **Think Before Coding** | Wrong assumptions, hidden confusion, missing tradeoffs | State assumptions, surface tradeoffs, push back when warranted. |
| **Simplicity First** | Overcomplication, bloated abstractions | Minimum code that solves the problem. Nothing speculative. |
| **Surgical Changes** | Orthogonal edits, touching code you shouldn't | Touch only what you must. Clean up only your own mess. |
| **Goal-Driven Execution** | Leverage through-tests first, verifiable success criteria | Define success criteria. Loop until verified.|
| **Never Guess, Always Cite** | Making up claims and criteria, copping out of background research | Research and cite sources before you make decisions. |
| **Always SSOT** | Hardcoding and handrolling several near-similar solutions | Import from a single source of truth, never re-export or copy-paste. |

## Context

The repository is Python 3, with `uv` package manager.
- **FastMCP** for the MCP interface.
	- **Scipy** for PPR GraphRAG ranking through the ontology layer
- **Cocoindex** for Single-Stage continuous indexing, chunking, and embedding the `raw/` notes into SQLite (`yaw.db`).
	- **spacy-llm** with Pydantic for extracting leaf ontologies, named entities, and Open IE relations.
- **SQLite Views & Engine** for instant global interconnection (`unified_edges`, `unified_nodes`) and optional compiled wiki materialization (`vault/wiki/`).

### Repository boundary
Before writing, verify that the working directory and Git remote match the repository explicitly named by the user. An attached workspace or a remembered checkout path does not override that location.

### Progressive Disclosure: Architecture & Implementation Plans
Before authoring code or proposing architectural changes, consult these authoritative documents based on your task scope:
- **Core System Specification:** Read [FOUNDATIONAL.md](FOUNDATIONAL.md) **first** for invariants and cross-component contracts.
- **Current execution contract:** Read [Epic 7.8](docs/epic7.8-open-ie-ontology-alignment.md#coding-agent-execution-contract) for the resume protocol, module ownership and ordered delivery packets. Select one unfinished packet step and its observable acceptance result before editing. [ADR-014](docs/adr/ADR-014.md) selects one production path with zero fallbacks.
- **Roadmap and entry gates:** Read [the roadmap](docs/epics-alignment-roadmap.md) before Epic 8 work. Every 7.8 runtime gate must pass before any 8.x implementation begins; documentation completion is not runtime completion.
- **Component guides:** Read [spaCy integration](docs/how-to-handle-spacy-llm.md) for NLP work and [continuous compilation](docs/knowledge-compilation.md) for orchestration/publication. Follow their links to the semantic, hierarchy and retrieval owners as needed. Historical ADR bodies and research clippings are not alternative implementation instructions.
- **Evidence:** [USER_STORIES](docs/USER_STORIES.md) owns E78 acceptance cases; [TEST_INFRA](TEST_INFRA.md) owns execution categories; [TEST_READY](TEST_READY.md) records actual results. Do not invent a pass, restore an obsolete path or use another checkout's results when the current runtime is incomplete.

## Tools

Ad-hoc commands often waste turns. Consult this list to find out repeating, succesful commands:
- [TEST_INFRA](TEST_INFRA.md#entry-commands) owns environment, focused-test, full-regression and actual-model commands. Use locked dependencies and explicit test targets; a selected/empty pytest run is not a full gate.
- `python tools/check_docs.py` and `git diff --check` verify documentation edits. The checker uses only the standard library and does not import or certify YAW.
