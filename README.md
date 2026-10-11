<div align="center">
  <img src="https://raw.githubusercontent.com/yawn-hub/assets/main/yaw-logo.png" alt="yaw logo" width="180" />
  <h1>yaw</h1>
  <p><strong>FastMCP Cocoindex LLM-Wiki Indexer & Ontology GraphRAG Engine</strong></p>
</div>

# YAW IS WIP!

Had to save to github to save the repository.

yaw is an autonomous, local-first GraphRAG engine combining **Cocoindex**, **spaCy + spacy-llm**, **SQLite** (`fts5` + `sqlite-vec`), and **SciPy** (Personalized PageRank through the ontology layer) served via **FastMCP**.

> [!NOTE]
> For the complete, authoritative system specification and invariant contracts, see [`FOUNDATIONAL.md`](FOUNDATIONAL.md). For implementation phases, see [`docs/epics-alignment-roadmap.md`](docs/epics-alignment-roadmap.md).

## Architecture

The system operates via single-stage continuous ingestion into a single SQLite database SSOT (`.cache/yaw.db`), providing instant global interconnection without multi-stage staging barriers:

```mermaid
flowchart TD
    subgraph Vault [Markdown Vault]
        Raw["vault/raw/ (Immutable Source)"]
        Wiki["vault/wiki/ (Optional Compiled dist/)"]
    end

    subgraph Ingestion [Single-Stage Continuous Ingestion]
        Raw -->|Chunk & Delta Watch via CocoIndex| Chunks[Deterministic Chunks]
        Chunks -->|title: doc \| text: chunk| Vec[sqlite-vec Chunks Vec]
        Chunks -->|BM25 indexing| FTS[query_fts FTS5]
        Chunks -->|Doc-integrity spaCy Pipeline| Leaf[Leaf Ontology Triples]
        Leaf -->|EDC Self-Canonicalization| Aliases[aliases & canon_verdicts]
    end

    subgraph Storage [Single SQLite Database SSOT: yaw.db]
        Vec & FTS & Leaf & Aliases --> Views["Views: unified_edges & unified_nodes"]
    end

    subgraph Retrieval [Zero-Allocation GraphRAG Retrieval]
        Query[User Query] --> Seed[BM25 + Dense Vector Seed Vector]
        Seed --> PPR[SciPy CSR PageRank Diffusion]
        Views --> PPR
        PPR --> Rerank[In-Memory Dot-Product Bi-Encoder Rerank]
        Rerank --> Payload[Ranked Evidence + Ontological Context]
    end

    subgraph Materializer [Optional Asynchronous Materializer]
        Views -.->|Decoupled Markdown Pass| Wiki
    end
```

## How It Works

### 1. Single-Stage Ingestion & Leaf Ontology Extraction
CocoIndex continuously monitors `vault/raw/` for markdown file modifications and computes deterministic chunk boundaries.
- **Doc Integrity (Annotate, Never Mutate):** The text passed to spaCy equals source chunk text byte-for-byte (`doc.text == chunk_text`).
- **LLM-Primary Entity Extraction:** `en_core_web_sm` is loaded with `exclude=['ner']`; `spacy.NER.v3` (Qwen) runs as the primary extractor on every chunk across 8 non-predicate categories (`entities`, `concepts`, `claims`, `invariants`, `decisions`, `processes`, `sources`, `contexts`).
- **Markdown Structure:** Code spans are annotated in `doc.spans["code"]` and filtered exclusively in `leaf_taxonomy_compiler`.
- **Open IE Relations:** `spacy.REL.v1` extracts unconstrained relations into `doc._.rel`; predicates are normalized via `Token.lemma_`.
- **Line-Range Provenance:** Chunks carry 1-indexed line numbers (`line_start`, `line_end`) into the raw note for zero-copy on-demand disk hydration.

### 2. Ontological Convergence (Two-Tier Convergence Protocol)
Prevents vocabulary fragmentation across an append-only corpus without runtime LLM calls ([`docs/adr/ADR-003.md`](file:///home/harri/yaw/docs/adr/ADR-003.md), [`docs/adr/ADR-010.md`](file:///home/harri/yaw/docs/adr/ADR-010.md)):
- **Compiled C/Cython KnowledgeBase:** In-process `InMemoryLookupKB` synchronizes canonical entities and active alias candidate prior distributions $P(e \mid a) \in (0, 1]$ directly from SQLite (`leaf_nodes`, `aliases`).
- **Tier 1 (Direct Match):** Exact slug match, active alias table lookup, or Cython KB candidate resolution. 0 vectors, 0 LLM calls.
- **Tier 2 (Vector Near-Match):** In-process vector similarity > 0.80 using `en_core_web_lg` static 300d GloVe vectors. Auto-merges and records alias mapping. 0 LLM calls.
- **Fallback:** New canonical induction under plural universal root (`concepts:{slug}`). Zero external LLM pairwise adjudication.
- **Zero-Copy Projection:** Nodes and relations are unified via zero-copy SQLite views (`unified_edges`, `unified_nodes`).

### 3. Associative Retrieval & Graph Diffusion
Retrieval queries execute against SQLite and in-memory SciPy CSR matrices:
- **Dual-Channel Seeding:** Merges top BM25 lexical hits (`query_fts`) and top dense vector hits (`chunks_vec`).
- **PageRank Diffusion:** Runs 30 fixed power iterations over `unified_edges` with directional conductance, reciprocal back-edges ($0.5$), and analytical hub dampening ($\gamma = 0.5$).
- **In-Memory Dot Product Reranking:** Stored 768-dim chunk embeddings are retrieved directly from SQLite and rescored against query vectors via in-memory dot product ($S(c) = 0.5 \cdot \text{PPR}(c) + 0.5 \cdot \cos(q, c)$). Zero HTTP calls for chunk embeddings are made during retrieval.

---

## Quickstart

### Prerequisites
- Python 3.11+
- `uv` package manager
- Running Ollama instance with `embed-gemma-300m-FLM` and `Qwen3-4B-Instruct-2507-GGUF`

### 1. Ingest a Vault
```bash
uv run python yaw.py index --vault /path/to/vault
```
To force a full re-indexing of all chunks:
```bash
uv run python yaw.py index --vault /path/to/vault --force
```

### 2. Start the FastMCP Server
Run the background daemon over Streamable HTTP:
```bash
uv run python yaw.py serve --vault /path/to/vault --port 8765
```
Or start over stdio transport:
```bash
uv run python yaw.py serve --vault /path/to/vault --transport stdio
```

### 3. Query via CLI
```bash
uv run python yaw.py query "how does embedding gemma reranking work?" --vault /path/to/vault --top-k 3
```

### 4. Inspect Schema & Index Health
```bash
uv run python yaw.py schema --vault /path/to/vault
```

---

## MCP Client Configuration

### Streamable HTTP (Recommended)
```json
{
  "mcpServers": {
    "yaw": {
      "url": "http://127.0.0.1:8765/mcp"
    }
  }
}
```

### Stdio Transport
```json
{
  "mcpServers": {
    "yaw": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/path/to/yaw",
        "python",
        "yaw.py",
        "serve",
        "--vault",
        "/path/to/vault",
        "--transport",
        "stdio"
      ]
    }
  }
}
```

### Exposed FastMCP Tools

#### Production Client Interface (Karpathy Contract)
- `query_wiki(query: str, top_k: int = 3) -> str`: Full hybrid GraphRAG retrieval returning dual-layer payload (Ontological Context + Retrieved Ground-Truth Evidence).
- `ingest_raw(content: str, title: str | None = None, url: str | None = None) -> dict`: Ingests an external URL or raw text string into a timestamped, slugged Markdown note in `vault/raw/`.
- `compile_wiki(force: bool = False) -> dict`: Triggers background consolidation, B-Tree HAC splitting, hypernym induction, fast closed-world SHACL (<200ms), and commits detected violations to SQLite `compiled_violations`.
- `lint_wiki(query: str | None = None, force: bool = False) -> list[dict]`: Instantaneous O(1) query of pre-compiled contradictions and shape violations from SQLite (<5ms); live pySHACL validation override on `force=True`.

#### Developer Probes
- `find_seed(query: str, top_k: int = 3) -> list[dict]`: Stage 1 seed generator returning top FTS BM25 and vector hits.
- `read_document(path: str, line_start: int | None = None, line_end: int | None = None) -> str`: Zero-copy line-based slice reader into `vault/raw/`.
- `get_ontology_schema() -> dict`: Returns active namespaces, base predicates, and graph node/edge counts.

---

## Configuration

Settings are managed in `config.toml` (or environment variables prefixed with `YAW_`):

```toml
[mcp]
host = "127.0.0.1"
port = 8765

[vault]
raw_dir = "raw"
wiki_dir = "wiki"
db_path = ".cache/yaw.db"

[models]
embedding_model = "embed-gemma-300m-FLM"
chat_model = "Qwen3-4B-Instruct-2507-GGUF"
ollama_host = "http://127.0.0.1:11434"

[retrieval]
alpha = 0.85
top_k = 3
relevance_threshold = 0.10
hub_dampening_gamma = 0.5
```
