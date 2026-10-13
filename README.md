# YAW: HippoRAG 2 evaluation

YAW is a minimal workspace for evaluating upstream HippoRAG 2 against a small, messy corpus. The previous YAW application and architecture have been retired. Read [FOUNDATIONAL.md](FOUNDATIONAL.md) for scope and the next evaluation steps.

## Install

Use Python 3.12 and uv:

```powershell
uv sync --locked
uv run --locked python -c "from hipporag import HippoRAG; print('HippoRAG import OK')"
```

HippoRAG is the only direct dependency. It is pinned to upstream commit `d5c8329422e0a0b834a15874545cb6a74b4f9b26`, which declares version `2.0.0a5`. This is a source installation, not a published a5 wheel. Public PyPI offered a4 when checked on 2026-10-13. Git is required for installation.

Python 3.14.7 was tested during setup: dependency resolution fails because upstream requires `torch==2.5.1`, which has no CPython 3.14 wheel. Python 3.12 avoids overriding upstream dependencies. Revisit this constraint when upstream changes its pins.

## Verified setup (2026-10-13)

The locked environment installed successfully in this repository on Windows with CPython 3.12.14. Importing HippoRAG succeeded and package metadata reported 2.0.0a5. The eight retained source documents were verified unchanged by SHA-256. No model inference, corpus indexing or retrieval-quality evaluation was performed during this reset.

## First model evaluation

Start Lemonade at `http://localhost:13305/api/v1`. Confirm the model IDs below with its `/models` endpoint. Run this Python example after installation; it creates an index in `outputs/smoke`:

```python
from hipporag import HippoRAG

with HippoRAG(
    save_dir="outputs/smoke",
    llm_model_name="Qwen3-4B-Instruct-2507-GGUF",
    llm_base_url="http://localhost:13305/api/v1",
    embedding_model_name="embed-gemma-300m-FLM",
    embedding_provider="openai",
    embedding_base_url="http://localhost:13305/api/v1",
) as hipporag:
    hipporag.index(docs=["George Rankin is a politician."])
    results = hipporag.rag_qa(queries=["What is George Rankin's occupation?"])
    print(results)
```

Configure credentials through the upstream API's environment-variable options if the server requires them. Model listing alone does not establish extraction, embedding or QA compatibility. The example is an evaluation starting point, not evidence of a successful model run.

Existing source notes remain in `sample-llm-wiki/raw/`. They are not automatically indexed. Start with a few manually selected passages before deciding how to prepare the full corpus.

Upstream: [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG/tree/d5c8329422e0a0b834a15874545cb6a74b4f9b26).
