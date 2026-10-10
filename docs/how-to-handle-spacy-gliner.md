# [DEPRECATED] spaCy-GLiNER Architecture

> [!WARNING]
> This guide is deprecated. In accordance with architectural decisions to avoid third-party wrappers, arbitrary internal sub-chunking loops, and custom glue code, YAW uses **100% official off-the-shelf `spacy-llm` components** (`spacy.SpanCat.v3`, `spacy.REL.v1`, `spacy.TextCat.v3`, `spacy.EntityLinker.v1`, and `spacy.Summarization.v1`) backed by `Qwen3-4B-Instruct-2507-GGUF` via `@registry.llm_models("yaw.Ollama.v1")`.
>
> Please consult the authoritative guide: [`docs/how-to-handle-spacy-llm.md`](how-to-handle-spacy-llm.md).
