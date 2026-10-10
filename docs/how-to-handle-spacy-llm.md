# How to Handle spaCy-LLM in YAW: Architecture, Protocols & Clean Integration

**Document Status:** Authoritative Architectural Guide  
**Location:** `docs/how-to-handle-spacy-llm.md`  
**Target Audience:** Core Engine Engineers & Autonomous Coding Agents  

---

## 1. Executive Summary & The Problem

In past iterations of the YAW codebase, the integration of `spacy-llm` accumulated convoluted patterns:
1. **Private Attribute Monkeypatching:** Creating `nlp.add_pipe("llm")` with placeholder OpenAI configs and then manually overwriting `pipe._model = custom_model`.
2. **Artificial Placeholders:** Declaring dummy `@llm_models = "spacy.REST.v1"` with fake API keys just to bypass initialization checks.
3. **Registry Bypass:** Failing to register local execution engines (Ollama, mock runners) within `spacy_llm.registry.llm_models`.
4. **Ad-Hoc String Parsing:** Manually hacking prompt strings rather than leveraging spaCy's native Jinja2 templating and structured tasks.

**Core Pipeline Architecture:** YAW is standardized on the official `spacy-llm` task ecosystem using `Qwen3-4B-Instruct-2507-GGUF` via the canonical `@registry.llm_models("yaw.Ollama.v1")` connector. To maintain pure off-the-shelf simplicity and zero hand-rolled extraction loops, exactly 5 native `spacy-llm` tasks govern extraction, relation mining, classification, linking, and description generation:
1. **`spacy.SpanCat.v3`**: Schema-free Open IE argument & predicate extraction with Chain-of-Thought prompting, saving overlapping/nested spans into `doc.spans["sc"]`.
2. **`spacy.REL.v1`**: Directed relational triple extraction between detected spans into `doc._.rel`.
3. **`spacy.TextCat.v3`**: Macro-domain classification into `doc.cats` (populating `note_meta.category` and providing a namespace prior).
4. **`spacy.EntityLinker.v1`**: EDC Self-Canonicalization & instance grounding into SQLite `leaf_nodes` (`span.kb_id_`).
5. **`spacy.Summarization.v1`**: Canonical node descriptions & wiki compilation into `doc._.description` (establishing that description == summary).

This guide documents the canonical conventions of `spacy-llm`, explains how models and tasks interact, and establishes the blueprint for maintaining clean, idiomatic pipelines in YAW.

---

## 2. Core Concepts: The Two Halves of `spacy-llm`

The `llm` component in spaCy (`spacy_llm.pipeline.llm.LLMWrapper`) coordinates exactly two decoupled subsystems:

```txt
┌────────────────────────────────────────────────────────────────────────┐
│                          spacy-llm Pipeline                            │
│                                                                        │
│   [Doc In] ──► 1. Task (Prompt Gen)  ──► [Prompts: list[list[str]]]   │
│                                                   │                    │
│                                                   ▼                    │
│   [Doc Out] ◄── 3. Task (Parse Res)  ◄── 2. Model (Execution)         │
│                 [doc.ents, doc._.rel,             │                    │
│                  doc.cats]              [Responses: list[list[str]]]   │
└────────────────────────────────────────────────────────────────────────┘
```

### 1. The Task (`@registry.llm_tasks`)
The **Task** defines *what* is extracted from the text:
- **`generate_prompts(docs: Iterable[Doc]) -> Iterable[Iterable[str]]`**: Renders Jinja2 prompt templates from document spans and task labels.
- **`parse_responses(docs: Iterable[Doc], responses: Iterable[Iterable[str]]) -> Iterable[Doc]`**: Parses model output (JSON, delimited lines) and populates native spaCy attributes (`doc.ents`, `doc.cats`, `doc._.rel`).
- Native tasks used in YAW:
  - `spacy.NER.v3`: High-recall domain entity extraction with optional descriptions and few-shot examples.
  - `spacy.REL.v1`: Directed entity relation extraction (`sub -> pred -> obj`) with customizable prompt templates.
  - `spacy.TextCat.v3`: Document-level category classification across configured namespaces.

### 2. The Model (`@registry.llm_models`)
The **Model** defines *how* prompts are executed against an LLM backend.
- A model in `spacy-llm` is any callable adhering to the model protocol:
  ```python
  def model(prompts: Iterable[Iterable[str]]) -> Iterable[Iterable[str]]: ...
  ```
  - **Outer Iterable**: Batches of documents.
  - **Inner Iterable**: Prompts for a specific document (e.g. sharded chunks or multi-task prompts).
- Models handle transport, timeouts, rate limits, and retries. They are completely agnostic to task logic or document annotations.

---

## 3. The Model Implementation Pattern

In official `spacy-llm` source (`spacy_llm/models/rest/base.py` and `openai/model.py`), REST models subclass `spacy_llm.models.rest.base.REST`:

### The Base REST Architecture
1. **Exponential Backoff (`retry`)**:
   Automatically retries requests upon network blips, 429 rate limits, or transient 5xx responses using `2^attempt * interval` backoff.
2. **Batch Request Processing**:
   Loops through outer document batches, sending prompts and collecting text completions into nested lists.
3. **Structured Error Handling**:
   Raises clean `ValueError` with API response messages rather than crashing with unhandled connection errors.

### The Canonical Ollama Model for YAW
Instead of monkeypatching `_model`, YAW registers a first-class `@registry.llm_models("yaw.Ollama.v1")`:

```python
from typing import Any, Callable, Iterable, List
import requests
from spacy_llm.registry import registry
from yaw import config


@registry.llm_models("yaw.Ollama.v1")
def make_ollama_model(
    name: str = "Qwen3-4B-Instruct-2507-GGUF",
    endpoint: str = "http://127.0.0.1:11434/api/chat",
    timeout_s: float = 60.0,
    temperature: float = 0.0,
) -> Callable[[Iterable[Iterable[str]]], Iterable[Iterable[str]]]:
    """Canonical REST model connector for local Ollama instances."""

    def _model(prompts: Iterable[Iterable[str]]) -> Iterable[Iterable[str]]:
        all_responses: List[List[str]] = []
        for doc_prompts in prompts:
            doc_res: List[str] = []
            for prompt_str in doc_prompts:
                payload = {
                    "model": name,
                    "messages": [{"role": "user", "content": prompt_str}],
                    "stream": False,
                    "options": {"temperature": temperature},
                }
                resp = requests.post(endpoint, json=payload, timeout=timeout_s)
                resp.raise_for_status()
                data = resp.json()
                doc_res.append(data.get("message", {}).get("content", ""))
            all_responses.append(doc_res)
        return all_responses

    return _model
```

---

## 4. The 5 Canonical `spacy-llm` Tasks in YAW

To prevent custom hand-rolled loops and fragile monkeypatching, YAW builds its entire NLP layer using 5 official, off-the-shelf tasks registered in `@llm_tasks`:

### 1. Open IE Span Extraction (`spacy.SpanCat.v3`)
- **Factory:** `@llm_tasks("spacy.SpanCat.v3")`
- **Output:** Stored in `Doc.spans[spans_key]` (`spans_key="sc"` by default).
- **Core Benefit:** Unlike `spacy.NER.v3`, `SpanCat.v3` natively supports **nested and overlapping spans**, enabling high-order propositions (`claims`, `invariants`) to contain nested concept arguments without collision.
- **Config:**
  ```ini
  [components.llm_spancat]
  factory = "llm"
  [components.llm_spancat.task]
  @llm_tasks = "spacy.SpanCat.v3"
  labels = ["subject", "relation_predicate", "object"]
  spans_key = "sc"
  [components.llm_spancat.model]
  @llm_models = "yaw.Ollama.v1"
  name = "Qwen3-4B-Instruct-2507-GGUF"
  ```

### 2. Relation Triple Extraction (`spacy.REL.v1`)
- **Factory:** `@llm_tasks("spacy.REL.v1")`
- **Output:** Stored in `Doc._.rel` as directed triples: `[{"dep": int, "dest": int, "relation": str}]`.
- **Open IE Template Requirement:**
  The default `rel.v1.jinja` template enumerates `{%- for label in labels -%}`. If `labels` is empty or omitted without few-shot examples, the prompt presents an empty label list, causing LLMs to extract 0 relations.
  To perform schema-free Open IE relation extraction, pass `template = OPEN_IE_REL_TEMPLATE` (from `src/yaw/nlp/templates.py`), which instructs the model to extract emergent, directed semantic relations without a closed label enumeration loop.
- **Native Normalizer Parameter (`normalizer = ...`):**
  `make_rel_task` inherits from `BuiltinTaskWithLabels`, which natively accepts `normalizer: Optional[Callable[[str], str]]`. In standard spaCy, normalizers are registered in `@spacy.registry.misc`. We leverage this official extension point to inject aggressive spaCy-based lemmatization and decamelization directly into the task.
- **spaCy-Based Predicate Normalizer Implementation:**
  Instead of fragile string preprocessing and regex replacement chains, spaCy's tokenizer natively handles punctuation, whitespace, and hyphens. By registering camelCase boundary detection into `nlp.tokenizer.infix_finditer` using `compile_infix_regex`, all segmentation and filtering is handled natively by spaCy's Cython tokenizer and `tok.is_alpha`:
  ```python
  from typing import Callable
  import spacy
  from spacy.util import compile_infix_regex, registry


  @registry.misc("yaw.PredicateNormalizer.v1")
  def make_predicate_normalizer(
      base_model: str = "en_core_web_sm",
  ) -> Callable[[str], str]:
      """Native spaCy predicate normalizer.
      Configures tokenizer with camelCase infixes and extracts alpha lemmas.
      E.g.: 'isComposedOf' -> 'be compose of', 'runs_on' -> 'run on', 'part-of' -> 'part of'.
      """
      nlp = spacy.load(base_model, disable=["ner"])
      # Add camelCase lookahead boundary to spaCy's native infix rules
      infixes = tuple(nlp.Defaults.infixes) + (r"(?<=[a-z])(?=[A-Z])",)
      nlp.tokenizer.infix_finditer = compile_infix_regex(infixes).finditer

      def _normalize(predicate: str) -> str:
          doc = nlp(predicate)
          return " ".join(tok.lemma_.lower() for tok in doc if tok.is_alpha)

      return _normalize
  ```
- **Declarative Config (`config.cfg`):**
  ```ini
  [components.llm_rel]
  factory = "llm"
  [components.llm_rel.task]
  @llm_tasks = "spacy.REL.v1"
  template = """${templates:open_ie_rel}"""
  [components.llm_rel.task.normalizer]
  @misc = "yaw.PredicateNormalizer.v1"
  base_model = "en_core_web_sm"
  [components.llm_rel.model]
  @llm_models = "yaw.Ollama.v1"
  name = "Qwen3-4B-Instruct-2507-GGUF"
  ```
- **Programmatic Config:**
  ```python
  rel_cfg = {
      "validate_types": False,
      "task": {
          "@llm_tasks": "spacy.REL.v1",
          "template": OPEN_IE_REL_TEMPLATE,
          "normalizer": {
              "@misc": "yaw.PredicateNormalizer.v1",
              "base_model": "en_core_web_sm",
          },
      },
      "model": _model_to_config(model),
  }
  nlp.add_pipe("llm", name="llm_rel", config=rel_cfg)
  ```
- **Parser Hooking (`parse_responses`):**
  Upstream `spacy.REL.v1`'s default `parse_responses_v1` parses JSON without invoking `task._normalizer`. To ensure the normalizer is applied at extraction time, YAW registers a response parser (or calls `pipe.task._normalizer` in the leaf compiler) that sets `rel_item.relation = task._normalizer(rel_item.relation)` before emitting `Doc._.rel`.

### 3. Ontological Typing (`spacy.TextCat.v3`)
- **Factory:** `@llm_tasks("spacy.TextCat.v3")`
- **Output:** Stored in `Doc.cats` (`{label: probability}`).
- **Ontological Grounding (Categories = Types):**
  - **The Chunk Categorization Fallacy:** A ~1000-character markdown chunk contains multiple entities of disparate types (a chunk may mention a `decision`, an `invariant`, a `concept`, and a `process` in consecutive sentences). Classifying an entire chunk into `["concepts", "claims", "invariants", ...]` commits a Rylean category mistake: assigning an entity's ontological type to an arbitrary document slice.
  - **Entity-Level Typing:** In formal ontology (BFO ISO/IEC 21838-2, DOLCE), `concepts`, `claims`, `invariants`, `decisions`, `processes`, `sources`, and `contexts` are **Universals (Types)** that classify **Particulars (Entity Instances)**.
  - **Pipeline Execution:** In EDC (Extract-Define-Canonicalize), `spacy.TextCat.v3` is applied directly to the entity's generated definition (`Doc._.description` from `spacy.Summarization.v1`). Classifying the semantic definition cleanly yields its ontological type:
    - *"A grounded architectural constraint guaranteeing byte offset equivalence..."* $\to$ `invariants`
    - *"An architectural choice to serve Model Context Protocol endpoints..."* $\to$ `decisions`
    - *"A graph-ranking algorithm computing the stationary probability distribution..."* $\to$ `concepts`
  - *(Optional document-level classification for `note_meta.category` classifies macro-thematic domains such as `architecture`, `graphrag`, `nlp`, rather than entity types).*
- **Config:**
  ```ini
  [components.llm_textcat]
  factory = "llm"
  [components.llm_textcat.task]
  @llm_tasks = "spacy.TextCat.v3"
  labels = ["concepts", "claims", "invariants", "decisions", "processes", "sources", "contexts"]
  [components.llm_textcat.model]
  @llm_models = "yaw.Ollama.v1"
  name = "Qwen3-4B-Instruct-2507-GGUF"
  ```

### 4. EDC Instance Convergence (`spacy.EntityLinker.v1`)
- **Factory:** `@llm_tasks("spacy.EntityLinker.v1")`
- **Output:** Assigns canonical SQLite URIs to `span.kb_id_` / `token.ent_kb_id_`.
- **Mechanism:** Integrates `spacy.CandidateSelector.v1` querying `leaf_nodes` via `embed-gemma-300m-FLM` vector distance. Auto-merges on $\ge 0.95$ cosine similarity; prompts Qwen for gray-zone candidate adjudication ($[0.80, 0.95)$).
- **Config:**
  ```ini
  [components.llm_entity_linker]
  factory = "llm"
  [components.llm_entity_linker.task]
  @llm_tasks = "spacy.EntityLinker.v1"
  [components.llm_entity_linker.model]
  @llm_models = "yaw.Ollama.v1"
  name = "Qwen3-4B-Instruct-2507-GGUF"
  temperature = 0.1
  ```

### 5. Node Descriptions & Wiki Summarization (`spacy.Summarization.v1`)
- **Factory:** `@llm_tasks("spacy.Summarization.v1")`
- **Core Principle (Description at Indexing Time):**
  1. Generates concise node definitions during EDC Phase 2 (Define), populating `leaf_nodes.definition` and feeding into `spacy.TextCat.v3` for ontological typing.
  2. Materializes `vault/wiki/` pages deterministically from SQLite row data (`unified_nodes.definition` frontmatter + `unified_edges` wikilinks) without redundant LLM authoring or synthesis passes.
- **Config:**
  ```ini
  [components.llm_summarization]
  factory = "llm"
  [components.llm_summarization.task]
  @llm_tasks = "spacy.Summarization.v1"
  field = "description"
  max_n_words = 50
  [components.llm_summarization.model]
  @llm_models = "yaw.Ollama.v1"
  name = "Qwen3-4B-Instruct-2507-GGUF"
  ```

---

## 5. FastMCP Protocol & Retrieval Contract (Zero Query-Time Synthesis)

In accordance with [`FOUNDATIONAL.md §5.3`](file:///home/harri/yaw/FOUNDATIONAL.md#component-3-retrieval-flow-graphragretriever) and Invariant 5, `query_wiki(query)` executes **purely deterministic GraphRAG retrieval with zero query-time LLM synthesis**:

1. **Dual-Layer Context Extraction:** The SciPy PPR engine selects the top winning concept nodes and zero-copy hydrates winning raw chunk byte slices directly from disk byte offsets.
2. **Deterministic Response Presentation:** Retrieval formats the token-dense Bridging Subgraph directly using indexing-time definitions (`Doc._.description`) and hydrated evidence chunks:
   - `# Result: [<k>] <rel_path>:<start>-<end>#chunk:<ordinal>`
   - `## Ontological Context`: active concept definitions and typed relations from SQLite.
   - `## Evidence Chunk`: raw hydrated markdown chunk.
3. **Zero Hallucination Guarantee:** No chat LLM prompt is constructed or evaluated during `query_wiki`. Qwen is scoped strictly to **indexing-time** EDC gray-zone adjudication (`canon_verdicts`) and compiled wiki page generation (`compile_wiki`).

---

## 6. Testing Patterns: How to Avoid Tautologies & Private Attribute Hacks

### The Anti-Pattern: Overwriting `_model`
```python
# AVOID THIS:
llm_pipe = nlp.add_pipe("llm", config=dummy_cfg)
llm_pipe._model = mock_callable  # Private attribute access! Fragile!
```

### The Canonical Pattern: Model Fixture / Registry Injection
1. **Direct Model Injection:** The `model` parameter of the `llm` pipe factory can receive any function/callable directly:
   ```python
   def test_pipeline_with_mock():
       mock_model = MockModel(['{"dep": 0, "dest": 1, "relation": "employed_at"}'])
       nlp = build_leaf_pipeline(model=mock_model)
       doc = nlp("Marie Curie worked at Sorbonne.")
       assert len(doc._.rel) == 1
   ```
2. **Ephemeral Registry Models:** For tests requiring full config serialization or `assemble()`, register a temporary test model in `registry.llm_models`:
   ```python
   @pytest.fixture
   def register_mock_llm():
       @registry.llm_models("test.MockModel.v1")
       def _make_mock(responses: list[str]):
           return lambda prompts: [[r] for r in responses]

       yield
   ```

---

## 7. Key Takeaways & Invariants

1. **Never Touch Private `_model`**: Always pass models through registry `@llm_models` or cleanly through the component factory signature.
2. **Never Hardcode Dummy Providers**: No fake `spacy.OpenAI.v1` declarations when using local Ollama models.
3. **Keep Tasks & Models Decoupled**: Tasks own prompt rendering and document annotation; models own HTTP transport, backoff, and text generation.
4. **Use Jinja2 for Prompt Variations**: Leverage `template` overrides for Open IE rather than modifying task code.
5. **Rely on Native spaCy Types**: Store entities in `doc.ents`, document classification in `doc.cats`, and relation triples in `doc._.rel`.
