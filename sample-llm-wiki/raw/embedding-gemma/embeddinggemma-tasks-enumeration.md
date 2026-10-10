`model.prompts` is a `dict` mapping task names to prefix strings; `model.encode(text, prompt_name=...)` prepends the prefix before embedding. There are 14 entries, printed in the notebook. [1](#1-0) 

---

## The 14 `model.prompts` entries

| `prompt_name` | Prefix string | Purpose |
|---|---|---|
| `query` | `task: search result \| query: ` | Generic alias for embedding a search query |
| `document` | `title: none \| text: ` | Generic alias for embedding a document (no title) |
| `BitextMining` | `task: search result \| query: ` | Finding parallel/translated sentence pairs across languages (MTEB task family) |
| `Clustering` | `task: clustering \| query: ` | Embeddings optimized for grouping similar texts |
| `Classification` | `task: classification \| query: ` | Single-label zero-shot classification — embed text and label names, take argmax similarity |
| `InstructionRetrieval` | `task: code retrieval \| query: ` | Retrieval of code given an instruction |
| `MultilabelClassification` | `task: classification \| query: ` | Same prefix as `Classification`; intended for multi-label assignment |
| `PairClassification` | `task: sentence similarity \| query: ` | Deciding whether a pair of texts share a relation (e.g., duplicate questions) |
| `Reranking` | `task: search result \| query: ` | Scoring query–candidate pairs to re-order retrieved results |
| `Retrieval` | `task: search result \| query: ` | Generic retrieval (queries) |
| `Retrieval-query` | `task: search result \| query: ` | **RAG query side** — asymmetric retrieval |
| `Retrieval-document` | `title: none \| text: ` | **RAG document side** — pair with `Retrieval-query` |
| `STS` | `task: sentence similarity \| query: ` | Semantic textual similarity between two sentences |
| `Summarization` | `task: summarization \| query: ` | Matching texts to their summaries |

All names and prefixes are enumerated verbatim in both notebooks. [1](#1-0) [2](#1-1) 

## Input / output schema

**Input** — always the same: a string or `list[str]` passed to `model.encode(sentences, prompt_name=...)`. The prompt is just a prefix prepended to your raw text; there is no structured input schema. The one exception is document encoding, where you may embed the title into the prompt itself via the `prompt` argument: `prompt="title: {title} | text: "`. [3](#1-2) 

**Output** — always the same regardless of prompt: a fixed-size embedding vector per input sentence. The model is a `Gemma3TextModel` transformer → mean pooling → two `Dense` layers → `Normalize`, producing 768-dim embeddings (truncatable via `truncate_dim`, e.g. 256, thanks to Matryoshka training). [4](#1-3) [5](#1-4) 

Downstream, embeddings are consumed by `model.similarity(a, b)` (cosine by default, configurable e.g. `"dot"`) → similarity tensor → `argmax` for classification/retrieval. [6](#1-5) 

## Canonical usage patterns in the repo

- **RAG (asymmetric)**: `prompt_name="Retrieval-query"` for the query, `prompt_name="Retrieval-document"` (or `prompt="title: ... | text: "`) for docs. [7](#1-6) [8](#1-7) 
- **Classification**: encode both labels and inputs with `prompt_name="Classification"`, then `similarities.argmax(1)`. [9](#1-8) 
- **STS**: encode both sentences with `prompt_name="STS"` and compare directly. [10](#1-9) 

## Notes

- Several names share identical prefixes (e.g. `query`, `BitextMining`, `Retrieval`, `Retrieval-query`, `Reranking` all use `"task: search result | query: "`) — they are aliases matching MTEB task names, so `prompt_name` selection among them is functionally equivalent.
- The docstrings note per-task behavior details are on the EmbeddingGemma model card, which isn't in this repo — my "purpose" column for `BitextMining`/`Reranking`/`InstructionRetrieval` reflects standard MTEB semantics, not repo documentation. [11](#1-10)

### Citations

**File:** docs/embeddinggemma/inference-embeddinggemma-with-sentence-transformers.ipynb (L356-370)
```text
      "Available tasks:\n",
      " query: \"task: search result | query: \"\n",
      " document: \"title: none | text: \"\n",
      " BitextMining: \"task: search result | query: \"\n",
      " Clustering: \"task: clustering | query: \"\n",
      " Classification: \"task: classification | query: \"\n",
      " InstructionRetrieval: \"task: code retrieval | query: \"\n",
      " MultilabelClassification: \"task: classification | query: \"\n",
      " PairClassification: \"task: sentence similarity | query: \"\n",
      " Reranking: \"task: search result | query: \"\n",
      " Retrieval: \"task: search result | query: \"\n",
      " Retrieval-query: \"task: search result | query: \"\n",
      " Retrieval-document: \"title: none | text: \"\n",
      " STS: \"task: sentence similarity | query: \"\n",
      " Summarization: \"task: summarization | query: \"\n",
```

**File:** docs/embeddinggemma/inference-embeddinggemma-with-sentence-transformers.ipynb (L390-394)
```text
    "for sentence in [sentence_high, sentence_medium, sentence_low]:\n",
    "  print(\"🙋‍♂️\")\n",
    "  print(sentence)\n",
    "  embeddings = model.encode(sentence, prompt_name=\"STS\")\n",
    "  similarities = model.similarity(embeddings[0], embeddings[1])\n",
```

**File:** docs/embeddinggemma/inference-embeddinggemma-with-sentence-transformers.ipynb (L404-430)
```text
    "#### Use Case: Retrieval-Augmented Generation (RAG)\n",
    "\n",
    "For RAG systems, use the following `prompt_name` values to create specialized embeddings for your queries and documents:\n",
    "\n",
    "* **For Queries:** Use `prompt_name=\"Retrieval-query\"`.<br>\n",
    "    ```python\n",
    "    query_embedding = model.encode(\n",
    "        \"How do I use prompts with this model?\",\n",
    "        prompt_name=\"Retrieval-query\"\n",
    "    )\n",
    "    ```\n",
    "\n",
    "* **For Documents:** Use `prompt_name=\"Retrieval-document\"`. To further improve document embeddings, you can also include a title by using the `prompt` argument directly:<br>\n",
    "  * **With a title:**<br>\n",
    "    ```python\n",
    "    doc_embedding = model.encode(\n",
    "        \"The document text...\",\n",
    "        prompt=\"title: Using Prompts in RAG | text: \"\n",
    "    )\n",
    "    ```\n",
    "  * **Without a title:**<br>\n",
    "    ```python\n",
    "    doc_embedding = model.encode(\n",
    "        \"The document text...\",\n",
    "        prompt=\"title: none | text: \"\n",
    "    )\n",
    "    ```\n",
```

**File:** docs/embeddinggemma/inference-embeddinggemma-with-sentence-transformers.ipynb (L432-436)
```text
    "#### Further Reading\n",
    "\n",
    "* For details on all available EmbeddingGemma prompts, see the [model card](http://ai.google.dev/gemma/docs/embeddinggemma/model_card#prompt_instructions).\n",
    "* For general information on prompt templates, see the [Sentence Transformer documentation](https://sbert.net/examples/sentence_transformer/applications/computing-embeddings/README.html#prompt-templates).\n",
    "* For a demo of RAG, see the [Simple RAG example](https://github.com/google-gemini/gemma-cookbook/blob/main/Gemma/%5BGemma_3%5DRAG_with_EmbeddingGemma.ipynb) in the Gemma Cookbook.\n"
```

**File:** docs/embeddinggemma/inference-embeddinggemma-with-sentence-transformers.ipynb (L472-491)
```text
    "labels = [\"Billing Issue\", \"Technical Support\", \"Sales Inquiry\"]\n",
    "\n",
    "sentence = [\n",
    "  \"Excuse me, the app freezes on the login screen. It won't work even when I try to reset my password.\",\n",
    "  \"I would like to inquire about your enterprise plan pricing and features for a team of 50 people.\",\n",
    "]\n",
    "\n",
    "# Calculate embeddings by calling model.encode()\n",
    "label_embeddings = model.encode(labels, prompt_name=\"Classification\")\n",
    "embeddings = model.encode(sentence, prompt_name=\"Classification\")\n",
    "\n",
    "# Calculate the embedding similarities\n",
    "similarities = model.similarity(embeddings, label_embeddings)\n",
    "print(similarities)\n",
    "\n",
    "idx = similarities.argmax(1)\n",
    "print(idx)\n",
    "\n",
    "for example in sentence:\n",
    "  print(\"🙋‍♂️\", example, \"-> 🤖\", labels[idx[sentence.index(example)]])"
```

**File:** docs/embeddinggemma/inference-embeddinggemma-with-sentence-transformers.ipynb (L616-620)
```text
    "model = SentenceTransformer(model_id, truncate_dim=256, similarity_fn_name=\"dot\").to(device=device)\n",
    "embeddings = model.encode(words, prompt_name=\"STS\", normalize_embeddings=True)\n",
    "\n",
    "for idx, embedding in enumerate(embeddings):\n",
    "  print(f\"Embedding {idx+1}: {embedding.shape}\")\n",
```

**File:** Gemma/[Gemma_3]RAG_with_EmbeddingGemma.ipynb (L174-181)
```text
      "SentenceTransformer(\n",
      "  (0): Transformer({'max_seq_length': 2048, 'do_lower_case': False, 'architecture': 'Gemma3TextModel'})\n",
      "  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})\n",
      "  (2): Dense({'in_features': 768, 'out_features': 3072, 'bias': False, 'activation_function': 'torch.nn.modules.linear.Identity'})\n",
      "  (3): Dense({'in_features': 3072, 'out_features': 768, 'bias': False, 'activation_function': 'torch.nn.modules.linear.Identity'})\n",
      "  (4): Normalize()\n",
      ")\n",
      "Total number of parameters in the model: 307581696\n"
```

**File:** Gemma/[Gemma_3]RAG_with_EmbeddingGemma.ipynb (L250-264)
```text
      "Available tasks:\n",
      " query: \"task: search result | query: \"\n",
      " document: \"title: none | text: \"\n",
      " BitextMining: \"task: search result | query: \"\n",
      " Clustering: \"task: clustering | query: \"\n",
      " Classification: \"task: classification | query: \"\n",
      " InstructionRetrieval: \"task: code retrieval | query: \"\n",
      " MultilabelClassification: \"task: classification | query: \"\n",
      " PairClassification: \"task: sentence similarity | query: \"\n",
      " Reranking: \"task: search result | query: \"\n",
      " Retrieval: \"task: search result | query: \"\n",
      " Retrieval-query: \"task: search result | query: \"\n",
      " Retrieval-document: \"title: none | text: \"\n",
      " STS: \"task: sentence similarity | query: \"\n",
      " Summarization: \"task: summarization | query: \"\n"
```

**File:** Gemma/[Gemma_3]RAG_with_EmbeddingGemma.ipynb (L459-467)
```text
    "    # Encode the query for retrieval\n",
    "    query_embedding = model.encode(query, prompt_name=\"Retrieval-query\")\n",
    "\n",
    "    # Encode the document for similarity check\n",
    "    doc_texts = [\n",
    "        f\"title: {doc.get('title', 'none')} | text: {doc.get('content', '')}\"\n",
    "        for doc in candidates\n",
    "    ]\n",
    "    candidate_embeddings = model.encode(doc_texts)\n",
```
