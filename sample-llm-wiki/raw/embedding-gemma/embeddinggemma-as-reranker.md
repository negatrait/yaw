# DeepWiki Q&A with Code Context for Repository: google-gemini/gemma-cookbook
## Q1
How does one configure embeddinggemma for reranking?
### Answer
Use `prompt_name="Reranking"` when encoding the query, then score candidate documents with `model.similarity`. EmbeddingGemma is a bi-encoder (no cross-encoder), so reranking is done by embedding similarity, not joint encoding.

---

## Configuration

### 1. Load the model

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("google/embeddinggemma-300M").to(device)
```

The model ships with built-in task prompts, including `Reranking`, which maps to `"task: search result | query: "` — the same prefix used for retrieval queries. gemma-cookbook:191-192 gemma-cookbook:250-264 

You can verify available prompts via `model.prompts`. gemma-cookbook:269-271 

### 2. Encode with the appropriate prompts

| Side | Config |
|---|---|
| Query | `model.encode(query, prompt_name="Reranking")` |
| Documents | `prompt_name="Retrieval-document"` or `prompt="title: {title} \| text: "` / `"title: none \| text: "` |

The prompt-argument mechanism is documented in the inference guide. gemma-cookbook:321-342 

### 3. Score and rank

Following the pattern used in the repo for query–document scoring:

```python
query_emb = model.encode(query, prompt_name="Reranking")
doc_embs = model.encode(documents, prompt_name="Retrieval-document")

scores = model.similarity(query_emb, doc_embs)[0]
ranked = sorted(zip(documents, scores.tolist()), key=lambda x: -x[1])
```

This mirrors the `get_scores` helper in the fine-tuning notebook, which encodes query and docs, computes `model.similarity`, and prints per-document scores. gemma-cookbook:270-279 

## Notes

- The `Reranking` prompt is identical to `Retrieval`/`Retrieval-query` (`"task: search result | query: "`), so in practice reranking with EmbeddingGemma behaves like re-scoring with retrieval embeddings — it's a bi-encoder similarity rank, not a true cross-encoder reranker. gemma-cookbook:356-370 
- If you don't specify `prompt_name`, the model uses its `default_prompt_name` or no prompt at all, so set it explicitly. gemma-cookbook:338-342 
- The RAG notebook itself only demonstrates retrieval (`Retrieval-query` / `Retrieval-document`); it does not include a dedicated reranking example. gemma-cookbook:207-231
### Citations
**File:** Gemma/[Gemma_3]RAG_with_EmbeddingGemma.ipynb (L191-192)
```text
    "model_id = \"google/embeddinggemma-300M\"\n",
    "model = SentenceTransformer(model_id).to(device=device)\n",
```
**File:** Gemma/[Gemma_3]RAG_with_EmbeddingGemma.ipynb (L207-231)
```text
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
**File:** Gemma/[Gemma_3]RAG_with_EmbeddingGemma.ipynb (L269-271)
```text
    "print(\"Available tasks:\")\n",
    "for name, prefix in model.prompts.items():\n",
    "  print(f\" {name}: \\\"{prefix}\\\"\")"
```
**File:** docs/embeddinggemma/inference-embeddinggemma-with-sentence-transformers.ipynb (L321-342)
```text
    "#### How to Apply Prompts\n",
    "\n",
    "You can apply a prompt during inference in three ways.\n",
    "\n",
    "1.  **Using the `prompt` argument**<br>\n",
    "    Pass the full prompt string directly to the `encode` method. This gives you precise control.\n",
    "    ```python\n",
    "    embeddings = model.encode(\n",
    "        sentence,\n",
    "        prompt=\"task: sentence similarity | query: \"\n",
    "    )\n",
    "    ```\n",
    "2.  **Using the `prompt_name` argument**<br>\n",
    "    Select a predefined prompt by its name. These prompts are loaded from the model's configuration or during its initialization.\n",
    "    ```python\n",
    "    embeddings = model.encode(sentence, prompt_name=\"STS\")\n",
    "    ```\n",
    "3.  **Using the Default Prompt**<br>\n",
    "    If you don't specify either `prompt` or `prompt_name`, the system will automatically use the prompt set as `default_prompt_name`, if no default is set, then no prompt is applied.\n",
    "    ```python\n",
    "    embeddings = model.encode(sentence)\n",
    "    ```\n"
```
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
**File:** docs/embeddinggemma/fine-tuning-embeddinggemma-with-sentence-transformers.ipynb (L270-279)
```text
    "def get_scores(query, documents):\n",
    "  # Calculate embeddings by calling model.encode()\n",
    "  query_embeddings = model.encode(query, prompt_name=task_name)\n",
    "  doc_embeddings = model.encode(documents, prompt_name=task_name)\n",
    "\n",
    "  # Calculate the embedding similarities\n",
    "  similarities = model.similarity(query_embeddings, doc_embeddings)\n",
    "\n",
    "  for idx, doc in enumerate(documents):\n",
    "    print(\"Document: \", doc, \"-> 🤖 Score: \", similarities.numpy()[0][idx])\n",
```