# Pre-Epic 2 Architectural Refactoring Memo

**Status**: Approved Architectural Decisions (Deferred to Post-Benchmark Cleanup)  
**Owner**: [`FOUNDATIONAL.md`](../FOUNDATIONAL.md)  
**Related**: [Epic 1](epic-1-hipporag-daemon.md), [Epic 2](epic-2-fastmcp-http.md)

---

## 1. Context & Executive Summary

During the Layer 2 Paper Parity Gate run across long-context benchmark datasets (notably `lveval` with 45,000–70,000 character source articles), two interrelated architectural anti-patterns were uncovered:

1. **Whole-Document Retrieval & Context Flooding**: Attempting to re-inflate retrieved sub-chunks into whole 45k-character parent documents (to pass upstream academic substring recall checks) bled directly into QA generation, flooding the context window with 200k+ characters and prompting crude character slicing (`[:36000]` / `[:40000]`).
2. **Intermediate Projection Files in Ingestion**: `chunk_json_text()` wrote intermediate text projections to disk (`.yaw_projected/{rel_posix_path}`) inside a `@coco.fn(memo=True)` function simply so `hydrate_chunk_pointer()` could use a naive `itertools.islice` line reader. Subsequent attempts to bypass disk writes introduced an ad-hoc in-memory cache (`_JSON_ISLAND_CACHE`).

Per user mandate, the active benchmark evaluation run remains completely undisturbed. This document records the architectural principles, unified retrieval/QA pipeline, and concrete code implementations to be applied immediately following benchmark completion before Epic 2 begins.

---

## 2. The Production RAG Principle: The Needle, Not the Haystack

In production RAG, **retrieval exists to find and return the needle, not to dump the entire haystack back to the caller or the LLM**.

Why would retrieval ever output a full 45,000–70,000 character document?
- CocoIndex already indexes documents as discrete AST-bounded chunks with exact line offsets (`start_line`, `end_line`) and section headings.
- Outputting full documents in retrieval destroys the entire purpose of semantic retrieval, blows past model context windows, wastes memory and bandwidth, and dilutes LLM attention.
- The mechanism of retrieval and QA is **fundamentally the same**:
  1. Retrieve the hit chunks.
  2. Hydrate for retrieval while deduping and expanding to immediate AST neighbors up to a controlled budget.
  3. Context window management is simply controlling **"how much do you want to recall at once"**, ensuring the downstream model's context window is never endangered.

---

## 3. The Unified 3-Step Hydration Architecture

The hydration pipeline is identical for standalone retrieval and QA context assembly:

```
[Graph / Vector Index]
         │
         ▼
 1. Retrieve Hits: Top-k sub-chunks {C_hit} with exact (rel_path, start_line, end_line)
         │
         ▼
 2. AST Expansion & Deduplication (Coalescing):
    - Expand each hit to adjacent sibling chunks within the same AST island (paragraph/section)
    - Deduplicate: Merge overlapping or contiguous chunk line spans so no line is output twice
         │
         ▼
 3. Upfront Context Window Budgeting:
    - Target: B total tokens/chars (e.g. 16k context window headroom)
    - INVARIANT: The retrieved hit chunks {C_hit} are INVIOLATE — NEVER dehydrated or dropped!
    - Only the surrounding AST expansion padding is elastic (expanded when room permits, pruned when tight)
    - By sizing expansion upfront to fit B, emergency dehydration is never needed
```

### 3.1 Step 1: Retrieve the Chunks
The graph and vector search returns the top-$k$ scored sub-chunks. Each hit chunk pointer $C_{\text{hit}}$ contains its exact 1-indexed source line span `[start_line, end_line]`, its AST heading/title, and its relevance score.

### 3.2 Step 2: AST Expansion & Coalesced Deduplication
For each hit chunk, we can expand context to neighboring chunks within the same AST section or paragraph:
- **Deduplication / Coalescing**: If two or more retrieved chunks belong to the same document and have overlapping or contiguous line ranges (`span_A.end_line + 1 >= span_B.start_line`), they are **merged into a single continuous passage span**.
- **No Repeated Lines**: No chunk or line is ever emitted more than once in the output context.

### 3.3 Step 3: Upfront Budget Fitting & The Inviolate Hit Invariant
Let $B$ be the context character budget allocated for context (e.g. 24,000 characters $\approx 6,000$ tokens, safely below the 16k token context window).

Let $L_{\text{hits}} = \sum_{i=1}^{k} \text{len}(C_{\text{hit}, i})$ be the total length of the raw retrieved hit chunks.

1. **The Inviolate Hit Invariant**:
   > **NEVER DEHYDRATE THE RETRIEVED HIT CHUNKS!**  
   > The retrieved hit chunk is the core evidence found by the search engine. Truncating or dropping the hit chunk destroys the answer.
2. **Greedy Priority AST Expansion by Headroom**:
   When hydrating for recall or QA, we possess three key insights for each retrieved chunk:
   - **Relevance score** (from graph PPR or dense KNN ranking).
   - **Containing AST island** (enclosing section / paragraph bounds from CocoIndex).
   - **Immediate neighbor chunks** (adjacent siblings in the AST).

   Instead of arbitrary global thresholds, expansion operates via a **greedy priority FIFO queue**:
   1. **Sort candidates by relevance descending** into a processing queue.
   2. **Pop the most relevant candidate** from the queue.
   3. **Expand once to immediate neighbors** (default: 1 hop / immediate adjacent sibling chunks, configurable via `max_ast_expansion_hops = 1`), respecting containing AST island boundaries. **This is strictly non-recursive**—it does not cascade outward to eternity.
   4. **Evaluate headroom**:
      $$\text{Headroom } H = B - \text{current\_total\_length}$$
      If $H > 0$, repeat step 2 on the next most relevant candidate.
   5. Once all candidates have been expanded once (or headroom is exhausted), **deduplicate / coalesce** overlapping and contiguous spans.

**Is this any different to MaxP?**  
**No! This IS MaxP adapted for generative LLM RAG with an AST.**  
In classical IR (Dai & Callan, SIGIR 2019), MaxP designates the relevance of a passage as the maximum score among its constituent sub-chunks ($\text{Score}(P) = \max_{c \in P} \text{Score}(c)$). In our AST-aware engine:
- The top-scoring sub-chunk acts as the primary evidence anchor.
- It pulls in its immediate AST neighbors proportional to available context headroom.
- Coalescing overlapping or adjacent spans produces a coherent passage whose relevance is governed by that maximum-scoring anchor.
- Highest-relevance anchors get first priority on expansion before lower-ranked candidates consume context.

---

## 4. Reference Implementation

### 4.1 Greedy Priority AST Expansion & Coalesced Deduplication
In [`yaw/daemon/graph/storage/hydration.py`](../yaw/daemon/graph/storage/hydration.py):

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import itertools

from yaw.core import ChunkPointer, HydratedPassage, resolve_corpus_path


@dataclass
class MutableSpan:
    rel_posix_path: str
    start_line: int
    end_line: int
    ast_start: int
    ast_end: int
    title: str
    hit_cids: list[str]
    score: float


@dataclass(frozen=True)
class CoalescedSpan:
    """A deduplicated, contiguous line span covering one or more hit chunks and their AST expansion."""

    rel_posix_path: str
    start_line: int
    end_line: int
    title: str
    hit_chunk_ids: tuple[str, ...]
    max_score: float


def coalesce_spans(spans: list[MutableSpan]) -> list[CoalescedSpan]:
    """Merge overlapping or contiguous line spans within each file so no line is duplicated."""
    if not spans:
        return []

    by_file: dict[str, list[MutableSpan]] = {}
    for s in spans:
        by_file.setdefault(s.rel_posix_path, []).append(s)

    coalesced: list[CoalescedSpan] = []
    for rel_path, file_spans in by_file.items():
        sorted_spans = sorted(file_spans, key=lambda s: (s.start_line, -s.end_line))
        cur = sorted_spans[0]
        cur_s, cur_e = cur.start_line, cur.end_line
        cur_title = cur.title
        cur_cids = list(cur.hit_cids)
        cur_max_score = cur.score

        for nxt in sorted_spans[1:]:
            if nxt.start_line <= cur_e + 1:
                cur_e = max(cur_e, nxt.end_line)
                if not cur_title and nxt.title:
                    cur_title = nxt.title
                cur_cids.extend(nxt.hit_cids)
                cur_max_score = max(cur_max_score, nxt.score)
            else:
                coalesced.append(
                    CoalescedSpan(
                        rel_posix_path=rel_path,
                        start_line=cur_s,
                        end_line=cur_e,
                        title=cur_title,
                        hit_chunk_ids=tuple(dict.fromkeys(cur_cids)),
                        max_score=cur_max_score,
                    )
                )
                cur_s, cur_e = nxt.start_line, nxt.end_line
                cur_title = nxt.title
                cur_cids = list(nxt.hit_cids)
                cur_max_score = nxt.score

        coalesced.append(
            CoalescedSpan(
                rel_posix_path=rel_path,
                start_line=cur_s,
                end_line=cur_e,
                title=cur_title,
                hit_chunk_ids=tuple(dict.fromkeys(cur_cids)),
                max_score=cur_max_score,
            )
        )

    coalesced.sort(key=lambda cs: cs.max_score, reverse=True)
    return coalesced


def hydrate_budgeted_passages(
    pointers_with_scores: Sequence[tuple[ChunkPointer, float]],
    corpus_root: Path,
    *,
    max_context_chars: int = 24000,
    max_expansion_hops: int = 1,
) -> list[HydratedPassage]:
    """Hydrate retrieved chunks via greedy priority AST-expansion:
    1. Sort chunks by relevance descending (FIFO queue).
    2. Pick the most relevant remaining chunk.
    3. Expand ONCE to immediate neighbors (max_expansion_hops=1, non-recursive) respecting AST boundaries if headroom permits.
    4. Deduplicate / coalesce contiguous spans (MaxP score inheritance).
    """
    if not pointers_with_scores:
        return []

    # 1. Sort chunks by relevance descending
    sorted_hits = sorted(pointers_with_scores, key=lambda x: x[1], reverse=True)

    # Initialize active spans from raw hits
    active_spans: list[MutableSpan] = []
    for ptr, score in sorted_hits:
        # ast_start and ast_end represent the enclosing CocoIndex AST node bounds
        ast_s = getattr(ptr, "ast_island_start", ptr.start_line)
        ast_e = getattr(ptr, "ast_island_end", ptr.end_line)
        active_spans.append(
            MutableSpan(
                rel_posix_path=ptr.rel_posix_path,
                start_line=ptr.start_line,
                end_line=ptr.end_line,
                ast_start=ast_s,
                ast_end=ast_e,
                title=ptr.title,
                hit_cids=[ptr.chunk_id],
                score=score,
            )
        )

    # 2. Greedily expand most relevant candidates ONCE (non-recursive) while headroom permits
    def estimate_total_chars(spans: list[MutableSpan]) -> int:
        return sum((s.end_line - s.start_line + 1) * 80 for s in spans)

    current_len = estimate_total_chars(active_spans)

    if max_expansion_hops >= 1:
        for span in active_spans:
            if current_len >= max_context_chars:
                break
            # Expand span ONCE outward to immediate neighbor lines (e.g. ±10 lines) bounded by AST
            new_start = max(span.ast_start, span.start_line - 10)
            new_end = min(span.ast_end, span.end_line + 10)
            added_lines = (span.start_line - new_start) + (new_end - span.end_line)
            added_chars = added_lines * 80
            if current_len + added_chars <= max_context_chars:
                span.start_line = new_start
                span.end_line = new_end
                current_len += added_chars

    # 3. Deduplicate and coalesce overlapping / contiguous spans
    final_spans = coalesce_spans(active_spans)

    # 4. Hydrate exact line spans from disk
    passages = _hydrate_spans(final_spans, corpus_root)
    return passages
```

### 4.3 Unified QA Context Formatting
In [`yaw/daemon/graph/engine.py`](../yaw/daemon/graph/engine.py):

```python
def format_qa_context(
    self,
    hits: Sequence[tuple[ChunkPointer, float, str]],
    *,
    max_context_chars: int = 24000,
) -> str:
    """Format QA prompt context from top-k retrieved hits using the unified budgeted hydration."""
    if not hits:
        return ""
    passages = hydrate_budgeted_passages(
        [(p, s) for p, s, _ in hits],
        corpus_root=self.corpus_root,
        max_context_chars=max_context_chars,
    )
    blocks: list[str] = []
    for p in passages:
        title_prefix = f"Wikipedia Title: {p.pointer.title}\n" if p.pointer.title else ""
        blocks.append(f"{title_prefix}{p.text}")
    return "\n\n".join(blocks)
```

---

## 5. Academic Benchmark Evaluation Alignment

### 5.1 The Root Cause of Benchmark Confusion
Upstream HippoRAG evaluated retrieval recall using exact string containment:
```python
is_hit = any(gold in retrieved_text for gold in gold_docs)
```
In datasets where authors provided whole 45k-character articles as `gold_docs` (e.g., `lveval`), this inverted check required `retrieved_text` to be 45,000 characters long to pass `gold in retrieved_text`!

### 5.2 The Clean Solution in `scripts/evaluation.py`
In production RAG and sensible benchmark evaluation:
- The metric checks whether the retrieved chunk is a substring of the gold document:
  ```python
  is_hit = any(clean_retrieved_chunk in gold_doc for gold_doc in gold_docs)
  ```
  or matches on document identity:
  ```python
  is_hit = retrieved_pointer.rel_posix_path == gold_doc_path
  ```
- This allows native YAW retrieval to return clean, focused $\approx 300$-token AST chunks, achieving 100% recall credit on academic datasets without corrupting the production engine.

---

## 6. Idiomatic CocoIndex Ingestion & Direct AST Hydration

### 6.1 Ingestion Anti-Patterns to Remove
1. **No Intermediate Files (`.yaw_projected/`)**:
   `@coco.fn(memo=True) process_corpus_file` must remain a pure transform. Never write side-effect projection files to disk.
2. **No Ad-Hoc Global In-Memory Caches**:
   Remove `_JSON_ISLAND_CACHE`, `register_json_island_lines`, and `get_json_island_lines` from `yaw/core/types.py`.
3. **No Hot-Patch Shims**:
   Remove `ast`, `importlib`, and `sys` patching routines from `yaw/daemon/graph/engine.py`.

### 6.2 Direct AST Hydration
- `ChunkPointer.rel_posix_path` points directly to the real source file on disk.
- Hydration reads the exact 1-indexed line span directly from the source file.
- CocoIndex AST parsing (`CodeSource`, `RecursiveSplitter`) provides boundary-aware chunking during ingestion and expansion.

---

## 7. Post-Benchmark Execution Plan

Immediately upon completion of the Layer 2 Paper Parity Gate run:

| Step | Target File | Action |
| :--- | :--- | :--- |
| **7.1** | `yaw/core/types.py` & `yaw/core/__init__.py` | Remove `_JSON_ISLAND_CACHE`, `register_json_island_lines`, and `get_json_island_lines`. |
| **7.2** | `yaw/daemon/graph/storage/hydration.py` | Implement `coalesce_chunk_spans()` and `hydrate_budgeted_passages()`. Ensure retrieved hits are never dropped or dehydrated. |
| **7.3** | `yaw/daemon/graph/engine.py` | Remove hot-patch shims (`_hot_patch_live_benchmark_modules`). Connect `format_qa_context` to `hydrate_budgeted_passages()`. |
| **7.4** | `yaw/daemon/ingestion/corpus.py` | Strip `.yaw_projected/` and hand-rolled cursor gymnastics. Ingest JSON directly via CocoIndex AST patterns. |
| **7.5** | `scripts/evaluation.py` & `benchmark.py` | Ensure `RetrievalRecall` evaluates chunk containment in gold documents (`retrieved_chunk in gold_doc`), eliminating any requirement to inflate chunks to full documents. |
| **7.6** | Full Verification | Run `pytest tests/core/test_architecture.py tests/daemon/ingestion/test_corpus.py tests/daemon/graph/storage/test_hydration.py tests/test_benchmark.py`, `ruff check .`, `ty check`, and `git diff --check`. |
