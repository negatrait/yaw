# Ontological Roots: Hierarchical Deepening & The Waveguide Effect

> [!NOTE]
> **RESEARCH & MATHEMATICAL SPECIFICATION:** This document details the mathematical rationale behind B-Tree hypernym induction and hierarchical deepening. In production, this architecture is codified in [`docs/adr/ADR-002.md`](file:///home/harri/yaw/docs/adr/ADR-002.md) (B-Tree Hypernym Induction for Saturated Hubs), [`docs/adr/ADR-005.md`](file:///home/harri/yaw/docs/adr/ADR-005.md) (9 Universal Root Namespaces), and [`FOUNDATIONAL.md §5.4`](file:///home/harri/yaw/FOUNDATIONAL.md).

**Governing Canon:**  
- [`docs/adr/ADR-002.md`](file:///home/harri/yaw/docs/adr/ADR-002.md)  
- [`docs/adr/ADR-005.md`](file:///home/harri/yaw/docs/adr/ADR-005.md)  
- [`FOUNDATIONAL.md §5.1, §5.4`](file:///home/harri/yaw/FOUNDATIONAL.md)  

---

## 1. Why Roots Must "Dig Deeper" (The PPR GraphRAG Math)

When a knowledge base expands from 10 notes to 1,000, its ontology cannot remain a flat 1-tier canopy (`Root -> Leaf`). If it does, root domains become massive super-hubs, and Personalized PageRank (PPR) graph diffusion degrades.

Suppose a wiki grows to 500 notes, and 120 different concepts attach directly to `concept:resilience`.
In the PPR transition matrix across the ontology layer, outgoing probability is distributed across neighbors:
$$P_{uv} = \frac{W_{uv}}{\sum_k W_{uk}}$$

If `concept:resilience` has 120 outgoing edges, each child concept receives less than **$0.008$ of the probability mass** ($1/120$). Energy injected by a vague query is pulverized into micro-droplets. No individual chunk accumulates enough mass to beat background noise.

### The Solution: Hierarchical Deepening (The Waveguide Effect)
As the wiki expands, intermediate **Meso-Concepts** must emerge to act as directed conduits (waveguides) [Edge et al., 2024, arXiv:2404.16130]:

```txt
Flat Structure (Energy Diffusion):
Root (Resilience) ──► 120 leaves (each gets ~0.8% energy)

Hierarchical Deepening (Waveguide Focus):
Root (Resilience)
  ├── Meso 1 (Ecological Resilience)  ──► 30 leaves (~8.3% per leaf)
  ├── Meso 2 (Cybernetic Resilience)  ──► 40 leaves (~6.2% per leaf)
  └── Meso 3 (Psychological Resilience) ──► 50 leaves (~5.0% per leaf)
```

The intermediate meso-concepts concentrate random walk energy into specific topical subgraphs, increasing signal-to-noise ratio by up to $10\times$.

---

## 2. Deterministic B-Tree Splitting vs Stochastic Clustering

Traditional GraphRAG systems run stochastic community detection algorithms (e.g. Louvain, Leiden) or LLM-driven batch clustering. In an incremental, local-first wiki, this causes **Ontological Churn (Thrashing)**: adding a single note can arbitrarily repartition the entire graph, rewriting dozens of concept boundaries.

YAW solves this by treating the ontology like a **B-Tree** ([`docs/adr/ADR-002.md`](file:///home/harri/yaw/docs/adr/ADR-002.md)):
1. **Saturation Threshold ($K_{\max} = 10$):** When any node's out-degree exceeds $10$, it is considered saturated.
2. **Deterministic Ward's HAC:** SciPy Agglomerative Hierarchical Clustering (`method='ward'`) partitions the saturated node's children into two balanced sub-clusters based on vector embedding distance.
3. **Hypernym Induction:** An LLM call (`temperature=0.1`) generates a concise 1–3 word hypernym and 1-sentence definition for the split cluster.
4. **Substrate Immutability:** Saturated hub splits persist in `compiled_nodes` and `compiled_relations` tables without mutating raw `leaf_nodes` or `leaf_relations`, strictly preserving Invariant 1.
5. **Taxonomic Conductance Asymmetry:** Edges are rewired with $1.0$ downward conductance (parent $\to$ child) and $0.2$ upward conductance (child $\to$ parent), preventing mass leakage into superclasses.
