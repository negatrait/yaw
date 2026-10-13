# SciPy GraphRAG research extensions

**Status:** Research alternatives; not selected runtime requirements. The [PPR guide](power-pagerank.md) owns the current retrieval algorithm.

Shortest paths can describe graph connectivity but do not prove that a chain of arbitrary predicates is a valid logical inference. Linear-system solvers need their own sparsity, conditioning and workload evaluation before replacing the selected power iteration. Preserve provenance and distinguish navigation from entailment.

The original research note follows as historical material; its unmeasured guarantees and architectural prescriptions are not current requirements.

<details>
<summary>Historical research note</summary>

---
title: "SciPy GraphRAG Extensions: Shortest Paths and Linear System PPR Solvers"
source: "internal://llm-wiki/architecture/scipy-graphrag-extensions"
author: "Antigravity"
published: 2026-10-01
created: 2026-10-01
description: "Architectural specification for multi-hop geodesic explanations via scipy.sparse.csgraph.shortest_path and exact linear system PPR solvers via scipy.sparse.linalg.spsolve in PPR GraphRAG through the ontology layer."
tags:
  - architecture
  - graphrag
  - ppr
  - scipy
---

# SciPy GraphRAG Extensions: Shortest Paths and Linear System Solvers

This specification examines two advanced graph capabilities available in `scipy` for continuous associative memory and knowledge graph reasoning:
1. **Multi-Hop Geodesic Path Traversal** via `scipy.sparse.csgraph.shortest_path`.
2. **Exact Linear System PPR Formulation** via `scipy.sparse.linalg.spsolve` vs. iterative power diffusion.

---

## 1. Multi-Hop Geodesic Explanations (`csgraph.shortest_path`)

### The Dialectical Reasoning Problem
Continuous graph diffusion (PPR GraphRAG through the ontology layer) ranks nodes by associative stationary probability, but does not provide an explanatory trace of *why* two distant concepts or notes are connected. When an agent or user queries the knowledge graph (e.g. asking for the conceptual bridge between `wiki/entities/person/albert-einstein` and `wiki/entities/organization/princeton-university`), the system must reconstruct the minimal transitive assertion chain:

$$\text{Path}(u, v) = \left( u \xrightarrow{p_1} n_1 \xrightarrow{p_2} \dots \xrightarrow{p_k} v \right)$$

### Algorithmic Formulation
Given the symmetric sparse CSR adjacency matrix $A \in \mathbb{R}^{|V| \times |V|}$ cached in `graph.get_graph()["M"]`:

- **Hop Distance (Unweighted)**: Setting all non-zero entries of $A$ to unit weight finds the minimum path length $k$.
- **Information Distance (Weighted)**: Setting edge cost $c(u, v) = -\ln P_{uv}$ where $P$ is the row-stochastic transition matrix yields the maximum likelihood associative path:
  $$\arg\min_{\text{path}} \sum_{i=0}^{k-1} -\ln P_{v_i, v_{i+1}} = \arg\max_{\text{path}} \prod_{i=0}^{k-1} P_{v_i, v_{i+1}}$$

### SciPy Implementation
SciPy provides `scipy.sparse.csgraph.shortest_path` implemented in compiled C using Dijkstra's algorithm with Fibonacci heaps for positive weights ($O(|E| + |V| \log |V|)$) or Breadth-First Search for unweighted graphs ($O(|V| + |E|)$):

```python
import numpy as np
import scipy.sparse.csgraph as csgraph


def get_shortest_path(
    source_uri: str,
    target_uri: str,
    graph_dict: dict,
) -> list[str]:
    node_to_idx = graph_dict["node_to_idx"]
    idx_to_node = graph_dict["idx_to_node"]
    M = graph_dict["M"]  # CSR matrix

    if source_uri not in node_to_idx or target_uri not in node_to_idx:
        return []

    s_idx = node_to_idx[source_uri]
    t_idx = node_to_idx[target_uri]

    # Compute shortest path and predecessor tree in C
    dist_matrix, predecessors = csgraph.shortest_path(
        csgraph=M,
        directed=False,
        indices=s_idx,
        return_predecessors=True,
        unweighted=True,
    )

    if np.isinf(dist_matrix[t_idx]):
        return []  # Disconnected components

    # Trace predecessors backwards from target to source
    path_indices = []
    curr = t_idx
    while curr != -9999:
        path_indices.append(curr)
        if curr == s_idx:
            break
        curr = predecessors[curr]

    path_indices.reverse()
    return [idx_to_node[i] for i in path_indices]
```

### Knowledge Graph Integration
- **Explaining PPR Hits**: When candidate chunks are hydrated in `pipeline.py`, the top-$k$ retrieved notes can be augmented with their shortest geodesic path to the initial query seeds.
- **Dialectical Contradiction Checking**: When evaluating conflicting propositions in `wiki_detect_contradictions`, geodesic path reconstruction identifies the exact sequence of triples bridging opposing assertions.

---

## 2. Linear System PPR vs. Power Iteration (`linalg.spsolve`)

### Theoretical Equivalence
Personalized PageRank (PPR) models an $\alpha$-discounted random walk with teleportation to a personalized seed distribution $p_0$. At stationarity:

$$p = \alpha P^T p + (1 - \alpha) p_0$$

Let $M = P^T$ (the column-stochastic transition matrix). Grouping $p$ terms:

$$(I - \alpha M) p = (1 - \alpha) p_0$$

Because $\alpha \in [0, 1)$ and $M$ is column-stochastic with spectral radius $\rho(M) = 1$, the matrix $I - \alpha M$ is strictly diagonally dominant and guaranteed to be non-singular and invertible:

$$p^* = (1 - \alpha) (I - \alpha M)^{-1} p_0$$

Thus, PPR can be computed either as:
1. **Iterative Power Method**: Successive matrix-vector multiplications $p^{(t+1)} = \alpha M p^{(t)} + (1 - \alpha) p_0$.
2. **Direct Linear System Solve**: Solving $(I - \alpha M) p = (1 - \alpha) p_0$ via `scipy.sparse.linalg.spsolve` (SuperLU sparse LU decomposition).

### SciPy Direct Solver Implementation
```python
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def ppr_direct_solve(
    M: sp.csr_matrix,
    p0: np.ndarray,
    alpha: float = 0.85,
) -> np.ndarray:
    n = M.shape[0]
    I = sp.eye(n, format="csr", dtype=np.float64)
    A = I - alpha * M
    b = (1.0 - alpha) * p0
    # Solve A * p = b directly using sparse LU decomposition
    p = spla.spsolve(A, b)
    return p
```

### Comparative Analysis: Direct Solve vs. Power Iteration

| Metric | Iterative Power Method (`M.dot(p)`) | Direct Solve (`spla.spsolve`) |
| :--- | :--- | :--- |
| **Time Complexity** | $O(K \cdot \|E\|)$ where $K \approx 20$ | $O(\|V\|^{1.5} \sim \|V\|^2)$ for sparse LU factor |
| **Memory Overhead** | $O(1)$ additional auxiliary vector | Fill-in during $L$ and $U$ factorization |
| **Precision** | Geometric error decay $O(\alpha^K) = 0.85^{20} \approx 0.038$ | Exact machine precision ($\epsilon \sim 10^{-16}$) |
| **Dynamic Graph Cost** | Zero setup; uses $M$ directly | Requires full LU re-factorization on graph mutation |
| **Empirical Latency ($N=500$)** | **0.15 ms** | **1.80 ms** (~12x slower) |
| **Empirical Latency ($N=50,000$)**| **3.2 ms** | **45.0 ms** (~14x slower) |

### Architectural Recommendation
1. **Online Query Diffusion**: Retain **iterative power method** (`M.dot(p)`) for `search_graph_activation` in `graph.py`. It runs in $<0.2\text{ ms}$, conserves total probability at every step without fill-in memory, and requires zero cache re-factorization when SQLite notes are updated live.
2. **Offline Precomputation & Inversion**: Use `spla.splu` only when precomputing the full all-pairs inverse resolvent matrix $(I - \alpha M)^{-1}$ for static graph snapshots, where repeated queries can be computed as a single sparse-dense product without re-solving.

</details>
