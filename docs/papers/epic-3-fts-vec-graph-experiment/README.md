# Epic 3 FTS + vector retrieval literature

Version-pinned arXiv PDFs downloaded for the [experiment proposal](../../reviews/epic-3-fts-vec-graph-experiment.md).

Downloaded files were checked for a PDF header and parsed with Poppler pdfinfo. This verifies document readability, not the correctness or reproducibility of the papers. Titles in sources.json are descriptive labels; venue versions may differ from the pinned arXiv versions.

The machine-readable [manifest](manifest.json) records exact source URLs, byte counts, page counts, SHA-256 hashes, and verification times. [sources.json](sources.json) is the requested bibliography. Excluded entries are retained to make the initial four-paper assessment reviewable.

## Readable copies

All **35 papers** have Markdown copies beside their original PDFs, extracted with **xberg-cli 1.3.7**, with token reduction and OCR disabled. For 34 papers, version-matched arXiv HTML preserves prose reading order better than native two-column PDF extraction. Embedded TeX equations were preserved through conversion; navigation and inline SVG diagrams were omitted. Any math not emitted in place is retained in an explicit appendix. EAGR has no available version-matched HTML endpoint, so its original LaTeX source was expanded and converted with xberg instead; its header records source preprocessing and rendering limitations.

These are reading copies, not verified typesetting replicas. Complex tables, algorithms, references, figures and equation layout can be imperfect; use the original PDF to resolve ambiguity. No paper was summarized or token-reduced. The [Markdown manifest](markdown-manifest.json) records source kinds, exact versions, source and output hashes, conversion options and limitations. The [PDF manifest](manifest.json) remains the integrity record for downloaded originals.

| Key | Paper (Markdown) | PDF | arXiv version | Pages |
| --- | --- | --- | --- | ---: |
| AIR | [Unsupervised Alignment-based Iterative Evidence Retrieval for Multi-hop Question Answering](2005.01218v1-air.md) | [PDF](2005.01218v1-air.pdf) | [2005.01218v1](https://arxiv.org/abs/2005.01218v1) | 12 |
| GoldEn | [Answering Complex Open-domain Questions Through Iterative Query Generation](1910.07000v1-golden-retriever.md) | [PDF](1910.07000v1-golden-retriever.pdf) | [1910.07000v1](https://arxiv.org/abs/1910.07000v1) | 13 |
| MDR | [Answering Complex Open-Domain Questions with Multi-Hop Dense Retrieval](2009.12756v2-multi-hop-dense-retrieval.md) | [PDF](2009.12756v2-multi-hop-dense-retrieval.pdf) | [2009.12756v2](https://arxiv.org/abs/2009.12756v2) | 19 |
| Beam | [End-to-End Beam Retrieval for Multi-Hop Question Answering](2308.08973v2-beam-retrieval.md) | [PDF](2308.08973v2-beam-retrieval.pdf) | [2308.08973v2](https://arxiv.org/abs/2308.08973v2) | 14 |
| Baleen | [Baleen: Robust Multi-Hop Reasoning at Scale via Condensed Retrieval](2101.00436v1-baleen.md) | [PDF](2101.00436v1-baleen.pdf) | [2101.00436v1](https://arxiv.org/abs/2101.00436v1) | 8 |
| IRCoT | [Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions](2212.10509v1-ircot.md) | [PDF](2212.10509v1-ircot.pdf) | [2212.10509v1](https://arxiv.org/abs/2212.10509v1) | 18 |
| Subspace | [Subspace Representations for Soft Set Operations and Sentence Similarities](2210.13034v4-subspace-set-operations.md) | [PDF](2210.13034v4-subspace-set-operations.pdf) | [2210.13034v4](https://arxiv.org/abs/2210.13034v4) | 13 |
| ColBERTv2 | [ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](2112.01488v3-colbertv2.md) | [PDF](2112.01488v3-colbertv2.pdf) | [2112.01488v3](https://arxiv.org/abs/2112.01488v3) | 20 |
| NBFNet | [Neural Bellman-Ford Networks: A General Graph Neural Network Framework for Link Prediction](2106.06935v1-neural-bellman-ford.md) | [PDF](2106.06935v1-neural-bellman-ford.pdf) | [2106.06935v1](https://arxiv.org/abs/2106.06935v1) | 21 |
| DGCNN | [Dynamic Graph CNN for Learning on Point Clouds](1801.07829v1-dynamic-graph-cnn.md) | [PDF](1801.07829v1-dynamic-graph-cnn.pdf) | [1801.07829v1](https://arxiv.org/abs/1801.07829v1) | 14 |
| NBA | [Non-backtracking Graph Neural Networks](2310.07430v2-non-backtracking-gnn.md) | [PDF](2310.07430v2-non-backtracking-gnn.pdf) | [2310.07430v2](https://arxiv.org/abs/2310.07430v2) | 40 |
| Schrodinger | [A survey of the Schroedinger problem and some of its connections with optimal transport](1308.0215v1-schrodinger-survey.md) | [PDF](1308.0215v1-schrodinger-survey.pdf) | [1308.0215v1](https://arxiv.org/abs/1308.0215v1) | 40 |
| GraphOT | [Lazy random walks and optimal transport on graphs](1308.0226v1-graph-optimal-transport.md) | [PDF](1308.0226v1-graph-optimal-transport.pdf) | [1308.0226v1](https://arxiv.org/abs/1308.0226v1) | 42 |
| FeynmanKac | [Cost-augmented Schroedinger bridges on graphs are exactly solvable: a Feynman-Kac tilt replaces learned control](2610.02195v1-cost-augmented-schrodinger.md) | [PDF](2610.02195v1-cost-augmented-schrodinger.pdf) | [2610.02195v1](https://arxiv.org/abs/2610.02195v1) | 15 |
| AdaptiveSubmodularity | [Adaptive Submodularity: Theory and Applications in Active Learning and Stochastic Optimization](1003.3967v5-adaptive-submodularity.md) | [PDF](1003.3967v5-adaptive-submodularity.pdf) | [1003.3967v5](https://arxiv.org/abs/1003.3967v5) | 60 |
| BayesianDesign | [A Unified Stochastic Gradient Approach to Designing Bayesian-Optimal Experiments](1911.00294v2-bayesian-experimental-design.md) | [PDF](1911.00294v2-bayesian-experimental-design.pdf) | [1911.00294v2](https://arxiv.org/abs/1911.00294v2) | 16 |
| GRetriever | [G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering](2402.07630v3-g-retriever.md) | [PDF](2402.07630v3-g-retriever.pdf) | [2402.07630v3](https://arxiv.org/abs/2402.07630v3) | 23 |
| GNNRAG | [GNN-RAG: Graph Neural Retrieval for Large Language Model Reasoning](2405.20139v1-gnn-rag.md) | [PDF](2405.20139v1-gnn-rag.pdf) | [2405.20139v1](https://arxiv.org/abs/2405.20139v1) | 21 |
| GFMRAG | [GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation](2502.01113v1-gfm-rag.md) | [PDF](2502.01113v1-gfm-rag.pdf) | [2502.01113v1](https://arxiv.org/abs/2502.01113v1) | 19 |
| GraphFlow | [Can Knowledge-Graph-based Retrieval Augmented Generation Really Retrieve What You Need?](2510.16582v1-graphflow.md) | [PDF](2510.16582v1-graphflow.pdf) | [2510.16582v1](https://arxiv.org/abs/2510.16582v1) | 30 |
| EAGR | [EAGR: Evidence-Guided Adaptive Graph Retrieval](2611.01910v1-eagr.md) | [PDF](2611.01910v1-eagr.pdf) | [2611.01910v1](https://arxiv.org/abs/2611.01910v1) | 25 |
| QueryFlow | [QueryFlow: Multi-Hop Graph Retrieval with Adaptive Query Evolution](2611.04625v1-queryflow.md) | [PDF](2611.04625v1-queryflow.pdf) | [2611.04625v1](https://arxiv.org/abs/2611.04625v1) | 14 |
| HippoRAG | [HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models](2405.14831v3-hipporag.md) | [PDF](2405.14831v3-hipporag.pdf) | [2405.14831v3](https://arxiv.org/abs/2405.14831v3) | 31 |
| HippoRAG2 | [From RAG to Memory: Non-Parametric Continual Learning for Large Language Models](2502.14802v1-hipporag2.md) | [PDF](2502.14802v1-hipporag2.pdf) | [2502.14802v1](https://arxiv.org/abs/2502.14802v1) | 18 |
| Youtu | [Youtu-GraphRAG: Vertically Unified Agents for Graph Retrieval-Augmented Complex Reasoning](2508.19855v3-youtu-graphrag.md) | [PDF](2508.19855v3-youtu-graphrag.pdf) | [2508.19855v3](https://arxiv.org/abs/2508.19855v3) | 19 |
| MuSiQue | [MuSiQue: Multihop Questions via Single-hop Question Composition](2108.00573v3-musique.md) | [PDF](2108.00573v3-musique.pdf) | [2108.00573v3](https://arxiv.org/abs/2108.00573v3) | 18 |
| ExcludedAutomorphisms | [Automorphisms of the subspace sum graphs on a vector space](1704.03787v1-subspace-sum-automorphisms.md) | [PDF](1704.03787v1-subspace-sum-automorphisms.pdf) | [1704.03787v1](https://arxiv.org/abs/1704.03787v1) | 10 |
| ExcludedSumGraph | [Subspace Sum Graph of a Vector Space](1702.08245v1-subspace-sum-graph.md) | [PDF](1702.08245v1-subspace-sum-graph.pdf) | [1702.08245v1](https://arxiv.org/abs/1702.08245v1) | 11 |
| ExcludedHNC | [Traversing the solution space of neural networks with Hessian Null Space Continuation](2609.38081v1-hessian-null-space-continuation.md) | [PDF](2609.38081v1-hessian-null-space-continuation.pdf) | [2609.38081v1](https://arxiv.org/abs/2609.38081v1) | 55 |
| ExcludedGraded | [Artificial neural networks on graded vector spaces](2407.19031v2-graded-vector-spaces.md) | [PDF](2407.19031v2-graded-vector-spaces.pdf) | [2407.19031v2](https://arxiv.org/abs/2407.19031v2) | 64 |
| NeuralSheafDiffusion | [Neural Sheaf Diffusion: A Topological Perspective on Heterophily and Oversmoothing in GNNs](2202.04579v4-neural-sheaf-diffusion.md) | [PDF](2202.04579v4-neural-sheaf-diffusion.pdf) | [2202.04579v4](https://arxiv.org/abs/2202.04579v4) | 29 |
| RetrievalDiffusion | [Efficient Diffusion on Region Manifolds: Recovering Small Objects with Compact CNN Representations](1611.05113v3-efficient-retrieval-diffusion.md) | [PDF](1611.05113v3-efficient-retrieval-diffusion.pdf) | [1611.05113v3](https://arxiv.org/abs/1611.05113v3) | 10 |
| TensorTrain | [Tensorizing Neural Networks](1509.06569v2-tensorizing-neural-networks.md) | [PDF](1509.06569v2-tensorizing-neural-networks.pdf) | [1509.06569v2](https://arxiv.org/abs/1509.06569v2) | 9 |
| PolynomialSheafDiffusion | [Polynomial Neural Sheaf Diffusion: A Spectral Filtering Approach on Cellular Sheaves](2512.00242v3-polynomial-sheaf-diffusion.md) | [PDF](2512.00242v3-polynomial-sheaf-diffusion.pdf) | [2512.00242v3](https://arxiv.org/abs/2512.00242v3) | 60 |
| JointSheafDiffusion | [Joint Diffusion Processes as an Inductive Bias in Sheaf Neural Networks](2407.20597v1-joint-sheaf-diffusion.md) | [PDF](2407.20597v1-joint-sheaf-diffusion.pdf) | [2407.20597v1](https://arxiv.org/abs/2407.20597v1) | 15 |

## Primary sources without an archived arXiv copy

- [Randomized shortest-path problems: two related models (2009)](https://pubmed.ncbi.nlm.nih.gov/19323635/).
- [Linearly-solvable Markov decision problems (Todorov, NIPS 2006)](https://papers.nips.cc/paper_files/paper/2006/hash/d806ca13ca3449af72a1ea5aedbed26a-Abstract.html).
- [MAP Inference for Customized Determinantal Point Processes via Maximum Inner Product Search (AISTATS 2020)](https://proceedings.mlr.press/v108/han20b.html).
- [A Class of Submodular Functions for Document Summarization (ACL 2011)](https://aclanthology.org/P11-1052/).

- [Reasoning With Neural Tensor Networks for Knowledge Base Completion (NeurIPS 2013)](https://papers.nips.cc/paper_files/paper/2013/hash/b337e84de8752b27eda3a12363109e80-Abstract.html).

No matching arXiv copy was established for these references during this review; no identifier has been guessed.
