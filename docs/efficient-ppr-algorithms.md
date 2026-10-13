---
title: "Efficient Algorithms for Personalized PageRank Computation: A Survey"
source: https://arxiv.org/html/2403.05198v1
author:
published: 2024-03-08
created: 2026-09-28
description:
tags:
  - clippings
---

> **Research reference:** The survey below is upstream material, not YAW's executable specification. Use the [PPR guide](power-pagerank.md) and [FOUNDATIONAL](../FOUNDATIONAL.md) for the current contract. Normalized graph relevance does not establish factual truth or hard identity.

Mingji Yang    Hanzhi Wang    Zhewei Wei    Sibo Wang    Ji-Rong Wen

## Abstract

Personalized PageRank (PPR) is a traditional measure for node proximity on large graphs. For a pair of nodes $\boldsymbol{s}$ and $\boldsymbol{t}$, the PPR value $\boldsymbol{\boldsymbol{\pi}_{s}(t)}$ equals the probability that an $\boldsymbol{\alpha}$ -discounted random walk from $\boldsymbol{s}$ terminates at $\boldsymbol{t}$ and reflects the importance between $\boldsymbol{s}$ and $\boldsymbol{t}$ in a bidirectional way. As a generalization of Google’s celebrated PageRank centrality, PPR has been extensively studied and has found multifaceted applications in many fields, such as network analysis, graph mining, and graph machine learning. Despite numerous studies devoted to PPR over the decades, efficient computation of PPR remains a challenging problem, and there is a dearth of systematic summaries and comparisons of existing algorithms. In this paper, we recap several frequently used techniques for PPR computation and conduct a comprehensive survey of various recent PPR algorithms from an algorithmic perspective. We classify these approaches based on the types of queries they address and review their methodologies and contributions. We also discuss some representative algorithms for computing PPR on dynamic graphs and in parallel or distributed environments.

## I Introduction

Graphs are a ubiquitous abstract data type that models a wide range of information in the real world. Personalized PageRank (PPR) [^1], as a direct variant of Google’s renowned PageRank centrality [^1], is a widely used measure of node proximity in graphs. Given a source node $s$ and a target node $t$, the PPR value $\boldsymbol{\pi}_{s}(t)$ offers a quantitative amount of the significance of $t$ w.r.t. $s$. On the other hand, $\boldsymbol{\pi}_{s}(t)$ also indicates the importance of $s$ in the view of $t$ [^2], and thus it essentially gauges whether $s$ and $t$ are mutually significant. The PageRank centrality of $t$, denoted as $\boldsymbol{\pi}(t)$, equals the average of $\boldsymbol{\pi}_{s}(t)$ over all source nodes $s$ and can be viewed as a special form of generalized PPR.

Although the seminal paper of PageRank and PPR [^1] considers utilizing them to rate webpages objectively and mechanically, recent years have witnessed a wide range of their applications beyond the web. In fact, due to its simplicity and profound influence, PageRank was named one of the top- $10$ algorithms in data mining in 2008 [^3]. We provide three representative instances of PPR’s applications below: local clustering, node embedding, and graph neural networks. More applications can be found in Gleich’s survey [^4].

Local Graph Clustering. As a traditional graph mining task, local graph clustering aims at identifying a well-connected cluster around a specified node by only examining a local region of the graph. A notable algorithm is PageRank-Nibble by Anderson, Chung, and Lang [^5] [^6], which leverages a local-push method for computing PPR to find the local cluster efficiently with strong theoretical guarantees. Following PageRank-Nibble, there arises a collection of recent works that rely on PPR or its extensions to compute local clusters, such as MAPPR [^7], TEA [^8], and [^9] [^10] [^11].

Node Embedding. Node embedding methods endeavor to learn low-dimensional representations of the nodes in the graph while preserving the similarities between them. These methods typically require a node proximity measure in the first place, and many of them choose PPR values, such as HOPE [^12], APP [^13], and VERSE [^14]. Some recent works [^15] [^16] [^17] also leverage PPR or its extensions.

Graph Neural Networks. Recently, Graph Neural Networks (GNNs) have fueled an explosion of research in graph machine learning. As a fundamental measure of node proximity, PPR has played an important role in designing various GNNs, especially scalable ones. For example, APPNP [^18] and PPRGo [^19] apply approximate PPR to GNNs, making them more efficient and scalable; GBP [^20] and AGP [^21] generalize the notion of PPR to propose novel methods for effective approximate graph propagation.

Other recent extensions and applications of PPR include [^22] [^23] [^24] [^25]. Despite the abundant applications, the computation of PageRank and PPR remains a bottleneck in some situations [^20] [^21], as the graphs under consideration are becoming increasingly massive. Even in 2002, the graph could contain $2.7$ billion nodes, and computing PageRank was consequently dubbed “The World’s Largest Matrix Computation” [^26]. Also, the complexity of computing PageRank and PPR is of individual theoretical interest. These factors have sparked a large corpus of studies concerning the efficient computation of PPR on large graphs in the past decades.

Based on these studies, several surveys on PPR algorithms have been conducted, such as Berkhin’s early survey in 2005 [^27], Chung’s brief survey in 2014 [^28], and a more recent one by Park et al. in 2019 [^29]. Nevertheless, none of them provides meticulous summaries of the fundamental techniques for computing PPR, nor do they review recent PPR algorithms comprehensively from a theoretical perspective. In contrast, this paper strives to address these gaps by delving into five atomic techniques and systematically categorizing numerous algorithms according to their query types. We provide theoretical analyses of the basic techniques, point out their pros and cons, and summarize the surveyed algorithms’ methodologies and contributions. Additionally, to cater to the emerging needs of handling evolving graphs, exploiting new computing environments, etc., we include a section for recent PPR algorithms tailored to these special settings. While we primarily focus on the proximity measure of PPR, we also include numerous algorithms for PageRank computation.

Paper Organization. The remainder of this paper is organized as follows. Sec. II provides preliminary knowledge about PPR and defines various PPR computation problems. Sec. III reviews five fundamental techniques extensively employed in PPR computation. Sec. IV gives an overview of various PPR algorithms. In Sec. V to IX, we summarize the salient ideas and contributions of numerous recent PPR algorithms under the typical setting, categorized by the query types. As a complement, Sec. X discusses the algorithms on dynamic graphs, in parallel/distributed environments, or for other special settings. After that, we discuss the research challenges and directions in Sec. XI. Finally, Sec. XII concludes the paper.

## II Preliminaries

This section offers preliminary knowledge for PPR computation, including basic notations for graphs, definitions of PPR and PPR queries, and some fundamental properties of PPR.

### II-A Notations for Graphs

PageRank and PPR were originally proposed by Google as a tool to assign an authority score to each webpage solely based on the links between them [^1]. Nonetheless, the notion of PageRank and PPR can be directly applied to any graph for measuring the importance of the nodes in the graph. In light of this, we begin with some basic definitions of graphs.

We consider a directed or undirected graph $G=(V,E)$ with $|V|=n$ nodes and $|E|=m$ edges, and we assume that the nodes in $V$ are numbered in some order as $v_{1},v_{2},\ldots,v_{n}$. Several fundamental concepts on $G$ are listed below.

Neighbors and Degrees. For a node $v\in V$, we denote the set of the in-neighbors of $v$ by $\mathcal{N}_{\mathrm{in}}(v)=\big\{u\mid(u,v)\in E\big\}$, and let $d_{\mathrm{in}}(v)=\big\lvert\mathcal{N}_{\mathrm{in}}(v)\big\rvert$. Similarly, we define $\mathcal{N}_{\mathrm{out}}(v)=\big\{u\mid(v,u)\in E\big\}$ and $d_{\mathrm{out}}(v)=\big|\mathcal{N}_{\mathrm{out}}(v)\big|$. For undirected graphs, we use $d(v)$ to denote the degree of $v$.

Matrices of $\boldsymbol{G}$. The adjacency matrix of $G$ is an $n\times n$ matrix $\mathbf{A}$ whose $(i,j)$ -th entry equals $\mathds{1}\big\{(v_{i},v_{j})\in E\big\}$, where $\mathds{1}\{\}$ is the indicator function. The degree matrix of $G$ is an $n\times n$ diagonal matrix $\mathbf{D}$ whose $i$ -th diagonal entry equals $d_{\mathrm{out}}(v_{i})$. The transition matrix of $G$ is formalized as $\mathbf{P}=\mathbf{A}^{\top}\mathbf{D}^{-1}$. To ensure that $\mathbf{P}$ is well-defined, we assume that $d_{\mathrm{out}}(v)>0$ for all $v\in V$ throughout this paper (see the last paragraph in Sec. II-B for discussions). In this case, $\mathbf{P}$ is column-stochastic and can thus be viewed as a Markov chain transition matrix.

Indicator Vector. For a node $v\in V$, we define $\boldsymbol{e}_{v}$ as an $n$ -dimensional column vector whose $i$ -th entry equals $\mathds{1}\{v_{i}=v\}$. Note that we use column vectors instead of row vectors throughout this survey. Also, for an $n$ -dimensional vector $\boldsymbol{x}$ associated with the nodes in $V$, we denote by $\boldsymbol{x}(v)$ the $i$ -th entry of $\boldsymbol{x}$, where $i$ satisfies $v_{i}=v$.

### II-B Definitions of PPR and PageRank

The concepts of PPR and PageRank are similar, and they can both be formulated as random-walk probabilities. While this paper mainly concentrates on PPR computation, we also consider PageRank and regard it as a special form of generalized PPR. We begin with the definition of PPR and then define PageRank based on it.

We first give the algebraic definition of PPR. We denote the PPR vector w.r.t. a source node $s$ as $\boldsymbol{\pi}_{s}$, which assigns a proximity score $\boldsymbol{\pi}_{s}(t)$ to each $t\in V$. $\boldsymbol{\pi}_{s}$ is defined as the unique solution to the linear equation

$$
\displaystyle\boldsymbol{\pi}_{s}
$$

$$
\displaystyle=(1-\alpha)\mathbf{P}\boldsymbol{\pi}_{s}+\alpha\cdot\boldsymbol{e}_{s},
$$

where $\alpha\in(0,1)$ is a predefined parameter called the teleportation probability (a.k.a. decay factor). Here, the indicator vector $\boldsymbol{e}_{s}$ is called the preference vector in defining PPR. To demonstrate that $\boldsymbol{\pi}_{s}$ is well-defined, note that the matrix $\mathbf{I}-(1-\alpha)\mathbf{P}$ is guaranteed to be nonsingular since it is strictly column diagonally dominant. Thus, $\boldsymbol{\pi}_{s}$ equals

$$
\displaystyle\boldsymbol{\pi}_{s}
$$

$$
\displaystyle=\alpha\big(\mathbf{I}-(1-\alpha)\mathbf{P}\big)^{-1}\boldsymbol{e}_{s},
$$

which establishes the existence and uniqueness of $\boldsymbol{\pi}_{s}$. A more general definition of PPR allows the preference vector in Eqn. (1) to be an arbitrary probability distribution vector, rather than being limited to indicator vectors. As we will see from Thm. 1, these generalized PPR vectors are linear combinations of the PPR vectors in our definition. *(In YAW, this generalized preference vector is operationalized as the multi-modal seed vector $\mathbf{p}_0 = \beta_1 \mathbf{s}_{\text{concepts}} + \beta_2 \mathbf{s}_{\text{chunks\_vec}} + \beta_3 \mathbf{s}_{\text{chunks\_fts}}$, where $\mathbf{s}_{\text{concepts}}[e]$ incorporates the alias candidate prior distribution $P(e \mid a) \in (0, 1]$ via `InMemoryLookupKB`; see [`docs/adr/ADR-010.md`](adr/ADR-010.md) and [`docs/power-pagerank.md`](power-pagerank.md)).*

Random-Walk-Based Interpretations. Apart from the algebraic definition, there are two more comprehensible interpretations of PPR. Both of them are based on a critical concept called random walk, with two slightly different definitions.

In the first possible definition, a random walk sets out from the source node $s$, and at each step, it either (i) jumps back to $s$ w.p. (shorthand for “with probability”) $\alpha$, or (ii) proceeds uniformly at random to an out-neighbor of the current node. This type of random walk is also called random walk with restart (RWR). The PPR vector $\boldsymbol{\pi}_{s}$ is then defined as the stationary distribution over $V$ for an RWR after infinitely many steps. For this reason, PPR values are occasionally referred to as RWR values.

Alternatively, one can modify the procedure above to define an $\alpha$ -discounted random walk, which starts at $s$ and terminates at the current node (instead of jumping back to $s$) w.p. $\alpha$ at each step. The vector $\boldsymbol{\pi}_{s}$ can then be defined as the termination distribution over the nodes at which such an $\alpha$ -discounted random walk eventually terminates. It can be verified that the three definitions of PPR above are essentially equivalent [^30].

Since the PPR value $\boldsymbol{\pi}_{s}(t)$ serves as a bidirectional proximity measure between nodes $s$ and $t$ [^2], the inverse PPR vector $\boldsymbol{\pi}^{-1}_{t}$ is also of research interest, where $\boldsymbol{\pi}^{-1}_{t}(s)$ is defined to be $\boldsymbol{\pi}_{s}(t)$. For convenience, we further define the PPR matrix $\mathbf{\Pi}$ as an $n\times n$ matrix, with its $(i,j)$ -th entry equaling $\boldsymbol{\pi}_{v_{j}}(v_{i})$. $\mathbf{\Pi}$ encompasses $n^{2}$ possible PPR values between all nodes in $V$, while its $i$ -th column equals $\boldsymbol{\pi}_{v_{i}}$ and $i$ -th row equals $\boldsymbol{\pi}^{-1}_{v_{i}}$.

PageRank. We denote the PageRank vector as $\boldsymbol{\pi}$, which assigns a centrality score $\boldsymbol{\pi}(t)$ to each $t\in V$. Algebraically, PageRank is a special form of generalized PPR with $1/n\cdot\boldsymbol{1}$ as the preference vector, where $\boldsymbol{1}$ is the $n$ -dimensional vector with value $1$ on each coordinate:

$$
\displaystyle\boldsymbol{\pi}
$$

$$
\displaystyle=(1-\alpha)\mathbf{P}\boldsymbol{\pi}+\alpha\cdot\frac{1}{n}\boldsymbol{1}.
$$

As we will see from Thm. 1, $\boldsymbol{\pi}$ equals the average of $\boldsymbol{\pi}_{s}$ over all $s\in V$. PageRank can also be defined using RWR or $\alpha$ -discounted random walks. For this purpose, an RWR sets out from an arbitrary node and jumps back to a uniformly random node in $V$ w.p. $\alpha$ at each step, and an $\alpha$ -discounted random walk starts from a uniformly random source node in $V$.

Eliminating Dangling Nodes. We call a node $v$ a dangling node if $d_{\mathrm{out}}(v)=0$. As mentioned, the formal definition of PPR and PageRank requires no dangling nodes in the graph. However, this is hardly the case in practice. When working with real-world graphs, several strategies for handling dangling nodes have been proposed. Three commonly used remedies are: (i) introduce a dummy node that has a self-loop to itself and add an edge from each dangling node to it; (ii) assume that each dangling node has outgoing edges to each node in $V$; and (iii) add a self-loop to each dangling node. Different methods may incur minor differences in the resulting PPR values, but the overall ranking remains the same. See, e.g., [^31] for more detailed descriptions and analyses.

### II-C Definitions of PPR Queries

As introduced above, the $n\times n$ PPR matrix $\mathbf{\Pi}$ contains the PPR values between all nodes in $V$ as useful proximity scores. However, computing and storing the dense matrix $\mathbf{\Pi}$ requires $\Omega\left(n^{2}\right)$ time and space, which is prohibitive when $n$ reaches, say, $10^{7}$. Fortunately, in many scenarios, it is unnecessary to compute the entire $\mathbf{\Pi}$; instead, we only need a single row/column of $\mathbf{\Pi}$ or even a single entry in it. This motivates us to define various query types for PPR computation.

Query Types. Common queries for PPR computation include:

- single-source PPR (SSPPR): given a source node $s\in V$, the PPR vector $\boldsymbol{\pi}_{s}$ is required.
- single-target PPR (STPPR): given a target node $t\in V$, the inverse PPR vector $\boldsymbol{\pi}^{-1}_{t}$ is required.
- single-pair PPR (SPPPR): given a source node $s\in V$ and a target node $t\in V$, the PPR value $\boldsymbol{\pi}_{s}(t)$ is required.
- top- $k$ PPR: given a source node $s$ and an integer $k$, the nodes with the top- $k$ largest values in $\boldsymbol{\pi}_{s}$ are required. In some cases, their corresponding values in $\boldsymbol{\pi}_{s}$ are also required.

Additionally, for PageRank computation, two common tasks are the PageRank vector query, which asks for the vector $\boldsymbol{\pi}$, and the single-node PageRank (SNPR) query, which asks for the value $\boldsymbol{\pi}(t)$ of a single target node $t$. They can be viewed as generalizations of the SSPPR and SPPPR queries, resp. The whole PPR matrix can be obtained by invoking the SSPPR queries for all $s\in V$ or the STPPR queries for all $t\in V$.

As we will see, these queries can be answered exactly using algorithms based on algebraic techniques. However, computing exact answers often incurs significant costs, which is impractical for efficient online queries. Particularly, for the SPPPR and top- $k$ PPR queries that have small output sizes, computing exact answers is unnecessarily wasteful. In real applications, it is generally acceptable that these queries are answered approximately in exchange for higher efficiency. Also, introducing approximation allows algorithms to answer certain queries with sparse vectors. The errors incurred by the approximate answers are typically quantified by the following four types of error measures.

Error Measures. For two $n$ -dimensional vectors $\boldsymbol{\hat{x}}$ and $\boldsymbol{x}$ or a positive real number $x$ along with its estimate $\hat{x}$, four commonly used error measures are defined as follows:

- $\ell_{1}$ -error: $\lVert\boldsymbol{\hat{x}}-\boldsymbol{x}\rVert_{1}=\sum_{v\in V}\big\lvert\boldsymbol{\hat{x}}(v)-\boldsymbol{x}(v)\big\rvert.$
- $\ell_{2}$ -error: $\lVert\boldsymbol{\hat{x}}-\boldsymbol{x}\rVert_{2}=\sqrt{\sum_{v\in V}\big(\boldsymbol{\hat{x}}(v)-\boldsymbol{x}(v)\big)^{2}}.$
- absolute error (a.k.a. additive error or $\ell_{\infty}$ -error for vectors): $\lVert\boldsymbol{\hat{x}}-\boldsymbol{x}\rVert_{\infty}=\max_{v\in V}\big\lvert\boldsymbol{\hat{x}}(v)-\boldsymbol{x}(v)\big\rvert$, or $|\hat{x}-x|$.
- relative error with threshold $\mu$:\
	$\max_{v\in V:\boldsymbol{x}(v)\geq\mu}\frac{|\boldsymbol{\hat{x}}(v)-\boldsymbol{x}(v)|}{\boldsymbol{x}(v)}$, or $\frac{\lvert\hat{x}-x\rvert}{x}\cdot\mathds{1}\{x\geq\mu\}$.

These definitions are straightforward, except for relative error: given the inherent difficulty of achieving the same relative error for smaller values, many relevant algorithms offer relative error bounds only for values above a threshold $\mu$. Also, note that these error measures are not directly applicable for the top- $k$ PPR query; we will discuss the approximation guarantees separately for the top- $k$ PPR query.

Probabilistic Algorithms. Furthermore, to achieve higher efficiency, it is common to allow the algorithms to be probabilistic, as long as they can answer the queries successfully with high probability. Concretely, these probabilistic algorithms receive an extra parameter $\delta$ as input and guarantee to return acceptable results w.p. at least $(1-\delta)$. With approximation and randomness, the algorithms can potentially answer the queries in sublinear time w.r.t. the size of the graph.

Combining the three factors above, a concrete algorithm is required to answer a specific type of PPR query exactly or under a designated error measure, and it may be allowed a small failure probability $\delta$. We denote the error bound by $\varepsilon$ and require that the designated error of the results does not exceed $\varepsilon$. Below, we define a specific query as an example.

#### Definition 1 (Probabilistic SSPPR Query with Relative Error Bounds).

Given a graph $G$, a decay factor $\alpha$, a source node $s$, a relative error parameter $\varepsilon$, a relative error threshold $\mu$, and a failure probability parameter $\delta$, the query requires an estimated PPR vector $\boldsymbol{\hat{\pi}}_{s}$ such that w.p. at least $(1-\delta)$, $\big\lvert\boldsymbol{\hat{\pi}}_{s}(v)-\boldsymbol{\pi}_{s}(v)\big\rvert\leq\varepsilon\cdot\boldsymbol{\pi}_{s}(v)$ holds for all $v$ with $\boldsymbol{\pi}_{s}(v)\geq\mu$.

Conversions between Error Measures. Despite the differences between the error measures, it is possible to directly convert some error guarantees to another type using the following facts. Here, we use $\varepsilon_{\ell}$, $\varepsilon_{a}$, and $\varepsilon_{r}$ to denote the $\ell_{1}$ -error, the absolute error, and the relative error, resp.

- from $\ell_{1}$ -error to absolute error: $\varepsilon_{\ell}\leq\varepsilon$ implies $\varepsilon_{a}\leq\varepsilon$.
- from absolute error to $\ell_{1}$ -error: $\varepsilon_{a}\leq\varepsilon/n$ implies $\varepsilon_{\ell}\leq\varepsilon$.
- from absolute error to relative error: $\varepsilon_{a}\leq\varepsilon\mu$ implies $\varepsilon_{r}\leq\varepsilon$ (with relative error threshold $\mu$).

However, these conversions are wasteful in general as they do not explicitly consider the features of the target error measures.

TABLE I: Summary of Notations

| Notation | Description |
| --- | --- |
| $G=(V,E)$ | graph with node set $V$ and edge set $E$ (Sec. II-A) |
| $n,m$ | number of nodes/edges in $G$ (Sec. II-A) |
| $\mathcal{N}_{\mathrm{in}}(v),\mathcal{N}_{\mathrm{out}}(v)$ | the set of in/out-neighbors of node $v$ (Sec. II-A) |
| $d_{\mathrm{in}}(v),d_{\mathrm{out}}(v)$ | in/out-degree of node $v$ (Sec. II-A) |
| $\mathbf{A},\mathbf{D}$ | adjacency and degree matrices of $G$ (Sec. II-A) |
| $\mathbf{P}$ | transition matrix of $G$, $\mathbf{P}=\mathbf{A}^{\top}\mathbf{D}^{-1}$ (Sec. II-A) |
| $\boldsymbol{e}_{s}$ | indicator vector (Sec. II-A) |
| $\alpha$ | teleportation probability (Sec. II-B) |
| $\boldsymbol{\pi}_{s}$ | PPR vector of source node $s$ (Sec. II-B) |
| $\boldsymbol{\pi}$ | PageRank vector of $G$ (Sec. II-B) |
| $\boldsymbol{\pi}^{-1}_{t}$ | inverse PPR vector of target node $t$ (Sec. II-B) |
| $\mathbf{\Pi}$ | PPR matrix (Sec. II-B) |
| $\varepsilon$ | error bound parameter (Sec. II-C) |
| $\mu$ | relative error threshold (Sec. II-C) |
| $\delta$ | failure probability parameter (Sec. II-C) |
| $r_{\mathrm{max}}^{(\mathrm{f})},r_{\mathrm{max}}^{(\mathrm{b})}$ | threshold of FP and BP, resp. (Sec. III) |
| $\boldsymbol{\hat{\pi}}^{(\mathrm{f})}_{s},\boldsymbol{\hat{\pi}}^{(\mathrm{b})}_{t}$ | reserve vector in FP and BP, resp. (Sec. III) |
| $\boldsymbol{r}^{(\mathrm{f})}_{s},\boldsymbol{r}^{(\mathrm{b})}_{t}$ | residue vector in FP and BP, resp. (Sec. III) |

### II-D Basic Properties of PPR

In this subsection, we overview several important properties of PPR without proving them. These properties will be frequently used when designing algorithms for computing PPR.

Algebraically, the PPR vector $\boldsymbol{\pi}_{s}$ can be expressed as the following infinite summation:

$$
\displaystyle\boldsymbol{\pi}_{s}=\alpha\sum_{\ell=0}^{\infty}(1-\alpha)^{\ell}\mathbf{P}^{\ell}\boldsymbol{e}_{s}.
$$

Here, $\mathbf{I}$ is the identity matrix of order $n$. The inverse PPR vector $\boldsymbol{\pi}^{-1}_{t}$ satisfies $\boldsymbol{\pi}^{-1}_{t}=(1-\alpha)\mathbf{P}^{\top}\boldsymbol{\pi}^{-1}_{t}+\alpha\boldsymbol{e}_{t}$ and equals

$$
\displaystyle\alpha\left(\mathbf{I}-(1-\alpha)\mathbf{P}^{\top}\right)^{-1}\boldsymbol{e}_{t}=\alpha\sum_{\ell=0}^{\infty}(1-\alpha)^{\ell}\left(\mathbf{P}^{\top}\right)^{\ell}\boldsymbol{e}_{t}.
$$

The PPR matrix $\mathbf{\Pi}$ satisfies $\mathbf{\Pi}=(1-\alpha)\mathbf{P}\mathbf{\Pi}+\alpha\mathbf{I}$ and

$$
\displaystyle\mathbf{\Pi}=\alpha\big(\mathbf{I}-(1-\alpha)\mathbf{P}\big)^{-1}=\alpha\sum_{\ell=0}^{\infty}(1-\alpha)^{\ell}\mathbf{P}^{\ell}.
$$

Some other algebraic properties of PPR are listed below, wherein $\lVert\boldsymbol{x}\rVert_{1}$ denotes the $\ell_{1}$ -norm of vector $\boldsymbol{x}$.

$$
\displaystyle\lVert\boldsymbol{\pi}_{s}\rVert_{1}=1
$$

$$
\displaystyle,\ \left\lVert\boldsymbol{\pi}^{-1}_{t}\right\rVert_{1}=n\boldsymbol{\pi}(t).
$$
$$
\displaystyle\boldsymbol{\pi}_{s}=\mathbf{\Pi}\boldsymbol{e}_{s}
$$

$$
\displaystyle,\ \boldsymbol{\pi}^{-1}_{t}=\mathbf{\Pi}^{\top}\boldsymbol{e}_{t}.
$$
$$
\displaystyle\lVert\mathbf{\Pi}\boldsymbol{x}\rVert_{1}=\lVert\boldsymbol{x}\rVert_{1}
$$

$$
\displaystyle,\ \mathbf{\Pi}^{\top}\boldsymbol{1}=\boldsymbol{1}.
$$

The following theorem considers the generalized PPR vector $\boldsymbol{\pi}_{\boldsymbol{\sigma}}$ for an arbitrary preference vector $\boldsymbol{\sigma}$. It states that $\boldsymbol{\pi}_{\boldsymbol{\sigma}}$ is a linear combination of the PPR vectors $\boldsymbol{\pi}_{s}$.

#### Theorem 1 (The Linearity Theorem ).

For a preference vector $\boldsymbol{\sigma}$, we have $\boldsymbol{\pi}_{\boldsymbol{\sigma}}=\sum_{s\in V}\boldsymbol{\sigma}(s)\cdot\boldsymbol{\pi}_{s}$.

The following theorem provides a method to decompose a node’s PPR vector or inverse PPR vector into a linear combination of the corresponding vectors of its neighbors. Intuitively, these formulas can be obtained by enumerating the first/last step of the random walk.

#### Theorem 2 (The Decomposition Theorem ).

$$
\displaystyle\boldsymbol{\pi}_{v}
$$

$$
\displaystyle=\frac{1-\alpha}{d_{\mathrm{out}}(v)}\sum_{u\in\mathcal{N}_{\mathrm{out}}(v)}\boldsymbol{\pi}_{u}+\alpha\boldsymbol{e}_{v},
$$
$$
\displaystyle\boldsymbol{\pi}^{-1}_{v}
$$

$$
\displaystyle=(1-\alpha)\sum_{u\in\mathcal{N}_{\mathrm{in}}(v)}\frac{\boldsymbol{\pi}^{-1}_{u}}{d_{\mathrm{out}}(u)}+\alpha\boldsymbol{e}_{v}.
$$

PPR on Undirected Graphs. There are some additional properties for PPR on undirected graphs. At a high level, as the main subject of spectral graph theory, undirected graphs have many special algebraic properties. As simple examples, we have $\mathbf{\Pi}\mathbf{D}\boldsymbol{1}=\mathbf{P}\mathbf{D}\boldsymbol{1}=\mathbf{D}\boldsymbol{1}=\mathbf{A}\boldsymbol{1}$. Also, random walks on undirected graphs are different from those on general directed graphs as they can be traversed oppositely. Besides, it is well-known that a typical random walk (without restart) on a connected and non-bipartite undirected graph converges to the stationary distribution, in which each entry is proportional to the degree of the corresponding node. As for PPR, the following key theorem forms the foundation for many PPR algorithms tailored to undirected graphs:

#### Theorem 3 (Symmetry of PPR on Undirected Graphs ).

$d(s)\cdot\boldsymbol{\pi}_{s}(t)=d(t)\cdot\boldsymbol{\pi}_{t}(s)$ holds for all $s,t\in V$.

Of course, these properties of PPR are far from comprehensive. Interested readers are referred to [^31] [^34] for detailed explanations of the mathematical properties of PPR. Tbl. I lists the frequently used notations in this paper.

## III Basic Techniques

In this section, we elaborate on five basic techniques for computing PPR: the Monte Carlo method [^35], Power Iteration [^1], Forward Push [^5] [^6], Reverse Power Iteration [^36] [^37], and Backward Push [^36] [^37]. They are the building blocks of more advanced algorithms for PPR computation. We give detailed descriptions and analyses of these techniques, as well as a comparison of their pros and cons.

### III-A The Monte Carlo Method

The Monte Carlo method (MC) is a classic approach for calculating sample-based estimates of desired values. Inspired by the interpretation of PPR based on $\alpha$ -discounted random walks, we can readily approximate the PPR vector $\boldsymbol{\pi}_{s}$ using MC. Albeit seemingly naïve, MC can obtain unbiased and accurate PPR estimates efficiently with high probability, and it can potentially break the theoretical lower bounds for deterministic algorithms. Another latent advantage of MC is its inherent suitability for parallelization.

The fundamental idea of MC is to simulate $\alpha$ -discounted random walks from $s$ and use the empirical termination distribution as the approximation for $\boldsymbol{\pi}_{s}$. This provides a natural way to answer the SSPPR query. Suppose that we generate $W$ random walks independently, which takes expected $\Theta(W/\alpha)$ time. By Chernoff bound, setting $W=\Theta\left(\log(n/\delta)\big/\varepsilon^{2}\right)$ suffices to answer the probabilistic SSPPR query with absolute error bounds and setting $W=\Theta\left(\log(n/\delta)\big/\left(\varepsilon^{2}\mu\right)\right)$ suffices for that with relative error bounds [^35].

Variants of MC. To improve the efficiency, numerous variants of MC have been proposed, with different approaches to generate random walks and extract information from the resultant walks. Avrachenkov et al. [^30] suggest several versions of MC, in particular one that takes into account the entire traversed path of the random walks, called “MC complete path.” Wei et al. [^38] propose a novel type of random walk called $\left(\sqrt{1-\alpha}\right)$ -walk, which has lower variances. Liao et al. [^39] present an algorithm for sampling a random spanning forest of the graph. They prove that such a sampling takes considerably less time than sampling $n$ random walks, especially for small $\alpha$. They also argue that for SSPPR and STPPR computation, such a forest sampling provides information comparable to that of $n$ random walk samplings.

### III-B Power Iteration

Power Iteration (PI) is a basic iterative approach for computing the whole PPR vector, as initially introduced in the seminal paper by Google [^1]. In fact, it is a direct adaptation of the Jacobi Method for solving linear equations in numerical linear algebra (see, e.g., [^40]), where $\boldsymbol{\pi}_{s}$ is viewed as the solution to $\boldsymbol{\pi}_{s}=(1-\alpha)\mathbf{P}\boldsymbol{\pi}_{s}+\alpha\boldsymbol{e}_{s}$ (Eqn. (1)).

PI uses the following iteration formula to compute estimates $\boldsymbol{\hat{\pi}}_{s}^{(0)},\boldsymbol{\hat{\pi}}_{s}^{(1)},\ldots,\boldsymbol{\hat{\pi}}_{s}^{(T)}$ for $\boldsymbol{\pi}_{s}$, where $T$ is a parameter:

$$
\displaystyle\boldsymbol{\hat{\pi}}_{s}^{(0)}=\boldsymbol{e}_{s},\quad\,\,\boldsymbol{\hat{\pi}}_{s}^{(L+1)}=(1-\alpha)\mathbf{P}\boldsymbol{\hat{\pi}}_{s}^{(L)}+\alpha\boldsymbol{e}_{s},\,\,\forall L\geq 0.
$$

Here, $\boldsymbol{\hat{\pi}}_{s}^{(L)}$ equals the probability distribution of an RWR starting from $s$ after $L$ steps. Note that since $\mathbf{P}$ has exactly $m$ nonzero entries, Eqn. (8) can be computed in $O(m)$ time. Furthermore, it can be shown that one can set $T=\Theta\left(\frac{1}{\alpha}\log\frac{1}{\varepsilon}\right)$ to ensure that $\left\lVert\boldsymbol{\hat{\pi}}_{s}^{(T)}-\boldsymbol{\pi}_{s}\right\rVert_{1}\leq\varepsilon$. Thus, by returning $\boldsymbol{\hat{\pi}}_{s}^{(T)}$ as an estimate for $\boldsymbol{\pi}_{s}$, PI solves the SSPPR query with $\ell_{1}$ -error bound $\varepsilon$ in $O(mT)=O\left(\frac{m}{\alpha}\log\frac{1}{\varepsilon}\right)$ time.

Cumulative Power Iteration. We additionally introduce a variant of PI since it can be seen as the prototype of Forward Push (see Sec. III-C). We call it Cumulative Power Iteration (CPI) [^41] to distinguish it from Standard Power Iteration (SPI) discussed above. CPI approaches $\boldsymbol{\pi}_{s}$ using Eqn. (4): $\boldsymbol{\pi}_{s}=\alpha\sum_{\ell=0}^{\infty}(1-\alpha)^{\ell}\mathbf{P}^{\ell}\boldsymbol{e}_{s}$. To this end, we define:

$$
\displaystyle\boldsymbol{r}^{(L)}
$$

$$
\displaystyle=(1-\alpha)^{L}\mathbf{P}^{L}\boldsymbol{e}_{s},
$$
$$
\displaystyle\boldsymbol{\hat{\pi}}_{s}^{(L)}
$$

$$
\displaystyle=\alpha\sum_{\ell=0}^{L-1}\boldsymbol{r}^{(\ell)}=\alpha\sum_{\ell=0}^{L-1}(1-\alpha)^{\ell}\mathbf{P}^{\ell}\boldsymbol{e}_{s}.
$$

We can compute $\boldsymbol{r}^{(L)}$ and $\boldsymbol{\hat{\pi}}_{s}^{(L)}$ iteratively as follows:

$$
\displaystyle\boldsymbol{r}^{(0)}
$$

$$
\displaystyle=\boldsymbol{e}_{s},
$$
$$
\displaystyle\quad\boldsymbol{r}^{(L+1)}
$$

$$
\displaystyle=(1-\alpha)\mathbf{P}\boldsymbol{r}^{(L)},
$$
$$
\displaystyle\boldsymbol{\hat{\pi}}_{s}^{(0)}
$$

$$
\displaystyle=\boldsymbol{0},
$$
$$
\displaystyle\quad\boldsymbol{\hat{\pi}}_{s}^{(L+1)}
$$

$$
\displaystyle=\boldsymbol{\hat{\pi}}_{s}^{(L)}+\alpha\boldsymbol{r}^{(L)}.
$$

Similar to SPI, each iteration of CPI also takes $O(m)$ work.

Again, we choose a parameter $T$, compute $\boldsymbol{\hat{\pi}}_{s}^{(0)}$ through $\boldsymbol{\hat{\pi}}_{s}^{(T)}$ and return $\boldsymbol{\hat{\pi}}_{s}^{(T)}$ as the final estimate. The $\ell_{1}$ -error between $\boldsymbol{\hat{\pi}}_{s}^{(T)}$ and $\boldsymbol{\pi}_{s}$ can be precisely calculated as

$$
\displaystyle\left\lVert\boldsymbol{\hat{\pi}}_{s}^{(T)}-\boldsymbol{\pi}_{s}\right\rVert_{1}=\left\lVert\alpha\sum_{\ell=T}^{\infty}(1-\alpha)^{\ell}\mathbf{P}^{\ell}\boldsymbol{e}_{s}\right\rVert_{1}
$$

$$
\displaystyle=
$$

$$
\displaystyle\alpha\sum_{\ell=T}^{\infty}(1-\alpha)^{\ell}\left\lVert\mathbf{P}^{\ell}\boldsymbol{e}_{s}\right\rVert_{1}=\alpha\sum_{\ell=T}^{\infty}(1-\alpha)^{\ell}=(1-\alpha)^{T}.
$$

Thus, we have $\left\lVert\boldsymbol{\hat{\pi}}_{s}^{(T)}-\boldsymbol{\pi}_{s}\right\rVert_{1}\leq\varepsilon$ if $T=\Theta\left(\frac{1}{\alpha}\log\frac{1}{\varepsilon}\right)$, so CPI guarantees an $\ell_{1}$ -error of $\varepsilon$ in $O\left(\frac{m}{\alpha}\log\frac{1}{\varepsilon}\right)$ time.

Reserve and Residue Vectors in CPI. By Eqns. (9) and (10), the following invariant holds for any $L\geq 0$:

$$
\displaystyle\boldsymbol{\pi}_{s}=\boldsymbol{\hat{\pi}}_{s}^{(L)}+\mathbf{\Pi}\boldsymbol{r}^{(L)}.
$$

Motivated by this, $\boldsymbol{r}^{(L)}$ can be regarded as the residue vector associated with $\boldsymbol{\hat{\pi}}_{s}^{(L)}$, the reserve vector. The reserve vector $\boldsymbol{\hat{\pi}}_{s}^{(L)}$ is always an underestimate of $\boldsymbol{\pi}_{s}$, where the associated residue vector determines the error between them.

Push-Based Interpretation of CPI. Let us inspect Eqns. (11) and (12) more carefully. By examining them from the perspective of an individual node $v\in V$, we find that, in effect, $\alpha$ fraction of $\boldsymbol{r}^{(L)}(v)$ (its residue) is converted to $\boldsymbol{\hat{\pi}}_{s}^{(L+1)}(v)$ (its reserve) and the rest $(1-\alpha)$ fraction is propagated evenly to the out-neighbors of $v$ in $\boldsymbol{r}^{(L+1)}$ (their residues) in each iteration. This process can be defined as a push operation, an important atomic operation widely used in recent algorithms. From this point of view, CPI can be seen as performing pushes synchronously to all nodes in $V$ in each iteration.

In practice, the two versions of PI have nearly identical effects, in the sense that they both solve the SSPPR query with $\ell_{1}$ -error bound $\varepsilon$ in time $O\left(\frac{m}{\alpha}\log\frac{1}{\varepsilon}\right)$. A minor technical distinction is that all estimated vectors in SPI are valid probability distributions, whereas, in CPI, they are underestimates of $\boldsymbol{\pi}_{s}$. Besides, the implementation of SPI is simpler as it does not need to maintain the additional residue vector. Thus, SPI may be more efficient in practice. Nevertheless, these two algorithms are often not finely distinguished and are collectively referred to as PI in the literature.

### III-C Forward Push

Forward Push (FP) [^5] [^6] is a local-push method that can answer SSPPR queries without having to scan the whole graph. In essence, FP can be viewed as an iterative coordinate solver [^9] and also a variant of the Gauss-Seidel method [^42]. For our purpose, it is best understood as a local version of CPI, as explained thoroughly in [^43] and briefly below.

Recall that CPI performs pushes to all nodes synchronously in each iteration. On the other hand, FP modifies this procedure and performs pushes asynchronously. It only performs one push operation for a single node at a time and allows the nodes to be pushed in a less structured order. Intuitively, nodes with larger reserve values should take priority to be pushed.

Specifically, the pseudocode of FP is shown in Alg. 1, which receives an extra parameter $r_{\mathrm{max}}^{(\mathrm{f})}$ as input. FP initializes the reserve and residue vectors as $\boldsymbol{\hat{\pi}}^{(\mathrm{f})}_{s}=\boldsymbol{0},\boldsymbol{r}^{(\mathrm{f})}_{s}=\boldsymbol{e}_{s}$ in Line 1. After that, FP performs push operations to nodes $v$ with $\frac{\boldsymbol{r}^{(\mathrm{f})}_{s}(v)}{d_{\mathrm{out}}(v)}>r_{\mathrm{max}}^{(\mathrm{f})}$ in an arbitrary order in Lines 1 to 1. When no nodes $v$ satisfy the condition $\frac{\boldsymbol{r}^{(\mathrm{f})}_{s}(v)}{d_{\mathrm{out}}(v)}>r_{\mathrm{max}}^{(\mathrm{f})}$ in Line 1, FP terminates the loop and returns the reserve and residue vectors in Line 1. Fig. 1 illustrates a running example of FP. In the following, we analyze the properties of this simple algorithm.

Algorithm 1 Forward Push [^5] [^6]

Input: graph $G=(V,E)$, source node $s$, teleportation probability $\alpha$, threshold $r_{\mathrm{max}}^{(\mathrm{f})}$

Output: reserve vector $\boldsymbol{\hat{\pi}}^{(\mathrm{f})}_{s}$ and residue vector $\boldsymbol{r}^{(\mathrm{f})}_{s}$

 $\boldsymbol{\hat{\pi}}^{(\mathrm{f})}_{s}\leftarrow\boldsymbol{0},\boldsymbol{r}^{(\mathrm{f})}_{s}\leftarrow\boldsymbol{e}_{s}$

while *$\exists v\in V$ s.t. $\frac{\boldsymbol{r}^{(\mathrm{f})}_{s}(v)}{d_{\mathrm{out}}(v)}>r_{\mathrm{max}}^{(\mathrm{f})}$* do

    pick an arbitrary node $v$ with $\frac{\boldsymbol{r}^{(\mathrm{f})}_{s}(v)}{d_{\mathrm{out}}(v)}>r_{\mathrm{max}}^{(\mathrm{f})}$

    $r\leftarrow\boldsymbol{r}^{(\mathrm{f})}_{s}(v),\boldsymbol{r}^{(\mathrm{f})}_{s}(v)\leftarrow 0$     $\boldsymbol{\hat{\pi}}^{(\mathrm{f})}_{s}(v)\leftarrow\boldsymbol{\hat{\pi}}^{(\mathrm{f})}_{s}(v)+\alpha r$

    for *each $u\in\mathcal{N}_{\mathrm{out}}(v)$* do

       $\boldsymbol{r}^{(\mathrm{f})}_{s}(u)\leftarrow\boldsymbol{r}^{(\mathrm{f})}_{s}(u)+\frac{1-\alpha}{d_{\mathrm{out}}(v)}\cdot r$

return $\boldsymbol{\hat{\pi}}^{(\mathrm{f})}_{s}$ and $\boldsymbol{r}^{(\mathrm{f})}_{s}$

![Refer to caption](https://arxiv.org/html/2403.05198v1/running_example_FP_3.png)

(a) Initial State

Invariant. Recall that CPI satisfies the invariant $\boldsymbol{\pi}_{s}=\boldsymbol{\hat{\pi}}_{s}^{(L)}+\mathbf{\Pi}\boldsymbol{r}^{(L)}$ (Eqn. (13)). Likewise, the following invariant holds before and after each push operation in FP:

$$
\displaystyle\boldsymbol{\pi}_{s}=\boldsymbol{\hat{\pi}}^{(\mathrm{f})}_{s}+\mathbf{\Pi}\boldsymbol{r}^{(\mathrm{f})}_{s}.
$$

This key invariant can be justified using induction and the decomposition theorem (Eqn. (6)).

Complexity. The total cost of performing pushes dominates the complexity of FP. Note that Line 1 constrains that each pushed node $v$ must satisfy $\frac{\boldsymbol{r}^{(\mathrm{f})}_{s}(v)}{d_{\mathrm{out}}(v)}>r_{\mathrm{max}}^{(\mathrm{f})}$, so one push operation for $v$ increases $\left\lVert\boldsymbol{\hat{\pi}}^{(\mathrm{f})}_{s}\right\rVert_{1}$ by at least $\alpha\cdot d_{\mathrm{out}}(v)\cdot r_{\mathrm{max}}^{(\mathrm{f})}$ at the cost of $O\big(d_{\mathrm{out}}(v)\big)$ time. Therefore, $\alpha\cdot r_{\mathrm{max}}^{(\mathrm{f})}$ can be viewed as the “unit cost-benefit” of the push operations in terms of increasing $\left\lVert\boldsymbol{\hat{\pi}}^{(\mathrm{f})}_{s}\right\rVert_{1}$. On the other hand, as the invariant (Eqn. (14)) implies, $\boldsymbol{\hat{\pi}}^{(\mathrm{f})}_{s}$ is always an underestimate of $\boldsymbol{\pi}_{s}$, and thus $\left\lVert\boldsymbol{\hat{\pi}}^{(\mathrm{f})}_{s}\right\rVert_{1}\leq\left\lVert\boldsymbol{\pi}_{s}\right\rVert_{1}=1$. Hence, the complexity of pushes is bounded by this upper bound of $\left\lVert\boldsymbol{\hat{\pi}}^{(\mathrm{f})}_{s}\right\rVert_{1}$ divided by the unit cost-benefit $\alpha\cdot r_{\mathrm{max}}^{(\mathrm{f})}$, which equals $O\left(1\Big/\left(\alpha\cdot r_{\mathrm{max}}^{(\mathrm{f})}\right)\right)$.

Error Bound. The condition in Line 1 also implies that, after termination, the residue vector satisfies $\boldsymbol{r}^{(\mathrm{f})}_{s}(v)\leq r_{\mathrm{max}}^{(\mathrm{f})}\cdot d_{\mathrm{out}}(v),\forall v\in V$. This implies that $\left\lVert\boldsymbol{\hat{\pi}}^{(\mathrm{f})}_{s}-\boldsymbol{\pi}_{s}\right\rVert_{1}=\left\lVert\mathbf{\Pi}\boldsymbol{r}^{(\mathrm{f})}_{s}\right\rVert_{1}\leq\left\lVert\mathbf{\Pi}\cdot r_{\mathrm{max}}^{(\mathrm{f})}\cdot\mathbf{D}\boldsymbol{1}\right\rVert_{1}=m\cdot r_{\mathrm{max}}^{(\mathrm{f})}$. In light of this, one can set $r_{\mathrm{max}}^{(\mathrm{f})}=\varepsilon/m$ and invoke FP to answer the SSPPR query with $\ell_{1}$ -error bound $\varepsilon$. Under such settings, the time complexity becomes $O\big(m/(\alpha\varepsilon)\big)$.

### III-D Reverse Power Iteration

Regarding STPPR, it is natural to ask whether a method called reverse power iteration (RPI) exists that works analogously to PI. Fortunately, the answer is positive. Similar to PI, RPI makes use of Eqn. (5): $\boldsymbol{\pi}^{-1}_{t}=\alpha\sum_{\ell=0}^{\infty}(1-\alpha)^{\ell}\left(\mathbf{P}^{\top}\right)^{\ell}\boldsymbol{e}_{t}$. Following CPI and using iterations akin to Eqns. (11) and (12), the estimate $\boldsymbol{\hat{\pi}}_{t}^{(-T)}=\alpha\sum_{\ell=0}^{T-1}(1-\alpha)^{\ell}\left(\mathbf{P}^{\top}\right)^{\ell}\boldsymbol{e}_{t}$ can be computed in $O(mT)$ time. Here, $\boldsymbol{\hat{\pi}}_{t}^{(-T)}$ denotes the $T$ -th estimate for $\boldsymbol{\pi}^{-1}_{t}$.

However, since $\mathbf{P}^{\top}$ is no longer column-stochastic as $\mathbf{P}$ does, the error between $\boldsymbol{\hat{\pi}}_{t}^{(-T)}$ and $\boldsymbol{\pi}^{-1}_{t}$ is a bit different. It turns out that considering the absolute error (instead of $\ell_{1}$ -error) between them is easier:

$$
\displaystyle\left\lVert\boldsymbol{\hat{\pi}}_{t}^{(-T)}-\boldsymbol{\pi}^{-1}_{t}\right\rVert_{\infty}\leq\left\lVert\alpha\sum_{\ell=T}^{\infty}(1-\alpha)^{\ell}\left(\mathbf{P}^{\top}\right)^{\ell}\boldsymbol{1}\right\lVert_{\infty}
$$

$$
\displaystyle=
$$

$$
\displaystyle\left\lVert\alpha\sum_{\ell=T}^{\infty}(1-\alpha)^{\ell}\boldsymbol{1}\right\rVert_{\infty}=(1-\alpha)^{T}.
$$

To guarantee that $\left\lVert\boldsymbol{\hat{\pi}}_{t}^{(-T)}-\boldsymbol{\pi}^{-1}_{t}\right\rVert_{\infty}\leq\varepsilon$, it suffices to set $T=\left\lceil\log_{1-\alpha}\varepsilon\right\rceil=\Theta\left(\frac{1}{\alpha}\log\frac{1}{\varepsilon}\right)$. In a word, RPI solves the STPPR query with absolute error bound $\varepsilon$ deterministically in $O(mT)=O\left(\frac{m}{\alpha}\log\frac{1}{\varepsilon}\right)$ time.

Analogous to CPI, RPI can also be viewed as performing reverse pushes to all nodes synchronously in each iteration. A reverse push operation to node $v$ also converts $\alpha$ fraction of its residue to its reserve but propagates the rest in a different way, which can be thought of as pushing according to the matrix $\mathbf{P}^{\top}$ instead of $\mathbf{P}$. To explain, as the $(i,j)$ -th entry of $\mathbf{P}^{\top}$ equals $\mathds{1}\big\{(v_{j},v_{i})\in E\big\}\big/d_{\mathrm{out}}(v_{j})$, the reverse push to $v$ should increment the residue of each $u\in\mathcal{N}_{\mathrm{in}}(v)$ by $(1-\alpha)/d_{\mathrm{out}}(u)\cdot\boldsymbol{r}^{(\mathrm{b})}(v)$. RPI can also be implemented similarly to SPI, with the same asymptotic complexity.

### III-E Backward Push

Backward Push (BP) [^36] [^37] is a local-push algorithm for answering the STPPR query. It can be viewed as the reverse version of FP or the local version of RPI. The framework of BP is extremely similar to that of FP, as shown in Alg. 2. It also receives a parameter $r_{\mathrm{max}}^{(\mathrm{b})}$ to determine which nodes to push, but the condition in Line 2 is slightly different.

Algorithm 2 Backward Push [^36] [^37]

Input: graph $G=(V,E)$, target node $t$, teleportation probability $\alpha$, threshold $r_{\mathrm{max}}^{(\mathrm{b})}$

Output: reserve vector $\boldsymbol{\hat{\pi}}^{(\mathrm{b})}_{t}$ and residue vector $\boldsymbol{r}^{(\mathrm{b})}_{t}$

 $\boldsymbol{\hat{\pi}}^{(\mathrm{b})}_{t}\leftarrow\boldsymbol{0},\boldsymbol{r}^{(\mathrm{b})}_{t}\leftarrow\boldsymbol{e}_{t}$

while *$\exists v\in V$ s.t. $\boldsymbol{r}^{(\mathrm{b})}_{t}(v)>r_{\mathrm{max}}^{(\mathrm{b})}$* do

    pick an arbitrary node $v$ with $\boldsymbol{r}^{(\mathrm{b})}_{t}(v)>r_{\mathrm{max}}^{(\mathrm{b})}$

    $r\leftarrow\boldsymbol{r}^{(\mathrm{b})}_{t}(v),\boldsymbol{r}^{(\mathrm{b})}_{t}(v)\leftarrow 0$     $\boldsymbol{\hat{\pi}}^{(\mathrm{b})}_{t}(v)\leftarrow\boldsymbol{\hat{\pi}}^{(\mathrm{b})}_{t}(v)+\alpha r$

    for *each $u\in\mathcal{N}_{\mathrm{in}}(v)$* do

       $\boldsymbol{r}^{(\mathrm{b})}_{t}(u)\leftarrow\boldsymbol{r}^{(\mathrm{b})}_{t}(u)+\frac{1-\alpha}{d_{\mathrm{out}}(u)}\cdot r$

return $\boldsymbol{\hat{\pi}}^{(\mathrm{b})}_{t}$ and $\boldsymbol{r}^{(\mathrm{b})}_{t}$

Invariant. Analogous to FP, BP also admits an invariant between the reserve vector $\boldsymbol{\hat{\pi}}^{(\mathrm{b})}_{t}$ and the residue vector $\boldsymbol{r}^{(\mathrm{b})}_{t}$:

$$
\displaystyle\boldsymbol{\pi}^{-1}_{t}=\boldsymbol{\hat{\pi}}^{(\mathrm{b})}_{t}+\mathbf{\Pi}^{\top}\boldsymbol{r}^{(\mathrm{b})}_{t}.
$$

The proof of this invariant also relies on induction and the decomposition theorem (Eqn. (7)).

Complexity. Similarly, the complexity of BP is also dominated by performing pushes. Following the reasoning in FP, we find that a push operation on node $v$ increases $\boldsymbol{\hat{\pi}}^{(\mathrm{b})}_{t}(v)$ by at least $\alpha\cdot r_{\mathrm{max}}^{(\mathrm{b})}$ in $O\big(d_{\mathrm{in}}(v)\big)$ time. By considering each node $v\in V$ separately, we attain the total complexity of

$$
\displaystyle O\left(\sum_{v\in V}\frac{\boldsymbol{\pi}^{-1}_{t}(v)}{\alpha\cdot r_{\mathrm{max}}^{(\mathrm{b})}}\cdot d_{\mathrm{in}}(v)\right)=O\left(\frac{\sum_{v\in V}\boldsymbol{\pi}^{-1}_{t}(v)\cdot d_{\mathrm{in}}(v)}{\alpha\cdot r_{\mathrm{max}}^{(\mathrm{b})}}\right).
$$

To derive a simpler complexity, we can consider BP’s average complexity over all target nodes $t\in V$, which equals

$$
\displaystyle O\left(\frac{1/n}{\alpha\cdot r_{\mathrm{max}}^{(\mathrm{b})}}\sum_{v\in V}d_{\mathrm{in}}(v)\sum_{t\in V}\boldsymbol{\pi}_{v}(t)\right)=O\left(\frac{m/n}{\alpha\cdot r_{\mathrm{max}}^{(\mathrm{b})}}\right).
$$

Error Bound. As Line 2 implies, after termination, the residue vector satisfies $\boldsymbol{r}^{(\mathrm{b})}_{t}(v)\leq r_{\mathrm{max}}^{(\mathrm{b})},\,\,\forall v\in V$. It follows that

$$
\displaystyle\left\lVert\boldsymbol{\hat{\pi}}^{(\mathrm{b})}_{t}-\boldsymbol{\pi}^{-1}_{t}\right\rVert_{\infty}=\left\lVert\mathbf{\Pi}^{\top}\boldsymbol{r}^{(\mathrm{b})}_{t}\right\rVert_{\infty}
$$

$$
\displaystyle\leq
$$

$$
\displaystyle\left\lVert\mathbf{\Pi}^{\top}\cdot r_{\mathrm{max}}^{(\mathrm{b})}\cdot\boldsymbol{1}\right\rVert_{\infty}=r_{\mathrm{max}}^{(\mathrm{b})}\cdot\lVert\boldsymbol{1}\rVert_{\infty}=r_{\mathrm{max}}^{(\mathrm{b})}.
$$

Therefore, one can invoke BP to answer the STPPR query with absolute error bound $\varepsilon$ by setting $r_{\mathrm{max}}^{(\mathrm{b})}=\varepsilon$, with an average complexity of $O\left(m/n/(\alpha\varepsilon)\right)$.

### III-F FP and BP on Undirected Graphs

FP and BP exhibit some additional properties on undirected graphs. In particular, using the specific properties discussed at the end of Sec. II-D, we can simplify the error bound of FP and the complexity bound of BP as follows.

FP on Undirected Graphs. The error bound of FP can be specific to each node on undirected graphs. To see this, note that $\boldsymbol{\hat{\pi}}^{(\mathrm{f})}_{s}=\boldsymbol{\pi}_{s}-\mathbf{\Pi}\boldsymbol{r}^{(\mathrm{f})}_{s}$ is lower bounded by

$$
\displaystyle\boldsymbol{\pi}_{s}-\mathbf{\Pi}\cdot r_{\mathrm{max}}^{(\mathrm{f})}\cdot\mathbf{D}\boldsymbol{1}=\boldsymbol{\pi}_{s}-r_{\mathrm{max}}^{(\mathrm{f})}\cdot\mathbf{D}\boldsymbol{1},
$$

where we used $\mathbf{\Pi}\mathbf{D}\boldsymbol{1}=\mathbf{D}\boldsymbol{1}$ for undirected graphs. Thus,

$$
\displaystyle\boldsymbol{\pi}_{s}(v)-r_{\mathrm{max}}^{(\mathrm{f})}\cdot d(v)\leq\boldsymbol{\hat{\pi}}^{(\mathrm{f})}_{s}(v)\leq\boldsymbol{\pi}_{s}(v),\quad\forall v\in V,
$$
$$
\displaystyle\Leftrightarrow
$$

$$
\displaystyle\frac{\boldsymbol{\pi}_{s}(v)}{d(v)}-r_{\mathrm{max}}^{(\mathrm{f})}\leq\frac{\boldsymbol{\hat{\pi}}^{(\mathrm{f})}_{s}(v)}{d(v)}\leq\frac{\boldsymbol{\pi}_{s}(v)}{d(v)},\quad\forall v\in V.
$$

This type of result is called degree-normalized absolute error, which means that the values $\boldsymbol{\pi}_{s}(v)/d(v)$ are estimated with absolute error bounds. We conclude that FP can solve the SSPPR query with degree-normalized absolute error bound $\varepsilon$ in $O\big(1/(\alpha\varepsilon)\big)$ time on undirected graphs by setting $r_{\mathrm{max}}^{(\mathrm{f})}=\varepsilon$. Although this type of error measure allows larger errors for nodes with larger degrees, it is useful for some applications (e.g., for local graph clustering [^5]).

BP on Undirected Graphs. Eqn. (16) can be directly simplified with the aid of Thm. 3 on undirected graphs, yielding $O\left(\sum_{v\in V}d(t)\cdot\boldsymbol{\pi}_{t}(v)\Big/\left(\alpha\cdot r_{\mathrm{max}}^{(\mathrm{b})}\right)\right)=O\big(d(t)/(\alpha\varepsilon)\big)$. Compared to $O\left(m/n/(\alpha\varepsilon)\right)$ on directed graphs, this complexity is parameterized instead of average.

### III-G Summary and Comparison of the Basic Techniques

We summarize the considered query types and complexities of the five basic techniques for different error bounds in Tbl. II. We remark that MC applies to probabilistic queries, while others pertain to deterministic ones. Also, the bounds of BP are averaged over all target nodes $t\in V$. Furthermore, we highlight the complexities derived above in red. The remaining complexities in black are derived using conversions between the error measures (discussed at the end of Sec. II-C) and thus may be pessimistic.

TABLE II: Summary of Basic Techniques (Black Complexities Are Derived using Error-Bound Conversions and May Be Inferior)

<table><thead><tr><th rowspan="2">Technique</th><th rowspan="2">Query Type</th><th colspan="4">Error Measures</th></tr><tr><th><math><semantics><msub><mi>ℓ</mi> <mn>𝟏</mn></msub> <annotation>\boldsymbol{\ell_{1}}</annotation></semantics></math></th><th>absolute</th><th>relative</th><th>degree-normalized absolute</th></tr></thead><tbody><tr><td>MC <sup><a href="#fn:35">35</a></sup></td><td>SSPPR (probabilistic)</td><td><math><semantics><mrow><mi>O</mi> <mo>⁡</mo> <mrow><mo>(</mo><mstyle><mfrac><mrow><msup><mi>n</mi> <mn>2</mn></msup> <mo></mo><mrow><mi>log</mi> <mo>⁡</mo> <mrow><mo>(</mo><mrow><mi>n</mi> <mo>/</mo> <mi>δ</mi></mrow><mo>)</mo></mrow></mrow></mrow> <mrow><mi>α</mi> <mo></mo><msup><mi>ε</mi> <mn>2</mn></msup></mrow></mfrac></mstyle><mo>)</mo></mrow></mrow> <annotation>O\left(\dfrac{n^{2}\log(n/\delta)}{\alpha\varepsilon^{2}}\right)</annotation></semantics></math></td><td><math><semantics><mrow><mi>O</mi> <mo>⁡</mo> <mrow><mo>(</mo><mstyle><mfrac><mrow><mi>log</mi> <mo>⁡</mo> <mrow><mo>(</mo><mrow><mi>n</mi> <mo>/</mo> <mi>δ</mi></mrow><mo>)</mo></mrow></mrow> <mrow><mi>α</mi> <mo></mo><msup><mi>ε</mi> <mn>2</mn></msup></mrow></mfrac></mstyle><mo>)</mo></mrow></mrow> <annotation>O\left(\dfrac{\log(n/\delta)}{\alpha\varepsilon^{2}}\right)</annotation></semantics></math></td><td><math><semantics><mrow><mi>O</mi> <mo>⁡</mo> <mrow><mo>(</mo><mstyle><mfrac><mrow><mi>log</mi> <mo>⁡</mo> <mrow><mo>(</mo><mrow><mi>n</mi> <mo>/</mo> <mi>δ</mi></mrow><mo>)</mo></mrow></mrow> <mrow><mi>α</mi> <mo></mo><msup><mi>ε</mi> <mn>2</mn></msup> <mo></mo><mi>μ</mi></mrow></mfrac></mstyle><mo>)</mo></mrow></mrow> <annotation>O\left(\dfrac{\log(n/\delta)}{\alpha\varepsilon^{2}\mu}\right)</annotation></semantics></math></td><td><math><semantics><mrow><mi>O</mi> <mo>⁡</mo> <mrow><mo>(</mo><mstyle><mfrac><mrow><mi>log</mi> <mo>⁡</mo> <mrow><mo>(</mo><mrow><mi>n</mi> <mo>/</mo> <mi>δ</mi></mrow><mo>)</mo></mrow></mrow> <mrow><mi>α</mi> <mo></mo><msup><mi>ε</mi> <mn>2</mn></msup></mrow></mfrac></mstyle><mo>)</mo></mrow></mrow> <annotation>O\left(\dfrac{\log(n/\delta)}{\alpha\varepsilon^{2}}\right)</annotation></semantics></math></td></tr><tr><td>PI <sup><a href="#fn:1">1</a></sup></td><td>SSPPR</td><td><math><semantics><mrow><mi>O</mi> <mo>⁡</mo> <mrow><mo>(</mo><mrow><mstyle><mfrac><mi>m</mi> <mi>α</mi></mfrac></mstyle> <mo></mo><mrow><mi>log</mi> <mo>⁡</mo> <mstyle><mfrac><mn>1</mn> <mi>ε</mi></mfrac></mstyle></mrow></mrow><mo>)</mo></mrow></mrow> <annotation>O\left(\dfrac{m}{\alpha}\log\dfrac{1}{\varepsilon}\right)</annotation></semantics></math></td><td><math><semantics><mrow><mi>O</mi> <mo>⁡</mo> <mrow><mo>(</mo><mrow><mstyle><mfrac><mi>m</mi> <mi>α</mi></mfrac></mstyle> <mo></mo><mrow><mi>log</mi> <mo>⁡</mo> <mstyle><mfrac><mn>1</mn> <mi>ε</mi></mfrac></mstyle></mrow></mrow><mo>)</mo></mrow></mrow> <annotation>O\left(\dfrac{m}{\alpha}\log\dfrac{1}{\varepsilon}\right)</annotation></semantics></math></td><td><math><semantics><mrow><mi>O</mi> <mo>⁡</mo> <mrow><mo>(</mo><mrow><mstyle><mfrac><mi>m</mi> <mi>α</mi></mfrac></mstyle> <mo></mo><mrow><mi>log</mi> <mo>⁡</mo> <mstyle><mfrac><mn>1</mn> <mrow><mi>ε</mi> <mo></mo><mi>μ</mi></mrow></mfrac></mstyle></mrow></mrow><mo>)</mo></mrow></mrow> <annotation>O\left(\dfrac{m}{\alpha}\log\dfrac{1}{\varepsilon\mu}\right)</annotation></semantics></math></td><td><math><semantics><mrow><mi>O</mi> <mo>⁡</mo> <mrow><mo>(</mo><mrow><mstyle><mfrac><mi>m</mi> <mi>α</mi></mfrac></mstyle> <mo></mo><mrow><mi>log</mi> <mo>⁡</mo> <mstyle><mfrac><mn>1</mn> <mi>ε</mi></mfrac></mstyle></mrow></mrow><mo>)</mo></mrow></mrow> <annotation>O\left(\dfrac{m}{\alpha}\log\dfrac{1}{\varepsilon}\right)</annotation></semantics></math></td></tr><tr><td>FP <sup><a href="#fn:6">6</a></sup></td><td>SSPPR</td><td><math><semantics><mrow><mi>O</mi> <mo>⁡</mo> <mrow><mo>(</mo><mstyle><mfrac><mi>m</mi> <mrow><mi>α</mi> <mo></mo><mi>ε</mi></mrow></mfrac></mstyle><mo>)</mo></mrow></mrow> <annotation>O\left(\dfrac{m}{\alpha\varepsilon}\right)</annotation></semantics></math></td><td><math><semantics><mrow><mi>O</mi> <mo>⁡</mo> <mrow><mo>(</mo><mstyle><mfrac><mi>m</mi> <mrow><mi>α</mi> <mo></mo><mi>ε</mi></mrow></mfrac></mstyle><mo>)</mo></mrow></mrow> <annotation>O\left(\dfrac{m}{\alpha\varepsilon}\right)</annotation></semantics></math></td><td><math><semantics><mrow><mi>O</mi> <mo>⁡</mo> <mrow><mo>(</mo><mstyle><mfrac><mi>m</mi> <mrow><mi>α</mi> <mo></mo><mi>ε</mi> <mo></mo><mi>μ</mi></mrow></mfrac></mstyle><mo>)</mo></mrow></mrow> <annotation>O\left(\dfrac{m}{\alpha\varepsilon\mu}\right)</annotation></semantics></math></td><td><math><semantics><mrow><mi>O</mi> <mo>⁡</mo> <mrow><mo>(</mo><mstyle><mfrac><mn>1</mn> <mrow><mi>α</mi> <mo></mo><mi>ε</mi></mrow></mfrac></mstyle><mo>)</mo></mrow></mrow> <annotation>O\left(\dfrac{1}{\alpha\varepsilon}\right)</annotation></semantics></math></td></tr><tr><td>RPI <sup><a href="#fn:37">37</a></sup></td><td>STPPR</td><td><math><semantics><mrow><mi>O</mi> <mo>⁡</mo> <mrow><mo>(</mo><mrow><mstyle><mfrac><mi>m</mi> <mi>α</mi></mfrac></mstyle> <mo></mo><mrow><mi>log</mi> <mo>⁡</mo> <mstyle><mfrac><mi>n</mi> <mi>ε</mi></mfrac></mstyle></mrow></mrow><mo>)</mo></mrow></mrow> <annotation>O\left(\dfrac{m}{\alpha}\log\dfrac{n}{\varepsilon}\right)</annotation></semantics></math></td><td><math><semantics><mrow><mi>O</mi> <mo>⁡</mo> <mrow><mo>(</mo><mrow><mstyle><mfrac><mi>m</mi> <mi>α</mi></mfrac></mstyle> <mo></mo><mrow><mi>log</mi> <mo>⁡</mo> <mstyle><mfrac><mn>1</mn> <mi>ε</mi></mfrac></mstyle></mrow></mrow><mo>)</mo></mrow></mrow> <annotation>O\left(\dfrac{m}{\alpha}\log\dfrac{1}{\varepsilon}\right)</annotation></semantics></math></td><td><math><semantics><mrow><mi>O</mi> <mo>⁡</mo> <mrow><mo>(</mo><mrow><mstyle><mfrac><mi>m</mi> <mi>α</mi></mfrac></mstyle> <mo></mo><mrow><mi>log</mi> <mo>⁡</mo> <mstyle><mfrac><mn>1</mn> <mrow><mi>ε</mi> <mo></mo><mi>μ</mi></mrow></mfrac></mstyle></mrow></mrow><mo>)</mo></mrow></mrow> <annotation>O\left(\dfrac{m}{\alpha}\log\dfrac{1}{\varepsilon\mu}\right)</annotation></semantics></math></td><td><math><semantics><mrow><mi>O</mi> <mo>⁡</mo> <mrow><mo>(</mo><mrow><mstyle><mfrac><mi>m</mi> <mi>α</mi></mfrac></mstyle> <mo></mo><mrow><mi>log</mi> <mo>⁡</mo> <mstyle><mfrac><mn>1</mn> <mi>ε</mi></mfrac></mstyle></mrow></mrow><mo>)</mo></mrow></mrow> <annotation>O\left(\dfrac{m}{\alpha}\log\dfrac{1}{\varepsilon}\right)</annotation></semantics></math></td></tr><tr><td>BP <sup><a href="#fn:37">37</a></sup></td><td>STPPR</td><td><math><semantics><mrow><mi>O</mi> <mo>⁡</mo> <mrow><mo>(</mo><mstyle><mfrac><mi>m</mi> <mrow><mi>α</mi> <mo></mo><mi>ε</mi></mrow></mfrac></mstyle><mo>)</mo></mrow></mrow> <annotation>O\left(\dfrac{m}{\alpha\varepsilon}\right)</annotation></semantics></math> (average)</td><td><math><semantics><mrow><mi>O</mi> <mo>⁡</mo> <mrow><mo>(</mo><mstyle><mfrac><mrow><mi>m</mi> <mo>/</mo> <mi>n</mi></mrow> <mrow><mi>α</mi> <mo></mo><mi>ε</mi></mrow></mfrac></mstyle><mo>)</mo></mrow></mrow> <annotation>O\left(\dfrac{m/n}{\alpha\varepsilon}\right)</annotation></semantics></math> (average)</td><td><math><semantics><mrow><mi>O</mi> <mo>⁡</mo> <mrow><mo>(</mo><mstyle><mfrac><mrow><mi>m</mi> <mo>/</mo> <mi>n</mi></mrow> <mrow><mi>α</mi> <mo></mo><mi>ε</mi> <mo></mo><mi>μ</mi></mrow></mfrac></mstyle><mo>)</mo></mrow></mrow> <annotation>O\left(\dfrac{m/n}{\alpha\varepsilon\mu}\right)</annotation></semantics></math> (average)</td><td><math><semantics><mrow><mi>O</mi> <mo>⁡</mo> <mrow><mo>(</mo><mstyle><mfrac><mrow><mi>m</mi> <mo>/</mo> <mi>n</mi></mrow> <mrow><mi>α</mi> <mo></mo><mi>ε</mi></mrow></mfrac></mstyle><mo>)</mo></mrow></mrow> <annotation>O\left(\dfrac{m/n}{\alpha\varepsilon}\right)</annotation></semantics></math> (average)</td></tr></tbody></table>

Generally speaking, PI and RPI are inefficient for massive graphs, as they need to scan the whole graph in each iteration. However, their dependence on $\varepsilon$, $O\big(\log(1/\varepsilon)\big)$, is the mildest among these methods. On the contrary, MC does not need to scan the graph, but its dependence on $\varepsilon$ is $O\left(1\big/\varepsilon^{2}\right)$, a considerable factor when $\varepsilon$ is small. The local-push methods seem to be a compromise between these two extremes: FP and BP work by locally exploring the graph, and their dependence on $\varepsilon$ is $O(1/\varepsilon)$. MC and the local-push methods are the key ingredients for designing sublinear algorithms.

Let us take a closer look at FP and BP. Despite their highly similar approaches, their properties have some key differences:

- In terms of error bounds, BP can elegantly provide absolute error guarantees, while FP only provides degree-normalized absolute error guarantees on undirected graphs. Note that although FP can provide $\ell_{1}$ -error guarantee on directed graphs, in this case, its complexity becomes proportional to $m$, and thus, it becomes nonlocal and inferior to PI.
- In terms of complexity, FP runs in $O\left(1\Big/\left(\alpha\cdot r_{\mathrm{max}}^{(\mathrm{f})}\right)\right)$ time, while BP’s complexity on directed graphs cannot be expressed compactly. BP exhibits a simple form of complexity only when considering average complexity or when applied to undirected graphs. In these two special cases, its complexity becomes $O\left(m/n\Big/\left(\alpha\cdot r_{\mathrm{max}}^{(\mathrm{b})}\right)\right)$ and $O\left(d(t)\Big/\left(\alpha\cdot r_{\mathrm{max}}^{(\mathrm{b})}\right)\right)$, resp.

## IV Overview of PPR Algorithms

The remainder of this paper focuses on more recent and advanced PPR algorithms. Although these algorithms cover various query types, error measures, and computing settings, they can be roughly divided into two categories by their methodologies: PPR algorithms based on PI and linear algebraic techniques and on local-push methods and MC.

1. PPR algorithms based on PI and linear algebraic techniques interpret PPR as the solution to Eqn. (1). They harness various numerical methods and optimization methods to expedite the solving process. Some works also leverage the structures of real-world graphs to simplify the problem. In general, these algorithms can obtain exact PPR values or approximate ones with tiny errors, but they need at least linear time in the graph size. Thus, they are more suitable for precomputing high-precision PageRank vectors or PPR matrices for small graphs. These works mainly consider improving the convergence speed and utilizing parallel/distributed settings to achieve higher scalability.
2. PPR algorithms based on local-push methods and MC rely on the random-walk-based interpretations of PPR. They use a combination of local-push methods and MC to derive approximate PPR values in a local manner. Although they cannot obtain extremely accurate results, some of them achieve sublinear complexities, which are more suitable for online queries and are particularly desirable for certain queries. These works mainly consider combining and optimizing the techniques of FP, BP, and MC to reduce the complexity, and some works also design effective precomputation to diminish the query cost.

We summarize the PPR algorithms surveyed below in Tbls. III and IV at the end of this paper.

## V Single-Source PPR Algorithms

The majority of recent PPR research concentrates on the SSPPR queries. We introduce them below based on the taxonomy of their techniques. For ease of presentation, unless explicitly stated otherwise, we assume that $\alpha$ is constant in the rest of this paper. We use $\widetilde{O}$ to denote the big- $O$ notation ignoring polylogarithmic factors.

### V-A SSPPR Algorithms Based on PI and Algebraic Techniques

This type of SSPPR algorithm starts from the algebraic definition of PPR (Eqn. (1)) and applies several advanced linear algebraic or optimization techniques to compute PPR vectors with small errors efficiently. However, their complexities are generally at least linear in the size of the graph.

PageRank Vector Computation. Several earlier works [^44] [^45] [^46] try to improve the efficiency of PI for computing the PageRank vector by leveraging linear algebraic techniques and the structural properties of real-world graphs. We refer to a detailed survey by Berkhin [^27] and the references therein for these algorithms. Additionally, Avrachenkov et al. [^30] use MC and its variants to estimate the PageRank vector, and Wu and Wei [^47] apply two advanced linear algebraic methods, Arnoldi and GMRES, to compute the PageRank vector.

Some earlier works for SSPPR include B\_LIN and NB\_LIN [^48], which leverage graph partitioning and low-rank approximation, and RPPR and BRPPR [^49], which use SPI on restricted subgraphs and expand the subgraph iteratively. Additionally, Jeh and Widom [^32] put forward a basic dynamic programming algorithm for computing SSPPR based on the decomposition theorem (Eqn. (6)). Sarlós et al. [^50] later improve this dynamic programming algorithm by applying Count-Min Sketch to the intermediate vectors. Under the framework of two-phase algorithms, the algorithms need to construct a database in the first precomputation phase and can only access the database instead of the graph in the second query phase. Their method reduces the database space from $O\big(n\log n\log(1/\delta)\big/\varepsilon^{2}\big)$ in their previous work [^35] to $O\big(n\log(1/\delta)/\varepsilon\big)$, where $\varepsilon$ denotes the absolute error bound. They demonstrate that this approach achieves optimal space usage for the database.

FastPPV, proposed by Zhu et al. [^51], supports a dynamic trade-off between efficiency and accuracy at query time based on scheduled approximation. Its basic idea is to partition and prioritize all paths from $s$ so that each iteration corresponds to a set of paths, and the critical paths can be considered in earlier iterations. To this end, FastPPV first carefully selects a set of hub nodes $H$ and conceptually categorizes the paths according to the number of hub nodes on the paths, and then precomputes the PPR vector of each hub node in a small subgraph. At query time, these vectors form the building blocks for the iterative computation, and the $\ell_{1}$ -error bound decreases exponentially.

Maehara et al. [^52] propose an SSPPR algorithm that exploits the graph structure and the iterative method GMRES. It first performs core-tree decomposition to transform the graph into a core and a small graph, then runs LU decomposition to the small graph, and finally applies GMRES to the whole graph, using the previous result as a preconditioner.

Jung et al. [^53] [^54] propose BEAR, a method that preprocesses several matrices and utilizes them via a block elimination approach. They propose two versions of BEAR, BEAR-Exact and BEAR-Approx, which give exact and approximate results, resp. The overall idea is to reduce the overhead of computing $\big(\mathbf{I}-(1-\alpha)\mathbf{P}\big)^{-1}$ (cf. Eqn. (2)) by using some linear algebraic techniques. In the preprocessing phase, BEAR-Exact reorders the nodes so that the matrix $\mathbf{I}-(1-\alpha)\mathbf{P}$ after transformation contains several easy-to-invert sub-matrices and uses the Schur complement method and LU decomposition to prepare auxiliary matrices. In the query phase, BEAR-Exact applies the block elimination method to compute the PPR vector efficiently while exploiting the preprocessed matrices. BEAR-Approx further heuristically omits some small entries in the preprocessed matrices to improve efficiency while sacrificing little accuracy in practice.

The follow-up work BePI by Jung et al. [^55] further improves over BEAR [^54]. BePI incorporates the advantages of preprocessing methods and iterative methods, i.e., high querying efficiency and scalability to large graphs, resp. Specifically, BePI combines the block elimination method (as used by BEAR) and GMRES, and it further applies some advanced techniques like reordering the nodes using the hub-and-spike structure of real-world graphs, sparsifying the Schur complement, and preconditioning a linear system. Compared to BEAR, BePI performs preprocessing much more efficiently and requires significantly less space. It is worth mentioning that BePI offers $\ell_{2}$ -error guarantee.

Proposed by Coşkun et al. [^56], Chopper modifies SPI by taking a linear combination of $\boldsymbol{\hat{\pi}}_{s}^{(0)},\boldsymbol{\hat{\pi}}_{s}^{(1)},\dots,\boldsymbol{\hat{\pi}}_{s}^{(T)}$ in Eqn. (8) to be an estimate of $\boldsymbol{\pi}_{s}$. They show that by assigning the coefficients of the combinations using Chebyshev polynomials, the obtained sequence of estimates can converge more rapidly, and the complexity of each iteration remains $O(m)$. Experimentally, Chopper can indeed reduce the number of required iterations significantly.

The TPA algorithm by Yoon et al. [^41] tries to approximate the PPR vector through Eqn. (4), as CPI does. It divides the summation into three parts, the family part, the neighbor part, and the stranger part, by limiting the index in three intervals from zero to infinity. The three parts are tackled differently, and in particular, the neighbor part is approximated by a factor times the family part, with the intuition that the block-wise structure of the real-world graphs causes the two parts to have similar distributions. TPA requires $O(n+m)$ memory space and takes $O\big(m\log(1/\varepsilon)\big)$ time for both preprocessing and online query, where $\varepsilon$ is the $\ell_{1}$ -error bound.

Chen et al. [^42] consider accelerating SSPPR computation using advanced optimization techniques. They propose FwdPushSOR, an improved version of FP based on Successive Over Relaxation [^40]. On undirected graphs, they formulate SSPPR computation as a convex optimization problem (cf. [^9]) and apply two momentum-based acceleration methods to it. To achieve a specified $\ell_{1}$ -error bound, their algorithms run in $\widetilde{O}\big(m\big/\sqrt{\alpha}\big)$ time, with $\alpha$ regarded as a parameter, improving over PI’s $\widetilde{O}(m/\alpha)$ bound.

### V-B SSPPR Algorithms Based on Local Push and MC

These algorithms mainly optimizes the vanilla FP algorithm and/or combines local-push techniques with MC for answering the SSPPR queries. By applying local pushes and randomness, they are potentially able to achieve sublinear complexities.

Pavel Berkhin [^57] put forward an approximate SSPPR algorithm called BCA, which is similar to FP but utilizes precomputed hub vectors to accelerate the computation. Conceptually, the pushes are performed as in standard FP, but the residues get blocked once they enter a hub node, where the hub nodes are selected beforehand. After performing the pushes, the obtained reserve vector is combined with the hub vectors of the hub nodes in accordance with the amount of blocked residues to obtain the final estimate. Chakrabarti further proposes HubRank [^58] to improve over Berkhin’s BCA [^57]. For preprocessing, HubRank carefully chooses some hub nodes according to the history query log statistics and runs MC to compute their PPR vectors as indices. At query time, the indices are used along with running PI over a small subgraph to yield the results.

FP with Exponential Convergence. Chung and Zhao [^59] consider optimizing FP when $r_{\mathrm{max}}^{(\mathrm{f})}$ is small. They propose to divide the process into $\Theta\left(\log\left(1\big/r_{\mathrm{max}}^{(\mathrm{f})}\right)\right)$ phases with exponentially decreasing thresholds of $1/2,1/4,1/8,\dots$ until it gets as small as $r_{\mathrm{max}}^{(\mathrm{f})}$. The obtained algorithm is shown to have an $O\left(m\log\left(1\big/r_{\mathrm{max}}^{(\mathrm{f})}\right)\right)$ complexity, similar to that of PI. Although this complexity is not sublinear, its dependence on $r_{\mathrm{max}}^{(\mathrm{f})}$ is $O\left(\log\left(1\big/r_{\mathrm{max}}^{(\mathrm{f})}\right)\right)$. Besides, Wu et al. [^43] revisits FP by revealing the connection between CPI and FP. They show that a common implementation of FP called FIFO-FP, which utilizes a queue to maintain the nodes to be pushed, actually admits a complexity of $O\left(\min\left(1\big/r_{\mathrm{max}}^{(\mathrm{f})},m\log\frac{1}{m\cdot r_{\mathrm{max}}^{(\mathrm{f})}}\right)\right)$. This implies that FIFO-FP is no worse than PI for answering the SSPPR query with $\ell_{1}$ -error bound. The authors also propose PowerPush, which unifies the advantages of both the local approach (FP) and the global approach (PI) to achieve superior efficiency and accuracy in practice. Specifically, PowerPush exploits FP’s asynchronous pushes to reduce the number of pushes and mimics PI to perform pushes in a cache-friendly way when most nodes have been involved.

FORA and Its Follow-Ups. FORA by Wang et al. [^60] [^61] is a representative algorithm for the SSPPR query with relative error guarantees. It combines FP and MC in an elegant and nontrivial way, and this “FP + MC” framework has been adopted by many subsequent works [^62] [^63] [^64] [^65] [^66] [^67] [^68] [^43] [^39] [^69] [^70], some of which are tailored for distributed environments (see Sec. X-B). Roughly speaking, FORA conducts random walks deterministically through FP before doing so randomly through MC, dramatically reducing the number of samples needed in MC. Technically, instead of ignoring the residues after FP, FORA replaces the $\mathbf{\Pi}$ matrix in the invariant of FP (Eqn. (14)) by PPR estimates obtained from MC. For the MC phase, FORA simulates several random walks from each node in $V$, where the number of walks depends on their residues. FORA’s overall complexity is shown to be $O\left(\min\left\{\frac{1}{\varepsilon}\sqrt{\frac{m\log(1/\delta)}{\mu}},\frac{\log(1/\delta)}{\varepsilon^{2}\mu}\right\}\right)$. Besides, a variant of FORA called FORA+ is proposed to achieve higher empirical performance, which avoids the cost of generating random walks on the fly by storing precomputed ones as indices.

Based on FORA, ResAcc is proposed by Lin et al. [^65] to achieve higher efficiency without using any indices. Motivated by the looping phenomenon discovered by BCA [^57], ResAcc first only performs pushes to the nodes that are no more than $h$ -hops away from $s$ (except $s$ itself after the first push) so that the residue can accumulate at $s$ and the $(h+1)$ -th layer, and then updates the reserves and residues for these nodes directly in accordance with the residue accumulated at $s$. After that, the residues at the $(h+1)$ -th layer are pushed, and FP is proceeded as usual, and finally, random walk samplings are performed as FORA does. Owing to the accumulated residue, ResAcc can perform fewer pushes and random walks in practice.

Wu et al. [^43] propose SpeedPPR to improve FORA by replacing FP in FORA’s first phase with their PowerPush [^43]. SpeedPPR runs in $O\left(m\log\frac{\log(1/\delta)}{\varepsilon^{2}\mu m}\right)$ time for relative error guarantees. Although this bound is not sublinear in $m$, SpeedPPR’s empirical efficiency surpasses that of previous algorithms, owing to PowerPush’s combination of FP and PI. Similar to FORA+, SpeedPPR can also store precomputed random walks as indices to achieve even higher performance.

Variational Formulation. Fountoulakis et al. [^9] revisit FP by deriving a variational formulation to explicitly express the problem solved by FP as an optimization problem. They show that FP is indeed an iterative coordinate solver for Eqn. (1). They further formulate an $\ell_{1}$ -regularized PageRank problem that decouples the sparsity of the solution from the specific algorithms, whose solution meets the requirements for local clustering in [^5]. The authors then apply the iterative shrinkage-thresholding algorithm to solve this problem and prove that it also solves the problem locally. Although they do not derive a directly better complexity bound, their formulation reveals a promising connection between local graph processing and numerical optimization. Whether one can apply some other optimization techniques to speed up the computation by a factor of $1/\sqrt{\alpha}$ is left as an open problem [^71]. By using conjugate directions and exploiting sparsity, [^72] gives some new bounds for the $\ell_{1}$ -regularized PageRank problem but does not fully resolve the open problem.

Optimizing the MC Technique. Liao et al. [^39] present a new implementation of MC by establishing a connection between PPR and the weights of random spanning forests of the graph. Armed with a new matrix forest theorem, they devise an approach for sampling a random spanning forest via simulating loop-erased $\alpha$ -discounted random walks. Such a sampling can substitute $n$ random walk samplings under FORA’s framework, and it takes expected $O\left(\sum_{v\in V}\pi(v,v)\right)$ time, which turns out to be significantly smaller than the corresponding $\Theta(n)$ complexity on real-world graphs. By modifying FORA’s FP phase and replacing the random walk samplings in the MC phase with their forest samplings, the authors propose FORAL, an efficient SSPPR algorithm that demonstrates robust performance for small $\alpha$.

A subsequent work by Liao et al. [^69] introduces two innovative techniques to reduce the variances of the MC estimators. The first technique performs several iterations of PI to MC estimates, leading to more accurate results. The second one, called the progressive sampling method, modifies the sampling scheme of MC by taking historical samplings into account. The authors give theoretical analyses of these techniques’ effects on variance reduction. They combine these techniques with their previous spanning forest sampling technique [^39] to propose novel and practically efficient SSPPR algorithms.

FP with Bounded Sensitivity. Epasto et al. [^23] consider computing approximate PPR vectors with provably bounded sensitivity to the edges in the undirected graph. To this end, they modify FP and restrict the maximum amount of residues being pushed along each edge. They show that this approach ensures low sensitivity and achieves a similar approximation quality as FP if the graph has a large minimum degree.

## VI Single-Target PPR Algorithms

So far, relatively few studies exist for the STPPR queries, which mainly rely on reverse local pushes. A basic difficulty is that MC cannot be used directly for STPPR computation.

Lofgren and Goel [^73] propose an implementation of BP using a Fibonacci heap. They prove that the complexity of this implementation can be alternatively bounded by $O\big(m\log(1/\varepsilon)\big)$, matching the complexity of RPI.

Wang et al. [^74] propose RBS, which applies randomness to the original BP. The key insight of RBS is that the cost of BP can be greatly reduced by only performing pushes to a randomly chosen subset of the in-neighbors of a node. For the STPPR query with relative error guarantees, this idea removes the dependence on $m/n$ in the average complexity of BP and yields an average complexity of $\widetilde{O}(1/\mu)$ or a parameterized complexity of $\widetilde{O}\big(n\boldsymbol{\pi}(t)/\mu\big)$ for node $t$ (treating $\varepsilon$ as a constant). These complexity bounds are optimal in the sense that they match the output size in the worst case. Additionally, in the same paper, it is shown that the complexity of BP can alternatively be bounded by $O\left(\sum_{v\in V}\boldsymbol{\pi}^{-1}_{t}(v)\cdot d_{\mathrm{out}}(v)/\varepsilon\right)$ and that RBS can answer the STPPR query with absolute error bounds in $\widetilde{O}\left(\sum_{v\in V}\boldsymbol{\pi}^{-1}_{t}(v)\cdot\sqrt{d_{\mathrm{out}}(v)}\big/\varepsilon\right)$ time, which also outperforms BP. These improved bounds of RBS hinge on a linear-time preprocessing phase that presorts the in-adjacency list of each node as per the out-degrees of the in-neighbors.

Liao et al. [^39] utilize their spanning forest sampling technique to propose BACKL, an STPPR algorithm as a counterpart to their FORAL [^39]. BACKL adopts the “BP+MC” framework, where the MC phase is implemented by sampling random spanning forests instead of random walks. Based on the invariant of BP, BACKL can approximate STPPR with relative error guarantees efficiently, especially for small $\alpha$.

## VII Single-Pair PPR Algorithms

For the SPPPR queries, Fujiwara et al. [^75] propose an exact algorithm based on node reordering, sparse matrix representations, and QR decomposition; Zhao et al. [^76] embed the nodes into geometric coordinate spaces to approximately answer the SPPPR query on undirected graphs. Recently, a series of methods combine local push and MC to approximately answer the SPPPR queries. Below, we review these methods in detail and also discuss a line of research tailored to the SNPR query.

### VII-A SPPPR Algorithms Based on Local Push and MC

Peter Lofgren et al. propose a series of bidirectional algorithms for the SPPPR query with relative error bounds [^77] [^78] [^79]. First, recall that MC offers a baseline complexity of $O\left(\frac{\log(1/\delta)}{\varepsilon^{2}\mu}\right)$ for this problem. Their seminal algorithm FAST-PPR [^77] first runs BP to construct a frontier set of the target node and then uses MC to detect the probability of reaching this frontier from the source node. They prove that FAST-PPR achieves an average complexity (over the target node $t$) of $\widetilde{O}\left(\frac{1}{\varepsilon^{2}}\sqrt{\frac{m/n}{\mu}}\right)$. In the same paper, a lower bound of $\Omega\left(\sqrt{1/\mu}\right)$ is also proved, implying that FAST-PPR’s dependence on $\mu$ is optimal.

Their subsequent work BiPPR [^78] achieves a better average complexity of $O\left(\frac{1}{\varepsilon}\sqrt{\frac{m/n\cdot\log(1/\delta)}{\mu}}\right)$. BiPPR also uses BP and MC but under a different analysis framework based on the invariant of BP (Eqn. (15)). In a nutshell, the invariant can be rewritten as

$$
\displaystyle\boldsymbol{\pi}^{-1}_{t}(s)=\boldsymbol{\hat{\pi}}^{(\mathrm{b})}_{t}(s)+\E_{v\sim\boldsymbol{\pi}_{s}}\left[\boldsymbol{r}^{(\mathrm{b})}_{t}(v)\right],
$$

where the PPR vector $\boldsymbol{\pi}_{s}$ is regarded as a probability distribution over $V$. Now, the expectation term can be straightforwardly estimated by running MC.

Note that FAST-PPR and BiPPR only provide average-case complexity guarantees instead of worst-case ones. In contrast, Lofgren et al. show that on undirected graphs, their algorithm Undirected-BiPPR [^79] can solve the query in $O\left(\frac{1}{\varepsilon}\sqrt{\frac{d(t)\log(1/\delta)}{\mu}}\right)$ time. Undirected-BiPPR combines FP and MC in a similar way as BiPPR with the aid of Thm. 3.

Wang et al. [^80] further propose HubPPR as an optimized version of BiPPR. HubPPR adopts the same bidirectional framework as BiPPR but uses two preprocessed backward oracle and forward oracle to speed up the BP and MC phases, resp. Particularly, HubPPR chooses some important “hub” nodes and preprocesses the results of backward pushes and random walks from them so that these results can be directly used to reduce the computational cost during query time.

### VII-B Algorithms for the Single-Node PageRank Query

The SPPPR algorithms FAST-PPR [^77], BiPPR [^78], and HubPPR [^80] can be naturally adapted for the SNPR query by modifying their MC phase from the source node. Besides, there exists a line of work tailored to the SNPR query. Aiming for sublinear complexities, these methods are also mainly based on local exploration and MC.

Graph-Access Models. This line of work gives upper and lower bounds on the computational complexity or query complexity of the SNPR query under different graph-access models, which specify how the algorithms can access the graph through queries to a graph oracle. The typical setting is the arc-centric graph-access model [^81] [^82], under which one can learn the in-degree/out-degree of a node or its $i$ -th incoming/outgoing edge using a single query. This model is also used by default in other works discussed in this paper. In another node-centric graph-access model [^83] [^84] [^81] [^82], one can learn the in-degree, out-degree, all in-neighbors, and all out-neighbors of a node using a single query. Crucially, to achieve sublinear complexities, the graph oracle must additionally support a $\textsc{jump}()$ query that returns a uniformly random node in $V$; otherwise, the problem cannot be answered with a sublinear query complexity in the worst case [^84] [^85].

The problem of estimating single-node PageRank with sublinear complexity is first introduced in [^83] by Chen et al. Under the node-centric graph-access model, they propose several methods with a general three-phase framework: expansion, estimation, and iteration. These algorithms expand backward from the target node to obtain a subgraph, then use various heuristic approaches to estimate the PageRank values of the nodes on the boundary of the subgraph, and finally iterate over the subgraph to obtain an approximate result.

Bar-Yossef and Mashiach [^84] reconsider this problem and give some lower bounds under the node-centric graph-access model. They prove that under this model and without the $\textsc{jump}()$ query, $\Omega\big(\sqrt{n}\big)$ queries are needed to estimate single-node PageRank in the worst case. This bound holds for randomized algorithms; for deterministic algorithms, a stronger bound of $\Omega(n)$ is verified. They show that the difficulty comes from two factors of the graph: (i) the existence of high-in-degree nodes and (ii) the slow convergence of the random walks. These lower bounds are later further discussed in [^85]. In [^84], the authors also propose an algorithm akin to those in [^83], which is efficient when the graph is free from these two difficulties. However, the existence of high-in-degree nodes in large real-world graphs makes their algorithm impractical.

Sublinear-Time SNPR Algorithms. Bressan, Peserico, and Pretto [^81] [^82] propose the first fully sublinear algorithms for the problem on general directed graphs. The authors propose two novel techniques called perfect weighted estimator and blacklisting high-degree nodes and combine them with RPI and MC to achieve a computational complexity of $\widetilde{O}\left(n^{2/3}\cdot\min\left(\Delta^{1/3},m^{1/6}\right)\right)$, where $\Delta$ is the maximum out-degree of the graph and $\text{poly}\left(\varepsilon^{-1}\right)$ factors are omitted. Their perfect weighted estimator has balanced coefficients and desirable concentration behavior. It is derived as a weighted sum of several subgraph estimators, which are similar to the estimators derived in FAST-PPR [^77]. The authors also establish query complexities of $\widetilde{O}\left(n^{1/2}\cdot\min\left(\Delta^{1/2},m^{1/4}\right)\right)$ under the arc-centric model and $\widetilde{O}\left(\min\left(n^{1/2}\Delta^{1/2},n^{2/3}\right)\right)$ under the node-centric one. They also give an $\Omega\left(\min\left(n^{1/2}\Delta^{1/2},n^{1/3}m^{1/3}\right)\right)$ lower bound under the arc-centric model and $\Omega\left(\min\left(n^{1/2}\Delta^{1/2},n^{2/3}\right)\right)$ under the node-centric one, implying that their algorithm is optimal under the node-centric model.

Recently, Wang and Wei [^86] propose SetPush as an algorithm for the SNPR query specifically on undirected graphs. By exploiting the symmetry property of PPR on undirected graphs (Thm. 3), SetPush designs a novel push operation that combines the reverse push and the MC sampling techniques, achieving a complexity of $\widetilde{O}\left(\min\big(d(t),\sqrt{m}\big)\right)$, breaking the lower bounds above for directed graphs. Notably, SetPush does not use the $\textsc{jump}()$ query.

## VIII Top-kk PPR Algorithms

There exists various algorithms devoted to the top- $k$ PPR query. Unlike the queries above, the quality of the results for the top- $k$ PPR queries cannot be easily evaluated using previous error measures, and existing algorithms provide error guarantees from diverse aspects. We will discuss the error guarantees explicitly for almost every algorithm. As we did for SSPPR algorithms, we categorize top- $k$ PPR algorithms into two types based on their methods.

### VIII-A Top-kk PPR Algorithms Based on Algebraic Techniques

Fujiwara et al. propose a series of algorithms for exact top- $k$ PPR computation [^87] [^75] [^88] [^89]. Roughly speaking, these algorithms iteratively compute PPR estimates with upper and lower bounds and use a pruning strategy to eliminate non-top- $k$ nodes, thereby constructing smaller subgraphs to expedite the estimation process. Specifically, K-dash [^87] constructs a breadth-first search tree and exploits sparse matrix computation, node reordering, and LU decomposition; [^75] applies the SPPPR algorithm in [^75]; Castanet [^88] is designed to return the exact ranking of the top- $k$ nodes among a target node set; and F-Rank [^89] is tailored to exact top- $k$ PageRank.

Wu et al. propose another exact top- $k$ PPR algorithm called FLoS\_RWR [^90]. FLoS\_RWR is derived as an extension of FLoS\_PHP for the computation of PHP (penalized hitting probability). Using the property of PHP, FLoS\_PHP searches the graph locally and uses SPI to compute the lower and upper bounds of the PHP values for the nodes on the boundary of the explored subgraph. By expanding the subgraph and tightening the lower and upper bounds are gradually tightened, the top- $k$ nodes can be determined. Based on an equation between PHP and RWR, FLoS\_PHP can be modified to FLoS\_RWR without modifying the overall process. In the paper, it is stated that the proximity values between two adjacent iterations are highly close. Thus, the number of iterations needed to update the proximity is fairly small.

Recall that Chopper [^56] computes the PPR vector by running modified SPI based on Chebyshev polynomials, as discussed in Sec. V-A. For the top- $k$ PPR query, Chopper takes advantage of the error bound of this Chebyshev acceleration to prune the nodes that cannot be the top- $k$ results, further reducing the number of iterations.

### VIII-B Top-kk PPR Algorithms Based on Local Push and MC

Sarlós et al. [^50] apply deterministic rounding to the dynamic programming algorithm [^32] for a version of approximate top- $k$ PPR query [^50] under the framework of two-phase algorithms (mentioned in Sec. V-B). They apply rounding to the intermediate vectors of the algorithm and build a database with optimal space for this type of query.

Gupta et al. [^91] propose the Basic Push Algorithm for the top- $k$ PPR query. The number of nodes returned by this algorithm varies in a specified range. The algorithm implements FP with a priority queue and precomputed hub vectors, obtaining lower and upper bounds for the PPR values. Once these bounds can guarantee valid results, the algorithm terminates and returns the results.

Chakrabarti et al. [^92] concentrate on building effective indices for top- $k$ PPR search so that a query can be processed in empirically constant time. Their two solutions, HubRankP and HubRankD, are based on BCA [^57] and the decomposition theorem (Thm. 2), resp., improving their early versions in [^91] [^93] [^58]. While both algortihms achieve small index space, efficient index building and query processing, and high accuracy, HubRankP is considered superior to HubRankD.

Avrachenkov et al. [^94] utilize MC to answer the approximate top- $k$ PPR query efficiently. They use the common MC method or “MC complete path” [^30] to estimate the PPR vector and obtain the top- $k$ nodes. They give approximations based on the central limit theorem to show that the algorithm achieves high efficiency if allowing a few erroneous results.

Adaptations to Top- $\boldsymbol{k}$ Query. Several SSPPR and SPPPR algorithms mentioned before can be adapted for top- $k$ PPR queries, including BiPPR [^78], HubPPR [^80], and FORA [^61]. BiPPR [^78] considers the top- $k$ PPR query among a node set in the context of personalized search, and it applies preprocessing and hierarchical sampling to the original BiPPR, achieving high performance along with theoretical guarantees. HubPPR [^80] studies a new version of approximate top- $k$ PPR query [^80] among a node set. Based on the original HubPPR, it increases the accuracy of BP and MC iteratively, and in each iteration, the lower and upper PPR bounds of all target nodes are updated, based on which some target nodes are pruned. FORA [^61] also solves the approximate top- $k$ PPR query defined in [^80] but on the entire node set $V$. It adopts a trial-and-error approach to iteratively run the original FORA with different relative error thresholds, and it also computes lower and upper PPR bounds to determine when to terminate.

Wei et al. [^38] meld FP, BP, and MC to create TopPPR. Given parameters $s$, $k$, and $\rho$, TopPPR returns approximate top- $k$ nodes with the highest PPR values w.r.t. $s$ and ensures a precision of at least $\rho$. Conceptually, TopPPR first performs FP from $s$ and simulates random walks from the nodes with nonzero residues, as FORA does; it then runs BP adaptively from some candidate nodes to determine the top- $k$ nodes. TopPPR estimates the PPR values through a combination of the invariants of both FP and BP, and it maintains and expands a set containing the nodes that are certain to be among the top- $k$. TopPPR runs in $O\left((m+n\log n)\big/\sqrt{\mathrm{gap}_{\rho}}\right)$ time and specifically in $O\left(k^{1/4}n^{3/4}\log n\big/\sqrt{\mathrm{gap}_{\rho}}\right)$ time on power-law graphs, where $\mathrm{gap}_{\rho}$ is the difference between the $\lceil\rho k\rceil$ -th and the $(k+1)$ -th largest PPR value w.r.t. $s$. Notably, the complexity is sublinear in $n$ on power-law graphs if $k=o(n)$.

## IX Algorithms for Other PPR Queries

This section discusses PPR algorithms tailored to some specific queries. These queries are relatively less studied but useful for some particular applications.

Subgraph PageRank Query. Davis and Dhillon [^95] study the problem of locally estimating the PageRank values in a given subgraph. They propose to expand the given subgraph by greedily selecting a series of nodes on the frontier and to utilize stochastic complementation to compute the PageRank values on the expanded subgraph. Their approach runs in linear time w.r.t. the number of nodes of the given subgraph, assuming that the number of performed iterations is constant.

Subgraph PageRank Ranking. Bressan and Pretto [^96] theoretically study the problem of determining the PageRank ranking of $k$ given nodes. They show that, in the worst case, a correct algorithm must explore $\Omega(n)$ nodes if it is deterministic and $\Omega\left(\sqrt{kn}\right)$ nodes if randomized. The later work [^85] by Bressan et al. further shows that Las Vegas algorithms must explore $n-o(n)$ nodes and that MC algorithms also need to explore $\Omega(n)$ nodes if the $\textsc{jump}()$ query (see Sec. VII-B) is not allowed. However, if the $\textsc{jump}()$ query is allowed, the lower bound of MC algorithms degrades to $\Omega\left(n^{2/3}\right)$. For this case, they propose SampleRank that achieves sublinear complexity when the PageRank values in question are not too small.

Significant PageRank Query. Borgs, Brautbar, Chayes, and Teng [^97] [^98] focus on the Significant PageRank problem: given $\Delta$ and $c>1$, with high probability compute a set that contains all nodes with PageRank value at least $\Delta$ and no node smaller than $\Delta/c$. Based on an abstraction of the problem to a basic matrix problem, their approach utilizes two novel components, multiscale matrix sampling and robust approximation of PPR, to solve this problem in $\widetilde{O}(1/\Delta)$ time. Their robust approximation of PPR aims to estimate PPR values within a relative error plus an absolute error, and their process resembles the standard MC method.

Reverse Top- $\boldsymbol{k}$ PPR Query. Yu et al. [^99] study the reverse top- $k$ PPR query, which asks for all nodes that have $t$ in their top- $k$ PPR node sets, given the target node $t$ and an integer $k$. Their proposed algorithm comprises an indexing technique and an online search process. In the preprocessing phase, BCA [^57] is invoked to produce lower bounds of the top PPR values, which are stored compactly as the index. In the online querying phase, the index is used in cooperation with RPI to prune or confirm the candidate nodes, where the computational overhead can be reduced, and the index can be refined for better future performance.

Heavy Hitters Query. Wang and Tao [^2] delve into identifying approximate PPR heavy hitters and propose BLOG for this purpose. Concretely, given two node sets $S$ and $T$, BLOG approximately finds all pairs $(s,t)$ such that $s\in S$, $t\in T$, and $\boldsymbol{\pi}_{s}(t)$ is above a specified fraction of $\boldsymbol{\pi}(t)$. BLOG primarily combines MC and BP to determine whether each pair of $(s,t)$ satisfies the property, and it incorporates a novel technique to circumvent high-in-degree nodes when performing BP. Moreover, BLOG shares some computational results across different node pairs to reduce redundancy. Also, BLOG leverages a new logarithmic bucketing technique that partitions the target nodes into groups and applies different parameter settings for each group. Overall, BLOG achieves superior theoretical complexity and empirical performance.

Batch One-Hop PPR Query. Luo et al. [^62] [^63] propose Baton for a new type of PPR query known as the batch one-hop PPR query: given a node set $S\subseteq V$, Baton provides an estimate of $\boldsymbol{\pi}_{s}(v)$ for each pair of $s\in S$ and $v\in\mathcal{N}_{\mathrm{out}}(s)$ with relative error guarantees. Baton follows the “FP+MC” framework of FORA [^61] but adopts novel techniques and analysis tailored to this query. In particular, Baton adopts distinct stopping criteria for the FP process and reuses the generated random walks. Its analysis exploits a lower bound for one-hop PPR and the concept of “generic Baton.” It is shown that Baton runs in $O\big(\sum_{s\in S}d_{\mathrm{out}}(s)\log(1/\delta)/\varepsilon^{2}\big)$ expected time, which is nearly optimal. The authors also consider applying a vertex-cover-based optimization on undirected graphs and multicore parallelization to Baton.

## X PPR Computation in Special Settings

This section reviews several algorithms for computing PPR in some special settings. Sarkar and Moore [^100] consider finding top nodes with the largest degree-normalized PPR values on large disk-resident undirected graphs, and their methods are based on a graph clustering algorithm and the dynamic programming algorithm in [^50]. Yang [^24] computes PPR and its extensions on bipartite graphs using modifications of PI and FP. Below, we particularly introduce the challenges and some algorithms for the settings of dynamic graphs, parallel/distributed settings, weighted graphs, and graph streams.

### X-A For Dynamic Graphs

Settings and Challenges. Although most relevant works focus on PPR computation on static graphs, many real-world graphs for practical applications, such as social networks, are constantly evolving. This necessitates the development of PPR algorithms tailored to dynamic graphs, where the graph undergoes continuous updates, including edge insertions and deletions. Efficiently maintaining PPR on dynamic graphs poses some unique challenges. Firstly, running linear-time PPR algorithms upon each update is prohibitive due to the frequent updates. Secondly, several efficient algorithms introduced before rely on precomputed indices to speed up the query process, but they cannot be directly applied to dynamic graphs. In fact, updating these indices over time can be rather costly. Finally, ensuring accuracy guarantees for evolving graphs introduces technical difficulties and probably requires more sophisticated approaches. To address these challenges, several PPR algorithms for dynamic graphs have been proposed.

Bahmani et al. [^101] apply MC to maintain PageRank values or top- $k$ PPR nodes for a dynamic graph, where $m$ edges arrive in an equally random order to a graph with $n$ nodes, and the edges can also be deleted randomly over time. This setting is called the random arrival model and is adopted in some later works [^102] [^68] [^70]. For PageRank values, their method is based on the “MC complete path” method [^30]. Roughly speaking, they perform several random walks from each node, store the entire paths, and use the number of occurrences of the nodes to estimate their PageRank values. Then, for an edge insertion, only a small portion of the stored paths need to be updated in expectation. They demonstrate that the expected complexity to maintain all PageRank values at all times with constant error is $O(n\ln m)$. This method also supports a random edge removal with $O(n/m)$ expected work. Also, a similar approach works for top- $k$ PPR maintenance under more complicated models and assumptions.

Unlike other studies for dynamic graphs, Bahmani et al. [^103] consider the case when the algorithm is oblivious to the changes in the graph. They propose effective crawling strategies for periodically probing the nodes in the graph to detect the changes and update the PageRank vector.

Yu and Lin [^104] propose IRWR for dynamically computing exact SSPPR. They first derive a formula for the new PPR vector $\boldsymbol{\pi}_{s}^{\prime}$ after inserting or deleting an edge $(u,v)$ to/from the graph. This formula expresses $\boldsymbol{\pi}_{s}^{\prime}$ as a linear combination of $\boldsymbol{\pi}_{u}$ and $\boldsymbol{\pi}_{v}$ before the operation. Based on this, given the original graph, its PPR matrix (computed using other algorithms), the source node $s$, and a batch of edge insertion/deletion operations, IRWR can incrementally update the PPR vectors of $s$ and other relevant nodes, and thus obtain the resultant PPR vector of $s$.

A later work of Yu and McCann [^105] considers maintaining the PPR matrix over dynamic graphs exactly, and their solution is called Inc-R. By characterizing the changes to $\mathbf{\Pi}$ as the outer product of two vectors and skipping unnecessary computation, Inc-R can handle a single edge insertion/deletion in time proportional to the number of affected entries in $\mathbf{\Pi}$.

Ohsaka et al. [^106] propose to adopt FP for maintaining dynamic SSPPR with absolute error guarantees on an evolving graph. Their algorithm TrackingPPR maintains the reserve and residue vectors dynamically by possibly performing pushes when the graph is updated. For a single edge insertion or deletion, their algorithm only needs to perform amortized $O(1/\varepsilon)$ pushes, and for $m$ randomly inserted edges, the total number of pushes is $O(\log m/\varepsilon)$ in expectation.

Zhang et al. [^102] adopt FP and BP to dynamic graphs, yielding efficient algorithms for dynamically maintaining approximate PPR and inverse PPR vectors, resp. The algorithms are dubbed LazyForward and LazyReverse. Conceptually, for every edge insertion/deletion, their algorithms first maintain the invariant between the reserve and residue vectors and then perform push operations to restore accuracy if needed. For PPR, LazyForward maintains degree-normalized error on an undirected graph, with the expected complexity of $O(1/\varepsilon)$ plus $O(1)$ per update, for a uniformly random source node $s$. For inverse PPR, LazyReverse maintains absolute error on a directed graph in the random arrival model [^101] or on an arbitrary undirected graph, with an expected complexity of $O(m/n/\varepsilon)$ plus $O(1)$ per update, for a uniformly random target node $t$.

Yoon et al. [^107] devise accurate dynamic SSPPR algorithms OSP and OSP-T based on CPI. For an edge insertion/deletion, OSP uses CPI to compute the difference between the old PPR vector and the new one by expressing it as a power series of the difference of the transition matrix. OSP-T further introduces a trade-off between accuracy and efficiency by limiting the number of iterations in CPI.

Mo and Luo [^68] [^108] present Agenda, an efficient algorithm for maintaining SSPPR with relative error guarantees on dynamic graphs. Agenda is designed to be “robust,” in the sense that it sustains efficiency regardless of whether the ratio of updates to queries is high or low. Following the framework of FORA+ [^61], Agenda stores precomputed random walks as indices, which are then combined with FP’s results at query time. Crucially, Agenda runs BP to compute a quantitative measure of the index inaccuracy and adopts a lazy-update mechanism to reduce the overhead of updating the indices over time. Theoretically, under the random arrival model [^101], the query complexity of Agenda matches that of FORA+, and the expected fraction of updated indices is also bounded.

A subsequent work, FIRM by Hou et al. [^70], also follows the framework of FORA+ [^61] but imposes an eager-update strategy to update the indices of random walks. Notably, FIRM maintains auxiliary data structures to support updating the indices in expected $O(1)$ time under the random arrival model [^101]. Furthermore, FIRM can use a novel sampling scheme to reduce the space consumption of these data structures while preserving the expected $O(1)$ update complexity.

### X-B In Parallel/Distributed Settings

Settings and Challenges. The enormous sizes of current graphs have made it imperative to develop efficient algorithms for computing PPR in parallel or distributed settings, where the algorithm can allocate the computation workload to multiple processors or machines for high performance and scalability. The detailed settings vary across different works, and we will introduce them briefly below. The core challenge under these settings lies in effectively managing the computational tasks across multiple processors or machines, each handling a different part of the graph. Meanwhile, the algorithms need to minimize the number of computational rounds and the communication cost while ensuring desirable convergence rates or error guarantees. Particularly, as it is nontrivial to parallelize the basic techniques of PI and FP, more sophisticated techniques are required to adapt existing methods for PPR computation to parallel or distributed settings.

Some early works [^109] [^110] [^111] [^112] aim at computing PageRank on a decentralized peer-to-peer network, where each peer computes PageRank values individually using iterative methods and communicates with others to refine and combine the results. Another early work [^113] uses divide-and-conquer to compute local PageRank vectors and then update them through communication with a coordinator.

Bahmani et al. [^114] propose DoublingPPR for approximately computing the PPR matrix under the MapReduce model. Their method is based on MC, and its basic idea is to generate random walks of a given length for all nodes efficiently using MapReduce. To achieve this, DoublingPPR merges short segments of random walks to obtain longer random walks in a doubling manner.

Mitliagkas et al. propose FrogWild [^115] for approximating top- $k$ PageRank using the GraphLab framework. FrogWild generates parallel random walks to estimate high PageRank values, and it adopts partial synchronization to reduce the network traffic. Although the generated random walks are no longer independent, the authors give analysis for their correlations and bound the approximation errors.

Liu et al. [^116] propose PowerWalk to approximate PPR vectors under a distributed framework. It simulates full-path random walks as offline indexing, and for online query, it computes linear combinations of these indices in a vertex-centric decomposition manner to obtain approximate PPR vectors. PowerWalk strikes an adaptive trade-off between offline preprocessing and online query based on the memory budget.

Guo et al. [^117] introduce GPA and HGPA for computing exact SSPPR on a coordinator-based share-nothing distributed computing platform. The salient idea of GPA is to partition the graph into disjoint subgraphs and distribute the computation accordingly, and HGPA further splits the subgraphs recursively to form a tree-structured hierarchy. Based on precomputed data, GPA and HGPA can allocate the computation evenly to separate machines, guaranteeing that only one time of communication between each machine and the coordinator is needed to obtain the aggregated PPR vector at query time.

Guo et al. [^118] migrate the algorithm for dynamic SSPPR in [^102] to parallel settings of GPUs or multicore CPUs. Their algorithm is a batch-processing method with novel techniques for updating the reserve and residue vectors in parallel, leading to reduced synchronization cost and a higher level of parallelism. While the algorithm’s complexity matches the original sequential one [^102], its specially designed optimizations yield extensive practical speedups.

Shi et al. [^119] propose kPAR for answering the approximate top- $k$ PPR query defined in [^80] using GPUs on a single commodity machine. To exploit the power of GPU, kPAR utilizes novel techniques called adaptive forward push and inverted random walks and performs careful system engineering specific to GPUs. kPAR runs in $O\left(n+\frac{1}{\varepsilon}\sqrt{\frac{m\log(1/\delta)}{\mu}}\right)$ time and can answer the query in real-time on billion-scale graphs.

Lin [^120] exploits the parallel pipeline framework to compute MC approximation of the PPR matrix with relative error bounds on large edge-weighted graphs. Their DistPPR utilizes three carefully designed techniques to efficiently generate many random walks from each node. In particular, these techniques help to handle nodes with large degrees, precompute walk segments for nodes with small degrees, and optimize the workload for each pipeline to reduce the overhead.

Wang et al. [^64] endeavor to parallelize FORA [^61] on shared-memory multicore systems under the work-depth model. Their work-efficient approach PAFO achieves linear speedup in the number of processors. PAFO exploits various novel techniques, including efficient maintenance of the active nodes in FP, cache-aware scheduling, and an integer-counting-based method for generating random walks.

Hou et al. [^67] propose Delta-Push for the SSPPR and top- $k$ PPR queries under the Massively Parallel Computing (MPC) model. Delta-Push follows the framework of FORA+ [^61] and combines a redesigned parallel push algorithm and MC in a nontrivial way. They show that Delta-Push reduces the number of rounds to $O\left(\log\frac{n^{2}\log n}{\varepsilon^{2}m}\right)$ with a load of $O(m/p)$, where $\varepsilon$ is the relative error bound and $p$ is the number of executors.

Other Theoretical Results. Some additional theoretical works focus on the round complexity and the communication bandwidth of computing PageRank under distributed settings. Most of these works use a stitching technique to generate long random walks by combining short random walks in a doubling manner. In [^121], the authors consider the round complexity of computing PageRank under a kind of congested clique model, where only limited messages can be sent over each edge in each round. They devise a special method to simulate random walks in parallel, which avoids communication congestion. Their algorithms take $O(\log n)$ rounds on directed graphs and $O\left(\sqrt{\log n}\right)$ rounds on undirected graphs, with edge bandwidth bounded by $O(\log n)$ and $O\left(\log^{3}n\right)$ bits, resp. On undirected graphs, the round complexity of $O\left(\sqrt{\log n}\right)$ is later improved to $O(\log\log n)$ by Luo [^122], and the bandwidth is also improved in a subsequent work by Luo [^66]. The Multiple-Round Radar-Push algorithm proposed in [^122] uses a novel approach to generate a number of low-variance random walks from each node, and [^66] further combines this algorithm with FP to reduce the communication cost.

Some other works investigate the MPC model instead of the congested clique model. [^123] proves the following bounds on the round complexity for computing PageRank under the MPC model: $O(\log\log n)$ for undirected graphs (with $\widetilde{O}(m)$ total space) and $\widetilde{O}\left(\log^{2}\log n\right)$ for directed graphs (with $\widetilde{O}\left(m+n^{1+o(1)}\right)$ total space). Kapralov et al. [^124] further study the complexity of computing PPR vectors under the MPC model and give some more involved bounds.

### X-C For Weighted Graphs

Settings and Challenges. This subsection considers computing PPR on graphs with weighted edges. On such weighted graphs, when an $\alpha$ -discounted random walk proceeds, each out-neighbor is selected with a probability proportional to the weight of the connecting edge instead of with an equal probability. Some applications of PPR on weighted graphs are introduced in [^125] [^126]. While some previously introduced PPR algorithms are adaptable for weighted graphs, specially designed algorithms can outperform these adaptations by a large margin. However, there is a limited body of research for this specific purpose. We introduce two works for computing SSPPR on weighted graphs below: the first employs algebraic techniques, and the second utilizes the local-push method.

The first empirically efficient algorithm tailored to weighted graphs is proposed by Xie et al. [^125]. Their method relies on model reduction, a general approach for approximating solutions to large linear systems. Note that on weighted graphs, the PPR vector still satisfies a linear equation similar to Eqn. (1), with $\mathbf{P}$ replaced by the corresponding weighted transition matrix. Based on model reduction, their algorithm can reduce this linear system to a low dimension, solve the reduced system, and reconstruct the approximate PPR vector. They also apply optimizations for parameterizations and cost-accuracy trade-offs to further improve their algorithm’s empirical performance. However, their algorithm is still costly as it needs to process the large matrices of the graph.

Wang et al. propose EdgePush [^126] to optimize FP on weighted graphs. EdgePush decomposes the atomic node-level push operation in FP into separate edge-level pushes, assigning different pushing thresholds for each edge. With a delicately designed data structure, each push operation can be done in amortized $O(1)$ time. This edge-based push strategy endows EdgePush with superior complexities over FP, especially on graphs where the edge weight distribution is imbalanced. In such cases, the complexity bound is improved by a factor up to $\Theta(n)$. The authors give detailed complexity bounds for both $\ell_{1}$ -error and degree-normalized absolute error guarantees.

### X-D For Graph Streams

Settings and Challenges. This subsection reviews algorithms for computing PageRank when the edges of the underlying graph are presented as a stream. The objective under this streaming model is to compute PageRank using minimal space and few passes over the stream. Under these constraints, even sampling a random walk becomes a complex task, and there are only a few relevant theoretical studies.

Sarma, Gollapudi, and Panigrahy [^127] [^128] consider estimating the PageRank vector on graph streams. Based on a new streaming algorithm for estimating the mixing time of random walks, their algorithm only requires $o(n)$ space and $\widetilde{O}\big(\sqrt{M}\big)$ passes to compute PageRank values above $M/n$, where $M$ denotes the mixing time. Kallaugher, Kapralov, and Price [^129] investigate the setting where the edges are given in a uniformly random order. They design a single-pass algorithm for generating nearly independent random walks and apply it to estimate the sum of PageRank values for a node set with absolute error guarantees using sublinear space.

## XI Research Challenges and Directions

Tbls. III and IV summarize the representative PPR algorithms under the typical setting and special settings, resp. Despite the extensive body of research, numerous unresolved issues persist in the realm of efficient PPR computation from both theoretical and practical perspectives.

- From the theoretical point of view, the major concern is the sublinear complexities of various PPR approximation problems. For example, for the basic techniques, can one optimize FP or BP by modifying their push strategies or termination conditions? As the error bound of FP and the worst-case complexity of BP on general directed graphs are still unclear, FP and BP’s properties may not have been fully understood. Also, the gaps between the upper and lower bounds for the SPPPR query [^77] [^79] and the SNPR query [^81] [^82], along with the problem of solving $\ell_{1}$ -regularized PageRank [^71], remain as open problems. It is also desirable to establish lower bounds for other PPR approximation problems and narrow the gaps between the upper and lower bounds. Finally, it is interesting to further investigate the complexities for undirected graphs, dynamic graphs, weighted graphs, etc.
- From the practical point of view, enhancing the efficiency and scalability of PPR computation is still challenging. The vast scale of real-world graphs greatly limits the applicability of classic iterative PPR algorithms, necessitating approximate algorithms that harness the power of parallel or distributed computing. An important but unsolved problem is to approximately compute or factorize the whole PPR matrix of large graphs efficiently [^15] [^19] [^17]. Moreover, there is a notable absence of a standardized benchmark or an extensive experimental survey for evaluating the practical performance of existing algorithms. As current algorithms cover various settings and traditional worst-case complexity analysis cannot adequately reflect their empirical performance, such a benchmark or experimental survey is meaningful and essential. Finally, some interesting future directions include using machine learning to accelerate PPR computation and using PPR to develop differentially private algorithms for downstream graph learning tasks [^23].

TABLE III: Summary of Representative PPR Algorithms under the Typical Setting

<table><tbody><tr><td>Query</td><td>Algorithm</td><td>Main Techniques/Tools</td><td>Error Bound</td><td>Notes</td></tr><tr><td rowspan="16">SSPPR</td><td><sup><a href="#fn:50">50</a></sup></td><td>dynamic programming, Count-Min Sketch</td><td>absolute</td><td>two-phase algorithm withoptimal database space</td></tr><tr><td>FastPPV <sup><a href="#fn:51">51</a></sup></td><td>prioritizing paths, scheduled estimation</td><td><math><semantics><msub><mi>ℓ</mi> <mn>1</mn></msub> <annotation>\ell_{1}</annotation></semantics></math></td><td></td></tr><tr><td><sup><a href="#fn:52">52</a></sup></td><td>core-tree decomposition, preconditioning</td><td><math><semantics><msub><mi>ℓ</mi> <mn>1</mn></msub> <annotation>\ell_{1}</annotation></semantics></math></td><td></td></tr><tr><td>BEAR <sup><a href="#fn:54">54</a></sup></td><td>block elimination, reordering, LU decomposition</td><td>exact</td><td>has approximate version</td></tr><tr><td>BePI <sup><a href="#fn:55">55</a></sup></td><td>block elimination, sparsifying, preconditioning</td><td><math><semantics><msub><mi>ℓ</mi> <mn>2</mn></msub> <annotation>\ell_{2}</annotation></semantics></math> -error</td><td>improves over BEAR</td></tr><tr><td>Chopper <sup><a href="#fn:56">56</a></sup></td><td>SPI, Chebyshev polynomials</td><td>not given</td><td></td></tr><tr><td>TPA <sup><a href="#fn:41">41</a></sup></td><td>dividing the summation of PPR into three parts</td><td><math><semantics><msub><mi>ℓ</mi> <mn>1</mn></msub> <annotation>\ell_{1}</annotation></semantics></math></td><td></td></tr><tr><td><sup><a href="#fn:42">42</a></sup></td><td>momentum-based acceleration methods</td><td><math><semantics><msub><mi>ℓ</mi> <mn>1</mn></msub> <annotation>\ell_{1}</annotation></semantics></math></td><td>on undirected graphs,<math><semantics><mrow><mn>1</mn> <mo>/</mo> <msqrt><mi>α</mi></msqrt></mrow> <annotation>1/\sqrt{\alpha}</annotation></semantics></math> dependence on <math><semantics><mi>α</mi> <annotation>\alpha</annotation></semantics></math></td></tr><tr><td><sup><a href="#fn:59">59</a></sup></td><td>FP with dynamic threshold</td><td>same as FP</td><td>exponential convergence</td></tr><tr><td>PowerPush <sup><a href="#fn:43">43</a></sup></td><td>unifying FIFO-FP and PI</td><td><math><semantics><msub><mi>ℓ</mi> <mn>1</mn></msub> <annotation>\ell_{1}</annotation></semantics></math></td><td>exponential convergence</td></tr><tr><td>FORA, FORA+ <sup><a href="#fn:61">61</a></sup></td><td>FP+MC, indexing</td><td>relative</td><td></td></tr><tr><td>ResAcc <sup><a href="#fn:65">65</a></sup></td><td>FP+MC, residue accumulation</td><td>relative</td><td>index-free</td></tr><tr><td>SpeedPPR <sup><a href="#fn:43">43</a></sup></td><td>PowerPush+MC, indexing</td><td>relative</td><td></td></tr><tr><td>FORAL <sup><a href="#fn:39">39</a></sup></td><td>FP+spanning forest sampling</td><td>see <sup><a href="#fn:39">39</a></sup></td><td>robust for small <math><semantics><mi>α</mi> <annotation>\alpha</annotation></semantics></math></td></tr><tr><td><sup><a href="#fn:69">69</a></sup></td><td>PI, progressive and spanning forest sampling</td><td>relative</td><td></td></tr><tr><td><sup><a href="#fn:23">23</a></sup></td><td>FP with residue restriction</td><td>absolute</td><td>bounded sensitivity</td></tr><tr><td rowspan="3">STPPR</td><td><sup><a href="#fn:73">73</a></sup></td><td>BP with priority queue</td><td>absolute</td><td>exponential convergence</td></tr><tr><td>RBS <sup><a href="#fn:74">74</a></sup></td><td>randomized BP, presorting</td><td>absolute/relative</td><td>optimal for relative error</td></tr><tr><td>BACKL <sup><a href="#fn:39">39</a></sup></td><td>BP+spanning forest sampling</td><td>relative</td><td>robust for small <math><semantics><mi>α</mi> <annotation>\alpha</annotation></semantics></math></td></tr><tr><td rowspan="6">SPPPR</td><td>FAST-PPR <sup><a href="#fn:77">77</a></sup></td><td>BP+MC, frontier set</td><td>relative</td><td>optimal dependence on <math><semantics><mi>μ</mi> <annotation>\mu</annotation></semantics></math></td></tr><tr><td>BiPPR <sup><a href="#fn:78">78</a></sup></td><td>BP+MC, bidirectional estimator</td><td>relative</td><td>improves over FAST-PPR</td></tr><tr><td>Undirected-BiPPR <sup><a href="#fn:79">79</a></sup></td><td>FP+MC, symmetry of PPR</td><td>relative</td><td>on undirected graphs</td></tr><tr><td>HubPPR <sup><a href="#fn:80">80</a></sup></td><td>BP+MC, indexing</td><td>relative</td><td>improves over BiPPR</td></tr><tr><td><sup><a href="#fn:81">81</a></sup> <sup><a href="#fn:82">82</a></sup></td><td>local exploration, MC, perfect estimator, blacklist</td><td>relative</td><td>SNPR, sublinear complexity</td></tr><tr><td>SetPush <sup><a href="#fn:86">86</a></sup></td><td>BP, MC, symmetry of PPR</td><td>relative</td><td>SNPR on undirected graphs</td></tr><tr><td rowspan="9">top- <math><semantics><mi>k</mi> <annotation>k</annotation></semantics></math> PPR</td><td><sup><a href="#fn:87">87</a></sup> <sup><a href="#fn:75">75</a></sup> <sup><a href="#fn:88">88</a></sup> <sup><a href="#fn:89">89</a></sup></td><td>computing lower/upper bounds, pruning</td><td>exact</td><td><sup><a href="#fn:88">88</a></sup> returns exact ranking,<sup><a href="#fn:89">89</a></sup> is for top- <math><semantics><mi>k</mi> <annotation>k</annotation></semantics></math> PageRank</td></tr><tr><td>FLoS_RWR <sup><a href="#fn:90">90</a></sup></td><td>local search, SPI on subgraphs</td><td>exact</td><td></td></tr><tr><td>Chopper <sup><a href="#fn:56">56</a></sup></td><td>Chebyshev polynomials, SPI, pruning</td><td>exact</td><td></td></tr><tr><td><sup><a href="#fn:50">50</a></sup></td><td>dynamic programming, rounding</td><td>see <sup><a href="#fn:50">50</a></sup></td><td>two-phase algorithm withoptimal database space</td></tr><tr><td>HubRank <sup><a href="#fn:92">92</a></sup></td><td>BCA, indexing</td><td>not given</td><td></td></tr><tr><td><sup><a href="#fn:78">78</a></sup> <sup><a href="#fn:80">80</a></sup> <sup><a href="#fn:61">61</a></sup></td><td>FP, BP, MC, hierarchical sampling</td><td>see <sup><a href="#fn:78">78</a></sup> <sup><a href="#fn:80">80</a></sup></td><td>adaptations for top- <math><semantics><mi>k</mi> <annotation>k</annotation></semantics></math> query</td></tr><tr><td>TopPPR <sup><a href="#fn:38">38</a></sup></td><td>FP+BP+MC, adaptive BP</td><td><math><semantics><mi>ρ</mi> <annotation>\rho</annotation></semantics></math> -precision</td><td>sublinear complexity in <math><semantics><mi>n</mi> <annotation>n</annotation></semantics></math> on power-law graphs</td></tr></tbody></table>

TABLE IV: Summary of Representative PPR Algorithms under Special Settings

<table><tbody><tr><th>Setting</th><th>Algorithm</th><td>Main Techniques/Tools</td><td>Notes</td></tr><tr><th rowspan="9">Dynamic Graphs</th><th><sup><a href="#fn:101">101</a></sup></th><td>MC, update random walks</td><td>for PageRank or top- <math><semantics><mi>k</mi> <annotation>k</annotation></semantics></math> PPR, random arrival model</td></tr><tr><th><sup><a href="#fn:103">103</a></sup></th><td>effective crawling strategies</td><td>for PageRank, oblivious to the changes in the graph</td></tr><tr><th>IRWR <sup><a href="#fn:104">104</a></sup></th><td>algebraically update PPR vector</td><td>for exact SSPPR</td></tr><tr><th>Inc-R <sup><a href="#fn:105">105</a></sup></th><td>algebraically update PPR matrix</td><td>for exact PPR matrix</td></tr><tr><th>TrackingPPR <sup><a href="#fn:106">106</a></sup></th><td>FP, pushes as per updates</td><td>for SSPPR with absolute error bounds</td></tr><tr><th><sup><a href="#fn:102">102</a></sup></th><td>FP, BP, maintaining invariants</td><td>for SSPPR or STPPR, similar error bounds to FP/BP</td></tr><tr><th>OSP, OSP-T <sup><a href="#fn:107">107</a></sup></th><td>CPI, computing the difference</td><td>for SSPPR, OSP-T balances accuracy and efficiency</td></tr><tr><th>Agenda <sup><a href="#fn:68">68</a></sup> <sup><a href="#fn:108">108</a></sup></th><td>FP+MC, indexing, lazy-update</td><td>for SSPPR with relative error bounds, workload robustness</td></tr><tr><th>FIRM <sup><a href="#fn:70">70</a></sup></th><td>FP+MC, indexing, eager-update</td><td>for SSPPR with relative error bounds</td></tr><tr><th rowspan="9">Parallel/Distributed</th><th>DoublingPPR <sup><a href="#fn:114">114</a></sup></th><td>MC, doubling</td><td>for PPR matrix, MapReduce model</td></tr><tr><th>FrogWild <sup><a href="#fn:115">115</a></sup></th><td>MC, partial synchronization</td><td>for approximate top- <math><semantics><mi>k</mi> <annotation>k</annotation></semantics></math> PageRank, GraphLab framework</td></tr><tr><th>PowerWalk <sup><a href="#fn:116">116</a></sup></th><td>MC, vertex-centric decomposition</td><td>for approximate SSPPR, balances indexing and query</td></tr><tr><th>GPA, HGPA <sup><a href="#fn:117">117</a></sup></th><td>partitioning, tree structure</td><td>for exact SSPPR, coordinator-based platform</td></tr><tr><th><sup><a href="#fn:118">118</a></sup></th><td><sup><a href="#fn:102">102</a></sup>, batch-processing</td><td>for SSPPR, on GPUs or multicore CPUs</td></tr><tr><th>kPAR <sup><a href="#fn:119">119</a></sup></th><td>adaptive FP, inverted random walks</td><td>for approximate top- <math><semantics><mi>k</mi> <annotation>k</annotation></semantics></math> PPR, on GPUs</td></tr><tr><th>DistPPR <sup><a href="#fn:120">120</a></sup></th><td>MC, workload optimization</td><td>for PPR matrix on weighted graphs, pipeline framework</td></tr><tr><th>PAFO <sup><a href="#fn:64">64</a></sup></th><td>FP+MC, cache-aware scheduling</td><td>for SSPPR, parallelizes FORA, work-depth model</td></tr><tr><th>Delta-Push <sup><a href="#fn:70">70</a></sup></th><td>FP+MC, parallel push</td><td>for SSPPR or top- <math><semantics><mi>k</mi> <annotation>k</annotation></semantics></math> PPR, MPC model</td></tr><tr><th rowspan="2">Weighted Graphs</th><th><sup><a href="#fn:125">125</a></sup></th><td>model reduction</td><td>for SSPPR, based on algebraic techniques</td></tr><tr><th>EdgePush <sup><a href="#fn:126">126</a></sup></th><td>edge-based FP</td><td>for SSPPR, superior complexity for imbalanced weights</td></tr><tr><th rowspan="2">Graph Streams</th><th><sup><a href="#fn:127">127</a></sup> <sup><a href="#fn:128">128</a></sup></th><td>estimating mixing time</td><td>estimates PageRank values above a threshold</td></tr><tr><th><sup><a href="#fn:129">129</a></sup></th><td>generating random walks</td><td>estimates PageRank values, single-pass, on random stream</td></tr></tbody></table>

## XII Conclusions

This survey studies efficient PPR computation and elaborates on five basic techniques: the Monte Carlo method, Power Iteration, Forward Push, Reverse Power Iteration, and Backward Push. We then conduct an extensive survey of recent PPR algorithms from an algorithmic perspective, summarizing their rationales and contributions. Our taxonomy is based on the query types, including Single-Source PPR, Single-Target PPR, Single-Pair PPR, and Top- $k$ PPR queries. In addition, we review several algorithms tailored to special settings, such as on dynamic graphs and in parallel/distributed environments. We also give high-level comparisons of PPR algorithms based on their techniques and point out future research directions, aiming to inspire and guide subsequent advancements in the field of PPR computation.

## Acknowledgments

This research was supported by NSFC of China (No. U2241212, No. 61932001, No. U2001212, No. U1936205), Beijing Natural Science Foundation (No. 4222028), Beijing Outstanding Young Scientist Program No.BJJWZYJH012019100020098, Hong Kong RGC GRF (No. 14217322), Hong Kong RGC CRF (No. C4158-20G), Hong Kong ITC ITF (No. MRP/071/20X), CUHK Direct Grant (No. 4055181), and Engineering Research Center of Next-Generation Intelligent Search and Recommendation, Ministry of Education. We thank the anonymous reviewers for their valuable comments.

## References

![[Uncaptioned image]](https://arxiv.org/html/2403.05198v1/Mingji_Yang.jpg)

Mingji Yang

![[Uncaptioned image]](https://arxiv.org/html/2403.05198v1/Hanzhi_Wang.jpg)

Hanzhi Wang

![[Uncaptioned image]](https://arxiv.org/html/2403.05198v1/Zhewei_Wei.jpg)

Zhewei Wei

![[Uncaptioned image]](https://arxiv.org/html/2403.05198v1/Sibo_Wang.png)

![[Uncaptioned image]](https://arxiv.org/html/2403.05198v1/Ji-Rong_Wen.jpeg)

Ji-Rong Wen

[^1]: S. Brin and L. Page, “The anatomy of a large-scale hypertextual web search engine,” *Comput. Netw.*, vol. 30, no. 1-7, pp. 107–117, 1998.

[^2]: S. Wang and Y. Tao, “Efficient algorithms for finding approximate heavy hitters in personalized pageranks,” in *Proc. ACM SIGMOD Int. Conf. Manage. Data*, 2018, pp. 1113–1127.

[^3]: X. Wu, V. Kumar, J. R. Quinlan, J. Ghosh, Q. Yang, H. Motoda, G. J. McLachlan, A. F. M. Ng, B. Liu, P. S. Yu, Z. Zhou, M. S. Steinbach, D. J. Hand, and D. Steinberg, “Top 10 algorithms in data mining,” *Knowl. Inf. Syst.*, vol. 14, no. 1, pp. 1–37, 2008.

[^4]: D. F. Gleich, “Pagerank beyond the web,” *SIAM Rev.*, vol. 57, no. 3, pp. 321–363, 2015.

[^5]: R. Andersen, F. R. K. Chung, and K. J. Lang, “Local graph partitioning using pagerank vectors,” in *Proc. 47th Annu. IEEE Symp. Found. Comput. Sci.*, 2006, pp. 475–486.

[^6]: ——, “Using pagerank to locally partition a graph,” *Internet Math.*, vol. 4, no. 1, pp. 35–64, 2007.

[^7]: H. Yin, A. R. Benson, J. Leskovec, and D. F. Gleich, “Local higher-order graph clustering,” in *Proc. 23rd ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining*, 2017, pp. 555–564.

[^8]: R. Yang, X. Xiao, Z. Wei, S. S. Bhowmick, J. Zhao, and R. Li, “Efficient estimation of heat kernel pagerank for local clustering,” in *Proc. ACM SIGMOD Int. Conf. Manage. Data*, 2019, pp. 1339–1356.

[^9]: K. Fountoulakis, F. Roosta-Khorasani, J. Shun, X. Cheng, and M. W. Mahoney, “Variational perspective on local graph clustering,” *Math. Program.*, vol. 174, no. 1-2, pp. 553–573, 2019.

[^10]: Y. Chang, H. Ma, L. Chang, and Z. Li, “Community detection with attributed random walk via seed replacement,” *Frontiers Comput. Sci.*, vol. 16, no. 5, p. 165324, 2022.

[^11]: Z. Yuan, Z. Wei, F. Lv, and J. Wen, “Index-free triangle-based graph local clustering,” *Frontiers Comput. Sci.*, vol. 18, no. 3, p. 183404, 2024.

[^12]: M. Ou, P. Cui, J. Pei, Z. Zhang, and W. Zhu, “Asymmetric transitivity preserving graph embedding,” in *Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining*, 2016, pp. 1105–1114.

[^13]: C. Zhou, Y. Liu, X. Liu, Z. Liu, and J. Gao, “Scalable graph embedding for asymmetric proximity,” in *Proc. 31st AAAI Conf. Artif. Intell.*, 2017, pp. 2942–2948.

[^14]: A. Tsitsulin, D. Mottin, P. Karras, and E. Müller, “Verse: Versatile graph embeddings from similarity measures,” in *Proc. Int. Conf. World Wide Web*, 2018, pp. 539–548.

[^15]: Y. Yin and Z. Wei, “Scalable graph embeddings via sparse transpose proximities,” in *Proc. 25th ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining*, 2019, pp. 1429–1437.

[^16]: R. Yang, J. Shi, X. Xiao, Y. Yang, and S. S. Bhowmick, “Homogeneous network embedding for massive graphs via reweighted personalized pagerank,” *Proc. VLDB Endowment*, vol. 13, no. 5, pp. 670–683, 2020.

[^17]: A. Tsitsulin, M. Munkhoeva, D. Mottin, P. Karras, I. V. Oseledets, and E. Müller, “Frede: Anytime graph embeddings,” *Proc. VLDB Endowment*, vol. 14, no. 6, pp. 1102–1110, 2021.

[^18]: J. Klicpera, A. Bojchevski, and S. Günnemann, “Predict then propagate: Graph neural networks meet personalized pagerank,” in *Proc. 7th Int. Conf. Learn. Representations*, 2019.

[^19]: A. Bojchevski, J. Klicpera, B. Perozzi, A. Kapoor, M. Blais, B. Rózemberczki, M. Lukasik, and S. Günnemann, “Scaling graph neural networks with approximate pagerank,” in *Proc. 26th ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining*, 2020, pp. 2464–2473.

[^20]: M. Chen, Z. Wei, B. Ding, Y. Li, Y. Yuan, X. Du, and J. Wen, “Scalable graph neural networks via bidirectional propagation,” in *Proc. Annu. Conf. Neural Inf. Process. Syst.*, 2020.

[^21]: H. Wang, M. He, Z. Wei, S. Wang, Y. Yuan, X. Du, and J. Wen, “Approximate graph propagation,” in *Proc. 27th ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining*, 2021, pp. 1686–1696.

[^22]: Y. Zhang, C. Li, C. Xie, and H. Chen, “Accuracy estimation of link-based similarity measures and its application,” *Frontiers Comput. Sci.*, vol. 10, no. 1, pp. 113–123, 2016.

[^23]: A. Epasto, V. Mirrokni, B. Perozzi, A. Tsitsulin, and P. Zhong, “Differentially private graph learning via sensitivity-bounded personalized pagerank,” in *Proc. Annu. Conf. Neural Inf. Process. Syst.*, 2022.

[^24]: R. Yang, “Efficient and effective similarity search over bipartite graphs,” in *ACM Web Conf.*, 2022, pp. 308–318.

[^25]: S. Zhang, R. Yang, X. Xiao, X. Yan, and B. Tang, “Effective and efficient pagerank-based positioning for graph visualization,” *Proc. ACM Manage. Data*, vol. 1, no. 1, pp. 76:1–76:27, 2023.

[^26]: C. Moler, “The world’s largest matrix computation,” *Matlab News Notes*, pp. 12–13, 2002.

[^27]: P. Berkhin, “Survey: A survey on pagerank computing,” *Internet Math.*, vol. 2, no. 1, pp. 73–120, 2005.

[^28]: F. Chung, “A brief survey of pagerank algorithms,” *IEEE Trans. Netw. Sci. Eng.*, vol. 1, no. 1, pp. 38–42, 2014.

[^29]: S. Park, W. Lee, B. Choe, and S. Lee, “A survey on personalized pagerank computation algorithms,” *IEEE Access*, vol. 7, pp. 163 049–163 062, 2019.

[^30]: K. Avrachenkov, N. Litvak, D. Nemirovsky, and N. Osipova, “Monte carlo methods in pagerank computation: When one iteration is sufficient,” *SIAM J. Numer. Anal.*, vol. 45, no. 2, pp. 890–904, 2007.

[^31]: M. Bianchini, M. Gori, and F. Scarselli, “Inside pagerank,” *ACM Trans. Internet Technol.*, vol. 5, no. 1, pp. 92–128, 2005.

[^32]: G. Jeh and J. Widom, “Scaling personalized web search,” in *Proc. Int. Conf. World Wide Web*, 2003, pp. 271–279.

[^33]: K. Avrachenkov, P. Gonçalves, and M. Sokol, “On the choice of kernel and labelled data in semi-supervised learning methods,” in *Proc. 10th Int. Workshop Algorithms Models Web Graph*, vol. 8305, 2013, pp. 56–67.

[^34]: A. N. Langville and C. D. Meyer, “Survey: Deeper inside pagerank,” *Internet Math.*, vol. 1, no. 3, pp. 335–380, 2003.

[^35]: D. Fogaras, B. Rácz, K. Csalogány, and T. Sarlós, “Towards scaling fully personalized pagerank: Algorithms, lower bounds, and experiments,” *Internet Math.*, vol. 2, no. 3, pp. 333–358, 2005.

[^36]: R. Andersen, C. Borgs, J. T. Chayes, J. E. Hopcroft, V. S. Mirrokni, and S. Teng, “Local computation of pagerank contributions,” in *Proc. 5th Int. Workshop Algorithms Models Web Graph*, vol. 4863, 2007, pp. 150–165.

[^37]: ——, “Local computation of pagerank contributions,” *Internet Math.*, vol. 5, no. 1, pp. 23–45, 2008.

[^38]: Z. Wei, X. He, X. Xiao, S. Wang, S. Shang, and J. Wen, “Topppr: Top-k personalized pagerank queries with precision guarantees on large graphs,” in *Proc. ACM SIGMOD Int. Conf. Manage. Data*, 2018, pp. 441–456.

[^39]: M. Liao, R. Li, Q. Dai, and G. Wang, “Efficient personalized pagerank computation: A spanning forests sampling based approach,” in *Proc. ACM SIGMOD Int. Conf. Manage. Data*, 2022, pp. 2048–2061.

[^40]: Y. Saad, *Iterative methods for sparse linear systems*. SIAM, 2003.

[^41]: M. Yoon, J. Jung, and U. Kang, “Tpa: Fast, scalable, and accurate method for approximate random walk with restart on billion scale graphs,” in *Proc. 34th Int. Conf. Data Eng.*, 2018, pp. 1132–1143.

[^42]: Z. Chen, X. Guo, B. Zhou, D. Yang, and S. Skiena, “Accelerating personalized pagerank vector computation,” in *Proc. 29th ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining*, 2023, pp. 262–273.

[^43]: H. Wu, J. Gan, Z. Wei, and R. Zhang, “Unifying the global and local approaches: An efficient power iteration with forward push,” in *Proc. ACM SIGMOD Int. Conf. Manage. Data*, 2021, pp. 1996–2008.

[^44]: S. D. Kamvar, T. H. Haveliwala, C. D. Manning, and G. H. Golub, “Extrapolation methods for accelerating pagerank computations,” in *Proc. Int. Conf. World Wide Web*, 2003, pp. 261–270.

[^45]: A. Z. Broder, R. Lempel, F. Maghoul, and J. O. Pedersen, “Efficient pagerank approximation via graph aggregation,” in *Proc. Int. Conf. World Wide Web*, 2004, pp. 484–485.

[^46]: S. Kamvar, T. Haveliwala, and G. Golub, “Adaptive methods for the computation of pagerank,” *Linear Algebra Appl.*, vol. 386, pp. 51–65, 2004.

[^47]: G. Wu and Y. Wei, “Arnoldi versus gmres for computing pagerank: A theoretical contribution to google’s pagerank problem,” *ACM Trans. Inf. Syst.*, vol. 28, no. 3, pp. 11:1–11:28, 2010.

[^48]: H. Tong, C. Faloutsos, and J. Pan, “Fast random walk with restart and its applications,” in *Proc. 6th Int. Conf. Data Mining*, 2006, pp. 613–622.

[^49]: D. F. Gleich and M. Polito, “Approximating personalized pagerank with minimal use of web graph data,” *Internet Math.*, vol. 3, no. 3, pp. 257–294, 2007.

[^50]: T. Sarlós, A. A. Benczúr, K. Csalogány, D. Fogaras, and B. Rácz, “To randomize or not to randomize: space optimal summaries for hyperlink analysis,” in *Proc. Int. Conf. World Wide Web*, 2006, pp. 297–306.

[^51]: F. Zhu, Y. Fang, K. C. Chang, and J. Ying, “Incremental and accuracy-aware personalized pagerank through scheduled approximation,” *Proc. VLDB Endowment*, vol. 6, no. 6, pp. 481–492, 2013.

[^52]: T. Maehara, T. Akiba, Y. Iwata, and K. Kawarabayashi, “Computing personalized pagerank quickly by exploiting graph structures,” *Proc. VLDB Endowment*, vol. 7, no. 12, pp. 1023–1034, 2014.

[^53]: K. Shin, J. Jung, L. Sael, and U. Kang, “Bear: Block elimination approach for random walk with restart on large graphs,” in *Proc. ACM SIGMOD Int. Conf. Manage. Data*, 2015, pp. 1571–1585.

[^54]: J. Jung, K. Shin, L. Sael, and U. Kang, “Random walk with restart on large graphs using block elimination,” *ACM Trans. Database Syst.*, vol. 41, no. 2, pp. 12:1–12:43, 2016.

[^55]: J. Jung, N. Park, L. Sael, and U. Kang, “Bepi: Fast and memory-efficient method for billion-scale random walk with restart,” in *Proc. ACM SIGMOD Int. Conf. Manage. Data*, 2017, pp. 789–804.

[^56]: M. Coşkun, A. Grama, and M. Koyutürk, “Efficient processing of network proximity queries via chebyshev acceleration,” in *Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining*, 2016, pp. 1515–1524.

[^57]: P. Berkhin, “Bookmark-coloring algorithm for personalized pagerank computing,” *Internet Math.*, vol. 3, no. 1, pp. 41–62, 2006.

[^58]: S. Chakrabarti, “Dynamic personalized pagerank in entity-relation graphs,” in *Proc. Int. Conf. World Wide Web*, 2007, pp. 571–580.

[^59]: F. C. Graham and W. Zhao, “A sharp pagerank algorithm with applications to edge ranking and graph sparsification,” in *Proc. 7th Int. Workshop Algorithms Models Web Graph*, vol. 6516, 2010, pp. 2–14.

[^60]: S. Wang, R. Yang, X. Xiao, Z. Wei, and Y. Yang, “Fora: Simple and effective approximate single-source personalized pagerank,” in *Proc. 23rd ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining*, 2017, pp. 505–514.

[^61]: S. Wang, R. Yang, R. Wang, X. Xiao, Z. Wei, W. Lin, Y. Yang, and N. Tang, “Efficient algorithms for approximate single-source personalized pagerank queries,” *ACM Trans. Database Syst.*, vol. 44, no. 4, pp. 18:1–18:37, 2019.

[^62]: S. Luo, X. Xiao, W. Lin, and B. Kao, “Efficient batch one-hop personalized pageranks,” in *Proc. 35th Int. Conf. Data Eng.*, 2019, pp. 1562–1565.

[^63]: ——, “Baton: Batch one-hop personalized pageranks with efficiency and accuracy,” *IEEE Trans. Knowl. Data Eng.*, vol. 32, no. 10, pp. 1897–1908, 2020.

[^64]: R. Wang, S. Wang, and X. Zhou, “Parallelizing approximate single-source personalized pagerank queries on shared memory,” *VLDB J.*, vol. 28, no. 6, pp. 923–940, 2019.

[^65]: D. Lin, R. C. Wong, M. Xie, and V. J. Wei, “Index-free approach with theoretical guarantee for efficient random walk with restart query,” in *Proc. 36th Int. Conf. Data Eng.*, 2020, pp. 913–924.

[^66]: S. Luo, “Improved communication cost in distributed pagerank computation - a theoretical study,” in *Proc. 37th Int. Conf. Mach. Learn.*, vol. 119, 2020, pp. 6459–6467.

[^67]: G. Hou, X. Chen, S. Wang, and Z. Wei, “Massively parallel algorithms for personalized pagerank,” *Proc. VLDB Endowment*, vol. 14, no. 9, pp. 1668–1680, 2021.

[^68]: D. Mo and S. Luo, “Agenda: Robust personalized pageranks in evolving graphs,” in *Proc. 30th ACM Int. Conf. Inf. Knowl. Manage.*, 2021, pp. 1315–1324.

[^69]: M. Liao, R. Li, Q. Dai, H. Chen, H. Qin, and G. Wang, “Efficient personalized pagerank computation: The power of variance-reduced monte carlo approaches,” *Proc. ACM Manage. Data*, vol. 1, no. 2, pp. 160:1–160:26, 2023.

[^70]: G. Hou, Q. Guo, F. Zhang, S. Wang, and Z. Wei, “Personalized pagerank on evolving graphs with an incremental index-update scheme,” *Proc. ACM Manage. Data*, vol. 1, no. 1, pp. 25:1–25:26, 2023.

[^71]: K. Fountoulakis and S. Yang, “Open problem: Running time complexity of accelerated $\ell_{1}$ -regularized pagerank,” in *Annu. Conf. Learn. Theory*, 2022, pp. 5630–5632.

[^72]: D. Martínez-Rubio, E. S. Wirth, and S. Pokutta, “Accelerated and sparse algorithms for approximate personalized pagerank and beyond,” in *Annu. Conf. Learn. Theory*, vol. 195, 2023, pp. 2852–2876.

[^73]: P. Lofgren and A. Goel, “Personalized pagerank to a target node,” *CoRR*, vol. abs/1304.4658, 2013.

[^74]: H. Wang, Z. Wei, J. Gan, S. Wang, and Z. Huang, “Personalized pagerank to a target node, revisited,” in *Proc. 26th ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining*, 2020, pp. 657–667.

[^75]: Y. Fujiwara, M. Nakatsuji, T. Yamamuro, H. Shiokawa, and M. Onizuka, “Efficient personalized pagerank with accuracy assurance,” in *Proc. 18th ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining*, 2012, pp. 15–23.

[^76]: X. Zhao, A. Chang, A. D. Sarma, H. Zheng, and B. Y. Zhao, “On the embeddability of random walk distances,” *Proc. VLDB Endowment*, vol. 6, no. 14, pp. 1690–1701, 2013.

[^77]: P. Lofgren, S. Banerjee, A. Goel, and S. Comandur, “Fast-ppr: scaling personalized pagerank estimation for large graphs,” in *Proc. 20th ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining*, 2014, pp. 1436–1445.

[^78]: P. Lofgren, S. Banerjee, and A. Goel, “Personalized pagerank estimation and search: A bidirectional approach,” in *Proc. 9th ACM Int. Conf. Web Search Data Mining*, 2016, pp. 163–172.

[^79]: ——, “Bidirectional pagerank estimation: From average-case to worst-case,” in *Proc. 12th Int. Workshop Algorithms Models Web Graph*, vol. 9479, 2015, pp. 164–176.

[^80]: S. Wang, Y. Tang, X. Xiao, Y. Yang, and Z. Li, “Hubppr: Effective indexing for approximate personalized pagerank,” *Proc. VLDB Endowment*, vol. 10, no. 3, pp. 205–216, 2016.

[^81]: M. Bressan, E. Peserico, and L. Pretto, “Sublinear algorithms for local graph centrality estimation,” in *Proc. 59th Annu. IEEE Symp. Found. Comput. Sci.*, 2018, pp. 709–718.

[^82]: ——, “Sublinear algorithms for local graph-centrality estimation,” *SIAM J. Comput.*, vol. 52, no. 4, pp. 968–1008, 2023.

[^83]: Y. Chen, Q. Gan, and T. Suel, “Local methods for estimating pagerank values,” in *Proc. 13th ACM Int. Conf. Inf. Knowl. Manage.*, 2004, pp. 381–389.

[^84]: Z. Bar-Yossef and L. Mashiach, “Local approximation of pagerank and reverse pagerank,” in *Proc. 17th ACM Int. Conf. Inf. Knowl. Manage.*, 2008, pp. 279–288.

[^85]: M. Bressan, E. Peserico, and L. Pretto, “The power of local information in pagerank,” in *Proc. Int. Conf. World Wide Web*, 2013, pp. 179–180.

[^86]: H. Wang and Z. Wei, “Estimating single-node pagerank in $\widetilde{O}\left(\min\big\{d_{t},\sqrt{m}\big\}\right)$ time,” *Proc. VLDB Endowment*, vol. 16, no. 11, pp. 2949–2961, 2023.

[^87]: Y. Fujiwara, M. Nakatsuji, M. Onizuka, and M. Kitsuregawa, “Fast and exact top-k search for random walk with restart,” *Proc. VLDB Endowment*, vol. 5, no. 5, pp. 442–453, 2012.

[^88]: Y. Fujiwara, M. Nakatsuji, H. Shiokawa, T. Mishima, and M. Onizuka, “Efficient ad-hoc search for personalized pagerank,” in *Proc. ACM SIGMOD Int. Conf. Manage. Data*, 2013, pp. 445–456.

[^89]: ——, “Fast and exact top-k algorithm for pagerank,” in *Proc. 27th AAAI Conf. Artif. Intell.*, 2013, pp. 1106–1112.

[^90]: Y. Wu, R. Jin, and X. Zhang, “Fast and unified local search for random walk based k-nearest-neighbor query in large graphs,” in *Proc. ACM SIGMOD Int. Conf. Manage. Data*, 2014, pp. 1139–1150.

[^91]: M. S. Gupta, A. Pathak, and S. Chakrabarti, “Fast algorithms for topk personalized pagerank queries,” in *Proc. Int. Conf. World Wide Web*, 2008, pp. 1225–1226.

[^92]: S. Chakrabarti, A. Pathak, and M. Gupta, “Index design and query processing for graph conductance search,” *VLDB J.*, vol. 20, no. 3, pp. 445–470, 2011.

[^93]: A. Pathak, S. Chakrabarti, and M. S. Gupta, “Index design for dynamic personalized pagerank,” in *Proc. 24th Int. Conf. Data Eng.*, 2008, pp. 1489–1491.

[^94]: K. Avrachenkov, N. Litvak, D. Nemirovsky, E. Smirnova, and M. Sokol, “Quick detection of top-k personalized pagerank lists,” in *Proc. 8th Int. Workshop Algorithms Models Web Graph*, vol. 6732, 2011, pp. 50–61.

[^95]: J. V. Davis and I. S. Dhillon, “Estimating the global pagerank of web communities,” in *Proc. 12th ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining*, 2006, pp. 116–125.

[^96]: M. Bressan and L. Pretto, “Local computation of pagerank: the ranking side,” in *Proc. 20th ACM Int. Conf. Inf. Knowl. Manage.*, 2011, pp. 631–640.

[^97]: C. Borgs, M. Brautbar, J. T. Chayes, and S. Teng, “A sublinear time algorithm for pagerank computations,” in *Proc. 9th Int. Workshop Algorithms Models Web Graph*, vol. 7323, 2012, pp. 41–53.

[^98]: ——, “Multiscale matrix sampling and sublinear-time pagerank computation,” *Internet Math.*, vol. 10, no. 1-2, pp. 20–48, 2014.

[^99]: A. W. Yu, N. Mamoulis, and H. Su, “Reverse top-k search using random walk with restart,” *Proc. VLDB Endowment*, vol. 7, no. 5, pp. 401–412, 2014.

[^100]: P. Sarkar and A. W. Moore, “Fast nearest-neighbor search in disk-resident graphs,” in *Proc. 16th ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining*, 2010, pp. 513–522.

[^101]: B. Bahmani, A. Chowdhury, and A. Goel, “Fast incremental and personalized pagerank,” *Proc. VLDB Endowment*, vol. 4, no. 3, pp. 173–184, 2010.

[^102]: H. Zhang, P. Lofgren, and A. Goel, “Approximate personalized pagerank on dynamic graphs,” in *Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining*, 2016, pp. 1315–1324.

[^103]: B. Bahmani, R. Kumar, M. Mahdian, and E. Upfal, “Pagerank on an evolving graph,” in *Proc. 18th ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining*, 2012, pp. 24–32.

[^104]: W. Yu and X. Lin, “Irwr: incremental random walk with restart,” in *Proc. 36th ACM SIGIR Int. Conf. Res. Develop. Inf. Retrieval*, 2013, pp. 1017–1020.

[^105]: W. Yu and J. A. McCann, “Random walk with restart over dynamic graphs,” in *Proc. 16th Int. Conf. Data Mining*, 2016, pp. 589–598.

[^106]: N. Ohsaka, T. Maehara, and K. Kawarabayashi, “Efficient pagerank tracking in evolving networks,” in *Proc. 21st ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining*, 2015, pp. 875–884.

[^107]: M. Yoon, W. Jin, and U. Kang, “Fast and accurate random walk with restart on dynamic graphs with guarantees,” in *Proc. Int. Conf. World Wide Web*, 2018, pp. 409–418.

[^108]: D. Mo and S. Luo, “Single-source personalized pageranks with workload robustness,” *IEEE Trans. Knowl. Data Eng.*, vol. 35, no. 6, pp. 6320–6334, 2023.

[^109]: K. Sankaralingam, S. Sethumadhavan, and J. C. Browne, “Distributed pagerank for p2p systems,” in *12th Int. Symp. High-Perform. Distrib. Comput.*, 2003, pp. 58–69.

[^110]: S. Shi, J. Yu, G. Yang, and D. Wang, “Distributed page ranking in structured p2p networks,” in *32nd Int. Conf. Parallel Process.*, 2003, pp. 179–186.

[^111]: Y. Wang and D. J. DeWitt, “Computing pagerank in a distributed internet search engine system,” in *Proc. 30th Int. Conf. Very Large Data Bases*, 2004, pp. 420–431.

[^112]: J. X. Parreira, D. Donato, S. Michel, and G. Weikum, “Efficient and decentralized pagerank approximation in a peer-to-peer web search network,” in *Proc. 32nd Int. Conf. Very Large Data Bases*, 2006, pp. 415–426.

[^113]: Y. Zhu, S. Ye, and X. Li, “Distributed pagerank computation based on iterative aggregation-disaggregation methods,” in *Proc. 14th ACM Int. Conf. Inf. Knowl. Manage.*, 2005, pp. 578–585.

[^114]: B. Bahmani, K. Chakrabarti, and D. Xin, “Fast personalized pagerank on mapreduce,” in *Proc. ACM SIGMOD Int. Conf. Manage. Data*, 2011, pp. 973–984.

[^115]: I. Mitliagkas, M. Borokhovich, A. G. Dimakis, and C. Caramanis, “Frogwild! - fast pagerank approximations on graph engines,” *Proc. VLDB Endowment*, vol. 8, no. 8, pp. 874–885, 2015.

[^116]: Q. Liu, Z. Li, J. C. S. Lui, and J. Cheng, “Powerwalk: Scalable personalized pagerank via random walks with vertex-centric decomposition,” in *Proc. 25th ACM Int. Conf. Inf. Knowl. Manage.*, 2016, pp. 195–204.

[^117]: T. Guo, X. Cao, G. Cong, J. Lu, and X. Lin, “Distributed algorithms on exact personalized pagerank,” in *Proc. ACM SIGMOD Int. Conf. Manage. Data*, 2017, pp. 479–494.

[^118]: W. Guo, Y. Li, M. Sha, and K. Tan, “Parallel personalized pagerank on dynamic graphs,” *Proc. VLDB Endowment*, vol. 11, no. 1, pp. 93–106, 2017.

[^119]: J. Shi, R. Yang, T. Jin, X. Xiao, and Y. Yang, “Realtime top-k personalized pagerank over large graphs on gpus,” *Proc. VLDB Endowment*, vol. 13, no. 1, pp. 15–28, 2019.

[^120]: W. Lin, “Distributed algorithms for fully personalized pagerank on large graphs,” in *Proc. Int. Conf. World Wide Web*, 2019, pp. 1084–1094.

[^121]: A. D. Sarma, A. R. Molla, G. Pandurangan, and E. Upfal, “Fast distributed pagerank computation,” *Theor. Comput. Sci.*, vol. 561, pp. 113–121, 2015.

[^122]: S. Luo, “Distributed pagerank computation: An improved theoretical study,” in *Proc. 33rd AAAI Conf. Artif. Intell.*, 2019, pp. 4496–4503.

[^123]: J. Łącki, S. Mitrović, K. Onak, and P. Sankowski, “Walking randomly, massively, and efficiently,” in *Proc. 52nd Annu. ACM SIGACT Symp. Theory Comput.*, 2020, pp. 364–377.

[^124]: M. Kapralov, S. Lattanzi, N. Nouri, and J. Tardos, “Efficient and local parallel random walks,” in *Proc. Annu. Conf. Neural Inf. Process. Syst.*, 2021, pp. 21 375–21 387.

[^125]: W. Xie, D. Bindel, A. J. Demers, and J. Gehrke, “Edge-weighted personalized pagerank: Breaking A decade-old performance barrier,” in *Proc. 21st ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining*, 2015, pp. 1325–1334.

[^126]: H. Wang, Z. Wei, J. Gan, Y. Yuan, X. Du, and J. Wen, “Edge-based local push for personalized pagerank,” *Proc. VLDB Endowment*, vol. 15, no. 7, pp. 1376–1389, 2022.

[^127]: A. D. Sarma, S. Gollapudi, and R. Panigrahy, “Estimating pagerank on graph streams,” in *Proc. 27th ACM SIGMOD-SIGACT-SIGART Symp. Princ. Database Syst.*, 2008, pp. 69–78.

[^128]: ——, “Estimating pagerank on graph streams,” *J. ACM*, vol. 58, no. 3, pp. 13:1–13:19, 2011.

[^129]: J. Kallaugher, M. Kapralov, and E. Price, “Simulating random walks in random streams,” in *Proc. ACM-SIAM Symp. Discrete Algorithms*, 2022, pp. 3091–3126.
