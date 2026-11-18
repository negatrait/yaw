# Sprint Spec: Parameter-Free MDL Evidence-DAG & Stage-1 RaBitQ Cutover (`mdl-dag-sprint`)

## 0. Agent Execution Rules (Anti-Spaghetti Override)

You are executing a clean-slate algorithmic replacement governed by `AGENTS.md` ("A minimal line count codebase that fails fast is infinitely better than a fallback-heavy codebase that silently degrades performance to mask failures").

**Hard Constraints — Violations Fail Review Automatically:**
1. **Zero New Modules:** Keep the exact flat package layout (`yaw/*.py` and `yaw/core/*.py`). Do **not** create `yaw/daemon/...`, `yaw/manifold.py`, `yaw/dag.py`, or helper subpackages.
2. **Zero Legacy Shims or Feature Flags:** Do **not** add `retriever_mode`, `use_mdl_dag`, `legacy_ppr`, or `try ... except sqlite3.OperationalError: pass` fallbacks. Delete superseded functions, dataclasses, and dependencies outright.
3. **Do Not Approximate the Math With Heuristics:** Implement the four reference kernels in §2 (`ingest_gate`, `fit_manifold`/`whiten_normalize`, `rabitq_encode`/`rabitq_query`, `cut_chain_atom`/`select_mdl_dag`) using exact linear algebra (`np.linalg.svd`, `np.linalg.qr`, `np.bitwise_count`, `scipy.optimize.nnls`, `graphlib.TopologicalSorter`, `zlib.compressobj`). Replacing `nnls` or QR bridge projection with ad-hoc scalar weighted sums is forbidden.
4. **Hard LOC Ceilings:**
   - `yaw/retrieval.py` <= **430 LOC** (currently ~370 LOC).
   - `yaw/pipeline.py` <= **430 LOC** (currently ~390 LOC).
   - Net repository LOC across `yaw/` **must decrease** by the end of `MDL-3`.
5. **Quality Gates After Every Checkpoint (`MDL-1`, `MDL-2`, `MDL-3`):**
   ```bash
   uv run ty check
   uv run ruff check --fix
   uv run ruff format
   uv run pytest -o addopts=""
   git diff --check
   ```
   (`-o addopts=""` bypasses `--testmon` in `pyproject.toml` so SQLite/schema/asset changes are never silently skipped.)

---

## 1. Evidence Ledger: Why Current `dev` Must Be Replaced

1. **Title & AST-Island Starvation (`yaw/pipeline.py`):** `chunk_file_content` hardcodes `SourceChunk(..., 1, len(lines), "", chunk.text)`. Every chunk embeds as `"title: none | text: ..."` in `format_doc_for_embedding`, `chunk_fts` receives empty `title` values, `json_document_fields` in `yaw/core/types.py` is never called, and `expand_neighbors` in `yaw/qa.py` adds `+0.50` to *every* chunk in the same file because `(1, len(lines)) != (1, 1)`.
2. **Broken FTS5 Ranking & Silent Fallback (`yaw/retrieval.py`):** `_bm25_scores` executes `SELECT rowid, -bm25(chunk_fts, 2.0, 1.0) FROM chunk_fts WHERE chunk_fts MATCH ? LIMIT ?` **without `ORDER BY bm25(chunk_fts)`**, returning arbitrary lowest-`rowid` matches on corpora with `>256` hits, followed by `except sqlite3.OperationalError: pass`.
3. **Unbounded Python Heap & Index-Time NLP Tax (`yaw/pipeline.py`, `yaw/retrieval.py`):** `build_vector_search` loads all `N x D` `float32` vectors (`3,072B/chunk`) plus `json.loads(tokens_json)` and `json.loads(entities_json)` (`spacy` `en_core_web_sm`) into Python RAM on every `PRAGMA data_version` change.
4. **Split-Brain Hyperparameters (`yaw/core/config.py`, `yaw/optimize.py`, `yaw/retrieval.py`):** `YawConfig.retrieval` defines 5 unused `RetrievalParams` (`peclet`, `alpha_sprt`, ...); `search_vectors` runs entity-PPR (`hybrid_alpha`, `ppr_alpha`, `seed_top_k`); `optimize_optuna` writes `.yaw/optuna_best_params.json`, which nothing loads.
5. **Eager LLM & Catalog Coupling (`yaw/runtime.py`, `yaw/cli.py`):** `start_runtime` eagerly instantiates `create_lm(cfg)` even when only zero-LLM `recall` is used, and `yaw/cli.py` eagerly imports `yaw.optimize` at module import time.

---

## 2. Verified Reference Kernels (Implement Directly; Do Not Improvise)

All four kernels below handle degenerate `pytest` fixtures (`N = 1..7`, arbitrary `D`) and scale to `N = 10^6` without `NaN`s or `CycleError`s.

### 2.1 Conditional `zlib(zdict)` Ingest Gate (`stdlib` `hashlib` + `zlib`, `~25µs`/chunk)
Approximates conditional Kolmogorov complexity $\eta_{\text{corpus}}(u) = |\texttt{zlib}(u \mid \text{zdict}=\text{proto})| / |\texttt{zlib}(u)|$ to skip redundant neural embedding calls on log/boilerplate clones while preserving full `FTS5` indexing.

```python
import hashlib
import re
import zlib

_VOLATILE_RE = re.compile(
    r"\b(?:0x[0-9a-fA-F]+|[0-9a-fA-F]{8}-[0-9a-fA-F-]{27}|\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[\w.:+-]*|\d+)\b"
)
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}")


def zlen_bytes(data: bytes, zdict: bytes = b"") -> int:
    if not data:
        return 1
    comp = zlib.compressobj(6, zdict=zdict[-32768:]) if zdict else zlib.compressobj(6)
    return max(1, len(comp.compress(data) + comp.flush()))


def simhash64(text: str) -> int:
    masked = _VOLATILE_RE.sub("#", text.lower())
    acc = [0] * 64
    tokens = _WORD_RE.findall(masked) or [masked[:64]]
    for tok in tokens:
        h = int.from_bytes(hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest(), "little")
        for bit in range(64):
            acc[bit] += 1 if ((h >> bit) & 1) else -1
    out = 0
    for bit, val in enumerate(acc):
        if val > 0:
            out |= 1 << bit
    return out - (1 << 64) if out >= (1 << 63) else out  # signed int64 for SQLite INTEGER


def corpus_novelty_q8(body_bytes: bytes, proto_bytes: bytes | None) -> int:
    if not proto_bytes:
        return 255
    ratio = min(1.0, zlen_bytes(body_bytes, proto_bytes) / float(zlen_bytes(body_bytes)))
    return max(1, min(255, int(round(ratio * 255.0))))
```

### 2.2 Parameter-Free Marchenko–Pastur Shrinkage Whitening (`NumPy`)
Computes mean $\mu$, low-rank spike attenuator $(V_{\text{spike}}, \Delta w)$ for eigenvalues exceeding the Marchenko–Pastur upper bulk edge $\lambda_+ = \sigma^2(1 + \sqrt{D/N_s})^2$, and participation-ratio intrinsic dimension $D_{\text{eff}} = (\sum \tilde{\lambda}_i)^2 / \sum \tilde{\lambda}_i^2$.

```python
import numpy as np
from numpy.typing import NDArray


def fit_manifold(
    raw_vecs: NDArray[np.float32], max_samples: int = 4096, seed: int = 0x594157
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], float]:
    n, d = raw_vecs.shape
    if n < 8:
        return (
            np.zeros(d, dtype=np.float32),
            np.zeros((d, 0), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            float(min(d, 64)),
        )
    sample = (
        raw_vecs[np.random.default_rng(seed).choice(n, size=max_samples, replace=False)]
        if n > max_samples
        else raw_vecs
    ).astype(np.float32, copy=False)
    ns = sample.shape[0]
    mu = sample.mean(axis=0, dtype=np.float64).astype(np.float32)
    xc = (sample - mu) / np.sqrt(float(ns - 1))
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    lam = np.maximum(s.astype(np.float64) ** 2, 1e-12)
    sigma2 = float(np.median(lam))
    lam_plus = sigma2 * ((1.0 + np.sqrt(d / float(ns))) ** 2)
    spikes = lam > lam_plus
    w = np.where(spikes, np.sqrt(lam_plus / lam), 1.0)
    lam_w = lam * (w**2)
    d_eff = float(np.clip((lam_w.sum() ** 2) / max(1e-12, float(np.sum(lam_w**2))), 8.0, float(d)))
    return mu, vt[spikes].T.astype(np.float32, copy=True), (w[spikes] - 1.0).astype(np.float32, copy=True), d_eff


def whiten_normalize(
    vecs: NDArray[np.float32],
    mu: NDArray[np.float32],
    v_spike: NDArray[np.float32],
    delta_w: NDArray[np.float32],
) -> NDArray[np.float32]:
    single = vecs.ndim == 1
    x = (vecs[None, :] if single else vecs).astype(np.float32) - mu
    if v_spike.shape[1] > 0:
        x = x + ((x @ v_spike) * delta_w[None, :]) @ v_spike.T
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return x[0] if single else x
```

### 2.3 Stage-1 `RaBitQ` (`100B`/chunk in RAM: `bits_u64` + `s_o` + `ones_count`)
Unbiased inner-product estimator $\hat{y}_o = \langle \bar{x}, P^\top q \rangle / s_o$ with concentration error bound $\epsilon_o = \frac{\sqrt{1 - s_o^2}}{s_o}\frac{t}{\sqrt{D - 1}}$ via 4-bit-plane SIMD `np.bitwise_count` (`numpy>=2.5.4`).

```python
def make_orthogonal_rotation(dim: int, seed: int = 0x594157) -> NDArray[np.float32]:
    q, r = np.linalg.qr(np.random.default_rng(seed + dim).standard_normal((dim, dim)).astype(np.float32))
    q *= np.sign(np.diag(r))
    return q.astype(np.float32, copy=False)


def rabitq_encode(
    unit_vecs: NDArray[np.float32], rot: NDArray[np.float32]
) -> tuple[NDArray[np.uint64], NDArray[np.float16], NDArray[np.uint16]]:
    x = unit_vecs @ rot
    s_o = np.maximum((np.sum(np.abs(x), axis=1) / np.sqrt(x.shape[1])).astype(np.float16), np.float16(1e-4))
    bits_u8 = np.packbits(x >= 0.0, axis=1, bitorder="little")
    if rem := (bits_u8.shape[1] % 8):
        bits_u8 = np.pad(bits_u8, ((0, 0), (0, 8 - rem)))
    bits_u64 = np.ascontiguousarray(bits_u8).view(np.uint64)
    ones = np.bitwise_count(bits_u64).sum(axis=1, dtype=np.uint16)
    return bits_u64, s_o, ones


def rabitq_query(
    bits_u64: NDArray[np.uint64],
    s_o: NDArray[np.float16],
    ones: NDArray[np.uint16],
    q_unit: NDArray[np.float32],
    rot: NDArray[np.float32],
    dim: int,
    t_bound: float = 2.8,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    q_rot = q_unit @ rot
    q_min, q_max = float(q_rot.min()), float(q_rot.max())
    span = max(q_max - q_min, 1e-8)
    q_u4 = np.clip(np.round((q_rot - q_min) * (15.0 / span)), 0, 15).astype(np.uint8)
    dot_b_q = np.zeros(bits_u64.shape[0], dtype=np.int32)
    for plane in range(4):
        p_u8 = np.packbits((q_u4 >> plane) & 1, bitorder="little")
        if rem := (len(p_u8) % 8):
            p_u8 = np.pad(p_u8, (0, 8 - rem))
        p_u64 = p_u8.view(np.uint64)
        dot_b_q += np.bitwise_count(bits_u64 & p_u64[None, :]).sum(axis=1, dtype=np.int32) << plane
    sum_b_q = q_min * ones.astype(np.float32) + (span / 15.0) * dot_b_q.astype(np.float32)
    xbar_dot_q = (2.0 * sum_b_q - float(q_rot.sum())) / np.sqrt(float(dim))
    s_f32 = s_o.astype(np.float32)
    est = (xbar_dot_q / s_f32).astype(np.float32)
    eps = ((np.sqrt(np.maximum(0.0, 1.0 - s_f32 * s_f32)) / s_f32) * (t_bound / np.sqrt(max(1, dim - 1)))).astype(
        np.float32
    )
    return est, eps
```

### 2.4 Hilbert-Space Cut Operator & Parameter-Free MDL `nnls` DAG Selector (`SciPy` + `stdlib`)
Projects internal existentially-bound bridge vectors $Q_B$ out of chain sums, fits $\hat{q}$ in the non-negative cone (`scipy.optimize.nnls`), gates text clones via `zlib(zdict=path_bytes)`, prices unlinked jumps at $\log_2 N$ bits and anchor hops at $\log_2\max(2, \text{df}(e)-1)$ bits, and enforces strict DAG acyclicity before `TopologicalSorter`.

```python
from graphlib import TopologicalSorter
from scipy.optimize import nnls


def cut_chain_atom(chunk_vecs: list[NDArray[np.float32]], bridge_vecs: list[NDArray[np.float32]]) -> NDArray[np.float32]:
    s = np.sum(chunk_vecs, axis=0, dtype=np.float32)
    if bridge_vecs:
        q_b, _ = np.linalg.qr(np.column_stack(bridge_vecs).astype(np.float32), mode="reduced")
        s = s - q_b @ (q_b.T @ s)
    return (s / max(float(np.linalg.norm(s)), 1e-12)).astype(np.float32)


def _creates_cycle(parents: dict[int, set[int]], src: int, dst: int) -> bool:
    if src == dst:
        return True
    stack = [src]
    seen: set[int] = set()
    while stack:
        cur = stack.pop()
        if cur == dst:
            return True
        for par in parents.get(cur, ()):
            if par not in seen:
                seen.add(par)
                stack.append(par)
    return False


def select_mdl_dag(
    q_hat: NDArray[np.float32],
    chains: list[tuple[tuple[int, ...], tuple[tuple[str, int], ...], NDArray[np.float32]]],
    chunk_bytes: dict[int, bytes],
    n_corpus: int,
    d_eff: float,
    max_branches: int = 6,
) -> tuple[tuple[int, ...], tuple[DagEdge, ...], float, float, NDArray[np.float32]]:
    if not chains:
        return (), (), 0.0, 0.0, q_hat
    d_mat = np.column_stack([c[2] for c in chains]).astype(np.float64)
    q64 = q_hat.astype(np.float64)
    root_cost = float(np.log2(max(2, n_corpus)))

    selected: list[int] = []
    seen_nodes: set[int] = set()
    seen_edges: set[tuple[int, int]] = set()
    dag_parents: dict[int, set[int]] = {}
    edge_ledger: list[DagEdge] = []
    r2_curr = 0.0
    net_bits = 0.0
    best_w = np.zeros(0, dtype=np.float64)

    for _ in range(min(max_branches, len(chains))):
        path_zdict = (
            b"\n".join(chunk_bytes.get(cid, b"") for cid in TopologicalSorter(dag_parents).static_order())
            if dag_parents
            else b""
        )
        best_j: int | None = None
        best_gain = 0.0
        best_r2 = r2_curr
        best_delta_s = 0.0
        best_w_trial = best_w

        for j, (nodes, bridges, _) in enumerate(chains):
            if j in selected:
                continue
            w_trial, rnorm = nnls(d_mat[:, [*selected, j]], q64)
            r2_new = float(np.clip(1.0 - rnorm * rnorm, 0.0, 0.999999))
            if r2_new <= r2_curr + 1e-6:
                continue
            delta_s = 0.5 * d_eff * float(np.log2((1.0 - r2_curr) / (1.0 - r2_new)))
            new_nodes = [nid for nid in nodes if nid not in seen_nodes]
            if new_nodes and path_zdict:
                blob = b"\n".join(chunk_bytes.get(nid, b"") for nid in new_nodes)
                eta_path = min(1.0, zlen_bytes(blob, path_zdict) / float(zlen_bytes(blob)))
            else:
                eta_path = 1.0
            delta_k = root_cost if nodes[0] not in seen_nodes else 0.0
            for step_i, (_, df_val) in enumerate(bridges):
                if (nodes[step_i], nodes[step_i + 1]) not in seen_edges:
                    delta_k += float(np.log2(max(2, df_val - 1)))
            gain = eta_path * delta_s - delta_k
            if gain > best_gain:
                best_gain, best_j, best_r2, best_delta_s, best_w_trial = gain, j, r2_new, eta_path * delta_s, w_trial

        if best_j is None:
            break
        selected.append(best_j)
        r2_curr, net_bits, best_w = best_r2, net_bits + best_gain, best_w_trial
        nodes, bridges, _ = chains[best_j]
        for nid in nodes:
            seen_nodes.add(nid)
            dag_parents.setdefault(nid, set())
        per_edge_save = best_delta_s / max(1, len(bridges))
        for step_i, (anchor, df_val) in enumerate(bridges):
            u, v = nodes[step_i], nodes[step_i + 1]
            if (u, v) not in seen_edges and not _creates_cycle(dag_parents, u, v):
                seen_edges.add((u, v))
                dag_parents[v].add(u)
                edge_ledger.append(DagEdge(u, v, anchor, per_edge_save, float(np.log2(max(2, df_val - 1)))))

    if not selected:
        j0 = int(np.argmax(d_mat.T @ q64))
        w0, rn0 = nnls(d_mat[:, [j0]], q64)
        r2_0 = float(np.clip(1.0 - rn0 * rn0, 0.0, 0.999999))
        return chains[j0][0], (), r2_0, 0.0, (q64 - d_mat[:, [j0]] @ w0).astype(np.float32)

    ordered = tuple(TopologicalSorter(dag_parents).static_order())
    residual = (q64 - d_mat[:, selected] @ best_w).astype(np.float32)
    return ordered, tuple(edge_ledger), r2_curr, net_bits, residual
```

---

## 3. Atomic Implementation Checkpoints

### Checkpoint `MDL-1`: Storage, AST Islands, Ingest Gate, and Dual `FTS5`
**Files touched:** `pyproject.toml`, `yaw/core/types.py`, `yaw/core/__init__.py`, `yaw/pipeline.py`.

1. **`pyproject.toml`:** Remove `spacy`, `en-core-web-sm` (and `[tool.uv.sources]`), and `optuna`. Add `tree-sitter-language-pack>=1.25`. Run `uv lock && uv sync`.
2. **`yaw/core/types.py` & `yaw/core/__init__.py`:**
   - Delete dead legacy types (`TextChunk`, `Entity`, `Triple`, `Quad`, `ExtractionRecord`, `DropReason`, `PassMetrics`, `AxiomViolation`, `ValidationSummary`, `ManifoldDiagnostics`, `OntologySummary`, `PipelineResult`, `DatasetStats`, `ChunkMetadata`, `SeedSelection`, `RecallHit`, `RecallResult`, `EvalQuestionResult`, `EvalAggregate`) and dead `.constants` lazy mappings in `yaw/core/__init__.py`.
   - Add `mtime_ns: int = 0` and `novelty_q8: int = 255` to `ChunkPointer`. Add `DagEdge` and `RecallPacket` (§2.4).
3. **`yaw/pipeline.py`:**
   - **Keep** `ChunkVecRow.embedding: NDArray[np.float32]` unchanged so `CocoIndex` `sqlite.Vec0TableDef` continues to mount `chunks` natively.
   - **Replace** `tokens_json` and `entities_json` in `ChunkVecRow` and `_CHUNK_AUXILIARY_COLS` with `mtime_ns: int` and `novelty_q8: int`. Add `anchors: str` to `SourceChunk`.
   - **In `chunk_file_content`:**
     - If `rel_path.endswith(".json")`, extract `(doc_title, raw_body)` via `json_document_fields(json.loads(text))`.
     - Extract AST `(ast_island_start, ast_island_end, title)` and structural `anchors` (symbols, imports, config keys, Markdown headings/links/inline `` `code` ``, plus regex paths/CVEs/identifiers matching `[A-Za-z0-9_./:@$-]+` containing at least one of `_ - . / : @ $`) via `tree-sitter-language-pack`. Never emit hardcoded `1, len(lines), ""` when AST/heading structure exists.
   - **In `_compute_chunk_features`:** Delete `extract_text_features` (`spacy`). Apply `simhash64` + `corpus_novelty_q8` (§2.1) against an `ingest_templates(simhash INTEGER PRIMARY KEY, proto_bytes BLOB, proto_vec BLOB)` table in `yaw.db`: when `novelty_q8 < 51` (`eta < 0.20`), reuse `proto_vec` **without** calling `embedder.embed()`; otherwise call `embedder.embed(format_doc_for_embedding(body, title=title, task_mode=task_mode))` and upsert prototype if bucket empty.
   - **In `_apply_fts_tables` & `_indexing_main`:** Keep `_FtsRowHandler` tuple `(title, body)` intact, and mount **two** contentless FTS5 targets:
     1. `"chunk_fts"`: `USING fts5(title, body, content='', contentless_delete=1, tokenize='unicode61')` + `CREATE VIRTUAL TABLE IF NOT EXISTS "chunk_vocab" USING fts5vocab("chunk_fts", 'row')`.
     2. `"anchor_fts"` (receiving `(piece.title, piece.anchors)`): `USING fts5(title, body, content='', contentless_delete=1, tokenize="unicode61 tokenchars '_-./:@$'")` + `CREATE VIRTUAL TABLE IF NOT EXISTS "anchor_vocab" USING fts5vocab("anchor_fts", 'row')`.
     Update `list_user_tables` `shadow_prefixes` to exclude `"anchor_fts_"`, `"chunk_vocab"`, `"anchor_vocab"`, and `"ingest_templates"`.

---

### Checkpoint `MDL-2`: Stage-1 RaBitQ View & Deterministic MDL DAG Engine
**Files touched:** `yaw/retrieval.py`, `yaw/core/config.py`, `yaw/optimize.py`.

1. **`yaw/retrieval.py`:**
   - **Delete:** `import spacy`, `_get_spacy_nlp`, `extract_text_features`, `entity_keys`, `entity_to_idx`, `entity_to_chunk`, `chunk_to_entity`, `doc_freqs`, `chunk_tokens`, and the silent `except sqlite3.OperationalError: pass` + Python token-loop fallback in `_bm25_scores`.
   - **`VectorSearch` dataclass (Stage-1 Compact View):** Hold **zero** full-corpus `float32` matrices. Store `chunk_ids`, `id_to_pos`, `pointers`, `dim`, `mu`, `v_spike`, `delta_w`, `d_eff`, `rot`, `rabitq_bits` (`uint64[N, ceil(D/64)]`), `rabitq_s` (`float16[N]`), `rabitq_ones` (`uint16[N]`), `csls_rk` (`float16[N]`, mean top-16 Stage-1 neighbor cosine), and `next_in_file` (`int32[N]`, index of the next sequential chunk in the same `rel_path`, or `-1`).
   - **Fix `_bm25_scores`:** Query `chunk_fts` and `anchor_fts` with explicit `ORDER BY bm25(...) ASC LIMIT ?`. Never swallow `sqlite3.OperationalError`.
   - **Implement `solve_evidence_dag(vs, query_embedding, *, query_text, conn, corpus_root, embed_fn=None, task_mode="distinct-titled", top_k=5) -> RecallPacket`:**
     1. **Vocabulary & Zero-Hit Probe:** Query `chunk_vocab` / `anchor_vocab` (`WHERE term IN (...)` — B-tree `INDEX 1` seek only). For any query token/identifier with `df == 0` and `len >= 4`, run `WHERE term >= ? AND term < ? LIMIT 5` using the 4-char prefix to populate `near_miss_terms`. Check adjacent non-stopword pairs via `chunk_fts MATCH '"w1 w2"'` for super-additive phrase surprisal.
     2. **Bidirectional Wavefront:** Whiten `query_embedding` $\to \hat{q}$. If `embed_fn` is provided and query anchors $E(q)$ exist, micro-batch embed unique anchor strings (caching `(model, anchor) -> blob` in SQLite `entity_vec_cache`) $\to Q_{E(q)}$, and compute target residual $\hat{q}_{\text{target}} = \text{normalize}((I - Q_{E(q)}Q_{E(q)}^\top)\hat{q})$. Score Stage-1 `rabitq_query` (`est - 0.5 * csls_rk`, pruning via `est + eps >= lb_cutoff`) + ordered `FTS5` for $\hat{q}$ (forward wave $F_{\text{fwd}}$) and $\hat{q}_{\text{target}}$ (backward wave $F_{\text{bwd}}$).
     3. **Stage-2 Fetch & Splice:** Fetch exact `embedding` blobs from `chunks WHERE chunk_id IN (...)` **only** for wavefront + anchor-bridge survivors (`<= 192` chunks), whiten via `whiten_normalize`, hydrate disk slices into a request-local `file_cache`, and build candidate chains ($k \in \{1, 2, 3, 4\}$) across shared anchors ($2 \le \text{df}(e) \le \max(32, 0.05 N)$ in `chunk_vocab`/`anchor_vocab`), `F_fwd <-> F_bwd` intersections, and `next_in_file` continuations, weighted by temporal proximity $\exp(-|\Delta\text{mtime}| / \tau)$ when `mtime_ns > 0`.
     4. **Cut & MDL Selection:** Embed unique candidate bridge strings ($\le 24$, cached) $\to \hat{z}_e$, compute `cut_chain_atom` (§2.4), prune extensions where $\hat{q}^\top \psi(p') \le \hat{q}^\top \psi(p) + 0.01$, and run `select_mdl_dag` (§2.4). Populate `stalled_anchors` with any query anchor whose whitened vector still has `dot(residual, z_a) > 0.25` when `explained_r2 < 0.55`.
   - **Implement `expand_neighbors_ranked(vs, conn, chunk_id, *, top_k=6) -> list[tuple[int, float]]`:** Fetch chunk `chunk_id`'s exact vector from `chunks`, whiten, score via `rabitq_query` + AST-island sibling bonus (only when `(ast_island_start, ast_island_end) != (1, 1)` and island span `< file_lines`) + `next_in_file` adjacency + shared rare `anchor_fts` terms.
2. **`yaw/core/config.py` & `yaw/optimize.py`:**
   - Remove `RetrievalParams`, `write_retrieval_section`, `write_manifold_section`, `manifold_overrides`, `retriever_mode`, and retired OpenIE/RLM/GBNF fields from `YawConfig`.
   - Delete `optimize_optuna` from `yaw/optimize.py`. In `optimize_dspy`: add `"insufficient references in corpus"` to `_ABSTAIN_PHRASES` (matching `QASignature` in `yaw/qa.py`), and strip `"tools"` keys from `Inputs` dicts inside `make_react_instruction_proposer` before `json.dumps` truncation.

---

### Checkpoint `MDL-3`: Lazy Tier-0 Runtime, MCP Librarian Surface, CLI, and Docs Sync
**Files touched:** `yaw/runtime.py`, `yaw/mcp_server.py`, `yaw/cli.py`, `yaw/qa.py`, `AGENTS.md`, `FOUNDATIONAL.md`.

1. **`yaw/runtime.py`:**
   - Change `YawRuntime.lm` to `dspy.LM | None = None`. Do **not** call `create_lm(cfg)` inside `start_runtime`; initialize it lazily under `rt.lm_sem` inside `ask()` so `start_runtime` and `recall()` run with **zero generative LLM endpoint configured or reachable**.
   - Update `recall(rt, query, *, top_k=5) -> RecallPacket` to call `solve_evidence_dag(..., embed_fn=rt.embed_fn)` and add `neighbors(rt, chunk_id, *, top_k=6) -> tuple[HydratedPassage, ...]`.
   - Enforce `len(cleaned.encode("utf-8")) <= 1024` inside `_validate_text_input` (`INVALID_INPUT`).
2. **`yaw/mcp_server.py`:**
   - Pass explicit `instructions` to `FastMCP("yaw", instructions=...)` documenting: (a) prefer concurrent zero-LLM `recall` + `neighbors` over serialized `Semaphore(1)` `qa`; (b) interpret `explained_r2`, `edges`, `stalled_anchors`, `near_miss_terms`, and `stale_on_disk`; (c) pass natural-language + exact identifier queries (`<= 1024` UTF-8 bytes).
   - Update `recall` response dict to include `explained_r2`, `net_bits_saved`, `edges`, `stalled_anchors`, `near_miss_terms`, and `results`.
   - Register read-only `@mcp.tool` `neighbors(chunk_id: int, top_k: int = 6)`.
3. **`yaw/cli.py` & `yaw/qa.py`:**
   - Lazy-import `yaw.optimize` inside the `args.command == "optimize"` branch of `yaw/cli.py` (remove top-level `from yaw.optimize import ...`) and remove the `optimize optuna` subparser.
   - Wire `yaw/qa.py`'s `search_passages` and `expand_neighbors` tools to `solve_evidence_dag` and `expand_neighbors_ranked`.
4. **`AGENTS.md` & `FOUNDATIONAL.md`:**
   - Reconcile stale `yaw/daemon/...` paths, retired SKOS/OWL/SHACL/GBNF/`RLM` rules, and `uv run yaw optimize dspy` CLI invocations against the active 14-file architecture.

---

## 4. Mandatory Falsifiable Acceptance Tests (`tests/test_mdl_dag.py`)

Before dispatching `critic (Model="flash")`, add `tests/test_mdl_dag.py` proving all six invariants without external network calls:

1. **`test_ingest_gate_skips_embedder_on_boilerplate_clones`:** Feed 20 log/policy clones differing only in timestamps/UUIDs + 1 unique ADR through `pipeline.py` with a counting mock `embed_fn`. Assert `embed_fn` is invoked **at most twice** (1 prototype + 1 ADR), while all 21 chunks are searchable by unique ID in `anchor_fts`.
2. **`test_ast_islands_and_json_titles_populated`:** Chunk a Python file with two classes, a Markdown file with `## Section`, and a benchmark `.json` doc. Assert `title != ""` on all three and `(ast_island_start, ast_island_end)` bounds each class/section strictly tighter than `(1, total_lines)`.
3. **`test_stage1_memory_footprint_has_no_float32_matrix`:** Build `VectorSearch` over 500 `768d` chunks; assert no attribute on `VectorSearch` holds an `(N, D)` float array and total NumPy array bytes per chunk `< 120` bytes.
4. **`test_hilbert_cut_beats_trivia_and_unlinked_decoy`:** Construct synthetic 3-hop silent-bridge + trivia + unlinked-decoy corpus (`s1: EU -> Munich-Edge`, `s2: Munich-Edge -> Core-Collector`, `s2_trivia: Munich-Edge -> 2021`, `s3: Core-Collector -> port 8443`, `decoy: Postgres -> port 5432`). Assert `solve_evidence_dag` returns topological order `(s1, s2, s3)`, excludes `s2_trivia` and `decoy`, and reports `explained_r2 > 0.80`.
5. **`test_diamond_dag_and_cycle_safety`:** Construct a 2-branch convergent query (`SvcA -> GwA -> Vault` and `SvcB -> GwB -> Vault` plus symmetric anchor sharing `u <-> v`). Assert `solve_evidence_dag` recovers both branches without raising `graphlib.CycleError`.
6. **`test_popqa_single_hop_stops_at_one_node_and_zero_hit_radar_fires`:** Query a 1-hop fact plus a non-existent identifier `webhook_retry_policy` when `webhook_backoff` is indexed. Assert `edges == ()` (1-hop MDL stop) and `"webhook_backoff"` appears in `near_miss_terms["webhook_retry_policy"]`.