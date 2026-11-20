# Probabilistic Graphless Graph Traversal Proposal

Two part post research report and proposal.

---

## Part 1: Post-Mortem on Mock Test Setup vs. Real Mathematical Shortcomings

### 1.1 Four Failure Modes in The Earlier Mock-Embedding Harness
There is a case for calling out the test setup. The synthetic benchmark used for researching this proposal had four artificial pathologies that do **not** occur in real Gemma embeddings:
1. **Setup #1 used i.i.d. random Gaussian vectors** (`rng.standard_normal`), and masked the broken multi-hop math by putting the Hop-3 answer token (`Pohjantähti99` / `tokenHelios77`) inside `query_text` (endpoint juking).
2. **Setup #2 disconnected distractor embeddings from distractor text:** The padding loop injected the exact query predicate tags (`qtags[1:]` at weight `0.82`) into 14 generic boilerplate sentences per question (`"Historical biographical survey k near Bristol..."`) whose text did not even mention those predicates—creating artificial super-decoys with higher query cosine (`0.83`) than any real passage could achieve without matching text.
3. **Orthogonal hash vectors (`blake2b(tag)`) had zero semantic correlation:** In a real embedding model, `"Mozart"` correlates with `"composer"` and `"opera"`, `"Salzburg"` correlates with `"Austria"` and `"federal state"`, and `"born in"` correlates (`cos ~ 0.8+`) with `"birthplace"`. Under random hash tags, $\mathbb{E}[\langle e_{\text{born\_in}}, e_{\text{birthplace}} \rangle] = 0$ unless manually tagged identically.
4. **Degenerate 3-vector background spectrum:** Reusing just three identical tag vectors (`"bg_en"`, `"bg_de"`, `"bg_zh"`) across 180 background chunks collapsed the covariance participation ratio $D_{\text{eff}}$ from `256` down to `9.7–15.0` and shifted the corpus mean `mu` so every non-background chunk had an artificial `+0.50` baseline cosine with every other non-background chunk.

### 1.2 The 6 Genuine Mathematical & Engineering Shortcomings (Independent of Mock Embeddings)
Stripping away those test-harness artifacts leaves **six hard, provable defects** in the original sprint plan (`MDL-2`) that degrade retrieval on real, unjuked English, German, and Chinese corpora:

1. **`RaBitQ` Was Only Used at Hop 0—Never at Hop 1 or Hop 2:**
   - `solve_evidence_dag` called `_rabitq_bnb_top` once at Hop 0, then ran Waves 1 & 2 (`lines 650–707`) purely via lexical `chunk_fts MATCH ... ORDER BY bm25(chunk_fts) ASC LIMIT per_b`.
   - When a bridge entity (`"Mozart"`, `df = 8`) appeared in 8 chunks, `ORDER BY bm25 LIMIT 2` picked the shortest/densest `Mozart` chunks (`Prague`, `Requiem`) and dropped `"Mozart was born in Salzburg"` before `_visit()` ever saw it; and when a hop crossed a paraphrase, acronym, or morphological shift with no shared trigram stem, Waves 1 & 2 were completely blind—despite `vs.rabitq_bits_t` (`~100 B/chunk`) sitting in RAM ready to scan all $N$ chunks in `< 0.5 ms`.
   - **Fix (Step-Wise `RaBitQ` Partial-Correlation Bridge Acquisition):** At each frontier wave, scan `vs.rabitq_bits_t` for top frontier vectors $\hat{u}$ (`est_u`) and compute **spherical partial correlation controlling for $\hat{q}$**:
     $$\rho_{uv \cdot q} = \frac{\hat{c}(u, v) - \hat{c}(u, q)\,\hat{c}(q, v)}{\sqrt{1 - \hat{c}(u, q)^2}\,\sqrt{1 - \hat{c}(q, v)^2}}$$
     Subtracting $\hat{c}(u, q)\hat{c}(q, v)$ removes shared query-predicate similarity (giving unlinked target-predicate decoys like `Wagner` $\rho_{uv \cdot q} \le 0$) while isolating true non-query bridge overlap (`Mozart` $\implies \rho_{uv \cdot q} > 0$), and ranks both dense and FTS5 bridge candidates by joint bridge + query surprisal at every step.

2. **Four Algebraic Defects in `_gibbs_lattice_pass` ("Bayesian" Objective):**
   - **Double-counted Hop-0 potential:** `seed_lp` included `q_nats0 + facet_nats(mask(v_0))`, and stopping weight `g[v_0, m_0]` multiplied by `exp(resid_nats[v_0] + facet_nats(m_0))` **a second time**, squaring 0-hop singleton likelihoods and miscalibrating `log_bayes_factor = log_z - log_z0`.
   - **Terminal-only `node_resid_nats` decay:** `u_term[v_L, m_L]` evaluated semantic query relevance **only at the stopping node $v_L$**, discarding $r(v_0), \dots, r(v_{L-1})$. Whenever Hop-1 ($v_0$) matched the query strongly (`r(v_0) = 8` nats) and Hop-2 ($v_1$) expressed the intermediate relation via paraphrase (`"born in"` vs. `"birthplace"`, so `cand_m == cm` and `r(v_1) = 4` nats), stepping $v_0 \to v_1$ **lost** $4$ nats instead of adding $v_1$'s evidence—causing `w_go <= w_st` to halt prematurely at Hop 0 (`hops = []`).
   - **Row-normalized bridge dilution (`w_cond / row_w`):** Normalizing outgoing weights per row gave a single accidental `df = 24` edge the exact same transition probability (`0.5`) as a `df = 2` smoking-gun entity edge, erasing bridge mutual information.
   - **Same-branch negative-bridge QR deflation:** Because Hop-1 embeds $\hat{v}_0 \approx e_{\text{src}} + e_{\text{bridge}}$, projecting $\hat{v}_0$ out of $\hat{q}$ (`q - (q·v_0)v_0`) injected $-\langle \hat{q}, \hat{v}_0 \rangle e_{\text{bridge}}$ into `q_rem`, actively penalizing Hop-2 chunks containing $+e_{\text{bridge}}$. (QR deflation is mathematically valid strictly for `exclude_chunk_ids` when seeking an orthogonal alternative branch across turns.)

3. **Contentless `FTS5 tokenize='trigram'` Returns `0` Rows on All 2-Character Chinese Words:**
   - Verified directly in SQLite: `SELECT rowid FROM chunk_fts WHERE chunk_fts MATCH '"鲁迅"'` returns `[]` on `content=''` `tokenize='trigram'` because strings `< 3` chars have no trigram tokens and `content=''` has no stored body to scan. Because ~80% of Chinese names, cities, and nouns are 2 Hanzi (`鲁迅`, `绍兴`, `苏轼`, `眉山`, `海思`, `杜甫`), unspaced Chinese failed whenever surrounded by varying particles (`的`, `于`, `在`, `。`).
   - **Fix:** $O(\log V)$ B-tree range seek on `chunk_vocab` (`WHERE term >= bg AND term < bg || '\uffff'`) expanded into bounded `MATCH` over the matching 3-grams, paired with **Church–Hanks Positive PMI** ($N \cdot \text{df}(AB) \ge 2\,\text{df}(A)\,\text{df}(B)$) to reject accidental cross-word 3-grams (`出生地隶`).

4. **Mid-Word Infix Slicing & Space-Gram Blindness in `_watershed_trigram_stems` (`EN` & `DE`):**
   - Character-only watershed sliced mid-word infixes (`arc[hite]ct`, `pub[lish]ed` $\leftrightarrow$ `Eng[lish]`, `disco[vere]d`, `finanz[iert]`), glued rare suffixes onto sibling German compounds (`Nordkanaldienst` `df=2` failed to back off to `Nordkanal` `df=3` for `Nordkanalverwaltung`), dropped `df=1` umlaut stems (`Medizinverbund` $\leftrightarrow$ `Medizinverbünde` sharing 11-char prefix `medizinverb` `df=2`), missed `Schloß` $\to$ `schloss` (`str.casefold()`), and discarded space-spanning trigrams (`g.isalnum()` rejected `' '`, blinding it to `"Red Sea"`, `"Bo Wu"`, `"Seventh Coalition"`, `"Friedrich Kind"`).
   - **Fix:** Left-anchored word stems + `.casefold()` + word-boundary-verified **Koehn–Knight symmetric decompounding** ($\min(\text{df}(w[:a]), \text{df}(w[a:])) > \text{df}(w)$ with `" " + w[a:]` verified in `chunk_fts`) + **Positive-PMI adjacent 2-word phrases** (admitting `"red sea"` and `"bo wu"` while rejecting negative-PMI glue like `"of the"` and `"die kapelle"` with zero stopword lists).

5. **Quadratic Participation Ratio $D_{\text{eff}}$ Collapse & Capped `4 * k_c` BnB:**
   - Boilerplate clusters spike $\lambda_1^2$, collapsing Rényi-2 participation ratio $(\sum \lambda)^2 / \sum \lambda^2$; replacing it with **Shannon Spectral Dimension** $D_{\text{eff}} = \exp(-\sum p_i \ln p_i)$ (Rényi-1 entropy) stabilizes spherical null nats. And replacing the `4 * k_c` BnB cap with **Two-Phase Progressive BnB** restores certified `R@10 = 1.000` with `7.4x` fewer exact fetches than static BnB.

6. **Incomplete `pathlib` Usage & Systems Bugs:**
   - `fnmatch.fnmatch("a/b/c.py", "*.py")` returns `True` (ignoring `/` separators and applying OS-dependent case rules); replaced with `PurePosixPath` (`full_match` on 3.13+ / `match` fallback) and pure `Path` methods (`stat`, `is_file`, `read_text`, `resolve`, `is_relative_to`).
   - Fixed the `PRAGMA query_only = ON` crash on `temp.chunk_vocab` DDL, removed `.yaw/stage1/` sidecar files in favor of single-file SQLite `yaw_stage1`, switched `RaBitQ` to SoA `(W, N)` (`8.1x` faster), cut `VectorSearch` metadata RAM by `52.3x` (`108 B/chunk`), and isolated leaf tasks on `_LEAF_POOL` (deadlock-free).

---

## Part 2: Reauthored Consolidated Sprint Specification (`mdl-dag-sprint` — Rev 3)

### 0. Sequencing, Hard Constraints & Mathematical Architecture

#### 0.1 Entry Gate, Boundaries & Single-File SQLite Contract
- **Entry Gate:** Execute strictly after [bare-bones-replacement-sprint.md](bare-bones-replacement-sprint.md) (`CS-6`) is green. Measure `CS-5` on real `gemma-4-e4b` / `embeddinggemma-2` as the paired `CS-6` control baseline without spending throwaway cycles on `Optuna` PPR sweeps, since `MDL-1` removes `optuna`, `spacy`, `optimize_optuna`, `SearchParams`, and `resolve_retrieval_params`.
- **Module Boundary & Schema Sync:** Preserve the exact 14-file boundary (`yaw/*.py`, `yaw/core/*.py`) with zero new Python modules. Reconcile the realistic `<= 5,500` LOC architecture ceiling established in `CS-4a` during `MDL-2` (trimming or adjusting the cap explicitly without code-golfing), and update [FOUNDATIONAL.md](../../FOUNDATIONAL.md) §4.1 from legacy dual `chunk_fts(unicode61)` + `anchor_fts` to the unified `chunk_fts(tokenize='trigram')` + `chunk_vocab` + `yaw_stage1` schema.
- **Single-File Storage Hygiene (Zero Sidecar Files):**
  - Do **not** create `.yaw/stage1/` or any `.npy`/`meta.json` sidecar files. `.yaw/` holds `yaw.db` (plus native SQLite `-wal`/`-shm` files) only.
  - Persist `chunk_vocab` (`fts5vocab(chunk_fts, 'row')`, `0 B` data overhead) and `yaw_stage1` (single-row ACID SoA `RaBitQ` snapshot) inside `yaw.db` during ingest (`persist_stage1=True`), and run `INSERT INTO chunk_fts(chunk_fts) VALUES('optimize')` on writer commit.
  - Update `list_user_tables(conn)` (from `CS-6 G1`) either to allow `("chunk_fts", "chunk_vocab", "chunks", "yaw_stage1")` or to filter internal `chunk_vocab` / `yaw_stage1` alongside `chunk_fts_*` shadow tables. For unmigrated 2-table DBs, `open_readonly_db` loads `sqlite_vec` and attaches `temp.chunk_vocab` **before** setting `PRAGMA query_only = ON`.
- **Read-Only Purity, Concurrency & API Contract:**
  - Preserve 100% of public signatures in `yaw/retrieval.py` (`ChunkPointer`, `HydratedPassage`, `max_chars`, `include_island`, `full_span`, `format_evidence_route`) and MCP tools (`recall`, `qa`, `daemon_status`), adding optional `exclude_chunk_ids: list[int] = []`, optional `path_glob: str | None = None` (evaluated via `PurePosixPath`), and `trace` on `recall`.
  - Initialize `dspy.LM` lazily inside `ask()` under a lock so Tier-0 `recall` serves with no LLM server running.

#### 0.2 Derived Information-Theoretic Formulation (Zero Free Tunable Hyperparameters; Fixed Structural Bounds)
Let $N = |V_{\text{corpus}}|$, $\hat{x}_i = \frac{x_i - \mu}{\|x_i - \mu\|_2}$, and $D_{\text{eff}} = \exp\!\big(-\sum_j p_j \ln p_j\big)$ where $p_j = \lambda_j / \sum_m \lambda_m$ are normalized sample covariance eigenvalues, with discrete structural bounds `horizon = 3`, wave cap `12`, and clone Jaccard threshold `3/4` (`novelty >= 0.25`):
1. **Structural Scales:** $d_{\text{bridge}} = \max(4, \lceil\sqrt{N}\rceil)$, $d_{\text{query}} = \max(8, \lfloor N / 2 \rfloor)$, $f_{\max} = \min(6, \max(2, \lfloor \frac{1}{2}\log_2 \max(4, N) \rfloor))$, $|C_T| \le 3 d_{\text{bridge}}$.
2. **Multilingual Anchor Extraction (`chunk_vocab` + `chunk_fts`):**
   - **Word-anchored stems & decompounding:** Left-anchored prefixes $w[:L]$ ($L \ge 4$, over $w$ and $w.\text{casefold}()$) where DF steps into $[d_{\min}, d_{\max}]$ or across a suffix seam, plus symmetric Koehn–Knight splits $(w[:a], w[a:])$ ($a \ge 4, |w|-a \ge 4$) verified via $\min(\text{df}(w[:a]), \text{df}(w[a:])) > \text{df}(w)$ and word-start `" " + w[a:]` in `chunk_fts`.
   - **2-word collocations:** Adjacent words `"w1 w2"` ($|w_1|, |w_2| \ge 2$) satisfying Church–Hanks Positive PMI: $N \cdot \text{df}(w_1 w_2) \ge 2 \cdot \text{df}(w_1)\cdot\text{df}(w_2)$.
   - **Unspaced CJK 2-char & 3/4-char anchors:** 2-char bigrams resolved via `chunk_vocab` B-tree range `[bg, bg + '\uffff')` $\to$ `MATCH` OR-expansion; 3/4-char spans filtered by Positive PMI against straddling bigrams.
3. **Stage-1 SoA `RaBitQ`, Two-Phase Progressive BnB & Step-Wise Partial Correlation:**
   - Unbiased cosine estimator $\hat{c}(u, v)$ with certified half-width $\varepsilon_i = \frac{\sqrt{1-s_i^2}}{s_i}\sqrt{\frac{2\ln\max(2, N)}{D - 1}}$.
   - At each frontier wave $w \in \{1, 2\}$, `RaBitQ` scans top frontier vectors $\hat{u}$ to compute partial bridge correlation $\rho_{uv \cdot q}$, admitting certified semantic bridges where $I_{\text{null}}(\rho_{uv \cdot q}; D_{\text{eff}} - 1) > \ln(N + 1)$ alongside FTS5 lexical bridges ranked by joint `RaBitQ` bridge + query surprisal.
4. **Additive Coverage-Lattice Stopped Gibbs Measure (in nats, $\tau \equiv 1.0$) & Nat-Weighted Completeness:**
   For any non-clone simple path $P = (v_0 \to \dots \to v_L)$ ($0 \le L \le 3$) with final union facet mask $m_L = \bigcup_{t=0}^L \text{mask}(v_t)$:
   $$\log W(P) = \underbrace{\sum_{b \in m_L} I(f_b)}_{\text{Union lexical nats (1x)}} + \underbrace{r(v_0) + \sum_{t=0}^{L-1} \big(1 - \max(0, \langle \hat{v}_t, \hat{v}_{t+1}\rangle)^2\big)\,r(v_{t+1})}_{\text{Additive non-redundant semantic nats}} - \underbrace{\left(\ln(N+1) + \sum_{t=0}^{L-1} c_{\text{hop}}(v_t \to v_{t+1}) - \ln\rho_{\text{stop}}(v_L)\right)}_{\text{Shannon path description length + Laplace stop prior}}$$
   where $I(s) = \ln\frac{N+2}{\text{df}(s)+1}$, $r(v) = \min\!\big(-\ln \frac{1}{2}I_{1-\langle\hat{v},\hat{q}\rangle^2}(\frac{D_{\text{eff}}-1}{2}, \frac{1}{2}),\, \ln(N+2)\big)$, $\rho_{\text{stop}}(v) = \frac{1}{1 + d_{\text{out}}(v)}$, and $c_{\text{hop}}(u \to v) = \ln(\text{df}(b_{uv}) + 1) / \text{nov}(u, v)$ for lexical bridges or $\ln\!\big(2 + (N+1)p_{\text{null}}(\rho_{uv\cdot q})\big)$ for pure semantic bridges (strictly $\in (\ln 2, \ln 3]$ upon Bonferroni admission). Route completeness puts both lexical facet surprisal and NNLS cone projection norm $r_{\text{cone}} = \min\!\big(I_{\text{null}}(\|\mathbf{V}_{\text{comp}}\mathbf{w}^+\|_2; D_{\text{eff}}),\, \ln(N+2)\big)$ on the same nat scale: $\frac{\sum_{b \in m(C)} I(f_b) + r_{\text{cone}}}{\sum_b I(f_b) + \ln(N+2)}$.

---

### 1. Task Breakdown & Copy-Paste Ready Production Code

#### `MDL-1`: Dependencies, Single-File SQLite Schema & Type Contracts
1. **`pyproject.toml`, `yaw/optimize.py` & `FOUNDATIONAL.md`:** Remove `spacy`, `en-core-web-sm` (`[tool.uv.sources]`), and `optuna` from `pyproject.toml`; delete `optimize_optuna` from `yaw/optimize.py`, delete `SearchParams` / `resolve_retrieval_params` and update `yaw/qa.py` / `yaw/runtime.py`; and update [FOUNDATIONAL.md](../../FOUNDATIONAL.md) §4.1 from dual `chunk_fts` + `anchor_fts` to the single `chunk_fts(tokenize='trigram')` + `chunk_vocab` + `yaw_stage1` schema.
2. **`yaw/pipeline.py` (Inside CocoIndex Managed Transactions — Zero Bare `conn.commit()`):**
   - In `_apply_fts_tables` (`conn_mgr.transaction()`), configure `chunk_fts` with `tokenize='trigram'`, create `main.chunk_vocab`, and run FTS5 segment optimization:
     ```python
     conn.execute(
         """
         CREATE VIRTUAL TABLE IF NOT EXISTS "chunk_fts"
         USING fts5(title, body, content='', contentless_delete=1, tokenize='trigram')
         """
     )
     conn.execute(
         """
         CREATE VIRTUAL TABLE IF NOT EXISTS "chunk_vocab"
         USING fts5vocab(chunk_fts, 'row')
         """
     )
     conn.execute("INSERT INTO chunk_fts(chunk_fts) VALUES('optimize')")
     ```
   - Never issue a bare `conn.commit()` inside CocoIndex's `ManagedConnection` transaction; persist `yaw_stage1` either inside the CocoIndex post-update managed transaction (`conn.in_transaction == True`) or via a short write session on `open_managed_db` after `app.update()` completes:
     ```python
     build_vector_search(conn, corpus_root=cfg.corpus_root, persist_stage1=True)
     ```
3. **`yaw/core/types.py` & `yaw/core/__init__.py`:** Add and export `EvidenceHop` and `SearchTrace` without altering `ChunkPointer` or `HydratedPassage`:
   ```python
   @dataclass(frozen=True, slots=True)
   class EvidenceHop:
       src_chunk_id: int
       dst_chunk_id: int
       anchor: str
       weight: float


   @dataclass(frozen=True, slots=True)
   class SearchTrace:
       hops: tuple[EvidenceHop, ...] = ()
       route_completeness: float = 0.0
       log_bayes_factor: float = 0.0
       unexplained_facets: tuple[str, ...] = ()
       dead_end_bridges: tuple[str, ...] = ()
   ```

---

#### `MDL-2`: Complete Drop-In `yaw/retrieval.py`

```python
"""Multilingual FTS5-Watershed, Step-Wise SoA RaBitQ & Additive Coverage-Lattice MDL Retrieval."""

from __future__ import annotations

from collections.abc import Sequence
import concurrent.futures
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
import os
from pathlib import Path, PurePosixPath
import re
import sqlite3
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import nnls
from scipy.special import betainc
import sqlite_vec

from yaw.core.types import (
    ChunkPointer,
    EvidenceHop,
    HydratedPassage,
    SearchTrace,
    YawRuntimeError,
    resolve_corpus_path,
)

_NO_ISLAND = (1, 1)
_MAX_HYDRATE_SPAN = 10_000
_CJK_CH_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]")
_CJK_RUN_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]{2,}")
_WORD_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_WORKERS = min(4, max(1, os.cpu_count() or 1))
# Strictly leaf-only thread pool: tasks running on _LEAF_POOL must NEVER submit child tasks to _LEAF_POOL.
_LEAF_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=_WORKERS, thread_name_prefix="yaw-leaf")
_STAGE1_DDL = (
    "CREATE TABLE IF NOT EXISTS yaw_stage1 ("
    "id INTEGER PRIMARY KEY CHECK (id = 1), "
    "n INTEGER NOT NULL, "
    "dim INTEGER NOT NULL, "
    "d_eff REAL NOT NULL, "
    "fingerprint TEXT NOT NULL, "
    "mu BLOB NOT NULL, "
    "bits_t BLOB NOT NULL, "
    "s BLOB NOT NULL, "
    "ones BLOB NOT NULL)"
)


@dataclass(frozen=True, slots=True)
class VectorSearch:
    """~108 B/chunk SoA Stage-1 RaBitQ view (zero sidecar files; zero float32[N,D] or ChunkPointer[N] in RAM)."""

    chunk_ids_arr: NDArray[np.int64]
    dim: int
    mu: NDArray[np.float32]
    d_eff: float
    rot: NDArray[np.float32]
    rabitq_bits_t: NDArray[np.uint64]  # SoA shape (W, N) for unit-stride SIMD popcount
    rabitq_s: NDArray[np.float16]
    rabitq_ones: NDArray[np.uint16]
    corpus_root: Path | None = None
    db_path: Path | None = None
    _mem_pointers: dict[int, ChunkPointer] | None = None

    @property
    def size(self) -> int:
        return int(self.chunk_ids_arr.size)

    @property
    def rabitq_bits(self) -> NDArray[np.uint64]:
        """Zero-copy (N, W) view preserving original attribute contract."""
        return self.rabitq_bits_t.T

    @property
    def chunk_ids(self) -> tuple[int, ...]:
        return tuple(self.chunk_ids_arr.tolist())

    @property
    def id_to_pos(self) -> dict[int, int]:
        return {int(c): i for i, c in enumerate(self.chunk_ids_arr.tolist())}

    @property
    def pointers(self) -> tuple[ChunkPointer, ...]:
        if self._mem_pointers is not None:
            return tuple(self._mem_pointers[int(c)] for c in self.chunk_ids_arr.tolist())
        if self.db_path is None:
            return ()
        conn = open_readonly_db(self.db_path)
        try:
            rows = conn.execute(
                "SELECT chunk_id, rel_path, source_hash, start_line, end_line, "
                "char_start, char_end, ast_island_start, ast_island_end, title "
                "FROM chunks ORDER BY chunk_id ASC"
            ).fetchall()
            return tuple(
                ChunkPointer(
                    int(r[0]), str(r[1]), str(r[2]), int(r[3]), int(r[4]),
                    int(r[5]), int(r[6]), int(r[7]), int(r[8]), str(r[9]),
                )
                for r in rows
            )
        finally:
            conn.close()

    def pos_of(self, cid: int) -> int | None:
        n = self.chunk_ids_arr.size
        if n == 0:
            return None
        idx = int(np.searchsorted(self.chunk_ids_arr, cid))
        return idx if idx < n and int(self.chunk_ids_arr[idx]) == cid else None


def open_readonly_db(db_path: Path | str) -> sqlite3.Connection:
    resolved = Path(db_path).resolve()
    if not resolved.is_file():
        raise YawRuntimeError("INDEX_NOT_READY", f"Database not found at {resolved}")
    conn = sqlite3.connect(
        f"file:{resolved.as_posix()}?mode=ro",
        uri=True,
        timeout=5.0,
        check_same_thread=False,
    )
    try:
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.execute("PRAGMA temp_store = MEMORY")
        conn.execute("PRAGMA mmap_size = 268435456")
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        if (
            conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunk_vocab'").fetchone()
            is None
        ):
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS temp.chunk_vocab "
                "USING fts5vocab(main, 'chunk_fts', 'row')"
            )
        conn.execute("PRAGMA query_only = ON")
    except Exception:
        conn.close()
        raise
    return conn


def _resolve_vocab_table(conn: sqlite3.Connection) -> str:
    if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunk_vocab'").fetchone():
        return "main.chunk_vocab"
    if conn.execute("SELECT 1 FROM sqlite_temp_master WHERE type='table' AND name='chunk_vocab'").fetchone():
        return "temp.chunk_vocab"
    conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS temp.chunk_vocab USING fts5vocab(main, 'chunk_fts', 'row')")
    return "temp.chunk_vocab"


def extract_text_features(title: str, body: str) -> tuple[str, str]:
    """Compatibility stub until auxiliary JSON columns are dropped."""
    del title, body
    return "{}", "[]"


def _fts_quote(term: str) -> str:
    return '"' + term.replace('"', '""') + '"'


def _posix_glob_match(rel_posix_path: str, pattern: str) -> bool:
    """Path-separator-aware POSIX glob matching via pathlib.PurePosixPath."""
    pp = PurePosixPath(rel_posix_path)
    full_match_fn = getattr(pp, "full_match", None)
    if full_match_fn is not None:
        return bool(full_match_fn(pattern))
    if "/" not in pattern and "**" not in pattern:
        return len(pp.parts) == 1 and pp.match(pattern)
    return pp.match(pattern)


def _extract_grams(s: str, *, include_space_grams: bool = True) -> list[str]:
    if len(s) < 3:
        return []
    out: list[str] = []
    for i in range(len(s) - 2):
        g = s[i : i + 3]
        if g.isalnum():
            out.append(g)
        elif include_space_grams:
            c0, c1, c2 = g[0], g[1], g[2]
            if (
                (c0 == " " and c1.isalnum() and c2.isalnum())
                or (c0.isalnum() and c1 == " " and c2.isalnum())
                or (c0.isalnum() and c1.isalnum() and c2 == " ")
            ):
                out.append(g)
    return out


def _center_normalize(vecs: NDArray[np.float32], mu: NDArray[np.float32]) -> NDArray[np.float32]:
    single = vecs.ndim == 1
    x = (vecs[None, :] if single else vecs).astype(np.float32) - mu
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return x[0] if single else x


def _null_nats(cos_val: NDArray[Any] | float, d_eff: float) -> NDArray[np.float64]:
    c = np.asarray(cos_val, dtype=np.float64)
    pos = c > 0.0
    x = np.where(pos, np.maximum(1e-15, 1.0 - np.minimum(c * c, 0.999999)), 1.0)
    p = 0.5 * betainc(0.5 * max(2.0, d_eff - 1.0), 0.5, x)
    return np.where(pos, -np.log(np.maximum(p, 1e-300)), 0.0)


def _db_file_path(conn: sqlite3.Connection) -> Path | None:
    for _, name, file_p in conn.execute("PRAGMA database_list").fetchall():
        if name == "main" and file_p:
            return Path(file_p).resolve()
    return None


def _make_rotation(dim: int, seed: int = 0x594157) -> NDArray[np.float32]:
    if dim == 0:
        return np.zeros((0, 0), dtype=np.float32)
    q_rot, r_rot = np.linalg.qr(np.random.default_rng(seed + dim).standard_normal((dim, dim)).astype(np.float32))
    return (q_rot * np.sign(np.diag(r_rot))).astype(np.float32, copy=False)


@lru_cache(maxsize=256)
def _cached_file_text(resolved_path: Path, mtime_ns: int, size_bytes: int) -> tuple[str, tuple[str, ...]]:
    del mtime_ns, size_bytes
    norm_text = resolved_path.read_text("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    return norm_text, tuple(norm_text.splitlines())


def _read_verified_text(full_path: Path) -> tuple[str, tuple[str, ...]] | None:
    try:
        if not full_path.is_file():
            return None
        st = full_path.stat()
    except OSError:
        return None
    return _cached_file_text(full_path, st.st_mtime_ns, st.st_size)


def _corpus_ids_and_fingerprint(conn: sqlite3.Connection) -> tuple[NDArray[np.int64], str]:
    h = hashlib.sha256()
    cids: list[int] = []
    for cid, sh in conn.execute("SELECT chunk_id, source_hash FROM chunks ORDER BY chunk_id ASC"):
        cids.append(int(cid))
        h.update(f"{cid}:{sh}\n".encode("ascii"))
    return np.array(cids, dtype=np.int64), h.hexdigest()


def _fetch_pointers(
    vs: VectorSearch, conn: sqlite3.Connection | None, cids: Sequence[int]
) -> dict[int, ChunkPointer]:
    if not cids:
        return {}
    if vs._mem_pointers is not None:
        return {c: vs._mem_pointers[c] for c in cids if c in vs._mem_pointers}
    own = conn is None
    if own:
        if vs.db_path is None:
            return {}
        conn = open_readonly_db(vs.db_path)
    try:
        uniq = list(dict.fromkeys(int(c) for c in cids))
        out: dict[int, ChunkPointer] = {}
        for k0 in range(0, len(uniq), 400):
            batch = uniq[k0 : k0 + 400]
            rows = conn.execute(
                "SELECT chunk_id, rel_path, source_hash, start_line, end_line, "
                "char_start, char_end, ast_island_start, ast_island_end, title "
                f"FROM chunks WHERE chunk_id IN ({','.join('?' * len(batch))})",
                batch,
            ).fetchall()
            for r in rows:
                out[int(r[0])] = ChunkPointer(
                    int(r[0]), str(r[1]), str(r[2]), int(r[3]), int(r[4]),
                    int(r[5]), int(r[6]), int(r[7]), int(r[8]), str(r[9]),
                )
        return out
    finally:
        if own and conn is not None:
            conn.close()


def _decode_embeddings(blobs: Sequence[bytes], dim: int) -> NDArray[np.float32]:
    if not blobs or dim == 0:
        return np.zeros((0, dim), dtype=np.float32)
    dtype = np.dtype("<f2") if len(blobs[0]) == dim * 2 else np.dtype("<f4")
    return np.vstack([np.frombuffer(b, dtype=dtype).astype(np.float32, copy=False) for b in blobs])


def build_vector_search(
    conn: sqlite3.Connection,
    corpus_root: Path | None = None,
    *,
    persist_stage1: bool = False,
) -> VectorSearch:
    db_file = _db_file_path(conn)
    root = corpus_root or (db_file.parent.parent if db_file and db_file.parent.name == ".yaw" else None)
    cids_arr, fp_hex = _corpus_ids_and_fingerprint(conn)
    n = int(cids_arr.size)

    if n > 0 and not persist_stage1:
        has_s1 = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='yaw_stage1'"
        ).fetchone()
        if has_s1 is not None:
            row = conn.execute(
                "SELECT n, dim, d_eff, fingerprint, mu, bits_t, s, ones FROM yaw_stage1 WHERE id = 1"
            ).fetchone()
            if row is not None and int(row[0]) == n and row[3] == fp_hex:
                dim = int(row[1])
                n_w = ((dim + 63) // 64) if dim > 0 else 0
                return VectorSearch(
                    cids_arr,
                    dim,
                    np.frombuffer(row[4], dtype="<f4"),
                    float(row[2]),
                    _make_rotation(dim),
                    np.frombuffer(row[5], dtype="<u8").reshape(n_w, n),
                    np.frombuffer(row[6], dtype="<f2"),
                    np.frombuffer(row[7], dtype="<u2"),
                    root,
                    db_file,
                )

    mem_ptrs: dict[int, ChunkPointer] | None = None
    if db_file is None and n > 0:
        ptr_rows = conn.execute(
            "SELECT chunk_id, rel_path, source_hash, start_line, end_line, "
            "char_start, char_end, ast_island_start, ast_island_end, title "
            "FROM chunks ORDER BY chunk_id ASC"
        ).fetchall()
        mem_ptrs = {
            int(r[0]): ChunkPointer(
                int(r[0]), str(r[1]), str(r[2]), int(r[3]), int(r[4]),
                int(r[5]), int(r[6]), int(r[7]), int(r[8]), str(r[9]),
            )
            for r in ptr_rows
        }

    if n == 0:
        return VectorSearch(
            np.zeros(0, dtype=np.int64),
            0,
            np.zeros(0, dtype=np.float32),
            8.0,
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0, 0), dtype=np.uint64),
            np.zeros(0, dtype=np.float16),
            np.zeros(0, dtype=np.uint16),
            root,
            db_file,
            mem_ptrs,
        )

    vec_rows = conn.execute("SELECT embedding FROM chunks ORDER BY chunk_id ASC").fetchall()
    dim = int(np.frombuffer(vec_rows[0][0], dtype="<f4").shape[0])
    raw = np.empty((n, dim), dtype=np.float32)
    for i, (blob,) in enumerate(vec_rows):
        raw[i] = np.frombuffer(blob, dtype="<f4")

    if n < 8 or dim == 0:
        mu = np.zeros(dim, dtype=np.float32)
        d_eff = float(max(8, dim))
    else:
        samp = raw[np.random.default_rng(0x594157).choice(n, size=min(n, 2048), replace=False)]
        mu = samp.mean(axis=0, dtype=np.float64).astype(np.float32)
        _, sv, _ = np.linalg.svd((samp - mu) / math.sqrt(float(len(samp) - 1)), full_matrices=False)
        lam = np.maximum(sv.astype(np.float64) ** 2, 1e-12)
        p_lam = lam / float(lam.sum())
        # Shannon spectral entropy dimension (Renyi alpha=1; robust to single-cluster boilerplate spikes).
        d_eff = float(np.clip(np.exp(-float(np.sum(p_lam * np.log(p_lam)))), 8.0, float(dim)))

    iso = _center_normalize(raw, mu)
    rot = _make_rotation(dim)
    x_rot = iso @ rot
    rb_s = np.maximum(
        (np.sum(np.abs(x_rot), axis=1) / math.sqrt(float(max(1, dim)))).astype(np.float16),
        np.float16(1e-4),
    )
    b_u8 = np.packbits(x_rot >= 0.0, axis=1, bitorder="little")
    if rem := (b_u8.shape[1] % 8):
        b_u8 = np.pad(b_u8, ((0, 0), (0, 8 - rem)))
    rb_bits = np.ascontiguousarray(b_u8).view(np.uint64)
    rb_ones = np.bitwise_count(rb_bits).sum(axis=1, dtype=np.uint16)
    rb_bits_t = np.ascontiguousarray(rb_bits.T)  # SoA (W, N)
    del raw, iso, x_rot, rb_bits

    if persist_stage1:
        args_s1 = (n, dim, d_eff, fp_hex, mu.tobytes(), rb_bits_t.tobytes(), rb_s.tobytes(), rb_ones.tobytes())
        sql_s1 = (
            "INSERT OR REPLACE INTO yaw_stage1("
            "id, n, dim, d_eff, fingerprint, mu, bits_t, s, ones"
            ") VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        if conn.in_transaction:
            conn.execute(_STAGE1_DDL)
            conn.execute(sql_s1, args_s1)
        else:
            with conn:
                conn.execute(_STAGE1_DDL)
                conn.execute(sql_s1, args_s1)

    return VectorSearch(cids_arr, dim, mu, d_eff, rot, rb_bits_t, rb_s, rb_ones, root, db_file, mem_ptrs)


def _rabitq_start_async(
    vs: VectorSearch, q_unit: NDArray[np.float32]
) -> tuple[float, float, float, NDArray[Any], list[concurrent.futures.Future[None]]]:
    qr = q_unit @ vs.rot
    qmin, qmax = float(qr.min()), float(qr.max())
    spn = max(qmax - qmin, 1e-8)
    qu4 = np.clip(np.round((qr - qmin) * (15.0 / spn)), 0, 15).astype(np.uint8)
    n_w = vs.rabitq_bits_t.shape[0]
    pu64s = np.empty((4, n_w), dtype=np.uint64)
    for pl in range(4):
        pu8 = np.packbits((qu4 >> pl) & 1, bitorder="little")
        if rem := (len(pu8) % 8):
            pu8 = np.pad(pu8, (0, 8 - rem))
        pu64s[pl] = pu8.view(np.uint64)

    n = vs.size
    acc_dt = np.uint16 if vs.dim <= 4096 else np.int32
    dot_bq = np.empty(n, dtype=acc_dt)

    def _scan_slice(lo: int, hi: int) -> None:
        sl = vs.rabitq_bits_t[:, lo:hi]
        m = hi - lo
        tmp = np.empty((n_w, m), dtype=np.uint64)
        cnt = np.empty((n_w, m), dtype=np.uint8)
        acc = np.zeros(m, dtype=acc_dt)
        for pl in range(4):
            np.bitwise_and(sl, pu64s[pl][:, None], out=tmp)
            np.bitwise_count(tmp, out=cnt)
            s = cnt.sum(axis=0, dtype=acc_dt)
            if pl:
                s <<= pl
            acc += s
        dot_bq[lo:hi] = acc

    if n >= 4096 and _WORKERS > 1:
        step = (n + _WORKERS - 1) // _WORKERS
        futs = [_LEAF_POOL.submit(_scan_slice, lo, min(n, lo + step)) for lo in range(0, n, step)]
    else:
        futs = [_LEAF_POOL.submit(_scan_slice, 0, n)]
    return qmin, spn, float(qr.sum()), dot_bq, futs


def _rabitq_finish(
    vs: VectorSearch,
    state: tuple[float, float, float, NDArray[Any], list[concurrent.futures.Future[None]]],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    qmin, spn, qr_sum, dot_bq, futs = state
    for fut in futs:
        fut.result()
    n = vs.size
    sbq = qmin * vs.rabitq_ones.astype(np.float32) + (spn / 15.0) * dot_bq.astype(np.float32)
    sf = vs.rabitq_s.astype(np.float32)
    est = (((2.0 * sbq - qr_sum) / math.sqrt(float(max(1, vs.dim)))) / sf).astype(np.float32)
    z_n = math.sqrt(2.0 * math.log(max(2, n)))
    eps = ((np.sqrt(np.maximum(0.0, 1.0 - sf * sf)) / sf) * (z_n / math.sqrt(max(1, vs.dim - 1)))).astype(np.float32)
    return est, eps


def _rabitq_estimate(
    vs: VectorSearch, q_unit: NDArray[np.float32]
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    return _rabitq_finish(vs, _rabitq_start_async(vs, q_unit))


def _rabitq_bnb_top(
    vs: VectorSearch,
    conn: sqlite3.Connection,
    q_unit: NDArray[np.float32],
    k_want: int,
    allowed_mask: NDArray[np.bool_] | None = None,
    precomputed: tuple[NDArray[np.float32], NDArray[np.float32]] | None = None,
) -> list[int]:
    """Certified Two-Phase Progressive Branch-and-Bound over RaBitQ's [est - eps, est + eps] bounds."""
    est, eps = precomputed if precomputed is not None else _rabitq_estimate(vs, q_unit)
    if allowed_mask is not None:
        est = np.where(allowed_mask, est, -2.0)
        eps = np.where(allowed_mask, eps, 0.0)
        n_ok = int(allowed_mask.sum())
        if n_ok == 0:
            return []
        k_c = min(n_ok, max(1, k_want))
    else:
        k_c = min(vs.size, max(1, k_want))

    lb = float(np.partition(est - eps, -k_c)[-k_c])
    ub = est + eps
    surv = np.flatnonzero(ub >= lb)
    if allowed_mask is not None:
        surv = surv[allowed_mask[surv]]
    if len(surv) == 0:
        return []

    batch_sz = max(64, k_c * 2)
    best: list[tuple[float, int]] = []

    def _eval_batch(pos_slice: NDArray[np.intp]) -> float:
        cids_b = vs.chunk_ids_arr[pos_slice].tolist()
        rows = conn.execute(
            f"SELECT chunk_id, embedding FROM chunks WHERE chunk_id IN ({','.join('?' * len(cids_b))})",
            cids_b,
        ).fetchall()
        if rows:
            vecs = _center_normalize(_decode_embeddings([r[1] for r in rows], vs.dim), vs.mu)
            for sc, r in zip((vecs @ q_unit).tolist(), rows, strict=True):
                best.append((float(sc), int(r[0])))
            best.sort(key=lambda x: -x[0])
            if len(best) > k_c:
                del best[k_c:]
        return best[-1][0] if len(best) == k_c else -2.0

    if len(surv) <= batch_sz:
        _eval_batch(surv)
        return [cid for _, cid in best]

    part = np.argpartition(-est[surv], batch_sz - 1)
    tau_k = _eval_batch(surv[part[:batch_sz]])
    rem = surv[part[batch_sz:]]
    rem = rem[ub[rem] >= tau_k]

    if len(rem) > 0:
        rem_ub = rem[np.argsort(-ub[rem])]
        for b0 in range(0, len(rem_ub), batch_sz):
            if len(best) == k_c and float(ub[rem_ub[b0]]) < tau_k:
                break
            tau_k = _eval_batch(rem_ub[b0 : b0 + batch_sz])

    return [cid for _, cid in best]


def _allowed_mask_for_glob(
    vs: VectorSearch, conn: sqlite3.Connection, path_glob: str | None
) -> NDArray[np.bool_] | None:
    if not path_glob:
        return None
    distinct_paths = [str(r[0]) for r in conn.execute("SELECT DISTINCT rel_path FROM chunks").fetchall()]
    matched_paths = [p for p in distinct_paths if _posix_glob_match(p, path_glob)]
    mask = np.zeros(vs.size, dtype=np.bool_)
    if not matched_paths:
        return mask
    matched_cids: list[int] = []
    for k0 in range(0, len(matched_paths), 400):
        batch = matched_paths[k0 : k0 + 400]
        matched_cids.extend(
            int(r[0])
            for r in conn.execute(
                f"SELECT chunk_id FROM chunks WHERE rel_path IN ({','.join('?' * len(batch))})", batch
            ).fetchall()
        )
    if matched_cids:
        arr = np.array(matched_cids, dtype=np.int64)
        idx = np.searchsorted(vs.chunk_ids_arr, arr)
        ok = (idx < vs.size) & (vs.chunk_ids_arr[np.minimum(idx, vs.size - 1)] == arr)
        mask[idx[ok]] = True
    return mask


def _anchor_matches_text(anchor: str, text_cf: str) -> bool:
    if _CJK_CH_RE.search(anchor):
        return anchor in text_cf
    if len(anchor) >= 5 and " " not in anchor:
        return anchor in text_cf
    idx = text_cf.find(anchor)
    while idx != -1:
        if idx == 0 or not text_cf[idx - 1].isalnum():
            return True
        idx = text_cf.find(anchor, idx + 1)
    return False


def _lookup_postings(
    sub: str,
    conn: sqlite3.Connection,
    vocab_tbl: str,
    cap_df: int,
    postings_cache: dict[str, tuple[int, tuple[int, ...]]],
) -> tuple[int, tuple[int, ...]]:
    """Bounded single-query postings + DF lookup (resolves 2-char CJK via fts5vocab B-tree prefix seek)."""
    if sub in postings_cache:
        d_prev, r_prev = postings_cache[sub]
        if r_prev or d_prev == 0 or d_prev > cap_df:
            return d_prev, r_prev

    if len(sub) == 2:
        if not (_CJK_CH_RE.match(sub[0]) and _CJK_CH_RE.match(sub[1])):
            postings_cache[sub] = (0, ())
            return (0, ())
        vrows = conn.execute(
            f"SELECT term, doc FROM {vocab_tbl} WHERE term >= ? AND term < ? LIMIT 64",
            (sub, sub + "\uffff"),
        ).fetchall()
        tot_est = sum(int(r[1]) for r in vrows)
        if not vrows or tot_est > cap_df * 3:
            res = (tot_est if tot_est > cap_df * 3 else 0, ())
            postings_cache[sub] = res
            return res
        q_or = " OR ".join(_fts_quote(str(r[0])) for r in vrows)
        rids = tuple(
            int(r[0])
            for r in conn.execute(
                "SELECT rowid FROM chunk_fts WHERE chunk_fts MATCH ? LIMIT ?",
                (q_or, cap_df + 1),
            ).fetchall()
        )
        res = (len(rids), rids if len(rids) <= cap_df else ())
        postings_cache[sub] = res
        return res

    rids = tuple(
        int(r[0])
        for r in conn.execute(
            "SELECT rowid FROM chunk_fts WHERE chunk_fts MATCH ? LIMIT ?",
            (_fts_quote(sub), cap_df + 1),
        ).fetchall()
    )
    res = (len(rids), rids if len(rids) <= cap_df else ())
    postings_cache[sub] = res
    return res


def _extract_multilingual_anchors(
    text_raw: str,
    gdf: dict[str, int],
    conn: sqlite3.Connection,
    vocab_tbl: str,
    n_tot: int,
    min_df: int,
    max_df: int,
    limit: int,
    postings_cache: dict[str, tuple[int, tuple[int, ...]]],
    anchor_positions: list[int] | None = None,
    *,
    dedup_query_spans: bool = False,
) -> list[tuple[str, float, int]]:
    """Word-boundary stems, Koehn-Knight decompounding, Positive-PMI 2-word phrases & 2-char CJK anchors."""
    text_low = text_raw.lower()
    if not text_low:
        return []

    def _prox(pos: int) -> float:
        return math.log(math.e + min(abs(pos - ap) for ap in anchor_positions)) if anchor_positions else 1.0

    raw_cands: list[tuple[int, int, str]] = []
    seen_s: set[str] = set()

    def _propose(sub: str, est_df: int, pos: int) -> None:
        if sub in seen_s or est_df < min_df or est_df > max_df:
            return
        seen_s.add(sub)
        raw_cands.append((pos, pos + len(sub), sub))

    # 1. Unspaced CJK runs: 2-char B-tree prefix bigrams + Positive-PMI 3/4-char spans.
    for m_cjk in _CJK_RUN_RE.finditer(text_low):
        run = m_cjk.group(0)
        r0 = m_cjk.start()
        rlen = len(run)
        bg_dfs: list[int] = []
        for i in range(rlen - 1):
            d_bg, _ = _lookup_postings(run[i : i + 2], conn, vocab_tbl, max(max_df, n_tot // 2), postings_cache)
            bg_dfs.append(d_bg)
            if min_df <= d_bg <= max_df:
                _propose(run[i : i + 2], d_bg, r0 + i)
        for i in range(rlen - 2):
            g = run[i : i + 3]
            d = gdf.get(g, 0)
            if min_df <= d <= max_df:
                if len(bg_dfs) >= i + 2 and bg_dfs[i] > 0 and bg_dfs[i + 1] > 0 and (n_tot * d) < (2 * bg_dfs[i] * bg_dfs[i + 1]):
                    continue
                if i > 0 and len(bg_dfs) >= i + 2 and bg_dfs[i - 1] > 0 and bg_dfs[i + 1] > 0 and (n_tot * d) < (2 * bg_dfs[i - 1] * bg_dfs[i + 1]):
                    continue
                if i + 3 < rlen:
                    g2 = run[i + 1 : i + 4]
                    d2 = gdf.get(g2, 0)
                    if min_df <= d2 <= 2 * (d + 1) - 1 and d2 <= max_df:
                        bad4 = (
                            len(bg_dfs) >= i + 3 and bg_dfs[i] > 0 and bg_dfs[i + 2] > 0 and (n_tot * d2) < (2 * bg_dfs[i] * bg_dfs[i + 2])
                        ) or (
                            len(bg_dfs) >= i + 4 and bg_dfs[i + 1] > 0 and bg_dfs[i + 3] > 0 and (n_tot * d2) < (2 * bg_dfs[i + 1] * bg_dfs[i + 3])
                        )
                        if not bad4:
                            _propose(run[i : i + 4], min(d, d2), r0 + i)
                _propose(g, d, r0 + i)

    # 2. Non-CJK words: left-anchored stems, Unicode casefold (ß -> ss), and Koehn-Knight decompounding.
    word_spans: list[tuple[int, int, str]] = [
        (m_w.start(), m_w.end(), m_w.group(0))
        for m_w in _WORD_TOKEN_RE.finditer(text_low)
        if not _CJK_CH_RE.search(m_w.group(0))
    ]
    extra_grams: list[str] = []
    for _, _, w in word_spans:
        w_cf = w.casefold()
        if w_cf != w and len(w_cf) >= 3:
            extra_grams.extend(w_cf[j : j + 3] for j in range(len(w_cf) - 2) if w_cf[j : j + 3] not in gdf)
        if len(w) == 2:
            for g_b in (f" {w}", f"{w} "):
                if g_b not in gdf:
                    extra_grams.append(g_b)
        elif len(w) == 3 and w not in gdf:
            extra_grams.append(w)
    if extra_grams:
        uq = list(dict.fromkeys(extra_grams))
        for k0 in range(0, len(uq), 400):
            batch = uq[k0 : k0 + 400]
            gdf.update(
                conn.execute(f"SELECT term, doc FROM {vocab_tbl} WHERE term IN ({','.join('?' * len(batch))})", batch).fetchall()
            )

    word_est_df: list[int] = []
    for ws, _, w in word_spans:
        if len(w) == 2:
            word_est_df.append(max(gdf.get(f" {w}", 0), gdf.get(f"{w} ", 0)))
            continue
        if len(w) == 3:
            word_est_df.append(gdf.get(w, 0))
            continue
        if len(w) < 4:
            word_est_df.append(n_tot)
            continue

        forms = [w]
        w_cf = w.casefold()
        if w_cf != w:
            forms.append(w_cf)
        best_w_df = n_tot + 1
        for fw in forms:
            t_dfs = [gdf.get(fw[j : j + 3], 0) for j in range(len(fw) - 2)]
            d_min_fw = min(t_dfs)
            best_w_df = min(best_w_df, d_min_fw)
            if min_df <= d_min_fw <= max_df:
                _propose(fw, d_min_fw, ws)
            if len(fw) >= 6:
                prev_d = t_dfs[-1]
                for ln in range(len(fw) - 1, 4, -1):
                    d_ln = min(t_dfs[: ln - 2])
                    g_end_d = t_dfs[ln - 3]
                    if min_df <= d_ln <= max_df and (prev_d < min_df or (ln >= 6 and g_end_d > prev_d and prev_d <= max_df)):
                        _propose(fw[:ln], d_ln, ws)
                    prev_d = min(prev_d, g_end_d)
            elif len(fw) == 5 and d_min_fw < min_df:
                d_4 = min(t_dfs[:2])
                if min_df <= d_4 <= max_df:
                    _propose(fw[:4], d_4, ws)

            if len(fw) >= 9:
                d_whole, _ = _lookup_postings(fw, conn, vocab_tbl, max_df, postings_cache)
                for a in range(4, len(fw) - 4):
                    left_p, right_p = fw[:a], fw[a:]
                    d_l_est = min(gdf.get(left_p[j : j + 3], 0) for j in range(len(left_p) - 2))
                    d_r_est = min(gdf.get(right_p[j : j + 3], 0) for j in range(len(right_p) - 2))
                    if d_l_est > d_whole and d_r_est > d_whole:
                        d_r_word, _ = _lookup_postings(" " + right_p, conn, vocab_tbl, max_df, postings_cache)
                        if d_r_word >= 1:
                            d_l_ex, _ = _lookup_postings(left_p, conn, vocab_tbl, max_df, postings_cache)
                            d_r_ex, _ = _lookup_postings(right_p, conn, vocab_tbl, max_df, postings_cache)
                            if min_df <= d_l_ex <= max_df:
                                _propose(left_p, d_l_ex, ws)
                            if min_df <= d_r_ex <= max_df:
                                _propose(right_p, d_r_ex, ws + a)
        word_est_df.append(best_w_df)

    # 3. Adjacent 2-word collocations filtered by Church-Hanks Positive PMI.
    if len(word_spans) >= 2:
        d_glue_cap = max(12, n_tot // 6)
        for idx_w in range(len(word_spans) - 1):
            ws1, we1, w1 = word_spans[idx_w]
            ws2, _, w2 = word_spans[idx_w + 1]
            if text_low[we1:ws2] != " " or len(w1) < 2 or len(w2) < 2:
                continue
            d1_est, d2_est = word_est_df[idx_w], word_est_df[idx_w + 1]
            if d1_est > d_glue_cap or d2_est > d_glue_cap:
                continue
            phrase = f"{w1} {w2}"
            if len(phrase) < 5:
                continue
            cross_gs = [w1[-1] + " " + w2[0], w1[-2:] + " ", " " + w2[:2]]
            ub_df = min(gdf.get(cg, 0) for cg in cross_gs)
            if ub_df < min_df or ub_df > max_df * 3:
                continue
            d_ph, _ = _lookup_postings(phrase, conn, vocab_tbl, max_df, postings_cache)
            if min_df <= d_ph <= max_df:
                d1_ex = _lookup_postings(w1, conn, vocab_tbl, n_tot, postings_cache)[0] if len(w1) >= 3 else max(d_ph, d1_est)
                d2_ex = _lookup_postings(w2, conn, vocab_tbl, n_tot, postings_cache)[0] if len(w2) >= 3 else max(d_ph, d2_est)
                if (n_tot * d_ph) >= max(1, 2 * d1_ex * d2_ex):
                    _propose(phrase, d_ph, ws1)

    verified: list[tuple[str, float, int, float, int, int]] = []
    for cs, ce, sub in raw_cands:
        exact_d, _ = _lookup_postings(sub, conn, vocab_tbl, max_df, postings_cache)
        if min_df <= exact_d <= max_df:
            surp = math.log((n_tot + 2.0) / (exact_d + 1.0))
            verified.append((sub, surp, exact_d, surp / _prox(cs), cs, ce))

    verified.sort(key=lambda x: (-x[3], -x[1], -len(x[0])))
    deduped: list[tuple[str, float, int, int, int]] = []
    for sub, surp, ex_d, _, cs, ce in verified:
        if any((sub in ps or ps in sub) and ex_d == pd and len(sub) <= len(ps) for ps, _, pd, _, _ in deduped):
            continue
        if dedup_query_spans:
            conflict = False
            for _, _, pd, pcs, pce in deduped:
                ov = max(0, min(ce, pce) - max(cs, pcs))
                if ov > 0 and ov >= 0.6 * min(ce - cs, pce - pcs) and pd > 1:
                    conflict = True
                    break
            if conflict:
                continue
        deduped.append((sub, surp, ex_d, cs, ce))
        if len(deduped) >= limit:
            break
    return [(x[0], x[1], x[2]) for x in deduped]


def _gibbs_additive_lattice_pass(
    init_log_phi: NDArray[np.float64],
    node_masks: NDArray[np.int32],
    step_log_gain: NDArray[np.float64],
    edge_mask: NDArray[np.bool_],
    node_log_rho_stop: NDArray[np.float64],
    facet_nats: NDArray[np.float64],
    horizon: int = 3,
) -> tuple[float, float, NDArray[np.float64], NDArray[np.float64], tuple[int, ...], int]:
    """Exact Forward-Backward Coverage-Lattice Stopped Gibbs over additive MDL path potentials (tau == 1.0 nat)."""
    vc, fc = len(init_log_phi), len(facet_nats)
    mc = 1 << fc
    masks = np.arange(mc, dtype=np.int32)
    or_tab = node_masks[:, None] | masks[None, :]
    v_idx = np.arange(vc, dtype=np.int32)[:, None]
    flat_or = (v_idx * mc + or_tab).ravel()

    bit_mat = ((masks[:, None] >> np.arange(fc, dtype=np.int32)) & 1).astype(np.float64) if fc else np.zeros((1, 0))
    u_lex = (bit_mat @ facet_nats)[None, :]  # shape (1, mc): union lexical coverage counted ONCE at stop!
    u_stop = node_log_rho_stop[:, None] + u_lex
    u_sh = float(max(float(u_lex.max()), float(u_stop.max())))
    g_h = np.exp(u_lex - u_sh) * np.ones((vc, 1), dtype=np.float64)
    g = np.exp(u_stop - u_sh)

    # Shift transition log-gains so t_mat = exp(step_log_gain - t_sh) never overflows float64.
    has_any_edge = bool(edge_mask.any())
    t_sh = float(step_log_gain[edge_mask].max()) if has_any_edge else 0.0
    t_mat = np.where(edge_mask, np.exp(np.clip(step_log_gain - t_sh, -700.0, 0.0)), 0.0)

    b = np.empty((horizon + 1, vc, mc), dtype=np.float64)
    b_sh = np.empty(horizon + 1, dtype=np.float64)
    b[horizon], b_sh[horizon] = g_h, u_sh
    for h in range(horizon - 1, -1, -1):
        if has_any_edge and np.isfinite(b_sh[h + 1]):
            cont = t_mat @ b[h + 1, v_idx, or_tab]
            c_sh = t_sh + float(b_sh[h + 1])
            sh = max(u_sh, c_sh)
            raw = g * np.exp(u_sh - sh) + cont * np.exp(c_sh - sh)
        else:
            sh = u_sh
            raw = g.copy()
        pk = float(raw.max())
        b[h] = raw / pk if pk > 0.0 else 0.0
        b_sh[h] = sh + math.log(pk) if pk > 0.0 else -np.inf

    sm = node_masks.copy()
    phi_sh = float(init_log_phi.max())
    phi_s = np.exp(init_log_phi - phi_sh)
    z0_vec = phi_s * b[0, np.arange(vc), sm]
    log_z = phi_sh + float(b_sh[0]) + math.log(max(float(z0_vec.sum()), 1e-300))
    log_z0 = phi_sh + u_sh + math.log(max(float((phi_s * g[np.arange(vc), sm]).sum()), 1e-300))

    a = np.zeros((horizon + 1, vc, mc), dtype=np.float64)
    a_sh = np.full(horizon + 1, -np.inf, dtype=np.float64)
    a[0, np.arange(vc), sm], a_sh[0] = phi_s, phi_sh
    tt = np.ascontiguousarray(t_mat.T)
    e_occ = np.zeros((vc, vc), dtype=np.float64)
    s_pr = np.zeros((vc, mc), dtype=np.float64)

    for h in range(horizon):
        if not np.isfinite(a_sh[h]):
            break
        s_pr += (a[h] * g) * np.exp(a_sh[h] + u_sh - log_z)
        if has_any_edge and np.isfinite(b_sh[h + 1]):
            bl = b[h + 1, v_idx, or_tab]
            e_occ += t_mat * ((a[h] @ bl.T) * np.exp(a_sh[h] + t_sh + b_sh[h + 1] - log_z))
            nxt = np.bincount(flat_or, weights=(tt @ a[h]).ravel(), minlength=vc * mc).reshape(vc, mc)
            pk = float(nxt.max())
            if pk > 0.0:
                a[h + 1], a_sh[h + 1] = nxt / pk, a_sh[h] + t_sh + math.log(pk)
    if np.isfinite(a_sh[horizon]):
        s_pr += (a[horizon] * g_h) * np.exp(a_sh[horizon] + u_sh - log_z)

    # Exact forward MAP route decoding (cycle-free):
    cur = int(np.argmax(z0_vec))
    cm = int(sm[cur])
    route = [cur]
    for h in range(horizon):
        if not has_any_edge or not np.isfinite(b_sh[h + 1]):
            break
        log_w_st = math.log(max(float(g[cur, cm]), 1e-300)) + u_sh
        cand_m = cm | node_masks
        b_next = b[h + 1, np.arange(vc), cand_m]
        valid_go = edge_mask[cur] & (b_next > 0.0)
        for pr in route:
            valid_go[pr] = False
        if not valid_go.any():
            break
        log_w_go = np.where(
            valid_go,
            step_log_gain[cur] + np.log(np.maximum(b_next, 1e-300)) + float(b_sh[h + 1]),
            -np.inf,
        )
        jb = int(np.argmax(log_w_go))
        if log_w_go[jb] <= log_w_st:
            break
        route.append(jb)
        cm, cur = int(cand_m[jb]), jb

    node_post = np.minimum(1.0, s_pr.sum(axis=1) + e_occ.sum(axis=1))
    return log_z, log_z0, node_post, e_occ, tuple(route), cm


def _route_completeness(
    chosen: list[int],
    edge_mask: NDArray[np.bool_],
    n_masks: NDArray[np.int32],
    f_nats: NDArray[np.float64],
    v_mat: NDArray[np.float32],
    qh: NDArray[np.float32],
    use_vec: bool,
    n_tot: int,
    d_eff: float,
) -> float:
    if not chosen:
        return 0.0
    fc = len(f_nats)
    tot_lex = float(f_nats.sum())
    max_vec_nats = math.log(float(n_tot) + 2.0)
    c_set = set(chosen)
    seen: set[int] = set()
    best = 0.0
    for start in chosen:
        if start in seen:
            continue
        comp_nodes: list[int] = []
        cm = 0
        stack = [start]
        seen.add(start)
        while stack:
            u = stack.pop()
            comp_nodes.append(u)
            cm |= int(n_masks[u])
            for v in c_set:
                if v not in seen and (bool(edge_mask[u, v]) or bool(edge_mask[v, u])):
                    seen.add(v)
                    stack.append(v)
        lex_cov = sum(float(f_nats[b]) for b in range(fc) if (cm >> b) & 1)
        if not use_vec:
            best = max(best, (lex_cov / tot_lex) if tot_lex > 1e-9 else 1.0)
            continue
        # At NNLS optimum, ||V_comp w+||_2 equals exact non-negative cone cosine fit against unit qh.
        v_comp = v_mat[comp_nodes].T.astype(np.float64)
        w_pos, _ = nnls(v_comp, qh.astype(np.float64))
        fit_norm = min(1.0, float(np.linalg.norm(v_comp @ w_pos)))
        vec_nats = min(float(_null_nats(fit_norm, d_eff)), max_vec_nats)
        comp_score = (
            (lex_cov + vec_nats) / (tot_lex + max_vec_nats)
            if tot_lex > 1e-9
            else (vec_nats / max_vec_nats)
        )
        best = max(best, comp_score)
    return best


def _is_sym_clone(r_u: frozenset[str], r_v: frozenset[str]) -> bool:
    return bool(r_u and r_v and (len(r_u & r_v) * 4) > (len(r_u | r_v) * 3))


def solve_evidence_dag(
    vs: VectorSearch,
    query_embedding: NDArray[np.float32] | None,
    *,
    query_text: str,
    conn: sqlite3.Connection,
    corpus_root: Path | None = None,
    exclude_chunk_ids: Sequence[int] = (),
    path_glob: str | None = None,
    top_k: int = 5,
) -> tuple[list[tuple[int, float]], SearchTrace]:
    if vs.size == 0 or top_k <= 0:
        return [], SearchTrace()

    vocab_tbl = _resolve_vocab_table(conn)
    db_file = _db_file_path(conn)
    root = corpus_root or vs.corpus_root or (db_file.parent.parent if db_file and db_file.parent.name == ".yaw" else None)
    root_res = root.resolve() if root is not None else None
    n_tot = vs.size
    use_vec = query_embedding is not None and vs.dim > 0
    excl_set = {int(x) for x in exclude_chunk_ids if vs.pos_of(int(x)) is not None}

    qh = _center_normalize(query_embedding, vs.mu) if use_vec and query_embedding is not None else np.zeros(0, np.float32)
    # QR subspace deflation is applied ONLY to explicit Turn-2+ `exclude_chunk_ids` (orthogonal branch discovery).
    if use_vec and excl_set:
        ex_list = sorted(excl_set)
        ex_rows = conn.execute(
            f"SELECT embedding FROM chunks WHERE chunk_id IN ({','.join('?' * len(ex_list))})", ex_list
        ).fetchall()
        if ex_rows:
            ex_vecs = _center_normalize(_decode_embeddings([r[0] for r in ex_rows], vs.dim), vs.mu)
            qx, _ = np.linalg.qr(ex_vecs.T, mode="reduced")
            rem0 = qh - qx @ (qx.T @ qh)
            if float(np.linalg.norm(rem0)) > 1e-6:
                qh = (rem0 / float(np.linalg.norm(rem0))).astype(np.float32)

    # Overlap Hop-0 Stage-1 SoA RaBitQ scan across CPU cores with SQLite glob/vocab/anchor work.
    rabitq_state = _rabitq_start_async(vs, qh) if use_vec else None
    allowed_mask = _allowed_mask_for_glob(vs, conn, path_glob)

    def _allowed(cid: int) -> bool:
        pos = vs.pos_of(cid)
        return pos is not None and (allowed_mask is None or bool(allowed_mask[pos]))

    d_bridge_max = max(4, int(math.ceil(math.sqrt(n_tot))))
    d_query_max = max(8, n_tot // 2)
    f_max = min(6, max(2, int(math.log2(max(4, n_tot)) // 2)))
    max_cand = 3 * d_bridge_max

    postings_cache: dict[str, tuple[int, tuple[int, ...]]] = {}
    gram_df_cache: dict[str, int] = {}
    qg = list(dict.fromkeys(_extract_grams(query_text.lower())))
    if qg:
        gram_df_cache.update(
            conn.execute(f"SELECT term, doc FROM {vocab_tbl} WHERE term IN ({','.join('?' * len(qg))})", qg).fetchall()
        )
    facets = _extract_multilingual_anchors(
        query_text, gram_df_cache, conn, vocab_tbl, n_tot, 1, d_query_max, f_max, postings_cache, dedup_query_spans=True
    )
    for w_miss in [w for w in _WORD_TOKEN_RE.findall(query_text.lower()) if len(w) >= 4 and not _CJK_CH_RE.search(w)]:
        if len(facets) < f_max and not any(st in w_miss or w_miss in st for st, _, _ in facets):
            d_m, _ = _lookup_postings(w_miss, conn, vocab_tbl, d_query_max, postings_cache)
            if d_m == 0:
                facets.append((w_miss, math.log(n_tot + 2.0), 0))

    est_q: NDArray[np.float32] | None = None
    eps_q: NDArray[np.float32] | None = None
    if rabitq_state is not None:
        est_q, eps_q = _rabitq_finish(vs, rabitq_state)

    live_facets = [f for f in facets if f[2] > 0]
    if not live_facets:
        if not use_vec or est_q is None or eps_q is None:
            return [], SearchTrace(unexplained_facets=tuple(f[0] for f in facets))
        cids_only = [
            c for c in _rabitq_bnb_top(vs, conn, qh, top_k + len(excl_set), allowed_mask, (est_q, eps_q)) if c not in excl_set
        ][:top_k]
        return [(c, 1.0) for c in cids_only], SearchTrace(unexplained_facets=tuple(f[0] for f in facets))

    # Hop-0 Stratified Seeding: rank each facet's postings via Stage-1 RaBitQ query estimates!
    fts_ids: list[int] = []
    per_f = max(2, d_bridge_max // len(live_facets))
    for f_str, _, f_df in live_facets:
        _, rids = _lookup_postings(f_str, conn, vocab_tbl, max(d_query_max, f_df), postings_cache)
        if not rids:
            rids = tuple(
                int(r[0])
                for r in conn.execute(
                    "SELECT rowid FROM chunk_fts WHERE chunk_fts MATCH ? ORDER BY bm25(chunk_fts) ASC LIMIT ?",
                    (_fts_quote(f_str), per_f * (4 if path_glob else 2)),
                ).fetchall()
            )
        cand_f = [ic for ic in rids if _allowed(ic)]
        if est_q is not None:
            cand_f.sort(key=lambda cid: -float(est_q[vs.pos_of(cid) or 0]))
        added_f = 0
        for ic in cand_f:
            if ic not in fts_ids:
                fts_ids.append(ic)
                added_f += 1
                if added_f >= per_f or len(fts_ids) >= d_bridge_max:
                    break

    top_d = (
        _rabitq_bnb_top(vs, conn, qh, max(8, d_bridge_max // 2), allowed_mask, (est_q, eps_q))
        if use_vec and est_q is not None and eps_q is not None
        else []
    )
    seeds = list(dict.fromkeys([*excl_set, *fts_ids, *top_d]))[: d_bridge_max + len(excl_set)]

    visited_vecs: dict[int, NDArray[np.float32]] = {}
    raw_texts: dict[int, str] = {}
    low_texts: dict[int, str] = {}
    cf_texts: dict[int, str] = {}
    rare_sets: dict[int, frozenset[str]] = {}
    file_cache: dict[str, tuple[str, tuple[str, ...]] | None] = {}
    ptr_cache: dict[int, ChunkPointer] = {}

    def _load_rel_text(rel: str) -> tuple[str, tuple[str, tuple[str, ...]] | None]:
        if root_res is None:
            return rel, None
        try:
            fp = resolve_corpus_path(root_res, rel)
        except ValueError:
            return rel, None
        return rel, _read_verified_text(fp)

    def _visit(cids: list[int]) -> list[int]:
        need = [c for c in cids if (c in excl_set or _allowed(c)) and c not in low_texts][: max(0, max_cand - len(low_texts))]
        if not need:
            return []
        missing_ptrs = [c for c in need if c not in ptr_cache]
        if missing_ptrs:
            ptr_cache.update(_fetch_pointers(vs, conn, missing_ptrs))
        if use_vec:
            rows = conn.execute(
                f"SELECT chunk_id, embedding FROM chunks WHERE chunk_id IN ({','.join('?' * len(need))})", need
            ).fetchall()
            if rows:
                vecs = _center_normalize(_decode_embeddings([r[1] for r in rows], vs.dim), vs.mu)
                for idx_r, r in enumerate(rows):
                    visited_vecs[int(r[0])] = vecs[idx_r]

        new_rels = list(
            dict.fromkeys(
                ptr_cache[cid].rel_posix_path
                for cid in need
                if cid in ptr_cache and ptr_cache[cid].rel_posix_path not in file_cache
            )
        )
        if len(new_rels) > 1 and _WORKERS > 1:
            for rel, loaded_t in _LEAF_POOL.map(_load_rel_text, new_rels):
                file_cache[rel] = loaded_t
        else:
            for rel in new_rels:
                _, file_cache[rel] = _load_rel_text(rel)

        new_grams: list[str] = []
        chunk_grams: dict[int, frozenset[str]] = {}
        added: list[int] = []
        for cid in need:
            ptr = ptr_cache.get(cid)
            if ptr is None:
                continue
            loaded_t = file_cache.get(ptr.rel_posix_path)
            if loaded_t is None or ptr.start_line < 1 or ptr.end_line < ptr.start_line:
                continue
            norm_text, lines = loaded_t
            if ptr.end_line > len(lines):
                continue
            raw_slice = "\n".join(lines[ptr.start_line - 1 : ptr.end_line])
            if ptr.source_hash and hashlib.sha256(raw_slice.encode("utf-8")).hexdigest() != ptr.source_hash:
                continue
            rt = (
                norm_text[ptr.char_start : ptr.char_end]
                if ptr.char_end > ptr.char_start
                else raw_slice
            )
            raw_texts[cid] = rt
            lt = rt.lower()
            low_texts[cid] = lt
            cf_texts[cid] = rt.casefold()
            gs = frozenset(_extract_grams(lt))
            chunk_grams[cid] = gs
            new_grams.extend(g for g in gs if g not in gram_df_cache)
            added.append(cid)

        unk = list(dict.fromkeys(new_grams))
        for k0 in range(0, len(unk), 400):
            batch = unk[k0 : k0 + 400]
            gram_df_cache.update(
                conn.execute(f"SELECT term, doc FROM {vocab_tbl} WHERE term IN ({','.join('?' * len(batch))})", batch).fetchall()
            )
        for cid, gs in chunk_grams.items():
            rare_sets[cid] = frozenset(g for g in gs if 1 <= gram_df_cache.get(g, 0) <= d_bridge_max)
        return added

    frontier = [c for c in _visit(seeds) if c not in excl_set]
    bridge_df: dict[str, int] = {}
    sem_bridge_edges: dict[tuple[int, int], float] = {}
    dead_end_cands: list[str] = []
    q_fset = {f[0] for f in facets}
    spec_q = {f[0] for f in live_facets if f[2] <= d_bridge_max}
    entry_stems: dict[int, set[str]] = {c: set(spec_q) for c in frontier}
    bonferroni_nats = math.log(float(n_tot) + 1.0)
    max_vec_nats = math.log(float(n_tot) + 2.0)

    # 2-Wave Step-Wise Hybrid Expansion: BOTH RaBitQ Partial-Correlation Scans AND Lexical Bridges at Every Step!
    for _ in range(2):
        frontier_ord = sorted(
            frontier,
            key=lambda c: -(
                max((f[1] for f in live_facets if _anchor_matches_text(f[0], cf_texts[c])), default=0.0)
                + (
                    min(float(_null_nats(float(visited_vecs[c] @ qh), vs.d_eff)), max_vec_nats)
                    if use_vec and c in visited_vecs
                    else 0.0
                )
            ),
        )
        dedup_f: list[int] = []
        seen_excl = [rare_sets[c] for c in excl_set if c in rare_sets]
        for fc in frontier_ord:
            rs = rare_sets[fc]
            if not any(_is_sym_clone(prev, rs) for prev in seen_excl) and not any(_is_sym_clone(rare_sets[p], rs) for p in dedup_f):
                dedup_f.append(fc)
            if len(dedup_f) >= 12:
                break
        if not dedup_f or len(low_texts) >= max_cand:
            break

        # Step-Wise RaBitQ Evidence Gathering from top frontier vectors (captures non-lexical / synonym / CJK / inflected hops):
        dense_wave_cids: list[int] = []
        if use_vec and est_q is not None:
            q_denom = np.sqrt(np.maximum(1e-4, 1.0 - np.minimum(est_q * est_q, 0.9999)))
            for u_cid in dedup_f[:4]:
                if u_cid not in visited_vecs:
                    continue
                u_hat = visited_vecs[u_cid]
                est_u, _ = _rabitq_estimate(vs, u_hat)
                c_uq = float(np.clip(u_hat @ qh, -0.999, 0.999))
                u_den = math.sqrt(max(1e-4, 1.0 - c_uq * c_uq))
                rho_part = np.clip((est_u - c_uq * est_q) / (u_den * q_denom), -0.999, 0.999)
                if allowed_mask is not None:
                    rho_part = np.where(allowed_mask, rho_part, -1.0)
                p_nats = _null_nats(rho_part, max(3.0, vs.d_eff - 1.0))
                u_pos = vs.pos_of(u_cid)
                if u_pos is not None:
                    p_nats[u_pos] = 0.0
                k_top = min(3, len(p_nats))
                if k_top > 0:
                    top_nbr = np.argpartition(-p_nats, k_top - 1)[:k_top]
                    for p_idx in top_nbr.tolist():
                        pn = float(p_nats[p_idx])
                        if pn > bonferroni_nats:
                            v_cid = int(vs.chunk_ids_arr[p_idx])
                            if v_cid not in excl_set and _allowed(v_cid):
                                sem_bridge_edges[(u_cid, v_cid)] = max(sem_bridge_edges.get((u_cid, v_cid), 0.0), pn)
                                dense_wave_cids.append(v_cid)

        # B) Lexical Anchor Bridges ranked by joint (bridge + RaBitQ query) evidence
        per_chunk_stems: list[list[tuple[str, int]]] = []
        for c in dedup_f:
            lt = low_texts[c]
            apos = [idx_p for st in entry_stems.get(c, ()) if (idx_p := lt.find(st)) != -1]
            c_ancs = _extract_multilingual_anchors(
                raw_texts[c], gram_df_cache, conn, vocab_tbl, n_tot, 1, d_bridge_max, f_max + 2, postings_cache, apos or None
            )
            c_st: list[tuple[str, int]] = []
            for st, _, mdf in c_ancs:
                if any(st in qf or qf in st for qf in q_fset):
                    continue
                if mdf == 1:
                    if st not in dead_end_cands:
                        dead_end_cands.append(st)
                elif st not in bridge_df:
                    c_st.append((st, mdf))
            if c_st:
                per_chunk_stems.append(c_st)

        wave_b: list[str] = []
        for r_idx in range(f_max + 2):
            for lst in per_chunk_stems:
                if r_idx < len(lst):
                    st, mdf = lst[r_idx]
                    if st not in bridge_df:
                        bridge_df[st] = mdf
                        wave_b.append(st)
                        if len(wave_b) >= 12:
                            break
            if len(wave_b) >= 12:
                break

        lex_wave_cids: list[int] = []
        if wave_b:
            per_b = max(2, (max_cand - len(low_texts)) // len(wave_b))
            for pb in wave_b:
                _, rids = _lookup_postings(pb, conn, vocab_tbl, d_bridge_max, postings_cache)
                if not rids:
                    rids = tuple(
                        int(r[0])
                        for r in conn.execute(
                            "SELECT rowid FROM chunk_fts WHERE chunk_fts MATCH ? ORDER BY bm25(chunk_fts) ASC LIMIT ?",
                            (_fts_quote(pb), per_b * 2),
                        ).fetchall()
                    )
                cand_b = [ic for ic in rids if _allowed(ic) and ic not in excl_set]
                if est_q is not None:
                    cand_b.sort(key=lambda cid: -float(est_q[pos if (pos := vs.pos_of(cid)) is not None else 0]))
                for ic in cand_b[:per_b]:
                    entry_stems.setdefault(ic, set()).add(pb)
                    lex_wave_cids.append(ic)

        wave_cids = list(dict.fromkeys([*lex_wave_cids, *dense_wave_cids]))
        if not wave_cids:
            break
        frontier = [c for c in _visit(wave_cids) if c not in excl_set]
        if not frontier:
            break

    c_list = sorted(c for c in low_texts if c not in excl_set)
    vc = len(c_list)
    if vc == 0:
        return [], SearchTrace(unexplained_facets=tuple(f[0] for f in facets))
    cid_to_local = {c: i for i, c in enumerate(c_list)}
    rsets = [rare_sets[c] for c in c_list]

    fc_len = len(facets)
    f_nats = np.array([f[1] for f in facets], dtype=np.float64)
    n_masks = np.array(
        [sum((1 << bi) for bi, (ph, _, _) in enumerate(facets) if _anchor_matches_text(ph, cf_texts[c_list[i]])) for i in range(vc)],
        dtype=np.int32,
    )

    def _rnov(u_r: frozenset[str], v_r: frozenset[str]) -> float:
        return 1.0 - (len(u_r & v_r) / float(len(v_r))) if v_r else 1.0

    # Build directed bridge graph with minimum Shannon hop cost c_hop(u -> v).
    edge_cost = np.full((vc, vc), np.inf, dtype=np.float64)
    edge_anchor: dict[tuple[int, int], str] = {}
    for ph, dfb in bridge_df.items():
        members = [i for i in range(vc) if _anchor_matches_text(ph, cf_texts[c_list[i]])]
        if len(members) < 2:
            continue
        raw_cost = math.log(float(dfb) + 1.0)
        for ui in members:
            for vi in members:
                if ui != vi and not _is_sym_clone(rsets[ui], rsets[vi]):
                    c_uv = raw_cost / max(0.25, _rnov(rsets[ui], rsets[vi]))
                    if c_uv < edge_cost[ui, vi]:
                        edge_cost[ui, vi] = c_uv
                        edge_anchor[(ui, vi)] = ph

    # Include pure Stage-1 RaBitQ semantic bridge edges discovered via partial correlation rho_{uv|q}.
    for (u_cid, v_cid), p_nats in sem_bridge_edges.items():
        ui = cid_to_local.get(u_cid)
        vi = cid_to_local.get(v_cid)
        if ui is not None and vi is not None and ui != vi and not _is_sym_clone(rsets[ui], rsets[vi]):
            sem_cost = math.log(2.0 + math.exp(bonferroni_nats - p_nats))
            if sem_cost < edge_cost[ui, vi]:
                edge_cost[ui, vi] = sem_cost
                edge_anchor.setdefault((ui, vi), "~vec")

    v_mat = np.vstack([visited_vecs[c] for c in c_list]) if use_vec else np.zeros((vc, 1), dtype=np.float32)
    q_nats = (
        np.minimum(_null_nats(v_mat @ qh, vs.d_eff), math.log(float(n_tot) + 2.0))
        if use_vec
        else np.zeros(vc, dtype=np.float64)
    )

    edge_mask = np.isfinite(edge_cost)
    out_deg = edge_mask.sum(axis=1)
    rho_stop = 1.0 / (1.0 + out_deg)
    if use_vec:
        sim_mat = np.clip(v_mat @ v_mat.T, 0.0, 0.9999)
        sin2_nov = 1.0 - sim_mat * sim_mat
    else:
        sin2_nov = np.ones((vc, vc), dtype=np.float64)

    # Additive MDL log-gain per step (u -> v): non-redundant query semantic nats minus Shannon bridge cost.
    step_log_gain = np.where(edge_mask, sin2_nov * q_nats[None, :] - edge_cost, -np.inf)
    init_log_phi = q_nats - math.log(float(n_tot) + 1.0)

    log_z, log_z0, node_post, e_occ, route, end_m = _gibbs_additive_lattice_pass(
        init_log_phi, n_masks, step_log_gain, edge_mask, np.log(rho_stop), f_nats, horizon=3
    )
    log_bf = float(log_z - log_z0)

    chosen: list[int] = list(route[:top_k])
    hops_out: list[EvidenceHop] = [
        EvidenceHop(
            c_list[route[i - 1]],
            c_list[route[i]],
            edge_anchor.get((route[i - 1], route[i]), ""),
            round(float(e_occ[route[i - 1], route[i]]), 4),
        )
        for i in range(1, len(route))
    ]

    for ni in np.argsort(-node_post):
        ni_int = int(ni)
        if len(chosen) >= top_k:
            break
        if ni_int in chosen:
            continue
        if any(_is_sym_clone(rsets[c_i], rsets[ni_int]) for c_i in chosen):
            continue
        chosen.append(ni_int)
    while len(chosen) < min(top_k, vc):
        for ni in range(vc):
            if ni not in chosen:
                chosen.append(ni)
                break

    final_chosen = chosen[:top_k]
    cc_comp = _route_completeness(final_chosen, edge_mask, n_masks, f_nats, v_mat, qh, use_vec, n_tot, vs.d_eff)
    unexplained = tuple(facets[b][0] for b in range(fc_len) if not ((end_m >> b) & 1))
    used_anchors = {h.anchor for h in hops_out if h.anchor}
    dead_ends = tuple(dict.fromkeys([*dead_end_cands, *(b for b in bridge_df if b not in used_anchors)]))[:6] if cc_comp < 0.95 else ()
    ranked = [(c_list[i], float(node_post[i])) for i in final_chosen]
    return ranked, SearchTrace(tuple(hops_out), round(cc_comp, 4), round(log_bf, 4), unexplained, dead_ends)


def search_vectors(
    vs: VectorSearch,
    query_embedding: NDArray[np.float32],
    *,
    query_text: str = "",
    conn: sqlite3.Connection | None = None,
    top_k: int = 10,
    **_ignored: Any,
) -> list[tuple[int, float]]:
    if vs.size == 0 or top_k <= 0:
        return []
    if conn is None:
        qh = _center_normalize(query_embedding, vs.mu)
        est, _ = _rabitq_estimate(vs, qh)
        k_c = min(vs.size, top_k)
        idx = np.argpartition(-est, k_c - 1)[:k_c]
        order = idx[np.argsort(-est[idx])]
        return [(int(vs.chunk_ids_arr[int(i)]), float(est[int(i)])) for i in order]
    ranked, _ = solve_evidence_dag(vs, query_embedding, query_text=query_text, conn=conn, top_k=top_k)
    return ranked


def expand_neighbors_ranked(
    vs: VectorSearch, conn: sqlite3.Connection, chunk_id: int, *, top_k: int = 6
) -> list[tuple[int, float]]:
    pos = vs.pos_of(chunk_id)
    if pos is None or vs.size <= 1 or top_k <= 0:
        return []
    row = conn.execute("SELECT embedding FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
    if row is None:
        return []
    u_hat = _center_normalize(_decode_embeddings([row[0]], vs.dim)[0], vs.mu)
    est, _ = _rabitq_estimate(vs, u_hat)
    scores = _null_nats(est, vs.d_eff)
    scores[pos] = -1.0
    k_c = min(top_k, vs.size - 1)
    idx = np.argpartition(-scores, k_c - 1)[:k_c]
    order = idx[np.argsort(-scores[idx])]
    return [(int(vs.chunk_ids_arr[int(i)]), float(scores[int(i)])) for i in order if scores[int(i)] >= 0.0]


def lexical_search(
    vs: VectorSearch, query_text: str, conn: sqlite3.Connection, *, top_k: int = 8
) -> list[tuple[int, float]]:
    if vs.size == 0 or top_k <= 0:
        return []
    vocab_tbl = _resolve_vocab_table(conn)
    q_low = query_text.lower()
    qg = list(dict.fromkeys(_extract_grams(q_low)))
    if not qg:
        return []
    gdf = dict(
        conn.execute(f"SELECT term, doc FROM {vocab_tbl} WHERE term IN ({','.join('?' * len(qg))})", qg).fetchall()
    )
    stems = _extract_multilingual_anchors(q_low, gdf, conn, vocab_tbl, vs.size, 1, max(8, vs.size // 2), 6, {}, dedup_query_spans=True)
    live = [s for s in stems if len(s[0]) >= 3]
    if not live:
        return []
    rows = conn.execute(
        "SELECT rowid, -bm25(chunk_fts) FROM chunk_fts WHERE chunk_fts MATCH ? ORDER BY bm25(chunk_fts) ASC LIMIT ?",
        (" OR ".join(_fts_quote(s[0]) for s in live), min(vs.size, top_k)),
    ).fetchall()
    return [(int(cid), float(sc)) for cid, sc in rows if vs.pos_of(int(cid)) is not None]


def build_query_hints(vs: VectorSearch, query_text: str, *, limit: int = 5) -> list[str]:
    del vs, query_text, limit
    return []


def _validate_max_chars(max_chars: int | None) -> int | None:
    if max_chars is None:
        return None
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 0:
        raise ValueError(f"max_chars must be a non-negative int or None, got {max_chars!r}")
    return max_chars


def _crop_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    lines = text.splitlines()
    if len(lines) >= 3 and max_chars >= 120:
        half = max(20, (max_chars - 48) // 2)
        head: list[str] = []
        used = 0
        for ln in lines:
            if head and used + len(ln) + 1 > half:
                break
            head.append(ln)
            used += len(ln) + 1
        tail: list[str] = []
        used_t = 0
        for ln in reversed(lines[len(head) :]):
            if tail and used_t + len(ln) + 1 > half:
                break
            tail.append(ln)
            used_t += len(ln) + 1
        tail.reverse()
        omitted_start = len(head) + 1
        omitted_end = len(lines) - len(tail)
        if omitted_end >= omitted_start and head and tail:
            marker = f"[... lines {omitted_start}-{omitted_end} omitted ...]"
            candidate = "\n".join([*head, marker, *tail])
            if len(candidate) <= max_chars:
                return candidate
    marker_c = "\n[... AST section cropped ...]\n"
    if max_chars <= len(marker_c) + 8:
        return text[:max_chars]
    rem = max_chars - len(marker_c)
    left = rem // 2
    right = rem - left
    return f"{text[:left]}{marker_c}{text[-right:]}"


def _hydrate_one(
    ptr: ChunkPointer,
    score: float,
    corpus_root: Path,
    file_cache: dict[str, tuple[str, tuple[str, ...]] | None],
    *,
    max_chars: int | None = None,
    include_island: bool = False,
    full_span: bool = False,
) -> HydratedPassage:
    limit = _validate_max_chars(max_chars)
    sl, el = int(ptr.start_line), int(ptr.end_line)
    if sl < 1 or el < sl or (el - sl + 1) > _MAX_HYDRATE_SPAN:
        raise ValueError(f"Invalid hydration line range [{sl}, {el}] for {ptr.rel_posix_path}")
    full_path = resolve_corpus_path(corpus_root, ptr.rel_posix_path)
    rel = ptr.rel_posix_path
    if rel not in file_cache:
        file_cache[rel] = _read_verified_text(full_path)
    loaded = file_cache[rel]
    if loaded is None:
        return HydratedPassage(ptr, "", float(score), stale_on_disk=True)
    text, lines = loaded
    if el > len(lines):
        return HydratedPassage(ptr, "", float(score), stale_on_disk=True)
    raw_slice = "\n".join(lines[sl - 1 : el])
    if ptr.source_hash and hashlib.sha256(raw_slice.encode("utf-8")).hexdigest() != ptr.source_hash:
        return HydratedPassage(ptr, "", float(score), stale_on_disk=True)
    if include_island and 0 < ptr.ast_island_start <= ptr.ast_island_end:
        es, ee = min(sl, int(ptr.ast_island_start)), min(len(lines), max(el, int(ptr.ast_island_end)))
        rendered = "\n".join(lines[es - 1 : ee])
    elif not full_span and ptr.char_end > ptr.char_start:
        rendered = text[ptr.char_start : ptr.char_end]
    else:
        rendered = raw_slice
    if (
        ptr.title
        and not rendered.lstrip().startswith("#")
        and not rendered.startswith(f"{ptr.title}\n")
        and rendered != ptr.title
    ):
        rendered = f"{ptr.title}\n{rendered}"
    if limit is not None and limit > 0:
        rendered = _crop_text(rendered, limit)
    return HydratedPassage(ptr, rendered, float(score), stale_on_disk=False)


def hydrate_passage(
    ptr: ChunkPointer,
    corpus_root: Path,
    *,
    score: float = 0.0,
    max_chars: int | None = None,
    include_island: bool = False,
    full_span: bool = False,
) -> HydratedPassage:
    return _hydrate_one(
        ptr,
        score,
        corpus_root,
        {},
        max_chars=max_chars,
        include_island=include_island,
        full_span=full_span,
    )


def hydrate_ranked_passages(
    vs: VectorSearch,
    ranked_chunks: Sequence[tuple[int, float]],
    corpus_root: Path,
    *,
    top_k: int = 5,
    max_chars: int | None = None,
    include_island: bool = False,
    full_span: bool = False,
    conn: sqlite3.Connection | None = None,
) -> tuple[HydratedPassage, ...]:
    limit = _validate_max_chars(max_chars)
    if top_k <= 0 or not ranked_chunks:
        return ()
    ptr_map = _fetch_pointers(vs, conn, [cid for cid, _ in ranked_chunks])

    unique_rels = list(dict.fromkeys(ptr_map[cid].rel_posix_path for cid, _ in ranked_chunks if cid in ptr_map))
    cache: dict[str, tuple[str, tuple[str, ...]] | None] = {}
    if len(unique_rels) > 1 and _WORKERS > 1:
        def _load(rel: str) -> tuple[str, tuple[str, tuple[str, ...]] | None]:
            return rel, _read_verified_text(resolve_corpus_path(corpus_root, rel))

        for rel, loaded_t in _LEAF_POOL.map(_load, unique_rels):
            cache[rel] = loaded_t

    out: list[HydratedPassage] = []
    seen_spans: set[tuple[str, int, int, int, int]] = set()
    for cid, sc in ranked_chunks:
        ptr = ptr_map.get(cid)
        if ptr is None:
            continue
        span_key = (
            ptr.rel_posix_path,
            int(ptr.start_line),
            int(ptr.end_line),
            int(ptr.char_start),
            int(ptr.char_end),
        )
        if span_key in seen_spans:
            continue
        seen_spans.add(span_key)
        out.append(
            _hydrate_one(
                ptr,
                sc,
                corpus_root,
                cache,
                max_chars=limit,
                include_island=include_island,
                full_span=full_span,
            )
        )
        if len(out) >= top_k:
            break
    return tuple(out)


def format_evidence_route(passages: Sequence[HydratedPassage]) -> str:
    """Format hydrated passages with explicit provenance headers for ReAct / recall output."""
    blocks: list[str] = []
    for idx, hp in enumerate(passages, start=1):
        if hp.stale_on_disk:
            blocks.append(
                f"[{idx}] [STALE: {hp.pointer.span_label} | chunk_id={hp.pointer.chunk_id} | score={hp.score:.4f}]"
            )
        else:
            blocks.append(
                f"[{idx}] [{hp.pointer.span_label} | chunk_id={hp.pointer.chunk_id} | score={hp.score:.4f}]\n{hp.text}"
            )
    return "\n\n".join(blocks)
```

---

### `MDL-3`: Wiring `yaw/runtime.py`, `yaw/qa.py`, `yaw/mcp_server.py` & `yaw/core/config.py`
1. **`yaw/qa.py`:** Replace `expand_neighbors` body to call `expand_neighbors_ranked(vs, conn, int(chunk_id), top_k=6)` and `hydrate_ranked_passages(..., conn=conn)`, removing all accesses to deleted `vs.embeddings` and `vs.entity_to_chunk`.
2. **`yaw/runtime.py`:**
   - Move `create_lm(cfg)` out of `start_runtime()` into lazy initialization inside `ask()` protected by a `threading.Lock` (`if runtime.lm is None: ...`).
   - Open per-request / thread-local `open_readonly_db(cfg.db_path)` connections inside `recall(...)` and `ask(...)` while sharing the immutable `VectorSearch` (`~108 B/chunk`) across all FastMCP worker threads.
   - Update `recall(...)` to accept `exclude_chunk_ids: Sequence[int] = ()` and `path_glob: str | None = None`, call `solve_evidence_dag(..., top_k=top_k)`, and return hydrated passages (`conn=conn`) plus serialized `SearchTrace` (computing `route_completeness` strictly on the final returned `top_k` set—never a `pool_k=50` over-fetch).
   - Include `mtime_ns` (via `(corpus_root / rel).stat().st_mtime_ns`) in `serialize_passage`'s `source` metadata so calling agents can detect newer superseding files.
3. **`yaw/mcp_server.py` & `yaw/core/config.py`:**
   - Expose optional `exclude_chunk_ids: list[int] = []` and `path_glob: str | None = None` on `recall` and include `trace` in the returned JSON payload.
   - Expand `DEFAULT_EXCLUDED_PATTERNS` in `yaw/core/config.py` and `yaw.toml` to exclude `node_modules/*`, `target/*`, `dist/*`, `build/*`, `venv/*` (`http_port` is already `13306`).

---

## 2. Mandatory Acceptance Gates (`MDL-4` & `MDL-5`)

1. **Adversarial, Single-File & Concurrency Suite (`tests/test_trigram_watershed_dag.py`):**
   - **Single-File Purity & Read-Only Gate:** Stage `title == ""` multi-line files in `tokenize='trigram'` `chunk_fts` + `main.chunk_vocab`, run `INSERT INTO chunk_fts(chunk_fts) VALUES('optimize')`, persist `yaw_stage1` via `persist_stage1=True`, assert **zero** `.yaw/stage1/` directory or sidecar files exist on disk, reopen over `open_readonly_db` (`?mode=ro`, `PRAGMA query_only = ON`) with `persist_stage1=False`, and assert `vs.rabitq_bits_t.flags.writeable is False` and `vs.rabitq_bits.shape == (N, (D + 63) // 64)`. Also test opening a legacy 2-table DB (without `main.chunk_vocab`) via `open_readonly_db` to verify `temp.chunk_vocab` attaches cleanly without `OperationalError`.
   - **Certified Progressive BnB Gate:** Verify `_rabitq_bnb_top` achieves exact `R@10 == 1.0` against brute-force mean-centered cosine on anisotropic vectors (`D = 768`) without any hardcoded `4 * k_c` truncation.
   - **Hydration Cache & Staleness Gate:** Verify `_cached_file_text` records cache hits when `hydrate_ranked_passages` runs after `solve_evidence_dag` (`stale_on_disk == False`), flips `stale_on_disk == True` immediately upon a 1-byte edit on disk, and enforces MaxP span deduplication.
   - **Unjuked Multilingual, Phrase, Decompounding & Semantic-Bridge Gate (Zero Query Endpoint Hints):**
     - **EN:** 3-hop chains with high-DF target-predicate decoys (`Mozart` / `Middlemarch` / `Polonium`), 2-word common-word bridges (`"Red Sea"`, `"Bo Wu"`), and paraphrase hops bridged via Step-Wise `RaBitQ` partial correlation (`~vec`).
     - **DE:** Split compounds (`Strahlenschutzkommission` $\to$ `Kommission für Strahlenschutz`), sibling compounds (`Nordkanaldienst` $\leftrightarrow$ `Nordkanalverwaltung`), umlaut plurals (`Medizinverbund` $\leftrightarrow$ `Medizinverbünde`), and `ß` $\leftrightarrow$ `ss` (`Schloß` $\leftrightarrow$ `Schloss`).
     - **ZH:** Unspaced 2-char Hanzi bridges (`鲁迅` $\to$ `绍兴`, `苏轼` $\to$ `眉山`, `海思` $\to$ `深圳`) including boundary EOF tokens (`碎叶`).
     - **FI & Logs:** Inflected Finnish (`Äänekorvella` $\to$ `Äänekorven` $\to$ `Järvenlahtisolmuun` $\leftrightarrow$ `Järvenlahtisolmussa`) and embedded log paths (`svc::Auriga0_Ledger` $\to$ `k8s/MercuryGate0` $\to$ `/var/SolisVault0.sock`) despite clone floods and distant red herrings.
   - **QR Deflation, `PurePosixPath` Globbing & Reject Diagnostics Gate:** Verify `exclude_chunk_ids=[branch_a_ids]` on Turn 2 returns an orthogonal second branch without repeating Turn 1 chunks, `_posix_glob_match("sub/a.py", "*.py") is False` while `"sub/*.py"` / `"sub/**/*.py"` match strictly, and `trace.route_completeness` + `trace.dead_end_bridges` separate complete chains from broken dead-end chains (`Reject AUROC` gated).
   - **16-Thread Concurrency Stress Gate:** Fire 32+ concurrent `solve_evidence_dag` calls across 16 threads sharing one `VectorSearch` with per-thread `open_readonly_db` connections; assert zero deadlocks (`_LEAF_POOL`), zero SQLite lock errors, and deterministic output.
2. **Real Corpus Benchmark Gate (`G3` — Real Gemma Embeddings + Real `MuSiQue`/`2Wiki`/`HotpotQA`/`PopQA`):**
   - Rebuild `.yaw/yaw.db` with `tokenize='trigram'`, `chunk_vocab`, `FTS5 'optimize'`, and `yaw_stage1` using **real** `EmbeddingGemma` vectors (never synthetic proxies) and run `[0, 10)` (`n = 40`) regression and `[10, 60)` (`n = 200`) evaluation across `2wikimultihopqa`, `musique`, `hotpotqa`, and `popqa`.
   - Report per-dataset `R@5`, `All-Hops@5`, `F1`/`EM`, `Reject AUROC`, p95 latency, resident RAM (`tracemalloc` + RSS), Progressive BnB Stage-2 fetch counts, and paired Wilcoxon $p$-values + leave-one-out ablations (Word-Boundary Morphology, Koehn–Knight Decompounding, Positive-PMI Phrases, 2-Char CJK Vocab Seek, Step-Wise `RaBitQ` Partial Correlation, Additive Lattice Gibbs vs. Rev-1 Gibbs, `0-Hop Hybrid`, and `CS-6`).