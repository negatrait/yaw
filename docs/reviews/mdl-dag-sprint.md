# Sprint Spec: Post-`CS-6` FTS5-Trigram Watershed, Stage-1 `mmap` RaBitQ & Coverage-Lattice Gibbs (`mdl-dag-sprint`)

## 0. Sequencing & Hard Constraints

1. **Entry Gate:** Execute strictly after `docs/reviews/bare-bones-replacement-sprint.md` (`CS-6`) is green.
2. **Zero New Modules & Preserved `CS-6` Schema/MCP Contracts:**
   - Keep the exact 14-file boundary (`yaw/*.py`, `yaw/core/*.py`).
   - Preserve `list_user_tables(conn) == ("chunk_fts", "chunks")` (`CS-6 G1`); attach `fts5vocab` strictly in `temp` (`CREATE VIRTUAL TABLE IF NOT EXISTS temp.chunk_vocab USING fts5vocab(main, 'chunk_fts', 'row')`).
   - Preserve exact `3` read-only MCP tools (`recall`, `qa`, `daemon_status`), adding optional `exclude_chunk_ids: list[int] = []` and `trace` diagnostics on `recall`.
3. **Zero Stopwords, Zero Title Heuristics, Zero Free Hyperparameters:**
   - Do **not** reintroduce `_STOP` lists, `_CJK_RE` spacing hacks, `title` heuristics (`title == ""` must achieve full accuracy), CSLS clone-penalizing offsets, MP eigenvalue crushing, or arbitrary scalar weights.
   - All scales derive strictly from corpus size $N$, covariance participation ratio $D_{\text{eff}} = (\sum \lambda_i)^2 / \sum \lambda_i^2$, and `temp.chunk_vocab` document frequencies:
     - $d_{\text{bridge}} = \lceil\sqrt{N}\rceil$, $d_{\text{query}} = \max(8, \lfloor N / 2 \rfloor)$, $f_{\max} = \min(6, \max(2, \lfloor \frac{1}{2}\log_2 N \rfloor))$, $|C_T| \le 3\lceil\sqrt{N}\rceil$.
     - Watershed stem boundary: $\text{df}(g_k) + 1 \le 2(d_{\text{peak}} + 1)$.
     - Laplace surprisal: $I(s) = -\ln\frac{\text{df}(s)+1}{N+2}$; spherical null surprisal: $-\ln \frac{1}{2}I_{1-c^2}(\frac{d-1}{2}, \frac{1}{2})$; Laplace stop prior: $\rho_{\text{stop}}(i) = \frac{1}{1 + d_{\text{out}}(i)}$ ($\tau \equiv 1.0$).
     - `RaBitQ` extreme-value Branch-and-Bound quantile: $z_N = \sqrt{2\ln N}$.
4. **Read-Only Query Purity & Lazy Tier-1 LM:** `recall` runs over `?mode=ro` (`PRAGMA query_only = ON; PRAGMA busy_timeout = 5000;`) with `0` SQLite writes, and `start_runtime` initializes `dspy.LM` lazily inside `ask()` so Tier-0 `recall` works when no LLM server is running.

---

## 1. File-by-File Implementation & Complete Code

### 1.1 `pyproject.toml` & `yaw/pipeline.py` (`MDL-1`)
1. Remove `spacy`, `en-core-web-sm` (`[tool.uv.sources]`), and `optuna` from `pyproject.toml`.
2. In `yaw/pipeline.py:_apply_fts_tables`, switch `chunk_fts` from default `unicode61` to `tokenize='trigram'`:
   ```sql
   CREATE VIRTUAL TABLE IF NOT EXISTS "chunk_fts"
   USING fts5(title, body, content='', contentless_delete=1, tokenize='trigram')
   ```
   *(Because `trigram` changes the FTS5 B-tree layout, bump schema version / rebuild `.yaw/yaw.db` cleanly on cutover.)*

### 1.2 `yaw/core/types.py` Contracts (`MDL-1`)
Ensure `EvidenceHop` and `SearchTrace` are present and exported in `yaw/core/__init__.py`:

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

### 1.3 Complete `yaw/retrieval.py` (`MDL-2`)

```python
"""Trigram-Watershed, Zero-Copy Stage-1 RaBitQ & Coverage-Lattice Gibbs Retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import betainc

from yaw.core.types import ChunkPointer, EvidenceHop, HydratedPassage, SearchTrace, YawRuntimeError

_NO_ISLAND = (1, 1)
_CJK_CH_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]")


@dataclass(frozen=True, slots=True)
class VectorSearch:
    """~100B/chunk Stage-1 RaBitQ view (mmap-backed when on disk; ZERO float32[N, D] arrays in RAM)."""

    chunk_ids: tuple[int, ...]
    id_to_pos: dict[int, int]
    pointers: tuple[ChunkPointer, ...]
    dim: int
    mu: NDArray[np.float32]
    d_eff: float
    rot: NDArray[np.float32]
    rabitq_bits: NDArray[np.uint64]
    rabitq_s: NDArray[np.float16]
    rabitq_ones: NDArray[np.uint16]
    corpus_root: Path | None = None

    @property
    def size(self) -> int:
        return len(self.chunk_ids)


def extract_text_features(title: str, body: str) -> tuple[str, str]:
    """Compatibility stub until tokens_json/entities_json auxiliary columns are dropped."""
    del title, body
    return "{}", "[]"


def _alnum_grams(s: str) -> list[str]:
    return [s[i : i + 3] for i in range(len(s) - 2) if s[i : i + 3].isalnum()] if len(s) >= 3 else []


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


def build_vector_search(conn: sqlite3.Connection, corpus_root: Path | None = None) -> VectorSearch:
    db_file = _db_file_path(conn)
    root = corpus_root or (db_file.parent.parent if db_file and db_file.parent.name == ".yaw" else None)
    ptr_rows = conn.execute(
        "SELECT chunk_id, rel_path, source_hash, start_line, end_line, "
        "char_start, char_end, ast_island_start, ast_island_end, title "
        "FROM chunks ORDER BY chunk_id ASC"
    ).fetchall()
    if not ptr_rows:
        z = np.zeros(0, dtype=np.float32)
        z2 = np.zeros((0, 0), dtype=np.float32)
        return VectorSearch(
            (), {}, (), 0, z, 8.0, z2,
            np.zeros((0, 0), dtype=np.uint64), np.zeros(0, dtype=np.float16), np.zeros(0, dtype=np.uint16), root,
        )

    cids = tuple(int(r[0]) for r in ptr_rows)
    ptrs = tuple(
        ChunkPointer(
            int(r[0]), str(r[1]), str(r[2]), int(r[3]), int(r[4]),
            int(r[5]), int(r[6]), int(r[7]), int(r[8]), str(r[9]),
        )
        for r in ptr_rows
    )
    n = len(cids)
    max_cid = cids[-1]
    stage1_dir = (db_file.parent / "stage1") if db_file else None

    # Fast zero-copy mmap attach when .yaw/stage1/ matches (count, max_chunk_id):
    if stage1_dir is not None and (stage1_dir / "meta.json").is_file():
        meta = json.loads((stage1_dir / "meta.json").read_text("utf-8"))
        if meta.get("n") == n and meta.get("max_cid") == max_cid:
            dim = int(meta["dim"])
            return VectorSearch(
                cids, {c: i for i, c in enumerate(cids)}, ptrs, dim,
                np.load(stage1_dir / "mu.npy"), float(meta["d_eff"]), _make_rotation(dim),
                np.load(stage1_dir / "bits.npy", mmap_mode="r"),
                np.load(stage1_dir / "s.npy", mmap_mode="r"),
                np.load(stage1_dir / "ones.npy", mmap_mode="r"),
                root,
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
        d_eff = float(np.clip((lam.sum() ** 2) / float(np.sum(lam**2)), 8.0, float(dim)))

    iso = _center_normalize(raw, mu)
    rot = _make_rotation(dim)
    x_rot = iso @ rot
    rb_s = np.maximum((np.sum(np.abs(x_rot), axis=1) / math.sqrt(float(max(1, dim)))).astype(np.float16), np.float16(1e-4))
    b_u8 = np.packbits(x_rot >= 0.0, axis=1, bitorder="little")
    if rem := (b_u8.shape[1] % 8):
        b_u8 = np.pad(b_u8, ((0, 0), (0, 8 - rem)))
    rb_bits = np.ascontiguousarray(b_u8).view(np.uint64)
    rb_ones = np.bitwise_count(rb_bits).sum(axis=1, dtype=np.uint16)
    del raw, iso, x_rot

    if stage1_dir is not None:
        try:
            stage1_dir.mkdir(parents=True, exist_ok=True)
            for fname, arr in (("mu.npy", mu), ("bits.npy", rb_bits), ("s.npy", rb_s), ("ones.npy", rb_ones)):
                mm = np.lib.format.open_memmap(stage1_dir / fname, mode="w+", dtype=arr.dtype, shape=arr.shape)
                mm[:] = arr
                mm.flush()
            (stage1_dir / "meta.json").write_text(
                json.dumps({"n": n, "max_cid": max_cid, "dim": dim, "d_eff": d_eff}), encoding="utf-8"
            )
        except OSError:
            pass  # Read-only filesystem sandbox; keep in-memory Stage-1 arrays

    return VectorSearch(cids, {c: i for i, c in enumerate(cids)}, ptrs, dim, mu, d_eff, rot, rb_bits, rb_s, rb_ones, root)


def _rabitq_estimate(vs: VectorSearch, q_unit: NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    qr = q_unit @ vs.rot
    qmin, qmax = float(qr.min()), float(qr.max())
    spn = max(qmax - qmin, 1e-8)
    qu4 = np.clip(np.round((qr - qmin) * (15.0 / spn)), 0, 15).astype(np.uint8)
    dot_bq = np.zeros(vs.size, dtype=np.int32)
    for pl in range(4):
        pu8 = np.packbits((qu4 >> pl) & 1, bitorder="little")
        if rem := (len(pu8) % 8):
            pu8 = np.pad(pu8, (0, 8 - rem))
        pu64 = pu8.view(np.uint64)
        dot_bq += np.bitwise_count(vs.rabitq_bits & pu64[None, :]).sum(axis=1, dtype=np.int32) << pl
    sbq = qmin * vs.rabitq_ones.astype(np.float32) + (spn / 15.0) * dot_bq.astype(np.float32)
    sf = vs.rabitq_s.astype(np.float32)
    est = (((2.0 * sbq - float(qr.sum())) / math.sqrt(float(max(1, vs.dim)))) / sf).astype(np.float32)
    z_n = math.sqrt(2.0 * math.log(max(2, vs.size)))
    eps = ((np.sqrt(np.maximum(0.0, 1.0 - sf * sf)) / sf) * (z_n / math.sqrt(max(1, vs.dim - 1)))).astype(np.float32)
    return est, eps


def _rabitq_bnb_top(vs: VectorSearch, conn: sqlite3.Connection, q_unit: NDArray[np.float32], k_want: int) -> list[int]:
    est, eps = _rabitq_estimate(vs, q_unit)
    k_c = min(vs.size, max(1, k_want))
    lb = float(np.partition(est - eps, -k_c)[-k_c])
    surv = np.flatnonzero((est + eps) >= lb)
    if len(surv) > 4 * k_c:
        surv = surv[np.argpartition(-est[surv], 4 * k_c)[: 4 * k_c]]
    surv_cids = [vs.chunk_ids[int(p)] for p in surv]
    rows = conn.execute(
        f"SELECT chunk_id, embedding FROM chunks WHERE chunk_id IN ({','.join('?' * len(surv_cids))})", surv_cids
    ).fetchall()
    vecs = _center_normalize(np.vstack([np.frombuffer(r[1], dtype="<f4") for r in rows]), vs.mu)
    order = np.argsort(-(vecs @ q_unit))[:k_c]
    return [int(rows[int(i)][0]) for i in order]


def _watershed_trigram_stems(
    text_low: str,
    gdf: dict[str, int],
    conn: sqlite3.Connection,
    n_tot: int,
    min_df: int,
    max_df: int,
    limit: int,
    match_cache: dict[str, int],
    anchor_positions: list[int] | None = None,
) -> list[tuple[str, float, int]]:
    """1D Watershed Stemmer over FTS5 Trigrams (cuts suffixes/compounds/particles at df+1 > 2*(d_peak+1))."""
    length = len(text_low) - 2
    if length <= 0:
        return []
    dfs: list[int] = []
    valid: list[bool] = []
    for i in range(length):
        g = text_low[i : i + 3]
        if g.isalnum():
            d = gdf.get(g, 0)
            ok = min_df <= d <= max_df
            dfs.append(d if ok else n_tot + 1)
            valid.append(ok)
        else:
            dfs.append(n_tot + 1)
            valid.append(False)

    used = [False] * length
    if anchor_positions:
        order = sorted(
            (i for i in range(length) if valid[i]),
            key=lambda i: -math.log((n_tot + 2.0) / (dfs[i] + 1.0))
            / math.log(math.e + min(abs(i - ap) for ap in anchor_positions)),
        )
    else:
        order = sorted((i for i in range(length) if valid[i]), key=lambda i: dfs[i])

    raw_cands: list[tuple[float, str]] = []
    seen_s: set[str] = set()
    max_alpha = int(math.ceil(math.log2(max(4, n_tot)) * 2))
    for k_star in order:
        if used[k_star]:
            continue
        d_peak = dfs[k_star]
        cap = 2 * (d_peak + 1) - 1
        is_cjk = bool(_CJK_CH_RE.search(text_low[k_star : k_star + 3]))
        max_ext = 1 if is_cjk else max_alpha
        lo = k_star
        while lo > 0 and (k_star - lo) < max_ext and not used[lo - 1] and valid[lo - 1] and dfs[lo - 1] <= cap:
            lo -= 1
        hi = k_star
        while hi + 1 < length and (hi - lo) < max_ext and not used[hi + 1] and valid[hi + 1] and dfs[hi + 1] <= cap:
            hi += 1
        for idx_u in range(lo, hi + 1):
            used[idx_u] = True
        sub = text_low[lo : hi + 3]
        if sub not in seen_s and (is_cjk or len(sub) >= 4):
            seen_s.add(sub)
            prox_div = math.log(math.e + min(abs(k_star - ap) for ap in anchor_positions)) if anchor_positions else 1.0
            raw_cands.append((math.log((n_tot + 2.0) / (d_peak + 1.0)) / prox_div, sub))
            if len(raw_cands) >= limit * 3:
                break

    verified: list[tuple[str, float, int, float]] = []
    for prio, sub in raw_cands:
        if sub not in match_cache:
            match_cache[sub] = int(
                conn.execute("SELECT COUNT(*) FROM chunk_fts WHERE chunk_fts MATCH ?", (f'"{sub}"',)).fetchone()[0]
            )
        exact_d = match_cache[sub]
        if min_df <= exact_d <= max_df:
            verified.append((sub, math.log((n_tot + 2.0) / (exact_d + 1.0)), exact_d, prio))
    verified.sort(key=lambda x: (-x[3], -x[1], -len(x[0])))
    return [(x[0], x[1], x[2]) for x in verified[:limit]]


def _gibbs_lattice_pass(
    seed_log_pi: NDArray[np.float64],
    node_masks: NDArray[np.uint8],
    k_mat: NDArray[np.float64],
    node_log_rho_stop: NDArray[np.float64],
    facet_nats: NDArray[np.float64],
    node_resid_nats: NDArray[np.float64],
    init_mask: int = 0,
    horizon: int = 4,
) -> tuple[float, float, NDArray[np.float64], NDArray[np.float64], tuple[int, ...], int]:
    vc, fc = len(seed_log_pi), len(facet_nats)
    mc = 1 << fc
    masks = np.arange(mc, dtype=np.int32)
    or_tab = node_masks[:, None].astype(np.int32) | masks[None, :]
    v_idx = np.arange(vc, dtype=np.int32)[:, None]
    flat_or = (v_idx * mc + or_tab).ravel()

    bit_mat = ((masks[:, None] >> np.arange(fc, dtype=np.int32)) & 1).astype(np.float64) if fc else np.zeros((1, 0))
    u_term = (bit_mat @ facet_nats)[None, :] + node_resid_nats[:, None]
    u_sh = float(u_term.max())
    g_h = np.exp(u_term - u_sh)
    g = np.exp((node_log_rho_stop[:, None] + u_term) - u_sh)

    b = np.empty((horizon + 1, vc, mc), dtype=np.float64)
    b_sh = np.empty(horizon + 1, dtype=np.float64)
    b[horizon], b_sh[horizon] = g_h, u_sh
    for h in range(horizon - 1, -1, -1):
        cont = k_mat @ b[h + 1, v_idx, or_tab]
        sh = max(u_sh, float(b_sh[h + 1]))
        raw = g * np.exp(u_sh - sh) + cont * np.exp(b_sh[h + 1] - sh)
        pk = float(raw.max())
        b[h] = raw / pk if pk > 0.0 else 0.0
        b_sh[h] = sh + math.log(pk) if pk > 0.0 else -np.inf

    sm = np.bitwise_or(init_mask, node_masks.astype(np.int32))
    pi_sh = float(seed_log_pi.max())
    pi_s = np.exp(seed_log_pi - pi_sh)
    z0 = pi_s * b[0, np.arange(vc), sm]
    log_z = pi_sh + float(b_sh[0]) + math.log(max(float(z0.sum()), 1e-300))
    log_z0 = pi_sh + u_sh + math.log(max(float((pi_s * g[np.arange(vc), sm]).sum()), 1e-300))

    a = np.zeros((horizon + 1, vc, mc), dtype=np.float64)
    a_sh = np.full(horizon + 1, -np.inf, dtype=np.float64)
    a[0, np.arange(vc), sm], a_sh[0] = pi_s, pi_sh
    kt = np.ascontiguousarray(k_mat.T)
    e_occ = np.zeros((vc, vc), dtype=np.float64)
    s_pr = np.zeros((vc, mc), dtype=np.float64)

    for h in range(horizon):
        if not np.isfinite(a_sh[h]):
            break
        s_pr += (a[h] * g) * np.exp(a_sh[h] + u_sh - log_z)
        bl = b[h + 1, v_idx, or_tab]
        e_occ += k_mat * ((a[h] @ bl.T) * np.exp(a_sh[h] + b_sh[h + 1] - log_z))
        nxt = np.bincount(flat_or, weights=(kt @ a[h]).ravel(), minlength=vc * mc).reshape(vc, mc)
        pk = float(nxt.max())
        if pk > 0.0:
            a[h + 1], a_sh[h + 1] = nxt / pk, a_sh[h] + math.log(pk)
    if np.isfinite(a_sh[horizon]):
        s_pr += (a[horizon] * g_h) * np.exp(a_sh[horizon] + u_sh - log_z)

    cur = int(np.argmax(z0))
    cm = int(sm[cur])
    route = [cur]
    for h in range(horizon):
        w_st = float(g[cur, cm]) * np.exp(u_sh - b_sh[h + 1]) if np.isfinite(b_sh[h + 1]) else 1.0
        cand_m = cm | node_masks.astype(np.int32)
        w_go = k_mat[cur] * b[h + 1, np.arange(vc), cand_m]
        for pr in route:
            w_go[pr] = 0.0
        jb = int(np.argmax(w_go))
        if w_go[jb] <= w_st or w_go[jb] <= 0.0:
            break
        route.append(jb)
        cm, cur = int(cand_m[jb]), jb

    return log_z, log_z0, s_pr.sum(1) + e_occ.sum(1), e_occ, tuple(route), cm


def _component_completeness(chosen: list[int], w_cond: NDArray[np.float64], n_masks: NDArray[np.uint8], f_nats: NDArray[np.float64]) -> float:
    tot = float(f_nats.sum())
    if tot <= 1e-9 or not chosen:
        return 0.0
    fc = len(f_nats)
    c_set = set(chosen)
    seen: set[int] = set()
    best = 0.0
    for start in chosen:
        if start in seen:
            continue
        cm = 0
        stack = [start]
        seen.add(start)
        while stack:
            u = stack.pop()
            cm |= int(n_masks[u])
            for v in c_set:
                if v not in seen and (w_cond[u, v] > 0.0 or w_cond[v, u] > 0.0):
                    seen.add(v)
                    stack.append(v)
        best = max(best, sum(float(f_nats[b]) for b in range(fc) if (cm >> b) & 1) / tot)
    return best


def solve_evidence_dag(
    vs: VectorSearch,
    query_embedding: NDArray[np.float32],
    *,
    query_text: str,
    conn: sqlite3.Connection,
    corpus_root: Path | None = None,
    exclude_chunk_ids: Sequence[int] = (),
    top_k: int = 5,
) -> tuple[list[tuple[int, float]], SearchTrace]:
    if vs.size == 0 or top_k <= 0:
        return [], SearchTrace()
    conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS temp.chunk_vocab USING fts5vocab(main, 'chunk_fts', 'row')")
    db_file = _db_file_path(conn)
    root = corpus_root or vs.corpus_root or (db_file.parent.parent if db_file and db_file.parent.name == ".yaw" else None)
    n_tot = vs.size
    qh = _center_normalize(query_embedding, vs.mu)

    d_bridge_max = max(4, int(math.ceil(math.sqrt(n_tot))))
    d_query_max = max(8, n_tot // 2)
    f_max = min(6, max(2, int(math.log2(max(4, n_tot)) // 2)))
    max_cand = 3 * d_bridge_max

    q_low = query_text.lower()
    match_cache: dict[str, int] = {}
    gram_df_cache: dict[str, int] = {}
    qg = list(dict.fromkeys(_alnum_grams(q_low)))
    if qg:
        gram_df_cache.update(
            conn.execute(f"SELECT term, doc FROM temp.chunk_vocab WHERE term IN ({','.join('?' * len(qg))})", qg).fetchall()
        )
    facets = _watershed_trigram_stems(q_low, gram_df_cache, conn, n_tot, 1, d_query_max, f_max, match_cache)
    for w_miss in [w for w in re.split(r"\W+", q_low) if len(w) >= 4 and "_" not in w]:
        if len(facets) < f_max and not any(st in w_miss or w_miss in st for st, _, _ in facets):
            if w_miss not in match_cache:
                match_cache[w_miss] = int(
                    conn.execute("SELECT COUNT(*) FROM chunk_fts WHERE chunk_fts MATCH ?", (f'"{w_miss}"',)).fetchone()[0]
                )
            if match_cache[w_miss] == 0:
                facets.append((w_miss, math.log(n_tot + 2.0), 0))

    live_facets = [f for f in facets if f[2] > 0]
    excl_set = {int(x) for x in exclude_chunk_ids if int(x) in vs.id_to_pos}
    if not live_facets:
        cids_only = [c for c in _rabitq_bnb_top(vs, conn, qh, top_k + len(excl_set)) if c not in excl_set][:top_k]
        return [(c, 1.0) for c in cids_only], SearchTrace()

    fts_ids: list[int] = []
    per_f = max(2, d_bridge_max // len(live_facets))
    for f_str, _, _ in live_facets:
        for (rc,) in conn.execute(
            "SELECT rowid FROM chunk_fts WHERE chunk_fts MATCH ? ORDER BY bm25(chunk_fts) ASC LIMIT ?",
            (f'"{f_str}"', per_f),
        ).fetchall():
            if int(rc) in vs.id_to_pos and int(rc) not in fts_ids:
                fts_ids.append(int(rc))
    top_d = _rabitq_bnb_top(vs, conn, qh, max(8, d_bridge_max // 2))
    seeds = list(dict.fromkeys([*excl_set, *fts_ids, *top_d]))[: d_bridge_max + len(excl_set)]

    visited_vecs: dict[int, NDArray[np.float32]] = {}
    low_texts: dict[int, str] = {}
    rare_sets: dict[int, frozenset[str]] = {}
    file_cache: dict[str, list[str]] = {}

    def _visit(cids: list[int]) -> list[int]:
        need = [c for c in cids if c in vs.id_to_pos and c not in visited_vecs][: max(0, max_cand - len(visited_vecs))]
        if not need:
            return []
        rows = conn.execute(
            f"SELECT chunk_id, embedding FROM chunks WHERE chunk_id IN ({','.join('?' * len(need))})", need
        ).fetchall()
        vecs = _center_normalize(np.vstack([np.frombuffer(r[1], dtype="<f4") for r in rows]), vs.mu)
        new_grams: list[str] = []
        chunk_grams: dict[int, frozenset[str]] = {}
        added: list[int] = []
        for idx_r, r in enumerate(rows):
            cid = int(r[0])
            visited_vecs[cid] = vecs[idx_r]
            ptr = vs.pointers[vs.id_to_pos[cid]]
            txt = ""
            if root is not None:
                if ptr.rel_posix_path not in file_cache:
                    fp = root / ptr.rel_posix_path
                    file_cache[ptr.rel_posix_path] = fp.read_text("utf-8", errors="ignore").splitlines() if fp.is_file() else []
                lines = file_cache[ptr.rel_posix_path]
                txt = "\n".join(lines[max(0, ptr.start_line - 1) : ptr.end_line])
            lt = txt.lower()
            low_texts[cid] = lt
            gs = frozenset(_alnum_grams(lt))
            chunk_grams[cid] = gs
            new_grams.extend(g for g in gs if g not in gram_df_cache)
            added.append(cid)
        unk = list(dict.fromkeys(new_grams))
        for k0 in range(0, len(unk), 400):
            batch = unk[k0 : k0 + 400]
            gram_df_cache.update(
                conn.execute(f"SELECT term, doc FROM temp.chunk_vocab WHERE term IN ({','.join('?' * len(batch))})", batch).fetchall()
            )
        for cid, gs in chunk_grams.items():
            rare_sets[cid] = frozenset(g for g in gs if 1 <= gram_df_cache.get(g, 0) <= d_bridge_max)
        return added

    frontier = [c for c in _visit(seeds) if c not in excl_set]
    bridge_df: dict[str, int] = {}
    q_fset = {f[0] for f in facets}
    spec_q = {f[0] for f in live_facets if f[2] <= d_bridge_max}
    entry_stems: dict[int, set[str]] = {c: set(spec_q) for c in frontier}

    for _ in range(2):
        dedup_f: list[int] = []
        union_r: set[str] = set().union(*(rare_sets[c] for c in excl_set if c in rare_sets))
        for fc in frontier:
            rs = rare_sets[fc]
            if not rs or not union_r or (len(rs - union_r) * 2 >= len(rs)):
                dedup_f.append(fc)
                union_r |= rs
            if len(dedup_f) >= 16:
                break
        cand_b: dict[str, tuple[float, int]] = {}
        for c in dedup_f:
            lt = low_texts[c]
            u_n = float(_null_nats(float(visited_vecs[c] @ qh), vs.d_eff))
            apos = [idx_p for st in entry_stems.get(c, ()) if (idx_p := lt.find(st)) != -1]
            for st, surp, mdf in _watershed_trigram_stems(lt, gram_df_cache, conn, n_tot, 2, d_bridge_max, f_max, match_cache, apos or None):
                if not any(st in qf or qf in st for qf in q_fset) and st not in bridge_df:
                    d_ch = min((abs(lt.find(st) - ap) for ap in apos), default=0) if apos else 0
                    prox = 1.0 / math.log(math.e + d_ch) if apos else 1.0
                    sc_b = surp * prox + (u_n / math.log(n_tot + 2.0))
                    if sc_b > cand_b.get(st, (-1.0, 0))[0]:
                        cand_b[st] = (sc_b, mdf)
        wave_b = [ph for ph, _ in sorted(cand_b.items(), key=lambda kv: -kv[1][0])[:10]]
        for ph in wave_b:
            bridge_df[ph] = cand_b[ph][1]
        if not wave_b or len(visited_vecs) >= max_cand:
            break
        per_b = max(2, (max_cand - len(visited_vecs)) // len(wave_b))
        wave_cids: list[int] = []
        for pb in wave_b:
            for (rc,) in conn.execute(
                "SELECT rowid FROM chunk_fts WHERE chunk_fts MATCH ? ORDER BY bm25(chunk_fts) ASC LIMIT ?",
                (f'"{pb}"', per_b),
            ).fetchall():
                ic = int(rc)
                entry_stems.setdefault(ic, set()).add(pb)
                wave_cids.append(ic)
        frontier = [c for c in _visit(wave_cids) if c not in excl_set]
        if not frontier:
            break

    c_list = sorted(c for c in visited_vecs if c not in excl_set)
    vc = len(c_list)
    if vc == 0:
        return [], SearchTrace()
    v_mat = np.vstack([visited_vecs[c] for c in c_list])
    rsets = [rare_sets[c] for c in c_list]

    fc_len = len(facets)
    f_nats = np.array([f[1] for f in facets], dtype=np.float64)
    n_masks = np.array(
        [sum((1 << bi) for bi, (ph, _, _) in enumerate(facets) if ph in low_texts[c_list[i]]) for i in range(vc)],
        dtype=np.uint8,
    )

    def _rnov(u_r: frozenset[str], v_r: frozenset[str]) -> float:
        return 1.0 - (len(u_r & v_r) / float(len(v_r))) if v_r else 1.0

    w_cond = np.zeros((vc, vc), dtype=np.float64)
    edge_anchor: dict[tuple[int, int], str] = {}
    for ph, dfb in bridge_df.items():
        members = [i for i in range(vc) if ph in low_texts[c_list[i]]]
        if len(members) < 2:
            continue
        base_c = math.log((n_tot + 2.0) / (dfb + 1.0)) / float(dfb - 1)
        for ui in members:
            for vi in members:
                if ui != vi:
                    w_cond[ui, vi] += base_c * _rnov(rsets[ui], rsets[vi])
                    edge_anchor.setdefault((ui, vi), ph)

    out_deg = (w_cond > 0.0).sum(axis=1)
    rho_stop = 1.0 / (1.0 + out_deg)
    row_w = w_cond.sum(axis=1)
    k_mat = np.where(row_w[:, None] > 0.0, (w_cond / np.maximum(row_w[:, None], 1e-12)) * (1.0 - rho_stop[:, None]), 0.0)

    # Seed stateful QR & rare-trigram exclusion from exclude_chunk_ids when exploring orthogonal branches:
    excl_vecs = [visited_vecs[c] for c in excl_set if c in visited_vecs]
    excl_rset: frozenset[str] = frozenset().union(*(rare_sets[c] for c in excl_set if c in rare_sets))
    q_rem = qh.copy()
    if excl_vecs:
        qx, _ = np.linalg.qr(np.column_stack(excl_vecs), mode="reduced")
        rem0 = qh - qx @ (qx.T @ qh)
        if np.linalg.norm(rem0) > 1e-6:
            q_rem = (rem0 / np.linalg.norm(rem0)).astype(np.float32)

    q_nats0 = _null_nats(v_mat @ q_rem, vs.d_eff)
    seed_n = q_nats0 + np.array([sum(f_nats[b] for b in range(fc_len) if (n_masks[i] >> b) & 1) for i in range(vc)])
    seed_lp = seed_n - float(np.max(seed_n))
    seed_lp -= float(np.log(np.sum(np.exp(seed_lp))))

    chosen: list[int] = []
    hops_out: list[EvidenceHop] = []
    cov_m = 0
    full_m = (1 << fc_len) - 1
    last_occ = np.zeros(vc, dtype=np.float64)
    first_bf = 0.0

    for pass_i in range(2):
        if len(chosen) >= top_k:
            break
        d_rem = max(8.0, vs.d_eff - len(chosen) - len(excl_vecs))
        r_nats = _null_nats(v_mat @ q_rem, d_rem)
        u_ch = excl_rset.union(*(rsets[j] for j in chosen))
        if u_ch:
            for iv in range(vc):
                r_nats[iv] *= _rnov(u_ch, rsets[iv])
        for p in chosen:
            r_nats[p] = 0.0

        log_z, log_z0, last_occ, e_occ, route, end_m = _gibbs_lattice_pass(
            seed_lp, n_masks, k_mat, np.log(rho_stop), f_nats, r_nats, init_mask=cov_m
        )
        if pass_i == 0:
            first_bf = float(log_z - log_z0)
        for s_i, node_i in enumerate(route):
            if node_i not in chosen and len(chosen) < top_k:
                chosen.append(node_i)
            if s_i > 0:
                ui, vi = route[s_i - 1], node_i
                hops_out.append(EvidenceHop(c_list[ui], c_list[vi], edge_anchor.get((ui, vi), ""), round(float(e_occ[ui, vi]), 4)))
        cov_m |= end_m
        if full_m > 0 and cov_m == full_m:
            break
        basis_vecs = [*excl_vecs, *(v_mat[i] for i in chosen)]
        qc, _ = np.linalg.qr(np.column_stack(basis_vecs), mode="reduced")
        rem = qh - qc @ (qc.T @ qh)
        if np.linalg.norm(rem) <= 1e-6:
            break
        q_rc = (rem / np.linalg.norm(rem)).astype(np.float32)
        if float(np.max(_null_nats(v_mat @ q_rc, d_rem))) < math.log(vc + 1.0):
            break
        q_rem = q_rc

    while len(chosen) < min(top_k, vc):
        u_ch = excl_rset.union(*(rsets[j] for j in chosen))
        best_ic = max((i for i in range(vc) if i not in chosen), key=lambda i: float(last_occ[i]) * _rnov(u_ch, rsets[i]))
        chosen.append(best_ic)

    cc_comp = _component_completeness(chosen, w_cond, n_masks, f_nats)
    unexplained = tuple(facets[b][0] for b in range(fc_len) if not ((cov_m >> b) & 1))
    used_anchors = {h.anchor for h in hops_out if h.anchor}
    dead_ends = tuple(b_st for b_st in list(bridge_df)[:6] if b_st not in used_anchors) if cc_comp < 0.99 else ()
    ranked = [(c_list[i], float(last_occ[i])) for i in chosen[:top_k]]
    return ranked, SearchTrace(tuple(hops_out), round(cc_comp, 4), round(first_bf, 4), unexplained, dead_ends)


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
        order = np.argsort(-est)[: min(vs.size, top_k)]
        return [(vs.chunk_ids[int(i)], float(est[int(i)])) for i in order]
    ranked, _ = solve_evidence_dag(vs, query_embedding, query_text=query_text, conn=conn, top_k=top_k)
    return ranked


def expand_neighbors_ranked(
    vs: VectorSearch, conn: sqlite3.Connection, chunk_id: int, *, top_k: int = 6
) -> list[tuple[int, float]]:
    pos = vs.id_to_pos.get(chunk_id)
    if pos is None or vs.size <= 1:
        return []
    row = conn.execute("SELECT embedding FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
    if row is None:
        return []
    u_hat = _center_normalize(np.frombuffer(row[0], dtype="<f4"), vs.mu)
    est, _ = _rabitq_estimate(vs, u_hat)
    scores = _null_nats(est, vs.d_eff)
    scores[pos] = -1.0
    order = np.argsort(-scores)[: min(top_k, vs.size - 1)]
    return [(vs.chunk_ids[int(i)], float(scores[int(i)])) for i in order if scores[int(i)] >= 0.0]


def lexical_search(vs: VectorSearch, query_text: str, conn: sqlite3.Connection, *, top_k: int = 8) -> list[tuple[int, float]]:
    if vs.size == 0 or top_k <= 0:
        return []
    conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS temp.chunk_vocab USING fts5vocab(main, 'chunk_fts', 'row')")
    q_low = query_text.lower()
    qg = list(dict.fromkeys(_alnum_grams(q_low)))
    if not qg:
        return []
    gdf = dict(conn.execute(f"SELECT term, doc FROM temp.chunk_vocab WHERE term IN ({','.join('?' * len(qg))})", qg).fetchall())
    stems = _watershed_trigram_stems(q_low, gdf, conn, vs.size, 1, max(8, vs.size // 2), 6, {})
    if not stems:
        return []
    rows = conn.execute(
        "SELECT rowid, -bm25(chunk_fts) FROM chunk_fts WHERE chunk_fts MATCH ? ORDER BY bm25(chunk_fts) ASC LIMIT ?",
        (" OR ".join(f'"{s[0]}"' for s in stems), min(vs.size, top_k)),
    ).fetchall()
    return [(int(cid), float(sc)) for cid, sc in rows if int(cid) in vs.id_to_pos]


def build_query_hints(vs: VectorSearch, query_text: str, *, limit: int = 5) -> list[str]:
    del vs, query_text, limit
    return []


def _hydrate_one(
    ptr: ChunkPointer, score: float, corpus_root: Path, file_cache: dict[str, tuple[str, list[str]]], *, include_island: bool = False
) -> HydratedPassage:
    rel = ptr.rel_posix_path
    full_path = (corpus_root / rel).resolve()
    if not full_path.is_relative_to(corpus_root.resolve()):
        raise YawRuntimeError("CONSISTENCY_ERROR", f"Pointer escaped corpus_root: {rel}")
    if rel not in file_cache:
        if not full_path.is_file():
            return HydratedPassage(ptr.chunk_id, score, "", ptr, stale_on_disk=True)
        raw_bytes = full_path.read_bytes()
        file_cache[rel] = (hashlib.sha256(raw_bytes).hexdigest(), raw_bytes.decode("utf-8", errors="replace").splitlines())
    else:
        digest, _ = file_cache[rel]
        if not full_path.is_file() or hashlib.sha256(full_path.read_bytes()).hexdigest() != digest:
            raise YawRuntimeError("CONSISTENCY_ERROR", f"File mutated mid-hydration: {rel}")

    actual_hash, lines = file_cache[rel]
    if actual_hash != ptr.source_hash:
        return HydratedPassage(ptr.chunk_id, score, "", ptr, stale_on_disk=True)
    use_island = include_island and (ptr.ast_island_start, ptr.ast_island_end) != _NO_ISLAND
    s_line = ptr.ast_island_start if use_island else ptr.start_line
    e_line = ptr.ast_island_end if use_island else ptr.end_line
    if s_line < 1 or e_line < s_line or e_line > max(1, len(lines)):
        return HydratedPassage(ptr.chunk_id, score, "", ptr, stale_on_disk=True)
    return HydratedPassage(ptr.chunk_id, score, "\n".join(lines[s_line - 1 : e_line]), ptr, stale_on_disk=False)


def hydrate_passage(vs: VectorSearch, chunk_id: int, corpus_root: Path, *, include_island: bool = False) -> HydratedPassage | None:
    pos = vs.id_to_pos.get(chunk_id)
    return None if pos is None else _hydrate_one(vs.pointers[pos], 1.0, corpus_root, {}, include_island=include_island)


def hydrate_ranked_passages(
    vs: VectorSearch, ranked: list[tuple[int, float]], corpus_root: Path, *, top_k: int | None = None, include_island: bool = False
) -> tuple[HydratedPassage, ...]:
    cache: dict[str, tuple[str, list[str]]] = {}
    items = ranked[:top_k] if top_k is not None else ranked
    return tuple(
        _hydrate_one(vs.pointers[vs.id_to_pos[cid]], sc, corpus_root, cache, include_island=include_island)
        for cid, sc in items
        if cid in vs.id_to_pos
    )
```

---

## 2. Mandatory Acceptance Gates (`MDL-3` & `MDL-4`)

1. **Adversarial Inflection, Compound, CJK, Clone & Stateful-Exclusion Suite (`tests/test_trigram_watershed_dag.py`):**
   - Stage `title == ""` chunks in `tokenize='trigram'` `chunk_fts` over `?mode=ro` (`PRAGMA query_only = ON`).
   - Verify inflected Finnish (`Äänekorvella` $\to$ `Äänekorven` $\to$ `Järvenlahtisolmuun` $\leftrightarrow$ `Järvenlahtisolmussa`), German compounds (`Prüfnordkanaldienst` $\to$ `Prüfnordkanal`), unspaced Chinese (`瀚海乾流`), and embedded log paths (`svc::Auriga0_Ledger`) all recover full 3-hop chains despite 5 clones + 4 distant red herrings + 3 decoy sinks.
   - Verify `exclude_chunk_ids=[branch_a_ids]` on Turn 2 returns an orthogonal second branch without repeating Turn 1's chunks.
   - Verify `trace.route_completeness` strictly separates connected 3-hop chains from dead-end `NullSink0` chains (`dead_end_bridges` non-empty).
2. **Real Corpus Benchmark (`G3`):** Run `[0, 10)` (`n = 40`) regression and `[10, 60)` (`n = 200`) across `2wikimultihopqa`, `musique`, `hotpotqa`, and `popqa`, reporting paired Wilcoxon $p$-values against `B5-S` Random Selection, `B5-A` Random Bridge Acquisition, and 0-Hop Hybrid.