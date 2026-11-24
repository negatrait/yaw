"""1-bit Extended SoA RaBitQ quantization, Beta-tail LUT, and spectral operators."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy import special

from .types import TANGENT_RHO_CLIP, Rotator

_SPECTRUM_SAMPLE_CAP = 2048
_RABITQ_SEED = 0x594157  # Fixed deterministic orthogonal rotator seed ("YAW").
_TANGENT_Z_MAX = math.sqrt(-math.log1p(-TANGENT_RHO_CLIP * TANGENT_RHO_CLIP))
_TANGENT_LUT_BINS = 16384

_BYTE_BITS_TBL: NDArray[np.float32] = (
    np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1, bitorder="little").astype(np.float32) * 2.0 - 1.0
)


def make_hadamard(n: int) -> NDArray[np.float32]:
    """Build a normalized n x n Walsh-Hadamard orthogonal matrix."""
    h = np.array([[1.0]], dtype=np.float32)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return (h / math.sqrt(n)).astype(np.float32)


def build_rotator(dim: int) -> Rotator:
    """Build the deterministic 3-round orthogonal rotator for dimension `dim`."""
    rng = np.random.default_rng(_RABITQ_SEED + dim)
    h64 = make_hadamard(64)
    signs = tuple(rng.choice([-1.0, 1.0], size=dim).astype(np.float32) for _ in range(3))
    perms = tuple(rng.permutation(dim) for _ in range(3))
    rem = dim % 64
    tail_q: NDArray[np.float32] | None = None
    if rem > 0:
        q_mat, r_mat = np.linalg.qr(rng.standard_normal((rem, rem)))
        diag = np.sign(np.diag(r_mat))
        diag[diag == 0.0] = 1.0
        tail_q = (q_mat * diag).astype(np.float32)
    return Rotator(dim=dim, h64=h64, signs=signs, perms=perms, tail_q=tail_q)


def rotate_vectors(rot: Rotator, arr: NDArray[np.float32]) -> NDArray[np.float32]:
    """Apply the 3-round orthogonal block-Hadamard + QR tail rotation."""
    is_1d = arr.ndim == 1
    y = arr[None, :].astype(np.float32, copy=True) if is_1d else arr.astype(np.float32, copy=True)
    n_blocks = rot.dim // 64
    head_end = n_blocks * 64
    for s, p in zip(rot.signs, rot.perms, strict=True):
        y = (y * s)[:, p]
        if n_blocks > 0:
            y[:, :head_end] = (y[:, :head_end].reshape(-1, n_blocks, 64) @ rot.h64).reshape(-1, head_end)
        if rot.tail_q is not None:
            y[:, head_end:] = y[:, head_end:] @ rot.tail_q
    return y[0] if is_1d else y


def transform_vectors(raw_vecs: NDArray[np.float32], mean_vec: NDArray[np.float32]) -> NDArray[np.float32]:
    """Mean-center and L2-normalize vectors with zero-norm protection."""
    centered = raw_vecs.astype(np.float32, copy=False) - mean_vec
    norms = np.linalg.norm(centered, axis=-1, keepdims=True).astype(np.float32)
    safe_norms = np.where(norms > 1e-12, norms, np.float32(1.0))
    return np.where(norms > 1e-12, centered / safe_norms, np.float32(0.0)).astype(np.float32, copy=False)


def compute_shannon_d_eff(centered_unit: NDArray[np.float32]) -> float:
    """Compute Shannon spectral entropy effective dimension over centered unit embeddings."""
    n, d = centered_unit.shape
    if n < 2:
        return float(max(2, d))
    sample = centered_unit
    if n > _SPECTRUM_SAMPLE_CAP:
        stride_idx = np.linspace(0, n - 1, _SPECTRUM_SAMPLE_CAP, dtype=np.int64)
        sample = centered_unit[stride_idx]
    m = sample.shape[0]
    _, svals, _ = np.linalg.svd(sample.astype(np.float64), full_matrices=False)
    eigvals = (svals * svals) / float(m)
    pos = eigvals[eigvals > 1e-12]
    total = float(pos.sum())
    if pos.size == 0 or total <= 1e-12:
        return float(max(2, min(d, max(2, n))))
    probs = pos / total
    ent = -float(np.sum(probs * np.log(probs)))
    return max(2.0, math.exp(ent))


def pack_rabitq_soa(unit_vecs: NDArray[np.float32], rot: Rotator) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
    """Quantize unit vectors into SoA uint8[W*8, N] bit-packed codes + ip_norm[N]."""
    n, d = unit_vecs.shape
    w = (d + 63) // 64
    if n == 0:
        return np.zeros((w * 8, 0), dtype=np.uint8), np.zeros(0, dtype=np.float32)
    rotated = rotate_vectors(rot, unit_vecs)
    bits = (rotated >= 0.0).astype(np.uint8)
    padded = np.zeros((n, w * 64), dtype=np.uint8)
    padded[:, :d] = bits
    codes = np.ascontiguousarray(np.packbits(padded, axis=1, bitorder="little").T, dtype=np.uint8)
    inv_sqrt_d = 1.0 / math.sqrt(d)
    signed_vertex = np.where(bits, inv_sqrt_d, -inv_sqrt_d).astype(np.float32)
    ip_norm = np.sum(rotated * signed_vertex, axis=1).astype(np.float32)
    ip_norm = np.maximum(ip_norm, inv_sqrt_d * 0.5).astype(np.float32)
    return codes, ip_norm


def rabitq_dot_lut(codes: NDArray[np.uint8], rotated_q: NDArray[np.float32], dim: int) -> NDArray[np.float32]:
    """Compute <x_bar_i, rotated_q> using vectorized byte-level 256-entry LUT indexing."""
    n = codes.shape[1]
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    active_bytes = (dim + 7) // 8
    inv_sqrt_d = np.float32(1.0 / math.sqrt(dim))
    q_scaled = np.zeros(active_bytes * 8, dtype=np.float32)
    q_scaled[:dim] = rotated_q * inv_sqrt_d
    lut = (q_scaled.reshape(active_bytes, 8) @ _BYTE_BITS_TBL.T).astype(np.float32, copy=False)
    acc = np.empty(n, dtype=np.float32)
    row_idx = np.arange(active_bytes)[:, None]
    block = 4096
    for s in range(0, n, block):
        e = min(n, s + block)
        acc[s:e] = lut[row_idx, codes[:active_bytes, s:e]].sum(axis=0)
    return acc


def unpack_1bit_signed(codes: NDArray[np.uint8], indices: Sequence[int], dim: int) -> NDArray[np.float32]:
    """Unpack 1-bit SoA codes into signed +/- 1/sqrt(dim) float32 vectors."""
    idx_arr = np.asarray(indices, dtype=np.intp).ravel()
    if idx_arr.size == 0:
        return np.zeros((0, dim), dtype=np.float32)
    active_bytes = (dim + 7) // 8
    sub = np.ascontiguousarray(codes[:active_bytes, idx_arr].T, dtype=np.uint8)
    bits = np.unpackbits(sub, axis=1, bitorder="little")[:, :dim]
    inv_sqrt_d = np.float32(1.0 / math.sqrt(dim))
    return ((bits.astype(np.float32) * np.float32(2.0) - np.float32(1.0)) * inv_sqrt_d).astype(np.float32, copy=False)


def pack_doc_1bit_codes(
    codes: NDArray[np.uint8],
    dim: int,
    el_idx: NDArray[np.int32],
    pidx: NDArray[np.int32],
    nd: int,
) -> NDArray[np.uint8]:
    """Pack uint8[active_bytes, nd] doc 1-bit sign codes (0.0 ties broken by first eligible chunk)."""
    active_bytes = (dim + 7) // 8
    if nd <= 0 or el_idx.size == 0:
        return np.zeros((active_bytes, max(0, nd)), dtype=np.uint8)
    pidx_el = pidx[el_idx].astype(np.int32, copy=False)
    first_c = np.full(nd, -1, dtype=np.int32)
    first_c[pidx_el[::-1]] = el_idx[::-1]
    is_first = first_c[pidx_el] == el_idx
    w = np.where(is_first, 3, 2).astype(np.int32)
    m_doc = sp.csr_array(
        (w, (pidx_el, np.arange(el_idx.size, dtype=np.int32))), shape=(nd, el_idx.size), dtype=np.int32
    )
    doc_codes = np.zeros((active_bytes, nd), dtype=np.uint8)
    b_step = 8
    for b0 in range(0, active_bytes, b_step):
        b1 = min(active_bytes, b0 + b_step)
        d0 = b0 * 8
        d1 = min(dim, b1 * 8)
        w_d = d1 - d0
        sub = np.ascontiguousarray(codes[b0:b1, el_idx].T, dtype=np.uint8)
        bits = np.unpackbits(sub, axis=1, bitorder="little")[:, :w_d]
        signed = bits.astype(np.int32) * 2 - 1
        sums = m_doc @ signed
        doc_bits = (sums > 0).astype(np.uint8)
        pad_w = (b1 - b0) * 8
        if w_d < pad_w:
            padded = np.zeros((nd, pad_w), dtype=np.uint8)
            padded[:, :w_d] = doc_bits
            doc_bits = padded
        doc_codes[b0:b1, :] = np.packbits(doc_bits, axis=1, bitorder="little").T
    return doc_codes


def compute_edge_ham_cos(
    codes: NDArray[np.uint8],
    doc_codes: NDArray[np.uint8],
    f_rows: NDArray[np.int32],
    f_cols: NDArray[np.int32],
    dim: int,
) -> NDArray[np.float32]:
    """Compute spherical edge cosines cos(pi * hamming(u, d) / dim) via bitwise XOR + popcount."""
    m_edges = int(f_rows.size)
    if m_edges == 0:
        return np.zeros(0, dtype=np.float32)
    active_bytes = (dim + 7) // 8
    h_ud = np.empty(m_edges, dtype=np.int32)
    rem = dim % 8
    last_mask = np.uint8((1 << rem) - 1) if rem != 0 else np.uint8(0xFF)
    block = 16384
    for s in range(0, m_edges, block):
        e = min(m_edges, s + block)
        xb = np.bitwise_xor(codes[:active_bytes, f_rows[s:e]], doc_codes[:active_bytes, f_cols[s:e]])
        if rem != 0:
            xb[active_bytes - 1] &= last_mask
        h_ud[s:e] = np.bitwise_count(xb).sum(axis=0, dtype=np.int32)
    return np.cos(math.pi * h_ud.astype(np.float64) / float(dim)).astype(np.float32)


def grouped_softmax(group_ids: NDArray[np.int32], logits: NDArray[np.float64], n_groups: int) -> NDArray[np.float32]:
    """Evaluate numerically stabilized per-group softmax over flat COO edge logits."""
    if group_ids.size == 0:
        return np.zeros(0, dtype=np.float32)
    g_max = np.full(n_groups, -np.inf, dtype=np.float64)
    np.maximum.at(g_max, group_ids, logits)
    ex = np.exp(logits - g_max[group_ids])
    g_sum = np.bincount(group_ids, weights=ex, minlength=n_groups)
    return (ex / np.maximum(g_sum[group_ids], 1e-30)).astype(np.float32)


def parametric_null_nats_vec(cos_vals: NDArray[np.float64], d_eff: float) -> NDArray[np.float64]:
    """Evaluate exact two-sided regularized incomplete Beta spherical-null surprisal nats."""
    out = np.zeros_like(cos_vals, dtype=np.float64)
    pos = cos_vals > 0.0
    if not np.any(pos):
        return out
    c = np.minimum(cos_vals[pos], 1.0 - 1e-15)
    a = max(0.5, 0.5 * (d_eff - 1.0))
    one_m_c2 = np.maximum(1e-30, 1.0 - c * c)
    p_two = special.betainc(a, 0.5, one_m_c2)
    asymp = a * np.log(one_m_c2) - math.log(a) - float(special.betaln(a, 0.5))
    vals = np.where(p_two <= 1e-300, -asymp, -np.log(np.maximum(p_two, 1e-300)))
    out[pos] = np.maximum(0.0, vals)
    return out


def build_beta_nats_lut(
    d_eff: float, n_bins: int = _TANGENT_LUT_BINS
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Precompute a piecewise-linear Beta-tail surprisal LUT parameterized by z = sqrt(-log1p(-c^2))."""
    z_grid = np.linspace(0.0, _TANGENT_Z_MAX, n_bins + 1, dtype=np.float64)
    c_grid = np.sqrt(-np.expm1(-z_grid * z_grid))
    lut = parametric_null_nats_vec(c_grid, d_eff)
    return lut, np.diff(lut), float(n_bins) / _TANGENT_Z_MAX


def eval_signed_beta_nats_lut(
    rho: NDArray[np.float64],
    lut: NDArray[np.float64],
    lut_diff: NDArray[np.float64],
    scale: float,
    inv_t_path: float,
) -> NDArray[np.float64]:
    """Interpolate signed Beta-tail surprisal logits from the precomputed LUT."""
    z = np.sqrt(-np.log1p(-rho * rho))
    n_bins = lut_diff.size
    pos = np.minimum(z * scale, float(n_bins))
    idx = np.minimum(pos.astype(np.int32), n_bins - 1)
    val = (lut[idx] + (pos - idx) * lut_diff[idx]) * inv_t_path
    return np.copysign(val, rho)


def sigmoid(x: float) -> float:
    """Numerically clamped logistic sigmoid."""
    if x >= 30.0:
        return 1.0
    if x <= -30.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))
