"""Frozen data contracts, regex constants, and index structures for read-only retrieval."""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from yaw.core import ChunkPointer

CACHE_LOCK = threading.Lock()
CACHE_MAX_ENTRIES = 4_096

CJK_RANGE = r"\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff\uac00-\ud7af"
CJK_RUN_RE = re.compile(f"[{CJK_RANGE}]+")
WORD_TOK_RE = re.compile(rf"[^\W_{CJK_RANGE}]+(?:[+#._-][^\W_{CJK_RANGE}]+|[+]{{2}}|#)?")
WS_RUN_RE = re.compile(r"\s+")
POSS_RE = re.compile(r"(?<=[0-9a-zA-Z\u00c0-\u024f])['’]s\b")

TANGENT_COS_CLIP = 0.9999
TANGENT_RHO_CLIP = 0.999


@dataclass(frozen=True, slots=True)
class Rotator:
    """Lightweight deterministic orthogonal rotator for 1-bit SoA RaBitQ."""

    dim: int
    h64: NDArray[np.float32]
    signs: tuple[NDArray[np.float32], ...]
    perms: tuple[NDArray[np.intp], ...]
    tail_q: NDArray[np.float32] | None


@dataclass(frozen=True, slots=True)
class CitationGraph:
    """Compact int32 CSR title-reference and 1-bit PMI bipartite citation graph."""

    fi: NDArray[np.int32]
    fx: NDArray[np.int32]
    ri: NDArray[np.int32]
    rx: NDArray[np.int32]
    dc_indptr: NDArray[np.int32]
    dc_indices: NDArray[np.int32]
    pmi_indptr: NDArray[np.int32]
    pmi_indices: NDArray[np.int32]
    cpmi_indptr: NDArray[np.int32]
    cpmi_indices: NDArray[np.int32]
    pmi_terms: tuple[str, ...]
    prop_nouns: frozenset[str]
    doc_titles: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DiffusionOperators:
    """Static CSR citation/PMI diffusion operators and 16,384-bin Beta-tail LUT buffers."""

    d_own: sp.csr_array | None
    m_fwd_t: sp.csr_array | None
    r_rev: sp.csr_array | None
    d_id_t: sp.csr_array | None
    p2c: sp.csr_array | None
    c2p_t: sp.csr_array | None
    self_pmi: NDArray[np.float32] | None
    pmi_scale: NDArray[np.float32] | None
    inv_m: NDArray[np.float32]
    edge_f_rows: NDArray[np.int32] | None = None
    edge_f_cols: NDArray[np.int32] | None = None
    edge_c_ud: NDArray[np.float32] | None = None
    edge_inv_s_ud: NDArray[np.float32] | None = None
    edge_fwd_indptr: NDArray[np.int32] | None = None
    edge_t_order: NDArray[np.int32] | None = None
    edge_t_indptr: NDArray[np.int32] | None = None
    edge_t_indices: NDArray[np.int32] | None = None
    edge_beta_lut: NDArray[np.float64] | None = None
    edge_beta_lut_diff: NDArray[np.float64] | None = None
    edge_beta_scale: float = 0.0
    edge_beta_d_tan: float = 0.0
    edge_nd: int = 0


@dataclass(slots=True)
class RetrievalIndex:
    """Read-only M_FINAL_1B retrieval index with 1-bit SoA RaBitQ codes, citation graph, and caches."""

    chunk_ids: tuple[int, ...]
    id_to_pos: dict[int, int]
    pointers: tuple[ChunkPointer, ...]
    path_idx: NDArray[np.uint32]
    distinct_paths: tuple[str, ...]
    dim: int
    mean_vec: NDArray[np.float32]
    d_eff: float
    codes: NDArray[np.uint8]
    ip_norm: NDArray[np.float32]
    rotator: Rotator | None
    doc_titles: dict[str, str]
    title_by_first_word: dict[str, tuple[int, dict[tuple[str, ...], tuple[int, ...]]]]
    citation_graph: CitationGraph | None
    diffusion_ops: DiffusionOperators | None
    vocab_doc_cache: dict[str, int] = field(default_factory=dict)
    resolved_path_cache: dict[tuple[Path, str], Path] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.chunk_ids)

    @property
    def num_pmi_terms(self) -> int:
        return len(self.citation_graph.pmi_terms) if self.citation_graph is not None else 0

    @property
    def num_citation_edges(self) -> int:
        return int(self.citation_graph.fx.size) if self.citation_graph is not None else 0


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Ranked chunk hits, certainty metrics, next candidates, and evidence links."""

    ranked: tuple[tuple[int, float], ...]
    log_bf: float
    max_chunk_score: float
    top1_cosine: float
    d_eff: float
    n_candidates: int
    coverage: float = 0.0
    next_candidates: tuple[tuple[int, float], ...] = ()
    links: tuple[tuple[Literal["cite", "pmi"], int, int, str], ...] = ()


@dataclass(frozen=True, slots=True)
class IndexArtifacts:
    """Materialized 1-bit SoA RaBitQ and compressed CitationGraph blobs for SQLite persistence."""

    n_chunks: int
    dim: int
    vec_sha256: str
    source_sha256: str
    n_words: int
    codes_blob: bytes
    ip_norm_blob: bytes
    mean_blob: bytes
    d_eff_shannon: float
    doc_titles_blob: bytes
    graph_blob: bytes


def empty_retrieval_index() -> RetrievalIndex:
    """Return an empty zero-chunk RetrievalIndex."""
    return RetrievalIndex(
        chunk_ids=(),
        id_to_pos={},
        pointers=(),
        path_idx=np.zeros(0, dtype=np.uint32),
        distinct_paths=(),
        dim=0,
        mean_vec=np.zeros(0, dtype=np.float32),
        d_eff=0.0,
        codes=np.zeros((0, 0), dtype=np.uint8),
        ip_norm=np.zeros(0, dtype=np.float32),
        rotator=None,
        doc_titles={},
        title_by_first_word={},
        citation_graph=None,
        diffusion_ops=None,
    )
