"""Entity Disambiguation and Canonicalization (EDC) Two-Tier Convergence Engine.

Implements the clean Two-Tier Convergence Protocol resolving surface form mentions into canonical nodes:
- Tier 1: Direct match (exact slug, active alias in `aliases`, or `InMemoryLookupKB`).
- Tier 2: Near match (in-process vector similarity > `CANON_SIM_THRESHOLD` (0.80) using static GloVe vectors or caller embeddings). Consolidates into existing canonical node via `_record_alias()`.
- Fallback: Novel canonical node creation via `_create_canonical_node()`.

Eliminates Tier 3 LLM pairwise adjudication: 0 Ollama `/api/chat` calls during canonicalization.
Eliminates O(N) SequenceMatcher loops and in-memory BLOB deserialization scan ladders.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from yaw import config
from yaw.nlp.components import get_base_spacy
from yaw.storage.db import (
    get_canonical_id,
    insert_alias,
    insert_leaf_node,
)
from yaw.utils import normalize_token, slugify

logger = logging.getLogger(__name__)


class TierEnum(str, Enum):
    """Execution tier where convergence resolved."""

    TIER_1_EXACT = "tier_1_exact"
    TIER_1_LEXICAL = "tier_1_lexical"  # Retained for backwards compatibility
    TIER_2_EMBEDDING = "tier_2_embedding"
    TIER_3_LLM = "tier_3_llm"  # Retained for backwards compatibility
    NEW_INDUCTION = "new_induction"
    FALLBACK_NEW = NEW_INDUCTION  # Retained for backwards compatibility alias


class MatchMethod(str, Enum):
    """Method recorded in aliases table or resolution result."""

    EXACT = "exact"
    ALIAS = "alias"
    LEXICAL = "lexical"
    AUTO = "auto"
    LLM = "llm"
    COMPILE = "compile"
    NEW = "new"


class AliasStatus(str, Enum):
    """Status recorded in aliases table."""

    ACTIVE = "active"
    CONTESTED = "contested"


@dataclass(frozen=True)
class CanonicalVerdict:
    """Record of an algorithmic or offline canonicalization verdict."""

    surface_id: str
    kind: str
    candidate_id: str
    cosine: float
    lexical: float
    verdict: str  # 'match' | 'distinct'
    phase: str = "ingest"
    decided_at: float = field(default_factory=time.time)
    confidence: float = 1.0
    reasoning: str | None = None
    rationale: str | None = None

    def __post_init__(self) -> None:
        if self.rationale is not None and self.reasoning is None:
            object.__setattr__(self, "reasoning", self.rationale)
        elif self.reasoning is not None and self.rationale is None:
            object.__setattr__(self, "rationale", self.reasoning)


@dataclass(frozen=True)
class ResolutionResult:
    """Result of the Two-Tier Convergence Engine resolve() invocation."""

    canonical_id: str
    method: str
    status: str = "active"
    tier: TierEnum = TierEnum.TIER_1_EXACT
    score: float = 1.0
    is_new: bool = False
    verdict: CanonicalVerdict | None = None


# Backward compatibility alias for roadmap specification (Invariant 5A.1)
Resolution = ResolutionResult


def _safe_insert_alias(
    conn: sqlite3.Connection,
    alias_id: str,
    kind: str,
    canonical_id: str,
    method: str,
    score: float,
    status: str = "active",
    source_leaf_id: str | None = None,
    evidence: str | None = None,
) -> None:
    """Safely insert alias row via SSOT storage.db.insert_alias."""
    insert_alias(
        conn=conn,
        alias_id=alias_id,
        canonical_id=canonical_id,
        kind=kind,
        method=method,
        score=score,
        status=status,
        source_leaf_id=source_leaf_id,
        evidence=evidence,
    )


def _record_alias(
    conn: sqlite3.Connection,
    alias_id: str,
    canonical_id: str,
    kind: str = "entity",
    method: str = "auto",
    score: float = 1.0,
    status: str = "active",
    source_leaf_id: str | None = None,
    evidence: str | None = None,
) -> None:
    """Record an alias mapping via SSOT storage.db.insert_alias."""
    insert_alias(
        conn=conn,
        alias_id=alias_id,
        canonical_id=canonical_id,
        kind=kind,
        method=method,
        score=score,
        status=status,
        source_leaf_id=source_leaf_id,
        evidence=evidence,
    )


def _create_canonical_node(
    conn: sqlite3.Connection,
    surface_id: str,
    name: str,
    namespace: str,
    source_leaf_id: str | None = None,
    definition: str = "",
    embedding: list[float] | np.ndarray | None = None,
) -> None:
    """Persist novel canonical entity in leaf_nodes via SSOT storage.db.insert_leaf_node."""
    if namespace == "predicates" or surface_id.startswith("predicates:"):
        return
    insert_leaf_node(
        conn=conn,
        node_id=surface_id,
        name=name,
        namespace=namespace,
        definition=definition,
        embedding=embedding,
        source_leaf_id=source_leaf_id or "system:canonical_induction",
    )


def _node_id(concept_name: str, namespace: str = "concepts") -> str:
    """Return canonical entity URI {namespace}:{slug} matching ^[a-z_]+:[a-z0-9_-]+$."""
    if ":" in concept_name:
        parts = concept_name.split(":", 1)
        ns_part = slugify(parts[0]).replace("-", "_")
        slug_part = slugify(parts[1])
        if ns_part and slug_part:
            return f"{ns_part}:{slug_part}"
    slug = slugify(concept_name)
    if not slug:
        slug = re.sub(r"[^a-z0-9_-]+", "-", normalize_token(concept_name)).strip("-")
    if not slug:
        slug = "unknown"
    ns = slugify(namespace).replace("-", "_") or "concepts"
    return f"{ns}:{slug}"


def _get_base_spacy() -> Any:
    """Acquire base spaCy pipeline cached in-process."""
    try:
        return get_base_spacy()
    except OSError:
        return None


def _get_or_compute_vector(
    name: str,
    embedding: list[float] | np.ndarray | None = None,
    nlp: Any | None = None,
) -> np.ndarray | None:
    """Extract or compute normalized vector representation. Zero external HTTP calls."""
    if embedding is not None and len(embedding) > 0:
        arr = np.asarray(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        return arr / norm if norm > 1e-12 else arr

    pipe = nlp if nlp is not None else _get_base_spacy()
    if pipe is None or getattr(pipe.vocab.vectors, "n_keys", 0) == 0:
        return None

    try:
        if name in pipe.vocab and pipe.vocab[name].has_vector:
            vec = np.asarray(pipe.vocab[name].vector, dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            if norm > 1e-12:
                return vec / norm

        doc = pipe(name)
        if doc.has_vector and doc.vector_norm > 1e-12:
            doc_vec = np.asarray(doc.vector, dtype=np.float32)
            return doc_vec / float(doc.vector_norm)
    except (ValueError, KeyError, AttributeError, TypeError) as spacy_err:
        logger.debug("spaCy vector retrieval failed for '%s': %s", name, spacy_err)

    return None


@dataclass(frozen=True, slots=True)
class SurfaceIdentity:
    surface_id: str
    slug: str
    namespace: str

    def __iter__(self):
        yield self.surface_id
        yield self.slug
        yield self.namespace


def _parse_surface_identity(
    name: str,
    kind: str = "entity",
    namespace: str = "concepts",
    proposed_ns: str | None = None,
) -> SurfaceIdentity:
    """Parse and normalize entity/predicate name into surface_id, slug, and effective namespace."""
    raw_ns = proposed_ns or namespace
    if kind == "predicate":
        effective_ns = (
            "predicates"
            if not raw_ns
            or raw_ns in ("concept", "concepts")
            or raw_ns not in config.TAXONOMY_NAMESPACES
            else raw_ns
        )
    else:
        effective_ns = (
            "concepts"
            if not raw_ns
            or raw_ns == "concept"
            or raw_ns not in config.TAXONOMY_NAMESPACES
            else raw_ns
        )

    if ":" in name and kind == "entity":
        parts = name.split(":", 1)
        ns_part = slugify(parts[0]).replace("-", "_")
        slug_part = slugify(parts[1])
        if ns_part and slug_part:
            if ns_part == "concept" or ns_part not in config.TAXONOMY_NAMESPACES:
                ns_part = "concepts"
            surface_id = f"{ns_part}:{slug_part}"
            return SurfaceIdentity(
                surface_id=surface_id, slug=slug_part, namespace=ns_part
            )

    slug = slugify(name)
    if not slug:
        import hashlib

        slug = "id-" + hashlib.md5(name.encode("utf-8")).hexdigest()[:8]

    if kind == "entity":
        surface_id = f"{effective_ns}:{slug}"
        return SurfaceIdentity(surface_id=surface_id, slug=slug, namespace=effective_ns)
    else:
        surface_id = slug
        return SurfaceIdentity(surface_id=surface_id, slug=slug, namespace=effective_ns)


def _lookup_active_alias(
    conn: sqlite3.Connection,
    surface_id: str,
    slug: str,
    name: str,
    kind: str,
) -> str | None:
    """Check active aliases by surface_id, slug, or raw name via SSOT storage.db.get_canonical_id."""
    queries = (
        [surface_id, slug, name.strip()]
        if kind == "entity"
        else [surface_id, name.strip()]
    )
    if kind == "entity" and surface_id.startswith("concepts:"):
        queries.append(f"concept:{slug}")
    for alias_key in queries:
        canon = get_canonical_id(conn, alias_key, kind=kind, fallback_to_alias=False)
        if canon is not None:
            return canon
    return None


def _lookup_kb(name: str, kb: Any) -> str | None:
    """Look up candidate from InMemoryLookupKB."""
    try:
        candidates = kb.get_candidates(name)
        if candidates:
            return str(candidates[0].entity_)
    except (AttributeError, KeyError, IndexError, TypeError) as kb_err:
        logger.debug("InMemoryLookupKB candidate lookup error: %s", kb_err)
    return None


def _lookup_predicate_direct(
    conn: sqlite3.Connection,
    name: str,
    slug: str,
) -> str | None:
    """Direct match against base predicates or leaf_relations."""
    pred_norm = slug.replace("-", "_").lower()
    pred_raw = name.strip()
    for bp in config.BASE_PREDICATES:
        if bp.lower() in (pred_norm, pred_raw.lower()):
            return bp
    try:
        row = conn.execute(
            """
            SELECT predicate FROM leaf_relations
            WHERE predicate = ? OR lower(predicate) = ? OR lower(replace(predicate, '-', '_')) = ?
            LIMIT 1
            """,
            (pred_raw, pred_raw.lower(), pred_norm),
        ).fetchone()
        if row and row[0]:
            return str(row[0])
    except sqlite3.OperationalError:
        pass
    return None


def _lookup_exact_node(
    conn: sqlite3.Connection,
    surface_id: str,
    slug: str,
    kind: str,
) -> str | None:
    """Direct match against leaf_nodes."""
    try:
        node_row = conn.execute(
            "SELECT node_id FROM leaf_nodes WHERE node_id = ? LIMIT 1",
            (surface_id,),
        ).fetchone()
        if not node_row and kind == "entity":
            node_row = conn.execute(
                "SELECT node_id FROM leaf_nodes WHERE node_id LIKE ? LIMIT 1",
                (f"%:{slug}",),
            ).fetchone()
        if node_row and node_row[0]:
            return str(node_row[0])
    except sqlite3.OperationalError:
        pass
    return None


def _resolve_tier1_direct(
    conn: sqlite3.Connection,
    surface_id: str,
    slug: str,
    name: str,
    kind: str,
    source_leaf_id: str | None,
    kb: Any | None = None,
) -> ResolutionResult | None:
    """Tier 1: Direct match via active alias table, exact leaf_nodes canonical match, or InMemoryLookupKB."""
    if kb is not None:
        cand_id = _lookup_kb(name, kb)
        if cand_id:
            return ResolutionResult(
                canonical_id=cand_id,
                method=MatchMethod.EXACT.value,
                tier=TierEnum.TIER_1_EXACT,
            )

    if kind == "predicate":
        matched_bp = _lookup_predicate_direct(conn, name, slug)
        if matched_bp is not None:
            if matched_bp != name.strip():
                _record_alias(
                    conn,
                    alias_id=name.strip(),
                    canonical_id=matched_bp,
                    kind="predicate",
                    method="lexical",
                    source_leaf_id=source_leaf_id,
                )
            return ResolutionResult(
                canonical_id=matched_bp,
                method=MatchMethod.EXACT.value,
                tier=TierEnum.TIER_1_EXACT,
            )

    canon_from_alias = _lookup_active_alias(conn, surface_id, slug, name, kind)
    if canon_from_alias:
        return ResolutionResult(
            canonical_id=canon_from_alias,
            method=MatchMethod.ALIAS.value,
            tier=TierEnum.TIER_1_EXACT,
        )

    exact_node = _lookup_exact_node(conn, surface_id, slug, kind)
    if exact_node:
        if exact_node != surface_id:
            _record_alias(
                conn,
                alias_id=surface_id,
                canonical_id=exact_node,
                kind=kind,
                method="lexical",
                source_leaf_id=source_leaf_id,
            )
        return ResolutionResult(
            canonical_id=exact_node,
            method=MatchMethod.EXACT.value,
            tier=TierEnum.TIER_1_EXACT,
        )

    return None


def _find_vec_nearest(
    conn: sqlite3.Connection,
    q_vec: np.ndarray,
    namespace: str | None = None,
) -> tuple[str | None, float]:
    """Query nearest candidate vector via sqlite-vec."""
    q_bytes = q_vec.astype(np.float32).tobytes()
    try:
        if namespace is not None:
            sql = """
                SELECT node_id, vec_distance_cosine(embedding, ?) AS distance
                FROM leaf_nodes
                WHERE embedding IS NOT NULL AND namespace = ?
                ORDER BY distance ASC
                LIMIT 1
            """
            params: tuple[Any, ...] = (q_bytes, namespace)
        else:
            sql = """
                SELECT node_id, vec_distance_cosine(embedding, ?) AS distance
                FROM leaf_nodes
                WHERE embedding IS NOT NULL
                ORDER BY distance ASC
                LIMIT 1
            """
            params = (q_bytes,)
        rows = conn.execute(sql, params).fetchall()
        if rows and rows[0][1] is not None:
            dist = float(rows[0][1])
            return str(rows[0][0]), max(0.0, min(1.0, 1.0 - dist))
    except (sqlite3.OperationalError, sqlite3.DatabaseError) as err:
        logger.debug("sqlite-vec distance query error: %s", err)
    return None, 0.0


def _find_spacy_entity_match(
    conn: sqlite3.Connection,
    q_vec: np.ndarray,
    pipe: Any,
    limit: int = 50,
) -> tuple[str | None, float]:
    """Find entity candidate via sqlite-vec or bounded in-process spaCy vector similarity."""
    if pipe is None or getattr(pipe.vocab.vectors, "n_keys", 0) == 0:
        return None, 0.0

    cand_id, cos = _find_vec_nearest(conn, q_vec)
    if cand_id is not None and cos > 0.0:
        return cand_id, cos

    best_cand_id: str | None = None
    best_cosine: float = 0.0
    try:
        nodes = conn.execute(
            "SELECT DISTINCT node_id, name FROM leaf_nodes WHERE name IS NOT NULL AND name != '' LIMIT ?",
            (limit,),
        ).fetchall()
        for nid, nm in nodes:
            cand_v = _get_or_compute_vector(str(nm or ""), nlp=pipe)
            if cand_v is not None and cand_v.shape == q_vec.shape:
                cos = max(0.0, min(1.0, float(np.dot(q_vec, cand_v))))
                if cos > best_cosine:
                    best_cosine = cos
                    best_cand_id = str(nid)
    except sqlite3.OperationalError:
        pass
    return best_cand_id, best_cosine


def _find_spacy_predicate_match(
    conn: sqlite3.Connection,
    q_vec: np.ndarray,
    pipe: Any,
) -> tuple[str | None, float]:
    """Scan predicate candidate names in SQLite via in-process spaCy vector similarity."""
    if pipe is None or getattr(pipe.vocab.vectors, "n_keys", 0) == 0:
        return None, 0.0

    best_cand_id: str | None = None
    best_cosine: float = 0.0
    try:
        preds = [
            str(r[0])
            for r in conn.execute(
                "SELECT DISTINCT predicate FROM leaf_relations"
            ).fetchall()
            if r[0]
        ]
        for bp in config.BASE_PREDICATES:
            if bp not in preds:
                preds.append(bp)
        for cand_pred in preds:
            cand_v = _get_or_compute_vector(cand_pred, nlp=pipe)
            if cand_v is not None and cand_v.shape == q_vec.shape:
                cos = max(0.0, min(1.0, float(np.dot(q_vec, cand_v))))
                if cos > best_cosine:
                    best_cosine = cos
                    best_cand_id = cand_pred
    except sqlite3.OperationalError:
        pass
    return best_cand_id, best_cosine


def _resolve_tier2_vector(
    conn: sqlite3.Connection,
    q_vec: np.ndarray,
    surface_id: str,
    kind: str,
    source_leaf_id: str | None,
    threshold: float,
    nlp: Any | None = None,
    namespace: str | None = None,
) -> ResolutionResult | None:
    """Tier 2: In-process vector similarity near match consolidating to existing canonical node."""
    best_cand_id, best_cosine = _find_vec_nearest(conn, q_vec, namespace=namespace)

    if best_cosine < threshold:
        pipe = nlp if nlp is not None else _get_base_spacy()
        if pipe is not None and getattr(pipe.vocab.vectors, "n_keys", 0) > 0:
            if kind == "entity":
                c_id, c_cos = _find_spacy_entity_match(conn, q_vec, pipe)
            else:
                c_id, c_cos = _find_spacy_predicate_match(conn, q_vec, pipe)
            if c_cos > best_cosine:
                best_cand_id, best_cosine = c_id, c_cos

    if best_cand_id is not None and best_cosine >= threshold:
        _record_alias(
            conn=conn,
            alias_id=surface_id,
            canonical_id=best_cand_id,
            kind=kind,
            method="auto",
            score=best_cosine,
            status="active",
            source_leaf_id=source_leaf_id,
        )
        return ResolutionResult(
            canonical_id=best_cand_id,
            method=MatchMethod.AUTO.value,
            status=AliasStatus.ACTIVE.value,
            tier=TierEnum.TIER_2_EMBEDDING,
            score=best_cosine,
            is_new=False,
        )

    if (
        best_cand_id is not None
        and best_cand_id != surface_id
        and (best_cosine >= config.CANON_SIM_THRESHOLD or best_cosine >= 0.80)
    ):
        _record_alias(
            conn=conn,
            alias_id=surface_id,
            canonical_id=best_cand_id,
            kind=kind,
            method="auto",
            score=best_cosine,
            status="contested",
            source_leaf_id=source_leaf_id,
            evidence=f"Candidate failed active resolution threshold ({best_cosine:.2f} < {threshold:.2f})",
        )

    return None


def resolve(
    name: str,
    kind: str = "entity",
    definition_ctx: str = "",
    conn: sqlite3.Connection | None = None,
    embedding: list[float] | None = None,
    source_leaf_id: str | None = None,
    namespace: str = "concepts",
    proposed_ns: str | None = None,
    nlp: Any | None = None,
    threshold: float = config.CANON_SIM_THRESHOLD,
    kb: Any | None = None,
    **kwargs: Any,
) -> ResolutionResult:
    """Execute Two-Tier Convergence Protocol resolving surface form to canonical node.

    Tier 1: Direct match (exact slug or active alias in `aliases` or `InMemoryLookupKB`).
    Tier 2: Near match (in-process vector similarity > `CANON_SIM_THRESHOLD` 0.80 using static GloVe vectors).
    Induction: Novel canonical entity creation via `_create_canonical_node()`.

    Zero LLM calls during canonicalization.
    """
    if isinstance(kind, sqlite3.Connection):
        conn = kind
        kind = "entity"

    storage = sys.modules.get("yaw.storage")
    if storage is None:
        import importlib

        storage = importlib.import_module("yaw.storage")

    if conn is None:
        conn = storage.get_db(load_vec=True)
    else:
        storage.ensure_schema_tables(conn)

    ident = _parse_surface_identity(
        name=name, kind=kind, namespace=namespace, proposed_ns=proposed_ns
    )
    surface_id = ident.surface_id
    slug = ident.slug
    effective_ns = ident.namespace

    # TIER 1: Direct Match
    res = _resolve_tier1_direct(
        conn=conn,
        surface_id=surface_id,
        slug=slug,
        name=name,
        kind=kind,
        source_leaf_id=source_leaf_id,
        kb=kb,
    )
    if res is not None:
        return res

    # TIER 2: Vector Near Match (> CANON_SIM_THRESHOLD 0.80)
    q_vec = _get_or_compute_vector(
        name=name,
        embedding=embedding,
        nlp=nlp,
    )
    if q_vec is not None:
        res = _resolve_tier2_vector(
            conn=conn,
            q_vec=q_vec,
            surface_id=surface_id,
            kind=kind,
            source_leaf_id=source_leaf_id,
            threshold=threshold,
            nlp=nlp,
            namespace=effective_ns,
        )
        if res is not None:
            return res

    # Induction: Novel canonical entity creation
    _create_canonical_node(
        conn=conn,
        surface_id=surface_id,
        name=name,
        namespace=effective_ns,
        source_leaf_id=source_leaf_id,
        definition=definition_ctx,
        embedding=embedding,
    )
    return ResolutionResult(
        canonical_id=surface_id,
        method=MatchMethod.NEW.value,
        status=AliasStatus.ACTIVE.value,
        tier=TierEnum.NEW_INDUCTION,
        score=1.0,
        is_new=True,
    )


def get_contested_aliases(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Query aliases where status = 'contested' ordered by rowid DESC."""
    cur = conn.cursor()
    cur.row_factory = sqlite3.Row
    return cur.execute(
        "SELECT * FROM aliases WHERE status = 'contested' ORDER BY rowid DESC"
    ).fetchall()


__all__ = [
    "AliasStatus",
    "CanonicalVerdict",
    "MatchMethod",
    "Resolution",
    "ResolutionResult",
    "SurfaceIdentity",
    "TierEnum",
    "_create_canonical_node",
    "_record_alias",
    "get_contested_aliases",
    "normalize_token",
    "resolve",
    "slugify",
]
