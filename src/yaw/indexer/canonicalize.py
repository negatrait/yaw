"""Entity Disambiguation and Canonicalization (EDC) Two-Tier Convergence Engine.

Implements the clean Two-Tier Convergence Protocol resolving surface form mentions into canonical nodes:
- Tier 1: Direct match (exact slug, active alias in `aliases`, or `InMemoryLookupKB`).
- Tier 2: Near match (in-process vector similarity > `CANON_SIM_THRESHOLD` (0.80) using `sqlite-vec` or `InMemoryLookupKB`). Consolidates into existing canonical node via `_record_alias()`.
- Fallback: Novel canonical node creation via `_create_canonical_node()`.

Zero Tier 3 LLM pairwise adjudication calls.
Zero procedural O(N) loops or SQLite scan ladders.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import spacy
from spacy.kb import InMemoryLookupKB

from yaw import config
from yaw.nlp import deserialize_embedding
from yaw.nlp.components import get_base_spacy
from yaw.storage.db import (
    ensure_schema_tables,
    get_canonical_id,
    get_db,
    insert_alias,
    insert_leaf_node,
)
from yaw.utils import normalize_token, slugify

logger = logging.getLogger(__name__)


class TierEnum(str, Enum):
    """Execution tier where convergence resolved."""

    TIER_1_EXACT = "tier_1_exact"
    TIER_2_EMBEDDING = "tier_2_embedding"
    NEW_INDUCTION = "new_induction"


class MatchMethod(str, Enum):
    """Method recorded in aliases table or resolution result."""

    EXACT = "exact"
    ALIAS = "alias"
    AUTO = "auto"
    NEW = "new"


class AliasStatus(str, Enum):
    """Status recorded in aliases table."""

    ACTIVE = "active"
    CONTESTED = "contested"


@dataclass(frozen=True)
class ResolutionResult:
    """Result of the Two-Tier Convergence Engine resolve() invocation."""

    canonical_id: str
    method: str
    status: str = "active"
    tier: TierEnum = TierEnum.TIER_1_EXACT
    score: float = 1.0
    is_new: bool = False


# Backward compatibility alias for roadmap specification (Invariant 5A.1)
Resolution = ResolutionResult

_active_kb: InMemoryLookupKB | None = None
_conn_kbs: dict[str, InMemoryLookupKB] = {}


def set_active_kb(kb: InMemoryLookupKB | None) -> None:
    """Set the globally active InMemoryLookupKB instance."""
    global _active_kb
    _active_kb = kb


def clear_kb_cache() -> None:
    """Clear all active and connection-associated InMemoryLookupKB instances."""
    global _active_kb
    _active_kb = None
    _conn_kbs.clear()


def create_in_memory_kb(
    nlp: Any | None = None,
    entity_vector_length: int = 300,
) -> InMemoryLookupKB:
    """Create a spaCy InMemoryLookupKB instance configured with given vocab and vector width."""
    pipe = nlp if nlp is not None else _get_base_spacy()
    vocab = pipe.vocab if pipe is not None else spacy.blank("en").vocab
    width = entity_vector_length
    if (
        getattr(vocab.vectors, "shape", None)
        and len(vocab.vectors.shape) > 1
        and vocab.vectors.shape[1] > 0
    ):
        width = vocab.vectors.shape[1]
    return InMemoryLookupKB(vocab=vocab, entity_vector_length=width)


def _register_alias_in_kb(
    kb: InMemoryLookupKB,
    alias: str,
    canonical_id: str,
    score: float = 1.0,
) -> None:
    """Safely register or append alias in InMemoryLookupKB."""
    clean_alias = alias.strip()
    if not clean_alias or not canonical_id:
        return
    if not kb.contains_entity(canonical_id):
        vec = np.zeros(kb.entity_vector_length, dtype=np.float32)
        kb.add_entity(entity=canonical_id, freq=1, entity_vector=vec)

    prob = min(1.0, max(0.01, float(score)))
    aliases_to_register = [clean_alias]
    if clean_alias.lower() != clean_alias:
        aliases_to_register.append(clean_alias.lower())

    for a_str in aliases_to_register:
        if not kb.contains_alias(a_str):
            kb.add_alias(alias=a_str, entities=[canonical_id], probabilities=[prob])
        else:
            cands = kb.get_alias_candidates(a_str)
            cand_entities = [c.entity_ for c in cands]
            if canonical_id not in cand_entities:
                current_sum = sum(c.prior_prob for c in cands)
                rem = max(0.0, 1.0 - current_sum)
                if rem > 0.01:
                    kb.append_alias(
                        alias=a_str, entity=canonical_id, prior_prob=min(rem, prob)
                    )


def _register_entity_in_kb(
    kb: InMemoryLookupKB,
    entity_id: str,
    name: str = "",
    embedding: list[float] | np.ndarray | None = None,
) -> None:
    """Register an entity and its aliases in InMemoryLookupKB."""
    if not kb.contains_entity(entity_id):
        vec = np.zeros(kb.entity_vector_length, dtype=np.float32)
        if embedding is not None and len(embedding) > 0:
            arr = np.asarray(embedding, dtype=np.float32)
            if len(arr) == kb.entity_vector_length:
                norm = float(np.linalg.norm(arr))
                vec = arr / norm if norm > 1e-12 else arr
        elif name and name in kb.vocab and kb.vocab[name].has_vector:
            arr = np.asarray(kb.vocab[name].vector, dtype=np.float32)
            if len(arr) == kb.entity_vector_length:
                norm = float(np.linalg.norm(arr))
                vec = arr / norm if norm > 1e-12 else arr
        kb.add_entity(entity=entity_id, freq=1, entity_vector=vec)
    _register_alias_in_kb(kb, alias=entity_id, canonical_id=entity_id, score=1.0)
    clean_name = name.strip()
    if clean_name and clean_name != entity_id:
        _register_alias_in_kb(kb, alias=clean_name, canonical_id=entity_id, score=1.0)


def sync_kb_from_db(kb: InMemoryLookupKB, conn: sqlite3.Connection) -> None:
    """Synchronize canonical entities, base predicates, and active aliases from yaw.db into InMemoryLookupKB."""
    defined_aliases: set[str] = set()
    try:
        a_rows = conn.execute(
            "SELECT alias_id FROM aliases WHERE status = 'active'"
        ).fetchall()
        defined_aliases = {str(r[0]).strip().lower() for r in a_rows if r[0]}
    except sqlite3.OperationalError:
        pass

    # 1. Load canonical entities from leaf_nodes
    try:
        rows = conn.execute(
            "SELECT DISTINCT node_id, name, embedding FROM leaf_nodes WHERE node_id IS NOT NULL"
        ).fetchall()
        for nid, name, emb_blob in rows:
            node_id = str(nid)
            clean_name = str(name).strip() if name else ""
            clean_name_lower = clean_name.lower()
            if not kb.contains_entity(node_id):
                vec = np.zeros(kb.entity_vector_length, dtype=np.float32)
                if emb_blob is not None:
                    try:
                        arr = deserialize_embedding(emb_blob)
                        if arr is not None and len(arr) == kb.entity_vector_length:
                            norm = float(np.linalg.norm(arr))
                            vec = arr / norm if norm > 1e-12 else arr
                    except (ValueError, TypeError, json.JSONDecodeError) as err:
                        logger.debug(
                            "Failed to deserialize embedding for node %s: %s",
                            node_id,
                            err,
                        )
                elif (
                    clean_name
                    and clean_name in kb.vocab
                    and kb.vocab[clean_name].has_vector
                ):
                    arr = np.asarray(kb.vocab[clean_name].vector, dtype=np.float32)
                    if len(arr) == kb.entity_vector_length:
                        norm = float(np.linalg.norm(arr))
                        vec = arr / norm if norm > 1e-12 else arr
                kb.add_entity(entity=node_id, freq=1, entity_vector=vec)
            if node_id.lower() not in defined_aliases:
                _register_alias_in_kb(
                    kb, alias=node_id, canonical_id=node_id, score=1.0
                )
            if (
                clean_name_lower
                and clean_name_lower != node_id.lower()
                and clean_name_lower not in defined_aliases
            ):
                _register_alias_in_kb(
                    kb, alias=clean_name, canonical_id=node_id, score=1.0
                )
    except sqlite3.OperationalError:
        pass

    # 2. Load base predicates and distinct leaf_relations predicates
    for bp in config.BASE_PREDICATES:
        if not kb.contains_entity(bp):
            bp_vec = np.zeros(kb.entity_vector_length, dtype=np.float32)
            if bp in kb.vocab and kb.vocab[bp].has_vector:
                arr = np.asarray(kb.vocab[bp].vector, dtype=np.float32)
                if len(arr) == kb.entity_vector_length:
                    norm = float(np.linalg.norm(arr))
                    bp_vec = arr / norm if norm > 1e-12 else arr
            kb.add_entity(entity=bp, freq=1, entity_vector=bp_vec)
        if bp.lower() not in defined_aliases:
            _register_alias_in_kb(kb, alias=bp, canonical_id=bp, score=1.0)

    try:
        rel_preds = conn.execute(
            "SELECT DISTINCT predicate FROM leaf_relations WHERE predicate IS NOT NULL"
        ).fetchall()
        for (pred,) in rel_preds:
            if pred:
                p_str = str(pred).strip()
                if not kb.contains_entity(p_str):
                    p_vec = np.zeros(kb.entity_vector_length, dtype=np.float32)
                    if p_str in kb.vocab and kb.vocab[p_str].has_vector:
                        arr = np.asarray(kb.vocab[p_str].vector, dtype=np.float32)
                        if len(arr) == kb.entity_vector_length:
                            norm = float(np.linalg.norm(arr))
                            p_vec = arr / norm if norm > 1e-12 else arr
                    kb.add_entity(entity=p_str, freq=1, entity_vector=p_vec)
                if p_str.lower() not in defined_aliases:
                    _register_alias_in_kb(
                        kb, alias=p_str, canonical_id=p_str, score=1.0
                    )
    except sqlite3.OperationalError:
        pass

    # 3. Load active aliases from aliases table with grouped prior normalization
    try:
        alias_rows = conn.execute(
            "SELECT alias_id, canonical_id, score FROM aliases WHERE status = 'active'"
        ).fetchall()
        alias_groups: dict[str, dict[str, float]] = {}
        for a_id, c_id, score in alias_rows:
            if not a_id or not c_id:
                continue
            raw_alias = str(a_id).strip()
            canon_str = str(c_id).strip()
            if not kb.contains_entity(canon_str):
                vec = np.zeros(kb.entity_vector_length, dtype=np.float32)
                kb.add_entity(entity=canon_str, freq=1, entity_vector=vec)
            curr_score = float(score) if score is not None else 1.0
            for a_key in {raw_alias, raw_alias.lower()}:
                sub_map = alias_groups.setdefault(a_key, {})
                sub_map[canon_str] = max(sub_map.get(canon_str, 0.0), curr_score)

        for a_key, cand_map in alias_groups.items():
            if kb.contains_alias(a_key):
                existing_cands = kb.get_alias_candidates(a_key)
                existing_ent_ids = [c.entity_ for c in existing_cands]
                current_sum = sum(c.prior_prob for c in existing_cands)
                rem = max(0.0, 1.0 - current_sum)
                new_cands = [c for c in cand_map if c not in existing_ent_ids]
                if new_cands and rem > 0.01:
                    raw_scores = [max(0.001, cand_map[c]) for c in new_cands]
                    tot_score = sum(raw_scores)
                    avail = rem
                    for i, c_str in enumerate(new_cands):
                        if i == len(new_cands) - 1:
                            allot = min(rem, max(0.001, rem))
                        else:
                            allot = min(rem, (cand_map[c_str] / tot_score) * avail)
                            rem -= allot
                        if allot > 0.001:
                            kb.append_alias(alias=a_key, entity=c_str, prior_prob=allot)
            else:
                cands = list(cand_map.keys())
                raw_probs = [max(0.001, cand_map[c]) for c in cands]
                prob_sum = sum(raw_probs)
                probs = (
                    [p / prob_sum for p in raw_probs]
                    if prob_sum > 0
                    else [1.0 / len(cands)] * len(cands)
                )
                kb.add_alias(alias=a_key, entities=cands, probabilities=probs)
    except sqlite3.OperationalError:
        pass


_conn_kbs: dict[str, InMemoryLookupKB] = {}


def _get_connection_uid(conn: sqlite3.Connection) -> str:
    """Acquire unique lifetime identifier for SQLite connection via connection-private temp table."""
    try:
        conn.execute(
            "CREATE TEMP TABLE IF NOT EXISTS _yaw_session (uid TEXT PRIMARY KEY)"
        )
        row = conn.execute("SELECT uid FROM _yaw_session LIMIT 1").fetchone()
        if row and row[0]:
            return str(row[0])
        import uuid

        uid = uuid.uuid4().hex
        conn.execute("INSERT INTO _yaw_session (uid) VALUES (?)", (uid,))
        return uid
    except sqlite3.OperationalError, sqlite3.ProgrammingError, sqlite3.DatabaseError:
        return f"conn_{id(conn)}"


def get_active_kb(
    conn: sqlite3.Connection | None = None,
    nlp: Any | None = None,
) -> InMemoryLookupKB | None:
    """Acquire the active or connection-associated InMemoryLookupKB instance."""
    if conn is not None:
        conn_uid = _get_connection_uid(conn)
        if conn_uid in _conn_kbs:
            return _conn_kbs[conn_uid]
        pipe = nlp if nlp is not None else _get_base_spacy()
        kb = create_in_memory_kb(pipe)
        sync_kb_from_db(kb, conn)
        _conn_kbs[conn_uid] = kb
        return kb
    return _active_kb


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
    kb: InMemoryLookupKB | None = None,
) -> None:
    """Record an alias mapping via SSOT storage.db.insert_alias and update KB."""
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
    if kb is not None and status == "active":
        _register_alias_in_kb(
            kb, alias=alias_id, canonical_id=canonical_id, score=score
        )


def _create_canonical_node(
    conn: sqlite3.Connection,
    surface_id: str,
    name: str,
    namespace: str,
    source_leaf_id: str | None = None,
    definition: str = "",
    embedding: list[float] | np.ndarray | None = None,
    kb: InMemoryLookupKB | None = None,
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
    if kb is not None:
        _register_entity_in_kb(
            kb=kb,
            entity_id=surface_id,
            name=name,
            embedding=embedding,
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


def _lookup_kb(name: str, kb: InMemoryLookupKB) -> str | None:
    """Look up candidate from InMemoryLookupKB using compiled Cython candidate generation."""
    clean = name.strip()
    if not clean:
        return None
    try:
        candidates = kb.get_alias_candidates(clean)
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
    kb: InMemoryLookupKB | None = None,
    canonical_id: str | None = None,
) -> ResolutionResult | None:
    """Tier 1: Direct match via canonical_id, InMemoryLookupKB, active aliases, or exact leaf_nodes."""
    if canonical_id:
        return ResolutionResult(
            canonical_id=canonical_id,
            method=MatchMethod.EXACT.value,
            tier=TierEnum.TIER_1_EXACT,
        )

    if kb is not None:
        cand_id = (
            _lookup_kb(name, kb) or _lookup_kb(surface_id, kb) or _lookup_kb(slug, kb)
        )
        if cand_id:
            method = (
                MatchMethod.EXACT.value
                if cand_id in (surface_id, name.strip())
                else MatchMethod.ALIAS.value
            )
            return ResolutionResult(
                canonical_id=cand_id,
                method=method,
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
                    method=MatchMethod.EXACT.value,
                    source_leaf_id=source_leaf_id,
                    kb=kb,
                )
            return ResolutionResult(
                canonical_id=matched_bp,
                method=MatchMethod.EXACT.value,
                tier=TierEnum.TIER_1_EXACT,
            )

    canon_from_alias = _lookup_active_alias(conn, surface_id, slug, name, kind)
    if canon_from_alias:
        if kb is not None:
            _register_alias_in_kb(kb, surface_id, canon_from_alias, 1.0)
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
                method=MatchMethod.EXACT.value,
                source_leaf_id=source_leaf_id,
                kb=kb,
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


def _find_kb_nearest(
    kb: InMemoryLookupKB,
    q_vec: np.ndarray,
    namespace: str | None = None,
) -> tuple[str | None, float]:
    """Find nearest entity in InMemoryLookupKB using vectorized matrix cosine similarity."""
    if kb.get_size_entities() == 0:
        return None, 0.0

    if q_vec.shape[0] != kb.entity_vector_length:
        return None, 0.0

    entities = kb.get_entity_strings()
    if namespace is not None and namespace != "predicates":
        prefix = f"{namespace}:"
        filtered_entities = [e for e in entities if e.startswith(prefix)]
        if filtered_entities:
            entities = filtered_entities
    elif namespace == "predicates":
        entities = [
            e
            for e in entities
            if e in config.BASE_PREDICATES or e.startswith("predicates:")
        ]

    if not entities:
        return None, 0.0

    raw_vecs = kb.get_vectors(entities)
    mat = np.asarray(raw_vecs, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    zero_mask = (norms < 1e-12).reshape(-1)
    safe_norms = np.where(norms < 1e-12, 1.0, norms)
    norm_mat = mat / safe_norms

    q_norm = float(np.linalg.norm(q_vec))
    if q_norm < 1e-12:
        return None, 0.0
    q_unit = q_vec / q_norm

    sims = (norm_mat @ q_unit).reshape(-1)
    if np.any(zero_mask):
        sims[zero_mask] = 0.0

    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])
    if best_sim > 0.0:
        return entities[best_idx], max(0.0, min(1.0, best_sim))
    return None, 0.0


def _resolve_tier2_vector(
    conn: sqlite3.Connection,
    q_vec: np.ndarray,
    surface_id: str,
    kind: str,
    source_leaf_id: str | None,
    threshold: float,
    nlp: Any | None = None,
    namespace: str | None = None,
    kb: InMemoryLookupKB | None = None,
    name: str = "",
) -> ResolutionResult | None:
    """Tier 2: In-process vector similarity near match consolidating to existing canonical node."""
    best_cand_id, best_cosine = _find_vec_nearest(conn, q_vec, namespace=namespace)

    if best_cosine < threshold and kb is not None:
        kb_cand_id, kb_cosine = _find_kb_nearest(kb, q_vec, namespace=namespace)
        if kb_cosine > best_cosine:
            best_cand_id, best_cosine = kb_cand_id, kb_cosine

    if best_cand_id is not None and best_cosine >= threshold:
        _record_alias(
            conn=conn,
            alias_id=surface_id,
            canonical_id=best_cand_id,
            kind=kind,
            method=MatchMethod.AUTO.value,
            score=best_cosine,
            status=AliasStatus.ACTIVE.value,
            source_leaf_id=source_leaf_id,
            kb=kb,
        )
        if kb is not None and name.strip() and name.strip() != surface_id:
            _register_alias_in_kb(
                kb, alias=name.strip(), canonical_id=best_cand_id, score=best_cosine
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
            method=MatchMethod.AUTO.value,
            score=best_cosine,
            status=AliasStatus.CONTESTED.value,
            source_leaf_id=source_leaf_id,
            evidence=f"Candidate failed active resolution threshold ({best_cosine:.2f} < {threshold:.2f})",
            kb=kb,
        )

    return None


def resolve(
    name: str,
    kind: str = "entity",
    definition_ctx: str = "",
    conn: sqlite3.Connection | None = None,
    embedding: list[float] | np.ndarray | None = None,
    source_leaf_id: str | None = None,
    namespace: str = "concepts",
    proposed_ns: str | None = None,
    nlp: Any | None = None,
    threshold: float = config.CANON_SIM_THRESHOLD,
    kb: InMemoryLookupKB | None = None,
    canonical_id: str | None = None,
    **kwargs: Any,
) -> ResolutionResult:
    """Execute Two-Tier Convergence Protocol resolving surface form to canonical node.

    Tier 1: Direct match (canonical_id, exact slug, active alias in `aliases`, or `InMemoryLookupKB`).
    Tier 2: Near match (in-process vector similarity > `CANON_SIM_THRESHOLD` (0.80) using `sqlite-vec` or `InMemoryLookupKB`).
    Induction: Novel canonical entity creation via `_create_canonical_node()`.

    Zero external HTTP or LLM calls during canonicalization.
    """
    if conn is None:
        conn = get_db(load_vec=True)
    else:
        ensure_schema_tables(conn)

    active_kb = kb if kb is not None else get_active_kb(conn, nlp=nlp)

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
        kb=active_kb,
        canonical_id=canonical_id,
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
            kb=active_kb,
            name=name,
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
        kb=active_kb,
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
    "MatchMethod",
    "Resolution",
    "ResolutionResult",
    "SurfaceIdentity",
    "TierEnum",
    "_create_canonical_node",
    "_record_alias",
    "clear_kb_cache",
    "create_in_memory_kb",
    "get_active_kb",
    "get_contested_aliases",
    "normalize_token",
    "resolve",
    "set_active_kb",
    "slugify",
    "sync_kb_from_db",
]
