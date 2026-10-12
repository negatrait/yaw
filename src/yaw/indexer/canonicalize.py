"""Entity Disambiguation and Canonicalization (EDC) Two-Tier Convergence Engine.

Implements the clean Two-Tier Convergence Protocol resolving surface form mentions into canonical nodes:
- Tier 1: Deterministic NLP normalization in Space L (NFKC, casefold, determiners, possessive clitics, NOUN lemmatization),
  followed by exact canonical_id, normalized key lookup in unified_nodes, active aliases, or InMemoryLookupKB.
- Tier 2: Blocked candidate generation in Space S (exact KNN on entity_vec, k=5, cosine >= 0.80) followed by
  deterministic linguistic corroboration L(a, e) (acronym, head-anchored containment, initial expansion).
  - Exactly one corroborated candidate -> hard merge into aliases (method='auto', score=cosine).
  - Otherwise (0 or >= 2 corroborated candidates) -> soft links in close_matches, novel node induction.
- Audit Logging: Every evaluated pair logs cosine, lexical (1.0 vs 0.0), and verdict ('match' vs 'distinct') to canon_verdicts.
- Predicates Space-L Guard: Predicates never enter Tier 2. Resolved strictly via synset, base predicate, or novel induction.

Zero Tier 3 LLM pairwise adjudication calls.
Zero procedural O(N) loops or SQLite scan ladders.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import time
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import spacy
from spacy.kb import InMemoryLookupKB
from spacy.language import Language

from yaw import config
from yaw.nlp.components import get_base_spacy, normalize_predicate_antonym_guarded
from yaw.storage.db import (
    ensure_schema_tables,
    get_canonical_id,
    get_db,
    insert_alias,
    insert_close_match,
    insert_entity_vec,
    insert_leaf_node,
    log_canon_verdict,
    serialize_embedding,
)
from yaw.utils import normalize_token, slugify

logger = logging.getLogger(__name__)

# Canonical alias for decision audit logging per ADR-012
insert_canon_verdict = log_canon_verdict


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


@dataclass(frozen=True)
class ResolutionResult:
    """Result of the Two-Tier Convergence Engine resolve() invocation."""

    canonical_id: str
    method: str
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
    entity_vector_length: int = 768,
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
    """Register an entity and its aliases in InMemoryLookupKB as an alias prior index."""
    if not kb.contains_entity(entity_id):
        vec = np.zeros(kb.entity_vector_length, dtype=np.float32)
        kb.add_entity(entity=entity_id, freq=1, entity_vector=vec)
    _register_alias_in_kb(kb, alias=entity_id, canonical_id=entity_id, score=1.0)
    clean_name = name.strip()
    if clean_name and clean_name != entity_id:
        _register_alias_in_kb(kb, alias=clean_name, canonical_id=entity_id, score=1.0)


def sync_kb_from_db(kb: InMemoryLookupKB, conn: sqlite3.Connection) -> None:
    """Synchronize canonical entities, base predicates, and active aliases from yaw.db into InMemoryLookupKB.

    Per ADR-012 §1 and §3, InMemoryLookupKB serves strictly as an alias -> prior probability index.
    Zero vector matrix operations are executed over KB entities.
    """
    defined_aliases: set[str] = set()
    try:
        a_rows = conn.execute("SELECT alias_id FROM aliases").fetchall()
        defined_aliases = {str(r[0]).strip().lower() for r in a_rows if r[0]}
    except sqlite3.OperationalError:
        pass

    # 1. Load canonical entities from leaf_nodes
    try:
        rows = conn.execute(
            """
            SELECT DISTINCT node_id, name 
            FROM leaf_nodes 
            WHERE node_id IS NOT NULL
            """
        ).fetchall()
        for nid, name in rows:
            node_id = str(nid)
            clean_name = str(name).strip() if name else ""
            clean_name_lower = clean_name.lower()
            if not kb.contains_entity(node_id):
                vec = np.zeros(kb.entity_vector_length, dtype=np.float32)
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
            vec = np.zeros(kb.entity_vector_length, dtype=np.float32)
            kb.add_entity(entity=bp, freq=1, entity_vector=vec)
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
                    vec = np.zeros(kb.entity_vector_length, dtype=np.float32)
                    kb.add_entity(entity=p_str, freq=1, entity_vector=vec)
                if p_str.lower() not in defined_aliases:
                    _register_alias_in_kb(
                        kb, alias=p_str, canonical_id=p_str, score=1.0
                    )
    except sqlite3.OperationalError:
        pass

    # 3. Load active aliases from aliases table with grouped prior normalization
    try:
        alias_rows = conn.execute(
            "SELECT alias_id, canonical_id, score FROM aliases"
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
        source_leaf_id=source_leaf_id,
        evidence=evidence,
    )
    if kb is not None:
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
        source_leaf_id=source_leaf_id or "system:canonical_induction",
    )
    if embedding is not None:
        vec_bytes = serialize_embedding(embedding)
        if vec_bytes:
            insert_entity_vec(conn, surface_id, namespace, vec_bytes)

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
    """Extract or normalize vector representation in Space S. Zero external HTTP calls."""
    if embedding is not None and len(embedding) > 0:
        arr = np.asarray(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        return arr / norm if norm > 1e-12 else arr
    return None


# ==============================================================================
# TIER 1: DETERMINISTIC NLP NORMALIZATION (SPACE L)
# ==============================================================================

_LIGATURE_MAP: dict[str, str] = {
    "œ": "oe",
    "æ": "ae",
    "ø": "o",
    "ł": "l",
    "đ": "d",
    "ð": "d",
    "þ": "th",
}


def _strip_namespace_prefix(text: str) -> str:
    """Strip namespace prefix from a surface mention, canonical ID, or entity label."""
    clean = text.strip()
    if ":" in clean:
        prefix, rest = clean.split(":", 1)
        prefix_slug = prefix.strip().lower().replace("-", "_")
        if (
            prefix_slug in config.TAXONOMY_NAMESPACES
            or prefix_slug in ("concept", "concepts", "predicate", "predicates")
            or (" " not in prefix and len(prefix) <= 32 and bool(rest.strip()))
        ):
            return rest.strip()
    return clean


def _normalize_entity_key(name: str, nlp: Language | None = None) -> str:
    """Deterministic Space-L normalization for entity surface forms.

    Per ADR-012 §2:
    1. NFKC normalization + casefold: unicodedata.normalize("NFKC", name).casefold().
    2. Strip leading determiners (^the\\s+, ^a\\s+, ^an\\s+).
    3. Strip possessive clitics ('s\\b, ’s\\b).
    4. Lemmatize NOUN tokens with spaCy (token.lemma_), keep PROPN tokens verbatim (token.text).
    5. Collapse punctuation and whitespace into hyphens (concepts:{normalized-key}).
    """
    if not name:
        return ""

    # 1. NFKC normalization + casefold
    text = unicodedata.normalize("NFKC", name).casefold().strip()
    if not text:
        return ""

    # 2. Strip leading determiners (preserve if pure determiner or single-letter initial like 'A.')
    stripped_det = re.sub(
        r"^(the\b[\s\-_:,.]+|an\b[\s\-_:,.]+|a\b(?!\.)[\s\-_:,]+)", "", text
    )
    if stripped_det.strip():
        text = stripped_det.strip()

    # 3. Strip possessive clitics
    text = re.sub(r"('s|’s)\b", "", text)
    text = text.strip()
    if not text:
        return ""

    # 4. Lemmatize NOUN tokens with spaCy, keep PROPN tokens verbatim
    pipe = nlp if nlp is not None else _get_base_spacy()
    if pipe is not None:
        try:
            doc = pipe(text)
            toks: list[str] = []
            for t in doc:
                if t.is_punct or t.is_space:
                    continue
                if t.pos_ == "NOUN":
                    toks.append(t.lemma_.lower())
                else:
                    toks.append(t.text.lower())
            processed = "-".join(toks)
        except (ValueError, TypeError, AttributeError, KeyError) as err:
            logger.debug("spaCy lemmatization failed in _normalize_entity_key: %s", err)
            processed = text
    else:
        processed = text

    # 5. Expand ligatures and special orthographic characters
    for lig, rep in _LIGATURE_MAP.items():
        processed = processed.replace(lig, rep)

    # Filter out combining diacritical marks via NFKD
    decomposed = unicodedata.normalize("NFKD", processed)
    stripped = "".join(c for c in decomposed if unicodedata.combining(c) == 0)

    # Collapse punctuation and whitespace into hyphens
    norm_key = "-".join(re.findall(r"[a-z0-9]+", stripped)).strip("-")
    if not norm_key:
        import hashlib

        norm_key = "id-" + hashlib.md5(name.encode("utf-8")).hexdigest()[:8]
    return norm_key


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
    nlp: Language | None = None,
) -> SurfaceIdentity:
    """Parse and normalize entity/predicate name into surface_id, normalized key slug, and effective namespace."""
    raw_ns = proposed_ns or namespace
    if kind == "predicate":
        effective_ns = (
            "predicates"
            if not raw_ns
            or raw_ns in ("concept", "concepts")
            or raw_ns not in config.TAXONOMY_NAMESPACES
            else raw_ns
        )
        pred_name = _strip_namespace_prefix(name)
        slug = slugify(pred_name)
        if not slug:
            import hashlib

            slug = "id-" + hashlib.md5(name.encode("utf-8")).hexdigest()[:8]
        return SurfaceIdentity(surface_id=slug, slug=slug, namespace=effective_ns)

    # kind == "entity"
    effective_ns = (
        "concepts"
        if not raw_ns or raw_ns == "concept" or raw_ns not in config.TAXONOMY_NAMESPACES
        else raw_ns
    )

    entity_name = name
    if ":" in name:
        parts = name.split(":", 1)
        ns_part = slugify(parts[0]).replace("-", "_")
        if ns_part in config.TAXONOMY_NAMESPACES or ns_part == "concept":
            effective_ns = "concepts" if ns_part in ("concept", "concepts") else ns_part
            entity_name = parts[1]

    norm_key = _normalize_entity_key(entity_name, nlp=nlp)
    surface_id = f"{effective_ns}:{norm_key}"
    return SurfaceIdentity(surface_id=surface_id, slug=norm_key, namespace=effective_ns)


def _lookup_active_alias(
    conn: sqlite3.Connection,
    surface_id: str,
    slug: str,
    name: str,
    kind: str,
    legacy_slug: str | None = None,
) -> str | None:
    """Check active aliases by surface_id, slug, raw name, or legacy slug via SSOT storage.db.get_canonical_id."""
    queries = (
        [surface_id, slug, name.strip()]
        if kind == "entity"
        else [surface_id, name.strip()]
    )
    if kind == "entity":
        if surface_id.startswith("concepts:"):
            queries.append(f"concept:{slug}")
        if legacy_slug and legacy_slug != slug:
            queries.extend(
                [legacy_slug, f"concepts:{legacy_slug}", f"concept:{legacy_slug}"]
            )
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
    pred_raw = name.strip()
    unprefixed = _strip_namespace_prefix(pred_raw)
    unprefixed_slug = slugify(unprefixed)
    pred_norm = slug.replace("-", "_").lower()
    unprefixed_norm = unprefixed_slug.replace("-", "_").lower()
    for bp in config.BASE_PREDICATES:
        if bp.lower() in (
            pred_norm,
            pred_raw.lower(),
            unprefixed_norm,
            unprefixed.lower(),
        ):
            return bp
    try:
        row = conn.execute(
            """
            SELECT predicate FROM leaf_relations
            WHERE predicate = ? OR lower(predicate) = ? OR lower(replace(predicate, '-', '_')) = ?
               OR predicate = ? OR lower(predicate) = ? OR lower(replace(predicate, '-', '_')) = ?
            LIMIT 1
            """,
            (
                pred_raw,
                pred_raw.lower(),
                pred_norm,
                unprefixed,
                unprefixed.lower(),
                unprefixed_norm,
            ),
        ).fetchone()
        if row and row[0]:
            return str(row[0])
    except sqlite3.OperationalError:
        pass
    return None


def _lookup_unified_node(
    conn: sqlite3.Connection,
    surface_id: str,
    slug: str,
    kind: str,
    legacy_slug: str | None = None,
) -> str | None:
    """Direct match against unified_nodes (with fallback to leaf_nodes)."""
    try:
        node_row = conn.execute(
            "SELECT node_id FROM unified_nodes WHERE node_id = ? LIMIT 1",
            (surface_id,),
        ).fetchone()
        if not node_row and kind == "entity":
            node_row = conn.execute(
                "SELECT node_id FROM unified_nodes WHERE node_id LIKE ? LIMIT 1",
                (f"%:{slug}",),
            ).fetchone()
            if not node_row and legacy_slug and legacy_slug != slug:
                node_row = conn.execute(
                    "SELECT node_id FROM unified_nodes WHERE node_id LIKE ? LIMIT 1",
                    (f"%:{legacy_slug}",),
                ).fetchone()
        if node_row and node_row[0]:
            nid = str(node_row[0])
            is_pred = nid in config.BASE_PREDICATES or nid.startswith("predicates:")
            if kind == "predicate" and not is_pred:
                return None
            if kind == "entity" and is_pred:
                return None
            return nid
    except sqlite3.OperationalError:
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
                if not node_row and legacy_slug and legacy_slug != slug:
                    node_row = conn.execute(
                        "SELECT node_id FROM leaf_nodes WHERE node_id LIKE ? LIMIT 1",
                        (f"%:{legacy_slug}",),
                    ).fetchone()
            if node_row and node_row[0]:
                nid = str(node_row[0])
                is_pred = nid in config.BASE_PREDICATES or nid.startswith("predicates:")
                if kind == "predicate" and not is_pred:
                    return None
                if kind == "entity" and is_pred:
                    return None
                return nid
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
    legacy_slug: str | None = None,
    nlp: Language | None = None,
) -> ResolutionResult | None:
    """Tier 1: Direct match via canonical_id, InMemoryLookupKB, active aliases, or unified_nodes."""
    if canonical_id:
        if kind == "predicate":
            for bp in config.BASE_PREDICATES:
                if (
                    canonical_id.lower() == bp.lower()
                    or canonical_id.lower() == f"predicates:{bp.lower()}"
                ):
                    canonical_id = bp
                    break
        return ResolutionResult(
            canonical_id=canonical_id,
            method=MatchMethod.EXACT.value,
            tier=TierEnum.TIER_1_EXACT,
        )

    # For predicates, evaluate base predicates and Space-L synset normalizer first
    if kind == "predicate":
        matched_bp = _lookup_predicate_direct(conn, name, slug)
        if matched_bp is not None:
            for bp in config.BASE_PREDICATES:
                if (
                    matched_bp.lower() == bp.lower()
                    or matched_bp.lower() == f"predicates:{bp.lower()}"
                ):
                    matched_bp = bp
                    break
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

        # Space L synset/polarity-guarded normalizer
        norm_pred = normalize_predicate_antonym_guarded(name, nlp=nlp)
        if norm_pred and norm_pred in config.BASE_PREDICATES:
            if norm_pred != name.strip():
                _record_alias(
                    conn,
                    alias_id=name.strip(),
                    canonical_id=norm_pred,
                    kind="predicate",
                    method=MatchMethod.EXACT.value,
                    source_leaf_id=source_leaf_id,
                    kb=kb,
                )
            return ResolutionResult(
                canonical_id=norm_pred,
                method=MatchMethod.EXACT.value,
                tier=TierEnum.TIER_1_EXACT,
            )

    if kb is not None:
        cand_id = (
            _lookup_kb(name, kb)
            or _lookup_kb(surface_id, kb)
            or _lookup_kb(slug, kb)
            or (legacy_slug and _lookup_kb(legacy_slug, kb))
        )
        if cand_id:
            is_cand_pred = cand_id in config.BASE_PREDICATES or cand_id.startswith(
                "predicates:"
            )
            if kind == "predicate" and is_cand_pred:
                for bp in config.BASE_PREDICATES:
                    if (
                        cand_id.lower() == bp.lower()
                        or cand_id.lower() == f"predicates:{bp.lower()}"
                    ):
                        cand_id = bp
                        break
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
            elif kind == "entity" and not is_cand_pred:
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

    canon_from_alias = _lookup_active_alias(
        conn, surface_id, slug, name, kind, legacy_slug=legacy_slug
    )
    if canon_from_alias:
        if kind == "predicate":
            for bp in config.BASE_PREDICATES:
                if (
                    canon_from_alias.lower() == bp.lower()
                    or canon_from_alias.lower() == f"predicates:{bp.lower()}"
                ):
                    canon_from_alias = bp
                    break
        if kb is not None:
            _register_alias_in_kb(kb, surface_id, canon_from_alias, 1.0)
        return ResolutionResult(
            canonical_id=canon_from_alias,
            method=MatchMethod.ALIAS.value,
            tier=TierEnum.TIER_1_EXACT,
        )

    exact_node = _lookup_unified_node(
        conn, surface_id, slug, kind, legacy_slug=legacy_slug
    )
    if exact_node:
        if kind == "predicate":
            for bp in config.BASE_PREDICATES:
                if (
                    exact_node.lower() == bp.lower()
                    or exact_node.lower() == f"predicates:{bp.lower()}"
                ):
                    exact_node = bp
                    break
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


# ==============================================================================
# TIER 2: BLOCKED CANDIDATE GENERATION & LINGUISTIC CORROBORATION (SPACES S & L)
# ==============================================================================


def is_acronym(span_a: Any, span_e: Any) -> bool:
    """True if one side is a single upper-case token of length 2-6 equal to initials of the other side's content tokens."""

    def _check_dir(d1: Any, d2: Any) -> bool:
        toks1 = [t for t in d1 if not (t.is_punct or t.is_space)]
        if len(toks1) != 1:
            return False
        clean1 = re.sub(r"[^A-Za-z0-9]", "", toks1[0].text)
        if not (2 <= len(clean1) <= 6 and clean1.isupper()):
            return False
        content = [
            t
            for t in d2
            if not (t.is_stop or t.is_punct or t.is_space) and len(t.text) > 0
        ]
        if len(content) >= 2 and clean1 == "".join(t.text[0].upper() for t in content):
            return True
        all_toks = [t for t in d2 if not (t.is_punct or t.is_space) and len(t.text) > 0]
        return len(all_toks) >= 2 and clean1 == "".join(
            t.text[0].upper() for t in all_toks
        )

    s_a = (
        span_a[:]
        if hasattr(span_a, "__getitem__") and not hasattr(span_a, "root")
        else span_a
    )
    s_e = (
        span_e[:]
        if hasattr(span_e, "__getitem__") and not hasattr(span_e, "root")
        else span_e
    )
    return _check_dir(s_a, s_e) or _check_dir(s_e, s_a)


def is_head_anchored_containment(span_a: Any, span_e: Any) -> bool:
    """True if one token sequence is a contiguous subsequence of the other, AND both share syntactic head."""
    s_a = (
        span_a[:]
        if hasattr(span_a, "__getitem__") and not hasattr(span_a, "root")
        else span_a
    )
    s_e = (
        span_e[:]
        if hasattr(span_e, "__getitem__") and not hasattr(span_e, "root")
        else span_e
    )

    toks_a = [t.text.lower() for t in s_a if not (t.is_punct or t.is_space)]
    toks_e = [t.text.lower() for t in s_e if not (t.is_punct or t.is_space)]
    if not toks_a or not toks_e:
        return False

    root_lemma_a = s_a.root.lemma_.strip().lower()
    root_lemma_e = s_e.root.lemma_.strip().lower()
    root_text_a = s_a.root.text.strip().lower()
    root_text_e = s_e.root.text.strip().lower()
    if root_lemma_a != root_lemma_e and root_text_a != root_text_e:
        return False

    def _is_subsequence(sub: list[str], full: list[str]) -> bool:
        n = len(sub)
        return any(full[i : i + n] == sub for i in range(len(full) - n + 1))

    if _is_subsequence(toks_a, toks_e) or _is_subsequence(toks_e, toks_a):
        return True

    lemmas_a = [t.lemma_.lower() for t in s_a if not (t.is_punct or t.is_space)]
    lemmas_e = [t.lemma_.lower() for t in s_e if not (t.is_punct or t.is_space)]
    return _is_subsequence(lemmas_a, lemmas_e) or _is_subsequence(lemmas_e, lemmas_a)


def is_initial_expansion(span_a: Any, span_e: Any) -> bool:
    """True if token-aligned match where initials match full tokens (e.g. 'A. Einstein' <-> 'Albert Einstein')."""
    s_a = (
        span_a[:]
        if hasattr(span_a, "__getitem__") and not hasattr(span_a, "root")
        else span_a
    )
    s_e = (
        span_e[:]
        if hasattr(span_e, "__getitem__") and not hasattr(span_e, "root")
        else span_e
    )

    toks1 = [
        t.text.strip(".").lower()
        for t in s_a
        if not (t.is_punct or t.is_space) and t.text.strip(".")
    ]
    toks2 = [
        t.text.strip(".").lower()
        for t in s_e
        if not (t.is_punct or t.is_space) and t.text.strip(".")
    ]
    if len(toks1) != len(toks2) or len(toks1) < 2:
        return False

    has_expansion = False
    for t1, t2 in zip(toks1, toks2):
        if t1 == t2:
            continue
        if (len(t1) == 1 and len(t2) >= 1 and t2.startswith(t1)) or (
            len(t2) == 1 and len(t1) >= 1 and t1.startswith(t2)
        ):
            has_expansion = True
        else:
            return False
    return has_expansion


def check_linguistic_corroboration(
    name_a: str,
    name_e: str,
    nlp: Language | None = None,
) -> bool:
    """Evaluate deterministic linguistic corroboration L(a, e) across acronym, containment, and initials."""
    clean_a = _strip_namespace_prefix(name_a)
    clean_e = _strip_namespace_prefix(name_e)
    pipe = nlp if nlp is not None else _get_base_spacy()
    if pipe is None:
        return False
    try:
        doc_a = pipe(clean_a)
        doc_e = pipe(clean_e)
        span_a = doc_a[:]
        span_e = doc_e[:]
        if is_acronym(span_a, span_e):
            return True
        if is_head_anchored_containment(span_a, span_e):
            return True
        if is_initial_expansion(span_a, span_e):
            return True
    except (ValueError, TypeError, AttributeError, KeyError, IndexError) as err:
        logger.debug("Linguistic corroboration check failed: %s", err)
    return False


def _get_entity_label(conn: sqlite3.Connection, entity_id: str) -> str:
    """Retrieve human-readable surface name for an entity node."""
    try:
        row = conn.execute(
            "SELECT name FROM unified_nodes WHERE node_id = ? LIMIT 1",
            (entity_id,),
        ).fetchone()
        if row and row[0]:
            return str(row[0]).strip()
    except sqlite3.OperationalError:
        try:
            row = conn.execute(
                "SELECT name FROM leaf_nodes WHERE node_id = ? LIMIT 1",
                (entity_id,),
            ).fetchone()
            if row and row[0]:
                return str(row[0]).strip()
        except sqlite3.OperationalError:
            pass
    if ":" in entity_id:
        return entity_id.split(":", 1)[1].replace("-", " ").replace("_", " ").strip()
    return entity_id.replace("-", " ").replace("_", " ").strip()


def _query_entity_vec_knn(
    conn: sqlite3.Connection,
    q_vec: np.ndarray,
    namespace: str | None = None,
    top_k: int = 5,
    min_cosine: float = 0.80,
    exclude_id: str | None = None,
) -> list[tuple[str, float]]:
    """Query entity_vec in Space S for top-k candidates within namespace partition and active ontologies."""
    q_bytes = q_vec.astype(np.float32).tobytes()
    partitions = [namespace] if namespace else ["concepts"]
    try:
        onto_rows = conn.execute(
            "SELECT ontology_id FROM dynamic_ontologies"
        ).fetchall()
        for (oid,) in onto_rows:
            if oid and str(oid) not in partitions:
                partitions.append(str(oid))
    except sqlite3.OperationalError:
        pass

    results: list[tuple[str, float]] = []
    try:
        placeholders = ", ".join("?" for _ in partitions)
        sql = f"""
            SELECT entity_id, vec_distance_cosine(embedding, ?) AS distance
            FROM entity_vec
            WHERE partition_key IN ({placeholders})
            ORDER BY distance ASC
            LIMIT ?
        """
        params: list[Any] = [q_bytes]
        params.extend(partitions)
        params.append(top_k * 2)
        rows = conn.execute(sql, tuple(params)).fetchall()
        for eid, dist in rows:
            if eid is None or dist is None:
                continue
            cand_id = str(eid)
            if exclude_id and cand_id == exclude_id:
                continue
            cosine = max(0.0, min(1.0, 1.0 - float(dist)))
            if cosine >= min_cosine:
                results.append((cand_id, cosine))
            if len(results) >= top_k:
                break
    except (sqlite3.OperationalError, sqlite3.DatabaseError) as err:
        logger.debug("sqlite-vec KNN query error: %s", err)
    return results


def _find_vec_nearest(
    conn: sqlite3.Connection,
    q_vec: np.ndarray,
    namespace: str | None = None,
) -> tuple[str | None, float]:
    """Query single nearest candidate vector via sqlite-vec."""
    candidates = _query_entity_vec_knn(
        conn=conn,
        q_vec=q_vec,
        namespace=namespace,
        top_k=1,
        min_cosine=0.0,
    )
    if candidates:
        return candidates[0][0], candidates[0][1]
    return None, 0.0


def _resolve_tier2_vector(
    conn: sqlite3.Connection,
    q_vec: np.ndarray,
    surface_id: str,
    kind: str,
    source_leaf_id: str | None,
    threshold: float,
    nlp: Language | None = None,
    namespace: str | None = None,
    kb: InMemoryLookupKB | None = None,
    name: str = "",
    definition_ctx: str = "",
    embedding: list[float] | np.ndarray | None = None,
) -> ResolutionResult | None:
    """Tier 2: Blocked KNN candidate generation in Space S + Linguistic Corroboration in Space L.

    Per ADR-012 §2:
    - Exactly 1 corroborated candidate -> hard merge into aliases (method='auto', score=cosine).
    - Otherwise (0 or >= 2 corroborated candidates) -> soft link in close_matches, novel induction.
    - Every evaluated pair logs decision to canon_verdicts.
    """
    if kind == "predicate":
        return None

    # Step 1: Blocking in Space S
    candidates = _query_entity_vec_knn(
        conn=conn,
        q_vec=q_vec,
        namespace=namespace,
        top_k=config.CANON_TOP_K,
        min_cosine=config.CANON_SIM_THRESHOLD,
        exclude_id=surface_id,
    )
    if not candidates:
        return None

    # Step 2: Matching via Linguistic Corroboration
    corroborated: list[tuple[str, float, str]] = []
    evaluated: list[tuple[str, float, str, bool]] = []

    for cand_id, cosine in candidates:
        cand_label = _get_entity_label(conn, cand_id)
        is_corr = check_linguistic_corroboration(name, cand_label, nlp=nlp)
        evaluated.append((cand_id, cosine, cand_label, is_corr))
        if is_corr:
            corroborated.append((cand_id, cosine, cand_label))

    now_ts = time.time()

    # Step 3: Decision
    if len(corroborated) == 1:
        matched_id, matched_cosine, matched_label = corroborated[0]
        for cand_id, cosine, cand_label, is_corr in evaluated:
            if cand_id == matched_id:
                log_canon_verdict(
                    conn=conn,
                    surface_id=surface_id,
                    kind=kind,
                    candidate_id=cand_id,
                    cosine=cosine,
                    lexical=1.0,
                    verdict="match",
                    phase="ingest",
                    decided_at=now_ts,
                    rationale=f"Corroborated Tier 2 match ({matched_label})",
                )
            else:
                log_canon_verdict(
                    conn=conn,
                    surface_id=surface_id,
                    kind=kind,
                    candidate_id=cand_id,
                    cosine=cosine,
                    lexical=0.0,
                    verdict="distinct",
                    phase="ingest",
                    decided_at=now_ts,
                    rationale="Distinct from matched candidate",
                )

        _record_alias(
            conn=conn,
            alias_id=surface_id,
            canonical_id=matched_id,
            kind=kind,
            method=MatchMethod.AUTO.value,
            score=matched_cosine,
            source_leaf_id=source_leaf_id,
            evidence=f"Corroborated Tier 2 match ({matched_label})",
            kb=kb,
        )
        if name.strip() and name.strip() != surface_id:
            _record_alias(
                conn=conn,
                alias_id=name.strip(),
                canonical_id=matched_id,
                kind=kind,
                method=MatchMethod.AUTO.value,
                score=matched_cosine,
                source_leaf_id=source_leaf_id,
                kb=kb,
            )

        return ResolutionResult(
            canonical_id=matched_id,
            method=MatchMethod.AUTO.value,
            tier=TierEnum.TIER_2_EMBEDDING,
            score=matched_cosine,
            is_new=False,
        )

    # 0 or >= 2 corroborated candidates -> NO merge!
    rationale_prefix = (
        "Ambiguous identity (multiple corroborated candidates)"
        if len(corroborated) >= 2
        else "Uncorroborated candidate"
    )

    for cand_id, cosine, cand_label, is_corr in evaluated:
        lexical = 1.0 if is_corr else 0.0
        rationale = f"{rationale_prefix}: {cand_label}"
        log_canon_verdict(
            conn=conn,
            surface_id=surface_id,
            kind=kind,
            candidate_id=cand_id,
            cosine=cosine,
            lexical=lexical,
            verdict="distinct",
            phase="ingest",
            decided_at=now_ts,
            rationale=rationale,
        )
        insert_close_match(
            conn=conn,
            src_id=surface_id,
            dst_id=cand_id,
            score=cosine,
            evidence=rationale,
        )

    _create_canonical_node(
        conn=conn,
        surface_id=surface_id,
        name=name,
        namespace=namespace or "concepts",
        source_leaf_id=source_leaf_id,
        definition=definition_ctx,
        embedding=embedding,
        kb=kb,
    )
    return ResolutionResult(
        canonical_id=surface_id,
        method=MatchMethod.NEW.value,
        tier=TierEnum.NEW_INDUCTION,
        score=1.0,
        is_new=True,
    )


def resolve(
    name: str,
    kind: str = "entity",
    definition_ctx: str = "",
    conn: sqlite3.Connection | None = None,
    embedding: list[float] | np.ndarray | None = None,
    source_leaf_id: str | None = None,
    namespace: str = "concepts",
    proposed_ns: str | None = None,
    nlp: Language | None = None,
    threshold: float = config.CANON_SIM_THRESHOLD,
    kb: InMemoryLookupKB | None = None,
    canonical_id: str | None = None,
    **kwargs: Any,
) -> ResolutionResult:
    """Execute Two-Tier Convergence Protocol resolving surface form to canonical node.

    Tier 1: Direct match in Space L (canonical_id, deterministic normalized key in unified_nodes,
            active alias in `aliases`, or `InMemoryLookupKB`).
    Tier 2: Blocked KNN candidate generation in Space S + Linguistic Corroboration in Space L.
            Unique corroborated candidate hard-merges into `aliases`.
            Uncorroborated/ambiguous candidates soft-link into `close_matches`.
    Induction: Novel canonical entity creation via `_create_canonical_node()`.

    Predicates are guarded: they never enter Tier 2.
    Zero external HTTP or LLM calls during canonicalization.
    """
    if conn is None:
        conn = get_db(load_vec=True)
    else:
        ensure_schema_tables(conn)

    active_kb = kb if kb is not None else get_active_kb(conn, nlp=nlp)

    ident = _parse_surface_identity(
        name=name, kind=kind, namespace=namespace, proposed_ns=proposed_ns, nlp=nlp
    )
    surface_id = ident.surface_id
    slug = ident.slug
    effective_ns = ident.namespace
    legacy_slug = slugify(name)

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
        legacy_slug=legacy_slug,
        nlp=nlp,
    )
    if res is not None:
        return res

    # TIER 2: Vector Near Match (Space S KNN + Space L Linguistic Corroboration)
    if kind != "predicate":
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
                definition_ctx=definition_ctx,
                embedding=embedding,
            )
            if res is not None:
                return res

    # Induction: Novel canonical creation
    if kind == "predicate":
        pred_canonical = (
            f"predicates:{slug}" if not slug.startswith("predicates:") else slug
        )
        _record_alias(
            conn=conn,
            alias_id=name.strip(),
            canonical_id=pred_canonical,
            kind="predicate",
            method=MatchMethod.NEW.value,
            source_leaf_id=source_leaf_id,
            kb=active_kb,
        )
        return ResolutionResult(
            canonical_id=pred_canonical,
            method=MatchMethod.NEW.value,
            tier=TierEnum.NEW_INDUCTION,
            score=1.0,
            is_new=True,
        )

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
        tier=TierEnum.NEW_INDUCTION,
        score=1.0,
        is_new=True,
    )


__all__ = [
    "MatchMethod",
    "Resolution",
    "ResolutionResult",
    "SurfaceIdentity",
    "TierEnum",
    "_create_canonical_node",
    "_normalize_entity_key",
    "_query_entity_vec_knn",
    "_record_alias",
    "_strip_namespace_prefix",
    "check_linguistic_corroboration",
    "clear_kb_cache",
    "create_in_memory_kb",
    "get_active_kb",
    "insert_canon_verdict",
    "is_acronym",
    "is_head_anchored_containment",
    "is_initial_expansion",
    "log_canon_verdict",
    "normalize_token",
    "resolve",
    "set_active_kb",
    "slugify",
    "sync_kb_from_db",
]
