"""SKOS/RDF/OWL graph CRUD, entity label resolution, entity merging, candidate mining, and cascading deletion."""

import difflib
import json
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from yaw.core import (
    LABEL_KIND_PRIORITY,
    SEMANTIC_EDGE_KINDS,
    OpenIERecord,
    StructuredTriple,
    resolve_corpus_path,
    text_processing,
)
from yaw.core import (
    bayesian_noisy_or as bayesian_noisy_or,
)
from yaw.core import (
    pool_confidences as pool_confidences,
)

from .db import batched, write_tx
from .hydration import normalize_json_passage_text, read_corpus_lines
from .vectors import get_entity_vectors

_EXISTING_ID_TABLES: dict[str, tuple[str, str]] = {
    "chunk": ("chunks", "chunk_id"),
    "chunk_vec": ("chunk_vec", "chunk_id"),
    "entity": ("entities", "entity_id"),
    "entity_vec": ("entity_vec", "entity_id"),
    "fact": ("facts", "fact_id"),
    "fact_vec": ("fact_vec", "fact_id"),
    "openie": ("openie_chunks", "chunk_id"),
}

_INIT_DEL_TEMP_TABLES: tuple[str, ...] = (
    "CREATE TEMP TABLE IF NOT EXISTS _del_chunks (chunk_id TEXT PRIMARY KEY)",
    """
    CREATE TEMP TABLE IF NOT EXISTS _affected_edges (
        src_id TEXT NOT NULL,
        dst_id TEXT NOT NULL,
        kind TEXT NOT NULL,
        PRIMARY KEY (src_id, dst_id, kind)
    )
    """,
    "CREATE TEMP TABLE IF NOT EXISTS _affected_facts (fact_id TEXT PRIMARY KEY)",
    "CREATE TEMP TABLE IF NOT EXISTS _affected_entities (entity_id TEXT PRIMARY KEY)",
    "CREATE TEMP TABLE IF NOT EXISTS _orphan_facts (fact_id TEXT PRIMARY KEY)",
    "CREATE TEMP TABLE IF NOT EXISTS _orphan_entities (entity_id TEXT PRIMARY KEY)",
    "DELETE FROM temp._del_chunks",
    "DELETE FROM temp._affected_edges",
    "DELETE FROM temp._affected_facts",
    "DELETE FROM temp._affected_entities",
    "DELETE FROM temp._orphan_facts",
    "DELETE FROM temp._orphan_entities",
)


@lru_cache(maxsize=4096)
def _lexical_name_parts(name: str) -> tuple[str, str]:
    return (
        name.strip().lower(),
        str(text_processing(name)),
    )


def _lexical_similarity_cascade(
    name_a: str,
    name_b: str,
    threshold: float,
    *,
    matcher: difflib.SequenceMatcher | None = None,
) -> float:
    """Run the 3-stage difflib.SequenceMatcher cascade (real_quick_ratio -> quick_ratio -> ratio)."""
    norm_a, tok_a = _lexical_name_parts(name_a)
    norm_b, tok_b = _lexical_name_parts(name_b)
    if not norm_a or not norm_b:
        return 0.0
    if norm_a == norm_b or (tok_a and tok_a == tok_b):
        return 1.0
    if norm_a.isdigit() and norm_b.isdigit():
        return 0.0

    if matcher is None:
        matcher = difflib.SequenceMatcher(autojunk=False)
        matcher.set_seq2(norm_a)
    matcher.set_seq1(norm_b)
    if matcher.real_quick_ratio() >= threshold and matcher.quick_ratio() >= threshold:
        ratio = matcher.ratio()
        if ratio >= threshold:
            return ratio
    return 0.0


def find_entity_synonym_candidates(
    conn: sqlite3.Connection,
    entity_ids: Sequence[str] | None = None,
    threshold: float = 0.8,
    top_k: int = 5,
) -> list[tuple[str, str, float]]:
    """Mine entity synonymy pairs in SQLite where cosine similarity >= threshold without RAM matrices."""
    if top_k <= 0:
        return []
    max_dist = max(0.0, 1.0 - threshold + 1e-6)
    pair_scores: dict[tuple[str, str], float] = {}

    adjudicated: set[tuple[str, str]] = {
        (str(r[0]), str(r[1])) if str(r[0]) < str(r[1]) else (str(r[1]), str(r[0]))
        for r in conn.execute(
            "SELECT src_id, dst_id FROM edges WHERE kind IN ('skos:related', 'skos:broader', 'owl:differentFrom')"
        )
    }

    def _scan_entity_rows(rows_iter: Iterator[sqlite3.Row]) -> None:
        for row in rows_iter:
            src_id = str(row[0])
            blob = row[1]
            knn_cursor = conn.execute(
                """
                SELECT entity_id, (1.0 - dist) AS similarity
                FROM (
                    SELECT entity_id, vec_distance_cosine(embedding, ?) AS dist
                    FROM entity_vec
                    WHERE entity_id != ?
                    ORDER BY dist ASC
                    LIMIT ?
                )
                WHERE dist <= ?
                """,
                (blob, src_id, top_k, max_dist),
            )
            for hit in knn_cursor:
                dst_id = str(hit[0])
                sim = float(hit[1])
                if sim >= threshold:
                    key = (src_id, dst_id) if src_id < dst_id else (dst_id, src_id)
                    if key not in adjudicated and sim > pair_scores.get(key, -1.0):
                        pair_scores[key] = sim

    if entity_ids is None:
        _scan_entity_rows(conn.execute("SELECT entity_id, embedding FROM entity_vec"))
    elif entity_ids:
        for batch in batched(entity_ids):
            placeholders = ",".join("?" for _ in batch)
            cursor = conn.execute(
                f"SELECT entity_id, embedding FROM entity_vec WHERE entity_id IN ({placeholders})",
                tuple(batch),
            )
            _scan_entity_rows(cursor)

    return [(a, b, sim) for (a, b), sim in sorted(pair_scores.items())]


def find_entity_disambiguation_candidates(
    conn: sqlite3.Connection,
    entity_ids: Sequence[str] | None = None,
    *,
    dense_threshold: float = 0.75,
    dense_top_k: int = 5,
    lexical_threshold: float = 0.70,
) -> list[tuple[str, str, float, float]]:
    """Two-channel candidate generation combining sqlite-vec dense kNN and 3-stage SequenceMatcher cascade."""
    adjudicated: set[tuple[str, str]] = {
        (str(r[0]), str(r[1])) if str(r[0]) < str(r[1]) else (str(r[1]), str(r[0]))
        for r in conn.execute(
            "SELECT src_id, dst_id FROM edges WHERE kind IN ('skos:related', 'skos:broader', 'owl:differentFrom')"
        )
    }

    dense_pairs: dict[tuple[str, str], float] = {
        (a, b): sim
        for a, b, sim in find_entity_synonym_candidates(
            conn, entity_ids=entity_ids, threshold=dense_threshold, top_k=dense_top_k
        )
    }

    all_entities = [
        (str(r[0]), str(r[1]), str(r[1]).strip().lower())
        for r in conn.execute("SELECT entity_id, name FROM entities ORDER BY entity_id ASC")
    ]
    entity_names = {eid: name for eid, name, _ in all_entities}
    target_set = set(entity_ids) if entity_ids is not None else None

    lexical_pairs: dict[tuple[str, str], float] = {}
    # Enumerate only pairs incident to the changed entities. The previous
    # triangular scan visited every graph pair even for a one-entity update.
    # Keep canonical orientation: SequenceMatcher can be order-sensitive.
    positions = {eid: i for i, (eid, _, _) in enumerate(all_entities)}
    target_positions = (
        sorted(positions[eid] for eid in target_set if eid in positions) if target_set is not None else None
    )
    indices = range(len(all_entities)) if target_positions is None else target_positions
    matcher = difflib.SequenceMatcher(autojunk=False)
    for i in indices:
        others = range(i + 1, len(all_entities)) if target_set is None else range(len(all_entities))
        for j in others:
            if j == i or (j < i and target_set is not None and all_entities[j][0] in target_set):
                continue
            a, b = sorted((i, j))
            eid_a, name_a, norm_a = all_entities[a]
            eid_b, name_b, norm_b = all_entities[b]
            if not norm_a or not norm_b:
                continue
            key = (eid_a, eid_b)
            if key in adjudicated:
                continue
            matcher.set_seq2(norm_a)
            best_ratio = _lexical_similarity_cascade(name_a, name_b, lexical_threshold, matcher=matcher)
            if best_ratio >= lexical_threshold:
                lexical_pairs[key] = best_ratio

    for row in conn.execute(
        """
        SELECT DISTINCT a.entity_id, b.entity_id
        FROM entity_labels a
        INNER JOIN entity_labels b
          ON a.label_norm = b.label_norm
         AND a.entity_id < b.entity_id
        """
    ):
        ea, eb = str(row[0]), str(row[1])
        if target_set is not None and ea not in target_set and eb not in target_set:
            continue
        key = (ea, eb)
        if key not in adjudicated:
            lexical_pairs[key] = 1.0

    combined_keys = sorted(set(dense_pairs.keys()) | set(lexical_pairs.keys()))
    if not combined_keys:
        return []

    missing_vec_ids: set[str] = set()
    for a, b in combined_keys:
        if (a, b) not in dense_pairs:
            missing_vec_ids.add(a)
            missing_vec_ids.add(b)
    extra_vecs = get_entity_vectors(conn, list(missing_vec_ids)) if missing_vec_ids else {}

    results: list[tuple[str, str, float, float]] = []
    for a, b in combined_keys:
        d_sim = dense_pairs.get((a, b))
        if d_sim is None:
            va = extra_vecs.get(a)
            vb = extra_vecs.get(b)
            if va is not None and vb is not None:
                denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
                d_sim = float(np.dot(va, vb) / denom) if denom > 1e-12 else 0.0
            else:
                d_sim = 0.0
        l_sim = lexical_pairs.get((a, b))
        if l_sim is None:
            l_sim = _lexical_similarity_cascade(entity_names.get(a, ""), entity_names.get(b, ""), 0.0)
        results.append((a, b, float(d_sim), float(l_sim)))

    return results


def deserialize_openie_record(
    raw_json: str | Mapping[str, Any],
    *,
    chunk_id: str | None = None,
) -> OpenIERecord:
    """Deserialize an OpenIERecord JSON payload or mapping from chunk_inference or openie_chunks."""
    data = json.loads(raw_json) if isinstance(raw_json, str) else raw_json
    cid = str(chunk_id if chunk_id is not None else data.get("chunk_id", ""))
    raw_entities = data.get("entities") or ()
    raw_defs = data.get("entity_definitions") or ()
    entities_list: list[str] = []
    inferred_defs: list[tuple[str, str]] = []
    has_inferred_def = False
    for item in raw_entities:
        if isinstance(item, (list, tuple)) and len(item) >= 1:
            ename = str(item[0]).strip()
            edef = str(item[1]).strip() if len(item) >= 2 and item[1] is not None else ""
            if ename:
                entities_list.append(ename)
                inferred_defs.append((ename, edef))
                if edef:
                    has_inferred_def = True
        elif item is not None:
            ename = str(item).strip()
            if ename:
                entities_list.append(ename)
                inferred_defs.append((ename, ""))
    explicit_defs = tuple(
        (str(pair[0]), str(pair[1])) for pair in raw_defs if isinstance(pair, (list, tuple)) and len(pair) >= 2
    )
    return OpenIERecord(
        chunk_id=cid,
        entities=tuple(entities_list),
        triples=tuple(
            (str(t[0]), str(t[1]), str(t[2]))
            for t in (data.get("triples") or ())
            if isinstance(t, (list, tuple)) and len(t) >= 3
        ),
        structured_triples=tuple(
            StructuredTriple(
                subj=str(st.get("subj", "")),
                subj_def=str(st.get("subj_def", "")),
                pred=str(st.get("pred", "")),
                obj=str(st.get("obj", "")),
                obj_def=str(st.get("obj_def", "")),
                spo_def=str(st.get("spo_def", "")),
                obj_kind="literal" if str(st.get("obj_kind", "entity")) == "literal" else "entity",
                obj_datatype=str(st.get("obj_datatype", "")),
                obj_lang=str(st.get("obj_lang", "")),
            )
            for st in (data.get("structured_triples") or ())
            if isinstance(st, Mapping) and "subj" in st and "pred" in st and "obj" in st
        ),
        entity_definitions=explicit_defs if explicit_defs else (tuple(inferred_defs) if has_inferred_def else ()),
    )


def get_openie_checkpoints(
    conn: sqlite3.Connection,
    chunk_ids: Sequence[str],
) -> dict[str, OpenIERecord]:
    """Retrieve cached OpenIE extraction records for chunk_ids in bounded SQL batches."""
    if not chunk_ids:
        return {}
    results: dict[str, OpenIERecord] = {}
    for batch in batched(chunk_ids):
        placeholders = ",".join("?" for _ in batch)
        cursor = conn.execute(
            f"""
            SELECT chunk_id, entities_json, triples_json, structured_triples_json
            FROM openie_chunks WHERE chunk_id IN ({placeholders})
            """,
            tuple(batch),
        )
        for row in cursor:
            cid = str(row[0])
            results[cid] = deserialize_openie_record(
                {
                    "chunk_id": cid,
                    "entities": json.loads(row[1]) if row[1] else (),
                    "triples": json.loads(row[2]) if row[2] else (),
                    "structured_triples": json.loads(row[3]) if row[3] else (),
                },
                chunk_id=cid,
            )
    missing_ids = [cid for cid in chunk_ids if cid not in results]
    if (
        missing_ids
        and conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunk_inference' LIMIT 1").fetchone()
        is not None
    ):
        for batch in batched(missing_ids):
            placeholders = ",".join("?" for _ in batch)
            cursor = conn.execute(
                f"SELECT chunk_id, extraction_json FROM chunk_inference WHERE chunk_id IN ({placeholders})",
                tuple(batch),
            )
            for row in cursor:
                cid = str(row[0])
                results[cid] = deserialize_openie_record(row[1] or "{}", chunk_id=cid)
    return results


def _serialize_openie_entities(rec: OpenIERecord) -> str:
    if rec.entity_definitions and any(edef for _, edef in rec.entity_definitions):
        def_map = {ename: edef for ename, edef in rec.entity_definitions}
        out: list[list[str]] = []
        seen: set[str] = set()
        for ename in rec.entities:
            seen.add(ename)
            out.append([ename, def_map.get(ename, "")])
        for ename, edef in rec.entity_definitions:
            if ename not in seen:
                seen.add(ename)
                out.append([ename, edef])
        return json.dumps(out)
    return json.dumps(list(rec.entities))


def save_openie_checkpoints(
    conn: sqlite3.Connection,
    records: Sequence[OpenIERecord],
) -> None:
    """Save OpenIE extraction records into openie_chunks for crash-resilient caching."""
    if not records:
        return
    with write_tx(conn):
        conn.executemany(
            """
            INSERT OR REPLACE INTO openie_chunks (
                chunk_id, entities_json, triples_json, structured_triples_json
            ) VALUES (?, ?, ?, ?)
            """,
            (
                (
                    rec.chunk_id,
                    _serialize_openie_entities(rec),
                    json.dumps([list(t) for t in rec.triples]),
                    json.dumps(
                        [
                            {
                                "subj": st.subj,
                                "subj_def": st.subj_def,
                                "pred": st.pred,
                                "obj": st.obj,
                                "obj_def": st.obj_def,
                                "spo_def": st.spo_def,
                                "obj_kind": st.obj_kind,
                                "obj_datatype": st.obj_datatype,
                                "obj_lang": st.obj_lang,
                            }
                            for st in rec.structured_triples
                        ]
                    ),
                )
                for rec in records
            ),
        )


def upsert_entities(
    conn: sqlite3.Connection,
    entities: Sequence[tuple[str, str] | tuple[str, str, str] | tuple[str, str, str, float | None]],
) -> None:
    """Upsert entity nodes (entity_id, name[, definition, confidence]) into entities and align existing SKOS labels."""
    if not entities:
        return
    normalized_rows: list[tuple[str, str, str, float | None]] = []
    for item in entities:
        if len(item) == 2:
            normalized_rows.append((str(item[0]), str(item[1]), "", None))
        elif len(item) == 3:
            normalized_rows.append((str(item[0]), str(item[1]), str(item[2]), None))
        else:
            c = float(item[3]) if item[3] is not None else None
            normalized_rows.append((str(item[0]), str(item[1]), str(item[2]), c))
    with write_tx(conn):
        conn.executemany(
            """
            INSERT INTO entities (entity_id, name, definition, confidence)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(entity_id) DO UPDATE SET
                name = excluded.name,
                definition = CASE
                    WHEN TRIM(excluded.definition) != '' THEN excluded.definition
                    ELSE entities.definition
                END,
                confidence = COALESCE(excluded.confidence, entities.confidence)
            """,
            normalized_rows,
        )
        for eid, ename, _, _ in normalized_rows:
            raw_norm = ename.strip().lower()
            tok_norm = str(text_processing(ename))
            if not raw_norm and not tok_norm:
                continue
            existing_rows = conn.execute(
                "SELECT DISTINCT label_norm, label_kind FROM entity_labels WHERE entity_id = ?",
                (eid,),
            ).fetchall()
            if not existing_rows:
                continue
            existing_norms = {str(r[0]) for r in existing_rows}
            pref_norm = (
                tok_norm
                if (tok_norm and tok_norm in existing_norms)
                else (raw_norm if raw_norm in existing_norms else "")
            )
            if pref_norm:
                conn.execute(
                    """
                    UPDATE entity_labels
                    SET label_kind = 'skos:prefLabel'
                    WHERE entity_id = ? AND label_norm = ? AND label_kind != 'skos:prefLabel'
                    """,
                    (eid, pref_norm),
                )
                conn.execute(
                    """
                    UPDATE entity_labels
                    SET label_kind = 'skos:altLabel'
                    WHERE entity_id = ? AND label_norm != ? AND label_kind = 'skos:prefLabel'
                    """,
                    (eid, pref_norm),
                )


def upsert_concept_schemes(
    conn: sqlite3.Connection,
    schemes: Sequence[tuple[str, str, str, str, float | None]],
) -> None:
    """Insert or replace concept schemes inside a write transaction."""
    if not schemes:
        return
    with write_tx(conn):
        conn.executemany(
            """
            INSERT INTO concept_schemes (scheme_id, pref_label, definition, history_note, confidence)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(scheme_id) DO UPDATE SET
                pref_label = excluded.pref_label,
                definition = excluded.definition,
                history_note = excluded.history_note,
                confidence = excluded.confidence
            """,
            schemes,
        )


def upsert_concept_scheme_members(
    conn: sqlite3.Connection,
    members: Sequence[tuple[str, str, int]],
) -> None:
    """Insert or replace concept scheme memberships inside a write transaction."""
    if not members:
        return
    with write_tx(conn):
        conn.executemany(
            """
            INSERT INTO concept_scheme_members (scheme_id, entity_id, is_top_concept)
            VALUES (?, ?, ?)
            ON CONFLICT(scheme_id, entity_id) DO UPDATE SET
                is_top_concept = excluded.is_top_concept
            """,
            members,
        )


def upsert_entity_labels(
    conn: sqlite3.Connection,
    labels: Sequence[tuple[str, str, str, str, str]],
) -> None:
    """Upsert SKOS entity labels while enforcing SKOS S13/S14 pairwise disjointness across all chunk_ids."""
    if not labels:
        return
    with write_tx(conn):
        affected_eids = list({item[1] for item in labels})
        entity_names: dict[str, str] = {}
        existing_pref: dict[str, str] = {}
        for batch in batched(affected_eids):
            placeholders = ",".join("?" for _ in batch)
            for row in conn.execute(
                f"""
                SELECT entity_id, name
                FROM entities
                WHERE entity_id IN ({placeholders})
                """,
                tuple(batch),
            ):
                if str(row[1]).strip():
                    entity_names[str(row[0])] = str(row[1]).strip()
            for row in conn.execute(
                f"""
                SELECT entity_id, label_norm
                FROM entity_labels
                WHERE label_kind = 'skos:prefLabel' AND entity_id IN ({placeholders})
                """,
                tuple(batch),
            ):
                existing_pref[str(row[0])] = str(row[1])

        for label_norm, entity_id, label_kind, raw_label, chunk_id in labels:
            norm = label_norm.strip().lower()
            if not norm:
                continue
            if label_kind not in LABEL_KIND_PRIORITY:
                raise ValueError(f"Invalid SKOS label_kind: {label_kind!r}")

            ename = entity_names.get(entity_id)
            curr_pref = existing_pref.get(entity_id)
            existing_kinds = [
                str(r[0])
                for r in conn.execute(
                    "SELECT DISTINCT label_kind FROM entity_labels WHERE entity_id = ? AND label_norm = ?",
                    (entity_id, norm),
                )
            ]

            if ename is not None:
                raw_ename_norm = ename.strip().lower()
                tok_ename_norm = str(text_processing(ename))
                canon_ename_norm = tok_ename_norm or raw_ename_norm
                matches_entity_name = norm in (canon_ename_norm, raw_ename_norm) or (
                    bool(tok_ename_norm) and str(text_processing(norm)) == tok_ename_norm
                )
                if matches_entity_name:
                    if curr_pref is not None and curr_pref != norm:
                        curr_matches_name = curr_pref in (canon_ename_norm, raw_ename_norm) or (
                            bool(tok_ename_norm) and str(text_processing(curr_pref)) == tok_ename_norm
                        )
                        if curr_pref == canon_ename_norm or (curr_matches_name and norm != canon_ename_norm):
                            # Entity already has canonical prefLabel norm; demote variant punctuation norm to altLabel
                            best_kind = "skos:altLabel"
                        else:
                            # Demote stale or punctuated prefLabel to altLabel and promote canonical name norm
                            conn.execute(
                                """
                                UPDATE entity_labels
                                SET label_kind = 'skos:altLabel'
                                WHERE entity_id = ? AND label_norm = ? AND label_kind = 'skos:prefLabel'
                                """,
                                (entity_id, curr_pref),
                            )
                            existing_pref[entity_id] = norm
                            best_kind = "skos:prefLabel"
                    else:
                        existing_pref[entity_id] = norm
                        best_kind = "skos:prefLabel"
                else:
                    candidate_kind = (
                        "skos:altLabel" if label_kind in ("skos:prefLabel", "skos:altLabel") else "skos:hiddenLabel"
                    )
                    if candidate_kind == "skos:altLabel" or any(
                        ek in ("skos:prefLabel", "skos:altLabel") for ek in existing_kinds
                    ):
                        best_kind = "skos:altLabel"
                    else:
                        best_kind = "skos:hiddenLabel"
            else:
                effective_kind = label_kind
                if curr_pref is not None and curr_pref == norm:
                    effective_kind = "skos:prefLabel"
                elif effective_kind == "skos:prefLabel":
                    if curr_pref is not None and curr_pref != norm:
                        effective_kind = "skos:altLabel"
                    else:
                        existing_pref[entity_id] = norm
                best_kind = effective_kind
                for ek in existing_kinds:
                    if LABEL_KIND_PRIORITY.get(ek, 0) > LABEL_KIND_PRIORITY.get(best_kind, 0):
                        best_kind = ek

            conn.execute(
                "\n"
                "                INSERT INTO entity_labels (label_norm, entity_id, label_kind, raw_label, "
                "chunk_id)\n"
                "                VALUES (?, ?, ?, ?, ?)\n"
                "                ON CONFLICT(label_norm, entity_id, chunk_id) DO UPDATE SET\n"
                "                    label_kind = excluded.label_kind,\n"
                "                    raw_label = CASE WHEN excluded.raw_label != '' THEN "
                "excluded.raw_label ELSE entity_labels.raw_label END\n"
                "                ",
                (norm, entity_id, best_kind, raw_label or label_norm, chunk_id),
            )
            if existing_kinds and any(k != best_kind for k in existing_kinds):
                conn.execute(
                    "UPDATE entity_labels SET label_kind = ? WHERE entity_id = ? AND label_norm = ?",
                    (best_kind, entity_id, norm),
                )


def get_entities_by_label_norm(
    conn: sqlite3.Connection,
    label_norms: Sequence[str],
) -> dict[str, list[str]]:
    """Look up entity_ids matching each normalized label across skos:prefLabel, altLabel, and hiddenLabel."""
    if not label_norms:
        return {}
    clean_norms = list(dict.fromkeys(n.strip().lower() for n in label_norms if n.strip()))
    if not clean_norms:
        return {}
    results: dict[str, list[str]] = {n: [] for n in clean_norms}
    for batch in batched(clean_norms):
        placeholders = ",".join("?" for _ in batch)
        cursor = conn.execute(
            f"""
            SELECT label_norm, entity_id,
                   MIN(CASE label_kind
                       WHEN 'skos:prefLabel' THEN 1
                       WHEN 'skos:altLabel' THEN 2
                       ELSE 3
                   END) AS prio
            FROM entity_labels
            WHERE label_norm IN ({placeholders})
            GROUP BY label_norm, entity_id
            ORDER BY label_norm ASC, prio ASC, entity_id ASC
            """,
            tuple(batch),
        )
        for row in cursor:
            norm = str(row[0])
            eid = str(row[1])
            if eid not in results[norm]:
                results[norm].append(eid)
    return results


def resolve_entity_labels(
    conn: sqlite3.Connection,
    label_norms: Sequence[str],
    query_vec: np.ndarray | Sequence[float] | None = None,
) -> dict[str, list[str]]:
    """Resolve surface labels to entity_ids, disambiguating polysemous/homograph senses by query_vec cosine."""
    by_norm = get_entities_by_label_norm(conn, label_norms)
    if not by_norm:
        return {}
    if query_vec is None:
        return by_norm

    ambiguous_eids: list[str] = [eid for eids in by_norm.values() if len(eids) > 1 for eid in eids]
    if not ambiguous_eids:
        return by_norm

    q_arr = np.asarray(query_vec, dtype=np.float32)
    q_norm = float(np.linalg.norm(q_arr))
    if q_norm <= 1e-12:
        return by_norm

    vecs = get_entity_vectors(conn, list(dict.fromkeys(ambiguous_eids)))
    resolved: dict[str, list[str]] = {}
    for norm, eids in by_norm.items():
        if len(eids) <= 1:
            resolved[norm] = eids
            continue
        scored: list[tuple[float, str]] = []
        for eid in eids:
            v = vecs.get(eid)
            if v is not None and v.size == q_arr.size:
                v_norm = float(np.linalg.norm(v))
                sim = float(np.dot(q_arr, v) / (q_norm * v_norm)) if v_norm > 1e-12 else -1.0
            else:
                sim = -1.0
            scored.append((sim, eid))
        scored.sort(key=lambda item: (-item[0], item[1]))
        resolved[norm] = [scored[0][1]]
    return resolved


def canonicalize_entity_labels_edc(
    conn: sqlite3.Connection,
    corpus_root: Path | str | None = None,
    *,
    entity_ids: Sequence[str] | None = None,
    hidden_label_norms_by_entity: Mapping[str, set[str]] | None = None,
) -> int:
    """Run deterministic end-of-round Active EDC label canonicalization over entity_labels and entities.

    For each candidate entity:
    1. Demotes any label norm in `hidden_label_norms_by_entity` or absent verbatim from its hydrated
       source chunk on disk (when `corpus_root` is available and chunk is non-stale) to `skos:hiddenLabel`.
    2. Scores all eligible (`skos:prefLabel` / `skos:altLabel`) label norms by provenance tuple:
       `(verbatim_in_source, in_ast_title, support_count, is_cased_or_multiword, -len(raw_label), label_norm)`.
    3. Promotes the single winning norm to `skos:prefLabel` and updates `entities.name` to its best
       surface `raw_label`; demotes all other non-hidden norms to `skos:altLabel` (enforcing SKOS S13 & S14).
    Returns the number of entities whose canonical `skos:prefLabel` or `entities.name` was updated.
    """
    root_path = Path(corpus_root) if corpus_root is not None else None

    with write_tx(conn):
        if entity_ids is not None:
            cand_eids = list(dict.fromkeys(str(eid) for eid in entity_ids if str(eid).strip()))
        else:
            cand_eids = [
                str(r[0])
                for r in conn.execute(
                    """
                    SELECT entity_id FROM entities
                    UNION
                    SELECT entity_id FROM entity_labels
                    ORDER BY entity_id ASC
                    """
                )
            ]

        if not cand_eids:
            return 0

        updated_entities_count = 0
        chunk_text_cache: dict[str, str | None] = {}

        for batch in batched(cand_eids, 500):
            batch_tuple = tuple(batch)
            placeholders = ",".join("?" for _ in batch)

            # 1. Fetch current entity names
            entity_names: dict[str, str] = {}
            for row in conn.execute(
                f"SELECT entity_id, name FROM entities WHERE entity_id IN ({placeholders})",
                batch_tuple,
            ):
                entity_names[str(row[0])] = str(row[1])

            # 2. Fetch all entity_labels for batch
            labels_by_eid: dict[str, dict[str, list[tuple[str, str, str]]]] = {eid: {} for eid in batch}
            chunk_ids_to_fetch: set[str] = set()
            for row in conn.execute(
                f"""
                SELECT entity_id, label_norm, label_kind, raw_label, chunk_id
                FROM entity_labels
                WHERE entity_id IN ({placeholders})
                """,
                batch_tuple,
            ):
                eid, lnorm, lkind, rlabel, cid = (
                    str(row[0]),
                    str(row[1]),
                    str(row[2]),
                    str(row[3]),
                    str(row[4]),
                )
                if eid in labels_by_eid:
                    labels_by_eid[eid].setdefault(lnorm, []).append((lkind, rlabel, cid))
                    if not cid.startswith("__"):
                        chunk_ids_to_fetch.add(cid)

            # 3. Fetch incident facts for support counting
            facts_by_eid: dict[str, list[tuple[str, str]]] = {eid: [] for eid in batch}
            for row in conn.execute(
                f"""
                SELECT fact_id, subj_id, obj_id, raw_triple_json
                FROM facts
                WHERE subj_id IN ({placeholders}) OR obj_id IN ({placeholders})
                """,
                batch_tuple + batch_tuple,
            ):
                fid = str(row[0])
                sid = str(row[1])
                oid = str(row[2]) if row[2] is not None else ""
                raw_json = str(row[3] or "").lower()
                if sid in facts_by_eid:
                    facts_by_eid[sid].append((fid, raw_json))
                if oid in facts_by_eid and oid != sid:
                    facts_by_eid[oid].append((fid, raw_json))

            # 4. Fetch chunk metadata for AST titles and verbatim verification
            chunks_meta: dict[str, tuple[str, int, int, str]] = {}
            unseen_cids = [cid for cid in chunk_ids_to_fetch if cid not in chunk_text_cache]
            if unseen_cids:
                for c_batch in batched(unseen_cids, 500):
                    c_placeholders = ",".join("?" for _ in c_batch)
                    for row in conn.execute(
                        f"""
                        SELECT chunk_id, rel_posix_path, start_line, end_line, title
                        FROM chunks
                        WHERE chunk_id IN ({c_placeholders})
                        """,
                        tuple(c_batch),
                    ):
                        chunks_meta[str(row[0])] = (
                            str(row[1]),
                            int(row[2]),
                            int(row[3]),
                            str(row[4] or ""),
                        )

            def _get_chunk_content(cid: str, meta: dict[str, tuple[str, int, int, str]]) -> str | None:
                if cid in chunk_text_cache:
                    return chunk_text_cache[cid]
                if root_path is None or cid not in meta:
                    chunk_text_cache[cid] = None
                    return None
                rel_path, s_line, e_line, title = meta[cid]
                target_path = resolve_corpus_path(root_path, rel_path)
                try:
                    if not target_path.is_file():
                        chunk_text_cache[cid] = None
                        return None
                    text = read_corpus_lines(target_path, s_line, e_line)
                    if text is not None and rel_path.lower().endswith((".json", ".jsonl", ".ndjson")):
                        text = normalize_json_passage_text(text, default_title=title)
                    chunk_text_cache[cid] = text
                    return text
                except OSError:
                    chunk_text_cache[cid] = None
                    return None

            name_updates: list[tuple[str, str]] = []
            label_updates: list[tuple[str, str, str]] = []

            for eid in batch:
                norms_map = labels_by_eid.get(eid, {})
                curr_name = entity_names.get(eid, "")

                # If entity has no labels but has a name, synthesize a prefLabel
                if not norms_map and curr_name:
                    synth_norm = str(text_processing(curr_name)) or curr_name.strip().lower()
                    conn.execute(
                        """
                        INSERT INTO entity_labels (label_norm, entity_id, label_kind, raw_label, chunk_id)
                        VALUES (?, ?, 'skos:prefLabel', ?, '__auto__')
                        """,
                        (synth_norm, eid, curr_name),
                    )
                    norms_map[synth_norm] = [("skos:prefLabel", curr_name, "__auto__")]

                if not norms_map:
                    continue

                curr_pref_norms = {
                    norm for norm, occs in norms_map.items() if any(occ[0] == "skos:prefLabel" for occ in occs)
                }

                raw_hidden = (
                    hidden_label_norms_by_entity.get(eid, set()) if hidden_label_norms_by_entity is not None else set()
                )
                explicit_hidden = {str(n).strip().lower() for n in raw_hidden} | {
                    str(text_processing(str(n))) for n in raw_hidden
                }

                eid_chunk_titles = [
                    chunks_meta[cid][3].strip().lower()
                    for occs in norms_map.values()
                    for _, _, cid in occs
                    if cid in chunks_meta and chunks_meta[cid][3].strip()
                ]

                norm_scores: dict[str, tuple[int, int, int, int, int, str]] = {}
                best_raw_by_norm: dict[str, str] = {}
                is_hidden_by_norm: dict[str, bool] = {}

                for norm, occs in norms_map.items():
                    raw_labels = list(dict.fromkeys(r[1].strip() for r in occs if r[1].strip()))
                    if not raw_labels:
                        raw_labels = [curr_name if curr_name.strip() else norm]

                    def _surface_key(rl: str) -> tuple[int, int, str]:
                        cased_or_multi = 1 if (any(c.isupper() for c in rl) or len(rl.split()) > 1) else 0
                        return (cased_or_multi, -len(rl), rl)

                    best_raw = max(raw_labels, key=_surface_key)
                    best_raw_by_norm[norm] = best_raw
                    is_cased_or_multiword = (
                        1 if (any(c.isupper() for c in best_raw) or len(best_raw.split()) > 1) else 0
                    )

                    if root_path is None:
                        verbatim_in_source = 1
                    else:
                        disk_grounded = 0
                        disk_failed = 0
                        for _, rlabel, cid in occs:
                            lbl_to_check = (rlabel.strip() or best_raw).lower()
                            if cid.startswith("__") or cid not in chunks_meta:
                                continue
                            c_text = _get_chunk_content(cid, chunks_meta)
                            if c_text is None:
                                disk_failed += 1
                                continue
                            if lbl_to_check in c_text.lower():
                                disk_grounded += 1
                            else:
                                disk_failed += 1

                        verbatim_in_source = 0 if (disk_failed > 0 and disk_grounded == 0) else 1

                    in_ast_title = 1 if any(rl.lower() in t for rl in raw_labels for t in eid_chunk_titles) else 0

                    chunk_count = len({cid for _, _, cid in occs})
                    facts_matching = {
                        fid
                        for fid, f_json in facts_by_eid.get(eid, [])
                        if any(rl.lower() in f_json for rl in raw_labels) or norm in f_json
                    }
                    support_count = chunk_count + len(facts_matching)

                    is_hidden = (norm in explicit_hidden) or (verbatim_in_source == 0)
                    is_hidden_by_norm[norm] = is_hidden

                    norm_scores[norm] = (
                        verbatim_in_source,
                        in_ast_title,
                        support_count,
                        is_cased_or_multiword,
                        -len(best_raw),
                        norm,
                    )

                eligible_norms = [n for n in norms_map if not is_hidden_by_norm[n]]
                hidden_norms = [n for n in norms_map if is_hidden_by_norm[n]]

                if not eligible_norms:
                    best_hidden = max(hidden_norms, key=lambda n: norm_scores[n])
                    eligible_norms.append(best_hidden)
                    hidden_norms.remove(best_hidden)

                winning_norm = max(eligible_norms, key=lambda n: norm_scores[n])
                winning_raw = best_raw_by_norm[winning_norm]

                if winning_raw != curr_name or curr_pref_norms != {winning_norm}:
                    updated_entities_count += 1

                if winning_raw != curr_name:
                    name_updates.append((winning_raw, eid))

                for norm, occs in norms_map.items():
                    if norm == winning_norm:
                        target_kind = "skos:prefLabel"
                    elif norm in hidden_norms:
                        target_kind = "skos:hiddenLabel"
                    else:
                        target_kind = "skos:altLabel"

                    if any(occ[0] != target_kind for occ in occs):
                        label_updates.append((target_kind, eid, norm))

            if name_updates:
                conn.executemany("UPDATE entities SET name = ? WHERE entity_id = ?", name_updates)
            if label_updates:
                conn.executemany(
                    "UPDATE entity_labels SET label_kind = ? WHERE entity_id = ? AND label_norm = ?",
                    label_updates,
                )

        return updated_entities_count


def merge_entities(
    conn: sqlite3.Connection,
    src_id: str,
    dst_id: str,
    label_kind: str = "skos:altLabel",
    corpus_root: Path | str | None = None,
    merged_definition: str | None = None,
) -> None:
    "Atomically collapse src_id into dst_id, demoting src_id's prefLabel to label_kind and re-pointing all edges/facts."
    if src_id == dst_id:
        return
    if label_kind not in ("skos:altLabel", "skos:hiddenLabel"):
        raise ValueError(f"merge_entities label_kind must be skos:altLabel or skos:hiddenLabel, got {label_kind!r}")

    with write_tx(conn):
        src_row = conn.execute("SELECT name, definition FROM entities WHERE entity_id = ?", (src_id,)).fetchone()
        dst_row = conn.execute("SELECT name, definition FROM entities WHERE entity_id = ?", (dst_id,)).fetchone()
        if src_row is None or dst_row is None:
            return

        src_name, src_def = str(src_row[0]), str(src_row[1])
        dst_name, dst_def = str(dst_row[0]), str(dst_row[1])
        dst_raw_norm = dst_name.strip().lower()
        dst_pref_norm = str(text_processing(dst_name)) or dst_raw_norm
        clean_merged_def = merged_definition.strip() if merged_definition else ""

        if clean_merged_def:
            conn.execute(
                "UPDATE entities SET definition = ? WHERE entity_id = ?",
                (clean_merged_def, dst_id),
            )
        elif not dst_def.strip() and src_def.strip():
            conn.execute(
                "UPDATE entities SET definition = ? WHERE entity_id = ?",
                (src_def, dst_id),
            )

        # 1. Re-point and demote src_id labels onto dst_id while maintaining SKOS S13
        src_labels = conn.execute(
            "SELECT label_norm, label_kind, raw_label, chunk_id FROM entity_labels WHERE entity_id = ?",
            (src_id,),
        ).fetchall()
        conn.execute("DELETE FROM entity_labels WHERE entity_id = ?", (src_id,))
        labels_to_upsert: list[tuple[str, str, str, str, str]] = []
        if src_labels:
            for l_norm, l_kind, raw_lbl, cid in src_labels:
                norm_str = str(l_norm)
                new_kind = (
                    "skos:prefLabel"
                    if norm_str in (dst_pref_norm, dst_raw_norm)
                    else (label_kind if str(l_kind) == "skos:prefLabel" else str(l_kind))
                )
                labels_to_upsert.append((norm_str, dst_id, new_kind, str(raw_lbl), str(cid)))
        else:
            norm_str = str(text_processing(src_name)) or src_name.strip().lower()
            new_kind = "skos:prefLabel" if norm_str in (dst_pref_norm, dst_raw_norm) else label_kind
            labels_to_upsert.append((norm_str, dst_id, new_kind, src_name, "__merged__"))
        upsert_entity_labels(conn, labels_to_upsert)

        # 2. Re-point facts (subj_id and obj_id), prune self-loop entity facts, and deduplicate colliding facts
        conn.execute(
            """
            UPDATE facts
            SET subj_id = ?,
                obj_id = CASE WHEN obj_kind = 'literal' AND obj_id = ? THEN NULL ELSE obj_id END
            WHERE subj_id = ?
            """,
            (dst_id, dst_id, src_id),
        )
        conn.execute(
            """
            UPDATE facts
            SET obj_id = CASE WHEN obj_kind = 'literal' AND subj_id = ? THEN NULL ELSE ? END
            WHERE obj_id = ?
            """,
            (dst_id, dst_id, src_id),
        )
        self_loop_fids = [
            str(r[0])
            for r in conn.execute(
                "SELECT fact_id FROM facts WHERE obj_kind = 'entity' AND subj_id = ? AND obj_id = ?",
                (dst_id, dst_id),
            ).fetchall()
        ]
        for fid in self_loop_fids:
            conn.execute("DELETE FROM fact_sources WHERE fact_id = ?", (fid,))
            conn.execute("DELETE FROM fact_vec WHERE fact_id = ?", (fid,))
            conn.execute("DELETE FROM fact_fts WHERE fact_id = ?", (fid,))
            conn.execute("DELETE FROM facts WHERE fact_id = ?", (fid,))

        dup_fact_groups = conn.execute(
            """
            SELECT
                MIN(fact_id) AS keep_fid,
                subj_id,
                pred,
                obj_kind,
                COALESCE(obj_id, '') AS oid,
                COALESCE(obj_literal_lex, '') AS olex,
                obj_datatype,
                obj_lang
            FROM facts
            WHERE subj_id = ? OR obj_id = ?
            GROUP BY
                subj_id, pred, obj_kind,
                COALESCE(obj_id, ''), COALESCE(obj_literal_lex, ''),
                obj_datatype, obj_lang
            HAVING COUNT(*) > 1
            """,
            (dst_id, dst_id),
        ).fetchall()
        for row in dup_fact_groups:
            keep_fid = str(row[0])
            dup_fids = [
                str(r[0])
                for r in conn.execute(
                    """
                    SELECT fact_id FROM facts
                    WHERE fact_id != ?
                      AND subj_id = ?
                      AND pred = ?
                      AND obj_kind = ?
                      AND COALESCE(obj_id, '') = ?
                      AND COALESCE(obj_literal_lex, '') = ?
                      AND obj_datatype = ?
                      AND obj_lang = ?
                    """,
                    (
                        keep_fid,
                        str(row[1]),
                        str(row[2]),
                        str(row[3]),
                        str(row[4]),
                        str(row[5]),
                        str(row[6]),
                        str(row[7]),
                    ),
                ).fetchall()
            ]
            for dfid in dup_fids:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO fact_sources (fact_id, chunk_id)
                    SELECT ?, chunk_id FROM fact_sources WHERE fact_id = ?
                    """,
                    (keep_fid, dfid),
                )
                conn.execute("DELETE FROM fact_sources WHERE fact_id = ?", (dfid,))
                conn.execute("DELETE FROM fact_vec WHERE fact_id = ?", (dfid,))
                conn.execute("DELETE FROM fact_fts WHERE fact_id = ?", (dfid,))
                conn.execute("DELETE FROM facts WHERE fact_id = ?", (dfid,))

        # 3. Re-point edge_sources
        conn.execute(
            """
            INSERT OR IGNORE INTO edge_sources (src_id, dst_id, kind, chunk_id, atomic_confidence)
            SELECT
                CASE WHEN src_id = ? THEN ? ELSE src_id END,
                CASE WHEN dst_id = ? THEN ? ELSE dst_id END,
                kind,
                chunk_id,
                atomic_confidence
            FROM edge_sources
            WHERE (src_id = ? OR dst_id = ?)
              AND (CASE WHEN src_id = ? THEN ? ELSE src_id END) != (CASE WHEN dst_id = ? THEN ? ELSE dst_id END)
            """,
            (src_id, dst_id, src_id, dst_id, src_id, src_id, src_id, dst_id, src_id, dst_id),
        )
        conn.execute(
            "DELETE FROM edge_sources WHERE src_id = ? OR dst_id = ?",
            (src_id, src_id),
        )

        # 4. Re-point edges (excluding self-loops)
        incident_edges = conn.execute(
            "SELECT src_id, dst_id, kind, weight FROM edges WHERE src_id = ? OR dst_id = ?",
            (src_id, src_id),
        ).fetchall()
        conn.execute(
            "DELETE FROM edges WHERE src_id = ? OR dst_id = ?",
            (src_id, src_id),
        )
        for e_src, e_dst, e_kind, e_weight in incident_edges:
            new_src = dst_id if str(e_src) == src_id else str(e_src)
            new_dst = dst_id if str(e_dst) == src_id else str(e_dst)
            kind_str = str(e_kind)
            weight_val = float(e_weight)
            if new_src == new_dst:
                continue
            if kind_str == "passage":
                conn.execute(
                    """
                    INSERT INTO edges (src_id, dst_id, kind, weight)
                    VALUES (?, ?, 'passage', ?)
                    ON CONFLICT(src_id, dst_id, kind) DO UPDATE SET
                        weight = MAX(edges.weight, excluded.weight)
                    """,
                    (new_src, new_dst, weight_val),
                )
            elif kind_str == "fact":
                conn.execute(
                    """
                    INSERT INTO edges (src_id, dst_id, kind, weight)
                    VALUES (?, ?, 'fact', ?)
                    ON CONFLICT(src_id, dst_id, kind) DO UPDATE SET
                        weight = excluded.weight
                    """,
                    (new_src, new_dst, weight_val),
                )
                conn.execute(
                    """
                    UPDATE edges
                    SET weight = COALESCE(
                        (
                            SELECT 1.0 - EXP(SUM(LN(MAX(
                                1e-6, 1.0 - MIN(1.0 - 1e-6, COALESCE(es.atomic_confidence, 0.5))
                            ))))
                            FROM edge_sources es
                            WHERE es.src_id = edges.src_id AND es.dst_id = edges.dst_id AND es.kind = edges.kind
                        ),
                        edges.weight
                    )
                    WHERE src_id = ? AND dst_id = ? AND kind = 'fact'
                    """,
                    (new_src, new_dst),
                )
            elif kind_str in ("skos:related", "skos:broader", "owl:differentFrom"):
                upsert_semantic_edges(conn, [(new_src, new_dst, kind_str, weight_val)])

        # 5. Remove src_id from entity_vec and entities
        conn.execute("DELETE FROM entity_vec WHERE entity_id = ?", (src_id,))
        conn.execute("DELETE FROM entities WHERE entity_id = ?", (src_id,))

        # 6. If corpus_root is provided, run Active EDC label canonicalization on dst_id
        if corpus_root is not None:
            src_norms = {str(text_processing(src_name)) or src_name.strip().lower()}
            for sl in src_labels:
                src_norms.add(str(sl[0]))
            src_demote_norms = src_norms - {dst_pref_norm, dst_raw_norm}
            hidden_map = {dst_id: src_demote_norms} if label_kind == "skos:hiddenLabel" else None
            canonicalize_entity_labels_edc(
                conn,
                corpus_root=corpus_root,
                entity_ids=[dst_id],
                hidden_label_norms_by_entity=hidden_map,
            )
            if clean_merged_def:
                conn.execute("UPDATE entities SET definition = ? WHERE entity_id = ?", (clean_merged_def, dst_id))
            elif src_def.strip():
                curr_dst_name = conn.execute("SELECT name FROM entities WHERE entity_id = ?", (dst_id,)).fetchone()
                if curr_dst_name is not None and str(curr_dst_name[0]) == src_name:
                    conn.execute("UPDATE entities SET definition = ? WHERE entity_id = ?", (src_def, dst_id))


def upsert_fact_sources(
    conn: sqlite3.Connection,
    fact_sources: Sequence[tuple[str, str] | tuple[str, str, float | None, float | None, float | None]],
) -> None:
    """Bulk record (fact_id, chunk_id[, llm_prob, vec_coherence, atomic_confidence]) provenance in fact_sources."""
    if not fact_sources:
        return
    rows_to_insert: list[tuple[str, str, float | None, float | None, float | None]] = []
    for item in fact_sources:
        fid = str(item[0])
        cid = str(item[1])
        if not fid or not cid:
            continue
        p_llm = float(item[2]) if len(item) > 2 and item[2] is not None else None
        s_vec = float(item[3]) if len(item) > 3 and item[3] is not None else None
        c_atom = float(item[4]) if len(item) > 4 and item[4] is not None else None
        rows_to_insert.append((fid, cid, p_llm, s_vec, c_atom))
    with write_tx(conn):
        conn.executemany(
            """
            INSERT INTO fact_sources (fact_id, chunk_id, llm_prob, vec_coherence, atomic_confidence)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(fact_id, chunk_id) DO UPDATE SET
                llm_prob = COALESCE(excluded.llm_prob, fact_sources.llm_prob),
                vec_coherence = COALESCE(excluded.vec_coherence, fact_sources.vec_coherence),
                atomic_confidence = COALESCE(excluded.atomic_confidence, fact_sources.atomic_confidence)
            """,
            rows_to_insert,
        )


def upsert_facts(
    conn: sqlite3.Connection,
    facts: Sequence[tuple[Any, ...]],
    *,
    fact_sources: Sequence[tuple[str, str]] | None = None,
) -> None:
    """Upsert fact nodes (supporting both entity and typed literal objects) and index in fact_fts."""
    if not facts:
        return
    normalized_facts: list[tuple[str, str, str, str, str | None, str | None, str, str, str, str]] = []
    for item in facts:
        fid = str(item[0])
        sid = str(item[1])
        pred = str(item[2])
        raw_oid = item[3]
        raw_json = str(item[4])
        fdef = str(item[5]) if len(item) >= 6 and item[5] is not None else ""
        if len(item) >= 7:
            obj_kind = "literal" if str(item[6]).strip().lower() == "literal" else "entity"
            obj_lit = str(item[7]) if len(item) >= 8 and item[7] is not None else None
            obj_dt = str(item[8]) if len(item) >= 9 and item[8] is not None else ""
            obj_lang = str(item[9]) if len(item) >= 10 and item[9] is not None else ""
        elif raw_oid is None:
            obj_kind = "literal"
            try:
                parsed_raw = json.loads(raw_json)
                obj_lit = str(parsed_raw[2]) if isinstance(parsed_raw, list) and len(parsed_raw) >= 3 else ""
            except json.JSONDecodeError, TypeError, IndexError:
                obj_lit = ""
            obj_dt = ""
            obj_lang = ""
        else:
            obj_kind = "entity"
            obj_lit = None
            obj_dt = ""
            obj_lang = ""

        if obj_kind == "literal":
            raw_oid_str = str(raw_oid).strip() if raw_oid is not None else ""
            oid_val: str | None = raw_oid_str if (raw_oid_str and raw_oid_str != sid) else None
            if obj_lit is None:
                try:
                    parsed_raw = json.loads(raw_json)
                    obj_lit = str(parsed_raw[2]) if isinstance(parsed_raw, list) and len(parsed_raw) >= 3 else ""
                except json.JSONDecodeError, TypeError, IndexError:
                    obj_lit = ""
        else:
            oid_val = str(raw_oid) if raw_oid is not None else ""
            obj_lit = None
            obj_dt = ""
            obj_lang = ""

        normalized_facts.append((fid, sid, pred, obj_kind, oid_val, obj_lit, obj_dt, obj_lang, raw_json, fdef))

    with write_tx(conn):
        conn.executemany(
            """
            INSERT INTO facts (
                fact_id, subj_id, pred, obj_kind, obj_id, obj_literal_lex,
                obj_datatype, obj_lang, raw_triple_json, definition
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(fact_id) DO UPDATE SET
                subj_id = excluded.subj_id,
                pred = excluded.pred,
                obj_kind = excluded.obj_kind,
                obj_id = excluded.obj_id,
                obj_literal_lex = excluded.obj_literal_lex,
                obj_datatype = excluded.obj_datatype,
                obj_lang = excluded.obj_lang,
                raw_triple_json = excluded.raw_triple_json,
                definition = CASE
                    WHEN TRIM(excluded.definition) != '' THEN excluded.definition
                    ELSE facts.definition
                END
            """,
            normalized_facts,
        )
        if fact_sources:
            conn.executemany(
                "INSERT OR IGNORE INTO fact_sources (fact_id, chunk_id) VALUES (?, ?)",
                ((str(fid), str(cid)) for fid, cid in fact_sources if fid and cid),
            )
        conn.executemany(
            "DELETE FROM fact_fts WHERE fact_id = ?",
            ((fid,) for fid, _, _, _, _, _, _, _, _, _ in normalized_facts),
        )
        fts_items: list[tuple[str, str]] = []
        for fid, _, pred, _, _, obj_lit, _, _, raw_json, _ in normalized_facts:
            try:
                items = json.loads(raw_json)
                fts_items.append((fid, f"{items[0]} {pred} {items[2]}"))
            except json.JSONDecodeError, TypeError, IndexError:
                fts_items.append((fid, f"{pred} {obj_lit or ''}".strip()))
        if fts_items:
            conn.executemany(
                "INSERT INTO fact_fts (fact_id, text) VALUES (?, ?)",
                fts_items,
            )


def upsert_edges_with_sources(
    conn: sqlite3.Connection,
    edges_with_sources: Sequence[tuple[str, str, str, float, str] | tuple[str, str, str, float, str, str]],
) -> None:
    """Bulk upsert graph edges and record chunk provenance in edge_sources with Bayesian weight aggregation."""
    if not edges_with_sources:
        return
    with write_tx(conn):
        conn.executemany(
            """
            INSERT INTO edge_sources (src_id, dst_id, kind, chunk_id, atomic_confidence)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(src_id, dst_id, kind, chunk_id) DO UPDATE SET
                atomic_confidence = COALESCE(excluded.atomic_confidence, edge_sources.atomic_confidence)
            """,
            ((row[0], row[1], row[2], row[4], float(row[3])) for row in edges_with_sources),
        )
        conn.executemany(
            """
            INSERT INTO edges (src_id, dst_id, kind, weight, reasoning)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(src_id, dst_id, kind) DO UPDATE SET
                weight = excluded.weight,
                reasoning = CASE WHEN excluded.reasoning != '' THEN excluded.reasoning ELSE edges.reasoning END
            """,
            ((row[0], row[1], row[2], row[3], str(row[5]) if len(row) > 5 else "") for row in edges_with_sources),
        )
        affected_edges = {(str(row[0]), str(row[1]), str(row[2])) for row in edges_with_sources}
        conn.executemany(
            """
            UPDATE edges
            SET weight = COALESCE(
                (
                    SELECT 1.0 - EXP(SUM(LN(MAX(1e-6, 1.0 - MIN(1.0 - 1e-6, COALESCE(es.atomic_confidence, 0.5))))))
                    FROM edge_sources es
                    WHERE es.src_id = edges.src_id AND es.dst_id = edges.dst_id AND es.kind = edges.kind
                ),
                edges.weight
            )
            WHERE src_id = ? AND dst_id = ? AND kind = ?
            """,
            affected_edges,
        )


def get_existing_ids(
    conn: sqlite3.Connection,
    table_kind: str,
    candidate_ids: Sequence[str],
) -> set[str]:
    """Check which candidate IDs already exist in the given table in bounded SQL batches."""
    if not candidate_ids:
        return set()
    if table_kind not in _EXISTING_ID_TABLES:
        raise ValueError(f"Unknown table kind: {table_kind!r}. Expected one of {list(_EXISTING_ID_TABLES.keys())}")
    table, col = _EXISTING_ID_TABLES[table_kind]
    existing: set[str] = set()
    for batch in batched(candidate_ids):
        placeholders = ",".join("?" for _ in batch)
        cursor = conn.execute(
            f"SELECT {col} FROM {table} WHERE {col} IN ({placeholders})",
            tuple(batch),
        )
        for row in cursor:
            existing.add(str(row[0]))
    return existing


def get_facts_by_ids(
    conn: sqlite3.Connection,
    fact_ids: Sequence[str],
) -> dict[str, tuple[str, str, str]]:
    """Retrieve original (subj, pred, obj) triples for candidate fact_ids in bounded batches."""
    if not fact_ids:
        return {}
    results: dict[str, tuple[str, str, str]] = {}
    for batch in batched(fact_ids):
        placeholders = ",".join("?" for _ in batch)
        cursor = conn.execute(
            f"SELECT fact_id, raw_triple_json FROM facts WHERE fact_id IN ({placeholders})",
            tuple(batch),
        )
        for row in cursor:
            fid = str(row[0])
            raw_list = json.loads(row[1])
            results[fid] = (str(raw_list[0]), str(raw_list[1]), str(raw_list[2]))
    return results


def get_facts_with_meta_by_ids(
    conn: sqlite3.Connection,
    fact_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Retrieve enriched fact records with statement, subject, object definitions, and literal metadata."""
    if not fact_ids:
        return {}
    results: dict[str, dict[str, Any]] = {}
    for batch in batched(fact_ids):
        placeholders = ",".join("?" for _ in batch)
        cursor = conn.execute(
            f"""
            SELECT
                f.fact_id,
                f.subj_id,
                f.pred,
                f.obj_id,
                f.raw_triple_json,
                f.definition,
                COALESCE(es.definition, '') AS subj_def,
                COALESCE(eo.definition, '') AS obj_def,
                f.obj_kind,
                f.obj_literal_lex,
                f.obj_datatype,
                f.obj_lang
            FROM facts f
            LEFT JOIN entities es ON es.entity_id = f.subj_id
            LEFT JOIN entities eo ON eo.entity_id = f.obj_id
            WHERE f.fact_id IN ({placeholders})
            """,
            tuple(batch),
        )
        for row in cursor:
            fid = str(row[0])
            raw_list = json.loads(row[4])
            item: dict[str, Any] = {
                "fact_id": fid,
                "subj_id": str(row[1]),
                "pred": str(row[2]),
                "obj_id": str(row[3]) if row[3] is not None else None,
                "triple": (str(raw_list[0]), str(raw_list[1]), str(raw_list[2])),
                "definition": str(row[5]),
                "subj_def": str(row[6]),
                "obj_def": str(row[7]),
            }
            obj_kind = str(row[8] or "entity")
            if obj_kind != "entity" or row[9] is not None or row[10] or row[11]:
                item["obj_kind"] = obj_kind
                item["obj_literal_lex"] = str(row[9]) if row[9] is not None else None
                item["obj_datatype"] = str(row[10] or "")
                item["obj_lang"] = str(row[11] or "")
            results[fid] = item
    return results


def get_entity_doc_frequencies(
    conn: sqlite3.Connection,
    entity_ids: Sequence[str],
) -> dict[str, int]:
    """Count number of passage edges incident to given entity_ids for PPR seed weighting."""
    if not entity_ids:
        return {}
    freqs: dict[str, int] = {eid: 0 for eid in entity_ids}
    for batch in batched(entity_ids):
        placeholders = ",".join("?" for _ in batch)
        cursor = conn.execute(
            f"""
            SELECT dst_id, COUNT(*)
            FROM edges
            WHERE kind = 'passage' AND dst_id IN ({placeholders})
            GROUP BY dst_id
            """,
            tuple(batch),
        )
        for row in cursor:
            freqs[str(row[0])] = int(row[1])
    return freqs


def get_entity_metadata_batch(
    conn: sqlite3.Connection,
    entity_ids: Sequence[str],
) -> dict[str, tuple[str, str]]:
    """Fetch {entity_id: (name, definition)} in 900-parameter SQLite batches."""
    if not entity_ids:
        return {}
    results: dict[str, tuple[str, str]] = {}
    for batch in batched(entity_ids, 900):
        placeholders = ",".join("?" for _ in batch)
        for row in conn.execute(
            f"SELECT entity_id, name, COALESCE(definition, '') FROM entities WHERE entity_id IN ({placeholders})",
            tuple(batch),
        ):
            results[str(row[0])] = (str(row[1]), str(row[2]))
    return results


def get_graph_counts(conn: sqlite3.Connection) -> tuple[int, int, int, int, int]:
    """Return (chunks_count, entities_count, facts_count, edges_count, openie_count) in a single query."""
    has_chunk_inference = (
        conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunk_inference' LIMIT 1").fetchone()
        is not None
    )
    openie_expr = (
        "MAX((SELECT COUNT(*) FROM openie_chunks), (SELECT COUNT(*) FROM chunk_inference))"
        if has_chunk_inference
        else "(SELECT COUNT(*) FROM openie_chunks)"
    )
    row = conn.execute(
        f"""
        SELECT
            (SELECT COUNT(*) FROM chunks),
            (SELECT COUNT(*) FROM entities),
            (SELECT COUNT(*) FROM facts),
            (SELECT COUNT(*) FROM edges),
            {openie_expr}
        """
    ).fetchone()
    if row is None:
        return (0, 0, 0, 0, 0)
    return (int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]))


def load_graph_topology(
    conn: sqlite3.Connection,
    *,
    include_kinds: bool = True,
) -> tuple[tuple[str, ...], tuple[str, ...], list[tuple[str, str, str, float]], dict[str, float | None]]:
    """Load all active chunk IDs, entity IDs, typed edges, and entity priors from yaw.db."""
    chunk_cursor = conn.execute("SELECT chunk_id FROM chunks ORDER BY chunk_id ASC")
    chunk_ids = tuple(str(row[0]) for row in chunk_cursor)

    entity_cursor = conn.execute("SELECT entity_id, confidence FROM entities ORDER BY entity_id ASC")
    entity_ids_list: list[str] = []
    entity_priors: dict[str, float | None] = {}
    for row in entity_cursor:
        eid = str(row[0])
        entity_ids_list.append(eid)
        entity_priors[eid] = float(row[1]) if row[1] is not None else None

    # Include emergent skos:ConceptScheme nodes in topology and priors for CSR/PPR
    try:
        scheme_cursor = conn.execute("SELECT scheme_id, confidence FROM concept_schemes ORDER BY scheme_id ASC")
        for row in scheme_cursor:
            sid = str(row[0])
            if sid not in entity_priors:
                entity_ids_list.append(sid)
                entity_priors[sid] = float(row[1]) if row[1] is not None else None
    except sqlite3.OperationalError:
        pass
    entity_ids = tuple(entity_ids_list)

    edge_cursor = conn.execute(
        """
        SELECT
            f.subj_id AS src_id,
            f.obj_id AS dst_id,
            'fact' AS kind,
            COALESCE(
                f.confidence,
                (SELECT 1.0 - EXP(SUM(LN(MAX(1e-6, 1.0 - MIN(1.0 - 1e-6, COALESCE(fs.atomic_confidence, 0.5))))))
                 FROM fact_sources fs WHERE fs.fact_id = f.fact_id),
                0.5
            ) AS weight
        FROM facts f
        WHERE f.obj_id IS NOT NULL
          AND f.obj_id != ''
          AND f.subj_id != f.obj_id
          AND COALESCE(f.obj_kind, 'entity') = 'entity'
          AND NOT EXISTS (
              SELECT 1 FROM edges e
              WHERE e.src_id = f.subj_id AND e.dst_id = f.obj_id AND e.kind = 'fact'
          )
        UNION ALL
        SELECT
            e.src_id AS src_id,
            e.dst_id AS dst_id,
            e.kind AS kind,
            e.weight AS weight
        FROM edges e
        WHERE e.src_id != e.dst_id
          AND e.weight > 0.0
          AND e.kind != 'owl:differentFrom'
          AND (
              e.kind NOT IN ('passage', 'skos:related', 'synonym')
              OR e.src_id < e.dst_id
              OR NOT EXISTS (
                  SELECT 1 FROM edges rev
                  WHERE rev.src_id = e.dst_id
                    AND rev.dst_id = e.src_id
                    AND rev.kind = e.kind
                    AND rev.weight > 0.0
              )
          )
        ORDER BY src_id ASC, dst_id ASC, kind ASC
        """
    )
    edges_with_kinds = [(str(row[0]), str(row[1]), str(row[2]), float(row[3])) for row in edge_cursor]
    return chunk_ids, entity_ids, edges_with_kinds, entity_priors


SemanticEdgeRow = tuple[str, str, str, float] | tuple[str, str, str, float, str]


def upsert_semantic_edges(
    conn: sqlite3.Connection,
    edges: Sequence[SemanticEdgeRow],
) -> None:
    """Bulk upsert symmetric skos:related / owl:differentFrom and directed skos:broader edges."""
    if not edges:
        return
    with write_tx(conn):
        for row in edges:
            ent1 = str(row[0])
            ent2 = str(row[1])
            kind = str(row[2])
            weight = float(row[3])
            reasoning = str(row[4]).strip() if len(row) > 4 and row[4] is not None else ""
            if ent1 == ent2:
                continue
            if kind not in SEMANTIC_EDGE_KINDS:
                raise ValueError(f"Unsupported semantic edge kind: {kind!r}")
            if kind == "owl:differentFrom":
                conn.execute(
                    """
                    DELETE FROM edges
                    WHERE kind IN ('skos:related', 'skos:broader')
                      AND ((src_id = ? AND dst_id = ?) OR (src_id = ? AND dst_id = ?))
                    """,
                    (ent1, ent2, ent2, ent1),
                )
                for s, d in ((ent1, ent2), (ent2, ent1)):
                    conn.execute(
                        """
                        INSERT INTO edges (src_id, dst_id, kind, weight, reasoning)
                        VALUES (?, ?, 'owl:differentFrom', 0.0, ?)
                        ON CONFLICT(src_id, dst_id, kind) DO UPDATE SET
                            weight = 0.0,
                            reasoning = CASE
                                WHEN excluded.reasoning != '' THEN excluded.reasoning
                                ELSE edges.reasoning
                            END
                        """,
                        (s, d, reasoning),
                    )
            elif kind == "skos:broader":
                has_diff = conn.execute(
                    """
                    SELECT 1 FROM edges
                    WHERE kind = 'owl:differentFrom'
                      AND ((src_id = ? AND dst_id = ?) OR (src_id = ? AND dst_id = ?))
                    LIMIT 1
                    """,
                    (ent1, ent2, ent2, ent1),
                ).fetchone()
                if has_diff is None:
                    # Enforce SKOS S27 disjointness with skos:related and prevent direct 2-cycles
                    conn.execute(
                        """
                        DELETE FROM edges
                        WHERE (
                            kind = 'skos:related'
                            AND ((src_id = ? AND dst_id = ?) OR (src_id = ? AND dst_id = ?))
                        )
                        OR (kind = 'skos:broader' AND src_id = ? AND dst_id = ?)
                        """,
                        (ent1, ent2, ent2, ent1, ent2, ent1),
                    )
                    broader_weight = max(float(weight), 1e-6)
                    conn.execute(
                        """
                        INSERT INTO edges (src_id, dst_id, kind, weight, reasoning)
                        VALUES (?, ?, 'skos:broader', ?, ?)
                        ON CONFLICT(src_id, dst_id, kind) DO UPDATE SET
                            weight = excluded.weight,
                            reasoning = CASE
                                WHEN excluded.reasoning != '' THEN excluded.reasoning
                                ELSE edges.reasoning
                            END
                        """,
                        (ent1, ent2, broader_weight, reasoning),
                    )
            elif kind == "skos:topConceptOf":
                tc_weight = max(float(weight), 1e-6)
                conn.execute(
                    """
                    INSERT INTO edges (src_id, dst_id, kind, weight, reasoning)
                    VALUES (?, ?, 'skos:topConceptOf', ?, ?)
                    ON CONFLICT(src_id, dst_id, kind) DO UPDATE SET
                        weight = excluded.weight,
                        reasoning = CASE
                            WHEN excluded.reasoning != '' THEN excluded.reasoning
                            ELSE edges.reasoning
                        END
                    """,
                    (ent1, ent2, tc_weight, reasoning),
                )
            else:
                has_conflict = conn.execute(
                    """
                    SELECT 1 FROM edges
                    WHERE kind IN ('owl:differentFrom', 'skos:broader')
                      AND ((src_id = ? AND dst_id = ?) OR (src_id = ? AND dst_id = ?))
                    LIMIT 1
                    """,
                    (ent1, ent2, ent2, ent1),
                ).fetchone()
                if has_conflict is None:
                    rel_weight = max(float(weight), 1e-6)
                    for s, d in ((ent1, ent2), (ent2, ent1)):
                        conn.execute(
                            """
                            INSERT INTO edges (src_id, dst_id, kind, weight, reasoning)
                            VALUES (?, ?, 'skos:related', ?, ?)
                            ON CONFLICT(src_id, dst_id, kind) DO UPDATE SET
                                weight = excluded.weight,
                                reasoning = CASE
                                    WHEN excluded.reasoning != '' THEN excluded.reasoning
                                    ELSE edges.reasoning
                                END
                            """,
                            (s, d, rel_weight, reasoning),
                        )


def delete_chunks(conn: sqlite3.Connection, chunk_ids: Sequence[str]) -> None:
    """Atomically delete chunks and cascade removal strictly across affected edges, facts, labels, and entities."""
    if not chunk_ids:
        return

    with write_tx(conn):
        for stmt in _INIT_DEL_TEMP_TABLES:
            conn.execute(stmt)

        conn.executemany(
            "INSERT OR IGNORE INTO temp._del_chunks (chunk_id) VALUES (?)",
            ((cid,) for cid in chunk_ids),
        )

        # 1. Capture only edges, facts, and entities incident to the deleted chunks before removing provenance
        conn.execute(
            """
            INSERT OR IGNORE INTO temp._affected_edges (src_id, dst_id, kind)
            SELECT src_id, dst_id, kind
            FROM edge_sources
            WHERE chunk_id IN (SELECT chunk_id FROM temp._del_chunks)
            """
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO temp._affected_facts (fact_id)
            SELECT fact_id
            FROM fact_sources
            WHERE chunk_id IN (SELECT chunk_id FROM temp._del_chunks)
            """
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO temp._affected_entities (entity_id)
            SELECT dst_id FROM temp._affected_edges WHERE kind IN ('passage', 'fact')
            UNION
            SELECT src_id FROM temp._affected_edges WHERE kind = 'fact'
            UNION
            SELECT src_id FROM temp._affected_edges WHERE kind = 'passage'
              AND src_id NOT IN (SELECT chunk_id FROM temp._del_chunks)
            UNION
            SELECT subj_id FROM facts WHERE fact_id IN (SELECT fact_id FROM temp._affected_facts)
            UNION
            SELECT obj_id FROM facts
            WHERE obj_id IS NOT NULL AND fact_id IN (SELECT fact_id FROM temp._affected_facts)
            UNION
            SELECT entity_id FROM entity_labels
            WHERE chunk_id IN (SELECT chunk_id FROM temp._del_chunks)
            """
        )
        # Include semantic-linked neighbors of affected entities so isolated aliases are also checked
        conn.execute(
            """
            INSERT OR IGNORE INTO temp._affected_entities (entity_id)
            SELECT dst_id FROM edges
            WHERE kind IN ('skos:related', 'skos:broader', 'owl:differentFrom')
              AND src_id IN (SELECT entity_id FROM temp._affected_entities)
            UNION
            SELECT src_id FROM edges
            WHERE kind IN ('skos:related', 'skos:broader', 'owl:differentFrom')
              AND dst_id IN (SELECT entity_id FROM temp._affected_entities)
            """
        )

        # 2. Remove chunk rows, label provenance, fact provenance, and edge provenance
        conn.executemany(
            "DELETE FROM chunk_fts WHERE chunk_id = ?",
            ((cid,) for cid in chunk_ids),
        )
        conn.execute("DELETE FROM chunk_vec WHERE chunk_id IN (SELECT chunk_id FROM temp._del_chunks)")
        conn.execute("DELETE FROM openie_chunks WHERE chunk_id IN (SELECT chunk_id FROM temp._del_chunks)")
        conn.execute("DELETE FROM chunks WHERE chunk_id IN (SELECT chunk_id FROM temp._del_chunks)")
        conn.execute("DELETE FROM fact_sources WHERE chunk_id IN (SELECT chunk_id FROM temp._del_chunks)")
        conn.execute("DELETE FROM edge_sources WHERE chunk_id IN (SELECT chunk_id FROM temp._del_chunks)")
        conn.execute("DELETE FROM entity_labels WHERE chunk_id IN (SELECT chunk_id FROM temp._del_chunks)")

        # 3. Prune affected passage/fact edges with zero remaining provenance and update surviving fact weights
        conn.execute(
            """
            DELETE FROM edges
            WHERE (src_id, dst_id, kind) IN (SELECT src_id, dst_id, kind FROM temp._affected_edges)
              AND NOT EXISTS (
                  SELECT 1 FROM edge_sources es
                  WHERE es.src_id = edges.src_id
                    AND es.dst_id = edges.dst_id
                    AND es.kind = edges.kind
              )
            """
        )
        conn.execute(
            """
            UPDATE edges
            SET weight = COALESCE(
                (
                    SELECT 1.0 - EXP(SUM(LN(MAX(1e-6, 1.0 - MIN(1.0 - 1e-6, COALESCE(es.atomic_confidence, 0.5))))))
                    FROM edge_sources es
                    WHERE es.src_id = edges.src_id
                      AND es.dst_id = edges.dst_id
                      AND es.kind = edges.kind
                ),
                edges.weight
            )
            WHERE (src_id, dst_id, kind) IN (SELECT src_id, dst_id, kind FROM temp._affected_edges)
            """
        )

        # 4. Prune orphaned facts and fact_vec across both fact_sources and affected fact edges
        conn.execute(
            """
            INSERT OR IGNORE INTO temp._orphan_facts (fact_id)
            SELECT af.fact_id
            FROM temp._affected_facts af
            WHERE NOT EXISTS (
                SELECT 1 FROM fact_sources fs WHERE fs.fact_id = af.fact_id
            )
            """
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO temp._orphan_facts (fact_id)
            SELECT f.fact_id
            FROM facts f
            INNER JOIN temp._affected_edges ae
              ON ae.kind = 'fact' AND ae.src_id = f.subj_id AND ae.dst_id = f.obj_id
            WHERE f.obj_kind = 'entity'
              AND NOT EXISTS (
                  SELECT 1 FROM edges e
                  WHERE e.src_id = f.subj_id AND e.dst_id = f.obj_id AND e.kind = 'fact'
              )
              AND NOT EXISTS (
                  SELECT 1 FROM fact_sources fs WHERE fs.fact_id = f.fact_id
              )
            """
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO temp._orphan_facts (fact_id)
            SELECT f.fact_id
            FROM facts f
            WHERE f.obj_kind = 'literal'
              AND NOT EXISTS (
                  SELECT 1 FROM fact_sources fs WHERE fs.fact_id = f.fact_id
              )
            """
        )
        conn.execute("DELETE FROM fact_sources WHERE fact_id IN (SELECT fact_id FROM temp._orphan_facts)")
        conn.execute("DELETE FROM fact_vec WHERE fact_id IN (SELECT fact_id FROM temp._orphan_facts)")
        conn.execute("DELETE FROM fact_fts WHERE fact_id IN (SELECT fact_id FROM temp._orphan_facts)")
        conn.execute("DELETE FROM facts WHERE fact_id IN (SELECT fact_id FROM temp._orphan_facts)")

        # 5. Prune orphaned entities and entity_vec strictly among candidate entities using index seeks
        conn.execute(
            """
            INSERT OR IGNORE INTO temp._orphan_entities (entity_id)
            SELECT ae.entity_id
            FROM temp._affected_entities ae
            WHERE NOT EXISTS (
                SELECT 1 FROM edges e
                WHERE e.src_id = ae.entity_id AND e.kind IN ('passage', 'fact')
            )
            AND NOT EXISTS (
                SELECT 1 FROM edges e
                WHERE e.dst_id = ae.entity_id AND e.kind IN ('passage', 'fact')
            )
            AND NOT EXISTS (
                SELECT 1 FROM facts f
                WHERE f.subj_id = ae.entity_id OR f.obj_id = ae.entity_id
            )
            """
        )
        conn.execute("DELETE FROM entity_labels WHERE entity_id IN (SELECT entity_id FROM temp._orphan_entities)")
        conn.execute("DELETE FROM entity_vec WHERE entity_id IN (SELECT entity_id FROM temp._orphan_entities)")
        conn.execute("DELETE FROM entities WHERE entity_id IN (SELECT entity_id FROM temp._orphan_entities)")

        # 6. Prune semantic edges incident to deleted entities
        conn.execute(
            """
            DELETE FROM edges
            WHERE kind IN ('skos:related', 'skos:broader', 'owl:differentFrom')
              AND (
                  src_id IN (SELECT entity_id FROM temp._orphan_entities)
                  OR dst_id IN (SELECT entity_id FROM temp._orphan_entities)
              )
            """
        )

        # 7. Promote a surviving label to skos:prefLabel for any surviving affected entity that lost its prefLabel
        surviving_missing_pref = conn.execute(
            """
            SELECT e.entity_id, e.name
            FROM entities e
            INNER JOIN temp._affected_entities ae ON ae.entity_id = e.entity_id
            WHERE NOT EXISTS (
                SELECT 1 FROM entity_labels el
                WHERE el.entity_id = e.entity_id AND el.label_kind = 'skos:prefLabel'
            )
            """
        ).fetchall()
        for r in surviving_missing_pref:
            eid, ename = str(r[0]), str(r[1])
            candidate_lbl = conn.execute(
                """
                SELECT label_norm, raw_label
                FROM entity_labels
                WHERE entity_id = ?
                ORDER BY CASE label_kind WHEN 'skos:altLabel' THEN 1 ELSE 2 END ASC, label_norm ASC
                LIMIT 1
                """,
                (eid,),
            ).fetchone()
            if candidate_lbl is not None:
                lnorm, rlabel = str(candidate_lbl[0]), str(candidate_lbl[1])
                conn.execute(
                    "UPDATE entity_labels SET label_kind = 'skos:prefLabel' WHERE entity_id = ? AND label_norm = ?",
                    (eid, lnorm),
                )
                if rlabel:
                    conn.execute("UPDATE entities SET name = ? WHERE entity_id = ?", (rlabel, eid))
                homograph_peers = [
                    str(hr[0])
                    for hr in conn.execute(
                        """
                        SELECT DISTINCT entity_id FROM entity_labels
                        WHERE label_norm = ? AND label_kind = 'skos:prefLabel' AND entity_id != ?
                        """,
                        (lnorm, eid),
                    ).fetchall()
                ]
                if homograph_peers:
                    upsert_semantic_edges(
                        conn,
                        [(eid, peer_id, "owl:differentFrom", 0.0) for peer_id in homograph_peers],
                    )
            else:
                surviving_chunk = conn.execute(
                    """
                    SELECT chunk_id FROM edge_sources
                    WHERE src_id = ? OR dst_id = ?
                    UNION ALL
                    SELECT fs.chunk_id FROM fact_sources fs
                    INNER JOIN facts f ON f.fact_id = fs.fact_id
                    WHERE f.subj_id = ? OR f.obj_id = ?
                    LIMIT 1
                    """,
                    (eid, eid, eid, eid),
                ).fetchone()
                cid = str(surviving_chunk[0]) if surviving_chunk is not None else "__survived__"
                conn.execute(
                    """
                    INSERT OR IGNORE INTO entity_labels (label_norm, entity_id, label_kind, raw_label, chunk_id)
                    VALUES (?, ?, 'skos:prefLabel', ?, ?)
                    """,
                    (ename.strip().lower(), eid, ename, cid),
                )
