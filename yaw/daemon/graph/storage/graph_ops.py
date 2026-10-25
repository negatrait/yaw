"""SKOS/RDF/OWL graph CRUD, entity label resolution, entity merging, candidate mining, and cascading deletion."""

import difflib
import json
import re
import sqlite3
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np

from yaw.core import (
    ChunkPointer,
    OpenIERecord,
    StructuredTriple,
    check_lexical_subsumption_inversion,
    detect_scalar_literal,
    is_acronym_pair,
    is_version_or_ordinal_sibling,
    text_processing,
)

from .db import YawConnection, batched, write_tx
from .vectors import FTS_STOPWORDS, get_entity_vectors

LABEL_KIND_PRIORITY: dict[str, int] = {
    "skos:prefLabel": 3,
    "skos:altLabel": 2,
    "skos:hiddenLabel": 1,
}

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


def _get_pure_literal_entity_ids(conn: sqlite3.Connection | YawConnection) -> set[str]:
    """Return entity IDs that represent pure rdfs:Literal resources and must not participate in SKOS taxonomy."""
    concept_participant_ids: set[str] = {
        str(r[0])
        for r in conn.execute(
            """
            SELECT subj_id FROM facts
            UNION
            SELECT obj_id FROM facts
            WHERE COALESCE(obj_kind, 'entity') = 'entity' AND obj_id IS NOT NULL
            """
        )
    }
    pure_literal_ids: set[str] = {
        str(r[0])
        for r in conn.execute(
            """
            SELECT DISTINCT obj_id FROM facts
            WHERE obj_kind = 'literal' AND obj_id IS NOT NULL
            """
        )
        if str(r[0]) not in concept_participant_ids
    }
    for row in conn.execute("SELECT entity_id, name FROM entities"):
        eid, name = str(row[0]), str(row[1])
        if eid not in concept_participant_ids and detect_scalar_literal(name)[0]:
            pure_literal_ids.add(eid)
    return pure_literal_ids


def _is_acronym_match(name_a: str, name_b: str) -> bool:
    """Return True if one label is an initialism/acronym of the other multi-token label."""
    if is_acronym_pair(name_a, name_b):
        return True
    clean_a = re.sub(r"[^a-zA-Z0-9\s]", "", name_a).strip()
    clean_b = re.sub(r"[^a-zA-Z0-9\s]", "", name_b).strip()
    if not clean_a or not clean_b:
        return False
    tokens_a = clean_a.split()
    tokens_b = clean_b.split()
    if len(tokens_a) == 1 and len(tokens_b) >= 2:
        short, words = tokens_a[0].lower(), tokens_b
    elif len(tokens_b) == 1 and len(tokens_a) >= 2:
        short, words = tokens_b[0].lower(), tokens_a
    else:
        return False
    if len(short) < 2:
        return False
    initials = "".join(w[0].lower() for w in words if w)
    if short == initials:
        return True
    non_stop_initials = "".join(w[0].lower() for w in words if w and w.lower() not in FTS_STOPWORDS)
    return len(non_stop_initials) >= 2 and short == non_stop_initials


def _lexical_similarity_cascade(
    name_a: str,
    name_b: str,
    threshold: float,
    *,
    matcher: difflib.SequenceMatcher | None = None,
) -> float:
    """Run the 3-stage difflib.SequenceMatcher cascade + acronym and alnum detector."""
    norm_a = name_a.strip().lower()
    norm_b = name_b.strip().lower()
    if not norm_a or not norm_b:
        return 0.0
    if norm_a == norm_b:
        return 1.0
    tok_a = str(text_processing(name_a))
    tok_b = str(text_processing(name_b))
    if tok_a and tok_a == tok_b:
        return 1.0
    if _is_acronym_match(name_a, name_b):
        return max(threshold, 0.85)
    if is_version_or_ordinal_sibling(name_a, name_b):
        return max(threshold, 0.80)
    if norm_a.isdigit() and norm_b.isdigit():
        return 0.0

    len_a, len_b = len(norm_a), len(norm_b)
    if min(len_a, len_b) / max(len_a, len_b) < 0.4:
        return 0.0

    if matcher is None:
        matcher = difflib.SequenceMatcher(autojunk=False)
        matcher.set_seq2(norm_a)
    matcher.set_seq1(norm_b)
    best_ratio = 0.0
    if matcher.real_quick_ratio() >= threshold and matcher.quick_ratio() >= threshold:
        best_ratio = matcher.ratio()
    if best_ratio < threshold and (" " in norm_a or " " in norm_b):
        alnum_a = re.sub(r"[^a-z0-9]", "", norm_a)
        alnum_b = re.sub(r"[^a-z0-9]", "", norm_b)
        if alnum_a and alnum_b and not (alnum_a.isdigit() and alnum_b.isdigit()):
            alnum_ratio = difflib.SequenceMatcher(None, alnum_b, alnum_a, autojunk=False).ratio()
            if alnum_ratio > best_ratio:
                best_ratio = alnum_ratio
    return best_ratio if best_ratio >= threshold else 0.0


def find_entity_synonym_candidates(
    conn: sqlite3.Connection | YawConnection,
    entity_ids: Sequence[str] | None = None,
    threshold: float = 0.8,
    top_k: int = 5,
) -> list[tuple[str, str, float]]:
    """Mine entity synonymy pairs in SQLite where cosine similarity >= threshold without RAM matrices."""
    if top_k <= 0:
        return []
    max_dist = max(0.0, 1.0 - threshold + 1e-6)
    pair_scores: dict[tuple[str, str], float] = {}
    pure_literals = _get_pure_literal_entity_ids(conn)

    adjudicated: set[tuple[str, str]] = {
        (str(r[0]), str(r[1])) if str(r[0]) < str(r[1]) else (str(r[1]), str(r[0]))
        for r in conn.execute(
            "SELECT src_id, dst_id FROM edges WHERE kind IN ('skos:related', 'skos:broader', 'owl:differentFrom')"
        )
    }

    def _scan_entity_rows(rows_iter: Iterator[sqlite3.Row]) -> None:
        for row in rows_iter:
            src_id = str(row[0])
            if src_id in pure_literals:
                continue
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
                if dst_id in pure_literals:
                    continue
                sim = float(hit[1])
                if sim >= threshold:
                    key = (src_id, dst_id) if src_id < dst_id else (dst_id, src_id)
                    if key not in adjudicated and sim > pair_scores.get(key, -1.0):
                        pair_scores[key] = sim

    if entity_ids is None:
        _scan_entity_rows(conn.execute("SELECT entity_id, embedding FROM entity_vec"))
    elif entity_ids:
        filtered_eids = [eid for eid in entity_ids if eid not in pure_literals]
        for batch in batched(filtered_eids):
            placeholders = ",".join("?" for _ in batch)
            cursor = conn.execute(
                f"SELECT entity_id, embedding FROM entity_vec WHERE entity_id IN ({placeholders})",
                tuple(batch),
            )
            _scan_entity_rows(cursor)

    return [(a, b, sim) for (a, b), sim in sorted(pair_scores.items())]


def find_entity_disambiguation_candidates(
    conn: sqlite3.Connection | YawConnection,
    entity_ids: Sequence[str] | None = None,
    *,
    dense_threshold: float = 0.75,
    dense_top_k: int = 5,
    lexical_threshold: float = 0.72,
) -> list[tuple[str, str, float, float]]:
    """Two-channel candidate generation combining sqlite-vec dense kNN and 3-stage SequenceMatcher cascade."""
    pure_literals = _get_pure_literal_entity_ids(conn)
    adjudicated: set[tuple[str, str]] = {
        (str(r[0]), str(r[1])) if str(r[0]) < str(r[1]) else (str(r[1]), str(r[0]))
        for r in conn.execute(
            "SELECT src_id, dst_id FROM edges WHERE kind IN ('skos:related', 'skos:broader', 'owl:differentFrom')"
        )
    }

    dense_pairs: dict[tuple[str, str], float] = {}
    for a, b, sim in find_entity_synonym_candidates(
        conn, entity_ids=entity_ids, threshold=dense_threshold, top_k=dense_top_k
    ):
        if a not in pure_literals and b not in pure_literals:
            dense_pairs[(a, b)] = sim

    all_entities = [
        (str(r[0]), str(r[1]), str(r[1]).strip().lower())
        for r in conn.execute("SELECT entity_id, name FROM entities ORDER BY entity_id ASC")
        if str(r[0]) not in pure_literals
    ]
    entity_names = {eid: name for eid, name, _ in all_entities}
    target_set = (set(entity_ids) - pure_literals) if entity_ids is not None else None

    lexical_pairs: dict[tuple[str, str], float] = {}
    n_ents = len(all_entities)
    matcher = difflib.SequenceMatcher(autojunk=False)
    for i in range(n_ents):
        eid_a, name_a, norm_a = all_entities[i]
        if not norm_a:
            continue
        a_in_target = target_set is None or eid_a in target_set
        matcher_seq2_set = False
        for j in range(i + 1, n_ents):
            eid_b, name_b, norm_b = all_entities[j]
            if not a_in_target and (target_set is not None and eid_b not in target_set):
                continue
            if not norm_b:
                continue
            key = (eid_a, eid_b) if eid_a < eid_b else (eid_b, eid_a)
            if key in adjudicated:
                continue
            if not matcher_seq2_set:
                matcher.set_seq2(norm_a)
                matcher_seq2_set = True
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
        if ea in pure_literals or eb in pure_literals:
            continue
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
            raw_a = entity_names.get(a, "")
            raw_b = entity_names.get(b, "")
            na = raw_a.strip().lower()
            nb = raw_b.strip().lower()
            if _is_acronym_match(raw_a, raw_b):
                l_sim = max(lexical_threshold, 0.85)
            elif na and nb:
                l_sim = difflib.SequenceMatcher(None, nb, na, autojunk=False).ratio()
            else:
                l_sim = 0.0
        results.append((a, b, float(d_sim), float(l_sim)))

    return results


def upsert_chunk_pointers(
    conn: sqlite3.Connection | YawConnection,
    chunks: Sequence[tuple[ChunkPointer, str]],
) -> None:
    """Bulk upsert ChunkPointer metadata into chunks and index passage text in contentless chunk_fts."""
    if not chunks:
        return
    with write_tx(conn):
        conn.executemany(
            "\n"
            "            INSERT OR REPLACE INTO chunks (chunk_id, rel_posix_path, start_line, "
            "end_line, title, ast_island_start, ast_island_end)\n"
            "            VALUES (?, ?, ?, ?, ?, ?, ?)\n"
            "            ",
            (
                (
                    p.chunk_id,
                    p.rel_posix_path,
                    p.start_line,
                    p.end_line,
                    p.title,
                    p.ast_island_start,
                    p.ast_island_end,
                )
                for p, _ in chunks
            ),
        )
        conn.executemany(
            "DELETE FROM chunk_fts WHERE chunk_id = ?",
            ((p.chunk_id,) for p, _ in chunks),
        )
        conn.executemany(
            "INSERT INTO chunk_fts (chunk_id, title, body) VALUES (?, ?, ?)",
            ((p.chunk_id, p.title, body) for p, body in chunks),
        )


def _row_to_chunk_pointer(row: sqlite3.Row) -> ChunkPointer:
    keys = row.keys()
    ast_s = int(row["ast_island_start"]) if "ast_island_start" in keys else 0
    ast_e = int(row["ast_island_end"]) if "ast_island_end" in keys else 0
    return ChunkPointer(
        chunk_id=row["chunk_id"],
        rel_posix_path=row["rel_posix_path"],
        start_line=row["start_line"],
        end_line=row["end_line"],
        title=row["title"],
        ast_island_start=ast_s,
        ast_island_end=ast_e,
    )


def get_chunk_pointer(conn: sqlite3.Connection | YawConnection, chunk_id: str) -> ChunkPointer | None:
    """Retrieve ChunkPointer metadata for a single chunk_id."""
    row = conn.execute(
        "SELECT chunk_id, rel_posix_path, start_line, end_line, title, ast_island_start, "
        "ast_island_end FROM chunks WHERE chunk_id = ?",
        (chunk_id,),
    ).fetchone()
    return _row_to_chunk_pointer(row) if row is not None else None


def get_chunk_pointers(
    conn: sqlite3.Connection | YawConnection,
    chunk_ids: Sequence[str],
) -> dict[str, ChunkPointer]:
    """Batch retrieve ChunkPointer metadata for given chunk_ids."""
    if not chunk_ids:
        return {}
    results: dict[str, ChunkPointer] = {}
    for batch in batched(chunk_ids):
        placeholders = ",".join("?" for _ in batch)
        cursor = conn.execute(
            f"""
            SELECT chunk_id, rel_posix_path, start_line, end_line, title, ast_island_start, ast_island_end
            FROM chunks WHERE chunk_id IN ({placeholders})
            """,
            tuple(batch),
        )
        for row in cursor:
            results[row["chunk_id"]] = _row_to_chunk_pointer(row)
    return results


def get_openie_checkpoints(
    conn: sqlite3.Connection | YawConnection,
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
            raw_entities = json.loads(row[1]) if row[1] else []
            entities_list: list[str] = []
            entity_defs_list: list[tuple[str, str]] = []
            has_any_def = False
            for item in raw_entities:
                if isinstance(item, (list, tuple)) and len(item) >= 1:
                    ename = str(item[0]).strip()
                    edef = str(item[1]).strip() if len(item) >= 2 and item[1] is not None else ""
                    if ename:
                        entities_list.append(ename)
                        entity_defs_list.append((ename, edef))
                        if edef:
                            has_any_def = True
                elif isinstance(item, str):
                    ename = item.strip()
                    if ename:
                        entities_list.append(ename)
                        entity_defs_list.append((ename, ""))
            triples = tuple((t[0], t[1], t[2]) for t in json.loads(row[2]))
            raw_st = json.loads(row[3]) if row[3] else []
            structured_triples = tuple(
                StructuredTriple(
                    subj=str(item["subj"]),
                    subj_def=str(item.get("subj_def", "")),
                    pred=str(item["pred"]),
                    obj=str(item["obj"]),
                    obj_def=str(item.get("obj_def", "")),
                    spo_def=str(item.get("spo_def", "")),
                    obj_kind="literal" if str(item.get("obj_kind", "entity")) == "literal" else "entity",
                    obj_datatype=str(item.get("obj_datatype", "")),
                    obj_lang=str(item.get("obj_lang", "")),
                )
                for item in raw_st
                if isinstance(item, dict) and "subj" in item and "pred" in item and "obj" in item
            )
            results[cid] = OpenIERecord(
                chunk_id=cid,
                entities=tuple(entities_list),
                triples=triples,
                structured_triples=structured_triples,
                entity_definitions=tuple(entity_defs_list) if has_any_def else (),
            )
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
    conn: sqlite3.Connection | YawConnection,
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
    conn: sqlite3.Connection | YawConnection,
    entities: Sequence[tuple[str, str] | tuple[str, str, str]],
) -> None:
    """Upsert entity nodes (entity_id, name[, definition]) into entities and align existing SKOS labels."""
    if not entities:
        return
    normalized_rows: list[tuple[str, str, str]] = []
    for item in entities:
        if len(item) == 2:
            normalized_rows.append((item[0], item[1], ""))
        else:
            normalized_rows.append((item[0], item[1], item[2]))
    with write_tx(conn):
        conn.executemany(
            """
            INSERT INTO entities (entity_id, name, definition)
            VALUES (?, ?, ?)
            ON CONFLICT(entity_id) DO UPDATE SET
                name = excluded.name,
                definition = CASE
                    WHEN TRIM(excluded.definition) != '' THEN excluded.definition
                    ELSE entities.definition
                END
            """,
            normalized_rows,
        )
        for eid, ename, _ in normalized_rows:
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


def upsert_entity_labels(
    conn: sqlite3.Connection | YawConnection,
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
    conn: sqlite3.Connection | YawConnection,
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
    conn: sqlite3.Connection | YawConnection,
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


def merge_entities(
    conn: sqlite3.Connection | YawConnection,
    src_id: str,
    dst_id: str,
    label_kind: str = "skos:altLabel",
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

        if not dst_def.strip() and src_def.strip():
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
            INSERT OR IGNORE INTO edge_sources (src_id, dst_id, kind, chunk_id)
            SELECT
                CASE WHEN src_id = ? THEN ? ELSE src_id END,
                CASE WHEN dst_id = ? THEN ? ELSE dst_id END,
                kind,
                chunk_id
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
                prov_count = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM edge_sources WHERE src_id = ? AND dst_id = ? AND kind = 'fact'",
                        (new_src, new_dst),
                    ).fetchone()[0]
                )
                has_ent_fact = (
                    conn.execute(
                        """
                        SELECT 1 FROM facts
                        WHERE ((subj_id = ? AND obj_id = ?) OR (subj_id = ? AND obj_id = ?))
                          AND COALESCE(obj_kind, 'entity') = 'entity'
                        LIMIT 1
                        """,
                        (new_src, new_dst, new_dst, new_src),
                    ).fetchone()
                    is not None
                )
                has_lit_fact = (
                    conn.execute(
                        """
                        SELECT 1 FROM facts
                        WHERE ((subj_id = ? AND obj_id = ?) OR (subj_id = ? AND obj_id = ?))
                          AND obj_kind = 'literal'
                        LIMIT 1
                        """,
                        (new_src, new_dst, new_dst, new_src),
                    ).fetchone()
                    is not None
                )
                base_mult = 0.5 if (has_lit_fact and not has_ent_fact) else 1.0
                eff_weight = float(prov_count) * base_mult if prov_count > 0 else weight_val
                conn.execute(
                    """
                    INSERT INTO edges (src_id, dst_id, kind, weight)
                    VALUES (?, ?, 'fact', ?)
                    ON CONFLICT(src_id, dst_id, kind) DO UPDATE SET
                        weight = CASE WHEN ? > 0 THEN excluded.weight ELSE edges.weight + excluded.weight END
                    """,
                    (new_src, new_dst, eff_weight, prov_count),
                )
            elif kind_str in ("skos:related", "skos:broader", "owl:differentFrom"):
                upsert_semantic_edges(conn, [(new_src, new_dst, kind_str, weight_val)])

        # 5. Remove src_id from entity_vec and entities
        conn.execute("DELETE FROM entity_vec WHERE entity_id = ?", (src_id,))
        conn.execute("DELETE FROM entities WHERE entity_id = ?", (src_id,))


def upsert_fact_sources(
    conn: sqlite3.Connection | YawConnection,
    fact_sources: Sequence[tuple[str, str]],
) -> None:
    """Bulk record (fact_id, chunk_id) provenance in fact_sources."""
    if not fact_sources:
        return
    with write_tx(conn):
        conn.executemany(
            "INSERT OR IGNORE INTO fact_sources (fact_id, chunk_id) VALUES (?, ?)",
            ((str(fid), str(cid)) for fid, cid in fact_sources if fid and cid),
        )


def upsert_facts(
    conn: sqlite3.Connection | YawConnection,
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
            except Exception:
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
                except Exception:
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
            except Exception:
                fts_items.append((fid, f"{pred} {obj_lit or ''}".strip()))
        if fts_items:
            conn.executemany(
                "INSERT INTO fact_fts (fact_id, text) VALUES (?, ?)",
                fts_items,
            )


def upsert_edges_with_sources(
    conn: sqlite3.Connection | YawConnection,
    edges_with_sources: Sequence[tuple[str, str, str, float, str]],
) -> None:
    """Bulk upsert graph edges and record chunk provenance in edge_sources."""
    if not edges_with_sources:
        return
    with write_tx(conn):
        conn.executemany(
            """
            INSERT OR IGNORE INTO edge_sources (src_id, dst_id, kind, chunk_id)
            VALUES (?, ?, ?, ?)
            """,
            ((src, dst, kind, cid) for src, dst, kind, _, cid in edges_with_sources),
        )
        conn.executemany(
            """
            INSERT INTO edges (src_id, dst_id, kind, weight)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(src_id, dst_id, kind) DO UPDATE SET weight = excluded.weight
            """,
            ((src, dst, kind, weight) for src, dst, kind, weight, _ in edges_with_sources),
        )
        fact_base_weights: dict[tuple[str, str], float] = {}
        for src, dst, kind, weight, _ in edges_with_sources:
            if kind == "fact":
                pair = (src, dst)
                fact_base_weights[pair] = max(fact_base_weights.get(pair, 0.0), float(weight))
        if fact_base_weights:
            conn.executemany(
                """
                UPDATE edges
                SET weight = MAX(
                    ?,
                    CASE
                        WHEN EXISTS (
                            SELECT 1 FROM facts f
                            WHERE f.subj_id = edges.src_id
                              AND f.obj_id = edges.dst_id
                              AND COALESCE(f.obj_kind, 'entity') = 'entity'
                        )
                        THEN 1.0
                        ELSE 0.0
                    END
                ) * (
                    SELECT COUNT(*) FROM edge_sources es
                    WHERE es.src_id = edges.src_id AND es.dst_id = edges.dst_id AND es.kind = 'fact'
                )
                WHERE src_id = ? AND dst_id = ? AND kind = 'fact'
                """,
                ((base_w, src, dst) for (src, dst), base_w in fact_base_weights.items()),
            )


def get_existing_ids(
    conn: sqlite3.Connection | YawConnection,
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
    conn: sqlite3.Connection | YawConnection,
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
    conn: sqlite3.Connection | YawConnection,
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
    conn: sqlite3.Connection | YawConnection,
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
    conn: sqlite3.Connection | YawConnection,
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


def get_graph_counts(conn: sqlite3.Connection | YawConnection) -> tuple[int, int, int, int, int]:
    """Return (chunks_count, entities_count, facts_count, edges_count, openie_count) in a single query."""
    row = conn.execute(
        """
        SELECT
            (SELECT COUNT(*) FROM chunks),
            (SELECT COUNT(*) FROM entities),
            (SELECT COUNT(*) FROM facts),
            (SELECT COUNT(*) FROM edges),
            (SELECT COUNT(*) FROM openie_chunks)
        """
    ).fetchone()
    if row is None:
        return (0, 0, 0, 0, 0)
    return (int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]))


def load_graph_topology(
    conn: sqlite3.Connection | YawConnection,
    *,
    include_kinds: bool = False,
) -> tuple[tuple[str, ...], tuple[str, ...], list[Any]]:
    """Load all active chunk IDs, entity IDs, and positive-weight edges from yaw.db to build the CSR graph."""
    chunk_cursor = conn.execute("SELECT chunk_id FROM chunks ORDER BY chunk_id ASC")
    chunk_ids = tuple(str(row[0]) for row in chunk_cursor)

    entity_cursor = conn.execute("SELECT entity_id FROM entities ORDER BY entity_id ASC")
    entity_ids = tuple(str(row[0]) for row in entity_cursor)

    if include_kinds:
        edge_cursor = conn.execute(
            """
            SELECT
                f.subj_id AS src_id,
                f.obj_id AS dst_id,
                'fact' AS kind,
                (CASE WHEN COALESCE(f.obj_kind, 'entity') = 'literal' THEN 0.5 ELSE 1.0 END)
                * CAST(MAX(1, COUNT(DISTINCT fs.chunk_id)) AS REAL) AS weight
            FROM facts f
            LEFT JOIN fact_sources fs ON fs.fact_id = f.fact_id
            WHERE f.obj_id IS NOT NULL
              AND f.obj_id != ''
              AND f.subj_id != f.obj_id
            GROUP BY f.fact_id, f.subj_id, f.obj_id
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
                  e.kind != 'fact'
                  OR NOT EXISTS (
                      SELECT 1 FROM facts f
                      WHERE f.obj_id IS NOT NULL
                        AND f.obj_id != ''
                        AND ((f.subj_id = e.src_id AND f.obj_id = e.dst_id)
                             OR (f.subj_id = e.dst_id AND f.obj_id = e.src_id))
                  )
              )
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
        return chunk_ids, entity_ids, edges_with_kinds

    edge_cursor = conn.execute(
        """
        SELECT src_id, dst_id, MAX(weight)
        FROM edges
        WHERE src_id != dst_id AND weight > 0.0 AND kind != 'owl:differentFrom'
        GROUP BY src_id, dst_id
        """
    )
    edges = [(str(row[0]), str(row[1]), float(row[2])) for row in edge_cursor]

    return chunk_ids, entity_ids, edges


def _are_same_label_homographs(conn: sqlite3.Connection | YawConnection, ent1: str, ent2: str) -> bool:
    "Return True if ent1 and ent2 share a skos:prefLabel norm or normalized entity name (e.g. 'a ball' vs 'a ball')."
    row = conn.execute(
        """
        SELECT 1
        FROM entities a
        INNER JOIN entities b ON a.entity_id = ? AND b.entity_id = ?
        WHERE LOWER(TRIM(a.name)) != '' AND LOWER(TRIM(a.name)) = LOWER(TRIM(b.name))
        UNION ALL
        SELECT 1
        FROM entity_labels la
        INNER JOIN entity_labels lb
          ON la.label_norm = lb.label_norm
         AND la.label_kind = 'skos:prefLabel'
         AND lb.label_kind = 'skos:prefLabel'
        WHERE la.entity_id = ? AND lb.entity_id = ?
        LIMIT 1
        """,
        (ent1, ent2, ent1, ent2),
    ).fetchone()
    if row is not None:
        return True
    names = conn.execute(
        "SELECT name FROM entities WHERE entity_id IN (?, ?)",
        (ent1, ent2),
    ).fetchall()
    if len(names) == 2:
        na, nb = str(names[0][0]).strip(), str(names[1][0]).strip()
        if na and nb:
            ta, tb = str(text_processing(na)), str(text_processing(nb))
            if ta and ta == tb:
                return True
    return False


def upsert_semantic_edges(
    conn: sqlite3.Connection | YawConnection,
    edges: Sequence[tuple[str, str, str, float]],
) -> None:
    """Bulk upsert symmetric skos:related / owl:differentFrom and directed skos:broader edges."""
    if not edges:
        return
    with write_tx(conn):
        pure_literals = _get_pure_literal_entity_ids(conn)
        for ent1, ent2, kind, weight in edges:
            if ent1 == ent2:
                continue
            if kind not in ("skos:related", "skos:broader", "owl:differentFrom"):
                raise ValueError(f"Unsupported semantic edge kind: {kind!r}")
            if kind in ("skos:related", "skos:broader") and (ent1 in pure_literals or ent2 in pure_literals):
                continue
            if kind == "skos:broader":
                row_s = conn.execute("SELECT name FROM entities WHERE entity_id = ?", (ent1,)).fetchone()
                row_d = conn.execute("SELECT name FROM entities WHERE entity_id = ?", (ent2,)).fetchone()
                name_s = str(row_s[0]) if row_s is not None else ""
                name_d = str(row_d[0]) if row_d is not None else ""
                if name_s and name_d:
                    if is_version_or_ordinal_sibling(name_s, name_d):
                        kind = "skos:related"
                    elif check_lexical_subsumption_inversion(name_s, name_d):
                        ent1, ent2 = ent2, ent1
            is_homograph = _are_same_label_homographs(conn, ent1, ent2)
            if kind == "owl:differentFrom":
                if not is_homograph:
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
                        INSERT INTO edges (src_id, dst_id, kind, weight)
                        VALUES (?, ?, 'owl:differentFrom', 0.0)
                        ON CONFLICT(src_id, dst_id, kind) DO UPDATE SET weight = 0.0
                        """,
                        (s, d),
                    )
            elif kind == "skos:broader":
                has_diff = (
                    None
                    if is_homograph
                    else conn.execute(
                        """
                        SELECT 1 FROM edges
                        WHERE kind = 'owl:differentFrom'
                          AND ((src_id = ? AND dst_id = ?) OR (src_id = ? AND dst_id = ?))
                        LIMIT 1
                        """,
                        (ent1, ent2, ent2, ent1),
                    ).fetchone()
                )
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
                        INSERT INTO edges (src_id, dst_id, kind, weight)
                        VALUES (?, ?, 'skos:broader', ?)
                        ON CONFLICT(src_id, dst_id, kind) DO UPDATE SET weight = excluded.weight
                        """,
                        (ent1, ent2, broader_weight),
                    )
            else:
                has_conflict = (
                    None
                    if is_homograph
                    else conn.execute(
                        """
                        SELECT 1 FROM edges
                        WHERE kind IN ('owl:differentFrom', 'skos:broader')
                          AND ((src_id = ? AND dst_id = ?) OR (src_id = ? AND dst_id = ?))
                        LIMIT 1
                        """,
                        (ent1, ent2, ent2, ent1),
                    ).fetchone()
                )
                if has_conflict is None:
                    rel_weight = max(float(weight), 1e-6)
                    for s, d in ((ent1, ent2), (ent2, ent1)):
                        conn.execute(
                            """
                            INSERT INTO edges (src_id, dst_id, kind, weight)
                            VALUES (?, ?, 'skos:related', ?)
                            ON CONFLICT(src_id, dst_id, kind) DO UPDATE SET weight = excluded.weight
                            """,
                            (s, d, rel_weight),
                        )
                    if is_homograph:
                        for s, d in ((ent1, ent2), (ent2, ent1)):
                            conn.execute(
                                """
                                INSERT INTO edges (src_id, dst_id, kind, weight)
                                VALUES (?, ?, 'owl:differentFrom', 0.0)
                                ON CONFLICT(src_id, dst_id, kind) DO UPDATE SET weight = 0.0
                                """,
                                (s, d),
                            )


def upsert_synonymy_edges(
    conn: sqlite3.Connection | YawConnection,
    edges: Sequence[tuple[str, str, float]],
) -> None:
    """Bulk upsert bidirectional skos:related semantic edges between entity pairs."""
    if not edges:
        return
    upsert_semantic_edges(
        conn,
        [(ent1, ent2, "skos:related", sim) for ent1, ent2, sim in edges],
    )


def delete_chunks(conn: sqlite3.Connection | YawConnection, chunk_ids: Sequence[str]) -> None:
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
            SET weight = (
                CASE
                    WHEN EXISTS (
                        SELECT 1 FROM facts f
                        WHERE ((f.subj_id = edges.src_id AND f.obj_id = edges.dst_id)
                               OR (f.subj_id = edges.dst_id AND f.obj_id = edges.src_id))
                          AND f.obj_kind = 'literal'
                    )
                    AND NOT EXISTS (
                        SELECT 1 FROM facts f
                        WHERE ((f.subj_id = edges.src_id AND f.obj_id = edges.dst_id)
                               OR (f.subj_id = edges.dst_id AND f.obj_id = edges.src_id))
                          AND COALESCE(f.obj_kind, 'entity') = 'entity'
                    )
                    THEN 0.5
                    ELSE 1.0
                END
            ) * (
                SELECT COUNT(*) FROM edge_sources es
                WHERE es.src_id = edges.src_id
                  AND es.dst_id = edges.dst_id
                  AND es.kind = 'fact'
            )
            WHERE kind = 'fact'
              AND (src_id, dst_id, kind) IN (SELECT src_id, dst_id, kind FROM temp._affected_edges)
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
