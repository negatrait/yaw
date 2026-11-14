"""SKOS/RDF/OWL graph staging, entity merging, candidate mining, and topology reads."""

import json
import sqlite3
from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix

from yaw.core import (
    LABEL_KIND_PRIORITY,
    SEMANTIC_EDGE_KINDS,
    OpenIERecord,
    StructuredTriple,
    clean_surface_label,
    is_irreflexive_statement,
    is_resource_label,
    is_scalar_label,
    resolve_corpus_path,
    text_processing,
)
from yaw.core import (
    bayesian_noisy_or as bayesian_noisy_or,
)
from yaw.core import (
    pool_confidences as pool_confidences,
)

from .db import batched, ensure_vec0_table, vec0_table_dimension, write_tx
from .evidence import CandidateVector, EntityMerge, fact_id
from .hydration import normalize_json_passage_text, read_corpus_lines

_EXISTING_ID_TABLES: dict[str, tuple[str, str]] = {
    "chunk": ("chunks", "chunk_id"),
    "chunk_vec": ("chunk_vec", "chunk_id"),
    "entity": ("entities", "entity_id"),
    "entity_vec": ("entity_vec", "entity_id"),
    "fact": ("facts", "fact_id"),
    "fact_vec": ("fact_vec", "fact_id"),
}


def find_entity_candidates(
    conn: sqlite3.Connection,
    entity_ids: Sequence[str] | None = None,
    *,
    dense_threshold: float = 0.8,
    lexical_threshold: float = 0.625,
    exclude_adjudicated: bool = True,
    candidate_vectors: Sequence[CandidateVector] = (),
) -> list[tuple[str, str, float, float]]:
    """Pair-local cosine/trigram candidates, with bounded native scoring blocks.

    Supplied source records let neighborhood discovery use this same policy before
    delta rows are staged. Exact names retain separate sense IDs for judging.
    """
    if entity_ids is not None and not entity_ids:
        return []
    eids = sorted(
        {str(row[0]) for row in conn.execute("SELECT DISTINCT entity_id FROM entity_candidate_vectors")}
        | {row.entity_id for row in candidate_vectors}
    )
    if len(eids) < 2:
        return []
    positions = {eid: i for i, eid in enumerate(eids)}
    targets = (
        sorted({positions[eid] for eid in entity_ids if eid in positions})
        if entity_ids is not None
        else list(range(len(eids)))
    )
    if not targets:
        return []
    labels: list[str] = []
    owners: list[int] = []
    known: set[tuple[str, str]] = set()
    for eid, name in chain(
        conn.execute("SELECT entity_id,name FROM entity_candidate_vectors"),
        ((row.entity_id, row.name) for row in candidate_vectors),
    ):
        if (eid, name) not in known:
            known.add((eid, name))
            owners.append(positions[eid])
            labels.append(name)
    owner = np.array(owners, dtype=np.int32)
    vocabulary: dict[str, int] = {}
    indices: list[int] = []
    indptr = [0]
    codes: dict[str, int] = {"": 0}
    norm_ids = np.array([codes.setdefault(name, len(codes)) for name in labels], dtype=np.int32)
    for name in labels:
        padded = "  " + name + "  "
        grams = {padded[i : i + 3] for i in range(len(padded) - 2)} if name else ()
        indices.extend(vocabulary.setdefault(gram, len(vocabulary)) for gram in grams)
        indptr.append(len(indices))
    lexical = csr_matrix((np.ones(len(indices), dtype=np.int32), indices, indptr), shape=(len(labels), len(vocabulary)))
    lexical.sort_indices()
    lexical_t = lexical.T.tocsr()
    lengths = np.diff(lexical.indptr)
    # Source vectors, never canonical replacement vectors, define eligibility.
    vector_rows = iter(conn.execute("SELECT entity_id,embedding FROM entity_candidate_vectors"))
    first = next(vector_rows, None)
    dimension = (
        len(first[1]) // 4
        if first is not None
        else (len(candidate_vectors[0].embedding) // 4 if candidate_vectors else 0)
    )
    vector_count = conn.execute("SELECT count(*) FROM entity_candidate_vectors").fetchone()[0]
    vectors = np.zeros((vector_count + len(candidate_vectors), dimension), dtype=np.float32)
    vector_owners = []
    for eid, blob in chain(
        (first,) if first is not None else (), vector_rows, ((r.entity_id, r.embedding) for r in candidate_vectors)
    ):
        if eid in positions:
            vectors[len(vector_owners)] = np.frombuffer(blob, dtype="<f4")
            vector_owners.append(positions[eid])
    vectors = vectors[: len(vector_owners)]
    vector_owner = np.array(vector_owners, dtype=np.int32)
    magnitudes = np.sqrt(np.einsum("ij,ij->i", vectors, vectors))[:, None]
    np.divide(vectors, magnitudes, out=vectors, where=magnitudes > 0)
    adjudicated = (
        {
            tuple(sorted((str(a), str(b))))
            for a, b in conn.execute(
                "SELECT src_id, dst_id FROM edges WHERE kind IN ('skos:related', 'skos:broader', 'owl:differentFrom')"
            )
        }
        if exclude_adjudicated
        else set()
    )
    target_mask = np.zeros(len(eids), dtype=bool)
    target_mask[targets] = True
    pair_scores: dict[tuple[str, str], tuple[float, float]] = {}
    query_vectors = np.flatnonzero(target_mask[vector_owner])
    dense = np.empty((min(16, len(query_vectors)), len(vectors)), dtype=np.float32)
    for start in range(0, len(query_vectors), 16):
        source = query_vectors[start : start + 16]
        count = len(source)
        np.matmul(vectors[source], vectors.T, out=dense[:count])
        for qi, j in zip(*np.nonzero(dense[:count] >= dense_threshold), strict=True):
            i = source[qi]
            a, b = sorted((eids[vector_owner[i]], eids[vector_owner[j]]))
            if a == b or (a, b) in adjudicated:
                continue
            old_dense, old_lexical = pair_scores.get((a, b), (-1.0, 0.0))
            pair_scores[a, b] = (max(old_dense, float(dense[qi, j])), old_lexical)
    vectors_by_owner: dict[int, list[int]] = defaultdict(list)
    for i, eid in enumerate(vector_owner):
        vectors_by_owner[int(eid)].append(i)

    # Fuzzy source labels survive canonical display-name changes. No dense
    # B-by-label matrix: score only nonzero sparse overlaps in each small block.
    query_labels = np.flatnonzero(target_mask[owner])
    for start in range(0, len(query_labels), 16):
        source = query_labels[start : start + 16]
        overlap = (lexical[source] @ lexical_t).tocoo()
        rows, cols = source[overlap.row], overlap.col
        scores = (2 * overlap.data) / (lengths[rows] + lengths[cols])
        selected = (scores >= lexical_threshold) | (norm_ids[rows] == norm_ids[cols])
        for row, col, score in zip(rows[selected], cols[selected], scores[selected], strict=True):
            i, j = owner[row], owner[col]
            if i == j:
                continue
            a, b = (eids[i], eids[j]) if i < j else (eids[j], eids[i])
            if (a, b) not in adjudicated and score > pair_scores.get((a, b), (0.0, -1.0))[1]:
                indices_a, indices_b = vectors_by_owner[i], vectors_by_owner[j]
                cosine = max((float(vectors[x] @ vectors[y]) for x in indices_a for y in indices_b), default=0.0)
                pair_scores[a, b] = (cosine, float(score))
    labels_by_owner: dict[int, list[int]] = defaultdict(list)
    for i, eid in enumerate(owner):
        labels_by_owner[int(eid)].append(i)
    for (a, b), (dense_sim, score) in pair_scores.items():
        if score == 0.0:
            score = max(
                (
                    float(2 * lexical[i].multiply(lexical[j]).sum() / (lengths[i] + lengths[j]))
                    for i in labels_by_owner[positions[a]]
                    for j in labels_by_owner[positions[b]]
                    if lengths[i] + lengths[j]
                ),
                default=0.0,
            )
            pair_scores[a, b] = (dense_sim, score)
    return [(a, b, d, score) for (a, b), (d, score) in sorted(pair_scores.items())]


def deserialize_openie_record(
    raw_json: str | Mapping[str, Any],
    *,
    chunk_id: str | None = None,
) -> OpenIERecord:
    """Deserialize an OpenIERecord JSON payload or mapping from the native chunk_inference target."""
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
                llm_prob=st.get("llm_prob"),
            )
            for st in (data.get("structured_triples") or ())
            if isinstance(st, Mapping) and "subj" in st and "pred" in st and "obj" in st
        ),
        entity_definitions=explicit_defs if explicit_defs else (tuple(inferred_defs) if has_inferred_def else ()),
        entity_probabilities=tuple(
            (str(name), str(definition), probability)
            for name, definition, probability in data.get("entity_probabilities", ())
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
            normalized_rows.append((item[0], item[1], "", None))
        elif len(item) == 3:
            normalized_rows.append((item[0], item[1], item[2], None))
        else:
            c = float(item[3]) if item[3] is not None else None
            normalized_rows.append((item[0], item[1], item[2], c))
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
                clean_ename = clean_surface_label(ename) or ename
                conn.execute(
                    """
                    UPDATE entity_labels
                    SET label_kind = 'skos:prefLabel', raw_label = ?
                    WHERE entity_id = ? AND label_norm = ? AND (label_kind != 'skos:prefLabel' OR raw_label != ?)
                    """,
                    (clean_ename, eid, pref_norm, clean_ename),
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

            clean_raw = clean_surface_label(raw_label) or raw_label.strip() or norm
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
                (norm, entity_id, best_kind, clean_raw, chunk_id),
            )
            if existing_kinds and any(k != best_kind for k in existing_kinds):
                conn.execute(
                    "UPDATE entity_labels SET label_kind = ? WHERE entity_id = ? AND label_norm = ?",
                    (best_kind, entity_id, norm),
                )


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

            # 3. Fetch chunk metadata for AST titles and verbatim verification
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
            label_updates: list[tuple[str, str, str, str]] = []

            for eid in batch:
                norms_map = labels_by_eid.get(eid, {})
                curr_name = entity_names.get(eid, "")

                # If entity has no labels but has a name, synthesize a prefLabel
                if not norms_map and curr_name:
                    clean_curr = clean_surface_label(curr_name) or curr_name.strip()
                    synth_norm = str(text_processing(clean_curr)) or clean_curr.lower()
                    conn.execute(
                        """
                        INSERT INTO entity_labels (label_norm, entity_id, label_kind, raw_label, chunk_id)
                        VALUES (?, ?, 'skos:prefLabel', ?, '__auto__')
                        """,
                        (synth_norm, eid, clean_curr),
                    )
                    norms_map[synth_norm] = [("skos:prefLabel", clean_curr, "__auto__")]

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
                    raw_labels = list(
                        dict.fromkeys((clean_surface_label(r[1]) or r[1].strip()) for r in occs if r[1].strip())
                    )
                    if not raw_labels:
                        fallback_name = clean_surface_label(curr_name) or curr_name.strip()
                        raw_labels = [fallback_name if fallback_name else norm]

                    def _surface_key(rl: str, target_norm: str = norm) -> tuple[int, int, int, int, str]:
                        matches_norm = 1 if str(text_processing(rl)) == target_norm else 0
                        is_cased = 1 if any(c.isupper() for c in rl) else 0
                        is_multiword = 1 if len(rl.split()) > 1 else 0
                        return (matches_norm, is_cased, is_multiword, -len(rl), rl)

                    best_raw = max(raw_labels, key=_surface_key)
                    best_raw_by_norm[norm] = best_raw
                    is_cased_or_multiword = (
                        1 if (any(c.isupper() for c in best_raw) or len(best_raw.split()) > 1) else 0
                    )

                    if not is_resource_label(best_raw) or is_scalar_label(best_raw):
                        verbatim_in_source = 0
                    elif root_path is None:
                        verbatim_in_source = 1
                    else:
                        disk_grounded = 0
                        disk_failed = 0
                        for _, rlabel, cid in occs:
                            lbl_to_check = (clean_surface_label(rlabel) or rlabel.strip() or best_raw).lower()
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

                    # Canonical fact text is a result of label selection, not source support.
                    support_count = len({cid for _, _, cid in occs})

                    is_hidden = (
                        (norm in explicit_hidden)
                        or (verbatim_in_source == 0)
                        or not is_resource_label(best_raw)
                        or is_scalar_label(best_raw)
                    )
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

                if (
                    winning_raw != curr_name
                    or curr_pref_norms != {winning_norm}
                    or any(occ[1] != winning_raw for occ in norms_map.get(winning_norm, ()))
                ):
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

                    target_raw = best_raw_by_norm[norm]
                    if any(occ[0] != target_kind or occ[1] != target_raw for occ in occs):
                        label_updates.append((target_kind, target_raw, eid, norm))

            if name_updates:
                conn.executemany("UPDATE entities SET name = ? WHERE entity_id = ?", name_updates)
            if label_updates:
                conn.executemany(
                    "UPDATE entity_labels SET label_kind = ?, raw_label = ? WHERE entity_id = ? AND label_norm = ?",
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
) -> EntityMerge:
    """Collapse equivalent resources to their smallest stable ID, preserving the preferred label."""
    if label_kind not in ("skos:altLabel", "skos:hiddenLabel"):
        raise ValueError(f"Invalid merge label kind: {label_kind!r}")
    if src_id == dst_id:
        return EntityMerge(src_id, dst_id)

    with write_tx(conn):
        requested_src, requested_dst = src_id, dst_id
        if conn.execute("SELECT 1 FROM sqlite_master WHERE name='entity_candidate_vectors'").fetchone():
            conn.execute(
                "UPDATE entity_candidate_vectors SET entity_id=? WHERE entity_id IN (?,?)",
                (min(src_id, dst_id), src_id, dst_id),
            )
        metadata = dict(
            (row[0], (row[1], row[2]))
            for row in conn.execute(
                "SELECT entity_id,name,definition FROM entities WHERE entity_id IN (?,?)", (src_id, dst_id)
            )
        )
        if set(metadata) != {src_id, dst_id}:
            raise ValueError("Entity merge requires both resources to exist")
        src_name, src_def = metadata[requested_src]
        dst_name, dst_def = metadata[requested_dst]
        dst_pref_norm = str(text_processing(dst_name))
        dst_raw_norm = dst_name.strip().lower()
        definition = max((src_def, dst_def, (merged_definition or "").strip()), key=lambda value: (len(value), value))
        dst_id, src_id = sorted((src_id, dst_id))
        conn.execute("UPDATE entities SET name=?,definition=? WHERE entity_id=?", (dst_name, definition, dst_id))
        conn.execute(
            "UPDATE entities SET confidence=(SELECT MAX(confidence) FROM entities WHERE entity_id IN (?,?)) "
            "WHERE entity_id=?",
            (src_id, dst_id, dst_id),
        )

        labels = conn.execute(
            "SELECT entity_id,label_norm,label_kind,raw_label,chunk_id FROM entity_labels "
            "WHERE entity_id IN (?,?) ORDER BY entity_id,label_norm,chunk_id",
            (src_id, dst_id),
        ).fetchall()
        src_labels = [row[1:] for row in labels if row[0] == requested_src]
        conn.execute("DELETE FROM entity_labels WHERE entity_id IN (?,?)", (src_id, dst_id))
        upsert_entity_labels(
            conn,
            [
                (
                    norm,
                    dst_id,
                    "skos:prefLabel"
                    if norm in (dst_pref_norm, dst_raw_norm)
                    else label_kind
                    if owner == requested_src and kind == "skos:prefLabel"
                    else kind,
                    raw,
                    chunk,
                )
                for owner, norm, kind, raw, chunk in labels
            ],
        )

        affected_facts = [
            row[0]
            for row in conn.execute(
                "SELECT fact_id FROM facts WHERE subj_id IN (?,?) OR obj_id IN (?,?)",
                (src_id, dst_id, src_id, dst_id),
            )
        ]
        conn.execute(
            "UPDATE facts SET subj_id=CASE WHEN subj_id=:src THEN :dst ELSE subj_id END, "
            "obj_id=CASE WHEN obj_kind='literal' AND "
            "(CASE WHEN obj_id=:src THEN :dst ELSE obj_id END)="
            "(CASE WHEN subj_id=:src THEN :dst ELSE subj_id END) THEN NULL "
            "WHEN obj_id=:src THEN :dst ELSE obj_id END WHERE subj_id=:src OR obj_id=:src",
            {"src": src_id, "dst": dst_id},
        )
        fact_bindings = rekey_facts(conn, affected_facts)

        # 3. Re-point edge_sources
        conn.execute(
            """
            INSERT INTO edge_sources (src_id, dst_id, kind, chunk_id, llm_prob, vec_coherence, atomic_confidence)
            SELECT
                CASE WHEN src_id = ? THEN ? ELSE src_id END,
                CASE WHEN dst_id = ? THEN ? ELSE dst_id END,
                kind,
                chunk_id,
                llm_prob,
                vec_coherence,
                atomic_confidence
            FROM edge_sources
            WHERE (src_id = ? OR dst_id = ?)
              AND (CASE WHEN src_id = ? THEN ? ELSE src_id END) != (CASE WHEN dst_id = ? THEN ? ELSE dst_id END)
            ON CONFLICT(src_id,dst_id,kind,chunk_id) DO UPDATE SET
                llm_prob=COALESCE(MAX(edge_sources.llm_prob,excluded.llm_prob),edge_sources.llm_prob,excluded.llm_prob),
                vec_coherence=COALESCE(MAX(edge_sources.vec_coherence,excluded.vec_coherence),
                                       edge_sources.vec_coherence,excluded.vec_coherence),
                atomic_confidence=COALESCE(MAX(edge_sources.atomic_confidence,excluded.atomic_confidence),
                                           edge_sources.atomic_confidence,excluded.atomic_confidence)
            """,
            (src_id, dst_id, src_id, dst_id, src_id, src_id, src_id, dst_id, src_id, dst_id),
        )
        conn.execute(
            "DELETE FROM edge_sources WHERE src_id = ? OR dst_id = ?",
            (src_id, src_id),
        )

        # 4. Re-point edges (excluding self-loops)
        incident_edges = conn.execute(
            "SELECT src_id, dst_id, kind, weight, reasoning FROM edges WHERE src_id = ? OR dst_id = ?",
            (src_id, src_id),
        ).fetchall()
        conn.execute(
            "DELETE FROM edges WHERE src_id = ? OR dst_id = ?",
            (src_id, src_id),
        )
        for e_src, e_dst, e_kind, e_weight, reasoning in incident_edges:
            new_src = dst_id if str(e_src) == src_id else str(e_src)
            new_dst = dst_id if str(e_dst) == src_id else str(e_dst)
            kind_str = str(e_kind)
            weight_val = float(e_weight)
            if new_src == new_dst:
                continue
            if kind_str == "passage":
                conn.execute(
                    """
                    INSERT INTO edges (src_id, dst_id, kind, weight, reasoning)
                    VALUES (?, ?, 'passage', ?, ?)
                    ON CONFLICT(src_id, dst_id, kind) DO UPDATE SET
                        weight = MAX(edges.weight, excluded.weight),
                        reasoning = MAX(edges.reasoning, excluded.reasoning)
                    """,
                    (new_src, new_dst, weight_val, reasoning),
                )
            elif kind_str == "fact":
                conn.execute(
                    """
                    INSERT INTO edges (src_id, dst_id, kind, weight, reasoning)
                    VALUES (?, ?, 'fact', ?, ?)
                    ON CONFLICT(src_id, dst_id, kind) DO UPDATE SET
                        weight = MAX(edges.weight, excluded.weight),
                        reasoning = MAX(edges.reasoning, excluded.reasoning)
                    """,
                    (new_src, new_dst, weight_val, reasoning),
                )
                conn.execute(
                    """
                    UPDATE edges
                    SET weight = (
                            SELECT 1.0 - EXP(SUM(LN(MAX(
                                1e-6, 1.0 - MIN(1.0 - 1e-6, es.atomic_confidence)
                            ))))
                            FROM edge_sources es
                            WHERE es.src_id = edges.src_id AND es.dst_id = edges.dst_id AND es.kind = edges.kind
                    )
                    WHERE src_id = ? AND dst_id = ? AND kind = 'fact'
                    """,
                    (new_src, new_dst),
                )
            elif kind_str in ("skos:related", "skos:broader", "owl:differentFrom"):
                upsert_semantic_edges(conn, [(new_src, new_dst, kind_str, weight_val, reasoning)])

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
        return EntityMerge(src_id, dst_id, fact_bindings)


def rekey_facts(conn: sqlite3.Connection, ids: list[str]) -> tuple[tuple[str, str | None], ...]:
    """Derive fact identities from canonical endpoints and return evidence redirects."""
    redirects = []
    for old_id in sorted(set(ids)):
        row = conn.execute(
            "SELECT facts.subj_id,facts.pred,facts.obj_kind,facts.obj_id,facts.obj_literal_lex,"
            "facts.obj_datatype,facts.obj_lang,facts.raw_triple_json,facts.definition,facts.confidence,"
            "COALESCE(es.name,''),COALESCE(es.definition,'') "
            "FROM facts LEFT JOIN entities es ON es.entity_id=facts.subj_id WHERE facts.fact_id=?",
            (old_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Cannot canonicalize missing fact {old_id!r}")
        (
            subject,
            predicate,
            kind,
            obj,
            lexical,
            datatype,
            language,
            raw_json,
            definition,
            confidence,
            subj_name,
            subj_def,
        ) = row
        new_id = (
            None
            if (kind == "entity" and subject == obj)
            or (
                kind == "literal"
                and not is_irreflexive_statement(subj_name, predicate, lexical or "", subj_def, "", "literal")
            )
            else fact_id(subject, predicate, kind, obj if kind == "entity" else lexical, datatype, language)
        )
        if new_id == old_id:
            continue
        redirects.append((old_id, new_id))
        if new_id is not None:
            existing = conn.execute("SELECT definition FROM facts WHERE fact_id=?", (new_id,)).fetchone()
            if existing is not None:
                definition = max((definition, existing[0]), key=lambda value: (len(value), value))
            upsert_facts(
                conn, [(new_id, subject, predicate, obj, raw_json, definition, kind, lexical, datatype, language)]
            )
            conn.execute(
                "UPDATE facts SET confidence=COALESCE(MAX(confidence,?),confidence,?) WHERE fact_id=?",
                (confidence, confidence, new_id),
            )
            conn.execute(
                "INSERT INTO fact_sources SELECT ?,chunk_id,llm_prob,vec_coherence,atomic_confidence "
                "FROM fact_sources WHERE fact_id=? ON CONFLICT(fact_id,chunk_id) DO UPDATE SET "
                + ",".join(
                    f"{column}=COALESCE(MAX(fact_sources.{column},excluded.{column}),"
                    f"fact_sources.{column},excluded.{column})"
                    for column in ("llm_prob", "vec_coherence", "atomic_confidence")
                ),
                (new_id, old_id),
            )
            old_vec_row = conn.execute("SELECT embedding FROM fact_vec WHERE fact_id=?", (old_id,)).fetchone()
            if old_vec_row is not None:
                new_vec_row = conn.execute("SELECT embedding FROM fact_vec WHERE fact_id=?", (new_id,)).fetchone()
                chosen_vec = (
                    min(bytes(old_vec_row[0]), bytes(new_vec_row[0]))
                    if new_vec_row is not None
                    else bytes(old_vec_row[0])
                )
                conn.execute("DELETE FROM fact_vec WHERE fact_id=?", (new_id,))
                conn.execute("INSERT INTO fact_vec (fact_id, embedding) VALUES (?, ?)", (new_id, chosen_vec))
        for table in ("fact_sources", "fact_vec", "fact_fts", "facts"):
            conn.execute(f"DELETE FROM {table} WHERE fact_id=?", (old_id,))
    return tuple(redirects)


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
            SET weight = (
                    SELECT 1.0 - EXP(SUM(LN(MAX(1e-6, 1.0 - MIN(1.0 - 1e-6, es.atomic_confidence)))))
                    FROM edge_sources es
                    WHERE es.src_id = edges.src_id AND es.dst_id = edges.dst_id AND es.kind = edges.kind
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


def get_entity_doc_frequencies(
    conn: sqlite3.Connection,
    entity_ids: Sequence[str],
) -> dict[str, int]:
    """Count number of passage edges incident to given entity_ids for HMM seed weighting."""
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
    openie_expr = "(SELECT COUNT(*) FROM chunk_inference)" if has_chunk_inference else "0"
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

    entity_ids = tuple(entity_ids_list)

    invalid_edge = conn.execute("SELECT weight FROM edges WHERE weight < 0 LIMIT 1").fetchone()
    if invalid_edge is not None:
        raise ValueError(f"Graph edge weight must be nonnegative: {invalid_edge[0]}")

    edge_cursor = conn.execute(
        """
        SELECT
            f.subj_id AS src_id,
            f.obj_id AS dst_id,
            'fact' AS kind,
            COALESCE(
                f.confidence,
                (SELECT 1.0 - EXP(SUM(LN(MAX(1e-6, 1.0 - MIN(1.0 - 1e-6, fs.atomic_confidence)))))
                 FROM fact_sources fs WHERE fs.fact_id = f.fact_id)
            ) AS weight
        FROM facts f
        WHERE f.obj_id IS NOT NULL
          AND f.obj_id != ''
          AND f.subj_id != f.obj_id
          AND COALESCE(f.obj_kind, 'entity') = 'entity'
          AND weight > 0.0
          AND NOT EXISTS (
              SELECT 1 FROM edges e
              WHERE e.src_id = f.subj_id AND e.dst_id = f.obj_id AND e.kind = 'fact'
          )
        UNION ALL
        SELECT
            e.src_id AS src_id,
            e.dst_id AS dst_id,
            e.kind AS kind,
            CASE WHEN e.kind = 'skos:related' THEN MAX(
                e.weight,
                COALESCE((SELECT rev.weight FROM edges rev
                          WHERE rev.src_id = e.dst_id AND rev.dst_id = e.src_id AND rev.kind = e.kind), e.weight)
            ) ELSE e.weight END AS weight
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
                        VALUES (?, ?, 'owl:differentFrom', ?, ?)
                        ON CONFLICT(src_id, dst_id, kind) DO UPDATE SET
                            weight = MAX(edges.weight, excluded.weight),
                            reasoning = CASE
                                WHEN excluded.reasoning != '' THEN excluded.reasoning
                                ELSE edges.reasoning
                            END
                        """,
                        (s, d, weight, reasoning),
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


def load_affected_neighborhood(
    ro_conn: sqlite3.Connection,
    stage_conn: sqlite3.Connection,
    *,
    seed_entity_ids: Sequence[str],
    seed_label_norms: Sequence[str],
    delta_entity_names: Mapping[str, str],
    candidate_dense_threshold: float = 0.8,
    candidate_lexical_threshold: float = 0.625,
    candidate_vectors: Sequence[CandidateVector] = (),
) -> set[str]:
    """Seed stage_conn with candidate partners and their affected semantic closure."""
    has_entities = (
        ro_conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='entities'").fetchone() is not None
    )
    if not has_entities:
        return set()
    has_entity_vec = (
        ro_conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='entity_vec'").fetchone() is not None
    )

    core_entity_ids: set[str] = set()

    # 1a. Existing entity IDs directly referenced by the delta batch or sense bindings
    for batch in batched(sorted({str(eid) for eid in seed_entity_ids if eid})):
        placeholders = ",".join("?" for _ in batch)
        for row in ro_conn.execute(
            f"SELECT entity_id FROM entities WHERE entity_id IN ({placeholders})",
            tuple(batch),
        ):
            core_entity_ids.add(str(row[0]))

    # 1b. Existing entities sharing a normalized label or token norm with the delta batch
    norm_candidates = {str(n) for n in seed_label_norms if n}
    for name in delta_entity_names.values():
        norm_candidates.add(name.strip().lower())
        norm_candidates.add(str(text_processing(name)))
    for batch in batched(sorted(norm_candidates)):
        placeholders = ",".join("?" for _ in batch)
        for row in ro_conn.execute(
            f"SELECT DISTINCT entity_id FROM entity_labels WHERE label_norm IN ({placeholders})",
            tuple(batch),
        ):
            core_entity_ids.add(str(row[0]))

    # 1c. Use the same pair-local policy before and after staging the delta.
    for a, b, _, _ in find_entity_candidates(
        ro_conn,
        sorted(set(seed_entity_ids) | set(delta_entity_names)),
        dense_threshold=candidate_dense_threshold,
        lexical_threshold=candidate_lexical_threshold,
        exclude_adjudicated=False,
        candidate_vectors=candidate_vectors,
    ):
        core_entity_ids.update((a, b))
    core_entity_ids.difference_update(
        set(delta_entity_names)
        - {
            str(row[0])
            for batch in batched(sorted(delta_entity_names))
            for row in ro_conn.execute(
                f"SELECT entity_id FROM entities WHERE entity_id IN ({','.join('?' for _ in batch)})", tuple(batch)
            )
        }
    )

    if not core_entity_ids:
        return set()

    # 2a. Expand transitive skos:broader ancestors and descendants
    ancestors = set(core_entity_ids)
    frontier = set(core_entity_ids)
    while frontier:
        next_frontier: set[str] = set()
        for batch in batched(sorted(frontier)):
            placeholders = ",".join("?" for _ in batch)
            for row in ro_conn.execute(
                f"SELECT dst_id FROM edges WHERE src_id IN ({placeholders}) AND kind = 'skos:broader'",
                tuple(batch),
            ):
                dst_id = str(row[0])
                if dst_id not in ancestors:
                    ancestors.add(dst_id)
                    next_frontier.add(dst_id)
        frontier = next_frontier

    descendants = set(core_entity_ids)
    frontier = set(core_entity_ids)
    while frontier:
        next_frontier = set()
        for batch in batched(sorted(frontier)):
            placeholders = ",".join("?" for _ in batch)
            for row in ro_conn.execute(
                f"SELECT src_id FROM edges WHERE dst_id IN ({placeholders}) AND kind = 'skos:broader'",
                tuple(batch),
            ):
                src_id = str(row[0])
                if src_id not in descendants:
                    descendants.add(src_id)
                    next_frontier.add(src_id)
        frontier = next_frontier

    core_entity_ids.update(ancestors)
    core_entity_ids.update(descendants)

    # 2b. Co-hyponym siblings sharing a skos:broader parent (ConceptShape/siblingNotBroader)
    parent_ids: set[str] = set()
    for batch in batched(sorted(core_entity_ids)):
        placeholders = ",".join("?" for _ in batch)
        for row in ro_conn.execute(
            f"SELECT dst_id FROM edges WHERE src_id IN ({placeholders}) AND kind = 'skos:broader'",
            tuple(batch),
        ):
            parent_ids.add(str(row[0]))
    for batch in batched(sorted(parent_ids)):
        placeholders = ",".join("?" for _ in batch)
        for row in ro_conn.execute(
            f"SELECT src_id FROM edges WHERE dst_id IN ({placeholders}) AND kind = 'skos:broader'",
            tuple(batch),
        ):
            core_entity_ids.add(str(row[0]))

    # 2c. 1-hop skos:related / owl:differentFrom neighbors of core_entity_ids
    sem_neighbors: set[str] = set()
    for batch in batched(sorted(core_entity_ids)):
        placeholders = ",".join("?" for _ in batch)
        b_tuple = tuple(batch)
        for row in ro_conn.execute(
            f"""
            SELECT dst_id FROM edges
            WHERE src_id IN ({placeholders}) AND kind IN ('skos:related', 'owl:differentFrom')
            UNION
            SELECT src_id FROM edges
            WHERE dst_id IN ({placeholders}) AND kind IN ('skos:related', 'owl:differentFrom')
            """,
            (*b_tuple, *b_tuple),
        ):
            sem_neighbors.add(str(row[0]))
    core_entity_ids.update(sem_neighbors)

    # Include any entity in ro_conn.entity_labels sharing a label_norm with core_entity_ids
    for batch in batched(sorted(core_entity_ids)):
        placeholders = ",".join("?" for _ in batch)
        for (eid,) in ro_conn.execute(
            f"""
            SELECT DISTINCT entity_id
            FROM entity_labels
            WHERE label_norm IN (
                SELECT label_norm FROM entity_labels WHERE entity_id IN ({placeholders})
            )
            """,
            tuple(batch),
        ):
            core_entity_ids.add(str(eid))

    affected_entity_ids = set(core_entity_ids)
    affected_fact_ids: set[str] = set()
    incident_edges: list[tuple[Any, ...]] = []
    incident_edge_keys: set[tuple[str, str, str]] = set()

    # 2d & 2e. Fixed-point expansion over incident facts/edges and passive-endpoint homographs
    expanded_core: set[str] = set()
    while True:
        pending_core = core_entity_ids - expanded_core
        if not pending_core:
            break
        expanded_core.update(pending_core)

        # 2d. Incident facts and their entity endpoints
        for batch in batched(sorted(pending_core)):
            placeholders = ",".join("?" for _ in batch)
            b_tuple = tuple(batch)
            for row in ro_conn.execute(
                f"""
                SELECT fact_id, subj_id, obj_id, obj_kind
                FROM facts
                WHERE subj_id IN ({placeholders}) OR obj_id IN ({placeholders})
                """,
                (*b_tuple, *b_tuple),
            ):
                affected_fact_ids.add(str(row[0]))
                affected_entity_ids.add(str(row[1]))
                if str(row[3] or "entity") == "entity" and row[2]:
                    affected_entity_ids.add(str(row[2]))

        # 2e. Incident edges and their entity endpoints
        for batch in batched(sorted(pending_core)):
            placeholders = ",".join("?" for _ in batch)
            b_tuple = tuple(batch)
            for row in ro_conn.execute(
                f"""
                SELECT src_id, dst_id, kind, weight, reasoning
                FROM edges
                WHERE src_id IN ({placeholders}) OR dst_id IN ({placeholders})
                """,
                (*b_tuple, *b_tuple),
            ):
                ekey = (str(row[0]), str(row[1]), str(row[2]))
                if ekey not in incident_edge_keys:
                    incident_edge_keys.add(ekey)
                    incident_edges.append(tuple(row))
                    if ekey[2] != "passage":
                        affected_entity_ids.add(ekey[0])
                        affected_entity_ids.add(ekey[1])

        # Promote any passive entities in affected_entity_ids that share a label_norm or definition
        eids_by_norm: dict[str, set[str]] = defaultdict(set)
        eids_by_def: dict[str, set[str]] = defaultdict(set)
        for batch in batched(sorted(affected_entity_ids)):
            placeholders = ",".join("?" for _ in batch)
            b_tuple = tuple(batch)
            for lnorm, eid in ro_conn.execute(
                f"SELECT label_norm, entity_id FROM entity_labels WHERE entity_id IN ({placeholders})",
                b_tuple,
            ):
                eids_by_norm[str(lnorm)].add(str(eid))
            for defn, eid in ro_conn.execute(
                f"SELECT definition, entity_id FROM entities WHERE entity_id IN ({placeholders})",
                b_tuple,
            ):
                def_norm = str(defn or "").strip().casefold()
                if def_norm:
                    eids_by_def[def_norm].add(str(eid))
        for eids in eids_by_norm.values():
            if len(eids) > 1:
                core_entity_ids.update(eids)
        for eids in eids_by_def.values():
            if len(eids) > 1:
                core_entity_ids.update(eids)

    # Copy passage edges and induced subgraph edges among passive affected_entity_ids
    for batch in batched(sorted(affected_entity_ids - core_entity_ids)):
        placeholders = ",".join("?" for _ in batch)
        b_tuple = tuple(batch)
        for row in ro_conn.execute(
            f"""
            SELECT src_id, dst_id, kind, weight, reasoning
            FROM edges
            WHERE src_id IN ({placeholders}) OR dst_id IN ({placeholders})
            """,
            (*b_tuple, *b_tuple),
        ):
            ekey = (str(row[0]), str(row[1]), str(row[2]))
            if ekey not in incident_edge_keys and (
                ekey[2] == "passage" or (ekey[0] in affected_entity_ids and ekey[1] in affected_entity_ids)
            ):
                incident_edge_keys.add(ekey)
                incident_edges.append(tuple(row))

    # Ensure vec0 dimensions match ro_conn before inserting vectors into stage_conn
    for vtable, pk_col in (("chunk_vec", "chunk_id"), ("entity_vec", "entity_id"), ("fact_vec", "fact_id")):
        vdim = vec0_table_dimension(ro_conn, vtable)
        if vdim is not None:
            ensure_vec0_table(stage_conn, vtable, pk_col, vdim)

    seed_chunk_ids: set[str] = set()

    with write_tx(stage_conn):
        # 3a. Copy entities, entity_vec, entity_labels, entity_mentions
        for batch in batched(sorted(affected_entity_ids)):
            placeholders = ",".join("?" for _ in batch)
            b_tuple = tuple(batch)
            ent_rows = ro_conn.execute(
                f"SELECT entity_id, name, definition, confidence FROM entities WHERE entity_id IN ({placeholders})",
                b_tuple,
            ).fetchall()
            if ent_rows:
                stage_conn.executemany(
                    "INSERT OR REPLACE INTO entities (entity_id, name, definition, confidence) VALUES (?, ?, ?, ?)",
                    ent_rows,
                )
            source_vectors = ro_conn.execute(
                f"SELECT sense_id,entity_id,name,definition,embedding FROM entity_candidate_vectors "
                f"WHERE entity_id IN ({placeholders})",
                b_tuple,
            ).fetchall()
            stage_conn.executemany("INSERT OR REPLACE INTO entity_candidate_vectors VALUES (?,?,?,?,?)", source_vectors)
            if has_entity_vec:
                evec_rows = ro_conn.execute(
                    f"SELECT entity_id, embedding FROM entity_vec WHERE entity_id IN ({placeholders})",
                    b_tuple,
                ).fetchall()
                for eid, emb in evec_rows:
                    stage_conn.execute("DELETE FROM entity_vec WHERE entity_id = ?", (eid,))
                    stage_conn.execute("INSERT INTO entity_vec (entity_id, embedding) VALUES (?, ?)", (eid, emb))
            lbl_rows = ro_conn.execute(
                f"SELECT label_norm, entity_id, label_kind, raw_label, chunk_id "
                f"FROM entity_labels WHERE entity_id IN ({placeholders})",
                b_tuple,
            ).fetchall()
            if lbl_rows:
                stage_conn.executemany(
                    "INSERT OR REPLACE INTO entity_labels (label_norm, entity_id, label_kind, raw_label, chunk_id) "
                    "VALUES (?, ?, ?, ?, ?)",
                    lbl_rows,
                )
                for row in lbl_rows:
                    cid = str(row[4])
                    if cid and not cid.startswith("__"):
                        seed_chunk_ids.add(cid)
            mention_rows = ro_conn.execute(
                f"SELECT mention_id, chunk_id, source_revision, sense_id, name, definition, "
                f"resolved_definition, entity_id, llm_prob "
                f"FROM entity_mentions WHERE entity_id IN ({placeholders})",
                b_tuple,
            ).fetchall()
            if mention_rows:
                stage_conn.executemany(
                    "INSERT OR REPLACE INTO entity_mentions "
                    "(mention_id, chunk_id, source_revision, sense_id, name, definition, "
                    "resolved_definition, entity_id, llm_prob) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    mention_rows,
                )
                for row in mention_rows:
                    cid = str(row[1])
                    if cid and not cid.startswith("__"):
                        seed_chunk_ids.add(cid)

        # 3b. Copy facts, fact_vec, fact_sources, fact_assertions, fact_fts
        has_fact_vec = (
            ro_conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='fact_vec'").fetchone() is not None
        )
        for batch in batched(sorted(affected_fact_ids)):
            placeholders = ",".join("?" for _ in batch)
            b_tuple = tuple(batch)
            fact_rows = ro_conn.execute(
                f"SELECT fact_id, subj_id, pred, obj_kind, obj_id, obj_literal_lex, obj_datatype, obj_lang, "
                f"raw_triple_json, definition, confidence FROM facts WHERE fact_id IN ({placeholders})",
                b_tuple,
            ).fetchall()
            if fact_rows:
                stage_conn.executemany(
                    "INSERT OR REPLACE INTO facts "
                    "(fact_id, subj_id, pred, obj_kind, obj_id, obj_literal_lex, obj_datatype, obj_lang, "
                    "raw_triple_json, definition, confidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    fact_rows,
                )
                fts_rows: list[tuple[str, str]] = []
                for f_row in fact_rows:
                    fid = str(f_row[0])
                    pred = str(f_row[2])
                    raw_json = str(f_row[8])
                    s_lbl, _, o_lbl = json.loads(raw_json)
                    fts_rows.append((fid, f"{s_lbl} {pred} {o_lbl}"))
                stage_conn.executemany(
                    "DELETE FROM fact_fts WHERE fact_id = ?",
                    ((fid,) for fid, _ in fts_rows),
                )
                stage_conn.executemany(
                    "INSERT INTO fact_fts (fact_id, text) VALUES (?, ?)",
                    fts_rows,
                )
            if has_fact_vec:
                fvec_rows = ro_conn.execute(
                    f"SELECT fact_id, embedding FROM fact_vec WHERE fact_id IN ({placeholders})",
                    b_tuple,
                ).fetchall()
                for fid, emb in fvec_rows:
                    stage_conn.execute("DELETE FROM fact_vec WHERE fact_id = ?", (fid,))
                    stage_conn.execute("INSERT INTO fact_vec (fact_id, embedding) VALUES (?, ?)", (fid, emb))
            fsrc_rows = ro_conn.execute(
                f"SELECT fact_id, chunk_id, llm_prob, vec_coherence, atomic_confidence "
                f"FROM fact_sources WHERE fact_id IN ({placeholders})",
                b_tuple,
            ).fetchall()
            if fsrc_rows:
                stage_conn.executemany(
                    "INSERT OR REPLACE INTO fact_sources "
                    "(fact_id, chunk_id, llm_prob, vec_coherence, atomic_confidence) VALUES (?, ?, ?, ?, ?)",
                    fsrc_rows,
                )
                for row in fsrc_rows:
                    cid = str(row[1])
                    if cid and not cid.startswith("__"):
                        seed_chunk_ids.add(cid)
            fassert_rows = ro_conn.execute(
                f"SELECT assertion_id, chunk_id, source_revision, subject_mention_id, object_mention_id, "
                f"raw_json, fact_id FROM fact_assertions WHERE fact_id IN ({placeholders})",
                b_tuple,
            ).fetchall()
            if fassert_rows:
                stage_conn.executemany(
                    "INSERT OR REPLACE INTO fact_assertions "
                    "(assertion_id, chunk_id, source_revision, subject_mention_id, object_mention_id, "
                    "raw_json, fact_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    fassert_rows,
                )
                for row in fassert_rows:
                    cid = str(row[1])
                    if cid and not cid.startswith("__"):
                        seed_chunk_ids.add(cid)

        # 3c. Copy incident edges and edge_sources
        if incident_edges:
            stage_conn.executemany(
                "INSERT OR REPLACE INTO edges (src_id, dst_id, kind, weight, reasoning) VALUES (?, ?, ?, ?, ?)",
                incident_edges,
            )
            for batch in batched(sorted({ekey[0] for ekey in incident_edge_keys})):
                placeholders = ",".join("?" for _ in batch)
                for row in ro_conn.execute(
                    f"SELECT src_id, dst_id, kind, chunk_id, llm_prob, vec_coherence, atomic_confidence "
                    f"FROM edge_sources WHERE src_id IN ({placeholders})",
                    tuple(batch),
                ):
                    ekey = (str(row[0]), str(row[1]), str(row[2]))
                    if ekey in incident_edge_keys:
                        stage_conn.execute(
                            "INSERT OR REPLACE INTO edge_sources "
                            "(src_id, dst_id, kind, chunk_id, llm_prob, vec_coherence, atomic_confidence) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?)",
                            tuple(row),
                        )
                        cid = str(row[3])
                        if cid and not cid.startswith("__"):
                            seed_chunk_ids.add(cid)

        # 3d. Copy referenced chunks and their source-coherence vectors.
        for batch in batched(sorted(seed_chunk_ids)):
            placeholders = ",".join("?" for _ in batch)
            chunk_rows = ro_conn.execute(
                f"SELECT chunk_id, rel_posix_path, start_line, end_line, title, ast_island_start, ast_island_end "
                f"FROM chunks WHERE chunk_id IN ({placeholders})",
                tuple(batch),
            ).fetchall()
            if chunk_rows:
                stage_conn.executemany(
                    "INSERT OR REPLACE INTO chunks "
                    "(chunk_id, rel_posix_path, start_line, end_line, title, ast_island_start, ast_island_end) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    chunk_rows,
                )
            for cid, embedding in ro_conn.execute(
                f"SELECT chunk_id,embedding FROM chunk_vec WHERE chunk_id IN ({placeholders})", tuple(batch)
            ):
                stage_conn.execute("DELETE FROM chunk_vec WHERE chunk_id=?", (cid,))
                stage_conn.execute("INSERT INTO chunk_vec (chunk_id,embedding) VALUES (?,?)", (cid, embedding))

    return affected_entity_ids
