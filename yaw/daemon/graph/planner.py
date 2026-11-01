"""Transaction-free CocoIndex graph planner and shared Stage 1-4 deterministic graph builders."""

import asyncio
import json
import re
import sqlite3
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cocoindex as coco
import numpy as np

from yaw.core import (
    ChunkPointer,
    OpenIERecord,
    SHACLViolation,
    StructuredTriple,
    YawRuntimeError,
    check_lexical_subsumption_inversion,
    compute_mdhash_id,
    is_acronym_pair,
    is_version_or_ordinal_sibling,
    text_processing,
)
from yaw.core.config import context_char_limit, scale_context_budget
from yaw.daemon.graph.inference import DSPyEmbeddingAdapter, EmbeddingRequest
from yaw.daemon.graph.inference.components import (
    EMBEDDING_ENDPOINT,
    OPENIE_ENDPOINT,
    MissingEntityInput,
    PredictorTrace,
    SHACLHealInput,
    SynonymJudgeInput,
    define_missing_entities,
    embed_text,
    heal_shacl_violations,
    judge_synonyms,
)
from yaw.daemon.graph.normalization import normalize_openie_record
from yaw.daemon.graph.storage import (
    PreparedGraphRows,
    apply_prepared_graph_rows,
    apply_shacl_repairs,
    deserialize_f32_vector,
    deserialize_openie_record,
    filter_valid_concept_entity_ids,
    find_entity_disambiguation_candidates,
    get_entities_by_label_norm,
    get_entity_metadata_batch,
    get_entity_vectors,
    inspect_focus_node,
    iter_synonym_evidence,
    lint_graph_shacl,
    merge_entities,
    open_yaw_db,
    resolve_chunk_text,
    upsert_semantic_edges,
)
from yaw.daemon.graph.storage.publication import GRAPH_DB


@dataclass(frozen=True, slots=True)
class GraphPlanSettings:
    corpus_root: Path
    program_identity: str
    embedding_identity: str
    synonym_identity: str = ""
    shacl_identity: str = ""
    same_sense_cosine_gate: float = 0.85
    synonymy_threshold: float = 0.8
    synonymy_top_k: int = 5
    datatype_property_weight: float = 0.5
    ast_context_hops: int = 1
    context_window_tokens: int = 32768
    max_output_tokens: int = 4096
    chunk_size: int = 1200
    chunk_overlap: int = 100
    min_chunk_size: int = 50


@dataclass(frozen=True, slots=True)
class GraphPlanResult:
    rows: PreparedGraphRows
    traces: tuple[PredictorTrace, ...] = ()
    telemetry: tuple[dict[str, Any], ...] = ()
    entity_def_audit_trail: tuple[dict[str, Any], ...] = ()
    synonym_audit_trail: tuple[dict[str, Any], ...] = ()
    shacl_audit_trail: tuple[dict[str, Any], ...] = ()
    rlm_stats: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SenseResolutionState:
    chunk_rel_path_by_id: dict[str, str]
    senses_by_label: dict[str, list[tuple[str, str, str, np.ndarray | None]]]
    existing_vecs_by_id: dict[str, np.ndarray]
    sense_docs_by_eid: dict[str, set[str]]
    sense_chunks_by_eid: dict[str, set[str]]
    multi_sense_in_chunk: set[tuple[str, str]]
    prebatch_texts: list[str]


@dataclass(slots=True)
class MaterializedGraphBatch:
    entity_nodes: list[tuple[str, str, str]]
    entity_labels_to_upsert: list[tuple[str, str, str, str, str]]
    fact_nodes: list[tuple[str, str, str, str | None, str, str, str, str, str, str | None, str, str]]
    fact_sources_batch: list[tuple[str, str]]
    edges_with_sources: list[tuple[str, str, str, float, str]]
    fact_linked_pairs: set[tuple[str, str]]
    precomputed_entity_vecs: dict[str, np.ndarray]
    homograph_splits: int


def load_active_chunk_inferences(
    conn: sqlite3.Connection,
    *,
    program_identity: str,
    embedding_identity: str,
) -> list[tuple[ChunkPointer, OpenIERecord, tuple[PredictorTrace, ...]]]:
    """Read active source_chunks joined with chunk_inference on a read-only snapshot."""
    tables = {
        str(r[0])
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('chunk_inference', 'source_chunks')"
        ).fetchall()
    }
    if tables != {"chunk_inference", "source_chunks"}:
        return []
    rows = conn.execute(
        """
        SELECT
            s.chunk_id,
            s.rel_posix_path,
            s.start_line,
            s.end_line,
            s.title,
            s.ast_island_start,
            s.ast_island_end,
            ci.extraction_json,
            ci.traces_json
        FROM source_chunks s
        JOIN chunk_inference ci
          ON ci.chunk_id = s.chunk_id
         AND ci.source_hash = s.source_hash
        WHERE ci.program_identity = ?
          AND ci.embedding_identity = ?
        ORDER BY s.chunk_id ASC
        """,
        (program_identity, embedding_identity),
    ).fetchall()
    items: list[tuple[ChunkPointer, OpenIERecord, tuple[PredictorTrace, ...]]] = []
    for r in rows:
        ptr = ChunkPointer(
            chunk_id=str(r[0]),
            rel_posix_path=str(r[1]),
            start_line=int(r[2]),
            end_line=int(r[3]),
            title=str(r[4] or ""),
            ast_island_start=int(r[5] or 0),
            ast_island_end=int(r[6] or 0),
        )
        rec = deserialize_openie_record(str(r[7]), chunk_id=str(r[0]))
        raw_traces = json.loads(str(r[8] or "[]"))
        traces = tuple(
            PredictorTrace(
                predictor=str(t.get("predictor", "")),
                inputs=dict(t.get("inputs") or {}),
                outputs=dict(t.get("outputs") or {}),
            )
            for t in raw_traces
            if isinstance(t, Mapping)
        )
        items.append((ptr, rec, traces))
    return items


def clone_graph_to_memory(conn: sqlite3.Connection) -> sqlite3.Connection:
    """Clone an existing yaw.db graph and vector tables into an isolated in-memory SQLite database."""
    stage_conn = open_yaw_db(":memory:")
    with stage_conn:
        for table, cols in (
            ("chunks", "chunk_id, rel_posix_path, start_line, end_line, title, ast_island_start, ast_island_end"),
            ("entities", "entity_id, name, definition"),
            ("entity_labels", "label_norm, entity_id, label_kind, raw_label, chunk_id"),
            (
                "facts",
                "fact_id, subj_id, pred, obj_id, raw_triple_json, definition, obj_kind, "
                "obj_literal_lex, obj_datatype, obj_lang",
            ),
            ("fact_sources", "fact_id, chunk_id"),
            ("edges", "src_id, dst_id, kind, weight"),
            ("edge_sources", "src_id, dst_id, kind, chunk_id"),
            ("chunk_vec", "chunk_id, embedding"),
            ("entity_vec", "entity_id, embedding"),
            ("fact_vec", "fact_id, embedding"),
        ):
            rows = conn.execute(f"SELECT {cols} FROM {table}").fetchall()
            if rows:
                placeholders = ",".join("?" for _ in cols.split(","))
                stage_conn.executemany(
                    f"INSERT INTO {table} ({cols}) VALUES ({placeholders})",
                    [tuple(r) for r in rows],
                )
    return stage_conn


def record_surface_form(
    surface_by_norm: dict[str, str],
    chunk_surface_by_norm: dict[tuple[str, str], str],
    norm_key: str,
    raw_val: str,
    cid_key: str = "",
) -> None:
    """Track preferred cased surface forms globally and per chunk."""
    clean_raw = raw_val.strip()
    if not norm_key or not clean_raw:
        return
    prev = surface_by_norm.get(norm_key)
    if prev is None or (prev == prev.lower() and clean_raw != clean_raw.lower()):
        surface_by_norm[norm_key] = clean_raw
    if cid_key:
        prev_c = chunk_surface_by_norm.get((cid_key, norm_key))
        if prev_c is None or (prev_c == prev_c.lower() and clean_raw != clean_raw.lower()):
            chunk_surface_by_norm[(cid_key, norm_key)] = clean_raw


def collect_missing_entity_norms(
    normalized_sts: Sequence[StructuredTriple],
    ner_entities: Sequence[tuple[str, str]],
    stage1_defs: Mapping[str, str],
    already_attempted: Iterable[str] = (),
) -> list[str]:
    """Identify normalized entity mentions lacking contextual definitions after Stage 1/2 OpenIE."""
    missing_norms: list[str] = []
    seen_missing: set[str] = set(already_attempted)
    for st in normalized_sts:
        if not st.subj_def and not stage1_defs.get(st.subj) and st.subj not in seen_missing:
            seen_missing.add(st.subj)
            missing_norms.append(st.subj)
        if st.obj_kind != "literal" and not st.obj_def and not stage1_defs.get(st.obj) and st.obj not in seen_missing:
            seen_missing.add(st.obj)
            missing_norms.append(st.obj)
    for e_norm, e_def in ner_entities:
        if not e_def and not stage1_defs.get(e_norm) and e_norm not in seen_missing:
            seen_missing.add(e_norm)
            missing_norms.append(e_norm)
    return missing_norms


def apply_stage1_definitions(
    cid: str,
    normalized_sts: list[StructuredTriple],
    ner_entities: list[tuple[str, str]],
    stage1_defs: dict[str, str],
    repaired_pairs: Sequence[tuple[str, str]],
    surface_by_norm: dict[str, str],
    chunk_surface_by_norm: dict[tuple[str, str], str],
) -> tuple[list[StructuredTriple], list[tuple[str, str]]]:
    """Merge repaired missing-entity definitions into Stage 1 definitions, triples, and NER entities."""
    for r_name, r_def in repaired_pairs:
        r_norm = str(text_processing(str(r_name)))
        record_surface_form(surface_by_norm, chunk_surface_by_norm, r_norm, str(r_name), cid)
        r_clean = str(r_def).strip()
        if r_norm and r_clean:
            stage1_defs[r_norm] = r_clean
    if stage1_defs:
        normalized_sts = [
            StructuredTriple(
                subj=st.subj,
                pred=st.pred,
                obj=st.obj,
                spo_def=st.spo_def,
                subj_def=st.subj_def or stage1_defs.get(st.subj, ""),
                obj_def=(
                    st.obj_def or stage1_defs.get(st.obj, "") or stage1_defs.get(str(text_processing(st.obj)), "")
                ),
                obj_kind=st.obj_kind,
                obj_datatype=st.obj_datatype,
                obj_lang=st.obj_lang,
            )
            for st in normalized_sts
        ]
        ner_entities = [(e_norm, e_def or stage1_defs.get(e_norm, "")) for e_norm, e_def in ner_entities]
    return normalized_sts, ner_entities


def synthesize_copular_is_a_triples(
    chunks: Sequence[tuple[ChunkPointer, str]],
    record_structured: Sequence[tuple[str, list[StructuredTriple], list[tuple[str, str]]]],
    all_norm_labels: set[str],
    surface_by_norm: dict[str, str],
    chunk_surface_by_norm: dict[tuple[str, str], str],
) -> None:
    """Bridge copular Stage 1 entity definitions to known graph concepts across multi-chunk corpora."""
    if len(chunks) <= 1:
        return
    known_concepts = {
        lbl for lbl in all_norm_labels if 3 <= len(lbl) <= 30 and not lbl.isdigit() and len(lbl.split()) <= 3
    }
    for ptr, _ in chunks:
        if ptr.title:
            t_norm = str(text_processing(ptr.title))
            record_surface_form(surface_by_norm, chunk_surface_by_norm, t_norm, ptr.title, ptr.chunk_id)
            if 3 <= len(t_norm) <= 30 and not t_norm.isdigit():
                known_concepts.add(t_norm)
    for _cid, normalized_sts, ner_entities in record_structured:
        existing_pairs = {(st.subj, st.obj) for st in normalized_sts} | {(st.obj, st.subj) for st in normalized_sts}
        for e_norm, e_def in ner_entities[:2]:
            if not e_def or len(e_norm) < 3:
                continue
            first_clause = re.split(r"\.(?:\s|$)", e_def, maxsplit=1)[0].lower()
            clause_words = set(str(text_processing(first_clause)).split()) | set(re.findall(r"[a-z0-9]+", first_clause))
            for c_norm in known_concepts:
                if c_norm == e_norm or c_norm in e_norm or e_norm in c_norm:
                    continue
                c_tokens = c_norm.split()
                if not all(tok in clause_words for tok in c_tokens):
                    continue
                if (e_norm, c_norm) in existing_pairs:
                    continue
                if re.search(rf"(?:^|(?<=\s)|(?<=\W)){re.escape(c_norm)}(?=$|\s|\W)", first_clause):
                    existing_pairs.add((e_norm, c_norm))
                    normalized_sts.append(
                        StructuredTriple(
                            subj=e_norm,
                            pred="is a",
                            obj=c_norm,
                            spo_def=e_def,
                            subj_def=e_def,
                            obj_def="",
                        )
                    )


def _doc_scope_key(rel_path: str, s_line: int, e_line: int, ast_s: int, ast_e: int) -> str:
    if not rel_path:
        return ""
    if rel_path.lower().endswith((".json", ".jsonl", ".ndjson")):
        span_s = ast_s if ast_s > 0 else s_line
        span_e = ast_e if ast_e > 0 else e_line
        return f"{rel_path}:{span_s}:{span_e}"
    return rel_path


def prepare_sense_resolution_state(
    chunks: Sequence[tuple[ChunkPointer, str]],
    record_structured: Sequence[tuple[str, list[StructuredTriple], list[tuple[str, str]]]],
    all_norm_labels: Iterable[str],
    surface_by_norm: Mapping[str, str],
    *,
    fmt_entity: Callable[[str, str], str],
    conn: sqlite3.Connection | None = None,
) -> SenseResolutionState:
    """Prepare existing sense state and pre-batch homograph disambiguation texts."""
    chunk_rel_path_by_id: dict[str, str] = {
        ptr.chunk_id: _doc_scope_key(
            ptr.rel_posix_path,
            ptr.start_line,
            ptr.end_line,
            ptr.ast_island_start,
            ptr.ast_island_end,
        )
        for ptr, _ in chunks
        if ptr.rel_posix_path
    }
    senses_by_label: dict[str, list[tuple[str, str, str, np.ndarray | None]]] = {}
    existing_vecs_by_id: dict[str, np.ndarray] = {}
    sense_docs_by_eid: dict[str, set[str]] = {}
    sense_chunks_by_eid: dict[str, set[str]] = {}

    if conn is not None:
        existing_by_label = get_entities_by_label_norm(conn, list(all_norm_labels))
        all_existing_eids = list({eid for eids in existing_by_label.values() for eid in eids})
        entity_info_by_id: dict[str, tuple[str, str]] = {}
        if all_existing_eids:
            entity_info_by_id = get_entity_metadata_batch(conn, all_existing_eids)
            raw_vecs = get_entity_vectors(conn, all_existing_eids)
            for eid_v, vec_v in raw_vecs.items():
                v_norm = float(np.linalg.norm(vec_v))
                existing_vecs_by_id[eid_v] = vec_v / v_norm if v_norm > 0 else vec_v
            ph = ",".join("?" for _ in all_existing_eids)
            for r_eid, r_cid, r_doc, r_s, r_e, r_ast_s, r_ast_e in conn.execute(
                f"""
                SELECT
                    el.entity_id,
                    el.chunk_id,
                    COALESCE(c.rel_posix_path, ''),
                    COALESCE(c.start_line, 0),
                    COALESCE(c.end_line, 0),
                    COALESCE(c.ast_island_start, 0),
                    COALESCE(c.ast_island_end, 0)
                FROM entity_labels el
                LEFT JOIN chunks c ON c.chunk_id = el.chunk_id
                WHERE el.entity_id IN ({ph}) AND el.chunk_id != ''
                """,
                all_existing_eids,
            ):
                s_eid = str(r_eid)
                s_cid = str(r_cid)
                s_doc = (
                    _doc_scope_key(str(r_doc), int(r_s), int(r_e), int(r_ast_s), int(r_ast_e))
                    if r_doc
                    else chunk_rel_path_by_id.get(s_cid, "")
                )
                sense_chunks_by_eid.setdefault(s_eid, set()).add(s_cid)
                if s_doc:
                    sense_docs_by_eid.setdefault(s_eid, set()).add(s_doc)
        for lbl, eids in existing_by_label.items():
            senses_by_label[lbl] = [
                (
                    eid,
                    entity_info_by_id.get(eid, (surface_by_norm.get(lbl, lbl), ""))[0],
                    entity_info_by_id.get(eid, (surface_by_norm.get(lbl, lbl), ""))[1],
                    existing_vecs_by_id.get(eid),
                )
                for eid in eids
            ]

    defs_by_label: dict[str, set[str]] = {}
    for lbl, senses in senses_by_label.items():
        for _, _, s_def, _ in senses:
            if s_def.strip():
                defs_by_label.setdefault(lbl, set()).add(s_def.strip().lower())

    cand_pairs_by_label: dict[str, list[tuple[str, str]]] = {}
    mentions_by_label: dict[str, list[tuple[str, str]]] = {}
    surfaces_by_label: dict[str, list[str]] = {}
    multi_sense_in_chunk: set[tuple[str, str]] = set()

    def _record_canon_surface(raw_mention: str, label_norm: str) -> None:
        raw_clean = str(raw_mention).strip()
        canon = (
            surface_by_norm.get(label_norm)
            or surface_by_norm.get(raw_clean)
            or (raw_clean if raw_clean != raw_clean.lower() else label_norm)
        )
        s_list = surfaces_by_label.setdefault(label_norm, [])
        if canon and canon not in s_list:
            s_list.append(canon)

    for cid_r, st_list, ner_entities in record_structured:
        chunk_defs_per_norm: dict[str, set[str]] = {}
        for st in st_list:
            s_norm = str(text_processing(st.subj)) or st.subj.strip().lower()
            s_def = st.subj_def.strip()
            if s_norm:
                _record_canon_surface(st.subj, s_norm)
                mentions_by_label.setdefault(s_norm, []).append((cid_r, s_def))
                if s_def:
                    defs_by_label.setdefault(s_norm, set()).add(s_def.lower())
                    cand_pairs_by_label.setdefault(s_norm, []).append((s_norm, s_def))
                    chunk_defs_per_norm.setdefault(s_norm, set()).add(s_def.lower())
            o_norm = str(text_processing(st.obj)) or st.obj.strip().lower()
            if st.obj_kind == "literal":
                if o_norm and o_norm != s_norm and not (st.obj_datatype.startswith("xsd:") and not st.obj_def.strip()):
                    _record_canon_surface(st.obj, o_norm)
                    o_cand_def = st.obj_def.strip() or st.spo_def.strip() or f"Literal value {st.obj}."
                    if o_cand_def:
                        defs_by_label.setdefault(o_norm, set()).add(o_cand_def.lower())
                        cand_pairs_by_label.setdefault(o_norm, []).append((o_norm, o_cand_def))
                        chunk_defs_per_norm.setdefault(o_norm, set()).add(o_cand_def.lower())
                    mentions_by_label.setdefault(o_norm, []).append((cid_r, o_cand_def))
            elif o_norm:
                _record_canon_surface(st.obj, o_norm)
                o_def = st.obj_def.strip()
                mentions_by_label.setdefault(o_norm, []).append((cid_r, o_def))
                if o_def:
                    defs_by_label.setdefault(o_norm, set()).add(o_def.lower())
                    cand_pairs_by_label.setdefault(o_norm, []).append((o_norm, o_def))
                    chunk_defs_per_norm.setdefault(o_norm, set()).add(o_def.lower())
        for e_raw_lbl, e_def in ner_entities:
            e_norm = str(text_processing(e_raw_lbl)) or e_raw_lbl.strip().lower()
            c_def = e_def.strip()
            if e_norm:
                _record_canon_surface(e_raw_lbl, e_norm)
                mentions_by_label.setdefault(e_norm, []).append((cid_r, c_def))
                if c_def:
                    defs_by_label.setdefault(e_norm, set()).add(c_def.lower())
                    cand_pairs_by_label.setdefault(e_norm, []).append((e_norm, c_def))
                    chunk_defs_per_norm.setdefault(e_norm, set()).add(c_def.lower())
        for m_norm, c_dset in chunk_defs_per_norm.items():
            if len(c_dset) >= 2:
                multi_sense_in_chunk.add((cid_r, m_norm))

    prebatch_texts: list[str] = []
    seen_prebatch: set[str] = set()
    for lbl, dset in defs_by_label.items():
        if len(dset) >= 2:
            existing_senses_lbl = senses_by_label.get(lbl, [])
            if len(existing_senses_lbl) <= 1:
                if existing_senses_lbl:
                    eid0, _, def0, _ = existing_senses_lbl[0]
                    possible_defs: list[str] = [def0] if def0 else []
                    seen_docs: set[str] = set(sense_docs_by_eid.get(eid0, set()))
                    seen_chunks: set[str] = set(sense_chunks_by_eid.get(eid0, set()))
                    seeded = True
                else:
                    possible_defs = []
                    seen_docs = set()
                    seen_chunks = set()
                    seeded = False

                for cid_m, c_def in mentions_by_label.get(lbl, ()):
                    doc_m = chunk_rel_path_by_id.get(cid_m, "")
                    if not seeded:
                        if c_def:
                            possible_defs = [c_def]
                        seen_chunks.add(cid_m)
                        if doc_m:
                            seen_docs.add(doc_m)
                        seeded = True
                        continue
                    if not c_def:
                        seen_chunks.add(cid_m)
                        if doc_m:
                            seen_docs.add(doc_m)
                        continue
                    if not possible_defs:
                        possible_defs = [c_def]
                        seen_chunks.add(cid_m)
                        if doc_m:
                            seen_docs.add(doc_m)
                        continue
                    if (
                        doc_m
                        and (cid_m, lbl) not in multi_sense_in_chunk
                        and doc_m in seen_docs
                        and cid_m not in seen_chunks
                    ):
                        next_defs: list[str] = []
                        for p_def in possible_defs:
                            if p_def.lower() == c_def.lower():
                                m_def = c_def if len(c_def) > len(p_def) else p_def
                            elif (
                                p_def
                                and c_def.lower() not in p_def.lower()
                                and p_def.lower() not in c_def.lower()
                                and len(p_def) + len(c_def) <= 200
                            ):
                                m_def = f"{p_def.rstrip('.')} — {c_def}"
                            else:
                                m_def = c_def if len(c_def) > len(p_def) else p_def
                            cand_pairs_by_label.setdefault(lbl, []).append((lbl, m_def))
                            if m_def not in next_defs:
                                next_defs.append(m_def)
                        for m_def in next_defs:
                            if m_def not in possible_defs:
                                possible_defs.append(m_def)
                    elif c_def not in possible_defs:
                        possible_defs.append(c_def)
                    seen_chunks.add(cid_m)
                    if doc_m:
                        seen_docs.add(doc_m)

            surface_names = list(surfaces_by_label.get(lbl) or [surface_by_norm.get(lbl, lbl)])
            default_surf = surface_by_norm.get(lbl, lbl)
            if default_surf and default_surf not in surface_names:
                surface_names.append(default_surf)
            for _, name_s, _, _ in existing_senses_lbl:
                if name_s and name_s not in surface_names:
                    surface_names.append(name_s)

            base_eid = compute_mdhash_id(lbl, prefix="entity-")
            for m_lbl, c_def in cand_pairs_by_label.get(lbl, ()):
                sense_cand_id = compute_mdhash_id(f"{m_lbl}::{c_def.lower()}", prefix="entity-")
                if sense_cand_id in existing_vecs_by_id and base_eid in existing_vecs_by_id:
                    continue
                for s_name in surface_names:
                    ct = fmt_entity(s_name, c_def)
                    if ct not in seen_prebatch:
                        seen_prebatch.add(ct)
                        prebatch_texts.append(ct)
            for _, name_s, prev_def_s, prev_vec_s in existing_senses_lbl:
                if prev_def_s and (prev_vec_s is None or len(surface_names) > 1):
                    for s_name in surface_names:
                        if prev_vec_s is not None and s_name == name_s:
                            continue
                        pt = fmt_entity(s_name, prev_def_s)
                        if pt not in seen_prebatch:
                            seen_prebatch.add(pt)
                            prebatch_texts.append(pt)

    return SenseResolutionState(
        chunk_rel_path_by_id=chunk_rel_path_by_id,
        senses_by_label=senses_by_label,
        existing_vecs_by_id=existing_vecs_by_id,
        sense_docs_by_eid=sense_docs_by_eid,
        sense_chunks_by_eid=sense_chunks_by_eid,
        multi_sense_in_chunk=multi_sense_in_chunk,
        prebatch_texts=prebatch_texts,
    )


def materialize_entities_and_facts(
    record_structured: Sequence[tuple[str, list[StructuredTriple], list[tuple[str, str]]]],
    surface_by_norm: Mapping[str, str],
    chunk_surface_by_norm: Mapping[tuple[str, str], str],
    sense_state: SenseResolutionState,
    prebatch_vec_by_text: Mapping[str, np.ndarray],
    *,
    fmt_entity: Callable[[str, str], str],
    same_sense_cosine_gate: float = 0.85,
    datatype_property_weight: float = 0.5,
    embed_one: Callable[[str], np.ndarray] | None = None,
) -> MaterializedGraphBatch:
    """Resolve entity mentions with homograph sense-splitting and materialize reified facts and edges."""
    senses_by_label = sense_state.senses_by_label
    existing_vecs_by_id = sense_state.existing_vecs_by_id
    chunk_rel_path_by_id = sense_state.chunk_rel_path_by_id
    sense_docs_by_eid = sense_state.sense_docs_by_eid
    sense_chunks_by_eid = sense_state.sense_chunks_by_eid
    multi_sense_in_chunk = sense_state.multi_sense_in_chunk

    entity_map: dict[str, tuple[str, str, str]] = {}
    entity_labels_to_upsert: list[tuple[str, str, str, str, str]] = []
    precomputed_entity_vecs: dict[str, np.ndarray] = {}
    homograph_splits = 0

    def _label_kind_for(entity_name: str, lbl_norm: str) -> str:
        return (
            "skos:prefLabel"
            if (
                lbl_norm == entity_name
                or lbl_norm.lower() == entity_name.lower()
                or lbl_norm == str(text_processing(entity_name))
            )
            else "skos:altLabel"
        )

    def _record_sense_provenance(eid: str, cid: str) -> None:
        sense_chunks_by_eid.setdefault(eid, set()).add(cid)
        doc_p = chunk_rel_path_by_id.get(cid, "")
        if doc_p:
            sense_docs_by_eid.setdefault(eid, set()).add(doc_p)

    def _unit_vec_for_text(text: str) -> np.ndarray:
        vec_opt = prebatch_vec_by_text.get(text)
        if vec_opt is not None:
            return vec_opt
        if embed_one is None:
            raise KeyError(f"Entity vector not found in prebatch for {text!r}")
        raw_v = np.asarray(embed_one(text), dtype=np.float32)
        norm_v = float(np.linalg.norm(raw_v))
        return raw_v / norm_v if norm_v > 0 else raw_v

    def _resolve_mention_entity_id(
        raw_mention: str,
        defn: str,
        chunk_id: str,
        *,
        exclude_eid: str | None = None,
    ) -> str:
        nonlocal homograph_splits
        raw_clean = str(raw_mention).strip()
        label_norm = str(text_processing(raw_clean)) or raw_clean.lower()
        clean_def = defn.strip()
        canon_surface = (
            surface_by_norm.get(label_norm)
            or surface_by_norm.get(raw_clean)
            or (raw_clean if raw_clean != raw_clean.lower() else label_norm)
        )
        mention_raw = (
            chunk_surface_by_norm.get((chunk_id, label_norm))
            or chunk_surface_by_norm.get((chunk_id, raw_clean))
            or canon_surface
        )
        existing_senses = senses_by_label.get(label_norm, [])
        if not existing_senses:
            eid = compute_mdhash_id(label_norm, prefix="entity-")
            senses_by_label[label_norm] = [(eid, canon_surface, clean_def, existing_vecs_by_id.get(eid))]
            entity_map[eid] = (eid, canon_surface, clean_def)
            entity_labels_to_upsert.append((label_norm, eid, "skos:prefLabel", mention_raw, chunk_id))
            _record_sense_provenance(eid, chunk_id)
            return eid

        def _best_surface(curr_name: str) -> str:
            if (
                curr_name == curr_name.lower()
                and canon_surface != canon_surface.lower()
                and (str(text_processing(curr_name)) or curr_name.lower()) == label_norm
            ):
                return canon_surface
            return curr_name

        if not clean_def:
            for idx_s, (eid, name, prev_def, prev_vec) in enumerate(existing_senses):
                if exclude_eid is not None and eid == exclude_eid:
                    continue
                best_name = _best_surface(name)
                if best_name != name:
                    existing_senses[idx_s] = (eid, best_name, prev_def, prev_vec)
                    entity_map[eid] = (eid, best_name, prev_def)
                elif eid not in entity_map:
                    entity_map[eid] = (eid, best_name, prev_def)
                entity_labels_to_upsert.append(
                    (label_norm, eid, _label_kind_for(best_name, label_norm), mention_raw, chunk_id)
                )
                _record_sense_provenance(eid, chunk_id)
                return eid
            eid, name, prev_def, prev_vec = existing_senses[0]
            best_name = _best_surface(name)
            if best_name != name:
                existing_senses[0] = (eid, best_name, prev_def, prev_vec)
                entity_map[eid] = (eid, best_name, prev_def)
            elif eid not in entity_map:
                entity_map[eid] = (eid, best_name, prev_def)
            entity_labels_to_upsert.append(
                (label_norm, eid, _label_kind_for(best_name, label_norm), mention_raw, chunk_id)
            )
            _record_sense_provenance(eid, chunk_id)
            return eid

        for idx_s, (eid, name, prev_def, prev_vec) in enumerate(existing_senses):
            if exclude_eid is not None and eid == exclude_eid:
                continue
            if not prev_def or prev_def.lower() == clean_def.lower():
                best_name = _best_surface(name)
                merged_def = clean_def if len(clean_def) > len(prev_def) else (prev_def or clean_def)
                existing_senses[idx_s] = (
                    eid,
                    best_name,
                    merged_def,
                    prev_vec if (merged_def == prev_def and best_name == name) else None,
                )
                entity_map[eid] = (eid, best_name, merged_def)
                entity_labels_to_upsert.append(
                    (label_norm, eid, _label_kind_for(best_name, label_norm), mention_raw, chunk_id)
                )
                _record_sense_provenance(eid, chunk_id)
                return eid

        curr_doc = chunk_rel_path_by_id.get(chunk_id, "")
        if (
            curr_doc
            and exclude_eid is None
            and len(existing_senses) == 1
            and (chunk_id, label_norm) not in multi_sense_in_chunk
        ):
            eid0, name0, prev_def0, prev_vec0 = existing_senses[0]
            if curr_doc in sense_docs_by_eid.get(eid0, set()) and chunk_id not in sense_chunks_by_eid.get(eid0, set()):
                best_name = _best_surface(name0)
                if (
                    prev_def0
                    and clean_def
                    and clean_def.lower() not in prev_def0.lower()
                    and prev_def0.lower() not in clean_def.lower()
                    and len(prev_def0) + len(clean_def) <= 200
                ):
                    merged_def = f"{prev_def0.rstrip('.')} — {clean_def}"
                else:
                    merged_def = clean_def if len(clean_def) > len(prev_def0) else prev_def0
                existing_senses[0] = (
                    eid0,
                    best_name,
                    merged_def,
                    prev_vec0 if (merged_def == prev_def0 and best_name == name0) else None,
                )
                entity_map[eid0] = (eid0, best_name, merged_def)
                entity_labels_to_upsert.append(
                    (label_norm, eid0, _label_kind_for(best_name, label_norm), mention_raw, chunk_id)
                )
                _record_sense_provenance(eid0, chunk_id)
                return eid0

        sense_id = compute_mdhash_id(f"{label_norm}::{clean_def.lower()}", prefix="entity-")
        cached_sense_vec = existing_vecs_by_id.get(sense_id)
        if cached_sense_vec is None:
            cached_sense_vec = precomputed_entity_vecs.get(sense_id)
        cand_vec = (
            cached_sense_vec
            if cached_sense_vec is not None
            else _unit_vec_for_text(fmt_entity(canon_surface, clean_def))
        )

        for idx_s, (eid, name, prev_def, prev_vec) in enumerate(existing_senses):
            if exclude_eid is not None and eid == exclude_eid:
                continue
            if prev_vec is None:
                prev_vec = _unit_vec_for_text(fmt_entity(name, prev_def))
                existing_senses[idx_s] = (eid, name, prev_def, prev_vec)
                precomputed_entity_vecs[eid] = prev_vec
            cos_sim = float(np.dot(cand_vec, prev_vec))
            if cos_sim >= same_sense_cosine_gate:
                best_name = _best_surface(name)
                merged_def = clean_def if len(clean_def) > len(prev_def) else prev_def
                if merged_def != prev_def or best_name != name:
                    existing_senses[idx_s] = (eid, best_name, merged_def, cand_vec)
                    precomputed_entity_vecs[eid] = cand_vec
                entity_map[eid] = (eid, best_name, merged_def)
                entity_labels_to_upsert.append(
                    (label_norm, eid, _label_kind_for(best_name, label_norm), mention_raw, chunk_id)
                )
                _record_sense_provenance(eid, chunk_id)
                return eid

        precomputed_entity_vecs[sense_id] = cand_vec
        existing_senses.append((sense_id, canon_surface, clean_def, cand_vec))
        entity_map[sense_id] = (sense_id, canon_surface, clean_def)
        entity_labels_to_upsert.append((label_norm, sense_id, "skos:prefLabel", mention_raw, chunk_id))
        _record_sense_provenance(sense_id, chunk_id)
        homograph_splits += 1
        return sense_id

    chunk_to_entities: dict[str, set[str]] = {}
    fact_nodes_map: dict[
        str,
        tuple[str, str, str, str | None, str, str, str, str, str, str | None, str, str],
    ] = {}
    fact_sources_batch: list[tuple[str, str]] = []
    edges_with_sources: list[tuple[str, str, str, float, str]] = []
    fact_linked_pairs: set[tuple[str, str]] = set()

    for cid, st_list, ner_entities in record_structured:
        chunk_to_entities.setdefault(cid, set())
        resolved_in_chunk: set[tuple[str, str]] = set()
        for st in st_list:
            s_norm_key = str(text_processing(st.subj)) or st.subj.strip().lower()
            s_id = _resolve_mention_entity_id(st.subj, st.subj_def, cid)
            resolved_in_chunk.add((s_norm_key, st.subj_def.strip().lower()))
            chunk_to_entities[cid].add(s_id)

            if st.obj_kind == "literal":
                o_id: str | None = None
                o_norm_key = str(text_processing(st.obj)) or st.obj.strip().lower()
                is_self_literal = o_norm_key == s_norm_key
                if not is_self_literal and not (st.obj_datatype.startswith("xsd:") and not st.obj_def.strip()):
                    has_existing_sense = bool(senses_by_label.get(o_norm_key))
                    eff_obj_def = (
                        st.obj_def.strip()
                        if (st.obj_def.strip() or has_existing_sense)
                        else (st.spo_def.strip() or f"Literal value {st.obj}.")
                    )
                    o_id = _resolve_mention_entity_id(st.obj, eff_obj_def, cid)
                    resolved_in_chunk.add((o_norm_key, st.obj_def.strip().lower()))
                    chunk_to_entities[cid].add(o_id)
                eff_obj_dt = st.obj_datatype or "rdfs:Literal"
                fact_oid = o_id if (o_id is not None and o_id != s_id) else None
                fid = compute_mdhash_id(
                    str((s_id, st.pred, f"literal:{st.obj}:{eff_obj_dt}:{st.obj_lang}")),
                    prefix="fact-",
                )
                raw_json = json.dumps([st.subj, st.pred, st.obj])
                prev_fact = fact_nodes_map.get(fid)
                if prev_fact is None or (not prev_fact[5] and st.spo_def):
                    fact_nodes_map[fid] = (
                        fid,
                        s_id,
                        st.pred,
                        fact_oid,
                        raw_json,
                        st.spo_def,
                        st.subj_def,
                        st.obj_def,
                        "literal",
                        st.obj,
                        eff_obj_dt,
                        st.obj_lang,
                    )
                fact_sources_batch.append((fid, cid))
                if fact_oid is not None:
                    fact_linked_pairs.add((min(s_id, fact_oid), max(s_id, fact_oid)))
                    edges_with_sources.append((s_id, fact_oid, "fact", datatype_property_weight, cid))
            else:
                o_norm_key = str(text_processing(st.obj)) or st.obj.strip().lower()
                distinct_homograph_pole = (
                    o_norm_key == s_norm_key
                    and bool(st.subj_def.strip())
                    and bool(st.obj_def.strip())
                    and st.subj_def.strip().lower() != st.obj_def.strip().lower()
                )
                o_id = _resolve_mention_entity_id(
                    st.obj,
                    st.obj_def,
                    cid,
                    exclude_eid=s_id if distinct_homograph_pole else None,
                )
                resolved_in_chunk.add((o_norm_key, st.obj_def.strip().lower()))
                chunk_to_entities[cid].add(o_id)

                base_s_id = compute_mdhash_id(s_norm_key, prefix="entity-")
                base_o_id = compute_mdhash_id(o_norm_key, prefix="entity-")
                if s_norm_key == o_norm_key or s_id != base_s_id or o_id != base_o_id:
                    fid = compute_mdhash_id(str((s_id, st.pred, o_id)), prefix="fact-")
                else:
                    fid = compute_mdhash_id(str((st.subj, st.pred, st.obj)), prefix="fact-")
                raw_json = json.dumps([st.subj, st.pred, st.obj])
                prev_fact = fact_nodes_map.get(fid)
                if prev_fact is None or (not prev_fact[5] and st.spo_def):
                    fact_nodes_map[fid] = (
                        fid,
                        s_id,
                        st.pred,
                        o_id,
                        raw_json,
                        st.spo_def,
                        st.subj_def,
                        st.obj_def,
                        "entity",
                        None,
                        "",
                        "",
                    )
                fact_sources_batch.append((fid, cid))
                if s_id != o_id:
                    fact_linked_pairs.add((min(s_id, o_id), max(s_id, o_id)))
                    edges_with_sources.append((s_id, o_id, "fact", 1.0, cid))

        for e_raw_lbl, e_def in ner_entities:
            e_norm = str(text_processing(e_raw_lbl)) or e_raw_lbl.strip().lower()
            e_key = (e_norm, e_def.strip().lower())
            if e_key in resolved_in_chunk:
                continue
            if not e_def.strip() and any(k[0] == e_norm for k in resolved_in_chunk):
                continue
            resolved_in_chunk.add(e_key)
            ner_eid = _resolve_mention_entity_id(e_raw_lbl, e_def.strip(), cid)
            chunk_to_entities[cid].add(ner_eid)

    for cid, eids in chunk_to_entities.items():
        for eid in eids:
            edges_with_sources.append((cid, eid, "passage", 1.0, cid))
            edges_with_sources.append((eid, cid, "passage", 1.0, cid))

    return MaterializedGraphBatch(
        entity_nodes=list(entity_map.values()),
        entity_labels_to_upsert=entity_labels_to_upsert,
        fact_nodes=list(fact_nodes_map.values()),
        fact_sources_batch=fact_sources_batch,
        edges_with_sources=edges_with_sources,
        fact_linked_pairs=fact_linked_pairs,
        precomputed_entity_vecs=precomputed_entity_vecs,
        homograph_splits=homograph_splits,
    )


def prepare_synonym_candidates(
    stage_conn: sqlite3.Connection,
    entity_nodes: Sequence[tuple[str, str, str]],
    *,
    synonymy_threshold: float,
    synonymy_top_k: int,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, tuple[str, str]], int]:
    """Filter valid SKOS concept entities and rank disambiguation candidate pairs."""
    valid_entity_ids = filter_valid_concept_entity_ids(stage_conn, entity_nodes)
    if not valid_entity_ids:
        return [], {}, 0
    if progress_callback is not None:
        progress_callback(f"indexing: disambiguation candidate search ({len(valid_entity_ids)} changed entities)")
    raw_candidates = find_entity_disambiguation_candidates(
        stage_conn,
        entity_ids=valid_entity_ids,
        dense_threshold=synonymy_threshold,
        dense_top_k=synonymy_top_k,
    )
    cand_eids = list({eid for ea, eb, _, _ in raw_candidates for eid in (ea, eb)})
    cand_meta = get_entity_metadata_batch(stage_conn, cand_eids)

    candidates: list[dict[str, Any]] = []
    for ea, eb, d_sim, l_sim in raw_candidates:
        name_a, def_a = cand_meta.get(ea, (ea, ""))
        name_b, def_b = cand_meta.get(eb, (eb, ""))
        sim = max(float(d_sim), float(l_sim))
        candidates.append(
            {
                "pair_id": f"{ea}|{eb}",
                "entity_a": ea,
                "name_a": name_a,
                "def_a": def_a,
                "entity_b": eb,
                "name_b": name_b,
                "def_b": def_b,
                "dense_sim": float(d_sim),
                "lexical_sim": float(l_sim),
                "similarity": sim,
            }
        )
    candidates.sort(
        key=lambda c: (
            text_processing(str(c["name_a"])) == text_processing(str(c["name_b"])),
            float(c["lexical_sim"]) >= 0.72,
            0.5 * float(c["lexical_sim"]) + 0.5 * float(c["dense_sim"]),
            float(c["similarity"]),
        ),
        reverse=True,
    )
    return candidates, cand_meta, len(raw_candidates)


_SNAKE_IDENT_RE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$", re.IGNORECASE)
_ID_TOKEN_RE = re.compile(r"(?:^|[_\s-])(?:id|ids)(?:$|[_\s-])", re.IGNORECASE)


def _is_incompatible_identifier_merge(name_a: str, name_b: str) -> bool:
    """Return True when merging would collapse distinct snake_case fields or an *_id field with non-ID concepts."""
    a = name_a.strip().lower()
    b = name_b.strip().lower()
    if not a or not b or a == b:
        return False
    a_snake = bool(_SNAKE_IDENT_RE.match(a))
    b_snake = bool(_SNAKE_IDENT_RE.match(b))
    if a_snake and b_snake:
        return True
    a_is_id_field = a.endswith("_id") or a.endswith(" id")
    b_is_id_field = b.endswith("_id") or b.endswith(" id")
    if a_is_id_field != b_is_id_field:
        other = b if a_is_id_field else a
        if not _ID_TOKEN_RE.search(other) and "identifier" not in other:
            return True
    if a_snake or b_snake:
        snake_val = a if a_snake else b
        other_str = b if a_snake else a
        if any("_" in tok and tok.strip(".,:;()[]") != snake_val for tok in other_str.split()):
            return True
        snake_toks = [t for t in snake_val.split("_") if t]
        other_toks = [t for t in re.split(r"[\s_-]+", other_str) if t]
        if any(t not in other_toks for t in snake_toks):
            return True
    return False


def apply_synonym_adjudication(
    stage_conn: sqlite3.Connection,
    candidates: Sequence[dict[str, Any]],
    cand_meta: Mapping[str, tuple[str, str]],
    decisions: Sequence[Mapping[str, Any]],
    fact_linked_pairs: set[tuple[str, str]],
    *,
    synonymy_threshold: float,
    rlm_stats: dict[str, Any],
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str, float]]]:
    """Apply Stage 3 synonym decisions and unadjudicated candidate fallbacks on stage_conn."""
    merged_into: dict[str, str] = {}

    def _resolve_active_eid(eid: str) -> str:
        curr = eid
        visited: set[str] = set()
        while curr in merged_into and curr not in visited:
            visited.add(curr)
            curr = merged_into[curr]
        return curr

    staged_merges: list[tuple[str, str, str]] = []
    semantic_batch: list[tuple[str, str, str, float]] = []
    cand_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for c in candidates:
        ea, eb = str(c["entity_a"]), str(c["entity_b"])
        cand_by_pair[(min(ea, eb), max(ea, eb))] = c

    adjudicated_pairs: set[tuple[str, str]] = set()
    with stage_conn:
        for dec in decisions:
            pid = str(dec.get("pair_id", "")).strip()
            if "|" in pid:
                ea, eb = pid.split("|", 1)
                raw_src, raw_tgt = ea.strip(), eb.strip()
            else:
                raw_src = str(dec.get("source_id", "")).strip()
                raw_tgt = str(dec.get("target_id", "")).strip()
            rel = str(dec.get("relation") or dec.get("decision") or "").strip()
            canon = str(dec.get("canonical", "")).strip()
            if not raw_src or not raw_tgt:
                continue
            pkey = (min(raw_src, raw_tgt), max(raw_src, raw_tgt))
            if pkey not in cand_by_pair:
                continue
            adjudicated_pairs.add(pkey)
            src = _resolve_active_eid(raw_src)
            tgt = _resolve_active_eid(raw_tgt)
            if src == tgt:
                continue
            cand_info = cand_by_pair.get(pkey, {})
            sim = float(cand_info.get("similarity", 0.5) or 0.5)
            name_src, _ = cand_meta.get(src, cand_meta.get(raw_src, (src, "")))
            name_tgt, _ = cand_meta.get(tgt, cand_meta.get(raw_tgt, (tgt, "")))
            same_name = text_processing(name_src) == text_processing(name_tgt)
            active_pkey = (min(src, tgt), max(src, tgt))
            is_fact_pole_pair = active_pkey in fact_linked_pairs
            if (
                rel not in ("owl:differentFrom", "different", "different_from")
                and is_acronym_pair(name_src, name_tgt)
                and not (same_name and is_fact_pole_pair)
            ):
                merge_src, merge_dst = (src, tgt) if len(name_src) < len(name_tgt) else (tgt, src)
                merge_entities(stage_conn, merge_src, merge_dst, label_kind="skos:altLabel")
                staged_merges.append((merge_src, merge_dst, "skos:altLabel"))
                merged_into[merge_src] = merge_dst
                rlm_stats["merged_entities"] += 1
                continue
            if rel in ("skos:altLabel", "skos:hiddenLabel", "exact_match") and not (same_name and is_fact_pole_pair):
                if is_version_or_ordinal_sibling(name_src, name_tgt) or _is_incompatible_identifier_merge(
                    name_src, name_tgt
                ):
                    semantic_batch.append((src, tgt, "skos:related", max(0.5, sim)))
                    rlm_stats["related_edges"] += 1
                    continue
                if canon and (
                    canon.upper() == "A"
                    or canon in (src, raw_src)
                    or (
                        text_processing(canon) == text_processing(name_src)
                        and text_processing(canon) != text_processing(name_tgt)
                    )
                ):
                    merge_src, merge_dst = tgt, src
                else:
                    merge_src, merge_dst = src, tgt
                lkind = (
                    "skos:hiddenLabel"
                    if rel == "skos:hiddenLabel" or str(dec.get("label_type", "")) == "hiddenLabel"
                    else "skos:altLabel"
                )
                merge_entities(stage_conn, merge_src, merge_dst, label_kind=lkind)
                staged_merges.append((merge_src, merge_dst, lkind))
                merged_into[merge_src] = merge_dst
                rlm_stats["merged_entities"] += 1
            elif rel in ("owl:differentFrom", "different", "different_from"):
                semantic_batch.append((src, tgt, "owl:differentFrom", 0.0))
                rlm_stats["different_from_edges"] += 1
            elif rel == "skos:broader":
                if is_version_or_ordinal_sibling(name_src, name_tgt):
                    semantic_batch.append((src, tgt, "skos:related", max(0.5, sim)))
                    rlm_stats["related_edges"] += 1
                    continue
                if canon and (
                    canon.upper() == "A"
                    or canon in (src, raw_src)
                    or (
                        text_processing(canon) == text_processing(name_src)
                        and text_processing(canon) != text_processing(name_tgt)
                    )
                ):
                    child_id, parent_id = tgt, src
                    child_name, parent_name = name_tgt, name_src
                elif canon and (
                    canon.upper() == "B"
                    or canon in (tgt, raw_tgt)
                    or (
                        text_processing(canon) == text_processing(name_tgt)
                        and text_processing(canon) != text_processing(name_src)
                    )
                ):
                    child_id, parent_id = src, tgt
                    child_name, parent_name = name_src, name_tgt
                else:
                    semantic_batch.append((src, tgt, "skos:related", max(0.5, sim)))
                    rlm_stats["related_edges"] += 1
                    continue
                if check_lexical_subsumption_inversion(child_name, parent_name):
                    child_id, parent_id = parent_id, child_id
                semantic_batch.append((child_id, parent_id, "skos:broader", max(0.5, sim)))
                rlm_stats["broader_edges"] = int(rlm_stats.get("broader_edges", 0)) + 1
            elif rel in ("skos:related", "related"):
                if same_name:
                    semantic_batch.append((src, tgt, "owl:differentFrom", 0.0))
                    rlm_stats["different_from_edges"] += 1
                    semantic_batch.append((src, tgt, "skos:related", max(0.5, sim)))
                    rlm_stats["related_edges"] += 1
                else:
                    semantic_batch.append((src, tgt, "skos:related", max(0.5, sim)))
                    rlm_stats["related_edges"] += 1
            elif same_name and is_fact_pole_pair:
                semantic_batch.append((src, tgt, "owl:differentFrom", 0.0))
                rlm_stats["different_from_edges"] += 1

        for c in candidates:
            ea, eb = str(c["entity_a"]), str(c["entity_b"])
            pkey = (min(ea, eb), max(ea, eb))
            if pkey in adjudicated_pairs:
                continue
            src = _resolve_active_eid(ea)
            tgt = _resolve_active_eid(eb)
            if src == tgt:
                continue
            name_src, def_src = cand_meta.get(src, (str(c["name_a"]), str(c.get("def_a", ""))))
            name_tgt, def_tgt = cand_meta.get(tgt, (str(c["name_b"]), str(c.get("def_b", ""))))
            same_name = text_processing(name_src) == text_processing(name_tgt)
            norm_def_src = str(text_processing(def_src)) if def_src else ""
            norm_def_tgt = str(text_processing(def_tgt)) if def_tgt else ""
            same_or_empty_def = not norm_def_src or not norm_def_tgt or norm_def_src == norm_def_tgt
            active_pkey = (min(src, tgt), max(src, tgt))
            if is_acronym_pair(name_src, name_tgt) and not (same_name and active_pkey in fact_linked_pairs):
                merge_src, merge_dst = (src, tgt) if len(name_src) < len(name_tgt) else (tgt, src)
                merge_entities(stage_conn, merge_src, merge_dst, label_kind="skos:altLabel")
                staged_merges.append((merge_src, merge_dst, "skos:altLabel"))
                merged_into[merge_src] = merge_dst
                rlm_stats["merged_entities"] += 1
            elif same_name:
                if (
                    float(c.get("dense_sim", 0.0)) >= 0.45 or same_or_empty_def
                ) and active_pkey not in fact_linked_pairs:
                    merge_entities(stage_conn, src, tgt, label_kind="skos:altLabel")
                    staged_merges.append((src, tgt, "skos:altLabel"))
                    merged_into[src] = tgt
                    rlm_stats["merged_entities"] += 1
                elif active_pkey in fact_linked_pairs:
                    semantic_batch.append((src, tgt, "owl:differentFrom", 0.0))
                    rlm_stats["different_from_edges"] += 1
            elif float(c.get("dense_sim", 0.0)) >= synonymy_threshold:
                sim = float(c.get("similarity", 0.5) or 0.5)
                semantic_batch.append((src, tgt, "skos:related", max(0.5, sim)))
                rlm_stats["related_edges"] += 1

        resolved_semantic_batch: list[tuple[str, str, str, float]] = []
        if semantic_batch:
            resolved_semantic_batch = [
                (_resolve_active_eid(s), _resolve_active_eid(d), k, w)
                for s, d, k, w in semantic_batch
                if _resolve_active_eid(s) != _resolve_active_eid(d)
            ]
            if resolved_semantic_batch:
                upsert_semantic_edges(stage_conn, resolved_semantic_batch)

    return staged_merges, resolved_semantic_batch


def collect_shacl_violation_dossiers(
    stage_conn: sqlite3.Connection,
    corpus_root: Path,
    *,
    ast_context_hops: int,
    context_window_tokens: int,
    attempted_keys: set[tuple[str, str]] | None = None,
    batch_node_ids: set[str] | None = None,
) -> tuple[list[SHACLViolation], list[SHACLViolation], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Lint stage_conn for SHACL violations, filter by batch scope if incremental, and build focus dossiers."""
    all_violations = lint_graph_shacl(stage_conn)
    if attempted_keys is not None and batch_node_ids is not None:
        violations = [
            v
            for v in all_violations
            if (v.focus_node, v.shape) not in attempted_keys
            or v.focus_node in batch_node_ids
            or ("|" in v.focus_node and any(p.strip() in batch_node_ids for p in v.focus_node.split("|")))
            or ("->" in v.focus_node and any(p.strip() in batch_node_ids for p in v.focus_node.split("->")))
        ]
    else:
        violations = list(all_violations)
    focus_dossiers: dict[str, dict[str, Any]] = {}
    focus_contexts: list[dict[str, Any]] = []
    for v in violations[: scale_context_budget(16, context_window_tokens)]:
        if v.focus_node not in focus_dossiers:
            focus_dossiers[v.focus_node] = inspect_focus_node(
                stage_conn,
                v.focus_node,
                corpus_root=corpus_root,
                ast_context_hops=ast_context_hops,
            )
        focus_contexts.append(focus_dossiers[v.focus_node])
    return all_violations, violations, focus_dossiers, focus_contexts


@coco.fn
async def plan_graph_publication(
    settings: GraphPlanSettings,
    chunk_rows: Sequence[tuple[ChunkPointer, OpenIERecord]] | None = None,
) -> GraphPlanResult:
    """Plan a complete SHACL-conformant PreparedGraphRows without holding any yaw.db write transaction."""
    embeddings = coco.use_context(EMBEDDING_ENDPOINT)
    extraction = coco.use_context(OPENIE_ENDPOINT)
    if (
        settings.program_identity != extraction.program_identity
        or settings.embedding_identity != embeddings.adapter.profile.index_identity
    ):
        raise ValueError("Graph plan settings identity does not match the bound inference endpoints")

    all_traces: list[PredictorTrace] = []
    all_telemetry: list[dict[str, Any]] = []
    entity_def_audit: list[dict[str, Any]] = []
    synonym_audit: list[dict[str, Any]] = []
    shacl_audit: list[dict[str, Any]] = []
    rlm_stats: dict[str, Any] = {
        "missing_entity_defs_count": 0,
        "missing_entity_defs_tokens": 0,
        "homograph_splits": 0,
        "candidate_pairs": 0,
        "synonym_judge_tokens": 0,
        "merged_entities": 0,
        "different_from_edges": 0,
        "broader_edges": 0,
        "related_edges": 0,
        "shacl_violations_before": 0,
        "shacl_violations_initial": 0,
        "shacl_violations_after": 0,
        "shacl_violations_remaining": 0,
        "shacl_healer_tokens": 0,
        "shacl_repairs_applied": 0,
        "shacl_conforms": True,
    }

    if chunk_rows is not None:
        active_items = [(ptr, rec, ()) for ptr, rec in sorted(chunk_rows, key=lambda item: item[0].chunk_id)]
    else:
        with coco.use_context(GRAPH_DB).readonly() as ro_conn:
            active_items = load_active_chunk_inferences(
                ro_conn,
                program_identity=settings.program_identity,
                embedding_identity=settings.embedding_identity,
            )

    if not active_items:
        empty_rows = PreparedGraphRows((), (), (), (), (), (), (), (), ())
        return GraphPlanResult(rows=empty_rows, rlm_stats=rlm_stats)

    chunks: list[tuple[ChunkPointer, str]] = []
    all_records: list[OpenIERecord] = []
    for ptr, rec, chunk_traces in active_items:
        chunk_text = resolve_chunk_text(
            ptr,
            settings.corpus_root,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            min_chunk_size=settings.min_chunk_size,
        )
        chunks.append((ptr, chunk_text))
        all_records.append(rec)
        all_traces.extend(chunk_traces)

    fmt_chunk = getattr(
        embeddings.adapter,
        "format_chunk_for_embedding",
        lambda p, t: DSPyEmbeddingAdapter.format_chunk_for_embedding(embeddings.adapter, p, t),
    )
    fmt_entity = DSPyEmbeddingAdapter.format_entity_for_embedding
    fmt_fact = DSPyEmbeddingAdapter.format_fact_for_embedding

    def _prepare_embed_req(text: str, stage: str) -> EmbeddingRequest:
        if hasattr(embeddings.adapter, "prepare_request"):
            return embeddings.adapter.prepare_request([text], batch_size=64, stage=stage)
        return EmbeddingRequest(
            texts=(text,),
            profile=embeddings.adapter.profile,
            task_mode=getattr(embeddings.adapter, "task_mode", "distinct-titled"),
            batch_size=64,
            max_batch_chars=getattr(embeddings.adapter, "max_batch_chars", 3500),
            stage=stage,
        )

    async def _embed_strings(texts: Sequence[str], *, stage: str) -> list[np.ndarray]:
        if not texts:
            return []
        unique_texts = list(dict.fromkeys(texts))
        embedded_unique = await asyncio.gather(*(embed_text(_prepare_embed_req(t, stage)) for t in unique_texts))
        vec_by_text: dict[str, np.ndarray] = {}
        for t, emb in zip(unique_texts, embedded_unique, strict=True):
            vec_by_text[t] = deserialize_f32_vector(emb.vector_f32)
            all_telemetry.append(dict(emb.telemetry))
        return [vec_by_text[t] for t in texts]

    # 1. Chunk dense vectors (hits @coco.fn(memo=True) from process_chunk when already computed)
    chunk_embed_texts = [fmt_chunk(ptr, text) for ptr, text in chunks]
    chunk_vecs = await _embed_strings(chunk_embed_texts, stage="chunk_embed")
    chunk_vec_pairs: list[tuple[str, np.ndarray]] = [
        (ptr.chunk_id, vec) for (ptr, _), vec in zip(chunks, chunk_vecs, strict=True)
    ]

    # 2. Normalize OpenIE records & synthesize missing entity definitions via memoized define_missing_entities
    chunk_text_by_id = {ptr.chunk_id: text for ptr, text in chunks}
    record_structured: list[tuple[str, list[StructuredTriple], list[tuple[str, str]]]] = []
    all_norm_labels: set[str] = set()
    surface_by_norm: dict[str, str] = {}
    chunk_surface_by_norm: dict[tuple[str, str], str] = {}

    for rec in all_records:
        cid = rec.chunk_id
        normalized = normalize_openie_record(rec)
        normalized_sts = normalized.triples
        ner_entities = normalized.entities
        stage1_defs = normalized.definitions
        all_norm_labels.update(normalized.labels)
        for norm_key, surface in normalized.surfaces.items():
            record_surface_form(surface_by_norm, chunk_surface_by_norm, norm_key, surface, cid)

        missing_norms = collect_missing_entity_norms(normalized_sts, ner_entities, stage1_defs)
        repaired_pairs: Sequence[tuple[str, str]] = ()
        if missing_norms:
            defined = await define_missing_entities(
                MissingEntityInput(
                    chunk_id=cid,
                    passage=chunk_text_by_id.get(cid, ""),
                    missing_entities=tuple(missing_norms),
                    triples=tuple(
                        {
                            "sub": st.subj,
                            "pred": st.pred,
                            "obj": st.obj,
                            "def": st.spo_def,
                        }
                        for st in normalized_sts
                    ),
                    program_identity=settings.program_identity,
                )
            )
            all_traces.extend(defined.traces)
            all_telemetry.extend(defined.telemetry)
            entity_def_audit.extend(defined.audit_trail)
            for entry in defined.audit_trail:
                rlm_stats["missing_entity_defs_count"] += int(entry.get("missing_count", 0))
                rlm_stats["missing_entity_defs_tokens"] += int(entry.get("tokens", 0))
            repaired_pairs = defined.definitions

        normalized_sts, ner_entities = apply_stage1_definitions(
            cid,
            normalized_sts,
            ner_entities,
            stage1_defs,
            repaired_pairs,
            surface_by_norm,
            chunk_surface_by_norm,
        )
        record_structured.append((cid, normalized_sts, ner_entities))

    synthesize_copular_is_a_triples(
        chunks,
        record_structured,
        all_norm_labels,
        surface_by_norm,
        chunk_surface_by_norm,
    )

    sense_state = prepare_sense_resolution_state(
        chunks,
        record_structured,
        all_norm_labels,
        surface_by_norm,
        fmt_entity=fmt_entity,
    )
    prebatch_vec_by_text: dict[str, np.ndarray] = {}
    if sense_state.prebatch_texts:
        p_vecs = await _embed_strings(sense_state.prebatch_texts, stage="entity_embed")
        for ct, pv in zip(sense_state.prebatch_texts, p_vecs, strict=True):
            n_val = float(np.linalg.norm(pv))
            prebatch_vec_by_text[ct] = pv / n_val if n_val > 0 else pv

    materialized = materialize_entities_and_facts(
        record_structured,
        surface_by_norm,
        chunk_surface_by_norm,
        sense_state,
        prebatch_vec_by_text,
        fmt_entity=fmt_entity,
        same_sense_cosine_gate=settings.same_sense_cosine_gate,
        datatype_property_weight=settings.datatype_property_weight,
    )
    rlm_stats["homograph_splits"] += materialized.homograph_splits
    entity_nodes = materialized.entity_nodes
    fact_nodes = materialized.fact_nodes
    precomputed_entity_vecs = materialized.precomputed_entity_vecs

    missing_entities = [(eid, name, defn) for eid, name, defn in entity_nodes if eid not in precomputed_entity_vecs]
    entity_vec_by_id: dict[str, np.ndarray] = dict(precomputed_entity_vecs)
    if missing_entities:
        e_texts = [fmt_entity(name, defn) for _, name, defn in missing_entities]
        e_vecs = await _embed_strings(e_texts, stage="entity_embed")
        for (eid, _, _), vec in zip(missing_entities, e_vecs, strict=True):
            entity_vec_by_id[eid] = vec

    fact_vec_by_id: dict[str, np.ndarray] = {}
    if fact_nodes:
        fact_texts: list[str] = []
        for f_row in fact_nodes:
            _, _s_id, pred, _o_id, raw_json, spo_d, subj_d, obj_d = f_row[:8]
            items = json.loads(raw_json)
            fact_texts.append(fmt_fact(items[0], pred, items[2], spo_d, subj_d, obj_d))
        f_vecs = await _embed_strings(fact_texts, stage="fact_embed")
        for f_row, vec in zip(fact_nodes, f_vecs, strict=True):
            fact_vec_by_id[f_row[0]] = vec

    initial_rows = PreparedGraphRows(
        chunks=tuple(chunks),
        entities=tuple(entity_nodes),
        entity_labels=tuple(materialized.entity_labels_to_upsert),
        facts=tuple(
            (fid, sid, pred, oid, raw, definition, kind, lexical, datatype, language)
            for fid, sid, pred, oid, raw, definition, _sd, _od, kind, lexical, datatype, language in fact_nodes
        ),
        fact_sources=tuple(materialized.fact_sources_batch),
        edges=tuple(materialized.edges_with_sources),
        chunk_vectors=tuple(chunk_vec_pairs),
        entity_vectors=tuple(entity_vec_by_id.items()),
        fact_vectors=tuple(fact_vec_by_id.items()),
    )

    # 3. Stage initial rows in an isolated in-memory SQLite database (zero yaw.db write lock)
    staging_conn = open_yaw_db(":memory:")
    try:
        with staging_conn:
            staging_conn.execute("BEGIN")
            apply_prepared_graph_rows(staging_conn, initial_rows)

        candidates, cand_meta, raw_cand_count = prepare_synonym_candidates(
            staging_conn,
            entity_nodes,
            synonymy_threshold=settings.synonymy_threshold,
            synonymy_top_k=settings.synonymy_top_k,
        )
        rlm_stats["candidate_pairs"] += raw_cand_count
        if candidates or raw_cand_count:
            decisions: list[dict[str, str]] = []
            if candidates:
                batch_limit = max(1, scale_context_budget(8, settings.context_window_tokens))
                max_dossier_chars = min(
                    40000,
                    context_char_limit(12000, settings.context_window_tokens, settings.max_output_tokens),
                )
                evidence_batches = list(
                    iter_synonym_evidence(
                        staging_conn,
                        candidates,
                        settings.corpus_root,
                        ast_context_hops=settings.ast_context_hops,
                        batch_size=batch_limit,
                        max_chars=max_dossier_chars,
                        fits=extraction.program.synonym_batch_fits,
                    )
                )
                judged_batches = await asyncio.gather(
                    *(
                        judge_synonyms(
                            SynonymJudgeInput(
                                batch=ev,
                                program_identity=settings.program_identity,
                                synonym_identity=settings.synonym_identity,
                            )
                        )
                        for ev in evidence_batches
                    )
                )
                for judged in judged_batches:
                    decisions.extend(judged.decisions)
                    all_traces.extend(judged.traces)
                    all_telemetry.extend(judged.telemetry)
                    synonym_audit.extend(judged.audit_trail)
                    rlm_stats["synonym_judge_tokens"] += sum(
                        int(x.get("tokens_used", 0)) for x in (judged.audit_trail or judged.telemetry)
                    )

            apply_synonym_adjudication(
                staging_conn,
                candidates,
                cand_meta,
                decisions,
                materialized.fact_linked_pairs,
                synonymy_threshold=settings.synonymy_threshold,
                rlm_stats=rlm_stats,
            )

        # 4. Closed-World SHACL Linting & RLM Healing on staging_conn
        _, violations, focus_dossiers, _ = collect_shacl_violation_dossiers(
            staging_conn,
            settings.corpus_root,
            ast_context_hops=settings.ast_context_hops,
            context_window_tokens=settings.context_window_tokens,
        )
        rlm_stats["shacl_violations_before"] = len(violations)
        rlm_stats["shacl_violations_initial"] = len(violations)
        if violations:
            healed = await heal_shacl_violations(
                SHACLHealInput(
                    violations=tuple(violations),
                    focus_dossiers=focus_dossiers,
                    focus_context=json.dumps(list(focus_dossiers.values()), ensure_ascii=False),
                    program_identity=settings.program_identity,
                    shacl_identity=settings.shacl_identity,
                )
            )
            all_traces.extend(healed.traces)
            all_telemetry.extend(healed.telemetry)
            shacl_audit.extend(healed.audit_trail)
            rlm_stats["shacl_healer_tokens"] += sum(
                int(x.get("tokens_used", 0)) for x in (healed.audit_trail or healed.telemetry)
            )
            if healed.repairs:
                applied = apply_shacl_repairs(staging_conn, healed.repairs)
                rlm_stats["shacl_repairs_applied"] += applied

        remaining_violations = lint_graph_shacl(staging_conn)
        rlm_stats["shacl_violations_after"] = len(remaining_violations)
        rlm_stats["shacl_violations_remaining"] = len(remaining_violations)
        rlm_stats["shacl_conforms"] = len(remaining_violations) == 0
        if remaining_violations:
            raise YawRuntimeError(
                "CONSISTENCY_ERROR",
                f"Graph build left {len(remaining_violations)} SHACL violations: {remaining_violations[:3]}",
            )

        # 5. Extract final post-merge, post-SHACL-repair PreparedGraphRows from staging_conn
        final_entities = tuple(
            (str(r[0]), str(r[1]), str(r[2]))
            for r in staging_conn.execute(
                "SELECT entity_id, name, definition FROM entities ORDER BY entity_id ASC"
            ).fetchall()
        )
        final_labels = tuple(
            (str(r[0]), str(r[1]), str(r[2]), str(r[3]), str(r[4]))
            for r in staging_conn.execute(
                "SELECT label_norm, entity_id, label_kind, raw_label, chunk_id "
                "FROM entity_labels ORDER BY entity_id ASC, label_norm ASC, chunk_id ASC"
            ).fetchall()
        )
        final_facts = tuple(
            (
                str(r[0]),
                str(r[1]),
                str(r[2]),
                str(r[3]) if r[3] is not None else None,
                str(r[4]),
                str(r[5]),
                str(r[6]),
                str(r[7]) if r[7] is not None else None,
                str(r[8]) if r[8] is not None else None,
                str(r[9]) if r[9] is not None else None,
            )
            for r in staging_conn.execute(
                "SELECT fact_id, subj_id, pred, obj_id, raw_triple_json, definition, "
                "obj_kind, obj_literal_lex, obj_datatype, obj_lang FROM facts ORDER BY fact_id ASC"
            ).fetchall()
        )
        final_fact_sources = tuple(
            (str(r[0]), str(r[1]))
            for r in staging_conn.execute(
                "SELECT fact_id, chunk_id FROM fact_sources ORDER BY fact_id ASC, chunk_id ASC"
            ).fetchall()
        )
        entity_fact_pairs = {
            (str(r[0]), str(r[1]))
            for r in staging_conn.execute(
                "SELECT DISTINCT subj_id, obj_id FROM facts "
                "WHERE COALESCE(obj_kind, 'entity') = 'entity' AND obj_id IS NOT NULL"
            ).fetchall()
        }
        final_edges = tuple(
            (
                str(r[0]),
                str(r[1]),
                str(r[2]),
                (
                    (1.0 if (str(r[0]), str(r[1])) in entity_fact_pairs else settings.datatype_property_weight)
                    if str(r[2]) == "fact"
                    else float(r[3])
                ),
                str(r[4] or ""),
            )
            for r in staging_conn.execute(
                """
                SELECT e.src_id, e.dst_id, e.kind, e.weight, COALESCE(es.chunk_id, '')
                FROM edges e
                LEFT JOIN edge_sources es
                  ON es.src_id = e.src_id AND es.dst_id = e.dst_id AND es.kind = e.kind
                ORDER BY e.src_id ASC, e.dst_id ASC, e.kind ASC, COALESCE(es.chunk_id, '') ASC
                """
            ).fetchall()
        )

        # Ensure any entity/fact whose text changed during merge or SHACL repair is embedded
        initial_entity_meta = {eid: (name, defn) for eid, name, defn in entity_nodes}
        reembed_entities = [
            (eid, name, defn)
            for eid, name, defn in final_entities
            if eid not in entity_vec_by_id or initial_entity_meta.get(eid) != (name, defn)
        ]
        if reembed_entities:
            re_vecs = await _embed_strings(
                [fmt_entity(name, defn) for _, name, defn in reembed_entities],
                stage="entity_embed",
            )
            for (eid, _, _), vec in zip(reembed_entities, re_vecs, strict=True):
                entity_vec_by_id[eid] = vec

        initial_fact_defs = {f_row[0]: f_row[5] for f_row in fact_nodes}
        reembed_facts = [
            f_row
            for f_row in final_facts
            if f_row[0] not in fact_vec_by_id or initial_fact_defs.get(f_row[0]) != f_row[5]
        ]
        if reembed_facts:
            rf_texts: list[str] = []
            for f_row in reembed_facts:
                items = json.loads(f_row[4])
                rf_texts.append(fmt_fact(items[0], f_row[2], items[2], f_row[5]))
            rf_vecs = await _embed_strings(rf_texts, stage="fact_embed")
            for f_row, vec in zip(reembed_facts, rf_vecs, strict=True):
                fact_vec_by_id[f_row[0]] = vec

        final_entity_vectors = tuple((eid, entity_vec_by_id[eid]) for eid, _, _ in final_entities)
        final_fact_vectors = tuple((f_row[0], fact_vec_by_id[f_row[0]]) for f_row in final_facts)

        final_rows = PreparedGraphRows(
            chunks=tuple(chunks),
            entities=final_entities,
            entity_labels=final_labels,
            facts=final_facts,
            fact_sources=final_fact_sources,
            edges=final_edges,
            chunk_vectors=tuple(chunk_vec_pairs),
            entity_vectors=final_entity_vectors,
            fact_vectors=final_fact_vectors,
        )
        return GraphPlanResult(
            rows=final_rows,
            traces=tuple(all_traces),
            telemetry=tuple(all_telemetry),
            entity_def_audit_trail=tuple(entity_def_audit),
            synonym_audit_trail=tuple(synonym_audit),
            shacl_audit_trail=tuple(shacl_audit),
            rlm_stats=rlm_stats,
        )
    finally:
        staging_conn.close()
