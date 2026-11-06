"""Transaction-free CocoIndex graph planner and shared Stage 1-4 deterministic graph builders."""

import asyncio
import json
import re
import sqlite3
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cocoindex as coco
import numpy as np
from cocoindex.connectorkits.fingerprint import fingerprint_object

from yaw.core import (
    ChunkPointer,
    OpenIERecord,
    SHACLViolation,
    StructuredTriple,
    YawRuntimeError,
    detect_scalar_literal,
    is_scalar_label,
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
from yaw.daemon.graph.normalization import normalize_openie_record, preferred_surface
from yaw.daemon.graph.storage import (
    apply_shacl_repairs,
    auto_heal_deterministic_shacl,
    canonicalize_entity_labels_edc,
    deserialize_f32_vector,
    deserialize_openie_record,
    find_entity_disambiguation_candidates,
    get_entity_metadata_batch,
    get_meta,
    inspect_focus_node,
    iter_synonym_evidence,
    lint_graph_shacl,
    merge_entities,
    open_yaw_db,
    pool_confidences,
    resolve_chunk_text,
    upsert_chunk_pointers,
    upsert_chunk_vectors,
    upsert_edges_with_sources,
    upsert_entities,
    upsert_entity_labels,
    upsert_entity_vectors,
    upsert_fact_sources,
    upsert_fact_vectors,
    upsert_facts,
    upsert_semantic_edges,
    write_tx,
)
from yaw.daemon.graph.storage.db import GRAPH_DB
from yaw.daemon.graph.storage.evidence import (
    EntityMention,
    EntityMerge,
    FactAssertion,
    bind_merged_evidence,
    build_evidence,
    canonicalize_fact_text,
    evidence_schemas,
    load_sense_bindings,
    sense_id,
    upsert_evidence,
)
from yaw.daemon.graph.storage.evidence import (
    fact_id as canonical_fact_id,
)
from yaw.daemon.graph.storage.managed_sqlite import TableSpec, create_table
from yaw.daemon.graph.storage.rows import DesiredTable, empty_graph_tables, graph_tables


@dataclass(frozen=True, slots=True)
class GraphPlanSettings:
    corpus_root: Path
    program_identity: str
    embedding_identity: str
    synonym_identity: str = ""
    shacl_identity: str = ""
    auto_merge_cosine_gate: float = 0.95
    auto_merge_difflib_gate: float = 0.96
    synonymy_threshold: float = 0.8
    synonymy_top_k: int = 5
    ast_context_hops: int = 1
    context_window_tokens: int = 32768
    max_output_tokens: int = 4096
    chunk_size: int = 1200
    chunk_overlap: int = 100
    min_chunk_size: int = 50


type FactRow = tuple[str, str, str, str | None, str, str, str, str | None, str | None, str | None]
type VectorRow = tuple[str, np.ndarray]


@dataclass(frozen=True, slots=True)
class PreparedGraphRows:
    chunks: tuple[tuple[ChunkPointer, str], ...]
    entities: tuple[tuple[str, str, str], ...] | tuple[tuple[str, str, str, float | None], ...]
    entity_labels: tuple[tuple[str, str, str, str, str], ...]
    facts: tuple[FactRow, ...]
    fact_sources: tuple[tuple[str, str], ...] | tuple[tuple[str, str, float | None, float | None, float | None], ...]
    edges: tuple[tuple[str, str, str, float, str], ...]
    chunk_vectors: tuple[VectorRow, ...]
    entity_vectors: tuple[VectorRow, ...]
    fact_vectors: tuple[VectorRow, ...]
    mentions: tuple[EntityMention, ...]
    assertions: tuple[FactAssertion, ...]
    input_fingerprints: tuple[tuple[str, str], ...] = ()


def stage_graph_rows(conn: sqlite3.Connection, rows: PreparedGraphRows) -> None:
    """Apply prepared rows and FTS updates; never perform inference or commit."""
    if not conn.in_transaction:
        raise RuntimeError("Graph publication requires an active transaction")
    if conn.execute("PRAGMA database_list").fetchone()[2] != "":
        raise ValueError("Staging must never write a persistent database")
    upsert_chunk_vectors(conn, rows.chunk_vectors)
    upsert_chunk_pointers(conn, rows.chunks)
    upsert_entities(conn, rows.entities)
    upsert_entity_labels(conn, rows.entity_labels)
    upsert_facts(conn, rows.facts)
    upsert_fact_sources(conn, rows.fact_sources)
    upsert_entity_vectors(conn, rows.entity_vectors)
    upsert_fact_vectors(conn, rows.fact_vectors)
    upsert_edges_with_sources(conn, rows.edges)
    upsert_evidence(conn, rows.mentions, rows.assertions)


@dataclass(frozen=True, slots=True)
class GraphPlanResult:
    tables: tuple[DesiredTable, ...]
    chunk_ids: tuple[str, ...]
    traces: tuple[PredictorTrace, ...] = ()
    telemetry: tuple[dict[str, Any], ...] = ()
    entity_def_audit_trail: tuple[dict[str, Any], ...] = ()
    label_judge_audit_trail: tuple[dict[str, Any], ...] = ()
    hierarchy_judge_audit_trail: tuple[dict[str, Any], ...] = ()
    related_judge_audit_trail: tuple[dict[str, Any], ...] = ()
    shacl_audit_trail: tuple[dict[str, Any], ...] = ()
    rlm_stats: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GraphPlanSummary:
    """Publication diagnostics without retaining a second copy of graph vectors."""

    chunk_ids: tuple[str, ...]
    traces: tuple[PredictorTrace, ...] = ()
    telemetry: tuple[dict[str, Any], ...] = ()
    entity_def_audit_trail: tuple[dict[str, Any], ...] = ()
    label_judge_audit_trail: tuple[dict[str, Any], ...] = ()
    hierarchy_judge_audit_trail: tuple[dict[str, Any], ...] = ()
    related_judge_audit_trail: tuple[dict[str, Any], ...] = ()
    shacl_audit_trail: tuple[dict[str, Any], ...] = ()
    rlm_stats: dict[str, Any] = field(default_factory=dict)


def summarize_graph_plan(plan: GraphPlanResult) -> GraphPlanSummary:
    return GraphPlanSummary(
        chunk_ids=plan.chunk_ids,
        traces=plan.traces,
        telemetry=plan.telemetry,
        entity_def_audit_trail=plan.entity_def_audit_trail,
        label_judge_audit_trail=plan.label_judge_audit_trail,
        hierarchy_judge_audit_trail=plan.hierarchy_judge_audit_trail,
        related_judge_audit_trail=plan.related_judge_audit_trail,
        shacl_audit_trail=plan.shacl_audit_trail,
        rlm_stats=dict(plan.rlm_stats),
    )


@dataclass(slots=True)
class MaterializedGraphBatch:
    entity_nodes: list[tuple[str, str, str]]
    entity_labels_to_upsert: list[tuple[str, str, str, str, str]]
    fact_nodes: list[tuple[str, str, str, str | None, str, str, str, str, str, str | None, str, str]]
    fact_sources_batch: list[tuple[str, str]]
    edges_with_sources: list[tuple[str, str, str, float, str]]
    fact_linked_pairs: set[tuple[str, str]]
    homograph_splits: int
    bindings: dict[tuple[str, str, str], tuple[str, str]]


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


def clone_graph_to_memory(conn: sqlite3.Connection, schemas: tuple[TableSpec, ...]) -> sqlite3.Connection:
    """Clone an existing yaw.db graph and vector tables into an isolated in-memory SQLite database."""
    stage_conn = open_yaw_db(":memory:")
    with write_tx(stage_conn):
        for schema in schemas:
            create_table(stage_conn, schema)
        for table, cols in (
            ("chunks", "chunk_id, rel_posix_path, start_line, end_line, title, ast_island_start, ast_island_end"),
            ("entities", "entity_id, name, definition, confidence"),
            ("entity_labels", "label_norm, entity_id, label_kind, raw_label, chunk_id"),
            (
                "entity_mentions",
                "mention_id, chunk_id, source_revision, sense_id, name, definition, resolved_definition, entity_id",
            ),
            (
                "fact_assertions",
                "assertion_id, chunk_id, source_revision, subject_mention_id, object_mention_id, raw_json, fact_id",
            ),
            (
                "facts",
                "fact_id, subj_id, pred, obj_id, raw_triple_json, definition, obj_kind, "
                "obj_literal_lex, obj_datatype, obj_lang, confidence",
            ),
            ("fact_sources", "fact_id, chunk_id, llm_prob, vec_coherence, atomic_confidence"),
            ("edges", "src_id, dst_id, kind, weight, reasoning"),
            ("edge_sources", "src_id, dst_id, kind, chunk_id, llm_prob, vec_coherence, atomic_confidence"),
            ("chunk_vec", "chunk_id, embedding"),
            ("entity_vec", "entity_id, embedding"),
            ("fact_vec", "fact_id, embedding"),
        ):
            placeholders = ",".join("?" for _ in cols.split(","))
            stage_conn.executemany(
                f"INSERT INTO {table} ({cols}) VALUES ({placeholders})",
                conn.execute(f"SELECT {cols} FROM {table}"),
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
    surface_by_norm[norm_key] = preferred_surface(
        surface_by_norm.get(norm_key, clean_raw), clean_raw, label_norm=norm_key
    )
    if cid_key:
        key = (cid_key, norm_key)
        chunk_surface_by_norm[key] = preferred_surface(
            chunk_surface_by_norm.get(key, clean_raw), clean_raw, label_norm=norm_key
        )


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


def materialize_entities_and_facts(
    record_structured: Sequence[tuple[str, list[StructuredTriple], list[tuple[str, str]]]],
    surface_by_norm: Mapping[str, str],
    chunk_surface_by_norm: Mapping[tuple[str, str], str],
    existing_bindings: Mapping[str, tuple[str, str, str]],
) -> MaterializedGraphBatch:
    """Resolve entity mentions with provisional sense-splitting and materialize reified facts and edges."""
    bindings: dict[tuple[str, str, str], tuple[str, str]] = {}
    definitions_by_label: dict[str, set[str]] = defaultdict(set)

    entity_map: dict[str, tuple[str, str, str]] = {}
    entity_labels_to_upsert: list[tuple[str, str, str, str, str]] = []

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

    def _resolve_mention_entity_id(raw_mention: str, defn: str, chunk_id: str) -> str:
        label_norm = str(text_processing(raw_mention))
        clean_def = defn.strip()
        if not label_norm or not clean_def:
            raise ValueError("Entity materialization requires a NER label and resolved definition")
        candidate = sense_id(label_norm, clean_def)
        surface = surface_by_norm[label_norm]
        eid, name, definition = existing_bindings.get(candidate, (candidate, surface, clean_def))
        previous = entity_map.get(eid, (eid, name, definition))
        name = preferred_surface(name, previous[1], surface, label_norm=str(text_processing(name)))
        definition = max((definition, previous[2], clean_def), key=lambda value: (len(value), value))
        entity_map[eid] = (eid, name, definition)
        mention_raw = chunk_surface_by_norm[(chunk_id, label_norm)]
        entity_labels_to_upsert.append((label_norm, eid, _label_kind_for(name, label_norm), mention_raw, chunk_id))
        bindings[(chunk_id, label_norm, clean_def.casefold())] = (eid, clean_def)
        definitions_by_label[label_norm].add(clean_def.casefold())
        return eid

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

            is_literal_obj = st.obj_kind == "literal"
            if is_literal_obj:
                eff_obj_dt = (
                    st.obj_datatype
                    if (st.obj_datatype and st.obj_datatype != "rdfs:Literal")
                    else detect_scalar_literal(st.obj)[1]
                )
                fid = canonical_fact_id(s_id, st.pred, "literal", st.obj, eff_obj_dt, st.obj_lang)
                raw_json = json.dumps([st.subj, st.pred, st.obj])
                prev_fact = fact_nodes_map.get(fid)
                if prev_fact is None or (not prev_fact[5] and st.spo_def):
                    fact_nodes_map[fid] = (
                        fid,
                        s_id,
                        st.pred,
                        None,  # obj_id is strictly None for scalar literals
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
                # Scalar literals NEVER create entities, entity_labels, or edges
            else:
                o_norm_key = str(text_processing(st.obj)) or st.obj.strip().lower()
                o_id = _resolve_mention_entity_id(st.obj, st.obj_def, cid)
                resolved_in_chunk.add((o_norm_key, st.obj_def.strip().lower()))
                chunk_to_entities[cid].add(o_id)

                fid = canonical_fact_id(s_id, st.pred, "entity", o_id)
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
                    edges_with_sources.append((s_id, o_id, "fact", 0.8, cid))

        for e_raw_lbl, e_def in ner_entities:
            if is_scalar_label(e_raw_lbl):
                continue
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
        homograph_splits=sum(max(0, len(values) - 1) for values in definitions_by_label.values()),
        bindings=bindings,
    )


def _is_acronym(short_str: str, long_str: str) -> bool:
    s = "".join(c for c in short_str if c.isalnum()).lower()
    if not s or len(s) < 2:
        return False
    variants = [s]
    if s.endswith("s") and len(s) > 2:
        variants.append(s[:-1])
    tokens = [w for w in re.split(r"[^a-zA-Z0-9]+", long_str) if w]
    if len(tokens) < 2:
        return False
    acronym_initials = "".join(w[0].lower() for w in tokens)
    return any(v == acronym_initials or acronym_initials.startswith(v) for v in variants)


def is_modifier_head_specialization(name_a: str, name_b: str) -> bool:
    """Return True if one concept name is a modifier prefix specialization of the other with a distinct head noun."""
    norm_a = str(text_processing(name_a)).strip().lower()
    norm_b = str(text_processing(name_b)).strip().lower()
    if not norm_a or not norm_b or norm_a == norm_b:
        return False

    tokens_a = norm_a.split()
    tokens_b = norm_b.split()

    if len(tokens_a) != len(tokens_b):
        shorter_toks, longer_toks = (tokens_a, tokens_b) if len(tokens_a) < len(tokens_b) else (tokens_b, tokens_a)
        if len(shorter_toks) > 0 and longer_toks[: len(shorter_toks)] == shorter_toks:
            return True

    clean_a = name_a.strip()
    clean_b = name_b.strip()
    if clean_a.casefold() != clean_b.casefold():
        short_name, long_name = (clean_a, clean_b) if len(clean_a) < len(clean_b) else (clean_b, clean_a)
        if long_name.casefold().startswith(short_name.casefold()):
            suffix = long_name[len(short_name) :]
            if suffix and (suffix[0].isupper() or suffix.startswith(("_", "-", " "))):
                return True

    return False


def _has_disjoint_non_acronym_tokens(name_a: str, name_b: str, lexical_sim: float = 0.0) -> bool:
    """Return True if token sets are completely disjoint and neither is an acronym/initialism of the other."""
    norm_a = str(text_processing(name_a)).strip().lower()
    norm_b = str(text_processing(name_b)).strip().lower()
    if not norm_a or not norm_b or norm_a == norm_b:
        return False
    tokens_a = set(norm_a.split())
    tokens_b = set(norm_b.split())
    if tokens_a & tokens_b:
        return False
    if _is_acronym(name_a, name_b) or _is_acronym(name_b, name_a):
        return False
    return lexical_sim < 0.85


def prepare_synonym_candidates(
    stage_conn: sqlite3.Connection,
    entity_nodes: Sequence[tuple[str, str, str]],
    *,
    synonymy_threshold: float,
    synonymy_top_k: int,
    auto_merge_cosine_gate: float = 0.95,
    auto_merge_difflib_gate: float = 0.96,
    fact_linked_pairs: set[tuple[str, str]] | None = None,
    rlm_stats: dict[str, Any] | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, tuple[str, str]], int, list[EntityMerge]]:
    """Filter valid SKOS concepts, apply the near-1 (cosine, difflib) auto-merge gate, and rank remaining pairs."""
    valid_entity_ids = [eid for eid, name, _ in entity_nodes if bool(text_processing(name))]
    if not valid_entity_ids:
        return [], {}, 0, []
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

    merged_into: dict[str, str] = {}
    auto_merges: list[EntityMerge] = []
    active_fact_pairs = fact_linked_pairs or set()

    def _resolve_active_eid(eid: str) -> str:
        curr = eid
        visited: set[str] = set()
        while curr in merged_into and curr not in visited:
            visited.add(curr)
            curr = merged_into[curr]
        return curr

    def _is_active_fact_pole(u: str, v: str) -> bool:
        target_pair = (min(u, v), max(u, v))
        for fa, fb in active_fact_pairs:
            ra, rb = _resolve_active_eid(fa), _resolve_active_eid(fb)
            if (min(ra, rb), max(ra, rb)) == target_pair:
                return True
        return False

    with write_tx(stage_conn):
        for raw_a, raw_b, d_sim, l_sim in raw_candidates:
            ea = _resolve_active_eid(raw_a)
            eb = _resolve_active_eid(raw_b)
            if ea == eb:
                continue
            name_a, def_a = cand_meta.get(ea, (ea, ""))
            name_b, def_b = cand_meta.get(eb, (eb, ""))

            # Guardrail 2: Never auto-merge modifier-head specializations or disjoint non-acronym concepts
            if is_modifier_head_specialization(name_a, name_b):
                continue
            if _has_disjoint_non_acronym_tokens(name_a, name_b, float(l_sim)):
                continue

            # Guardrail 3: Case-insensitive identity consolidation before disambiguation
            is_case_variant = (
                text_processing(name_a) == text_processing(name_b) or name_a.casefold() == name_b.casefold()
            )
            meets_gate = (
                is_case_variant and float(d_sim) >= synonymy_threshold and not _is_active_fact_pole(ea, eb)
            ) or (float(d_sim) >= auto_merge_cosine_gate and float(l_sim) >= auto_merge_difflib_gate)
            if not meets_gate:
                continue

            if text_processing(name_a) != text_processing(name_b) and _is_active_fact_pole(ea, eb):
                continue

            casing_a = 1 if any(c.isupper() for c in name_a) else 0
            casing_b = 1 if any(c.isupper() for c in name_b) else 0
            key_a = (casing_a, len(def_a), -len(name_a), ea)
            key_b = (casing_b, len(def_b), -len(name_b), eb)
            dst, src = (ea, eb) if key_a >= key_b else (eb, ea)
            merged = merge_entities(stage_conn, src, dst, label_kind="skos:altLabel")
            src, dst = merged.removed_id, merged.canonical_id
            auto_merges.append(merged)
            merged_into[src] = dst
            cand_meta.update(get_entity_metadata_batch(stage_conn, [dst]))
            if rlm_stats is not None:
                rlm_stats["merged_entities"] = int(rlm_stats.get("merged_entities", 0)) + 1

    dedup_remaining: dict[tuple[str, str], tuple[float, float]] = {}
    for raw_a, raw_b, d_sim, l_sim in raw_candidates:
        ea = _resolve_active_eid(raw_a)
        eb = _resolve_active_eid(raw_b)
        if ea == eb:
            continue
        pkey = (min(ea, eb), max(ea, eb))
        prev = dedup_remaining.get(pkey)
        if prev is None:
            dedup_remaining[pkey] = (float(d_sim), float(l_sim))
        else:
            dedup_remaining[pkey] = (max(prev[0], float(d_sim)), max(prev[1], float(l_sim)))

    candidates: list[dict[str, Any]] = []
    for (ea, eb), (d_sim, l_sim) in dedup_remaining.items():
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
            float(c["lexical_sim"]) >= 0.70,
            0.5 * float(c["lexical_sim"]) + 0.5 * float(c["dense_sim"]),
            float(c["similarity"]),
        ),
        reverse=True,
    )
    return candidates, cand_meta, len(raw_candidates), auto_merges


def apply_synonym_adjudication(
    stage_conn: sqlite3.Connection,
    candidates: Sequence[dict[str, Any]],
    cand_meta: Mapping[str, tuple[str, str]],
    decisions: Sequence[Mapping[str, Any]],
    fact_linked_pairs: set[tuple[str, str]],
    *,
    synonymy_threshold: float,
    rlm_stats: dict[str, Any],
    corpus_root: Path | str | None = None,
) -> tuple[list[EntityMerge], list[tuple[str, str, str, float, str]]]:
    """Apply Stage 3 Gemma 4 synonym decisions on stage_conn with two-phase topological resolution."""
    _ = synonymy_threshold
    cand_meta = dict(cand_meta)
    merged_into: dict[str, str] = {}
    hidden_label_norms: dict[str, set[str]] = defaultdict(set)

    def _resolve_active_eid(eid: str) -> str:
        curr = eid
        visited: set[str] = set()
        while curr in merged_into and curr not in visited:
            visited.add(curr)
            curr = merged_into[curr]
        return curr

    staged_merges: list[EntityMerge] = []
    raw_semantic_batch: list[tuple[str, str, str, float, str]] = []
    cand_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for c in candidates:
        ea, eb = str(c["entity_a"]), str(c["entity_b"])
        cand_by_pair[(min(ea, eb), max(ea, eb))] = c

    # Empirical mean lexical similarity of accepted skos:altLabel pairs
    alt_label_sims: list[float] = []
    for dec in decisions:
        rel_check = str(dec.get("relation") or dec.get("decision") or "").strip()
        if rel_check in ("skos:altLabel", "exact_match"):
            pid_check = str(dec.get("pair_id", "")).strip()
            if pid_check and "|" in pid_check:
                r_src, r_tgt = pid_check.split("|", 1)
                pk = (min(r_src.strip(), r_tgt.strip()), max(r_src.strip(), r_tgt.strip()))
                c_item = cand_by_pair.get(pk)
                if c_item:
                    alt_label_sims.append(float(c_item.get("lexical_sim", 0.0)))
    avg_alt_label_sim: float | None = sum(alt_label_sims) / len(alt_label_sims) if alt_label_sims else None

    def _is_active_fact_pole(u: str, v: str) -> bool:
        target_pair = (min(u, v), max(u, v))
        for fa, fb in fact_linked_pairs:
            ra, rb = _resolve_active_eid(fa), _resolve_active_eid(fb)
            if (min(ra, rb), max(ra, rb)) == target_pair:
                return True
        return False

    with write_tx(stage_conn):
        for dec in decisions:
            pid = str(dec.get("pair_id", "")).strip()
            if dec.get("source_id") and dec.get("target_id"):
                raw_src = str(dec["source_id"]).strip()
                raw_tgt = str(dec["target_id"]).strip()
            elif "|" in pid:
                ea, eb = pid.split("|", 1)
                raw_src, raw_tgt = ea.strip(), eb.strip()
            else:
                raw_src = str(dec.get("source_id", "")).strip()
                raw_tgt = str(dec.get("target_id", "")).strip()
            rel = str(dec.get("relation") or dec.get("decision") or "").strip()
            if not raw_src or not raw_tgt:
                continue
            pkey = (min(raw_src, raw_tgt), max(raw_src, raw_tgt))
            if pkey not in cand_by_pair:
                continue
            src = _resolve_active_eid(raw_src)
            tgt = _resolve_active_eid(raw_tgt)
            if src == tgt:
                continue
            cand_info = cand_by_pair[pkey]
            sim = float(cand_info.get("similarity", 0.5) or 0.5)
            name_src, _ = cand_meta.get(src, cand_meta.get(raw_src, (src, "")))
            name_tgt, _ = cand_meta.get(tgt, cand_meta.get(raw_tgt, (tgt, "")))

            reas = str(dec.get("reasoning", "") or "").strip()
            dec_prob = dec.get("llm_prob")
            dec_llm_prob = float(dec_prob) if dec_prob is not None else None
            edge_weight = pool_confidences(dec_llm_prob, sim) or sim
            if rel in ("skos:altLabel", "skos:hiddenLabel", "exact_match"):
                # Guardrail 2: Head-Noun & Proper-Noun Modifier Invariant
                if is_modifier_head_specialization(name_src, name_tgt):
                    child_id = src if len(name_src) >= len(name_tgt) else tgt
                    parent_id = tgt if child_id == src else src
                    raw_semantic_batch.append((child_id, parent_id, "skos:broader", edge_weight, reas))
                    continue
                if _has_disjoint_non_acronym_tokens(name_src, name_tgt, sim):
                    raw_semantic_batch.append((src, tgt, "skos:related", edge_weight, reas))
                    continue

                if text_processing(name_src) != text_processing(name_tgt) and _is_active_fact_pole(src, tgt):
                    raw_semantic_batch.append((src, tgt, "skos:related", edge_weight, reas))
                    continue
                canon = str(dec.get("canonical", "") or "").strip()
                canon_id = _resolve_active_eid(canon) if canon else ""
                if canon and (
                    canon.upper() == "A"
                    or canon in (src, raw_src)
                    or canon_id == src
                    or (
                        text_processing(canon) == text_processing(name_src)
                        and text_processing(canon) != text_processing(name_tgt)
                    )
                ):
                    merge_src, merge_dst = tgt, src
                elif canon and (
                    canon.upper() == "B"
                    or canon in (tgt, raw_tgt)
                    or canon_id == tgt
                    or (
                        text_processing(canon) == text_processing(name_tgt)
                        and text_processing(canon) != text_processing(name_src)
                    )
                ):
                    merge_src, merge_dst = src, tgt
                else:
                    merge_src, merge_dst = src, tgt
                lkind = (
                    "skos:hiddenLabel"
                    if rel == "skos:hiddenLabel" or str(dec.get("label_type", "")) == "hiddenLabel"
                    else "skos:altLabel"
                )
                if lkind == "skos:hiddenLabel":
                    typo_name = name_src if merge_src == src else name_tgt
                    src_norm = str(text_processing(typo_name)) or typo_name.strip().lower()
                    if src_norm:
                        hidden_label_norms[merge_dst].add(src_norm)
                if merge_src in hidden_label_norms:
                    hidden_label_norms[merge_dst].update(hidden_label_norms.pop(merge_src))

                merged_def = str(dec.get("definition", "") or "").strip() or None
                merged = merge_entities(
                    stage_conn,
                    merge_src,
                    merge_dst,
                    label_kind=lkind,
                    corpus_root=corpus_root,
                    merged_definition=merged_def,
                )
                merge_src, merge_dst = merged.removed_id, merged.canonical_id
                # Label preference and the stable resource representative are independent.
                for owner in (src, tgt):
                    if owner != merge_dst and owner in hidden_label_norms:
                        hidden_label_norms[merge_dst].update(hidden_label_norms.pop(owner))
                cand_meta.update(get_entity_metadata_batch(stage_conn, [merge_dst]))
                staged_merges.append(merged)
                merged_into[merge_src] = merge_dst
                rlm_stats["merged_entities"] = int(rlm_stats.get("merged_entities", 0)) + 1
            else:
                if rel in ("owl:differentFrom", "different", "different_from"):
                    # Guardrail 1: Homograph-Only owl:differentFrom
                    is_homograph = text_processing(name_src) == text_processing(name_tgt)
                    lex_sim = float(cand_info.get("lexical_sim", 0.0))
                    meets_homograph_boundary = avg_alt_label_sim is not None and lex_sim >= avg_alt_label_sim
                    if is_homograph or meets_homograph_boundary:
                        raw_semantic_batch.append((src, tgt, "owl:differentFrom", 0.0, reas))
                elif rel == "skos:broader":
                    child_id, parent_id = src, tgt
                    raw_semantic_batch.append((child_id, parent_id, "skos:broader", edge_weight, reas))
                elif rel in ("skos:related", "related"):
                    raw_semantic_batch.append((src, tgt, "skos:related", edge_weight, reas))

        edge_priority = {"skos:related": 1, "skos:broader": 2, "owl:differentFrom": 3}
        pair_edges: dict[tuple[str, str], tuple[str, str, str, float, str]] = {}
        for s_raw, o_raw, r_kind, w, r_reas in raw_semantic_batch:
            s_act = _resolve_active_eid(s_raw)
            o_act = _resolve_active_eid(o_raw)
            if s_act == o_act:
                continue
            upair = (min(s_act, o_act), max(s_act, o_act))
            prev = pair_edges.get(upair)
            if prev is None:
                pair_edges[upair] = (s_act, o_act, r_kind, w, r_reas)
            else:
                _, _, prev_kind, prev_w, prev_reas = prev
                if edge_priority[r_kind] > edge_priority[prev_kind] or (
                    edge_priority[r_kind] == edge_priority[prev_kind] and w > prev_w
                ):
                    pair_edges[upair] = (s_act, o_act, r_kind, w, r_reas or prev_reas)

        resolved_semantic_batch: list[tuple[str, str, str, float, str]] = []
        for s_act, o_act, r_kind, w, r_reas in pair_edges.values():
            resolved_semantic_batch.append((s_act, o_act, r_kind, w, r_reas))
            if r_kind == "owl:differentFrom":
                rlm_stats["different_from_edges"] = int(rlm_stats.get("different_from_edges", 0)) + 1
            elif r_kind == "skos:broader":
                rlm_stats["broader_edges"] = int(rlm_stats.get("broader_edges", 0)) + 1
            elif r_kind == "skos:related":
                rlm_stats["related_edges"] = int(rlm_stats.get("related_edges", 0)) + 1

        if resolved_semantic_batch:
            upsert_semantic_edges(stage_conn, resolved_semantic_batch)

        canonicalize_entity_labels_edc(
            stage_conn,
            corpus_root=corpus_root,
            hidden_label_norms_by_entity=hidden_label_norms,
        )

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
    frozen_text: Mapping[str, str] | None = None,
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
    label_judge_audit: list[dict[str, Any]] = []
    hierarchy_judge_audit: list[dict[str, Any]] = []
    related_judge_audit: list[dict[str, Any]] = []
    shacl_audit: list[dict[str, Any]] = []
    rlm_stats: dict[str, Any] = {
        "missing_entity_defs_count": 0,
        "missing_entity_defs_tokens": 0,
        "homograph_splits": 0,
        "candidate_pairs": 0,
        "label_judge_tokens": 0,
        "hierarchy_judge_tokens": 0,
        "related_judge_tokens": 0,
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

    if frozen_text is not None:
        # Retired child targets may still exist until the parent finishes syncing.
        # The declared frozen source set, rather than those old rows, owns liveness.
        active_items = [item for item in active_items if item[0].chunk_id in frozen_text]
        if {item[0].chunk_id for item in active_items} != frozen_text.keys():
            raise ValueError("Frozen source snapshot has incomplete inference artifacts")

    if not active_items:
        return GraphPlanResult(tables=empty_graph_tables(), chunk_ids=(), rlm_stats=rlm_stats)

    chunks: list[tuple[ChunkPointer, str]] = []
    all_records: list[OpenIERecord] = []
    for ptr, rec, chunk_traces in active_items:
        chunk_text = (
            frozen_text[ptr.chunk_id]
            if frozen_text is not None
            else resolve_chunk_text(
                ptr,
                settings.corpus_root,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                min_chunk_size=settings.min_chunk_size,
            )
        )
        chunks.append((ptr, chunk_text))
        all_records.append(rec)
        all_traces.extend(chunk_traces)

    # Reuse only verified unchanged contributions. Edits, removals and semantic
    # profile changes require recomputing their graph interpretation.
    input_fingerprints = {
        ptr.chunk_id: fingerprint_object((settings, ptr, text, record)).hex()
        for (ptr, text), record in zip(chunks, all_records, strict=True)
    }
    with coco.use_context(GRAPH_DB).readonly() as ro_conn:
        saved = get_meta(ro_conn, "graph_input_fingerprints")
        published_inputs = dict(json.loads(saved)) if saved is not None else {}
        published_ids = {row[0] for row in ro_conn.execute("SELECT chunk_id FROM chunks")}
    incremental = (
        bool(published_inputs)
        and published_ids == published_inputs.keys()
        and all(input_fingerprints.get(cid) == digest for cid, digest in published_inputs.items())
    )
    complete_chunks = tuple(chunks)
    if incremental:
        new_items = [
            (chunk, record)
            for chunk, record in zip(chunks, all_records, strict=True)
            if chunk[0].chunk_id not in published_inputs
        ]
        chunks = [chunk for chunk, _ in new_items]
        all_records = [record for _, record in new_items]
        all_traces = [
            trace for ptr, _, traces in active_items if ptr.chunk_id not in published_inputs for trace in traces
        ]
    rlm_stats["reused_chunks"] = len(published_inputs) if incremental else 0
    rlm_stats["processed_chunks"] = len(chunks)
    del active_items

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

    with coco.use_context(GRAPH_DB).readonly() as ro_conn:
        existing_bindings = load_sense_bindings(ro_conn, all_norm_labels) if incremental else {}
    materialized = materialize_entities_and_facts(
        record_structured,
        surface_by_norm,
        chunk_surface_by_norm,
        existing_bindings,
    )
    mentions: list[EntityMention] = []
    assertions: list[FactAssertion] = []
    for (pointer, text), record in zip(chunks, all_records, strict=True):
        raw_mentions, raw_assertions = build_evidence(
            pointer, text, settings.program_identity, record, materialized.bindings
        )
        mentions.extend(raw_mentions)
        assertions.extend(raw_assertions)
    rlm_stats["homograph_splits"] += materialized.homograph_splits
    entity_nodes = materialized.entity_nodes
    fact_nodes = materialized.fact_nodes

    entity_vec_by_id: dict[str, np.ndarray] = {}
    if entity_nodes:
        e_texts = [fmt_entity(name, defn) for _, name, defn in entity_nodes]
        e_vecs = await _embed_strings(e_texts, stage="entity_embed")
        for (eid, _, _), vec in zip(entity_nodes, e_vecs, strict=True):
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
        mentions=tuple(mentions),
        assertions=tuple(assertions),
    )

    schemas = await evidence_schemas()

    # 3. Stage initial rows in an isolated in-memory SQLite database (zero yaw.db write lock)
    with coco.use_context(GRAPH_DB).readonly() as ro_conn:
        staging_conn = clone_graph_to_memory(ro_conn, schemas) if incremental else open_yaw_db(":memory:")
    try:
        with write_tx(staging_conn):
            for schema in schemas:
                create_table(staging_conn, schema)
        initial_entity_meta = {
            eid: (name, definition)
            for eid, name, definition in staging_conn.execute("SELECT entity_id, name, definition FROM entities")
        }
        initial_entity_meta.update({eid: (name, definition) for eid, name, definition in entity_nodes})
        initial_fact_defs = {
            row[0]: (row[1], row[2])
            for row in staging_conn.execute("SELECT fact_id, raw_triple_json, definition FROM facts")
        }
        initial_fact_defs.update({row[0]: (row[4], row[5]) for row in fact_nodes})
        for eid, blob in staging_conn.execute("SELECT entity_id, embedding FROM entity_vec"):
            entity_vec_by_id.setdefault(eid, deserialize_f32_vector(blob))
        for fid, blob in staging_conn.execute("SELECT fact_id, embedding FROM fact_vec"):
            fact_vec_by_id.setdefault(fid, deserialize_f32_vector(blob))
        with staging_conn:
            staging_conn.execute("BEGIN")
            stage_graph_rows(staging_conn, initial_rows)

        candidates, cand_meta, raw_cand_count, merges = prepare_synonym_candidates(
            staging_conn,
            entity_nodes,
            synonymy_threshold=settings.synonymy_threshold,
            synonymy_top_k=settings.synonymy_top_k,
            auto_merge_cosine_gate=settings.auto_merge_cosine_gate,
            auto_merge_difflib_gate=settings.auto_merge_difflib_gate,
            fact_linked_pairs=materialized.fact_linked_pairs,
            rlm_stats=rlm_stats,
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
                evidence_batches = iter_synonym_evidence(
                    staging_conn,
                    candidates,
                    settings.corpus_root,
                    ast_context_hops=settings.ast_context_hops,
                    batch_size=batch_limit,
                    max_chars=max_dossier_chars,
                    fits=extraction.program.synonym_batch_fits,
                )
                for ev in evidence_batches:
                    judged = await judge_synonyms(
                        SynonymJudgeInput(
                            batch=ev,
                            program_identity=settings.program_identity,
                            synonym_identity=settings.synonym_identity,
                        )
                    )
                    decisions.extend(judged.decisions)
                    all_traces.extend(judged.traces)
                    all_telemetry.extend(judged.telemetry)
                    for entry in judged.audit_trail:
                        pred_name = entry.get("predictor") or entry.get("stage")
                        if pred_name == "hierarchy_judge":
                            hierarchy_judge_audit.append(entry)
                        elif pred_name == "related_judge":
                            related_judge_audit.append(entry)
                        else:
                            label_judge_audit.append(entry)
                    rlm_stats["label_judge_tokens"] += sum(
                        int(x.get("tokens_used", 0))
                        for x in (judged.audit_trail or judged.telemetry)
                        if (x.get("predictor") or x.get("stage")) == "label_judge"
                    )
                    rlm_stats["hierarchy_judge_tokens"] += sum(
                        int(x.get("tokens_used", 0))
                        for x in (judged.audit_trail or judged.telemetry)
                        if (x.get("predictor") or x.get("stage")) == "hierarchy_judge"
                    )
                    rlm_stats["related_judge_tokens"] += sum(
                        int(x.get("tokens_used", 0))
                        for x in (judged.audit_trail or judged.telemetry)
                        if (x.get("predictor") or x.get("stage")) == "related_judge"
                    )

            judged_merges, _ = apply_synonym_adjudication(
                staging_conn,
                candidates,
                cand_meta,
                decisions,
                materialized.fact_linked_pairs,
                synonymy_threshold=settings.synonymy_threshold,
                rlm_stats=rlm_stats,
                corpus_root=settings.corpus_root,
            )
            merges.extend(judged_merges)

        for merged in merges:
            bind_merged_evidence(staging_conn, merged)
        canonicalize_fact_text(staging_conn)

        # 4. Closed-World SHACL Deterministic Auto-Healing & RLM Healing on staging_conn
        auto_healed = auto_heal_deterministic_shacl(staging_conn)
        rlm_stats["shacl_deterministic_healed"] = auto_healed
        rlm_stats["shacl_repairs_applied"] += auto_healed

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
                int(x.get("tokens_used", 0)) for x in healed.audit_trail if x.get("mode") == "rlm"
            )
            if healed.repairs:

                def merge_repaired_entities(src: str, dst: str, kind: str) -> bool:
                    merged = merge_entities(staging_conn, src, dst, label_kind=kind)
                    bind_merged_evidence(staging_conn, merged)
                    return True

                applied = apply_shacl_repairs(staging_conn, healed.repairs, merge_entities_fn=merge_repaired_entities)
                canonicalize_fact_text(staging_conn)
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
        # Ensure any entity/fact whose text changed during merge or SHACL repair is embedded
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
            upsert_entity_vectors(staging_conn, [(eid, entity_vec_by_id[eid]) for eid, _, _ in reembed_entities])

        reembed_facts = [
            f_row
            for f_row in final_facts
            if f_row[0] not in fact_vec_by_id or initial_fact_defs.get(f_row[0]) != (f_row[4], f_row[5])
        ]
        if reembed_facts:
            rf_texts: list[str] = []
            for f_row in reembed_facts:
                items = json.loads(f_row[4])
                rf_texts.append(fmt_fact(items[0], f_row[2], items[2], f_row[5]))
            rf_vecs = await _embed_strings(rf_texts, stage="fact_embed")
            for f_row, vec in zip(reembed_facts, rf_vecs, strict=True):
                fact_vec_by_id[f_row[0]] = vec

        upsert_fact_vectors(staging_conn, [(row[0], fact_vec_by_id[row[0]]) for row in final_facts])
        return GraphPlanResult(
            tables=graph_tables(staging_conn, complete_chunks, tuple(sorted(input_fingerprints.items()))),
            chunk_ids=tuple(sorted(pointer.chunk_id for pointer, _ in complete_chunks)),
            traces=tuple(all_traces),
            telemetry=tuple(all_telemetry),
            entity_def_audit_trail=tuple(entity_def_audit),
            label_judge_audit_trail=tuple(label_judge_audit),
            hierarchy_judge_audit_trail=tuple(hierarchy_judge_audit),
            related_judge_audit_trail=tuple(related_judge_audit),
            shacl_audit_trail=tuple(shacl_audit),
            rlm_stats=rlm_stats,
        )
    finally:
        staging_conn.close()
