"""Source-owned evidence and deterministic graph identities."""

import hashlib
import json
import math
import sqlite3
from collections.abc import Iterable, Mapping
from dataclasses import asdict, astuple, dataclass
from itertools import groupby

from yaw.core import (
    ChunkPointer,
    OpenIERecord,
    bayesian_noisy_or,
    compute_mdhash_id,
    detect_scalar_literal,
    is_irreflexive_statement,
    is_scalar_label,
    pool_confidences,
    text_processing,
)

from .db import batched, write_tx
from .managed_sqlite import TableSpec

EVIDENCE_INDEXES = {
    "entity_candidate_vectors": (("entity_id",),),
    "entity_mentions": (("chunk_id",), ("entity_id",), ("sense_id",)),
    "fact_assertions": (("chunk_id",), ("fact_id",)),
}


def sense_id(label: str, definition: str) -> str:
    return compute_mdhash_id(json.dumps([text_processing(label), definition.strip().casefold()]), prefix="entity-")


def fact_id(subject: str, predicate: str, kind: str, obj: str, datatype: str = "", language: str = "") -> str:
    return compute_mdhash_id(json.dumps([subject, predicate, kind, obj, datatype, language]), prefix="fact-")


@dataclass(frozen=True, slots=True)
class EntityMention:
    mention_id: str
    chunk_id: str
    source_revision: str
    sense_id: str
    name: str
    definition: str
    resolved_definition: str
    entity_id: str
    llm_prob: float | None = None


@dataclass(frozen=True, slots=True)
class FactAssertion:
    assertion_id: str
    chunk_id: str
    source_revision: str
    subject_mention_id: str
    object_mention_id: str | None
    raw_json: str
    fact_id: str | None


@dataclass(frozen=True, slots=True)
class CandidateVector:
    sense_id: str
    entity_id: str
    name: str
    definition: str
    embedding: bytes


async def evidence_schemas() -> tuple[TableSpec, ...]:
    from cocoindex.connectors import sqlite

    return (
        TableSpec(
            "entity_candidate_vectors",
            await sqlite.TableSchema[CandidateVector].from_class(CandidateVector, primary_key=["sense_id"]),
            indexes=EVIDENCE_INDEXES["entity_candidate_vectors"],
        ),
        TableSpec(
            "entity_mentions",
            await sqlite.TableSchema[EntityMention].from_class(EntityMention, primary_key=["mention_id"]),
            indexes=EVIDENCE_INDEXES["entity_mentions"],
        ),
        TableSpec(
            "fact_assertions",
            await sqlite.TableSchema[FactAssertion].from_class(FactAssertion, primary_key=["assertion_id"]),
            indexes=EVIDENCE_INDEXES["fact_assertions"],
        ),
    )


def build_evidence(
    pointer: ChunkPointer,
    text: str,
    program_identity: str,
    record: OpenIERecord,
    bindings: Mapping[tuple[str, str, str], tuple[str, str]],
) -> tuple[tuple[EntityMention, ...], tuple[FactAssertion, ...]]:
    occurrence = (pointer.rel_posix_path, pointer.start_line, pointer.end_line)
    revision = compute_mdhash_id(
        json.dumps(
            [
                occurrence,
                hashlib.sha256(text.encode()).hexdigest(),
                program_identity,
                pointer.ast_island_start,
                pointer.ast_island_end,
            ]
        ),
        prefix="revision-",
    )
    mentions: dict[tuple[str, str], EntityMention] = {}
    probabilities = {(name, definition): probability for name, definition, probability in record.entity_probabilities}

    def mention(name: str, definition: str) -> EntityMention:
        key = (name, definition)
        if key not in mentions:
            normalized = str(text_processing(name))
            choices = [
                (key, value)
                for key, value in bindings.items()
                if key[0] == pointer.chunk_id
                and key[1] == normalized
                and (not definition.strip() or key[2] == definition.strip().casefold())
            ]
            if len(choices) != 1:
                raise ValueError(f"Resource {name!r} does not have one resolved NER sense")
            _, (entity, resolved_definition) = choices[0]
            mentions[key] = EntityMention(
                compute_mdhash_id(json.dumps([occurrence, revision, name, definition]), prefix="mention-"),
                pointer.chunk_id,
                revision,
                sense_id(name, resolved_definition),
                name,
                definition,
                resolved_definition,
                entity,
                probabilities.get(key),
            )
        return mentions[key]

    for name, definition in record.entity_definitions:
        if not is_scalar_label(name):
            mention(name, definition)
    defined = {name for name, _ in record.entity_definitions}
    for name in record.entities:
        if name not in defined and not is_scalar_label(name):
            mention(name, "")
    assertions = []
    for triple in record.structured_triples:
        subject = mention(triple.subj, triple.subj_def)
        obj = mention(triple.obj, triple.obj_def) if triple.obj_kind == "entity" else None
        raw_data = asdict(triple)
        raw = json.dumps(raw_data, sort_keys=True, ensure_ascii=False)
        raw_data.pop("llm_prob")
        identity = json.dumps(raw_data, sort_keys=True, ensure_ascii=False)
        assertions.append(
            FactAssertion(
                compute_mdhash_id(json.dumps([occurrence, revision, identity]), prefix="assertion-"),
                pointer.chunk_id,
                revision,
                subject.mention_id,
                obj.mention_id if obj else None,
                raw,
                None
                if (obj is not None and subject.entity_id == obj.entity_id)
                or not is_irreflexive_statement(
                    triple.subj,
                    triple.pred,
                    triple.obj,
                    subject.resolved_definition,
                    obj.resolved_definition if obj else "",
                    triple.obj_kind,
                )
                else fact_id(
                    subject.entity_id,
                    triple.pred.strip(),
                    triple.obj_kind,
                    obj.entity_id if obj else triple.obj.strip(),
                    (
                        triple.obj_datatype
                        if triple.obj_datatype and triple.obj_datatype != "rdfs:Literal"
                        else detect_scalar_literal(triple.obj)[1]
                    )
                    if obj is None
                    else "",
                    triple.obj_lang,
                ),
            )
        )
    return tuple(mentions.values()), tuple(assertions)


def upsert_evidence(
    conn: sqlite3.Connection, mentions: tuple[EntityMention, ...], assertions: tuple[FactAssertion, ...]
) -> None:
    conn.executemany(
        "INSERT INTO entity_mentions VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(mention_id) "
        "DO UPDATE SET entity_id=excluded.entity_id,llm_prob=excluded.llm_prob",
        (astuple(row) for row in mentions),
    )
    conn.executemany(
        "INSERT INTO fact_assertions VALUES (?,?,?,?,?,?,?) ON CONFLICT(assertion_id) "
        "DO UPDATE SET fact_id=excluded.fact_id,raw_json=excluded.raw_json",
        (astuple(row) for row in assertions),
    )


def restore_asserted_fact_sources(conn: sqlite3.Connection) -> int:
    """Restore missing provenance from bound source assertions without inventing scores."""
    with write_tx(conn):
        rows = conn.execute(
            """
            SELECT a.fact_id, a.chunk_id, MAX(json_extract(a.raw_json, '$.llm_prob'))
            FROM fact_assertions a
            JOIN facts f ON f.fact_id = a.fact_id
            JOIN chunks c ON c.chunk_id = a.chunk_id
            JOIN entity_mentions s ON s.mention_id = a.subject_mention_id
                AND s.chunk_id = a.chunk_id AND s.source_revision = a.source_revision
                AND s.entity_id = f.subj_id
            LEFT JOIN entity_mentions o ON o.mention_id = a.object_mention_id
                AND o.chunk_id = a.chunk_id AND o.source_revision = a.source_revision
                AND o.entity_id = f.obj_id
            WHERE ((f.obj_kind = 'entity' AND o.mention_id IS NOT NULL)
                OR (f.obj_kind = 'literal' AND a.object_mention_id IS NULL))
                AND NOT EXISTS (
                    SELECT 1 FROM fact_sources fs
                    WHERE fs.fact_id = a.fact_id AND fs.chunk_id = a.chunk_id
                )
            GROUP BY a.fact_id, a.chunk_id
            """
        )
        return conn.executemany(
            "INSERT INTO fact_sources (fact_id,chunk_id,llm_prob,atomic_confidence) VALUES (?,?,?,?)",
            ((fid, cid, probability, pool_confidences(probability)) for fid, cid, probability in rows),
        ).rowcount


def refresh_source_confidences(conn: sqlite3.Connection) -> None:
    """Pool token votes with immutable source-role/context cosine measurements."""

    def vote(probability: float | None, coherences: tuple[float, ...]) -> tuple[float | None, float, float]:
        if any(c is None or not math.isfinite(c) for c in coherences):
            raise ValueError("Source semantic coherence must be measured and finite")
        if probability is not None and (not math.isfinite(probability) or not 0 <= probability <= 1):
            raise ValueError("Invalid source model probability")
        coherence = pool_confidences(*coherences)
        assert coherence is not None
        atomic = pool_confidences(probability, coherence)
        assert atomic is not None
        return probability, coherence, atomic

    rows = conn.execute(
        "SELECT m.entity_id,m.chunk_id,m.llm_prob,1-vec_distance_cosine(v.embedding,c.embedding) "
        "FROM entity_mentions m LEFT JOIN entity_candidate_vectors v ON v.sense_id=m.sense_id "
        "LEFT JOIN chunk_vec c ON c.chunk_id=m.chunk_id ORDER BY m.entity_id,m.chunk_id,m.mention_id"
    )
    for (entity, chunk), observations in groupby(rows, key=lambda row: tuple(row[:2])):
        probability, coherence, atomic = max((vote(row[2], (row[3],)) for row in observations), key=lambda v: v[2])
        for src, dst in ((entity, chunk), (chunk, entity)):
            conn.execute(
                "INSERT INTO edge_sources (src_id,dst_id,kind,chunk_id,llm_prob,vec_coherence,atomic_confidence) "
                "VALUES (?,?,'passage',?,?,?,?) ON CONFLICT(src_id,dst_id,kind,chunk_id) DO UPDATE SET "
                "llm_prob=excluded.llm_prob,vec_coherence=excluded.vec_coherence,atomic_confidence=excluded.atomic_confidence",
                (src, dst, chunk, probability, coherence, atomic),
            )
            conn.execute(
                "INSERT INTO edges (src_id,dst_id,kind,weight) VALUES (?,?,'passage',?) "
                "ON CONFLICT(src_id,dst_id,kind) DO UPDATE SET weight=excluded.weight",
                (src, dst, atomic),
            )
    rows = conn.execute(
        "SELECT a.fact_id,a.chunk_id,json_extract(a.raw_json,'$.llm_prob'),a.object_mention_id,"
        "1-vec_distance_cosine(sv.embedding,c.embedding),"
        "CASE WHEN a.object_mention_id IS NOT NULL THEN 1-vec_distance_cosine(ov.embedding,c.embedding) END "
        "FROM fact_assertions a JOIN facts f ON f.fact_id=a.fact_id "
        "JOIN entity_mentions s ON s.mention_id=a.subject_mention_id "
        "AND s.chunk_id=a.chunk_id AND s.source_revision=a.source_revision AND s.entity_id=f.subj_id "
        "LEFT JOIN entity_mentions o ON o.mention_id=a.object_mention_id "
        "AND o.chunk_id=a.chunk_id AND o.source_revision=a.source_revision AND o.entity_id=f.obj_id "
        "LEFT JOIN entity_candidate_vectors sv ON sv.sense_id=s.sense_id "
        "LEFT JOIN entity_candidate_vectors ov ON ov.sense_id=o.sense_id "
        "JOIN chunks source ON source.chunk_id=a.chunk_id LEFT JOIN chunk_vec c ON c.chunk_id=a.chunk_id "
        "WHERE (f.obj_kind='entity' AND o.mention_id IS NOT NULL) "
        "OR (f.obj_kind='literal' AND a.object_mention_id IS NULL) "
        "ORDER BY a.fact_id,a.chunk_id,a.assertion_id"
    )
    for (fid, cid), observations in groupby(rows, key=lambda row: tuple(row[:2])):
        probability, coherence, atomic = max(
            (vote(row[2], (row[4], row[5]) if row[3] is not None else (row[4],)) for row in observations),
            key=lambda v: v[2],
        )
        conn.execute(
            "UPDATE fact_sources SET llm_prob=?,vec_coherence=?,atomic_confidence=? WHERE fact_id=? AND chunk_id=?",
            (probability, coherence, atomic, fid, cid),
        )
    rows = conn.execute(
        "SELECT f.subj_id,f.obj_id,fs.chunk_id,fs.llm_prob,fs.vec_coherence,fs.atomic_confidence "
        "FROM facts f JOIN fact_sources fs ON fs.fact_id=f.fact_id WHERE f.obj_kind='entity' "
        "ORDER BY f.subj_id,f.obj_id,fs.chunk_id,f.fact_id"
    )
    for (src, dst, cid), observations in groupby(rows, key=lambda row: tuple(row[:3])):
        probability, coherence, atomic = max((tuple(row[3:]) for row in observations), key=lambda v: v[2])
        conn.execute(
            "UPDATE edge_sources SET llm_prob=?,vec_coherence=?,atomic_confidence=? "
            "WHERE kind='fact' AND src_id=? AND dst_id=? AND chunk_id=?",
            (probability, coherence, atomic, src, dst, cid),
        )
    for table, key, query in (
        ("facts", "fact_id", "SELECT fact_id,atomic_confidence FROM fact_sources ORDER BY fact_id,chunk_id"),
        (
            "entities",
            "entity_id",
            "SELECT dst_id,MAX(atomic_confidence) FROM edge_sources "
            "WHERE kind='passage' AND src_id=chunk_id GROUP BY dst_id,chunk_id ORDER BY dst_id,chunk_id",
        ),
    ):
        for owner, observations in groupby(conn.execute(query), key=lambda row: row[0]):
            probability = bayesian_noisy_or([row[1] for row in observations])
            conn.execute(
                f"UPDATE {table} SET confidence=? WHERE {key}=? AND confidence IS NOT ?",
                (probability, owner, probability),
            )
    rows = conn.execute(
        "SELECT src_id,dst_id,kind,atomic_confidence FROM edge_sources "
        "WHERE kind IN ('passage','fact') ORDER BY src_id,dst_id,kind,chunk_id"
    )
    for edge, observations in groupby(rows, key=lambda row: tuple(row[:3])):
        probability = bayesian_noisy_or([row[3] for row in observations])
        if probability is None:
            raise ValueError("Structural edge has no measured confidence")
        conn.execute(
            "UPDATE edges SET weight=? WHERE src_id=? AND dst_id=? AND kind=? AND weight IS NOT ?",
            (probability, *edge, probability),
        )


def load_sense_bindings(conn: sqlite3.Connection, labels: Iterable[str]) -> dict[str, tuple[str, str, str]]:
    result = {}
    for batch in batched(sorted(set(labels))):
        placeholders = ",".join("?" for _ in batch)
        for row in conn.execute(
            "SELECT DISTINCT m.sense_id,m.entity_id,e.name,e.definition FROM entity_mentions m "
            "JOIN entities e ON e.entity_id=m.entity_id JOIN entity_labels l ON l.entity_id=m.entity_id "
            f"WHERE l.label_norm IN ({placeholders})",
            tuple(batch),
        ):
            value = (row[1], row[2], row[3])
            if row[0] in result and result[row[0]] != value:
                raise ValueError("One resource sense has conflicting canonical bindings")
            result[row[0]] = value
    return result


def canonicalize_fact_definitions(
    conn: sqlite3.Connection,
    prior_definitions: Mapping[str, str] | None = None,
) -> int:
    """Resolve facts.definition deterministically from valid source assertions or valid prior repairs."""
    fact_rows = conn.execute(
        """
        SELECT
            f.fact_id,
            COALESCE(f.definition, ''),
            COALESCE(f.obj_kind, 'entity'),
            LOWER(TRIM(COALESCE(es.definition, ''))),
            LOWER(TRIM(COALESCE(eo.definition, ''))),
            COALESCE(es.name, f.subj_id),
            f.pred,
            COALESCE(eo.name, f.obj_id, ''),
            COALESCE(f.obj_literal_lex, ''),
            COALESCE(f.raw_triple_json, '')
        FROM facts f
        LEFT JOIN entities es ON es.entity_id = f.subj_id
        LEFT JOIN entities eo ON eo.entity_id = f.obj_id
        """
    ).fetchall()
    if not fact_rows:
        return 0

    assertions_by_fact: dict[str, list[tuple[str, str]]] = {}
    for row in conn.execute(
        """
        SELECT fact_id, assertion_id, COALESCE(json_extract(raw_json, '$.spo_def'), '')
        FROM fact_assertions
        WHERE fact_id IS NOT NULL
        ORDER BY assertion_id ASC
        """
    ):
        assertions_by_fact.setdefault(str(row[0]), []).append((str(row[1]), str(row[2])))

    def _is_valid_spo_def(cand: str, subj_def_norm: str, obj_kind: str, obj_def_norm: str) -> bool:
        norm = cand.strip().lower()
        if not norm or norm == subj_def_norm:
            return False
        return not (obj_kind == "entity" and norm == obj_def_norm)

    updates: list[tuple[str, str]] = []
    for row in fact_rows:
        fid = str(row[0])
        current_def = str(row[1])
        obj_kind = str(row[2])
        subj_def_norm = str(row[3])
        obj_def_norm = str(row[4])
        subj_name = str(row[5])
        pred = str(row[6])
        obj_name = str(row[7])
        obj_literal_lex = str(row[8])
        raw_triple_str = str(row[9])

        obj_surface = obj_name if obj_kind == "entity" else obj_literal_lex
        stale_synth_defs: set[str] = set()
        if raw_triple_str.startswith("["):
            try:
                parsed_raw = json.loads(raw_triple_str)
            except ValueError:
                parsed_raw = None
            if isinstance(parsed_raw, list) and len(parsed_raw) == 3:
                raw_s, raw_p, raw_o = str(parsed_raw[0]), str(parsed_raw[1]), str(parsed_raw[2])
                if (raw_s, raw_o) != (subj_name, obj_surface):
                    stale_synth_defs.add(f"{raw_s} {raw_p} {raw_o}.".strip())
                    stale_synth_defs.add(f"{raw_s} {raw_p} {raw_o} (statement).".strip())

        assertions = assertions_by_fact.get(fid, ())
        valid_assertions = [
            spo_def for _, spo_def in assertions if _is_valid_spo_def(spo_def, subj_def_norm, obj_kind, obj_def_norm)
        ]
        if valid_assertions:
            resolved_def = valid_assertions[0]
        else:
            prior_cand = (prior_definitions.get(fid) or current_def) if prior_definitions is not None else current_def
            if prior_cand not in stale_synth_defs and _is_valid_spo_def(
                prior_cand, subj_def_norm, obj_kind, obj_def_norm
            ):
                resolved_def = prior_cand
            elif current_def not in stale_synth_defs and _is_valid_spo_def(
                current_def, subj_def_norm, obj_kind, obj_def_norm
            ):
                resolved_def = current_def
            else:
                synth_def = f"{subj_name} {pred} {obj_surface}.".strip()
                if not _is_valid_spo_def(synth_def, subj_def_norm, obj_kind, obj_def_norm):
                    synth_def = f"{subj_name} {pred} {obj_surface} (statement).".strip()
                resolved_def = synth_def

        if resolved_def != current_def:
            updates.append((resolved_def, fid))

    if updates:
        with write_tx(conn):
            conn.executemany("UPDATE facts SET definition = ? WHERE fact_id = ?", updates)
    return len(updates)


def canonicalize_fact_text(conn: sqlite3.Connection) -> int:
    """Update raw_triple_json and fact_fts only for facts whose canonical surface triple changed."""
    rows = conn.execute(
        "SELECT f.fact_id,s.name,f.pred,f.obj_kind,o.name,f.obj_literal_lex,f.raw_triple_json "
        "FROM facts f JOIN entities s ON s.entity_id=f.subj_id LEFT JOIN entities o ON o.entity_id=f.obj_id"
    ).fetchall()
    updates: list[tuple[str, str]] = []
    fts_deletes: list[tuple[str]] = []
    fts_inserts: list[tuple[str, str]] = []
    for fid, subject, predicate, kind, obj, lexical, current_raw in rows:
        raw = json.dumps([subject, predicate, obj if kind == "entity" else lexical])
        if raw != current_raw:
            updates.append((raw, fid))
            fts_deletes.append((fid,))
            fts_inserts.append((fid, f"{subject} {predicate} {obj or lexical}"))
    if updates:
        with write_tx(conn):
            conn.executemany("UPDATE facts SET raw_triple_json=? WHERE fact_id=?", updates)
            conn.executemany("DELETE FROM fact_fts WHERE fact_id=?", fts_deletes)
            conn.executemany("INSERT INTO fact_fts(fact_id,text) VALUES (?,?)", fts_inserts)
    return len(updates)


@dataclass(frozen=True)
class EntityMerge:
    removed_id: str
    canonical_id: str
    fact_bindings: tuple[tuple[str, str | None], ...] = ()


def bind_merged_evidence(conn: sqlite3.Connection, merge: EntityMerge) -> None:
    """Update canonical bindings only; raw source evidence remains immutable."""
    conn.execute("UPDATE entity_mentions SET entity_id=? WHERE entity_id=?", (merge.canonical_id, merge.removed_id))
    conn.executemany(
        "UPDATE fact_assertions SET fact_id=? WHERE fact_id=?",
        ((new_id, old_id) for old_id, new_id in merge.fact_bindings),
    )
