"""Bounded, shared source evidence for native DSPy synonym judgment."""

import copy
import json
import sqlite3
import time
from collections.abc import Iterator, Sequence
from dataclasses import replace
from itertools import chain
from pathlib import Path
from typing import Any

from yaw.core import SynonymEvidenceBatch

from .chunks import get_chunk_pointers
from .hydration import hydrate_chunk_pointer
from .shacl import inspect_entity_or_fact


class SynonymConstraints:
    """Plan-local identity proofs and separation adjacency; never scored observations."""

    def __init__(self, conn: sqlite3.Connection, fact_pairs: Sequence[tuple[str, str]] = ()) -> None:
        self.parents: dict[str, str] = {}
        self.apart: dict[str, set[str]] = {}
        self.fact_pairs = set(fact_pairs)
        self.fact_pairs.update(conn.execute("SELECT subj_id,obj_id FROM facts WHERE obj_kind='entity'"))
        for a, b in chain(
            self.fact_pairs,
            conn.execute(
                "SELECT src_id,dst_id FROM edges WHERE kind IN ('skos:broader','skos:related','owl:differentFrom')"
            ),
        ):
            if a != b:
                self.apart.setdefault(a, set()).add(b)
                self.apart.setdefault(b, set()).add(a)

    def resolve(self, entity: str) -> str:
        while entity in self.parents:
            parent = self.parents[entity]
            self.parents[entity] = self.parents.get(parent, parent)
            entity = parent
        return entity

    def separated(self, a: str, b: str) -> bool:
        return self.resolve(b) in self.apart.get(self.resolve(a), ())

    def separate(self, a: str, b: str) -> None:
        a, b = self.resolve(a), self.resolve(b)
        if a != b:
            self.apart.setdefault(a, set()).add(b)
            self.apart.setdefault(b, set()).add(a)

    def merge(self, a: str, b: str) -> bool:
        a, b = sorted((self.resolve(a), self.resolve(b)))
        if a == b or b in self.apart.get(a, ()):
            return False
        self.parents[b] = a
        for neighbor in self.apart.pop(b, ()):
            self.apart[neighbor].discard(b)
            self.apart[neighbor].add(a)
            self.apart.setdefault(a, set()).add(neighbor)
        return True

    def prune(self, batch: SynonymEvidenceBatch) -> SynonymEvidenceBatch:
        """Omit collapsed pairs and route established separation past label judging."""
        retained = []
        separated = set()
        for candidate in batch.candidates:
            a, b = candidate["entity_a"], candidate["entity_b"]
            if self.resolve(a) == self.resolve(b):
                continue
            retained.append(candidate["pair_id"])
            if self.separated(a, b):
                separated.add(candidate["pair_id"])
        if len(retained) != len(batch.candidates):
            batch = batch.subset(retained)
        if not separated:
            return batch
        payload = json.loads(batch.payload)
        for pair in payload["pairs"]:
            a, b = pair["pair_id"].split("|", 1)
            if f"{batch.identifiers.get(a, a)}|{batch.identifiers.get(b, b)}" in separated:
                pair["merge_forbidden"] = True
        return replace(batch, payload=json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


def query_shared_role_facts(
    conn: sqlite3.Connection,
    pairs: Sequence[tuple[str, str]],
) -> dict[tuple[str, str], list[list[str]]]:
    """Find facts where entity pairs share a predicate role (Rule W2) in a single batched query."""
    if not pairs:
        return {}
    entity_ids = sorted({e for p in pairs for e in p})
    placeholders = ",".join("?" for _ in entity_ids)
    target_pairs = set(pairs) | {(b, a) for a, b in pairs}

    sql = f"""
    WITH c_facts AS (
        SELECT fact_id, subj_id, pred, obj_id, obj_literal_lex, raw_triple_json
        FROM facts
        WHERE subj_id IN ({placeholders}) OR obj_id IN ({placeholders})
    )
    SELECT
        f1.subj_id, f1.obj_id, f1.raw_triple_json,
        f2.subj_id, f2.obj_id, f2.raw_triple_json,
        1 AS role_type
    FROM c_facts f1
    JOIN c_facts f2 ON f1.pred = f2.pred
        AND f1.fact_id != f2.fact_id
        AND f1.subj_id != f2.subj_id
        AND (
            (f1.obj_id IS NOT NULL AND f1.obj_id = f2.obj_id AND f1.obj_id NOT IN (f1.subj_id, f2.subj_id))
            OR
            (f1.obj_literal_lex IS NOT NULL AND f1.obj_literal_lex = f2.obj_literal_lex)
        )
    UNION ALL
    SELECT
        f1.subj_id, f1.obj_id, f1.raw_triple_json,
        f2.subj_id, f2.obj_id, f2.raw_triple_json,
        2 AS role_type
    FROM c_facts f1
    JOIN c_facts f2 ON f1.pred = f2.pred
        AND f1.fact_id != f2.fact_id
        AND f1.subj_id = f2.subj_id
        AND f1.subj_id NOT IN (f1.obj_id, f2.obj_id)
    WHERE f1.obj_id IS NOT NULL AND f2.obj_id IS NOT NULL AND f1.obj_id != f2.obj_id
    """
    res: dict[tuple[str, str], list[list[str]]] = {}
    rows = conn.execute(sql, (*entity_ids, *entity_ids)).fetchall()
    for r in rows:
        s1, o1, t1_json, s2, o2, t2_json = r[0], r[1], r[2], r[3], r[4], r[5]
        for pair in ((str(s1), str(s2)), (str(o1), str(o2))):
            if pair in target_pairs and pair not in res:
                t1 = json.loads(t1_json)
                t2 = json.loads(t2_json)
                res[pair] = [t1, t2]
                res[(pair[1], pair[0])] = [t2, t1]
    return res


class _Packet:
    def __init__(
        self,
        conn: sqlite3.Connection,
        root: Path,
        hops: int = 0,
        *,
        shared_role_facts_map: dict[tuple[str, str], list[list[str]]] | None = None,
    ) -> None:
        self.conn, self.root, self.hops = conn, root, hops
        self.candidates: list[dict[str, Any]] = []
        self.identifiers: dict[str, str] = {}
        self.aliases: dict[tuple[str, str], str] = {}
        self.infos: dict[str, dict[str, Any]] = {}
        self.pairs: list[dict[str, Any]] = []
        self.entities: dict[str, dict[str, Any]] = {}
        self.sources: dict[str, dict[str, Any]] = {}
        self.source_aliases: dict[tuple[str, int, int, str], str] = {}
        self.shared_role_facts_map = shared_role_facts_map

    def fork(self) -> _Packet:
        other = copy.copy(self)
        other.identifiers = dict(self.identifiers)
        other.aliases = dict(self.aliases)
        other.infos = dict(self.infos)
        other.entities = dict(self.entities)
        other.sources = dict(self.sources)
        other.source_aliases = dict(self.source_aliases)
        other.candidates = list(self.candidates)
        other.pairs = list(self.pairs)
        return other

    def ref(self, kind: str, original: str) -> str:
        key = (kind, original)
        if key not in self.aliases:
            alias = f"{kind}{sum(k == kind for k, _ in self.aliases)}"
            self.aliases[key] = alias
            self.identifiers[alias] = original
        return self.aliases[key]

    def build_entity_dict(self, eid: str) -> dict[str, Any]:
        info = self.infos.setdefault(eid, {})
        if not info:
            info.update(inspect_entity_or_fact(self.conn, eid, self.root))
        ent = info["entity"]
        if ent is None:
            raise ValueError(f"Missing synonym candidate entity {eid}")

        missing_chunks = set(info["chunk_ids"]) - {cm["chunk_id"] for cm in info["chunks_meta"]}
        if missing_chunks:
            raise ValueError(f"Missing source pointers for {eid}: {sorted(missing_chunks)}")

        pref_labels = sorted({str(x["raw_label"]) for x in info["labels"] if x["label_kind"] == "skos:prefLabel"})
        if not pref_labels:
            pref_labels = [str(ent["name"])]
        alt_labels = sorted({str(x["raw_label"]) for x in info["labels"] if x["label_kind"] == "skos:altLabel"})
        hidden_labels = sorted({str(x["raw_label"]) for x in info["labels"] if x["label_kind"] == "skos:hiddenLabel"})
        facts = [{"triple": list(f["raw_triple"]), "definition": str(f["definition"])} for f in info["facts"]]
        source_refs = []
        for cm in info["chunks_meta"]:
            cid = cm["chunk_id"]
            pointer = get_chunk_pointers(self.conn, [cid]).get(cid)
            if pointer is None:
                raise ValueError(f"Missing source pointer {cid}")
            key = (pointer.rel_posix_path, pointer.start_line, pointer.end_line, pointer.title)
            hp = hydrate_chunk_pointer(pointer, self.root, max_chars=0)
            if hp.stale_on_disk:
                raise ValueError(f"Stale source pointer {cid}")
            if key not in self.source_aliases:
                ref = f"s{len(self.source_aliases)}"
                self.source_aliases[key] = ref
                self.sources[ref] = {
                    "path": pointer.rel_posix_path,
                    "title": pointer.title,
                    "start_line": pointer.start_line,
                    "end_line": pointer.end_line,
                    "text": hp.text,
                }
            ref = self.source_aliases[key]
            if ref not in source_refs:
                source_refs.append(ref)
        return {
            "name": str(ent["name"]),
            "definition": str(ent["definition"]),
            "labels": {"skos:prefLabel": pref_labels, "skos:altLabel": alt_labels, "skos:hiddenLabel": hidden_labels},
            "facts": facts,
            "sources": source_refs,
        }

    def add(self, candidate: dict[str, Any]) -> None:
        ea, eb = str(candidate["entity_a"]), str(candidate["entity_b"])
        a = self.ref("e", ea)
        b = self.ref("e", eb)
        if a not in self.entities:
            self.entities[a] = self.build_entity_dict(ea)
        if b not in self.entities:
            self.entities[b] = self.build_entity_dict(eb)
        ia, ib = self.infos[ea], self.infos[eb]

        def docs(info: dict[str, Any]) -> set[str]:
            return {cm["rel_posix_path"] for cm in info["chunks_meta"]}

        def islands(info: dict[str, Any]) -> set[tuple[str, int, int]]:
            return {
                (
                    cm["rel_posix_path"],
                    cm["parent_ast_start"] or cm["start_line"],
                    cm["parent_ast_end"] or cm["end_line"],
                )
                for cm in info["chunks_meta"]
            }

        shared_docs = sorted(docs(ia) & docs(ib))
        shared_islands = [list(x) for x in sorted(islands(ia) & islands(ib))]

        direct_facts: list[list[str]] = []
        seen_direct: set[tuple[str, str, str]] = set()
        for f in ia.get("facts", []):
            if {f.get("subj_id"), f.get("obj_id")} == {ea, eb}:
                triple = tuple(f["raw_triple"])
                if triple not in seen_direct:
                    seen_direct.add(triple)
                    direct_facts.append(list(triple))

        if self.shared_role_facts_map is None:
            self.shared_role_facts_map = query_shared_role_facts(self.conn, [(ea, eb)])

        shared_role = self.shared_role_facts_map.get((ea, eb), [])

        self.pairs.append(
            {
                "pair_id": f"{a}|{b}",
                "dense_sim": float(candidate.get("dense_sim", 0.0)),
                "lexical_sim": float(candidate.get("lexical_sim", 0.0)),
                "shared_documents": shared_docs,
                "shared_islands": shared_islands,
                "direct_facts": direct_facts,
                "shared_role_facts": shared_role,
                "entity_a": a,
                "entity_b": b,
            }
        )
        self.candidates.append(candidate)

    def to_json(self) -> str:
        return json.dumps(
            {
                "format": "synonym-evidence-v3",
                "pairs": self.pairs,
                "entities": self.entities,
                "sources": self.sources,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def batch(self, elapsed: float) -> SynonymEvidenceBatch:
        return SynonymEvidenceBatch(tuple(self.candidates), self.to_json(), dict(self.identifiers), elapsed)


def iter_semantic_evidence(
    conn: sqlite3.Connection,
    candidates: Sequence[dict[str, Any]],
    corpus_root: Path,
    *,
    ast_context_hops: int = 0,
    batch_size: int = 6,
    max_chars: int = 40000,
) -> Iterator[SynonymEvidenceBatch]:
    """Yield complete evidence packets, splitting whole pairs without dropping evidence."""
    if batch_size < 1 or max_chars < 1:
        raise ValueError("Synonym packet limits must be positive")
    pairs = [(str(c["entity_a"]), str(c["entity_b"])) for c in candidates]
    shared_role_facts_map = query_shared_role_facts(conn, pairs)

    packet = _Packet(
        conn,
        corpus_root,
        ast_context_hops,
        shared_role_facts_map=shared_role_facts_map,
    )
    started = time.perf_counter()

    for candidate in candidates:
        trial = packet.fork()
        trial.add(candidate)
        if packet.candidates and (len(trial.candidates) > batch_size or len(trial.batch(0).payload) > max_chars):
            yield packet.batch(time.perf_counter() - started)
            packet = _Packet(
                conn,
                corpus_root,
                ast_context_hops,
                shared_role_facts_map=shared_role_facts_map,
            )
            packet.infos = {eid: trial.infos[eid] for eid in (candidate["entity_a"], candidate["entity_b"])}
            started = time.perf_counter()
            packet.add(candidate)
        else:
            packet = trial
        if len(packet.batch(0).payload) > max_chars:
            raise ValueError(
                f"Complete disambiguation evidence for {candidate['entity_a']}|{candidate['entity_b']} "
                f"exceeds {max_chars} characters"
            )
    if packet.candidates:
        yield packet.batch(time.perf_counter() - started)
