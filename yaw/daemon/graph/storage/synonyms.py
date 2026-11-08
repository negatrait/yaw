"""Bounded, shared source evidence for native DSPy synonym judgment."""

import copy
import json
import sqlite3
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

from yaw.core import SynonymEvidenceBatch

from .chunks import get_chunk_pointers
from .hydration import hydrate_chunk_pointer
from .shacl import inspect_entity_or_fact


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
        self.source_texts: dict[str, str] = {}
        self.shared_role_facts_map = shared_role_facts_map

    def fork(self) -> _Packet:
        other = copy.copy(self)
        other.identifiers = dict(self.identifiers)
        other.aliases = dict(self.aliases)
        other.infos = dict(self.infos)
        other.source_texts = dict(self.source_texts)
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

        pref_labels = sorted({str(x["raw_label"]) for x in info["labels"] if x["label_kind"] == "skos:prefLabel"})[:4]
        if not pref_labels:
            pref_labels = [str(ent["name"])]
        alt_labels = sorted({str(x["raw_label"]) for x in info["labels"] if x["label_kind"] == "skos:altLabel"})[:8]

        facts = [list(f["raw_triple"]) for f in info["facts"][:6]]

        sources = []
        for cm in info["chunks_meta"][:2]:
            cid = cm["chunk_id"]
            if cid not in self.source_texts:
                ptr_map = get_chunk_pointers(self.conn, [cid])
                pointer = ptr_map.get(cid)
                if pointer is not None:
                    hp = hydrate_chunk_pointer(pointer, self.root, max_chars=900)
                    body = "\n".join(hp.text.strip().splitlines()[:14]) if not hp.stale_on_disk else ""
                else:
                    body = ""
                self.source_texts[cid] = body
            sources.append(
                {
                    "path": cm["rel_posix_path"],
                    "title": cm["title"],
                    "text": self.source_texts[cid],
                }
            )

        return {
            "name": str(ent["name"]),
            "definition": str(ent["definition"]),
            "type": "skos:Concept",
            "labels": {
                "skos:prefLabel": pref_labels,
                "skos:altLabel": alt_labels,
            },
            "facts": facts,
            "sources": sources,
        }

    def add(self, candidate: dict[str, Any]) -> None:
        ea, eb = str(candidate["entity_a"]), str(candidate["entity_b"])
        a = self.ref("e", ea)
        b = self.ref("e", eb)
        ent_a = self.build_entity_dict(ea)
        ent_b = self.build_entity_dict(eb)
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

        shared_docs = sorted(docs(ia) & docs(ib))[:2]
        shared_islands = [list(x) for x in sorted(islands(ia) & islands(ib))[:2]]

        direct_facts: list[list[str]] = []
        seen_direct: set[tuple[str, str, str]] = set()
        for f in ia.get("facts", []):
            if {f.get("subj_id"), f.get("obj_id")} == {ea, eb}:
                triple = tuple(f["raw_triple"])
                if triple not in seen_direct:
                    seen_direct.add(triple)
                    direct_facts.append(list(triple))
                    if len(direct_facts) >= 2:
                        break

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
                "entity_a": ent_a,
                "entity_b": ent_b,
            }
        )
        self.candidates.append(candidate)

    def to_json(self) -> str:
        return json.dumps(
            {
                "format": "synonym-evidence-v2",
                "pairs": self.pairs,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def batch(self, elapsed: float) -> SynonymEvidenceBatch:
        return SynonymEvidenceBatch(tuple(self.candidates), self.to_json(), dict(self.identifiers), elapsed)


def iter_synonym_evidence(
    conn: sqlite3.Connection,
    candidates: Sequence[dict[str, Any]],
    corpus_root: Path,
    *,
    ast_context_hops: int = 0,
    batch_size: int = 6,
    max_chars: int = 40000,
    fits: Callable[[SynonymEvidenceBatch], bool] | None = None,
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

    def emit(current: _Packet, elapsed: float) -> Iterator[SynonymEvidenceBatch]:
        batch = current.batch(elapsed)
        if fits is None or fits(batch):
            yield batch
        elif len(current.candidates) == 1:
            raise ValueError(f"Complete synonym evidence exceeds context budget for {current.candidates[0]['pair_id']}")
        else:
            middle = len(current.candidates) // 2
            for part in (current.candidates[:middle], current.candidates[middle:]):
                child_started = time.perf_counter()
                child = _Packet(
                    conn,
                    corpus_root,
                    ast_context_hops,
                    shared_role_facts_map=shared_role_facts_map,
                )
                child.infos = {
                    eid: info
                    for eid, info in current.infos.items()
                    if any(eid in (c["entity_a"], c["entity_b"]) for c in part)
                }
                child.source_texts = dict(current.source_texts)
                for candidate in part:
                    child.add(candidate)
                yield from emit(
                    child, elapsed * len(part) / len(current.candidates) + time.perf_counter() - child_started
                )

    for candidate in candidates:
        trial = packet.fork()
        trial.add(candidate)
        if packet.candidates and (len(trial.candidates) > batch_size or len(trial.batch(0).payload) > max_chars):
            yield from emit(packet, time.perf_counter() - started)
            packet = _Packet(
                conn,
                corpus_root,
                ast_context_hops,
                shared_role_facts_map=shared_role_facts_map,
            )
            packet.infos = {eid: trial.infos[eid] for eid in (candidate["entity_a"], candidate["entity_b"])}
            packet.source_texts = dict(trial.source_texts)
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
        yield from emit(packet, time.perf_counter() - started)
