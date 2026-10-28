"""Bounded, shared source evidence for native DSPy synonym judgment."""

import copy
import json
import sqlite3
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

from yaw.core import SynonymEvidenceBatch

from .db import YawConnection
from .graph_ops import get_chunk_pointers
from .hydration import hydrate_ast_parent_context, hydrate_chunk_pointer
from .shacl import inspect_entity_or_fact


class _Packet:
    def __init__(self, conn: sqlite3.Connection | YawConnection, root: Path, hops: int) -> None:
        self.conn, self.root, self.hops = conn, root, hops
        self.candidates: list[dict[str, Any]] = []
        self.identifiers: dict[str, str] = {}
        self.aliases: dict[tuple[str, str], str] = {}
        self.infos: dict[str, dict[str, Any]] = {}
        self.pairs: list[dict[str, Any]] = []
        self.entities: dict[str, Any] = {}
        self.facts: dict[str, Any] = {}
        self.sources: dict[str, Any] = {}
        self.spans: dict[tuple[Any, ...], str] = {}

    def fork(self) -> _Packet:
        other = copy.copy(self)
        for name in ("identifiers", "aliases", "infos", "entities", "facts", "sources", "spans"):
            setattr(other, name, dict(getattr(self, name)))
        other.candidates, other.pairs = list(self.candidates), list(self.pairs)
        return other

    def ref(self, kind: str, original: str) -> str:
        key = (kind, original)
        if key not in self.aliases:
            alias = f"{kind}{sum(k == kind for k, _ in self.aliases)}"
            self.aliases[key] = alias
            self.identifiers[alias] = original
        return self.aliases[key]

    def fact(self, fact: dict[str, Any]) -> str:
        alias = self.ref("f", fact["fact_id"])
        if alias not in self.facts:
            self.facts[alias] = {
                "triple": fact["raw_triple"],
                "definition": fact["definition"],
                "property_type": fact["property_type"],
                **(
                    {"datatype": fact["obj_datatype"], "language": fact["obj_lang"]}
                    if fact["obj_kind"] == "literal"
                    else {}
                ),
            }
        return alias

    def source(self, cm: dict[str, Any]) -> dict[str, str]:
        chunk = self.ref("c", cm["chunk_id"])
        span = tuple(
            cm[k]
            for k in (
                "rel_posix_path",
                "start_line",
                "end_line",
                "title",
                "ast_island_start",
                "ast_island_end",
                "parent_ast_start",
                "parent_ast_end",
                "parent_ast_title",
            )
        )
        if span not in self.spans:
            pointer = get_chunk_pointers(self.conn, [cm["chunk_id"]])[cm["chunk_id"]]
            hp = hydrate_chunk_pointer(pointer, self.root, max_chars=900)
            parent = hydrate_ast_parent_context(pointer, self.root, hops=self.hops, max_chars=450) if self.hops else ""
            body = "\n".join(hp.text.strip().splitlines()[:14]) if not hp.stale_on_disk else ""
            parent = "\n".join(parent.splitlines()[:6])
            # The physical span, AST context and exact rendered text define a source.
            key = (*span, parent, body)
            alias = self.spans.get(key, f"s{len(self.sources)}")
            self.spans[key] = self.spans[span] = alias
            self.sources[alias] = {
                "path": cm["rel_posix_path"],
                "lines": [cm["start_line"], cm["end_line"]],
                "ast": [cm["ast_island_start"], cm["ast_island_end"]],
                "title": cm["title"],
                "parent": parent,
                "text": body,
                "available": bool(parent or body),
            }
        return {"source": self.spans[span], "chunk": chunk}

    def entity(self, eid: str) -> str:
        alias = self.ref("e", eid)
        if alias in self.entities:
            return alias
        info = self.infos.setdefault(eid, {})
        if not info:
            info.update(inspect_entity_or_fact(self.conn, eid, self.root))
        ent = info["entity"]
        if ent is None:
            raise ValueError(f"Missing synonym candidate entity {eid}")
        facts = info["facts"]
        literal = [f for f in facts if f["obj_id"] == eid and f["obj_kind"] == "literal"]
        is_literal = bool(literal) and not any(
            f["subj_id"] == eid or (f["obj_id"] == eid and f["obj_kind"] == "entity") for f in facts
        )
        labels = {}
        for kind, limit in (("skos:prefLabel", 4), ("skos:altLabel", 8), ("skos:hiddenLabel", 6)):
            labels[kind] = sorted(
                {
                    (x["raw_label"], x["label_norm"]) if kind == "skos:prefLabel" else (x["raw_label"],)
                    for x in info["labels"]
                    if x["label_kind"] == kind
                }
            )[:limit]
        if not labels["skos:prefLabel"]:
            labels["skos:prefLabel"] = [(ent["name"],)]
        links: dict[str, list[Any]] = {}
        for edge in info["edges"]:
            kind = edge["kind"]
            if kind == "skos:broader" and edge["dst_id"] == eid:
                kind = "skos:narrower"
            elif edge["src_id"] != eid:
                continue
            if kind not in ("skos:broader", "skos:narrower", "skos:related", "owl:differentFrom"):
                continue
            target = edge["neighbor_id"]
            entry = [self.ref("e", target), edge["neighbor_name"] or target]
            if kind == "owl:differentFrom" and edge["neighbor_definition"]:
                entry.append(edge["neighbor_definition"].strip()[:80])
            values = links.setdefault(kind, [])
            if entry not in values and len(values) < (6 if kind == "skos:related" else 5):
                values.append(entry)
        self.entities[alias] = {
            "name": ent["name"],
            "definition": ent["definition"],
            "type": "rdfs:Literal" if is_literal else "skos:Concept",
            "labels": labels,
            "relations": links,
            "facts": [self.fact(f) for f in facts[:6]],
            "sources": [self.source(cm) for cm in info["chunks_meta"][:2]],
            **(
                {
                    "datatypes": sorted({f["obj_datatype"] for f in literal if f["obj_datatype"]}),
                    "languages": sorted({f["obj_lang"] for f in literal if f["obj_lang"]}),
                }
                if is_literal
                else {}
            ),
        }
        return alias

    def add(self, candidate: dict[str, Any]) -> None:
        ea, eb = str(candidate["entity_a"]), str(candidate["entity_b"])
        a, b = self.entity(ea), self.entity(eb)
        ia, ib = self.infos[ea], self.infos[eb]

        def docs(info: dict[str, Any]) -> set[str]:
            return {cm["rel_posix_path"] for cm in info["chunks_meta"]}

        def islands(info: dict[str, Any]) -> set[tuple[str, int, int]]:
            return {(cm["rel_posix_path"], cm["parent_ast_start"], cm["parent_ast_end"]) for cm in info["chunks_meta"]}

        direct = [f for f in ia["facts"] if {f["subj_id"], f["obj_id"]} == {ea, eb}][:2]
        self.pairs.append(
            {
                "pair_id": f"{a}|{b}",
                "dense_sim": float(candidate["dense_sim"]),
                "lexical_sim": float(candidate["lexical_sim"]),
                "shared_documents": sorted(docs(ia) & docs(ib))[:2],
                "shared_islands": sorted(islands(ia) & islands(ib))[:2],
                "direct_facts": [self.fact(f) for f in direct],
            }
        )
        self.candidates.append(candidate)

    def batch(self, elapsed: float) -> SynonymEvidenceBatch:
        payload = json.dumps(
            {
                "format": "synonym-evidence-v1",
                "pairs": self.pairs,
                "entities": self.entities,
                "facts": self.facts,
                "sources": self.sources,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return SynonymEvidenceBatch(tuple(self.candidates), payload, dict(self.identifiers), elapsed)


def iter_synonym_evidence(
    conn: sqlite3.Connection | YawConnection,
    candidates: Sequence[dict[str, Any]],
    corpus_root: Path,
    *,
    ast_context_hops: int = 0,
    batch_size: int = 16,
    max_chars: int = 40000,
    fits: Callable[[SynonymEvidenceBatch], bool] | None = None,
) -> Iterator[SynonymEvidenceBatch]:
    """Yield complete evidence packets, splitting whole pairs without dropping evidence."""
    if batch_size < 1 or max_chars < 1:
        raise ValueError("Synonym packet limits must be positive")
    packet = _Packet(conn, corpus_root, ast_context_hops)
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
                child = _Packet(conn, corpus_root, ast_context_hops)
                child.infos = {
                    eid: info
                    for eid, info in current.infos.items()
                    if any(eid in (c["entity_a"], c["entity_b"]) for c in part)
                }
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
            packet = _Packet(conn, corpus_root, ast_context_hops)
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
        yield from emit(packet, time.perf_counter() - started)
