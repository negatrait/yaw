"""Tests verifying graph build-time storage invariants, guards, and edge direction."""

import json
from pathlib import Path
from typing import Any

import dspy
import numpy as np
import pytest

from daemon.graph.synonym_protocol import candidate_rows, complete_decisions
from yaw.core import ChunkPointer, OpenIERecord, RelationTraversalPolicy, StructuredTriple
from yaw.daemon.graph.engine import YawGraphEngine
from yaw.daemon.graph.inference import HippoRAGProgram, parse_synonym_decisions
from yaw.daemon.graph.ppr import build_transition_matrix
from yaw.daemon.graph.storage import (
    delete_chunks,
    lint_graph_shacl,
    load_graph_topology,
    merge_entities,
    upsert_semantic_edges,
)


class DeterministicMockLM(dspy.LM):
    def __init__(
        self,
        responses: list[str] | None = None,
        *,
        synonym_fn: Any = None,
    ) -> None:
        super().__init__(model="mock-lm")
        self._responses = list(responses or ['{"decisions": []}'])
        self._idx = 0
        self._ner_json: str | None = None
        self._triples_json: str | None = None
        self._synonym_fn = synonym_fn
        self.observed_candidates: list[dict[str, Any]] = []

    def set_stage_responses(self, ner_json: str, triples_json: str) -> None:
        self._ner_json = ner_json
        self._triples_json = triples_json

    @staticmethod
    def _extract_prompt_text(
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> str:
        parts: list[str] = []
        if prompt:
            parts.append(str(prompt))
        if messages:
            for m in messages:
                parts.append(str(m.get("content", "")))
        return "\n".join(parts)

    @staticmethod
    def _parse_candidate_pairs_from_text(text: str) -> list[dict[str, Any]]:
        return candidate_rows(text)

    def __call__(
        self, prompt: str | None = None, messages: list[dict[str, Any]] | None = None, **kwargs: Any
    ) -> list[str]:
        full_text = self._extract_prompt_text(prompt, messages)
        if "candidate_pairs" in full_text or '"pair_id"' in full_text:
            cands = self._parse_candidate_pairs_from_text(full_text)
            self.observed_candidates.extend(cands)
            decisions = complete_decisions(cands, self._synonym_fn(cands) if self._synonym_fn is not None else [])
            return [json.dumps({"decisions": decisions})]
        if "triples" in full_text and "named_entities" in full_text and self._triples_json is not None:
            return [self._triples_json]
        if "named_entities" in full_text and self._ner_json is not None:
            return [self._ner_json]
        out = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return [out]


class DeterministicMockEmbedder:
    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def encode(self, texts: Any, instruction: str = "", **kwargs: Any) -> np.ndarray:
        raw = np.ones((len(texts), self.dim), dtype=np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        return (raw / norms).astype(np.float32)

    def __call__(self, texts: list[str]) -> list[list[float]]:
        return self.encode(texts).tolist()


def _make_engine(
    synonym_token_counter,
    tmp_path: Path,
    db_name: str,
    synonym_callback: Any = None,
) -> tuple[YawGraphEngine, Any]:
    db_path = tmp_path / db_name
    mock_lm = DeterministicMockLM(synonym_fn=synonym_callback)
    dspy.settings.configure(lm=mock_lm)
    program = HippoRAGProgram(
        synonym_token_counter=synonym_token_counter, enable_gbnf=False, orchestrator_lm=mock_lm, rlm_sub_lm=mock_lm
    )
    embedder: Any = DeterministicMockEmbedder(dim=64)
    engine = YawGraphEngine(
        db_path=db_path,
        workspace_root=tmp_path,
        program=program,
        embedder=embedder,
    )
    return engine, engine.conn


def _index_chunk_with_record(
    engine: YawGraphEngine,
    chunk: ChunkPointer,
    text: str,
    record: OpenIERecord,
) -> None:
    ner_json = json.dumps(
        {
            "named_entities": [
                [str(e[0]), str(e[1])] if isinstance(e, (list, tuple)) and len(e) >= 2 else [str(e), ""]
                for e in record.entities
            ]
        }
    )
    triples_json = json.dumps(
        {
            "triples": [
                {
                    "sub": [st.subj, st.subj_def],
                    "pred": [
                        st.pred,
                        "owl:DatatypeProperty" if st.obj_kind == "literal" else "owl:ObjectProperty",
                    ],
                    "obj": [st.obj, st.obj_def],
                    "def": st.spo_def or f"{st.subj} {st.pred} {st.obj}.",
                }
                for st in record.structured_triples
            ]
        }
    )
    mock_lm = engine.program.rlm_sub_lm
    assert isinstance(mock_lm, DeterministicMockLM)
    mock_lm.set_stage_responses(ner_json, triples_json)
    engine.index_chunks([(chunk, text)])


def test_datatype_property_literal_objects_stored_with_weighted_edges_and_excluded_from_skos(
    synonym_token_counter,
    tmp_path: Path,
) -> None:
    """Verify owl:DatatypeProperty literal objects are materialized in entities with contextual definitions,
    connected via 0.5-weighted forward fact edges and passage edges, and excluded from SKOS disambiguation."""
    # First verify HippoRAGProgram.extract_structured_triples preserves obj_def on obj_kind == 'literal'
    lm_triple_json = json.dumps(
        {
            "triples": [
                {
                    "sub": ["YAW Service", "A high performance knowledge graph daemon"],
                    "pred": ["listens on port", "owl:DatatypeProperty"],
                    "obj": ["8080", "Default HTTP listen port for YAW Service"],
                    "def": "YAW Service listens on port 8080.",
                }
            ]
        }
    )
    dspy.settings.configure(lm=DeterministicMockLM([lm_triple_json]))
    prog = HippoRAGProgram(synonym_token_counter=synonym_token_counter, enable_gbnf=False)
    extracted_sts = prog.extract_structured_triples(
        "YAW Service listens on port 8080.",
        [
            ("YAW Service", "A high performance knowledge graph daemon"),
            ("8080", "Default HTTP listen port for YAW Service"),
        ],
    )
    assert len(extracted_sts) == 1
    assert extracted_sts[0].obj_kind == "literal"
    assert extracted_sts[0].obj_def == "Default HTTP listen port for YAW Service"

    called_candidates: list[dict[str, Any]] = []

    def _judge(candidates: list[dict[str, Any]]) -> list[dict[str, str]]:
        called_candidates.extend(candidates)
        return [
            {
                "pair_id": c["pair_id"],
                "relation": "skos:related",
                "canonical": "A",
            }
            for c in candidates
        ]

    engine, conn = _make_engine(synonym_token_counter, tmp_path, "test_lit_edges.db", synonym_callback=_judge)

    chunk = ChunkPointer(
        chunk_id="chunk-1",
        rel_posix_path="doc1.md",
        start_line=1,
        end_line=5,
        title="Server Config",
    )
    record = OpenIERecord(
        chunk_id="chunk-1",
        entities=[
            ("YAW Service", "A high performance knowledge graph daemon"),
            ("8080", "Default HTTP listen port for YAW Service"),
            ("$1.2B", "Annual division revenue"),
        ],
        triples=[
            ("YAW Service", "listens on port", "8080"),
            ("YAW Service", "generated revenue", "$1.2B"),
        ],
        structured_triples=[
            StructuredTriple(
                subj="YAW Service",
                pred="listens on port",
                obj="8080",
                obj_kind="literal",
                obj_datatype="rdfs:Literal",
                subj_def="A high performance knowledge graph daemon",
                obj_def="Default HTTP listen port for YAW Service",
            ),
            StructuredTriple(
                subj="YAW Service",
                pred="generated revenue",
                obj="$1.2B",
                obj_kind="literal",
                obj_datatype="rdfs:Literal",
                subj_def="A high performance knowledge graph daemon",
                obj_def="Annual division revenue",
            ),
        ],
    )

    _index_chunk_with_record(engine, chunk, "YAW Service listens on port 8080 and generated $1.2B.", record)

    # Verify both 8080 and non-scalar $1.2B are stored as literal nodes with no duplicate '1 2b' concept node
    all_entities = {
        r[1]: (r[0], r[2]) for r in conn.execute("SELECT entity_id, name, definition FROM entities").fetchall()
    }
    assert "8080" in all_entities
    assert "$1.2B" in all_entities
    assert "1 2b" not in all_entities
    lit_id = all_entities["8080"][0]
    rev_id = all_entities["$1.2B"][0]
    assert "Default HTTP listen port" in all_entities["8080"][1]
    assert "Annual division revenue" in all_entities["$1.2B"][1]

    # Verify facts table contains both literal facts with obj_id and obj_literal_lex
    facts = conn.execute(
        "SELECT fact_id, subj_id, pred, obj_id, obj_kind, obj_literal_lex, obj_datatype FROM facts ORDER BY pred"
    ).fetchall()
    assert len(facts) == 2
    subj_id = facts[0][1]
    assert {f[3] for f in facts} == {lit_id, rev_id}
    assert all(f[4] == "literal" for f in facts)

    # Verify edges contains directed forward fact edges (subj_id -> lit_id, subj_id -> rev_id) with weight 0.5
    fact_edges = {
        (r[0], r[1]): float(r[3])
        for r in conn.execute("SELECT src_id, dst_id, kind, weight FROM edges WHERE kind = 'fact'").fetchall()
    }
    assert fact_edges == {(subj_id, lit_id): 0.5, (subj_id, rev_id): 0.5}

    # Verify passage edges connect chunk-1 to subj_id, lit_id, and rev_id
    passage_edges = conn.execute("SELECT src_id, dst_id FROM edges WHERE kind = 'passage'").fetchall()
    passage_pairs = {(r[0], r[1]) for r in passage_edges}
    assert ("chunk-1", subj_id) in passage_pairs
    assert ("chunk-1", lit_id) in passage_pairs
    assert ("chunk-1", rev_id) in passage_pairs

    # Verify pure literals '8080' and '$1.2B' were excluded from SKOS disambiguation and have zero skos:* edges
    assert len(called_candidates) == 0
    skos_edges = conn.execute(
        "SELECT src_id, dst_id, kind FROM edges WHERE kind IN ('skos:broader', 'skos:related')"
    ).fetchall()
    assert len(skos_edges) == 0

    # Verify even direct storage calls to upsert_semantic_edges reject pure literal endpoints
    upsert_semantic_edges(
        conn,
        [
            (subj_id, lit_id, "skos:related", 1.0),
            (subj_id, rev_id, "skos:broader", 1.0),
        ],
    )
    skos_after = conn.execute(
        "SELECT src_id, dst_id, kind FROM edges WHERE kind IN ('skos:broader', 'skos:related')"
    ).fetchall()
    assert len(skos_after) == 0


def test_dual_role_literal_promoted_to_concept_and_weight_preserved_across_merge_and_delete(
    synonym_token_counter,
    tmp_path: Path,
) -> None:
    """Verify a surface form used first as a literal and later as an ObjectProperty endpoint is promoted
    to a skos:Concept, and merge_entities/delete_chunks preserve 0.5 weights on pure literal fact edges."""
    engine, conn = _make_engine(synonym_token_counter, tmp_path, "test_dual_role.db")

    c1 = ChunkPointer("c1", "doc1.md", 1, 5, "Rollout")
    r1 = OpenIERecord(
        chunk_id="c1",
        entities=[
            ("Gateway Alpha", "Primary API gateway"),
            ("Gateway Beta", "Secondary API gateway alias"),
            ("8080", "Listen port"),
            ("canary", "Rollout mode string"),
        ],
        triples=[
            ("Gateway Alpha", "uses port", "8080"),
            ("Gateway Beta", "uses port", "8080"),
            ("Gateway Alpha", "rollout mode", "canary"),
        ],
        structured_triples=[
            StructuredTriple(
                subj="Gateway Alpha",
                pred="uses port",
                obj="8080",
                obj_kind="literal",
                obj_datatype="rdfs:Literal",
                subj_def="Primary API gateway",
                obj_def="Listen port",
            ),
            StructuredTriple(
                subj="Gateway Beta",
                pred="uses port",
                obj="8080",
                obj_kind="literal",
                obj_datatype="rdfs:Literal",
                subj_def="Secondary API gateway alias",
                obj_def="Listen port",
            ),
            StructuredTriple(
                subj="Gateway Alpha",
                pred="rollout mode",
                obj="canary",
                obj_kind="literal",
                obj_datatype="rdfs:Literal",
                subj_def="Primary API gateway",
                obj_def="Rollout mode string",
            ),
        ],
    )
    c2 = ChunkPointer("c2", "doc2.md", 1, 5, "Backup")
    r2 = OpenIERecord(
        chunk_id="c2",
        entities=[
            ("Gateway Alpha", "Primary API gateway"),
            ("8080", "Listen port"),
        ],
        triples=[("Gateway Alpha", "fallback port", "8080")],
        structured_triples=[
            StructuredTriple(
                subj="Gateway Alpha",
                pred="fallback port",
                obj="8080",
                obj_kind="literal",
                obj_datatype="rdfs:Literal",
                subj_def="Primary API gateway",
                obj_def="Listen port",
            )
        ],
    )
    _index_chunk_with_record(engine, c1, "Gateway Alpha and Beta use port 8080 in canary mode.", r1)
    _index_chunk_with_record(engine, c2, "Gateway Alpha fallback port 8080.", r2)

    ent_rows = {r[0].lower(): r[1] for r in conn.execute("SELECT name, entity_id FROM entities").fetchall()}
    alpha_id = ent_rows["gateway alpha"]
    beta_id = ent_rows["gateway beta"]
    lit_id = ent_rows["8080"]
    canary_id = ent_rows["canary"]

    # While 'canary' only participates in obj_kind='literal' facts, it cannot participate in SKOS edges
    upsert_semantic_edges(conn, [(canary_id, alpha_id, "skos:related", 0.8)])
    assert (
        len(
            conn.execute(
                "SELECT src_id, dst_id FROM edges WHERE kind = 'skos:related' AND src_id = ? AND dst_id = ?",
                (canary_id, alpha_id),
            ).fetchall()
        )
        == 0
    )

    # Merge beta into alpha: fact edge (alpha -> 8080) has 2 literal facts across c1 and c2 -> weight = 0.5 * 2 = 1.0
    # Delete chunk c2: remaining literal fact in c1 between alpha and 8080 must have weight 0.5 (not 1.0!)
    merge_entities(conn, beta_id, alpha_id, label_kind="skos:altLabel")
    delete_chunks(conn, ["c2"])
    assert lint_graph_shacl(conn, auto_heal=False) == []
    w_row = conn.execute(
        "SELECT weight FROM edges WHERE src_id = ? AND dst_id = ? AND kind = 'fact'",
        (alpha_id, lit_id),
    ).fetchone()
    assert w_row is not None
    assert abs(float(w_row[0]) - 0.5) < 1e-6

    # Now promote 'canary' to a skos:Concept by referencing it in an ObjectProperty fact
    c3 = ChunkPointer("c3", "doc3.md", 1, 5, "Rollout Controller")
    r3 = OpenIERecord(
        chunk_id="c3",
        entities=[
            ("canary", "Progressive rollout strategy"),
            ("Argo Rollouts", "Kubernetes progressive delivery controller"),
        ],
        triples=[("canary", "orchestrated by", "Argo Rollouts")],
        structured_triples=[
            StructuredTriple(
                subj="canary",
                pred="orchestrated by",
                obj="Argo Rollouts",
                obj_kind="entity",
                subj_def="Progressive rollout strategy",
                obj_def="Kubernetes progressive delivery controller",
            )
        ],
    )
    _index_chunk_with_record(engine, c3, "Canary is orchestrated by Argo Rollouts.", r3)
    ent_rows_after = {r[0].lower(): r[1] for r in conn.execute("SELECT name, entity_id FROM entities").fetchall()}
    canary_concept_id = ent_rows_after["canary"]
    argo_id = ent_rows_after["argo rollouts"]
    # Because 'canary' is now a subject of an ObjectProperty fact, it can participate in SKOS edges
    upsert_semantic_edges(conn, [(canary_concept_id, argo_id, "skos:related", 0.8)])
    skos_rows = conn.execute(
        "SELECT src_id, dst_id FROM edges WHERE kind = 'skos:related' AND src_id = ? AND dst_id = ?",
        (canary_concept_id, argo_id),
    ).fetchall()
    assert len(skos_rows) == 1
    assert lint_graph_shacl(conn, auto_heal=False) == []


def test_forward_fact_edges_strictly_directed(synonym_token_counter, tmp_path: Path) -> None:
    """Verify that entity-to-entity facts store ONLY forward (subj -> obj) edges with weight 1.0,
    and topology + transition matrix distinguish ObjectProperty (1.0) vs DatatypeProperty (0.5)."""
    engine, conn = _make_engine(synonym_token_counter, tmp_path, "test_directed_edges.db")

    chunk = ChunkPointer(
        chunk_id="chunk-2",
        rel_posix_path="doc2.md",
        start_line=1,
        end_line=5,
        title="Architecture",
    )
    record = OpenIERecord(
        chunk_id="chunk-2",
        entities=[
            ("YAW Daemon", "A service daemon"),
            ("SQLite Engine", "A database engine"),
            ("8080", "Listen port"),
        ],
        triples=[
            ("YAW Daemon", "embeds", "SQLite Engine"),
            ("YAW Daemon", "uses port", "8080"),
        ],
        structured_triples=[
            StructuredTriple(
                subj="YAW Daemon",
                pred="embeds",
                obj="SQLite Engine",
                obj_kind="entity",
                subj_def="A service daemon",
                obj_def="A database engine",
            ),
            StructuredTriple(
                subj="YAW Daemon",
                pred="uses port",
                obj="8080",
                obj_kind="literal",
                obj_datatype="rdfs:Literal",
                subj_def="A service daemon",
                obj_def="Listen port",
            ),
        ],
    )

    _index_chunk_with_record(engine, chunk, "YAW Daemon embeds SQLite Engine on port 8080.", record)

    # Fetch entity IDs (lowercased)
    ent_rows = {r[0].lower(): r[1] for r in conn.execute("SELECT name, entity_id FROM entities").fetchall()}
    s_id = ent_rows["yaw daemon"]
    o_id = ent_rows["sqlite engine"]
    lit_id = ent_rows["8080"]

    fact_edges = {
        (r[0], r[1]): float(r[2])
        for r in conn.execute("SELECT src_id, dst_id, weight FROM edges WHERE kind = 'fact'").fetchall()
    }
    # Must contain ONLY forward edges with 1.0 for ObjectProperty and 0.5 for DatatypeProperty
    assert (s_id, o_id) in fact_edges
    assert (o_id, s_id) not in fact_edges
    assert abs(fact_edges[(s_id, o_id)] - 1.0) < 1e-6
    assert (s_id, lit_id) in fact_edges
    assert (lit_id, s_id) not in fact_edges
    assert abs(fact_edges[(s_id, lit_id)] - 0.5) < 1e-6

    # Verify load_graph_topology and build_transition_matrix preserve 1.0 vs 0.5
    chunk_ids, entity_ids, topo_edges = load_graph_topology(conn, include_kinds=True)
    topo_fact_weights = {(src, dst): float(w) for src, dst, k, w in topo_edges if k == "fact"}
    assert abs(topo_fact_weights[(s_id, o_id)] - 1.0) < 1e-6
    assert abs(topo_fact_weights[(s_id, lit_id)] - 0.5) < 1e-6

    policy = RelationTraversalPolicy()
    csr = build_transition_matrix(chunk_ids, entity_ids, topo_edges, policy=policy, directed=True)
    raw_s_to_o = float(csr.transition_t[csr.node_to_idx[o_id], csr.node_to_idx[s_id]])
    raw_s_to_lit = float(csr.transition_t[csr.node_to_idx[lit_id], csr.node_to_idx[s_id]])
    assert raw_s_to_o > raw_s_to_lit > 0.0


def test_lexical_subsumption_guard_inverts_erroneous_direction_and_rejects_missing_canonical(
    synonym_token_counter,
    tmp_path: Path,
) -> None:
    """Verify that inverted subsumption assertions are corrected from child to parent,
    missing canonical on skos:broader is rejected, and direct
    upsert_semantic_edges calls also enforce lexical subsumption inversion."""
    # First check parse_synonym_decisions does not default missing canonical to 'A' on skos:broader
    parsed = parse_synonym_decisions('{"decisions": [{"pair_id": 0, "relation": "skos:broader"}]}')
    assert len(parsed) == 1
    assert parsed[0]["canonical"] == ""

    def _judge(candidates: list[dict[str, Any]]) -> list[dict[str, str]]:
        return [
            {
                "pair_id": c["pair_id"],
                "relation": "skos:broader",
                "canonical": "redwood hosted inference",  # Erroneous canonical pointing to compound!
            }
            for c in candidates
        ]

    engine, conn = _make_engine(synonym_token_counter, tmp_path, "test_subsumption.db", synonym_callback=_judge)

    chunk = ChunkPointer(
        chunk_id="chunk-3",
        rel_posix_path="doc3.md",
        start_line=1,
        end_line=5,
        title="Inference Systems",
    )
    record = OpenIERecord(
        chunk_id="chunk-3",
        entities=[
            ("inference", "Generic model inference process"),
            ("redwood hosted inference", "A specialized hosted inference cluster"),
        ],
        triples=[],
        structured_triples=[],
    )

    _index_chunk_with_record(engine, chunk, "Inference and redwood hosted inference systems.", record)

    ent_rows = {r[0].lower(): r[1] for r in conn.execute("SELECT name, entity_id FROM entities").fetchall()}
    inf_id = ent_rows["inference"]
    rhi_id = ent_rows["redwood hosted inference"]

    broader_edges = conn.execute("SELECT src_id, dst_id FROM edges WHERE kind = 'skos:broader'").fetchall()
    assert len(broader_edges) == 1
    assert tuple(broader_edges[0]) == (rhi_id, inf_id)

    # Verify direct upsert_semantic_edges also inverts erroneous (head -> modifier_head) skos:broader
    conn.execute("DELETE FROM edges WHERE kind = 'skos:broader'")
    upsert_semantic_edges(conn, [(inf_id, rhi_id, "skos:broader", 0.85)])
    direct_broader = conn.execute("SELECT src_id, dst_id FROM edges WHERE kind = 'skos:broader'").fetchall()
    assert len(direct_broader) == 1
    assert tuple(direct_broader[0]) == (rhi_id, inf_id)

    # Missing canonical is invalid model output, not a symmetric relation.
    def _judge_empty_canonical(candidates: list[dict[str, Any]]) -> list[dict[str, str]]:
        return [
            {
                "pair_id": c["pair_id"],
                "relation": "skos:broader",
                "canonical": "",
            }
            for c in candidates
        ]

    engine2, conn2 = _make_engine(
        synonym_token_counter, tmp_path, "test_missing_canon.db", synonym_callback=_judge_empty_canonical
    )
    record2 = OpenIERecord(
        chunk_id="chunk-3b",
        entities=[
            ("alpha engine", "First subsystem"),
            ("beta router", "Second subsystem"),
        ],
        triples=[],
        structured_triples=[],
    )
    with pytest.raises(ValueError, match="invalid canonical"):
        _index_chunk_with_record(
            engine2,
            ChunkPointer("chunk-3b", "doc3b.md", 1, 5, "Subsystems"),
            "Alpha engine and beta router.",
            record2,
        )
    assert len(conn2.execute("SELECT src_id, dst_id FROM edges WHERE kind = 'skos:broader'").fetchall()) == 0
    assert len(conn2.execute("SELECT src_id, dst_id FROM edges WHERE kind = 'skos:related'").fetchall()) == 0


def test_version_sibling_guard_demotes_to_related(synonym_token_counter, tmp_path: Path) -> None:
    """Verify that coordinate version siblings are demoted from skos:broader to skos:related
    both in JudgeSynonyms and in direct upsert_semantic_edges calls."""

    def _judge(candidates: list[dict[str, Any]]) -> list[dict[str, str]]:
        return [
            {
                "pair_id": c["pair_id"],
                "relation": "skos:broader",
                "canonical": "HTTP 3",
            }
            for c in candidates
        ]

    engine, conn = _make_engine(synonym_token_counter, tmp_path, "test_siblings.db", synonym_callback=_judge)

    chunk = ChunkPointer(
        chunk_id="chunk-4",
        rel_posix_path="doc4.md",
        start_line=1,
        end_line=5,
        title="Web Protocols",
    )
    record = OpenIERecord(
        chunk_id="chunk-4",
        entities=[
            ("HTTP 2", "Hypertext Transfer Protocol version 2"),
            ("HTTP 3", "Hypertext Transfer Protocol version 3 over QUIC"),
        ],
        triples=[],
        structured_triples=[],
    )

    _index_chunk_with_record(engine, chunk, "Protocols HTTP 2 and HTTP 3.", record)

    # skos:broader must NOT be emitted
    broader_edges = conn.execute("SELECT src_id, dst_id FROM edges WHERE kind = 'skos:broader'").fetchall()
    assert len(broader_edges) == 0

    # Symmetric skos:related must be emitted instead
    related_edges = conn.execute("SELECT src_id, dst_id FROM edges WHERE kind = 'skos:related'").fetchall()
    assert len(related_edges) == 2

    # Verify direct upsert_semantic_edges call also demotes version siblings from skos:broader to skos:related
    ent_rows = {r[0].lower(): r[1] for r in conn.execute("SELECT name, entity_id FROM entities").fetchall()}
    h2_id, h3_id = ent_rows["http 2"], ent_rows["http 3"]
    conn.execute("DELETE FROM edges WHERE kind IN ('skos:broader', 'skos:related')")
    upsert_semantic_edges(conn, [(h2_id, h3_id, "skos:broader", 0.9)])
    assert len(conn.execute("SELECT src_id, dst_id FROM edges WHERE kind = 'skos:broader'").fetchall()) == 0
    assert len(conn.execute("SELECT src_id, dst_id FROM edges WHERE kind = 'skos:related'").fetchall()) == 2


def test_acronym_guard_routes_to_alt_label_merge(synonym_token_counter, tmp_path: Path) -> None:
    """Verify that acronym pairs are merged as skos:altLabel instead of emitting skos:broader,
    both in the JudgeSynonyms loop and in the fallback loop."""

    def _judge(candidates: list[dict[str, Any]]) -> list[dict[str, str]]:
        return [
            {
                "pair_id": c["pair_id"],
                "relation": "skos:broader",
                "canonical": "service level objectives",
            }
            for c in candidates
        ]

    engine, conn = _make_engine(synonym_token_counter, tmp_path, "test_acronym.db", synonym_callback=_judge)

    chunk = ChunkPointer(
        chunk_id="chunk-5",
        rel_posix_path="doc5.md",
        start_line=1,
        end_line=5,
        title="SRE Concepts",
    )
    record = OpenIERecord(
        chunk_id="chunk-5",
        entities=[
            ("SLOs", "Reliability goals"),
            ("service level objectives", "Formal reliability commitments"),
        ],
        triples=[],
        structured_triples=[],
    )

    _index_chunk_with_record(engine, chunk, "SRE SLOs and service level objectives.", record)

    # Zero skos:broader edges
    broader_edges = conn.execute("SELECT src_id, dst_id FROM edges WHERE kind = 'skos:broader'").fetchall()
    assert len(broader_edges) == 0

    # The entities must be merged into 1 entity with skos:altLabel
    entities = conn.execute("SELECT entity_id, name FROM entities").fetchall()
    assert len(entities) == 1
    assert entities[0][1].lower() == "service level objectives"

    labels = conn.execute("SELECT label_norm, label_kind FROM entity_labels").fetchall()
    kinds = {lkind for _, lkind in labels}
    assert "skos:prefLabel" in kinds
    assert "skos:altLabel" in kinds

    # Verify fallback loop also merges acronyms when JudgeSynonyms returns empty decisions
    engine_fb, conn_fb = _make_engine(
        synonym_token_counter, tmp_path, "test_acronym_fallback.db", synonym_callback=lambda _: []
    )
    _index_chunk_with_record(engine_fb, chunk, "SRE SLOs and service level objectives.", record)
    entities_fb = conn_fb.execute("SELECT entity_id, name FROM entities").fetchall()
    assert len(entities_fb) == 1
    assert entities_fb[0][1].lower() == "service level objectives"


def test_technical_punctuation_and_surface_forms_preserved(synonym_token_counter, tmp_path: Path) -> None:
    """Verify C++, C#, and Python 3.10 remain distinct entities with preserved surface forms in entities.name
    and entity_labels.raw_label, normalized label_norms in entity_labels, and COLLATE NOCASE on entities.name."""
    engine, conn = _make_engine(synonym_token_counter, tmp_path, "test_tech_punct.db", synonym_callback=lambda _: [])
    chunk = ChunkPointer(
        chunk_id="chunk-tech-1",
        rel_posix_path="languages.md",
        start_line=1,
        end_line=5,
        title="Programming Languages",
    )
    record = OpenIERecord(
        chunk_id="chunk-tech-1",
        entities=[
            ("C++", "A systems programming language"),
            ("C#", "A managed object-oriented programming language"),
            ("Python 3.10", "A release of the Python programming language"),
        ],
        triples=[
            ("C++", "contrasts with", "C#"),
            ("Python 3.10", "interoperates with", "C++"),
        ],
        structured_triples=[
            StructuredTriple(
                subj="C++",
                pred="contrasts with",
                obj="C#",
                obj_kind="entity",
                subj_def="A systems programming language",
                obj_def="A managed object-oriented programming language",
                spo_def="C++ contrasts with C#.",
            ),
            StructuredTriple(
                subj="Python 3.10",
                pred="interoperates with",
                obj="C++",
                obj_kind="entity",
                subj_def="A release of the Python programming language",
                obj_def="A systems programming language",
                spo_def="Python 3.10 interoperates with C++.",
            ),
        ],
    )

    _index_chunk_with_record(engine, chunk, "C++ contrasts with C#. Python 3.10 interoperates with C++.", record)

    rows = conn.execute("SELECT entity_id, name FROM entities ORDER BY name ASC").fetchall()
    names = {str(r[1]): str(r[0]) for r in rows}
    assert "C++" in names
    assert "C#" in names
    assert "Python 3.10" in names
    assert names["C++"] != names["C#"]

    # COLLATE NOCASE on entities.name allows case-insensitive SQL lookup while preserving exact surface casing
    nocase_row = conn.execute("SELECT name FROM entities WHERE name = 'python 3.10'").fetchone()
    assert nocase_row is not None
    assert str(nocase_row[0]) == "Python 3.10"

    label_rows = conn.execute(
        "SELECT label_norm, raw_label, label_kind FROM entity_labels WHERE chunk_id = 'chunk-tech-1'"
    ).fetchall()
    labels_by_norm = {str(r[0]): (str(r[1]), str(r[2])) for r in label_rows}
    assert labels_by_norm["c++"] == ("C++", "skos:prefLabel")
    assert labels_by_norm["c#"] == ("C#", "skos:prefLabel")
    assert labels_by_norm["python 3.10"] == ("Python 3.10", "skos:prefLabel")
    assert lint_graph_shacl(conn, auto_heal=False) == []


def test_cased_literal_objects_and_same_doc_consecutive_chunks_never_produce_false_different_from(
    synonym_token_counter,
    tmp_path: Path,
) -> None:
    """Verify cased literal objects ('AWS TGW', 'Pricing addendum', 'KMS/region support matrix', 'Diego Morales')
    and consecutive chunks of the same document ('Priya Nair', 'Ethan Cole', 'Diego Morales') never split into
    duplicate entities or produce false owl:differentFrom edges."""
    engine, conn = _make_engine(synonym_token_counter, tmp_path, "test_no_false_diff.db", synonym_callback=lambda _: [])

    doc_path = tmp_path / "meeting.md"
    doc_path.write_text(
        "# Architecture & Procurement Sync\n"
        "Participants: Priya Nair (Helix Procurement), Ethan Cole (Cloud Network Lead), Diego Morales (Security).\n"
        "Topics: AWS TGW, Pricing addendum, KMS/region support matrix.\n\n"
        "## Discussion\n"
        "Priya Nair reviewed the Pricing addendum. Ethan Cole presented AWS TGW route scaling.\n"
        "Diego Morales verified the KMS/region support matrix.\n",
        encoding="utf-8",
    )

    c1 = ChunkPointer(
        chunk_id="chunk-meet-1",
        rel_posix_path="meeting.md",
        start_line=1,
        end_line=3,
        title="Architecture & Procurement Sync",
        ast_island_start=1,
        ast_island_end=7,
    )
    r1 = OpenIERecord(
        chunk_id="chunk-meet-1",
        entities=[
            ("Priya Nair", "Helix Procurement"),
            ("Ethan Cole", "Cloud Network Lead"),
            ("Diego Morales", "Security"),
            ("AWS TGW", "AWS Transit Gateway networking component"),
            ("Pricing addendum", "Contract pricing document under review"),
            ("KMS/region support matrix", "Matrix of KMS regional support"),
        ],
        triples=[
            ("AWS TGW", "has topic", "AWS TGW"),
            ("Pricing addendum", "status", "Pricing addendum"),
            ("KMS/region support matrix", "documented in", "KMS/region support matrix"),
        ],
        structured_triples=[
            StructuredTriple(
                subj="AWS TGW",
                pred="has topic",
                obj="AWS TGW",
                obj_kind="literal",
                obj_datatype="rdfs:Literal",
                subj_def="AWS Transit Gateway networking component",
                obj_def="AWS Transit Gateway networking component",
                spo_def="AWS TGW is a topic in the meeting.",
            ),
            StructuredTriple(
                subj="Pricing addendum",
                pred="status",
                obj="Pricing addendum",
                obj_kind="literal",
                obj_datatype="rdfs:Literal",
                subj_def="Contract pricing document under review",
                obj_def="",
                spo_def="Pricing addendum is under procurement review.",
            ),
            StructuredTriple(
                subj="KMS/region support matrix",
                pred="documented in",
                obj="KMS/region support matrix",
                obj_kind="literal",
                obj_datatype="rdfs:Literal",
                subj_def="Matrix of KMS regional support",
                obj_def="Matrix of KMS regional support",
                spo_def="KMS/region support matrix is documented.",
            ),
        ],
    )

    c2 = ChunkPointer(
        chunk_id="chunk-meet-2",
        rel_posix_path="meeting.md",
        start_line=5,
        end_line=7,
        title="Discussion",
        ast_island_start=5,
        ast_island_end=7,
    )
    r2 = OpenIERecord(
        chunk_id="chunk-meet-2",
        entities=[
            ("Priya Nair", "A participant in the meeting."),
            ("Ethan Cole", "A participant in the meeting."),
            ("Diego Morales", "A participant in the meeting."),
            ("Meeting", "Architecture and procurement sync meeting"),
        ],
        triples=[
            ("Meeting", "hasDate", "Diego Morales"),
            ("Priya Nair", "reviewed", "Pricing addendum"),
        ],
        structured_triples=[
            StructuredTriple(
                subj="Meeting",
                pred="hasDate",
                obj="Diego Morales",
                obj_kind="literal",
                obj_datatype="rdfs:Literal",
                subj_def="Architecture and procurement sync meeting",
                obj_def="",
                spo_def="The meeting date is associated with Diego Morales.",
            ),
            StructuredTriple(
                subj="Priya Nair",
                pred="reviewed",
                obj="Pricing addendum",
                obj_kind="entity",
                subj_def="A participant in the meeting.",
                obj_def="Contract pricing document under review",
                spo_def="Priya Nair reviewed the Pricing addendum.",
            ),
        ],
    )

    _index_chunk_with_record(
        engine,
        c1,
        "# Architecture & Procurement Sync\nParticipants: Priya Nair, Ethan Cole, Diego Morales.",
        r1,
    )
    _index_chunk_with_record(
        engine,
        c2,
        "## Discussion\nPriya Nair reviewed the Pricing addendum.",
        r2,
    )

    # Every named entity must exist exactly once (no duplicate sense splits)
    for target_label in (
        "priya nair",
        "ethan cole",
        "diego morales",
        "aws tgw",
        "pricing addendum",
        "kms/region support matrix",
    ):
        eids = [
            str(r[0])
            for r in conn.execute(
                "SELECT DISTINCT entity_id FROM entity_labels WHERE label_norm = ?",
                (target_label,),
            ).fetchall()
        ]
        assert len(eids) == 1, f"Expected 1 entity for {target_label!r}, got {eids}"

    # Verify Priya Nair combined her header role and dialogue role in her definition
    priya_def = str(conn.execute("SELECT definition FROM entities WHERE name = 'Priya Nair'").fetchone()[0])
    assert "Helix Procurement" in priya_def
    assert "participant in the meeting" in priya_def

    # Zero owl:differentFrom edges in the entire graph
    diff_edges = conn.execute("SELECT src_id, dst_id FROM edges WHERE kind = 'owl:differentFrom'").fetchall()
    assert diff_edges == []
    assert lint_graph_shacl(conn, auto_heal=False) == []
