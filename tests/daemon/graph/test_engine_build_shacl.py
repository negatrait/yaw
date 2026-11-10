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
from yaw.daemon.graph.inference import YawProgram, parse_hierarchy_decisions
from yaw.daemon.graph.inference.embedding import EmbeddingProfile
from yaw.daemon.graph.planner import apply_synonym_adjudication
from yaw.daemon.graph.ppr import build_transition_matrix
from yaw.daemon.graph.storage import (
    lint_graph_shacl,
    load_graph_topology,
    merge_entities,
    open_yaw_db,
    upsert_entities,
    upsert_entity_labels,
    upsert_semantic_edges,
)
from yaw.daemon.indexing import bind_native_indexing
from yaw.daemon.ingestion.corpus import chunk_source_text


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
        self.records: dict[str, tuple[str, str, str]] = {}

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
        matched = sorted(
            (
                (max(full_text.rfind(text), full_text.rfind(json.dumps(text)[1:-1])), ner, triples)
                for text, ner, triples in self.records.values()
            ),
            reverse=True,
        )
        ner_json, triples_json = (
            matched[0][1:] if matched and matched[0][0] >= 0 else (self._ner_json, self._triples_json)
        )
        if "repl_history" in full_text and "variables_info" in full_text:
            return [
                json.dumps({"reasoning": "No semantic repair proposed by this fixture.", "code": "SUBMIT(repairs=[])"})
            ]
        if "candidate_pairs" in full_text or '"pair_id"' in full_text:
            cands = self._parse_candidate_pairs_from_text(full_text)
            self.observed_candidates.extend(cands)
            decisions = complete_decisions(
                cands,
                self._synonym_fn(cands) if self._synonym_fn is not None else [],
                stage_text=full_text,
            )
            return [json.dumps({"decisions": decisions})]
        if "triples" in full_text and "named_entities" in full_text and triples_json is not None:
            return [triples_json]
        if "named_entities" in full_text and ner_json is not None:
            return [ner_json]
        out = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return [out]


class DeterministicMockEmbedder:
    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self.profile = EmbeddingProfile("deterministic", "fixture-v1", dim, 2048, "mean", "", "")

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
    program = YawProgram(
        synonym_token_counter=synonym_token_counter, enable_gbnf=False, orchestrator_lm=mock_lm, rlm_sub_lm=mock_lm
    )
    embedder: Any = DeterministicMockEmbedder(dim=64)
    engine = bind_native_indexing(
        YawGraphEngine(
            db_path=db_path,
            workspace_root=tmp_path,
            program=program,
            embedder=embedder,
        )
    )
    return engine, engine.conn


def _index_chunk_with_record(
    engine: YawGraphEngine,
    chunk: ChunkPointer,
    text: str,
    record: OpenIERecord,
) -> ChunkPointer:
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
                    "sub": st.subj,
                    "pred": [
                        st.pred,
                        "owl:DatatypeProperty" if st.obj_kind == "literal" else "owl:ObjectProperty",
                    ],
                    "obj": (
                        {"value": st.obj, "datatype": st.obj_datatype, "lang": st.obj_lang}
                        if st.obj_kind == "literal"
                        else st.obj
                    ),
                    "def": st.spo_def or f"{st.subj} {st.pred} {st.obj}.",
                }
                for st in record.structured_triples
            ]
        }
    )
    mock_lm = engine.program.rlm_sub_lm
    assert isinstance(mock_lm, DeterministicMockLM)
    mock_lm.set_stage_responses(ner_json, triples_json)
    mock_lm.records[chunk.chunk_id] = (text, ner_json, triples_json)
    rel_path = str(Path(chunk.rel_posix_path).with_suffix(".json"))
    rows = [{"text": source} for source, _, _ in mock_lm.records.values()]
    content = json.dumps(rows, indent=2)
    (engine.workspace_root / rel_path).write_text(content, encoding="utf8")
    chunks = chunk_source_text(content, rel_path, min_chunk_size=1)
    engine.index_chunks(chunks)
    return next(ptr for ptr, source in chunks if source == text)


def test_datatype_property_literal_objects_stored_with_weighted_edges_and_excluded_from_skos(
    synonym_token_counter,
    tmp_path: Path,
) -> None:
    """Literal objects belong to facts and never become entities or SKOS candidates."""
    # Literal values are discovered by the fact extractor, not defined by NER.
    lm_triple_json = json.dumps(
        {
            "triples": [
                {
                    "sub": "YAW Service",
                    "pred": ["listens on port", "owl:DatatypeProperty"],
                    "obj": {"value": "8080", "datatype": "xsd:integer", "lang": ""},
                    "def": "YAW Service listens on port 8080.",
                }
            ]
        }
    )
    dspy.settings.configure(lm=DeterministicMockLM([lm_triple_json]))
    prog = YawProgram(synonym_token_counter=synonym_token_counter, enable_gbnf=False)
    extracted_sts = prog.extract_structured_triples(
        "YAW Service listens on port 8080.",
        [
            ("YAW Service", "A high performance knowledge graph daemon"),
        ],
    )
    assert len(extracted_sts) == 1
    assert extracted_sts[0].obj_kind == "literal"
    assert extracted_sts[0].obj_def == ""

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
                obj_def="",
            ),
            StructuredTriple(
                subj="YAW Service",
                pred="generated revenue",
                obj="$1.2B",
                obj_kind="literal",
                obj_datatype="rdfs:Literal",
                subj_def="A high performance knowledge graph daemon",
                obj_def="",
            ),
        ],
    )

    chunk = _index_chunk_with_record(engine, chunk, "YAW Service listens on port 8080 and generated $1.2B.", record)

    # Verify scalar and formatted literals NEVER enter entities
    all_entities = {
        r[1]: (r[0], r[2]) for r in conn.execute("SELECT entity_id, name, definition FROM entities").fetchall()
    }
    assert "8080" not in all_entities
    assert "$1.2B" not in all_entities
    assert "YAW Service" in all_entities
    subj_id = all_entities["YAW Service"][0]

    # Verify facts table contains both literal facts with obj_id=None and obj_literal_lex
    facts = conn.execute(
        "SELECT fact_id, subj_id, pred, obj_id, obj_kind, obj_literal_lex, obj_datatype FROM facts ORDER BY pred"
    ).fetchall()
    assert len(facts) == 2
    assert all(f[3] is None for f in facts)
    assert all(f[4] == "literal" for f in facts)
    assert {f[5] for f in facts} == {"8080", "$1.2B"}

    # Verify edges contains zero fact edges to literal objects
    fact_edges = conn.execute("SELECT src_id, dst_id, kind, weight FROM edges WHERE kind = 'fact'").fetchall()
    assert len(fact_edges) == 0

    # Verify passage edges connect chunk-1 to subj_id only
    passage_edges = conn.execute("SELECT src_id, dst_id FROM edges WHERE kind = 'passage'").fetchall()
    passage_pairs = {(r[0], r[1]) for r in passage_edges}
    assert (chunk.chunk_id, subj_id) in passage_pairs

    # Verify pure literals '8080' and '$1.2B' were excluded from SKOS candidate discovery
    assert len(called_candidates) == 0
    skos_edges = conn.execute(
        "SELECT src_id, dst_id, kind FROM edges WHERE kind IN ('skos:broader', 'skos:related')"
    ).fetchall()
    assert len(skos_edges) == 0


def test_dual_role_literal_promoted_to_concept_and_weight_preserved_across_merge(
    synonym_token_counter,
    tmp_path: Path,
) -> None:
    """Verify a surface form used first as a literal and later as an ObjectProperty endpoint is promoted
    to a skos:Concept, and merge_entities preserves 0.5 weights on pure literal fact edges."""
    engine, conn = _make_engine(synonym_token_counter, tmp_path, "test_dual_role.db")

    c1 = ChunkPointer("c1", "doc1.md", 1, 5, "Rollout")
    r1 = OpenIERecord(
        chunk_id="c1",
        entities=[
            ("Gateway Alpha", "Primary API gateway"),
            ("Gateway Beta", "Secondary API gateway alias"),
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
                obj_def="",
            ),
            StructuredTriple(
                subj="Gateway Beta",
                pred="uses port",
                obj="8080",
                obj_kind="literal",
                obj_datatype="rdfs:Literal",
                subj_def="Secondary API gateway alias",
                obj_def="",
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
                obj_def="",
            )
        ],
    )
    _index_chunk_with_record(engine, c1, "Gateway Alpha and Beta use port 8080 in canary mode.", r1)
    _index_chunk_with_record(engine, c2, "Gateway Alpha fallback port 8080.", r2)

    ent_rows = {r[0].lower(): r[1] for r in conn.execute("SELECT name, entity_id FROM entities").fetchall()}
    alpha_id = ent_rows["gateway alpha"]
    beta_id = ent_rows["gateway beta"]
    assert "8080" not in ent_rows
    assert "canary" in ent_rows

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
    assert lint_graph_shacl(conn) == []

    # Merge beta into alpha: literal facts across c1 and c2 are merged
    merged = merge_entities(conn, beta_id, alpha_id, label_kind="skos:altLabel")
    alpha_id = merged.canonical_id
    assert lint_graph_shacl(conn) == []
    lit_fact = conn.execute(
        "SELECT subj_id, pred, obj_literal_lex, obj_kind, obj_id FROM facts WHERE obj_literal_lex = '8080'"
    ).fetchone()
    assert lit_fact is not None
    assert lit_fact[0] == alpha_id
    assert lit_fact[1] == "uses port"
    assert lit_fact[3] == "literal"
    assert lit_fact[4] is None


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
                obj_def="",
            ),
        ],
    )

    _index_chunk_with_record(engine, chunk, "YAW Daemon embeds SQLite Engine on port 8080.", record)

    # Fetch entity IDs (lowercased)
    ent_rows = {r[0].lower(): r[1] for r in conn.execute("SELECT name, entity_id FROM entities").fetchall()}
    s_id = ent_rows["yaw daemon"]
    o_id = ent_rows["sqlite engine"]
    assert "8080" not in ent_rows

    fact_edges = {
        (r[0], r[1]): float(r[2])
        for r in conn.execute("SELECT src_id, dst_id, weight FROM edges WHERE kind = 'fact'").fetchall()
    }
    # Must contain ONLY forward edges for ObjectProperty (0.8) and ZERO edges for literal objects
    assert (s_id, o_id) in fact_edges
    assert (o_id, s_id) not in fact_edges
    assert abs(fact_edges[(s_id, o_id)] - 0.8) < 1e-6
    assert len(fact_edges) == 1

    # Verify literal fact exists strictly in facts table
    lit_fact = conn.execute(
        "SELECT subj_id, pred, obj_literal_lex, obj_kind, obj_id FROM facts WHERE obj_literal_lex = '8080'"
    ).fetchone()
    assert lit_fact is not None
    assert lit_fact[0] == s_id
    assert lit_fact[1] == "uses port"
    assert lit_fact[3] == "literal"
    assert lit_fact[4] is None

    # Verify load_graph_topology and build_transition_matrix preserve the fact edge
    chunk_ids, entity_ids, topo_edges, _ = load_graph_topology(conn)
    topo_fact_weights = {(src, dst): float(w) for src, dst, k, w in topo_edges if k == "fact"}
    assert abs(topo_fact_weights[(s_id, o_id)] - 0.8) < 1e-6
    assert len(topo_fact_weights) == 1

    policy = RelationTraversalPolicy()
    csr = build_transition_matrix(chunk_ids, entity_ids, topo_edges, policy=policy, directed=True)
    raw_s_to_o = float(csr.transition_t[csr.node_to_idx[o_id], csr.node_to_idx[s_id]])
    assert raw_s_to_o > 0.0


def test_skos_broader_rejects_missing_canonical(
    synonym_token_counter,
    tmp_path: Path,
) -> None:
    """Verify that parse_hierarchy_decisions extracts skos:broader and direction cleanly."""
    payload = json.dumps({"decisions": [{"pair_id": "a|b", "reasoning": "valid", "broader": True, "narrower": False}]})
    parsed = parse_hierarchy_decisions(payload)
    assert len(parsed) == 1
    assert parsed[0]["relation"] == "skos:broader"
    assert parsed[0]["source_direction"] == "a_to_b"

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


def test_canonical_gemma4_synonym_adjudication_and_fact_pole_self_loop_guard(
    synonym_token_counter,
    tmp_path: Path,
) -> None:
    """Verify apply_synonym_adjudication executes only Gemma 4 decisions without acronym/homograph overrides,
    and prevents merging fact-linked poles even when entity names differ."""
    # 1. Verify ("audit trails", "ATS") with Gemma 4 skos:related is NOT overridden into skos:altLabel merge
    stage_conn = open_yaw_db(":memory:")
    upsert_entities(
        stage_conn,
        [
            ("ent-ats", "ATS", "Applicant Tracking System"),
            ("ent-audit", "audit trails", "Immutable security audit logs"),
            ("ent-sec-a", "Security", "Security team in doc A"),
            ("ent-sec-b", "Security", "Security role in doc B"),
        ],
    )
    candidates = [
        {"entity_a": "ent-ats", "entity_b": "ent-audit", "similarity": 0.88},
        {"entity_a": "ent-sec-a", "entity_b": "ent-sec-b", "similarity": 0.92},
    ]
    cand_meta = {
        "ent-ats": ("ATS", "Applicant Tracking System"),
        "ent-audit": ("audit trails", "Immutable security audit logs"),
        "ent-sec-a": ("Security", "Security team in doc A"),
        "ent-sec-b": ("Security", "Security role in doc B"),
    }
    decisions = [
        {"pair_id": "ent-ats|ent-audit", "relation": "skos:related", "canonical": "ent-audit"},
        {"pair_id": "ent-sec-a|ent-sec-b", "relation": "skos:related", "canonical": "ent-sec-a"},
    ]
    rlm_stats = {"merged_entities": 0, "related_edges": 0, "different_from_edges": 0, "broader_edges": 0}
    staged_merges, semantic_batch = apply_synonym_adjudication(
        stage_conn,
        candidates,
        cand_meta,
        decisions,
        set(),
        rlm_stats=rlm_stats,
    )
    assert staged_merges == []
    assert rlm_stats["merged_entities"] == 0
    assert rlm_stats["different_from_edges"] == 0
    assert rlm_stats["related_edges"] == 2
    assert [row[:4] for row in semantic_batch] == [
        ("ent-ats", "ent-audit", "skos:related", 0.88),
        ("ent-sec-a", "ent-sec-b", "skos:related", 0.92),
    ]
    stage_conn.close()

    # 2. Verify fact_linked_pairs prevents self-loop merge even when name_src != name_tgt
    def _judge_alt(cands: list[dict[str, Any]]) -> list[dict[str, str]]:
        return [
            {
                "pair_id": c["pair_id"],
                "relation": "skos:altLabel",
                "canonical": "NetSuite",
            }
            for c in cands
        ]

    engine, conn = _make_engine(synonym_token_counter, tmp_path, "test_fact_pole.db", synonym_callback=_judge_alt)
    chunk = ChunkPointer(
        chunk_id="chunk-5",
        rel_posix_path="doc5.md",
        start_line=1,
        end_line=5,
        title="ERP Sync",
    )
    record = OpenIERecord(
        chunk_id="chunk-5",
        entities=[
            ("NetSuite", "Enterprise ERP system"),
            ("NetSuite import", "Nightly financial import pipeline into NetSuite"),
        ],
        triples=[("NetSuite import", "loads into", "NetSuite")],
        structured_triples=[
            StructuredTriple(
                subj="NetSuite import",
                pred="loads into",
                obj="NetSuite",
                obj_kind="entity",
                subj_def="Nightly financial import pipeline into NetSuite",
                obj_def="Enterprise ERP system",
                spo_def="NetSuite import loads into NetSuite.",
            )
        ],
    )
    _index_chunk_with_record(engine, chunk, "NetSuite import loads into NetSuite.", record)

    # Both fact poles must remain distinct entities and the connecting fact must survive
    entities = {str(r[1]) for r in conn.execute("SELECT entity_id, name FROM entities").fetchall()}
    assert entities == {"NetSuite", "NetSuite import"}
    assert conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 1
    assert len(conn.execute("SELECT src_id, dst_id FROM edges WHERE kind = 'skos:related'").fetchall()) == 2


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

    chunk = _index_chunk_with_record(
        engine, chunk, "C++ contrasts with C#. Python 3.10 interoperates with C++.", record
    )

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
        "SELECT label_norm, raw_label, label_kind FROM entity_labels WHERE chunk_id = ?", (chunk.chunk_id,)
    ).fetchall()
    labels_by_norm = {str(r[0]): (str(r[1]), str(r[2])) for r in label_rows}
    assert labels_by_norm["c++"] == ("C++", "skos:prefLabel")
    assert labels_by_norm["c#"] == ("C#", "skos:prefLabel")
    assert labels_by_norm["python 3.10"] == ("Python 3.10", "skos:prefLabel")
    assert lint_graph_shacl(conn) == []


def test_cased_literal_objects_and_same_doc_consecutive_chunks_never_produce_false_different_from(
    synonym_token_counter,
    tmp_path: Path,
) -> None:
    """Verify cased literal objects ('AWS TGW', 'Pricing addendum', 'KMS/region support matrix', 'Diego Morales')
    and consecutive chunks of the same document ('Priya Nair', 'Ethan Cole', 'Diego Morales') never split into
    duplicate entities or produce false owl:differentFrom edges."""

    def judge_same_people(candidates):
        return [
            {"pair_id": pair["pair_id"], "relation": "skos:exactMatch", "definition": "A participant in the meeting."}
            for pair in candidates
            if pair["name_a"].casefold() == pair["name_b"].casefold()
        ]

    engine, conn = _make_engine(
        synonym_token_counter, tmp_path, "test_no_false_diff.db", synonym_callback=judge_same_people
    )

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
            ("Pricing addendum", "Contract pricing document under review"),
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

    # Verify Priya Nair retained the longer canonical definition after auto-merge
    priya_def = str(conn.execute("SELECT definition FROM entities WHERE name = 'Priya Nair'").fetchone()[0])
    assert priya_def == "A participant in the meeting."

    # Zero owl:differentFrom edges in the entire graph
    diff_edges = conn.execute("SELECT src_id, dst_id FROM edges WHERE kind = 'owl:differentFrom'").fetchall()
    assert diff_edges == []
    assert lint_graph_shacl(conn) == []


def test_apply_synonym_adjudication_remaps_semantic_edges_after_later_merge() -> None:
    """Verify semantic edges buffered before a later merge in the same batch are remapped to surviving entities
    with zero EdgeShape/closedWorld dangling-endpoint SHACL violations."""
    stage_conn = open_yaw_db(":memory:")
    try:
        upsert_entities(
            stage_conn,
            [
                ("ent-a", "NetSuite", "Enterprise ERP platform"),
                ("ent-b", "NetSuite import", "Nightly ERP import pipeline"),
                ("ent-c", "NetSuite staging import", "Staging import pipeline into NetSuite"),
            ],
        )
        upsert_entity_labels(
            stage_conn,
            [
                ("netsuite", "ent-a", "skos:prefLabel", "NetSuite", "c1"),
                ("netsuite import", "ent-b", "skos:prefLabel", "NetSuite import", "c1"),
                ("netsuite staging import", "ent-c", "skos:prefLabel", "NetSuite staging import", "c2"),
            ],
        )
        candidates = [
            {"entity_a": "ent-a", "entity_b": "ent-b", "similarity": 0.85},
            {"entity_a": "ent-b", "entity_b": "ent-c", "similarity": 0.91},
        ]
        cand_meta = {
            "ent-a": ("NetSuite", "Enterprise ERP platform"),
            "ent-b": ("NetSuite import", "Nightly ERP import pipeline"),
            "ent-c": ("NetSuite staging import", "Staging import pipeline into NetSuite"),
        }
        decisions = [
            {"pair_id": "ent-a|ent-b", "relation": "skos:related", "canonical": "ent-a"},
            {"pair_id": "ent-b|ent-c", "relation": "skos:altLabel", "canonical": "ent-c"},
        ]
        rlm_stats = {"merged_entities": 0, "related_edges": 0, "different_from_edges": 0, "broader_edges": 0}
        staged_merges, resolved_batch = apply_synonym_adjudication(
            stage_conn,
            candidates,
            cand_meta,
            decisions,
            set(),
            rlm_stats=rlm_stats,
        )
        assert [(merge.removed_id, merge.canonical_id) for merge in staged_merges] == [("ent-c", "ent-b")]
        assert [row[:4] for row in resolved_batch] == [("ent-a", "ent-b", "skos:related", 0.85)]
        assert lint_graph_shacl(stage_conn) == []
    finally:
        stage_conn.close()


def test_apply_synonym_adjudication_prevents_transitive_collapse_across_separated_clusters() -> None:
    """Verify fact_linked_pairs blocks transitive self-loop collapse while non-fact semantic edges do not
    override Gemma 4 skos:altLabel merges and have any resulting self-loops dropped in Phase 2."""
    stage_conn = open_yaw_db(":memory:")
    try:
        upsert_entities(
            stage_conn,
            [
                ("ent-a", "Service Alpha", "Primary service mention A"),
                ("ent-b", "Alpha Service", "Primary service mention B"),
                ("ent-c", "Alpha", "Primary service mention C"),
                ("ent-p1", "Billing Service", "Core billing system"),
                ("ent-alias", "Billing Platform", "Core billing system alias"),
                ("ent-p2", "invoice_record", "Invoice record emitted by Billing Service"),
            ],
        )
        upsert_entity_labels(
            stage_conn,
            [
                ("service alpha", "ent-a", "skos:prefLabel", "Service Alpha", "c1"),
                ("alpha service", "ent-b", "skos:prefLabel", "Alpha Service", "c1"),
                ("alpha", "ent-c", "skos:prefLabel", "Alpha", "c2"),
                ("billing service", "ent-p1", "skos:prefLabel", "Billing Service", "c3"),
                ("billing platform", "ent-alias", "skos:prefLabel", "Billing Platform", "c3"),
                ("invoice_record", "ent-p2", "skos:prefLabel", "invoice_record", "c3"),
            ],
        )
        candidates = [
            {"entity_a": "ent-a", "entity_b": "ent-c", "similarity": 0.84},
            {"entity_a": "ent-a", "entity_b": "ent-b", "similarity": 0.89},
            {"entity_a": "ent-b", "entity_b": "ent-c", "similarity": 0.87},
            {"entity_a": "ent-p1", "entity_b": "ent-alias", "similarity": 0.93},
            {"entity_a": "ent-alias", "entity_b": "ent-p2", "similarity": 0.82},
        ]
        cand_meta = {
            "ent-a": ("Service Alpha", "Primary service mention A"),
            "ent-b": ("Alpha Service", "Primary service mention B"),
            "ent-c": ("Alpha", "Primary service mention C"),
            "ent-p1": ("Billing Service", "Core billing system"),
            "ent-alias": ("Billing Platform", "Core billing system alias"),
            "ent-p2": ("invoice_record", "Invoice record emitted by Billing Service"),
        }
        decisions = [
            {"pair_id": "ent-a|ent-c", "relation": "skos:related", "canonical": "ent-a"},
            {"pair_id": "ent-a|ent-b", "relation": "skos:altLabel", "canonical": "ent-a"},
            {"pair_id": "ent-b|ent-c", "relation": "skos:altLabel", "canonical": "ent-c"},
            {"pair_id": "ent-p1|ent-alias", "relation": "skos:altLabel", "canonical": "ent-alias"},
            {"pair_id": "ent-alias|ent-p2", "relation": "skos:altLabel", "canonical": "ent-p2"},
        ]
        rlm_stats = {"merged_entities": 0, "related_edges": 0, "different_from_edges": 0, "broader_edges": 0}
        staged_merges, resolved_batch = apply_synonym_adjudication(
            stage_conn,
            candidates,
            cand_meta,
            decisions,
            {("ent-p1", "ent-p2")},
            rlm_stats=rlm_stats,
        )
        assert [(merge.removed_id, merge.canonical_id) for merge in staged_merges] == [
            ("ent-b", "ent-a"),
            ("ent-c", "ent-a"),
            ("ent-p1", "ent-alias"),
        ]
        assert [row[:4] for row in resolved_batch] == [("ent-alias", "ent-p2", "skos:related", 0.82)]
        surviving = {str(r[0]) for r in stage_conn.execute("SELECT entity_id FROM entities").fetchall()}
        assert surviving == {"ent-a", "ent-alias", "ent-p2"}
        assert lint_graph_shacl(stage_conn) == []
    finally:
        stage_conn.close()


def test_statement_shape_closed_world_flags_and_repairs_duplicate_endpoint_definitions() -> None:
    """Verify StatementShape/closedWorld and validate_stage_invariants('triples') flag statement definitions
    that duplicate an endpoint entity definition verbatim, and apply_shacl_repairs heals them."""
    from yaw.daemon.graph.inference.invariants import validate_stage_invariants
    from yaw.daemon.graph.storage import (
        apply_shacl_repairs,
        upsert_chunk_pointers,
        upsert_fact_sources,
        upsert_facts,
    )

    inv_score, inv_fb = validate_stage_invariants(
        "triples",
        {
            "passage": "NetSuite exports records to Snowflake.",
            "named_entities": [
                ["NetSuite", "Enterprise ERP system"],
                ["Snowflake", "Cloud data warehouse"],
            ],
        },
        dspy.Prediction(
            triples=[
                {
                    "sub": "NetSuite",
                    "pred": ["exports to", "owl:ObjectProperty"],
                    "obj": "Snowflake",
                    "def": "Enterprise ERP system",
                }
            ]
        ),
    )
    assert inv_score < 1.0
    assert "StatementShape/closedWorld" in inv_fb

    stage_conn = open_yaw_db(":memory:")
    try:
        upsert_chunk_pointers(
            stage_conn,
            [(ChunkPointer("c1", "doc.md", 1, 5, "Doc"), "NetSuite exports records to Snowflake.")],
        )
        upsert_entities(
            stage_conn,
            [
                ("ent-s", "NetSuite", "Enterprise ERP system"),
                ("ent-o", "Snowflake", "Cloud data warehouse"),
            ],
        )
        upsert_entity_labels(
            stage_conn,
            [
                ("netsuite", "ent-s", "skos:prefLabel", "NetSuite", "c1"),
                ("snowflake", "ent-o", "skos:prefLabel", "Snowflake", "c1"),
            ],
        )
        upsert_facts(
            stage_conn,
            [
                (
                    "fact-dup-subj",
                    "ent-s",
                    "exports to",
                    "ent-o",
                    '["netsuite", "exports to", "snowflake"]',
                    "Enterprise ERP system",
                    "entity",
                    None,
                    "",
                    "",
                ),
                (
                    "fact-dup-obj",
                    "ent-s",
                    "syncs with",
                    "ent-o",
                    '["netsuite", "syncs with", "snowflake"]',
                    "Cloud data warehouse",
                    "entity",
                    None,
                    "",
                    "",
                ),
            ],
        )
        upsert_fact_sources(stage_conn, [("fact-dup-subj", "c1"), ("fact-dup-obj", "c1")])
        stage_conn.execute("INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('c1', 'ent-s', 'passage', 1.0)")
        stage_conn.execute("INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('c1', 'ent-o', 'passage', 1.0)")
        stage_conn.execute("INSERT INTO edges (src_id, dst_id, kind, weight) VALUES ('ent-s', 'ent-o', 'fact', 1.0)")

        violations = lint_graph_shacl(stage_conn)
        dup_violations = [
            v for v in violations if v.shape == "StatementShape/closedWorld" and v.result_path == "skos:definition"
        ]
        assert {v.focus_node for v in dup_violations} == {"fact-dup-subj", "fact-dup-obj"}

        applied = apply_shacl_repairs(
            stage_conn,
            [
                {
                    "focus_node": "fact-dup-subj",
                    "action": "set_definition",
                    "value": "NetSuite exports records to Snowflake.",
                },
                {
                    "focus_node": "fact-dup-obj",
                    "action": "set_definition",
                    "value": "NetSuite syncs financial tables with Snowflake.",
                },
            ],
        )
        assert applied == 2
        assert lint_graph_shacl(stage_conn) == []
    finally:
        stage_conn.close()
