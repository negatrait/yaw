"""End-to-end integration test verifying that engine.index_chunks enforces the zero-violation SHACL build gate."""

import hashlib
import json
from pathlib import Path
from typing import Any

import dspy

from yaw.core import ChunkPointer
from yaw.daemon.graph.engine import YawGraphEngine
from yaw.daemon.graph.inference import HippoRAGProgram
from yaw.daemon.graph.ppr import run_ppr
from yaw.daemon.graph.storage import (
    lint_graph_shacl,
    open_yaw_db,
    serialize_graph_turtle,
)


class DummyEmbedder:
    """Deterministic unit-norm embedder that gives concepts high cosine similarity for candidate pairing."""

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    def __call__(self, texts: list[str]) -> list[list[float]]:
        res = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            vec = [0.85 + 0.15 * (float(b) / 255.0) for b in h[: self.dim]]
            norm = sum(x * x for x in vec) ** 0.5 or 1.0
            res.append([x / norm for x in vec])
        return res

    def encode(self, texts: list[str], instruction: str = "", stage: str | None = None) -> list[list[float]]:
        return self(texts)


class PipelineGateMockLM(dspy.LM):
    """Deterministic multi-stage DSPy LM driving Stage 1 NER, Stage 2 Triples, Stage 3 JudgeSynonyms, and Retrieval."""

    def __init__(self, *, mode: str = "gate") -> None:
        super().__init__(model="mock-lm")
        self.mode = mode
        self.observed_synonym_candidates: list[dict[str, Any]] = []
        self.rlm_healer_called: bool = False

    def _extract_prompt_text(
        self,
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

    def _parse_candidate_pairs_from_text(self, text: str) -> list[dict[str, Any]]:
        decoder = json.JSONDecoder()
        found: list[dict[str, Any]] = []
        for idx, ch in enumerate(text):
            if ch != "[":
                continue
            try:
                parsed, _ = decoder.raw_decode(text[idx:])
            except json.JSONDecodeError:
                continue
            if (
                isinstance(parsed, list)
                and parsed
                and all(isinstance(item, dict) and "pair_id" in item for item in parsed)
            ):
                found = parsed
        return found

    def __call__(
        self, prompt: str | None = None, messages: list[dict[str, Any]] | None = None, **kwargs: Any
    ) -> list[str]:
        full_text = self._extract_prompt_text(prompt, messages)

        # RLM healer or direct SHACL repair detection
        if "HealSHACLViolations" in full_text or "ProposeSHACLRepairs" in full_text or "violations" in full_text:
            self.rlm_healer_called = True
            return ['{"repairs": []}']

        # Query intent stage
        if "query_intent" in full_text and "question" in full_text:
            return ['{"query_intent": "Yet Another Wiki listens on port 8080 and launched on 2026-10-25"}']

        # Recognition memory triple filter stage
        if "fact_before_filter" in full_text:
            return [
                json.dumps(
                    {
                        "fact": [
                            ["yet another wiki", "listens on port", "8080"],
                            ["yet another wiki", "launched on", "2026-10-25"],
                        ]
                    }
                )
            ]

        # Stage 3: JudgeSynonyms
        if "candidate_pairs" in full_text or '"pair_id"' in full_text:
            cands = self._parse_candidate_pairs_from_text(full_text)
            self.observed_synonym_candidates.extend(cands)
            decisions: list[dict[str, str]] = []
            if self.mode == "gate":
                for c in cands:
                    pid = str(c["pair_id"])
                    na = str(c.get("name_a", "")).strip().lower()
                    nb = str(c.get("name_b", "")).strip().lower()
                    pair_names = {na, nb}
                    if pair_names == {"engineering", "security engineering"}:
                        # Adversarial inverted skos:broader: erroneously picking the specialized compound as canonical
                        decisions.append(
                            {
                                "pair_id": pid,
                                "relation": "skos:broader",
                                "canonical": "security engineering",
                            }
                        )
                    elif pair_names == {"http 2", "http 3"}:
                        # Adversarial skos:broader on coordinate version siblings
                        decisions.append(
                            {
                                "pair_id": pid,
                                "relation": "skos:broader",
                                "canonical": "HTTP 3",
                            }
                        )
                    elif pair_names == {"yaw", "yet another wiki"}:
                        # Adversarial skos:broader on an acronym pair (must be routed to skos:altLabel merge)
                        decisions.append(
                            {
                                "pair_id": pid,
                                "relation": "skos:broader",
                                "canonical": "Yet Another Wiki",
                            }
                        )
                    else:
                        decisions.append(
                            {
                                "pair_id": pid,
                                "relation": "owl:differentFrom",
                                "canonical": "A",
                            }
                        )
            elif self.mode == "pre_rlm_heal":
                for c in cands:
                    pid = str(c["pair_id"])
                    na = str(c.get("name_a", "")).strip().lower()
                    nb = str(c.get("name_b", "")).strip().lower()
                    pair_names = {na, nb}
                    if pair_names == {"cryptanalysis", "discrete mathematics"}:
                        decisions.append(
                            {
                                "pair_id": pid,
                                "relation": "skos:broader",
                                "canonical": "discrete mathematics",
                            }
                        )
                    elif pair_names == {"discrete mathematics", "number theory"}:
                        decisions.append(
                            {
                                "pair_id": pid,
                                "relation": "skos:broader",
                                "canonical": "number theory",
                            }
                        )
                    elif pair_names == {"cryptanalysis", "number theory"}:
                        # Both closes a 3-cycle in skos:broader (if broader) or triggers S27 (if related);
                        # emit skos:related to trigger SKOS S27 against the 2-hop broaderTransitive path!
                        decisions.append(
                            {
                                "pair_id": pid,
                                "relation": "skos:related",
                                "canonical": "A",
                            }
                        )
            return [json.dumps({"decisions": decisions})]

        # Stage 2: ExtractTriples (has both `named_entities` input and `triples` output)
        if "triples" in full_text and "named_entities" in full_text:
            if self.mode == "pre_rlm_heal":
                return [
                    json.dumps(
                        {
                            "triples": [
                                {
                                    "sub": ["cryptanalysis", "Study of analyzing information systems"],
                                    "pred": ["applies", "owl:ObjectProperty"],
                                    "obj": ["discrete mathematics", "Study of discrete mathematical structures"],
                                    "def": "Cryptanalysis applies discrete mathematics.",
                                },
                                {
                                    "sub": ["discrete mathematics", "Study of discrete mathematical structures"],
                                    "pred": ["includes", "owl:ObjectProperty"],
                                    "obj": ["number theory", "Branch of pure mathematics devoted to integers"],
                                    "def": "Discrete mathematics includes number theory.",
                                },
                            ]
                        }
                    )
                ]
            return [
                json.dumps(
                    {
                        "triples": [
                            {
                                "sub": ["Yet Another Wiki", "A graph retrieval daemon and wiki system"],
                                "pred": ["launched on", "owl:DatatypeProperty"],
                                "obj": ["2026-10-25", "Release date of Yet Another Wiki"],
                                "def": "Yet Another Wiki was launched on 2026-10-25.",
                            },
                            {
                                "sub": ["Yet Another Wiki", "A graph retrieval daemon and wiki system"],
                                "pred": ["listens on port", "owl:DatatypeProperty"],
                                "obj": ["8080", "Default TCP listen port of Yet Another Wiki"],
                                "def": "Yet Another Wiki listens on port 8080.",
                            },
                            {
                                "sub": [
                                    "security engineering",
                                    "Engineering domain focused on system security",
                                ],
                                "pred": ["branch of", "owl:ObjectProperty"],
                                "obj": ["engineering", "General engineering discipline"],
                                "def": "Security engineering is a branch of engineering.",
                            },
                            {
                                "sub": ["HTTP 2", "Version 2 of HTTP protocol"],
                                "pred": ["version of", "owl:ObjectProperty"],
                                "obj": ["HTTP 3", "Version 3 of HTTP protocol"],
                                "def": "HTTP 2 and HTTP 3 are networking protocol versions.",
                            },
                        ]
                    }
                )
            ]

        # Stage 1: ExtractEntities (`ner`)
        if self.mode == "pre_rlm_heal":
            return [
                json.dumps(
                    {
                        "named_entities": [
                            ["cryptanalysis", "Study of analyzing information systems"],
                            ["discrete mathematics", "Study of discrete mathematical structures"],
                            ["number theory", "Branch of pure mathematics devoted to integers"],
                        ]
                    }
                )
            ]
        return [
            json.dumps(
                {
                    "named_entities": [
                        ["Yet Another Wiki", "A graph retrieval daemon and wiki system"],
                        ["YAW", "Acronym for Yet Another Wiki"],
                        ["security engineering", "Engineering domain focused on system security"],
                        ["engineering", "General engineering discipline"],
                        ["HTTP 2", "Version 2 of HTTP protocol"],
                        ["HTTP 3", "Version 3 of HTTP protocol"],
                    ]
                }
            )
        ]


def _make_engine(tmp_path: Path, *, mode: str = "gate") -> tuple[YawGraphEngine, PipelineGateMockLM]:
    db_path = tmp_path / f"yaw_{mode}.db"
    open_yaw_db(db_path).close()
    mock_lm = PipelineGateMockLM(mode=mode)
    dspy.settings.configure(lm=mock_lm)
    embedder = DummyEmbedder(dim=16)
    prog = HippoRAGProgram(enable_gbnf=False, orchestrator_lm=mock_lm, rlm_sub_lm=mock_lm)
    engine = YawGraphEngine(
        db_path=db_path,
        workspace_root=tmp_path,
        program=prog,
        embedder=embedder,
    )
    return engine, mock_lm


def test_build_time_shacl_gate_success(tmp_path: Path) -> None:
    """Verify that index_chunks executes Stage 1 -> Stage 2 -> Stage 3 -> SHACL build gate end-to-end without
    checkpoint pre-seeding, materializing 0.5-weighted owl:DatatypeProperty literal nodes and healing SKOS edges."""
    engine, mock_lm = _make_engine(tmp_path, mode="gate")
    try:
        text = (
            "Yet Another Wiki (YAW) was launched on 2026-10-25. It listens on port 8080.\n"
            "Security engineering is a specialization of engineering.\n"
            "HTTP 2 and HTTP 3 are networking protocols.\n"
        )
        (tmp_path / "gate.md").write_text(text, encoding="utf-8")
        chunk = ChunkPointer(
            chunk_id="chunk-shacl-gate-1",
            rel_posix_path="gate.md",
            start_line=1,
            end_line=3,
            title="Gate Test Doc",
        )

        # Execute full pipeline end-to-end (no save_openie_checkpoints pre-seeding)
        indexed = engine.index_chunks([(chunk, text)])
        assert indexed == 1

        # Verify Stage 3 JudgeSynonyms was invoked, all 3 adversarial pairs were adjudicated,
        # and pure literals were excluded from candidate pairs
        assert len(mock_lm.observed_synonym_candidates) > 0
        observed_pair_sets = {
            frozenset({str(c.get("name_a", "")).strip().lower(), str(c.get("name_b", "")).strip().lower()})
            for c in mock_lm.observed_synonym_candidates
        }
        assert frozenset({"engineering", "security engineering"}) in observed_pair_sets
        assert frozenset({"http 2", "http 3"}) in observed_pair_sets
        assert frozenset({"yaw", "yet another wiki"}) in observed_pair_sets
        cand_names = {name for pair in observed_pair_sets for name in pair}
        assert "2026-10-25" not in cand_names
        assert "8080" not in cand_names

        with engine.managed_conn.readonly() as conn:
            # 1. Zero SHACL violations exist immediately after index_chunks and Turtle conforms
            violations = lint_graph_shacl(conn, auto_heal=False)
            assert violations == [], f"Build gate leaked SHACL violations: {violations}"
            ttl = serialize_graph_turtle(conn)
            assert "sh:conforms true" in ttl
            assert "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> ." in ttl

            # 2. owl:DatatypeProperty literal objects ("2026-10-25", "8080") ARE materialized in entities
            ent_by_name = {
                str(r[1]).lower(): (str(r[0]), str(r[2]))
                for r in conn.execute("SELECT entity_id, name, definition FROM entities").fetchall()
            }
            assert "2026-10-25" in ent_by_name
            assert "8080" in ent_by_name
            assert "yet another wiki" in ent_by_name
            assert "security engineering" in ent_by_name
            assert "engineering" in ent_by_name
            assert "http 2" in ent_by_name
            assert "http 3" in ent_by_name

            # Acronym "yaw" must have been merged into "yet another wiki" as skos:altLabel
            assert "yaw" not in ent_by_name
            yaw_id = ent_by_name["yet another wiki"][0]
            yaw_labels = {
                (str(r[0]), str(r[1]))
                for r in conn.execute(
                    "SELECT label_norm, label_kind FROM entity_labels WHERE entity_id = ?",
                    (yaw_id,),
                ).fetchall()
            }
            assert ("yet another wiki", "skos:prefLabel") in yaw_labels
            assert ("yaw", "skos:altLabel") in yaw_labels

            date_id = ent_by_name["2026-10-25"][0]
            port_id = ent_by_name["8080"][0]
            sec_eng_id = ent_by_name["security engineering"][0]
            eng_id = ent_by_name["engineering"][0]
            h2_id = ent_by_name["http 2"][0]
            h3_id = ent_by_name["http 3"][0]

            # Passage edges connect the chunk to both literal nodes
            passage_pairs = {
                (str(r[0]), str(r[1]))
                for r in conn.execute("SELECT src_id, dst_id FROM edges WHERE kind = 'passage'").fetchall()
            }
            assert (chunk.chunk_id, date_id) in passage_pairs
            assert (chunk.chunk_id, port_id) in passage_pairs

            # Fact edges from subject to literal nodes have weight == 0.5, while ObjectProperty fact edges have 1.0
            fact_edges = {
                (str(r[0]), str(r[1])): float(r[2])
                for r in conn.execute("SELECT src_id, dst_id, weight FROM edges WHERE kind = 'fact'").fetchall()
            }
            assert abs(fact_edges[(yaw_id, date_id)] - 0.5) < 1e-6
            assert abs(fact_edges[(yaw_id, port_id)] - 0.5) < 1e-6
            assert abs(fact_edges[(sec_eng_id, eng_id)] - 1.0) < 1e-6
            assert abs(fact_edges[(h2_id, h3_id)] - 1.0) < 1e-6

            # Facts table records obj_kind == 'literal' and obj_id IS NOT NULL for both DatatypeProperty facts
            lit_facts = conn.execute(
                "SELECT pred, obj_kind, obj_id, obj_literal_lex FROM facts WHERE obj_kind = 'literal' ORDER BY pred ASC"
            ).fetchall()
            assert len(lit_facts) == 2
            assert {str(r[2]) for r in lit_facts} == {date_id, port_id}
            assert {str(r[3]) for r in lit_facts} == {"2026-10-25", "8080"}

            # Literal nodes have zero skos:broader or skos:related edges
            lit_skos_edges = conn.execute(
                """
                SELECT src_id, dst_id, kind FROM edges
                WHERE kind IN ('skos:broader', 'skos:related')
                  AND (src_id IN (?, ?) OR dst_id IN (?, ?))
                """,
                (date_id, port_id, date_id, port_id),
            ).fetchall()
            assert lit_skos_edges == []

            # 3. Verify security engineering -> engineering has properly oriented skos:broader edge
            broader_edges = {
                (str(r[0]), str(r[1]))
                for r in conn.execute("SELECT src_id, dst_id FROM edges WHERE kind = 'skos:broader'").fetchall()
            }
            assert (sec_eng_id, eng_id) in broader_edges
            assert (eng_id, sec_eng_id) not in broader_edges
            assert (h2_id, h3_id) not in broader_edges
            assert (h3_id, h2_id) not in broader_edges

            # 4. Verify http 2 and http 3 have symmetric skos:related edges (and are the ONLY skos:related edges)
            related_edges = {
                (str(r[0]), str(r[1]))
                for r in conn.execute("SELECT src_id, dst_id FROM edges WHERE kind = 'skos:related'").fetchall()
            }
            assert related_edges == {(h2_id, h3_id), (h3_id, h2_id)}

        # 5. Verify YawGraphEngine.retrieve (PPR) traverses from the subject through the 0.5-weighted
        # owl:DatatypeProperty literal edge to rank the chunk
        csr = engine._ensure_csr_graph()
        assert float(csr.transition_t[csr.node_to_idx[port_id], csr.node_to_idx[yaw_id]]) > 0.0
        assert float(csr.transition_t[csr.node_to_idx[date_id], csr.node_to_idx[yaw_id]]) > 0.0
        assert float(csr.transition_t[csr.node_to_idx[chunk.chunk_id], csr.node_to_idx[port_id]]) > 0.0
        assert float(csr.transition_t[csr.node_to_idx[chunk.chunk_id], csr.node_to_idx[date_id]]) > 0.0

        ppr_from_subj = run_ppr(csr, {yaw_id: 1.0})
        assert float(ppr_from_subj[csr.node_to_idx[port_id]]) > 0.0
        assert float(ppr_from_subj[csr.node_to_idx[date_id]]) > 0.0
        ppr_from_lit = run_ppr(csr, {port_id: 1.0})
        assert float(ppr_from_lit[csr.node_to_idx[chunk.chunk_id]]) > 0.0

        hits, kept_facts = engine.retrieve_with_seeds("What port does Yet Another Wiki listen on?", top_k=1)
        assert ("yet another wiki", "listens on port", "8080") in kept_facts
        assert len(hits) == 1
        assert hits[0][0].chunk_id == chunk.chunk_id
        assert "8080" in hits[0][2]

        retrieve_hits = engine.retrieve("What port does Yet Another Wiki listen on?", top_k=1)
        assert len(retrieve_hits) == 1
        assert retrieve_hits[0][0].chunk_id == chunk.chunk_id
        assert "8080" in retrieve_hits[0][2]
    finally:
        engine.close()


def test_pre_rlm_deterministic_auto_heal_skips_rlm_healer(tmp_path: Path) -> None:
    """Verify that when deterministic auto-healing run before RLM healing resolves all structural
    SHACL violations (such as SKOS S27 broaderTransitive/related conflicts), shacl_healer_tokens == 0."""
    engine, mock_lm = _make_engine(tmp_path, mode="pre_rlm_heal")
    try:
        text = (
            "Cryptanalysis applies discrete mathematics.\n"
            "Discrete mathematics includes number theory.\n"
        )
        (tmp_path / "math.md").write_text(text, encoding="utf-8")
        chunk = ChunkPointer(
            chunk_id="chunk-pre-rlm-1",
            rel_posix_path="math.md",
            start_line=1,
            end_line=2,
            title="Mathematics Hierarchy",
        )

        indexed = engine.index_chunks([(chunk, text)])
        assert indexed == 1

        # Structural S27 violation existed prior to deterministic auto-heal
        assert int(engine._graph_rlm_stats.get("shacl_violations_before", 0)) >= 1
        # Deterministic auto-heal resolved it before RLM healing, so RLM healer was never invoked
        assert int(engine._graph_rlm_stats.get("shacl_healer_tokens", 0)) == 0
        assert float(engine._chunk_rlm_stats[chunk.chunk_id].get("shacl_healer_tokens", 0.0)) == 0.0
        assert mock_lm.rlm_healer_called is False
        assert len(engine.program.shacl_audit_trail) == 0
        assert int(engine._graph_rlm_stats.get("shacl_violations_after", 0)) == 0

        with engine.managed_conn.readonly() as conn:
            assert lint_graph_shacl(conn, auto_heal=False) == []
    finally:
        engine.close()

