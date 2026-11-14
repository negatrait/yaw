"""Real-substrate tests for YawProgram two-stage W3C semantic cascade."""

import json
from typing import Any

import dspy

from yaw.core import SynonymEvidenceBatch
from yaw.daemon.graph.inference.diagnostics import DiagnosticHistory
from yaw.daemon.graph.inference.optimizer import (
    label_judge_fingerprint,
    relation_judge_fingerprint,
)
from yaw.daemon.graph.inference.program import YawProgram


class CascadeMockLM(dspy.LM):
    """Native DSPy request/response provider for semantic decision tests."""

    def __init__(self) -> None:
        super().__init__(model="test-cascade-lm", cache=False)
        self.stage_calls = {"label_judge": 0, "relation_judge": 0}
        self.stage_payloads: dict[str, list[list[dict[str, Any]]]] = {"label_judge": [], "relation_judge": []}
        self.label_responses: dict[str, tuple[bool, bool | str]] = {}
        self.relation_responses: dict[str, str] = {}

    @property
    def supported_params(self) -> set[str]:
        return set()

    def __call__(self, prompt: Any = None, **kwargs: Any) -> Any:
        request = prompt
        from dspy.lm15 import Message, Response, TextPart, Usage

        content = "\n".join(
            part.text for message in request.messages for part in message.parts if hasattr(part, "text")
        )
        start = content.index('{"format":')
        packet, _ = json.JSONDecoder().raw_decode(content[start:])
        pairs = packet["pairs"]
        is_label = "LabelDecision" in str(request.system)
        stage = "label_judge" if is_label else "relation_judge"
        self.stage_calls[stage] += 1
        self.stage_payloads[stage].append(pairs)
        decisions = []
        for pair in pairs:
            pid = pair["pair_id"]
            item: dict[str, Any] = {"pair_id": pid, "reasoning": f"Grounded decision for {pid}"}
            if is_label:
                equivalent, hidden = self.label_responses.get(pid, (False, False))
                item["equivalent"] = bool(equivalent or hidden)
                if item["equivalent"]:
                    item["definition"] = hidden if isinstance(hidden, str) else f"Unified definition for {pid}"
                    item["hidden_label"] = hidden is True
            else:
                item["relation"] = self.relation_responses.get(pid, "skos:related")
            decisions.append(item)
        return Response(
            id="test",
            model="test-cascade-lm",
            message=Message(role="assistant", parts=(TextPart(text=json.dumps({"decisions": decisions})),)),
            finish_reason="stop",
            usage=Usage(input_tokens=32, output_tokens=16),
        )


def _make_batch(pairs: list[dict[str, Any]]) -> SynonymEvidenceBatch:
    identifiers = {}
    candidates = []
    for p in pairs:
        pid = p["pair_id"]
        ea, eb = pid.split("|", 1)
        real_a = f"real_{ea}"
        real_b = f"real_{eb}"
        identifiers[ea] = real_a
        identifiers[eb] = real_b
        candidates.append(
            {
                "pair_id": pid,
                "entity_a": real_a,
                "entity_b": real_b,
                "similarity": p.get("dense_sim", 0.8),
            }
        )
    payload = json.dumps(
        {
            "format": "synonym-evidence-v3",
            "pairs": pairs,
            "entities": {k: {"sources": []} for k in identifiers},
            "sources": {},
        }
    )
    return SynonymEvidenceBatch(
        candidates=tuple(candidates),
        payload=payload,
        identifiers=identifiers,
    )


def _make_program(lm: dspy.LM) -> YawProgram:
    return YawProgram(
        orchestrator_lm=lm,
        rlm_sub_lm=lm,
        enable_gbnf=False,
    )


def test_program_named_predictors_and_audit_trails() -> None:
    """Verify YawProgram exposes 3 separate judges and per-stage DiagnosticHistory trails."""
    lm = CascadeMockLM()
    dspy.settings.configure(lm=lm)
    program = _make_program(lm)

    named_predictors = dict(program.named_predictors())
    assert "label_judge" in named_predictors
    assert "relation_judge" in named_predictors
    assert "semantic_judge" not in named_predictors

    assert isinstance(program.label_judge_audit_trail, DiagnosticHistory)
    assert isinstance(program.relation_judge_audit_trail, DiagnosticHistory)
    assert not hasattr(program, "synonym_audit_trail")

    trails = program.dump_audit_trails()
    assert "label_judge_audit_trail" in trails
    assert "relation_judge_audit_trail" in trails
    assert "synonym_audit_trail" not in trails


def test_cascade_rule_w1_direct_facts_skips_stage1_to_stage2() -> None:
    """Verify Rule W1 pairs with direct facts bypass Stage 1 and go straight to Stage 2."""
    lm = CascadeMockLM()
    dspy.settings.configure(lm=lm)
    program = _make_program(lm)

    pairs = [
        {
            "pair_id": "e0|e1",
            "direct_facts": [["pilot tenant", "participate in", "test"]],
            "shared_role_facts": [],
        },
    ]
    lm.relation_responses["e0|e1"] = "skos:broader"
    batch = _make_batch(pairs)

    decisions = program.judge_semantics(batch)
    assert len(decisions) == 1
    assert lm.stage_calls["label_judge"] == 0
    assert lm.stage_calls["relation_judge"] == 1

    assert decisions[0]["relation"] == "skos:broader"
    assert decisions[0]["source_id"] == "real_e0"
    assert decisions[0]["target_id"] == "real_e1"


def test_cascade_stage1_truth_table_short_circuiting() -> None:
    """Verify Stage 1 branched truth table and advancing equivalent=False to Stage 2."""
    lm = CascadeMockLM()
    dspy.settings.configure(lm=lm)
    program = _make_program(lm)

    pairs = [
        {"pair_id": "e0|e1", "direct_facts": [], "shared_role_facts": []},
        {"pair_id": "e2|e3", "direct_facts": [], "shared_role_facts": []},
        {"pair_id": "e4|e5", "direct_facts": [], "shared_role_facts": []},
        {"pair_id": "e6|e7", "direct_facts": [], "shared_role_facts": []},
    ]
    # e0|e1: equivalent=True, hidden_label=False -> skos:altLabel with unified definition
    lm.label_responses["e0|e1"] = (True, "Unified Jira issue tracking platform.")
    # e2|e3: equivalent=True, hidden_label=True -> skos:hiddenLabel
    lm.label_responses["e2|e3"] = (False, True)
    # e4|e5: equivalent=True, hidden_label=False -> skos:altLabel
    lm.label_responses["e4|e5"] = (True, False)
    # e6|e7: equivalent=False -> distinct concepts, advances to Stage 2
    lm.label_responses["e6|e7"] = (False, False)
    lm.relation_responses["e6|e7"] = "skos:broader"

    batch = _make_batch(pairs)
    decisions = program.judge_semantics(batch)

    assert lm.stage_calls["label_judge"] == 1
    assert lm.stage_calls["relation_judge"] == 1

    d_equiv = next(d for d in decisions if d["pair_id"] == "real_e0|real_e1")
    assert d_equiv["relation"] == "skos:altLabel"
    assert d_equiv["equivalent"] is True
    assert d_equiv["definition"] == "Unified Jira issue tracking platform."

    d_hid = next(d for d in decisions if d["pair_id"] == "real_e2|real_e3")
    assert d_hid["relation"] == "skos:hiddenLabel"
    assert d_hid["equivalent"] is True

    d_alt2 = next(d for d in decisions if d["pair_id"] == "real_e4|real_e5")
    assert d_alt2["relation"] == "skos:altLabel"
    assert d_alt2["equivalent"] is True

    d_adv = next(d for d in decisions if d["pair_id"] == "real_e6|real_e7")
    assert d_adv["relation"] == "skos:broader"


def test_cascade_stage2_hierarchy_truth_table() -> None:
    """Verify Stage 2 (broader, narrower) truth table including inverse and asymmetry degradation."""
    lm = CascadeMockLM()
    dspy.settings.configure(lm=lm)
    program = _make_program(lm)

    pairs = [
        {"pair_id": "e0|e1", "direct_facts": [], "shared_role_facts": []},
        {"pair_id": "e2|e3", "direct_facts": [], "shared_role_facts": []},
        {"pair_id": "e4|e5", "direct_facts": [], "shared_role_facts": []},
    ]
    # All pass Stage 1 as distinct concepts
    for p in pairs:
        lm.label_responses[str(p["pair_id"])] = (False, False)

    # e0|e1: (True, False) -> A skos:broader B (source=e0, target=e1)
    lm.relation_responses["e0|e1"] = "skos:broader"
    # e2|e3: (False, True) -> B skos:broader A (source=e3, target=e2)
    lm.relation_responses["e2|e3"] = "skos:narrower"
    # e4|e5: (True, True) -> S27 degrade to skos:related
    lm.relation_responses["e4|e5"] = "skos:related"

    batch = _make_batch(pairs)
    decisions = program.judge_semantics(batch)

    assert lm.stage_calls["label_judge"] == 1
    assert lm.stage_calls["relation_judge"] == 1

    d0 = next(d for d in decisions if d["pair_id"] == "real_e0|real_e1")
    assert d0["relation"] == "skos:broader"
    assert d0["source_id"] == "real_e0"
    assert d0["target_id"] == "real_e1"

    d1 = next(d for d in decisions if d["pair_id"] == "real_e2|real_e3")
    assert d1["relation"] == "skos:broader"
    assert d1["source_id"] == "real_e3"
    assert d1["target_id"] == "real_e2"

    d2 = next(d for d in decisions if d["pair_id"] == "real_e4|real_e5")
    assert d2["relation"] == "skos:related"


def test_cascade_rule_w1_w2_stage2_residuals_deterministic_related() -> None:
    """Verify non-hierarchical pairs with direct facts or shared roles resolve to skos:related with 0 Stage-3 calls."""
    lm = CascadeMockLM()
    dspy.settings.configure(lm=lm)
    program = _make_program(lm)

    pairs = [
        {
            "pair_id": "e0|e1",
            "direct_facts": [["pilot", "uses", "plane"]],
            "shared_role_facts": [],
        },
        {
            "pair_id": "e2|e3",
            "direct_facts": [],
            "shared_role_facts": [["cat", "eats", "fish"], ["dog", "eats", "meat"]],
        },
    ]
    # e0|e1 skips Stage 1 to Stage 2 due to direct_facts; e2|e3 goes to Stage 1
    lm.label_responses["e2|e3"] = (False, False)
    # Both are non-hierarchical in Stage 2
    lm.relation_responses["e0|e1"] = "skos:related"
    lm.relation_responses["e2|e3"] = "skos:related"

    batch = _make_batch(pairs)
    decisions = program.judge_semantics(batch)

    assert lm.stage_calls["relation_judge"] == 1

    d0 = next(d for d in decisions if d["pair_id"] == "real_e0|real_e1")
    assert d0["relation"] == "skos:related"

    d1 = next(d for d in decisions if d["pair_id"] == "real_e2|real_e3")
    assert d1["relation"] == "skos:related"


def test_cascade_stage3_related_and_different_from() -> None:
    """Verify Stage 3 resolves residual pairs to skos:related vs owl:differentFrom."""
    lm = CascadeMockLM()
    dspy.settings.configure(lm=lm)
    program = _make_program(lm)

    pairs = [
        {"pair_id": "e0|e1", "direct_facts": [], "shared_role_facts": []},
        {"pair_id": "e2|e3", "direct_facts": [], "shared_role_facts": []},
    ]
    for p in pairs:
        lm.label_responses[str(p["pair_id"])] = (False, False)
        lm.relation_responses[str(p["pair_id"])] = "skos:related"

    lm.relation_responses["e0|e1"] = "skos:related"
    lm.relation_responses["e2|e3"] = "owl:differentFrom"

    batch = _make_batch(pairs)
    decisions = program.judge_semantics(batch)

    assert lm.stage_calls["label_judge"] == 1
    assert lm.stage_calls["relation_judge"] == 1

    d0 = next(d for d in decisions if d["pair_id"] == "real_e0|real_e1")
    assert d0["relation"] == "skos:related"

    d1 = next(d for d in decisions if d["pair_id"] == "real_e2|real_e3")
    assert d1["relation"] == "owl:differentFrom"


def test_per_stage_fingerprint_isolation_on_signature_mutation() -> None:
    """Verify mutating only relation_judge instruction isolates Stage 1 & 2 fingerprints without dead stage caches."""
    import inspect

    lm = CascadeMockLM()
    dspy.settings.configure(lm=lm)
    program = _make_program(lm)

    for fn in (program.judge_semantics, program._run_judge_stage):
        assert set(inspect.signature(fn).parameters).isdisjoint({"stage_" + "cache", "stage_" + "caches"})

    pairs = [
        {"pair_id": "e0|e1", "direct_facts": [], "shared_role_facts": []},
    ]
    lm.label_responses["e0|e1"] = (False, False)
    lm.relation_responses["e0|e1"] = "skos:related"
    lm.relation_responses["e0|e1"] = "skos:related"

    batch = _make_batch(pairs)

    decisions1 = program.judge_semantics(batch)
    assert len(decisions1) == 1
    assert lm.stage_calls["label_judge"] == 1
    assert lm.stage_calls["relation_judge"] == 1

    # Mutate ONLY relation_judge instructions
    cand_prog = program.deepcopy()
    cand_prog.relation_judge.signature = cand_prog.relation_judge.signature.with_instructions(
        "Mutated related judge instructions v2"
    )

    # Check fingerprints: label and hierarchy unchanged, related changed, and batch subset cache_identity preserved
    assert label_judge_fingerprint(cand_prog) == label_judge_fingerprint(program)
    assert relation_judge_fingerprint(cand_prog) != relation_judge_fingerprint(program)
    assert batch.subset(["e0|e1"]).cache_identity == _make_batch(pairs).subset(["e0|e1"]).cache_identity

    cand_prog.set_lm(lm)
    decisions2 = cand_prog.judge_semantics(batch, lm=lm)
    assert decisions2 == decisions1


def test_equivalent_decision_updates_merged_entity_definition() -> None:
    """Verify equivalent=True decision merges duplicate entities and updates entities.definition."""
    from yaw.daemon.graph.planner import apply_synonym_adjudication
    from yaw.daemon.graph.storage import open_yaw_db, upsert_entities

    conn = open_yaw_db(":memory:")
    try:
        upsert_entities(
            conn,
            [
                ("ent-jira-1", "Jira", "Issue tracker used for sprint tasks."),
                ("ent-jira-2", "Jira", "Ticketing system used for change requests."),
            ],
        )
        candidates = [
            {
                "pair_id": "ent-jira-1|ent-jira-2",
                "entity_a": "ent-jira-1",
                "entity_b": "ent-jira-2",
                "similarity": 0.92,
            }
        ]
        cand_meta = {
            "ent-jira-1": ("Jira", "Issue tracker used for sprint tasks."),
            "ent-jira-2": ("Jira", "Ticketing system used for change requests."),
        }
        decisions = [
            {
                "pair_id": "ent-jira-1|ent-jira-2",
                "reasoning": "Both entities represent the same Jira platform across different teams.",
                "equivalent": True,
                "definition": "Unified Jira issue tracking and change management platform.",
                "relation": "skos:altLabel",
            }
        ]
        rlm_stats: dict[str, Any] = {}
        staged_merges, _ = apply_synonym_adjudication(
            conn,
            candidates,
            cand_meta,
            decisions,
            set(),
            rlm_stats=rlm_stats,
        )
        assert len(staged_merges) == 1
        rows = conn.execute("SELECT entity_id, name, definition FROM entities").fetchall()
        assert len(rows) == 1
        assert rows[0][1] == "Jira"
        assert rows[0][2] == "Unified Jira issue tracking and change management platform."
    finally:
        conn.close()
