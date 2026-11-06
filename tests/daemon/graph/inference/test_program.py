"""Real-substrate tests for YawProgram 3-stage W3C semantic cascade."""

import json
from typing import Any

import dspy

from yaw.core import SynonymEvidenceBatch
from yaw.daemon.graph.inference.diagnostics import DiagnosticHistory
from yaw.daemon.graph.inference.optimizer import (
    hierarchy_judge_fingerprint,
    label_judge_fingerprint,
    related_judge_fingerprint,
)
from yaw.daemon.graph.inference.program import YawProgram


class CascadeMockLM(dspy.LM):
    """Deterministic mock LM for testing 3-stage semantic cascade execution."""

    def __init__(self) -> None:
        super().__init__(model="test-cascade-lm")
        self.stage_calls: dict[str, int] = {
            "label_judge": 0,
            "hierarchy_judge": 0,
            "related_judge": 0,
        }
        self.stage_payloads: dict[str, list[list[dict[str, Any]]]] = {
            "label_judge": [],
            "hierarchy_judge": [],
            "related_judge": [],
        }
        self.label_responses: dict[str, tuple[bool, bool | str]] = {}
        self.hierarchy_responses: dict[str, tuple[bool, bool]] = {}
        self.related_responses: dict[str, bool] = {}

    def __call__(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        if getattr(dspy.settings, "usage_tracker", None) is not None:
            dspy.settings.usage_tracker.add_usage(
                self.model,
                {"prompt_tokens": 32, "completion_tokens": 16, "total_tokens": 48},
            )
        req_str = str(prompt or messages or kwargs)
        req_lower = req_str.lower()
        pairs = self._extract_pairs(messages, req_str)

        is_related = "judgerelated" in req_lower or "differentfrom" in req_lower
        is_related = is_related or (
            "related" in req_lower and "narrower" not in req_lower and "alt_label" not in req_lower
        )
        if is_related:
            self.stage_calls["related_judge"] += 1
            self.stage_payloads["related_judge"].append(pairs)
            decisions = []
            for p in pairs:
                pid = p["pair_id"]
                related = self.related_responses.get(pid, False)
                decisions.append(
                    {
                        "pair_id": pid,
                        "reasoning": f"Stage 3 related reasoning for {pid}",
                        "related": related,
                    }
                )
            return [json.dumps({"decisions": decisions})]

        if "judgehierarchy" in req_lower or "narrower" in req_lower:
            self.stage_calls["hierarchy_judge"] += 1
            self.stage_payloads["hierarchy_judge"].append(pairs)
            decisions = []
            for p in pairs:
                pid = p["pair_id"]
                broader, narrower = self.hierarchy_responses.get(pid, (False, False))
                decisions.append(
                    {
                        "pair_id": pid,
                        "reasoning": f"Stage 2 hierarchy reasoning for {pid}",
                        "broader": broader,
                        "narrower": narrower,
                    }
                )
            return [json.dumps({"decisions": decisions})]

        if "judgelabels" in req_lower or "alt_label" in req_lower or "equivalent" in req_lower:
            self.stage_calls["label_judge"] += 1
            self.stage_payloads["label_judge"].append(pairs)
            decisions = []
            for p in pairs:
                pid = p["pair_id"]
                resp = self.label_responses.get(pid, (False, False))
                if isinstance(resp[1], str):
                    decisions.append(
                        {
                            "pair_id": pid,
                            "reasoning": f"Stage 1 label reasoning for {pid}",
                            "equivalent": True,
                            "definition": resp[1],
                        }
                    )
                elif resp[0]:
                    decisions.append(
                        {
                            "pair_id": pid,
                            "reasoning": f"Stage 1 label reasoning for {pid}",
                            "equivalent": False,
                            "alt_label": True,
                        }
                    )
                else:
                    decisions.append(
                        {
                            "pair_id": pid,
                            "reasoning": f"Stage 1 label reasoning for {pid}",
                            "equivalent": False,
                            "alt_label": False,
                            "hidden_label": bool(resp[1]),
                        }
                    )
            return [json.dumps({"decisions": decisions})]

        return ['{"decisions": []}']

    def _extract_pairs(self, messages: list[dict[str, Any]] | None, req_str: str) -> list[dict[str, Any]]:
        import re

        if messages:
            for msg in reversed(messages):
                content = str(msg.get("content", ""))
                m = re.search(r'\{\s*"format"\s*:\s*"synonym-evidence-v2"', content)
                if m:
                    try:
                        payload, _ = json.JSONDecoder().raw_decode(content[m.start() :])
                        parsed_pairs = payload.get("pairs", [])
                        if parsed_pairs:
                            return parsed_pairs
                    except Exception:
                        pass
        pids = list(dict.fromkeys(re.findall(r"e\d+\|e\d+", req_str)))
        return [{"pair_id": pid} for pid in pids]


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
            "format": "synonym-evidence-v2",
            "pairs": pairs,
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
        synonym_token_counter=lambda messages, **_kwargs: len(str(messages)) // 4,
        enable_gbnf=False,
    )


def test_program_named_predictors_and_audit_trails() -> None:
    """Verify YawProgram exposes 3 separate judges and per-stage DiagnosticHistory trails."""
    lm = CascadeMockLM()
    dspy.settings.configure(lm=lm)
    program = _make_program(lm)

    named_predictors = dict(program.named_predictors())
    assert "label_judge" in named_predictors
    assert "hierarchy_judge" in named_predictors
    assert "related_judge" in named_predictors
    assert "synonym_judge" not in named_predictors

    assert isinstance(program.label_judge_audit_trail, DiagnosticHistory)
    assert isinstance(program.hierarchy_judge_audit_trail, DiagnosticHistory)
    assert isinstance(program.related_judge_audit_trail, DiagnosticHistory)
    assert not hasattr(program, "synonym_audit_trail")

    trails = program.dump_audit_trails()
    assert "label_judge_audit_trail" in trails
    assert "hierarchy_judge_audit_trail" in trails
    assert "related_judge_audit_trail" in trails
    assert "synonym_audit_trail" not in trails


def test_cascade_rule_w3_broader_tc_short_circuits() -> None:
    """Verify Rule W3 broader_tc pairs resolve immediately with 0 LLM calls."""
    lm = CascadeMockLM()
    dspy.settings.configure(lm=lm)
    program = _make_program(lm)

    pairs = [
        {"pair_id": "e0|e1", "broader_tc": "a_under_b", "direct_facts": [], "shared_role_facts": []},
        {"pair_id": "e2|e3", "broader_tc": "b_under_a", "direct_facts": [], "shared_role_facts": []},
    ]
    batch = _make_batch(pairs)

    decisions = program.judge_synonyms(batch)
    assert len(decisions) == 2
    assert lm.stage_calls["label_judge"] == 0
    assert lm.stage_calls["hierarchy_judge"] == 0
    assert lm.stage_calls["related_judge"] == 0

    d0 = next(d for d in decisions if d["pair_id"] == "real_e0|real_e1")
    assert d0["relation"] == "skos:broader"
    assert d0["source_id"] == "real_e0"
    assert d0["target_id"] == "real_e1"

    d1 = next(d for d in decisions if d["pair_id"] == "real_e2|real_e3")
    assert d1["relation"] == "skos:broader"
    assert d1["source_id"] == "real_e3"
    assert d1["target_id"] == "real_e2"


def test_cascade_rule_w1_direct_facts_skips_stage1_to_stage2() -> None:
    """Verify Rule W1 pairs with direct facts bypass Stage 1 and go straight to Stage 2."""
    lm = CascadeMockLM()
    dspy.settings.configure(lm=lm)
    program = _make_program(lm)

    pairs = [
        {
            "pair_id": "e0|e1",
            "broader_tc": "",
            "direct_facts": [["pilot tenant", "participate in", "test"]],
            "shared_role_facts": [],
        },
    ]
    lm.hierarchy_responses["e0|e1"] = (True, False)
    batch = _make_batch(pairs)

    decisions = program.judge_synonyms(batch)
    assert len(decisions) == 1
    assert lm.stage_calls["label_judge"] == 0
    assert lm.stage_calls["hierarchy_judge"] == 1
    assert lm.stage_calls["related_judge"] == 0

    assert decisions[0]["relation"] == "skos:broader"
    assert decisions[0]["source_id"] == "real_e0"
    assert decisions[0]["target_id"] == "real_e1"


def test_cascade_stage1_truth_table_short_circuiting() -> None:
    """Verify Stage 1 branched truth table and advancing (False, False) to Stage 2."""
    lm = CascadeMockLM()
    dspy.settings.configure(lm=lm)
    program = _make_program(lm)

    pairs = [
        {"pair_id": "e0|e1", "broader_tc": "", "direct_facts": [], "shared_role_facts": []},
        {"pair_id": "e2|e3", "broader_tc": "", "direct_facts": [], "shared_role_facts": []},
        {"pair_id": "e4|e5", "broader_tc": "", "direct_facts": [], "shared_role_facts": []},
        {"pair_id": "e6|e7", "broader_tc": "", "direct_facts": [], "shared_role_facts": []},
    ]
    # e0|e1: equivalent=True -> skos:altLabel with unified definition
    lm.label_responses["e0|e1"] = (True, "Unified Jira issue tracking platform.")
    # e2|e3: equivalent=False, alt_label=False, hidden_label=True -> skos:hiddenLabel
    lm.label_responses["e2|e3"] = (False, True)
    # e4|e5: equivalent=False, alt_label=True -> skos:altLabel
    lm.label_responses["e4|e5"] = (True, False)
    # e6|e7: equivalent=False, alt_label=False, hidden_label=False -> distinct concepts, advances to Stage 2
    lm.label_responses["e6|e7"] = (False, False)
    lm.hierarchy_responses["e6|e7"] = (True, False)

    batch = _make_batch(pairs)
    decisions = program.judge_synonyms(batch)

    assert lm.stage_calls["label_judge"] == 1
    assert lm.stage_calls["hierarchy_judge"] == 1
    assert lm.stage_calls["related_judge"] == 0

    d_equiv = next(d for d in decisions if d["pair_id"] == "real_e0|real_e1")
    assert d_equiv["relation"] == "skos:altLabel"
    assert d_equiv["equivalent"] is True
    assert d_equiv["definition"] == "Unified Jira issue tracking platform."

    d_hid = next(d for d in decisions if d["pair_id"] == "real_e2|real_e3")
    assert d_hid["relation"] == "skos:hiddenLabel"

    d_alt2 = next(d for d in decisions if d["pair_id"] == "real_e4|real_e5")
    assert d_alt2["relation"] == "skos:altLabel"
    assert d_alt2["equivalent"] is False

    d_adv = next(d for d in decisions if d["pair_id"] == "real_e6|real_e7")
    assert d_adv["relation"] == "skos:broader"


def test_cascade_stage2_hierarchy_truth_table() -> None:
    """Verify Stage 2 (broader, narrower) truth table including inverse and asymmetry degradation."""
    lm = CascadeMockLM()
    dspy.settings.configure(lm=lm)
    program = _make_program(lm)

    pairs = [
        {"pair_id": "e0|e1", "broader_tc": "", "direct_facts": [], "shared_role_facts": []},
        {"pair_id": "e2|e3", "broader_tc": "", "direct_facts": [], "shared_role_facts": []},
        {"pair_id": "e4|e5", "broader_tc": "", "direct_facts": [], "shared_role_facts": []},
    ]
    # All pass Stage 1 as distinct concepts
    for p in pairs:
        lm.label_responses[str(p["pair_id"])] = (False, False)

    # e0|e1: (True, False) -> A skos:broader B (source=e0, target=e1)
    lm.hierarchy_responses["e0|e1"] = (True, False)
    # e2|e3: (False, True) -> B skos:broader A (source=e3, target=e2)
    lm.hierarchy_responses["e2|e3"] = (False, True)
    # e4|e5: (True, True) -> S27 degrade to skos:related
    lm.hierarchy_responses["e4|e5"] = (True, True)

    batch = _make_batch(pairs)
    decisions = program.judge_synonyms(batch)

    assert lm.stage_calls["label_judge"] == 1
    assert lm.stage_calls["hierarchy_judge"] == 1
    assert lm.stage_calls["related_judge"] == 0

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
            "broader_tc": "",
            "direct_facts": [["pilot", "uses", "plane"]],
            "shared_role_facts": [],
        },
        {
            "pair_id": "e2|e3",
            "broader_tc": "",
            "direct_facts": [],
            "shared_role_facts": [["cat", "eats", "fish"], ["dog", "eats", "meat"]],
        },
    ]
    # e0|e1 skips Stage 1 to Stage 2 due to direct_facts; e2|e3 goes to Stage 1
    lm.label_responses["e2|e3"] = (False, False)
    # Both are non-hierarchical in Stage 2
    lm.hierarchy_responses["e0|e1"] = (False, False)
    lm.hierarchy_responses["e2|e3"] = (False, False)

    batch = _make_batch(pairs)
    decisions = program.judge_synonyms(batch)

    assert lm.stage_calls["hierarchy_judge"] == 1
    assert lm.stage_calls["related_judge"] == 0

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
        {"pair_id": "e0|e1", "broader_tc": "", "direct_facts": [], "shared_role_facts": []},
        {"pair_id": "e2|e3", "broader_tc": "", "direct_facts": [], "shared_role_facts": []},
    ]
    for p in pairs:
        lm.label_responses[str(p["pair_id"])] = (False, False)
        lm.hierarchy_responses[str(p["pair_id"])] = (False, False)

    lm.related_responses["e0|e1"] = True
    lm.related_responses["e2|e3"] = False

    batch = _make_batch(pairs)
    decisions = program.judge_synonyms(batch)

    assert lm.stage_calls["label_judge"] == 1
    assert lm.stage_calls["hierarchy_judge"] == 1
    assert lm.stage_calls["related_judge"] == 1

    d0 = next(d for d in decisions if d["pair_id"] == "real_e0|real_e1")
    assert d0["relation"] == "skos:related"

    d1 = next(d for d in decisions if d["pair_id"] == "real_e2|real_e3")
    assert d1["relation"] == "owl:differentFrom"


def test_per_stage_cache_isolation_on_signature_mutation() -> None:
    """Verify mutating only related_judge instruction reuses 100% of Stage 1 & 2 caches."""
    lm = CascadeMockLM()
    dspy.settings.configure(lm=lm)
    program = _make_program(lm)

    # In-memory stage caches simulating CandidateStageCache
    stage_caches: dict[str, dict[tuple[str, str], Any]] = {
        "label_judge": {},
        "hierarchy_judge": {},
        "related_judge": {},
    }

    pairs = [
        {"pair_id": "e0|e1", "broader_tc": "", "direct_facts": [], "shared_role_facts": []},
    ]
    lm.label_responses["e0|e1"] = (False, False)
    lm.hierarchy_responses["e0|e1"] = (False, False)
    lm.related_responses["e0|e1"] = True

    batch = _make_batch(pairs)

    # Run 1: Cold caches -> all 3 stages run live
    decisions1 = program.judge_synonyms(batch, stage_caches=stage_caches)
    assert len(decisions1) == 1
    assert lm.stage_calls["label_judge"] == 1
    assert lm.stage_calls["hierarchy_judge"] == 1
    assert lm.stage_calls["related_judge"] == 1

    assert len(stage_caches["label_judge"]) == 1
    assert len(stage_caches["hierarchy_judge"]) == 1
    assert len(stage_caches["related_judge"]) == 1

    # Mutate ONLY related_judge instructions
    cand_prog = program.deepcopy()
    cand_prog.related_judge.signature = cand_prog.related_judge.signature.with_instructions(
        "Mutated related judge instructions v2"
    )

    # Check fingerprints: label and hierarchy unchanged, related changed
    assert label_judge_fingerprint(cand_prog) == label_judge_fingerprint(program)
    assert hierarchy_judge_fingerprint(cand_prog) == hierarchy_judge_fingerprint(program)
    assert related_judge_fingerprint(cand_prog) != related_judge_fingerprint(program)

    # Run 2 on mutated candidate: Stage 1 and Stage 2 hit cache (0 LLM calls)!
    s1_before = lm.stage_calls["label_judge"]
    s2_before = lm.stage_calls["hierarchy_judge"]
    s3_before = lm.stage_calls["related_judge"]

    # Debug: inspect stage caches and fingerprint
    subset_fp = (related_judge_fingerprint(cand_prog), batch.subset(["e0|e1"]).cache_identity)
    assert subset_fp not in stage_caches["related_judge"]

    cand_prog.set_lm(lm)
    decisions2 = cand_prog.judge_synonyms(batch, stage_caches=stage_caches, lm=lm)
    assert len(decisions2) == 1

    # Stage 1 and Stage 2 did NOT execute LLM calls
    assert lm.stage_calls["label_judge"] == s1_before
    assert lm.stage_calls["hierarchy_judge"] == s2_before
    # Stage 3 executed 1 live LLM call due to mutated fingerprint
    assert lm.stage_calls["related_judge"] == s3_before + 1


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
            synonymy_threshold=0.75,
            rlm_stats=rlm_stats,
        )
        assert len(staged_merges) == 1
        rows = conn.execute("SELECT entity_id, name, definition FROM entities").fetchall()
        assert len(rows) == 1
        assert rows[0][1] == "Jira"
        assert rows[0][2] == "Unified Jira issue tracking and change management platform."
    finally:
        conn.close()
