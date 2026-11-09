"""Real HTTP invariants for native predictor binding, copying and contracts."""

import asyncio
import copy
import inspect
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import dspy
import pytest
from dspy.utils.exceptions import AdapterParseError

from yaw.core import VALID_SHACL_ACTIONS
from yaw.daemon.graph.inference import RLMInstructionProposer, YawProgram, create_dspy_lm
from yaw.daemon.graph.inference.adapter import YawGBNFAdapter
from yaw.daemon.graph.inference.prompts import (
    AnswerQuestion,
    DefineMissingEntities,
    ExtractEntities,
    ExtractQueryIntent,
    ExtractTriples,
    FilterTriples,
    HealSHACLViolations,
    JudgeHierarchy,
    JudgeLabels,
    JudgeRelated,
    ProposeSHACLRepairs,
    ReflectiveInnerDistillerSignature,
)


@pytest.fixture
def provider():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(request)
            body = json.dumps(
                {
                    "id": "native-binding",
                    "object": "chat.completion",
                    "model": request["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": json.dumps(
                                    {
                                        "named_entities": [["Widget", "An object described in the source."]],
                                        "reasoning": "Supported by the source.",
                                        "answer": "Widget",
                                        "query_intent": "Widget is an object",
                                    }
                                ),
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/v1", requests
        finally:
            server.shutdown()
            thread.join(timeout=5)


def lm_for(url, name="bound", **kwargs):
    return create_dspy_lm(name, base_url=url, api_key="test", cache=False, **kwargs)


def test_native_binding_and_copy_send_actual_generation_overrides(provider):
    url, requests = provider
    lm = lm_for(url, temperature=0.1, max_tokens=77)
    program = YawProgram(orchestrator_lm=lm, rlm_sub_lm=lm)
    assert program.ner.lm is lm
    assert program.get_lm() is lm
    copied = program.ner.lm.copy(temperature=0.9, max_tokens=123)
    with dspy.context(adapter=program.json_adapter):
        result = program.ner(passage="Widget is an object.", lm=copied)
    assert result.named_entities[0][0] == "Widget"
    assert requests[-1]["temperature"] == 0.9
    assert requests[-1].get("max_completion_tokens", requests[-1].get("max_tokens")) == 123
    assert lm.kwargs["temperature"] == 0.1
    assert lm.kwargs["max_tokens"] == 77


def test_native_predict_acall_reaches_real_provider(provider):
    url, requests = provider
    lm = lm_for(url)
    program = YawProgram(orchestrator_lm=lm, rlm_sub_lm=lm)

    async def invoke():
        with dspy.context(adapter=program.json_adapter, disable_history=True, track_usage=True):
            return await program.ner.acall(passage="Widget is an object.")

    result = asyncio.run(invoke())
    assert result.named_entities[0][0] == "Widget"
    assert len(requests) == 1
    assert not lm.history


def test_native_precedence_per_call_then_bound_then_context(provider):
    url, requests = provider
    bound = lm_for(url, "bound")
    ambient = lm_for(url, "ambient")
    override = lm_for(url, "override")
    program = YawProgram(orchestrator_lm=bound, rlm_sub_lm=bound)
    with dspy.context(lm=ambient, adapter=program.json_adapter):
        program.ner(passage="first")
        program.ner(passage="second", lm=override)
        program.extract_entities("third", lm=override)
        program.answer_question(context="Widget is an object.", question="What is it?", lm=override)
    assert [request["model"] for request in requests] == ["bound", "override", "override", "override"]
    assert program.ner.lm is bound


def test_unbound_program_uses_context_set_after_construction(provider):
    url, requests = provider
    program = YawProgram()
    assert all(predictor.lm is None for _, predictor in program.named_predictors())
    with dspy.context(lm=lm_for(url, "contextual"), adapter=program.json_adapter):
        assert program.extract_entities("Widget is an object.") == ("Widget",)
    assert requests[-1]["model"] == "contextual"


def test_task_roles_and_query_intent_override_keep_models_independent(provider):
    url, requests = provider
    qa = lm_for(url, "qa")
    extraction = lm_for(url, "extraction")
    override = lm_for(url, "override")
    program = YawProgram(orchestrator_lm=qa, rlm_sub_lm=extraction)
    with dspy.context(lm=override):
        program.extract_entities("Widget is an object.")
        program.answer_question(context="Widget is an object.", question="What is it?")
        assert program.extract_query_intent("What is Widget?", lm=override) == "Widget is an object"
    assert [request["model"] for request in requests] == ["extraction", "qa", "override"]


def test_candidate_deepcopy_preserves_native_lm_copy_settings(provider):
    url, requests = provider
    lm = lm_for(url, temperature=0.1, max_tokens=77)
    program = YawProgram(orchestrator_lm=lm, rlm_sub_lm=lm)
    program.ner.set_lm(lm.copy(temperature=0.8, max_tokens=119))
    candidate = copy.deepcopy(program)
    candidate.extract_entities("Widget is an object.")
    assert requests[-1]["temperature"] == 0.8
    assert requests[-1].get("max_completion_tokens", requests[-1].get("max_tokens")) == 119
    assert program.qa.predict.lm.kwargs["temperature"] == 0.1


def test_native_state_roundtrip_keeps_demos_and_explicit_deployment_lms(tmp_path, provider):
    url, requests = provider
    lm = lm_for(url)
    program = YawProgram(orchestrator_lm=lm, rlm_sub_lm=lm)
    program.ner.demos = [dspy.Example(passage="A source", named_entities=[["A", "An example concept."]])]
    program.qa.predict.signature = program.qa.predict.signature.with_instructions("Use only supplied evidence.")
    path = tmp_path / "compiled.json"
    program.save(str(path), save_program=False)
    replacement = lm_for(url, "replacement")
    restored = YawProgram(orchestrator_lm=replacement, rlm_sub_lm=replacement)
    restored.load(str(path))
    assert restored.ner.demos[0]["named_entities"] == [["A", "An example concept."]]
    assert restored.qa.predict.signature.instructions == "Use only supplied evidence."
    assert restored.get_lm() is replacement
    restored.extract_entities("Widget is an object.")
    assert requests[-1]["model"] == "replacement"


def test_fresh_programs_are_zero_shot():
    program = YawProgram()
    proposer = RLMInstructionProposer()
    assert all(not predictor.demos for _, predictor in program.named_predictors())
    assert not proposer.llm_proposer.demos and not proposer.llm_compressor.demos
    assert all(not predictor.demos for _, predictor in proposer.inner_rlm.named_predictors())


def test_legacy_state_missing_a_predictor_uses_native_configured_defaults(provider):
    url, _requests = provider
    lm = lm_for(url)
    program = YawProgram(orchestrator_lm=lm, rlm_sub_lm=lm)
    state = program.dump_state()
    del state["entity_definer"]
    restored = YawProgram(orchestrator_lm=lm, rlm_sub_lm=lm)
    restored.load_state(state)
    assert restored.entity_definer.lm is lm
    assert not restored.entity_definer.demos


@pytest.mark.parametrize("payload", ['{"reasoning":"Incomplete"}', '{"answer":"Incomplete"}', "{}"])
def test_repaired_multifield_responses_must_satisfy_complete_signature(payload):
    signature = YawProgram().qa.predict.signature
    with pytest.raises(AdapterParseError):
        YawGBNFAdapter().parse(signature, payload)


def test_repaired_multifield_response_preserves_all_typed_outputs():
    signature = YawProgram().qa.predict.signature
    result = YawGBNFAdapter().parse(signature, "{'reasoning': 'Evidence', 'answer': 'Widget'}")
    assert result == {"reasoning": "Evidence", "answer": "Widget"}


def test_native_parser_rejects_wrong_field_names_types_and_bare_arrays():
    class Count(dspy.Signature):
        number: int = dspy.OutputField()

    with pytest.raises(AdapterParseError):
        YawGBNFAdapter().parse(Count, '{"number": "not an integer"}')
    with pytest.raises(AdapterParseError):
        YawGBNFAdapter().parse(Count, '{"alternate": 42}')
    with pytest.raises(AdapterParseError):
        YawGBNFAdapter().parse(ExtractEntities, '[["Widget", "An object."]]')
    assert inspect.unwrap(YawGBNFAdapter.parse) is inspect.unwrap(dspy.JSONAdapter.parse)


def test_history_remains_explicit_conversation_input_not_diagnostic_storage():
    class Conversation(dspy.Signature):
        question: str = dspy.InputField()
        history: dspy.History = dspy.InputField()
        answer: str = dspy.OutputField()

    history = dspy.History(messages=[{"question": "Which object?", "answer": "Widget"}])
    messages = YawGBNFAdapter().format(Conversation, [], {"question": "What was its name?", "history": history})
    assert [message["role"] for message in messages] == ["system", "user", "assistant", "user"]
    assert "Widget" in messages[-2]["content"]
    assert len(history.messages) == 1


def test_typed_schema_contracts_exported_on_prompts_and_inference_facade():
    import yaw.daemon.graph.inference as inf_pkg
    import yaw.daemon.graph.inference.prompts as prompts_mod

    expected_types = (
        "EntitySensePair",
        "FactTriple",
        "ObjectPropertyPredicate",
        "DatatypePropertyPredicate",
        "LiteralObject",
        "ObjectPropertyStatement",
        "DatatypePropertyStatement",
        "ReifiedStatement",
        "EquivalentLabelDecision",
        "DistinctLabelDecision",
        "LabelDecision",
        "HierarchyDecision",
        "RelatedDecision",
        "SHACLRepairAction",
        "SHACLRepair",
        "DistillerReference",
        "DistillerSelection",
    )
    for sym in expected_types:
        assert sym in prompts_mod.__all__, f"{sym} missing from prompts.__all__"
        assert sym in inf_pkg.__all__, f"{sym} missing from inference.__all__"
        assert getattr(prompts_mod, sym) is getattr(inf_pkg, sym)


def test_predictor_signatures_emit_strict_json_schema_hints():
    adapter = YawGBNFAdapter()
    all_signatures = (
        ExtractEntities,
        ExtractTriples,
        DefineMissingEntities,
        JudgeLabels,
        JudgeHierarchy,
        JudgeRelated,
        ProposeSHACLRepairs,
        HealSHACLViolations,
        FilterTriples,
        ExtractQueryIntent,
        AnswerQuestion,
        ReflectiveInnerDistillerSignature,
    )
    for sig in all_signatures:
        rendered = adapter.format_field_structure(sig).replace('\\"', '"')
        assert '"items": {}' not in rendered, f"{sig.__name__} has untyped items: {rendered}"
        assert '"additionalProperties": true' not in rendered, (
            f"{sig.__name__} has open dict additionalProperties: {rendered}"
        )

    object_schema_signatures = (
        ExtractTriples,
        JudgeLabels,
        JudgeHierarchy,
        JudgeRelated,
        ProposeSHACLRepairs,
        HealSHACLViolations,
        ReflectiveInnerDistillerSignature,
    )
    for sig in object_schema_signatures:
        rendered = adapter.format_field_structure(sig).replace('\\"', '"')
        assert '"additionalProperties": false' in rendered, (
            f"{sig.__name__} missing closed-world additionalProperties=false: {rendered}"
        )

    ner_schema = adapter.format_field_structure(ExtractEntities).replace('\\"', '"')
    assert '"prefixItems"' in ner_schema
    assert '"minItems": 2' in ner_schema and '"maxItems": 2' in ner_schema

    definer_schema = adapter.format_field_structure(DefineMissingEntities).replace('\\"', '"')
    assert '"prefixItems"' in definer_schema
    assert '"minItems": 2' in definer_schema and '"maxItems": 2' in definer_schema

    filter_schema = adapter.format_field_structure(FilterTriples).replace('\\"', '"')
    assert '"prefixItems"' in filter_schema
    assert '"minItems": 3' in filter_schema and '"maxItems": 3' in filter_schema

    triples_schema = adapter.format_field_structure(ExtractTriples).replace('\\"', '"')
    for token in (
        "ObjectPropertyStatement",
        "DatatypePropertyStatement",
        "LiteralObject",
        "owl:ObjectProperty",
        "owl:DatatypeProperty",
        '"sub"',
        '"pred"',
        '"obj"',
        '"def"',
    ):
        assert token in triples_schema, f"{token} missing from ExtractTriples schema: {triples_schema}"

    label_schema = adapter.format_field_structure(JudgeLabels).replace('\\"', '"')
    for token in (
        "EquivalentLabelDecision",
        "DistinctLabelDecision",
        '"pair_id"',
        '"reasoning"',
        '"equivalent"',
        '"definition"',
        '"hidden_label"',
    ):
        assert token in label_schema, f"{token} missing from JudgeLabels schema: {label_schema}"

    hierarchy_schema = adapter.format_field_structure(JudgeHierarchy).replace('\\"', '"')
    for token in ("HierarchyDecision", '"pair_id"', '"reasoning"', '"broader"', '"narrower"'):
        assert token in hierarchy_schema

    related_schema = adapter.format_field_structure(JudgeRelated).replace('\\"', '"')
    for token in ("RelatedDecision", '"pair_id"', '"reasoning"', '"related"'):
        assert token in related_schema

    for shacl_sig in (ProposeSHACLRepairs, HealSHACLViolations):
        shacl_schema = adapter.format_field_structure(shacl_sig).replace('\\"', '"')
        for token in ("SHACLRepair", '"focus_node"', '"action"', '"value"', *VALID_SHACL_ACTIONS):
            assert token in shacl_schema, f"{token} missing from {shacl_sig.__name__} schema"

    distiller_schema = adapter.format_field_structure(ReflectiveInnerDistillerSignature).replace('\\"', '"')
    for token in ("DistillerSelection", "DistillerReference", '"diagnosis"', '"references"', '"example_id"', '"path"'):
        assert token in distiller_schema


def test_predictor_signatures_parse_valid_and_reject_invalid_payloads():
    adapter = YawGBNFAdapter()

    assert adapter.parse(ExtractEntities, '{"named_entities": [["Radio City", "Private FM station"]]}') == {
        "named_entities": [("Radio City", "Private FM station")]
    }
    assert adapter.parse(DefineMissingEntities, '{"entity_definitions": [["Radio City", "Private FM station"]]}') == {
        "entity_definitions": [("Radio City", "Private FM station")]
    }
    assert adapter.parse(FilterTriples, '{"fact": [["Radio City", "located in", "India"]]}') == {
        "fact": [("Radio City", "located in", "India")]
    }
    triples_parsed = adapter.parse(
        ExtractTriples,
        json.dumps(
            {
                "triples": [
                    {
                        "sub": "Radio City",
                        "pred": ["started on", "owl:DatatypeProperty"],
                        "obj": {"value": "2001-07-03", "datatype": "xsd:date", "lang": ""},
                        "def": "Radio City started on 3 July 2001.",
                    }
                ]
            }
        ),
    )
    assert triples_parsed["triples"][0]["pred"] == ("started on", "owl:DatatypeProperty")

    assert (
        adapter.parse(
            JudgeLabels,
            json.dumps(
                {
                    "decisions": [
                        {
                            "pair_id": "e0|e1",
                            "reasoning": "Same entity",
                            "equivalent": True,
                            "definition": "Unified def",
                            "hidden_label": False,
                        },
                        {"pair_id": "e1|e2", "reasoning": "Distinct", "equivalent": False},
                    ]
                }
            ),
        )["decisions"][0]["equivalent"]
        is True
    )

    assert (
        adapter.parse(
            JudgeHierarchy,
            '{"decisions": [{"pair_id": "e0|e1", "reasoning": "Subtype", "broader": true, "narrower": false}]}',
        )["decisions"][0]["broader"]
        is True
    )

    assert (
        adapter.parse(
            JudgeRelated,
            '{"decisions": [{"pair_id": "e0|e1", "reasoning": "Related", "related": true}]}',
        )["decisions"][0]["related"]
        is True
    )

    for shacl_sig in (ProposeSHACLRepairs, HealSHACLViolations):
        assert adapter.parse(
            shacl_sig,
            '{"repairs": [{"focus_node": "ent-1", "action": "set_definition", "value": "Updated definition."}]}',
        ) == {"repairs": [{"focus_node": "ent-1", "action": "set_definition", "value": "Updated definition."}]}

    assert (
        adapter.parse(
            ReflectiveInnerDistillerSignature,
            json.dumps(
                {
                    "selection": {
                        "diagnosis": "Missing bridge",
                        "references": [{"example_id": "ex-1", "path": ["Feedback"], "start": 0, "end": 10}],
                    }
                }
            ),
        )["selection"]["diagnosis"]
        == "Missing bridge"
    )

    invalid_cases = [
        (ExtractEntities, '{"named_entities": [["OnlyLabel"]]}'),
        (ExtractEntities, '{"named_entities": [["A", "B", "C"]]}'),
        (DefineMissingEntities, '{"entity_definitions": [["OnlyName"]]}'),
        (ExtractTriples, '{"triples": [{"sub": "A", "pred": "p", "obj": "B"}]}'),
        (
            ExtractTriples,
            '{"triples": [{"sub": "A", "pred": ["p", "owl:InvalidProperty"], "obj": "B", "def": "A p B."}]}',
        ),
        (
            ExtractTriples,
            (
                '{"triples": [{"sub": "A", "pred": ["p", "owl:DatatypeProperty"], '
                '"obj": {"value": "1"}, "def": "A p 1."}]}'
            ),
        ),
        (
            ExtractTriples,
            (
                '{"triples": [{"sub": "A", "pred": ["started on", "owl:DatatypeProperty"], '
                '"obj": "2001-07-03", "def": "A started on 2001-07-03."}]}'
            ),
        ),
        (
            ExtractTriples,
            (
                '{"triples": [{"sub": "A", "pred": ["located in", "owl:ObjectProperty"], '
                '"obj": {"value": "India", "datatype": "xsd:string", "lang": ""}, "def": "A located in India."}]}'
            ),
        ),
        (JudgeLabels, '{"decisions": [{"pair_id": "e0|e1", "reasoning": "Missing equivalent"}]}'),
        (JudgeLabels, '{"decisions": [{"pair_id": "a|b", "reasoning": "r", "equivalent": true}]}'),
        (
            JudgeLabels,
            '{"decisions": [{"pair_id": "a|b", "reasoning": "r", "equivalent": false, "hidden_label": false}]}',
        ),
        (JudgeHierarchy, '{"decisions": [{"pair_id": "e0|e1", "reasoning": "Missing narrower", "broader": true}]}'),
        (JudgeRelated, '{"decisions": [{"pair_id": "e0|e1", "reasoning": "Missing related"}]}'),
        (
            JudgeRelated,
            '{"decisions": [{"pair_id": "e0|e1", "reasoning": "Extra field", "related": true, "extra": 1}]}',
        ),
        (ProposeSHACLRepairs, '{"repairs": [{"focus_node": "ent-1", "action": "invent_action", "value": "x"}]}'),
        (HealSHACLViolations, '{"repairs": [{"focus_node": "ent-1", "action": "invent_action", "value": "x"}]}'),
        (FilterTriples, '{"fact": [["Subj", "Pred"]]}'),
        (FilterTriples, '{"fact": [["S", "P", "O", "Extra"]]}'),
        (ReflectiveInnerDistillerSignature, '{"selection": {"diagnosis": "Missing references"}}'),
    ]
    for sig, bad_payload in invalid_cases:
        with pytest.raises(AdapterParseError):
            adapter.parse(sig, bad_payload)


def test_signature_instructions_and_stage_contracts_avoid_schema_repetition_and_validate_compressor():
    from yaw.daemon.graph.inference.invariants import STAGE_SCHEMA_CONTRACTS, validate_stage_invariants

    assert "[skos:prefLabel, skos:definition]" not in ExtractEntities.instructions
    extra = ExtractEntities.output_fields["named_entities"].json_schema_extra
    assert isinstance(extra, dict)
    desc = extra.get("desc")
    assert isinstance(desc, str) and "[skos:prefLabel, skos:definition]" not in desc
    assert "[skos:prefLabel, skos:definition]" not in STAGE_SCHEMA_CONTRACTS["ner"][2]
    assert "fact_filter" not in STAGE_SCHEMA_CONTRACTS
    for stage_key in ("qa", "query_intent", "shacl_healer_rlm"):
        assert stage_key in STAGE_SCHEMA_CONTRACTS

    assert validate_stage_invariants(
        "llm_compressor", {}, dspy.Prediction(shortened_instruction="Concise instruction.")
    ) == (1.0, "PASS [llm_compressor]")
    comp_empty_score, comp_empty_fb = validate_stage_invariants(
        "llm_compressor", {}, dspy.Prediction(shortened_instruction="")
    )
    assert comp_empty_score == 0.0
    assert "shortened_instruction" in comp_empty_fb

    prop_score, prop_fb = validate_stage_invariants(
        "llm_proposer",
        {"component_name": "qa.predict", "current_instruction": "Return answer"},
        dspy.Prediction(new_instruction="Unrelated text"),
    )
    assert prop_score < 1.0
    assert "answer" in prop_fb
