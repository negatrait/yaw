"""NER owns resource discovery; statements may add literal values only."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from yaw.daemon.graph.inference import YawProgram, create_dspy_lm
from yaw.daemon.graph.inference.grammars import build_dspy_triples_grammar
from yaw.daemon.graph.inference.parsers import extract_structured_triples

RESOURCES = (("Ada", "A named individual who wrote notes."), ("mathematician", "A general category of scholars."))


def statement(obj, kind="owl:ObjectProperty"):
    return {"sub": "Ada", "pred": ["has value", kind], "obj": obj, "def": "Ada has the stated value."}


@pytest.mark.parametrize("obj", ["Charles", ["Charles", "A computer pioneer."]])
def test_triples_cannot_discover_resources(obj):
    with pytest.raises(ValueError, match="absent from NER"):
        extract_structured_triples(
            [statement(obj)], entity_definitions=RESOURCES, known_entities=[name for name, _ in RESOURCES]
        )


def test_literal_is_discovered_without_becoming_an_entity():
    literal = {"value": "English", "datatype": "xsd:string", "lang": ""}
    (triple,) = extract_structured_triples(
        [statement(literal, "owl:DatatypeProperty")],
        entity_definitions=RESOURCES,
        known_entities=[name for name, _ in RESOURCES],
    )
    assert (triple.obj, triple.obj_kind, triple.obj_datatype) == ("English", "literal", "xsd:string")
    assert triple.subj_def == RESOURCES[0][1]
    assert triple.obj_def == ""


def test_known_resource_definition_cannot_be_redefined_by_triples():
    raw = {**statement("mathematician"), "subjDef": "Invented replacement.", "objDef": "Another replacement."}
    (triple,) = extract_structured_triples(
        [raw], entity_definitions=RESOURCES, known_entities=[name for name, _ in RESOURCES]
    )
    assert (triple.subj_def, triple.obj_def) == (RESOURCES[0][1], RESOURCES[1][1])


def test_predicate_type_cannot_disguise_a_new_resource_as_a_literal():
    with pytest.raises(ValueError, match="property type conflicts"):
        extract_structured_triples(
            [statement({"value": "English", "datatype": "xsd:string", "lang": ""})],
            entity_definitions=RESOURCES,
            known_entities=[name for name, _ in RESOURCES],
        )


def test_homograph_senses_must_already_exist_in_ner():
    senses = (("ball", "A spherical toy."), ("ball", "A formal dance."))
    raw = {"sub": list(senses[0]), "pred": "appears at", "obj": list(senses[1]), "def": "A toy appears at a dance."}
    (triple,) = extract_structured_triples([raw], entity_definitions=senses, known_entities=["ball"])
    assert (triple.subj_def, triple.obj_def) == (senses[0][1], senses[1][1])
    with pytest.raises(ValueError, match="existing NER resources"):
        extract_structured_triples(
            [{**raw, "obj": ["ball", "An invented third sense."]}],
            entity_definitions=senses,
            known_entities=["ball"],
        )


def test_default_demonstrations_obey_resource_ownership():
    from yaw.daemon.graph.inference.prompts import TRIPLE_DEMOS

    for demo in TRIPLE_DEMOS:
        triples = extract_structured_triples(
            demo.triples,
            entity_definitions=demo.named_entities,
            known_entities=[name for name, _ in demo.named_entities],
        )
        assert len(triples) == len(demo.triples)
        assert any(triple.obj_kind == "literal" for triple in triples)


def test_grammar_has_no_new_resource_slot_and_empty_ner_has_no_statements():
    grammar = build_dspy_triples_grammar(known_entities=RESOURCES)
    assert "entity-slot ::= known-entity\n" in grammar
    assert "new-entity" not in grammar
    assert "literal-slot" in grammar
    assert build_dspy_triples_grammar(known_entities=[]) == 'root ::= "{\\"triples\\": []}"\n'


def test_two_stage_resource_handoff_over_real_http():
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            prompt = str(request["messages"][-1]["content"])
            if "[[ ## missing_entities ## ]]" in prompt:
                result = {"entity_definitions": [["Charles", "An unrequested computer pioneer."]]}
                calls.append("definition_repair")
            elif "[[ ## named_entities ## ]]" in prompt:
                result = {
                    "triples": [
                        statement("mathematician"),
                        statement({"value": "37", "datatype": "xsd:integer", "lang": ""}, "owl:DatatypeProperty"),
                    ]
                }
                calls.append("triples")
            else:
                result = {"named_entities": list(RESOURCES)}
                calls.append("ner")
            raw = json.dumps(
                {
                    "id": "roles",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "roles",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": json.dumps(result),
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, format, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            lm = create_dspy_lm("roles", base_url=f"http://127.0.0.1:{server.server_port}/v1", cache=False)
            program = YawProgram(orchestrator_lm=lm, rlm_sub_lm=lm, enable_gbnf=False)
            record = program.extract_openie("chunk", "Ada was a mathematician who wrote 37 notes.")
            assert calls == ["ner", "triples"]
            assert record.entities == ("Ada", "mathematician")
            assert record.entity_definitions == RESOURCES
            assert [triple.obj_kind for triple in record.structured_triples] == ["entity", "literal"]
            with pytest.raises(ValueError, match="unrequested resource"):
                program.define_missing_entities("Ada wrote notes.", ["Ada"])
            assert calls[-1] == "definition_repair"
        finally:
            server.shutdown()
            thread.join()
