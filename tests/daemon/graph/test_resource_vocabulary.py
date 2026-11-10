"""Resource vocabulary boundary (native graph spec §5.3) on verbatim boundary-violating model output.

NER alone produces resources (`owl:NamedIndividual`, `skos:Concept`); their labels are the closed
`rdf:subject` vocabulary. GBNF prevents a violation, the parser rejects it and SHACL rejects it in
published state. Every parser-accepted record must normalize, bind and build evidence.
"""

import json
import threading
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from yaw.core import ChunkPointer, OpenIERecord, StructuredTriple, is_resource_label
from yaw.daemon.graph.inference import YawProgram, create_dspy_lm
from yaw.daemon.graph.inference.grammars import (
    DSPY_NER_GBNF_GRAMMAR,
    build_dspy_ner_grammar,
    build_dspy_triples_grammar,
)
from yaw.daemon.graph.inference.parsers import extract_structured_triples
from yaw.daemon.graph.normalization import normalize_openie_record
from yaw.daemon.graph.planner import (
    apply_stage1_definitions,
    collect_missing_entity_norms,
    materialize_entities_and_facts,
    record_surface_form,
)
from yaw.daemon.graph.storage import (
    apply_shacl_repairs,
    canonicalize_entity_labels_edc,
    lint_graph_shacl,
    open_yaw_db,
    upsert_chunk_pointers,
    upsert_entities,
    upsert_entity_labels,
    upsert_fact_sources,
    upsert_facts,
)
from yaw.daemon.graph.storage.evidence import build_evidence

BROKER = [
    "Admission broker",
    "A small, deterministic service colocated with the request router responsible for converting traffic "
    'intent into kernel "slot" requests.',
]

# Verbatim NER outputs from yaw_optimize_20261105_175308.db traces that broke graph publication.
TEMPLATE_ECHO = [BROKER, [", ", "]]}        # note: the value you produce must adhere to the JSON schema: {"]]
TRUNCATED_ECHO = [[", ", ""]]
NUMERIC_LABEL = [["gateway", "System/Component"], ["429", "HTTP Status Code/Response"]]
COLON_NUMERIC_LABEL = [
    [":Redwood", "The system or service generating 429 responses"],
    [":429", "HTTP status code indicating too many requests (rate limiting)"],
]
LETTERLESS_LABELS = [",", ", ", "**", "429", ":429", "65%", "535.86", "2026-03-18", "$100"]

# Parser-accepted shapes observed in the same run; these must publish.
ACCEPTED_NER = [
    BROKER,
    [":Redwood", "The system or service generating 429 responses"],
    [":401s", "HTTP 401 responses returned by the gateway"],
    ["5xx errors", "Server-side HTTP failures"],
    ["ball", "A spherical toy."],
    ["ball", "A formal dance."],
]
ACCEPTED_TRIPLES = [
    {
        "sub": "Admission broker",
        "pred": ["rejects with", "owl:DatatypeProperty"],
        "obj": {"value": "429", "datatype": "xsd:integer", "lang": ""},
        "def": "The broker rejects excess requests with status 429.",
    },
    {
        "sub": ":Redwood",
        "pred": ["emits", "owl:ObjectProperty"],
        "obj": "5xx errors",
        "def": "Redwood emits server-side failures.",
    },
    {
        "sub": ["ball", "A spherical toy."],
        "pred": ["appears at", "owl:ObjectProperty"],
        "obj": ["ball", "A formal dance."],
        "def": "A toy appears at a dance.",
    },
]


def grammar_rules(grammar: str) -> dict[str, str]:
    return dict(line.split(" ::= ", 1) for line in grammar.splitlines())


def chat_completion(result: dict) -> bytes:
    return json.dumps(
        {
            "id": "boundary",
            "object": "chat.completion",
            "created": 0,
            "model": "boundary",
            "choices": [
                {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps(result)}}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        }
    ).encode()


def extract_over_http(named_entities: list, triples: list):
    """Run the real two-stage OpenIE program against a loopback OpenAI-compatible endpoint."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            prompt = str(request["messages"][-1]["content"])
            is_triples = "[[ ## named_entities ## ]]" in prompt
            raw = chat_completion({"triples": triples} if is_triples else {"named_entities": named_entities})
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
            lm = create_dspy_lm("boundary", base_url=f"http://127.0.0.1:{server.server_port}/v1", cache=False)
            program = YawProgram(orchestrator_lm=lm, rlm_sub_lm=lm, enable_gbnf=False)
            return program.extract_openie("chunk-boundary", "Passage text.")
        finally:
            server.shutdown()
            thread.join()


def publish_without_inference(record):
    """The deterministic planner path from one record to evidence rows (no repair stage needed)."""
    norm = normalize_openie_record(record)
    surfaces: dict[str, str] = {}
    chunk_surfaces: dict[tuple[str, str], str] = {}
    for key, value in norm.surfaces.items():
        record_surface_form(surfaces, chunk_surfaces, key, value, record.chunk_id)
    assert collect_missing_entity_norms(norm.triples, norm.entities, norm.definitions) == []
    triples, entities = apply_stage1_definitions(
        record.chunk_id, norm.triples, norm.entities, norm.definitions, (), surfaces, chunk_surfaces
    )
    materialized = materialize_entities_and_facts([(record.chunk_id, triples, entities)], surfaces, chunk_surfaces, {})
    pointer = ChunkPointer(record.chunk_id, "doc.md", 1, 2, "Doc")
    return build_evidence(pointer, "Passage text.", "program", record, materialized.bindings)


@pytest.mark.parametrize("label", LETTERLESS_LABELS)
def test_letterless_labels_are_not_resource_labels(label):
    assert not is_resource_label(label)


@pytest.mark.parametrize("label", ["Admission broker", ":Redwood", ":401s", "5xx errors", "409A", "Zürich", "東京"])
def test_lexical_labels_are_resource_labels(label):
    assert is_resource_label(label)


@pytest.mark.parametrize("grammar", [DSPY_NER_GBNF_GRAMMAR, build_dspy_ner_grammar(3)], ids=["open", "capped"])
def test_ner_grammar_labels_require_a_letter(grammar):
    rules = grammar_rules(grammar)
    assert rules["entity-pair"] == '"[" label ", " string "]"'
    assert rules["label"] == '"\\"" non-letter* letter char* "\\""'
    assert rules["letter"] == "[a-zA-Z\\x80-\\U0010FFFF]"
    # Printable ASCII except the quote, backslash and letters; escapes may follow the first letter.
    assert rules["non-letter"] == "[\\x20\\x21\\x23-\\x40\\x5B\\x5D-\\x60\\x7B-\\x7E]"


@pytest.mark.parametrize(
    "named_entities", [TEMPLATE_ECHO, TRUNCATED_ECHO, NUMERIC_LABEL, COLON_NUMERIC_LABEL], ids=lambda e: e[-1][0]
)
def test_ner_stage_rejects_labels_without_lexical_content(named_entities):
    with pytest.raises(ValueError, match="NER resource label"):
        extract_over_http(named_entities, [])


def test_parser_accepted_record_publishes_every_statement():
    record = extract_over_http(ACCEPTED_NER, ACCEPTED_TRIPLES)
    assert len(record.structured_triples) == len(ACCEPTED_TRIPLES)
    mentions, assertions = publish_without_inference(record)
    assert len(assertions) == len(ACCEPTED_TRIPLES)
    bound = {mention.mention_id for mention in mentions}
    assert all(a.subject_mention_id in bound for a in assertions)
    assert all(a.object_mention_id is None or a.object_mention_id in bound for a in assertions)
    assert len({m.sense_id for m in mentions if m.name == "ball"}) == 2


def test_shacl_rejects_letterless_resource_labels(tmp_path):
    with closing(open_yaw_db(tmp_path / "labels.db")) as conn:
        upsert_entities(conn, [("ent-comma", ",", "Echoed output template"), ("ent-broker", *BROKER)])
        upsert_entity_labels(
            conn,
            [
                (",", "ent-comma", "skos:prefLabel", ",", "c1"),
                ("admission broker", "ent-broker", "skos:prefLabel", BROKER[0], "c1"),
            ],
        )
        flagged = {v.focus_node for v in lint_graph_shacl(conn) if v.shape == "ConceptShape/resourceLabel"}
    assert flagged == {"ent-comma"}


def test_shacl_set_pref_label_heals_non_resource_and_survives_edc(tmp_path):
    doc_file = tmp_path / "doc.md"
    doc_file.write_text("Admission broker configuration.\n", encoding="utf-8")
    with closing(open_yaw_db(tmp_path / "labels_heal.db")) as conn:
        upsert_chunk_pointers(conn, [(ChunkPointer("c1", "doc.md", 1, 1, "Doc"), "Admission broker configuration.")])
        upsert_entities(conn, [("ent-comma", ",", "Punctuation artifact that needs a lexical concept label.")])
        upsert_entity_labels(conn, [(",", "ent-comma", "skos:prefLabel", ",", "c1")])

        with pytest.raises(ValueError):
            apply_shacl_repairs(conn, [{"focus_node": "ent-comma", "action": "set_pref_label", "value": "---"}])

        apply_shacl_repairs(
            conn,
            [{"focus_node": "ent-comma", "action": "set_pref_label", "value": "Comma delimiter"}],
        )
        canonicalize_entity_labels_edc(conn, corpus_root=tmp_path)
        row = conn.execute("SELECT name FROM entities WHERE entity_id = 'ent-comma'").fetchone()
        assert row is not None and row[0] == "Comma delimiter"
        assert [v for v in lint_graph_shacl(conn) if v.focus_node == "ent-comma"] == []


# --- V2: a statement never relates a resource sense to itself ------------------------------------

HUBSPOT = ["HubSpot", "CRM platform/software provided by HubSpot."]
# Verbatim reflexive statements from the same run's triple traces (509 of 1,094 calls had one).
REFLEXIVE_DEFINITION_AS_PREDICATE = {
    "sub": "HubSpot",
    "pred": ["CRM platform/software provided by HubSpot.", "owl:ObjectProperty"],
    "obj": "HubSpot",
    "def": "CRM platform/software provided by HubSpot.",
}
REFLEXIVE_TYPE_AS_PREDICATE = {
    "sub": "Omar El-Sayed",
    "pred": ["Developer", "owl:ObjectProperty"],
    "obj": "Omar El-Sayed",
    "def": "Omar El-Sayed",
}
SENSES: list[tuple[str, str]] = [
    (HUBSPOT[0], HUBSPOT[1]),
    ("Hubspot", HUBSPOT[1]),  # same sense: differs only in normalized form
    ("Omar El-Sayed", "Developer"),
    ("ball", "A spherical toy."),
    ("ball", "A formal dance."),
]


def endpoint_objects(grammar: str) -> dict[str, set[str]]:
    """Map each subject endpoint (JSON text) to the resource-object endpoints the grammar offers it."""
    rules = grammar_rules(grammar)
    literal = {name: json.loads(json.loads(body)) for name, body in rules.items() if name.startswith("endpoint-")}
    as_text = {name: json.dumps(value, ensure_ascii=False) for name, value in literal.items()}
    offered: dict[str, set[str]] = {}
    for name, body in rules.items():
        if name.startswith("subject-"):
            index = name.removeprefix("subject-")
            objects = rules.get(f"objects-{index}", "")
            offered[as_text[f"endpoint-{index}"]] = {as_text[ref] for ref in objects.split(" | ") if ref}
            assert body.startswith(f'endpoint-{index} ", \\"pred\\": "')
    return offered


def test_triples_grammar_offers_each_subject_only_other_senses():
    offered = endpoint_objects(build_dspy_triples_grammar(known_entities=SENSES))
    hubspot, hubspot_alias, omar = '"HubSpot"', '"Hubspot"', '"Omar El-Sayed"'
    toy, dance = json.dumps(SENSES[3]), json.dumps(SENSES[4])
    assert set(offered) == {hubspot, hubspot_alias, omar, toy, dance}
    assert offered[hubspot] == offered[hubspot_alias] == {omar, toy, dance}
    assert offered[omar] == {hubspot, hubspot_alias, toy, dance}
    assert toy in offered[dance] and dance in offered[toy]
    assert all(subject not in objects for subject, objects in offered.items())


def test_single_sense_catalog_allows_only_literal_statements():
    rules = grammar_rules(build_dspy_triples_grammar(known_entities=[HUBSPOT]))
    assert "objects-0" not in rules
    assert rules["subject-0"] == 'endpoint-0 ", \\"pred\\": " literal-statement'


@pytest.mark.parametrize(
    "triple",
    [
        REFLEXIVE_DEFINITION_AS_PREDICATE,
        REFLEXIVE_TYPE_AS_PREDICATE,
        {**REFLEXIVE_DEFINITION_AS_PREDICATE, "obj": "Hubspot", "def": "HubSpot relates to Hubspot."},
    ],
    ids=["definition-predicate", "type-predicate", "normalized-alias"],
)
def test_statement_parser_rejects_same_sense_statements(triple):
    with pytest.raises(ValueError, match="relates a resource sense to itself"):
        extract_structured_triples([triple], entity_definitions=SENSES, known_entities=[n for n, _ in SENSES])


def test_reflexive_statement_is_rejected_over_http():
    with pytest.raises(ValueError, match="relates a resource sense to itself"):
        extract_over_http([HUBSPOT, ["Zendesk", "Customer support platform."]], [REFLEXIVE_DEFINITION_AS_PREDICATE])


def test_normalization_never_drops_a_same_sense_statement():
    record = OpenIERecord(
        chunk_id="chunk-reflexive",
        entities=("HubSpot",),
        entity_definitions=(tuple(HUBSPOT),),
        triples=(),
        structured_triples=(
            StructuredTriple(
                subj="HubSpot",
                subj_def=HUBSPOT[1],
                pred="CRM platform/software provided by HubSpot.",
                obj="HubSpot",
                obj_def=HUBSPOT[1],
                spo_def=HUBSPOT[1],
                obj_kind="entity",
            ),
        ),
    )
    with pytest.raises(ValueError, match="relates a resource sense to itself"):
        normalize_openie_record(record)


def test_shacl_rejects_reflexive_facts(tmp_path):
    with closing(open_yaw_db(tmp_path / "facts.db")) as conn:
        upsert_chunk_pointers(conn, [(ChunkPointer("c1", "doc.md", 1, 2, "Doc"), "HubSpot sells CRM to Acme.")])
        upsert_entities(conn, [("ent-hubspot", *HUBSPOT), ("ent-acme", "Acme", "A customer company.")])
        upsert_facts(
            conn,
            [
                ("fact-self", "ent-hubspot", "provides", "ent-hubspot", "[]", "HubSpot provides itself.", "entity"),
                ("fact-sells", "ent-hubspot", "sells to", "ent-acme", "[]", "HubSpot sells CRM to Acme.", "entity"),
            ],
        )
        upsert_fact_sources(conn, [("fact-self", "c1"), ("fact-sells", "c1")])
        flagged = {v.focus_node for v in lint_graph_shacl(conn) if v.shape == "StatementShape/irreflexive"}
    assert flagged == {"fact-self"}


# --- V4: publication failure aborts optimization fast -------------------------------------------


def test_graph_publication_failure_aborts_optimization_fast(tmp_path, monkeypatch):
    """V4: A publication failure during candidate indexing aborts optimization with the original exception."""
    import dspy

    from tests.daemon.graph.test_engine import DeterministicMockEmbedder, DeterministicMockLM
    from yaw.core import FatalGraphPublicationError
    from yaw.daemon.graph.engine import YawGraphEngine

    lm = DeterministicMockLM()
    dspy.settings.configure(lm=lm)
    prog = YawProgram(enable_gbnf=False, orchestrator_lm=lm, rlm_sub_lm=lm)
    embedder = DeterministicMockEmbedder(dim=64)

    engine = YawGraphEngine(
        db_path=tmp_path / "v4_abort.db",
        workspace_root=tmp_path,
        program=prog,
        embedder=embedder,
    )
    try:
        engine._opt_chunks_pool = [(ChunkPointer("c1", "doc.md", 1, 2, "Doc"), "Some passage text.")]
        engine._in_optimization = True

        # Invariant 1: Inside optimization, index_chunks failure raises FatalGraphPublicationError
        def failing_index_chunks(*args, **kwargs):
            raise ValueError("Resource '429' does not have one resolved NER sense")

        monkeypatch.setattr(engine, "index_chunks", failing_index_chunks)

        with pytest.raises(FatalGraphPublicationError) as exc_info:
            engine._ensure_candidate_chunks_indexed(prog)
        assert isinstance(exc_info.value.original, ValueError)
        assert "Resource '429' does not have one resolved NER sense" in str(exc_info.value.original)

        # Invariant 2: In evolve_prompts / optimize, FatalGraphPublicationError unwraps to the original exception
        class MockGEPA:
            def __init__(self, *args, **kwargs):
                pass

            def compile(self, student, trainset, valset=None):
                # Candidate evaluation triggers _ensure_candidate_chunks_indexed
                engine._ensure_candidate_chunks_indexed(student)
                return student

        monkeypatch.setattr(dspy, "GEPA", MockGEPA, raising=False)
        devset = [dspy.Example(question="Q?", gold_titles=["Doc"], answer="A").with_inputs("question")]

        with pytest.raises(ValueError, match="Resource '429' does not have one resolved NER sense"):
            engine.evolve_prompts(devset, mode="gepa", reflection_token_counter=lambda text: len(str(text).split()))
    finally:
        engine.close()
