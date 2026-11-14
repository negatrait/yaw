"""Compact evidence over real SQLite, files, native DSPy and loopback HTTP."""

import json
import subprocess
import sys
import threading
from contextlib import closing
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import dspy
import pytest

from yaw.core import ChunkPointer, SynonymEvidenceBatch
from yaw.daemon.graph.inference import YawProgram, create_dspy_lm
from yaw.daemon.graph.storage import (
    iter_semantic_evidence,
    open_yaw_db,
    upsert_chunk_pointers,
    upsert_edges_with_sources,
    upsert_entities,
)

RELATIONS = ("skos:altLabel", "skos:hiddenLabel", "skos:related", "skos:broader", "owl:differentFrom")


@pytest.fixture
def packet_provider():
    calls, rendered = [], []
    state = {"mode": "valid", "context": 131072}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def send(self, data):
            raw = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(raw)
            self.close_connection = True

        def do_GET(self):
            self.send_error(404)

        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            completion = self.path.endswith("/v1/completions")
            assert completion or self.path.endswith("/chat/completions")
            if state["mode"] == "overflow":
                calls.append(data)
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error":{"message":"context_length_exceeded", "code":"context_length_exceeded"}}')
                return
            calls.append(data)
            text = (
                data["prompt"]
                if completion
                else next(m["content"] for m in data["messages"] if '{"format":"synonym-evidence-v3"' in m["content"])
            )
            packet = json.JSONDecoder().raw_decode(text[text.index('{"format":') :])[0]
            decisions = [
                (
                    {
                        "pair_id": p["pair_id"],
                        "reasoning": f"Reasoning {i}",
                        "equivalent": True,
                        "definition": f"Unified definition {i}",
                        "hidden_label": False,
                    }
                    if i % 2 == 0
                    else {
                        "pair_id": p["pair_id"],
                        "reasoning": f"Reasoning {i}",
                        "equivalent": True,
                        "definition": f"Unified definition {i}",
                        "hidden_label": True,
                    }
                )
                for i, p in enumerate(packet["pairs"])
            ]
            if state["mode"] == "two-stage":
                is_relation = "RelationDecision" in text or "JudgeRelations" in text
                if is_relation:
                    choices = ("skos:broader", "skos:narrower", "skos:related", "owl:differentFrom")
                    decisions = [
                        {"pair_id": p["pair_id"], "reasoning": "Grounded relation", "relation": choices[i % 4]}
                        for i, p in enumerate(packet["pairs"])
                    ]
                else:
                    decisions = [
                        {"pair_id": p["pair_id"], "reasoning": "Distinct senses", "equivalent": False}
                        for p in packet["pairs"]
                    ]
            if state["mode"] == "missing":
                decisions.pop()
            elif state["mode"] == "duplicate":
                decisions.append(decisions[0])
            elif state["mode"] == "foreign":
                decisions[0]["pair_id"] = "e999|e1000"
            elif state["mode"] == "canonical":
                decisions[0]["reasoning"] = ""
            elif state["mode"] == "relation":
                decisions[0]["hidden_label"] = "not-a-bool"
            if completion:
                self.send(
                    {
                        "id": "packet",
                        "model": "packet-model",
                        "choices": [{"text": json.dumps({"decisions": decisions}), "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": len(text), "completion_tokens": 50, "total_tokens": len(text) + 50},
                    }
                )
                return
            self.send(
                {
                    "id": "packet",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "packet-model",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": json.dumps({"decisions": decisions})},
                        }
                    ],
                    "usage": {
                        "prompt_tokens": len(json.dumps(data["messages"])),
                        "completion_tokens": 50,
                        "total_tokens": len(json.dumps(data["messages"])) + 50,
                    },
                }
            )

        def log_message(self, format: str, *args: object) -> None:
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        server.daemon_threads = True
        server.block_on_close = False
        url = f"http://127.0.0.1:{server.server_port}"
        worker = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
        worker.start()
        try:
            yield url, calls, rendered, state
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=0.5)


def make_batch(tmp_path: Path, count: int = 5) -> SynonymEvidenceBatch:
    with closing(open_yaw_db(tmp_path / "evidence.db")) as conn:
        source = tmp_path / "shared.json"
        source.write_text(
            json.dumps({"title": "Shared", "text": "Evidence with Unicode: café.\nSame source."}), encoding="utf-8"
        )
        entities = [(f"entity-original-{i}", f"Concept {i}", f"Distinct meaning {i}") for i in range(count + 1)]
        upsert_entities(conn, entities)
        for i, (eid, *_rest) in enumerate(entities):
            pointer = ChunkPointer(f"chunk-{i}", source.name, 1, 1, "Shared", 1, 1)
            upsert_chunk_pointers(conn, [(pointer, "Evidence")])
            upsert_edges_with_sources(conn, [(pointer.chunk_id, eid, "passage", 1.0, pointer.chunk_id)])
        # Seed an already validated directed relation; sibling demotion belongs to other tests.
        conn.execute(
            "INSERT INTO edges (src_id, dst_id, kind, weight) VALUES (?, ?, 'skos:broader', 1.0)",
            (entities[1][0], entities[0][0]),
        )
        candidates = [
            {
                "pair_id": f"{entities[0][0]}|{e[0]}",
                "entity_a": entities[0][0],
                "entity_b": e[0],
                "dense_sim": 0.8,
                "lexical_sim": 0.9,
            }
            for e in entities[1:]
        ]
        conn.commit()
        return next(iter_semantic_evidence(conn, candidates, tmp_path))


def make_program(url: str, *, context: int = 131072) -> YawProgram:
    # Tests assert exact provider call counts; DSPy retries would add calls, so transport errors must fail directly.
    lm = create_dspy_lm(
        "packet-model", base_url=url + "/v1", max_tokens=512, context_window_tokens=context, num_retries=0, cache=False
    )
    program = YawProgram(rlm_sub_lm=lm, context_window_tokens=context, max_output_tokens=512)
    return program


def test_shared_physical_sources_keep_origin_ids_and_directed_profiles(tmp_path):
    batch = make_batch(tmp_path)
    packet = json.loads(batch.payload)
    assert packet["format"] == "synonym-evidence-v3"
    assert len(packet["pairs"]) == 5
    assert "café" in packet["sources"][packet["entities"][packet["pairs"][0]["entity_a"]]["sources"][0]]["text"]
    assert len([original for key, original in batch.identifiers.items() if key.startswith("e")]) == 6
    changed_ids = replace(batch, identifiers={**batch.identifiers, "e0": "another-entity"})
    assert changed_ids.payload == batch.payload
    assert changed_ids.cache_identity != batch.cache_identity


def test_complete_native_request_and_original_id_decisions(tmp_path, packet_provider):
    url, calls, rendered, _ = packet_provider
    program = make_program(url)
    batch = make_batch(tmp_path)
    decisions = program.judge_semantics(batch)
    assert [x["relation"] for x in decisions] == [
        "skos:altLabel",
        "skos:hiddenLabel",
        "skos:altLabel",
        "skos:hiddenLabel",
        "skos:altLabel",
    ]
    assert [x["pair_id"] for x in decisions] == [x["pair_id"] for x in batch.candidates]
    assert len(calls) == 1
    assert rendered == []
    usage = program.label_judge_audit_trail[-1]["usage"]
    assert {
        k: v
        for k, v in next(iter(usage.values())).items()
        if k in ("prompt_tokens", "completion_tokens", "total_tokens")
    } == {
        "prompt_tokens": len(json.dumps(calls[0]["messages"])),
        "completion_tokens": 50,
        "total_tokens": len(json.dumps(calls[0]["messages"])) + 50,
    }
    assert "candidate_pairs" not in program.label_judge_audit_trail[-1]


def test_packet_specific_grammar_reaches_native_provider(tmp_path, packet_provider):
    from yaw.daemon.graph.inference.grammars import build_dspy_label_judge_grammar

    url, calls, _, _ = packet_provider
    program = make_program(url)
    from yaw.daemon.graph.inference.chat import YawDSPyLM
    from yaw.daemon.graph.inference.completions import GemmaCompletionsEngine

    lm = YawDSPyLM(
        "openai/packet-model", engine=GemmaCompletionsEngine(url + "/v1", "key"), max_tokens=512, cache=False
    )
    lm._yaw_supports_gbnf = True
    program = YawProgram(rlm_sub_lm=lm, enable_gbnf=True, max_output_tokens=512)
    batch = make_batch(tmp_path)
    pairs = [p["pair_id"] for p in json.loads(batch.payload)["pairs"]]
    program.judge_semantics(batch)
    grammar = calls[0]["grammar"]
    assert grammar == build_dspy_label_judge_grammar(pair_ids=pairs)
    root = grammar.splitlines()[0]
    assert 'decision-0 ", " decision-1 ", " decision-2 ", " decision-3 ", " decision-4' in root
    assert "?" not in grammar
    assert len([line for line in grammar.splitlines() if line.startswith("decision-")]) == len(pairs)
    with pytest.raises(ValueError, match="unique"):
        build_dspy_label_judge_grammar(pair_ids=[pairs[0], pairs[0]])


@pytest.mark.parametrize("mode", ["missing", "duplicate", "foreign", "canonical", "relation"])
def test_bad_decisions_fail_before_graph_application(tmp_path, packet_provider, mode):
    url, calls, _, state = packet_provider
    state["mode"] = mode
    program = make_program(url)
    with pytest.raises((ValueError, dspy.utils.exceptions.AdapterParseError)):
        program.judge_semantics(make_batch(tmp_path))
    assert len(calls) == 1
    assert not program.label_judge_audit_trail


def test_provider_context_overflow_propagates_without_reduced_retry(tmp_path, packet_provider):
    from dspy._vendor.lm15.errors import ContextLengthError
    from dspy.utils.exceptions import ContextWindowExceededError

    url, calls, _, state = packet_provider
    program = make_program(url, context=20000)
    state["mode"] = "overflow"
    batch = make_batch(tmp_path)
    program.label_judge.demos = [dspy.Example(candidate_pairs="unabridged demo " * 2000, decisions=[])]
    with pytest.raises(ContextWindowExceededError) as failure:
        program.judge_semantics(batch)
    assert isinstance(failure.value.__cause__, ContextLengthError)
    assert len(calls) == 1
    assert not program.label_judge_audit_trail


def test_packet_splits_on_configured_pair_limit_without_losing_pairs(tmp_path, packet_provider):
    batch = make_batch(tmp_path)
    with closing(open_yaw_db(tmp_path / "evidence.db")) as conn:
        packets = list(iter_semantic_evidence(conn, batch.candidates, tmp_path, batch_size=2))
    assert [len(b.candidates) for b in packets] == [2, 2, 1]
    assert [c for b in packets for c in b.candidates] == list(batch.candidates)


def test_fresh_process_cache_survives_rollback_without_repeated_generation(tmp_path, packet_provider):
    url, calls, _, _ = packet_provider
    batch = make_batch(tmp_path)
    request = tmp_path / "batch.json"
    request.write_text(
        json.dumps({"candidates": batch.candidates, "payload": batch.payload, "identifiers": batch.identifiers}),
        encoding="utf-8",
    )
    script = """
import json,sys
from pathlib import Path
from yaw.core import SynonymEvidenceBatch
from yaw.daemon.graph.inference import (
 YawProgram,create_dspy_lm,configure_yaw_dspy_cache,label_judge_fingerprint,
)
from yaw.daemon.graph.storage import open_managed_yaw_db
root=Path(sys.argv[1]);url=sys.argv[2];crash=sys.argv[3]=='crash'
data=json.loads((root/'batch.json').read_text(encoding='utf-8'))
batch=SynonymEvidenceBatch(tuple(data['candidates']),data['payload'],data['identifiers'])
configure_yaw_dspy_cache(root/'responses',enable_memory_cache=False)
# len(calls) == 1 must count cache misses only; DSPy retries would add calls.
lm=create_dspy_lm('packet-model',base_url=url+'/v1',max_tokens=512,context_window_tokens=131072,num_retries=0)
program=YawProgram(rlm_sub_lm=lm,max_output_tokens=512)
managed=open_managed_yaw_db(root/'recovery.db')
key=f"label_judge:{label_judge_fingerprint(program)}:{batch.cache_identity}"
try:
 with managed.transaction() as conn:
  row=conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
  decisions=json.loads(row[0]) if row is not None else None
  if decisions is None:
   decisions=program.judge_semantics(batch)
   conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, json.dumps(decisions)))
  if crash:raise RuntimeError('interrupted before graph commit')
 print(json.dumps(decisions))
except RuntimeError:
 if not crash:raise
finally:managed.close()
"""
    for mode in ("crash", "resume", "resume"):
        result = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path), url, mode],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        if mode != "crash":
            assert len(json.loads(result.stdout)) == 5
        assert len(calls) == 1


def test_lossless_shared_sources_and_transitive_subset(tmp_path):
    with closing(open_yaw_db(tmp_path / "long.db")) as conn:
        lines = ["café " + "x" * 1000] + [f"line {i}" for i in range(1, 20)] + ["tail Ω"]
        source = tmp_path / "long.txt"
        source.write_text("\n".join(lines), encoding="utf-8")
        upsert_entities(conn, [(f"e{i}", f"Entity {i}", f"Definition {i}") for i in range(3)])
        pointer = ChunkPointer("shared", source.name, 1, len(lines), "Title", 1, len(lines))
        upsert_chunk_pointers(conn, [(pointer, "Evidence")])
        for i in range(3):
            upsert_edges_with_sources(conn, [("shared", f"e{i}", "passage", 1.0, "shared")])
        candidates = [
            {"pair_id": "e0|e1", "entity_a": "e0", "entity_b": "e1"},
            {"pair_id": "e0|e2", "entity_a": "e0", "entity_b": "e2"},
        ]
        batch = next(iter_semantic_evidence(conn, candidates, tmp_path))
        packet = json.loads(batch.payload)
        assert len(packet["entities"]) == 3
        assert len(packet["sources"]) == 1
        assert batch.payload.count("tail Ω") == 1
        assert next(iter(packet["sources"].values()))["text"] == "\n".join(lines)
        assert all("type" not in entity for entity in packet["entities"].values())
        subset = json.loads(batch.subset(["e0|e1"]).payload)
        assert set(subset["entities"]) == {"e0", "e1"}
        assert set(subset["sources"]) == {ref for entity in subset["entities"].values() for ref in entity["sources"]}
        source.unlink()
        with pytest.raises(ValueError, match="Stale source"):
            next(iter_semantic_evidence(conn, candidates, tmp_path))


def test_all_source_chunks_and_oversize_failure(tmp_path):
    with closing(open_yaw_db(tmp_path / "many.db")) as conn:
        upsert_entities(conn, [("a", "Alpha", "Alpha definition"), ("b", "Beta", "Beta definition")])
        for i in range(3):
            source = tmp_path / f"source{i}.txt"
            source.write_text(f"source {i} Ω", encoding="utf-8")
            pointer = ChunkPointer(f"chunk{i}", source.name, 1, 1, f"Title {i}")
            upsert_chunk_pointers(conn, [(pointer, "Evidence")])
            upsert_edges_with_sources(conn, [(pointer.chunk_id, "a", "passage", 1.0, pointer.chunk_id)])
        candidates = [{"pair_id": "a|b", "entity_a": "a", "entity_b": "b"}]
        packet = json.loads(next(iter_semantic_evidence(conn, candidates, tmp_path)).payload)
        assert len(packet["sources"]) == 3
        assert all(f"source {i} Ω" in json.dumps(packet, ensure_ascii=False) for i in range(3))
        with pytest.raises(ValueError, match="exceeds"):
            next(iter_semantic_evidence(conn, candidates, tmp_path, max_chars=100))


def test_native_two_stage_relations_and_orientations(tmp_path, packet_provider):
    url, calls, _, state = packet_provider
    state["mode"] = "two-stage"
    batch = make_batch(tmp_path, count=4)
    program = make_program(url)
    decisions = program.judge_semantics(batch)
    assert len(calls) == 2
    assert [d["relation"] for d in decisions] == ["skos:broader", "skos:broader", "skos:related", "owl:differentFrom"]
    a, b = decisions[0]["pair_id"].split("|")
    assert (decisions[0]["source_id"], decisions[0]["target_id"]) == (a, b)
    a, b = decisions[1]["pair_id"].split("|")
    assert (decisions[1]["source_id"], decisions[1]["target_id"]) == (b, a)
    assert len(program.label_judge_audit_trail) == len(program.relation_judge_audit_trail) == 1
    recorded = program.relation_judge_audit_trail[0]["decisions"]
    assert [d["relation"] for d in recorded] == ["skos:broader", "skos:narrower", "skos:related", "owl:differentFrom"]
    assert "source_id" not in recorded[1] and "target_id" not in recorded[1]


def test_relation_grammar_and_invariants_constrain_fact_pairs(tmp_path):
    from yaw.daemon.graph.inference.grammars import build_dspy_relation_judge_grammar
    from yaw.daemon.graph.inference.invariants import validate_stage_invariants
    from yaw.daemon.graph.inference.parsers import parse_relation_decisions

    batch = make_batch(tmp_path, count=2)
    packet = json.loads(batch.payload)
    packet["pairs"][0]["direct_facts"] = [{"predicate": "uses"}]
    pair_ids = [p["pair_id"] for p in packet["pairs"]]
    grammar = build_dspy_relation_judge_grammar(pair_ids=pair_ids, associated_pair_ids=pair_ids[:1])
    rules = [line for line in grammar.splitlines() if line.startswith("decision-")]
    assert "owl:differentFrom" not in rules[0]
    assert "owl:differentFrom" in rules[1]
    for relation in ("skos:broader", "skos:narrower", "skos:related", "owl:differentFrom"):
        raw = [{"pair_id": pair_ids[1], "reasoning": "Evidence", "relation": relation}]
        assert parse_relation_decisions(raw)[0]["relation"] == relation
    decisions = [{"pair_id": pid, "reasoning": "Evidence", "relation": "owl:differentFrom"} for pid in pair_ids]
    score, _ = validate_stage_invariants(
        "relation_judge", {"candidate_pairs": json.dumps(packet)}, dspy.Prediction(decisions=decisions)
    )
    assert score < 1


def test_missing_indexed_source_pointer_aborts_evidence(tmp_path):
    with closing(open_yaw_db(tmp_path / "missing.db")) as conn:
        upsert_entities(conn, [("a", "Alpha", "Alpha definition"), ("b", "Beta", "Beta definition")])
        conn.execute(
            "INSERT INTO entity_labels (label_norm, entity_id, label_kind, raw_label, chunk_id) VALUES (?, ?, ?, ?, ?)",
            ("alpha", "a", "skos:prefLabel", "Alpha", "missing-chunk"),
        )
        candidates = [{"pair_id": "a|b", "entity_a": "a", "entity_b": "b"}]
        with pytest.raises(ValueError, match="Missing source pointers.*missing-chunk"):
            next(iter_semantic_evidence(conn, candidates, tmp_path))
        conn.execute("UPDATE entity_labels SET chunk_id = '__healed__' WHERE entity_id = 'a'")
        packet = json.loads(next(iter_semantic_evidence(conn, candidates, tmp_path)).payload)
        assert packet["sources"] == {}


def test_native_direct_fact_skips_identity_but_not_relation(tmp_path, packet_provider):
    url, calls, _, state = packet_provider
    state["mode"] = "two-stage"
    batch = make_batch(tmp_path, count=1)
    packet = json.loads(batch.payload)
    packet["pairs"][0]["direct_facts"] = [{"predicate": "is part of"}]
    batch = replace(batch, payload=json.dumps(packet, ensure_ascii=False, separators=(",", ":")))
    program = make_program(url)
    decisions = program.judge_semantics(batch)
    assert len(calls) == 1
    assert not program.label_judge_audit_trail
    assert len(program.relation_judge_audit_trail) == 1
    assert decisions[0]["relation"] == "skos:broader"


def test_hidden_label_and_full_fact_definition_are_preserved(tmp_path):
    with closing(open_yaw_db(tmp_path / "facts.db")) as conn:
        upsert_entities(conn, [("a", "Alpha", "Alpha definition"), ("b", "Beta", "Beta definition")])
        conn.execute(
            "INSERT INTO entity_labels VALUES (?, ?, ?, ?, ?)",
            ("alhpa", "a", "skos:hiddenLabel", "Alhpa", "__healed__"),
        )
        triple = ["Alpha", "uses", "Beta"]
        definition = "Fact evidence Ω " + "details " * 300 + "definition tail"
        conn.execute(
            "INSERT INTO facts (fact_id, subj_id, pred, obj_id, raw_triple_json, definition) VALUES (?, ?, ?, ?, ?, ?)",
            ("f1", "a", "uses", "b", json.dumps(triple), definition),
        )
        candidates = [{"pair_id": "a|b", "entity_a": "a", "entity_b": "b"}]
        packet = json.loads(next(iter_semantic_evidence(conn, candidates, tmp_path)).payload)
        entity = packet["entities"][packet["pairs"][0]["entity_a"]]
        assert entity["labels"]["skos:hiddenLabel"] == ["Alhpa"]
        assert entity["facts"] == [{"triple": triple, "definition": definition}]
