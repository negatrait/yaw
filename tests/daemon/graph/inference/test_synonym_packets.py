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
from yaw.daemon.graph.inference import HippoRAGProgram, ServingTokenCounter, create_dspy_lm
from yaw.daemon.graph.storage import (
    iter_synonym_evidence,
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
        def send(self, data):
            raw = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path.endswith("/health"):
                self.send(
                    {
                        "all_models_loaded": [
                            {
                                "id": "packet-model",
                                "backend_url": url,
                                "launch_command": ["-m", "packet.gguf"],
                            }
                        ]
                    }
                )
            else:
                self.send(
                    {
                        "model_path": "packet.gguf",
                        "chat_template": "loopback-byte-tokenizer",
                        "default_generation_settings": {"n_ctx": state["context"]},
                    }
                )

        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path.endswith("/apply-template"):
                rendered.append(data["messages"])
                self.send({"prompt": json.dumps(data["messages"], ensure_ascii=False)})
            elif self.path.endswith("/tokenize"):
                self.send({"tokens": list(data["content"].encode("utf-8"))})
            else:
                calls.append(data)
                text = next(m["content"] for m in data["messages"] if '{"format":"synonym-evidence-v1"' in m["content"])
                packet = json.JSONDecoder().raw_decode(text[text.index('{"format":') :])[0]
                decisions = [
                    {
                        "pair_id": p["pair_id"],
                        "relation": RELATIONS[i % 5],
                        "canonical": p["pair_id"].split("|")[1 if i % 5 == 3 else 0],
                    }
                    for i, p in enumerate(packet["pairs"])
                ]
                if state["mode"] == "missing":
                    decisions.pop()
                elif state["mode"] == "duplicate":
                    decisions.append(decisions[0])
                elif state["mode"] == "foreign":
                    decisions[0]["pair_id"] = "e999|e1000"
                elif state["mode"] == "canonical":
                    decisions[0]["canonical"] = "not-in-pair"
                elif state["mode"] == "relation":
                    decisions[0]["relation"] = "made-up"
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
                            "total_tokens": 100,
                        },
                    }
                )

        def log_message(self, format: str, *args: object) -> None:
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        url = f"http://127.0.0.1:{server.server_port}"
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            yield url, calls, rendered, state
        finally:
            server.shutdown()
            worker.join(timeout=5)


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
        conn.execute("INSERT INTO edges VALUES (?, ?, 'skos:broader', 1.0)", (entities[1][0], entities[0][0]))
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
        return next(iter_synonym_evidence(conn, candidates, tmp_path))


def make_program(url: str, *, context: int = 131072) -> HippoRAGProgram:
    lm = create_dspy_lm(
        "packet-model", base_url=url + "/v1", max_tokens=512, context_window_tokens=context, num_retries=0, cache=False
    )
    counter = ServingTokenCounter(url + "/v1", "packet-model")
    program = HippoRAGProgram(
        rlm_sub_lm=lm, synonym_token_counter=counter, context_window_tokens=context, max_output_tokens=512
    )
    program.max_invariant_retries = 0
    return program


def test_shared_physical_sources_keep_origin_ids_and_directed_profiles(tmp_path):
    batch = make_batch(tmp_path)
    packet = json.loads(batch.payload)
    assert len(packet["sources"]) == 1
    assert len({s["chunk"] for e in packet["entities"].values() for s in e["sources"]}) == 6
    assert "café" in next(iter(packet["sources"].values()))["text"]
    profiles = {e["name"]: e for e in packet["entities"].values()}
    assert "skos:narrower" in profiles["Concept 0"]["relations"]
    assert "skos:broader" in profiles["Concept 1"]["relations"]
    assert len([original for key, original in batch.identifiers.items() if key.startswith("c")]) == 6
    changed_ids = replace(batch, identifiers={**batch.identifiers, "c0": "another-chunk"})
    assert changed_ids.payload == batch.payload
    assert changed_ids.cache_identity != batch.cache_identity


def test_complete_native_request_and_original_id_decisions(tmp_path, packet_provider):
    url, calls, rendered, _ = packet_provider
    program = make_program(url)
    batch = make_batch(tmp_path)
    assert program.synonym_batch_fits(batch)
    decisions = program.judge_synonyms(batch)
    assert [x["relation"] for x in decisions] == list(RELATIONS)
    assert [x["pair_id"] for x in decisions] == [x["pair_id"] for x in batch.candidates]
    assert decisions[3]["canonical"] == batch.candidates[3]["entity_b"]
    assert len(calls) == 1
    assert rendered[-1] == calls[0]["messages"]
    assert program.synonym_audit_trail[-1]["input_tokens"] == len(json.dumps(rendered[-1], ensure_ascii=False).encode())
    assert "candidate_pairs" not in program.synonym_audit_trail[-1]


def test_packet_specific_grammar_reaches_native_provider(tmp_path, packet_provider):
    from yaw.daemon.graph.inference.grammars import build_dspy_synonym_judge_grammar

    url, calls, _, _ = packet_provider
    program = make_program(url)
    program.enable_gbnf = True
    program.synonym_judge.lm._yaw_supports_gbnf = True
    batch = make_batch(tmp_path)
    pairs = [p["pair_id"] for p in json.loads(batch.payload)["pairs"]]
    program.judge_synonyms(batch)
    grammar = calls[0]["grammar"]
    assert grammar == build_dspy_synonym_judge_grammar(pair_ids=pairs)
    root = grammar.splitlines()[0]
    assert 'decision-0 ", " decision-1 ", " decision-2 ", " decision-3 ", " decision-4' in root
    assert "?" not in grammar
    assert len([line for line in grammar.splitlines() if line.startswith("decision-")]) == len(pairs)
    with pytest.raises(ValueError, match="unique"):
        build_dspy_synonym_judge_grammar(pair_ids=[pairs[0], pairs[0]])


@pytest.mark.parametrize("mode", ["missing", "duplicate", "foreign", "canonical", "relation"])
def test_bad_decisions_fail_before_graph_application(tmp_path, packet_provider, mode):
    url, calls, _, state = packet_provider
    state["mode"] = mode
    program = make_program(url)
    with pytest.raises(ValueError):
        program.judge_synonyms(make_batch(tmp_path))
    assert len(calls) == 1
    assert not program.synonym_audit_trail


def test_complete_request_budget_counts_demos_and_never_generates_over_budget(tmp_path, packet_provider):
    url, calls, _, _ = packet_provider
    program = make_program(url, context=20000)
    batch = make_batch(tmp_path)
    program.synonym_judge.demos = [dspy.Example(candidate_pairs="unabridged demo " * 2000, decisions=[])]
    assert not program.synonym_batch_fits(batch)
    with pytest.raises(ValueError, match="exceeds context"):
        program.judge_synonyms(batch)
    assert not calls


def test_packet_splits_on_real_tokenizer_budget_without_losing_pairs(tmp_path, packet_provider):
    url, _, _, _ = packet_provider
    batch = make_batch(tmp_path)
    program = make_program(url)
    with closing(open_yaw_db(tmp_path / "evidence.db")) as conn:
        singles = list(iter_synonym_evidence(conn, batch.candidates, tmp_path, batch_size=1))
        guard = program.synonym_request_budget()
        counts = [
            guard.count(
                program.synonym_judge.lm,
                {},
                program.synonym_judge.signature,
                program.synonym_judge.demos,
                {"candidate_pairs": b.payload},
            )
            for b in singles
        ]
        guard.context_tokens = max(counts) + 512 + 512
        # Use this fixed verified budget directly; don't re-resolve deployment capacity.
        packets = list(
            iter_synonym_evidence(
                conn,
                batch.candidates,
                tmp_path,
                fits=lambda b: guard.fits(b, program.synonym_judge, program.synonym_judge.lm),
            )
        )
        assert len(packets) > 1
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
 HippoRAGProgram,ServingTokenCounter,create_dspy_lm,configure_yaw_dspy_cache,synonym_fingerprint,
)
from yaw.daemon.graph.storage import CandidateCacheStore,CandidateStageCache,open_managed_yaw_db
root=Path(sys.argv[1]);url=sys.argv[2];crash=sys.argv[3]=='crash'
data=json.loads((root/'batch.json').read_text(encoding='utf-8'))
batch=SynonymEvidenceBatch(tuple(data['candidates']),data['payload'],data['identifiers'])
configure_yaw_dspy_cache(root/'responses',enable_memory_cache=False)
lm=create_dspy_lm('packet-model',base_url=url+'/v1',max_tokens=512,context_window_tokens=131072,num_retries=0)
program=HippoRAGProgram(rlm_sub_lm=lm,synonym_token_counter=ServingTokenCounter(url+'/v1','packet-model'),max_output_tokens=512)
program.max_invariant_retries=0
managed=open_managed_yaw_db(root/'recovery.db')
with managed.transaction() as conn:
 cache=CandidateStageCache(CandidateCacheStore(conn),managed,'synonym',lambda:'packet-test')
key=(synonym_fingerprint(program),batch.cache_identity)
try:
 with managed.transaction():
  decisions=cache.get(key)
  if decisions is None:
   decisions=program.judge_synonyms(batch);cache[key]=decisions
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
