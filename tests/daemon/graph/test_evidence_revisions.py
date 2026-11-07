"""Source evidence revisions and canonical bindings through the native CocoIndex app."""

from contextlib import contextmanager
from dataclasses import replace
from itertools import permutations

import cocoindex as coco
import pytest

from daemon.graph.test_batch_equivalence import assert_same_graph, canonical_graph
from daemon.graph.test_indexing_overlap import adapter
from daemon.graph.test_indexing_overlap import provider as provider
from yaw.core import ChunkPointer
from yaw.daemon.graph.engine import YawGraphEngine
from yaw.daemon.graph.inference import YawProgram, create_dspy_lm, openie_fingerprint
from yaw.daemon.graph.inference.components import EMBEDDING_ENDPOINT, OPENIE_ENDPOINT, EmbeddingEndpoint, OpenIEEndpoint
from yaw.daemon.graph.planner import GraphPlanSettings
from yaw.daemon.graph.storage import open_managed_yaw_db
from yaw.daemon.graph.storage.db import GRAPH_DB
from yaw.daemon.graph.storage.evidence import sense_id
from yaw.daemon.graph.storage.managed_sqlite import VALIDATE_SQLITE
from yaw.daemon.graph.storage.rows import validate_graph_state
from yaw.daemon.indexing import (
    FrozenChunkStream,
    InferenceIdentity,
    bind_native_indexing,
    frozen_indexing_app_main,
    indexing_app_main,
)
from yaw.daemon.ingestion.source_targets import SOURCE_DB, SOURCE_ROOT, SourceSettings


@contextmanager
def native_environment(root, state_dir, url):
    db = open_managed_yaw_db(state_dir / "yaw.db")
    embedder = adapter(url)
    lm = create_dspy_lm("overlap", base_url=url + "/v1", cache=False, temperature=1.0, max_tokens=512)
    program = YawProgram(orchestrator_lm=lm, rlm_sub_lm=lm, enable_gbnf=False, max_output_tokens=512)
    identity = openie_fingerprint(program)
    context = coco.ContextProvider()
    context.provide(GRAPH_DB, db)
    context.provide(SOURCE_DB, db)
    context.provide(SOURCE_ROOT, root)
    context.provide(VALIDATE_SQLITE, validate_graph_state)
    context.provide(EMBEDDING_ENDPOINT, EmbeddingEndpoint(embedder))
    context.provide(OPENIE_ENDPOINT, OpenIEEndpoint(program, lm, identity))
    env = coco.Environment(settings=coco.Settings(db_path=state_dir / "native-state"), context_provider=context)
    settings = GraphPlanSettings(
        corpus_root=root,
        program_identity=identity,
        embedding_identity=embedder.profile.index_identity,
        synonymy_top_k=8,
    )
    try:
        yield db, env, settings
    finally:
        env.close()
        db.close()


@contextmanager
def native_corpus(root, state_dir, url):
    with native_environment(root, state_dir, url) as (db, env, settings):
        stream = FrozenChunkStream()
        app = coco.App(
            coco.AppConfig(name="evidence_revisions", environment=env), frozen_indexing_app_main, stream, settings
        )
        yield db, stream, app


def filesystem_app(env, settings):
    return coco.App(
        coco.AppConfig(name="filesystem_evidence", environment=env),
        indexing_app_main,
        SourceSettings(("**/*.md",), (), min_chunk_size=1),
        InferenceIdentity(settings.program_identity, settings.embedding_identity),
        settings,
    )


def extraction(label, definition, value="007"):
    return {
        "named_entities": [[label, definition], ["Anchor", "A telemetry anchor service."]],
        "triples": [
            {
                "sub": label,
                "pred": ["routes Via", "owl:ObjectProperty"],
                "obj": "Anchor",
                "def": "The service routes via Anchor.",
            },
            *(
                {
                    "sub": label,
                    "pred": ["has Value", "owl:DatatypeProperty"],
                    "obj": {"value": value, "datatype": datatype, "lang": language},
                    "def": "The service reports a value.",
                }
                for datatype, language in (
                    ("xsd:integer", ""),
                    ("xsd:string", ""),
                    ("rdf:langString", "en"),
                    ("rdf:langString", "fi"),
                )
            ),
        ],
    }


def fixed_vector(text):
    if "orion" in text.casefold():
        return [1.0, 0.0, 0.0, 0.0]
    return [0.0, 1.0, 0.0, 0.0]


def publish(stream, app, chunks):
    stream.chunks = tuple(chunks)
    app.update_blocking()


def native_engine(root, state_dir, url, **settings):
    lm = create_dspy_lm("overlap", base_url=url + "/v1", cache=False, temperature=1.0, max_tokens=512)
    program = YawProgram(orchestrator_lm=lm, rlm_sub_lm=lm, enable_gbnf=False, max_output_tokens=512)
    return bind_native_indexing(
        YawGraphEngine(state_dir / "yaw.db", root, program, adapter(url), synonymy_top_k=8, **settings), batch_size=2
    )


@pytest.mark.parametrize("restart", [False, True])
@pytest.mark.parametrize("raw_documents", [False, True])
def test_engine_snapshots_withdraw_shared_support_and_reintroduce(tmp_path, provider, restart, raw_documents):
    url, state = provider
    state["delay"] = 0
    state["embedding_vector"] = fixed_vector
    texts = ("Orion first evidence.", "Orion second evidence.")
    state["extractions"] = {text: extraction("Orion", "A telemetry service.") for text in texts}
    chunks = [(ChunkPointer(str(index), f"{index}.md", 1, 1), text) for index, text in enumerate(texts)]
    for ptr, text in chunks:
        (tmp_path / ptr.rel_posix_path).write_text(text, encoding="utf-8")
    engine = native_engine(tmp_path, tmp_path / "incremental", url)
    try:
        for step, desired in enumerate((chunks, chunks[1:], [], chunks[:1])):
            inputs = [text for _, text in desired] if raw_documents else desired
            publish_snapshot = engine.index if raw_documents else engine.index_chunks
            assert publish_snapshot(inputs) == len(desired)
            with engine.managed_conn.readonly() as conn:
                actual = canonical_graph(conn)
                assert conn.execute("SELECT COUNT(*) FROM fact_assertions").fetchone()[0] == 5 * len(desired)
                assert conn.execute("SELECT COUNT(*) FROM fact_sources").fetchone()[0] == 5 * len(desired)
                assert conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == (5 if desired else 0)
                assert conn.execute("SELECT COUNT(*) FROM source_chunks").fetchone()[0] == len(desired)
                assert conn.execute("SELECT COUNT(*) FROM chunk_inference").fetchone()[0] == len(desired)
            reference = native_engine(tmp_path, tmp_path / f"reference-{step}", url)
            try:
                (reference.index if raw_documents else reference.index_chunks)(inputs)
                with reference.managed_conn.readonly() as conn:
                    assert_same_graph(actual, canonical_graph(conn))
            finally:
                reference.close()
            if restart:
                engine.close()
                engine = native_engine(tmp_path, tmp_path / "incremental", url)
            calls = len(state["intervals"])
            changes = engine.conn.total_changes
            (engine.index if raw_documents else engine.index_chunks)(inputs)
            assert len(state["intervals"]) == calls
            assert engine.conn.total_changes == changes
    finally:
        engine.close()


def test_retrieval_seeds_selected_fact_endpoints_despite_homograph_vector_conflict(tmp_path, provider):
    url, state = provider
    state["delay"] = 0
    planet_text = "Mercury is a planet that orbits Sun."
    metal_text = "Mercury is a metal that has Conductivity."
    state["extractions"] = {
        planet_text: {
            "named_entities": [["Mercury", "A planet."], ["Sun", "A star."]],
            "triples": [{"sub": "Mercury", "pred": "orbits", "obj": "Sun", "def": "Mercury orbits Sun."}],
        },
        metal_text: {
            "named_entities": [["Mercury", "A metal."], ["Conductivity", "A property."]],
            "triples": [{"sub": "Mercury", "pred": "has", "obj": "Conductivity", "def": "Mercury conducts."}],
        },
    }
    state["embedding_vector"] = lambda text: (
        [1.0, 0.0, 0.0, 0.0]
        if "planet" in text.casefold()
        else [0.0, 0.0, 1.0, 0.0]
        if "star" in text.casefold()
        else [0.0, 0.0, 0.0, 1.0]
        if "property" in text.casefold()
        else [0.0, 1.0, 0.0, 0.0]
    )
    state["selected_facts"] = [["Mercury", "orbits", "Sun"]]
    engine = native_engine(
        tmp_path, tmp_path / "index", url, linking_top_k=1, fact_fts_weight=10.0, passage_node_weight=0.0
    )
    try:
        engine.index([planet_text, metal_text])
        assert engine.conn.execute("SELECT COUNT(*) FROM entities WHERE name='Mercury'").fetchone()[0] == 2
        queries = []
        engine.conn.set_trace_callback(queries.append)
        hits, facts = engine.retrieve_with_seeds("Which body orbits Sun?", top_k=1, allow_intent_path=False)
        engine.conn.set_trace_callback(None)
        assert facts == [("Mercury", "orbits", "Sun")]
        assert len(hits) == 1 and "planet" in hits[0][2]
        assert not any("entity_labels" in sql for sql in queries)
    finally:
        engine.close()


@pytest.mark.parametrize("restart", [False, True])
def test_native_evidence_replacement_withdrawal_and_late_representative(tmp_path, provider, restart):
    url, state = provider
    state["delay"] = 0
    state["embedding_vector"] = fixed_vector
    definitions = sorted(
        ("A telemetry service.", "A service collecting telemetry.", "An event telemetry service."),
        key=lambda definition: sense_id("Orion", definition),
        reverse=True,
    )
    texts = [f"Orion routes via Anchor. Evidence {index}: {definition}" for index, definition in enumerate(definitions)]
    revised_text = "Orion routes via Anchor. Its revised value is 008."
    state["extractions"] = {
        text: extraction("Orion", definition) for text, definition in zip(texts, definitions, strict=True)
    }
    state["extractions"][revised_text] = extraction("Orion", definitions[1], "008")
    chunks = [(ChunkPointer(f"c{index}", f"source-{index}.md", 1, 1), text) for index, text in enumerate(texts)]
    duplicate = (ChunkPointer("duplicate", "duplicate.md", 1, 1), texts[0])
    revised = (chunks[1][0], revised_text)
    renamed = (replace(duplicate[0], rel_posix_path="renamed.md"), duplicate[1])
    snapshots = (
        [chunks[0], duplicate],
        [*chunks, duplicate],
        [chunks[0], revised, chunks[2], duplicate],
        [chunks[0], revised, duplicate],
        [revised, duplicate],
        [revised, renamed],
        [revised],
        [],
        [chunks[0]],
    )
    holder = None
    previous_raw = {}
    try:
        for step, desired in enumerate(snapshots):
            for pointer, text in desired:
                (tmp_path / pointer.rel_posix_path).write_text(text + "\n", encoding="utf-8")
            if holder is None:
                holder = native_corpus(tmp_path, tmp_path / "incremental", url)
                db, stream, app = holder.__enter__()
            publish(stream, app, desired)
            with db.readonly() as conn:
                actual = canonical_graph(conn)
                assert conn.execute("SELECT COUNT(*) FROM fact_assertions").fetchone()[0] == 5 * len(desired)
                assert conn.execute("SELECT COUNT(*) FROM fact_sources").fetchone()[0] == 5 * len(desired)
                assert {
                    (row[0], row[1], row[2])
                    for row in conn.execute("SELECT chunk_id,name,definition FROM entity_mentions")
                } == {
                    (pointer.chunk_id, name, definition)
                    for pointer, text in desired
                    for name, definition in state["extractions"][text]["named_entities"]
                }
                assert set(
                    map(
                        tuple,
                        conn.execute(
                            "SELECT obj_literal_lex,obj_datatype,obj_lang FROM facts WHERE obj_kind='literal'"
                        ),
                    )
                ) == {
                    (triple["obj"]["value"], triple["obj"]["datatype"], triple["obj"]["lang"])
                    for _, text in desired
                    for triple in state["extractions"][text]["triples"]
                    if isinstance(triple["obj"], dict)
                }
                assert {row[0] for row in conn.execute("SELECT pred FROM facts")} == (
                    {"routes Via", "has Value"} if desired else set()
                )
                raw = dict(conn.execute("SELECT assertion_id,raw_json FROM fact_assertions"))
                assert all(previous_raw[key] == value for key, value in raw.items() if key in previous_raw)
                previous_raw = raw
                assert {row[0] for row in conn.execute("SELECT chunk_id FROM entity_mentions")} == {
                    ptr.chunk_id for ptr, _ in desired
                }
                if desired:
                    representative = min(
                        sense_id("Orion", state["extractions"][text]["named_entities"][0][1]) for _, text in desired
                    )
                    assert {
                        row[0] for row in conn.execute("SELECT entity_id FROM entity_mentions WHERE name='Orion'")
                    } == {representative}
                    assert {row[0] for row in conn.execute("SELECT obj_lang FROM facts WHERE obj_kind='literal'")} == {
                        "",
                        "en",
                        "fi",
                    }
                if step == 0:
                    revisions = conn.execute("SELECT DISTINCT chunk_id,source_revision FROM entity_mentions").fetchall()
                    assert len({row[1] for row in revisions}) == 2
            with native_corpus(tmp_path, tmp_path / f"reference-{step}", url) as (
                reference_db,
                reference_stream,
                reference_app,
            ):
                publish(reference_stream, reference_app, desired)
                with reference_db.readonly() as conn:
                    assert_same_graph(actual, canonical_graph(conn))
            calls = len(state["intervals"])
            with db.readonly() as conn:
                changes = conn.total_changes
            publish(stream, app, desired)
            assert len(state["intervals"]) == calls
            with db.readonly() as conn:
                assert conn.total_changes == changes
            if restart:
                holder.__exit__(None, None, None)
                holder = native_corpus(tmp_path, tmp_path / "incremental", url)
                db, stream, app = holder.__enter__()
                calls = len(state["intervals"])
                with db.readonly() as conn:
                    changes = conn.total_changes
                publish(stream, app, desired)
                assert len(state["intervals"]) == calls
                with db.readonly() as conn:
                    assert conn.total_changes == changes
                    assert_same_graph(actual, canonical_graph(conn))
    finally:
        if holder is not None:
            holder.__exit__(None, None, None)


def test_filesystem_evidence_revisions_match_surviving_sources(tmp_path, provider):
    url, state = provider
    state["delay"] = 0
    state["embedding_vector"] = fixed_vector
    root = tmp_path / "corpus"
    root.mkdir()
    texts = ("Orion original passage.", "Orion independent passage.", "Orion replacement passage.")
    definitions = ("A telemetry service.", "An event telemetry service.", "A service collecting telemetry.")
    state["extractions"] = {
        text: extraction("Orion", definition, value)
        for text, definition, value in zip(texts, definitions, ("007", "007", "008"), strict=True)
    }
    snapshots = (
        {"first.md": texts[0]},
        {"first.md": texts[0], "second.md": texts[1]},
        {"first.md": texts[2], "second.md": texts[1]},
        {"second.md": texts[1]},
        {"renamed.md": texts[1]},
        {},
    )
    with native_environment(root, tmp_path / "incremental", url) as (db, env, settings):
        app = filesystem_app(env, settings)
        previous_raw = {}
        for step, desired in enumerate(snapshots):
            for source in root.glob("*.md"):
                if source.name not in desired:
                    source.unlink()
            for name, text in desired.items():
                (root / name).write_text(text, encoding="utf-8")
            app.update_blocking()
            with db.readonly() as conn:
                actual = canonical_graph(conn)
                assert conn.execute("SELECT COUNT(*) FROM fact_assertions").fetchone()[0] == 5 * len(desired)
                raw = dict(conn.execute("SELECT assertion_id,raw_json FROM fact_assertions"))
                assert all(previous_raw[key] == value for key, value in raw.items() if key in previous_raw)
                previous_raw = raw
            with native_environment(root, tmp_path / f"reference-{step}", url) as (reference_db, ref_env, ref_settings):
                filesystem_app(ref_env, ref_settings).update_blocking()
                with reference_db.readonly() as conn:
                    assert_same_graph(actual, canonical_graph(conn))


def test_native_same_sense_metadata_is_independent_of_arrival(tmp_path, provider):
    url, state = provider
    state["delay"] = 0
    state["embedding_vector"] = fixed_vector
    state["extractions"] = {
        "Uppercase Orion source.": extraction("ORION", "A TELEMETRY SERVICE."),
        "Titlecase Orion source.": extraction("Orion", "A telemetry service."),
    }
    chunks = [
        (ChunkPointer("z", "upper.md", 1, 1), "Uppercase Orion source."),
        (ChunkPointer("a", "title.md", 1, 1), "Titlecase Orion source."),
    ]
    for pointer, text in chunks:
        (tmp_path / pointer.rel_posix_path).write_text(text + "\n", encoding="utf-8")
    with native_corpus(tmp_path, tmp_path / "incremental", url) as (db, stream, app):
        publish(stream, app, chunks[:1])
        publish(stream, app, chunks)
        with db.readonly() as conn:
            actual = canonical_graph(conn)
    with native_corpus(tmp_path, tmp_path / "reference", url) as (db, stream, app):
        publish(stream, app, chunks)
        with db.readonly() as conn:
            assert_same_graph(actual, canonical_graph(conn))


@pytest.mark.parametrize("labels", [("Orion", "ORION", "orion"), ("Orion Service", "Orion Services", "OrionSvc")])
def test_native_alias_classes_converge_for_every_arrival_order(tmp_path, provider, labels):
    url, state = provider
    state["delay"] = 0
    state["embedding_vector"] = fixed_vector
    state["relation"] = "skos:altLabel"
    rows = zip(
        labels,
        ("A telemetry service.", "A service collecting telemetry events.", "An event telemetry service."),
        strict=True,
    )
    chunks = []
    state["extractions"] = {}
    for index, (name, definition) in enumerate(rows):
        text = f"Alias evidence {index}: {name} routes through Anchor."
        pointer = ChunkPointer(f"alias-{index}", f"alias-{index}.md", 1, 1)
        (tmp_path / pointer.rel_posix_path).write_text(text + "\n", encoding="utf-8")
        chunks.append((pointer, text))
        state["extractions"][text] = extraction(name, definition)
    with native_corpus(tmp_path, tmp_path / "reference", url) as (db, stream, app):
        publish(stream, app, chunks)
        with db.readonly() as conn:
            expected = canonical_graph(conn)
    for order in permutations(range(3)):
        with native_corpus(tmp_path, tmp_path / ("order-" + "".join(map(str, order))), url) as (db, stream, app):
            for count in range(1, 4):
                publish(stream, app, [chunks[index] for index in order[:count]])
            with db.readonly() as conn:
                assert_same_graph(canonical_graph(conn), expected)


@pytest.mark.parametrize(
    "mutation",
    [
        "UPDATE fact_assertions SET chunk_id='other' WHERE chunk_id='owner'",
        "UPDATE fact_assertions SET source_revision='wrong-revision' WHERE chunk_id='owner'",
        "UPDATE entity_mentions SET sense_id='entity-zzzz' WHERE name='Orion'",
        "UPDATE entity_mentions SET entity_id=(SELECT entity_id FROM entities WHERE name='Anchor') "
        "WHERE name='Orion' AND chunk_id='owner'",
    ],
)
def test_source_evidence_bindings_are_validated_before_commit(tmp_path, provider, mutation):
    url, state = provider
    state["delay"] = 0
    state["embedding_vector"] = fixed_vector
    texts = ("Orion source one.", "Orion source two.")
    state["extractions"] = {text: extraction("Orion", "A telemetry service.") for text in texts}
    chunks = [(ChunkPointer(cid, f"{cid}.md", 1, 1), text) for cid, text in zip(("owner", "other"), texts, strict=True)]
    for pointer, text in chunks:
        (tmp_path / pointer.rel_posix_path).write_text(text + "\n", encoding="utf-8")
    with native_corpus(tmp_path, tmp_path / "index", url) as (db, stream, app):
        publish(stream, app, chunks)
        with db.readonly() as conn:
            before = canonical_graph(conn)
        # Deliberately corrupt a proposed row in a real transaction, not the committed graph.
        with pytest.raises(ValueError, match="source (mention|assertion) binding"), db.transaction() as conn:
            conn.execute(mutation)
            validate_graph_state(conn)
        with db.readonly() as conn:
            assert_same_graph(canonical_graph(conn), before)


def test_cross_batch_fact_definition_repair_persists_and_invalidates_on_endpoint_collision(tmp_path, provider):
    import asyncio
    import sqlite3
    from pathlib import Path

    from yaw.daemon.graph.planner import clone_graph_to_memory
    from yaw.daemon.graph.storage import lint_graph_shacl
    from yaw.daemon.graph.storage.evidence import canonicalize_fact_definitions, evidence_schemas

    url, state = provider
    state["delay"] = 0
    state["embedding_vector"] = lambda text: (
        [1.0, 0.0, 0.0, 0.0] if text.casefold().startswith("text: # orion") else [0.0, 1.0, 0.0, 0.0]
    )
    healed_v1 = "Orion routes telemetry traffic through the Anchor service."
    state["healed_definition"] = healed_v1

    orion_def_hi = "A telemetry service for cluster routing."
    orion_def_lo = next(
        d
        for d in ("A telemetry service.", "Telemetry node.", "Core router.", "Edge service.")
        if sense_id("orion", d) < sense_id("orion", orion_def_hi)
    )
    assert len(orion_def_hi) > len(orion_def_lo)

    c1_text = "Orion routes via Anchor in batch one."
    c2_text = "Orion routes via Anchor again and Beacon monitors Pulse in batch two."
    anchor_def = "A telemetry anchor service."
    state["extractions"] = {
        c1_text: {
            "named_entities": [
                ["Orion", orion_def_hi],
                ["Anchor", anchor_def],
            ],
            "triples": [
                {
                    "sub": "Orion",
                    "pred": ["routes Via", "owl:ObjectProperty"],
                    "obj": "Anchor",
                    "def": orion_def_hi,
                }
            ],
        },
        c2_text: {
            "named_entities": [
                ["Orion", orion_def_lo],
                ["Anchor", anchor_def],
                ["Beacon", "A monitoring agent."],
                ["Pulse", "A health signal."],
            ],
            "triples": [
                {
                    "sub": "Orion",
                    "pred": ["routes Via", "owl:ObjectProperty"],
                    "obj": "Anchor",
                    "def": orion_def_hi,
                },
                {
                    "sub": "Beacon",
                    "pred": ["monitors", "owl:ObjectProperty"],
                    "obj": "Pulse",
                    "def": "Beacon monitors the Pulse health signal.",
                },
            ],
        },
    }
    c1 = (ChunkPointer("c1", "c1.md", 1, 1), c1_text)
    c2 = (ChunkPointer("c2", "c2.md", 1, 1), c2_text)
    for ptr, text in (c1, c2):
        (tmp_path / ptr.rel_posix_path).write_text(text + "\n", encoding="utf-8")

    with native_corpus(tmp_path, tmp_path / "inc", url) as (db, stream, app):
        publish(stream, app, [c1])
        healer_calls_batch1 = len(state.get("healer_requests", []))
        assert healer_calls_batch1 == 3
        with db.readonly() as conn:
            assert lint_graph_shacl(conn) == []
            fid_b1 = conn.execute("SELECT fact_id FROM facts WHERE pred = 'routes Via'").fetchone()[0]
            defs_b1 = dict(conn.execute("SELECT pred, definition FROM facts").fetchall())
            assert defs_b1 == {"routes Via": healed_v1}

        publish(stream, app, [c1, c2])
        assert len(state.get("healer_requests", [])) == healer_calls_batch1
        with db.readonly() as conn:
            assert lint_graph_shacl(conn) == []
            fid_b2 = conn.execute("SELECT fact_id FROM facts WHERE pred = 'routes Via'").fetchone()[0]
            assert fid_b2 != fid_b1
            defs_b2 = dict(conn.execute("SELECT pred, definition FROM facts").fetchall())
            assert defs_b2["routes Via"] == healed_v1
            assert defs_b2["monitors"] == "Beacon monitors the Pulse health signal."

        healed_v2 = "Orion forwards telemetry packets to the Anchor endpoint."
        state["healed_definition"] = healed_v2
        c3_text = "Anchor endpoint description update."
        state["extractions"][c3_text] = {
            "named_entities": [
                ["Anchor", healed_v1],
                ["Pulse", "A health signal."],
            ],
            "triples": [
                {
                    "sub": "Anchor",
                    "pred": ["emits", "owl:ObjectProperty"],
                    "obj": "Pulse",
                    "def": "Anchor emits the Pulse health signal.",
                }
            ],
        }
        c3 = (ChunkPointer("c3", "c3.md", 1, 1), c3_text)
        (tmp_path / c3[0].rel_posix_path).write_text(c3_text + "\n", encoding="utf-8")
        publish(stream, app, [c1, c2, c3])
        assert len(state.get("healer_requests", [])) > healer_calls_batch1
        with db.readonly() as conn:
            assert lint_graph_shacl(conn) == []
            defs_b3 = dict(conn.execute("SELECT pred, definition FROM facts").fetchall())
            assert defs_b3["routes Via"] == healed_v2

    opt_db = Path(".yaw/yaw_optimize_20261106_221811.db")
    if opt_db.exists():
        schemas = asyncio.run(evidence_schemas())
        ro_conn = sqlite3.connect(f"file:{opt_db.resolve().as_posix()}?mode=ro", uri=True)
        try:
            mem_conn = clone_graph_to_memory(ro_conn, schemas)
        finally:
            ro_conn.close()
        try:
            prior_defs = {str(r[0]): str(r[1]) for r in mem_conn.execute("SELECT fact_id, definition FROM facts")}
            healed_before = mem_conn.execute(
                """
                SELECT COUNT(*) FROM facts f
                WHERE NOT EXISTS (
                    SELECT 1 FROM fact_assertions a
                    WHERE a.fact_id = f.fact_id
                      AND COALESCE(json_extract(a.raw_json, '$.spo_def'), '') = f.definition
                )
                """
            ).fetchone()[0]
            assert healed_before == 42
            canonicalize_fact_definitions(mem_conn, prior_definitions=prior_defs)
            healed_after = mem_conn.execute(
                """
                SELECT COUNT(*) FROM facts f
                WHERE NOT EXISTS (
                    SELECT 1 FROM fact_assertions a
                    WHERE a.fact_id = f.fact_id
                      AND COALESCE(json_extract(a.raw_json, '$.spo_def'), '') = f.definition
                )
                """
            ).fetchone()[0]
            assert healed_after == 42
            closed_world_violations = [v for v in lint_graph_shacl(mem_conn) if v.shape == "StatementShape/closedWorld"]
            assert closed_world_violations == []
        finally:
            mem_conn.close()


def test_focus_isolated_shacl_memo_reuses_healed_focus_when_earlier_violation_added(tmp_path, provider):
    from yaw.daemon.graph.storage import lint_graph_shacl
    from yaw.daemon.graph.storage.evidence import fact_id as canonical_fact_id

    url, state = provider
    state["delay"] = 0
    state["embedding_vector"] = fixed_vector
    state["healed_definition"] = "A distinct non-colliding statement definition."

    specs = [
        ("Alpha", "Alpha node.", "Beta", "Beta node."),
        ("Gamma", "Gamma node.", "Delta", "Delta node."),
        ("Epsilon", "Epsilon node.", "Zeta", "Zeta node."),
    ]
    ranked = sorted(
        specs,
        key=lambda item: canonical_fact_id(
            sense_id(item[0].lower(), item[1]),
            "links to",
            "entity",
            sense_id(item[2].lower(), item[3]),
        ),
    )
    chunks = []
    state["extractions"] = {}
    for idx, (s_lbl, s_def, o_lbl, o_def) in enumerate(ranked):
        text = f"Passage {idx}: {s_lbl} links to {o_lbl}."
        ptr = ChunkPointer(f"m{idx}", f"m{idx}.md", 1, 1)
        (tmp_path / ptr.rel_posix_path).write_text(text + "\n", encoding="utf-8")
        chunks.append((ptr, text))
        state["extractions"][text] = {
            "named_entities": [[s_lbl, s_def], [o_lbl, o_def]],
            "triples": [
                {
                    "sub": s_lbl,
                    "pred": ["links To", "owl:ObjectProperty"],
                    "obj": o_lbl,
                    "def": s_def,
                }
            ],
        }

    ephemeral_text = "Passage ephemeral: Beacon monitors Pulse."
    ephemeral_ptr = ChunkPointer("m_ephemeral", "m_ephemeral.md", 1, 1)
    (tmp_path / ephemeral_ptr.rel_posix_path).write_text(ephemeral_text + "\n", encoding="utf-8")
    state["extractions"][ephemeral_text] = {
        "named_entities": [["Beacon", "A monitoring agent."], ["Pulse", "A health signal."]],
        "triples": [
            {
                "sub": "Beacon",
                "pred": ["monitors", "owl:ObjectProperty"],
                "obj": "Pulse",
                "def": "Beacon monitors the Pulse health signal.",
            }
        ],
    }
    chunk_ephemeral = (ephemeral_ptr, ephemeral_text)
    chunk_lo, chunk_hi_1, chunk_hi_2 = chunks
    with native_corpus(tmp_path, tmp_path / "memo", url) as (db, stream, app):
        publish(stream, app, [chunk_ephemeral, chunk_hi_1, chunk_hi_2])
        healer_calls_after_first = len(state.get("healer_requests", []))
        direct_calls_after_first = state.get("healer_requests", []).count("direct")
        assert healer_calls_after_first == 6
        assert direct_calls_after_first == 2

        publish(stream, app, [chunk_lo, chunk_hi_1, chunk_hi_2])
        assert len(state.get("healer_requests", [])) - healer_calls_after_first == 3
        assert state.get("healer_requests", []).count("direct") - direct_calls_after_first == 1
        with db.readonly() as conn:
            assert lint_graph_shacl(conn) == []
            assert conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 3


def test_single_pass_fact_embedding_and_delta_scoped_batch_publication(tmp_path, provider):
    url, state = provider
    state["delay"] = 0
    state["embedding_vector"] = fixed_vector

    c1_text = "Orion routes via Anchor."
    c2_text = "Vega links to Sirius."
    state["extractions"] = {
        c1_text: {
            "named_entities": [
                ["Orion", "A telemetry service."],
                ["Anchor", "A telemetry anchor service."],
            ],
            "triples": [
                {
                    "sub": "Orion",
                    "pred": ["routes Via", "owl:ObjectProperty"],
                    "obj": "Anchor",
                    "def": "The service routes via Anchor.",
                }
            ],
        },
        c2_text: {
            "named_entities": [
                ["Vega", "A relay node."],
                ["Sirius", "A core router."],
            ],
            "triples": [
                {
                    "sub": "Vega",
                    "pred": ["links To", "owl:ObjectProperty"],
                    "obj": "Sirius",
                    "def": "Vega links to the Sirius router.",
                }
            ],
        },
    }
    c1 = (ChunkPointer("c1", "c1.md", 1, 1), c1_text)
    c2 = (ChunkPointer("c2", "c2.md", 1, 1), c2_text)
    for ptr, text in (c1, c2):
        (tmp_path / ptr.rel_posix_path).write_text(text + "\n", encoding="utf-8")

    with native_corpus(tmp_path, tmp_path / "inc_embed", url) as (db, stream, app):
        publish(stream, app, [c1])
        batch1_embeds = list(state.get("embedding_inputs", []))
        orion_fact_embeds_b1 = [
            s
            for s in batch1_embeds
            if "routes via" in s.casefold() and "the service routes via anchor." in s.casefold()
        ]
        assert len(orion_fact_embeds_b1) == 1
        assert "Orion routes Via Anchor." in orion_fact_embeds_b1[0]

        publish(stream, app, [c1, c2])
        batch2_new_embeds = state.get("embedding_inputs", [])[len(batch1_embeds) :]
        orion_fact_embeds_b2 = [s for s in batch2_new_embeds if "routes via" in s.casefold()]
        vega_fact_embeds_b2 = [
            s
            for s in batch2_new_embeds
            if "links to" in s.casefold() and "vega links to the sirius router." in s.casefold()
        ]
        assert orion_fact_embeds_b2 == []
        assert len(vega_fact_embeds_b2) == 1
        assert "Vega links To Sirius." in vega_fact_embeds_b2[0]
        with db.readonly() as conn:
            actual = canonical_graph(conn)

    with native_corpus(tmp_path, tmp_path / "ref_embed", url) as (ref_db, ref_stream, ref_app):
        publish(ref_stream, ref_app, [c1, c2])
        with ref_db.readonly() as conn:
            assert_same_graph(actual, canonical_graph(conn))
