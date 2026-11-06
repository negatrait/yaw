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
