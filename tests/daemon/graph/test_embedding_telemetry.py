"""Native CocoIndex embedding call telemetry and persistent memo behavior."""

import asyncio

import cocoindex as coco

from daemon.graph.test_indexing_overlap import adapter, embedding_fixture
from daemon.graph.test_indexing_overlap import provider as provider
from yaw.daemon.graph.inference.components import EMBEDDING_ENDPOINT, EmbeddingEndpoint


def test_batch_execution_event_and_no_event_on_persistent_memo_reuse(tmp_path, provider):
    url, state = provider
    state["delay"] = 0
    embedder = adapter(url)
    requests = tuple(
        embedder.prepare_request([f"Distinct evidence {i}."], batch_size=64, stage="chunk_embed") for i in range(64)
    )
    direct_endpoint = EmbeddingEndpoint(embedder)
    direct = asyncio.run(direct_endpoint.execute_batch(list(requests)))
    direct_events = direct_endpoint.drain_embedding_executions()
    assert len(direct) == 64
    assert len(state["intervals"]) == len(direct_events) == 1
    assert direct_events[0]["batch_sizes"] == [64]
    assert direct_events[0]["num_inputs"] == 64
    assert direct_events[0]["tokens_used"] is None
    assert direct_events[0]["memo_replayed"] is False
    assert {item.request_id for item in direct} == {direct_events[0]["request_id"]}
    state["intervals"].clear()

    split_embedder = adapter(url)
    split_embedder.max_batch_chars = 100
    split_requests = tuple(
        split_embedder.prepare_request([f"Distinct evidence {i}."], batch_size=64, stage="chunk_embed")
        for i in range(64)
    )
    split_endpoint = EmbeddingEndpoint(split_embedder)
    split = asyncio.run(split_endpoint.execute_batch(list(split_requests)))
    split_events = split_endpoint.drain_embedding_executions()
    assert len(split) == 64 and len(split_events) == 1
    assert split_events[0]["num_batches"] > 1
    assert split_events[0]["num_batches"] == len(state["intervals"])
    assert sum(split_events[0]["batch_sizes"]) == 64
    assert {item.request_id for item in split} == {split_events[0]["request_id"]}
    state["intervals"].clear()

    first_call_count = 0
    for attempt in range(2):
        endpoint = EmbeddingEndpoint(embedder)
        context = coco.ContextProvider()
        context.provide(EMBEDDING_ENDPOINT, endpoint)
        env = coco.Environment(settings=coco.Settings.from_env(tmp_path / "memo"), context_provider=context)
        app = coco.App(
            coco.AppConfig(name="telemetry_batch", environment=env, max_inflight_components=64),
            embedding_fixture,
            requests,
        )
        try:
            result = app.update_blocking()
            events = endpoint.drain_embedding_executions()
            assert len(result) == 64
            if attempt == 0:
                first_call_count = len(state["intervals"])
                assert len(events) == first_call_count
                assert len({event["request_id"] for event in events}) == first_call_count
                assert sum(event["num_inputs"] for event in events) == 64
                assert {record.request_id for record in result} == {event["request_id"] for event in events}
                assert all(event["tokens_used"] is None and event["memo_replayed"] is False for event in events)
            else:
                assert len(state["intervals"]) == first_call_count
                assert events == ()
        finally:
            env.close()
