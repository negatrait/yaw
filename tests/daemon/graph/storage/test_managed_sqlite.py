"""Public typed targets: one producing component, one shared backend transaction."""

from dataclasses import dataclass

import cocoindex as coco
import pytest

from yaw.daemon.graph.storage import open_managed_yaw_db
from yaw.daemon.graph.storage.db import GRAPH_DB
from yaw.daemon.graph.storage.managed_sqlite import VALIDATE_SQLITE, ManagedTableTarget, mount_table


@dataclass(frozen=True)
class RevisionRow:
    key: str
    revision: int


class Source:
    revision = 1


@coco.fn
async def publish(source):
    nodes = await mount_table("probe_nodes", RevisionRow, ["key"])
    checkpoint = await mount_table("probe_checkpoint", RevisionRow, ["key"])
    nodes.declare_row(row=RevisionRow("node", source.revision))
    checkpoint.declare_row(row=RevisionRow("checkpoint", source.revision))


def test_shared_sink_rolls_back_and_recovers_without_partial_publication(tmp_path):
    db = open_managed_yaw_db(tmp_path / "yaw.db")
    source = Source()
    reject = True
    trace = []

    def validate(conn):
        node = conn.execute("SELECT revision FROM probe_nodes").fetchone()[0]
        checkpoint = conn.execute("SELECT revision FROM probe_checkpoint").fetchone()[0]
        assert node == checkpoint
        if reject and node == 2:
            raise ValueError("Rejected proposed revision")

    context = coco.ContextProvider()
    context.provide(GRAPH_DB, db)
    context.provide(VALIDATE_SQLITE, validate)
    env = coco.Environment(settings=coco.Settings(db_path=tmp_path / "native-state"), context_provider=context)
    app = coco.App(coco.AppConfig(name="typed_atomic_probe", environment=env), publish, source)
    try:
        app.update_blocking()
        with db.transaction() as conn:
            conn.set_trace_callback(trace.append)
        source.revision = 2
        with pytest.raises(ValueError, match="Rejected proposed revision"):
            app.update_blocking()
        with db.readonly() as conn:
            assert conn.execute("SELECT revision FROM probe_nodes").fetchone()[0] == 1
            assert conn.execute("SELECT revision FROM probe_checkpoint").fetchone()[0] == 1
        assert sum(statement == "ROLLBACK" for statement in trace) >= 1
        reject = False
        app.update_blocking()
        with db.readonly() as conn:
            assert conn.execute("SELECT revision FROM probe_nodes").fetchone()[0] == 2
            assert conn.execute("SELECT revision FROM probe_checkpoint").fetchone()[0] == 2
        with db.transaction() as conn:
            changes = conn.total_changes
        app.update_blocking()
        with db.readonly() as conn:
            assert conn.total_changes == changes
    finally:
        env.close()
        db.close()


def test_managed_table_target_delete_row(tmp_path):
    db = open_managed_yaw_db(tmp_path / "yaw.db")

    @dataclass
    class SourceState:
        deleted: bool = False

    state = SourceState(deleted=False)

    @coco.fn
    async def publish_deletable(source: SourceState) -> None:
        nodes = await mount_table("deletable_nodes", RevisionRow, ["key"])
        assert isinstance(nodes, ManagedTableTarget)
        if not source.deleted:
            nodes.declare_row(row=RevisionRow("node_1", 42))
        else:
            nodes.delete_row(("node_1",))

    context = coco.ContextProvider()
    context.provide(GRAPH_DB, db)
    context.provide(VALIDATE_SQLITE, lambda conn: None)
    env = coco.Environment(settings=coco.Settings(db_path=tmp_path / "native-state"), context_provider=context)
    app = coco.App(coco.AppConfig(name="delete_probe", environment=env), publish_deletable, state)
    try:
        app.update_blocking()
        with db.readonly() as conn:
            assert conn.execute("SELECT revision FROM deletable_nodes WHERE key='node_1'").fetchone()[0] == 42
        state.deleted = True
        app.update_blocking()
        with db.readonly() as conn:
            assert conn.execute("SELECT COUNT(*) FROM deletable_nodes WHERE key='node_1'").fetchone()[0] == 0
    finally:
        env.close()
        db.close()
