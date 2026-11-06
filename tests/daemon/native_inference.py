"""Publish extraction-only fixtures through the real native artifact target."""

import hashlib
import json
from collections.abc import Sequence
from contextlib import closing
from dataclasses import asdict
from pathlib import Path

import cocoindex as coco
from cocoindex.connectors import sqlite

from yaw.core import OpenIERecord
from yaw.daemon.graph.storage import open_managed_yaw_db
from yaw.daemon.indexing import ChunkInference
from yaw.daemon.ingestion.source_targets import SOURCE_DB


@coco.fn
async def declare_inference(records: tuple[OpenIERecord, ...]) -> None:
    target = await sqlite.mount_table_target(
        SOURCE_DB,
        table_name="chunk_inference",
        table_schema=await sqlite.TableSchema[ChunkInference].from_class(ChunkInference, primary_key=["chunk_id"]),
    )
    for record in records:
        raw = json.dumps(asdict(record))
        target.declare_row(
            row=ChunkInference(
                record.chunk_id, hashlib.sha256(raw.encode()).hexdigest(), "fixture", "fixture", raw, "[]"
            )
        )


def publish_inference(path: Path, records: Sequence[OpenIERecord]) -> None:
    with closing(open_managed_yaw_db(path)) as db:
        context = coco.ContextProvider()
        context.provide(SOURCE_DB, db)
        env = coco.Environment(
            settings=coco.Settings(db_path=path.with_suffix(".native-state")), context_provider=context
        )
        try:
            coco.App(
                coco.AppConfig(name="artifact_fixture", environment=env), declare_inference, tuple(records)
            ).update_blocking()
        finally:
            env.close()
