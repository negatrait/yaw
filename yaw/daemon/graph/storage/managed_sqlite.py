"""Typed SQLite targets sharing one transactional sink per publication component.

Uses CocoIndex's public container/child-handler pattern. Processing declares rows;
only this connector applies reconciled row actions. No application-side writes.
"""

import re
import sqlite3
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, NamedTuple, cast

import cocoindex as coco
from cocoindex.connectorkits.fingerprint import fingerprint_object
from cocoindex.connectors import sqlite

from .db import GRAPH_DB, ManagedConnection, ensure_vec0_table, resolve_db_identity

VALIDATE_SQLITE = coco.ContextKey[Callable[[sqlite3.Connection], None]]("yaw/managed_sqlite/validate")


def identifier(value: str) -> str:
    if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", value) is None:
        raise ValueError(f"Invalid SQLite identifier: {value!r}")
    return f'"{value}"'


class TableKey(NamedTuple):
    db_identity: str
    name: str


@dataclass(frozen=True)
class VecSpec:
    column: str = "embedding"
    dim: int | None = None
    distance_metric: str = "cosine"


@dataclass(frozen=True)
class TableSpec:
    name: str
    schema: sqlite.TableSchema[Any]
    fts_unindexed: tuple[str, ...] | None = None
    indexes: tuple[tuple[str, ...], ...] = ()
    db_identity: str = ""
    vec: VecSpec | None = None
    is_delta: bool = False


@dataclass(frozen=True)
class TableAction:
    spec: TableSpec
    delete: bool
    create: bool


DELETE_ROW: dict[str, Any] = {"__yaw_delete_row__": True}
_DELETE_RECORD: bytes = fingerprint_object(DELETE_ROW)


def _schema_base(spec: TableSpec, *, ignore_vec_dim: bool = False) -> TableSpec:
    return replace(
        spec,
        vec=replace(spec.vec, dim=None) if (ignore_vec_dim and spec.vec is not None) else spec.vec,
        is_delta=False,
    )


@dataclass(frozen=True)
class RowAction:
    spec: TableSpec
    key: tuple[Any, ...]
    values: Mapping[str, Any] | None


def create_table(conn: sqlite3.Connection, spec: TableSpec) -> None:
    if spec.vec is not None:
        ensure_vec0_table(
            conn,
            spec.name,
            spec.schema.primary_key[0],
            spec.vec.dim,
            column=spec.vec.column,
            metric=spec.vec.distance_metric,
        )
        return
    table = identifier(spec.name)
    if spec.fts_unindexed is not None:
        columns = ",".join(
            identifier(name) + (" UNINDEXED" if name in spec.fts_unindexed else "") for name in spec.schema.columns
        )
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING fts5("
            f"{columns},content='',contentless_delete=1,contentless_unindexed=1)"
        )
    else:
        columns = ",".join(
            f"{identifier(name)} {col.type}" + ("" if col.nullable else " NOT NULL")
            for name, col in spec.schema.columns.items()
        )
        primary_key = ",".join(identifier(name) for name in spec.schema.primary_key)
        conn.execute(f"CREATE TABLE IF NOT EXISTS {table} ({columns},PRIMARY KEY ({primary_key}))")
    for fields in spec.indexes:
        name = identifier(spec.name + "_" + "_".join(fields))
        columns = ",".join(identifier(field) for field in fields)
        conn.execute(f"CREATE INDEX IF NOT EXISTS {name} ON {table} ({columns})")


@dataclass(frozen=True)
class DatabaseSink:
    def __call__(self, context: coco.ContextProvider, actions: Sequence[TableAction | RowAction]) -> None:
        try:
            managed_db = context.get(GRAPH_DB)
        except LookupError:
            managed_db = context.get("yaw/source_db", ManagedConnection)
        with managed_db.transaction() as conn:
            active_db_identity = resolve_db_identity(conn)
            actions = [
                action
                for action in actions
                if not action.spec.db_identity or action.spec.db_identity == active_db_identity
            ]
            if not actions:
                return
            has_work = any(isinstance(action, RowAction) or action.create or action.delete for action in actions)
            if not has_work:
                return
            for action in actions:
                if isinstance(action, TableAction) and action.create:
                    create_table(conn, action.spec)
            for action in actions:
                if not isinstance(action, RowAction):
                    continue
                table = identifier(action.spec.name)
                pk = action.spec.schema.primary_key
                where = " AND ".join(f"{identifier(name)}=?" for name in pk)
                is_virtual = action.spec.fts_unindexed is not None or action.spec.vec is not None
                if action.values is None or is_virtual:
                    if action.values is not None and action.spec.vec is not None:
                        blob = action.values[action.spec.vec.column]
                        ensure_vec0_table(
                            conn,
                            action.spec.name,
                            pk[0],
                            len(blob) // 4,
                            column=action.spec.vec.column,
                            metric=action.spec.vec.distance_metric,
                        )
                    conn.execute(f"DELETE FROM {table} WHERE {where}", action.key)
                if action.values is not None:
                    columns = tuple(action.spec.schema.columns)
                    names = ",".join(identifier(name) for name in columns)
                    placeholders = ",".join("?" for _ in columns)
                    conflict = ""
                    if not is_virtual:
                        updates = ",".join(
                            f"{identifier(name)}=excluded.{identifier(name)}" for name in columns if name not in pk
                        )
                        conflict = (
                            " ON CONFLICT ("
                            + ",".join(identifier(name) for name in pk)
                            + ") DO "
                            + ("UPDATE SET " + updates if updates else "NOTHING")
                        )
                    conn.execute(
                        f"INSERT INTO {table} ({names}) VALUES ({placeholders}){conflict}",
                        tuple(action.values[name] for name in columns),
                    )
            for action in actions:
                if isinstance(action, TableAction) and action.delete:
                    conn.execute(f"DROP TABLE {identifier(action.spec.name)}")
            if any(
                isinstance(action, RowAction) and action.spec.name not in {"source_chunks", "chunk_inference"}
                for action in actions
            ):
                context.get(VALIDATE_SQLITE)(conn)


@dataclass(frozen=True)
class RowHandlerFactory:
    def __call__(self, context: coco.ContextProvider, action: TableAction) -> RowHandler:
        return RowHandler(action.spec)


def shared_sink() -> coco.TargetActionSink[TableAction | RowAction]:
    return coco.TargetActionSink.from_fn(DatabaseSink(), child_handler=RowHandlerFactory())


@dataclass(frozen=True)
class RowHandler(coco.TargetHandler[dict[str, Any], bytes, None]):
    spec: TableSpec
    tracks_value_fingerprint = True

    def reconcile(
        self,
        key: coco.StableKey,
        desired: dict[str, Any] | coco.NonExistenceType,
        previous: Collection[bytes],
        prev_may_be_missing: bool,
        /,
    ) -> coco.TargetReconcileOutput[TableAction | RowAction, bytes, None] | None:
        if coco.is_non_existence(desired):
            if (
                self.spec.is_delta
                or (not previous and not prev_may_be_missing)
                or (previous and all(value == _DELETE_RECORD for value in previous))
            ):
                return None
            record = coco.NON_EXISTENCE
            values = None
        elif desired == DELETE_ROW:
            if not prev_may_be_missing and previous and all(value == _DELETE_RECORD for value in previous):
                return None
            record = _DELETE_RECORD
            values = None
        else:
            record = fingerprint_object(desired)
            if not prev_may_be_missing and previous and all(value == record for value in previous):
                return None
            values = desired
        if not isinstance(key, tuple):
            raise TypeError("SQLite row target requires a tuple primary key")
        return coco.TargetReconcileOutput[TableAction | RowAction, bytes, None](
            action=RowAction(self.spec, key, values), sink=shared_sink(), tracking_record=record
        )


class TableHandler(coco.TargetHandler[TableSpec, TableSpec, RowHandler]):
    def reconcile(
        self,
        key: coco.StableKey,
        desired: TableSpec | coco.NonExistenceType,
        previous: Collection[TableSpec],
        prev_may_be_missing: bool,
        /,
    ) -> coco.TargetReconcileOutput[TableAction | RowAction, TableSpec, RowHandler] | None:
        if coco.is_non_existence(desired):
            if not previous:
                return None
            (spec,) = previous
            record = coco.NON_EXISTENCE
            create = False
        else:
            spec = desired
            record = desired
            create = prev_may_be_missing or not previous
            if previous:
                if desired.vec is not None:
                    desired_base = _schema_base(desired, ignore_vec_dim=True)
                    if any(
                        fingerprint_object(_schema_base(old, ignore_vec_dim=True)) != fingerprint_object(desired_base)
                        for old in previous
                    ):
                        raise ValueError("Changed graph table schema requires a fresh index")
                    if desired.vec.dim is None:
                        spec = replace(desired, vec=next(iter(previous)).vec)
                        record = spec
                    elif any(old.vec is None or old.vec.dim != desired.vec.dim for old in previous):
                        create = True
                elif any(
                    fingerprint_object(_schema_base(old)) != fingerprint_object(_schema_base(spec)) for old in previous
                ):
                    raise ValueError("Changed graph table schema requires a fresh index")
        return coco.TargetReconcileOutput[TableAction | RowAction, TableSpec, RowHandler](
            action=TableAction(spec, coco.is_non_existence(desired), create),
            sink=shared_sink(),
            tracking_record=record,
        )


TABLES = coco.register_root_target_states_provider("yaw/managed_sqlite/tables", TableHandler())


async def mount_table[RowT](
    name: str,
    row_type: type[RowT],
    primary_key: Sequence[str],
    *,
    fts_unindexed: tuple[str, ...] | None = None,
    indexes: tuple[tuple[str, ...], ...] = (),
    db_identity: str = "",
    vec: VecSpec | None = None,
    is_delta: bool = False,
) -> sqlite.TableTarget[RowT]:
    schema = await sqlite.TableSchema[RowT].from_class(row_type, primary_key=list(primary_key))
    spec = TableSpec(
        name,
        sqlite.TableSchema(schema.columns, schema.primary_key),
        fts_unindexed,
        indexes,
        db_identity=db_identity,
        vec=vec,
        is_delta=is_delta,
    )
    # SDK inference includes the deletion sentinel in the row value type.
    provider = cast(
        coco.TargetStateProvider[dict[str, Any], None],
        await coco.mount_target(TABLES.target_state(TableKey(db_identity, name), spec)),
    )
    return sqlite.TableTarget(provider, schema)
