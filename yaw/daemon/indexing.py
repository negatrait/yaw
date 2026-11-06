"""Native source, inference, and declarative graph target composition at the daemon boundary."""

import asyncio
import hashlib
import json
import sqlite3
import threading
import uuid
import weakref
from collections.abc import AsyncIterator, Callable, Collection, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cocoindex as coco
import msgspec
from cocoindex.connectorkits import statediff
from cocoindex.connectors import localfs, sqlite
from cocoindex.resources.file import FileLike, PatternFilePathMatcher

from yaw.core import ChunkPointer
from yaw.daemon.graph.engine import YawGraphEngine
from yaw.daemon.graph.inference import (
    DSPyEmbeddingAdapter,
    EmbeddingRequest,
    YawProgram,
    hierarchy_judge_fingerprint,
    label_judge_fingerprint,
    openie_fingerprint,
    related_judge_fingerprint,
    shacl_fingerprint,
)
from yaw.daemon.graph.inference.components import (
    EMBEDDING_ENDPOINT,
    OPENIE_ENDPOINT,
    EmbeddingEndpoint,
    OpenIEEndpoint,
    OpenIEInput,
    embed_text,
    extract_text,
)
from yaw.daemon.graph.planner import GraphPlanSettings, GraphPlanSummary, plan_graph_publication, summarize_graph_plan
from yaw.daemon.graph.storage.db import GRAPH_DB
from yaw.daemon.graph.storage.hydration import (
    hydrate_ast_parent_context,
    resolve_chunk_text,
)
from yaw.daemon.graph.storage.managed_sqlite import VALIDATE_SQLITE
from yaw.daemon.graph.storage.rows import declare_tables, validate_graph_state
from yaw.daemon.ingestion.source_targets import (
    SOURCE_DB,
    SOURCE_ROOT,
    SourceMetadata,
    SourceSettings,
    build_source_path_matcher,
    declare_source_file,
    normalize_chunking_settings,
)

RELOAD_SIGNAL_ROOT = coco.ContextKey[Path]("yaw/reload_signal_root", detect_change=True)
RELOAD_SIGNAL_HANDLER = coco.ContextKey[Callable[[str], None]]("yaw/reload_signal_handler")


@dataclass(frozen=True, slots=True)
class SyncOutcome:
    added_chunks: tuple[str, ...] = ()
    deleted_chunk_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class InferenceIdentity:
    program: str
    embedding: str


@dataclass(frozen=True, slots=True)
class ChunkInference:
    chunk_id: str
    source_hash: str
    program_identity: str
    embedding_identity: str
    extraction_json: str
    traces_json: str


class ChunkStateRecord(msgspec.Struct, frozen=True):
    chunk_id: str
    source_hash: str
    program_identity: str
    embedding_identity: str


@dataclass(frozen=True, slots=True)
class ChunkStateAction:
    chunk_id: str
    desired: ChunkStateRecord | coco.NonExistenceType


class GraphDeltaCoordinator:
    """Event-driven state-diff coordinator for declarative live graph publication (zero polling)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._dirty_generation = 0
        self._flushed_generation = 0
        self._ready_epoch = 0
        self._consumed_epoch = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._event: asyncio.Event | None = None
        self.last_transitions: tuple[ChunkStateAction, ...] = ()
        self.last_plan_summary: GraphPlanSummary | None = None
        self.recovery_nonce: str = ""
        self.last_error: BaseException | None = None
        self._cycle_error: BaseException | None = None

    def record_error(self, exc: BaseException) -> None:
        with self._lock:
            self.last_error = exc
            self._cycle_error = exc

    def consume_cycle_error(self) -> BaseException | None:
        with self._lock:
            err = self._cycle_error
            self._cycle_error = None
            return err

    def pop_error(self) -> BaseException | None:
        with self._lock:
            err = self.last_error
            self.last_error = None
            self._cycle_error = None
            return err

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        with self._lock:
            if self._loop is not loop or self._event is None:
                self._loop = loop
                self._event = asyncio.Event()
                if self._ready_epoch > self._consumed_epoch:
                    self._event.set()

    def record_transitions(self, actions: Sequence[ChunkStateAction]) -> None:
        if not actions:
            return
        with self._lock:
            self.last_transitions = tuple(actions)
            self._dirty_generation += 1

    def notify_external_change(self) -> None:
        with self._lock:
            self._dirty_generation += 1
        self.flush_ready()

    def flush_ready(self) -> None:
        with self._lock:
            if self._dirty_generation <= self._flushed_generation:
                return
            self._flushed_generation = self._dirty_generation
            self._ready_epoch += 1
            loop = self._loop
            event = self._event
        if loop is not None and event is not None and not loop.is_closed():
            loop.call_soon_threadsafe(event.set)

    def mark_published(self, summary: GraphPlanSummary) -> None:
        with self._lock:
            self.last_plan_summary = summary
            self._flushed_generation = self._dirty_generation
            self._consumed_epoch = self._ready_epoch
            if self._event is not None:
                self._event.clear()

    async def wait_for_delta(self) -> None:
        loop = asyncio.get_running_loop()
        self.bind_loop(loop)
        while True:
            with self._lock:
                if self._ready_epoch > self._consumed_epoch:
                    self._consumed_epoch = self._ready_epoch
                    if self._event is not None:
                        self._event.clear()
                    return
                event = self._event
            assert event is not None
            await event.wait()


_COORDINATOR_LOCK = threading.Lock()
_COORDINATORS: dict[int, GraphDeltaCoordinator] = {}


def get_graph_delta_coordinator(db: sqlite.ManagedConnection) -> GraphDeltaCoordinator:
    db_id = id(db)
    with _COORDINATOR_LOCK:
        coordinator = _COORDINATORS.get(db_id)
        if coordinator is None:
            coordinator = GraphDeltaCoordinator()
            _COORDINATORS[db_id] = coordinator
            weakref.finalize(db, _COORDINATORS.pop, db_id, None)
        return coordinator


class ChunkStateTargetHandler(coco.TargetHandler[ChunkStateRecord, ChunkStateRecord]):
    """Declarative state-diff handler that triggers graph publication when chunk state changes."""

    def __init__(self) -> None:
        self.sink = coco.TargetActionSink.from_fn(self.apply_actions)

    def apply_actions(self, context: coco.ContextProvider, actions: Sequence[ChunkStateAction]) -> None:
        if not actions:
            return
        db = context.get(SOURCE_DB)
        get_graph_delta_coordinator(db).record_transitions(actions)

    def reconcile(
        self,
        key: coco.StableKey,
        desired: ChunkStateRecord | coco.NonExistenceType,
        previous: Collection[ChunkStateRecord],
        prev_may_be_missing: bool,
        /,
    ) -> coco.TargetReconcileOutput[Any, Any, Any] | None:
        record = coco.NON_EXISTENCE if coco.is_non_existence(desired) else desired
        change = statediff.diff(
            statediff.TrackingRecordTransition(
                desired=record,
                prev=previous,
                prev_may_be_missing=prev_may_be_missing,
            )
        )
        if change is None:
            return None
        return coco.TargetReconcileOutput(
            action=ChunkStateAction(str(key), record),
            sink=self.sink,
            tracking_record=record,
        )


CHUNK_STATE_TARGET = coco.register_root_target_states_provider(
    "yaw/indexing/chunk_state",
    ChunkStateTargetHandler(),
)


def declare_chunk_state(record: ChunkStateRecord) -> None:
    coco.declare_target_state(CHUNK_STATE_TARGET.target_state(record.chunk_id, record))


@coco.fn
async def process_chunk(
    metadata: SourceMetadata,
    target: sqlite.TableTarget[ChunkInference],
    identity: InferenceIdentity,
    settings: SourceSettings | None = None,
    frozen_text: str | None = None,
) -> None:
    """Declare a complete inference artifact only after both operations succeed."""
    try:
        embeddings = coco.use_context(EMBEDDING_ENDPOINT)
        extraction = coco.use_context(OPENIE_ENDPOINT)
        if (
            identity.program != extraction.program_identity
            or identity.embedding != embeddings.adapter.profile.index_identity
        ):
            raise ValueError("Indexing identity does not match the bound inference resources")
        pointer = ChunkPointer(
            metadata.chunk_id,
            metadata.rel_posix_path,
            metadata.start_line,
            metadata.end_line,
            metadata.title,
            metadata.ast_island_start,
            metadata.ast_island_end,
        )
        source_root = coco.use_context(SOURCE_ROOT)
        effective = settings if settings is not None else SourceSettings((), ())
        chunk_text = (
            frozen_text
            if frozen_text is not None
            else resolve_chunk_text(
                pointer,
                source_root,
                chunk_size=effective.chunk_size,
                chunk_overlap=effective.chunk_overlap,
                min_chunk_size=effective.min_chunk_size,
            )
        )
        fmt_chunk = getattr(
            embeddings.adapter,
            "format_chunk_for_embedding",
            lambda p, t: DSPyEmbeddingAdapter.format_chunk_for_embedding(embeddings.adapter, p, t),
        )
        embedding_text = fmt_chunk(pointer, chunk_text)
        if hasattr(embeddings.adapter, "prepare_request"):
            embed_req = embeddings.adapter.prepare_request([embedding_text], batch_size=64, stage="chunk_embed")
        else:
            embed_req = EmbeddingRequest(
                texts=(embedding_text,),
                profile=embeddings.adapter.profile,
                task_mode=getattr(embeddings.adapter, "task_mode", "distinct-titled"),
                batch_size=64,
                max_batch_chars=getattr(embeddings.adapter, "max_batch_chars", 3500),
                stage="chunk_embed",
            )
        openie_passage = chunk_text
        if frozen_text is None and pointer.start_line > 1:
            ast_parent_ctx = hydrate_ast_parent_context(pointer, source_root, hops=1)
            if ast_parent_ctx:
                openie_passage = (
                    f"[Document AST Context ({pointer.rel_posix_path})]\n{ast_parent_ctx}\n\n"
                    f"[Target Chunk]\n{chunk_text}"
                )
        _, record = await asyncio.gather(
            embed_text(embed_req),
            extract_text(OpenIEInput(metadata.chunk_id, openie_passage, extraction.program_identity)),
        )
        target.declare_row(
            row=ChunkInference(
                metadata.chunk_id,
                metadata.source_hash,
                extraction.program_identity,
                embeddings.adapter.profile.index_identity,
                json.dumps(asdict(record.record)),
                json.dumps([asdict(trace) for trace in record.traces]),
            )
        )
        declare_chunk_state(
            ChunkStateRecord(
                metadata.chunk_id,
                metadata.source_hash,
                extraction.program_identity,
                embeddings.adapter.profile.index_identity,
            )
        )
    except BaseException as exc:
        get_graph_delta_coordinator(coco.use_context(SOURCE_DB)).record_error(exc)
        raise


@coco.fn
async def process_file(
    file: FileLike,
    sources: sqlite.TableTarget[SourceMetadata],
    artifacts: sqlite.TableTarget[ChunkInference],
    settings: SourceSettings,
    identity: InferenceIdentity,
) -> None:
    try:
        chunking_settings = normalize_chunking_settings(settings)
        metadata_rows = await declare_source_file(file, sources, chunking_settings)
        for metadata in metadata_rows:
            await coco.mount(
                coco.component_subpath("chunk", metadata.chunk_id),
                process_chunk,
                metadata,
                artifacts,
                identity,
                chunking_settings,
            )
    except BaseException as exc:
        get_graph_delta_coordinator(coco.use_context(SOURCE_DB)).record_error(exc)
        raise


def compute_active_inference_fingerprint(
    conn: sqlite3.Connection,
    program_identity: str,
    embedding_identity: str,
) -> str:
    """Compute a deterministic digest of all active source + inference rows in yaw.db."""
    hasher = hashlib.sha256()
    hasher.update(f"{program_identity}\0{embedding_identity}\n".encode())
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('source_chunks', 'chunk_inference')"
        ).fetchall()
    }
    if {"source_chunks", "chunk_inference"} <= tables:
        rows = conn.execute(
            """
            SELECT s.chunk_id, s.rel_posix_path, s.source_hash, s.start_line, s.end_line,
                   s.title, s.ast_island_start, s.ast_island_end,
                   i.program_identity, i.embedding_identity, i.extraction_json
            FROM source_chunks s
            JOIN chunk_inference i ON i.chunk_id = s.chunk_id
            WHERE i.source_hash = s.source_hash
              AND i.program_identity = ?
              AND i.embedding_identity = ?
            ORDER BY s.chunk_id
            """,
            (program_identity, embedding_identity),
        ).fetchall()
        for row in rows:
            hasher.update(json.dumps(tuple(row), ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            hasher.update(b"\n")
    return hasher.hexdigest()


@coco.fn(memo=True, version=7)
async def publish_graph(
    settings: GraphPlanSettings,
    state_fingerprint: str,
    recovery_nonce: str = "",
    frozen_text: dict[str, str] | None = None,
) -> GraphPlanSummary:
    """Plan and declare the atomic validated graph without storing vectors in LMDB memos."""
    _ = state_fingerprint, recovery_nonce
    plan = await plan_graph_publication(settings, frozen_text=frozen_text)
    await declare_tables(plan.tables)
    return summarize_graph_plan(plan)


def _is_published_graph_aligned(conn: sqlite3.Connection, expected_chunk_ids: tuple[str, ...]) -> bool:
    table_exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunks'").fetchone()
    if table_exists is None:
        return not expected_chunk_ids
    actual = tuple(row[0] for row in conn.execute("SELECT chunk_id FROM chunks ORDER BY chunk_id").fetchall())
    return actual == expected_chunk_ids


class _SubscriberProxy:
    """Forwards LiveMapSubscriber calls and flushes committed state-diff transitions to the coordinator."""

    def __init__(self, inner: Any, coordinator: GraphDeltaCoordinator) -> None:
        self._inner = inner
        self._coordinator = coordinator

    async def update_all(self) -> None:
        await self._inner.update_all()
        self._coordinator.flush_ready()

    async def mark_ready(self) -> None:
        await self._inner.mark_ready()

    async def update(self, key: Any, value: Any) -> Any:
        handle = await self._inner.update(key, value)
        await handle.ready()
        self._coordinator.flush_ready()
        return handle

    async def delete(self, key: Any) -> Any:
        handle = await self._inner.delete(key)
        await handle.ready()
        self._coordinator.flush_ready()
        return handle

    async def read_committed_state(self, key: Any) -> Any:
        return await self._inner.read_committed_state(key)

    async def write_committed_state(self, key: Any, value: Any) -> None:
        await self._inner.write_committed_state(key, value)


class _DeltaNotifyingLiveItems:
    """Wraps a LiveMapView so state-diff transitions flush after file components commit."""

    def __init__(self, inner: Any, coordinator: GraphDeltaCoordinator) -> None:
        self._inner = inner
        self._coordinator = coordinator

    def __aiter__(self) -> AsyncIterator[tuple[str, FileLike]]:
        return self._inner.__aiter__()

    async def watch(self, subscriber: Any) -> None:
        await self._inner.watch(_SubscriberProxy(subscriber, self._coordinator))


class GraphPublicationLiveComponent:
    """Event-driven LiveComponent that publishes the validated graph when chunk state transitions commit."""

    def __init__(self, settings: GraphPlanSettings) -> None:
        self._settings = settings

    async def process(self) -> None:
        db = coco.use_context(GRAPH_DB)
        coordinator = get_graph_delta_coordinator(coco.use_context(SOURCE_DB))
        err = coordinator.consume_cycle_error()
        if err is not None:
            raise err
        with db.readonly() as conn:
            state_fingerprint = compute_active_inference_fingerprint(
                conn,
                self._settings.program_identity,
                self._settings.embedding_identity,
            )
        try:
            summary = await coco.use_mount(
                coco.component_subpath("publication"),
                publish_graph,
                self._settings,
                state_fingerprint,
                coordinator.recovery_nonce,
            )
        except BaseException as exc:
            coordinator.record_error(exc)
            raise
        coordinator.mark_published(summary)

    async def _update_and_verify(self, operator: coco.LiveComponentOperator) -> None:
        await operator.update_full()
        db = coco.use_context(GRAPH_DB)
        coordinator = get_graph_delta_coordinator(coco.use_context(SOURCE_DB))
        summary = coordinator.last_plan_summary
        if summary is None:
            return
        with db.readonly() as conn:
            aligned = _is_published_graph_aligned(conn, summary.chunk_ids)
        if not aligned:
            coordinator.recovery_nonce = uuid.uuid4().hex
            await operator.update_full()

    async def process_live(self, operator: coco.LiveComponentOperator) -> None:
        coordinator = get_graph_delta_coordinator(coco.use_context(SOURCE_DB))
        coordinator.bind_loop(asyncio.get_running_loop())
        await self._update_and_verify(operator)
        await operator.mark_ready()
        while True:
            await coordinator.wait_for_delta()
            await self._update_and_verify(operator)


@coco.fn
async def process_reload_signal(file: FileLike) -> None:
    raw = await file.read_text()
    if not raw.strip():
        return
    handler = coco.use_context(RELOAD_SIGNAL_HANDLER)
    await asyncio.to_thread(handler, raw)


@coco.fn
async def indexing_app_main(
    settings: SourceSettings,
    identity: InferenceIdentity,
    graph_settings: GraphPlanSettings | None = None,
    watch_reload_signal: bool = False,
) -> None:
    sources = await sqlite.mount_table_target(
        SOURCE_DB,
        table_name="source_chunks",
        table_schema=await sqlite.TableSchema[SourceMetadata].from_class(SourceMetadata, primary_key=["chunk_id"]),
    )
    artifacts = await sqlite.mount_table_target(
        SOURCE_DB,
        table_name="chunk_inference",
        table_schema=await sqlite.TableSchema[ChunkInference].from_class(ChunkInference, primary_key=["chunk_id"]),
    )
    coordinator = get_graph_delta_coordinator(coco.use_context(SOURCE_DB))
    files = localfs.walk_dir(
        SOURCE_ROOT,
        recursive=True,
        live=True,
        path_matcher=build_source_path_matcher(settings),
        rescan_interval=None,
    )
    files_handle = await coco.mount_each(
        process_file,
        _DeltaNotifyingLiveItems(files.items(), coordinator),
        sources,
        artifacts,
        settings,
        identity,
    )
    if graph_settings is not None:
        await files_handle.ready()
        graph_handle = await coco.mount(
            coco.component_subpath("graph"),
            GraphPublicationLiveComponent,
            graph_settings,
        )
        await graph_handle.ready()
    if watch_reload_signal:
        signal_files = localfs.walk_dir(
            RELOAD_SIGNAL_ROOT,
            recursive=False,
            live=True,
            path_matcher=PatternFilePathMatcher(included_patterns=["reload.signal"]),
            rescan_interval=None,
        )
        await coco.mount_each(
            coco.component_subpath("reload_signal"),
            process_reload_signal,
            signal_files.items(),
        )


@coco.fn
async def process_frozen_chunk(
    chunk: tuple[ChunkPointer, str],
    sources: sqlite.TableTarget[SourceMetadata],
    artifacts: sqlite.TableTarget[ChunkInference],
    identity: InferenceIdentity,
) -> None:
    pointer, text = chunk
    metadata = frozen_chunk_metadata(pointer, text)
    sources.declare_row(row=metadata)
    await process_chunk(metadata, artifacts, identity, frozen_text=text)


def frozen_chunk_metadata(pointer: ChunkPointer, text: str) -> SourceMetadata:
    """Use the same occurrence/revision key for admission and native declaration."""
    return SourceMetadata(
        pointer.chunk_id,
        pointer.rel_posix_path,
        hashlib.sha256(text.encode()).hexdigest(),
        pointer.start_line,
        pointer.end_line,
        pointer.title,
        pointer.ast_island_start,
        pointer.ast_island_end,
    )


def partition_frozen_chunks(
    db: sqlite.ManagedConnection,
    chunks: Sequence[tuple[ChunkPointer, str]],
    identity: InferenceIdentity,
) -> tuple[list[tuple[ChunkPointer, str]], list[tuple[ChunkPointer, str]]]:
    """Keep unchanged native declarations present while admitting changed revisions."""
    stored: dict[str, SourceMetadata] = {}
    with db.readonly() as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('source_chunks', 'chunk_inference')"
            )
        }
        if tables == {"source_chunks", "chunk_inference"}:
            for start in range(0, len(chunks), 64):
                ids = [pointer.chunk_id for pointer, _ in chunks[start : start + 64]]
                placeholders = ",".join("?" for _ in ids)
                rows = conn.execute(
                    "SELECT s.chunk_id, s.rel_posix_path, s.source_hash, s.start_line, s.end_line, s.title, "
                    "s.ast_island_start, s.ast_island_end FROM source_chunks s JOIN chunk_inference ci "
                    "ON ci.chunk_id=s.chunk_id AND ci.source_hash=s.source_hash "
                    f"WHERE s.chunk_id IN ({placeholders}) AND ci.program_identity=? AND ci.embedding_identity=?",
                    (*ids, identity.program, identity.embedding),
                )
                stored.update((row[0], SourceMetadata(*row)) for row in rows)
    retained = []
    changed = []
    for pointer, text in chunks:
        destination = retained if stored.get(pointer.chunk_id) == frozen_chunk_metadata(pointer, text) else changed
        destination.append((pointer, text))
    return retained, changed


class FrozenChunkStream:
    """Dynamic chunk stream for batching frozen candidate indexing through a single CocoIndex app."""

    __slots__ = ("chunks",)

    def __init__(self, chunks: tuple[tuple[ChunkPointer, str], ...] = ()) -> None:
        self.chunks = chunks


@coco.fn
async def frozen_indexing_app_main(
    stream: FrozenChunkStream | tuple[tuple[ChunkPointer, str], ...],
    settings: GraphPlanSettings,
) -> GraphPlanSummary:
    """Reconcile an explicit immutable candidate corpus through the native targets."""
    active_chunks = stream.chunks if isinstance(stream, FrozenChunkStream) else stream
    sources = await sqlite.mount_table_target(
        SOURCE_DB,
        table_name="source_chunks",
        table_schema=await sqlite.TableSchema[SourceMetadata].from_class(SourceMetadata, primary_key=["chunk_id"]),
    )
    artifacts = await sqlite.mount_table_target(
        SOURCE_DB,
        table_name="chunk_inference",
        table_schema=await sqlite.TableSchema[ChunkInference].from_class(ChunkInference, primary_key=["chunk_id"]),
    )
    identity = InferenceIdentity(settings.program_identity, settings.embedding_identity)
    handles = [
        await coco.mount(
            coco.component_subpath("chunk", chunk[0].chunk_id),
            process_frozen_chunk,
            chunk,
            sources,
            artifacts,
            identity,
        )
        for chunk in active_chunks
    ]
    for handle in handles:
        await handle.ready()
    coordinator = get_graph_delta_coordinator(coco.use_context(SOURCE_DB))
    error = coordinator.pop_error()
    if error is not None:
        raise error
    with coco.use_context(SOURCE_DB).readonly() as conn:
        fingerprint = compute_active_inference_fingerprint(conn, settings.program_identity, settings.embedding_identity)
    return await coco.use_mount(
        coco.component_subpath("publication"),
        publish_graph,
        settings,
        fingerprint,
        frozen_text={p.chunk_id: text for p, text in active_chunks},
    )


def bind_native_indexing(engine: YawGraphEngine, *, batch_size: int = 64) -> YawGraphEngine:
    """Compose frozen indexing at the daemon boundary; graph code owns no scheduler."""

    def execute(chunks: Sequence[tuple[ChunkPointer, str]], program: YawProgram) -> GraphPlanSummary:
        program_identity = openie_fingerprint(program)
        embedding_identity = str(engine.embedder.profile.index_identity)
        context = coco.ContextProvider()
        context.provide(SOURCE_DB, engine.managed_conn)
        context.provide(GRAPH_DB, engine.managed_conn)
        context.provide(VALIDATE_SQLITE, validate_graph_state)
        context.provide(SOURCE_ROOT, engine.workspace_root)
        context.provide(EMBEDDING_ENDPOINT, EmbeddingEndpoint(engine.embedder))
        context.provide(
            OPENIE_ENDPOINT,
            OpenIEEndpoint(program, program.rlm_sub_lm or program.orchestrator_lm, program_identity),
        )
        state_root = Path(engine.db_path).resolve().parent / "candidate_coco" / Path(engine.db_path).name
        settings = GraphPlanSettings(
            corpus_root=engine.workspace_root,
            program_identity=program_identity,
            embedding_identity=embedding_identity,
            synonym_identity=":".join(
                (
                    label_judge_fingerprint(program),
                    hierarchy_judge_fingerprint(program),
                    related_judge_fingerprint(program),
                )
            ),
            shacl_identity=shacl_fingerprint(program),
            auto_merge_cosine_gate=engine.auto_merge_cosine_gate,
            auto_merge_difflib_gate=engine.auto_merge_difflib_gate,
            synonymy_threshold=engine.synonymy_threshold,
            synonymy_top_k=engine.synonymy_top_k,
            ast_context_hops=engine.ast_context_hops,
            context_window_tokens=program.context_window_tokens,
            max_output_tokens=program.max_output_tokens,
        )
        environment = coco.Environment(settings=coco.Settings.from_env(state_root), context_provider=context)
        stream = FrozenChunkStream()
        try:
            app = coco.App(
                coco.AppConfig(name="yaw_frozen_indexing", environment=environment, max_inflight_components=4),
                frozen_indexing_app_main,
                stream,
                settings,
            )

            async def run_streaming() -> GraphPlanSummary:
                retained, changed = partition_frozen_chunks(
                    engine.managed_conn, chunks, InferenceIdentity(program_identity, embedding_identity)
                )
                if not changed:
                    stream.chunks = tuple(retained)
                    return await app.update().result()
                summary = GraphPlanSummary(())
                for start in range(0, len(changed), batch_size):
                    stream.chunks = (*retained, *changed[: start + batch_size])
                    summary = await app.update().result()
                return summary

            return asyncio.run(run_streaming())
        finally:
            environment.close()

    engine.indexing_runner = execute
    return engine
