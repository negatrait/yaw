"""Native source, inference, and declarative graph target composition at the daemon boundary."""

import asyncio
import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cocoindex as coco
from cocoindex.connectorkits.fingerprint import fingerprint_object
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
    ChunkInference,
    ChunkProcessOutput,
    EmbeddingEndpoint,
    OpenIEEndpoint,
    OpenIEInput,
    embed_text,
    extract_text,
)
from yaw.daemon.graph.planner import GraphPlanSettings, GraphPlanSummary, plan_graph_publication, summarize_graph_plan
from yaw.daemon.graph.storage.db import GRAPH_DB, get_meta, resolve_coco_state_root, resolve_db_identity
from yaw.daemon.graph.storage.hydration import (
    hydrate_ast_parent_context,
    resolve_chunk_text,
)
from yaw.daemon.graph.storage.managed_sqlite import VALIDATE_SQLITE, mount_table
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
INDEXING_ERROR_SINK = coco.ContextKey[Callable[[BaseException], None] | None]("yaw/indexing_error_sink")


@dataclass(frozen=True, slots=True)
class SyncOutcome:
    added_chunks: tuple[str, ...] = ()
    deleted_chunk_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class InferenceIdentity:
    program: str
    embedding: str


def _managed_db_identity(db: sqlite.ManagedConnection) -> str:
    with db.readonly() as conn:
        has_meta = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta'").fetchone()
        if has_meta is not None:
            existing = get_meta(conn, "db_identity")
            if existing:
                return existing
    with db.transaction() as conn:
        return resolve_db_identity(conn)


@coco.fn
async def process_chunk(
    metadata: SourceMetadata,
    target: sqlite.TableTarget[ChunkInference],
    identity: InferenceIdentity,
    settings: SourceSettings | None = None,
    frozen_text: str | None = None,
    db_identity: str = "",
) -> ChunkProcessOutput:
    """Declare a complete inference artifact only after both operations succeed."""
    _ = db_identity
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
                f"[Document AST Context ({pointer.rel_posix_path})]\n{ast_parent_ctx}\n\n[Target Chunk]\n{chunk_text}"
            )
    embedded, record = await asyncio.gather(
        embed_text(embed_req),
        extract_text(OpenIEInput(metadata.chunk_id, openie_passage, extraction.program_identity)),
    )
    row = ChunkInference(
        metadata.chunk_id,
        metadata.source_hash,
        extraction.program_identity,
        embeddings.adapter.profile.index_identity,
        json.dumps(asdict(record.record)),
        json.dumps([asdict(trace) for trace in record.traces]),
    )
    target.declare_row(row=row)
    return ChunkProcessOutput(
        pointer=pointer,
        text=chunk_text,
        inference=row,
        embedding=embedded.vector_f32,
    )


@coco.fn
async def process_file(
    file: FileLike,
    sources: sqlite.TableTarget[SourceMetadata],
    artifacts: sqlite.TableTarget[ChunkInference],
    settings: SourceSettings,
    identity: InferenceIdentity,
    db_identity: str = "",
) -> tuple[ChunkProcessOutput, ...]:
    source_root = coco.use_context(SOURCE_ROOT)
    target_file = source_root / file.file_path.path
    if not target_file.is_file():
        return ()
    chunking_settings = normalize_chunking_settings(settings)
    metadata_rows = await coco.use_mount(
        coco.component_subpath("source"),
        declare_source_file,
        file,
        sources,
        chunking_settings,
    )
    if not target_file.is_file():
        return ()
    try:
        outputs = await asyncio.gather(
            *(
                coco.use_mount(
                    coco.component_subpath("chunk", metadata.chunk_id),
                    process_chunk,
                    metadata,
                    artifacts,
                    identity,
                    chunking_settings,
                    None,
                    db_identity,
                )
                for metadata in metadata_rows
            )
        )
    except ValueError:
        if not target_file.is_file():
            return ()
        raise
    return tuple(outputs)


@coco.fn(memo=True, version=2)
async def publish_graph(
    settings: GraphPlanSettings,
    chunk_outputs: tuple[ChunkProcessOutput, ...],
    db_identity: str = "",
) -> GraphPlanSummary:
    """Plan and declare the atomic validated graph without storing vectors in LMDB memos."""
    plan = await plan_graph_publication(settings, chunk_outputs, db_identity=db_identity)
    await declare_tables(plan.tables, db_identity=db_identity)
    return summarize_graph_plan(plan)


class _CompletedHandle:
    async def ready(self) -> None:
        return None


class _CorpusFeedSubscriber:
    """Forwards live localfs file events into serialized CorpusIndexingLiveComponent updates."""

    def __init__(self, operator: coco.LiveComponentOperator) -> None:
        self._operator = operator
        self._lock = asyncio.Lock()

    async def _refresh(self) -> None:
        async with self._lock:
            await self._operator.update_full()

    async def update_all(self) -> None:
        await self._refresh()

    async def mark_ready(self) -> None:
        await self._operator.mark_ready()

    async def update(self, key: Any, value: Any) -> Any:
        _ = key, value
        await self._refresh()
        return _CompletedHandle()

    async def delete(self, key: Any) -> Any:
        _ = key
        await self._refresh()
        return _CompletedHandle()

    async def read_committed_state(self, key: Any) -> Any:
        _ = key
        return coco.NON_EXISTENCE

    async def write_committed_state(self, key: Any, value: Any) -> None:
        _ = key, value


class CorpusIndexingLiveComponent:
    """Native LiveComponent that coordinates file indexing and direct ChunkProcessOutput publication."""

    def __init__(
        self,
        files: Any,
        sources: sqlite.TableTarget[SourceMetadata],
        artifacts: sqlite.TableTarget[ChunkInference],
        settings: SourceSettings,
        identity: InferenceIdentity,
        graph_settings: GraphPlanSettings | None = None,
        db_identity: str = "",
        live: bool = True,
    ) -> None:
        self._files = files
        self._sources = sources
        self._artifacts = artifacts
        self._settings = settings
        self._identity = identity
        self._graph_settings = graph_settings
        self._db_identity = db_identity
        self._live = live
        self.last_summary: GraphPlanSummary | None = None

    async def process(self) -> GraphPlanSummary | None:
        try:
            entries = [item async for item in self._files.items()]
            file_batches = await asyncio.gather(
                *(
                    coco.use_mount(
                        coco.component_subpath("process_file", rel_path),
                        process_file,
                        file,
                        self._sources,
                        self._artifacts,
                        self._settings,
                        self._identity,
                        self._db_identity,
                    )
                    for rel_path, file in entries
                )
            )
            chunk_outputs = tuple(out for batch in file_batches for out in batch)
            if self._graph_settings is None:
                self.last_summary = None
                return None
            self.last_summary = await coco.use_mount(
                coco.component_subpath("publication"),
                publish_graph,
                self._graph_settings,
                chunk_outputs,
                self._db_identity,
            )
            return self.last_summary
        except BaseException as exc:
            try:
                sink = coco.use_context(INDEXING_ERROR_SINK)
            except KeyError:
                sink = None
            if sink is not None:
                sink(exc)
            raise

    async def process_live(self, operator: coco.LiveComponentOperator) -> None:
        items = self._files.items()
        if not self._live or not isinstance(items, coco.LiveMapView):
            await operator.update_full()
            await operator.mark_ready()
            return
        await items.watch(_CorpusFeedSubscriber(operator))


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
    live: bool = True,
) -> None:
    source_db = coco.use_context(SOURCE_DB)
    db_identity = _managed_db_identity(source_db)
    sources = await mount_table("source_chunks", SourceMetadata, ["chunk_id"], db_identity=db_identity)
    artifacts = await mount_table("chunk_inference", ChunkInference, ["chunk_id"], db_identity=db_identity)
    files = localfs.walk_dir(
        SOURCE_ROOT,
        recursive=True,
        live=live,
        path_matcher=build_source_path_matcher(settings),
        rescan_interval=None,
    )
    corpus_handle = await coco.mount(
        coco.component_subpath("corpus"),
        CorpusIndexingLiveComponent,
        files,
        sources,
        artifacts,
        settings,
        identity,
        graph_settings,
        db_identity,
        live,
    )
    await corpus_handle.ready()
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
    db_identity: str = "",
) -> ChunkProcessOutput:
    pointer, text = chunk
    metadata = frozen_chunk_metadata(pointer, text)
    sources.declare_row(row=metadata)
    return await process_chunk(metadata, artifacts, identity, frozen_text=text, db_identity=db_identity)


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
    settings: GraphPlanSettings | None = None,
) -> tuple[list[tuple[ChunkPointer, str]], list[tuple[ChunkPointer, str]]]:
    """Keep published, unchanged native declarations in retained while admitting unpublished or revised chunks."""
    stored: dict[str, tuple[SourceMetadata, str]] = {}
    published_inputs: dict[str, str] | None = None
    with db.readonly() as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name IN ('source_chunks', 'chunk_inference', 'chunks')"
            )
        }
        if tables == {"source_chunks", "chunk_inference", "chunks"}:
            if settings is not None:
                has_meta = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta'").fetchone()
                if has_meta is not None:
                    saved = get_meta(conn, "graph_input_fingerprints")
                    published_inputs = dict(json.loads(saved)) if saved is not None else {}
                else:
                    published_inputs = {}
            for start in range(0, len(chunks), 64):
                ids = [pointer.chunk_id for pointer, _ in chunks[start : start + 64]]
                placeholders = ",".join("?" for _ in ids)
                rows = conn.execute(
                    "SELECT s.chunk_id, s.rel_posix_path, s.source_hash, s.start_line, s.end_line, s.title, "
                    "s.ast_island_start, s.ast_island_end, ci.extraction_json "
                    "FROM source_chunks s "
                    "JOIN chunk_inference ci ON ci.chunk_id=s.chunk_id AND ci.source_hash=s.source_hash "
                    "JOIN chunks c ON c.chunk_id=s.chunk_id "
                    "AND c.rel_posix_path=s.rel_posix_path AND c.start_line=s.start_line "
                    "AND c.end_line=s.end_line AND c.title=s.title "
                    "AND c.ast_island_start=s.ast_island_start AND c.ast_island_end=s.ast_island_end "
                    f"WHERE s.chunk_id IN ({placeholders}) AND ci.program_identity=? AND ci.embedding_identity=?",
                    (*ids, identity.program, identity.embedding),
                )
                stored.update((row[0], (SourceMetadata(*row[:8]), str(row[8]))) for row in rows)
    retained = []
    changed = []
    for pointer, text in chunks:
        entry = stored.get(pointer.chunk_id)
        is_retained = False
        if entry is not None and entry[0] == frozen_chunk_metadata(pointer, text):
            if settings is None or published_inputs is None:
                is_retained = True
            else:
                expected_fp = fingerprint_object((settings, pointer, text, entry[1])).hex()
                is_retained = published_inputs.get(pointer.chunk_id) == expected_fp
        destination = retained if is_retained else changed
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
    settings: GraphPlanSettings | None = None,
    identity: InferenceIdentity | None = None,
) -> GraphPlanSummary | None:
    """Reconcile an explicit immutable candidate corpus through the native targets."""
    active_chunks = stream.chunks if isinstance(stream, FrozenChunkStream) else stream
    source_db = coco.use_context(SOURCE_DB)
    db_identity = _managed_db_identity(source_db)
    sources = await mount_table("source_chunks", SourceMetadata, ["chunk_id"], db_identity=db_identity)
    artifacts = await mount_table("chunk_inference", ChunkInference, ["chunk_id"], db_identity=db_identity)
    effective_identity = (
        identity
        if identity is not None
        else (
            InferenceIdentity(settings.program_identity, settings.embedding_identity)
            if settings is not None
            else InferenceIdentity("", "")
        )
    )
    chunk_outputs = tuple(
        await asyncio.gather(
            *(
                coco.use_mount(
                    coco.component_subpath("chunk", chunk[0].chunk_id),
                    process_frozen_chunk,
                    chunk,
                    sources,
                    artifacts,
                    effective_identity,
                    db_identity,
                )
                for chunk in active_chunks
            )
        )
    )
    if settings is None:
        return None
    return await coco.use_mount(
        coco.component_subpath("publication"),
        publish_graph,
        settings,
        chunk_outputs,
        db_identity,
    )


def bind_native_indexing(
    engine: YawGraphEngine,
    *,
    batch_size: int = 64,
    state_root: Path | None = None,
) -> YawGraphEngine:
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
            OpenIEEndpoint(
                program,
                program.rlm_sub_lm or program.orchestrator_lm,
                program_identity,
                progress_callback=engine.emit_indexing_progress,
            ),
        )
        resolved_state_root = resolve_coco_state_root(state_root if state_root is not None else engine.db_path)
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
            synonymy_threshold=engine.synonymy_threshold,
            synonymy_top_k=engine.synonymy_top_k,
            ast_context_hops=engine.ast_context_hops,
            context_window_tokens=program.context_window_tokens,
            max_output_tokens=program.max_output_tokens,
        )
        environment = coco.Environment(settings=coco.Settings.from_env(resolved_state_root), context_provider=context)
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
                    engine.managed_conn,
                    chunks,
                    InferenceIdentity(program_identity, embedding_identity),
                    settings=settings,
                )
                if not changed:
                    stream.chunks = tuple(retained)
                    summary = await app.update().result() or GraphPlanSummary(())
                    engine.report_indexing_batch(summary, len(chunks))
                    return summary
                summary = GraphPlanSummary(())
                for start in range(0, len(changed), batch_size):
                    stream.chunks = (*retained, *changed[: start + batch_size])
                    summary = await app.update().result() or GraphPlanSummary(())
                    engine.report_indexing_batch(summary, len(chunks))
                return summary

            return asyncio.run(run_streaming())
        finally:
            environment.close()

    engine.indexing_runner = execute
    return engine
