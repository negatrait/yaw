"""Native target ownership against actual SQLite and filesystem changes."""

import asyncio
import hashlib
from collections.abc import AsyncGenerator
from pathlib import Path

import cocoindex as coco
import pytest
from cocoindex.connectors import localfs, sqlite

import yaw.daemon.ingestion.source_targets as source_targets
from yaw.daemon.ingestion.source_targets import (
    SOURCE_DB,
    SOURCE_ROOT,
    SourceMetadata,
    SourceSettings,
    build_source_path_matcher,
    declare_source_file,
)


@coco.fn
async def source_target_fixture(settings: SourceSettings) -> None:
    table = await sqlite.mount_table_target(
        SOURCE_DB,
        table_name="source_chunks",
        table_schema=await sqlite.TableSchema[SourceMetadata].from_class(SourceMetadata, primary_key=["chunk_id"]),
    )
    files = localfs.walk_dir(
        SOURCE_ROOT,
        recursive=True,
        live=True,
        path_matcher=build_source_path_matcher(settings),
        rescan_interval=None,
    )
    await coco.mount_each(declare_source_file, files.items(), table, settings)


@coco.fn
async def declare_files_fixture(settings: SourceSettings) -> tuple[tuple[SourceMetadata, ...], ...]:
    table = await sqlite.mount_table_target(
        SOURCE_DB,
        table_name="source_chunks",
        table_schema=await sqlite.TableSchema[SourceMetadata].from_class(SourceMetadata, primary_key=["chunk_id"]),
    )
    files = localfs.walk_dir(
        SOURCE_ROOT,
        recursive=True,
        path_matcher=build_source_path_matcher(settings),
        rescan_interval=None,
    )
    return tuple([await declare_source_file(file, table, settings) async for _, file in files.items()])


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_native_live_source_edit_rename_delete(tmp_path: Path) -> None:
    root = tmp_path / "documents"
    root.mkdir()
    source = root / "source.md"
    source.write_text("Original grounded evidence.", encoding="utf-8")
    db = sqlite.connect(tmp_path / "yaw.db", load_vec=False)
    context = coco.ContextProvider()
    context.provide(SOURCE_DB, db)
    context.provide(SOURCE_ROOT, root)
    env = coco.Environment(settings=coco.Settings.from_env(tmp_path / "state"), context_provider=context)
    app = coco.App(
        coco.AppConfig(name="live_source", environment=env, max_inflight_components=4),
        source_target_fixture,
        SourceSettings(("**/*.md",), (), min_chunk_size=1),
    )
    handle = app.update(live=True)
    events = handle.watch()
    # UpdateHandle.watch() is an async generator annotated as AsyncIterator; aclose() is required for cleanup.
    assert isinstance(events, AsyncGenerator)
    ready = False

    async def expect_rows(expected: list[tuple[str, str]]) -> None:
        nonlocal ready
        expected = [(path, hashlib.sha256(text.encode()).hexdigest()) for path, text in expected]
        async with asyncio.timeout(15):
            async for event in events:
                assert event.stats.total.num_errors == 0
                ready = ready or event.status == coco.UpdateStatus.READY
                if not ready:
                    continue
                with db.readonly() as conn:
                    actual = conn.execute(
                        "SELECT rel_posix_path, source_hash FROM source_chunks ORDER BY rel_posix_path"
                    ).fetchall()
                if actual == expected and event.stats.total.num_in_progress == 0:
                    return
            raise AssertionError("Live source terminated before applying the expected change")

    try:
        await expect_rows([("source.md", "Original grounded evidence.")])
        source.write_text("Updated grounded evidence.", encoding="utf-8")
        await expect_rows([("source.md", "Updated grounded evidence.")])
        renamed = root / "renamed.md"
        source.rename(renamed)
        await expect_rows([("renamed.md", "Updated grounded evidence.")])
        renamed.unlink()
        await expect_rows([])
        await asyncio.wait_for(app.update().result(), timeout=10)
        await asyncio.wait_for(handle.result(), timeout=10)
        source.write_text("Evidence after live shutdown.", encoding="utf-8")
        await asyncio.sleep(0.2)
        with db.readonly() as conn:
            assert conn.execute("SELECT COUNT(*) FROM source_chunks").fetchone()[0] == 0
        await asyncio.wait_for(app.update().result(), timeout=10)
        with db.readonly() as conn:
            assert conn.execute("SELECT source_hash FROM source_chunks").fetchall() == [
                (hashlib.sha256(source.read_bytes()).hexdigest(),)
            ]
    finally:
        await events.aclose()
        env.close()
        db.close()


@pytest.mark.anyio
async def test_failed_replacement_preserves_targets_and_retries(tmp_path: Path) -> None:
    root = tmp_path / "documents"
    root.mkdir()
    source = root / "source.json"
    source.write_text('{"body": "Original grounded evidence."}', encoding="utf-8")
    db = sqlite.connect(tmp_path / "yaw.db", load_vec=False)
    context = coco.ContextProvider()
    context.provide(SOURCE_DB, db)
    context.provide(SOURCE_ROOT, root)
    env = coco.Environment(settings=coco.Settings.from_env(tmp_path / "state"), context_provider=context)
    app = coco.App(
        coco.AppConfig(name="failure_test", environment=env, max_inflight_components=4),
        source_target_fixture,
        SourceSettings(("**/*.json",), (), min_chunk_size=1),
    )
    try:
        await app.update()
        with db.readonly() as conn:
            original = conn.execute("SELECT * FROM source_chunks ORDER BY chunk_id").fetchall()
        assert original
        source.write_text('{"body":', encoding="utf-8")
        failed = app.update()
        await failed
        stats = failed.stats()
        assert stats is not None
        assert stats.total.num_errors == 1
        with db.readonly() as conn:
            assert conn.execute("SELECT * FROM source_chunks ORDER BY chunk_id").fetchall() == original
        source.write_text('{"body": "Corrected grounded evidence."}', encoding="utf-8")
        repaired = app.update()
        await repaired
        stats = repaired.stats()
        assert stats is not None
        assert stats.total.num_errors == 0
        with db.readonly() as conn:
            corrected = conn.execute("SELECT source_hash FROM source_chunks").fetchall()
        assert corrected
        assert all(row[0] == hashlib.sha256(source.read_bytes()).hexdigest() for row in corrected)
    finally:
        env.close()
        db.close()


def test_source_targets_replace_delete_and_restart(tmp_path: Path) -> None:
    assert not hasattr(source_targets, "SourceChunk")
    assert not hasattr(source_targets, "source_app_main")
    root = tmp_path / "documents"
    root.mkdir()
    source = root / "source.md"
    source.write_text("# First\n\nOriginal grounded source evidence.\n", encoding="utf-8")
    db = sqlite.connect(tmp_path / "yaw.db", load_vec=False)
    try:
        for phase in range(2):
            context = coco.ContextProvider()
            context.provide(SOURCE_DB, db)
            context.provide(SOURCE_ROOT, root)
            env = coco.Environment(settings=coco.Settings.from_env(tmp_path / "state"), context_provider=context)
            app = coco.App(
                coco.AppConfig(name="source_test", environment=env, max_inflight_components=4),
                declare_files_fixture,
                SourceSettings(("**/*.md",), (), min_chunk_size=1),
            )
            try:
                batches = app.update_blocking()
                assert isinstance(batches, tuple) and len(batches) == 1
                assert isinstance(batches[0], tuple) and len(batches[0]) == 1
                assert isinstance(batches[0][0], SourceMetadata)
                assert not hasattr(batches[0][0], "text")
                assert "Original grounded source evidence." not in repr(batches)
                with db.readonly() as conn:
                    rows = conn.execute("SELECT chunk_id, title, source_hash FROM source_chunks").fetchall()
                    assert "text" not in {row[1] for row in conn.execute("PRAGMA table_info(source_chunks)")}
                assert len(rows) == 1
                assert rows[0][1] == "First"
                assert app.update_blocking() == batches
                with db.readonly() as conn:
                    assert conn.execute("SELECT chunk_id, title, source_hash FROM source_chunks").fetchall() == rows
                if phase == 1:
                    source.write_text("# Second\n\nReplacement source evidence.\n", encoding="utf-8")
                    updated_batches = app.update_blocking()
                    assert len(updated_batches) == 1 and isinstance(updated_batches[0][0], SourceMetadata)
                    assert "Replacement source evidence." not in repr(updated_batches)
                    with db.readonly() as conn:
                        changed = conn.execute("SELECT title, source_hash FROM source_chunks").fetchall()
                    assert len(changed) == 1
                    assert changed[0][0] == "Second"
                    assert changed[0][1] != rows[0][2]
                    source.unlink()
                    assert app.update_blocking() == ()
                    with db.readonly() as conn:
                        assert conn.execute("SELECT COUNT(*) FROM source_chunks").fetchone()[0] == 0
            finally:
                env.close()
    finally:
        db.close()


def test_source_settings_included_files_selects_only_requested_paths(tmp_path: Path) -> None:
    root = tmp_path / "documents"
    (root / "nested").mkdir(parents=True)
    (root / "unselected_dir").mkdir(parents=True)
    (root / "keep.md").write_text("Root selected evidence.", encoding="utf-8")
    (root / "ignore.md").write_text("Root sibling ignored.", encoding="utf-8")
    (root / "nested" / "keep_nested.md").write_text("Nested selected evidence.", encoding="utf-8")
    (root / "nested" / "ignore_nested.md").write_text("Nested sibling ignored.", encoding="utf-8")
    (root / "unselected_dir" / "other.md").write_text("Other directory ignored.", encoding="utf-8")

    settings = SourceSettings(
        ("**/*.md",),
        (),
        min_chunk_size=1,
        included_files=("keep.md", "nested/keep_nested.md"),
    )
    matcher = build_source_path_matcher(settings)
    assert matcher.is_file_included(Path("keep.md"))
    assert not matcher.is_file_included(Path("ignore.md"))
    assert matcher.is_dir_included(Path("nested"))
    assert not matcher.is_dir_included(Path("unselected_dir"))
    assert matcher.is_file_included(Path("nested/keep_nested.md"))
    assert not matcher.is_file_included(Path("nested/ignore_nested.md"))

    db = sqlite.connect(tmp_path / "yaw.db", load_vec=False)
    context = coco.ContextProvider()
    context.provide(SOURCE_DB, db)
    context.provide(SOURCE_ROOT, root)
    env = coco.Environment(settings=coco.Settings.from_env(tmp_path / "state"), context_provider=context)
    app = coco.App(
        coco.AppConfig(name="selected_files", environment=env, max_inflight_components=4),
        declare_files_fixture,
        settings,
    )
    try:
        batches = app.update_blocking()
        returned_paths = sorted(row.rel_posix_path for batch in batches for row in batch)
        assert returned_paths == ["keep.md", "nested/keep_nested.md"]
        with db.readonly() as conn:
            db_paths = [
                row[0]
                for row in conn.execute("SELECT rel_posix_path FROM source_chunks ORDER BY rel_posix_path").fetchall()
            ]
        assert db_paths == ["keep.md", "nested/keep_nested.md"]
    finally:
        env.close()
        db.close()
