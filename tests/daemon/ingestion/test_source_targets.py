"""Native target ownership against actual SQLite and filesystem changes."""

from pathlib import Path

import cocoindex as coco
import pytest
from cocoindex.connectors import sqlite

from yaw.daemon.ingestion.source_targets import SOURCE_DB, SOURCE_ROOT, SourceSettings, source_app_main


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


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
        source_app_main,
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
            corrected = conn.execute("SELECT text FROM source_chunks").fetchall()
        assert corrected
        assert any("Corrected grounded" in row[0] for row in corrected)
        assert all("Original grounded" not in row[0] for row in corrected)
    finally:
        env.close()
        db.close()


def test_source_targets_replace_delete_and_restart(tmp_path: Path) -> None:
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
                source_app_main,
                SourceSettings(("**/*.md",), (), min_chunk_size=1),
            )
            try:
                app.update_blocking()
                with db.readonly() as conn:
                    rows = conn.execute("SELECT chunk_id, text, source_hash FROM source_chunks").fetchall()
                assert len(rows) == 1
                assert "Original grounded" in rows[0][1]
                app.update_blocking()
                with db.readonly() as conn:
                    assert conn.execute("SELECT chunk_id, text, source_hash FROM source_chunks").fetchall() == rows
                if phase == 1:
                    source.write_text("# Second\n\nReplacement source evidence.\n", encoding="utf-8")
                    app.update_blocking()
                    with db.readonly() as conn:
                        changed = conn.execute("SELECT text, source_hash FROM source_chunks").fetchall()
                    assert len(changed) == 1
                    assert "Replacement" in changed[0][0]
                    assert changed[0][1] != rows[0][2]
                    source.unlink()
                    app.update_blocking()
                    with db.readonly() as conn:
                        assert conn.execute("SELECT COUNT(*) FROM source_chunks").fetchone()[0] == 0
            finally:
                env.close()
    finally:
        db.close()
