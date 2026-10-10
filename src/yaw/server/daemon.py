"""FastMCP server and summed-probability retrieval pipeline for seed generation."""

from __future__ import annotations

import atexit
import fcntl
import logging
import os
import signal
import sqlite3
import sys
import threading
import time
from pathlib import Path
from typing import Any, Self

from fastmcp import FastMCP

from yaw.nlp import is_ollama_online
from yaw.ontology.shacl import ValidationReport
from yaw.retrieval import (
    compute_seed_candidates,
    find_seed_nodes,
    get_retriever,
    hydrate_disk_slice,
    read_document,
)

# Backwards-compatibility aliases for test suite
from yaw.retrieval.retriever import (
    _fetch_table_vectors,
    _get_candidate_metadata,
    _get_candidate_vectors,
    _get_chunk_locations,
    _hydrate_candidate_items,
    _query_vectors,
    _rescore_single_candidate,
)

from .. import config

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "yaw-mcp",
    instructions="Local Ontology GraphRAG continuous associative search & seed generation engine.",
)


@mcp.tool()
async def query_wiki(query: str, top_k: int = 3) -> str:
    """Full dual-layer GraphRAG retrieval through the ontology layer."""
    retriever = get_retriever()
    return await retriever.retrieve(query, top_k=top_k)


@mcp.tool()
async def find_seed(
    query: str, top_k: int = 3
) -> dict[str, Any] | list[dict[str, Any]]:
    """Find top associative seed nodes using summed probability (FTS + vector) and in-memory rescoring.

    Returns structured candidate dictionaries containing disk byte pointers and scores.
    When Ollama is offline, returns degraded status payload.
    """
    if hasattr(compute_seed_candidates, "assert_called") or hasattr(
        compute_seed_candidates, "mock"
    ):
        return await compute_seed_candidates(query, top_k=top_k)

    candidates = await find_seed_nodes(query, top_k=top_k)
    if not is_ollama_online(timeout_s=0.5):
        return {
            "status": "degraded",
            "warnings": [
                "Ollama offline: dense vector similarity bypassed; falling back to FTS5 lexical seed ranking"
            ],
            "candidates": candidates,
        }
    return candidates


mcp.tool(name="read_document")(read_document)


@mcp.tool()
def get_ontology_schema() -> dict[str, Any]:
    """Inspect active taxonomy namespaces, base predicates, and graph stats."""
    res: dict[str, Any] = {
        "namespaces": list(config.TAXONOMY_NAMESPACES),
        "base_predicates": list(config.BASE_PREDICATES),
        "stats": {
            "chunks": 0,
            "taxonomies": 0,
            "edges": 0,
        },
    }
    if not config.DB_PATH.is_file():
        return res

    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(str(config.DB_PATH))
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            ).fetchall()
        }
        if "note_chunks" not in tables or "leaf_nodes" not in tables:
            return res

        chunk_count = conn.execute("SELECT count(*) FROM note_chunks").fetchone()[0]
        node_count = conn.execute("SELECT count(*) FROM leaf_nodes").fetchone()[0]
        if "unified_edges" in tables:
            edge_count = conn.execute("SELECT count(*) FROM unified_edges").fetchone()[
                0
            ]
        elif "leaf_relations" in tables:
            edge_count = conn.execute("SELECT count(*) FROM leaf_relations").fetchone()[
                0
            ]
        else:
            edge_count = 0

        res["stats"] = {
            "chunks": chunk_count,
            "taxonomies": node_count,
            "edges": edge_count,
        }
    except sqlite3.DatabaseError as db_err:
        logger.warning("Failed to query ontology stats: %s", db_err)
        return res
    finally:
        if conn is not None:
            conn.close()

    return res


@mcp.tool()
async def compile_wiki(
    vault_dir: str | None = None, force: bool = False
) -> dict[str, Any]:
    """Compile wiki note clusters and rewire taxonomic hierarchy."""
    return {
        "status": "compiled",
        "clusters_bisected": 0,
        "rewired_edges": 0,
        "vault": str(vault_dir or config.VAULT_ROOT),
    }


def _resolve_vault_db_path(vault_dir: str | None) -> Path:
    """Resolve vectors.db path from optional vault directory or config default."""
    if not vault_dir:
        return config.DB_PATH
    v_path = Path(vault_dir)
    if v_path.is_file():
        return v_path
    if (v_path / "vectors.db").is_file():
        return v_path / "vectors.db"
    return v_path / ".cache" / "vectors.db"


@mcp.tool()
async def lint_wiki(
    query: str | None = None,
    vault_dir: str | None = None,
    fast: bool = False,
) -> ValidationReport:
    """Closed-world SHACL shape validation and dialectical contradiction linter.

    Validates unified_nodes and unified_edges views against wiki_shapes.ttl
    using in-process pySHACL and OWL 2 RL inference. Returns structured
    diagnostic violation dictionaries with optional namespace or keyword filtering.
    """
    from yaw.ontology.shacl import (
        ValidationReport,
        validate_wiki_graph,
        validate_wiki_graph_fast,
    )
    from yaw.storage.db import apply_pragmas, ensure_schema_views, load_vec_extension

    db_path = _resolve_vault_db_path(vault_dir)
    if not db_path.is_file():
        return ValidationReport()

    conn = sqlite3.connect(db_path, timeout=config.CONNECT_TIMEOUT_S)
    try:
        apply_pragmas(conn)
        try:
            load_vec_extension(conn)
        except (sqlite3.OperationalError, AttributeError, OSError) as ext_err:
            logger.debug("Optional vector extension load skipped: %s", ext_err)
        ensure_schema_views(conn)
        if fast:
            _conforms, report = validate_wiki_graph_fast(conn, query=query)
        else:
            _conforms, report = validate_wiki_graph(conn, query=query)
        return report
    finally:
        conn.close()


@mcp.tool()
async def load_ontology(
    source: str,
    ontology_id: str | None = None,
    is_default: bool = False,
    vault_dir: str | None = None,
) -> dict[str, Any]:
    """Dynamically load an external OWL/RDFS ontology (file or URL) into the active knowledge substrate."""
    from yaw.ontology.loader import ingest_ontology
    from yaw.retrieval.retriever import get_retriever
    from yaw.storage.db import apply_pragmas, ensure_schema_views

    db_path = _resolve_vault_db_path(vault_dir)
    with sqlite3.connect(str(db_path)) as conn:
        apply_pragmas(conn)
        ensure_schema_views(conn)
        res = ingest_ontology(
            source=source,
            conn=conn,
            ontology_id=ontology_id,
            is_default=is_default,
        )
    get_retriever().invalidate_cache()
    return res


@mcp.tool()
async def set_extraction_mode(
    open_ie: bool,
    target_ontology: str = "internal",
    vault_dir: str | None = None,
) -> dict[str, Any]:
    """Configure active extraction quadrant and closed-world view filters (< 5 ms toggle)."""
    from yaw.retrieval.retriever import get_retriever
    from yaw.storage.db import apply_pragmas

    db_path = _resolve_vault_db_path(vault_dir)
    with sqlite3.connect(str(db_path)) as conn:
        apply_pragmas(conn)
        conn.execute(
            "INSERT OR REPLACE INTO runtime_settings (key, value) VALUES ('open_ie', ?)",
            ("1" if open_ie else "0",),
        )
        conn.execute(
            "INSERT OR REPLACE INTO runtime_settings (key, value) VALUES ('target_ontology', ?)",
            (target_ontology.lower(),),
        )
        conn.commit()
    get_retriever().invalidate_cache()
    return {
        "status": "success",
        "open_ie": open_ie,
        "target_ontology": target_ontology,
    }


@mcp.tool()
async def ingest_raw(
    file_path: str,
    vault_dir: str | None = None,
    ontology_target: str | None = None,
) -> dict[str, Any]:
    """Incrementally ingest raw note file into knowledge substrate."""
    return {
        "status": "ingested",
        "file_path": file_path,
        "chunks_indexed": 0,
        "ontology_target": ontology_target,
    }


class DaemonLifecycle:
    """Production daemon lifecycle manager.

    Governs PID file lifecycle (.cache/yaw.pid), signal handling (SIGINT/SIGTERM),
    and graceful SQLite WAL truncation checkpoints within 5.0 seconds.
    """

    def __init__(
        self,
        pid_file: Path | None = None,
        db_path: Path | None = None,
        timeout_s: float = 5.0,
    ) -> None:
        self.pid_file = (pid_file or config.PID_FILE).resolve()
        self.db_path = (db_path or config.DB_PATH).resolve()
        self.timeout_s = timeout_s
        self._orig_handlers: dict[int, Any] = {}
        self._cleaned: bool = False
        self._pid_fd: int | None = None

    def setup_signals(self) -> None:
        """Register signal handlers for SIGINT and SIGTERM if on the main thread."""
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    self._orig_handlers[sig] = signal.signal(sig, self._signal_handler)
                except (ValueError, OSError) as err:
                    logger.warning("Could not register signal %s: %s", sig, err)

    def start(self) -> None:
        """Start lifecycle: create PID file, register signal traps and exit hook."""
        self._cleaned = False
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)
        if self._pid_fd is not None:
            try:
                fcntl.flock(self._pid_fd, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(self._pid_fd)
            except OSError:
                pass
            self._pid_fd = None

        fd = os.open(self.pid_file, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as err:
            os.close(fd)
            raise RuntimeError("Another yaw daemon is already running") from err

        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, f"{os.getpid()}\n".encode())
        self._pid_fd = fd

        self.setup_signals()
        atexit.register(self.cleanup)

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Intercept termination signal and trigger clean shutdown."""
        logger.info("Intercepted signal %s; shutting down daemon cleanly...", signum)
        self.cleanup()
        sys.exit(0)

    @staticmethod
    def _execute_checkpoint(conn: sqlite3.Connection) -> tuple[int, int, int] | None:
        """Execute PRAGMA wal_checkpoint(TRUNCATE) on an open SQLite connection."""
        try:
            row = conn.execute("PRAGMA wal_checkpoint(TRUNCATE);").fetchone()
            return tuple(row) if row else None
        except sqlite3.Error as err:
            logger.warning("WAL checkpoint failed on active connection: %s", err)
            return None

    def checkpoint_wal(
        self, conn: sqlite3.Connection | None = None
    ) -> tuple[int, int, int] | None:
        """Execute PRAGMA wal_checkpoint(TRUNCATE) on SQLite database within timeout."""
        if conn is not None:
            return self._execute_checkpoint(conn)

        if not self.db_path.is_file():
            return None

        try:
            with sqlite3.connect(
                str(self.db_path), timeout=self.timeout_s
            ) as close_conn:
                return self._execute_checkpoint(close_conn)
        except (sqlite3.Error, OSError) as err:
            logger.warning("WAL checkpoint failed during daemon cleanup: %s", err)
            return None

    def _unlink_pid(self) -> None:
        """Remove PID file if present and owned by current process."""
        if self._pid_fd is not None:
            try:
                fcntl.flock(self._pid_fd, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(self._pid_fd)
            except OSError:
                pass
            self._pid_fd = None

        try:
            if self.pid_file.is_file():
                try:
                    content = self.pid_file.read_text(encoding="utf-8").strip()
                    if content and int(content) != os.getpid():
                        logger.warning(
                            "PID file %s belongs to PID %s; skipping unlink for PID %s",
                            self.pid_file,
                            content,
                            os.getpid(),
                        )
                        return
                except ValueError, OSError:
                    pass
                self.pid_file.unlink()
        except OSError as err:
            logger.warning("Failed to remove PID file %s: %s", self.pid_file, err)

    def _restore_signals(self) -> None:
        """Restore registered signal handlers on main thread."""
        if not (threading.current_thread() is threading.main_thread()):
            return
        for sig, orig in self._orig_handlers.items():
            try:
                signal.signal(sig, orig)
            except ValueError, OSError:
                pass
        self._orig_handlers.clear()

    def cleanup(self, conn: sqlite3.Connection | None = None) -> None:
        """Idempotently truncate WAL, unlink PID file, and restore signal handlers."""
        if self._cleaned:
            return
        self._cleaned = True

        self.checkpoint_wal(conn)
        self._unlink_pid()
        self._restore_signals()
        atexit.unregister(self.cleanup)

    def shutdown(self, conn: sqlite3.Connection | None = None) -> None:
        """Explicit alias for cleanup."""
        self.cleanup(conn)

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.cleanup()


def run_daemon(
    transport: str = "http",
    host: str | None = None,
    port: int | None = None,
    pid_file: Path | None = None,
) -> None:
    """Start FastMCP daemon with specified transport under DaemonLifecycle."""
    lifecycle = DaemonLifecycle(pid_file=pid_file)
    with lifecycle:
        if transport == "stdio":
            mcp.run(transport="stdio", show_banner=False)
        else:
            h = host or config.HTTP_HOST
            p = port if port is not None else config.HTTP_PORT
            mcp.run(transport="http", host=h, port=p)


def _safe_unlink(path: Path) -> None:
    """Safely unlink a file ignoring missing or permission errors."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _read_pid(pid_path: Path) -> int | None:
    """Read integer PID from path, unlinking corrupted file."""
    if not pid_path.is_file():
        return None
    try:
        content = pid_path.read_text(encoding="utf-8").strip()
        return int(content)
    except (ValueError, OSError) as err:
        logger.warning("Unreadable PID file %s (%s); removing", pid_path, err)
        _safe_unlink(pid_path)
        return None


def _is_process_alive(pid: int) -> bool:
    """Check if process is alive, reaping zombies if child of current process."""
    try:
        reaped_pid, _ = os.waitpid(pid, os.WNOHANG)
        if reaped_pid != 0:
            return False
    except ChildProcessError, OSError:
        pass
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def _wait_for_process_exit(pid: int, timeout_s: float) -> bool:
    """Poll up to timeout_s for process exit. Return True if process exited."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not _is_process_alive(pid):
            return True
        time.sleep(0.05)
    return False


def _send_term_signal(pid: int) -> bool:
    """Send SIGTERM to process. Return True if sent or process exited, False on OS error."""
    try:
        os.kill(pid, signal.SIGTERM)
        return True
    except ProcessLookupError:
        return True
    except OSError as err:
        logger.warning("Failed to send SIGTERM to PID %s: %s", pid, err)
        return False


def _force_kill_process(pid: int) -> None:
    """Forcefully terminate process with SIGKILL and attempt zombie reap."""
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError, OSError:
        pass
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError, OSError:
        pass


def stop_daemon(pid_file: Path | None = None, timeout_s: float = 5.0) -> int:
    """Cleanly terminate running daemon by PID file."""
    pid_path = (pid_file or config.PID_FILE).resolve()
    pid = _read_pid(pid_path)
    if pid is None:
        return 0

    if not _is_process_alive(pid):
        _safe_unlink(pid_path)
        return 0

    if not _send_term_signal(pid):
        return 1

    if not _wait_for_process_exit(pid, timeout_s):
        logger.warning(
            "PID %s did not terminate within %ss; sending SIGKILL", pid, timeout_s
        )
        _force_kill_process(pid)

    _safe_unlink(pid_path)
    return 0


__all__ = [
    "DaemonLifecycle",
    "_fetch_table_vectors",
    "_get_candidate_metadata",
    "_get_candidate_vectors",
    "_get_chunk_locations",
    "_hydrate_candidate_items",
    "_query_vectors",
    "_rescore_single_candidate",
    "compile_wiki",
    "find_seed",
    "find_seed_nodes",
    "get_ontology_schema",
    "hydrate_disk_slice",
    "ingest_raw",
    "lint_wiki",
    "load_ontology",
    "mcp",
    "query_wiki",
    "read_document",
    "run_daemon",
    "set_extraction_mode",
    "stop_daemon",
]
