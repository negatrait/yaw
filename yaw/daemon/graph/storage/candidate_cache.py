"""Bounded, transactional candidate-result storage in the existing yaw.db."""

import sqlite3


class CandidateCacheStore:
    """Store opaque, versioned payloads; callers own the SQLite transaction.

    The shared payload budget is maintained by triggers, avoiding a full cache
    scan on each embedding write. FIFO eviction never touches graph tables.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        conn.execute("""
            CREATE TABLE IF NOT EXISTS candidate_cache_size (
                singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                total_bytes INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.execute("INSERT OR IGNORE INTO candidate_cache_size(singleton) VALUES (1)")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS candidate_stage_cache (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                stage TEXT NOT NULL,
                cache_key TEXT NOT NULL,
                payload BLOB NOT NULL,
                size_bytes INTEGER NOT NULL,
                UNIQUE(stage, cache_key)
            )
        """)
        for operation, adjustment in (
            ("INSERT", "+ NEW.size_bytes"),
            ("DELETE", "- OLD.size_bytes"),
            ("UPDATE", "+ NEW.size_bytes - OLD.size_bytes"),
        ):
            conn.execute(f"""
                CREATE TRIGGER IF NOT EXISTS candidate_cache_{operation.lower()}
                AFTER {operation} ON candidate_stage_cache BEGIN
                    UPDATE candidate_cache_size SET total_bytes = total_bytes {adjustment}
                    WHERE singleton = 1;
                END
            """)

    def get(self, stage: str, key: str) -> bytes | None:
        row = self.conn.execute(
            "SELECT payload FROM candidate_stage_cache WHERE stage = ? AND cache_key = ?",
            (stage, key),
        ).fetchone()
        return bytes(row[0]) if row is not None else None

    def contains(self, stage: str, key: str) -> bool:
        return (
            self.conn.execute(
                "SELECT 1 FROM candidate_stage_cache WHERE stage = ? AND cache_key = ?",
                (stage, key),
            ).fetchone()
            is not None
        )

    def delete(self, stage: str, key: str) -> None:
        self.conn.execute(
            "DELETE FROM candidate_stage_cache WHERE stage = ? AND cache_key = ?", (stage, key)
        )

    def put(self, stage: str, key: str, payload: bytes, *, max_bytes: int) -> None:
        if max_bytes < 0:
            raise ValueError("Candidate cache budget must be nonnegative.")
        self.delete(stage, key)
        if len(payload) > max_bytes or max_bytes == 0:
            return
        self.conn.execute(
            "INSERT INTO candidate_stage_cache(stage, cache_key, payload, size_bytes) "
            "VALUES (?, ?, ?, ?)",
            (stage, key, payload, len(payload)),
        )
        while (
            self.conn.execute(
                "SELECT total_bytes FROM candidate_cache_size WHERE singleton = 1"
            ).fetchone()[0]
            > max_bytes
        ):
            self.conn.execute(
                "DELETE FROM candidate_stage_cache WHERE seq = "
                "(SELECT MIN(seq) FROM candidate_stage_cache)"
            )
