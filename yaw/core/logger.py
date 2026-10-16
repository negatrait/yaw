"""FIFO log rotation and noise-suppression configuration for YAW."""

import contextlib
import logging
import os
from collections import deque
from pathlib import Path


class FIFOLineRotatingHandler(logging.Handler):
    """A logging handler that maintains at most `max_lines` in a FIFO ring buffer on disk."""

    def __init__(
        self,
        filename: str | Path,
        max_lines: int = 500,
        encoding: str = "utf-8",
        flush_interval: int = 50,
    ) -> None:
        super().__init__()
        self.filename = Path(filename)
        self.max_lines = max(1, max_lines)
        self.encoding = encoding
        self.flush_interval = max(1, flush_interval)
        self._buffer: deque[str] = deque(maxlen=self.max_lines)
        self._unflushed: int = 0
        with contextlib.suppress(OSError):
            self.filename.parent.mkdir(parents=True, exist_ok=True)
        self._load_existing()

    def _load_existing(self) -> None:
        if self.filename.exists():
            with contextlib.suppress(OSError):
                lines = self.filename.read_text(encoding=self.encoding).splitlines()
                self._buffer.extend(lines[-self.max_lines :])

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            for line in msg.splitlines():
                self._buffer.append(line)
            self._unflushed += 1
            if self._unflushed >= self.flush_interval or record.levelno >= logging.ERROR:
                self.flush()
        except Exception:
            self.handleError(record)

    def flush(self) -> None:
        if self._unflushed == 0 and self.filename.exists():
            return
        if not self._buffer and not self.filename.exists():
            return
        try:
            tmp_path = self.filename.with_suffix(self.filename.suffix + ".tmp")
            content = "\n".join(self._buffer) + ("\n" if self._buffer else "")
            tmp_path.write_text(content, encoding=self.encoding)
            tmp_path.replace(self.filename)
            self._unflushed = 0
        except OSError:
            pass

    def close(self) -> None:
        if self._unflushed > 0:
            self.flush()
        super().close()


def configure_logging(
    log_file: str | Path | None = None,
    level: int = logging.INFO,
    max_lines: int = 500,
    quiet_third_party: bool = True,
    disable_tqdm: bool = True,
) -> list[logging.Handler]:
    """Configure stream and FIFO-rotating file handlers, silencing chatty third-party loggers."""
    if disable_tqdm:
        os.environ.setdefault("TQDM_DISABLE", "1")

    if quiet_third_party:
        for name in ("httpx", "httpcore", "openai", "urllib3"):
            logging.getLogger(name).setLevel(logging.WARNING)

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        handlers.append(FIFOLineRotatingHandler(log_file, max_lines=max_lines))

    # force=True reconfigures any previously installed handlers
    logging.basicConfig(level=level, handlers=handlers, force=True)
    return handlers
