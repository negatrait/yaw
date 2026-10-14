"""FIFO log rotation and noise-suppression configuration for YAW."""

from collections import deque
import logging
import os
from pathlib import Path


class FIFOLineRotatingHandler(logging.Handler):
    """A logging handler that maintains at most `max_lines` in a FIFO ring buffer on disk."""

    def __init__(self, filename: str | Path, max_lines: int = 500, encoding: str = "utf-8"):
        super().__init__()
        self.filename = Path(filename)
        self.max_lines = max(1, max_lines)
        self.encoding = encoding
        self._buffer: deque[str] = deque(maxlen=self.max_lines)
        self._load_existing()

    def _load_existing(self) -> None:
        if self.filename.exists():
            try:
                lines = self.filename.read_text(encoding=self.encoding).splitlines()
                self._buffer.extend(lines[-self.max_lines :])
            except Exception:
                pass

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            for line in msg.splitlines():
                self._buffer.append(line)
            self.flush()
        except Exception:
            self.handleError(record)

    def flush(self) -> None:
        try:
            self.filename.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.filename.with_suffix(self.filename.suffix + ".tmp")
            content = "\n".join(self._buffer) + ("\n" if self._buffer else "")
            tmp_path.write_text(content, encoding=self.encoding)
            tmp_path.replace(self.filename)
        except Exception:
            pass

    def close(self) -> None:
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
