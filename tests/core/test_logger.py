import logging
import os
import tempfile
import unittest
from pathlib import Path

from yaw.core.logger import FIFOLineRotatingHandler, configure_logging


class LoggerTests(unittest.TestCase):
    def test_fifo_line_rotating_handler_caps_file_at_max_lines(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "test.log"
            handler = FIFOLineRotatingHandler(log_path, max_lines=5)
            logger = logging.getLogger("test_fifo")
            logger.setLevel(logging.INFO)
            logger.addHandler(handler)

            for i in range(12):
                logger.info("Message %d", i)

            handler.flush()
            lines = log_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 5)
            self.assertEqual(lines, [f"Message {i}" for i in range(7, 12)])
            logger.removeHandler(handler)
            handler.close()

    def test_fifo_handler_reloads_existing_file_and_retains_fifo(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "existing.log"
            log_path.write_text("Old 1\nOld 2\nOld 3\nOld 4\nOld 5\n", encoding="utf-8")

            handler = FIFOLineRotatingHandler(log_path, max_lines=4)
            logger = logging.getLogger("test_existing")
            logger.setLevel(logging.INFO)
            logger.addHandler(handler)

            logger.info("New 1")
            handler.flush()

            lines = log_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 4)
            self.assertEqual(lines, ["Old 3", "Old 4", "Old 5", "New 1"])
            logger.removeHandler(handler)
            handler.close()

    def test_configure_logging_quiets_noisy_loggers_and_sets_tqdm(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "run.log"
            handlers = configure_logging(log_file=log_path, max_lines=500)
            self.assertEqual(os.environ.get("TQDM_DISABLE"), "1")
            for name in ("httpx", "httpcore", "openai", "urllib3"):
                self.assertEqual(logging.getLogger(name).level, logging.WARNING)
            self.assertEqual(len(handlers), 2)
            self.assertIsInstance(handlers[1], FIFOLineRotatingHandler)

    def test_fifo_handler_buffers_writes_until_flush_interval_or_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "buffered.log"
            handler = FIFOLineRotatingHandler(log_path, max_lines=50, flush_interval=5)
            logger = logging.getLogger("test_buffered_fifo")
            logger.setLevel(logging.INFO)
            logger.addHandler(handler)

            # 3 INFO messages (< flush_interval=5) remain buffered in memory without disk write
            for i in range(3):
                logger.info("Buffered %d", i)
            self.assertFalse(log_path.exists())

            # 2 more INFO messages reach flush_interval=5 and trigger a single disk flush
            logger.info("Buffered 3")
            logger.info("Buffered 4")
            self.assertTrue(log_path.exists())
            self.assertEqual(len(log_path.read_text(encoding="utf-8").splitlines()), 5)

            # ERROR message flushes immediately even before flush_interval
            logger.error("Immediate error")
            self.assertEqual(len(log_path.read_text(encoding="utf-8").splitlines()), 6)

            logger.removeHandler(handler)
            handler.close()


if __name__ == "__main__":
    unittest.main()
