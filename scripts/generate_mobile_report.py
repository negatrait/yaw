"""Backward compatibility stub for scripts.watchdog."""
from __future__ import annotations

import sys
from scripts.watchdog import main, run_once

__all__ = ["main", "run_once"]

if __name__ == "__main__":
    sys.exit(main())
