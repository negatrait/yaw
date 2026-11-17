"""Package entrypoint for `python -m yaw`."""

import sys

from yaw.cli import main

if __name__ == "__main__":
    sys.exit(main())
