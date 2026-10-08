"""The repo's scripts import each other by module name (sys.path holds sim/ and hjepa/ when they run)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for d in ("sim", "hjepa"):
    if str(ROOT / d) not in sys.path:
        sys.path.insert(0, str(ROOT / d))
