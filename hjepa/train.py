"""Train an SO-101 world model with the H-JEPA code (third_party/H-JEPA, run from source).

    sh scripts/uvr python hjepa/train.py CONFIG [hydra overrides ...]   # e.g. so101_hjepa_l2 seed=42

Datasets and checkpoints live in data/ (HJEPA_HOME): data/<name>.h5, data/ckpts/<env>/<model>/.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    env = dict(os.environ, HJEPA_HOME=str(ROOT / "data"))
    cmd = [sys.executable, "main_hjepa.py", "--config-dir", str(ROOT / "hjepa/config/train"),
           "--config-name", sys.argv[1], *sys.argv[2:]]
    sys.exit(subprocess.call(cmd, cwd=ROOT / "third_party/H-JEPA/h_jepa", env=env))
