"""The table of hjepa/lewam_repro_node.sh's evals (outputs/repro/eval_RUN_TASK_MODE_SEED.log) beside the paper's
(third_party/lewam/docs/REPRODUCE.md): goal-reaching success (%), mean +- std over the seeds.

    python hjepa/lewam_repro_table.py [outputs/repro]
"""

import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PAPER = {  # REPRODUCE.md, "Results of the released checkpoints"
    "pusht": {"policy": "85.3 +- 5.0", "best_of_k": "95.3 +- 2.3", "grad": "98.7 +- 1.2"},
    "cube": {"policy": "99.3 +- 1.2", "best_of_k": "100.0 +- 0.0", "grad": "100.0 +- 0.0"},
    "scene": {"policy": "90.7 +- 5.8", "best_of_k": "96.0 +- 2.0", "grad": "97.3 +- 1.2"},
    "puzzle": {"policy": "76.7 +- 8.1", "best_of_k": "76.7 +- 10.1", "grad": "78.7 +- 9.5"},
}
MODES = ["policy", "best_of_k", "grad"]
NAME = re.compile(r"eval_(.+)_(pusht|cube|scene|puzzle)_(policy|best_of_k|grad)_(\d+)\.log")


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/repro")
    rates = defaultdict(dict)
    for log in sorted(out.glob("eval_*.log")):
        m = NAME.fullmatch(log.name)
        found = re.search(r"'success_rate': ([0-9.]+)", log.read_text(errors="replace")) if m else None
        if found:
            run, task, mode, seed = m.groups()
            rates[(run, task, mode)][int(seed)] = float(found.group(1))
    print(f"{'run':<16}{'task':<7}{'mode':<11}{'ours':<16}{'paper':<13}seeds")
    for run, task in sorted({(r, t) for r, t, _ in rates}):
        for mode in MODES:
            seeds = rates.get((run, task, mode))
            if not seeds:
                continue
            v = list(seeds.values())
            ours = f"{statistics.mean(v):.1f} +- {statistics.stdev(v):.1f}" if len(v) > 1 else f"{v[0]:.1f}"
            detail = ", ".join(f"{s}: {r:.0f}" for s, r in sorted(seeds.items()))
            print(f"{run:<16}{task:<7}{mode:<11}{ours:<16}{PAPER.get(task, {}).get(mode, '-'):<13}{detail}")


if __name__ == "__main__":
    main()
