#!/bin/bash
# kat-pc WSL (or any CUDA Linux): LeWAM's venv once (~/lewam-venv), tiny data, then graphs_check.py rng, train per
# variant (each in its own process: graph pools freed) and compare.
set -uo pipefail
V=$HOME/lewam-venv
[ -x $V/bin/python ] || sed 's/\r$//' hjepa/lewam_repro_env.sh | sh -s -- $V cu128
export MPLBACKEND=Agg STABLEWM_HOME=/tmp/swm PYTHONPATH=third_party/lewam LEWAM_SPLIT=frame
[ -f /tmp/tiny.h5 ] || $V/bin/python hjepa/jobs/tiny_data.py /tmp/tiny.h5
O=outputs/graphs_check
$V/bin/python hjepa/jobs/graphs_check.py rng 2>&1 | grep -v "UserWarning\|warnings.warn"
for v in eager wenc+csig+closs genc+csig+closs wenc+gsig+closs wenc+csig+gloss genc+gsig+gloss; do
    $V/bin/python hjepa/jobs/graphs_check.py train repro-cube /tmp/tiny.h5 $v ${STEPS:-20} $O 2>&1 |
        grep -v "UserWarning\|warnings.warn\|^W1"
done
$V/bin/python hjepa/jobs/graphs_check.py compare $O
