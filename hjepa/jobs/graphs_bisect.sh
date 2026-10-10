#!/bin/bash
# Which part of LEWAM_GRAPHS changes training? The real trainer (lewam_train.py) on tiny data, 1 GPU, 2 epochs, as
# graphs_test.sh, once per setting: VAR=VALUE ... lines below. kat-pc WSL: venv ~/lewam-venv (graphs_check.sh).
set -uo pipefail
V=$HOME/lewam-venv
export MPLBACKEND=Agg STABLEWM_HOME=/tmp/swm PYTHONPATH=third_party/lewam LEWAM_SPLIT=frame LEWAM_TIMER=0
[ -f /tmp/tiny.h5 ] || $V/bin/python hjepa/jobs/tiny_data.py /tmp/tiny.h5
args=(/tmp/tiny.h5 data.source=h5 data.num_workers=2 train.precision=fp32 train.batch_size=${BS:-8} train.warmup_epochs=1 train.epochs=2)
while read -r setting; do
    [ -z "$setting" ] && continue
    echo "== $setting"
    rm -rf data/ckpts/lewam/repro-cube
    env $setting $V/bin/python hjepa/lewam_train.py repro-cube "${args[@]}" 2>&1 | grep "\] ep \|graphs on\|Error\|error:"
done <<'LIST'
LEWAM_GRAPHS=0
LEWAM_GRAPHS=1
LEWAM_GRAPHS=enc
LEWAM_GRAPHS=sig
LEWAM_GRAPHS=loss
LEWAM_GRAPHS=1 LEWAM_FUSED=0
LEWAM_GRAPHS=1 LEWAM_FALLBACK_RANDOM=1
LIST
echo "== done"
