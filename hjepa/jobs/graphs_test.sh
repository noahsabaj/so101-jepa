#!/bin/bash
# Test of CUDA graphs in training (lewam_train.py speedups, LEWAM_GRAPHS) on Kaggle's 2x T4: one GPU with and
# without graphs from the same seed (the losses must match), then 2 GPUs with graphs and a resume (val and the
# last, smaller batch take graphs of their own).
set -euo pipefail
V=/tmp/lewam-venv
command -v uv >/dev/null || pip install -q uv
[ -x $V/bin/python ] || sh hjepa/lewam_repro_env.sh $V cu126
nvidia-smi -L
$V/bin/python hjepa/jobs/tiny_data.py /tmp/tiny.h5
export MPLBACKEND=Agg STABLEWM_HOME=/tmp/swm PYTHONPATH=third_party/lewam LEWAM_SPLIT=frame LEWAM_TIMER=20
args=(/tmp/tiny.h5 data.source=h5 data.num_workers=2 train.precision=fp32 train.batch_size=16 train.warmup_epochs=1)
for g in 0 1; do
    echo "== 1 GPU, 2 epochs, LEWAM_GRAPHS=$g"
    rm -rf data/ckpts/lewam/repro-cube
    LEWAM_GRAPHS=$g $V/bin/python hjepa/lewam_train.py repro-cube "${args[@]}" train.epochs=2 2>&1 |
        grep -v "^W1\|UserWarning\|warnings.warn"
done
echo "== 2 GPUs, 2 epochs, graphs"
LEWAM_GRAPHS=1 $V/bin/torchrun --standalone --nproc_per_node 2 hjepa/lewam_train.py repro-pusht "${args[@]}" \
    train.epochs=2 2>&1 | grep -v "W1\|UserWarning\|warnings.warn"
echo "== 2 GPUs, resumed to 3 epochs, graphs"
LEWAM_GRAPHS=1 $V/bin/torchrun --standalone --nproc_per_node 2 hjepa/lewam_train.py repro-pusht "${args[@]}" \
    train.epochs=3 2>&1 | grep -v "W1\|UserWarning\|warnings.warn"
echo "== done"
