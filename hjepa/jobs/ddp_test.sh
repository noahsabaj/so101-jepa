#!/bin/bash
# Test of the data-parallel trainer (hjepa/lewam_common.py Ddp) on 2 GPUs with NCCL (Kaggle's 2x T4), before a paid
# node: weights in sync every epoch, the loss falls, rank 0 alone saves, a resumed run goes on; one GPU for comparison.
set -eu
V=/tmp/lewam-venv
command -v uv >/dev/null || pip install -q uv
[ -x $V/bin/python ] || sh hjepa/lewam_repro_env.sh $V cu126
nvidia-smi -L
$V/bin/python hjepa/jobs/tiny_data.py /tmp/tiny.h5
export MPLBACKEND=Agg STABLEWM_HOME=/tmp/swm PYTHONPATH=third_party/lewam LEWAM_SPLIT=frame LEWAM_TIMER=20
args=(/tmp/tiny.h5 data.source=h5 data.num_workers=2 train.precision=fp32 train.batch_size=16 train.warmup_epochs=1)
echo "== 2 GPUs, 2 epochs"
$V/bin/torchrun --standalone --nproc_per_node 2 hjepa/lewam_train.py repro-pusht "${args[@]}" train.epochs=2
echo "== 2 GPUs, resumed to 3 epochs"
$V/bin/torchrun --standalone --nproc_per_node 2 hjepa/lewam_train.py repro-pusht "${args[@]}" train.epochs=3
ls -la data/ckpts/lewam/repro-pusht/seed42/
echo "== 1 GPU, 2 epochs"
$V/bin/python hjepa/lewam_train.py repro-cube "${args[@]}" train.epochs=2
echo "== done"
