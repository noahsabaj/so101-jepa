#!/bin/bash
# Kaggle 2x T4: hjepa/lewam_plus.py data-parallel on tiny data (hjepa/jobs/tiny_data.py), every change on (do the ranks
# stay in sync through the plus terms?), then every change off.
set -uo pipefail
V=/tmp/lewam-venv
command -v uv >/dev/null || pip install -q uv
[ -x $V/bin/python ] || sh hjepa/lewam_repro_env.sh $V cu126
nvidia-smi -L
export PYTHONPATH=third_party/lewam MPLBACKEND=Agg
$V/bin/python hjepa/jobs/tiny_data.py /tmp/tiny.h5
A="data.source=h5 data.num_workers=2 data.load_in_ram=false train.batch_size=8 train.steps_per_epoch=12 train.epochs=3
   train.warmup_epochs=1 train.precision=fp32"
ALL="plus.proprio_keys=[proprio] plus.roll_k=3 plus.shortcut=true plus.distill=true plus.distill_from=1 plus.distill_every=2
     plus.distill_k=4 plus.distill_steps=3 plus.hier=true plus.hier_far=8 plus.hier_sub=3 train.p_drop_goal=0.1"
$V/bin/torchrun --standalone --nproc_per_node 2 hjepa/lewam_plus.py plus ddp-all /tmp/tiny.h5 $A $ALL 2>&1 |
    grep -v "Warning\|warn\|^W1"
$V/bin/torchrun --standalone --nproc_per_node 2 hjepa/lewam_plus.py plus ddp-off /tmp/tiny.h5 $A 2>&1 |
    grep -v "Warning\|warn\|^W1"
echo "== done"
