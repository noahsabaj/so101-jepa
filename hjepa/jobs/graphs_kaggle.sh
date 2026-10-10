#!/bin/bash
# Kaggle T4, where job 1010-122303-3n showed the graphs gap (train state 0.59 against 0.014) and kat-pc (4060 Ti,
# batch 8) did not: is it the GPU, the batch size, or the compile cache left by the run before? 1 GPU, 2 epochs.
set -uo pipefail
V=/tmp/lewam-venv
command -v uv >/dev/null || pip install -q uv
[ -x $V/bin/python ] || sh hjepa/lewam_repro_env.sh $V cu126
nvidia-smi -L
$V/bin/python hjepa/jobs/tiny_data.py /tmp/tiny.h5
export MPLBACKEND=Agg STABLEWM_HOME=/tmp/swm PYTHONPATH=third_party/lewam LEWAM_SPLIT=frame LEWAM_TIMER=0 \
    CUDA_VISIBLE_DEVICES=0
args=(/tmp/tiny.h5 data.source=h5 data.num_workers=2 train.precision=fp32 train.warmup_epochs=1 train.epochs=2)
run() {  # BS, then VAR=VALUE ...
    bs=$1; shift
    echo "== batch $bs $*"
    rm -rf data/ckpts/lewam/repro-cube
    env "$@" $V/bin/python hjepa/lewam_train.py repro-cube "${args[@]}" train.batch_size=$bs 2>&1 |
        grep "\] ep \|graphs on\|Error\|error:"
}
run 16 LEWAM_GRAPHS=1 TORCHINDUCTOR_CACHE_DIR=/tmp/ic1   # graphs first, fresh cache
run 16 LEWAM_GRAPHS=0 TORCHINDUCTOR_CACHE_DIR=/tmp/ic2   # no graphs, fresh cache
run 16 LEWAM_GRAPHS=1 TORCHINDUCTOR_CACHE_DIR=/tmp/ic2   # graphs after a no-graphs run, its cache (the 3n order)
run 8 LEWAM_GRAPHS=1 TORCHINDUCTOR_CACHE_DIR=/tmp/ic3    # batch 8, as kat-pc
run 8 LEWAM_GRAPHS=0 TORCHINDUCTOR_CACHE_DIR=/tmp/ic4
echo "== done"
