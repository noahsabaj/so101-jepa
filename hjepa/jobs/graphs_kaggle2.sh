#!/bin/bash
# Kaggle 2x T4. Job 1010-131012-8c: the graphs gap shows at batch 16 (train state 0.59 against 0.014, any compile
# cache) and not at batch 8. Which compiled part, at batch 16? Two lanes, one GPU each, own compile caches.
set -uo pipefail
V=/tmp/lewam-venv
command -v uv >/dev/null || pip install -q uv
[ -x $V/bin/python ] || sh hjepa/lewam_repro_env.sh $V cu126
nvidia-smi -L
$V/bin/python hjepa/jobs/tiny_data.py /tmp/tiny.h5
export MPLBACKEND=Agg STABLEWM_HOME=/tmp/swm PYTHONPATH=third_party/lewam LEWAM_SPLIT=frame LEWAM_TIMER=0
args=(/tmp/tiny.h5 data.source=h5 data.num_workers=2 train.precision=fp32 train.warmup_epochs=1 train.epochs=2
      train.batch_size=16)
run() {  # GPU NAME, then VAR=VALUE ...
    gpu=$1 name=$2; shift 2
    mkdir -p /tmp/lane$gpu && cp -r hjepa third_party /tmp/lane$gpu/ 2>/dev/null
    (cd /tmp/lane$gpu && rm -rf data/ckpts/lewam/repro-cube &&
     env CUDA_VISIBLE_DEVICES=$gpu TORCHINDUCTOR_CACHE_DIR=/tmp/ic_$name "$@" \
         $V/bin/python hjepa/lewam_train.py repro-cube "${args[@]}" 2>&1 |
         grep "\] ep \|graphs on\|starts=\|Error\|error:" | sed "s/^/[$name] /")
}
lane0() {
    run 0 enc LEWAM_GRAPHS=enc
    run 0 sig LEWAM_GRAPHS=sig
    run 0 loss LEWAM_GRAPHS=loss
}
lane1() {
    run 1 nofused LEWAM_GRAPHS=1 LEWAM_FUSED=0
    run 1 fbrand LEWAM_GRAPHS=1 LEWAM_FALLBACK_RANDOM=1
    O=/tmp/gc16
    for v in eager genc+gsig+gloss wenc+csig+gloss; do
        CUDA_VISIBLE_DEVICES=1 GC_BS=16 TORCHINDUCTOR_CACHE_DIR=/tmp/ic_gc_$v \
            $V/bin/python hjepa/jobs/graphs_check.py train repro-cube /tmp/tiny.h5 $v 20 $O 2>&1 |
            grep -v "UserWarning\|warnings.warn\|^W1" | sed "s/^/[gc16 $v] /"
    done
    $V/bin/python hjepa/jobs/graphs_check.py compare $O 2>&1 | sed "s/^/[gc16] /"
}
lane0 > /tmp/lane0.log 2>&1 &
lane1 > /tmp/lane1.log 2>&1 &
wait
cat /tmp/lane0.log /tmp/lane1.log
echo "== done"
