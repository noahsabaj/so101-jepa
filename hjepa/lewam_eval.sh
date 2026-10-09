#!/bin/sh
# Tests of a trained LeWAM run (PLAN.md A25): the encoder probe, then rungs 1 (reach, 100 trials) and 2 (pick,
# 50 trials) with its action head alone (so101_lewam_policy) and with its gradient planner (so101_lewam_grad),
# all at once, 10 processes each (seeds 5000..). Results in outputs/TAG/.
#   sh hjepa/lewam_eval.sh TAG        (e.g. v47; v48* use the 224 px val set and render at 224 px)
. "$(dirname "$0")/../scripts/gpu.sh"
M=sojepa-$1 O=outputs/$1 L=data/ckpts/lewam/sojepa-$1/seed42/lewam_best.pt
R=""; case $1 in v48*) R=_224; export SO101_IMAGE=224 ;; esac
mkdir -p $O
uvr python hjepa/probe.py $L so101_lewam_policy data/so101_val$R.h5 outputs/probe_$M.json > $O/probe.log 2>&1 &
pids="$!"
for cfg in so101_lewam_policy so101_lewam_grad; do
  for task_n in reach:10 pick:5; do
    task=${task_n%%:*} n=${task_n##*:}
    for c in 0 1 2 3 4 5 6 7 8 9; do
      uvr python sim/closed_loop.py $task $L $cfg $((5000 + c * n)) $n $O/${task}_$cfg.jsonl \
        > $O/${task}_${cfg}_$c.log 2>&1 & pids="$pids $!"
    done
  done
done
fail=0
for p in $pids; do wait $p || fail=1; done
exit $fail
