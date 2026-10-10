#!/bin/sh
# LeWAM data scaling (SO-JEPA v53, PLAN.md A25): v53 on a fraction of the train episodes, all fractions at once on
# this GPU, then the offline action check of each on the val episodes. Results in outputs/v53-fNN/.
#   sh hjepa/lewam_scaling.sh [FRACTIONS]   (default "0.125 0.25 0.5 1")
. "$(dirname "$0")/../scripts/gpu.sh"
FRACTIONS=${1:-0.125 0.25 0.5 1}
name() { echo v53-f$(awk "BEGIN{print int($1 * 100)}"); }
pids=""
for f in $FRACTIONS; do
  R=$(name $f); mkdir -p outputs/$R
  LEWAM_DATA_FRACTION=$f uvr python hjepa/lewam_train.py sojepa-$R data/lewam/so101_train.h5 > outputs/$R/train.log 2>&1 &
  pids="$pids $!"
done
fail=0
for p in $pids; do wait $p || fail=1; done
pids=""
for f in $FRACTIONS; do
  R=$(name $f)
  uvr python hjepa/lewam_check.py data/ckpts/lewam/sojepa-$R/seed42/lewam_best.pt data/so101_val.h5 400 \
    > outputs/$R/check.log 2>&1 & pids="$pids $!"
done
for p in $pids; do wait $p || fail=1; done
exit $fail
