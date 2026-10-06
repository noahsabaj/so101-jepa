#!/bin/sh
# Phase 0, step 5, on one Linux GPU machine: train LeWM and H-JEPA on the community SO-100/SO-101
# data (data/community_{train,test}.h5), then the offline tests on the held-out setups (test split).
# Usage: sh hjepa/run_step5.sh train|offline
# Options (environment): MODELS (default "lewm hjepa_l2"); TRAIN_ARGS adds Hydra overrides.
# A failed process does not stop the others; the exit status is 1 if any failed.
MODELS=${MODELS:-lewm hjepa_l2}
mkdir -p outputs
fail=0
for m in $MODELS; do
  ckpt=data/ckpts/community/community_$m/seed42/community_${m}_object.ckpt
  case $1 in
  train)
    uv run python hjepa/train.py "community_$m" seed=42 $TRAIN_ARGS > "outputs/train_community_$m.log" 2>&1 || fail=1 ;;
  offline)
    case $m in hjepa_l2) cfg=so101_l2 ;; *) cfg=so101_flat ;; esac
    uv run python hjepa/offline.py "$ckpt" $cfg data/community_test.h5 "outputs/offline_community_$m.json" \
      > "outputs/offline_community_$m.log" 2>&1 || fail=1 ;;
  *)
    echo "usage: sh hjepa/run_step5.sh train|offline" >&2
    exit 2 ;;
  esac
done
exit $fail
