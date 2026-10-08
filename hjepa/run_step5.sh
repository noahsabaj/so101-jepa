#!/bin/sh
# Phase 0, step 5, on one Linux GPU machine: train LeWM and H-JEPA on the community SO-100/SO-101
# data (data/community_{train,test}.h5), then the offline tests on the held-out setups (test split).
# Usage: sh hjepa/run_step5.sh train|offline
# Options (environment): MODELS (default "sojepa-v45 sojepa-v46": flat and 2-level, vision only);
# TRAIN_ARGS adds Hydra overrides. The offline planner config follows the model's num_levels.
# A failed process does not stop the others; the exit status is 1 if any failed.
. "$(dirname "$0")/../scripts/gpu.sh"  # GPU, uvr (uv run with this GPU's torch build), GL
MODELS=${MODELS:-sojepa-v45 sojepa-v46}
mkdir -p outputs
fail=0
for m in $MODELS; do
  ckpt=data/ckpts/community/$m/seed42/${m}_object.ckpt
  case $1 in
  train)  # 3 tries: after a crash (e.g. a GPU reset), it resumes from lightning_resume/last.ckpt
    ok=0
    for try in 1 2 3; do
      uvr python hjepa/train.py "$m" seed=42 $TRAIN_ARGS >> "outputs/train_community_$m.log" 2>&1 \
        && { ok=1; break; }
      echo "run_step5: training $m failed (try $try of 3)" >> "outputs/train_community_$m.log"
      sleep 60
    done
    [ $ok = 1 ] || fail=1 ;;
  offline)
    if grep -q "^num_levels: 2" "$(dirname "$ckpt")/config.yaml"; then cfg=so101_l2; else cfg=so101_flat; fi
    uvr python hjepa/offline.py "$ckpt" $cfg data/community_test.h5 "outputs/offline_community_$m.json" \
      > "outputs/offline_community_$m.log" 2>&1 || fail=1 ;;
  *)
    echo "usage: sh hjepa/run_step5.sh train|offline" >&2
    exit 2 ;;
  esac
done
exit $fail
