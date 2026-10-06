#!/bin/sh
# Phase 0, steps 4 and 6, on one Linux GPU machine (the rental), after data/so101_{train,val}.h5 exist.
# Trains flat LeWM and 2-level H-JEPA at the same time, then runs the offline tests and the
# closed-loop rung 1 and 2 test trials (seeds 5000-5099 and 5000-5049), 10 processes at a time.
# Usage: sh hjepa/run_phase0.sh train|offline|closed_loop
set -e
export MUJOCO_GL=egl
case $1 in
train)
  uv run python hjepa/train.py so101_lewm seed=42 > outputs/train_lewm.log 2>&1 &
  uv run python hjepa/train.py so101_hjepa_l2 seed=42 > outputs/train_l2.log 2>&1 &
  wait ;;
offline)
  uv run python hjepa/offline.py data/ckpts/so101/so101_lewm/seed42/so101_lewm_object.ckpt so101_flat data/so101_val.h5 outputs/offline_lewm.json &
  uv run python hjepa/offline.py data/ckpts/so101/so101_hjepa_l2/seed42/so101_hjepa_l2_object.ckpt so101_l2 data/so101_val.h5 outputs/offline_l2.json &
  wait ;;
closed_loop)
  for model in lewm:so101_flat hjepa_l2:so101_l2; do
    name=${model%%:*}; cfg=${model##*:}
    ckpt=data/ckpts/so101/so101_$name/seed42/so101_${name}_object.ckpt
    for c in 0 1 2 3 4 5 6 7 8 9; do
      uv run python sim/closed_loop.py reach $ckpt $cfg $((5000 + c * 10)) 10 outputs/reach_$name.jsonl > outputs/reach_${name}_$c.log 2>&1 &
    done
    wait
    for c in 0 1 2 3 4 5 6 7 8 9; do
      uv run python sim/closed_loop.py pick $ckpt $cfg $((5000 + c * 5)) 5 outputs/pick_$name.jsonl > outputs/pick_${name}_$c.log 2>&1 &
    done
    wait
  done ;;
esac
