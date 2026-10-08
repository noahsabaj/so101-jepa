#!/bin/sh
# Planner checks after the 2026-10-07 fixes, on the tuning seeds (rule 6): reach with the 4-frame
# history, fp32 (so101_flat_cem) against fp16 + compile (so101_flat_cem_fast), 5 trials each, paired (a trial takes ~26 min in fp32 on kat-pc);
# and the time-step planner with equal-time strides (so101_flat_cem_dt, v31), 3 trials.
# Usage: sh hjepa/check_planner.sh  (results: outputs/check_*.jsonl)
. "$(dirname "$0")/../scripts/gpu.sh"
export MUJOCO_GL=${MUJOCO_GL:-$GL}
C="uvr python sim/closed_loop.py reach"
K=data/ckpts/so101
$C $K/sojepa-v6/seed42/sojepa-v6_object.ckpt so101_flat_cem 1000 5 outputs/check_reach_v6_fp32.jsonl
$C $K/sojepa-v6/seed42/sojepa-v6_object.ckpt so101_flat_cem_fast 1000 5 outputs/check_reach_v6_fast.jsonl
$C $K/sojepa-v31/seed42/sojepa-v31_object.ckpt so101_flat_cem_dt 1000 3 outputs/check_reach_v31_dt.jsonl
