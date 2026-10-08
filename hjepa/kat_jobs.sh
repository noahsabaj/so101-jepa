#!/bin/sh
# The kat-pc queue after the 2026-10-07 node run, one fleet job per step (chained with --after), each
# through hjepa/wsl_job.sh. 8 GB GPU: the encoder uses gradient checkpointing.
#   check     planner checks (hjepa/check_planner.sh)
#   v42       train SO-JEPA v42 (the new default recipe), probe, jump test
#   selfplay  self-improvement round 1c: v21 plays up to 96 episodes for 6 h (4 processes, fast planner), merged
#             with the play data into data/so101_train_play_r1.h5
#   v43       train SO-JEPA v43 on it, probe
#   v44       train SO-JEPA v44 (2-level H-JEPA, vision only), offline tests, probe, rungs 1 and 2
#   step5     train SO-JEPA v45 and v46 on the community data, offline tests on the held-out setups
. "$(dirname "$0")/../scripts/gpu.sh"
CKPT_ON="level1.encoder.pixel_encoder.encoder.gradient_checkpointing=true num_workers=8"
K=data/ckpts/so101
case $1 in
check) sh hjepa/check_planner.sh ;;
v42)
  MODELS=sojepa-v42 TRAIN_ARGS="$CKPT_ON" sh hjepa/run_phase0.sh train &&
    MODELS=sojepa-v42 sh hjepa/run_phase0.sh probe &&
    uvr python hjepa/jump_test.py $K/sojepa-v42/seed42/sojepa-v42_object.ckpt data/so101_val.h5 \
      outputs/jump_sojepa-v42.json 1,2,3,5,10 300 ;;
selfplay)
  STOP_AT="$(date -d "+6 hours" "+%F %T")" GPUS=0 EVAL=so101_flat_cem_fast sh hjepa/self_improve.sh $K/sojepa-v21/seed42/sojepa-v21_object.ckpt \
    so101_train_play so101_train_play_r1 96 4 3300000 ;;
v43) MODELS=sojepa-v43 TRAIN_ARGS="$CKPT_ON" sh hjepa/run_phase0.sh train && MODELS=sojepa-v43 sh hjepa/run_phase0.sh probe ;;
v44)
  MODELS=sojepa-v44 TRAIN_ARGS="$CKPT_ON" sh hjepa/run_phase0.sh train &&
    RUNS=sojepa-v44:so101_l2 sh hjepa/run_phase0.sh offline &&
    MODELS=sojepa-v44 sh hjepa/run_phase0.sh probe &&
    RUNS=sojepa-v44:so101_l2 sh hjepa/run_phase0.sh closed_loop ;;
step5) TRAIN_ARGS="$CKPT_ON" sh hjepa/run_step5.sh train && sh hjepa/run_step5.sh offline ;;
*) echo "usage: sh hjepa/kat_jobs.sh check|v42|selfplay|v43|v44|step5" >&2; exit 2 ;;
esac
