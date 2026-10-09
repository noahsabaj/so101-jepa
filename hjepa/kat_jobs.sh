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
#   v47       LeWAM (PLAN.md A25) on sim-2: convert the data, train, probe, rungs 1 and 2 with its action head
#             alone and with its gradient planner (fleet project so101-jepa-lewam: LeWAM is a submodule,
#             which `fleet submit --git` does not push)
#   v47b      the same, warm-started from the authors' OGBench Cube checkpoint (downloaded here)
#   v49       v47 for the paper's 50 epochs
#   v48, v48b v47, v47b at 224 px (data/so101_train_224.h5: SO101_IMAGE=224 SUFFIX=_224 sh sim/make_dataset.sh)
#   v50, v50b v47, v47b with the 64 px views upsampled to 224 (the paper's 7 x 7 map)
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
v47|v47b|v48|v48b|v49|v50|v50b)
  M=sojepa-$1 O=outputs/$1 U=data/ckpts/lewam/upstream/lewam-cube
  R=""; case $1 in v48*) R=_224; export SO101_IMAGE=224 ;; v50*) export LEWAM_UPSAMPLE=224 ;; esac  # v48: 224 px data and sim
  mkdir -p data/lewam $O $U
  if [ "${1%b}" != $1 ] && [ ! -f $U/lewam_best.pt ]; then
    for f in lewam_best.pt lewam_config.json; do curl -sfL -o $U/$f https://huggingface.co/LeWAM/lewam-cube/resolve/main/$f; done
  fi
  { [ -f data/lewam/so101_train$R.h5 ] || uvr python hjepa/lewam_data.py data/so101_train$R.h5 data/lewam/so101_train$R.h5; } &&
    { [ -f data/decomp/so101_train$R/pixels_wrist.npy ] || PYTHONPATH=third_party/lewam uvr python \
      third_party/lewam/scripts/decompress_h5.py --dataset_path data/lewam/so101_train$R.h5 --decomp_dir data/decomp \
      --views pixels_scene,pixels_wrist > $O/decompress.log 2>&1; } &&
    uvr python hjepa/lewam_train.py $M data/lewam/so101_train$R.h5 > $O/train.log 2>&1 &&
    sh hjepa/lewam_eval.sh $1 ;;
step5) TRAIN_ARGS="$CKPT_ON" sh hjepa/run_step5.sh train && sh hjepa/run_step5.sh offline ;;
*) echo "usage: sh hjepa/kat_jobs.sh check|v42|selfplay|v43|v44|step5|v47|v47b|v48|v48b|v49|v50|v50b" >&2; exit 2 ;;
esac
