#!/bin/sh
# Phase 0, steps 4 and 6, on one Linux GPU machine, after data/so101_{train,val}.h5 exist.
# Trains flat LeWM, flat LpWM and 2-level H-JEPA, then runs the offline tests and the closed-loop
# rung 1 and 2 test trials (seeds 5000-5099 and 5000-5049) for each model and planner, 10
# processes at a time. probe: linear probes of each model's latent for the cube and the grasp point
# (hjepa/probe.py).
# Usage: sh hjepa/run_phase0.sh train|offline|closed_loop|probe
# Options (environment): MODELS (train; default: all three); RUNS (offline, closed_loop: model:eval
# config pairs; default: each flat model with gradient descent and with CEM, H-JEPA with its
# solver); SEQUENTIAL=1 trains one model after the other (a GPU with 8 GB); GPU_<model> chooses a
# model's GPU (default 0); TRAIN_ARGS adds Hydra overrides (on a T4: "trainer.precision=16-mixed
# num_workers=2"); MUJOCO_GL (default egl).
# A failed process does not stop the others; the exit status is 1 if any failed.
export MUJOCO_GL=${MUJOCO_GL:-egl}
MODELS=${MODELS:-lewm lpwm hjepa_l2}
RUNS=${RUNS:-lewm:so101_flat lewm:so101_flat_cem lpwm:so101_flat lpwm:so101_flat_cem hjepa_l2:so101_l2}
mkdir -p outputs
fail=0 pids=""
waitall() {  # wait for the processes in $pids
  for p in $pids; do wait "$p" || fail=1; done
  pids=""
}
ckpt() {  # model -> its checkpoint
  echo "data/ckpts/so101/so101_$1/seed42/so101_$1_object.ckpt"
}
train() {  # model; 3 tries: after a crash (e.g. a GPU reset), it resumes from lightning_resume/last.ckpt
  gpu=$(eval echo "\${GPU_$1:-0}")
  for try in 1 2 3; do
    CUDA_VISIBLE_DEVICES=$gpu uv run python hjepa/train.py "so101_$1" seed=42 $TRAIN_ARGS >> "outputs/train_$1.log" 2>&1 \
      && return 0
    echo "run_phase0: training so101_$1 failed (try $try of 3)" >> "outputs/train_$1.log"
    sleep 60
  done
  return 1
}
case $1 in
train)
  for m in $MODELS; do
    if [ "$SEQUENTIAL" = 1 ]; then
      train "$m" || fail=1
    else
      train "$m" & pids="$pids $!"
    fi
  done
  waitall ;;
offline)
  for run in $RUNS; do
    name=${run%%:*} cfg=${run##*:}
    uv run python hjepa/offline.py "$(ckpt "$name")" "$cfg" data/so101_val.h5 "outputs/offline_${name}_$cfg.json" \
      > "outputs/offline_${name}_$cfg.log" 2>&1 & pids="$pids $!"
  done
  waitall ;;
closed_loop)
  for run in $RUNS; do
    name=${run%%:*} cfg=${run##*:}
    for c in 0 1 2 3 4 5 6 7 8 9; do
      uv run python sim/closed_loop.py reach "$(ckpt "$name")" "$cfg" $((5000 + c * 10)) 10 \
        "outputs/reach_${name}_$cfg.jsonl" > "outputs/reach_${name}_${cfg}_$c.log" 2>&1 & pids="$pids $!"
    done
    waitall
    for c in 0 1 2 3 4 5 6 7 8 9; do
      uv run python sim/closed_loop.py pick "$(ckpt "$name")" "$cfg" $((5000 + c * 5)) 5 \
        "outputs/pick_${name}_$cfg.jsonl" > "outputs/pick_${name}_${cfg}_$c.log" 2>&1 & pids="$pids $!"
    done
    waitall
  done ;;
probe)
  for m in $MODELS; do
    case $m in hjepa_l2) cfg=so101_l2 ;; *) cfg=so101_flat ;; esac
    uv run python hjepa/probe.py "$(ckpt "$m")" $cfg data/so101_val.h5 "outputs/probe_$m.json" \
      > "outputs/probe_$m.log" 2>&1 || fail=1
  done ;;
*)
  echo "usage: sh hjepa/run_phase0.sh train|offline|closed_loop|probe" >&2
  exit 2 ;;
esac
exit $fail
