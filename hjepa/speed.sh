#!/bin/sh
# Training speed and GPU memory per model and batch size: 40 batches each on DATA_train/DATA_val.
# Usage: sh hjepa/speed.sh DATA "so101_lewm:32 so101_lewm:64 so101_hjepa_l2:32" [extra Hydra overrides]
# Prints batches/s and samples/s (from the progress bar), peak GPU memory (all processes) and GPU use.
. "$(dirname "$0")/../scripts/gpu.sh"  # GPU, uvr (uv run with this GPU's torch build), GL
DATA=$1 RUNS=$2
shift 2
OUT=outputs/speed
mkdir -p $OUT
for run in $RUNS; do
  cfg=${run%%:*} bs=${run##*:}
  while :; do gpu_sample; sleep 0.5; done > $OUT/gpu_${cfg}_$bs.log &
  smi=$!
  uvr python hjepa/train.py $cfg seed=0 output_model_name=speed_${cfg}_$bs data.dataset.name=${DATA}_train \
    data.dataset.val_name=${DATA}_val loader.batch_size=$bs trainer.max_epochs=1 +trainer.limit_train_batches=40 \
    +trainer.limit_val_batches=2 "$@" > $OUT/train_${cfg}_$bs.log 2>&1
  status=$?
  kill $smi
  rate=$(tr '\r' '\n' < $OUT/train_${cfg}_$bs.log | grep -o "[0-9.]*it/s" | tail -1)
  mem=$(cut -d, -f1 $OUT/gpu_${cfg}_$bs.log | sort -n | tail -1)
  use=$(cut -d, -f2 $OUT/gpu_${cfg}_$bs.log | awk '{s += $1; n++} END {if (n) printf "%.0f", s / n}')
  echo "$cfg batch $bs: exit $status, $rate (x $bs samples), peak GPU memory $mem MiB, mean GPU use $use%"
  rm -rf data/ckpts/so101/speed_${cfg}_$bs
done
