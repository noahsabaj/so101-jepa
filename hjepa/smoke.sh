#!/bin/sh
# Check a Linux GPU machine before the Phase 0 runs: CUDA, MuJoCo rendering (EGL, else OSMesa),
# a tiny dataset (20 + 2 episodes; made once), 2 epochs of each model on it (time and GPU memory),
# and one closed-loop reach trial per model and planner. Logs in outputs/smoke/.
# Usage: MODELS="so101_lewm so101_lpwm" sh hjepa/smoke.sh   (default: so101_lewm so101_hjepa_l2)
set -e
unset MUJOCO_GL
MODELS=${MODELS:-so101_lewm so101_hjepa_l2}
OUT=outputs/smoke
mkdir -p $OUT data/smoke
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv
uv sync -q
uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

for gl in egl osmesa; do
  if MUJOCO_GL=$gl uv run python -c "
import sys, time
sys.path.insert(0, 'sim')
from env import SO101Env
env = SO101Env()
env.reset(0)
t = time.perf_counter()
for _ in range(100):
    env.render()
print('MUJOCO_GL=$gl:', round(100 / (time.perf_counter() - t)), 'frames/s (scene and wrist)')
" 2> /dev/null; then export MUJOCO_GL=$gl; break; fi
done
[ -n "$MUJOCO_GL" ] || { echo "no MuJoCo renderer works"; exit 1; }

if [ ! -f data/smoke_train.h5 ] || [ ! -f data/smoke_val.h5 ]; then
  t0=$(date +%s)
  cd sim
  for i in 0 1 2 3; do
    uv run python collect.py shard ../data/smoke/train_$i.h5 $((1000000 + i * 5)) 5 > ../$OUT/collect_$i.log 2>&1 &
  done
  uv run python collect.py shard ../data/smoke/val_0.h5 2000000 2 > ../$OUT/collect_val.log 2>&1
  wait
  uv run python collect.py merge ../data/smoke_train.h5 ../data/smoke/train_0.h5 ../data/smoke/train_1.h5 ../data/smoke/train_2.h5 ../data/smoke/train_3.h5 > /dev/null
  uv run python collect.py merge ../data/smoke_val.h5 ../data/smoke/val_0.h5 > /dev/null
  cd ..
  echo "tiny dataset: $(( $(date +%s) - t0 )) s for 6,600 steps (5 processes)"
fi

for cfg in $MODELS; do
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 1000 > $OUT/gpu_$cfg.log &
  smi=$!
  t0=$(date +%s)
  uv run python hjepa/train.py $cfg seed=42 output_model_name=smoke_$cfg data.dataset.name=smoke_train \
    data.dataset.val_name=smoke_val trainer.max_epochs=2 > $OUT/train_$cfg.log 2>&1 || { kill $smi; tail -30 $OUT/train_$cfg.log; exit 1; }
  kill $smi
  echo "$cfg: $(( $(date +%s) - t0 )) s for 2 epochs; GPU memory used (all processes) peak $(sort -n $OUT/gpu_$cfg.log | tail -1) MiB"
  tr '\r' '\n' < $OUT/train_$cfg.log | grep -o "[0-9.]*it/s" | tail -1 || true
done

for cfg in $MODELS; do
  case $cfg in
    *hjepa_l2) evals=so101_l2 ;;
    *) evals="so101_flat so101_flat_cem" ;;
  esac
  for ev in $evals; do
    t0=$(date +%s)
    uv run python sim/closed_loop.py reach data/ckpts/so101/smoke_$cfg/seed42/smoke_${cfg}_object.ckpt \
      $ev 1000 1 $OUT/reach_${cfg}_$ev.jsonl > $OUT/reach_${cfg}_$ev.log 2>&1 || { tail -30 $OUT/reach_${cfg}_$ev.log; exit 1; }
    echo "closed loop, $cfg with $ev, 1 reach trial (100 steps): $(( $(date +%s) - t0 )) s: $(cat $OUT/reach_${cfg}_$ev.jsonl)"
  done
done
