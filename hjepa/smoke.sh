#!/bin/sh
# Check a Linux GPU machine (NVIDIA or AMD, GPU.md) before the Phase 0 runs: torch on the GPU, MuJoCo rendering (EGL, else OSMesa),
# a tiny dataset (20 + 2 episodes; made once), 2 epochs of each model on it (time and GPU memory),
# and one closed-loop reach trial per model and planner. Logs in outputs/smoke/.
# Each run has a fresh id (RUN): new model names and output files, so nothing resumes or is skipped;
# it fails unless both training epochs ran and the planner replanned.
# Usage: MODELS="so101_lewm so101_lpwm" sh hjepa/smoke.sh   (default: so101_lewm so101_hjepa_l2)
set -e
. "$(dirname "$0")/../scripts/gpu.sh"  # GPU, uvr (uv run with this GPU's torch build), GL
unset MUJOCO_GL
MODELS=${MODELS:-so101_lewm so101_hjepa_l2}
OUT=outputs/smoke
RUN=$(date +%Y%m%d-%H%M%S)-$$
mkdir -p $OUT data/smoke
if [ $GPU = cuda ]; then nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv
else rocm-smi --showproductname --showdriverversion --showmeminfo vram; fi
uv sync $UV_GROUPS -q
uvr python -c "import torch; print('torch', torch.__version__, 'gpu', torch.cuda.is_available(), torch.cuda.get_device_name(0), 'hip', torch.version.hip)"

for gl in egl osmesa; do
  if MUJOCO_GL=$gl uvr python -c "
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
    uvr python collect.py shard ../data/smoke/train_$i.h5 $((1000000 + i * 5)) 5 > ../$OUT/collect_$i.log 2>&1 &
  done
  uvr python collect.py shard ../data/smoke/val_0.h5 2000000 2 > ../$OUT/collect_val.log 2>&1
  wait
  uvr python collect.py merge ../data/smoke_train.h5 ../data/smoke/train_0.h5 ../data/smoke/train_1.h5 ../data/smoke/train_2.h5 ../data/smoke/train_3.h5 > /dev/null
  uvr python collect.py merge ../data/smoke_val.h5 ../data/smoke/val_0.h5 > /dev/null
  cd ..
  echo "tiny dataset: $(( $(date +%s) - t0 )) s for 6,600 steps (5 processes)"
fi

for cfg in $MODELS; do
  while :; do gpu_sample; sleep 1; done > $OUT/gpu_$cfg.log &
  smi=$!
  t0=$(date +%s)
  uvr python hjepa/train.py $cfg seed=42 output_model_name=smoke_${cfg}_$RUN data.dataset.name=smoke_train \
    data.dataset.val_name=smoke_val trainer.max_epochs=2 > $OUT/train_$cfg.log 2>&1 || { kill $smi; tail -30 $OUT/train_$cfg.log; exit 1; }
  kill $smi
  if grep -q "Resuming full training state" $OUT/train_$cfg.log || ! grep -q "\[Epoch 1/2\] done" $OUT/train_$cfg.log; then
    echo "$cfg: training did not run both epochs from scratch"; tail -30 $OUT/train_$cfg.log; exit 1
  fi
  echo "$cfg: $(( $(date +%s) - t0 )) s for 2 epochs; GPU memory used (all processes) peak $(cut -d, -f1 $OUT/gpu_$cfg.log | sort -n | tail -1) MiB"
  tr '\r' '\n' < $OUT/train_$cfg.log | grep -o "[0-9.]*it/s" | tail -1 || true
done

for cfg in $MODELS; do
  case $cfg in
    *hjepa_l2) evals=so101_l2 ;;
    *) evals="so101_flat so101_flat_cem" ;;
  esac
  for ev in $evals; do
    t0=$(date +%s)
    res=$OUT/reach_${cfg}_${ev}_$RUN.jsonl
    uvr python sim/closed_loop.py reach data/ckpts/so101/smoke_${cfg}_$RUN/seed42/smoke_${cfg}_${RUN}_object.ckpt \
      $ev 1000 1 $res > $OUT/reach_${cfg}_$ev.log 2>&1 || { tail -30 $OUT/reach_${cfg}_$ev.log; exit 1; }
    grep -q '"replans": [1-9]' $res || { echo "$cfg with $ev: no replan ran"; tail -30 $OUT/reach_${cfg}_$ev.log; exit 1; }
    echo "closed loop, $cfg with $ev, 1 reach trial (100 steps): $(( $(date +%s) - t0 )) s: $(cat $res)"
  done
done
