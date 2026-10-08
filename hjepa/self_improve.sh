#!/bin/sh
# One self-improvement round (PLAN.md A19): model k plays toward goals sampled from the data it was
# trained on (sim/self_play.py, no expert, no task), and the new episodes join that data.
# Usage: sh hjepa/self_improve.sh CKPT DATA OUT_DATA EPISODES PROCESSES FIRST_SEED
#   DATA, OUT_DATA: dataset names in data/ (e.g. so101_train_play -> so101_train_play_r1)
# Then train model k+1 on OUT_DATA. Seeds: FIRST_SEED.. (use a new range each round, below the test seeds
# 5000.. and the train seeds 1000000..: e.g. 3000000 + 100000 * round).
# EVAL: the planner's eval config (default so101_flat_cem). GPUS (e.g. "0 1 2 3") spreads the processes over GPUs, round robin (default: CUDA_VISIBLE_DEVICES, else 0);
# about 10 processes per GPU, and never a GPU that trains. Each episode is its own file
# (data/self/OUT_DATA/SEED.h5), so a stop (kill -TERM) loses only the episodes in progress, and a rerun
# continues. With STOP_AT (e.g. "2026-10-08 20:35"), the processes end at that time and the merge follows.
. "$(dirname "$0")/../scripts/gpu.sh"
export MUJOCO_GL=${MUJOCO_GL:-$GL}
CKPT=$1 DATA=$2 OUT=$3 N=$4 P=$5 FIRST=$6
GPUS=${GPUS:-${CUDA_VISIBLE_DEVICES:-0}} EVAL=${EVAL:-so101_flat_cem}
set -- $GPUS
PER=$(( (N + P - 1) / P ))
mkdir -p data/self/$OUT
i=0 pids=""
while [ $((i * PER)) -lt "$N" ]; do
  n=$(( N - i * PER < PER ? N - i * PER : PER ))
  g=$(eval echo "\${$(( i % $# + 1 ))}")
  CUDA_VISIBLE_DEVICES=$g uvr python sim/self_play.py "$CKPT" $EVAL data/$DATA.h5 data/self/$OUT \
    $((FIRST + i * PER)) $n > data/self/$OUT/$i.log 2>&1 &
  pids="$pids $!" i=$((i + 1))
done
watch=""
if [ -n "$STOP_AT" ]; then  # SIGTERM to the python processes themselves (uv does not pass it on from sh)
  ( while [ "$(date +%s)" -lt "$(date -d "$STOP_AT" +%s)" ]; do sleep 30; done
    pkill -TERM -f "python sim/self_play.py .* data/self/$OUT " ) &
  watch=$!
fi
for p in $pids; do wait $p || echo "self_improve: a self-play process ended with status $? (its finished episodes are kept)" >&2; done
[ -z "$watch" ] || kill $watch 2> /dev/null
episodes=$(ls data/self/$OUT/*.h5 | sort -V | sed 's#^#../#')
echo "self_improve: $(echo $episodes | wc -w) episodes" >&2
cd sim && uvr python collect.py merge ../data/$OUT.h5 ../data/$DATA.h5 $episodes
