#!/bin/sh
# One self-improvement round (PLAN.md A19): model k plays toward goals sampled from the data it was
# trained on (sim/self_play.py, no expert, no task), and the new episodes join that data.
# Usage: sh hjepa/self_improve.sh CKPT DATA OUT_DATA EPISODES PROCESSES FIRST_SEED
#   DATA, OUT_DATA: dataset names in data/ (e.g. so101_train_play -> so101_train_play_r1)
# Then train model k+1 on OUT_DATA. Seeds: FIRST_SEED.. (use a new range each round, below the test seeds
# 5000.. and the train seeds 1000000..: e.g. 3000000 + 100000 * round).
. "$(dirname "$0")/../scripts/gpu.sh"
export MUJOCO_GL=${MUJOCO_GL:-$GL}
CKPT=$1 DATA=$2 OUT=$3 N=$4 P=$5 FIRST=$6
PER=$(( (N + P - 1) / P ))
mkdir -p data/self/$OUT
i=0 pids=""
while [ $((i * PER)) -lt "$N" ]; do
  n=$(( N - i * PER < PER ? N - i * PER : PER ))
  f=data/self/$OUT/$i.h5
  [ -f $f.done ] || { uvr python sim/self_play.py "$CKPT" so101_flat_cem data/$DATA.h5 $f $((FIRST + i * PER)) $n > $f.log 2>&1 && touch $f.done; } &
  pids="$pids $!" i=$((i + 1))
done
for p in $pids; do wait $p; done
cd sim && uvr python collect.py merge ../data/$OUT.h5 ../data/$DATA.h5 $(ls ../data/self/$OUT/*.h5 | sort -V)
