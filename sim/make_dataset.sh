#!/bin/sh
# Make the SO-101 sim dataset on one Linux machine: train seeds 1000000.., val seeds 2000000..
# Usage: sh sim/make_dataset.sh TRAIN_EPISODES VAL_EPISODES PARALLEL
# Output: data/so101_train$SUFFIX.h5 and data/so101_val$SUFFIX.h5 (H-JEPA layout), shards in
# data/sim/shards$SUFFIX/. A split with 0 episodes is skipped. SO101_IMAGE renders larger views (the
# same episodes): SO101_IMAGE=224 SUFFIX=_224 sh sim/make_dataset.sh 0 100 8 makes data/so101_val_224.h5.
set -e
. "$(dirname "$0")/../scripts/gpu.sh"  # GPU, uvr (uv run with this GPU's torch build), GL
TRAIN=$1; VAL=$2; P=$3; PER=25; SUFFIX=${SUFFIX:-}
export MUJOCO_GL=${MUJOCO_GL:-$GL} SO101_IMAGE=${SO101_IMAGE:-64}
mkdir -p data/sim/shards$SUFFIX
cd sim
uvr python -c "import mujoco" >/dev/null
jobs=""
for split in train val; do
  if [ $split = train ]; then n=$TRAIN; base=1000000; else n=$VAL; base=2000000; fi
  i=0
  while [ $i -lt $n ]; do
    jobs="$jobs $split:$((base + i)):$(( n - i < PER ? n - i : PER ))"
    i=$((i + PER))
  done
done
echo $jobs | tr ' ' '\n' | xargs -P "$P" -I{} sh -c 'IFS=:; set -- $(echo {}); f=../data/sim/shards'"$SUFFIX"'/$1_$2.h5; [ -f $f.done ] || { uv run $UV_GROUPS python collect.py shard $f $2 $3 > $f.log 2>&1 && touch $f.done; }'
for split in train val; do
  if [ $split = train ]; then n=$TRAIN; else n=$VAL; fi
  [ "$n" -gt 0 ] || continue
  uvr python collect.py merge ../data/so101_$split$SUFFIX.h5 $(ls ../data/sim/shards$SUFFIX/${split}_*.h5 | sort)  # seeds have the same width
done
ls -la ../data/so101_*$SUFFIX.h5
