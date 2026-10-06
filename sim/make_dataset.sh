#!/bin/sh
# Make the SO-101 sim dataset on one Linux machine: train seeds 1000000.., val seeds 2000000..
# Usage: sh sim/make_dataset.sh TRAIN_EPISODES VAL_EPISODES PARALLEL
# Output: data/so101_train.h5 and data/so101_val.h5 (H-JEPA layout), shards in data/sim/shards/.
set -e
TRAIN=$1; VAL=$2; P=$3; PER=25
export MUJOCO_GL=egl
mkdir -p data/sim/shards
cd sim
uv run python -c "import mujoco" >/dev/null
jobs=""
for split in train val; do
  if [ $split = train ]; then n=$TRAIN; base=1000000; else n=$VAL; base=2000000; fi
  i=0
  while [ $i -lt $n ]; do
    jobs="$jobs $split:$((base + i)):$(( n - i < PER ? n - i : PER ))"
    i=$((i + PER))
  done
done
echo $jobs | tr ' ' '\n' | xargs -P "$P" -I{} sh -c 'IFS=:; set -- $(echo {}); f=../data/sim/shards/$1_$2.h5; [ -f $f.done ] || { uv run python collect.py shard $f $2 $3 > $f.log 2>&1 && touch $f.done; }'
uv run python collect.py merge ../data/so101_train.h5 $(ls ../data/sim/shards/train_*.h5 | sort -t_ -k2 -n)
uv run python collect.py merge ../data/so101_val.h5 $(ls ../data/sim/shards/val_*.h5 | sort -t_ -k2 -n)
ls -la ../data/so101_*.h5
