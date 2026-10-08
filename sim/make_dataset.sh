#!/bin/sh
# Make the SO-101 sim dataset on one Linux machine: train seeds 1000000.., val seeds 2000000..
# Usage: sh sim/make_dataset.sh TRAIN_EPISODES VAL_EPISODES PARALLEL
# Output: data/so101_train$SUFFIX.h5 and data/so101_val$SUFFIX.h5 (H-JEPA layout), shards in
# data/sim/shards_${POLICY}_$SO101_IMAGE$SUFFIX/. A split with 0 episodes is skipped. SO101_IMAGE renders larger views (the
# same episodes): SO101_IMAGE=224 SUFFIX=_224 sh sim/make_dataset.sh 0 100 8 makes data/so101_val_224.h5.
# POLICY=play (no expert, sim/collect.py) with SUFFIX=_play makes the play dataset.
set -e
. "$(dirname "$0")/../scripts/gpu.sh"  # GPU, uvr (uv run with this GPU's torch build), GL
TRAIN=$1; VAL=$2; P=$3; PER=25; SUFFIX=${SUFFIX:-}; export POLICY=${POLICY:-expert}
export MUJOCO_GL=${MUJOCO_GL:-$GL} SO101_IMAGE=${SO101_IMAGE:-64}
SHARDS=data/sim/shards_${POLICY}_$SO101_IMAGE$SUFFIX  # policy and image size in the path: no run reuses another's shards
mkdir -p $SHARDS
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
# A shard counts as done only if its .done marker exists and it holds the episodes asked for
echo $jobs | tr ' ' '
' | xargs -P "$P" -I{} sh -c 'IFS=:; set -- $(echo {}); f=../'"$SHARDS"'/$1_$2.h5; { [ -f $f.done ] && [ "$(sh ../scripts/uvr python collect.py count $f 2>/dev/null)" = "$3" ]; } || { rm -f $f.done; sh ../scripts/uvr python collect.py shard $f $2 $3 $POLICY > $f.log 2>&1 && touch $f.done; }'
for split in train val; do
  if [ $split = train ]; then n=$TRAIN; else n=$VAL; fi
  [ "$n" -gt 0 ] || continue
  # Merge exactly the shards scheduled above (not every ${split}_*.h5 left by an earlier, different run)
  list="" want=""
  for j in $jobs; do
    case $j in $split:*) ;; *) continue ;; esac
    rest=${j#*:}
    list="$list ../$SHARDS/${split}_${rest%%:*}.h5" want="$want ${rest##*:}"
  done
  got=$(uvr python collect.py count $list)
  [ "$got" = "${want# }" ] || { echo "make_dataset: $split shard episode counts are '$got', not '${want# }'" >&2; exit 1; }
  uvr python collect.py merge ../data/so101_$split$SUFFIX.h5 $list
  total=$(uvr python collect.py count ../data/so101_$split$SUFFIX.h5)
  [ "$total" = "$n" ] || { echo "make_dataset: ../data/so101_$split$SUFFIX.h5 has $total episodes, not $n" >&2; exit 1; }
done
ls -la ../data/so101_*$SUFFIX.h5
