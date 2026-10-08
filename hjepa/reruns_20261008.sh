#!/bin/sh
# Reruns after the Codex audit fixes (2026-10-08), on the fixed code, one after another on one GPU:
#   A20  v7 = v6 + its learned value, retrained with the terminal-target fix (finding 9); offline expert
#        rank with real history (finding 11) and rung 1 on the test seeds, latent distance against value
#   A21  jump test on one common case list for every model (finding 10): v6, v10 (fixed steps) against
#        v31, v36 (time-step jumps)
#   #11  offline tests of the time-step models with real history
# Usage: sh hjepa/reruns_20261008.sh [PY]   (PY: the python to run, default "sh scripts/uvr python")
# Results: outputs/rerun_1008/
PY=${1:-sh scripts/uvr python}
K=data/ckpts/so101
O=outputs/rerun_1008
V6=$K/sojepa-v6/seed42/sojepa-v6_object.ckpt
mkdir -p $O
fail=0
run() {  # name command...
  name=$1; shift
  echo "$(date '+%H:%M') $name" >&2
  "$@" > $O/$name.log 2>&1 || { fail=1; echo "$(date '+%H:%M') $name failed ($O/$name.log)" >&2; }
}
run value_v6 $PY hjepa/value.py $V6 so101_flat data/so101_train.h5 data/so101_val.h5
cp "$(dirname $V6)/value.json" $O/value_v6.json 2> /dev/null
run offline_v6_latent $PY hjepa/offline.py $V6 so101_flat data/so101_val.h5 $O/offline_v6_latent.json
run offline_v6_value $PY hjepa/offline.py $V6 so101_flat_value data/so101_val.h5 $O/offline_v6_value.json
run reach_v6_latent $PY sim/closed_loop.py reach $V6 so101_flat_cem 5000 100 $O/reach_v6_latent.jsonl
run reach_v6_value $PY sim/closed_loop.py reach $V6 so101_flat_cem_value 5000 100 $O/reach_v6_value.jsonl
run jump_v6 $PY hjepa/jump_test.py $V6 data/so101_val.h5 $O/jump_v6.json 1 300
run jump_v10 $PY hjepa/jump_test.py $K/sojepa-v10/seed43/sojepa-v10_object.ckpt data/so101_val.h5 $O/jump_v10.json 1 300
run jump_v31 $PY hjepa/jump_test.py $K/sojepa-v31/seed42/sojepa-v31_object.ckpt data/so101_val.h5 $O/jump_v31.json 1,2,3,5,10 300
run jump_v36 $PY hjepa/jump_test.py $K/sojepa-v36/seed43/sojepa-v36_object.ckpt data/so101_val.h5 $O/jump_v36.json 1,2,3,5,10 300
run offline_v31 $PY hjepa/offline.py $K/sojepa-v31/seed42/sojepa-v31_object.ckpt so101_flat data/so101_val.h5 $O/offline_v31.json
run offline_v36 $PY hjepa/offline.py $K/sojepa-v36/seed43/sojepa-v36_object.ckpt so101_flat data/so101_val.h5 $O/offline_v36.json
echo "$(date '+%H:%M') done (fail=$fail)" >&2
exit $fail
