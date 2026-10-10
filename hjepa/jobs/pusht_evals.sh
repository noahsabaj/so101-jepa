#!/bin/sh
# The Push-T evals of the released checkpoint that noah-pc did not finish (docs/REPRODUCE.md), on Colab's T4:
# best-of-32 seed 1, gradient planner seeds 42, 0, 1. Logs in outputs/repro/.
set -eu
V=/tmp/lewam-venv
export STABLEWM_HOME=/tmp/swm LEWAM_VENV=$V MPLBACKEND=Agg  # Colab sets a notebook-only backend
command -v uv >/dev/null || pip install -q uv
[ -x $V/bin/python ] || sh hjepa/lewam_repro_env.sh $V cu126
nvidia-smi -L; df -h /tmp | tail -1
mkdir -p $STABLEWM_HOME/datasets $STABLEWM_HOME/checkpoints
if [ ! -f $STABLEWM_HOME/datasets/pusht.h5 ]; then
    curl -sfL https://huggingface.co/datasets/LeWAM/lewam-pusht/resolve/main/pusht.h5.zst |
        $V/bin/python -c "import sys, zstandard; zstandard.ZstdDecompressor(max_window_size=2**31).copy_stream(sys.stdin.buffer, open('$STABLEWM_HOME/datasets/pusht.h5', 'wb'))"
fi
ls -la $STABLEWM_HOME/datasets
HF_HUB_OFFLINE=0 $V/bin/hf download LeWAM/lewam-pusht --local-dir $STABLEWM_HOME/checkpoints/lewam-pusht >/dev/null
for ms in "best_of_k 1" "grad 42" "grad 0" "grad 1"; do
    set -- $ms
    echo "== $1 seed $2 $(date +%T)"
    sh hjepa/lewam_repro_node.sh one lewam-pusht pusht 0 "$1" "$2"
    grep -o "'success_rate': [0-9.]*" "outputs/repro/eval_lewam-pusht_pusht_$1_$2.log" || tail -20 "outputs/repro/eval_lewam-pusht_pusht_$1_$2.log"
done
