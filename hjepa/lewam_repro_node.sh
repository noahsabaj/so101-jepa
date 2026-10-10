#!/bin/sh
# LeWAM reproduction on an 8-GPU node (PLAN.md A27): the paper's goal-reaching runs on Cube and Push-T from scratch
# (hjepa/config/lewam/repro-gr.yaml), then LeWAM's own eval (third_party/lewam/docs/REPRODUCE.md: 50 episodes for
# each of the seeds 42, 0, 1; reactive policy, best-of-32 and the gradient planner) of our runs and the authors'.
#
#   sh hjepa/lewam_repro_node.sh all        # everything, in order, logs in outputs/repro/
#   sh hjepa/lewam_repro_node.sh STEP ...   # env | data TASK | train TASK | eval RUN TASK [GPU] | released
# Elsewhere (e.g. WSL with a CUDA GPU): set STABLEWM_HOME, LEWAM_VENV (hjepa/lewam_repro_env.sh), MUJOCO_GL, PAR.
#
# Order in `all`: env; Cube data; Cube training on all 8 GPUs while the Push-T data is made (CPU) and the
# authors' checkpoints are evaluated; then Push-T training; then the evals of our two runs.
# The h5 files and the uncompressed copies are on the NVMe ($STABLEWM_HOME): Cube 102 + 303 GB, Push-T 46 + 352 GB.
set -eu
cd "$(dirname "$0")/.."
export STABLEWM_HOME=${STABLEWM_HOME:-/mnt/nvme/lewam-home} HF_HUB_OFFLINE=1
export MUJOCO_GL=${MUJOCO_GL:-osmesa} PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa} SDL_VIDEODRIVER=dummy
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=third_party/lewam
V=${LEWAM_VENV:-/mnt/nvme/lewam-venv}
PY=$V/bin/python
H=$STABLEWM_HOME
O=outputs/repro
GPUS=${GPUS:-8}
mkdir -p "$H/datasets" "$H/decomp" "$H/checkpoints" "$O"

env_() {  # a fresh node: zstd, OSMesa (MuJoCo renders on the CPU), uv, then LeWAM's environment
    command -v zstd >/dev/null && [ -e /usr/lib/x86_64-linux-gnu/libOSMesa.so.8 ] ||
        { sudo apt-get update -qq && sudo apt-get install -y -qq zstd libosmesa6 libgl1 libegl1 >/dev/null; }
    command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
    [ -x "$PY" ] || sh hjepa/lewam_repro_env.sh "$V" rocm7.2
}

data() {  # TASK: pusht | cube
    case $1 in pusht) f=pusht ;; cube) f=cube ;; *) echo "no task $1"; exit 2 ;; esac
    if [ ! -f "$H/datasets/$f.h5" ]; then
        curl -sfL -C - -o "$H/datasets/$f.h5.zst" "https://huggingface.co/datasets/LeWAM/lewam-$1/resolve/main/$f.h5.zst"
        zstd -d -q -T0 --long=27 -f "$H/datasets/$f.h5.zst" -o "$H/datasets/$f.h5.partial"
        mv "$H/datasets/$f.h5.partial" "$H/datasets/$f.h5" && rm -f "$H/datasets/$f.h5.zst"
    fi
    $PY hjepa/lewam_decompress.py "$H/datasets/$f.h5" "$H/decomp" --workers 96
}

released() {  # the authors' checkpoints
    for t in pusht cube; do
        [ -f "$H/checkpoints/lewam-$t/lewam_best.pt" ] || HF_HUB_OFFLINE=0 $V/bin/hf download "LeWAM/lewam-$t" \
            --local-dir "$H/checkpoints/lewam-$t" >/dev/null
    done
}

train() {  # TASK: Cube or Push-T from scratch, data-parallel on all GPUs, LeWAM's own split
    LEWAM_SPLIT=frame $V/bin/torchrun --standalone --nproc_per_node "$GPUS" hjepa/lewam_train.py "repro-$1" \
        "$H/datasets/$1.h5" "data.decomp_dir=$H/decomp" > "$O/train_$1.log" 2>&1
}

one() {  # RUN TASK GPU MODE SEED: one eval_lewam.py run of 50 episodes
    case $4 in
        policy) args="eval.mode=lewam_policy" ;;
        best_of_k) args="eval.mode=lewam_plan eval.plan_mode=best_of_k" ;;
        grad) args="eval.mode=lewam_plan eval.plan_mode=grad eval.grad_all_k=true eval.grad_tr=0.01" ;;
    esac
    log="$O/eval_$(basename "$1")_$2_$4_$5.log"
    grep -q "RESULTS" "$log" 2>/dev/null && return
    HIP_VISIBLE_DEVICES=$3 CUDA_VISIBLE_DEVICES=$3 $PY third_party/lewam/scripts/eval_lewam.py --config-name "$2"         "policy=$1" "seed=$5" $args "hydra.run.dir=$O/hydra/$(basename "$1")-$2-$4-$5" > "$log" 2>&1
}

eval_() {  # RUN TASK [GPU]: every mode and seed of the paper's table, PAR (default 9) at a time on one GPU
    for seed in 42 0 1; do for mode in policy best_of_k grad; do echo "$mode $seed"; done; done |
        xargs -P "${PAR:-9}" -n 2 sh "$0" one "$1" "$2" "${3:-0}"
}

ours() {  # our run of TASK, as eval_lewam.py finds it under $STABLEWM_HOME/checkpoints
    ln -sfn "$PWD/data/ckpts/lewam/repro-$1/seed42" "$H/checkpoints/repro-$1"
    echo "repro-$1"
}

case ${1:-all} in
    env) env_ ;;
    data) data "$2" ;;
    train) train "$2" ;;
    eval) eval_ "$2" "$3" "${4:-0}" ;;
    one) one "$2" "$3" "$4" "$5" "$6" ;;
    released) released; eval_ lewam-cube cube 0 & eval_ lewam-pusht pusht 1 & wait ;;
    all)
        env_
        released
        data cube > "$O/data_cube.log" 2>&1
        echo "cube data done $(date +%T)"
        train cube & tr=$!
        { data pusht > "$O/data_pusht.log" 2>&1; echo "pusht data done $(date +%T)"; } &
        { eval_ lewam-cube cube 0 & eval_ lewam-pusht pusht 1 & wait; echo "released evals done $(date +%T)"; } &
        wait $tr; echo "cube training done $(date +%T)"
        wait
        train pusht & tr=$!
        eval_ "$(ours cube)" cube 0; echo "cube evals done $(date +%T)"
        wait $tr; echo "pusht training done $(date +%T)"
        eval_ "$(ours pusht)" pusht 0; echo "pusht evals done $(date +%T)"
        ;;
    *) echo "unknown step $1"; exit 2 ;;
esac
