#!/bin/sh
# PLAN.md A27 step 3 on an 8-GPU node: the test bed. Each change of hjepa/lewam_plus.py alone against the baseline
# (config plus.yaml, every change off), trained from scratch on TASK (default pusht) with the episode split,
# data-parallel on all GPUs as the A27 reproduction (batch 8 x 128, lr 4.24e-4, bf16) for EPOCHS epochs (default 10,
# a screen; the paper trains 50), one run after another; then LeWAM's eval of each run (lewam_repro_node.sh one:
# 50 episodes x seeds 42, 0, 1 x reactive policy, best of 32, gradient planner; logs outputs/repro/eval_plus-*), and
# the eval-only variants (guidance weights, flow steps, proprio off, subgoals on the long-horizon protocol).
#
#   sh hjepa/lewam_plus_node.sh all              # env, data, every run in RUNS, then the evals and the variants
#   sh hjepa/lewam_plus_node.sh train NAME | eval NAME [GPU] | variants | evals
# Elsewhere: STABLEWM_HOME, LEWAM_VENV, GPUS, EPOCHS, TASK, RUNS as in lewam_repro_node.sh.
set -euf  # -f: no globbing (the overrides hold [ ])
cd "$(dirname "$0")/.."
export STABLEWM_HOME=${STABLEWM_HOME:-/mnt/nvme/lewam-home} HF_HUB_OFFLINE=1 PYTHONPATH=third_party/lewam
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
V=${LEWAM_VENV:-/mnt/nvme/lewam-venv}
H=$STABLEWM_HOME
O=outputs/plus
GPUS=${GPUS:-8}
EPOCHS=${EPOCHS:-10}
TASK=${TASK:-pusht}
RUNS=${RUNS:-base proprio hier distill roll gdrop sflow shortcut vit}
REPRO="sh hjepa/lewam_repro_node.sh"
mkdir -p "$O" "$H/checkpoints"

overrides() {  # NAME: its one change
    case $1 in
        base) ;;
        proprio) if [ "$TASK" = cube ]; then echo "plus.proprio_keys=[proprio_joint_pos,proprio_gripper_opening]"
                 else echo "plus.proprio_keys=[proprio]"; fi ;;           # Push-T: agent x, y, vx, vy
        hier) echo "plus.hier=true" ;;                                    # A26
        distill) echo "plus.distill=true plus.distill_from=$((EPOCHS / 2))" ;;  # A28: from the middle of the run
        roll) echo "plus.roll_k=5" ;;                                     # A29
        gdrop) echo "train.p_drop_goal=0.1" ;;                            # A30 (guidance: variants)
        sflow) echo "model.state_head=flow" ;;                            # A31
        shortcut) echo "plus.shortcut=true" ;;                            # A32 (flow steps: variants)
        vit) echo "model.encoder_backbone=vit model.encoder_size=tiny" ;; # A33
        *) echo "no run $1" >&2; exit 2 ;;
    esac
}

train() {  # NAME
    overrides "$1" > /dev/null  # an unknown NAME stops here
    run=plus-$TASK-$1
    if ! grep -q "DONE best_val" "$O/train_$1.log" 2>/dev/null; then
        $V/bin/torchrun --standalone --nproc_per_node "$GPUS" hjepa/lewam_plus.py plus "$run" "$H/datasets/$TASK.h5" \
            "data.decomp_dir=$H/decomp" "train.epochs=$EPOCHS" "train.warmup_epochs=$(((EPOCHS + 9) / 10))" \
            $(overrides "$1") >> "$O/train_$1.log" 2>&1
    fi
    ln -sfn "$PWD/data/ckpts/lewam/$run/seed42" "$H/checkpoints/$run"
}

eval_() {  # NAME [GPU]: 3 seeds x 3 modes, PAR (default 9) at a time on one GPU
    for mode in policy best_of_k grad; do for seed in 42 0 1; do echo "$mode $seed"; done; done |
        LEWAM_EVAL_SCRIPT=hjepa/lewam_eval_plus.py xargs -P "${PAR:-9}" -n 2 $REPRO one "plus-$TASK-$1" "$TASK" "${2:-0}"
}

variant() {  # NAME ALIAS GPU MODES [VAR=VALUE ...]: run NAME's checkpoint as ALIAS, MODES x 3 seeds, under VAR=VALUE
    name=$1 alias=plus-$TASK-$1-$2 gpu=$3 modes=$4
    shift 4
    ln -sfn "$PWD/data/ckpts/lewam/plus-$TASK-$name/seed42" "$H/checkpoints/$alias"
    for mode in $modes; do for seed in 42 0 1; do echo "$mode $seed"; done; done |
        env "$@" LEWAM_EVAL_SCRIPT=hjepa/lewam_eval_plus.py xargs -P 9 -n 2 $REPRO one "$alias" "$TASK" "$gpu"
}

LONG="eval.random_start=false eval.min_episode_len=101 eval.exec_actions=25 eval.goal_offset_steps=100 eval.eval_budget=200"

variants() {  # the eval-only tests, two GPUs' worth each
    variant gdrop cfg1.5 0 policy LEWAM_CFG=1.5 &           # A30: guidance weight
    variant gdrop cfg2 1 policy LEWAM_CFG=2 &
    variant gdrop cfg3 2 policy LEWAM_CFG=3 &
    variant shortcut s1 3 policy LEWAM_FLOW_STEPS=1 &       # A32: fewer flow steps, with and without shortcut training
    variant shortcut s2 4 policy LEWAM_FLOW_STEPS=2 &
    variant base s1 5 policy LEWAM_FLOW_STEPS=1 &
    variant base s2 6 policy LEWAM_FLOW_STEPS=2 &
    variant proprio vision 7 "policy best_of_k" LEWAM_PROPRIO=0 &  # A17: the proprio model without proprio
    wait
    variant base long 0 "policy best_of_k" "LEWAM_EVAL_ARGS=$LONG" &               # A26: goals 20 chunks away
    variant hier long 1 "policy best_of_k" "LEWAM_EVAL_ARGS=$LONG" &
    variant hier longsub 2 "policy best_of_k" LEWAM_HIER=1 "LEWAM_EVAL_ARGS=$LONG" &
    variant hier sub 3 "policy best_of_k" LEWAM_HIER=1 &                          # subgoals on the usual goals
    variant shortcut s4 4 policy LEWAM_FLOW_STEPS=4 &
    variant base s4 5 policy LEWAM_FLOW_STEPS=4 &
    wait
}

evals() {  # every run in RUNS, one GPU each (round-robin), then the variants
    gpu=0
    for r in $RUNS; do
        eval_ "$r" $gpu &
        gpu=$(((gpu + 1) % GPUS))
        [ $gpu -eq 0 ] && wait
    done
    wait
    variants
}

case ${1:-} in
    train) train "$2" ;;
    eval) eval_ "$2" "${3:-0}" ;;
    variants) variants ;;
    evals) evals ;;
    all)
        echo "start $(date +%T)"
        $REPRO env
        $REPRO data "$TASK"
        echo "data done $(date +%T)"
        for r in $RUNS; do train "$r"; echo "trained $r $(date +%T)"; done
        evals
        echo "evals done $(date +%T)" ;;
    *) echo "usage: $0 all | train NAME | eval NAME [GPU] | variants | evals"; exit 2 ;;
esac
