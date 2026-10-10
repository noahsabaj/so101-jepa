#!/bin/bash
# Kaggle T4: LeWAM's whole training step (hjepa/jobs/step_ref.py) compiled, as CUDA graphs and with Inductor's own
# GroupNorm (the old bug), against eager fp32, at batch 16 and 32 (64 and 128 images; the bug shows from 48).
set -uo pipefail
V=/tmp/lewam-venv
command -v uv >/dev/null || pip install -q uv
[ -x $V/bin/python ] || sh hjepa/lewam_repro_env.sh $V cu126
nvidia-smi -L
export PYTHONPATH=third_party/lewam MPLBACKEND=Agg
$V/bin/python hjepa/jobs/step_ref.py repro-cube 16 32 2>&1 | grep -v "Warning\|warnings.warn\|^W1"
echo "== done"
