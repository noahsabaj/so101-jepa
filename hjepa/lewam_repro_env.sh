#!/bin/sh
# LeWAM's own environment for reproducing its results on its own tasks (PLAN.md A27): the pins of
# third_party/lewam/pyproject.toml and install.sh (mujoco 3.8.1, ogbench 1.2.1, stable-worldmodel 0.1.1, ...), with uv
# instead of conda, not our pins (mujoco 3.14: versions change contact outcomes). robomimic and DexMimicGen (Drawer,
# Transport, Tool Hang only) are left out. The lewam package itself runs from source (PYTHONPATH=third_party/lewam),
# so the submodule stays clean.
#   sh hjepa/lewam_repro_env.sh VENV [cu128|rocm7.2]
set -eu
VENV=$1; GPU=${2:-cu128}
uv venv -q --clear --python 3.11 "$VENV"
uv pip install -q -p "$VENV/bin/python" "cmake<4"  # egl_probe builds with cmake (install.sh: conda's cmake<4)
export PATH="$(cd "$VENV/bin" && pwd):$PATH"
uv pip install -q -p "$VENV/bin/python" --index-strategy unsafe-best-match \
    --extra-index-url "https://download.pytorch.org/whl/$GPU" \
    "torch==2.11.0+$GPU" "torchvision==0.26.0+$GPU" \
    "stable-worldmodel[format,train]==0.1.1" "stable-pretraining==0.1.7" "timm>=1.0.20" pygame pymunk shapely \
    "ogbench==1.2.1" "mujoco==3.8.1" "dm_control==1.0.41" "robosuite==1.5.1" egl_probe termcolor tensorboardX psutil \
    hdf5plugin hydra-core "datasets==5.1.0" "huggingface_hub==1.33.0"  # as in uv.lock: a newer hub drags datasets back to 1.x
"$VENV/bin/python" -c "import torch, mujoco, ogbench, stable_worldmodel; print('torch', torch.__version__, 'gpu', torch.cuda.is_available(), 'mujoco', mujoco.__version__)"
