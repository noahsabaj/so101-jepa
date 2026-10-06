#!/bin/sh
# Install Isaac Sim and Isaac Lab in their own environment (.isaac) and run the insertion test.
# The NVIDIA Isaac Sim EULA is accepted through OMNI_KIT_ACCEPT_EULA (Noah's approval, 2026-10-06).
# Usage (Linux, NVIDIA RTX GPU): sh bakeoff/isaac_setup.sh [trials]
set -e
export OMNI_KIT_ACCEPT_EULA=YES
if [ ! -x .isaac/bin/python ]; then
  uv venv --python 3.11 .isaac
  VIRTUAL_ENV=.isaac uv pip install "isaaclab[isaacsim,all]" trimesh scipy \
    --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match
fi
.isaac/bin/python bakeoff/insert_isaaclab.py "${1:-100}"
