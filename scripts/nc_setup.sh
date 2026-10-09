#!/bin/sh
# Set up a National Compute node (8x MI355X, ROCm) for this project, in /mnt/nvme (memory: national-compute):
#   ssh NODE 'sh -s' < scripts/nc_setup.sh
# Clones the repo with its submodules (H-JEPA fork, LeWAM), installs MuJoCo's GL libraries and uv, and syncs the
# ROCm environment. The data comes separately (rsync from noah-pc to /mnt/nvme/so101-jepa/data).
set -e
cd /mnt/nvme
[ -d so101-jepa ] || git clone -q --recursive https://github.com/noahsabaj/so101-jepa.git
cd so101-jepa && git pull -q && git submodule update -q --init --recursive
sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libegl1 libgl1 libosmesa6 rsync > /dev/null
command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null
export PATH=$HOME/.local/bin:$PATH
mkdir -p data outputs
uv sync -q --no-group cuda --group rocm
sh scripts/uvr python -c "import torch; print('torch', torch.__version__, 'hip', torch.version.hip, 'gpus', torch.cuda.device_count())"
