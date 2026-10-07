# NVIDIA or AMD GPUs

The project runs on NVIDIA (CUDA) and AMD (ROCm) GPUs with the same code and the same torch version.

## How

- **torch:** `pyproject.toml` has one dependency group per GPU vendor, `cuda` (CUDA 12.8 wheels) and
  `rocm` (ROCm 7.2 wheels, Linux only), with the same torch, torchvision and torchaudio versions.
  `cuda` is the default group, so plain `uv run` gives the CUDA build.
- **scripts/uvr** is `uv run` with this computer's build: `sh scripts/uvr python hjepa/train.py ...`.
  On a computer with an AMD GPU (`/dev/kfd`) and no `nvidia-smi`, it adds
  `--no-group cuda --group rocm`. `SO101_GPU=cuda|rocm` overrides the detection.
- **Shell scripts** source `scripts/gpu.sh`, which sets `GPU`, `UV_GROUPS` and `GL` and defines
  `uvr` and `gpu_sample` (GPU memory and use, from nvidia-smi or rocm-smi). Write new scripts the
  same way: `uvr python ...`, never `uv run python ...`, and `uv sync $UV_GROUPS`.
- **Python code** uses the `cuda` device name. ROCm builds of torch answer to it (HIP), so
  `torch.device("cuda")`, `.cuda()`, `torch.cuda.*` and `CUDA_VISIBLE_DEVICES` work on both. Do not
  call vendor tools or vendor-only libraries (pynvml, flash-attn, xformers, apex, bitsandbytes) from
  Python; use `torch.nn.functional.scaled_dot_product_attention`.
- **MuJoCo rendering:** `MUJOCO_GL` defaults to `egl` on NVIDIA and `osmesa` (CPU) on AMD, as AMD
  Instinct GPUs (MI300, MI355X) have no graphics. A Radeon card can set `MUJOCO_GL=egl`.
- **Several GPUs:** `scripts/run_lanes.sh LANES_FILE` runs one lane of jobs per GPU.

## Change the torch version

Pick a version with wheels for both vendors (torchaudio too: h-jepa needs it; its last release is
2.11). Check https://download.pytorch.org/whl/cu128/torch/ and .../rocm7.2/torch/, change the
version in `dependencies` and in both groups, then `uv lock`.

## On an AMD node

```sh
sudo apt-get install -y libegl1 libgl1 libosmesa6   # MuJoCo; or use a container that has them
sh scripts/uvr python -c "import torch; print(torch.version.hip, torch.cuda.device_count())"
```

The user needs access to `/dev/kfd` and `/dev/dri` (groups `render` and `video`), or run in Docker
with `--device=/dev/kfd --device=/dev/dri --group-add video`.
