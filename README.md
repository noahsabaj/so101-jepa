# so101-jepa

Written in ASD-STE100 (Simplified Technical English).

Research: an SO-101 arm learns to pick and place, and later to assemble another SO-101 arm, with only
JEPA models (no VLM or VLA). The target design is one JEPA with a learned action head (LeWAM): the head
acts, and a planner is only an optional refinement.

## Status (2026-10-10)

- Sim only (MuJoCo). The real arms are not built yet.
- Now (PLAN.md A27): we reproduce LeWAM on its own benchmarks, then use them as a test bed for our
  changes. Our Cube retrain from scratch (SO-JEPA v55) gives the paper's results. The Push-T retrain
  and the test-bed runs (hjepa/lewam_plus.py) wait for a GPU lease.

## Documents

- [PLAN.md](PLAN.md): the north star, the rules, the assumptions (each with a test), the ladder and
  every result.
- [MODELS.md](MODELS.md): every trained model (SO-JEPA vN), with its recipe and results.
- [GPU.md](GPU.md): NVIDIA and AMD GPUs, the torch build and MuJoCo rendering.

## Layout

| Folder | Contents |
|---|---|
| `sim/` | MuJoCo scene and env, scripted expert, data collection, closed-loop tests (rungs 1 and 2), self-play |
| `hjepa/` | Training, planning and offline tests of our world models; LeWAM wrappers (`lewam_*.py`); configs in `hjepa/config/` |
| `hjepa/jobs/` | Small jobs for Kaggle and Colab (`pack.sh` makes the job folder) |
| `community/` | Build of the community SO-100/SO-101 dataset |
| `bakeoff/` | Simulator test (servo insertion) and the convex parts for rung 3 |
| `scripts/` | GPU setup (`gpu.sh`, `uvr`), lanes on many GPUs, node setup |
| `tests/` | Tests that run on the CPU |
| `third_party/` | H-JEPA fork ([noahsabaj/H-JEPA](https://github.com/noahsabaj/H-JEPA), branch `so101`) and [LeWAM](https://github.com/MinghaoFu/lewam) |
| `assets/so101/` | The official SO-101 model |

Data, checkpoints and results (`data/`, `outputs/`) are not in git.

## Setup

```sh
git clone --recursive https://github.com/noahsabaj/so101-jepa
cd so101-jepa
uv sync          # NVIDIA (CUDA 12.8). AMD: use scripts/uvr (GPU.md).
```

H-JEPA training and planning run on Linux only (stable-pretraining uses POSIX signals). On Windows, use WSL.

## Tests

```sh
CUDA_VISIBLE_DEVICES=-1 uv run python -m pytest tests -q
```
