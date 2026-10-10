"""Shared by the LeWAM scripts (PLAN.md A25): LeWAM's code (third_party/lewam, MIT, Fu, Siebert, Halicki,
Balestriero 2026) on the path, and its encoder sized for our 64 x 64 views.

LeWAM's resnet18dp encoder (Diffusion Policy's ResNet-18, GroupNorm, a 32-keypoint spatial softmax) crops
224 px frames to 202. Our views are 64 px: a ResNet-18 makes a 2 x 2 map of them, too coarse for keypoints.
So the frames are upsampled to UPSAMPLE px on the GPU and cropped to the same 202/224 fraction (a 3 x 3 map,
as Diffusion Policy's 84 px robomimic frames). Parameter names are LeWAM's own, so its checkpoints load as is.
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "lewam"))

import lewam.models.lewam as lewam_model  # noqa: E402
from lewam.models.encoders import ResNetEncoder, build_encoder  # noqa: E402

UPSAMPLE = int(os.environ.get("LEWAM_UPSAMPLE", "96"))  # 224: the paper's 7 x 7 map from our 64 px (v50)
VIEWS = ["pixels_scene", "pixels_wrist"]  # the two halves of our 64 x 128 frame, scene first


class SmallImageResNet(ResNetEncoder):
    def features(self, pixels):
        pixels = F.interpolate(pixels, size=(UPSAMPLE, UPSAMPLE), mode="bilinear", align_corners=False)
        return super().features(pixels)


def _build_encoder(backbone, output_dim, proj_hidden, size="tiny", img_size=224, checkpoint=None):
    if backbone == "resnet18dp" and img_size < 224:
        return SmallImageResNet(output_dim, proj_hidden, group_norm=True, crop_size=round(UPSAMPLE * 202 / 224))
    return build_encoder(backbone, output_dim, proj_hidden, size=size, img_size=img_size, checkpoint=checkpoint)


lewam_model.build_encoder = _build_encoder

_forward_mot = lewam_model.LeWAM.forward_mot


def _forward_mot_fp32_tokens(self, z_history, *args, **kwargs):
    # Under bf16 autocast the encoder gives bf16 latents, and forward_mot allocates its token buffer in their
    # dtype; the embeddings written into it add fp32 parameters (fp32), and the index put fails. An fp32
    # buffer fixes that; the blocks still run in bf16 under autocast. A no-op in fp32.
    return _forward_mot(self, z_history.float(), *args, **kwargs)


lewam_model.LeWAM.forward_mot = _forward_mot_fp32_tokens


def _build_attn_mask_nosync(self, history_pad, goal_keep=None):
    # LeWAM's mask with no GPU -> CPU sync (its history_pad.any() and boolean index put each stall the CPU
    # until the GPU is idle, every batch). The same mask: an all-False key_pad changes nothing.
    B = history_pad.shape[0]
    key_pad = torch.zeros(B, self.n_tokens, dtype=torch.bool, device=history_pad.device)
    key_pad[:, self.token_indices["hist"]] = history_pad.repeat(1, self.num_views)
    if goal_keep is not None:
        key_pad[:, self.token_indices["goal"]] = ~goal_keep[:, None]
    attends = self.attends[None].expand(B, -1, -1) & ~key_pad[:, None, :]
    mask = torch.zeros(B, self.n_tokens, self.n_tokens, device=history_pad.device)
    return mask.masked_fill(~attends, float("-inf"))[:, None]


lewam_model.LeWAM._build_attn_mask = _build_attn_mask_nosync

TRAINER = ROOT / "third_party" / "lewam" / "scripts" / "train_lewam.py"
SPLIT = "for starts in (possible_starts[n_val:], possible_starts[:n_val])"


def episode_split(possible_starts, n_val, ep_starts):
    """The train/val split by episode: val is a fixed 10% of the episodes, train the start points of a fraction
    F (LEWAM_DATA_FRACTION, default 1; data scaling, v53) of the other episodes, repeated to the length of the
    full train set so that every F gets the same epochs, steps and val passes. LeWAM's own split is by start
    point: nearly every val start has train starts in its episode, a frame or two away, so its val loss rewards
    recall, and lewam_best.pt (the best val) cannot show overfitting. Runs up to v51 and v47c used it
    (LEWAM_SPLIT=frame reproduces them)."""
    if os.environ.get("LEWAM_SPLIT", "episode") == "frame":
        return possible_starts[n_val:], possible_starts[:n_val]
    fraction = float(os.environ.get("LEWAM_DATA_FRACTION", "1"))
    episodes = torch.unique(ep_starts)
    episodes = episodes[torch.randperm(len(episodes), generator=torch.Generator().manual_seed(0))]
    held, pool = episodes[:len(episodes) // 10], episodes[len(episodes) // 10:]
    keep = pool[:max(1, round(fraction * len(pool)))]
    episode_of = ep_starts[possible_starts]
    full = possible_starts[torch.isin(episode_of, pool)]
    train = possible_starts[torch.isin(episode_of, keep)]
    train = train.repeat(-(-len(full) // len(train)))[:len(full)]
    val = possible_starts[torch.isin(episode_of, held)]
    print(f"[lewam] data fraction {fraction}: {len(keep)} of {len(pool)} train episodes ({len(held)} held out "
          f"for val), {len(train)} train starts (repeated), {len(val)} val starts", flush=True)
    return train, val


class Ddp:
    """Data-parallel training of LeWAM's trainer under torchrun (PLAN.md A27: the paper's 50 epochs on Cube or Push-T
    take ~9 h on one MI355X, ~1.2 h on eight). Each rank trains on its own 1/W of every epoch with data.batch_size
    samples, so the batch is W x data.batch_size; the gradients are averaged over the ranks before the clip, the
    epoch's losses before the log, the best-checkpoint choice and the collapse check. Only rank 0 prints and saves.
    The model has no batch statistics (GroupNorm), and SIGReg sees each rank's own batch, as on one GPU."""

    def __init__(self):
        import torch.distributed as dist

        self.dist, self.world = dist, int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.on, self.checked = self.world > 1, False
        if self.on and not dist.is_initialized():
            # device "cuda" is then this rank's GPU (a test with more ranks than GPUs shares them, over gloo)
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]) % torch.cuda.device_count())
            dist.init_process_group(os.environ.get("LEWAM_DDP_BACKEND", "nccl"))

    def sampler(self, dataset, num_samples):
        if not self.on:
            return torch.utils.data.RandomSampler(dataset, num_samples=num_samples)
        return _ShardSampler(len(dataset), num_samples // self.world, self.rank, self.world)

    def grads(self, optimizer):
        if not self.on:
            return
        grads = [p.grad for g in optimizer.param_groups for p in g["params"] if p.grad is not None]
        flat = torch._utils._flatten_dense_tensors(grads)
        self.dist.all_reduce(flat)
        flat /= self.world
        for grad, avg in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
            grad.copy_(avg)

    def stats(self, store, n):
        """The sums of the logged losses and of the sample counts over the ranks."""
        if not self.on:
            return store, n
        keys = sorted(store)
        t = torch.tensor([float(store[k]) for k in keys] + [float(n)], device="cuda", dtype=torch.float64)
        self.dist.all_reduce(t)
        if not self.checked:  # once an epoch (stats() runs for train, then val): the weights must stay the same
            in_sync = self.in_sync()
            if self.rank == 0:
                print(f"[ddp] {self.world} ranks, weights {'in sync' if in_sync else 'OUT OF SYNC'}", flush=True)
        self.checked = not self.checked
        return dict(zip(keys, t[:-1].tolist())), int(t[-1].item())

    def in_sync(self):
        total = torch.stack([p.detach().double().sum() for p in self.model.parameters()]).sum()
        lo, hi = total.clone(), total.clone()
        self.dist.all_reduce(lo, self.dist.ReduceOp.MIN)
        self.dist.all_reduce(hi, self.dist.ReduceOp.MAX)
        return bool(hi - lo <= 1e-6 * max(1.0, abs(float(total))))

    def broadcast(self, model, seed):
        """Rank 0's initial weights on every rank, then a different random stream per rank (dropout, goals)."""
        self.model = model
        if self.on:
            for tensor in list(model.parameters()) + list(model.buffers()):
                self.dist.broadcast(tensor.data, 0)
            torch.manual_seed(seed + self.rank)


class _ShardSampler(torch.utils.data.Sampler):
    """One random permutation per epoch, the same on every rank (the same seed); each rank takes every W-th index."""

    def __init__(self, n, num_samples, rank, world):
        self.n, self.num_samples, self.rank, self.world, self.epoch = n, num_samples, rank, world, 0

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        g = torch.Generator().manual_seed(1234 + self.epoch)
        self.epoch += 1
        perm = torch.randperm(self.n, generator=g)
        if self.n < self.num_samples * self.world:
            perm = perm.repeat(-(-self.num_samples * self.world // self.n))
        return iter(perm[self.rank::self.world][:self.num_samples].tolist())


DDP_PATCHES = [  # (LeWAM's line, the line in its place); each must occur exactly once
    ("sampler=RandomSampler(dataset, num_samples=num_samples)", "sampler=_ddp.sampler(dataset, num_samples)"),
    ("    model = build_model(lewam_cfg).to(device)\n",
     "    model = build_model(lewam_cfg).to(device)\n    _ddp.broadcast(model, cfg.seed)\n"),
    ("            loss.backward()\n", "            loss.backward()\n            _ddp.grads(optimizer)\n"),
    ("        val_action_loss = ", "        train_stats, train_n = _ddp.stats(train_stats, train_n)\n"
     "        val_stats, val_n = _ddp.stats(val_stats, val_n)\n        val_action_loss = "),
]


def trainer_namespace(name, nosync=True):
    """Run LeWAM's train_lewam.py as module `name` ("__main__" trains) and return its globals. nosync: its
    run_batch keeps the logged losses on the GPU (.detach(), not .item(): five CPU-GPU syncs a batch), read
    once an epoch; the normalization constants live on the GPU (a pageable copy syncs the stream). Under torchrun
    (WORLD_SIZE > 1) it trains data-parallel (Ddp)."""
    import lewam.train.utils as utils

    ddp = Ddp()
    src = TRAINER.read_text()
    for old, new in DDP_PATCHES:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    if nosync:
        a, b = src.index("def run_batch("), src.index("def accumulate(")
        src = src[:a] + src[a:b].replace(".item()", ".detach().float()") + src[b:]
        for line in ('val_action_loss = val_stats.get("act", 0.0) / max(val_n, 1)',
                     'val_zstd = val_stats.get("zstd", float("inf")) / max(val_n, 1)'):
            assert line in src, line
            lhs, rhs = line.split(" = ", 1)
            src = src.replace(line, f"{lhs} = float({rhs})")
        if torch.cuda.is_available():
            utils._IMG_MEAN, utils._IMG_STD = utils._IMG_MEAN.cuda(), utils._IMG_STD.cuda()
    assert SPLIT in src, SPLIT
    src = src.replace(SPLIT, "for starts in _episode_split(possible_starts, n_val, ep_starts)")
    namespace = {"__name__": name, "__file__": str(TRAINER), "_episode_split": episode_split, "_ddp": ddp}
    if ddp.rank > 0:  # only rank 0 logs and writes files (checkpoints, configs)
        import pathlib

        namespace["print"] = lambda *args, **kwargs: None
        torch.save = lambda *args, **kwargs: None
        pathlib.Path.write_text = lambda *args, **kwargs: None
    exec(compile(src, str(TRAINER), "exec"), namespace)
    return namespace


def split_views(pixels):
    """(..., 64, 128, 3) frame -> (scene, wrist), each (..., 64, 64, 3)."""
    w = pixels.shape[-2] // 2
    return pixels[..., :w, :], pixels[..., w:, :]
