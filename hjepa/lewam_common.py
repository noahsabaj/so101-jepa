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


def trainer_namespace(name, nosync=True):
    """Run LeWAM's train_lewam.py as module `name` ("__main__" trains) and return its globals. nosync: its
    run_batch keeps the logged losses on the GPU (.detach(), not .item(): five CPU-GPU syncs a batch), read
    once an epoch; the normalization constants live on the GPU (a pageable copy syncs the stream)."""
    import lewam.train.utils as utils

    src = TRAINER.read_text()
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
    namespace = {"__name__": name, "__file__": str(TRAINER)}
    exec(compile(src, str(TRAINER), "exec"), namespace)
    return namespace


def split_views(pixels):
    """(..., 64, 128, 3) frame -> (scene, wrist), each (..., 64, 64, 3)."""
    w = pixels.shape[-2] // 2
    return pixels[..., :w, :], pixels[..., w:, :]
