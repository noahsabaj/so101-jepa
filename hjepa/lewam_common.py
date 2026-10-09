"""Shared by the LeWAM scripts (PLAN.md A25): LeWAM's code (third_party/lewam, MIT, Fu, Siebert, Halicki,
Balestriero 2026) on the path, and its encoder sized for our 64 x 64 views.

LeWAM's resnet18dp encoder (Diffusion Policy's ResNet-18, GroupNorm, a 32-keypoint spatial softmax) crops
224 px frames to 202. Our views are 64 px: a ResNet-18 makes a 2 x 2 map of them, too coarse for keypoints.
So the frames are upsampled to UPSAMPLE px on the GPU and cropped to the same 202/224 fraction (a 3 x 3 map,
as Diffusion Policy's 84 px robomimic frames). Parameter names are LeWAM's own, so its checkpoints load as is.
"""

import sys
from pathlib import Path

import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "lewam"))

import lewam.models.lewam as lewam_model  # noqa: E402
from lewam.models.encoders import ResNetEncoder, build_encoder  # noqa: E402

UPSAMPLE = 96
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


def split_views(pixels):
    """(..., 64, 128, 3) frame -> (scene, wrist), each (..., 64, 64, 3)."""
    w = pixels.shape[-2] // 2
    return pixels[..., :w, :], pixels[..., w:, :]
