#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

from typing import Tuple
import torch.nn.functional as F

def _infer_target_hw(image: torch.Tensor) -> Tuple[int, int]:
    """
    从 image 推断目标高宽 (H_img, W_img)。
    支持 (H,W), (C,H,W), (H,W,C), (B,C,H,W), (B,H,W,C)。
    """
    if image.dim() == 2:                 # (H,W)
        return image.shape[0], image.shape[1]
    elif image.dim() == 3:
        # 可能是 (C,H,W) 或 (H,W,C)
        if image.shape[0] <= 4 and image.shape[1] >= 8 and image.shape[2] >= 8:
            return image.shape[1], image.shape[2]    # (C,H,W)
        else:
            return image.shape[0], image.shape[1]    # (H,W,C)
    elif image.dim() == 4:
        # 可能是 (B,C,H,W) 或 (B,H,W,C)
        if image.shape[1] <= 4 and image.shape[2] >= 8 and image.shape[3] >= 8:
            return image.shape[2], image.shape[3]    # (B,C,H,W)
        else:
            return image.shape[1], image.shape[2]    # (B,H,W,C)
    else:
        raise ValueError(f"Unsupported image dim: {image.dim()}")


def resize_mask_nearest_hw(mask: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    """
    将 (H,W) 的 mask 用最近邻插值对齐到 image 的空间尺寸 (H_img,W_img)。
    - 仅改变尺寸，不改变取值（最近邻复制原像素）。
    - 返回形状为 (H_img, W_img)，dtype 与输入 mask 相同。
    """
    if mask.dim() != 2:
        raise ValueError(f"mask must be 2D (H,W). Got shape: {tuple(mask.shape)}")

    H_img, W_img = _infer_target_hw(image)

    # 尺寸一致则直接返回原 mask（零拷贝）
    if mask.shape[0] == H_img and mask.shape[1] == W_img:
        return mask

    orig_dtype = mask.dtype
    src_dev = mask.device
    tgt_dev = image.device

    # 最近邻插值（需要浮点），插值在 image 的 device 上做，最后再原样转回 dtype 和 device
    x = mask.to(device=tgt_dev, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    y = F.interpolate(x, size=(H_img, W_img), mode="nearest")                  # (1,1,H_img,W_img)
    out = y.squeeze(0).squeeze(0).to(dtype=orig_dtype, device=src_dev)         # (H_img, W_img) & same dtype

    return out

