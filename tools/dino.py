#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch DINO similarity map generator with GT-based normalization.
"""

import os
import argparse
from argparse import Namespace
from typing import Tuple, List

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as tvT
import numpy as np
from tqdm import tqdm
import albumentations as A

#from scene.dynamic_model import *  # noqa
from dino_model import *

def load_jpg_as_tensor(path: str, add_batch_dim: bool = True) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    t = tvT.ToTensor()(img)
    if add_batch_dim:
        t = t.unsqueeze(0)
    return t

def save_similarity_map(sim: np.ndarray, out_path: str) -> None:
    if sim.min() < 0.0 or sim.max() > 1.0:
        sim = (sim + 1.0) * 0.5
    sim = np.clip(sim, 0.0, 1.0)
    #sim = sim / sim.max()
    img8 = (sim * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(img8).save(out_path)

def intersect_filenames(gt_dir: str, ren_dir: str, exts: Tuple[str, ...]) -> List[str]:
    def list_files(d: str) -> set:
        return {f for f in os.listdir(d) if f.lower().endswith(exts)}
    return sorted(list(list_files(gt_dir) & list_files(ren_dir)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--exts", default=".jpg,.jpeg,.png")
    args = parser.parse_args()

    root = args.root
    gt_dir = os.path.join(root, "gt")
    ren_dir = os.path.join(root, "renders")
    out_dir = os.path.join(root, "dino_try")
    os.makedirs(out_dir, exist_ok=True)

    exts = tuple(e.strip().lower() for e in args.exts.split(",") if e.strip())
    model = Dino().to(args.device)
    model.eval()

    files = intersect_filenames(gt_dir, ren_dir, exts)
    if not files:
        raise RuntimeError("No matching files found.")

    # Compute normalization stats from all GT images
    means, stds = [], []
    for fname in tqdm(files, desc="GT stats"):
        gt = load_jpg_as_tensor(os.path.join(gt_dir, fname), add_batch_dim=False)
        means.append(gt.mean(dim=(1, 2)))
        stds.append(gt.std(dim=(1, 2)))
    mean_all = torch.stack(means).mean(dim=0).tolist()
    std_all = torch.stack(stds).mean(dim=0).tolist()

    alb_transform = A.Compose(A.Normalize(mean=mean_all, std=std_all))

    for fname in tqdm(files, desc="Processing", unit="pair"):
        gt_path = os.path.join(gt_dir, fname)
        ren_path = os.path.join(ren_dir, fname)
        out_path = os.path.join(out_dir, fname)

        gt = load_jpg_as_tensor(gt_path, add_batch_dim=True)
        pred = load_jpg_as_tensor(ren_path, add_batch_dim=True)
        _, _, H, W = gt.shape

        gt_down = dino_downsample(gt, max_size=model.max_size).to(args.device)
        pred_down = dino_downsample(pred, max_size=model.max_size).to(args.device)

        with torch.no_grad():
            # gt_feats = GetDinov2RegFeats(un_model.original_model, un_model.fine_model, gt_down, alb_transform)
            # pred_feats = GetDinov2RegFeats(un_model.original_model, un_model.fine_model, pred_down, alb_transform)
            
            gt_down = process_image(gt_down, 14, alb_transform, args.device)
            pred_down = process_image(pred_down, 14, alb_transform, args.device)
            
            gt_feats = model.fine_model.get_intermediate_layers(gt_down, n=[11], reshape=True, norm=True)[-1]
            pred_feats = model.fine_model.get_intermediate_layers(pred_down, n=[11], reshape=True, norm=True)[-1]

            dino_cosine = F.cosine_similarity(gt_feats, pred_feats, dim=1).unsqueeze(1)
            dino_cosine_up = F.interpolate(dino_cosine, size=(H, W), mode="bilinear", align_corners=False)

        sim_np = dino_cosine_up.cpu().squeeze(0).squeeze(0).numpy().astype(np.float32)
        save_similarity_map(sim_np, out_path)

    print(f"Done. Results saved to: {out_dir}")

if __name__ == "__main__":
    main()