from typing import Callable, Dict, Optional, Any, Iterable, Tuple
import numpy as np
from PIL import Image, ImageOps
import torch
import os

import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation

# 几何过滤阈值（可按分辨率微调）
MIN_AREA = 32           # 最小像素面积：<64 像素直接视为细屑
MIN_FILL_RATIO = 0.05   # 最小填充率：实际像素 / bbox面积，过低通常是“毛刺/细丝”
MIN_GLOBAL_FILL_RATIO = 0.0001   # 最小全局充率：实际像素 / 图像面积，过低通常是“毛刺/细丝”
MIN_SHORT_SIDE = 5      # bbox短边阈值：过细（如1~2px宽）剔除

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def _geom_filter_single_label(labels_2d: np.ndarray, lb: int) -> bool:
    """几何过滤：返回 True 表示通过；False 表示丢弃为细屑。"""
    ys, xs = np.where(labels_2d == lb)
    if ys.size == 0:
        return False
    area = ys.size
    if area < MIN_AREA:
        return False
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    h = y2 - y1 + 1
    w = x2 - x1 + 1
    if min(h, w) < MIN_SHORT_SIDE:
        return False
    bbox_area = h * w
    fill_ratio = float(area) / float(bbox_area) if bbox_area > 0 else 0.0
    if fill_ratio < MIN_FILL_RATIO:
        return False
    H, W = labels_2d.shape
    global_fill_ratio = float(area) / (H * W)
    if global_fill_ratio < MIN_GLOBAL_FILL_RATIO:
        return False
    return True


def dilate_labels(labels: np.ndarray, radius: int = 3) -> np.ndarray:
    """
    对标签图做形态学膨胀，每个实例 mask 会扩大指定的半径。
    radius: 膨胀的像素半径
    """
    out = np.zeros_like(labels)
    for lb in np.unique(labels):
        if lb == 0:
            continue
        mask = (labels == lb)
        dilated = binary_dilation(mask, iterations=radius)
        out[dilated] = lb
    return out

def filter_labels_by_similarity(
    gt_path: str,
    render_path: str,
    similarity_path: str,
    label_npy_path: str,
    output_colored_path: Optional[str] = None,   # 输出彩色可视化 PNG
    output_label_path: Optional[str] = None,     # 输出过滤后的标签 .npy
    predicate: Optional[Callable[[float, int, int, Dict[str, float]], bool]] = None,
    background_label: int = 0,                   # 认为是背景的标签编号（默认 0）
    ignore_labels: Optional[Iterable[int]] = None,
    keep_background: bool = True,                # 彩色可视化时，背景是否填背景色；否则填黑
    strict_shape: bool = False,                  # True 时尺寸不一致直接报错；False 时把相似度图 resize 到标签大小
    bg_rgb: Tuple[int, int, int] = (0, 0, 0),    # 可视化的背景颜色
    palette_name: str = "tab20",                 # 可视化调色板
    seed: Optional[int] = 0,                     # 若调色板不够，用随机补色，则使用该种子
) -> Dict[int, Dict[str, float]]:
    """
    根据标签图（每像素为整数标签）在相似度图上计算每个标签区域的统计量（均值/方差/像素数），
    通过 predicate(mean, area, label, stats)->bool 判定是否保留，输出：
      1) 仅保留区域着色的 PNG 可视化
      2) 可选：过滤后的标签图（未保留的置为 background_label）

    返回：{label: {'mean':..., 'std':..., 'area':...}} 仅包含保留下来的标签的统计。
    """
    # 读取并校正 EXIF 方向的相似度图 -> 灰度
    sim_img = Image.open(similarity_path)
    sim_img = ImageOps.exif_transpose(sim_img).convert("L")

    gt_img = Image.open(gt_path)
    render_img = Image.open(render_path)

    # 读取标签图
    labels = np.load(label_npy_path)
    if labels.ndim == 3 and labels.shape[0] == 1:
        labels = labels[0]
    if labels.ndim != 2:
        raise ValueError(f"label_npy_path should have shape HxW or 1xHxW, got {labels.shape}")

    def resize(img, H, W, strict_shape):
        h, w = img.size
        if (h, w) != (H, W):
            if strict_shape:
                raise ValueError(
                    f"Size mismatch: {w}x{h} vs labels {W}x{H}. "
                    f"Set strict_shape=False to auto-resize similarity to label size."
                )
            # 调整相似度图到标签图大小（双线性）
            img = img.resize((W, H), resample=Image.BILINEAR)
        return img

    uniq = np.unique(labels)
    kept_labels = np.zeros_like(labels)
    for lb in uniq.tolist():
        # step-1: 先剔除几何上“细屑”的mask
        if not _geom_filter_single_label(labels_2d=labels, lb=lb):
            continue
        kept_labels[labels == lb] = lb

    #labels = dilate_labels(kept_labels, radius=3)
    labels = kept_labels

    H, W = labels.shape
    sim_img = resize(sim_img, H, W, strict_shape)
    gt_img = resize(gt_img, H, W, strict_shape)
    render_img = resize(render_img, H, W, strict_shape)
    
    gt = PILtoTorch(gt_img, gt_img.size[-2:])[:3, ...].cuda()
    render = PILtoTorch(render_img, render_img.size[-2:])[:3, ...].cuda()
    with torch.no_grad():
        l1_loss = torch.abs((render - gt)).mean(dim=0)
        loss = l1_loss

    loss = loss.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

    # 转数组并归一化到 [0,1]
    sim = np.asarray(sim_img, dtype=np.float32)
    sim_min, sim_max = float(sim.min()), float(sim.max())
    if sim_max > sim_min:
        sim_norm = (sim - sim_min) / (sim_max - sim_min)
    else:
        sim_norm = np.zeros_like(sim, dtype=np.float32)

    labels = labels.astype(np.int64, copy=False)
    flat_lbl = labels.ravel()
    flat_sim = sim_norm.ravel()
    flat_loss = loss.ravel()

    # 可忽略标签集合
    ignore_set = set(ignore_labels or [])
    ignore_set.add(background_label)  # 背景默认不参与判定

    # 找到所有出现过的标签
    uniq_labels = np.unique(flat_lbl)

    # 通过 bincount 一次性统计每个标签的 area、sum、sumsq（高效且避免逐像素循环）
    max_label = int(uniq_labels.max()) if uniq_labels.size > 0 else 0
    # 注意：如果标签号非常稀疏且很大，bincount 可能较大；通常问题不大，若特别大可换散列表方式
    area = np.bincount(flat_lbl, minlength=max_label + 1)
    sum_vals = np.bincount(flat_lbl, weights=flat_sim, minlength=max_label + 1)
    sumsq_vals = np.bincount(flat_lbl, weights=flat_sim * flat_sim, minlength=max_label + 1)

    sum_loss = np.bincount(flat_lbl, weights=flat_loss, minlength=max_label + 1)

    # 默认 predicate：均值 >= 0.5
    if predicate is None:
        def predicate(mean, area_i, label_i, stats):
            return mean >= 0.5

    kept = set()
    kept_stats: Dict[int, Dict[str, float]] = {}

    # 判定每个标签是否保留
    for lb in uniq_labels.tolist():

        lb = int(lb)
        if lb in ignore_set:
            continue
        a = int(area[lb])
        if a == 0:
            continue
        s = float(sum_vals[lb])
        mean = s / a
        # 使用 sumsq 计算方差 -> std
        ss = float(sumsq_vals[lb])
        var = max(0.0, ss / a - mean * mean)
        std = var ** 0.5
        
        loss_mean = float(sum_loss[lb]) / a

        stats = {"mean": mean, "std": std, "area": float(a), "loss_mean": loss_mean}
        if predicate(mean, a, lb, stats):
            kept.add(lb)
            kept_stats[lb] = stats

    # 生成可视化彩色图（仅保留标签上色）
    # 调色板：先用 matplotlib 的离散色，不够时用随机颜色补齐
    cmap = plt.get_cmap(palette_name)
    palette_colors = [np.array(cmap(i % cmap.N)[:3]) for i in range(max(1, len(kept)))]
    rng = np.random.default_rng(seed)

    def color_for_label(idx: int) -> np.ndarray:
        if idx < len(palette_colors):
            return palette_colors[idx]
        return rng.random(3)

    kept_list = sorted(kept)
    color_map: Dict[int, np.ndarray] = {lb: color_for_label(i) for i, lb in enumerate(kept_list)}

    filtered_labels = labels.copy()
    mask_keep = np.isin(filtered_labels, kept_list)
    filtered_labels[~mask_keep] = background_label
    labels = dilate_labels(filtered_labels, radius=3)

    # 生成 RGB 可视化
    vis = np.zeros((H, W, 3), dtype=np.float32)
    if keep_background:
        vis[:] = np.array(bg_rgb, dtype=np.float32) / 255.0  # 背景底色

    # 为每个保留标签着色（矢量化赋值）
    for lb, color in color_map.items():
        vis[labels == lb] = color

    # 保存可视化 PNG
    if output_colored_path is None:
        base = os.path.splitext(os.path.basename(label_npy_path))[0]
        output_colored_path = os.path.join(os.path.dirname(label_npy_path), f"{base}_filtered.png")
    Image.fromarray((vis * 255).astype(np.uint8), mode="RGB").save(output_colored_path)

    # 可选：保存过滤后的标签 .npy（未保留的标签设为 background_label）
    if output_label_path is not None:
        filtered_labels = labels.copy()
        mask_keep = np.isin(filtered_labels, kept_list)
        filtered_labels[~mask_keep] = background_label
        np.save(output_label_path, filtered_labels)

    return kept_stats

def get_image_names(folder_path):
    # 支持的图片后缀
    img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    
    # 遍历文件夹并筛选图片文件
    image_list = [
        f for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f)) 
        and os.path.splitext(f)[1].lower() in img_exts
    ]
    
    return image_list

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str)
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--root", type=str)
parser.add_argument("--masks", type=str)
parser.add_argument("--iteration", type=int, default=10000)
parser.add_argument("--output_folder", type=str, default="masks")
parser.add_argument("--sim_folder", type=str, default="dino")
parser.add_argument("--tau1", type=float, default=0.75)
parser.add_argument("--tau2", type=float, default=0.05)
args = parser.parse_args()

# model_name = "patio"
# dataset_name = "patio"

model_name = args.model_name
output_folder = args.output_folder
sim_folder = args.sim_folder
dataset_name = args.dataset_name
if dataset_name is None:
    dataset_name = model_name
iteration = args.iteration


masks_path = args.masks
root = args.root
sims_path = f"{root}/{model_name}/train/ours_{iteration}/{sim_folder}/"
gts_path = f"{root}/{model_name}/train/ours_{iteration}/gt/"
renders_path = f"{root}/{model_name}/train/ours_{iteration}/renders/"
output_path = f"{root}/{model_name}/train/ours_{iteration}/{output_folder}/"
output_npy_path = f"{root}/{model_name}/train/ours_{iteration}/{output_folder}_npy/"
print(output_path)

def rule(mean, area, label_id, stats):
    if mean <= args.tau1: #0.75 onthego
        if stats["loss_mean"] >= args.tau2: # 0.05 onthego
            return True
    return False

os.makedirs(output_path, exist_ok=True)
os.makedirs(output_npy_path, exist_ok=True)

image_list = get_image_names(sims_path)
from tqdm import tqdm
for image in tqdm(image_list):
    npy = image.split(".")[0] + ".npy"
    sim_path = os.path.join(sims_path, image)
    gt_path = os.path.join(gts_path, image)
    render_path = os.path.join(renders_path, image)
    npy_path = os.path.join(masks_path, npy)
    stats = filter_labels_by_similarity(
        gt_path=gt_path,
        render_path=render_path,
        similarity_path=sim_path,
        label_npy_path=npy_path,
        output_colored_path=os.path.join(output_path, image),
        output_label_path=os.path.join(output_npy_path, npy),
        background_label=0,
        predicate=rule
    )
