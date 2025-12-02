# -*- coding: utf-8 -*-
"""
WHU-OPT-SAR / Potsdam 整图滑窗推理 + 拼回整图

使用方式示例（单张 WHU-OPT-SAR 大图）：
python whuoptsar_Large_test.py \
    -c config/whuoptsar/2_modality/your_config.py \
    -o fig_results/whuoptsar/freASam_Large_image \
    --img_path /path/to/your/optical_big.tif \
    --aux2_path /path/to/your/sar_big.tif \
    --mask_path /path/to/your/gt_big.tif \
    --patch_size 1024 1024 \
    --stride 512 512 \
    --rgb

如果你仍然希望像 Potsdam 一样，对整个数据集目录做整图推理：
python whuoptsar_Large_test.py \
    -c config/whuoptsar/2_modality/your_config.py \
    -o fig_results/whuoptsar/freASam_Large_image \
    --data_root /path/to/WHU-OPT-SAR/full_images_root \
    --patch_size 1024 1024 \
    --stride 512 512 \
    --rgb

注意：
- 单图模式下，必须显式给出 --img_path 与 --aux2_path；
- 如果提供 --mask_path，就会计算整图指标；不提供则只保存预测结果，不算指标；
- data_root 模式沿用原来逻辑：假设有 rgb/、aux2(or dsm)/、label/ 子目录，文件名（不含扩展名）一一对应。
"""

import os
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import albumentations as albu
import ttach as tta

# 直接复用你 train 文件里的对象
from train_for_2_modality import Supervision_Train, py2cfg, seed_everything, Evaluator


# ------------------ 工具函数 ------------------ #

def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, required=True, help="Path to config .py")
    arg("-o", "--output_path", type=Path, required=True, help="Path to save resulting masks")

    # 方式一：整套数据集（与 Potsdam 相同逻辑）
    arg("--data_root", type=Path, default=None,
        help="Root dir of full images (rgb/aux2/label subdirs). "
             "If None and --img_path is also None, will try config.data_root or config.test_data_root.")

    # 方式二：单张大图推理（WHU-OPT-SAR 推荐用这个）
    arg("--img_path", type=Path, default=None,
        help="Path to a single RGB/optical big image for large-tile inference.")
    arg("--aux2_path", type=Path, default=None,
        help="Path to the corresponding aux2 image (e.g., SAR) of the single big image.")
    arg("--mask_path", type=Path, default=None,
        help="(Optional) Path to the GT mask of the single big image (for metrics).")

    arg("--patch_size", type=int, nargs=2, default=[1024, 1024],
        help="Sliding window patch size: H W")
    arg("--stride", type=int, nargs=2, default=None,
        help="Sliding window stride: H W (default: same as patch_size)")
    arg("-t", "--tta", help="Test time augmentation.", default=None, choices=[None, "d4", "lr"])
    arg("--rgb", help="Whether output RGB masks (label2rgb)", action='store_true')
    # 新增：是否保存每个滑窗 patch 的预测结果
    arg("--save_patches", help="Also save per-patch prediction masks.", action='store_true')
    return parser.parse_args()


def label2rgb(mask):
    """
    WHU-OPT-SAR 7 类调色板（你给的是 BGR，这里内部用 RGB，保存时再转 BGR 给 cv2）：

    类别索引：
      0: farmland
      1: city
      2: village
      3: water
      4: forest
      5: road
      6: others

    你给的配色（B, G, R）：
      0: [0,   102, 204]  farmland
      1: [0,   0,   255]  city
      2: [0,   255, 255]  village
      3: [255, 0,   0]    water
      4: [0,   167, 85]   forest
      5: [255, 255, 0]    road
      6: [153, 102, 153]  others

    这里存成 RGB：
      0: (204, 102,   0)
      1: (255,   0,   0)
      2: (255, 255,   0)
      3: (  0,   0, 255)
      4: ( 85, 167,   0)
      5: (  0, 255, 255)
      6: (153, 102, 153)
    """
    h, w = mask.shape
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    m = mask[np.newaxis, :, :]

    # farmland = 0, BGR=(0,102,204) -> RGB=(204,102,0)
    mask_rgb[np.all(m == 0, axis=0)] = (204, 102, 0)
    # city = 1, BGR=(0,0,255) -> RGB=(255,0,0)
    mask_rgb[np.all(m == 1, axis=0)] = (255, 0, 0)
    # village = 2, BGR=(0,255,255) -> RGB=(255,255,0)
    mask_rgb[np.all(m == 2, axis=0)] = (255, 255, 0)
    # water = 3, BGR=(255,0,0) -> RGB=(0,0,255)
    mask_rgb[np.all(m == 3, axis=0)] = (0, 0, 255)
    # forest = 4, BGR=(0,167,85) -> RGB=(85,167,0)
    mask_rgb[np.all(m == 4, axis=0)] = (85, 167, 0)
    # road = 5, BGR=(255,255,0) -> RGB=(0,255,255)
    mask_rgb[np.all(m == 5, axis=0)] = (0, 255, 255)
    # others = 6, BGR=(153,102,153) -> RGB=(153,102,153)
    mask_rgb[np.all(m == 6, axis=0)] = (153, 102, 153)

    return mask_rgb


def get_image_ids(data_root: Path, rgb_dir: str = "rgb"):
    """
    从 rgb 目录读取所有大图文件名（不含扩展名）并排序。
    """
    rgb_path = data_root / rgb_dir
    img_ids = sorted([p.stem for p in rgb_path.iterdir() if p.is_file()])
    return img_ids


def build_sliding_grid(h, w, patch_h, patch_w, stride_h, stride_w):
    """
    给定整图大小 (h,w) 和 patch/stride，生成所有窗口左上角坐标 (y,x)。
    末尾一行/一列不整除时，将最后一个 patch 贴到图像下/右边界。
    """
    ys = list(range(0, max(h - patch_h + 1, 1), stride_h))
    xs = list(range(0, max(w - patch_w + 1, 1), stride_w))
    if ys[-1] != h - patch_h:
        ys.append(max(h - patch_h, 0))
    if xs[-1] != w - patch_w:
        xs.append(max(w - patch_w, 0))
    grid = [(y, x) for y in ys for x in xs]
    return grid


def normalize_patch_pair(rgb_patch, aux2_patch):
    """
    使用 Albumentations 的 Normalize 对 RGB 和 aux2 逐通道做相同归一化。
    这里简单用默认 mean/std（和你 val_aug 一致：albu.Normalize()）。
    """
    norm = albu.Normalize()
    rgb = norm(image=rgb_patch)['image']
    aux2 = norm(image=aux2_patch)['image']
    return rgb, aux2


def infer_one_full_image(model,
                         rgb_path: Path,
                         aux2_path: Path,
                         mask_path: Path,
                         num_classes: int,
                         patch_size,
                         stride,
                         device,
                         patch_out_dir: Path = None,
                         save_patches: bool = False,
                         save_patches_rgb: bool = False):
    """
    对一张整图做滑窗推理并拼回整图：
    - model: 已经 .eval() 并搬到 device 的模型（可以是 TTA wrapper）
    - 返回：pred_mask (H,W)，gt_mask (H,W or None)
    - 若 save_patches=True，则在 patch_out_dir 下保存每个 patch 的预测结果
    """
    # 读取整图
    rgb = np.array(Image.open(rgb_path).convert('RGB'))  # H,W,3
    aux2 = np.array(Image.open(aux2_path).convert('RGB'))

    h, w, _ = rgb.shape
    ph, pw = patch_size
    sh, sw = stride

    # 初始化 logits 累加器 & 计数器 (在 CPU 上)
    logits_sum = torch.zeros(num_classes, h, w, dtype=torch.float32)
    count = torch.zeros(h, w, dtype=torch.float32)

    grid = build_sliding_grid(h, w, ph, pw, sh, sw)

    # 若要保存 patch，确保目录存在
    if save_patches and patch_out_dir is not None:
        patch_out_dir.mkdir(parents=True, exist_ok=True)

    img_id = rgb_path.stem

    for (y, x) in grid:
        rgb_patch = rgb[y:y + ph, x:x + pw, :]
        aux2_patch = aux2[y:y + ph, x:x + pw, :]

        # 做和验证阶段一致的 Normalize
        rgb_norm, aux2_norm = normalize_patch_pair(rgb_patch, aux2_patch)

        # 转 tensor，搬到 GPU
        rgb_t = torch.from_numpy(rgb_norm).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
        aux2_t = torch.from_numpy(aux2_norm).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)

        with torch.no_grad():
            out = model(rgb_t, aux2_t)  # 兼容 TTA wrapper

            # 兼容含 aux loss 的结构
            if isinstance(out, (list, tuple)):
                logits = out[0]  # B,C,H,W（主输出）
            else:
                logits = out

            # softmax 可以放在最后整图一起做，这里直接累加 logits 即可
            logits = logits.squeeze(0).detach().cpu()  # C,H,W

        # 累加到整图 logits
        logits_sum[:, y:y + ph, x:x + pw] += logits
        count[y:y + ph, x:x + pw] += 1.0

        # 同时保存当前 patch 的预测结果（可选）
        if save_patches and patch_out_dir is not None:
            patch_pred = logits.argmax(dim=0).numpy().astype(np.uint8)  # H,W
            patch_name = patch_out_dir / f"{img_id}_y{y}_x{x}.png"
            if save_patches_rgb:
                patch_rgb = label2rgb(patch_pred)  # RGB
                # cv2 需要 BGR

