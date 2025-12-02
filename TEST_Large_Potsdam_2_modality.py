# -*- coding: utf-8 -*-
"""
Potsdam 整图滑窗推理 + 拼回整图

使用方式示例：
python potsdam_Large_test.py \
    -c config/potsdam/2_modality/old_freASam_tiny.py \
    -o fig_results/potsdam/freASam_Large_image \
    --data_root H:/Datasets/Potsdam_New/Images/original_test \
    --patch_size 768 768 \
    --stride 384 384 \
    --rgb \
    --save_patches

注意：
- 默认假设 data_root 下结构为：
    rgb/   *.tif
    dsm/   *.jpg  (作为 aux2)
    label/ *.tif  (整图的单通道标签图)
- 文件名（不含扩展名）在三个目录中一一对应。
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
from TRAIN_FreASam_for_2_modality import Supervision_Train, py2cfg, seed_everything, Evaluator


# ------------------ 工具函数 ------------------ #

def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, required=True, help="Path to config .py")
    arg("-o", "--output_path", type=Path, required=True, help="Path to save resulting masks")
    arg("--data_root", type=Path, default=None,
        help="Root dir of Potsdam full images (rgb/dsm/label subdirs). "
             "If None, will try config.data_root or config.test_data_root.")
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
    按 Potsdam 5/6 类调色板上色：
      0: 白(255,255,255) Impervious surfaces
      1: 蓝(0,0,255)     Building
      2: 青(0,255,255)   Low vegetation
      3: 绿(0,255,0)     Tree
      4: 黄(255,255,0)   Car
      5: (可选) 其他类
    """
    h, w = mask.shape
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    m = mask[np.newaxis, :, :]

    mask_rgb[np.all(m == 0, axis=0)] = (255, 255, 255)
    mask_rgb[np.all(m == 1, axis=0)] = (0,   0,   255)
    mask_rgb[np.all(m == 2, axis=0)] = (0, 255,   255)
    mask_rgb[np.all(m == 3, axis=0)] = (0,   255,   0)
    mask_rgb[np.all(m == 4, axis=0)] = (255, 255,   0)
    # 如果你有第 5 类，可以在这里再加一行
    # mask_rgb[np.all(m == 5, axis=0)] = (255,   0,   0)
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
                cv2.imwrite(str(patch_name), patch_rgb[:, :, ::-1])
            else:
                cv2.imwrite(str(patch_name), patch_pred)

    # 防止除 0（理论上 count>0）
    count = count.clamp(min=1.0)
    logits_avg = logits_sum / count.unsqueeze(0)

    # 整图预测
    pred_mask = logits_avg.argmax(dim=0).numpy().astype(np.uint8)

    # 读 GT（如果有）
    gt_mask = None
    if mask_path is not None and mask_path.exists():
        gt_mask = np.array(Image.open(mask_path).convert('L'))

    return pred_mask, gt_mask


# ------------------ 主流程 ------------------ #

def main():
    args = get_args()
    seed_everything(42)

    config = py2cfg(args.config_path)

    # 输出目录
    args.output_path.mkdir(parents=True, exist_ok=True)

    # data_root & 子目录名
    if args.data_root is not None:
        data_root = args.data_root
    else:
        # 优先用 config.test_data_root，其次 config.data_root，最后手动填默认
        dr = getattr(config, "test_data_root", None)
        if dr is None:
            dr = getattr(config, "data_root", None)
        if dr is None:
            raise ValueError("请通过 --data_root 或 config.test_data_root / config.data_root 指定 Potsdam 数据根目录")
        data_root = Path(dr)

    rgb_dir = getattr(config, "rgb_dir", "rgb")
    aux2_dir = getattr(config, "aux2_dir", getattr(config, "dsm_dir", "dsm"))
    mask_dir = getattr(config, "mask_dir", "label")

    patch_size = tuple(args.patch_size)
    if args.stride is None:
        stride = patch_size
    else:
        stride = tuple(args.stride)

    print(f"[Data] root={data_root}, rgb_dir={rgb_dir}, aux2_dir={aux2_dir}, mask_dir={mask_dir}")
    print(f"[Slide] patch_size={patch_size}, stride={stride}")
    print(f"[Save patches] {args.save_patches}")

    # --------- 构建模型 & 加载权重 --------- #
    ckpt_path = os.path.join(config.weights_path, config.test_weights_name + '.ckpt')
    print(f"[Model] loading checkpoint: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    model = Supervision_Train(config=config)
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    model.freeze()

    # --------- TTA 设置（保持和原 test 一致） --------- #
    if args.tta == "lr":
        transforms = tta.Compose([
            tta.HorizontalFlip(),
            tta.VerticalFlip()
        ])
        model = tta.SegmentationTTAWrapper(model, transforms)
        print("[TTA] lr (flip) enabled")
    elif args.tta == "d4":
        transforms = tta.Compose([
            tta.HorizontalFlip(),
            tta.VerticalFlip(),
            tta.Rotate90(angles=[90]),
            # 注意：缩放 TTA 在滑窗上会比较诡异，我这里暂时不加 Scale
        ])
        model = tta.SegmentationTTAWrapper(model, transforms)
        print("[TTA] d4 (flip + rotate90) enabled")
    else:
        print("[TTA] disabled")

    # --------- 遍历所有大图，整图滑窗推理 --------- #
    img_ids = get_image_ids(data_root, rgb_dir=rgb_dir)
    print(f"[Data] found {len(img_ids)} full images")

    evaluator = Evaluator(num_class=config.num_classes)
    evaluator.reset()

    # 若保存 patch，则统一放在 output_path/patches/<img_id> 下面
    patches_root = args.output_path / "patches" if args.save_patches else None

    for img_id in tqdm(img_ids, desc="Sliding-window inference on full images"):
        rgb_path = data_root / rgb_dir / f"{img_id}.tif"
        aux2_path = data_root / aux2_dir / f"{img_id}.jpg"
        mask_path = data_root / mask_dir / f"{img_id}.tif"

        if not rgb_path.exists():
            print(f"  [Warn] RGB not found: {rgb_path}, skip")
            continue
        if not aux2_path.exists():
            print(f"  [Warn] aux2 not found: {aux2_path}, skip")
            continue

        # 每张大图的 patch 输出子目录
        if patches_root is not None:
            patch_out_dir = patches_root / img_id
        else:
            patch_out_dir = None

        pred_mask, gt_mask = infer_one_full_image(
            model=model,
            rgb_path=rgb_path,
            aux2_path=aux2_path,
            mask_path=mask_path,
            num_classes=config.num_classes,
            patch_size=patch_size,
            stride=stride,
            device=device,
            patch_out_dir=patch_out_dir,
            save_patches=args.save_patches,
            save_patches_rgb=args.rgb
        )

        # 评估（如果有 GT）
        if gt_mask is not None:
            evaluator.add_batch(pre_image=pred_mask, gt_image=gt_mask)

            # 写整图预测
        out_name = args.output_path / f"{img_id}.png"
        if args.rgb:
            out_img = label2rgb(pred_mask)  # RGB
            cv2.imwrite(str(out_name), out_img[:, :, ::-1])  # cv2 需要 BGR
        else:
            out_img = pred_mask.astype(np.uint8)
            cv2.imwrite(str(out_name), out_img)

    # --------- 计算整套大图的指标 --------- #
    iou_per_class = evaluator.Intersection_over_Union()
    f1_per_class = evaluator.F1()

    # ========= 关键改动：去掉最后一类参与 per-class 评价 =========
    # 假设最后一类是 clutter/background，不参与我们关心的类别评价
    # iou_valid = iou_per_class[:-1]
    # f1_valid = f1_per_class[:-1]

    iou_valid = iou_per_class
    f1_valid = f1_per_class

    pa_per_class = evaluator.Pixel_Accuracy_Class()
    oa_cls_binary = evaluator.OA_per_class()

    # pa_valid = pa_per_class[:-1]
    # oa_cls_valid = oa_cls_binary[:-1]

    pa_valid = pa_per_class
    oa_cls_valid = oa_cls_binary

    OA = evaluator.OA()
    mIoU_macro_all = evaluator.mIoU_macro()
    mF1_macro_all = evaluator.mF1_macro()
    mKappa = evaluator.mKappa()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()

    # 自己再算一份 “去掉最后一类后的 macro”
    mIoU_macro_valid = float(np.nanmean(iou_valid))
    mF1_macro_valid = float(np.nanmean(f1_valid))

    print("========== Metrics on full images (ignore last class in per-class stats) ==========")
    for cls_name, cls_iou, cls_f1, cls_pa, cls_oabin in zip(
            config.classes[:-1], iou_valid, f1_valid, pa_valid, oa_cls_valid):
        print(
            f'F1_{cls_name}:{cls_f1:.6f}, IOU_{cls_name}:{cls_iou:.6f}, '
            f'PA_{cls_name}:{cls_pa:.6f}, OAcls_{cls_name}:{cls_oabin:.6f}'
        )

    print(
        f'mF1_macro(valid,no_last):{mF1_macro_valid:.6f}, '
        f'mIoU_macro(valid,no_last):{mIoU_macro_valid:.6f}'
    )
    print(
        f'mF1_macro(all):{mF1_macro_all:.6f}, mIoU_macro(all):{mIoU_macro_all:.6f}, '
        f'OA:{OA:.6f}, mKappa:{mKappa:.6f}, FWIoU:{FWIoU:.6f}'
    )


if __name__ == "__main__":
    main()

