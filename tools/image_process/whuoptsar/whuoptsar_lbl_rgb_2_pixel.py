import os
from pathlib import Path
import numpy as np
from PIL import Image

# 将原始标签中的 10..70 映射到 0..6，背景 0 不在映射表里 → 会被映射成 255(ignore)
LABEL_MAP = {10: 0, 20: 1, 30: 2, 40: 3, 50: 4, 60: 5, 70: 6}
IGNORE_VALUE = 255
LUT_SIZE = 256  # 固定 8-bit 查找表，安全且快速

def build_lut(label_map=LABEL_MAP, ignore_value: int = IGNORE_VALUE) -> np.ndarray:
    """
    构造 0..255 的查找表：
      - 默认值 = ignore(255)
      - 指定的键（10..70）映射到 0..6
      - 关键点：0 不在映射表内 → 0 会保持为 255(忽略)
    """
    lut = np.full(LUT_SIZE, ignore_value, dtype=np.uint8)
    for k, v in label_map.items():
        if 0 <= k < LUT_SIZE:
            lut[k] = v
    # lut[0] 仍为 255（忽略）
    return lut

def remap_labels_np(mask: np.ndarray,
                    label_map=LABEL_MAP,
                    ignore_value: int = IGNORE_VALUE,
                    lut: np.ndarray = None) -> np.ndarray:
    """
    mask: HxW (uint8/uint16/uint32 均可，但值域须在 0..255 内)
    return: HxW (uint8), 取值集合为 {0..6, 255}
    """
    mask = np.asarray(mask)
    if lut is None:
        lut = build_lut(label_map, ignore_value)
    out = lut[mask]  # 依据 0..255 查找
    return out

def remap_folder(in_dir, out_dir, suffixes=(".png", ".tif", ".tiff"),
                 print_first_n: int = 5, save_debug_vis: bool = False):
    """
    批量映射，并在前几张上打印唯一值用于自检。
    若 save_debug_vis=True，会额外导出 *_debug.png（忽略像素=255显示为白色）。
    """
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lut = build_lut(LABEL_MAP, IGNORE_VALUE)

    shown = 0
    for p in in_dir.rglob("*"):
        if p.suffix.lower() in suffixes:
            arr = np.array(Image.open(p))
            out = remap_labels_np(arr, lut=lut)
            # 保存映射结果（单通道 L）
            out_path = out_dir / p.name
            Image.fromarray(out, mode="L").save(out_path)

            # 前几张做唯一值检查（不要用颜色判断，直接看数值）
            if shown < print_first_n:
                uniq = np.unique(out)
                print(f"[CHECK] {p.name}: unique={uniq.tolist()}")
                shown += 1

            # 可选：调试可视化（把 255 用白色显示，便于肉眼确认是否存在忽略像素）
            if save_debug_vis:
                vis = out.copy()
                # 255 已经是 255，这里只是明确一下逻辑
                vis[out == IGNORE_VALUE] = 255
                Image.fromarray(vis, mode="L").save(out_path.with_name(p.stem + "_debug.png"))

# --- 用法示例 ---
# 单张：
# arr = np.array(Image.open("raw_label.png"))
# mapped = remap_labels_np(arr)  # 结果仅含 {0..6, 255}
# Image.fromarray(mapped, "L").save("mapped_label.png")

# 批量：
original_dir = r'H:\Datasets\WHUOPTSAR\original_images\lbl_rgb'
ouput_dir = r'H:\Datasets\WHUOPTSAR\original_images\new_lbl'
remap_folder(original_dir, ouput_dir, print_first_n=5, save_debug_vis=False)
