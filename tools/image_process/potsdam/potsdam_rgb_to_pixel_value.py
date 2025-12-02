# filename: potsdam_rgb_to_pixel_value.py
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

IGNORE_VALUE = 255

# ---------------------------
# ISPRS Potsdam 官方色标（RGB）
# ---------------------------
# Impervious surfaces: (255,255,255) -> 0
# Building:            (0,0,255)     -> 1
# Low vegetation:      (0,255,255)   -> 2
# Tree:                (0,255,0)     -> 3
# Car:                 (255,255,0)   -> 4
# Clutter/background:  (255,0,0)     -> 5   ✅ 改为有效类别 5
#
# 额外忽略:
# Pure black:          (0,0,0)       -> 255 ✅ 作为 ignore

# 像素值定义（0..5），Clutter 映射为 5
# 像素值定义（0..5），Clutter 映射为 5
RGB2PIXEL = {
    (255, 255, 255): 0,  # impervious
    (0,   0,   255): 1,  # building
    (0,   255, 255): 2,  # low vegetation
    (0,   255, 0  ): 3,  # tree
    (255, 255, 0  ): 4,  # car
    (252, 255, 0  ): 4,  # ✅ 新增：近似黄色也算 car
    # (255, 0,   0  ): 5   # clutter
}


# 需要直接忽略为 255 的颜色
EXTRA_IGNORE_COLORS = { (0,0,0) }

def rgb_tuple_to_code(rgb):
    """把 (R,G,B) 压成 24bit 整数，便于向量化比较。"""
    r, g, b = rgb
    return (np.uint32(r) << 16) | (np.uint32(g) << 8) | np.uint32(b)

def make_color_code_maps():
    """准备颜色编码集合与映射。"""
    known_map_codes = {rgb_tuple_to_code(k): v for k, v in RGB2PIXEL.items()}
    ignore_codes = set()
    if EXTRA_IGNORE_COLORS:
        ignore_codes |= {rgb_tuple_to_code(c) for c in EXTRA_IGNORE_COLORS}
    return known_map_codes, ignore_codes

def rgb_mask_to_ids(arr_rgb: np.ndarray,
                    ignore_value: int = IGNORE_VALUE) -> np.ndarray:
    """
    arr_rgb: HxW x 3 (uint8) 的 RGB 彩色标签图
    return:  HxW (uint8)，像素值 {0..5, 255}
    """
    if arr_rgb.ndim != 3 or arr_rgb.shape[2] != 3:
        raise ValueError("输入必须是 RGB 彩色标签图，形状应为 HxW x 3")

    H, W, _ = arr_rgb.shape
    # 压成 24bit 颜色编码
    codes = (arr_rgb[..., 0].astype(np.uint32) << 16) | \
            (arr_rgb[..., 1].astype(np.uint32) << 8)  | \
             arr_rgb[..., 2].astype(np.uint32)

    out = np.full((H, W), ignore_value, dtype=np.uint8)

    known_map, ignore_codes = make_color_code_maps()

    # 逐类赋值（6 类，含 clutter=5）
    for code_val, cls_id in known_map.items():
        mask = (codes == code_val)
        if mask.any():
            out[mask] = np.uint8(cls_id)

    # 显式忽略（如纯黑 0,0,0）
    if ignore_codes:
        mask_ignore = np.isin(codes, np.fromiter(ignore_codes, dtype=np.uint32))
        if mask_ignore.any():
            out[mask_ignore] = np.uint8(ignore_value)

    # 统计未知颜色（既不在 6 类、也不在忽略列表）
    known_and_ign = set(known_map.keys()) | set(ignore_codes)
    unknown_mask = ~np.isin(codes, np.fromiter(known_and_ign, dtype=np.uint32))
    if unknown_mask.any():
        # 取若干示例打印出来，便于你发现异常像素
        unknown_codes = np.unique(codes[unknown_mask])
        examples = unknown_codes[:10]
        def code_to_rgb(c):
            r = (c >> 16) & 255
            g = (c >> 8) & 255
            b = c & 255
            return int(r), int(g), int(b)
        print("[WARN] 检测到未知颜色（将被置为 255）：",
              [code_to_rgb(int(c)) for c in examples])

    return out

def process_folder(in_dir,
                   out_dir,
                   suffixes=(".png", ".tif", ".tiff", ".jpg", ".jpeg"),
                   recursive=True,
                   print_first_n=5):
    """
    批量把 Potsdam 的 RGB 标签图转为单通道像素值图（0..5, 255）。
    会保留原有的子目录结构。
    """
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = in_dir.rglob("*") if recursive else in_dir.iterdir()
    shown = 0
    count = 0
    for p in files:
        if p.is_file() and p.suffix.lower() in suffixes:
            # 用 PIL 强制转为 RGB
            img = Image.open(p).convert("RGB")
            arr = np.array(img, dtype=np.uint8)

            ids = rgb_mask_to_ids(arr, ignore_value=IGNORE_VALUE)

            # 保持子目录结构与扩展名
            out_path = out_dir / p.relative_to(in_dir)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(ids, mode="L").save(out_path.with_suffix(p.suffix))
            count += 1

            if shown < print_first_n:
                uniq = np.unique(ids)
                print(f"[CHECK] {p}: unique(ids)={uniq.tolist()}")
                shown += 1

    print(f"Done. {count} file(s) saved to: {out_dir}")

def main():
    parser = argparse.ArgumentParser(
        description="将 ISPRS Potsdam RGB 标签批量映射为像素值（0..5），Clutter→5，纯黑(0,0,0)→255(ignore）"
    )
    parser.add_argument("--in-dir", required=True, help="输入 RGB 标签文件夹")
    parser.add_argument("--out-dir", required=True, help="输出 单通道像素值 标签文件夹")
    parser.add_argument("--ext", nargs="+",
                        default=[".png", ".tif", ".tiff", ".jpg", ".jpeg"],
                        help="处理的文件后缀")
    parser.add_argument("--non-recursive", action="store_true",
                        help="不递归子目录（默认递归）")
    parser.add_argument("--print-first-n", type=int, default=5,
                        help="前 N 张打印唯一值用于自检")
    args = parser.parse_args()

    process_folder(
        in_dir=args.in_dir,
        out_dir=args.out_dir,
        suffixes=tuple([s.lower() for s in args.ext]),
        recursive=not args.non_recursive,
        print_first_n=args.print_first_n
    )

if __name__ == "__main__":
    main()

# ---------------------------
# 使用示例
# --in-dir D:\data\potsdam\labels_rgb --out-dir D:\data\potsdam\labels_id
# ---------------------------
