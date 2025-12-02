# filename: batch_label2rgb.py
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

# ------------------------------
# 把你给的 B,G,R 规则转换为 RGB
# 类别: 0 farmland, 1 city, 2 village, 3 water, 4 forest, 5 road, 6 others
# BGR -> RGB:
# [0,102,204]->[204,102,0], [0,0,255]->[255,0,0], [0,255,255]->[255,255,0]
# [255,0,0]->[0,0,255], [0,167,85]->[85,167,0], [255,255,0]->[0,255,255], [153,102,153]->不变
# ------------------------------
RGB_TABLE = np.zeros((256, 3), dtype=np.uint8)
RGB_TABLE[0] = [204, 102,   0]   # farmland
RGB_TABLE[1] = [255,   0,   0]   # city
RGB_TABLE[2] = [255, 255,   0]   # village
RGB_TABLE[3] = [  0,   0, 255]   # water
RGB_TABLE[4] = [ 85, 167,   0]   # forest
RGB_TABLE[5] = [  0, 255, 255]   # road
RGB_TABLE[6] = [153, 102, 153]   # others

def label_to_rgb(label_arr: np.ndarray,
                 ignore_value: int | None = None,
                 ignore_color_rgb=(0, 0, 0)) -> np.ndarray:
    """
    label_arr: HxW 或 HxW x 1 的单通道标签数组 (0..6)
    返回: HxW x 3 的 RGB uint8 彩图
    """
    lab = np.asarray(label_arr)
    if lab.ndim == 3:
        lab = lab[..., 0]
    if lab.dtype != np.uint8:
        lab = lab.astype(np.uint8)

    rgb = RGB_TABLE[lab]  # 直接查表映射成 RGB

    if ignore_value is not None:
        m = (lab == np.uint8(ignore_value))
        if m.any():
            rgb = rgb.copy()
            rgb[m] = np.array(ignore_color_rgb, dtype=np.uint8)
    return rgb

def process_dir(in_dir: Path,
                out_dir: Path,
                suffixes=(".png", ".tif", ".tiff"),
                recursive=True,
                ignore_value: int | None = None,
                ignore_color_rgb=(0, 0, 0),
                print_first_n: int = 5):
    out_dir.mkdir(parents=True, exist_ok=True)
    files = in_dir.rglob("*") if recursive else in_dir.iterdir()
    n = 0
    shown = 0

    for p in files:
        if p.is_file() and p.suffix.lower() in suffixes:
            arr = np.array(Image.open(p))
            uniq = np.unique(arr)
            rgb = label_to_rgb(arr, ignore_value=ignore_value, ignore_color_rgb=ignore_color_rgb)

            # 保留子目录结构与文件名，写到 out_dir
            out_path = out_dir / p.relative_to(in_dir)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb, mode="RGB").save(out_path.with_suffix(p.suffix))
            n += 1

            if shown < print_first_n:
                print(f"[CHECK] {p}: unique(labels)={uniq.tolist()}")
                shown += 1

    print(f"Done. {n} file(s) saved to: {out_dir}")

def main():
    parser = argparse.ArgumentParser(description="批量将单通道 label 映射为 RGB 彩色掩膜并保存（按 RGB 通道写出）")
    parser.add_argument("--in-dir", required=True, help="输入标签文件夹（单通道 0..6）")
    parser.add_argument("--out-dir", required=True, help="输出彩色结果文件夹")
    parser.add_argument("--ext", nargs="+", default=[".png", ".tif", ".tiff"], help="处理的文件后缀")
    parser.add_argument("--non-recursive", action="store_true", help="不递归子目录（默认会递归）")
    parser.add_argument("--ignore-value", type=int, default=None, help="忽略像素值，如 255；不需要则不填")
    parser.add_argument("--ignore-color", type=str, default=None, help="忽略像素可视化颜色 'R,G,B'（默认黑色）")
    parser.add_argument("--print-first-n", type=int, default=5, help="前 N 张打印唯一值用于自检")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    recursive = not args.non_recursive

    ignore_color_rgb = (0, 0, 0)
    if args.ignore_color is not None:
        parts = [int(x) for x in args.ignore_color.split(",")]
        if len(parts) != 3:
            raise ValueError("ignore 颜色需为 'R,G,B' 三个整数，用逗号分隔")
        ignore_color_rgb = tuple(np.clip(parts, 0, 255).astype(np.uint8))

    process_dir(
        in_dir=in_dir,
        out_dir=out_dir,
        suffixes=tuple([s.lower() for s in args.ext]),
        recursive=recursive,
        ignore_value=args.ignore_value,
        ignore_color_rgb=ignore_color_rgb,
        print_first_n=args.print_first_n
    )

if __name__ == "__main__":
    # —— 方式 A：命令行（推荐）——
    # 在 PyCharm 的 Run/Debug Configurations 里添加参数，例如：
    # --in-dir D:\data\labels --out-dir D:\data\labels_rgb
    # main()

    # —— 方式 B：直接在代码里写死路径（如不想传参，可把上面的 main() 注释掉，改用下面 4 行）——
    process_dir(
         in_dir=Path(r"H:\Datasets\WHUOPTSAR\data_512\data\test\label"),
         out_dir=Path(r"H:\Datasets\WHUOPTSAR\data_512\data\test\label_rgb"),
    )
