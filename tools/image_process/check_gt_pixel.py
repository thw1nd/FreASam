import os
import glob
import numpy as np

# 可修改为你的标签目录
LABEL_DIR = r"H:\Datasets\WHUOPTSAR\data_512\data\aug_train\label"

# 支持的后缀
EXTS = ("*.tif", "*.tiff", "*.png")

# 允许的类别集合
ALLOWED = set(range(8))  # {0,1,2,3,4,5}

def read_label(path):
    """
    读取单通道标签为 [H,W] 的 numpy 数组（int64）。
    优先尝试 rasterio；若失败则退回 imageio（应付 PNG 等）。
    """
    try:
        import rasterio
        with rasterio.open(path) as src:
            arr = src.read(1)  # 读第一通道
    except Exception:
        # 兼容非tif的情况
        import imageio.v2 as imageio
        arr = imageio.imread(path)
        if arr.ndim == 3:
            arr = arr[..., 0]
    # 统一为整数类型
    if arr.dtype.kind == "f":
        # 浮点标签：尝试就地四舍五入到最近整数（也可改成 np.floor / np.ceil）
        arr = np.rint(arr)
    arr = arr.astype(np.int64, copy=False)
    return arr

def main():
    # 收集文件
    files = []
    for ext in EXTS:
        files.extend(glob.glob(os.path.join(LABEL_DIR, ext)))
    files = sorted(files)

    if not files:
        print(f"未在 {LABEL_DIR} 下找到标签文件（支持后缀：{EXTS}）。")
        return

    all_ok = True
    for fp in files:
        arr = read_label(fp)
        # 去掉可能的无效值（按需可屏蔽 NoData；这里不做额外屏蔽）
        uniq = np.unique(arr)
        uniq_list = uniq.tolist()
        print(f"{os.path.basename(fp)} 唯一像素值: {uniq_list}")

        # 检查越界类别
        bad = set(uniq_list) - ALLOWED
        if bad:
            all_ok = False
            print(f"图像 {os.path.basename(fp)} 不属于：发现异常值 {sorted(bad)}")

    if all_ok:
        print("检查完成：所有标签像素值均在 {0,1,2,3,4,5,6} 范围内 ✅")
    else:
        print("检查完成：存在不在允许集合内的像素值，请上方逐条查看 ❗")

if __name__ == "__main__":
    main()
