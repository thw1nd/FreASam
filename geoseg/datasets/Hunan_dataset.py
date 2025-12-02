# -*- coding: utf-8 -*-
"""
Hunan_dataset.py  —— 综合改进版

主要特性：
1) 使用 rasterio 读取 GeoTIFF，稳健支持 S1/S2/DEM/TOPO/TRI 等单/多通道与浮点栅格；
2) 统一把除标签外的所有模态强制转成 3 通道（形状适配，便于后续增强/模型）；
3) 灵活的文件名前缀映射：默认使用“目录名_”作为前缀，支持从 rgb_dir 扫描时剥除前缀；
4) transform 兼容两种风格：
   - Albumentations.Compose（使用关键字 image/aux1/aux2/mask 调用）；
   - 普通函数 transform(img, aux1, aux2, mask) -> (img, aux1, aux2, mask)；
5) 内置几何增强构造器（只做几何，保证多模态对齐）；示例提供恒等 Normalize。

返回字典：
{
    'img': Tensor[3,H,W],   # S2 或主分支（已强制 3 通道）
    'aux1': Tensor[3,H,W],  # S1（1->3 复制）
    'aux2': Tensor[3,H,W],  # TOPO/DEM/TRI（1->3 复制）
    'gt_semantic_seg': LongTensor[H,W],
    'img_id': str
}

注意：把单通道强制 3 通道只是形状适配，不会增加信息。更科学做法是保留原通道数，或在模型第一层用 1×1 卷积适配。
"""

CLASSES = ('cropland', 'forest', 'grassland', 'wetland', 'water', 'unused land', 'built-up area')

PALETTE = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0],
           [255, 255, 0], [255, 0, 0], [255, 0, 255]]
import os
import os.path as osp
from typing import List, Optional, Tuple, Sequence, Dict
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset

import rasterio
import albumentations as A

# 可选：屏蔽 rasterio 的未地理参考提醒（对训练无影响）
from rasterio.errors import NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


IGBP2HUNAN = np.array([255, 0, 1, 2, 1, 3, 4, 6, 6, 5, 6, 7, 255], dtype=np.int64)
ALLOWED_LABEL_SET = set(range(7)) | {255}  # {0..6, 255}

def remap_label_to_0_6_255(lc: np.ndarray) -> np.ndarray:
    """
    将原始 IGBP 标签重映射为 {0..6, 255}：
      1) 先把 255 替换成 12；2) 用 IGBP2HUNAN 查表映射；
      3) 把 7 合并到 6；最终仅保留 {0..6,255}
    """
    lc = lc.copy()
    lc[lc == 255] = 12
    lc = IGBP2HUNAN[lc]
    # 兜底：有些瓦片会出现 7（作者映射里保留了 7），合并为 6，确保只有 0..6/255
    lc[lc == 7] = 6
    return lc.astype(np.int64, copy=False)


# =========================
# 工具函数
# =========================

def read_geotiff_as_hwc(path: str, band_indices: Optional[Sequence[int]] = None) -> np.ndarray:
    """
    读取 GeoTIFF -> [H, W, C] np.float32
    band_indices: 选择波段（1-based）；None = 读全部波段
    """
    with rasterio.open(path) as src:
        if band_indices is None:
            arr = src.read()  # [C, H, W]
        else:
            arr = src.read(band_indices)  # [C, H, W]
    arr = np.transpose(arr, (1, 2, 0)).astype(np.float32)  # -> [H, W, C]
    if arr.ndim == 2:
        arr = arr[..., None]
    return arr


def read_mask_as_hw(path: str) -> np.ndarray:
    """读取语义标签 -> [H, W] np.int64（保持原类别编号）"""
    with rasterio.open(path) as src:
        mask = src.read(1)

    # 若是浮点，先取最近整数；然后转 int64
    if mask.dtype.kind == "f":
        mask = np.rint(mask).astype(np.int64)
    else:
        mask = mask.astype(np.int64)

    # ★ 关键：做 IGBP -> Hunan7 映射，并把 7 并入 6
    mask = remap_label_to_0_6_255(mask)
    return mask


def ensure_3ch(arr: np.ndarray, select_idx: Tuple[int, int, int] = (0, 1, 2), fill_value: float = 0.0) -> np.ndarray:
    """
    把 [H,W,C] 强制转成 [H,W,3]
    - C==1: 复制三份
    - C==2: 补一个常数通道
    - C>=3: 选取指定的三个通道（默认取前 3）
    - 若输入为 [H,W]，先扩成 [H,W,1]
    """
    if arr.ndim == 2:
        arr = arr[..., None]
    C = arr.shape[2]
    if C == 3:
        return arr
    if C == 1:
        return np.repeat(arr, 3, axis=2)
    if C == 2:
        pad = np.full_like(arr[..., :1], fill_value)
        return np.concatenate([arr, pad], axis=2)
    idx = np.clip(np.array(select_idx, dtype=int), 0, C - 1)
    return arr[..., idx]


# =========================
# Albumentations 几何增强（只做几何，强制对齐）
# =========================

def build_geo_transform(img_size: Optional[Tuple[int, int]] = None,
                        is_train: bool = True) -> Optional[A.Compose]:
    tfms = []
    if is_train:
        # 如需，可解开以下任意几何变换（示例默认关闭以先跑通流程）
        # tfms.append(A.HorizontalFlip(p=0.5))
        # tfms.append(A.RandomRotate90(p=0.5))
        # tfms.append(A.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.0, rotate_limit=15, border_mode=0, p=0.3))
        pass
    if img_size is not None:
        h, w = img_size
        tfms.append(A.Resize(height=h, width=w, interpolation=1))  # 1=LINEAR
    if not tfms:
        return None
    return A.Compose(tfms, additional_targets={
        'aux1': 'image',
        'aux2': 'image'
    })


# 示例：恒等 Normalize（避免把 SAR/DEM 当 RGB 归一化）

def get_training_transform():
    return A.Compose([A.Normalize(mean=(0, 0, 0), std=(1, 1, 1))])

def train_aug(img, aux1, aux2, mask):
    img, aux1, aux2, mask = np.array(img), np.array(aux1), np.array(aux2), np.array(mask)
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    aug_aux1 = get_training_transform()(image=aux1.copy())
    aux1 = aug_aux1['image']
    aug_aux2 = get_training_transform()(image=aux2.copy())
    aux2 = aug_aux2['image']
    return img, aux1, aux2, mask


def get_val_transform():
    return A.Compose([A.Normalize(mean=(0, 0, 0), std=(1, 1, 1))])

def val_aug(img, aux1, aux2, mask):
    img, aux1, aux2, mask = np.array(img), np.array(aux1), np.array(aux2), np.array(mask)
    aug = get_val_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    aug_aux1 = get_val_transform()(image=aux1.copy())
    aux1 = aug_aux1['image']
    aug_aux2 = get_val_transform()(image=aux2.copy())
    aux2 = aug_aux2['image']
    return img, aux1, aux2, mask


# =========================
# Dataset 实现
# =========================

class HunanDataset(Dataset):
    """
    目录结构示例：
        data_root/
            s2/   s2_XXXXX.tif
            s1/   s1_XXXXX.tif
            topo/ topo_XXXXX.tif   或 tri/ tri_XXXXX.tif
            lc/   lc_XXXXX.tif

    默认按“目录名_”作为各自文件名前缀；若某目录无前缀，可用 prefix_map 覆盖为 ""。
    """

    def __init__(
        self,
        data_root: str,
        rgb_dir: str = 's2',
        aux1_dir: str = 's1',
        aux2_dir: str = 'topo',            # 或 'tri'
        mask_dir: str = 'lc',
        suffix: str = '.tif',
        band_indices_s2: Optional[Sequence[int]] = None,  # 如 [2,3,4] 表示 S2 取 RGB 三波段
        img_size: Optional[Tuple[int, int]] = None,       # 若需统一尺寸，传 (H,W)
        train: bool = True,
        transform: Optional[object] = None,               # A.Compose 或 普通函数
        id_list: Optional[List[str]] = None,              # 外部指定 id 列表；否则自动扫描
        prefix_map: Optional[Dict[str, str]] = None,      # 各目录的文件名前缀；默认 "{dir}_"
        strip_rgb_prefix: bool = True,                    # 扫描 rgb_dir 时是否剥除前缀
    ):
        super().__init__()
        self.data_root = data_root
        self.rgb_dir = rgb_dir
        self.aux1_dir = aux1_dir
        self.aux2_dir = aux2_dir
        self.mask_dir = mask_dir
        self.suffix = suffix
        self.band_indices_s2 = band_indices_s2
        self.img_size = img_size
        self.train = train
        self.strip_rgb_prefix = strip_rgb_prefix

        # 前缀映射
        if prefix_map is None:
            self.prefix_map = {
                rgb_dir:  f"{rgb_dir}_",
                aux1_dir: f"{aux1_dir}_",
                aux2_dir: f"{aux2_dir}_",
                mask_dir: f"{mask_dir}_",
            }
        else:
            self.prefix_map = prefix_map

        # 扫描 id
        self.img_ids = id_list if id_list is not None else self._scan_ids()
        if not isinstance(self.img_ids, list) or len(self.img_ids) == 0:
            raise RuntimeError(f"No samples found under {osp.join(self.data_root, self.rgb_dir)} with suffix {self.suffix}")

        # transform：若未提供，则只做几何对齐（可选）
        self.transform = transform if transform is not None else build_geo_transform(img_size, is_train=train)

    # ---------- 扫描与长度 ----------
    def _scan_ids(self) -> List[str]:
        img_dir = osp.join(self.data_root, self.rgb_dir)
        if not osp.isdir(img_dir):
            raise FileNotFoundError(f"Not a directory: {img_dir}")
        rgb_prefix = self.prefix_map.get(self.rgb_dir, "")
        ids = []
        for fn in os.listdir(img_dir):
            if not fn.lower().endswith(self.suffix.lower()):
                continue
            stem = osp.splitext(fn)[0]  # 例如 "s2_11624"
            if self.strip_rgb_prefix and rgb_prefix and stem.startswith(rgb_prefix):
                stem = stem[len(rgb_prefix):]  # -> "11624"
            ids.append(stem)
        ids.sort()
        return ids

    def __len__(self) -> int:
        return len(self.img_ids)

    # ---------- 读取单样本 ----------
    def load_img_and_mask(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        img_id = self.img_ids[index]
        p_rgb = self.prefix_map.get(self.rgb_dir, "")
        p_a1  = self.prefix_map.get(self.aux1_dir, "")
        p_a2  = self.prefix_map.get(self.aux2_dir, "")
        p_msk = self.prefix_map.get(self.mask_dir, "")

        img_name = osp.join(self.data_root, self.rgb_dir,  p_rgb + img_id + self.suffix)
        a1_name  = osp.join(self.data_root, self.aux1_dir, p_a1  + img_id + self.suffix)
        a2_name  = osp.join(self.data_root, self.aux2_dir, p_a2  + img_id + self.suffix)
        msk_name = osp.join(self.data_root, self.mask_dir, p_msk + img_id + self.suffix)

        # 读取原始数组
        img  = read_geotiff_as_hwc(img_name, band_indices=self.band_indices_s2)  # 可能多波段
        aux1 = read_geotiff_as_hwc(a1_name)                                      # 常单通道
        aux2 = read_geotiff_as_hwc(a2_name)                                      # 常单通道
        mask = read_mask_as_hw(msk_name)                                         # [H,W] int64

        # 统一转为 3 通道（仅形状适配）
        # 对 S2：若 band_indices_s2 选了 >=3 个，这里默认取前三个；也可自定义 select_idx
        img  = ensure_3ch(img,  select_idx=(0, 1, 2))
        aux1 = ensure_3ch(aux1, select_idx=(0, 0, 0))
        aux2 = ensure_3ch(aux2, select_idx=(0, 0, 0))

        return img, aux1, aux2, mask

    # ---------- __getitem__ ----------
    def __getitem__(self, index: int):
        img, aux1, aux2, mask = self.load_img_and_mask(index)

        # 兼容两种 transform 形态
        if self.transform is not None:
            if isinstance(self.transform, A.Compose):
                data = self.transform(image=img, aux1=aux1, aux2=aux2, mask=mask)
                img, aux1, aux2, mask = data['image'], data['aux1'], data['aux2'], data['mask']
            else:
                # 普通函数：f(img, aux1, aux2, mask) -> (img, aux1, aux2, mask)
                img, aux1, aux2, mask = self.transform(img, aux1, aux2, mask)

        # np(HWC/HW) -> torch(CHW/HW)
        img_t  = torch.from_numpy(img).permute(2, 0, 1).contiguous().float()
        aux1_t = torch.from_numpy(aux1).permute(2, 0, 1).contiguous().float()
        aux2_t = torch.from_numpy(aux2).permute(2, 0, 1).contiguous().float()
        mask_t = torch.from_numpy(mask).long()

        return {
            'img': img_t,
            'aux1': aux1_t,
            'aux2': aux2_t,
            'gt_semantic_seg': mask_t,
            'img_id': self.img_ids[index],
        }


# =========================
# 用法示例（按需放到你的 config/脚本中）：
# =========================
# train_set = HunanDataset(
#     data_root=r'H:\Datasets\Hunan_Dataset\Hunan_Dataset\train',
#     rgb_dir='s2', aux1_dir='s1', aux2_dir='topo', mask_dir='lc',
#     suffix='.tif', band_indices_s2=[2,3,4],  # 只取 S2 的 RGB 三波段（可选）
#     img_size=(512, 512), train=True,
#     transform=build_geo_transform((512,512), is_train=True),  # 只做几何对齐；Normalize 可另行处理
#     # 若某目录无前缀，例如 lc/11624.tif，使用：
#     # prefix_map={'s2':'s2_', 's1':'s1_', 'topo':'topo_', 'lc':''},
# )
#
# val_set = HunanDataset(
#     data_root=r'H:\Datasets\Hunan_Dataset\Hunan_Dataset\test',
#     rgb_dir='s2', aux1_dir='s1', aux2_dir='topo', mask_dir='lc',
#     suffix='.tif', band_indices_s2=[2,3,4], img_size=(512, 512), train=False,
#     transform=build_geo_transform((512,512), is_train=False),
# )
#
# 若 config 里想使用你自己的 val_aug/train_aug（普通函数），也可：
# def val_aug(img, aux1, aux2, mask):
#     # 这里可以调用 get_val_transform()(image=..., aux1=..., aux2=..., mask=...)
#     tfm = get_val_transform()
#     d = tfm(image=img, aux1=aux1, aux2=aux2, mask=mask)
#     return d['image'], d['aux1'], d['aux2'], d['mask']
#
# val_set = HunanDataset(..., transform=val_aug)
