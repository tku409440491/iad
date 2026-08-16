#!/usr/bin/env python3
"""
patchcore_train_mvtecad2.py

在 VisA 資料集上訓練 PatchCore baseline（與 LOCO / MVTec AD / AD2 版分開）。
結構：
  - 只用 train/good 建特徵庫（FROM_TRAIN 切 20% 校準）
  - 異常固定在 test/bad，正常測試在 test/good（圖片為 .JPG，anomalib 自動處理）
  - checkpoint 存到 patchcore_models_visa/

使用方式：
    python patchcore_train_visa.py                                    # 全部 12 類
    python patchcore_train_visa.py --category candle                  # 單一類別
    python patchcore_train_visa.py --category candle --backbone dinov2 --layers 3,9
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torchvision.transforms import v2 as T
from anomalib.data import Folder
from anomalib.data.utils import TestSplitMode, ValSplitMode
from anomalib.engine import Engine
from anomalib.models import Patchcore
from anomalib.pre_processing import PreProcessor

# ---------------------------------------------------------------------------
# 路徑與常數
# ---------------------------------------------------------------------------

VISA_ROOT = Path.home() / "data/iad/visa/visa_pytorch"
MODELS_DIR   = Path.home() / "data/iad/code/results/patchcore_models_visa"

ALL_CATEGORIES = [
    "candle", "capsules", "cashew", "chewinggum", "fryum",
    "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
]


# ---------------------------------------------------------------------------
# 單一類別訓練
# ---------------------------------------------------------------------------

def train_category(
    category: str,
    seed: int = 42,
    backbone: str = "wide_resnet50_2",
    layers_str: str = "11",
) -> None:
    layer_tag = layers_str.replace(",", "_") if backbone == "dinov2" else ""

    print(f"\n{'='*55}")
    print(f"PatchCore 訓練 (MVTec AD)：{category}  backbone={backbone}" +
          (f"  layers={layers_str}" if backbone == "dinov2" else ""))
    print(f"{'='*55}")
    t0 = time.time()

    # checkpoint 儲存路徑
    if backbone == "dinov2":
        save_dir = MODELS_DIR / category / f"dinov2_{layer_tag}"
    else:
        save_dir = MODELS_DIR / category
    save_dir.mkdir(parents=True, exist_ok=True)

    # MVTec AD 2：只用 train/good 建特徵庫（validation 資料夾不使用，改用 FROM_TRAIN 切分）
    normal_dir = ["train/good"]
    # 異常固定在 test_public/bad，正常測試在 test_public/good
    abnormal_dirs = ["test/bad"]
    print(f"  異常來源：{abnormal_dirs}")

    datamodule = Folder(
        name=category,
        root=VISA_ROOT / category,
        normal_dir=normal_dir,
        abnormal_dir=abnormal_dirs,
        normal_test_dir="test/good",
        test_split_mode=TestSplitMode.FROM_DIR,
        val_split_mode=ValSplitMode.FROM_TRAIN,
        val_split_ratio=0.2,
        seed=seed,
    )

    # backbone 選擇；DINOv2 ViT-B/14 固定要求 518×518（patch_size=14, 37×14=518）
    if backbone == "dinov2":
        backbone_name = "vit_base_patch14_dinov2"
        layers = [f"blocks.{l.strip()}" for l in layers_str.split(",")]
        pre_processor = PreProcessor(transform=T.Compose([
            T.Resize((518, 518)),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]))
    else:
        backbone_name = "wide_resnet50_2"
        layers = ["layer2", "layer3"]
        pre_processor = True  # 使用 anomalib 預設 pre_processor

    model = Patchcore(
        backbone=backbone_name,
        layers=layers,
        coreset_sampling_ratio=0.1,
        pre_processor=pre_processor,
    )

    engine = Engine(
        max_epochs=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        default_root_dir=str(save_dir),
    )

    engine.fit(model=model, datamodule=datamodule)

    elapsed = time.time() - t0

    # 找最新的 checkpoint
    ckpt_files = sorted(save_dir.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    latest_ckpt = ckpt_files[-1] if ckpt_files else None

    # 記憶庫大小（訓練後特徵向量數量）
    memory_bank_size: int | None = None
    try:
        mb = getattr(model, "memory_bank", None) \
          or getattr(getattr(model, "model", None), "memory_bank", None)
        if mb is not None:
            memory_bank_size = mb.shape[0]
    except Exception:
        pass

    print(f"\n[{category}] 訓練完成")
    if latest_ckpt:
        size_mb = latest_ckpt.stat().st_size / 1024 / 1024
        print(f"  checkpoint：{latest_ckpt}")
        print(f"  檔案大小  ：{size_mb:.1f} MB")
    else:
        print("  警告：找不到 checkpoint 檔案")
    if memory_bank_size is not None:
        print(f"  記憶庫大小：{memory_bank_size:,} 個特徵向量")
    print(f"  訓練耗時  ：{elapsed:.0f} 秒")


# ---------------------------------------------------------------------------
# 主程式
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="在 VisA 上訓練 PatchCore 模型")
    parser.add_argument(
        "--category",
        nargs="+",
        choices=ALL_CATEGORIES,
        default=None,
        dest="categories",
        metavar="CATEGORY",
        help="指定類別（預設：所有 12 個類別）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="取樣隨機種子（預設：42）",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="wide_resnet50_2",
        choices=["wide_resnet50_2", "dinov2"],
        help="PatchCore backbone，預設 wide_resnet50_2",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="11",
        help="DINOv2 使用的 blocks，逗號分隔，例如 '11' 或 '3,9'，ResNet 時忽略此參數",
    )
    args = parser.parse_args()

    categories = args.categories if args.categories else ALL_CATEGORIES

    layers_info = f"  layers={args.layers}" if args.backbone == "dinov2" else ""
    print(f"準備訓練 {len(categories)} 個類別：{categories}")
    print(f"backbone={args.backbone}{layers_info}")

    for cat in categories:
        train_category(cat, seed=args.seed, backbone=args.backbone, layers_str=args.layers)

    print(f"\n{'='*55}")
    print(f"全部完成！共訓練 {len(categories)} 個類別。")
    print(f"Checkpoint 根目錄：{MODELS_DIR}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
