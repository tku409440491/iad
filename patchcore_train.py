#!/usr/bin/env python3
"""
patchcore_train.py

訓練 PatchCore baseline，合併 train/good 與 validation/good 當訓練資料，
每個類別獨立訓練並儲存 checkpoint。

使用方式：
    python patchcore_train.py                                            # 全部類別，full-shot
    python patchcore_train.py --category breakfast_box                   # 單一類別，full-shot
    python patchcore_train.py --category breakfast_box --max-train-images 5  # few-shot（5 張）
    python patchcore_train.py --max-train-images 5 --seed 0              # 全部類別，few-shot
"""

from __future__ import annotations

import argparse
import random
import shutil
import tempfile
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

LOCO_ROOT  = Path.home() / "data/iad/mvtec_loco"
MODELS_DIR = Path.home() / "data/iad/code/results/patchcore_models"

CATEGORIES = [
    "breakfast_box", "juice_bottle", "pushpins",
    "screw_bag", "splicing_connectors",
]


# ---------------------------------------------------------------------------
# 單一類別訓練
# ---------------------------------------------------------------------------

def train_category(
    category: str,
    max_train_images: int | None = None,
    seed: int = 42,
    backbone: str = "wide_resnet50_2",
    layers_str: str = "11",
) -> None:
    # DINOv2 層標籤，例如 "3,9" → "3_9"
    layer_tag = layers_str.replace(",", "_") if backbone == "dinov2" else ""

    print(f"\n{'='*55}")
    print(f"PatchCore 訓練：{category}  backbone={backbone}" +
          (f"  layers={layers_str}" if backbone == "dinov2" else ""))
    if max_train_images is not None:
        print(f"模式：few-shot（最多 {max_train_images} 張，seed={seed}）")
    else:
        print(f"模式：full-shot（全部訓練圖片）")
    print(f"{'='*55}")
    t0 = time.time()

    # 依據 backbone、layers、訓練模式決定 checkpoint 儲存路徑
    if backbone == "dinov2":
        model_subdir = f"dinov2_{layer_tag}"
        if max_train_images is not None:
            save_dir = MODELS_DIR / category / f"few{max_train_images}_{model_subdir}"
        else:
            save_dir = MODELS_DIR / category / model_subdir
    else:
        if max_train_images is not None:
            save_dir = MODELS_DIR / category / f"few{max_train_images}"
        else:
            save_dir = MODELS_DIR / category
    save_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有正常訓練圖片（train/good + validation/good 合併）
    train_imgs = sorted((LOCO_ROOT / category / "train" / "good").glob("*.png"))
    val_imgs   = sorted((LOCO_ROOT / category / "validation" / "good").glob("*.png"))
    all_imgs   = [(p, "tr") for p in train_imgs] + [(p, "va") for p in val_imgs]
    total      = len(all_imgs)

    if max_train_images is not None and max_train_images < total:
        rng      = random.Random(seed)
        selected = rng.sample(all_imgs, max_train_images)
        use_n    = max_train_images
    else:
        selected = all_imgs
        use_n    = total

    print(f"  使用訓練圖片：{use_n} / 總計 {total} 張")

    # few-shot：在 LOCO_ROOT/category 內建立暫存目錄，用 symlink 指向選取的圖片
    # 前綴 tr_/va_ 避免 train/good 和 validation/good 的同名檔案衝突
    tmp_dir = None
    if max_train_images is not None and use_n < total:
        tmp_dir = Path(tempfile.mkdtemp(
            dir=LOCO_ROOT / category,
            prefix="fewshot_",
        ))
        for img_path, prefix in selected:
            link = tmp_dir / f"{prefix}_{img_path.name}"
            link.symlink_to(img_path.resolve())
        normal_dir = [tmp_dir.name]
    else:
        normal_dir = ["train/good", "validation/good"]

    try:
        # few-shot 時用更高的 val ratio，確保 threshold 校準有足夠樣本
        val_ratio = 0.4 if max_train_images is not None else 0.2

        datamodule = Folder(
            name=category,
            root=LOCO_ROOT / category,
            normal_dir=normal_dir,
            abnormal_dir=["test/logical_anomalies", "test/structural_anomalies"],
            normal_test_dir="test/good",
            test_split_mode=TestSplitMode.FROM_DIR,
            val_split_mode=ValSplitMode.FROM_TRAIN,
            val_split_ratio=val_ratio,
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

        # few-shot 時保留全部 patch，full-shot 時用 10% coreset
        coreset_ratio = 1.0 if max_train_images is not None else 0.1
        model = Patchcore(
            backbone=backbone_name,
            layers=layers,
            coreset_sampling_ratio=coreset_ratio,
            pre_processor=pre_processor,
        )

        engine = Engine(
            max_epochs=1,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            default_root_dir=str(save_dir),
        )

        engine.fit(model=model, datamodule=datamodule)

    finally:
        if tmp_dir is not None and tmp_dir.exists():
            shutil.rmtree(tmp_dir)

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
    parser = argparse.ArgumentParser(description="訓練 PatchCore 模型")
    parser.add_argument(
        "--category",
        nargs="+",
        choices=CATEGORIES,
        default=None,
        dest="categories",
        metavar="CATEGORY",
        help="指定類別（預設：所有 5 個類別）",
    )
    parser.add_argument(
        "--max-train-images",
        type=int,
        default=None,
        dest="max_train_images",
        help="最多使用幾張正常訓練圖片（預設：全部）",
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
        help="DINOv2 使用的 blocks，逗號分隔，例如 '11' 或 '3,9' 或 '3,6,9,11'，ResNet 時忽略此參數",
    )
    args = parser.parse_args()

    categories = args.categories if args.categories else CATEGORIES

    mode_str = (
        f"few-shot（{args.max_train_images} 張，seed={args.seed}）"
        if args.max_train_images else "full-shot"
    )
    layers_info = f"  layers={args.layers}" if args.backbone == "dinov2" else ""
    print(f"準備訓練 {len(categories)} 個類別：{categories}")
    print(f"訓練模式：{mode_str}  backbone={args.backbone}{layers_info}")

    for cat in categories:
        train_category(cat, max_train_images=args.max_train_images,
                       seed=args.seed, backbone=args.backbone, layers_str=args.layers)

    print(f"\n{'='*55}")
    print(f"全部完成！共訓練 {len(categories)} 個類別。")
    print(f"Checkpoint 根目錄：{MODELS_DIR}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
