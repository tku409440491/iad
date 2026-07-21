#!/usr/bin/env python3
"""
efficientad_train.py
訓練 EfficientAD 異常偵測模型。

使用方式：
    python efficientad_train.py                           # 所有類別
    python efficientad_train.py --category breakfast_box  # 單一類別
    python efficientad_train.py --model-size m            # 使用 M size
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from anomalib.data import Folder
from anomalib.data.utils import TestSplitMode, ValSplitMode
from anomalib.engine import Engine
from anomalib.models import EfficientAd

# ---------------------------------------------------------------------------
# 路徑與常數
# ---------------------------------------------------------------------------

LOCO_ROOT    = Path.home() / "data/iad/mvtec_loco"
MODELS_DIR   = Path.home() / "data/iad/code/results/efficientad_models"
IMAGENET_DIR = Path.home() / "data/iad/code/datasets/imagenet/train"

CATEGORIES = [
    "breakfast_box", "juice_bottle", "pushpins",
    "screw_bag", "splicing_connectors",
]


# ---------------------------------------------------------------------------
# 單一類別訓練
# ---------------------------------------------------------------------------

def train_category(
    category: str,
    model_size: str = "s",
    max_steps: int = 20000,
) -> None:
    size_label = "S" if model_size == "s" else "M"
    print(f"\n{'='*55}")
    print(f"EfficientAD 訓練：{category}  model_size={size_label}")
    print(f"{'='*55}")
    print(f"  訓練步數：{max_steps}")
    t0 = time.time()

    save_dir = MODELS_DIR / category
    save_dir.mkdir(parents=True, exist_ok=True)

    # 合併 train/good 與 validation/good 作為訓練資料
    datamodule = Folder(
        name=category,
        root=LOCO_ROOT / category,
        normal_dir=["train/good", "validation/good"],
        abnormal_dir=["test/logical_anomalies", "test/structural_anomalies"],
        normal_test_dir="test/good",
        test_split_mode=TestSplitMode.FROM_DIR,
        val_split_mode=ValSplitMode.FROM_TRAIN,
        val_split_ratio=0.2,
        train_batch_size=1,
    )

    size_value = "small" if model_size == "s" else "medium"
    model = EfficientAd(
        imagenet_dir=IMAGENET_DIR,
        model_size=size_value,
    )

    engine = Engine(
        max_steps=max_steps,
        max_epochs=-1,  # 不限制 epoch，由 max_steps 控制
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        default_root_dir=str(save_dir),
    )

    engine.fit(model=model, datamodule=datamodule)

    elapsed = time.time() - t0

    ckpt_files = sorted(save_dir.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    latest_ckpt = ckpt_files[-1] if ckpt_files else None

    print(f"\n[{category}] 訓練完成")
    if latest_ckpt:
        size_mb = latest_ckpt.stat().st_size / 1024 / 1024
        print(f"  checkpoint：{latest_ckpt}")
        print(f"  檔案大小  ：{size_mb:.1f} MB")
    else:
        print("  警告：找不到 checkpoint 檔案")
    print(f"  訓練耗時  ：{elapsed:.0f} 秒")


# ---------------------------------------------------------------------------
# 主程式
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="訓練 EfficientAD 模型")
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
        "--model-size",
        type=str,
        default="s",
        choices=["s", "m"],
        dest="model_size",
        help="EfficientAD 模型大小：s（small，預設）或 m（medium）",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20000,
        dest="max_steps",
        help="訓練步數，EfficientAD 論文建議 70000，預設 20000",
    )
    args = parser.parse_args()

    categories = args.categories if args.categories else CATEGORIES
    size_label = "S" if args.model_size == "s" else "M"
    print(f"準備訓練 {len(categories)} 個類別：{categories}")
    print(f"模型大小：EfficientAD-{size_label}")
    print(f"ImageNet 路徑：{IMAGENET_DIR}")

    for cat in categories:
        train_category(cat, model_size=args.model_size, max_steps=args.max_steps)

    print(f"\n{'='*55}")
    print(f"全部完成！共訓練 {len(categories)} 個類別。")
    print(f"Checkpoint 根目錄：{MODELS_DIR}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
