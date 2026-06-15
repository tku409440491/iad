#!/usr/bin/env python3
"""
patchcore_train.py

訓練 PatchCore baseline，合併 train/good 與 validation/good 當訓練資料，
每個類別獨立訓練並儲存 checkpoint。

使用方式：
    python patchcore_train.py                             # 訓練所有 5 個類別
    python patchcore_train.py breakfast_box               # 只訓練一個類別
    python patchcore_train.py breakfast_box juice_bottle  # 訓練指定多個
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from anomalib.data import Folder
from anomalib.data.utils import TestSplitMode, ValSplitMode
from anomalib.engine import Engine
from anomalib.models import Patchcore

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

def train_category(category: str) -> None:
    print(f"\n{'='*55}")
    print(f"PatchCore 訓練：{category}")
    print(f"{'='*55}")
    t0 = time.time()

    save_dir = MODELS_DIR / category
    save_dir.mkdir(parents=True, exist_ok=True)

    # normal_dir 接受 list，直接合併 train/good 和 validation/good
    datamodule = Folder(
        name=category,
        root=LOCO_ROOT / category,
        normal_dir=["train/good", "validation/good"],
        abnormal_dir=["test/logical_anomalies", "test/structural_anomalies"],
        normal_test_dir="test/good",
        test_split_mode=TestSplitMode.FROM_DIR,
        val_split_mode=ValSplitMode.FROM_TRAIN,
        val_split_ratio=0.2,
    )

    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.1,
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
    categories = sys.argv[1:] if len(sys.argv) > 1 else CATEGORIES

    unknown = [c for c in categories if c not in CATEGORIES]
    if unknown:
        print(f"未知類別：{unknown}")
        print(f"可用類別：{CATEGORIES}")
        sys.exit(1)

    print(f"準備訓練 {len(categories)} 個類別：{categories}")

    for cat in categories:
        train_category(cat)

    print(f"\n{'='*55}")
    print(f"全部完成！共訓練 {len(categories)} 個類別。")
    print(f"Checkpoint 根目錄：{MODELS_DIR}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
