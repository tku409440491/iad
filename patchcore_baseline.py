#!/usr/bin/env python3
"""PatchCore baseline 評估"""
from pathlib import Path
from anomalib.models import Patchcore
from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.data.utils import TestSplitMode, ValSplitMode
import torch

LOCO_ROOT  = Path.home() / "data/iad/mvtec_loco"
RESULTS_DIR = Path.home() / "data/iad/code/results/patchcore"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = [
    "breakfast_box", "juice_bottle", "pushpins",
    "screw_bag", "splicing_connectors",
]

def run_category(category: str):
    print(f"\n{'='*50}\nRunning PatchCore: {category}\n{'='*50}")

    datamodule = Folder(
        name=category,
        root=LOCO_ROOT / category,
        normal_dir="train/good",
        abnormal_dir=["test/logical_anomalies", "test/structural_anomalies"],
        normal_test_dir="test/good",
        test_split_mode=TestSplitMode.FROM_DIR,
        val_split_mode=ValSplitMode.FROM_TEST,
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
        default_root_dir=str(RESULTS_DIR / category),
    )

    engine.fit(model=model, datamodule=datamodule)
    results = engine.test(model=model, datamodule=datamodule)
    print(f"\n[{category}] 結果：{results}")
    return results

if __name__ == "__main__":
    import sys
    categories = [sys.argv[1]] if len(sys.argv) > 1 else [CATEGORIES[0]]
    for cat in categories:
        run_category(cat)
