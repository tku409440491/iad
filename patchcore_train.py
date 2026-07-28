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
import json
import random
import shutil
import tempfile
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import v2 as T
from torchvision.transforms.functional import to_tensor
from anomalib.data import Folder
from anomalib.data.dataclasses.torch.image import ImageBatch
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
# 獨立驗證集校準（--val-mode independent 專用，不影響 from_train 邏輯）
# ---------------------------------------------------------------------------

def predict_one_image(model, img_path) -> float:
    """用訓練好的 model 對單張圖片推理，回傳 raw anomaly score。"""
    transform = model.pre_processor.transform
    device    = next(model.parameters()).device
    img = Image.open(img_path).convert("RGB")
    t   = transform(to_tensor(img)).to(device)
    batch = ImageBatch(image=t.unsqueeze(0))
    with torch.no_grad():
        out = model.predict_step(batch, 0)
    return out.pred_score[0].item()


def calibrate_with_validation(model, category, save_dir, engine=None, ckpt_path=None) -> dict:
    """
    用 validation/good 的圖片校準 threshold、min、max（獨立驗證集模式）。

    1. 載入 validation/good 所有圖片
    2. 用訓練好的 model 跑推理，收集 raw_score
    3. 計算 min, max, threshold（threshold = max，與 FROM_TRAIN 校準一致，只用正常品）
    4. 更新 model 的 post_processor
    5. 校準結果存成 JSON
    6. 重新儲存 checkpoint（含校準後的 threshold）
    """
    # 剛訓練完的 model 仍在 train 模式，PatchCore 的 forward 會回傳 embeddings 而非
    # InferenceBatch，需切到 eval 模式才能取得 pred_score
    model.eval()

    val_dir = LOCO_ROOT / category / "validation" / "good"
    val_images = sorted(val_dir.glob("*.png"))

    scores = [predict_one_image(model, p) for p in val_images]

    img_min   = min(scores)
    img_max   = max(scores)
    threshold = max(scores)

    # 更新 model 的 post_processor（三者皆為 Tensor buffer）
    with torch.no_grad():
        model.post_processor.image_min.copy_(torch.tensor(float(img_min)))
        model.post_processor.image_max.copy_(torch.tensor(float(img_max)))
        model.post_processor._image_threshold.copy_(torch.tensor(float(threshold)))

    # 存成 JSON
    calibration = {
        "img_min":      float(img_min),
        "img_max":      float(img_max),
        "threshold":    float(threshold),
        "n_val_images": len(scores),
    }
    cal_path = save_dir / "calibration_indval.json"
    with cal_path.open("w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2)

    # 重新儲存 checkpoint（含校準後的 threshold），之後載入時直接就有正確的 threshold
    if engine is not None and ckpt_path is not None:
        engine.trainer.save_checkpoint(str(ckpt_path))

    print(f"  校準完成（{len(val_images)} 張 validation 圖片）")
    print(f"  threshold={threshold:.4f}  min={img_min:.4f}  max={img_max:.4f}")
    print(f"  校準檔：{cal_path}")
    return calibration


def _read_checkpoint_calib(ckpt_path) -> dict | None:
    """讀取指定 checkpoint 的 post_processor min/max/threshold（用於比較），失敗回 None。"""
    try:
        m = Patchcore.load_from_checkpoint(str(ckpt_path), map_location="cpu", weights_only=False)
        pp = m.post_processor
        return {
            "img_min":   float(pp.image_min),
            "img_max":   float(pp.image_max),
            "threshold": float(pp._image_threshold),
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 單一類別訓練
# ---------------------------------------------------------------------------

def train_category(
    category: str,
    max_train_images: int | None = None,
    seed: int = 42,
    backbone: str = "wide_resnet50_2",
    layers_str: str = "11",
    val_mode: str = "from_train",
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

    # independent 模式：checkpoint 存到獨立子目錄，與 from_train 完全分開
    if val_mode == "independent":
        indval_tag = f"indval_dinov2_{layer_tag}" if backbone == "dinov2" else "indval"
        save_dir = MODELS_DIR / category / indval_tag

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

        # independent 模式：100% train/good 建特徵庫、不切驗證集（訓練後另用 validation/good 校準）
        val_split_mode = ValSplitMode.FROM_TRAIN
        if val_mode == "independent":
            normal_dir     = ["train/good"]
            val_split_mode = ValSplitMode.NONE
            val_ratio      = 0.0

        datamodule = Folder(
            name=category,
            root=LOCO_ROOT / category,
            normal_dir=normal_dir,
            abnormal_dir=["test/logical_anomalies", "test/structural_anomalies"],
            normal_test_dir="test/good",
            test_split_mode=TestSplitMode.FROM_DIR,
            val_split_mode=val_split_mode,
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

        # independent 模式無驗證集，關閉 Lightning 的 val 迴圈與 sanity check
        # （記憶庫仍於訓練階段建立，min/max/threshold 之後用 validation/good 校準）
        engine_kwargs = {}
        if val_mode == "independent":
            engine_kwargs = {"limit_val_batches": 0, "num_sanity_val_steps": 0}

        engine = Engine(
            max_epochs=1,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            default_root_dir=str(save_dir),
            **engine_kwargs,
        )

        engine.fit(model=model, datamodule=datamodule)

    finally:
        if tmp_dir is not None and tmp_dir.exists():
            shutil.rmtree(tmp_dir)

    elapsed = time.time() - t0

    # 找最新的 checkpoint
    ckpt_files = sorted(save_dir.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    latest_ckpt = ckpt_files[-1] if ckpt_files else None

    # independent 模式：用 validation/good 校準 threshold/min/max，並重存 checkpoint
    if val_mode == "independent" and latest_ckpt is not None:
        print(f"\n[{category}] 獨立驗證集校準…")
        calib = calibrate_with_validation(model, category, save_dir, engine, latest_ckpt)
        # 與 from_train 模式的 checkpoint 比較（若存在）
        ft_dir   = MODELS_DIR / category / "Patchcore" / category
        ft_ckpts = sorted(ft_dir.rglob("model.ckpt"), key=lambda p: p.stat().st_mtime) if ft_dir.exists() else []
        ft = _read_checkpoint_calib(ft_ckpts[-1]) if ft_ckpts else None
        if ft is not None:
            print(f"  [比較] from_train  ：threshold={ft['threshold']:.4f}  min={ft['img_min']:.4f}  max={ft['img_max']:.4f}")
            print(f"  [比較] independent ：threshold={calib['threshold']:.4f}  min={calib['img_min']:.4f}  max={calib['img_max']:.4f}")
        else:
            print("  [比較] 找不到 from_train checkpoint，略過比較")

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
    parser.add_argument(
        "--val-mode",
        type=str,
        default="from_train",
        choices=["from_train", "independent"],
        dest="val_mode",
        help="驗證集模式：from_train（現有，從訓練集切 20%%）或 independent（用 100%% train 建特徵庫，再用 validation/good 做閾值校準）",
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
                       seed=args.seed, backbone=args.backbone, layers_str=args.layers,
                       val_mode=args.val_mode)

    print(f"\n{'='*55}")
    print(f"全部完成！共訓練 {len(categories)} 個類別。")
    print(f"Checkpoint 根目錄：{MODELS_DIR}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
