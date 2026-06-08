#!/usr/bin/env python3
"""
visualize_detections.py

並排顯示三張圖：原圖 / YOLO 偵測結果 / Ground Truth mask，並存成 PNG。

輸出說明：
    左  ── 原圖（無任何標注）
    中  ── YOLO 偵測框（紅色=結構異常, 藍色=邏輯異常）
    右  ── GT mask 疊在原圖上（每種 defect 各自不同顏色 + 圖例）

用法：
    python visualize_detections.py \\
        --category breakfast_box \\
        --anomaly-type logical_anomalies \\
        --img-id 000

    # 指定輸出資料夾（預設 ~/data/iad/code/viz/）
    python visualize_detections.py ... --output-dir ./output
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# 常數
# ---------------------------------------------------------------------------

DATASET_ROOT = Path.home() / "data/iad/mvtec_loco"
OUTPUT_DIR   = Path.home() / "data/iad/code/viz"
BEST_PT      = Path.home() / "data/iad/code/runs/yolo_baseline/weights/best.pt"

# YOLO 偵測框顏色（RGB）
_YOLO_COLOR = {0: (220, 50, 50), 1: (50, 90, 220)}     # 0=紅(structural), 1=藍(logical)
_YOLO_LABEL = {0: "structural", 1: "logical"}

# GT mask 的 defect 上色用 20 色調色盤（matplotlib tab20）
_PALETTE_20 = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44),  (214, 39, 40),
    (148, 103, 189),(140, 86, 75),  (227, 119, 194),(127, 127, 127),
    (188, 189, 34), (23, 190, 207),(174, 199, 232),(255, 187, 120),
    (152, 223, 138),(255, 152, 150),(197, 176, 213),(196, 156, 148),
    (247, 182, 210),(199, 199, 199),(219, 219, 141),(158, 218, 229),
]

_FONT_PATH  = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_FONT_SIZE  = 22    # YOLO 標籤文字大小（px）


# ---------------------------------------------------------------------------
# YOLO 推理
# ---------------------------------------------------------------------------

def _run_yolo(image_path: Path, conf: float = 0.10) -> list[dict]:
    """
    對單張圖片執行 YOLO 推理，回傳像素座標格式的偵測結果。

    回傳
    ----
    list[dict]：每個元素包含 class_id, conf, x1, y1, x2, y2（原圖像素座標）
    """
    from ultralytics import YOLO

    model   = YOLO(str(BEST_PT))
    results = model.predict(
        source=str(image_path),
        conf=conf,
        iou=0.5,
        imgsz=640,
        verbose=False,
    )

    detections = []
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "class_id": int(box.cls[0]),
                "conf":     float(box.conf[0]),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })
    return detections


# ---------------------------------------------------------------------------
# 中圖：在原圖上畫 YOLO 框
# ---------------------------------------------------------------------------

def _draw_yolo_boxes(orig: Image.Image, detections: list[dict]) -> Image.Image:
    """
    在原圖副本上畫 YOLO bounding box，並標註類別名稱與信心分數。

    structural（class 0）→ 紅框
    logical   （class 1）→ 藍框
    """
    img  = orig.copy()
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(_FONT_PATH, _FONT_SIZE)
    except OSError:
        font = ImageFont.load_default(size=_FONT_SIZE)

    for det in detections:
        cls_id           = int(det["class_id"])
        x1, y1, x2, y2  = det["x1"], det["y1"], det["x2"], det["y2"]
        color            = _YOLO_COLOR.get(cls_id, (128, 128, 128))
        label            = f"{_YOLO_LABEL.get(cls_id, str(cls_id))} {det['conf']:.2f}"

        # 框線（3px）
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # 標籤背景
        tb = draw.textbbox((x1, y1 - _FONT_SIZE - 4), label, font=font)
        draw.rectangle([tb[0]-2, tb[1]-2, tb[2]+2, tb[3]+2], fill=color)
        draw.text((x1, y1 - _FONT_SIZE - 4), label, fill=(255, 255, 255), font=font)

    return img


# ---------------------------------------------------------------------------
# 右圖：GT mask 上色疊圖
# ---------------------------------------------------------------------------

def _colorize_gt(
    category: str,
    anomaly_type: str,
    img_id: str,
    orig: Image.Image,
    dataset_root: Path,
) -> tuple[Image.Image, list[mpatches.Patch]]:
    """
    讀取 ground truth mask，每種 defect 賦予不同顏色，疊在原圖上。

    回傳
    ----
    gt_overlay : 疊色後的 PIL.Image（RGB）
    legend_patches : matplotlib Patch 列表，供圖例使用
    """
    # 讀取 defect 名稱對照表
    cfg      = json.loads((dataset_root / category / "defects_config.json").read_text())
    pv_name  = {d["pixel_value"]: d["defect_name"] for d in cfg}

    # 讀取 GT mask
    mask_path = (
        dataset_root / category / "ground_truth" / anomaly_type / img_id / "000.png"
    )
    mask = np.array(Image.open(mask_path))   # shape: (H, W), dtype: uint8

    # 找出本圖實際出現的 defect pixel value
    unique_pvs = sorted(int(pv) for pv in np.unique(mask) if pv > 0)

    # 為每個 defect 指定調色盤顏色
    pv_color: dict[int, tuple[int, int, int]] = {
        pv: _PALETTE_20[i % len(_PALETTE_20)]
        for i, pv in enumerate(unique_pvs)
    }

    # 建立 RGBA overlay（只在 defect 區域填色，alpha≈70%）
    overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
    for pv, (r, g, b) in pv_color.items():
        overlay[mask == pv] = [r, g, b, 178]   # 178/255 ≈ 0.70

    # 原圖與 overlay 合成
    base         = orig.convert("RGBA")
    overlay_img  = Image.fromarray(overlay, "RGBA")
    gt_display   = Image.alpha_composite(base, overlay_img).convert("RGB")

    # 建立圖例 patches
    legend_patches = [
        mpatches.Patch(
            color=tuple(c / 255 for c in pv_color[pv]),
            label=pv_name.get(pv, f"defect_{pv}"),
        )
        for pv in unique_pvs
    ]

    return gt_display, legend_patches


# ---------------------------------------------------------------------------
# 主視覺化函式
# ---------------------------------------------------------------------------

def visualize(
    category: str,
    anomaly_type: str,
    img_id: str,
    output_dir: Path = OUTPUT_DIR,
    dataset_root: Path = DATASET_ROOT,
    conf: float = 0.10,
) -> Path:
    """
    並排繪製三張圖（原圖 / YOLO / GT mask），存成 PNG 並回傳輸出路徑。

    參數
    ----
    category     : 產品類別，例如 'breakfast_box'
    anomaly_type : 'logical_anomalies' 或 'structural_anomalies'
    img_id       : 圖片編號，例如 '000'
    output_dir   : PNG 輸出資料夾
    dataset_root : mvtec_loco 根目錄
    conf         : YOLO 偵測信心閾值（預設 0.10）

    回傳
    ----
    Path：儲存的 PNG 路徑
    """
    # ── 載入原圖 ────────────────────────────────────────────────────────
    img_path = dataset_root / category / "test" / anomaly_type / f"{img_id}.png"
    if not img_path.exists():
        raise FileNotFoundError(f"找不到圖片：{img_path}")

    orig = Image.open(img_path).convert("RGB")

    # ── 中圖：YOLO 推理 ─────────────────────────────────────────────────
    print(f"  執行 YOLO 推理（conf={conf}）…")
    detections = _run_yolo(img_path, conf=conf)
    n_str = sum(1 for d in detections if d["class_id"] == 0)
    n_log = sum(1 for d in detections if d["class_id"] == 1)
    print(f"  偵測到 {len(detections)} 框（structural={n_str}, logical={n_log}）")
    yolo_img = _draw_yolo_boxes(orig, detections)

    # ── 右圖：GT mask 上色 ──────────────────────────────────────────────
    gt_img, legend_patches = _colorize_gt(
        category, anomaly_type, img_id, orig, dataset_root
    )

    # ── 繪製三欄並排圖 ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle(
        f"{category}  /  {anomaly_type}  /  {img_id}.png  "
        f"(YOLO conf={conf})",
        fontsize=13, fontweight="bold", y=1.01,
    )

    # 左：原圖
    axes[0].imshow(orig)
    axes[0].set_title("Original", fontsize=12)
    axes[0].axis("off")

    # 中：YOLO 偵測
    yolo_title = (
        f"YOLO Detections  ({len(detections)} boxes)"
        if detections
        else "YOLO Detections  (none)"
    )
    axes[1].imshow(yolo_img)
    axes[1].set_title(yolo_title, fontsize=12)
    axes[1].axis("off")

    # 右：GT mask（含圖例）
    axes[2].imshow(gt_img)
    axes[2].set_title("Ground Truth Mask", fontsize=12)
    axes[2].axis("off")

    if legend_patches:
        axes[2].legend(
            handles=legend_patches,
            loc="lower right",
            fontsize=8,
            framealpha=0.85,
            title="defect",
            title_fontsize=8,
        )

    plt.tight_layout()

    # ── 儲存輸出 ────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    atype_short = "log" if anomaly_type == "logical_anomalies" else "str"
    out_path    = output_dir / f"{category}_{atype_short}_{img_id}.png"

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  儲存完成：{out_path}")
    return out_path


# ---------------------------------------------------------------------------
# 主程式（argparse）
# ---------------------------------------------------------------------------

def main() -> None:
    """解析命令列參數並呼叫 visualize()。"""
    parser = argparse.ArgumentParser(
        description="並排顯示原圖、YOLO 偵測、GT mask，存成 PNG。"
    )
    parser.add_argument(
        "--category",
        required=True,
        choices=[
            "breakfast_box", "juice_bottle",
            "pushpins", "screw_bag", "splicing_connectors",
        ],
        help="產品類別",
    )
    parser.add_argument(
        "--anomaly-type",
        required=True,
        choices=["logical_anomalies", "structural_anomalies"],
        help="異常類型資料夾名稱",
    )
    parser.add_argument(
        "--img-id",
        required=True,
        help="圖片編號，例如 000",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"PNG 輸出資料夾（預設：{OUTPUT_DIR}）",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.10,
        help="YOLO 偵測信心閾值（預設 0.10）",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT,
        help=f"mvtec_loco 根目錄（預設：{DATASET_ROOT}）",
    )
    args = parser.parse_args()

    print(f"視覺化：{args.category} / {args.anomaly_type} / {args.img_id}.png")
    out = visualize(
        category     = args.category,
        anomaly_type = args.anomaly_type,
        img_id       = args.img_id,
        output_dir   = args.output_dir,
        dataset_root = args.dataset_root,
        conf         = args.conf,
    )
    print(f"完成：{out}")


if __name__ == "__main__":
    main()

"""
# breakfast_box 邏輯異常，圖片 000
python code/visualize_detections.py \
    --category juice_bottle \
    --anomaly-type logical_anomalies \
    --img-id 054


# splicing_connectors 結構異常，圖片 005
python code/visualize_detections.py \
    --category splicing_connectors \
    --anomaly-type structural_anomalies \
    --img-id 005

# 調高 conf threshold，輸出到自訂資料夾
python code/visualize_detections.py \
    --category screw_bag \
    --anomaly-type logical_anomalies \
    --img-id 002 \
    --conf 0.25 \
    --output-dir ~/data/iad/code/viz/screw_bag
"""