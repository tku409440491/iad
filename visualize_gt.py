#!/usr/bin/env python3
"""
visualize_gt.py

將 MVTec LOCO-AD 的 ground truth mask 轉成有標註 bounding box 的視覺化圖片。

使用方式：
    python visualize_gt.py                          # 跑所有類別
    python visualize_gt.py --category breakfast_box # 只跑一個類別
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# 路徑與常數
# ---------------------------------------------------------------------------

LOCO_ROOT  = Path.home() / "data/iad/mvtec_loco"
OUTPUT_DIR = Path.home() / "data/iad/code/results/gt_visualized"

CATEGORIES = [
    "breakfast_box", "juice_bottle", "pushpins",
    "screw_bag", "splicing_connectors",
]

ANOMALY_TYPES = ["logical_anomalies", "structural_anomalies"]

BOX_COLOR   = (255, 0, 0)      # 紅色框
BOX_WIDTH   = 3
TEXT_COLOR  = (255, 255, 255)  # 白色文字
BG_COLOR    = (255, 0, 0)      # 紅色文字背景
FONT_SIZE   = 18


# ---------------------------------------------------------------------------
# 載入瑕疵對應表
# ---------------------------------------------------------------------------

def load_defect_map(category: str) -> dict[int, str]:
    """pixel_value → defect_name"""
    cfg_path = LOCO_ROOT / category / "defects_config.json"
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    return {item["pixel_value"]: item["defect_name"] for item in data}


# ---------------------------------------------------------------------------
# 單張影像視覺化
# ---------------------------------------------------------------------------

def visualize_gt(category: str, anomaly_type: str, img_id: str) -> Path:
    """
    讀取原圖 + GT mask，畫出 bounding box 並標註瑕疵名稱，存檔並回傳路徑。
    img_id: 不含副檔名的字串，如 "014"
    """
    defect_map = load_defect_map(category)

    img_path  = LOCO_ROOT / category / "test" / anomaly_type / f"{img_id}.png"
    mask_path = LOCO_ROOT / category / "ground_truth" / anomaly_type / img_id / "000.png"

    orig = Image.open(img_path).convert("RGB")
    mask = np.array(Image.open(mask_path).convert("L"))

    draw = ImageDraw.Draw(orig)

    # 嘗試載入系統字型，失敗則用預設字型
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", FONT_SIZE)
    except OSError:
        font = ImageFont.load_default()

    unique_vals = [v for v in np.unique(mask) if v != 0]

    for pv in unique_vals:
        defect_name = defect_map.get(int(pv), f"pixel_{pv}")

        ys, xs = np.where(mask == pv)
        if len(xs) == 0:
            continue

        min_x, max_x = int(xs.min()), int(xs.max())
        min_y, max_y = int(ys.min()), int(ys.max())

        # 畫 bounding box
        for w in range(BOX_WIDTH):
            draw.rectangle(
                [min_x - w, min_y - w, max_x + w, max_y + w],
                outline=BOX_COLOR,
            )

        # 計算文字背景範圍
        bbox = draw.textbbox((min_x, min_y), defect_name, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        label_x = min_x
        label_y = max(0, min_y - text_h - 4)

        draw.rectangle(
            [label_x, label_y, label_x + text_w + 4, label_y + text_h + 4],
            fill=BG_COLOR,
        )
        draw.text((label_x + 2, label_y + 2), defect_name, fill=TEXT_COLOR, font=font)

    # 存檔
    out_path = OUTPUT_DIR / category / anomaly_type / f"{img_id}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    orig.save(out_path)
    return out_path


# ---------------------------------------------------------------------------
# 跑單一類別 × 單一異常類型
# ---------------------------------------------------------------------------

def run_category_type(category: str, anomaly_type: str) -> int:
    gt_dir = LOCO_ROOT / category / "ground_truth" / anomaly_type
    if not gt_dir.exists():
        return 0

    img_ids = sorted(p.name for p in gt_dir.iterdir() if p.is_dir())
    print(f"  {anomaly_type}: {len(img_ids)} 張")

    for img_id in img_ids:
        out = visualize_gt(category, anomaly_type, img_id)
        print(f"    儲存：{out}")

    return len(img_ids)


# ---------------------------------------------------------------------------
# 主程式
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="GT mask 視覺化工具")
    parser.add_argument(
        "--category",
        choices=CATEGORIES,
        default=None,
        help="只處理指定類別（預設跑全部）",
    )
    args = parser.parse_args()

    categories = [args.category] if args.category else CATEGORIES
    total = 0

    for cat in categories:
        print(f"\n[{cat}]")
        for atype in ANOMALY_TYPES:
            total += run_category_type(cat, atype)

    print(f"\n完成！共產生 {total} 張視覺化圖片。")
    print(f"輸出目錄：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
