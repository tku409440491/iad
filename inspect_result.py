#!/usr/bin/env python3
"""
inspect_result.py

給定一張圖片，顯示三欄視覺化（原圖 / YOLO / GT mask）+ 從 JSONL 讀取
Gemma 的文字描述，存成 PNG 到 results/inspect/。

用法：
    # 指定 stem
    python code/inspect_result.py --stem breakfast_box__str__004

    # 指定類別與編號
    python code/inspect_result.py --category breakfast_box --anomaly_type str --img_id 004

    # 批次輸出所有 FP
    python code/inspect_result.py --all_fp

    # 批次輸出所有 FN
    python code/inspect_result.py --all_fn
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from gemini_inference import draw_detections

# ---------------------------------------------------------------------------
# 中文字型設定（Noto Sans TC）
# ---------------------------------------------------------------------------

_NOTO_TC_PATH = Path.home() / "data/iad/Noto_Sans_TC/NotoSansTC-Regular.ttf"
if _NOTO_TC_PATH.exists():
    font_manager.fontManager.addfont(str(_NOTO_TC_PATH))
    _noto_prop = font_manager.FontProperties(fname=str(_NOTO_TC_PATH))
    plt.rcParams["font.family"] = _noto_prop.get_name()
else:
    print(f"警告：找不到 Noto Sans TC 字型（{_NOTO_TC_PATH}），中文可能顯示為框框")


# ---------------------------------------------------------------------------
# 常數設定
# ---------------------------------------------------------------------------

DATASET_ROOT = Path.home() / "data/iad/mvtec_loco"
BEST_PT      = Path.home() / "data/iad/code/runs/yolo_baseline/weights/best.pt"
RESULTS_DIR  = Path.home() / "data/iad/code/results"
JSONL_PATH   = RESULTS_DIR / "pipeline_results.jsonl"
INSPECT_DIR  = RESULTS_DIR / "inspect"

YOLO_CONF = 0.10

_GT_MAP = {"log": "logical", "str": "structural"}

# anomaly_type CLI 輸入 → stem 用的短寫
_ATYPE_NORM = {
    "str": "str", "structural": "str",
    "log": "log", "logical":    "log",
}

# GT mask 上色調色盤（matplotlib tab20）
_PALETTE_20 = [
    (31, 119, 180),  (255, 127, 14),  (44, 160, 44),   (214, 39, 40),
    (148, 103, 189), (140, 86, 75),   (227, 119, 194),  (127, 127, 127),
    (188, 189, 34),  (23, 190, 207),  (174, 199, 232),  (255, 187, 120),
    (152, 223, 138), (255, 152, 150), (197, 176, 213),  (196, 156, 148),
    (247, 182, 210), (199, 199, 199), (219, 219, 141),  (158, 218, 229),
]

# Verdict 顯示顏色
_VERDICT_COLOR = {"TP": "#1a7d1a", "FP": "#cc2222", "FN": "#cc7700"}


# ---------------------------------------------------------------------------
# 工具函式
# ---------------------------------------------------------------------------

def parse_stem(stem: str) -> tuple[str, str, str]:
    """解析 stem → (category, gt_class, img_id)"""
    parts = stem.split("__")
    if len(parts) != 3:
        raise ValueError(f"無法解析 stem：{stem}（預期格式 category__log|str__id）")
    category, gt_short, img_id = parts
    gt_class = _GT_MAP.get(gt_short)
    if gt_class is None:
        raise ValueError(f"未知 GT 簡寫：{gt_short}（來源：{stem}）")
    return category, gt_class, img_id


def load_jsonl(jsonl_path: Path) -> dict[str, dict]:
    """讀取 JSONL，回傳以 stem 為 key 的 dict。"""
    records: dict[str, dict] = {}
    if not jsonl_path.exists():
        return records
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            records[rec["stem"]] = rec
        except (json.JSONDecodeError, KeyError):
            pass
    return records


def run_yolo(model, image_path: Path) -> list[dict]:
    """對單張圖片執行 YOLO 推理，回傳偵測框列表。"""
    results = model.predict(
        source=str(image_path),
        conf=YOLO_CONF,
        iou=0.50,
        imgsz=640,
        verbose=False,
    )
    detections: list[dict] = []
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "class_id": int(box.cls[0]),
                "conf":     float(box.conf[0]),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })
    return detections


def colorize_gt(
    category: str,
    gt_class: str,
    img_id: str,
    orig: Image.Image,
) -> tuple[Image.Image, list[mpatches.Patch]]:
    """GT mask 上色疊圖，回傳 (疊色圖, legend patches)。"""
    atype_folder = f"{gt_class}_anomalies"
    cfg_path     = DATASET_ROOT / category / "defects_config.json"
    mask_path    = DATASET_ROOT / category / "ground_truth" / atype_folder / img_id / "000.png"

    pv_name = {
        d["pixel_value"]: d["defect_name"]
        for d in json.loads(cfg_path.read_text())
    }
    mask = np.array(Image.open(mask_path))

    unique_pvs = sorted(int(pv) for pv in np.unique(mask) if pv > 0)
    pv_color   = {pv: _PALETTE_20[i % len(_PALETTE_20)] for i, pv in enumerate(unique_pvs)}

    overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
    for pv, (r, g, b) in pv_color.items():
        overlay[mask == pv] = [r, g, b, 178]   # alpha ≈ 70%

    gt_display = Image.alpha_composite(
        orig.convert("RGBA"),
        Image.fromarray(overlay, "RGBA"),
    ).convert("RGB")

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

def inspect_one(
    stem: str,
    record: dict | None,
    yolo_model,
    output_dir: Path = INSPECT_DIR,
) -> Path:
    """
    為單一 stem 生成 inspect PNG：上半部三欄圖片，下半部文字資訊。

    參數
    ----
    stem       : 例如 'breakfast_box__str__004'
    record     : 從 JSONL 讀取的結果 dict；找不到時傳 None
    yolo_model : 預載的 YOLO 模型
    output_dir : PNG 輸出資料夾

    回傳
    ----
    儲存路徑
    """
    category, gt_class, img_id = parse_stem(stem)
    atype_folder = f"{gt_class}_anomalies"
    img_path     = DATASET_ROOT / category / "test" / atype_folder / f"{img_id}.png"

    if not img_path.exists():
        raise FileNotFoundError(f"找不到圖片：{img_path}")

    orig = Image.open(img_path).convert("RGB")

    # ── 中：YOLO 標注圖 ────────────────────────────────────────────────────
    detections = run_yolo(yolo_model, img_path)
    yolo_img   = draw_detections(img_path, detections)

    # ── 右：GT mask 疊圖 ───────────────────────────────────────────────────
    gt_img, legend_patches = colorize_gt(category, gt_class, img_id, orig)

    # ── 從 JSONL 記錄取得文字資訊 ──────────────────────────────────────────
    if record:
        n_str      = record.get("n_structural", "?")
        n_log      = record.get("n_logical",    "?")
        verdict    = record.get("verdict",      "N/A")
        gemma_type = record.get("anomaly_type", "N/A")
        reason_raw = record.get("reason",       "（無資料）")
    else:
        n_str = n_log = "N/A"
        verdict    = "N/A"
        gemma_type = "N/A"
        reason_raw = "（JSONL 中找不到此 stem，僅顯示視覺化）"

    verdict_color = _VERDICT_COLOR.get(verdict, "#333333")

    # reason 換行處理（label 寬度 = 14 字元，後續行縮排對齊）
    reason_label  = "Gemma reason:  "
    reason_filled = textwrap.fill(
        str(reason_raw),
        width=125,
        initial_indent=reason_label,
        subsequent_indent=" " * len(reason_label),
    )

    # ── 建立圖形 ───────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(24, 14), facecolor="white")
    gs  = gridspec.GridSpec(
        2, 3, figure=fig,
        height_ratios=[5, 2.5],
        hspace=0.22, wspace=0.05,
    )

    fig.suptitle(
        f"{category}  /  {gt_class}  /  {img_id}.png",
        fontsize=14, fontweight="bold", y=0.995,
    )

    # ── 上排：三張圖 ────────────────────────────────────────────────────────
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_yolo = fig.add_subplot(gs[0, 1])
    ax_gt   = fig.add_subplot(gs[0, 2])

    ax_orig.imshow(orig)
    ax_orig.set_title("Original", fontsize=12)
    ax_orig.axis("off")

    n_yolo = len(detections)
    ax_yolo.imshow(yolo_img)
    ax_yolo.set_title(
        f"YOLO Detections  ({n_yolo} boxes, conf={YOLO_CONF})",
        fontsize=12,
    )
    ax_yolo.axis("off")

    ax_gt.imshow(gt_img)
    ax_gt.set_title("Ground Truth Mask", fontsize=12)
    ax_gt.axis("off")
    if legend_patches:
        ax_gt.legend(
            handles=legend_patches,
            loc="lower right",
            fontsize=8,
            framealpha=0.85,
            title="defect",
            title_fontsize=8,
        )

    # ── 下排：文字區域（橫跨三欄）──────────────────────────────────────────
    ax_txt = fig.add_subplot(gs[1, :])
    ax_txt.set_facecolor("#f7f7f7")
    ax_txt.axis("off")

    # 分隔線（axhline 不接受 transform，改用 plot 畫在 axes 最上緣）
    ax_txt.plot([0, 1], [1, 1], color="#cccccc", linewidth=1,
                transform=ax_txt.transAxes, clip_on=False)

    fs = 12       # 一般欄位字體大小
    x  = 0.013   # 左邊距（axes 座標）

    # 各欄位 y 位置（由上往下，間距約 0.19）
    ax_txt.text(x, 0.88, f"GT class:    {gt_class}",
                transform=ax_txt.transAxes, fontsize=fs, va="top")

    ax_txt.text(x, 0.69, f"Verdict:     {verdict}",
                transform=ax_txt.transAxes, fontsize=fs, va="top",
                fontweight="bold", color=verdict_color)

    ax_txt.text(x, 0.50, f"YOLO 偵測:   str={n_str},  log={n_log}",
                transform=ax_txt.transAxes, fontsize=fs, va="top")

    ax_txt.text(x, 0.31, f"Gemma 判斷:  {gemma_type}",
                transform=ax_txt.transAxes, fontsize=fs, va="top")

    # reason（含自動換行）
    ax_txt.text(x, 0.12, reason_filled,
                transform=ax_txt.transAxes, fontsize=fs - 1, va="top",
                clip_on=True)

    # ── 儲存 ────────────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{stem}_inspect.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    verdict_str = f"[{verdict}]" if verdict != "N/A" else "[no record]"
    print(f"  {verdict_str}  {out_path.name}")
    return out_path


# ---------------------------------------------------------------------------
# 主程式
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="視覺化單張或批次 FP/FN 的 YOLO+Gemma 結果"
    )

    # 互斥：指定單一目標 的三種方式
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--stem",
        help="直接指定 stem，例如 breakfast_box__str__004",
    )
    mode_group.add_argument(
        "--all_fp",
        action="store_true",
        help="批次輸出 JSONL 中所有 verdict=FP 的圖片",
    )
    mode_group.add_argument(
        "--all_fn",
        action="store_true",
        help="批次輸出 JSONL 中所有 verdict=FN 的圖片",
    )

    # 透過 category + anomaly_type + img_id 指定單張
    parser.add_argument("--category",     help="產品類別，例如 breakfast_box")
    parser.add_argument("--anomaly_type", help="str / log / structural / logical")
    parser.add_argument("--img_id",       help="圖片編號，例如 004")

    parser.add_argument(
        "--jsonl",
        type=Path,
        default=JSONL_PATH,
        help=f"JSONL 來源（預設：{JSONL_PATH}）",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=INSPECT_DIR,
        help=f"PNG 輸出資料夾（預設：{INSPECT_DIR}）",
    )
    args = parser.parse_args()

    # ── 讀取 JSONL ──────────────────────────────────────────────────────────
    records = load_jsonl(args.jsonl)
    if not records:
        print(f"警告：JSONL 為空或不存在（{args.jsonl}）")

    # ── 決定處理目標 ────────────────────────────────────────────────────────
    if args.all_fp or args.all_fn:
        filter_verdict = "FP" if args.all_fp else "FN"
        targets = [
            (stem, rec)
            for stem, rec in sorted(records.items())
            if rec.get("verdict") == filter_verdict
        ]
        if not targets:
            print(f"JSONL 中沒有 verdict={filter_verdict} 的記錄。")
            return
        print(f"批次模式：共 {len(targets)} 張 {filter_verdict}\n")

    elif args.stem:
        stem   = args.stem
        record = records.get(stem)
        if record is None:
            print(f"警告：JSONL 中找不到 {stem}，仍顯示視覺化。")
        targets = [(stem, record)]

    elif args.category and args.anomaly_type and args.img_id:
        gt_short = _ATYPE_NORM.get(args.anomaly_type)
        if gt_short is None:
            parser.error(
                f"--anomaly_type 不支援：'{args.anomaly_type}'  "
                f"（接受：str / log / structural / logical）"
            )
        stem   = f"{args.category}__{gt_short}__{args.img_id}"
        record = records.get(stem)
        if record is None:
            print(f"警告：JSONL 中找不到 {stem}，仍顯示視覺化。")
        targets = [(stem, record)]

    else:
        parser.error(
            "請指定以下其中一種：\n"
            "  --stem STEM\n"
            "  --category CAT --anomaly_type TYPE --img_id ID\n"
            "  --all_fp  或  --all_fn"
        )
        return

    # ── 載入 YOLO 模型（一次）────────────────────────────────────────────────
    from ultralytics import YOLO
    print("載入 YOLO 模型…")
    yolo_model = YOLO(str(BEST_PT))
    print()

    # ── 執行 ────────────────────────────────────────────────────────────────
    for i, (stem, record) in enumerate(targets, start=1):
        if len(targets) > 1:
            print(f"[{i:>3}/{len(targets)}] {stem}")
        inspect_one(stem, record, yolo_model, output_dir=args.output_dir)

    print(f"\n完成！共輸出 {len(targets)} 張  →  {args.output_dir}")


if __name__ == "__main__":
    main()

"""
# 指定 stem
python code/inspect_result.py --stem breakfast_box__str__004

# 指定類別與編號（等價）
python code/inspect_result.py --category breakfast_box --anomaly_type str --img_id 004

# 批次輸出所有 FP
python code/inspect_result.py --all_fp

# 批次輸出所有 FN
python code/inspect_result.py --all_fn

"""