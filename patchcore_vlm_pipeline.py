#!/usr/bin/env python3
"""
patchcore_vlm_pipeline.py

PatchCore + VLM 整合推論管線。
  text   mode：PatchCore anomaly_map → 九宮格文字描述 → Gemma
  heatmap mode：PatchCore anomaly_map → jet colormap 熱力圖 → Gemma

使用方式：
    python patchcore_vlm_pipeline.py --category breakfast_box --mode text
    python patchcore_vlm_pipeline.py --category breakfast_box --mode heatmap
    python patchcore_vlm_pipeline.py --mode text          # 跑所有類別
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from anomalib.data.dataclasses.torch.image import ImageBatch
from anomalib.models import Patchcore

import google.genai as genai
from gemini_inference import gemini_inference_patchcore

# ---------------------------------------------------------------------------
# 路徑常數
# ---------------------------------------------------------------------------

LOCO_ROOT  = Path.home() / "data/iad/mvtec_loco"
MODELS_DIR = Path.home() / "data/iad/code/results/patchcore_models"
RESULTS_DIR = Path.home() / "data/iad/code/results"

CATEGORIES = [
    "breakfast_box", "juice_bottle", "pushpins",
    "screw_bag", "splicing_connectors",
]

# 九宮格標籤
_GRID_LABELS = [
    ["左上", "中上", "右上"],
    ["左中", "中央", "右中"],
    ["左下", "中下", "右下"],
]


# ---------------------------------------------------------------------------
# PatchCore 載入
# ---------------------------------------------------------------------------

def load_patchcore(category: str, subdir: str = "Patchcore"):
    """載入指定類別最新的 PatchCore checkpoint，回傳 (model, ckpt_path)。"""
    cat_dir = MODELS_DIR / category / subdir / category
    if not cat_dir.exists():
        raise FileNotFoundError(f"找不到 {category} checkpoint 目錄：{cat_dir}")

    latest = cat_dir / "latest" / "weights" / "lightning" / "model.ckpt"
    if not latest.exists():
        ckpt_files = sorted(cat_dir.rglob("model.ckpt"), key=lambda p: p.stat().st_mtime)
        if not ckpt_files:
            raise FileNotFoundError(f"找不到 {category} model.ckpt")
        latest = ckpt_files[-1]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Patchcore.load_from_checkpoint(
        str(latest), map_location=device, weights_only=False
    )
    model.eval()
    return model, latest


# ---------------------------------------------------------------------------
# 單張圖片推論
# ---------------------------------------------------------------------------

def infer_image(model: Patchcore, img_path: Path):
    """
    對單張圖片執行 PatchCore 推論。
    回傳 (raw_score: float, anomaly_map: np.ndarray[H, W])
    """
    transform = model.pre_processor.transform
    device    = next(model.parameters()).device
    img = Image.open(img_path).convert("RGB")
    t   = to_tensor(img)
    t   = transform(t).to(device)
    batch = ImageBatch(image=t.unsqueeze(0))
    with torch.no_grad():
        out = model.predict_step(batch, 0)
    score = out.pred_score[0].item()
    amap  = out.anomaly_map[0].cpu().numpy()   # shape (H, W)
    return score, amap


# ---------------------------------------------------------------------------
# anomaly_map 轉換函式
# ---------------------------------------------------------------------------

def anomaly_map_to_text(
    anomaly_map: np.ndarray,
    anomaly_score: float,
    img_min: float,
    img_max: float,
    pixel_min: float,
    pixel_max: float,
    threshold: float = 0.5,
) -> str:
    """把 anomaly_map 分成 3×3 九宮格，回傳繁體中文文字描述。"""
    norm_score = float(np.clip((anomaly_score - img_min) / (img_max - img_min + 1e-8), 0, 1))

    H, W = anomaly_map.shape
    gh, gw = H // 3, W // 3
    cells: dict[str, float] = {}
    for ri in range(3):
        for ci in range(3):
            patch    = anomaly_map[ri * gh : (ri + 1) * gh, ci * gw : (ci + 1) * gw]
            raw_mean = float(patch.mean())
            norm_v   = float(np.clip((raw_mean - pixel_min) / (pixel_max - pixel_min + 1e-8), 0, 1))
            cells[_GRID_LABELS[ri][ci]] = norm_v

    sorted_cells = sorted(cells.items(), key=lambda x: x[1], reverse=True)
    high = [(l, v) for l, v in sorted_cells if v >= threshold]
    low  = [(l, v) for l, v in sorted_cells if v <  threshold]

    lines = [f"PatchCore 偵測結果（整體異常分數：{norm_score:.2f}）"]
    if high:
        lines.append("高異常區域：" + "、".join(f"{l}（{v:.2f}）" for l, v in high))
    if low:
        lines.append("低異常區域：" + "、".join(f"{l}（{v:.2f}）" for l, v in low[:4]))
    if high:
        focus = "、".join(l for l, _ in sorted_cells[:3])
        lines.append(f"請重點檢查：{focus}區域")
    return "\n".join(lines)


def anomaly_map_to_image(
    anomaly_map: np.ndarray,
    pixel_min: float,
    pixel_max: float,
) -> Image.Image:
    """把 anomaly_map 用 jet colormap 轉成 PIL.Image（藍=正常，紅=異常）。"""
    norm = np.clip((anomaly_map - pixel_min) / (pixel_max - pixel_min + 1e-8), 0, 1)
    rgba = cm.jet(norm)
    rgb  = (rgba[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb)


def make_side_by_side(orig: Image.Image, heatmap: Image.Image) -> Image.Image:
    """把原圖和熱力圖並排，組成雙格 PIL.Image。"""
    w, h    = orig.size
    canvas  = Image.new("RGB", (w * 2, h))
    canvas.paste(orig,    (0, 0))
    canvas.paste(heatmap.resize((w, h)), (w, 0))
    return canvas


# ---------------------------------------------------------------------------
# 載入測試集
# ---------------------------------------------------------------------------

def collect_test_items(category: str) -> list[dict]:
    """收集測試集中所有圖片，附帶 gt_class 標籤。"""
    test_root = LOCO_ROOT / category / "test"
    items: list[dict] = []
    for subdir, gt_class in [
        ("good",               "good"),
        ("logical_anomalies",  "logical"),
        ("structural_anomalies", "structural"),
    ]:
        folder = test_root / subdir
        if not folder.exists():
            continue
        for p in sorted(folder.glob("*.png")):
            items.append({"path": p, "stem": p.stem, "gt_class": gt_class})
    return items


# ---------------------------------------------------------------------------
# 評估指標
# ---------------------------------------------------------------------------

def compute_metrics(records: list[dict]) -> dict:
    """計算整體及各子類別的 F1、Precision、Recall。"""
    def _prf(tp, fp, fn):
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return prec, rec, f1

    # --- 整體 anomaly vs. normal ---
    tp = sum(1 for r in records if r["gt_class"] != "good" and r["is_anomaly"])
    fp = sum(1 for r in records if r["gt_class"] == "good" and r["is_anomaly"])
    fn = sum(1 for r in records if r["gt_class"] != "good" and not r["is_anomaly"])
    prec, rec, f1 = _prf(tp, fp, fn)
    out = {"overall": {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}}

    # --- logical anomaly ---
    lg = [r for r in records if r["gt_class"] == "logical"]
    if lg:
        tp_l = sum(1 for r in lg if r["is_anomaly"])
        fn_l = sum(1 for r in lg if not r["is_anomaly"])
        prec_l, rec_l, f1_l = _prf(tp_l, 0, fn_l)
        out["logical"] = {"precision": prec_l, "recall": rec_l, "f1": f1_l, "n": len(lg), "tp": tp_l, "fn": fn_l}

    # --- structural anomaly ---
    st = [r for r in records if r["gt_class"] == "structural"]
    if st:
        tp_s = sum(1 for r in st if r["is_anomaly"])
        fn_s = sum(1 for r in st if not r["is_anomaly"])
        prec_s, rec_s, f1_s = _prf(tp_s, 0, fn_s)
        out["structural"] = {"precision": prec_s, "recall": rec_s, "f1": f1_s, "n": len(st), "tp": tp_s, "fn": fn_s}

    # --- good image FP rate ---
    gd = [r for r in records if r["gt_class"] == "good"]
    if gd:
        fp_g = sum(1 for r in gd if r["is_anomaly"])
        out["good_fp_rate"] = fp_g / len(gd)
        out["good_fp_count"] = fp_g
        out["good_total"]    = len(gd)

    return out


def print_metrics(category: str, metrics: dict) -> None:
    """格式化輸出評估指標。"""
    ov = metrics["overall"]
    print(f"\n{'='*55}")
    print(f"[{category}] 評估結果")
    print(f"{'='*55}")
    print(f"  整體  F1={ov['f1']:.3f}  Prec={ov['precision']:.3f}  Rec={ov['recall']:.3f}"
          f"  (TP={ov['tp']}, FP={ov['fp']}, FN={ov['fn']})")
    if "logical" in metrics:
        lg = metrics["logical"]
        print(f"  邏輯  F1={lg['f1']:.3f}  Rec={lg['recall']:.3f}  ({lg['tp']}/{lg['n']})")
    if "structural" in metrics:
        st = metrics["structural"]
        print(f"  結構  F1={st['f1']:.3f}  Rec={st['recall']:.3f}  ({st['tp']}/{st['n']})")
    if "good_fp_rate" in metrics:
        print(f"  正常品 FP率={metrics['good_fp_rate']:.3f}"
              f"  ({metrics['good_fp_count']}/{metrics['good_total']})")
    print()


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run_category(category: str, mode: str, client, jsonl_path: Path,
                 test_only: bool = False, anomaly_only: bool = False) -> list[dict]:
    """對單一類別執行 PatchCore → VLM 管線。"""
    print(f"\n{'='*55}")
    print(f"PatchCore-VLM  [{mode}]  類別：{category}")
    print(f"{'='*55}")

    # 載入 PatchCore
    model, ckpt = load_patchcore(category)
    img_min   = float(model.post_processor.image_min)
    img_max   = float(model.post_processor.image_max)
    pixel_min = float(model.post_processor.pixel_min)
    pixel_max = float(model.post_processor.pixel_max)
    threshold = float(model.post_processor._image_threshold)
    print(f"  checkpoint：{ckpt.parent.parent.parent.name}")
    print(f"  threshold  ：{threshold:.4f}  (min={img_min:.2f}, max={img_max:.2f})")

    # 載入已完成記錄，用 (stem, gt_class) 避免不同子集的同名 stem 衝突
    completed: set[tuple[str, str]] = set()
    all_records: list[dict] = []
    if jsonl_path.exists():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                r = json.loads(line)
                if r.get("category") == category:
                    completed.add((r["stem"], r["gt_class"]))
                    all_records.append(r)

    items = collect_test_items(category)
    if anomaly_only:
        items = [it for it in items if it["gt_class"] != "good"]
    if test_only:
        # 每個子集只取第一張
        seen: set[str] = set()
        test_items: list[dict] = []
        for it in items:
            if it["gt_class"] not in seen:
                test_items.append(it)
                seen.add(it["gt_class"])
        items = test_items
    todo  = [it for it in items if (it["stem"], it["gt_class"]) not in completed]
    print(f"  測試集：{len(items)} 張，待處理：{len(todo)} 張\n")

    t0 = time.time()
    new_records: list[dict] = []
    for i, item in enumerate(todo):
        img_path = item["path"]
        stem     = item["stem"]
        gt_class = item["gt_class"]

        # PatchCore 推論
        raw_score, amap = infer_image(model, img_path)
        norm_score = float(np.clip((raw_score - img_min) / (img_max - img_min + 1e-8), 0, 1))
        patchcore_positive = raw_score >= threshold

        orig_img = Image.open(img_path).convert("RGB")

        # 根據 mode 建立送給 Gemma 的 patchcore_text
        patchcore_text = anomaly_map_to_text(
            amap, raw_score, img_min, img_max, pixel_min, pixel_max
        )
        if mode == "heatmap":
            heatmap = anomaly_map_to_image(amap, pixel_min, pixel_max)
            heatmap_desc = f"（附圖右半部為 PatchCore 熱力圖，藍色=正常，紅色=異常）"
            patchcore_text = patchcore_text + "\n" + heatmap_desc
            send_img = make_side_by_side(orig_img, heatmap)
        else:
            send_img = orig_img

        # Gemma 推論
        result = gemini_inference_patchcore(
            image=send_img,
            patchcore_text=patchcore_text,
            category=category,
            client=client,
        )

        logical_score    = result.get("logical_score", 10)
        structural_score = result.get("structural_score", 10)
        vlm_score        = round(max(logical_score, structural_score) / 100, 4)

        record = {
            "category":              category,
            "stem":                  stem,
            "gt_class":              gt_class,
            "is_anomaly":            result["is_anomaly"],
            "anomaly_type":          result["anomaly_type"],
            "logical_score":         logical_score,
            "structural_score":      structural_score,
            "vlm_score":             vlm_score,
            "reason":                result["reason"],
            "raw_score":             round(raw_score, 4),
            "norm_score":            round(norm_score, 4),
            "patchcore_above_threshold": patchcore_positive,
            "mode":                  mode,
        }
        all_records.append(record)
        new_records.append(record)

        verdict_str = "異常" if result["is_anomaly"] else "正常"
        gt_str      = {"good": "正常品", "logical": "邏輯異常", "structural": "結構異常"}[gt_class]
        correct     = (result["is_anomaly"]) == (gt_class != "good")
        tag         = "✓" if correct else "✗"
        elapsed = time.time() - t0
        avg     = elapsed / (i + 1)
        print(f"[{i+1:3d}/{len(todo)}] {stem} ({gt_str}) → {verdict_str}  score={norm_score:.2f}  {tag}"
              f"  [{elapsed/60:.1f}min  avg {avg:.1f}s/張]")
        print(f"   PatchCore: {norm_score:.2f}  VLM: log={logical_score}  str={structural_score}  vlm={vlm_score:.2f}")
        print(f"   reason: {result['reason'][:100]}")

        # 即時寫入
        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    cat_records = [r for r in all_records if r.get("category") == category]
    metrics = compute_metrics(cat_records)
    print_metrics(category, metrics)
    return cat_records


def main() -> None:
    run_start = time.time()
    parser = argparse.ArgumentParser(description="PatchCore + VLM 整合推論管線")
    parser.add_argument("--category", nargs="+", default=None,
                        help="指定類別（預設：所有 5 個類別）")
    parser.add_argument("--mode", choices=["text", "heatmap"], default="text",
                        help="text（文字描述）或 heatmap（熱力圖）")
    parser.add_argument("--test-only", action="store_true",
                        help="只跑每類別每子集第 1 張（logical/good/structural 各 1 張）用於快速測試")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="清除既有結果，從頭重跑所有圖片",
    )
    parser.add_argument(
        "--anomaly-only",
        action="store_true",
        dest="anomaly_only",
        help="只跑異常圖片（149 張），跳過 good 圖片",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        dest="retry_errors",
        help="只重跑上次因 API 錯誤失敗的圖片（reason 含 api_error 或 parse_error）",
    )
    args = parser.parse_args()

    categories = args.category or CATEGORIES
    unknown = [c for c in categories if c not in CATEGORIES]
    if unknown:
        print(f"未知類別：{unknown}")
        sys.exit(1)

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("請先設定環境變數 GEMINI_API_KEY")
        sys.exit(1)
    client = genai.Client(api_key=api_key)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = RESULTS_DIR / f"patchcore_vlm_{args.mode}_results.jsonl"

    # --reset：刪除既有結果從頭跑
    if args.reset and jsonl_path.exists():
        jsonl_path.unlink()
        print(f"已清除既有結果：{jsonl_path.name}")

    # --retry-errors：只保留非錯誤記錄，讓錯誤圖片重跑
    if args.retry_errors and jsonl_path.exists():
        all_lines = [l for l in jsonl_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        clean, error_keys = [], set()
        for line in all_lines:
            r = json.loads(line)
            reason = r.get("reason", "")
            if "api_error" in reason or reason == "parse_error":
                error_keys.add((r["stem"], r.get("gt_class", "")))
            else:
                clean.append(line)
        if not error_keys:
            print("JSONL 中沒有錯誤記錄，無需重跑。")
        else:
            jsonl_path.write_text("\n".join(clean) + "\n", encoding="utf-8")
            print(f"找到 {len(error_keys)} 筆錯誤記錄，重跑這些圖片…")

    print(f"輸出檔案：{jsonl_path.name}")
    print(f"模式    ：{args.mode}")
    if args.anomaly_only:
        print("模式    ：只跑異常圖片（跳過 good）")

    all_records: list[dict] = []
    for cat in categories:
        records = run_category(
            cat, args.mode, client, jsonl_path,
            test_only=args.test_only,
            anomaly_only=args.anomaly_only,
        )
        all_records.extend(records)

    total_elapsed = time.time() - run_start
    print(f"\n總花費時間：{total_elapsed/60:.1f} 分鐘（{total_elapsed:.0f} 秒）")

    if len(categories) > 1:
        print(f"\n{'='*55}")
        print(f"全部 {len(categories)} 個類別合計")
        metrics = compute_metrics(all_records)
        print_metrics("ALL", metrics)


if __name__ == "__main__":
    main()
