#!/usr/bin/env python3
"""
patchcore_vlm_pipeline_mvtecad.py

MVTec AD 版 PatchCore + VLM 評估管線（沿用 LQ/SQ + 計數的共用推理函式）。
與 LOCO 版差異：
  - 路徑指向 mvtec_ad / patchcore_models_mvtecad / auto_questions_mvtecad
  - test/ 下自動掃描所有異常類型資料夾，gt_class = 資料夾名
  - 評估只算 overall（good vs 所有異常）+ 各異常類型 Recall + good FP 率
最終判斷：combined_score = patchcore_norm_score * 0.5 + vlm_score * 0.5，>= 0.5 判為異常。

使用方式：
    python patchcore_vlm_pipeline_mvtecad.py --category screw --mode text
    python patchcore_vlm_pipeline_mvtecad.py --mode text                          # 跑全部 15 類
    python patchcore_vlm_pipeline_mvtecad.py --mode text --backbone dinov2 --layers 3,9
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
import gemini_inference
import vlm_counting
from gemini_inference import gemini_inference_patchcore_lqsq_counting
from vlm_counting import vlm_count

# ---------------------------------------------------------------------------
# 路徑常數
# ---------------------------------------------------------------------------

MVTECAD_ROOT  = Path.home() / "data/iad/mvtec_ad"
MODELS_DIR    = Path.home() / "data/iad/code/results/patchcore_models_mvtecad"
QUESTIONS_DIR = Path.home() / "data/iad/code/results/auto_questions_mvtecad"
RESULTS_DIR   = Path.home() / "data/iad/code/results"

# 重指向共用模組的問題/計數規格書目錄到 MVTec AD 版（不修改現有檔案，只在執行時覆寫模組常數）
# gemini_inference 的 _load_auto_questions_lq_sq / _load_comad_spec、vlm_counting 的
# load_counting_spec 都在呼叫時才查各自的模組常數，故此覆寫生效。
gemini_inference._AUTO_QUESTIONS_DIR = QUESTIONS_DIR
vlm_counting.SPEC_DIR = QUESTIONS_DIR

ALL_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]
CATEGORIES = ALL_CATEGORIES

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
    amap  = out.anomaly_map[0].cpu().numpy()
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
    w, h   = orig.size
    canvas = Image.new("RGB", (w * 2, h))
    canvas.paste(orig,                  (0, 0))
    canvas.paste(heatmap.resize((w, h)), (w, 0))
    return canvas


# ---------------------------------------------------------------------------
# 載入測試集
# ---------------------------------------------------------------------------

def collect_test_items(category: str) -> list[dict]:
    """收集測試集中所有圖片；MVTec AD 的 gt_class 是各異常類型資料夾名稱。"""
    test_dir = MVTECAD_ROOT / category / "test"
    items: list[dict] = []
    for subset_dir in sorted(test_dir.iterdir()):
        if not subset_dir.is_dir():
            continue
        gt_class = subset_dir.name  # "good", "scratch_head", "thread_side", ...
        for img_path in sorted(subset_dir.glob("*.png")):
            items.append({
                "path":       img_path,
                "stem":       img_path.stem,
                "gt_class":   gt_class,
                "is_anomaly": gt_class != "good",
            })
    return items


# ---------------------------------------------------------------------------
# 評估指標
# ---------------------------------------------------------------------------

def _predicted_anomaly(r: dict) -> bool:
    """預測是否異常：combined_score >= 0.5（record 的 is_anomaly 存的是 ground truth）。"""
    return r["combined_score"] >= 0.5


def compute_metrics(records: list[dict]) -> dict:
    """MVTec AD：只算 overall（good vs 所有異常）+ 各異常類型 Recall + good FP 率。"""
    def _prf(tp, fp, fn):
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return prec, rec, f1

    tp = sum(1 for r in records if r["gt_class"] != "good" and _predicted_anomaly(r))
    fp = sum(1 for r in records if r["gt_class"] == "good" and _predicted_anomaly(r))
    fn = sum(1 for r in records if r["gt_class"] != "good" and not _predicted_anomaly(r))
    prec, rec, f1 = _prf(tp, fp, fn)
    out = {"overall": {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}}

    # 各異常類型的 Recall（gt_class 為各瑕疵名稱）
    defect_types = sorted({r["gt_class"] for r in records if r["gt_class"] != "good"})
    per_type = {}
    for t in defect_types:
        sub = [r for r in records if r["gt_class"] == t]
        tp_t = sum(1 for r in sub if _predicted_anomaly(r))
        per_type[t] = {"recall": tp_t / len(sub) if sub else 0.0, "tp": tp_t, "n": len(sub)}
    out["per_type"] = per_type

    gd = [r for r in records if r["gt_class"] == "good"]
    if gd:
        fp_g = sum(1 for r in gd if _predicted_anomaly(r))
        out["good_fp_rate"]  = fp_g / len(gd)
        out["good_fp_count"] = fp_g
        out["good_total"]    = len(gd)

    return out


def print_metrics(category: str, metrics: dict) -> None:
    """格式化輸出評估指標（MVTec AD：overall + 各異常類型 Recall + good FP 率）。"""
    ov = metrics["overall"]
    print(f"\n{'='*55}")
    print(f"[{category}] 評估結果（MVTec AD, Combined）")
    print(f"{'='*55}")
    print(f"  整體  F1={ov['f1']:.3f}  Prec={ov['precision']:.3f}  Rec={ov['recall']:.3f}"
          f"  (TP={ov['tp']}, FP={ov['fp']}, FN={ov['fn']})")
    for t, m in metrics.get("per_type", {}).items():
        print(f"  {t:<20} Recall={m['recall']:.3f}  ({m['tp']}/{m['n']})")
    if "good_fp_rate" in metrics:
        print(f"  正常品 FP率={metrics['good_fp_rate']:.3f}"
              f"  ({metrics['good_fp_count']}/{metrics['good_total']})")
    print()


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run_category(category: str, mode: str, client, jsonl_path: Path,
                 test_only: bool = False, backbone: str = "resnet",
                 layers_str: str = "11") -> list[dict]:
    """對單一類別執行 PatchCore → VLM → Combined 管線（MVTec AD）。"""
    print(f"\n{'='*55}")
    print(f"PatchCore-VLM MVTec AD [{mode}]  類別：{category}")
    print(f"{'='*55}")

    if backbone == "dinov2":
        layer_tag = layers_str.replace(",", "_")
        subdir = f"dinov2_{layer_tag}/Patchcore"
    else:
        subdir = "Patchcore"
    model, ckpt = load_patchcore(category, subdir=subdir)
    img_min   = float(model.post_processor.image_min)
    img_max   = float(model.post_processor.image_max)
    pixel_min = float(model.post_processor.pixel_min)
    pixel_max = float(model.post_processor.pixel_max)
    threshold = float(model.post_processor._image_threshold)
    print(f"  checkpoint：{ckpt.parent.parent.parent.name}")
    print(f"  threshold  ：{threshold:.4f}  (min={img_min:.2f}, max={img_max:.2f})")

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
    if test_only:
        seen: set[str] = set()
        test_items: list[dict] = []
        for it in items:
            if it["gt_class"] not in seen:
                test_items.append(it)
                seen.add(it["gt_class"])
        items = test_items
    todo = [it for it in items if (it["stem"], it["gt_class"]) not in completed]
    print(f"  測試集：{len(items)} 張，待處理：{len(todo)} 張\n")

    t0 = time.time()
    new_records: list[dict] = []
    for i, item in enumerate(todo):
        img_path = item["path"]
        stem     = item["stem"]
        gt_class = item["gt_class"]

        # PatchCore 推論
        raw_score, amap = infer_image(model, img_path)
        patchcore_norm = float(np.clip((raw_score - img_min) / (img_max - img_min + 1e-8), 0, 1))
        patchcore_positive = raw_score >= threshold

        orig_img = Image.open(img_path).convert("RGB")

        # 計數分析（原圖）；該類別無可數物件時回 None，推理函式會退回一般 LQ/SQ prompt
        counting_text = vlm_count(orig_img, category, client)

        patchcore_text = anomaly_map_to_text(
            amap, raw_score, img_min, img_max, pixel_min, pixel_max
        )
        if mode == "heatmap":
            heatmap = anomaly_map_to_image(amap, pixel_min, pixel_max)
            heatmap_desc = "（附圖右半部為 PatchCore 熱力圖，藍色=正常，紅色=異常）"
            patchcore_text = patchcore_text + "\n" + heatmap_desc
            send_img = make_side_by_side(orig_img, heatmap)
        else:
            send_img = orig_img

        result = gemini_inference_patchcore_lqsq_counting(
            image=send_img,
            patchcore_text=patchcore_text,
            counting_text=counting_text,
            category=category,
            client=client,
        )

        logical_score    = result.get("logical_score", 10)
        structural_score = result.get("structural_score", 10)
        vlm_score        = round(max(logical_score, structural_score) / 100, 4)

        # Combined 判斷（predicted）
        combined_score  = round(patchcore_norm * 0.5 + vlm_score * 0.5, 4)
        pred_anomaly    = combined_score >= 0.5
        gt_is_anomaly   = gt_class != "good"

        record = {
            "category":             category,
            "stem":                 stem,
            "gt_class":             gt_class,
            "is_anomaly":           gt_is_anomaly,           # ground truth（good=False，其他=True）
            "patchcore_norm_score": round(patchcore_norm, 4),
            "vlm_logical_score":    logical_score,
            "vlm_structural_score": structural_score,
            "vlm_score":            vlm_score,
            "combined_score":       combined_score,
            "norm_score":           combined_score,          # 供 AUROC 計算用
            "counting_text":        counting_text,
            "reason":               result["reason"],
        }
        all_records.append(record)
        new_records.append(record)

        verdict_str = "異常" if pred_anomaly else "正常"
        gt_str      = "正常品" if gt_class == "good" else gt_class
        correct     = pred_anomaly == gt_is_anomaly
        tag         = "✓" if correct else "✗"
        elapsed     = time.time() - t0
        avg         = elapsed / (i + 1)
        print(f"[{i+1:3d}/{len(todo)}] {stem} ({gt_str}) → {verdict_str}"
              f"  combined={combined_score:.2f}  {tag}"
              f"  [{elapsed/60:.1f}min  avg {avg:.1f}s/張]")
        print(f"   PatchCore: {patchcore_norm:.2f}  VLM: log={logical_score}  str={structural_score}"
              f"  vlm={vlm_score:.2f}  combined={combined_score:.2f}")
        if counting_text is not None:
            print(f"   {counting_text}")
        print(f"   reason: {result['reason'][:100]}")

        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    cat_records = [r for r in all_records if r.get("category") == category]
    metrics = compute_metrics(cat_records)
    print_metrics(category, metrics)
    return cat_records


def main() -> None:
    run_start = time.time()
    parser = argparse.ArgumentParser(description="PatchCore + VLM MVTec AD 評估管線")
    parser.add_argument("--category", nargs="+", default=None,
                        help="指定類別（預設：所有 15 個類別）")
    parser.add_argument("--mode", choices=["text", "heatmap"], default="text",
                        help="text（文字描述）或 heatmap（熱力圖）")
    parser.add_argument("--test-only", action="store_true",
                        help="每個子集（good 與各異常類型）只跑第 1 張，用於快速測試")
    parser.add_argument("--reset", action="store_true",
                        help="清除既有結果，從頭重跑所有圖片")
    parser.add_argument("--retry-errors", action="store_true", dest="retry_errors",
                        help="重跑結果中標記為錯誤（parse_error、timeout、API error）的項目")
    parser.add_argument("--backbone", type=str, default="resnet",
                        choices=["resnet", "dinov2"],
                        help="使用哪個 backbone 的 checkpoint，預設 resnet")
    parser.add_argument("--layers", type=str, default="11",
                        help="DINOv2 使用的 blocks，逗號分隔，例如 '11' 或 '3,9'，ResNet 時忽略")
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
    suffix = f"_dinov2_{args.layers.replace(',', '_')}" if args.backbone == "dinov2" else ""
    jsonl_path = RESULTS_DIR / f"patchcore_vlm_mvtecad{suffix}_results.jsonl"

    if args.reset and jsonl_path.exists():
        jsonl_path.unlink()
        print(f"已清除既有結果：{jsonl_path.name}")

    # --retry-errors：把 reason 含 parse_error/timeout/error 的記錄從 JSONL 移除，
    # 其餘保留；被移除的圖片會因不在 completed 集合而被重新推理，其他記錄不動。
    if args.retry_errors and jsonl_path.exists():
        all_lines = [l for l in jsonl_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        clean, error_keys = [], set()
        for line in all_lines:
            r = json.loads(line)
            reason = str(r.get("reason", "")).lower()
            if any(kw in reason for kw in ("parse_error", "timeout", "error")):
                error_keys.add((r.get("category", ""), r["stem"], r.get("gt_class", "")))
            else:
                clean.append(line)
        if not error_keys:
            print("JSONL 中沒有錯誤記錄，無需重跑（0 筆）。")
        else:
            jsonl_path.write_text("\n".join(clean) + "\n", encoding="utf-8")
            print(f"找到 {len(error_keys)} 筆錯誤記錄，重跑這些圖片…")

    print(f"輸出檔案：{jsonl_path.name}")
    print(f"模式    ：{args.mode}")
    layers_info = f"  layers={args.layers}" if args.backbone == "dinov2" else ""
    print(f"Backbone ：{args.backbone}{layers_info}")

    all_records: list[dict] = []
    for cat in categories:
        records = run_category(
            cat, args.mode, client, jsonl_path,
            test_only=args.test_only,
            backbone=args.backbone,
            layers_str=args.layers,
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
