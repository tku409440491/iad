#!/usr/bin/env python3
"""
evaluate_pipeline.py

對完整 test set 執行 YOLO → draw_detections → Gemini 完整 pipeline，
計算 Precision / Recall / F1（整體 + structural + logical）。

執行方式：
    python evaluate_pipeline.py

支援斷點續跑：
    結果逐行寫入 results/pipeline_results.jsonl，
    重新執行時自動跳過已處理的圖片。

輸出：
    results/pipeline_results.jsonl  ── 每張圖一筆 JSON 記錄
    results/evaluation_report.json  ── 最終彙總指標
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

from google import genai
from ultralytics import YOLO

# 同目錄的 gemini_inference 模組
sys.path.insert(0, str(Path(__file__).parent))
from gemini_inference import draw_detections, gemini_inference


# ---------------------------------------------------------------------------
# 路徑 / 參數設定
# ---------------------------------------------------------------------------

TEST_IMG_DIR = Path.home() / "data/iad/yolo_dataset/images/test"
BEST_PT      = Path.home() / "data/iad/code/runs/yolo_baseline/weights/best.pt"
RESULTS_DIR  = Path.home() / "data/iad/code/results"
LOCO_ROOT    = Path.home() / "data/iad/mvtec_loco"
JSONL_PATH   = RESULTS_DIR / "pipeline_results.jsonl"
REPORT_PATH  = RESULTS_DIR / "evaluation_report.json"

YOLO_CONF  = 0.10
YOLO_IOU   = 0.50
CATEGORIES = [
    "breakfast_box", "juice_bottle", "pushpins",
    "screw_bag", "splicing_connectors",
]

# GT class 簡寫對照（symlink 檔名中的部分）
_GT_MAP = {"log": "logical", "str": "structural"}


# ---------------------------------------------------------------------------
# YOLO 推理（使用預載模型，不每張圖重新載入）
# ---------------------------------------------------------------------------

def run_yolo(model: YOLO, image_path: Path) -> list[dict]:
    """對單張圖片執行 YOLO 推理，回傳像素座標格式的偵測框列表。"""
    results = model.predict(
        source=str(image_path),
        conf=YOLO_CONF,
        iou=YOLO_IOU,
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


# ---------------------------------------------------------------------------
# 解析 symlink 檔名
# ---------------------------------------------------------------------------

def parse_stem(stem: str) -> tuple[str, str, str]:
    """
    解析 symlink 檔名，回傳 (category, gt_class, img_id)。

    格式：{category}__{log|str}__{img_id}
    範例：breakfast_box__log__000 → ("breakfast_box", "logical", "000")
    """
    parts = stem.split("__")
    if len(parts) != 3:
        raise ValueError(f"無法解析檔名：{stem}（預期格式：category__log|str__id）")
    category, gt_short, img_id = parts
    gt_class = _GT_MAP.get(gt_short)
    if gt_class is None:
        raise ValueError(f"未知 GT 簡寫：'{gt_short}'（來源：{stem}）")
    return category, gt_class, img_id


# ---------------------------------------------------------------------------
# 建立完整圖片清單（異常圖片 + good 圖片）
# ---------------------------------------------------------------------------

def build_image_list() -> list[dict]:
    """
    回傳按類別排序的圖片清單，每個類別的異常圖片後接 good 圖片。

    每個元素：{"path": Path, "stem": str, "category": str, "gt_class": str, "img_id": str}
    Good 圖片的 stem 格式：{category}__good__{img_id}
    """
    cat_items: dict[str, list[dict]] = {c: [] for c in CATEGORIES}

    # 異常圖片（來自 yolo_dataset 的 symlinks）
    for img_path in sorted(TEST_IMG_DIR.glob("*.png")):
        stem = img_path.stem
        try:
            category, gt_class, img_id = parse_stem(stem)
            if category in cat_items:
                cat_items[category].append({
                    "path": img_path, "stem": stem,
                    "category": category, "gt_class": gt_class, "img_id": img_id,
                })
        except ValueError:
            pass

    # Good 圖片（來自 mvtec_loco/{category}/test/good/）
    for category in CATEGORIES:
        good_dir = LOCO_ROOT / category / "test" / "good"
        for img_path in sorted(good_dir.glob("*.png")):
            img_id = img_path.stem
            stem   = f"{category}__good__{img_id}"
            cat_items[category].append({
                "path": img_path, "stem": stem,
                "category": category, "gt_class": "good", "img_id": img_id,
            })

    # 按 CATEGORIES 順序合併，同類別內保持路徑排序
    return [item for c in CATEGORIES for item in cat_items[c]]


# ---------------------------------------------------------------------------
# 斷點續跑：讀取已完成的 stems
# ---------------------------------------------------------------------------

def load_completed_records(jsonl_path: Path) -> tuple[set[str], list[dict]]:
    """
    讀取既有 JSONL，回傳已完成的 stem 集合與記錄列表。
    """
    completed: set[str]  = set()
    records:   list[dict] = []

    if not jsonl_path.exists():
        return completed, records

    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            completed.add(rec["stem"])
            records.append(rec)
        except (json.JSONDecodeError, KeyError):
            pass

    return completed, records


# ---------------------------------------------------------------------------
# 計算指標（Precision / Recall / F1）
# ---------------------------------------------------------------------------

def compute_metrics(records: list[dict]) -> dict:
    """
    根據所有記錄計算 TP/FP/FN/TN 並推算 P/R/F1。

    異常圖片（gt_class in {logical, structural}）：
        is_anomaly=True  AND anomaly_type == gt_class → TP（該類別）
        is_anomaly=True  AND anomaly_type != gt_class → FP（預測類別）、FN（GT 類別）
        is_anomaly=False OR  anomaly_type='none'      → FN（GT 類別）

    Good 圖片（gt_class == "good"）：
        is_anomaly=False → TN（正確辨識為正常品）
        is_anomaly=True  → FP（誤報，納入整體 fp_total）

    整體 Precision = TP / (TP + FP)，FP 包含異常誤分類 + good 誤報。
    整體 Recall    = TP / (TP + FN)，FN 只算異常圖片未偵測到。
    """
    tp_cls: dict[str, int] = defaultdict(int)
    fp_cls: dict[str, int] = defaultdict(int)
    fn_cls: dict[str, int] = defaultdict(int)
    tp_total = fp_total = fn_total = tn_total = 0
    fp_good  = 0  # good 圖片被誤判為異常的數量

    for r in records:
        gt_cls = r["gt_class"]

        if gt_cls == "good":
            if r.get("is_anomaly"):
                fp_good  += 1
                fp_total += 1
            else:
                tn_total += 1
        else:
            pred_cls = r["anomaly_type"] if r.get("is_anomaly") else None
            if pred_cls == gt_cls:
                tp_cls[gt_cls] += 1
                tp_total       += 1
            elif pred_cls in ("logical", "structural"):
                # 預測了錯誤的類別
                fp_cls[pred_cls] += 1
                fn_cls[gt_cls]   += 1
                fp_total         += 1
            else:
                # is_anomaly=False、anomaly_type=none、api_error 等
                fn_cls[gt_cls] += 1
                fn_total       += 1

    def _prf(t: int, fpos: int, fneg: int, tn: int = 0) -> dict:
        p  = t / (t + fpos) if (t + fpos) > 0 else 0.0
        r  = t / (t + fneg) if (t + fneg) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        result = {
            "precision": round(p,  4),
            "recall":    round(r,  4),
            "f1":        round(f1, 4),
            "tp": t, "fp": fpos, "fn": fneg,
        }
        if tn > 0:
            result["tn"] = tn
        return result

    return {
        "overall":    _prf(tp_total, fp_total, fn_total, tn_total),
        "logical":    _prf(tp_cls["logical"],    fp_cls["logical"],    fn_cls["logical"]),
        "structural": _prf(tp_cls["structural"], fp_cls["structural"], fn_cls["structural"]),
        "good":       {"fp": fp_good, "tn": tn_total},
    }


# ---------------------------------------------------------------------------
# 主程式
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO + Gemini pipeline 評估")
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

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise EnvironmentError("請先設定環境變數 GEMINI_API_KEY（export GEMINI_API_KEY=...）")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── reset：刪除既有結果 ────────────────────────────────────────────────
    if args.reset:
        for path in (JSONL_PATH, REPORT_PATH):
            if path.exists():
                path.unlink()
                print(f"已刪除：{path}")
        print("重置完成，從頭開始跑。\n")

    # ── 載入模型 ──────────────────────────────────────────────────────────
    print("載入 YOLO 模型…")
    yolo_model = YOLO(str(BEST_PT))

    print("初始化 Gemini client…")
    client = genai.Client(api_key=api_key)

    # ── 建立完整圖片清單（異常圖片 + good 圖片，按類別分組）─────────────────
    all_items = build_image_list()
    if args.anomaly_only:
        all_items = [it for it in all_items if it["gt_class"] != "good"]
    total     = len(all_items)
    n_good    = sum(1 for it in all_items if it["gt_class"] == "good")
    n_anomaly = total - n_good
    if args.anomaly_only:
        print(f"Test set：{total} 張圖片（僅異常）")
    else:
        print(f"Test set：{total} 張圖片（異常 {n_anomaly} + good {n_good}）")

    # ── 斷點續跑 ────────────────────────────────────────────────────────────
    completed, all_records = load_completed_records(JSONL_PATH)
    if completed:
        print(f"已完成 {len(completed)} 張，跳過繼續…\n")

    # ── retry-errors：只重跑錯誤記錄 ────────────────────────────────────────
    if args.retry_errors:
        error_stems = {
            r["stem"] for r in all_records
            if "api_error" in r.get("reason", "") or r.get("reason") == "parse_error"
        }
        if not error_stems:
            print("JSONL 中沒有錯誤記錄，無需重跑。")
            return
        # 把錯誤記錄從 JSONL 移除，讓主迴圈重新寫入
        clean_records = [r for r in all_records if r["stem"] not in error_stems]
        JSONL_PATH.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in clean_records) + "\n",
            encoding="utf-8",
        )
        all_records = clean_records
        completed   = {r["stem"] for r in clean_records}
        # 只保留 error stems 對應的圖片
        all_items = [it for it in all_items if it["stem"] in error_stems]
        print(f"找到 {len(error_stems)} 筆錯誤記錄，重跑這些圖片…\n")

    # ── 主迴圈 ──────────────────────────────────────────────────────────────
    errors          = 0
    t_start         = time.time()
    cur_category    = None
    cat_buf: list[dict] = []  # 暫存當前類別的記錄，用於印小計

    def _print_cat_subtotal(cat: str, buf: list[dict]) -> None:
        """印出單一類別的 TP/FP/FN/TN 小計。"""
        tp = sum(1 for r in buf if r["verdict"] == "TP")
        fp = sum(1 for r in buf if r["verdict"] == "FP")
        fn = sum(1 for r in buf if r["verdict"] == "FN")
        tn = sum(1 for r in buf if r["verdict"] == "TN")
        print(
            f"  ── [{cat}] 小計：TP={tp}  FP={fp}  FN={fn}  TN={tn}  "
            f"（共 {len(buf)} 張）"
        )

    with JSONL_PATH.open("a", encoding="utf-8") as fout:
        for idx, item in enumerate(all_items, start=1):
            img_path = item["path"]
            stem     = item["stem"]
            category = item["category"]
            gt_class = item["gt_class"]
            img_id   = item["img_id"]

            # 切換類別時印上一個類別的小計
            if category != cur_category:
                if cur_category is not None and cat_buf:
                    _print_cat_subtotal(cur_category, cat_buf)
                    cat_buf = []
                cur_category = category

            # 跳過已完成
            if stem in completed:
                # 把舊記錄加回 cat_buf 供小計使用
                for r in all_records:
                    if r["stem"] == stem:
                        cat_buf.append(r)
                        break
                continue

            # YOLO 推理
            detections   = run_yolo(yolo_model, img_path)
            n_str        = sum(1 for d in detections if d["class_id"] == 0)
            n_log        = sum(1 for d in detections if d["class_id"] == 1)
            yolo_max_conf = max((d["conf"] for d in detections), default=0.0)

            # 畫框（good 圖片也送框；框為空時等同原圖）
            annotated = draw_detections(img_path, detections)

            # Gemini 推理（內含 time.sleep(5)；retry 等待 20s）
            result       = gemini_inference(annotated, detections, category, client)
            is_anomaly   = result["is_anomaly"]
            anomaly_type = result["anomaly_type"]
            reason       = result["reason"]

            # 判斷 TP / FP / FN / TN
            is_error = ("api_error" in reason) or (reason == "parse_error")
            if gt_class == "good":
                if is_error:
                    verdict = "FN"   # 保守處理：API 錯誤的 good 圖片不算 TN
                    errors += 1
                elif is_anomaly:
                    verdict = "FP"   # 正常品被誤判為異常
                else:
                    verdict = "TN"   # 正確判斷為正常品
            else:
                if is_error:
                    verdict = "FN"
                    errors += 1
                elif is_anomaly and anomaly_type == gt_class:
                    verdict = "TP"
                elif is_anomaly and anomaly_type in ("logical", "structural"):
                    verdict = "FP"
                else:
                    verdict = "FN"

            # 進度列
            icon = "✓" if verdict in ("TP", "TN") else "✗"
            if verdict == "TP":
                verdict_str = gt_class
            elif verdict == "TN":
                verdict_str = "TN（正常品正確）"
            elif verdict == "FP":
                if gt_class == "good":
                    verdict_str = f"FP→{anomaly_type}（GT=good）"
                else:
                    verdict_str = f"FP→{anomaly_type}（GT={gt_class}）"
            else:
                cause = "error" if is_error else f"not_detected({anomaly_type})"
                verdict_str = f"FN({cause})"

            gt_short = gt_class[:3] if gt_class != "good" else "goo"
            print(
                f"[{idx:>3}/{total}] {category}/{gt_short}/{img_id}.png  "
                f"{icon} {verdict_str}  "
                f"[yolo str={n_str} log={n_log} max={yolo_max_conf:.2f}]"
            )

            # 寫入 JSONL（每張圖立即 flush）
            record: dict = {
                "stem":          stem,
                "category":      category,
                "gt_class":      gt_class,
                "img_id":        img_id,
                "n_detections":  len(detections),
                "n_structural":  n_str,
                "n_logical":     n_log,
                "yolo_max_conf": round(yolo_max_conf, 4),
                "is_anomaly":    is_anomaly,
                "anomaly_type":  anomaly_type,
                "reason":        reason,
                "verdict":       verdict,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            all_records.append(record)
            completed.add(stem)
            cat_buf.append(record)

    # 最後一個類別的小計
    if cur_category and cat_buf:
        _print_cat_subtotal(cur_category, cat_buf)

    # ── 計算並輸出最終指標 ──────────────────────────────────────────────────
    runtime = time.time() - t_start
    metrics = compute_metrics(all_records)

    report: dict = {
        "overall":         metrics["overall"],
        "structural":      metrics["structural"],
        "logical":         metrics["logical"],
        "good":            metrics["good"],
        "total_images":    total,
        "n_anomaly":       n_anomaly,
        "n_good":          n_good,
        "processed":       len(all_records),
        "errors":          errors,
        "runtime_seconds": round(runtime, 1),
    }

    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ── 印出最終報告 ────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(
        f"評估完成  │  共 {total} 張（異常 {n_anomaly} + good {n_good}）  │  "
        f"錯誤 {errors} 張  │  耗時 {runtime:.0f}s"
    )
    print("=" * 65)
    for cls in ("overall", "structural", "logical"):
        m = metrics[cls]
        tn_str = f"  TN={m['tn']}" if "tn" in m else ""
        print(
            f"  {cls:<12}  "
            f"P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}  "
            f"(TP={m['tp']}  FP={m['fp']}  FN={m['fn']}{tn_str})"
        )
    g = metrics["good"]
    print(f"  {'good':<12}  FP（誤報）={g['fp']}  TN（正確）={g['tn']}")
    print(f"\n報告儲存：{REPORT_PATH}")


if __name__ == "__main__":
    main()
