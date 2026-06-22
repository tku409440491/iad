#!/usr/bin/env python3
"""
patchcore_vlm_pipeline_nopc.py

純 VLM pipeline，不使用 PatchCore。
直接將原圖 + 自動生成問題清單送給 Gemma，不計算 anomaly map。

使用方式：
    python patchcore_vlm_pipeline_nopc.py --category breakfast_box
    python patchcore_vlm_pipeline_nopc.py                          # 跑所有類別
    python patchcore_vlm_pipeline_nopc.py --test-only --reset
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from PIL import Image

import google.genai as genai
from gemini_inference import gemini_inference_auto

# ---------------------------------------------------------------------------
# 路徑常數
# ---------------------------------------------------------------------------

LOCO_ROOT   = Path.home() / "data/iad/mvtec_loco"
RESULTS_DIR = Path.home() / "data/iad/code/results"

CATEGORIES = [
    "breakfast_box", "juice_bottle", "pushpins",
    "screw_bag", "splicing_connectors",
]


# ---------------------------------------------------------------------------
# 載入測試集
# ---------------------------------------------------------------------------

def collect_test_items(category: str) -> list[dict]:
    """收集測試集中所有圖片，附帶 gt_class 標籤。"""
    test_root = LOCO_ROOT / category / "test"
    items: list[dict] = []
    for subdir, gt_class in [
        ("good",                 "good"),
        ("logical_anomalies",    "logical"),
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

    tp = sum(1 for r in records if r["gt_class"] != "good" and r["is_anomaly"])
    fp = sum(1 for r in records if r["gt_class"] == "good" and r["is_anomaly"])
    fn = sum(1 for r in records if r["gt_class"] != "good" and not r["is_anomaly"])
    prec, rec, f1 = _prf(tp, fp, fn)
    out = {"overall": {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}}

    lg = [r for r in records if r["gt_class"] == "logical"]
    if lg:
        tp_l = sum(1 for r in lg if r["is_anomaly"])
        fn_l = sum(1 for r in lg if not r["is_anomaly"])
        prec_l, rec_l, f1_l = _prf(tp_l, 0, fn_l)
        out["logical"] = {"precision": prec_l, "recall": rec_l, "f1": f1_l, "n": len(lg), "tp": tp_l, "fn": fn_l}

    st = [r for r in records if r["gt_class"] == "structural"]
    if st:
        tp_s = sum(1 for r in st if r["is_anomaly"])
        fn_s = sum(1 for r in st if not r["is_anomaly"])
        prec_s, rec_s, f1_s = _prf(tp_s, 0, fn_s)
        out["structural"] = {"precision": prec_s, "recall": rec_s, "f1": f1_s, "n": len(st), "tp": tp_s, "fn": fn_s}

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

def run_category(category: str, client, jsonl_path: Path,
                 test_only: bool = False, anomaly_only: bool = False) -> list[dict]:
    """對單一類別執行純 VLM 推論管線。"""
    print(f"\n{'='*55}")
    print(f"NoPC-VLM  類別：{category}")
    print(f"{'='*55}")

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

        orig_img = Image.open(img_path).convert("RGB")

        result = gemini_inference_auto(orig_img, category, client)

        logical_score    = result.get("logical_score", 10)
        structural_score = result.get("structural_score", 10)
        vlm_score        = round(max(logical_score, structural_score) / 100, 4)

        record = {
            "category":         category,
            "stem":             stem,
            "gt_class":         gt_class,
            "is_anomaly":       result["is_anomaly"],
            "anomaly_type":     result["anomaly_type"],
            "logical_score":    logical_score,
            "structural_score": structural_score,
            "vlm_score":        vlm_score,
            "reason":           result["reason"],
            "norm_score":       0.0,
        }
        all_records.append(record)
        new_records.append(record)

        verdict_str = "異常" if result["is_anomaly"] else "正常"
        gt_str      = {"good": "正常品", "logical": "邏輯異常", "structural": "結構異常"}[gt_class]
        correct     = result["is_anomaly"] == (gt_class != "good")
        tag         = "✓" if correct else "✗"
        elapsed = time.time() - t0
        avg     = elapsed / (i + 1)
        print(f"[{i+1:3d}/{len(todo)}] {stem} ({gt_str}) → {verdict_str}  {tag}"
              f"  [{elapsed/60:.1f}min  avg {avg:.1f}s/張]")
        print(f"   VLM: log={logical_score}  str={structural_score}  vlm={vlm_score:.2f}")
        print(f"   reason: {result['reason'][:100]}")

        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    cat_records = [r for r in all_records if r.get("category") == category]
    metrics = compute_metrics(cat_records)
    print_metrics(category, metrics)
    return cat_records


def main() -> None:
    run_start = time.time()
    parser = argparse.ArgumentParser(description="純 VLM 推論管線（不使用 PatchCore）")
    parser.add_argument("--category", nargs="+", default=None,
                        help="指定類別（預設：所有 5 個類別）")
    parser.add_argument("--mode", choices=["text", "heatmap"], default="text",
                        help="保留參數（此管線無 heatmap 模式，固定使用原圖）")
    parser.add_argument("--test-only", action="store_true",
                        help="只跑每類別每子集第 1 張（3 張）用於快速測試")
    parser.add_argument("--reset", action="store_true",
                        help="清除既有結果，從頭重跑所有圖片")
    parser.add_argument("--anomaly-only", action="store_true", dest="anomaly_only",
                        help="只跑異常圖片（跳過 good 圖片）")
    parser.add_argument("--retry-errors", action="store_true", dest="retry_errors",
                        help="只重跑上次因 API 錯誤失敗的圖片")
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
    jsonl_path = RESULTS_DIR / "patchcore_vlm_text_nopc_results.jsonl"

    if args.reset and jsonl_path.exists():
        jsonl_path.unlink()
        print(f"已清除既有結果：{jsonl_path.name}")

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
    print(f"管線模式：純 VLM（不使用 PatchCore）")
    if args.anomaly_only:
        print("篩選    ：只跑異常圖片（跳過 good）")

    all_records: list[dict] = []
    for cat in categories:
        records = run_category(
            cat, client, jsonl_path,
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
