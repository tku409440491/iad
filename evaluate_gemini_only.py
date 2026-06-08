#!/usr/bin/env python3
"""
evaluate_gemini_only.py

Ablation study B 組：純 Gemini，不跑 YOLO，不畫框，直接送原圖。
用來和 evaluate_pipeline.py（YOLO + Gemini，有框）做效果對比。

執行方式：
    python evaluate_gemini_only.py

支援斷點續跑：
    結果逐行寫入 results/gemini_only_results.jsonl，
    重新執行時自動跳過已處理的圖片。

輸出：
    results/gemini_only_results.jsonl  ── 每張圖一筆 JSON 記錄
    results/gemini_only_report.json    ── 最終彙總指標
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
from PIL import Image

# 從同目錄的 gemini_inference 借用產品規格與回應解析函式
sys.path.insert(0, str(Path(__file__).parent))
from gemini_inference import _CATEGORY_RULES, _parse_response


# ---------------------------------------------------------------------------
# 路徑 / 參數設定
# ---------------------------------------------------------------------------

TEST_IMG_DIR = Path.home() / "data/iad/yolo_dataset/images/test"
RESULTS_DIR  = Path.home() / "data/iad/code/results"
JSONL_PATH   = RESULTS_DIR / "gemini_only_results.jsonl"
REPORT_PATH  = RESULTS_DIR / "gemini_only_report.json"

MODEL_NAME  = "gemma-4-31b-it"
SLEEP_SECS  = 5
RETRY_DELAY = 20.0
MAX_RETRIES = 3

_GT_MAP = {"log": "logical", "str": "structural"}

_FALLBACK: dict = {"is_anomaly": False, "anomaly_type": "none", "reason": "parse_error"}


# ---------------------------------------------------------------------------
# Prompt（純視覺判斷，不含任何 YOLO 框說明）
# ---------------------------------------------------------------------------

def _build_prompt(category: str) -> str:
    """組成純視覺品管 prompt，不提及 YOLO 偵測框。"""
    rule = _CATEGORY_RULES.get(category, "請根據圖片判斷是否有工業瑕疵。")
    return f"""你是一個工業產品視覺品管專家。

【產品規格】
產品類別：{category}
規格說明：{rule}

【檢測任務】
這是一張工業檢測圖片，產品類別是 {category}。

請根據圖片內容判斷：
1. 這張圖片是否有異常？
2. 如果有，主要是結構異常還是邏輯異常？
3. 具體的異常原因是什麼？

只回傳 JSON，不要其他文字：
{{"is_anomaly": true/false, "anomaly_type": "structural"/"logical"/"none", "reason": "..."}}"""


# ---------------------------------------------------------------------------
# Gemini 推理（送原圖，不畫框）
# ---------------------------------------------------------------------------

def gemini_inference_no_yolo(
    image_path: Path,
    category: str,
    client,
) -> dict:
    """
    直接送原圖給 Gemma 進行異常判斷，不經過 YOLO、不畫任何框。

    參數
    ----
    image_path : 原圖路徑（symlink 亦可）
    category   : 產品類別字串，例如 'breakfast_box'
    client     : genai.Client 實例

    回傳
    ----
    dict：{"is_anomaly": bool, "anomaly_type": str, "reason": str}
    """
    img    = Image.open(image_path).convert("RGB")
    prompt = _build_prompt(category)

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        # 每次呼叫前固定等待，控制 RPM
        time.sleep(SLEEP_SECS)
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[img, prompt],
            )
            return _parse_response(response.text)
        except Exception as exc:
            last_error = exc
            err_lower  = str(exc).lower()
            is_retryable = any(kw in err_lower for kw in (
                "429", "quota", "rate", "resource exhausted",
                "503", "unavailable", "timeout", "deadline exceeded",
            ))
            if is_retryable and attempt < MAX_RETRIES:
                print(
                    f"  [重試 {attempt}/{MAX_RETRIES}] "
                    f"API 暫時錯誤，{RETRY_DELAY:.0f} 秒後重試…  ({exc})"
                )
                time.sleep(RETRY_DELAY)
            else:
                break

    print(f"  [錯誤] Gemini API 呼叫失敗（嘗試 {MAX_RETRIES} 次）：{last_error}")
    return {**_FALLBACK, "reason": f"api_error: {last_error}"}


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
# 斷點續跑：讀取已完成的 stems
# ---------------------------------------------------------------------------

def load_completed_records(jsonl_path: Path) -> tuple[set[str], list[dict]]:
    """讀取既有 JSONL，回傳已完成的 stem 集合與記錄列表。"""
    completed: set[str]   = set()
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
    根據所有記錄計算 TP/FP/FN 並推算 P/R/F1。

    判斷邏輯（所有 test 圖片皆為異常）：
        is_anomaly=True  AND anomaly_type == gt_class → TP（該類別）
        is_anomaly=True  AND anomaly_type != gt_class → FP（預測類別）、FN（GT 類別）
        is_anomaly=False OR  anomaly_type='none'      → FN（GT 類別）

    整體指標以每張圖為單位計算，互不重複。
    """
    tp_cls: dict[str, int] = defaultdict(int)
    fp_cls: dict[str, int] = defaultdict(int)
    fn_cls: dict[str, int] = defaultdict(int)
    tp_total = fp_total = fn_total = 0

    for r in records:
        gt_cls   = r["gt_class"]
        pred_cls = r["anomaly_type"] if r.get("is_anomaly") else None

        if pred_cls == gt_cls:
            tp_cls[gt_cls] += 1
            tp_total       += 1
        elif pred_cls in ("logical", "structural"):
            fp_cls[pred_cls] += 1
            fn_cls[gt_cls]   += 1
            fp_total         += 1
        else:
            fn_cls[gt_cls] += 1
            fn_total       += 1

    def _prf(t: int, fpos: int, fneg: int) -> dict:
        p  = t / (t + fpos) if (t + fpos) > 0 else 0.0
        r  = t / (t + fneg) if (t + fneg) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return {
            "precision": round(p,  4),
            "recall":    round(r,  4),
            "f1":        round(f1, 4),
            "tp": t, "fp": fpos, "fn": fneg,
        }

    return {
        "overall":    _prf(tp_total,              fp_total,              fn_total),
        "logical":    _prf(tp_cls["logical"],    fp_cls["logical"],    fn_cls["logical"]),
        "structural": _prf(tp_cls["structural"], fp_cls["structural"], fn_cls["structural"]),
    }


# ---------------------------------------------------------------------------
# 主程式
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="純 Gemini（無 YOLO）pipeline 評估")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="清除既有結果，從頭重跑所有圖片",
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

    print("初始化 Gemini client…")
    client = genai.Client(api_key=api_key)

    # ── 掃描 test set ──────────────────────────────────────────────────────
    all_images = sorted(TEST_IMG_DIR.glob("*.png"))
    total      = len(all_images)
    print(f"Test set：{total} 張圖片（純 Gemini，無 YOLO 框）")

    # ── 斷點續跑 ────────────────────────────────────────────────────────────
    completed, all_records = load_completed_records(JSONL_PATH)
    if completed:
        print(f"已完成 {len(completed)} 張，跳過繼續…\n")

    # ── 主迴圈 ──────────────────────────────────────────────────────────────
    errors  = 0
    t_start = time.time()

    with JSONL_PATH.open("a", encoding="utf-8") as fout:
        for idx, img_path in enumerate(all_images, start=1):
            stem = img_path.stem

            if stem in completed:
                continue

            try:
                category, gt_class, img_id = parse_stem(stem)
            except ValueError as e:
                print(f"  [{idx:>3}/{total}] 解析失敗，略過：{e}")
                continue

            # Gemini 推理（原圖，無框；內含 time.sleep(5)）
            result       = gemini_inference_no_yolo(img_path, category, client)
            is_anomaly   = result["is_anomaly"]
            anomaly_type = result["anomaly_type"]
            reason       = result["reason"]

            # 判斷 TP / FP / FN
            is_error = ("api_error" in reason) or (reason == "parse_error")
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
            icon = "✓" if verdict == "TP" else "✗"
            if verdict == "TP":
                verdict_str = gt_class
            elif verdict == "FP":
                verdict_str = f"FP→{anomaly_type}（GT={gt_class}）"
            else:
                cause = "error" if is_error else f"not_detected({anomaly_type})"
                verdict_str = f"FN({cause})"

            print(f"[{idx:>3}/{total}] {category}/{gt_class[:3]}/{img_id}.png  {icon} {verdict_str}")

            # 寫入 JSONL（每張立即 flush）
            record: dict = {
                "stem":         stem,
                "category":     category,
                "gt_class":     gt_class,
                "img_id":       img_id,
                "n_detections": 0,
                "n_structural": 0,
                "n_logical":    0,
                "is_anomaly":   is_anomaly,
                "anomaly_type": anomaly_type,
                "reason":       reason,
                "verdict":      verdict,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            all_records.append(record)
            completed.add(stem)

    # ── 計算並輸出最終指標 ──────────────────────────────────────────────────
    runtime = time.time() - t_start
    metrics = compute_metrics(all_records)

    report: dict = {
        "overall":         metrics["overall"],
        "structural":      metrics["structural"],
        "logical":         metrics["logical"],
        "total_images":    total,
        "processed":       len(all_records),
        "errors":          errors,
        "runtime_seconds": round(runtime, 1),
    }

    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print(
        f"評估完成（純 Gemini）  │  共 {total} 張  │  "
        f"錯誤 {errors} 張  │  耗時 {runtime:.0f}s"
    )
    print("=" * 65)
    for cls in ("overall", "structural", "logical"):
        m = metrics[cls]
        print(
            f"  {cls:<12}  "
            f"P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}  "
            f"(TP={m['tp']}  FP={m['fp']}  FN={m['fn']})"
        )
    print(f"\n報告儲存：{REPORT_PATH}")


if __name__ == "__main__":
    main()
