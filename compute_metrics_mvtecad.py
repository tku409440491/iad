#!/usr/bin/env python3
"""
compute_metrics_mvtecad.py
計算 MVTec AD 格式結果的 F1-max、AUROC、fixed-F1、good-FP率。
gt_class="good" 為正常，其餘（各異常類型名稱）為異常。

使用方式：
    python compute_metrics_mvtecad.py --file results/patchcore_vlm_mvtecad_results.jsonl
    python compute_metrics_mvtecad.py --file results/xxx.jsonl --score patchcore
    python compute_metrics_mvtecad.py --file results/xxx.jsonl --score vlm

分數來源（--score）：
    combined  ：用 combined_score（0.5 PC + 0.5 VLM，預設）
    patchcore ：用 patchcore_norm_score
    vlm       ：用 vlm_score
"""
from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

FIXED_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# 分數取值
# ---------------------------------------------------------------------------

def score_of(r: dict, source: str) -> float:
    """依 source 取單筆記錄的分數。"""
    if source == "patchcore":
        return float(r.get("patchcore_norm_score", 0.0))
    if source == "vlm":
        return float(r.get("vlm_score", 0.0))
    # combined：優先 combined_score，退回 norm_score
    if "combined_score" in r and r["combined_score"] is not None:
        return float(r["combined_score"])
    return float(r.get("norm_score", 0.0))


# ---------------------------------------------------------------------------
# 指標計算
# ---------------------------------------------------------------------------

def _f1_at(labels: np.ndarray, scores: np.ndarray, thr: float) -> float:
    """在指定 threshold 計算 F1（score >= thr 判為正）。"""
    pred = scores >= thr
    tp = int(np.sum(pred & (labels == 1)))
    fp = int(np.sum(pred & (labels == 0)))
    fn = int(np.sum(~pred & (labels == 1)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def _f1_max(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """掃描所有候選 threshold，回傳 (最大 F1, 對應 threshold)。"""
    candidates = np.unique(scores)
    thresholds = np.concatenate([[candidates.min() - 1e-6], candidates])
    best_f1, best_thr = 0.0, float(thresholds[0])
    for thr in thresholds:
        f1 = _f1_at(labels, scores, thr)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_f1, best_thr


def calc_metrics(records: list[dict], source: str) -> dict | None:
    """
    對單一類別（good vs 所有異常）計算 F1-max、AUROC、fixed-F1、good-FP率，
    以及各異常類型在 threshold=0.5 下的 Recall。
    樣本不足（缺正或缺負）回傳 None。
    """
    labels = np.array([0 if r["gt_class"] == "good" else 1 for r in records])
    scores = np.array([score_of(r, source) for r in records])

    n_pos = int(labels.sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None

    auroc = 0.5 if len(np.unique(scores)) == 1 else float(roc_auc_score(labels, scores))
    f1_max, f1_max_thr = _f1_max(labels, scores)
    fixed_f1 = _f1_at(labels, scores, FIXED_THRESHOLD)
    good_mask = labels == 0
    fixed_fp_rate = float((scores[good_mask] >= FIXED_THRESHOLD).mean())

    # 各異常類型在 fixed threshold=0.5 下的 Recall
    per_type = {}
    for t in sorted({r["gt_class"] for r in records if r["gt_class"] != "good"}):
        sub = [r for r in records if r["gt_class"] == t]
        tp_t = sum(1 for r in sub if score_of(r, source) >= FIXED_THRESHOLD)
        per_type[t] = {"recall": tp_t / len(sub) if sub else 0.0, "tp": tp_t, "n": len(sub)}

    return {
        "f1_max":           f1_max,
        "f1_max_threshold": f1_max_thr,
        "auroc":            auroc,
        "fixed_f1":         fixed_f1,
        "fixed_fp_rate":    fixed_fp_rate,
        "n_pos":            n_pos,
        "n_neg":            n_neg,
        "per_type":         per_type,
    }


# ---------------------------------------------------------------------------
# 載入與彙總
# ---------------------------------------------------------------------------

def load_records(path: Path) -> list[dict]:
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def macro_avg(rows: dict[str, dict]) -> dict:
    keys = ["f1_max", "auroc", "fixed_f1", "fixed_fp_rate"]
    return {k: float(np.mean([m[k] for m in rows.values()])) for k in keys}


# ---------------------------------------------------------------------------
# CJK 對齊輔助
# ---------------------------------------------------------------------------

def _disp_w(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def _ljust(s: str, width: int) -> str:
    return s + " " * max(0, width - _disp_w(s))


def _rjust(s: str, width: int) -> str:
    return " " * max(0, width - _disp_w(s)) + s


def _score_label(source: str) -> str:
    if source == "combined":
        return "combined (0.5 PC + 0.5 VLM)"
    if source == "patchcore":
        return "patchcore（僅偵測器分數）"
    return "vlm（僅 VLM 分數）"


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run_file(path: Path, source: str) -> None:
    records = load_records(path)
    print(f"\n檔案：{path.name}")
    print(f"分數：{_score_label(source)}")

    # 逐類別
    categories = sorted({r["category"] for r in records})
    rows: dict[str, dict] = {}
    for cat in categories:
        cat_recs = [r for r in records if r["category"] == cat]
        m = calc_metrics(cat_recs, source)
        if m is not None:
            rows[cat] = m

    cat_w = 20
    header = (_ljust("類別", cat_w) + _rjust("F1-max", 8) + _rjust("AUROC", 8)
              + _rjust("fixed-F1", 10) + _rjust("good-FP率", 11))
    print("\nOVERALL")
    print(header)
    print("-" * _disp_w(header))
    if not rows:
        print("（無足夠樣本）")
        return
    for cat, m in rows.items():
        print(_ljust(cat, cat_w)
              + _rjust(f"{m['f1_max']:.4f}", 8)
              + _rjust(f"{m['auroc']:.4f}", 8)
              + _rjust(f"{m['fixed_f1']:.4f}", 10)
              + _rjust(f"{m['fixed_fp_rate']:.4f}", 11))
    avg = macro_avg(rows)
    print("-" * _disp_w(header))
    print(_ljust("平均", cat_w)
          + _rjust(f"{avg['f1_max']:.4f}", 8)
          + _rjust(f"{avg['auroc']:.4f}", 8)
          + _rjust(f"{avg['fixed_f1']:.4f}", 10)
          + _rjust(f"{avg['fixed_fp_rate']:.4f}", 11))

    # 各異常類型 Recall（fixed threshold=0.5）
    print("\n各異常類型 Recall（threshold=0.5）")
    for cat in rows:
        pt = rows[cat]["per_type"]
        if not pt:
            continue
        print(f"  [{cat}]")
        for t, mm in pt.items():
            print(f"    {_ljust(t, 22)}Recall={mm['recall']:.3f}  ({mm['tp']}/{mm['n']})")


def main() -> None:
    parser = argparse.ArgumentParser(description="計算 MVTec AD 結果的 AUROC、F1-max")
    parser.add_argument("--file", type=str, required=True, help="指定 JSONL 結果檔")
    parser.add_argument("--score", type=str, default="combined",
                        choices=["combined", "patchcore", "vlm"],
                        help="分數來源：combined(預設)、patchcore、vlm")
    args = parser.parse_args()
    run_file(Path(args.file), args.score)


if __name__ == "__main__":
    main()
