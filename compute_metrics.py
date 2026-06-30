#!/usr/bin/env python3
"""
compute_metrics.py
從 JSONL 結果檔計算 AUROC、F1-max、fixed F1、good FP率。

使用方式：
    # 指定單一檔案
    python compute_metrics.py --file results/patchcore_vlm_text_auto_results.jsonl

    # 指定分數來源
    python compute_metrics.py --file results/xxx.jsonl --score combined    # detector*w_pc + vlm*w_vlm
    python compute_metrics.py --file results/xxx.jsonl --score patchcore   # 只用偵測器分數
    python compute_metrics.py --file results/xxx.jsonl --score vlm         # 只用 vlm_score

    # 自訂權重
    python compute_metrics.py --file results/xxx.jsonl --score combined --w-pc 0.3 --w-vlm 0.7

    # 一次跑多個檔案比較
    python compute_metrics.py --all

說明：
    不同 pipeline 的 JSONL 中 norm_score 意義不同：
      - auto 版：norm_score = PatchCore 偵測分數（0–1）
      - combined 版：norm_score = combined_score（已加權），偵測器分數在
        patchcore_norm_score / efficientad_norm_score
      - nopc 版：norm_score 固定為 0
    為了讓 --score 在所有檔案語意一致，本程式以 _detector_score() 取「偵測器原始分數」：
    優先取 patchcore_norm_score / efficientad_norm_score，沒有才退回 norm_score。
"""
from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# 路徑常數
# ---------------------------------------------------------------------------

RESULTS_DIR = Path.home() / "data/iad/code/results"

CATEGORIES = [
    "breakfast_box", "juice_bottle", "pushpins",
    "screw_bag", "splicing_connectors",
]

FIXED_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# 分數取值
# ---------------------------------------------------------------------------

def _detector_score(r: dict) -> float:
    """
    取偵測器（PatchCore / EfficientAD）原始 0–1 分數。
    優先 patchcore_norm_score / efficientad_norm_score，沒有才退回 norm_score。
    """
    for key in ("patchcore_norm_score", "efficientad_norm_score"):
        if key in r and r[key] is not None:
            return float(r[key])
    return float(r.get("norm_score", 0.0))


def _vlm_score(r: dict) -> float:
    """取 VLM 分數（0–1）；舊版若缺 vlm_score 則回傳 0。"""
    return float(r.get("vlm_score", 0.0))


def compute_score(r: dict, score: str, w_pc: float, w_vlm: float) -> float:
    """
    依 score 來源計算單筆記錄的最終分數。

    score:
      "patchcore" → 只用偵測器分數
      "vlm"       → 只用 VLM 分數
      "combined"  → detector*w_pc + vlm*w_vlm
    """
    if score == "patchcore":
        return _detector_score(r)
    if score == "vlm":
        return _vlm_score(r)
    # combined
    return _detector_score(r) * w_pc + _vlm_score(r) * w_vlm


# ---------------------------------------------------------------------------
# 核心指標計算
# ---------------------------------------------------------------------------

def calc_metrics(records: list[dict], atype: str, score: str,
                 w_pc: float, w_vlm: float) -> dict | None:
    """
    計算指定異常類型的 F1-max、AUROC、fixed F1、good FP率。

    atype: "logical"、"structural" 或 "overall"
      - logical    ：正樣本=邏輯異常，負樣本=正常品（排除結構異常）
      - structural ：正樣本=結構異常，負樣本=正常品（排除邏輯異常）
      - overall    ：正樣本=所有異常，負樣本=正常品

    回傳 dict：{
        "f1_max", "f1_max_threshold", "auroc",
        "fixed_f1", "fixed_fp_rate", "n_pos", "n_neg"
    }
    若無足夠樣本（缺正樣本或負樣本）回傳 None。
    """
    if atype == "logical":
        pos_classes = {"logical"}
    elif atype == "structural":
        pos_classes = {"structural"}
    else:  # overall
        pos_classes = {"logical", "structural"}

    # 子集：正樣本（指定異常類型）+ 全部正常品
    subset = [r for r in records
              if r["gt_class"] in pos_classes or r["gt_class"] == "good"]
    labels = np.array([1 if r["gt_class"] in pos_classes else 0 for r in subset])
    scores = np.array([compute_score(r, score, w_pc, w_vlm) for r in subset])

    n_pos = int(labels.sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None

    # --- AUROC ---
    if len(np.unique(scores)) == 1:
        auroc = 0.5  # 分數全相同，無區分力
    else:
        auroc = float(roc_auc_score(labels, scores))

    # --- F1-max：掃描所有候選 threshold ---
    f1_max, f1_max_thr = _f1_max(labels, scores)

    # --- fixed threshold（0.5）指標 ---
    fixed_f1 = _f1_at(labels, scores, FIXED_THRESHOLD)
    good_mask = labels == 0
    fixed_fp_rate = float((scores[good_mask] >= FIXED_THRESHOLD).mean()) if n_neg else 0.0

    return {
        "f1_max":           f1_max,
        "f1_max_threshold": f1_max_thr,
        "auroc":            auroc,
        "fixed_f1":         fixed_f1,
        "fixed_fp_rate":    fixed_fp_rate,
        "n_pos":            n_pos,
        "n_neg":            n_neg,
    }


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
    # 額外加入「略低於最小值」的 threshold，涵蓋全部判為正的情況
    thresholds = np.concatenate([[candidates.min() - 1e-6], candidates])
    best_f1, best_thr = 0.0, thresholds[0]
    for thr in thresholds:
        f1 = _f1_at(labels, scores, thr)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_f1, best_thr


# ---------------------------------------------------------------------------
# JSONL 載入
# ---------------------------------------------------------------------------

def load_records(path: Path) -> list[dict]:
    """讀取 JSONL 結果檔，回傳記錄 list。"""
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# 逐類別彙總
# ---------------------------------------------------------------------------

_AVG_KEYS = ["f1_max", "auroc", "fixed_f1", "fixed_fp_rate"]


def category_table(records: list[dict], atype: str, score: str,
                   w_pc: float, w_vlm: float) -> dict[str, dict]:
    """回傳 {category: metrics}（只含有足夠樣本的類別）。"""
    rows: dict[str, dict] = {}
    for cat in CATEGORIES:
        cat_recs = [r for r in records
                    if r.get("category") == cat and "gt_class" in r]
        if not cat_recs:
            continue
        m = calc_metrics(cat_recs, atype, score, w_pc, w_vlm)
        if m is not None:
            rows[cat] = m
    return rows


def macro_avg(rows: dict[str, dict]) -> dict | None:
    """對逐類別指標取 macro 平均（MVTec LOCO 標準作法）。"""
    if not rows:
        return None
    return {k: float(np.mean([m[k] for m in rows.values()])) for k in _AVG_KEYS}


# ---------------------------------------------------------------------------
# CJK 對齊輔助
# ---------------------------------------------------------------------------

def _disp_w(s: str) -> int:
    """計算字串顯示寬度（CJK 全形字算 2）。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def _ljust(s: str, width: int) -> str:
    return s + " " * max(0, width - _disp_w(s))


def _rjust(s: str, width: int) -> str:
    return " " * max(0, width - _disp_w(s)) + s


def _score_label(score: str, w_pc: float, w_vlm: float) -> str:
    if score == "combined":
        return f"combined ({w_pc:g} PC + {w_vlm:g} VLM)"
    if score == "patchcore":
        return "patchcore（僅偵測器分數）"
    return "vlm（僅 VLM 分數）"


# ---------------------------------------------------------------------------
# Step 2：單一檔案評估
# ---------------------------------------------------------------------------

def run_file(path: Path, score: str, w_pc: float, w_vlm: float) -> None:
    """評估單一 JSONL 檔，逐異常類型印出逐類別表格。"""
    records = load_records(path)
    print(f"\n檔案：{path.name}")
    print(f"分數：{_score_label(score, w_pc, w_vlm)}")

    cat_w = 24
    header = (_ljust("類別", cat_w) + _rjust("F1-max", 8) + _rjust("AUROC", 8)
              + _rjust("fixed-F1", 10) + _rjust("good-FP率", 11))
    for atype in ("logical", "structural", "overall"):
        rows = category_table(records, atype, score, w_pc, w_vlm)
        print(f"\n{atype.upper()}")
        print(header)
        print("-" * _disp_w(header))
        if not rows:
            print("（無足夠樣本）")
            continue
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


# ---------------------------------------------------------------------------
# Step 3：批次比較
# ---------------------------------------------------------------------------

def run_all(w_pc: float, w_vlm: float) -> None:
    """掃描 results/ 下所有 *_results.jsonl，對三種分數來源各印一張比較表。"""
    files = sorted(RESULTS_DIR.glob("*_results.jsonl"))
    if not files:
        print(f"在 {RESULTS_DIR} 找不到任何 *_results.jsonl")
        return

    MIN_RECORDS = 100

    # 預先載入並計算每個檔案、每種分數、每個 atype 的 macro 平均
    # cache[(file, score, atype)] = macro_avg dict or None
    cache: dict[tuple[str, str, str], dict | None] = {}
    counts: dict[str, int] = {}
    valid_files: list[str] = []
    skipped: list[tuple[str, str]] = []  # (檔名, 原因)
    for path in files:
        try:
            records = load_records(path)
        except Exception as exc:
            skipped.append((path.name, f"讀取失敗：{exc}"))
            continue
        name = path.name.replace("_results.jsonl", "")
        n = len(records)
        if n < MIN_RECORDS:
            skipped.append((path.name, f"記錄數 {n} < {MIN_RECORDS}"))
            continue
        has_any = False
        for score in ("patchcore", "vlm", "combined"):
            for atype in ("logical", "structural", "overall"):
                rows = category_table(records, atype, score, w_pc, w_vlm)
                avg = macro_avg(rows)
                cache[(name, score, atype)] = avg
                if avg is not None:
                    has_any = True
        if has_any:
            counts[name] = n
            valid_files.append(name)
        else:
            skipped.append((path.name, "無法計算任何指標（缺 gt_class/分數欄位）"))

    if skipped:
        print(f"\n已跳過 {len(skipped)} 個檔案：")
        for fname, reason in skipped:
            print(f"  - {fname}（{reason}）")

    if not valid_files:
        print("\n沒有符合條件的檔案可比較。")
        return

    name_w = max((_disp_w(n) for n in valid_files), default=10) + 2

    for score in ("patchcore", "vlm", "combined"):
        title = f"=== 總比較表（{_score_label(score, w_pc, w_vlm)}）==="
        print(f"\n{title}")
        header = (_ljust("檔案", name_w) + _rjust("N", 6)
                  + _rjust("log F1-max", 12) + _rjust("log AUROC", 11)
                  + _rjust("str F1-max", 12) + _rjust("str AUROC", 11)
                  + _rjust("ovr F1-max", 12) + _rjust("ovr AUROC", 11))
        print(header)
        print("-" * _disp_w(header))
        for name in valid_files:
            cells = [_ljust(name, name_w) + _rjust(str(counts[name]), 6)]
            for atype in ("logical", "structural", "overall"):
                avg = cache.get((name, score, atype))
                if avg is None:
                    cells.append(_rjust("-", 12) + _rjust("-", 11))
                else:
                    cells.append(_rjust(f"{avg['f1_max']:.4f}", 12)
                                 + _rjust(f"{avg['auroc']:.4f}", 11))
            print("".join(cells))


# ---------------------------------------------------------------------------
# 主程式
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="從 JSONL 結果檔計算 AUROC、F1-max")
    parser.add_argument("--file", type=str, help="指定 JSONL 結果檔")
    parser.add_argument("--score", type=str, default="combined",
                        choices=["combined", "patchcore", "vlm"],
                        help="分數來源：combined(預設)、patchcore、vlm")
    parser.add_argument("--w-pc", type=float, default=0.5, dest="w_pc",
                        help="偵測器（PatchCore/EfficientAD）權重，預設 0.5")
    parser.add_argument("--w-vlm", type=float, default=0.5, dest="w_vlm",
                        help="VLM 權重，預設 0.5")
    parser.add_argument("--all", action="store_true",
                        help="掃描 results/ 下所有結果檔並比較")
    args = parser.parse_args()

    if args.all:
        run_all(args.w_pc, args.w_vlm)
    elif args.file:
        run_file(Path(args.file), args.score, args.w_pc, args.w_vlm)
    else:
        parser.error("請指定 --file <檔案> 或 --all")


if __name__ == "__main__":
    main()
