#!/usr/bin/env python3
"""
recompute_patchcore_raw.py
重新計算純 PatchCore 的 AUROC，使用未經 clip 的 raw_score。
不呼叫 VLM API。

對每張測試圖取得 PatchCore 的原始異常分數（out.pred_score，不做正規化），
逐類別計算 AUROC；同時輸出 clip 到 [0,1] 正規化後的 AUROC 作為對照。
（clip 只在飽和產生 ties 時才會改變 AUROC，此程式用來量化該影響。）

使用方式：
    python recompute_patchcore_raw.py --dataset mvtecad
    python recompute_patchcore_raw.py --dataset visa
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

RESULTS_DIR = Path.home() / "data/iad/code/results"


def _load_pipeline(dataset: str):
    """依 dataset 載入對應 pipeline 模組（沿用其 load_patchcore/infer_image/collect_test_items）。"""
    if dataset == "mvtecad":
        import patchcore_vlm_pipeline_mvtecad as P
    elif dataset == "visa":
        import patchcore_vlm_pipeline_visa as P
    else:
        raise ValueError(f"未知 dataset：{dataset}")
    return P


def main() -> None:
    parser = argparse.ArgumentParser(description="重新計算純 PatchCore 的 raw AUROC（不呼叫 VLM）")
    parser.add_argument("--dataset", choices=["mvtecad", "visa"], required=True,
                        help="資料集：mvtecad 或 visa")
    args = parser.parse_args()

    P = _load_pipeline(args.dataset)
    out_path = RESULTS_DIR / f"patchcore_raw_{args.dataset}.jsonl"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"資料集：{args.dataset}")
    print(f"Checkpoint 根目錄：{P.MODELS_DIR}")
    print(f"輸出檔案：{out_path.name}\n")

    all_records: list[dict] = []
    rows: dict[str, tuple[float, float, int, int]] = {}  # cat -> (raw_auroc, clip_auroc, n_pos, n_neg)

    for cat in P.ALL_CATEGORIES:
        # 載入該類別 PatchCore checkpoint
        try:
            model, ckpt = P.load_patchcore(cat)
        except FileNotFoundError:
            print(f"[跳過] {cat}：找不到 checkpoint")
            continue

        img_min = float(model.post_processor.image_min)
        img_max = float(model.post_processor.image_max)

        items = P.collect_test_items(cat)
        raws, clippeds, gts = [], [], []
        for it in items:
            raw, _amap = P.infer_image(model, it["path"])
            norm = float(np.clip((raw - img_min) / (img_max - img_min + 1e-8), 0, 1))
            gt = 1 if it["gt_class"] != "good" else 0
            raws.append(raw)
            clippeds.append(norm)
            gts.append(gt)
            all_records.append({
                "category":  cat,
                "stem":      it["stem"],
                "gt_class":  it["gt_class"],
                "raw_score": round(float(raw), 6),
            })

        gts_arr = np.array(gts)
        n_pos = int(gts_arr.sum())
        n_neg = int((gts_arr == 0).sum())
        if n_pos == 0 or n_neg == 0:
            print(f"[跳過] {cat}：樣本不足（正 {n_pos} 負 {n_neg}）")
            continue

        raw_auroc  = float(roc_auc_score(gts_arr, raws))
        clip_auroc = float(roc_auc_score(gts_arr, clippeds))
        rows[cat] = (raw_auroc, clip_auroc, n_pos, n_neg)
        print(f"  {cat:<22} raw={raw_auroc*100:6.2f}  clipped={clip_auroc*100:6.2f}  "
              f"(正 {n_pos} 負 {n_neg}, 共 {len(items)} 張)")

    # 寫入 JSONL（每次覆蓋）
    with out_path.open("w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 輸出比較表
    if not rows:
        print("\n沒有可計算的類別。")
        return

    print(f"\n{'='*60}")
    print(f"{'類別':<22}{'raw AUROC':>11}{'clipped AUROC':>15}{'差距':>9}")
    print("-" * 60)
    for cat, (ra, ca, _np, _nn) in rows.items():
        diff = (ra - ca) * 100
        print(f"{cat:<22}{ra*100:>11.2f}{ca*100:>15.2f}{diff:>+9.2f}")
    print("-" * 60)
    raw_avg  = np.mean([v[0] for v in rows.values()]) * 100
    clip_avg = np.mean([v[1] for v in rows.values()]) * 100
    print(f"{'平均':<22}{raw_avg:>11.2f}{clip_avg:>15.2f}{raw_avg - clip_avg:>+9.2f}")
    print(f"{'='*60}")
    print(f"\n共 {len(all_records)} 筆 raw_score 已存至 {out_path}")


if __name__ == "__main__":
    main()
