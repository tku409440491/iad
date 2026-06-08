#!/usr/bin/env python3
"""用 YOLO 最佳 threshold 計算 YOLO+Gemma 的 F1"""
from pathlib import Path
from ultralytics import YOLO
import json
import numpy as np

dataset_root = Path.home() / "data/iad/mvtec_loco"
best_pt = Path.home() / "data/iad/code/runs/yolo_baseline/weights/best.pt"
results_path = Path.home() / "data/iad/code/results/pipeline_results.jsonl"

model = YOLO(str(best_pt))

# 讀取 pipeline 結果
pipeline = {}
with open(results_path) as f:
    for line in f:
        r = json.loads(line)
        pipeline[r["stem"]] = r

# 對每個 stem 取得 YOLO max conf
anomaly_map = {"logical_anomalies": "log", "structural_anomalies": "str"}
categories = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]

records = []
for cat in categories:
    for atype, short in anomaly_map.items():
        img_dir = dataset_root / cat / "test" / atype
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.glob("*.png")):
            stem = f"{cat}__{short}__{img_path.stem}"
            if stem not in pipeline:
                continue
            preds = model.predict(str(img_path), conf=0.01, iou=0.5, imgsz=640, verbose=False)
            boxes = preds[0].boxes
            yolo_score = float(boxes.conf.max()) if boxes is not None and len(boxes) > 0 else 0.0
            p = pipeline[stem]
            records.append({
                "gt_class": p["gt_class"],
                "yolo_score": yolo_score,
                "gemma_anomaly": p["is_anomaly"],
            })

# YOLO 最佳 threshold（從 yolo_auroc.py 結果）
thresholds = {"overall": 0.29, "logical": 0.39, "structural": 0.47}

def calc_f1(records, gt_filter, yolo_thresh):
    subset = [r for r in records if gt_filter == "overall" or r["gt_class"] == gt_filter]
    tp = sum(1 for r in subset if r["yolo_score"] >= yolo_thresh and r["gemma_anomaly"])
    fp = sum(1 for r in subset if r["yolo_score"] >= yolo_thresh and not r["gemma_anomaly"])
    fn = sum(1 for r in subset if r["yolo_score"] < yolo_thresh or not r["gemma_anomaly"])
    # 重新算：預測為異常 = YOLO conf >= thresh AND Gemma says anomaly
    pred_pos = [r["yolo_score"] >= yolo_thresh and r["gemma_anomaly"] for r in subset]
    gt_pos   = [True] * len(subset)  # 全部都是異常樣本
    tp = sum(p and g for p,g in zip(pred_pos, gt_pos))
    fp = sum(p and not g for p,g in zip(pred_pos, gt_pos))
    fn = sum(not p and g for p,g in zip(pred_pos, gt_pos))
    prec = tp/(tp+fp) if tp+fp > 0 else 0
    rec  = tp/(tp+fn) if tp+fn > 0 else 0
    f1   = 2*prec*rec/(prec+rec) if prec+rec > 0 else 0
    return f1, prec, rec, tp, fp, fn

print(f"{'':12} {'F1':>8}  {'P':>8}  {'R':>8}  TP  FP  FN")
for label, gt_f, thresh_key in [
    ("overall",    "overall",    "overall"),
    ("logical",    "logical",    "logical"),
    ("structural", "structural", "structural"),
]:
    f1, p, r, tp, fp, fn = calc_f1(records, gt_f, thresholds[thresh_key])
    print(f"{label:12} {f1:>8.4f}  {p:>8.4f}  {r:>8.4f}  {tp:3d} {fp:3d} {fn:3d}")
