#!/usr/bin/env python3
"""純 YOLO baseline 的 AUROC 與 F1-max 計算（含 good 圖片）"""
from pathlib import Path
from ultralytics import YOLO
import numpy as np
from sklearn.metrics import roc_auc_score

dataset_root = Path.home() / "data/iad/mvtec_loco"
best_pt = Path.home() / "data/iad/code/runs/yolo_baseline/weights/best.pt"
model = YOLO(str(best_pt))

categories = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
anomaly_types = ["logical_anomalies", "structural_anomalies"]

y_true, scores = [], []
y_true_log, scores_log = [], []
y_true_str, scores_str = [], []

for cat in categories:
    # 異常圖片
    for atype in anomaly_types:
        img_dir = dataset_root / cat / "test" / atype
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.glob("*.png")):
            preds = model.predict(str(img_path), conf=0.01, iou=0.5, imgsz=640, verbose=False)
            boxes = preds[0].boxes
            score = float(boxes.conf.max()) if boxes is not None and len(boxes) > 0 else 0.0
            y_true.append(1)
            scores.append(score)
            if atype == "logical_anomalies":
                y_true_log.append(1); scores_log.append(score)
            else:
                y_true_str.append(1); scores_str.append(score)

    # 正常圖片
    good_dir = dataset_root / cat / "test" / "good"
    if not good_dir.exists():
        continue
    for img_path in sorted(good_dir.glob("*.png")):
        preds = model.predict(str(img_path), conf=0.01, iou=0.5, imgsz=640, verbose=False)
        boxes = preds[0].boxes
        score = float(boxes.conf.max()) if boxes is not None and len(boxes) > 0 else 0.0
        y_true.append(0)
        scores.append(score)
        y_true_log.append(0); scores_log.append(score)
        y_true_str.append(0); scores_str.append(score)

# AUROC
auroc_all = roc_auc_score(y_true, scores)
auroc_log = roc_auc_score(y_true_log, scores_log)
auroc_str = roc_auc_score(y_true_str, scores_str)

# F1-max
def f1_max(y_true, scores):
    best_f1, best_t = 0, 0
    for t in np.arange(0.01, 1.0, 0.01):
        preds = [1 if s >= t else 0 for s in scores]
        tp = sum(p==1 and g==1 for p,g in zip(preds, y_true))
        fp = sum(p==1 and g==0 for p,g in zip(preds, y_true))
        fn = sum(p==0 and g==1 for p,g in zip(preds, y_true))
        if tp+fp == 0 or tp+fn == 0: continue
        prec = tp/(tp+fp); rec = tp/(tp+fn)
        f1 = 2*prec*rec/(prec+rec) if prec+rec > 0 else 0
        if f1 > best_f1: best_f1, best_t = f1, t
    return best_f1, best_t

f1_all, t_all = f1_max(y_true, scores)
f1_log, t_log = f1_max(y_true_log, scores_log)
f1_str, t_str = f1_max(y_true_str, scores_str)

print(f"樣本數：異常 {sum(y_true)} 張，正常 {len(y_true)-sum(y_true)} 張")
print(f"{'':12} {'AUROC':>8}  {'F1-max':>8}  {'threshold':>10}")
print(f"{'overall':12} {auroc_all:>8.4f}  {f1_all:>8.4f}  {t_all:>10.2f}")
print(f"{'logical':12} {auroc_log:>8.4f}  {f1_log:>8.4f}  {t_log:>10.2f}")
print(f"{'structural':12} {auroc_str:>8.4f}  {f1_str:>8.4f}  {t_str:>10.2f}")
