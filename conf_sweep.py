"""
conf_sweep.py

Evaluates the trained YOLOv8 model at multiple confidence thresholds by running
inference once (conf=0.001) to collect all raw detections, then filtering at each
threshold and computing TP/FP/FN directly against ground-truth boxes.

ultralytics model.val() reports P/R at the *optimal F1 threshold* from the PR
curve regardless of the conf argument — so it always returns the same P/R values.
This script bypasses that by matching predictions to GT manually.

Matching rule: prediction is a TP if its IoU with a GT box of the same class >= 0.5.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BEST_PT      = Path.home() / "data/iad/code/runs/yolo_baseline/weights/best.pt"
TEST_IMG_DIR = Path.home() / "data/iad/yolo_dataset/images/test"
TEST_LBL_DIR = Path.home() / "data/iad/yolo_dataset/labels/test"
OUT_PATH     = Path.home() / "data/iad/code/conf_sweep_results.json"

THRESHOLDS  = [0.25, 0.15, 0.10, 0.05]
MATCH_IOU   = 0.50


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def box_iou(a: list, b: list) -> float:
    """IoU of two [x1, y1, x2, y2] boxes."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Collect all raw predictions + GT (single forward pass)
# ---------------------------------------------------------------------------

def collect(model: YOLO) -> tuple[dict, dict]:
    """
    Run inference at conf=0.001 so every detection is kept, then return
    all predictions and ground-truth boxes indexed by image stem.

    Returns
    -------
    all_preds : {stem: [(cls_id, conf, [x1,y1,x2,y2])]}
    all_gt    : {stem: [(cls_id, [x1,y1,x2,y2])]}
    """
    all_preds: dict = {}
    all_gt: dict    = {}

    for result in model.predict(
        source=str(TEST_IMG_DIR),
        conf=0.001,
        iou=0.7,       # NMS IoU — keep overlapping boxes we'd otherwise lose
        imgsz=640,
        verbose=False,
        stream=True,
    ):
        stem = Path(result.path).stem
        img_h, img_w = result.orig_shape   # original resolution (1280 x 1600)

        # ── ground truth ────────────────────────────────────────────────
        lbl_path = TEST_LBL_DIR / f"{stem}.txt"
        gt: list = []
        if lbl_path.exists():
            for line in lbl_path.read_text().strip().splitlines():
                if not line:
                    continue
                parts = line.split()
                cls_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                gt.append((cls_id, [
                    (cx - w / 2) * img_w, (cy - h / 2) * img_h,
                    (cx + w / 2) * img_w, (cy + h / 2) * img_h,
                ]))
        all_gt[stem] = gt

        # ── predictions ─────────────────────────────────────────────────
        preds: list = []
        if result.boxes is not None and len(result.boxes):
            for box in result.boxes:
                preds.append((
                    int(box.cls[0]),
                    float(box.conf[0]),
                    box.xyxy[0].tolist(),
                ))
        all_preds[stem] = preds

    return all_preds, all_gt


# ---------------------------------------------------------------------------
# Metrics at a fixed threshold
# ---------------------------------------------------------------------------

def compute_metrics(
    all_preds: dict,
    all_gt: dict,
    conf_thresh: float,
    iou_thresh: float = MATCH_IOU,
) -> dict:
    """
    Filter predictions at conf_thresh, match to GT boxes (greedy, highest-conf
    first), accumulate TP/FP/FN, and return precision/recall/F1 per class.

    A prediction is a TP when:
      - its class matches a GT box of the same class, AND
      - IoU with that GT box >= iou_thresh, AND
      - that GT box has not already been matched.
    """
    tp: dict[int, int] = defaultdict(int)
    fp: dict[int, int] = defaultdict(int)
    fn: dict[int, int] = defaultdict(int)

    for stem, gts in all_gt.items():
        # Filter predictions by confidence threshold
        preds = sorted(
            [(c, s, b) for c, s, b in all_preds.get(stem, []) if s >= conf_thresh],
            key=lambda x: -x[1],   # descending confidence
        )

        matched_gt = [False] * len(gts)

        for pred_cls, _, pred_box in preds:
            best_iou, best_j = 0.0, -1
            for j, (gt_cls, gt_box) in enumerate(gts):
                if gt_cls != pred_cls or matched_gt[j]:
                    continue
                iou = box_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou, best_j = iou, j

            if best_iou >= iou_thresh and best_j >= 0:
                tp[pred_cls] += 1
                matched_gt[best_j] = True
            else:
                fp[pred_cls] += 1

        for j, (gt_cls, _) in enumerate(gts):
            if not matched_gt[j]:
                fn[gt_cls] += 1

    def _prf(t: int, fpos: int, fneg: int) -> tuple[float, float, float]:
        p  = t / (t + fpos) if (t + fpos) > 0 else 0.0
        r  = t / (t + fneg) if (t + fneg) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return round(p, 4), round(r, 4), round(f1, 4)

    row: dict = {}
    total_tp = total_fp = total_fn = 0
    for cls_id, name in [(0, "structural"), (1, "logical")]:
        p, r, f1 = _prf(tp[cls_id], fp[cls_id], fn[cls_id])
        row[name] = {
            "precision": p, "recall": r, "f1": f1,
            "tp": tp[cls_id], "fp": fp[cls_id], "fn": fn[cls_id],
        }
        total_tp += tp[cls_id]
        total_fp += fp[cls_id]
        total_fn += fn[cls_id]

    p, r, f1 = _prf(total_tp, total_fp, total_fn)
    row["overall"] = {"precision": p, "recall": r, "f1": f1}
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    model = YOLO(str(BEST_PT))

    print("Collecting all detections (single inference pass, conf=0.001) …")
    all_preds, all_gt = collect(model)
    n_gt = sum(len(v) for v in all_gt.values())
    print(f"  {len(all_gt)} images  |  {n_gt} GT boxes total")

    report: dict = {}

    print(f"\n{'conf':>5}  {'class':<12}  {'P':>7}  {'R':>7}  {'F1':>7}  {'TP':>5}  {'FP':>5}  {'FN':>5}")
    print("-" * 65)

    for conf in THRESHOLDS:
        m = compute_metrics(all_preds, all_gt, conf)
        report[conf] = m

        for cls in ("overall", "structural", "logical"):
            d = m[cls]
            counts = (
                f"  {d['tp']:>5}  {d['fp']:>5}  {d['fn']:>5}"
                if cls != "overall"
                else "  " + "-" * 17
            )
            print(
                f"{conf:>5.2f}  {cls:<12}  "
                f"{d['precision']:>7.4f}  {d['recall']:>7.4f}  {d['f1']:>7.4f}"
                f"{counts}"
            )
        print()

    OUT_PATH.write_text(json.dumps(report, indent=2))
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
