#!/usr/bin/env python3
"""
train_yolo.py

Fine-tunes YOLOv8s on the MVTec LOCO-AD anomaly dataset (class 0 = structural,
class 1 = logical), then evaluates the best checkpoint on the held-out test
split and writes per-class precision / recall / F1 to baseline_results.json.

Training parameters
-------------------
  model   : yolov8s.pt (COCO pretrained)
  epochs  : 50
  imgsz   : 640
  batch   : 16
  seed    : 42

Evaluation parameters
---------------------
  conf    : 0.25
  iou     : 0.50
  split   : test

Usage
-----
  python train_yolo.py                   # full train + eval
  python train_yolo.py --skip-train      # eval only (reuse existing best.pt)
  python train_yolo.py --epochs 100 --batch 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Paths & defaults
# ---------------------------------------------------------------------------

DATA_YAML = Path.home() / "data/iad/yolo_dataset/data.yaml"
CODE_DIR  = Path.home() / "data/iad/code"
RUNS_DIR  = CODE_DIR / "runs"
RUN_NAME  = "yolo_baseline"

MODEL_WEIGHTS = "yolov8s.pt"
EPOCHS  = 50
IMGSZ   = 640
BATCH   = 16
SEED    = 42
CONF    = 0.25
IOU_TH  = 0.50


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> Path:
    """Train YOLOv8s and return the path to best.pt.

    Checkpoints are saved every 10 epochs in addition to the final best/last
    weights.  Training metrics (loss, mAP) are printed to stdout each epoch
    by ultralytics automatically.

    Parameters
    ----------
    args:
        Parsed CLI arguments with ``epochs``, ``batch``, ``imgsz``.

    Returns
    -------
    Path
        Absolute path to ``best.pt`` produced by this training run.
    """
    model = YOLO(MODEL_WEIGHTS)

    model.train(
        data        = str(DATA_YAML),
        epochs      = args.epochs,
        imgsz       = args.imgsz,
        batch       = args.batch,
        seed        = SEED,
        project     = str(RUNS_DIR),
        name        = RUN_NAME,
        exist_ok    = True,
        save        = True,
        save_period = 10,
        plots       = True,
        verbose     = True,
    )

    return RUNS_DIR / RUN_NAME / "weights" / "best.pt"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _safe_f1(p: float, r: float) -> float:
    denom = p + r
    return 2.0 * p * r / denom if denom > 0.0 else 0.0


def evaluate(best_pt: Path, args: argparse.Namespace) -> dict:
    """Evaluate best.pt on the test split and return a metrics dict.

    Uses conf=0.25 and iou=0.50 as specified.  Per-class precision, recall,
    and F1 are extracted from ``results.box`` (ultralytics ``Metric`` object)
    where ``.p`` / ``.r`` / ``.f1`` are arrays aligned to ``ap_class_index``.

    Parameters
    ----------
    best_pt:
        Path to the trained best.pt weights file.
    args:
        Parsed CLI arguments (``imgsz``, ``batch``).

    Returns
    -------
    dict
        ``{overall, structural, logical}`` each with precision/recall/f1,
        plus top-level ``map50`` and ``map50_95`` entries.
    """
    model = YOLO(str(best_pt))

    results = model.val(
        data    = str(DATA_YAML),
        split   = "test",
        conf    = CONF,
        iou     = IOU_TH,
        imgsz   = args.imgsz,
        batch   = args.batch,
        verbose = True,
    )

    box = results.box  # ultralytics Metric object

    # --- per-class metrics ---
    # ap_class_index[i] is the dataset class id for the i-th entry in p/r/f1
    class_metrics: dict[int, dict] = {}
    for i, cls_idx in enumerate(box.ap_class_index):
        cls_idx = int(cls_idx)
        p  = float(box.p[i])
        r  = float(box.r[i])
        # box.f1 may be empty depending on ultralytics version; compute as fallback
        f1 = float(box.f1[i]) if len(box.f1) > i else _safe_f1(p, r)
        class_metrics[cls_idx] = {"precision": round(p, 4),
                                   "recall":    round(r, 4),
                                   "f1":        round(f1, 4)}

    # --- overall = mean across classes present ---
    mp = float(box.mp)
    mr = float(box.mr)
    mf1 = _safe_f1(mp, mr)

    def _cls_entry(cls_id: int) -> dict:
        if cls_id in class_metrics:
            return class_metrics[cls_id]
        # class had zero GT or zero detections
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    return {
        "overall":    {"precision": round(mp,  4),
                       "recall":    round(mr,  4),
                       "f1":        round(mf1, 4)},
        "structural": _cls_entry(0),   # class 0
        "logical":    _cls_entry(1),   # class 1
        "map50":      round(float(box.map50), 4),
        "map50_95":   round(float(box.map),   4),
    }


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_metrics(metrics: dict) -> None:
    """Print a compact metrics table to stdout."""
    print("\n" + "=" * 55)
    print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 55)
    for name in ("overall", "structural", "logical"):
        m = metrics[name]
        print(
            f"{name:<15} {m['precision']:>10.4f}"
            f" {m['recall']:>10.4f} {m['f1']:>10.4f}"
        )
    print("=" * 55)
    print(f"  mAP@50      : {metrics['map50']:.4f}")
    print(f"  mAP@50-95   : {metrics['map50_95']:.4f}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse CLI args, run training and/or evaluation, save results."""
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8s on MVTec LOCO-AD and evaluate on test split."
    )
    parser.add_argument("--epochs",     type=int,  default=EPOCHS)
    parser.add_argument("--batch",      type=int,  default=BATCH)
    parser.add_argument("--imgsz",      type=int,  default=IMGSZ)
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and evaluate the existing best.pt directly.",
    )
    args = parser.parse_args()

    best_pt = RUNS_DIR / RUN_NAME / "weights" / "best.pt"

    # ── Training ─────────────────────────────────────────────────────────
    if args.skip_train:
        print(f"Skipping training — using existing weights: {best_pt}")
    else:
        print("=" * 60)
        print(
            f"Fine-tuning YOLOv8s  "
            f"epochs={args.epochs}  batch={args.batch}  imgsz={args.imgsz}"
        )
        print("=" * 60)
        best_pt = train(args)
        print(f"\nTraining complete.  Best weights → {best_pt}")

    if not best_pt.exists():
        raise FileNotFoundError(
            f"best.pt not found at {best_pt}\n"
            "Run without --skip-train to train first."
        )

    # ── Evaluation ───────────────────────────────────────────────────────
    print(f"\nEvaluating on test split  conf={CONF}  iou={IOU_TH} …")
    metrics = evaluate(best_pt, args)

    print_metrics(metrics)

    # ── Save results ─────────────────────────────────────────────────────
    out_path = CODE_DIR / "baseline_results.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
