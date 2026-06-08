#!/usr/bin/env python3
"""
prepare_yolo_dataset.py

Assembles the MVTec LOCO-AD anomaly images and their pre-generated YOLO labels
into a single train/val/test dataset directory ready for YOLO training.

Split ratios: train 70 % · val 15 % · test 15 %  (random seed 42)
Stratification: category × anomaly_type (10 strata) so every group
maintains the same proportions across splits.

Images and labels are referenced via symbolic links — no data is copied.
Symlink names encode origin to avoid collisions across categories:
  {category}__{log|str}__{img_id}.{ext}

Output layout
-------------
yolo_dataset/
  images/
    train/  val/  test/
  labels/
    train/  val/  test/
  data.yaml

Usage
-----
  python prepare_yolo_dataset.py
  python prepare_yolo_dataset.py --dataset-root /path/to/mvtec_loco
  python prepare_yolo_dataset.py --output-dir /path/to/yolo_dataset
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import yaml


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_ROOT = Path.home() / "data" / "iad" / "mvtec_loco"
OUTPUT_DIR   = Path.home() / "data" / "iad" / "yolo_dataset"

ALL_CATEGORIES = [
    "breakfast_box",
    "juice_bottle",
    "pushpins",
    "screw_bag",
    "splicing_connectors",
]

ANOMALY_TYPES = {
    "logical_anomalies":    "log",
    "structural_anomalies": "str",
}

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO  = 0.15  (remainder)

SEED = 42


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    """One image/label pair with its stratum metadata."""
    img_path:    Path    # absolute path to source image
    lbl_path:    Path    # absolute path to source label
    category:   str
    anomaly_type: str   # 'logical_anomalies' | 'structural_anomalies'
    img_id:     str     # e.g. '000'
    symlink_stem: str   # unique stem used for both image and label symlinks


# ---------------------------------------------------------------------------
# Sample collection
# ---------------------------------------------------------------------------

def collect_samples(dataset_root: Path) -> List[Sample]:
    """Walk all categories and anomaly types and return every valid sample.

    A sample is valid when both the source image and its YOLO label exist.
    Missing labels trigger a warning and are skipped.

    Parameters
    ----------
    dataset_root:
        Root directory containing all category subdirectories.

    Returns
    -------
    list[Sample]
        All valid (image, label) pairs across all categories and anomaly types.
    """
    samples: List[Sample] = []

    for category in ALL_CATEGORIES:
        for anomaly_type, atype_short in ANOMALY_TYPES.items():
            img_dir = dataset_root / category / "test"   / anomaly_type
            lbl_dir = dataset_root / category / "labels" / anomaly_type

            if not img_dir.exists():
                print(f"  [skip] {category}/{anomaly_type}: image dir not found")
                continue

            for img_path in sorted(img_dir.glob("*.png")):
                img_id   = img_path.stem
                lbl_path = lbl_dir / f"{img_id}.txt"

                if not lbl_path.exists():
                    print(f"  WARNING: label missing for {category}/{anomaly_type}/{img_id}.png")
                    continue

                stem = f"{category}__{atype_short}__{img_id}"
                samples.append(
                    Sample(
                        img_path=img_path.resolve(),
                        lbl_path=lbl_path.resolve(),
                        category=category,
                        anomaly_type=anomaly_type,
                        img_id=img_id,
                        symlink_stem=stem,
                    )
                )

    return samples


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------

def stratified_split(
    samples: List[Sample],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float   = VAL_RATIO,
    seed: int          = SEED,
) -> Dict[str, List[Sample]]:
    """Split samples into train / val / test preserving per-stratum ratios.

    Stratification key: ``category + anomaly_type``.  Each stratum is split
    independently and the parts are concatenated, so proportions hold even
    when strata sizes differ.

    Parameters
    ----------
    samples:
        All collected samples.
    train_ratio:
        Fraction of total data for training (default 0.70).
    val_ratio:
        Fraction of total data for validation (default 0.15).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    dict[str, list[Sample]]
        Keys: ``'train'``, ``'val'``, ``'test'``.
    """
    rng = random.Random(seed)

    # Group by stratum
    strata: Dict[str, List[Sample]] = {}
    for s in samples:
        key = f"{s.category}__{s.anomaly_type}"
        strata.setdefault(key, []).append(s)

    splits: Dict[str, List[Sample]] = {"train": [], "val": [], "test": []}

    for key, group in sorted(strata.items()):
        shuffled = group.copy()
        rng.shuffle(shuffled)
        n = len(shuffled)

        n_test  = max(1, round(n * (1.0 - train_ratio - val_ratio)))
        n_val   = max(1, round(n * val_ratio))
        n_train = n - n_val - n_test

        # Guard: ensure at least 1 sample in each split
        if n_train < 1:
            n_train = 1

        train_part = shuffled[:n_train]
        val_part   = shuffled[n_train : n_train + n_val]
        test_part  = shuffled[n_train + n_val :]

        splits["train"].extend(train_part)
        splits["val"].extend(val_part)
        splits["test"].extend(test_part)

    return splits


# ---------------------------------------------------------------------------
# Symlink creation
# ---------------------------------------------------------------------------

def make_symlinks(
    splits: Dict[str, List[Sample]],
    output_dir: Path,
) -> None:
    """Create the output directory tree and populate it with symlinks.

    Existing symlinks at the target path are replaced; existing regular files
    or directories raise an error to avoid silent overwrites.

    Parameters
    ----------
    splits:
        Output of :func:`stratified_split`.
    output_dir:
        Root of the yolo_dataset directory to create.
    """
    for split_name, samples in splits.items():
        img_dir = output_dir / "images" / split_name
        lbl_dir = output_dir / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for s in samples:
            img_link = img_dir / f"{s.symlink_stem}.png"
            lbl_link = lbl_dir / f"{s.symlink_stem}.txt"

            for link, target in [(img_link, s.img_path), (lbl_link, s.lbl_path)]:
                if link.is_symlink():
                    link.unlink()
                elif link.exists():
                    raise FileExistsError(
                        f"Cannot create symlink — file already exists: {link}"
                    )
                link.symlink_to(target)


# ---------------------------------------------------------------------------
# data.yaml
# ---------------------------------------------------------------------------

def write_data_yaml(output_dir: Path) -> None:
    """Write the YOLO data.yaml configuration file.

    Parameters
    ----------
    output_dir:
        Root of the yolo_dataset directory.
    """
    config = {
        "path":  str(output_dir),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    2,
        "names": ["structural", "logical"],
    }
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"data.yaml saved → {yaml_path}")


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def print_distribution(splits: Dict[str, List[Sample]]) -> None:
    """Print per-split image counts broken down by category and anomaly type."""
    total_all = sum(len(v) for v in splits.values())

    print("\n" + "=" * 72)
    print(f"{'Split':<8} {'Category':<25} {'Anomaly type':<22} {'N':>5}")
    print("-" * 72)

    split_totals: Dict[str, Dict[str, int]] = {}

    for split_name in ("train", "val", "test"):
        samples = splits[split_name]
        by_key: Dict[str, int] = {}
        for s in samples:
            key = f"{s.category}|{s.anomaly_type}"
            by_key[key] = by_key.get(key, 0) + 1

        split_totals[split_name] = {"logical": 0, "structural": 0, "total": 0}
        for key, count in sorted(by_key.items()):
            cat, atype = key.split("|")
            short = atype.replace("_anomalies", "")
            print(f"{split_name:<8} {cat:<25} {short:<22} {count:>5}")
            split_totals[split_name]["total"] += count
            if "logical" in atype:
                split_totals[split_name]["logical"] += count
            else:
                split_totals[split_name]["structural"] += count
        print()

    print("=" * 72)
    print(f"{'Split':<8} {'Total':>7}  {'Logical':>9}  {'Structural':>12}  {'%':>6}")
    print("-" * 72)
    for split_name in ("train", "val", "test"):
        t = split_totals[split_name]
        pct = 100 * t["total"] / total_all if total_all else 0
        print(
            f"{split_name:<8} {t['total']:>7}  {t['logical']:>9}  "
            f"{t['structural']:>12}  {pct:>5.1f}%"
        )
    print("=" * 72)
    print(f"{'TOTAL':<8} {total_all:>7}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: collect, split, link, report."""
    parser = argparse.ArgumentParser(
        description="Prepare MVTec LOCO-AD anomaly images as a YOLO dataset."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT,
        help=f"Path to mvtec_loco directory (default: {DATASET_ROOT}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory for the YOLO dataset (default: {OUTPUT_DIR}).",
    )
    args = parser.parse_args()

    print("Collecting samples …")
    samples = collect_samples(args.dataset_root)
    print(f"  Found {len(samples)} valid (image, label) pairs")

    print("Splitting (stratified, seed=42) …")
    splits = stratified_split(samples)
    for name, part in splits.items():
        print(f"  {name:<6}: {len(part)} samples")

    print(f"\nCreating symlinks in {args.output_dir} …")
    make_symlinks(splits, args.output_dir)

    write_data_yaml(args.output_dir)

    print_distribution(splits)


if __name__ == "__main__":
    main()
