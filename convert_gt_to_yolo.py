#!/usr/bin/env python3
"""
convert_gt_to_yolo.py

Converts MVTec LOCO-AD ground-truth segmentation masks to YOLO bounding-box
label files.

For every anomaly test image the matching single-channel mask is read, each
non-zero pixel region is extracted via connected-component analysis, and a
YOLO label line is written:

    <class> <cx> <cy> <w> <h>   (class 0 = structural, class 1 = logical)

All coordinates are normalised to [0, 1].  Each defect's class is determined
by classifying its name from defects_config.json with semantic pattern rules —
no per-category name lists are hard-coded.

Output structure
----------------
mvtec_loco/{category}/labels/
  logical_anomalies/{img_id}.txt
  structural_anomalies/{img_id}.txt

Usage
-----
  python convert_gt_to_yolo.py                         # all five categories
  python convert_gt_to_yolo.py --category breakfast_box
  python convert_gt_to_yolo.py --dataset-root /path/to/mvtec_loco
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_ROOT = Path.home() / "data" / "iad" / "mvtec_loco"

ALL_CATEGORIES = [
    "breakfast_box",
    "juice_bottle",
    "pushpins",
    "screw_bag",
    "splicing_connectors",
]

ANOMALY_TYPES = ["logical_anomalies", "structural_anomalies"]

YOLO_CLASS: Dict[str, int] = {"logical": 1, "structural": 0}


# ---------------------------------------------------------------------------
# Semantic defect classifier
# ---------------------------------------------------------------------------

# Compound-name overrides: checked BEFORE general patterns because the same
# sub-token carries opposite meanings depending on context.
# (cable_color = wrong component by colour → logical;
#  cable_cut   = cable cut to wrong length → logical assembly error;
#  unknown_cable_color = unidentifiable colour = visual defect → structural)
_COMPOUND_OVERRIDES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"cable_color"), "logical"),       # wrong colour cable = wrong part
    (re.compile(r"cable_cut"), "logical"),          # cut cable = wrong length for assembly
]

# Physical damage / deformation / contamination → structural.
_STRUCTURAL_PATTERNS: List[re.Pattern] = [
    re.compile(r"contaminat"),       # contamination
    re.compile(r"damage"),           # damaged, box_damaged, fruit_damaged
    re.compile(r"crush"),            # toppings_crushed
    re.compile(r"broken"),           # broken, broken_connector, bag_broken, part_broken
    re.compile(r"bent"),             # front_bent
    re.compile(r"^color$"),          # standalone "color" = physical discoloration
    re.compile(r"juice_color"),      # wrong physical property of juice
    re.compile(r"unknown_"),         # unknown identity = visual inspection defect
    re.compile(r"mixed_"),           # physical mixing of materials (contamination-like)
    re.compile(r"rotated"),          # physical orientation defect (rotated_label)
    re.compile(r"flipped"),          # physically inverted object (flipped_connector)
    re.compile(r"not_plugged"),      # cable physically absent from socket
    re.compile(r"^open_"),           # physical open state (open_lever)
    re.compile(r"^incomplete_"),     # physically incomplete icon / label
    re.compile(r"_text_incomplete"), # label text physically missing/cut-off
]

# Global rule violations: count, arrangement, type, quantity → logical.
_LOGICAL_PATTERNS: List[re.Pattern] = [
    re.compile(r"missing_"),                          # missing item
    re.compile(r"_missing_"),                         # 1_missing_long_screw, …
    re.compile(r"wrong_"),                            # wrong type / ratio / fill / location
    re.compile(r"swapped"),                           # items swapped in position
    re.compile(r"misplaced"),                         # item placed incorrectly
    re.compile(r"extra_"),                            # extra / unexpected component
    re.compile(r"overflow"),                          # quantity exceeds maximum
    re.compile(r"underflow"),                         # quantity below minimum
    re.compile(r"additional"),                        # 1_additional_pushpin, …
    re.compile(r"empty_"),                            # container should hold content
    re.compile(r"too_(short|long|much|little|few)"),  # dimension / quantity deviation
    re.compile(r"very_short"),                        # extreme dimensional deviation
]


def classify_defect_semantic(defect_name: str) -> str:
    """Classify a defect name as ``'logical'`` or ``'structural'``.

    **Logical** defects are violations of global assembly rules: missing or
    extra components, wrong component type, wrong quantity, wrong arrangement
    or position.

    **Structural** defects are local physical defects: damage, contamination,
    deformation, wrong material colour, or incorrect physical state.

    Rules were derived from MVTec LOCO-AD defects_config.json for all five
    categories and validated against the ground-truth folder organisation.

    Parameters
    ----------
    defect_name:
        Snake-case identifier from defects_config.json.

    Returns
    -------
    str
        ``'logical'`` or ``'structural'``
    """
    name = defect_name.lower()

    # 1. Compound overrides (context-sensitive tokens)
    for pattern, label in _COMPOUND_OVERRIDES:
        if pattern.search(name):
            # Extra guard: if "unknown" prefix is present it becomes structural
            if label == "logical" and "unknown" in name:
                return "structural"
            return label

    # 2. Count / digit prefix → logical
    #    e.g. "2_additional_pushpins", "1_nectarine_1_tangerine"
    if re.match(r"^\d+_", name):
        return "logical"

    # 3. Structural patterns (physical damage / deformation / contamination)
    for pattern in _STRUCTURAL_PATTERNS:
        if pattern.search(name):
            return "structural"

    # 4. Logical patterns (count / arrangement / type violations)
    for pattern in _LOGICAL_PATTERNS:
        if pattern.search(name):
            return "logical"

    print(f"  WARNING: no pattern matched '{defect_name}' → defaulting to structural")
    return "structural"


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------

def load_pixel_class_map(
    category_path: Path,
) -> Tuple[Dict[int, Tuple[str, int]], Dict[str, str]]:
    """Load defects_config.json and build lookup tables.

    Parameters
    ----------
    category_path:
        Root directory of the category (contains defects_config.json).

    Returns
    -------
    pixel_class_map : dict[int, tuple[str, int]]
        ``{pixel_value: (defect_name, yolo_class)}``
    defect_type_map : dict[str, str]
        ``{defect_name: 'logical' | 'structural'}`` — for the report.
    """
    config = json.loads((category_path / "defects_config.json").read_text())
    pixel_class_map: Dict[int, Tuple[str, int]] = {}
    defect_type_map: Dict[str, str] = {}

    for entry in config:
        name: str = entry["defect_name"]
        pv: int = int(entry["pixel_value"])
        defect_type = classify_defect_semantic(name)
        pixel_class_map[pv] = (name, YOLO_CLASS[defect_type])
        defect_type_map[name] = defect_type

    return pixel_class_map, defect_type_map


# ---------------------------------------------------------------------------
# Mask → YOLO bounding boxes
# ---------------------------------------------------------------------------

def mask_to_yolo_lines(
    mask: np.ndarray,
    pixel_class_map: Dict[int, Tuple[str, int]],
) -> List[str]:
    """Extract YOLO label lines from a single-channel mask array.

    For each distinct non-zero pixel value, connected-component analysis
    identifies spatially separate regions; each component becomes one label
    line.  Empty list is returned for masks with no anomalies.

    Parameters
    ----------
    mask:
        2-D uint8 array (H × W).  Pixel value 0 = background.
    pixel_class_map:
        Output of :func:`load_pixel_class_map`.

    Returns
    -------
    list[str]
        YOLO label lines (no trailing newline per entry).
    """
    h, w = mask.shape
    lines: List[str] = []

    for pv in np.unique(mask):
        if pv == 0:
            continue

        pv_int = int(pv)
        if pv_int not in pixel_class_map:
            print(f"  WARNING: pixel value {pv_int} not in defects_config → skipping")
            continue

        _, yolo_cls = pixel_class_map[pv_int]
        binary = (mask == pv).astype(np.uint8)
        labeled, num_components = ndimage.label(binary)

        for comp_id in range(1, num_components + 1):
            ys, xs = np.where(labeled == comp_id)
            if ys.size == 0:
                continue

            x_min, x_max = int(xs.min()), int(xs.max())
            y_min, y_max = int(ys.min()), int(ys.max())

            cx = ((x_min + x_max) / 2.0) / w
            cy = ((y_min + y_max) / 2.0) / h
            bw = (x_max - x_min + 1) / w
            bh = (y_max - y_min + 1) / h

            lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    return lines


# ---------------------------------------------------------------------------
# Per-category processing
# ---------------------------------------------------------------------------

def process_category(
    category: str,
    dataset_root: Path,
) -> Dict:
    """Convert all ground-truth masks for one category to YOLO label files.

    Label files are written to::

        {dataset_root}/{category}/labels/{anomaly_type}/{img_id}.txt

    Parameters
    ----------
    category:
        Category name, e.g. ``'breakfast_box'``.
    dataset_root:
        Root directory containing all category subdirectories.

    Returns
    -------
    dict
        ``{anomaly_type: {'images': int, 'labels': int, 'defects': {name: int}}}``
        plus a top-level ``'defect_classification'`` entry for the report.
    """
    cat_path = dataset_root / category
    pixel_class_map, defect_type_map = load_pixel_class_map(cat_path)

    stats: Dict = {"defect_classification": defect_type_map}

    for anomaly_type in ANOMALY_TYPES:
        gt_dir = cat_path / "ground_truth" / anomaly_type
        labels_dir = cat_path / "labels" / anomaly_type
        labels_dir.mkdir(parents=True, exist_ok=True)

        n_images = 0
        n_labels = 0
        defect_instance_counts: Dict[str, int] = {}

        if not gt_dir.exists():
            print(f"  [skip] {anomaly_type}: ground_truth dir not found")
            stats[anomaly_type] = {"images": 0, "labels": 0, "defects": {}}
            continue

        for img_folder in sorted(gt_dir.iterdir()):
            if not img_folder.is_dir():
                continue
            mask_path = img_folder / "000.png"
            if not mask_path.exists():
                print(f"  WARNING: mask missing at {mask_path}")
                continue

            img_id = img_folder.name
            mask = np.array(Image.open(mask_path))

            yolo_lines = mask_to_yolo_lines(mask, pixel_class_map)

            label_path = labels_dir / f"{img_id}.txt"
            label_path.write_text(
                "\n".join(yolo_lines) + ("\n" if yolo_lines else "")
            )

            n_images += 1
            n_labels += len(yolo_lines)

            # Count images containing each defect type
            for pv in np.unique(mask):
                if pv == 0:
                    continue
                pv_int = int(pv)
                if pv_int in pixel_class_map:
                    name = pixel_class_map[pv_int][0]
                    defect_instance_counts[name] = (
                        defect_instance_counts.get(name, 0) + 1
                    )

        stats[anomaly_type] = {
            "images": n_images,
            "labels": n_labels,
            "defects": defect_instance_counts,
        }

    return stats


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(report: Dict) -> None:
    """Print a human-readable conversion table to stdout."""
    total_images = 0
    total_labels = 0

    print("\n" + "=" * 66)
    print(f"{'Category':<25} {'Anomaly type':<22} {'Images':>7} {'Labels':>7}")
    print("-" * 66)

    for category, cat_stats in report.items():
        for atype in ANOMALY_TYPES:
            astats = cat_stats.get(atype, {})
            short = atype.replace("_anomalies", "")
            imgs = astats.get("images", 0)
            lbls = astats.get("labels", 0)
            print(f"{category:<25} {short:<22} {imgs:>7} {lbls:>7}")
            total_images += imgs
            total_labels += lbls

    print("=" * 66)
    print(f"{'TOTAL':<25} {'':22} {total_images:>7} {total_labels:>7}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: parse arguments, process categories, write report."""
    parser = argparse.ArgumentParser(
        description="Convert MVTec LOCO-AD ground-truth masks to YOLO label files."
    )
    parser.add_argument(
        "--category",
        choices=ALL_CATEGORIES,
        default=None,
        help="Process only this category (default: all categories).",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT,
        help=f"Path to the mvtec_loco directory (default: {DATASET_ROOT}).",
    )
    args = parser.parse_args()

    categories = [args.category] if args.category else ALL_CATEGORIES
    report: Dict = {}

    for category in categories:
        print(f"\n[{category}]")
        cat_stats = process_category(category, args.dataset_root)
        report[category] = cat_stats

        for atype in ANOMALY_TYPES:
            astats = cat_stats.get(atype, {})
            short = atype.replace("_anomalies", "")
            print(
                f"  {short:<22}  images={astats.get('images', 0):>3}"
                f"  labels={astats.get('labels', 0):>4}"
            )

    print_summary(report)

    report_path = args.dataset_root / "conversion_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Report saved → {report_path}")


if __name__ == "__main__":
    main()
