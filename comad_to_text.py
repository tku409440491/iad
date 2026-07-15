#!/usr/bin/env python3
"""
comad_to_text.py
把 ComAD 的分割結果轉成文字描述，供 VLM 使用。
"""

import cv2
import numpy as np
from pathlib import Path

COMAD_DIR = Path.home() / "data/iad/code/comad_repo"

# 九宮格方位標籤：欄（左/中/右）× 列（上/中/下），正中央特稱「中央」
_COL_LABELS = ["左", "中", "右"]
_ROW_LABELS = ["上", "中", "下"]


def _grid_position(cx: float, cy: float) -> str:
    """把正規化中心座標 (cx, cy) ∈ [0,1] 轉成九宮格方位字串。"""
    col = 0 if cx < 1/3 else (1 if cx < 2/3 else 2)
    row = 0 if cy < 1/3 else (1 if cy < 2/3 else 2)
    if col == 1 and row == 1:
        return "中央"
    return _COL_LABELS[col] + _ROW_LABELS[row]


def _count_components(folder: Path) -> int:
    """偵測資料夾內 heatresult{i}.jpg 的數量（不含純 heatresult.jpg）。"""
    n = 0
    while (folder / f"heatresult{n}.jpg").exists():
        n += 1
    return n


def comad_describe(category: str, img_id: str, subset: str, n_components: int = 5) -> str:
    """
    把 ComAD 分割結果轉成文字描述。

    Args:
        category: 類別名稱（breakfast_box 等）
        img_id: 圖片編號（資料夾名，例如 "0", "1"）
        subset: "train", "test/good", "test/logical_anomalies", "test/stru", "test/validation"
        n_components: 預期 component 數量（預設 5）；實際以資料夾內 heatresult 檔數為準

    Returns:
        文字描述字串
    """
    folder = COMAD_DIR / f"{category}_heat" / subset / img_id
    if not folder.exists():
        return f"[錯誤] 找不到分割結果資料夾：{folder}"

    # 以實際 heatresult 檔數為準（避免只讀部分 component 導致 argmax 分割錯誤）
    actual_n = _count_components(folder)
    if actual_n == 0:
        return f"[錯誤] {folder} 內沒有 heatresult{{i}}.jpg"
    note = ""
    if actual_n != n_components:
        note = f"（注意：實際偵測到 {actual_n} 個 component，與預期 {n_components} 不同）\n"

    # 讀取每個 component 的活化熱圖（灰階 224×224），堆疊後做 argmax 硬分割
    heatmaps = []
    for i in range(actual_n):
        hm = cv2.imread(str(folder / f"heatresult{i}.jpg"), cv2.IMREAD_GRAYSCALE)
        heatmaps.append(hm.astype(np.float32))
    stack = np.stack(heatmaps, axis=0)          # (N, H, W)
    label_map = np.argmax(stack, axis=0)         # (H, W)，每像素所屬 component
    H, W = label_map.shape

    # 讀原圖並縮到熱圖尺寸，轉 RGB 供取色（cv2 讀進來是 BGR）
    imgo = cv2.imread(str(folder / "imgo.jpg"))
    imgo = cv2.resize(imgo, (W, H), interpolation=cv2.INTER_AREA)
    imgo_rgb = cv2.cvtColor(imgo, cv2.COLOR_BGR2RGB)

    total = H * W
    regions = []
    for comp in range(actual_n):
        mask = label_map == comp
        area = int(mask.sum())
        if area == 0:
            continue  # 空 component 略過
        ys, xs = np.nonzero(mask)
        cx = xs.mean() / W
        cy = ys.mean() / H
        pos = _grid_position(cx, cy)
        mean_rgb = imgo_rgb[mask].mean(axis=0).round().astype(int)
        regions.append({
            "area_ratio": area / total,
            "position":   pos,
            "rgb":        tuple(int(v) for v in mean_rgb),
        })

    # 依面積由大到小排序，重新以 A, B, C… 命名（避免與 KMeans component 編號混淆）
    regions.sort(key=lambda r: r["area_ratio"], reverse=True)

    lines = [f"{note}ComAD 物件組成分析（{len(regions)} 個區域）："]
    for idx, r in enumerate(regions):
        label = chr(ord("A") + idx)
        pct = round(r["area_ratio"] * 100)
        rr, gg, bb = r["rgb"]
        lines.append(
            f"  區域 {label}：位置={r['position']}，面積={pct}%，"
            f"平均顏色 RGB({rr},{gg},{bb})"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    tests = [
        ("train", "0"),
        ("test/good", "0"),
        ("test/logical_anomalies", "0"),
    ]
    for subset, img_id in tests:
        print(f"=== {subset}/{img_id} ===")
        print(comad_describe("breakfast_box", img_id, subset))
        print()
