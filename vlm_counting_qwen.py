#!/usr/bin/env python3
"""
vlm_counting_qwen.py
用本地 Qwen2.5-VL 對圖片做物件計數，回傳計數結果文字描述。
邏輯與 vlm_counting.py 相同，只把 API 呼叫換成 _qwen_generate()，計數 prompt 保持完全相同。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image

from qwen_inference import _qwen_generate

# ---------------------------------------------------------------------------
# 路徑與常數
# ---------------------------------------------------------------------------

SPEC_DIR = Path.home() / "data/iad/code/results/auto_questions"


# ---------------------------------------------------------------------------
# 工具函式
# ---------------------------------------------------------------------------

def _parse_int(text: str) -> int | None:
    """從 VLM 回應中取出第一個整數。"""
    match = re.search(r"-?\d+", text)
    return int(match.group()) if match else None


def load_counting_spec(category: str) -> list[dict] | None:
    """載入計數規格書，回傳 countable_objects list，無檔案回傳 None。"""
    path = SPEC_DIR / f"{category}_counting_spec.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("countable_objects")


def _count_one_object(image: Image.Image, obj: dict) -> int | None:
    """對單一可數物件用 Qwen 數數，回傳數量（失敗回 None）。"""
    # 計數 prompt 與 vlm_counting.py 完全相同，確保消融公平
    count_prompt = (
        f"請仔細數一下圖片中有幾個{obj['name']}（{obj.get('description', '')}）。"
        f"請逐行或逐區檢查，最後只回答一個數字，不要其他文字。"
    )
    try:
        raw = _qwen_generate(image, count_prompt, max_new_tokens=64, temperature=0.1)
        return _parse_int(raw)
    except Exception as exc:
        print(f"  [計數失敗] {obj.get('name', '物件')}：{exc}")
        return None


def vlm_count(image: Image.Image, category: str, client=None) -> str | None:
    """
    對圖片做計數（本地 Qwen 版）。
    client 參數保留但忽略（為了與 vlm_counting.vlm_count 介面相容）。

    回傳格式：
    "計數分析：偵測到 16 個圖釘（正常品預期 15 個）⚠️ 數量不符"
    或
    "計數分析：偵測到 15 個圖釘（正常品預期 15 個）✓ 數量正確"
    或
    None（不需要計數）
    """
    spec = load_counting_spec(category)
    if not spec:
        return None  # 無規格書或無可數物件

    segments = []
    for obj in spec:
        name     = obj.get("name", "物件")
        expected = obj.get("expected_count")
        detected = _count_one_object(image, obj)

        if detected is None:
            segments.append(f"偵測到 ?（計數失敗）個{name}（正常品預期 {expected} 個）")
            continue

        if expected is not None and detected != expected:
            mark = "⚠️ 數量不符"
        else:
            mark = "✓ 數量正確"
        segments.append(f"偵測到 {detected} 個{name}（正常品預期 {expected} 個）{mark}")

    return "計數分析：" + "；".join(segments)
