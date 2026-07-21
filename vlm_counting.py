#!/usr/bin/env python3
"""
vlm_counting.py
用 VLM 對圖片做物件計數，回傳計數結果文字描述。
"""

from __future__ import annotations

import io
import json
import re
import time
from pathlib import Path

from PIL import Image
from google.genai import types as _genai_types

# ---------------------------------------------------------------------------
# 路徑與常數
# ---------------------------------------------------------------------------

SPEC_DIR    = Path.home() / "data/iad/code/results/auto_questions"
MODEL_NAME  = "gemma-4-31b-it"

MAX_RETRIES = 4
RETRY_DELAY = 20.0
RATE_LIMIT_SLEEP = 5

# 單次 API 請求逾時（毫秒），避免連線 stall 時無限阻塞
_API_TIMEOUT_MS = 120_000


# ---------------------------------------------------------------------------
# 工具函式
# ---------------------------------------------------------------------------

def _pil_to_jpeg_part(img: Image.Image) -> _genai_types.Part:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return _genai_types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")


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


def _count_one_object(image: Image.Image, obj: dict, client) -> int | None:
    """對單一可數物件呼叫 VLM 數數，回傳數量（失敗回 None）。"""
    count_prompt = (
        f"請仔細數一下圖片中有幾個{obj['name']}（{obj.get('description', '')}）。"
        f"請逐行或逐區檢查，最後只回答一個數字，不要其他文字。"
    )
    parts = [
        _pil_to_jpeg_part(image),
        _genai_types.Part.from_text(text=count_prompt),
    ]
    contents = [_genai_types.Content(role="user", parts=parts)]

    for attempt in range(1, MAX_RETRIES + 1):
        time.sleep(RATE_LIMIT_SLEEP)
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config=_genai_types.GenerateContentConfig(
                    temperature=0.1,
                    top_p=0.9,
                    top_k=40,
                    http_options=_genai_types.HttpOptions(timeout=_API_TIMEOUT_MS),
                ),
            )
            return _parse_int(response.text)

        except Exception as exc:
            err_lower = f"{type(exc).__name__} {exc}".lower()
            is_retryable = any(kw in err_lower for kw in (
                "429", "quota", "rate", "resource exhausted",
                "503", "unavailable", "timeout", "deadline exceeded",
                "500", "internal",
            ))
            if is_retryable and attempt < MAX_RETRIES:
                print(f"  [計數重試 {attempt}/{MAX_RETRIES}] {RETRY_DELAY:.0f} 秒後重試… ({exc})")
                time.sleep(RETRY_DELAY)
            else:
                break

    return None


def vlm_count(image: Image.Image, category: str, client) -> str | None:
    """
    對圖片做計數。

    1. 載入計數規格書
    2. 如果沒有可數物件，回傳 None（不需要計數）
    3. 對每個可數物件呼叫 VLM 數數
    4. 回傳文字描述

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
        detected = _count_one_object(image, obj, client)

        if detected is None:
            segments.append(f"偵測到 ?（計數失敗）個{name}（正常品預期 {expected} 個）")
            continue

        if expected is not None and detected != expected:
            mark = "⚠️ 數量不符"
        else:
            mark = "✓ 數量正確"
        segments.append(f"偵測到 {detected} 個{name}（正常品預期 {expected} 個）{mark}")

    return "計數分析：" + "；".join(segments)
