#!/usr/bin/env python3
"""
generate_counting_spec.py
用 VLM 分析每個類別有哪些可數的重複物件，生成計數規格書。

使用方式：
    python generate_counting_spec.py                           # 所有類別
    python generate_counting_spec.py --category pushpins       # 單一類別
    python generate_counting_spec.py --n-images 5              # 用 5 張正常圖
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import time
from pathlib import Path

from PIL import Image
from google import genai
from google.genai import types as _genai_types

# ---------------------------------------------------------------------------
# 路徑與常數
# ---------------------------------------------------------------------------

MVTECAD_ROOT   = Path.home() / "data/iad/mvtec_ad"
OUTPUT_DIR  = Path.home() / "data/iad/code/results/auto_questions_mvtecad"
MODEL_NAME  = "gemma-4-31b-it"

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

MAX_RETRIES  = 4
RETRY_DELAY  = 20.0
RATE_LIMIT_SLEEP = 5


# ---------------------------------------------------------------------------
# 工具函式
# ---------------------------------------------------------------------------

def _pil_to_jpeg_part(img: Image.Image) -> _genai_types.Part:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return _genai_types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")


def _load_normal_images(category: str, n: int) -> list[Image.Image]:
    good_dir = MVTECAD_ROOT / category / "train" / "good"
    paths = sorted(good_dir.glob("*.png"))[:n]
    return [Image.open(p).convert("RGB") for p in paths]


def _parse_spec(text: str) -> list[dict]:
    """從 VLM 回傳文字中解析 JSON，取出 countable_objects 清單。"""
    # 直接解析
    try:
        return json.loads(text.strip())["countable_objects"]
    except Exception:
        pass

    # markdown code block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))["countable_objects"]
        except Exception:
            pass

    # 第一個 { ... }
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))["countable_objects"]
        except Exception:
            pass

    raise ValueError(f"無法從回應中解析 JSON：{text[:300]}")


# ---------------------------------------------------------------------------
# 主要函式
# ---------------------------------------------------------------------------

def generate_counting_spec(
    category: str,
    client,
    n_images: int = 5,
) -> list[dict]:
    """
    送正常圖片給 VLM，分析有哪些可數的重複物件，生成計數規格書。

    回傳 countable_objects 清單，並存到 OUTPUT_DIR/{category}_counting_spec.json。
    """
    images = _load_normal_images(category, n_images)
    if not images:
        raise FileNotFoundError(f"找不到 {category} 的正常圖片：{MVTECAD_ROOT / category / 'train' / 'good'}")

    prompt = f"""你是一個工業品管專家。以下是 {category} 的 {len(images)} 張正常產品圖片。

請分析這些圖片中有哪些「可精確計數的重複物件」。

只列出符合以下全部條件的物件：
1. 每個物件有明確的邊界，可以清楚區分個體
2. 不是散裝、堆疊、密集排列或部分遮擋的物件
3. 物件之間有明確的間隔或分隔（例如放在各自的格子裡）
4. 從圖片中可以精確計數，不需要估計

例如：放在格子裡的圖釘（✓可數）、分開擺放的螺絲（✓可數）
反例：堆在一起的杏仁（✗散裝）、密集的穀物（✗無法精確數）

對每種可數物件，請描述：
1. 物件名稱
2. 正常品中的精確數量
3. 物件的外觀特徵（用於辨識）

如果產品中沒有符合條件的可精確計數物件，回傳空列表。

只回傳 JSON，不要其他文字：
{{"countable_objects": [
    {{"name": "物件名稱", "expected_count": 數量, "description": "外觀描述"}},
    ...
]}}"""

    parts: list[_genai_types.Part] = []
    for img in images:
        parts.append(_pil_to_jpeg_part(img))
    parts.append(_genai_types.Part.from_text(text=prompt))
    contents = [_genai_types.Content(role="user", parts=parts)]

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        time.sleep(RATE_LIMIT_SLEEP)
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
            )
            spec = _parse_spec(response.text)

            # 存檔
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            out_path = OUTPUT_DIR / f"{category}_counting_spec.json"
            out_path.write_text(
                json.dumps({"category": category, "countable_objects": spec}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return spec

        except Exception as exc:
            last_error = exc
            err_lower  = f"{type(exc).__name__} {exc}".lower()

            is_retryable = any(kw in err_lower for kw in (
                "429", "quota", "rate", "resource exhausted",
                "503", "unavailable", "timeout", "deadline exceeded",
                "500", "internal",
            ))

            if is_retryable and attempt < MAX_RETRIES:
                print(f"  [重試 {attempt}/{MAX_RETRIES}] API 暫時錯誤，{RETRY_DELAY:.0f} 秒後重試… ({exc})")
                time.sleep(RETRY_DELAY)
            else:
                break

    raise RuntimeError(f"[{category}] API 呼叫失敗（嘗試 {MAX_RETRIES} 次）：{last_error}")


# ---------------------------------------------------------------------------
# 主程式
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="用 VLM 生成可數物件計數規格書")
    parser.add_argument("--category", choices=CATEGORIES, default=None, help="只處理指定類別")
    parser.add_argument("--n-images", type=int, default=5, help="使用幾張正常圖（預設 5）")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("請設定環境變數 GEMINI_API_KEY")

    client = genai.Client(api_key=api_key)
    categories = [args.category] if args.category else CATEGORIES

    for cat in categories:
        print(f"\n[{cat}] 分析可數物件中（使用 {args.n_images} 張正常圖）…")
        spec = generate_counting_spec(cat, client, n_images=args.n_images)
        if not spec:
            print("  （無可數的重複物件）")
        else:
            print(f"  共 {len(spec)} 種可數物件：")
            for item in spec:
                print(f"    - {item.get('name')}：預期數量={item.get('expected_count')}，"
                      f"特徵={item.get('description')}")

    print(f"\n完成！結果存於 {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
