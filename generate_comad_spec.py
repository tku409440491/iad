#!/usr/bin/env python3
"""
generate_comad_spec.py
用 VLM 生成每個類別的正常品組成規格書。
使用 ComAD 分割結果 + 正常品圖片，讓 VLM 描述正常品的物件組成。

使用方式：
    python generate_comad_spec.py                           # 所有類別
    python generate_comad_spec.py --category breakfast_box  # 單一類別
    python generate_comad_spec.py --n-images 5              # 用 5 張正常圖
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

from comad_to_text import comad_describe

# ---------------------------------------------------------------------------
# 路徑與常數
# ---------------------------------------------------------------------------

LOCO_ROOT   = Path.home() / "data/iad/mvtec_loco"
OUTPUT_DIR  = Path.home() / "data/iad/code/results/auto_questions"
MODEL_NAME  = "gemma-4-31b-it"

CATEGORIES = [
    "breakfast_box", "juice_bottle", "pushpins",
    "screw_bag", "splicing_connectors",
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
    good_dir = LOCO_ROOT / category / "train" / "good"
    paths = sorted(good_dir.glob("*.png"))[:n]
    return [Image.open(p).convert("RGB") for p in paths]


def _parse_spec(text: str) -> list[dict]:
    """從 VLM 回傳文字中解析 JSON，取出 component_spec 清單。"""
    # 直接解析
    try:
        data = json.loads(text.strip())
        return data["component_spec"]
    except Exception:
        pass

    # markdown code block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))["component_spec"]
        except Exception:
            pass

    # 第一個 { ... }
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))["component_spec"]
        except Exception:
            pass

    raise ValueError(f"無法從回應中解析 JSON：{text[:300]}")


# ---------------------------------------------------------------------------
# 主要函式
# ---------------------------------------------------------------------------

def generate_comad_spec(
    category: str,
    client,
    n_images: int = 5,
) -> list[dict]:
    """
    送正常圖片 + ComAD 分析結果給 VLM，生成正常品組成規格書。

    回傳 component_spec 清單，並存到 OUTPUT_DIR/{category}_comad_spec.json。
    """
    images = _load_normal_images(category, n_images)
    if not images:
        raise FileNotFoundError(f"找不到 {category} 的正常圖片：{LOCO_ROOT / category / 'train' / 'good'}")

    # 對每張正常圖取得 ComAD 文字描述（train/good 第 i 張 ↔ ComAD train/{i}）
    desc_blocks = []
    for i in range(len(images)):
        desc = comad_describe(category, str(i), "train")
        desc_blocks.append(f"【圖片 {i+1} 的 ComAD 分析】\n{desc}")
    comad_descriptions = "\n\n".join(desc_blocks)

    prompt = f"""你是一個工業品管專家。以下是 {category} 的 {len(images)} 張正常產品圖片，
以及每張圖片的 ComAD 物件組成分析結果。

{comad_descriptions}

請根據這些正常品的組成分析，寫出一份「正常品組成規格書」。
規格書需要描述：
1. 每個主要物件區域通常在什麼位置
2. 每個區域大約佔多少面積比例
3. 每個區域的顏色特徵

注意：
- ComAD 的區域編號（A, B, C...）在不同圖片間可能不對應同一物件
- 請根據顏色和位置來辨識對應的物件
- 面積和顏色允許一定的波動範圍

只回傳 JSON：
{{"component_spec": [
    {{"name": "物件名稱", "typical_position": "位置", "area_range": "X-Y%", "color_desc": "顏色描述", "typical_rgb": [R, G, B]}},
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
            out_path = OUTPUT_DIR / f"{category}_comad_spec.json"
            out_path.write_text(
                json.dumps({"category": category, "component_spec": spec}, ensure_ascii=False, indent=2),
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
    parser = argparse.ArgumentParser(description="用 VLM 生成正常品組成規格書")
    parser.add_argument("--category", choices=CATEGORIES, default=None, help="只處理指定類別")
    parser.add_argument("--n-images", type=int, default=5, help="使用幾張正常圖（預設 5）")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("請設定環境變數 GEMINI_API_KEY")

    client = genai.Client(api_key=api_key)
    categories = [args.category] if args.category else CATEGORIES

    for cat in categories:
        print(f"\n[{cat}] 生成組成規格書中（使用 {args.n_images} 張正常圖）…")
        spec = generate_comad_spec(cat, client, n_images=args.n_images)
        print(f"  共 {len(spec)} 個物件區域：")
        for item in spec:
            print(f"    - {item.get('name')}：位置={item.get('typical_position')}，"
                  f"面積={item.get('area_range')}，顏色={item.get('color_desc')}，"
                  f"RGB={item.get('typical_rgb')}")

    print(f"\n完成！結果存於 {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
