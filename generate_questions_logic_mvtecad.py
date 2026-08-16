#!/usr/bin/env python3
"""
generate_questions_logic.py

從正常圖片自動生成每個類別的邏輯性 Yes/No 問題清單（數量/位置/顏色/種類關係），
取代手寫的 _CATEGORY_RULES。

使用方式：
    python generate_questions_logic.py                           # 所有類別
    python generate_questions_logic.py --category breakfast_box  # 單一類別
    python generate_questions_logic.py --n-images 5              # 用 5 張正常圖
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

# 此版本的 JSON key 與輸出檔尾綴
QUESTIONS_KEY = "logical_questions"
FILE_SUFFIX   = "_logic"


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


def _parse_questions(text: str) -> list[str]:
    """從 Gemma 回傳文字中解析 JSON，取出問題清單。"""
    # 嘗試直接解析
    try:
        data = json.loads(text.strip())
        return data[QUESTIONS_KEY]
    except Exception:
        pass

    # 嘗試從 markdown code block 中提取
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            return data[QUESTIONS_KEY]
        except Exception:
            pass

    # 嘗試找第一個 { ... }
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            return data[QUESTIONS_KEY]
        except Exception:
            pass

    raise ValueError(f"無法從回應中解析 JSON：{text[:300]}")


# ---------------------------------------------------------------------------
# 主要函式
# ---------------------------------------------------------------------------

def generate_questions(
    category: str,
    client,
    n_images: int = 3,
) -> list[str]:
    """
    送正常圖片給 Gemma，生成邏輯性 Yes/No 問題清單。

    回傳問題清單，並存到 OUTPUT_DIR/{category}_logic.json。
    """
    images = _load_normal_images(category, n_images)
    if not images:
        raise FileNotFoundError(f"找不到 {category} 的正常圖片：{MVTECAD_ROOT / category / 'train' / 'good'}")

    prompt = f"""你是一個工業品管專家。以下是 {category} 的正常產品圖片（共 {len(images)} 張）。
請仔細觀察這些正常圖片，找出產品的固定規格，
然後生成一個問題清單，每個問題都是 Yes/No 格式，用來檢查一張新圖片是否符合正常規格。

請從以下面向分別出題：
- 物品種類：圖中有哪些不同種類的物品？
- 物品數量：每種物品各有幾個？數量是否固定？
- 物品位置：各物品在圖片的哪個區域？位置是否固定？
- 物品顏色：各物品應該是什麼顏色？顏色是否與其他物品對應？
- 物品外觀：物品的形狀、紋路、完整度是否正常？有無損壞或污染？
- 物品關係：不同物品之間的相對位置或比例是否正確？

要求：
1. 每個問題只問一件事
2. 正常品的答案應該全部是 Yes
3. 問題要具體，能夠從圖片直接判斷（避免模糊問題如「看起來正常嗎？」）
4. 生成 8–15 個問題
5. 問題編號使用 LQ1, LQ2... 格式
6. 只回傳 JSON 格式，不要其他文字：
{{"logical_questions": ["LQ1: ...", "LQ2: ...", ...]}}"""

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
            questions = _parse_questions(response.text)

            # 存檔
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            out_path = OUTPUT_DIR / f"{category}{FILE_SUFFIX}.json"
            out_path.write_text(
                json.dumps({"category": category, QUESTIONS_KEY: questions}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return questions

        except Exception as exc:
            last_error = exc
            err_lower  = str(exc).lower()

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
    parser = argparse.ArgumentParser(description="從正常圖片自動生成邏輯性 Yes/No 問題清單")
    parser.add_argument("--category", choices=CATEGORIES, default=None, help="只處理指定類別")
    parser.add_argument("--n-images", type=int, default=3, help="使用幾張正常圖（預設 3）")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("請設定環境變數 GEMINI_API_KEY")

    client = genai.Client(api_key=api_key)
    categories = [args.category] if args.category else CATEGORIES

    for cat in categories:
        print(f"\n[{cat}] 生成邏輯問題中（使用 {args.n_images} 張正常圖）…")
        questions = generate_questions(cat, client, n_images=args.n_images)
        print(f"  共生成 {len(questions)} 個問題：")
        for q in questions:
            print(f"    {q}")

    print(f"\n完成！結果存於 {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
