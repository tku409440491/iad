#!/usr/bin/env python3
"""
測試 Gemma 4 的備用 anomaly score 方案
TEST A：強制輸出 JSON 格式
TEST B：簡化 prompt，只問 Yes/No + 數字
TEST C：分兩步驟（先問 Yes/No，再問信心）
執行方式：
    python test_confidence.py --api_key YOUR_API_KEY
"""

import argparse
import json
import re
import sys


def test_confidence(api_key: str, model_name: str = "gemma-4-31b-it"):
    try:
        import google.generativeai as genai
    except ImportError:
        print("[ERROR] 請先安裝：pip install google-generativeai")
        sys.exit(1)

    genai.configure(api_key=api_key)
    print(f"[INFO] 使用模型：{model_name}")
    print("=" * 60)

    model = genai.GenerativeModel(model_name)
    config = genai.GenerationConfig(
        temperature=0.2,
        max_output_tokens=100,
    )

    # ── TEST A：JSON 格式輸出 ─────────────────────────────────────
    print("\n[TEST A] JSON 格式輸出")
    prompt_a = """Look at this question and answer in JSON only.

Question: Are there exactly two oranges visible?

Return only this JSON, nothing else:
{"result": "Yes", "confidence": 0.9}
or
{"result": "No", "confidence": 0.8}"""

    try:
        response = model.generate_content(prompt_a, generation_config=config)
        text = response.text.strip()
        print(f"  原始回應：{text}")

        # 嘗試解析 JSON
        json_match = re.search(r'\{.*?\}', text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            print(f"  解析結果：result={data.get('result')}, confidence={data.get('confidence')}")
            print("  ✅ TEST A 成功")
        else:
            print("  ❌ 找不到 JSON 格式")
    except Exception as e:
        print(f"  ❌ 失敗：{e}")

    # ── TEST B：極簡 prompt ───────────────────────────────────────
    print("\n[TEST B] 極簡 prompt")
    prompt_b = """Answer with exactly one line.
Format: Yes 0.9  (or No 0.2)
Question: Are there exactly two oranges?
Answer:"""

    try:
        response = model.generate_content(prompt_b, generation_config=config)
        text = response.text.strip()
        print(f"  原始回應：{text}")

        # 解析 "Yes 0.9" 或 "No 0.2"
        match = re.search(r'(Yes|No)\s+([\d.]+)', text, re.IGNORECASE)
        if match:
            result = match.group(1)
            confidence = float(match.group(2))
            print(f"  解析結果：result={result}, confidence={confidence}")
            print("  ✅ TEST B 成功")
        else:
            print("  ❌ 解析失敗")
    except Exception as e:
        print(f"  ❌ 失敗：{e}")

    # ── TEST C：分兩步驟 ─────────────────────────────────────────
    print("\n[TEST C] 分兩步驟（先問 Yes/No，再問信心）")

    # Step 1：只問 Yes/No
    prompt_c1 = """Answer only Yes or No.
Question: Are there exactly two oranges?
Answer:"""

    try:
        response1 = model.generate_content(prompt_c1, generation_config=config)
        answer = response1.text.strip()
        print(f"  Step 1 回應：{answer}")

        # 解析 Yes/No
        if re.search(r'\byes\b', answer, re.IGNORECASE):
            result = "Yes"
        elif re.search(r'\bno\b', answer, re.IGNORECASE):
            result = "No"
        else:
            result = None
            print("  ❌ Step 1 無法解析 Yes/No")

        if result:
            # Step 2：問信心分數
            prompt_c2 = f"""You just answered "{result}" to this question:
"Are there exactly two oranges?"

How confident are you? Reply with only a number between 0.0 and 1.0.
Confidence:"""

            response2 = model.generate_content(prompt_c2, generation_config=config)
            conf_text = response2.text.strip()
            print(f"  Step 2 回應：{conf_text}")

            conf_match = re.search(r'[\d.]+', conf_text)
            if conf_match:
                confidence = float(conf_match.group())
                confidence = min(max(confidence, 0.0), 1.0)  # 限制在 0~1
                print(f"  解析結果：result={result}, confidence={confidence}")
                print("  ✅ TEST C 成功（但需要 2 次 API 呼叫）")
            else:
                print("  ❌ Step 2 無法解析信心分數")

    except Exception as e:
        print(f"  ❌ 失敗：{e}")

    # ── 總結 ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[總結與建議]")
    print("  TEST A 成功 → 用 JSON 格式輸出（推薦，最容易解析）")
    print("  TEST B 成功 → 用極簡格式（次選，省 token）")
    print("  TEST C 成功 → 用兩步驟（需 2 次 API，但最穩定）")
    print("  全部失敗   → 放棄信心分數，只用 Yes/No 計算 Accuracy")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--model", type=str, default="gemma-4-31b-it")
    args = parser.parse_args()
    test_confidence(api_key=args.api_key, model_name=args.model)