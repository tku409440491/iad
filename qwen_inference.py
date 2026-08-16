#!/usr/bin/env python3
"""
qwen_inference.py
用本地 Qwen2.5-VL 7B（4-bit 量化）做推論，取代 Gemma API。
Prompt 建構完全重用 gemini_inference.py 的函式，確保消融實驗公平。
"""

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

MODEL_PATH = "/home/wilson/data/iad/models/qwen25vl-7b"

_model = None
_processor = None


def load_qwen():
    """載入模型（只載入一次，之後重用）"""
    global _model, _processor
    if _model is None:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH, quantization_config=bnb, torch_dtype=torch.float16, device_map="cuda:0"
        )
        _processor = AutoProcessor.from_pretrained(
            MODEL_PATH, min_pixels=256 * 28 * 28, max_pixels=768 * 28 * 28
        )
    return _model, _processor


def _qwen_generate(image, prompt, max_new_tokens=512, temperature=0.1):
    """單次推論：傳入 PIL Image 和 prompt，回傳文字"""
    model, processor = load_qwen()
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs, vids = process_vision_info(msgs)
    inputs = processor(text=[text], images=imgs, videos=vids, padding=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else None,
            top_p=0.9,
            top_k=40,
        )
    return processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]


def qwen_inference_patchcore_lqsq_counting(image, patchcore_text, counting_text, category, client=None, **kwargs):
    """
    跟 gemini_inference_patchcore_lqsq_counting() 相同介面，但用本地 Qwen。
    client 參數保留但忽略（為了介面相容）。
    Prompt 選擇邏輯完全複製 gemini 版，確保消融公平。
    """
    from gemini_inference import (
        _load_auto_questions_lq_sq, _load_auto_questions,
        _build_prompt_patchcore_lqsq_counting, _build_prompt_patchcore_lq_sq,
        _build_prompt_patchcore_auto, _build_prompt_patchcore,
    )

    lq_sq = _load_auto_questions_lq_sq(category)
    if lq_sq is not None:
        lq, sq = lq_sq
        if counting_text is not None:
            prompt = _build_prompt_patchcore_lqsq_counting(patchcore_text, counting_text, lq, sq, category)
        else:
            prompt = _build_prompt_patchcore_lq_sq(patchcore_text, lq, sq, category)
    else:
        questions = _load_auto_questions(category)
        if questions:
            print(f"  [警告] {category} 缺 LQ/SQ 問題檔，fallback 到單組 auto 問題")
            prompt = _build_prompt_patchcore_auto(patchcore_text, questions, category)
        else:
            print(f"  [警告] {category} 無任何問題清單，fallback 到 PatchCore 預設 prompt")
            prompt = _build_prompt_patchcore(patchcore_text, category)

    raw = _qwen_generate(image, prompt, max_new_tokens=512, temperature=0.1)
    return _parse_scores(raw)


# ---------------------------------------------------------------------------
# JSON 容錯解析
# ---------------------------------------------------------------------------

def _finalize(logical, structural, reason) -> dict:
    """把分數收斂到 0–100 並推算 is_anomaly / anomaly_type（與 gemini _parse_response 一致）。"""
    try:
        logical = int(max(0, min(100, int(float(logical)))))
    except Exception:
        logical = 0
    try:
        structural = int(max(0, min(100, int(float(structural)))))
    except Exception:
        structural = 0
    is_anomaly = max(logical, structural) >= 50
    if logical < 50 and structural < 50:
        anomaly_type = "none"
    elif logical >= structural:
        anomaly_type = "logical"
    else:
        anomaly_type = "structural"
    return {
        "is_anomaly":       is_anomaly,
        "anomaly_type":     anomaly_type,
        "logical_score":    logical,
        "structural_score": structural,
        "reason":           str(reason),
    }


def _loose_fix(s: str) -> str:
    """修常見 JSON 瑕疵：單引號、尾逗號、全形冒號、智慧引號。"""
    import re
    s = s.strip()
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    s = s.replace("：", ":")
    s = re.sub(r",\s*}", "}", s)   # 物件尾逗號
    s = re.sub(r",\s*]", "]", s)   # 陣列尾逗號
    s = s.replace("'", '"')        # 單引號 → 雙引號
    return s


def _parse_scores(raw_text: str) -> dict:
    """
    從 VLM 回應中抽取 logical_score, structural_score, reason。
    強化容錯：處理 ```json 區塊、缺引號、單引號、尾逗號、額外文字等情況。
    """
    import json
    import re

    text = (raw_text or "").strip()

    # 收集候選 JSON 字串：優先 ```json 區塊 → 第一個 {...} → 原文
    candidates: list[str] = []
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1))
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        candidates.append(brace.group(0))
    candidates.append(text)

    # 逐一嘗試（原字串 + 寬鬆修正後）
    for cand in candidates:
        for attempt in (cand, _loose_fix(cand)):
            try:
                d = json.loads(attempt)
            except Exception:
                continue
            if isinstance(d, dict) and ("logical_score" in d or "structural_score" in d):
                return _finalize(d.get("logical_score", 0), d.get("structural_score", 0), d.get("reason", ""))

    # 正則直接抽數字（連 JSON 結構都壞掉時的最後手段）
    lg = re.search(r"logical[_\s]*score\D{0,10}(\d{1,3})", text, re.IGNORECASE)
    st = re.search(r"structural[_\s]*score\D{0,10}(\d{1,3})", text, re.IGNORECASE)
    rs = re.search(r"reason[\"'）]?\s*[:：]\s*[\"']?([^\"'\n}]{1,300})", text, re.IGNORECASE)
    if lg or st:
        return _finalize(
            int(lg.group(1)) if lg else 0,
            int(st.group(1)) if st else 0,
            rs.group(1).strip() if rs else "regex_extracted",
        )

    # 全失敗
    return {
        "is_anomaly":       False,
        "anomaly_type":     "none",
        "logical_score":    0,
        "structural_score": 0,
        "reason":           f"parse_error: {text[:200]}",
    }
