#!/usr/bin/env python3
"""
gemini_inference_logicqa.py

gemini_inference.py 的 LogicQA 版本：使用逐題問答式 prompt 取代原本的規格描述。
除 _CATEGORY_RULES 外，所有程式碼與 gemini_inference.py 完全相同。
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from google import genai


# ---------------------------------------------------------------------------
# 常數設定
# ---------------------------------------------------------------------------

# 類別對應顏色（RGB）與名稱
_CLASS_COLOR: dict[int, tuple[int, int, int]] = {
    0: (220, 50,  50),   # 結構異常（structural） → 紅色
    1: (50,  90, 220),   # 邏輯異常（logical）    → 藍色
}
_CLASS_NAME: dict[int, str] = {0: "structural", 1: "logical"}

# 各產品類別的規格說明（LogicQA 版本：逐題問答式 prompt）
_CATEGORY_RULES: dict[str, str] = {
    "breakfast_box": (
        "請逐一回答以下問題（Yes/No），任何一題回答 No 即表示有異常：\n"
        "Q1: Are there exactly two tangerines visible on the left-hand side of the breakfast box?\n"
        "Q2: Is there one nectarine visible on the left-hand side of the breakfast box?\n"
        "Q3: Does the right-hand side of the breakfast box have cereals in the upper portion?\n"
        "Q4: Is there a mix of banana chips and almonds in the lower portion of the right-hand side of the breakfast box?\n"
        "Q5: Are the fruits (tangerines and nectarine) only on the left-hand side, and are the cereals with banana chips and almonds only on the right-hand side?\n"
        "若所有問題皆為 Yes，判定正常；否則判定為邏輯異常並說明哪題為 No。"
    ),
    "juice_bottle": (
        "請先判斷瓶內果汁顏色（深紅褐色=cherry、淡黃色=orange、乳白色=banana），再逐一回答對應問題：\n"
        "【Cherry】Q1: Is the fill level at least 90% capacity with a small gap at top?\n"
        "Q2: Is there a central label with a cherry icon clearly visible?\n"
        "Q3: Is there a lower label displaying '100% Juice'?\n"
        "Q4: Is the juice color a deep reddish-brown consistent with cherry juice?\n"
        "Q5: Are the two labels vertically aligned with center label above lower label?\n"
        "【Orange】Q1: Is the fill level at least 90% capacity but not completely full?\n"
        "Q2: Is the center label positioned in the middle with an orange icon clearly visible?\n"
        "Q3: Does the center label have a light orange background?\n"
        "Q4: Is the lower label attached to the lower part with '100% Juice' in bold font?\n"
        "Q5: Are the labels vertically aligned?\n"
        "【Banana】Q1: Does the juice appear creamy light yellow, typical of banana juice?\n"
        "Q2: Is there a central label with a banana icon at the center?\n"
        "Q3: Is there a lower label reading '100% Juice'?\n"
        "Q4: Does the fill level reach at least 90% capacity with a small gap at top?\n"
        "Q5: Are there exactly two labels, one center and one lower?\n"
        "若所有問題皆為 Yes，判定正常；否則判定為邏輯異常並說明哪題為 No。"
    ),
    "pushpins": (
        "請針對圖中每個格子逐一回答以下問題（Yes/No），任何格子有任何一題為 No 即表示有異常：\n"
        "Q1: Is there exactly one pushpin visible in the compartment?\n"
        "Q2: Is the pushpin yellow in color?\n"
        "Q3: Is the compartment transparent, allowing the pushpin to be visible?\n"
        "Q4: Is the pushpin visible against a contrasting background?\n"
        "若所有格子所有問題皆為 Yes，判定正常；否則判定為邏輯異常並說明哪個格子哪題為 No。"
    ),
    "screw_bag": (
        "請逐一回答以下問題（Yes/No），任何一題回答 No 即表示有異常：\n"
        "Q1: Are there exactly 6 components visible in the bag?\n"
        "Q2: Is there exactly one long screw (with hexagonal head)?\n"
        "Q3: Is there exactly one short screw, approximately half the length of the long screw?\n"
        "Q4: Is the length difference between the two screws clearly visible (long screw is about 2x the short screw)?\n"
        "Q5: Are there exactly two nuts (hexagonal metal pieces)?\n"
        "Q6: Are there exactly two washers (circular open metal rings)?\n"
        "Q7: Are all components silver metallic in color without damage or discoloration?\n"
        "若所有問題皆為 Yes，判定正常；否則判定為邏輯異常並說明哪題為 No。"
    ),
    "splicing_connectors": (
        "請先判斷電線顏色（yellow=T2、blue=T3、red=T5），再逐一回答對應問題：\n"
        "【共用問題】Q1: Are there exactly two splicing connectors visible?\n"
        "Q2: Do both connectors have the same number of wire clamps?\n"
        "Q3: Is there exactly one cable connecting the two connectors?\n"
        "Q4: Do the connectors have transparent bodies with orange levers?\n"
        "Q5: Is the cable connected to the same position on both connectors (mirror symmetry)?\n"
        "Q6: Does the cable appear longer than the length of one splicing connector?\n"
        "Q7: Is the cable continuous without any cuts or breaks?\n"
        "【Yellow/T2】Q8: Does each connector have exactly two orange levers?\n"
        "【Blue/T3】Q8: Does each connector have exactly three orange levers?\n"
        "【Red/T5】Q8: Does each connector have exactly five orange levers?\n"
        "若所有問題皆為 Yes，判定正常；否則判定為邏輯異常並說明哪題為 No。"
    ),
}

# 字型路徑（找不到時 fallback 到 Pillow 內建字型）
_FONT_BOLD  = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_FONT_LABEL_SIZE = 22    # 標籤文字大小（px）

# API 呼叫失敗時的預設回傳值
_FALLBACK: dict = {
    "is_anomaly":   False,
    "anomaly_type": "none",
    "reason":       "parse_error",
}


# ---------------------------------------------------------------------------
# 在圖片上繪製偵測框
# ---------------------------------------------------------------------------

def draw_detections(
    image_path: str | Path,
    detections: list[dict],
) -> Image.Image:
    """
    在原圖上畫出 YOLO 偵測框並標註類別名稱與信心分數。

    框的顏色：
        structural（class 0） → 紅色
        logical   （class 1） → 藍色

    參數
    ----
    image_path : 原圖路徑
    detections : YOLO 偵測結果列表，每個元素為 dict，包含：
        class_id (int)  : 類別編號（0 或 1）
        conf     (float): 信心分數（0.0 ~ 1.0）
        x1, y1, x2, y2 (float): 框的像素座標（原圖尺寸）

    回傳
    ----
    PIL.Image：畫好框的 RGB 圖片
    """
    img  = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # 載入字型：優先使用系統 TrueType，失敗則用 Pillow 內建
    try:
        font = ImageFont.truetype(_FONT_BOLD, _FONT_LABEL_SIZE)
    except OSError:
        font = ImageFont.load_default(size=_FONT_LABEL_SIZE)

    for det in detections:
        cls_id = int(det["class_id"])
        conf   = float(det["conf"])
        x1, y1 = float(det["x1"]), float(det["y1"])
        x2, y2 = float(det["x2"]), float(det["y2"])

        color = _CLASS_COLOR.get(cls_id, (128, 128, 128))
        label = f"{_CLASS_NAME.get(cls_id, str(cls_id))} {conf:.2f}"

        # 框線（3px）
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # 標籤背景（與框同色）
        txt_box = draw.textbbox((x1, y1 - _FONT_LABEL_SIZE - 4), label, font=font)
        draw.rectangle(
            [txt_box[0] - 2, txt_box[1] - 2, txt_box[2] + 2, txt_box[3] + 2],
            fill=color,
        )
        # 白色文字
        draw.text((x1, y1 - _FONT_LABEL_SIZE - 4), label, fill=(255, 255, 255), font=font)

    return img


# ---------------------------------------------------------------------------
# 組成 Gemini Prompt
# ---------------------------------------------------------------------------

def _build_prompt(detections: list[dict], category: str) -> str:
    """
    根據類別規格與偵測摘要，組成送給 Gemini 的完整 prompt。
    """
    rule = _CATEGORY_RULES.get(category, "請根據圖片判斷是否有工業瑕疵。")

    n_structural = sum(1 for d in detections if int(d["class_id"]) == 0)
    n_logical    = sum(1 for d in detections if int(d["class_id"]) == 1)

    if detections:
        det_summary = (
            f"YOLO 共偵測到 {len(detections)} 個疑似異常區域："
            f"{n_structural} 個結構異常（紅框）、{n_logical} 個邏輯異常（藍框）。"
        )
    else:
        det_summary = "YOLO 未偵測到任何異常框（模型信心不足或可能為正常品）。"

    return f"""你是一個工業產品視覺品管專家。

【產品規格】
產品類別：{category}
規格說明：{rule}

【YOLO 偵測摘要】
{det_summary}

【【檢測任務】
這是一張工業檢測圖片，產品類別是 {category}。
圖中紅色框是 YOLO 偵測到的疑似結構異常區域（物理損傷、污染等）。
藍色框是 YOLO 偵測到的疑似邏輯異常區域（數量錯誤、位置錯誤、缺少元件等）。

重要：若圖中有標注框，請優先以框住的區域作為判斷依據，仔細檢查框內的內容是否符合產品規格。
不要忽略框的位置而只憑整體印象判斷。
請根據圖片內容判斷：
1. 這張圖片是否有異常？
2. 如果有，主要是結構異常還是邏輯異常？
3. 具體的異常原因是什麼？

只回傳 JSON，不要其他文字：
{{"is_anomaly": true/false, "anomaly_type": "structural"/"logical"/"none", "reason": "..."}}"""


# ---------------------------------------------------------------------------
# 解析 Gemini 回應
# ---------------------------------------------------------------------------

def _parse_response(text: str) -> dict:
    """
    解析 Gemini 回傳的文字為結構化 dict。

    能處理以下格式：
    - 純 JSON：{"is_anomaly": true, ...}
    - Markdown 包裝：```json\\n{...}\\n```

    解析失敗時回傳 _FALLBACK。
    """
    # 去除 markdown code fence 標記
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()

    # 取出第一個完整的 JSON 物件
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        return {**_FALLBACK}

    try:
        raw = json.loads(match.group())
        return {
            "is_anomaly":   bool(raw.get("is_anomaly", False)),
            "anomaly_type": str(raw.get("anomaly_type", "none")),
            "reason":       str(raw.get("reason", "")),
        }
    except (json.JSONDecodeError, ValueError):
        return {**_FALLBACK}


# ---------------------------------------------------------------------------
# Gemini 推理主函式
# ---------------------------------------------------------------------------

def gemini_inference(
    annotated_img: Image.Image,
    detections: list[dict],
    category: str,
    client,
    max_retries: int = 4,
    retry_delay: float = 20.0,
) -> dict:
    """
    將標注後的圖片送給 Gemini 2.5 Flash 進行異常判斷。

    若 YOLO 沒有偵測到任何框（detections 為空），仍會送原圖讓 Gemini 自行判斷。

    參數
    ----
    annotated_img : 已畫好 YOLO bbox 的 PIL.Image
    detections    : YOLO 偵測結果，用於組成 prompt 摘要
    category      : 產品類別字串，例如 'breakfast_box'
    client        : genai.Client 實例
    max_retries   : 遇到 rate limit 等可重試錯誤時的最大嘗試次數（預設 3）
    retry_delay   : 每次重試前等待秒數（預設 20）

    回傳
    ----
    dict：{"is_anomaly": bool, "anomaly_type": str, "reason": str}
          解析失敗時 reason 為 "parse_error" 或 "api_error: ..."
    """
    prompt = _build_prompt(detections, category)

    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        # 每次呼叫前固定等待 5 秒
        time.sleep(5)
        try:
            response = client.models.generate_content(
                model="gemma-4-31b-it",
                contents=[annotated_img, prompt],
            )
            return _parse_response(response.text)

        except Exception as exc:
            last_error = exc
            err_lower  = str(exc).lower()

            # 判斷是否為可重試的暫時性錯誤
            is_retryable = any(kw in err_lower for kw in (
                "429", "quota", "rate", "resource exhausted",
                "503", "unavailable", "timeout", "deadline exceeded",
                "500", "internal",
            ))

            if is_retryable and attempt < max_retries:
                print(
                    f"  [重試 {attempt}/{max_retries}] "
                    f"API 暫時錯誤，{retry_delay:.0f} 秒後重試…  ({exc})"
                )
                time.sleep(retry_delay)
            else:
                break   # 非可重試錯誤，或已達最大次數

    print(f"  [錯誤] Gemini API 呼叫失敗（嘗試 {max_retries} 次）：{last_error}")
    return {**_FALLBACK, "reason": f"api_error: {last_error}"}


# ---------------------------------------------------------------------------
# 測試用輔助函式
# ---------------------------------------------------------------------------

def _get_gt_defect_names(
    category: str,
    img_id: str,
    anomaly_type: str,
    dataset_root: Path,
) -> list[str]:
    """
    從 ground truth mask 讀取該圖片實際含有的瑕疵名稱列表。

    參數
    ----
    category     : 產品類別
    img_id       : 圖片編號（例如 '000'）
    anomaly_type : 'logical_anomalies' 或 'structural_anomalies'
    dataset_root : mvtec_loco 根目錄

    回傳
    ----
    list[str]：瑕疵名稱列表
    """
    cfg_path  = dataset_root / category / "defects_config.json"
    mask_path = (
        dataset_root / category / "ground_truth" / anomaly_type / img_id / "000.png"
    )

    pv_to_name = {
        d["pixel_value"]: d["defect_name"]
        for d in json.loads(cfg_path.read_text())
    }

    mask = np.array(Image.open(mask_path))
    return [
        pv_to_name.get(int(pv), f"unknown_{pv}")
        for pv in np.unique(mask)
        if pv > 0
    ]


def _yolo_inference(image_path: Path, conf: float = 0.10) -> list[dict]:
    """
    對單張圖片執行 YOLO 推理，回傳像素座標格式的偵測結果。

    參數
    ----
    image_path : 圖片路徑
    conf       : 信心分數門檻（預設 0.10）

    回傳
    ----
    list[dict]：每個元素包含 class_id, conf, x1, y1, x2, y2（像素座標）
    """
    from ultralytics import YOLO

    best_pt = Path.home() / "data/iad/code/runs/yolo_baseline/weights/best.pt"
    model   = YOLO(str(best_pt))

    results = model.predict(
        source=str(image_path),
        conf=conf,
        iou=0.5,
        imgsz=640,
        verbose=False,
    )

    detections = []
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "class_id": int(box.cls[0]),
                "conf":     float(box.conf[0]),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })
    return detections


# ---------------------------------------------------------------------------
# 測試入口
# ---------------------------------------------------------------------------

def run_test() -> None:
    """
    測試：對 breakfast_box 前 3 張 logical anomaly 圖片跑完整 pipeline，
    印出 ground truth 瑕疵名稱、Gemini 回應及判斷正確性。
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise EnvironmentError("請先設定環境變數 GEMINI_API_KEY")

    client = genai.Client(api_key=api_key)

    dataset_root = Path.home() / "data/iad/mvtec_loco"
    category     = "breakfast_box"
    anomaly_type = "logical_anomalies"
    test_img_dir = dataset_root / category / "test" / anomaly_type

    img_files = sorted(test_img_dir.glob("*.png"))[:3]

    correct = 0
    print("=" * 64)
    print(f"測試目標：{category} / {anomaly_type}（前 3 張）")
    print("=" * 64)

    for img_path in img_files:
        img_id = img_path.stem
        print(f"\n── 圖片：{img_id}.png")

        # 1. Ground truth 瑕疵名稱
        gt_defects = _get_gt_defect_names(
            category, img_id, anomaly_type, dataset_root
        )
        print(f"   GT 瑕疵  ：{', '.join(gt_defects)}")

        # 2. YOLO 推理（conf=0.10）
        detections = _yolo_inference(img_path, conf=0.10)
        n_str = sum(1 for d in detections if d["class_id"] == 0)
        n_log = sum(1 for d in detections if d["class_id"] == 1)
        print(f"   YOLO 偵測：共 {len(detections)} 框（結構 {n_str}、邏輯 {n_log}）")

        # 3. 畫框
        annotated = draw_detections(img_path, detections)

        # 4. Gemini 推理
        result = gemini_inference(annotated, detections, category, client)

        # 5. 評估：logical anomaly 應判為 is_anomaly=True 且 anomaly_type="logical"
        is_correct = result["is_anomaly"] and result["anomaly_type"] == "logical"
        if is_correct:
            correct += 1

        verdict = "✓ 正確" if is_correct else "✗ 錯誤"
        print(f"   Gemini   ：{verdict}")
        print(f"     is_anomaly   = {result['is_anomaly']}")
        print(f"     anomaly_type = {result['anomaly_type']}")
        print(f"     reason       = {result['reason']}")

    print("\n" + "=" * 64)
    print(f"正確率：{correct} / {len(img_files)} 張")
    print("=" * 64)


if __name__ == "__main__":
    run_test()
