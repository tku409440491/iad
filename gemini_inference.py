#!/usr/bin/env python3
"""
gemini_inference.py

使用 Gemini 2.5 Flash 對工業瑕疵圖片進行視覺推理。

Pipeline：
    YOLO 偵測（conf=0.10）
    → draw_detections()：在原圖畫上 bbox 標注
    → gemini_inference()：送標注圖給 Gemini，取得異常判斷結果

主要公開函式：
    draw_detections(image_path, detections) -> PIL.Image
    gemini_inference(annotated_img, detections, category, client) -> dict

使用範例：
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    annotated = draw_detections(image_path, detections)
    result = gemini_inference(annotated, detections, "breakfast_box", client)
"""

from __future__ import annotations

import io
import json
import os
import re
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from google import genai
from google.genai import types as _genai_types

# ---------------------------------------------------------------------------
# 常數設定
# ---------------------------------------------------------------------------

# 類別對應顏色（RGB）與名稱
_CLASS_COLOR: dict[int, tuple[int, int, int]] = {
    0: (220, 50,  50),   # 結構異常（structural） → 紅色
    1: (50,  90, 220),   # 邏輯異常（logical）    → 藍色
}
_CLASS_NAME: dict[int, str] = {0: "structural", 1: "logical"}

# Few-shot 正常品範例設定
_LOCO_ROOT = Path.home() / "data/iad/mvtec_loco"
_FEWSHOT_N = 2

# 高置信度 YOLO structural 直接信任
_HIGH_CONF_THRESHOLD = 0.80
_HIGH_CONF_SKIP_CATEGORIES = {"screw_bag"}


def _load_fewshot_images(category: str) -> list:
    """載入該類別 train/good/ 的前 N 張正常圖片"""
    good_dir = _LOCO_ROOT / category / "train" / "good"
    imgs = sorted(good_dir.glob("*.png"))[:_FEWSHOT_N]
    return [Image.open(p).convert("RGB") for p in imgs]


def _pil_to_jpeg_part(img: Image.Image) -> _genai_types.Part:
    """將 PIL 圖片轉成 JPEG bytes 並包裝為 types.Part，避免直接傳 PIL 物件觸發 SDK 內部編碼問題"""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return _genai_types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")


# 各產品類別的規格說明（注入 Gemini prompt 的背景知識）
_CATEGORY_RULES: dict[str, str] = {
    "breakfast_box": (
    "早餐盒正常狀態：左側格子恰好有 2 顆橘子（tangerine，圓形橘色）與 1 顆油桃（nectarine，扁圓形、紅黃色外皮），"
    "三顆水果的排列順序不固定，但數量與種類必須完全正確（2橘子＋1油桃），不可多也不可少。"
    "右側格子上半部為穀物（cereals），下半部為香蕉片（banana chips）與杏仁（almonds）混合，"
    "兩者的比例與相對位置是固定的，不可過多或過少。"
    "左右格子內容不可互換。"
    ),
    "juice_bottle": (
    "果汁瓶正常狀態：瓶身中央貼有一張正方形標籤，標籤上有對應口味的水果圖示（cherry/orange/banana）；"
    "瓶身下方貼有一張長方形標籤，印有『100% Juice』文字。"
    "兩張標籤皆需端正貼附（不可旋轉、不可偏移），文字與圖示需完整清晰。"
    "瓶內液體顏色需與水果圖示一致：櫻桃→深紅褐色、橘子→淡黃色、香蕉→乳白色。"
    "液位正常範圍為瓶肩至瓶頸之間，不可低於瓶身一半（過少）或完全看不到空氣（過多）。"
    ),
    "pushpins": (
    "圖釘盒正常狀態：透明收納盒內有 3 列 × 5 欄共 15 個格子，每個格子中恰好放有 1 支黃色圖釘。"
    "圖釘方向不固定，但每格數量必須剛好是 1 支，不可多也不可少。"
    "所有格子之間的分隔片（separator）必須存在且完整。"
    "圖釘本體（針尖與塑膠帽）不可彎曲、破損或變色。"
    ),
    "screw_bag": (
    "螺絲袋正常狀態：透明夾鏈袋內恰好包含 6 個零件：\n"
    "  - 長螺絲（long screw）× 1：帶六角帽頭，螺紋桿較長\n"
    "  - 短螺絲（short screw）× 1：帶六角帽頭，長度約為長螺絲的一半\n"
    "  - 螺帽（nut）× 2：六角形金屬螺帽\n"
    "  - 墊圈（washer）× 2：圓形開口金屬環\n"
    "判斷步驟：請逐一數出圖中每種零件的數量，確認是否與規格完全一致，數量差一個也算異常。\n"
    "螺絲長度判斷：長螺絲約為短螺絲的 2 倍長，兩者差異需明顯。"
    "若圖中兩支螺絲長度相近（差異不到 1.5 倍），則視為異常（可能是兩支都是短螺絲或兩支都是長螺絲）。\n"
    "所有零件應為銀色金屬光澤，不可破損、斷裂或變色。"
    "袋子本身不可破損。"
    ),
    "splicing_connectors": (
    "接線連接器正常狀態：畫面中有左右兩個連接器，中間由恰好一條電線相連，不可多也不可少。"
    "電線顏色與連接器夾孔數量必須對應：黃色線→2孔連接器（T2）、藍色線→3孔連接器（T3）、紅色線→5孔連接器（T5）。"
    "左右兩個連接器的夾孔數量必須完全相同（對稱），電線插入左側的第幾個孔，右側也必須插入對應位置的孔。"
    "電線必須完整連續，中間不可有斷裂或切斷的痕跡。"
    "電線長度需足夠，左右兩個連接器之間應有明顯間距（至少連接器自身寬度的兩倍以上），不可過短導致兩個連接器緊靠在一起。"
    "所有夾孔的橙色壓桿需閉合（不可翻起），電線需確實插入接口內。"
    "電線絕緣層不可破損，連接器本體不可破裂。"
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
    # 高置信度 structural 直接信任（screw_bag 除外）
    if category not in _HIGH_CONF_SKIP_CATEGORIES:
        high_conf_str = [
            d for d in detections
            if int(d["class_id"]) == 0 and float(d["conf"]) >= _HIGH_CONF_THRESHOLD
        ]
        if high_conf_str:
            best_conf = max(float(d["conf"]) for d in high_conf_str)
            return {
                "is_anomaly":   True,
                "anomaly_type": "structural",
                "reason":       f"YOLO 高置信度偵測到結構異常（conf={best_conf:.2f}），直接判定。",
            }

    prompt = _build_prompt(detections, category)

    fewshot_imgs = _load_fewshot_images(category)
    parts: list[_genai_types.Part] = []
    for i, img in enumerate(fewshot_imgs):
        parts.append(_pil_to_jpeg_part(img))
        parts.append(_genai_types.Part.from_text(text=(
            f"Reference image {i+1}: a normal (defect-free) product. "
            "Use this to understand the overall appearance and layout. "
            "Do NOT match pixel-level details; rely on the product spec rules below."
        )))
    parts.append(_pil_to_jpeg_part(annotated_img))
    parts.append(_genai_types.Part.from_text(text=prompt))
    contents = [_genai_types.Content(role="user", parts=parts)]

    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        # 每次呼叫前固定等待 5 秒
        time.sleep(5)
        try:
            response = client.models.generate_content(
                model="gemma-4-31b-it",
                contents=contents,
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
