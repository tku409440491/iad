# PatchCore-VLM：結合 PatchCore 與視覺語言模型之可解釋工業異常檢測

> 兩階段、**無需訓練任何模型參數**的工業異常檢測框架。PatchCore 提供統計層面的異常分數，視覺語言模型（Gemma 4 31B）提供語意層面的推理判斷，兩者以等權重融合（`0.5 + 0.5`）為最終異常分數，同時偵測**結構性異常**與**邏輯性異常**。

---

## 目錄

- [核心貢獻](#核心貢獻)
- [方法架構](#方法架構)
- [實驗結果](#實驗結果)
- [環境需求與安裝](#環境需求與安裝)
- [資料集下載與目錄結構](#資料集下載與目錄結構)
- [API Key 設定](#api-key-設定)
- [完整執行流程](#完整執行流程)
- [輸出檔案格式](#輸出檔案格式)
- [程式功能對照表](#程式功能對照表)
- [重現消融實驗](#重現消融實驗)
- [其他資料集（MVTec AD / VisA）](#其他資料集mvtec-ad--visa)
- [常見問題排除](#常見問題排除)
- [引用](#引用)
- [授權](#授權)

---

## 核心貢獻

1. **無需訓練的融合框架**：不微調任何參數，即可同時偵測結構性與邏輯性異常。PatchCore 只用正常品建立記憶庫，VLM 直接以預訓練能力推理。
2. **揭示 VLM 的「否決現象」**：當 VLM 擁有最終判定權時，會**系統性推翻 PatchCore 的正確偵測**，導致結構性異常漏檢。改採**分數融合**後，整體（overall）AUROC 從 **0.825 → 0.915**。
3. **LQSQ 分離問題機制 + 計數輔助模組**：把邏輯性問題（LQ）與結構性問題（SQ）分兩組獨立生成與評分，並額外設計計數模組，提升可解釋性與數量判斷準確度。

---

## 方法架構

```
                       ┌──────────────────────────────────────┐
   輸入測試圖片 ──────▶ │  PatchCore（ResNet wide_resnet50_2）  │
                       │  layer2 + layer3 → 池化+上採樣+串接    │
                       │  最近鄰距離 → raw_score               │
                       └──────────────┬───────────────────────┘
                                      │ norm_score = clip((raw-min)/(max-min),0,1)
                                      │ + 3×3 九宮格區域異常描述
                                      ▼
   5 張正常圖 ─┐        ┌──────────────────────────────────────┐
   （離線一次）│        │  VLM（Gemma 4 31B, temperature=0.1）  │
   LQ 問題 ────┼──────▶ │  prompt = 類別 + PatchCore 報告        │
   SQ 問題 ────┤        │        + 計數結果 + LQ/SQ + 評分標準   │
   計數規格書 ─┘        │  回傳 JSON: logical_score/structural_ │
                       │  score/reason                         │
                       └──────────────┬───────────────────────┘
                                      │ vlm_score = max(logical, structural)/100
                                      ▼
                       ┌──────────────────────────────────────┐
                       │  融合  combined = norm×0.5 + vlm×0.5   │
                       │  is_anomaly = combined >= 0.5          │
                       └──────────────────────────────────────┘
```

### 各模組技術細節

**PatchCore 模組**
- 框架：`anomalib 2.5.0`，骨幹 `wide_resnet50_2`（ImageNet 預訓練）。
- 特徵：從 ResNet 的 `layer2`（512 通道）+ `layer3`（1024 通道）提取 patch-level 特徵；各層先做 `AvgPool2d(3,1,1)` 局部平均，再把 `layer3` 雙線性上採樣到 `layer2` 解析度，沿通道串接為 1536 維向量。輸入 resize 至 256×256（layer2 → 32×32 = 1024 個 patch/圖）。
- coreset subsampling 比例 `0.1`。
- 分數正規化：把原始 `train/good` 與 `validation/good` **合併**後 80/20 分割，80% 建立記憶庫、20% 作為驗證集校準 min/max。
  `norm_score = clip((raw_score - img_min) / (img_max - img_min), 0, 1)`

**自動問題生成（離線，每類別一次）**
- 從每類別 `train/good` 隨機抽 5 張正常圖，連同「檢查面向提示」送入 Gemma，生成 Yes/No 問題（每組至多 15 題，生成時 `temperature` 使用預設 1.0）。
- **LQ（邏輯性）面向**：種類、數量、位置、顏色、外觀、物件間關係。
- **SQ（結構性）面向**：表面品質、完整度、污染、顏色一致性、容器/包裝狀態、紋路質感。

**計數輔助模組（離線建規格書 + 推論時清點）**
- 離線：由 VLM 從 5 張正常圖識別「可精確計數」物件並記錄正常數量，產生計數規格書。篩選四條規則：①邊界明確可區分個體 ②非散裝/堆疊/密集/遮擋 ③物件間有明確間隔 ④能精確計數不需估計。
- 推論時：依規格書對**每種**可數物件各呼叫一次 VLM 專門清點，與規格書比對後回饋給主推理。**以推論呼叫次數換取準確度**——有幾種可數物件就多幾次呼叫。

**VLM 推理**
- prompt 內容：產品類別 + PatchCore 偵測報告（整體異常分數 + 3×3 九宮格區域描述）+ 計數結果 + LQ/SQ 問題清單 + 五級評分標準。
- 五級評分：`0–20` 非常確定正常、`21–40` 大致正常、`41–60` 不確定、`61–80` 明確異常、`81–100` 嚴重異常。
- 回傳 JSON：`logical_score`、`structural_score`、`reason`。推論 `temperature=0.1`（穩定輸出）。
- `vlm_score = max(logical_score, structural_score) / 100`。

**融合評分**
- `combined_score = norm_score × 0.5 + vlm_score × 0.5`，閾值 `0.5` 判定異常。

---

## 實驗結果

### 最佳配置（LQSQ + 計數模組）於 MVTec LOCO-AD

| 指標 | Logical | Structural | Overall |
|------|:-------:|:----------:|:-------:|
| F1-max | 0.870 | 0.859 | **0.891** |
| AUROC  | 0.916 | 0.917 | **0.915** |

Good（正常品）誤判率 FP rate：**0.041**

### 與現有方法比較（Logical AUROC）

| 方法 | Logical AUROC | 需訓練 | 可解釋輸出 |
|------|:-------------:|:------:|:----------:|
| **本方法（PatchCore-VLM）** | **0.916** | ❌ | ✅ |
| LogicQA | 0.876 | — | — |
| LogicAD | 0.860 | — | — |
| EfficientAD | 0.868 | ✅ | ❌ |
| SALAD | 0.965 | ✅（需 A100） | ❌ |

> 在**不需訓練**的方法中本方法最高；SALAD 雖更高但需在 A100 上訓練且無可解釋輸出。

### 消融實驗（Overall AUROC）

| 配置 | Overall AUROC |
|------|:-------------:|
| 純 PatchCore | 0.825 |
| 純 VLM（不接收 PatchCore 資訊） | 0.768 |
| PatchCore 輔助 VLM（VLM 做最終判定） | 0.825 |
| **Combined 融合（未加計數）** | **0.898** |

**計數模組增益（Overall F1）**

| 類別 | 未加計數 | 加計數 |
|------|:--------:|:------:|
| pushpins | 0.788 | **0.856** |
| screw_bag | 0.663 | **0.723** |

> **已知限制**：`pushpins` 為均質物件、正常品之間外觀差異大，是表現最弱的類別。

---

## 環境需求與安裝

**硬體**：實驗環境為 WSL2 上的 Ubuntu、NVIDIA RTX 4060 Laptop（8GB VRAM）。ResNet 骨幹在 8GB 上可順利運行；DINOv2 骨幹（消融用）較吃記憶體。

**軟體**：Python 3.10、CUDA 12.x。以下版本取自實際實驗環境（`pip freeze`），建議固定以利重現：

| 套件 | 版本 |
|------|------|
| python | 3.10.20 |
| anomalib | 2.5.0 |
| torch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| lightning / pytorch-lightning | 2.6.5 |
| timm | 1.0.27 |
| scikit-learn | 1.7.2 |
| matplotlib | 3.10.9 |
| opencv-python | 4.13.0.92 |
| pillow | 12.2.0 |
| google-genai | 2.7.0 |

> 消融實驗中的**本地 Qwen2.5-VL 後端**額外需要：`transformers==5.12.0`、`qwen-vl-utils==0.0.14`、`bitsandbytes==0.50.1`、`accelerate==1.14.0`。若不跑 Qwen 消融可略過。

### 安裝步驟

```bash
# 1. 建立 conda 環境
conda create -n iad python=3.10 -y
conda activate iad

# 2. 安裝 PyTorch（依你的 CUDA 版本，以下為 cu128 範例）
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128

# 3. 安裝核心套件
pip install anomalib==2.5.0
pip install google-genai==2.7.0 scikit-learn==1.7.2 matplotlib opencv-python pillow

# 4.（選用）本地 Qwen 消融後端
pip install transformers==5.12.0 qwen-vl-utils==0.0.14 bitsandbytes==0.50.1 accelerate==1.14.0
```

或直接用鎖定檔一次安裝：

```bash
pip install -r requirements.txt
```

> ⚠️ **CUDA 版本注意**：`requirements.txt` 中的 `torch` / `torchvision` 帶有 `+cu128` 標記（對應 CUDA 12.8）。若你的 CUDA 版本不同，`pip install -r requirements.txt` 可能無法安裝這兩個套件——請改用上方「安裝步驟」的手動流程，從 [PyTorch 官方 index](https://pytorch.org/get-started/locally/) 安裝對應你 CUDA 版本的 torch，其餘套件再用 `requirements.txt` 補齊。

---

## 資料集下載與目錄結構

**本 repo 不包含任何資料集**，需自行下載。

| 資料集 | 類別數 | 授權 | 下載 |
|--------|:------:|------|------|
| **MVTec LOCO-AD**（主要） | 5 | CC BY-NC-SA 4.0（僅限非商業） | [MVTec 官網](https://www.mvtec.com/company/research/datasets/mvtec-loco) |
| MVTec AD（選用） | 15 | CC BY-NC-SA 4.0 | [MVTec 官網](https://www.mvtec.com/company/research/datasets/mvtec-ad) |
| VisA（選用） | 12 | CC BY 4.0 | [VisA GitHub](https://github.com/amazon-science/spot-diff) |

### 目錄結構

> ⚠️ **重要**：目前程式碼以 `Path.home() / "data/iad/..."` 寫死絕對路徑（**未來版本將改為相對路徑/可設定路徑**）。在改版前，請務必把 repo 與資料集放到下列位置，否則路徑會找不到。

```
~/data/iad/
├── code/                       ← clone 本 repo 到這裡
│   ├── patchcore_train.py
│   ├── generate_questions_logic.py
│   ├── ... （其餘程式）
│   └── results/                ← 自動產生（已 .gitignore，不進版控）
│       ├── patchcore_models/           # PatchCore checkpoints
│       ├── auto_questions/             # LQ/SQ/計數規格書 JSON
│       └── *_results.jsonl             # 推論結果
│
├── mvtec_loco/                 ← 自行下載解壓
│   ├── breakfast_box/
│   │   ├── train/good/
│   │   ├── validation/good/
│   │   └── test/{good, logical_anomalies, structural_anomalies}/
│   ├── juice_bottle/
│   ├── pushpins/
│   ├── screw_bag/
│   └── splicing_connectors/
│
├── mvtec_ad/                   ← 選用；{類別}/test/{good, 各異常類型}/
└── visa/visa_pytorch/          ← 選用；{類別}/test/{good, bad}/（.JPG）
```

### MVTec LOCO-AD 各類別圖片數量

| 類別 | train/good | validation/good | test/good | logical | structural |
|------|:----:|:----:|:----:|:----:|:----:|
| breakfast_box | 351 | 62 | 102 | 83 | 90 |
| juice_bottle | 335 | 54 | 94 | 142 | 94 |
| pushpins | 372 | 69 | 138 | 91 | 81 |
| screw_bag | 360 | 60 | 122 | 137 | 82 |
| splicing_connectors | 360 | 60 | 119 | 108 | 85 |

> 測試集合計 **1,568 張**（good 575 + logical 561 + structural 432）。

---

## API Key 設定

VLM 透過 **Google AI Studio API** 呼叫（免費額度每日約 1,500 次）。API key 以環境變數讀取：

```bash
export GEMINI_API_KEY="你的_API_KEY"
# 建議寫入 ~/.bashrc 以便持久：
echo 'export GEMINI_API_KEY="你的_API_KEY"' >> ~/.bashrc && source ~/.bashrc
```

> ⚠️ **切勿**把 API key 寫進程式碼或提交到 git。所有需要 API 的程式都只從環境變數 `GEMINI_API_KEY` 讀取。

---

## 完整執行流程

以下以 **MVTec LOCO-AD、最佳配置（LQSQ + 計數）** 為例。一個完全沒接觸過本專案的人，照此順序即可跑完整套實驗。

> 所有指令都在 `~/data/iad/code/` 下、`conda activate iad` 後執行。
> 想先小規模驗證，各推論指令都可加 `--test-only`（每類別每子集只跑第 1 張）。

### 步驟總覽與依賴關係

```
[步驟1 訓練 PatchCore] ──┐
[步驟2 生成 LQ 問題] ────┤
[步驟3 生成 SQ 問題] ────┼──▶ [步驟5 融合推論] ──▶ [步驟6 計算指標]
[步驟4 生成計數規格書] ──┘
```

步驟 1–4 彼此獨立、可任意順序；步驟 5 需要 1–4 全部完成；步驟 6 需要步驟 5 的 JSONL。

---

### 步驟 1：訓練 PatchCore（建立記憶庫）

**不需 API key。** 只用正常品建庫。

```bash
# 全部 5 個類別：
python patchcore_train.py

# 或單一類別：
python patchcore_train.py --category breakfast_box
```

- **輸入**：`~/data/iad/mvtec_loco/{類別}/{train,validation}/good/`
- **輸出**：`results/patchcore_models/{類別}/Patchcore/{類別}/vN/weights/lightning/model.ckpt`
- **預期時間**：ResNet 骨幹約每類別 2–4 分鐘（RTX 4060）。
- **常用參數**：`--backbone {wide_resnet50_2, dinov2}`、`--layers`（DINOv2 用）、`--max-train-images N`（few-shot）、`--seed`。

### 步驟 2：生成邏輯性問題（LQ）

**需 API key。**

```bash
python generate_questions_logic.py --n-images 5           # 全部類別
# python generate_questions_logic.py --category breakfast_box --n-images 5
```

- **輸入**：`{類別}/train/good/` 前 5 張。
- **輸出**：`results/auto_questions/{類別}_logic.json`
- **預期時間**：每類別數十秒（1 次 API 呼叫）。

### 步驟 3：生成結構性問題（SQ）

```bash
python generate_questions_structural.py --n-images 5
```

- **輸出**：`results/auto_questions/{類別}_structural.json`

### 步驟 4：生成計數規格書

```bash
python generate_counting_spec.py --n-images 5
```

- **輸出**：`results/auto_questions/{類別}_counting_spec.json`
- **說明**：無「可精確計數」物件的類別會產生空清單（`countable_objects: []`）——此為正常，推論時該類別會自動略過計數。

### 步驟 5：融合推論（最佳配置）

**需 API key。** 依賴步驟 1–4 的全部產物。

```bash
# 快速驗證（每類別每子集第 1 張）：
python patchcore_vlm_pipeline_combined_lqsq_counting.py --category breakfast_box --mode text --test-only --reset

# 完整跑全部類別：
python patchcore_vlm_pipeline_combined_lqsq_counting.py --mode text
```

- **輸入**：PatchCore checkpoint + LQ/SQ 問題 + 計數規格書 + 測試圖片。
- **輸出**：`results/patchcore_vlm_text_combined_lqsq_counting_results.jsonl`（逐圖一行）。
- **預期時間**：Gemma 每張約 30 秒（含 rate-limit 等待）；完整 1,568 張很久，且可能超過每日免費額度——**本程式支援斷點續跑**（見下）。
- **常用參數**：
  - `--reset`：清除既有結果從頭跑。**不加 `--reset` 直接重跑 = 從斷點繼續**（已完成的圖會跳過）。
  - `--test-only`：每子集第 1 張，快速煙霧測試。
  - `--anomaly-only`：只跑異常圖，跳過 good。
  - `--retry-errors`：只重跑上次失敗（reason 含 api_error / parse_error）的圖。
  - `--mode {text, heatmap}`：PatchCore 資訊以文字九宮格或熱力圖傳給 VLM。
  - `--backbone {resnet, dinov2}`、`--layers`：對應不同 checkpoint。

### 步驟 6：計算指標

**不需 API key。** 純讀 JSONL。

```bash
python compute_metrics.py --file results/patchcore_vlm_text_combined_lqsq_counting_results.jsonl --score combined
```

- **輸出**：印出 LOGICAL / STRUCTURAL / OVERALL 三段，各含逐類別 + 平均的 F1-max、AUROC、fixed-F1、good-FP 率。
- **`--score` 選項**：
  - `combined`（預設）：`norm_score×w_pc + vlm_score×w_vlm`（可用 `--w-pc / --w-vlm` 調權重）。
  - `patchcore`：只用偵測器分數。
  - `vlm`：只用 VLM 分數。
- **`--all`**：掃描 `results/` 下所有 `*_results.jsonl`，一次比較多個實驗（自動跳過樣本數 < 100 的檔案）。

---

## 輸出檔案格式

推論結果 JSONL 每行一張圖片，主要欄位如下（以 `..._combined_lqsq_counting_results.jsonl` 為例）：

```json
{
  "category": "breakfast_box",
  "stem": "000",
  "gt_class": "good",              // good / logical / structural（LOCO）
  "is_anomaly": false,             // 預測結果（combined >= 0.5）
  "anomaly_type": "none",          // none / logical / structural
  "patchcore_norm_score": 0.51,    // PatchCore 正規化分數 (0–1)
  "vlm_logical_score": 0,          // VLM 邏輯分數 (0–100)
  "vlm_structural_score": 0,       // VLM 結構分數 (0–100)
  "vlm_score": 0.0,                // max(logical, structural)/100
  "combined_score": 0.25,          // norm×0.5 + vlm×0.5
  "norm_score": 0.25,              // = combined_score（供 AUROC 計算統一欄位）
  "counting_text": "計數分析：偵測到 15 個圖釘（正常品預期 15 個）✓ 數量正確",
  "reason": "…VLM 的推理說明…",
  "raw_score": 35.9,               // PatchCore 原始分數
  "patchcore_above_threshold": false,
  "mode": "text"
}
```

問題檔（`results/auto_questions/`）格式：

```json
// {category}_logic.json
{"category": "breakfast_box", "logical_questions": ["LQ1: …", "LQ2: …"]}

// {category}_structural.json
{"category": "breakfast_box", "structural_questions": ["SQ1: …", "SQ2: …"]}

// {category}_counting_spec.json
{"category": "pushpins", "countable_objects": [
    {"name": "圖釘", "expected_count": 15, "description": "金屬圖釘…"}
]}
```

---

## 程式功能對照表

### 主線（MVTec LOCO-AD）

| 程式 | 功能 |
|------|------|
| `patchcore_train.py` | 訓練 PatchCore 記憶庫（`--backbone`、`--max-train-images` few-shot、`--val-mode`、`--few-mode`） |
| `generate_questions_logic.py` | 生成 LQ（邏輯性）問題清單 |
| `generate_questions_structural.py` | 生成 SQ（結構性）問題清單 |
| `generate_counting_spec.py` | 生成計數規格書 |
| `gemini_inference.py` | VLM 呼叫與 prompt 建構的**共用核心**（各 pipeline 皆重用其 prompt 函式） |
| `vlm_counting.py` | 計數模組（`load_counting_spec` / `vlm_count`） |
| `compute_metrics.py` | 計算 AUROC / F1-max / fixed-F1 / good-FP（`--file` / `--all`） |

### 推論 Pipeline（由簡到繁，對應消融）

| 程式 | 對應配置 |
|------|----------|
| `patchcore_vlm_pipeline_nopc.py` | 純 VLM（不接收 PatchCore 資訊） |
| `patchcore_vlm_pipeline_auto.py` | PatchCore 輔助 VLM（VLM 做最終判定） |
| `patchcore_vlm_pipeline_combined.py` | Combined 融合（未加 LQSQ/計數） |
| `patchcore_vlm_pipeline_combined_lqsq.py` | Combined + LQSQ 分離問題 |
| **`patchcore_vlm_pipeline_combined_lqsq_counting.py`** | **Combined + LQSQ + 計數（最佳配置）** |

### 消融/延伸變體

| 程式 | 差異 |
|------|------|
| `patchcore_vlm_pipeline_combined_lqsq_counting_maxpixel.py` | PatchCore 報告改「異常最高點百分比座標」+ 一致性引導 |
| `patchcore_vlm_pipeline_combined_lqsq_counting_indval.py` | 用獨立 `validation/good` 校準（`indval/` checkpoint） |
| `patchcore_vlm_pipeline_combined_lqsq_counting_fewshot.py` | few-shot full 模式（5 張建庫、同批圖校準） |
| `patchcore_vlm_pipeline_combined_lqsq_counting_qwen.py` | VLM 後端換本地 **Qwen2.5-VL 7B（4-bit）** |
| `qwen_inference.py` / `vlm_counting_qwen.py` | 本地 Qwen 推論與計數（prompt 與 Gemma 版完全相同） |
| `patchcore_vlm_pipeline_combined_comad_lqsq.py` | 加入 ComAD 組成分析（需 ComAD 分割輸出） |

---

## 重現消融實驗

消融的關鍵在於：**同一套訓練好的 PatchCore 與問題檔**，只更換推論 pipeline，再用同一支 `compute_metrics.py` 評估。

```bash
# 前置：步驟 1–4 完成一次即可（所有消融共用）

# 純 PatchCore（AUROC 0.825）：用任一含 patchcore_norm_score 的結果檔，--score patchcore
python compute_metrics.py --file results/patchcore_vlm_text_combined_lqsq_counting_results.jsonl --score patchcore

# 純 VLM（0.768）：
python patchcore_vlm_pipeline_nopc.py --mode text
python compute_metrics.py --file results/patchcore_vlm_text_nopc_results.jsonl --score vlm

# PatchCore 輔助 VLM（VLM 最終判定，0.825）：
python patchcore_vlm_pipeline_auto.py --mode text
python compute_metrics.py --file results/patchcore_vlm_text_auto_results.jsonl --score vlm

# Combined 未加計數（0.898）：
python patchcore_vlm_pipeline_combined_lqsq.py --mode text
python compute_metrics.py --file results/patchcore_vlm_text_combined_lqsq_results.jsonl --score combined

# Combined + 計數（最佳，0.915）：
python patchcore_vlm_pipeline_combined_lqsq_counting.py --mode text
python compute_metrics.py --file results/patchcore_vlm_text_combined_lqsq_counting_results.jsonl --score combined

# 一次比較所有結果檔：
python compute_metrics.py --all
```

---

## 其他資料集（MVTec AD / VisA）

流程與 LOCO 相同，程式檔名加對應後綴，checkpoint / 問題檔 / 結果各自獨立儲存，**不與 LOCO 混用**。評估為 **overall only**（good vs 所有異常）+ 各異常類型 Recall。

| 資料集 | 訓練 | 問題生成 | 推論 | 指標 |
|--------|------|---------|------|------|
| **MVTec AD**（15 類） | `patchcore_train_mvtecad.py` | `generate_*_mvtecad.py` | `patchcore_vlm_pipeline_mvtecad.py` | `compute_metrics_mvtecad.py` |
| **MVTec AD 2**（8 類） | `patchcore_train_mvtecad2.py` | `generate_*_mvtecad2.py` | `patchcore_vlm_pipeline_mvtecad2.py` | `compute_metrics_mvtecad.py` |
| **VisA**（12 類） | `patchcore_train_visa.py` | `generate_*_visa.py` | `patchcore_vlm_pipeline_visa.py` | `compute_metrics_mvtecad.py` |

範例（VisA 的 candle）：

```bash
python patchcore_train_visa.py --category candle
python generate_questions_logic_visa.py --category candle --n-images 5
python generate_questions_structural_visa.py --category candle --n-images 5
python generate_counting_spec_visa.py --category candle --n-images 5
python patchcore_vlm_pipeline_visa.py --category candle --mode text --test-only --reset
python compute_metrics_mvtecad.py --file results/patchcore_vlm_visa_results.jsonl --score combined
```

> VisA 圖片副檔名為 `.JPG`（大寫），程式已處理；MVTec AD 2 的測試集在 `test_public/{good,bad}`。

---

## 常見問題排除

**Q：找不到資料集 / checkpoint 路徑？**
A：程式目前以 `~/data/iad/...` 寫死絕對路徑。請確認 repo 在 `~/data/iad/code/`、LOCO 資料集在 `~/data/iad/mvtec_loco/`。（未來版本會改相對路徑。）

**Q：`請先設定環境變數 GEMINI_API_KEY`？**
A：`export GEMINI_API_KEY="…"`（見 [API Key 設定](#api-key-設定)）。訓練與計算指標不需要，只有問題生成與推論需要。

**Q：推論跑到一半中斷了，要重跑全部嗎？**
A：不用。推論結果**逐圖即時寫入 JSONL**，直接**重跑同一指令且不加 `--reset`** 即可從斷點續跑（已完成的圖會跳過）。

**Q：出現 API 錯誤 / JSON 解析失敗（reason 含 api_error / parse_error）？**
A：先確認 API 額度未用盡，再用 `--retry-errors` 只重跑失敗的圖，不影響已成功的記錄。程式對 API 逾時/暫時錯誤有自動重試（含 120 秒逾時保護，避免連線 stall 卡死）。

**Q：DINOv2 骨幹報錯 / 記憶體不足？**
A：DINOv2（`vit_base_patch14_dinov2`）固定需要 518×518 輸入且較吃 VRAM，在 8GB 上可能吃緊；主結果用 ResNet 即可。DINOv2 僅供骨幹消融。

**Q：某類別的計數規格書是空的？**
A：正常。該類別沒有符合「可精確計數」四條規則的物件（如單一瓶子），推論時會自動略過計數、退回一般 LQ/SQ 流程。

**Q：PatchCore 的 threshold ≈ max、`patchcore_above_threshold` 幾乎都是 False？**
A：這是預期行為。驗證集只含正常品，門檻校準為「正常分數的最大值」；本框架最終判定用的是**融合分數 `combined_score >= 0.5`**，不是這個 threshold。

---

## 引用

若本專案對你的研究有幫助，歡迎引用。**論文正式發表後，此處會更新為正式引用資訊與連結：**

> 📄 **論文連結（發表後更新）**：`<TODO: 在此填入論文/預印本連結>`

```bibtex
@article{patchcore_vlm,
  title   = {結合 PatchCore 與視覺語言模型之可解釋工業異常檢測},
  author  = {TODO: 作者},
  year    = {TODO: 年份},
  note    = {論文連結 TODO},
}
```

---

## 授權

- **程式碼**：`<TODO: 選擇並填入授權，如 MIT / Apache-2.0>`
- **資料集**：本 repo 不含任何資料集。MVTec LOCO-AD 與 MVTec AD 採 **CC BY-NC-SA 4.0（僅限非商業用途）**，VisA 採 CC BY 4.0；請至各自官網下載並遵守其授權條款。
- **模型**：Gemma 4 透過 Google AI Studio API 使用，請遵守 Google 的使用條款；Qwen2.5-VL 遵守其模型授權。
