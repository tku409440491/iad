"""
路徑設定：所有路徑相對於 code/ 或資料根目錄，不寫死個人家目錄。

- CODE_DIR    ：專案 code/ 目錄（本檔位於 code/src/）。
- DATA_ROOT   ：資料集根目錄（mvtec_loco、mvtec_ad、visa 等放在這裡）。
                預設為 CODE_DIR 的上一層；可用環境變數 IAD_DATA_ROOT 覆寫。
- RESULTS_DIR ：實驗結果（checkpoint、問題檔、JSONL、圖）存放於 code/results/。
"""
import os
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("IAD_DATA_ROOT", CODE_DIR.parent)).expanduser()
RESULTS_DIR = CODE_DIR / "results"
MODELS_DIR = RESULTS_DIR / "patchcore_models"   # LOCO 版 PatchCore checkpoint
