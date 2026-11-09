# add_patchcore

相對 base 新增：把 PatchCore 偵測報告（整體分數 + 九宮格區域）餵給 VLM，由 VLM 做最終判定（auto 版，使用自動生成的問題清單）。

檔案：`patchcore_vlm_pipeline_auto.py`（另含初版 `patchcore_vlm_pipeline.py`）
