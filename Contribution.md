# 自 `5ee18db` 起在本專案上的改動與優化摘要

基準提交：`5ee18db0646de74741714c83493a8d3f17c1a8c2`。以下為之後為讓 DehazeDDPM 在自建冷庫／微調數據集上可訓練、可推理、可多卡穩定運行所做的變更歸納。

## 1. 數據與配置（接軌本地數據集）

- **微調訓練**：新增 `config/Dehaze_ColdFog_finetune.json`，指向 `data/finetune` 下 `train_hazy`／`train_gt`、`val_hazy`／`val_gt`，訓練解析度設為 **448×576**，batch、workers、迭代與驗證頻率等按本地資源調整；沿用公開權重路徑 `resume_state`（擴散）與 `resume_stateH`（第一階段 PreNet）。
- **同時微調 PreNet（netH）**：新增 `config/Dehaze_ColdFog_finetune_netH.json`，在上一條基礎上開啟 `finetune_netH`，並為 netH 單獨配置較小學習率 `lr_netH`。
- **測試／驗證**：新增 `config/test_ColdFog_finetune.json`（微調後檢查用，含佔位 `resume_state` 前綴需替換）；新增 `config/test_DENSE_diy.json`、`config/test_NH_diy.json` 並調整 `test_DENSE.json`、`test_NH.json`，用於在自建數據上跑推理與指標。
- **一鍵腳本**：`trainColdFog.sh`、`trainColdFogNetH.sh`、`testColdFogFinetune.sh`、`testDENSEDIY.sh`、`testNHDIY.sh` 指定對應 config，減少手敲路徑錯誤。

## 2. 核心代碼修復與擴展（`model/model.py`）

這些改動是「能正確載入權重、能按意圖凍結／解凍子網、多卡不報錯」的關鍵。

- **擴散網 G 的載入（訓練階段）**  
  舊邏輯對 checkpoint 做固定鍵 `denoise_fn.downs.0.weight` 的 `pop` 再以 `strict=False` 載入，易與實際權重結構不一致。改為在 **train** 階段只合併「當前模型中存在且 shape 一致」的預訓練參數，其餘記錄日誌樣例，避免錯誤覆蓋或依賴硬編碼刪鍵。
- **PreNet（netH）載入與凍結語義**  
  - 未開 `finetune_netH` 時：對 **netH 全部參數** 設 `requires_grad=False`，確保優化器只更新擴散部分，與「只微調 G」的預期一致。  
  - 開啟 `finetune_netH` 時：按層名前綴**選擇性解凍** netH（主幹／融合／注意力等相關模塊），其餘凍結；優化器使用 **兩組 param group**（G 用 `lr`，netH 用可選的 `lr_netH`）。
- **可選載入微調後的 netH**：支持配置項 `path.resume_stateH_finetune`，優先於原始 `resume_stateH` 載入微調後的 PreNet。
- **保存檢查點**：在 `finetune_netH` 時額外保存 `I{iter}_E{epoch}_netH.pth`，便於推理或第二階段單獨載入 netH。
- **多卡 `DataParallel` 與測試**：在 `optimize_parameters` 相關路徑中調用 `super_resolution` 時，若 `netG` 為 `DataParallel`，改為使用 **`netG.module.super_resolution`**，修復多卡下測試／前向訪問錯誤（對應提交說明中的多卡 test 問題）。

## 3. 工程與可復現性

- **`.gitignore`**：忽略 `__pycache__` 等；並從版本庫中移除已追蹤的 `.pyc`，減少噪音與合併衝突。
- **`README.md`**：補充 conda 環境名、PyTorch／CUDA 121 安裝示例、依賴包、指定 GPU、`infer.py` 在 DENSE／NH 自建配置上的用法、tmux 後台跑法，以及 baseline 驗證 PSNR 記錄與 `data/results_analysis.ipynb` 的分析入口。

## 4. Git 提交鏈（按時間從舊到新）

| 提交     | 主題（簡述）                         |
|----------|--------------------------------------|
| `2d4e3e4` | 增加 `.gitignore`                    |
| `7edb30b` | 取消追蹤 `pycache`                   |
| `daf9efb` | 按數據集更新測試相關配置             |
| `c3d6434` | 修復 load network 凍結／載入問題     |
| `1ffe351` | 修復多卡訓練中的 test 問題           |
| `e37352a` | 冷庫微調腳本與配置；netH 解凍與保存  |

---

以上內容可直接用於畢設「工程實現與適配」小節：說明在保留原兩階段 DehazeDDPM 流程的前提下，如何將數據路徑、訓練配置與 `model.py` 中的載入／凍結／多卡行為對齊到自建冷庫微調場景。
