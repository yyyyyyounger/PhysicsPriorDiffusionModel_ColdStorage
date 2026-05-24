# 自 `5ee18db` 起在本專案上的改動與貢獻摘要

基準提交：`5ee18db0646de74741714c83493a8d3f17c1a8c2`。

本文整理從該提交之後，為了讓 DehazeDDPM 適配自建冷庫／ColdFog 數據、完成微調與推理實驗、加入 netH 物理一致性約束、並提升實驗可復現性所做的主要改動。統計範圍以 `git diff 5ee18db0646de74741714c83493a8d3f17c1a8c2..HEAD` 為主：共 58 個文件變更，約 3369 行新增、140 行刪除；另有少量當前工作區未提交的文檔與配置修訂，見文末註記。

## 1. 將原始 DehazeDDPM 適配到 ColdFog 微調場景

原倉庫主要面向既有 DENSE / NH 配置與公開預訓練權重。這一階段的工作，是把訓練、驗證、推理入口整理成可在本地 ColdFog 數據上穩定運行的實驗流程。

- 新增 `config/Dehaze_ColdFog_finetune.json`、`config/Dehaze_ColdFog_finetune_netH.json`、`config/Dehaze_ColdFog_finetune_resume.json` 等訓練配置，將數據路徑接到本地 `data/finetune`，並統一使用 448 x 576 的冷庫圖像解析度。
- 新增 `config/test_ColdFog_finetune*.json` 系列測試配置，覆蓋只微調 diffusion、同時微調 netH、DDIM 推理、DPM-Solver++ 推理、netH physical 版本等實驗條件。
- 新增 `trainColdFog*.sh`、`testColdFogFinetune*.sh`、`testDENSEDIY.sh`、`testNHDIY.sh` 等腳本，將常用訓練與推理命令固化，降低手動輸入配置路徑與 GPU 參數時出錯的概率。
- 調整 `config/test_DENSE.json`、`config/test_NH.json`，並新增 `config/test_DENSE_diy.json`、`config/test_NH_diy.json`，方便用統一入口在自建數據與原始基準配置上做對比。

可寫入報告的表述：本工作首先完成了 DehazeDDPM 從公開數據集配置到自建 ColdFog 數據集的工程遷移，包括數據路徑、圖像尺寸、權重載入、訓練腳本與推理腳本的系統化整理，為後續微調和消融實驗提供可重複的基礎。

## 2. 修復與擴展模型權重載入、凍結與接續訓練邏輯

原始程式在載入 diffusion 權重、凍結 PreNet / netH，以及多卡測試時存在若干不穩定點。這部分主要集中在 `model/model.py`、`model/base_model.py`、`core/logger.py` 與 `sr.py`。

- 修復 diffusion 網路 `netG` 的 checkpoint 載入方式：不再依賴硬編碼刪除特定 key，而是在訓練階段只合併「當前模型存在且 shape 一致」的參數，避免因模型結構或 schedule buffer 差異造成錯誤載入。
- 修復 netH 的凍結語義：未開啟 `finetune_netH` 時明確凍結所有 netH 參數，確保只更新 diffusion；開啟 `finetune_netH` 時則按層名前綴選擇性解凍主幹、融合、注意力、transmission 與 atmospheric light 相關模塊。
- 為 optimizer 增加兩組 param group：diffusion 使用主學習率 `lr`，netH 使用較小的 `lr_netH`，使兩個子網能以不同步幅聯合微調。
- 增加 `path.resume_stateH_finetune`，支持推理或續訓時優先載入已微調過的 netH 權重，而不是只能載入原始 PreNet 權重。
- 在開啟 `finetune_netH` 時額外保存 `I{iter}_E{epoch}_netH.pth`，使 diffusion checkpoint 與 netH checkpoint 能成對保存和復現。
- 支持接續訓練時復用同一實驗目錄：新增 `path.reuse_experiments_root`，同時在存在 `*_opt.pth` 時恢復 optimizer、iteration 和 epoch；log 文件改為追加模式，避免 resume 後覆蓋先前訓練日誌。

可寫入報告的表述：本工作對原始 checkpoint 載入與參數凍結流程做了工程修復，使模型能在「只微調 diffusion」與「聯合微調 diffusion + netH」兩種模式間明確切換，並支持訓練中斷後的狀態恢復。

## 3. 新增 netH 物理一致性 loss

這是本輪改動中最核心的模型訓練擴展。原模型中 netH 主要作為第一階段條件生成器，提供去霧估計與 transmission 圖給 diffusion。新增的 physical loss 進一步利用深度與散射係數信息約束 netH 的物理中間量。

- 在 `model/model.py` 中新增 `lambda_t`、`lambda_asm`、`current_physical_losses` 等訓練狀態，並在 `optimize_parameters()` 中將總 loss 改為：

```text
loss_total = l_pix + lambda_t * loss_t + lambda_asm * loss_asm
```

- `loss_t`：將 netH 預測的 `out_T` 與由深度和 beta 計算出的 transmission 目標 `exp(-beta * depth)` 做 L1 約束。
- `loss_asm`：用大氣散射模型重建出的 `out_I` 與輸入 hazy 圖做 L1 約束，讓 netH 的 `out_J / out_T / out_A` 組合具有物理一致性。
- 在 `config/Dehaze_ColdFog_finetune_netH_physical.json` 中加入 physical 版本訓練配置：`finetune_netH=true`、`lambda_t=0.01`、`lambda_asm=0.05`，並將訓練與驗證數據接到帶 metadata 的 ColdFog finetune 數據。
- 在 `sr.py` 的 train / validation 流程中統計 `loss_t`、`loss_asm`、`loss_physical_total`，並同步寫入 train log、val log、TensorBoard 與 W&B。
- 在 `model/model.py` 的 validation/test 流程中新增 `update_physical_log()`，使驗證階段也能記錄 physical loss 指標，便於觀察 netH 中間物理量是否穩定。

可寫入報告的表述：在保留 diffusion 去霧主 loss 的基礎上，本工作加入了面向 netH 中間物理量的輔助監督，通過 transmission 一致性與大氣散射重建一致性約束，使第一階段條件圖不只服務於最終生成結果，也具備更清晰的物理解釋。

## 4. 擴展數據讀取以支持 metadata、depth 與 beta

為了讓 physical loss 可訓練，數據層需要在原有 hazy / gt 圖像對之外提供深度圖與散射係數。

- 在 `data/util.py` 中新增 metadata 讀取邏輯，支持 `metadata.csv` 與 `metadata.jsonl`，並可根據 `split` 過濾 train / val / test 樣本。
- metadata 記錄可包含 `hazy`、`gt` / `clear`、`depth`、`beta` 等欄位；相對路徑會根據 `finetune_root` 自動解析。
- 新增 depth 清洗、轉 tensor、resize 等工具函數，處理 NaN / inf / 負值，並在 resize 時保持與圖像相同的空間尺寸。
- 修改 `data/LRHR_dataset.py`，使 `__getitem__()` 在樣本帶有 `depth_path` 時同時返回 `depth` 和 `beta`，並確保圖像與 depth 在 resize、水平翻轉等增強上保持對齊。
- 修改 `data/__init__.py`，使 dataset 建立時可讀取 `metadata_csv`、`metadata_jsonl`、`finetune_root` 等配置。

可寫入報告的表述：為支撐物理先驗監督，本工作將原本只讀取圖像對的數據管線擴展為可讀取 metadata 的多模態樣本管線，使每個訓練樣本可同時攜帶 hazy 圖、gt 圖、depth 與 beta。

## 5. 新增 DDIM 與 DPM-Solver++ 加速推理

原始 `sr3` diffusion 推理主要使用完整 DDPM 反向鏈，推理步數高、速度慢。這一部分在 `model/sr3_modules/diffusion.py` 與新增的 `model/sr3_modules/dpm_solver_pp.py` 中完成。

- 在 `beta_schedule.val` 中新增 `sampler`、`sample_steps`、`ddim_eta` 三個推理配置項。
- 新增 DDIM 採樣流程，支持確定性 DDIM（`ddim_eta=0.0`）與帶隨機性的 DDIM。
- 新增離散時間 VP diffusion 下的二階 multistep DPM-Solver++ 實作，用於少步數快速推理。
- 保留原完整 DDPM 路徑作為 baseline；若 `sampler` 未指定，仍回到原始 DDPM 行為，保持與既有 checkpoint 相容。
- 新增 `config/test_ColdFog_finetune_ddim.json`、`config/test_ColdFog_finetune_dpm_solver_pp.json` 與對應 shell 腳本，方便在相同權重下比較完整 DDPM、DDIM 和 DPM-Solver++ 的速度與效果。
- 在 `README.md` 中補充採樣器配置說明，強調少步推理應調整 `sample_steps`，而不是直接把 `val.n_timestep` 改小。

可寫入報告的表述：本工作在不重新訓練模型的前提下，為驗證與推理階段加入 DDIM 和 DPM-Solver++ 採樣器，使模型可以在完整 DDPM 品質基線之外，進行少步數推理效率與圖像品質的對照實驗。

## 6. 新增多 GPU 推理與多卡訓練測試修復

原始推理流程主要面向單進程推理，多卡使用時容易出現重複處理樣本、DataParallel 訪問方法錯誤或輸出覆蓋問題。

- 重構 `infer.py`，新增 `torch.multiprocessing.spawn` 多 GPU 推理模式：每張 GPU 啟動一個 worker，按樣本 index 交錯切分驗證集。
- 多卡推理時每個 worker 綁定本地 `cuda:{rank}`，並避免在子進程內再次包一層 `DataParallel`。
- 多卡輸出文件名使用原資料集 `Index + 1`，避免不同 worker 同時寫同名文件。
- 每個 worker 分別輸出 `infer_rank{rank}.log` 與局部 metrics，主進程根據樣本數加權彙總整體 PSNR。
- 修復 `model/model.py` 中 `netG` 被 `DataParallel` 包裝時直接訪問 `super_resolution()` 的問題，改為在需要時使用 `netG.module.super_resolution()`。

可寫入報告的表述：本工作將原本單卡推理流程擴展為按樣本切分的多 GPU 推理流程，並修復 DataParallel 下模型方法訪問錯誤，使大規模測試集推理和指標統計更高效、更穩定。

## 7. 加強可復現性與公平對比

為了比較不同 checkpoint、不同採樣器與不同訓練策略，實驗流程需要固定隨機性來源。

- 新增 `core/seed.py`，統一設置 Python、NumPy、PyTorch 和 CUDA 隨機種子。
- 在 `core/logger.py` 中加入 `manual_seed` 與 `manual_seed_deterministic` 默認配置。
- 在 `sr.py` 和 `infer.py` 啟動時調用統一 seed 設置，並將 DataLoader 的 generator / worker seed 一併固定。
- validation / inference 階段按樣本 index 設置 sample seed，使同一張圖在不同 GPU 數、不同分片方式下有一致的隨機起點，利於公平比較 DDPM / DDIM / DPM-Solver++ 與不同 checkpoint。

可寫入報告的表述：本工作統一了訓練、驗證和推理階段的隨機種子控制，降低多卡推理與隨機採樣帶來的評估波動，提升不同實驗設置間比較的公平性。

## 8. 增加可視化、指標保存與實驗分析工具

除最終去霧結果外，physical 版本還需要觀察 netH 的中間輸出。

- 在 `core/metrics.py` 中新增 `save_physical_visuals()`，自動保存 `out_T`、`out_A`、`out_I`、`stage1_output`、`output` 等 netH 中間結果。
- 支持單通道 transmission 圖保存，以及 `out_A` 這類可能是全局或低解析度大氣光圖的尺寸展開。
- 修改 `sr.py` 與 `infer.py`，在 validation / inference 時保存 physical 中間可視化圖，便於分析 transmission、atmospheric light 與重建 hazy 圖是否合理。
- 新增 `plot/plot_train_log.ipynb` 用於繪製訓練曲線；新增 `plot/compare_checkpoints_testset.ipynb` 用於多 checkpoint 測試效果比較。
- 新增 `docs/ColdFog_netH_physical_losses.md`，系統整理 physical loss 的計算來源、日誌含義、可視化文件命名與分析重點。

可寫入報告的表述：本工作不只保存最終去霧結果，也保存 netH 的多個物理中間量，從而支持對模型內部物理分解結果的定性分析，並提供訓練曲線與 checkpoint 對比工具輔助實驗總結。

## 9. 工程清理與文檔補充

- 新增 `.gitignore`，忽略 `__pycache__`、訓練輸出權重等不應進入版本庫的文件。
- 從版本庫中移除已追蹤的 `.pyc` 文件，減少無關二進制文件對 diff 和後續維護的干擾。
- 新增 `.gitattributes`，強制 shell 腳本使用 LF 行尾，修復 Linux 下因 CRLF 導致腳本無法執行的問題。
- 更新 `README.md`，補充 conda 環境、依賴安裝、GPU 指定、ColdFog / DENSE / NH 推理命令、resume 訓練、採樣器配置、多 GPU 推理與 tmux 後台運行說明。

## 10. 提交鏈按主題歸納

| 主題 | 代表提交 |
|------|----------|
| 基礎工程清理 | `2d4e3e4`、`7edb30b`、`61d1c1f`、`71863e2` |
| ColdFog 配置與訓練腳本 | `daf9efb`、`e37352a`、`7fb434b`、`4af9b04`、`f9a9814` |
| 權重載入、凍結與多卡修復 | `c3d6434`、`1ffe351`、`adf17bf` |
| 訓練 resume 與可復現性 | `a5a10ec`、`676c12e` |
| 推理加速與多 GPU 推理 | `4845d49`、`8d06b94`、`d104946`、`e0c91fc` |
| plot / checkpoint 對比 | `a381d21`、`0d8729f`、`da604cb`、`860e003`、`4ada189`、`8b24c57`、`2e1dd02`、`10d56ce`、`877821e` |
| netH physical loss | `ae83e3b`、`f855b8c`、`c215313` |

## 11. 報告中可強調的貢獻點

1. **數據集適配**：將原始 DehazeDDPM 工程遷移到自建 ColdFog 數據，完成訓練、驗證、推理配置與腳本整理。
2. **模型微調策略**：支持只微調 diffusion 與聯合微調 diffusion + netH，並修復權重載入、參數凍結、checkpoint 保存與續訓邏輯。
3. **物理先驗引入**：基於 depth 和 beta 設計 `loss_t` 與 `loss_asm`，對 netH 的 transmission 與大氣散射重建結果施加物理一致性約束。
4. **推理效率提升**：在不重新訓練的情況下加入 DDIM 和 DPM-Solver++，支持少步數推理與完整 DDPM baseline 的公平對比。
5. **實驗可靠性**：增加固定 seed、多 GPU 推理樣本切分、DataParallel 修復、resume 訓練、物理中間量保存與訓練曲線分析工具。

## 12. 當前工作區註記

截至本文件更新時，`git status` 顯示除 `Contribution.md` 外，工作區另有以下尚未提交內容：`config/test_ColdFog_finetune_netH_physical_ddim20.json`、`docs/ColdFog_netH_physical_losses.md` 的修改，以及若干 `docs/*.html` / `docs/ColdFog_netH_physical_losses/` 可視化文檔。這些內容與 physical loss 解釋、DDIM 20-step 測試配置和 HTML 視覺化有關，可在最終提交前再決定是否納入版本庫。
