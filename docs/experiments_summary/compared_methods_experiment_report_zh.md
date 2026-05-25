# 對比方法與實驗配置報告

生成日期：2026-05-25，供論文第 5 章對比方法 / 配置小節使用。

本報告基於對 `config/`、`experiments/`、日誌、檢查點、腳本及模型程式碼的唯讀探索。本次僅產出本 Markdown 報告。

## 執行摘要

本倉庫目前支援基於 DehazeDDPM 的 ColdFog 適配實驗線。可復現且有程式碼支撐的方法包括：

1. 在 DENSE/NH 檢查點上的 DehazeDDPM 預訓練基線。
2. 僅微調擴散模型的 ColdFog 微調。
3. 聯合微調 `netH` 的 ColdFog 微調。
4. 聯合微調 `netH` 並加入物理一致性損失的 ColdFog 微調。
5. 推理採樣器變體：完整 DDPM、DDIM 與 DPM-Solver++。

DCP、DehazeNet、AOD-Net 與 PPDM 未在本倉庫中實作。除非後續補充外部實作或結果，否則不應將其表述為本地復現方法。它們可保留為計劃中 / 外部基線，或從最終定量表格中移除。

目前最強的已完成訓練為物理損失 ColdFog 實驗：

- `experiments/Dehaze_ColdFog_finetune_netH_physical_v1_260508_123402`
- 最佳驗證 PSNR：`20.322`（`I75000_E268`）
- 最終驗證 PSNR：`20.261`（`I100000_E358`）

然而，未找到該物理損失實驗在 ColdFog 測試集上已驗證的推理結果。這是撰寫最終主要定量結論前最高優先級的缺失實驗。

## 論文撰寫建議框架

對於 5.1.3 小節，最清晰的表述並非「大量無關基線均在本地復現」，而是：

> 對比重點在於逐步增強的 DehazeDDPM 系 ColdFog 適配設定，並以預訓練 DehazeDDPM 檢查點量化域間差距。DCP、DehazeNet、AOD-Net 等傳統與 CNN 基線作為候選外部基線，僅在相同 ColdFog 測試協議下完成推理後，才應進入最終定量表格。

方法表可分為兩組：

1. **已驗證本地方法**：有霧輸入、DehazeDDPM 預訓練 DENSE/NH（若已測試）、僅微調擴散、微調擴散 + `netH`、微調擴散 + `netH` + 物理損失。
2. **候選外部基線 / 計劃方法**：DCP、DehazeNet、AOD-Net、PPDM。

DDIM、DPM-Solver++ 等採樣器變體應描述為推理配置或效率消融，而非獨立去霧方法。

## 方法證據鏈

| 方法或組件 | 本地狀態 | 證據 |
| --- | --- | --- |
| DehazeDDPM / SR3 擴散 | 已實作 | `README.md`；`model/networks.py` 選擇 `sr3`/`ddpm`；`model/model.py` 定義包裝類 `DDPM`；`model/sr3_modules/diffusion.py` 實作擴散訓練與採樣。 |
| 第一階段 `netH` / PreNet | 已實作 | `model/networkHelper.py` 定義 `MPRfusion`；日誌顯示 `Network H structure: MPRfusion, with parameters: 2,779,939`。 |
| ColdFog 僅微調擴散 | 已實作，後續實驗已完成 | `config/Dehaze_ColdFog_finetune.json`、`config/Dehaze_ColdFog_finetune_resume.json`、`experiments/Dehaze_ColdFog_finetune_only_diffusion_260417_152053`。 |
| ColdFog 微調擴散 + `netH` | 已實作且已完成 | `config/Dehaze_ColdFog_finetune_netH.json`、`experiments/Dehaze_ColdFog_finetune_netH_260406_223953`。 |
| ColdFog 物理一致性損失 | 已實作，驗證階段已完成 | `config/Dehaze_ColdFog_finetune_netH_physical.json`、`model/model.py`、`docs/ColdFog_netH_physical_losses.md`、`experiments/Dehaze_ColdFog_finetune_netH_physical_v1_260508_123402`。 |
| DDIM | 已實作 | `model/sr3_modules/diffusion.py` 含 `ddim_sample_loop`；`config/test_ColdFog_finetune_ddim.json` 設定 `sampler: ddim`、`sample_steps: 100`、`ddim_eta: 0.0`。 |
| DPM-Solver++ | 已實作 | `model/sr3_modules/dpm_solver_pp.py`；`model/sr3_modules/diffusion.py` 含 `dpm_solver_pp_sample_loop`；`config/test_ColdFog_finetune_dpm_solver_pp.json` 設定 `sampler: dpm_solver_pp`。 |
| DCP | 未實作 | 未找到可執行的本地程式碼 / 配置 / 入口。 |
| DehazeNet | 未實作 | 未找到可執行的本地程式碼 / 配置 / 入口。 |
| AOD-Net | 未實作 | 未找到可執行的本地程式碼 / 配置 / 入口。 |
| PPDM | 未實作 | 未找到可執行的本地程式碼 / 配置 / 入口。 |

## 配置清單

### 訓練配置

| 配置 | 角色 | 數據 | 初始化 | 關鍵設定 | 狀態 |
| --- | --- | --- | --- | --- | --- |
| `config/Dehaze_DENSE.json` | 原始 DENSE 訓練基線 | `./data/Dense_Haze/*` | 擴散從零訓練；`netH` 使用 `DENSE_net_g_120000.pth` | `lr=1e-4`、`n_iter=2000000`、DDPM 2000 步 | 基線配置；重跑前應檢查舊的絕對路徑 `resume_stateH`。 |
| `config/Dehaze_NH.json` | 原始 NH 訓練基線 | `./data/NH-HAZE/*` | 擴散從零訓練；`netH` 使用 `NH_net_g_80000.pth` | `lr=1e-4`、`n_iter=2000000`、DDPM 2000 步 | 基線配置；重跑前應檢查舊的絕對路徑 `resume_stateH`。 |
| `config/Dehaze_ColdFog_finetune.json` | ColdFog 僅微調擴散 | ColdFog 訓練 / 驗證 | `Diffusion_trained_pth/DENSE_I130000_E2600`；凍結 DENSE `netH` | `batch_size=3`、`lr=5e-5`、`n_iter=100000`、DDPM 2000 | 初始配置；早期實驗中斷，後續 resume 實驗已完成。 |
| `config/Dehaze_ColdFog_finetune_resume.json` | 僅微調擴散實驗的 resume | ColdFog 訓練 / 驗證 | 從 `experiments/Dehaze_ColdFog_finetune_only_diffusion_260417_152053/checkpoint/I85000_E304` 恢復 | 與僅微調擴散相同 | Resume 配置，非獨立方法。 |
| `config/Dehaze_ColdFog_finetune_netH.json` | ColdFog 微調擴散 + `netH` | ColdFog 訓練 / 驗證 | DENSE 擴散 + DENSE `netH` | `finetune_netH=true`、`lr=5e-5`、`lr_netH=1e-5`、`n_iter=100000` | 已完成。 |
| `config/Dehaze_ColdFog_finetune_netH_physical.json` | ColdFog 微調擴散 + `netH` + 物理損失 | 含元數據的 ColdFog 訓練 / 驗證 | 最佳 `netH` 實驗 `I90000_E322`；載入 `resume_stateH_finetune` | `lambda_t=0.01`、`lambda_asm=0.05`、驗證採樣器 `ddim`、`sample_steps=20` | 驗證階段已完成；最終測試推理缺失。 |

### 測試 / 推理配置

| 配置 | 角色 | 檢查點 | 採樣器 | 狀態說明 |
| --- | --- | --- | --- | --- |
| `config/test_DENSE.json` | DENSE 預訓練在 DENSE 測試集上測試 | `DENSE_I130000_E2600` | DDPM 2000 | 數據集長度 5；非 ColdFog。 |
| `config/test_NH.json` | NH 預訓練在 NH 測試集上測試 | `NH_I230000_E4600` | DDPM 2000 | 數據集長度 5；非 ColdFog。 |
| `config/test_DENSE_diy.json` | DENSE 預訓練在 ColdFog 測試集上零樣本 | `DENSE_I130000_E2600` | DDPM 2000 | 已在 `experiments/test/pretrain_model_domain_gap` 完成。 |
| `config/test_NH_diy.json` | NH 預訓練在 ColdFog 測試集上零樣本 | `NH_I230000_E4600` | DDPM 2000 | 配置存在，但未找到已驗證的 ColdFog 測試運行。 |
| `config/test_ColdFog_finetune.json` | 僅微調擴散的 ColdFog 測試 | 舊 `I15000_E54` | DDPM 2000 | 現有測試使用早期中斷實驗，非已完成僅微調擴散實驗。 |
| `config/test_ColdFog_finetune_netH.json` | 微調擴散 + `netH` 的 ColdFog 測試 | `I90000_E322` + `I90000_E322_netH.pth` | DDPM 2000 | 已完成。 |
| `config/test_ColdFog_finetune_ddim.json` | `netH` 模型 + DDIM | `I90000_E322` + `netH` | DDIM 100，eta 0 | 已完成；seed42 運行可復現。 |
| `config/test_ColdFog_finetune_dpm_solver_pp.json` | `netH` 模型 + DPM-Solver++ | `I90000_E322` + `netH` | DPM-Solver++ 200 | 已完成多種步數。 |
| `config/test_ColdFog_finetune_netH_physical_ddim20.json` | 物理損失模型測試候選 | 目前指向不存在的無 `_v1` 路徑及 `I25000_E90` | 檔名為 DDIM20，現有內容為 `sample_steps=100` | 髒 / 未提交配置；修正並重跑前勿作最終證據。 |

## 已完成訓練實驗

| 實驗 | 訓練狀態 | 最佳驗證 | 最終驗證 | 檢查點 | 解讀 |
| --- | --- | --- | --- | --- | --- |
| `experiments/Dehaze_ColdFog_finetune_260405_120026` | 中斷 / 半完成 | PSNR `14.673`（`I15000_E54`） | 無最終；日誌在 `iter 20000` 訓練行後停止 | 僅 `I5000`、`I10000`、`I15000` | 歷史早期僅微調擴散實驗。勿作最終方法。 |
| `experiments/Dehaze_ColdFog_finetune_only_diffusion_260417_152053` | 從 `I85000_E304` resume 後完成 | PSNR `17.879`（`I85000_E304`） | PSNR `16.930`（`I100000_E358`） | 完整 5k 間隔檢查點至 `I100000_E358` | 已完成僅微調擴散消融，但未找到對應最終測試運行。 |
| `experiments/Dehaze_ColdFog_finetune_netH_260406_223953` | 已完成 | PSNR `19.580`（`I90000_E322`） | PSNR `19.059`（`I100000_E358`） | `gen`、`netH`、`opt` 檢查點至 `I100000_E358` | 強已完成消融；當前最佳已驗證測試結果使用 `I90000_E322`。 |
| `experiments/Dehaze_ColdFog_finetune_netH_physical_v1_260508_123402` | 已完成 | PSNR `20.322`（`I75000_E268`） | PSNR `20.261`（`I100000_E358`） | `gen`、`netH`、`opt` 檢查點至少至 `I100000_E358` | 最強驗證結果，可能為提出方法 / PPDM 風格方法；需最終測試推理。 |

重要路徑說明：磁碟上物理實驗目錄為 `Dehaze_ColdFog_finetune_netH_physical_v1_260508_123402`，但日誌內部記錄的路徑不含 `_v1`。實際檔案位於 `_v1` 目錄下。

## 已找到的 ColdFog 測試運行

以下為暫定表格最可靠的數字，因使用 ColdFog 測試數據且有推理日誌與輸出圖像。

| 候選行 | 檢查點 / 設定 | 測試集 PSNR | 證據路徑 | 是否用於最終表格？ |
| --- | --- | ---: | --- | --- |
| DehazeDDPM-DENSE 預訓練零樣本 | `DENSE_I130000_E2600` | `14.319` | `experiments/test/pretrain_model_domain_gap/Dehaze_test_DENSE_diy_260418_100213/logs/train.log` | 是，作為域間差距基線。 |
| ColdFog 僅微調擴散，早期不完整檢查點 | `I15000_E54`，DDPM | `15.122` | `experiments/test/sampler_ddpm/Dehaze_ColdFog_finetune_test_only_diffusion_260417_180356/logs/train.log` | 僅作歷史說明；非最終消融。 |
| ColdFog 微調擴散 + `netH` | `I90000_E322`，DDPM | `18.275` | `experiments/test/sampler_ddpm/Dehaze_ColdFog_finetune_test_netH_260418_004817/logs/train.log` | 是。 |
| ColdFog 微調擴散 + `netH`，DDIM seed42 | `I90000_E322`，DDIM 100 | `18.730` | `experiments/test/sampler_ddim/with_seed/Dehaze_ColdFog_finetune_test_ddim_netH_seed42_260505_113113/logs/train.log` | 是，但作採樣器 / 效率結果，非獨立方法。 |
| ColdFog 微調擴散 + `netH`，DDIM seed42 雙 GPU | `I90000_E322`，DDIM 100 | `18.730` | `experiments/test/sampler_ddim/with_seed/Dehaze_ColdFog_finetune_test_ddim_netH_seed42_2gpu_260505_130757/logs/train.log` | 確認確定性分片推理。 |
| ColdFog 微調擴散 + `netH`，DPM-Solver++ 50 | `I90000_E322`，50 步 | `16.095` | `experiments/test/sampler_dpm_solver_pp/...sample50.../logs/train.log` | 僅採樣器消融。 |
| ColdFog 微調擴散 + `netH`，DPM-Solver++ 100 | `I90000_E322`，100 步 | `17.630` | `experiments/test/sampler_dpm_solver_pp/...sample100.../logs/train.log` | 僅採樣器消融。 |
| ColdFog 微調擴散 + `netH`，DPM-Solver++ 150 | `I90000_E322`，150 步 | `17.407` | `experiments/test/sampler_dpm_solver_pp/...sample150.../logs/train.log` | 僅採樣器消融。 |
| ColdFog 微調擴散 + `netH`，DPM-Solver++ 200 | `I90000_E322`，200 步 | `17.680` | `experiments/test/sampler_dpm_solver_pp/...sample200.../logs/train.log` | 僅採樣器消融。 |

不建議將無 seed 的 DDIM 結果用於最終表格：一次運行報告 `19.262`，無 seed 雙 GPU 運行報告 `18.438`。seed42 的 DDIM 結果更站得住腳，因單 GPU 與雙 GPU 一致。

## 表 5.1 候選內容建議

| 方法 | 配置狀態 | 在第 5 章中的用途 |
| --- | --- | --- |
| 有霧輸入 | 始終可用；直接由有霧輸入與 GT 計算 PSNR/SSIM | 無處理基線，用於判斷去霧是否改善重建。 |
| DCP | 本地未實作；僅外部基線 | 傳統物理先驗基線，僅在外部運行完成後納入。 |
| DehazeNet / AOD-Net | 本地未實作；僅在外部運行完成後擇一納入 | 基於 CNN 的學習基線。 |
| DehazeDDPM-DENSE 預訓練 | 已配置並在 ColdFog 上測試（`PSNR=14.319`） | 同族擴散基線，用於域間差距驗證。 |
| DehazeDDPM-NH 預訓練 | 配置存在，未找到已驗證 ColdFog 測試運行 | 可選同族域間差距基線；重跑或移除。 |
| ColdFog 僅微調擴散 | 訓練已完成，但缺少已完成檢查點的最終測試 | 僅微調擴散階段的消融。 |
| ColdFog 微調擴散 + `netH` | 已完成並測試（`I90000_E322`） | 消融，展示適配第一階段條件網路的收益。 |
| ColdFog 微調擴散 + `netH` + 物理損失 | 訓練已完成且驗證最強；最終測試缺失 | 提出 / PPDM 風格變體，測試冷庫適配與物理一致性。 |
| DDIM / DPM-Solver++ | 針對 `netH` 模型測試的採樣器變體 | 效率與少步推理分析，非獨立去霧基線。 |

## 現階段可撰寫內容

論文可安全表述：

- 本地實驗框架基於 DehazeDDPM 的兩階段設計：`netH` 先預測去霧 / 結構條件及透射率相關資訊，再由 SR3/DDPM 條件擴散模型恢復最終清晰圖像。
- 所有 ColdFog 微調配置使用線性 beta 調度：`n_timestep=2000`、`linear_start=1e-6`、`linear_end=1e-2`。
- ColdFog 微調將圖像 resize/crop 至 `448 x 576`，batch size `3`，擴散學習率 `5e-5`，驗證 / 檢查點頻率 `5000`。
- 聯合 `netH` 微調使用 `finetune_netH=true` 與 `lr_netH=1e-5`。
- 物理損失微調增加兩項輔助損失：
  - `loss_t = L1(out_T, exp(-beta * depth))`
  - `loss_asm = L1(out_I, hazy_input)`
  - 總訓練損失：`l_pix + 0.01 * loss_t + 0.05 * loss_asm`
- 物理版本中，元數據在訓練 / 驗證階段提供 `depth` 與 `beta`。當前 ColdFog 測試配置不含元數據，故除非測試集補充元數據，否則不應聲稱測試集上使用了物理損失。

## 現階段不應聲稱的內容

在補充更多證據前，避免以下表述：

- 勿聲稱 DCP、DehazeNet、AOD-Net 或 PPDM 已在本倉庫復現。
- 勿將 DCP/DehazeNet/AOD-Net/PPDM 納入最終定量表格，除非在相同評估協議下產出 ColdFog 測試輸出。
- 勿聲稱物理損失方法已有最終 ColdFog 測試集 PSNR/SSIM；目前僅找到驗證 PSNR。
- 勿以當前狀態的 `test_ColdFog_finetune_netH_physical_ddim20.json` 作最終證據，因其為髒 / 未提交配置，指向磁碟上不存在的無 `_v1` 物理實驗路徑，且檔名為 `ddim20` 但現設 `sample_steps=100`。
- 勿將 DDIM 或 DPM-Solver++ 視為獨立訓練的去霧方法；它們是推理採樣器。

## 優先待辦實驗

### 優先級 0：最終主要定量表格

撰寫最終結論前最高價值的實驗。

| 優先級 | 待辦 | 重要性 | 建議設定 |
| --- | --- | --- | --- |
| P0-1 | 為物理損失模型運行最終 ColdFog 測試推理 | 可能為提出方法，但測試集指標缺失 | 最佳驗證檢查點：`experiments/Dehaze_ColdFog_finetune_netH_physical_v1_260508_123402/checkpoint/I75000_E268`；可選最終檢查點 `I100000_E358`。 |
| P0-2 | 為已完成僅微調擴散模型運行最終 ColdFog 測試推理 | 現有測試使用舊不完整 `I15000_E54`，對 `netH` 與物理損失實驗不公平 | 最佳驗證檢查點：`experiments/Dehaze_ColdFog_finetune_only_diffusion_260417_152053/checkpoint/I85000_E304`；可選 `I100000_E358`。 |
| P0-3 | 在同一 80 張 ColdFog 測試集上計算有霧輸入 PSNR/SSIM | 無處理基線行所需 | 直接比較 `/data/dehazeddpm_test/hazy_test` 與 `/data/dehazeddpm_test/gt_test`。 |
| P0-4 | 為所有保留方法計算 SSIM | 現有推理日誌主要提供 PSNR；論文表格通常需 PSNR/SSIM | 使用已保存的 `*_out.png` 與 GT 圖像，或擴展小評估腳本（`core.metrics.calculate_ssim`）。 |
| P0-5 | 將最終表格統一至同一評估協議 | 避免混用驗證 PSNR、舊中斷實驗與測試 PSNR | 相同 ColdFog 測試劃分、相同圖像尺寸、相同指標程式碼；採樣器隨機時固定 seed。 |

P0 完成後建議的最終主要表格行：

1. 有霧輸入。
2. DehazeDDPM-DENSE 預訓練零樣本。
3. DehazeDDPM-NH 預訓練零樣本（僅在重跑完成後納入；否則省略）。
4. ColdFog 僅微調擴散，已完成最佳驗證檢查點。
5. ColdFog 微調擴散 + `netH`，`I90000_E322`。
6. ColdFog 微調擴散 + `netH` + 物理損失，最佳驗證或最終檢查點。

### 優先級 1：外部基線

可提升覆蓋面，但若章節框架為 DehazeDDPM 適配則非必需。

| 優先級 | 待辦 | 建議 |
| --- | --- | --- |
| P1-1 | DCP 在 ColdFog 測試集上 | 僅在有時間時添加。作為傳統先驗基線有用，但需外部實作或新腳本。 |
| P1-2 | 在 DehazeNet 與 AOD-Net 中擇一 | 除非已有檢查點與推理程式碼，否則勿同時納入兩者。AOD-Net 通常更易作為緊湊 CNN 基線呈現。 |
| P1-3 | PPDM | 僅在外部程式碼 / 檢查點可可靠運行時納入。當前倉庫無 PPDM 實作。 |

若時間緊張，從最終定量結論中移除上述方法，並說明為「因未完成已驗證運行而未納入的候選基線」。

### 優先級 2：效率 / 採樣器分析

支撐 5.7 小節。

| 優先級 | 待辦 | 原因 |
| --- | --- | --- |
| P2-1 | 測量 DDPM 2000、DDIM 100、DPM-Solver++ 50/100/150/200 的秒 / 圖 | 現有日誌有起止時間戳，但受控計時表更清晰。 |
| P2-2 | 僅以固定 seed 重跑 DDIM | 無 seed 的 DDIM 結果不穩定；固定 seed 更站得住腳。 |
| P2-3 | 可選：在同一檢查點上測試 DDIM 20/50/100/200 | 可使「DDIM 採樣」小節比僅 DDIM 100 更完整。 |

### 優先級 3：物理一致性分析

支撐 5.6 小節，而非主要去霧表格。

| 優先級 | 待辦 | 原因 |
| --- | --- | --- |
| P3-1 | 在含元數據 / depth / beta 的劃分上評估物理模型 | 測試配置缺元數據，故當前 ColdFog 測試集上的物理損失無意義。 |
| P3-2 | 保存代表性 `out_T`、`out_A`、`out_I`、`output`、`stage1_output` 面板 | 這些可視化說明物理先驗如何改變 `netH`，而非僅 PSNR。 |
| P3-3 | 比較訓練驗證日誌中的 `loss_t`、`loss_asm`、`loss_physical_total` | 物理損失實驗已每 5000 步記錄這些值。 |

## 建議 LaTeX 級文字段落

### 對比方法段落

對比方法按其在實驗設計中的角色組織。首先，保留有霧輸入作為無處理基線。其次，在公開去霧數據集上訓練的 DehazeDDPM 預訓練檢查點直接在 ColdFog 圖像上評估，以量化跨域退化。第三，比較若干 ColdFog 適配的 DehazeDDPM 變體：僅微調擴散模型、聯合微調擴散模型與第一階段 `netH`、以及基於透射率與大氣散射約束加入物理一致性損失。DDPM、DDIM、DPM-Solver++ 等推理採樣器單獨作為效率配置分析，而非獨立去霧模型。

### 物理損失段落

物理變體在聯合 `netH` 微調配置基礎上引入兩項輔助約束。第一項以 `t(x)=exp(-beta d(x))` 監督預測透射圖，其中 depth 與散射係數從 ColdFog 元數據讀取。第二項透過大氣散射模型 `I = T J + (1 - T) A` 重建有霧輸入，其中 `J`、`T`、`A` 由 `netH` 預測。最終訓練目標為擴散損失加上 `0.01 loss_t + 0.05 loss_asm`。

### 表格標題草稿

候選對比方法及其在實驗設計中的角色。最終定量表格僅應納入在相同評估協議下已完成且已驗證的 ColdFog 測試集結果；無本地實作或未完成運行的方法應報告為計劃中或從最終定量結論中排除。

## 最終納入建議

最終論文定量表格僅納入已完成且已驗證的 ColdFog 測試集運行。依現有證據，表格尚未就緒以作最終結論，因提出方法（物理損失）與已完成的僅微調擴散消融仍缺測試集推理。補全後，本章可作出更強、更清晰的結論：

- 預訓練 DehazeDDPM 在 ColdFog 上呈現明顯域間差距；
- ColdFog 微調改善重建；
- 聯合 `netH` 適配優於僅微調擴散；
- 物理一致性帶來最強驗證表現，應作為提出最終方法進行測試。
