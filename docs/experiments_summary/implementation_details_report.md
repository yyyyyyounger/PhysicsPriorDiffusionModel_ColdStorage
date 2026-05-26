# 5.1.4 Implementation Details and Checkpoint Selection

生成日期：2026-05-26。供論文 §5.1.4 實作細節表格直接引用。

本報告依據 `config/Dehaze_ColdFog_finetune*.json`、`config/test_ColdFog_finetune*.json`、`data/finetune/metadata.csv` 及 `data/dehazeddpm_test/dehazeddpm_manifest_seed42.json` 整理。Conda 環境：`dehazeddpm`。

---

## 論文可直接引用的表格（英文）

| Item | Setting |
| :--- | :--- |
| Input resolution | 448 × 576 |
| Train / validation / test split | 280 / 40 / 80 clear-image IDs; expanded to 840 / 120 / 240 hazy–clear pairs |
| Main evaluation subset | **80 one-to-one hazy–clear test pairs** (one fog level per test ID; see note below) |
| Main sampler for diffusion variants | DDIM-100, η = 0, seed = 42 |
| Optimizer | Adam |
| netG learning rate | 5 × 10⁻⁵ |
| netH learning rate | 1 × 10⁻⁵ |
| Batch size | 3 |
| Training iterations | 100,000 |
| Validation / checkpoint interval | Every 5,000 iterations |
| Diffusion schedule | Linear β, 2,000 training timesteps |
| Random seed | 42 |
| Checkpoint selection | Best validation PSNR; no test-set selection |
| Hardware | 2 × NVIDIA RTX 3090 |

---

## 各欄位依據與撰寫說明

### Input resolution

- 所有 ColdFog 微調配置統一為 `img_sizeH=448`、`img_sizeW=576`。
- 配置來源：`config/Dehaze_ColdFog_finetune_netH.json` L20–21、L32–33。

### Train / validation / test split

| 劃分 | 清晰圖 ID 數 | 有霧–清晰配對數 | 說明 |
| --- | ---: | ---: | --- |
| Train | 280 | 840 | 每 ID 對應 light / medium / heavy 三檔霧濃度 |
| Val | 40 | 120 | 同上 |
| Test | 80 | 240（完整） | 80 個測試 ID × 3 檔霧濃度 |

- 統計來源：`data/finetune/metadata.csv`（train 840 行、val 120 行；test 不在該 CSV 中，但 ID 數與資料集設計一致）。

### Main evaluation subset（原 TODO 項，已填寫）

**建議寫法：80 one-to-one hazy–clear test pairs。**

理由：

1. 目錄 `data/dehazeddpm_test/hazy_test` 與 `gt_test` 各含 **80** 張圖像（2026-05-26 實測）。
2. 所有已完成的 ColdFog 測試推理配置均指向上述目錄，例如 `config/test_ColdFog_finetune_netH.json`、`config/test_ColdFog_finetune_ddim.json`。
3. 測試子集由 `dehazeddpm_manifest_seed42.json` 定義：對 80 個 test ID **各選一個霧濃度檔**（`chosen_level`），形成 80 組一對一配對；`selection_seed=43` 用於檔位選擇，`seed=42` 用於推理隨機性控制。
4. 完整 240 對（80 ID × 3 檔）尚未作為統一主評估協議使用；若論文需報告 240 對結果，應另起一行說明評估協議，並對所有方法重跑。

**論文可選補充句（英文）：**

> The main quantitative comparison uses 80 one-to-one hazy–clear pairs, where each of the 80 held-out clear-image IDs is paired with a single fog level selected from the three synthesis levels (light, medium, heavy). The full 240-pair test split is reserved for fog-level analysis if needed.

### Main sampler for diffusion variants

| 階段 | 採樣器 | 步數 | η | seed |
| --- | --- | ---: | ---: | ---: |
| 主測試 / 定量對比 | DDIM | 100 | 0 | 42 |
| 訓練期驗證（物理損失實驗） | DDIM | 20 | 0 | 42 |
| 部分消融 / 域間差距基線 | DDPM | 2000 | — | 42 |

- 主表格應寫 **DDIM-100, η=0, seed=42**，與 `config/test_ColdFog_finetune_ddim.json` 及 seed42 可復現結果一致（單卡 / 雙卡 PSNR 均為 18.730）。
- 不建議將無 seed 的 DDIM 結果寫入主表（見 `compared_methods_experiment_report_zh.md`）。

### Optimizer & learning rates

- Optimizer：Adam（`train.optimizer.type = "adam"`）。
- netG（擴散 U-Net）：`lr = 5e-5`。
- netH（MPRfusion，僅 `finetune_netH=true` 時）：`lr_netH = 1e-5`。
- 配置來源：`config/Dehaze_ColdFog_finetune_netH.json` L86–89。

### Batch size / iterations / validation

| 項目 | 值 |
| --- | --- |
| Batch size | 3 |
| Training iterations | 100,000 |
| Validation frequency | 5,000 iterations |
| Checkpoint save frequency | 5,000 iterations |
| Approx. epochs | ≈ 358（100,000 ÷ ⌈840/3⌉） |

- 配置來源：各 `Dehaze_ColdFog_finetune*.json` 的 `train` 區塊。

### Diffusion schedule

- Linear β schedule；`n_timestep = 2000`；`linear_start = 1e-6`；`linear_end = 1e-2`。
- 與原始 DehazeDDPM / 第 4 章設定一致，論文中可寫「follow Ch.4 / DehazeDDPM」而不重述公式。

### Random seed

- 全局 `manual_seed = 42`（所有 ColdFog 微調與主測試配置）。
- 推理階段另按樣本 index 設置 sample seed，保證多 GPU 分片結果一致（見 `Contribution.md` §6）。

### Checkpoint selection

- **規則：依驗證集 PSNR 選最佳 checkpoint，不用測試集選模。**
- 各消融最佳驗證 checkpoint（供表格 / 复現引用）：

| 方法 | 最佳驗證 PSNR | Checkpoint |
| --- | ---: | --- |
| 僅微調擴散 | 17.879 | `I85000_E304` |
| 微調擴散 + netH | 19.580 | `I90000_E322` |
| 微調擴散 + netH + 物理損失 | 20.322 | `I75000_E268` |

- 若某方法在測試集上的最佳 checkpoint 與驗證最佳不同，應在論文中說明並統一為「validation-best」規則以保公平。

### Hardware

- 訓練配置 `gpu_ids: [2, 3]` 或 `[0, 1]`，均為 **2 × RTX 3090**。
- 論文表格寫 **RTX 3090 × 2** 即可。

---

## 與原始 DehazeDDPM 的差異（可選寫入正文）

| 項目 | 原始 DehazeDDPM（DENSE/NH） | ColdFog 微調 |
| --- | --- | --- |
| 輸入解析度 | 512 × 672（DENSE）等 | 448 × 576 |
| netG 學習率 | 1e-4 | 5e-5 |
| 訓練迭代 | 2,000,000 | 100,000 |
| netH 微調 | 否（凍結 PreNet） | 可選（`finetune_netH`） |
| 物理損失 | 無 | 可選（λ_t=0.01, λ_asm=0.05） |

---

## 復現命令（conda: dehazeddpm）

```bash
conda activate dehazeddpm
cd /mnt/newdisk/Documents/linzhanyang/DehazeDDPM

# 訓練（微調擴散 + netH）
CUDA_VISIBLE_DEVICES=2,3 python sr.py --config config/Dehaze_ColdFog_finetune_netH.json

# 測試（DDIM-100, seed=42）
CUDA_VISIBLE_DEVICES=2 python sr.py --config config/test_ColdFog_finetune_ddim.json
```

---

## 待確認 / 可選補充

| 項目 | 狀態 | 建議 |
| --- | --- | --- |
| 物理損失模型 ColdFog 測試 PSNR | 未完成 | P0-1：用 `I75000_E268` 跑 `test_ColdFog_finetune_netH_physical_ddim20.json`（需先修正配置路徑） |
| 僅微調擴散最終測試 | 未完成 | P0-2：用 `I85000_E304` 重跑 ColdFog 測試 |
| SSIM 指標 | 日誌多為 PSNR | P0-4：對已保存輸出補算 SSIM |
| 240 對完整測試集 | 未作主協議 | 若需要 fog-level 分析可另表，勿與 80 對主表混用 |

---

## 修訂記錄

| 日期 | 變更 |
| --- | --- |
| 2026-05-26 | 初版；填寫 Main evaluation subset TODO 為 80 one-to-one pairs |
