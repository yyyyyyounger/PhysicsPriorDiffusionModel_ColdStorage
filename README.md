# DehazeDDPM
This is the codebase for [High-quality Image Dehazing with Diffusion Model](https://arxiv.org/abs/2308.11949).

# Download pre-trained models

This method employs a two-stage pipline. The pre-trained model of the first stage is within the 'pretrained_PreNet_pth' file.
Here are the download links for the second-stage model checkpoint: [Diffusion_trained_pth](https://drive.google.com/drive/folders/1I7sH6vb9oWOZeIVu6-xh9Xm5lnwdzHa7?usp=drive_link)


使用conda環境：
```
conda activate lzy_dehazeddpm

pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

pip install tensorboardX wandb numpy opencv-python Pillow tqdm lmdb matplotlib
```

指定GPU運行：
```
CUDA_VISIBLE_DEVICES=3 bash testDENSE.sh
```

## 訓練：從頭與接續

訓練入口為 `python sr.py --config <json>`（例如 `bash trainColdFog.sh`）。以下路徑皆相對於專案根目錄、且需在根目錄執行。

### 從頭訓練（新實驗目錄）

- **不要**在設定檔的 `path` 裡加入 `reuse_experiments_root`。程式會自動建立 `experiments/<name>_<時間戳>/`，並將 `logs`、`checkpoint`、`tb_logger` 等寫入該目錄。
- **`resume_state`**：填二階段擴散預訓練的**前綴**（不含 `_gen.pth`），例如 `./Diffusion_trained_pth/DENSE_I130000_E2600`。程式會載入 `..._gen.pth`。
- 若該前綴路徑**沒有**對應的 `..._opt.pth`（僅預訓練權重），則只載入網路權重，**iteration 從 0 開始**，屬預期行為。
- **`resume_stateH`**：維持第一階段 PreNet 權重路徑（與原專案相同）。

### 接續訓練（同一實驗目錄 + log 接續）

中斷後若要從**最近一次存檔**繼續跑到 `train.n_iter`（例如 100000），並讓 **`train.log` / `val.log` 接在舊檔後面**（不覆寫）：

1. 在 `path` 中設定 **`reuse_experiments_root`** 為既有實驗資料夾的**完整相對路徑**，例如：  
   `experiments/Dehaze_ColdFog_finetune_only_diffusion_260417_152053`
2. 將 **`resume_state`** 改為該實驗下 `checkpoint` 裡**一組存檔的前綴**（同樣不含 `_gen.pth`），例如：  
   `experiments/Dehaze_ColdFog_finetune_only_diffusion_260417_152053/checkpoint/I85000_E304`
3. 必須同時存在 **`I85000_E304_gen.pth`** 與 **`I85000_E304_opt.pth`**，才會還原 **optimizer** 以及 **`iter` / `epoch`**，訓練才會從該 iteration 繼續；若只有 `*_gen.pth`，行為等同只載權重並從 iter 0 重跑。
4. **`train.n_iter`** 仍為「總 iteration 上限」；接續時會從 checkpoint 記錄的 iter 繼續遞增，直到達到 `n_iter`。

**注意：** 存檔頻率由 `train.save_checkpoint_freq` 決定。若中斷發生在兩次存檔之間，磁碟上只有**上一個** checkpoint，接續後會從該點重跑中間未落盤的 iteration。TensorBoard 沿用同一 `tb_logger` 目錄時可能產生多個 event 檔，一般仍可一併檢視。

### 驗證／推理採樣器設定（`beta_schedule.val`）

僅當 **`model.which_model_G` 為 `"sr3"`** 時，`GaussianDiffusion.super_resolution()` 會依設定切換採樣器；`ddpm` 路徑仍為完整反向鏈，與既有 checkpoint 相容。

設定寫在 JSON 的 **`model.beta_schedule.val`**（與 `schedule`、`n_timestep`、`linear_start`、`linear_end` 同層）。**訓練**仍只使用 **`beta_schedule.train`**，不需也不應在 `train` 區塊加 `sampler`。

| 欄位 | 說明 |
|------|------|
| **`sampler`** | 選採樣器：省略或省略整個鍵行為時，在載入 **僅含 train 的 schedule**（例如 `sr.py` 驗證後切回 train）會重設為 **`ddpm`**。可選字串（不分大小寫）：**`ddpm`**（完整 \(T\) 步）、**`ddim`**、**`dpm_solver_pp`**（亦可寫 **`dpm_solver++`**，程式會正規化成底線形式）。 |
| **`sample_steps`** | **僅對 `ddim` / `dpm_solver_pp` 有意義**。從完整時間表（長度為 **`n_timestep`**）做跳步採樣的目標步數；**不要**為了加速而把 **`val.n_timestep`** 改成遠小於訓練時的步數來冒充少步推理（會破壞與訓練一致的 \(\bar\alpha\) 離散化）。若省略：**`ddim` 預設 100**，**`dpm_solver_pp` 預設 50**。**`ddpm`** 會忽略此欄，固定跑 **`n_timestep`** 步。 |
| **`ddim_eta`** | **僅 `ddim` 使用**。\( \eta = 0 \) 為確定性 DDIM；\( \eta > 0 \) 會注入隨機性（類 DDPM）。預設可設 **`0.0`**。 |

**範例片段（DDIM，100 步）：**

```json
"val": {
  "schedule": "linear",
  "n_timestep": 2000,
  "linear_start": 1e-6,
  "linear_end": 1e-2,
  "sampler": "ddim",
  "sample_steps": 100,
  "ddim_eta": 0.0
}
```

**範例片段（DPM-Solver++，50 步）：**

```json
"val": {
  "schedule": "linear",
  "n_timestep": 2000,
  "linear_start": 1e-6,
  "linear_end": 1e-2,
  "sampler": "dpm_solver_pp",
  "sample_steps": 50
}
```

**採樣器調參建議：**

- **DDIM / DPM-Solver++ 不需要重新訓練。** 它們只改變驗證／推理時的反向採樣方式，不引入新的可訓練參數；後續訓練仍保持 `beta_schedule.train` 原設定即可。
- **不要為了加速改小 `val.n_timestep`。** `n_timestep`、`linear_start`、`linear_end` 應與訓練 schedule 保持一致，例如本專案 finetune 設定為 `2000 / 1e-6 / 1e-2`。少步推理請只調 `sample_steps`。
- **DPM-Solver++ 優先調 `sample_steps`。** 目前程式中 `order=2`、`skip_type='time_uniform'`、`clip_denoised=True` 是固定的；若 50 步結果有噪點，建議依序測 `75`、`100`、`150`、`200`，通常步數增加會提升穩定性但推理更慢。
- **DDIM 建議先用確定性設定。** `ddim_eta: 0.0` 通常更穩；若影像已有噪點，不建議先增大 `ddim_eta`，因為 `eta > 0` 會額外注入隨機性。可對照測 `sample_steps: 100` 與 `200`。
- **實驗比較建議。** 用完整 DDPM（`ddpm`，2000 步）作為品質 baseline，再比較 `DPM-Solver++ 50/100/150`、`DDIM 100/200` 的 PSNR、SSIM 與視覺效果；若高步數仍有明顯噪點，問題更可能來自模型權重、冷庫資料分佈或第一階段 `netH` 條件圖品質，而不是採樣器本身。

**程式入口對應：** `sr.py` 在驗證階段會載入 `beta_schedule.val`；`infer.py` 亦使用 **`beta_schedule.val`** 做推理。專案中已有對照用設定：`config/test_ColdFog_finetune_ddim.json`、`config/test_ColdFog_finetune_dpm_solver_pp.json`，以及 `testColdFogFinetune_ddim.sh`、`testColdFogFinetune_dpm_solver_pp.sh`；完整 DDPM baseline 仍為 `config/test_ColdFog_finetune.json` 與 `testColdFogFinetune.sh`。

跑我的數據集
```
conda activate lzy_dehazeddpm

CUDA_VISIBLE_DEVICES=3 python infer.py --config ./config/test_DENSE_diy.json

CUDA_VISIBLE_DEVICES=2 python infer.py --config ./config/test_NH_diy.json
```

### `infer.py` 多 GPU 推理（依樣本切分）

當設定檔中 **`gpu_ids` 超過一張**（例如 `[0, 1]`），或使用 **`python infer.py ... -gpu 0,1`** 時，程式會以 **`torch.multiprocessing.spawn`** 為每張可見 GPU 啟動一個進程，並將驗證集按 **樣本索引交錯切分**（rank `r` 處理索引 `r, r+W, r+2W, ...`）。每個進程各自載入完整模型並綁定 **`cuda:{local_rank}`**，且會將 **`distributed` 設為 False**，避免子進程再包一層 `DataParallel`。

- **輸出檔名**：使用資料集中的 **`Index + 1`** 作為檔名中的序號（與單卡時「第 k 張圖對應 `{step}_{k+1}_*.png`」一致），多進程並行寫入同一 `results` 目錄時不會互相覆蓋。
- **整體 PSNR**：主進程依各 worker 回傳的樣本加權平均彙總。
- **日誌**：各 GPU 的細節寫入 **`logs/infer_rank{0,1,...}.log`**；主進程終端仍會印出總平均 PSNR。
- **W&B**：多 GPU 模式下若開啟 **`log_infer`**，為避免多進程重複上傳，**會跳過**逐張 `log_eval_data`／表格；若需要完整 W&B 推理紀錄請改用 **單卡**。

範例：

```bash
bash testColdFogFinetune_ddim.sh -gpu 0,1
# 或在 JSON 中設定 "gpu_ids": [0, 1] 後直接：
python infer.py --config config/test_ColdFog_finetune_ddim.json
```


後臺運行
```
# 1) 開新 session
tmux new -s dehaze

# 2) 在 tmux 內執行
conda activate lzy_dehazeddpm
cd /mnt/newdisk/Documents/linzhanyang/DehazeDDPM
python infer.py --config ./config/test_DENSE_diy.json
# 或 NH:
python infer.py --config ./config/test_NH_diy.json


離開但不中斷：Ctrl + a，不鬆手再按 d
之後重連 SSH 回來看：tmux attach -t dehaze
看目前 session：tmux ls
```

20260226 Baseline結果：
NH 80 張Test結果：# Validation # PSNR: 1.4448e+01
Dense 結果：# Validation # PSNR: 1.4432e+01

到 data/results_analysis.ipynb 跑PSNR、SSIM的分析結果