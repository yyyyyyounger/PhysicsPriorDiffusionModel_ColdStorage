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

**程式入口對應：** `sr.py` 在驗證階段會載入 `beta_schedule.val`；`infer.py` 亦使用 **`beta_schedule.val`** 做推理。專案中已有對照用設定：`config/test_ColdFog_finetune_ddim.json`、`config/test_ColdFog_finetune_dpm_solver_pp.json`，以及 `testColdFogFinetune_ddim.sh`、`testColdFogFinetune_dpm_solver_pp.sh`；完整 DDPM baseline 仍為 `config/test_ColdFog_finetune.json` 與 `testColdFogFinetune.sh`。

跑我的數據集
```
conda activate lzy_dehazeddpm

CUDA_VISIBLE_DEVICES=3 python infer.py --config ./config/test_DENSE_diy.json

CUDA_VISIBLE_DEVICES=2 python infer.py --config ./config/test_NH_diy.json
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