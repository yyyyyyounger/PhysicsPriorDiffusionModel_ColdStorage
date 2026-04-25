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