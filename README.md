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