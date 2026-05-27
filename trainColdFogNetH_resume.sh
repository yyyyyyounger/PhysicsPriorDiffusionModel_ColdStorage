#!/usr/bin/env bash
# 以 best_val_checkpoints_0406/I90000_E322 權重初始化，寫入新 experiments 目錄
# （l_pix netG+netH，validation 使用 DDIM 20 steps；見 config 內 resume_state）。
set -euo pipefail
cd "$(dirname "$0")"
optpath='./config/Dehaze_ColdFog_finetune_netH_resume.json'
python sr.py --config "$optpath" "$@"
