#!/usr/bin/env bash
# 接續 experiments/Dehaze_ColdFog_finetune_only_diffusion_260417_152053 的訓練（見 config 內 resume_state）。
set -euo pipefail
cd "$(dirname "$0")"
optpath='./config/Dehaze_ColdFog_finetune_resume.json'
python sr.py --config "$optpath"
