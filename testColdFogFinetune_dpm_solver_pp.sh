#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python infer.py --config config/test_ColdFog_finetune_dpm_solver_pp.json "$@"
