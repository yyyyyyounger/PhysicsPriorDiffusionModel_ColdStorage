#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python sr.py --config config/Dehaze_ColdFog_finetune_netH_physical.json "$@"
