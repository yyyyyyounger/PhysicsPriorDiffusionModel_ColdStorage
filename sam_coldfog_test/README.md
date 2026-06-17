# Cold-Fog SAM 評測
SAM Github: https://github.com/facebookresearch/segment-anything

冷庫霧 test 集上的 Segment Anything（SAM）自動分割評測流程。
使用了SAM的 `sam_vit_h_4b8939.pth` checkpoint。

## 準備SAM評測用的數據集
參考 `data/` 下的兩個py腳本。

## GPU 設定（優先閱讀）

SAM 推理建議在 GPU 上執行。依腳本不同，選擇下列方式之一。

### 方式 A：雙 GPU 並行 `--gpu-ids`（推薦，加速）

將 sample 用 **seed 42** 打亂後均分到各 GPU，各卡獨立載入 SAM 並行推理，最後合併結果。

```bash
cd /mnt/newdisk/Documents/linzhanyang/segment-anything

# GPU 0 和 GPU 3 各跑一半（80 張 → 40 + 40）
python coldfog_test/run_sam_infer.py \
  --input-dir data/sam_eval/clear \
  --tag clear \
  --manifest data/sam_eval/manifest.json \
  --gpu-ids 0,3 \
  --save-overlays
```

`run_sam_eval.py` 同樣支援 `--gpu-ids`；`run_sam_infer.py` 建議用於分目錄推理。

| 參數 | 說明 |
|------|------|
| `--gpu-ids` | 逗號分隔的卡號，例如 `0,1` 或 `0,3`；與 `--gpu-id` 互斥 |
| 拆分方式 | `random.Random(42).shuffle` 後 round-robin 均分 |
| 進度輸出 | 各 worker 前綴 `[GPU N]`，例如 `[GPU 0] [12/40] ...` |
| 合併 | 全部 shard 完成後寫入 `results/run_*/`；`summary.json` 的 `config.split` 記錄各卡 sample 清單 |

80 張 test 集、2 卡時 shard 大小為 `[40, 40]`。奇數 sample 時各 shard 相差最多 1 張。

### 方式 B：單 GPU `--gpu-id`

```bash
# 使用第 0 號 GPU
python coldfog_test/run_sam_eval.py --gpu-id 0

# 使用第 3 號 GPU
python coldfog_test/run_sam_eval.py --gpu-id 3

# 強制 CPU（忽略 GPU 參數）
python coldfog_test/run_sam_eval.py --device cpu
```

| 參數 | 預設 | 說明 |
|------|------|------|
| `--gpu-id` | 無 | 單卡 CUDA 索引；實際 device 為 `cuda:N` |
| `--device` | `cuda` | `cuda` 或 `cpu` |

### 方式 C：環境變數 `CUDA_VISIBLE_DEVICES`

適用於 `run_sam_overlay.py`（該腳本尚未提供 `--gpu-id`）：

```bash
CUDA_VISIBLE_DEVICES=0 python coldfog_test/run_sam_overlay.py
```

`run_sam_infer.py` / `run_sam_eval.py` 建議直接用 `--gpu-id` / `--gpu-ids`，無需再設此變數。

### 執行前檢查

```bash
nvidia-smi          # 確認卡號與空閒顯存
conda activate <你的 segment-anything 環境>
```

SAM ViT-H 權重預設路徑：`checkpoints/sam_vit_h_4b8939.pth`（需自行下載）。

---

## 流程概覽

**推薦（分輪推理 + 比較，可復用 clear reference）：**

```
data/sam_eval/          run_sam_infer.py (clear)
     │                         │
     │                  results/infer/clear_*/
     │
     ├── heavy/  ──►  run_sam_infer.py (heavy)  ──►  results/infer/heavy_*/
     │
DehazeDDPM results/     prepare_dehazed_sam_eval.py
  {step}_{index}_out.png            │
                                    ▼
     └── dehazed/ ──► run_sam_infer.py (dehazed) ──► results/infer/dehazed_*/
                                    │
                          run_sam_compare.py
                                    │
                                    ▼
                          results/compare/
```

**一鍵評測（舊流程，仍可用）：**

```
data/splits/test          data/prepare_sam_eval.py
        │                          │
        └──────────────►  data/sam_eval/
                                    │
                          run_sam_eval.py
                                    │
                                    ▼
                          coldfog_test/results/
```

1. **準備資料**（一次性，無需 GPU）：`data/prepare_sam_eval.py`
2. **準備去霧結果**（每次新實驗，無需 GPU）：`data/prepare_dehazed_sam_eval.py`
3. **分目錄推理**（需 GPU，推薦）：`run_sam_infer.py` → 各條件各跑一輪
4. **比較結果**（無需 GPU）：`run_sam_compare.py`
5. **深度分層比較**（無需 GPU）：`run_sam_depth_compare.py`
6. **論文繪圖**（無需 GPU）：`plot/plot_depth_heatmaps.py`、`plot/plot_qualitative_depth_case.py`
7. **一鍵評測**（可選）：`run_sam_eval.py`（clear + 三霧度一次跑完）
8. **單張可視化**（可選）：`run_sam_overlay.py`

---

## 2A. 分目錄推理 `run_sam_infer.py`（推薦）

對**任意圖片目錄**跑 SAM，保存 mask、統計與可選 overlay。clear 只需推理一次，後續 heavy / dehazed 可獨立追加。

### 第一輪：clear（GT-SAM reference）

```bash
cd /mnt/newdisk/Documents/linzhanyang/segment-anything

python coldfog_test/run_sam_infer.py \
  --input-dir data/sam_eval/clear \
  --tag clear \
  --manifest data/sam_eval/manifest.json \
  --gpu-id 0 \
  --save-overlays
```

### 第二輪：heavy（或其他條件）

```bash
python coldfog_test/run_sam_infer.py \
  --input-dir data/sam_eval/heavy \
  --tag heavy \
  --manifest data/sam_eval/manifest.json \
  --gpu-id 0 \
  --save-overlays
```

### 第三輪：DDPM 去霧結果

DehazeDDPM 輸出檔名為 `{step}_{index}_out.png`（`index` 對應 `test_metadata.json`），
需先用 `prepare_dehazed_sam_eval.py` 對齊成 `{sample_id}.png`（見 **1B**），再跑 SAM：

```bash
# 0. 準備去霧 staging 目錄（只需每次新實驗做一次）
python data/prepare_dehazed_sam_eval.py \
  --dehaze-results /path/to/DehazeDDPM/experiments/.../results \
  --with-hazy-input

# 1. 去霧結果 SAM
python coldfog_test/run_sam_infer.py \
  --input-dir data/sam_eval/dehazed \
  --tag dehazed_ddim100 \
  --manifest data/sam_eval/manifest.json \
  --gpu-id 0 \
  --save-overlays

# 2. （可選）各 sample 實際輸入霧圖 SAM，作為去霧前 baseline
python coldfog_test/run_sam_infer.py \
  --input-dir data/sam_eval/hazy_input \
  --tag hazy_input \
  --manifest data/sam_eval/manifest.json \
  --gpu-id 0 \
  --save-overlays
```

去霧 vs clear / 去霧 vs 輸入霧圖：

```bash
python coldfog_test/run_sam_compare.py \
  --reference coldfog_test/results/infer/latest_clear \
  --query coldfog_test/results/infer/latest_hazy_input \
            coldfog_test/results/infer/latest_dehazed_ddim100 \
  --save-tex
```

### 輸出

```
coldfog_test/results/infer/
  latest_clear -> clear_<UTC>/
  clear_<UTC>/
    meta.json          # input_dir, tag, AMG 參數
    index.csv          # 每張圖 num_valid_masks, mean_stability
    masks/*.npz        # 完整 SAM mask（compare 時讀取）
    overlays/*.png     # --save-overlays 時
```

| 參數 | 預設 | 說明 |
|------|------|------|
| `--input-dir` | （必填） | 圖片目錄 |
| `--tag` | 目錄 basename | run 名稱前綴 |
| `--manifest` | 無 | 可選，控制 sample 順序/篩選 |
| `--save-overlays` | 關 | 輸出每張 overlay PNG |
| `--gpu-ids` | 無 | 多卡並行，seed 42 均分 |

---

## 2B. 比較推理結果 `run_sam_compare.py`

讀兩次（或多個）infer run，以 reference（通常 clear）計算 Matched IoU、Recall@0.5 等。**無需 GPU**。

```bash
python coldfog_test/run_sam_compare.py \
  --reference coldfog_test/results/infer/clear_20250602T120000Z \
  --query coldfog_test/results/infer/heavy_20250602T130000Z \
  --save-tex
```

多條件一次比較：

```bash
python coldfog_test/run_sam_compare.py \
  --reference coldfog_test/results/infer/latest_clear \
  --query coldfog_test/results/infer/latest_light \
            coldfog_test/results/infer/latest_medium \
            coldfog_test/results/infer/latest_heavy
```

### 輸出

```
coldfog_test/results/compare/
  clear_vs_heavy_<UTC>/
    per_sample.csv       # 每 sample 明細
    summary.json         # 按 query 聚合 mean/std
    summary_metrics.pdf  # 四項指標柱狀圖
    summary_table.tex    # --save-tex 時
```

| 指標欄位 | 意義 |
|----------|------|
| `matched_iou_vs_reference` | query mask 與 reference mask 的匹配 IoU |
| `mask_recall_at_0_5` | reference mask 中 best IoU ≥ 0.5 的比例 |
| `num_valid_masks` | query 圖有效 mask 數 |
| `mean_stability` | query mask 平均 stability |

---

## 2C. 深度分層比較 `run_sam_depth_compare.py`

在已完成 `run_sam_infer.py` 的基礎上，讀取 clear reference 的 SAM masks 與 DAv2 metric depth map，按 **reference mask 的 median depth** 分成 Near / Middle / Far，再統計各霧濃度在不同距離上的 SAM 退化程度。這一步是離線後處理，**無需 GPU**，也不會重新跑 SAM。

預設 depth 目錄為：

```
/mnt/newdisk/Documents/linzhanyang/data/cold_depth_metric_vitl_980
```

depth 檔名需與 sample_id 對齊：

```
{sample_id}_raw_depth_meter.npy
```

### 基本用法

```bash
cd /mnt/newdisk/Documents/linzhanyang/segment-anything

python coldfog_test/run_sam_depth_compare.py \
  --reference coldfog_test/results/infer/latest_clear \
  --query coldfog_test/results/infer/latest_light \
          coldfog_test/results/infer/latest_medium \
          coldfog_test/results/infer/latest_heavy \
  --save-tex
```

### Depth bins

| 分組 | 條件 | 解釋 |
|------|------|------|
| Near | `d <= 4m` | 近距離貨物、避障、抓取區域 |
| Middle | `4m < d <= 10m` | 貨架、通道中段、一般巡檢距離 |
| Far | `d > 10m` | 遠端通道、貨架盡頭、導航前視區域 |

其中 `d` 是每個 clear reference SAM mask 內 depth 的 median。若 depth map 尺寸與 SAM mask 尺寸不同，腳本會把 depth resize 到 mask 尺寸後再分組。

### 輸出

```
coldfog_test/results/compare/
  clear_vs_light_vs_medium_vs_heavy_depth_<UTC>/
    per_reference_mask.csv    # 每個 reference mask 的 depth bin 與 best IoU
    per_depth_bin.csv         # 每 query × depth bin 的聚合表
    summary.json              # 配置 + depth-bin 聚合結果
    summary_depth_table.tex   # --save-tex 時
```

| 指標欄位 | 意義 |
|----------|------|
| `depth_bin` | Near / Middle / Far |
| `sample_count` | 該 depth bin 涉及的 sample 數 |
| `reference_num_masks` | 該 depth bin 的 clear reference mask 數 |
| `query_num_valid_masks` | query 圖中落入該 depth bin 的有效 mask 數 |
| `query_valid_mask_ratio` | `query_num_valid_masks / reference_num_masks` |
| `matched_iou_vs_reference` | 該 depth bin 內 reference masks 的平均 best IoU |
| `mask_recall_at_0_5` | 該 depth bin 內 best IoU ≥ 0.5 的 reference mask 比例 |
| `missing_mask_rate_at_0_5` | `1 - mask_recall_at_0_5` |

### 常用參數

| 參數 | 預設 | 說明 |
|------|------|------|
| `--depth-dir` | `data/cold_depth_metric_vitl_980` | DAv2 metric depth NPY 目錄 |
| `--depth-suffix` | `_raw_depth_meter.npy` | depth 檔名後綴 |
| `--near-max-m` | `4.0` | Near 上界 |
| `--far-min-m` | `10.0` | Far 下界；Middle 為兩者之間 |
| `--min-match-area` | `100` | 匹配與 depth binning 時忽略小 mask |

---

## 2D. 論文繪圖腳本 `coldfog_test/plot/`

這兩個腳本都是離線繪圖，讀取已保存的 infer / compare 結果，**無需 GPU**，不會重新跑 SAM。

### 深度分層 heatmap

用 `per_depth_bin.csv` 畫兩張並排 heatmap：

- `(a) Matched IoU`
- `(b) Recall@0.5`

基本用法：

```bash
cd /mnt/newdisk/Documents/linzhanyang/segment-anything

python coldfog_test/plot/plot_depth_heatmaps.py
```

預設輸入：

```
coldfog_test/results/compare/latest_clear_vs_light_vs_medium_vs_heavy_depth/per_depth_bin.csv
```

預設輸出：

```
coldfog_test/plot/depth_stratified_sam_heatmaps.pdf
coldfog_test/plot/depth_stratified_sam_heatmaps.png
```

可指定輸入、輸出與 colormap：

```bash
python coldfog_test/plot/plot_depth_heatmaps.py \
  --input-csv coldfog_test/results/compare/latest_clear_vs_light_vs_medium_vs_heavy_depth/per_depth_bin.csv \
  --output-stem coldfog_test/plot/depth_stratified_sam_heatmaps \
  --cmap cividis
```

### Qualitative case visualization

用已保存的 `results/infer/` masks 畫 qualitative 對比圖：

```
Top row:    Clear image | Depth map | Light fog | Medium fog | Heavy fog
Bottom row:                         | Light masks | Medium masks | Heavy masks
```

Light / Medium / Heavy 的 fog 原圖與 masks overlay 分開顯示，便於直觀看出霧圖本身和 SAM 退化的對應關係。Masks overlay 會疊加所選 depth bins 的 clear reference masks，預設同時展示 **Near / Middle / Far**：

- Green：matched masks，IoU >= 0.5
- Red：missing masks，IoU < 0.5

先列出可選 sample：

```bash
cd /mnt/newdisk/Documents/linzhanyang/segment-anything

python coldfog_test/plot/plot_qualitative_depth_case.py --list-samples
```

按 `sample_id` 選圖：

```bash
python coldfog_test/plot/plot_qualitative_depth_case.py \
  --sample-id sdm_20260212_0292
```

也可以按 `clear/index.csv` 的行序號選圖：

```bash
python coldfog_test/plot/plot_qualitative_depth_case.py \
  --sample-index 34
```

若 mask 太多，可只顯示面積最大的前 N 個 masks：

```bash
python coldfog_test/plot/plot_qualitative_depth_case.py \
  --sample-id sdm_20260212_0292 \
  --max-masks 30
```

若想只突出某些深度範圍，例如只看 Far：

```bash
python coldfog_test/plot/plot_qualitative_depth_case.py \
  --sample-id sdm_20260212_0292 \
  --depth-bins Far
```

也可以組合多個 depth bins：

```bash
python coldfog_test/plot/plot_qualitative_depth_case.py \
  --sample-id sdm_20260212_0292 \
  --depth-bins Middle Far
```

批次處理 `latest_clear/index.csv` 中所有 sample，逐張輸出到 `coldfog_test/plot/`；單張失敗不會中斷整批：

```bash
cd /mnt/newdisk/Documents/linzhanyang/segment-anything

tail -n +2 coldfog_test/results/infer/latest_clear/index.csv | cut -d, -f1 | while read -r sid; do
  echo "=== $sid ==="
  python coldfog_test/plot/plot_qualitative_depth_case.py --sample-id "$sid" || echo "FAILED: $sid"
done
```

預設輸入：

```
coldfog_test/results/infer/latest_clear
coldfog_test/results/infer/latest_light
coldfog_test/results/infer/latest_medium
coldfog_test/results/infer/latest_heavy
```

若不存在 `latest_*` 軟連結，腳本會自動選擇 `results/infer/` 下對應條件最新的 `clear_*`、`light_*`、`medium_*`、`heavy_*` 目錄。

預設輸出：

```
coldfog_test/plot/qualitative_depth_case_<sample_id>.pdf
coldfog_test/plot/qualitative_depth_case_<sample_id>.png
```

常用參數：

| 參數 | 預設 | 說明 |
|------|------|------|
| `--sample-id` | 無 | 指定要可視化的 sample id |
| `--sample-index` | 無 | 按 `clear/index.csv` 行序號選圖 |
| `--list-samples` | 關 | 列出可選 sample id 後退出 |
| `--depth-bins` | `Near Middle Far` | 要展示的 reference mask 深度分組 |
| `--near-max-m` | `4.0` | median depth 小於等於此值的 reference mask 視為 Near |
| `--far-min-m` | `10.0` | median depth 大於此值的 reference mask 視為 Far；Middle 為兩者之間 |
| `--iou-threshold` | `0.5` | 判斷 matched / missing 的 IoU 閾值 |
| `--max-masks` | `0` | 限制所選 masks 顯示數量；`0` 表示不限制 |
| `--overlay-alpha` | `0.42` | 紅/綠 mask 填充透明度 |

---

## 1. 準備評測資料

從 `data/splits/test` 複製配對影像到 `data/sam_eval/`，不修改原目錄。

```bash
cd /mnt/newdisk/Documents/linzhanyang/segment-anything

python data/prepare_sam_eval.py
python data/prepare_sam_eval.py --clean   # 清空舊輸出後重建
```

輸出結構：

```
data/sam_eval/
  clear/    # GT 清晰圖
  light/
  medium/
  heavy/
  manifest.json
```

| 參數 | 預設 | 說明 |
|------|------|------|
| `--metadata` | `DehazeDDPM/plot/test_metadata.json` | 決定 80 個 test sample ID |
| `--source` | `data/splits/test` | 原始 test split 根目錄 |
| `--output` | `data/sam_eval` | 輸出根目錄 |
| `--clean` | 關 | 刪除已有 `--output` 後重新複製 |

---

## 1B. 準備去霧結果 `prepare_dehazed_sam_eval.py`

將 DehazeDDPM `infer.py` 輸出的 `{step}_{index}_out.png` 對齊到 SAM 使用的
`{sample_id}.png`。`index` 與 `DehazeDDPM/plot/test_metadata.json` 的 `index`
欄位一致；`sample_id` 由 `filename` 去掉霧濃度後綴得到（與 `manifest.json` 相同）。

### 檔名對應

| 來源 | 範例 | 對應鍵 |
|------|------|--------|
| DehazeDDPM results | `0_10_out.png` | `test_metadata.samples[9].index == 10` |
| test_metadata | `sd_20260215_0321_medium.jpg` | `fog_level=medium` |
| SAM manifest | `sd_20260215_0321.png` | `sample_id` |

### 基本用法

```bash
cd /mnt/newdisk/Documents/linzhanyang/segment-anything

python data/prepare_dehazed_sam_eval.py \
  --dehaze-results \
    /mnt/newdisk/Documents/linzhanyang/DehazeDDPM/experiments/test/physical_v1/Dehaze_ColdFog_finetune_test_ddim100_physical_v1_260526_091235/results
```

預設以 **symlink** 寫入 `data/sam_eval/dehazed/{sample_id}.png`，不複製大圖。
需要實體複製時加 `--copy`。

### 同時準備「實際輸入霧圖」baseline

test 集每張樣本的 `fog_level` 不同（low/medium/heavy）。若要比較
「去霧前輸入霧圖 vs 去霧後」，加 `--with-hazy-input`：腳本會依 metadata
從 `sam_eval/light|medium|heavy` 取對應霧圖，寫入 `sam_eval/hazy_input/`。
需先執行 **1. prepare_sam_eval.py**。

```bash
python data/prepare_dehazed_sam_eval.py \
  --dehaze-results /path/to/DehazeDDPM/experiments/.../results \
  --with-hazy-input \
  --clean
```

### 輸出

```
data/sam_eval/
  dehazed/
    {sample_id}.png           # symlink 或 copy
    dehazed_mapping.json      # index ↔ sample_id ↔ 源檔路徑
  hazy_input/                 # --with-hazy-input 時
    {sample_id}.png
```

### 參數

| 參數 | 預設 | 說明 |
|------|------|------|
| `--dehaze-results` | （必填） | DehazeDDPM infer 的 `results/` 目錄 |
| `--metadata` | `DehazeDDPM/plot/test_metadata.json` | 決定 index ↔ sample_id 對應 |
| `--step` | `0` | infer 輸出檔名前綴，對應 `{step}_{index}_out.png` |
| `--output` | `data/sam_eval/dehazed` | staging 輸出目錄 |
| `--copy` | 關 | 複製檔案；預設為 symlink |
| `--with-hazy-input` | 關 | 額外輸出 `hazy_input/` |
| `--hazy-input-dir` | `data/sam_eval/hazy_input` | hazy baseline 輸出目錄 |
| `--sam-eval-root` | `data/sam_eval` | `--with-hazy-input` 的霧圖來源根目錄 |
| `--clean` | 關 | 刪除已有 dehazed / hazy_input 後重建 |

---

## 2. 批量評測 `run_sam_eval.py`（一鍵舊流程）

對每個 sample 在 clear 上生成 **GT-SAM**（參考分割），再評估 light / medium / heavy 霧圖相對於 GT-SAM 的指標。若需分輪推理或接入去霧結果，請改用 **2A + 2B**。

### 基本用法

```bash
cd /mnt/newdisk/Documents/linzhanyang/segment-anything

python coldfog_test/run_sam_eval.py --gpu-id 0
```

### 快速試跑

```bash
python coldfog_test/run_sam_eval.py --gpu-id 0 --limit 2
```

### 輸出

```
coldfog_test/results/
  latest -> run_<UTC時間戳>/
  run_<UTC時間戳>/
    per_sample.csv       # 每 sample × fog level 明細
    summary.json         # 按霧濃度聚合 mean/std + 運行配置
    summary_metrics.pdf  # 四項指標柱狀圖
```

### 評測指標

| 欄位 | 意義 |
|------|------|
| `num_valid_masks` | fog 圖上 SAM 有效 mask 數量 |
| `mean_stability` | fog mask 平均 stability score |
| `matched_iou_vs_gt_sam` | 每個 GT-SAM mask 的最佳 IoU 再平均 |
| `mask_recall_at_0_5` | GT-SAM mask 中 best IoU ≥ 0.5 的比例 |
| `gt_sam_num_masks` | clear 上 GT-SAM mask 數（參考） |

GT-SAM = 無霧 clear 圖上的 SAM 自動分割，**非人工標註**。

### 路徑參數

| 參數 | 預設 | 說明 |
|------|------|------|
| `--manifest` | `data/sam_eval/manifest.json` | sample 清單 |
| `--data-root` | `data/sam_eval` | 含 clear/light/medium/heavy 的目錄 |
| `--checkpoint` | `checkpoints/sam_vit_h_4b8939.pth` | SAM ViT-H 權重 |
| `--output` | `coldfog_test/results` | 結果根目錄 |

### AMG 參數（Automatic Mask Generator）

四個霧濃度與 clear 的 GT-SAM **必須使用同一套設定**，結果才可比較。

| 參數 | 預設 | 說明 |
|------|------|------|
| `--points-per-side` | `32` | 每邊 prompt 點數（32×32=1024 點） |
| `--pred-iou-thresh` | `0.86` | 低於此 predicted IoU 的 mask 被過濾 |
| `--stability-score-thresh` | `0.90` | 低於此 stability 的 mask 被過濾 |
| `--crop-n-layers` | `1` | 多層 crop 推理（大圖/小物體通常更好） |
| `--crop-n-points-downscale-factor` | `2` | crop 層點密度縮放 |
| `--min-mask-region-area` | `100` | AMG 後處理：去掉面積過小的碎 mask（px） |

### 匹配參數（IoU / Recall 計算）

| 參數 | 預設 | 說明 |
|------|------|------|
| `--min-match-area` | `100` | 匹配時忽略 area 小於此值的 mask |

### 完整範例

```bash
python coldfog_test/run_sam_eval.py \
  --gpu-id 1 \
  --limit 5 \
  --pred-iou-thresh 0.86 \
  --stability-score-thresh 0.90 \
  --min-match-area 100 \
  --output coldfog_test/results
```

---

## 3. 可視化 `run_sam_overlay.py`（可選）

單張 sample 的 clear / light / medium 對比圖，用於論文插圖或肉眼檢查。

1. 編輯腳本內 `image_id`（例如 `ggl_20260202_0064.png`）
2. 確認 `dirs` 中路徑指向 `data/sam_eval/`（若仍為 `sam_eval/` 需改為 `data/sam_eval/clear` 等）
3. 指定 GPU 後執行：

```bash
cd /mnt/newdisk/Documents/linzhanyang/segment-anything
CUDA_VISIBLE_DEVICES=0 python coldfog_test/run_sam_overlay.py
```

輸出：`sam_outputs/sam_overlay_compare.png`

腳本內 AMG 參數與 `run_sam_eval.py` 預設一致；修改時請兩邊保持一致。

---

## 耗時參考

完整 80 個 sample：每 sample 需 SAM 推理 4 次（clear + 3 fog levels），ViT-H 在 GPU 上通常需數十分鐘至數小時，視卡型與解析度而定。建議先用 `--limit 2` 驗證流程。

---

## 目錄說明

| 路徑 | 內容 |
|------|------|
| `coldfog_test/sam_core.py` | 共用工具（mask 讀寫、指標、繪圖） |
| `coldfog_test/run_sam_infer.py` | 分目錄 SAM 推理（推薦） |
| `coldfog_test/run_sam_compare.py` | 比較 infer runs（無 GPU） |
| `coldfog_test/run_sam_depth_compare.py` | 按 reference mask depth bin 比較 infer runs（無 GPU） |
| `coldfog_test/run_sam_eval.py` | 一鍵批量評測（舊流程） |
| `coldfog_test/run_sam_overlay.py` | 單張可視化 |
| `coldfog_test/plot/plot_depth_heatmaps.py` | depth-bin 指標 heatmap 論文圖 |
| `coldfog_test/plot/plot_qualitative_depth_case.py` | 可選 sample 的五列 qualitative case 論文圖 |
| `coldfog_test/results/infer/` | infer 輸出 |
| `coldfog_test/results/compare/` | compare 輸出 |
| `coldfog_test/results/` | run_sam_eval 輸出 |
| `data/prepare_sam_eval.py` | 從 test split 複製 clear / 三霧度影像 |
| `data/prepare_dehazed_sam_eval.py` | 將 DehazeDDPM 去霧結果對齊為 SAM 檔名 |
| `data/sam_eval/` | 評測用配對影像（含 `dehazed/`、`hazy_input/`） |
