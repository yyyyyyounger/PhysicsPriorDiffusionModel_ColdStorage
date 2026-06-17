# 冷庫去霧資料集（Cold-Fog）說明

## 目錄概要

| 路徑 / 檔案 | 內容說明 |
| --- | --- |
| `cold_inputs/` | 約 400 張原始清晰影像，作為合成與切分的基礎。 |
| `cold_fog_synthesized/` | 依霧濃度合成的有霧影像（如 light / medium / heavy），各濃度約 400 張。 |
| `splits/` | 由 `data/split.ipynb` 依 `split_manifest_seed42.json` 劃分的 train / val / test（例如 train 280、val 40、test 80）。有霧圖按濃度分子目錄，GT 為對應檔名的清晰圖。 |
| `dehazeddpm_test/` | 由 `data/split.ipynb` 從 test split 建立的扁平測試集（`hazy_test` / `gt_test`），80 張，檔名一一對應；清單見 `dehazeddpm_manifest_seed42.json`。供 DehazeDDPM 基線與微調後對照評測。 |
| `finetune/` | 由 `prepare_finetune_data.py` 產生的扁平目錄，供微調使用：`train_hazy` / `train_gt`（840 對 = 280×3 濃度）、`val_hazy` / `val_gt`（120 對 = 40×3）。hazy 與 GT 使用相同檔名（含副檔名）；GT 端可為指向 `.jpg` 的 `.png` 符號連結，以便 `LRHR_dataset.paired_paths_from_folder` 配對。 |
| `baseline_results_metrics.csv` | 預訓練 DENSE 模型在測試集上的基線指標。 |
| `stats/splits/` | Split 組成統計（貨物類別、真實/AI 分佈）；讀取 `splits/split_manifest_seed42.json`，見 `split_composition_stats.ipynb`。 |

## 建立來源（目錄 / 檔案 ↔ 腳本）

以下列出 `data/` 下各目錄與關鍵檔案**由誰建立**。上游依賴順序見文末流程。

| 路徑 / 檔案 | 建立方式 | 建立腳本 / 工具 | 備註 |
| --- | --- | --- | --- |
| `cold_inputs/` | 人工採集 | — | 約 400 張清晰冷庫圖，非程式生成 |
| `cold_depth_metric_vitl_980/` | 程式輸出 | `Depth-Anything-V2/metric_depth/run.py` | 每張清晰圖對應 `{id}_raw_depth_meter.npy` 與可視化 `.png` |
| `cold_fog_synthesized/` | Notebook | `Depth-Anything-V2/synthesize_fog.ipynb` | 子目錄 `light/`、`medium/`、`heavy/`，各約 400 張 |
| `splits/` | Notebook | `data/split.ipynb` | 含 `train/`、`val/`、`test/`（各含 `gt/`、`hazy/{light,medium,heavy}/` 與 depth npy） |
| `splits/split_manifest_seed42.json` | Notebook | `data/split.ipynb` | 切分 seed=42；train 280 / val 40 / test 80 |
| `dehazeddpm_test/` | Notebook | `data/split.ipynb` | **非** `prepare_finetune_data.py` 建立 |
| `dehazeddpm_test/gt_test/` | Notebook | `data/split.ipynb` | 80 張 GT，檔名含 `_low` / `_medium` / `_heavy` 後綴 |
| `dehazeddpm_test/hazy_test/` | Notebook | `data/split.ipynb` | 80 張有霧圖，與 `gt_test/` **同名** |
| `dehazeddpm_test/dehazeddpm_manifest_seed42.json` | Notebook | `data/split.ipynb` | 記錄每 ID 抽到的霧級（選級 seed = 43，即 `SEED + 1`） |
| `finetune/` | Python 腳本 | `data/prepare_finetune_data.py` | 自 `splits/train`、`splits/val` 展平為 symlink |
| `finetune/train_hazy/`、`train_gt/`、`train_depth/` | Python 腳本 | `data/prepare_finetune_data.py` | 840 對（280×3 霧級） |
| `finetune/val_hazy/`、`val_gt/`、`val_depth/` | Python 腳本 | `data/prepare_finetune_data.py` | 120 對（40×3 霧級） |
| `finetune/metadata.csv`、`metadata.jsonl` | Python 腳本 | `data/prepare_finetune_data.py` | 每對樣本的 hazy / gt / depth 路徑與 beta 等 |
| `stats/splits/id_prefix_labels.json` | 人工維護 | — | ID 前綴 → 貨物類別、real/ai 標籤；統計 notebook 的輸入 |
| `stats/splits/split_id_labels.csv` 等 | Notebook | `data/stats/splits/split_composition_stats.ipynb` | 讀 `splits/split_manifest_seed42.json` 產生統計 |
| `stats/depth/` | Python 腳本 | `data/stats/depth/check_depth_saturation.py` | 深度飽和度檢查摘要 |
| `stats/postprocess/` | Notebook | `data/stats/postprocess/color_stats_and_dehaze_postprocess.ipynb` | GT / 去霧後色彩統計 |
| `baseline_results_metrics.csv` | Notebook | `data/results_analysis.ipynb` | 基線模型在 `dehazeddpm_test` 上的指標，非資料集構建產物 |
| `backup/` | 人工備份 | — | 例如 `backup/splits_0506/`，非 pipeline 自動步驟 |

### 重建指令（依賴順序）

```text
1. cold_inputs/                          ← 人工放置清晰圖
2. Depth-Anything-V2/metric_depth/run.py ← → cold_depth_metric_vitl_980/
3. Depth-Anything-V2/synthesize_fog.ipynb ← → cold_fog_synthesized/
4. data/split.ipynb                      ← → splits/ + dehazeddpm_test/
5. python data/prepare_finetune_data.py  ← → finetune/
```

`split.ipynb` 內 `CLEAN_OUTPUT_DIRS=True` 會先清空再重建 `splits/` 與 `dehazeddpm_test/`。  
`prepare_finetune_data.py` 預設也會清空 `finetune/` 下受管目錄（可用 `--no-clean` 保留）。

### 常見混淆

| 目錄 | 用途 | 建立者 |
| --- | --- | --- |
| `dehazeddpm_test/` | DehazeDDPM **推論 / 測試評測**（80 對，每 ID 一霧級） | `split.ipynb` |
| `finetune/` | DehazeDDPM **微調**（train 840 + val 120，三霧級全展開） | `prepare_finetune_data.py` |

兩者都服務 DehazeDDPM，但分工不同：**測試集在 split 階段建立，微調集在 prepare 階段建立**。  
專案內**不存在** `dehazeddpm_train/`、`dehazeddpm_val/` 目錄。

更完整的物理模型與切分規則見 repo 根目錄 [`cold_storage_dataset_pipeline.md`](../cold_storage_dataset_pipeline.md)。

## 微調資料準備

在專案內執行：

```bash
python data/prepare_finetune_data.py
```

可選：`--dry-run` 僅列印將建立的連結；`--splits-root` / `--out-root` 自訂路徑。
