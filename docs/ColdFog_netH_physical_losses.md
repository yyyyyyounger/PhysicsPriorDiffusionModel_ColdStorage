# ColdFog netH Physical 版本 Loss 說明

本文檔整理當前實驗 `Dehaze_ColdFog_finetune_netH_physical_260508_123402` 中各個 loss 的計算方式、對比項來源與日誌含義，便於導入給其他 AI 或用於後續結果分析。

## 1. 實驗版本與入口

- 入口腳本：`trainColdFogNetHPhysical.sh`
- 實際命令：`python sr.py --config config/Dehaze_ColdFog_finetune_netH_physical.json`
- 實驗目錄：`experiments/Dehaze_ColdFog_finetune_netH_physical_260508_123402/`
- 模型類：`model.model.DDPM`
- 擴散模型：`which_model_G = "sr3"`，即 `model/sr3_modules/diffusion.py` 的 `GaussianDiffusion + UNet`
- 第一階段/條件網路：`netH = model.networkHelper.MPRfusion()`
- 當前配置開啟：`finetune_netH = true`

訓練時 `netH` 輸入 hazy 圖，輸出：

| 名稱 | 含義 | 來源 |
| --- | --- | --- |
| `output` / `out_J` | netH 預測的清晰圖 / 去霧估計 | `MPRfusion.forward()` |
| `stage1_output` | 第一階段中間輸出去霧圖 | `MPRfusion.forward()` |
| `out_T` | netH 預測的 transmission / 傳輸圖 | `conv_T_1 + conv_T_2` |
| `out_A` | netH 預測的大氣光 / atmospheric light 圖 | `ANet(hazy)` |
| `out_I` | 按大氣散射模型重建出的 hazy 圖 | `out_T * out_J + (1 - out_T) * out_A` |

`netG` 的條件輸入為：

```python
condition = torch.cat([output / 0.5 - 1, out_T / 0.5 - 1], dim=1)
```

也就是 3 通道 `out_J` 加 1 通道 `out_T`，共 4 通道。UNet 配置 `in_channel=7`，對應 `condition` 4 通道加 noisy HR 3 通道。

## 2. 日誌中各 loss 快速對照

`train.log` 中訓練行目前會出現：

```text
l_pix loss_t loss_asm loss_physical_total loss_total
```

| 日誌 key | 對比項 | 公式概要 | 是否反傳 | 權重 |
| --- | --- | --- | --- | --- |
| `l_pix` | diffusion 預測噪聲 vs 真實噪聲；另含頻域重建項 | `(L1_sum(noise, pred_noise) + 0.01 * L1_sum(abs(FFT(x0_pred)), abs(FFT(HR)))) / (B*C*H*W)` | 是 | 1.0 |
| `loss_t` | `out_T` vs `exp(-beta * depth)` | `L1(clamp(out_T), clamp(t_gt))` | 是，當 `finetune_netH=true` 且有 `depth/beta` | `lambda_t=0.01` |
| `loss_asm` | `out_I` vs 輸入 hazy 圖 | `L1(out_I, hazy_input_01)` | 是，當 `finetune_netH=true` 且有 `depth/beta` | `lambda_asm=0.05` |
| `loss_physical_total` | 物理約束總和 | `0.01 * loss_t + 0.05 * loss_asm` | 是 | 已加權 |
| `loss_total` | 總訓練 loss | `l_pix + loss_physical_total` | 是 | 已加權 |

注意：`loss_t` 本身是未乘權重的原始 L1；真正加進總 loss 的是 `0.01 * loss_t`。`loss_asm` 同理，真正加進總 loss 的是 `0.05 * loss_asm`。

## 3. `l_pix`: diffusion 訓練 loss

### 3.1 來源文件

- `model/model.py`: `DDPM.optimize_parameters()`
- `model/networks.py`: `define_G()`
- `model/sr3_modules/diffusion.py`: `GaussianDiffusion.p_losses()`

### 3.2 計算流程

`model/networks.py` 中建立 `netG` 時固定使用：

```python
loss_type = "l1"
```

因此 diffusion 內部 loss 函數為：

```python
nn.L1Loss(reduction="sum")
```

一次訓練 step 中：

1. 取 GT 清晰圖 `HR` 作為 diffusion 的 `x_start`。
2. 隨機抽一個 diffusion timestep。
3. 隨機生成高斯噪聲 `noise`。
4. 按正向擴散公式加噪：

```python
x_noisy = sqrt_alpha * x_start + sqrt(1 - sqrt_alpha**2) * noise
```

5. UNet 接收 `[condition, x_noisy]`，預測噪聲：

```python
pred_noise = denoise_fn(torch.cat([condition, x_noisy], dim=1), sqrt_alpha)
```

6. 噪聲預測 loss：

```python
loss_noise = L1_sum(noise, pred_noise)
```

7. 用預測噪聲反推 `x0_pred`：

```python
x0_pred = (x_noisy - sqrt(1 - sqrt_alpha**2) * pred_noise) / sqrt_alpha
```

8. 頻域 loss：

```python
loss_frequency = L1_sum(abs(FFT(x0_pred)), abs(FFT(HR)))
```

9. diffusion 返回：

```python
loss_diffusion_raw = loss_noise + 0.01 * loss_frequency
```

10. `model/model.py` 再按像素數平均：

```python
l_pix = loss_diffusion_raw / (B * C * H * W)
```

### 3.3 解讀

雖然名稱叫 `l_pix`，但它不是簡單的「輸出去霧圖 vs GT 清晰圖」像素 L1。它實際上是 diffusion 的噪聲預測 L1，加上一個基於 FFT 幅值的頻域重建約束，再除以 batch 內像素總數。

## 4. `loss_t`: 傳輸圖物理監督

### 4.1 來源文件

- `model/model.py`: `_compute_physical_losses()`
- `data/LRHR_dataset.py`: `__getitem__()`
- `data/util.py`: `paired_paths_from_metadata()`, `load_depth_npy()`

### 4.2 對比項

`loss_t` 比較的是：

- 預測項：`netH` 輸出的 `out_T`
- 目標項：由 metadata 中的 `depth` 和 `beta` 按物理模型算出的 `t_gt`

目標傳輸圖：

```python
t_gt = exp(-beta * depth)
```

其中：

- `depth` 從 metadata 指定的 `.npy` 文件讀入；
- `beta` 從 metadata 的 `beta` 欄位讀入；
- 若尺寸與 `out_T` 不一致，`depth` 會 bilinear resize 到 `out_T` 的尺寸；
- `out_T` 和 `t_gt` 都會 clamp 到 `[1e-4, 1.0]`。

最終：

```python
loss_t = F.l1_loss(clamp(out_T, 1e-4, 1.0), clamp(t_gt, 1e-4, 1.0))
```

`F.l1_loss` 默認 `reduction="mean"`，所以這裡是平均 L1，而不是 sum。

### 4.3 解讀

`loss_t` 是 transmission map 的物理先驗監督。它不直接比較最終去霧輸出和 GT 清晰圖，而是在約束 `netH` 預測的傳輸圖要接近 Beer-Lambert / 大氣散射模型中的：

```text
t(x) = exp(-beta * depth(x))
```

在當前配置中，加進總 loss 的實際項是：

```python
0.01 * loss_t
```

## 5. `loss_asm`: 大氣散射模型重建 loss

### 5.1 來源文件

- `model/networkHelper.py`: `MPRfusion.forward()`
- `model/model.py`: `_compute_physical_losses()`

### 5.2 對比項

`netH` 先按大氣散射模型合成 hazy 圖：

```python
out_I = out_T * out_J + (1 - out_T) * out_A
```

其中：

- `out_J`: netH 預測的清晰圖 / 去霧估計；
- `out_T`: netH 預測的傳輸圖；
- `out_A`: netH 預測的大氣光，來自 `ANet(hazy)`；
- `out_I`: 用上述三者合成回來的 hazy 圖。

輸入 hazy 圖原本在 dataset 中被歸一化到 `[-1, 1]`，訓練時會轉回 `[0, 1]`：

```python
hazy_input_01 = (SR + 1.0) / 2.0
```

最終：

```python
loss_asm = F.l1_loss(out_I, hazy_input_01)
```

同樣是默認 `reduction="mean"`。

### 5.3 解讀

`loss_asm` 約束 `netH` 的三個物理輸出 `out_J / out_T / out_A` 能夠重新合成原始 hazy 輸入。它是自一致性約束，不需要額外的大氣光 GT。

當前代碼中沒有看到對 `out_A` 的直接 ground truth supervision；`out_A` 是由 `ANet` 預測，通過 `loss_asm` 間接受到約束。

在當前配置中，加進總 loss 的實際項是：

```python
0.05 * loss_asm
```

## 6. `loss_physical_total` 和 `loss_total`

### 6.1 當前權重

配置文件：

```json
"lambda_t": 0.01,
"lambda_asm": 0.05
```

因此：

```python
loss_physical_total = 0.01 * loss_t + 0.05 * loss_asm
loss_total = l_pix + loss_physical_total
```

以日誌中的一行為例：

```text
l_pix: 5.2307e-01 loss_t: 1.3905e-01 loss_asm: 7.7069e-02 loss_physical_total: 5.2440e-03 loss_total: 5.2832e-01
```

對應：

```text
0.01 * 0.13905 + 0.05 * 0.077069 = 0.005244
0.52307 + 0.005244 = 0.528314
```

與日誌中的 `loss_physical_total`、`loss_total` 一致。

### 6.2 物理 loss 何時為 0

訓練時只有同時滿足以下條件才加入物理 loss：

- `finetune_netH = true`
- batch 裡有 `depth`
- batch 裡有 `beta`
- 已經有 `out_T`
- 已經有 `out_I`

如果不滿足，`loss_t` 和 `loss_asm` 都會是 0。

## 7. metadata 與物理參數來源

當前配置指定：

```json
"metadata_csv": "/mnt/newdisk/Documents/linzhanyang/data/finetune/metadata.csv",
"finetune_root": "/mnt/newdisk/Documents/linzhanyang/data/finetune"
```

dataset 會優先從 metadata 讀：

| metadata 欄位 | 用途 |
| --- | --- |
| `hazy` | 輸入霧圖路徑，對應 batch 裡的 `SR` |
| `gt` 或 `clear` | GT 清晰圖路徑，對應 batch 裡的 `HR` |
| `depth` | 深度 `.npy` 路徑，用於計算 `t_gt` |
| `beta` | 散射係數，用於計算 `t_gt` |
| `split` | 過濾 train / val |

如果 metadata 不存在，代碼才會回退到資料夾配對。但當前 physical 配置有 metadata，因此 `loss_t` 的 `depth/beta` 來源就是 metadata。

## 8. 驗證階段的 loss

訓練過程中每 `val_freq=5000` step 會進行 validation。驗證時：

1. 使用 `beta_schedule.val`；
2. 當前配置驗證採樣器為 DDIM，`sample_steps=20`，`ddim_eta=0.0`；
3. 調用 `diffusion.test()` 生成輸出；
4. 再調用 `update_physical_log()` 計算 `loss_t`、`loss_asm`、`loss_physical_total`；
5. 對整個 val set 求平均。

驗證日誌中的 `loss_t/loss_asm/loss_physical_total` 是物理 loss 的驗證集平均值，不含 `l_pix`，因為 validation 主要在跑生成和 PSNR，不重新做 diffusion training loss。

## 9. 給其他 AI 分析時的重點提示

1. `l_pix` 名稱容易誤導：它不是最終輸出圖和 GT 的普通像素 loss，而是 diffusion 訓練 loss。
2. `loss_t` 是 transmission map 監督：`out_T` 對 `exp(-beta * depth)`。
3. `loss_asm` 是大氣散射重建自一致性：`out_I` 對輸入 hazy 圖。
4. `out_A` 沒有直接 GT loss，目前靠 `loss_asm` 間接學。
5. 物理 loss 權重較小：`loss_t` 乘 0.01，`loss_asm` 乘 0.05；分析數值時要看加權後的 `loss_physical_total`。
6. `loss_total` 中 diffusion loss 仍然占主導，物理 loss 主要是輔助約束 `netH` 的物理中間量。
7. validation 的物理 loss 是在推理/採樣後重新計算的物理一致性指標，不等同於訓練 step 的 `loss_total`。

## 10. 主要代碼位置

| 文件 | 內容 |
| --- | --- |
| `trainColdFogNetHPhysical.sh` | physical 版本訓練入口 |
| `config/Dehaze_ColdFog_finetune_netH_physical.json` | 當前實驗配置、loss 權重、資料集、採樣器 |
| `sr.py` | 訓練循環、日誌輸出、驗證流程 |
| `model/model.py` | `DDPM.optimize_parameters()`、物理 loss 計算、總 loss 組合 |
| `model/sr3_modules/diffusion.py` | diffusion noise loss 和 frequency loss |
| `model/networkHelper.py` | `MPRfusion` 輸出 `out_J/out_T/out_A/out_I` |
| `data/LRHR_dataset.py` | batch 中 `HR/SR/depth/beta` 的組裝 |
| `data/util.py` | metadata 讀取、depth 載入與 resize |

