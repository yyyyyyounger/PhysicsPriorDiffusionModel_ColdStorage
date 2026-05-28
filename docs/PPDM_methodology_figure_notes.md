# PPDM Implementation Notes for Methodology Figure

本文整理目前倉庫中可作為 PPDM 方法圖依據的實作細節。這裡的「PPDM」不是一個獨立命名的 Python class 或 package；目前 repo 中對應的是 `ColdFog_finetune_netH_physical` 這條實驗線：在原 DehazeDDPM 的兩階段框架上，加入 ColdFog 資料適配、`netH` 聯合微調、transmission/depth 物理監督，以及大氣散射模型重建約束。

> 繪圖時可將它表述為：Physical-prior DehazeDDPM / PPDM variant = `netH` physical prior estimator + conditional diffusion denoiser.

## 1. 一句話架構

輸入有霧圖 `I` 先經 `netH/MPRfusion` 估計初步清晰圖 `J`、透射率圖 `T`、大氣光 `A`，並由大氣散射模型重建 `I_hat = T * J + (1 - T) * A`；其中 `J` 與 `T` 被拼接成 diffusion 條件，引導 `netG/SR3` 從噪聲反向生成最終去霧結果。

核心資料流：

```text
Hazy image I
  -> netH / MPRfusion
      -> J: preliminary dehazed image
      -> T: transmission map
      -> A: atmospheric light
      -> I_hat = T * J + (1 - T) * A

concat(J, T)
  -> conditional SR3/DDPM netG
  -> final dehazed image x0
```

## 2. 與原 DehazeDDPM 圖二的對照

你上傳的原 DehazeDDPM 圖二可以理解成：

| 原圖元素 | 原 DehazeDDPM 含義 | 本倉庫 PPDM 對應畫法 |
| --- | --- | --- |
| Stage1 | 物理建模 / 第一階段先驗估計 | `netH/MPRfusion` 從 hazy input 估計 `J/T/A/I_hat` |
| PM | Physical Modelling | 可改成 `Physical-prior netH` 或 `Physical Prior Estimator` |
| ASM | Atmospheric Scattering Model | 保留，公式改明確寫 `I_hat = T J + (1 - T) A` |
| `J` | 初步去霧圖 | `out_J` / `output`，作為 diffusion 條件之一 |
| `trmap` | transmission map | `out_T`，同時作為 diffusion 條件與 `loss_t` 預測項 |
| `A` | atmospheric light | `out_A`，由 `ANet(hazy)` 預測 |
| Stage2 | diffusion process / denoise process | `netG/GaussianDiffusion + UNet` 條件反向採樣 |
| FDC/CDF | 原圖中的條件模組示意 | 本實作未命名為 FDC/CDF；建議改畫成 `Condition concat [J,T]` 與 `Conditional UNet denoiser` |
| Loss | Stage1/Stage2 訓練信號 | PPDM 多三條明確 loss：`l_pix`、`loss_t`、`loss_asm` |

建議首圖不要照搬原圖中的 `FDC`、`CDF` 命名，因為本倉庫沒有這兩個同名模組。若需要保留視覺語義，可把 FDC 的位置改成 `Condition Builder: concat(J,T)`，把 CDF 的位置改成 `Conditional Denoising UNet`。

## 3. 模組與張量清單

| 圖中節點 | 建議標籤 | 程式中的名稱 | 形狀/通道 | 來源 |
| --- | --- | --- | --- | --- |
| 有霧輸入 | Hazy image `I` | `data['SR']`，轉成 `hazy_input_01` | 3 ch | `data/LRHR_dataset.py`, `model/model.py` |
| 清晰 GT | Clean target `J_gt` / `x0` | `data['HR']` | 3 ch | diffusion training target |
| 第一階段網路 | Physical-prior netH / MPRfusion | `self.netH = MPRfusion()` | network | `model/model.py` |
| 初步去霧圖 | Preliminary dehazed image `J` | `out_J` / `output` | 3 ch | `MPRfusion.forward()` |
| 中間輸出 | Stage-1 intermediate | `stage1_output` | 3 ch | `SAM` branch |
| 透射率圖 | Transmission map `T` | `out_T` | 1 ch | `conv_T_1 + conv_T_2` |
| 大氣光 | Atmospheric light `A` | `out_A` | 3 ch, often global/low spatial information | `ANet(xcopy)` |
| 重建霧圖 | Reconstructed hazy `I_hat` | `out_I` | 3 ch | `out_T * out_J + (1 - out_T) * out_A` |
| 條件拼接 | Condition `c=[J,T]` | `condition` | 4 ch | `torch.cat([out_J, out_T], dim=1)` |
| 擴散輸入 | Noisy clean image `x_t` | `x_noisy` | 3 ch | `q_sample(HR, t)` |
| 條件擴散網路 | Conditional SR3/DDPM denoiser | `netG/GaussianDiffusion/UNet` | 7 ch input, 3 ch output | `condition` 4 ch + `x_t` 3 ch |
| 最終輸出 | Final dehazed image | `self.SR` / `visuals['Out']` | 3 ch | `super_resolution(condition)` |

最重要的繪圖事實：

- `netH` 輸出 5 個值：`out_J, stage1_output, out_T, out_A, out_I`。
- diffusion 條件不是直接使用原 hazy 圖，而是使用 `concat(out_J, out_T)`。
- UNet 配置 `in_channel=7`，因為反向去噪時輸入為 4 通道條件加 3 通道 noisy image。
- `out_A` 沒有直接 GT supervision，主要透過 `loss_asm` 間接受約束。

## 4. Training Flow

訓練時的完整流程建議畫成「主幹前向箭頭 + 上方/下方 loss 箭頭」：

```text
metadata.csv
  -> hazy I, clean GT, depth d, beta

hazy I
  -> netH
      -> J, T, A
      -> I_hat = T J + (1 - T) A

depth d + beta
  -> T_gt = exp(-beta * d)

[J, T] + clean GT
  -> q(x_t | x0)
  -> Conditional UNet predicts noise
  -> l_pix

T vs T_gt
  -> loss_t

I_hat vs I
  -> loss_asm

loss_total = l_pix + 0.01 loss_t + 0.05 loss_asm
  -> update netG + selected netH parameters
```

訓練程式入口：

- `trainColdFogNetHPhysical.sh`: 執行 `python sr.py --config config/Dehaze_ColdFog_finetune_netH_physical.json`。
- `sr.py`: 建立 dataset/model，對每個 batch 呼叫 `feed_data()` 與 `optimize_parameters()`。
- `model/model.py`: `DDPM.optimize_parameters()` 中同時跑 `netH`、`netG` 與三個 loss。

## 5. Inference Flow

推理圖可以比訓練圖簡化，保留兩階段：

```text
hazy I
  -> netH
      -> J, T
  -> concat(J, T)
  -> reverse diffusion sampler
      DDPM / DDIM / DPM-Solver++
  -> final dehazed image
```

注意推理時：

- 不需要 `depth`、`beta` 也能生成結果。
- `depth/beta` 只在 physical loss 訓練或帶 metadata 的驗證中用於計算 `loss_t`。
- 本倉庫已支援 `ddpm`、`ddim`、`dpm_solver_pp` 三種驗證/推理 sampler；方法首圖若不討論效率，可只畫「reverse diffusion」。

## 6. Loss Definition for Figure

### 6.1 Diffusion loss

`l_pix` 這個名稱在程式中不是最終圖像與 GT 的普通 pixel L1。它實際包含：

```text
l_pix = L1(noise, predicted_noise)
        + 0.01 * L1(|FFT(x0_pred)|, |FFT(x0)|)
```

然後在 `model/model.py` 中除以 `B*C*H*W`。

圖中可以簡化標為：

```text
Diffusion denoising loss + frequency consistency
```

### 6.2 Transmission loss

metadata 提供 depth `d` 與散射係數 `beta`，程式生成 transmission target：

```text
T_gt = exp(-beta * d)
loss_t = L1(T, T_gt)
```

圖中建議畫一條從 `Depth + beta` 到 `T_gt`，再到 `loss_t` 的輔助分支。

### 6.3 Atmospheric scattering reconstruction loss

`netH` 先用大氣散射模型重建有霧圖：

```text
I_hat = T * J + (1 - T) * A
loss_asm = L1(I_hat, I)
```

這條 loss 對繪圖很重要，因為它能直接說明 physical prior 如何約束 `J/T/A` 不是任意中間量，而是要能合成回輸入霧圖。

### 6.4 Total loss

physical 版本配置：

```text
loss_total = l_pix + lambda_t * loss_t + lambda_asm * loss_asm
lambda_t = 0.01
lambda_asm = 0.05
```

圖中若空間不足，可以寫：

```text
L = L_diff + 0.01 L_T + 0.05 L_ASM
```

## 7. 建議首圖版面

### 7.1 三帶式版面

最適合報告方法論章節首圖的結構：

```text
Top band: Physical supervision
  depth + beta -> T_gt -> L_T
  I_hat vs I -> L_ASM

Middle band: PPDM forward path
  Hazy I -> netH -> J, T, A, I_hat -> concat(J,T) -> conditional diffusion -> dehazed x0

Bottom band: Diffusion process
  x0 -> q(x_t | x0) -> x_t
  [concat(J,T), x_t, t] -> UNet -> predicted noise -> reverse step x_{t-1}
```

這樣能同時保留原 DehazeDDPM 圖二的 Stage1/Stage2 視覺語言，又能突出你的新增物理監督。

### 7.2 節點命名建議

建議使用以下英文標籤，適合放進論文章節圖：

- `Hazy input I`
- `Physical-prior estimator netH`
- `Preliminary dehazed image J`
- `Transmission map T`
- `Atmospheric light A`
- `Atmospheric scattering reconstruction I_hat`
- `Condition c = concat(J, T)`
- `Conditional diffusion denoiser netG`
- `Final dehazed image`
- `T_gt = exp(-beta d)`
- `L_T`
- `L_ASM`
- `L_diff`

如果要中文圖：

- `有霧輸入 I`
- `物理先驗估計器 netH`
- `初步去霧圖 J`
- `透射率圖 T`
- `大氣光 A`
- `大氣散射重建 I_hat`
- `條件拼接 c=[J,T]`
- `條件擴散去噪器 netG`
- `最終去霧結果`

### 7.3 顏色建議

可延續原圖語義：

- `netH / physical branch`: green
- `transmission / depth supervision`: blue or cyan
- `ASM reconstruction`: orange
- `diffusion denoiser`: purple
- `loss arrows`: dashed red/orange
- `inference forward arrows`: solid black
- `training-only arrows`: dashed gray

## 8. 和 Baseline 的差異應畫在哪裡

相對原始 DehazeDDPM baseline，本實作最值得在圖中突出三個差異：

1. `netH` 不只是 frozen physical prior provider，而是可聯合微調。
2. `netH` 的 `T` 被 depth/beta 形成的 `T_gt` 直接監督。
3. `J/T/A` 被 ASM reconstruction loss 約束，使 `I_hat` 接近原 hazy input。

可以在圖 caption 或圖內小框寫：

```text
Compared with DehazeDDPM, PPDM adds physical supervision on netH:
T_gt = exp(-beta d), I_hat = T J + (1 - T) A.
The refined J and T are used as conditional inputs for diffusion.
```

## 9. 原始 DehazeDDPM vs 本工作修改

本節按基準提交 `5ee18db0646de74741714c83493a8d3f17c1a8c2` 劃分：基準提交中已存在的內容視為「原始 DehazeDDPM 已有」，之後為了 ColdFog/PPDM 實驗加入或改造的內容視為「本工作修改」。

### 9.1 原始 DehazeDDPM 已有的內容

| 圖中可保留的 baseline 元素 | 原始已有內容 | 對應文件 |
| --- | --- | --- |
| 兩階段架構 | `netH` 先估計物理/先驗條件，`netG` 再做 conditional diffusion 去噪 | `model/model.py`, `model/networkHelper.py`, `model/sr3_modules/diffusion.py` |
| `netH/MPRfusion` | 第一階段網路 `MPRfusion()` 已存在 | `model/networkHelper.py` |
| `J/T/A/I_hat` 物理中間量 | `out_J`, `stage1_output`, `out_T`, `out_A`, `out_I` 已由 `MPRfusion.forward()` 輸出 | `model/networkHelper.py` |
| 大氣散射重建公式 | `out_I = out_T * out_J + (1 - out_T) * out_A` 已存在 | `model/networkHelper.py` |
| diffusion 條件 | `condition = concat(out_J, out_T)` 已存在，作為 `netG` 條件輸入 | `model/model.py` |
| SR3/DDPM 反向生成 | `GaussianDiffusion + UNet` 的 conditional sampling 已存在 | `model/sr3_modules/diffusion.py`, `model/sr3_modules/unet.py` |
| diffusion 訓練 loss | 噪聲預測 L1 與 FFT magnitude consistency 已存在 | `model/sr3_modules/diffusion.py` |
| DENSE/NH 原始配置 | 公開數據集 DENSE/NH 的 baseline 配置與 PreNet 權重接口已存在 | `config/Dehaze_DENSE.json`, `config/Dehaze_NH.json` |
| folder-pair dataset | 原始資料讀取是 hazy/GT 圖像資料夾配對 | `data/LRHR_dataset.py`, `data/util.py` |

對比圖中，這些可以畫成灰色或淡色的「Inherited from DehazeDDPM」部分。尤其是 `netH -> J/T/A -> condition -> diffusion` 這條主幹不應被說成完全新發明；新意在於下面的 ColdFog 適配與 physical supervision。

### 9.2 本工作新增或改造的內容

| 新增/改造點 | 對方法圖的意義 | 對應文件 |
| --- | --- | --- |
| ColdFog 訓練/測試配置 | 將原 DENSE/NH 場景遷移到自建 ColdFog 資料，圖中可標 `ColdFog adaptation` | `config/Dehaze_ColdFog_finetune*.json`, `config/test_ColdFog_finetune*.json` |
| ColdFog 腳本入口 | 固化訓練/推理命令，方便重現 PPDM 實驗線 | `trainColdFog*.sh`, `testColdFogFinetune*.sh` |
| `finetune_netH` 控制 | 原 baseline 中 `netH` 主要作為 PreNet/條件提供者；本工作支持選擇性解凍並聯合微調 `netH` | `model/model.py`, `config/Dehaze_ColdFog_finetune_netH*.json` |
| `lr_netH` optimizer group | `netG` 和 `netH` 使用不同 learning rate，圖中可標 `joint optimization` | `model/model.py` |
| `resume_stateH_finetune` | 可載入已微調 `netH` 權重，讓 `netG`/`netH` checkpoint 成對復現 | `model/model.py`, config files |
| netH checkpoint 保存 | `finetune_netH=true` 時額外保存 `I*_netH.pth` | `model/model.py` |
| metadata/depth/beta 管線 | 讓每個訓練樣本除了 `I/GT` 外攜帶 `depth` 與 `beta`，支撐 physical loss | `data/util.py`, `data/LRHR_dataset.py`, `data/__init__.py` |
| transmission physical loss | 新增 `T_gt = exp(-beta * depth)`，用 `loss_t` 監督 `out_T` | `model/model.py`, `config/Dehaze_ColdFog_finetune_netH_physical.json` |
| ASM reconstruction loss | 新增 `loss_asm = L1(out_I, hazy input)`，約束 `J/T/A` 的物理一致性 | `model/model.py` |
| total loss 組合 | 將訓練目標改為 `l_pix + 0.01 loss_t + 0.05 loss_asm` | `model/model.py`, physical config |
| physical validation log | 驗證/推理階段記錄 `loss_t/loss_asm/loss_physical_total` | `model/model.py`, `sr.py` |
| physical visual saving | 保存 `out_T/out_A/out_I/stage1_output/output`，可直接作圖中示例小圖 | `core/metrics.py`, `sr.py`, `infer.py` |
| DDIM / DPM-Solver++ | 推理加速與效率消融，不是 PPDM 主體新結構，但可作額外 sampler 對比圖 | `model/sr3_modules/diffusion.py`, `model/sr3_modules/dpm_solver_pp.py` |
| 多 GPU 推理 | 測試集推理分片與指標彙總，屬實驗工程改造 | `infer.py` |
| 固定 seed | 保證不同 sampler/GPU 分片的公平比較 | `core/seed.py`, `sr.py`, `infer.py` |
| 實驗/繪圖工具 | 主結果、消融、訓練曲線、失敗案例等圖表生成 | `plot/`, `docs/experiments_summary/` |

對比圖中，這些可以畫成彩色或高亮的「Proposed / Our modifications」部分。最適合放在主架構圖上的新增內容是：

```text
Depth + beta -> T_gt -> L_T
J, T, A -> I_hat -> L_ASM
L = L_diff + 0.01 L_T + 0.05 L_ASM
Joint finetuning of netG and selected netH modules
```

### 9.3 對比圖建議畫法

可以做一張左右對比圖：

```text
Left: DehazeDDPM baseline
  Hazy I -> netH -> J, T, A
  J, T -> conditional diffusion netG -> output
  Training: diffusion loss only
  netH: pretrained / mostly fixed

Right: PPDM / this work
  Hazy I -> finetuned netH -> J, T, A, I_hat
  Depth + beta -> T_gt -> L_T
  I_hat vs I -> L_ASM
  J, T -> conditional diffusion netG -> output
  Training: L_diff + 0.01 L_T + 0.05 L_ASM
  netG + selected netH modules jointly optimized on ColdFog
```

也可以做一張「baseline 主幹 + 新增紅色分支」圖：保留原 DehazeDDPM 的 `netH -> condition -> diffusion` 黑色主幹，將 `depth/beta -> loss_t`、`I_hat -> loss_asm`、`netH finetune` 用紅/橙色虛線標出。這張會更適合作為方法論首圖，因為讀者能一眼看出「不是重畫一個完全不同模型，而是在 DehazeDDPM 兩階段架構上加入物理監督與 ColdFog 適配」。

### 9.4 報告可用表述

英文：

> The proposed PPDM keeps the two-stage conditional diffusion backbone of DehazeDDPM, where netH predicts physical priors and netG performs conditional diffusion denoising. Our modification lies in adapting the framework to ColdFog and introducing physical supervision for netH. Specifically, the predicted transmission map is constrained by depth- and beta-derived transmission, and the predicted dehazed image, transmission, and atmospheric light are required to reconstruct the hazy input through the atmospheric scattering model.

中文：

> PPDM 保留了 DehazeDDPM 原有的兩階段條件擴散主幹，即先由 netH 估計物理先驗，再由 netG 執行條件擴散去噪。本工作的主要改動在於將該框架適配到 ColdFog 場景，並對 netH 的物理中間量加入顯式監督：一方面利用深度與散射係數構造透射率目標約束 `T`，另一方面利用大氣散射模型約束 `J/T/A` 能夠重建輸入有霧圖。

## 10. Source Map

| 實作點 | 文件 |
| --- | --- |
| 訓練入口 | `trainColdFogNetHPhysical.sh` |
| 訓練主循環 | `sr.py` |
| 推理入口 | `testColdFogFinetune_physical_ddim20.sh`, `infer.py` |
| `netG + netH` 包裝 | `model/model.py` |
| `netH/MPRfusion` 結構與 `J/T/A/I_hat` | `model/networkHelper.py` |
| SR3/DDPM diffusion 與 DDIM/DPM-Solver++ | `model/sr3_modules/diffusion.py`, `model/sr3_modules/dpm_solver_pp.py` |
| UNet denoiser | `model/sr3_modules/unet.py` |
| network factory | `model/networks.py` |
| metadata/depth/beta 資料流 | `data/util.py`, `data/LRHR_dataset.py`, `data/__init__.py` |
| physical 訓練配置 | `config/Dehaze_ColdFog_finetune_netH_physical.json` |
| physical 可視化保存 | `core/metrics.py` |
| 既有 loss 詳解 | `docs/ColdFog_netH_physical_losses.md` |

## 11. Caption Draft

英文版：

> Overview of the proposed PPDM framework. Given a hazy input image, the physical-prior estimator netH predicts a preliminary dehazed image, a transmission map, and atmospheric light. The predicted transmission is supervised by depth- and beta-derived physical transmission, while the predicted physical components reconstruct the hazy input through the atmospheric scattering model. The preliminary dehazed image and transmission map are concatenated as conditions for the diffusion denoiser netG, which progressively recovers the final haze-free image.

中文版：

> PPDM 方法總覽。有霧輸入首先經由物理先驗估計器 netH 預測初步去霧圖、透射率圖與大氣光；其中透射率受到由深度與散射係數計算得到的物理目標監督，三個物理分量則通過大氣散射模型重建有霧輸入。隨後，初步去霧圖與透射率圖被拼接為條件，引導擴散去噪器 netG 逐步恢復最終清晰圖像。
