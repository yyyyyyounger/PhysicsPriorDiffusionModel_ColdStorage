# PPDM - Physics-Prior Diffusion Model for Image Dehazing in Cold Storage Scenes

中文版: [README_CN.md](README_CN.md).

**PPDM** (Physics-Prior Diffusion Model) is a single-image dehazing model for **low-temperature fogging in cold storage scenes**. It uses a conditional diffusion dehazing backbone and injects **physics priors** including depth, transmission, and the atmospheric scattering model. The target problem is the short-range, high-density fog caused by water vapor condensation in cold storage environments.

General-purpose dehazing models have a clear domain gap in cold storage scenes: the fog is caused by low-temperature condensation rather than distant atmospheric haze, and it is denser with a different spatial distribution. PPDM addresses this with:

- **A physics-based synthetic cold-storage fog dataset**: [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) is used as the depth prior, and foggy images are synthesized with the atmospheric scattering model at Light / Medium / Heavy density levels.
- **Physics-prior injection with three-stage progressive training**: **finetune diffusion backbone netG only -> jointly finetune netG + condition network netH -> add physics losses for transmission and atmospheric scattering consistency**. The restored images are supervised by both reconstruction and physics constraints.
- **Efficient inference and evaluation**: DDIM and DPM-Solver++ fast samplers, multi-GPU inference, paper figure generation, and SAM-based downstream segmentation evaluation are included.

> The diffusion backbone and condition network are adapted from [DehazeDDPM](https://github.com/yuhuUSTC/DehazeDDPM). Names such as `netG` (diffusion backbone) and `netH` (condition / PreNet) follow that baseline.

<p align="center">
  <img src="docs/images/ppdm_whole.png" alt="PPDM architecture and physics losses" width="850"><br>
  <em>Overall PPDM architecture. Stage 1 estimates physics priors such as transmission t_hat and atmospheric light A_hat with netH. Stage 2 restores the clear image with conditional diffusion netG. Besides the original reconstruction loss L_base, training adds two physics constraints: <strong>transmission loss L_t</strong> and <strong>ASM reconstruction loss L_ASM</strong>. The total loss is L_total = lambda_ASM * L_ASM + lambda_t * L_t + L_base.</em>
</p>

---

## Repository Structure

| Path | Description |
| --- | --- |
| `sr.py` | Training entry point: `python sr.py --config <json>`. |
| `infer.py` | Inference / sampling entry point, with multi-GPU sample splitting. |
| `config/` | Experiment configs for training and testing. See the tables below for naming. |
| `model/` | PPDM model definitions: diffusion backbone netG (`sr3_modules`, `ddpm_modules`), condition network netH, and physics losses (`model.py`). |
| `core/` | Utilities for logging, metrics, W&B, seeds, and related helpers. |
| `data/` | Data loading (`LRHR_dataset.py`) and dataset construction scripts. |
| `data/dataset/` | Cold-storage dataset construction notes and scripts. See [data/dataset/README.md](data/dataset/README.md). |
| `Diffusion_trained_pth/` | Stage-2 diffusion checkpoints. This directory is ignored by git and must be obtained separately. |
| `pretrained_PreNet_pth/` | Stage-1 PreNet (`netH`) weights. |
| `experiments/` | Training outputs: logs, checkpoints, and TensorBoard logs. This directory is ignored by git. |
| `plot/` | Paper figure generation and analysis notebooks for Chapter 5. See [Figures and Analysis](#figures-and-analysis). |
| `sam_coldfog_test/` | SAM downstream segmentation evaluation subproject. See [sam_coldfog_test/README.md](sam_coldfog_test/README.md). |
| `*.sh` | Training and testing launch scripts for the experiments. |

---

## Code Sources and External Dependencies

The full PPDM workflow spans three codebases. Dataset construction uses depth estimation, and downstream evaluation uses SAM. These were originally run inside the official [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) and [Segment Anything](https://github.com/facebookresearch/segment-anything) repositories. For a clean submission, this repository only includes the key scripts directly related to PPDM, such as the synthesis scripts under `data/dataset/` and the evaluation scripts under `sam_coldfog_test/`.

To fully reproduce the project, you need to clone the two official repositories separately:

| Step | Code location | Documentation in this repository |
| --- | --- | --- |
| Dehazing training / inference (PPDM main code) | **This repository** | This README |
| Depth prior / fog synthesis | Clone [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) separately | [data/dataset/README.md](data/dataset/README.md) |
| SAM downstream segmentation evaluation | Clone [Segment Anything](https://github.com/facebookresearch/segment-anything) separately | [sam_coldfog_test/README.md](sam_coldfog_test/README.md) |

> If the two sub-READMEs contain local absolute paths such as `/mnt/newdisk/.../Depth-Anything-V2` or `/mnt/newdisk/.../segment-anything`, replace them with your own clone locations.

## Environment Setup

Because the three codebases use different dependencies, using **three separate conda environments** is recommended. The names below are only suggestions.

### 1. Main PPDM Dehazing Environment

Use this environment for training and inference of the dehazing model.

```bash
conda create -n ppdm python=3.10 -y
conda activate ppdm

pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

pip install tensorboardX wandb numpy opencv-python Pillow tqdm lmdb matplotlib
```

### 2. Depth Estimation Environment for Dataset Construction

Use this environment to run Depth Anything V2 for depth priors and cold-storage fog synthesis. Install the official dependencies **inside the Depth Anything V2 repository**:

```bash
conda create -n ppdm-depth python=3.10 -y
conda activate ppdm-depth

# In Depth-Anything-V2/
pip install -r requirements.txt
# If using metric depth, as this project does, also install:
pip install -r metric_depth/requirements.txt
```

### 3. SAM Evaluation Environment

Use this environment for the segmentation evaluation under `sam_coldfog_test/`. Install dependencies **inside the Segment Anything repository**:

```bash
conda create -n ppdm-sam python=3.10 -y
conda activate ppdm-sam

# In segment-anything/
pip install -e .
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

Run on a specific GPU from the main environment:

```bash
CUDA_VISIBLE_DEVICES=3 bash testDENSE.sh
```

---

## Data and Checkpoint Download

The PPDM finetuned checkpoints and the synthetic cold-storage fog dataset are stored in one Baidu Netdisk folder:

| Content | Target location |
| --- | --- |
| PPDM finetuned checkpoints | Extract to `Diffusion_trained_pth/` or `experiments/`, depending on the folder layout in the archive. For inference, set `path.resume_state` in the config to the checkpoint prefix without `_gen.pth`. |
| Synthetic cold-storage fog dataset | Extract under the project `data/` directory, then update `datasets.*.dataroot`, `finetune_root`, and `metadata_csv` in the configs according to the actual paths. |

**Baidu Netdisk:** [https://pan.baidu.com/s/1Uj3H0aOS8PqIfT_x7NAYNA?pwd=8w2i](https://pan.baidu.com/s/1Uj3H0aOS8PqIfT_x7NAYNA?pwd=8w2i)

You can also generate checkpoints through the [training workflow](#training), or rebuild the dataset by following [Build From Scratch](#build-from-scratch).

---

## Pretrained Weights

PPDM uses the DehazeDDPM pretrained weights as the finetuning starting point. These are only needed when reproducing the **baseline** or finetuning from scratch:

- Stage-1 PreNet (`netH`) weights are stored in `pretrained_PreNet_pth/`.
- Stage-2 diffusion backbone checkpoint: [Diffusion_trained_pth](https://drive.google.com/drive/folders/1I7sH6vb9oWOZeIVu6-xh9Xm5lnwdzHa7?usp=drive_link), provided by the DehazeDDPM authors.

### PPDM Pretrained Checkpoint

The PPDM checkpoint finetuned on cold-storage data is not included in the repository. Download it from [Data and Checkpoint Download](#data-and-checkpoint-download).

---

## Dataset

<p align="center">
  <img src="docs/images/ppdm_teaser.png" alt="Synthetic cold-storage fog dataset construction overview" width="850"><br>
  <em>Dataset construction overview. About 400 clear cold-storage images are processed by Depth Anything V2 for depth estimation, then Light / Medium / Heavy fog images are synthesized with the atmospheric scattering model for PPDM training and evaluation.</em>
</p>

### Download

Download the synthetic cold-storage fog dataset, including clear images, fog images at different density levels, splits, and flattened finetuning folders, from [Data and Checkpoint Download](#data-and-checkpoint-download).

### Build From Scratch

To rebuild the dataset from the original clear images, follow the complete workflow here:

**[data/dataset/README.md](data/dataset/README.md)**

The workflow covers clear image collection, Depth Anything V2 depth estimation, atmospheric-scattering-based Light / Medium / Heavy fog synthesis, split generation, and flattened finetuning data preparation.

Related scripts: `data/dataset/synthesize_fog.ipynb`, `data/dataset/split.ipynb`, `data/dataset/prepare_finetune_data.py`, and `data/prepare_data.py`. The depth estimation step must be run inside the Depth Anything V2 repository with the `ppdm-depth` environment.

> The `datasets.*.dataroot`, `finetune_root`, and `metadata_csv` fields in the configs currently point to local absolute paths such as `/mnt/newdisk/.../data/finetune`. Update these fields on a new machine.

---

## Training

The training entry point is `python sr.py --config <json>`. Run the following commands from the project root. Paths below are relative to the project root.

### Three Finetuning Settings

| Setting / launch script | Config | Description |
| --- | --- | --- |
| `trainColdFog.sh` | `Dehaze_ColdFog_finetune.json` | **Finetune diffusion netG only** while keeping netH frozen. |
| `trainColdFogNetH.sh` | `Dehaze_ColdFog_finetune_netH.json` | **Jointly finetune netG + netH** with `finetune_netH: true`; netH uses a smaller `lr_netH`, default `1e-5`. |
| `trainColdFogNetHPhysical.sh` | `Dehaze_ColdFog_finetune_netH_physical.json` | Finetune netG + netH **with physics losses**: transmission loss `lambda_t` and atmospheric scattering model loss `lambda_asm`. Total loss = `l_pix + lambda_t * loss_t + lambda_asm * loss_asm`. Validation uses DDIM with 20 steps. Requires `resume_stateH_finetune` for the finetuned netH weights and physics targets from `metadata_csv` / `finetune_root`. |

Resume scripts: `trainColdFog_resume.sh` and `trainColdFogNetH_resume.sh`. The latter resumes netH physical training up to `n_iter=200000`.

### Train From Scratch in a New Experiment Directory

- Do **not** add `reuse_experiments_root` under `path` in the config. The program will automatically create `experiments/<name>_<timestamp>/` and write logs, checkpoints, and TensorBoard logs there.
- **`resume_state`** should be the prefix of the stage-2 diffusion pretrained checkpoint, without `_gen.pth`, for example `./Diffusion_trained_pth/DENSE_I130000_E2600`. The code will load `..._gen.pth`.
- If there is no matching `..._opt.pth` for the prefix, meaning it is only a pretrained network weight, the code loads only model weights and starts the iteration counter from 0. This is expected.
- **`resume_stateH`** should be the Stage-1 PreNet (`netH`) weight path, as in the original project.
- **`resume_stateH_finetune`** is only used by the netH physical config. It points to a finetuned netH weight prefix ending in `..._netH.pth`, so the physical stage can continue from that netH.

### Resume Training in the Same Experiment Directory

To resume from the latest saved checkpoint after interruption, continue until `train.n_iter` such as 100000, and append to the existing **`train.log` / `val.log`** instead of overwriting them:

1. Set **`reuse_experiments_root`** under `path` to the full relative path of the existing experiment directory, for example:
   `experiments/Dehaze_ColdFog_finetune_only_diffusion_260417_152053`
2. Set **`resume_state`** to the checkpoint prefix inside that experiment's `checkpoint` directory, again without `_gen.pth`, for example:
   `experiments/Dehaze_ColdFog_finetune_only_diffusion_260417_152053/checkpoint/I85000_E304`
3. Both **`I85000_E304_gen.pth`** and **`I85000_E304_opt.pth`** must exist to restore the optimizer and the recorded **`iter` / `epoch`**. If only `*_gen.pth` exists, the behavior is the same as loading weights only and restarting from iteration 0.
4. **`train.n_iter`** is still the total iteration limit. After resuming, training continues from the iteration recorded in the checkpoint until it reaches `n_iter`.

**Note:** Checkpoint frequency is controlled by `train.save_checkpoint_freq`. If training stops between checkpoints, only the previous checkpoint exists on disk, and resuming will rerun the unsaved iterations. When reusing the same `tb_logger` directory, TensorBoard may contain multiple event files; they can usually be viewed together.

---

## Inference and Sampler Settings

### Run on This Project's Dataset

```bash
conda activate ppdm

CUDA_VISIBLE_DEVICES=3 python infer.py --config ./config/test_DENSE_diy.json
CUDA_VISIBLE_DEVICES=2 python infer.py --config ./config/test_NH_diy.json
```

Cold-storage finetuned model test configs and launch scripts:

| Sampler | Config | Script |
| --- | --- | --- |
| Full DDPM quality baseline, 2000 steps | `test_ColdFog_finetune.json` | `testColdFogFinetune.sh` |
| DDIM | `test_ColdFog_finetune_ddim.json` | `testColdFogFinetune_ddim.sh` |
| DPM-Solver++ | `test_ColdFog_finetune_dpm_solver_pp.json` | `testColdFogFinetune_dpm_solver_pp.sh` |
| netH model | `test_ColdFog_finetune_netH.json` | `testColdFogFinetune_netH.sh` |
| netH + physics, DDIM 20 steps | `test_ColdFog_finetune_netH_physical_ddim20.json` | `testColdFogFinetune_physical_ddim20.sh` |

### `beta_schedule.val` Sampler Settings

Sampler switching in `GaussianDiffusion.super_resolution()` is used only when **`model.which_model_G` is `"sr3"`**. The `ddpm` path still runs the full reverse chain and remains compatible with existing checkpoints.

The settings are placed under **`model.beta_schedule.val`** in the JSON, at the same level as `schedule`, `n_timestep`, `linear_start`, and `linear_end`. **Training** still uses only **`beta_schedule.train`**. Do not add `sampler` under the `train` block.

| Field | Description |
| --- | --- |
| **`sampler`** | Sampler name. If omitted, or if the key is missing after loading a train-only schedule such as when `sr.py` switches back after validation, the sampler resets to **`ddpm`**. Accepted strings are case-insensitive: **`ddpm`** for full T-step sampling, **`ddim`**, and **`dpm_solver_pp`**. **`dpm_solver++`** is also accepted and normalized internally. |
| **`sample_steps`** | Used only by `ddim` and `dpm_solver_pp`. It is the target number of skipped sampling steps from the full schedule whose length is **`n_timestep`**. Do **not** reduce **`val.n_timestep`** to simulate fast inference, because that breaks the alpha-bar discretization used in training. If omitted, **`ddim` defaults to 100** and **`dpm_solver_pp` defaults to 50**. **`ddpm`** ignores this field and always runs **`n_timestep`** steps. |
| **`ddim_eta`** | Used only by `ddim`. `eta = 0` gives deterministic DDIM; `eta > 0` injects randomness similar to DDPM. A stable default is **`0.0`**. |

**DDIM example, 100 steps:**

```json
"val": {
  "schedule": "linear",
  "n_timestep": 2000,
  "linear_start": 1e-6,
  "linear_end": 1e-2,
  "sampler": "ddim",
  "sample_steps": 100,
  "ddim_eta": 0.0
}
```

**DPM-Solver++ example, 50 steps:**

```json
"val": {
  "schedule": "linear",
  "n_timestep": 2000,
  "linear_start": 1e-6,
  "linear_end": 1e-2,
  "sampler": "dpm_solver_pp",
  "sample_steps": 50
}
```

**Sampler tuning suggestions:**

- **DDIM / DPM-Solver++ do not require retraining.** They only change the reverse sampling method during validation or inference and introduce no trainable parameters. Future training can keep the original `beta_schedule.train` settings.
- **Do not reduce `val.n_timestep` for speed.** Keep `n_timestep`, `linear_start`, and `linear_end` consistent with the training schedule, for example `2000 / 1e-6 / 1e-2` for this project's finetuning configs. Use only `sample_steps` for fewer inference steps.
- **Tune `sample_steps` first for DPM-Solver++.** The current code fixes `order=2`, `skip_type='time_uniform'`, and `clip_denoised=True`. If 50 steps produce noisy results, try `75`, `100`, `150`, and `200` in order. More steps usually improve stability at the cost of speed.
- **Start with deterministic DDIM.** `ddim_eta: 0.0` is usually more stable. If the image is already noisy, increasing `ddim_eta` is not recommended as the first fix, because `eta > 0` injects extra randomness. Compare `sample_steps: 100` and `200` first.
- **Recommended comparison protocol.** Use full DDPM (`ddpm`, 2000 steps) as the quality baseline, then compare `DPM-Solver++ 50/100/150` and `DDIM 100/200` with PSNR, SSIM, and visual quality. If high-step samplers still produce obvious noise, the issue is more likely from model weights, cold-storage data distribution, or Stage-1 `netH` condition quality rather than the sampler itself.

**Code entry points:** `sr.py` loads `beta_schedule.val` during validation, and `infer.py` also uses **`beta_schedule.val`** for inference.

### Multi-GPU Inference in `infer.py`

When **`gpu_ids` contains more than one GPU** in the config, for example `[0, 1]`, or when running **`python infer.py ... -gpu 0,1`**, the program launches one process per visible GPU with **`torch.multiprocessing.spawn`**. The validation set is split by interleaved sample indices, so rank `r` processes indices `r, r+W, r+2W, ...`. Each process loads a full model, binds to **`cuda:{local_rank}`**, and sets **`distributed` to False** to avoid wrapping another `DataParallel` layer inside each process.

- **Output filenames** use **`Index + 1`** from the dataset as the sequence number, matching the single-GPU convention where the k-th image maps to `{step}_{k+1}_*.png`. Multiple processes can write to the same `results` directory without overwriting each other.
- **Overall PSNR** is aggregated by the main process with sample-count weighting from all workers.
- **Logs** are written to **`logs/infer_rank{0,1,...}.log`** for each GPU. The main process still prints the total average PSNR to the terminal.
- **W&B** per-image `log_eval_data` / table logging is skipped in multi-GPU mode when **`log_infer`** is enabled, to avoid duplicate uploads from multiple processes. Use single-GPU inference if complete W&B inference records are needed.

Example:

```bash
bash testColdFogFinetune_ddim.sh -gpu 0,1
# Or set "gpu_ids": [0, 1] in JSON and run:
python infer.py --config config/test_ColdFog_finetune_ddim.json
```

---

## Figures and Analysis

The `plot/` directory contains paper figures and analysis notebooks for Chapter 5:

| Path | Description |
| --- | --- |
| `plot/plot_train_log.ipynb` | Plot training curves from `train.log` / `val.log`. |
| `plot/compare_checkpoints_testset.ipynb` | Compare metrics from different checkpoints on the test set. |
| `plot/ch5_main_results/` | Main result figures. |
| `plot/ch5_ablation/` | Ablation study figures. |
| `plot/ch5_inference/` | Inference / sampler comparison figures. |
| `plot/ch5_zero_shot_failure_cases/` | Zero-shot failure cases of the original model. |
| `plot/build_test_metadata.py`, `plot/test_metadata.json` | Test-set metadata construction. |

---

## Downstream Evaluation with SAM

Segment Anything (SAM) is used to automatically segment cold-storage dehazing results and quantify how dehazing helps downstream segmentation. For the workflow, GPU settings, and scripts, see:

**[sam_coldfog_test/README.md](sam_coldfog_test/README.md)**

This part uses the `ppdm-sam` environment and requires cloning the Segment Anything repository separately.

---

## Acknowledgements and Citation

The PPDM diffusion dehazing backbone is based on [DehazeDDPM](https://github.com/yuhuUSTC/DehazeDDPM). Physics priors use [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2), and downstream evaluation uses [Segment Anything](https://github.com/facebookresearch/segment-anything). We thank the authors of these projects.

<!-- If this repository is useful for your research, please cite the corresponding report:

> Physics-Prior Diffusion Model for Image Dehazing in Cold Storage Scenes (PPDM). -->
