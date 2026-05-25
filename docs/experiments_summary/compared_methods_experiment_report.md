# Compared Methods and Experimental Configurations Report

Generated on 2026-05-25 for the thesis Chapter 5 compared-methods / configuration section.

This report is based on read-only exploration of `config/`, `experiments/`, logs, checkpoints, scripts, and the model code. The only file written by this pass is this Markdown report.

## Executive Summary

The repository currently supports a DehazeDDPM-based experimental line for ColdFog adaptation. The reproducible and code-backed methods are:

1. DehazeDDPM pretrained baselines on DENSE/NH checkpoints.
2. ColdFog finetuning with diffusion-only adaptation.
3. ColdFog finetuning with joint `netH` adaptation.
4. ColdFog finetuning with joint `netH` adaptation plus physical consistency losses.
5. Inference sampler variants: full DDPM, DDIM, and DPM-Solver++.

DCP, DehazeNet, AOD-Net, and PPDM are not implemented in this repository. They should not be presented as locally reproduced methods unless external implementations/results are added later. They can remain as planned/external baselines or be removed from the final quantitative table.

The strongest completed training run is the physical-loss ColdFog run:

- `experiments/Dehaze_ColdFog_finetune_netH_physical_v1_260508_123402`
- best validation PSNR: `20.322` at `I75000_E268`
- final validation PSNR: `20.261` at `I100000_E358`

However, no verified ColdFog test-set inference result was found for this physical run. This is the highest-priority missing experiment before writing the final main quantitative conclusion.

## Recommended Thesis Framing

For Section 5.1.3, the cleanest framing is not "many unrelated baselines were all reproduced", but:

> The comparison focuses on progressively stronger DehazeDDPM-based ColdFog adaptation settings, with pretrained DehazeDDPM checkpoints used to quantify the domain gap. Traditional and CNN baselines such as DCP, DehazeNet, and AOD-Net are treated as candidate external baselines and should only enter the final quantitative table if their inference is completed under the same ColdFog test protocol.

The method table can be split into two groups:

1. **Verified local methods**: foggy input, DehazeDDPM pretrained DENSE/NH if tested, diffusion-only finetune, diffusion+netH finetune, diffusion+netH+physical-loss finetune.
2. **Candidate external baselines / planned methods**: DCP, DehazeNet, AOD-Net, PPDM.

Sampler variants such as DDIM and DPM-Solver++ should be described as inference configurations or efficiency ablations, not as separate dehazing methods.

## Method Evidence Chain

| Method or component | Local status | Evidence |
| --- | --- | --- |
| DehazeDDPM / SR3 diffusion | Implemented | `README.md`; `model/networks.py` selects `sr3`/`ddpm`; `model/model.py` defines wrapper class `DDPM`; `model/sr3_modules/diffusion.py` implements diffusion training and sampling. |
| First-stage `netH` / PreNet | Implemented | `model/networkHelper.py` defines `MPRfusion`; logs report `Network H structure: MPRfusion, with parameters: 2,779,939`. |
| ColdFog diffusion-only finetune | Implemented and completed in later run | `config/Dehaze_ColdFog_finetune.json`, `config/Dehaze_ColdFog_finetune_resume.json`, `experiments/Dehaze_ColdFog_finetune_only_diffusion_260417_152053`. |
| ColdFog diffusion + `netH` finetune | Implemented and completed | `config/Dehaze_ColdFog_finetune_netH.json`, `experiments/Dehaze_ColdFog_finetune_netH_260406_223953`. |
| ColdFog physical consistency losses | Implemented and completed for validation | `config/Dehaze_ColdFog_finetune_netH_physical.json`, `model/model.py`, `docs/ColdFog_netH_physical_losses.md`, `experiments/Dehaze_ColdFog_finetune_netH_physical_v1_260508_123402`. |
| DDIM | Implemented | `model/sr3_modules/diffusion.py` has `ddim_sample_loop`; `config/test_ColdFog_finetune_ddim.json` sets `sampler: ddim`, `sample_steps: 100`, `ddim_eta: 0.0`. |
| DPM-Solver++ | Implemented | `model/sr3_modules/dpm_solver_pp.py`; `model/sr3_modules/diffusion.py` has `dpm_solver_pp_sample_loop`; `config/test_ColdFog_finetune_dpm_solver_pp.json` sets `sampler: dpm_solver_pp`. |
| DCP | Not implemented | No executable local code/config/entry point was found. |
| DehazeNet | Not implemented | No executable local code/config/entry point was found. |
| AOD-Net | Not implemented | No executable local code/config/entry point was found. |
| PPDM | Not implemented | No executable local code/config/entry point was found. |

## Configuration Inventory

### Training Configurations

| Config | Role | Data | Initialization | Key settings | Status |
| --- | --- | --- | --- | --- | --- |
| `config/Dehaze_DENSE.json` | Original DENSE training baseline | `./data/Dense_Haze/*` | diffusion from scratch; `DENSE_net_g_120000.pth` for `netH` | `lr=1e-4`, `n_iter=2000000`, DDPM 2000 steps | Baseline config; old absolute `resume_stateH` path should be checked before rerun. |
| `config/Dehaze_NH.json` | Original NH training baseline | `./data/NH-HAZE/*` | diffusion from scratch; `NH_net_g_80000.pth` for `netH` | `lr=1e-4`, `n_iter=2000000`, DDPM 2000 steps | Baseline config; old absolute `resume_stateH` path should be checked before rerun. |
| `config/Dehaze_ColdFog_finetune.json` | ColdFog diffusion-only finetune | ColdFog train/val | `Diffusion_trained_pth/DENSE_I130000_E2600`; frozen DENSE `netH` | `batch_size=3`, `lr=5e-5`, `n_iter=100000`, DDPM 2000 | Initial config; early run was interrupted, later resume run completed. |
| `config/Dehaze_ColdFog_finetune_resume.json` | Resume for diffusion-only run | ColdFog train/val | resumes `experiments/Dehaze_ColdFog_finetune_only_diffusion_260417_152053/checkpoint/I85000_E304` | same as diffusion-only | Resume config, not a separate method. |
| `config/Dehaze_ColdFog_finetune_netH.json` | ColdFog diffusion + `netH` finetune | ColdFog train/val | DENSE diffusion + DENSE `netH` | `finetune_netH=true`, `lr=5e-5`, `lr_netH=1e-5`, `n_iter=100000` | Completed. |
| `config/Dehaze_ColdFog_finetune_netH_physical.json` | ColdFog diffusion + `netH` + physical losses | ColdFog train/val with metadata | best `netH` run at `I90000_E322`; `resume_stateH_finetune` loaded | `lambda_t=0.01`, `lambda_asm=0.05`, validation sampler `ddim`, `sample_steps=20` | Completed validation run; final test inference missing. |

### Test / Inference Configurations

| Config | Role | Checkpoint | Sampler | Status notes |
| --- | --- | --- | --- | --- |
| `config/test_DENSE.json` | DENSE pretrained test on DENSE test | `DENSE_I130000_E2600` | DDPM 2000 | Dataset len 5; not ColdFog. |
| `config/test_NH.json` | NH pretrained test on NH test | `NH_I230000_E4600` | DDPM 2000 | Dataset len 5; not ColdFog. |
| `config/test_DENSE_diy.json` | DENSE pretrained zero-shot on ColdFog test | `DENSE_I130000_E2600` | DDPM 2000 | Completed in `experiments/test/pretrain_model_domain_gap`. |
| `config/test_NH_diy.json` | NH pretrained zero-shot on ColdFog test | `NH_I230000_E4600` | DDPM 2000 | Config exists, but no verified ColdFog test run found. |
| `config/test_ColdFog_finetune.json` | diffusion-only ColdFog test | old `I15000_E54` | DDPM 2000 | Existing test uses early interrupted run, not the completed diffusion-only run. |
| `config/test_ColdFog_finetune_netH.json` | diffusion + `netH` ColdFog test | `I90000_E322` + `I90000_E322_netH.pth` | DDPM 2000 | Completed. |
| `config/test_ColdFog_finetune_ddim.json` | `netH` model with DDIM | `I90000_E322` + `netH` | DDIM 100, eta 0 | Completed; seed42 runs are reproducible. |
| `config/test_ColdFog_finetune_dpm_solver_pp.json` | `netH` model with DPM-Solver++ | `I90000_E322` + `netH` | DPM-Solver++ 200 | Completed for several step counts. |
| `config/test_ColdFog_finetune_netH_physical_ddim20.json` | physical model test candidate | currently points to non-existing no-`_v1` path and `I25000_E90` | filename says DDIM20, current content says `sample_steps=100` | Dirty/uncommitted config; do not use as final evidence until corrected and rerun. |

## Completed Training Runs

| Experiment | Training status | Best validation | Final validation | Checkpoints | Interpretation |
| --- | --- | --- | --- | --- | --- |
| `experiments/Dehaze_ColdFog_finetune_260405_120026` | Interrupted / half-complete | PSNR `14.673` at `I15000_E54` | no final; log stops after `iter 20000` training line | `I5000`, `I10000`, `I15000` only | Historical early diffusion-only run. Do not use as final method. |
| `experiments/Dehaze_ColdFog_finetune_only_diffusion_260417_152053` | Completed after resume from `I85000_E304` | PSNR `17.879` at `I85000_E304` | PSNR `16.930` at `I100000_E358` | full 5k interval checkpoints to `I100000_E358` | Completed diffusion-only ablation, but corresponding final test run was not found. |
| `experiments/Dehaze_ColdFog_finetune_netH_260406_223953` | Completed | PSNR `19.580` at `I90000_E322` | PSNR `19.059` at `I100000_E358` | `gen`, `netH`, `opt` checkpoints to `I100000_E358` | Strong completed ablation; current best verified test results use `I90000_E322`. |
| `experiments/Dehaze_ColdFog_finetune_netH_physical_v1_260508_123402` | Completed | PSNR `20.322` at `I75000_E268` | PSNR `20.261` at `I100000_E358` | `gen`, `netH`, `opt` checkpoints to at least `I100000_E358` | Strongest validation result and likely proposed/PPDM-style method; needs final test inference. |

Important path note: the physical experiment directory on disk is `Dehaze_ColdFog_finetune_netH_physical_v1_260508_123402`, but the log text internally records paths without `_v1`. The actual files are under the `_v1` directory.

## Completed ColdFog Test Runs Found

These are the currently safest numbers for a provisional table because they use ColdFog test data and have inference logs plus output images.

| Candidate row | Checkpoint / setting | Test-set PSNR | Evidence path | Use in final table? |
| --- | --- | ---: | --- | --- |
| DehazeDDPM-DENSE pretrained zero-shot | `DENSE_I130000_E2600` | `14.319` | `experiments/test/pretrain_model_domain_gap/Dehaze_test_DENSE_diy_260418_100213/logs/train.log` | Yes, as domain-gap baseline. |
| ColdFog diffusion-only, early incomplete checkpoint | `I15000_E54`, DDPM | `15.122` | `experiments/test/sampler_ddpm/Dehaze_ColdFog_finetune_test_only_diffusion_260417_180356/logs/train.log` | Only as historical note; not final ablation. |
| ColdFog diffusion + `netH` | `I90000_E322`, DDPM | `18.275` | `experiments/test/sampler_ddpm/Dehaze_ColdFog_finetune_test_netH_260418_004817/logs/train.log` | Yes. |
| ColdFog diffusion + `netH`, DDIM seed42 | `I90000_E322`, DDIM 100 | `18.730` | `experiments/test/sampler_ddim/with_seed/Dehaze_ColdFog_finetune_test_ddim_netH_seed42_260505_113113/logs/train.log` | Yes, but as sampler/efficiency result, not a separate method. |
| ColdFog diffusion + `netH`, DDIM seed42 2-GPU | `I90000_E322`, DDIM 100 | `18.730` | `experiments/test/sampler_ddim/with_seed/Dehaze_ColdFog_finetune_test_ddim_netH_seed42_2gpu_260505_130757/logs/train.log` | Confirms deterministic sharded inference. |
| ColdFog diffusion + `netH`, DPM-Solver++ 50 | `I90000_E322`, 50 steps | `16.095` | `experiments/test/sampler_dpm_solver_pp/...sample50.../logs/train.log` | Sampler ablation only. |
| ColdFog diffusion + `netH`, DPM-Solver++ 100 | `I90000_E322`, 100 steps | `17.630` | `experiments/test/sampler_dpm_solver_pp/...sample100.../logs/train.log` | Sampler ablation only. |
| ColdFog diffusion + `netH`, DPM-Solver++ 150 | `I90000_E322`, 150 steps | `17.407` | `experiments/test/sampler_dpm_solver_pp/...sample150.../logs/train.log` | Sampler ablation only. |
| ColdFog diffusion + `netH`, DPM-Solver++ 200 | `I90000_E322`, 200 steps | `17.680` | `experiments/test/sampler_dpm_solver_pp/...sample200.../logs/train.log` | Sampler ablation only. |

The no-seed DDIM result is not recommended for the final table: one run reports `19.262`, while the no-seed 2-GPU run reports `18.438`. The seed42 DDIM result is more defensible because single-GPU and 2-GPU agree.

## Suggested Table 5.1 Candidate Content

| Method | Configuration status | Purpose in Chapter 5 |
| --- | --- | --- |
| Foggy input | Always available; compute PSNR/SSIM directly from hazy input and GT | No-processing baseline for judging whether dehazing improves reconstruction. |
| DCP | Not implemented locally; external baseline only | Traditional physical-prior baseline, include only if external run is completed. |
| DehazeNet / AOD-Net | Not implemented locally; choose one only if external run is completed | CNN-based learning baseline. |
| DehazeDDPM-DENSE pretrained | Configured and tested on ColdFog (`PSNR=14.319`) | Same-family diffusion baseline for domain-gap verification. |
| DehazeDDPM-NH pretrained | Config exists, no verified ColdFog test run found | Optional same-family domain-gap baseline; run or remove. |
| ColdFog diffusion-only | Training completed, but final completed-checkpoint test missing | Ablation of adapting only the diffusion stage. |
| ColdFog diffusion + `netH` | Completed and tested (`I90000_E322`) | Ablation showing the benefit of adapting the first-stage condition network. |
| ColdFog diffusion + `netH` + physical losses | Training completed and strongest validation; final test missing | Proposed/PPDM-style variant testing cold-storage adaptation and physical consistency. |
| DDIM / DPM-Solver++ | Tested sampler variants for `netH` model | Efficiency and reduced-step inference analysis, not separate dehazing baselines. |

## What Can Be Written Now

The thesis can safely state:

- The local experimental framework is based on DehazeDDPM with a two-stage design: `netH` first predicts a dehazed/structural condition and transmission-related information, then an SR3/DDPM conditional diffusion model restores the final clean image.
- All ColdFog finetuning configurations use a linear beta schedule with `n_timestep=2000`, `linear_start=1e-6`, and `linear_end=1e-2`.
- ColdFog finetuning uses images resized/cropped to `448 x 576`, batch size `3`, diffusion learning rate `5e-5`, and validation/checkpoint frequency `5000`.
- Joint `netH` finetuning uses `finetune_netH=true` and `lr_netH=1e-5`.
- Physical-loss finetuning adds two auxiliary terms:
  - `loss_t = L1(out_T, exp(-beta * depth))`
  - `loss_asm = L1(out_I, hazy_input)`
  - total training loss: `l_pix + 0.01 * loss_t + 0.05 * loss_asm`
- In the physical version, metadata supplies `depth` and `beta` during training/validation. The ColdFog test configuration currently does not include metadata, so physical losses should not be claimed on the test set unless metadata is added there.

## What Should Not Be Claimed Yet

Avoid these statements until more evidence is added:

- Do not claim DCP, DehazeNet, AOD-Net, or PPDM were reproduced in this repository.
- Do not include DCP/DehazeNet/AOD-Net/PPDM in the final quantitative table unless their ColdFog test outputs are produced under the same evaluation protocol.
- Do not claim the physical-loss method has a final ColdFog test-set PSNR/SSIM yet; only validation PSNR was found.
- Do not use `test_ColdFog_finetune_netH_physical_ddim20.json` as final evidence in its current state because it is dirty/uncommitted, points to a no-`_v1` physical experiment path that is not on disk, and currently sets `sample_steps=100` despite the filename saying `ddim20`.
- Do not treat DDIM or DPM-Solver++ as separate trained dehazing methods. They are inference samplers.

## Best TODO Experiments

### Priority 0: Final Main Quantitative Table

These are the highest-value experiments to complete before writing final conclusions.

| Priority | TODO | Why it matters | Suggested setting |
| --- | --- | --- | --- |
| P0-1 | Run final ColdFog test inference for the physical-loss model | This is likely the proposed method, but test-set metrics are missing | Use `experiments/Dehaze_ColdFog_finetune_netH_physical_v1_260508_123402/checkpoint/I75000_E268` as best-val checkpoint, and optionally `I100000_E358` as final checkpoint. |
| P0-2 | Run final ColdFog test inference for the completed diffusion-only model | Existing test uses old incomplete `I15000_E54`; unfair against `netH` and physical runs | Use `experiments/Dehaze_ColdFog_finetune_only_diffusion_260417_152053/checkpoint/I85000_E304` as best-val checkpoint, and optionally `I100000_E358`. |
| P0-3 | Compute Foggy input PSNR/SSIM on the same 80-image ColdFog test set | Needed for the no-processing baseline row | Directly compare `/data/dehazeddpm_test/hazy_test` with `/data/dehazeddpm_test/gt_test`. |
| P0-4 | Compute SSIM for all retained methods | Current inference logs mainly provide PSNR; thesis tables usually need PSNR/SSIM | Use existing saved `*_out.png` and GT images, or extend a small evaluation script using `core.metrics.calculate_ssim`. |
| P0-5 | Normalize final table to one evaluation protocol | Prevents mixing validation PSNR, old interrupted runs, and test PSNR | Use the same ColdFog test split, same image size, same metric code, and fixed seed if sampler is stochastic. |

Recommended final main table rows after P0:

1. Foggy input.
2. DehazeDDPM-DENSE pretrained zero-shot.
3. DehazeDDPM-NH pretrained zero-shot, only if rerun is completed; otherwise omit.
4. ColdFog diffusion-only, completed best-val checkpoint.
5. ColdFog diffusion + `netH`, `I90000_E322`.
6. ColdFog diffusion + `netH` + physical losses, best-val or final checkpoint.

### Priority 1: External Baselines

These improve breadth but are not necessary if the chapter is framed as DehazeDDPM adaptation.

| Priority | TODO | Recommendation |
| --- | --- | --- |
| P1-1 | DCP on ColdFog test | Add only if there is time. It is a useful traditional prior baseline, but requires external implementation or a new script. |
| P1-2 | Choose either DehazeNet or AOD-Net | Do not try to include both unless checkpoints and inference code are already available. AOD-Net is usually easier to present as a compact CNN baseline. |
| P1-3 | PPDM | Include only if external code/checkpoint can be run reliably. Current repo has no PPDM implementation. |

If time is short, remove these from final quantitative conclusions and leave them as "candidate baselines not included due to incomplete verified runs".

### Priority 2: Efficiency / Sampler Analysis

These support Section 5.7.

| Priority | TODO | Why |
| --- | --- | --- |
| P2-1 | Measure seconds/image for DDPM 2000, DDIM 100, DPM-Solver++ 50/100/150/200 | Existing logs show start/end timestamps, but a controlled timing table would be cleaner. |
| P2-2 | Rerun DDIM with fixed seed only | No-seed DDIM results vary; fixed seed is defensible. |
| P2-3 | Optionally test DDIM 20/50/100/200 on the same checkpoint | This would make the "DDIM sampling" section more complete than only DDIM 100. |

### Priority 3: Physical Consistency Analysis

These support Section 5.6 rather than the main dehazing table.

| Priority | TODO | Why |
| --- | --- | --- |
| P3-1 | Evaluate physical model on a split with metadata/depth/beta | Test-set config lacks metadata, so physical losses on the current ColdFog test set are not meaningful. |
| P3-2 | Save representative `out_T`, `out_A`, `out_I`, `output`, `stage1_output` panels | These visualizations explain how the physical prior changes `netH`, not just PSNR. |
| P3-3 | Compare `loss_t`, `loss_asm`, and `loss_physical_total` over training validation logs | The physical run already logs these values every 5000 steps. |

## Suggested LaTeX-Level Text Blocks

### Compared Methods Paragraph

The compared methods are organized according to their role in the experimental design. First, the hazy input is retained as a no-processing baseline. Second, pretrained DehazeDDPM checkpoints trained on public dehazing datasets are evaluated directly on ColdFog images to quantify the cross-domain degradation. Third, several ColdFog-adapted DehazeDDPM variants are compared: adapting only the diffusion model, jointly adapting the diffusion model and the first-stage `netH`, and adding physical consistency losses based on transmission and atmospheric scattering constraints. Inference samplers such as DDPM, DDIM, and DPM-Solver++ are analyzed separately as efficiency configurations rather than as independent dehazing models.

### Physical Loss Paragraph

The physical variant extends the joint `netH` finetuning configuration by introducing two auxiliary constraints. The first supervises the predicted transmission map with `t(x)=exp(-beta d(x))`, where depth and scattering coefficient are read from the ColdFog metadata. The second reconstructs the hazy input through the atmospheric scattering model `I = T J + (1 - T) A`, where `J`, `T`, and `A` are predicted by `netH`. The final training objective is the diffusion loss plus `0.01 loss_t + 0.05 loss_asm`.

### Table Caption Draft

Candidate compared methods and their roles in the experimental design. The final quantitative table should include only methods with completed and verified ColdFog test-set results under the same evaluation protocol; methods without local implementations or completed runs should be reported as planned or excluded from the final quantitative conclusion.

## Final Inclusion Recommendation

For the final thesis quantitative table, include only completed and verified ColdFog test-set runs. Based on current evidence, the table is not yet ready for final claims because the proposed physical-loss model and the completed diffusion-only ablation still need test-set inference. Once those are added, the chapter can make a much stronger and cleaner claim:

- pretrained DehazeDDPM shows a clear domain gap on ColdFog,
- ColdFog finetuning improves reconstruction,
- joint `netH` adaptation improves over diffusion-only adaptation,
- physical consistency gives the strongest validation behavior and should be tested as the proposed final method.
