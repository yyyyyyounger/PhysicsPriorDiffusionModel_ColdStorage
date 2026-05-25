import argparse
import copy
import gc
import json
import logging
import os

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset
from tensorboardX import SummaryWriter

import core.logger as Logger
import core.metrics as Metrics
import core.seed as Seed
import data as Data
import model as Model
from core.wandb_logger import WandbLogger


def _opt_to_plain(opt):
    """Recursively convert NoneDict / OrderedDict to plain dict for multiprocessing."""
    if isinstance(opt, dict):
        return {k: _opt_to_plain(v) for k, v in opt.items()}
    if isinstance(opt, list):
        return [_opt_to_plain(v) for v in opt]
    return opt


def run_inference_worker(opt, rank, world_size, wandb_logger=None,
                         log_full_opt=True, setup_worker_log=False):
    """
    Run validation inference on a shard of the val dataset (stride sharding by rank).

    Args:
        opt: NoneDict-like options (must include path, datasets, model, enable_wandb, log_infer).
        rank: Worker rank in [0, world_size).
        world_size: Total workers; use 1 for single-process inference.
        wandb_logger: Optional WandbLogger (single-GPU only recommended).
        log_full_opt: If True, log full opt dict once (single-GPU default).
        setup_worker_log: If True, write worker-only log file infer_rank{rank}.log.

    Returns:
        (psnr_sum, count): Weighted PSNR sum and number of samples processed.
    """
    logger = logging.getLogger('base')
    if setup_worker_log:
        Logger.setup_logger(
            None, opt['path']['log'], 'infer_rank{}'.format(rank),
            level=logging.INFO, screen=False)

    if log_full_opt:
        logger.info(Logger.dict2str(opt))

    val_loader = None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            full_len = len(val_set)
            indices = list(range(rank, full_len, world_size))
            subset = Subset(val_set, indices)
            if world_size > 1:
                # Avoid nested multiprocessing: each GPU worker is already a
                # spawned process, so DataLoader workers can segfault at exit.
                val_loader = DataLoader(
                    subset, batch_size=1, shuffle=False,
                    num_workers=0, pin_memory=False)
            else:
                val_loader = Data.create_dataloader(
                    subset, dataset_opt, phase,
                    manual_seed=opt['manual_seed'])
            prefix = ('Worker rank {}: '.format(rank)) if world_size > 1 else ''
            logger.info(
                '{}val shard {} / {} samples (world_size={}).'.format(
                    prefix, len(indices), full_len, world_size))
            break

    if val_loader is None:
        raise RuntimeError('No val dataset in opt[datasets].')

    logger.info('Initial Dataset Finished')

    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['val'], schedule_phase='val')

    prefix = ('Worker rank {}: '.format(rank)) if world_size > 1 else ''
    logger.info('{}Begin Model Inference.'.format(prefix))

    current_step = 0
    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)

    psnr_sum = 0.0
    count = 0

    manual_seed = opt['manual_seed']
    if manual_seed is None:
        manual_seed = 42

    for val_data in val_loader:
        idx_tensor = val_data['Index']
        sample_idx_1based = int(idx_tensor.detach().cpu().view(-1)[0].item()) + 1
        Seed.set_sample_seed(int(manual_seed) + sample_idx_1based)

        diffusion.feed_data(val_data)
        diffusion.test(continous=True)
        visuals = diffusion.get_current_visuals()

        hr_img = Metrics.tensor2img(visuals['HR'])
        lr_img = Metrics.tensor2img(visuals['LR'])
        sr_img_mode = 'grid'
        if sr_img_mode == 'single':
            sr_img = visuals['Out']
            sample_num = sr_img.shape[0]
            for it in range(0, sample_num):
                Metrics.save_img(
                    Metrics.tensor2img(sr_img[it]),
                    '{}/{}_{}_out_{}.png'.format(
                        result_path, current_step, sample_idx_1based, it))
        else:
            Metrics.save_img(
                Metrics.tensor2img(visuals['Out'][-1]),
                '{}/{}_{}_out.png'.format(result_path, current_step, sample_idx_1based))
        Metrics.save_img(
            lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, sample_idx_1based))
        Metrics.save_physical_visuals(
            visuals, result_path,
            '{}_{}'.format(current_step, sample_idx_1based))

        psnr_sum += Metrics.calculate_psnr(
            Metrics.tensor2img(visuals['Out'][-1]), hr_img)
        count += 1

        if wandb_logger is not None and opt.get('log_infer'):
            wandb_logger.log_eval_data(
                lr_img, Metrics.tensor2img(visuals['Out'][-1]), hr_img)

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['train'], schedule_phase='train')

    if world_size > 1:
        logger.info(
            'Worker rank {}: local PSNR mean {:.4e} over {} samples.'.format(
                rank, psnr_sum / count if count else 0.0, count))

    if wandb_logger is not None and opt.get('log_infer'):
        wandb_logger.log_eval_table(commit=True)

    del diffusion
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()

    return psnr_sum, count


def _infer_spawn_fn(rank, opt_plain, metrics_dir):
    """torch.multiprocessing.spawn entry: one CUDA device per rank."""
    torch.cuda.set_device(rank)
    opt = Logger.dict_to_nonedict(copy.deepcopy(opt_plain))
    opt['local_rank'] = rank
    opt['distributed'] = False

    Seed.set_seed(
        opt['manual_seed'],
        deterministic=bool(opt.get('manual_seed_deterministic')),
    )

    psnr_sum, count = run_inference_worker(
        opt, rank=rank, world_size=len(opt_plain['gpu_ids']),
        wandb_logger=None,
        log_full_opt=False,
        setup_worker_log=True)
    metrics_path = os.path.join(
        metrics_dir, 'infer_rank{}_metrics.json'.format(rank))
    with open(metrics_path, 'w') as f:
        json.dump({'psnr_sum': float(psnr_sum), 'count': int(count)}, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', type=str,
        default="/home/yuhu/experiments/Dehaze_NH_221024_163156/checkpoint/I80000_E1455",
        help='JSON file for configuration')
    parser.add_argument(
        '-p', '--phase', type=str, choices=['val'],
        help='val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_infer', action='store_true')

    args = parser.parse_args()
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)

    Seed.set_seed(
        opt['manual_seed'],
        deterministic=bool(opt.get('manual_seed_deterministic')),
    )

    gpu_ids = opt['gpu_ids']
    num_gpus = len(gpu_ids) if gpu_ids is not None else 0

    Logger.setup_logger(
        None, opt['path']['log'], 'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')

    logger.info('Config file: {}'.format(os.path.abspath(args.config)))
    logger.info(Logger.dict2str(opt))
    config_snapshot = os.path.join(opt['path']['log'], 'config_resolved.json')
    with open(config_snapshot, 'w', encoding='utf-8') as f:
        json.dump(_opt_to_plain(opt), f, indent=2, ensure_ascii=False)
    logger.info('Resolved config saved to {}'.format(config_snapshot))

    if num_gpus <= 1:
        wandb_logger = WandbLogger(opt) if opt['enable_wandb'] else None
        tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

        psnr_sum, count = run_inference_worker(
            opt, rank=0, world_size=1,
            wandb_logger=wandb_logger,
            log_full_opt=False,
            setup_worker_log=False)

        avg_psnr = psnr_sum / count if count else 0.0
        logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
        tb_logger.close()
        return

    if num_gpus > 1 and not torch.cuda.is_available():
        raise RuntimeError('Multiple GPUs requested but CUDA is not available.')

    logger.info(
        'Multi-GPU inference: {} GPUs (sample sharding). '
        'Per-GPU logs: infer_rank*.log'.format(num_gpus))

    if opt['enable_wandb'] and opt.get('log_infer'):
        logger.warning(
            'W&B log_infer is disabled for multi-GPU spawn runs '
            '(use single GPU if you need eval tables).')

    opt_plain = _opt_to_plain(opt)
    metrics_dir = opt['path']['log']
    mp.spawn(
        _infer_spawn_fn,
        args=(opt_plain, metrics_dir),
        nprocs=num_gpus,
        join=True)

    total_psnr_sum = 0.0
    total_count = 0
    for rank in range(num_gpus):
        metrics_path = os.path.join(
            metrics_dir, 'infer_rank{}_metrics.json'.format(rank))
        with open(metrics_path, 'r') as f:
            part_metrics = json.load(f)
        total_psnr_sum += float(part_metrics['psnr_sum'])
        total_count += int(part_metrics['count'])

    avg_psnr = total_psnr_sum / total_count if total_count else 0.0
    logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))


if __name__ == "__main__":
    main()
