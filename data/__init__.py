'''create dataset and dataloader'''
import logging
import random
from re import split

import numpy as np
import torch
import torch.utils.data


def _make_worker_init_fn(base_seed):
    """Per-worker seed offset so DataLoader workers are reproducible."""
    base_seed = int(base_seed)

    def _fn(worker_id):
        s = base_seed + int(worker_id)
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)

    return _fn


def create_dataloader(dataset, dataset_opt, phase, manual_seed=None):
    '''create dataloader '''
    generator = None
    worker_init_fn = None
    if manual_seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(manual_seed))
        if int(dataset_opt.get('num_workers', 0) or 0) > 0:
            worker_init_fn = _make_worker_init_fn(int(manual_seed))

    if phase == 'train':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=dataset_opt['use_shuffle'],
            num_workers=dataset_opt['num_workers'],
            pin_memory=True,
            generator=generator,
            worker_init_fn=worker_init_fn)
    elif phase == 'val':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            generator=generator,
            worker_init_fn=worker_init_fn)
    else:
        raise NotImplementedError(
            'Dataloader [{:s}] is not found.'.format(phase))


def create_dataset(dataset_opt, phase):
    '''create dataset'''
    from data.LRHR_dataset import LRHRDataset as D
    dataset = D(datarootlq=dataset_opt['datarootlq'],
                dataroothq=dataset_opt['dataroothq'],
                datatype=dataset_opt['datatype'],
                split=phase,
                data_len=dataset_opt['len'],
                img_sizeH=dataset_opt['img_sizeH'],
                img_sizeW=dataset_opt['img_sizeW']
                )
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset
