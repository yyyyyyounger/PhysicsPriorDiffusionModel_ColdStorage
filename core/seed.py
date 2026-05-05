"""Shared RNG seeding for reproducible training and inference."""
import random

import numpy as np
import torch


def set_seed(seed, deterministic=False):
    """
    Set Python / NumPy / PyTorch (and CUDA) RNG seeds.

    Args:
        seed: Integer seed.
        deterministic: If True, use cudnn deterministic algorithms and disable
            autotune benchmark (slower but more reproducible).
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def set_sample_seed(seed):
    """
    Seed RNG for a single sample (e.g. one DDIM forward) without changing
    cuDNN benchmark/deterministic flags.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
