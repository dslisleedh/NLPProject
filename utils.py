import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import timm.optim as toptim

from omegaconf import DictConfig

from typing import Optional
from copy import deepcopy

import logging


optimizer_torch = [cls for cls in dir(optim) if isinstance(getattr(optim, cls), type)]
optimizer_timm = [cls for cls in dir(toptim) if isinstance(getattr(toptim, cls), type)]

optimizer_logger = logging.getLogger('Optimizer')


def get_optimizer(optimizer_cfg: DictConfig, model: nn.Module, additional_str: Optional[str] = None):
    optimizer_cfg = deepcopy(optimizer_cfg)
    if optimizer_cfg.name in optimizer_torch:
        name = optimizer_cfg.name
        del optimizer_cfg.name
        optimizer = getattr(optim, name)
        optimizer = optimizer(model.parameters(), **optimizer_cfg)
        from_ = 'torch.optim'
    elif optimizer_cfg.name in optimizer_timm:
        name = optimizer_cfg.name
        del optimizer_cfg.name
        optimizer = getattr(toptim, name)
        optimizer = optimizer(model.parameters(), **optimizer_cfg)
        from_ = 'timm.optim'
    else:
        raise ValueError('Optimizer {} not found.'.format(optimizer_cfg.name))
    additional_str = additional_str or ''
    optimizer_logger.info(additional_str + 'Optimizer_loaded: {} from {}'.format(name, from_))
    return optimizer