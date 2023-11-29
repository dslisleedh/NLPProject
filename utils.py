import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import timm.optim as toptim

import numpy as np
import random

from omegaconf import DictConfig

from typing import Optional
from copy import deepcopy
import os
import logging


optimizer_torch = [cls for cls in dir(optim) if isinstance(getattr(optim, cls), type)]
optimizer_timm = [cls for cls in dir(toptim) if isinstance(getattr(toptim, cls), type)]

optimizer_logger = logging.getLogger('Optimizer')
early_stopping_logger = logging.getLogger('EarlyStopping')


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


def seed_everything(seed: Optional[int] = 42):
    """Seed everything for reproducibility.
    
    Args:
        seed (int, optional): Seed value. Defaults to 42.
    """
    if seed is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    
class EarlyStopping(nn.Module):
    def __init__(self, criterion: str, patience: int, maximize: bool):
        self.criterion = criterion
        self.patience = patience
        self.maximize = maximize
        
        self.reset()
        
    def __call__(self, metric: dict, model: nn.Module):
        current = metric[self.criterion]
        if self.maximize:
            is_best = current > self.best
        else:
            is_best = current < self.best
        
        if is_best:
            self.best = current
            self.counter = 0
            self.best_state_dict_model = deepcopy(model.state_dict())
            
        else:
            self.counter += 1
            early_stopping_logger.info('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                early_stopping_logger.info('EarlyStopping: Stop training')
                self.is_stop = True
                
        return self.is_stop
                 
    def reset(self):
        self.best = -np.inf if self.maximize else np.inf
        self.counter = 0
        self.is_stop = False
        self.best_state_dict_model = None
        