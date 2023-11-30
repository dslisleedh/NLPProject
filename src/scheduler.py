import torch
import torch.nn as nn
import torch.optim as optim

from omegaconf import DictConfig
from typing import Optional
from copy import deepcopy

import logging

scheduler_logger = logging.getLogger('scheduler')

scheduler_all = optim.lr_scheduler.__all__
custom_optimizers_all = [
    'LinearWarmupCosineAnnealingLR', 'ConstantMultiplicativeLR', 'LinearWarmupLinearDecayLR'
]
__all__ = scheduler_all + custom_optimizers_all


class LinearWarmupCosineAnnealingLR(optim.lr_scheduler.SequentialLR):
    def __init__(
            self, opt: optim.Optimizer, warmup_iter: int, max_iter: int, eta_min: float = 0.,
            start_factor: float = 1e-4,
    ):
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            opt, start_factor=start_factor, end_factor=1., total_iters=warmup_iter
        )
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max_iter - warmup_iter, eta_min=eta_min
        )
        scheduler = [warmup_scheduler, cosine_scheduler]
        milestones = [warmup_iter]
        super(LinearWarmupCosineAnnealingLR, self).__init__(
            opt, schedulers=scheduler, milestones=milestones
        )


class LinearWarmupLinearDecayLR(optim.lr_scheduler.SequentialLR):
    def __init__(
            self, opt: optim.Optimizer, warmup_iter: int, max_iter: int, eta_min: float = 0.,
            start_factor: float = 1e-4,
    ):
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            opt, start_factor=start_factor, end_factor=1., total_iters=warmup_iter
        )
        decay_scheduler = optim.lr_scheduler.LinearLR(
            opt, start_factor=1., end_factor=eta_min, total_iters=max_iter - warmup_iter
        )
        scheduler = [warmup_scheduler, decay_scheduler]
        milestones = [warmup_iter]
        super(LinearWarmupLinearDecayLR, self).__init__(
            opt, schedulers=scheduler, milestones=milestones
        )


class ConstantMultiplicativeLR(optim.lr_scheduler.MultiplicativeLR):
    def __init__(self, opt: optim.Optimizer, multiplier: float):
        super(ConstantMultiplicativeLR, self).__init__(opt, lr_lambda=lambda _: multiplier)


def get_scheduler(
        scheduler_cfg: DictConfig, optimizer: optim.Optimizer, additional_str: Optional[str] = None
):
    scheduler_cfg = deepcopy(scheduler_cfg)
    name = scheduler_cfg.name
    del scheduler_cfg.name
    if name in custom_optimizers_all:
        cls_ = globals()[name]
        scheduler = cls_(optimizer, **scheduler_cfg)
    elif name in scheduler_all:
        cls_ = getattr(optim.lr_scheduler, name)
        scheduler = cls_(optimizer, **scheduler_cfg)
    else:
        raise ValueError('Unknown scheduler: {}'.format(name))
    additional_str = additional_str or ''
    scheduler_logger.info(additional_str + 'Scheduler_loaded: {}'.format(name))
    return scheduler
