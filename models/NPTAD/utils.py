"""Learning rate scheduler.
Forked from https://github.com/OATML/non-parametric-transformers."""

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import (
    LambdaLR, CosineAnnealingLR)

class ConcatLR(torch.optim.lr_scheduler._LRScheduler):
    """
    From Over9000
    https://github.com/mgrankin/over9000/blob/master/train.py
    """
    def __init__(self, optimizer, scheduler1, scheduler2, total_steps,
                 pct_start=0.5, last_epoch=-1):
        self.scheduler1 = scheduler1
        self.scheduler2 = scheduler2
        self.step_start = float(pct_start * total_steps) - 1
        self.curr_epoch = 0
        super(ConcatLR, self).__init__(optimizer, last_epoch)

    def step(self):
        if self.curr_epoch <= self.step_start:
            self.scheduler1.step()
        else:
            self.scheduler2.step()
        self.curr_epoch += 1
        super().step()

    def get_lr(self):
        if self.curr_epoch <= self.step_start:
            return self.scheduler1.get_last_lr()
        else:
            return self.scheduler2.get_last_lr()


class LRScheduler:
    """Flat and Anneal"""
    def __init__(self, c, optimizer):
        self.c = c
        self.optimizer = optimizer
        self.num_steps = 0

        self.construct_auto_scheduler()

        print(f'Initialized flat and anneal learning rate scheduler.')

    def construct_auto_scheduler(self):
        total_steps = self.c.exp_num_total_steps

        if self.c.exp_optimizer_warmup_proportion >= 0:
            num_warmup_steps = (
                    total_steps * self.c.exp_optimizer_warmup_proportion)
        else:
            num_warmup_steps = self.c.exp_optimizer_warmup_fixed_n_steps

        print(f'Warming up for {num_warmup_steps}/{total_steps} steps.')

        def d(x):
            return 1

        assert self.c.exp_optimizer_warmup_proportion >= 0

        # We use exp_optimizer_warmup_proportion to denote the
        # flat LR regime, prior to annealing
        dummy = LambdaLR(self.optimizer, d)
        cosine = CosineAnnealingLR(
            self.optimizer, int(total_steps * (
                1 - self.c.exp_optimizer_warmup_proportion)))
        self.scheduler = ConcatLR(
            self.optimizer, dummy, cosine, total_steps,
            self.c.exp_optimizer_warmup_proportion)

    def step(self):
        self.num_steps += 1
        num = self.num_steps
        self.scheduler.step_update(num_updates=num)