import torch
import torch.nn as nn
from torch.optim import Optimizer, Scheduler
from typing import Tuple

def get_optimizer_and_scheduler(
        model: nn.Module,
        model_dim=512,
        warmup_steps=4000,
        beta_1 = 0.9,
        beta_2 = 0.98,
        eps = 1e-9,
        lr_mul=1.0,
) -> Tuple[Optimizer, Scheduler]:
    optimizer = torch.optim.Adam(model.parameters(), betas=(beta_1, beta_2), eps=eps)

    def lr_lambda(step):
        return lr_mul * model_dim**-0.5 * min(step**-0.5, step * warmup_steps**-1.5)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler
