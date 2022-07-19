# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Optimizer."""
import torch
import torch.nn as nn

from . import lr_policy
from . import logs

logger = logs.get_logger(__name__)

def construct_optimizer(model, cfg):
    """
    Args:
        model (model): model to perform stochastic gradient descent
        optimization or ADAM optimization.
        cfg (config): configs of hyper-parameters of SGD or ADAM, includes base
        learning rate,  momentum, weight_decay, dampening, and etc.
    """
    no_weight_decay = {}
    if hasattr(model, 'no_weight_decay'):
        no_weight_decay = model.no_weight_decay()

    optim_params = param_groups_weight_decay(model,
                                             weight_decay=cfg.SOLVER.WEIGHT_DECAY,
                                             no_weight_decay_list=no_weight_decay)

    if cfg.SOLVER.OPTIMIZING_METHOD == "sgd":
        return torch.optim.SGD(
            optim_params,
            lr=cfg.SOLVER.LR,
            momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=0.0,
            dampening=cfg.SOLVER.DAMPENING,
            nesterov=cfg.SOLVER.NESTEROV,
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == "adam":
        return torch.optim.Adam(
            optim_params,
            lr=cfg.SOLVER.LR,
            betas=cfg.SOLVER.BETAS,
            weight_decay=0.0,
            amsgrad=cfg.SOLVER.AMSGRAD
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == "adamw":
        return torch.optim.AdamW(
            optim_params,
            lr=cfg.SOLVER.LR,
            betas=cfg.SOLVER.BETAS,
            weight_decay=0.0,
            amsgrad=cfg.SOLVER.AMSGRAD
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == "mt_adamw":
        return torch.optim._multi_tensor.AdamW(
            optim_params,
            lr=cfg.SOLVER.LR,
            betas=cfg.SOLVER.BETAS,
            weight_decay=0.0,
            amsgrad=cfg.SOLVER.AMSGRAD
        )
    else:
        raise NotImplementedError(
            "Does not support {} optimizer".format(cfg.SOLVER.OPTIMIZING_METHOD)
        )

def param_groups_weight_decay(
        model: nn.Module,
        weight_decay: float,
        no_weight_decay_list=()
):
    no_weight_decay_list = set(no_weight_decay_list)
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

def get_epoch_lr(cur_epoch, cfg):
    """
    Retrieves the lr for the given epoch (as specified by the lr policy).
    Args:
        cfg (config): configs of hyper-parameters of ADAM, includes base
        learning rate, betas, and weight decays.
        cur_epoch (float): the number of epoch of the current training stage.
    """
    return lr_policy.get_lr_at_epoch(cfg, cur_epoch)


def set_lr(optimizer, new_lr):
    """
    Sets the optimizer lr to the specified value.
    Args:
        optimizer (optim): the optimizer using to optimize the current network.
        new_lr (float): the new learning rate to set.
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr
