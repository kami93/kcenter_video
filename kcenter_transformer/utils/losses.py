# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""
import torch.nn as nn

_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
    "mse": nn.MSELoss,
}

def get_loss_func(cfg):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    loss_name = cfg.MODEL.LOSS_FUNC
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))

    loss_creator = _LOSSES[loss_name]
    if loss_name in {"cross_entropy", }:
        smoothing = cfg.SOLVER.SMOOTHING
        loss_func = loss_creator(label_smoothing=smoothing)
    else:
        loss_func = loss_creator()

    return loss_func
