# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Metric functions."""
import torch
import torch.nn as nn

class TopkErrors(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, preds, targets, ks=(1, 5)):
        """
        Given the predictions, labels, and a list of top-k values, compute the
        number of correct predictions for each top-k value.

        Args:
            preds (array): array of predictions. Dimension is batchsize
                N x ClassNum.
            labels (array): array of labels. Dimension is batchsize N.
            ks (list): list of top-k values. For example, ks = [1, 5] correspods
                to top-1 and top-5.

        Returns:
            topks_correct (dict).
        """
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        if isinstance(targets, (list, tuple)):
            targets = targets[0]

        assert preds.size(0) == targets.size(
            0
        ), "Batch dim of predictions and labels must match"
        # Find the top max_k predictions for each sample
        _top_max_k_vals, top_max_k_inds = torch.topk(
            preds, max(ks), dim=1, largest=True, sorted=True
        )
        # (batch_size, max_k) -> (max_k, batch_size).
        top_max_k_inds = top_max_k_inds.t()
        # (batch_size, ) -> (max_k, batch_size).
        rep_max_k_labels = targets.view(1, -1).expand_as(top_max_k_inds)
        # (i, j) = 1 if top i-th prediction for the j-th sample is correct.
        top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
        # Compute the number of topk correct predictions for each k.
        topk_errors = {f"Top{k}_err": (1.0 - top_max_k_correct[:k, :].float().sum() / preds.size(0)) * 100.0 for k in ks}

        return topk_errors

class NoneMetric(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, preds, targets):
        return {}

_METRICS = {
    "none": NoneMetric,
    "topk_errors": TopkErrors
}

def get_metric_func(cfg):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    metric_name = cfg.MODEL.METRIC_FUNC
    if metric_name not in _METRICS.keys():
        raise NotImplementedError("Metric {} is not supported".format(metric_name))

    metrics_creator = _METRICS[metric_name]
    metric_func = metrics_creator()

    return metric_func