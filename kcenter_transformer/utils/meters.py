# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Meters."""

import datetime
import numpy as np
from collections import deque
import torch
from fvcore.common.timer import Timer
from collections import OrderedDict

from . import logs

logger = logs.get_logger(__name__)

class ScalarMeter(object):
    """
    A scalar meter uses a deque to track a series of scaler values with a given
    window size. It supports calculating the median and average values of the
    window, and also supports calculating the global average.
    """

    def __init__(self, window_size):
        """
        Args:
            window_size (int): size of the max length of the deque.
        """
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def reset(self):
        """
        Reset the deque.
        """
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def add_value(self, value):
        """
        Add a new scalar value to the deque.
        """
        self.deque.append(value)
        self.count += 1
        self.total += value

    def get_win_median(self):
        """
        Calculate the current median value of the deque.
        """
        return np.median(self.deque)

    def get_win_avg(self):
        """
        Calculate the current average value of the deque.
        """
        return np.mean(self.deque)

    def get_global_avg(self):
        """
        Calculate the global mean value.
        """
        return self.total / self.count


class TrainMeter(object):
    """
    Measure training stats.
    """

    def __init__(self, epoch_iters, cfg):
        """
        Args:
            epoch_iters (int): the overall number of iterations of one epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.epoch_iters = epoch_iters
        self.MAX_EPOCH = cfg.SOLVER.MAX_EPOCH * epoch_iters
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.update_timer = Timer()

        self.lr = None

        # Current minibatch metrics (smoothed over a window).
        self.mb_metric = {}

        # Epoch metrics
        self.total_metric = {}
        
        # Processing time stats
        self.dt = ScalarMeter(cfg.LOG_PERIOD)
        self.dt_data = ScalarMeter(cfg.LOG_PERIOD)
        self.dt_net = ScalarMeter(cfg.LOG_PERIOD)
        self.dt_update = ScalarMeter(cfg.LOG_PERIOD)

        self.num_samples = 0
        self.num_accumulations = 1
        self.output_dir = cfg.OUTPUT_DIR
        self.extra_stats = OrderedDict()
        self.extra_stats_total = OrderedDict()
        self.log_period = cfg.LOG_PERIOD
    
    def set_accumulations(self, num_accumulations):
        if self.num_accumulations == num_accumulations:
            return
            
        self.num_accumulations = num_accumulations
        new_epoch_iters = self.epoch_iters // num_accumulations
        self.MAX_EPOCH = int(self.MAX_EPOCH / self.epoch_iters * new_epoch_iters)
        self.epoch_iters = new_epoch_iters

    def reset(self):
        """
        Reset the Meter.
        """
        self.lr = None
        for mb_metric in self.mb_metric.values():
            mb_metric.reset()
        
        for key in self.total_metric.keys():
            self.total_metric[key] = 0.0

        self.num_samples = 0

        for key in self.extra_stats.keys():
            self.extra_stats[key].reset()
            self.extra_stats_total[key] = 0.0

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()
        self.net_timer.reset()
        self.net_timer.pause()
        self.update_timer.reset()
        self.update_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.resume()

    def net_toc(self):
        self.net_timer.pause()
        self.data_timer.resume()
        
    def update_tic(self):
        self.data_timer.pause()
        self.update_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.update_timer.pause()
        self.iter_timer.pause()

    def update_stats(self, metrics, lr, mb_size, stats={}):
        """
        Update the current stats.
        Args:
            metrics (dict)
            lr list(float): learning rates.
            mb_size (int): mini batch size.
        """
        self.lr = lr
        self.num_samples += mb_size

        for key in metrics.keys():
            if key not in self.mb_metric:
                self.mb_metric[key] = ScalarMeter(self.log_period)
            
            if key not in self.total_metric:
                self.total_metric[key] = 0.0

            self.mb_metric[key].add_value(metrics[key])
            self.total_metric[key] += metrics[key] * mb_size

        self.dt.add_value(self.iter_timer.seconds())
        self.dt_data.add_value(self.data_timer.seconds())
        self.dt_net.add_value(self.net_timer.seconds())
        self.dt_update.add_value(self.update_timer.seconds())

        for key in stats.keys():
            if key not in self.extra_stats:
                self.extra_stats[key] = ScalarMeter(self.log_period)
                self.extra_stats_total[key] = 0.0
            self.extra_stats[key].add_value(stats[key])
            self.extra_stats_total[key] += stats[key] * mb_size

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.dt.get_win_avg() * (
            self.MAX_EPOCH - (cur_epoch * self.epoch_iters + cur_iter + 1)
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = OrderedDict(
            _type="train_iter",
            epoch=f"{cur_epoch+1}/{self._cfg.SOLVER.MAX_EPOCH}",
            iter=f"{cur_iter+1}/{self.epoch_iters}",
            eta=eta,
            dt=self.dt.get_win_avg(),
            dt_data=self.dt_data.get_win_avg(),
            dt_net=self.dt_net.get_win_avg(),
            dt_update=self.dt_update.get_win_avg(),
            lr=self.lr,
        )

        for key in self.mb_metric.keys():
            stats[key] = self.mb_metric[key].get_win_median()

        for key in self.extra_stats.keys():
            stats[key] = self.extra_stats[key].get_win_median()

        logs.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        stats = OrderedDict(
            _type="train_epoch",
            epoch=f"{cur_epoch+1}/{self._cfg.SOLVER.MAX_EPOCH}",
            # eta=eta,
            lr=self.lr
        )

        for key in self.total_metric.keys():
            stats[key] = self.total_metric[key] / self.num_samples

        for key in self.extra_stats_total.keys():
            stats[key] = self.extra_stats_total[key] / self.num_samples
        
        logs.log_json_stats(stats)

class ValMeter(object):
    """
    Measures validation stats.
    """

    def __init__(self, max_iter, cfg):
        """
        Args:
            max_iter (int): the max number of iteration of the current epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.max_iter = max_iter
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()

        # Current minibatch errors (smoothed over a window).
        self.mb_metric = {}

        # Epoch metrics
        self.total_metric = {}

        self.num_samples = 0
        self.output_dir = cfg.OUTPUT_DIR
        self.extra_stats = OrderedDict()
        self.extra_stats_total = OrderedDict()
        self.log_period = cfg.LOG_PERIOD

    def reset(self):
        """
        Reset the Meter.
        """
        for mb_metric in self.mb_metric.values():
            mb_metric.reset()
        
        for key in self.total_metric.keys():
            self.total_metric[key] = 0.0

        self.num_samples = 0

        for key in self.extra_stats.keys():
            self.extra_stats[key].reset()
            self.extra_stats_total[key] = 0.0

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def update_stats(self, metrics, mb_size, stats={}):
        """
        Update the current stats.
        Args:
            metrics (dict)
            mb_size (int): mini batch size.
        """
        self.num_samples += mb_size

        for key in metrics.keys():
            if key not in self.mb_metric:
                self.mb_metric[key] = ScalarMeter(self.log_period)
            
            if key not in self.total_metric:
                self.total_metric[key] = 0.0

            self.mb_metric[key].add_value(metrics[key])
            self.total_metric[key] += metrics[key] * mb_size

        for key in stats.keys():
            if key not in self.extra_stats:
                self.extra_stats[key] = ScalarMeter(self.log_period)
                self.extra_stats_total[key] = 0.0
            self.extra_stats[key].add_value(stats[key])
            self.extxra_stats_total[key] += stats[key] * mb_size

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (self.max_iter - cur_iter - 1)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        
        stats = OrderedDict(
            _type="val_iter",
            epoch=f"{cur_epoch+1}/{self._cfg.SOLVER.MAX_EPOCH}",
            iter=f"{cur_iter+1}/{self.max_iter}",
            time_diff=self.iter_timer.seconds(),
            eta=eta
        )

        for key in self.mb_metric.keys():
            stats[key] = self.mb_metric[key].get_win_median()

        for key in self.extra_stats.keys():
            stats[key] = self.extra_stats[key].get_win_median()

        logs.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        stats = OrderedDict(
            _type="val_epoch",
            epoch=f"{cur_epoch+1}/{self._cfg.SOLVER.MAX_EPOCH}",
            time_diff=self.iter_timer.seconds()
        )

        for key in self.total_metric.keys():
            stats[key] = self.total_metric[key] / self.num_samples

        for key in self.extra_stats_total.keys():
            stats[key] = self.extra_stats_total[key] / self.num_samples

        logs.log_json_stats(stats)

class TestMeter(object):
    """
    Perform the multi-view ensemble for testing: each video with an unique index
    will be sampled with multiple clips, and the predictions of the clips will
    be aggregated to produce the final prediction for the video.
    The accuracy is calculated with the given ground truth labels.
    """

    def __init__(
        self,
        num_videos,
        num_clips,
        num_cls,
        overall_iters,
        ensemble_method="sum",
    ):
        """
        Construct tensors to store the predictions and labels. Expect to get
        num_clips predictions from each video, and calculate the metrics on
        num_videos videos.
        Args:
            num_videos (int): number of videos to test.
            num_clips (int): number of clips sampled from each video for
                aggregating the final prediction for the video.
            num_cls (int): number of classes for each prediction.
            overall_iters (int): overall iterations for testing.
            ensemble_method (str): method to perform the ensemble, options
                include "sum", and "max".
        """

        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.num_clips = num_clips
        self.overall_iters = overall_iters
        self.ensemble_method = ensemble_method
        # Initialize tensors.
        self.video_preds = torch.zeros((num_videos, num_cls))

        self.video_labels = torch.zeros((num_videos)).long()
        self.clip_count = torch.zeros((num_videos)).long()
        self.topk_accs = []
        self.stats = {}

        # Reset metric.
        self.reset()

    def reset(self):
        """
        Reset the metric.
        """
        self.clip_count.zero_()
        self.video_preds.zero_()
        self.video_labels.zero_()

    def update_stats(self, preds, labels, clip_ids):
        """
        Collect the predictions from the current batch and perform on-the-flight
        summation as ensemble.
        Args:
            preds (tensor): predictions from the current batch. Dimension is
                N x C where N is the batch size and C is the channel size
                (num_cls).
            labels (tensor): the corresponding labels of the current batch.
                Dimension is N.
            clip_ids (tensor): clip indexes of the current batch, dimension is
                N.
        """
        for ind in range(preds.shape[0]):
            vid_id = int(clip_ids[ind]) // self.num_clips
            if self.video_labels[vid_id].sum() > 0:
                assert torch.equal(
                    self.video_labels[vid_id].type(torch.FloatTensor),
                    labels[ind].type(torch.FloatTensor),
                )
            self.video_labels[vid_id] = labels[ind]
            if self.ensemble_method == "sum":
                self.video_preds[vid_id] += preds[ind]
            elif self.ensemble_method == "max":
                self.video_preds[vid_id] = torch.max(
                    self.video_preds[vid_id], preds[ind]
                )
            else:
                raise NotImplementedError(
                    "Ensemble Method {} is not supported".format(
                        self.ensemble_method
                    )
                )
            self.clip_count[vid_id] += 1

    def log_iter_stats(self, cur_iter):
        """
        Log the stats.
        Args:
            cur_iter (int): the current iteration of testing.
        """
        eta_sec = self.iter_timer.seconds() * (self.overall_iters - cur_iter)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "split": "test_iter",
            "cur_iter": "{}".format(cur_iter + 1),
            "eta": eta,
            "time_diff": self.iter_timer.seconds(),
        }
        logs.log_json_stats(stats)

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def finalize_metrics(self, ks=(1, 5)):
        """
        Calculate and log the final ensembled metrics.
        ks (tuple): list of top-k values for topk_accuracies. For example,
            ks = (1, 5) correspods to top-1 and top-5 accuracy.
        """
        if not all(self.clip_count == self.num_clips):
            logger.warning(
                "clip count {} ~= num clips {}".format(
                    ", ".join(
                        [
                            "{}: {}".format(i, k)
                            for i, k in enumerate(self.clip_count.tolist())
                        ]
                    ),
                    self.num_clips,
                )
            )

        self.stats = {"split": "test_final"}

        num_topks_correct = topks_correct(
            self.video_preds, self.video_labels, ks
        )
        topks = [
            (x / self.video_preds.size(0)) * 100.0
            for x in num_topks_correct
        ]

        assert len({len(ks), len(topks)}) == 1
        for k, topk in zip(ks, topks):
            self.stats["top{}_acc".format(k)] = "{:.{prec}f}".format(
                topk, prec=2
            )
            
        logs.log_json_stats(self.stats)

def topks_correct(preds, labels, ks):
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
        topks_correct (list): list of numbers, where the `i`-th entry
            corresponds to the number of top-`ks[i]` correct predictions.
    """
    assert preds.size(0) == labels.size(
        0
    ), "Batch dim of predictions and labels must match"
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )
    # (batch_size, max_k) -> (max_k, batch_size).
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size).
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct.
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k.
    topks_correct = [top_max_k_correct[:k, :].float().sum() for k in ks]
    return topks_correct
