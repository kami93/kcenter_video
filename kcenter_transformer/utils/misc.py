# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import math
import random

import numpy as np
from datetime import datetime
import torch

from . import logs
from . import multiprocessing as mpu

logger = logs.get_logger(__name__)

def set_random_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def check_nan_losses(loss):
    """
    Determine whether the loss is NaN (not a number).
    Args:
        loss (loss): loss to check whether is NaN.
    """
    if math.isnan(loss):
        raise RuntimeError("ERROR: Got NaN losses {}".format(datetime.now()))

def is_eval_epoch(cfg, cur_epoch):
    """
    Determine if the model should be evaluated at the current epoch.
    Args:
        cfg (CfgNode): configs. Details can be found in
            lib/config/defaults.py
        cur_epoch (int): current epoch.
        multigrid_schedule (List): schedule for multigrid training.
    """
    if cur_epoch + 1 == cfg.SOLVER.MAX_EPOCH:
        return True

    return (cur_epoch + 1) % cfg.TRAIN.EVAL_PERIOD == 0

def launch_job(cfg, init_method, func, daemon=False):
    """
    Run 'func' on one or more GPUs, specified in cfg
    Args:
        cfg (CfgNode): configs. Details can be found in
            lib/config/defaults.py
        init_method (str): initialization method to launch the job with multiple
            devices.
        func (function): job to run on GPU(s)
        daemon (bool): The spawned processes’ daemon flag. If set to True,
            daemonic processes will be created
    """
    if cfg.NUM_GPUS > 1:
        if cfg.MP_SPAWN:
            torch.multiprocessing.spawn(
                mpu.run,
                nprocs=cfg.NUM_GPUS,
                args=(
                    cfg.NUM_GPUS,
                    func,
                    init_method,
                    cfg.SHARD_ID,
                    cfg.NUM_SHARDS,
                    cfg.DIST_BACKEND,
                    cfg,
                ),
                daemon=daemon,
            )

        else:
            mpu.run(local_rank=cfg.GPU_ID,
                    num_proc=cfg.NUM_GPUS,
                    func=func,
                    init_method=init_method,
                    shard_id=cfg.SHARD_ID,
                    num_shards=cfg.NUM_SHARDS,
                    backend=cfg.DIST_BACKEND,
                    cfg=cfg
                    )

    else:
        func(cfg=cfg)