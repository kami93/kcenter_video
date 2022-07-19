# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging as log
import os

from torch.utils.tensorboard import SummaryWriter

from . import logs

logger = logs.get_logger(__name__)
log.getLogger("matplotlib").setLevel(log.ERROR)

class TensorboardWriter(object):
    """
    Helper class to log information to Tensorboard.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode): configs. Details can be found in
                slowfast/config/defaults.py
        """
        # class_names: list of class names.
        # cm_subset_classes: a list of class ids -- a user-specified subset.
        # parent_map: dictionary where key is the parent class name and
        #   value is a list of ids of its children classes.
        # hist_subset_classes: a list of class ids -- user-specified to plot histograms.
        (
            self.class_names,
            self.cm_subset_classes,
            self.parent_map,
            self.hist_subset_classes,
        ) = (None, None, None, None)
        self.cfg = cfg
        self.global_step = 0

        log_dir = os.path.join(cfg.OUTPUT_DIR, cfg.TENSORBOARD.LOG_DIR)

        self.writer = SummaryWriter(log_dir=log_dir)
        logger.info(
            "To see logged results in Tensorboard, please launch using the command \
            `tensorboard  --port=<port-number> --logdir {}`".format(
                log_dir
            )
        )

    def add_scalars(self, data_dict, global_step=None):
        """
        Add multiple scalars to Tensorboard logs.
        Args:
            data_dict (dict): key is a string specifying the tag of value.
            global_step (Optinal[int]): Global step value to record.
        """
        if global_step is not None:
            self.global_step = global_step
        else:
            self.global_step += 1

        if self.writer is not None:
            for key, item in data_dict.items():
                self.writer.add_scalar(key, item, global_step)

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.flush()
        self.writer.close()