# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""
import os
import os.path as osp
import pickle as pkl

from fvcore.common.file_io import PathManager
import torch

import kcenter_transformer.utils.metrics as metrics
import kcenter_transformer.utils.loader as loader
import kcenter_transformer.utils.checkpoint as cu
import kcenter_transformer.utils.distributed as du
import kcenter_transformer.utils.logs as logs
import kcenter_transformer.utils.misc as misc
import kcenter_transformer.utils.tensorboard_vis as tb
from kcenter_transformer.utils.meters import TestMeter
from kcenter_transformer.models import build_model

logger = logs.get_logger(__name__)

def test(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    misc.set_random_seed(cfg.RNG_SEED)

    # Setup logging format.
    logs.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Test with config:")
    logger.info(cfg)

    # Build the video model and print model statistics.
    model = build_model(cfg)

    cu.load_test_checkpoint(cfg, model)

    # Create video testing loaders.
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    assert (
        len(test_loader.dataset)
        % (cfg.TEST.NUM_ENSEMBLE_TEMPORAL * cfg.TEST.NUM_ENSEMBLE_SPATIAL)
        == 0
    )
    # Create meters for multi-view testing.
    test_meter = TestMeter(
        len(test_loader.dataset)
        // (cfg.TEST.NUM_ENSEMBLE_TEMPORAL * cfg.TEST.NUM_ENSEMBLE_SPATIAL),
        cfg.TEST.NUM_ENSEMBLE_TEMPORAL * cfg.TEST.NUM_ENSEMBLE_SPATIAL,
        cfg.MODEL.NUM_CLASSES,
        len(test_loader),
        ensemble_method=cfg.DATA.ENSEMBLE_METHOD,
    )

    # Set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        cfg.TENSORBOARD.LOG_DIR = "runs-{}".format(cfg.TEST.DATASET) if cfg.TENSORBOARD.LOG_DIR == "" else cfg.TENSORBOARD.LOG_DIR
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # # Perform multi-view test on the entire dataset.
    test_meter = perform_test(test_loader, model, test_meter, cfg, writer)
    
    if writer is not None:
        writer.close()

@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg, writer=None):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter object, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable eval mode.
    model.eval()
    test_meter.iter_tic()

    metric_fun = metrics.get_metric_func(cfg)
    for cur_iter, (inputs, labels, meta) in enumerate(test_loader):
        # Transfer the data to the current GPU device.
        if isinstance(inputs, torch.Tensor):
            inputs = [inputs]

        extra_inputs = {}
        if 'extra_inputs' in meta:
            extra_inputs = meta.pop('extra_inputs')

        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)

            for key in extra_inputs.keys():
                value = extra_inputs[key]
                extra_inputs[key] = value.cuda(non_blocking=True)

            labels = labels.cuda(non_blocking=True)
            meta['labels'] = labels

        targets = labels

        inputs.append(extra_inputs)
        online_batch_size = inputs[0].size(0)
        test_meter.data_toc()

        # Perform the forward pass.
        extra_outputs = {}
        preds = model(*inputs)
        if isinstance(preds, (list, tuple)):
            preds, extra_outputs = preds

        # Extra preds & labels (e.g., MAE reconstruction)
        if 'extra_targets' in extra_outputs:
            targets = [targets, extra_outputs['extra_targets']]
            
        if 'extra_predictions' in extra_outputs:
            preds = [preds, extra_outputs['extra_predictions']]

        meta.update(extra_outputs)
        if preds is not None:
            video_index = meta['video_index'].cuda(non_blocking=True)
            # Gather all the predictions across all the devices to perform ensemble.
            if cfg.NUM_GPUS > 1:
                preds, labels, video_index = du.all_gather(
                    [preds, labels, video_index]
                )
            if cfg.NUM_GPUS:
                preds = preds.cpu()
                labels = labels.cpu()
                video_index = video_index.cpu()

            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats(
                preds.detach(), labels.detach(), video_index.detach()
            )
            test_meter.log_iter_stats(cur_iter)
        else:
            test_meter.iter_toc()

        test_meter.iter_tic()

    test_meter.finalize_metrics()

    # Log epoch stats and print the final testing results.
    all_preds = test_meter.video_preds.clone().detach()
    all_labels = test_meter.video_labels
    stats = test_meter.stats
    if cfg.NUM_GPUS:
        all_preds = all_preds.cpu()
        all_labels = all_labels.cpu()

    save_path = osp.join(cfg.OUTPUT_DIR, 'predictions.pkl')
    with PathManager.open(save_path, "wb") as f:
        pkl.dump([all_preds, all_labels, stats], f)

    logger.info(
        "Successfully saved prediction results to {}".format(save_path)
    )
    
    return test_meter