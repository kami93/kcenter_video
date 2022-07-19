# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""
import contextlib
import pprint

import torch
from torch import _assert
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F

import kcenter_transformer.utils.losses as losses
import kcenter_transformer.utils.metrics as metrics
import kcenter_transformer.utils.optimizer as optim
import kcenter_transformer.utils.loader as loader
import kcenter_transformer.utils.checkpoint as cu
import kcenter_transformer.utils.distributed as du
import kcenter_transformer.utils.logs as logs
import kcenter_transformer.utils.misc as misc
import kcenter_transformer.utils.tensorboard_vis as tb
from kcenter_transformer.utils.meters import TrainMeter, ValMeter

from kcenter_transformer.models import build_model


from timm.models import model_parameters

logger = logs.get_logger(__name__)

def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    misc.set_random_seed(cfg.RNG_SEED)

    # Setup logging format.
    logs.setup_logging(cfg.OUTPUT_DIR, cfg.PRINT_ALL_PROCS)

    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model.
    model = build_model(cfg)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Construct the gradient scaler for AMP
    scaler = GradScaler(enabled=cfg.SOLVER.USE_MIXED_PRECISION)

    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(cfg, model, optimizer, scaler)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")

    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    # set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch))
    if cfg.EVALUATE_INIT:
        eval_epoch(val_loader, model, val_meter, start_epoch-1, cfg, writer)

    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)

        # Train for one epoch.
        train_epoch(
            train_loader, model, optimizer, scaler, train_meter, cur_epoch, cfg, writer
        )

        # Save a checkpoint.
        if cu.is_checkpoint_epoch(cfg, cur_epoch):
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, scaler, cur_epoch, cfg)
        
        # Evaluate the model on validation set.
        if misc.is_eval_epoch(cfg, cur_epoch):
            eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer)

    if writer is not None:
        writer.close()
    
def train_epoch(
    train_loader, model, optimizer, scaler, train_meter, cur_epoch, cfg, writer=None):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable train mode.
    model.train()
    iterations = len(train_loader)

    cur_global_batch_size = cfg.NUM_SHARDS * cfg.NUM_GPUS * cfg.TRAIN.BATCH_SIZE
    num_accumulation_iters = max(1, cfg.TRAIN.GLOBAL_BATCH_SIZE // cur_global_batch_size)
    effective_batch_size = cur_global_batch_size * num_accumulation_iters
    optimizer.zero_grad()

    drop_iterations = iterations % num_accumulation_iters
    iterations = iterations - drop_iterations
    effective_iterations = iterations // num_accumulation_iters
    if 1 < num_accumulation_iters:
        _assert(cfg.TRAIN.GLOBAL_BATCH_SIZE % cur_global_batch_size == 0, "global batch size should be divisible by current batch size.")
        logger.info(f"Gradient accumulation x{num_accumulation_iters} enabled!")
        logger.info(f"cur_global_batch_size: {cur_global_batch_size}, target_global_batch_size: {cfg.TRAIN.GLOBAL_BATCH_SIZE}") 
        logger.info(f"# iterations after drop: {iterations}/{iterations+drop_iterations}")
        logger.info(f"effective_iterations: {effective_iterations}")
        train_meter.set_accumulations(num_accumulation_iters)
        
    loss_func = losses.get_loss_func(cfg)
    metric_func = metrics.get_metric_func(cfg)

    scaler_scale = scaler.get_scale()
    accumulate_metric = {}
    train_meter.iter_tic()
    du.synchronize() # wait until other devices finish training setups
    for cur_iter, (inputs, labels, meta) in enumerate(train_loader):
        if cur_iter == iterations:
            break # break loop
        
        if isinstance(inputs, torch.Tensor):
            inputs = [inputs]
        
        extra_inputs = {}
        if 'extra_inputs' in meta:
            extra_inputs = meta.pop('extra_inputs')

        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)

            for key in extra_inputs.keys():
                value = extra_inputs[key]
                extra_inputs[key] = value.cuda(non_blocking=True)

            labels = labels.cuda(non_blocking=True)

        meta['cur_epoch'] = cur_epoch
        meta['phase'] = 'train'

        targets = labels
        
        inputs.append(extra_inputs)
        online_batch_size = inputs[0].size(0)
        train_meter.data_toc()

        update_iter = (cur_iter + 1) % num_accumulation_iters == 0
        maybe_no_sync = model.no_sync if (hasattr(model, "no_sync") and not update_iter) else contextlib.nullcontext
        extra_outputs = {}
        with maybe_no_sync():
            with autocast(cfg.SOLVER.USE_MIXED_PRECISION):
                preds = model(*inputs)
                if isinstance(preds, tuple):
                    preds, extra_outputs = preds
                
                loss_record = {}
                loss = loss_func(preds, targets)
                if isinstance(loss, (list, tuple)):
                    loss, loss_record = loss
                else:
                    loss_record['total_loss'] = loss.detach()
            
                loss_ = loss / num_accumulation_iters

            scaler.scale(loss_).backward()

        metric = metric_func(preds, labels)
        metric.update(loss_record)
        meta.update(extra_outputs)
        # sync all metrics across all the devices.
        with torch.no_grad():
            if cfg.NUM_GPUS > 1:
                metric_names = list(metric.keys())
                metric_names.sort()
                du.all_reduce([metric[name] for name in metric_names])
            
            for key, val in metric.items():
                if key not in accumulate_metric:
                    accumulate_metric[key] = 0.0
                accumulate_metric[key] += val.item()

        train_meter.net_toc()
        if update_iter:
            du.synchronize() # wait until other devices finish accumulations
            train_meter.update_tic()

            # Update the learning rate.
            lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / iterations, cfg)
            optim.set_lr(optimizer, lr)

            scaler.step(optimizer)

            grad_norm = {} # for record
            for name, param in model.named_parameters():
                if param.grad is None:
                    grad_norm[name] = 0.0
                    continue
                with torch.no_grad():
                    grad_norm[name] = param.grad.abs().sum().item()

            scaler.update()
            scaler_scale = scaler.get_scale()
            optimizer.zero_grad()

            train_meter.iter_toc()
            # Update and log stats.
            for key in accumulate_metric.keys():
                accumulate_metric[key] = accumulate_metric[key] / num_accumulation_iters

            train_meter.update_stats(
                accumulate_metric,
                lr,
                effective_batch_size
            )

            train_meter.log_iter_stats(cur_epoch, cur_iter//num_accumulation_iters)

            # write to tensorboard format if available.
            if writer is not None:
                total_global_steps = effective_iterations**cfg.SOLVER.MAX_EPOCH
                global_step=effective_iterations*cur_epoch + cur_iter//num_accumulation_iters
                record = {}
                record["Train/lr"] = lr
                record["Train/scaler_scale"] = scaler_scale

                for key, val in accumulate_metric.items():
                    record[f"Train/{key}"] = val

                for key, val in grad_norm.items():
                    record[f"Grad_norm/{key}"] = val

                writer.add_scalars(record,
                                   global_step=global_step)
            

            du.synchronize() # wait until other devices finish updates
            accumulate_metric = {}
            train_meter.iter_tic()
        
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()

    return cur_epoch

@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer=None):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()
    
    loss_func = losses.get_loss_func(cfg)
    metric_func = metrics.get_metric_func(cfg)
    for cur_iter, (inputs, labels, meta) in enumerate(val_loader):
        if isinstance(inputs, torch.Tensor):
            inputs = [inputs]
        
        extra_inputs = {}
        if 'extra_inputs' in meta:
            extra_inputs = meta.pop('extra_inputs')

        if cfg.NUM_GPUS:
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)

            for key in extra_inputs.keys():
                value = extra_inputs[key]
                extra_inputs[key] = value.cuda(non_blocking=True)

            targets = labels = labels.cuda(non_blocking=True)

        extra_inputs['labels'] = labels

        inputs.append(extra_inputs)
        online_batch_size = inputs[0].size(0)
        val_meter.data_toc()
        
        loss_record = {}
        extra_outputs = {}
        with autocast(cfg.SOLVER.USE_MIXED_PRECISION):
            preds = model(*inputs)
            if isinstance(preds, tuple):
                preds, extra_outputs = preds

                # Extra preds & labels (e.g., MAE reconstruction)
                if 'extra_targets' in extra_outputs:
                    targets = [targets, extra_outputs['extra_targets']]
                if 'extra_predictions' in meta:
                    preds = [preds, extra_outputs['extra_predictions']]

                loss = loss_func(preds, targets)
                if isinstance(loss, (list, tuple)):
                    loss, loss_record = loss
                else:
                    loss_record['total_loss'] = loss.detach()

        # Compute the errors.
        metric = metric_func(preds, labels)
        metric.update(loss_record)
        meta.update(extra_outputs)

        # sync all metrics across all the devices.
        if cfg.NUM_GPUS > 1:
            metric_names = list(metric.keys())
            metric_names.sort()
            du.all_reduce([metric[name] for name in metric_names])

        # Copy the stats from GPU to CPU (sync point).
        for key in metric.keys():
            metric[key] = metric[key].item()

        # Update and log stats.
        val_meter.update_stats(
            metric,
            online_batch_size * max(cfg.NUM_GPUS, 1)
            )

        val_meter.iter_toc()
        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    if writer is not None:
        global_step = writer.global_step

        record = {}
        for key, val in val_meter.total_metric.items():
            record[f"Val/{key}"] = val / val_meter.num_samples

        writer.add_scalars(record,
                           global_step=global_step)

    val_meter.reset()