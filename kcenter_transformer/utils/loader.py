# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Data loader."""
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

from ..datasets.build import build_dataset
from . import logs

logger = logs.get_logger(__name__)

def create_sampler(dataset, shuffle, cfg):
    """
    Create sampler for the given dataset.
    Args:
        dataset (torch.utils.data.Dataset): the given dataset.
        shuffle (bool): set to ``True`` to have the data reshuffled
            at every epoch.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
    Returns:
        sampler (Sampler): the created sampler.
    """
    if cfg.NUM_GPUS > 1:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        logger.info(f"Initialized {type(sampler)}")
        logger.info(f"num_replicas: {dist.get_world_size()}, rank: {dist.get_rank()}")

    else:
        sampler = None

    return sampler

def construct_loader(cfg, split):
    """
    Constructs the data loader for the given dataset.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
    """
    assert split in ["train", "val", "test"]
    if split in ["train"]:
        dataset_name = cfg.TRAIN.DATASET
        batch_size = int(cfg.TRAIN.BATCH_SIZE)
        shuffle = True
        drop_last = True
    elif split in ["val"]:
        dataset_name = cfg.TRAIN.DATASET
        batch_size = int(cfg.TRAIN.BATCH_SIZE)
        shuffle = False
        drop_last = False
    elif split in ["test"]:
        dataset_name = cfg.TEST.DATASET
        batch_size = int(cfg.TEST.BATCH_SIZE)
        shuffle = False
        drop_last = False

    # Construct the dataset
    dataset = build_dataset(dataset_name, cfg, split)

    # Create a sampler for multi-process training
    sampler = create_sampler(dataset, shuffle, cfg)
    # Create a loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        collate_fn=None,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=drop_last
    )

    return loader

def shuffle_dataset(loader, cur_epoch):
    """ "
    Shuffles the data.
    Args:
        loader (loader): data loader to perform shuffle.
        cur_epoch (int): number of the current epoch.
    """
    sampler = loader.sampler
    dataset = loader.dataset

    assert isinstance(
        sampler, (RandomSampler, DistributedSampler)
    ), "Sampler type '{}' not supported".format(type(sampler))
    # RandomSampler handles shuffling automatically
    if isinstance(sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        sampler.set_epoch(cur_epoch)
    
    if hasattr(dataset, 'cur_epoch'):
        setattr(dataset, 'cur_epoch', cur_epoch)