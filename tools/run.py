# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os.path as osp
import argparse
import sys

from kcenter_transformer.configs.defaults import get_cfg
import kcenter_transformer.utils.checkpoint as cu
from kcenter_transformer.utils.misc import launch_job

from train_net import train
from test_net import test

""""
General launcher script for Emprical Risk Minimization (i.e., supervised learning) cases.
"""

def parse_args(default=False):
    parser = argparse.ArgumentParser(
        description='Parse arguments')
    parser.add_argument(
        "--shard_id",
        help="The shard id of current node, Starts from 0 to num_shards - 1",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--num_shards",
        help="Number of shards using by the job",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--init_method",
        help="Initialization method, includes TCP or shared file-system",
        default="tcp://localhost:9999",
        type=str,
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--sym",
        default=0,
        type=int,
    )
    parser.add_argument(
        "opts",
        help="See configs/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
    
    return parser.parse_args()

def load_config(args):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # Inherit parameters from args.
    if hasattr(args, "num_shards") and hasattr(args, "shard_id"):
        cfg.NUM_SHARDS = args.num_shards
        cfg.SHARD_ID = args.shard_id
    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed
    if hasattr(args, "output_dir"):
        cfg.OUTPUT_DIR = args.output_dir

    if cfg.DEBUGGING:
        cfg.NUM_GPUS = 1
        cfg.DATA_LOADER.NUM_WORKERS = 0
        cfg.OUTPUT_DIR = osp.join(cfg.OUTPUT_DIR, 'debugging_logs')

    # Calculate learning rates
    if cfg.SOLVER.LR is not None:
        pass

    elif cfg.SOLVER.REFERENCE_BATCH_SIZE is not None:
        cfg.SOLVER.LR = cfg.SOLVER.BASE_LR * cfg.TRAIN.GLOBAL_BATCH_SIZE / cfg.SOLVER.REFERENCE_BATCH_SIZE
        cfg.SOLVER.COSINE_END_LR = cfg.SOLVER.COSINE_END_LR * cfg.TRAIN.GLOBAL_BATCH_SIZE / cfg.SOLVER.REFERENCE_BATCH_SIZE
        cfg.SOLVER.WARMUP_START_LR = cfg.SOLVER.WARMUP_START_LR * cfg.TRAIN.GLOBAL_BATCH_SIZE / cfg.SOLVER.REFERENCE_BATCH_SIZE

    else:
        cfg.SOLVER.LR = cfg.SOLVER.BASE_LR

    # Create the checkpoint dir.
    cu.make_checkpoint_dir(cfg.OUTPUT_DIR)
    return cfg

def main():
    """ argument define """
    args = parse_args()

    """ set torch device"""

    cfg = load_config(args)

    if cfg.TRAIN.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=train)

    if cfg.TEST.ENABLE:
        if cfg.TRAIN.ENABLE:
            cfg.SHARD_ID = 0
            cfg.NUM_SHARDS = 1

        launch_job(cfg=cfg, init_method="tcp://localhost:9999", func=test)

if __name__ == "__main__":
    main()
