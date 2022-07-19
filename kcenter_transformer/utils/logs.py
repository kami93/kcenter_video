# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Logging."""
import atexit
import builtins
import decimal
import functools
import logging
import os
import sys
import simplejson
from fvcore.common.file_io import PathManager
from collections import OrderedDict
from copy import copy

from . import distributed as du

MAPPING = {
    'WARNING': 33, # yellow,
    'INFO': 37, # white
    'DEBUG': 36, # cyan
    'CRITICAL': 41, # white on red bg
    'ERROR': 31, # red
}
PREFIX = '\033['
SUFFIX = '\033[0m'

class ColoredFormatter(logging.Formatter):

    def __init__(self, pattern, datefmt):
        logging.Formatter.__init__(self, pattern, datefmt=datefmt)

    def format(self, record):
        colored_record = copy(record)
        levelname = colored_record.levelname
        seq = MAPPING.get(levelname, 37) # default white
        colored_levelname = ('{0}{1}m{2}{3}') \
            .format(PREFIX, seq, levelname, SUFFIX)
        colored_record.levelname = colored_levelname

        return logging.Formatter.format(self, colored_record)

def _suppress_print():
    """
    Suppresses printing from the current process.
    """

    def print_pass(*objects, sep=" ", end="\n", file=sys.stdout, flush=False):
        pass

    builtins.print = print_pass


@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    io = PathManager.open(filename, "a", buffering=1024)
    atexit.register(io.close)
    return io


def setup_logging(output_dir=None, print_all_procs=False):
    """
    Sets up the logging for multiple processes. Only enable the logging for the
    master process, and suppress logging for the non-master processes.
    """
    if du.is_master_proc() or print_all_procs:
        # Enable logging for the master process.
        logging.root.handlers = []

        logger = logging.getLogger()
        logger.propagate = False
        logger.setLevel(logging.DEBUG)

        formatter = ColoredFormatter(
            "[%(asctime)s][%(levelname)s] %(filename)s: %(lineno)3d: %(message)s",
            datefmt="%m/%d %H:%M:%S",
        )

        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
        if output_dir is not None:
            rank = du.get_rank()
            filename = os.path.join(output_dir, f"rank{rank:02d}_stdout.log")
            fh = logging.StreamHandler(_cached_log_stream(filename))
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    else:
        # Suppress logging for non-master processes.
        _suppress_print()
        logger = logging.getLogger()
        logger.propagate = False
        logger.setLevel(logging.CRITICAL)

def get_logger(name):
    """
    Retrieve the logger with the specified name or, if name is None, return a
    logger which is the root logger of the hierarchy.
    Args:
        name (string): name of the logger.
    """
    return logging.getLogger(name)

def log_json_stats(stats):
    """
    Logs json stats.
    Args:
        stats (dict): a dictionary of statistical information to log.
    """
    stats_out = OrderedDict()
    for k, v in stats.items():
        if isinstance(v, float):
            if k == 'lr':
                stats_out[k] = decimal.Decimal("{:.8f}".format(v))
            else:
                stats_out[k] = decimal.Decimal("{:.2f}".format(v))
        else:
            stats_out[k] = v
    json_stats = simplejson.dumps(stats_out, sort_keys=False, use_decimal=True)
    logger = get_logger(__name__)

    if du.is_master_proc():
        logger.info("json_stats: {:s}".format(json_stats))
