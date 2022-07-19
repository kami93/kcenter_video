"""Configs."""
from fvcore.common.config import CfgNode
# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()

_C.TRAIN = CfgNode()
_C.TRAIN.ENABLE = False
_C.TRAIN.DATASET = ''
_C.TRAIN.BATCH_SIZE = 8
_C.TRAIN.GLOBAL_BATCH_SIZE = 256 # Global Batch Size = BATCH_SIZE * NUM_GPU
_C.TRAIN.EVAL_PERIOD = 1
_C.TRAIN.CHECKPOINT_PERIOD = 1
_C.TRAIN.AUTO_RESUME = True
_C.TRAIN.CHECKPOINT_FILE_PATH = ""

# If True, only restore the model states_dict (optimizer and scaler states are ignored.)
_C.TRAIN.LOAD_MODEL_STATES_ONLY = False
# If True, reset epochs when loading checkpoint.
_C.TRAIN.CHECKPOINT_EPOCH_RESET = False
# If set, clear all layer names according to the pattern provided.
_C.TRAIN.CHECKPOINT_CLEAR_NAME_PATTERN = ()  # e.g., ("backbone.", )

_C.TEST = CfgNode()
_C.TEST.ENABLE = False
_C.TEST.DATASET = ''
_C.TEST.BATCH_SIZE = 8
_C.TEST.CHECKPOINT_FILE_PATH = ""
_C.TEST.NUM_ENSEMBLE_SPATIAL = 3
_C.TEST.NUM_ENSEMBLE_TEMPORAL = 1

# ----------------------------------------------------------------------------
# Model options
# -----------------------------------------------------------------------------
_C.MODEL = CfgNode()

# Model name
_C.MODEL.MODEL_NAME = ""
_C.MODEL.NUM_CLASSES = 0

# Loss function.
_C.MODEL.METRIC_FUNC = "none"
_C.MODEL.LOSS_FUNC = "cross_entropy"

# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()

# Data path configurations
_C.DATA.PATH_TO_ANNOTATION = ""
_C.DATA.PATH_TO_RAWDATA = ""
_C.DATA.PATH_TO_JPEG = ""
_C.DATA.PATH_TO_MP4 = ""

_C.DATA.TRAIN_SPLIT_DIR = ""
_C.DATA.VAL_SPLIT_DIR = ""
_C.DATA.TEST_SPLIT_DIR = ""

_C.DATA.CACHE_PREFIX = "cache"
_C.DATA.CACHE_DIR = './'

_C.DATA.CROP_SIZE = 224 # The spatial crop size.
_C.DATA.PATCH_SIZE = 16 # The patch size (used in patch-based loaders)
_C.DATA.NUM_FRAMES = 8

# The mean & std value of the video raw pixels across the R G B channels.
_C.DATA.MEAN = [0.5, 0.5, 0.5]
_C.DATA.STD = [0.5, 0.5, 0.5]

# Data Augmentation
_C.DATA.TRAIN_JITTER_SCALES = [256, 320] # Spatial jitter scales for augmentation.
_C.DATA.TRAIN_COLORJITTER = False # Use color jittering augmentation
_C.DATA.TRAIN_RANDOM_FLIP = False # Perform random horizontal flip on the video frames during training.
_C.DATA.TRAIN_RAND_AUGMENT = None # Use RandAug (rand-m9-mstd0.5-inc1)

# Method to perform the ensemble, options include "sum" and "max".
_C.DATA.ENSEMBLE_METHOD = 'sum'

# Configs for Video (MP4) dataloaders
# Input videos may has different fps, convert it to the target video fps before frame sampling.
_C.DATA.TARGET_FPS = 30 
# The video sampling rate of the input clip.
_C.DATA.SAMPLING_RATE = 8

_C.DATA.INV_UNIFORM_SAMPLE = True
_C.DATA.CHANNEL_STANDARD = 'rgb' # 'rgb' | 'bgr'


# -----------------------------------------------------------------------------
# K-Center ViT model options
# -----------------------------------------------------------------------------
_C.KCENTER_VIT = CfgNode()
_C.KCENTER_VIT.NUM_HYBRID_FRAMES = 0

_C.KCENTER_VIT.TOTAL_SAMPLE_PATCHES = 1568

_C.KCENTER_VIT.KCENTER_SPATIAL_COEFFICIENT = 1.0
_C.KCENTER_VIT.KCENTER_TEMPORAL_COEFFICIENT = 1.0

_C.KCENTER_VIT.PATCH_SIZE = 16
_C.KCENTER_VIT.CHANNELS = 3
_C.KCENTER_VIT.EMBED_DIM = 768
_C.KCENTER_VIT.DEPTH = 12
_C.KCENTER_VIT.NUM_HEADS = 12
_C.KCENTER_VIT.MLP_RATIO = 4
_C.KCENTER_VIT.QKV_BIAS = True

_C.KCENTER_VIT.DROP = 0.0
_C.KCENTER_VIT.DROP_PATH = 0.1
_C.KCENTER_VIT.POS_DROPOUT = 0.0
_C.KCENTER_VIT.ATTN_DROPOUT = 0.0

_C.KCENTER_VIT.PRETRAINED = True # Use ImageNet pretrained weights
_C.KCENTER_VIT.PRETRAINED_WEIGHTS = "vit_1k" # Pretrained weights type

# -----------------------------------------------------------------------------
# K-Center TimeSformer model options
# -----------------------------------------------------------------------------
_C.KCENTER_TIMESFORMER = CfgNode()
_C.KCENTER_TIMESFORMER.NUM_HYBRID_FRAMES = 0

_C.KCENTER_TIMESFORMER.TOTAL_SAMPLE_PATCHES = 1568

_C.KCENTER_TIMESFORMER.KCENTER_SPATIAL_COEFFICIENT = 1.0
_C.KCENTER_TIMESFORMER.KCENTER_TEMPORAL_COEFFICIENT = 1.0

_C.KCENTER_TIMESFORMER.KCENTER_SPATIAL_DIVISION = 14
_C.KCENTER_TIMESFORMER.KCENTER_TEMPORAL_DIVISION = 8

_C.KCENTER_TIMESFORMER.PATCH_SIZE = 16
_C.KCENTER_TIMESFORMER.CHANNELS = 3
_C.KCENTER_TIMESFORMER.EMBED_DIM = 768
_C.KCENTER_TIMESFORMER.DEPTH = 12
_C.KCENTER_TIMESFORMER.NUM_HEADS = 12
_C.KCENTER_TIMESFORMER.MLP_RATIO = 4
_C.KCENTER_TIMESFORMER.QKV_BIAS = True

_C.KCENTER_TIMESFORMER.DROP = 0.0
_C.KCENTER_TIMESFORMER.DROP_PATH = 0.1
_C.KCENTER_TIMESFORMER.POS_DROPOUT = 0.0
_C.KCENTER_TIMESFORMER.ATTN_DROPOUT = 0.0

_C.KCENTER_TIMESFORMER.PRETRAINED = True # Use ImageNet pretrained weights
_C.KCENTER_TIMESFORMER.PRETRAINED_WEIGHTS = "vit_1k" # Pretrained weights type

# -----------------------------------------------------------------------------
# K-Center Motionformer model options
# -----------------------------------------------------------------------------
_C.KCENTER_MOTIONFORMER = CfgNode()
_C.KCENTER_MOTIONFORMER.NUM_HYBRID_FRAMES = 0

_C.KCENTER_MOTIONFORMER.TOTAL_SAMPLE_PATCHES = 1568

_C.KCENTER_MOTIONFORMER.KCENTER_SPATIAL_COEFFICIENT = 1.0
_C.KCENTER_MOTIONFORMER.KCENTER_TEMPORAL_COEFFICIENT = 1.0

_C.KCENTER_MOTIONFORMER.KCENTER_TEMPORAL_DIVISION = 8

_C.KCENTER_MOTIONFORMER.PATCH_SIZE = 16
_C.KCENTER_MOTIONFORMER.CHANNELS = 3
_C.KCENTER_MOTIONFORMER.EMBED_DIM = 768
_C.KCENTER_MOTIONFORMER.DEPTH = 12
_C.KCENTER_MOTIONFORMER.NUM_HEADS = 12
_C.KCENTER_MOTIONFORMER.MLP_RATIO = 4
_C.KCENTER_MOTIONFORMER.QKV_BIAS = True

_C.KCENTER_MOTIONFORMER.DROP = 0.0
_C.KCENTER_MOTIONFORMER.DROP_PATH = 0.1
_C.KCENTER_MOTIONFORMER.POS_DROPOUT = 0.0
_C.KCENTER_MOTIONFORMER.ATTN_DROPOUT = 0.0

_C.KCENTER_MOTIONFORMER.PRETRAINED = True # Use ImageNet pretrained weights
_C.KCENTER_MOTIONFORMER.PRETRAINED_WEIGHTS = "vit_1k" # Pretrained weights type

_C.KCENTER_MOTIONFORMER.ATTN_LAYER = 'trajectory' # 'joint' | 'trajectory' | 'divided'

_C.KCENTER_MOTIONFORMER.HEAD_DROPOUT = 0.0 # Dropout rate for the head

_C.KCENTER_MOTIONFORMER.USE_MLP = False # use MLP classification head
_C.KCENTER_MOTIONFORMER.HEAD_ACT = "tanh" # Activation for MLP head, 'tanh' | 'gelu' | 'relu'

# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()

# Effective learning rate. If set None, automatically calculated from BASE_LR.
_C.SOLVER.LR = None

# Base learning rate.
_C.SOLVER.BASE_LR = 5e-3

# LR reference batch-size
_C.SOLVER.REFERENCE_BATCH_SIZE = None

# Use Mixed Precision Training
_C.SOLVER.USE_MIXED_PRECISION = False

# Label smoothing
_C.SOLVER.SMOOTHING = 0.0

_C.SOLVER.LR_UPDATE_PERIOD = 'epoch'

# Learning rate policy (see utils/lr_policy.py for options and examples).
_C.SOLVER.LR_POLICY = "cosine"

# Final learning rates for 'cosine' policy.
_C.SOLVER.COSINE_END_LR = 0.0

# Exponential decay factor.
_C.SOLVER.GAMMA = 0.1

# Steps for 'steps_' policies (in epochs).
_C.SOLVER.STEPS = []
# Learning rates for 'steps_' policies.
_C.SOLVER.LRS = []

# Maximal number of epochs.
_C.SOLVER.MAX_EPOCH = 10

# Momentum.
_C.SOLVER.MOMENTUM = 0.9

# Betas for ADAM & ADAMW.
_C.SOLVER.BETAS = (0.9, 0.999)

# Momentum dampening.
_C.SOLVER.DAMPENING = 0.0

# Nesterov momentum in SGD.
_C.SOLVER.NESTEROV = False

# AMSGRAD in Adam.
_C.SOLVER.AMSGRAD = False

# weight decay regularization.
_C.SOLVER.WEIGHT_DECAY = 1e-4

# Gradually warm up the SOLVER.BASE_LR over this number of epochs.
_C.SOLVER.WARMUP_EPOCHS = 0.0

# The start learning rate of the warm up.
_C.SOLVER.WARMUP_START_LR = 0.01

# Optimization method.
_C.SOLVER.OPTIMIZING_METHOD = "sgd"

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.DEBUGGING = False
_C.MP_SPAWN = True
_C.PRINT_ALL_PROCS = False
_C.GPU_ID = None

_C.EVALUATE_INIT = False

# Number of GPUs to use (applies to both training and testing).
_C.NUM_GPUS = 1
# Number of machine to use for the job.
_C.NUM_SHARDS = 1
# The index of the current machine.
_C.SHARD_ID = 0

# Output basedir.
_C.OUTPUT_DIR = "./experiments/"

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries.
_C.RNG_SEED = 0

# Log period in iters.
_C.LOG_PERIOD = 10

# Distributed backend.
_C.DIST_BACKEND = "nccl"

# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CfgNode()
# Number of data loader workers per training process.
_C.DATA_LOADER.NUM_WORKERS = 1
# Load data to pinned host memory.
_C.DATA_LOADER.PIN_MEMORY = True
# Enable multi thread decoding.
_C.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE = True

# -----------------------------------------------------------------------------
# Tensorboard Visualization Options
# -----------------------------------------------------------------------------
_C.TENSORBOARD = CfgNode()
# Log to summary writer, this will automatically.
# log loss, lr and metrics during train/eval.
_C.TENSORBOARD.ENABLE = False
# Provide path to prediction results for visualization.
# This is a pickle file of [prediction_tensor, label_tensor]
_C.TENSORBOARD.PREDICTIONS_PATH = ""
# Path to directory for tensorboard logs.
# Default to to cfg.OUTPUT_DIR/runs-{cfg.TRAIN.DATASET}.
_C.TENSORBOARD.LOG_DIR = "tensorboard"

def _assert_and_infer_cfg(cfg):

    return cfg

def get_cfg():
    """
    Get a copy of the default config.
    """
    return _assert_and_infer_cfg(_C.clone())
