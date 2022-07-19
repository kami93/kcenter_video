import math

import torch
from einops import rearrange

from ...utils import logs
from .kinetics_mp4 import Kinetics_mp4 as Kinetics_baseclass
from ..build import DATASET_REGISTRY

logger = logs.get_logger(__name__)

@DATASET_REGISTRY.register()
class Kinetics_patch(Kinetics_baseclass):
    """ Kinetics dataset for patch-based models"""
    
    def __init__(self, cfg, mode, num_retries=10):
        super().__init__(cfg, mode, num_retries)
        self.num_frames = cfg.DATA.NUM_FRAMES
        self.num_patches = cfg.DATA.CROP_SIZE // cfg.DATA.PATCH_SIZE

        data_mean, data_std = cfg.DATA.MEAN[0], cfg.DATA.STD[0] # assume uniform mand&std for all channels.
        rgb_range = 1.0 / data_std

        spatial_index_range = math.sqrt(cfg.DATA.PATCH_SIZE/2) * rgb_range
        temporal_index_range = math.sqrt(cfg.DATA.PATCH_SIZE) * rgb_range

        T_index = torch.linspace(-temporal_index_range/2, temporal_index_range/2, self.num_frames)
        H_index = torch.linspace(-spatial_index_range/2, spatial_index_range/2, self.num_patches)
        W_index = torch.linspace(-spatial_index_range/2, spatial_index_range/2, self.num_patches)
        self.patch_index = torch.stack(torch.meshgrid(T_index, H_index, W_index), dim=-1)

    def __getitem__(self, index, seed=None):
        """
        Given the video index, return the list of frames, label, and video
        index if the video frames can be fetched.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): the index of the video.
        """
        frames, label, meta = super().__getitem__(index, seed=seed)
        patches = rearrange(frames, 'c t (ph p1) (pw p2) -> t ph pw (c p1 p2)', ph=self.num_patches, pw=self.num_patches)

        # append self.patch_index to patches vector
        patches_ = torch.cat([patches, self.patch_index], dim=-1)
    
        return patches_, label, meta