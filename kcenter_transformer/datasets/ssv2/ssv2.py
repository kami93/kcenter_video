# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import json
import os
import os.path as osp
from pathlib import Path as P
import time
import random
from itertools import chain as chain
import copy
from multiprocessing import Pool

import numpy as np
import torch
from torch.utils.data import Dataset
from fvcore.common.file_io import PathManager
from einops import rearrange
from tqdm import tqdm
from torchvision import transforms

from .. import transform
from .. import autoaugment as autoaugment
from ...utils import logs
from ...utils import distributed as du
from .. import utils as dataset_utils
from ..build import DATASET_REGISTRY

logger = logs.get_logger(__name__)

class cache_callback():
    def __init__(self, num_samples):
        self.pbar = tqdm(total=num_samples)
        self.success_list = []
        self.fail_list = []
    
    def __call__(self, ret):
        self.pbar.update(1)
        video_size = ret[0]
        if video_size is not None:
            self.success_list.append(ret)
        else:
            self.fail_list.append(ret)
    
    def close(self):
        self.pbar.close()

    def get_outputs(self):
        return copy.deepcopy(self.success_list), copy.deepcopy(self.fail_list)

def check_video_file(frames_dir, index):
    frame_list = [path for path in P(frames_dir).glob("*.jpeg")]
    video_size = None

    length = len(frame_list)
    if length > 0:
        frame = dataset_utils.retry_load_images(frame_list[0:1], 10)[0]
        height, width  = frame.shape[1:]
        video_size = (length, height, width)

    else:
        logger.warn(f"Frames from {frames_dir} is empty!")
    
    return video_size, frames_dir, index

@DATASET_REGISTRY.register()
class Ssv2(Dataset):
    """
    ssv2 dataset base class. Uses JPEG files.
    """
    def __init__(self, cfg, mode, num_retries=10):
        """
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Something-Something V2".format(mode)
        self.mode = mode
        self.cfg = cfg

        self._video_meta = {}
        self._num_retries = num_retries

        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            assert cfg.TEST.NUM_ENSEMBLE_TEMPORAL == 1
            self._num_clips = cfg.TEST.NUM_ENSEMBLE_SPATIAL

        if du.is_master_proc():
            logger.info("Constructing Something-Something V2 {}...".format(mode))
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        if self.mode == 'train':
            data_split = 'train'
            split_dir = self.cfg.DATA.TRAIN_SPLIT_DIR
        
        else:
            # Note that we do not consider 'testset' for ssv2, as it does not come with labels.
            data_split = 'val'
            split_dir = self.cfg.DATA.VAL_SPLIT_DIR

        videos_dir = osp.join(self.cfg.DATA.PATH_TO_JPEG, split_dir)
        cache_dir = self.cfg.DATA.CACHE_DIR
        
        if du.is_master_proc() and not osp.isdir(cache_dir):
            os.mkdir(cache_dir)
        cache_file = osp.join(cache_dir, f'{self.cfg.DATA.CACHE_PREFIX}_{data_split}.cache')
        
        if du.is_master_proc() and not osp.isfile(cache_file):
            logger.info(f"Start building cache {cache_file}")

            # Loading label names.
            with PathManager.open(
                osp.join(
                    self.cfg.DATA.PATH_TO_ANNOTATION,
                    "something-something-v2-labels.json",
                ),
                "r",
            ) as f:
                label_dict = json.load(f)

            # Loading labels.
            label_file = osp.join(
                self.cfg.DATA.PATH_TO_ANNOTATION,
                "something-something-v2-{}.json".format(
                    "train" if self.mode == "train" else "validation"
                ),
            )
            with PathManager.open(label_file, "r") as f:
                label_json = json.load(f)

            _video_names = []
            _labels = []
            for video in label_json:
                video_name = video["id"]
                template = video["template"]
                template = template.replace("[", "")
                template = template.replace("]", "")
                label = int(label_dict[template])
                _video_names.append(video_name)
                _labels.append(label)

            _path_to_videos = [osp.join(videos_dir, f"{name}") for name in _video_names]
            pool = Pool(40)
            callback = cache_callback(len(_path_to_videos))
            # sort videos by length, width*height
            for index in range(len(_path_to_videos)):
                frames_dir = _path_to_videos[index]
                # check_video_file(frames_dir, index)
                pool.apply_async(check_video_file, args=(frames_dir, index), callback=callback)

            pool.close()
            pool.join()
            callback.close()

            success_list, fail_list = callback.get_outputs()
            if len(fail_list) > 0:
                filenames = [fail_output[1] for fail_output in fail_list]
                with open('fail_list.txt', 'w') as f:
                    for filename in filenames:
                        f.write(f'{filename}\n') 
                raise Exception("Failed to read some video files. Check fail_list.txt.")

            video_sizes = [success_output[0] for success_output in success_list]
            indices = [success_output[2] for success_output in success_list]

            sort_key = [f"{length:04d}{height*width:06d}" for length, height, width in video_sizes]
            sorting_ = list(zip(indices, video_sizes, sort_key))
            sorting_.sort(key=lambda x: x[-1])
            
            sorted_indices = [x[0] for x in sorting_]
            self._video_names = [_video_names[index] for index in sorted_indices]
            self._labels = [_labels[index] for index in sorted_indices]

            video_sizes = [x[1] for x in sorting_]
            self._video_lengths = [x[0] for x in video_sizes]
            self._video_heights = [x[1] for x in video_sizes]
            self._video_widths = [x[2] for x in video_sizes]
            
            cache_dict = {'_labels': self._labels, '_video_names': self._video_names,
                          '_video_lengths': self._video_lengths, '_video_heights': self._video_heights, '_video_widths': self._video_widths}
            torch.save(cache_dict, cache_file)
            loaded_from = videos_dir

        else:
            while True:
                try:
                    cache_dict = torch.load(cache_file)
                    break

                except:
                    time.sleep(5) # wait until master_proc finished building cache.

            self._labels = cache_dict['_labels']
            self._video_names = cache_dict['_video_names']
            self._video_lengths = cache_dict['_video_lengths']
            self._video_heights = cache_dict['_video_heights']
            self._video_widths = cache_dict['_video_widths']
            loaded_from = cache_file

        self._path_to_videos = [osp.join(videos_dir, f"{name}") for name in self._video_names]

        # Extend self when self._num_clips > 1 (during testing).
        self._labels = list(
            chain.from_iterable([[x] * self._num_clips for x in self._labels])
        )
        self._path_to_videos = list(
            chain.from_iterable(
                [[x] * self._num_clips for x in self._path_to_videos]
            )
        )
        self._video_names = list(
            chain.from_iterable(
                [[x] * self._num_clips for x in self._video_names]
            )
        )
        self._video_lengths = list(
            chain.from_iterable(
                [[x] * self._num_clips for x in self._video_lengths]
            )
        )
        self._video_heights = list(
            chain.from_iterable(
                [[x] * self._num_clips for x in self._video_heights]
            )
        )
        self._video_widths = list(
            chain.from_iterable(
                [[x] * self._num_clips for x in self._video_widths]
            )
        )

        self._spatial_temporal_idx = list(
            chain.from_iterable(
                [
                    range(self._num_clips)
                    for _ in range(len(self._path_to_videos))
                ]
            )
        )

        logger.info(
            "Something-Something V2 dataset constructed "
            " (size: {}) from {}".format(
                len(self._path_to_videos), loaded_from
            )
        )
    
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
        if seed is None:
            seed = random.randint(0, 4294967295)
        
        np.random.seed(seed)
        random.seed(seed)

        label = self._labels[index]
        video_name = self._video_names[index]
        video_path = self._path_to_videos[index]
        video_length = self._video_lengths[index]
        video_height = self._video_heights[index]
        video_width = self._video_widths[index]

        crop_size = self.cfg.DATA.CROP_SIZE

        if self.mode in ["train", "val"]: #or self.cfg.MODEL.ARCH in ['resformer', 'vit']:
            spatial_sample_index = -1 if self.mode == 'train' else 1
            
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0] if self.mode == 'train' else crop_size
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1] if self.mode == 'train' else crop_size
            
        elif self.mode in ["test"]:
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                (
                    self._spatial_temporal_idx[index]
                    % self.cfg.TEST.NUM_ENSEMBLE_SPATIAL
                )
                if self.cfg.TEST.NUM_ENSEMBLE_SPATIAL > 1
                else 1
            )
            min_scale, max_scale = (
                [crop_size] * 2
            )
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expected to be the same.
            assert len({min_scale, max_scale}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )

        num_frames = self.cfg.DATA.NUM_FRAMES
        seg_size = float(video_length - 1) / num_frames
        seq = []

        for i in range(num_frames):
            start = int(np.round(seg_size * i))
            end = int(np.round(seg_size * (i + 1)))
            if self.mode == "train":
                seq.append(random.randint(start, end))
            else:
                seq.append((start + end) // 2)

        # frames are read in self.cfg.DATA.CHANNEL_STANDARD
        frames = dataset_utils.retry_load_images(
            [osp.join(video_path, f'{video_name}_{frame:06d}.jpeg') for frame in seq],
            self._num_retries)

        if self.cfg.DATA.TRAIN_RAND_AUGMENT and self.mode == "train":
            # Transform to PIL Image
            frames = [transforms.ToPILImage()(frame.squeeze().numpy()) for frame in frames]

            # Perform RandAugment
            img_size_min = crop_size
            auto_augment_desc = "rand-m20-mstd0.5-inc1"
            aa_params = dict(
                translate_const=int(img_size_min * 0.45),
                img_mean=tuple([min(255, round(255 * x)) for x in self.cfg.DATA.MEAN]),
            )

            frames = [autoaugment.rand_augment_transform(
                auto_augment_desc, aa_params, seed)(frame) for frame in frames]

            # To Tensor: T H W C
            frames = [torch.tensor(np.array(frame)) for frame in frames]
            frames = torch.stack(frames)

        frames = dataset_utils.tensor_normalize(
            frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
        )

        frames = rearrange(frames, 't h w c -> t c h w')

        # Perform data augmentation.
        frames = dataset_utils.spatial_sampling(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
            random_horizontal_flip=self.cfg.DATA.TRAIN_RANDOM_FLIP,
            inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            seed=seed,
        )

        if self.cfg.DATA.TRAIN_COLORJITTER and self.mode == "train":
            # dataset_utils.frames_augmentation requires 'bgr' standard.
            if self.cfg.DATA.CHANNEL_STANDARD == 'rgb':
                frames = dataset_utils.reverse_color_channels(frames, channel_dim=1)

            frames = dataset_utils.frames_augmentation(
                frames,
                colorjitter=True,
                use_grayscale=False,
                use_gaussian=False,
                seed=seed
            )

            if self.cfg.DATA.CHANNEL_STANDARD == 'rgb':
                frames = dataset_utils.reverse_color_channels(frames, channel_dim=1)

        if self.cfg.DATA.CHANNEL_STANDARD == 'bgr':
            frames = dataset_utils.reverse_color_channels(frames, channel_dim=1)

        frames = rearrange(frames, 't c h w -> c t h w')
        original_shape = torch.as_tensor([video_length, video_height, video_width])

        return frames, label, {"video_name": video_name, "video_index": index, "original_shape": original_shape,
                               "spatial_sample_index": spatial_sample_index, "seed": seed}
    
    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)