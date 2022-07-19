# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import os.path as osp
from itertools import chain as chain
import random
import time
import copy
from multiprocessing import Pool

import av
import numpy as np
import torch
from torch.utils.data import Dataset
from fvcore.common.file_io import PathManager
from einops import rearrange
from tqdm import tqdm
from torchvision import transforms

from .. import autoaugment
from .. import transform
from ...utils import logs
from ...utils import distributed as du
from .. import utils as dataset_utils

from . import decoder as decoder

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

def check_video_file(videofile, index):
    video_size = None
    try:
        video_container = av.open(videofile)
        if video_container is not None:
            length = video_container.streams.video[0].frames
            height = video_container.streams.video[0].height
            width = video_container.streams.video[0].width

            video_size = (length, height, width)
    except:
        pass
    
    return video_size, videofile, index

@DATASET_REGISTRY.register()
class Kinetics_mp4(Dataset):
    """
    Kinetics video loader. Construct the Kinetics video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, cfg, mode, num_retries=10):
        """
        Construct the Kinetics video loader with a given csv file. The format of
        the csv file is:
        ```
        label, youtube_id, time_start, time_end, split, is_cc
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Kinetics".format(mode)
        self.mode = mode
        self.cfg = cfg

        self._num_retries = num_retries
        # For training or validation mode, one single clip is sampled from every video.
        # For testing:
        # NUM_ENSEMBLE_TEMPORAL clips are sampled from every video.
        # For every clip, NUM_ENSEMBLE_SPATIAL is cropped spatially from the frames.
        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (
                cfg.TEST.NUM_ENSEMBLE_TEMPORAL * cfg.TEST.NUM_ENSEMBLE_SPATIAL
            )

        logger.info("Constructing Kinetics {}...".format(mode))
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        if self.mode == 'train':
            csv_filename = "train.csv"
            split_dir = self.cfg.DATA.TRAIN_SPLIT_DIR
        
        elif self.mode == 'val':
            csv_filename = "val.csv"
            split_dir = self.cfg.DATA.VAL_SPLIT_DIR

        else:
            csv_filename = "val.csv"
            split_dir = self.cfg.DATA.TEST_SPLIT_DIR

        videos_dir = osp.join(self.cfg.DATA.PATH_TO_MP4, split_dir)
        cache_dir = self.cfg.DATA.CACHE_DIR
        if not osp.isdir(cache_dir):
            os.mkdir(cache_dir)
        cache_file = osp.join(cache_dir, f'{self.cfg.DATA.CACHE_PREFIX}_{split_dir}.cache')
        
        if du.is_master_proc() and not osp.isfile(cache_file):
            path_to_csv = osp.join(self.cfg.DATA.PATH_TO_ANNOTATION, csv_filename)
            logger.info(f"Start building cache {cache_file}")

            _video_names = []
            _path_to_videos = []
            _labels = []
            
            _text_labels = []
            with PathManager.open(path_to_csv, "r") as f:
                for idx, path_label in enumerate(f.read().splitlines()):
                    if idx == 0:
                        continue
                    
                    text_label, youtube_id, time_start, time_end, split, is_cc = [x.strip() for x in path_label.split(',')]
                    _video_names.append(f"{youtube_id}_{int(time_start):06d}_{int(time_end):06d}")
                    _text_labels.append(text_label)
                        
            assert (
                len(_video_names) > 0
            ), "Failed to load Kinetics split {} from {}".format(
                self.mode, path_to_csv)

            unique_labels = list(set(_text_labels))
            label_dict = {label: idx for idx, label in enumerate(sorted(unique_labels))}
            _labels = [label_dict[text_label] for text_label in _text_labels]

            _path_to_videos = [osp.join(videos_dir, f"{video_name}.mp4") for video_name in _video_names]
            
            pool = Pool(40)
            callback = cache_callback(len(_path_to_videos))
            
            # sort videos by length, width*height
            for index in range(len(_path_to_videos)):
                videofile = _path_to_videos[index]
                pool.apply_async(check_video_file, args=(videofile, index), callback=callback)
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
            
            sort_key = [f"{length:04d}{height*width:07d}{index:07d}" for (length, height, width), index in zip(video_sizes, indices)]
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

        self._path_to_videos = [osp.join(videos_dir, f"{name}.mp4") for name in self._video_names]
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
            "Kinetics dataset constructed "
            " (size: {}) from {}".format(
                len(self._path_to_videos), loaded_from
            )
        )

    def __getitem__(self, index, seed=None):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
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

        if self.mode in ["train", "val"]:
            num_clips = 1
            # -1 indicates random sampling.
            temporal_sample_index = -1 if self.mode == 'train' else 0
            spatial_sample_index = -1 if self.mode == 'train' else 1
            
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0] if self.mode == 'train' else crop_size
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1] if self.mode == 'train' else crop_size
            
        elif self.mode in ["test"]:
            num_clips = self.cfg.TEST.NUM_ENSEMBLE_TEMPORAL
            temporal_sample_index = (
                self._spatial_temporal_idx[index]
                // self.cfg.TEST.NUM_ENSEMBLE_SPATIAL
            )
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

        sampling_rate = self.cfg.DATA.SAMPLING_RATE
        for i_try in range(self._num_retries):
            video_container = None
            try:
                video_container = av.open(self._path_to_videos[index])
                if self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE:
                    # Enable multiple threads for decoding.
                    video_container.streams.video[0].thread_type = "AUTO"
            
                # Select a random video if the current video was not able to access.
                if video_container is None:
                    raise Exception("The video container is empty.")
                
                break

            except Exception as e:
                logger.info(
                    "Failed to load video from {} with error {}".format(
                        self._path_to_videos[index], e
                    )
                )
        
        # Decode video. Perform uniform random sampling of frames.
        frames = decoder.decode(
            video_container,
            sampling_rate,
            self.cfg.DATA.NUM_FRAMES,
            clip_idx=temporal_sample_index,
            num_clips=num_clips, # if clip_idx == -1 (random cliping), this option will be igonored
            target_fps=self.cfg.DATA.TARGET_FPS,
            seed=seed
        )

        if self.cfg.DATA.TRAIN_RAND_AUGMENT and self.mode == "train":
            # Transform to PIL Image
            frames = [transforms.ToPILImage()(frame.squeeze().numpy()) for frame in frames]

            # Perform RandAugment
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

        # Perform color normalization.
        frames = dataset_utils.tensor_normalize(
            frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
        )

        # Permute frames
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