# K-centered Patch Sampling for Efficient Video Recognition

PyTorch implementation for K-centered Patch Sampling for Efficient Video Recognition (accepted Poster presentation in ECCV 2022)

by [Seong Hyeon Park](https://shpark.org/), [Jihoon Tack](https://jihoontack.github.io/), [Byeongho Heo](https://sites.google.com/view/byeongho-heo/home), [Jung-Woo Ha](https://aidljwha.wordpress.com/) and [Jinwoo Shin](https://alinlab.kaist.ac.kr/shin.html)

<p align="center">
<img width="721" alt="thumbnail" src="https://user-images.githubusercontent.com/20102/179505916-98828802-dca7-4a09-9d4d-1ad4ec20f023.png">
</p>

## Requirements
- `torch>=1.10`
- `torchvision>=0.11`
- `einops`
- `simplejson`
- `opencv-python`
- `tensorboard`
- [fvcore](https://github.com/facebookresearch/fvcore/)
- [timm](https://github.com/rwightman/pytorch-image-models)
- [av (PyAV)](https://github.com/PyAV-Org/PyAV)

## Installing
```
git clone https://github.com/kami93/kcenter_video
cd kcenter_video
python setup.py build develop
```

## Preparing datasets
Download [Kinetics](https://www.deepmind.com/open-source/kinetics) and [Something-Something](https://developer.qualcomm.com/software/ai-datasets/something-something) datasets from the official providers.

For Kinetics dataset, convert every videos to mp4 resized to 256p (short-side) using scripts such as [mmaction2-resize_videos](https://github.com/open-mmlab/mmaction2/blob/master/tools/data/resize_videos.py). The filename should follow the rule: *`{youtube_id}_{time_start:06d}_{time_end:06d}.mp4`*

For Something-Something dataset, convert every *webm* videos to jpeg frames using video transcoders such as [ffmpeg-python](https://github.com/kkroening/ffmpeg-python). The filename should follow the rule: *`{video_id}/{video_id}_{frame_index:06d}.jpeg`*

An example Python script for the conversion is as follows:
```python
import ffmpeg
import numpy as np
import imageio as iio

(
    probe = ffmpeg.probe(video_file)
    stream_dict = probe['streams'][0]
    width, height = stream_dict['width'], stream_dict['height']
    
    out, _ = ffmpeg
    .input('filename.webm')
    .output('pipe:', format='rawvideo', pix_fmt='rgb24', v='trace')
    .run(capture_stdout=True, capture_stderr=True)
    
    frames = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
    for idx, image in enumerate(frames):
        iio.imwrite(f'filename_{idx:06d}.jpeg', image)
)
```

After the conversion, an example filetree is as follows:

```
# SSv2
somethingv2
    |-annotation
    |-webm/train_test
        |-"1.webm"
        ...
        |-"220847.webm"

        (220,847 files)
    |-jpeg/train_test
        |-"1"
            |-"1_000000.jpeg"
            ...
            |-"1_000046.jpeg"
            (30 files)

        ...
        |"220847"
            |-"220847_000000.jpeg"
            ...
            |-"220847_000029.jpeg"
            (47 files)

        (220,847 directories)

# Kinetics
kinetics
    |-annotation
    |-train
        |-"---QUuC4vJs_000084_000094.mp4"
        ...
        |-"zzzzE0ncP1Y_000232_000242.mp4"

        (241,199 files)
    |-val
        |-"--07WQ2iBlw_000001_000011.mp4"
        ...
        |-"zzokQ8DKETs_000112_000122.mp4"

        (19,877 files)
```

## Training Models

Example scripts for training models are as follows:

### Kinetics
```bash
PATH_TO_MP4='kinetics'
PATH_TO_ANNOTATION='kinetics/annotation'
TRAIN_SPLIT_DIR='train'
VAL_SPLIT_DIR='val'
OUTPUT_DIR='training_outputs'
BATCH_SIZE=4
NUM_GPUS=4

python tools/run.py --cfg kcenter_transformer/configs/kcenter_models/{model_name}_kinetics.yaml \
TRAIN.ENABLE True \
TRAIN.BATCH_SIZE $BATCH_SIZE \
DATA.PATH_TO_MP4 $PATH_TO_MP4 \
DATA.PATH_TO_ANNOTATION $PATH_TO_ANNOTATION \
DATA.TRAIN_SPLIT_DIR $TRAIN_SPLIT_DIR \
DATA.VAL_SPLIT_DIR $VAL_SPLIT_DIR \
OUTPUT_DIR $OUTPUT_DIR \
NUM_GPUS $NUM_GPUS
```

### Something-Something V2
```bash
PATH_TO_JPEG='somethingv2/jpeg_data'
PATH_TO_ANNOTATION='somethingv2/annotation'
TRAIN_SPLIT_DIR='train_test'
VAL_SPLIT_DIR='train_test'
OUTPUT_DIR='training_outputs'
BATCH_SIZE=4
NUM_GPUS=4

python tools/run.py --cfg kcenter_transformer/configs/kcenter_models/{model_name}_ssv2.yaml \
TRAIN.ENABLE True \
TRAIN.BATCH_SIZE $BATCH_SIZE \
DATA.PATH_TO_JPEG $PATH_TO_JPEG \
DATA.PATH_TO_ANNOTATION $PATH_TO_ANNOTATION \
DATA.TRAIN_SPLIT_DIR $TRAIN_SPLIT_DIR \
DATA.VAL_SPLIT_DIR $VAL_SPLIT_DIR \
OUTPUT_DIR $OUTPUT_DIR \
NUM_GPUS $NUM_GPUS
```

BATCH_SIZE and NUM_GPUS should be set appropriately depending on your system environment. Note that the global batch size is preset in each config yaml files so that 
"GLOBAL_BATCH_SIZE=BATCH_SIZExNUM_GPUSxNUM_SHARDSxNUM_ACCUMULATION".

## Testing Models

Example scripts for testing models are as follows:

### Kinetics
```bash
PATH_TO_MP4='kinetics'
PATH_TO_ANNOTATION='kinetics/annotation'
TEST_SPLIT_DIR='val'
OUTPUT_DIR='training_outputs'
BATCH_SIZE=4
NUM_GPUS=4

python tools/run.py --cfg kcenter_transformer/configs/kcenter_models/{model_name}_kinetics.yaml \
TEST.ENABLE True \
TEST.BATCH_SIZE $BATCH_SIZE \
DATA.PATH_TO_MP4 $PATH_TO_MP4 \
DATA.PATH_TO_ANNOTATION $PATH_TO_ANNOTATION \
DATA.TEST_SPLIT_DIR $TEST_SPLIT_DIR \
OUTPUT_DIR $OUTPUT_DIR \
NUM_GPUS $NUM_GPUS
```

### Something-Something V2
```bash
PATH_TO_JPEG='somethingv2/jpeg_data'
PATH_TO_ANNOTATION='somethingv2/annotation'
TEST_SPLIT_DIR='train_test'
OUTPUT_DIR='training_outputs'
BATCH_SIZE=4
NUM_GPUS=4

python tools/run.py --cfg kcenter_transformer/configs/kcenter_models/{model_name}_ssv2.yaml \
TEST.ENABLE True \
TEST.BATCH_SIZE $BATCH_SIZE \
DATA.PATH_TO_JPEG $PATH_TO_JPEG \
DATA.PATH_TO_ANNOTATION $PATH_TO_ANNOTATION \
DATA.TEST_SPLIT_DIR $TEST_SPLIT_DIR \
OUTPUT_DIR $OUTPUT_DIR \
NUM_GPUS $NUM_GPUS
```

## Acknowledgement
Our code base is built partly upon the packages: 
<a href="https://github.com/facebookresearch/SlowFast">PySlowFast</a>, <a href="https://github.com/facebookresearch/TimeSformer">TimeSformer</a>, <a href="https://github.com/facebookresearch/Motionformer">Motionformer</a>, <a href="https://github.com/rwightman/pytorch-image-models">PyTorch Image Models</a> and <a href="https://github.com/1adrianb/video-transformers">XViT</a>


## Citation
If you use this code for your research, please cite our papers.
```
@InProceedings{Park_2022_ECCV,
    author    = {Park, Seong Hyeon and Tack, Jihoon and Heo, Byeongho and Ha, Jung-Woo and Shin, Jinwoo},
    title     = {K-centered Patch Sampling for Efficient Video Recognition},
    booktitle = {European Conference on Computer Vision (ECCV)},
    month     = {October},
    year      = {2022}
}
```
