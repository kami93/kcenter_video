# K-centered Patch Sampling for Efficient Video Recognition

PyTorch implementation for K-centered Patch Sampling for Efficient Video Recognition (accepted Poster presentation in ECCV 2022)

<p align="center">
<img width="721" alt="thumbnail" src="https://user-images.githubusercontent.com/20102/179505916-98828802-dca7-4a09-9d4d-1ad4ec20f023.png">
</p>

## Requirements
- `torch>=1.10`
- `torchvision>=0.11`
- 'fvcore' https://github.com/facebookresearch/fvcore/
- 'timm' https://github.com/rwightman/pytorch-image-models
- 'einops'
- 'simplejson'
- 'av (PyAV)' https://github.com/PyAV-Org/PyAV
- 'opencv-python'
- 'tensorboard

## Installing
```
git clone https://github.com/kami93/kcenter_video
cd kcenter_video
python setup.py build develop
```

## Preparing datasets
Download [Kinetics](https://www.deepmind.com/open-source/kinetics) and [Something-Something](https://developer.qualcomm.com/software/ai-datasets/something-something) datasets from the official providers.

For Something-Something dataset, convert every *webm* videos to jpeg frames using video transcoders such as [ffmpeg-python](https://github.com/kkroening/ffmpeg-python):
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
        iio.imwrite(f'filename_{idx:06d}'.jpeg, image)
)
```

For Kinetics dataset, convert every videos to mp4 resized to 256p (short-side) using scripts such as [mmaction2-resize_videos](https://github.com/open-mmlab/mmaction2/blob/master/tools/data/resize_videos.py).

## Training on Kinetics
```
python tools/run.py --cfg ./kcenter_transformer/configs/kcenter_models/kcenter_vit_kinetics.yaml
```


## Training on Something-Something V2
```
python tools/run.py --cfg ./kcenter_transformer/configs/kcenter_models/kcenter_vit_ssv2.yaml
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
