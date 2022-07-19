# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from setuptools import find_packages, setup

setup(
    name="kcenter_transformer",
    version="1.0",
    author="SH",
    url="https://github.com/kami93/kcenter_video",
    description="K-centered Patch Sampling for Efficient Video Recognition",
    keywords = [
    'artificial intelligence'
    ],
    packages=find_packages(exclude=("dataset", "tools")),
)
