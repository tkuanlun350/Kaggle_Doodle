#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: config.py

import numpy as np

# for new data
NORM = 'BN'
RESNET_MODE = "resnet" #preact
BATCH = 680
INFERENCE_BATCH = 128
IMAGE_SIZE = 64
NUM_CLASS = 340
RESNET = True
# dataset -----------------------
BASEDIR = '/data/kaggle/doodle/data/'
TRAIN_DATASET = 'train'
VAL_DATASET = 'val'
TEST_DATASET = 'test'


# basemodel ----------------------
RESNET_NUM_BLOCK = [3, 4, 6, 3]     # resnet50
#RESNET_NUM_BLOCK = [3, 4, 23, 3]     # resnet101
