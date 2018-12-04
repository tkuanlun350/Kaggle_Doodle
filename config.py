#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: config.py

import numpy as np

# for new data
ACCU = False
RAW = False
TIME_COLOR = True
AUG_TRAIN = True
FOLD = "shuffle_csv2" if not RAW else "shuffle_csv_raw"
FREEZE = False
TTA = False
LW = 2
SEQ = False
PADDING = False
NORM = 'BN' 
RESNET_MODE = "se" #preact
BATCH = 256
INFERENCE_BATCH = 256
IMAGE_SIZE = 144
NUM_CLASS = 340
RESNET = "ResXt" #ResXt
# dataset -----------------------
BASEDIR = '/data/kaggle/doodle/data/'
TRAIN_DATASET = 'train'
VAL_DATASET = 'val'
TEST_DATASET = 'test'


# basemodel ----------------------
RESNET_NUM_BLOCK = [3, 4, 6, 3]     # resnet50
#RESNET_NUM_BLOCK = [3, 4, 23, 3]     # resnet101
