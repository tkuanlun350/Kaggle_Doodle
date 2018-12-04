#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: model.py

import tensorflow as tf
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.tfutils import get_current_tower_context
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.argscope import argscope
from tensorpack.tfutils.scope_utils import under_name_scope
from tensorpack.models import ( 
    MaxPooling, BatchNorm, Conv2DTranspose, BNReLU, Conv2D, FullyConnected, GlobalAvgPooling, layer_register, Deconv2D, Dropout)
from resnet_model import (
    resnet_backbone_dropout, preresnet_group, preresnet_basicblock, preresnet_bottleneck,
    resnet_group, resnet_basicblock, resnet_bottleneck, se_resnet_bottleneck,
    resnet_backbone)
import numpy as np
import config
import math
from tensorflow.python.keras.metrics import categorical_accuracy, top_k_categorical_accuracy

def get_logit(image, num_blocks, block_func):
    with argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm], data_format="NCHW"):
            return resnet_backbone_dropout(
                image, num_blocks,
                preresnet_group if config.RESNET_MODE == 'preact' else resnet_group, block_func)

@under_name_scope()
def cls_loss(label_logits, label):
    with tf.name_scope('cls_label_metrics'):
        label_pred = tf.nn.softmax(label_logits)
        top3_acc = top_k_categorical_accuracy(label, label_pred, k=3)
        add_moving_summary(top3_acc)

    #label_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
    #    labels=label, logits=label_logits)
    label_loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=label, logits=label_logits)
    label_loss = tf.reduce_mean(label_loss, name='label_loss')
    return label_loss

@layer_register(log_shape=True)
def cls_head(feature):
    feature = GlobalAvgPooling('gap', feature, data_format='NCHW')
    
    fc1 = FullyConnected(
        'fc1', feature, 1024,
        W_init=tf.random_normal_initializer(stddev=0.01))
    fc1 = Dropout(fc1)
    fc2 = FullyConnected(
        'fc2', fc1, config.NUM_CLASS,
        W_init=tf.random_normal_initializer(stddev=0.01))
    
    return fc2