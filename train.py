#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: train.py
import matplotlib
matplotlib.use('Agg')
import os
import argparse
import cv2
import shutil
import itertools
import tqdm
import math
import numpy as np
import json
import tensorflow as tf
import zipfile
import pickle
from glob import glob

from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils import optimizer
import tensorpack.utils.viz as tpviz
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.dataflow import (
    DataFromGenerator, MapData, imgaug, AugmentImageComponent, TestDataSpeed, MultiProcessMapData,
    MapDataComponent, DataFromList, PrefetchDataZMQ, BatchData)

from resnet_model import (
    preresnet_group, preresnet_basicblock, preresnet_bottleneck,
    resnet_group, resnet_basicblock, resnet_bottleneck, se_resnet_bottleneck,
    resnet_backbone)
from basemodel import (
    image_preprocess, pretrained_resnet_conv4, resnet_conv5)
from model import *
import config
import collections
import ast
import pandas as pd
from utils import *
from tensorflow.python.keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from tensorflow import keras
from custom_utils import ReduceLearningRateOnPlateau

NCSVS = 100

def get_batch_factor():
    nr_gpu = get_nr_gpu()
    assert nr_gpu in [1, 2, 4, 8], nr_gpu
    return 8 // nr_gpu

def get_resnet_model_output_names():
    return ['final_probs', 'final_labels']

def image_generator(csv_file, label_map, batchsize=config.BATCH, image_size=config.IMAGE_SIZE, lw=6):
    while True:
        for df in pd.read_csv(csv_file, chunksize=batchsize):
            df['drawing'] = df['drawing'].apply(ast.literal_eval)
            x = [] #np.zeros((len(df), size, size))
            y = []
            for i, raw_strokes in enumerate(df.drawing.values):
                #x[i] = draw_cv2(raw_strokes, size=size, lw=lw)
                x.append(drawing2tensor(list2drawing(raw_strokes, size=image_size, lw=lw)))
            y = [label_map[x] for x in df.word.values]
            x = np.array(x)
            y = np.array(y)
            yield [x, y]

def get_train_dataflow():
    train_datagen = image_generator_xd(size=config.IMAGE_SIZE, batchsize=config.BATCH, ks=range(NCSVS - 1))
    ds = DataFromGenerator(train_datagen)
    #ds = PrefetchDataZMQ(ds, 6)
    return ds

def get_eval_dataflow():
    valid_df = pd.read_csv(os.path.join('/data/kaggle/doodle/shuffle_csv2/train_k{}.csv.gz'.format(NCSVS - 1)), nrows=34000)
    x_valid = df_to_image_array_xd(valid_df, size)
    y_valid = keras.utils.to_categorical(valid_df.y, num_classes=config.NUM_CLASS)
    return x_valid, y_valid
        
class ResnetModel(ModelDesc):
    def _get_inputs(self):
        ret = [
            InputDesc(tf.float32, (None, None, None, 1), 'image'),
            InputDesc(tf.int32, (None, config.NUM_CLASS), 'labels'),
        ]
        return ret

    def _build_graph(self, inputs):
        is_training = get_current_tower_context().is_training
        image, label = inputs
        tf.summary.image('viz', image, max_outputs=10)
        #image = image_preprocess(image, bgr=True)
        image = image * (1.0 / 255)
        image = tf.transpose(image, [0, 3, 1, 2])
        
        depth = 50
        basicblock = preresnet_basicblock if config.RESNET_MODE == 'preact' else resnet_basicblock
        bottleneck = {
            'resnet': resnet_bottleneck,
            'preact': preresnet_bottleneck,
            'se': se_resnet_bottleneck}[config.RESNET_MODE]
        num_blocks, block_func = {
            18: ([2, 2, 2, 2], basicblock),
            34: ([3, 4, 6, 3], basicblock),
            50: ([3, 4, 6, 3], bottleneck),
            101: ([3, 4, 23, 3], bottleneck),
            152: ([3, 8, 36, 3], bottleneck)
        }[depth]
        logits = get_logit(image, num_blocks, block_func)
        """
        conv4 = pretrained_resnet_conv4(image, config.RESNET_NUM_BLOCK[:3])
        # B*c*h*w
        conv5 = resnet_conv5(conv4, config.RESNET_NUM_BLOCK[-1])
        
        # train head only
        logits = cls_head('cls', conv5)
        """
        if is_training:
            loss = cls_loss(logits, label)
            wd_cost = regularize_cost(
                '.*/W',
                l2_regularizer(1e-4), name='wd_cost')

            self.cost = tf.add_n([
                loss, wd_cost], 'total_cost')                

            add_moving_summary(self.cost, wd_cost)
        else:
            final_prob = tf.nn.softmax(logits)
            tf.identity(final_prob, name='final_probs')
            values, indices = tf.nn.top_k(final_prob, k=3)
            final_label = tf.identity(indices, name='final_labels')
    
    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.01, trainable=False)
        tf.summary.scalar('learning_rate', lr)

        factor = get_batch_factor()
        if factor != 1:
            lr = lr / float(factor)
            opt = tf.train.MomentumOptimizer(lr, 0.9)
            opt = optimizer.AccumGradOptimizer(opt, factor)
        else:
            opt = tf.train.MomentumOptimizer(lr, 0.9)
        opt = tf.train.AdamOptimizer(lr)
        #opt = tf.train.MomentumOptimizer(lr, 0.9)
        return opt

class ResnetEvalCallback(Callback):
    
    def _setup_graph(self):
        self.pred = self.trainer.get_predictor(
            ['image'],
            get_resnet_model_output_names())
        valid_df = pd.read_csv(os.path.join('/data/kaggle/doodle/shuffle_csv2/train_k{}.csv.gz'.format(NCSVS - 1)), nrows=34000)
        x_valid = df_to_image_array_xd(valid_df, config.IMAGE_SIZE)
        y_valid = keras.utils.to_categorical(valid_df.y, num_classes=config.NUM_CLASS)
        self.data = [x_valid, y_valid]
        self.valid_df = valid_df
    def _eval(self):
        from tensorpack.utils.utils import get_tqdm_kwargs
        score = 0.0
        ind = 0.0
        #x_valid, y_valid = get_eval_dataflow()
        #valid_df = pd.read_csv(os.path.join('/data/kaggle/doodle/shuffle_csv/train_k{}.csv.gz'.format(NCSVS - 1)), nrows=34000)
        #x_valid = df_to_image_array_xd(valid_df, config.IMAGE_SIZE)
        #y_valid = keras.utils.to_categorical(valid_df.y, num_classes=config.NUM_CLASS)
        x_valid, y_valid = self.data
        valid_predictions = []
        with tqdm.tqdm(total=len(x_valid) // config.INFERENCE_BATCH + 1, **get_tqdm_kwargs()) as pbar:
            start = 0
            end = 0
            for i in range(len(x_valid) // config.INFERENCE_BATCH + 1):
                start = i * config.INFERENCE_BATCH
                end = start + config.INFERENCE_BATCH if start + config.INFERENCE_BATCH < len(x_valid) else len(x_valid)
                x = x_valid[start:end]
                final_probs, final_labels = self.pred(x)
                valid_predictions.extend(final_probs)
                #score += mapk(la, final_labels)
                pbar.update()
        valid_predictions = np.array(valid_predictions)
        map3 = mapk(self.valid_df[['y']].values, preds2catids(valid_predictions).values)
        print('Map3: {:.3f}'.format(map3))
        #loss = tf.get_default_graph().get_tensor_by_name("tower0/cls_loss/label_loss:0")
        #print("train loss: ", loss.eval())
        self.trainer.monitors.put_scalar("Map3", map3)

    def _trigger_epoch(self):
        #if self.epoch_num > 0 and self.epoch_num % 5 == 0:
        self._eval()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--logdir', help='logdir', default='train_log/fastrcnn')
    parser.add_argument('--datadir', help='override config.BASEDIR')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--evaluate', help='path to the output json eval file')
    parser.add_argument('--predict', help='path to the input image file')
    parser.add_argument('--lr_find', action='store_true')
    parser.add_argument('--cyclic', action='store_true')
    parser.add_argument('--auto_reduce', action='store_true')
    args = parser.parse_args()
    if args.datadir:
        config.BASEDIR = args.datadir

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.visualize or args.evaluate or args.predict:
        # autotune is too slow for inference
        os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'

        assert args.load
        print_config()

        if args.visualize:
            visualize(args.load)
#            imgs = [img['file_name'] for img in imgs]
#            predict_many(pred, imgs)
        else:
            if args.evaluate:
                if config.RESNET:
                    pred = OfflinePredictor(PredictConfig(
                        model=ResnetModel(),
                        session_init=get_model_loader(args.load),
                        input_names=['image'],
                        output_names=get_resnet_model_output_names()))
                    
                    valid_df = pd.read_csv(os.path.join('/data/kaggle/doodle/shuffle_csv2/train_k{}.csv.gz'.format(NCSVS - 1)), nrows=34000)
                    x_valid = df_to_image_array_xd(valid_df, config.IMAGE_SIZE)
                    y_valid = keras.utils.to_categorical(valid_df.y, num_classes=config.NUM_CLASS)
                    valid_predictions = []
                    with tqdm.tqdm(total=len(x_valid) // config.INFERENCE_BATCH + 1) as pbar:
                        start = 0
                        end = 0
                        for i in range(len(x_valid) // config.INFERENCE_BATCH + 1):
                            start = i * config.INFERENCE_BATCH
                            end = start + config.INFERENCE_BATCH if start + config.INFERENCE_BATCH < len(x_valid) else len(x_valid)
                            x = x_valid[start:end]
                            final_probs, final_labels = pred(x)
                            valid_predictions.extend(final_probs)
                            pbar.update()
                    valid_predictions = np.array(valid_predictions)
                    map3 = mapk(valid_df[['y']].values, preds2catids(valid_predictions).values)
                    print('Map3: {:.3f}'.format(map3))
            elif args.predict:
                if config.RESNET:
                    pred = OfflinePredictor(PredictConfig(
                        model=ResnetModel(),
                        session_init=get_model_loader(args.load),
                        input_names=['image'],
                        output_names=get_resnet_model_output_names()))
                    size = config.IMAGE_SIZE
                    lw = 6
                    predictions = []
                    test = pd.read_csv(os.path.join(config.BASEDIR, 'test_simplified.csv'))
                    x_test = df_to_image_array_xd(test, config.IMAGE_SIZE)
                    with tqdm.tqdm(total=(len(x_test)) // config.INFERENCE_BATCH + 1) as pbar:
                        start = 0
                        end = 0
                        for i in range(len(x_test) // config.INFERENCE_BATCH + 1):
                            start = i * config.INFERENCE_BATCH
                            end = start + config.INFERENCE_BATCH if start + config.INFERENCE_BATCH < len(x_test) else len(x_test)
                            x = x_test[start:end]
                            final_probs, final_labels = pred(x)
                            predictions.extend(final_probs)
                            pbar.update()
                        predictions = np.array(predictions)
                    
                    top3 = preds2catids(predictions)
                    cats = list_all_categories()
                    id2cat = {k: cat.replace(' ', '_') for k, cat in enumerate(cats)}
                    top3cats = top3.replace(id2cat)
                    test['word'] = top3cats['a'] + ' ' + top3cats['b'] + ' ' + top3cats['c']
                    submission = test[['key_id', 'word']]
                    submission.to_csv('submission.csv', index=False)

    else:
       
        if args.lr_find:
            base_lr = 0.00001
            max_lr = 0.1
            stepnum = 1000
            max_epoch = 200 # run max_epoch to find lr
            schedule = [(0, base_lr)]
            for e in range(1, max_epoch):
                offset = (max_lr-base_lr)/(max_epoch-1)
                schedule.append((e, base_lr+offset*e))
            LR_RANGE_TEST_SCHEDULE = ScheduledHyperParamSetter('learning_rate', schedule)
            TRAINING_SCHEDULE = LR_RANGE_TEST_SCHEDULE
        elif args.cyclic:
            from custom_utils import CyclicLearningRateSetter
            if config.RESNET:
                base_lr = 0.0001
                max_lr = 0.004
                step_size = (180000 // 64) * 2
            stepnum = 5000 # step to save model and eval
            max_epoch = 20 # how many cycle / 4 = 5 cycle (2*step_size = 1 cycle)
            CYCLIC_SCHEDULE = CyclicLearningRateSetter('learning_rate', base_lr=base_lr, max_lr=max_lr, step_size=step_size)
            TRAINING_SCHEDULE = CYCLIC_SCHEDULE
        elif args.auto_reduce:
            stepnum = 800
            base_lr = 2e-3
            min_lr = 1e-5
            max_epoch = 120
            TRAINING_SCHEDULE = ReduceLearningRateOnPlateau('learning_rate', 
                                        factor=0.1, patience=5, 
                                        base_lr=base_lr, min_lr=min_lr, window_size=800)
        else:
            # heuristic setting for baseline
            if config.RESNET:
                stepnum = 800
                max_epoch = 120
                TRAINING_SCHEDULE = ScheduledHyperParamSetter('learning_rate', [(0, 2e-3), (40, 1e-3), (80, 1e-4)])

        #==========LR Range Test===============#
        if config.RESNET:
            logger.set_logger_dir(args.logdir)
            print_config()

            cfg = TrainConfig(
                model=ResnetModel(),
                data=QueueInput(get_train_dataflow()),
                callbacks=[
                    ModelSaver(max_to_keep=10, keep_checkpoint_every_n_hours=1),
                    TRAINING_SCHEDULE,
                    ResnetEvalCallback(),
                    GPUUtilizationTracker(),
                ],
                steps_per_epoch=stepnum,
                max_epoch=max_epoch,
                session_init=get_model_loader(args.load) if args.load else None,
            )
            trainer = SyncMultiGPUTrainerReplicated(get_nr_gpu(), mode='nccl')
            launch_train_with_config(cfg, trainer)