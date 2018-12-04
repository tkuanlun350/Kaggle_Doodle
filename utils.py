import cv2
import numpy as np
import pandas as pd
from tensorflow.python.keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from tensorpack.utils import logger
import config
import os
from tensorflow import keras
import ast
import math

def draw_raw(drawing, H, W):
    point_xy = []
    point_v = []
    for d in drawing:
        x_r, y_r, t_r = np.array(d)
        d_s = ((x_r[1:]-x_r[:-1])**2 + (y_r[1:]-y_r[:-1])**2 )**0.5
        delta_t = t_r[1:]-t_r[:-1]
        v_r = d_s / (delta_t + 1e-6)
        v_r = np.insert(v_r, 0, 0)
        v_r = v_r / (np.max(v_r) + 1e-6)
        point_xy.append(np.dstack([x_r, y_r])[0])
        point_v.append(v_r)
   
    point = np.concatenate(point_xy).astype(np.float32)
    velocity  = np.concatenate(point_v ).astype(np.float32)

    #--------
    image  = np.full((H,W),0,np.uint8)
    image_1 = np.full((H,W),0,np.uint8)
    image_2 = np.full((H,W),0,np.uint8)
    x_max = point[:,0].max()
    x_min = point[:,0].min()
    y_max = point[:,1].max()
    y_min = point[:,1].min()
    w = x_max-x_min
    h = y_max-y_min

    s = max(w,h)
    norm_point = (point-[x_min,y_min])/s
    norm_point = (norm_point-[w/s*0.5,h/s*0.5])*max(W,H)*0.85
    norm_point = np.floor(norm_point + [W/2,H/2]).astype(np.int32)

    #--------
    
    # random drop stroke
    for i in range(point.shape[0]-1):
        if config.AUG_TRAIN:
            if np.random.rand() > 0.8:
                continue

        x0,y0 = norm_point[i]
        x1,y1 = norm_point[i+1]
        v = velocity[i+1]
        if math.isnan(v):
            v = 0
        color_v = int(v * 255)
        color_t = min(i, 255)
        cv2.line(image,(x0,y0),(x1,y1),(255), config.LW,cv2.LINE_AA)
        cv2.line(image_1,(x0,y0),(x1,y1),(color_v),config.LW,cv2.LINE_AA)
        cv2.line(image_2,(x0,y0),(x1,y1),(color_t),config.LW,cv2.LINE_AA)
    
    if config.SEQ:
        im = np.dstack([image, image_1, image_2])
        if config.AUG_TRAIN:
            if np.random.rand() > 0.5:
                im = np.fliplr(im)
        return im
    else:
        return image

def draw_cv2_seq(raw_strokes, size=256, lw=6, time_color=True, padding=config.PADDING):
    img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
    imgs = []
    pad = 10
    gap = len(raw_strokes) // 3
    for st in range(3):
        start = st * gap
        end = start + gap if start + gap < len(raw_strokes) and st < 2 else len(raw_strokes)
        img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
        for t, stroke in enumerate(raw_strokes[:end]):
            for i in range(len(stroke[0]) - 1):
                #color = 255 - min(t, 10) * 13 if time_color else 255
                color = 255
                _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                             (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
        imgs.append(img)
    imgs.append(np.clip((imgs[0] + imgs[1]), 0, 255))
    imgs.append(np.clip((imgs[1] + imgs[2]), 0, 255))
    imgs.append(np.clip((imgs[0] + imgs[1] + imgs[2]), 0, 255))
    if padding:
        imgs = [cv2.resize(img, (size-pad*2, size-pad*2), interpolation=cv2.INTER_NEAREST) for img in imgs]
        imgs = [np.pad(img, ((pad, pad), (pad, pad)), 'constant') for img in imgs]
        return np.array(imgs).transpose((1,2,0))
    img = np.array(imgs).transpose((1,2,0))
    if size != BASE_SIZE:
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
    else:
        return img

def draw_cv2(raw_strokes, size=256, lw=6, time_color=True, padding=config.PADDING, is_training=True):
    img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
    pad = 10
    for t, stroke in enumerate(raw_strokes):
        for i in range(len(stroke[0]) - 1):
            color = 255 - min(t, 10) * 13 if config.TIME_COLOR else 255
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                         (stroke[0][i + 1], stroke[1][i + 1]), color, lw, cv2.LINE_AA) #cv2.LINE_AA
    if padding:
        img = cv2.resize(img, (size-pad*2, size-pad*2), interpolation=cv2.INTER_LINEAR)
        img = np.pad(img, ((pad, pad), (pad, pad)), 'constant')
        return img

    if config.AUG_TRAIN and is_training:
            if np.random.rand() > 0.5:
                img = np.fliplr(img)

    if size != BASE_SIZE:
        return cv2.resize(img, (size, size))
    else:
        return img
        
def image_generator_xd(size, batchsize, ks, lw=6, time_color=True):
    while True:
        for k in np.random.permutation(ks):
            filename = os.path.join('/data/kaggle/doodle/{}/train_k{}.csv.gz'.format(config.FOLD, k))
            for df in pd.read_csv(filename, chunksize=batchsize):
                if not config.RAW:
                    df['drawing'] = df['drawing'].apply(ast.literal_eval)
                if config.SEQ:
                    x = np.zeros((len(df), size, size, 3))
                else:
                    x = np.zeros((len(df), size, size, 1))
                for i, raw_strokes in enumerate(df.drawing.values):
                    if config.SEQ and not config.RAW:
                        x[i] = draw_cv2_seq(raw_strokes, size=size, lw=lw,
                                            time_color=time_color, padding=config.PADDING)
                    elif config.RAW:
                        _raw_strokes = eval(raw_strokes)
                        x[i, :, :, :] = draw_raw(_raw_strokes, size, size)
                        
                    else:
                        x[i, :, :, 0] = draw_cv2(raw_strokes, size=size, lw=lw,
                                             time_color=time_color, padding=config.PADDING)
                x = x.astype(np.float32)
                y = keras.utils.to_categorical(df.y, num_classes=config.NUM_CLASS)
                yield x, y

def df_to_image_array_xd(df, size, lw=6, time_color=True):
    if not config.RAW:
        df['drawing'] = df['drawing'].apply(ast.literal_eval)
    if config.SEQ:
        x = np.zeros((len(df), size, size, 3))
    else:
        x = np.zeros((len(df), size, size, 1))
    for i, raw_strokes in enumerate(df.drawing.values):
        if config.SEQ and not config.RAW:
            x[i] = draw_cv2_seq(raw_strokes, size=size, lw=lw, time_color=time_color, padding=config.PADDING)
        elif config.RAW:
            _raw_strokes = eval(raw_strokes)
            x[i, :, :, :] = draw_raw(_raw_strokes, size, size)
        else:
            x[i, :, :, 0] = draw_cv2(raw_strokes, size=size, lw=lw, time_color=time_color, padding=config.PADDING, is_training=False)
    x = x.astype(np.float32)
    return x

def f2cat(filename: str) -> str:
    return filename.split('.')[0]

def list_all_categories():
    files = os.listdir(os.path.join(config.BASEDIR, 'train'))
    return sorted([f2cat(f) for f in files], key=str.lower)

def apk(actual, predicted, k=3):
    """
    Source: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    """
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    if not actual:
        return 0.0
    return score / min(len(actual), k)


def mapk(actual, predicted, k=3):
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

def preds2catids(predictions):
    return pd.DataFrame(np.argsort(-predictions, axis=1)[:, :3], columns=['a', 'b', 'c'])

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def print_config():
    logger.info("Config: ------------------------------------------")
    for k in dir(config):
        if k == k.upper():
            logger.info("{} = {}".format(k, getattr(config, k)))
    logger.info("--------------------------------------------------")

# https://www.kaggle.com/gaborfodor/greyscale-mobilenet-lb-0-892
BASE_SIZE = 256
def list2drawing(raw_strokes, size=256, lw=6, time_color=True):
    img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
    for t, stroke in enumerate(raw_strokes):
        for i in range(len(stroke[0]) - 1):
            color = 255 - min(t, 10) * 13 if time_color else 255
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                         (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
    if size != BASE_SIZE:
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    else:
        return img

def drawing2tensor(drawing):
    return drawing[...,np.newaxis]
    rgb = cv2.cvtColor(drawing,cv2.COLOR_GRAY2RGB)
    #rgb = rgb.transpose(2,0,1).astype(np.float32)
    return rgb