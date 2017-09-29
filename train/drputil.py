import cv2
import numpy as np
import os
import pickle
import sys
import yaml
import tables
import srperm as srp


class Bbox:
    def __init__(self, x, y, w, h):
        self.x = x # first col
        self.y = y # first row
        self.w = w # width
        self.h = h # height
    def __repr__(self):
        return str(self)
    def __str__(self):
        return "bbox(%i,%i)[%i,%i]" % (self.x, self.y, self.w, self.h)
    

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def float64_feature(value):
    return tf.train.Feature(float64_list=tf.train.Float64List(value=[value]))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def getPatchBbox(source_config, target_config, camera):
    """
    Currently we assume that orientations and positions are identical
    """
    
    patch = target_config['patch'][camera]
    frame = source_config['camera'][camera]
    
    hfov_ratio = patch['hfov'] / frame['hfov']
    vfov_ratio = patch['vfov'] / frame['vfov']

    width = frame['width'] * hfov_ratio
    height = frame['height'] * vfov_ratio
    x = (frame['width'] - width) // 2
    y = frame['height'] - height

    return Bbox(x, y, width, height)


def getPatchSize(target_config, camera):
    patch = target_config['patch'][camera]
    return patch['width'], patch['size']


def cropImage(image, bbox):
    crop = image[bbox.y : bbox.y + bbox.h, bbox.x : bbox.x + bbox.w]
    return crop

def resizeImage(image, size):
    return cv2.resize(image, size)
        

def readState(path):
    """
    Read thhrough the state file and get our timestamps and recorded values.
    Returns the non-timestamp headers, timestamps as a double array, and
    all non-timestamp values in one 2D float32 array.
    """
    timestamps = []
    states = []
    with open(path) as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            states.append([float(x) for x in row])
            
    timestamps_arr = np.array(timestamps, dtype=np.double)
    states_arr = np.array(states, dtype=np.float32)

    return headers[1:], timestamps_arr, states_arr

