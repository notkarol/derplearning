import csv
import cv2
import numpy as np
import os
import pickle
import sys
import yaml

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

class Bbox:
    def __init__(self, x, y, w, h):
        self.x = int(x + 0.5) # first col
        self.y = int(y + 0.5) # first row
        self.w = int(w + 0.5) # width
        self.h = int(h + 0.5) # height
    def __repr__(self):
        return str(self)
    def __str__(self):
        return "bbox(%i,%i)[%i,%i]" % (self.x, self.y, self.w, self.h)

def has_image_ext(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
    

def load_image(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')                    

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def loadConfig(path, name='config'):
    if os.path.isdir(path):
        config_path = os.path.join(path, name + '.yaml')
    else:
        config_path = path        
    with open(config_path) as f:
        config = yaml.load(f)
    return config

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
    return patch['width'], patch['height']


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
    if os.path.isdir(path):
        state_path = os.path.join(path, 'state.csv')
    else:
        state_path = path
    timestamps = []
    states = []
    with open(state_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            state = {}
            for h, r in zip(header, row):
                if h == 'timestamp':
                    timestamps.append(int(1E6 * float(r)))
                    continue
                if len(r) == 0:
                    continue
                state[h] = float(r)
            states.append(state)

    return timestamps, states

