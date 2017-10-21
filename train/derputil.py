import csv
import cv2
import numpy as np
import os
import pickle
import PIL.Image
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
        with PIL.Image.open(f) as img:
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

def getPatchBbox(source_config, target_config, perspective='record'):
    """
    Currently we assume that orientations and positions are identical
    """
    
    patch = target_config['patch']
    frame = source_config[perspective]
    
    hfov_ratio = patch['hfov'] / frame['hfov']
    vfov_ratio = patch['vfov'] / frame['vfov']

    width = frame['width'] * hfov_ratio
    height = frame['height'] * vfov_ratio
    x = (frame['width'] - width) // 2
    y = frame['height'] - height

    return Bbox(x, y, width, height)


def getPatchSize(target_config):
    return target_config['patch']['width'], target_config['patch']['height']


def read_csv(path, floats=True):
    """
    Read through the state file and get our timestamps and recorded values.
    Returns the non-timestamp headers, timestamps as 
    """
    timestamps = []
    states = []
    with open(path) as f:
        reader = csv.reader(f)
        headers = next(reader)[1:]
        for line in reader:
            if not len(line):
                continue
            state = []
            timestamps.append(int(line[0]))
            for value in line[1:]:
                value = float(value) if floats else value
                state.append(value)
            states.append(state)
    timestamps = np.array(timestamps, dtype=np.uint64)
    if floats:
        states = np.array(states, dtype=np.float)
    return timestamps, headers, states

