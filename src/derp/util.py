import csv
import cv2
import numpy as np
import os
import pickle
import PIL.Image
import sys
import yaml
import re
import evdev
import socket
from datetime import datetime
from time import time
from importlib import import_module
from collections import OrderedDict

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


def get_name(path):
    return os.path.splitext(os.path.basename(path.rstrip('/')))[0]


def get_record_folder():
     dt = datetime.utcfromtimestamp(time()).strftime("%Y%m%d-%H%M%S")
     hn = socket.gethostname()
     return os.path.join(os.environ['DERP_DATA'], "%s-%s" % (dt, hn))


def has_image_ext(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
    

def load_image(path):
    with open(path, 'rb') as f:
        with PIL.Image.open(f) as img:
            return img.convert('RGB')                    


def save_image(path, img):
    if type(img) == np.ndarray:
        img = PIL.Image.fromarray((img * 255).astype(np.uint8))
    img.save(path)
        

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

        
def load_config(path):

    # Don't load blank paths
    if path is None:
        return None
    
    config_path = os.path.join(path, 'config.yaml') if os.path.isdir(path) else path
    with open(config_path) as f:
        config = yaml.load(f)
    if 'name' not in config:
        config['name'] = get_name(config_path)
    return config


def load_module(path):
    return import_module(path)


def load_class(path, name):
    m = load_module(path)
    c = getattr(m, name)
    return c
    

def load_components(config, state):
    out = []

    # Initialize components
    for component in config['components']:
        c = load_class("derp.components." + component['class'].lower(), component['class'])
        obj = c(component)

        # Preset all state keys
        if 'state' in component:
            for key in component['state']:
                val = component['state'][key]

                # Don't set the key if it's already set and the proposed value is None
                # This allows us to have components request fields, but have a master
                # initializer. Useful for servo or car-specific steer_offset
                if key in state and val is None:
                    continue

                state[key] = val
        
        # Discover
        discover = obj.discover()
        print("Connecting to %s %s [%s]" % ('required' if component['required'] else 'optional',
                                            component['name'], discover))

        # Exit if we're missing a component
        if not discover and component['required']:
            sys.exit(1)

        if discover:
            out.append(obj)
    return out
    

def get_patch_bbox(source_config, target_config):
    """
    Currently we assume that orientations and positions are identical
    """
    
    patch = target_config['patch']
    
    hfov_ratio = patch['hfov'] / source_config['hfov']
    vfov_ratio = patch['vfov'] / source_config['vfov']

    width = source_config['width'] * hfov_ratio
    height = source_config['height'] * vfov_ratio
    x = (source_config['width'] - width) // 2
    y = source_config['height'] - height

    return Bbox(x, y, width, height)



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
            timestamps.append(int(re.sub('\D', '', line[0] ) ) )
            #regex to remove any non-decimal characters from the timestamp so that 
            #it can be read as an int
            for value in line[1:]:
                value = float(value) if floats else value
                state.append(value)
            states.append(state)
    timestamps = np.array(timestamps, dtype=np.uint64)
    if floats:
        states = np.array(states, dtype=np.float)
    return timestamps, headers, states



'''
Produces side shifts and z axis rotations of training data
'''

#shift label function
'''
    dsteer is the stearing in degrees
        (positive steering is turning left atm)
    drot is the number of degrees the car is rotated about zaxis for perm
        (rotate car left is postiive in degrees)
    mshift is the number of car is shifted along yaxis for perm
        (shift car left is positive in meters)
'''
def shiftsteer(steer, drot, mshift):
    maxsteer = 1 # maximum value which can be returned for steering
    shifttosteer = 1 # ratio of steering correction to lateral shift in steers/m
    spd = 1 / 30 # linear coefficient for angle correction in steers/degree

    permsteer = steer - drot * spd - mshift*shifttosteer
    if permsteer>0:
        return min(maxsteer, permsteer)
    else:
        return max(-maxsteer, permsteer)


#shift image function
'''
    img is the input image
    drot is the number of pixels the image is rotated about zaxis
        (rotate car left is postiive in degrees)
    mshift is the number of meters the car is shifted along yaxis
        (shift car left is positive in meters)
'''
def shiftimg(img, drot, mshift, cfovz, cfovy):
    perm = np.zeros(np.shape(img), dtype=np.uint8)
    phorz = horizonset(len(img), cfovy)
    prot = zdegtopixel(drot, len(img[0]), cfovz )
    pshift = ymetertopixel(mshift, len(img[0]), cfovz )

    for z,row in enumerate(img):
        
        #Calculates the shift distance for a given row
        shift_dist = prot + pshift*(max(0,(z+1-phorz) )/(len(img)-phorz ) )
        shift_count = int(round(shift_dist,0) )

        #Executes the called for shift accross the row
        if shift_count == 0:
            perm[z] = row
        elif shift_count >0:
            for y, pixel in enumerate(row):
                if y+shift_count <len(row):
                    perm[z][y+shift_count] = row[y]
        elif shift_count <0 :
            for y, pixel in enumerate(row):
                if y+shift_count >=0:
                    perm[z][y+shift_count] = row[y]

    return perm


#calculates the number of pixels the image has rotated
# for a given degree rotation of the camera
def zdegtopixel(deg, iydim, cfovz = 100):
    return deg * iydim/cfovz


#converts a displacement of the car in y meters to pixels along the bottom row of the image
def ymetertopixel(disp, iydim, cfovz = 100):
    cheight = .38 #camera height in meters
    minvis = .7 #FIXME minimum distance camera can see (must be measured)

    botwidth = 2 * np.tan(cfovz * pi / (2 * 180) ) * (cheight**2 + minvis**2)**.5

    return disp * iydim / botwidth


def horizonset(izdim, cfovy = 60):
    cheight = .38 #camera height in meters
    minvis = .7 #FIXME minimum distance camera can see (must be measured)

    return izdim*( (cfovy-np.arctan(cheight/minvis)*180/pi )/cfovy )


def plot_batch(example, label, name):
    import matplotlib.pyplot as plt
    dim = int(np.sqrt(len(example)))
    fig, axs = plt.subplots(dim, dim, figsize=(dim, dim))
    for i in range(len(example)):
        x = i % dim
        y = int(i // dim)

        # change from CHW to HWC and only show first three channels
        img = np.transpose(example[i].numpy(), (1, 2, 0))[:, :, :3]
        axs[y, x].imshow(img)
        axs[y, x].set_title(" ".join(["%.2f" % x for x in label[i]]))
        
    plt.savefig("%s.png" % name, bbox_inches='tight', dpi=160)
    print("Saved batch [%s]" % name)

def find_device(name):
    out = []
    for filename in evdev.list_devices():
        device = evdev.InputDevice(filename)
        if device.name == name:
            return device
    return None


def find_value(haystack, key, values, interpolate=False):
    """
    Find the nearest value in the sorted haystack to the specified key.
    """

    nearest = 0
    diff = np.abs(haystack - key)
    if interpolate:
        nearest = diff.argsort()[:2]
        return (values[nearest[0]] + values[nearest[1]]) / 2

    nearest = diff.argmin()
    return values[nearest]
