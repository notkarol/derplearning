import csv
import cv2
from datetime import datetime
import numpy as np
import os
import re
import scipy.misc
import socket
import time
import torch
from torch.autograd import Variable
import yaml


class Bbox:
    def __init__(self, x, y, w, h):
        self.x = int(x + 0.5) # first col
        self.y = int(y + 0.5) # first row
        self.w = int(w + 0.5) # width
        self.h = int(h + 0.5) # height
    def __repr__(self):
        return str(self)
    def __str__(self):
        return "Bbox(x: %i y: %i w: %i h: %i)" % (self.x, self.y, self.w, self.h)

    
def get_patch_bbox(target_config, source_config):
    """
    Currently we assume that orientations and positions are identical
    """

    source_top = source_config['pitch'] + source_config['vfov'] / 2
    source_bot = source_config['pitch'] - source_config['vfov'] / 2
    target_top = target_config['pitch'] + target_config['vfov'] / 2
    target_bot = target_config['pitch'] - target_config['vfov'] / 2
    print("Top: %7.3f %7.3f" % (source_top, target_top))
    print("Bot: %7.3f %7.3f" % (source_bot, target_bot))
    
    hfov_ratio = target_config['hfov'] / source_config['hfov']
    vfov_ratio = target_config['vfov'] / source_config['vfov']
    hfov_offset = source_config['yaw'] - target_config['yaw']
    vfov_offset = source_config['pitch'] - target_config['pitch']
    width = source_config['width'] * hfov_ratio
    height = source_config['height'] * vfov_ratio
    x_center = (source_config['width'] - width) // 2
    y_center = (source_config['height'] - height) // 2
    x_offset = (hfov_offset / source_config['hfov']) * source_config['width']
    y_offset = (vfov_offset / source_config['vfov']) * source_config['height']
    x = x_center + x_offset
    y = y_center + y_offset
    assert x >= 0 and x + width <= source_config['width']
    assert y >= 0 and y + height <= source_config['height']
    bbox = Bbox(x, y, width, height)
    return bbox


def crop(image, bbox):
    out = image[bbox.y : bbox.y + bbox.h, bbox.x : bbox.x + bbox.w]
    return out


def resize(image, size):
    out = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return out


def perturb(frame, config, perts):

    # Estimate how many pixels to rotate by, assuming fixed degrees per pixel
    pixels_per_degree = config['width'] / config['hfov']
    rotate_pixels = (perts['rotate'] if 'rotate' in perts else 0) * pixels_per_degree

    # Figure out where the horizon is in the image
    horizon_frac = ((config['vfov'] / 2) + config['pitch']) / config['vfov']

    # For each row in the frame shift/rotate it
    indexs = np.arange(len(frame))
    vertical_fracs = np.linspace(0, 1, len(frame))

    # For each vertical line, apply shift/rotation rolls
    for index, vertical_frac in zip(indexs, vertical_fracs):

        # We always adjust for rotation
        magnitude = rotate_pixels

        # based on the distance adjust for shift
        if 'shift' in perts and vertical_frac > horizon_frac:
            ground_angle = (vertical_frac - horizon_frac) * config['vfov']
            ground_distance = config['z'] / np.tan(deg2rad(ground_angle))
            ground_width = 2 * ground_distance * np.tan(deg2rad(config['hfov']) / 2)
            shift_pixels = (perts['shift'] / ground_width) * config['width']
            magnitude += shift_pixels

        # Find the nearest integer
        magnitude = int(magnitude + 0.5 * np.sign(magnitude))

        if magnitude > 0:
            frame[index, magnitude:, :] = frame[index, : frame.shape[1] - magnitude]
            frame[index, :magnitude, :] = 0
        elif magnitude < 0:
            frame[index, :magnitude, :] = frame[index, abs(magnitude):]
            frame[index, frame.shape[1] + magnitude:] = 0


def deg2rad(val):
    return val * np.pi / 180


def rad2deg(val):
    return val * 180 / np.pi


def load_image(path):
    return cv2.imread(path)


def save_image(path, image):
    return cv2.imwrite(path, image)


def get_name(path):
    """
    The name of a script is it's filename without the extension
    """
    clean_path = path.rstrip('/')
    bn = os.path.basename(clean_path)
    name, ext = os.path.splitext(bn)
    return name


def get_hostname():
    return socket.gethostname()


def create_record_folder():
    """
    Generate the name of the record folder and created it
    """
    dt = datetime.utcfromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")
    hn = socket.gethostname()
    path = os.path.join(os.environ['DERP_ROOT'], "data", "%s-%s" % (dt, hn))
    print("Creating", path)
    os.mkdir(path)
    return path


def load_config(config_path):
    """ 
    Loads the vehicle config and all requisite components configs
    """

    # Make sure we have a path to a file
    if os.path.isdir(config_path):
        config_path  = os.path.join(config_path, 'config.yaml')
    
    # First load the car's config
    with open(config_path) as f:
        config = yaml.load(f)

    # Make sure we set the name and path of the config stored
    if 'name' not in config:
        config['name'] = get_name(config_path)
    if 'path' not in config:
        config['path'] = config_path

    # Then load the each component
    dirname = os.path.dirname(config_path)
    for component_config in config['components']:

        # Check if we need to load more parameters from elsewhere
        if 'path' in component_config:
            component_path = os.path.join(os.environ['DERP_ROOT'], 'config', component_config['path'])
            with open(component_path) as f:
                default_component_config = yaml.load(f)

            # Load paramters only if they're not found in default
            for key in default_component_config:
                if key not in component_config:
                    component_config[key] = default_component_config[key]

            # Make sure we have a name for this component
            if 'name' not in component_config:
                component_config['name'] = os.path.basename(os.path.dirname(component_config['path']))

        # Make sure we were able to find a name
        if 'name' not in component_config:
            raise ValueError("load_config: all components must have a name or a path")

        # Make sure we were able to find a class
        if 'class' not in component_config:
            raise ValueError("load_config: all components must have a class in components/")

    # Make sure we also have a state
    if 'state' not in config:
        config['state'] = {}
        
    return config


def load_component(component_config, config):

    # Load the component from its module
    module_name = "derp.components." + component_config['class'].lower()
    class_fn = load_class(module_name, component_config['class'])
    component = class_fn(component_config, config)

    # If we're ready, add it, otherwise make sure it's required
    if not component.ready and component_config['required']:
        raise ValueError("load_components: missing required", component_config['name'])

    print("Loaded component", module_name)
    return component


def load_components(config):
    """
    Load the class of each component by its name and initialize all state keys.
    """
    from derp.state import State
    state = State(config['state'], config)
    components = [state]

    # Initialize components
    for component_config in config['components']:

        # Load the component object
        component = load_component(component_config, config)

        # Skip a non-ready component. Raise an error if it's required as we can't continue
        if not component.ready:
            if component_config['required']:
                raise ValueError("load_components: required component [%s] not available"
                                 % component_config['name'])
            print("load_components: skipping", component_config['name'])
            continue
        
        # if we survived the cull, add the component to 
        components.append(component)

        # Preset all state keys
        if 'state' in component_config:
            for key in component_config['state']:

                # Don't set the key if it's already set and the proposed value is None
                # This allows us to have components request fields, but have a master
                # initializer. Useful for servo or car-specific steer_offset
                val = component_config['state'][key]
                if key not in state or state[key] is None:
                    state[key] = val

    return state, components


def find_component_config(full_config, name):
    """
    Finds the matching component by name of the component and script if needed
    """
    for component_config in full_config['components']:
        if name in component_config['name']:
            return component_config
    

def load_class(path, name):
    """ 
    Loads the class "name" at relative path (period separated) "path" and returns it
    """
    from importlib import import_module
    m = import_module(path)
    c = getattr(m, name)
    return c


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
            timestamps.append(float(line[0]))
            for value in line[1:]:
                try:
                    value = float(value) if floats else value
                except:
                    value = 0
                state.append(value)
            states.append(state)
    timestamps = np.array(timestamps, dtype=np.double)
    if floats:
        states = np.array(states, dtype=np.float)
    return timestamps, headers, states


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


def find_matching_file(path, name_pattern):
    """
    Finds a file that matches the given name regex
    """
    pattern = re.compile(name_pattern)
    if os.path.exists(path):        
        for filename in os.listdir(path):
            if pattern.search(filename) is not None:
                return os.path.join(path, filename)
    return None


def extractList(config, state):
    if len(config) == 0:
        return
    vector = np.zeros(len(config), dtype=np.float32)
    for i, d in enumerate(config):
        scale = d['scale'] if 'scale' in d else 1
        vector[i] = state[d['field']] * scale
    return vector


def unscale(config, vector):
    if len(config) == 0:
        return
    state = {}
    for i, d in enumerate(config):
        scale = d['scale'] if 'scale' in d else 1
        vector[i] /= scale
    return vector
    

def unbatch(batch):
    if torch.cuda.is_available():
        out = batch.data.cpu().numpy()
    else:
        out = batch.data.numpy()
    if len(out) == 1:
        return out[0]
    return out


def prepareVectorBatch(vector, cuda=True):
    """ Common vector to batch preparation script for training and inference """
    if vector is None:
        return

    # Treat it as if it's a row in a larger batch
    if len(vector.shape) == 1:
        vector = np.reshape(vector, [1] + list(vector.shape))

    # Pepare the torch representation
    batch = Variable(torch.from_numpy(vector).float())
    if cuda:
        batch = batch.cuda()
    return batch


def prepareImageBatch(image, cuda=True):
    """ Common image to batch preparation script for training and inference """
    if image is None:
        return

    # Make sure it's a 4d tensor
    if len(image.shape) < 4:
        batch = np.reshape(image, [1] * (4 - len(image.shape)) + list(image.shape))

    # Make sure that we have BCHW
    batch = batch.transpose((0, 3, 1, 2))

    # Normalize input to range [0, 1)
    batch = Variable(torch.from_numpy(batch).float())
    if cuda:
        batch = batch.cuda()
        batch /= 256

    return batch
    
    
        
    
    
