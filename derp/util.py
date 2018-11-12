"""
Common utilities for derp used by various classes.
"""
import csv
import cv2
from datetime import datetime
import evdev
import pathlib
import numpy as np
import os
import re
import scipy.misc
import socket
import subprocess
import time
import torch
from torch.autograd import Variable
import yaml


ROOT = pathlib.Path(os.environ["DERP_ROOT"])

class Bbox:
    """
    A bounding box is a 4-tuple of integers representing x, y, width, height
    """
    def __init__(self, x, y, w, h):
        """ Creates class x, y, w,h variables from the arguments. """
        self.x = int(x + 0.5) # first col
        self.y = int(y + 0.5) # first row
        self.w = int(w + 0.5) # width
        self.h = int(h + 0.5) # height

    def __repr__(self):
        """
        For this class's representation just use all the local class variables.
        """
        return "Bbox(x: %i y: %i w: %i h: %i)" % (self.x, self.y, self.w, self.h)

    def __str__(self):
        return repr(self)

def find_device(names):
    """
    Searches for an input devices. Assuming it is found that device is returned
    """
    for filename in sorted(evdev.list_devices()):
        device = evdev.InputDevice(filename)
        device_name = device.name.lower()
        for name in names:
            if name in device_name:
                print("Using evdev:", device_name)
                return device
    return None


def get_car_config_path(name):
    return ROOT / "config" / "car" / (name + ".yaml")


def get_controller_config_path(name):
    return ROOT / "config" / "controller" / (name + ".yaml")


def get_controller_models_path(name):
    return ROOT / "models" / name


def get_experiment_path(name):
    return ROOT / "scratch" / name


def encode_video(folder, name, suffix, fps=30):
    cmd = " ".join(["gst-launch-1.0",
                    "multifilesrc",
                    "location='%s/%s/%%06d.%s'" % (folder, name, suffix),
                    "!", "'image/jpeg,framerate=%i/1'" % fps,
                    "!", "jpegparse",
                    "!", "jpegdec",
                    "!", "omxh264enc", "bitrate=8000000",
                    "!", "'video/x-h264, stream-format=(string)byte-stream'",
                    "!", "h264parse",
                    "!", "mp4mux",
                    "!", "filesink location='%s/%s.mp4'" % (folder, name)])
    subprocess.Popen(cmd, shell=True)


def print_image_config(name, config):
    """ Prints some useful variables about the camera for debugging purposes """
    top = config["pitch"] + config["vfov"] / 2
    bot = config["pitch"] - config["vfov"] / 2
    left = config["yaw"] - config["hfov"] / 2
    right = config["yaw"] + config["hfov"] / 2
    hppd = config["width"] / config["hfov"]
    vppd = config["height"] / config["vfov"]
    print("%s top: %6.2f bot: %6.2f left: %6.2f right: %6.2f hppd: %5.1f vppd: %5.1f" %
          (name, top, bot, left, right, hppd, vppd))


def get_patch_bbox(target_config, source_config):
    """
    Currently we assume that orientations and positions are identical
    """
    if 'resize' not in source_config:
        source_config['resize'] = 1
    source_width = int(source_config['width'] * source_config['resize'] + 0.5)
    source_height = int(source_config['height'] * source_config['resize'] + 0.5)
    hfov_ratio = target_config["hfov"] / source_config['hfov']
    vfov_ratio = target_config["vfov"] / source_config['vfov']
    hfov_offset = source_config['yaw'] - target_config["yaw"]
    vfov_offset = source_config['pitch'] - target_config["pitch"]
    patch_width = source_width * hfov_ratio
    patch_height = source_height * vfov_ratio
    x_center = (source_width - patch_width) // 2
    y_center = (source_height - patch_height) // 2
    x_offset = (hfov_offset / source_config['hfov']) * source_width
    y_offset = (vfov_offset / source_config['vfov']) * source_height
    x = int(x_center + x_offset + 0.5)
    y = int(y_center + y_offset + 0.5)
    patch_width = int(patch_width + 0.5)
    patch_height = int(patch_height + 0.5)
    print('Using bbox:', x, y, patch_width, patch_height,
          'in', source_width, source_height)
    if (x >= 0 and x + patch_width <= source_width
        and y >= 0 and y + patch_height <= source_height):
        return Bbox(x, y, patch_width, patch_height)
    return None


def crop(image, bbox, copy=False):
    """ Crops the Bbox(x,y,w,h) from the image. Copy indicates to copy of the ROI"s memory"""
    roi = image[bbox.y : bbox.y + bbox.h, bbox.x : bbox.x + bbox.w]
    if copy:
        return roi.copy()
    return roi


def resize(image, size):
    """ Resize the image to the target (w, h) """
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def perturb(frame, config, perts):

    # Estimate how many pixels to rotate by, assuming fixed degrees per pixel
    pixels_per_degree = config["width"] / config["hfov"]
    rotate_pixels = (perts["rotate"] if "rotate" in perts else 0) * pixels_per_degree

    # Figure out where the horizon is in the image
    horizon_frac = ((config["vfov"] / 2) + config["pitch"]) / config["vfov"]

    # For each row in the frame shift/rotate it
    indexs = np.arange(len(frame))
    vertical_fracs = np.linspace(0, 1, len(frame))

    # For each vertical line, apply shift/rotation rolls
    for index, vertical_frac in zip(indexs, vertical_fracs):

        # We always adjust for rotation
        magnitude = rotate_pixels

        # based on the distance adjust for shift
        if "shift" in perts and vertical_frac > horizon_frac:
            ground_angle = (vertical_frac - horizon_frac) * config["vfov"]
            ground_distance = config["z"] / np.tan(deg2rad(ground_angle))
            ground_width = 2 * ground_distance * np.tan(deg2rad(config["hfov"]) / 2)
            shift_pixels = (perts["shift"] / ground_width) * config["width"]
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
    return cv2.imread(str(path))


def save_image(path, image):
    return cv2.imwrite(str(path), image)


def get_name(path):
    """ The name of a script is it"s filename without the extension """
    return pathlib.Path(str(path).rstrip("/")).stem


def get_hostname():
    return socket.gethostname()


def create_record_folder():
    """ Generate the name of the record folder and created it """
    dt = datetime.utcfromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")
    hn = socket.gethostname()
    path = ROOT / "data" / ("%s-%s" % (dt, hn))
    path.mkdir(parents=True, exist_ok=True)
    return path


def pass_config(config_path, dict_0, list_ind=0, dict_1=0, dict_2=0, dict_3=0):
    """
    Passes a single config dict entry as a return value so it can be used by a shell script.
    """
    with open(str(config_path)) as config_fd:
        config = yaml.load(config_fd)

    if dict_3 != 0:
        value = config[dict_0][list_ind][dict_1][dict_2][dict_3]
    elif dict_2 != 0:
        value = config[dict_0][list_ind][dict_1][dict_2]
    elif dict_1 != 0:
        value = config[dict_0][list_ind][dict_1]
    else:
        value = config[dict_0]

    return value


def load_config(config_path):

    # First load the car"s config
    with open(str(config_path)) as config_fd:
        config = yaml.load(config_fd)

    # Make sure we set the name and path of the config stored
    if "name" not in config:
        config["name"] = get_name(config_path)
    if "path" not in config:
        config["path"] = config_path

    # Load component configs recursively if they exist, and eventually return the full config
    if "components" not in config:
        return config
    for component_config in config["components"]:

        # Check if we need to load more parameters from elsewhere
        if "path" in component_config:
            component_path = ROOT / "config" / component_config["path"]
            with open(str(component_path)) as component_fd:
                default_component_config = yaml.load(component_fd)

            # Load paramters only if they"re not found in default
            for key in default_component_config:
                if key not in component_config:
                    component_config[key] = default_component_config[key]

            # Make sure we have a name for this component
            if "name" not in component_config:
                component_config["name"] = pathlib.Path(component_config["path"]).parent.name

        # Make sure we were able to find a name
        if "name" not in component_config:
            raise ValueError("load_config: all components must have a name or a path")

        # Make sure we were able to find a class
        if "class" not in component_config:
            raise ValueError("load_config: all components must have a class in components/")
    return config


def load_component(config, state):
    module_name = "derp.components.%s" % (config["class"].lower())
    class_fn = load_class(module_name, config["class"])
    script = class_fn(config, state)
    if not script.ready and config["required"]:
        raise ValueError("load_script: failed", config["name"])
    print("Loaded %s" % module_name)
    return script


def load_controller(config, car_config, state):
    module_name = "derp.controllers.%s" % (config["class"].lower())
    print(module_name)
    class_fn = load_class(module_name, config["class"])
    script = class_fn(config, car_config, state)
    if not script.ready:
        raise ValueError("load_controller: failed")
    print("Loaded %s" % module_name)
    return script


def load_components(config, state):
    # Load the class of each component by its name and initialize all state keys.
    components = []
    for component_config in config:

        # Load the component object
        component = load_component(component_config, state)

        # Skip a non-ready component. Raise an error if it"s required as we can"t continue
        if not component.ready:
            if component_config["required"]:
                raise ValueError("load_components: required component [%s] not available"
                                 % component_config["name"])
            print("load_components: skipping", component_config["name"])
            continue

        # if we survived the cull, add the component to
        components.append(component)

        # Preset all state keys
        if "state" in component_config:
            for key in component_config["state"]:

                # Don"t set the key if it"s already set and the proposed value is None
                # This allows us to have components request fields, but have a master
                # initializer. Useful for servo or car-specific steer_offset
                val = component_config["state"][key]
                if key not in state or state[key] is None:
                    state[key] = val
    return components


def find_component_config(full_config, name):
    """
    Finds the matching component by name of the component and script if needed
    """
    for component_config in full_config["components"]:
        if name in component_config["name"]:
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
    Returns the non-timestamp headers, timestamps as numpy arrays.
    """
    timestamps = []
    states = []
    with open(str(path)) as csv_fd:
        reader = csv.reader(csv_fd)
        headers = next(reader)[1:]
        for line in reader:
            if not len(line):
                continue
            state = []
            timestamps.append(float(line[0]))
            for value in line[1:]:
                try:
                    if floats:
                        value = float(value)
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
    path = pathlib.Path(path)
    if path.exists():
        for filename in path.glob('*'):
            if pattern.search(str(filename)) is not None:
                return path / filename
    return None


def extractList(config, state):
    if len(config) == 0:
        return
    vector = np.zeros(len(config), dtype=np.float32)
    for i, d in enumerate(config):
        scale = d["scale"] if "scale" in d else 1
        vector[i] = state[d["field"]] * scale
    return vector


def unscale(config, vector):
    if len(config) == 0:
        return
    state = {}
    for i, d in enumerate(config):
        scale = d["scale"] if "scale" in d else 1
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

    # Treat it as if it"s a row in a larger batch
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


def plot_batch(path, example, status, label, guess):
    import matplotlib.pyplot as plt
    dim = int(len(example) ** 0.5)
    if (dim * dim) < len(example):
        dim += 1
    fig, axs = plt.subplots(dim, dim, figsize=(dim, dim))

    # Change from CHW to HWC, and move RGB to GBR
    example = np.transpose(example, (0, 2, 3, 1))[...,[2,1,0]]
    for i in range(len(example)):
        x, y = i % dim, int(i // dim)
        axs[y, x].imshow(example[i])

        # Prepare Title
        label_str = " ".join(["%5.2f" % x for x in label[i]])
        guess_str = " ".join(["%5.2f" % x for x in guess[i]])
        axs[y, x].set_title('L: %s\nG: %s' % (label_str, guess_str), fontsize=8)
        axs[y, x].set_xticks([])
        axs[y, x].set_yticks([])

    plt.savefig("%s.png" % str(path), bbox_inches='tight', dpi=160)
    print("Saved batch %s" % path)
