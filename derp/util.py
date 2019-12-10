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
import yaml
import zmq
import capnp
import messages_capnp

TOPIC_CLASSES = {'camera': messages_capnp.Camera,
                 'state': messages_capnp.State,
                 'control': messages_capnp.Control,
                 'imu': messages_capnp.Imu}

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


def get_timestamp():
    return time.time_ns() // 1000000 


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
    print("Could not find devices", names, 'in', evdev.list_devices())
    return None


def get_config_path(name):
    return ROOT / "config" / (name + ".yaml")


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


def publisher(path):
    context = zmq.Context()
    sock = context.socket(zmq.PUB)
    sock.bind("ipc://" + path)
    #sock.bind("tcp://*:%s" % port)
    return context, sock

def subscriber(paths):
    context = zmq.Context()
    sock = context.socket(zmq.SUB)
    #sock.connect("tcp://localhost:%s" % port)
    for path in paths:
        sock.connect("ipc://" + path)
    sock.setsockopt(zmq.SUBSCRIBE, b'')
    return context, sock


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


def create_record_folder():
    """ Generate the name of the record folder and created it """
    dt = datetime.utcfromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")
    hn = socket.gethostname()
    path = ROOT / "data" / ("%s-%s" % (dt, hn))
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_config(config_path):
    """ Load a configuration file, also reading any component configs """
    with open(str(config_path)) as config_fd:
        config = yaml.load(config_fd, Loader=yaml.FullLoader)
    for component in config:
        if isinstance(config[component], dict) and "path" in config[component]:
            component_path = ROOT / "config" / config[component]["path"]
            with open(str(component_path)) as component_fd:
                component_config = yaml.load(component_fd, Loader=yaml.FullLoader)
            component_config.update(config[component])
            config[component] = component_config
    return config


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


def unscale(config, vector):
    if len(config) == 0:
        return
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
    batch = torch.from_numpy(vector).float()
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
    batch = torch.from_numpy(batch).float()
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
