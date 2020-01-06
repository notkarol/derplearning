"""
Common utilities for derp used by various classes.
"""
from collections import namedtuple
import cv2
import heapq
import pathlib
import numpy as np
import os
import time
import yaml
import zmq
import capnp
import messages_capnp
import scipy.signal as signal
from scipy.interpolate import interp1d

Bbox = namedtuple('Bbox', ['x', 'y', 'w', 'h'])

TOPICS = {
    "camera": messages_capnp.Camera,
    "controller": messages_capnp.Controller,
    "action": messages_capnp.Action,
    "imu": messages_capnp.Imu,
    "quality": messages_capnp.Quality,
}

ROOT = pathlib.Path(os.environ["DERP_ROOT"])

def get_timestamp():
    return int(time.time() * 1E9)


def sleep_hertz(start_timestamp, hertz):
    end_timestamp = start_timestamp + 1E9 / hertz
    duration = end_timestamp - get_timestamp() - 1E3
    if duration > 0:
        time.sleep(duration)

def publisher(path):
    context = zmq.Context()
    sock = context.socket(zmq.PUB)
    sock.bind("ipc://" + path)
    # sock.bind("tcp://*:%s" % port)
    return context, sock


def subscriber(paths):
    context = zmq.Context()
    sock = context.socket(zmq.SUB)
    # sock.connect("tcp://localhost:%s" % port)
    for path in paths:
        sock.connect("ipc://" + path)
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    return context, sock


def loop(config, exit_event, func):
    obj = func(config)
    while not exit_event.is_set() and obj.run():
        pass
    print("Exiting")

def topic_file_reader(folder, topic):
    return open("%s/%s.bin" % (folder, topic), "rb") 


def topic_exists(folder, topic):
    path = folder / ('%s.bin' % topic)
    return path.exists()


def topic_file_writer(folder, topic):
    return open("%s/%s.bin" % (folder, topic), "wb")     


def print_image_config(name, config):
    """ Prints some useful variables about the camera for debugging purposes """
    top = config["pitch"] + config["vfov"] / 2
    bot = config["pitch"] - config["vfov"] / 2
    left = config["yaw"] - config["hfov"] / 2
    right = config["yaw"] + config["hfov"] / 2
    hppd = config["width"] / config["hfov"]
    vppd = config["height"] / config["vfov"]
    print(
        "%s top: %6.2f bot: %6.2f left: %6.2f right: %6.2f hppd: %5.1f vppd: %5.1f"
        % (name, top, bot, left, right, hppd, vppd)
    )


def get_patch_bbox(target_config, source_config):
    """
    Currently we assume that orientations and positions are identical
    """
    if "resize" not in source_config:
        source_config["resize"] = 1
    source_width = int(source_config["width"] * source_config["resize"] + 0.5)
    source_height = int(source_config["height"] * source_config["resize"] + 0.5)
    hfov_ratio = target_config["hfov"] / source_config["hfov"]
    vfov_ratio = target_config["vfov"] / source_config["vfov"]
    hfov_offset = source_config["yaw"] - target_config["yaw"]
    vfov_offset = source_config["pitch"] - target_config["pitch"]
    patch_width = source_width * hfov_ratio
    patch_height = source_height * vfov_ratio
    x_center = (source_width - patch_width) // 2
    y_center = (source_height - patch_height) // 2
    x_offset = (hfov_offset / source_config["hfov"]) * source_width
    y_offset = (vfov_offset / source_config["vfov"]) * source_height
    x = int(x_center + x_offset + 0.5)
    y = int(y_center + y_offset + 0.5)
    patch_width = int(patch_width + 0.5)
    patch_height = int(patch_height + 0.5)
    #print("Using bbox:", x, y, patch_width, patch_height, "in", source_width, source_height)
    if x >= 0 and x + patch_width <= source_width and y >= 0 and y + patch_height <= source_height:
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


def perturb(frame, config, perturbs):
    # Estimate how many pixels to rotate by, assuming fixed degrees per pixel
    pixels_per_degree = config["width"] / config["hfov"]
    rotate_pixels = (perturbs["rotate"] if "rotate" in perturbs else 0) * pixels_per_degree

    # Figure out where the horizon is in the image
    horizon_frac = ((config["vfov"] / 2) + config["pitch"]) / config["vfov"]

    # For each row in the frame shift/rotate it
    indexs = np.arange(len(frame))
    vertical_fracs = np.linspace(0, 1, len(frame))

    # For each vertical line, apply shift/rotation rolls
    for index, vertical_frac in zip(indexs, vertical_fracs):
        magnitude = rotate_pixels
        if "shift" in perturbs and vertical_frac > horizon_frac:
            ground_angle = (vertical_frac - horizon_frac) * config["vfov"]
            ground_distance = config["z"] / np.tan(deg2rad(ground_angle))
            ground_width = 2 * ground_distance * np.tan(deg2rad(config["hfov"]) / 2)
            shift_pixels = (perturbs["shift"] / ground_width) * config["width"]
            magnitude += shift_pixels
        magnitude = int(magnitude + 0.5 * np.sign(magnitude))
        if magnitude > 0:
            frame[index, magnitude:, :] = frame[index, : frame.shape[1] - magnitude]
            frame[index, :magnitude, :] = 0
        elif magnitude < 0:
            frame[index, :magnitude, :] = frame[index, abs(magnitude) :]
            frame[index, frame.shape[1] + magnitude :] = 0


def deg2rad(val):
    return val * np.pi / 180


def rad2deg(val):
    return val * 180 / np.pi


def load_image(path):
    return cv2.imread(str(path))


def save_image(path, image):
    return cv2.imwrite(str(path), image)


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
            if 'name' not in config[component]:
                config[component]['name'] = component_path.stem
    if 'name' not in config:
        config['name'] = config_path.stem
    return config


def smooth(vals):
    b, a = signal.butter(3, 0.05, output="ba")
    return signal.filtfilt(b, a, vals)


def extract_latest(desired_times, source_times, source_values):
    out = []
    pos = 0
    val = 0
    for desired_time in desired_times:
        while pos < len(source_times) and source_times[pos] < desired_time:
            val = source_values[pos]
            pos += 1
        out.append(val)
    return np.array(out)


def interpolate(vals, n_out, intmult=None):
    fn_interpolate = interp1d(np.linspace(0, 1, len(vals)), vals)
    if intmult is None:
        out = np.array([-fn_interpolate(x) for x in np.linspace(0, 1, n_out)])
    else:
        out = np.array([-fn_interpolate(x) * intmult + 0.5 for x in
                        np.linspace(0, 1, n_out)], dtype=np.int)
    return out
                    

def load_topics(folder):
    out = {}
    for topic in TOPICS:
        if not topic_exists(folder, topic):
            continue
        topic_fd = topic_file_reader(folder, topic)
        out[topic] = [msg for msg in TOPICS[topic].read_multiple(topic_fd)]
        topic_fd.close()
    return out


def replay(topics):
    heap = []
    for topic in topics:
        for msg in topics[topic]:
            heapq.heappush(heap, [msg.publishNS, topic, msg])
    while heap:
        yield heapq.heappop(heap)


def decode_jpg(jpg):
    return cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)    


def extract_car_actions(topics):
    out = []
    autonomous = False
    speed_offset = 0
    steer_offset = 0
    for timestamp, topic, msg in replay(topics):
        if topic == 'controller':
            autonomous = msg.isAutonomous
            speed_offset = msg.speedOffset
            steer_offset = msg.steerOffset
        elif topic == 'action':
            if autonomous or msg.isManual:
                out.append([timestamp, msg.speed + speed_offset, msg.steer + steer_offset])
    return np.array(out)

