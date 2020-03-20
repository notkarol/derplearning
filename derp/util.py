"""
Common utilities for derp used by various classes.
"""
from collections import namedtuple
import cv2
from datetime import datetime
import heapq
import logging
import pathlib
import numpy as np
import os
import socket
import time
import yaml
import zmq
import capnp
import messages_capnp

Bbox = namedtuple("Bbox", ["x", "y", "w", "h"])

TOPICS = {
    "camera": messages_capnp.Camera,
    "controller": messages_capnp.Controller,
    "action": messages_capnp.Action,
    "imu": messages_capnp.Imu,
    "quality": messages_capnp.Quality,
}

DERP_ROOT = pathlib.Path(os.environ["DERP_ROOT"])
MODEL_ROOT = DERP_ROOT / "models"
RECORDING_ROOT = DERP_ROOT / "recordings"
CONFIG_ROOT = DERP_ROOT / "config"
MSG_STEM = "/tmp/derp_"


def is_already_running(path):
    """ For the given PID path check if the PID exists """
    if isinstance(path, str):
        path = pathlib.Path(path)
    if not path.exists():
        return False
    with open(str(path)) as pid_file:
        pid = int(pid_file.read())
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def write_pid(path):
    with open(str(path), 'w') as pid_file:
        pid_file.write(str(os.getpid()))
        pid_file.flush()


def init_logger(name, recording_path, level=logging.INFO):
    logger = logging.getLogger(name)
    formatter = logging.Formatter('%(asctime)s %(levelname)-5s %(message)s')
    fileHandler = logging.FileHandler(recording_path / ('%s.log' % name), mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    logger.setLevel(level)
    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)
    return logger


def make_recording_path():
    date = datetime.utcfromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")
    folder = RECORDING_ROOT / ("recording-%s-%s" % (date, socket.gethostname()))
    folder.mkdir(parents=True)
    return folder


def get_timestamp():
    return int(time.time() * 1e9)


def sleep_hertz(start_timestamp, hertz):
    end_timestamp = start_timestamp + 1e9 / hertz
    duration = end_timestamp - get_timestamp() - 1e3
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


def topic_file_reader(folder, topic):
    return open("%s/%s.bin" % (folder, topic), "rb")


def topic_exists(folder, topic):
    path = folder / ("%s.bin" % topic)
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
    Gets a different sub-persepective given a smaller desired hfov/vfov and different yaw/pitch
    """
    hfov_ratio = target_config["hfov"] / source_config["hfov"]
    vfov_ratio = target_config["vfov"] / source_config["vfov"]
    hfov_offset = source_config["yaw"] - target_config["yaw"]
    vfov_offset = source_config["pitch"] - target_config["pitch"]
    patch_width = int(source_config["width"] * hfov_ratio + 0.5)
    patch_height = int(source_config["height"] * vfov_ratio + 0.5)
    x_center = (source_config["width"] - patch_width) // 2
    y_center = (source_config["height"] - patch_height) // 2
    x_offset = int(hfov_offset / source_config["hfov"] * source_config["width"] + 0.5)
    y_offset = int(vfov_offset / source_config["vfov"] * source_config["height"] + 0.5)
    x = x_center + x_offset
    y = y_center + y_offset
    if (x >= 0 and x + patch_width <= source_config["width"] and
        y >= 0 and y + patch_height <= source_config["height"]):
        return Bbox(x, y, patch_width, patch_height)
    return None


def crop(image, bbox):
    """ Crops the Bbox(x,y,w,h) from the image. Copy indicates to copy of the ROI"s memory"""
    return image[bbox.y : bbox.y + bbox.h, bbox.x : bbox.x + bbox.w]


def resize(image, size):
    """ Resize the image to the target (w, h) """
    is_larger = size[0] > image.shape[1] or size[1] > image.shape[0]
    interpolation = cv2.INTER_LINEAR if is_larger else cv2.INTER_AREA
    return cv2.resize(image, size, interpolation=interpolation)


def perturb(frame, camera_config, shift=0, rotate=0):
    # Estimate how many pixels to rotate by, assuming fixed degrees per pixel
    pixels_per_degree = camera_config["width"] / camera_config["hfov"]

    # Figure out where the horizon is in the image
    horizon_frac = ((camera_config["vfov"] / 2) + camera_config["pitch"]) / camera_config["vfov"]

    # For each row in the frame shift/rotate it
    indexs = np.arange(len(frame))
    vertical_fracs = np.linspace(0, 1, len(frame))

    # For each vertical line, apply shift/rotation rolls
    for index, vertical_frac in zip(indexs, vertical_fracs):
        magnitude = rotate * pixels_per_degree
        if vertical_frac > horizon_frac:
            ground_angle = (vertical_frac - horizon_frac) * camera_config["vfov"]
            ground_distance = camera_config["z"] / np.tan(deg2rad(ground_angle))
            ground_width = 2 * ground_distance * np.tan(deg2rad(camera_config["hfov"]) / 2)
            magnitude += (shift / ground_width) * camera_config["width"]
        magnitude = int(magnitude + 0.5 * np.sign(magnitude))
        if magnitude > 0:
            frame[index, magnitude:, :] = frame[index, : frame.shape[1] - magnitude]
            frame[index, :magnitude, :] = 0
        elif magnitude < 0:
            frame[index, :magnitude, :] = frame[index, abs(magnitude) :]
            frame[index, frame.shape[1] + magnitude :] = 0
    return frame


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
            component_path = CONFIG_ROOT / config[component]["path"]
            with open(str(component_path)) as component_fd:
                component_config = yaml.load(component_fd, Loader=yaml.FullLoader)
            component_config.update(config[component])
            config[component] = component_config
            if "name" not in config[component]:
                config[component]["name"] = component_path.stem
    if "name" not in config:
        config["name"] = config_path.stem
    return config


def dump_config(config, config_path):
    """ Write a configuration file """
    with open(str(config_path), 'w') as config_fd:
        yaml.dump(config, config_fd)


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


def load_topics(folder):
    if isinstance(folder, str):
        folder = pathlib.Path(folder)
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


def encode_jpg(image, quality):
    return cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tostring()


def extract_car_actions(topics):
    out = []
    autonomous = False
    speed_offset = 0
    steer_offset = 0
    for timestamp, topic, msg in replay(topics):
        if topic == "controller":
            autonomous = msg.isAutonomous
            speed_offset = msg.speedOffset
            steer_offset = msg.steerOffset
        elif topic == "action":
            if autonomous or msg.isManual:
                out.append([timestamp, msg.speed + speed_offset, msg.steer + steer_offset])
    return np.array(out)
