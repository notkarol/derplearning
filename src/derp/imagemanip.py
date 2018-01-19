#import imageio
import numpy as np
import scipy.misc

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


def get_patch_bbox(target_config, source_config):
    """
    Currently we assume that orientations and positions are identical
    """
    hfov_ratio = target_config['hfov'] / source_config['hfov']
    vfov_ratio = target_config['vfov'] / source_config['vfov']

    if hfov_ratio > 1 or vfov_ratio > 1:
        raise ValueError("get_patch_bbox: hfov_ratio [%.3f] or vhov_ratio [%.3f] greater than 1" %
                         (hfov_ratio, vfov_ratio))
    
    width = source_config['width'] * hfov_ratio
    height = source_config['height'] * vfov_ratio
    x = (source_config['width'] - width) // 2
    y = source_config['height'] - height

    return Bbox(x, y, width, height)


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
            ground_distance = config['y'] / np.tan(deg2rad(ground_angle))
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
