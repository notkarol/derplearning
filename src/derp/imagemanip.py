#import imageio
import numpy as np

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


def get_patch_bbox(sw_config, hw_config):
    """
    Currently we assume that orientations and positions are identical
    """
    hfov_ratio = sw_config['hfov'] / hw_config['hfov']
    vfov_ratio = sw_config['vfov'] / hw_config['vfov']

    width = hw_config['width'] * hfov_ratio
    height = hw_config['height'] * vfov_ratio
    x = (hw_config['width'] - width) // 2 #use floor division("//")
    y = hw_config['height'] - height
    #y assumes that the crop will always include the lowest row of pixels

    return Bbox(x, y, width, height)


'''applies geometric transforms to create the effect of y axis vehical shifting
and z axis vehicle rotation assuming a flat world'''
'''this function could be changed to compute and return only the contents of the bbox'''
def perturb(frame, rotate_degrees, shift_meters, config, margin):
    out = np.zeros(np.shape(frame), dtype=np.uint8)
    

    # the row on which the horizon is found
    horizon_index = horizonset(config['height'], config['vfov'],
                                config['bottom_row_x'], config['z'])

    # determines how many pixel shifts will be needed to effect rotation
    rotate_pixels = rotate_degrees * config['width']  / config['hfov']
    # determines the max pixel shift required for side shifting
    shift_pixels = ymetertopixel(shift_meters, config['width'], config['hfov'])

    for z, row in enumerate(frame):
        
        # Calculates the shift distance for a given row
        shift_dist = (rotate_pixels +
                      shift_pixels * (max(0, (z + 1 - horizon_index))
                                      / (len(frame) - horizon_index)))
        shift_count = int(round(shift_dist, 0))

        # FIXME: modify the loop to only apply to rows needed by the bbox to keep the 
        # margin check from being oversensitive.
        # Executes the called for shift accross the row
        if shift_count == 0:
            out[z] = row
        elif np.abs(shift_count) > margin:
            print("Error shift of %i pixels exceeded margin of %i pixels. This was caused by %f degrees rotation and %f meters of shifting." % (shift_count, margin, rotate_degrees, shift_pixels))
        elif shift_count > 0:
            for y, pixel in enumerate(row):
                if y + shift_count < len(row):
                    out[z][y + shift_count] = row[y]
        elif shift_count < 0:
            for y, pixel in enumerate(row):
                if y + shift_count >= 0:
                    out[z][y + shift_count] = row[y]
    return out


def deg2rad(val):
    return val * np.pi / 180

def rad2deg(val):
    return val * 180 / np.pi


# Returns the number of pixels which the bottom of the frame needs to be shifted
# to approximate the car being shifted by the same number of meters(y)
# bottom_row_x is a measure of the camera's minimum view range (xaxis in meters)
# hfov is the field of view as measured about the 
# z is the height of the camera in meters (note camera is at 0,0,0)
def ymetertopixel(meter_shift, pix_width, hfov, bottom_row_x, z):
    meter_width = 2 * np.tan(deg2rad(hfov) / 2) * (z**2 + bottom_row_x**2) ** 0.5
    return meter_shift * pix_width / meter_width

# returns the pixel elevation of the camera's horizon
def horizonset(pix_height, vfov, bottom_row_x, z):
    return (pix_height * (vfov -  rad2deg(np.arctan(z / bottom_row_x) ) ) / vfov)
