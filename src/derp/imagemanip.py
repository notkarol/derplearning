import imageio

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
    x = (hw_config['width'] - width) // 2
    y = hw_config['height'] - height

    return Bbox(x, y, width, height)


def perturb(frame, rotate_degrees, shift_meters, config):
    out = np.zeros(np.shape(img), dtype=np.uint8)
    horizon_pixels = horizonset(config['height'], config['vfov'],
                                config['bottom_row_x'], config['z'])
    rotate_pixels = config['width'] * rotate_degrees / config['hfov']
    shift_pixels = ymetertopixel(shift_meters, config['width'], config['hfov'])

    for z, row in enumerate(img):
        
        # Calculates the shift distance for a given row
        shift_dist = (rotate_pixels +
                      shift_pixels * (max(0, (z + 1 - horizon_pixels))
                                      / (len(img) - horizon_pixels)))
        shift_count = int(round(shift_dist, 0))

        # Executes the called for shift accross the row
        if shift_count == 0:
            out[z] = row
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
    
def ymetertopixel(shift, width, hfov, bottom_row_x, z):
    blah = 2 * np.tan(deg2rad(hfov) / 2) * (z**2 + bottom_row_x**2) ** 0.5
    return shift * width / blah

def horizonset(height, vfov, bottom_row_x, z):
    return (height * (vfov - rad2deg(np.arctan(z / bottom_row_x))) / vfov)



