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


