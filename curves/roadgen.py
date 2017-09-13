import PIL
import numpy as np
from skimage.draw import line_aa
from bezier import bezier_curve

'''
v1 only draws one line on the ground. it is a work in progress.
'''

# Configuration
n_lines = 3
n_points = 3
n_dimensions = 2

n_channels = 1
train_width = 64
gen_width = train_width+64
cropsize = int( (gen_width-train_width)/2)
height = 32

max_road_width = gen_width/4
horz_noise_fraction = 0.25

n_segments = 10
n_datapoints = int(1E5)
train_split = 0.8
n_train_datapoints = int(train_split * n_datapoints)


# Data to store
X_train = np.zeros((n_datapoints, n_channels, height, train_width), dtype=np.float)
y_train = np.zeros((n_datapoints, n_lines, n_dimensions, n_points ), np.float)

# Generate Y
#Centerline:
y_train[:, 1, 0, : ] = np.random.randint(max_road_width, (gen_width-max_road_width), (n_datapoints, n_points))
y_train[:, 1, 1, 1:] = np.sort(np.random.randint(height/4, height, (n_datapoints, (n_points-1) ) ) )
#note that by sorting the height control points we get non-uniform distributions

#noise for the side lines
y_noise = np.zeros((n_datapoints, n_lines, n_dimensions, n_points) , np.float)
y_noise = np.random.randint(0, max_road_width*horz_noise_fraction, (n_datapoints, n_lines, n_dimensions, n_points) )

#Left lines
y_train[:, 0, 0, : ] = y_train[:, 1, 0, : ] - np.multiply( (max_road_width*(1-horz_noise_fraction) - y_noise[:, 0, 1, :]),(1 - .8*y_train[:, 1, 1, : ]/height) )
y_train[:, 0, 1, : ] = y_train[:, 1, 1, : ]

#Right lines
y_train[:, 2, 0, : ] = y_train[:, 1, 0, : ] + np.multiply( (max_road_width*(1-horz_noise_fraction) - y_noise[:, 2, 1, :]),(1 - .8*y_train[:, 1, 1, : ]/height) ) 
y_train[:, 2, 1, : ] = y_train[:, 1, 1, : ]


# Generate X
for dp_i in range(n_datapoints):
  #Temporary generation location
  X_gen = np.zeros((n_channels, height, gen_width), dtype=np.float)

  x0, y0 = bezier_curve(y_train[dp_i, 0, 0, : ], y_train[dp_i, 0, 1, :], n_segments)
  for ls_i in range(len(x0) - 1):
    rr, cc, val = line_aa(int(x0[ls_i]), int(y0[ls_i]), int(x0[ls_i + 1]), int(y0[ls_i + 1]))
    X_gen[0, cc, rr] = val

  x1, y1 = bezier_curve(y_train[dp_i, 1, 0, : ], y_train[dp_i, 1, 1, :], n_segments)
  for ls_i in range(len(x1) - 1):
    rr, cc, val = line_aa(int(x1[ls_i]), int(y1[ls_i]), int(x1[ls_i + 1]), int(y1[ls_i + 1]))
    X_gen[0, cc, rr] = val

  x2, y2 = bezier_curve(y_train[dp_i, 2, 0, : ], y_train[dp_i, 2, 1, :], n_segments)
  for ls_i in range(len(x2) - 1):
    rr, cc, val = line_aa(int(x2[ls_i]), int(y2[ls_i]), int(x2[ls_i + 1]), int(y2[ls_i + 1]))
    X_gen[ 0, cc, rr] = val

  X_train[dp_i, :, :, :] = X_gen[:, :, cropsize : (gen_width-cropsize) ]

# Normalize training and testing
X_train *= (1. / np.max(X_train))
y_train[:, :, 0, :] /= gen_width
y_train[:, :, 1, :] /= height

#fixes the label array for use by the learning model
y_train = np.reshape(y_train, (n_datapoints, n_lines * n_points * n_dimensions) )

# Save Files
np.save("X_train.npy", X_train[:n_train_datapoints])
np.save("X_val.npy", X_train[n_train_datapoints:])
np.save("y_train.npy", y_train[:n_train_datapoints])
np.save("y_val.npy", y_train[n_train_datapoints:])