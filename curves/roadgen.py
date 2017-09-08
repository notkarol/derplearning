import PIL
import numpy as np
from skimage.draw import line_aa
from bezier import bezier_curve

'''
v1 only draws one line on the ground. it is a work in progress.
'''

# Configuration
n_points = 3
n_channels = 1
n_dimensions = 2
width = 32
height = 32
n_segments = 10
n_datapoints = int(2E5)
train_split = 0.8
n_train_datapoints = int(train_split * n_datapoints)

# Data to store
X_train = np.zeros((n_datapoints, n_channels, height, width), dtype=np.float)
y_train = np.zeros((n_datapoints, n_points * n_dimensions), np.float)

# Generate Y
y_train[:, : n_points] = np.random.randint(0, width, (n_datapoints, n_points))
y_train[:, (n_points+1) :] = np.sort(np.random.randint(0, height, (n_datapoints, (n_points-1) ) ) )
#note that by sorting the height control points we get non-uniform distributions

# Generate X
for dp_i in range(n_datapoints):
    x, y = bezier_curve(y_train[dp_i, : n_points], y_train[dp_i, n_points :], n_segments)
    for ls_i in range(len(x) - 1):
        rr, cc, val = line_aa(int(x[ls_i]), int(y[ls_i]), int(x[ls_i + 1]), int(y[ls_i + 1]))
        X_train[dp_i, 0, cc, rr] = val

# Normalize training and testing
X_train *= (1. / np.max(X_train))
y_train[:, :n_points] /= width
y_train[:, n_points:] /= height

# Save Files
np.save("X_train.npy", X_train[:n_train_datapoints])
np.save("X_val.npy", X_train[n_train_datapoints:])
np.save("y_train.npy", y_train[:n_train_datapoints])
np.save("y_val.npy", y_train[n_train_datapoints:])