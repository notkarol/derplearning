import PIL
import numpy as np
from scipy.misc import comb
from skimage.draw import line_aa
import matplotlib.pyplot as plt

'''
function list:
  bernstein_polynomials
  bezier_curve
  verify_plot

function: bezier_curve converts control point coordinates into strait line segment coordinates
'''

# Calculate bezier polynomials
# Page 2: http://www.idav.ucdavis.edu/education/CAGDNotes/CAGDNotes/Bernstein-Polynomials.pdf
def bernstein_polynomials(i, n, t):
    return comb(n, i) * (t ** (n - i)) * ((1 - t) ** i)


# Get points for bezier curve
def bezier_curve(x_arr, y_arr, n_segments=5):
    t = np.linspace(0.0, 1.0, n_segments)
    arr = np.array([bernstein_polynomials(i, len(x_arr) - 1, t) for i in range(len(x_arr))])
    x = np.dot(x_arr, arr)
    y = np.dot(y_arr, arr)
    return x, y

# Verify by saving both pyplot as generated image
def verify_plot(points, x, y, w, h):
    fig = plt.figure(figsize=(w / 10, h / 10))
    plt.plot(x, y, 'k-')
    plt.plot(points[:, 0], points[:, 1], 'ro')
    plt.grid()
    plt.xlim(0, w)
    plt.ylim(0, h)
    plt.gca().invert_yaxis()
    plt.savefig('plt.png', dpi=100, bbox_inches='tight')    
    
# Generate board size and plot
if __name__ == "__main__":

    # Configuration
    n_points = 3
    n_channels = 1
    n_dimensions = 2
    width = 32
    height = 32
    radius = 5
    n_segments = 10
    n_datapoints = int(1E5)
    train_split = 0.8
    n_train_datapoints = int(train_split * n_datapoints)
    
    # Data to store
    X_train = np.zeros((n_datapoints, n_channels, height, width), dtype=np.float)
    y_train = np.zeros((n_datapoints, n_points * n_dimensions), np.float)

    # Generate Y
    y_train[:, : n_points] = np.random.randint(0, width, (n_datapoints, n_points))
    y_train[:, n_points :] = np.random.randint(0, height, (n_datapoints, n_points))

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
