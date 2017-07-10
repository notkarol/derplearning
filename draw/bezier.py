import PIL
import numpy as np
from scipy.misc import comb
from skimage.draw import line_aa
import matplotlib.pyplot as plt

# Calculate bezier polynomials
# Page 2: http://www.idav.ucdavis.edu/education/CAGDNotes/CAGDNotes/Bernstein-Polynomials.pdf
def bernstein_polynomials(i, n, t):
    return comb(n, i) * (t ** (n - i)) * ((1 - t) ** i)


# Get points for bezier curve
def bezier_curve(points, n_segments=5):
    t = np.linspace(0.0, 1.0, n_segments)
    arr = np.array([bernstein_polynomials(i, len(points) - 1, t) for i in range(len(points))])
    x = np.dot(points[:, 0], arr)
    y = np.dot(points[:, 1], arr)
    return x, y

# Verify by saving both pyplot as generated image
def verify_plot(points, x, y, w, h, array):
    fig = plt.figure(figsize=(w / 100, h / 100))
    plt.plot(x, y, 'k-')
    plt.plot(points[:, 0], points[:, 1], 'ro')
    plt.grid()
    plt.xlim(0, w)
    plt.ylim(0, h)
    plt.gca().invert_yaxis()
    plt.savefig('plt.png', dpi=100, bbox_inches='tight')

    image = PIL.Image.fromarray(np.array(array, dtype=np.uint8))
    image.save("pil.png")
    plt.close()
    
    
# Generate board size and plot
if __name__ == "__main__":

    # Configuration
    n_points = 3
    width = 720
    height = 480

    # Prepare Points
    points = np.zeros((n_points, 2), dtype=np.int)
    points[:, 0] = np.random.randint(0, width, n_points)
    points[:, 1] = np.random.randint(0, height, n_points)

    # Get points of curve
    x, y = bezier_curve(points, 32)

    # Plot lines on array
    array = np.zeros((height, width))
    for i in range(len(x) - 1):
        rr, cc, val = line_aa(int(x[i]), int(y[i]), int(x[i + 1]), int(y[i + 1]))
        array[cc, rr] = val * 255

    # Convert to PIL image and save png
    verify_plot(points, x, y, width, height, array)
                              
