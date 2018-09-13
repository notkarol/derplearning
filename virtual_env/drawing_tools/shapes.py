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


class Shapes:
    
    def __init__(self):
        pass


    #returns a unit vector perpendicular to the input vector
    #Aka a unit vector normal to the curve as defined by delta
    def perpendicular(self, delta):
        u_vector =  np.matmul( delta, [[0,-1],[1,0]] ) / np.sqrt(np.matmul( np.multiply(delta, delta), [[1],[1]] ))
            
        return u_vector


    #Converts a vector into a unit vector with the same orientation
    def unit_vector(self, delta):
        '''if np.absolute(delta) == 0
            raise Value_Error('Cannot calculate the unit vector of a size 0 vector!')'''
        return delta / np.sqrt(np.matmul(np.multiply(delta,delta),[1, 1]) )

    #measures a vector's length and returns that as a scalar
    def vector_len(self, vector):
        return np.sqrt(np.matmul(np.multiply(vector,vector),[1, 1]) )

    def rot_by_vector(self, rot_vect, vector):
        unit_rot_vect = self.unit_vector( rot_vect)
        rot_mat = np.array([[unit_rot_vect[0], -unit_rot_vect[1]], 
                            [unit_rot_vect[1],  unit_rot_vect[0]] ])
        return np.matmul(vector, rot_mat)


    # Calculate bezier polynomials
    # Page 2: http://www.idav.ucdavis.edu/education/CAGDNotes/CAGDNotes/Bernstein-Polynomials.pdf
    def bernstein_polynomials(self, i, n, t):
        return comb(n, i) * (t ** (n - i)) * ((1 - t) ** i)


    # Get points for bezier curve
    def bezier_curve(self, x_arr, y_arr, n_segments=5):
        t = np.linspace(0.0, 1.0, n_segments)
        arr = np.array([ self.bernstein_polynomials(i, len(x_arr) - 1, t) for i in range(len(x_arr))])
        x = np.dot(x_arr, arr)
        y = np.dot(y_arr, arr)
        return x, y

    def poly_line(self, view_res, coordinates, line_width, seg_noise = 0):
        #Note that we subtract generation offsets from the curve coordinates before calculating the line segment locations
        x,y = self.bezier_curve(coordinates[ 0, : ] - self.cropsize[0],
                 coordinates[1, :] - self.cropsize[1], self.n_segments)
        true_line = np.array([x, y])

        #Add some noise to the line so it's harder to over fit
        noise_line = true_line + seg_noise * np.random.randn(2, true_line.shape[1])
        #Create the virtual point path needed to give the line width when drawn by polygon:

        polygon_path = np.zeros( (true_line.shape[0], 2 * true_line.shape[1] + 1) , dtype=float)

        #Now we offset the noisy line perpendicularly by the line width to give it depth (rhs)
        polygon_path[:, 1:(true_line.shape[1]-1) ] = (noise_line[:,1:true_line.shape[1]-1]
             + line_width * np.transpose(self.perpendicular(
            np.transpose(noise_line[:,2:] - noise_line[:, :noise_line.shape[1]-2]) ) ) )
        #Same code but subtracting width and reverse order to produce the lhs of the line
        polygon_path[:, (2*true_line.shape[1]-2):(true_line.shape[1]) :-1 ] = (noise_line[:,1:true_line.shape[1]-1]
             - line_width * np.transpose(self.perpendicular(
            np.transpose(noise_line[:,2:] - noise_line[:, :noise_line.shape[1]-2]) ) ) )

        #These points determine the bottom end of the line:
        polygon_path[:, true_line.shape[1]-1] = noise_line[:, true_line.shape[1]-1] - [line_width, 0]
        polygon_path[:, true_line.shape[1]  ] = noise_line[:, true_line.shape[1]-1] + [line_width, 0]

        #Now we set the start and endpoints (they must be the same!)
        polygon_path[:, 0]                       = noise_line[:, 0] - [line_width, 0]
        polygon_path[:, 2*true_line.shape[1] -1] = noise_line[:, 0] + [line_width, 0] #This is the last unique point
        polygon_path[:, 2*true_line.shape[1]   ] = noise_line[:, 0] - [line_width, 0]

        #Actually draw the polygon
        rr, cc = polygon((polygon_path.astype(int)[1]), polygon_path.astype(int)[0], (view_res[1], view_res[0]) )

        return rr, cc

    # Draws dashed lines like the one in the center of the road
    # FIXME add noise to the dashed line generator to cut down on over-fitting(may be superfluous)
    def dashed_line(self, view_res, coordinates, dash_length, dash_width ):
        #estimate the curve length to generate a segment count which will approximate the desired dash lenght
        est_curve_len = (self.vector_len(coordinates[:,2] - coordinates[:,0] ) + 
                        self.vector_len(coordinates[:,1] - coordinates[:,0] ) + 
                        self.vector_len(coordinates[:,2] - coordinates[:,1] ) )/2
        segments = int(est_curve_len/dash_length)
        x, y = self.bezier_curve(coordinates[0, :] - self.cropsize[0], 
                coordinates[1, :] - self.cropsize[1], segments)
        dash_line = np.array([x, y])

        #initializing empty indices
        rrr = np.empty(0, dtype=int)
        ccc = np.empty(0, dtype=int)
        for dash in range( int(segments/2) ):
            offset = .5*dash_width * self.perpendicular(dash_line[:,dash*2]-dash_line[:,dash*2+1])
            d_path = np.array( [ dash_line[:,dash*2] + offset, dash_line[:,dash*2 +1] + offset, 
                        dash_line[:,dash*2+1] - offset, dash_line[:,dash*2 ] - offset,
                        dash_line[:,dash*2] + offset] )
            rr, cc = polygon(d_path.astype(int)[:,1], d_path.astype(int)[:,0], 
                            (view_res[1], view_res[0]) )
            rrr = np.append(rrr, rr)
            ccc = np.append(ccc, cc)

        return rrr, ccc


    #Makes randomly shaped polygon noise to screw with the learning algorithm
    def poly_noise(self, view_res, origin, max_size=[128,24],  max_verticies=10):
        vert_count = np.random.randint(3,max_verticies)
        verts = np.matmul(np.ones([vert_count+1, 1]), [origin] )
        verts[1:vert_count, 0] = origin[ 0] + np.random.randint(0, max_size[0], vert_count -1)
        verts[1:vert_count, 1] = origin[ 1] + np.random.randint(0, max_size[1], vert_count -1)

        return polygon(verts[:,1], verts[:,0], (view_res[1], view_res[0]) )


    # Verify by saving both pyplot as generated image
    def verify_plot(self, points, x, y, w, h):
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
    n_datapoints = int(1E4)
    train_split = 0.8
    n_train_datapoints = int(train_split * n_datapoints)

    shaper0 = Shapes()
    
    # Data to store
    X_train = np.zeros((n_datapoints, n_channels, height, width), dtype=np.float)
    y_train = np.zeros((n_datapoints, n_points * n_dimensions), np.float)

    # Generate Y
    y_train[:, : n_points] = np.random.randint(0, width, (n_datapoints, n_points))
    y_train[:, n_points :] = np.random.randint(0, height, (n_datapoints, n_points))

    # Generate X
    for dp_i in range(n_datapoints):
        x, y = shaper0.bezier_curve(y_train[dp_i, : n_points], y_train[dp_i, n_points :], n_segments)
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
