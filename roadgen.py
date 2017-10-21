 #!/usr/bin/env python3
import os
import argparse
import PIL
import cv2
import numpy as np
import numpy.random as rng
from scipy.misc import imsave
from scipy.misc import imread
from skimage.draw import polygon
from bezier import bezier_curve
#from model import Model
import matplotlib
import matplotlib.pyplot as plt
import imageio

'''
Running this file generates 3 line road training data
v5 draws 3 color lines on the ground the middle one being dashed.
'''

import yaml
with open("config/line_model.yaml", 'r') as yamlfile:
    cfg = yaml.load(yamlfile)

''' The Rooagen class is responsible for all things related to the generation of virtual training data
    Roadgen's init function defines several key image parameters used by line_model validation systems
    So the class is frequently invoked in line_validate '''

class Roadgen:
    """Load the generation configuration parameters"""
    def __init__(self, config=cfg):
        #characteristics of the lines to be detected:
        self.n_lines = config['line']['n_lines']
        self.n_points = config['line']['n_points']
        self.n_dimensions = config['line']['n_dimensions']
        self.n_channels = config['line']['n_channels']
        self.max_intensity = 256

        #parameters of the virtual view window
        self.view_res = (cfg['line']['gen_width'], 
                            cfg['line']['gen_height'])
        self.input_res = (cfg['line']['input_width'] , 
                            cfg['line']['input_height'])

        self.gen_res = self.view_res * 2
        self.cropsize = ((self.gen_res[0] - self.view_res[0])/2,
                         (self.gen_res[1] - self.view_res[1])/2 )

        #define camera characteristics
        #linear measurements given in mm
        self.cam_height = 380
        self.cam_min_view = 500 #FIXME remeasure distance
        self.cam_res = np.array([640, 540])
        #arcs measured in radians
        #arc from bottom of camera view to vertical
        self.cam_to_ground_arc = np.arctan(self.cam_min_view / self.cam_height)
        self.cam_arc_y = 80 * (np.pi / 180)
        self.cam_arc_x = 60 * (np.pi / 180)
        #measure of how close the center of the camera's view is to horizontal
        self.cam_tilt_y = self.cam_to_ground_arc + self.cam_arc_x/2 - np.pi/2
        #self.crop_ratio = [c / s for c, s in zip(self.crop_size, self.source_size) ]
        #1/2 Minimum road view width of the camera in mm
        
        #Attributes of the road line drawing.
        self.max_road_width = self.view_res[0] * 0.5 #The widest possible road that can be drawn (radius)
        self.horz_noise_fraction = 0.5 #size of the noise envelope below max_width where road lines may exist
                    #horz_noise_fraction = 1 allows the road edge lines to exist anywhere between the center and max width
        self.lane_convergence = .6 #rate at which lanes converge approaching horizon

        #parameters to be used by the drawing function road_gen
        self.n_segments = config['line']['n_segments']
        self.line_width = self.view_res[0] * .007
        self.line_wiggle = self.view_res[0] * 0.00005

    def __del__(self):
        #Deconstructor
        pass

    #Reshape a label tensor to 2d and normalizes data for use by the ANN:
    def label_norm(self, nd_labels):
        
        #normalization
        nd_labels[:, :, 0, :] /= self.gen_res[0]
        nd_labels[:, :, 1, :] /= self.gen_res[1]
        nd_labels -= 0.5

        #reshaping
        twod_labels =  np.reshape(nd_labels, (nd_labels.shape[0], self.n_lines * self.n_points *
                                     self.n_dimensions) )

        return twod_labels

    #Reshape the label tensor for use by all other functions and reverses normalization
    def model_interpret(self, twod_labels):
        #reshape labels
        nd_labels = np.reshape(twod_labels, (twod_labels.shape[0], self.n_lines, self.n_dimensions,
                                        self.n_points))
        #denormalize labels
        nd_labels += 0.5
        nd_labels[:,:,0,:] *= self.gen_res[0]
        nd_labels[:,:,1,:] *= self.gen_res[1]

        return nd_labels

    #normalizes an image tensor to be floats on the range 0. - 1.
    def normalize(self, frames):
        return  ((frames.astype(float) - np.mean(frames, axis=0, dtype=float))/ 
                np.std(frames, axis=0, dtype=float) )

    #Scales an image tensor by a maximum intensity and recasts as uint8 for display purposes
    #FIXME I don't remember if this is still used and it's currently out of date
    def denormalize(self, frame):
        
        frame =  np.uint8(frame *self.max_intensity)
        
        return frame


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

    #function defines the location of the middle control points
    #for a single curve frame:
    def middle_points(self, y_train, y_noise, orientation=1):
        #Assign the central control point:
        y_train[ 1, 0, 1] = np.random.randint(0, self.view_res[0]) + self.cropsize[0]
        y_train[ 1, 1, 1] = np.random.randint(0, self.view_res[1]) + self.cropsize[1]

        '''Calculate the control point axis:
        control point axis is defined by a unit vector parallel to a line
        originating at the midpoint of the line's start and end and 
        terminating at the curves second control point'''
        u_delta = self.unit_vector(y_train[ 1, :, 1 ] - 
                (y_train[ 1, :, 0 ] + (y_train[ 1, :, 2 ] - y_train[ 1, :, 0 ] )/2 ) )

        #print(u_delta)
        #scales the sideways road outputs to account for pitch
        vertical_excursion = np.absolute(u_delta[1]) * self.max_road_width

        #rotational correction:
        u_rot = self.rot_by_vector(y_train[ 1, :, 2 ] - y_train[ 1, :, 0 ], u_delta)
        #Checks to see which side of the line between start and end the midpoint falls on
        if u_rot[1] > 0:
            #If u_delta has a negative y component then it falls south of the line and need to be inverted
            orientation = -1

        #Set the left side no.1 control point
        y_train[ 0, :, 1] = (y_train[ 1, :, 1 ] + u_delta
            * np.multiply( (self.max_road_width - y_noise[ 0, 1]),
            (1 - self.lane_convergence + self.lane_convergence *
                np.maximum(.2, (y_train[ 1, 1, 1 ] + vertical_excursion) )/self.gen_res[1]) ) )
        #Set the right side no.1 control point
        y_train[ 2, :, 1] = (y_train[ 1, :, 1 ] - u_delta
            * np.multiply( (self.max_road_width - y_noise[ 1, 1]),
            (1 - self.lane_convergence + self.lane_convergence *
                np.maximum(.2, (y_train[ 1, 1, 1 ] - vertical_excursion) )/self.gen_res[1]) ) )

        return y_train

    # Generate coordinate of bezier control points for a road
    #FIXME: control point generation for turns is currently garbage
    #FIXME: Split the vanishing point into 3 points
    #Needs: more variability, realism (esp for right hand turns)
    def coord_gen(self, n_datapoints):
        y_train = np.zeros( (n_datapoints, self.n_lines, self.n_dimensions, 
            self.n_points), dtype=np.float)

        #noise for the side lines
        y_noise = np.zeros((n_datapoints, self.n_lines-1 ,
            self.n_points) , dtype=np.int)
        y_noise = np.random.randint(0, self.max_road_width * self.horz_noise_fraction,
            (n_datapoints, self.n_lines - 1, self.n_points) )

        #Defining the road's 'start' at the base of the camera's view point (not the base of the generation window)
        #Centerline base definition:
        y_train[:, 1, 0, 0 ] = np.random.randint(self.cropsize[0], 
            (self.gen_res[0] - self.cropsize[0]), (n_datapoints))
        #Left line base point
        y_train[:, 0, 0, 0 ] = y_train[:, 1, 0, 0 ] - (self.max_road_width - y_noise[:, 0, 0])
        #Right line base point
        y_train[:, 2, 0, 0 ] = y_train[:, 1, 0, 0 ] + (self.max_road_width - y_noise[:, 1, 0])
        #Road start elevation:
        y_train[:, :, 1, 0] = int(self.gen_res[1] - self.cropsize[1] * .75)

        #places the vanishing point either on the side of the view window or at the top of the screen
        vanishing_point = np.random.randint(0, (self.gen_res[0] + self.gen_res[1]*2),  (n_datapoints) )

        '''This loop applies the vanishing point and then generates control points in a semi-logical way
        between the road's origin and vanishing point '''
        for dp_i in range(n_datapoints):
            if(vanishing_point[dp_i] < self.gen_res[1]):
                #Assign the vanishing point:
                y_train[dp_i, :, 1, 2] = vanishing_point[dp_i]
                
                #These lines give the vanishing point width when sideways:
                term_width = np.multiply( (self.max_road_width - y_noise[dp_i, 0, 2]),
                    (1 - self.lane_convergence + self.lane_convergence * 
                        y_train[dp_i, 1, 1, 2 ]/self.gen_res[1]) )
                y_train[dp_i, 0, 1, 2 ] = y_train[dp_i, 1, 1, 2 ] + term_width
                y_train[dp_i, 2, 1, 2 ] = y_train[dp_i, 1, 1, 2 ] - term_width
                
                #Assign the control points for the side lines
                y_train[dp_i] = self.middle_points(y_train[dp_i], y_noise[dp_i], orientation=[1, -1] )
                
            elif(vanishing_point[dp_i] < self.gen_res[1] + self.gen_res[0]):
                #define the vanishing point at the top of the camera's perspective
                y_train[dp_i, :, 0, 2] = vanishing_point[dp_i] - self.gen_res[1]
                y_train[dp_i, :, 1, 2] = 0

                # Define the middle control points as members of a horizontal line chosen with the center point lying in the view window
                #First assign the line containing the control points an elevation:
                y_train[dp_i, :, 1, 1] = np.random.randint(0, self.view_res[1]) + self.cropsize[1]
                #Centerline middle control point definition:
                y_train[dp_i, 1, 0, 1 ] = np.random.randint(self.cropsize[0], 
                    (self.gen_res[0] - self.cropsize[0]) )
                #Left line base point
                y_train[dp_i, 0, 0, 1 ] = y_train[dp_i, 1, 0, 1 ] - np.multiply( 
                    (self.max_road_width - y_noise[dp_i, 0, 1]),
                    (1 - self.lane_convergence * y_train[dp_i, 1, 1, 1 ]/self.gen_res[1]) ) 
                #Right line base point
                y_train[dp_i, 2, 0, 1 ] = y_train[dp_i, 1, 0, 1 ] + np.multiply( 
                    (self.max_road_width - y_noise[dp_i, 1, 1]),
                    (1 - self.lane_convergence * y_train[dp_i, 1, 1, 1 ]/self.gen_res[1]) ) 
            else:
                #Assign the vanishing point to the rhs boundary
                y_train[dp_i, :, 0, 2] = self.gen_res[0] - 1
                y_train[dp_i, :, 1, 2] = vanishing_point[dp_i] - (self.gen_res[1] + self.gen_res[0])

                #These lines give the vanishing point width when sideways:
                term_width = np.multiply( (self.max_road_width - y_noise[dp_i, 0, 2]),
                    (1 - self.lane_convergence + self.lane_convergence
                        * y_train[dp_i, 1, 1, 2 ]/self.gen_res[1]) )
                y_train[dp_i, 0, 1, 2 ] = y_train[dp_i, 1, 1, 2 ] - term_width
                y_train[dp_i, 2, 1, 2 ] = y_train[dp_i, 1, 1, 2 ] + term_width
                
                #Assign side lines
                y_train[dp_i] = self.middle_points(y_train[dp_i], y_noise[dp_i], orientation=[ -1, -1] )

        return y_train


    def poly_line(self, coordinates, line_width, seg_noise = 0):
        #Note that we subtract generation offsets from the curve coordinates before calculating the line segment locations
        x,y = bezier_curve(coordinates[ 0, : ] - self.cropsize[0],
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
        polygon_path[:, true_line.shape[1] ] = noise_line[:, true_line.shape[1]-1] + [line_width, 0]

        #Now we set the start and endpoints (they must be the same!)
        polygon_path[:, 0] = noise_line[:, 0] - [line_width, 0]
        polygon_path[:, 2*true_line.shape[1] -1] = noise_line[:, 0] + [line_width, 0] #This is the last unique point
        polygon_path[:, 2*true_line.shape[1] ] = noise_line[:, 0] - [line_width, 0]

        #Actually draw the polygon
        rr, cc = polygon((polygon_path.astype(int)[1]), polygon_path.astype(int)[0], ( self.view_res[1], self.view_res[0]) )

        return rr, cc


    # Draws dashed lines like the one in the center of the road
    # FIXME add noise to the dashed line generator to cut down on over-fitting(may be superfluous)
    def dashed_line(self, coordinates, dash_length, dash_width ):
        #estimate the curve length to generate a segment count which will approximate the desired dash lenght
        est_curve_len = (self.vector_len(coordinates[:,2] - coordinates[:,0] ) + 
                        self.vector_len(coordinates[:,1] - coordinates[:,0] ) + 
                        self.vector_len(coordinates[:,2] - coordinates[:,1] ) )/2
        segments = int(est_curve_len/dash_length)
        x, y = bezier_curve(coordinates[0, :] - self.cropsize[0], 
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
                            (self.view_res[1], self.view_res[0]) )
            rrr = np.append(rrr, rr)
            ccc = np.append(ccc, cc)

        return rrr, ccc

    #Makes randomly shaped polygon noise to screw with the learning algorithm
    def poly_noise(self, origin, max_size=[128,24],  max_verticies=10):
        vert_count = np.random.randint(3,max_verticies)
        verts = np.matmul(np.ones([vert_count+1, 1]), [origin] )
        verts[1:vert_count, 0] = origin[ 0] + np.random.randint(0, max_size[0], vert_count -1)
        verts[1:vert_count, 1] = origin[ 1] + np.random.randint(0, max_size[1], vert_count -1)

        return polygon(verts[:,1], verts[:,0], (self.view_res[1], self.view_res[0]) )

    #converts coordinates into images with curves on them
    def road_generator(self, y_train, line_width, rand_gen=1, seg_noise = 0, poly_noise=0):
        road_frame = np.ones((self.view_res[1], self.view_res[0], self.n_channels),
             dtype=np.uint8)

        #set max intensity:
        max_intensity = 256

        if rand_gen:
            #Initialize a single tone background:
            if np.random.rand() < 0.5:
                road_frame[ :, :, 0] *= rng.randint(0, .7 * max_intensity)
                road_frame[ :, :, 1] *= rng.randint(0, .7 * max_intensity)
                road_frame[ :, :, 2] *= rng.randint(0, .7 * max_intensity)
            else:
                road_frame[ :, :, :] *= rng.randint(0, .7 * max_intensity)
    
        if seg_noise:
            #line width randomizer:
            line_width += rand_gen * max(line_width/3 *np.random.randn(), -line_width*3/4)

        while poly_noise:
            rr, cc = self.poly_noise([np.random.randint(0, self.gen_res[0]),
                    np.random.randint(0, self.gen_res[1]) ] )
            road_frame[rr,cc, :] = (road_frame[rr,cc,:] +
                                    rng.randint(0, .5 *max_intensity, self.n_channels)) / 2
            poly_noise -= 1

        rr, cc = self.poly_line( y_train[0], line_width, seg_noise)
        road_frame[rr, cc, :] = rng.randint(.8* max_intensity, max_intensity, self.n_channels)

        rr, cc = self.dashed_line(y_train[1], self.view_res[1]/4, line_width*2)
        road_frame[rr, cc, 1:] = rng.randint(.7 * max_intensity, max_intensity) 
        road_frame[rr, cc, 0] = rng.randint(0,  .3 * max_intensity) 

        rr, cc = self.poly_line( y_train[2], line_width, seg_noise)
        road_frame[rr, cc, :] = rng.randint(.8* max_intensity, max_intensity, self.n_channels) 

        return road_frame


    #Saves images in side by side plots
    def save_images(self, Top_Images, Bot_Images, directory , titles = ('Input Image','Model Perception') ):
        
        max_intensity = 256

        curves_to_print = min(Top_Images.shape[0],Bot_Images.shape[0])

        if not os.path.exists(directory):
            os.makedirs(directory)

        for dp_i in range(curves_to_print):

            #This is the actual save images part
            plt.subplot(2, 1, 1)
            plt.title(titles[0])
            plt.imshow(Top_Images.astype(np.uint8)[dp_i,:,:,::-1], vmin=0, vmax=255)
            #plt.gca().invert_yaxis()

            plt.subplot(2, 1, 2)
            plt.title(titles[1])
            plt.imshow(Bot_Images.astype(np.uint8)[dp_i,:,:,::-1], vmin=0, vmax=255)
            #plt.gca().invert_yaxis()

            plt.savefig('%s/%06i.png' % (directory, dp_i), dpi=200, bbox_inches='tight')
            plt.close()

    #Function generates roads using a label vector passed to it
    # saves those generated roads in a location designated by the save name.
    def training_saver(self, y_train, save_name):
        patch = self.road_generator(y_train=y_train,
                            line_width=self.line_width,
                            seg_noise=self.line_wiggle * rng.randint(0, 4),
                            poly_noise=rng.randint(0, 20) )
        #Scales the image down to the input size and reproduces an important resizing artifact
        thumb = cv2.resize(patch, self.input_res, interpolation=cv2.INTER_AREA)

        imsave(save_name, thumb)


    '''batch gen creats a folder and then fills that folder with:
        n_datapoints of png files
        a numpy file with the label array
        a meta file describing batch breakpoints (useful for matching lable files with images)'''
    def batch_gen(self, n_datapoints, data_dir):
        #Cast n_datapoints as an int if it has not already happened:
        n_datapoints = int(n_datapoints)

        #Creates needed directories
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        #branch builds on previous branch
        if os.path.exists('%s/batch_meta.npy' % data_dir):
            batch_sizes = np.load('%s/batch_meta.npy' % data_dir)
            batch_iter = batch_sizes.shape[0] -1
            
            #Make a new larger array to store batch data
            newb_sizes = np.zeros(batch_iter + 2)
            newb_sizes[:batch_iter+1] = batch_sizes
            newb_sizes[batch_iter+1] = n_datapoints + newb_sizes[batch_iter]
        else :
            batch_iter = 0
            newb_sizes = [0, n_datapoints]
        
        #Generate Labels
        y_train = self.coord_gen(n_datapoints)

        # Generate X
        for dp_i in range(n_datapoints ):
            self.training_saver(y_train=y_train[dp_i], 
                                save_name='%s/%09i.png' % (data_dir, dp_i + newb_sizes[batch_iter]) )
            print("Generation %.2f%% complete." % ((100.0 * dp_i / n_datapoints)), end='\r')
        

        np.save("%s/y_%03i.npy" % (data_dir, batch_iter) , y_train)
        np.save("%s/batch_meta.npy" % (data_dir) , newb_sizes)
        print("%s dataset %i generated." % (data_dir, batch_iter) )

    '''Loads the designated batch from the given directory
     and returns a tensor of the selected images.'''
    def batch_loader(self, data_dir, batch_iter):

        batch_meta = np.load('%s/batch_meta.npy' % data_dir)
        batch_size = int(batch_meta[batch_iter+1] - batch_meta[batch_iter] )
        batch = np.zeros( ( (batch_size),
                            self.input_res[1],
                            self.input_res[0],
                            self.n_channels), 
                        dtype= np.uint8 )

        for dp_i in range(batch_size):
            batch[dp_i] = imread('%s/%09i.png' % (data_dir, (batch_meta[batch_iter] + dp_i) ) )
        return batch



#Generates a batch of training data    
def main():

    parser = argparse.ArgumentParser(description='Roadlike virtual data generator')
    parser.add_argument('--tests', type=int, default=0, metavar='TS', 
        help='creates a test batch and compares the batch to video data (default is off)')
    parser.add_argument('--frames', type=int, default=1E4,
        help='determines how many frames per batch')
    parser.add_argument('--batches', type=int, default = 9,
        help='determines how many training batches to generate')
    parser.add_argument('--val_batches', type=int, default = 1,
        help='determines how many validation batches to generate')    
    args = parser.parse_args()

    roads = Roadgen(cfg)

    #generate the training data
    for batch in range(args.batches):
        roads.batch_gen(n_datapoints=args.frames, data_dir=cfg['dir']['train_data'])

    #generate the validation data
    for batch in range(args.val_batches):
        roads.batch_gen(n_datapoints=args.frames, data_dir=cfg['dir']['val_data'])        

if __name__ == "__main__":
    main()