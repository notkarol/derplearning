 #!/usr/bin/env python3
import os
import PIL
import cv2
import numpy as np
import argparse
from skimage.draw import polygon
from bezier import bezier_curve
from model import Model
import matplotlib.pyplot as plt

'''
Running this file generates 3 line road training data
v2 only draws 3 lines on the ground.
'''

import yaml
with open("config/line_model.yaml", 'r') as yamlfile:
    cfg = yaml.load(yamlfile)

''' The Rooagen class is responsible for all things related to the generation of virtual training data
    Roadgen's init function defines several key image parameters used by line_model validation systems
    So the class is frequently invoked in line_validate '''

class Roadgen:
    """Load the generation configuration parameters"""
    def __init__(self, config):
        #characteristics of the lines to be detected:
        self.n_lines = config['line']['n_lines']
        self.n_points = config['line']['n_points']
        self.n_dimensions = config['line']['n_dimensions']
        self.n_channels = config['line']['n_channels']

        #parameters of the virtual view window
        self.view_height = config['line']['input_height']
        self.view_width = config['line']['input_width']
        self.gen_width = self.view_width * 2 # buffer onto each horizontal side to allow us to draw curves
        self.gen_height = self.view_height * 2
        self.cropsize = (int((self.gen_width - self.view_width) / 2), 
                int((self.gen_height - self.view_height)/2) )

        #parameters of the final image size to be passed to the CNN
        self.input_width = config['line']['input_width']
        self.input_height = config['line']['input_height']

        #Attributes of the road line drawing.
        self.max_road_width = self.view_width * 0.75 #The widest possible road that can be drawn (radius)
        #self.min_road_height = self.gen_height/2 #The minimum value for the 2nd and 3rd road control points
        self.horz_noise_fraction = 0.8 #size of the noise envelope below max_width where road lines may exist
            #horz_noise_fraction = 1 allows the road edge lines to exist anywhere between the center and max width
        self.lane_convergence = .6 #rate at which lanes converge approaching horizon

        #parameters to be used by the drawing function road_gen
        self.n_segments = config['line']['n_segments']
        self.line_width = self.view_width * .006
        self.line_wiggle = self.view_width * 0.0001

    def __del__(self):
        #Deconstructor
        pass

    #Reshape a label tensor to 2d and normalizes data for use by the ANN:
    def model_tranform(self, nd_labels):
        
        #normalization
        nd_labels[:, :, 0, :] /= self.gen_width
        nd_labels[:, :, 1, :] /= self.gen_height

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
        nd_labels[:,:,0,:] *= self.gen_width
        nd_labels[:,:,1,:] *= self.view_height

        '''#Clamp model outputs (consider making this switched)
        nd_labels[:,:,0,:] = self.clamp(nd_labels[:,:,0,:], self.gen_width)
        nd_labels[:,:,1,:] = self.clamp(nd_labels[:,:,1,:], self.view_height)
        '''
        return nd_labels

    #returns a unit vector perpendicular to the input vector
    #Aka a unit vector normal to the curve as defined by delta
    def perpendicular(self, delta):
        u_vector =  np.matmul( delta, [[0,-1],[1,0]] ) / np.sqrt(np.matmul( np.multiply(delta, delta), [[1],[1]] ))
            
        return u_vector


    #Converts a vector into a unit vector with the same orientation
    def unit_vector(self, delta):
        return delta / np.sqrt(np.matmul(np.multiply(delta,delta),[1, 1]) )

    #measures a vector's length and returns that
    def vector_len(self, vector):
        return np.sqrt(np.matmul(np.multiply(vector,vector),[1, 1]) )


    # Generate coordinate of beizier control points for a road
    def coord_gen(self, n_datapoints):
        y_train = np.zeros( (n_datapoints, self.n_lines, self.n_dimensions, 
            self.n_points), np.float)

        #noise for the side lines
        y_noise = np.zeros((n_datapoints, self.n_lines-1 ,
            self.n_points) , dtype=np.int)
        y_noise = np.random.randint(0, self.max_road_width * self.horz_noise_fraction,
            (n_datapoints, self.n_lines - 1, self.n_points) )

        #Defining the road's 'start' at the base of the camera's view point (not the base of the generation window)
        #Centerline base definition:
        y_train[:, 1, 0, 0 ] = np.random.randint(self.cropsize[0], 
            (self.gen_width - self.cropsize[0]), (n_datapoints))
        #Left line base point
        y_train[:, 0, 0, 0 ] = y_train[:, 1, 0, 0 ] - (self.max_road_width - y_noise[:, 0, 0])
        #Right line base point
        y_train[:, 2, 0, 0 ] = y_train[:, 1, 0, 0 ] + (self.max_road_width - y_noise[:, 1, 0])
        #Road start elevation:
        y_train[:, :, 1, 0] = int(self.cropsize[1] * .5)

        #places the vanishing point either on the side of the view window or at the top of the screen
        vanishing_point = np.random.randint(0, (self.gen_width + self.gen_height*2),  (n_datapoints) )

        '''This loop applies the vanishing point and then generates control points in a semi-logical way
        between the road's orgin and vanishing point '''
        for dp_i in range(n_datapoints):
            if(vanishing_point[dp_i] < self.gen_height):
                #Assign the vanishing point:
                y_train[dp_i, :, 1, 2] = vanishing_point[dp_i]
                #Assign the central control point:
                y_train[dp_i, 1, 0, 1] = np.random.randint(0, self.view_width) + self.cropsize[0]
                y_train[dp_i, 1, 1, 1] = np.random.randint(0, self.view_height) + self.cropsize[1]

                #Set the left side no.1 control point
                y_train[dp_i, 0, :, 1] = (y_train[dp_i, 1, :, 1 ] - self.unit_vector(y_train[dp_i, 1, :, 1 ])
                    * np.multiply( (self.max_road_width - y_noise[dp_i, 0, 1]),
                    (1 - self.lane_convergence * y_train[dp_i, 1, 1, 1 ]/self.gen_height) ) )
                #Set the right side no.1 control point
                y_train[dp_i, 2, :, 1] = (y_train[dp_i, 1, :, 1 ] + self.unit_vector(y_train[dp_i, 1, :, 1 ])
                    * np.multiply( (self.max_road_width - y_noise[dp_i, 1, 1]),
                    (1 - self.lane_convergence * y_train[dp_i, 1, 1, 1 ]/self.gen_height) ) )
            elif(vanishing_point[dp_i] < self.gen_height + self.gen_width):
                #define the vanishing point at the top of the camera's perspective
                y_train[dp_i, :, 0, 2] = vanishing_point[dp_i] - self.gen_height
                y_train[dp_i, :, 1, 2] = self.gen_height - 1

                # Define the middle control points as members of a horizontal line chosen with the center point lying in the view window
                #First assign the line containing the control points an elevation:
                y_train[dp_i, :, 1, 1] = np.random.randint(0, self.view_height) + self.cropsize[1]
                #Centerline middle control point definition:
                y_train[dp_i, 1, 0, 1 ] = np.random.randint(self.cropsize[0], 
                    (self.gen_width - self.cropsize[0]) )
                #Left line base point
                y_train[dp_i, 0, 0, 1 ] = y_train[dp_i, 1, 0, 1 ] - np.multiply( 
                    (self.max_road_width - y_noise[dp_i, 0, 1]),
                    (1 - self.lane_convergence * y_train[dp_i, 1, 1, 1 ]/self.gen_height) ) 
                #Right line base point
                y_train[dp_i, 2, 0, 1 ] = y_train[dp_i, 1, 0, 1 ] + np.multiply( 
                    (self.max_road_width - y_noise[dp_i, 1, 1]),
                    (1 - self.lane_convergence * y_train[dp_i, 1, 1, 1 ]/self.gen_height) ) 
            else:
                #Assign the vanishing point to the rhs boundary
                y_train[dp_i, :, 0, 2] = self.gen_width - 1
                y_train[dp_i, :, 1, 2] = vanishing_point[dp_i] - (self.gen_height + self.gen_width)

                #Assign the central control point:
                y_train[dp_i, 1, 0, 1] = np.random.randint(0, self.view_width) + self.cropsize[0]
                y_train[dp_i, 1, 1, 1] = np.random.randint(0, self.view_height) + self.cropsize[1]

                #Set the left side no.1 control point
                y_train[dp_i, 0, :, 1] = (y_train[dp_i, 1, :, 1 ] - self.unit_vector(y_train[dp_i, 1, :, 1 ] - [0, self.gen_width])
                    * np.multiply( (self.max_road_width - y_noise[dp_i, 0, 1]),
                    (1 - self.lane_convergence * y_train[dp_i, 1, 1, 1 ]/self.gen_height) ) )
                #Set the right side no.1 control point
                y_train[dp_i, 2, :, 1] = (y_train[dp_i, 1, :, 1 ] + self.unit_vector(y_train[dp_i, 1, :, 1 ] - [0, self.gen_width])
                    * np.multiply( (self.max_road_width - y_noise[dp_i, 1, 1]),
                    (1 - self.lane_convergence * y_train[dp_i, 1, 1, 1 ]/self.gen_height) ) )

        return y_train

    def poly_line(self, coordinates, line_width, seg_noise = 0):
        #Note that we subtrack generation offsets from the curve coordinates before calculating the line segment locations
        x,y = bezier_curve(coordinates[ 0, : ]-self.cropsize[0],
                 coordinates[1, :] - self.cropsize[1], self.n_segments)
        true_line = np.array([x, y])

        #Add some noise to the line so it's harder to overfit
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
        rr, cc = polygon((polygon_path.astype(int)[1]), polygon_path.astype(int)[0], ( self.view_height, self.view_width) )

        return rr, cc

    # Draws dashed lines like the one in the center of the road
    def dashed_line(self, coordinates, dash_length, dash_width ):
        #estimate the curve lenght to generate a segment count which will approxmiate the desired dash lenght
        est_curve_len = (self.vector_len(coordinates[:,2] - coordinates[:,0] ) + 
                        self.vector_len(coordinates[:,1] - coordinates[:,0] ) + 
                        self.vector_len(coordinates[:,2] - coordinates[:,1] ) )/2
        segments = int(est_curve_len/dash_length)
        x, y = bezier_curve(coordinates[0, :] - self.cropsize[0], 
                coordinates[1, :] - self.cropsize[1], segments)
        dash_line = np.array([x, y])

        #initializing empty indexs
        rrr = np.empty(0, dtype=int)
        ccc = np.empty(0, dtype=int)
        for dash in range( int(segments/2) ):
            offset = .5*dash_width * self.perpendicular(dash_line[:,dash*2]-dash_line[:,dash*2+1])
            d_path = np.array( [ dash_line[:,dash*2] + offset, dash_line[:,dash*2 +1] + offset, 
                        dash_line[:,dash*2+1] - offset, dash_line[:,dash*2 ] - offset,
                        dash_line[:,dash*2] + offset] )
            rr, cc = polygon(d_path.astype(int)[:,1], d_path.astype(int)[:,0], 
                            (self.view_height, self.view_width) )
            rrr = np.append(rrr, rr)
            ccc = np.append(ccc, cc)

        return rrr, ccc

    #converts coordinates into images with curves on them
    def road_generator(self, y_train, line_width, seg_noise = 0):
        road_frame = np.zeros((self.view_height, self.view_width, self.n_channels),
             dtype=np.uint8)

        #line width randomizer:

        line_width += max(line_width/3 *np.random.randn(), -line_width*3/4)

        rr, cc = self.poly_line( y_train[0], line_width, seg_noise)
        road_frame[rr, cc, 0] = 255

        rr, cc = self.dashed_line(y_train[1], self.view_height/4, line_width*2)
        road_frame[rr, cc, 0] = 255

        rr, cc = self.poly_line( y_train[2], line_width, seg_noise)
        road_frame[rr, cc, 0] = 255

        return road_frame #[ :, self.cropsize:(self.cropsize+self.view_width),:]

    #applies canny edge detection algorithm to make generated data behave more like real world data
    def road_refiner(self, road_frame):

        #road_frame = np.reshape(road_frame, (16, 96,1))
        road_frame = cv2.Canny(road_frame, 50, 200)
        #road_frame[road_frame < 128] = 0
        #road_frame[road_frame >= 128] = 255

        return np.reshape(road_frame, (self.input_height, self.input_width, 1) )

    #Saves images in side by side plots
    def save_images(self, Left_Images, Right_Images, directory , titles = ('Input Image','Model Perception') ):
        
        max_intensity = 255
        curves_to_print = min(Right_Images.shape[0],Left_Images.shape[0])


        if not os.path.exists(directory):
            os.makedirs(directory)

        for dp_i in range(curves_to_print):

            #This is the actual save images part
            plt.subplot(1, 2, 1)
            plt.title(titles[0])
            plt.imshow(Left_Images[dp_i,:,:,0], cmap=plt.cm.gray)
            plt.gca().invert_yaxis()

            plt.subplot(1, 2, 2)
            plt.title(titles[1])
            plt.imshow(Right_Images[dp_i,:,:,0], cmap=plt.cm.gray)
            plt.gca().invert_yaxis()

            plt.savefig('%s/%06i.png' % (directory, dp_i), dpi=200, bbox_inches='tight')
            plt.close()


def main():

    parser = argparse.ArgumentParser(description='Roadlike virtual data generator')
    parser.add_argument('--tests', type=int, default=0, metavar='TS', 
                        help='creates a test batch and compares the batch to video data (default is off)')
    args = parser.parse_args()

    roads = Roadgen(cfg)

    #Move these parameters into an argparser
    n_datapoints = int(1E5)
    train_split = 0.9
    n_train_datapoints = int(train_split * n_datapoints)

    train_data_dir = cfg['dir']['train_data']

    #test condition switch
    if args.tests > 0:
        n_datapoints = args.tests

    # Data to store
    #print((n_datapoints, n_channels, height, train_width))
    X_train = np.zeros((n_datapoints, roads.view_height, roads.view_width,
         roads.n_channels), dtype=np.uint8)
    
    y_train = roads.coord_gen(n_datapoints)

    # Generate X
    #Temporary generation location
    for dp_i in range(int(n_datapoints) ):
        X_train[dp_i, :, :, :] = roads.road_refiner(
                                    roads.road_generator(y_train[dp_i], 
                                    roads.line_width, roads.line_wiggle * dp_i%5) )
        print("%.2f%%" % ((100.0 * dp_i / n_datapoints)), end='\r')
    print("Done")

    if args.tests > 0:
        subdir = 'test'
        model = Model(None, None, None)
        roads.save_images(model.video_to_frames(), X_train, '%s/%s' % 
            (train_data_dir, subdir), ['Camera', 'Virtual Generator'] )
    else:
        # Normalize  testing

        X_train = (X_train / np.max( X_train  ) )
        
        X_train = X_train.astype(np.uint8)

        #fixes the label array for use by the learning model
        y_train = roads.model_tranform(y_train)

        #file management stuff
        if not os.path.exists(train_data_dir):
            os.makedirs(train_data_dir)

        # Save Files
        np.save("%s/line_X_train.npy" % (train_data_dir) , X_train[:n_train_datapoints])
        np.save("%s/line_X_val.npy" % (train_data_dir) , X_train[n_train_datapoints:])
        np.save("%s/line_y_train.npy" % (train_data_dir) , y_train[:n_train_datapoints])
        np.save("%s/line_y_val.npy" % (train_data_dir) , y_train[n_train_datapoints:])

if __name__ == "__main__":
    main()