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
        self.cropsize = int((self.gen_width - self.view_width) / 2)

        #parameters of the final image size to be passed to the CNN
        self.input_width = config['line']['input_width']
        self.input_height = config['line']['input_height']

        #Attributes of the road line drawing.
        self.max_road_width = self.gen_width/3.5 #The widest possible road that can be drawn (radius)
        self.min_road_height = self.view_height/2 #The minimum value for the 2nd and 3rd road control points
        self.horz_noise_fraction = 0.3 #size of the noise envelope below max_width where road lines may exist
        self.lane_convergence = .6 #rate at which lanes converge approaching horizon

        #parameters to be used by the drawing function road_gen
        self.n_segments = config['line']['n_segments']
        self.line_width = self.view_width/30

    def __del__(self):
        #Deconstructor
        pass

    #Reshape a label tensor to 2d and normalizes data for use by the ANN:
    def model_tranform(self, nd_labels):
        
        #normalization
        nd_labels[:, :, 0, :] /= self.gen_width
        nd_labels[:, :, 1, :] /= self.view_height

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

        #Clamp model outputs (consider making this switched)
        nd_labels[:,:,0,:] = self.clamp(nd_labels[:,:,0,:], self.gen_width)
        nd_labels[:,:,1,:] = self.clamp(nd_labels[:,:,1,:], self.view_height)

        return nd_labels

    #clamp function to prevent predictions from exceeding the drawing boundaries of plot_curves
    def clamp(self, array, max_val):
        array[array >= max_val] = max_val - 1
        array[array < 0] = 0

        return array

    # Generate coordinate of beizier control points for a road
    def coord_gen(self, n_datapoints):
        y_train = np.zeros( (n_datapoints, self.n_lines, self.n_dimensions, 
            self.n_points), np.float)

        #Centerline:
        y_train[:, 1, 0, : ] = np.random.randint(self.max_road_width, 
            (self.gen_width - self.max_road_width), (n_datapoints, self.n_points))
        y_train[:, 1, 1, 1:] = np.sort(np.random.randint(self.min_road_height, 
            self.view_height, (n_datapoints, (self.n_points - 1) ) ) )
        #note that by sorting the height control points we get non-uniform distributions

        #noise for the side lines
        y_noise = np.zeros((n_datapoints, self.n_lines, self.n_dimensions,
            self.n_points) , np.float)
        y_noise = np.random.randint(0, self.max_road_width * self.horz_noise_fraction,
            (n_datapoints, self.n_lines, self.n_dimensions, self.n_points) )

        #Left lines
        y_train[:, 0, 0, : ] = y_train[:, 1, 0, : ] - np.multiply( 
                (self.max_road_width * (1 - self.horz_noise_fraction) -
                 y_noise[:, 0, 1, :]),(1 - self.lane_convergence * 
                 y_train[:, 1, 1, : ]/self.view_height) )
        y_train[:, 0, 1, : ] = y_train[:, 1, 1, : ]

        #Right lines
        y_train[:, 2, 0, : ] = y_train[:, 1, 0, : ] + np.multiply( 
                (self.max_road_width * (1 - self.horz_noise_fraction) - 
                y_noise[:, 2, 1, :]),(1 - self.lane_convergence * 
                y_train[:, 1, 1, : ]/self.view_height) ) 
        y_train[:, 2, 1, : ] = y_train[:, 1, 1, : ]

        #Invert generation values so that roads are drawn right side up:
        #y_train[:,:, 1, :] = self.view_height - 1 - y_train[:,:, 1, :]

        return y_train

    #converts coordinates into images with curves on them
    def road_generator(self, y_train, line_width):
        road_frame = np.zeros((self.view_height, self.gen_width, self.n_channels),
             dtype=np.uint8)

        for y_line in y_train:
            x, y = bezier_curve(y_line[ 0, : ], y_line[1, :], self.n_segments)
            for ls_i in range(len(x) - 1):
                c = [int(x[ls_i] ), int(x[ls_i] + line_width), int(x[ls_i+1] + 
                    line_width), int(x[ls_i+1]), int(x[ls_i])]
                r = [int(y[ls_i]), int(y[ls_i]), int(y[ls_i+1]), 
                    int(y[ls_i+1]), int(y[ls_i] ) ]
                rr, cc = polygon(r, c, ( self.view_height, self.gen_width) )
                road_frame[rr, cc, 0] = 255

        return road_frame[ :, self.cropsize:(self.cropsize+self.view_width),:]

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
            #plt.plot(x0, y0, 'r-')
            #plt.plot(x1, y1, 'y-')
            #plt.plot(x2, y2, 'r-')
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
         roads.n_channels), dtype=np.float)
    
    y_train = roads.coord_gen(n_datapoints)

    # Generate X
    #Temporary generation location
    for dp_i in range(int(n_datapoints/4) ):
        X_train[4*dp_i, :, :, :] = roads.road_refiner(
                                 roads.road_generator(y_train[4*dp_i], roads.line_width) )
        X_train[4*dp_i+1, :, :, :] = roads.road_refiner(
                                 roads.road_generator(y_train[4*dp_i+1], roads.line_width/2) )
        X_train[4*dp_i+2, :, :, :] = roads.road_generator(y_train[4*dp_i+2], roads.line_width)
        X_train[4*dp_i+3, :, :, :] = roads.road_generator(y_train[4*dp_i+3], roads.line_width/2)

        print("%.2f%%" % ((100.0 * dp_i*4 / n_datapoints)), end='\r')
    print("Done")

    if args.tests > 0:
        subdir = 'test'
        model = Model(None, None, None)
        roads.save_images(model.video_to_frames(), X_train, '%s/%s' %(train_data_dir, subdir))
    else:
        # Normalize  testing
        X_train *= (1. / np.max( [np.max(X_train ), 0] ) )
        
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