import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import argparse
import sys
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_yaml
#from skimage.draw import line_aa
from scipy.misc import imread
from model import Model
#from bezier import bezier_curve, verify_plot
from roadgen3d import Roadgen

'''
Contains functions which validate the quality of road_gen, a model on virtual data,
and a model on video data.
'''

import yaml
with open("config/line_model.yaml", 'r') as yamlfile:
        cfg = yaml.load(yamlfile)
'''
#Loads png images as arrays and assembles them into block
def load_images(directory, filenames)
    first = imread('%s/%s' % (directory, filenames[0] ) )

    block = np.zeros( (filenames.shape[0], first.shape), dtype= np.uint8 )

    for frame in range(filenames):
        block[frame] = imread('%s/%s' % (directory, filenames[0] ) )
    return block
'''

#Prints to command line the points used to produce an image
def print_points( labels, model_out):
    print("The image was created using points: ")
    for l in labels:
        print("{} ".format(l ) )
    print("The model returned points: ")
    for m in model_out:
        print("{} ".format(m ) ) 

    
#Saves images in side by side plots
def compare_io(X_val, model_raw, directory = '%s/image_comparison' % cfg['dir']['validation']):
    #Creates tensors to compare to source images, plots both side by side, and saves the plots
    road = Roadgen(cfg)

    curves_to_print = model_raw.shape[0]

    #reshaping the model output vector to make it easier to work with
    model_out = road.model_interpret(model_raw)
    print('predictions denormalized')
    #initialize the model view tensor
    model_view = np.zeros( (curves_to_print, road.view_height, road.view_width, road.n_channels), dtype=np.uint8)

    for prnt_i in range(curves_to_print):
        model_view[prnt_i] = road.road_generator(model_out[prnt_i], road.line_width/2, rand_gen=0) 

    road.save_images(X_val, model_view, directory )


#Prints plot of curves against the training data Saves plots in files
#Because the model outputs are not put into a drawing function it is easier for audiences 
# to understand the model output data.
#FIXME function is still built to work like v1 generation also may have bugs in the plotter function
def plot_curves( val_points, model_out):

    n_points = 3
    n_segments = 10
    width = 64
    height = 64
    max_intensity = 255
    curves_to_print = model_out.shape[0]

    for i in range(curves_to_print):
     
        #Turns the validation points into a plottable curve
        x_val_points = val_points[i, :n_points]*width
        y_val_points = val_points[i, n_points:]*height
        x_val_curve, y_val_curve = bezier_curve(x_val_points, y_val_points, n_segments)
        
        #Turns the model output curve into a plottable curve
        x_model_points = model_out[i, :n_points]*width
        y_model_points = model_out[i, n_points:]*height
        x_mod_curve, y_mod_curve = bezier_curve(x_model_points, y_model_points, n_segments)
        
        fig = plt.figure(figsize=(width / 10, height / 10))
        plt.plot(x_val_curve, y_val_curve, 'k-')
        plt.plot(x_mod_curve, y_mod_curve, 'r-')
        plt.xlim(0, width)
        plt.ylim(0, height)
        plt.gca().invert_yaxis()
        plt.savefig('valplots/model_%06i.png' % i, dpi=100, bbox_inches='tight')
        plt.close()


#function invokes Model class, and saves the predictions as images
def val_training(X_val, loaded_model, directory ):
    #predictions = np.zeros((X_val.shape[0], cfg['line']['n_lines'], 
     #       cfg['line']['n_dimensions'], cfg['line']['n_points']), np.float )
    road = Roadgen(cfg)

    #apply the model to generate coordinate prediction
    predictions = loaded_model.road_spotter(road.normalize(X_val) )
    print('predictions made')

    compare_io(X_val=X_val, model_raw=predictions, directory=directory)
    print("Validation images saved to: %s" % directory )


'''
def mp4_validate(X_raw, X_val, loaded_model, directory):
    #Creates tensors to compare to source images, plots both side by side, and saves the plots
    road = Roadgen(cfg)

    #Runing the model on the loaded data:
    predictions = loaded_model.road_spotter(X_val)
    curves_to_print = X_val.shape[0]

    #reshaping the model output vector to make it easier to work with
    model_out = road.model_interpret(predictions)

    #initialize the model view tensor
    model_view = np.zeros( (curves_to_print, road.view_height, road.view_width, road.n_channels), dtype=np.uint8)

    for prnt_i in range(curves_to_print):
        model_view[prnt_i] = road.denormalize(
            road.road_generator(model_out[prnt_i], road.line_width/2) )
        
    road.save_video(X_raw, model_view, '%s' % (directory) )
    print("Validation gif saved to: %s" %(directory) )
 '''

def main():
    
    #Arguments which select which form of validation to conduct
    parser = argparse.ArgumentParser(
        description='Functions for validating model related code and trained models')
    parser.add_argument('--roadgen', type=int, default=0,
        help='designates a number of training images to be compared against camera data. Does not require a trained model. (default=0)')
    parser.add_argument('--vr', type=int, default=0, 
        help='defines a number of images from the validation data to load for validation analysis(default=0)')
    #"data/20170812T214343Z-paras" is a good default video to select
    parser.add_argument('--video', type=int, default=0,
        help='defines a video source file for real world validation. When empty disables video validation')
    parser.add_argument('--gif', type=int, default=0, 
        help='When!=0 creates a gif of the video validation')
    args = parser.parse_args()

    #Set max number of frames to validate:
    val_count = 256

    #file management stuff
    directory = "%s/ver_%s" % (cfg['dir']['validation'], cfg['dir']['model_name'])
    subdirectory = 'virtual_comparison'
    
    #loading the model
    model_path = "%s/%s" % (cfg['dir']['model'], cfg['dir']['model_name'] )
    loaded_model = Model(None, '%s.yaml' % model_path, '%s.h5' % model_path)

    #Pulling up roadgen to help deal with virtually generated data:
    road = Roadgen(cfg)

    if args.roadgen:
        test_dir = 'line_train_data/test_data'
        comp_dir = 'line_train_data/test_comparison'

        road.batch_gen(n_datapoints=args.roadgen, data_dir=test_dir)
        
        X_train = road.batch_loader(data_dir=test_dir, batch_iter=0)
        #y_train = np.load("%s/y_%03i.npy" % (test_dir, 0) )

        road.save_images(loaded_model.video_to_frames(edge_detect=0, channels_out=road.n_channels),
             X_train, '%s' % (comp_dir), 
             ['Camera', 'Virtual Generator'] )

    #Virtual Validation branch
    if args.vr:
        #load data
        X_large = road.batch_loader(cfg['dir']['train_data'], 0)
        #y_val = np.load('%s/line_y_val_000.npy' % cfg['dir']['train_data'])

        #Validates against virtually generated data
        val_training(X_large[:args.vr], loaded_model, '%s/%s' % (directory, subdirectory) )
    
    if args.video:
        #Loading video data for validation:
        folder = "data/20170812T214343Z-paras"
        X_video = loaded_model.video_to_frames(folder, args.video, edge_detect=0, channels_out=3)
        
        #Validates model against recorded data
        subdirectory = 'video_comparison' 
        
        #Creating video validation comparison images:
        val_training(X_video, loaded_model, '%s/%s' % (directory, subdirectory) )

    if args.gif:
        from giffer import create_gif
        subdirectory = 'video_comparison'
        create_gif(n_images=args.gif, source_directory='%s/%s' % (directory, subdirectory), 
            output_name='%s/%s' % (directory, subdirectory), duration=.1)
        print('gif created with name: %s/%s' % (directory, subdirectory) )
        
    

if __name__ == "__main__":
        main()
