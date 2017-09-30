import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_yaml
from skimage.draw import line_aa

from model import Model
from bezier import bezier_curve, verify_plot
from roadgen import Roadgen
'''
Function List:
    print_points
    clamp
    save_images
    plot_curves
    val_training
    video_to_frames
    main
'''

import yaml
with open("config/line_model.yaml", 'r') as yamlfile:
        cfg = yaml.load(yamlfile)

#Prints to command line the points used to produce an image
def print_points( labels, model_out):
    print("The image was created using points: ")
    for l in labels:
        print("{} ".format(l ) )
    print("The model returned points: ")
    for m in model_out:
        print("{} ".format(m ) )

    
#   
def save_images(X_val, model_raw, directory = cfg['dir']['validation']):
    #file management stuff
    if not os.path.exists(directory):
        os.makedirs(directory)

    #Creates tensors to compare to source images, plots both side by side, and saves the plots
    road = Roadgen(cfg)

    max_intensity = 256
    curves_to_print = model_raw.shape[0]

    #reshaping the model output vector to make it easier to work with
    model_out = road.model_interpret(model_raw)

    #Restoring the training data to a displayable color range
    X_large = X_val * max_intensity

    #file management stuff
    subdirectory = ('image_comparison_%s' % 
                (cfg['dir']['model_name']) )
    if not os.path.exists('%s/%s' % (directory, subdirectory)):
        os.makedirs('%s/%s' % (directory, subdirectory) )

    for dp_i in range(curves_to_print):
        model_view = road.road_generator(model_out[dp_i])

        #This is the actual save images part
        plt.subplot(1, 2, 1)
        plt.title('Input Image')
        plt.imshow(X_large[dp_i,:,:,0], cmap=plt.cm.gray)
        #plt.plot(x0, y0, 'r-')
        #plt.plot(x1, y1, 'y-')
        #plt.plot(x2, y2, 'r-')
        plt.gca().invert_yaxis()

        plt.subplot(1, 2, 2)
        plt.title('Model Perception')
        plt.imshow(model_view[:,:,0], cmap=plt.cm.gray)
        plt.gca().invert_yaxis()

        plt.savefig('%s/%s/%06i.png' % (directory, subdirectory, dp_i), dpi=100, bbox_inches='tight')
        plt.close()

#Prints plot of curves against the training data Saves plots in files
#Because the model outputs are not put into a drawing function it is easier for audiences 
# to understand the model output data.
#FIXME function is still built to work like v1 generation also may have bugs in the plotter function
def plot_curves( val_points, model_out):

    n_points = 3
    n_segments = 10
    width = 64
    height = 64
    max_intensity = 256
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

#function loads model parameters
def load_model(model_path):
    # load YAML and create model
    yaml_file = open('%s.yaml' % model_path, 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    loaded_model.load_weights('%s.h5' % model_path)
    print("Loaded model from disk")

    return loaded_model

#function calls load_model, runs the model, and saves the predictions
#FIXME: add either a save directory passed variable or a filename prefix variable (or both)
#   such that we can easily run validation on both realworld images and virtual images simultaneously
def val_training(X_val, model_path = "%s/%s" % (cfg['dir']['model'], cfg['dir']['model_name'] ) ):
    
    loaded_model = load_model(model_path)

    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    #score = loaded_model.evaluate(X_val[:64], y_val[:64], verbose=0)
    #print("%s: %.2f%%" % (loaded_model.metrics_names[0], score[0]) )

    #apply the model to generate coordinate prediction
    predictions = loaded_model.predict(X_val)

    directory = cfg['dir']['validation']

    #file management stuff
    if not os.path.exists(directory):
        os.makedirs(directory)

    save_images(X_val, predictions, directory)
    print("Validation images saved to: %s" %directory)

#extracts frames from video for use by the validation function
#this allows us to validate the model with real world images instead of simulated images.
def video_to_frames(folder="data/20170812T214343Z-paras", max_frames=32):
    # Prepare video frames by extracting the patch and thumbnail for training
    video_path = os.path.join(folder, 'video.mp4')
    print(video_path)
    video_cap = cv2.VideoCapture(video_path)

    #initializing the car's perspective
    viewer = Model(None, None, None)

    #initializing the output array
    frames = np.zeros([max_frames, 64, 128, 1])

    counter = 0
    while video_cap.isOpened() and counter < max_frames:
        # Get the frame
        ret, frame = video_cap.read()
        if not ret: break

        prepared = viewer.preprocess(frame)[0]
        prepared = cv2.cvtColor(prepared, cv2.COLOR_BGR2GRAY)
        prepared = cv2.flip(prepared, 0)
        prepared = cv2.Canny(prepared,100,200)
        prepared[prepared < 128] = 0
        prepared[prepared >= 128] = 255
        prepared = np.reshape(prepared, (64, 128,1))
        frames[counter] = prepared
        counter += 1

    #cleanup      
    video_cap.release()

    # Return our batch
    return frames



def main():
    
    #load data
    #X_val = np.load('%s/line_X_val.npy' % cfg['dir']['train_data'])
    #y_val = np.load('%s/line_y_val.npy' % cfg['dir']['train_data'])
    X_val = video_to_frames()

    val_count = 64
    val_training(X_val[:val_count])
    

if __name__ == "__main__":
        main()
