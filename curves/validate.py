import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from PIL import Image
import matplotlib.pyplot as plt

import numpy as np

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import model_from_yaml

from bezier import bezier_curve, verify_plot

def print_points( labels, model_out):
  print("The image was created using points: ")
  for l in labels:
    print("{} ".format(l ) )
  print("The model returned points: ")
  for m in model_out:
    print("{} ".format(m ) )


def print_curves( val_array, val_points, model_out):

  n_points = 3
  n_segments = 10
  width = 32
  height = 32
  max_intensity = 256
  curves_to_print = val_points.shape[0]

  for i in range(curves_to_print):
   #valdiation data rescalling:
    big_val = np.floor(val_array[i, 0]*max_intensity)

    #Currently raw input to model is disabled.
    #figval = plt.figure(figsize=(width / 10, height / 10))
    #plt.imshow(big_val)
    #plt.imsave('real_%06i.png' % i, big_val, dpi=100)

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


def main():
  
  #Validation parameters
  #number of curves to print:
  image_count = 64

  #load data
  X_val = np.load('X_val.npy')
  y_val = np.load('y_val.npy')

  # load YAML and create model
  yaml_file = open('model.yaml', 'r')
  loaded_model_yaml = yaml_file.read()
  yaml_file.close()
  loaded_model = model_from_yaml(loaded_model_yaml)
  # load weights into new model
  loaded_model.load_weights("model.h5")
  print("Loaded model from disk")

  # evaluate loaded model on test data
  loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
  score = loaded_model.evaluate(X_val[:64], y_val[:64], verbose=0)
  print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]))

  predictions = loaded_model.predict(X_val[:image_count])

  #print_points(y_val[0], predictions[0] )

  print_curves(X_val[:image_count], y_val[:image_count], predictions[:image_count])


if __name__ == "__main__":
    main()