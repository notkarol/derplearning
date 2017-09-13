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
from skimage.draw import line_aa

def print_points( labels, model_out):
  print("The image was created using points: ")
  for l in labels:
    print("{} ".format(l ) )
  print("The model returned points: ")
  for m in model_out:
    print("{} ".format(m ) )

def clamp(array, max_val):
  array[array >= max_val] = max_val - 1
  array[array < 0] = 0
  
    
def print_images(X_val, model_raw):
  #file management stuff
  directory = 'validation_images'
  if not os.path.exists(directory):
    os.makedirs(directory)

  n_lines = 3
  n_points = 3
  n_dimensions = 2

  n_segments = 10

  n_channels = 1
  train_width = 64
  gen_width = train_width+64
  cropsize = int( (gen_width-train_width)/2)
  height = 32
  max_intensity = 256
  curves_to_print = model_raw.shape[0]

  #reshaping the model output vector to make it easier to work with
  model_shaped = np.reshape(model_raw, (model_raw.shape[0], n_lines, n_dimensions, n_points))
  model_out = np.zeros(np.shape(model_shaped), dtype=np.float)
  model_out[:,:,0,:] = model_shaped[:,:,0,:] * gen_width
  model_out[:,:,1,:] = model_shaped[:,:,1,:] * height

  #creating the array to hold the perception projections
  model_view = np.zeros( (curves_to_print, n_channels, height, train_width), dtype=np.float)

  X_large = X_val*max_intensity

  for dp_i in range(curves_to_print):
    model_vgen = np.zeros( (n_channels, height, gen_width), dtype=np.float)
  
    # Generate model perception image
    x0, y0 = bezier_curve(model_out[dp_i, 0, 0, : ], model_out[dp_i, 0, 1, :], n_segments)
    for ls_i in range(len(x0) - 1):
      rr, cc, val = line_aa(int(x0[ls_i]), int(y0[ls_i]), int(x0[ls_i + 1]), int(y0[ls_i + 1]))
      clamp(rr, model_vgen.shape[-1])
      clamp(cc, model_vgen.shape[-2])
      model_vgen[ 0, cc, rr] = val

    x1, y1 = bezier_curve(model_out[dp_i, 1, 0, : ], model_out[dp_i, 1, 1, :], n_segments)
    for ls_i in range(len(x1) - 1):
      rr, cc, val = line_aa(int(x1[ls_i]), int(y1[ls_i]), int(x1[ls_i + 1]), int(y1[ls_i + 1]))
      clamp(rr, model_vgen.shape[-1])
      clamp(cc, model_vgen.shape[-2])
      model_vgen[ 0, cc, rr] = val

    x2, y2 = bezier_curve(model_out[dp_i, 2, 0, : ], model_out[dp_i, 2, 1, :], n_segments)
    for ls_i in range(len(x2) - 1):
      rr, cc, val = line_aa(int(x2[ls_i]), int(y2[ls_i]), int(x2[ls_i + 1]), int(y2[ls_i + 1]))
      clamp(rr, model_vgen.shape[-1])
      clamp(cc, model_vgen.shape[-2])
      model_vgen[ 0, cc, rr] = val

    model_view[dp_i] = model_vgen[:,:, cropsize : (gen_width-cropsize) ]

    plt.subplot(1, 2, 1)
    plt.title('Input Image')
    plt.imshow(X_large[dp_i,0])
    plt.gca().invert_yaxis()

    plt.subplot(1, 2, 2)
    plt.title('Model Perception')
    plt.imshow(model_view[dp_i,0])
    plt.gca().invert_yaxis()

    plt.savefig('%s/image_comparison_%06i.png' % (directory, dp_i), dpi=100, bbox_inches='tight')
    plt.close()


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

  #plot_curves( y_val[:image_count], predictions[:image_count])

  print_images(X_val[:image_count], predictions[:image_count])

if __name__ == "__main__":
    main()
