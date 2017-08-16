import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import model_from_yaml

from bezier.py import bezier_curve, verify_plot

def print_points( labels, model_out):
	print("The image was created using points: ")
	for l in labels:
		print("{} ".format(l ) )
	print("The model returned points: ")
	for m in model_out:
		print("{} ".format(m ) )


def print_curves( val_image, model_out):



def main():

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
	score = loaded_model.evaluate(X_val[0:9], y_val[0:9], verbose=0)
	print("%s: %.2f%%" % (loaded_model.metrics_names[2], score[2]))

	predictions = loaded_model.predict(X_val[0:31])

	print_points(predictions[0] , y_val[0])



if __name__ == "__main__":
    main()