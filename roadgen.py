 #!/usr/bin/env python3
import os
import PIL
import numpy as np
from skimage.draw import line_aa
from bezier import bezier_curve

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
		self.n_lines = config['line']['n_lines']
		self.n_points = config['line']['n_points']
		self.n_dimensions = config['line']['n_dimensions']
		self.n_channels = config['line']['n_channels']

		self.view_height = config['line']['view_height']
		self.view_width = config['line']['view_width']
		self.gen_width = self.view_width * 2 # buffer onto each horizontal side to allow us to draw curves
		self.cropsize = int((self.gen_width - self.view_width) / 2)

		self.max_road_width = self.gen_width/4
		self.horz_noise_fraction = 0.25

		self.n_segments = config['line']['n_segments']


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
		y_train[:, 1, 1, 1:] = np.sort(np.random.randint(self.view_height/4, 
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
				 y_noise[:, 0, 1, :]),(1 - .8*y_train[:, 1, 1, : ]/self.view_height) )
		y_train[:, 0, 1, : ] = y_train[:, 1, 1, : ]

		#Right lines
		y_train[:, 2, 0, : ] = y_train[:, 1, 0, : ] + np.multiply( 
				(self.max_road_width * (1 - self.horz_noise_fraction) - 
				y_noise[:, 2, 1, :]),(1 - .8*y_train[:, 1, 1, : ]/self.view_height) ) 
		y_train[:, 2, 1, : ] = y_train[:, 1, 1, : ]

		#Invert generation values so that roads are drawn right side up:
		#y_train[:,:, 1, :] = self.view_height - 1 - y_train[:,:, 1, :]

		return y_train

	def road_generator(self, y_train):
		road_frame = np.zeros((self.view_height, self.gen_width, self.n_channels), dtype=np.float)

		for y_line in y_train:
			x, y = bezier_curve(y_line[ 0, : ], y_line[1, :], self.n_segments)
			for ls_i in range(len(x) - 1):
				rr, cc, val = line_aa(int(x[ls_i]), int(y[ls_i]), int(x[ls_i + 1]), int(y[ls_i + 1]))
				road_frame[cc, rr, 0] = val

		return road_frame[:, self.cropsize : (self.gen_width - self.cropsize), :]


def main():

	roads = Roadgen(cfg)

	#Move these parameters into an argparser
	n_datapoints = int(1E5)
	train_split = 0.8
	n_train_datapoints = int(train_split * n_datapoints)

	# Data to store
	#print((n_datapoints, n_channels, height, train_width))
	X_train = np.zeros((n_datapoints, roads.view_height, roads.view_width,
		 roads.n_channels), dtype=np.float)
	
	y_train = roads.coord_gen(n_datapoints)

	# Generate X
	#Temporary generation location
	for dp_i in range(n_datapoints):
		X_train[dp_i, :, :, :] = roads.road_generator(y_train[dp_i])
		print("%.2f%%" % ((100.0 * dp_i / n_datapoints)), end='\r')
	print("Done")

	# Normalize  testing
	X_train *= (1. / np.max(X_train))
	
	#fixes the label array for use by the learning model
	y_train = roads.model_tranform(y_train)


	train_data_dir = cfg['dir']['train_data']
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