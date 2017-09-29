 #!/usr/bin/env python3
import os
import PIL
import numpy as np
from skimage.draw import line_aa
from bezier import bezier_curve

'''
generates 3 line road training data does not have any functions
v2 only draws 3 lines on the ground. it is a work in progress.
'''

import yaml
with open("config/line_model.yaml", 'r') as yamlfile:
    cfg = yaml.load(yamlfile)

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
		y_train[:,:, 1, :] = self.view_height - 1 - y_train[:,:, 1, :]

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
	n_datapoints = int(1E3)
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

	# Normalize training and testing
	X_train *= (1. / np.max(X_train))
	y_train[:, :, 0, :] /= roads.gen_width
	y_train[:, :, 1, :] /= roads.view_height

	#fixes the label array for use by the learning model
	y_train = np.reshape(y_train, (n_datapoints, roads.n_lines * roads.n_points *
									 roads.n_dimensions) )


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