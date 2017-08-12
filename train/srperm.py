#import cv2
import numpy as np
from math import pi
#import os
#import sys
#import tensorflow as tf

'''
Produces side shifts and z axis rotations of training data
'''

#shift function
'''
	img is the input image
	horz is the measure of pixels above the horizon
	rot is the number of pixels the image is rotated about zaxis
		(rotate car left is postiive)
	shift is the number of pixes the bottom of the image is shifted along yaxis
		(shift car left is positive)
'''
def shiftimg(img, drot, mshift):
	perm = np.zeros( np.shape(img) )

	phorz = horizonset(len(img))

	prot = zdegtopixel(drot, len(img[0]) )
	pshift = ymetertopixel(mshift, len(img[0]) )

	for z,row in enumerate(img):
		
		#Calculates the shift distance for a given row
		shift_dist = prot + pshift*(max(0,(z+1-phorz) )/(len(img)-phorz ) )
		shift_count = int(round(shift_dist,0) )

		#Executes the called for shift accross the row
		if shift_count == 0:
			perm[z] = row
		elif shift_count >0:
			for y, pixel in enumerate(row):
				if y+shift_count <len(row):
					perm[z][y+shift_count] = row[y]
		elif shift_count <0 :
			for y, pixel in enumerate(row):
				if y+shift_count >=0:
					perm[z][y+shift_count] = row[y]

	return perm


#calculates the number of pixels the image has rotated
# for a given degree rotation of the camera
def zdegtopixel(deg, iydim):
	cfovz = 80 #camera view arc about zaxis in degrees

	return deg*iydim/cfovz


#converts a displacement of the car in y meters to pixels along the bottom row of the image
def ymetertopixel(disp, iydim):
	cfovz = 80 #camera view arc about zaxis in degrees

	cheight = .38 #camera height in meters
	minvis = .7 #FIXME minimum distance camera can see (must be measured)

	botwidth = 2*np.tan(cfovz*pi/(2*180) )*(cheight**2+minvis**2)**.5

	return disp*iydim/botwidth


def horizonset(izdim):
	cfovy = 60 #camera view arc about yaxis in degrees

	cheight = .38 #camera height in meters
	minvis = .7 #FIXME minimum distance camera can see (must be measured)

	return izdim*( (cfovy-np.arctan(cheight/minvis)*180/pi )/cfovy )



def main():
	img = [[1,2,3,4,5,6,7,8,9,0],
				[1,2,3,4,5,6,7,8,9,0],
				[1,2,3,4,5,6,7,8,9,0],
				[1,2,3,4,5,6,7,8,9,0],
				[1,2,3,4,5,6,7,8,9,0],
				[1,2,3,4,5,6,7,8,9,0],
				[1,2,3,4,5,6,7,8,9,0],
				[1,2,3,4,5,6,7,8,9,0],
				[1,2,3,4,5,6,7,8,9,0],
				[1,2,3,4,5,6,7,8,9,0]]

	'''
	for x in img:
		print(x)
	'''
	perm = shiftimg(img, 0, 0.5)

	for y in perm:
		print(y)


if __name__ == "__main__":
	main()