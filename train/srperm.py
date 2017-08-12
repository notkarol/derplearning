#import cv2
import numpy as np
from math import pi
#import os
#import sys
#import tensorflow as tf

'''
Produces side shifts and z axis rotations of training data
'''

#shift label function
'''
	dsteer is the stearing in degrees
		(positive steering is turning left atm)
	drot is the number of degrees the car is rotated about zaxis for perm
		(rotate car left is postiive in degrees)
	mshift is the number of car is shifted along yaxis for perm
		(shift car left is positive in meters)
'''
def shiftsteer(steer, drot, mshift):
	maxsteer = 1 #maximum value which can be returned for steering
	shifttosteer = 1 # ratio of steering correction to lateral shift in steers/m
	spd = 1/30 #linear coefficient for angle correction in steers/degree

	permsteer = steer - drot*spd - mshift*shifttosteer
	if permsteer>0:
		return min(maxsteer, permsteer)
	else:
		return max(-maxsteer, permsteer)


#shift image function
'''
	img is the input image
	drot is the number of pixels the image is rotated about zaxis
		(rotate car left is postiive in degrees)
	mshift is the number of meters the car is shifted along yaxis
		(shift car left is positive in meters)
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

	steer = 5

	mshift = .25
	drot = -5

	'''
	for x in img:
		print(x)
	'''
	perm = shiftimg(img, drot, mshift)

	for y in perm:
		print(y)


	print(steer)

	newsteer = shiftsteer(steer, drot, mshift)

	print(newsteer)


if __name__ == "__main__":
	main()