'''Used to generate txt world files from heightmap images'''

import numpy as np
import cv2
import sys
import math

# what is the range of heights we want our world to have?
HEIGHT_RANGE = 24
# how many columns do we want our world to have
NUM_BLOCKS = 50000


def depthimage2world(img, outpath, height_range = HEIGHT_RANGE, num_blocks = NUM_BLOCKS):
	'''Takes a numpy array and saves a txt world to the filepath'''
	# open our output file
	o = open(outpath, 'w')
	
	yscale = calc_y_scale(img, height_range)
	xzscale = calc_xz_scale(img, num_blocks, yscale)

	height, width = img.shape

	# how far do we step over in our image for each sample?
	xzstep = 1.0 / xzscale

	# initiate our index variables for our blocks
	i, j = 0, 0
	# go through and sample the image
	for x in frange(0, height, xzstep):
		x = int(round(x))
		for z in frange(0, width, xzstep):
			z = int(round(z))
			# get the raw data, then round and convert to int
			y = yscale*img[x,z]
			y = int(round(y))
			# fill in this column at (i,j)
			for eachy in range(1, y):
				o.write('{} {} {}\n'.format(i, eachy, j))
			i+=1
		i=0
		j+=1
	o.close()

def frange(x, y, jump):
	while x < y:
		yield x
		x += jump

def calc_y_scale(img, height_range):
	imgrange = np.max(img) - np.min(img)
	return float(height_range)/imgrange

def calc_xz_scale(img, desired_num, yscale):
	'''Determines how much to scale each dimension to get about the right number of blocks'''
	numblocks = np.sum(img) * yscale
	# since this factors into each dimension individually, the effect of the scale is squared
	return math.sqrt(float(desired_num)/numblocks)

if __name__ == '__main__':
	if len(sys.argv) == 5:
		# read the imput image path
		# the output .txt file path
		# the desired height range of the world
		# and the desired number of blocks in the world
		inpath, outpath, height_range, num_blocks = sys.argv[1:]
	else:
		print 'usage: python heightmap.py inpath outpath height_range num_blocks'
		exit()

	# open the image
	img = cv2.imread(inpath, cv2.IMREAD_GRAYSCALE)
	# convert the image to a txt world file
	depthimage2world(img, outpath, height_range = int(height_range), num_blocks = int(num_blocks))
