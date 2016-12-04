import cv2
import numpy as np
from skimage.segmentation import slic, mark_boundaries
from constants import *

# this method takes the PATH to the folder as input and loads all the files in the folder into the dataset
def load_data():
	
	image_locs = np.genfromtxt("images.txt",dtype = None)
	#imagePATH = PATH + image_locs[1][1]
	for i in range(len(image_locs)):
		imagePath = PATH + image_locs[i][1]
		#print imagePath		
		image = cv2.imread(imagePath)
		#print image
		cv2.imshow("image",image)
		cv2.waitKey(100)
	return imagePath #will be returning the training and testing sets after this


def segment_image(image):
	segments = slic(img, n_segments=N_SEGMENTS, compactness=10, sigma=1)
	return segments

def get_subsquares(image):
	return

def extract_features(paths):
	return
