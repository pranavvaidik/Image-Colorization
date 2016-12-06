import cv2
import numpy as np
from skimage.segmentation import slic, mark_boundaries
from constants import *

# this method takes the PATH to the folder as input and loads all the files in the folder into the dataset
def load_training_data():
	image_locs = np.genfromtxt("images.txt",dtype = None)
	#imagePATH = PATH + image_locs[1][1]
	features = np.array([]).reshape(0, SQUARE_SIZE * SQUARE_SIZE)
        train_L = np.array([])
        train_a = np.array([])
        train_b = np.array([])
	for i in range(len(image_locs)):
		imagePath = PATH + image_locs[i][1]
		print "loading data from "+ imagePath + " ... "
		subsquares, L, a, b = extract_features(imagePath)
		features = np.concatenate((features, subsquares), axis=0)
            	train_L = np.concatenate((train_L, L), axis=0)
            	train_a = np.concatenate((train_a, a), axis=0)
		train_b = np.concatenate((train_b, b), axis=0)
		#image = cv2.imread(imagePath)
		#print image
		#cv2.imshow("image",image)
		#cv2.waitKey(100)
	return features, train_L, train_a, train_b #will be returning the training and testing sets after this

def load_test_data():
	image_locs = np.genfromtxt("images.txt",dtype = None)
	#imagePATH = PATH + image_locs[1][1]
	features = np.array([]).reshape(0, SQUARE_SIZE * SQUARE_SIZE)
        test_L = np.array([])
        test_a = np.array([])
        test_b = np.array([])
	for i in range(len(image_locs)):
		imagePath = PATH + image_locs[i][1]
		#print imagePath
		subsquares, L, a, b = extract_features(imagePath)
		features = np.concatenate((features, subsquares), axis=0)
            	test_L = np.concatenate((test_L, L), axis=0)
            	test_a = np.concatenate((test_a, a), axis=0)
		test_b = np.concatenate((test_b, b), axis=0)
		#image = cv2.imread(imagePath)
		#print image
		#cv2.imshow("image",image)
		#cv2.waitKey(100)
	return features, test_L, test_a, test_b #will be returning the training and testing sets after this
	

def segment_image(path):
	image = cv2.imread(path)	
	gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	Lab  = cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
	segments = slic(gray_image, n_segments=N_SEGMENTS, compactness=0.1, sigma=1)
	return gray_image, Lab, segments

def extract_features(path):
	#image = cv2.imread(path)
	#segments = slic(gray_image, n_segments=N_SEGMENTS, compactness=10, sigma=1)
	gray_image, Lab , segments = segment_image(path)
	n_segments = segments.max() + 1

	#compute centroids
	point_count = np.zeros(n_segments)
	centroids = np.zeros((n_segments, 2))
    	L = np.zeros(n_segments)
    	a = np.zeros(n_segments)
    	b = np.zeros(n_segments)
	for (i,j), value in np.ndenumerate(segments):
		point_count[value] += 1
		centroids[value][0] += i
		centroids[value][1] += j
		L[value] += Lab[i][j][0]        
		a[value] += Lab[i][j][1]
		b[value] += Lab[i][j][2]

	for k in range(n_segments):
		centroids[k] /= point_count[k]
		L[k] /= point_count[k]
		a[k] /= point_count[k]
		b[k] /= point_count[k]

	subsquares = np.zeros((n_segments, SQUARE_SIZE * SQUARE_SIZE))
	for k in range(n_segments):
        	# Check that the square lies completely within the image
	        top = max(int(centroids[k][0]), 0)
        	if top + SQUARE_SIZE >= gray_image.shape[0]:
            		top = gray_image.shape[0] - 1 - SQUARE_SIZE
        	left = max(int(centroids[k][1]), 0)
        	if left + SQUARE_SIZE >= gray_image.shape[1]:
            		left = gray_image.shape[1] - 1 - SQUARE_SIZE
        	for i in range(0, SQUARE_SIZE):
            		for j in range(0, SQUARE_SIZE):
                		subsquares[k][i*SQUARE_SIZE + j] = gray_image[i + top][j + left]
        	subsquares[k] = np.fft.fft2(subsquares[k].reshape(SQUARE_SIZE, SQUARE_SIZE)).reshape(SQUARE_SIZE * SQUARE_SIZE)

	return np.abs(subsquares), L, a, b
