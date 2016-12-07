import cv2
import numpy as np
from skimage.segmentation import slic, mark_boundaries
from constants import *

# this method takes the PATH to the folder as input and loads all the files in the folder into the dataset
def load_training_data():
	image_locs = np.genfromtxt("images.txt",dtype = None)
	#imagePATH = PATH + image_locs[1][1]
	features = np.array([]).reshape(0, SQUARE_SIZE * SQUARE_SIZE)
        #train_L = np.array([])
        #train_a = np.array([])
        #train_b = np.array([])
	train_Y = np.array([])
	train_U = np.array([])
	train_V = np.array([])	

	for i in range(len(image_locs)):
		#imagePath = PATH + image_locs[i][1]
		imagePath = image_locs[i][1]
		print "loading data from "+ imagePath + " ... "
		#subsquares, L, a, b = extract_features(imagePath)
		subsquares, Y,U,V = extract_features(imagePath)
		features = np.concatenate((features, subsquares), axis=0)
            	train_Y = np.concatenate((train_Y, Y), axis=0)
            	train_U = np.concatenate((train_U, U), axis=0)
		train_V = np.concatenate((train_V, V), axis=0)

		
		
		#image = cv2.imread(imagePath)
		#print image
		#cv2.imshow("image",image)
		#cv2.waitKey(100)
	#return features, train_L, train_a, train_b #will be returning the training and testing sets after this
	return features, train_Y, train_U, train_V #will be returning the training and testing sets after this

def load_test_data():
	image_locs = np.genfromtxt("images.txt",dtype = None)
	#imagePATH = PATH + image_locs[1][1]
	features = np.array([]).reshape(0, SQUARE_SIZE * SQUARE_SIZE)
        test_Y = np.array([])
        test_U = np.array([])
        test_V = np.array([])
	for i in range(len(image_locs)):
		imagePath = PATH + image_locs[i][1]
		#print imagePath
		subsquares, Y,U,V = extract_features(imagePath)
		features = np.concatenate((features, subsquares), axis=0)
            	test_Y = np.concatenate((test_Y, Y), axis=0)
            	test_U = np.concatenate((test_U, U), axis=0)
		test_V = np.concatenate((test_V, V), axis=0)
		#image = cv2.imread(imagePath)
		#print image
		#cv2.imshow("image",image)
		#cv2.waitKey(100)
	return features, test_Y, test_U, test_V #will be returning the training and testing sets after this
	

def segment_image(path):
	image = cv2.imread(path)	
	gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	yuv  = cv2.cvtColor(image,cv2.COLOR_BGR2YUV)
	segments = slic(gray_image, n_segments=N_SEGMENTS, compactness=0.1, sigma=1)
	
	test = cv2.cvtColor(yuv,cv2.COLOR_YUV2BGR)
	cv2.imshow('test',test)
	return gray_image, yuv, segments

def extract_features(path):
	#image = cv2.imread(path)
	#segments = slic(gray_image, n_segments=N_SEGMENTS, compactness=10, sigma=1)
	gray_image, yuv , segments = segment_image(path)
	n_segments = segments.max() + 1

	#compute centroids
	point_count = np.zeros(n_segments)
	centroids = np.zeros((n_segments, 2))
    	Y = np.zeros(n_segments)
    	U = np.zeros(n_segments)
    	V = np.zeros(n_segments)
	for (i,j), value in np.ndenumerate(segments):
		point_count[value] += 1
		centroids[value][0] += i
		centroids[value][1] += j
		Y[value] += yuv[i][j][0]        
		U[value] += yuv[i][j][1]
		V[value] += yuv[i][j][2]
		#print "Y is ", yuv[i][j][1], " U is :",yuv[i][j][2]

	for k in range(n_segments):
		centroids[k] /= point_count[k]
		Y[k] /= point_count[k]
		U[k] /= point_count[k]
		V[k] /= point_count[k]

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

	return np.abs(subsquares), Y,U,V

def predict_image(svr_Y, svr_U, svr_V, path):
	image = cv2.imread(path)	
	gray_image, yuv , segments = segment_image(path)
	n_segments = segments.max() + 1
	subsquares, Y,U,V = extract_features(path)
	
	Y_image = np.zeros_like(gray_image)
	U_image = np.zeros_like(gray_image)
	V_image = np.zeros_like(gray_image)
	#prediction of LAB
	predicted_Y = np.zeros(n_segments)
    	predicted_U = np.zeros(n_segments)
    	predicted_V = np.zeros(n_segments)	
	for k in range(n_segments):
		predicted_Y[k] = svr_Y.predict(subsquares[k])

		predicted_U[k] = svr_U.predict(subsquares[k])
		
		predicted_V[k] = svr_V.predict(subsquares[k])

	# Apply MRF to smooth out colorings
#    	predicted_u, predicted_v = apply_mrf(predicted_u, predicted_v, segments, n_segments, img, subsquares)
	
	print "problem is here"

	#image reconstruction	
	for (i,j), value in np.ndenumerate(segments):
		Y_image[i,j] = predicted_Y[value]
		
		U_image[i,j] = predicted_U[value]
		V_image[i,j] = predicted_V[value]
    	#rgb = retrieveRGB(yuv)
	Y_image = gray_image	
	merged = cv2.merge([Y_image,U_image,V_image])
	
	predicted_image = cv2.cvtColor(merged,cv2.COLOR_YUV2BGR)
	print "predicted"
	cv2.imshow("merged YUV",merged)
	cv2.imshow("pred", predicted_image)

	return predicted_image

