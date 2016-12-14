import cv2
import numpy as np
from skimage.segmentation import slic, mark_boundaries
from constants import *
from skimage.data import imread
from skimage.util import img_as_float

def retrieveRGB(img):
    rgb = np.dot(img, RGB_FROM_YUV)
    for (i, j, k), value in np.ndenumerate(rgb):
        rgb[i][j][k] = clamp(rgb[i][j][k], 0, 1)
    return rgb

def retrieveYUV(img):
    return np.dot(img, YUV_FROM_RGB)

def clamp(val, low, high):
    return max(min(val, high), low)

def clampU(val):
    return clamp(val, -U_MAX, U_MAX)

def clampV(val):
    return clamp(val, -V_MAX, U_MAX)



# this method takes the PATH to the folder as input and loads all the files in the folder into the dataset
def load_training_data():
	image_locs = np.genfromtxt("images.txt",dtype = None)
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
	#image_locs = np.genfromtxt("images.txt",dtype = None)
	imagePATH = PATH + image_locs[1][1]
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
	img = img_as_float(imread(path))
	image = cv2.imread(path)
	gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	gray_image = img_as_float(gray_image)
	yuv  = retrieveYUV(img)#cv2.cvtColor(image,cv2.COLOR_BGR2YUV)
	segments = slic(gray_image, n_segments=N_SEGMENTS, compactness=0.1, sigma=1)
	
	##test
	#(R,G,B) = cv2.split(retrieveRGB(yuv))
	##
	#cv2.imshow('testing',cv2.merge([B,G,R]))
	#cv2.imshow('segments',255*segments)
	#cv2.waitKey(0)	

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
        	#subsquares[k] = np.fft.fft2(subsquares[k].reshape(SQUARE_SIZE, SQUARE_SIZE)).reshape(SQUARE_SIZE * SQUARE_SIZE)
		subsquares[k] = subsquares[k].reshape(SQUARE_SIZE * SQUARE_SIZE)

	return subsquares, Y,U,V

def predict_image(svr_Y, svr_U, svr_V, path,pca):
    	img = img_as_float(imread(path))
	gray_image, yuv , segments = segment_image(path)
	n_segments = segments.max() + 1
	subsquares, Y,U,V = extract_features(path)
	
	subsquares_pca = pca.transform(subsquares)
	
	Y_image = img_as_float(gray_image)#np.zeros_like(gray_image)
	#Y_image = np.ones_like(gray_image)	
	U_image = np.zeros_like(gray_image)
	V_image = np.zeros_like(gray_image)
	#prediction of LAB
	predicted_Y = np.zeros(n_segments)
    	predicted_U = np.zeros(n_segments)
    	predicted_V = np.zeros(n_segments)	
	for k in range(n_segments):
		#predicted_Y[k] = svr_Y.predict(subsquares[k])

		predicted_U[k] = clampU(svr_U.predict(subsquares_pca[k])*2)
		#predicted_U[k] = svr_U.predict(subsquares[k])
		predicted_V[k] = clampU(svr_V.predict(subsquares_pca[k])*2)
		#predicted_V[k] = svr_V.predict(subsquares[k])
	# Apply MRF to smooth out colorings
    	predicted_U, predicted_V = apply_mrf(predicted_U, predicted_V, segments, n_segments, img, subsquares_pca)
	
	#image reconstruction	
	for (i,j), value in np.ndenumerate(segments):
		#Y_image[i,j] = predicted_Y[value]
		yuv[i][j][0] = Y_image[i][j]#predicted_Y[value]
		yuv[i][j][1] = predicted_U[value]
		yuv[i][j][2] = predicted_V[value]
    	predicted_image = retrieveRGB(yuv)
	#Y_image = gray_image	
	#merged = cv2.merge([Y_image,U_image,V_image])
	
	print 'Green channel'
	print yuv[:,:,1]
	print 'Blue channel'
	print yuv[:,:,2]
	
	#predicted_image = cv2.cvtColor(merged,cv2.COLOR_YUV2BGR)
	print "predicted"
	#cv2.imshow("pred", gray_image)

	return predicted_image




def generate_adjacencies(segments, n_segments, img, subsquares):
    adjacency_list = []
    for i in range(n_segments):
        adjacency_list.append(set())
    for (i,j), value in np.ndenumerate(segments):
        # Check vertical adjacency
        if i < img.shape[0] - 1:
            newValue = segments[i + 1][j]
            if value != newValue and np.linalg.norm(subsquares[value] - subsquares[newValue]) < THRESHOLD:
                adjacency_list[value].add(newValue)
                adjacency_list[newValue].add(value)

        # Check horizontal adjacency
        if j < img.shape[1] - 1:
            newValue = segments[i][j + 1]
            if value != newValue and np.linalg.norm(subsquares[value] - subsquares[newValue]) < THRESHOLD:
                adjacency_list[value].add(newValue)
                adjacency_list[newValue].add(value)

    return adjacency_list

# Given the prior observed_u and observed_v, which are generated using the SVR,
# represent the system as a Markov Random Field and optimize over it using
# Iterated Conditional Modes. Return the prediction of the hidden U and V values
# of the segments.
# For now, we assume that the U and V channels behave independently.
def apply_mrf(observed_u, observed_v, segments, n_segments, img, subsquares):
    hidden_u = np.copy(observed_u)  # Initialize hidden U and V to the observed
    hidden_v = np.copy(observed_v)

    adjacency_list = generate_adjacencies(segments, n_segments, img, subsquares)

    for iteration in range(ICM_ITERATIONS):
        new_u = np.zeros(n_segments)
        new_v = np.zeros(n_segments)

        for k in range(n_segments):

            u_potential = 100000
            v_potential = 100000
            u_min = -1
            v_min = -1

            # Compute conditional probability over all possibilities of U
            for u in np.arange(-U_MAX, U_MAX, .001):
                u_computed = (u - observed_u[k]) ** 2 / (2 * COVAR)
                for adjacency in adjacency_list[k]:
                    u_computed += WEIGHT_DIFF * ((u - hidden_u[adjacency]) ** 2)
                if u_computed < u_potential:
                    u_potential = u_computed
                    u_min = u
            new_u[k] = u_min

            # Compute conditional probability over all possibilities of V
            for v in np.arange(-V_MAX, V_MAX, .001):
                v_computed = (v - observed_v[k]) ** 2 / (2 * COVAR)
                for adjacency in adjacency_list[k]:
                    v_computed += WEIGHT_DIFF * ((v - hidden_v[adjacency]) ** 2)
                if v_computed < v_potential:
                    v_potential = v_computed
                    v_min = v
            new_v[k] = v_min

        u_diff = np.linalg.norm(hidden_u - new_u)
        v_diff = np.linalg.norm(hidden_v - new_v)
        hidden_u = new_u
        hidden_v = new_v
        if u_diff < ITER_EPSILON and v_diff < ITER_EPSILON:
            break

    return hidden_u, hidden_v


