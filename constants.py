import numpy as np

#Defining Constants reading images
PATH = "CUB_200_2011/CUB_200_2011/images/"


#Definfing constants for segmentation
N_SEGMENTS = 250
SQUARE_SIZE = 10

#Constants for training the SVR
C = 0.125
SVR_EPSILON = 0.0625


# Constants for running ICM on the MRF
ICM_ITERATIONS = 10
ITER_EPSILON = .01
COVAR = 0.25       # Covariance of predicted chrominance from SVR and actual covariance
WEIGHT_DIFF = 2    # Relative importance of neighboring superpixels
THRESHOLD = 25     # Threshold for comparing adjacent superpixels.
                   # Setting a higher threshold reduces error, but causes the image to appear more uniform.


U_MAX = 0.436
V_MAX = 0.615



RGB_FROM_YUV = np.array([[1, 0, 1.13983],
                         [1, -0.39465, -.58060],
                         [1, 2.03211, 0]]).T
YUV_FROM_RGB = np.array([[0.299, 0.587, 0.114],
                         [-0.14713, -0.28886, 0.436],
                         [0.615, -0.51499, -0.10001]]).T
