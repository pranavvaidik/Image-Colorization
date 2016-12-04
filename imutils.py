import cv2
import numpy as np



# this method takes the path to the folder as input and loads all the files in the folder into the dataset
def load_data():
	path = "CUB_200_2011/CUB_200_2011/images/"
	image_locs = np.genfromtxt("images.txt",dtype = None)
	#imagePath = path + image_locs[1][1]
	for i in range(len(image_locs)):
		imagePath = path + image_locs[i][1]
		#print imagePath		
		image = cv2.imread(imagePath)
		#print image
		cv2.imshow("image",image)
		cv2.waitKey(100)
	return imagePath #will be returning the training and testing sets after this


load_data()

	
