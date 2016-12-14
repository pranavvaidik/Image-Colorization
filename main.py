import cv2
from imutils import *
from constants import *
from sklearn.svm import SVR
from sklearn.decomposition import PCA

#path = "Indigo_Bunting_0018_11883.jpg"
path = "TrainingData/214000.jpg"
#cv2.figure(2)
img = cv2.imread(path)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#cv2.imshow("gray",gray)

features, train_Y, train_U, train_V = load_training_data()

pca = PCA(n_components = 30)
pca.fit(features)

Y_svr = SVR(C=C, epsilon=SVR_EPSILON)
U_svr = SVR(C=C, epsilon=SVR_EPSILON)
V_svr = SVR(C=C, epsilon=SVR_EPSILON)

features_pca = pca.transform(features)

Y_svr.fit(features_pca,train_Y)
print "Y trained"
U_svr.fit(features_pca,train_U)
print "U trained"
V_svr.fit(features_pca,train_V)
print "V trained"


test_image_locs = np.genfromtxt("test_images.txt",dtype = None)
for i in range(len(test_image_locs)):
	#path = PATH+test_image_locs[i][1]
	path = test_image_locs[i][1]
	predicted_image = predict_image(Y_svr, U_svr, V_svr, path, pca)
	(R,G,B) = cv2.split(predicted_image)
	I = cv2.merge([B,G,R])
	cv2.imwrite("results/"+`i`+".jpg",I*255)
	cv2.imshow("img",I)
	
cv2.waitKey(0)





