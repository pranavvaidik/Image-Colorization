import cv2
from imutils import *
from constants import *
from sklearn.svm import SVR


path = "TrainingData/1.jpg"
#cv2.figure(2)
img = cv2.imread(path)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cv2.imshow("gray",gray)
cv2.waitKey(0)
features, train_L, train_a, train_b = load_training_data()


L_svr = SVR(C=C, epsilon=SVR_EPSILON)
a_svr = SVR(C=C, epsilon=SVR_EPSILON)
b_svr = SVR(C=C, epsilon=SVR_EPSILON)


L_svr.fit(features,train_L)
print "L trained"
a_svr.fit(features,train_L)
print "a trained"
b_svr.fit(features,train_L)
print "b trained"





predicted_image = predict_image(L_svr, a_svr, b_svr, path)

cv2.imshow("predicted colors", predicted_image*255)
cv2.waitKey(0)



