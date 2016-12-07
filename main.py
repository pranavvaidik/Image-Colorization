import cv2
from imutils import *
from constants import *
from sklearn.svm import SVR


path = "TrainingData/1.jpg"
#cv2.figure(2)
img = cv2.imread(path)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cv2.imshow("gray",gray)

features, train_Y, train_U, train_V = load_training_data()


Y_svr = SVR(C=C, epsilon=SVR_EPSILON)
U_svr = SVR(C=C, epsilon=SVR_EPSILON)
V_svr = SVR(C=C, epsilon=SVR_EPSILON)


Y_svr.fit(features,train_Y)
print "L trained"
U_svr.fit(features,train_U)
print "a trained"
V_svr.fit(features,train_V)
print "b trained"





predicted_image = predict_image(Y_svr, U_svr, V_svr, path)


cv2.waitKey(0)



