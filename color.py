import cv2
import numpy as np
from matplotlib import pyplot as plt
 
def nothing(x):
    pass
 
cap = cv2.VideoCapture(0) #capture camera img
cv2.namedWindow("Trackbars")
 
cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing) #fifth paramter not required
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)
l_h = cv2.getTrackbarPos("L - H", "Trackbars") #lower bound
l_s = cv2.getTrackbarPos("L - S", "Trackbars")
l_v = cv2.getTrackbarPos("L - V", "Trackbars")
u_h = cv2.getTrackbarPos("U - H", "Trackbars") #upper bound
u_s = cv2.getTrackbarPos("U - S", "Trackbars")
u_v = cv2.getTrackbarPos("U - V", "Trackbars")
 
 
while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #convert from RGB to HSV type:uint8***----------- convolution
 
    l_h = cv2.getTrackbarPos("L - H", "Trackbars") #lower bound
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars") #upper bound
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")
 
    lower_color = np.array([l_h, l_s, l_v])
    upper_color = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hsv, lower_color, upper_color)
    #desired color will be shown in mask frame in white color,others black
 
    result = cv2.bitwise_and(frame, frame, mask=mask)
    #by bitwise_and function only white part in the mask will be shown in result frame


    
    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)
    cv2.imshow("result", result)


    #color = ('b','g','r')
    #for i,col in enumerate(color):
    # histr = cv2.calcHist([frame],[i],None,[256],[0,256])
    # plt.plot(histr,color = col)
    # plt.xlim([0,256])
    #plt.show()

    #print(lower_color[0])
    #print(upper_color[0])
    
    #if(lower_color[0]<=hsv[0][0] and hsv[0][0]<=upper_color[0]):
    #   print('orange')
    key = cv2.waitKey(1)
    if key == 27: #27=ESC
        break
 
cap.release()
cv2.destroyAllWindows()
