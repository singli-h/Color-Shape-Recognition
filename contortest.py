import cv2   
import numpy as np
import time

#capturing video through camera
cap=cv2.VideoCapture(0)

#Color range value greatly depends on the camera and light envionment
#defining the Range of Red color
red_lower=np.array([0,40,140])
red_upper=np.array([6,228,238])

#defining the Range of Blue color
blue_lower=np.array([60,100,100])
blue_upper=np.array([110,255,255])

#defining the Range of Brown color
brown_lower=np.array([34,5,98])
brown_upper=np.array([114,80,238])

#defining the Range of yellow color
yellow_lower=np.array([67,24,57])
yellow_upper=np.array([87,154,238])

line1 = cv2.imread('line1.png',0)
line2 = cv2.imread('line2.png',0)
river = cv2.imread('river.png',0)

printed=False
printed2=False
printed3=False
status="Start"
status2="Start"
newStatus="Start"
newStatus2="Start"



while True:
    #capture current frame from live camera video
    _, frame = cap.read()

    #blur the image since we dont need precious and small image
    #detection but a whole view of the game field characteristic
    #this can reduce scattered contour
    frame=cv2.GaussianBlur(frame,(15,15),0)
    
    #converting frame(img i.e BGR) to HSV (hue-saturation-value)
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    brown_b=False
    blue_b=False
    yellow_b=False
    red_b=False
    

        
    #status print for image matching
    if newStatus!=status:
        status=newStatus
        printed=False
        
    if printed!=True:
        print(status)
        printed=True
        
    #status print for color line detection
    if newStatus2!=status2:
        status2=newStatus2
        printed2=False
        
    if printed2!=True:
        print(status2)
        printed2=True

    #check combined result
    if status!=status2 and printed2 and printed:
        printed3=False
        
    if status==status2 and printed3==False:
        print("real:" + status)
        printed3=True
        
    img_rgb = frame
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    #match for river
    res = cv2.matchTemplate(img_gray,river,cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    threshold = 0.5
    if max_val> threshold:
        newStatus="river"
        loc = np.where( res >= threshold)
        
        # draw rectangle to locate the recognised images
        w, h = river.shape[::-1]
        for pt in zip(*loc[::-1]):
            cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    
    #match for line1
    res = cv2.matchTemplate(img_gray,line1,cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    threshold = 0.5
    if max_val> threshold:
        newStatus="line1"
        loc = np.where( res >= threshold)
        
        # draw rectangle to locate the recognised images
        w, h = river.shape[::-1]
        for pt in zip(*loc[::-1]):
            cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

    #match for line2
    res = cv2.matchTemplate(img_gray,line2,cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    threshold = 0.5
    if max_val> threshold:
        newStatus="line2"
        loc = np.where( res >= threshold)
        
        # draw rectangle to locate the recognised images
        w, h = line2.shape[::-1]
        for pt in zip(*loc[::-1]):
            cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
        
    #create mask where color in the range of red,blue and yellow in the image
    red=cv2.inRange(hsv, red_lower, red_upper)
    blue=cv2.inRange(hsv,blue_lower,blue_upper)
    brown=cv2.inRange(hsv,brown_lower,brown_upper)
    yellow=cv2.inRange(hsv,yellow_lower,yellow_upper)

    #Tracking Blue Color and line
    blue_mask=cv2.bitwise_and(frame, frame, mask=blue)
    gray=cv2.cvtColor(blue_mask,cv2.COLOR_BGR2GRAY)
    edges=cv2.Canny(gray,50,150)
    lines=cv2.HoughLinesP(edges,1,np.pi/180,50,minLineLength=100,maxLineGap=200)

    contours,_=cv2.findContours(blue,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if(area>2000):
            blue_b=True
            if lines is not None:
                #print("blue line")
                for line in lines:
                    x1,y1,x2,y2=line[0]
                    cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),5)
            #print("Blue")
                    
    #Tracking Brown Color
    brown_mask=cv2.bitwise_and(frame, frame, mask=brown)
    gray=cv2.cvtColor(brown_mask,cv2.COLOR_BGR2GRAY)
    edges=cv2.Canny(gray,50,150)
    lines=cv2.HoughLinesP(edges,1,np.pi/180,50,minLineLength=100,maxLineGap=200)

    contours,_=cv2.findContours(brown,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if(area>2000):
            brown_b=True
            #print("Brown")          

    #Tracking Red Color and line
    red_mask=cv2.bitwise_and(frame, frame, mask=red)
    gray=cv2.cvtColor(red_mask,cv2.COLOR_BGR2GRAY)
    edges=cv2.Canny(gray,150,350)
    lines=cv2.HoughLinesP(edges,1,np.pi/180,50,maxLineGap=50)
    
    contours,_=cv2.findContours(red,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if(area>3000): 
            if lines is not None:
                red_b=True
                #print("red line")
                for line in lines:
                    x1,y1,x2,y2=line[0]
                    cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),5)
                
            #print("Red")
            
    #Tracking Yellow Color and line
    yellow_mask=cv2.bitwise_and(frame, frame, mask=yellow)
    gray=cv2.cvtColor(yellow_mask,cv2.COLOR_BGR2GRAY)
    edges=cv2.Canny(gray,50,150)
    lines=cv2.HoughLinesP(edges,1,np.pi/180,50,maxLineGap=50)
    
    contours,_=cv2.findContours(yellow,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if(area>2000):
            yellow_b=True
            #print("Yellow")  

    #check color status
    if blue_b and brown_b :
        newStatus2="river"
    if yellow_b and red_b :
        newStatus2="line2"
            
    cv2.imshow("Color Tracking",frame)
    #cv2.imshow("yellow",red_mask)
    #cv2.imshow("Brown",brown_mask)
    #time.sleep(0.05)
    
    #break while loop and close program if 'esc' clicked
    if cv2.waitKey(1) == 27:
        break      
cap.release()
cv2.destroyAllWindows()
