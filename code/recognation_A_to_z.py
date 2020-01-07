import cv2
import  numpy as np
import math
from gtts import gTTS
import os

frame=cv2.imread('C:\\Users\\Prateek\\Desktop\\hand\\A.jpg')
#frame=cv2.imread('P:\\Python\\DSP\\data\\frame130.jpg')
#hand=cv2.imread('P:\\Python\\DSP\data\\frame25.jpg',0)
#dim=frame.shape
#print(dim)
#edged=cv2.Canny(frame,100,200)
#cv2.imshow('Canny',edged)
#ret,the=cv2.threshold(edged,70,255,cv2.THRESH_BINARY)
#_,contours,_=cv2.findContours(the.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#crop_image = frame[100:300, 100:300]

    # Apply Gaussian blur
blur = cv2.GaussianBlur(frame, (3, 3), 0)

    # Change color-space from BGR -> HSV
hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Create a binary image with where white will be skin colors and rest is black
mask2 = cv2.inRange(hsv, np.array([2, 0, 0]), np.array([20, 255, 255]))

    # Kernel for morphological transformation
kernel = np.ones((5, 5))

    # Apply morphological transformations to filter out the background noise
dilation = cv2.dilate(mask2, kernel, iterations=1)
erosion = cv2.erode(dilation, kernel, iterations=1)

    # Apply Gaussian Blur and Threshold
filtered = cv2.GaussianBlur(erosion, (3, 3), 0)
ret, thresh = cv2.threshold(filtered, 100, 255, 0)

    # Show threshold image
cv2.imshow("Thresholded", thresh)

    # Find contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



contour = max(contours, key=lambda x: cv2.contourArea(x))

        # Create bounding rectangle around the contour
#x, y, w, h = cv2.boundingRect(contour)
#cv2.rectangle(crop_image, (x, y), (x + w, y + h), (0, 0, 255), 0)

        # Find convex hull
hull = cv2.convexHull(contour)
#final=cv2.drawContours(edged,hull,-1,(255,255,255),3)
#print(final)
        

        # Draw contour
drawing = np.zeros(frame.shape, np.uint8)
cv2.drawContours(drawing, [contour], -1, (0, 255, 0), 0)
cv2.drawContours(drawing, [hull], -1, (0, 0, 255), 0)

        # Find convexity defects
hull = cv2.convexHull(contour, returnPoints=False)
#cv2.imshow('hull',hull)
defects = cv2.convexityDefects(contour, hull)
print(len(defects))
value=len(defects)
print('Hand gesture:')
if(value==17):
    mytext='A'
    language='en'
    myobj = gTTS(text=mytext, lang=language, slow=False)
    print('A')
elif(value==13):
    mytext='B'
    language='en'
    myobj = gTTS(text=mytext, lang=language, slow=False)
    print('B')
elif(value==25):
    mytext='C'
    language='en'
    myobj = gTTS(text=mytext, lang=language, slow=False)
    print('C')
elif(value==16):
    mytext='D'
    language='en'
    myobj = gTTS(text=mytext, lang=language, slow=False)
elif(value==15):
    mytext='F'
    language='en'
    myobj = gTTS(text=mytext, lang=language, slow=False)
elif(value==20):
    mytext='G'
    language='en'
    myobj = gTTS(text=mytext, lang=language, slow=False)
elif(value==27):
    mytext='H'
    language='en'
    myobj = gTTS(text=mytext, lang=language, slow=False)
elif(value==14):
    mytext='I'
    language='en'
    myobj = gTTS(text=mytext, lang=language, slow=False)
elif(value==11):
    mytext='J'
    language='en'
    myobj = gTTS(text=mytext, lang=language, slow=False)
elif(value==10):
    mytext='K'
    language='en'
    myobj = gTTS(text=mytext, lang=language, slow=False)
elif(value==9):
    mytext='L'
    language='en'
    myobj = gTTS(text=mytext, lang=language, slow=False)
elif(value==18):
    mytext='M'
    language='en'
    myobj = gTTS(text=mytext, lang=language, slow=False)
elif(value==21):
    mytext='N'
    language='en'
    myobj = gTTS(text=mytext, lang=language, slow=False)
elif(value==19):
    mytext='O'
    language='en'
    myobj = gTTS(text=mytext, lang=language, slow=False)
elif(value==12):
    mytext='P'
    language='en'
    myobj = gTTS(text=mytext, lang=language, slow=False)
elif(value==23):
    mytext='Q'
    language='en'
    myobj = gTTS(text=mytext, lang=language, slow=False)
elif(value==29):
    mytext='R'
    language='en'
    myobj = gTTS(text=mytext, lang=language, slow=False)
elif(value==26):
    mytext='S'
    language='en'
    myobj = gTTS(text=mytext, lang=language, slow=False)
elif(value==24):
    mytext='T'
    language='en'
    myobj = gTTS(text=mytext, lang=language, slow=False)



myobj.save("welcome.mp3")
os.system("mpg321 welcome.mp3")


        # Use cosine rule to find angle of the far point from the start and end point i.e. the convex points (the finger
        # tips) for all defects
count_defects = 0

for i in range(defects.shape[0]):
    s, e, f, d = defects[i, 0]
    start = tuple(contour[s][0])
    end = tuple(contour[e][0])
    far = tuple(contour[f][0])

    a= math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
    b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
    c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
    angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14
    count=0

            # if angle > 90 draw a circle at the far point
    if angle <= 90:
            count_defects += 1
            cv2.circle(frame, far, 1, [0, 0, 255], -1)
            print(count_defects)
    if(count_defects==1):
        count+=1         

    cv2.line(frame, start, end, [0, 255, 0], 2)
    print(count_defects)
    
    #print(count)
print('count')
print(count)    
print('output')
count_edge=count_defects+1
print(count_edge)
# Print number of fingers
cv2.imshow("Gesture", frame)
all_image = np.hstack((drawing, frame))
cv2.imshow('Contours', all_image) 
#final=cv2.drawContours(edged,hull,-1,(255,255,255),3)
#print(final)
print("cbh")
cv2.imshow('Original Image',frame)
#cv2.imshow('Threshold',the)
#cv2.waitKey(0)
#cv2.imshow('Convex Hull',final)
cv2.waitKey(0)
cv2.destroyAllWindows()
