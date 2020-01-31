
# coding: utf-8
"""
Created on Mon Jan 13 19:50:53 2020

@author: K S RAKSHIT prateek pannaga rahul
FILE NAME:Image_capturing.py
"""

# In[11]:



import cv2    #import numpy and cv2
import numpy as np

vc = cv2.VideoCapture(0)
pic_no = 1       #the image number will start with 1 till 2000 image
total_pic =2000
flag_capturing = False
path = 'D:/closingdilation/_'   #for creating database for _ mention _ if you are creating for A write location/A'
while(vc.isOpened()):
    # read image
    rval, frame = vc.read()  #read the frame from the webcam
    frame = cv2.flip(frame, 1)   #flip the image horizontally
    
    # get hand data from the rectangle sub window on the screen
    cv2.rectangle(frame, (300,300), (100,100), (0,255,0),0) #draw the ROI(REGION OF INTEREST)
    
    cv2.imshow("image", frame)#SHOW THE IMAGE
    
    crop_img = frame[100:300, 100:300]  #CREATE THE BOX 
    
    if flag_capturing:
        
        pic_no += 1   #START CAPTURING
        grey1 = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)  #CONVERT ROI TO GRAY
        thresh1 = cv2.threshold(grey1, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1] #APPLY THE THRESHOLD
        cv2.imshow('grey',thresh1)
        
        save_img = cv2.resize( crop_img, (50,50) )  #RESIZE THE IMAGE T0 50 CROSS 50
        save_img = np.array(save_img)
        
        grey = cv2.cvtColor(save_img, cv2.COLOR_BGR2GRAY)  #CONVERT TO GREY
        thresh = cv2.threshold(grey, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]#APPLY THE THRESHOLD
        kernel = np.ones((5,5),np.uint8)    #ADD THE KERNEL
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)#USE DILATION FOR FILLING UP THE GLARE
        #cv2.imshow('grey',thresh)
        save_img = cv2.resize(closing, (50,50))   
        
        
        cv2.imwrite(path + "/" + str(pic_no) + ".jpg", save_img)#SAVE THE IMAGE
        print(path + "/" + str(pic_no) + ".jpg")
        
    
    keypress = cv2.waitKey(1)  #wait for the waitkey
    
    if pic_no == total_pic:   #if the image number is 2000(2000 images captured break)
        flag_capturing = False
        break
    
    if keypress == ord('q'):  #exit if q is pressed
        break
    elif keypress == ord('c'):  #start capturing only when c is pressed
        flag_capturing = True

vc.release()  #relase the webscam
cv2.destroyAllWindows()#destroy all the window
cv2.waitKey(1)

