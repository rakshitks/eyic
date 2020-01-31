# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 19:55:27 2020

@author: K S RAKSHIT, Prateek Janaj, Pannaga S, Rahul Rajakumar
filename: gesture.py
Global variable: None 
"""


# coding: utf-8

# In[2]:


import cv2
import numpy as np
from keras.models import load_model
import time
import textwrap
import keras
import tensorflow as tf 
model = tf.keras.models.load_model('closingdilationunder_15.h5')


# In[3]:


#model = load_model('C:/Users/K S RAKSHIT/Desktop/new_eyic/Sign-Language-Recognition-master/model30.h5')


# In[4]:


gestures = {
   1:'A',
    2:'B',
    3:'C',
    4:'D',
    5:'E',
    6:'F',
    7:'G',
    8:'H',
    9:'I',
    10:'J',
    11:'K',
    12:'L',
    13:'M',
    14:'N',
    15:'O',
    16:'P',
    17:'Q',
    18:'R',
    19:'S',
    20:'T',
    21:'U',
    22:'V',
    23:'W',
    24:'X',
    25:'Y',
    26:'Z',
    27:'',
    28:'_',
   
}



# In[10]:

#function for prediction of hand gsture
def predict(gesture):
    img = cv2.resize(gesture, (50,50)) #resize the gesture to 50x50
    img = img.reshape(1,50,50,1) #
    img = img/255.0
    prd = model.predict(img)
    print(prd)
   # time.sleep(2)
    index = prd.argmax()
    print(index)
    #print("index" ,index)
    #print(gestures[index],"is")
    
    return gestures[index]
   


# In[15]:


vc = cv2.VideoCapture(0) #capture the gesture
rval, frame = vc.read() #read the video
old_text = ''
pred_text = ''
count_frames = 0 
total_str = ''
flag = False #intialize the flag  as zero
list1=[]


# In[16]:


while True:
    
    if frame is not None: 
        #j=0
        
        frame = cv2.flip(frame, 1) #flip the frames
        frame = cv2.resize( frame, (400,400) ) #resize the frame 
        
        cv2.rectangle(frame, (300,300), (100,100), (0,255,0), 2) #form a rectangle inside the frame
        
        crop_img = frame[100:300, 100:300] #consider the only the rectangle 
        grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)#convert to graay
        
        thresh = cv2.threshold(grey,210,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]#Apply threshold
        kernel = np.ones((3,3),np.uint8)#kernal of 3x3 matrix
        thresh= cv2.dilate(thresh,kernel,iterations = 1)#apply dilation to fill the small gaps
        #cv2.imshow('windows',thresh)
      
        
        blackboard = np.zeros(frame.shape, dtype=np.uint8)#create the blackboard of size frame
        cv2.putText(blackboard, "Predicted text - ", (30, 40), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0)) #put the predicted text
        if count_frames > 30 and pred_text != "": #consider the  30 frames each 
            total_str += pred_text #combine the predicted text and total string and store it in total_string
            count_frames = 0 #make count frames to 0
            #print('counting frame')            
        if flag == True:
            old_text = pred_text #if flag is true means the predicted text is same as old text
            list1=pred_text #put the predicted text in the list1
            pred_text = predict(thresh)  #then call the predict function for predicting the gesture, threshold image as parameter               
           # print('flag is true')        
            if old_text == pred_text: #if old_text is equal to predicted text
                count_frames += 1 #increase the count_frames by 1
                #print('old test')
            else:
                count_frames = 0 #else make count =0             
            font = cv2.FONT_HERSHEY_SIMPLEX #font type
            font_size = 1 
            font_thickness = 2
            wrapped_text = textwrap.wrap(total_str, width=35) #35 words in the each line
            x, y = 30, 80
            i = 0
            for line in wrapped_text:
                textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0] #get the line word to be printed
                gap = textsize[1] +10 #gap between the each line
               # print(textsize[0])
               # print(textsize[1])
                y = int((blackboard.shape[0] + textsize[1]) / 4)+ i * gap #y axis position to print the word       
                x=30 #x axis position to print the word
                #print(y,x)                           
                cv2.putText(blackboard, line, (x,y), cv2.FONT_HERSHEY_TRIPLEX,0.5,(255, 255, 127)) #put the text in the (x,y) position            i +=1
    
        res = np.hstack((frame, blackboard)) #to combine the two images horizontally
        
        cv2.imshow("image", res)
        cv2.imshow("hand", thresh)
        
    rval, frame = vc.read()
    keypress = cv2.waitKey(1)
    if keypress == ord('c'):
        flag = True
    if keypress == ord('q'):
        break

vc.release()
cv2.destroyAllWindows()
cv2.waitKey(1)


# In[17]:


vc.release()

