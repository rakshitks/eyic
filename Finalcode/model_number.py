# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 19:50:53 2020

@author: K S RAKSHIT Prateek Pannaga Rahul
File name:model_number.py
global variable:None
"""

import numpy as np
from keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D
from keras.models import Sequential, save_model
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle       #import required modules
import os
import cv2
path='D:/closingdilation_1/'
gestures = os.listdir(path)

dict_labels = {
    'A':1,        #dictionary is created
    'B':2,
    'C':3,
    'D':4,
    'E':5,
    'F':6,
    'G':7,
    'H':8,
    'I':9,
    'J':10,
    'K':11,
    'L':12,
    'M':13,
    'N':14,
    'O':15,
    'P':16,
    'Q':17,
    'R':18,
    'S':19,
    'T':20,
    'U':21,
    'V':22,
    'W':23,
    'X':24,
    'Y':25,
    'Z':26,
    'nothing':27,
    '_':28,
    
    
    
    
}

print(list(dict_labels.keys()))  #printing the dictionary

x, y = [], []
for ix in gestures:    #loop through all the folders of A,B...nothing,'_' etc
    images = os.listdir(path + ix)
    for cx in images:
        img_path = path + ix + '/' + cx
        img = cv2.imread(img_path,0)  #grayscale reading
        img = img.reshape((50,50,1))  #reshape it into 50 cross 50
        #print(img)
        #print("-------------------------------------------------------------")
        img = img/255.0  #normalise the image
        x.append(img)     #append it to the list
        y.append(dict_labels[ix])  #append the required labels such as 1,2,etc
        
       # img1 = cv2.imread(img_path,0)
        #img1=cv2.flip(img1, 1)         #flipping horizontally and then training images
        #img1 = img1.reshape((50,50,1))
       # img1 = img1/255.0
        #x.append(img1)
       # y.append(dict_labels[ix])
    print("done with ",ix)
      
X = np.array(x)  #convert it into array since deep learning does not acccept list convert x to list
Y = np.array(y)  #simiarly y list is converted into array
Y = np_utils.to_categorical(Y)
Y.shape

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize = (18,8))
sns.countplot(x=list(dict_labels.keys()))

Y.shape

categories = Y.shape[1]

X, Y = shuffle(X, Y, random_state=0)   #shuffle so that training will be good

X.shape

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)   #10% is for testing 

print(X_train.shape, X_test.shape)  #print shape of X_test
print(Y_train.shape, Y_test.shape) #print shape of Y_test

model = Sequential()  #sequential module
model.add(Conv2D(64, kernel_size=(3,3), activation = 'relu', input_shape=(50,50 ,1) ))  #add a convolution layer of 64 nodes with activation relu function and input 50 cross 50
model.add(MaxPooling2D(pool_size = (2, 2)))   #adding the 2*2 matrix filter

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu')) #add a convolution layer of 64 nodes with activation relu function no need to mention input shape
model.add(MaxPooling2D(pool_size = (2, 2)))  #adding 2*2 matrix filter

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))#same as earlier
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())  #flatten it so that u can use dense layer
model.add(Dense(128, activation = 'relu'))  #use dense layer of 128 nodes
model.add(Dropout(0.30))  #drop 30% of neurons for better accuracy
model.add(Dense(categories, activation = 'softmax'))  #now categories will be 28 and the activation is softmax which gives output within 0 to 1

model.summary()

model.compile(optimizer='Adam', metrics=['accuracy'], loss='categorical_crossentropy')#adam algorithm and cat
history = model.fit(X_train, Y_train, batch_size=15, epochs=25, validation_data=[X_test, Y_test])  #train the model 25 epochs(25 times) and fit the data

import matplotlib.pyplot as plt
#matplotlib inline

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title("Accuracy")
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train','test'])
plt.show()

model.save('C:/Users/K S RAKSHIT/Desktop/new_eyic/Sign-Language-Recognition-master/closingdilationunder_15.h5')#save th model

from keras.models import load_model

m = load_model('C:/Users/K S RAKSHIT/Desktop/new_eyic/Sign-Language-Recognition-master/closingdilationunder_15.h5')