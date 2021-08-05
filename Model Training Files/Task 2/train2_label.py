#!/usr/bin/env python
# coding: utf-8

# In[263]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout


# In[264]:


#Loading images from provided dataset 
path = 'SoML/SoML-50/data/'   
img_arr = []
size = 1000 #number of samples to load from dataset
files = os.listdir(path)
files.sort(key=lambda x:int(x[:-4]))


# In[265]:


count = 0

for dir1 in files:
    if (count < size):
        img_path = os.path.join(path,dir1)
        img = 255 - cv2.resize(cv2.imread(img_path,cv2.IMREAD_GRAYSCALE),(96,32))
        
        #applying binarization using threshold value = 50 
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if (img[i][j]>50):
                    img[i][j]=255
                else:
                    img[i][j]=0
            
        img = img.astype('float32')/255
        img_arr.append(img)
        count += 1
    else:
        break
    
img_arr = np.array(img_arr)


# In[266]:


#reading the annotations file
annot = pd.read_csv('SoML/SoML-50/annotations.csv',nrows=size)['Label']
annot = np.array(annot).reshape(-1,1)


# In[267]:


#One Hot Encoding of annotation
temp = []
for i in annot:
    if (i == 'infix'):
        temp.append([0,1,0])
    elif (i == 'prefix'):
        temp.append([1,0,0])
    else:
        temp.append([0,0,1])

annot = np.array(temp)


# In[268]:


x = img_arr.reshape(size,32,96,1)


# In[269]:


y = annot


# In[274]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1)


# In[275]:


#model description

model = Sequential()
model.add(Conv2D(100,(8,8),input_shape=(32,96,1),activation='relu'))
model.add(Conv2D(64,(4,4),activation='relu'))
model.add(MaxPooling2D((8,8)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(3,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[276]:


model.fit(x=x_train,y=y_train,epochs=5)


# In[278]:


model.evaluate(x_test,y_test)


# In[371]:


model.save('label_predictor.h5')
