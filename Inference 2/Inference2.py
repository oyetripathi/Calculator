#!/usr/bin/env python
# coding: utf-8

# In[135]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from joblib import load
import keras
import sys


# In[136]:

digit_model = keras.models.load_model('digit_classification.h5')


# In[137]:



symbol_model = load('model.joblib')



# In[138]:


label_model = keras.models.load_model('label_predictor.h5')


# In[139]:


def predict_label(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = 255 - cv2.resize(img,(96,32))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (img[i][j]>50):
                img[i][j]=255
            else:
                img[i][j]=0
    img = img.astype('float32')/255
    A = np.argmax(label_model.predict(img.reshape((1,32,96,1))))
    return A


# In[140]:


def predict_symbol(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(32,32))
    img = np.reshape(img,1024).astype(np.float32)/255.0
    A = symbol_model.predict([img])
    return A[0]


# In[141]:


def predict_digit(img):
    img = cv2.resize(img,(32,32))
    img = np.array(img).astype('float32')/255
    img = img.reshape((1,32,32,3))
    A = np.argmax(digit_model.predict(img))
    return A


# In[142]:


def result(img):
    wid = int(img.shape[1]/3)
    img1 = img[:,:1*wid]
    img2 = img[:,1*wid:2*wid]
    img3 = img[:,2*wid:]
    label = predict_label(img)
    if (label==0):
        operation = predict_symbol(img1)
        operand_1 = predict_digit(img2)
        operand_2 = predict_digit(img3)
    elif (label==1):
        operation = predict_symbol(img2)
        operand_1 = predict_digit(img1)
        operand_2 = predict_digit(img3)
    else:
        operation = predict_symbol(img3)
        operand_1 = predict_digit(img1)
        operand_2 = predict_digit(img2)
    if (operation==10):
        return operand_1 + operand_2
    elif(operation==11):
        return operand_1 - operand_2
    elif (operation==12):
        return operand_1 * operand_2
    elif (operation==13):
        if(operand_1 == 0 or operand_2 == 0):
            return 0
        else:
            return  int(operand_1/operand_2)


# In[150]:


path = sys.argv[1]
if (path[-1] != '/'):
    path = path + '/'
value_arr = []
name_arr = []
files = os.listdir(path)


# In[149]:


for dir1 in files:
    img_path = os.path.join(path,dir1)
    img = cv2.imread(img_path)
    A = result(img)
    value_arr.append(A)
    name_arr.append(dir1)


# In[151]:


out = pd.DataFrame(name_arr,columns=['Image_Name'])


# In[152]:


out['Value'] = value_arr


# In[153]:


out.to_csv('Team_VNV_2.csv',index=False)


# In[ ]:

