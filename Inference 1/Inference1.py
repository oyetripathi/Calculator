#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import cv2
import keras
import os
import sys


# In[1]:


model = keras.models.load_model('label_predictor.h5')


# In[10]:


def predict (path):
    img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    img = 255 - cv2.resize(img,(96,32))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (img[i][j]>50):
                img[i][j]=255
            else:
                img[i][j]=0
    img = img.astype('float32')/255
    A = np.argmax(model.predict(img.reshape((1,32,96,1))))
    if (A==0):
        return 'prefix'
    elif (A==1):
        return 'infix'
    else:
        return 'postfix'


# In[12]:


label_arr = []
name_arr = []
path = sys.argv[1]
if (path[-1]!='/'):
    path += '/'
file = os.listdir(path)


# In[13]:


for dir1 in file:
    img_path = os.path.join(path,dir1)
    A = predict(img_path)
    label_arr.append(A)
    name_arr.append(dir1)


# In[15]:


out = pd.DataFrame(name_arr,columns=['Image_Name'])
out['Label'] = label_arr
out.to_csv('Team_VNV_1.csv',index=False)


# In[ ]:



