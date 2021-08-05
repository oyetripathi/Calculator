#!/usr/bin/env python
# coding: utf-8

# Importing Libraries that we require Here we have imported a few basic libraries

# In[38]:


import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import pandas as pd


# Mounting our drive. Since our training dataset was stored in drive

# In[39]:


from google.colab import drive
drive.mount('/content/gdrive')


# Further steps are for loading images

# In[40]:


import os
os.environ['CONFIG_DIR'] = "/content/gdrive/My Drive"
# /content/gdrive/My Drive/Kaggle is the path where kaggle.json is present in the Google Drive


# In[41]:


#changing the working directory
get_ipython().run_line_magic('cd', '/content/gdrive/My Drive')
#Check the present working directory using pwd command


# In[42]:


from PIL import Image


# In[43]:



import numpy as np
 

img = Image.open("/content/gdrive/MyDrive/train_numbers/0/10014.jpg", 'r').convert('RGB')
 
 #image_arr = np.array (img) # is converted into an array numpy
img = np.asarray(img, 'float32')


# In[44]:


img = img/255


# In[45]:


plt.figure(figsize=(10,10))
plt.imshow(img)
plt.show()


# Compiling our dataset (will take 19 mins if the code is run)

# In[46]:


import numpy as np
from glob import glob 
import cv2
class_names = ['0', '1', '2', '3','4','5','6','7','8','9']
training_data = []
training_label = []

 # Img = Image.open ( "1.jpg") # open the image of PIL Image.open
TRAIN_DATA_DIR = '/content/gdrive/MyDrive/train_numbers/*/*.jpg'
data = glob(TRAIN_DATA_DIR)
# TEST_DATA_DIR = '/content/gdrive/MyDrive/eval_numbers/*/*.jpg'
# data_test = glob(TEST_DATA_DIR)
# AUG_DATA_DIR = '/content/gdrive/MyDrive/augmented/*/*.jpg'
# data_aug = glob(AUG_DATA_DIR)

data_count= len(data)
for d in data:
    training_data.append(cv2.resize(cv2.imread(d),(32,32)))
    training_label.append(d.split('/')[-2])
# data_count_test= len(data_test)
# for d in data_test:
#     training_data.append(cv2.resize(cv2.imread(d),(32,32)))
#     training_label.append(d.split('/')[-2])
# for d in data_aug:
#     training_data.append(cv2.resize(cv2.imread(d),(32,32)))
#     training_label.append(d.split('/')[-2])
print("Data Read Complete")
training_data = np.asarray(training_data)
training_label = pd.DataFrame(training_label)


# Thus we have our  dataset called 'training_data'

# In[47]:


training_data = np.asarray(training_data, 'float32')
training_data = training_data/255
training_data.shape


# Plot and visualize the images

# In[48]:


plt.imshow(training_data[1])


# In[49]:


plt.figure(figsize=(10,10))
for i in range(3):
    plt.imshow(training_data[i])
    #plt.xlabel(class_names[training_label[i][0]])
    plt.show()


# Spliting our data into text and train 

# In[50]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(training_data, training_label, test_size=0.1)


# In[51]:


import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import BatchNormalization


# encoding y_test and y_train

# In[52]:


y_test_coded = pd.get_dummies(y_test)
y_train_coded = pd.get_dummies(y_train)
print(y_train.shape)
print(X_train.shape)


# Taking in image from contest dataset to check whether its detecting

# In[80]:


import matplotlib.pyplot as plt
im = cv2.resize(cv2.imread('/content/1.33.jpg'),(32,32))
plt.imshow(im)
im.shape


# In[81]:


im_test = np.asarray(im, 'float32')
im_test = im_test/255
im_test = im_test.reshape(1,32,32,3)
im_test.shape


# Creating our model

# In[55]:


classifier = Sequential()
classifier.add(Conv2D(64,(3,3),activation='relu',input_shape=(32,32,3)))
classifier.add(Conv2D(32,(1,1),activation='relu'))
classifier.add(MaxPooling2D((2,2)))
classifier.add(BatchNormalization(axis=3))
classifier.add(Conv2D(32,(3,3),activation='relu'))
classifier.add(Conv2D(16,(1,1),activation='relu'))
classifier.add(MaxPooling2D((2,2)))
classifier.add(BatchNormalization(axis=3))
classifier.add(BatchNormalization(axis=3))
classifier.add(MaxPooling2D((2,2)))
classifier.add(Conv2D(16,(1,1),activation='relu'))
classifier.add(BatchNormalization(axis=3))
classifier.add(Flatten())
classifier.add(Dense(128,activation='relu'))
classifier.add(Dense(10,activation='softmax'))


# In[57]:


classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = classifier.fit(X_train, y_train_coded, validation_data= (X_test, y_test_coded) ,epochs=15)


# Using it to predict our data

# In[82]:


pred_Y = classifier.predict(im_test)
predYC = np.argmax(pred_Y, axis = 1) 


# In[83]:


pred_Y


# Thus we get our prediction

# In[84]:


predYC


# Saving the prediction

# In[61]:


classifier.save('digit_classification',save_format='h5')
