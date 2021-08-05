#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing Libraries
import cv2
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import numpy as np
import os
import matplotlib.pyplot as plt
from joblib import dump


# In[ ]:


X_test = []
y_test = []
X_train = []
y_train = []

# dir_train is the path to folder containing the training images 
dir_train = "/content/drive/MyDrive/IITD/data_final/train/"

for dir in os.listdir(dir_train) :
    cm = 0
    for file in os.listdir(dir_train + str(dir)) :
      if cm < 868 :
        img = cv2.resize(cv2.imread("{}{}/{}".format(dir_train,dir,file),0),(32,32))
        img = np.reshape(img,1024)
        img = img / 255.0
        img.astype(np.float32)
        X_train.append(img)
        y_train.append(int(dir))
        cm += 1

# dir_test is the path to folder containing the testing images 
dir_test = "/content/drive/MyDrive/IITD/data_final/eval/"

for dir in os.listdir(dir_test) :
    cm = 0
    for file in os.listdir("{}{}".format(dir_test,dir)) :
      if cm < 868 :
        img = cv2.resize(cv2.imread("{}{}/{}".format(dir_test,dir,file),0),(32,32))
        img = np.reshape(img,1024)
        img = img / 255.0
        img.astype(np.float32)
        X_test.append(img)
        y_test.append(int(dir))
        cm += 1
        
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)


# In[ ]:


# Create a classifier: a support vector classifier
clf = svm.SVC(C=10, gamma=0.005, kernel="rbf")


# In[ ]:


# Training the SVM model on training data
clf.fit(X_train, y_train)


# In[ ]:


# Evaluating the model on testing data
predicted = clf.predict(X_test)


# In[ ]:


# Checking the evaluation statistics of our model on testing data
print("accuracy", metrics.accuracy_score(y_test, predicted), "\n") # Accuracy

scores=metrics.classification_report(y_test, predicted, labels=[10,11,12,13]) # F1 score and overall analysis
print(scores)


# In[ ]:


# Saving the model
dump(clf,"model.joblib")
