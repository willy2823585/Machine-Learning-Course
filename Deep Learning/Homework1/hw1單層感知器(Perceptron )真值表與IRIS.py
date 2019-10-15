# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 11:04:41 2019

@author: willywanghw
"""

#7108029211 王皓威
#hw1 單層感知器(Perceptron)真值表與iris
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout
from keras import backend as K
from tensorflow.python.client import device_lib
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
#GPU CODE
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config)) 
K.tensorflow_backend._get_available_gpus()
print(device_lib.list_local_devices())


iris_data = load_iris() # load the iris dataset

print('Example data: ')
print(iris_data.data[:5])
print('Example labels: ')
print(iris_data.target[:5])

x = iris_data.data
y_ = iris_data.target.reshape(-1, 1) # Convert data to a single column
# One Hot encode the class labels
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y_)
#print(y)
# Split the data for training and testing
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.20)
# Build the model

# Initialising the ANN
classifier = Sequential()
# Adding the Single Perceptron or Shallow network
classifier.add(Dense(10, input_shape=(4,), activation='relu', name='fc1'))
# Adding dropout to prevent overfitting
classifier.add(Dropout(p=0.1))
# Adding the output layer
classifier.add(Dense(3, activation='softmax', name='output'))
# Adam optimizer with learning rate of 0.001
optimizer = Adam(lr=0.001)

# criterion loss and optimizer 
classifier.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
# Fitting the ANN to the Training set
classifier.fit(train_x, train_y, batch_size=4, nb_epoch=200)

results = classifier.evaluate(test_x, test_y)
print('Final test set loss: {:4f}'.format(results[0]))
print('Final test set accuracy: {:4f}'.format(results[1]))

print("--------------------------------------------------")

#keras use in 真值表  單層感知器
class SLPTruthtableBuild(object):
    def __init__(self):
        self.truthtablemodel  =  Sequential()  #build the model 
    def TruthtableBuildTrain(self,label):  #傳入Truthtable 與他的label值(OR AND XOR)分別不同
        X = np.array([[1,1],[1,0],[0,1],[0,0]])
        Y = label
        self.truthtablemodel.add(Dense(1,input_shape=(2,),activation='sigmoid', name='output_layer'))
#        self.truthtablemodel.add(Dense(1, activation='hard_sigmoid', name='output_layer'))
        self.truthtablemodel.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.truthtablemodel.fit(X,Y,batch_size=4, nb_epoch=2000)
    def Predict(self,name):  
        X_Predict = np.array([[1,1],[1,0],[0,1],[0,0]])
        Y_Result = self.truthtablemodel.predict(X_Predict)
        print("Test:",name,Y_Result)


AndSLP = SLPTruthtableBuild()
andlabel = np.array([1,0,0,0])
AndSLP.TruthtableBuildTrain(andlabel)
AND="AND Table"
AndSLP.Predict(AND)

OrSLP = SLPTruthtableBuild()
orlabel = np.array([1,1,1,0])
OrSLP.TruthtableBuildTrain(orlabel)
OR="OR Table"
OrSLP.Predict(OR)

XorSLP = SLPTruthtableBuild()
xorlabel = np.array([0,1,1,0])
XorSLP.TruthtableBuildTrain(xorlabel)
XOR = "XOR Table"
XorSLP.Predict(XOR)

