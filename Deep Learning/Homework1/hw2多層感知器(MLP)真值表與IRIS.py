# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 21:07:42 2019

@author: willywanghw

"""
#7108029211 王皓威
#Hw2 MLP應用於IRIS資料與真值表
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout
#from keras import backend as K
#from tensorflow.python.client import device_lib
#import tensorflow as tf
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
#K.tensorflow_backend._get_available_gpus()
#print(device_lib.list_local_devices())
iris_data = load_iris() # load the iris dataset

print('Example data: ')
print(iris_data.data[:5])
print('Example labels: ')
print(iris_data.target[:5])
#preprocessing
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
#Create neural network
classifier.add(Dense(10, input_shape=(4,), activation='relu', name='hidden_layer1'))
classifier.add(Dense(10, input_shape=(4,), activation='relu', name='hidden_layer2'))
classifier.add(Dense(10, input_shape=(4,), activation='elu', name='hidden_layer3'))
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

#真值表Truthtable use MLP(多層感知器)
print("--------------------------------------------------")

class MLPTruthtableBuild(object):
    def __init__(self):
        self.truthtablemodel  =  Sequential()  #build the model 
    def TruthtableBuildTrain(self,label):  #傳入Truthtable 與他的label值(OR AND XOR)分別不同
        X = np.array([[1,1],[1,0],[0,1],[0,0]])
        Y = label
        self.truthtablemodel.add(Dense(10, input_shape=(2,), activation='relu', name='hidden_layer1'))
        self.truthtablemodel.add(Dense(20, activation='hard_sigmoid', name='hidden_layer2'))
        self.truthtablemodel.add(Dense(1, activation='hard_sigmoid', name='hidden_layer3'))
        self.truthtablemodel.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.truthtablemodel.fit(X,Y,batch_size=4, nb_epoch=1000)
    def Predict(self,name):  
        X_Predict = np.array([[1,1],[1,0],[0,1],[0,0]])
        Y_Result = self.truthtablemodel.predict(X_Predict)
        print("Test [1,1],[1,0],[0,1],[0,0]:\n",name,Y_Result)


AndMLP = MLPTruthtableBuild()
andlabel = np.array([1,0,0,0])
AndMLP.TruthtableBuildTrain(andlabel)
AND="AND\n"
AndMLP.Predict(AND)

OrMLP = MLPTruthtableBuild()
orlabel = np.array([1,1,1,0])
OrMLP.TruthtableBuildTrain(orlabel)
OR="OR\n"
OrMLP.Predict(OR)

XorMLP = MLPTruthtableBuild()
xorlabel = np.array([0,1,1,0])
XorMLP.TruthtableBuildTrain(xorlabel)
XOR = "XOR\n"
XorMLP.Predict(XOR)


