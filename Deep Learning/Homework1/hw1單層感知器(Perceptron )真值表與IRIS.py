# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 11:04:41 2019

@author: willyWanghw
"""

import numpy as np
class Perceptron(object):

    def __init__(self, no_of_inputs, threshold=100, learning_rate=0.01): #threshold (epochs)
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)
        self.training_inputs = []
        #用於訓練真值表的LIST(輸入)
        self.training_inputs.append(np.array([1, 1]))  
        self.training_inputs.append(np.array([1, 0]))
        self.training_inputs.append(np.array([0, 1]))
        self.training_inputs.append(np.array([0, 0]))
    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if summation > 0:
          activation = 1
        else:
          activation = 0            
        return activation

    def train(self,labels): #傳入你要訓練的真值表label ex: AND>labels = np.array([1, 0, 0, 0])
        training_inputs = self.training_inputs
        for _ in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)

#用於訓練真值表AND的LABEL
labels_AND = np.array([1, 0, 0, 0])
P_AND = Perceptron(2)
P_AND.train(labels_AND)

inputs = np.array([1,1])
print("AND感知器訓練結果測試")
print("輸入[1,1]  輸出:",P_AND.predict(inputs)) 
inputs = np.array([1,0])
print("輸入[1,0]  輸出:",P_AND.predict(inputs))
inputs = np.array([0,1])
print("輸入[0,1]  輸出:",P_AND.predict(inputs)) 
inputs = np.array([0,0])
print("輸入[0,0]  輸出:",P_AND.predict(inputs)) 
#=> 1


#用於訓練真值表OR的LABEL
labels_or = np.array([1, 1, 1, 0])
P_OR =  Perceptron(2)
P_OR.train(labels_or)


inputs_or = np.array([1,1])
print("OR感知器訓練結果測試")
print("輸入[1,1 輸出]",P_OR.predict(inputs_or))
inputs_or = np.array([1,0])
print("輸入[1,0 輸出]",P_OR.predict(inputs_or))
inputs_or = np.array([0,1])
print("輸入[0,1 輸出]",P_OR.predict(inputs_or))
inputs_or = np.array([0,0])
print("輸入[0,0 輸出]",P_OR.predict(inputs_or))


#用於訓練真值表XOR的LABEL
label_xor = [0,1,1,0]
P_XOR = Perceptron(2)
P_XOR.train(label_xor)

input_xor = np.array([1,1])
print("XOR感知器訓練結果測試")
print("輸入[1,1 輸出]",P_XOR.predict(input_xor))
input_xor = np.array([1,0])
print("輸入[1,0 輸出]",P_XOR.predict(input_xor))
input_xor = np.array([0,1])
print("輸入[0,1 輸出]",P_XOR.predict(input_xor))
input_xor = np.array([0,0])
print("輸入[0,0 輸出]",P_XOR.predict(input_xor))
print("XOR無法運用單層感知器")



print("----------------------------------------------------")


import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout
from keras import backend as K

K.tensorflow_backend._get_available_gpus()
from tensorflow.python.client import device_lib
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
classifier.fit(train_x, train_y, batch_size=100, nb_epoch=200)

results = classifier.evaluate(test_x, test_y)
print('Final test set loss: {:4f}'.format(results[0]))
print('Final test set accuracy: {:4f}'.format(results[1]))

print("--------------------------------------------------")

#keras use in 真值表
#AND 
truthtable = Sequential()
X = np.array([[1,1],[1,0],[0,1],[0,0]])
y = np.array([1,0,0,0])

truthtable.add(Dense(1, input_shape=(2,), activation='hard_sigmoid', name='input_layer'))
#truthtable.add(Dense(1, activation='softmax', name='output'))
truthtable.compile(optimizer, loss='binary_crossentropy', metrics=['accuracy'])
truthtable.fit(X,y,batch_size=100, nb_epoch=300)

X_Predict = np.array([[1,1],[1,0],[0,1],[0,0]])
Y_Result = truthtable.predict(X_Predict)
print(Y_Result)

