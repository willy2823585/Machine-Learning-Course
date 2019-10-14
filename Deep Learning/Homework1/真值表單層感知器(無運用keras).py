# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 23:59:33 2019

@author: willywanghw
"""
#7108029211 王皓威
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