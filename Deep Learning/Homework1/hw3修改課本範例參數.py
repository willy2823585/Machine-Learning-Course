# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 01:09:20 2019

@author: EXIA
"""
#7108029211 王皓威
#hw3 修改課本第二張範例參數

import keras
keras.__version__

from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images.shape

len(train_labels)

train_labels
test_images.shape
len(test_labels)
test_labels
from keras import models
from keras import layers

network = models.Sequential()
#參數修改  試過sigmoid,relu..等
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
#層數加入  準確度0.9752   神經元512>>1024
network.add(layers.Dense(1024, activation='relu'))
#參數修改 
network.add(layers.Dense(10,activation='softmax'))

#Keras優化器的公共參數 RMSprop,SGD,Adagrad,Adadelta,Adam
#adam 約98%  Adadelta 0.98  Adagrad 約0.98  SGD 約0.9246   RMSprop  0.9781
network.compile(optimizer='Adadelta',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
#batch_size =128 改成64
network.fit(train_images, train_labels, epochs=5, batch_size=64)

test_loss, test_acc = network.evaluate(test_images, test_labels)

print('test_acc:', test_acc)