# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:59:12 2019

@author: willywanghw
"""
#7108029211 王皓威

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
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(512, activation='sigmoid'))
network.add(layers.Dense(10,activation='softmax'))

network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#測試集放進驗證集
networkht = network.fit(train_images, train_labels, epochs=25, batch_size=256,validation_data=(test_images,test_labels), verbose=1)
net_dict = networkht.history
test_loss, test_acc = network.evaluate(test_images, test_labels)
import matplotlib.pyplot as plt 
loss = networkht.history['loss']
val_loss = networkht.history['val_loss']
epochs = range(1,len(loss)+1)
plt.plot(epochs,loss,'r',label='Training loss')
plt.plot(epochs,val_loss,'b',label = 'validation loss')
plt.title('Training and validataion loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()

acc = networkht.history['acc']
val_acc = networkht.history['val_acc']

plt.plot(epochs,acc,'r',label = 'Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

print('test_acc:', test_acc)
