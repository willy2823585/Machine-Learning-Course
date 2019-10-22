# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 16:57:28 2019

@author: willy
"""
#Hw2 使用不同優化器、調整epoch、batch size、隱藏層數、隱藏層節點數


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
import matplotlib.pyplot as plt 
from keras.utils import to_categorical


train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28)) 
test_images = test_images.astype('float32') / 255

optimizerlist = ['rmsprop','Adam','SGD','Adagrad','Adadelta','Adamax','Nadam']
b_size = [64,128,256,512]
   
for opt in optimizerlist :   
    plt.clf()
    network = models.Sequential()

    network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))

    network.add(layers.Dense(256, activation='sigmoid'))

    network.add(layers.Dense(10,activation='sigmoid'))

    network.compile(optimizer=opt,
    loss='categorical_crossentropy',
    metrics=['accuracy'])
    #batch_size =128 改成64
    networkht = network.fit(train_images, train_labels, epochs=20, batch_size=256,validation_data=(test_images,test_labels), verbose=1)
    net_dict = networkht.history
    test_loss, test_acc = network.evaluate(test_images, test_labels)
    print('test_acc:', test_acc)
    loss = networkht.history['loss']
    val_loss = networkht.history['val_loss']
    epochs = range(1,len(loss)+1)    
    plt.plot(epochs,loss,'r',label='Training loss')
    plt.plot(epochs,val_loss,'b',label = 'validation loss')
    plt.title('Training and validataion loss '+opt)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(opt+'+loss')
    plt.show()
    acc = networkht.history['acc']
    val_acc = networkht.history['val_acc']
    plt.plot(epochs,acc,'r',label = 'Training acc')
    plt.plot(epochs,val_acc,'b',label='Validation acc')
    plt.title('Training and Validation accuracy '+opt)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(opt+'+acc')
    plt.show()

